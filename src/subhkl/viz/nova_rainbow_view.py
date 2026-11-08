"""NOVA / Trame and NDIP Galaxy Interactive Tool integration for Rainbow Zones.

Supports three execution modes for NDIP / Galaxy / NOVA and local beamline workstations:
1. **NOVA / Trame View (`RainbowZonesTrameApp`)**: When ``trame`` (and optionally
   ``nova-trame``) is installed inside the container, mounts the Rainbow Zones
   interactive view inside a Trame/Vuetify server with bidirectional state sync.
2. **Built-in HTTP Interactive Server (`serve_rainbow_zones`)**: Zero-dependency
   HTTP server on ``0.0.0.0:8080`` (honoring ``$EP_PATH`` for Galaxy's
   ``gx-it-proxy`` / nginx reverse proxy) so the ``ghcr.io/neutrons/subhkl``
   container can run as a Galaxy Interactive Tool without requiring extra packages.
3. **Standalone HTML Export**: Produces a self-contained HTML artifact for
   Galaxy dataset history viewing.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from urllib.parse import parse_qs, urlparse

from subhkl.viz.rainbow_zones import (
    build_rainbow_zone_payload,
    render_rainbow_html,
)


class NovaGalaxyPipelineRunner:
    """Orchestrates NDIP / Galaxy IMAGINE-X workflow steps on GPU Pulsar runners.

    Designed so the lightweight visualization / NOVA GUI runs on a CPU container
    (``JAX_PLATFORMS=cpu``) and dispatches heavy compute steps (``solve``,
    ``predict``, ``integrate``) to Galaxy, which routes them to a GPU Pulsar
    runner (e.g. ``imaginex1``):
      - Stage 1 (Reduce & Merge): ``imaginex_subhkl_prepare_peaks`` (``merged.h5``)
      - Stage 2 (Solve Lattice Ladder): ``imaginex_subhkl_solve`` (finder-less ``solve.h5``)
      - Stage 3 (Predict & Integrate): ``imaginex_subhkl_predict`` -> ``imaginex_subhkl_integrate``
      - Stage 4 (Merge & MTZ): ``imaginex_subhkl_integrate`` (MTZ export)
    """

    STAGE_TOOLS: dict[str, list[str]] = {
        "reduce": [
            "imaginex_subhkl_prepare_peaks",
        ],
        "solve": ["imaginex_subhkl_solve"],
        "integrate": [
            "imaginex_subhkl_predict",
            "imaginex_subhkl_integrate",
        ],
        "merge": ["imaginex_subhkl_mtz", "imaginex_merge", "imaginex_analysis"],
    }

    def __init__(
        self,
        galaxy_url: str | None = None,
        galaxy_api_key: str | None = None,
        history_id: str | None = None,
    ) -> None:
        raw_url = galaxy_url or os.environ.get("GALAXY_URL") or ""
        if raw_url.startswith("$"):
            raw_url = ""
        raw_key = galaxy_api_key or os.environ.get("GALAXY_API_KEY") or ""
        if raw_key.startswith("$"):
            raw_key = ""
        raw_hid = history_id or os.environ.get("HISTORY_ID") or ""
        if raw_hid.startswith("$"):
            raw_hid = ""

        self.galaxy_url = raw_url.rstrip("/") if raw_url else None
        self.galaxy_api_key = raw_key or None
        self.history_id = raw_hid or None
        self._conn_helper: Any = None
        self.active_job: dict[str, Any] | None = None
        self._auto_discover_local_galaxy()

    def _auto_discover_local_galaxy(self) -> None:
        """Auto-discover local Planemo / Galaxy server if env vars were not injected."""
        if not self.galaxy_url:
            for candidate in ("http://127.0.0.1:9090", "http://localhost:9090"):
                try:
                    req = urllib.request.Request(f"{candidate}/api/version")
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        if resp.status == 200:
                            self.galaxy_url = candidate
                            break
                except Exception:
                    continue

        if self.galaxy_url and not self.galaxy_api_key:
            # Check if Planemo's default master_api_key ('test_key') can mint/fetch a user key
            try:
                users = self._api_request("GET", "/api/users", api_key="test_key")
                if isinstance(users, list) and users:
                    uid = users[0]["id"]
                    ukey = self._api_request(
                        "POST", f"/api/users/{uid}/api_key", api_key="test_key"
                    )
                    if isinstance(ukey, str) and ukey:
                        self.galaxy_api_key = ukey
                    else:
                        self.galaxy_api_key = "test_key"
            except Exception:
                pass

        if self.galaxy_url and self.galaxy_api_key and not self.history_id:
            try:
                histories = self._api_request("GET", "/api/histories")
                if isinstance(histories, list) and histories:
                    self.history_id = histories[0]["id"]
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return bool(self.galaxy_url and self.galaxy_api_key)

    def _api_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        api_key: str | None = None,
        raw: bool = False,
    ) -> Any:
        if not self.galaxy_url:
            raise RuntimeError("GALAXY_URL is not configured.")
        key = api_key or self.galaxy_api_key
        sep = "&" if "?" in path else "?"
        url = f"{self.galaxy_url}{path}"
        if key:
            url = f"{url}{sep}key={urllib.parse.quote(key)}"
        headers = {"Accept": "application/json"}
        if key:
            headers["x-api-key"] = key
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            body = resp.read()
            if raw:
                return body
            return json.loads(body.decode("utf-8"))

    def list_history_datasets(self) -> list[dict[str, Any]]:
        """Return active HDF5/MTZ datasets in the current Galaxy history."""
        if not self.is_connected:
            return []
        if not self.history_id:
            self._auto_discover_local_galaxy()
        if not self.history_id:
            return []
        try:
            items = self._api_request(
                "GET",
                f"/api/histories/{self.history_id}/contents?v=dev&q=deleted&qv=false",
            )
            if not isinstance(items, list):
                return []
            return [
                {
                    "id": it.get("id"),
                    "hid": it.get("hid"),
                    "name": it.get("name"),
                    "state": it.get("state"),
                    "extension": it.get("extension"),
                }
                for it in items
                if isinstance(it, dict)
                and it.get("history_content_type") == "dataset"
                and not it.get("deleted")
            ]
        except Exception:
            return []

    def connect(self) -> Any:
        if self._conn_helper is not None:
            return self._conn_helper
        if not self.is_connected:
            raise RuntimeError(
                "GALAXY_URL and GALAXY_API_KEY environment variables are required "
                "for live NDIP Galaxy execution."
            )
        from nova.galaxy import Connection

        conn = Connection(self.galaxy_url, self.galaxy_api_key)
        self._conn_helper = conn.connect()
        return self._conn_helper

    def get_datastore(self, history_name: str = "IMAGINE-X SubHKL") -> Any:
        helper = self.connect()
        return helper.get_data_store(history_name, create=True)

    def submit_galaxy_stage(
        self,
        stage_id: str,
        params: dict[str, Any] | None = None,
        on_complete_h5: Any = None,
    ) -> dict[str, Any]:
        """Submit a coarse NDIP workflow stage to Galaxy REST API (dispatched to Pulsar GPU)."""
        if stage_id not in self.STAGE_TOOLS:
            raise ValueError(f"Unknown pipeline stage: {stage_id}")
        params = dict(params or {})
        tool_ids = self.STAGE_TOOLS[stage_id]
        if not self.is_connected:
            return {
                "stage_id": stage_id,
                "mode": "local_preview",
                "tool_ids": tool_ids,
                "status": "simulated",
            }

        datasets = self.list_history_datasets()
        h5_ok = [
            d
            for d in datasets
            if d.get("state") == "ok" and d.get("extension") in ("h5", "hdf5", "nxs.h5")
        ]

        # Match appropriate input datasets from the active Galaxy history
        merged_ds = next(
            (
                d
                for d in reversed(h5_ok)
                if any(k in (d.get("name") or "").lower() for k in ("merged", "image", "scan", "prepare"))
            ),
            h5_ok[-1] if h5_ok else None,
        )
        solve_ds = next(
            (
                d
                for d in reversed(h5_ok)
                if any(k in (d.get("name") or "").lower() for k in ("solve", "index"))
            ),
            h5_ok[0] if h5_ok else None,
        )

        primary_tool = tool_ids[0]
        tool_inputs: dict[str, Any] = {}

        if stage_id == "solve":
            primary_tool = "imaginex_subhkl_solve"
            input_id = params.get("input_h5_id") or (merged_ds["id"] if merged_ds else None)
            if not input_id:
                raise RuntimeError("No input HDF5 dataset found in Galaxy history for Solve.")
            cell = params.get("cell") or [68.7, 69.8, 156.9, 90.0, 90.0, 90.0]
            wl = params.get("wavelength_range") or [1.8, 4.5]
            tool_inputs = {
                "input_h5": {"src": "hda", "id": input_id},
                "instrument": params.get("instrument", "CG4D"),
                "space_group": params.get("space_group", "P 21 21 21"),
                "a": float(cell[0]),
                "b": float(cell[1]),
                "c": float(cell[2]),
                "alpha": float(cell[3]),
                "beta": float(cell[4]),
                "gamma": float(cell[5]),
                "wavelength_min": float(wl[0]),
                "wavelength_max": float(wl[1]),
                "objective": params.get("objective", "band"),
                "d_min": float(params.get("d_min", 2.5)),
                "n_zones": int(params.get("n_zones", 400)),
                "steps": int(params.get("steps", 250)),
            }
        elif stage_id == "integrate":
            primary_tool = "imaginex_subhkl_predict"
            img_id = params.get("images_h5_id") or (merged_ds["id"] if merged_ds else None)
            idx_id = params.get("indexed_h5_id") or (solve_ds["id"] if solve_ds else img_id)
            if not img_id or not idx_id:
                raise RuntimeError("Need both merged images and solve.h5 in Galaxy history.")
            wl = params.get("wavelength_range") or [1.8, 4.5]
            tool_inputs = {
                "input": {"src": "hda", "id": img_id},
                "indexed_peaks": {"src": "hda", "id": idx_id},
                "instrument": params.get("instrument", "CG4D"),
                "d_min": float(params.get("d_min", 1.8)),
                "wavelength_min": float(wl[0]),
                "wavelength_max": float(wl[1]),
            }
        else:
            input_id = params.get("input_h5_id") or (merged_ds["id"] if merged_ds else None)
            if input_id:
                tool_inputs["input"] = {"src": "hda", "id": input_id}

        submit_body = {
            "history_id": self.history_id,
            "tool_id": primary_tool,
            "inputs": tool_inputs,
        }
        resp = self._api_request("POST", "/api/tools", payload=submit_body)
        jobs = resp.get("jobs", []) if isinstance(resp, dict) else []
        outputs = resp.get("outputs", []) if isinstance(resp, dict) else []
        job_id = jobs[0].get("id") if jobs else None

        job_record = {
            "job_id": job_id,
            "tool_id": primary_tool,
            "stage_id": stage_id,
            "state": "queued",
            "ladder_rung_idx": 0 if stage_id == "solve" else 4,
            "outputs": [
                {"id": o.get("id"), "name": o.get("name"), "extension": o.get("extension")}
                for o in outputs
                if isinstance(o, dict)
            ],
            "updated_payload_ready": False,
            "started_at": time.time(),
        }
        self.active_job = job_record

        if job_id:
            threading.Thread(
                target=self._poll_galaxy_job,
                args=(job_record, on_complete_h5),
                daemon=True,
            ).start()

        return {
            "stage_id": stage_id,
            "mode": "ndip_galaxy_rest",
            "galaxy_url": self.galaxy_url,
            "history_id": self.history_id,
            "job": job_record,
            "status": "submitted",
        }

    def _poll_galaxy_job(self, job_record: dict[str, Any], on_complete_h5: Any = None) -> None:
        job_id = job_record.get("job_id")
        if not job_id:
            return
        step_counter = 0
        while True:
            time.sleep(1.8)
            try:
                jinfo = self._api_request("GET", f"/api/jobs/{job_id}")
                state = jinfo.get("state", "running")
                job_record["state"] = state
                if job_record.get("stage_id") == "solve" and state in ("queued", "running"):
                    step_counter = min(3, step_counter + 1)
                    job_record["ladder_rung_idx"] = step_counter
                if state in ("ok", "error", "deleted", "paused"):
                    if state == "ok":
                        job_record["ladder_rung_idx"] = 4
                        h5_outs = [
                            o
                            for o in job_record.get("outputs", [])
                            if o.get("extension") in ("h5", "hdf5")
                        ]
                        if h5_outs and callable(on_complete_h5):
                            ds_id = h5_outs[0]["id"]
                            raw_bytes = self._api_request(
                                "GET", f"/api/datasets/{ds_id}/display", raw=True
                            )
                            fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="galaxy_solve_")
                            with os.fdopen(fd, "wb") as f:
                                f.write(raw_bytes)
                            on_complete_h5(tmp_path)
                            job_record["updated_payload_ready"] = True
                    break
            except Exception as exc:
                job_record["error"] = str(exc)
                break

    def run_stage(
        self,
        stage_id: str,
        params_by_tool: dict[str, dict[str, Any]] | None = None,
        *,
        history_name: str = "IMAGINE-X SubHKL",
        wait: bool = True,
    ) -> dict[str, Any]:
        """Run one coarse NDIP pipeline stage in Galaxy for full artifact provenance."""
        if stage_id not in self.STAGE_TOOLS:
            raise ValueError(f"Unknown pipeline stage: {stage_id}")
        tool_ids = self.STAGE_TOOLS[stage_id]
        if not self.is_connected:
            return {
                "stage_id": stage_id,
                "mode": "local_preview",
                "tool_ids": tool_ids,
                "status": "simulated",
            }

        try:
            from nova.galaxy import Parameters, Tool
        except ImportError:
            flat_params: dict[str, Any] = {}
            for pdict in (params_by_tool or {}).values():
                flat_params.update(pdict)
            return self.submit_galaxy_stage(stage_id, flat_params)

        datastore = self.get_datastore(history_name)
        outputs_summary: list[dict[str, Any]] = []
        for tool_id in tool_ids:
            tool = Tool(tool_id)
            params = Parameters()
            for k, v in (params_by_tool or {}).get(tool_id, {}).items():
                params.add_input(k, v)
            outputs = tool.run(datastore, params, wait=wait)
            outputs_summary.append(
                {
                    "tool_id": tool_id,
                    "job_id": getattr(getattr(tool, "_job", None), "id", None),
                    "outputs": [getattr(o, "id", str(o)) for o in (outputs or [])],
                }
            )
        return {
            "stage_id": stage_id,
            "mode": "ndip_galaxy",
            "galaxy_url": self.galaxy_url,
            "tool_runs": outputs_summary,
            "status": "completed" if wait else "running",
        }


class RainbowZonesViewModel:
    """MVVM ViewModel managing Rainbow Zones state for NOVA / Trame or HTTP views."""

    def __init__(
        self,
        replay_h5: str | os.PathLike | None = None,
        payload: dict[str, Any] | None = None,
        default_mode: str = "arcade",
        **payload_kwargs: Any,
    ) -> None:
        self.replay_h5 = str(replay_h5) if replay_h5 is not None else None
        self.default_mode = default_mode
        self.payload_kwargs = dict(payload_kwargs)
        self.pipeline_runner = NovaGalaxyPipelineRunner()
        if payload is not None:
            self.payload = payload
        elif replay_h5 is not None:
            self.payload = build_rainbow_zone_payload(
                replay_h5=replay_h5, **self.payload_kwargs
            )
        else:
            self.payload = build_rainbow_zone_payload(**self.payload_kwargs)

    def reload(self, **overrides: Any) -> dict[str, Any]:
        kwargs = {**self.payload_kwargs, **overrides}
        if self.replay_h5 is not None:
            self.payload = build_rainbow_zone_payload(
                replay_h5=self.replay_h5, **kwargs
            )
        else:
            self.payload = build_rainbow_zone_payload(**kwargs)
        return self.payload

    def to_html(self, mode: str | None = None) -> str:
        return render_rainbow_html(
            self.payload,
            default_mode=mode or self.default_mode,
        )


class RainbowZonesTrameApp:
    """Trame / NOVA application wrapper for Rainbow Zones."""

    def __init__(
        self,
        view_model: RainbowZonesViewModel,
        server: Any = None,
    ) -> None:
        self.vm = view_model
        self.server = server
        self._setup_trame()

    def _setup_trame(self) -> None:
        try:
            from trame.app import get_server
            from trame.ui.html import DivLayout
            from trame.widgets import html as html_widgets
        except ImportError as exc:
            raise RuntimeError(
                "Trame is not installed in this environment; use "
                "serve_rainbow_zones() or export_rainbow_widget() instead."
            ) from exc

        if self.server is None:
            self.server = get_server(client_type="vue3")

        state = self.server.state
        state.rainbow_mode = self.vm.default_mode
        state.rainbow_payload = self.vm.payload
        state.rainbow_iframe_srcdoc = self.vm.to_html()

        with DivLayout(self.server) as layout:
            layout.root.style = "margin:0;padding:0;width:100vw;height:100vh;overflow:hidden;background:#060719;"
            html_widgets.Iframe(
                srcdoc=("rainbow_iframe_srcdoc",),
                style="border:none;width:100%;height:100%;",
            )

    def start(self, host: str = "0.0.0.0", port: int = 8080, **kwargs: Any) -> None:
        self.server.start(
            host=host,
            port=port,
            open_browser=False,
            timeout=0,
            **kwargs,
        )


def serve_rainbow_zones(
    replay_h5: str | os.PathLike | None = None,
    payload: dict[str, Any] | None = None,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    default_mode: str = "arcade",
    prefer_trame: bool = False,
    **payload_kwargs: Any,
) -> None:
    """Serve the Rainbow Zones GUI over HTTP (or Trame if requested and installed).

    Compatible with NDIP / Galaxy Interactive Tools (`$EP_PATH`, port 8080) as well
    as local container port-forwarding (`podman run -p 8080:8080 ...`).
    """
    vm = RainbowZonesViewModel(
        replay_h5=replay_h5,
        payload=payload,
        default_mode=default_mode,
        **payload_kwargs,
    )

    if prefer_trame:
        try:
            app = RainbowZonesTrameApp(vm)
            app.start(host=host, port=port)
            return
        except RuntimeError:
            pass

    ep_path = os.environ.get("EP_PATH", "").rstrip("/")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            req_path = parsed.path
            if ep_path and req_path.startswith(ep_path):
                req_path = req_path[len(ep_path) :] or "/"

            if req_path in ("/api/payload", "/payload.json"):
                body = json.dumps(vm.payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/status":
                status_doc = {
                    "galaxy_connected": vm.pipeline_runner.is_connected,
                    "galaxy_url": vm.pipeline_runner.galaxy_url,
                    "history_id": vm.pipeline_runner.history_id,
                    "history_datasets": vm.pipeline_runner.list_history_datasets(),
                    "active_job": vm.pipeline_runner.active_job,
                    "pipeline_stages": vm.payload.get("pipeline_stages", []),
                    "ladder_steps": vm.payload.get("ladder_steps", []),
                }
                body = json.dumps(status_doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            mode = qs.get("mode", [vm.default_mode])[0]
            body = vm.to_html(mode=mode).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            req_path = parsed.path
            if ep_path and req_path.startswith(ep_path):
                req_path = req_path[len(ep_path) :] or "/"

            if req_path == "/api/pipeline/run":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    stage_id = req_json.get("stage_id", "solve")
                    params = dict(req_json.get("params") or {})
                    params.setdefault("instrument", vm.payload.get("instrument", "CG4D"))
                    params.setdefault("space_group", vm.payload.get("space_group", "P 1"))
                    if vm.payload.get("cell"):
                        params.setdefault("cell", vm.payload.get("cell"))
                    if vm.payload.get("wavelength_range"):
                        params.setdefault("wavelength_range", vm.payload.get("wavelength_range"))

                    def _on_complete(new_h5_path: str) -> None:
                        vm.replay_h5 = new_h5_path
                        vm.reload()

                    result = vm.pipeline_runner.submit_galaxy_stage(
                        stage_id, params=params, on_complete_h5=_on_complete
                    )
                    code = 200
                except Exception as exc:
                    result = {"status": "error", "error": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, int(port)), _Handler) as httpd:
        print(
            f"Rainbow Zones server listening on http://{host}:{port}{ep_path or '/'} "
            f"(mode={default_mode})"
        )
        httpd.serve_forever()
