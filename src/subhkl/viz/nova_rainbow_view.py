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

    @staticmethod
    def _is_galaxy_server(url: str) -> bool:
        """Return True only if ``url`` responds with a valid Galaxy JSON ``/api/version`` payload."""
        try:
            req = urllib.request.Request(
                f"{url.rstrip('/')}/api/version",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status != 200:
                    return False
                doc = json.loads(resp.read().decode("utf-8"))
                return isinstance(doc, dict) and "version_major" in doc
        except Exception:
            return False

    @staticmethod
    def _clean_err_msg(exc: Exception) -> str:
        """Strip any HTML/XML/SVG body or multi-line traceback from an exception message."""
        msg = str(exc).strip()
        for marker in ("<!doctype", "<html", "<?xml", "<svg", "\n"):
            idx = msg.lower().find(marker)
            if idx > 0:
                msg = msg[:idx].strip().rstrip(":,;")
        return msg[:220]

    def _auto_discover_local_galaxy(self) -> None:
        """Auto-discover local Planemo / Galaxy server if env vars were not injected."""
        if self.galaxy_url and self.galaxy_url.startswith("https://"):
            http_alt = "http://" + self.galaxy_url[len("https://") :]
            if (
                any(h in self.galaxy_url for h in ("localhost", "127.0.0.1"))
                and not self._is_galaxy_server(self.galaxy_url)
                and self._is_galaxy_server(http_alt)
            ):
                self.galaxy_url = http_alt

        if not self.galaxy_url:
            for candidate in ("http://127.0.0.1:9090", "http://localhost:9090"):
                if self._is_galaxy_server(candidate):
                    self.galaxy_url = candidate
                    break

        if self.galaxy_url and (
            not self.galaxy_api_key or self.galaxy_api_key == "test_key"
        ):
            # Check if Planemo's default master_api_key ('test_key') can mint/fetch a user key
            try:
                users = self._api_request("GET", "/api/users", api_key="test_key")
                if isinstance(users, list) and users:
                    uid = users[0]["id"]
                    ukey = self._api_request(
                        "GET", f"/api/users/{uid}/api_key", api_key="test_key"
                    )
                    if not (isinstance(ukey, str) and ukey):
                        ukey = self._api_request(
                            "POST", f"/api/users/{uid}/api_key", api_key="test_key"
                        )
                    if isinstance(ukey, str) and ukey:
                        self.galaxy_api_key = ukey
                    elif not self.galaxy_api_key:
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
        timeout: float | None = None,
    ) -> Any:
        if not self.galaxy_url:
            raise RuntimeError("GALAXY_URL is not configured.")
        key = api_key or self.galaxy_api_key
        sep = "&" if "?" in path else "?"
        url = f"{self.galaxy_url}{path}"
        if key:
            url = f"{url}{sep}key={urllib.parse.quote(key)}"
        headers = {"Accept": "*/*" if raw else "application/json"}
        if key:
            headers["x-api-key"] = key
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req_timeout = timeout if timeout is not None else (180.0 if raw else 30.0)
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=req_timeout) as resp:
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

    def resolve_dataset_to_local_h5(
        self, ds_id: str, fallback_dir: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a Galaxy dataset ID to a local HDF5 path (via local file_name, sibling dir, or /display download)."""
        info = self._api_request("GET", f"/api/datasets/{ds_id}")
        if not isinstance(info, dict):
            raise RuntimeError(f"Dataset {ds_id} not found on {self.galaxy_url}")
        name = str(info.get("name") or f"{ds_id}.h5")
        file_name = str(info.get("file_name") or "")
        if file_name and os.path.isfile(file_name) and os.path.getsize(file_name) > 0:
            return file_name, info
        if fallback_dir and os.path.isdir(fallback_dir):
            cand = os.path.join(fallback_dir, os.path.basename(name))
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand, info
        safe_name = "".join(
            ch if (ch.isalnum() or ch in "._-") else "_"
            for ch in os.path.basename(name)
        )
        if not safe_name.endswith((".h5", ".hdf5")):
            safe_name += ".h5"
        tmp_dest = os.path.join(
            tempfile.gettempdir(), f"subhkl_gx_{ds_id}_{safe_name}"
        )
        if os.path.isfile(tmp_dest) and os.path.getsize(tmp_dest) > 0:
            return tmp_dest, info
        url = f"{self.galaxy_url}/api/datasets/{ds_id}/display"
        headers: dict[str, str] = {}
        if self.galaxy_api_key:
            headers["x-api-key"] = self.galaxy_api_key
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            data = resp.read()
        with open(tmp_dest, "wb") as fh:
            fh.write(data)
        return tmp_dest, info

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
        # Preserve explicit port (e.g. :9090 for local Planemo/Galaxy) stripped by nova.galaxy.Connection.__init__
        conn.galaxy_url = self.galaxy_url
        self._conn_helper = conn.connect()
        return self._conn_helper

    def get_datastore(self, history_name: str = "IMAGINE-X SubHKL") -> Any:
        helper = self.connect()
        return helper.get_data_store(history_name, create=True)

    def _has_tool_on_galaxy(self, tool_id: str) -> bool:
        try:
            doc = self._api_request("GET", f"/api/tools/{tool_id}")
            return isinstance(doc, dict) and doc.get("id") == tool_id
        except Exception:
            return False

    def submit_galaxy_stage(
        self,
        stage_id: str,
        params: dict[str, Any] | None = None,
        on_complete_h5: Any = None,
    ) -> dict[str, Any]:
        """Submit a coarse NDIP workflow stage via nova-galaxy -> Galaxy -> Pulsar GPU container."""
        if stage_id not in self.STAGE_TOOLS:
            raise ValueError(f"Unknown pipeline stage: {stage_id}")
        params = dict(params or {})
        if not self.is_connected:
            return self._submit_local_subhkl_stage(
                stage_id,
                params,
                on_complete_h5,
                fallback_reason="GALAXY_URL / GALAXY_API_KEY not set",
            )

        job_record: dict[str, Any] = {
            "job_id": f"nova-{int(time.time())}",
            "tool_id": self.STAGE_TOOLS[stage_id][0],
            "stage_id": stage_id,
            "mode": "galaxy",
            "state": "queued",
            "command_line": f"nova-galaxy -> {self.galaxy_url}",
            "stdout": (
                f"[nova -> galaxy] Connecting to {self.galaxy_url} (datastore: 'IMAGINE-X SubHKL')...\n"
                f"[pulsar] Target GPU worker queue on this node (IMAGINEX-1): imaginex1-test / imaginex1\n"
                f"[pulsar] Staging dir on IMAGINEX-1: /data/staging_test/ (or /data/staging/)\n"
            ),
            "stderr": "",
            "ladder_rung_idx": 0 if stage_id == "solve" else 4,
            "outputs": [],
            "updated_payload_ready": False,
            "started_at": time.time(),
        }
        self.active_job = job_record

        threading.Thread(
            target=self._dispatch_nova_galaxy_thread,
            args=(stage_id, params, job_record, on_complete_h5),
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

    def _ensure_galaxy_h5_dataset(
        self,
        store: Any,
        local_path: str,
        job_record: dict[str, Any],
        *,
        force_upload: bool = False,
        file_ext: str = "h5",
    ) -> str:
        """Link an existing 'ok' dataset in the Galaxy history or upload with file_type=file_ext, to_posix_lines=False."""
        fname = os.path.basename(local_path)
        gi = store.nova_connection.galaxy_instance
        if not force_upload:
            existing = gi.datasets.get_datasets(history_id=store.history_id, name=fname)
            for ds in existing:
                ds_id = ds.get("id")
                if not ds_id:
                    continue
                info = gi.datasets.show_dataset(ds_id)
                st = info.get("state")
                if st == "ok":
                    job_record["stdout"] += f"[nova -> galaxy] Linked existing dataset '{fname}' (id={ds_id}, state=ok)\n"
                    return ds_id
                if st in ("queued", "running", "new", "upload"):
                    job_record["stdout"] += (
                        f"[nova -> galaxy] Waiting for in-progress upload of '{fname}' (id={ds_id}, state={st})...\n"
                    )
                    gi.datasets.wait_for_dataset(ds_id, maxwait=120, interval=2.0, check=True)
                    return ds_id

        if file_ext == "mtz":
            file_ext = "binary"
        is_local_gx = any(
            h in (getattr(gi, "base_url", "") or self.galaxy_url or "")
            for h in ("127.0.0.1", "localhost")
        )
        job_record["stdout"] += (
            f"[nova -> galaxy] Uploading '{fname}' to Galaxy history (file_type={file_ext})...\n"
        )
        up_info = gi.tools.upload_file(
            path=local_path,
            history_id=store.history_id,
            file_name=fname,
            file_type=file_ext,
            to_posix_lines=False,
            link_to_files=is_local_gx,
        )
        ds_id = up_info["outputs"][0]["id"]
        gi.datasets.wait_for_dataset(ds_id, maxwait=180, interval=2.0, check=True)
        job_record["stdout"] += f"[nova -> galaxy] Saved '{fname}' in Galaxy history (id={ds_id})\n"
        return ds_id

    def _dispatch_nova_galaxy_thread(
        self,
        stage_id: str,
        params: dict[str, Any],
        job_record: dict[str, Any],
        on_complete_h5: Any = None,
    ) -> None:
        """Background worker that links/uploads datasets, dispatches to Pulsar on IMAGINEX-1, and streams console_output."""
        try:
            store = self.get_datastore("IMAGINE-X SubHKL")
            self.history_id = store.history_id
            gi = store.nova_connection.galaxy_instance
            job_record["stdout"] += (
                f"[nova -> galaxy] Attached to Galaxy history_id={store.history_id} on {self.galaxy_url}\n"
            )

            if all(k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")):
                raw_cell = [
                    params["a"],
                    params["b"],
                    params["c"],
                    params["alpha"],
                    params["beta"],
                    params["gamma"],
                ]
            else:
                raw_cell = params.get("cell") or [
                    params.get("a", 68.7),
                    params.get("b", 69.8),
                    params.get("c", 156.9),
                    params.get("alpha", 90.0),
                    params.get("beta", 90.0),
                    params.get("gamma", 90.0),
                ]
            if "wavelength_min" in params or "wavelength_max" in params:
                wl = [
                    float(params.get("wavelength_min", 2.8)),
                    float(params.get("wavelength_max", 4.5)),
                ]
            else:
                wl = params.get("wavelength_range") or [2.8, 4.5]
            inst = str(params.get("instrument", "CG4D"))
            sg = str(params.get("space_group", "P 21 21 21"))
            merged_path = params.get("images_file") or "/home/xvg/subhkl-examples/betalac-7/merged.h5"
            work_dir = os.path.dirname(os.path.abspath(merged_path))
            solve_path = params.get("peaks_file") or os.path.join(work_dir, "solve.h5")
            if not os.path.isfile(solve_path) or os.path.basename(solve_path) in ("merged.h5", "integrated.h5"):
                cand_solve = os.path.join(work_dir, "solve.h5")
                if os.path.isfile(cand_solve):
                    solve_path = cand_solve

            tool_inputs: dict[str, Any] = {}
            if stage_id in ("reduce", "solve", "integrate"):
                required_tool = (
                    "imaginex_subhkl_prepare_peaks"
                    if stage_id == "reduce"
                    else "imaginex_subhkl_solve"
                )
                if not self._has_tool_on_galaxy(required_tool):
                    if (
                        self.galaxy_url
                        and "127.0.0.1" not in self.galaxy_url
                        and "localhost" not in self.galaxy_url
                    ):
                        if self._is_galaxy_server("http://127.0.0.1:9090"):
                            from nova.galaxy import Connection

                            job_record["stdout"] += (
                                f"[nova -> galaxy] '{required_tool}' not yet merged on {self.galaxy_url}; "
                                f"switching to local Galaxy instance http://127.0.0.1:9090 (GPU container runner)...\n"
                            )
                            local_conn = Connection(galaxy_url="http://127.0.0.1:9090", galaxy_key="test_key")
                            store = local_conn.connect().create_data_store("IMAGINE-X SubHKL")
                            gi = store.nova_connection.galaxy_instance
                            self.galaxy_url = "http://127.0.0.1:9090"
                            self.api_key = gi.key
                            self.history_id = store.history_id

                    if not self._has_tool_on_galaxy(required_tool):
                        raise RuntimeError(
                            f"subhkl GPU tool '{required_tool}' not yet deployed on {self.galaxy_url} "
                            f"(awaiting MR merge on ndip/galaxy-tools)"
                        )

            if stage_id == "solve":
                primary_tool = "imaginex_subhkl_solve"
                scan_id = self._ensure_galaxy_h5_dataset(store, merged_path, job_record)
                tool_inputs = {
                    "scan": {"src": "hda", "id": scan_id},
                    "instrument": inst,
                    "spaceGroup": sg,
                    "a": float(raw_cell[0]),
                    "b": float(raw_cell[1]),
                    "c": float(raw_cell[2]),
                    "alpha": float(raw_cell[3]),
                    "beta": float(raw_cell[4]),
                    "gamma": float(raw_cell[5]),
                    "wavelengthMin": float(wl[0]),
                    "wavelengthMax": float(wl[1]),
                    "refinement": str(params.get("objective", "band")),
                    "dMin": float(params.get("d_min", 2.8)),
                }
            elif stage_id == "integrate":
                primary_tool = "imaginex_subhkl_predict"
                scan_id = self._ensure_galaxy_h5_dataset(store, merged_path, job_record)
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                tool_inputs = {
                    "scan": {"src": "hda", "id": scan_id},
                    "indexer": {"src": "hda", "id": idx_id},
                    "instrument": inst,
                    "dMin": float(params.get("d_min", 2.8)),
                    "wavelengthMin": float(wl[0]),
                    "wavelengthMax": float(wl[1]),
                    "space": sg,
                }
            elif stage_id == "reduce":
                raw_nxs = self._resolve_raw_nexus_inputs(params, work_dir)
                if not raw_nxs:
                    raise RuntimeError(
                        f"Using pre-merged scan {os.path.basename(merged_path)} (no raw *.nxs.h5 specified or in {work_dir})"
                    )
                missing_nxs = [p for p in raw_nxs if p.startswith("/") and not os.path.isfile(p)]
                if missing_nxs:
                    chk = self.check_neutron_paths(
                        {
                            "facility": params.get("facility", "SNS" if missing_nxs[0].startswith("/SNS/") else "HFIR"),
                            "instrument": inst,
                            "ipts": params.get("ipts", ""),
                            "paths": raw_nxs,
                        }
                    )
                    raise FileNotFoundError(chk.get("message") or f"File not found: {missing_nxs[0]}")
                job_record["raw_nxs"] = raw_nxs
                job_record["instrument"] = inst
                primary_tool = "imaginex_subhkl_prepare_peaks"
                tool_inputs = {
                    "instrument": inst,
                    "a": float(raw_cell[0]),
                    "b": float(raw_cell[1]),
                    "c": float(raw_cell[2]),
                    "alpha": float(raw_cell[3]),
                    "beta": float(raw_cell[4]),
                    "gamma": float(raw_cell[5]),
                    "spaceGroup": sg,
                    "wavelengthMin": float(wl[0]),
                    "wavelengthMax": float(wl[1]),
                }
                for i_nxs, nxs_p in enumerate(raw_nxs):
                    tool_inputs[f"inputs_{i_nxs}|input"] = nxs_p
            else:
                primary_tool = "imaginex_subhkl_metrics"
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                tool_inputs = {
                    "peaks": {"src": "hda", "id": idx_id},
                }

            is_two_step_integrate = stage_id == "integrate" and self._has_tool_on_galaxy(
                "imaginex_subhkl_integrate"
            )
            job_record["tool_id"] = primary_tool
            results = gi.tools.run_tool(
                history_id=store.history_id,
                tool_id=primary_tool,
                tool_inputs=tool_inputs,
            )
            job_id = results["jobs"][0]["id"]
            job_record["job_id"] = job_id
            job_record["outputs"] = [
                {
                    "id": d.get("id"),
                    "name": d.get("output_name") or d.get("name"),
                    "extension": d.get("file_ext") or d.get("extension") or "h5",
                }
                for d in (results.get("outputs") or [])
                if isinstance(d, dict)
            ]
            step_lbl = "Step 1/2 " if is_two_step_integrate else ""
            job_record["stdout"] += (
                f"[galaxy -> pulsar] Dispatched {step_lbl}{primary_tool} (Galaxy job_id={job_id}) -> Pulsar queue on IMAGINEX-1\n"
            )
            self._poll_galaxy_job(
                job_record,
                None if is_two_step_integrate else on_complete_h5,
                final_step=not is_two_step_integrate,
                work_dir=work_dir,
            )

            # Chain Step 2/2: imaginex_subhkl_integrate (integrator + mtz-exporter)
            if (
                is_two_step_integrate
                and job_record.get("state") not in ("error", "deleted", "paused")
                and job_record.get("outputs")
            ):
                pred_ds_id = job_record["outputs"][0]["id"]
                gi.datasets.wait_for_dataset(pred_ds_id, maxwait=60, interval=1.5, check=True)
                int_tool = "imaginex_subhkl_integrate"
                int_inputs = {
                    "scan": {"src": "hda", "id": scan_id},
                    "predicted": {"src": "hda", "id": pred_ds_id},
                    "instrument": inst,
                    "anisotropic": True,
                    "mosaicFit": True,
                    "normalized": True,
                    "snr": 5.0,
                    "matrix": True,
                    "matrixProfile": "gaussian",
                    "createVisualizations": False,
                }
                job_record["tool_id"] = int_tool
                int_results = gi.tools.run_tool(
                    history_id=store.history_id,
                    tool_id=int_tool,
                    tool_inputs=int_inputs,
                )
                int_job_id = int_results["jobs"][0]["id"]
                job_record["job_id"] = int_job_id
                job_record["outputs"] = [
                    {
                        "id": d.get("id"),
                        "name": d.get("output_name") or d.get("name"),
                        "extension": d.get("file_ext") or d.get("extension") or "h5",
                    }
                    for d in (int_results.get("outputs") or [])
                    if isinstance(d, dict)
                ]
                job_record["stdout"] += (
                    f"\n[galaxy -> pulsar] Dispatched Step 2/2 {int_tool} (Galaxy job_id={int_job_id}) -> Pulsar queue on IMAGINEX-1\n"
                )
                self._poll_galaxy_job(
                    job_record,
                    on_complete_h5,
                    final_step=True,
                    work_dir=work_dir,
                )
        except Exception as exc:
            clean_msg = self._clean_err_msg(exc)
            job_record["stdout"] += f"[nova -> galaxy] {clean_msg}; running GPU container on IMAGINEX-1...\n"
            self._submit_local_subhkl_stage(
                stage_id,
                params,
                on_complete_h5,
                fallback_reason=clean_msg,
                existing_job_record=job_record,
            )

    def check_neutron_paths(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate NeXus / HDF5 file paths for the Load Neutron Data modal and suggest IPTS if mismatched."""
        import glob as _glob
        import re as _re

        fac = str(body.get("facility") or "HFIR").strip()
        inst = str(body.get("instrument") or "CG4D").strip()
        ipts_raw = _re.sub(r"^IPTS-", "", str(body.get("ipts") or "").strip(), flags=_re.I)
        raw_paths = body.get("paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [s.strip() for s in raw_paths.replace("\n", ",").split(",") if s.strip()]

        expanded: list[str] = []
        for p in raw_paths:
            p_str = str(p).strip()
            if not p_str:
                continue
            if any(ch in p_str for ch in ("*", "?", "[")):
                matches = sorted(_glob.glob(p_str))
                if matches:
                    expanded.extend(matches)
                    continue
            expanded.append(p_str)

        existing = [p for p in expanded if os.path.isfile(p)]
        missing = [p for p in expanded if not os.path.isfile(p)]
        ipts_dir = f"/{fac}/{inst}/IPTS-{ipts_raw}" if ipts_raw else ""
        ipts_exists = bool(ipts_dir and os.path.isdir(ipts_dir))

        suggested_ipts: str | None = None
        suggested_paths: list[str] = []
        if missing:
            # Try to locate the first missing run across /<facility>/<instrument>/IPTS-*/nexus/
            first_missing_base = os.path.basename(missing[0])
            candidates = sorted(_glob.glob(f"/{fac}/{inst}/IPTS-*/nexus/{first_missing_base}"))
            if not candidates:
                alt_fac = "SNS" if fac == "HFIR" else "HFIR"
                candidates = sorted(_glob.glob(f"/{alt_fac}/{inst}/IPTS-*/nexus/{first_missing_base}"))
            if candidates:
                m_ipts = _re.search(r"/IPTS-([^/]+)/", candidates[0])
                if m_ipts:
                    suggested_ipts = m_ipts.group(1)
                    sug_fac = "SNS" if candidates[0].startswith("/SNS/") else "HFIR"
                    for p in expanded:
                        bn = os.path.basename(p)
                        sp = f"/{sug_fac}/{inst}/IPTS-{suggested_ipts}/nexus/{bn}"
                        if os.path.isfile(sp):
                            suggested_paths.append(sp)

        sample_title = ""
        sample_goniometer = ""
        probe_file = existing[0] if existing else (suggested_paths[0] if suggested_paths else None)
        if probe_file and os.path.isfile(probe_file):
            try:
                import h5py
                from subhkl.instrument.goniometer import (
                    _read_nexus_title,
                    get_rotation_data_from_nexus,
                )

                with h5py.File(probe_file, "r") as f:
                    sample_title = _read_nexus_title(f).strip()
                _, angs, nms = get_rotation_data_from_nexus(probe_file, inst)
                seen_m: set[str] = set()
                parts_g: list[str] = []
                for nm, ang in zip(nms, angs):
                    short_m = nm.split(":")[-1]
                    if short_m not in seen_m:
                        seen_m.add(short_m)
                        parts_g.append(f"{short_m}={ang:.1f}°")
                sample_goniometer = " ".join(parts_g)
            except Exception:
                pass

        ok = len(existing) > 0 and len(missing) == 0
        if ok:
            msg = f"✓ Verified {len(existing)} file(s)"
            if sample_title:
                msg += f' · "{sample_title}"'
            if sample_goniometer:
                msg += f" ({sample_goniometer})"
        elif suggested_ipts:
            msg = (
                f"Not found in IPTS-{ipts_raw or '?'} — "
                f"found {len(suggested_paths)}/{len(expanded)} run(s) in IPTS-{suggested_ipts}"
            )
            if sample_title:
                msg += f' ("{sample_title}")'
        elif ipts_raw and not ipts_exists:
            msg = f"Directory /{fac}/{inst}/IPTS-{ipts_raw} does not exist"
        elif missing:
            msg = f"File not found: {missing[0]}" + (
                f" (+{len(missing) - 1} more)" if len(missing) > 1 else ""
            )
        else:
            msg = "No NeXus / HDF5 file paths specified"

        return {
            "ok": ok,
            "existing": existing,
            "missing": missing,
            "ipts_exists": ipts_exists,
            "suggested_ipts": suggested_ipts,
            "suggested_paths": suggested_paths,
            "sample_title": sample_title,
            "sample_goniometer": sample_goniometer,
            "message": msg,
        }

    def _resolve_raw_nexus_inputs(self, params: dict[str, Any], work_dir: str) -> list[str]:
        """Resolve raw .nxs.h5 paths from explicit nexus_files, galaxy_dataset_id, IPTS+runs, or work_dir."""
        import glob as _glob

        resolved: list[str] = []
        raw_list = params.get("nexus_files")
        if isinstance(raw_list, str):
            raw_list = [s.strip() for s in raw_list.replace("\n", ",").split(",") if s.strip()]
        if isinstance(raw_list, list) and raw_list:
            for item in raw_list:
                p_str = str(item).strip()
                if not p_str:
                    continue
                if any(ch in p_str for ch in ("*", "?", "[")):
                    matches = sorted(_glob.glob(p_str))
                    if matches:
                        resolved.extend(matches)
                        continue
                resolved.append(os.path.abspath(p_str))
            if resolved:
                return resolved

        ds_id = params.get("galaxy_dataset_id")
        if ds_id:
            try:
                local_p, _ = self.resolve_dataset_to_local_h5(str(ds_id), fallback_dir=work_dir)
                if local_p and os.path.isfile(local_p):
                    return [local_p]
            except Exception:
                pass

        if work_dir and os.path.isdir(work_dir):
            return sorted(_glob.glob(os.path.join(work_dir, "*.nxs.h5")))
        return []

    def _submit_local_subhkl_stage(
        self,
        stage_id: str,
        params: dict[str, Any],
        on_complete_h5: Any = None,
        fallback_reason: str | None = None,
        existing_job_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run subhkl stage inside the GPU Docker/Podman container on IMAGINEX-1 and stream live stdout/stderr."""
        import shlex
        import shutil
        import socket
        import subprocess
        import sys

        if all(k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")):
            raw_cell = [
                params["a"],
                params["b"],
                params["c"],
                params["alpha"],
                params["beta"],
                params["gamma"],
            ]
        else:
            raw_cell = params.get("cell") or [
                params.get("a", 68.7),
                params.get("b", 69.8),
                params.get("c", 156.9),
                params.get("alpha", 90.0),
                params.get("beta", 90.0),
                params.get("gamma", 90.0),
            ]
        cell_str = ",".join(str(float(x)) for x in raw_cell[:6])
        if "wavelength_min" in params or "wavelength_max" in params:
            wl = [
                float(params.get("wavelength_min", 2.8)),
                float(params.get("wavelength_max", 4.5)),
            ]
        else:
            wl = params.get("wavelength_range") or [2.8, 4.5]
        inst = params.get("instrument", "CG4D")
        sg = params.get("space_group", "P 21 21 21")
        merged_path = os.path.abspath(
            params.get("images_file") or "/home/xvg/subhkl-examples/betalac-7/merged.h5"
        )
        work_dir = os.path.dirname(merged_path)
        solve_dest = os.path.join(work_dir, "solve.h5")
        pred_dest = os.path.join(work_dir, "predictor.h5")
        integrated_dest = os.path.join(work_dir, "integrated.h5")
        mtz_dest = os.path.join(work_dir, "reflections.mtz")

        solve_path = os.path.abspath(params.get("peaks_file") or solve_dest)
        if os.path.basename(solve_path) in ("merged.h5", "integrated.h5", "predictor.h5") and os.path.isfile(solve_dest):
            solve_path = solve_dest

        ts = int(time.time())
        out_h5 = os.path.join(tempfile.gettempdir(), f"subhkl_{stage_id}_{ts}.h5")
        pred_tmp_h5 = os.path.join(tempfile.gettempdir(), f"subhkl_pred_{ts}.h5")
        mtz_tmp_path = os.path.join(tempfile.gettempdir(), f"subhkl_refl_{ts}.mtz")
        reduce_tmp_dir = os.path.join(tempfile.gettempdir(), f"subhkl_reduce_{ts}")
        raw_nxs_local: list[str] = []
        if stage_id == "reduce":
            raw_nxs_local = [
                p for p in self._resolve_raw_nexus_inputs(params, work_dir) if os.path.isfile(p)
            ]

        if stage_id == "reduce" and raw_nxs_local:
            os.makedirs(reduce_tmp_dir, exist_ok=True)
            sitecustomize_path = os.path.join(reduce_tmp_dir, "sitecustomize.py")
            with open(sitecustomize_path, "w", encoding="utf-8") as sc_f:
                sc_f.write(
                    "import re, h5py, numpy as np\n"
                    "try:\n"
                    "    import subhkl.instrument.goniometer as _gonio\n"
                    "    _orig_get_rot = _gonio.get_rotation_data_from_nexus\n"
                    "    _PATS = {\n"
                    "        'phi': re.compile(r'(?:axis0|phi)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "        'kappa': re.compile(r'(?:axis1|kappa)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "        'omega': re.compile(r'(?:axis2|omega)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "        'chi': re.compile(r'(?:chi)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "        'alpha': re.compile(r'(?:alpha)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "    }\n"
                    "    _DEF = {'omega': 0.0, 'alpha': 45.0, 'kappa': 0.0, 'phi': 0.0, 'chi': 0.0}\n"
                    "    def _patched_get_rot(filename, instrument):\n"
                    "        settings = _gonio.reduction_settings[instrument]\n"
                    "        with h5py.File(filename, 'r') as f:\n"
                    "            das = f['entry/DASlogs'] if 'entry/DASlogs' in f else {}\n"
                    "            has_all = all(k.split('#')[0] in das and 'average_value' in das[k.split('#')[0]] for k in settings['Goniometer'])\n"
                    "            if has_all:\n"
                    "                return _orig_get_rot(filename, instrument)\n"
                    "            t = ''\n"
                    "            if 'entry/title' in f:\n"
                    "                raw = f['entry/title'][()]\n"
                    "                if isinstance(raw, np.ndarray) and raw.size > 0: raw = raw.flat[0]\n"
                    "                t = raw.decode('utf-8', 'replace') if isinstance(raw, (bytes, np.bytes_)) else str(raw)\n"
                    "            parsed = {k: float(m.group(1)) for k, p in _PATS.items() if (m := p.search(t))}\n"
                    "            axes, angles, names = [], [], []\n"
                    "            for ax_name, ax_spec in settings['Goniometer'].items():\n"
                    "                ln = ax_name.split('#')[0]\n"
                    "                sk = ln.split(':')[-1].lower()\n"
                    "                val = float(das[ln]['average_value'][0]) if (ln in das and 'average_value' in das[ln]) else parsed.get(sk, _DEF.get(sk))\n"
                    "                if val is not None:\n"
                    "                    axes.append(np.array(ax_spec, dtype=float))\n"
                    "                    angles.append(float(val))\n"
                    "                    names.append(ln)\n"
                    "            return axes, angles, names\n"
                    "    _gonio.get_rotation_data_from_nexus = _patched_get_rot\n"
                    "except Exception:\n"
                    "    pass\n"
                )
            reduce_cmds: list[str] = []
            for i_nxs, nxs_p in enumerate(raw_nxs_local):
                red_out = os.path.join(reduce_tmp_dir, f"run_{i_nxs:03d}.reduce.h5")
                r_cmd = [
                    "python",
                    "-u",
                    "-m",
                    "subhkl.io.parser",
                    "reduce",
                    nxs_p,
                    red_out,
                    str(inst),
                    "--wavelength-min",
                    str(wl[0]),
                    "--wavelength-max",
                    str(wl[1]),
                ]
                reduce_cmds.append(shlex.join(r_cmd))
            merge_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "merge-images",
                os.path.join(reduce_tmp_dir, "*.reduce.h5"),
                out_h5,
                str(float(raw_cell[0])),
                str(float(raw_cell[1])),
                str(float(raw_cell[2])),
                str(float(raw_cell[3])),
                str(float(raw_cell[4])),
                str(float(raw_cell[5])),
                str(sg),
            ]
            reduce_cmds.append(shlex.join(merge_cmd))
            chained_reduce = (
                f"set -e; export PYTHONPATH={shlex.quote(reduce_tmp_dir)}:${{PYTHONPATH:-}}; "
                + " && ".join(reduce_cmds)
            )
            inner_cmd = ["bash", "-c", chained_reduce]
        elif stage_id == "reduce" and os.path.isfile(merged_path):
            inner_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "sum-images",
                out_h5,
                merged_path,
            ]
        elif stage_id == "solve" and os.path.isfile(merged_path):
            inner_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "solve",
                merged_path,
                out_h5,
                "--instrument",
                str(inst),
                "--cell",
                cell_str,
                "--space-group",
                str(sg),
                "--wavelength-min",
                str(wl[0]),
                "--wavelength-max",
                str(wl[1]),
                "--d-min",
                str(params.get("d_min", 2.8)),
                "--refinement",
                str(params.get("objective", "band")),
            ]
        elif stage_id == "integrate" and os.path.isfile(merged_path) and os.path.isfile(solve_path):
            d_min_val = str(params.get("d_min", 2.8))
            pred_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "peak-predictor",
                merged_path,
                str(inst),
                solve_path,
                pred_tmp_h5,
                "--d-min",
                d_min_val,
                "--wavel-min",
                str(wl[0]),
                "--wavel-max",
                str(wl[1]),
                "--space-group",
                str(sg),
            ]
            int_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "integrator",
                merged_path,
                str(inst),
                pred_tmp_h5,
                out_h5,
                "--anisotropic",
                "--fit-mosaicity",
                "--shape-fit-normalized",
                "--matrix-free",
                "--show-progress",
            ]
            mtz_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "mtz-exporter",
                out_h5,
                mtz_tmp_path,
                "--predictions-file",
                pred_tmp_h5,
                "--corrections-file",
                solve_path,
            ]
            chained = (
                f"set -e; "
                f"{shlex.join(pred_cmd)} && "
                f"{shlex.join(int_cmd)} && "
                f"{shlex.join(mtz_cmd)}"
            )
            inner_cmd = ["bash", "-c", chained]
        elif stage_id == "merge" and os.path.isfile(integrated_dest if os.path.isfile(integrated_dest) else solve_path):
            target_h5 = integrated_dest if os.path.isfile(integrated_dest) else solve_path
            inner_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "metrics",
                target_h5,
                "--instrument",
                str(inst),
                "--per-run",
            ]
        else:
            inner_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "indexer-visualize",
                solve_path,
                "--instrument",
                str(inst),
                "--output-dir",
                tempfile.gettempdir(),
                "--dpi",
                "72",
            ]

        container_image = "ghcr.io/neutrons/subhkl:feat-rainbow-zones"
        if shutil.which("podman"):
            raw_dirs = {os.path.dirname(os.path.abspath(p)) for p in raw_nxs_local}
            mount_dirs = sorted(
                {
                    os.path.dirname(merged_path),
                    os.path.dirname(solve_path),
                    tempfile.gettempdir(),
                    *raw_dirs,
                }
            )
            cmd = [
                "podman",
                "run",
                "--rm",
                "--pids-limit=-1",
                "--security-opt",
                "label=disable",
                "--device",
                "nvidia.com/gpu=all",
                "-e",
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
                "-e",
                "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "MKL_NUM_THREADS=1",
            ]
            for d in mount_dirs:
                if os.path.isdir(d):
                    cmd.extend(["-v", f"{d}:{d}"])
            cmd.append(container_image)
            cmd.extend(inner_cmd)
        else:
            if inner_cmd[0] == "bash":
                cmd = inner_cmd
            else:
                cmd = [sys.executable] + inner_cmd[1:]

        gpu_env = os.environ.copy()
        gpu_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        gpu_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        gpu_env.setdefault("OMP_NUM_THREADS", "1")
        gpu_env.setdefault("MKL_NUM_THREADS", "1")
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                text=True,
                timeout=2.0,
            ).strip().splitlines()[0]
        except Exception:
            gpu_info = "CPU fallback"

        host_tag = socket.gethostname()
        header_lines = []
        if fallback_reason and existing_job_record is None:
            header_lines.append(f"[ndip] Note: {fallback_reason}")
        header_lines.append(
            f"[container] Host={host_tag} | GPU={gpu_info} | Image={container_image}"
        )
        cmd_str = " ".join(cmd)
        header_lines.append(f"$ {cmd_str}\n")

        job_id = f"gpu-{int(time.time())}"
        if existing_job_record is not None:
            job_record = existing_job_record
            job_record["job_id"] = job_id
            job_record["mode"] = "container_gpu"
            job_record["state"] = "running"
            job_record["command_line"] = cmd_str
            job_record["stdout"] = (job_record.get("stdout", "") + "\n".join(header_lines))[-24000:]
        else:
            job_record = {
                "job_id": job_id,
                "tool_id": f"subhkl::{stage_id}",
                "stage_id": stage_id,
                "mode": "container_gpu",
                "state": "running",
                "command_line": cmd_str,
                "stdout": "\n".join(header_lines),
                "stderr": "",
                "ladder_rung_idx": 0 if stage_id == "solve" else 4,
                "outputs": [{"id": job_id, "name": os.path.basename(out_h5), "extension": "h5"}],
                "updated_payload_ready": False,
                "started_at": time.time(),
            }
            self.active_job = job_record

        def _stream_proc() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=gpu_env,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    job_record["stdout"] = (job_record.get("stdout", "") + line)[-24000:]
                    low = line.lower()
                    if stage_id == "solve":
                        if "ladder" in low or "blind" in low:
                            job_record["ladder_rung_idx"] = max(job_record["ladder_rung_idx"], 1)
                        elif "global" in low or "60-zone" in low:
                            job_record["ladder_rung_idx"] = max(job_record["ladder_rung_idx"], 2)
                        elif "120" in low or "cell" in low or "stage" in low:
                            job_record["ladder_rung_idx"] = max(job_record["ladder_rung_idx"], 3)
                rc = proc.wait()
                if rc == 0:
                    job_record["ladder_rung_idx"] = 0 if stage_id == "reduce" else 4
                    final_h5 = out_h5
                    if stage_id == "reduce" and os.path.isfile(out_h5):
                        merged_dest = os.path.join(work_dir, "merged.h5")
                        try:
                            shutil.copy2(out_h5, merged_dest)
                            final_h5 = merged_dest
                            job_record["stdout"] += f"\n[subhkl] Saved merged.h5 -> {merged_dest}\n"
                        except Exception as copy_exc:
                            job_record["stdout"] += f"\n[subhkl] Warning saving {merged_dest}: {copy_exc}\n"

                        if self.is_connected and os.path.isfile(final_h5):
                            try:
                                store = self.get_datastore("IMAGINE-X SubHKL")
                                self._ensure_galaxy_h5_dataset(
                                    store, final_h5, job_record, force_upload=True, file_ext="h5"
                                )
                            except Exception as up_exc:
                                job_record["stdout"] += (
                                    f"[nova -> galaxy] Warning uploading merged.h5: {self._clean_err_msg(up_exc)}\n"
                                )
                    elif stage_id == "solve" and os.path.isfile(out_h5):
                        try:
                            shutil.copy2(out_h5, solve_dest)
                            for suffix in (".stages.h5", ".blind.h5"):
                                src_sub = out_h5[:-3] + suffix
                                dst_sub = solve_dest[:-3] + suffix
                                if os.path.isfile(src_sub):
                                    shutil.copy2(src_sub, dst_sub)
                            final_h5 = solve_dest
                            job_record["stdout"] += f"\n[subhkl] Saved solve.h5 -> {solve_dest}\n"
                        except Exception as copy_exc:
                            job_record["stdout"] += f"\n[subhkl] Warning saving {solve_dest}: {copy_exc}\n"

                        if self.is_connected and os.path.isfile(final_h5):
                            try:
                                store = self.get_datastore("IMAGINE-X SubHKL")
                                self._ensure_galaxy_h5_dataset(
                                    store, final_h5, job_record, force_upload=True, file_ext="h5"
                                )
                            except Exception as up_exc:
                                job_record["stdout"] += (
                                    f"[nova -> galaxy] Warning uploading solve.h5: {self._clean_err_msg(up_exc)}\n"
                                )
                    elif stage_id == "integrate" and os.path.isfile(out_h5):
                        try:
                            if os.path.isfile(pred_tmp_h5):
                                shutil.copy2(pred_tmp_h5, pred_dest)
                                job_record["stdout"] += f"\n[subhkl] Saved predictor.h5 -> {pred_dest}\n"
                            shutil.copy2(out_h5, integrated_dest)
                            final_h5 = integrated_dest
                            job_record["stdout"] += f"[subhkl] Saved integrated.h5 -> {integrated_dest}\n"
                            if os.path.isfile(mtz_tmp_path):
                                shutil.copy2(mtz_tmp_path, mtz_dest)
                                job_record["stdout"] += f"[subhkl] Saved reflections.mtz -> {mtz_dest}\n"
                        except Exception as copy_exc:
                            job_record["stdout"] += f"\n[subhkl] Warning saving integrated outputs: {copy_exc}\n"

                        if self.is_connected:
                            try:
                                store = self.get_datastore("IMAGINE-X SubHKL")
                                if os.path.isfile(pred_dest):
                                    self._ensure_galaxy_h5_dataset(
                                        store, pred_dest, job_record, force_upload=True, file_ext="h5"
                                    )
                                if os.path.isfile(final_h5):
                                    self._ensure_galaxy_h5_dataset(
                                        store, final_h5, job_record, force_upload=True, file_ext="h5"
                                    )
                                if os.path.isfile(mtz_dest):
                                    self._ensure_galaxy_h5_dataset(
                                        store, mtz_dest, job_record, force_upload=True, file_ext="binary"
                                    )
                            except Exception as up_exc:
                                job_record["stdout"] += (
                                    f"[nova -> galaxy] Warning uploading integrated outputs: {self._clean_err_msg(up_exc)}\n"
                                )

                    job_record["state"] = "ok"
                    job_record["stdout"] += f"[subhkl] Stage '{stage_id}' completed (exit 0).\n"
                    if os.path.isfile(final_h5) and callable(on_complete_h5):
                        on_complete_h5(final_h5)
                        job_record["updated_payload_ready"] = True
                else:
                    job_record["state"] = "error"
                    job_record["stderr"] += f"\n[subhkl] Process exited with code {rc}\n"
            except Exception as exc:
                job_record["state"] = "error"
                job_record["stderr"] += f"\n[subhkl] Error: {exc}\n"

        threading.Thread(target=_stream_proc, daemon=True).start()
        return {
            "stage_id": stage_id,
            "mode": "local_stream",
            "galaxy_url": self.galaxy_url,
            "history_id": self.history_id,
            "job": job_record,
            "status": "submitted",
        }

    def _poll_galaxy_job(
        self,
        job_record: dict[str, Any],
        on_complete_h5: Any = None,
        nova_job: Any = None,
        *,
        final_step: bool = True,
        work_dir: str | None = None,
    ) -> None:
        job_id = job_record.get("job_id")
        if not job_id:
            return
        step_counter = 0
        poll_errors = 0
        base_log = job_record.get("stdout", "")
        seen_pulsar_dirs: set[str] = set()
        latest_pulsar_dir: str | None = None
        for droot in ("/data/staging_test", "/data/staging"):
            if os.path.isdir(droot):
                try:
                    for entry in os.listdir(droot):
                        seen_pulsar_dirs.add(os.path.join(droot, entry))
                except Exception:
                    pass

        while True:
            time.sleep(1.5)
            try:
                # 1. Detect live Pulsar staging directory on this GPU node (IMAGINEX-1)
                pulsar_live_out = ""
                pulsar_live_err = ""
                for droot in ("/data/staging_test", "/data/staging"):
                    if os.path.isdir(droot):
                        try:
                            for entry in os.listdir(droot):
                                full_dir = os.path.join(droot, entry)
                                if not os.path.isdir(full_dir):
                                    continue
                                if full_dir not in seen_pulsar_dirs:
                                    seen_pulsar_dirs.add(full_dir)
                                    latest_pulsar_dir = full_dir
                                    pid_path = os.path.join(full_dir, "pid")
                                    ctr_path = os.path.join(
                                        full_dir, "metadata", "__instrument_core_container"
                                    )
                                    pid_str = (
                                        open(pid_path).read().strip()
                                        if os.path.isfile(pid_path)
                                        else "?"
                                    )
                                    ctr_str = (
                                        open(ctr_path).read().strip()
                                        if os.path.isfile(ctr_path)
                                        else "docker"
                                    )
                                    base_log += (
                                        f"[pulsar] Staged on IMAGINEX-1: {full_dir} "
                                        f"(PID={pid_str}, container={ctr_str})\n"
                                    )
                                p_out = os.path.join(full_dir, "metadata", "job_stdout")
                                p_err = os.path.join(full_dir, "metadata", "job_stderr")
                                if os.path.isfile(p_out):
                                    pulsar_live_out = open(p_out, errors="replace").read()
                                if os.path.isfile(p_err):
                                    pulsar_live_err = open(p_err, errors="replace").read()
                        except Exception:
                            pass

                # 2. Fetch job status & live Pulsar console_output stream from NDIP Galaxy
                jinfo = self._api_request("GET", f"/api/jobs/{job_id}?full=true")
                poll_errors = 0
                state = jinfo.get("state", "running")
                if state != "ok":
                    job_record["state"] = state
                else:
                    job_record["state"] = "running"
                if jinfo.get("command_line"):
                    job_record["command_line"] = jinfo["command_line"]

                console_out = ""
                console_err = ""
                try:
                    cinfo = self._api_request(
                        "GET",
                        f"/api/jobs/{job_id}/console_output?stdout_position=0&stdout_length=65536&stderr_position=0&stderr_length=65536",
                    )
                    if isinstance(cinfo, dict):
                        console_out = cinfo.get("stdout") or ""
                        console_err = cinfo.get("stderr") or ""
                except Exception:
                    pass

                out_txt = (
                    console_out
                    or pulsar_live_out
                    or jinfo.get("tool_stdout")
                    or jinfo.get("stdout")
                    or ""
                )
                err_txt = (
                    console_err
                    or pulsar_live_err
                    or jinfo.get("tool_stderr")
                    or jinfo.get("stderr")
                    or ""
                )
                cmd_header = (
                    f"$ {job_record['command_line']}\n" if job_record.get("command_line") else ""
                )
                job_record["stdout"] = f"{base_log}{cmd_header}{out_txt}"[-24000:]
                if err_txt:
                    job_record["stderr"] = err_txt[-16000:]

                if job_record.get("stage_id") == "solve" and state in ("queued", "running"):
                    step_counter = min(3, step_counter + 1)
                    job_record["ladder_rung_idx"] = step_counter
                if state in ("ok", "error", "deleted", "paused"):
                    if state == "ok":
                        try:
                            job_record["ladder_rung_idx"] = (
                                0 if job_record.get("stage_id") == "reduce" else 4
                            )
                            job_record["stdout"] += (
                                f"\n[galaxy -> pulsar] Job {job_id} finished (state=ok).\n"
                            )
                            h5_outs = [
                                o
                                for o in job_record.get("outputs", [])
                                if o.get("extension") in ("h5", "hdf5")
                            ]
                            mtz_outs = [
                                o
                                for o in job_record.get("outputs", [])
                                if o.get("extension") == "mtz"
                                or "mtz" in str(o.get("name", "")).lower()
                            ]
                            if h5_outs:
                                ds_id = h5_outs[0]["id"]
                                target_path = None
                                if work_dir and os.path.isdir(work_dir):
                                    if not final_step and job_record.get("stage_id") == "integrate":
                                        target_path = os.path.join(work_dir, "predictor.h5")
                                    elif job_record.get("stage_id") == "integrate":
                                        target_path = os.path.join(work_dir, "integrated.h5")
                                    elif job_record.get("stage_id") == "solve":
                                        target_path = os.path.join(work_dir, "solve.h5")
                                    elif job_record.get("stage_id") == "reduce":
                                        target_path = os.path.join(work_dir, "merged.h5")

                                # Check if Pulsar staging still has the output file locally before downloading over HTTPS
                                local_staged_h5 = None
                                if latest_pulsar_dir and os.path.isdir(latest_pulsar_dir):
                                    for rel_cand in (
                                        "working/out/scan_master.h5",
                                        "working/out/solve.h5",
                                        "working/out/peak_predictor.h5",
                                        "working/out/integrated.h5",
                                    ):
                                        cand_p = os.path.join(latest_pulsar_dir, rel_cand)
                                        if os.path.isfile(cand_p) and os.path.getsize(cand_p) > 0:
                                            local_staged_h5 = cand_p
                                            break

                                tmp_path = None
                                if local_staged_h5:
                                    import shutil as _shutil

                                    if target_path:
                                        _shutil.copyfile(local_staged_h5, target_path)
                                        tmp_path = target_path
                                        job_record["stdout"] += f"[subhkl] Saved {os.path.basename(target_path)} -> {target_path}\n"
                                    else:
                                        fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="galaxy_solve_")
                                        os.close(fd)
                                        _shutil.copyfile(local_staged_h5, tmp_path)
                                else:
                                    raw_bytes = None
                                    for attempt in range(3):
                                        try:
                                            raw_bytes = self._api_request(
                                                "GET",
                                                f"/api/datasets/{ds_id}/display",
                                                raw=True,
                                                timeout=180.0,
                                            )
                                            if raw_bytes:
                                                break
                                        except Exception as dl_exc:
                                            if attempt == 2:
                                                raise dl_exc
                                            time.sleep(2.0)
                                    if raw_bytes:
                                        if target_path:
                                            with open(target_path, "wb") as f:
                                                f.write(raw_bytes)
                                            tmp_path = target_path
                                            job_record["stdout"] += f"[subhkl] Saved {os.path.basename(target_path)} -> {target_path}\n"
                                        else:
                                            fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="galaxy_solve_")
                                            with os.fdopen(fd, "wb") as f:
                                                f.write(raw_bytes)

                                # If reduce ran inside a container without CG4D title goniometer fallback, patch angles in merged.h5
                                if (
                                    tmp_path
                                    and os.path.isfile(tmp_path)
                                    and job_record.get("stage_id") == "reduce"
                                    and job_record.get("raw_nxs")
                                ):
                                    try:
                                        import h5py
                                        import numpy as np
                                        from subhkl.instrument.goniometer import get_rotation_data_from_nexus

                                        raw_nxs_list = job_record["raw_nxs"]
                                        inst_name = str(job_record.get("instrument") or "CG4D")
                                        with h5py.File(tmp_path, "r+") as hf:
                                            if "goniometer/angles" in hf:
                                                cur_ang = np.asarray(hf["goniometer/angles"][()])
                                                if np.allclose(cur_ang, 0.0):
                                                    new_axes, new_angles, new_names = [], [], []
                                                    for nxs_f in raw_nxs_list:
                                                        if os.path.isfile(nxs_f):
                                                            ax, ang, nms = get_rotation_data_from_nexus(nxs_f, inst_name)
                                                            new_axes.append(ax)
                                                            new_angles.append(ang)
                                                            new_names.append(nms)
                                                    if new_angles and not np.allclose(new_angles, 0.0):
                                                        del hf["goniometer/angles"]
                                                        hf["goniometer/angles"] = np.asarray(new_angles, dtype=np.float64)
                                                        if "goniometer/axes" in hf:
                                                            del hf["goniometer/axes"]
                                                        hf["goniometer/axes"] = np.asarray(new_axes, dtype=np.float64)
                                                        if "goniometer/names" in hf:
                                                            del hf["goniometer/names"]
                                                        hf["goniometer/names"] = np.asarray(new_names, dtype="S")
                                                        job_record["stdout"] += (
                                                            f"[subhkl] Patched goniometer angles from NeXus title into {os.path.basename(tmp_path)}\n"
                                                        )
                                    except Exception:
                                        pass

                                if final_step and tmp_path and os.path.isfile(tmp_path) and callable(on_complete_h5):
                                    on_complete_h5(tmp_path)
                                    job_record["updated_payload_ready"] = True
                            if final_step and mtz_outs and work_dir and os.path.isdir(work_dir):
                                try:
                                    mtz_id = mtz_outs[0]["id"]
                                    mtz_bytes = self._api_request(
                                        "GET",
                                        f"/api/datasets/{mtz_id}/display",
                                        raw=True,
                                        timeout=180.0,
                                    )
                                    mtz_dest = os.path.join(work_dir, "reflections.mtz")
                                    with open(mtz_dest, "wb") as f:
                                        f.write(mtz_bytes)
                                    job_record["stdout"] += f"[subhkl] Saved reflections.mtz -> {mtz_dest}\n"
                                except Exception:
                                    pass
                        except Exception as post_exc:
                            job_record["stdout"] += (
                                f"[nova -> galaxy] Warning fetching job output: {self._clean_err_msg(post_exc)}\n"
                            )
                        finally:
                            if final_step:
                                job_record["state"] = "ok"
                    else:
                        job_record["state"] = state
                    break
            except Exception as exc:
                poll_errors += 1
                if poll_errors < 5:
                    continue
                job_record["error"] = str(exc)
                if job_record.get("state") in ("queued", "running", None):
                    job_record["state"] = "error"
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
            from subhkl.viz.nova_app.views.main_view import MainApp

            nova_app = MainApp(
                peaks_h5=replay_h5,
                images_h5=payload_kwargs.get("images_h5"),
                instrument=payload_kwargs.get("instrument") or "CG4D",
            )
            nova_app.server.start(
                host=host,
                port=port,
                open_browser=False,
                timeout=0,
            )
            return
        except Exception:
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
                if "file_path" in qs and qs["file_path"]:
                    try:
                        fp = os.path.abspath(str(qs["file_path"][0]).strip())
                        if os.path.isfile(fp):
                            fp_low = os.path.basename(fp).lower()
                            if any(k in fp_low for k in ("merged", "scan", "image")):
                                vm.payload_kwargs["images_h5"] = fp
                                vm.replay_h5 = fp
                                vm.reload(images_h5=fp)
                            else:
                                vm.replay_h5 = fp
                                vm.reload()
                    except Exception:
                        pass
                if "dataset_id" in qs and qs["dataset_id"]:
                    try:
                        ds_id = str(qs["dataset_id"][0]).strip()
                        fb_dir = (
                            os.path.dirname(os.path.abspath(vm.replay_h5))
                            if vm.replay_h5
                            else None
                        )
                        local_h5, ds_info = (
                            vm.pipeline_runner.resolve_dataset_to_local_h5(
                                ds_id, fallback_dir=fb_dir
                            )
                        )
                        ds_name = str(ds_info.get("name") or os.path.basename(local_h5)).lower()
                        if any(k in ds_name for k in ("merged", "scan", "image")):
                            vm.payload_kwargs["images_h5"] = local_h5
                            vm.replay_h5 = local_h5
                            vm.reload(images_h5=local_h5)
                        else:
                            vm.replay_h5 = local_h5
                            vm.reload()
                    except Exception:
                        pass
                if "frame" in qs and qs["frame"]:
                    try:
                        vm.reload(frame_index=int(qs["frame"][0]))
                    except Exception:
                        pass
                body = json.dumps(vm.payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path.startswith("/api/datasets/") and req_path.endswith("/display"):
                ds_id = req_path.split("/")[3]
                try:
                    fb_dir = (
                        os.path.dirname(os.path.abspath(vm.replay_h5))
                        if vm.replay_h5
                        else None
                    )
                    local_h5, _ = vm.pipeline_runner.resolve_dataset_to_local_h5(
                        ds_id, fallback_dir=fb_dir
                    )
                    with open(local_h5, "rb") as fh:
                        h5_bytes = fh.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-hdf5")
                    self.send_header("Content-Length", str(len(h5_bytes)))
                    self.end_headers()
                    self.wfile.write(h5_bytes)
                except Exception as exc:
                    err_body = str(exc).encode("utf-8")
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(err_body)))
                    self.end_headers()
                    self.wfile.write(err_body)
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

            if req_path == "/api/pipeline/check_neutron_paths":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    result = vm.pipeline_runner.check_neutron_paths(req_json)
                    code = 200
                except Exception as exc:
                    result = {"ok": False, "error": str(exc), "message": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/run":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    stage_id = req_json.get("stage_id", "solve")
                    params = dict(req_json.get("params") or {})
                    params.setdefault("instrument", vm.payload.get("instrument", "CG4D"))
                    params.setdefault("space_group", vm.payload.get("space_group", "P 1"))
                    if all(k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")):
                        params["cell"] = [
                            float(params["a"]),
                            float(params["b"]),
                            float(params["c"]),
                            float(params["alpha"]),
                            float(params["beta"]),
                            float(params["gamma"]),
                        ]
                    elif vm.payload.get("cell"):
                        params.setdefault("cell", vm.payload.get("cell"))
                    if vm.payload.get("wavelength_range"):
                        params.setdefault("wavelength_range", vm.payload.get("wavelength_range"))

                    def _on_complete(new_h5_path: str) -> None:
                        if stage_id == "reduce" or "merged" in os.path.basename(new_h5_path).lower():
                            vm.payload_kwargs["images_h5"] = new_h5_path
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

            if req_path == "/api/pipeline/upload_h5":
                import email.parser

                content_len = int(self.headers.get("Content-Length", "0") or 0)
                ctype = self.headers.get("Content-Type", "")
                raw_body = self.rfile.read(content_len) if content_len > 0 else b""
                try:
                    fb_dir = (
                        os.path.dirname(os.path.abspath(vm.replay_h5))
                        if vm.replay_h5 and os.path.isfile(vm.replay_h5)
                        else tempfile.gettempdir()
                    )
                    saved_paths: list[str] = []
                    header_bytes = f"Content-Type: {ctype}\r\n\r\n".encode("utf-8")
                    msg = email.parser.BytesParser().parsebytes(header_bytes + raw_body)
                    for part in msg.walk():
                        fname = part.get_filename()
                        if not fname:
                            continue
                        safe_name = os.path.basename(fname)
                        payload_bytes = part.get_payload(decode=True)
                        if payload_bytes is None:
                            continue
                        dest_path = os.path.join(fb_dir, safe_name)
                        with open(dest_path, "wb") as f_out:
                            f_out.write(payload_bytes)
                        saved_paths.append(dest_path)
                    if len(saved_paths) == 1 and not saved_paths[0].lower().endswith(".nxs.h5"):
                        sp = saved_paths[0]
                        if "merged" in os.path.basename(sp).lower():
                            vm.payload_kwargs["images_h5"] = sp
                        vm.replay_h5 = sp
                        vm.reload()
                    result = {"status": "ok", "saved_paths": saved_paths}
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
