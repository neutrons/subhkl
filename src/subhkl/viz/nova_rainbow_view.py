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
    ) -> str:
        """Link an existing 'ok' dataset in the Galaxy history or upload with file_type='h5', to_posix_lines=False."""
        fname = os.path.basename(local_path)
        gi = store.nova_connection.galaxy_instance
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

        is_local_gx = any(
            h in (getattr(gi, "base_url", "") or self.galaxy_url)
            for h in ("127.0.0.1", "localhost")
        )
        job_record["stdout"] += (
            f"[nova -> galaxy] Uploading/linking '{fname}' (file_type=h5, link_to_files={is_local_gx})...\n"
        )
        up_info = gi.tools.upload_file(
            path=local_path,
            history_id=store.history_id,
            file_name=fname,
            file_type="h5",
            to_posix_lines=False,
            link_to_files=is_local_gx,
        )
        ds_id = up_info["outputs"][0]["id"]
        gi.datasets.wait_for_dataset(ds_id, maxwait=180, interval=2.0, check=True)
        job_record["stdout"] += f"[nova -> galaxy] Upload of '{fname}' ready (id={ds_id})\n"
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

            raw_cell = params.get("cell") or [
                params.get("a", 68.7),
                params.get("b", 69.8),
                params.get("c", 156.9),
                params.get("alpha", 90.0),
                params.get("beta", 90.0),
                params.get("gamma", 90.0),
            ]
            wl = params.get("wavelength_range") or [
                params.get("wavelength_min", 1.8),
                params.get("wavelength_max", 4.5),
            ]
            inst = str(params.get("instrument", "CG4D"))
            sg = str(params.get("space_group", "P 21 21 21"))
            merged_path = params.get("images_file") or "/home/xvg/subhkl-examples/betalac-7/merged.h5"
            solve_path = params.get("peaks_file") or "/home/xvg/subhkl-examples/betalac-7/solve.h5"

            tool_inputs: dict[str, Any] = {}
            if stage_id == "solve":
                if not self._has_tool_on_galaxy("imaginex_subhkl_solve"):
                    # Check if local Planemo Galaxy (http://127.0.0.1:9090) has imaginex_subhkl_solve deployed
                    from nova.galaxy import Connection

                    job_record["stdout"] += (
                        f"[nova -> galaxy] 'imaginex_subhkl_solve' not yet merged on {self.galaxy_url}; "
                        f"switching to local Galaxy instance http://127.0.0.1:9090 (GPU container runner)...\n"
                    )
                    local_conn = Connection(galaxy_url="http://127.0.0.1:9090", galaxy_key="test_key")
                    store = local_conn.connect().create_data_store("IMAGINE-X SubHKL")
                    gi = store.nova_connection.galaxy_instance
                    self.galaxy_url = "http://127.0.0.1:9090"
                    self.api_key = gi.key
                    self.history_id = store.history_id

                if self._has_tool_on_galaxy("imaginex_subhkl_solve"):
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
                else:
                    raise RuntimeError(
                        f"'imaginex_subhkl_solve' is not yet deployed on {self.galaxy_url} "
                        f"(awaiting MR merge on ndip/galaxy-tools)"
                    )
            elif stage_id == "integrate":
                primary_tool = "imaginex_subhkl_predict"
                scan_id = self._ensure_galaxy_h5_dataset(store, merged_path, job_record)
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                tool_inputs = {
                    "scan": {"src": "hda", "id": scan_id},
                    "indexer": {"src": "hda", "id": idx_id},
                    "instrument": inst,
                    "dMin": float(params.get("d_min", 2.1)),
                    "wavelengthMin": float(wl[0]),
                    "wavelengthMax": float(wl[1]),
                    "space": sg,
                }
            elif stage_id == "reduce":
                primary_tool = "imaginex_subhkl_prepare_peaks"
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                tool_inputs = {
                    "inputs_0|input": {"src": "hda", "id": idx_id},
                    "instrument": inst,
                    "a": float(raw_cell[0]),
                    "b": float(raw_cell[1]),
                    "c": float(raw_cell[2]),
                    "alpha": float(raw_cell[3]),
                    "beta": float(raw_cell[4]),
                    "gamma": float(raw_cell[5]),
                    "spaceGroup": sg,
                }
            else:
                primary_tool = "imaginex_subhkl_metrics"
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                tool_inputs = {
                    "peaks": {"src": "hda", "id": idx_id},
                }

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
            job_record["stdout"] += (
                f"[galaxy -> pulsar] Dispatched {primary_tool} (Galaxy job_id={job_id}) -> Pulsar queue on IMAGINEX-1\n"
            )
            self._poll_galaxy_job(job_record, on_complete_h5)
        except Exception as exc:
            job_record["stdout"] += f"[nova -> galaxy] {exc}; running GPU container on IMAGINEX-1...\n"
            self._submit_local_subhkl_stage(
                stage_id,
                params,
                on_complete_h5,
                fallback_reason=str(exc),
                existing_job_record=job_record,
            )

    def _submit_local_subhkl_stage(
        self,
        stage_id: str,
        params: dict[str, Any],
        on_complete_h5: Any = None,
        fallback_reason: str | None = None,
        existing_job_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run subhkl stage inside the GPU Docker/Podman container on IMAGINEX-1 and stream live stdout/stderr."""
        import shutil
        import socket
        import subprocess
        import sys

        raw_cell = params.get("cell") or [
            params.get("a", 68.7),
            params.get("b", 69.8),
            params.get("c", 156.9),
            params.get("alpha", 90.0),
            params.get("beta", 90.0),
            params.get("gamma", 90.0),
        ]
        cell_str = ",".join(str(float(x)) for x in raw_cell[:6])
        wl = params.get("wavelength_range") or [
            params.get("wavelength_min", 1.8),
            params.get("wavelength_max", 4.5),
        ]
        inst = params.get("instrument", "CG4D")
        sg = params.get("space_group", "P 21 21 21")
        merged_path = os.path.abspath(
            params.get("images_file") or "/home/xvg/subhkl-examples/betalac-7/merged.h5"
        )
        solve_path = os.path.abspath(
            params.get("peaks_file") or "/home/xvg/subhkl-examples/betalac-7/solve.h5"
        )
        out_h5 = os.path.join(tempfile.gettempdir(), f"subhkl_{stage_id}_{int(time.time())}.h5")

        if stage_id == "reduce" and os.path.isfile(merged_path):
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
            inner_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "peak-predictor",
                merged_path,
                str(inst),
                solve_path,
                out_h5,
                "--d-min",
                str(params.get("d_min", 2.2)),
                "--wavel-min",
                str(wl[0]),
                "--wavel-max",
                str(wl[1]),
                "--space-group",
                str(sg),
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
            mount_dirs = sorted(
                {os.path.dirname(merged_path), os.path.dirname(solve_path), tempfile.gettempdir()}
            )
            cmd = [
                "podman",
                "run",
                "--rm",
                "--security-opt",
                "label=disable",
                "--device",
                "nvidia.com/gpu=all",
                "-e",
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
            ]
            for d in mount_dirs:
                if os.path.isdir(d):
                    cmd.extend(["-v", f"{d}:{d}"])
            cmd.append(container_image)
            cmd.extend(inner_cmd)
        else:
            cmd = [sys.executable] + inner_cmd[1:]

        gpu_env = os.environ.copy()
        gpu_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
        if fallback_reason:
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
            job_record["stdout"] = (job_record.get("stdout", "") + "\n" + "\n".join(header_lines))[-24000:]
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
                    job_record["state"] = "ok"
                    job_record["ladder_rung_idx"] = 4
                    job_record["stdout"] += f"\n[subhkl] Stage '{stage_id}' completed (exit 0).\n"
                    if os.path.isfile(out_h5) and callable(on_complete_h5):
                        on_complete_h5(out_h5)
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
    ) -> None:
        job_id = job_record.get("job_id")
        if not job_id:
            return
        step_counter = 0
        base_log = job_record.get("stdout", "")
        seen_pulsar_dirs: set[str] = set()
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
                state = jinfo.get("state", "running")
                job_record["state"] = state
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
                        job_record["ladder_rung_idx"] = 4
                        job_record["stdout"] += (
                            f"\n[galaxy -> pulsar] Job {job_id} finished (state=ok).\n"
                        )
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
