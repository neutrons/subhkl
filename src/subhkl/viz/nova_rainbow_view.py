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
import shutil
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
        "merge": ["imaginex_subhkl_merge"],
    }

    # Upstream NDIP ``neutrons_load_data`` tool hardcodes ``<select name="instrument">``
    # options when ``params|ingest_mode == "oncat"``, omitting instruments such as
    # ``CG4D`` / ``IMAGINE``. When an instrument is not in this allow-list, we pass
    # the ONCat-resolved path via ``params|ingest_mode == "file"`` (``params|filepath``).
    NEUTRONS_LOAD_DATA_ONCAT_INSTRUMENTS: dict[str, set[str]] = {
        "SNS": {
            "ARCS",
            "BL0",
            "CNCS",
            "CORELLI",
            "EQSANS",
            "FNPB",
            "HYSPEC",
            "MANDI",
            "NOMAD",
            "NSE",
            "PG3",
            "POWGEN",
            "REF_L",
            "REF_M",
            "SEQ",
            "SNAP",
            "TOPAZ",
            "USANS",
            "VENUS",
            "VISION",
            "VULCAN",
        },
        "HFIR": {
            "CG1B",
            "CG1D",
            "CG2",
            "CG3",
            "CG4C",
            "HB1",
            "HB1A",
            "HB2A",
            "HB2B",
            "HB2C",
            "HB3",
            "HB3A",
        },
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
        self.current_frame_index: int | None = None
        self._auto_discover_local_galaxy()

    def _build_neutrons_load_data_inputs(
        self,
        *,
        ingest_mode: str,
        fac: str,
        inst: str,
        ipts_raw: str,
        run_num: int | None,
        filepath: str,
    ) -> dict[str, Any]:
        """Build ``neutrons_load_data`` inputs, falling back to ``file`` mode for instruments
        (such as HFIR ``CG4D``) omitted from upstream's hardcoded ``oncat`` dropdown."""
        fac_u = str(
            fac or ("SNS" if str(filepath).startswith("/SNS/") else "HFIR")
        ).upper()
        inst_u = str(inst or "").upper()
        allowed_oncat = self.NEUTRONS_LOAD_DATA_ONCAT_INSTRUMENTS.get(fac_u, set())
        if (
            str(ingest_mode or "oncat") != "file"
            and ipts_raw
            and run_num is not None
            and inst_u in allowed_oncat
        ):
            return {
                "params|ingest_mode": "oncat",
                "params|facility": fac_u,
                "params|instrument": inst_u,
                "params|ipts": str(ipts_raw),
                "params|run": int(run_num),
            }
        return {
            "params|ingest_mode": "file",
            "params|filepath": str(filepath),
        }

    def _run_neutrons_load_data_tool(
        self,
        gi: Any,
        history_id: str,
        nld_inputs: dict[str, Any],
        filepath: str,
    ) -> dict[str, Any]:
        """Execute ``neutrons_load_data`` and automatically retry in ``file`` mode if
        upstream Galaxy rejects an ``oncat`` parameter option (HTTP 400)."""
        try:
            return gi.tools.run_tool(
                history_id=history_id,
                tool_id="neutrons_load_data",
                tool_inputs=nld_inputs,
            )
        except Exception:
            if nld_inputs.get("params|ingest_mode") == "oncat" and filepath:
                fallback_inputs = {
                    "params|ingest_mode": "file",
                    "params|filepath": str(filepath),
                }
                nld_inputs.clear()
                nld_inputs.update(fallback_inputs)
                return gi.tools.run_tool(
                    history_id=history_id,
                    tool_id="neutrons_load_data",
                    tool_inputs=nld_inputs,
                )
            raise

    def _lookup_oncat_run_title(self, nxs_path: str) -> str:
        """Look up a NeXus run's ONCat title and goniometer angles without reading the host filesystem."""
        import re as _re

        p_str = str(nxs_path or "")
        m_loc = _re.search(r"/(SNS|HFIR)/([^/]+)/IPTS-([^/]+)/", p_str, _re.I)
        m_run = _re.search(r"_([0-9]+)\.nxs", os.path.basename(p_str), _re.I)
        if not m_loc:
            return ""
        fac_u = m_loc.group(1).upper()
        inst_u = m_loc.group(2).upper()
        ipts_num = m_loc.group(3)
        run_num = int(m_run.group(1)) if m_run else None
        try:
            cat = self.browse_oncat_catalog(
                {
                    "facility": fac_u,
                    "instrument": inst_u,
                    "ipts": f"IPTS-{ipts_num}",
                }
            )
            for r in cat.get("runs") or []:
                if str(r.get("location") or "") == p_str or (
                    run_num is not None and r.get("run_number") == run_num
                ):
                    t_part = str(r.get("title") or "").strip()
                    g_part = str(r.get("angles") or "").strip()
                    return f"{t_part} {g_part}".strip()
        except Exception:
            pass
        return ""

    def apply_active_solve_overlay(
        self,
        payload: dict[str, Any],
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        """If a solve job is currently active and has produced an intermediate blind solution,
        overlay its zones and ladder steps projected for ``frame_index`` onto ``payload``."""
        if frame_index is not None:
            self.current_frame_index = int(frame_index)
        aj = self.active_job
        if not isinstance(aj, dict):
            return payload
        if aj.get("stage_id") != "solve" or aj.get("state") not in (
            "queued",
            "running",
        ):
            return payload
        partial_h5 = aj.get("partial_h5_path")
        images_h5 = aj.get("images_h5_path")
        sp = aj.get("solve_params") or {}
        if (
            partial_h5
            and os.path.isfile(partial_h5)
            and os.path.getsize(partial_h5) > 0
        ):
            try:
                from subhkl.viz.rainbow_zones import build_rainbow_zone_payload

                p_blind = build_rainbow_zone_payload(
                    replay_h5=partial_h5,
                    images_h5=(
                        images_h5 if (images_h5 and os.path.isfile(images_h5)) else None
                    ),
                    instrument=str(
                        sp.get("instrument") or payload.get("instrument") or "CG4D"
                    ),
                    space_group=str(
                        sp.get("space_group") or payload.get("space_group") or "P 1"
                    ),
                    cell=list(
                        sp.get("cell")
                        or payload.get("cell")
                        or [10, 10, 10, 90, 90, 90]
                    ),
                    wavelength_range=tuple(
                        sp.get("wavelength_range")
                        or payload.get("wavelength_range")
                        or (2.0, 4.5)
                    ),
                    max_zone_index=1,
                    max_zones=13,
                    include_panels=False,
                    include_peaks=False,
                    n_arc=720,
                    frame_index=(
                        frame_index
                        if frame_index is not None
                        else self.current_frame_index
                    ),
                )
                if p_blind.get("zones"):
                    payload["zones"] = p_blind["zones"]
                    if p_blind.get("ladder_steps"):
                        payload["ladder_steps"] = p_blind["ladder_steps"]
                    payload["active_rung_idx"] = int(aj.get("ladder_rung_idx") or 1)
                    return payload
            except Exception:
                pass
        ov = aj.get("partial_overlay")
        if isinstance(ov, dict) and ov.get("zones") and not payload.get("zones"):
            payload["zones"] = ov["zones"]
            if ov.get("ladder_steps"):
                payload["ladder_steps"] = ov["ladder_steps"]
            payload["active_rung_idx"] = int(aj.get("ladder_rung_idx") or 1)
        return payload

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

    @staticmethod
    def _format_terminal_stream(raw_text: str) -> str:
        """Emulate terminal carriage-return (\\r) and ANSI line-erase/cursor-up for tqdm progress bars."""
        if not raw_text:
            return ""
        import re as _re

        text = raw_text.replace("\r\n", "\n")
        tokens = _re.split(r"(\x1b\[[0-9;?]*[A-Za-z]|\r|\n)", text)
        lines: list[str] = [""]
        row = 0
        col = 0
        for tok in tokens:
            if not tok:
                continue
            if tok == "\n":
                lines[row] = lines[row].rstrip()
                row += 1
                if row >= len(lines):
                    lines.append("")
                col = 0
            elif tok == "\r":
                col = 0
            elif tok.startswith("\x1b["):
                cmd_ch = tok[-1]
                params_s = tok[2:-1].lstrip("?;")
                first_num = (
                    int(params_s.split(";")[0])
                    if params_s and params_s.split(";")[0].isdigit()
                    else 1
                )
                if cmd_ch == "A":
                    row = max(0, row - max(1, first_num))
                    col = 0
                elif cmd_ch == "B":
                    row = min(len(lines) - 1, row + max(1, first_num))
                    col = 0
                elif cmd_ch == "G":
                    col = max(0, first_num - 1)
                elif cmd_ch == "K":
                    mode_k = int(params_s) if params_s and params_s.isdigit() else 0
                    if mode_k == 2:
                        lines[row] = ""
                        col = 0
                    elif mode_k == 1:
                        lines[row] = (" " * col) + lines[row][col:]
                    else:
                        lines[row] = lines[row][:col]
            else:
                cur = lines[row]
                if col > len(cur):
                    cur = cur + (" " * (col - len(cur)))
                lines[row] = cur[:col] + tok + cur[col + len(tok) :]
                col += len(tok)

        tqdm_re = _re.compile(r"^\s*(?P<pfx>[^%|]*?)\s*\d{1,3}%\|.*\|")
        collapsed: list[str] = []
        for ln in lines:
            clean_ln = ln.rstrip()
            m_cur = tqdm_re.match(clean_ln)
            if m_cur and collapsed:
                m_prev = tqdm_re.match(collapsed[-1])
                if m_prev and m_prev.group("pfx").strip() == m_cur.group("pfx").strip():
                    collapsed[-1] = clean_ln
                    continue
            collapsed.append(clean_ln)
        return "\n".join(collapsed)

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
        """Resolve a Galaxy dataset ID to a local HDF5 path (via local file_name, cached ds_id, or /display download)."""
        import glob as _glob

        info = self._api_request("GET", f"/api/datasets/{ds_id}")
        if not isinstance(info, dict):
            raise RuntimeError(f"Dataset {ds_id} not found on {self.galaxy_url}")
        name = str(info.get("name") or f"{ds_id}.h5")
        file_name = str(info.get("file_name") or "")
        if file_name and os.path.isfile(file_name) and os.path.getsize(file_name) > 0:
            return file_name, info

        # 1. Check cached dataset file in /tmp by exact ds_id first
        cached_matches = sorted(
            _glob.glob(os.path.join(tempfile.gettempdir(), f"subhkl_gx_{ds_id}_*"))
        )
        for cm in cached_matches:
            if os.path.isfile(cm) and os.path.getsize(cm) > 0:
                return cm, info

        # 2. If dataset name includes a relative folder (e.g. 'betalac-7/merged.h5'), check /home/xvg/subhkl-examples/
        if "/" in name:
            if os.path.isfile(name) and os.path.getsize(name) > 0:
                return name, info
            ex_cand = os.path.join("/home/xvg/subhkl-examples", name.lstrip("/"))
            if os.path.isfile(ex_cand) and os.path.getsize(ex_cand) > 0:
                return ex_cand, info

        # 3. Only fall back to fallback_dir if file_size matches (prevents generic 'scan.h5'/'merged.h5' collisions)
        expected_size = int(info.get("file_size") or 0)
        if fallback_dir and os.path.isdir(fallback_dir):
            cand = os.path.join(fallback_dir, os.path.basename(name))
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                if (expected_size > 0 and os.path.getsize(cand) == expected_size) or (
                    expected_size == 0 and not self.is_connected
                ):
                    return cand, info

        safe_name = "".join(
            ch if (ch.isalnum() or ch in "._-") else "_"
            for ch in os.path.basename(name)
        )
        if not safe_name.endswith((".h5", ".hdf5", ".mtz", ".json", ".nxs")):
            safe_name += ".h5"
        tmp_dest = os.path.join(tempfile.gettempdir(), f"subhkl_gx_{ds_id}_{safe_name}")
        if os.path.isfile(tmp_dest) and os.path.getsize(tmp_dest) > 0:
            return tmp_dest, info
        data = self._api_request(
            "GET",
            f"/api/datasets/{ds_id}/display",
            raw=True,
            timeout=180.0,
        )
        with open(tmp_dest, "wb") as fh:
            fh.write(data)
        return tmp_dest, info

    @staticmethod
    def _classify_artifact_stage(filename: str) -> str | None:
        """Map an artifact filename to one of ('reduce', 'solve', 'integrate', 'merge')."""
        low = os.path.basename(filename).lower()
        if not low or low.startswith("."):
            return None
        if any(
            low == k or low.endswith(k)
            for k in (
                "merged.mtz",
                "predictions.mtz",
                "xval.mtz",
                "careless_metrics.json",
                "careless_0.mtz",
                "careless_predictions_0.mtz",
                "careless_xval_0.mtz",
            )
        ):
            return "merge"
        if any(
            low == k or low.endswith(k)
            for k in ("predictor.h5", "integrated.h5", "reflections.mtz")
        ):
            return "integrate"
        if "solve" in low and low.endswith(".h5"):
            return "solve"
        if any(
            k in low
            for k in (
                "merged.h5",
                "_merged.h5",
                ".reduce.h5",
                ".nxs.h5",
                "scan_master.h5",
            )
        ):
            return "reduce"
        return None

    def _active_work_dir(self, current_h5: str | None = None) -> str | None:
        cand = current_h5
        if not cand and isinstance(self.active_job, dict):
            cand = self.active_job.get("output_h5_path")
        if cand and os.path.isfile(str(cand)):
            wd, _ = self._resolve_work_dir_and_merged(str(cand))
            if os.path.isdir(wd):
                return wd
            parent = os.path.dirname(os.path.abspath(str(cand)))
            if os.path.isdir(parent):
                return parent
        return None

    def list_step_artifacts(
        self,
        current_h5: str | None = None,
        history_datasets: list[dict[str, Any]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return downloadable artifacts grouped by pipeline stage ('reduce', 'solve', 'integrate', 'merge')."""
        stages: dict[str, dict[str, dict[str, Any]]] = {
            "reduce": {},
            "solve": {},
            "integrate": {},
            "merge": {},
        }
        if history_datasets is None:
            history_datasets = self.list_history_datasets()

        for ds in history_datasets or []:
            if ds.get("state") and ds.get("state") != "ok":
                continue
            raw_name = str(ds.get("name") or "")
            base_name = os.path.basename(raw_name)
            st = self._classify_artifact_stage(base_name)
            if not st:
                continue
            stages[st][base_name] = {
                "name": base_name,
                "stage_id": st,
                "dataset_id": str(ds.get("id") or ""),
                "hid": ds.get("hid"),
                "source": "NDIP",
            }

        work_dir = self._active_work_dir(current_h5)
        if work_dir and os.path.isdir(work_dir):
            try:
                for fn in sorted(os.listdir(work_dir)):
                    fp = os.path.join(work_dir, fn)
                    if not os.path.isfile(fp) or os.path.getsize(fp) <= 0:
                        continue
                    st = self._classify_artifact_stage(fn)
                    if not st:
                        continue
                    entry = stages[st].get(
                        fn, {"name": fn, "stage_id": st, "source": "LOCAL"}
                    )
                    entry["size_bytes"] = int(os.path.getsize(fp))
                    entry["has_local"] = True
                    stages[st][fn] = entry
                    if fn == "integrated.h5" and os.path.isfile(
                        os.path.join(work_dir, "careless_metrics.json")
                    ):
                        m_entry = dict(entry)
                        m_entry["stage_id"] = "merge"
                        stages["merge"][fn] = m_entry
            except Exception:
                pass

        return {k: list(v.values()) for k, v in stages.items()}

    def resolve_artifact_for_download(
        self,
        *,
        dataset_id: str = "",
        name: str = "",
        current_h5: str | None = None,
    ) -> tuple[str, str]:
        """Resolve an artifact by dataset_id or filename to `(local_file_path, download_filename)`."""
        clean_name = os.path.basename(name.strip()) if name else ""
        work_dir = self._active_work_dir(current_h5)
        if clean_name and work_dir and os.path.isdir(work_dir):
            cand = os.path.join(work_dir, clean_name)
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand, clean_name

        ds_id = dataset_id.strip()
        if not ds_id and clean_name:
            for ds in reversed(self.list_history_datasets()):
                if (
                    ds.get("state") == "ok"
                    and os.path.basename(str(ds.get("name") or "")) == clean_name
                ):
                    ds_id = str(ds.get("id") or "")
                    break

        if ds_id:
            local_p, info = self.resolve_dataset_to_local_h5(
                ds_id, fallback_dir=work_dir
            )
            dl_name = clean_name or os.path.basename(
                str(info.get("name") or os.path.basename(local_p))
            )
            return local_p, dl_name

        raise FileNotFoundError(
            f"Artifact '{clean_name or dataset_id}' not found in work_dir or NDIP Galaxy history."
        )

    def _work_dir_for_reduce(
        self, raw_nxs: list[str], fallback_dir: str
    ) -> tuple[str, str]:
        """Return an isolated per-dataset work directory and descriptive merged filename for raw NeXus reduction."""
        if not raw_nxs:
            return fallback_dir, "merged.h5"
        first_stem = os.path.basename(raw_nxs[0]).split(".")[0]
        last_stem = os.path.basename(raw_nxs[-1]).split(".")[0]
        tag = first_stem if len(raw_nxs) == 1 else f"{first_stem}_{last_stem}"
        safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
        work_d = os.path.join(tempfile.gettempdir(), f"subhkl_work_{safe_tag}")
        os.makedirs(work_d, exist_ok=True)
        return work_d, f"{safe_tag}_merged.h5"

    def _resolve_work_dir_and_merged(self, merged_path: str) -> tuple[str, str]:
        """Ensure a merged HDF5 file inside /tmp (such as /tmp/subhkl_gx_<id>_...) uses a dedicated /tmp/subhkl_work_<tag> directory."""
        import re as _re
        import shutil as _shutil

        abs_m = os.path.abspath(merged_path)
        parent_d = os.path.dirname(abs_m)
        tmp_root = os.path.abspath(tempfile.gettempdir())
        if parent_d != tmp_root:
            return parent_d, abs_m

        tag = ""
        if os.path.isfile(abs_m):
            try:
                import h5py

                with h5py.File(abs_m, "r") as hf:
                    if "files" in hf:
                        raw_f = [
                            (
                                x.decode("utf-8", "ignore")
                                if isinstance(x, bytes)
                                else str(x)
                            )
                            for x in list(hf["files"][()])
                        ]
                        raw_f = [x for x in raw_f if x]
                        if raw_f:
                            s0 = os.path.basename(raw_f[0]).split(".")[0]
                            s1 = os.path.basename(raw_f[-1]).split(".")[0]
                            tag = s0 if s0 == s1 else f"{s0}_{s1}"
            except Exception:
                pass
        if not tag:
            bn = os.path.basename(abs_m)
            bn = _re.sub(r"^subhkl_gx_[0-9a-fA-F]+_", "", bn)
            bn = _re.sub(r"(_merged)?\.h5$", "", bn, flags=_re.I)
            tag = bn or "dataset"
        safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
        work_d = os.path.join(tmp_root, f"subhkl_work_{safe_tag}")
        os.makedirs(work_d, exist_ok=True)
        dest_m = os.path.join(work_d, "merged.h5")
        if os.path.isfile(abs_m) and not os.path.isfile(dest_m):
            try:
                _shutil.copy2(abs_m, dest_m)
            except Exception:
                return work_d, abs_m
        return work_d, (dest_m if os.path.isfile(dest_m) else abs_m)

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

    def _get_tool_version(self, tool_id: str) -> str:
        try:
            doc = self._api_request("GET", f"/api/tools/{tool_id}")
            if isinstance(doc, dict):
                return str(doc.get("version") or "")
        except Exception:
            pass
        return ""

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

    def _stamp_merged_metadata(self, h5_path: str, job_record: dict[str, Any]) -> None:
        if not h5_path or not os.path.isfile(h5_path):
            return
        try:
            import h5py
            import numpy as np

            params = job_record.get("params") or {}
            inst_val = str(
                job_record.get("instrument") or params.get("instrument") or ""
            ).strip()
            sg_val = str(params.get("space_group") or "").strip()
            raw_cell = params.get("cell")
            if not (isinstance(raw_cell, (list, tuple)) and len(raw_cell) == 6):
                if all(k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")):
                    raw_cell = [
                        float(params["a"]),
                        float(params["b"]),
                        float(params["c"]),
                        float(params["alpha"]),
                        float(params["beta"]),
                        float(params["gamma"]),
                    ]
            with h5py.File(h5_path, "r+") as hf:
                if inst_val and "instrument" not in hf:
                    hf["instrument"] = np.bytes_(inst_val)
                if sg_val and "space_group" not in hf:
                    hf["space_group"] = np.bytes_(sg_val)
                raw_nxs_list = (
                    job_record.get("raw_nxs") or params.get("nexus_files") or []
                )
                if isinstance(raw_nxs_list, str):
                    raw_nxs_list = [
                        s.strip() for s in raw_nxs_list.split(",") if s.strip()
                    ]
                if (
                    raw_nxs_list
                    and "files" in hf
                    and len(hf["files"]) == len(raw_nxs_list)
                ):
                    orig_names = [
                        os.path.basename(str(p).strip()) for p in raw_nxs_list
                    ]
                    del hf["files"]
                    dt_str = h5py.string_dtype(encoding="utf-8")
                    hf.create_dataset("files", data=orig_names, dtype=dt_str)
                if (
                    isinstance(raw_cell, (list, tuple))
                    and len(raw_cell) == 6
                    and "sample/a" not in hf
                ):
                    grp = hf.require_group("sample")
                    for idx_c, k_c in enumerate(
                        ("a", "b", "c", "alpha", "beta", "gamma")
                    ):
                        if k_c not in grp:
                            grp[k_c] = float(raw_cell[idx_c])
        except Exception:
            pass

    def _stamp_careless_merge_into_h5(
        self,
        integrated_h5_path: str,
        work_dir: str,
        job_record: dict[str, Any],
    ) -> None:
        """Persist Careless merge metrics and per-observation residuals/scales into ``integrated.h5``."""
        import h5py
        import gemmi
        import numpy as np

        metrics_path = os.path.join(work_dir, "careless_metrics.json")
        pred_mtz_path = os.path.join(work_dir, "predictions.mtz")
        merged_mtz_path = os.path.join(work_dir, "merged.mtz")
        xval_mtz_path = os.path.join(work_dir, "xval.mtz")
        metrics_raw = "{}"
        metrics_doc: dict[str, Any] = {}
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as fh:
                metrics_raw = fh.read()
                try:
                    metrics_doc = json.loads(metrics_raw)
                except Exception:
                    metrics_doc = {}

        if os.path.isfile(merged_mtz_path):
            try:
                mtz_m = gemmi.read_mtz_file(merged_mtz_path)
                mcols = {c.label: np.array(c, dtype=np.float64) for c in mtz_m.columns}
                mI = mcols.get("I", mcols.get("I(+)", np.array([])))
                mS = mcols.get("SigI", mcols.get("SigI(+)", np.array([])))
                val_m = np.isfinite(mI) & np.isfinite(mS) & (mS > 0)
                if np.any(val_m):
                    hm_v = mcols["H"][val_m].astype(int)
                    km_v = mcols["K"][val_m].astype(int)
                    lm_v = mcols["L"][val_m].astype(int)
                    isigi_v = mI[val_m] / mS[val_m]
                    d_m = np.array(
                        [
                            mtz_m.cell.calculate_d([int(hh), int(kk), int(ll)])
                            for hh, kk, ll in zip(hm_v, km_v, lm_v)
                        ],
                        dtype=np.float64,
                    )
                    inv_d2_m = 1.0 / np.maximum(d_m**2, 1e-6)
                    edges = np.linspace(
                        float(inv_d2_m.min()), float(inv_d2_m.max()), 11
                    )
                    dmin_obs = float(np.min(d_m))
                    sg_m = mtz_m.spacegroup or gemmi.find_spacegroup_by_name("P 1")
                    asu_m = gemmi.ReciprocalAsu(sg_m)
                    ops_m = sg_m.operations()
                    hmax = int(np.ceil(mtz_m.cell.a / max(dmin_obs, 0.5))) + 1
                    kmax = int(np.ceil(mtz_m.cell.b / max(dmin_obs, 0.5))) + 1
                    lmax = int(np.ceil(mtz_m.cell.c / max(dmin_obs, 0.5))) + 1
                    theo_set: set[tuple[int, int, int]] = set()
                    for hh in range(-hmax, hmax + 1):
                        for kk in range(-kmax, kmax + 1):
                            for ll in range(-lmax, lmax + 1):
                                if hh == 0 and kk == 0 and ll == 0:
                                    continue
                                if ops_m.is_systematically_absent([hh, kk, ll]):
                                    continue
                                d_val = mtz_m.cell.calculate_d([hh, kk, ll])
                                if d_val >= dmin_obs - 1e-5:
                                    hasu, _ = asu_m.to_asu([hh, kk, ll], ops_m)
                                    theo_set.add(
                                        (int(hasu[0]), int(hasu[1]), int(hasu[2]))
                                    )
                    d_theo = (
                        np.array(
                            [
                                mtz_m.cell.calculate_d([hh, kk, ll])
                                for hh, kk, ll in theo_set
                            ],
                            dtype=np.float64,
                        )
                        if theo_set
                        else np.array([], dtype=np.float64)
                    )
                    inv_d2_t = (
                        1.0 / np.maximum(d_theo**2, 1e-6)
                        if len(d_theo)
                        else np.array([], dtype=np.float64)
                    )

                    xv_v0 = np.array([], dtype=np.float64)
                    xv_v1 = np.array([], dtype=np.float64)
                    inv_d2_x = np.array([], dtype=np.float64)
                    if os.path.isfile(xval_mtz_path):
                        xv = gemmi.read_mtz_file(xval_mtz_path)
                        xcols = {
                            c.label: np.array(c, dtype=np.float64) for c in xv.columns
                        }
                        xh = xcols["H"].astype(int)
                        xk = xcols["K"].astype(int)
                        xl = xcols["L"].astype(int)
                        xhalf = xcols.get("half", np.zeros_like(xh)).astype(int)
                        xI = xcols.get("I", xcols.get("F", np.zeros_like(xcols["H"])))
                        xd = np.array(
                            [
                                xv.cell.calculate_d([int(hh), int(kk), int(ll)])
                                for hh, kk, ll in zip(xh, xk, xl)
                            ],
                            dtype=np.float64,
                        )
                        m0 = xhalf == 0
                        m1 = xhalf == 1
                        map0 = {
                            (int(hh), int(kk), int(ll)): (float(val), float(dd))
                            for hh, kk, ll, val, dd in zip(
                                xh[m0], xk[m0], xl[m0], xI[m0], xd[m0]
                            )
                            if np.isfinite(val)
                        }
                        map1 = {
                            (int(hh), int(kk), int(ll)): float(val)
                            for hh, kk, ll, val in zip(xh[m1], xk[m1], xl[m1], xI[m1])
                            if np.isfinite(val)
                        }
                        common = [k for k in map0 if k in map1]
                        if common:
                            xv_v0 = np.array(
                                [map0[k][0] for k in common], dtype=np.float64
                            )
                            xv_v1 = np.array(
                                [map1[k] for k in common], dtype=np.float64
                            )
                            d_c = np.array(
                                [map0[k][1] for k in common], dtype=np.float64
                            )
                            inv_d2_x = 1.0 / np.maximum(d_c**2, 1e-6)

                    bins = []
                    for b in range(10):
                        hi_edge = edges[b + 1]
                        lo_edge = edges[b]
                        bm_m = (inv_d2_m >= lo_edge) & (
                            inv_d2_m <= hi_edge if b == 9 else inv_d2_m < hi_edge
                        )
                        bm_t = (
                            (inv_d2_t >= lo_edge)
                            & (inv_d2_t <= hi_edge if b == 9 else inv_d2_t < hi_edge)
                            if len(inv_d2_t)
                            else np.array([], dtype=bool)
                        )
                        n_m_b = int(np.sum(bm_m))
                        n_t_b = max(n_m_b, int(np.sum(bm_t))) if len(bm_t) else n_m_b
                        comp_b = (
                            round(float(100.0 * n_m_b / max(1, n_t_b)), 2)
                            if n_t_b > 0
                            else 0.0
                        )
                        isigi_b = (
                            round(float(np.mean(isigi_v[bm_m])), 2)
                            if n_m_b > 0
                            else 0.0
                        )
                        cc_b: float | None = None
                        if len(inv_d2_x):
                            bm_x = (inv_d2_x >= lo_edge) & (
                                inv_d2_x <= hi_edge if b == 9 else inv_d2_x < hi_edge
                            )
                            if int(np.sum(bm_x)) >= 2:
                                s0 = xv_v0[bm_x]
                                s1 = xv_v1[bm_x]
                                if np.std(s0) > 1e-12 and np.std(s1) > 1e-12:
                                    r_val = float(np.corrcoef(s0, s1)[0, 1])
                                    if np.isfinite(r_val):
                                        cc_b = round(r_val, 4)
                        d_hi = float(1.0 / np.sqrt(max(hi_edge, 1e-6)))
                        d_lo = float(1.0 / np.sqrt(max(lo_edge, 1e-6)))
                        bins.append(
                            {
                                "bin": b,
                                "d_max": round(d_lo, 2),
                                "d_min": round(d_hi, 2),
                                "n_refl": n_m_b,
                                "n_theo": n_t_b,
                                "completeness_pct": comp_b,
                                "mean_isigi": isigi_b,
                                "cc_half": cc_b,
                            }
                        )
                    metrics_doc["resolution_bins"] = bins
                    metrics_raw = json.dumps(metrics_doc, indent=2)
                    with open(metrics_path, "w", encoding="utf-8") as fh_out:
                        fh_out.write(metrics_raw)
            except Exception:
                pass

        with h5py.File(integrated_h5_path, "r+") as hf:
            cgrp = hf.require_group("careless")
            if "metrics_json" in cgrp:
                del cgrp["metrics_json"]
            cgrp.create_dataset(
                "metrics_json",
                data=np.bytes_(metrics_raw.encode("utf-8")),
            )

            if "peaks" in hf and os.path.isfile(pred_mtz_path):
                pgrp = hf["peaks"]
                ph = np.asarray(pgrp["h"][()], dtype=int)
                pk = np.asarray(pgrp["k"][()], dtype=int)
                pl = np.asarray(pgrp["l"][()], dtype=int)
                pI = np.asarray(pgrp["intensity"][()], dtype=float)
                pS = np.asarray(pgrp["sigma"][()], dtype=float)
                n_peaks = len(ph)

                mtz_pred = gemmi.read_mtz_file(pred_mtz_path)
                sg = mtz_pred.spacegroup or gemmi.find_spacegroup_by_name("P 1")
                asu = gemmi.ReciprocalAsu(sg)
                ops = sg.operations()
                pcols = {
                    c.label: np.array(c, dtype=np.float64) for c in mtz_pred.columns
                }
                pred_map: dict[
                    tuple[int, int, int],
                    list[tuple[float, float, float, float, float]],
                ] = {}
                for j in range(mtz_pred.nreflections):
                    key = (
                        int(pcols["H"][j]),
                        int(pcols["K"][j]),
                        int(pcols["L"][j]),
                    )
                    pred_map.setdefault(key, []).append(
                        (
                            float(
                                pcols.get("Iobs", np.zeros(mtz_pred.nreflections))[j]
                            ),
                            float(
                                pcols.get("SigIobs", np.ones(mtz_pred.nreflections))[j]
                            ),
                            float(
                                pcols.get("Ipred", np.zeros(mtz_pred.nreflections))[j]
                            ),
                            float(
                                pcols.get("SigIpred", np.ones(mtz_pred.nreflections))[j]
                            ),
                            float(
                                pcols.get("Scale", np.ones(mtz_pred.nreflections))[j]
                            ),
                        )
                    )

                merged_map: dict[tuple[int, int, int], tuple[float, float]] = {}
                if os.path.isfile(merged_mtz_path):
                    mtz_m = gemmi.read_mtz_file(merged_mtz_path)
                    mcols = {
                        c.label: np.array(c, dtype=np.float64) for c in mtz_m.columns
                    }
                    mI = mcols.get("I", mcols.get("I(+)", np.zeros(mtz_m.nreflections)))
                    mS = mcols.get(
                        "SigI", mcols.get("SigI(+)", np.ones(mtz_m.nreflections))
                    )
                    for j in range(mtz_m.nreflections):
                        mkey = (
                            int(mcols["H"][j]),
                            int(mcols["K"][j]),
                            int(mcols["L"][j]),
                        )
                        merged_map[mkey] = (float(mI[j]), float(mS[j]))

                residuals = np.full(n_peaks, np.nan, dtype=np.float64)
                scales = np.full(n_peaks, np.nan, dtype=np.float64)
                ipreds = np.full(n_peaks, np.nan, dtype=np.float64)
                merged_i = np.full(n_peaks, np.nan, dtype=np.float64)
                merged_sigi = np.full(n_peaks, np.nan, dtype=np.float64)
                asu_hkl_arr = np.zeros((n_peaks, 3), dtype=np.int32)

                for i in range(n_peaks):
                    h_asu, _ = asu.to_asu([int(ph[i]), int(pk[i]), int(pl[i])], ops)
                    key = (int(h_asu[0]), int(h_asu[1]), int(h_asu[2]))
                    asu_hkl_arr[i] = key
                    if key in merged_map:
                        merged_i[i], merged_sigi[i] = merged_map[key]
                    cands = pred_map.get(key)
                    if cands:
                        best = min(
                            cands,
                            key=lambda c: abs(c[0] - pI[i]) + abs(c[1] - pS[i]),
                        )
                        iobs, sigiobs, ipred, sigipred, scale = best
                        denom = max(float(np.hypot(sigiobs, 0.5 * sigipred)), 1e-6)
                        residuals[i] = (iobs - ipred) / denom
                        scales[i] = scale
                        ipreds[i] = ipred

                for ds_name, arr in (
                    ("careless_residual", residuals),
                    ("careless_scale", scales),
                    ("careless_ipred", ipreds),
                    ("merged_i", merged_i),
                    ("merged_sigi", merged_sigi),
                    ("asu_hkl", asu_hkl_arr),
                ):
                    if ds_name in pgrp:
                        del pgrp[ds_name]
                    pgrp.create_dataset(ds_name, data=arr)

        if metrics_doc:
            cc_val = metrics_doc.get("cc_half")
            cc_str = f"{cc_val:.4f}" if isinstance(cc_val, (int, float)) else "N/A"
            comp_val = metrics_doc.get("completeness_pct")
            comp_str = (
                f"{comp_val:.1f}%" if isinstance(comp_val, (int, float)) else "N/A"
            )
            job_record["stdout"] += (
                f"\n[careless] Merged {metrics_doc.get('n_merged_asu', 0)} ASU reflections "
                f"from {metrics_doc.get('n_observations', 0)} observations | "
                f"CC1/2={cc_str} | <I/sigma>={metrics_doc.get('mean_isigi', 0.0):.2f} | "
                f"Multiplicity={metrics_doc.get('mean_multiplicity', 0.0):.2f}x | "
                f"Completeness={comp_str}\n"
            )

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
                    job_record["stdout"] += (
                        f"[nova -> galaxy] Linked existing dataset '{fname}' (id={ds_id}, state=ok)\n"
                    )
                    return ds_id
                if st in ("queued", "running", "new", "upload"):
                    job_record["stdout"] += (
                        f"[nova -> galaxy] Waiting for in-progress upload of '{fname}' (id={ds_id}, state={st})...\n"
                    )
                    gi.datasets.wait_for_dataset(
                        ds_id, maxwait=120, interval=2.0, check=True
                    )
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
        try:
            safe_name = "".join(
                ch if (ch.isalnum() or ch in "._-") else "_"
                for ch in os.path.basename(fname)
            )
            cache_dest = os.path.join(
                tempfile.gettempdir(), f"subhkl_gx_{ds_id}_{safe_name}"
            )
            if os.path.isfile(local_path) and not os.path.exists(cache_dest):
                try:
                    os.link(local_path, cache_dest)
                except Exception:
                    shutil.copy2(local_path, cache_dest)
        except Exception:
            pass
        job_record["stdout"] += (
            f"[nova -> galaxy] Saved '{fname}' in Galaxy history (id={ds_id})\n"
        )
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
            merged_path = (
                params.get("images_file")
                or "/home/xvg/subhkl-examples/betalac-7/merged.h5"
            )
            work_dir, merged_path = self._resolve_work_dir_and_merged(merged_path)
            solve_path = params.get("peaks_file") or os.path.join(work_dir, "solve.h5")
            if not os.path.isfile(solve_path) or os.path.basename(solve_path) in (
                "merged.h5",
                "integrated.h5",
            ):
                cand_solve = os.path.join(work_dir, "solve.h5")
                if os.path.isfile(cand_solve):
                    solve_path = cand_solve

            tool_inputs: dict[str, Any] = {}
            if stage_id in ("reduce", "solve", "integrate", "merge"):
                required_tool = (
                    "imaginex_subhkl_prepare_peaks"
                    if stage_id == "reduce"
                    else (
                        "imaginex_subhkl_merge"
                        if stage_id == "merge"
                        else "imaginex_subhkl_solve"
                    )
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
                            local_conn = Connection(
                                galaxy_url="http://127.0.0.1:9090",
                                galaxy_key="test_key",
                            )
                            store = local_conn.connect().create_data_store(
                                "IMAGINE-X SubHKL"
                            )
                            gi = store.nova_connection.galaxy_instance
                            self.galaxy_url = "http://127.0.0.1:9090"
                            self.api_key = gi.key
                            self.history_id = store.history_id

                    if not self._has_tool_on_galaxy(required_tool):
                        raise RuntimeError(
                            f"subhkl GPU tool '{required_tool}' not yet deployed on {self.galaxy_url} "
                            f"(awaiting MR merge on ndip/galaxy-tools)"
                        )
                    if (
                        stage_id != "reduce"
                        and getattr(self, "_pulsar_queue_stalled_until", 0.0)
                        > time.time()
                    ):
                        raise RuntimeError(
                            f"Galaxy Pulsar queue on {self.galaxy_url} is stalled in state=new"
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
                    "dMin": float(params.get("d_min") or round(0.9 * float(wl[0]), 2)),
                }
            elif stage_id == "integrate":
                primary_tool = "imaginex_subhkl_predict"
                scan_id = self._ensure_galaxy_h5_dataset(store, merged_path, job_record)
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                max_edge = max(float(x) for x in raw_cell[:3])
                min_edge = min(float(x) for x in raw_cell[:3])
                d_min_req = float(params.get("d_min") or round(0.9 * float(wl[0]), 2))
                d_min_int = (
                    min(d_min_req, max(0.75, min_edge * 0.085))
                    if max_edge < 35.0
                    else d_min_req
                )
                tool_inputs = {
                    "scan": {"src": "hda", "id": scan_id},
                    "indexer": {"src": "hda", "id": idx_id},
                    "instrument": inst,
                    "dMin": round(d_min_int, 3),
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
                work_dir, merged_label = self._work_dir_for_reduce(raw_nxs, work_dir)
                job_record["raw_nxs"] = raw_nxs
                job_record["instrument"] = inst
                job_record["merged_label"] = merged_label
                # Route NeXus verification through ONCat OAuth first (instant, no file copy),
                # falling back to NDIP neutrons_load_data only when ONCat is unavailable.
                verified_ids = list(params.get("verified_dataset_ids") or [])
                if not verified_ids and params.get("ingest_mode") in (
                    "oncat",
                    "file",
                    None,
                ):
                    import re as _re

                    first_p = raw_nxs[0]
                    ipts_str = _re.sub(
                        r"^IPTS-",
                        "",
                        str(params.get("ipts") or "").strip(),
                        flags=_re.I,
                    )
                    if not ipts_str:
                        m_ip = _re.search(r"/IPTS-([^/]+)/", first_p, _re.I)
                        if m_ip:
                            ipts_str = m_ip.group(1)
                    fac_str = str(
                        params.get("facility")
                        or ("SNS" if first_p.startswith("/SNS/") else "HFIR")
                    )
                    if ipts_str:
                        try:
                            cat = self.browse_oncat_catalog(
                                {
                                    "facility": fac_str,
                                    "instrument": inst,
                                    "ipts": ipts_str,
                                }
                            )
                            run_map = {
                                int(r["run"]): r
                                for r in (cat.get("runs") or [])
                                if r.get("run") is not None
                            }
                            if run_map:
                                missing_r = []
                                for p_chk in raw_nxs:
                                    m_rc = _re.search(
                                        r"_([0-9]+)\.nxs",
                                        os.path.basename(p_chk),
                                        _re.I,
                                    )
                                    if m_rc and int(m_rc.group(1)) not in run_map:
                                        missing_r.append(os.path.basename(p_chk))
                                if missing_r:
                                    raise FileNotFoundError(
                                        f"ONCat IPTS-{ipts_str}: run(s) not found ({', '.join(missing_r[:4])})"
                                    )
                                verified_ids = [
                                    f"oncat:{inst}:IPTS-{ipts_str}:{len(raw_nxs)}"
                                ]
                        except FileNotFoundError:
                            raise
                        except Exception:
                            pass

                if verified_ids:
                    job_record["stdout"] += (
                        f"[galaxy -> ndip] Verified dataset(s): {', '.join(str(x) for x in verified_ids)}\n"
                    )
                elif self._has_tool_on_galaxy("neutrons_load_data") and params.get(
                    "ingest_mode"
                ) in ("oncat", "file"):
                    import re as _re

                    nld_inputs: dict[str, Any] = {}
                    first_p = raw_nxs[0]
                    m_run = _re.search(
                        r"_([0-9]+)\.nxs", os.path.basename(first_p), _re.I
                    )
                    ipts_str = _re.sub(
                        r"^IPTS-",
                        "",
                        str(params.get("ipts") or "").strip(),
                        flags=_re.I,
                    )
                    if not ipts_str:
                        m_ip = _re.search(r"/IPTS-([^/]+)/", first_p, _re.I)
                        if m_ip:
                            ipts_str = m_ip.group(1)
                    nld_inputs = self._build_neutrons_load_data_inputs(
                        ingest_mode=str(params.get("ingest_mode") or "oncat"),
                        fac=str(
                            params.get("facility")
                            or ("SNS" if first_p.startswith("/SNS/") else "HFIR")
                        ),
                        inst=inst,
                        ipts_raw=ipts_str,
                        run_num=int(m_run.group(1)) if m_run else None,
                        filepath=first_p,
                    )
                    nld_res = self._run_neutrons_load_data_tool(
                        gi=gi,
                        history_id=store.history_id,
                        nld_inputs=nld_inputs,
                        filepath=first_p,
                    )
                    nld_jid = (nld_res.get("jobs") or [{}])[0].get("id")
                    nld_outs = [
                        o.get("id")
                        for o in (nld_res.get("outputs") or [])
                        if isinstance(o, dict) and o.get("id")
                    ]
                    job_record["stdout"] += (
                        f"[galaxy -> ndip] Dispatched neutrons_load_data ({nld_inputs.get('params|ingest_mode')}, job_id={nld_jid})\n"
                    )
                    for _ in range(18):
                        if not nld_jid:
                            break
                        j_nld = self._api_request(
                            "GET", f"/api/jobs/{nld_jid}?full=true"
                        )
                        st_nld = j_nld.get("state")
                        if st_nld == "ok":
                            break
                        if st_nld in ("error", "deleted"):
                            err_out = (
                                j_nld.get("tool_stdout")
                                or j_nld.get("tool_stderr")
                                or "File not found"
                            ).strip()
                            for oid in nld_outs:
                                try:
                                    self._api_request(
                                        "PUT",
                                        f"/api/histories/{store.history_id}/contents/{oid}",
                                        payload={"deleted": True},
                                    )
                                except Exception:
                                    pass
                            raise FileNotFoundError(
                                f"NDIP neutrons_load_data failed: {err_out.splitlines()[0]}"
                            )
                        time.sleep(1.0)
                if getattr(self, "_pulsar_queue_stalled_until", 0.0) > time.time():
                    raise RuntimeError(
                        f"Galaxy Pulsar queue on {self.galaxy_url} is stalled in state=new"
                    )
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
            elif stage_id == "merge":
                primary_tool = "imaginex_subhkl_merge"
                mtz_in_path = os.path.join(work_dir, "reflections.mtz")
                if not os.path.isfile(mtz_in_path):
                    raise RuntimeError(
                        f"Unmerged reflections.mtz not found in {work_dir}; run Integrate first."
                    )
                mtz_id = self._ensure_galaxy_h5_dataset(
                    store,
                    mtz_in_path,
                    job_record,
                    force_upload=True,
                    file_ext="mtz",
                )
                tool_inputs = {
                    "reflections": {"src": "hda", "id": mtz_id},
                    "iterations": int(params.get("iterations") or 10000),
                    "studenttDof": float(params.get("studentt_dof") or 4.0),
                    "mergeHalfDatasets": True,
                    "anomalous": bool(params.get("anomalous", False)),
                }
            else:
                primary_tool = "imaginex_subhkl_metrics"
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                tool_inputs = {
                    "peaks": {"src": "hda", "id": idx_id},
                }

            is_two_step_integrate = (
                stage_id == "integrate"
                and self._has_tool_on_galaxy("imaginex_subhkl_integrate")
            )
            is_multi_run_legacy_reduce = (
                stage_id == "reduce"
                and len(job_record.get("raw_nxs") or []) > 1
                and self._get_tool_version("imaginex_subhkl_prepare_peaks").startswith(
                    "0.1"
                )
            )
            if is_multi_run_legacy_reduce:
                import concurrent.futures
                import re as _re

                raw_nxs_all = list(job_record.get("raw_nxs") or [])
                if all(os.path.isfile(p) for p in raw_nxs_all):
                    raise RuntimeError(
                        f"imaginex_subhkl_prepare_peaks 0.1.0-dev on {self.galaxy_url} lacks multi-input batching; "
                        f"switching to multi-core container reduction ({len(raw_nxs_all)} runs)"
                    )
                out_sub = os.path.join(work_dir, "out")
                os.makedirs(out_sub, exist_ok=True)
                # Pre-warm ONCat catalog cache once for the sweep so parallel workers don't re-fetch it per run
                try:
                    m_ip = _re.search(r"/IPTS-([0-9]+)/", raw_nxs_all[0], _re.I)
                    if m_ip:
                        fac_pre = (
                            "HFIR"
                            if (
                                raw_nxs_all[0].startswith("/HFIR")
                                or inst.upper().startswith(("CG", "HB"))
                            )
                            else "SNS"
                        )
                        self.browse_oncat_catalog(
                            {
                                "facility": fac_pre,
                                "instrument": inst,
                                "ipts": m_ip.group(1),
                            }
                        )
                except Exception:
                    pass

                job_record["stdout"] += (
                    f"[galaxy -> pulsar] Multi-run reduction ({len(raw_nxs_all)} runs): "
                    f"dispatching parallel per-run {primary_tool} jobs on {self.galaxy_url}...\n"
                )

                def _submit_one_part(
                    arg: tuple[int, str],
                ) -> tuple[int, str, str, list[dict[str, Any]], str]:
                    idx_r, nxs_p = arg
                    bn = os.path.basename(nxs_p).split(".")[0]
                    part_path = os.path.join(out_sub, f"{bn}.reduce.h5")
                    part_inputs = {
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
                        "inputs_0|input": nxs_p,
                    }
                    part_res = gi.tools.run_tool(
                        history_id=store.history_id,
                        tool_id=primary_tool,
                        tool_inputs=part_inputs,
                    )
                    p_jid = part_res["jobs"][0]["id"]
                    p_outs = [
                        {
                            "id": d.get("id"),
                            "name": d.get("output_name") or d.get("name"),
                            "extension": d.get("file_ext")
                            or d.get("extension")
                            or "h5",
                        }
                        for d in (part_res.get("outputs") or [])
                        if isinstance(d, dict)
                    ]
                    return idx_r, nxs_p, p_jid, p_outs, part_path

                submit_workers = min(len(raw_nxs_all), 6)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=submit_workers
                ) as sub_pool:
                    submitted_indexed = list(
                        sub_pool.map(_submit_one_part, list(enumerate(raw_nxs_all)))
                    )
                submitted_indexed.sort(key=lambda item: item[0])
                submitted_parts = [
                    (nxs_p, p_jid, p_outs, part_path)
                    for idx_r, nxs_p, p_jid, p_outs, part_path in submitted_indexed
                ]
                for idx_r, (nxs_p, p_jid, _p_outs, _part_path) in enumerate(
                    submitted_parts
                ):
                    job_record["stdout"] += (
                        f"[galaxy -> pulsar] Dispatched Run {idx_r + 1}/{len(raw_nxs_all)} "
                        f"({os.path.basename(nxs_p)}, job_id={p_jid})\n"
                    )

                log_lock = threading.Lock()

                def _poll_one_part(
                    arg: tuple[int, tuple[str, str, list[dict[str, Any]], str]],
                ) -> tuple[int, str, str, str]:
                    idx_r, (nxs_p, p_jid, p_outs, part_path) = arg
                    part_rec: dict[str, Any] = {
                        "stage_id": "reduce",
                        "job_id": p_jid,
                        "outputs": p_outs,
                        "raw_nxs": [nxs_p],
                        "instrument": inst,
                        "merged_label": os.path.basename(part_path),
                        "reduce_part_path": part_path,
                        "state": "running",
                        "stdout": "",
                        "stderr": "",
                    }
                    self._poll_galaxy_job(
                        part_rec,
                        None,
                        final_step=False,
                        work_dir=work_dir,
                    )
                    for po in p_outs:
                        if po.get("id") and self.history_id:
                            try:
                                self._api_request(
                                    "PUT",
                                    f"/api/histories/{self.history_id}/contents/{po['id']}",
                                    payload={"deleted": True, "purged": True},
                                )
                            except Exception:
                                pass
                    st_part = str(part_rec.get("state") or "ok")
                    with log_lock:
                        job_record["stdout"] += (
                            f"[nova <- galaxy] Completed Run {idx_r + 1}/{len(raw_nxs_all)} "
                            f"({os.path.basename(nxs_p)}, state={st_part})\n"
                        )
                    return idx_r, part_path, st_part, str(part_rec.get("stderr") or "")

                poll_workers = min(len(submitted_parts), 8)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=poll_workers
                ) as poll_pool:
                    polled_results = list(
                        poll_pool.map(_poll_one_part, list(enumerate(submitted_parts)))
                    )
                polled_results.sort(key=lambda item: item[0])
                part_paths: list[str] = []
                for _idx_r, part_path, st_part, err_part in polled_results:
                    if st_part in ("error", "deleted", "paused"):
                        job_record["state"] = st_part
                        if err_part:
                            job_record["stderr"] = err_part
                        return
                    if os.path.isfile(part_path):
                        part_paths.append(part_path)

                job_record.pop("reduce_part_path", None)
                job_record["raw_nxs"] = raw_nxs_all
                if not part_paths:
                    raise RuntimeError(
                        "No reduced HDF5 files were produced by multi-run reduction."
                    )
                merged_label = str(job_record.get("merged_label") or "merged.h5")
                merged_dest = os.path.join(work_dir, merged_label)
                merged_link = os.path.join(work_dir, "merged.h5")
                from subhkl.commands import run_merge_images
                import h5py
                import numpy as np

                job_record["stdout"] += (
                    f"\n[subhkl] Merging {len(part_paths)} reduced runs (LZF compressed) -> {merged_dest}...\n"
                )
                run_merge_images(
                    " ".join(part_paths),
                    merged_dest,
                    float(raw_cell[0]),
                    float(raw_cell[1]),
                    float(raw_cell[2]),
                    float(raw_cell[3]),
                    float(raw_cell[4]),
                    float(raw_cell[5]),
                    sg,
                )
                try:
                    with h5py.File(merged_dest, "r+") as hf:
                        if "files" in hf:
                            del hf["files"]
                        hf["files"] = np.array(
                            [
                                f"out/{os.path.basename(p)}".encode("utf-8")
                                for p in part_paths
                            ]
                        )
                except Exception:
                    pass
                self._stamp_merged_metadata(merged_dest, job_record)
                if os.path.abspath(merged_dest) != os.path.abspath(merged_link):
                    try:
                        if os.path.exists(merged_link):
                            os.remove(merged_link)
                        os.link(merged_dest, merged_link)
                    except Exception:
                        shutil.copy2(merged_dest, merged_link)
                job_record["output_h5_path"] = merged_dest
                if callable(on_complete_h5):
                    on_complete_h5(merged_dest)
                    job_record["updated_payload_ready"] = True
                job_record["stdout"] += (
                    f"\n[galaxy -> pulsar] Multi-run reduction ({len(part_paths)} runs) finished (state=ok).\n"
                )
                job_record["state"] = "ok"

                def _bg_preupload() -> None:
                    try:
                        bg_rec: dict[str, Any] = {"stdout": ""}
                        up_id = self._ensure_galaxy_h5_dataset(
                            store,
                            merged_link,
                            bg_rec,
                            force_upload=True,
                            file_ext="h5",
                        )
                        job_record["outputs"] = [
                            {
                                "id": up_id,
                                "name": merged_label,
                                "extension": "h5",
                            }
                        ]
                    except Exception:
                        pass

                threading.Thread(target=_bg_preupload, daemon=True).start()
                return

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
                gi.datasets.wait_for_dataset(
                    pred_ds_id, maxwait=60, interval=1.5, check=True
                )
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
        except FileNotFoundError as fnf_exc:
            clean_msg = self._clean_err_msg(fnf_exc)
            job_record["state"] = "error"
            job_record["stderr"] = f"{clean_msg}\n"
            job_record["stdout"] += f"\n[galaxy -> ndip] Error: {clean_msg}\n"
        except Exception as exc:
            clean_msg = self._clean_err_msg(exc)
            job_record["stdout"] += (
                f"[nova -> galaxy] {clean_msg}; running GPU container on IMAGINEX-1...\n"
            )
            self._submit_local_subhkl_stage(
                stage_id,
                params,
                on_complete_h5,
                fallback_reason=clean_msg,
                existing_job_record=job_record,
            )

    def check_neutron_paths(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate NeXus / HDF5 file paths strictly via NDIP's ``neutrons_load_data`` tool.

        Never probes ``/SNS`` or ``/HFIR`` directly on the host/container filesystem so
        user permissions are always enforced by NDIP / Galaxy.
        """
        import re as _re

        fac = str(body.get("facility") or "HFIR").strip()
        inst = str(body.get("instrument") or "CG4D").strip()
        ipts_raw = _re.sub(
            r"^IPTS-", "", str(body.get("ipts") or "").strip(), flags=_re.I
        )
        raw_paths = body.get("paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [
                s.strip() for s in raw_paths.replace("\n", ",").split(",") if s.strip()
            ]

        expanded: list[str] = [str(p).strip() for p in raw_paths if str(p).strip()]
        if not expanded:
            return {
                "ok": False,
                "ndip_checked": False,
                "existing": [],
                "missing": [],
                "ipts_exists": False,
                "suggested_ipts": None,
                "suggested_paths": [],
                "verified_dataset_ids": [],
                "sample_title": "",
                "sample_goniometer": "",
                "message": "No NeXus / HDF5 file paths specified",
            }

        ok = False
        ndip_checked = False
        verified_dataset_ids: list[str] = []
        msg = ""
        sample_title = ""
        sample_goniometer = ""
        oncat_existing: list[str] = []
        oncat_missing: list[str] = []

        if self.is_connected and self._has_tool_on_galaxy("neutrons_load_data"):
            try:
                ndip_checked = True
                m_p0 = _re.match(
                    r"^/(SNS|HFIR)/([^/]+)/IPTS-([^/]+)/", expanded[0], _re.I
                )
                if m_p0 and (
                    str(body.get("ingest_mode") or "oncat") == "file" or not ipts_raw
                ):
                    fac = m_p0.group(1).upper()
                    inst = m_p0.group(2).upper()
                    ipts_raw = m_p0.group(3)
                elif not ipts_raw:
                    m_ip0 = _re.search(r"/IPTS-([^/]+)/", expanded[0], _re.I)
                    if m_ip0:
                        ipts_raw = m_ip0.group(1)
                # Verify via authenticated pyoncat catalog first (instant metadata check, no file fetch)
                if ipts_raw:
                    try:
                        cat = self.browse_oncat_catalog(
                            {
                                "facility": fac,
                                "instrument": inst,
                                "ipts": ipts_raw,
                                "skip_experiments": True,
                            }
                        )
                        run_map = {
                            int(r["run"]): r
                            for r in (cat.get("runs") or [])
                            if r.get("run") is not None
                        }
                        if run_map:
                            titles_seen: list[str] = []
                            gonio_seen: list[str] = []
                            for p in expanded:
                                m_r = _re.search(
                                    r"_([0-9]+)\.nxs", os.path.basename(p), _re.I
                                )
                                if m_r:
                                    r_num = int(m_r.group(1))
                                    info = run_map.get(r_num)
                                    if info:
                                        oncat_existing.append(info.get("location") or p)
                                        t_str = str(info.get("title") or "").strip()
                                        g_str = str(info.get("angles") or "").strip()
                                        if t_str and t_str not in titles_seen:
                                            titles_seen.append(t_str)
                                        if g_str and g_str not in gonio_seen:
                                            gonio_seen.append(g_str)
                                    else:
                                        oncat_missing.append(p)
                            sample_title = "; ".join(titles_seen[:3])
                            sample_goniometer = "; ".join(gonio_seen[:3])
                            if oncat_existing or oncat_missing:
                                if oncat_missing:
                                    ok = False
                                    msg = (
                                        f"ONCat: {len(oncat_existing)} found, {len(oncat_missing)} missing "
                                        f"({', '.join(os.path.basename(x) for x in oncat_missing[:4])})"
                                    )
                                else:
                                    ok = True
                                    verified_dataset_ids = [
                                        f"oncat:{fac}:{inst}:IPTS-{ipts_raw}:{len(oncat_existing)}"
                                    ]
                                    msg = (
                                        f"Verified {len(oncat_existing)} run(s) via ONCat "
                                        f"({inst} IPTS-{ipts_raw})"
                                    )
                                    if sample_title:
                                        msg += f" — {sample_title[:60]}"
                        elif (cat.get("auth_status") or {}).get("authenticated"):
                            oncat_missing = list(expanded)
                            ok = False
                            msg = f"ONCat: No accessible runs found for {inst} IPTS-{ipts_raw}"
                    except Exception:
                        pass

                if not ok and not msg:
                    store = self.get_datastore("IMAGINE-X SubHKL")
                    gi = store.nova_connection.galaxy_instance
                    first_p = oncat_existing[0] if oncat_existing else expanded[0]
                    m_run = _re.search(
                        r"_([0-9]+)\.nxs", os.path.basename(first_p), _re.I
                    )
                    nld_in = self._build_neutrons_load_data_inputs(
                        ingest_mode=str(body.get("ingest_mode") or "oncat"),
                        fac=fac,
                        inst=inst,
                        ipts_raw=ipts_raw,
                        run_num=int(m_run.group(1)) if m_run else None,
                        filepath=first_p,
                    )
                    nld_res = self._run_neutrons_load_data_tool(
                        gi=gi,
                        history_id=store.history_id,
                        nld_inputs=nld_in,
                        filepath=first_p,
                    )
                    jid = (nld_res.get("jobs") or [{}])[0].get("id")
                    out_ids = [
                        o.get("id")
                        for o in (nld_res.get("outputs") or [])
                        if isinstance(o, dict) and o.get("id")
                    ]
                    for _ in range(45):
                        if not jid:
                            break
                        j = self._api_request("GET", f"/api/jobs/{jid}?full=true")
                        st = j.get("state")
                        if st == "ok":
                            ok = True
                            verified_dataset_ids = out_ids or [
                                f"verified:{len(expanded)}"
                            ]
                            msg = (
                                f"Verified {len(expanded)} run(s) via NDIP Load Neutron Data "
                                f"({inst} IPTS-{ipts_raw or '?'})"
                            )
                            if sample_title:
                                msg += f" — {sample_title[:60]}"
                            if len(expanded) > 1 or not ok:
                                for oid in out_ids:
                                    try:
                                        self._api_request(
                                            "PUT",
                                            f"/api/histories/{store.history_id}/contents/{oid}",
                                            payload={"deleted": True},
                                        )
                                    except Exception:
                                        pass
                            break
                        if st in ("error", "deleted"):
                            err_out = (
                                j.get("tool_stdout") or j.get("tool_stderr") or ""
                            ).strip()
                            first_line = (
                                err_out.splitlines()[0]
                                if err_out
                                else "File or run not found"
                            )
                            msg = f"NDIP Load Neutron Data: {first_line} ({inst} IPTS-{ipts_raw or '?'})"
                            for oid in out_ids:
                                try:
                                    self._api_request(
                                        "PUT",
                                        f"/api/histories/{store.history_id}/contents/{oid}",
                                        payload={"deleted": True},
                                    )
                                except Exception:
                                    pass
                            break
                        time.sleep(1.0)
                    if not ok and not msg:
                        msg = f"NDIP Load Neutron Data check timed out for {first_p}"
            except Exception as exc:
                msg = f"NDIP Load Neutron Data error: {self._clean_err_msg(exc)}"
        else:
            msg = "NDIP Galaxy connection required to verify /SNS or /HFIR paths via neutrons_load_data"

        return {
            "ok": ok,
            "ndip_checked": ndip_checked,
            "checked_via": "ndip_neutrons_load_data"
            if ndip_checked
            else "ndip_unavailable",
            "existing": oncat_existing
            if oncat_existing and ok
            else (list(expanded) if ok else []),
            "missing": oncat_missing
            if oncat_missing
            else ([] if ok else list(expanded)),
            "ipts_exists": ok,
            "suggested_ipts": None,
            "suggested_paths": [],
            "verified_dataset_ids": verified_dataset_ids,
            "sample_title": sample_title,
            "sample_goniometer": sample_goniometer,
            "message": msg,
        }

    ONCAT_DEFAULT_URL = "https://oncat.ornl.gov"
    # Public human-user device-authorization client ID published by ONCat docs.
    ONCAT_DEFAULT_CLIENT_ID = "eaeb036a-2602-4bb9-8530-0bb5812da7a1"
    ONCAT_DEFAULT_SCOPES = ["api:read", "openid"]

    def _get_oncat_token_path(self) -> str:
        return os.path.expanduser(
            os.environ.get("ONCAT_TOKEN_PATH", "~/.oncat/token.json")
        )

    def _seed_oidc_token_if_available(self) -> None:
        """If an outer NDIP/Galaxy OIDC token or token_endpoint is present in env, seed pyoncat's token store."""
        import pyoncat  # type: ignore[import-untyped]

        token_path = self._get_oncat_token_path()
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        store = pyoncat.FileSystemTokenStore(token_path)
        if store.read_token():
            return

        raw_token = (
            os.environ.get("ONCAT_TOKEN") or os.environ.get("NDIP_OIDC_TOKEN") or ""
        ).strip()
        if not raw_token:
            token_ep = (
                os.environ.get("GALAXY_TOKEN_ENDPOINT")
                or os.environ.get("PULSAR_TOKEN_ENDPOINT")
                or os.environ.get("TOKEN_ENDPOINT")
                or ""
            ).strip()
            if token_ep:
                provider = os.environ.get("OIDC_PROVIDER", "pingfed").strip()
                try:
                    import urllib.request

                    sep = "&" if "?" in token_ep else "?"
                    with urllib.request.urlopen(
                        f"{token_ep}{sep}provider={provider}", timeout=5
                    ) as resp:
                        raw_token = resp.read().decode("utf-8").strip().strip('"')
                except Exception:
                    raw_token = ""

        if raw_token:
            try:
                if raw_token.startswith("{"):
                    tok_obj = json.loads(raw_token)
                else:
                    tok_obj = {
                        "access_token": raw_token,
                        "token_type": "Bearer",
                    }
                store.write_token(tok_obj)
            except Exception:
                pass

    def _build_oauth_oncat_client(self, verification_handler: Any = None) -> Any:
        """Build a pyoncat.ONCat client configured with FileSystemTokenStore and OAuth flow."""
        import pyoncat  # type: ignore[import-untyped]

        self._seed_oidc_token_if_available()
        url = os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
        client_id = os.environ.get("ONCAT_CLIENT_ID", self.ONCAT_DEFAULT_CLIENT_ID)
        client_secret = os.environ.get("ONCAT_CLIENT_SECRET")
        username = os.environ.get("ONCAT_USERNAME")
        password = os.environ.get("ONCAT_PASSWORD")
        token_path = self._get_oncat_token_path()
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        store = pyoncat.FileSystemTokenStore(token_path)

        if client_id and client_secret:
            client = pyoncat.ONCat(
                url,
                client_id=client_id,
                client_secret=client_secret,
                flow=pyoncat.CLIENT_CREDENTIALS_FLOW,
                token_getter=store.read_token,
                token_setter=store.write_token,
            )
            if not client.has_stored_token():
                client.login()
            return client

        if client_id and username and password:
            client = pyoncat.ONCat(
                url,
                client_id=client_id,
                flow=pyoncat.RESOURCE_OWNER_CREDENTIALS_FLOW,
                token_getter=store.read_token,
                token_setter=store.write_token,
            )
            if not client.has_stored_token():
                client.login(username, password)
            return client

        return pyoncat.ONCat(
            url,
            client_id=client_id,
            flow=pyoncat.DEVICE_AUTHORIZATION_FLOW,
            scopes=list(self.ONCAT_DEFAULT_SCOPES),
            token_getter=store.read_token,
            token_setter=store.write_token,
            reauth_on_expired=pyoncat.REAUTH_INTERACTION_REQUIRED,
            verification_handler=verification_handler,
        )

    def _get_oncat_client(self) -> tuple[Any, bool]:
        """Return (pyoncat.ONCat client, is_authenticated).

        Uses stored OAuth/OIDC token via pyoncat when available; falls back to
        anonymous pyoncat client for public catalog fields when not signed in.
        """
        url = os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
        oauth_client = None
        try:
            oauth_client = self._build_oauth_oncat_client()
            if oauth_client.has_stored_token():
                return oauth_client, True
        except Exception:
            pass
        try:
            import pyoncat  # type: ignore[import-untyped]

            return pyoncat.ONCat(url), False
        except Exception:
            return oauth_client, False

    def _invalidate_oncat_caches(self) -> None:
        self._oncat_exp_cache = {}
        self._oncat_runs_cache = {}

    def get_oncat_auth_status(self) -> dict[str, Any]:
        """Return current pyoncat OAuth session state and any active device-flow challenge."""
        state: dict[str, Any] = getattr(self, "_oncat_auth_state", {})
        try:
            oauth_client = self._build_oauth_oncat_client()
            has_tok = bool(oauth_client.has_stored_token())
        except Exception:
            oauth_client = None
            has_tok = False

        if has_tok and oauth_client is not None:
            user_name = state.get("user")
            if not user_name:
                try:
                    u = oauth_client.User.retrieve("me")
                    if isinstance(u, dict):
                        user_name = u.get("name")
                        if not user_name or user_name == "Unknown":
                            user_name = u.get("id")
                    state["user"] = user_name or "ORNL User"
                    self._oncat_auth_state = state
                except Exception as exc:
                    import pyoncat  # type: ignore[import-untyped]

                    if isinstance(exc, pyoncat.InteractionRequiredError):
                        try:
                            tp = self._get_oncat_token_path()
                            if os.path.isfile(tp):
                                os.remove(tp)
                        except Exception:
                            pass
                        has_tok = False
                    else:
                        user_name = "ORNL User"
            if has_tok:
                return {
                    "ok": True,
                    "authenticated": True,
                    "status": "signed_in",
                    "user": user_name or "ORNL User",
                    "verification_uri": None,
                    "verification_uri_complete": None,
                    "user_code": None,
                    "expires_in": None,
                    "message": f"Signed in to ONCat as {user_name or 'ORNL User'}",
                }

        if state.get("status") == "pending_verification":
            return {
                "ok": True,
                "authenticated": False,
                "status": "pending_verification",
                "user": None,
                "verification_uri": state.get("verification_uri"),
                "verification_uri_complete": state.get("verification_uri_complete"),
                "user_code": state.get("user_code"),
                "expires_in": state.get("expires_in"),
                "message": state.get("message")
                or "Waiting for browser approval via ORNL PingFed...",
            }

        return {
            "ok": True,
            "authenticated": False,
            "status": state.get("status") or "signed_out",
            "user": None,
            "verification_uri": None,
            "verification_uri_complete": None,
            "user_code": None,
            "expires_in": None,
            "message": state.get("message")
            or "Anonymous mode (sign in to ONCat for run titles & sample metadata)",
        }

    def handle_oncat_auth(self, body: dict[str, Any]) -> dict[str, Any]:
        """Start/cancel pyoncat Device Authorization sign-in, accept forwarded OIDC token, or sign out."""
        action = str(body.get("action") or "status").strip().lower()
        if action == "status":
            return self.get_oncat_auth_status()

        if action == "logout":
            cancel_ev = getattr(self, "_oncat_login_cancel_event", None)
            if cancel_ev is not None:
                cancel_ev.set()
            tp = self._get_oncat_token_path()
            try:
                if os.path.isfile(tp):
                    os.remove(tp)
            except Exception:
                pass
            self._oncat_auth_state = {
                "status": "signed_out",
                "user": None,
                "message": "Signed out of ONCat.",
            }
            self._invalidate_oncat_caches()
            return self.get_oncat_auth_status()

        if action == "cancel":
            cancel_ev = getattr(self, "_oncat_login_cancel_event", None)
            if cancel_ev is not None:
                cancel_ev.set()
            self._oncat_auth_state = {
                "status": "signed_out",
                "user": None,
                "message": "Sign-in cancelled.",
            }
            return self.get_oncat_auth_status()

        if action == "token":
            raw_tok = str(body.get("access_token") or "").strip()
            if not raw_tok:
                return {
                    "ok": False,
                    "authenticated": False,
                    "status": "error",
                    "message": "No access_token provided",
                }
            tp = self._get_oncat_token_path()
            os.makedirs(os.path.dirname(tp), exist_ok=True)
            try:
                import pyoncat  # type: ignore[import-untyped]

                store = pyoncat.FileSystemTokenStore(tp)
                store.write_token({"access_token": raw_tok, "token_type": "Bearer"})
            except ImportError:
                with open(tp, "w", encoding="utf-8") as fh:
                    json.dump({"access_token": raw_tok, "token_type": "Bearer"}, fh)
            self._oncat_auth_state = {"status": "signed_in", "user": None}
            self._invalidate_oncat_caches()
            return self.get_oncat_auth_status()

        if action == "start":
            prev_cancel = getattr(self, "_oncat_login_cancel_event", None)
            if prev_cancel is not None:
                prev_cancel.set()

            cancel_ev = threading.Event()
            ready_ev = threading.Event()
            self._oncat_login_cancel_event = cancel_ev
            state: dict[str, Any] = {
                "status": "pending_verification",
                "user": None,
                "verification_uri": None,
                "verification_uri_complete": None,
                "user_code": None,
                "expires_in": None,
                "message": "Requesting device code from ORNL PingFed...",
            }
            self._oncat_auth_state = state

            def _on_challenge(challenge: Any) -> None:
                uri = getattr(challenge, "verification_uri", None)
                uri_complete = (
                    getattr(challenge, "verification_uri_complete", None) or uri
                )
                code = getattr(challenge, "user_code", None)
                exp = getattr(challenge, "expires_in", None)
                state.update(
                    {
                        "status": "pending_verification",
                        "verification_uri": uri,
                        "verification_uri_complete": uri_complete,
                        "user_code": code,
                        "expires_in": exp,
                        "message": f"Approve in browser (Code: {code})",
                    }
                )
                ready_ev.set()

            def _worker() -> None:
                try:
                    client = self._build_oauth_oncat_client(
                        verification_handler=_on_challenge
                    )
                    client.login(
                        cancel_event=cancel_ev,
                        device_authorization_timeout=300.0,
                    )
                    user_name = "ORNL User"
                    try:
                        u = client.User.retrieve("me")
                        if isinstance(u, dict):
                            user_name = (
                                u.get("name")
                                if u.get("name") and u.get("name") != "Unknown"
                                else (u.get("id") or "ORNL User")
                            )
                    except Exception:
                        pass
                    state.update(
                        {
                            "status": "signed_in",
                            "user": user_name,
                            "verification_uri": None,
                            "verification_uri_complete": None,
                            "user_code": None,
                            "expires_in": None,
                            "message": f"Signed in to ONCat as {user_name}",
                        }
                    )
                    self._invalidate_oncat_caches()
                except Exception as exc:
                    if not cancel_ev.is_set():
                        state.update(
                            {
                                "status": "signed_out",
                                "user": None,
                                "verification_uri": None,
                                "verification_uri_complete": None,
                                "user_code": None,
                                "expires_in": None,
                                "message": f"ONCat sign-in ended: {self._clean_err_msg(exc)}",
                            }
                        )
                finally:
                    ready_ev.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            ready_ev.wait(timeout=3.0)
            return self.get_oncat_auth_status()

        return self.get_oncat_auth_status()

    def browse_oncat_catalog(self, params: dict[str, Any]) -> dict[str, Any]:
        """Browse IPTS projects and runs via pyoncat (using OAuth token when signed in)."""
        import re as _re

        fac = str(params.get("facility") or "HFIR").strip().upper()
        inst = str(params.get("instrument") or "CG4D").strip().upper()
        if inst in ("MANDI", "TOPAZ") and fac == "HFIR":
            fac = "SNS"
        elif inst in ("CG4D", "IMAGINE") and fac == "SNS":
            fac = "HFIR"
        ipts_raw = _re.sub(
            r"^IPTS-", "", str(params.get("ipts") or "").strip(), flags=_re.I
        )

        # Multi-run support:
        # NDIP's raw `neutrons_load_data` (oncat mode) accepts 1 integer run per tool invocation,
        # whereas `imaginex_subhkl_prepare_peaks` (and local GPU container reduction) accepts
        # multiple run locations resolved by pyoncat and merges them via `subhkl merge-images`.
        has_prepare_peaks = not self.is_connected or self._has_tool_on_galaxy(
            "imaginex_subhkl_prepare_peaks"
        )
        multi_run_supported = bool(has_prepare_peaks)
        multi_run_note = (
            "Multi-run selection supported via pyoncat + subhkl merge-images"
            if multi_run_supported
            else "Single run only (NDIP neutrons_load_data integer run limit)"
        )

        raw_client = self._get_oncat_client()
        if isinstance(raw_client, tuple):
            oncat, is_auth = raw_client
        else:
            oncat, is_auth = raw_client, False
        auth_status = self.get_oncat_auth_status()
        is_auth = bool(auth_status.get("authenticated") or is_auth)

        exp_cache: dict[tuple[str, str, bool], tuple[float, list[dict[str, Any]]]] = (
            getattr(self, "_oncat_exp_cache", {})
        )
        now = time.time()
        experiments: list[dict[str, Any]] = []
        skip_exps = bool(params.get("skip_experiments")) and bool(ipts_raw)
        cached_exp = exp_cache.get((fac, inst, is_auth))
        if cached_exp and (now - cached_exp[0]) < 300.0:
            experiments = cached_exp[1]
        elif not skip_exps:
            try:
                try:
                    raw_exps = oncat.Experiment.list(
                        facility=fac,
                        instrument=inst,
                    )
                except Exception:
                    import pyoncat as _pyoncat  # type: ignore[import-untyped]

                    oncat = _pyoncat.ONCat(
                        os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
                    )
                    is_auth = False
                    raw_exps = oncat.Experiment.list(facility=fac, instrument=inst)

                for e in raw_exps:
                    eid = str(e.get("id") or "")
                    m_ip = _re.search(r"IPTS-(\d+)", eid, _re.I)
                    ip_num = m_ip.group(1) if m_ip else eid
                    n_runs = int(e.get("size") or 0)
                    r_info = (e.get("indexed") or {}).get("run_number") or {}
                    ranges_str = str(r_info.get("ranges") or "")
                    latest_dt = str(((e.get("latest") or {}).get("created") or ""))[:10]
                    title_str = str(e.get("title") or "").strip()
                    experiments.append(
                        {
                            "ipts": ip_num,
                            "id": f"IPTS-{ip_num}",
                            "n_runs": n_runs,
                            "run_ranges": ranges_str,
                            "min_run": r_info.get("min"),
                            "max_run": r_info.get("max"),
                            "latest_date": latest_dt,
                            "title": title_str,
                        }
                    )
                experiments.sort(
                    key=lambda x: (
                        1 if (x.get("n_runs") or 0) > 0 else 0,
                        int(x["ipts"]) if str(x.get("ipts", "")).isdigit() else 0,
                    ),
                    reverse=True,
                )
                exp_cache[(fac, inst, is_auth)] = (now, experiments)
                self._oncat_exp_cache = exp_cache
            except Exception:
                pass

        if not ipts_raw and experiments:
            ipts_raw = str(experiments[0]["ipts"])

        runs: list[dict[str, Any]] = []
        if ipts_raw:
            runs_cache: dict[
                tuple[str, str, str, bool], tuple[float, list[dict[str, Any]]]
            ] = getattr(self, "_oncat_runs_cache", {})
            cached_runs = runs_cache.get((fac, inst, ipts_raw, is_auth))
            if cached_runs and (now - cached_runs[0]) < 300.0:
                runs = cached_runs[1]
            else:
                by_run: dict[int, dict[str, Any]] = {}
                try:
                    proj = [
                        "indexed.run_number",
                        "location",
                        "size",
                        "created",
                        "indexed.sample_id",
                        "metadata.entry.title",
                        "metadata.entry.sample.name",
                        "metadata.entry.daslogs.omega.average_value",
                        "metadata.entry.daslogs.chi.average_value",
                        "metadata.entry.daslogs.phi.average_value",
                        "metadata.entry.duration",
                        "metadata.entry.total_counts",
                    ]
                    try:
                        dfs = oncat.Datafile.list(
                            facility=fac,
                            instrument=inst,
                            experiment=f"IPTS-{ipts_raw}",
                            projection=proj,
                            exts=[".nxs.h5"],
                        )
                    except Exception:
                        import pyoncat as _pyoncat  # type: ignore[import-untyped]

                        oncat = _pyoncat.ONCat(
                            os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
                        )
                        is_auth = False
                        dfs = oncat.Datafile.list(
                            facility=fac,
                            instrument=inst,
                            experiment=f"IPTS-{ipts_raw}",
                            projection=proj,
                            exts=[".nxs.h5"],
                        )

                    for d in dfs:
                        idx_d = d.get("indexed") or {}
                        r_val = idx_d.get("run_number")
                        if r_val is None:
                            m_r = _re.search(
                                r"_([0-9]+)\.nxs",
                                str(d.get("location") or ""),
                                _re.I,
                            )
                            if m_r:
                                r_val = int(m_r.group(1))
                        if r_val is None:
                            continue
                        r_num = int(r_val)
                        meta_entry = (d.get("metadata") or {}).get("entry") or {}
                        loc = str(
                            d.get("location")
                            or f"/{fac}/{inst}/IPTS-{ipts_raw}/nexus/{inst}_{r_num}.nxs.h5"
                        )
                        title = str(meta_entry.get("title") or "").strip()
                        sample_name = str(
                            (meta_entry.get("sample") or {}).get("name") or ""
                        ).strip()
                        daslogs = meta_entry.get("daslogs") or {}
                        angle_parts: list[str] = []
                        for ax_name in ("omega", "chi", "phi"):
                            ax_obj = daslogs.get(ax_name)
                            if (
                                isinstance(ax_obj, dict)
                                and ax_obj.get("average_value") is not None
                            ):
                                try:
                                    val_f = float(ax_obj["average_value"])
                                    angle_parts.append(f"{ax_name}={val_f:.1f}")
                                except (TypeError, ValueError):
                                    pass
                        angles = ";".join(angle_parts)
                        if not angles and title:
                            ax_m = _re.findall(
                                r"(axis\s*\d+|omega|phi|chi|kappa)\s*=?\s*([-+]?\d+(?:\.\d+)?)",
                                title,
                                flags=_re.I,
                            )
                            if ax_m:
                                angles = ";".join(
                                    f"{k.replace(' ', '').lower()}={v}" for k, v in ax_m
                                )
                        sz_bytes = int(d.get("size") or 0)
                        created_str = str(d.get("created") or "")[:16].replace("T", " ")
                        dur_s = float(meta_entry.get("duration") or 0.0)
                        if dur_s > 0:
                            tot_sec = int(round(dur_s))
                            hh, rem = divmod(tot_sec, 3600)
                            mm, ss = divmod(rem, 60)
                            dur_hms = f"{hh:02d}:{mm:02d}:{ss:02d}"
                        else:
                            dur_hms = ""
                        by_run[r_num] = {
                            "run": r_num,
                            "location": loc,
                            "title": title,
                            "sample_name": sample_name,
                            "angles": angles,
                            "size_mb": round(sz_bytes / (1024.0 * 1024.0), 1),
                            "created": created_str,
                            "duration_s": dur_s,
                            "duration_hms": dur_hms,
                            "total_counts": int(
                                float(meta_entry.get("total_counts") or 0)
                            ),
                        }
                except Exception:
                    pass

                runs = [by_run[k] for k in sorted(by_run.keys())]
                runs_cache[(fac, inst, ipts_raw, is_auth)] = (now, runs)
                self._oncat_runs_cache = runs_cache

        return {
            "ok": True,
            "facility": fac,
            "instrument": inst,
            "ipts": ipts_raw,
            "oncat_authenticated": is_auth,
            "oncat_auth": auth_status,
            "multi_run_supported": multi_run_supported,
            "multi_run_note": multi_run_note,
            "experiments": experiments,
            "runs": runs,
        }

    def _resolve_raw_nexus_inputs(
        self, params: dict[str, Any], work_dir: str
    ) -> list[str]:
        """Resolve raw .nxs.h5 paths from explicit nexus_files, galaxy_dataset_id, IPTS+runs, or work_dir."""
        import glob as _glob

        resolved: list[str] = []
        raw_list = params.get("nexus_files")
        if isinstance(raw_list, str):
            raw_list = [
                s.strip() for s in raw_list.replace("\n", ",").split(",") if s.strip()
            ]
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
                local_p, _ = self.resolve_dataset_to_local_h5(
                    str(ds_id), fallback_dir=work_dir
                )
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
        work_dir, merged_path = self._resolve_work_dir_and_merged(merged_path)
        solve_dest = os.path.join(work_dir, "solve.h5")
        pred_dest = os.path.join(work_dir, "predictor.h5")
        integrated_dest = os.path.join(work_dir, "integrated.h5")
        mtz_dest = os.path.join(work_dir, "reflections.mtz")

        solve_path = os.path.abspath(params.get("peaks_file") or solve_dest)
        if os.path.basename(solve_path) in (
            "merged.h5",
            "integrated.h5",
            "predictor.h5",
        ) and os.path.isfile(solve_dest):
            solve_path = solve_dest

        ts = int(time.time())
        out_h5 = os.path.join(tempfile.gettempdir(), f"subhkl_{stage_id}_{ts}.h5")
        pred_tmp_h5 = os.path.join(tempfile.gettempdir(), f"subhkl_pred_{ts}.h5")
        mtz_tmp_path = os.path.join(tempfile.gettempdir(), f"subhkl_refl_{ts}.mtz")
        reduce_tmp_dir = os.path.join(tempfile.gettempdir(), f"subhkl_reduce_{ts}")
        raw_nxs_local: list[str] = []
        merged_label = "merged.h5"
        if stage_id == "reduce":
            raw_nxs_local = [
                p
                for p in self._resolve_raw_nexus_inputs(params, work_dir)
                if os.path.isfile(p)
            ]
            if raw_nxs_local:
                work_dir, merged_label = self._work_dir_for_reduce(
                    raw_nxs_local, work_dir
                )
                solve_dest = os.path.join(work_dir, "solve.h5")
                pred_dest = os.path.join(work_dir, "predictor.h5")
                integrated_dest = os.path.join(work_dir, "integrated.h5")
                mtz_dest = os.path.join(work_dir, "reflections.mtz")

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
                    "    _orig_cd = h5py.File.create_dataset\n"
                    "    def _lzf_cd(self, name, shape=None, dtype=None, data=None, **kw):\n"
                    "        sh = shape if shape is not None else getattr(data, 'shape', ())\n"
                    "        if name == 'images' and len(sh) >= 3 and sh[0] > 0:\n"
                    "            kw.setdefault('compression', 'lzf')\n"
                    "            kw.setdefault('chunks', (1,) + tuple(sh[1:]))\n"
                    "        return _orig_cd(self, name, shape=shape, dtype=dtype, data=data, **kw)\n"
                    "    h5py.File.create_dataset = _lzf_cd\n"
                    "except Exception:\n"
                    "    pass\n"
                )
            batch_py_path = os.path.join(reduce_tmp_dir, "_run_batch_reduce.py")
            with open(batch_py_path, "w") as bf:
                bf.write(
                    "import os, sys, concurrent.futures, multiprocessing as mp\n"
                    "os.environ.setdefault('JAX_PLATFORMS', 'cpu')\n"
                    "from subhkl.commands import run_reduce\n"
                    "def _w(t):\n"
                    "    run_reduce(t[0], t[1], t[2], wavelength_min=t[3], wavelength_max=t[4])\n"
                    "if __name__ == '__main__':\n"
                    "    inst = sys.argv[1]\n"
                    "    wl_min, wl_max = float(sys.argv[2]), float(sys.argv[3])\n"
                    "    out_dir = sys.argv[4]\n"
                    "    nxs_files = sys.argv[5:]\n"
                    "    tasks = [(p, os.path.join(out_dir, f\"{os.path.basename(p).split('.')[0] or f'run_{i:03d}'}.reduce.h5\"), inst, wl_min, wl_max) for i, p in enumerate(nxs_files)]\n"
                    "    if len(tasks) == 1:\n"
                    "        _w(tasks[0])\n"
                    "    else:\n"
                    "        workers = min(len(tasks), os.cpu_count() or 8, 16)\n"
                    "        print(f'Reducing {len(tasks)} runs in parallel across {workers} CPU cores...')\n"
                    "        ctx = mp.get_context('fork')\n"
                    "        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:\n"
                    "            list(ex.map(_w, tasks))\n"
                )
            reduced_parts_ordered = [
                os.path.join(
                    reduce_tmp_dir,
                    f"{os.path.basename(p).split('.')[0] or f'run_{i:03d}'}.reduce.h5",
                )
                for i, p in enumerate(raw_nxs_local)
            ]
            r_cmd = [
                "python",
                "-u",
                batch_py_path,
                str(inst),
                str(wl[0]),
                str(wl[1]),
                reduce_tmp_dir,
                *raw_nxs_local,
            ]
            merge_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "merge-images",
                " ".join(reduced_parts_ordered),
                out_h5,
                str(float(raw_cell[0])),
                str(float(raw_cell[1])),
                str(float(raw_cell[2])),
                str(float(raw_cell[3])),
                str(float(raw_cell[4])),
                str(float(raw_cell[5])),
                str(sg),
            ]
            reduce_cmds = [
                shlex.join(r_cmd),
                shlex.join(merge_cmd),
                f"rm -f {shlex.quote(reduce_tmp_dir)}/*.reduce.h5",
            ]
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
                str(params.get("d_min") or round(0.9 * float(wl[0]), 2)),
                "--refinement",
                str(params.get("objective", "band")),
            ]
        elif (
            stage_id == "integrate"
            and os.path.isfile(merged_path)
            and os.path.isfile(solve_path)
        ):
            max_edge = max(float(x) for x in raw_cell[:3])
            min_edge = min(float(x) for x in raw_cell[:3])
            d_min_req = float(params.get("d_min") or round(0.9 * float(wl[0]), 2))
            d_min_int = (
                min(d_min_req, max(0.75, min_edge * 0.085))
                if max_edge < 35.0
                else d_min_req
            )
            d_min_val = str(round(d_min_int, 3))
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
        elif stage_id == "merge":
            if not os.path.isfile(mtz_dest):
                alt_mtz = os.path.join(
                    os.path.dirname(os.path.abspath(self.replay_h5)),
                    "reflections.mtz",
                )
                if os.path.isfile(alt_mtz):
                    shutil.copy2(alt_mtz, mtz_dest)
                else:
                    src_int = (
                        integrated_dest
                        if os.path.isfile(integrated_dest)
                        else self.replay_h5
                    )
                    if os.path.isfile(src_int):
                        try:
                            from subhkl.io.export import MTZExporter

                            exp = MTZExporter(
                                src_int,
                                space_group_name=str(sg),
                                predictions_file=(
                                    pred_dest if os.path.isfile(pred_dest) else None
                                ),
                                corrections_file=(
                                    solve_path if os.path.isfile(solve_path) else None
                                ),
                            )
                            exp.write_mtz(mtz_dest)
                        except Exception:
                            pass
            careless_iters = int(params.get("iterations") or 10000)
            studentt_dof = float(params.get("studentt_dof") or 4.0)
            merge_work = os.path.join(work_dir, "careless_work")
            os.makedirs(merge_work, exist_ok=True)
            unmerged_mtz = os.path.join(merge_work, "input_unmerged.mtz")
            out_prefix = os.path.join(merge_work, "careless")
            merged_mtz_out = os.path.join(work_dir, "merged.mtz")
            pred_mtz_out = os.path.join(work_dir, "predictions.mtz")
            xval_mtz_out = os.path.join(work_dir, "xval.mtz")
            metrics_json_out = os.path.join(work_dir, "careless_metrics.json")
            prep_py = (
                "import gemmi, reciprocalspaceship as rs\n"
                f"mtz = gemmi.read_mtz_file({mtz_dest!r})\n"
                "c = mtz.cell\n"
                "na = 90.0 if abs(c.alpha - 90) < 0.2 else (120.0 if abs(c.alpha - 120) < 0.2 else c.alpha)\n"
                "nb = 90.0 if abs(c.beta - 90) < 0.2 else (120.0 if abs(c.beta - 120) < 0.2 else c.beta)\n"
                "ng = 90.0 if abs(c.gamma - 90) < 0.2 else (120.0 if abs(c.gamma - 120) < 0.2 else c.gamma)\n"
                "mtz.cell.set(c.a, c.b, c.c, na, nb, ng)\n"
                f"cell_mtz = {os.path.join(merge_work, 'input_cell.mtz')!r}\n"
                "mtz.write_to_file(cell_mtz)\n"
                "ds = rs.read_mtz(cell_mtz)\n"
                "if 'SIGI' in ds:\n"
                "    ds = ds[ds['SIGI'] > 0]\n"
                "if not ds.get_m_isym_keys():\n"
                "    ds = ds.hkl_to_asu()\n"
                "ds.merged = False\n"
                f"ds.write_mtz({unmerged_mtz!r})\n"
                "cols = {col.label for col in mtz.columns}\n"
                "keys = ['WAVEL','THETA','PHI','dHKL','H_0','K_0','L_0','H','K','L','image_id','file_id','asu_id','harmonic_id']\n"
                "for extra in ('SIGEFF', 'SNAPD', 'DPHI', 'DTX', 'DTY', 'DTZ'):\n"
                "    if extra in cols:\n"
                "        keys.append(extra)\n"
                "print(','.join(keys))\n"
            )
            post_py = (
                "import json, os, shutil, gemmi, numpy as np\n"
                "import reciprocalspaceship as rs\n"
                "from scipy.stats import pearsonr, spearmanr\n"
                f"shutil.copy2({(out_prefix + '_0.mtz')!r}, {merged_mtz_out!r})\n"
                f"shutil.copy2({(out_prefix + '_predictions_0.mtz')!r}, {pred_mtz_out!r})\n"
                f"if os.path.exists({(out_prefix + '_xval_0.mtz')!r}):\n"
                f"    shutil.copy2({(out_prefix + '_xval_0.mtz')!r}, {xval_mtz_out!r})\n"
                "summary = {'status': 'ok'}\n"
                f"mtz = gemmi.read_mtz_file({merged_mtz_out!r})\n"
                "cols = {c.label: np.array(c, dtype=np.float64) for c in mtz.columns}\n"
                "I = cols.get('I', cols.get('I(+)', np.array([])))\n"
                "SigI = cols.get('SigI', cols.get('SigI(+)', np.array([])))\n"
                "valid = np.isfinite(I) & np.isfinite(SigI) & (SigI > 0)\n"
                "summary['n_merged_asu'] = int(np.sum(valid))\n"
                "summary['mean_isigi'] = float(np.mean(I[valid] / SigI[valid])) if np.any(valid) else 0.0\n"
                f"pmtz = gemmi.read_mtz_file({pred_mtz_out!r})\n"
                "n_obs = int(pmtz.nreflections)\n"
                "summary['n_observations'] = n_obs\n"
                "summary['mean_multiplicity'] = float(n_obs / max(1, summary['n_merged_asu'])) if summary['n_merged_asu'] > 0 else 0.0\n"
                "try:\n"
                "    h_m = cols['H'][valid].astype(int); k_m = cols['K'][valid].astype(int); l_m = cols['L'][valid].astype(int)\n"
                "    d_m = np.array([mtz.cell.calculate_d([int(hh), int(kk), int(ll)]) for hh, kk, ll in zip(h_m, k_m, l_m)])\n"
                "    dmin_obs = float(np.min(d_m)) if len(d_m) else 2.0\n"
                "    theo_asu = rs.utils.generate_reciprocal_asu(mtz.cell, mtz.spacegroup, dmin_obs)\n"
                "    summary['n_theoretical_asu'] = int(len(theo_asu))\n"
                "    summary['completeness_pct'] = float(100.0 * summary['n_merged_asu'] / max(1, len(theo_asu)))\n"
                "    summary['d_min'] = round(dmin_obs, 3)\n"
                "    summary['d_max'] = round(float(np.max(d_m)), 3) if len(d_m) else 10.0\n"
                "except Exception as e:\n"
                "    summary['completeness_error'] = str(e)\n"
                f"if os.path.exists({xval_mtz_out!r}):\n"
                f"    xv = gemmi.read_mtz_file({xval_mtz_out!r})\n"
                "    xcols = {c.label: np.array(c, dtype=np.float64) for c in xv.columns}\n"
                "    h = xcols['H'].astype(int); k = xcols['K'].astype(int); l = xcols['L'].astype(int)\n"
                "    half = xcols.get('half', np.zeros_like(h)).astype(int)\n"
                "    xI = xcols.get('I', xcols.get('F', np.zeros_like(xcols['H'])))\n"
                "    d_arr = np.array([xv.cell.calculate_d([int(hh), int(kk), int(ll)]) for hh, kk, ll in zip(h, k, l)])\n"
                "    m0 = half == 0; m1 = half == 1\n"
                "    map0 = {(int(hh), int(kk), int(ll)): (float(val), float(dd)) for hh, kk, ll, val, dd in zip(h[m0], k[m0], l[m0], xI[m0], d_arr[m0]) if np.isfinite(val)}\n"
                "    map1 = {(int(hh), int(kk), int(ll)): float(val) for hh, kk, ll, val in zip(h[m1], k[m1], l[m1], xI[m1]) if np.isfinite(val)}\n"
                "    common = [key for key in map0 if key in map1]\n"
                "    if len(common) >= 2:\n"
                "        v0 = np.array([map0[k][0] for k in common]); v1 = np.array([map1[k] for k in common]); d_c = np.array([map0[k][1] for k in common])\n"
                "        summary['cc_half'] = float(pearsonr(v0, v1)[0]) if len(common) >= 3 else float(np.corrcoef(v0, v1)[0, 1])\n"
                "        summary['cc_half_spearman'] = float(spearmanr(v0, v1)[0]) if len(common) >= 3 else summary['cc_half']\n"
                "        inv_d2 = 1.0 / np.maximum(d_c ** 2, 1e-6)\n"
                "        edges = np.linspace(inv_d2.min(), inv_d2.max(), 11)\n"
                "        bins = []\n"
                "        for b in range(10):\n"
                "            bm = (inv_d2 >= edges[b]) & (inv_d2 <= edges[b+1] if b == 9 else inv_d2 < edges[b+1])\n"
                "            cc_b = float(np.corrcoef(v0[bm], v1[bm])[0, 1]) if np.sum(bm) >= 2 else float('nan')\n"
                "            d_hi = float(1.0 / np.sqrt(max(edges[b+1], 1e-6)))\n"
                "            d_lo = float(1.0 / np.sqrt(max(edges[b], 1e-6)))\n"
                "            bins.append({'bin': b, 'd_max': round(d_lo, 2), 'd_min': round(d_hi, 2), 'n_refl': int(np.sum(bm)), 'cc_half': round(cc_b, 4) if np.isfinite(cc_b) else None})\n"
                "        summary['resolution_bins'] = bins\n"
                f"with open({metrics_json_out!r}, 'w') as fh:\n"
                "    json.dump(summary, fh, indent=2)\n"
                f"for _p in ({merged_mtz_out!r}, {pred_mtz_out!r}, {xval_mtz_out!r}, {metrics_json_out!r}):\n"
                "    if os.path.exists(_p):\n"
                "        try: os.chmod(_p, 0o666)\n"
                "        except Exception: pass\n"
                "print('[careless] Merge summary:', json.dumps(summary))\n"
            )
            prep_script = os.path.join(merge_work, "prep_merge.py")
            post_script = os.path.join(merge_work, "post_merge.py")
            with open(prep_script, "w", encoding="utf-8") as fh:
                fh.write(prep_py)
            with open(post_script, "w", encoding="utf-8") as fh:
                fh.write(post_py)
            try:
                os.chmod(prep_script, 0o666)
                os.chmod(post_script, 0o666)
            except Exception:
                pass
            chained_merge = (
                f"set -e; "
                f"MKEYS=$(python {shlex.quote(prep_script)}) && "
                f'careless poly --jit-compile --iterations={careless_iters} --merge-half-datasets --wavelength-key=WAVEL --studentt-likelihood-dof={studentt_dof} "$MKEYS" {shlex.quote(unmerged_mtz)} {shlex.quote(out_prefix)} && '
                f"python {shlex.quote(post_script)}"
            )
            inner_cmd = ["bash", "-c", chained_merge]
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

        container_image = (
            "ghcr.io/jglaser/careless:cuda"
            if stage_id == "merge"
            else "ghcr.io/neutrons/subhkl:feat-rainbow-zones"
        )
        if shutil.which("podman"):
            raw_dirs = {os.path.dirname(os.path.abspath(p)) for p in raw_nxs_local}
            mount_dirs = sorted(
                {
                    os.path.dirname(merged_path),
                    os.path.dirname(solve_path),
                    work_dir,
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
                "JAX_PLATFORMS=",
                "-e",
                "MKL_NUM_THREADS=1",
            ]
            for d in mount_dirs:
                if os.path.isdir(d):
                    cmd.extend(["-v", f"{d}:{d}"])
            local_subhkl_pkg = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            if os.path.isdir(local_subhkl_pkg) and stage_id != "merge":
                cmd.extend(
                    [
                        "-v",
                        f"{local_subhkl_pkg}:/usr/local/lib/python3.13/site-packages/subhkl:ro",
                    ]
                )
            cmd.append(container_image)
            cmd.extend(inner_cmd)
        else:
            if inner_cmd[0] == "bash":
                cmd = inner_cmd
            else:
                cmd = [sys.executable] + inner_cmd[1:]

        gpu_env = os.environ.copy()
        gpu_env["JAX_PLATFORMS"] = ""
        gpu_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        gpu_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        gpu_env.setdefault("OMP_NUM_THREADS", "1")
        gpu_env.setdefault("MKL_NUM_THREADS", "1")
        try:
            gpu_info = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    timeout=2.0,
                )
                .strip()
                .splitlines()[0]
            )
        except Exception:
            gpu_info = "CPU fallback"

        host_tag = socket.gethostname()
        cmd_str = " ".join(cmd)
        header_lines: list[str] = []
        if stage_id != "merge":
            if fallback_reason and existing_job_record is None:
                header_lines.append(f"[ndip] Note: {fallback_reason}")
            header_lines.append(
                f"[container] Host={host_tag} | GPU={gpu_info} | Image={container_image}"
            )
            header_lines.append(f"$ {cmd_str}\n")

        job_id = f"gpu-{int(time.time())}"
        if existing_job_record is not None:
            job_record = existing_job_record
            job_record["job_id"] = job_id
            job_record["mode"] = "container_gpu"
            job_record["state"] = "running"
            job_record["command_line"] = cmd_str
            if stage_id == "merge":
                job_record["stdout"] = ""
            else:
                job_record["stdout"] = (
                    job_record.get("stdout", "") + "\n".join(header_lines)
                )[-24000:]
        else:
            job_record = {
                "job_id": job_id,
                "tool_id": f"subhkl::{stage_id}",
                "stage_id": stage_id,
                "mode": "container_gpu",
                "state": "running",
                "command_line": cmd_str,
                "stdout": "" if stage_id == "merge" else "\n".join(header_lines),
                "stderr": "",
                "ladder_rung_idx": 0 if stage_id == "solve" else 4,
                "outputs": [
                    {"id": job_id, "name": os.path.basename(out_h5), "extension": "h5"}
                ],
                "updated_payload_ready": False,
                "started_at": time.time(),
            }
            self.active_job = job_record

        if stage_id == "solve":
            job_record["images_h5_path"] = merged_path
            job_record["solve_params"] = {
                "instrument": str(inst),
                "space_group": str(sg),
                "cell": [float(x) for x in raw_cell[:6]],
                "wavelength_range": (float(wl[0]), float(wl[1])),
            }

        def _stream_proc() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    env=gpu_env,
                )
                assert proc.stdout is not None
                blind_src = out_h5[:-3] + ".blind.h5"
                blind_dst = solve_dest[:-3] + ".blind.h5"
                last_blind_mtime = 0.0
                partial_seq = 0
                initial_prefix = job_record.get("stdout", "")
                if initial_prefix and not initial_prefix.endswith("\n"):
                    initial_prefix += "\n"
                raw_stream = ""
                fd = proc.stdout.fileno()
                while True:
                    chunk_bytes = os.read(fd, 4096)
                    if not chunk_bytes:
                        break
                    chunk_str = chunk_bytes.decode("utf-8", errors="replace")
                    raw_stream += chunk_str
                    if len(raw_stream) > 96000 and "\n" in raw_stream[:-8192]:
                        cut_nl = raw_stream.rfind("\n", 0, len(raw_stream) - 8192)
                        if cut_nl > 0:
                            raw_stream = (
                                self._format_terminal_stream(raw_stream[:cut_nl])
                                + "\n"
                                + raw_stream[cut_nl + 1 :]
                            )
                    formatted_stream = self._format_terminal_stream(raw_stream)
                    job_record["stdout"] = (initial_prefix + formatted_stream)[-24000:]
                    low = chunk_str.lower()
                    if stage_id == "solve":
                        trigger_blind_sync = False
                        if (
                            "blind band search" in low
                            or "band table" in low
                            or "exhaustive rung" in low
                            or "rung 13 zones" in low
                        ):
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 1
                            )
                            trigger_blind_sync = True
                        elif "rung 30 zones" in low or "rung 60 zones" in low:
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 2
                            )
                            trigger_blind_sync = True
                        elif (
                            "rung 145 zones" in low
                            or "joint band refinement" in low
                            or "candidate " in low
                            or "full-data scores" in low
                            or "global" in low
                            or "60-zone" in low
                            or "stage 0" in low
                            or "120" in low
                            or "cell" in low
                            or "stage 1" in low
                            or "stage 3" in low
                        ):
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 3
                            )
                            trigger_blind_sync = True
                        if (
                            trigger_blind_sync
                            and os.path.isfile(blind_src)
                            and os.path.getsize(blind_src) > 0
                        ):
                            try:
                                cur_mtime = os.path.getmtime(blind_src)
                                if cur_mtime >= last_blind_mtime:
                                    last_blind_mtime = cur_mtime
                                    shutil.copy2(blind_src, blind_dst)
                                    job_record["partial_h5_path"] = blind_dst
                                    from subhkl.viz.rainbow_zones import (
                                        build_rainbow_zone_payload,
                                    )

                                    p_blind = build_rainbow_zone_payload(
                                        replay_h5=blind_dst,
                                        images_h5=merged_path,
                                        instrument=str(inst),
                                        space_group=str(sg),
                                        cell=[float(x) for x in raw_cell[:6]],
                                        wavelength_range=(
                                            float(wl[0]),
                                            float(wl[1]),
                                        ),
                                        max_zone_index=1,
                                        max_zones=13,
                                        include_panels=False,
                                        include_peaks=False,
                                        n_arc=720,
                                        frame_index=self.current_frame_index,
                                    )
                                    if p_blind.get("zones"):
                                        partial_seq += 1
                                        job_record["partial_overlay"] = {
                                            "seq": partial_seq,
                                            "zones": p_blind.get("zones", []),
                                            "peaks": [],
                                            "ladder_steps": p_blind.get(
                                                "ladder_steps", []
                                            ),
                                        }
                            except Exception:
                                pass
                rc = proc.wait()
                if rc == 0:
                    job_record["ladder_rung_idx"] = 0 if stage_id == "reduce" else 4
                    final_h5 = out_h5
                    if stage_id == "reduce" and os.path.isfile(out_h5):
                        merged_dest = os.path.join(work_dir, merged_label)
                        merged_link = os.path.join(work_dir, "merged.h5")
                        try:
                            self._stamp_merged_metadata(out_h5, job_record)
                            shutil.move(out_h5, merged_dest)
                            if os.path.abspath(merged_dest) != os.path.abspath(
                                merged_link
                            ):
                                try:
                                    if os.path.lexists(merged_link):
                                        os.remove(merged_link)
                                    os.link(merged_dest, merged_link)
                                except Exception:
                                    shutil.copy2(merged_dest, merged_link)
                            final_h5 = merged_dest
                            job_record["stdout"] += (
                                f"\n[subhkl] Saved {merged_label} -> {merged_dest}\n"
                            )
                        except Exception as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving {merged_dest}: {copy_exc}\n"
                            )

                        if self.is_connected and os.path.isfile(final_h5):

                            def _bg_upload_reduced(
                                _f_h5: str = final_h5,
                                _lbl: str = merged_label,
                            ) -> None:
                                try:
                                    store = self.get_datastore("IMAGINE-X SubHKL")
                                    bg_rec: dict[str, Any] = {"stdout": ""}
                                    up_id = self._ensure_galaxy_h5_dataset(
                                        store,
                                        _f_h5,
                                        bg_rec,
                                        force_upload=True,
                                        file_ext="h5",
                                    )
                                    job_record["outputs"] = [
                                        {
                                            "id": up_id,
                                            "name": os.path.basename(_f_h5),
                                            "extension": "h5",
                                        }
                                    ]
                                except Exception:
                                    pass

                            threading.Thread(
                                target=_bg_upload_reduced, daemon=True
                            ).start()
                    elif stage_id == "solve" and os.path.isfile(out_h5):
                        try:
                            shutil.copy2(out_h5, solve_dest)
                            for suffix in (".stages.h5", ".blind.h5"):
                                src_sub = out_h5[:-3] + suffix
                                dst_sub = solve_dest[:-3] + suffix
                                if os.path.isfile(src_sub):
                                    shutil.copy2(src_sub, dst_sub)
                            final_h5 = solve_dest
                            job_record["stdout"] += (
                                f"\n[subhkl] Saved solve.h5 -> {solve_dest}\n"
                            )
                        except Exception as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving {solve_dest}: {copy_exc}\n"
                            )

                        if self.is_connected and os.path.isfile(final_h5):
                            try:
                                store = self.get_datastore("IMAGINE-X SubHKL")
                                up_id = self._ensure_galaxy_h5_dataset(
                                    store,
                                    final_h5,
                                    job_record,
                                    force_upload=True,
                                    file_ext="h5",
                                )
                                job_record["outputs"] = [
                                    {
                                        "id": up_id,
                                        "name": "solve.h5",
                                        "extension": "h5",
                                    }
                                ]
                            except Exception as up_exc:
                                job_record["stdout"] += (
                                    f"[nova -> galaxy] Warning uploading solve.h5: {self._clean_err_msg(up_exc)}\n"
                                )
                    elif stage_id == "integrate" and os.path.isfile(out_h5):
                        try:
                            if os.path.isfile(pred_tmp_h5):
                                shutil.copy2(pred_tmp_h5, pred_dest)
                                job_record["stdout"] += (
                                    f"\n[subhkl] Saved predictor.h5 -> {pred_dest}\n"
                                )
                            shutil.copy2(out_h5, integrated_dest)
                            final_h5 = integrated_dest
                            job_record["stdout"] += (
                                f"[subhkl] Saved integrated.h5 -> {integrated_dest}\n"
                            )
                            if os.path.isfile(mtz_tmp_path):
                                shutil.copy2(mtz_tmp_path, mtz_dest)
                                job_record["stdout"] += (
                                    f"[subhkl] Saved reflections.mtz -> {mtz_dest}\n"
                                )
                        except Exception as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving integrated outputs: {copy_exc}\n"
                            )

                        if self.is_connected:
                            try:
                                store = self.get_datastore("IMAGINE-X SubHKL")
                                outs_up = []
                                if os.path.isfile(final_h5):
                                    int_id = self._ensure_galaxy_h5_dataset(
                                        store,
                                        final_h5,
                                        job_record,
                                        force_upload=True,
                                        file_ext="h5",
                                    )
                                    outs_up.append(
                                        {
                                            "id": int_id,
                                            "name": "integrated.h5",
                                            "extension": "h5",
                                        }
                                    )
                                if os.path.isfile(pred_dest):
                                    pred_id = self._ensure_galaxy_h5_dataset(
                                        store,
                                        pred_dest,
                                        job_record,
                                        force_upload=True,
                                        file_ext="h5",
                                    )
                                    outs_up.append(
                                        {
                                            "id": pred_id,
                                            "name": "predictor.h5",
                                            "extension": "h5",
                                        }
                                    )
                                if os.path.isfile(mtz_dest):
                                    mtz_id = self._ensure_galaxy_h5_dataset(
                                        store,
                                        mtz_dest,
                                        job_record,
                                        force_upload=True,
                                        file_ext="binary",
                                    )
                                    outs_up.append(
                                        {
                                            "id": mtz_id,
                                            "name": "reflections.mtz",
                                            "extension": "binary",
                                        }
                                    )
                                if outs_up:
                                    job_record["outputs"] = outs_up
                            except Exception as up_exc:
                                job_record["stdout"] += (
                                    f"[nova -> galaxy] Warning uploading integrated outputs: {self._clean_err_msg(up_exc)}\n"
                                )
                    elif stage_id == "merge":
                        target_int = (
                            integrated_dest
                            if os.path.isfile(integrated_dest)
                            else self.replay_h5
                        )
                        try:
                            if os.path.isfile(target_int):
                                self._stamp_careless_merge_into_h5(
                                    target_int, work_dir, job_record
                                )
                                final_h5 = target_int
                        except Exception as merge_exc:
                            job_record["stdout"] += (
                                f"\n[careless] Warning stamping merge results into HDF5: {merge_exc}\n"
                            )
                        if self.is_connected:
                            try:
                                store = self.get_datastore("IMAGINE-X SubHKL")
                                outs_up = []
                                for fn, ext in (
                                    ("merged.mtz", "binary"),
                                    ("predictions.mtz", "binary"),
                                    ("xval.mtz", "binary"),
                                    ("careless_metrics.json", "json"),
                                ):
                                    fp = os.path.join(work_dir, fn)
                                    if os.path.isfile(fp):
                                        did = self._ensure_galaxy_h5_dataset(
                                            store,
                                            fp,
                                            job_record,
                                            force_upload=True,
                                            file_ext=ext,
                                        )
                                        outs_up.append(
                                            {
                                                "id": did,
                                                "name": fn,
                                                "extension": ext,
                                            }
                                        )
                                if outs_up:
                                    job_record["outputs"] = outs_up
                            except Exception as up_exc:
                                job_record["stdout"] += (
                                    f"[nova -> galaxy] Warning uploading Careless outputs: {self._clean_err_msg(up_exc)}\n"
                                )

                    if os.path.isfile(final_h5):
                        job_record["output_h5_path"] = final_h5
                        if callable(on_complete_h5):
                            on_complete_h5(final_h5)
                            job_record["updated_payload_ready"] = True
                    job_record["stdout"] += (
                        f"[subhkl] Stage '{stage_id}' completed (exit 0).\n"
                    )
                    job_record["state"] = "ok"
                else:
                    job_record["state"] = "error"
                    job_record["stderr"] += (
                        f"\n[subhkl] Process exited with code {rc}\n"
                    )
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
        new_state_ticks = 0
        base_log = job_record.get("stdout", "")

        while True:
            time.sleep(1.5)
            try:
                # Fetch job status & live Pulsar console_output stream from NDIP Galaxy API
                jinfo = self._api_request("GET", f"/api/jobs/{job_id}?full=true")
                poll_errors = 0
                state = jinfo.get("state", "running")
                if state == "new":
                    new_state_ticks += 1
                    if new_state_ticks >= 10:
                        self._pulsar_queue_stalled_until = time.time() + 300.0
                        try:
                            self._api_request("DELETE", f"/api/jobs/{job_id}")
                        except Exception:
                            pass
                        raise TimeoutError(
                            f"Galaxy Pulsar queue did not pick up job {job_id} after 15s (stuck in state=new)"
                        )
                else:
                    new_state_ticks = 0
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

                out_txt = self._format_terminal_stream(
                    console_out or jinfo.get("tool_stdout") or jinfo.get("stdout") or ""
                )
                err_txt = self._format_terminal_stream(
                    console_err or jinfo.get("tool_stderr") or jinfo.get("stderr") or ""
                )
                if job_record.get("stage_id") == "merge":
                    job_record["stdout"] = out_txt[-24000:]
                else:
                    raw_cmd = str(job_record.get("command_line") or "").strip()
                    cmd_header = (
                        f"$ {raw_cmd}\n"
                        if raw_cmd and "\n" not in raw_cmd and "PYEOF" not in raw_cmd
                        else ""
                    )
                    job_record["stdout"] = f"{base_log}{cmd_header}{out_txt}"[-24000:]
                if err_txt:
                    job_record["stderr"] = err_txt[-16000:]

                if job_record.get("stage_id") == "solve" and state in (
                    "queued",
                    "running",
                ):
                    step_counter = min(3, step_counter + 1)
                    job_record["ladder_rung_idx"] = step_counter
                if state in ("ok", "error", "deleted", "paused"):
                    if state == "ok":
                        try:
                            job_record["ladder_rung_idx"] = (
                                0 if job_record.get("stage_id") == "reduce" else 4
                            )
                            if not final_step:
                                job_record["stdout"] += (
                                    f"\n[galaxy -> pulsar] Step job {job_id} finished.\n"
                                )
                            h5_outs = [
                                o
                                for o in job_record.get("outputs", [])
                                if o.get("extension")
                                in ("h5", "hdf5", "data", "auto", "binary", None)
                                and not str(o.get("name", ""))
                                .lower()
                                .endswith((".mtz", ".png", ".pdf", ".log"))
                                and "mtz" not in str(o.get("name", "")).lower()
                            ]
                            mtz_outs = [
                                o
                                for o in job_record.get("outputs", [])
                                if o.get("extension") == "mtz"
                                or "mtz" in str(o.get("name", "")).lower()
                            ]
                            if h5_outs:
                                ds_id = h5_outs[0]["id"]
                                st_id = str(job_record.get("stage_id") or "")
                                default_lbl = (
                                    "merged.h5"
                                    if st_id == "reduce"
                                    else (
                                        "solve.h5"
                                        if st_id == "solve"
                                        else (
                                            "predictor.h5"
                                            if (st_id == "integrate" and not final_step)
                                            else "integrated.h5"
                                        )
                                    )
                                )
                                merged_label = str(
                                    job_record.get("merged_label") or default_lbl
                                )
                                if (
                                    job_record.get("stage_id") == "reduce"
                                    and final_step
                                    and self.history_id
                                ):
                                    try:
                                        self._api_request(
                                            "PUT",
                                            f"/api/histories/{self.history_id}/contents/{ds_id}",
                                            payload={"name": merged_label},
                                        )
                                        h5_outs[0]["name"] = merged_label
                                        job_record["stdout"] += (
                                            f"[galaxy -> history] Created dataset '{merged_label}' (id={ds_id}) in Galaxy history\n"
                                        )
                                    except Exception:
                                        pass
                                target_path = None
                                if (
                                    not final_step
                                    and job_record.get("stage_id") == "reduce"
                                    and job_record.get("reduce_part_path")
                                ):
                                    target_path = str(job_record["reduce_part_path"])
                                elif work_dir and os.path.isdir(work_dir):
                                    if (
                                        not final_step
                                        and job_record.get("stage_id") == "integrate"
                                    ):
                                        target_path = os.path.join(
                                            work_dir, "predictor.h5"
                                        )
                                    elif job_record.get("stage_id") == "integrate":
                                        target_path = os.path.join(
                                            work_dir, "integrated.h5"
                                        )
                                    elif job_record.get("stage_id") == "solve":
                                        target_path = os.path.join(work_dir, "solve.h5")
                                    elif job_record.get("stage_id") == "reduce":
                                        target_path = os.path.join(
                                            work_dir, "merged.h5"
                                        )

                                tmp_path = None
                                cache_gx_path = os.path.join(
                                    tempfile.gettempdir(),
                                    f"subhkl_gx_{ds_id}_{merged_label}",
                                )
                                raw_bytes = None
                                for attempt in range(6):
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
                                        if attempt == 5:
                                            raise dl_exc
                                    time.sleep(2.0)
                                if raw_bytes:
                                    if target_path:
                                        with open(target_path, "wb") as f:
                                            f.write(raw_bytes)
                                        tmp_path = target_path
                                        if not job_record.get("reduce_part_path"):
                                            try:
                                                with open(cache_gx_path, "wb") as cf:
                                                    cf.write(raw_bytes)
                                            except Exception:
                                                pass
                                        job_record["stdout"] += (
                                            f"[nova <- galaxy] Synced '{merged_label}' (id={ds_id}) -> {target_path}\n"
                                        )
                                    else:
                                        fd, tmp_path = tempfile.mkstemp(
                                            suffix=".h5", prefix="galaxy_solve_"
                                        )
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
                                        from subhkl.instrument.goniometer import (
                                            calc_goniometer_rotation_matrix,
                                            get_rotation_data_from_nexus,
                                            get_rotation_data_from_title,
                                        )

                                        raw_nxs_list = job_record["raw_nxs"]
                                        inst_name = str(
                                            job_record.get("instrument") or "CG4D"
                                        )
                                        with h5py.File(tmp_path, "r+") as hf:
                                            if "goniometer/angles" in hf:
                                                cur_ang = np.asarray(
                                                    hf["goniometer/angles"][()]
                                                )
                                                if np.allclose(cur_ang, 0.0):
                                                    new_axes, new_angles, new_names = (
                                                        [],
                                                        [],
                                                        [],
                                                    )
                                                    for nxs_f in raw_nxs_list:
                                                        if nxs_f.startswith(
                                                            tempfile.gettempdir()
                                                        ) and os.path.isfile(nxs_f):
                                                            ax, ang, nms = (
                                                                get_rotation_data_from_nexus(
                                                                    nxs_f, inst_name
                                                                )
                                                            )
                                                        else:
                                                            title_meta = self._lookup_oncat_run_title(
                                                                nxs_f
                                                            )
                                                            ax, ang, nms = (
                                                                get_rotation_data_from_title(
                                                                    title_meta,
                                                                    inst_name,
                                                                )
                                                            )
                                                        if ang:
                                                            new_axes.append(ax)
                                                            new_angles.append(ang)
                                                            new_names.append(nms)
                                                    if new_angles and not np.allclose(
                                                        new_angles, 0.0
                                                    ):
                                                        del hf["goniometer/angles"]
                                                        hf["goniometer/angles"] = (
                                                            np.asarray(
                                                                new_angles,
                                                                dtype=np.float64,
                                                            )
                                                        )
                                                        if "goniometer/axes" in hf:
                                                            del hf["goniometer/axes"]
                                                        hf["goniometer/axes"] = (
                                                            np.asarray(
                                                                new_axes,
                                                                dtype=np.float64,
                                                            )
                                                        )
                                                        if "goniometer/names" in hf:
                                                            del hf["goniometer/names"]
                                                        hf["goniometer/names"] = (
                                                            np.asarray(
                                                                new_names, dtype="S"
                                                            )
                                                        )
                                                        new_R = [
                                                            calc_goniometer_rotation_matrix(
                                                                ax_r, ang_r
                                                            )
                                                            for ax_r, ang_r in zip(
                                                                new_axes, new_angles
                                                            )
                                                        ]
                                                        if "goniometer/R" in hf:
                                                            del hf["goniometer/R"]
                                                        hf["goniometer/R"] = np.asarray(
                                                            new_R, dtype=np.float64
                                                        )
                                                        job_record["stdout"] += (
                                                            f"[subhkl] Patched goniometer angles from NeXus/ONCat title into {os.path.basename(tmp_path)}\n"
                                                        )
                                        if os.path.isfile(cache_gx_path):
                                            shutil.copy2(tmp_path, cache_gx_path)
                                    except Exception:
                                        pass

                                if (
                                    tmp_path
                                    and os.path.isfile(tmp_path)
                                    and job_record.get("stage_id") == "reduce"
                                ):
                                    self._stamp_merged_metadata(tmp_path, job_record)
                                    if os.path.isfile(cache_gx_path):
                                        self._stamp_merged_metadata(
                                            cache_gx_path, job_record
                                        )

                                if final_step and tmp_path and os.path.isfile(tmp_path):
                                    job_record["output_h5_path"] = tmp_path
                                    if callable(on_complete_h5):
                                        on_complete_h5(tmp_path)
                                        job_record["updated_payload_ready"] = True
                            if (
                                final_step
                                and job_record.get("stage_id") == "merge"
                                and work_dir
                                and os.path.isdir(work_dir)
                            ):
                                for out_d in job_record.get("outputs", []):
                                    oid = out_d.get("id")
                                    oname = str(out_d.get("name") or "")
                                    if not oid:
                                        continue
                                    dest_name = None
                                    for cand in (
                                        "merged.mtz",
                                        "predictions.mtz",
                                        "xval.mtz",
                                        "careless_metrics.json",
                                    ):
                                        if cand in oname or oname == cand.replace(
                                            ".", "_"
                                        ):
                                            dest_name = cand
                                            break
                                    if dest_name:
                                        try:
                                            bdata = self._api_request(
                                                "GET",
                                                f"/api/datasets/{oid}/display",
                                                raw=True,
                                                timeout=180.0,
                                            )
                                            dest_p = os.path.join(work_dir, dest_name)
                                            with open(dest_p, "wb") as f:
                                                f.write(bdata)
                                            job_record["stdout"] += (
                                                f"[careless] Saved {dest_name} -> {dest_p}\n"
                                            )
                                        except Exception:
                                            pass
                                int_target = os.path.join(work_dir, "integrated.h5")
                                if os.path.isfile(int_target):
                                    self._stamp_careless_merge_into_h5(
                                        int_target, work_dir, job_record
                                    )
                                    job_record["output_h5_path"] = int_target
                                    if callable(on_complete_h5):
                                        on_complete_h5(int_target)
                                        job_record["updated_payload_ready"] = True
                            elif (
                                final_step
                                and mtz_outs
                                and work_dir
                                and os.path.isdir(work_dir)
                            ):
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
                                    job_record["stdout"] += (
                                        f"[subhkl] Saved reflections.mtz -> {mtz_dest}\n"
                                    )
                                except Exception:
                                    pass
                        except Exception as post_exc:
                            job_record["stdout"] += (
                                f"[nova -> galaxy] Warning fetching job output: {self._clean_err_msg(post_exc)}\n"
                            )
                        finally:
                            if final_step:
                                job_record["stdout"] += (
                                    f"\n[galaxy -> pulsar] Job {job_id} finished (state=ok).\n"
                                )
                                job_record["state"] = "ok"
                    else:
                        job_record["state"] = state
                    break
            except TimeoutError:
                raise
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
                        ds_name = str(
                            ds_info.get("name") or os.path.basename(local_h5)
                        ).lower()
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
                        f_idx = int(qs["frame"][0])
                        vm.reload(frame_index=f_idx)
                        vm.pipeline_runner.apply_active_solve_overlay(
                            vm.payload, frame_index=f_idx
                        )
                    except Exception:
                        pass
                else:
                    try:
                        vm.pipeline_runner.apply_active_solve_overlay(vm.payload)
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

            if req_path == "/api/artifacts/download":
                ds_id = str(qs.get("dataset_id", [""])[0]).strip()
                fname = str(qs.get("name", [""])[0]).strip()
                try:
                    local_fp, dl_name = (
                        vm.pipeline_runner.resolve_artifact_for_download(
                            dataset_id=ds_id,
                            name=fname,
                            current_h5=vm.replay_h5,
                        )
                    )
                    with open(local_fp, "rb") as fh:
                        raw_bytes = fh.read()
                    safe_dl = dl_name.replace('"', "")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="{safe_dl}"',
                    )
                    self.send_header("Content-Length", str(len(raw_bytes)))
                    self.end_headers()
                    self.wfile.write(raw_bytes)
                except Exception as exc:
                    err_body = str(exc).encode("utf-8")
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(err_body)))
                    self.end_headers()
                    self.wfile.write(err_body)
                return

            if req_path == "/api/pipeline/oncat_catalog":
                cat_doc = vm.pipeline_runner.browse_oncat_catalog(
                    {
                        "facility": str(qs.get("facility", ["HFIR"])[0]).strip(),
                        "instrument": str(qs.get("instrument", ["CG4D"])[0]).strip(),
                        "ipts": str(qs.get("ipts", [""])[0]).strip(),
                    }
                )
                body = json.dumps(cat_doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/oncat_auth":
                auth_doc = vm.pipeline_runner.get_oncat_auth_status()
                body = json.dumps(auth_doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/status":
                hist_ds = vm.pipeline_runner.list_history_datasets()
                status_doc = {
                    "galaxy_connected": vm.pipeline_runner.is_connected,
                    "galaxy_url": vm.pipeline_runner.galaxy_url,
                    "history_id": vm.pipeline_runner.history_id,
                    "history_datasets": hist_ds,
                    "step_artifacts": vm.pipeline_runner.list_step_artifacts(
                        current_h5=vm.replay_h5,
                        history_datasets=hist_ds,
                    ),
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

            if req_path == "/api/pipeline/oncat_auth":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    result = vm.pipeline_runner.handle_oncat_auth(req_json)
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
                    params.setdefault(
                        "instrument", vm.payload.get("instrument", "CG4D")
                    )
                    params.setdefault(
                        "space_group", vm.payload.get("space_group", "P 1")
                    )
                    if all(
                        k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")
                    ):
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
                        params.setdefault(
                            "wavelength_range", vm.payload.get("wavelength_range")
                        )
                    cur_imgs = (
                        vm.payload_kwargs.get("images_h5")
                        or vm.payload.get("images_h5")
                        or vm.replay_h5
                    )
                    if cur_imgs:
                        params.setdefault("images_file", str(cur_imgs))
                    if vm.replay_h5:
                        params.setdefault("peaks_file", str(vm.replay_h5))

                    def _on_complete(new_h5_path: str) -> None:
                        if (
                            stage_id == "reduce"
                            or "merged" in os.path.basename(new_h5_path).lower()
                        ):
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
                    if len(saved_paths) == 1 and not saved_paths[0].lower().endswith(
                        ".nxs.h5"
                    ):
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
