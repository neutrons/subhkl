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

import contextlib
import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

import numpy as np

from subhkl import get_container_image
from subhkl.viz.logo_svg import SUBHKL_FAVICON_DATA_URI, SUBHKL_FAVICON_SVG
from subhkl.viz.rainbow_zones import (
    build_rainbow_zone_payload,
    render_rainbow_html,
)

_NOVA_RECOVERABLE_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    ImportError,
    ArithmeticError,
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

    STAGE_TOOLS: ClassVar[dict[str, list[str]]] = {
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
    NEUTRONS_LOAD_DATA_ONCAT_INSTRUMENTS: ClassVar[dict[str, set[str]]] = {
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
        m_loc = _re.search(r"/(SNS|HFIR)/([^/]+)/IPTS-([^/]+)/", p_str, _re.IGNORECASE)
        m_run = _re.search(r"_([0-9]+)\.nxs", os.path.basename(p_str), _re.IGNORECASE)
        if not m_loc:
            return ""
        fac_u = m_loc.group(1).upper()
        inst_u = m_loc.group(2).upper()
        ipts_num = m_loc.group(3)
        run_num = int(m_run.group(1)) if m_run else None
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
        target_frame = int(
            frame_index if frame_index is not None else (self.current_frame_index or 0)
        )
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
        if sp.get("wavelength_range") and len(sp["wavelength_range"]) >= 2:
            payload["wavelength_range"] = [
                float(sp["wavelength_range"][0]),
                float(sp["wavelength_range"][1]),
            ]
        if sp.get("d_min"):
            payload["d_min"] = float(sp["d_min"])
        if (
            partial_h5
            and os.path.isfile(partial_h5)
            and os.path.getsize(partial_h5) > 0
        ):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                    cell=None,
                    wavelength_range=tuple(
                        sp.get("wavelength_range")
                        or payload.get("wavelength_range")
                        or (2.0, 4.5)
                    ),
                    d_min=(
                        float(sp["d_min"])
                        if sp.get("d_min")
                        else (float(payload["d_min"]) if payload.get("d_min") else None)
                    ),
                    max_zone_index=3,
                    max_zones=60,
                    include_panels=True,
                    include_panel_images=False,
                    include_peaks=True,
                    n_arc=720,
                    frame_index=target_frame,
                )
                if p_blind.get("zones"):
                    payload["zones"] = p_blind["zones"]
                    if p_blind.get("peaks"):
                        payload["peaks"] = p_blind["peaks"]
                    if p_blind.get("sample_peaks"):
                        payload["sample_peaks"] = p_blind["sample_peaks"]
                    if p_blind.get("panels") and payload.get("panels"):
                        p_by_b = {
                            int(p["bank"]): p for p in p_blind["panels"] if "bank" in p
                        }
                        for cur_p in payload["panels"]:
                            upd_p = p_by_b.get(int(cur_p.get("bank", -1)))
                            if upd_p:
                                for k_g in (
                                    "az_min",
                                    "az_max",
                                    "y_min",
                                    "y_max",
                                    "el_min",
                                    "el_max",
                                    "corners",
                                    "corners_sph",
                                    "outline",
                                    "outline_sph",
                                    "mesh",
                                    "mesh_sph",
                                ):
                                    if k_g in upd_p:
                                        cur_p[k_g] = upd_p[k_g]
                    if p_blind.get("cell"):
                        payload["cell"] = p_blind["cell"]
                    if p_blind.get("ub_matrix"):
                        payload["ub_matrix"] = p_blind["ub_matrix"]
                    if p_blind.get("goniometer_R"):
                        payload["goniometer_R"] = p_blind["goniometer_R"]
                    if p_blind.get("ladder_steps"):
                        payload["ladder_steps"] = p_blind["ladder_steps"]
                    payload["active_rung_idx"] = int(aj.get("ladder_rung_idx") or 1)
                    prev_ov = aj.get("partial_overlay")
                    next_seq = (
                        int(prev_ov.get("seq", 0)) + 1
                        if isinstance(prev_ov, dict)
                        and prev_ov.get("frame_index") != target_frame
                        else (
                            int(prev_ov.get("seq", 1))
                            if isinstance(prev_ov, dict)
                            else 1
                        )
                    )
                    aj["partial_overlay"] = {
                        "seq": next_seq,
                        "frame_index": target_frame,
                        "zones": p_blind.get("zones", []),
                        "peaks": p_blind.get("peaks", []),
                        "sample_peaks": p_blind.get("sample_peaks", []),
                        "panels": p_blind.get("panels", []),
                        "cell": p_blind.get("cell"),
                        "ub_matrix": p_blind.get("ub_matrix"),
                        "goniometer_R": p_blind.get("goniometer_R"),
                        "ladder_steps": p_blind.get("ladder_steps", []),
                    }
                    return payload
        ov = aj.get("partial_overlay")
        if (
            isinstance(ov, dict)
            and ov.get("zones")
            and not payload.get("zones")
            and (
                ov.get("frame_index") is None or int(ov["frame_index"]) == target_frame
            )
        ):
            payload["zones"] = ov["zones"]
            if ov.get("peaks"):
                payload["peaks"] = ov["peaks"]
            if ov.get("sample_peaks"):
                payload["sample_peaks"] = ov["sample_peaks"]
            if ov.get("ladder_steps"):
                payload["ladder_steps"] = ov["ladder_steps"]
            payload["active_rung_idx"] = int(aj.get("ladder_rung_idx") or 1)
        return payload

    def sync_frame_index(self, frame_index: int | None) -> None:
        """Update ``current_frame_index`` and ensure any active live solve overlay matches."""
        if frame_index is None:
            return
        self.current_frame_index = int(frame_index)
        aj = self.active_job
        if not isinstance(aj, dict):
            return
        if aj.get("stage_id") != "solve" or aj.get("state") not in (
            "queued",
            "running",
        ):
            return
        ov = aj.get("partial_overlay")
        if isinstance(ov, dict) and ov.get("frame_index") != self.current_frame_index:
            self.apply_active_solve_overlay({}, frame_index=self.current_frame_index)

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
        except _NOVA_RECOVERABLE_ERRORS:
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
            if clean_ln.lstrip().startswith("[subhkl-live-state]"):
                continue
            m_cur = tqdm_re.match(clean_ln)
            if m_cur:
                while collapsed and collapsed[-1] == "":
                    collapsed.pop()
                if collapsed:
                    m_prev = tqdm_re.match(collapsed[-1])
                    if (
                        m_prev
                        and m_prev.group("pfx").strip() == m_cur.group("pfx").strip()
                    ):
                        collapsed[-1] = clean_ln
                        continue
            collapsed.append(clean_ln)
        while (
            len(collapsed) >= 2 and collapsed[-1] == "" and tqdm_re.match(collapsed[-2])
        ):
            collapsed.pop()
        return "\n".join(collapsed)

    @staticmethod
    def _materialize_live_state_h5(b64_token: str, dest_h5_path: str) -> bool:
        """Decode a ``[subhkl-live-state] <b64>`` stdout snapshot into an HDF5 file for live zone rendering."""
        import base64
        import zlib

        import h5py
        import numpy as np
        from scipy.spatial.transform import Rotation

        from subhkl.config import beamlines
        from subhkl.instrument.detector import Detector

        raw_json = zlib.decompress(base64.b64decode(b64_token.strip())).decode("utf-8")
        doc = json.loads(raw_json)
        if not isinstance(doc, dict) or "U" not in doc or "B" not in doc:
            return False
        inst = str(doc.get("instrument") or "CG4D")
        sg = str(doc.get("space_group") or "P 1")
        wl = doc.get("wavelength") or [2.8, 4.5]
        cell_vals = [
            float(x) for x in (doc.get("cell") or [68.7, 69.8, 156.9, 90.0, 90.0, 90.0])
        ]
        tmp_p = dest_h5_path + ".tmp.h5"
        try:
            with h5py.File(tmp_p, "w") as hf:
                hf.attrs["instrument"] = inst
                hf["instrument/name"] = inst
                hf["instrument/wavelength"] = np.asarray(wl[:2], dtype=float)
                hf["sample/cell"] = np.asarray(cell_vals[:6], dtype=float)
                for k_c, v_c in zip(
                    ("a", "b", "c", "alpha", "beta", "gamma"), cell_vals[:6]
                ):
                    hf[f"sample/{k_c}"] = float(v_c)
                hf["sample/space_group"] = sg
                hf["sample/U"] = np.asarray(doc["U"], dtype=float)
                hf["sample/B"] = np.asarray(doc["B"], dtype=float)
                if doc.get("R") is not None:
                    hf["goniometer/R"] = np.asarray(doc["R"], dtype=float)
                banks_map = doc.get("banks")
                if isinstance(banks_map, dict) and banks_map:
                    cal_g = hf.create_group("detector_calibration")
                    for b_id, b_info in banks_map.items():
                        bg = cal_g.create_group(f"bank_{b_id}")
                        bg["center"] = np.asarray(b_info["center"], dtype=float)
                        bg["uhat"] = np.asarray(b_info["uhat"], dtype=float)
                        bg["vhat"] = np.asarray(b_info["vhat"], dtype=float)
                        bg["width"] = float(b_info["width"])
                        bg["height"] = float(b_info["height"])
                elif (
                    "radial_scale" in doc
                    or "global_rot_vec" in doc
                    or "global_trans_m" in doc
                ) and inst in beamlines:
                    r_sc = float(doc.get("radial_scale", 0.0))
                    r_vec = np.asarray(
                        doc.get("global_rot_vec", [0.0, 0.0, 0.0]), dtype=float
                    )
                    t_vec = np.asarray(
                        doc.get("global_trans_m", [0.0, 0.0, 0.0]), dtype=float
                    )
                    rot_m = Rotation.from_rotvec(r_vec).as_matrix()
                    cal_g = hf.create_group("detector_calibration")
                    for b_k, b_cfg in beamlines[inst].items():
                        try:
                            b_id_int = int(b_k)
                        except (ValueError, TypeError):
                            continue
                        det_b = Detector(dict(b_cfg))
                        mid_b = (
                            det_b.center
                            + 0.5 * det_b.width * det_b.uhat
                            + 0.5 * det_b.height * det_b.vhat
                        )
                        mid_cal = mid_b * (1.0 + r_sc) @ rot_m.T + t_vec
                        uh_cal = det_b.uhat @ rot_m.T
                        vh_cal = det_b.vhat @ rot_m.T
                        cen_cal = (
                            mid_cal
                            - 0.5 * det_b.width * uh_cal
                            - 0.5 * det_b.height * vh_cal
                        )
                        bg = cal_g.create_group(f"bank_{b_id_int}")
                        bg["center"] = np.asarray(cen_cal, dtype=float)
                        bg["uhat"] = np.asarray(uh_cal, dtype=float)
                        bg["vhat"] = np.asarray(vh_cal, dtype=float)
                        bg["width"] = float(det_b.width)
                        bg["height"] = float(det_b.height)
                if doc.get("kind") == "stages":
                    rep = hf.create_group("scan_refinement")
                    rep["U"] = np.asarray(doc["U"], dtype=float)
                    rep["B"] = np.asarray(doc["B"], dtype=float)
                    rep["band_wavelength"] = np.asarray(
                        doc.get("band_wavelength") or wl[:2], dtype=float
                    )
                    rep.attrs["stages"] = json.dumps(doc.get("stages") or [])
                    rep.attrs["status"] = "running"
                    rep.attrs["n_zones"] = int(doc.get("n_zones", 120))
                    rep.attrs["d_min"] = float(doc.get("d_min", 1.5))
                else:
                    sg_grp = hf.create_group("search")
                    sg_grp.attrs["instrument"] = inst
                    sg_grp.attrs["d_min"] = float(doc.get("d_min", 1.5))
                    sg_grp.attrs["ladder_stage"] = str(doc.get("stage") or "blind")
                    sg_grp.attrs["ladder_zones"] = int(doc.get("n_zones", 60))
            os.replace(tmp_p, dest_h5_path)
            return True
        finally:
            if os.path.exists(tmp_p):
                with contextlib.suppress(OSError):
                    os.unlink(tmp_p)

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
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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

        if self.galaxy_url and self.galaxy_api_key and not self.history_id:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                histories = self._api_request("GET", "/api/histories")
                if isinstance(histories, list) and histories:
                    self.history_id = histories[0]["id"]

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

    def touch_dataset(self, ds_id: str | None) -> None:
        """Record that a Galaxy dataset ID was just accessed, created, or linked."""
        if not ds_id:
            return
        touch_map: dict[str, float] = getattr(self, "_dataset_touch_times", {})
        touch_map[str(ds_id)] = time.time()
        self._dataset_touch_times = touch_map

    @staticmethod
    def _normalize_cell_meta(
        cell: Any,
        space_group: Any = "",
        instrument: Any = "",
    ) -> dict[str, Any] | None:
        if not isinstance(cell, (list, tuple)) or len(cell) < 6:
            return None
        try:
            vals = [round(float(x), 5) for x in cell[:6]]
        except (TypeError, ValueError):
            return None
        if any(not np.isfinite(v) or v <= 0.0 for v in vals):
            return None
        if any(v >= 180.0 for v in vals[3:6]):
            return None
        sg_str = str(space_group or "").strip()
        inst_str = str(instrument or "").strip()
        res: dict[str, Any] = {
            "cell": vals,
            "space_group": sg_str,
        }
        if inst_str:
            res["instrument"] = inst_str
        return res

    @classmethod
    def _parse_dataset_annotation_meta(
        cls, annotation: Any, tags: Any = None
    ) -> dict[str, Any] | None:
        if isinstance(annotation, dict):
            if "annotation" in annotation or "tags" in annotation:
                if tags is None:
                    tags = annotation.get("tags")
                annotation = annotation.get("annotation")
            else:
                cand = cls._normalize_cell_meta(
                    annotation.get("subhkl_cell") or annotation.get("cell"),
                    annotation.get("space_group"),
                    annotation.get("instrument"),
                )
                if cand:
                    return cand
        if isinstance(annotation, str):
            s = annotation.strip()
            if s.startswith("{") and s.endswith("}"):
                with contextlib.suppress(ValueError, TypeError):
                    doc = json.loads(s)
                    if isinstance(doc, dict):
                        cand = cls._normalize_cell_meta(
                            doc.get("subhkl_cell") or doc.get("cell"),
                            doc.get("space_group"),
                            doc.get("instrument"),
                        )
                        if cand:
                            return cand
        if isinstance(tags, list):
            for t in tags:
                ts = str(t or "").strip()
                if ts.startswith("subhkl_cell:"):
                    parts = ts[len("subhkl_cell:") :].split(":")
                    if len(parts) >= 1:
                        with contextlib.suppress(ValueError, TypeError):
                            c_vals = [
                                float(x) for x in parts[0].split(",") if x.strip()
                            ]
                            sg_tag = (
                                parts[1].replace("_", " ") if len(parts) > 1 else ""
                            )
                            inst_tag = parts[2] if len(parts) > 2 else ""
                            cand = cls._normalize_cell_meta(c_vals, sg_tag, inst_tag)
                            if cand:
                                return cand
        return None

    @classmethod
    def _extract_cell_meta_from_local_file(
        cls, path: str | None
    ) -> dict[str, Any] | None:
        if not path or not os.path.isfile(str(path)):
            return None
        p_str = str(path)
        low = p_str.lower()
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            if low.endswith(".mtz"):
                import gemmi

                mtz = gemmi.read_mtz_file(p_str)
                c_params = list(mtz.cell.parameters)
                sg_hm = mtz.spacegroup.hm if mtz.spacegroup else ""
                return cls._normalize_cell_meta(c_params, sg_hm, "")
            import h5py

            with h5py.File(p_str, "r") as hf:
                raw_cell: list[float] | None = None
                if all(
                    f"sample/{k}" in hf
                    for k in ("a", "b", "c", "alpha", "beta", "gamma")
                ):
                    raw_cell = [
                        float(np.asarray(hf[f"sample/{k}"][()]).flat[0])
                        for k in ("a", "b", "c", "alpha", "beta", "gamma")
                    ]
                elif "sample/B" in hf:
                    from subhkl.viz.rainbow_zones import cell_params_from_B

                    B = np.asarray(hf["sample/B"][()], dtype=np.float64)
                    if B.shape == (3, 3):
                        raw_cell = list(cell_params_from_B(B))
                if not raw_cell:
                    return None
                sg_val = ""
                for sk in ("space_group", "sample/space_group"):
                    if sk in hf:
                        v = hf[sk][()]
                        if hasattr(v, "flat") and v.size > 0:
                            v = v.flat[0]
                        sg_val = (
                            v.decode("utf-8", "replace")
                            if isinstance(v, (bytes, np.bytes_))
                            else str(v)
                        ).strip()
                        if sg_val:
                            break
                if not sg_val and "space_group" in hf.attrs:
                    av = hf.attrs["space_group"]
                    sg_val = (
                        av.decode("utf-8", "replace")
                        if isinstance(av, (bytes, np.bytes_))
                        else str(av)
                    ).strip()
                inst_val = ""
                if "instrument" in hf.attrs:
                    iv = hf.attrs["instrument"]
                    inst_val = (
                        iv.decode("utf-8", "replace")
                        if isinstance(iv, (bytes, np.bytes_))
                        else str(iv)
                    ).strip()
                elif "instrument/name" in hf:
                    iv = hf["instrument/name"][()]
                    if hasattr(iv, "flat") and iv.size > 0:
                        iv = iv.flat[0]
                    inst_val = (
                        iv.decode("utf-8", "replace")
                        if isinstance(iv, (bytes, np.bytes_))
                        else str(iv)
                    ).strip()
                elif "instrument" in hf and isinstance(hf["instrument"], h5py.Dataset):
                    iv = hf["instrument"][()]
                    if hasattr(iv, "flat") and iv.size > 0:
                        iv = iv.flat[0]
                    inst_val = (
                        iv.decode("utf-8", "replace")
                        if isinstance(iv, (bytes, np.bytes_))
                        else str(iv)
                    ).strip()
                return cls._normalize_cell_meta(raw_cell, sg_val, inst_val)
        return None

    def attach_dataset_cell_metadata(
        self,
        ds_id: str | None,
        cell: Any,
        space_group: str | None = "",
        instrument: str | None = "",
        *,
        sync_remote: bool = True,
    ) -> dict[str, Any] | None:
        """Attach unit cell + space group metadata to a Galaxy dataset ID (in-memory + Galaxy HDA annotation)."""
        import re as _re

        if not ds_id:
            return None
        meta = self._normalize_cell_meta(cell, space_group, instrument)
        if not meta:
            return None
        ds_key = str(ds_id).strip()
        cell_map: dict[str, dict[str, Any]] = getattr(self, "_dataset_cell_meta", {})
        prev = cell_map.get(ds_key)
        if prev and not meta.get("space_group") and prev.get("space_group"):
            meta["space_group"] = prev["space_group"]
        if prev and not meta.get("instrument") and prev.get("instrument"):
            meta["instrument"] = prev["instrument"]
        cell_map[ds_key] = meta
        self._dataset_cell_meta = cell_map

        if (
            sync_remote
            and self.is_connected
            and self.history_id
            and _re.match(r"^[0-9a-fA-F]{8,}$", ds_key)
        ):
            synced_map: dict[str, dict[str, Any]] = getattr(
                self, "_dataset_remote_synced_meta", {}
            )
            if synced_map.get(ds_key) != meta:
                synced_map[ds_key] = dict(meta)
                self._dataset_remote_synced_meta = synced_map
                ann_doc: dict[str, Any] = {
                    "subhkl_cell": meta["cell"],
                    "space_group": meta.get("space_group") or "",
                }
                if meta.get("instrument"):
                    ann_doc["instrument"] = meta["instrument"]
                ann_str = json.dumps(ann_doc, separators=(",", ":"))
                hist_id = str(self.history_id)

                def _push_ann() -> None:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        self._api_request(
                            "PUT",
                            f"/api/histories/{hist_id}/contents/{ds_key}",
                            payload={"annotation": ann_str},
                            timeout=5.0,
                        )

                threading.Thread(target=_push_ann, daemon=True).start()
        return meta

    def attach_dataset_cell_metadata_from_file(
        self,
        ds_id: str | None,
        local_path: str | None,
        fallback_params: dict[str, Any] | None = None,
        *,
        sync_remote: bool = True,
    ) -> dict[str, Any] | None:
        if not ds_id:
            return None
        meta = self._extract_cell_meta_from_local_file(local_path)
        params = fallback_params or {}
        if not meta and params:
            raw_cell = params.get("cell")
            if not (isinstance(raw_cell, (list, tuple)) and len(raw_cell) >= 6) and all(
                k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")
            ):
                with contextlib.suppress(TypeError, ValueError):
                    raw_cell = [
                        float(params["a"]),
                        float(params["b"]),
                        float(params["c"]),
                        float(params["alpha"]),
                        float(params["beta"]),
                        float(params["gamma"]),
                    ]
            meta = self._normalize_cell_meta(
                raw_cell,
                params.get("space_group"),
                params.get("instrument"),
            )
        elif meta and params:
            if not meta.get("space_group") and params.get("space_group"):
                meta["space_group"] = str(params["space_group"]).strip()
            if not meta.get("instrument") and params.get("instrument"):
                meta["instrument"] = str(params["instrument"]).strip()
        if not meta:
            return None
        return self.attach_dataset_cell_metadata(
            ds_id,
            meta["cell"],
            meta.get("space_group"),
            meta.get("instrument"),
            sync_remote=sync_remote,
        )

    def _local_cached_files_by_ds_id(self) -> dict[str, str]:
        """Return a fast mapping of ds_id -> cached local file path in /tmp."""
        import re as _re

        tmp_dir = tempfile.gettempdir()
        with contextlib.suppress(OSError):
            mtime = os.stat(tmp_dir).st_mtime_ns
            cached_pair = getattr(self, "_tmp_gx_index_cache", None)
            if (
                isinstance(cached_pair, tuple)
                and len(cached_pair) == 2
                and cached_pair[0] == mtime
            ):
                return cached_pair[1]
            mapping: dict[str, str] = {}
            for fn in os.listdir(tmp_dir):
                if not fn.startswith("subhkl_gx_"):
                    continue
                m = _re.match(r"^subhkl_gx_([0-9a-fA-F]{8,})_", fn)
                if m:
                    ds_id = m.group(1)
                    if ds_id not in mapping or fn.endswith((".h5", ".mtz")):
                        mapping[ds_id] = os.path.join(tmp_dir, fn)
            self._tmp_gx_index_cache = (mtime, mapping)
            return mapping
        return {}

    @staticmethod
    def _parse_galaxy_timestamp(ts_str: str | None) -> float:
        if not ts_str:
            return 0.0
        s = str(ts_str).strip()
        if not s:
            return 0.0
        with contextlib.suppress(ValueError, TypeError):
            from datetime import datetime

            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return float(dt.timestamp())
        return 0.0

    def list_history_datasets(
        self, limit: int | None = None, *, use_cached: bool = False
    ) -> list[dict[str, Any]]:
        """Return active datasets in the current Galaxy history, sorted most-recently-touched first."""
        if not self.is_connected:
            return []
        if not self.history_id:
            self._auto_discover_local_galaxy()
        if not self.history_id:
            return []
        cached_list = getattr(self, "_cached_history_datasets", None)
        if use_cached and isinstance(cached_list, list) and cached_list:
            return (
                cached_list[:limit]
                if isinstance(limit, int) and limit > 0
                else list(cached_list)
            )
        try:
            items = self._api_request(
                "GET",
                (
                    f"/api/histories/{self.history_id}/contents"
                    "?v=dev&q=deleted&qv=false"
                    "&keys=id,name,hid,state,extension,deleted,history_content_type,update_time,create_time,tags,annotation"
                ),
            )
            if not isinstance(items, list):
                return []
            from subhkl.viz.rainbow_zones import infer_dataset_extension

            touch_map: dict[str, float] = getattr(self, "_dataset_touch_times", {})
            cell_map: dict[str, dict[str, Any]] = getattr(
                self, "_dataset_cell_meta", {}
            )
            probed_local: set[str] = getattr(self, "_probed_local_cell_ds_ids", set())
            local_by_id = self._local_cached_files_by_ds_id()
            synced_map: dict[str, dict[str, Any]] = getattr(
                self, "_dataset_remote_synced_meta", {}
            )
            out: list[dict[str, Any]] = []
            for it in items:
                if (
                    not isinstance(it, dict)
                    or it.get("history_content_type") != "dataset"
                    or it.get("deleted")
                ):
                    continue
                ds_id = str(it.get("id") or "")
                ut_str = str(it.get("update_time") or it.get("create_time") or "")
                ut_epoch = self._parse_galaxy_timestamp(ut_str)
                touched_at = max(ut_epoch, float(touch_map.get(ds_id, 0.0)))
                ext_norm = infer_dataset_extension(it.get("name"), it.get("extension"))
                cell_meta = self._parse_dataset_annotation_meta(
                    it.get("annotation"), it.get("tags")
                )
                if cell_meta and ds_id:
                    cell_map[ds_id] = cell_meta
                    synced_map[ds_id] = dict(cell_meta)
                elif ds_id in cell_map:
                    cell_meta = cell_map[ds_id]
                elif ds_id and ds_id in local_by_id and ds_id not in probed_local:
                    probed_local.add(ds_id)
                    cell_meta = self.attach_dataset_cell_metadata_from_file(
                        ds_id, local_by_id[ds_id], sync_remote=True
                    )
                entry: dict[str, Any] = {
                    "id": it.get("id"),
                    "hid": it.get("hid"),
                    "name": it.get("name"),
                    "state": it.get("state"),
                    "extension": ext_norm,
                    "update_time": ut_str,
                    "touched_at": round(touched_at, 3),
                }
                if cell_meta:
                    entry["cell"] = cell_meta["cell"]
                    entry["space_group"] = cell_meta.get("space_group") or ""
                    if cell_meta.get("instrument"):
                        entry["instrument"] = cell_meta["instrument"]
                out.append(entry)
            self._dataset_cell_meta = cell_map
            self._probed_local_cell_ds_ids = probed_local
            self._dataset_remote_synced_meta = synced_map
            out.sort(
                key=lambda d: (
                    float(d.get("touched_at") or 0.0),
                    int(d.get("hid") or 0),
                ),
                reverse=True,
            )
            self._cached_history_datasets = list(out)
            if isinstance(limit, int) and limit > 0:
                return out[:limit]
            return out
        except _NOVA_RECOVERABLE_ERRORS:
            return []

    def resolve_dataset_to_local_h5(
        self, ds_id: str, fallback_dir: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a Galaxy dataset ID to a local HDF5 path (via local file_name, cached ds_id, or /display download)."""
        import glob as _glob

        info = self._api_request("GET", f"/api/datasets/{ds_id}")
        if not isinstance(info, dict):
            raise TypeError(f"Dataset {ds_id} not found on {self.galaxy_url}")
        name = str(info.get("name") or f"{ds_id}.h5")
        state = str(info.get("state") or "ok")
        if state != "ok":
            hid = info.get("hid")
            lbl = f"#{hid} '{name}'" if hid else f"'{name}'"
            raise ValueError(
                f"Cannot ingest Galaxy dataset {lbl}: dataset state is '{state}' (must be 'ok')."
            )
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})

        def _remember(path: str) -> str:
            known_map[os.path.abspath(path)] = ds_id
            known_map[os.path.realpath(path)] = ds_id
            self._dataset_id_by_path = known_map
            self.touch_dataset(ds_id)
            self.attach_dataset_cell_metadata_from_file(ds_id, path, sync_remote=True)
            return path

        file_name = str(info.get("file_name") or "")
        if file_name and os.path.isfile(file_name) and os.path.getsize(file_name) > 0:
            return _remember(file_name), info

        # 1. Check cached dataset file in /tmp by exact ds_id first
        cached_matches = sorted(
            _glob.glob(os.path.join(tempfile.gettempdir(), f"subhkl_gx_{ds_id}_*"))
        )
        for cm in cached_matches:
            if os.path.isfile(cm) and os.path.getsize(cm) > 0:
                return _remember(cm), info

        # 2. If dataset name includes a relative folder (e.g. 'betalac-7/merged.h5'), check /home/xvg/subhkl-examples/
        if "/" in name:
            if os.path.isfile(name) and os.path.getsize(name) > 0:
                return _remember(name), info
            ex_cand = os.path.join("/home/xvg/subhkl-examples", name.lstrip("/"))
            if os.path.isfile(ex_cand) and os.path.getsize(ex_cand) > 0:
                return _remember(ex_cand), info

        # 3. Only fall back to fallback_dir if file_size matches (prevents generic 'scan.h5'/'merged.h5' collisions)
        expected_size = int(info.get("file_size") or 0)
        if fallback_dir and os.path.isdir(fallback_dir):
            cand = os.path.join(fallback_dir, os.path.basename(name))
            if (
                os.path.isfile(cand)
                and os.path.getsize(cand) > 0
                and (
                    (expected_size > 0 and os.path.getsize(cand) == expected_size)
                    or (expected_size == 0 and not self.is_connected)
                )
            ):
                return _remember(cand), info

        safe_name = "".join(
            ch if (ch.isalnum() or ch in "._-") else "_"
            for ch in os.path.basename(name)
        )
        if not safe_name.endswith((".h5", ".hdf5", ".mtz", ".json", ".nxs")):
            safe_name += ".h5"
        tmp_dest = os.path.join(tempfile.gettempdir(), f"subhkl_gx_{ds_id}_{safe_name}")
        if os.path.isfile(tmp_dest) and os.path.getsize(tmp_dest) > 0:
            return _remember(tmp_dest), info
        data = self._api_request(
            "GET",
            f"/api/datasets/{ds_id}/display",
            raw=True,
            timeout=180.0,
        )
        with open(tmp_dest, "wb") as fh:
            fh.write(data)
        return _remember(tmp_dest), info

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

    STAGE_ORDER: ClassVar[tuple[str, ...]] = (
        "reduce",
        "solve",
        "integrate",
        "merge",
    )

    @classmethod
    def _stage_for_kind_or_stage(cls, val: str | None) -> str | None:
        k = str(val or "").strip().lower()
        if k in cls.STAGE_ORDER:
            return k
        if k in ("merged", "raw_nexus"):
            return "reduce"
        if k in ("solve", "finder"):
            return "solve"
        if k in ("integrator", "predictor"):
            return "integrate"
        if k == "merged_careless":
            return "merge"
        return None

    def invalidate_subsequent_stages(
        self,
        current_stage_or_kind: str | None,
        work_dir: str | None = None,
    ) -> None:
        """Invalidate artifacts and cached state for pipeline stages after ``current_stage_or_kind``."""
        mapped = self._stage_for_kind_or_stage(current_stage_or_kind)
        self._max_valid_stage = mapped
        if not mapped or mapped == "merge":
            return

        wd = work_dir or self._active_work_dir()
        if not wd or not os.path.isdir(wd):
            return

        is_tmp_wd = os.path.abspath(wd).startswith(
            os.path.abspath(tempfile.gettempdir())
        )
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})

        to_remove: list[str] = [
            "careless_metrics.json",
            "merged.mtz",
            "predictions.mtz",
            "xval.mtz",
            "careless_0.mtz",
            "careless_predictions_0.mtz",
            "careless_xval_0.mtz",
        ]
        if is_tmp_wd and mapped in ("reduce", "solve"):
            to_remove.extend(
                [
                    "predictor.h5",
                    "peak_predictor.h5",
                    "integrated.h5",
                    "integrator.h5",
                    "reflections.mtz",
                ]
            )
        if is_tmp_wd and mapped == "solve":
            to_remove.extend(
                [
                    "solve.blind.h5",
                    "solve.stages.h5",
                ]
            )
        if is_tmp_wd and mapped == "reduce":
            to_remove.extend(
                [
                    "solve.h5",
                    "peaks.h5",
                    "indexer.h5",
                    "solve.blind.h5",
                    "solve.stages.h5",
                ]
            )

        for fn in to_remove:
            fp = os.path.join(wd, fn)
            for p_key in (os.path.abspath(fp), os.path.realpath(fp)):
                known_map.pop(p_key, None)
            if os.path.isfile(fp):
                with contextlib.suppress(OSError):
                    os.remove(fp)

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
            if not st or base_name in stages[st]:
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
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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

        max_stage = getattr(self, "_max_valid_stage", None)
        if not max_stage and current_h5:
            max_stage = self._classify_artifact_stage(os.path.basename(str(current_h5)))
            if (
                max_stage == "integrate"
                and work_dir
                and os.path.isfile(os.path.join(work_dir, "careless_metrics.json"))
            ):
                max_stage = "merge"
        if max_stage in self.STAGE_ORDER:
            max_idx = self.STAGE_ORDER.index(max_stage)
            for st_k in self.STAGE_ORDER[max_idx + 1 :]:
                stages[st_k] = {}

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
            for ds in self.list_history_datasets():
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
        """Ensure a merged HDF5 file inside /tmp or Pulsar staging uses a dedicated /tmp/subhkl_work_<tag> directory."""
        import glob as _glob
        import re as _re
        import shutil as _shutil

        abs_m = os.path.abspath(merged_path)
        parent_d = os.path.dirname(abs_m)
        orig_bn = os.path.basename(abs_m)
        tmp_root = os.path.abspath(tempfile.gettempdir())
        is_staging_dat = "/staging" in abs_m or (
            orig_bn.startswith("dataset_") and orig_bn.endswith(".dat")
        )
        if parent_d != tmp_root and not is_staging_dat:
            return parent_d, abs_m

        m_gx = _re.match(r"^subhkl_gx_([0-9a-fA-F]+)_(.+)$", orig_bn)
        gx_ds_id = m_gx.group(1) if m_gx else None
        clean_bn = m_gx.group(2) if m_gx else orig_bn

        tag = ""
        is_images_h5 = False
        if os.path.isfile(abs_m):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py

                with h5py.File(abs_m, "r") as hf:
                    is_images_h5 = "images" in hf
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
        if not tag:
            bn = _re.sub(
                r"(_merged|_solve|_integrated|_predictor)?\.(h5|dat)$",
                "",
                clean_bn,
                flags=_re.IGNORECASE,
            )
            tag = bn or "dataset"
        safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
        work_d = os.path.join(tmp_root, f"subhkl_work_{safe_tag}")
        os.makedirs(work_d, exist_ok=True)
        dest_name = (
            clean_bn
            if clean_bn.lower().endswith("_merged.h5")
            else f"{safe_tag}_merged.h5"
        )
        dest_m = os.path.join(work_d, dest_name)
        legacy_m = os.path.join(work_d, "merged.h5")
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})

        def _has_h5_images(path_cand: str) -> bool:
            if not os.path.isfile(path_cand):
                return False
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py

                with h5py.File(path_cand, "r") as hf_c:
                    return "images" in hf_c
            return False

        if os.path.isfile(abs_m) and is_images_h5:
            src_sz = os.path.getsize(abs_m)
            if (
                not os.path.isfile(dest_m)
                or os.path.getsize(dest_m) != src_sz
                or not _has_h5_images(dest_m)
            ):
                try:
                    _shutil.copy2(abs_m, dest_m)
                except OSError:
                    return work_d, abs_m
            if dest_m != legacy_m and (
                not os.path.isfile(legacy_m)
                or os.path.getsize(legacy_m) != src_sz
                or not _has_h5_images(legacy_m)
            ):
                with contextlib.suppress(OSError):
                    _shutil.copy2(abs_m, legacy_m)
            final_m = dest_m if os.path.isfile(dest_m) else abs_m
            mapped_id = (
                known_map.get(abs_m)
                or known_map.get(os.path.realpath(abs_m))
                or gx_ds_id
            )
            if not mapped_id:
                for gx_f in _glob.glob(
                    os.path.join(tmp_root, f"subhkl_gx_*_{safe_tag}_merged.h5")
                ):
                    if not _has_h5_images(gx_f):
                        continue
                    m_cand = _re.match(
                        r"^subhkl_gx_([0-9a-fA-F]+)_", os.path.basename(gx_f)
                    )
                    if m_cand:
                        mapped_id = m_cand.group(1)
                        break
            if mapped_id:
                for p_alias in (abs_m, final_m, legacy_m):
                    known_map[os.path.abspath(p_alias)] = mapped_id
                    known_map[os.path.realpath(p_alias)] = mapped_id
                self._dataset_id_by_path = known_map
            return work_d, final_m

        # Non-image HDF5 (e.g. solve.h5, integrated.h5, predictor.h5):
        # Never overwrite <tag>_merged.h5 or merged.h5 with it!
        if gx_ds_id and os.path.isfile(abs_m):
            known_map[os.path.abspath(abs_m)] = gx_ds_id
            known_map[os.path.realpath(abs_m)] = gx_ds_id
            low_c = clean_bn.lower()
            stage_copy = None
            if "solve" in low_c and low_c.endswith(".h5"):
                stage_copy = os.path.join(work_d, "solve.h5")
            elif "integrat" in low_c and low_c.endswith(".h5"):
                stage_copy = os.path.join(work_d, "integrated.h5")
            elif "predict" in low_c and low_c.endswith(".h5"):
                stage_copy = os.path.join(work_d, "predictor.h5")
            if stage_copy:
                if not os.path.isfile(stage_copy) or os.path.getsize(
                    stage_copy
                ) != os.path.getsize(abs_m):
                    with contextlib.suppress(OSError):
                        _shutil.copy2(abs_m, stage_copy)
                known_map[os.path.abspath(stage_copy)] = gx_ds_id
                known_map[os.path.realpath(stage_copy)] = gx_ds_id
            self._dataset_id_by_path = known_map

        # Self-heal dest_m / legacy_m if missing or previously overwritten by a non-image HDF5
        if not _has_h5_images(dest_m):
            for bad_p in (dest_m, legacy_m):
                if os.path.isfile(bad_p) and not _has_h5_images(bad_p):
                    with contextlib.suppress(OSError):
                        os.remove(bad_p)
                known_map.pop(os.path.abspath(bad_p), None)
                known_map.pop(os.path.realpath(bad_p), None)
            for gx_f in sorted(
                _glob.glob(os.path.join(tmp_root, f"subhkl_gx_*_{safe_tag}_merged.h5")),
                key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else 0.0,
                reverse=True,
            ):
                if _has_h5_images(gx_f):
                    with contextlib.suppress(OSError):
                        _shutil.copy2(gx_f, dest_m)
                        if dest_m != legacy_m:
                            _shutil.copy2(gx_f, legacy_m)
                    m_cand = _re.match(
                        r"^subhkl_gx_([0-9a-fA-F]+)_", os.path.basename(gx_f)
                    )
                    if m_cand:
                        cid = m_cand.group(1)
                        for p_alias in (gx_f, dest_m, legacy_m):
                            known_map[os.path.abspath(p_alias)] = cid
                            known_map[os.path.realpath(p_alias)] = cid
                        self._dataset_id_by_path = known_map
                    break

        final_m = dest_m if _has_h5_images(dest_m) else abs_m
        return work_d, final_m

    def _discover_or_fetch_galaxy_images_h5(
        self, peaks_h5_path: str | None
    ) -> str | None:
        """Locate the companion merged scan HDF5 (containing ``images``) on disk or fetch it from Galaxy history."""
        if not peaks_h5_path or not os.path.isfile(str(peaks_h5_path)):
            return None
        from subhkl.viz.rainbow_zones import _discover_images_h5

        # First allow _resolve_work_dir_and_merged to self-heal any local /tmp/subhkl_gx_*_<tag>_merged.h5
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            self._resolve_work_dir_and_merged(str(peaks_h5_path))

        disc = _discover_images_h5(str(peaks_h5_path))
        if disc and os.path.isfile(str(disc)):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py

                with h5py.File(str(disc), "r") as hf_d:
                    if "images" in hf_d:
                        _, resolved_m = self._resolve_work_dir_and_merged(str(disc))
                        return resolved_m

        # Not found locally — inspect `files` in peaks_h5_path and look up <tag>_merged.h5 in Galaxy history
        tag = ""
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py

            with h5py.File(str(peaks_h5_path), "r") as hf:
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
        if not tag or not self.is_connected:
            return None
        safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
        target_names = {
            f"{safe_tag}_merged.h5".lower(),
            f"{tag}_merged.h5".lower(),
        }
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            hist_items = self.list_history_datasets(limit=200)
            for ds in hist_items:
                if ds.get("state") != "ok":
                    continue
                dname = str(ds.get("name") or "").strip().lower()
                if dname in target_names or (
                    safe_tag.lower() in dname and dname.endswith("_merged.h5")
                ):
                    ds_id = str(ds.get("id") or "").strip()
                    if ds_id:
                        local_m, _ = self.resolve_dataset_to_local_h5(ds_id)
                        _, final_m = self._resolve_work_dir_and_merged(local_m)
                        return final_m
        return None

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
        except _NOVA_RECOVERABLE_ERRORS:
            return False

    def _get_tool_version(self, tool_id: str) -> str:
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            doc = self._api_request("GET", f"/api/tools/{tool_id}")
            if isinstance(doc, dict):
                return str(doc.get("version") or "")
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
            ),
            "stderr": "",
            "ladder_rung_idx": 0 if stage_id == "solve" else 4,
            "outputs": [],
            "updated_payload_ready": False,
            "started_at": time.time(),
        }
        if stage_id == "solve":
            wl_r = params.get("wavelength_range")
            job_record["solve_params"] = {
                "instrument": str(params.get("instrument") or "CG4D"),
                "space_group": str(params.get("space_group") or "P 1"),
                "cell": params.get("cell"),
                "wavelength_range": (
                    (float(wl_r[0]), float(wl_r[1]))
                    if isinstance(wl_r, (list, tuple)) and len(wl_r) >= 2
                    else None
                ),
                "d_min": (
                    float(params["d_min"]) if params.get("d_min") is not None else None
                ),
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

    def cancel_active_job(self, stage_id: str | None = None) -> dict[str, Any]:
        """Cancel the currently running NDIP Galaxy or local container job."""
        import signal
        import subprocess

        aj = self.active_job
        if not isinstance(aj, dict):
            return {"ok": True, "status": "idle", "message": "No active job to cancel"}

        aj["cancel_requested"] = True
        prev_state = str(aj.get("state") or "")
        aj["state"] = "deleted"
        aj["stdout"] = (
            aj.get("stdout", "")
            + "\n[subhkl] Stop requested by user — cancelling job...\n"
        )[-24000:]

        cancelled_galaxy_ids: list[str] = []
        gx_ids: list[str] = list(aj.get("galaxy_job_ids") or [])
        cur_jid = str(aj.get("job_id") or "")
        if (
            cur_jid
            and not cur_jid.startswith(("nova-", "gpu-"))
            and cur_jid not in gx_ids
        ):
            gx_ids.append(cur_jid)

        for jid in gx_ids:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                self._api_request("DELETE", f"/api/jobs/{jid}")
                cancelled_galaxy_ids.append(jid)

        proc = getattr(self, "_active_proc", None)
        if proc is not None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except OSError:
                with contextlib.suppress(OSError):
                    proc.terminate()

        cname = getattr(self, "_active_container_name", None)
        if cname:
            with contextlib.suppress(
                OSError, subprocess.SubprocessError, ValueError, TypeError
            ):
                subprocess.run(
                    ["podman", "rm", "-f", str(cname)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    check=False,
                )

        return {
            "ok": True,
            "status": "cancelled",
            "stage_id": aj.get("stage_id") or stage_id,
            "previous_state": prev_state,
            "cancelled_job_id": cur_jid,
            "cancelled_galaxy_jobs": cancelled_galaxy_ids,
            "job": aj,
        }

    def _stamp_merged_metadata(self, h5_path: str, job_record: dict[str, Any]) -> None:
        if not h5_path or not os.path.isfile(h5_path):
            return
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py
            import numpy as np

            params = job_record.get("params") or {}
            inst_val = str(
                job_record.get("instrument") or params.get("instrument") or ""
            ).strip()
            sg_val = str(params.get("space_group") or "").strip()
            raw_cell = params.get("cell")
            if not (isinstance(raw_cell, (list, tuple)) and len(raw_cell) == 6) and all(
                k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")
            ):
                raw_cell = [
                    float(params["a"]),
                    float(params["b"]),
                    float(params["c"]),
                    float(params["alpha"]),
                    float(params["beta"]),
                    float(params["gamma"]),
                ]
            with h5py.File(h5_path, "r+") as hf:
                if inst_val:
                    hf.attrs["instrument"] = inst_val
                    if "instrument" not in hf:
                        hf["instrument"] = np.bytes_(inst_val)
                    elif (
                        isinstance(hf["instrument"], h5py.Group)
                        and "name" not in hf["instrument"]
                    ):
                        hf["instrument/name"] = np.bytes_(inst_val)
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

    def _stamp_careless_merge_into_h5(
        self,
        integrated_h5_path: str,
        work_dir: str,
        job_record: dict[str, Any],
    ) -> None:
        """Persist Careless merge metrics and per-observation residuals/scales into ``integrated.h5``."""
        import gemmi
        import h5py
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
                except ValueError:
                    metrics_doc = {}

        if os.path.isfile(merged_mtz_path):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                    metrics_doc.setdefault("n_merged_asu", len(hm_v))
                    metrics_doc.setdefault(
                        "mean_isigi", round(float(np.mean(isigi_v)), 2)
                    )
                    metrics_doc.setdefault(
                        "median_isigi", round(float(np.median(isigi_v)), 2)
                    )
                    if theo_set:
                        metrics_doc.setdefault(
                            "completeness_pct",
                            round(
                                float(100.0 * len(hm_v) / max(1, len(theo_set))),
                                2,
                            ),
                        )
                    if (
                        len(xv_v0) >= 2
                        and np.std(xv_v0) > 1e-12
                        and np.std(xv_v1) > 1e-12
                    ):
                        r_all = float(np.corrcoef(xv_v0, xv_v1)[0, 1])
                        if np.isfinite(r_all):
                            metrics_doc.setdefault("cc_half", round(r_all, 4))
                    metrics_doc["resolution_bins"] = bins
                    metrics_raw = json.dumps(metrics_doc, indent=2)
                    with open(metrics_path, "w", encoding="utf-8") as fh_out:
                        fh_out.write(metrics_raw)

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
        import glob as _glob
        import re as _re

        fname = os.path.basename(local_path)
        if not os.path.isfile(local_path):
            raise RuntimeError(
                f"Required input file '{fname}' was not found at {local_path}; "
                "run the previous pipeline stage first or select it from History."
            )
        local_size = os.path.getsize(local_path)
        gi = store.nova_connection.galaxy_instance
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})
        if not force_upload:
            cand_paths = [
                os.path.abspath(local_path),
                os.path.realpath(local_path),
            ]
            parent_d = os.path.dirname(os.path.abspath(local_path))
            parent_bn = os.path.basename(parent_d)
            if parent_bn.startswith("subhkl_work_") and (
                fname == "merged.h5" or fname.lower().endswith("_merged.h5")
            ):
                tag_part = parent_bn[len("subhkl_work_") :]
                cand_paths.append(os.path.join(parent_d, f"{tag_part}_merged.h5"))
                cand_paths.append(os.path.join(parent_d, "merged.h5"))
                for gx_f in _glob.glob(
                    os.path.join(
                        tempfile.gettempdir(),
                        f"subhkl_gx_*_{tag_part}_merged.h5",
                    )
                ):
                    cand_paths.append(os.path.abspath(gx_f))

            for cand_p in cand_paths:
                known_id = known_map.get(cand_p)
                if not known_id:
                    m_gx = _re.match(
                        r"^subhkl_gx_([0-9a-fA-F]+)_",
                        os.path.basename(cand_p),
                    )
                    if m_gx:
                        known_id = m_gx.group(1)
                if known_id:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        k_info = gi.datasets.show_dataset(known_id)
                        k_name = str(k_info.get("name") or "")
                        is_merged_req = fname == "merged.h5" or fname.lower().endswith(
                            "_merged.h5"
                        )
                        is_wrong_stage = is_merged_req and any(
                            w in k_name.lower()
                            for w in ("solve", "integrat", "predict", ".mtz")
                        )
                        if (
                            k_info.get("state") == "ok"
                            and not k_info.get("deleted")
                            and "/" not in k_name
                            and not is_wrong_stage
                        ):
                            known_map[os.path.abspath(local_path)] = known_id
                            self._dataset_id_by_path = known_map
                            self.touch_dataset(known_id)
                            self.attach_dataset_cell_metadata_from_file(
                                known_id,
                                local_path,
                                fallback_params=job_record.get("params"),
                            )
                            job_record["stdout"] += (
                                f"[nova -> galaxy] Linked cached dataset '{fname}' (id={known_id}, state=ok)\n"
                            )
                            return known_id

            search_names = [fname]
            if (
                fname == "merged.h5"
                and parent_bn.startswith("subhkl_work_")
                and len(parent_bn) > len("subhkl_work_")
            ):
                tag_part = parent_bn[len("subhkl_work_") :]
                search_names = [f"{tag_part}_merged.h5", "merged.h5"]

            generic_names = {
                "merged.h5",
                "solve.h5",
                "integrated.h5",
                "predictor.h5",
                "merged.mtz",
                "scan.h5",
            }
            in_prog_candidates: list[tuple[str, str]] = []
            all_hist_ds: list[dict[str, Any]] | None = None
            for sname in search_names:
                is_generic = sname.lower() in generic_names
                existing = list(
                    gi.datasets.get_datasets(history_id=store.history_id, name=sname)
                    or []
                )
                if sname in ("solve.h5", "integrated.h5", "merged.mtz"):
                    if all_hist_ds is None:
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            all_hist_ds = list(
                                gi.datasets.get_datasets(history_id=store.history_id)
                                or []
                            )
                    for d_item in all_hist_ds or []:
                        d_nm = str(d_item.get("name") or "")
                        if d_nm.startswith(f"{sname} (") and d_item not in existing:
                            existing.append(d_item)
                existing.sort(key=lambda d: int(d.get("hid") or 0))
                # Pass 1: Newest-first search for any dataset already in state='ok'
                for ds in reversed(existing):
                    ds_id = ds.get("id")
                    if not ds_id:
                        continue
                    info: dict[str, Any] = {}
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        raw_info = gi.datasets.show_dataset(ds_id)
                        if isinstance(raw_info, dict):
                            info = raw_info
                    if info.get("deleted") is True or ds.get("deleted") is True:
                        continue
                    ds_actual_name = str(info.get("name") or ds.get("name") or "")
                    if "/" in ds_actual_name or not (
                        ds_actual_name == sname
                        or ds_actual_name.startswith(f"{sname} (")
                    ):
                        continue
                    st = (
                        info.get("state")
                        if isinstance(info.get("state"), str)
                        else ds.get("state")
                    )
                    ds_sz = int(info.get("file_size") or 0)
                    if st == "ok":
                        if (
                            is_generic
                            and local_size > 0
                            and ds_sz > 0
                            and ds_sz != local_size
                        ):
                            continue
                        known_map[os.path.abspath(local_path)] = ds_id
                        self._dataset_id_by_path = known_map
                        self.touch_dataset(ds_id)
                        self.attach_dataset_cell_metadata_from_file(
                            ds_id,
                            local_path,
                            fallback_params=job_record.get("params"),
                        )
                        job_record["stdout"] += (
                            f"[nova -> galaxy] Linked existing dataset '{info.get('name') or sname}' (id={ds_id}, state=ok)\n"
                        )
                        return ds_id
                    if st in ("queued", "running", "new", "upload"):
                        in_prog_candidates.append((ds_id, str(st)))

            # Pass 2: If no 'ok' dataset exists yet, briefly check the newest in-progress dataset
            if in_prog_candidates:
                ds_id, st = in_prog_candidates[0]
                job_record["stdout"] += (
                    f"[nova -> galaxy] Waiting up to 15s for in-progress dataset '{fname}' (id={ds_id}, state={st})...\n"
                )
                with contextlib.suppress(Exception):
                    gi.datasets.wait_for_dataset(
                        ds_id, maxwait=15, interval=2.0, check=False
                    )
                    info = gi.datasets.show_dataset(ds_id)
                    ds_sz = int(info.get("file_size") or 0)
                    if (
                        info.get("state") == "ok"
                        and not info.get("deleted")
                        and (local_size <= 0 or ds_sz <= 0 or ds_sz == local_size)
                    ):
                        known_map[os.path.abspath(local_path)] = ds_id
                        self._dataset_id_by_path = known_map
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
        with contextlib.suppress(Exception):
            gi.datasets.wait_for_dataset(ds_id, maxwait=120, interval=2.0, check=False)
        known_map[os.path.abspath(local_path)] = ds_id
        self._dataset_id_by_path = known_map
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                except OSError:
                    shutil.copy2(local_path, cache_dest)
        self.touch_dataset(ds_id)
        self.attach_dataset_cell_metadata_from_file(
            ds_id,
            local_path,
            fallback_params=job_record.get("params"),
        )
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
            merged_path = params.get("images_file") or ""
            merged_has_images = False
            if merged_path and os.path.isfile(str(merged_path)):
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    import h5py

                    with h5py.File(str(merged_path), "r") as hf_chk:
                        merged_has_images = "images" in hf_chk
            if (
                not merged_path
                or not os.path.isfile(merged_path)
                or not merged_has_images
                or "/staging" in str(merged_path)
                or (
                    os.path.basename(str(merged_path)).startswith("dataset_")
                    and str(merged_path).endswith(".dat")
                )
            ) and params.get("peaks_file"):
                disc_m = self._discover_or_fetch_galaxy_images_h5(
                    params.get("peaks_file")
                )
                if disc_m:
                    merged_path = disc_m
                elif not merged_has_images:
                    merged_path = ""
            if not merged_path and stage_id not in ("reduce", "merge"):
                raise RuntimeError(
                    "No merged scan HDF5 file is loaded; run Step 1 (Reduce & Merge) "
                    "or select a merged.h5 dataset from History first."
                )
            work_dir, merged_path = (
                self._resolve_work_dir_and_merged(merged_path)
                if merged_path
                else (tempfile.gettempdir(), "")
            )
            self.invalidate_subsequent_stages(stage_id, work_dir)
            solve_path = params.get("peaks_file") or os.path.join(work_dir, "solve.h5")
            solve_bn = os.path.basename(solve_path).lower()
            cand_solve = os.path.join(work_dir, "solve.h5")
            if (
                not os.path.isfile(solve_path)
                or solve_bn in ("merged.h5", "integrated.h5", "predictor.h5")
                or solve_bn.endswith("_merged.h5")
            ):
                solve_path = cand_solve
            elif os.path.dirname(os.path.abspath(solve_path)) != os.path.abspath(
                work_dir
            ):
                from subhkl.viz.rainbow_zones import _discover_images_h5

                disc_for_solve = _discover_images_h5(solve_path)
                if not disc_for_solve or os.path.abspath(
                    disc_for_solve
                ) != os.path.abspath(merged_path):
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
                        and self._is_galaxy_server("http://127.0.0.1:9090")
                    ):
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
                d_min_solve = float(params.get("d_min") or round(0.5 * float(wl[0]), 2))
                job_record["images_h5_path"] = merged_path
                job_record["solve_params"] = {
                    "instrument": str(inst),
                    "space_group": str(sg),
                    "cell": [float(x) for x in raw_cell[:6]],
                    "wavelength_range": (float(wl[0]), float(wl[1])),
                    "d_min": d_min_solve,
                }
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
                    "dMin": d_min_solve,
                }
            elif stage_id == "integrate":
                primary_tool = "imaginex_subhkl_predict"
                scan_id = self._ensure_galaxy_h5_dataset(store, merged_path, job_record)
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                d_min_req = float(params.get("d_min") or round(0.5 * float(wl[0]), 2))
                tool_inputs = {
                    "scan": {"src": "hda", "id": scan_id},
                    "indexer": {"src": "hda", "id": idx_id},
                    "instrument": inst,
                    "dMin": round(d_min_req, 3),
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
                        flags=_re.IGNORECASE,
                    )
                    if not ipts_str:
                        m_ip = _re.search(r"/IPTS-([^/]+)/", first_p, _re.IGNORECASE)
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
                                        _re.IGNORECASE,
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
                        except _NOVA_RECOVERABLE_ERRORS:
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
                        r"_([0-9]+)\.nxs", os.path.basename(first_p), _re.IGNORECASE
                    )
                    ipts_str = _re.sub(
                        r"^IPTS-",
                        "",
                        str(params.get("ipts") or "").strip(),
                        flags=_re.IGNORECASE,
                    )
                    if not ipts_str:
                        m_ip = _re.search(r"/IPTS-([^/]+)/", first_p, _re.IGNORECASE)
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
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    self._api_request(
                                        "PUT",
                                        f"/api/histories/{store.history_id}/contents/{oid}",
                                        payload={"deleted": True},
                                    )
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
                if (
                    work_dir == os.path.abspath(tempfile.gettempdir())
                    and params.get("peaks_file")
                    and os.path.isfile(str(params.get("peaks_file")))
                ):
                    work_dir, _ = self._resolve_work_dir_and_merged(
                        str(params.get("peaks_file"))
                    )
                mtz_in_path = os.path.join(work_dir, "reflections.mtz")
                if not os.path.isfile(mtz_in_path) and self.is_connected:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        for ds in self.list_history_datasets(limit=100):
                            if ds.get("state") != "ok":
                                continue
                            dname = str(ds.get("name") or "").lower()
                            if "reflections.mtz" in dname or (
                                "unmerged" in dname and dname.endswith(".mtz")
                            ):
                                ds_id = str(ds.get("id") or "").strip()
                                if ds_id:
                                    dl_p, _ = self.resolve_dataset_to_local_h5(
                                        ds_id, fallback_dir=work_dir
                                    )
                                    if os.path.isfile(dl_p):
                                        import shutil as _shutil

                                        _shutil.copy2(dl_p, mtz_in_path)
                                        break
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
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    m_ip = _re.search(
                        r"/IPTS-([0-9]+)/", raw_nxs_all[0], _re.IGNORECASE
                    )
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
                            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                self._api_request(
                                    "PUT",
                                    f"/api/histories/{self.history_id}/contents/{po['id']}",
                                    payload={"deleted": True, "purged": True},
                                )
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
                import h5py
                import numpy as np

                from subhkl.commands import run_merge_images

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
                with (
                    contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS),
                    h5py.File(merged_dest, "r+") as hf,
                ):
                    if "files" in hf:
                        del hf["files"]
                    hf["files"] = np.array(
                        [f"out/{os.path.basename(p)}".encode() for p in part_paths]
                    )
                self._stamp_merged_metadata(merged_dest, job_record)
                if os.path.abspath(merged_dest) != os.path.abspath(merged_link):
                    try:
                        if os.path.exists(merged_link):
                            os.remove(merged_link)
                        os.link(merged_dest, merged_link)
                    except OSError:
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
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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

                threading.Thread(target=_bg_preupload, daemon=True).start()
                return

            if job_record.get("cancel_requested"):
                job_record["state"] = "deleted"
                return
            job_record["tool_id"] = primary_tool
            results = gi.tools.run_tool(
                history_id=store.history_id,
                tool_id=primary_tool,
                tool_inputs=tool_inputs,
            )
            job_id = results["jobs"][0]["id"]
            job_record["job_id"] = job_id
            job_record.setdefault("galaxy_job_ids", []).append(job_id)
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
                f"[galaxy -> pulsar] Dispatched {step_lbl}{primary_tool} (Galaxy job_id={job_id}) -> Pulsar GPU queue\n"
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
                and not job_record.get("cancel_requested")
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
                job_record.setdefault("galaxy_job_ids", []).append(int_job_id)
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
                    f"\n[galaxy -> pulsar] Dispatched Step 2/2 {int_tool} (Galaxy job_id={int_job_id}) -> Pulsar GPU queue\n"
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
        except _NOVA_RECOVERABLE_ERRORS as exc:
            if job_record.get("cancel_requested"):
                job_record["state"] = "deleted"
                return
            import shutil as _shutil

            clean_msg = self._clean_err_msg(exc)
            is_remote_galaxy = bool(
                self.galaxy_url
                and not any(h in self.galaxy_url for h in ("127.0.0.1", "localhost"))
            )
            has_container_rt = bool(_shutil.which("podman") or _shutil.which("docker"))
            if is_remote_galaxy and not has_container_rt:
                job_record["state"] = "error"
                job_record["stderr"] = f"{clean_msg}\n"
                job_record["stdout"] += f"\n[nova -> galaxy] Error: {clean_msg}\n"
                return
            from subhkl.io.parser import detect_runner_hostname

            local_host = detect_runner_hostname()
            job_record["stdout"] += (
                f"[nova -> galaxy] {clean_msg}; running GPU container on {local_host}...\n"
            )
            self._submit_local_subhkl_stage(
                stage_id,
                params,
                on_complete_h5,
                fallback_reason=clean_msg,
                existing_job_record=job_record,
            )

    SNS_HFIR_ROOTS: ClassVar[dict[str, str]] = {"SNS": "/SNS", "HFIR": "/HFIR"}

    @staticmethod
    def _resolve_local_podman_image(
        preferred_image: str,
        *,
        podman_bin: str = "podman",
    ) -> tuple[str, str | None]:
        """Resolve a usable Podman base image when ``preferred_image`` is an unbuilt branch tag or offline."""
        if (
            os.environ.get("SUBHKL_CONTAINER_IMAGE", "").strip()
            or os.environ.get("SUBHKL_CONTAINER_TAG", "").strip()
        ):
            return preferred_image, None

        with contextlib.suppress(OSError, subprocess.SubprocessError):
            exists_res = subprocess.run(
                [podman_bin, "image", "exists", preferred_image],
                capture_output=True,
                timeout=3.0,
                check=False,
            )
            if exists_res.returncode == 0:
                return preferred_image, None

        repo, sep, tag = preferred_image.rpartition(":")
        if not sep or "/" in tag:
            repo, tag = preferred_image, "latest"

        local_images: list[str] = []
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            list_res = subprocess.run(
                [
                    podman_bin,
                    "images",
                    "--format",
                    "{{.Repository}}:{{.Tag}}",
                    repo,
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            if list_res.returncode == 0:
                for raw_line in list_res.stdout.splitlines():
                    img = raw_line.strip()
                    if (
                        img
                        and img.startswith(f"{repo}:")
                        and not img.endswith(":<none>")
                        and img not in local_images
                    ):
                        local_images.append(img)

        local_fallback: str | None = None
        for candidate in (f"{repo}:main", f"{repo}:latest"):
            if candidate in local_images:
                local_fallback = candidate
                break
        if local_fallback is None and local_images:
            local_fallback = local_images[0]

        is_published_tag = bool(
            tag in ("main", "latest")
            or re.match(r"^[vV]?\d+\.\d+", tag)
            or re.match(r"^pr-\d+$", tag)
        )
        if not is_published_tag:
            if local_fallback is not None:
                return (
                    local_fallback,
                    f"{preferred_image} not built locally; using cached {local_fallback} with bind-mounted subhkl",
                )
            return (
                f"{repo}:main",
                f"{preferred_image} is a git branch tag; falling back to {repo}:main with bind-mounted subhkl",
            )

        if local_fallback is not None:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                manifest_res = subprocess.run(
                    [podman_bin, "manifest", "inspect", preferred_image],
                    capture_output=True,
                    timeout=4.0,
                    check=False,
                )
                if manifest_res.returncode != 0:
                    return (
                        local_fallback,
                        f"{preferred_image} unreachable in registry; using cached {local_fallback} with bind-mounted subhkl",
                    )

        return preferred_image, None

    @staticmethod
    def _parse_run_range_spec(spec_str: str) -> list[int]:
        """Parse run specifications such as '2625-2638', '2632-2638/2', '2632:2638:2', or '2632..2638..2'."""
        import re as _re

        runs: list[int] = []
        seen: set[int] = set()
        norm = _re.sub(r"\s*(\.\.|[-–:/])\s*", r"\1", str(spec_str or ""))
        for tok in _re.split(r"[,\s]+", norm):
            t = tok.strip()
            if not t:
                continue
            m = _re.match(r"^(\d+)(?:\.\.|[-–:])(\d+)(?:(?:\.\.|[/:])(\d+)?)?$", t)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                step = int(m.group(3)) if m.group(3) else 1
                if step <= 0:
                    continue
                lo, hi = min(a, b), max(a, b)
                for r in range(lo, hi + 1, step):
                    if len(runs) >= 20000:
                        break
                    if r not in seen:
                        seen.add(r)
                        runs.append(r)
            elif t.isdigit():
                r = int(t)
                if r not in seen:
                    seen.add(r)
                    runs.append(r)
        return runs

    @staticmethod
    def _format_run_list_to_spec(run_numbers: list[int]) -> str:
        uniq = sorted({int(n) for n in run_numbers})
        if not uniq:
            return ""
        parts: list[str] = []
        start = prev = uniq[0]
        for cur in uniq[1:]:
            if cur == prev + 1:
                prev = cur
            else:
                if prev == start:
                    parts.append(str(start))
                elif prev == start + 1:
                    parts.append(f"{start},{prev}")
                else:
                    parts.append(f"{start}-{prev}")
                start = prev = cur
        if prev == start:
            parts.append(str(start))
        elif prev == start + 1:
            parts.append(f"{start},{prev}")
        else:
            parts.append(f"{start}-{prev}")
        return ",".join(parts)

    def _facility_mount_root(self, fac: str) -> str:
        key = str(fac or "HFIR").strip().upper()
        return self.SNS_HFIR_ROOTS.get(key, "/HFIR")

    def _has_local_sns_hfir_fs(self) -> bool:
        return any(
            os.path.isdir(root)
            for root in (
                self._facility_mount_root("SNS"),
                self._facility_mount_root("HFIR"),
            )
        )

    @staticmethod
    def _read_local_nexus_summary(path: str) -> dict[str, Any]:
        """Lightweight read of NeXus HDF5 header metadata (title, sample, angles, duration) without touching event/image arrays."""
        import datetime as _dt
        import re as _re

        info: dict[str, Any] = {
            "title": "",
            "sample_name": "",
            "angles": "",
            "duration_s": 0.0,
            "duration_hms": "",
            "total_counts": 0,
            "size_mb": 0.0,
            "created": "",
        }
        with contextlib.suppress(OSError):
            st = os.stat(path)
            info["size_mb"] = round(st.st_size / (1024.0 * 1024.0), 1)
            info["created"] = _dt.datetime.fromtimestamp(
                st.st_mtime, tz=_dt.UTC
            ).strftime("%Y-%m-%d %H:%M")

        def _read_str(h5_obj: Any, ds_path: str) -> str:
            if ds_path not in h5_obj:
                return ""
            raw = h5_obj[ds_path][()]
            if hasattr(raw, "size") and getattr(raw, "size", 0) > 0:
                raw = raw.flat[0]
            if isinstance(raw, (bytes, bytearray)):
                return raw.decode("utf-8", "replace").strip()
            return str(raw).strip()

        with contextlib.suppress(Exception):
            import h5py

            with h5py.File(path, "r") as f:
                title = _read_str(f, "entry/title")
                sample_name = _read_str(f, "entry/sample/name")
                info["title"] = title
                info["sample_name"] = sample_name
                if "entry/duration" in f:
                    dur_raw = f["entry/duration"][()]
                    dur_s = float(
                        dur_raw.flat[0] if hasattr(dur_raw, "flat") else dur_raw
                    )
                    if dur_s > 0:
                        info["duration_s"] = dur_s
                        tot_sec = round(dur_s)
                        hh, rem = divmod(tot_sec, 3600)
                        mm, ss = divmod(rem, 60)
                        info["duration_hms"] = f"{hh:02d}:{mm:02d}:{ss:02d}"
                if "entry/total_counts" in f:
                    tc_raw = f["entry/total_counts"][()]
                    info["total_counts"] = int(
                        float(tc_raw.flat[0] if hasattr(tc_raw, "flat") else tc_raw)
                    )
                angle_parts: list[str] = []
                if "entry/DASlogs" in f:
                    das = f["entry/DASlogs"]
                    for ax_key, label in (
                        ("omega", "omega"),
                        ("BL11B:Mot:omega", "omega"),
                        ("HB2C:Mot:s1", "omega"),
                        ("chi", "chi"),
                        ("BL11B:Mot:chi", "chi"),
                        ("phi", "phi"),
                        ("BL11B:Mot:phi", "phi"),
                    ):
                        if ax_key in das and "average_value" in das[ax_key]:
                            av = das[ax_key]["average_value"][()]
                            val_f = float(av.flat[0] if hasattr(av, "flat") else av)
                            tag = f"{label}={val_f:.1f}"
                            if not any(p.startswith(f"{label}=") for p in angle_parts):
                                angle_parts.append(tag)
                angles = ";".join(angle_parts)
                if not angles and title:
                    ax_m = _re.findall(
                        r"(axis\s*\d+|omega|phi|chi|kappa)\s*=?\s*([-+]?\d+(?:\.\d+)?)",
                        title,
                        flags=_re.IGNORECASE,
                    )
                    if ax_m:
                        angles = ";".join(
                            f"{k.replace(' ', '').lower()}={v}" for k, v in ax_m
                        )
                info["angles"] = angles
        return info

    def _browse_local_sns_hfir_catalog(
        self,
        fac: str,
        inst: str,
        ipts_raw: str,
        *,
        skip_experiments: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Scan mounted ``/SNS/<INST>`` or ``/HFIR/<INST>`` directories when Galaxy / ONCat is unavailable."""
        import datetime as _dt
        import re as _re

        fac_root = self._facility_mount_root(fac)
        inst_dir = os.path.join(fac_root, inst)
        experiments: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        if not os.path.isdir(inst_dir):
            return experiments, runs

        if not skip_experiments:
            with contextlib.suppress(OSError):
                for entry in os.scandir(inst_dir):
                    m_ip = _re.match(r"^IPTS-(\d+)$", entry.name, _re.IGNORECASE)
                    if not m_ip or not entry.is_dir():
                        continue
                    ip_num = m_ip.group(1)
                    nxs_dir = os.path.join(entry.path, "nexus")
                    run_nums: list[int] = []
                    latest_mtime = 0.0
                    if os.path.isdir(nxs_dir) and os.access(nxs_dir, os.R_OK | os.X_OK):
                        with contextlib.suppress(OSError):
                            for nf in os.scandir(nxs_dir):
                                if not nf.name.endswith((".nxs.h5", ".nxs")):
                                    continue
                                m_r = _re.search(
                                    r"_([0-9]+)\.nxs", nf.name, _re.IGNORECASE
                                )
                                if m_r:
                                    run_nums.append(int(m_r.group(1)))
                                with contextlib.suppress(OSError):
                                    latest_mtime = max(latest_mtime, nf.stat().st_mtime)
                    else:
                        continue
                    run_nums_sorted = sorted(set(run_nums))
                    latest_dt = (
                        _dt.datetime.fromtimestamp(latest_mtime, tz=_dt.UTC).strftime(
                            "%Y-%m-%d"
                        )
                        if latest_mtime > 0
                        else ""
                    )
                    experiments.append(
                        {
                            "ipts": ip_num,
                            "id": f"IPTS-{ip_num}",
                            "n_runs": len(run_nums_sorted),
                            "run_ranges": self._format_run_list_to_spec(
                                run_nums_sorted
                            ),
                            "min_run": run_nums_sorted[0] if run_nums_sorted else None,
                            "max_run": run_nums_sorted[-1] if run_nums_sorted else None,
                            "latest_date": latest_dt,
                            "title": "",
                        }
                    )
            experiments.sort(
                key=lambda x: (
                    1 if (x.get("n_runs") or 0) > 0 else 0,
                    int(x["ipts"]) if str(x.get("ipts", "")).isdigit() else 0,
                ),
                reverse=True,
            )

        if ipts_raw:
            nxs_dir = os.path.join(inst_dir, f"IPTS-{ipts_raw}", "nexus")
            if os.path.isdir(nxs_dir) and os.access(nxs_dir, os.R_OK | os.X_OK):
                entries_by_run: list[tuple[int, str]] = []
                with contextlib.suppress(OSError):
                    for nf in os.scandir(nxs_dir):
                        if (
                            not nf.name.endswith((".nxs.h5", ".nxs"))
                            or not nf.is_file()
                        ):
                            continue
                        m_r = _re.search(r"_([0-9]+)\.nxs", nf.name, _re.IGNORECASE)
                        if m_r:
                            entries_by_run.append((int(m_r.group(1)), nf.path))
                entries_by_run.sort(key=lambda t: t[0])
                read_headers = len(entries_by_run) <= 250
                for r_num, r_path in entries_by_run:
                    if read_headers:
                        meta = self._read_local_nexus_summary(r_path)
                    else:
                        meta = {
                            "title": "",
                            "sample_name": "",
                            "angles": "",
                            "duration_s": 0.0,
                            "duration_hms": "",
                            "total_counts": 0,
                            "size_mb": 0.0,
                            "created": "",
                        }
                        with contextlib.suppress(OSError):
                            st = os.stat(r_path)
                            meta["size_mb"] = round(st.st_size / (1024.0 * 1024.0), 1)
                    runs.append(
                        {
                            "run": r_num,
                            "location": r_path,
                            "title": meta["title"],
                            "sample_name": meta["sample_name"],
                            "angles": meta["angles"],
                            "size_mb": meta["size_mb"],
                            "created": meta["created"],
                            "duration_s": meta["duration_s"],
                            "duration_hms": meta["duration_hms"],
                            "total_counts": meta["total_counts"],
                        }
                    )
        return experiments, runs

    def _remap_to_facility_root(self, p_str: str) -> str:
        import re as _re

        m = _re.match(r"^/(SNS|HFIR)(/.*)$", p_str, _re.IGNORECASE)
        if m:
            fac_key = m.group(1).upper()
            root = self._facility_mount_root(fac_key)
            return root.rstrip("/") + m.group(2)
        return p_str

    def check_neutron_paths(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate NeXus / HDF5 file paths via NDIP ``neutrons_load_data`` when Galaxy is available,
        falling back to direct local ``/SNS`` / ``/HFIR`` filesystem inspection when Galaxy is unavailable.
        """
        import glob as _glob
        import re as _re

        fac = str(body.get("facility") or "HFIR").strip()
        inst = str(body.get("instrument") or "CG4D").strip()
        ipts_raw = _re.sub(
            r"^IPTS-", "", str(body.get("ipts") or "").strip(), flags=_re.IGNORECASE
        )
        raw_paths = body.get("paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [
                s.strip() for s in raw_paths.replace("\n", ",").split(",") if s.strip()
            ]

        expanded: list[str] = [str(p).strip() for p in raw_paths if str(p).strip()]
        if not expanded and ipts_raw and body.get("runs"):
            fac_norm = fac.upper()
            inst_norm = inst.upper()
            if inst_norm in ("MANDI", "TOPAZ") and fac_norm == "HFIR":
                fac_norm = "SNS"
            elif inst_norm in ("CG4D", "IMAGINE") and fac_norm == "SNS":
                fac_norm = "HFIR"
            root_dir = self._facility_mount_root(fac_norm).rstrip("/")
            expanded = [
                f"{root_dir}/{inst_norm}/IPTS-{ipts_raw}/nexus/{inst_norm}_{r}.nxs.h5"
                for r in self._parse_run_range_spec(str(body.get("runs") or ""))
            ]
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
        checked_via = "ndip_unavailable"
        verified_dataset_ids: list[str] = []
        msg = ""
        sample_title = ""
        sample_goniometer = ""
        suggested_ipts: str | None = None
        suggested_paths: list[str] = []
        oncat_existing: list[str] = []
        oncat_missing: list[str] = []
        galaxy_unreachable = not self.is_connected

        if self.is_connected and self._has_tool_on_galaxy("neutrons_load_data"):
            try:
                ndip_checked = True
                checked_via = "ndip_neutrons_load_data"
                m_p0 = _re.match(
                    r"^/(SNS|HFIR)/([^/]+)/IPTS-([^/]+)/", expanded[0], _re.IGNORECASE
                )
                if m_p0 and (
                    str(body.get("ingest_mode") or "oncat") == "file" or not ipts_raw
                ):
                    fac = m_p0.group(1).upper()
                    inst = m_p0.group(2).upper()
                    ipts_raw = m_p0.group(3)
                elif not ipts_raw:
                    m_ip0 = _re.search(r"/IPTS-([^/]+)/", expanded[0], _re.IGNORECASE)
                    if m_ip0:
                        ipts_raw = m_ip0.group(1)
                # Verify via authenticated pyoncat catalog first (instant metadata check, no file fetch)
                if ipts_raw:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                                    r"_([0-9]+)\.nxs",
                                    os.path.basename(p),
                                    _re.IGNORECASE,
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

                if not ok and not msg:
                    store = self.get_datastore("IMAGINE-X SubHKL")
                    gi = store.nova_connection.galaxy_instance
                    first_p = oncat_existing[0] if oncat_existing else expanded[0]
                    m_run = _re.search(
                        r"_([0-9]+)\.nxs", os.path.basename(first_p), _re.IGNORECASE
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
                                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                        self._api_request(
                                            "PUT",
                                            f"/api/histories/{store.history_id}/contents/{oid}",
                                            payload={"deleted": True},
                                        )
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
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    self._api_request(
                                        "PUT",
                                        f"/api/histories/{store.history_id}/contents/{oid}",
                                        payload={"deleted": True},
                                    )
                            break
                        time.sleep(1.0)
                    if not ok and not msg:
                        msg = f"NDIP Load Neutron Data check timed out for {first_p}"
            except _NOVA_RECOVERABLE_ERRORS as exc:
                galaxy_unreachable = True
                ndip_checked = False
                msg = f"NDIP Load Neutron Data error: {self._clean_err_msg(exc)}"
        else:
            galaxy_unreachable = True

        # Fallback: when NDIP Galaxy is unconfigured or unreachable, inspect local /SNS and /HFIR directly
        if galaxy_unreachable and not ok:
            if self._has_local_sns_hfir_fs() or any(
                os.path.exists(self._remap_to_facility_root(p)) for p in expanded
            ):
                checked_via = "local_filesystem"
                fs_existing: list[str] = []
                fs_missing: list[str] = []
                titles_local: list[str] = []
                gonio_local: list[str] = []
                for p in expanded:
                    p_loc = self._remap_to_facility_root(p)
                    if any(ch in p_loc for ch in ("*", "?", "[")):
                        matches = sorted(
                            m
                            for m in _glob.glob(p_loc)
                            if os.path.isfile(m) and os.access(m, os.R_OK)
                        )
                        if matches:
                            fs_existing.extend(matches)
                        else:
                            fs_missing.append(p_loc)
                    elif os.path.isfile(p_loc) and os.access(p_loc, os.R_OK):
                        fs_existing.append(p_loc)
                    else:
                        fs_missing.append(p_loc)

                for p_ok in fs_existing[:6]:
                    meta = self._read_local_nexus_summary(p_ok)
                    t_str = str(meta.get("title") or "").strip()
                    g_str = str(meta.get("angles") or "").strip()
                    if t_str and t_str not in titles_local:
                        titles_local.append(t_str)
                    if g_str and g_str not in gonio_local:
                        gonio_local.append(g_str)
                if titles_local:
                    sample_title = "; ".join(titles_local[:3])
                if gonio_local:
                    sample_goniometer = "; ".join(gonio_local[:3])

                oncat_existing = fs_existing
                oncat_missing = fs_missing
                if fs_existing and not fs_missing:
                    ok = True
                    verified_dataset_ids = [
                        f"local_fs:{fac.upper()}:{inst.upper()}:IPTS-{ipts_raw or '?'}:{len(fs_existing)}"
                    ]
                    msg = (
                        f"Verified {len(fs_existing)} file(s) on local /SNS or /HFIR filesystem "
                        f"({inst} IPTS-{ipts_raw or '?'})"
                    )
                    if sample_title:
                        msg += f" — {sample_title[:60]}"
                else:
                    ok = False
                    # If the user entered the wrong IPTS, scan sibling IPTS-*/nexus for the missing run filename
                    if fs_missing:
                        first_miss_bn = os.path.basename(fs_missing[0])
                        fac_guess = (
                            "SNS"
                            if inst.upper() in ("MANDI", "TOPAZ")
                            else (
                                "HFIR"
                                if inst.upper() in ("CG4D", "IMAGINE")
                                else fac.upper()
                            )
                        )
                        inst_root = os.path.join(
                            self._facility_mount_root(fac_guess), inst.upper()
                        )
                        if os.path.isdir(inst_root):
                            with contextlib.suppress(OSError):
                                for ip_ent in os.scandir(inst_root):
                                    m_ip = _re.match(
                                        r"^IPTS-(\d+)$", ip_ent.name, _re.IGNORECASE
                                    )
                                    if not m_ip or not ip_ent.is_dir():
                                        continue
                                    cand_f = os.path.join(
                                        ip_ent.path, "nexus", first_miss_bn
                                    )
                                    if os.path.isfile(cand_f) and os.access(
                                        cand_f, os.R_OK
                                    ):
                                        suggested_ipts = m_ip.group(1)
                                        suggested_paths = [
                                            os.path.join(
                                                ip_ent.path,
                                                "nexus",
                                                os.path.basename(mp),
                                            )
                                            for mp in expanded
                                        ]
                                        break
                    if fs_existing:
                        msg = (
                            f"Local filesystem: {len(fs_existing)} found, {len(fs_missing)} missing "
                            f"({', '.join(os.path.basename(x) for x in fs_missing[:4])})"
                        )
                    else:
                        msg = (
                            f"Local filesystem: file not found or permission denied "
                            f"({', '.join(os.path.basename(x) for x in fs_missing[:3])})"
                        )
                        if suggested_ipts:
                            msg += f" — found in IPTS-{suggested_ipts}"
            elif not msg:
                msg = "NDIP Galaxy connection or local /SNS, /HFIR mount required to verify neutron data paths"

        return {
            "ok": ok,
            "ndip_checked": ndip_checked,
            "checked_via": checked_via,
            "existing": oncat_existing
            if oncat_existing and ok
            else (list(expanded) if ok else []),
            "missing": oncat_missing
            if oncat_missing
            else ([] if ok else list(expanded)),
            "ipts_exists": ok,
            "suggested_ipts": suggested_ipts,
            "suggested_paths": suggested_paths,
            "verified_dataset_ids": verified_dataset_ids,
            "sample_title": sample_title,
            "sample_goniometer": sample_goniometer,
            "message": msg,
        }

    ONCAT_DEFAULT_URL = "https://oncat.ornl.gov"
    # Public human-user device-authorization client ID published by ONCat docs.
    ONCAT_DEFAULT_CLIENT_ID = "eaeb036a-2602-4bb9-8530-0bb5812da7a1"
    ONCAT_DEFAULT_SCOPES: ClassVar[list[str]] = ["api:read", "openid"]

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
                except _NOVA_RECOVERABLE_ERRORS:
                    raw_token = ""

        if raw_token:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                if raw_token.startswith("{"):
                    tok_obj = json.loads(raw_token)
                else:
                    tok_obj = {
                        "access_token": raw_token,
                        "token_type": "Bearer",
                    }
                store.write_token(tok_obj)

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
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            oauth_client = self._build_oauth_oncat_client()
            if oauth_client.has_stored_token():
                return oauth_client, True
        try:
            import pyoncat  # type: ignore[import-untyped]

            return pyoncat.ONCat(url), False
        except _NOVA_RECOVERABLE_ERRORS:
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
        except _NOVA_RECOVERABLE_ERRORS:
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
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    import pyoncat  # type: ignore[import-untyped]

                    if isinstance(exc, pyoncat.InteractionRequiredError):
                        with contextlib.suppress(OSError):
                            tp = self._get_oncat_token_path()
                            if os.path.isfile(tp):
                                os.remove(tp)
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
            with contextlib.suppress(OSError):
                if os.path.isfile(tp):
                    os.remove(tp)
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
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        u = client.User.retrieve("me")
                        if isinstance(u, dict):
                            user_name = (
                                u.get("name")
                                if u.get("name") and u.get("name") != "Unknown"
                                else (u.get("id") or "ORNL User")
                            )
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
                except _NOVA_RECOVERABLE_ERRORS as exc:
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
            r"^IPTS-", "", str(params.get("ipts") or "").strip(), flags=_re.IGNORECASE
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
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                try:
                    raw_exps = oncat.Experiment.list(
                        facility=fac,
                        instrument=inst,
                    )
                except _NOVA_RECOVERABLE_ERRORS:
                    if is_auth:
                        import pyoncat as _pyoncat  # type: ignore[import-untyped]

                        oncat = _pyoncat.ONCat(
                            os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
                        )
                        is_auth = False
                        raw_exps = oncat.Experiment.list(facility=fac, instrument=inst)
                    else:
                        raise

                for e in raw_exps:
                    eid = str(e.get("id") or "")
                    m_ip = _re.search(r"IPTS-(\d+)", eid, _re.IGNORECASE)
                    ip_num = m_ip.group(1) if m_ip else eid
                    n_runs = int(e.get("size") or 0)
                    r_info = (e.get("indexed") or {}).get("run_number") or {}
                    ranges_str = str(r_info.get("ranges") or "")
                    latest_dt = str((e.get("latest") or {}).get("created") or "")[:10]
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
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                    except _NOVA_RECOVERABLE_ERRORS:
                        if is_auth:
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
                        else:
                            raise

                    for d in dfs:
                        idx_d = d.get("indexed") or {}
                        r_val = idx_d.get("run_number")
                        if r_val is None:
                            m_r = _re.search(
                                r"_([0-9]+)\.nxs",
                                str(d.get("location") or ""),
                                _re.IGNORECASE,
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
                                flags=_re.IGNORECASE,
                            )
                            if ax_m:
                                angles = ";".join(
                                    f"{k.replace(' ', '').lower()}={v}" for k, v in ax_m
                                )
                        sz_bytes = int(d.get("size") or 0)
                        created_str = str(d.get("created") or "")[:16].replace("T", " ")
                        dur_s = float(meta_entry.get("duration") or 0.0)
                        if dur_s > 0:
                            tot_sec = round(dur_s)
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

                runs = [by_run[k] for k in sorted(by_run.keys())]
                runs_cache[(fac, inst, ipts_raw, is_auth)] = (now, runs)
                self._oncat_runs_cache = runs_cache

        catalog_source = "oncat"
        if (not experiments and not skip_exps) or (ipts_raw and not runs):
            fs_exps, fs_runs = self._browse_local_sns_hfir_catalog(
                fac,
                inst,
                ipts_raw,
                skip_experiments=bool(experiments or skip_exps),
            )
            if not experiments and fs_exps:
                experiments = fs_exps
                catalog_source = "local_filesystem"
            if ipts_raw and not runs and fs_runs:
                runs = fs_runs
                catalog_source = "local_filesystem"

        return {
            "ok": True,
            "facility": fac,
            "instrument": inst,
            "ipts": ipts_raw,
            "catalog_source": catalog_source,
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
        import re as _re

        resolved: list[str] = []
        raw_list = params.get("nexus_files")
        if isinstance(raw_list, str):
            raw_list = [
                s.strip() for s in raw_list.replace("\n", ",").split(",") if s.strip()
            ]
        if isinstance(raw_list, list) and raw_list:
            for item in raw_list:
                p_str = self._remap_to_facility_root(str(item).strip())
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

        ipts_raw = _re.sub(
            r"^IPTS-", "", str(params.get("ipts") or "").strip(), flags=_re.IGNORECASE
        )
        runs_spec = str(params.get("runs") or "").strip()
        if ipts_raw and runs_spec:
            inst_norm = str(params.get("instrument") or "CG4D").strip().upper()
            fac_norm = str(params.get("facility") or "HFIR").strip().upper()
            if inst_norm in ("MANDI", "TOPAZ") and fac_norm == "HFIR":
                fac_norm = "SNS"
            elif inst_norm in ("CG4D", "IMAGINE") and fac_norm == "SNS":
                fac_norm = "HFIR"
            root_dir = self._facility_mount_root(fac_norm).rstrip("/")
            nxs_dir = os.path.join(root_dir, inst_norm, f"IPTS-{ipts_raw}", "nexus")
            for r_num in self._parse_run_range_spec(runs_spec):
                cand_h5 = os.path.join(nxs_dir, f"{inst_norm}_{r_num}.nxs.h5")
                cand_nxs = os.path.join(nxs_dir, f"{inst_norm}_{r_num}.nxs")
                if os.path.isfile(cand_nxs) and not os.path.isfile(cand_h5):
                    resolved.append(cand_nxs)
                else:
                    resolved.append(cand_h5)
            if resolved:
                return resolved

        ds_id = params.get("galaxy_dataset_id")
        if ds_id:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                local_p, _ = self.resolve_dataset_to_local_h5(
                    str(ds_id), fallback_dir=work_dir
                )
                if local_p and os.path.isfile(local_p):
                    return [local_p]

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
        raw_img = params.get("images_file") or ""
        if (
            not raw_img
            or not os.path.isfile(raw_img)
            or "/staging" in str(raw_img)
            or (
                os.path.basename(str(raw_img)).startswith("dataset_")
                and str(raw_img).endswith(".dat")
            )
        ) and params.get("peaks_file"):
            from subhkl.viz.rainbow_zones import _discover_images_h5

            disc_m = _discover_images_h5(params.get("peaks_file"))
            if disc_m:
                raw_img = disc_m
        if not raw_img and stage_id != "reduce":
            raise RuntimeError(
                "No merged scan HDF5 file is loaded; run Step 1 (Reduce & Merge) "
                "or select a merged.h5 dataset from History first."
            )
        merged_path = os.path.abspath(
            raw_img or os.path.join(tempfile.gettempdir(), "merged.h5")
        )
        work_dir, merged_path = self._resolve_work_dir_and_merged(merged_path)
        self.invalidate_subsequent_stages(stage_id, work_dir)
        solve_dest = os.path.join(work_dir, "solve.h5")
        pred_dest = os.path.join(work_dir, "predictor.h5")
        integrated_dest = os.path.join(work_dir, "integrated.h5")
        mtz_dest = os.path.join(work_dir, "reflections.mtz")

        solve_path = os.path.abspath(params.get("peaks_file") or solve_dest)
        solve_bn = os.path.basename(solve_path).lower()
        if (
            not os.path.isfile(solve_path)
            or solve_bn in ("merged.h5", "integrated.h5", "predictor.h5")
            or solve_bn.endswith("_merged.h5")
        ):
            solve_path = solve_dest
        elif os.path.dirname(os.path.abspath(solve_path)) != os.path.abspath(work_dir):
            from subhkl.viz.rainbow_zones import _discover_images_h5

            disc_for_solve = _discover_images_h5(solve_path)
            if not disc_for_solve or os.path.abspath(disc_for_solve) != os.path.abspath(
                merged_path
            ):
                solve_path = solve_dest

        ts = int(time.time())
        out_h5 = os.path.join(tempfile.gettempdir(), f"subhkl_{stage_id}_{ts}.h5")
        pred_tmp_h5 = os.path.join(tempfile.gettempdir(), f"subhkl_pred_{ts}.h5")
        mtz_tmp_path = os.path.join(tempfile.gettempdir(), f"subhkl_refl_{ts}.mtz")
        reduce_tmp_dir = os.path.join(tempfile.gettempdir(), f"subhkl_reduce_{ts}")
        raw_nxs_local: list[str] = []
        merged_label = "merged.h5"
        if stage_id == "reduce":
            resolved_candidates = self._resolve_raw_nexus_inputs(params, work_dir)
            raw_nxs_local = [
                p
                for p in resolved_candidates
                if os.path.isfile(p) and os.access(p, os.R_OK)
            ]
            if resolved_candidates and not raw_nxs_local:
                missing_sample = ", ".join(
                    os.path.basename(p) for p in resolved_candidates[:4]
                )
                raise FileNotFoundError(
                    f"Local NeXus file(s) not found or unreadable on /SNS or /HFIR: {missing_sample}"
                )
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
            refine_modes_str = str(
                params.get(
                    "refine_modes",
                    "cylindrical,axial_stretch,area,global_rot,global_trans,sample,cell,per_run,per_run_trans",
                )
            )
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
                str(params.get("d_min") or round(0.5 * float(wl[0]), 2)),
                "--refinement",
                str(params.get("objective", "band")),
            ]
            if str(params.get("objective", "band")) == "band" and refine_modes_str:
                inner_cmd.extend(["--refine-modes", refine_modes_str])
        elif (
            stage_id == "integrate"
            and os.path.isfile(merged_path)
            and os.path.isfile(solve_path)
        ):
            d_min_req = float(params.get("d_min") or round(0.5 * float(wl[0]), 2))
            d_min_val = str(round(d_min_req, 3))
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
            rel_border = float(params.get("rel_border_width", 0.03))
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
                "--rel-border-width",
                str(round(rel_border, 4)),
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
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
            with contextlib.suppress(OSError):
                os.chmod(prep_script, 0o666)
                os.chmod(post_script, 0o666)
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
            else get_container_image()
        )
        image_fallback_note: str | None = None
        podman_bin = shutil.which("podman")
        if podman_bin:
            if stage_id != "merge":
                container_image, image_fallback_note = self._resolve_local_podman_image(
                    container_image,
                    podman_bin=podman_bin,
                )
            raw_dirs = {
                d
                for p in raw_nxs_local
                for d in (
                    os.path.dirname(os.path.abspath(p)),
                    os.path.dirname(os.path.realpath(p)),
                )
            }
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
        except (OSError, subprocess.SubprocessError, IndexError):
            gpu_info = "CPU fallback"
            gpu_env["JAX_PLATFORMS"] = "cpu"

        job_id = f"gpu-{int(time.time())}"
        container_name: str | None = None
        if cmd and cmd[0] == "podman":
            container_name = f"subhkl-{job_id}"
            cmd.insert(3, f"--name={container_name}")

        from subhkl.io.parser import detect_runner_hostname

        host_tag = detect_runner_hostname()
        cmd_str = " ".join(cmd)
        header_lines: list[str] = []
        if stage_id != "merge":
            if fallback_reason and existing_job_record is None:
                header_lines.append(f"[ndip] Note: {fallback_reason}")
            if image_fallback_note:
                header_lines.append(f"[container] Note: {image_fallback_note}")
            header_lines.append(f"[pulsar] Runner host: {host_tag}")
            header_lines.append(
                f"[container] Host={host_tag} | GPU={gpu_info} | Image={container_image}"
            )
            header_lines.append(f"$ {cmd_str}\n")

        if existing_job_record is not None:
            job_record = existing_job_record
            job_record["job_id"] = job_id
            job_record["mode"] = "container_gpu"
            job_record["state"] = "running"
            job_record["runner_host"] = host_tag
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
                "runner_host": host_tag,
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
                "d_min": float(params.get("d_min") or round(0.5 * float(wl[0]), 2)),
            }

        def _stream_proc() -> None:
            try:
                self._active_container_name = container_name
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    env=gpu_env,
                    start_new_session=True,
                )
                self._active_proc = proc
                assert proc.stdout is not None
                blind_src = out_h5[:-3] + ".blind.h5"
                stages_src = out_h5[:-3] + ".stages.h5"
                blind_dst = solve_dest[:-3] + ".blind.h5"
                last_live_key: tuple[str, int] = ("", 0)
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
                        if (
                            "blind band search" in low
                            or "band table" in low
                            or "exhaustive rung" in low
                            or "rung 13 zones" in low
                        ):
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 1
                            )
                        if "rung 30 zones" in low or "rung 60 zones" in low:
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 2
                            )
                        if (
                            "rung 145 zones" in low
                            or "joint band refinement" in low
                            or "candidate " in low
                            or "full-data scores" in low
                            or "ewald pre-polish" in low
                            or "global" in low
                            or "60-zone" in low
                            or "[stage 0" in low
                            or "120" in low
                            or "cell" in low
                            or "[stage 1" in low
                            or "[stage 2" in low
                            or "[stage 3" in low
                            or "[stage 4" in low
                        ):
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 3
                            )
                        if "band refinement: converged" in low:
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 4
                            )
                        active_live_src = None
                        if (
                            os.path.isfile(stages_src)
                            and os.path.getsize(stages_src) > 0
                        ):
                            active_live_src = stages_src
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 3
                            )
                        elif (
                            os.path.isfile(blind_src) and os.path.getsize(blind_src) > 0
                        ):
                            active_live_src = blind_src
                        if active_live_src is not None:
                            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                cur_mtime_ns = int(os.stat(active_live_src).st_mtime_ns)
                                cur_key = (active_live_src, cur_mtime_ns)
                                if cur_key != last_live_key:
                                    last_live_key = cur_key
                                    shutil.copy2(active_live_src, blind_dst)
                                    job_record["partial_h5_path"] = blind_dst
                                    from subhkl.viz.rainbow_zones import (
                                        build_rainbow_zone_payload,
                                    )

                                    ov_frame = int(self.current_frame_index or 0)
                                    p_blind = build_rainbow_zone_payload(
                                        replay_h5=blind_dst,
                                        images_h5=merged_path,
                                        instrument=str(inst),
                                        space_group=str(sg),
                                        cell=None,
                                        wavelength_range=(
                                            float(wl[0]),
                                            float(wl[1]),
                                        ),
                                        max_zone_index=3,
                                        max_zones=60,
                                        include_panels=True,
                                        include_panel_images=False,
                                        include_peaks=True,
                                        n_arc=720,
                                        frame_index=ov_frame,
                                    )
                                    if p_blind.get("zones"):
                                        partial_seq += 1
                                        job_record["partial_overlay"] = {
                                            "seq": partial_seq,
                                            "frame_index": int(
                                                p_blind.get("frame_index", ov_frame)
                                            ),
                                            "zones": p_blind.get("zones", []),
                                            "peaks": p_blind.get("peaks", []),
                                            "sample_peaks": p_blind.get(
                                                "sample_peaks", []
                                            ),
                                            "panels": p_blind.get("panels", []),
                                            "cell": p_blind.get("cell"),
                                            "ub_matrix": p_blind.get("ub_matrix"),
                                            "goniometer_R": p_blind.get("goniometer_R"),
                                            "ladder_steps": p_blind.get(
                                                "ladder_steps", []
                                            ),
                                        }
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
                                except OSError:
                                    shutil.copy2(merged_dest, merged_link)
                            final_h5 = merged_dest
                            job_record["stdout"] += (
                                f"\n[subhkl] Saved {merged_label} -> {merged_dest}\n"
                            )
                        except _NOVA_RECOVERABLE_ERRORS as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving {merged_dest}: {copy_exc}\n"
                            )

                        if self.is_connected and os.path.isfile(final_h5):

                            def _bg_upload_reduced(
                                _f_h5: str = final_h5,
                                _lbl: str = merged_label,
                            ) -> None:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                        except _NOVA_RECOVERABLE_ERRORS as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving {solve_dest}: {copy_exc}\n"
                            )

                        if self.is_connected and os.path.isfile(final_h5):

                            def _bg_upload_solve(_f_h5: str = final_h5) -> None:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                                            "name": "solve.h5",
                                            "extension": "h5",
                                        }
                                    ]

                            threading.Thread(
                                target=_bg_upload_solve, daemon=True
                            ).start()
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
                        except _NOVA_RECOVERABLE_ERRORS as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving integrated outputs: {copy_exc}\n"
                            )

                        if self.is_connected:

                            def _bg_upload_integrate(
                                _f_h5: str = final_h5,
                                _p_h5: str = pred_dest,
                                _m_mtz: str = mtz_dest,
                            ) -> None:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    store = self.get_datastore("IMAGINE-X SubHKL")
                                    bg_rec: dict[str, Any] = {"stdout": ""}
                                    outs_up = []
                                    if os.path.isfile(_f_h5):
                                        int_id = self._ensure_galaxy_h5_dataset(
                                            store,
                                            _f_h5,
                                            bg_rec,
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
                                    if os.path.isfile(_p_h5):
                                        pred_id = self._ensure_galaxy_h5_dataset(
                                            store,
                                            _p_h5,
                                            bg_rec,
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
                                    if os.path.isfile(_m_mtz):
                                        mtz_id = self._ensure_galaxy_h5_dataset(
                                            store,
                                            _m_mtz,
                                            bg_rec,
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

                            threading.Thread(
                                target=_bg_upload_integrate, daemon=True
                            ).start()
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
                        except _NOVA_RECOVERABLE_ERRORS as merge_exc:
                            job_record["stdout"] += (
                                f"\n[careless] Warning stamping merge results into HDF5: {merge_exc}\n"
                            )
                        if self.is_connected:

                            def _bg_upload_merge(_wdir: str = work_dir) -> None:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    store = self.get_datastore("IMAGINE-X SubHKL")
                                    bg_rec: dict[str, Any] = {"stdout": ""}
                                    outs_up = []
                                    for fn, ext in (
                                        ("merged.mtz", "binary"),
                                        ("predictions.mtz", "binary"),
                                        ("xval.mtz", "binary"),
                                        ("careless_metrics.json", "json"),
                                    ):
                                        fp = os.path.join(_wdir, fn)
                                        if os.path.isfile(fp):
                                            did = self._ensure_galaxy_h5_dataset(
                                                store,
                                                fp,
                                                bg_rec,
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

                            threading.Thread(
                                target=_bg_upload_merge, daemon=True
                            ).start()

                    if os.path.isfile(final_h5):
                        job_record["output_h5_path"] = final_h5
                        if callable(on_complete_h5):
                            on_complete_h5(final_h5)
                            job_record["updated_payload_ready"] = True
                    job_record["stdout"] += (
                        f"[subhkl] Stage '{stage_id}' completed (exit 0).\n"
                    )
                    job_record["state"] = "ok"
                elif job_record.get("cancel_requested"):
                    job_record["state"] = "deleted"
                else:
                    job_record["state"] = "error"
                    job_record["stderr"] += (
                        f"\n[subhkl] Process exited with code {rc}\n"
                    )
            except _NOVA_RECOVERABLE_ERRORS as exc:
                if job_record.get("cancel_requested"):
                    job_record["state"] = "deleted"
                else:
                    job_record["state"] = "error"
                    job_record["stderr"] += f"\n[subhkl] Error: {exc}\n"
            finally:
                self._active_proc = None
                self._active_container_name = None

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
        last_live_token = ""
        partial_seq = 0
        base_log = job_record.get("stdout", "")
        is_localhost_galaxy = bool(
            self.galaxy_url
            and any(h in self.galaxy_url for h in ("127.0.0.1", "localhost"))
        )

        while True:
            if job_record.get("cancel_requested"):
                job_record["state"] = "deleted"
                break
            time.sleep(1.5)
            if job_record.get("cancel_requested"):
                job_record["state"] = "deleted"
                break
            try:
                # Fetch job status & live Pulsar console_output stream from NDIP Galaxy API
                jinfo = self._api_request("GET", f"/api/jobs/{job_id}?full=true")
                poll_errors = 0
                state = jinfo.get("state", "running")
                if state == "new":
                    new_state_ticks += 1
                    if is_localhost_galaxy and new_state_ticks >= 10:
                        self._pulsar_queue_stalled_until = time.time() + 300.0
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            self._api_request("DELETE", f"/api/jobs/{job_id}")
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
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    cinfo = self._api_request(
                        "GET",
                        f"/api/jobs/{job_id}/console_output?stdout_position=0&stdout_length=65536&stderr_position=0&stderr_length=65536",
                    )
                    if isinstance(cinfo, dict):
                        console_out = cinfo.get("stdout") or ""
                        console_err = cinfo.get("stderr") or ""

                raw_stdout_src = (
                    console_out or jinfo.get("tool_stdout") or jinfo.get("stdout") or ""
                )
                out_txt = self._format_terminal_stream(raw_stdout_src)
                err_txt = self._format_terminal_stream(
                    console_err or jinfo.get("tool_stderr") or jinfo.get("stderr") or ""
                )
                import re as _re_h

                m_host = _re_h.search(
                    r"\[(?:pulsar|subhkl)\]\s*Runner host:\s*([A-Za-z0-9._-]+)",
                    f"{raw_stdout_src}\n{console_err}",
                    _re_h.IGNORECASE,
                )
                if m_host:
                    r_host = m_host.group(1).strip()
                    if r_host:
                        job_record["runner_host"] = r_host
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
                    low_out = out_txt.lower()
                    cur_rung = int(job_record.get("ladder_rung_idx") or 0)
                    if (
                        "blind band search" in low_out
                        or "band table" in low_out
                        or "exhaustive rung" in low_out
                        or "rung 13 zones" in low_out
                    ):
                        cur_rung = max(cur_rung, 1)
                    if "rung 30 zones" in low_out or "rung 60 zones" in low_out:
                        cur_rung = max(cur_rung, 2)
                    if (
                        "rung 145 zones" in low_out
                        or "joint band refinement" in low_out
                        or "candidate " in low_out
                        or "full-data scores" in low_out
                        or "[stage " in low_out
                    ):
                        cur_rung = max(cur_rung, 3)
                    if "band refinement: converged" in low_out:
                        cur_rung = max(cur_rung, 4)
                    if cur_rung == 0 and state == "running":
                        step_counter = min(3, step_counter + 1)
                        cur_rung = step_counter
                    job_record["ladder_rung_idx"] = cur_rung

                    if "[subhkl-live-state]" in raw_stdout_src:
                        live_tokens = [
                            ln.strip().split("[subhkl-live-state]", 1)[1].strip()
                            for ln in raw_stdout_src.splitlines()
                            if "[subhkl-live-state]" in ln
                        ]
                        if live_tokens and live_tokens[-1] != last_live_token:
                            last_live_token = live_tokens[-1]
                            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                blind_dst = os.path.join(
                                    work_dir or tempfile.gettempdir(),
                                    "solve.blind.h5",
                                )
                                if self._materialize_live_state_h5(
                                    last_live_token, blind_dst
                                ):
                                    job_record["partial_h5_path"] = blind_dst
                                    partial_seq += 1
                                    self.apply_active_solve_overlay(
                                        {},
                                        frame_index=int(self.current_frame_index or 0),
                                    )
                                    if isinstance(
                                        job_record.get("partial_overlay"), dict
                                    ):
                                        job_record["partial_overlay"]["seq"] = (
                                            partial_seq
                                        )
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
                                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                        self._api_request(
                                            "PUT",
                                            f"/api/histories/{self.history_id}/contents/{ds_id}",
                                            payload={"name": merged_label},
                                        )
                                        h5_outs[0]["name"] = merged_label
                                        job_record["stdout"] += (
                                            f"[galaxy -> history] Created dataset '{merged_label}' (id={ds_id}) in Galaxy history\n"
                                        )
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
                                    except _NOVA_RECOVERABLE_ERRORS:
                                        if attempt == 5:
                                            raise
                                    time.sleep(2.0)
                                if raw_bytes:
                                    if target_path:
                                        with open(target_path, "wb") as f:
                                            f.write(raw_bytes)
                                        tmp_path = target_path
                                        if not job_record.get("reduce_part_path"):
                                            with (
                                                contextlib.suppress(
                                                    *_NOVA_RECOVERABLE_ERRORS
                                                ),
                                                open(cache_gx_path, "wb") as cf,
                                            ):
                                                cf.write(raw_bytes)
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
                                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                                    self.touch_dataset(ds_id)
                                    self.attach_dataset_cell_metadata_from_file(
                                        ds_id,
                                        tmp_path,
                                        fallback_params=job_record.get("params"),
                                    )
                                    known_map = getattr(self, "_dataset_id_by_path", {})
                                    known_map[os.path.abspath(tmp_path)] = ds_id
                                    if (
                                        work_dir
                                        and job_record.get("stage_id") == "reduce"
                                    ):
                                        lbl_path = os.path.join(work_dir, merged_label)
                                        known_map[os.path.abspath(lbl_path)] = ds_id
                                        if os.path.abspath(lbl_path) != os.path.abspath(
                                            tmp_path
                                        ):
                                            try:
                                                if os.path.lexists(lbl_path):
                                                    os.remove(lbl_path)
                                                os.link(tmp_path, lbl_path)
                                            except _NOVA_RECOVERABLE_ERRORS:
                                                with contextlib.suppress(
                                                    *_NOVA_RECOVERABLE_ERRORS
                                                ):
                                                    shutil.copy2(tmp_path, lbl_path)
                                    self._dataset_id_by_path = known_map
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
                                    if (
                                        oname
                                        in ("metrics_json", "careless_metrics_json")
                                        or "careless_metrics" in oname
                                    ):
                                        dest_name = "careless_metrics.json"
                                    else:
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
                                        with contextlib.suppress(
                                            *_NOVA_RECOVERABLE_ERRORS
                                        ):
                                            bdata = self._api_request(
                                                "GET",
                                                f"/api/datasets/{oid}/display",
                                                raw=True,
                                                timeout=180.0,
                                            )
                                            dest_p = os.path.join(work_dir, dest_name)
                                            with open(dest_p, "wb") as f:
                                                f.write(bdata)
                                            known_map = getattr(
                                                self, "_dataset_id_by_path", {}
                                            )
                                            known_map[os.path.abspath(dest_p)] = oid
                                            known_map[os.path.realpath(dest_p)] = oid
                                            self._dataset_id_by_path = known_map
                                            job_record["stdout"] += (
                                                f"[careless] Saved {dest_name} -> {dest_p}\n"
                                            )
                                int_target = os.path.join(work_dir, "integrated.h5")
                                if not os.path.isfile(int_target):
                                    pf_cand = str(
                                        (job_record.get("params") or {}).get(
                                            "peaks_file"
                                        )
                                        or ""
                                    )
                                    if pf_cand and os.path.isfile(pf_cand):
                                        import shutil as _shutil

                                        with contextlib.suppress(OSError):
                                            _shutil.copy2(pf_cand, int_target)
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
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
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
                        except _NOVA_RECOVERABLE_ERRORS as post_exc:
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
                        misc_info = ""
                        for out_d in job_record.get("outputs", []):
                            oid = out_d.get("id")
                            if not oid:
                                continue
                            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                dmeta = self._api_request("GET", f"/api/datasets/{oid}")
                                if isinstance(dmeta, dict) and dmeta.get("misc_info"):
                                    misc_info = str(dmeta["misc_info"]).strip()
                                    break
                        if not misc_info and jinfo.get("job_messages"):
                            misc_info = "; ".join(
                                str(m.get("desc") or m)
                                if isinstance(m, dict)
                                else str(m)
                                for m in jinfo["job_messages"]
                            ).strip()
                        if misc_info:
                            if misc_info not in job_record.get("stderr", ""):
                                job_record["stderr"] = (
                                    f"{job_record.get('stderr', '')}{misc_info}\n"
                                )[-16000:]
                            if misc_info not in job_record.get("stdout", ""):
                                job_record["stdout"] += (
                                    f"\n[galaxy -> error] {misc_info}\n"
                                )
                        if state == "error" and not jinfo.get("command_line"):
                            raise RuntimeError(
                                misc_info
                                or f"Galaxy job {job_id} failed before command execution"
                            )
                        job_record["state"] = state
                    break
            except TimeoutError:
                raise
            except _NOVA_RECOVERABLE_ERRORS as exc:
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
        if "frame_index" in overrides and overrides["frame_index"] is not None:
            self.payload_kwargs["frame_index"] = int(overrides["frame_index"])
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
        state.trame__title = "SUBHKL // Diffraction Parallax"
        state.trame__favicon = SUBHKL_FAVICON_DATA_URI
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
        except (ImportError, *_NOVA_RECOVERABLE_ERRORS):
            try:
                app = RainbowZonesTrameApp(vm)
                app.start(host=host, port=port)
                return
            except RuntimeError:
                pass

    ep_path = os.environ.get("EP_PATH", "").rstrip("/")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            req_path = parsed.path
            if ep_path and req_path.startswith(ep_path):
                req_path = req_path[len(ep_path) :] or "/"

            if req_path in ("/favicon.ico", "/favicon.svg", "/logo.png"):
                fav_bytes = SUBHKL_FAVICON_SVG.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
                self.send_header("Content-Length", str(len(fav_bytes)))
                self.end_headers()
                self.wfile.write(fav_bytes)
                return

            if req_path in ("/api/payload", "/payload.json"):
                f_idx: int | None = None
                if qs.get("frame"):
                    try:
                        f_idx = int(qs["frame"][0])
                        vm.payload_kwargs["frame_index"] = f_idx
                        vm.pipeline_runner.current_frame_index = f_idx
                    except (ValueError, TypeError, IndexError):
                        f_idx = None
                preserve_q = qs.get("preserve_params", ["0"])[0] == "1"
                d_min_val: float | None = None
                if qs.get("d_min"):
                    with contextlib.suppress(ValueError, TypeError, IndexError):
                        parsed_dmin = float(qs["d_min"][0])
                        if 0.2 <= parsed_dmin <= 20.0:
                            d_min_val = parsed_dmin
                            vm.payload_kwargs["d_min"] = parsed_dmin
                reloaded = False
                if qs.get("file_path"):
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        fp = os.path.abspath(str(qs["file_path"][0]).strip())
                        if os.path.isfile(fp):
                            fp_low = os.path.basename(fp).lower()
                            if any(k in fp_low for k in ("merged", "scan", "image")):
                                _, fp = vm.pipeline_runner._resolve_work_dir_and_merged(
                                    fp
                                )
                                vm.payload_kwargs["images_h5"] = fp
                                vm.payload_kwargs.pop("d_min", None)
                                vm.replay_h5 = fp
                                vm.reload(images_h5=fp, frame_index=f_idx)
                            else:
                                if not preserve_q and d_min_val is None:
                                    vm.payload_kwargs.pop("d_min", None)
                                disc_img = vm.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                    fp
                                )
                                if disc_img:
                                    vm.payload_kwargs["images_h5"] = str(disc_img)
                                vm.replay_h5 = fp
                                vm.reload(
                                    images_h5=vm.payload_kwargs.get("images_h5"),
                                    frame_index=f_idx,
                                )
                            vm.pipeline_runner.invalidate_subsequent_stages(
                                (vm._payload or {}).get("artifact_kind"),
                                os.path.dirname(fp),
                            )
                            reloaded = True
                if qs.get("dataset_id"):
                    try:
                        from subhkl.viz.rainbow_zones import (
                            _discover_images_h5,
                            apply_mtz_to_payload,
                            validate_subhkl_ingestible_dataset,
                        )

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
                        cur_p = vm._payload or {}
                        has_active_geom = bool(
                            (
                                cur_p.get("artifact_kind") != "empty"
                                and cur_p.get("panels")
                                and (cur_p.get("peaks") or cur_p.get("ub_matrix"))
                            )
                            or qs.get("has_active_geometry", ["0"])[0] == "1"
                        )
                        ingest_kind = validate_subhkl_ingestible_dataset(
                            local_h5,
                            ds_info,
                            has_active_geometry=has_active_geom,
                        )
                        if ingest_kind == "mtz_overlay":
                            vm._payload = apply_mtz_to_payload(cur_p, local_h5, ds_info)
                            body = json.dumps(vm._payload).encode("utf-8")
                            self.send_response(200)
                            self.send_header(
                                "Content-Type", "application/json; charset=utf-8"
                            )
                            self.send_header("Content-Length", str(len(body)))
                            self.end_headers()
                            self.wfile.write(body)
                            return
                        if ingest_kind == "raw_nexus":
                            raw_body = json.dumps(
                                {
                                    "status": "raw_nexus",
                                    "dataset_id": ds_id,
                                    "name": ds_info.get("name")
                                    or os.path.basename(local_h5),
                                }
                            ).encode("utf-8")
                            self.send_response(200)
                            self.send_header(
                                "Content-Type", "application/json; charset=utf-8"
                            )
                            self.send_header("Content-Length", str(len(raw_body)))
                            self.end_headers()
                            self.wfile.write(raw_body)
                            return
                        ds_name = str(
                            ds_info.get("name") or os.path.basename(local_h5)
                        ).lower()
                        if any(k in ds_name for k in ("merged", "scan", "image")):
                            _, local_h5 = (
                                vm.pipeline_runner._resolve_work_dir_and_merged(
                                    local_h5
                                )
                            )
                            vm.payload_kwargs["images_h5"] = local_h5
                            vm.replay_h5 = local_h5
                            vm.reload(images_h5=local_h5, frame_index=f_idx)
                        else:
                            disc_img = (
                                vm.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                    local_h5
                                )
                                or _discover_images_h5(local_h5)
                            )
                            vm.payload_kwargs["images_h5"] = (
                                str(disc_img) if disc_img else None
                            )
                            vm.replay_h5 = local_h5
                            vm.reload(
                                images_h5=(str(disc_img) if disc_img else None),
                                frame_index=f_idx,
                            )
                        vm.pipeline_runner.invalidate_subsequent_stages(
                            (vm._payload or {}).get("artifact_kind"),
                            os.path.dirname(os.path.abspath(local_h5)),
                        )
                        reloaded = True
                    except _NOVA_RECOVERABLE_ERRORS as exc:
                        err_body = json.dumps(
                            {"status": "error", "error": str(exc)}
                        ).encode("utf-8")
                        self.send_response(400)
                        self.send_header(
                            "Content-Type", "application/json; charset=utf-8"
                        )
                        self.send_header("Content-Length", str(len(err_body)))
                        self.end_headers()
                        self.wfile.write(err_body)
                        return
                if not reloaded:
                    cur_p = vm._payload
                    cur_f = (
                        cur_p.get("frame_index") if isinstance(cur_p, dict) else None
                    )
                    cur_dmin = cur_p.get("d_min") if isinstance(cur_p, dict) else None
                    if (
                        not isinstance(cur_p, dict)
                        or (f_idx is not None and cur_f != f_idx)
                        or (
                            d_min_val is not None
                            and (
                                cur_dmin is None
                                or abs(float(cur_dmin) - d_min_val) > 1e-3
                            )
                        )
                    ):
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            vm.reload(frame_index=f_idx)
                    elif (
                        isinstance(cur_p, dict) and d_min_val is not None and preserve_q
                    ):
                        cur_p["d_min"] = round(d_min_val, 3)
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    vm.pipeline_runner.apply_active_solve_overlay(
                        vm.payload, frame_index=f_idx
                    )
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
                except _NOVA_RECOVERABLE_ERRORS as exc:
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
                except _NOVA_RECOVERABLE_ERRORS as exc:
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
                if qs.get("frame"):
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        f_idx_st = int(qs["frame"][0])
                        vm.payload_kwargs["frame_index"] = f_idx_st
                        vm.pipeline_runner.sync_frame_index(f_idx_st)
                is_poll = qs.get("poll", ["0"])[0] == "1"
                aj_now = vm.pipeline_runner.active_job
                aj_running = isinstance(aj_now, dict) and aj_now.get("state") in (
                    "queued",
                    "running",
                )
                hist_ds = vm.pipeline_runner.list_history_datasets(
                    use_cached=bool(is_poll and aj_running)
                )
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

        def do_POST(self) -> None:
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
                except _NOVA_RECOVERABLE_ERRORS as exc:
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
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"ok": False, "error": str(exc), "message": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/cancel":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    result = vm.pipeline_runner.cancel_active_job(
                        req_json.get("stage_id")
                    )
                    code = 200
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"ok": False, "error": str(exc)}
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
                    if "wavelength_min" in params and "wavelength_max" in params:
                        params["wavelength_range"] = [
                            float(params["wavelength_min"]),
                            float(params["wavelength_max"]),
                        ]
                    elif vm.payload.get("wavelength_range"):
                        params.setdefault(
                            "wavelength_range", vm.payload.get("wavelength_range")
                        )
                    if vm.payload.get("d_min"):
                        params.setdefault("d_min", vm.payload.get("d_min"))
                    if stage_id != "reduce":
                        if params.get("wavelength_range"):
                            wl_r = params["wavelength_range"]
                            vm.payload["wavelength_range"] = [
                                float(wl_r[0]),
                                float(wl_r[1]),
                            ]
                            vm.payload_kwargs["wavelength_range"] = (
                                float(wl_r[0]),
                                float(wl_r[1]),
                            )
                        if params.get("d_min"):
                            vm.payload["d_min"] = float(params["d_min"])
                            vm.payload_kwargs["d_min"] = float(params["d_min"])
                    cur_imgs = vm.payload_kwargs.get("images_h5") or vm.payload.get(
                        "images_h5"
                    )
                    if not cur_imgs and vm.replay_h5:
                        cur_imgs = (
                            vm.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                vm.replay_h5
                            )
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
                            vm.payload_kwargs["frame_index"] = 0
                            vm.payload_kwargs.pop("wavelength_range", None)
                            vm.payload_kwargs.pop("d_min", None)
                        else:
                            vm.payload_kwargs["frame_index"] = int(
                                vm.pipeline_runner.current_frame_index or 0
                            )
                            if params.get("d_min"):
                                vm.payload_kwargs["d_min"] = float(params["d_min"])
                        vm.replay_h5 = new_h5_path
                        vm.reload()
                        if stage_id != "reduce":
                            if params.get("wavelength_range"):
                                wl_r = params["wavelength_range"]
                                vm.payload["wavelength_range"] = [
                                    float(wl_r[0]),
                                    float(wl_r[1]),
                                ]
                            if params.get("d_min"):
                                vm.payload["d_min"] = float(params["d_min"])
                                vm.payload_kwargs["d_min"] = float(params["d_min"])

                    result = vm.pipeline_runner.submit_galaxy_stage(
                        stage_id, params=params, on_complete_h5=_on_complete
                    )
                    code = 200
                except _NOVA_RECOVERABLE_ERRORS as exc:
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
                    header_bytes = f"Content-Type: {ctype}\r\n\r\n".encode()
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
                except _NOVA_RECOVERABLE_ERRORS as exc:
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
