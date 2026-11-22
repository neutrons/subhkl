"""NOVA Trame View for SubHKL Diffraction Parallax & Lattice Ladder (full-bleed hero layout)."""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

try:
    from nova.mvvm.trame_binding import TrameBinding
    from trame.app import get_server
    from trame.ui.html import DivLayout
    from trame.widgets import html

    from subhkl.viz.nova_app.mvvm_factory import create_viewmodels
except ModuleNotFoundError:  # pragma: no cover - optional `nova` extra
    TrameBinding = None  # type: ignore[assignment,misc]
    get_server = None  # type: ignore[assignment]
    DivLayout = None  # type: ignore[assignment,misc]
    html = None  # type: ignore[assignment]
    create_viewmodels = None  # type: ignore[assignment]

from subhkl.viz.logo_svg import SUBHKL_FAVICON_DATA_URI, SUBHKL_FAVICON_SVG
from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner

if TYPE_CHECKING:
    from trame_server import Server


class MainApp:
    """Full-bleed NOVA Trame application exhausting the browser window with the Diffraction Parallax hero view."""

    def __init__(
        self,
        server: Server | None = None,
        peaks_h5: str | Path | None = None,
        images_h5: str | Path | None = None,
        instrument: str = "CG4D",
    ) -> None:
        import threading

        from subhkl.io.shared_storage import resolve_input_h5_path

        self.server = server or get_server(None, client_type="vue3")
        self.state = self.server.state
        self._peaks_h5 = (
            resolve_input_h5_path(peaks_h5, must_exist=False)[0]
            if peaks_h5 is not None
            else None
        )
        self._images_h5 = (
            resolve_input_h5_path(images_h5, must_exist=False)[0]
            if images_h5 is not None
            else None
        )
        self._instrument = instrument
        self._payload_lock = threading.RLock()
        self.pipeline_runner = NovaGalaxyPipelineRunner()
        if self.pipeline_runner.is_explicit_galaxy_exec_mode():
            self.pipeline_runner.require_shared_dir_for_galaxy_mode()

        self.create_vm()
        sub_init = self.view_model.model.subhkl
        if (
            sub_init.data.peaks_file
            and os.path.isfile(sub_init.data.peaks_file)
            and (sub_init.payload or {}).get("intensity_source") != "raw_h5"
        ):
            with contextlib.suppress(
                OSError, RuntimeError, ValueError, TypeError, KeyError
            ):
                from subhkl.viz.rainbow_zones import _discover_images_h5

                _, staged_init = self.pipeline_runner.stage_derived_h5_in_work_dir(
                    sub_init.data.peaks_file,
                    name_hint=os.path.basename(sub_init.data.peaks_file),
                )
                sub_init.data.peaks_file = staged_init
                disc_init = self.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                    staged_init
                ) or _discover_images_h5(staged_init)
                if disc_init:
                    sub_init.data.images_file = str(disc_init)
                    sub_init.refresh_payload(preserve_params=True)
                if (sub_init.payload or {}).get("intensity_source") != "raw_h5":
                    hyd_init = self.pipeline_runner._maybe_trigger_hydrate_reduce(
                        sub_init,
                        self._payload_lock,
                        staged_init,
                    )
                    if hyd_init and isinstance(sub_init.payload, dict):
                        sub_init.payload["hydration_job"] = hyd_init
        self.view_model.config_bind.connect("config")
        self.view_model.view_state_bind.connect("view_state")
        self.view_model.parallax_html_bind.connect("parallax_html")
        self.server.controller.on_server_bind.add(self._bind_routes)
        self.create_ui()
        self.view_model.update_view()

    def _bind_routes(self, wslink_server: Any) -> None:
        """Bind JSON endpoints for live Galaxy/Pulsar job dispatch and stdout/stderr streaming."""
        app = getattr(wslink_server, "app", None)
        if app is None:
            return

        import asyncio

        payload_lock = getattr(self, "_payload_lock", None)
        if payload_lock is None:
            import threading

            payload_lock = threading.RLock()
            self._payload_lock = payload_lock

        def _stage_derived_safely(h5_p: str, *, name_hint: str = "") -> tuple[str, str]:
            default_d = os.path.dirname(os.path.abspath(h5_p))
            with contextlib.suppress(
                OSError, RuntimeError, ValueError, TypeError, AttributeError
            ):
                res = self.pipeline_runner.stage_derived_h5_in_work_dir(
                    h5_p, name_hint=name_hint
                )
                if (
                    isinstance(res, tuple)
                    and len(res) == 2
                    and isinstance(res[0], str)
                    and isinstance(res[1], str)
                    and os.path.isfile(res[1])
                ):
                    return res
            return default_d, h5_p

        def _valid_h5_str(cand: Any) -> str | None:
            if isinstance(cand, (str, Path)) and os.path.isfile(str(cand)):
                return str(cand)
            return None

        def _build_status_doc(
            frame_q: str | None,
            is_poll: bool,
            overlay_seq_q: str | None = None,
        ) -> dict[str, Any]:
            f_idx_st: int | None = None
            if frame_q is not None and str(frame_q).strip() != "":
                with contextlib.suppress(
                    ValueError, TypeError, RuntimeError, AttributeError
                ):
                    f_idx_st = int(frame_q)
                    self.view_model.model.subhkl.frame_index = f_idx_st
                    self.pipeline_runner.sync_frame_index(f_idx_st)
            ov_seq_client: int | None = None
            if overlay_seq_q is not None and str(overlay_seq_q).strip() != "":
                with contextlib.suppress(ValueError, TypeError):
                    ov_seq_client = int(overlay_seq_q)
            payload = self.view_model.model.subhkl.payload or {}
            aj_now = self.pipeline_runner.active_job
            aj_running = isinstance(aj_now, dict) and aj_now.get("state") in (
                "queued",
                "running",
            )
            hist_ds = self.pipeline_runner.list_history_datasets(
                use_cached=bool(is_poll and aj_running)
            )
            cur_h5 = (
                self.view_model.model.subhkl.data.peaks_file
                or self.view_model.model.subhkl.data.images_file
            )
            step_timings_snap = self.pipeline_runner.get_step_timings_snapshot()
            aj_resp = self.pipeline_runner.active_job
            if (
                is_poll
                and ov_seq_client is not None
                and isinstance(aj_resp, dict)
                and isinstance(aj_resp.get("partial_overlay"), dict)
            ):
                ov_now = aj_resp["partial_overlay"]
                ov_seq_now = int(ov_now.get("seq") or 0)
                ov_frame_now = ov_now.get("frame_index")
                if ov_seq_now == ov_seq_client and (
                    f_idx_st is None
                    or ov_frame_now is None
                    or int(ov_frame_now) == f_idx_st
                ):
                    aj_resp = dict(aj_resp)
                    aj_resp["partial_overlay"] = {
                        "seq": ov_seq_now,
                        "frame_index": ov_frame_now,
                        "unchanged": True,
                    }
            return {
                "galaxy_connected": self.pipeline_runner.is_connected,
                "galaxy_url": self.pipeline_runner.galaxy_url,
                "history_id": self.pipeline_runner.history_id,
                "history_datasets": hist_ds,
                "step_artifacts": self.pipeline_runner.list_step_artifacts(
                    current_h5=cur_h5,
                    history_datasets=hist_ds,
                ),
                "active_job": aj_resp,
                "step_timings": step_timings_snap,
                "server_time": time.time(),
                "pipeline_stages": payload.get("pipeline_stages", []),
                "ladder_steps": payload.get("ladder_steps", []),
            }

        async def handle_status(request: web.Request) -> web.Response:
            frame_q = request.rel_url.query.get("frame")
            is_poll = request.rel_url.query.get("poll") == "1"
            overlay_seq_q = request.rel_url.query.get("overlay_seq")
            doc = await asyncio.to_thread(
                _build_status_doc, frame_q, is_poll, overlay_seq_q
            )
            return web.json_response(doc)

        async def handle_artifact_download(request: web.Request) -> web.Response:
            ds_id = str(request.rel_url.query.get("dataset_id") or "").strip()
            fname = str(request.rel_url.query.get("name") or "").strip()
            cur_h5 = (
                self.view_model.model.subhkl.data.peaks_file
                or self.view_model.model.subhkl.data.images_file
            )
            try:
                local_fp, dl_name = await asyncio.to_thread(
                    self.pipeline_runner.resolve_artifact_for_download,
                    dataset_id=ds_id,
                    name=fname,
                    current_h5=cur_h5,
                )
                safe_dl = dl_name.replace('"', "")
                return web.FileResponse(
                    path=local_fp,
                    headers={
                        "Content-Disposition": f'attachment; filename="{safe_dl}"'
                    },
                )
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                return web.Response(text=str(exc), status=404)

        latest_frame_only_target: list[int | None] = [None]

        def _compute_payload_response(
            query_dict: dict[str, str],
        ) -> tuple[dict[str, Any], int]:
            if query_dict.get("textures_only") == "1":
                from subhkl.viz.rainbow_zones import build_rainbow_panel_textures_only

                f_idx_tex: int | None = None
                frame_q_tex = query_dict.get("frame")
                if frame_q_tex is not None and str(frame_q_tex).strip() != "":
                    with contextlib.suppress(ValueError, TypeError):
                        f_idx_tex = int(frame_q_tex)
                rb_tex = int(self.view_model.model.subhkl.render_binning or 4)
                rbin_q_tex = query_dict.get("render_binning")
                if rbin_q_tex is not None and str(rbin_q_tex).strip() != "":
                    with contextlib.suppress(ValueError, TypeError):
                        parsed_rb_tex = int(rbin_q_tex)
                        if 1 <= parsed_rb_tex <= 16:
                            rb_tex = parsed_rb_tex
                            self.view_model.model.subhkl.render_binning = parsed_rb_tex
                sub_m = self.view_model.model.subhkl
                tex_doc = build_rainbow_panel_textures_only(
                    replay_h5=sub_m.data.peaks_file or sub_m.data.images_file,
                    images_h5=sub_m.data.images_file,
                    instrument=sub_m.data.instrument or "CG4D",
                    render_binning=rb_tex,
                    frame_index=(
                        f_idx_tex
                        if f_idx_tex is not None
                        else int(sub_m.frame_index or 0)
                    ),
                )
                if payload_lock.acquire(blocking=False):
                    try:
                        cur_pl = sub_m.payload
                        if isinstance(cur_pl, dict) and isinstance(
                            cur_pl.get("panels"), list
                        ):
                            cur_pl["render_binning"] = tex_doc["render_binning"]
                            if int(cur_pl.get("frame_index") or 0) == int(
                                tex_doc.get("frame_index") or 0
                            ):
                                by_bank = {
                                    int(p["bank"]): p
                                    for p in tex_doc.get("panels", [])
                                    if "bank" in p
                                }
                                for cp in cur_pl["panels"]:
                                    np_tex = by_bank.get(int(cp.get("bank", -1)))
                                    if np_tex:
                                        if np_tex.get("texture_b64"):
                                            cp["texture_b64"] = np_tex["texture_b64"]
                                        if np_tex.get("mask_b64"):
                                            cp["mask_b64"] = np_tex["mask_b64"]
                    finally:
                        payload_lock.release()
                return (tex_doc, 200)

            frame_only_q = query_dict.get("frame_only") == "1"
            f_idx_pre: int | None = None
            frame_q_pre = query_dict.get("frame")
            if frame_q_pre is not None and str(frame_q_pre).strip() != "":
                with contextlib.suppress(ValueError, TypeError):
                    f_idx_pre = int(frame_q_pre)
            if frame_only_q and f_idx_pre is not None:
                latest_frame_only_target[0] = f_idx_pre

            with payload_lock:
                if (
                    frame_only_q
                    and f_idx_pre is not None
                    and latest_frame_only_target[0] not in (None, f_idx_pre)
                ):
                    return (
                        {
                            "status": "superseded",
                            "frame_only": True,
                            "frame_index": f_idx_pre,
                        },
                        200,
                    )
                frame_q = query_dict.get("frame")
                f_idx: int | None = None
                if frame_q is not None and str(frame_q).strip() != "":
                    try:
                        f_idx = int(frame_q)
                        self.view_model.model.subhkl.frame_index = f_idx
                        self.pipeline_runner.current_frame_index = f_idx
                    except (ValueError, TypeError):
                        f_idx = None

                refreshed = False
                preserve_q = query_dict.get("preserve_params") == "1"
                d_min_q = query_dict.get("d_min")
                d_min_val: float | None = None
                if d_min_q is not None and str(d_min_q).strip() != "":
                    with contextlib.suppress(ValueError, TypeError):
                        parsed_dmin = float(d_min_q)
                        if 0.2 <= parsed_dmin <= 20.0:
                            d_min_val = parsed_dmin
                            self.view_model.model.subhkl.data.d_min = parsed_dmin
                rbin_q = query_dict.get("render_binning")
                rbin_val: int | None = None
                if rbin_q is not None and str(rbin_q).strip() != "":
                    with contextlib.suppress(ValueError, TypeError):
                        parsed_rb = int(rbin_q)
                        if 1 <= parsed_rb <= 16:
                            rbin_val = parsed_rb
                            self.view_model.model.subhkl.render_binning = parsed_rb

                fp_q = query_dict.get("file_path")
                if fp_q is not None and str(fp_q).strip() != "":
                    local_fp_raw = os.path.abspath(str(fp_q).strip())
                    if local_fp_raw.lower().endswith(".mtz"):
                        if not os.path.isfile(local_fp_raw):
                            return (
                                {
                                    "status": "error",
                                    "error": f"MTZ file not found: {local_fp_raw}",
                                },
                                404,
                            )
                        try:
                            from subhkl.viz.rainbow_zones import apply_mtz_to_payload

                            ds_info_mtz = {"name": os.path.basename(local_fp_raw)}
                            sub_model = self.view_model.model.subhkl
                            sub_model.payload = apply_mtz_to_payload(
                                sub_model.payload or {}, local_fp_raw, ds_info_mtz
                            )
                            sub_model.mtz_overlay_path = local_fp_raw
                            sub_model.mtz_overlay_info = ds_info_mtz
                            return (sub_model.payload, 200)
                        except (
                            OSError,
                            RuntimeError,
                            ValueError,
                            TypeError,
                            KeyError,
                        ) as exc:
                            return ({"status": "error", "error": str(exc)}, 400)
                    try:
                        local_h5 = local_fp_raw
                        if os.path.isfile(local_h5):
                            ds_name = os.path.basename(local_h5).lower()
                            is_reduce_file = (
                                any(k in ds_name for k in ("merged", "scan", "image"))
                                and "careless" not in ds_name
                            )
                            if is_reduce_file:
                                self.view_model.model.subhkl.mtz_overlay_path = None
                                self.view_model.model.subhkl.mtz_overlay_info = None
                            if is_reduce_file:
                                _, local_h5 = (
                                    self.pipeline_runner._resolve_work_dir_and_merged(
                                        local_h5
                                    )
                                )
                                self.view_model.model.subhkl.data.images_file = local_h5
                                self.view_model.model.subhkl.data.peaks_file = local_h5
                            else:
                                from subhkl.viz.rainbow_zones import _discover_images_h5

                                work_d_fp, staged_fp = _stage_derived_safely(
                                    local_h5, name_hint=ds_name
                                )
                                local_h5 = staged_fp
                                self.view_model.model.subhkl.data.peaks_file = staged_fp
                                disc_img = _valid_h5_str(
                                    _discover_images_h5(staged_fp)
                                ) or _valid_h5_str(
                                    self.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                        staged_fp
                                    )
                                )
                                self.pipeline_runner._discover_or_fetch_galaxy_predictor_h5(
                                    staged_fp
                                )
                                self.view_model.model.subhkl.data.images_file = (
                                    disc_img or ""
                                )
                            self.view_model.model.subhkl.refresh_payload(
                                frame_index=f_idx,
                                preserve_params=bool(
                                    (preserve_q or d_min_val is not None)
                                    and not is_reduce_file
                                ),
                            )
                            self.pipeline_runner.invalidate_subsequent_stages(
                                (self.view_model.model.subhkl.payload or {}).get(
                                    "artifact_kind"
                                ),
                                (
                                    os.path.dirname(os.path.abspath(local_h5))
                                    if is_reduce_file
                                    else work_d_fp
                                ),
                            )
                            if (
                                not is_reduce_file
                                and (self.view_model.model.subhkl.payload or {}).get(
                                    "intensity_source"
                                )
                                != "raw_h5"
                            ):
                                hyd_fp = (
                                    self.pipeline_runner._maybe_trigger_hydrate_reduce(
                                        self.view_model.model.subhkl,
                                        payload_lock,
                                        local_h5,
                                    )
                                )
                                if (
                                    isinstance(hyd_fp, dict)
                                    and hyd_fp.get("job_id")
                                    and isinstance(
                                        self.view_model.model.subhkl.payload, dict
                                    )
                                ):
                                    self.view_model.model.subhkl.payload[
                                        "hydration_job"
                                    ] = hyd_fp
                            refreshed = True
                    except (
                        OSError,
                        RuntimeError,
                        ValueError,
                        TypeError,
                        KeyError,
                    ) as exc:
                        return ({"status": "error", "error": str(exc)}, 404)

                ds_id_q = query_dict.get("dataset_id")
                if ds_id_q is not None and str(ds_id_q).strip() != "":
                    try:
                        from subhkl.viz.rainbow_zones import (
                            _discover_images_h5,
                            apply_mtz_to_payload,
                            validate_subhkl_ingestible_dataset,
                        )

                        ds_id = str(ds_id_q).strip()
                        cur_peaks = self.view_model.model.subhkl.data.peaks_file
                        cur_imgs = self.view_model.model.subhkl.data.images_file
                        fb_dir = None
                        for p in (cur_peaks, cur_imgs):
                            if p and os.path.isfile(p):
                                fb_dir = os.path.dirname(os.path.abspath(p))
                                break
                        local_h5, ds_info = (
                            self.pipeline_runner.resolve_dataset_to_local_h5(
                                ds_id, fallback_dir=fb_dir
                            )
                        )
                        cur_p = self.view_model.model.subhkl.payload or {}
                        has_active_geom = bool(
                            (
                                cur_p.get("artifact_kind") != "empty"
                                and cur_p.get("panels")
                                and (cur_p.get("peaks") or cur_p.get("ub_matrix"))
                            )
                            or query_dict.get("has_active_geometry") == "1"
                        )
                        ingest_kind = validate_subhkl_ingestible_dataset(
                            local_h5,
                            ds_info,
                            has_active_geometry=has_active_geom,
                        )
                        if ingest_kind == "mtz_overlay":
                            sub_model = self.view_model.model.subhkl
                            sub_model.payload = apply_mtz_to_payload(
                                cur_p, local_h5, ds_info
                            )
                            sub_model.mtz_overlay_path = local_h5
                            sub_model.mtz_overlay_info = ds_info
                            return (sub_model.payload, 200)
                        if ingest_kind == "raw_nexus":
                            self.view_model.model.subhkl.mtz_overlay_path = None
                            self.view_model.model.subhkl.mtz_overlay_info = None
                            return (
                                {
                                    "status": "raw_nexus",
                                    "dataset_id": ds_id,
                                    "name": ds_info.get("name")
                                    or os.path.basename(local_h5),
                                },
                                200,
                            )
                        ds_name = str(
                            ds_info.get("name") or os.path.basename(local_h5)
                        ).lower()
                        is_reduce_ds = (
                            any(k in ds_name for k in ("merged", "scan", "image"))
                            and "careless" not in ds_name
                        )
                        if is_reduce_ds:
                            self.view_model.model.subhkl.mtz_overlay_path = None
                            self.view_model.model.subhkl.mtz_overlay_info = None
                            work_d_ds, local_h5 = (
                                self.pipeline_runner._resolve_work_dir_and_merged(
                                    local_h5
                                )
                            )
                            self.view_model.model.subhkl.data.images_file = local_h5
                            self.view_model.model.subhkl.data.peaks_file = local_h5
                        else:
                            work_d_ds, staged_ds_h5 = _stage_derived_safely(
                                local_h5, name_hint=ds_name
                            )
                            local_h5 = staged_ds_h5
                            self.view_model.model.subhkl.data.peaks_file = staged_ds_h5
                            disc_img = _valid_h5_str(
                                self.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                    staged_ds_h5
                                )
                            ) or _valid_h5_str(_discover_images_h5(staged_ds_h5))
                            ds_hid_raw = (ds_info or {}).get("hid")
                            ds_hid = (
                                int(ds_hid_raw)
                                if isinstance(ds_hid_raw, (int, str))
                                and str(ds_hid_raw).isdigit()
                                else None
                            )
                            self.pipeline_runner._discover_or_fetch_galaxy_predictor_h5(
                                staged_ds_h5, target_hid=ds_hid
                            )
                            self.view_model.model.subhkl.data.images_file = (
                                disc_img or ""
                            )
                        self.view_model.model.subhkl.refresh_payload(frame_index=f_idx)
                        self.pipeline_runner.invalidate_subsequent_stages(
                            (self.view_model.model.subhkl.payload or {}).get(
                                "artifact_kind"
                            ),
                            work_d_ds,
                        )
                        active_img = str(
                            self.view_model.model.subhkl.data.images_file or ""
                        )
                        self.pipeline_runner.release_shared_scan_intermediates(
                            keep_work_dir=work_d_ds,
                            keep_paths={os.path.realpath(active_img)}
                            if active_img and os.path.isfile(active_img)
                            else set(),
                        )
                        if (
                            not is_reduce_ds
                            and (self.view_model.model.subhkl.payload or {}).get(
                                "intensity_source"
                            )
                            != "raw_h5"
                        ):
                            hyd_ds = self.pipeline_runner._maybe_trigger_hydrate_reduce(
                                self.view_model.model.subhkl,
                                payload_lock,
                                local_h5,
                                ds_info=ds_info,
                            )
                            if (
                                isinstance(hyd_ds, dict)
                                and hyd_ds.get("job_id")
                                and isinstance(
                                    self.view_model.model.subhkl.payload, dict
                                )
                            ):
                                self.view_model.model.subhkl.payload[
                                    "hydration_job"
                                ] = hyd_ds
                        refreshed = True
                    except (
                        OSError,
                        RuntimeError,
                        ValueError,
                        TypeError,
                        KeyError,
                    ) as exc:
                        return ({"status": "error", "error": str(exc)}, 400)

                if not refreshed:
                    from subhkl.viz.rainbow_zones import (
                        _discover_images_h5,
                        build_frame_only_response,
                    )

                    cur_payload = self.view_model.model.subhkl.payload
                    cur_f = (
                        cur_payload.get("frame_index")
                        if isinstance(cur_payload, dict)
                        else None
                    )
                    cur_dmin = (
                        cur_payload.get("d_min")
                        if isinstance(cur_payload, dict)
                        else None
                    )
                    cur_rb = (
                        int(cur_payload.get("render_binning", 4))
                        if isinstance(cur_payload, dict)
                        else None
                    )
                    newly_hydrated = (
                        isinstance(cur_payload, dict)
                        and cur_payload.get("intensity_source") != "raw_h5"
                        and bool(
                            self.view_model.model.subhkl.data.peaks_file
                            and _discover_images_h5(
                                self.view_model.model.subhkl.data.peaks_file
                            )
                        )
                    )
                    fast_frame_only = bool(
                        frame_only_q
                        and isinstance(cur_payload, dict)
                        and not newly_hydrated
                        and (
                            cur_payload.get("sample_peaks")
                            or cur_payload.get("integrated_sample_peaks")
                            or cur_payload.get("predicted_sample_peaks")
                            or len(cur_payload.get("frames") or ()) <= 1
                        )
                    )
                    needs_refresh = (
                        not isinstance(cur_payload, dict)
                        or newly_hydrated
                        or (f_idx is not None and cur_f != f_idx)
                        or (rbin_val is not None and cur_rb != rbin_val)
                        or (
                            d_min_val is not None
                            and (
                                cur_dmin is None
                                or abs(float(cur_dmin) - d_min_val) > 1e-3
                            )
                        )
                    )
                    if needs_refresh and (
                        newly_hydrated
                        or f_idx is not None
                        or d_min_val is not None
                        or rbin_val is not None
                    ):
                        with contextlib.suppress(
                            OSError, RuntimeError, ValueError, TypeError, KeyError
                        ):
                            self.view_model.model.subhkl.refresh_payload(
                                frame_index=f_idx,
                                preserve_params=True,
                                include_all_runs_peaks=not fast_frame_only,
                            )
                    elif (
                        isinstance(cur_payload, dict)
                        and d_min_val is not None
                        and preserve_q
                    ):
                        cur_payload["d_min"] = round(d_min_val, 3)
                else:
                    fast_frame_only = False
                if isinstance(self.view_model.model.subhkl.payload, dict):
                    with contextlib.suppress(
                        OSError, RuntimeError, ValueError, TypeError, KeyError
                    ):
                        self.pipeline_runner.apply_active_solve_overlay(
                            self.view_model.model.subhkl.payload,
                            frame_index=f_idx,
                        )
                    if fast_frame_only:
                        from subhkl.viz.rainbow_zones import build_frame_only_response

                        return (
                            build_frame_only_response(
                                self.view_model.model.subhkl.payload
                            ),
                            200,
                        )
                return (self.view_model.model.subhkl.payload or {}, 200)

        async def handle_payload(request: web.Request) -> web.Response:
            q_snap = dict(request.rel_url.query)
            payload_doc, status_code = await asyncio.to_thread(
                _compute_payload_response, q_snap
            )
            return web.json_response(
                payload_doc,
                status=status_code,
                dumps=lambda obj: json.dumps(obj, separators=(",", ":")),
            )

        def _compute_tof_delta_theta_response(
            req_json: dict[str, Any],
        ) -> tuple[dict[str, Any], int]:
            try:
                from subhkl.viz.rainbow_zones import (
                    compute_live_tof_delta_theta_payload,
                )

                sub_m = self.view_model.model.subhkl
                cur_pl = sub_m.payload if isinstance(sub_m.payload, dict) else {}
                cur_imgs = sub_m.data.images_file or cur_pl.get("images_h5")
                cur_replay = sub_m.data.peaks_file
                raw_inst = getattr(
                    sub_m.data.instrument, "value", sub_m.data.instrument
                )
                def_inst = str(cur_pl.get("instrument") or raw_inst or "IMAGINE")
                def_sg = str(
                    cur_pl.get("space_group") or sub_m.data.space_group or "P 1"
                )
                def_cell = cur_pl.get("cell")
                def_wl = cur_pl.get("wavelength_range") or [
                    float(sub_m.data.wavelength_min),
                    float(sub_m.data.wavelength_max),
                ]
                def_dmin = float(cur_pl.get("d_min") or sub_m.data.d_min or 1.5)
                def_ki = cur_pl.get("ki") or [0.0, 0.0, 1.0]
                if not cur_imgs and cur_replay:
                    cur_imgs = self.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                        cur_replay
                    )
                wl_in = req_json.get("wavelength_range") or def_wl
                wl_tup = (float(wl_in[0]), float(wl_in[1]))
                result = compute_live_tof_delta_theta_payload(
                    images_h5=cur_imgs,
                    replay_h5=cur_replay,
                    frame_index=int(req_json.get("frame_index") or 0),
                    d_min=float(req_json.get("d_min") or def_dmin),
                    wavelength_range=wl_tup,
                    ub_matrix=req_json.get("ub_matrix"),
                    R_run=req_json.get("R_run"),
                    sample_offset_lab=req_json.get("sample_offset_lab"),
                    ki=req_json.get("ki") or def_ki,
                    instrument=str(req_json.get("instrument") or def_inst),
                    space_group=str(req_json.get("space_group") or def_sg),
                    cell=req_json.get("cell") or def_cell,
                    detector_configs=req_json.get("detector_configs"),
                    tof_l1=req_json.get("tof_l1"),
                    zone_axes_crystal=req_json.get("zone_axes_crystal"),
                    zone_tolerance_deg=float(
                        req_json.get("zone_tolerance_deg") or 0.75
                    ),
                )
                return (result, 200 if result.get("ok") else 400)
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
            ) as exc:
                return ({"ok": False, "error": str(exc)}, 400)

        async def handle_tof_delta_theta(request: web.Request) -> web.Response:
            try:
                body = await request.json()
                if not isinstance(body, dict):
                    body = {}
            except (OSError, RuntimeError, ValueError, TypeError):
                body = {}
            res_doc, status_code = await asyncio.to_thread(
                _compute_tof_delta_theta_response, body
            )
            return web.json_response(
                res_doc,
                status=status_code,
                dumps=lambda obj: json.dumps(obj, separators=(",", ":")),
            )

        async def handle_dataset_display(request: web.Request) -> web.Response:
            ds_id = request.match_info.get("ds_id", "")
            try:
                cur_peaks = self.view_model.model.subhkl.data.peaks_file
                cur_imgs = self.view_model.model.subhkl.data.images_file
                fb_dir = None
                for p in (cur_peaks, cur_imgs):
                    if p and os.path.isfile(p):
                        fb_dir = os.path.dirname(os.path.abspath(p))
                        break
                local_h5, _ = await asyncio.to_thread(
                    self.pipeline_runner.resolve_dataset_to_local_h5,
                    ds_id,
                    fallback_dir=fb_dir,
                )
                return web.FileResponse(path=local_h5)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                return web.Response(text=str(exc), status=404)

        async def handle_standalone_html(_request: web.Request) -> web.Response:
            html_doc = self.view_model.model.subhkl.get_widget_html()
            return web.Response(text=html_doc, content_type="text/html")

        async def handle_run(request: web.Request) -> web.Response:
            try:
                body = await request.json()
            except (OSError, RuntimeError, ValueError, TypeError):
                body = {}
            stage_id = body.get("stage_id", "solve")
            params = dict(body.get("params") or {})
            data = self.view_model.model.subhkl.data
            params.setdefault("instrument", data.instrument.value)
            params.setdefault("space_group", data.space_group)
            params["cell"] = [
                float(params.get("a", data.a)),
                float(params.get("b", data.b)),
                float(params.get("c", data.c)),
                float(params.get("alpha", data.alpha)),
                float(params.get("beta", data.beta)),
                float(params.get("gamma", data.gamma)),
            ]
            wl_min = float(params.get("wavelength_min", data.wavelength_min))
            wl_max = float(params.get("wavelength_max", data.wavelength_max))
            params["wavelength_range"] = [wl_min, wl_max]
            params.setdefault("d_min", data.d_min)
            d_min_req = float(params.get("d_min", data.d_min))
            if stage_id != "reduce":
                with contextlib.suppress(
                    ValueError, TypeError, KeyError, AttributeError
                ):
                    data.wavelength_min = wl_min
                    data.wavelength_max = wl_max
                    data.d_min = d_min_req
                    if isinstance(self.view_model.model.subhkl.payload, dict):
                        self.view_model.model.subhkl.payload["wavelength_range"] = [
                            wl_min,
                            wl_max,
                        ]
                        self.view_model.model.subhkl.payload["d_min"] = d_min_req
            params.setdefault("n_zones", data.n_zones)
            params.setdefault("peaks_file", data.peaks_file)
            params.setdefault("images_file", data.images_file)

            def _on_complete(new_h5_path: str) -> None:
                from subhkl.viz.nova_app.models.subhkl_model import InstrumentOptions

                with payload_lock:
                    new_base_low = os.path.basename(new_h5_path).lower()
                    is_reduce_out = stage_id == "reduce" or (
                        "merged" in new_base_low and "careless" not in new_base_low
                    )
                    if is_reduce_out:
                        self.view_model.model.subhkl.mtz_overlay_path = None
                        self.view_model.model.subhkl.mtz_overlay_info = None

                    with contextlib.suppress(
                        ValueError, TypeError, KeyError, AttributeError
                    ):
                        if (
                            "instrument" in params
                            and str(params["instrument"])
                            in InstrumentOptions.__members__
                        ):
                            data.instrument = InstrumentOptions(
                                str(params["instrument"])
                            )
                        for attr in ("a", "b", "c", "alpha", "beta", "gamma"):
                            if attr in params:
                                setattr(data, attr, float(params[attr]))
                        if "space_group" in params:
                            data.space_group = str(params["space_group"])
                        if "wavelength_min" in params:
                            data.wavelength_min = float(params["wavelength_min"])
                        if "wavelength_max" in params:
                            data.wavelength_max = float(params["wavelength_max"])
                        if "d_min" in params:
                            data.d_min = float(params["d_min"])
                    if is_reduce_out:
                        self.view_model.model.subhkl.frame_index = 0
                        self.pipeline_runner.current_frame_index = 0
                        self.view_model.model.subhkl.data.images_file = new_h5_path
                        self.view_model.model.subhkl.data.peaks_file = new_h5_path
                    else:
                        self.view_model.model.subhkl.frame_index = int(
                            self.pipeline_runner.current_frame_index or 0
                        )
                        self.view_model.model.subhkl.data.peaks_file = new_h5_path
                    self.view_model.model.subhkl.refresh_payload(
                        frame_index=self.view_model.model.subhkl.frame_index,
                        preserve_params=not is_reduce_out,
                    )

            try:
                res = self.pipeline_runner.submit_galaxy_stage(
                    stage_id, params=params, on_complete_h5=_on_complete
                )
                return web.json_response(res)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                return web.json_response(
                    {"status": "error", "error": str(exc)}, status=400
                )

        async def handle_cancel(request: web.Request) -> web.Response:
            import asyncio

            try:
                body = await request.json()
            except (OSError, RuntimeError, ValueError, TypeError):
                body = {}
            res = await asyncio.to_thread(
                self.pipeline_runner.cancel_active_job, body.get("stage_id")
            )
            return web.json_response(res)

        async def handle_upload_h5(request: web.Request) -> web.Response:
            import tempfile
            import time

            try:
                from subhkl.viz.nova_rainbow_view import register_temp_artifact

                upload_dir = os.path.join(
                    tempfile.gettempdir(), f"subhkl_upload_{int(time.time())}"
                )
                os.makedirs(upload_dir, exist_ok=True)
                register_temp_artifact(upload_dir)
                reader = await request.multipart()
                saved_paths: list[str] = []
                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    fname = part.filename
                    if not fname:
                        await part.read()
                        continue
                    safe_name = os.path.basename(fname)
                    dest_path = Path(upload_dir) / safe_name
                    payload_bytes = bytearray()
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        payload_bytes.extend(chunk)
                    dest_path.write_bytes(payload_bytes)
                    saved_paths.append(str(dest_path))
                if len(saved_paths) == 1 and not saved_paths[0].lower().endswith(
                    (".nxs.h5", ".nxs", ".mtz")
                ):
                    sp = saved_paths[0]
                    self.view_model.model.subhkl.mtz_overlay_path = None
                    self.view_model.model.subhkl.mtz_overlay_info = None
                    if (
                        "merged" in os.path.basename(sp).lower()
                        or "scan" in os.path.basename(sp).lower()
                    ):
                        self.view_model.model.subhkl.data.images_file = sp
                    self.view_model.model.subhkl.data.peaks_file = sp
                    self.view_model.model.subhkl.refresh_payload()
                return web.json_response({"status": "ok", "saved_paths": saved_paths})
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                return web.json_response(
                    {"status": "error", "error": str(exc)}, status=400
                )

        async def handle_resolve_local_files(request: web.Request) -> web.Response:
            import glob as _glob
            import tempfile

            try:
                body = await request.json()
            except (OSError, RuntimeError, ValueError, TypeError):
                body = {}
            req_files = body.get("files") or []
            matched_paths: list[str] = []
            cur_peaks = self.view_model.model.subhkl.data.peaks_file
            cur_imgs = self.view_model.model.subhkl.data.images_file
            search_dirs: list[str] = []
            for p in (cur_peaks, cur_imgs):
                if p and os.path.isfile(p):
                    d = os.path.dirname(os.path.abspath(p))
                    if d not in search_dirs:
                        search_dirs.append(d)
            for extra_root in ("/home/xvg/subhkl-examples", tempfile.gettempdir()):
                if os.path.isdir(extra_root) and extra_root not in search_dirs:
                    search_dirs.append(extra_root)

            for fmeta in req_files:
                if not isinstance(fmeta, dict):
                    continue
                fname = os.path.basename(str(fmeta.get("name") or "").strip())
                fsize = int(fmeta.get("size") or -1)
                if not fname or fsize < 0:
                    break
                found: str | None = None
                for sdir in search_dirs:
                    direct = os.path.join(sdir, fname)
                    if os.path.isfile(direct) and os.path.getsize(direct) == fsize:
                        found = direct
                        break
                    for sub_cand in _glob.glob(os.path.join(sdir, "*", fname)):
                        if (
                            os.path.isfile(sub_cand)
                            and os.path.getsize(sub_cand) == fsize
                        ):
                            found = sub_cand
                            break
                    if found:
                        break
                if found:
                    matched_paths.append(found)
            return web.json_response(
                {
                    "matched": len(matched_paths) == len(req_files)
                    and len(matched_paths) > 0,
                    "matched_paths": matched_paths,
                }
            )

        async def handle_check_neutron_paths(request: web.Request) -> web.Response:
            try:
                body = await request.json()
            except (OSError, RuntimeError, ValueError, TypeError):
                body = {}
            res = self.pipeline_runner.check_neutron_paths(body)
            return web.json_response(res)

        async def handle_oncat_catalog(request: web.Request) -> web.Response:
            import asyncio

            q = request.rel_url.query
            params = {
                "facility": str(q.get("facility") or "HFIR").strip(),
                "instrument": str(q.get("instrument") or "CG4D").strip(),
                "ipts": str(q.get("ipts") or "").strip(),
            }
            res = await asyncio.to_thread(
                self.pipeline_runner.browse_oncat_catalog, params
            )
            return web.json_response(res)

        async def handle_oncat_auth_get(request: web.Request) -> web.Response:
            import asyncio

            res = await asyncio.to_thread(self.pipeline_runner.get_oncat_auth_status)
            return web.json_response(res)

        async def handle_oncat_auth_post(request: web.Request) -> web.Response:
            import asyncio

            try:
                body = await request.json()
            except (OSError, RuntimeError, ValueError, TypeError):
                body = {}
            res = await asyncio.to_thread(self.pipeline_runner.handle_oncat_auth, body)
            return web.json_response(res)

        app._client_max_size = 4 * 1024**3
        app.add_routes(
            [
                web.get("/api/pipeline/status", handle_status),
                web.get("/api/pipeline/oncat_catalog", handle_oncat_catalog),
                web.get("/api/pipeline/oncat_auth", handle_oncat_auth_get),
                web.post("/api/pipeline/oncat_auth", handle_oncat_auth_post),
                web.get("/api/artifacts/download", handle_artifact_download),
                web.get("/api/payload", handle_payload),
                web.post("/api/tof_delta_theta", handle_tof_delta_theta),
                web.get("/api/datasets/{ds_id}/display", handle_dataset_display),
                web.post("/api/pipeline/run", handle_run),
                web.post("/api/pipeline/cancel", handle_cancel),
                web.post("/api/pipeline/upload_h5", handle_upload_h5),
                web.post(
                    "/api/pipeline/resolve_local_files", handle_resolve_local_files
                ),
                web.post(
                    "/api/pipeline/check_neutron_paths", handle_check_neutron_paths
                ),
                web.get("/subhkl_parallax.html", handle_standalone_html),
            ]
        )

        orig_handle = app._handle

        async def _ep_handle(request: web.Request) -> web.StreamResponse:
            ep_path = os.environ.get("EP_PATH", "").strip().rstrip("/")
            req_path = request.rel_url.path
            if req_path.endswith(("/logo.png", "/favicon.ico", "/favicon.svg")):
                return web.Response(
                    text=SUBHKL_FAVICON_SVG, content_type="image/svg+xml"
                )
            served_index = False
            if ep_path and ep_path.startswith("/"):
                ep_parent = os.path.dirname(ep_path)
                if req_path == ep_path or req_path == ep_path + "/":
                    request = request.clone(
                        rel_url=request.rel_url.with_path(
                            "/index.html", keep_query=True, keep_fragment=True
                        )
                    )
                    served_index = True
                elif req_path.startswith(ep_path + "/"):
                    sub_path = req_path[len(ep_path) :] or "/"
                    if sub_path == "/":
                        sub_path = "/index.html"
                        served_index = True
                    elif sub_path == "/index.html":
                        served_index = True
                    request = request.clone(
                        rel_url=request.rel_url.with_path(
                            sub_path, keep_query=True, keep_fragment=True
                        )
                    )
                elif ep_parent not in ("", "/") and req_path.startswith(
                    ep_parent + "/"
                ):
                    sub_path = req_path[len(ep_parent) :] or "/"
                    request = request.clone(
                        rel_url=request.rel_url.with_path(
                            sub_path, keep_query=True, keep_fragment=True
                        )
                    )
            elif req_path in ("/", "/index.html"):
                request = request.clone(
                    rel_url=request.rel_url.with_path(
                        "/index.html", keep_query=True, keep_fragment=True
                    )
                )
                served_index = True

            resp = await orig_handle(request)
            if served_index and isinstance(resp, web.FileResponse):
                with contextlib.suppress(OSError, RuntimeError, ValueError, TypeError):
                    idx_fp = getattr(resp, "_path", None)
                    if idx_fp and os.path.isfile(idx_fp):
                        html_txt = Path(idx_fp).read_text(encoding="utf-8")
                        if ep_path:
                            html_txt = html_txt.replace(
                                '<base href="." />',
                                f'<base href="{ep_path}/" />',
                                1,
                            )
                        html_txt = html_txt.replace(
                            '<link rel="icon" href="logo.png" />',
                            f'<link rel="icon" type="image/svg+xml" href="{SUBHKL_FAVICON_DATA_URI}" />',
                            1,
                        ).replace(
                            "<title>trame is built by Kitware</title>",
                            "<title>SUBHKL // Diffraction Parallax</title>",
                            1,
                        )
                        return web.Response(text=html_txt, content_type="text/html")
            return resp

        app._handle = _ep_handle

    def create_vm(self) -> None:
        binding = TrameBinding(self.server.state)
        self.local_view_models = create_viewmodels(
            binding,
            peaks_h5=self._peaks_h5,
            images_h5=self._images_h5,
            instrument=self._instrument,
        )
        self.view_model = self.local_view_models["main"]

    def create_ui(self) -> None:
        self.state.trame__title = "SUBHKL // Diffraction Parallax"
        self.state.trame__favicon = SUBHKL_FAVICON_DATA_URI
        with DivLayout(self.server) as layout:
            layout.root.style = (
                "position: fixed; inset: 0; margin: 0; padding: 0; "
                "width: 100%; height: 100%; overflow: hidden; background: #070913;"
            )
            html.Iframe(
                srcdoc=("parallax_html",),
                style="width: 100%; height: 100%; border: none; display: block;",
            )
