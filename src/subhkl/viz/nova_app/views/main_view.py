"""NOVA Trame View for SubHKL Diffraction Parallax & Lattice Ladder (full-bleed hero layout)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aiohttp import web
from nova.mvvm.trame_binding import TrameBinding
from trame.app import get_server
from trame.ui.html import DivLayout
from trame.widgets import html
from trame_server import Server

from subhkl.viz.nova_app.mvvm_factory import create_viewmodels
from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner


class MainApp:
    """Full-bleed NOVA Trame application exhausting the browser window with the Diffraction Parallax hero view."""

    def __init__(
        self,
        server: Server | None = None,
        peaks_h5: str | Path | None = None,
        images_h5: str | Path | None = None,
        instrument: str = "CG4D",
    ) -> None:
        self.server = server or get_server(None, client_type="vue3")
        self.state = self.server.state
        self._peaks_h5 = peaks_h5
        self._images_h5 = images_h5
        self._instrument = instrument
        self.pipeline_runner = NovaGalaxyPipelineRunner()

        self.create_vm()
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

        async def handle_status(_request: web.Request) -> web.Response:
            payload = self.view_model.model.subhkl.payload or {}
            hist_ds = self.pipeline_runner.list_history_datasets()
            cur_h5 = (
                self.view_model.model.subhkl.data.peaks_file
                or self.view_model.model.subhkl.data.images_file
            )
            doc = {
                "galaxy_connected": self.pipeline_runner.is_connected,
                "galaxy_url": self.pipeline_runner.galaxy_url,
                "history_id": self.pipeline_runner.history_id,
                "history_datasets": hist_ds,
                "step_artifacts": self.pipeline_runner.list_step_artifacts(
                    current_h5=cur_h5,
                    history_datasets=hist_ds,
                ),
                "active_job": self.pipeline_runner.active_job,
                "pipeline_stages": payload.get("pipeline_stages", []),
                "ladder_steps": payload.get("ladder_steps", []),
            }
            return web.json_response(doc)

        async def handle_artifact_download(request: web.Request) -> web.Response:
            ds_id = str(request.rel_url.query.get("dataset_id") or "").strip()
            fname = str(request.rel_url.query.get("name") or "").strip()
            cur_h5 = (
                self.view_model.model.subhkl.data.peaks_file
                or self.view_model.model.subhkl.data.images_file
            )
            try:
                local_fp, dl_name = self.pipeline_runner.resolve_artifact_for_download(
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
            except Exception as exc:
                return web.Response(text=str(exc), status=404)

        async def handle_payload(request: web.Request) -> web.Response:
            fp_q = request.rel_url.query.get("file_path")
            if fp_q is not None and str(fp_q).strip() != "":
                try:
                    local_h5 = os.path.abspath(str(fp_q).strip())
                    if os.path.isfile(local_h5):
                        ds_name = os.path.basename(local_h5).lower()
                        if any(k in ds_name for k in ("merged", "scan", "image")):
                            self.view_model.model.subhkl.data.images_file = local_h5
                            self.view_model.model.subhkl.data.peaks_file = local_h5
                        else:
                            self.view_model.model.subhkl.data.peaks_file = local_h5
                        self.view_model.model.subhkl.refresh_payload()
                except Exception as exc:
                    return web.json_response(
                        {"status": "error", "error": str(exc)}, status=404
                    )

            ds_id_q = request.rel_url.query.get("dataset_id")
            if ds_id_q is not None and str(ds_id_q).strip() != "":
                try:
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
                    ds_name = str(
                        ds_info.get("name") or os.path.basename(local_h5)
                    ).lower()
                    if any(k in ds_name for k in ("merged", "scan", "image")):
                        self.view_model.model.subhkl.data.images_file = local_h5
                        self.view_model.model.subhkl.data.peaks_file = local_h5
                    else:
                        self.view_model.model.subhkl.data.peaks_file = local_h5
                    self.view_model.model.subhkl.refresh_payload()
                except Exception as exc:
                    return web.json_response(
                        {"status": "error", "error": str(exc)}, status=404
                    )

            frame_q = request.rel_url.query.get("frame")
            if frame_q is not None and str(frame_q).strip() != "":
                try:
                    self.view_model.model.subhkl.refresh_payload(
                        frame_index=int(frame_q)
                    )
                except Exception:
                    pass
            return web.json_response(self.view_model.model.subhkl.payload or {})

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
                local_h5, _ = self.pipeline_runner.resolve_dataset_to_local_h5(
                    ds_id, fallback_dir=fb_dir
                )
                return web.FileResponse(path=local_h5)
            except Exception as exc:
                return web.Response(text=str(exc), status=404)

        async def handle_standalone_html(_request: web.Request) -> web.Response:
            html_doc = self.view_model.model.subhkl.get_widget_html()
            return web.Response(text=html_doc, content_type="text/html")

        async def handle_run(request: web.Request) -> web.Response:
            try:
                body = await request.json()
            except Exception:
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
            params.setdefault("n_zones", data.n_zones)
            params.setdefault("steps", data.steps)
            params.setdefault("peaks_file", data.peaks_file)
            params.setdefault("images_file", data.images_file)

            def _on_complete(new_h5_path: str) -> None:
                from subhkl.viz.nova_app.models.subhkl_model import InstrumentOptions

                try:
                    if (
                        "instrument" in params
                        and str(params["instrument"]) in InstrumentOptions.__members__
                    ):
                        data.instrument = InstrumentOptions(str(params["instrument"]))
                    for attr in ("a", "b", "c", "alpha", "beta", "gamma"):
                        if attr in params:
                            setattr(data, attr, float(params[attr]))
                    if "space_group" in params:
                        data.space_group = str(params["space_group"])
                    if "wavelength_min" in params:
                        data.wavelength_min = float(params["wavelength_min"])
                    if "wavelength_max" in params:
                        data.wavelength_max = float(params["wavelength_max"])
                except Exception:
                    pass
                if (
                    stage_id == "reduce"
                    or "merged" in os.path.basename(new_h5_path).lower()
                ):
                    self.view_model.model.subhkl.frame_index = 0
                    self.view_model.model.subhkl.data.images_file = new_h5_path
                    self.view_model.model.subhkl.data.peaks_file = new_h5_path
                else:
                    self.view_model.model.subhkl.data.peaks_file = new_h5_path
                self.view_model.model.subhkl.refresh_payload()

            try:
                res = self.pipeline_runner.submit_galaxy_stage(
                    stage_id, params=params, on_complete_h5=_on_complete
                )
                return web.json_response(res)
            except Exception as exc:
                return web.json_response(
                    {"status": "error", "error": str(exc)}, status=400
                )

        async def handle_upload_h5(request: web.Request) -> web.Response:
            import tempfile
            import time

            try:
                upload_dir = os.path.join(
                    tempfile.gettempdir(), f"subhkl_upload_{int(time.time())}"
                )
                os.makedirs(upload_dir, exist_ok=True)
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
                    dest_path = os.path.join(upload_dir, safe_name)
                    with open(dest_path, "wb") as f_out:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            f_out.write(chunk)
                    saved_paths.append(dest_path)
                if len(saved_paths) == 1 and not saved_paths[0].lower().endswith(
                    ".nxs.h5"
                ):
                    sp = saved_paths[0]
                    if (
                        "merged" in os.path.basename(sp).lower()
                        or "scan" in os.path.basename(sp).lower()
                    ):
                        self.view_model.model.subhkl.data.images_file = sp
                    self.view_model.model.subhkl.data.peaks_file = sp
                    self.view_model.model.subhkl.refresh_payload()
                return web.json_response({"status": "ok", "saved_paths": saved_paths})
            except Exception as exc:
                return web.json_response(
                    {"status": "error", "error": str(exc)}, status=400
                )

        async def handle_resolve_local_files(request: web.Request) -> web.Response:
            import glob as _glob
            import tempfile

            try:
                body = await request.json()
            except Exception:
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
            except Exception:
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

            res = await asyncio.to_thread(
                self.pipeline_runner.get_oncat_auth_status
            )
            return web.json_response(res)

        async def handle_oncat_auth_post(request: web.Request) -> web.Response:
            import asyncio

            try:
                body = await request.json()
            except Exception:
                body = {}
            res = await asyncio.to_thread(
                self.pipeline_runner.handle_oncat_auth, body
            )
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
                web.get("/api/datasets/{ds_id}/display", handle_dataset_display),
                web.post("/api/pipeline/run", handle_run),
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
        with DivLayout(self.server) as layout:
            layout.root.style = (
                "position: fixed; inset: 0; margin: 0; padding: 0; "
                "width: 100%; height: 100%; overflow: hidden; background: #070913;"
            )
            html.Iframe(
                srcdoc=("parallax_html",),
                style="width: 100%; height: 100%; border: none; display: block;",
            )
