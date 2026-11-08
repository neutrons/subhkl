"""NOVA Trame View for SubHKL Diffraction Parallax & Lattice Ladder (full-bleed hero layout)."""

from __future__ import annotations

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
            doc = {
                "galaxy_connected": self.pipeline_runner.is_connected,
                "galaxy_url": self.pipeline_runner.galaxy_url,
                "history_id": self.pipeline_runner.history_id,
                "history_datasets": self.pipeline_runner.list_history_datasets(),
                "active_job": self.pipeline_runner.active_job,
                "pipeline_stages": payload.get("pipeline_stages", []),
                "ladder_steps": payload.get("ladder_steps", []),
            }
            return web.json_response(doc)

        async def handle_payload(_request: web.Request) -> web.Response:
            return web.json_response(self.view_model.model.subhkl.payload or {})

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
            params.setdefault(
                "cell", [data.a, data.b, data.c, data.alpha, data.beta, data.gamma]
            )
            params.setdefault(
                "wavelength_range", [data.wavelength_min, data.wavelength_max]
            )
            params.setdefault("d_min", data.d_min)
            params.setdefault("n_zones", data.n_zones)
            params.setdefault("steps", data.steps)
            params.setdefault("peaks_file", data.peaks_file)
            params.setdefault("images_file", data.images_file)

            def _on_complete(new_h5_path: str) -> None:
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

        app.add_routes(
            [
                web.get("/api/pipeline/status", handle_status),
                web.get("/api/payload", handle_payload),
                web.post("/api/pipeline/run", handle_run),
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
                "margin: 0; padding: 0; width: 100vw; height: 100vh; "
                "overflow: hidden; background: #070913;"
            )
            html.Iframe(
                srcdoc=("parallax_html",),
                style="width: 100vw; height: 100vh; border: none; display: block;",
            )

