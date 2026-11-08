"""NOVA Trame View for SubHKL Diffraction Parallax & Lattice Ladder (full-bleed hero layout)."""

from __future__ import annotations

from pathlib import Path

from nova.mvvm.trame_binding import TrameBinding
from trame.app import get_server
from trame.ui.html import DivLayout
from trame.widgets import html
from trame_server import Server

from subhkl.viz.nova_app.mvvm_factory import create_viewmodels


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

        self.create_vm()
        self.view_model.config_bind.connect("config")
        self.view_model.view_state_bind.connect("view_state")
        self.view_model.parallax_html_bind.connect("parallax_html")
        self.create_ui()
        self.view_model.update_view()

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

