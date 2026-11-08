"""NOVA ThemedApp View for SubHKL Diffraction Parallax & Lattice Ladder."""

from __future__ import annotations

from pathlib import Path

from nova.mvvm.trame_binding import TrameBinding
from nova.trame import ThemedApp
from nova.trame.view.components import InputField, RemoteFileInput
from nova.trame.view.layouts import GridLayout, HBoxLayout, VBoxLayout
from trame.widgets import html, vuetify3 as vuetify
from trame_server import Server

from subhkl.viz.nova_app.mvvm_factory import create_viewmodels


class MainApp(ThemedApp):
    """Main NOVA Trame application adhering to NOVA Tutorial Episode 5 (`ThemedApp`)."""

    def __init__(
        self,
        server: Server | None = None,
        peaks_h5: str | Path | None = None,
        images_h5: str | Path | None = None,
        instrument: str = "CG4D",
    ) -> None:
        super().__init__(server=server)
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
        self.set_theme("CompactTheme")
        self.state.trame__title = "SUBHKL // Diffraction Parallax & Lattice Ladder"

        with super().create_ui() as layout:
            layout.toolbar_title.set_text(
                "SUBHKL // Diffraction Parallax & Lattice Ladder"
            )

            with layout.pre_content:
                with HBoxLayout(classes="align-center justify-space-between px-2"):
                    with vuetify.VTabs(
                        v_model="view_state.active_tab", density="compact"
                    ):
                        vuetify.VTab("1. Diffraction Parallax & Lattice Ladder", value="0")
                        vuetify.VTab("2. NDIP / Crystal Parameters", value="1")
                    html.Span(
                        "{{ view_state.status_text }}",
                        classes="text-caption text-medium-emphasis",
                    )

            with layout.content:
                with VBoxLayout(
                    v_show="view_state.active_tab == '0'", stretch=True
                ):
                    html.Iframe(
                        srcdoc=("parallax_html",),
                        style="width: 100%; height: 82vh; border: 1px solid #30363d; border-radius: 6px;",
                    )

                with VBoxLayout(
                    v_show="view_state.active_tab == '1'", classes="pa-3"
                ):
                    with GridLayout(columns=3, gap="0.75em"):
                        InputField(v_model="config.galaxy_url")
                        InputField(v_model="config.galaxy_key", type="password")
                        InputField(v_model="config.instrument", type="select")

                    with GridLayout(columns=2, gap="0.75em", classes="mt-2"):
                        RemoteFileInput(
                            v_model="config.peaks_file",
                            base_paths=["/HFIR", "/SNS", "/home/xvg", "/tmp"],
                            extensions=[".h5", ".hdf5"],
                        )
                        RemoteFileInput(
                            v_model="config.images_file",
                            base_paths=["/HFIR", "/SNS", "/home/xvg", "/tmp"],
                            extensions=[".h5", ".hdf5"],
                        )

                    with GridLayout(columns=4, gap="0.75em", classes="mt-2"):
                        InputField(v_model="config.space_group")
                        InputField(v_model="config.wavelength_min", type="number")
                        InputField(v_model="config.wavelength_max", type="number")
                        InputField(v_model="config.d_min", type="number")

                    with GridLayout(columns=6, gap="0.75em", classes="mt-2"):
                        InputField(v_model="config.a", type="number")
                        InputField(v_model="config.b", type="number")
                        InputField(v_model="config.c", type="number")
                        InputField(v_model="config.alpha", type="number")
                        InputField(v_model="config.beta", type="number")
                        InputField(v_model="config.gamma", type="number")

                    with GridLayout(columns=3, gap="0.75em", classes="mt-2"):
                        InputField(v_model="config.n_zones", type="number")
                        InputField(v_model="config.steps", type="number")
                        InputField(v_model="config.max_zone_index", type="number")

            with layout.post_content:
                with HBoxLayout(
                    gap="0.75em", classes="align-center justify-center py-1"
                ):
                    vuetify.VBtn(
                        "Reload Local HDF5",
                        variant="tonal",
                        click=self.view_model.reload_local_files,
                    )
                    vuetify.VBtn(
                        "⚡ Run Solve on NDIP (nova-galaxy)",
                        v_if="!view_state.running",
                        color="primary",
                        click=self.view_model.run_solve,
                    )
                    vuetify.VProgressCircular(
                        v_if="view_state.running", indeterminate=True, size=24
                    )
