"""NOVA ViewModel linking SubhklGalaxyModel to Trame View via nova-mvvm."""

from __future__ import annotations

from typing import Any

from nova.mvvm.interface import BindingInterface
from pydantic import BaseModel, Field

from subhkl.viz.nova_app.models.main_model import MainModel
from subhkl.viz.nova_app.models.subhkl_model import PipelineStageOptions


class ViewState(BaseModel):
    """Reactive UI state bound to the NOVA Trame view."""

    active_tab: str = Field(default="0", title="Active Tab")
    running: bool = Field(default=False, title="NDIP Job Running")
    active_stage: PipelineStageOptions = Field(
        default=PipelineStageOptions.solve, title="Active Pipeline Stage"
    )
    ladder_rung: int = Field(default=5, ge=1, le=5, title="Lattice Ladder Rung")
    status_text: str = Field(
        default="Ready — interactive Diffraction Parallax & Lattice Ladder active",
        title="Status",
    )


class MainViewModel:
    """ViewModel adhering to NOVA Tutorial Episodes 4 & 5 (`binding.new_bind` & `binding.new_worker`)."""

    def __init__(self, model: MainModel, binding: BindingInterface) -> None:
        self.model = model
        self.binding = binding
        self.view_state = ViewState()

        self.config_bind = binding.new_bind(
            self.model.subhkl.data, callback_after_update=self.change_callback
        )
        self.view_state_bind = binding.new_bind(self.view_state)
        self.parallax_html_bind = binding.new_bind()

    def change_callback(self, results: dict[str, Any]) -> None:
        """Invoked automatically by nova-mvvm when bound Pydantic fields change in the UI."""
        if results.get("error"):
            self.view_state.status_text = f"Validation error: {results['error']}"
            self.view_state_bind.update_in_view(self.view_state)
            return

        updated_fields = set(results.get("updated", []))
        if updated_fields & {"peaks_file", "images_file", "instrument", "max_zone_index"}:
            self.model.subhkl.refresh_payload()
            self.update_view()

    def update_view(self) -> None:
        """Push current model config, view state, and rendered Parallax HTML to the View."""
        self.config_bind.update_in_view(self.model.subhkl.data)
        self.view_state_bind.update_in_view(self.view_state)
        self.parallax_html_bind.update_in_view(self.model.subhkl.render_html())

    def reload_local_files(self) -> None:
        """Manually refresh the Diffraction Parallax view from selected HDF5 files."""
        self.model.subhkl.refresh_payload()
        n_peaks = (
            len(self.model.subhkl.payload.get("peaks", []))
            if self.model.subhkl.payload is not None
            else 0
        )
        self.view_state.status_text = (
            f"Loaded {n_peaks} reflections ({self.model.subhkl.data.instrument.value})"
        )
        self.update_view()

    def run_solve(self) -> None:
        """Dispatch `imaginex_subhkl_solve` to NDIP in a background worker thread (`binding.new_worker`)."""
        self.view_state.running = True
        self.view_state.active_stage = PipelineStageOptions.solve
        self.view_state.status_text = (
            f"Running {self.view_state.active_stage.value} on {self.model.subhkl.data.galaxy_url}..."
        )
        self.view_state_bind.update_in_view(self.view_state)

        worker = self.binding.new_worker(
            self.model.subhkl.run_subhkl_tool,
            stage=self.view_state.active_stage.value,
        )
        worker.connect_finished(self.on_solve_finished)
        worker.start()

    def on_solve_finished(self, result: Any = None) -> None:
        """Callback invoked when the background NDIP tool execution completes."""
        self.view_state.running = False
        if isinstance(result, Exception):
            self.view_state.status_text = f"NDIP execution error: {result}"
        else:
            last = self.model.subhkl.last_job_result
            self.view_state.status_text = (
                f"Completed {last.get('tool_id', 'solve')} -> "
                f"{last.get('downloaded_path') or 'done'}"
            )
        self.update_view()
