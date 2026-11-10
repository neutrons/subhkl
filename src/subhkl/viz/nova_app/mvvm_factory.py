"""MVVM Factory wiring MainModel and MainViewModel to a BindingInterface."""

from __future__ import annotations

from pathlib import Path

from nova.mvvm.interface import BindingInterface

from subhkl.viz.nova_app.models.main_model import MainModel
from subhkl.viz.nova_app.view_models.main_view_model import MainViewModel


def create_viewmodels(
    binding: BindingInterface,
    peaks_h5: str | Path | None = None,
    images_h5: str | Path | None = None,
    instrument: str = "CG4D",
) -> dict[str, MainViewModel]:
    """Instantiate Model and ViewModels following the NOVA Tutorial Episode 4 factory pattern."""
    model = MainModel(
        peaks_h5=peaks_h5,
        images_h5=images_h5,
        instrument=instrument,
    )
    return {
        "main": MainViewModel(model, binding),
    }
