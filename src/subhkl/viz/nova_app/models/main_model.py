"""Top-level NOVA Model container for SubHKL Diffraction Parallax."""

from __future__ import annotations

from pathlib import Path

from .subhkl_model import SubhklGalaxyModel


class MainModel:
    """Root application model adhering to the NOVA tutorial Episode 4 structure."""

    def __init__(
        self,
        peaks_h5: str | Path | None = None,
        images_h5: str | Path | None = None,
        instrument: str = "CG4D",
    ) -> None:
        self.subhkl = SubhklGalaxyModel(
            peaks_h5=peaks_h5,
            images_h5=images_h5,
            instrument=instrument,
        )
