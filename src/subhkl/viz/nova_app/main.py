"""Entry point for starting the SubHKL Diffraction Parallax NOVA Application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from subhkl.viz.nova_app.views.main_view import MainApp


def main(
    peaks_h5: str | Path | None = None,
    images_h5: str | Path | None = None,
    instrument: str = "CG4D",
    **kwargs: Any,
) -> None:
    """Launch the NOVA Trame application server (`MainApp().server.start(**kwargs)`)."""
    app = MainApp(
        peaks_h5=peaks_h5,
        images_h5=images_h5,
        instrument=instrument,
    )
    for arg in sys.argv[1:]:
        try:
            key, value = arg.split("=", 1)
            kwargs[key] = int(value)
        except Exception:
            pass
    app.server.start(**kwargs)


if __name__ == "__main__":
    main()
