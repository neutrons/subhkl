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
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--peaks", "-p") and i + 1 < len(args):
            peaks_h5 = args[i + 1]
            i += 2
            continue
        if arg in ("--images", "-i") and i + 1 < len(args):
            images_h5 = args[i + 1]
            i += 2
            continue
        if arg == "--instrument" and i + 1 < len(args):
            instrument = args[i + 1]
            i += 2
            continue
        if "=" in arg and not arg.startswith("--"):
            try:
                key, value = arg.split("=", 1)
                kwargs[key] = int(value)
            except Exception:
                pass
        i += 1

    app = MainApp(
        peaks_h5=peaks_h5,
        images_h5=images_h5,
        instrument=instrument,
    )
    app.server.start(**kwargs)


if __name__ == "__main__":
    main()
