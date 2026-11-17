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
    import os

    args = sys.argv[1:]
    remaining_argv = [sys.argv[0]]
    i = 0
    while i < len(args):
        arg = args[i]
        if (
            arg == "--peaks"
            or (arg == "-p" and i + 1 < len(args) and not args[i + 1].isdigit())
        ) and i + 1 < len(args):
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
            key, value = arg.split("=", 1)
            try:
                kwargs[key] = int(value)
                i += 1
                continue
            except ValueError:
                pass
        remaining_argv.append(arg)
        i += 1

    sys.argv = remaining_argv
    if os.environ.get("EP_PATH") or os.environ.get("GALAXY_URL"):
        kwargs.setdefault("timeout", 0)
        kwargs.setdefault("open_browser", False)

    app = MainApp(
        peaks_h5=peaks_h5,
        images_h5=images_h5,
        instrument=instrument,
    )
    app.server.start(**kwargs)


if __name__ == "__main__":
    main()
