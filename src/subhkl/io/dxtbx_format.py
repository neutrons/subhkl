"""dxtbx ``Format`` class for subhkl image containers.

This module is imported by dxtbx's format registry (via the ``dxtbx.format``
entry point declared in ``pyproject.toml``), not by subhkl itself, so it may
import dxtbx at module scope. It reads the self-describing HDF5 container written
by :mod:`subhkl.io.dials_imageset` and returns raw pixel data per panel.

Geometry is intentionally *not* provided here: the subhkl DIALS export builds the
beam/detector/goniometer/crystal models itself and overrides them on the imageset,
so this format only supplies pixels. That keeps the exact subhkl geometry
authoritative rather than round-tripping it through a NeXus schema.
"""

from __future__ import annotations

from dxtbx.format.FormatHDF5 import FormatHDF5
from scitbx.array_family import flex

from subhkl.io import dials_imageset


class FormatSubhkl(FormatHDF5):
    """Reads a subhkl image container (see :mod:`subhkl.io.dials_imageset`).

    A container holds one image per subhkl frame; each image carries one panel
    per detector bank, returned as a tuple of ``flex.double`` arrays.
    """

    @staticmethod
    def understand(image_file):
        return dials_imageset.is_image_container(image_file)

    def _start(self):
        self._num_images = dials_imageset.container_num_frames(self._image_file)
        self._panel_ids = dials_imageset.container_panel_ids(self._image_file)

    def get_num_images(self):
        return self._num_images

    def get_raw_data(self, index):
        panels = dials_imageset.container_frame(self._image_file, index)
        return tuple(flex.double(panel.astype(float)) for panel in panels)

    # Geometry is supplied by the exporter via imageset.set_*(); returning None
    # here lets those overrides win.
    def _detector(self, index=None):
        return None

    def _beam(self, index=None):
        return None

    def _goniometer(self, index=None):
        return None

    def _scan(self, index=None):
        return None


def make_imageset(container_path):
    """Build a dxtbx ``ImageSet`` (one image per subhkl frame) from a container.

    Geometry is left unset here; the caller overrides beam/detector/goniometer per
    experiment. Returns the imageset, or raises if dxtbx cannot build one.
    """
    n = dials_imageset.container_num_frames(container_path)
    return FormatSubhkl.get_imageset(
        [container_path],
        single_file_indices=list(range(n)),
        as_imageset=True,
        check_format=True,
    )
