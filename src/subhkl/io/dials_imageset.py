"""Image-container support for the DIALS export.

subhkl's outputs don't reference their raw images in a way DIALS can open, so the
DIALS ``.expt`` normally has no ``ImageSet`` and must be read with
``check_format=False`` (no ``dials.image_viewer``). The pixel data subhkl
actually used, however, is present as the ``images`` stack written by ``reduce``
and ``merge_images`` (and available to the downstream programs as their input
stack).

This module writes that pixel data into a small self-describing HDF5 *image
container* that a companion dxtbx ``Format`` class (:mod:`subhkl.io.dxtbx_format`)
reads back, so the geometry stays authoritative in the ``.expt`` while the
container only supplies pixels. Everything here is pure numpy/h5py and carries no
dxtbx dependency, so it is importable and testable without DIALS.

Container layout::

    attrs["subhkl_imageset"] = True
    attrs["instrument"]      = <name>
    data       : float32 (n_frames, n_panels, slow, fast)
    panel_ids  : int32 (n_panels,)   # subhkl bank id per panel index
    frame_ids  : int32 (n_frames,)   # subhkl frame (run) id per image index
"""

from __future__ import annotations

import h5py
import numpy as np


def read_image_stack_layout(f):
    """Describe the (frame, panel) structure of a subhkl ``images`` stack.

    ``f`` is an open HDF5 file that contains an ``images`` dataset of shape
    ``(N, slow, fast)`` with a per-image ``bank_ids`` array, and optionally
    ``file_offsets`` marking per-run boundaries. Returns ``None`` if there is no
    image stack.
    """
    if "images" not in f:
        return None

    n_images = int(f["images"].shape[0])
    if "bank_ids" in f:
        bank_ids = np.asarray(f["bank_ids"][()]).astype(int)
    else:
        bank_ids = np.zeros(n_images, dtype=int)

    if "file_offsets" in f and len(f["file_offsets"]):
        file_offsets = np.asarray(f["file_offsets"][()]).astype(int)
        frame_of = np.searchsorted(file_offsets, np.arange(n_images), side="right") - 1
    else:
        frame_of = np.zeros(n_images, dtype=int)

    frame_ids = sorted({int(x) for x in frame_of})
    panel_bank_ids = sorted({int(b) for b in bank_ids})
    bank_to_panel = {b: i for i, b in enumerate(panel_bank_ids)}

    # image row for each (frame, panel), and the first row of each frame.
    row_of = {}
    frame_first_row = {}
    for i in range(n_images):
        fr = int(frame_of[i])
        panel = bank_to_panel[int(bank_ids[i])]
        row_of[(fr, panel)] = i
        frame_first_row.setdefault(fr, i)

    return {
        "n_images": n_images,
        "frame_ids": frame_ids,
        "panel_bank_ids": panel_bank_ids,
        "bank_to_panel": bank_to_panel,
        "row_of": row_of,
        "frame_first_row": frame_first_row,
    }


def frame_angle_rows(f, layout):
    """Return ``{frame_id: angle_row}`` for a stack, using each frame's first image.

    ``goniometer/angles`` in a stack is stored per image, so per-frame angles are
    taken from the representative (first) image of each frame. Returns ``None`` if
    no angles are present.
    """
    if "goniometer/angles" not in f:
        return None
    angles = np.asarray(f["goniometer/angles"][()])
    rows = {}
    for frame_id, first_row in layout["frame_first_row"].items():
        if angles.ndim == 2 and first_row < angles.shape[0]:
            rows[frame_id] = angles[first_row]
        elif angles.ndim == 1:
            rows[frame_id] = angles
    return rows or None


def write_image_container(container_path, images_ds, layout, instrument=None):
    """Write the ``(n_frames, n_panels, slow, fast)`` container from a stack.

    ``images_ds`` is the ``images`` dataset of the stack. Missing (frame, panel)
    cells are left as zeros. Returns the container path, or ``None`` on failure.
    """
    frame_ids = layout["frame_ids"]
    n_panels = len(layout["panel_bank_ids"])
    slow, fast = int(images_ds.shape[1]), int(images_ds.shape[2])

    frame_index = {fr: i for i, fr in enumerate(frame_ids)}
    data = np.zeros((len(frame_ids), n_panels, slow, fast), dtype=np.float32)
    for (fr, panel), row in layout["row_of"].items():
        data[frame_index[fr], panel] = np.asarray(images_ds[row], dtype=np.float32)

    with h5py.File(container_path, "w") as g:
        g.attrs["subhkl_imageset"] = True
        g.attrs["instrument"] = instrument or ""
        g.create_dataset("data", data=data, compression="lzf")
        g.create_dataset(
            "panel_ids", data=np.asarray(layout["panel_bank_ids"], dtype=np.int32)
        )
        g.create_dataset("frame_ids", data=np.asarray(frame_ids, dtype=np.int32))
    return container_path


def is_image_container(path):
    """True if ``path`` is a subhkl image container written by this module."""
    try:
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("subhkl_imageset", False))
    except Exception:
        return False


def container_num_frames(path):
    with h5py.File(path, "r") as f:
        return int(f["data"].shape[0])


def container_panel_ids(path):
    with h5py.File(path, "r") as f:
        return [int(x) for x in f["panel_ids"][()]]


def container_frame(path, index):
    """Return the per-panel image list ``[array, ...]`` for frame ``index``."""
    with h5py.File(path, "r") as f:
        frame = f["data"][index]
    return [np.asarray(frame[p]) for p in range(frame.shape[0])]
