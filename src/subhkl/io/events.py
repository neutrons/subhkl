"""Pixel-sorted neutron events stored next to the detector images.

A reduced file collapses each detector bank to one image and so discards
the time of flight.  For pulsed-source instruments the events are kept
alongside, in a layout the image itself indexes:

- ``events/tof``            float32 [microsecond], all events of the file,
  sorted by (image, flat row-major pixel of the stored image, tof);
- ``events/image_offsets``  int64 [N_images + 1], where each image's events
  start in ``events/tof``;
- ``instrument/L1``         [m] moderator-to-sample distance.

No per-pixel index is stored: ``images[i]`` is by construction the
histogram of its own events over pixels, so the per-pixel offsets are
``cumsum(images[i].ravel())``.  The events of a pixel range are then one
contiguous slice, already sorted in time.
"""

import h5py
import numpy as np

# lambda [Angstrom] = ANGSTROM_M_PER_US * tof [microsecond] / path [m]
ANGSTROM_M_PER_US = 3.956034e-3

EVENTS_TOF = "events/tof"
EVENTS_OFFSETS = "events/image_offsets"
L1_KEY = "instrument/L1"


def sort_bank_events(local_id, tof, m, n, y_fast):
    """Image and pixel-sorted event times of one detector bank.

    Args:
        local_id: [-] event pixel id minus the bank offset
        tof: [microsecond] event time of flight
        m, n: [-] panel pixel counts (columns, rows)
        y_fast: the event id's fast index is the image row (the loader's
            ``bincount.reshape(m, n).T`` convention)
    Returns:
        image [photons/Pixel] (n, m) or the loader's legacy shape, and the
        event times sorted by (flat image pixel, tof), float32
    """
    local_id = np.asarray(local_id, dtype=np.int64)
    tof = np.asarray(tof, dtype=np.float32)
    ok = (local_id >= 0) & (local_id < m * n)
    if not ok.all():
        local_id, tof = local_id[ok], tof[ok]
    if y_fast:
        shape = (n, m)
        flat = (local_id % n) * m + (local_id // n)
    else:
        shape = (n, m)
        flat = local_id
    order = np.lexsort((tof, flat))
    image = np.bincount(flat, minlength=m * n).reshape(shape)
    return image, tof[order]


def write_events(f, tof_per_image, l1=None):
    """Write the event datasets for the images of ``f`` in stack order."""
    counts = np.array([len(t) for t in tof_per_image], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    total = int(offsets[-1])
    dset = f.create_dataset(
        EVENTS_TOF,
        shape=(total,),
        dtype=np.float32,
        chunks=(min(max(total, 1), 1 << 20),),
        compression="lzf",
        shuffle=True,
    )
    for i, t in enumerate(tof_per_image):
        if len(t):
            dset[offsets[i] : offsets[i + 1]] = t
    dset.attrs["units"] = "microsecond"
    f.create_dataset(EVENTS_OFFSETS, data=offsets)
    if l1 is not None and L1_KEY not in f:
        f[L1_KEY] = float(l1)


def has_events(filename):
    try:
        with h5py.File(filename, "r") as f:
            return EVENTS_TOF in f and EVENTS_OFFSETS in f
    except OSError:
        return False


def merge_events(h5_files, f_out):
    """Concatenate the event datasets of reduced files into a merged file.

    Events are carried only when every input has them: a merged stack whose
    images are partly without events could not be gated consistently.
    Returns True if the events were written.
    """
    counts = []
    l1 = None
    for name in h5_files:
        with h5py.File(name, "r") as f_in:
            if EVENTS_TOF not in f_in or EVENTS_OFFSETS not in f_in:
                return False
            counts.append(f_in[EVENTS_OFFSETS][()])
            if L1_KEY in f_in:
                l1 = float(f_in[L1_KEY][()])
    total = int(sum(c[-1] for c in counts))
    dset = f_out.create_dataset(
        EVENTS_TOF,
        shape=(total,),
        dtype=np.float32,
        chunks=(min(max(total, 1), 1 << 20),),
        compression="lzf",
        shuffle=True,
    )
    dset.attrs["units"] = "microsecond"
    merged_offsets = [np.zeros(1, dtype=np.int64)]
    start = 0
    step = 1 << 24
    for name, c in zip(h5_files, counts):
        n_ev = int(c[-1])
        with h5py.File(name, "r") as f_in:
            src = f_in[EVENTS_TOF]
            for lo in range(0, n_ev, step):
                hi = min(lo + step, n_ev)
                dset[start + lo : start + hi] = src[lo:hi]
        merged_offsets.append(c[1:] + start)
        start += n_ev
    f_out.create_dataset(EVENTS_OFFSETS, data=np.concatenate(merged_offsets))
    if l1 is not None and L1_KEY not in f_out:
        f_out[L1_KEY] = l1
    return True


class EventStore:
    """Read access to the pixel-sorted events of a reduced or merged file."""

    def __init__(self, filename):
        self.filename = filename
        self._f = h5py.File(filename, "r")
        if EVENTS_TOF not in self._f or EVENTS_OFFSETS not in self._f:
            self._f.close()
            raise KeyError(f"{filename} carries no events")
        self.offsets = self._f[EVENTS_OFFSETS][()]
        self.l1 = float(self._f[L1_KEY][()]) if L1_KEY in self._f else None

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def image_events(self, img_idx, image):
        """Events of one image and their per-pixel offsets.

        Args:
            img_idx: [-] position of the image in the file's stack
            image: [photons/Pixel] (H, W) the stored image, which is the
                pixel histogram of these events
        Returns:
            tof [microsecond] sorted by (flat pixel, tof), and pixel
            offsets [H * W + 1] into it
        """
        lo, hi = int(self.offsets[img_idx]), int(self.offsets[img_idx + 1])
        tof = self._f[EVENTS_TOF][lo:hi]
        counts = np.rint(np.asarray(image, dtype=np.float64)).astype(np.int64).ravel()
        pix_offsets = np.concatenate([[0], np.cumsum(counts)])
        if pix_offsets[-1] != hi - lo:
            raise ValueError(
                f"image {img_idx} of {self.filename} holds {pix_offsets[-1]} counts "
                f"but {hi - lo} events; the image is no longer the histogram of "
                "its events"
            )
        return tof, pix_offsets


def box_events(tof, pix_offsets, width, r0, r1, c0, c1):
    """Events of the pixel box [r0, r1) x [c0, c1) of one image.

    Returns:
        tof [microsecond] of the box's events, and for each its flat index
        within the box, row-major over (r1 - r0, c1 - c0)
    """
    n_cols = c1 - c0
    rows = np.arange(r0, r1)
    starts = pix_offsets[rows * width + c0]
    stops = pix_offsets[rows * width + c1]
    parts = [tof[a:b] for a, b in zip(starts, stops)]
    box_tof = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    # Per-pixel counts of the box, row by row, give each event its pixel.
    pix_counts = np.diff(
        pix_offsets[(rows[:, None] * width + np.arange(c0, c1 + 1)[None, :])], axis=1
    ).ravel()
    box_pix = np.repeat(np.arange(len(rows) * n_cols), pix_counts)
    return box_tof, box_pix
