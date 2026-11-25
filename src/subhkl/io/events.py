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

import ctypes
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

# lambda [Angstrom] = ANGSTROM_M_PER_US * tof [microsecond] / path [m]
ANGSTROM_M_PER_US = 3.956034e-3

EVENTS_TOF = "events/tof"
EVENTS_OFFSETS = "events/image_offsets"
L1_KEY = "instrument/L1"

_H5Z_FILTER_SHUFFLE = 2
_H5Z_FILTER_LZF = 32000
_LZF_DECOMPRESS = None
_LZF_PROBED = False


def _get_lzf_decompress():
    """Return h5py's bundled C ``lzf_decompress`` function (GIL-free), or None."""
    global _LZF_DECOMPRESS, _LZF_PROBED
    if _LZF_PROBED:
        return _LZF_DECOMPRESS
    _LZF_PROBED = True
    try:
        import h5py.h5z

        lib = ctypes.CDLL(h5py.h5z.__file__)
        fn = lib.lzf_decompress
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        fn.restype = ctypes.c_uint
        _LZF_DECOMPRESS = fn
    except (ImportError, OSError, AttributeError, TypeError):
        _LZF_DECOMPRESS = None
    return _LZF_DECOMPRESS


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

    def __init__(self, filename, rdcc_nbytes: int = 64 * 1024 * 1024):
        self.filename = filename
        self._io_lock = threading.Lock()
        self._decode_pool: ThreadPoolExecutor | None = None
        self._f = h5py.File(
            filename,
            "r",
            rdcc_nbytes=int(rdcc_nbytes),
            rdcc_nslots=4099,
            rdcc_w0=0.75,
        )
        if EVENTS_TOF not in self._f or EVENTS_OFFSETS not in self._f:
            self._f.close()
            raise KeyError(f"{filename} carries no events")
        self.offsets = self._f[EVENTS_OFFSETS][()]
        self.l1 = float(self._f[L1_KEY][()]) if L1_KEY in self._f else None

        self._fast_lzf = False
        self._chunk_len = 0
        self._shuf_bit = 0
        self._lzf_bit = 0
        self._init_fast_chunk_decoder()

    def _init_fast_chunk_decoder(self) -> None:
        """Enable GIL-free parallel LZF+shuffle chunk decoding when supported."""
        dset = self._f[EVENTS_TOF]
        native_order = "<" if sys.byteorder == "little" else ">"
        if (
            dset.dtype.kind != "f"
            or dset.dtype.itemsize != 4
            or dset.dtype.byteorder not in ("=", "|", native_order)
            or dset.chunks is None
            or len(dset.chunks) != 1
            or dset.chunks[0] <= 0
        ):
            return
        plist = dset.id.get_create_plist()
        nfilt = plist.get_nfilters()
        if nfilt == 0:
            return
        shuf_bit = 0
        lzf_bit = 0
        seen_lzf = False
        for idx in range(nfilt):
            fid, _flags, cd_values, _name = plist.get_filter(idx)
            if fid == _H5Z_FILTER_SHUFFLE and not seen_lzf and shuf_bit == 0:
                if len(cd_values) >= 1 and int(cd_values[0]) != 4:
                    return
                shuf_bit = 1 << idx
            elif fid == _H5Z_FILTER_LZF and lzf_bit == 0:
                lzf_bit = 1 << idx
                seen_lzf = True
            else:
                return
        if lzf_bit and _get_lzf_decompress() is None:
            return
        self._chunk_len = int(dset.chunks[0])
        self._shuf_bit = shuf_bit
        self._lzf_bit = lzf_bit
        self._fast_lzf = True
        n_workers = min(8, max(1, os.cpu_count() or 1))
        if n_workers > 1:
            self._decode_pool = ThreadPoolExecutor(max_workers=n_workers)

    def close(self):
        if self._decode_pool is not None:
            self._decode_pool.shutdown(wait=True)
            self._decode_pool = None
        with self._io_lock:
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
        with self._io_lock:
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

    def image_event_count(self, img_idx: int) -> int:
        """Number of stored events in ``img_idx`` without reading ``events/tof``."""
        return int(self.offsets[int(img_idx) + 1]) - int(self.offsets[int(img_idx)])

    def _read_runs(self, out_tof: np.ndarray, runs: list[tuple[int, int, int]]) -> None:
        """Read ``[(run_lo, run_hi, dst), ...]`` into ``out_tof``."""
        if not runs:
            return
        if not self._fast_lzf:
            with self._io_lock:
                dset = self._f[EVENTS_TOF]
                for run_lo, run_hi, dst in runs:
                    dset.read_direct(
                        out_tof,
                        np.s_[run_lo:run_hi],
                        np.s_[dst : dst + (run_hi - run_lo)],
                    )
            return

        chunk_len = self._chunk_len
        chunk_targets: dict[int, list[tuple[int, int, int, int]]] = {}
        for run_lo, run_hi, dst in runs:
            c_first = run_lo // chunk_len
            c_last = (run_hi - 1) // chunk_len
            for c in range(c_first, c_last + 1):
                c_start = c * chunk_len
                s_lo = max(run_lo, c_start) - c_start
                s_hi = min(run_hi, c_start + chunk_len) - c_start
                d_lo = dst + max(run_lo, c_start) - run_lo
                d_hi = d_lo + (s_hi - s_lo)
                chunk_targets.setdefault(c, []).append((s_lo, s_hi, d_lo, d_hi))

        raw_chunks = []
        try:
            with self._io_lock:
                dset_id = self._f[EVENTS_TOF].id
                for c in sorted(chunk_targets):
                    mask, raw = dset_id.read_direct_chunk((c * chunk_len,))
                    raw_chunks.append((c, mask, raw, chunk_targets[c]))
        except OSError:
            with self._io_lock:
                dset = self._f[EVENTS_TOF]
                for run_lo, run_hi, dst in runs:
                    dset.read_direct(
                        out_tof,
                        np.s_[run_lo:run_hi],
                        np.s_[dst : dst + (run_hi - run_lo)],
                    )
            return

        lzf_fn = _get_lzf_decompress()
        lzf_bit = self._lzf_bit
        shuf_bit = self._shuf_bit
        nbytes = chunk_len * 4

        def _decode_chunk(item) -> None:
            _c, mask, raw, targets = item
            if lzf_bit and not (mask & lzf_bit):
                buf = np.empty(nbytes, dtype=np.uint8)
                got = lzf_fn(raw, len(raw), buf.ctypes.data, nbytes)
                if got != nbytes:
                    raise RuntimeError(
                        f"LZF chunk decompression returned {got} bytes, expected {nbytes}"
                    )
            else:
                buf = np.frombuffer(raw, dtype=np.uint8, count=nbytes)

            if shuf_bit and not (mask & shuf_bit):
                planes_all = buf.reshape(4, chunk_len)
                for s_lo, s_hi, d_lo, d_hi in targets:
                    out_tof[d_lo:d_hi].view(np.uint8).reshape(s_hi - s_lo, 4)[:] = (
                        planes_all[:, s_lo:s_hi].T
                    )
            else:
                f32_all = buf.view(np.float32)
                for s_lo, s_hi, d_lo, d_hi in targets:
                    out_tof[d_lo:d_hi] = f32_all[s_lo:s_hi]

        if len(raw_chunks) < 2 or self._decode_pool is None:
            for item in raw_chunks:
                _decode_chunk(item)
        else:
            list(self._decode_pool.map(_decode_chunk, raw_chunks))

    def read_slab(
        self,
        img_indices,
        images_slab: np.ndarray,
        out_tof: np.ndarray | None = None,
        out_pix_offsets: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Coalesced multi-image event read into preallocated slab buffers.

        Adjacent images in ``events/tof`` are read with a single contiguous
        chunk pass so LZF chunks straddling image boundaries are decompressed
        once, and per-pixel offsets are shifted into slab-global ``int32``
        coordinates.
        """
        idx_arr = np.asarray(img_indices, dtype=np.int64)
        m_imgs = len(idx_arr)
        images_arr = np.asarray(images_slab)
        n_pix = int(images_arr.shape[1]) * int(images_arr.shape[2])
        if m_imgs == 0:
            tof_buf = out_tof if out_tof is not None else np.zeros(1, dtype=np.float32)
            off_buf = (
                out_pix_offsets
                if out_pix_offsets is not None
                else np.zeros((1, n_pix + 1), dtype=np.int32)
            )
            return (
                tof_buf,
                out_pix_offsets if out_pix_offsets is not None else off_buf,
                0,
            )

        los = self.offsets[idx_arr].astype(np.int64, copy=False)
        his = self.offsets[idx_arr + 1].astype(np.int64, copy=False)
        lens = his - los
        total_ev = int(lens.sum())

        if out_tof is None or len(out_tof) < max(total_ev, 1):
            out_tof = np.zeros(max(total_ev, 1), dtype=np.float32)
        if (
            out_pix_offsets is None
            or out_pix_offsets.shape[0] < m_imgs
            or out_pix_offsets.shape[1] != n_pix + 1
        ):
            out_pix_offsets = np.zeros((m_imgs, n_pix + 1), dtype=np.int32)

        runs: list[tuple[int, int, int]] = []
        dst = 0
        i = 0
        while i < m_imgs:
            run_lo = int(los[i])
            run_hi = int(his[i])
            j = i + 1
            while j < m_imgs and int(los[j]) == run_hi:
                run_hi = int(his[j])
                j += 1
            n_run = run_hi - run_lo
            if n_run > 0:
                runs.append((run_lo, run_hi, dst))
                dst += n_run
            i = j
        self._read_runs(out_tof, runs)

        counts_2d = np.rint(images_arr[:m_imgs]).astype(np.int64).reshape(m_imgs, n_pix)
        cum_counts = np.cumsum(counts_2d, axis=1, dtype=np.int64)
        bad = np.flatnonzero(cum_counts[:, -1] != lens)
        if len(bad) > 0:
            b0 = int(bad[0])
            raise ValueError(
                f"image {int(idx_arr[b0])} of {self.filename} holds "
                f"{int(cum_counts[b0, -1])} counts but {int(lens[b0])} events; "
                "the image is no longer the histogram of its events"
            )
        img_bases = np.zeros(m_imgs, dtype=np.int32)
        if m_imgs > 1:
            np.cumsum(lens[:-1], dtype=np.int32, out=img_bases[1:])
        out_pix_offsets[:m_imgs, 0] = img_bases
        out_pix_offsets[:m_imgs, 1:] = cum_counts.astype(np.int32) + img_bases[:, None]
        if out_pix_offsets.shape[0] > m_imgs:
            out_pix_offsets[m_imgs:, :] = total_ev
        return out_tof, out_pix_offsets, total_ev


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
