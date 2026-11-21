"""Time-of-flight gated integration of predicted reflections.

On a pulsed source every event carries its time of flight, and a Bragg
reflection occupies a narrow slice of the frame while the incoherent
background fills all of it.  The 2D integrator sees each reflection on the
background of the whole wavelength band.  Here each predicted reflection
gets its own small detector tile built only from the events inside a time
window around its predicted time of flight, and the unchanged 2D Poisson
amplitude solve (``matrix_free._solve_image_amplitudes``) runs on that
tile:

- data:        events of the tile's pixels inside the gate;
- background:  the rate of the same pixels in time sidebands on both sides
               of the gate, scaled to the gate width -- measured on the
               very pixels under the peak, which a 2D image cannot do;
- atoms:       the reflection itself plus any other prediction of the image
               whose footprint reaches the tile and whose gate overlaps in
               time, so genuinely overlapping reflections are still solved
               jointly.  Reflections that share pixels but not time (the
               harmonics of a Laue multiplet) simply never meet.

The gate is relative, [t (1 + lo), t (1 + hi)] around the predicted time
t.  By default it is measured from the data: the events near all predicted
positions are stacked in the relative coordinate t_event / t_pred - 1.
Reflections that are absent or weak add only a smooth background to the
stack, so no selection is needed and the gate is the same for strong and
weak reflections.  The stacked excess gives a common offset (a cell-scale
or flight-path error moves every reflection by the same relative amount)
and a width; the gate is the interval holding a chosen fraction of it.

Everything is done one image at a time on the accelerator.  The events of
an image are histogrammed into a (row, column, time) cube whose time bins
are logarithmic, so a relative gate is the same number of bins at every
wavelength.  Cumulative sums of that cube turn every count the method
needs into a handful of lookups: a tile inside a gate is a difference of
two time slices, and the sideband counts of a tile are corner differences
of the three-way cumulative sum.  Gate and sideband edges are therefore
bin edges; the bin is at most a thirtieth of the gate.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from subhkl.io.events import ANGSTROM_M_PER_US

ATOM_PAD = 8  # atoms per tile: the reflection and up to seven neighbours
MIN_TILE_HALF = 10
MAX_TILE_HALF = 30
SIDEBAND_BINS = 8  # sub-intervals per side
STACK_HALF_BOX = 2  # 5x5 pixel core for the stacked time profile
STACK_MAX_IMAGES = 96  # images sampled for the stacked time profile
MAX_TIME_BINS = 2048
MAX_LOG_BIN = 1.0e-3  # [-] coarsest relative time bin
PEAK_PAD = 512  # retained for legacy compatibility
TILE_CHUNK_SMALL = 256  # fixed tile chunk for small images/tests
TILE_CHUNK_LARGE = 2048  # fixed tile chunk for dense macromolecular runs (1 compile)
EVENT_CHUNK_SMALL = 1 << 16  # 65,536 events per static scatter step (small images)
EVENT_CHUNK = 1 << 20  # 1,048,576 events per static scatter step (1 compile)
EVENT_PAD = EVENT_CHUNK

_JIT = {}


@dataclass
class TofGate:
    """Relative time-of-flight gate around the predicted time."""

    lo: float
    hi: float
    coverage: float | None = None  # fraction of the stacked excess inside
    source: str = "manual"

    @property
    def width(self):
        return self.hi - self.lo


@dataclass
class TimeGrid:
    """Logarithmic time bins: edge b sits at t0 * exp(b * dl)."""

    t0: float  # [microsecond]
    dl: float  # [-]
    nt: int

    def edge_index(self, t):
        """[-] fractional edge index of time t [microsecond]."""
        return np.log(np.asarray(t, dtype=np.float64) / self.t0) / self.dl

    def edge_time(self, b):
        """[microsecond] time of edge index b."""
        return self.t0 * np.exp(np.asarray(b, dtype=np.float64) * self.dl)

    @classmethod
    def spanning(cls, t_lo, t_hi, dl):
        dl = max(dl, np.log(t_hi / t_lo) / MAX_TIME_BINS)
        return cls(
            t0=float(t_lo), dl=float(dl), nt=int(np.ceil(np.log(t_hi / t_lo) / dl))
        )


def predicted_tof(wavelength, l1, l2):
    """[microsecond] time of flight of wavelength [Angstrom] over l1 + l2 [m]."""
    return np.asarray(wavelength, dtype=np.float64) * (l1 + l2) / ANGSTROM_M_PER_US


def wavelength_band(predicted, peaks_band=None, images_band=None):
    """[Angstrom] (min, max) of the band the gates and sidebands may use.

    The band the reflections were predicted in is the one that counts: the
    images file only records the instrument default from reduce time, which
    is wrong whenever the choppers ran a different band.  Whatever its
    source, the band is widened to hold every predicted wavelength, so a
    stale band can never rule a predicted reflection out.

    Args:
        predicted: [Angstrom] wavelengths of the predicted reflections
        peaks_band: (min, max) stored with the predictions, or None
        images_band: (min, max) stored with the images, or None
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    band = peaks_band if peaks_band is not None else images_band
    if band is None:
        return float(predicted.min()), float(predicted.max())
    return (
        min(float(band[0]), float(predicted.min())),
        max(float(band[1]), float(predicted.max())),
    )


def parse_gate(spec):
    """'auto' -> None, 'lo,hi' -> TofGate with relative edges."""
    if spec is None or isinstance(spec, TofGate):
        return spec
    text = str(spec).strip().lower()
    if text == "auto":
        return None
    parts = [float(p) for p in text.split(",")]
    if len(parts) == 1:
        parts = [-abs(parts[0]), abs(parts[0])]
    lo, hi = parts
    if not lo < hi:
        raise ValueError(f"time-of-flight gate needs lo < hi, got {spec!r}")
    return TofGate(lo=lo, hi=hi, source="manual")


def _scatter_events_impl(cube_flat, flat_chunk):
    """Accumulate one fixed-size chunk of flat (pixel, time) event indices."""
    return cube_flat.at[flat_chunk].add(1)


def _finalize_prefix_impl(cube_flat, valid, shape, need_ct=True):
    """Cumulative sums of one image's accumulated (row, column, time) event cube."""
    import jax.numpy as jnp

    H, W, nt = shape
    cube = cube_flat[:-1].reshape(H, W, nt) * valid[:, :, None]
    ct = jnp.pad(jnp.cumsum(cube, axis=2), ((0, 0), (0, 0), (1, 0)))
    p3 = jnp.pad(jnp.cumsum(jnp.cumsum(ct, axis=0), axis=1), ((1, 0), (1, 0), (0, 0)))
    if need_ct:
        return ct, p3
    return None, p3


def _prefix_impl(flat_idx, valid, shape):
    """Cumulative sums of one image's (row, column, time) event cube.

    Args:
        flat_idx: [-] per event, (row * W + col) * nt + time bin; events to
            leave out (padding, out of range) point at the slot H * W * nt
        valid: [-] (H, W) int32, 1 = usable pixel
    Returns:
        ct [photons] (H, W, nt + 1): counts of a pixel before each time edge
        p3 [photons] (H + 1, W + 1, nt + 1): ct summed over the pixels above
            and left of each corner
    """
    import jax.numpy as jnp

    H, W, nt = shape
    cube_flat = jnp.zeros(H * W * nt + 1, dtype=jnp.int32).at[flat_idx].add(1)
    return _finalize_prefix_impl(cube_flat, valid, shape, need_ct=True)


def _box_sums_impl(p3, r0, r1, c0, c1, edges):
    """[photons] (K, S): counts of each pixel box before each time edge."""
    r0, r1, c0, c1 = r0[:, None], r1[:, None], c0[:, None], c1[:, None]
    return p3[r1, c1, edges] - p3[r0, c1, edges] - p3[r1, c0, edges] + p3[r0, c0, edges]


def _tiles_impl(ct, valid, org_r, org_c, b_lo, b_hi, T):
    """Tiles of the counts between two time edges, and their usable pixels."""
    import jax.numpy as jnp

    H, W = valid.shape
    rr = org_r[:, None] + jnp.arange(T)[None, :]
    cc = org_c[:, None] + jnp.arange(T)[None, :]
    inside = ((rr >= 0) & (rr < H))[:, :, None] & ((cc >= 0) & (cc < W))[:, None, :]
    rr = jnp.clip(rr, 0, H - 1)[:, :, None]
    cc = jnp.clip(cc, 0, W - 1)[:, None, :]
    counts = ct[rr, cc, b_hi[:, None, None]] - ct[rr, cc, b_lo[:, None, None]]
    ok = inside & (valid[rr, cc] > 0)
    return (counts * ok).astype(jnp.float32), ok.astype(jnp.float32)


def _jit(name):
    if name not in _JIT:
        from jax import jit

        if name == "prefix":
            _JIT[name] = partial(jit, static_argnames=["shape"])(_prefix_impl)
        elif name == "scatter":
            _JIT[name] = jit(_scatter_events_impl, donate_argnums=(0,))
        elif name == "finalize_prefix":
            _JIT[name] = partial(jit, static_argnames=["shape", "need_ct"])(
                _finalize_prefix_impl
            )
        elif name == "box":
            _JIT[name] = jit(_box_sums_impl)
        elif name == "tiles":
            _JIT[name] = partial(jit, static_argnames=["T"])(_tiles_impl)
        elif name == "snap":
            from subhkl.search.matrix_free import _snap_positions_impl

            _JIT[name] = jit(_snap_positions_impl)
    return _JIT[name]


def _solve_tiles(has_profile, use_l1_gate, allow_negative):
    """vmapped ``_solve_image_amplitudes`` over a batch of tiles."""
    key = ("solve", has_profile, use_l1_gate, allow_negative)
    if key not in _JIT:
        from jax import jit, vmap

        from subhkl.search.matrix_free import _solve_image_amplitudes_impl

        def solve(tiles, bgs, valid, r, c, vu, vv, cuv, ok, z, prof):
            one = partial(
                _solve_image_amplitudes_impl,
                admission_z=z,
                profile=prof,
                use_l1_gate=use_l1_gate,
                allow_negative=allow_negative,
            )
            return vmap(one)(tiles, bgs, valid, r, c, vu, vv, cuv, ok)

        _JIT[key] = jit(solve)
    return _JIT[key]


def _prepare_flat_events(tof, pix_offsets, H, W, grid):
    """Map one image's (tof, pix_offsets) into flat int32 cube indices on CPU."""
    n_ev = len(tof)
    dump = H * W * grid.nt
    if n_ev == 0:
        return np.empty(0, dtype=np.int32), dump
    pix = np.repeat(np.arange(H * W, dtype=np.int64), np.diff(pix_offsets))
    log_t0 = np.log(grid.t0)
    inv_dl = 1.0 / grid.dl
    tb = np.floor(
        (np.log(np.maximum(np.asarray(tof, dtype=np.float64), 1e-6)) - log_t0) * inv_dl
    ).astype(np.int64)
    in_range = (tb >= 0) & (tb < grid.nt)
    flat = np.where(in_range, pix * grid.nt + tb, dump).astype(np.int32)
    return flat, dump


def _image_prefix_from_flat(flat, dump, valid_img, grid, need_ct=True):
    """Build (ct, p3) from pre-binned flat indices using fixed-size event chunks."""
    import jax.numpy as jnp

    H, W = valid_img.shape
    cube_flat = jnp.zeros(H * W * grid.nt + 1, dtype=jnp.int32)
    n_ev = len(flat)
    if n_ev > 0:
        step = EVENT_CHUNK_SMALL if n_ev <= EVENT_CHUNK_SMALL else EVENT_CHUNK
        scatter_fn = _jit("scatter")
        for lo in range(0, n_ev, step):
            hi = min(n_ev, lo + step)
            if hi - lo == step:
                chunk = flat[lo:hi]
            else:
                chunk = np.full(step, dump, dtype=np.int32)
                chunk[: hi - lo] = flat[lo:hi]
            cube_flat = scatter_fn(cube_flat, jnp.asarray(chunk))
    return _jit("finalize_prefix")(
        cube_flat,
        jnp.asarray((valid_img > 0.5).astype(np.int32)),
        shape=(H, W, grid.nt),
        need_ct=need_ct,
    )


def _image_prefix(tof, pix_offsets, valid_img, grid, need_ct=True):
    """Cumulative event cube of one image on the accelerator."""
    H, W = valid_img.shape
    flat, dump = _prepare_flat_events(tof, pix_offsets, H, W, grid)
    return _image_prefix_from_flat(flat, dump, valid_img, grid, need_ct=need_ct)


def _padded(arr, n_pad, fill=0):
    if len(arr) == n_pad:
        return arr
    out = np.full((n_pad,) + arr.shape[1:], fill, dtype=arr.dtype)
    out[: len(arr)] = arr
    return out


def _pad_len(n):
    return max(PEAK_PAD, -(-n // PEAK_PAD) * PEAK_PAD)


def _select_tile_chunk(max_peaks_per_image: int) -> int:
    return (
        TILE_CHUNK_SMALL
        if max_peaks_per_image <= TILE_CHUNK_SMALL
        else TILE_CHUNK_LARGE
    )


def _group_frame_indices(frames: np.ndarray, n_frames: int) -> list[np.ndarray]:
    """O(N) partition of peak indices by frame index (preserving within-frame order)."""
    if len(frames) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n_frames)]
    order = np.argsort(frames, kind="mergesort")
    counts = np.bincount(frames, minlength=n_frames)
    splits = np.cumsum(counts[:-1])
    parts = np.split(order, splits)
    return parts


def _box_sums_chunked(p3, r0, r1, c0, c1, edges, chunk_size: int) -> np.ndarray:
    """Evaluate _jit('box') in fixed ``chunk_size`` slices so XLA compiles once."""
    import jax.numpy as jnp

    n = len(r0)
    s_dim = edges.shape[1]
    out = np.empty((n, s_dim), dtype=np.int32)
    box_fn = _jit("box")
    r0_i = r0.astype(np.int32, copy=False)
    r1_i = r1.astype(np.int32, copy=False)
    c0_i = c0.astype(np.int32, copy=False)
    c1_i = c1.astype(np.int32, copy=False)
    ed_i = edges.astype(np.int32, copy=False)
    for lo in range(0, n, chunk_size):
        hi = min(n, lo + chunk_size)
        m = hi - lo
        res = box_fn(
            p3,
            jnp.asarray(_padded(r0_i[lo:hi], chunk_size)),
            jnp.asarray(_padded(r1_i[lo:hi], chunk_size)),
            jnp.asarray(_padded(c0_i[lo:hi], chunk_size)),
            jnp.asarray(_padded(c1_i[lo:hi], chunk_size)),
            jnp.asarray(_padded(ed_i[lo:hi], chunk_size)),
        )
        out[lo:hi] = np.asarray(res)[:m]
    return out


def stack_relative_tof(
    reader,
    static_valid,
    frames,
    rs,
    cs,
    tof_pred,
    grid,
    search=0.05,
    n_bins=100,
    show_progress=False,
):
    """Histogram of t_event / t_pred - 1 near the predicted positions.

    A sample of the images is enough: the stack only has to show where the
    Bragg excess sits relative to the predictions and how wide it is.

    Args:
        reader: frame -> (tof [microsecond], pixel offsets) of that image
        static_valid: [-] (B, H, W), 1 = usable pixel
        frames, rs, cs: image index and predicted position [Pixel] per peak
        tof_pred: [microsecond] predicted time of flight per peak
        grid: TimeGrid of the event cube
        search: [-] half-range of the relative coordinate
    Returns:
        bin edges [-] and summed counts [photons]
    """
    B, H, W = static_valid.shape
    edges = np.linspace(-search, search, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.float64)
    hb = STACK_HALF_BOX
    nb = int(np.ceil(search / grid.dl)) + 1
    offs = np.arange(-nb, nb + 2)

    frame_groups = _group_frame_indices(frames, B)
    with_peaks = np.array([f for f in range(B) if len(frame_groups[f]) > 0], dtype=int)
    if len(with_peaks) == 0:
        return edges, hist
    if len(with_peaks) > STACK_MAX_IMAGES:
        pick = np.linspace(0, len(with_peaks) - 1, STACK_MAX_IMAGES).round().astype(int)
        with_peaks = with_peaks[pick]

    max_n = max(len(frame_groups[int(f)]) for f in with_peaks)
    chunk_size = _select_tile_chunk(max_n)

    def _load_one(f_int: int):
        tof, pix_offsets = reader(f_int)
        return _prepare_flat_events(tof, pix_offsets, H, W, grid)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_load_one, int(with_peaks[0]))
        pbar = tqdm(
            range(len(with_peaks)),
            desc="Time-of-flight profile",
            disable=not show_progress,
        )
        for k_i in pbar:
            f = int(with_peaks[k_i])
            flat, dump = fut.result()
            if k_i + 1 < len(with_peaks):
                fut = pool.submit(_load_one, int(with_peaks[k_i + 1]))

            idx = frame_groups[f]
            _, p3 = _image_prefix_from_flat(
                flat, dump, static_valid[f], grid, need_ct=False
            )
            ri, ci = np.rint(rs[idx]).astype(int), np.rint(cs[idx]).astype(int)
            r0, r1 = np.clip(ri - hb, 0, H), np.clip(ri + hb + 1, 0, H)
            c0, c1 = np.clip(ci - hb, 0, W), np.clip(ci + hb + 1, 0, W)
            centre = np.rint(grid.edge_index(tof_pred[idx])).astype(int)
            e = np.clip(centre[:, None] + offs[None, :], 0, grid.nt)
            sums = _box_sums_chunked(p3, r0, r1, c0, c1, e, chunk_size)
            del p3
            counts = np.diff(sums, axis=1)
            x = grid.edge_time(0.5 * (e[:, :-1] + e[:, 1:])) / tof_pred[idx, None] - 1.0
            hist += np.histogram(x.ravel(), bins=edges, weights=counts.ravel())[0]
    return edges, hist


def gate_from_stack(edges, hist, coverage=0.95, min_z=5.0):
    """The gate holding ``coverage`` of the stacked Bragg excess.

    The baseline under the excess is a straight line through the outer
    parts of the search range, where no reflection contributes.  Returns
    None when the excess is not significant (``min_z`` standard errors):
    the predictions then do not line up with the events in time and no
    gate can be read off.
    """
    x = 0.5 * (edges[:-1] + edges[1:])
    search = edges[-1]
    dx = edges[1] - edges[0]
    outer = np.abs(x) > 0.6 * search
    slope, icpt = np.polyfit(x[outer], hist[outer], 1)
    base = slope * x + icpt
    excess = hist - base
    inner = ~outer

    # Smooth just enough to locate the maximum without chasing a single bin.
    n_smooth = 3
    kern = np.ones(n_smooth) / n_smooth
    smooth = np.convolve(excess, kern, mode="same")
    peak = int(np.argmax(np.where(inner, smooth, -np.inf)))

    # The support of the excess: outward from the maximum until the smoothed
    # excess reaches the baseline noise, bounded by the inner range.
    noise = np.sqrt(np.maximum(base, 1.0) / n_smooth)
    left = peak
    while left > 0 and inner[left - 1] and smooth[left - 1] > noise[left - 1]:
        left -= 1
    right = peak
    while (
        right < len(x) - 1 and inner[right + 1] and smooth[right + 1] > noise[right + 1]
    ):
        right += 1
    support = slice(left, right + 1)
    total = float(excess[support].sum())
    z = total / np.sqrt(max(float(base[support].sum()), 1.0))
    if total <= 0 or z < min_z:
        return None

    cdf = np.cumsum(np.maximum(excess[support], 0.0))
    cdf = cdf / cdf[-1]
    tail = 0.5 * (1.0 - coverage)
    i_lo = int(np.searchsorted(cdf, tail, side="left"))
    i_hi = int(np.searchsorted(cdf, 1.0 - tail, side="left"))
    lo = edges[left + i_lo]
    hi = edges[left + min(i_hi, right - left) + 1]
    if hi - lo < 2 * dx:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - dx, mid + dx
    return TofGate(lo=float(lo), hi=float(hi), coverage=coverage, source="stack")


def _tile_half(var_us, var_vs):
    sig_major = np.sqrt(np.maximum(var_us, var_vs))
    half = int(np.ceil(3.5 * float(np.percentile(sig_major, 99)))) + 2
    return int(np.clip(half, MIN_TILE_HALF, MAX_TILE_HALF))


def _sideband_edges(b_lo, b_hi, nominal, lim_lo, lim_hi, factor):
    """Time-edge indices of the sideband sub-intervals of each gate.

    Args:
        b_lo, b_hi: [-] gate edges, (K,)
        nominal: [-] gate width in bins before clipping to the band
        lim_lo, lim_hi: [-] edges of the wavelength band, (K,)
        factor: [-] sideband length on each side in units of the gate width
    Returns:
        edges (K, 2, SIDEBAND_BINS + 1) int, and (K, 2) bool: side is long
        enough to use
    """
    guard = int(np.ceil(0.5 * nominal))
    span = int(np.ceil(factor * nominal))
    a = np.stack([b_lo - guard - span, b_hi + guard], axis=1)
    b = np.stack([b_lo - guard, b_hi + guard + span], axis=1)
    a = np.clip(a, lim_lo[:, None], lim_hi[:, None])
    b = np.clip(b, lim_lo[:, None], lim_hi[:, None])
    frac = np.arange(SIDEBAND_BINS + 1) / SIDEBAND_BINS
    edges = np.rint(a[:, :, None] + (b - a)[:, :, None] * frac).astype(np.int64)
    return edges, (b - a) >= max(1.0, 0.25 * nominal)


def _masked_row_median(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact row-wise median of finite masked entries without np.nanmedian overhead."""
    valid = mask & np.isfinite(vals)
    cnt = valid.sum(axis=1)
    out = np.full(len(vals), np.nan, dtype=np.float64)
    ok = cnt > 0
    if not np.any(ok):
        return out
    filled = np.where(valid[ok], vals[ok], np.inf)
    s = np.sort(filled, axis=1)
    c_ok = cnt[ok]
    rows = np.arange(len(c_ok))
    out[ok] = 0.5 * (s[rows, (c_ok - 1) // 2] + s[rows, c_ok // 2])
    return out


def _sideband_rates(counts, widths, keep, keep_unblocked, n_valid):
    """Background rate [photons/Pixel/microsecond] of each tile.

    Args:
        counts: [photons] (K, S) per sideband sub-interval, usable pixels
        widths: [microsecond] (K, S)
        keep: [-] (K, S) sub-intervals free of other reflections' gates
        keep_unblocked: [-] (K, S) the same without that exclusion; used
            where it leaves nothing
        n_valid: [-] (K,) usable pixels of the tile
    """
    none_left = ~keep.any(axis=1)
    keep = np.where(none_left[:, None], keep_unblocked, keep)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_width = np.where(keep, counts / widths, np.nan)
        any_kept = keep.any(axis=1)
        med = _masked_row_median(per_width, keep)
        med_w = np.ones(len(counts), dtype=np.float64)
        if np.any(any_kept):
            med_w[any_kept] = _masked_row_median(widths[any_kept], keep[any_kept])
        # An unpredicted reflection in a sideband is a count far above the
        # others; leave such sub-intervals out.  Counts are compared per
        # unit width because clipping at the band edge can shorten a side.
        tol = 5.0 * np.sqrt(np.maximum(med * med_w, 1.0)) / med_w
        trimmed = keep & (per_width <= (med + tol)[:, None])
        trimmed = np.where(trimmed.any(axis=1)[:, None], trimmed, keep)
        total_c = np.where(trimmed, counts, 0.0).sum(axis=1)
        total_w = np.where(trimmed, widths, 0.0).sum(axis=1)
        rate = total_c / (total_w * np.maximum(n_valid, 1))
    return np.where(any_kept & (total_w > 0), rate, np.nan)


def _build_neighbor_plan(
    rs_f: np.ndarray,
    cs_f: np.ndarray,
    reach_f: np.ndarray,
    in_band_f: np.ndarray,
    b_lo_f: np.ndarray,
    b_hi_f: np.ndarray,
    sb_edges: np.ndarray,
    keep0: np.ndarray,
    half: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build sideband exclusion mask and nearest joint-atom table in O(n) via cKDTree."""
    n = len(rs_f)
    keep = keep0.copy()
    local = np.zeros((n, K), dtype=np.int64)
    local[:, 0] = np.arange(n, dtype=np.int64)
    has_atom = np.zeros((n, K), dtype=bool)
    has_atom[:, 0] = True
    n_dropped = 0

    if n <= 1 or not np.any(in_band_f):
        return keep, local, has_atom, n_dropped

    r_max = float(half + np.max(reach_f))
    pts = np.column_stack((rs_f, cs_f))
    pairs = cKDTree(pts).query_pairs(r=r_max + 1e-9, p=np.inf, output_type="ndarray")
    if len(pairs) == 0:
        return keep, local, has_atom, n_dropped

    pa_c = np.concatenate((pairs[:, 0], pairs[:, 1]))
    pj_c = np.concatenate((pairs[:, 1], pairs[:, 0]))
    dr_c = np.abs(rs_f[pa_c] - rs_f[pj_c])
    dc_c = np.abs(cs_f[pa_c] - cs_f[pj_c])
    thr_near = half + reach_f[pj_c]
    ok_near = in_band_f[pj_c] & (dr_c <= thr_near) & (dc_c <= thr_near)
    pa = pa_c[ok_near]
    pj = pj_c[ok_near]
    if len(pa) == 0:
        return keep, local, has_atom, n_dropped

    hit = (sb_edges[pa, :, :-1] < b_hi_f[pj, None, None]) & (
        sb_edges[pa, :, 1:] > b_lo_f[pj, None, None]
    )
    blocked = np.zeros(keep.shape, dtype=bool)
    np.logical_or.at(blocked, pa, hit)
    keep &= ~blocked

    dr_n = dr_c[ok_near]
    dc_n = dc_c[ok_near]
    touch_n = reach_f[pa] + reach_f[pj]
    ok_joint = (
        (dr_n <= touch_n)
        & (dc_n <= touch_n)
        & (b_lo_f[pa] < b_hi_f[pj])
        & (b_hi_f[pa] > b_lo_f[pj])
    )
    ja = pa[ok_joint]
    jj = pj[ok_joint]
    if len(ja) == 0:
        return keep, local, has_atom, n_dropped

    d2_j = dr_n[ok_joint] ** 2 + dc_n[ok_joint] ** 2
    # Lexicographic sort by (ja, d2_j, jj) preserves exact tie-breaking of np.argsort.
    ord_j = np.lexsort((jj, d2_j, ja))
    ja_s = ja[ord_j]
    jj_s = jj[ord_j]
    counts_j = np.bincount(ja_s, minlength=n)
    n_dropped = int(np.sum(counts_j > (K - 1)))
    starts_j = np.zeros(n, dtype=np.int64)
    starts_j[1:] = np.cumsum(counts_j[:-1])
    ranks = np.arange(len(ja_s), dtype=np.int64) - starts_j[ja_s]
    keep_k = ranks < (K - 1)
    sel_row = ja_s[keep_k]
    sel_col = ranks[keep_k] + 1
    local[sel_row, sel_col] = jj_s[keep_k]
    has_atom[sel_row, sel_col] = True
    return keep, local, has_atom, n_dropped


def integrate_reflections_tof_gated(
    images_batch,
    reader,
    frames,
    rs,
    cs,
    tof_pred,
    tof_lim_lo,
    tof_lim_hi,
    var_us,
    var_vs,
    cov_uvs,
    gate=None,
    gate_coverage=0.95,
    gate_search=0.05,
    sideband_factor=4.0,
    static_valid=None,
    profile="gaussian",
    fp_target=None,
    border_width=0,
    allow_negative=False,
    show_progress=False,
):
    """Integrate every predicted reflection on its own time-gated tile.

    Same row contract as ``integrate_reflections_matrix_free``:
    [intensity, r, c, sigma_eff, sigI] per input peak, censored rows
    carrying sigI = 0.

    Args:
        images_batch: [photons/Pixel] (B, H, W), the event histograms
        reader: frame -> (tof [microsecond], pixel offsets) of that image
        frames: [-] image index per peak
        rs, cs: [Pixel] predicted positions
        tof_pred: [microsecond] predicted time of flight per peak
        tof_lim_lo, tof_lim_hi: [microsecond] per peak, the times of the
            wavelength band edges at its pixel; gate and sidebands are
            clipped to them
        var_us, var_vs, cov_uvs: [Pixel^2] projected 2D shape per peak
        gate: TofGate, or None to measure it from the stacked time profile
        gate_coverage: [-] fraction of the stacked excess the measured
            gate holds
        gate_search: [-] half-range of t / t_pred - 1 searched for it
        sideband_factor: [-] sideband length on each side of the gate in
            units of the gate width
        static_valid: [-] optional (B, H, W), 1 = usable pixel
        profile: "gaussian" or a path to a radial profile JSON
    Returns:
        (rows, info): info holds the gate, the per-peak gate centre
        ``tof`` [microsecond], expected background ``bg`` [photons/Pixel]
        inside the gate, and the stacked time profile
    """
    import jax.numpy as jnp

    from subhkl.search.ssn import calibrated_admission_z

    B, H, W = images_batch.shape
    n_peaks = len(frames)
    info = {"gate": gate, "stack_edges": None, "stack_hist": None}
    if n_peaks == 0:
        return np.empty((0, 5)), info

    frames = np.asarray(frames, dtype=int)
    rs = np.asarray(rs, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)
    tof_pred = np.asarray(tof_pred, dtype=np.float64)
    tof_lim_lo = np.asarray(tof_lim_lo, dtype=np.float64)
    tof_lim_hi = np.asarray(tof_lim_hi, dtype=np.float64)
    var_us = np.asarray(var_us, dtype=np.float32)
    var_vs = np.asarray(var_vs, dtype=np.float32)
    cov_uvs = np.asarray(cov_uvs, dtype=np.float32)

    if static_valid is None:
        static_valid = np.ones((B, H, W), dtype=np.float32)
    else:
        static_valid = np.asarray(static_valid, dtype=np.float32).copy()
    bw_px = min(max(0, int(border_width)), H // 4, W // 4)
    if bw_px > 0:
        static_valid[:, :bw_px, :] = 0.0
        static_valid[:, H - bw_px :, :] = 0.0
        static_valid[:, :, :bw_px] = 0.0
        static_valid[:, :, W - bw_px :] = 0.0

    margin = gate_search + 0.01
    t_span = (tof_lim_lo.min() * (1.0 - margin), tof_lim_hi.max() * (1.0 + margin))

    stack_grid = TimeGrid.spanning(*t_span, MAX_LOG_BIN)
    edges, hist = stack_relative_tof(
        reader,
        static_valid,
        frames,
        rs,
        cs,
        tof_pred,
        stack_grid,
        search=gate_search,
        show_progress=show_progress,
    )
    info["stack_edges"], info["stack_hist"] = edges, hist
    if gate is None:
        gate = gate_from_stack(edges, hist, coverage=gate_coverage)
        if gate is None:
            raise ValueError(
                "No Bragg excess in the stacked time-of-flight profile within "
                f"+-{gate_search:.3f} of the predicted times: the predictions do "
                "not line up with the events in time. Check the solve (cell "
                "scale, orientation) and the flight path, give the gate "
                "explicitly (--tof-gate lo,hi), or integrate the whole band "
                "(--tof-gate off)."
            )
    info["gate"] = gate
    if show_progress:
        print(
            f"  > Time-of-flight gate: t/t_pred - 1 in [{gate.lo:+.4f}, {gate.hi:+.4f}] "
            f"({gate.source}"
            + (
                f", {100 * gate.coverage:.0f}% of the stacked excess"
                if gate.coverage
                else ""
            )
            + ")"
        )

    # Time bins a thirtieth of the gate or finer; gates become bin edges.
    grid = TimeGrid.spanning(*t_span, min(MAX_LOG_BIN, gate.width / 30.0))
    nominal = (np.log1p(gate.hi) - np.log1p(gate.lo)) / grid.dl  # [-] bins
    lim_lo = np.ceil(grid.edge_index(tof_lim_lo)).astype(np.int64)
    lim_hi = np.floor(grid.edge_index(tof_lim_hi)).astype(np.int64)
    b_lo = np.rint(grid.edge_index(tof_pred * (1.0 + gate.lo))).astype(np.int64)
    b_hi = np.rint(grid.edge_index(tof_pred * (1.0 + gate.hi))).astype(np.int64)
    b_lo = np.clip(b_lo, lim_lo, lim_hi)
    b_hi = np.clip(b_hi, lim_lo, lim_hi)
    in_band = (b_hi - b_lo) >= max(1.0, 0.25 * nominal)
    gate_time = grid.edge_time(b_hi) - grid.edge_time(b_lo)  # [microsecond]

    half = _tile_half(var_us, var_vs)
    T = 2 * half + 1
    K = ATOM_PAD
    origin_r = np.rint(rs).astype(int) - half
    origin_c = np.rint(cs).astype(int) - half
    reach = 3.0 * np.sqrt(np.maximum(var_us, var_vs))

    trunk_jax = None
    if profile not in (None, "gaussian", "auto"):
        import json

        with open(profile, encoding="utf-8") as fh:
            prof = json.load(fh)
        u_, f_ = np.asarray(prof["u"], float), np.asarray(prof["f"], float)
        trunk_jax = (
            jnp.array(u_, dtype=jnp.float32),
            jnp.array(f_, dtype=jnp.float32),
            jnp.float32(np.trapezoid(f_ * u_, u_)),
        )
    elif profile == "auto" and show_progress:
        print(
            "  > Measured radial profile is not available with gating; Gaussian atoms."
        )

    admission_z = (
        0.0 if fp_target is None else calibrated_admission_z(n_peaks, fp_target)
    )
    use_l1_gate = bool(admission_z > 0.0)
    solve = _solve_tiles(trunk_jax is not None, use_l1_gate, bool(allow_negative))

    out = np.zeros((n_peaks, 5), dtype=np.float64)
    out[:, 1], out[:, 2] = rs, cs
    tile_bg = np.full(n_peaks, np.nan, dtype=np.float64)  # [photons/Pixel]
    n_dropped_neighbours = 0

    frame_groups = _group_frame_indices(frames, B)
    active_frames = [f for f in range(B) if len(frame_groups[f]) > 0]
    if not active_frames:
        out[:, 3] = np.sqrt(np.sqrt(np.maximum(var_us * var_vs - cov_uvs**2, 1e-6)))
        info["tof"] = 0.5 * (grid.edge_time(b_lo) + grid.edge_time(b_hi))
        info["bg"] = np.zeros(n_peaks, dtype=np.float64)
        info["tile_half"] = half
        return out, info

    max_n = max(len(frame_groups[f]) for f in active_frames)
    chunk_size = _select_tile_chunk(max_n)
    tiles_fn = _jit("tiles")
    snap_fn = _jit("snap")
    frame_iota = jnp.arange(chunk_size, dtype=jnp.int32)
    z_f32 = jnp.float32(admission_z)

    def _load_one(f_int: int):
        tof, pix_offsets = reader(f_int)
        return _prepare_flat_events(tof, pix_offsets, H, W, grid)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut_map = {active_frames[0]: pool.submit(_load_one, active_frames[0])}
        next_active_ptr = 1

        pbar = tqdm(
            range(B), desc="Time-of-flight gated solve", disable=not show_progress
        )
        for f in pbar:
            idx = frame_groups[f]
            n = len(idx)
            if n == 0:
                continue

            flat, dump = fut_map.pop(f).result()
            if next_active_ptr < len(active_frames):
                nf = active_frames[next_active_ptr]
                fut_map[nf] = pool.submit(_load_one, nf)
                next_active_ptr += 1

            valid_img = static_valid[f]
            valid_img_dev = jnp.asarray(valid_img)
            ct, p3 = _image_prefix_from_flat(flat, dump, valid_img, grid, need_ct=True)

            # 1. Sideband counts in fixed-size chunks, then release 3D prefix p3.
            org_r_f = origin_r[idx].astype(np.int32)
            org_c_f = origin_c[idx].astype(np.int32)
            b_lo_f = b_lo[idx]
            b_hi_f = b_hi[idx]
            r0 = np.clip(org_r_f, 0, H)
            r1 = np.clip(org_r_f + T, 0, H)
            c0 = np.clip(org_c_f, 0, W)
            c1 = np.clip(org_c_f + T, 0, W)
            sb_edges, side_ok = _sideband_edges(
                b_lo_f, b_hi_f, nominal, lim_lo[idx], lim_hi[idx], sideband_factor
            )
            sums_flat = _box_sums_chunked(
                p3, r0, r1, c0, c1, sb_edges.reshape(n, -1), chunk_size
            )
            del p3
            sums = sums_flat.reshape(n, 2, SIDEBAND_BINS + 1)
            counts = np.diff(sums, axis=2).astype(np.float64)
            widths = np.diff(grid.edge_time(sb_edges), axis=2)
            keep0 = (widths > 0) & side_ok[:, :, None]

            # 2. Sparse O(n) neighbor lookup + sideband exclusion + joint-atom table.
            keep, local, has_atom, dropped_f = _build_neighbor_plan(
                rs[idx],
                cs[idx],
                reach[idx],
                in_band[idx],
                b_lo_f,
                b_hi_f,
                sb_edges,
                keep0,
                half,
                K,
            )
            n_dropped_neighbours += dropped_f

            vp = np.pad(
                np.cumsum(np.cumsum(valid_img > 0.5, axis=0), axis=1), ((1, 0), (1, 0))
            )
            n_valid = vp[r1, c1] - vp[r0, c1] - vp[r1, c0] + vp[r0, c0]
            rate = _sideband_rates(
                counts.reshape(n, -1),
                widths.reshape(n, -1),
                keep.reshape(n, -1),
                keep0.reshape(n, -1),
                n_valid,
            )
            bg = rate * gate_time[idx]
            usable = in_band[idx] & np.isfinite(bg) & (n_valid > 0)
            tile_bg[idx] = np.where(usable, bg, np.nan)
            bg_safe = np.where(usable, np.maximum(bg, 1e-3), 1.0).astype(np.float32)

            # 3. Snap every reflection on its own gated tile in fixed-size chunks.
            b_lo_i32 = b_lo_f.astype(np.int32)
            b_hi_i32 = b_hi_f.astype(np.int32)
            loc_r = (rs[idx] - org_r_f).astype(np.float32)
            loc_c = (cs[idx] - org_c_f).astype(np.float32)
            snap_r = np.empty(n, dtype=np.float64)
            snap_c = np.empty(n, dtype=np.float64)
            cached_single_chunk = None
            for lo in range(0, n, chunk_size):
                hi = min(n, lo + chunk_size)
                m = hi - lo
                tiles_c, tile_valid_c = tiles_fn(
                    ct,
                    valid_img_dev,
                    jnp.asarray(_padded(org_r_f[lo:hi], chunk_size)),
                    jnp.asarray(_padded(org_c_f[lo:hi], chunk_size)),
                    jnp.asarray(_padded(b_lo_i32[lo:hi], chunk_size)),
                    jnp.asarray(_padded(b_hi_i32[lo:hi], chunk_size)),
                    T=T,
                )
                bg_tiles_c = jnp.broadcast_to(
                    jnp.asarray(_padded(bg_safe[lo:hi], chunk_size, 1.0))[
                        :, None, None
                    ],
                    (chunk_size, T, T),
                )
                sr_c, sc_c = snap_fn(
                    tiles_c,
                    bg_tiles_c,
                    tile_valid_c,
                    frame_iota,
                    jnp.asarray(_padded(loc_r[lo:hi], chunk_size)),
                    jnp.asarray(_padded(loc_c[lo:hi], chunk_size)),
                )
                snap_r[lo:hi] = np.asarray(sr_c, dtype=np.float64)[:m] + org_r_f[lo:hi]
                snap_c[lo:hi] = np.asarray(sc_c, dtype=np.float64)[:m] + org_c_f[lo:hi]
                if n <= chunk_size:
                    cached_single_chunk = (tiles_c, bg_tiles_c, tile_valid_c)

            # 4. Joint Poisson solve in fixed-size chunks (1 compile for entire run).
            a_r = np.where(has_atom, snap_r[local] - org_r_f[:, None], -1e4).astype(
                np.float32
            )
            a_c = np.where(has_atom, snap_c[local] - org_c_f[:, None], -1e4).astype(
                np.float32
            )
            g = idx[local]
            vu_f = np.where(has_atom, var_us[g], 1.0).astype(np.float32)
            vv_f = np.where(has_atom, var_vs[g], 1.0).astype(np.float32)
            cuv_f = np.where(has_atom, cov_uvs[g], 0.0).astype(np.float32)
            ok_f = has_atom.astype(np.float32)

            flux_all = np.empty(n, dtype=np.float64)
            sigma_all = np.empty(n, dtype=np.float64)
            for lo in range(0, n, chunk_size):
                hi = min(n, lo + chunk_size)
                m = hi - lo
                if cached_single_chunk is not None:
                    tiles_c, bg_tiles_c, tile_valid_c = cached_single_chunk
                else:
                    tiles_c, tile_valid_c = tiles_fn(
                        ct,
                        valid_img_dev,
                        jnp.asarray(_padded(org_r_f[lo:hi], chunk_size)),
                        jnp.asarray(_padded(org_c_f[lo:hi], chunk_size)),
                        jnp.asarray(_padded(b_lo_i32[lo:hi], chunk_size)),
                        jnp.asarray(_padded(b_hi_i32[lo:hi], chunk_size)),
                        T=T,
                    )
                    bg_tiles_c = jnp.broadcast_to(
                        jnp.asarray(_padded(bg_safe[lo:hi], chunk_size, 1.0))[
                            :, None, None
                        ],
                        (chunk_size, T, T),
                    )
                fl_c, sg_c = solve(
                    tiles_c,
                    bg_tiles_c,
                    tile_valid_c,
                    jnp.asarray(_padded(a_r[lo:hi], chunk_size, -1e4)),
                    jnp.asarray(_padded(a_c[lo:hi], chunk_size, -1e4)),
                    jnp.asarray(_padded(vu_f[lo:hi], chunk_size, 1.0)),
                    jnp.asarray(_padded(vv_f[lo:hi], chunk_size, 1.0)),
                    jnp.asarray(_padded(cuv_f[lo:hi], chunk_size, 0.0)),
                    jnp.asarray(_padded(ok_f[lo:hi], chunk_size, 0.0)),
                    z_f32,
                    trunk_jax,
                )
                flux_all[lo:hi] = np.asarray(fl_c, dtype=np.float64)[:m, 0]
                sigma_all[lo:hi] = np.asarray(sg_c, dtype=np.float64)[:m, 0]

            del ct, cached_single_chunk
            if not allow_negative or use_l1_gate:
                sigma_all[flux_all <= 0.0] = 0.0
            # A reflection whose gate falls outside the band, or whose
            # background could not be measured, was not observed.
            out[idx, 0] = np.where(usable, flux_all, 0.0)
            out[idx, 4] = np.where(usable, sigma_all, 0.0)
            out[idx, 1] = snap_r
            out[idx, 2] = snap_c

    if show_progress and n_dropped_neighbours:
        print(
            f"  > {n_dropped_neighbours} tiles had more than {K - 1} overlapping "
            "neighbours; the farthest were left out of the joint solve."
        )

    out[:, 3] = np.sqrt(np.sqrt(np.maximum(var_us * var_vs - cov_uvs**2, 1e-6)))
    info["tof"] = 0.5 * (grid.edge_time(b_lo) + grid.edge_time(b_hi))
    info["bg"] = np.where(np.isfinite(tile_bg), tile_bg, 0.0)
    info["tile_half"] = half
    return out, info
