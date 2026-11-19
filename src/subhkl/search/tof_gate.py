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
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
from tqdm import tqdm

from subhkl.io.events import ANGSTROM_M_PER_US, box_events

ATOM_PAD = 8  # atoms per tile: the reflection and up to seven neighbours
MIN_TILE_HALF = 10
MAX_TILE_HALF = 30
SIDEBAND_BINS = 8  # sub-intervals per side
STACK_HALF_BOX = 2  # 5x5 pixel core for the stacked time profile

_SOLVE_TILES_JIT = {}
_SNAP_TILES_JIT = None


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


def predicted_tof(wavelength, l1, l2):
    """[microsecond] time of flight of wavelength [Angstrom] over l1 + l2 [m]."""
    return np.asarray(wavelength, dtype=np.float64) * (l1 + l2) / ANGSTROM_M_PER_US


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


def stack_relative_tof(
    reader, n_frames, frames, rs, cs, tof_pred, shape, search=0.05, n_bins=400
):
    """Histogram of t_event / t_pred - 1 near all predicted positions.

    Args:
        reader: frame -> (tof [microsecond], pixel offsets) of that image
        frames, rs, cs: image index and predicted position [Pixel] per peak
        tof_pred: [microsecond] predicted time of flight per peak
        shape: (H, W) of the images
        search: [-] half-range of the relative coordinate
    Returns:
        bin edges [-] and summed counts [photons]
    """
    H, W = shape
    edges = np.linspace(-search, search, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.float64)
    scale = n_bins / (2.0 * search)
    hb = STACK_HALF_BOX
    for f in range(n_frames):
        idx = np.flatnonzero(frames == f)
        if len(idx) == 0:
            continue
        tof, pix_offsets = reader(f)
        for k in idx:
            ri, ci = round(float(rs[k])), round(float(cs[k]))
            r0, r1 = max(ri - hb, 0), min(ri + hb + 1, H)
            c0, c1 = max(ci - hb, 0), min(ci + hb + 1, W)
            if r0 >= r1 or c0 >= c1:
                continue
            t, _pix = box_events(tof, pix_offsets, W, r0, r1, c0, c1)
            b = np.floor((t / tof_pred[k] - 1.0 + search) * scale).astype(np.int64)
            b = b[(b >= 0) & (b < n_bins)]
            hist += np.bincount(b, minlength=n_bins)
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
    kern = np.ones(5) / 5.0
    smooth = np.convolve(excess, kern, mode="same")
    peak = int(np.argmax(np.where(inner, smooth, -np.inf)))

    # The support of the excess: outward from the maximum until the smoothed
    # excess reaches the baseline noise, bounded by the inner range.
    noise = np.sqrt(np.maximum(base, 1.0) / 5.0)
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


def _sideband_rate(t, ok, n_valid, gate_lo, gate_hi, lim_lo, lim_hi, blocked, factor):
    """Background rate [photons/Pixel/microsecond] from time sidebands.

    Args:
        t: [microsecond] events of the tile
        ok: [-] per event, True on a usable pixel
        n_valid: [-] number of usable pixels in the tile
        gate_lo, gate_hi: [microsecond] the gate
        lim_lo, lim_hi: [microsecond] edges of the wavelength band here
        blocked: list of (lo, hi) [microsecond] gates of other reflections
            reaching this tile; sub-intervals touching them are left out
        factor: [-] sideband length on each side in units of the gate width
    """
    width = gate_hi - gate_lo
    guard = 0.5 * width
    span = factor * width
    t_ok = t[ok]
    counts, widths = [], []
    for a, b in (
        (gate_lo - guard - span, gate_lo - guard),
        (gate_hi + guard, gate_hi + guard + span),
    ):
        a, b = max(a, lim_lo), min(b, lim_hi)
        if b - a < 0.25 * width:
            continue
        sub = np.linspace(a, b, SIDEBAND_BINS + 1)
        c = np.histogram(t_ok, bins=sub)[0].astype(np.float64)
        w = np.diff(sub)
        keep = np.ones(SIDEBAND_BINS, dtype=bool)
        for blo, bhi in blocked:
            keep &= ~((sub[:-1] < bhi) & (sub[1:] > blo))
        counts.append(c[keep])
        widths.append(w[keep])
    if not counts or sum(len(c) for c in counts) == 0:
        return np.nan
    counts = np.concatenate(counts)
    widths = np.concatenate(widths)
    # An unpredicted reflection in a sideband is a count far above the
    # others; leave such sub-intervals out.  Counts are compared per unit
    # width because clipping at the band edge can shorten one side.
    per_width = counts / widths
    med = np.median(per_width)
    tol = 5.0 * np.sqrt(max(med * np.median(widths), 1.0)) / np.median(widths)
    keep = per_width <= med + tol
    if not keep.any():
        keep[:] = True
    return float(counts[keep].sum() / (widths[keep].sum() * max(n_valid, 1)))


def _solve_tiles(profile, use_l1_gate, allow_negative):
    """vmapped ``_solve_image_amplitudes`` over a batch of tiles."""
    key = (profile is not None, use_l1_gate, allow_negative)
    if key not in _SOLVE_TILES_JIT:
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

        _SOLVE_TILES_JIT[key] = jit(solve)
    return _SOLVE_TILES_JIT[key]


def _snap_tiles(tiles, bgs, valid, r, c):
    global _SNAP_TILES_JIT
    if _SNAP_TILES_JIT is None:
        from jax import jit

        from subhkl.search.matrix_free import _snap_positions_impl

        _SNAP_TILES_JIT = jit(_snap_positions_impl)
    import jax.numpy as jnp

    return _SNAP_TILES_JIT(tiles, bgs, valid, jnp.arange(tiles.shape[0]), r, c)


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
    chunk_size=2048,
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

    edges, hist = stack_relative_tof(
        reader, B, frames, rs, cs, tof_pred, (H, W), search=gate_search
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

    g_lo = np.clip(tof_pred * (1.0 + gate.lo), tof_lim_lo, tof_lim_hi)
    g_hi = np.clip(tof_pred * (1.0 + gate.hi), tof_lim_lo, tof_lim_hi)
    in_band = g_hi - g_lo > 0.25 * tof_pred * gate.width

    half = _tile_half(var_us, var_vs)
    T = 2 * half + 1
    K = ATOM_PAD

    tiles = np.zeros((n_peaks, T, T), dtype=np.float32)
    tile_valid = np.zeros((n_peaks, T, T), dtype=np.float32)
    tile_bg = np.full(n_peaks, np.nan, dtype=np.float64)  # [photons/Pixel]
    atom_idx = np.full((n_peaks, K), -1, dtype=np.int64)
    atom_idx[:, 0] = np.arange(n_peaks)
    origin_r = np.rint(rs).astype(int) - half
    origin_c = np.rint(cs).astype(int) - half
    reach = 3.0 * np.sqrt(np.maximum(var_us, var_vs))
    n_dropped_neighbours = 0

    pbar = tqdm(range(B), desc="Time-of-flight gating", disable=not show_progress)
    for f in pbar:
        idx = np.flatnonzero(frames == f)
        if len(idx) == 0:
            continue
        tof, pix_offsets = reader(f)
        valid_img = static_valid[f]

        # Neighbours: footprints reaching the tile, gates overlapping in time.
        dr = np.abs(rs[idx, None] - rs[None, idx])
        dc = np.abs(cs[idx, None] - cs[None, idx])
        near = (dr <= half + reach[None, idx]) & (dc <= half + reach[None, idx])
        overlap = (g_lo[idx, None] < g_hi[None, idx]) & (
            g_hi[idx, None] > g_lo[None, idx]
        )
        np.fill_diagonal(near, False)

        for a, k in enumerate(idx):
            if not in_band[k]:
                continue
            r0, c0 = origin_r[k], origin_c[k]
            rr0, rr1 = max(r0, 0), min(r0 + T, H)
            cc0, cc1 = max(c0, 0), min(c0 + T, W)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            t, pix = box_events(tof, pix_offsets, W, rr0, rr1, cc0, cc1)
            box_shape = (rr1 - rr0, cc1 - cc0)
            v_box = valid_img[rr0:rr1, cc0:cc1]
            in_gate = (t >= g_lo[k]) & (t < g_hi[k])
            counts = np.bincount(pix[in_gate], minlength=box_shape[0] * box_shape[1])
            sl = (slice(rr0 - r0, rr1 - r0), slice(cc0 - c0, cc1 - c0))
            tiles[k][sl] = counts.reshape(box_shape)
            tile_valid[k][sl] = v_box

            neigh = np.flatnonzero(near[a])
            blocked = [(g_lo[idx[j]], g_hi[idx[j]]) for j in neigh if in_band[idx[j]]]
            rate = _sideband_rate(
                t,
                v_box.ravel()[pix] > 0.5,
                int(v_box.sum()),
                g_lo[k],
                g_hi[k],
                tof_lim_lo[k],
                tof_lim_hi[k],
                blocked,
                sideband_factor,
            )
            if not np.isfinite(rate):
                rate = _sideband_rate(
                    t,
                    v_box.ravel()[pix] > 0.5,
                    int(v_box.sum()),
                    g_lo[k],
                    g_hi[k],
                    tof_lim_lo[k],
                    tof_lim_hi[k],
                    [],
                    sideband_factor,
                )
            tile_bg[k] = rate * (g_hi[k] - g_lo[k])

            joint = neigh[overlap[a, neigh]]
            if len(joint) > K - 1:
                d2 = dr[a, joint] ** 2 + dc[a, joint] ** 2
                joint = joint[np.argsort(d2)[: K - 1]]
                n_dropped_neighbours += 1
            atom_idx[k, 1 : 1 + len(joint)] = idx[joint]

    usable = in_band & np.isfinite(tile_bg)
    if show_progress and n_dropped_neighbours:
        print(
            f"  > {n_dropped_neighbours} tiles had more than {K - 1} overlapping "
            "neighbours; the farthest were left out of the joint solve."
        )

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

    if fp_target is None:
        admission_z = 0.0
    else:
        admission_z = calibrated_admission_z(n_peaks, fp_target)
    use_l1_gate = bool(admission_z > 0.0)
    solve = _solve_tiles(trunk_jax, use_l1_gate, bool(allow_negative))

    # Snap each reflection on its own gated tile, then solve with every
    # atom (the reflection and its neighbours) at its snapped position.
    bg_safe = np.where(usable, np.maximum(tile_bg, 1e-3), 1.0).astype(np.float32)
    snap_r = rs.copy()
    snap_c = cs.copy()
    for lo in range(0, n_peaks, chunk_size):
        sl = slice(lo, min(lo + chunk_size, n_peaks))
        n = sl.stop - sl.start
        bg_tiles = jnp.broadcast_to(jnp.array(bg_safe[sl])[:, None, None], (n, T, T))
        sr, sc = _snap_tiles(
            jnp.array(tiles[sl]),
            bg_tiles,
            jnp.array(tile_valid[sl]),
            jnp.array((rs[sl] - origin_r[sl]).astype(np.float32)),
            jnp.array((cs[sl] - origin_c[sl]).astype(np.float32)),
        )
        snap_r[sl] = np.array(sr, dtype=np.float64) + origin_r[sl]
        snap_c[sl] = np.array(sc, dtype=np.float64) + origin_c[sl]

    out = np.zeros((n_peaks, 5), dtype=np.float64)
    has_atom = atom_idx >= 0
    safe_idx = np.where(has_atom, atom_idx, 0)
    pbar = tqdm(
        range(0, n_peaks, chunk_size),
        desc="Gated amplitude solve",
        disable=not show_progress,
    )
    for lo in pbar:
        sl = slice(lo, min(lo + chunk_size, n_peaks))
        n = sl.stop - sl.start
        ai, ok = safe_idx[sl], has_atom[sl]
        a_r = np.where(ok, snap_r[ai] - origin_r[sl, None], -1e4).astype(np.float32)
        a_c = np.where(ok, snap_c[ai] - origin_c[sl, None], -1e4).astype(np.float32)
        a_vu = np.where(ok, var_us[ai], 1.0).astype(np.float32)
        a_vv = np.where(ok, var_vs[ai], 1.0).astype(np.float32)
        a_cuv = np.where(ok, cov_uvs[ai], 0.0).astype(np.float32)
        bg_tiles = jnp.broadcast_to(jnp.array(bg_safe[sl])[:, None, None], (n, T, T))
        flux, sigma = solve(
            jnp.array(tiles[sl]),
            bg_tiles,
            jnp.array(tile_valid[sl]),
            jnp.array(a_r),
            jnp.array(a_c),
            jnp.array(a_vu),
            jnp.array(a_vv),
            jnp.array(a_cuv),
            jnp.array(ok.astype(np.float32)),
            jnp.float32(admission_z),
            trunk_jax,
        )
        flux = np.array(flux[:, 0], dtype=np.float64)
        sigma = np.array(sigma[:, 0], dtype=np.float64)
        if not allow_negative or use_l1_gate:
            sigma[flux <= 0.0] = 0.0
        out[sl, 0] = flux
        out[sl, 4] = sigma

    out[:, 1] = snap_r
    out[:, 2] = snap_c
    out[:, 3] = np.sqrt(np.sqrt(np.maximum(var_us * var_vs - cov_uvs**2, 1e-6)))
    # A reflection whose gate falls outside the band, or whose background
    # could not be measured, was not observed.
    out[~usable, 0] = 0.0
    out[~usable, 4] = 0.0

    info["tof"] = 0.5 * (g_lo + g_hi)
    info["bg"] = np.where(usable, tile_bg, 0.0)
    info["tile_half"] = half
    return out, info
