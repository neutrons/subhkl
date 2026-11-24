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

Events are stored sorted by ``(flat_pixel, tof)``.  Multi-image slabs are
streamed directly to the accelerator where fused binary-search kernels
extract gated tiles and sideband counts in place without materialising
dense ``(H, W, nt)`` 3D cubes, and ``snap -> joint Poisson solve`` runs
across the full slab in ``TILE_CHUNK_LARGE`` batches with minimal host
synchronisation.
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
MIN_SPECTRUM_EVENTS_PER_BIN = 20  # fewer events per time bin: no usable spectrum
SIDEBAND_BINS = 8  # sub-intervals per side
STACK_HALF_BOX = 2  # 5x5 pixel core for the stacked time profile
STACK_MAX_IMAGES = 96  # images sampled for the stacked time profile
MAX_TIME_BINS = 2048
MAX_LOG_BIN = 1.0e-3  # [-] coarsest relative time bin
PEAK_PAD = 512  # retained for legacy compatibility
TILE_CHUNK_SMALL = 256  # fixed tile chunk for small batches/tests
TILE_CHUNK_LARGE = 2048  # fixed tile chunk for multi-image GPU slabs (1 compile)
EVENT_CHUNK_SMALL = 1 << 16  # 65,536 events
EVENT_CHUNK = 1 << 20  # 1,048,576 events (slab cap = 8 * EVENT_CHUNK)
EVENT_PAD = EVENT_CHUNK
SLAB_MAX_IMAGES = 32  # static image slots per GPU slab
SLAB_MAX_PEAKS = 32768  # cap cached GPU tiles per slab (~250 MB max VRAM)
BSEARCH_STEPS = 20  # covers up to 2^20 = 1,048,576 events in one pixel

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


def _edge_cutoffs_f32(grid: TimeGrid) -> np.ndarray:
    """IEEE-754 exact float32 lower-bound cutoffs matching ``_prepare_flat_events``."""
    b_idx = np.arange(grid.nt + 1, dtype=np.int64)
    c = grid.edge_time(b_idx).astype(np.float32)
    log_t0 = np.log(grid.t0)
    inv_dl = 1.0 / grid.dl

    def _tb(val_f32: np.ndarray) -> np.ndarray:
        return np.floor(
            (np.log(np.maximum(val_f32.astype(np.float64), 1e-6)) - log_t0) * inv_dl
        ).astype(np.int64)

    for _ in range(4):
        prev_c = np.nextafter(c, np.float32(-np.inf))
        step_down = (c > np.float32(0.0)) & (_tb(prev_c) >= b_idx)
        if not np.any(step_down):
            break
        c = np.where(step_down, prev_c, c)
    for _ in range(4):
        step_up = _tb(c) < b_idx
        if not np.any(step_up):
            break
        c = np.where(step_up, np.nextafter(c, np.float32(np.inf)), c)
    return c


def _bsearch_lower(tof_slab, lo, hi, cuts, steps=BSEARCH_STEPS):
    """Vectorised lower-bound index in sorted ``tof_slab[lo:hi]`` where ``tof >= cuts``."""
    import jax
    import jax.numpy as jnp

    target_shape = jnp.broadcast_shapes(lo.shape, hi.shape, cuts.shape)
    l0 = jnp.broadcast_to(lo, target_shape)
    h0 = jnp.broadcast_to(hi, target_shape)

    def _step(_i, state):
        l_cur, h_cur = state
        active = l_cur < h_cur
        mid = l_cur + ((h_cur - l_cur) >> 1)
        mid_safe = jnp.where(active, mid, jnp.int32(0))
        val = tof_slab[mid_safe]
        lt = val < cuts
        l_nxt = jnp.where(active & lt, mid + jnp.int32(1), l_cur)
        h_nxt = jnp.where(active & ~lt, mid, h_cur)
        return (l_nxt, h_nxt)

    l_fin, _ = jax.lax.fori_loop(0, steps, _step, (l0, h0))
    return l_fin


def _slab_box_impl(
    tof_slab, pix_offsets_slab, valid_slab, s_idx, org_r, org_c, cuts_e, box_t
):
    """[photons] (K, S - 1): interval counts in ``box_t x box_t`` pixel boxes."""
    import jax.numpy as jnp

    _, H, W = valid_slab.shape
    rr = org_r[:, None] + jnp.arange(box_t, dtype=jnp.int32)[None, :]
    cc = org_c[:, None] + jnp.arange(box_t, dtype=jnp.int32)[None, :]
    inside = ((rr >= 0) & (rr < H))[:, :, None] & ((cc >= 0) & (cc < W))[:, None, :]
    rr_c = jnp.clip(rr, 0, H - 1)[:, :, None]
    cc_c = jnp.clip(cc, 0, W - 1)[:, None, :]
    s_b = s_idx[:, None, None]
    ok = inside & (valid_slab[s_b, rr_c, cc_c] > 0)

    flat_pix = rr_c * jnp.int32(W) + cc_c
    pix_lo = jnp.where(ok, pix_offsets_slab[s_b, flat_pix], jnp.int32(0))
    pix_hi = jnp.where(ok, pix_offsets_slab[s_b, flat_pix + jnp.int32(1)], jnp.int32(0))

    env_cuts = jnp.stack([cuts_e[:, 0], cuts_e[:, -1]], axis=-1)[:, None, None, :]
    env_idx = _bsearch_lower(tof_slab, pix_lo[..., None], pix_hi[..., None], env_cuts)
    p_first = env_idx[..., 0]
    p_last = env_idx[..., 1]

    if cuts_e.shape[1] > 2:
        mid_idx = _bsearch_lower(
            tof_slab,
            p_first[..., None],
            p_last[..., None],
            cuts_e[:, 1:-1][:, None, None, :],
        )
        all_idx = jnp.concatenate(
            [p_first[..., None], mid_idx, p_last[..., None]], axis=-1
        )
    else:
        all_idx = jnp.stack([p_first, p_last], axis=-1)
    counts = ((all_idx[..., 1:] - all_idx[..., :-1]) * ok[..., None]).sum(axis=(1, 2))
    return counts


def _slab_tiles_impl(
    tof_slab, pix_offsets_slab, valid_slab, s_idx, org_r, org_c, cuts_20, T
):
    """Fused extraction of gated ``(T, T)`` tiles and 16 sideband sub-interval counts."""
    import jax.numpy as jnp

    _, H, W = valid_slab.shape
    rr = org_r[:, None] + jnp.arange(T, dtype=jnp.int32)[None, :]
    cc = org_c[:, None] + jnp.arange(T, dtype=jnp.int32)[None, :]
    inside = ((rr >= 0) & (rr < H))[:, :, None] & ((cc >= 0) & (cc < W))[:, None, :]
    rr_c = jnp.clip(rr, 0, H - 1)[:, :, None]
    cc_c = jnp.clip(cc, 0, W - 1)[:, None, :]
    s_b = s_idx[:, None, None]
    ok = inside & (valid_slab[s_b, rr_c, cc_c] > 0)

    flat_pix = rr_c * jnp.int32(W) + cc_c
    pix_lo = jnp.where(ok, pix_offsets_slab[s_b, flat_pix], jnp.int32(0))
    pix_hi = jnp.where(ok, pix_offsets_slab[s_b, flat_pix + jnp.int32(1)], jnp.int32(0))

    # Hierarchical bisection narrows sub-intervals before searching odd edges,
    # turning >85% of leaf searches into 0-width L1 cache hits on sparse panels.
    env_cuts = jnp.stack([cuts_20[:, 0], cuts_20[:, 19]], axis=-1)[:, None, None, :]
    env_idx = _bsearch_lower(tof_slab, pix_lo[..., None], pix_hi[..., None], env_cuts)
    p0 = env_idx[..., 0]
    p19 = env_idx[..., 1]

    gate_cuts = jnp.stack([cuts_20[:, 9], cuts_20[:, 10]], axis=-1)[:, None, None, :]
    gate_idx = _bsearch_lower(tof_slab, p0[..., None], p19[..., None], gate_cuts)
    p9 = gate_idx[..., 0]
    p10 = gate_idx[..., 1]

    # Boundary edges p8 (<= p9) and p11 (>= p10), plus sideband midpoints p4, p15.
    lvl1_lo = jnp.stack([p0, p0, p10, p10], axis=-1)
    lvl1_hi = jnp.stack([p9, p9, p19, p19], axis=-1)
    lvl1_cuts = jnp.stack(
        [cuts_20[:, 4], cuts_20[:, 8], cuts_20[:, 11], cuts_20[:, 15]], axis=-1
    )[:, None, None, :]
    lvl1 = _bsearch_lower(tof_slab, lvl1_lo, lvl1_hi, lvl1_cuts, steps=16)
    p4, p8, p11, p15 = lvl1[..., 0], lvl1[..., 1], lvl1[..., 2], lvl1[..., 3]

    lvl2_lo = jnp.stack([p0, p4, p11, p15], axis=-1)
    lvl2_hi = jnp.stack([p4, p8, p15, p19], axis=-1)
    lvl2_cuts = jnp.stack(
        [cuts_20[:, 2], cuts_20[:, 6], cuts_20[:, 13], cuts_20[:, 17]], axis=-1
    )[:, None, None, :]
    lvl2 = _bsearch_lower(tof_slab, lvl2_lo, lvl2_hi, lvl2_cuts, steps=16)
    p2, p6, p13, p17 = lvl2[..., 0], lvl2[..., 1], lvl2[..., 2], lvl2[..., 3]

    lvl3_lo = jnp.stack([p0, p2, p4, p6, p11, p13, p15, p17], axis=-1)
    lvl3_hi = jnp.stack([p2, p4, p6, p8, p13, p15, p17, p19], axis=-1)
    lvl3_cuts = jnp.stack(
        [
            cuts_20[:, 1],
            cuts_20[:, 3],
            cuts_20[:, 5],
            cuts_20[:, 7],
            cuts_20[:, 12],
            cuts_20[:, 14],
            cuts_20[:, 16],
            cuts_20[:, 18],
        ],
        axis=-1,
    )[:, None, None, :]
    lvl3 = _bsearch_lower(tof_slab, lvl3_lo, lvl3_hi, lvl3_cuts, steps=16)
    p1, p3, p5, p7 = lvl3[..., 0], lvl3[..., 1], lvl3[..., 2], lvl3[..., 3]
    p12, p14, p16, p18 = lvl3[..., 4], lvl3[..., 5], lvl3[..., 6], lvl3[..., 7]

    tile_counts = (p10 - p9) * ok
    ok_i32 = ok.astype(jnp.int32)
    left_all = jnp.stack([p0, p1, p2, p3, p4, p5, p6, p7, p8], axis=-1)
    right_all = jnp.stack([p11, p12, p13, p14, p15, p16, p17, p18, p19], axis=-1)
    sb_left = ((left_all[..., 1:] - left_all[..., :-1]) * ok_i32[..., None]).sum(
        axis=(1, 2)
    )
    sb_right = ((right_all[..., 1:] - right_all[..., :-1]) * ok_i32[..., None]).sum(
        axis=(1, 2)
    )
    sb_counts = jnp.concatenate([sb_left, sb_right], axis=-1)
    n_valid = ok_i32.sum(axis=(1, 2))
    return tile_counts.astype(jnp.float32), ok.astype(jnp.float32), sb_counts, n_valid


def _nbr_block_impl(
    q_frame,
    q_gid,
    q_r,
    q_c,
    q_reach,
    q_b_lo,
    q_b_hi,
    q_sb_l,
    q_sb_r,
    k_frame,
    k_gid,
    k_r,
    k_c,
    k_reach,
    k_in_band,
    k_b_lo,
    k_b_hi,
    half_f,
):
    """GPU block neighbor search: sideband blocking + top-7 overlapping atoms."""
    import jax
    import jax.numpy as jnp

    def _one_query(qf, qg, qr, qc, qrc, qblo, qbhi, qsbl, qsbr):
        same = (k_frame == qf) & k_in_band & (k_gid != qg)
        dr = jnp.abs(qr - k_r)
        dc = jnp.abs(qc - k_c)
        thr = half_f + k_reach
        ok_near = same & (dr <= thr) & (dc <= thr)

        ok_sb = ok_near & (qsbl[0] < k_b_hi) & (qsbr[-1] > k_b_lo)
        blk_l = jnp.any(
            ok_sb[None, :]
            & (qsbl[:-1, None] < k_b_hi[None, :])
            & (qsbl[1:, None] > k_b_lo[None, :]),
            axis=1,
        )
        blk_r = jnp.any(
            ok_sb[None, :]
            & (qsbr[:-1, None] < k_b_hi[None, :])
            & (qsbr[1:, None] > k_b_lo[None, :]),
            axis=1,
        )
        blk_16 = jnp.concatenate([blk_l, blk_r], axis=0)

        touch = qrc + k_reach
        ok_joint = (
            ok_near & (dr <= touch) & (dc <= touch) & (qblo < k_b_hi) & (qbhi > k_b_lo)
        )
        n_joint = jnp.sum(ok_joint.astype(jnp.int32))
        d2 = jnp.where(ok_joint, dr * dr + dc * dc, jnp.float32(jnp.inf))

        best_d2 = []
        best_gid = []
        cur_d2 = d2
        for _ in range(ATOM_PAD - 1):
            pos = jnp.argmin(cur_d2)
            best_d2.append(cur_d2[pos])
            best_gid.append(k_gid[pos])
            cur_d2 = cur_d2.at[pos].set(jnp.float32(jnp.inf))
        return blk_16, n_joint, jnp.stack(best_d2), jnp.stack(best_gid)

    return jax.vmap(_one_query)(
        q_frame, q_gid, q_r, q_c, q_reach, q_b_lo, q_b_hi, q_sb_l, q_sb_r
    )


def _merge_nbr_impl(
    acc_blk, acc_cnt, acc_d2, acc_gid, step_blk, step_cnt, step_d2, step_gid
):
    """Merge two fixed-shape top-7 neighbor candidate tables on GPU."""
    import jax
    import jax.numpy as jnp

    comb_d2 = jnp.concatenate([acc_d2, step_d2], axis=1)
    comb_gid = jnp.concatenate([acc_gid, step_gid], axis=1)

    def _pick_7(d2_row, gid_row):
        out_d2 = []
        out_gid = []
        cur = d2_row
        for _ in range(ATOM_PAD - 1):
            p = jnp.argmin(cur)
            out_d2.append(cur[p])
            out_gid.append(gid_row[p])
            cur = cur.at[p].set(jnp.float32(jnp.inf))
        return jnp.stack(out_d2), jnp.stack(out_gid)

    m_d2, m_gid = jax.vmap(_pick_7)(comb_d2, comb_gid)
    return acc_blk | step_blk, acc_cnt + step_cnt, m_d2, m_gid


def _sideband_bg_impl(counts, widths, keep0, blocked, n_valid, gate_time, in_band):
    """On-GPU robust median sideband background estimation (zero host sync)."""
    import jax.numpy as jnp

    keep = keep0 & ~blocked
    none_left = ~jnp.any(keep, axis=1)
    keep_eff = jnp.where(none_left[:, None], keep0, keep)
    cnt_f = counts.astype(jnp.float32)
    safe_w = jnp.where(keep_eff & (widths > 0), widths, jnp.float32(1.0))
    per_width = jnp.where(keep_eff & (widths > 0), cnt_f / safe_w, jnp.float32(jnp.nan))
    any_kept = jnp.any(keep_eff, axis=1)

    def _row_med_16(vals, msk):
        valid = msk & jnp.isfinite(vals)
        cnt = jnp.sum(valid.astype(jnp.int32), axis=1)
        s = jnp.sort(jnp.where(valid, vals, jnp.float32(jnp.inf)), axis=1)
        i0 = jnp.maximum((cnt - 1) // 2, 0)
        i1 = jnp.maximum(cnt // 2, 0)
        rows = jnp.arange(vals.shape[0], dtype=jnp.int32)
        med = jnp.float32(0.5) * (s[rows, i0] + s[rows, i1])
        return jnp.where(cnt > 0, med, jnp.float32(jnp.nan))

    med = _row_med_16(per_width, keep_eff)
    med_w = jnp.where(any_kept, _row_med_16(widths, keep_eff), jnp.float32(1.0))
    tol = (
        jnp.float32(5.0) * jnp.sqrt(jnp.maximum(med * med_w, jnp.float32(1.0))) / med_w
    )
    trimmed = keep_eff & (per_width <= (med + tol)[:, None])
    trimmed = jnp.where(jnp.any(trimmed, axis=1)[:, None], trimmed, keep_eff)
    total_c = jnp.sum(jnp.where(trimmed, cnt_f, jnp.float32(0.0)), axis=1)
    total_w = jnp.sum(jnp.where(trimmed, widths, jnp.float32(0.0)), axis=1)
    denom = total_w * jnp.maximum(n_valid.astype(jnp.float32), jnp.float32(1.0))
    rate = total_c / jnp.where(denom > 0, denom, jnp.float32(1.0))
    rate = jnp.where(any_kept & (total_w > 0), rate, jnp.float32(jnp.nan))
    bg = rate * gate_time
    usable = in_band & jnp.isfinite(bg) & (n_valid > 0)
    tile_bg_out = jnp.where(usable, bg, jnp.float32(jnp.nan))
    bg_safe = jnp.where(usable, jnp.maximum(bg, jnp.float32(1e-3)), jnp.float32(1.0))
    return tile_bg_out, bg_safe, usable


def _nbr_step_impl(
    acc_blk,
    acc_cnt,
    acc_d2,
    acc_gid,
    q_frame,
    q_gid,
    q_r,
    q_c,
    q_reach,
    q_b_lo,
    q_b_hi,
    q_sb_l,
    q_sb_r,
    k_frame,
    k_gid,
    k_r,
    k_c,
    k_reach,
    k_in_band,
    k_b_lo,
    k_b_hi,
    half_f,
):
    """Fused GPU neighbor block search + top-7 merge (zero intermediate dispatch)."""
    s_blk, s_cnt, s_d2, s_gid = _nbr_block_impl(
        q_frame,
        q_gid,
        q_r,
        q_c,
        q_reach,
        q_b_lo,
        q_b_hi,
        q_sb_l,
        q_sb_r,
        k_frame,
        k_gid,
        k_r,
        k_c,
        k_reach,
        k_in_band,
        k_b_lo,
        k_b_hi,
        half_f,
    )
    return _merge_nbr_impl(acc_blk, acc_cnt, acc_d2, acc_gid, s_blk, s_cnt, s_d2, s_gid)


def _slab_spectrum_impl(
    tof_dev,
    off_dev,
    n_m_i32,
    nt_i32,
    flat_pad_dev,
    inv_t0,
    inv_dl,
):
    """Batched GPU cumulative time-spectrum measure across all images in a slab."""
    import jax
    import jax.numpy as jnp

    M = off_dev.shape[0]
    nt_cap = flat_pad_dev.shape[0] - 1
    cap = tof_dev.shape[0]
    m_arange = jnp.arange(M, dtype=jnp.int32)
    last_m = jnp.maximum(n_m_i32 - jnp.int32(1), jnp.int32(0))
    total_ev = jnp.where(n_m_i32 > 0, off_dev[last_m, -1], jnp.int32(0))
    img_starts = jnp.where(m_arange < n_m_i32, off_dev[:, 0], total_ev)
    img_ends = jnp.where(m_arange < n_m_i32, off_dev[:, -1], total_ev)
    img_counts = img_ends - img_starts

    ev_idx = jnp.arange(cap, dtype=jnp.int32)
    m_id = jnp.searchsorted(img_ends, ev_idx, side="right")
    tb = jnp.floor(
        jnp.log(jnp.maximum(tof_dev, jnp.float32(1e-6)) * inv_t0) * inv_dl
    ).astype(jnp.int32)
    valid_ev = (ev_idx < total_ev) & (tb >= 0) & (tb < nt_i32)
    flat_bin = jnp.where(valid_ev, m_id * nt_cap + tb, M * nt_cap)
    hist_flat = jnp.zeros(M * nt_cap + 1, dtype=jnp.int32).at[flat_bin].add(1)
    hist = hist_flat[: M * nt_cap].reshape(M, nt_cap).astype(jnp.float32)

    kernel = jnp.ones((5,), dtype=jnp.float32) / jnp.float32(5.0)
    hist_sm = jax.vmap(lambda h: jnp.convolve(h, kernel, mode="same"))(hist)
    bin_mask = jnp.arange(nt_cap, dtype=jnp.int32)[None, :] < nt_i32
    hist_sm = jnp.where(bin_mask, hist_sm, jnp.float32(0.0))
    cum = jnp.concatenate(
        [jnp.zeros((M, 1), dtype=jnp.float32), jnp.cumsum(hist_sm, axis=1)], axis=1
    )
    enough = (img_counts >= MIN_SPECTRUM_EVENTS_PER_BIN * nt_i32) & (cum[:, -1] > 0.0)
    scale = flat_pad_dev[-1] / jnp.where(enough, cum[:, -1], jnp.float32(1.0))
    return jnp.where(enough[:, None], cum * scale[:, None], flat_pad_dev[None, :])


def _chunk_stage1_impl(
    sr_cap,
    sc_cap,
    tof_dev,
    off_dev,
    val_dev,
    measure_slab_dev,
    s_idx_c,
    org_r_c,
    org_c_c,
    cuts_20_c,
    b_lo_f32_c,
    b_hi_f32_c,
    sb_l_f32_c,
    sb_r_f32_c,
    side_ok_c,
    in_band_c,
    loc_r_c,
    loc_c_c,
    slab_gid_c,
    acc_blk,
    acc_cnt,
    acc_d2,
    acc_gid,
    lo_i32,
    n_s_i32,
    T,
):
    """Fused Stage-1 chunk kernel: tiles + sideband bg + snap + neighbor slots."""
    import jax
    import jax.numpy as jnp

    C = s_idx_c.shape[0]
    tiles_c, valid_c, sb_c, nv_c = _jit("tiles")(
        tof_dev, off_dev, val_dev, s_idx_c, org_r_c, org_c_c, cuts_20_c, T=T
    )
    sb_l_i = sb_l_f32_c.astype(jnp.int32)
    sb_r_i = sb_r_f32_c.astype(jnp.int32)
    b_lo_i = b_lo_f32_c.astype(jnp.int32)
    b_hi_i = b_hi_f32_c.astype(jnp.int32)
    m_l = measure_slab_dev[s_idx_c[:, None], sb_l_i]
    m_r = measure_slab_dev[s_idx_c[:, None], sb_r_i]
    widths_c = jnp.concatenate([jnp.diff(m_l, axis=1), jnp.diff(m_r, axis=1)], axis=1)
    gate_time_c = measure_slab_dev[s_idx_c, b_hi_i] - measure_slab_dev[s_idx_c, b_lo_i]
    keep0_c = (widths_c > 0.0) & jnp.repeat(side_ok_c, SIDEBAND_BINS, axis=1)
    tbg_c, bg_safe_c, usable_c = _sideband_bg_impl(
        sb_c, widths_c, keep0_c, acc_blk, nv_c, gate_time_c, in_band_c
    )
    bg_c = jnp.broadcast_to(bg_safe_c[:, None, None], (C, T, T))
    sr_c, sc_c = _jit("snap")(
        tiles_c,
        bg_c,
        valid_c,
        jnp.arange(C, dtype=jnp.int32),
        loc_r_c,
        loc_c_c,
    )
    sr_cap_out = jax.lax.dynamic_update_slice_in_dim(sr_cap, sr_c, lo_i32, axis=0)
    sc_cap_out = jax.lax.dynamic_update_slice_in_dim(sc_cap, sc_c, lo_i32, axis=0)

    active_c = slab_gid_c < n_s_i32
    self_gid = slab_gid_c[:, None]
    nbr_has = jnp.isfinite(acc_d2) & active_c[:, None]
    nbr_gid = jnp.where(nbr_has, acc_gid, self_gid)
    local_c = jnp.concatenate([self_gid, nbr_gid], axis=1)
    has_atom_c = jnp.concatenate([active_c[:, None], nbr_has], axis=1)
    dropped_c = jnp.sum(((acc_cnt > (ATOM_PAD - 1)) & active_c).astype(jnp.int32))
    return (
        sr_cap_out,
        sc_cap_out,
        tiles_c,
        bg_c,
        valid_c,
        sr_c,
        sc_c,
        tbg_c,
        usable_c,
        local_c,
        has_atom_c,
        dropped_c,
    )


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
            _JIT[name] = partial(jit, static_argnames=["box_t"])(_slab_box_impl)
        elif name == "tiles":
            _JIT[name] = partial(jit, static_argnames=["T"])(_slab_tiles_impl)
        elif name == "nbr_block":
            _JIT[name] = jit(_nbr_block_impl)
        elif name == "merge_nbr":
            _JIT[name] = jit(_merge_nbr_impl)
        elif name == "nbr_step":
            _JIT[name] = jit(_nbr_step_impl)
        elif name == "sideband_bg":
            _JIT[name] = jit(_sideband_bg_impl)
        elif name == "slab_spectrum":
            _JIT[name] = jit(_slab_spectrum_impl)
        elif name == "chunk_stage1":
            _JIT[name] = partial(jit, static_argnames=["T"], donate_argnums=(0, 1))(
                _chunk_stage1_impl
            )
        elif name == "snap":
            from subhkl.search.matrix_free import _snap_positions_impl

            _JIT[name] = jit(_snap_positions_impl)
    return _JIT[name]


def _chunk_stage2(
    has_profile, use_l1_gate, allow_negative, max_inflation=None, ridge="fixed"
):
    """Fused neighbor gather + joint Poisson solve for a static chunk."""
    key = (
        "chunk_stage2",
        has_profile,
        use_l1_gate,
        allow_negative,
        max_inflation,
        ridge,
    )
    if key not in _JIT:
        import jax.numpy as jnp
        from jax import jit

        solve_fn = _solve_tiles(
            has_profile, use_l1_gate, allow_negative, max_inflation, ridge
        )

        def _run_stage2(
            tiles_c,
            bg_c,
            valid_c,
            local_c,
            has_atom_c,
            org_r_c,
            org_c_c,
            org_r_cap,
            org_c_cap,
            sr_cap,
            sc_cap,
            vu_cap,
            vv_cap,
            cuv_cap,
            z_f32,
            prof,
        ):
            dr_int_c = (org_r_cap[local_c] - org_r_c[:, None]).astype(jnp.float32)
            dc_int_c = (org_c_cap[local_c] - org_c_c[:, None]).astype(jnp.float32)
            a_r_c = jnp.where(has_atom_c, sr_cap[local_c] + dr_int_c, jnp.float32(-1e4))
            a_c_c = jnp.where(has_atom_c, sc_cap[local_c] + dc_int_c, jnp.float32(-1e4))
            vu_c = jnp.where(has_atom_c, vu_cap[local_c], jnp.float32(1.0))
            vv_c = jnp.where(has_atom_c, vv_cap[local_c], jnp.float32(1.0))
            cuv_c = jnp.where(has_atom_c, cuv_cap[local_c], jnp.float32(0.0))
            ok_c = has_atom_c.astype(jnp.float32)
            fl_c, sg_c = solve_fn(
                tiles_c,
                bg_c,
                valid_c,
                a_r_c,
                a_c_c,
                vu_c,
                vv_c,
                cuv_c,
                ok_c,
                z_f32,
                prof,
            )
            return fl_c[:, 0], sg_c[:, 0]

        _JIT[key] = jit(_run_stage2)
    return _JIT[key]


def _solve_tiles(
    has_profile, use_l1_gate, allow_negative, max_inflation=None, ridge="fixed"
):
    """vmapped ``_solve_image_amplitudes`` over a batch of tiles."""
    key = ("solve", has_profile, use_l1_gate, allow_negative, max_inflation, ridge)
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
                max_variance_inflation=max_inflation,
                ridge=ridge,
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


def _iter_slabs(
    active_frames,
    frame_groups,
    reader,
    static_valid,
    images_batch=None,
):
    """Yield double-buffered ``(slab_frames, tof_slab, pix_offsets_slab, valid_slab)``."""
    _, H, W = static_valid.shape
    n_pix = H * W
    mult = 16 if int(EVENT_CHUNK) >= (1 << 20) else 8
    default_cap = max(mult * int(EVENT_CHUNK), 1)
    event_store = getattr(reader, "event_store", None)
    frame_img_keys = getattr(reader, "frame_img_keys", None)
    use_store = event_store is not None and images_batch is not None

    if use_store:
        slabs = []
        cur = []
        cur_ev = 0
        cur_pk = 0
        max_single_ev = 0
        for f in active_frames:
            f_int = int(f)
            img_key = (
                int(frame_img_keys[f_int]) if frame_img_keys is not None else f_int
            )
            ev_f = event_store.image_event_count(img_key)
            pk_f = len(frame_groups[f_int])
            max_single_ev = max(max_single_ev, ev_f)
            if cur and (
                len(cur) >= SLAB_MAX_IMAGES
                or cur_pk + pk_f > SLAB_MAX_PEAKS
                or cur_ev + ev_f > default_cap
            ):
                slabs.append(cur)
                cur = []
                cur_ev = 0
                cur_pk = 0
            cur.append(f_int)
            cur_ev += ev_f
            cur_pk += pk_f
        if cur:
            slabs.append(cur)

        slab_cap = default_cap
        if max_single_ev > slab_cap:
            slab_cap = 1 << int(np.ceil(np.log2(max_single_ev)))

        tof_bufs = [np.zeros(slab_cap, dtype=np.float32) for _ in range(2)]
        off_bufs = [
            np.zeros((SLAB_MAX_IMAGES, n_pix + 1), dtype=np.int32) for _ in range(2)
        ]
        val_bufs = [np.zeros((SLAB_MAX_IMAGES, H, W), dtype=np.uint8) for _ in range(2)]

        def _load_store_slab(s_idx: int):
            slot = s_idx % 2
            s_frames = slabs[s_idx]
            m = len(s_frames)
            img_keys = [
                int(frame_img_keys[f]) if frame_img_keys is not None else int(f)
                for f in s_frames
            ]
            event_store.read_slab(
                img_keys,
                images_batch[s_frames],
                out_tof=tof_bufs[slot],
                out_pix_offsets=off_bufs[slot],
            )
            val_buf = val_bufs[slot]
            val_buf.fill(0)
            val_buf[:m] = (static_valid[s_frames] > 0.5).astype(np.uint8)
            return s_frames, tof_bufs[slot], off_bufs[slot], val_buf

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_load_store_slab, 0)
            for s_idx in range(len(slabs)):
                res = fut.result()
                if s_idx + 1 < len(slabs):
                    fut = pool.submit(_load_store_slab, s_idx + 1)
                yield res
        return

    # Generic reader(frame) path (unit tests and custom event readers).
    tof_bufs = [np.zeros(default_cap, dtype=np.float32) for _ in range(2)]
    off_bufs = [
        np.zeros((SLAB_MAX_IMAGES, n_pix + 1), dtype=np.int32) for _ in range(2)
    ]
    val_bufs = [np.zeros((SLAB_MAX_IMAGES, H, W), dtype=np.uint8) for _ in range(2)]
    state = {"ptr": 0, "carry": None}

    def _load_callable_slab(slot: int):
        ptr = state["ptr"]
        carry = state["carry"]
        if ptr >= len(active_frames) and carry is None:
            return None
        tof_buf = tof_bufs[slot]
        off_buf = off_bufs[slot]
        val_buf = val_bufs[slot]
        val_buf.fill(0)

        s_frames = []
        cursor = 0
        cur_pk = 0
        while len(s_frames) < SLAB_MAX_IMAGES:
            if carry is not None:
                f_int, tof_f, off_f = carry
                carry = None
            elif ptr < len(active_frames):
                f_int = int(active_frames[ptr])
                ptr += 1
                tof_raw, off_raw = reader(f_int)
                tof_f = np.asarray(tof_raw, dtype=np.float32)
                off_f = np.asarray(off_raw, dtype=np.int64)
            else:
                break

            ev_f = len(tof_f)
            pk_f = len(frame_groups[f_int])
            if s_frames and (
                cur_pk + pk_f > SLAB_MAX_PEAKS or cursor + ev_f > len(tof_buf)
            ):
                carry = (f_int, tof_f, off_f)
                break
            if ev_f > len(tof_buf):
                cap_big = 1 << int(np.ceil(np.log2(ev_f)))
                tof_buf = np.zeros(cap_big, dtype=np.float32)
                tof_bufs[slot] = tof_buf

            m_pos = len(s_frames)
            if ev_f > 0:
                tof_buf[cursor : cursor + ev_f] = tof_f
            off_buf[m_pos] = (off_f + cursor).astype(np.int32)
            val_buf[m_pos] = (static_valid[f_int] > 0.5).astype(np.uint8)
            s_frames.append(f_int)
            cursor += ev_f
            cur_pk += pk_f

        if len(s_frames) < SLAB_MAX_IMAGES:
            off_buf[len(s_frames) :] = cursor
        state["ptr"] = ptr
        state["carry"] = carry
        return s_frames, tof_buf, off_buf, val_buf

    with ThreadPoolExecutor(max_workers=1) as pool:
        slot = 0
        fut = pool.submit(_load_callable_slab, slot)
        while True:
            item = fut.result()
            if item is None:
                break
            slot = 1 - slot
            fut = pool.submit(_load_callable_slab, slot)
            yield item


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
    _images_batch=None,
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
    B = int(static_valid.shape[0])
    edges = np.linspace(-search, search, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.float64)
    hb = STACK_HALF_BOX
    box_t = 2 * hb + 1
    nb = int(np.ceil(search / grid.dl)) + 1
    offs = np.arange(-nb, nb + 2)

    frame_groups = _group_frame_indices(frames, B)
    with_peaks = np.array([f for f in range(B) if len(frame_groups[f]) > 0], dtype=int)
    if len(with_peaks) == 0:
        return edges, hist
    if len(with_peaks) > STACK_MAX_IMAGES:
        pick = np.linspace(0, len(with_peaks) - 1, STACK_MAX_IMAGES).round().astype(int)
        with_peaks = with_peaks[pick]

    total_stack_peaks = sum(len(frame_groups[int(f)]) for f in with_peaks)
    chunk_size = _select_tile_chunk(total_stack_peaks)
    edge_cut_f32 = _edge_cutoffs_f32(grid)
    box_fn = _jit("box")

    pbar = tqdm(
        total=len(with_peaks),
        desc="Time-of-flight profile",
        disable=not show_progress,
    )
    for s_frames, tof_slab_np, off_slab_np, val_slab_np in _iter_slabs(
        with_peaks, frame_groups, reader, static_valid, images_batch=_images_batch
    ):
        import jax

        idx_parts = [frame_groups[f] for f in s_frames]
        idx_slab = np.concatenate(idx_parts)
        s_idx_slab = np.repeat(
            np.arange(len(s_frames), dtype=np.int32),
            [len(p) for p in idx_parts],
        )
        n_s = len(idx_slab)
        n_chunks = -(-n_s // chunk_size)
        n_pad = n_chunks * chunk_size
        org_r_s = (np.rint(rs[idx_slab]).astype(np.int32) - hb).astype(np.int32)
        org_c_s = (np.rint(cs[idx_slab]).astype(np.int32) - hb).astype(np.int32)
        centre = np.rint(grid.edge_index(tof_pred[idx_slab])).astype(np.int64)
        e = np.clip(centre[:, None] + offs[None, :], 0, grid.nt)
        cuts_e = edge_cut_f32[e]

        s_idx_ch = _padded(s_idx_slab, n_pad).reshape(n_chunks, chunk_size)
        org_r_ch = _padded(org_r_s, n_pad).reshape(n_chunks, chunk_size)
        org_c_ch = _padded(org_c_s, n_pad).reshape(n_chunks, chunk_size)
        cuts_ch = _padded(cuts_e, n_pad, 0.0).reshape(
            n_chunks, chunk_size, cuts_e.shape[1]
        )
        chunk_args = [
            (s_idx_ch[c], org_r_ch[c], org_c_ch[c], cuts_ch[c]) for c in range(n_chunks)
        ]
        tof_dev, off_dev, val_dev, chunk_args_dev = jax.device_put(
            (tof_slab_np, off_slab_np, val_slab_np, chunk_args)
        )
        cnt_devs = [
            box_fn(tof_dev, off_dev, val_dev, sc, rc, cc, ec, box_t=box_t)
            for sc, rc, cc, ec in chunk_args_dev
        ]
        del tof_dev, off_dev, val_dev, chunk_args_dev
        counts_pad = np.concatenate(jax.device_get(cnt_devs), axis=0)
        counts_slab = counts_pad[:n_s]

        x = (
            grid.edge_time(0.5 * (e[:, :-1] + e[:, 1:])) / tof_pred[idx_slab, None]
            - 1.0
        )
        hist += np.histogram(x.ravel(), bins=edges, weights=counts_slab.ravel())[0]
        pbar.update(len(s_frames))
    pbar.close()
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


def _spectrum_measure(tof, grid, flat_measure, smooth=5):
    """Cumulative time spectrum of one image on the grid's edges.

    The sideband background is scaled from the sidebands to the gate by the
    "length" of each interval.  Measured in plain time, that assumes the
    background rate is flat in time between them.  It is not: the incident
    spectrum is curved and falls off steeply at the band edges, so a
    reflection's background came out too high or too low by an amount that
    depends on its wavelength.  For weak reflections that offset was as
    large as their uncertainty, and because it repeats between
    measurements of the same reflection it survives merging as apparent
    signal.

    Measured instead in the image's own cumulative spectrum, the
    background is assumed proportional to what the image as a whole shows
    at each time.  Bragg peaks are a small, time-scattered part of an
    image's events and barely change that shape.  Intervals the spectrum
    does not reach (beyond a band edge) get zero length and drop out.

    Args:
        tof: [microsecond] all events of the image
        grid: TimeGrid of the gates and sidebands
        flat_measure: [microsecond] (nt + 1,) edge times minus the first,
            returned when the image has too few events for a spectrum
        smooth: [-] time bins the spectrum is averaged over
    Returns:
        [microsecond-equivalent] (nt + 1,) monotone measure of the edges
    """
    if len(tof) < MIN_SPECTRUM_EVENTS_PER_BIN * grid.nt:
        return flat_measure
    tb = np.floor(np.log(np.maximum(tof, 1e-6) * (1.0 / grid.t0)) * (1.0 / grid.dl))
    tb = tb.astype(np.int64)
    hist = np.bincount(tb[(tb >= 0) & (tb < grid.nt)], minlength=grid.nt)
    hist = np.convolve(hist.astype(np.float64), np.ones(smooth) / smooth, mode="same")
    cum = np.concatenate([[0.0], np.cumsum(hist)])
    if cum[-1] <= 0:
        return flat_measure
    return cum * (flat_measure[-1] / cum[-1])


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
    t_env_lo = np.minimum(b_lo_f, sb_edges[:, 0, 0]).astype(np.float64)
    t_env_hi = np.maximum(b_hi_f, sb_edges[:, 1, -1]).astype(np.float64)
    t_mid = 0.5 * (t_env_lo + t_env_hi)
    max_t_span = max(float(np.max(t_env_hi - t_env_lo)), 1.0)
    pts = np.column_stack((rs_f, cs_f, t_mid * (r_max / max_t_span)))
    pairs = cKDTree(pts).query_pairs(r=r_max + 1e-7, p=np.inf, output_type="ndarray")
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
    sideband_measure="spectrum",
    max_variance_inflation=1000.0,
    amplitude_ridge="fixed",
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
        max_variance_inflation: [-] reflections whose amplitude variance
            is inflated by more than this factor by overlapping neighbours
            in the same gate are censored (sigI = 0) instead of reported;
            see ``_solve_image_amplitudes_impl``.  Only their sum is
            determined, and a per-reflection table cannot carry a sum:
            whatever value is written for a member repeats between
            measurements and merges into correlation without signal.
            None or 0 keeps every row.
        amplitude_ridge: "fixed" or "counts"; what keeps the joint fit of
            overlapping reflections invertible (see the ``ridge`` argument
            of ``_solve_image_amplitudes_impl``).  "counts" gives
            unresolved reflections a sigma on the scale of the counts
            they could hold, but on a crowded lattice it also shrinks
            weak, partly overlapped reflections, so it is not the default.
        sideband_measure: "spectrum" scales the sideband background to the
            gate by the image's own time spectrum (see
            ``_spectrum_measure``); "time" assumes a rate flat in time
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
        _images_batch=images_batch,
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
    lim_lo = np.clip(np.ceil(grid.edge_index(tof_lim_lo)).astype(np.int64), 0, grid.nt)
    lim_hi = np.clip(
        np.floor(grid.edge_index(tof_lim_hi)).astype(np.int64), lim_lo, grid.nt
    )
    b_lo = np.rint(grid.edge_index(tof_pred * (1.0 + gate.lo))).astype(np.int64)
    b_hi = np.rint(grid.edge_index(tof_pred * (1.0 + gate.hi))).astype(np.int64)
    b_lo = np.clip(b_lo, lim_lo, lim_hi)
    b_hi = np.clip(b_hi, lim_lo, lim_hi)
    in_band = (b_hi - b_lo) >= max(1.0, 0.25 * nominal)
    if sideband_measure not in ("spectrum", "time"):
        raise ValueError(
            f"sideband_measure must be 'spectrum' or 'time', got {sideband_measure!r}"
        )
    # Edge times relative to the first edge: the measure of a time interval
    # when the background is taken as flat in time.
    flat_measure = grid.edge_time(np.arange(grid.nt + 1)) - grid.edge_time(0)
    edge_cut_f32 = _edge_cutoffs_f32(grid)

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
    max_inflation = (
        None if not max_variance_inflation else float(max_variance_inflation)
    )
    stage2_fn = _chunk_stage2(
        trunk_jax is not None,
        use_l1_gate,
        bool(allow_negative),
        max_inflation,
        amplitude_ridge,
    )

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

    import jax

    total_active_peaks = sum(len(frame_groups[f]) for f in active_frames)
    chunk_size = _select_tile_chunk(total_active_peaks)
    nbr_block_fn = _jit("nbr_block")
    nbr_step_fn = _jit("nbr_step")
    stage1_fn = _jit("chunk_stage1")
    slab_spectrum_fn = _jit("slab_spectrum") if sideband_measure == "spectrum" else None
    z_f32 = jnp.float32(admission_z)
    half_f32 = jnp.float32(half)

    nt_cap = max(2048, 1 << int(np.ceil(np.log2(max(grid.nt, 1)))))
    flat_pad_np = np.empty(nt_cap + 1, dtype=np.float32)
    flat_pad_np[: grid.nt + 1] = flat_measure.astype(np.float32)
    flat_pad_np[grid.nt + 1 :] = np.float32(flat_measure[-1])
    flat_pad_dev = jax.device_put(flat_pad_np)
    flat_measure_slab_dev = (
        jax.device_put(
            np.broadcast_to(
                flat_pad_np[None, :], (int(SLAB_MAX_IMAGES), nt_cap + 1)
            ).copy()
        )
        if sideband_measure != "spectrum"
        else None
    )
    nt_i32_dev = jax.device_put(np.int32(grid.nt))
    inv_t0_f32 = jnp.float32(1.0 / grid.t0)
    inv_dl_f32 = jnp.float32(1.0 / grid.dl)

    pbar = tqdm(total=B, desc="Time-of-flight gated solve", disable=not show_progress)
    progressed_to = 0

    for s_frames, tof_slab_np, off_slab_np, val_slab_np in _iter_slabs(
        active_frames, frame_groups, reader, static_valid, images_batch=images_batch
    ):
        idx_parts = [frame_groups[f] for f in s_frames]
        part_lens = [len(p) for p in idx_parts]
        idx_slab = np.concatenate(idx_parts)
        n_s = len(idx_slab)
        n_chunks = -(-n_s // chunk_size)
        n_pad = n_chunks * chunk_size
        gather_cap = max(int(SLAB_MAX_PEAKS), 1 << int(np.ceil(np.log2(max(n_pad, 1)))))

        slab_frames_s = np.repeat(np.arange(len(s_frames), dtype=np.int32), part_lens)
        s_idx_ch = _padded(slab_frames_s, n_pad, 0).reshape(n_chunks, chunk_size)
        q_frame_ch = _padded(slab_frames_s, n_pad, -1).reshape(n_chunks, chunk_size)
        k_frame_ch = _padded(slab_frames_s, n_pad, -2).reshape(n_chunks, chunk_size)
        slab_gid_ch = np.arange(n_pad, dtype=np.int32).reshape(n_chunks, chunk_size)

        org_r_slab = origin_r[idx_slab].astype(np.int32)
        org_c_slab = origin_c[idx_slab].astype(np.int32)
        org_r_ch = _padded(org_r_slab, n_pad, 0).reshape(n_chunks, chunk_size)
        org_c_ch = _padded(org_c_slab, n_pad, 0).reshape(n_chunks, chunk_size)

        rs_ch = _padded(rs[idx_slab].astype(np.float32), n_pad, 0.0).reshape(
            n_chunks, chunk_size
        )
        cs_ch = _padded(cs[idx_slab].astype(np.float32), n_pad, 0.0).reshape(
            n_chunks, chunk_size
        )
        reach_ch = _padded(reach[idx_slab].astype(np.float32), n_pad, 0.0).reshape(
            n_chunks, chunk_size
        )
        in_band_ch = _padded(in_band[idx_slab], n_pad, False).reshape(
            n_chunks, chunk_size
        )

        b_lo_slab = b_lo[idx_slab]
        b_hi_slab = b_hi[idx_slab]
        b_lo_ch = _padded(b_lo_slab.astype(np.float32), n_pad, 0.0).reshape(
            n_chunks, chunk_size
        )
        b_hi_ch = _padded(b_hi_slab.astype(np.float32), n_pad, 0.0).reshape(
            n_chunks, chunk_size
        )

        sb_edges_slab, side_ok_slab = _sideband_edges(
            b_lo_slab,
            b_hi_slab,
            nominal,
            lim_lo[idx_slab],
            lim_hi[idx_slab],
            sideband_factor,
        )
        sb_l_ch = _padded(
            sb_edges_slab[:, 0, :].astype(np.float32), n_pad, 0.0
        ).reshape(n_chunks, chunk_size, SIDEBAND_BINS + 1)
        sb_r_ch = _padded(
            sb_edges_slab[:, 1, :].astype(np.float32), n_pad, 0.0
        ).reshape(n_chunks, chunk_size, SIDEBAND_BINS + 1)
        side_ok_ch = _padded(side_ok_slab, n_pad, False).reshape(
            n_chunks, chunk_size, 2
        )

        cuts_20_pad = np.zeros((n_pad, 20), dtype=np.float32)
        cuts_20_pad[:n_s, 0:9] = edge_cut_f32[sb_edges_slab[:, 0, :]]
        cuts_20_pad[:n_s, 9] = edge_cut_f32[b_lo_slab]
        cuts_20_pad[:n_s, 10] = edge_cut_f32[b_hi_slab]
        cuts_20_pad[:n_s, 11:20] = edge_cut_f32[sb_edges_slab[:, 1, :]]
        cuts_20_ch = cuts_20_pad.reshape(n_chunks, chunk_size, 20)

        loc_r_ch = _padded(
            (rs[idx_slab] - org_r_slab).astype(np.float32), n_pad, 0.0
        ).reshape(n_chunks, chunk_size)
        loc_c_ch = _padded(
            (cs[idx_slab] - org_c_slab).astype(np.float32), n_pad, 0.0
        ).reshape(n_chunks, chunk_size)

        org_r_cap_np = _padded(org_r_slab, gather_cap, 0)
        org_c_cap_np = _padded(org_c_slab, gather_cap, 0)
        vu_cap_np = _padded(var_us[idx_slab].astype(np.float32), gather_cap, 1.0)
        vv_cap_np = _padded(var_vs[idx_slab].astype(np.float32), gather_cap, 1.0)
        cuv_cap_np = _padded(cov_uvs[idx_slab].astype(np.float32), gather_cap, 0.0)
        sr_cap_np = np.zeros(gather_cap, dtype=np.float32)
        sc_cap_np = np.zeros(gather_cap, dtype=np.float32)

        q_tuples_np = [
            (
                q_frame_ch[c],
                slab_gid_ch[c],
                rs_ch[c],
                cs_ch[c],
                reach_ch[c],
                b_lo_ch[c],
                b_hi_ch[c],
                sb_l_ch[c],
                sb_r_ch[c],
            )
            for c in range(n_chunks)
        ]
        k_tuples_np = [
            (
                k_frame_ch[c],
                slab_gid_ch[c],
                rs_ch[c],
                cs_ch[c],
                reach_ch[c],
                in_band_ch[c],
                b_lo_ch[c],
                b_hi_ch[c],
            )
            for c in range(n_chunks)
        ]
        st1_tuples_np = [
            (
                s_idx_ch[c],
                org_r_ch[c],
                org_c_ch[c],
                cuts_20_ch[c],
                side_ok_ch[c],
                in_band_ch[c],
                loc_r_ch[c],
                loc_c_ch[c],
                slab_gid_ch[c],
                np.int32(c * chunk_size),
            )
            for c in range(n_chunks)
        ]

        # Single batched H2D transfer for the entire slab (zero per-chunk PCIe setup).
        (
            tof_dev,
            off_dev,
            val_dev,
            org_r_cap_dev,
            org_c_cap_dev,
            sr_cap_dev,
            sc_cap_dev,
            vu_cap_dev,
            vv_cap_dev,
            cuv_cap_dev,
            n_s_dev,
            n_m_dev,
            q_devs,
            k_devs,
            st1_devs,
        ) = jax.device_put(
            (
                tof_slab_np,
                off_slab_np,
                val_slab_np,
                org_r_cap_np,
                org_c_cap_np,
                sr_cap_np,
                sc_cap_np,
                vu_cap_np,
                vv_cap_np,
                cuv_cap_np,
                np.int32(n_s),
                np.int32(len(s_frames)),
                q_tuples_np,
                k_tuples_np,
                st1_tuples_np,
            )
        )

        # Compute cumulative time-spectrum measures directly on GPU from tof_dev/off_dev.
        if slab_spectrum_fn is not None:
            measure_slab_dev = slab_spectrum_fn(
                tof_dev,
                off_dev,
                n_m_dev,
                nt_i32_dev,
                flat_pad_dev,
                inv_t0_f32,
                inv_dl_f32,
            )
        else:
            measure_slab_dev = flat_measure_slab_dev

        chunk_f_ranges = [
            (
                int(slab_frames_s[c * chunk_size]),
                int(slab_frames_s[min((c + 1) * chunk_size, n_s) - 1]),
            )
            for c in range(n_chunks)
        ]

        # 1. Pure @jit Stage-1 loop (zero eager JAX ops in Python).
        stage1_kept = []
        for c_i in range(n_chunks):
            fi_min, fi_max = chunk_f_ranges[c_i]
            q_c = q_devs[c_i]
            acc = None
            for c_j in range(n_chunks):
                fj_min, fj_max = chunk_f_ranges[c_j]
                if fj_max < fi_min or fj_min > fi_max:
                    continue
                if acc is None:
                    acc = nbr_block_fn(*q_c, *k_devs[c_j], half_f32)
                else:
                    acc = nbr_step_fn(*acc, *q_c, *k_devs[c_j], half_f32)

            (
                sr_c_in,
                org_r_c,
                org_c_c,
                cuts_c,
                sok_c,
                ib_c,
                lr_c,
                lc_c,
                gid_c,
                lo_dev,
            ) = st1_devs[c_i]
            b_lo_c, b_hi_c, sb_l_c, sb_r_c = q_c[5], q_c[6], q_c[7], q_c[8]
            (
                sr_cap_dev,
                sc_cap_dev,
                tiles_c,
                bg_c,
                valid_c,
                sr_c,
                sc_c,
                tbg_c,
                usable_c,
                local_c,
                has_atom_c,
                dropped_c,
            ) = stage1_fn(
                sr_cap_dev,
                sc_cap_dev,
                tof_dev,
                off_dev,
                val_dev,
                measure_slab_dev,
                sr_c_in,
                org_r_c,
                org_c_c,
                cuts_c,
                b_lo_c,
                b_hi_c,
                sb_l_c,
                sb_r_c,
                sok_c,
                ib_c,
                lr_c,
                lc_c,
                gid_c,
                *acc,
                lo_dev,
                n_s_dev,
                T=T,
            )
            stage1_kept.append(
                (
                    tiles_c,
                    bg_c,
                    valid_c,
                    local_c,
                    has_atom_c,
                    org_r_c,
                    org_c_c,
                    sr_c,
                    sc_c,
                    tbg_c,
                    usable_c,
                    dropped_c,
                )
            )

        if slab_spectrum_fn is not None:
            del measure_slab_dev
        del tof_dev, off_dev, val_dev, n_m_dev, q_devs, k_devs, st1_devs

        # 2. Pure @jit Stage-2 loop (neighbor gather + joint Poisson solve).
        out_devs = []
        for (
            tiles_c,
            bg_c,
            valid_c,
            local_c,
            has_atom_c,
            org_r_c,
            org_c_c,
            sr_c,
            sc_c,
            tbg_c,
            usable_c,
            dropped_c,
        ) in stage1_kept:
            fl_0, sg_0 = stage2_fn(
                tiles_c,
                bg_c,
                valid_c,
                local_c,
                has_atom_c,
                org_r_c,
                org_c_c,
                org_r_cap_dev,
                org_c_cap_dev,
                sr_cap_dev,
                sc_cap_dev,
                vu_cap_dev,
                vv_cap_dev,
                cuv_cap_dev,
                z_f32,
                trunk_jax,
            )
            out_devs.append((fl_0, sg_0, sr_c, sc_c, tbg_c, usable_c, dropped_c))

        del (
            stage1_kept,
            org_r_cap_dev,
            org_c_cap_dev,
            sr_cap_dev,
            sc_cap_dev,
            vu_cap_dev,
            vv_cap_dev,
            cuv_cap_dev,
        )

        # Single batched D2H transfer for the entire slab (zero eager concat kernels).
        out_host = jax.device_get(out_devs)
        del out_devs

        fl_np = np.concatenate([row[0] for row in out_host], axis=0)
        sg_np = np.concatenate([row[1] for row in out_host], axis=0)
        sr_np = np.concatenate([row[2] for row in out_host], axis=0)
        sc_np = np.concatenate([row[3] for row in out_host], axis=0)
        tbg_np = np.concatenate([row[4] for row in out_host], axis=0)
        usable_np = np.concatenate([row[5] for row in out_host], axis=0)
        n_dropped_neighbours += sum(int(row[6]) for row in out_host)

        usable_slab = usable_np[:n_s]
        tile_bg[idx_slab] = np.where(
            usable_slab, tbg_np[:n_s].astype(np.float64), np.nan
        )
        flux_slab = fl_np[:n_s].astype(np.float64)
        sigma_slab = sg_np[:n_s].astype(np.float64)
        if not allow_negative or use_l1_gate:
            sigma_slab[flux_slab <= 0.0] = 0.0
        out[idx_slab, 0] = np.where(usable_slab, flux_slab, 0.0)
        out[idx_slab, 4] = np.where(usable_slab, sigma_slab, 0.0)
        out[idx_slab, 1] = sr_np[:n_s].astype(np.float64) + org_r_slab
        out[idx_slab, 2] = sc_np[:n_s].astype(np.float64) + org_c_slab

        last_done = int(s_frames[-1]) + 1
        if last_done > progressed_to:
            pbar.update(last_done - progressed_to)
            progressed_to = last_done

    if B > progressed_to:
        pbar.update(B - progressed_to)
    pbar.close()

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
