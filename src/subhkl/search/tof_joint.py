"""Time-resolved joint fit of all reflections on a detector image.

Every reflection of an image is fitted at once, in position and time:

    counts(pixel, time bin) ~ Poisson(background + sum_k I_k P_k(pixel) T_k(bin))

P_k is the predicted spot (its projected 2D shape) and T_k the time profile
measured by stacking events around the predicted times.  Reflections that
share pixels are told apart by when their neutrons arrive, so nothing is
summed over a time gate before the fit and no reflection is limited to a
handful of neighbours.

Nothing the size of the voxel grid is factorised.  Each reflection carries
a window (pixels x time bins); the model cube is a scatter-add of those
windows, and the information matrix (reflections x reflections) is
assembled pair by pair from a neighbour list.  Because the profile is
separable, a pair's overlap is a (pixels, bins) sum on one reflection's
window with the other's spot and time profile evaluated there.

The matrix, scaled to unit diagonal, is solved by Cholesky.  The diagonal
of its inverse is each reflection's variance inflation: the variance of its
amplitude relative to what it would be with no neighbour.  Only where that
exceeds ``max_variance_inflation`` is the image redone with an
eigendecomposition that leaves out the directions the data do not
determine; reflections with a share in those directions are censored.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

from subhkl.search.tof_gate import (
    TofGate,
    gate_from_stack,
    stack_relative_tof,
)

BINS_PER_PROFILE = 20  # time bins across the support of the time profile
MAX_TIME_BINS = 1024
MIN_HALF, MAX_HALF = 4, 12  # [Pixel] half-size of a reflection's window
SPOT_REACH = 3.0  # [-] window half-size in units of the spot's major sigma
PAIR_BATCH = 64  # retained for compatibility
N_SCORING = 4  # Fisher-scoring steps
BG_BLUR = 6.0  # [Pixel] smoothing of the background map
MIN_SPOT_MASS = 0.65  # [-] share of the spot on usable pixels
MIN_TIME_MASS = 0.5  # [-] share of the time profile inside the band
LEVERAGE_LIMIT = 0.01  # [-] share in undetermined directions that censors a reflection
PEAK_QUANTUM = 512
EVENT_QUANTUM = 1 << 20
DEFAULT_NBR_PAD = (
    32  # initial GPU neighbour slots per reflection (escalates if exceeded)
)
NBR_ROW_CHUNK = 512  # row tile when n_pad > 2048 to bound temporary overlap memory
MAX_BATCH_IMAGES = 8  # images solved simultaneously on GPU via vmap
DEFAULT_BATCH_EVENTS = 16 << 20  # target events per multi-image slab (64 MiB float32)
MAX_BATCH_EVENTS = 32 << 20  # upper bound on automatic slab sizing (128 MiB float32)
TARGET_BATCH_VRAM_BYTES = 5 << 30  # ~5 GiB ceiling for transient multi-image cubes
PREFETCH_SLOTS = 4  # host ring-buffer slots (2 prefetch + H2D staging + in-flight GPU)

_JIT = {}


def stack_profile(edges, hist):
    """Cumulative time profile of the stacked Bragg excess.

    Args:
        edges, hist: histogram of t / t_pred - 1 from ``stack_relative_tof``
    Returns:
        (x [-], cdf [-]): edges of the support and the cumulative share of
        the excess up to each, or None when there is no excess
    """
    x = 0.5 * (edges[:-1] + edges[1:])
    outer = np.abs(x) > 0.6 * edges[-1]
    base = np.polyval(np.polyfit(x[outer], hist[outer], 1), x)
    excess = hist - base
    smooth = np.convolve(excess, np.ones(3) / 3, mode="same")
    noise = np.sqrt(np.maximum(base, 1.0) / 3)
    peak = int(np.argmax(np.where(~outer, smooth, -np.inf)))
    lo = peak
    while lo > 0 and not outer[lo - 1] and smooth[lo - 1] > noise[lo - 1]:
        lo -= 1
    hi = peak
    while hi < len(x) - 1 and not outer[hi + 1] and smooth[hi + 1] > noise[hi + 1]:
        hi += 1
    mass = np.maximum(excess[lo : hi + 1], 0.0)
    if mass.sum() <= 0:
        return None
    return edges[lo : hi + 2], np.concatenate([[0.0], np.cumsum(mass) / mass.sum()])


def _blur(x, sigma):
    """Separable Gaussian blur of a 2D map, edges reflected."""
    import jax
    import jax.numpy as jnp

    half = int(4 * sigma)
    k = jnp.exp(-0.5 * (jnp.arange(-half, half + 1) / sigma) ** 2).astype(x.dtype)
    k = k / k.sum()
    xp = jnp.pad(x, half, mode="symmetric")
    xp = jax.vmap(lambda row: jnp.convolve(row, k, mode="valid"))(xp)
    return jax.vmap(
        lambda col: jnp.convolve(col, k, mode="valid"), in_axes=1, out_axes=1
    )(xp)


def _bin_cutoffs_f32(l0: float, dl: float, nt: int) -> np.ndarray:
    """IEEE-754 exact float32 lower-bound cutoffs for log-time bins ``0..nt``."""
    b_idx = np.arange(nt + 1, dtype=np.int64)
    c = np.exp(l0 + b_idx * dl).astype(np.float32)

    def _tb(val_f32: np.ndarray) -> np.ndarray:
        return np.floor(
            (np.log(np.maximum(val_f32.astype(np.float64), 1e-6)) - l0) / dl
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


def _gpu_neighbour_list(r0, c0, b0, real, *, nb, half, k_pad):
    """Build padded ``(n, k_pad)`` overlapping neighbour table on GPU (self first)."""
    import jax.numpy as jnp
    from jax import lax

    n = r0.shape[0]
    cols = jnp.arange(n, dtype=jnp.int32)

    def _query_rows(q_idx):
        ov = (
            real[q_idx, None]
            & real[None, :]
            & (jnp.abs(r0[q_idx, None] - r0[None, :]) <= 2 * half)
            & (jnp.abs(c0[q_idx, None] - c0[None, :]) <= 2 * half)
            & (jnp.abs(b0[q_idx, None] - b0[None, :]) < nb)
        )
        deg_q = jnp.sum(ov.astype(jnp.int32), axis=1)
        scores = jnp.where(
            ov,
            jnp.where(q_idx[:, None] == cols[None, :], jnp.int32(2), jnp.int32(1)),
            jnp.int32(0),
        )
        top_vals, top_idx = lax.top_k(scores, k_pad)
        nbr_q = jnp.where(top_vals > 0, top_idx, jnp.int32(n))
        return nbr_q, deg_q

    if n <= 2048 or n % NBR_ROW_CHUNK != 0:
        nbr, deg = _query_rows(cols)
    else:
        nbr_b, deg_b = lax.map(_query_rows, cols.reshape(-1, NBR_ROW_CHUNK))
        nbr = nbr_b.reshape(n, k_pad)
        deg = deg_b.reshape(n)

    max_deg = jnp.max(jnp.where(real, deg, jnp.int32(0)))
    return nbr, max_deg


def _fit_from_counts_impl(
    N_2d,
    valid_px,
    r0,
    c0,
    pr,
    pc,
    vu,
    vv,
    cuv,
    T,
    b0,
    real,
    trunk,
    eig_floor,
    *,
    nt,
    half,
    k_pad,
    truncate,
):
    """Joint position-and-time Poisson fit of one image from ``(H*W, nt)`` counts."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.linalg import cho_solve, solve_triangular

    H_, W_ = valid_px.shape
    n, nb = T.shape
    size = H_ * W_ * nt
    f32 = jnp.float32
    valid_flat = valid_px.ravel().astype(f32)

    nbr, max_deg = _gpu_neighbour_list(r0, c0, b0, real, nb=nb, half=half, k_pad=k_pad)

    o = jnp.arange(-half, half + 1, dtype=jnp.int32)
    rr = (r0[:, None, None] + o[None, :, None] + 0 * o[None, None, :]).reshape(n, -1)
    cc = (c0[:, None, None] + 0 * o[None, :, None] + o[None, None, :]).reshape(n, -1)
    inb = (rr >= 0) & (rr < H_) & (cc >= 0) & (cc < W_)
    pflat = jnp.where(inb, rr * W_ + cc, 0)
    pix_ok = (inb & (valid_flat[pflat] > 0)).astype(f32)

    def spot(dv, du, vu_, vv_, cuv_):
        det = jnp.maximum(vu_ * vv_ - cuv_**2, 1e-6)
        m2 = (vv_ * du**2 - 2.0 * cuv_ * du * dv + vu_ * dv**2) / det
        if trunk is None:
            return jnp.exp(-0.5 * m2) / (2.0 * jnp.pi * jnp.sqrt(det))
        u, f, norm = trunk
        g = jnp.interp(jnp.sqrt(m2), u, f, left=f[0], right=0.0)
        return g / (2.0 * jnp.pi * jnp.sqrt(det) * norm)

    P = (
        spot(rr - pr[:, None], cc - pc[:, None], vu[:, None], vv[:, None], cuv[:, None])
        * pix_ok
    ).astype(f32)
    active = real & (P.sum(1) >= MIN_SPOT_MASS) & (T.sum(1) >= MIN_TIME_MASS)
    act = active.astype(f32)

    tbk = b0[:, None] + jnp.arange(nb, dtype=jnp.int32)[None, :]
    tbc = jnp.clip(tbk, 0, nt - 1)
    tb_ok = ((tbk >= 0) & (tbk < nt)).astype(f32)
    vox = pflat[:, :, None] * nt + tbc[:, None, :]
    a = P[:, :, None] * T[:, None, :]

    # Background: a pixel map times one time spectrum, from the voxels no
    # predicted reflection touches.
    occupied = (a > 0) & real[:, None, None]
    cover = (
        jnp.zeros(size, jnp.int32)
        .at[vox]
        .add(occupied.astype(jnp.int32))
        .reshape(H_ * W_, nt)
        > 0
    )
    free = (~cover) & (valid_flat[:, None] > 0)
    n_free_t = jnp.sum(free, axis=0, dtype=f32)
    nf_t = jnp.sum(jnp.where(free, N_2d, f32(0.0)), axis=0)
    nf_p = jnp.sum(jnp.where(free, N_2d, f32(0.0)), axis=1)
    spectrum = nf_t / jnp.maximum(n_free_t, f32(1.0))
    spectrum = jnp.convolve(spectrum, jnp.ones(5, f32) / f32(5.0), mode="same")
    weight = jnp.sum(jnp.where(free, spectrum[None, :], f32(0.0)), axis=1).reshape(
        H_, W_
    )
    bmap = _blur(nf_p.reshape(H_, W_), BG_BLUR) / jnp.maximum(
        _blur(weight, BG_BLUR), f32(1e-9)
    )
    bmap_flat = bmap.ravel()

    # Gather window counts and background once; Fisher scoring operates purely on
    # compact (n, P_win, nb) windows and precomputed (n, k_pad, .) pair factors.
    N_win = N_2d[pflat[:, :, None], tbc[:, None, :]]
    bg_win = jnp.maximum(
        bmap_flat[pflat][:, :, None] * spectrum[tbc][:, None, :], f32(1e-4)
    )
    val_win = pix_ok[:, :, None] * tb_ok[:, None, :]

    nbr_c = jnp.clip(nbr, 0, n - 1)
    nbr_act = (nbr < n).astype(f32) * act[nbr_c]

    def _prep_pair(k):
        j = nbr_c[k]
        pj = spot(
            rr[k][None, :] - pr[j][:, None],
            cc[k][None, :] - pc[j][:, None],
            vu[j][:, None],
            vv[j][:, None],
            cuv[j][:, None],
        )
        in_j = (jnp.abs(rr[k][None, :] - r0[j][:, None]) <= half) & (
            jnp.abs(cc[k][None, :] - c0[j][:, None]) <= half
        )
        rel = tbk[k][None, :] - b0[j][:, None]
        tj = jnp.take_along_axis(T[j], jnp.clip(rel, 0, nb - 1), axis=1)
        tj = tj * ((rel >= 0) & (rel < nb)) * tb_ok[k][None, :]
        return (
            (pj * in_j * pix_ok[k][None, :]).astype(f32),
            (tj * nbr_act[k][:, None]).astype(f32),
        )

    pj_all, tj_all = jax.vmap(_prep_pair)(jnp.arange(n, dtype=jnp.int32))
    t_pair = tj_all * T[:, None, :]
    prec = lax.Precision.HIGHEST

    def _eval_fisher(amp):
        amp_j = amp[nbr_c]
        model_win = jnp.einsum(
            "kjt,kjp->kpt", amp_j[:, :, None] * tj_all, pj_all, precision=prec
        )
        mu_win = jnp.maximum(bg_win + model_win, f32(0.3) * bg_win)
        inv_mu = val_win / mu_win
        sc = jnp.sum(a * ((N_win - mu_win) * inv_mu), axis=(1, 2)) * act
        x = jnp.einsum("kjp,kpt->kjt", pj_all, inv_mu * P[:, :, None], precision=prec)
        rows = jnp.sum(x * t_pair, axis=2) * act[:, None]
        Hm = (
            jnp.zeros((n, n), f32)
            .at[jnp.arange(n, dtype=jnp.int32)[:, None], nbr]
            .add(rows, mode="drop")
        )
        Hm = f32(0.5) * (Hm + Hm.T) + jnp.diag(f32(1.0) - act)
        return Hm, sc

    def _solver(Hm, need_infl: bool):
        d = jnp.sqrt(jnp.maximum(jnp.diag(Hm), f32(1e-30)))
        R = Hm / jnp.outer(d, d)
        if truncate:
            lam, V = jnp.linalg.eigh(R)
            kept = lam > eig_floor
            inv = jnp.where(kept, f32(1.0) / jnp.where(kept, lam, f32(1.0)), f32(0.0))

            def solve(rhs):
                return V @ ((V.T @ rhs) * inv)

            infl = jnp.sum(V**2 * inv[None, :], axis=1) if need_infl else None
            lev = (
                jnp.sum(jnp.where(kept[None, :], f32(0.0), V**2), axis=1)
                if need_infl
                else None
            )
            return d, solve, infl, lev, jnp.sum(~kept)
        L = jnp.linalg.cholesky(R)

        def solve(rhs):
            return cho_solve((L, True), rhs)

        if need_infl:
            linv = solve_triangular(L, jnp.eye(n, dtype=f32), lower=True)
            infl = jnp.sum(linv * linv, axis=0)
            lev = jnp.zeros(n, f32)
        else:
            infl, lev = None, None
        return d, solve, infl, lev, jnp.int32(0)

    def _step(_, amp):
        Hm, sc = _eval_fisher(amp)
        d, solve, *_ = _solver(Hm, False)
        return amp + solve(sc / d) / d

    amp = lax.fori_loop(0, N_SCORING, _step, jnp.zeros(n, f32))
    Hm, _ = _eval_fisher(amp)
    d, _, infl, lev, dropped = _solver(Hm, True)
    sigma = jnp.sqrt(jnp.maximum(infl, f32(0.0))) / d * act
    centre = jnp.clip(r0, 0, H_ - 1) * W_ + jnp.clip(c0, 0, W_ - 1)
    bg_centre = jnp.sum(
        jnp.maximum(bmap_flat[centre][:, None] * spectrum[tbc], f32(1e-4)) * (T > 0),
        axis=1,
    )
    return amp * act, sigma, infl, lev, dropped, active, bg_centre, max_deg


def _scatter_batch_events(tof_slab, off_batch, cuts_f32, *, nt):
    """Bin sorted slab events of ``M`` images into ``(M, n_pix, nt)`` float32 counts."""
    import jax.numpy as jnp

    M, n_pix_p1 = off_batch.shape
    n_pix = n_pix_p1 - 1
    ev_start = off_batch[0, 0]
    ev_end = off_batch[-1, -1]
    ev_idx = jnp.arange(tof_slab.shape[0], dtype=jnp.int32)
    in_sub = (ev_idx >= ev_start) & (ev_idx < ev_end)
    flat_ends = off_batch[:, 1:].ravel()
    sp = jnp.searchsorted(flat_ends, jnp.where(in_sub, ev_idx, ev_start), side="right")
    tb = jnp.searchsorted(
        cuts_f32, jnp.where(in_sub, tof_slab, jnp.float32(-1.0)), side="right"
    ) - jnp.int32(1)
    ok = in_sub & (tb >= 0) & (tb < nt)
    dump = M * n_pix * nt
    if dump < (1 << 30):
        vox = jnp.where(ok, sp * jnp.int32(nt) + tb, jnp.int32(dump))
    else:
        vox = jnp.where(
            ok,
            sp.astype(jnp.int64) * np.int64(nt) + tb.astype(jnp.int64),
            np.int64(dump),
        )
    counts = jnp.zeros(dump + 1, dtype=jnp.float32).at[vox].add(1.0)
    return counts[:dump].reshape(M, n_pix, nt)


def _fit_batch_impl(
    tof_slab,
    off_batch,
    cuts_f32,
    valid_batch,
    r0_b,
    c0_b,
    pr_b,
    pc_b,
    vu_b,
    vv_b,
    cuv_b,
    T_b,
    b0_b,
    real_b,
    trunk,
    eig_floor,
    *,
    nt,
    half,
    k_pad,
):
    """Fit a static batch of ``M`` images in parallel on GPU."""
    import jax

    N_batch = _scatter_batch_events(tof_slab, off_batch, cuts_f32, nt=nt)
    single = partial(
        _fit_from_counts_impl,
        trunk=trunk,
        eig_floor=eig_floor,
        nt=nt,
        half=half,
        k_pad=k_pad,
        truncate=False,
    )
    return jax.vmap(single)(
        N_batch,
        valid_batch,
        r0_b,
        c0_b,
        pr_b,
        pc_b,
        vu_b,
        vv_b,
        cuv_b,
        T_b,
        b0_b,
        real_b,
    )


def _fit_single_slab_impl(
    tof_slab,
    off_1d,
    cuts_f32,
    valid_px,
    r0,
    c0,
    pr,
    pc,
    vu,
    vv,
    cuv,
    T,
    b0,
    real,
    trunk,
    eig_floor,
    *,
    nt,
    half,
    k_pad,
    truncate,
):
    """Single-image fallback for eigendecomposition truncation or dense overlaps."""
    N_2d = _scatter_batch_events(tof_slab, off_1d[None, :], cuts_f32, nt=nt)[0]
    return _fit_from_counts_impl(
        N_2d,
        valid_px,
        r0,
        c0,
        pr,
        pc,
        vu,
        vv,
        cuv,
        T,
        b0,
        real,
        trunk,
        eig_floor,
        nt=nt,
        half=half,
        k_pad=k_pad,
        truncate=truncate,
    )


def _fit_batch_fn(nt, half, k_pad, has_trunk):
    key = ("batch", nt, half, k_pad, has_trunk)
    if key not in _JIT:
        from jax import jit

        _JIT[key] = jit(partial(_fit_batch_impl, nt=nt, half=half, k_pad=k_pad))
    return _JIT[key]


def _fit_single_fn(nt, half, k_pad, truncate, has_trunk):
    key = ("single", nt, half, k_pad, truncate, has_trunk)
    if key not in _JIT:
        from jax import jit

        _JIT[key] = jit(
            partial(
                _fit_single_slab_impl,
                nt=nt,
                half=half,
                k_pad=k_pad,
                truncate=truncate,
            )
        )
    return _JIT[key]


def _choose_n_pad(max_n: int) -> int:
    if max_n <= 512:
        return max(16, 1 << int(np.ceil(np.log2(max(max_n, 1)))))
    return -(-max_n // PEAK_QUANTUM) * PEAK_QUANTUM


def _precompute_time_shares(
    b0_all: np.ndarray,
    lt: np.ndarray,
    tof_lim_lo: np.ndarray,
    tof_lim_hi: np.ndarray,
    prof_l: np.ndarray,
    prof_cdf: np.ndarray,
    l0: float,
    dl: float,
    nt: int,
    nb: int,
) -> np.ndarray:
    """Vectorised ``(n_peaks, nb)`` float32 profile time shares cut at band edges."""
    n_peaks = len(b0_all)
    T_all = np.empty((n_peaks, nb), dtype=np.float32)
    bins = np.arange(nb + 1, dtype=np.int64)[None, :]
    chunk = 65536
    for lo in range(0, n_peaks, chunk):
        hi = min(lo + chunk, n_peaks)
        be = b0_all[lo:hi, None] + bins
        t_edges = l0 + be * dl
        cdf = np.interp(
            t_edges - lt[lo:hi, None], prof_l, prof_cdf, left=0.0, right=1.0
        )
        lo_l = np.log(tof_lim_lo[lo:hi])[:, None]
        hi_l = np.log(tof_lim_hi[lo:hi])[:, None]
        inside = np.clip(
            (np.minimum(t_edges[:, 1:], hi_l) - np.maximum(t_edges[:, :-1], lo_l)) / dl,
            0.0,
            1.0,
        )
        T_all[lo:hi] = (
            np.diff(cdf, axis=1) * inside * ((be[:, :-1] >= 0) & (be[:, :-1] < nt))
        ).astype(np.float32)
    return T_all


def integrate_reflections_tof_joint(
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
    max_variance_inflation=1000.0,
    static_valid=None,
    profile="gaussian",
    border_width=0,
    allow_negative=False,
    show_progress=False,
):
    """Integrate every predicted reflection by one time-resolved fit per image.

    Same row contract as ``integrate_reflections_matrix_free``:
    [intensity, r, c, sigma_eff, sigI] per input peak, censored rows
    carrying sigI = 0.  The intensity is the reflection's total flux (the
    whole time profile, not the share inside a gate).

    Args:
        images_batch: [photons/Pixel] (B, H, W), the event histograms
        reader: frame -> (tof [microsecond], pixel offsets) of that image
        frames: [-] image index per peak
        rs, cs: [Pixel] predicted positions
        tof_pred: [microsecond] predicted time of flight per peak
        tof_lim_lo, tof_lim_hi: [microsecond] per peak, the times of the
            wavelength band edges at its pixel; the time profile is cut
            there
        var_us, var_vs, cov_uvs: [Pixel^2] projected 2D shape per peak
        gate: None to measure the time profile from the stacked events, or
            a TofGate, taken as a flat profile between its edges
        gate_coverage: [-] share of the stacked excess inside the reported
            gate (the fit itself uses the whole profile)
        gate_search: [-] half-range of t / t_pred - 1 searched for it
        max_variance_inflation: [-] where neighbours inflate a
            reflection's variance beyond this, the image is refitted
            without the directions whose eigenvalue (of the unit-diagonal
            information matrix) is below its inverse, and reflections with
            a share in those directions are censored.  None or 0 keeps
            every direction.
        static_valid: [-] optional (B, H, W), 1 = usable pixel
        profile: "gaussian" or a path to a radial profile JSON
        border_width: [Pixel] rim of each image treated as unusable
        allow_negative: keep signed amplitudes; otherwise negative
            estimates are reported as zero
    Returns:
        (rows, info): info holds the gate, the per-peak centre time ``tof``
        [microsecond], the background ``bg`` [photons/Pixel] under the
        time profile at the peak's pixel, the stacked time profile and the
        fit's bookkeeping (``inflation`` per peak, ``dropped`` directions)
    """
    import jax
    import jax.numpy as jnp

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

    # The stacked profile around the predicted times: where the Bragg
    # excess sits relative to the predictions, and its shape.
    from subhkl.search.tof_gate import MAX_LOG_BIN, TimeGrid

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
        prof = stack_profile(edges, hist) if gate is not None else None
        if gate is None or prof is None:
            raise ValueError(
                "No Bragg excess in the stacked time-of-flight profile within "
                f"+-{gate_search:.3f} of the predicted times: the predictions do "
                "not line up with the events in time. Check the solve (cell "
                "scale, orientation) and the flight path, give the gate "
                "explicitly (--tof-gate lo,hi), or integrate the whole band "
                "(--tof-gate off)."
            )
        prof_x, prof_cdf = prof
    else:
        gate = TofGate(
            lo=gate.lo, hi=gate.hi, coverage=gate.coverage, source=gate.source
        )
        prof_x, prof_cdf = np.array([gate.lo, gate.hi]), np.array([0.0, 1.0])
    info["gate"] = gate
    # Profile edges in log time relative to the prediction.
    prof_l = np.log1p(prof_x)
    if show_progress:
        print(
            f"  > Time profile: t/t_pred - 1 in [{prof_x[0]:+.4f}, {prof_x[-1]:+.4f}] "
            f"({gate.source}); one joint fit per image in position and time"
        )

    l0 = np.log(t_span[0])
    dl = max(
        (prof_l[-1] - prof_l[0]) / BINS_PER_PROFILE,
        np.log(t_span[1] / t_span[0]) / MAX_TIME_BINS,
    )
    nt = int(np.ceil(np.log(t_span[1] / t_span[0]) / dl))
    nb = int(np.ceil((prof_l[-1] - prof_l[0]) / dl)) + 1
    n_pix = H * W
    sig_major = np.sqrt(np.maximum(var_us, var_vs))
    half = int(
        np.clip(np.ceil(SPOT_REACH * np.percentile(sig_major, 99)), MIN_HALF, MAX_HALF)
    )

    has_trunk = profile not in (None, "gaussian", "auto")
    trunk = None
    if has_trunk:
        import json

        with open(profile, encoding="utf-8") as fh:
            prof_json = json.load(fh)
        u_, f_ = np.asarray(prof_json["u"], float), np.asarray(prof_json["f"], float)
        trunk = (
            jnp.asarray(u_, dtype=jnp.float32),
            jnp.asarray(f_, dtype=jnp.float32),
            jnp.float32(np.trapezoid(f_ * u_, u_)),
        )
    eig_floor = 1.0 / max_variance_inflation if max_variance_inflation else 0.0
    eig_floor_f32 = np.float32(eig_floor)

    # Precompute per-reflection coordinates and time-profile shares once on CPU.
    lt = np.log(tof_pred)
    b0_all = np.floor((lt + prof_l[0] - l0) / dl).astype(np.int64)
    T_all = _precompute_time_shares(
        b0_all, lt, tof_lim_lo, tof_lim_hi, prof_l, prof_cdf, l0, dl, nt, nb
    )
    r0_all = np.rint(rs).astype(np.int32)
    c0_all = np.rint(cs).astype(np.int32)
    rs_f32 = rs.astype(np.float32)
    cs_f32 = cs.astype(np.float32)
    b0_i32 = b0_all.astype(np.int32)
    cuts_dev = jax.device_put(_bin_cutoffs_f32(l0, dl, nt))

    out = np.zeros((n_peaks, 5), dtype=np.float64)
    out[:, 1], out[:, 2] = rs, cs
    out[:, 3] = np.sqrt(np.sqrt(np.maximum(var_us * var_vs - cov_uvs**2, 1e-6)))
    inflation = np.zeros(n_peaks)
    bg_out = np.zeros(n_peaks)
    dropped_total, redone = 0, 0

    order = np.argsort(frames, kind="stable")
    bounds = np.searchsorted(frames[order], np.arange(B + 1))
    frame_ix = [order[bounds[f] : bounds[f + 1]] for f in range(B)]
    active_frames = [f for f in range(B) if len(frame_ix[f]) > 0]

    if active_frames:
        max_n = max(len(frame_ix[f]) for f in active_frames)
        n_pad = _choose_n_pad(max_n)
        k_pad = min(n_pad, DEFAULT_NBR_PAD)
        p_win = (2 * half + 1) ** 2
        per_img_bytes = max(
            n_pix * nt * 8,
            n_pad * k_pad * p_win * 16 + n_pad * n_pad * 16,
            1,
        )
        batch_imgs = int(
            np.clip(
                TARGET_BATCH_VRAM_BYTES // per_img_bytes,
                1,
                min(MAX_BATCH_IMAGES, len(active_frames)),
            )
        )

        event_store = getattr(reader, "event_store", None)
        frame_img_keys = getattr(reader, "frame_img_keys", None)
        use_store = event_store is not None

        if use_store:
            img_ev_counts = [
                event_store.image_event_count(
                    int(frame_img_keys[f]) if frame_img_keys is not None else int(f)
                )
                for f in active_frames
            ]
        else:
            img_ev_counts = [
                max(int(np.rint(images_batch[f]).sum()), 0) for f in active_frames
            ]
        total_est_ev = int(sum(img_ev_counts))
        max_single_ev = max(img_ev_counts) if img_ev_counts else 0
        p95_ev = int(np.ceil(np.percentile(img_ev_counts, 95))) if img_ev_counts else 0
        target_ev = min(
            MAX_BATCH_EVENTS, max(DEFAULT_BATCH_EVENTS, p95_ev * batch_imgs)
        )
        if total_est_ev <= target_ev:
            batch_ev_cap = max(1024, 1 << int(np.ceil(np.log2(max(total_est_ev, 1)))))
        else:
            batch_ev_cap = max(
                target_ev,
                1 << int(np.ceil(np.log2(max(max_single_ev, 1)))),
            )
            batch_ev_cap = 1 << int(np.ceil(np.log2(max(batch_ev_cap, 1))))

        # Allocate ring-buffer slots so HDF5 decompression, H2D transfer, and GPU
        # multi-image execution overlap continuously without host-device stalls.
        slots = [
            {
                "tof": np.zeros(batch_ev_cap, dtype=np.float32),
                "off": np.zeros((batch_imgs, n_pix + 1), dtype=np.int32),
                "val": np.zeros((batch_imgs, H, W), dtype=np.float32),
                "r0": np.zeros((batch_imgs, n_pad), dtype=np.int32),
                "c0": np.zeros((batch_imgs, n_pad), dtype=np.int32),
                "pr": np.zeros((batch_imgs, n_pad), dtype=np.float32),
                "pc": np.zeros((batch_imgs, n_pad), dtype=np.float32),
                "vu": np.ones((batch_imgs, n_pad), dtype=np.float32),
                "vv": np.ones((batch_imgs, n_pad), dtype=np.float32),
                "cuv": np.zeros((batch_imgs, n_pad), dtype=np.float32),
                "T": np.zeros((batch_imgs, n_pad, nb), dtype=np.float32),
                "b0": np.zeros((batch_imgs, n_pad), dtype=np.int32),
                "real": np.zeros((batch_imgs, n_pad), dtype=bool),
            }
            for _ in range(PREFETCH_SLOTS)
        ]

        def _stage_peaks(buf, batch_frames):
            buf["val"].fill(0.0)
            buf["r0"].fill(0)
            buf["c0"].fill(0)
            buf["pr"].fill(0.0)
            buf["pc"].fill(0.0)
            buf["vu"].fill(1.0)
            buf["vv"].fill(1.0)
            buf["cuv"].fill(0.0)
            buf["T"].fill(0.0)
            buf["b0"].fill(0)
            buf["real"].fill(False)
            for i, f in enumerate(batch_frames):
                ix = frame_ix[f]
                n = len(ix)
                buf["val"][i] = static_valid[f]
                buf["r0"][i, :n] = r0_all[ix]
                buf["c0"][i, :n] = c0_all[ix]
                buf["pr"][i, :n] = rs_f32[ix]
                buf["pc"][i, :n] = cs_f32[ix]
                buf["vu"][i, :n] = var_us[ix]
                buf["vv"][i, :n] = var_vs[ix]
                buf["cuv"][i, :n] = cov_uvs[ix]
                buf["T"][i, :n] = T_all[ix]
                buf["b0"][i, :n] = b0_i32[ix]
                buf["real"][i, :n] = True

        def _iter_batches():
            if use_store:
                batches = []
                cur = []
                cur_ev = 0
                for f, ev_f in zip(active_frames, img_ev_counts):
                    if cur and (len(cur) >= batch_imgs or cur_ev + ev_f > batch_ev_cap):
                        batches.append(cur)
                        cur = []
                        cur_ev = 0
                    cur.append(int(f))
                    cur_ev += ev_f
                if cur:
                    batches.append(cur)

                def _load_store(b_idx: int):
                    buf = slots[b_idx % PREFETCH_SLOTS]
                    b_frames = batches[b_idx]
                    m = len(b_frames)
                    img_keys = [
                        int(frame_img_keys[f]) if frame_img_keys is not None else f
                        for f in b_frames
                    ]
                    _, _, total_ev = event_store.read_slab(
                        img_keys,
                        images_batch[b_frames],
                        out_tof=buf["tof"],
                        out_pix_offsets=buf["off"],
                    )
                    if m < batch_imgs:
                        buf["off"][m:] = total_ev
                    _stage_peaks(buf, b_frames)
                    return b_frames, buf

                n_batches = len(batches)
                n_prefetch = min(2, n_batches)
                with ThreadPoolExecutor(max_workers=n_prefetch) as pool:
                    in_flight = [pool.submit(_load_store, k) for k in range(n_prefetch)]
                    next_submit = n_prefetch
                    while in_flight:
                        res = in_flight.pop(0).result()
                        if next_submit < n_batches:
                            in_flight.append(pool.submit(_load_store, next_submit))
                            next_submit += 1
                        yield res
                return

            state = {"ptr": 0, "carry": None}

            def _load_callable(slot_idx: int):
                ptr = state["ptr"]
                carry = state["carry"]
                if ptr >= len(active_frames) and carry is None:
                    return None
                buf = slots[slot_idx]
                tof_buf = buf["tof"]
                off_buf = buf["off"]
                b_frames = []
                cursor = 0
                while len(b_frames) < batch_imgs:
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
                    if b_frames and cursor + ev_f > len(tof_buf):
                        carry = (f_int, tof_f, off_f)
                        break
                    if ev_f > len(tof_buf):
                        cap_big = 1 << int(np.ceil(np.log2(ev_f)))
                        tof_buf = np.zeros(cap_big, dtype=np.float32)
                        buf["tof"] = tof_buf
                    i = len(b_frames)
                    if ev_f > 0:
                        tof_buf[cursor : cursor + ev_f] = tof_f
                    off_buf[i] = (off_f + cursor).astype(np.int32)
                    b_frames.append(f_int)
                    cursor += ev_f
                if len(b_frames) < batch_imgs:
                    off_buf[len(b_frames) :] = cursor
                _stage_peaks(buf, b_frames)
                state["ptr"] = ptr
                state["carry"] = carry
                return b_frames, buf

            with ThreadPoolExecutor(max_workers=1) as pool:
                slot_idx = 0
                fut = pool.submit(_load_callable, slot_idx)
                while True:
                    res = fut.result()
                    if res is None:
                        break
                    slot_idx = (slot_idx + 1) % PREFETCH_SLOTS
                    fut = pool.submit(_load_callable, slot_idx)
                    yield res

        batch_fn = _fit_batch_fn(nt, half, k_pad, has_trunk)
        pbar = tqdm(
            total=n_peaks, desc="Joint time-resolved fit", disable=not show_progress
        )

        def _launch_gpu(b_frames, buf):
            dev_args = jax.device_put(
                (
                    buf["tof"],
                    buf["off"],
                    buf["val"],
                    buf["r0"],
                    buf["c0"],
                    buf["pr"],
                    buf["pc"],
                    buf["vu"],
                    buf["vv"],
                    buf["cuv"],
                    buf["T"],
                    buf["b0"],
                    buf["real"],
                )
            )
            gpu_res = batch_fn(
                dev_args[0],
                dev_args[1],
                cuts_dev,
                *dev_args[2:],
                trunk,
                eig_floor_f32,
            )
            return b_frames, dev_args, gpu_res

        def _retire_gpu(pending):
            nonlocal dropped_total, redone
            b_frames, dev_args, gpu_res = pending
            (
                amp_b,
                sig_b,
                infl_b,
                lev_b,
                _dropped_b,
                active_b,
                bg_b,
                max_deg_b,
            ) = jax.device_get(gpu_res)

            for i, f in enumerate(b_frames):
                ix = frame_ix[f]
                n = len(ix)
                amp_i = amp_b[i]
                sig_i = sig_b[i]
                infl_i = infl_b[i]
                lev_i = lev_b[i]
                active_i = active_b[i]
                bg_i = bg_b[i]
                k_req = k_pad

                if int(max_deg_b[i]) > k_pad:
                    k_req = min(n_pad, 1 << int(np.ceil(np.log2(int(max_deg_b[i])))))
                    single_res = _fit_single_fn(nt, half, k_req, False, has_trunk)(
                        dev_args[0],
                        dev_args[1][i],
                        cuts_dev,
                        *(d[i] for d in dev_args[2:]),
                        trunk,
                        eig_floor_f32,
                    )
                    (
                        amp_i,
                        sig_i,
                        infl_i,
                        lev_i,
                        _dropped_i,
                        active_i,
                        bg_i,
                        _,
                    ) = jax.device_get(single_res)

                fitted = active_i[:n]
                bad = not np.all(np.isfinite(amp_i[:n][fitted])) or not np.all(
                    np.isfinite(sig_i[:n][fitted])
                )
                over = (
                    eig_floor > 0
                    and fitted.any()
                    and infl_i[:n][fitted].max() > max_variance_inflation
                )
                if bad or over:
                    floor = eig_floor if eig_floor > 0 else 1e-6
                    trunc_res = _fit_single_fn(nt, half, k_req, True, has_trunk)(
                        dev_args[0],
                        dev_args[1][i],
                        cuts_dev,
                        *(d[i] for d in dev_args[2:]),
                        trunk,
                        np.float32(floor),
                    )
                    (
                        amp_i,
                        sig_i,
                        infl_i,
                        lev_i,
                        dropped_i,
                        active_i,
                        bg_i,
                        _,
                    ) = jax.device_get(trunc_res)
                    fitted = active_i[:n]
                    dropped_total += int(dropped_i)
                    redone += 1

                keep = (
                    fitted
                    & (lev_i[:n] < LEVERAGE_LIMIT)
                    & np.isfinite(amp_i[:n])
                    & (sig_i[:n] > 0)
                )
                flux = amp_i[:n] if allow_negative else np.maximum(amp_i[:n], 0.0)
                out[ix, 0] = np.where(keep, flux, 0.0)
                out[ix, 4] = np.where(keep, sig_i[:n], 0.0)
                inflation[ix] = np.where(fitted, infl_i[:n], 0.0)
                bg_out[ix] = bg_i[:n]
                pbar.update(n)

        pending = None
        for b_frames, buf in _iter_batches():
            nxt = _launch_gpu(b_frames, buf)
            if pending is not None:
                _retire_gpu(pending)
            pending = nxt
        if pending is not None:
            _retire_gpu(pending)
        pbar.close()

    info["tof"] = tof_pred * (1.0 + 0.5 * (gate.lo + gate.hi))
    info["bg"] = bg_out
    info["tile_half"] = half
    info["inflation"] = inflation
    info["dropped"] = dropped_total
    info["images_truncated"] = redone
    if show_progress:
        n_ok = int((out[:, 4] > 0).sum())
        print(
            f"  > {n_ok} of {n_peaks} reflections fitted; variance inflation above 10 for "
            f"{100.0 * np.mean(inflation[out[:, 4] > 0] > 10.0) if n_ok else 0.0:.1f}%; "
            f"{dropped_total} undetermined direction(s) left out on {redone} image(s)"
        )
    return out, info
