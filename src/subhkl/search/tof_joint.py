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
PAIR_BATCH = 64  # reflections whose neighbour overlaps are evaluated together
N_SCORING = 4  # Fisher-scoring steps
BG_BLUR = 6.0  # [Pixel] smoothing of the background map
MIN_SPOT_MASS = 0.65  # [-] share of the spot on usable pixels
MIN_TIME_MASS = 0.5  # [-] share of the time profile inside the band
LEVERAGE_LIMIT = 0.01  # [-] share in undetermined directions that censors a reflection
PEAK_QUANTUM = 1024
EVENT_QUANTUM = 1 << 20

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


def _fit_image_impl(
    events,
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
    nbr,
    trunk,
    eig_floor,
    *,
    nt,
    half,
    truncate,
):
    """Fit one image.

    Args:
        events: [-] (E,) flat voxel index (pixel * nt + bin) per event;
            out-of-range values are padding
        valid_px: [-] (H, W), 1 = usable pixel
        r0, c0: [Pixel] (n,) window centres (rounded positions)
        pr, pc: [Pixel] (n,) predicted positions
        vu, vv, cuv: [Pixel^2] (n,) projected spot covariance
        T: [-] (n, nb) share of the time profile in each of the
            reflection's time bins
        b0: [-] (n,) first time bin of each reflection
        real: [-] (n,) False for padding
        nbr: [-] (n, K) neighbour list including the reflection itself;
            values >= n are padding
        trunk: None, or (u, f, norm): measured radial profile in model
            sigmas and its integral of f(m) m dm
        eig_floor: [-] smallest eigenvalue kept when ``truncate``
    Returns:
        flux [photons], sigma [photons], variance inflation [-], share in
        dropped directions [-], number of dropped directions, fitted
        mask, background under the profile at the centre [photons/Pixel]
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.linalg import cho_solve

    H_, W_ = valid_px.shape
    n, nb = T.shape
    size = H_ * W_ * nt
    f32 = jnp.float32
    valid_flat = valid_px.ravel().astype(f32)

    o = jnp.arange(-half, half + 1)
    rr = (r0[:, None, None] + o[None, :, None] + 0 * o[None, None, :]).reshape(n, -1)
    cc = (c0[:, None, None] + 0 * o[None, :, None] + o[None, None, :]).reshape(n, -1)
    inb = (rr >= 0) & (rr < H_) & (cc >= 0) & (cc < W_)
    pflat = jnp.where(inb, rr * W_ + cc, 0)

    def spot(dv, du, vu_, vv_, cuv_):
        det = jnp.maximum(vu_ * vv_ - cuv_**2, 1e-6)
        m2 = (vv_ * du**2 - 2.0 * cuv_ * du * dv + vu_ * dv**2) / det
        if trunk is None:
            return jnp.exp(-0.5 * m2) / (2.0 * jnp.pi * jnp.sqrt(det))
        u, f, norm = trunk
        g = jnp.interp(jnp.sqrt(m2), u, f, left=f[0], right=0.0)
        return g / (2.0 * jnp.pi * jnp.sqrt(det) * norm)

    P = spot(rr - pr[:, None], cc - pc[:, None], vu[:, None], vv[:, None], cuv[:, None])
    P = (P * inb * valid_flat[pflat]).astype(f32)
    active = real & (P.sum(1) >= MIN_SPOT_MASS) & (T.sum(1) >= MIN_TIME_MASS)
    act = active.astype(f32)

    tbk = b0[:, None] + jnp.arange(nb)[None, :]
    tbc = jnp.clip(tbk, 0, nt - 1)
    vox = pflat[:, :, None] * nt + tbc[:, None, :]
    a = P[:, :, None] * T[:, None, :]
    N = jnp.zeros(size, f32).at[events].add(1.0, mode="drop")
    valid_vox = jnp.repeat(valid_flat, nt)

    # Background: a pixel map times one time spectrum, from the voxels no
    # predicted reflection touches.
    occupied = (a > 0) & real[:, None, None]
    cover = jnp.zeros(size, f32).at[vox].add(occupied.astype(f32)) > 0
    free = ((~cover) & (valid_vox > 0)).reshape(H_ * W_, nt).astype(f32)
    Nf = N.reshape(H_ * W_, nt) * free
    spectrum = Nf.sum(0) / jnp.maximum(free.sum(0), 1.0)
    spectrum = jnp.convolve(spectrum, jnp.ones(5, f32) / 5.0, mode="same")
    weight = (free * spectrum[None, :]).sum(1).reshape(H_, W_)
    bmap = _blur(Nf.sum(1).reshape(H_, W_), BG_BLUR) / jnp.maximum(
        _blur(weight, BG_BLUR), 1e-9
    )
    mu_bg = jnp.maximum((bmap.reshape(-1, 1) * spectrum[None, :]).ravel(), 1e-4)

    nbr_c = jnp.clip(nbr, 0, n - 1)
    nbr_ok = nbr < n

    def expected(amp):
        model = jnp.zeros(size, f32).at[vox].add((amp * act)[:, None, None] * a)
        return jnp.maximum(mu_bg + model, 0.3 * mu_bg)

    def information(mu):
        w2 = (valid_vox / mu).reshape(H_ * W_, nt)

        def row(k):
            wk = w2[pflat[k]][:, tbc[k]]  # (pixels, bins) of k's window
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
            tj = tj * ((rel >= 0) & (rel < nb))
            x = jnp.einsum("jp,pt->jt", pj * in_j * P[k][None, :], wk)
            return jnp.sum(x * tj * T[k][None, :], axis=1) * nbr_ok[k] * act[k] * act[j]

        rows = lax.map(row, jnp.arange(n), batch_size=PAIR_BATCH)
        Hm = (
            jnp.zeros((n, n), f32)
            .at[jnp.arange(n)[:, None], nbr]
            .add(rows, mode="drop")
        )
        Hm = 0.5 * (Hm + Hm.T)
        return Hm + jnp.diag(1.0 - act)

    def score(mu):
        ratio = (N - mu) / mu * valid_vox
        return jnp.sum(a * ratio[vox], axis=(1, 2)) * act

    def solver(Hm):
        d = jnp.sqrt(jnp.maximum(jnp.diag(Hm), 1e-30))
        R = Hm / jnp.outer(d, d)
        if truncate:
            lam, V = jnp.linalg.eigh(R)
            kept = lam > eig_floor
            inv = jnp.where(kept, 1.0 / jnp.where(kept, lam, 1.0), 0.0)

            def solve(rhs):
                return V @ ((V.T @ rhs) * inv)

            infl = jnp.sum(V**2 * inv[None, :], axis=1)
            lev = jnp.sum(jnp.where(kept[None, :], 0.0, V**2), axis=1)
            return d, solve, infl, lev, jnp.sum(~kept)
        L = jnp.linalg.cholesky(R)

        def solve(rhs):
            return cho_solve((L, True), rhs)

        return d, solve, None, None, 0

    def step(_, amp):
        mu = expected(amp)
        d, solve, *_ = solver(information(mu))
        return amp + solve(score(mu) / d) / d

    amp = lax.fori_loop(0, N_SCORING, step, jnp.zeros(n, f32))
    mu = expected(amp)
    d, solve, infl, lev, dropped = solver(information(mu))
    if not truncate:
        infl = jnp.diag(solve(jnp.eye(n, dtype=f32)))
        lev = jnp.zeros(n, f32)
    sigma = jnp.sqrt(jnp.maximum(infl, 0.0)) / d * act
    centre = jnp.clip(r0, 0, H_ - 1) * W_ + jnp.clip(c0, 0, W_ - 1)
    bg_centre = jnp.sum(
        mu_bg.reshape(H_ * W_, nt)[centre[:, None], tbc] * (T > 0), axis=1
    )
    return amp * act, sigma, infl, lev, dropped, active, bg_centre


def _fit_fn(nt, half, truncate, has_trunk):
    key = (nt, half, truncate, has_trunk)
    if key not in _JIT:
        from jax import jit

        _JIT[key] = jit(partial(_fit_image_impl, nt=nt, half=half, truncate=truncate))
    return _JIT[key]


def _neighbour_list(r0, c0, b0, nb, half, n_pad):
    """Padded (n_pad, K) list of reflections whose windows overlap, self first."""
    from scipy.spatial import cKDTree

    n = len(r0)
    tree = cKDTree(np.stack([r0, c0], 1).astype(float))
    pairs = tree.query_pairs(2 * half + 0.5, p=np.inf, output_type="ndarray")
    pairs = pairs[np.abs(b0[pairs[:, 0]] - b0[pairs[:, 1]]) < nb]
    src = np.concatenate([np.arange(n), pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([np.arange(n), pairs[:, 1], pairs[:, 0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    count = np.bincount(src, minlength=n)
    k_pad = max(16, int(2 ** np.ceil(np.log2(count.max()))))
    nbr = np.full((n_pad, k_pad), n_pad, dtype=np.int32)
    slot = np.arange(len(src)) - np.repeat(np.cumsum(count) - count, count)
    nbr[src, slot] = dst
    return nbr


def _pad(x, n_pad, fill=0):
    out = np.full((n_pad, *x.shape[1:]), fill, dtype=x.dtype)
    out[: len(x)] = x
    return out


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
    size = H * W * nt
    sig_major = np.sqrt(np.maximum(var_us, var_vs))
    half = int(
        np.clip(np.ceil(SPOT_REACH * np.percentile(sig_major, 99)), MIN_HALF, MAX_HALF)
    )

    trunk = None
    if profile not in (None, "gaussian", "auto"):
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

    # Time bins of every reflection and the share of its profile in each,
    # cut at the band edges.
    lt = np.log(tof_pred)
    b0_all = np.floor((lt + prof_l[0] - l0) / dl).astype(np.int64)
    out = np.zeros((n_peaks, 5), dtype=np.float64)
    out[:, 1], out[:, 2] = rs, cs
    out[:, 3] = np.sqrt(np.sqrt(np.maximum(var_us * var_vs - cov_uvs**2, 1e-6)))
    inflation = np.zeros(n_peaks)
    bg_out = np.zeros(n_peaks)
    dropped_total, redone = 0, 0

    order = np.argsort(frames, kind="stable")
    bounds = np.searchsorted(frames[order], np.arange(B + 1))
    pbar = tqdm(
        total=n_peaks, desc="Joint time-resolved fit", disable=not show_progress
    )
    for f in range(B):
        ix = order[bounds[f] : bounds[f + 1]]
        n = len(ix)
        if n == 0:
            continue
        tof, off = reader(f)
        tof = np.asarray(tof, dtype=np.float64)
        tb = np.floor((np.log(np.maximum(tof, 1e-6)) - l0) / dl).astype(np.int64)
        pix = np.repeat(np.arange(H * W, dtype=np.int64), np.diff(off))
        ev = np.where((tb >= 0) & (tb < nt), pix * nt + tb, size).astype(np.int32)
        ev_pad = np.full(
            max(EVENT_QUANTUM, -(-len(ev) // EVENT_QUANTUM) * EVENT_QUANTUM),
            size,
            dtype=np.int32,
        )
        ev_pad[: len(ev)] = ev

        b0 = b0_all[ix]
        be = b0[:, None] + np.arange(nb + 1)[None, :]
        t_edges = l0 + be * dl
        cdf = np.interp(t_edges - lt[ix, None], prof_l, prof_cdf, left=0.0, right=1.0)
        lo_l, hi_l = np.log(tof_lim_lo[ix])[:, None], np.log(tof_lim_hi[ix])[:, None]
        # share of each bin inside the band, by linear interpolation of its edges
        inside = np.clip(
            (np.minimum(t_edges[:, 1:], hi_l) - np.maximum(t_edges[:, :-1], lo_l)) / dl,
            0.0,
            1.0,
        )
        T = np.diff(cdf, axis=1) * inside * ((be[:, :-1] >= 0) & (be[:, :-1] < nt))

        n_pad = -(-n // PEAK_QUANTUM) * PEAK_QUANTUM
        r0 = np.rint(rs[ix]).astype(np.int32)
        c0 = np.rint(cs[ix]).astype(np.int32)
        nbr = _neighbour_list(r0, c0, b0, nb, half, n_pad)
        real = np.zeros(n_pad, dtype=bool)
        real[:n] = True
        args = (
            ev_pad,
            static_valid[f],
            _pad(r0, n_pad),
            _pad(c0, n_pad),
            _pad(rs[ix].astype(np.float32), n_pad),
            _pad(cs[ix].astype(np.float32), n_pad),
            _pad(var_us[ix], n_pad, 1.0),
            _pad(var_vs[ix], n_pad, 1.0),
            _pad(cov_uvs[ix], n_pad),
            _pad(T.astype(np.float32), n_pad),
            _pad(b0.astype(np.int32), n_pad),
            real,
            nbr,
        )
        dev = jax.device_put(args)
        res = _fit_fn(nt, half, False, trunk is not None)(
            *dev, trunk, np.float32(eig_floor)
        )
        amp, sig, infl, lev, dropped, active, bg_c = (np.asarray(r) for r in res)
        fitted = active[:n]
        bad = not np.all(np.isfinite(amp[:n][fitted])) or not np.all(
            np.isfinite(sig[:n][fitted])
        )
        over = (
            eig_floor > 0
            and fitted.any()
            and infl[:n][fitted].max() > max_variance_inflation
        )
        if bad or over:
            # Some direction is not determined: leave it out instead of
            # letting it set the amplitudes.
            floor = eig_floor if eig_floor > 0 else 1e-6
            res = _fit_fn(nt, half, True, trunk is not None)(
                *dev, trunk, np.float32(floor)
            )
            amp, sig, infl, lev, dropped, active, bg_c = (np.asarray(r) for r in res)
            fitted = active[:n]
            dropped_total += int(dropped)
            redone += 1
        keep = (
            fitted & (lev[:n] < LEVERAGE_LIMIT) & np.isfinite(amp[:n]) & (sig[:n] > 0)
        )
        flux = amp[:n] if allow_negative else np.maximum(amp[:n], 0.0)
        out[ix, 0] = np.where(keep, flux, 0.0)
        out[ix, 4] = np.where(keep, sig[:n], 0.0)
        inflation[ix] = np.where(fitted, infl[:n], 0.0)
        bg_out[ix] = bg_c[:n]
        pbar.update(n)
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
