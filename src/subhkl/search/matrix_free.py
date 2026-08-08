import warnings
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.signal
import jax.scipy.sparse.linalg
import numpy as np
from jax import jit, lax, vmap

# --- Carpet-fragmentation control --------------------------------------------
#
# A truth width sigma* strictly between two bank widths can be rendered two
# ways: as a point mixture of the two bracketing atoms (concentrated, but
# inexact -- the shape error is 4th order in the gap), or as a spatial carpet
# of the narrower atom (exact for any sigma* >= sigma_k, by the Gaussian
# semigroup G_sigma* = G_sigma_k * G_sqrt(sigma*^2-sigma_k^2), but taxed by
# the sigma**gamma penalty, 1st order in the gap).  The solver takes whichever
# is cheaper in the objective, and when the bank is too sparse in sigma that
# is the carpet: one physical peak comes back as a cluster of narrow atoms.
# No gamma < 1 can prevent it, because the tax is levied per unit mass while
# the fidelity gap grows with the 4th power of the gap -- the penalty only
# wins below a critical bank density.  The helpers below price both options
# so the bank can be sized to that density (num_sigmas=None) or an explicit
# bank flagged when it is predicted to fragment.
#
# Fidelity is the exact Poisson-core-weighted projection residual of the
# bracketing pair (no small-gap expansion: at the narrow end of a uniform
# bank the relative gap is O(1) and a Taylor form is badly wrong).  The tax
# is the flux-matched carpet surcharge priced at the calibrated per-height
# threshold lam_k ~ a_corr(z * w_k) sqrt(pi sigma_k^2 / bg), with z solved
# from the finder's own false-alarm calibration sum_k N_k Q(z w_k) = m0 --
# the same equation effective_alpha solves -- and the same Cornish-Fisher
# skew correction at low counts.  Because z multiplies the raw weights
# w_k = (sigma_k/ref_sigma)**gamma, any rescaling of the weights (any choice
# of ref_sigma) is absorbed into z exactly, so the criterion inherits the
# calibration's ref_sigma invariance by construction.
#
# What each input does, measured on [1, 25] at gamma = 0 (the steepness of
# the mechanism bounds every influence: the fidelity/tax ratio grows with
# the ~3.5th power of the gap, so the bank size responds to a multiplicative
# error in any input only as its ~0.28th power):
#
#   peak amplitude    the real scene driver: 40/160/640 photons -> 9/11/14
#                     channels.  Brighter peaks put more photons behind the
#                     same shape mismatch while the tax stays fixed.
#   background        nearly cancels at fixed amplitude: 0.4..40 photons/px
#                     -> 10..11 channels.  A darker background raises the
#                     tax (lam ~ 1/sqrt(bg)) but raises the contrast, and
#                     with it the misfit weight, by the same order.  (A
#                     fixed-contrast sweep suggests a strong bg dependence,
#                     but fixing contrast silently scales the peak with the
#                     background and answers a different question.)
#   frame size, m0    enter only through z, logarithmically: +-1 channel
#                     across frames 64..4096 and m0 0.1..10.
#   residual x/÷ 2    -> 9..14 channels.
#
# The construction-time frame side is nominal (z is re-evaluated against the
# real frame and the measured background in find_peaks_batch, which warns if
# the built bank falls short on the actual data).

_FRAG_FID_RESIDUAL = 20.0  # realised / idealised carpet advantage; factory
# default of the fid_residual constructor argument, and the one empirical
# constant left.  Measured 9.3-9.9 on the calibration case and kept at ~2x
# that, so the bank errs dense: the steepness of the mechanism prices the
# safety factor at only a few channels (e.g. [1,25] at gamma 0: 11 -> 14).
# calibrate_fragmentation_residual fits it per instrument
# against a requested unsupported-atom rate, the way m0 fixes the threshold: the measured factor (bank [1,25]x5, truth
# sigma*=15 at height 120 over background 7.6: 886/245 nats measured against
# the model's idealised ratio) by which reality favours the carpet beyond
# the projection model.  It absorbs the carpet's pointwise-shrinkage evasion
# of the flux-matched tax and its absorption of the background-estimate bias
# under wide peaks, neither of which the idealised model prices.
_FRAG_PEAK_AMP = 160.0  # default expected_peak_amplitude [photons]
_FRAG_BG = 10.0  # default expected_background [photons/Pixel]
_FRAG_NOMINAL_SIDE = 512  # frame side assumed before any data exists [Pixel];
# auto-sized banks re-derive the grid from the real frame and the measured
# background on every find_peaks_batch call and rebuild when it differs
_NUM_SIGMAS_SOFT_CAP = 16  # solve cost is linear in the channel count


def _frag_calibrated_z(sigmas, gamma, ref_sigma, m0, height, width):
    """Solve the false-alarm calibration sum_k N_k Q(z w_k) = m0 for z.

    Numpy twin of the bisection in effective_alpha, for use at bank-sizing
    time (before any jax tracing is warranted)."""
    from scipy.special import erfc

    sig = np.asarray(sigmas, dtype=float)
    w = (sig / ref_sigma) ** gamma
    n_k = np.maximum(height * width / (2.0 * np.pi * np.maximum(sig**2, 1e-6)), 2.0)
    lo, hi = 0.5, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(n_k * 0.5 * erfc(mid * w / np.sqrt(2.0)))) > m0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _moment_census(images, bg_map, bg_hi, counting=False):
    """Moment amplitudes ``A = F / (2 pi sigma^2)`` of every window that
    carries enough flux to measure, as an array (possibly empty).

    ``counting=True`` keeps only the window that *owns* its flux centroid
    (|centroid| <= stride/2 in both axes) -- a Voronoi assignment of peaks
    to window centres, so each bright peak is counted exactly once however
    it sits relative to the grid.  That is the population estimate for the
    fragmentation-rate mapping.  The default scan keeps every measurable
    window and is the *amplitude* configuration used for quantiles.
    """
    excess = np.asarray(images, dtype=np.float64) - np.asarray(bg_map, dtype=np.float64)
    if excess.ndim == 2:
        excess = excess[None, ...]
    amps = []
    for frame in excess:
        H, W = frame.shape
        step = max(8, int(H // 64))
        for r0 in range(step, H - step, step):
            for c0 in range(step, W - step, step):
                win = frame[r0 - step : r0 + step + 1, c0 - step : c0 + step + 1]
                flux = float(win.sum())
                # The floor must clear the Poisson noise of the window sum,
                # not just an absolute count: on a peak-free frame the excess
                # sum fluctuates with sd sqrt(area * bg), and a fixed floor of
                # 50 sits within reach of its tail (measured: a flat
                # Poisson(0.5) frame produced a census entry of 0.41 photons
                # that resized the bank to protect nothing).  Eight sigma
                # keeps every real peak -- their fluxes are thousands -- and
                # nothing else.
                area = float(win.size)
                if flux < max(
                    20.0 * bg_hi, 50.0, 8.0 * np.sqrt(area * max(bg_hi, 1e-3))
                ):
                    continue
                w = np.maximum(win, 0.0)
                tot = w.sum()
                if tot <= 0:
                    continue
                rr, cc = np.mgrid[-step : step + 1, -step : step + 1]
                dr = (w * rr).sum() / tot
                dc = (w * cc).sum() / tot
                # Counting mode: neighbouring windows see the same peak (the
                # skirt clears the flux floor several windows out), but only
                # the window nearest the peak owns the centroid.  Half the
                # stride in each axis tiles the plane exactly once, so the
                # ownership test counts each peak once wherever it sits.
                if counting and max(abs(dr), abs(dc)) > 0.5 * step:
                    continue
                m2 = (w * ((rr - dr) ** 2 + (cc - dc) ** 2)).sum() / tot
                if not np.isfinite(m2) or m2 <= 0:
                    continue
                amps.append(flux / (2.0 * np.pi * (0.5 * m2)))  # 2D: m2 = 2 sigma^2
    return np.asarray(amps, dtype=np.float64)


def _measure_radial_profile(
    images, bg_map, bg_hi, max_sigma, min_windows=8, u_max=4.0, du=0.1
):
    """The peak family's radial profile, measured from the frames themselves.

    The low-rank result that motivates this: stacking bright isolated peaks
    (recentred on moment centroids, rescaled by moment widths) leaves the mean
    profile carrying ~95% of the family's energy, so one measured trunk is the
    family up to anisotropy.  Measuring it is the same box arithmetic as the
    amplitude census -- no solve, no fit, no functional form:

      per qualifying window: float centroid, m2 -> sigma_w; bin the
      background-subtracted counts by u = r/sigma_w; normalise to unit flux;
      average windows weighted by flux.

    Binning around the *float* centroid handles sub-pixel centring without
    interpolation.  Windows are the Voronoi (counting) census, so each peak
    contributes once; the flux floor already clears 8 sigma of window-sum
    noise, so a peak-free frame contributes nothing rather than a noise
    profile.  Returns ``(u, f)`` with ``f[0] = 1``, or ``None`` when fewer
    than ``min_windows`` qualify -- the caller then stays on the Gaussian,
    which is the safe default rather than a profile made of too few peaks.
    """
    excess = np.asarray(images, dtype=np.float64) - np.asarray(bg_map, dtype=np.float64)
    if excess.ndim == 2:
        excess = excess[None, ...]
    edges = np.arange(0.0, u_max + du, du)
    centres = 0.5 * (edges[1:] + edges[:-1])
    acc = np.zeros(centres.size)
    wgt = np.zeros(centres.size)
    n_used = 0
    for frame in excess:
        H, W = frame.shape
        # The window must hold a max_sigma peak out to ~2.5 sigma, or its
        # second moment is so truncated that every measured width -- and with
        # it the u = r/sigma axis -- is wrong.  Measured cost of the census
        # default (17 px on a 512 frame, sigma up to 6.5): a pure-Gaussian
        # synthetic read 43% too broad at u = 1.5.
        step = max(8, int(np.ceil(2.5 * float(max_sigma))))
        yy, xx = np.mgrid[-step : step + 1, -step : step + 1].astype(float)
        for r0 in range(step, H - step, step):
            for c0 in range(step, W - step, step):
                win = frame[r0 - step : r0 + step + 1, c0 - step : c0 + step + 1]
                flux = float(win.sum())
                area = win.size
                if flux < max(
                    20.0 * bg_hi, 50.0, 8.0 * np.sqrt(max(bg_hi, 0.05) * area)
                ):
                    continue
                # Centroid from clipped weights (stability); everything after
                # from *unclipped* values inside a *circular* mask.  Clipping
                # rectifies background noise into a pedestal that inflates m2,
                # and a square window's corners do the same relative to the
                # circular truncation model inverted below -- together they
                # read a pure-Gaussian synthetic 20-40% wrong in width.
                w = np.maximum(win, 0.0)
                tot = w.sum()
                if tot <= 0:
                    continue
                dr = (w * yy).sum() / tot
                dc = (w * xx).sum() / tot
                # Voronoi ownership: only the window that owns its centroid.
                if abs(dr) > step / 2 or abs(dc) > step / 2:
                    continue
                d2 = (yy - dr) ** 2 + (xx - dc) ** 2
                disk = d2 <= step**2
                flux_d = float(win[disk].sum())
                if flux_d <= 0:
                    continue
                m2 = float((win[disk] * d2[disk]).sum()) / flux_d
                # Invert the truncation: inside radius R a Gaussian returns
                # m2 = sigma^2 * 2(1 - (1+x)e^-x)/(1 - e^-x), x = R^2/2sigma^2.
                # A damped fixed point converges in a few steps and needs no
                # bracketing solver.
                sigma_w = np.sqrt(max(m2, 1e-9) / 2.0)
                for _ in range(8):
                    x = step**2 / (2.0 * sigma_w**2)
                    ex = np.exp(-min(x, 60.0))
                    ratio = 2.0 * (1.0 - (1.0 + x) * ex) / max(1.0 - ex, 1e-9)
                    new_sigma = np.sqrt(max(m2, 1e-9) / max(ratio, 1e-9))
                    if abs(new_sigma - sigma_w) < 0.005:
                        sigma_w = new_sigma
                        break
                    sigma_w = 0.5 * (sigma_w + new_sigma)
                # A width at the window scale was never bracketed.
                if sigma_w < 0.5 or sigma_w > step / 2.5:
                    continue
                u = np.sqrt(d2[disk]).ravel() / sigma_w
                v = win[disk].ravel() / flux_d
                which = np.clip((u / du).astype(int), 0, centres.size - 1)
                inside = u < u_max
                # Flux-weighted mean of the unit-flux pixel values per u bin:
                # acc/wgt is then the average profile, bright peaks weighted by
                # their better statistics.
                np.add.at(acc, which[inside], flux * v[inside])
                counts = np.bincount(which[inside], minlength=centres.size).astype(
                    float
                )
                wgt += flux * counts
                n_used += 1
    if n_used < min_windows:
        return None
    ok = wgt > 0
    f = np.zeros(centres.size)
    f[ok] = acc[ok] / wgt[ok]
    # Per-bin averages of per-pixel values: convert back to a profile by
    # normalising the centre to 1; the kernel builder only needs the shape.
    if f[0] <= 0:
        return None
    f = np.maximum(f / f[0], 0.0)
    return centres[ok], f[ok]


def _moment_peak_amplitude(images, bg_map, bg_hi, quantile=90.0):
    """Bright-peak amplitude from window moments, not from a top-quantile count.

    ``A = F / (2 pi sigma^2)``, with the flux and the second moment taken over
    the same window of the background-subtracted frame.  Both are aggregates
    over a footprint, so unlike ``percentile(images, 99.99)`` a handful of hot
    pixels cannot pass for a bright peak population -- which is what kept the
    measured amplitude out of the bank resize.

    A high quantile rather than the median: the fidelity gap that drives
    fragmentation grows with brightness, so the bank is sized to protect the
    brightest peaks.  Precision is not the point -- bank size responds to this
    input as roughly its 0.28th power, so a 50% error is under one channel, and
    p75 through p99 give the same bank on real CG4D data.

    Returns ``None`` when nothing carries enough flux to measure, leaving the
    caller on its declared default.
    """
    amps = _moment_census(images, bg_map, bg_hi)
    if not amps.size:
        return None
    return float(np.percentile(amps, quantile))


def _frag_protected_quantile(max_fragmentation_rate, n_bright_per_image):
    """The brightness quantile the bank must protect to meet the rate.

    The scaling argument that replaces the solve ladder: given a bank, the
    criterion already says which peaks fragment -- those brighter than the
    protected amplitude -- and a marginally fragmented peak is a pair, both
    atoms of which fail the leave-one-out support test.  So

        E[unsupported/image] ~ 2 * (bright peaks above the quantile)/image,

    and inverting for the quantile is arithmetic on the moment census (box
    sums, no solve):  q = 1 - rate / (2 * N_bright).  Floored at the median
    -- allowing more fragmentation than the census can express just means
    every measured peak is fair game -- and capped at protecting the
    brightest censused peak outright.
    """
    allowed_peaks = max_fragmentation_rate / 2.0
    q = 1.0 - allowed_peaks / max(n_bright_per_image, 1e-9)
    return 100.0 * float(np.clip(q, 0.5, 1.0))


def _frag_pair_ratio(sa, sb, gamma, ref_sigma, z, bg, amp, fid_res=None):
    """Worst fidelity/tax ratio over off-grid widths in the gap (sa, sb).

    A value above 1 predicts that a peak of amplitude ``amp`` and some width
    inside the gap is cheaper to represent as a carpet of sigma-sa atoms
    than as a point mixture of the two bracketing atoms, i.e. that the fit
    will fragment it.  ``z`` is the calibrated threshold for the bank under
    construction (see _frag_calibrated_z)."""
    contrast = amp / bg
    va, vb = sa * sa, sb * sb
    r = np.linspace(0.0, 6.0 * sb, 1500)
    dA = 2.0 * np.pi * r
    fa = np.exp(-r * r / (2.0 * va))
    fb = np.exp(-r * r / (2.0 * vb))
    # Per-height threshold exactly as the solver builds it: calibrated
    # z * w_k, Cornish-Fisher corrected with the pure-background skewness
    # gamma1 = (2/3) / sqrt(pi sigma^2 bg), times sqrt(H_diag).
    lam = []
    for s, v in ((sa, va), (sb, vb)):
        a = z * (s / ref_sigma) ** gamma
        g1 = np.clip((2.0 / 3.0) / np.sqrt(np.pi * v * bg), 0.0, 2.0)
        lam.append((a + g1 * (a * a - 1.0) / 6.0) * np.sqrt(np.pi * v / bg))
    lam_a, lam_b = lam
    worst = 0.0
    for t in (0.2, 0.35, 0.5, 0.65, 0.8):
        vs = (1.0 - t) * va + t * vb
        fs = np.exp(-r * r / (2.0 * vs))
        wt = dA / (fs + 1.0 / contrast)
        G = np.array(
            [
                [np.trapezoid(fa * fa * wt, r), np.trapezoid(fa * fb * wt, r)],
                [np.trapezoid(fa * fb * wt, r), np.trapezoid(fb * fb * wt, r)],
            ]
        )
        h = np.array([np.trapezoid(fa * fs * wt, r), np.trapezoid(fb * fs * wt, r)])
        w = np.maximum(np.linalg.solve(G, h), 0.0)
        fid = 0.5 * (np.trapezoid(fs * fs * wt, r) - 2.0 * w @ h + w @ G @ w)
        tax = max(lam_a * vs / va - (w[0] * lam_a + w[1] * lam_b), 0.0)
        if fid_res is None:
            fid_res = _FRAG_FID_RESIDUAL
        worst = max(worst, fid_res * fid / max(tax, 1e-12))
    return worst


def _frag_bank_ratio(
    sigmas, gamma, ref_sigma, m0, bg, amp, height, width, fid_res=None
):
    """Worst pair ratio over a built bank, with z calibrated for that bank.

    Returns (ratio, sigma near the worst gap)."""
    # A shape-expanded bank repeats each scale once per variant; the pair
    # criterion is about scale gaps, and a zero-width gap makes its 2x2 Gram
    # singular.
    sigmas = np.unique(np.round(np.asarray(sigmas, dtype=float), 6))
    z = _frag_calibrated_z(sigmas, gamma, ref_sigma, m0, height, width)
    worst, where = 0.0, float(sigmas[0])
    for sa, sb in zip(sigmas[:-1], sigmas[1:]):
        ratio = _frag_pair_ratio(
            float(sa), float(sb), gamma, ref_sigma, z, bg, amp, fid_res
        )
        if ratio > worst:
            worst, where = ratio, float(np.sqrt(0.5 * (sa * sa + sb * sb)))
    return worst, where


def _required_uniform_num_sigmas(
    min_sigma, max_sigma, gamma, ref_sigma, m0, bg, amp, fid_res=None, nmax=64
):
    """Smallest uniform-grid num_sigmas with no fragmenting gap, or None."""
    side = _FRAG_NOMINAL_SIDE
    for n in range(2, nmax + 1):
        sigmas = np.linspace(min_sigma, max_sigma, n)
        z = _frag_calibrated_z(sigmas, gamma, ref_sigma, m0, side, side)
        if all(
            _frag_pair_ratio(float(sa), float(sb), gamma, ref_sigma, z, bg, amp) <= 1.0
            for sa, sb in zip(sigmas[:-1], sigmas[1:])
        ):
            return n
    return None


def _auto_sigma_grid(
    min_sigma,
    max_sigma,
    gamma,
    ref_sigma,
    m0,
    bg,
    amp,
    height=_FRAG_NOMINAL_SIDE,
    width=_FRAG_NOMINAL_SIDE,
    fid_res=None,
):
    """Smallest bank with no fragmenting gap: greedy widest-safe-step placement.

    The safe gap is roughly a constant *ratio* (~1.3-1.6x, tightening as
    sigma**(1-gamma) toward the wide end), so the minimal grid is close to
    geometric.  A uniform grid pays that ratio at the narrowest pair and
    needs several times more channels for the same guarantee (roughly 3x,
    e.g. ~14 adaptive over [1, 25] at gamma = 0).

    The calibrated z depends on the bank being built, so the build runs as a
    short fixed-point iteration: place the grid under the current z, re-solve
    z for that grid, repeat until the grid reproduces itself.  z varies only
    logarithmically with the channel count, so this settles in 2-3 passes.
    """
    z, prev = 4.0, None
    for _ in range(4):
        grid = [float(min_sigma)]
        while grid[-1] < max_sigma * (1.0 - 1e-9) and len(grid) < 64:
            sa = grid[-1]
            if (
                _frag_pair_ratio(sa, max_sigma, gamma, ref_sigma, z, bg, amp, fid_res)
                <= 1.0
            ):
                grid.append(float(max_sigma))
                break
            lo, hi = sa, float(max_sigma)
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                if (
                    _frag_pair_ratio(sa, mid, gamma, ref_sigma, z, bg, amp, fid_res)
                    <= 1.0
                ):
                    lo = mid
                else:
                    hi = mid
            # A vanishing safe step means the criterion cannot be met locally
            # (pathological parameters); take a small fixed ratio, don't spin.
            grid.append(max(lo, sa * 1.05))
        if grid[-1] < max_sigma:
            grid.append(float(max_sigma))
        if prev is not None and len(grid) == len(prev) and np.allclose(grid, prev):
            break
        prev = grid
        z = _frag_calibrated_z(grid, gamma, ref_sigma, m0, height, width)

    # Greedy-from-below leaves its remnant at the wide end -- when the last
    # bisection lands just short of max_sigma the forced endpoint creates a
    # near-duplicate pair (e.g. ..., 4.94, 5), a wasted channel next to an
    # unevenly stretched gap.  Relax the interior points to equalise the two
    # adjacent pair margins (left ratio rises and right ratio falls in s_i,
    # so the balance point is unique); this spreads the slack over the whole
    # grid without changing the channel count.  Keep the greedy grid if the
    # relaxed one ever violates the criterion.
    if len(grid) > 2:
        relaxed = list(grid)
        for _ in range(6):
            for i in range(1, len(relaxed) - 1):
                lo, hi = relaxed[i - 1] * 1.001, relaxed[i + 1] * 0.999
                for _ in range(15):
                    mid = np.sqrt(lo * hi)
                    left = _frag_pair_ratio(
                        relaxed[i - 1], mid, gamma, ref_sigma, z, bg, amp, fid_res
                    )
                    right = _frag_pair_ratio(
                        mid, relaxed[i + 1], gamma, ref_sigma, z, bg, amp, fid_res
                    )
                    if left > right:
                        hi = mid
                    else:
                        lo = mid
                relaxed[i] = float(np.sqrt(lo * hi))
        ok = all(
            _frag_pair_ratio(sa, sb, gamma, ref_sigma, z, bg, amp, fid_res) <= 1.0
            for sa, sb in zip(relaxed[:-1], relaxed[1:])
        )
        if ok:
            grid = relaxed
    return grid


class MatrixFreeSparseRBFPeakFinder:
    """
    Matrix-Free Global L1 Peak Finder.
    Replaces greedy Matching Pursuit with Global Convolutional Basis Pursuit.

    A note on ``gamma``, because the default sits on a degenerate value.

    The penalty on an atom of scale sigma works out proportional to
    sigma**(gamma + 1), while that atom's flux is proportional to sigma**2, so
    the penalty *per unit flux* goes as sigma**(gamma - 1).  At ``gamma == 1``
    that is constant -- measured constant to 1% across a 10x range of sigma --
    which is the Radon-measure / total-variation point: the penalty charges an
    atom for its flux and is blind to the scale carrying it.

    That is a genuine degeneracy rather than a mere preference, because the
    Gaussian scale space is closed under mass-preserving superposition
    (G_sigma = G_sigma' * G_sigma'' with sigma^2 = sigma'^2 + sigma''^2).  One
    broad atom and a spread of narrower atoms of equal total flux therefore
    predict the *same* image at the *same* penalty, so the minimiser is not
    unique in the scale coordinate, and since extra atoms always absorb a
    little more noise, the fit breaks the tie towards splitting.  In practice
    this shows up as one broad peak being reported as a cluster of narrower
    ones, plus spurious atoms between genuinely overlapping peaks.

    ``gamma < 1`` breaks the symmetry and makes a single broad atom strictly
    cheaper than any spread of the same flux, which is what a deconvolution
    ought to prefer.  Lowering ``gamma`` strengthens that preference
    continuously -- the penalty per unit flux goes as sigma**(gamma - 1), so
    the smaller ``gamma`` is, the more the fit prefers to explain structure
    with one wide atom rather than several narrow ones.  ``gamma = 0`` (a
    flat, scale-independent weight) is a point on that continuum, not a
    boundary, and negative values continue it: they bias towards broad,
    smooth solutions, which is what you want when the features of interest
    are diffuse rather than compact.

    Measured on the overlap regression cases, at ``min_sigma = 2``,
    ``max_sigma = 8``, over a flat background of 10 counts/pixel:

        gamma       -1.0   -0.5    0.0    0.5    1.0
        two blended peaks resolved
                     yes    yes    yes    yes    yes (as 32 atoms)
        weak peak in a strong tail found
                     yes    yes    yes    yes    yes (as 20 atoms)
        one broad + one compact feature: atoms reported
                       2      2      2      4     21
        median recovered sigma
                    4.35   4.35   4.35   3.36   2.00

    An earlier version of this docstring claimed ``gamma = 0`` over-merges
    and swallows genuine neighbours, and that the usable range is therefore
    the open interval.  That does not reproduce: at and below zero both
    overlap cases resolve cleanly into exactly two atoms, and the recovered
    widths simply track the broad-atom preference.  What the sweep does show
    is the fragmentation at ``gamma = 1`` the paragraphs above predict.

    The default is 0, and unlike its predecessor (0.5, inherited from the
    test-suite operating point) it is derived rather than tuned.  Under the
    calibrated threshold the minimum detectable *flux* at scale sigma works
    out to

        F_min(sigma)  ~  z * sigma**(gamma + 1) * sqrt(U),

    (coefficient threshold z * sigma**gamma * sd_c with sd_c ~ sqrt(U)/sigma,
    times the atom's flux-per-coefficient 2 pi sigma**2), while pure photon
    statistics -- the matched-filter sensitivity for a flux F at scale sigma
    on background U -- gives F_min ~ z * sigma * sqrt(U).  The two agree
    exactly at ``gamma = 0`` and only there: any positive gamma demands more
    flux than counting statistics requires at broad scales (missing real
    broad reflections), any negative gamma does the same at fine scales.  So
    gamma = 0 is the detection-neutral, likelihood-ratio operating point --
    and it still breaks the splitting degeneracy, because the penalty per
    unit flux ~ sigma**(gamma-1) = 1/sigma remains strictly decreasing;
    only gamma = 1 is degenerate.  The ten-dataset benchmark survey found
    gamma = 0 optimal across instruments, which is this argument measured.

    gamma is therefore a physics switch, not a tuning knob: 0 for point-like
    reflections, negative to impose a smoothness prior when the features of
    interest are genuinely diffuse, and never near 1.  Note the false-alarm
    calibration holds E[FP] = m0 at every gamma, so moving gamma reshapes
    *which scales* the budget is spent on without changing the budget --
    the two axes are orthogonal by construction.

    One caveat bounds all of the above: gamma's broad-atom preference is only
    enforceable when the competing broad atom exists in the bank.  For a truth
    width strictly between two bank widths the atomic option is missing; the
    fit then chooses between an inexact point mixture of the bracketing atoms
    (shape error 4th order in the gap) and an exact semigroup carpet of the
    narrower one (penalty surcharge 1st order in the gap), and once the bank
    is too sparse in sigma the carpet wins at any gamma < 1 -- one bright peak
    is reported as a cluster of narrow atoms (measured: an 886-nat fidelity
    gap against a 245-nat collected tax on a [1, 25] x 5 bank).  The default
    ``num_sigmas=None`` therefore sizes the bank automatically to the
    smallest, roughly geometric grid whose every gap keeps the tax ahead of
    the shape error, and an explicit ``num_sigmas`` below that density
    triggers a fragmentation warning at construction.  See the
    carpet-fragmentation helpers at module scope for the criterion and its
    calibration.
    """

    def __init__(
        self,
        alpha: float | None = None,
        gamma: float = 0.0,
        min_sigma: float = 1.0,
        max_sigma: float = 5.0,
        num_sigmas: int | None = None,
        loss: str = "poisson",
        show_steps: bool = False,
        ref_sigma: float = 1.0,
        chunk_size: int = 64,
        refine_positions: bool = True,
        reject_boundary_sigma: bool = False,
        boundary_sigma_frac: float = 0.98,
        false_alarms_per_image: float = 1.0,
        expected_background: float = _FRAG_BG,
        expected_peak_amplitude: float | None = None,
        fid_residual: float | None = None,
        max_fragmentation_rate: float = 1.0,
        profile_file: str | None = "auto",
        shape_ratio: float = 1.2,
        shape_orientations: int = 4,
        **kwargs,
    ):
        if max_sigma < min_sigma:
            raise ValueError(
                f"max_sigma ({max_sigma}) is below min_sigma ({min_sigma}); "
                "the basis bank would be empty."
            )
        if num_sigmas is not None and num_sigmas < 1:
            raise ValueError(f"num_sigmas must be at least 1, got {num_sigmas}")

        # A zero-width range is a single scale, whatever num_sigmas asks for.
        # linspace(s, s, k) returns k identical widths, and the solve then
        # carries k copies of one channel: k times the cost, and gamma --
        # which only ever compares scales against each other -- becomes
        # completely inert, silently.  Outputs at gamma 0.3, 0.5 and 0.75 came
        # back bit-identical, which is how this was noticed.  Collapse it here
        # rather than failing, because a single-width bank is a legitimate
        # request (the finder is then a matched filter at one scale) and only
        # the duplication is wrong.
        if max_sigma == min_sigma:
            if num_sigmas is not None and num_sigmas > 1 and show_steps:
                print(
                    f"  > min_sigma == max_sigma == {min_sigma:g}: collapsing the "
                    f"bank from {num_sigmas} identical widths to 1 "
                    "(gamma has no effect on a single-scale bank)."
                )
            num_sigmas = 1

        # Bank sizing against carpet fragmentation (see the helpers at module
        # scope).  num_sigmas=None sizes the bank to the smallest grid whose
        # every sigma gap keeps the penalty tax ahead of the mixture's shape
        # error; an explicit num_sigmas keeps the historical uniform grid and
        # is checked against the same criterion.
        # None means "measure it from the first batch".  The bank must exist
        # before any data does, so construction uses the nominal and the first
        # batch re-derives -- the contract `expected_background` already has.
        amp_is_auto = expected_peak_amplitude is None
        amp_for_build = (
            _FRAG_PEAK_AMP if amp_is_auto else float(expected_peak_amplitude)
        )
        # The criterion's one empirical constant, as an instance parameter:
        # the factory default is _FRAG_FID_RESIDUAL, and
        # calibrate_fragmentation_residual fits it per instrument against a
        # requested fragmentation rate, the way m0 fixes the threshold.
        fid_residual = (
            _FRAG_FID_RESIDUAL if fid_residual is None else float(fid_residual)
        )

        sigma_grid = None
        if num_sigmas is None:
            sigma_grid = _auto_sigma_grid(
                min_sigma,
                max_sigma,
                gamma,
                ref_sigma,
                false_alarms_per_image,
                expected_background,
                amp_for_build,
                fid_res=fid_residual,
            )
            num_sigmas = len(sigma_grid)
            print(
                f"  > num_sigmas auto-tuned to {num_sigmas} for sigma in "
                f"[{min_sigma:g}, {max_sigma:g}] at gamma = {gamma:g} "
                "(fragmentation control; pass num_sigmas to override): "
                + ", ".join(f"{s:.3g}" for s in sigma_grid)
            )
            if num_sigmas > _NUM_SIGMAS_SOFT_CAP:
                warnings.warn(
                    f"auto-tuned num_sigmas={num_sigmas} exceeds "
                    f"{_NUM_SIGMAS_SOFT_CAP}; solve cost and memory scale "
                    "linearly with the channel count.  The narrow end of the "
                    "range dominates the requirement, so raising min_sigma is "
                    "the strongest lever; alternatively narrow the sigma range "
                    "or pass an explicit (smaller) num_sigmas and accept that "
                    "off-grid peak widths may be reported fragmented.",
                    stacklevel=2,
                )
        elif num_sigmas >= 2 and max_sigma > min_sigma:
            ratio, near = _frag_bank_ratio(
                np.linspace(min_sigma, max_sigma, num_sigmas),
                gamma,
                ref_sigma,
                false_alarms_per_image,
                expected_background,
                amp_for_build,
                _FRAG_NOMINAL_SIDE,
                _FRAG_NOMINAL_SIDE,
                fid_res=fid_residual,
            )
            if ratio > 1.0:
                n_req = _required_uniform_num_sigmas(
                    min_sigma,
                    max_sigma,
                    gamma,
                    ref_sigma,
                    false_alarms_per_image,
                    expected_background,
                    amp_for_build,
                    fid_res=fid_residual,
                )
                n_auto = len(
                    _auto_sigma_grid(
                        min_sigma,
                        max_sigma,
                        gamma,
                        ref_sigma,
                        false_alarms_per_image,
                        expected_background,
                        amp_for_build,
                        fid_res=fid_residual,
                    )
                )
                warnings.warn(
                    f"num_sigmas={num_sigmas} is below the fragmentation "
                    f"threshold for sigma in [{min_sigma:g}, {max_sigma:g}] at "
                    f"gamma = {gamma:g}: bright peaks of off-grid width (worst "
                    f"near sigma ~ {near:.1f}, fidelity/tax ratio {ratio:.1f}) "
                    "are predicted to be reported as clusters of narrower "
                    "atoms rather than one atom.  A uniform grid needs "
                    f"num_sigmas >= {n_req if n_req else '> 64'}; "
                    f"num_sigmas=None auto-tunes an adaptive {n_auto}-channel "
                    "bank instead.",
                    stacklevel=2,
                )

        self.alpha = alpha
        self.gamma = gamma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigmas = num_sigmas
        self.loss = loss
        self.show_steps = show_steps
        self.ref_sigma = ref_sigma
        self.expected_background = float(expected_background)
        self.expected_peak_amplitude = amp_for_build
        self._amp_is_auto = amp_is_auto
        self.fid_residual = fid_residual
        # Tolerable unsupported atoms per image; drives which quantile of the
        # measured brightness census the auto bank protects (see
        # _frag_protected_quantile).  Non-positive keeps the fixed p90.
        self.max_fragmentation_rate = float(max_fragmentation_rate)
        # "auto" (default): measure the radial profile from the first batch and
        # rebuild the bank -- the same first-batch contract as the background
        # and amplitude.  A path uses that measured profile file; "gaussian"
        # or None keeps the analytic Gaussian.
        self.profile_file = profile_file
        self._measured_trunk = None
        self.shape_ratio = float(shape_ratio)
        self.shape_orientations = int(shape_orientations)
        self.chunk_size = chunk_size
        self.refine_positions = refine_positions
        self.reject_boundary_sigma = reject_boundary_sigma
        self.boundary_sigma_frac = boundary_sigma_frac
        if false_alarms_per_image <= 0:
            raise ValueError(
                f"false_alarms_per_image must be positive, got {false_alarms_per_image}"
            )
        self.false_alarms_per_image = float(false_alarms_per_image)

        # 1. Pre-build the Filter Bank
        self._auto_bank = sigma_grid is not None
        self.max_k_rad = int(3.0 * max_sigma)
        if sigma_grid is not None:
            self._set_bank(sigma_grid)
        else:
            self._set_bank(np.linspace(min_sigma, max_sigma, num_sigmas))

    def _set_bank(self, sigma_grid):
        """(Re)build every tensor derived from the sigma grid.

        Auto-sized banks call this again from find_peaks_batch when the real
        frame and measured background imply a different grid than the
        construction-time assumptions did."""
        self.num_sigmas = len(sigma_grid)
        self.sigmas = jnp.asarray(np.asarray(sigma_grid), dtype=jnp.float32)
        # Per-channel scale.  ``sigmas`` is the public scale grid and never
        # expands; shape variants multiply *channels*, tracked here, so the
        # explicit-count contract (num_sigmas == len(sigmas)) and the
        # calibration equation keep their historical meaning.
        self._channel_sigmas = self.sigmas
        # Use strictly unnormalized physical bases to preserve flux
        # relationships.  _build_kernel_bank also refreshes the separable
        # row/column factors; the identity-dispatch caches key on the new
        # arrays, so nothing stale survives the rebuild.
        self.K_weights, self.kernel_sq_norms = self._build_kernel_bank()
        self.K_sq = self.K_weights**2
        self.K_cu = self.K_weights**3
        # Both transform caches key on id(weights).  Emptying them is not
        # optional tidiness: the freed old bank's id can be recycled by a
        # later allocation -- the replacement bank included -- and a recycled
        # id would resurrect stale entries with the old channel count.
        # (Observed as a K x K' shape error in the FFT solve path on CPU,
        # where the allocator reuses the freed block readily.)
        self._pc_cache = {}
        self._fft_cache = {}
        # The jitted methods take `self` as a static argument, so their
        # compiled traces bake in the bank tensors read at trace time.  A
        # rebuilt bank with unchanged input shapes would otherwise hit those
        # stale traces silently -- _extract_peaks_all would read widths off
        # the old grid.  Resizing is rare, so drop every cached trace and
        # let the next call retrace against the new bank.
        for name in dir(type(self)):
            fn = getattr(type(self), name, None)
            if hasattr(fn, "clear_cache"):
                fn.clear_cache()

    def effective_alpha(self, height, width):
        """Per-scale significance threshold, in units of the dual's noise.

        The number returned here multiplies ``sd[p(omega)] = sqrt(H_diag)``
        (the standard deviation of the studentized dual variable under the
        background-only null) to form the L1 weight, so it is a z-score:
        an atom is admitted where the matched-filter correlation of the
        Poisson residual exceeds this many standard deviations of what noise
        alone produces.

        Admission is a binary classification run simultaneously over every
        resolution element of (position, scale) space: ``N_k = area /
        (2 pi sigma_k^2)`` elements at scale k, each with one-sided
        false-positive probability ``Q(z) = P(N(0,1) > z)``.  The threshold
        is fixed by the *calibration equation*

            E[FP](z) = sum_k N_k * Q(z * w_k) = m0,

        solved for ``z`` by bisection, where ``w_k = (sigma_k/ref_sigma) **
        gamma`` is the scale prior and ``m0 = false_alarms_per_image`` is the
        expected number of false atoms per image.  This replaces an earlier
        anchoring rule, ``max(floor_k / w_k) * w_k`` with per-scale floors
        ``sqrt(2 log N_k)``, which had two measured defects: it admitted the
        finest scale at its bare floor for every gamma > 0 while holding
        broad scales 2-3x above theirs (systematically favouring the split
        solutions gamma < 1 exists to suppress), and its realised E[FP]
        drifted with gamma (0.40 at gamma=0 down to 0.095 at gamma=1 on a
        256^2 frame), so gamma sweeps silently changed the detection budget.
        Under the calibration equation E[FP] = m0 identically for every
        gamma: the prior reshapes the threshold across scales but no longer
        moves the overall rate, and any rescaling of the weights -- including
        the choice of ``ref_sigma`` -- is absorbed exactly into z, so the
        unexposed reference constant provably cannot change the result.

        Multiplicity is sum-pooled over the bank as given.  By linearity of
        expectation this is exact for the per-slot count and a valid
        Bonferroni bound on distinct false atoms regardless of the strong
        correlation between adjacent scales (0.99 for neighbouring bank
        entries; one noise bump lighting several correlated scales still
        yields one admitted atom, so the realised count sits at or below
        m0).  The cost of that conservatism is logarithmically damped and
        includes a mild dependence on the bank sampling: densifying
        num_sigmas 5 -> 33 at fixed range raises z by ~0.4, because
        near-duplicate scales are counted as new tests.

        Passing an explicit ``alpha`` keeps its meaning as a z-score and is
        floored at the calibrated level: the final threshold is
        ``max(alpha, z*) * w_k``, so a user can demand more evidence than
        false-alarm control requires but not less.
        """
        # The calibration sums over *scales*, not channels.  Shape variants at
        # one (site, scale) are the same statistical test -- their kernels
        # correlate at ~0.99 -- so counting each channel would solve z against
        # a multiplicity ~n_shapes too high; measured cost when that happened:
        # the faint half of every detection.  Summing over the scale grid is
        # the exact deduplication, and on the Gaussian path it is identical to
        # the historical equation.
        w = (self.sigmas / self.ref_sigma) ** self.gamma
        n_k = jnp.maximum(
            (height * width) / (2.0 * jnp.pi * jnp.maximum(self.sigmas**2, 1e-6)),
            2.0,
        )
        m0 = self.false_alarms_per_image

        def expected_fp(z):
            q = 0.5 * jax.scipy.special.erfc(z * w / jnp.sqrt(2.0))
            return jnp.sum(n_k * q)

        # E[FP](z) is strictly decreasing, so bisection converges
        # unconditionally.  The bracket top is generous: even at w_min = 0.1
        # and N ~ 1e8, z* < 60 / w_min never binds.  60 halvings resolve z to
        # ~1e-16 of the bracket.
        lo = jnp.asarray(0.0, dtype=jnp.float32)
        hi = jnp.asarray(60.0, dtype=jnp.float32) / jnp.minimum(jnp.min(w), 1.0)

        def bisect(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            too_low = expected_fp(mid) > m0
            return jnp.where(too_low, mid, lo), jnp.where(too_low, hi, mid)

        lo, hi = lax.fori_loop(0, 60, bisect, (lo, hi))
        z_star = 0.5 * (lo + hi)

        # The returned vector is applied per *channel*.
        w_channel = (self._channel_sigmas / self.ref_sigma) ** self.gamma
        if self.alpha is None:
            return z_star * w_channel
        return jnp.maximum(self.alpha, z_star) * w_channel

    def _build_kernel_bank(self):
        """The atom family: Gaussian by default, measured-profile on request.

        ``profile_file`` points at a radial profile ``f(u)``, ``u = r/sigma``
        (JSON ``{"u": [...], "f": [...]}``), measured by stacking bright
        isolated peaks recentred on their moment centroids and rescaled by
        their moment widths.  On cg4d-garnet the family is essentially rank-1
        after scale: the mean profile carries 95.5% of the energy, and it is
        flat-topped -- 14-24% above a Gaussian at u = 1-2.

        ``shape_ratio > 1`` adds elliptical variants of the trunk at
        ``shape_orientations`` position angles, area-preserving (a*b =
        sigma^2), so every kernel keeps flux = C * sigma^2 with a single C and
        the width-mixture collapse in ``_read_chunk`` stays exactly valid.
        The residual after the mean profile *is* anisotropy (the leading
        scale-normalised PCA modes), and a radially symmetric atom -- Gaussian
        or measured -- cannot represent a ratio-1.2 peak with one atom at any
        bank density.

        Non-Gaussian or anisotropic kernels are not separable, so the solver
        uses its FFT path; the separable erf fast path stays for the default
        Gaussian bank.
        """
        mode = self.profile_file
        gaussian_trunk = mode in (None, "gaussian", "none") or (
            mode == "auto" and self._measured_trunk is None
        )
        if gaussian_trunk and self.shape_ratio == 1.0:
            return self._build_gaussian_bank()

        if mode == "auto" and self._measured_trunk is not None:
            _u, _f = self._measured_trunk

            def trunk(u):
                return np.interp(u, _u, _f, left=float(_f[0]), right=0.0)

        elif not gaussian_trunk:
            import json as _json

            with open(mode, encoding="utf-8") as fh:
                _prof = _json.load(fh)
            _u = np.asarray(_prof["u"], dtype=float)
            _f = np.asarray(_prof["f"], dtype=float)

            def trunk(u):
                return np.interp(u, _u, _f, left=float(_f[0]), right=0.0)

        else:

            def trunk(u):
                return np.exp(-0.5 * u * u)

        if self.shape_ratio > 1.0:
            angles = (
                np.pi * np.arange(self.shape_orientations) / self.shape_orientations
            )
            shapes = [(1.0, 0.0)] + [(self.shape_ratio, float(a)) for a in angles]
        else:
            shapes = [(1.0, 0.0)]

        base = np.asarray(self.sigmas, dtype=float)
        # Guard against double expansion on a rebuild that did not reset
        # ``sigmas``: an expanded bank repeats each scale len(shapes) times.
        n_sh = len(shapes)
        if n_sh > 1 and base.size >= n_sh and base.size % n_sh == 0:
            folded = base.reshape(-1, n_sh)
            if np.all(folded == folded[:, :1]):
                base = folded[:, 0]

        grid = np.arange(-self.max_k_rad, self.max_k_rad + 1, dtype=float)
        # Pixel integration by 3x3 supersampling: adequate for the smooth
        # measured profile at sigma >= 1 px, where the erf shortcut for the
        # Gaussian is not available.
        off = (np.arange(3) - 1.0) / 3.0
        kernels, sig_out = [], []
        for s in base:
            for ratio, theta in shapes:
                a = s * np.sqrt(ratio)
                b = s / np.sqrt(ratio)
                co, si = np.cos(theta), np.sin(theta)
                acc = np.zeros((grid.size, grid.size))
                for dy in off:
                    for dx in off:
                        yy, xx = np.meshgrid(grid + dy, grid + dx, indexing="ij")
                        uu = (yy * co + xx * si) / a
                        vv = (-yy * si + xx * co) / b
                        acc += trunk(np.hypot(uu, vv))
                kernels.append(acc / 9.0)
                sig_out.append(s)
        kernels_2d = jnp.asarray(np.stack(kernels), dtype=jnp.float32)
        sq_norms = jnp.sum(kernels_2d**2, axis=(1, 2))
        self._channel_sigmas = jnp.asarray(np.asarray(sig_out), dtype=jnp.float32)
        self.use_separable = False

        if self.show_steps:
            print(
                f"  > learned basis: {len(base)} scales x {n_sh} shape(s) = "
                f"{len(sig_out)} channels; threshold calibrated over the "
                "scale grid (shape variants are one statistical test)"
            )
        return kernels_2d[:, None, :, :], sq_norms

    def _build_gaussian_bank(self):
        k_grid = jnp.arange(-self.max_k_rad, self.max_k_rad + 1)
        yy, xx = jnp.meshgrid(k_grid, k_grid, indexing="ij")

        def build_one(s):
            sig_sq2 = s * jnp.sqrt(2.0) + 1e-6
            erf_y = jax.scipy.special.erf((yy + 0.5) / sig_sq2) - jax.scipy.special.erf(
                (yy - 0.5) / sig_sq2
            )
            erf_x = jax.scipy.special.erf((xx + 0.5) / sig_sq2) - jax.scipy.special.erf(
                (xx - 0.5) / sig_sq2
            )
            k_2d = (jnp.pi / 2.0) * (s**2) * erf_y * erf_x
            return k_2d

        kernels_2d = vmap(build_one)(self.sigmas)
        sq_norms = jnp.sum(kernels_2d**2, axis=(1, 2))

        # Separable factorization.  The pixel-integrated Gaussian is an exact
        # outer product, k_2d = (pi/2) s^2 * e1 (x) e1 with e1 the 1D erf
        # profile, and so is its square.  Two 1D depthwise passes therefore
        # apply the *identical* operator at (2r+1)+(2r+1) taps instead of
        # (2r+1)^2 -- a ~15x FLOP reduction at max_sigma = 5 -- differing
        # from the dense path only by floating-point reassociation.
        sig_sq2 = self.sigmas * jnp.sqrt(2.0) + 1e-6
        g = k_grid[None, :]
        e1 = jax.scipy.special.erf(
            (g + 0.5) / sig_sq2[:, None]
        ) - jax.scipy.special.erf((g - 0.5) / sig_sq2[:, None])  # [K, taps]
        amp = (jnp.pi / 2.0) * self.sigmas**2
        self._rows_w = (amp[:, None] * e1)[:, None, :, None]  # [K,1,taps,1]
        self._cols_w = e1[:, None, None, :]  # [K,1,1,taps]
        self._rows_sq = ((amp**2)[:, None] * e1**2)[:, None, :, None]
        self._cols_sq = (e1**2)[:, None, None, :]
        # Cubed bank, for the third cumulant of the dual variable (the
        # Cornish-Fisher skewness correction in the solve).  The cube of an
        # outer product is the outer product of cubes, so it stays separable.
        self._rows_cu = ((amp**3)[:, None] * e1**3)[:, None, :, None]
        self._cols_cu = (e1**3)[:, None, None, :]
        self.use_separable = True

        return kernels_2d[:, None, :, :], sq_norms

    @staticmethod
    def _sep_depthwise(x, rows, cols):
        """Two 1D depthwise passes; rows [K,1,t,1], cols [K,1,1,t]."""
        K = rows.shape[0]
        y = lax.conv_general_dilated(
            x,
            rows,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=K,
        )
        return lax.conv_general_dilated(
            y,
            cols,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=K,
        )

    def _sep_factors(self, weights):
        """Trace-time dispatch: identify which bank `weights` is."""
        if not getattr(self, "use_separable", False):
            return None
        if weights is self.K_weights:
            return self._rows_w, self._cols_w
        if weights is self.K_sq:
            return self._rows_sq, self._cols_sq
        if weights is self.K_cu:
            return self._rows_cu, self._cols_cu
        return None

    # The kernel bank is applied by FFT at every width; the separable direct
    # path below survives as the `use_fft = False` escape hatch and as the
    # reference the transform is validated against.  The transform computes the
    # same linear map to float32 round-off: zero-padded past the
    # linear-convolution length, so there is no wraparound being traded away,
    # it agrees with the direct path to ~1e-06 relative over the whole array,
    # borders included, with the adjoint identity holding to the same level.
    # It is also never slower -- measured on a 512x512 frame with K=5 it is
    # 2.1x faster at the narrowest bank tried (19 taps) and 4.6x at
    # max_sigma = 25 (151 taps), where it takes the solve from 5.9 s to 0.78 s
    # per frame, since a separable pass is O(taps) per pixel per channel while
    # the transform is O(log n) and independent of kernel width.
    #
    # An earlier revision gated this behind a minimum tap count so the default
    # max_sigma = 5 configuration kept its exact previous arithmetic.  The
    # gate cost a hard-tuned constant and left ~2x on the table at the
    # default; with the finder's test suite passing under the transform at
    # every width, it went.  The arithmetic difference is real but last-bit
    # (~1e-06), and at these settings the low-amplitude tail of the peak list
    # is chaotic well above that level from run-to-run GPU nondeterminism
    # alone (see MEDIAN_MAX_SAMPLES).

    @staticmethod
    def _fft_len(n):
        """Next power of two >= n.

        A smaller 7-smooth length (e.g. 840 rather than 1024 for a 662-pixel
        frame at max_sigma = 25) was tried and measured no faster at any
        configuration -- cuFFT's radix-2 kernels beat its mixed-radix ones by
        more than the extra padding costs, by up to 28% -- so the simplest
        length wins on both counts.
        """
        return 1 << (max(int(n), 1) - 1).bit_length()

    def _fft_kernels(self, weights, nf):
        """Transform of the kernel bank, centred at the origin.  Cached per shape.

        Held in the cache as a numpy array, not a device array.  Anything
        produced by a jnp call inside the solve is a tracer belonging to that
        trace, so caching one would leak it out of the enclosing ``fori_loop``
        and fail on the next call; the numpy value is converted at each use
        site instead and lands in the compiled program as a constant.
        """
        cache = self.__dict__.setdefault("_fft_cache", {})
        key = (id(weights), nf)
        if key not in cache:
            # Convert the whole bank before indexing: slicing a captured array
            # inside a trace is itself a JAX operation and would hand back a
            # tracer, concrete though the bank is.
            k2d = np.asarray(weights, dtype=np.float64)[:, 0, :, :]  # [K, taps, taps]
            r = self.max_k_rad
            ker = np.zeros((k2d.shape[0], nf, nf), dtype=np.float64)
            ker[:, : 2 * r + 1, : 2 * r + 1] = k2d
            ker = np.roll(ker, (-r, -r), axis=(1, 2))
            cache[key] = np.fft.rfft2(ker).astype(np.complex64)
        return cache[key]

    def _use_fft(self, weights):
        # The shape test keeps any weights that are not the full bank (nothing
        # passes one today) on the general direct path rather than through a
        # transform built for the bank's dimensions.
        return (
            getattr(self, "use_fft", True)
            and weights.shape[-1] == 2 * self.max_k_rad + 1
        )

    def _pc_kernels(self, weights, H, W):
        """Fourier symbols of the kernel bank on the solver's (H, W) grid.

        These feed the Sherman-Morrison preconditioner in the Newton CG
        solves: with a scalar weight the Gram operator A^T W A is circulant,
        and because the dictionary maps K coefficient planes into a single
        image its per-frequency K x K block is exactly the rank-one outer
        product of this vector with itself -- invertible in closed form once
        a diagonal ridge completes it.  Cached as numpy for the tracer-leak
        reason documented on _fft_kernels.  Keyed on the grid shape: the
        preconditioner must live on exactly the solver's torus, since
        transforming on a padded grid and cropping would break the symmetry
        CG requires of M.
        """
        cache = self.__dict__.setdefault("_pc_cache", {})
        key = (id(weights), H, W)
        if key not in cache:
            k2d = np.asarray(weights, dtype=np.float64)[:, 0, :, :]
            r = self.max_k_rad
            ker = np.zeros((k2d.shape[0], H, W), dtype=np.float64)
            ker[:, : 2 * r + 1, : 2 * r + 1] = k2d
            ker = np.roll(ker, (-r, -r), axis=(1, 2))
            Kh = np.fft.rfft2(ker)
            cache[key] = (
                Kh.astype(np.complex64),
                (np.abs(Kh) ** 2).astype(np.float32),
            )
        return cache[key]

    def _use_sm_precond(self):
        # Opt-in (finder.use_sm_precond = True), off by default: the solver-
        # level win is unambiguous (see _solve_ssn_cg_global), but on wide
        # peaks whose true sigma falls between bank scales the better-
        # converged solution represents the flux as the cluster of
        # neighbouring bank atoms that the discretized L1 optimum actually
        # is, where the stalled default solve reports one dominant atom that
        # sigma-refinement then fits -- cleaner output, worse optimality.
        # Measured on a 512^2 frame with peaks at sigma = 3/8/15 against a
        # 5-scale bank to max_sigma = 25: 6 -> 22 reported atoms, BIC worse
        # by ~1.5k despite ~200 nats better NLL.  On pure-noise controls the
        # two paths return identical peak lists, so the false-alarm
        # calibration is not affected.  Flip the default once extraction
        # merges same-reflection clusters (or the bank is dense enough in
        # sigma that single atoms are representable).
        return getattr(self, "use_sm_precond", False) and self._use_fft(self.K_weights)

    def _forward_op(self, c, weights):
        if self._use_fft(weights):
            H, W = c.shape[-2:]
            nf = self._fft_len(max(H, W) + 2 * self.max_k_rad)
            Kf = jnp.asarray(self._fft_kernels(weights, nf))
            Cf = jnp.fft.rfft2(c[0], s=(nf, nf))
            out = jnp.fft.irfft2(jnp.sum(Cf * Kf, axis=0), s=(nf, nf))
            return out[None, None, :H, :W]
        factors = self._sep_factors(weights)
        if factors is not None:
            filtered = self._sep_depthwise(c, *factors)
            return jnp.sum(filtered, axis=1, keepdims=True)
        weights_fwd = weights.transpose(1, 0, 2, 3)
        return lax.conv_general_dilated(
            c,
            weights_fwd,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

    def _adjoint_op(self, u, weights):
        if self._use_fft(weights):
            H, W = u.shape[-2:]
            nf = self._fft_len(max(H, W) + 2 * self.max_k_rad)
            Kf = jnp.asarray(self._fft_kernels(weights, nf))
            Uf = jnp.fft.rfft2(u[0, 0], s=(nf, nf))
            out = jnp.fft.irfft2(Uf[None] * jnp.conj(Kf), s=(nf, nf))
            return out[None, :, :H, :W]
        factors = self._sep_factors(weights)
        if factors is not None:
            K = weights.shape[0]
            tiled = jnp.broadcast_to(u, u.shape[:1] + (K,) + u.shape[2:])
            return self._sep_depthwise(tiled, *factors)
        return lax.conv_general_dilated(
            u,
            weights,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

    @partial(jit, static_argnames=["self", "max_iter"])
    def _solve_ssn_cg_global(self, y_img, bg_img, max_iter=100):
        H, W = y_img.shape
        y = y_img[None, None, :, :]
        bg = bg_img[None, None, :, :]

        # Channel count from the bank itself: with shape variants the bank
        # carries num_sigmas * n_shapes channels while ``num_sigmas`` keeps
        # its public meaning as the scale count.
        K = int(self.K_weights.shape[0])
        c_init = jnp.zeros((1, K, H, W))
        q_init = jnp.zeros((1, K, H, W))

        bg_med = jnp.maximum(jnp.median(bg), 1e-3).astype(jnp.float32)

        # 1. Exact Spatially Varying Variance Map & Lipschitz Bounds
        if self.loss == "gaussian":
            W_ref = jnp.ones_like(bg)
        else:
            W_ref = 1.0 / jnp.maximum(bg, 1e-3)

        H_diag = self._adjoint_op(W_ref, self.K_sq)
        H_diag_safe = jnp.maximum(H_diag, 1e-6)

        # Variance of each channel's coefficient estimate.  This has to be the
        # channel's own curvature: taking the across-channel maximum instead
        # understates the noise on every channel but the broadest -- by a factor
        # of ~11 for the narrowest basis in a typical bank -- so the fine scales
        # get thresholded far too weakly and fit noise into a smear of shifted
        # basis copies rather than one coefficient per peak.
        if self.loss == "gaussian":
            var_c = bg_med / H_diag_safe
        else:
            var_c = 1.0 / H_diag_safe

        # The multiplicity in the calibration is the number of tests actually
        # performed.  The solver sees the padded image, but extraction
        # discards maxima within pad + MARGIN of every edge (find_peaks_batch:
        # pad = max_k_rad, MARGIN = max(3, max_k_rad)), so the padded border
        # holds no admissible candidates and must not be counted -- at
        # max_sigma = 25 on a 256^2 frame it would otherwise inflate the test
        # count 2.5x and the threshold by ~0.2 sigma.
        border = self.max_k_rad + max(3, self.max_k_rad)
        H_int = max(H - 2 * border, 8)
        W_int = max(W - 2 * border, 8)
        alpha_vec = self.effective_alpha(H_int, W_int)

        # 2. L1 penalty weight, in units of the objective rather than of the
        # prox step.  Soft-thresholding at lam/H_diag recovers the intended
        # "alpha standard deviations of coefficient noise" cut, so lam must be
        # alpha * weight * H_diag * sqrt(var_c).  Deriving the penalty from the
        # step size instead makes the objective itself move whenever the step
        # does, which leaves the line search minimising a different problem on
        # every iteration.
        a_gauss = jnp.broadcast_to(alpha_vec[None, :, None, None], H_diag_safe.shape)
        if self.loss == "poisson":
            # Cornish-Fisher skewness correction.  The calibration converts a
            # false-alarm budget to a *Gaussian* quantile, but the dual is a
            # weighted sum of Poisson residuals whose third cumulant per pixel
            # is E[(y - U)^3]/U^3 = 1/U^2, so its standardized skewness is
            #     gamma1(omega) = (Phi^3 adj 1/U^2) / H_diag^{3/2},
            # exactly computable with one more separable convolution (the
            # cube of an outer product is an outer product).  Where the atom
            # footprint collects many photons gamma1 -> 0 and this is inert;
            # in the photon-starved regime it is decisive: at 0.5 counts/px
            # the finest scale sums ~3 photons, the upper tail is far fatter
            # than Gaussian, and the uncorrected calibrated threshold admits
            # 14 false atoms per frame against a budget of 1 (measured).  The
            # second-order Cornish-Fisher quantile transform
            #     z_corr = z + gamma1 (z^2 - 1) / 6
            # restores the budget; validity requires gamma1 * z / 2 < 1, and
            # the clip keeps the correction inside that regime rather than
            # extrapolating the expansion where it has no meaning.
            kappa3 = self._adjoint_op(W_ref * W_ref, self.K_cu)
            gamma1 = jnp.clip(kappa3 / H_diag_safe**1.5, 0.0, 2.0)
            a_corr = a_gauss + gamma1 * (a_gauss**2 - 1.0) / 6.0
        else:
            a_corr = a_gauss
        lam = a_corr * H_diag_safe * jnp.sqrt(var_c)

        # 3. Step size.  A prox-gradient step is only guaranteed to descend for
        # tau <= 1/lambda_max(A^T W A).  The diagonal of that operator is not a
        # bound on its largest eigenvalue: these basis functions overlap heavily,
        # and for a typical bank lambda_max exceeds max(diag) by ~400x, so a step
        # built from the diagonal overshoots by that factor and collapses the
        # line search onto its smallest permitted step every iteration.  A few
        # power iterations give the bound directly; the kernels and weights are
        # positive, so a constant vector is a good starting guess.
        def power_step(_, v):
            Av = self._adjoint_op(
                W_ref * self._forward_op(v, self.K_weights), self.K_weights
            )
            return Av / (jnp.linalg.norm(Av) + 1e-12)

        v0 = jnp.ones((1, K, H, W), dtype=jnp.float32)
        v_top = lax.fori_loop(0, 15, power_step, v0 / jnp.linalg.norm(v0))
        Av_top = self._adjoint_op(
            W_ref * self._forward_op(v_top, self.K_weights), self.K_weights
        )
        L_max = jnp.sum(v_top * Av_top) / jnp.maximum(jnp.sum(v_top * v_top), 1e-12)
        tau_local = 1.0 / (L_max + 1e-4)

        tau_alpha = tau_local * lam

        # Sherman-Morrison circulant preconditioner for the Newton CG solves.
        # Jacobi preconditioning collapses in the wide-kernel regime: the
        # overlap ratio lambda_max/diag grows as the footprint area 4*pi*
        # sigma^2 (theory notes, Prop. 2), and at max_sigma = 25 a Jacobi-
        # preconditioned CG leaves the Newton system at a relative residual
        # of 0.05-0.5 no matter the iteration budget (measured at maxiter up
        # to 150), so every Newton step is rejected and the outer loop
        # degenerates into prox-gradient steps of size 1/lambda_max -- it
        # stalls at a KKT residual ~1e4 above STOP_TOL.  The circulant
        # approximation A^T Wbar A + eps captures exactly the overlap
        # structure Jacobi ignores: per frequency it is a rank-one K x K
        # block (see _pc_kernels) plus a diagonal, inverted in closed form
        # below at the cost of one round-trip FFT per application.  Measured
        # at max_sigma = 25 on a 512^2 Poisson frame: Newton acceptance
        # 13% -> 57%, final KKT residual 12.0 -> 0.6, active set 4.3k -> 1.1k
        # atoms, per-iteration cost +35%.  The damped variants (adding a
        # fraction of the rank-one diagonal to eps) were swept and always
        # lost to the undamped form.  Off by default -- see _use_sm_precond
        # for why better convergence is not yet better output.
        use_sm = self._use_sm_precond()
        if use_sm:
            Kh_np, Kh2_np = self._pc_kernels(self.K_weights, H, W)
            Kh = jnp.asarray(Kh_np)
            Kh2 = jnp.asarray(Kh2_np)

        def get_loss_grad_hess(c_curr):
            u = self._forward_op(c_curr, self.K_weights) + bg
            if self.loss == "gaussian":
                res = u - y
                nll = 0.5 * jnp.sum(res**2)
                grad = self._adjoint_op(res, self.K_weights)
                W_diag = jnp.ones_like(u)
            else:
                u_safe = jnp.maximum(u, 1e-6)
                nll = jnp.sum(u_safe - y * jnp.log(u_safe))
                res_1d = 1.0 - (y / u_safe)
                grad = self._adjoint_op(res_1d, self.K_weights)
                W_diag = 1.0 / jnp.maximum(u_safe, 1e-3)
            return nll, grad, W_diag, u

        # Stop on the per-coordinate KKT residual max|G_i|/lam_i, i.e. when
        # the first-order residual is everywhere below STOP_TOL of the local
        # penalty scale -- the scale at which the estimator makes activation
        # decisions.  The previous test, ||q+ - q|| > 1e-3, was an absolute
        # threshold on the step norm: once the fallback/acceptance logic
        # takes forward-backward steps (step norm tau*||G||, with tau ~
        # 1/lambda_max ~ 1e-4), it fired at ||G|| ~ 1e-3/tau ~ 10 -- a
        # residual of the ORDER OF the penalty itself, i.e. it read "steps
        # got small" as "converged" three to four orders before the
        # activation decisions were settled.  Measured on a 4-peak synthetic
        # case the relative residual reaches 2e-5 (float32 floor ~1e-4 in
        # the prox-gradient map), so 1e-3 certifies with a ~50x margin.
        STOP_TOL = 1e-3

        def cond_fn(state):
            step, _, _, kkt_rel = state
            return (step < max_iter) & (kkt_rel > STOP_TOL)

        def body_fn(state):
            step, q, c, _ = state
            _, grad, W_diag, u_curr = get_loss_grad_hess(c)
            Gq = (q - c) / tau_local + grad

            # Residual of the CURRENT iterate.  ||G|| is not monotone under
            # the acceptance rule below (J is the merit function; an accepted
            # Newton step can transiently inflate the residual by orders),
            # so once the current iterate is certified the state is frozen
            # rather than stepped -- the loop then exits returning the
            # certified iterate, never a post-certification transient.
            kkt_rel = jnp.max(jnp.abs(Gq) / lam)
            converged = kkt_rel <= STOP_TOL

            # Strict Independent L1 Block Soft-Thresholding
            D_mat = (q > tau_alpha).astype(jnp.float32)

            # Semi-smooth Newton system for G(q) = (q - c(q))/tau + grad(c(q)).
            # Since c(q) = max(0, q - tau_alpha), the generalised Jacobian is the
            # Gauss-Newton Hessian A^T W A on the active set (D_mat == 1) and
            # 1/tau_local on the inactive set.  Both blocks are carried in a
            # single operator that is symmetric by construction: CG is only
            # valid for symmetric positive-definite operators, so the Jacobi
            # scaling is supplied through the preconditioner argument rather
            # than multiplied into the operator, which would break symmetry and
            # leave CG returning a direction that is not a descent direction.
            # H = A^T W A carries [counts]/[c]^2, so a bare additive constant is
            # not a regularisation strength but whatever fraction of diag(H) the
            # data makes it: 6.4e-4 of the diagonal at 500 counts and 8.2e-7 at
            # the 0.64-count mean of a real MANDI frame.  main folds the Jacobi
            # scaling into the operator (ssn.py, eta[:, None] * hess) so its
            # 1e-4 * I sits on a unit diagonal and is genuinely relative; that
            # is only possible there because it solves densely.  Keeping the
            # operator symmetric for CG moved the scaling out to M=, which left
            # this ridge absolute.  Scale it by diag(H) to restore the same
            # meaning, with main's value.
            RIDGE_REL = 1e-4

            H_diag_local = jnp.maximum(self._adjoint_op(W_diag, self.K_sq), 1e-6)

            def apply_jacobian(v):
                v_active = v * D_mat
                Av = self._forward_op(v_active, self.K_weights)
                At_W_Av = self._adjoint_op(W_diag * Av, self.K_weights)
                ridge = RIDGE_REL * H_diag_local * v_active
                return (At_W_Av + ridge) * D_mat + (1.0 - D_mat) * v

            if use_sm:
                # Closed-form inverse of E + Wbar u u^H per frequency
                # (Sherman-Morrison), with u the kernel-bank symbol vector
                # and E the per-channel mean of the true ridge.  The mask to
                # the active set keeps M symmetric positive definite; the
                # mismatch it introduces (the true Jacobian is masked, the
                # circulant is not) is localized to active-set boundaries,
                # which is what CG's remaining iterations are for.
                Wbar = jnp.mean(W_diag)
                eps_ch = RIDGE_REL * jnp.mean(H_diag_local, axis=(0, 2, 3))
                Ei = 1.0 / eps_ch[:, None, None]
                den = 1.0 + Wbar * jnp.sum(Kh2 * Ei, axis=0)

                def precond(v):
                    vh = jnp.fft.rfft2((v * D_mat)[0])
                    s = jnp.sum(Kh * vh * Ei, axis=0)
                    out = vh * Ei - jnp.conj(Kh) * Ei * (Wbar * s / den)
                    w = jnp.fft.irfft2(out, s=(H, W))[None]
                    return w * D_mat + (1.0 - D_mat) * v

            else:
                eta = 1.0 / H_diag_local

                def precond(v):
                    return eta * v * D_mat + (1.0 - D_mat) * v

            # Active rows solve A^T W A dq = -G; inactive rows reduce to the
            # explicit prox-gradient step dq = -tau_local * G.
            rhs = -Gq * D_mat - tau_local * Gq * (1.0 - D_mat)
            dq, _ = jax.scipy.sparse.linalg.cg(
                apply_jacobian, rhs, M=precond, tol=1e-3, maxiter=20
            )
            dq = jnp.where(jnp.isfinite(dq), dq, 0.0)

            # Change in objective from c to c_test, accumulated as a single
            # reduction over per-pixel *differences*.
            #
            # Forming it as j(c_test) - j(c) instead cannot work in float32.
            # Both are sums over the whole image of magnitude |J|, so their
            # difference is unresolvable below one ulp, ~6e-8 * |J| -- on a
            # 126x126 frame that floor is ~9e-3, and on a 542x542 detector
            # frame it is ~0.2.  The true decrease falls under it long before
            # the iterate is converged, at which point no amount of
            # backtracking can show an improvement: every halving looks like an
            # increase, the budget runs out, the step is rejected and the outer
            # loop stops.  Measured on a 6-frame synthetic case, all 12 solves
            # ended that way -- none reached the convergence test or the
            # iteration cap -- and the stopping iteration then moved with
            # anything that perturbed the arithmetic.
            #
            # Differencing per pixel first ties the accuracy to the decrease
            # rather than to |J|.  log1p keeps the Poisson term exact when the
            # trial point is close, which is precisely the regime that matters.
            # This is the same quantity, not an approximation; it also drops the
            # adjoint convolution the old objective computed and threw away on
            # every backtracking trial.
            def obj_delta(c_test):
                u_test = self._forward_op(c_test, self.K_weights) + bg
                if self.loss == "gaussian":
                    d_nll = 0.5 * jnp.sum(
                        (u_test - u_curr) * ((u_test - y) + (u_curr - y))
                    )
                else:
                    u_c = jnp.maximum(u_curr, 1e-6)
                    du = jnp.maximum(u_test, 1e-6) - u_c
                    d_nll = jnp.sum(du - y * jnp.log1p(du / u_c))
                return d_nll + jnp.sum(lam * (c_test - c))

            def bt_cond(bt_state):
                bt_i, _step_size, _, _, d_test = bt_state
                is_valid = jnp.isfinite(d_test)
                return (bt_i < 12) & ((d_test > 0.0) | ~is_valid)

            def bt_body(bt_state):
                bt_i, step_size, _, _, _ = bt_state
                step_size = jnp.float32(step_size * 0.5)

                q_test = q + step_size * dq
                c_test = jnp.maximum(0.0, q_test - tau_alpha)

                return (bt_i + 1, step_size, q_test, c_test, obj_delta(c_test))

            q_test_init = q + dq
            c_test_init = jnp.maximum(0.0, q_test_init - tau_alpha)

            bt_init = (
                0,
                jnp.float32(1.0),
                q_test_init,
                c_test_init,
                obj_delta(c_test_init),
            )
            _, _, q_try, c_try, d_try = lax.while_loop(bt_cond, bt_body, bt_init)

            # Sufficient-decrease acceptance against the forward-backward
            # step q - tau_local * G(q).  The identity
            # q - tau G(q) = c(q) - tau grad(c(q)) makes that step the
            # prox-gradient iteration, and tau <= 1/L holds by construction
            # (power iteration on A^T W(0) A with W(u) <= W(0) elementwise),
            # so the descent lemma guarantees its decrease -- it is the
            # certified step this iteration must beat.  Accepting the Newton
            # step whenever it merely does not increase the objective, as
            # this loop previously did, discards a better step the residual
            # already contains: measured on a 4-peak synthetic case, the FB
            # step decreased J more than the backtracked Newton step on ~75%
            # of iterations, and taking the better of the two drove the
            # optimality measure ||c - prox(c - tau grad)||/tau from 1.9e-1
            # to 3.7e-4 at gamma=0.5 within the same iteration budget.  Each
            # iteration now decreases J at least as much as the FB step, so
            # the solve is globally convergent at no less than the FB rate;
            # the cost is one extra fused-difference evaluation (~4%).
            q_fb = q - tau_local * Gq
            c_fb = jnp.maximum(0.0, q_fb - tau_alpha)
            d_fb = obj_delta(c_fb)
            accept = jnp.isfinite(d_try) & (d_try <= d_fb)
            q_final = jnp.where(accept, q_try, q_fb)
            c_final = jnp.where(accept, c_try, c_fb)
            q_final = jnp.where(converged, q, q_final)
            c_final = jnp.where(converged, c, c_final)

            return (step + 1, q_final, c_final, kkt_rel)

        init_state = (0, q_init, c_init, jnp.float32(1e9))
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        _, _, c_l1, _ = final_state

        # Augmented-penalty proximal Newton endgame.  The zero-count pixels'
        # fidelity terms are exactly linear in c (u >= bg > 0), so they fold
        # into the penalty: lam_aug = lam + A^T 1_{y=0}.  What remains is a
        # fidelity over counted pixels only, whose every term is standard
        # self-concordant (integer counts), with the exact Hessian weight
        # y/u^2 -- supported on counted pixels automatically.  Two damped
        # proximal Newton steps from the certified phase-1 endpoint measure
        # a ~6x tighter KKT residual and deactivate atoms the better-resolved
        # solution does not contain; each step is accepted only if it beats
        # the certified forward-backward step, so the guarantee of the main
        # loop is preserved (the FB step below is bit-identical to the
        # phase-1 fallback: grad f_+ + A^T 1_{y=0} = A^T(1 - y/u) exactly).
        # The subproblem keeps the constraint and the augmented penalty, so
        # its solution exists for singular Hessians with no ridge (feasible
        # recession directions have slope lam_aug > 0), and the null
        # directions of the exact weight are resolved by deactivation rather
        # than damped through a blind metric.
        P_mask = (y >= 1.0).astype(jnp.float32)
        a0 = self._adjoint_op(1.0 - P_mask, self.K_weights)
        lam_aug = lam + a0

        def apn_grad(c_curr):
            u = jnp.maximum(self._forward_op(c_curr, self.K_weights) + bg, 1e-6)
            gp = self._adjoint_op(P_mask * (1.0 - y / u), self.K_weights)
            Wt = P_mask * y / (u * u)
            return gp, Wt, u

        def apn_Hop(v, Wt):
            return self._adjoint_op(
                Wt * self._forward_op(v, self.K_weights), self.K_weights
            )

        def apn_obj_delta(c_curr, u_curr, c_test):
            u_t = self._forward_op(c_test, self.K_weights) + bg
            du = jnp.maximum(u_t, 1e-6) - u_curr
            return jnp.sum(P_mask * (du - y * jnp.log1p(du / u_curr))) + jnp.sum(
                lam_aug * (c_test - c_curr)
            )

        def apn_body(_, c_curr):
            gp, Wt, u_curr = apn_grad(c_curr)
            Dg = jnp.maximum(apn_Hop(jnp.ones_like(c_curr), Wt), 1e-8)
            Hj = jnp.maximum(self._adjoint_op(Wt, self.K_sq), 1e-8)
            gl = gp + lam_aug

            def q_of(x):
                return apn_Hop(x - c_curr, Wt) + gl

            def psi(x):
                dx = x - c_curr
                return 0.5 * jnp.sum(dx * apn_Hop(dx, Wt)) + jnp.sum(gl * dx)

            # GPCG-lite inner solve of the constrained quadratic subproblem:
            # a Gershgorin projected-gradient step identifies the active set
            # (D = diag(H 1) majorizes H since H is entrywise nonnegative),
            # then Jacobi-preconditioned CG runs on the free variables and
            # the projected result is kept only if the model decreases.
            x = c_curr
            for _round in range(2):
                x = jnp.maximum(0.0, x - q_of(x) / Dg)
                qx = q_of(x)
                Fm = ((x > 0.0) | (qx < 0.0)).astype(jnp.float32)

                def Aop(v, Fm=Fm, Wt=Wt):
                    return apn_Hop(v * Fm, Wt) * Fm + (1.0 - Fm) * v

                if use_sm:
                    # The endgame's subproblem must be preconditioned like
                    # the main loop's, and solved at least as well: handed
                    # the tighter phase-1 iterate the Jacobi/maxiter-8
                    # configuration returns garbage directions in the
                    # wide-kernel regime, and its free-set rule then
                    # RE-inflates the support 1.7k -> 15k atoms within two
                    # steps (measured at max_sigma = 25).  With this
                    # preconditioner and budget the decrement stays in the
                    # full-step regime (nu ~ 0.09) and the support shrinks.
                    Wt_bar = jnp.sum(Wt) / jnp.maximum(jnp.sum(P_mask), 1.0)
                    eps_apn = 1e-4 * jnp.mean(Hj, axis=(0, 2, 3))
                    Ei_a = 1.0 / eps_apn[:, None, None]
                    den_a = 1.0 + Wt_bar * jnp.sum(Kh2 * Ei_a, axis=0)

                    def Mop(v, Fm=Fm, Ei_a=Ei_a, den_a=den_a, Wt_bar=Wt_bar):
                        vh = jnp.fft.rfft2((v * Fm)[0])
                        s = jnp.sum(Kh * vh * Ei_a, axis=0)
                        out = vh * Ei_a - jnp.conj(Kh) * Ei_a * (Wt_bar * s / den_a)
                        w = jnp.fft.irfft2(out, s=(H, W))[None]
                        return w * Fm + (1.0 - Fm) * v

                    apn_cg_iters = 40
                else:

                    def Mop(v, Fm=Fm, Hj=Hj):
                        return (v / Hj) * Fm + (1.0 - Fm) * v

                    apn_cg_iters = 8

                dx, _ = jax.scipy.sparse.linalg.cg(
                    Aop, -qx * Fm, M=Mop, tol=1e-4, maxiter=apn_cg_iters
                )
                dx = jnp.where(jnp.isfinite(dx), dx, 0.0)
                x_cand = jnp.maximum(0.0, x + dx)
                x = jnp.where(psi(x_cand) < psi(x), x_cand, x)

            # Damped step from the decrement (full step in the quadratic
            # phase); c + alpha*d is a convex combination of feasible points.
            d = x - c_curr
            nu = jnp.sqrt(jnp.maximum(jnp.sum(d * apn_Hop(d, Wt)), 0.0))
            alpha_step = jnp.where(nu < 0.2, 1.0, 1.0 / (1.0 + nu))
            c_try = c_curr + alpha_step * d
            dJ_n = apn_obj_delta(c_curr, u_curr, c_try)

            g_full = gp + a0
            c_fb = jnp.maximum(0.0, c_curr - tau_local * g_full - tau_alpha)
            dJ_f = apn_obj_delta(c_curr, u_curr, c_fb)
            take = jnp.isfinite(dJ_n) & (dJ_n <= dJ_f)
            return jnp.where(take, c_try, c_fb)

        c_l1 = lax.fori_loop(0, 2, apn_body, c_l1)

        return c_l1[0]

    @partial(jit, static_argnames=["self", "border"])
    def _rank_support(self, c_tensor, border=0):
        """Rank the significance-gated support of one image, capacity-free.

        Everything here is independent of how many peaks there are -- the
        smoothing, the local-maximum test, and the full argsort all have
        shapes fixed by the image alone -- so this compiles exactly once per
        image shape no matter how large the support turns out to be.  Returns
        the descending coefficient order over the flattened image and the
        exact support count; the capacity-dependent work (measuring and
        refining candidates) happens downstream in fixed-size chunks.
        """
        c_tot = jnp.sum(c_tensor, axis=0)  # [H, W]

        # Smooth the discrete L1 coefficients to recover true continuous center
        # of mass.  L1 splinters a peak across adjacent pixels, so this only
        # needs to span that one-pixel scale: a wider kernel throws away exactly
        # the separation the fine basis channels were there to resolve, merging
        # neighbouring peaks a few pixels apart into one maximum.  Cap it at the
        # finest scale in the bank.
        smooth_sigma = min(1.0, float(self.min_sigma))
        sig_sq2 = smooth_sigma * jnp.sqrt(2.0) + 1e-6
        k_half = max(1, round(2.0 * smooth_sigma))
        k_grid = jnp.arange(-k_half, k_half + 1)
        k_1d = jax.scipy.special.erf((k_grid + 0.5) / sig_sq2) - jax.scipy.special.erf(
            (k_grid - 0.5) / sig_sq2
        )

        c_smooth_temp = jax.scipy.signal.correlate2d(c_tot, k_1d[:, None], mode="same")
        c_smooth = jax.scipy.signal.correlate2d(
            c_smooth_temp, k_1d[None, :], mode="same"
        )

        window = (3, 3)
        c_max = lax.reduce_window(
            c_smooth, -jnp.inf, jax.lax.max, window, (1, 1), "SAME"
        )
        # Support membership is exact, not approximate: the prox step returns
        # c = max(0, q - tau * lam), so every coefficient below the alpha
        # threshold is *identically* zero and everything above it is strictly
        # positive.  Smoothing (a nonnegative kernel) preserves that sign, so
        # `> 0` admits exactly the maxima whose support cleared alpha.  An
        # epsilon here would be a second, hidden significance level.
        is_max = (c_smooth == c_max) & (c_smooth > 0.0)

        # Discard maxima in the replicated border before ranking, not after.
        # The edge padding is a constant strip that the finest basis fits
        # readily, so it carries many strong spurious maxima; leaving them in
        # until after the top-capacity cut lets them consume the whole budget
        # and silently drop the real peaks from the interior.
        if border > 0:
            interior = jnp.zeros_like(is_max)
            interior = interior.at[border:-border, border:-border].set(True)
            is_max = is_max & interior

        c_flat = jnp.where(is_max.flatten(), c_smooth.flatten(), -1.0)

        # Full descending order plus the exact support count.  Every support
        # maximum is strictly positive and every non-maximum slot is -1, so
        # the first `count` entries of `order` are exactly the support --
        # membership in it is the alpha test, and no cap or epsilon takes part
        # in the decision.
        order = jnp.argsort(c_flat)[::-1]
        return order, jnp.sum(is_max)

    @partial(jit, static_argnames=["self"])
    def _measure_candidates(self, c_tensor, idx):
        """Read (amplitude, row, column, sigma) for one fixed-size chunk of
        ranked candidate indices.  Shapes depend only on the image and the
        chunk length, never on the support size, so this compiles once."""
        H, W = c_tensor.shape[1], c_tensor.shape[2]

        def process_peak(idx):
            r = idx // W
            c = idx % W

            r_safe = jnp.clip(r, 1, H - 2)
            c_safe = jnp.clip(c, 1, W - 2)

            # Extract 3x3 patch to integrate splintered coefficients
            c_patch = lax.dynamic_slice(
                c_tensor, (0, r_safe - 1, c_safe - 1), (c_tensor.shape[0], 3, 3)
            )
            c_channels = jnp.sum(c_patch, axis=(1, 2))

            # Exact Flux & Variance Preservation
            # Flux of basis k is A_k * sigma_k^2
            flux_k = c_channels * (self._channel_sigmas**2)
            total_flux_scaled = jnp.sum(flux_k) + 1e-9

            # Variance of mixture is sum(Flux_k * sigma_k^2) / sum(Flux_k).
            # Floor it at the finest basis: a slot that matched nothing has zero
            # flux in every channel, and dividing by a zero width below turns its
            # amplitude into an infinity.  Those slots are discarded by the
            # validity mask, but an infinity multiplied by a zero mask is a NaN,
            # which then contaminates anything that consumes the whole array.
            sigma_sq_eff = jnp.maximum(
                jnp.sum(flux_k * (self._channel_sigmas**2)) / total_flux_scaled,
                float(self.min_sigma) ** 2,
            )
            sigma_eff = jnp.sqrt(sigma_sq_eff)

            # Convert flux back to effective central amplitude
            amp_eff = total_flux_scaled / sigma_sq_eff

            # Localise from the raw coefficients, not from the smoothed map used
            # to find the maximum.  Those are two different jobs: smoothing helps
            # decide *whether* there is a peak here, but it also drags the apex
            # toward any neighbouring peak, so reading the position off it makes
            # the centre wander.  L1 splinters a peak across the pixels it
            # straddles, so the coefficient-weighted centroid is the sub-pixel
            # position, and it is exact for a peak sitting on a pixel.
            #
            # The window is kept at the one-pixel splintering scale on purpose.
            # Widening it to follow the fitted width was tried and is worse: the
            # broad channels carry diffuse coefficients, so a wider window pulls
            # the centre off the peak and reaches into close neighbours.
            patch_tot = jnp.sum(c_patch, axis=0)
            patch_mass = jnp.sum(patch_tot) + 1e-9
            offsets = jnp.array([-1.0, 0.0, 1.0])
            dr = jnp.sum(patch_tot * offsets[:, None]) / patch_mass
            dc = jnp.sum(patch_tot * offsets[None, :]) / patch_mass

            r_cont = r_safe + jnp.clip(dr, -1.0, 1.0)
            c_cont = c_safe + jnp.clip(dc, -1.0, 1.0)

            return jnp.array([amp_eff, r_cont, c_cont, sigma_eff])

        return vmap(process_peak)(idx)

    @partial(jit, static_argnames=["self"])
    def _render_atoms(self, y_img, peaks, active):
        """Render one chunk of atoms into an [H, W] image, boxes clipped at
        the borders exactly as ``_refine_peaks`` renders them."""
        H, W = y_img.shape
        P = 2 * self.max_k_rad + 1
        off = jnp.arange(P, dtype=jnp.float32)

        finite = jnp.all(jnp.isfinite(peaks), axis=1) & active
        amp = jnp.where(finite, peaks[:, 0], 0.0)
        r = jnp.where(finite, peaks[:, 1], float(self.max_k_rad))
        c = jnp.where(finite, peaks[:, 2], float(self.max_k_rad))
        sig = jnp.where(finite, peaks[:, 3], float(self.min_sigma))

        s2 = sig * jnp.sqrt(2.0) + 1e-6
        r0 = jnp.clip(jnp.round(r).astype(jnp.int32) - self.max_k_rad, 0, H - P)
        c0 = jnp.clip(jnp.round(c).astype(jnp.int32) - self.max_k_rad, 0, W - P)
        rr = r0[:, None].astype(jnp.float32) + off[None, :]
        cc = c0[:, None].astype(jnp.float32) + off[None, :]

        def erf_span(grid, centre):
            d = grid - centre[:, None]
            return jax.scipy.special.erf(
                (d + 0.5) / s2[:, None]
            ) - jax.scipy.special.erf((d - 0.5) / s2[:, None])

        ey = erf_span(rr, r)
        ex = erf_span(cc, c)
        amp_s = amp * (jnp.pi / 2.0) * (sig**2)
        patch = amp_s[:, None, None] * ey[:, :, None] * ex[:, None, :]

        idx_r = jnp.broadcast_to(
            (r0[:, None] + jnp.arange(P))[:, :, None], (peaks.shape[0], P, P)
        )
        idx_c = jnp.broadcast_to(
            (c0[:, None] + jnp.arange(P))[:, None, :], (peaks.shape[0], P, P)
        )
        return jnp.zeros((H, W)).at[idx_r, idx_c].add(patch)

    # Fixed chunk length for the capacity-dependent stages.  This is a tile
    # size, not a limit: the number of chunks grows with the support while
    # every jitted shape stays constant, so nothing recompiles however many
    # peaks an image carries, and per-chunk memory is bounded (the refinement
    # working set is O(chunk * patch^2) rather than O(support * patch^2)).
    EXTRACT_CHUNK = 256

    def _extract_peaks_all(self, c_tensor, y_img, bg_img, border=0):
        """Extract and refine the *entire* significance-gated support.

        The admission criterion is support membership alone: the prox step
        soft-thresholds at ``alpha`` standard deviations of coefficient noise,
        so a coefficient is nonzero exactly when it cleared the significance
        level, and every positive local maximum of that support is admitted.
        No count cap and no epsilon takes part in the decision.

        jit needs static shapes, so the work is transposed rather than sized:
        the capacity-independent stages (smoothing, maximum test, full sort)
        run once with image-fixed shapes in ``_rank_support``, and the
        capacity-dependent stages (candidate measurement, refinement) sweep
        the ranked support in fixed-length chunks.  Each jitted function
        therefore compiles exactly once per image shape; a larger support
        means more chunk iterations, never a new compilation.  The earlier
        capacity-doubling ladder recompiled the sort and the 200-step
        refinement at every rung -- the largest graphs in the finder, twice
        over.

        Refinement stays a minimisation of the one joint Poisson NLL: the
        atoms outside the chunk being refined are frozen and rendered into
        that chunk's background, which is exact block-coordinate descent on
        the joint objective (one sweep, strongest block first).  When the
        whole support fits a single chunk -- the common case -- this is
        bit-identical to refining everything jointly.

        Returns the valid peaks only, [n, 4], already mask-applied.
        """
        order, count = self._rank_support(c_tensor, border=border)
        n = int(count)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32)

        chunk = self.EXTRACT_CHUNK
        n_chunks = -(-n // chunk)
        # The order vector has H*W entries and a 3x3-window maximum occupies
        # at least a 2x2 cell, so n_chunks*chunk <= n + chunk - 1 << H*W and
        # this slice never truncates.
        order_np = np.asarray(order[: n_chunks * chunk])

        peaks_chunks = []
        masks = []
        for k in range(n_chunks):
            idx = jnp.asarray(order_np[k * chunk : (k + 1) * chunk])
            peaks_chunks.append(self._measure_candidates(c_tensor, idx))
            masks.append(np.arange(k * chunk, (k + 1) * chunk) < n)

        if self.refine_positions:
            renders = [
                self._render_atoms(y_img, p, jnp.asarray(m))
                for p, m in zip(peaks_chunks, masks)
            ]
            total = renders[0]
            for rimg in renders[1:]:
                total = total + rimg
            refined = []
            for p, m, rimg in zip(peaks_chunks, masks, renders):
                bg_eff = bg_img + (total - rimg)
                refined.append(self._refine_peaks(y_img, bg_eff, p, jnp.asarray(m)))
            peaks_chunks = refined

        peaks_all = np.concatenate([np.asarray(p) for p in peaks_chunks])
        mask_all = np.concatenate(masks)
        return peaks_all[mask_all].astype(np.float32)

    @partial(jit, static_argnames=["self", "n_steps"])
    def _refine_peaks(self, y_img, bg_img, peaks, active, n_steps=200):
        """Continuous ("sliding") refinement of the selected support.

        The convex solve picks *which* atoms are present, but it can only place
        them on the integer grid the dictionary is built on, so a peak that sits
        between pixels is split across neighbours and its position is only
        recoverable to whatever the centroid heuristic manages.  Re-fitting the
        selected peaks' continuous (amplitude, row, column, sigma) against the
        same Poisson objective lifts the answer off the grid: this is the
        sliding step of a Frank-Wolfe scheme over measures, and it is what makes
        the recovered positions sub-pixel rather than sub-grid.

        Peaks are rendered on their own bounding box and scattered into the
        model, so cost is O(n_peaks * patch^2) rather than O(n_peaks * H * W)
        and this stays affordable on full detector images.
        """
        H, W = y_img.shape
        P = 2 * self.max_k_rad + 1
        off = jnp.arange(P, dtype=jnp.float32)
        mask = active.astype(jnp.float32)

        # Replace unused slots outright instead of relying on the mask to cancel
        # them.  Masking by multiplication cannot neutralise a non-finite value
        # -- inf * 0 is NaN -- and one such row is enough to make the objective,
        # and therefore every gradient, NaN.  The guard against non-finite
        # gradients further down would then silently turn refinement into a
        # no-op rather than reporting anything.
        safe = jnp.stack(
            [
                jnp.ones_like(peaks[:, 0]),
                jnp.full_like(peaks[:, 1], float(self.max_k_rad)),
                jnp.full_like(peaks[:, 2], float(self.max_k_rad)),
                jnp.full_like(peaks[:, 3], float(self.min_sigma)),
            ],
            axis=1,
        )
        finite = jnp.all(jnp.isfinite(peaks), axis=1) & active
        peaks = jnp.where(finite[:, None], peaks, safe)
        mask = finite.astype(jnp.float32)

        # Unconstrained parameterisation keeps amplitude positive and sigma
        # inside the bank's range without needing a projection each step.
        lo, hi = float(self.min_sigma), float(self.max_sigma)
        sig0 = jnp.clip(peaks[:, 3], lo + 1e-3, hi - 1e-3)
        u_init = jnp.stack(
            [
                jnp.log(jnp.maximum(peaks[:, 0], 1e-3)),
                peaks[:, 1],
                peaks[:, 2],
                jnp.log((sig0 - lo) / jnp.maximum(hi - sig0, 1e-6)),
            ],
            axis=1,
        )

        def physical(u):
            amp = jnp.exp(u[:, 0])
            sig = lo + (hi - lo) * jax.nn.sigmoid(u[:, 3])
            return amp, u[:, 1], u[:, 2], sig

        def render(u):
            amp, r, c, sig = physical(u)
            s2 = sig * jnp.sqrt(2.0) + 1e-6
            r0 = jnp.clip(jnp.round(r).astype(jnp.int32) - self.max_k_rad, 0, H - P)
            c0 = jnp.clip(jnp.round(c).astype(jnp.int32) - self.max_k_rad, 0, W - P)
            rr = r0[:, None].astype(jnp.float32) + off[None, :]
            cc = c0[:, None].astype(jnp.float32) + off[None, :]

            def erf_span(grid, centre):
                d = grid - centre[:, None]
                return jax.scipy.special.erf(
                    (d + 0.5) / s2[:, None]
                ) - jax.scipy.special.erf((d - 0.5) / s2[:, None])

            ey = erf_span(rr, r)
            ex = erf_span(cc, c)
            amp_s = amp * (jnp.pi / 2.0) * (sig**2) * mask
            patch = amp_s[:, None, None] * ey[:, :, None] * ex[:, None, :]

            idx_r = jnp.broadcast_to(
                (r0[:, None] + jnp.arange(P))[:, :, None], (peaks.shape[0], P, P)
            )
            idx_c = jnp.broadcast_to(
                (c0[:, None] + jnp.arange(P))[:, None, :], (peaks.shape[0], P, P)
            )
            return jnp.zeros((H, W)).at[idx_r, idx_c].add(patch)

        def nll(u):
            model = jnp.maximum(render(u) + bg_img, 1e-6)
            return jnp.sum(model - y_img * jnp.log(model))

        grad_fn = jax.value_and_grad(nll)

        def adam_step(state, _):
            u, m, v, t = state
            _, g = grad_fn(u)
            g = jnp.where(jnp.isfinite(g), g, 0.0) * mask[:, None]
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            t = t + 1.0
            upd = (m / (1.0 - 0.9**t)) / (jnp.sqrt(v / (1.0 - 0.999**t)) + 1e-8)
            return (u - 0.05 * upd, m, v, t), None

        (u_fin, _, _, _), _ = lax.scan(
            adam_step,
            (u_init, jnp.zeros_like(u_init), jnp.zeros_like(u_init), 0.0),
            None,
            length=n_steps,
        )
        u_fin = jnp.where(jnp.isfinite(u_fin), u_fin, u_init)

        amp, r, c, sig = physical(u_fin)
        refined = jnp.stack([amp, r, c, sig], axis=1)
        # A refined peak that ran away from its own bounding box is not a
        # refinement of that peak any more, so keep the pre-fit values there.
        moved = jnp.sqrt((r - peaks[:, 1]) ** 2 + (c - peaks[:, 2]) ** 2)
        keep = (
            active
            & (moved < float(self.max_k_rad))
            & jnp.all(jnp.isfinite(refined), axis=1)
        )
        return jnp.where(keep[:, None], refined, peaks)

    def position_curvature(self, y_img, bg_img, peaks):
        """Direction-resolved second-order position certificate.

        For each peak (amp, r, c, sigma), returns the eigenvalues of the
        2x2 Hessian of the full Poisson NLL with respect to that peak's
        (r, c) at its reported position (other peaks held fixed), together
        with the eigenvalues of the silence part alone -- the Hessian of
        the peak's overlap mass with the zero-count set, amp * d^2(sum_Z
        Phi)/dxi^2.  A nonpositive smallest total eigenvalue means the
        subpixel position is not identifiable at this significance: the
        refined position sits on a saddle or ridge of the likelihood and
        can bifurcate under arithmetic noise.  The silence part isolates
        the deterministic stiffness contributed by nearby dark regions
        ("walls"): a straight dark boundary at distance D contributes
        transverse stiffness 2*pi*amp*Q''(D/sigma), peaking near D=sigma
        at ~24% of the peak's full position Fisher information.

        Diagnostic helper (Python loop over peaks, O(N^2) renders); not on
        the batched hot path.  Returns (eig_total, eig_silence), each of
        shape [N, 2], ascending.
        """
        y = jnp.asarray(y_img)
        bg = jnp.asarray(bg_img)
        peaks = jnp.asarray(peaks)
        H_img, W_img = y.shape
        rows = jnp.arange(H_img, dtype=jnp.float32)[:, None]
        cols = jnp.arange(W_img, dtype=jnp.float32)[None, :]
        z_mask = (y < 1.0).astype(jnp.float32)

        def atom(amp, r, c, sig):
            s2 = sig * jnp.sqrt(2.0) + 1e-6
            ey = jax.scipy.special.erf((rows - r + 0.5) / s2) - jax.scipy.special.erf(
                (rows - r - 0.5) / s2
            )
            ex = jax.scipy.special.erf((cols - c + 0.5) / s2) - jax.scipy.special.erf(
                (cols - c - 0.5) / s2
            )
            return amp * (jnp.pi / 2.0) * (sig**2) * ey * ex

        n_peaks = int(peaks.shape[0])
        eig_total = np.zeros((n_peaks, 2), dtype=np.float64)
        eig_silence = np.zeros((n_peaks, 2), dtype=np.float64)
        for n in range(n_peaks):
            amp, r0, c0, sig = (float(peaks[n, i]) for i in range(4))
            others = bg + sum(
                (
                    atom(*(float(peaks[m, i]) for i in range(4)))
                    for m in range(n_peaks)
                    if m != n
                ),
                jnp.zeros_like(y),
            )

            def nll_rc(rc, amp=amp, sig=sig, others=others):
                U = jnp.maximum(others + atom(amp, rc[0], rc[1], sig), 1e-6)
                return jnp.sum(U - y * jnp.log(U))

            def silence_mass(rc, amp=amp, sig=sig):
                return jnp.sum(z_mask * atom(amp, rc[0], rc[1], sig))

            rc0 = jnp.asarray([r0, c0], dtype=jnp.float32)
            h_tot = np.asarray(jax.hessian(nll_rc)(rc0), dtype=np.float64)
            h_sil = np.asarray(jax.hessian(silence_mass)(rc0), dtype=np.float64)
            eig_total[n] = np.linalg.eigvalsh(0.5 * (h_tot + h_tot.T))
            eig_silence[n] = np.linalg.eigvalsh(0.5 * (h_sil + h_sil.T))
        return eig_total, eig_silence

    def find_peaks_batch(self, images_batch):
        # Detector frames arrive as integer counts.  Everything below is a
        # convolution, and cuDNN will not lower an integer convolution, so the
        # cast is required rather than cosmetic: without it the background
        # estimate fails outright on real data.
        images_batch = np.asarray(images_batch, dtype=np.float32)
        B, H, W = images_batch.shape

        # Odd, as the greedy path already forces it (sparse_rbf.py).  An even
        # window has no centre pixel: the old exact filter then returned an
        # H+1 x W+1 map that the shape guard below silently cropped, which is a
        # half-pixel shift of the background against the image rather than a
        # harmless size mismatch.  max_sigma = 8 (window 40) hits this.
        filter_size = max(15, int(self.max_sigma * 5))
        if filter_size % 2 == 0:
            filter_size += 1
        bg_map = np.full_like(images_batch, 10.0)
        try:
            # The quantile-inversion rate map, not the median background: the
            # median of Poisson(mu) is identically zero below mu = log 2, so
            # on sparse frames the median map collapses to its clamp and every
            # significance downstream is measured against a background
            # hundreds of times too small.  See compute_rate_batch.  It also
            # retires this branch's subsampled-median workaround
            # (MEDIAN_MAX_SAMPLES): there is no window-sized sort left to
            # subsample.  The legacy greedy finder keeps the median path
            # unchanged.
            from subhkl.search.sparse_rbf import compute_rate_batch

            # Chunked for the same reason the greedy finder chunks it: a full
            # detector scan is far too much to hold on the device at once.
            bg_chunk = min(self.chunk_size, max(1, B // 4))
            pieces = []
            for start in range(0, B, bg_chunk):
                piece = compute_rate_batch(
                    jnp.asarray(
                        images_batch[start : start + bg_chunk], dtype=jnp.float32
                    ),
                    filter_size,
                )
                piece.block_until_ready()
                pieces.append(np.asarray(piece, dtype=np.float32))
            bg_map = np.concatenate(pieces, axis=0)
            if bg_map.shape != images_batch.shape:
                bg_map_fixed = np.zeros_like(images_batch)
                mh, mw = min(H, bg_map.shape[1]), min(W, bg_map.shape[2])
                bg_map_fixed[:, :mh, :mw] = bg_map[:, :mh, :mw]
                bg_map = bg_map_fixed
        except ImportError:
            pass

        self._last_bg_map = bg_map

        # The bank was sized against expected_background and
        # expected_peak_amplitude, with z calibrated for a nominal frame.
        # Amplitude is the driver (a brighter peak puts more photons behind
        # the same shape mismatch while the tax stays fixed; the background
        # nearly cancels at fixed amplitude), so re-check the built bank with
        # the quantities this batch actually measured -- the rate map's bright
        # quantile for the background, a top-quantile count above it as the
        # bright-peak amplitude proxy, and the real frame for z -- and warn.
        # The bank is already built, so this cannot resize it, but it names
        # the fix.
        if self.num_sigmas >= 2:
            bg_hi = float(np.percentile(bg_map, 90.0))
            amp_hi = max(float(np.percentile(images_batch, 99.99)) - bg_hi, 1.0)
            # The moment amplitude may stand in for the top-quantile count: its
            # inputs are aggregates over a footprint, so it is safe to let into
            # the resize, which is what amp_hi itself is excluded from.
            # Default preprocessing: the peak family's own radial profile,
            # measured from this batch by the same box arithmetic as the
            # amplitude census, replaces the Gaussian trunk.  One uniform
            # stretch of the u axis (measured ~6% on synthetics, from
            # neighbour-tail spill in the window moments) is absorbed by the
            # sigma grid, so only the shape relative to a Gaussian at fixed
            # second moment matters -- which is exactly what the solver pays
            # atoms for.  Too few qualifying peaks leaves the Gaussian in
            # place, stated rather than silent.
            if self.profile_file == "auto" and self._measured_trunk is None:
                measured_trunk = _measure_radial_profile(
                    images_batch, bg_map, bg_hi, self.max_sigma
                )
                if measured_trunk is not None:
                    self._measured_trunk = measured_trunk
                    self._set_bank(np.asarray(self.sigmas))
                    if self.show_steps:
                        u, f = measured_trunk
                        g = np.exp(-0.5 * u**2)
                        i = int(np.argmin(np.abs(u - 1.5)))
                        print(
                            f"  > peak profile measured from this batch "
                            f"({len(u)} radial bins; f/gaussian at u=1.5: "
                            f"{f[i] / max(g[i], 1e-9):.2f}); bank rebuilt "
                            "on the measured trunk"
                        )
                elif self.show_steps:
                    print(
                        "  > peak profile: too few qualifying peaks in this "
                        "batch; keeping the Gaussian atom"
                    )

            amp_for_resize = self.expected_peak_amplitude
            if self._amp_is_auto:
                # The requested fragmentation rate selects which quantile of
                # the measured brightness distribution the bank protects:
                # peaks above it may fragment, and each contributes ~2
                # unsupported atoms, so the quantile is arithmetic on the
                # disjoint-window census -- no solve involved (see
                # _frag_protected_quantile).
                if self.max_fragmentation_rate > 0:
                    census = _moment_census(images_batch, bg_map, bg_hi, counting=True)
                    if census.size:
                        q = _frag_protected_quantile(
                            self.max_fragmentation_rate, census.size / B
                        )
                        measured = float(np.percentile(census, q))
                    else:
                        measured = None
                else:
                    measured = _moment_peak_amplitude(images_batch, bg_map, bg_hi)
                if measured is not None:
                    amp_for_resize = measured
                    self.expected_peak_amplitude = measured

            # An auto-sized bank re-derives its grid from what the data
            # implies -- the real frame (which sets z through the calibration)
            # and the measured background -- and rebuilds itself when that
            # differs from the construction-time assumptions.  The measured
            # peak amplitude deliberately stays out of the resize: a
            # top-quantile count is one hot pixel away from silently
            # inflating the bank, so it only feeds the warning below, where
            # the user decides.
            if self._auto_bank:
                grid = _auto_sigma_grid(
                    self.min_sigma,
                    self.max_sigma,
                    self.gamma,
                    self.ref_sigma,
                    self.false_alarms_per_image,
                    bg_hi,
                    amp_for_resize,
                    H,
                    W,
                    fid_res=self.fid_residual,
                )
                cur = np.asarray(self.sigmas, dtype=float)
                # Rebuild only for a change that matters: a different channel
                # count, or grid points shifted beyond the ~5% level below
                # which the criterion's steepness makes placement immaterial.
                # (A rebuild clears the jitted-method caches, so cosmetic
                # rebuilds would recompile the whole solve for nothing.)
                if len(grid) != len(cur) or not np.allclose(grid, cur, rtol=0.05):
                    self._set_bank(grid)
                    if self.show_steps:
                        print(
                            f"  > sigma bank re-sized to {int(self.K_weights.shape[0])} "
                            f"channels for this data (frame {H}x{W}, measured "
                            f"background ~{bg_hi:.2f} photons/px): "
                            + ", ".join(f"{v:.3g}" for v in grid)
                        )

            ratio, near = _frag_bank_ratio(
                np.asarray(self.sigmas),
                self.gamma,
                self.ref_sigma,
                self.false_alarms_per_image,
                bg_hi,
                amp_hi,
                H,
                W,
                fid_res=self.fid_residual,
            )
            if ratio > 1.0:
                warnings.warn(
                    f"the sigma bank (sized for peaks of "
                    f"{self.expected_peak_amplitude:g} photons over a "
                    f"{self.expected_background:g} photons/px background) is "
                    "predicted to fragment the brightest peaks in this data "
                    f"(measured background ~{bg_hi:.2f} photons/px, peak "
                    f"amplitudes up to ~{amp_hi:.0f}; worst near sigma ~ "
                    f"{near:.1f}, fidelity/tax ratio {ratio:.1f}): they may "
                    "be reported as clusters of narrower atoms.  Reconstruct "
                    f"the finder with expected_peak_amplitude={amp_hi:.0f} "
                    f"(and expected_background={bg_hi:.1f}, num_sigmas=None) "
                    "to size the bank for this data.",
                    stacklevel=2,
                )

        PAD = 2 * self.max_k_rad + 1
        pad_y = PAD // 2
        pad_x = PAD // 2

        images_padded = jnp.pad(
            images_batch, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="edge"
        )
        bg_padded = jnp.pad(
            bg_map, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="edge"
        )

        results = []
        rejected_counts = []

        MARGIN = max(3, self.max_k_rad)

        # vmap multiplies the solver's per-image working set by the batch, so a
        # full detector scan does not fit on the card: on the 1114-frame CG4D
        # garnet stack XLA asks for an 82 GiB program and the allocator gives up
        # at 69.8 GiB on a 96 GB H100.  Chunk it for the same reason the
        # background estimate above is chunked.  Peaks are pulled back to the
        # host inside the chunk loop, which keeps the coefficient tensors, at
        # [chunk, num_sigmas, H, W] the largest thing here, from accumulating.
        #
        # The images are solved independently, so chunking is not an
        # approximation -- but it is not bit-identical either.  XLA compiles a
        # separate kernel per batch shape, and the resulting rounding
        # differences can carry a coefficient across the selection threshold: on
        # a 6-frame synthetic case, solving one image at a time rather than all
        # six moved two frames' peak counts.  A batch of chunk_size or fewer
        # still takes a single chunk and so matches the unchunked path exactly,
        # which at the default of 64 covers every unit test here.
        #
        # Unlike the background rule this does not also divide by four: the
        # solver is the expensive stage, chunk_size is the knob documented to
        # control exactly this, and subdividing a batch that already fits only
        # gives up vmap parallelism.
        solve_chunk = max(1, min(self.chunk_size, B))
        solve_batch = jax.jit(jax.vmap(self._solve_ssn_cg_global))

        for start in range(0, B, solve_chunk):
            stop = min(start + solve_chunk, B)
            c_tensors = solve_batch(images_padded[start:stop], bg_padded[start:stop])

            for i in range(start, stop):
                # Extraction and sliding refinement in one chunked sweep; the
                # coordinates still match the padded image the model is
                # rendered on, and only the valid rows come back.
                valid_peaks = self._extract_peaks_all(
                    c_tensors[i - start],
                    images_padded[i],
                    bg_padded[i],
                    border=pad_y + MARGIN,
                )

                valid_peaks[:, 1] -= pad_y
                valid_peaks[:, 2] -= pad_x

                keep = (
                    (valid_peaks[:, 1] >= MARGIN)
                    & (valid_peaks[:, 1] < H - MARGIN)
                    & (valid_peaks[:, 2] >= MARGIN)
                    & (valid_peaks[:, 2] < W - MARGIN)
                )

                # An atom whose width has run to the edge of the bank is the solver
                # asking for a wider basis than it was given.  That can mean
                # unmodelled smooth background -- or simply that max_sigma is too
                # small for the data, in which case every real peak saturates the
                # bank too and this discards them.  On a real MANDI scan, where the
                # peaks have a median width of ~34 px against a max_sigma of 5, it
                # removed 87% of genuine detections (466 peaks down to 60), so it is
                # off by default and is a diagnostic to reach for once the bank is
                # known to be wide enough.  The case
                # that motivated this is a diffuse halo whose background estimate
                # falls ~20% short at its centre, leaving a broad positive residual
                # that is then reported as a reflection sitting on the halo.  On real
                # Laue data the structures that trigger it -- thermal diffuse
                # scattering, powder rings, halos around strong reflections -- are
                # exactly the ones that should not be reported as peaks.
                if self.reject_boundary_sigma:
                    at_boundary = (
                        valid_peaks[:, 3] >= self.boundary_sigma_frac * self.max_sigma
                    ) & keep
                    n_rejected = int(np.count_nonzero(at_boundary))
                    keep &= ~at_boundary
                else:
                    n_rejected = 0

                # Record rather than drop silently: a run that discards many atoms
                # here is telling you the background model is leaving structure
                # behind, or that max_sigma is too small for this data, and either
                # is worth knowing.
                rejected_counts.append(n_rejected)
                if self.show_steps and n_rejected:
                    print(
                        f"  > Rejected {n_rejected} atom(s) with sigma at the bank "
                        f"edge (>= {self.boundary_sigma_frac:.2f} * "
                        f"{self.max_sigma:g}); likely unmodelled background."
                    )

                results.append(valid_peaks[keep])

            # Release the chunk before the next one is dispatched, so peak
            # device memory stays at one chunk rather than two.
            del c_tensors

        self.n_boundary_rejected = rejected_counts

        # Goodness-of-fit exit report, matching the greedy pipeline
        # (sparse_rbf calls compute_metrics on its final peaks): total NLL,
        # BIC, and residual deviance per degree of freedom
        # (n_pixels - 4 * n_atoms).  Deviance/DoF near 1 is a calibrated
        # Poisson fit; well above 1 flags unmodelled structure (background
        # or missed peaks), well below 1 flags over-parameterization.
        # Stored on the instance for programmatic use and printed under
        # show_steps, exactly as the greedy code did.
        self.fit_metrics = self.compute_metrics(images_batch, bg_map, results, 1.0)

        # Per-peak counterpart of that global number: two values per reported
        # atom, aligned with `results`.  They answer different questions and
        # neither substitutes for the other; see `compute_peak_metrics`.
        self.peak_deviance, self.peak_residual_deviance = self.compute_peak_metrics(
            images_batch, bg_map, results
        )

        # Unsupported-atom rate: atoms whose leave-one-out deviance falls
        # below the chi^2_4 95% point are atoms the data do not independently
        # support.  This is a different statistic from the m0 false-alarm
        # budget, not a restatement of it: a false positive was admitted on a
        # >= z noise excursion, so its leave-one-out deviance is typically
        # ~ z^2 (~17 nats) and it *passes* this test.  What fails it is
        # redundancy -- an atom whose contribution its neighbours absorb,
        # which is exactly a fragment of a peak reported as a cluster.  This
        # is the observable max_fragmentation_rate bounds.  Stored always,
        # printed with the final stats.
        n_unsupported = sum(int(np.count_nonzero(d < 9.49)) for d in self.peak_deviance)
        self.unsupported_atoms_per_image = n_unsupported / max(B, 1)
        if self.show_steps:
            print(
                f"  > Unsupported atoms: {n_unsupported} "
                f"({self.unsupported_atoms_per_image:.2f}/image).  Not the "
                f"m0 = {self.false_alarms_per_image:g}/image false-alarm "
                "count: noise admissions carry their ~z^2 of support and "
                "pass this test; failing it means redundancy -- a peak "
                "reported as a cluster of atoms none of which the data "
                "support alone."
            )

        # The false-positive diagnostic, parallel to the unsupported count:
        # an atom admitted on a marginal noise excursion carries the
        # admission bar's worth of leave-one-out support and no more --
        # ~z_bar^2 plus the median chi^2_3 refit gain.  Atoms inside
        # [chi^2_4 95%, z_bar^2 + 2.4) are therefore consistent with a noise
        # admission.  This bounds rather than counts false positives --
        # genuinely dim peaks land in the same band -- but the calibration
        # holds E[FP] = m0, so a count far above m0 means either a large dim
        # population or a broken calibration, and a count near m0 says the
        # budget is being spent as designed.
        a_bar = float(np.mean(np.asarray(self.effective_alpha(H, W))))
        fp_band_hi = a_bar * a_bar + 2.4
        n_fp_like = sum(
            int(np.count_nonzero((d >= 9.49) & (d < fp_band_hi)))
            for d in self.peak_deviance
        )
        self.fp_consistent_atoms_per_image = n_fp_like / max(B, 1)
        if self.show_steps:
            print(
                f"  > False-positive-consistent atoms: {n_fp_like} "
                f"({self.fp_consistent_atoms_per_image:.2f}/image), with "
                f"leave-one-out support in [9.5, {fp_band_hi:.1f}) nats -- "
                "no more than a marginal admission carries -- against the "
                f"budget m0 = {self.false_alarms_per_image:g}/image.  An "
                "upper bound on realised false positives: dim real peaks "
                "land in this band too."
            )

        # Bank-edge report, both ends.  An atom pinned at the ceiling is a
        # width the bank could not represent (max_sigma too small -- and the
        # solver then tiles the reflection with several atoms, which the
        # per-peak metrics score as a *better* fit); one pinned at the floor
        # is the mirror statement about min_sigma.  These are the
        # configuration-selection statistics: the operating point is the
        # smallest ceiling at which saturation reaches zero, with no
        # hand-chosen threshold involved.  Always computed and stored;
        # printed under show_steps.
        all_sigma = np.concatenate(
            [np.asarray(p)[:, 3] for p in results if len(p)] or [np.zeros(0)]
        )
        n_atoms = all_sigma.size
        self.bank_saturation = {
            "ceiling": float(
                np.mean(all_sigma >= self.boundary_sigma_frac * self.max_sigma)
            )
            if n_atoms
            else 0.0,
            "floor": float(
                np.mean(all_sigma <= self.min_sigma / self.boundary_sigma_frac)
            )
            if n_atoms
            else 0.0,
        }
        if self.show_steps and n_atoms:
            print(
                f"  > Bank saturation: {100 * self.bank_saturation['ceiling']:.1f}% "
                f"of atoms at the ceiling (max_sigma={self.max_sigma:g}), "
                f"{100 * self.bank_saturation['floor']:.1f}% at the floor "
                f"(min_sigma={self.min_sigma:g}).  Nonzero ceiling saturation "
                "means widths were imposed, not measured: raise max_sigma."
            )

        return results

    @partial(jit, static_argnames=["self"])
    def _peak_metrics_image(self, peaks, target, bg):
        """Leave-one-out and local residual deviance for every atom of one image.

        **Leave-one-out deviance.**

        For each atom n this returns

            dD_n = D(model without n) - D(model) = 2 sum_pix [ y log(U/U_-n) - r_n ]

        with U the full model (background plus every atom), r_n atom n's own
        contribution and U_-n = U - r_n.  That is the likelihood-ratio
        statistic for the presence of atom n, holding the other atoms fixed,
        and it is calibrated against chi^2 on the atom's four parameters
        (95% point 9.49): dD well above that is a peak the data insist on,
        dD near zero is one the data are indifferent to, and dD < 0 is an atom
        that actively degrades the fit at its reported parameters.  It is the
        per-peak reading of the same currency as the global Deviance/DoF, so
        the two can be compared directly.

        The sum runs over the whole image by definition, but the finder's
        atoms are the kernel-bank atoms, which are identically zero outside a
        radius of ``max_k_rad`` pixels; every term outside that window has
        r_n = 0 and so contributes exactly nothing.  Summing over the window
        is therefore not a 3-sigma approximation to the image-wide sum, it *is*
        the image-wide sum for this model, at a cost of (2 max_k_rad + 1)^2
        pixels per peak instead of H*W.

        **Local residual deviance.**

        dD says whether an atom is carrying real signal; it says almost
        nothing about whether it is carrying it *correctly*.  An atom fitted
        with too large a sigma still explains a great deal of density, so
        removing it still costs a great deal: on a sigma = 2 peak, dD reads
        26154 at the true width and still 15364 at sigma = 6, a factor of 1.7
        where the width is wrong by a factor of 3.  Every one of those passes
        any significance cut.

        The residual deviance answers the complementary question -- is this
        neighbourhood explained? -- by summing the goodness-of-fit deviance of
        the *fitted* model over the atom's own footprint,

            D_res,n / dof = sum_{|x - x_n| <= 3 sigma_n} 2 [ y log(y/U) - (y - U) ]
                            / (n_window - 4),

        with U the full model again, so a neighbour's tails and any unmodelled
        background count against the atom sitting in them.  This is calibrated
        the same way the global Deviance/DoF is: near 1 for a model that fits,
        above 1 for structure left behind.  On the same sigma = 2 peak it reads
        1.06 at the true width against 19.1 at sigma = 1.5 and 9.5 at
        sigma = 3, and on a genuinely broad sigma = 5 peak it reads 0.93 at
        sigma = 5 and 62.8 at sigma = 3 -- so it tracks whether the width is
        *right*, not whether it is large.

        Two caveats, both measured on Poisson nulls with an exact model.  The
        statistic carries a mild positive bias at low count rates -- E[D/dof]
        is 1.04 at 10 counts/pixel, 1.19 at 1 count/pixel, with a spread of
        about 0.15 -- so 1.2 is the null there, not 1.0.  Below roughly 0.5
        counts/pixel it breaks down in the other direction (0.49 at 0.1
        counts/pixel): almost every pixel is empty, the deviance has nothing
        to test, and a low value stops meaning a good fit.  Read it as a fit
        diagnostic only where the background is counted rather than dark.

        The 3-sigma footprint is a choice, not an identity -- unlike the dD
        window there is nothing exact to preserve here, and the alternative of
        using the full kernel box dilutes the signal badly (4.0 rather than
        116.8 on the sigma = 1 case above) because it averages the mismatch
        against a large annulus of pure background.
        """
        R = self.max_k_rad
        d = jnp.arange(-R, R + 1)
        dy, dx = jnp.meshgrid(d, d, indexing="ij")

        amp = peaks[:, 0][:, None, None]
        r_c = peaks[:, 1][:, None, None]
        c_c = peaks[:, 2][:, None, None]
        sig = peaks[:, 3][:, None, None]

        H, W = target.shape
        ri = jnp.round(r_c).astype(jnp.int32) + dy[None]
        ci = jnp.round(c_c).astype(jnp.int32) + dx[None]
        inside = (ri >= 0) & (ri < H) & (ci >= 0) & (ci < W)
        ri_c = jnp.clip(ri, 0, H - 1)
        ci_c = jnp.clip(ci, 0, W - 1)

        # Same pixel-integrated Gaussian the forward operator applies, at the
        # atom's refined (sub-pixel) centre and width.
        sig_sq2 = sig * jnp.sqrt(2.0) + 1e-6
        erf_y = jax.scipy.special.erf(
            (ri - r_c + 0.5) / sig_sq2
        ) - jax.scipy.special.erf((ri - r_c - 0.5) / sig_sq2)
        erf_x = jax.scipy.special.erf(
            (ci - c_c + 0.5) / sig_sq2
        ) - jax.scipy.special.erf((ci - c_c - 0.5) / sig_sq2)
        r_n = amp * (jnp.pi / 2.0) * (sig**2) * erf_y * erf_x
        r_n = jnp.where(inside & (amp > 0), r_n, 0.0)

        # Model image assembled from those same truncated atoms, so that the
        # U appearing in the window is consistent with the r_n subtracted from
        # it -- including the tails of *other* atoms reaching into the window.
        model = jnp.zeros((H, W), dtype=r_n.dtype).at[ri_c, ci_c].add(r_n)
        u_full = jnp.maximum(bg + model, 1e-9)

        u_win = u_full[ri_c, ci_c]
        u_minus = jnp.maximum(u_win - r_n, 1e-9)
        y_win = target[ri_c, ci_c]

        loo = 2.0 * jnp.sum(y_win * jnp.log(u_win / u_minus) - r_n, axis=(1, 2))

        # Residual deviance over the atom's own 3-sigma footprint, clipped to
        # the kernel box that is actually gathered above.
        radius = jnp.minimum(3.0 * sig, float(R))
        foot = inside & (amp > 0) & ((ri - r_c) ** 2 + (ci - c_c) ** 2 <= radius**2)
        y_safe = jnp.maximum(y_win, 1.0)  # the y log y term vanishes at y = 0
        dev_pix = 2.0 * (
            jnp.where(y_win > 0, y_win * jnp.log(y_safe / u_win), 0.0) - (y_win - u_win)
        )
        n_foot = jnp.sum(foot, axis=(1, 2))
        dof = jnp.maximum(n_foot - 4, 1)
        resid = jnp.sum(jnp.where(foot, dev_pix, 0.0), axis=(1, 2)) / dof

        return loo, resid

    @classmethod
    def calibrate_fragmentation_residual(
        cls,
        images_batch,
        max_fragmentation_rate=1.0,
        residual_ladder=(2.5, 5.0, 10.0, 20.0, 40.0),
        **finder_kwargs,
    ):
        """Fit the criterion's empirical constant to a requested rate.

        The bank-sizing analogue of the false-alarm calibration: ``m0``
        turns "how many spurious peaks per image are tolerable" into a
        threshold, and this turns "how many unsupported atoms per image are
        tolerable" into ``fid_residual``.  The observable is the
        leave-one-out deviance already computed by compute_peak_metrics: an
        atom whose removal barely changes the fit (dD below the chi^2_4 95%
        point) is one the data do not independently support, which is
        exactly what a carpet fragment is -- its neighbours absorb it.  The
        same statistic also counts the redundant atoms an *over*-dense bank
        re-introduces, so the measured rate is a U-shaped curve in bank size
        and the calibration picks the cheapest bank meeting the target
        rather than the largest.

        (The fit-quality caveat on compute_peak_metrics -- that splitting
        *improves* within-configuration statistics -- applies to the
        residual deviance, not to this count: leave-one-out support of a
        redundant atom collapses instead of improving.)

        Runs one full solve of ``images_batch`` per ladder rung, so this is
        an offline *validation* tool, not a pipeline step: the pipeline meets
        the requested rate without solving, by mapping it onto the protected
        brightness quantile of the moment census (_frag_protected_quantile).
        Use this when the analytic mapping itself is in question --
        e.g. commissioning a new instrument -- to confirm the measured
        unsupported-atom curve against the residual ladder.  Returns
        ``(fid_residual, rows)`` with ``rows`` of ``(residual, num_sigmas,
        unsupported_atoms_per_image)`` for the report; pass the returned
        value back as ``fid_residual=`` (it is not stored globally).
        """
        images_batch = np.asarray(images_batch, dtype=np.float32)
        rows = []
        for res in residual_ladder:
            finder = cls(num_sigmas=None, fid_residual=float(res), **finder_kwargs)
            finder.find_peaks_batch(images_batch)
            # find_peaks_batch measures the unsupported-atom rate as part of
            # its final statistics; reuse it rather than re-deriving.
            rows.append(
                (float(res), finder.num_sigmas, finder.unsupported_atoms_per_image)
            )
        met = [r for r in rows if r[2] <= max_fragmentation_rate]
        # Cheapest bank that meets the rate; smaller residual breaks ties.
        # If no rung meets it, take the best rate observed -- the report rows
        # show the caller how far off the target was.
        chosen = (
            min(met, key=lambda r: (r[1], r[0]))
            if met
            else min(rows, key=lambda r: r[2])
        )
        return chosen[0], rows

    def compute_peak_metrics(self, images_raw, bg_map, peaks_list):
        """Per-peak quality metrics: (leave-one-out, residual), one array each
        per image.

        The two are complementary and a peak needs both to be trusted: a high
        dD with a residual near 1 is a real peak, well fitted; a high dD with a
        large residual is a real peak fitted with the wrong shape (a mis-sized
        sigma, or a neighbour it has swallowed); a dD near or below zero is an
        atom the data do not support at all.  See ``_peak_metrics_image``.

        .. warning::
            Both metrics are *within-configuration* statistics.  They rank
            peaks and track regressions at a fixed bank; they must not be
            used to compare configurations that change how reflections are
            tiled (``max_sigma``, ``num_sigmas``, ``gamma``).  N narrow atoms
            tiling one broad reflection each fit their own sub-footprint
            extremely well, so splitting *improves* both statistics by
            construction: measured on real CG4D data, a ceiling-starved bank
            that tiled three reflections with 23 atoms scored 0% residual
            misfit while the correct three-atom solution scored 24% -- the
            natural "lower is better" reading is exactly inverted across
            bank changes.  For choosing a configuration use the ceiling
            saturation fraction reported by ``find_peaks_batch`` (smallest
            ceiling at which it reaches zero) and the global BIC from
            ``compute_metrics``, whose per-atom parameter penalty is the
            anti-tiling term the local metrics lack.
        """
        B = images_raw.shape[0]
        max_k = max([len(p) for p in peaks_list] + [1])

        # Pad to a single shape so the batch compiles once; padded rows carry
        # zero amplitude and drop out of the statistic.
        peaks_padded = np.zeros((B, max_k, 4), dtype=np.float32)
        for b in range(B):
            n = len(peaks_list[b])
            if n > 0:
                peaks_padded[b, :n, :] = peaks_list[b]
            peaks_padded[b, n:, 3] = 1.0

        out, out_res = [], []
        for b in range(B):
            n = len(peaks_list[b])
            if n == 0:
                out.append(np.zeros(0, dtype=np.float32))
                out_res.append(np.zeros(0, dtype=np.float32))
                continue
            dev, resid = self._peak_metrics_image(
                jnp.asarray(peaks_padded[b]),
                jnp.asarray(images_raw[b]),
                jnp.asarray(bg_map[b]),
            )
            out.append(np.asarray(dev)[:n].astype(np.float32))
            out_res.append(np.asarray(resid)[:n].astype(np.float32))

        if self.show_steps:
            flat = np.concatenate(out) if any(len(o) for o in out) else np.zeros(0)
            flat_res = (
                np.concatenate(out_res) if any(len(o) for o in out_res) else np.zeros(0)
            )
            if flat.size:
                weak = int(np.count_nonzero(flat < 9.49))
                misfit = int(np.count_nonzero(flat_res > 2.0))
                print(
                    f"  > Peak deviance: median {np.median(flat):.3g}, "
                    f"{weak}/{flat.size} below the chi^2_4 95% point (9.49)"
                )
                print(
                    f"  > Peak residual deviance/DoF: median "
                    f"{np.median(flat_res):.3g}, {misfit}/{flat_res.size} above 2 "
                    f"(shape mismatch; target ~ 1)"
                )

        return out, out_res

    @partial(jit, static_argnames=["self"])
    def _predict_batch_scan(self, peaks, x_grid):
        def render_peak(p):
            total_I, r, c, sigma = p[0], p[1], p[2], p[3]
            sig_sq2 = sigma * jnp.sqrt(2.0) + 1e-6
            erf_y = jax.scipy.special.erf(
                (x_grid[0] - r + 0.5) / sig_sq2
            ) - jax.scipy.special.erf((x_grid[0] - r - 0.5) / sig_sq2)
            erf_x = jax.scipy.special.erf(
                (x_grid[1] - c + 0.5) / sig_sq2
            ) - jax.scipy.special.erf((x_grid[1] - c - 0.5) / sig_sq2)
            return total_I * (jnp.pi / 2.0) * (sigma**2) * erf_y * erf_x * (total_I > 0)

        def render_image(peaks_img):
            return jnp.sum(vmap(render_peak)(peaks_img), axis=0)

        return render_image(peaks)

    def compute_metrics(self, images_raw, bg_map, peaks_list, global_max):
        """
        Args:
            images_raw, bg_map: [photons/Pixel]
            global_max: [photons/Pixel]
        """
        B, H, W = images_raw.shape
        yy, xx = np.indices((H, W))  # [Pixel^0.5]
        x_grid = jnp.array([yy, xx])  # [Pixel^0.5]

        if self.show_steps:
            print("\n  [Metrics] Calculating goodness-of-fit...")

        max_k = max([len(p) for p in peaks_list] + [1])
        peaks_padded = np.zeros((B, max_k, 4), dtype=np.float32)
        counts_per_image = np.zeros(B, dtype=np.float32)

        for b in range(B):
            n = len(peaks_list[b])
            if n > 0:
                peaks_padded[b, :n, :] = peaks_list[b]
                if n < max_k:
                    peaks_padded[b, n:, 3] = 1.0
            counts_per_image[b] = n

        if self.loss == "poisson":
            loss_code = 1
        elif self.loss == "gaussian":
            loss_code = 0
        else:
            raise ValueError("Unsupported loss")

        @jit
        def process_one_image(peaks, target_raw, median_val, k_val):
            recon_peaks = self._predict_batch_scan(peaks, x_grid)  # [photons/Pixel]
            recon_total = jnp.maximum(recon_peaks + median_val, 1e-9)  # [photons/Pixel]

            if loss_code == 1:
                # 1. Exact Poisson NLL using xlogy
                nll = jnp.sum(
                    recon_total - jax.scipy.special.xlogy(target_raw, recon_total)
                )  # [photons/Pixel]
                # 2. Exact Poisson Deviance
                term = jax.scipy.special.xlogy(target_raw, target_raw / recon_total) - (
                    target_raw - recon_total
                )
                dev = 2 * jnp.sum(term)
            else:
                diff = recon_total - target_raw  # [photons/Pixel]
                nll = 0.5 * jnp.sum(diff**2)  # [photons^2 / Pixel^2]
                dev = jnp.sum((diff**2) / recon_total)  # [-]

            n_pix = target_raw.size
            n_params = k_val * 4
            bic = n_params * jnp.log(n_pix) + 2 * nll
            return nll, bic, dev

        nll_total, bic_total, deviance_total = 0.0, 0.0, 0.0

        for b in range(B):
            nll, bic, dev = process_one_image(
                jnp.array(peaks_padded[b]),
                jnp.array(images_raw[b]),
                jnp.array(bg_map[b]),
                jnp.array(counts_per_image[b]),
            )
            nll_total += float(nll)
            bic_total += float(bic)
            deviance_total += float(dev)

        pixels_total = B * H * W
        params_total = float(np.sum(counts_per_image)) * 4
        dof = max(pixels_total - params_total, 1)
        dev_per_dof = deviance_total / dof

        if self.show_steps:
            target_str = (
                "(Target ~ 1.0)" if loss_code == 1 else "(MSE/Variance of noise)"
            )
            print(f"  > Total NLL: {nll_total:.2e}")
            print(f"  > Total BIC: {bic_total:.2e}")
            print(f"  > Deviance/DoF: {dev_per_dof:.4f} {target_str}")

        return {"nll": nll_total, "bic": bic_total, "deviance_nu": dev_per_dof}
