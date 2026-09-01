"""Orientation search on SO(3) by bandlimited spherical correlation.

The Laue geometry collapses indexing onto the unit sphere: for incident beam
k_i the elastic condition fixes the wavelength of every reflection, so the
scattered direction depends only on the *unit* vector of Q -- Ghat =
(k_f_hat - k_i_hat)/|..| -- never on |Q|.  Each detector spot therefore
measures one direction on S^2, harmonics n*G land on the same spot, and a
crystal orientation R is a rotation aligning the fixed set of reciprocal-
lattice unit directions {B hkl / |B hkl|} with the measured set.  Finding R
is a sparse recovery problem over SO(3):

    f  =  sum_r c_r Lambda(R_r) g  +  clutter,    c >= 0 sparse,

with f the measured direction density, g the model density, and Lambda(R)
rotation of functions on the sphere.  The dual variable of that problem --
the matched filter -- is the spherical cross-correlation

    C(R) = <f, Lambda(R) g>,

and this module computes it in the spherical-harmonic domain, where the
per-degree coupling f^l (g^l)^dagger is an outer product: rank one per
degree, O(L^3) total work, no spherical grid at any point of the forward
projection (points and rings project analytically; the only grid is the
Euler-angle grid the correlogram is *read out* on).

The data side takes either found peaks (``project_points`` on Ghat
directions) or -- with no peak finding at all -- the raw frame
(``project_counts`` on background-subtracted pixel counts): the correlogram
is then the matched filter of every photon rather than every found peak,
and recovers orientations from spots individually below the finder's own
detection floor (measured: F = 5 counts per spot against a floor of 15).

Two model bases:

- ``project_points``: one smoothed delta per reciprocal-lattice direction
  (moderate cells, |directions| manageable);
- ``project_rings``: one great circle per crystallographic zone.  A zone
  [uvw] contains the reciprocal directions perpendicular to the real-space
  vector t = u a + v b + w c, a great circle on the sphere whose
  spherical-harmonic expansion is analytic (Funk--Hecke): ring axis n has
  coefficients 2 pi P_l(0) Y_lm(n)* -- rank one per degree again, so a
  ring costs exactly what a point costs.

  Which basis for which cell is a measured result, and it is the reverse
  of the first intuition.  A zone's excess over the isotropic direction
  background is a 1D chain of ~b/d_min points per radian against a 2D
  background of ~abc/d_min^3 / 4 pi points per steradian, so the contrast
  of the zone ribbons scales as d_min^2 / (a c sigma): zones dominate for
  SMALL cells (the classic Laue photographs full of conics are all
  small-cell crystals) and wash out for large ones (protein Laue is a
  uniform spot carpet).  Measured at sigma = 0.4 deg: an 8 A cell at
  d_min = 1.2 recovers through rings at z = 13; a 120 A cell at
  d_min = 2 gives z = 0.1 -- statistically blind -- while the point basis
  still works.  Large cells are instead a *bandwidth* problem: directions
  are resolvable only while their spacing ~ d_min/a exceeds pi/L, so
  L = 96 handles cells to roughly 60 A at d_min = 2; beyond that the
  correlation needs L ~ 3 a / (d_min) -- a few hundred -- which is the
  same O(L^3) per beta on an accelerator rather than in numpy, not a
  different algorithm.

Where an explicit threshold is not needed (and where it is).  Each
orientation atom aggregates M ~ hundreds of predicted directions.  A wrong
orientation matches the data at chance; a right one matches a finite
fraction.  The separation between the two, in units of the null's own
fluctuation, grows like sqrt(M) (measured by ``null_zscore``; typical peaks
sit far above the null even at 40% outliers), so admission needs no tuned
alpha here -- the aggregation is the evidence.  This is the SO(3) analogue
of the zero-count pixels' implicit L1 penalty in the image-domain solver
(the terms of the Poisson loss linear in the intensity), and it *suffices*
in this domain precisely because one atom explains many spots.  In the
image domain one atom explains one bump and the same argument gives no
false-alarm control; see tests/test_implicit_l1.py for both measurements.

Conventions.  Beam +z by default (subhkl.instrument.physics).  Active
rotations, R acting on model directions: predicted = R @ H.  Euler angles
z-y-z: R = Rz(alpha) Ry(beta) Rz(gamma).  All direction sets are treated as
antipodally symmetric (a lattice contains -G with G), so only even degrees
carry signal and odd degrees are dropped; orientations are recovered up to
the Laue symmetry of the cell, as in any Laue experiment.

Units: angles [rad] unless a name says _deg; directions are unit vectors in
the lab frame; d_min [Angstrom]; wavelengths [Angstrom]; B as returned by
subhkl.core.crystallography.cartesian_matrix_metric_tensor (|B hkl| = 1/d,
no 2 pi).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def ghat_from_kf(kf_dirs, ki=(0.0, 0.0, 1.0)):
    """Measured Q directions from scattered-beam unit vectors.

    Q = k_f - k_i and only the direction survives the Laue condition, so
    Ghat = (kf_hat - ki_hat) normalized.  [unit vectors, lab frame]
    """
    kf = np.asarray(kf_dirs, dtype=float)
    kf = kf / np.linalg.norm(kf, axis=-1, keepdims=True)
    ki_hat = np.asarray(ki, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)
    g = kf - ki_hat
    n = np.linalg.norm(g, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("a scattered direction coincides with the beam")
    return g / n


def lattice_directions(B, d_min, wavelength_band=None, ki=(0.0, 0.0, 1.0)):
    """Primitive reciprocal-lattice unit directions and their weights.

    Enumerates hkl with |B hkl| <= 1/d_min, reduces each to its primitive
    direction (gcd), and keeps one representative per +-pair.  With a
    wavelength band, a direction is kept if any harmonic q of its primitive
    vector is excited in band -- lambda_q = 2 |Ghat . ki_hat| / (q g0) with
    g0 = |B hkl_prim| [1/Angstrom] -- and its weight is the number of such
    harmonics; the sign of the pair that diffracts (Ghat . ki < 0) is the
    one tested.  Note the band cut depends on the *orientation* of the
    crystal, which is unknown during the search, so passing a band models
    the vertical extent of the excited set only approximately; the default
    (None, weight 1, resolution cut only) is the conservative choice.

    Returns (dirs [N, 3], weights [N]).
    """
    B = np.asarray(B, dtype=float)
    binv = np.linalg.inv(B)
    bounds = np.ceil(np.linalg.norm(binv, axis=1) / d_min).astype(int)
    h, k, l_ = np.meshgrid(*(np.arange(-b, b + 1) for b in bounds), indexing="ij")
    hkl = np.stack([h.ravel(), k.ravel(), l_.ravel()], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    g = hkl @ B.T
    gnorm = np.linalg.norm(g, axis=1)
    keep = gnorm <= 1.0 / d_min
    hkl, gnorm = hkl[keep], gnorm[keep]

    # primitive reduction and +- dedup: divide by the gcd, then orient each
    # primitive index to a canonical hemisphere in index space.
    gcd = np.gcd.reduce(np.abs(hkl), axis=1)
    prim = hkl // gcd[:, None]
    lead = np.where(
        prim[:, 0] != 0, prim[:, 0], np.where(prim[:, 1] != 0, prim[:, 1], prim[:, 2])
    )
    prim = prim * np.where(lead < 0, -1, 1)[:, None]
    prim = np.unique(prim, axis=0)

    gprim = prim @ B.T
    g0 = np.linalg.norm(gprim, axis=1)
    dirs = gprim / g0[:, None]

    if wavelength_band is None:
        return dirs, np.ones(len(dirs))

    lam_lo, lam_hi = map(float, wavelength_band)
    ki_hat = np.asarray(ki, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)
    # the member of the +- pair with Ghat . ki < 0 is the one that diffracts
    cosb = np.abs(dirs @ ki_hat)
    # harmonics q with lambda_q in band and q g0 within resolution
    q_hi = np.minimum(2.0 * cosb / (g0 * lam_lo), 1.0 / (d_min * g0))
    q_lo = 2.0 * cosb / (g0 * lam_hi)
    weights = np.floor(q_hi) - np.ceil(q_lo) + 1.0
    keep = weights >= 1.0
    return dirs[keep], weights[keep]


def zone_axes(cell_basis, max_index=3):
    """Zone axes [uvw] as unit vectors, weighted by their direction density.

    ``cell_basis`` is the real-space basis as rows [Angstrom].  The
    reciprocal directions of zone [uvw] lie in the plane perpendicular to
    t = u a + v b + w c; their 2D lattice in that plane has point density
    proportional to 1/|t| (adjacent zone layers are 2 pi/|t| apart), so
    low-index zones carry the most directions and the weight is 1/|t|.
    """
    A = np.asarray(cell_basis, dtype=float)
    rng = np.arange(-max_index, max_index + 1)
    u, v, w = np.meshgrid(rng, rng, rng, indexing="ij")
    uvw = np.stack([u.ravel(), v.ravel(), w.ravel()], axis=1)
    uvw = uvw[np.any(uvw != 0, axis=1)]
    gcd = np.gcd.reduce(np.abs(uvw), axis=1)
    uvw = uvw // gcd[:, None]
    lead = np.where(
        uvw[:, 0] != 0, uvw[:, 0], np.where(uvw[:, 1] != 0, uvw[:, 1], uvw[:, 2])
    )
    uvw = uvw * np.where(lead < 0, -1, 1)[:, None]
    uvw = np.unique(uvw, axis=0)
    t = uvw @ A
    tn = np.linalg.norm(t, axis=1)
    return t / tn[:, None], 1.0 / tn


# ---------------------------------------------------------------------------
# spherical harmonics
# ---------------------------------------------------------------------------


def _legendre_norm(L, x):
    """Fully normalized associated Legendre Pbar[l, m, :] for m >= 0.

    Pbar includes sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!) and the Condon-Shortley
    phase, so Y_lm = Pbar_lm(cos theta) e^{i m phi}.  Standard forward
    recursions, stable to l of a few thousand.
    """
    x = np.asarray(x, dtype=float)
    s = np.sqrt(np.maximum(1.0 - x * x, 0.0))
    P = np.zeros((L + 1, L + 1) + x.shape)
    P[0, 0] = np.sqrt(1.0 / (4.0 * np.pi))
    for m in range(L):
        P[m + 1, m + 1] = -np.sqrt((2.0 * m + 3.0) / (2.0 * m + 2.0)) * s * P[m, m]
        P[m + 1, m] = np.sqrt(2.0 * m + 3.0) * x * P[m, m]
    for m in range(L + 1):
        for l_ in range(m + 2, L + 1):
            a = np.sqrt((4.0 * l_**2 - 1.0) / (l_**2 - m**2))
            b = np.sqrt(((l_ - 1.0) ** 2 - m**2) / (4.0 * (l_ - 1.0) ** 2 - 1.0))
            P[l_, m] = a * (x * P[l_ - 1, m] - b * P[l_ - 2, m])
    return P


def sph_harm_all(L, dirs):
    """Y[l, m + L, j] for all l <= L, |m| <= l, at unit vectors dirs [N, 3]."""
    d = np.asarray(dirs, dtype=float)
    ct = np.clip(d[:, 2], -1.0, 1.0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    P = _legendre_norm(L, ct)  # [l, m>=0, j]
    Y = np.zeros((L + 1, 2 * L + 1, len(d)), dtype=complex)
    for m in range(L + 1):
        e = np.exp(1j * m * phi)
        Y[m:, L + m] = P[m:, m] * e
        if m:
            Y[m:, L - m] = (-1.0) ** m * np.conj(Y[m:, L + m])
    return Y


def _smoothing(L, sigma):
    """Per-degree Gaussian smoothing exp(-l(l+1) sigma^2 / 2).  [1]"""
    ell = np.arange(L + 1, dtype=float)
    return np.exp(-ell * (ell + 1.0) * sigma * sigma / 2.0)


def project_points(dirs, weights, L, sigma, even_only=True):
    """SH coefficients of a sum of smoothed deltas.  f[l, m + L] complex.

    f_lm = sum_j w_j conj(Y_lm(dir_j)) s_l -- a direct nonuniform sum; no
    spherical grid.  even_only drops odd degrees (antipodal symmetry).
    """
    w = np.ones(len(dirs)) if weights is None else np.asarray(weights, dtype=float)
    Y = sph_harm_all(L, dirs)
    f = np.einsum("lmj,j->lm", np.conj(Y), w)
    f *= _smoothing(L, sigma)[:, None]
    if even_only:
        f[1::2] = 0.0
    return f


def project_counts(pixel_dirs, excess, L, sigma, even_only=True, chunk=8192):
    """SH coefficients of a raw count map -- no peak finding required.

    The data function is the *excess count* density on the sphere: each
    detector pixel contributes its lab direction (detector.pixel_to_lab,
    normalized after subtracting the sample position) weighted by
    ``excess = y - U``, its counts above the expected background.  The
    correlogram then computes C(R) = sum_px (y - U)_px [Lambda(R) g](Ghat_px)
    -- the matched filter of the raw counts against the predicted direction
    density, the same statistic the peak path uses with every photon voting
    instead of every found peak.  Subtracting U (the rate map of
    subhkl.search.sparse_rbf.compute_rate_batch, in practice) is what makes
    the null mean zero *per pixel*, so the detector's partial sky coverage
    -- which is anisotropic and would otherwise correlate with the rotated
    model -- cancels in expectation rather than biasing the search.

    Because no threshold intervenes, this works below the finder's own
    detection floor: peaks individually far under the matched-filter
    F_min = z sqrt(N_bg) still contribute their few photons coherently, and
    the orientation aggregates all of them (measured in
    test_orientation_from_raw_counts_below_the_finder_floor).  The intended
    use is bootstrap: orientation first, then the predicted directions hand
    the image-domain finder a positional prior.

    Chunked direct sum; memory is O(L^2 chunk) and cost O(L^2 N_pixels)
    (~seconds for 10^5 pixels at L = 64 in numpy -- this loop is the
    module's first candidate for a jax port, being one fixed-geometry
    matvec per frame once the basis is cached).  Negative weights are fine.
    """
    d = np.asarray(pixel_dirs, dtype=float)
    w = np.asarray(excess, dtype=float)
    f = np.zeros((L + 1, 2 * L + 1), dtype=complex)
    for start in range(0, len(d), int(chunk)):
        dd = d[start : start + int(chunk)]
        ww = w[start : start + int(chunk)]
        ct = np.clip(dd[:, 2], -1.0, 1.0)
        phi = np.arctan2(dd[:, 1], dd[:, 0])
        P = _legendre_norm(L, ct)  # [l, m>=0, j]
        E = np.exp(-1j * np.outer(np.arange(L + 1), phi))  # [m, j]
        # f_lm = sum_j w_j Pbar_lm e^{-i m phi_j}; negative m by symmetry
        block = np.einsum("lmj,mj,j->lm", P, E, ww)
        f[:, L:] += block
    m = np.arange(1, L + 1)
    f[:, L - m] = (-1.0) ** m * np.conj(f[:, L + m])
    f *= _smoothing(L, sigma)[:, None]
    if even_only:
        f[1::2] = 0.0
    return f


def _legendre_p0(L):
    """P_l(0): (-1)^{l/2} (l-1)!! / l!! for even l, 0 for odd.  [1]"""
    p = np.zeros(L + 1)
    p[0] = 1.0
    for l_ in range(2, L + 1, 2):
        p[l_] = -p[l_ - 2] * (l_ - 1.0) / l_
    return p


def project_rings(axes, weights, L, sigma):
    """SH coefficients of a sum of smoothed great circles (zones).

    Funk--Hecke: a unit-mass great circle about axis n has coefficients
    2 pi P_l(0) conj(Y_lm(n)), so a ring is represented by its axis alone --
    rank one per degree, the same cost as a point.  Odd degrees vanish
    identically (P_l(0) = 0), so the antipodal projection is automatic.
    """
    w = np.ones(len(axes)) if weights is None else np.asarray(weights, dtype=float)
    Y = sph_harm_all(L, axes)
    f = np.einsum("lmj,j->lm", np.conj(Y), w)
    f *= (2.0 * np.pi * _legendre_p0(L) * _smoothing(L, sigma))[:, None]
    return f


# ---------------------------------------------------------------------------
# Wigner d and SO(3) machinery
# ---------------------------------------------------------------------------


def _factln(n):
    """ln n! lookup table 0..n.  [1]"""
    return gammaln(np.arange(n + 1) + 1.0)


def wigner_d_matrix(L, beta):
    """d[l, m + L, n + L] = d^l_{m n}(beta) for all l <= L, float64.

    Jacobi-polynomial form (one three-term recursion per (m, n), all l at
    once): with mx = max(|m|, |n|), k = l - mx, and per-(m, n) constants
    (a, b, lam) from the standard case table,

        d^l_{m n} = (-1)^lam sqrt(binom(2l-k, k+a) / binom(k+b, b))
                    (sin b/2)^a (cos b/2)^b P_k^{(a,b)}(cos beta).

    Entries with mx > l are zero.  Validated against the explicit factorial
    sum and closed forms in tests/test_spherical.py.

    ``beta`` may be a scalar (returns (L+1, M, M)) or a 1D array (returns
    (L+1, M, M, n_beta)): the correlogram's profile showed 45% of a run
    inside per-slice rebuilds of this tensor -- ~100 small-array numpy ops
    per beta, overhead-dominated -- and batching the trailing beta axis
    turns them into a few hundred large-array ops for the whole grid.
    """
    beta_arr = np.atleast_1d(np.asarray(beta, dtype=float))
    scalar = np.ndim(beta) == 0
    nb = beta_arr.size

    M = 2 * L + 1
    m = np.arange(-L, L + 1)
    mm, nn = np.meshgrid(m, m, indexing="ij")  # first index m (row), second n
    mx = np.maximum(np.abs(mm), np.abs(nn))

    # case table for k = l - mx: which of (l+n, l-n, l+m, l-m) is smallest
    # decides (a, lam); b = 2 mx - a always.
    #   k = l + n  (n = -mx): a = m - n, lam = m - n
    #   k = l - n  (n = +mx): a = n - m, lam = 0
    #   k = l + m  (m = -mx): a = n - m, lam = 0
    #   k = l - m  (m = +mx): a = m - n, lam = m - n
    a = np.where(
        np.abs(nn) >= np.abs(mm),
        np.where(nn < 0, mm - nn, nn - mm),
        np.where(mm < 0, nn - mm, mm - nn),
    )
    lam = np.where(
        np.abs(nn) >= np.abs(mm),
        np.where(nn < 0, mm - nn, 0),
        np.where(mm < 0, 0, mm - nn),
    )
    b = 2 * mx - a

    x = np.cos(beta_arr)[None, None, :]  # [1, 1, nb]
    sh, ch = np.sin(beta_arr / 2.0), np.cos(beta_arr / 2.0)
    af, bf = a.astype(float)[..., None], b.astype(float)[..., None]

    # P_k^{(a,b)}(x) for k = 0..L, elementwise (a, b), batched over beta:
    # standard three-term recursion of the Jacobi polynomials, forward
    # stable on [-1, 1].
    P = np.zeros((L + 1, M, M, nb))
    P[0] = 1.0
    if L >= 1:
        P[1] = 0.5 * (af - bf) + (1.0 + 0.5 * (af + bf)) * x
    for k in range(2, L + 1):
        n2 = 2.0 * k + af + bf
        c1 = 2.0 * k * (k + af + bf) * (n2 - 2.0)
        c2 = (n2 - 1.0) * (n2 * (n2 - 2.0) * x + af * af - bf * bf)
        c3 = 2.0 * (k + af - 1.0) * (k + bf - 1.0) * n2
        P[k] = (c2 * P[k - 1] - c3 * P[k - 2]) / c1

    fl = _factln(2 * L + max(2 * L, 1))
    sign = np.where(lam % 2 == 0, 1.0, -1.0)
    # angular factors; 0^0 = 1 is the correct limit here
    with np.errstate(divide="ignore"):
        log_sh = np.log(np.maximum(sh, 1e-300))[None, None, :]
        log_ch = np.log(np.maximum(ch, 1e-300))[None, None, :]
        ang = np.where(
            (a[..., None] == 0) & (sh[None, None, :] == 0.0),
            1.0,
            np.exp(log_sh * af),
        ) * np.where(
            (b[..., None] == 0) & (ch[None, None, :] == 0.0),
            1.0,
            np.exp(log_ch * bf),
        )

    d = np.zeros((L + 1, M, M, nb))
    for l_ in range(L + 1):
        valid = mx <= l_
        k = l_ - mx
        # ln binom(2l - k, k + a) - ln binom(k + b, b), all indices >= 0
        top = (
            fl[l_ + mx] - fl[np.maximum(k + a, 0)] - fl[np.maximum(l_ + mx - k - a, 0)]
        )
        bot = fl[np.maximum(k + b, 0)] - fl[np.maximum(k, 0)] - fl[b]
        pref = np.exp(0.5 * (top - bot))
        Pk = np.take_along_axis(P, np.maximum(k, 0)[None, ..., None], axis=0)[0]
        d[l_] = np.where(valid[..., None], (sign * pref)[..., None] * ang * Pk, 0.0)
    return d[..., 0] if scalar else d


def rot_zyz(alpha, beta, gamma):
    """Active rotation R = Rz(alpha) Ry(beta) Rz(gamma).  [3, 3]"""

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def ry(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    return rz(alpha) @ ry(beta) @ rz(gamma)


def euler_zyz(R):
    """Euler angles (alpha, beta, gamma) with R = Rz(a) Ry(b) Rz(g).  [rad]"""
    R = np.asarray(R, dtype=float)
    beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    if np.abs(R[2, 2]) > 1.0 - 1e-12:
        # gimbal: only alpha + gamma (or difference) is defined
        return np.arctan2(R[1, 0], R[0, 0]), beta, 0.0
    alpha = np.arctan2(R[1, 2], R[0, 2])
    gamma = np.arctan2(R[2, 1], -R[2, 0])
    return alpha, beta, gamma


def rotate_coeffs(f, R):
    """Coefficients of the actively rotated function Lambda(R) f = f(R^-1 x).

    (Lambda(R) f)_{l m} = sum_n D^l_{m n}(R) f_{l n} with
    D^l_{m n} = e^{-i m alpha} d^l_{m n}(beta) e^{-i n gamma}.  Convention
    pinned by test_rotation_identity (coefficients of rotated points equal
    the rotated coefficients).
    """
    L = f.shape[0] - 1
    alpha, beta, gamma = euler_zyz(R)
    d = wigner_d_matrix(L, beta)
    m = np.arange(-L, L + 1)
    ea = np.exp(-1j * m * alpha)
    eg = np.exp(-1j * m * gamma)
    return np.einsum("m,lmn,n,ln->lm", ea, d, eg, f)


def so3_inner(f, g, R):
    """C(R) = <f, Lambda(R) g> = sum_l f^l dagger D^l(R) g^l.  [real]"""
    rf = rotate_coeffs(g, R)
    return float(np.real(np.sum(np.conj(f) * rf)))


def correlogram(f, g, n_beta=None, backend="jax"):
    """C(alpha, beta, gamma) = <f, Lambda(R) g> on an Euler-angle grid.

    For each beta the coupling S_{m n} = sum_l conj(f^l_m) d^l_{m n} g^l_n
    is a single contraction over degrees (the per-degree coupling is the
    rank-one outer product f^l (g^l)^dagger), and the (alpha, gamma)
    dependence is a 2D Fourier sum.

    backend="jax" (default) runs a fused float32 kernel: profiling showed
    the numpy path memory-bound on materializing the full Wigner stack
    ((n_beta, L+1, M, M) = 2.8 GB at L = 96), so the kernel never builds
    it -- the coupling is reindexed by recursion depth,
    coefK[k, m, n] = coef[l = k + mx(m, n), m, n], and accumulated inside
    the Jacobi recursion with two rolling P rows: peak memory O(M^2
    n_beta) ~ 90 MB, every beta in one fused pass, on whatever device jax
    holds.  backend="numpy" is the float64 reference the exactness tests
    pin (agreement with direct kernel sums at 1e-10); the jax path is
    tested against it at float32 tolerance.

    Returns (C [n_a, n_b, n_g] real, alphas, betas, gammas).
    """
    L = f.shape[0] - 1
    n_beta = L + 1 if n_beta is None else int(n_beta)
    n_ang = 2 * L + 2
    alphas = 2.0 * np.pi * np.arange(n_ang) / n_ang
    gammas = alphas
    betas = np.pi * (np.arange(n_beta) + 0.5) / n_beta

    m = np.arange(-L, L + 1)
    Ea = np.exp(-1j * np.outer(m, alphas))  # [M, n_ang]

    if backend == "jax":
        C = _correlogram_kernel_jax(np.asarray(f), np.asarray(g), betas, Ea)
        return np.asarray(C, dtype=float), alphas, betas, gammas

    C = np.empty((n_ang, n_beta, n_ang))
    # batched Wigner build, chunked so the (L+1, M, M, chunk) tensor stays
    # a few hundred MB (29 MB per beta at L = 96)
    chunk = max(1, int(2.5e8 // (8 * (L + 1) * (2 * L + 1) ** 2)))
    for t0 in range(0, n_beta, chunk):
        bs = betas[t0 : t0 + chunk]
        d = wigner_d_matrix(L, bs)  # [L+1, M, M, nb]
        S = np.einsum("lm,lmnb,ln->bmn", np.conj(f), d, g, optimize=True)
        for j in range(len(bs)):
            C[:, t0 + j, :] = np.real(Ea.T @ S[j] @ Ea)
    return C, alphas, betas, gammas


def _wigner_case_tables(L):
    """Static (m, n) tables of the Jacobi form: mx, a, b, and the
    sign * prefactor(l) stack.  Pure geometry, numpy float64."""
    M = 2 * L + 1
    m = np.arange(-L, L + 1)
    mm, nn = np.meshgrid(m, m, indexing="ij")
    mx = np.maximum(np.abs(mm), np.abs(nn))
    a = np.where(
        np.abs(nn) >= np.abs(mm),
        np.where(nn < 0, mm - nn, nn - mm),
        np.where(mm < 0, nn - mm, mm - nn),
    )
    lam = np.where(
        np.abs(nn) >= np.abs(mm),
        np.where(nn < 0, mm - nn, 0),
        np.where(mm < 0, 0, mm - nn),
    )
    b = 2 * mx - a
    fl = _factln(4 * L + 1)
    sign = np.where(lam % 2 == 0, 1.0, -1.0)
    pref = np.zeros((L + 1, M, M))
    for l_ in range(L + 1):
        valid = mx <= l_
        k = l_ - mx
        top = (
            fl[l_ + mx] - fl[np.maximum(k + a, 0)] - fl[np.maximum(l_ + mx - k - a, 0)]
        )
        bot = fl[np.maximum(k + b, 0)] - fl[np.maximum(k, 0)] - fl[b]
        pref[l_] = np.where(valid, sign * np.exp(0.5 * (top - bot)), 0.0)
    return mx, a, b, pref


def _correlogram_kernel_jax(f, g, betas, Ea):
    """The fused correlogram: coupling accumulated inside the recursion, no
    Wigner tensor ever materialized.

    Numerical form matters here: the Jacobi factorization d = pref * ang *
    P_k balances a prefactor that reaches ~1e28 at L = 96 against an
    equally extreme angular factor and polynomial -- fine in float64,
    overflow at L ~ 40 in float32.  The kernel therefore recurs directly on
    the BOUNDED quantity D_k = pref_k ang P_k (the d-matrix entries
    themselves, |d| <= 1): ang cancels from the linear recursion because it
    is k-independent, and pref enters only through the polynomially bounded
    ratios rho = pref_k / pref_{k-1}, precomputed in float64 and cast.
    Everything the kernel touches is O(1).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    L = f.shape[0] - 1
    mx, a, b, pref = _wigner_case_tables(L)
    # coupling gathered into recursion-depth order: step k of the recursion
    # computes degree l = k + mx(m, n) for every (m, n) at once
    coef = np.conj(f)[:, :, None] * g[:, None, :]
    l_idx = np.minimum(np.arange(L + 1)[:, None, None] + mx[None], L)
    in_band = (np.arange(L + 1)[:, None, None] + mx[None]) <= L
    coefK = np.where(in_band, np.take_along_axis(coef, l_idx, axis=0), 0.0)
    prefK = np.where(in_band, np.take_along_axis(pref, l_idx, axis=0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho1 = np.where(prefK != 0.0, prefK / np.roll(prefK, 1, axis=0), 0.0)
        rho2 = np.where(prefK != 0.0, prefK / np.roll(prefK, 2, axis=0), 0.0)
    rho1[~np.isfinite(rho1)] = 0.0
    rho2[~np.isfinite(rho2)] = 0.0

    sh, ch = np.sin(betas / 2.0), np.cos(betas / 2.0)
    with np.errstate(divide="ignore"):
        ang = np.exp(
            np.log(np.maximum(sh, 1e-300))[None, None, :] * a[..., None]
            + np.log(np.maximum(ch, 1e-300))[None, None, :] * b[..., None]
        )  # float64: exact where float32 would over/underflow
    x64 = np.cos(betas)
    D0 = prefK[0][..., None] * ang  # = d at k = 0 (P_0 = 1), O(1)
    P1_64 = 0.5 * (a - b)[..., None] + (1.0 + 0.5 * (a + b))[..., None] * x64
    D1 = prefK[1][..., None] * ang * P1_64

    af = jnp.asarray(a, dtype=jnp.float32)[..., None]
    bf = jnp.asarray(b, dtype=jnp.float32)[..., None]
    coefK_j = jnp.asarray(coefK, dtype=jnp.complex64)
    rho1_j = jnp.asarray(rho1, dtype=jnp.float32)
    rho2_j = jnp.asarray(rho2, dtype=jnp.float32)
    x = jnp.asarray(x64, dtype=jnp.float32)[None, None, :]
    D0_j = jnp.asarray(D0, dtype=jnp.float32)
    D1_j = jnp.asarray(D1, dtype=jnp.float32)
    Ea_j = jnp.asarray(Ea, dtype=jnp.complex64)

    @jax.jit
    def kernel(coefK_j, rho1_j, rho2_j, af, bf, x, D0_j, D1_j, Ea_j):
        S = coefK_j[0][..., None] * D0_j.astype(jnp.complex64) + coefK_j[1][
            ..., None
        ] * D1_j.astype(jnp.complex64)

        def body(k, state):
            Dm2, Dm1, S = state
            kf = jnp.asarray(k, dtype=jnp.float32)
            n2 = 2.0 * kf + af + bf
            c1 = 2.0 * kf * (kf + af + bf) * (n2 - 2.0)
            c2 = (n2 - 1.0) * (n2 * (n2 - 2.0) * x + af * af - bf * bf)
            c3 = 2.0 * (kf + af - 1.0) * (kf + bf - 1.0) * n2
            Dk = (
                c2 * rho1_j[k][..., None] * Dm1 - c3 * rho2_j[k][..., None] * Dm2
            ) / c1
            S = S + coefK_j[k][..., None] * Dk.astype(jnp.complex64)
            return (Dm1, Dk, S)

        _, _, S = lax.fori_loop(2, coefK_j.shape[0], body, (D0_j, D1_j, S))
        return jnp.real(jnp.einsum("ma,mnb,nc->abc", Ea_j, S, Ea_j))

    return kernel(coefK_j, rho1_j, rho2_j, af, bf, x, D0_j, D1_j, Ea_j)


# ---------------------------------------------------------------------------
# peak extraction, refinement, admission
# ---------------------------------------------------------------------------


def _quat_angle(R1, R2):
    """Rotation angle between two orientations.  [rad]"""
    tr = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))


def top_orientations(C, alphas, betas, gammas, n=4, min_sep_deg=10.0):
    """Local maxima of the correlogram as rotations, best first.

    Wrap-aware local-maximum test in alpha and gamma, then greedy selection
    with a minimum quaternion separation so one broad peak is not returned
    n times.  Returns [(R, value), ...].
    """
    up = np.roll(C, 1, axis=0) < C
    dn = np.roll(C, -1, axis=0) <= C
    lf = np.roll(C, 1, axis=2) < C
    rt = np.roll(C, -1, axis=2) <= C
    bu = np.roll(C, 1, axis=1) < C
    bd = np.roll(C, -1, axis=1) <= C
    bu[:, 0, :] = True
    bd[:, -1, :] = True
    is_max = up & dn & lf & rt & bu & bd
    idx = np.argwhere(is_max)
    vals = C[is_max]
    order = np.argsort(vals)[::-1]
    out = []
    for o in order:
        a, b, g = idx[o]
        R = rot_zyz(alphas[a], betas[b], gammas[g])
        if any(_quat_angle(R, R0) < np.deg2rad(min_sep_deg) for R0, _ in out):
            continue
        out.append((R, float(vals[o])))
        if len(out) >= n:
            break
    return out


def refine_wahba(R, data_dirs, model_dirs, match_deg=3.0, n_iter=3):
    """Polish an orientation by matched-pair alignment (Wahba, via SVD).

    Assign each measured direction to the nearest rotated model *line*
    (|dot|, both directions of the +- pair), keep matches within match_deg,
    and solve for the rotation maximizing sum |Ghat . R Hhat| in closed
    form.  Iterated; assignment is refreshed each round.  Returns
    (R, n_matched).  Bandlimit-free: precision is set by the data, not by L.
    """
    D = np.asarray(data_dirs, dtype=float)
    H = np.asarray(model_dirs, dtype=float)
    cut = np.cos(np.deg2rad(match_deg))
    n_matched = 0
    for _ in range(n_iter):
        pred = H @ R.T  # [N, 3]
        dots = D @ pred.T  # [M, N]
        j = np.argmax(np.abs(dots), axis=1)
        best = np.abs(dots[np.arange(len(D)), j])
        keep = best >= cut
        n_matched = int(np.sum(keep))
        if n_matched < 2:
            return R, n_matched
        sgn = np.sign(dots[np.arange(len(D)), j])[keep]
        Bm = (D[keep] * sgn[:, None]).T @ H[j[keep]]
        U, _, Vt = np.linalg.svd(Bm)
        S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R = U @ S @ Vt
    return R, n_matched


def _rodrigues(v):
    """exp([v]x): rotation by |v| about v/|v|.  [3, 3]"""
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def refine_local(f, g, R, span_deg=3.0):
    """Grid-free polish by maximizing C(R) itself (Nelder--Mead on the
    rotation vector).  Works for any model -- rings included, where Wahba
    has no point pairs to match -- at the cost of being bandlimited: the
    objective is the L-bandlimited correlation, so precision is ~kernel
    width / sqrt(matches), not the Wahba floor."""
    from scipy.optimize import minimize

    def neg(v):
        return -so3_inner(f, g, _rodrigues(v) @ R)

    res = minimize(
        neg,
        np.zeros(3),
        method="Nelder-Mead",
        options={"maxiter": 60, "xatol": np.deg2rad(0.02), "fatol": 1e-10},
        bounds=[(-np.deg2rad(span_deg), np.deg2rad(span_deg))] * 3,
    )
    return _rodrigues(res.x) @ R


def null_zscore(C, value, trim=0.01):
    """How far a correlogram value stands above the no-orientation null.

    The null is the correlogram itself away from its peaks: trim the top
    fraction, use the remaining mean and standard deviation.  A correct
    orientation aggregates hundreds of matched directions, so its z is
    large by construction and needs no tuned threshold -- this is the
    quantitative form of the implicit-penalty argument for this domain.
    """
    flat = np.sort(C.ravel())
    body = flat[: int(len(flat) * (1.0 - trim))]
    mu, sd = float(np.mean(body)), float(np.std(body))
    return (value - mu) / max(sd, 1e-12)


def nonneg_lasso(f, g, rotations, lam=0.0, n_iter=300):
    """Sparse nonnegative coefficients over refined orientation candidates.

    Solves min_{c >= 0} 1/2 ||f - sum_r c_r Lambda(R_r) g||^2 + lam sum c
    in the SH domain by projected gradient.  With lam = 0 the nonnegativity
    and the candidates' mutual coherence (the Gram below, generically small
    between distinct orientations) already produce exact zeros for
    candidates the data does not support -- the L1-on-coefficients stage of
    the sparse recovery, run on the few survivors rather than all of SO(3).
    Returns c [n_candidates].
    """
    n = len(rotations)
    G = np.empty((n, n))
    b = np.empty(n)
    for r, Rr in enumerate(rotations):
        b[r] = so3_inner(f, g, Rr)
        for s_, Rs in enumerate(rotations):
            if s_ < r:
                G[r, s_] = G[s_, r]
            else:
                G[r, s_] = so3_inner(g, g, Rr.T @ Rs)
    step = 1.0 / (np.linalg.norm(G, 2) + 1e-12)
    c = np.zeros(n)
    for _ in range(n_iter):
        c = np.maximum(0.0, c - step * (G @ c - b + lam))
    return c


def find_orientations(
    data_dirs,
    model_dirs=None,
    model_weights=None,
    ring_axes=None,
    ring_weights=None,
    data_weights=None,
    kernel_deg=1.0,
    L=None,
    n_candidates=4,
    min_sep_deg=10.0,
    refine=True,
    lam=0.0,
):
    """Find crystal orientations from measured Q directions.

    data_dirs: measured Ghat unit vectors (use ghat_from_kf).  Provide
    either model_dirs (+ optional weights; see lattice_directions) or
    ring_axes (+ weights; see zone_axes) as the model.  kernel_deg is the
    angular width [deg] given to every direction (mosaic + measurement);
    L defaults to the bandwidth at which that kernel has decayed to ~1%,
    capped at 96 -- candidates only need to be separated here, refinement
    restores full precision off-grid.

    Returns a list of dicts {R, score, z, n_matched, c}, best first.
    """
    # kernel_deg is a FWHM; sigma = FWHM / sqrt(8 ln 2)
    sigma = np.deg2rad(kernel_deg) / np.sqrt(8.0 * np.log(2.0))
    if L is None:
        L = int(min(max(np.ceil(3.0 / sigma), 16), 96))
    f = project_points(np.asarray(data_dirs, float), data_weights, L, sigma)
    if (model_dirs is None) == (ring_axes is None):
        raise ValueError("provide exactly one of model_dirs or ring_axes")
    if model_dirs is not None:
        g = project_points(np.asarray(model_dirs, float), model_weights, L, sigma)
    else:
        g = project_rings(np.asarray(ring_axes, float), ring_weights, L, sigma)

    C, al, be, ga = correlogram(f, g)
    cands = top_orientations(C, al, be, ga, n=n_candidates, min_sep_deg=min_sep_deg)
    out = []
    rotations = []
    gnorm = np.sqrt(so3_inner(g, g, np.eye(3)))
    for R, val in cands:
        n_matched = 0
        if refine and model_dirs is not None:
            R, n_matched = refine_wahba(R, data_dirs, model_dirs)
            val = so3_inner(f, g, R)
        elif refine:
            R = refine_local(f, g, R)
            val = so3_inner(f, g, R)
        # Mutual coherence against the candidates already kept: two rotations
        # equivalent under the lattice's own point group carry the *same*
        # rotated model (Lambda_S g = g for a symmetry S), coherence 1, and
        # the sparse stage cannot apportion mass between identical columns.
        # Orientations are only ever defined up to the Laue group, so keep
        # one representative.  Distinct orientations are generically nearly
        # orthogonal here -- that near-diagonal Gram over the zero-overlap
        # region is what makes the recovery well posed.
        mu = max(
            (abs(so3_inner(g, g, R.T @ R2)) / gnorm**2 for R2 in rotations),
            default=0.0,
        )
        if mu > 0.99:
            continue
        out.append(
            {
                "R": R,
                "score": val,
                "z": null_zscore(C, val),
                "n_matched": n_matched,
                "coherence": mu,
            }
        )
        rotations.append(R)
    if rotations:
        c = nonneg_lasso(f, g, rotations, lam=lam)
        for rec, cr in zip(out, c):
            rec["c"] = float(cr)
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# real detector panels
# ---------------------------------------------------------------------------


def panel_directions(
    detector, rows=None, cols=None, sample_offset=None, ki=(0.0, 0.0, 1.0)
):
    """Measured Q directions for detector pixels, via the instrument geometry.

    ``detector`` is a subhkl.instrument.detector.Detector (or a config dict
    for one).  With rows/cols omitted, returns Ghat for *every* pixel of the
    panel, flattened in the same (row, col) C-order as a counts frame
    ``counts[row, col]`` -- the raw-counts path: pool
    ``panel_directions(det)`` and ``(counts - rate_map).ravel()`` across
    banks and hand them to project_counts.  With explicit rows/cols (e.g. a
    finder peak table's i/j), returns Ghat per peak -- the peaks path.

    Multi-bank is concatenation: directions from every panel live in the
    one lab frame already, so a global search across panels is
    ``np.vstack([panel_directions(d) for d in dets])``.  For a rotated
    sample (goniometer), rotate the returned directions back to the sample
    frame with the run's R and pool across runs the same way.
    """
    from subhkl.instrument.detector import Detector

    det = detector if hasattr(detector, "pixel_to_lab") else Detector(detector)
    if (rows is None) != (cols is None):
        raise ValueError("give both rows and cols, or neither")
    if rows is None:
        # full frame: counts[row, col] <-> pixel_to_lab(row, col)
        n_rows, n_cols = det.n, det.m
        rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
        rows, cols = rr.ravel(), cc.ravel()
    xyz = det.pixel_to_lab(np.asarray(rows), np.asarray(cols))
    if sample_offset is not None:
        xyz = xyz - np.asarray(sample_offset, dtype=float)
    kf = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    return ghat_from_kf(kf, ki=ki)


# ---------------------------------------------------------------------------
# matching-free refinement (jax)
# ---------------------------------------------------------------------------
#
# Refinement needs no Wigner machinery and no peak list.  The correlation
#
#     C(R, B) = Re sum_lm conj(f_lm) g_lm(R, B),
#     g_lm    = sum_i w_i conj(Y_lm(R Hhat_i(B))) s_l,
#     Hhat_i  = B hkl_i / |B hkl_i|,
#
# is differentiable end to end -- through the associated-Legendre recursion,
# the normalization, and the rotation -- so orientation and cell SHAPE refine
# jointly by gradient ascent of the same objective the search maximized,
# with the data side f frozen (peaks or raw counts, it no longer matters).
# There is no assignment step: the kernel overlap IS the soft assignment,
# integrated exactly.  Cell SCALE is structurally invisible here (directions
# are homogeneous of degree zero in B) and must come from the wavelength/TOF
# axis downstream; the symmetric-traceless parameterization below makes that
# explicit instead of letting scale wander.


def _sph_coeffs_jax(dirs, weights, L, sigma):
    """g_lm as a jax array [(L+1), (2L+1)] complex; differentiable in dirs.

    One lax.fori_loop over degrees carrying two Legendre rows -- linear
    compile time in L (the unrolled per-(l, m) graph took ~90 s to trace at
    L = 48; this compiles in seconds and evaluates in milliseconds)."""
    import jax.numpy as jnp
    from jax import lax

    ct = jnp.clip(dirs[:, 2], -1.0, 1.0)
    st = jnp.sqrt(jnp.maximum(1.0 - ct * ct, 1e-20))
    phi = jnp.arctan2(dirs[:, 1], dirs[:, 0])
    n = dirs.shape[0]
    m_idx = jnp.arange(L + 1, dtype=jnp.float32)
    ell = np.arange(L + 1, dtype=float)
    s_l = jnp.asarray(np.exp(-ell * (ell + 1.0) * sigma * sigma / 2.0))
    # conj(Y_lm) = Pbar_lm e^{-i m phi}; E[m, j] = w_j e^{-i m phi_j}
    E = weights[None, :] * jnp.exp(-1j * jnp.outer(m_idx, phi))
    signs = jnp.asarray((-1.0) ** np.arange(1, L + 1))

    def coeffs_from_row(P_row, l_):
        cpos = jnp.einsum("mj,mj->m", P_row * 1.0, jnp.real(E)) + 1j * jnp.einsum(
            "mj,mj->m", P_row * 1.0, jnp.imag(E)
        )
        cneg = signs * jnp.conj(cpos[1:])
        return jnp.concatenate([jnp.flip(cneg), cpos]) * s_l[l_]

    p00 = jnp.full((n,), float(np.sqrt(1.0 / (4.0 * np.pi))))
    P0 = jnp.zeros((L + 1, n)).at[0].set(p00)  # row l = 0
    g = jnp.zeros((L + 1, 2 * L + 1), dtype=jnp.complex64)
    g = g.at[0].set(coeffs_from_row(P0, 0))
    if L == 0:
        return g
    # row l = 1: m = 0 and the diagonal m = 1
    P1 = (
        jnp.zeros((L + 1, n))
        .at[0]
        .set(np.sqrt(3.0) * ct * p00)
        .at[1]
        .set(-np.sqrt(3.0 / 2.0) * st * p00)
    )
    g = g.at[1].set(coeffs_from_row(P1, 1))

    def body(l_, state):
        Pm2, Pm1, g = state
        lf = jnp.asarray(l_, dtype=jnp.float32)
        # general recursion, valid for m <= l - 2 (masked elsewhere)
        a = jnp.sqrt(
            jnp.abs(
                (4.0 * lf**2 - 1.0)
                / jnp.where(lf**2 - m_idx**2 != 0.0, lf**2 - m_idx**2, 1.0)
            )
        )
        b = jnp.sqrt(
            jnp.abs(((lf - 1.0) ** 2 - m_idx**2) / (4.0 * (lf - 1.0) ** 2 - 1.0))
        )
        row = a[:, None] * (ct[None, :] * Pm1 - b[:, None] * Pm2)
        # m = l - 1 (subdiagonal) and m = l (diagonal), from row l-1's diagonal
        diag_prev = Pm1[l_ - 1]
        row = row.at[l_ - 1].set(jnp.sqrt(2.0 * lf + 1.0) * ct * diag_prev)
        row = row.at[l_].set(-jnp.sqrt((2.0 * lf + 1.0) / (2.0 * lf)) * st * diag_prev)
        # zero out m > l
        row = jnp.where((m_idx <= lf)[:, None], row, 0.0)
        g = g.at[l_].set(coeffs_from_row(row, l_))
        return (Pm1, row, g)

    _, _, g = lax.fori_loop(2, L + 1, body, (P0, P1, g))
    return g


def _rodrigues_jax(v):
    """exp([v]x), autodiff-safe at v = 0.

    The naive form has NaN gradients at the origin -- which is exactly
    where every refinement starts -- through 0/0 in sin(th)/th and the
    norm's own derivative.  The standard fix: Taylor branches near zero,
    with *safe inputs to the exact branch* so the untaken branch cannot
    poison the gradient through jnp.where."""
    import jax.numpy as jnp

    th2 = v @ v
    small = th2 < 1e-8
    th = jnp.sqrt(jnp.where(small, 1.0, th2))
    A = jnp.where(small, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    B = jnp.where(
        small, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / jnp.where(small, 1.0, th2)
    )
    K = jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return jnp.eye(3) + A * K + B * (K @ K)


def refine_matching_free(
    f,
    R0,
    hkl,
    B0,
    weights=None,
    kernel_deg=1.0,
    refine_cell=False,
    maxiter=60,
):
    """Refine orientation (and cell shape) against the correlation itself.

    Maximizes C(R, B) by L-BFGS with jax gradients, starting from the
    search's candidate R0 and the nominal B0.  Parameters: a rotation
    vector delta (R = exp([delta]x) R0) and, with refine_cell, a symmetric
    traceless 3x3 strain A (B = (I + A) B0): symmetric because the
    antisymmetric part is a rotation (already delta's job), traceless
    because pure scale does not move any direction and would be a flat
    direction of the objective.  Five shape parameters, exactly the cell's
    direction-visible degrees of freedom.

    Replaces peak-list refinement for everything the sphere can see: no
    finder, no indexing labels, no predicted-to-observed assignment --
    the kernel overlap is the soft assignment, integrated exactly.  Scale
    (and only scale) still needs the wavelength/TOF axis.

    Returns (R, B, C_value).
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    L = f.shape[0] - 1
    sigma = np.deg2rad(kernel_deg) / np.sqrt(8.0 * np.log(2.0))
    hkl_j = np.asarray(hkl, dtype=float)
    w = np.ones(len(hkl_j)) if weights is None else np.asarray(weights, float)
    fj = np.asarray(f, dtype=complex)

    def unpack(p):
        delta = p[:3]
        A = jnp.zeros((3, 3))
        if refine_cell:
            a = p[3:]
            A = jnp.array(
                [
                    [a[0], a[2], a[3]],
                    [a[2], a[1], a[4]],
                    [a[3], a[4], -a[0] - a[1]],
                ]
            )
        return delta, A

    def neg_corr(p):
        delta, A = unpack(p)
        R = _rodrigues_jax(delta) @ jnp.asarray(R0)
        B = (jnp.eye(3) + A) @ jnp.asarray(B0)
        g_vec = hkl_j @ B.T
        dirs = g_vec / jnp.linalg.norm(g_vec, axis=1, keepdims=True)
        g = _sph_coeffs_jax(dirs @ R.T, jnp.asarray(w), L, sigma)
        return -jnp.real(jnp.sum(jnp.conj(fj) * g))

    val_grad = jax.jit(jax.value_and_grad(neg_corr))

    def fun(p):
        v, g = val_grad(jnp.asarray(p, dtype=jnp.float32))
        return float(v), np.asarray(g, dtype=float)

    n = 8 if refine_cell else 3
    res = minimize(
        fun, np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": maxiter}
    )
    delta, A = unpack(res.x)
    R = np.asarray(_rodrigues_jax(jnp.asarray(delta))) @ np.asarray(R0)
    B = (np.eye(3) + np.asarray(A)) @ np.asarray(B0)
    return R, B, float(-res.fun)


def refine_instrument_matching_free(
    peaks,
    detector_nominal,
    hkl,
    B0,
    R0,
    modes=("independent",),
    det_bounds=None,
    gonio=None,
    weights=None,
    kernel_deg=1.0,
    ki=(0.0, 0.0, 1.0),
    sample_offset=None,
    refine_cell=False,
    maxiter=600,
    floor=0.01,
):
    """Joint matching-free refinement of orientation, cell shape, detector
    panels and goniometer offsets -- one likelihood, no peak assignment.

    The objective is the per-peak log-likelihood of the model direction
    density, a von Mises mixture with a uniform floor -- NOT the
    correlation.  The search stage maximizes <f, Lambda g> with the data
    side frozen, where the matched filter is the right detection statistic;
    but once the DATA side has free parameters, raw correlation rewards
    concentration -- geometry can bunch data directions onto whichever
    model kernels are nearest (measured on the two-bank synthetic: it
    preferred a wrong geometry at C = 42768 over the truth at 42344) --
    and normalized correlation is nearly flat (truth 0.2447 vs nominal
    0.2389) because the unobserved part of the model dominates the norms.
    The likelihood has neither disease: each datum rewards only its own
    kernel, unobserved model directions cost nothing, and the floor makes
    outliers a bounded penalty instead of a veto.  No spherical harmonics
    appear at all here -- the kernel sum is exact, so refinement carries
    no bandlimit bias either.

    The detector parameterization is *shared* with the peak-list path:
    subhkl.instrument.refinables.apply_detector_modes is the very code
    VectorizedObjective runs, so a bank translation means the same thing
    to both refinements, with the same modes and normalized-parameter
    convention (0.5 = nominal).  The goniometer follows sample_to_lab's
    composition (outermost first).  Identifiability of an axis offset is a
    theorem worth stating exactly, because the first guess is wrong both
    ways: an offset on axis k is pure gauge with the crystal orientation
    whenever every axis *inner* to k (closer to the sample) holds a
    constant angle across the pooled runs -- R_inner constant lets
    R_k(delta) commute through and fold into U exactly, so in particular
    the innermost axis offset is ALWAYS gauge, whatever the outer axes do
    (measured: a 0.6 deg innermost offset came back as 0.36 with the
    remaining 0.24 deg absorbed into R, at any outer span).  An offset is
    identifiable precisely when some inner axis varies across the pooled
    runs (measured: a 0.7 deg outer-axis offset recovered to 0.689 with
    the inner angle swept 0-60 deg).  Put only identifiable offsets in the
    refine mask; the gauge ones belong to U by convention.

    peaks: dict with det_idx (N,), u_off (N,) [m], v_off (N,) [m] --
    physical offsets from the panel center, the convention commands.py
    stores -- and optionally run_idx (N,) when gonio is given.
    detector_nominal: dict with centers (nb, 3), uhats, vhats [unit],
    widths, heights (nb,) [m].
    gonio: dict with axes (K, 3 or 4; direction multiplier honored),
    angles_deg (n_runs, K), refine_mask (K,) bool, bound_deg, optionally
    nominal_offsets_deg (K,).

    Returns dict with R, B, det_params (normalized, 0.5 = nominal),
    gonio_offsets_deg (full K vector), and the achieved mean
    log-likelihood per peak ("loglik").
    """
    import jax
    import jax.numpy as jnp

    from subhkl.instrument.refinables import (
        apply_detector_modes,
        forward_map_param,
        gonio_rotation_jax,
        peak_lab_xyz,
    )

    sigma = np.deg2rad(kernel_deg) / np.sqrt(8.0 * np.log(2.0))
    det_idx = np.asarray(peaks["det_idx"], dtype=int)
    u_off = jnp.asarray(np.asarray(peaks["u_off"], dtype=float))
    v_off = jnp.asarray(np.asarray(peaks["v_off"], dtype=float))
    n_peaks = len(det_idx)
    w_data = (
        jnp.ones(n_peaks)
        if weights is None
        else jnp.asarray(np.asarray(weights, float))
    )
    centers = jnp.asarray(np.asarray(detector_nominal["centers"], float))[None]
    uhats = jnp.asarray(np.asarray(detector_nominal["uhats"], float))[None]
    vhats = jnp.asarray(np.asarray(detector_nominal["vhats"], float))[None]
    widths = jnp.asarray(np.asarray(detector_nominal["widths"], float))[None]
    heights = jnp.asarray(np.asarray(detector_nominal["heights"], float))[None]
    n_banks = centers.shape[1]
    ki_hat = np.asarray(ki, float)
    ki_hat = jnp.asarray(ki_hat / np.linalg.norm(ki_hat))
    s_off = jnp.zeros(3) if sample_offset is None else jnp.asarray(sample_offset)

    if det_bounds is None:
        det_bounds = {
            "independent_trans": 0.01,  # [m]
            "independent_rot": np.deg2rad(1.0),  # [rad]
        }
    n_det = n_banks * 6 if "independent" in modes else 0
    param_slices = {"independent": slice(0, n_banks * 6)}

    if gonio is not None:
        axes = np.asarray(gonio["axes"], dtype=float)
        angles_deg = np.asarray(gonio["angles_deg"], dtype=float)
        refine_mask = np.asarray(gonio["refine_mask"], dtype=bool)
        gonio_bound = float(gonio.get("bound_deg", 2.0))
        nominal_off = np.asarray(
            gonio.get("nominal_offsets_deg", np.zeros(axes.shape[0])), dtype=float
        )
        run_idx = np.asarray(peaks["run_idx"], dtype=int)
        n_goff = int(refine_mask.sum())
    else:
        n_goff = 0

    hkl_j = np.asarray(hkl, dtype=float)
    w_model = jnp.ones(len(hkl_j))
    n_cell = 5 if refine_cell else 0

    # every parameter normalized to [0, 1] with 0.5 = nominal, the shared
    # forward-map convention -- and not only for consistency: mixed physical
    # units (radians, meters, degrees) give the optimizer gradient blocks
    # scaled apart by orders of magnitude, and the joint refinement then
    # stalls on whichever block is smallest.
    rot_bound = np.deg2rad(3.0)
    strain_bound = 0.03

    def unpack(p):
        rotvec = forward_map_param(p[:3], rot_bound)
        i = 3
        A = jnp.zeros((3, 3))
        if refine_cell:
            a = forward_map_param(p[i : i + 5], strain_bound)
            A = jnp.array(
                [[a[0], a[2], a[3]], [a[2], a[1], a[4]], [a[3], a[4], -a[0] - a[1]]]
            )
            i += 5
        det_norm = p[i : i + n_det]
        i += n_det
        goff = forward_map_param(p[i : i + n_goff], gonio_bound) if n_goff else p[i:i]
        return rotvec, A, det_norm, goff

    def neg_loglik(p):
        rotvec, A, det_norm, goff = unpack(p)
        # data side: peak directions from the perturbed instrument
        c, u, v, _, _, _ = apply_detector_modes(
            det_norm[None, :],
            centers,
            uhats,
            vhats,
            widths,
            heights,
            modes,
            param_slices,
            det_bounds,
        )
        xyz = peak_lab_xyz(c, u, v, det_idx, u_off, v_off)[0] - s_off
        kf = xyz / jnp.linalg.norm(xyz, axis=1, keepdims=True)
        g_lab = kf - ki_hat
        d_lab = g_lab / jnp.linalg.norm(g_lab, axis=1, keepdims=True)
        if gonio is not None:
            offs = jnp.asarray(nominal_off).at[np.where(refine_mask)[0]].add(goff)
            R_runs = jnp.stack(
                [
                    gonio_rotation_jax(axes, jnp.asarray(angles_deg[r]), offs)
                    for r in range(angles_deg.shape[0])
                ]
            )
            d_lab = jnp.einsum("nij,ni->nj", R_runs[run_idx], d_lab)  # R^T d
        # model side: orientation and (optionally) cell shape
        R = _rodrigues_jax(rotvec) @ jnp.asarray(R0)
        B = (jnp.eye(3) + A) @ jnp.asarray(B0)
        g_vec = hkl_j @ B.T
        dirs = g_vec / jnp.linalg.norm(g_vec, axis=1, keepdims=True)
        dots = jnp.abs(d_lab @ (dirs @ R.T).T)  # lines: +- are one direction
        ang2 = 2.0 * (1.0 - jnp.clip(dots, 0.0, 1.0))
        dens = jnp.sum(
            w_model[None, :] * jnp.exp(-ang2 / (2.0 * sigma * sigma)), axis=1
        )
        return -jnp.sum(w_data * jnp.log(dens + floor)) / jnp.sum(w_data)

    val_grad = jax.jit(jax.value_and_grad(neg_loglik))

    # Adam with a cosine-decayed step, not a quasi-Newton line search: the
    # objective is float32 (about seven digits), and line searches read
    # that noise floor as convergence long before the minimum.  Bounds by
    # projection; all parameters live in [0, 1] so one rate serves all.
    x = jnp.full(3 + n_cell + n_det + n_goff, 0.5, dtype=jnp.float32)
    m = jnp.zeros_like(x)
    v2 = jnp.zeros_like(x)
    lr0, beta1, beta2, eps = 0.03, 0.9, 0.999, 1e-8
    best_x, best_v = x, np.inf
    for t in range(1, int(maxiter) + 1):
        val, g = val_grad(x)
        if float(val) < best_v:
            best_v, best_x = float(val), x
        m = beta1 * m + (1 - beta1) * g
        v2 = beta2 * v2 + (1 - beta2) * g * g
        mh = m / (1 - beta1**t)
        vh = v2 / (1 - beta2**t)
        lr = lr0 * 0.5 * (1.0 + np.cos(np.pi * t / maxiter))
        x = jnp.clip(x - lr * mh / (jnp.sqrt(vh) + eps), 0.0, 1.0)
    val = val_grad(x)[0]
    if float(val) < best_v:
        best_v, best_x = float(val), x

    rotvec, A, det_norm, goff = unpack(best_x)
    out = {
        "R": np.asarray(_rodrigues_jax(jnp.asarray(rotvec))) @ np.asarray(R0),
        "B": (np.eye(3) + np.asarray(A)) @ np.asarray(B0),
        "det_params": np.asarray(det_norm),
        "loglik": float(-best_v),
    }
    if gonio is not None:
        offs = np.array(nominal_off)
        offs[refine_mask] += np.asarray(goff)
        out["gonio_offsets_deg"] = offs
    return out
