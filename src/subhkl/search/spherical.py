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
    """
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

    x = np.cos(beta)
    sh, ch = np.sin(beta / 2.0), np.cos(beta / 2.0)
    af, bf = a.astype(float), b.astype(float)

    # P_k^{(a,b)}(x) for k = 0..L, elementwise (a, b): standard three-term
    # recursion of the Jacobi polynomials, forward stable on [-1, 1].
    P = np.zeros((L + 1, M, M))
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
        ang = np.where(
            (a == 0) & (sh == 0.0), 1.0, np.exp(np.log(np.maximum(sh, 1e-300)) * af)
        ) * np.where(
            (b == 0) & (ch == 0.0), 1.0, np.exp(np.log(np.maximum(ch, 1e-300)) * bf)
        )

    d = np.zeros((L + 1, M, M))
    for l_ in range(L + 1):
        valid = mx <= l_
        k = l_ - mx
        # ln binom(2l - k, k + a) - ln binom(k + b, b), all indices >= 0
        top = (
            fl[l_ + mx] - fl[np.maximum(k + a, 0)] - fl[np.maximum(l_ + mx - k - a, 0)]
        )
        bot = fl[np.maximum(k + b, 0)] - fl[np.maximum(k, 0)] - fl[b]
        pref = np.exp(0.5 * (top - bot))
        Pk = np.take_along_axis(P, np.maximum(k, 0)[None], axis=0)[0]
        d[l_] = np.where(valid, sign * pref * ang * Pk, 0.0)
    return d


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


def correlogram(f, g, n_beta=None):
    """C(alpha, beta, gamma) = <f, Lambda(R) g> on an Euler-angle grid.

    For each beta the coupling S_{m n} = sum_l conj(f^l_m) d^l_{m n} g^l_n
    is a single contraction over degrees (the per-degree coupling is the
    rank-one outer product f^l (g^l)^dagger), and the (alpha, gamma)
    dependence is a 2D Fourier sum.  Cost O(L^3) per beta; no SO(3) grid
    finer than the readout grid ever exists.

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
    C = np.empty((n_ang, n_beta, n_ang))
    for t, beta in enumerate(betas):
        d = wigner_d_matrix(L, beta)
        S = np.einsum("lm,lmn,ln->mn", np.conj(f), d, g)
        C[:, t, :] = np.real(Ea.T @ S @ Ea)
    return C, alphas, betas, gammas


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
