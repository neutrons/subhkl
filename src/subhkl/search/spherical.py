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

from functools import partial

import numpy as np
from scipy.special import gammaln


def _precise(fn):
    """Run a jax-backed function with full float32 matmuls.

    On Ampere-class GPUs jax evaluates float32 matmuls at TF32 by
    default (10-bit mantissa): measured here, a unit-vector dot product
    carries up to 6e-4 of error, which next to 1 is a 2 deg angle -- the
    width of the whole kernel these functions integrate over.  On a
    synthetic scene with known geometry the instrument refinement's
    orientation error doubled (0.16 -> 0.28 deg) and its panel-parameter
    error tripled at the default.  Nothing here is a large GEMM, so the
    cost of asking for the real float32 is nil; the config is scoped to
    the call so the rest of the process keeps its own setting.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import jax

        with jax.default_matmul_precision("highest"):
            return fn(*args, **kwargs)

    return wrapper


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


def radial_basis(n_radial, q_max):
    """Orthonormal radial functions on the resolution ball, [0, q_max].

    Shifted Legendre polynomials, R_n(q) = sqrt((2n+1)/q_max)
    P_n(2 q / q_max - 1): orthonormal under dq, so a function on the ball
    expanded as sum_{n l m} a_{nlm} R_n(|q|) Y_lm(q^) has the 3D inner
    product sum |a|^2 -- and a rotation acts on (l, m) alone, which is
    what lets the correlogram absorb the radial sum into its per-degree
    coupling (correlogram accepts the [n, l, m] stack directly).

    Returns (shell, segment): shell(q) -> [len(q), n_radial] evaluates the
    basis at radii (a lattice point's |G|); segment(q_lo, q_hi) ->
    [len, n_radial] integrates it over radial intervals (a Laue spot's
    admissible |Q| range, [2 sin(theta) / lambda_max, 2 sin(theta) /
    lambda_min] clipped to the ball).  A spot and a lattice point then
    overlap in this basis exactly when the shell lies inside the segment,
    resolved to ~ q_max / n_radial -- the wavelength-band consistency rule
    (band_masks) as a single 3D inner product, with no stratification.
    """
    from numpy.polynomial import legendre as _leg

    n_radial = int(n_radial)
    norm = np.sqrt((2.0 * np.arange(n_radial) + 1.0) / float(q_max))

    def shell(q):
        x = np.clip(2.0 * np.asarray(q, float) / q_max - 1.0, -1.0, 1.0)
        V = _leg.legvander(x, n_radial - 1)  # [len, n]
        out = V * norm[None, :]
        out[np.asarray(q, float) > q_max] = 0.0
        return out

    # 12-point Gauss-Legendre on each interval: exact for the polynomials
    xg, wg = np.polynomial.legendre.leggauss(12)

    def segment(q_lo, q_hi):
        lo = np.asarray(q_lo, float)
        hi = np.minimum(np.asarray(q_hi, float), q_max)
        h = np.clip(hi - lo, 0.0, None)
        mid = 0.5 * (lo + hi)
        qs = mid[:, None] + 0.5 * h[:, None] * xg[None, :]  # [len, 12]
        vals = shell(qs.ravel()).reshape(len(lo), len(xg), n_radial)
        out = np.einsum("g,pgn->pn", wg, vals) * (0.5 * h)[:, None]
        out[h <= 0.0] = 0.0
        return out

    return shell, segment


def radial_model_stack(dirs, q, weights, n_radial, q_max, L, sigma):
    """The model on the ball: every reflection (both signs) as a shell at
    its |G|, carrying its ray's dictionary weight.  Returns [n_radial,
    L+1, 2L+1]; feed it to correlogram() against radial_data_stack()."""
    shell, _ = radial_basis(n_radial, q_max)
    d = np.asarray(dirs, float)
    qq = np.asarray(q, float)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, float)
    keep = (w > 0) & (qq <= q_max)
    d, qq, w = d[keep], qq[keep], w[keep]
    dd = np.vstack([d, -d])
    W = (shell(np.concatenate([qq, qq])) * np.concatenate([w, w])[:, None]).T
    return project_counts_device(dd, W, L, sigma)


def radial_data_weights(sin_theta, wavelength_band, q_max, n_radial):
    """Per-datum radial channel weights: the integral of the basis over
    the Laue segment [2 sin(theta) / lambda_max, 2 sin(theta) /
    lambda_min] clipped to the ball -- [n, n_radial].  Multiply into the
    data weights and project (project_counts_device with (frame x
    channel) rows, or radial_data_stack)."""
    _, segment = radial_basis(n_radial, q_max)
    s_ = np.asarray(sin_theta, float)
    lam_lo, lam_hi = float(wavelength_band[0]), float(wavelength_band[1])
    return segment(2.0 * s_ / lam_hi, np.minimum(2.0 * s_ / lam_lo, q_max))


def radial_data_stack(
    dirs, weights, sin_theta, wavelength_band, q_max, n_radial, L, sigma
):
    """The data on the ball: each datum smeared over its admissible |Q|
    segment.  Returns [n_radial, L+1, 2L+1]."""
    w = np.ones(len(dirs)) if weights is None else np.asarray(weights, float)
    seg = radial_data_weights(sin_theta, wavelength_band, q_max, n_radial)  # [n, N]
    return project_counts_device(
        np.asarray(dirs, float), (seg * w[:, None]).T, L, sigma
    )


def _canonical_sign(v):
    """Orient integer triples so the first nonzero component is positive:
    +-v are one direction on the sphere (and one line for the kernel)."""
    lead = np.where(v[:, 0] != 0, v[:, 0], np.where(v[:, 1] != 0, v[:, 1], v[:, 2]))
    return v * np.where(lead < 0, -1, 1)[:, None]


def nodal_points(B, max_index=3):
    """Nodal points: the crossings of the zone-ring family, as hkl.

    Zone [uvw] draws a great circle on the direction sphere (the Funk
    transform of its reciprocal directions), and two zones cross at the
    reciprocal direction hkl = uvw_1 x uvw_2 -- so the crossings are
    themselves lattice directions, the low-index ones, each weighted by
    the zone pairs meeting there (1/|t_1| 1/|t_2| per pair, the ring
    weights, summed over pairs).  A 0-D dictionary derived from the 1-D
    one: it keeps the ring basis's orientation-selective low-index content
    but has a point-sharp autocorrelation and a mutual-coherence floor far
    below the uniform reflection set's -- measured on CG4D garnet at a 1
    deg kernel, 0.003 at |uvw| <= 2 against 0.063 for every reflection to
    1 A and 0.31 for the rings themselves -- so its correlogram basins
    separate at modest bandwidth.  Its size follows the zone cut, not the
    reflection count: 357 / 3,233 / 13,117 points at |uvw| <= 2 / 3 / 4
    where a 100 A cell has 157k reflections (65k directions) at 2.5 A.
    The price is leverage: the high-index reflections that pin the finest
    orientation precision are not in it (0.2 deg against 0.12 deg on
    garnet), which is what a final pass on the full list restores.

    Returns (hkl [N, 3] int, weights [N]).  Directions are hkl @ B.T
    normalized, so a refined B moves the nodes exactly as it moves the
    reflections -- pass the triples, not the directions, to a refinement.
    """
    B = np.asarray(B, dtype=float)
    A = np.linalg.inv(B).T  # real-space basis as rows, a . a* = 1
    rng = np.arange(-max_index, max_index + 1)
    u, v, w = np.meshgrid(rng, rng, rng, indexing="ij")
    uvw = np.stack([u.ravel(), v.ravel(), w.ravel()], axis=1)
    uvw = uvw[np.any(uvw != 0, axis=1)]
    gcd = np.gcd.reduce(np.abs(uvw), axis=1)
    uvw = np.unique(_canonical_sign(uvw // gcd[:, None]), axis=0)
    wz = 1.0 / np.linalg.norm(uvw @ A, axis=1)
    i, j = np.triu_indices(len(uvw), k=1)
    hkl = np.cross(uvw[i], uvw[j])
    keep = np.any(hkl != 0, axis=1)
    hkl, wp = hkl[keep], wz[i][keep] * wz[j][keep]
    gcd = np.gcd.reduce(np.abs(hkl), axis=1)
    hkl = _canonical_sign(hkl // gcd[:, None])
    uniq, inv = np.unique(hkl, axis=0, return_inverse=True)
    weights = np.zeros(len(uniq))
    np.add.at(weights, inv.ravel(), wp)
    return uniq, weights


def primitive_spacing(hkl, B):
    """|G| of the primitive lattice vector along each hkl's ray [1/Angstrom].

    The Laue collapse keeps a spot's direction and loses its |Q|; what it
    does NOT lose is that along that ray the lattice only has points at
    multiples n g0 of the primitive spacing.  That is the scale-free
    remnant of the reciprocal lattice's additive structure, and
    band_masks turns it into a per-spot constraint.
    """
    hkl = np.asarray(hkl, dtype=int)
    gcd = np.maximum(np.gcd.reduce(np.abs(hkl), axis=1), 1)
    prim = hkl // gcd[:, None]
    return np.linalg.norm(prim @ np.asarray(B, dtype=float).T, axis=1)


def band_masks(g0, sin_edges, wavelength_band, d_min, n_sub=7):
    """Which dictionary directions can produce a spot in each sin(theta) band.

    A spot whose Q direction makes |Ghat . ki| = sin(theta) with the beam
    has |Q| = 2 sin(theta) / lambda, and with lambda anywhere in the band
    that is the interval [2 s / lambda_max, 2 s / lambda_min]; the lattice
    can only supply |G| = n g0, and only within the resolution cut.  So a
    direction with primitive spacing g0 is a POSSIBLE origin of the spot
    iff some integer n >= 1 has n g0 in [2 s / lambda_max,
    min(2 s / lambda_min, 1 / d_min)] -- no wavelength measured, just
    consistency with one existing.  For a factor-2 band the condition
    collapses to g0 <= s below the cut and to a narrow window above it.
    Measured on a 100 A cell at 2.5 A in a 2-4 A band, the legal model per
    spot shrinks from 65k directions to an effective 19k and the
    spurious-match rate from 27% to 8%, concentrated on the low-angle
    spots that carry most of the intensity; on garnet (2-10 A) it lifts
    the point model's search z from 16.8 to 27.4.  Spots with
    2 s / lambda_max > 1 / d_min can be explained by nothing in the model
    at all and get an empty mask.

    Returns a list of boolean masks over the dictionary, one per band
    (sin_edges[i], sin_edges[i+1]], each the union over n_sub sample
    points of the band so no legal direction is dropped at a band edge.
    """
    g0 = np.asarray(g0, dtype=float)
    lam_lo, lam_hi = float(wavelength_band[0]), float(wavelength_band[1])
    gmax = 1.0 / float(d_min)
    masks = []
    for lo, hi in zip(sin_edges[:-1], sin_edges[1:]):
        m = np.zeros(len(g0), dtype=bool)
        for s in np.linspace(lo, hi, n_sub)[1:]:
            q_lo, q_hi = 2.0 * s / lam_hi, min(2.0 * s / lam_lo, gmax)
            if q_hi < q_lo:
                continue
            m |= np.floor(q_hi / g0 + 1e-12) >= np.ceil(q_lo / g0 - 1e-12)
        masks.append(m)
    return masks


def sin_theta_edges(sin_theta, weights=None, n_bands=4):
    """Weighted-quantile band edges over the data's sin(theta), so every
    band carries the same evidence; the lowest band is where the rule
    bites hardest."""
    s = np.asarray(sin_theta, dtype=float)
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float)
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / max(np.sum(w), 1e-12)
    qs = np.linspace(0.0, 1.0, n_bands + 1)[1:-1]
    inner = [float(s[o][min(np.searchsorted(cw, q), len(s) - 1)]) for q in qs]
    return np.array([0.0] + inner + [1.0 + 1e-9])


@_precise
def banded_search(f_bands, dirs, weights, masks, L, sigma, min_dirs=8):
    """The band-consistent correlogram: each sin(theta) band of the data
    correlated against only the directions that could have produced it,
    each band's correlogram standardized against its own null (median /
    MAD over SO(3)) and the z-maps summed.  The per-band models are
    normalized to unit norm so that a band restricted to few directions
    -- the informative one -- is not outweighed by an unrestricted band
    seven times its size; the standardization then makes the sum a
    Fisher combination of independent tests rather than a sum of
    unrelated scales.  Returns (Z, alphas, betas, gammas) on the same
    grid as correlogram()."""
    Z = None
    grid = None
    for f_b, m in zip(f_bands, masks):
        if f_b is None or int(np.sum(m)) < min_dirs or not np.any(f_b):
            continue
        w_b = None if weights is None else np.asarray(weights, float)[m]
        g_b = project_points(np.asarray(dirs, float)[m], w_b, L, sigma)
        g_b = g_b / max(np.linalg.norm(g_b), 1e-300)
        C, al, be, ga = correlogram(f_b, g_b)
        med = np.median(C)
        mad = 1.4826 * np.median(np.abs(C - med))
        Zb = (C - med) / max(mad, 1e-12)
        Z = Zb if Z is None else Z + Zb
        grid = (al, be, ga)
    if Z is None:
        raise ValueError("no sin(theta) band has both data and a permitted model")
    return Z, grid[0], grid[1], grid[2]


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
    """SH coefficients of raw count maps -- no peak finding required.

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

    ``excess`` may be 1D (N,) -> coefficients (L+1, 2L+1), or 2D
    (n_frames, N) -> (n_frames, L+1, 2L+1): every frame sharing the same
    pixel directions is projected in ONE pass, so the associated-Legendre
    evaluation -- the dominant, memory-bound cost -- is paid per detector
    bank instead of per frame.  That is the point: a bank's binned
    geometry is fixed in the lab frame, so project all its frames here and
    rotate the *coefficients* per run (rotate_coeffs, O(L^3)) instead of
    rotating directions and re-evaluating the basis per frame
    (O(L^2 N) each).  Measured at benchmark scale (cg4d-garnet, 1114
    frames) the per-frame form was ~1.5 s/frame -- half an hour of
    projection; the batched form is a few large BLAS contractions.
    Negative weights are fine.
    """
    d = np.asarray(pixel_dirs, dtype=float)
    w = np.asarray(excess, dtype=float)
    one_d = w.ndim == 1
    W = w[None, :] if one_d else w
    nf = W.shape[0]
    f = np.zeros((nf, L + 1, 2 * L + 1), dtype=complex)
    for start in range(0, len(d), int(chunk)):
        dd = d[start : start + int(chunk)]
        Wc = W[:, start : start + int(chunk)]
        ct = np.clip(dd[:, 2], -1.0, 1.0)
        phi = np.arctan2(dd[:, 1], dd[:, 0])
        P = _legendre_norm(L, ct)  # [l, m>=0, j]
        # f[fr, l, m] = sum_j W[fr, j] Pbar_lm(j) e^{-i m phi_j}: one GEMM
        # per order m, each (n_frames x chunk) @ (chunk x (L+1-m))
        for m in range(L + 1):
            WE = Wc * np.exp(-1j * m * phi)[None, :]
            f[:, m:, L + m] += WE @ P[m:, m].T
    mneg = np.arange(1, L + 1)
    f[:, :, L - mneg] = (-1.0) ** mneg * np.conj(f[:, :, L + mneg])
    f *= _smoothing(L, sigma)[None, :, None]
    if even_only:
        f[:, 1::2] = 0.0
    return f[0] if one_d else f


_PROJECT_KERNEL = None


def _get_project_kernel():
    """Jitted batched-weights projection (module scope, one cache key):
    the numpy per-order GEMMs are 14-row skinny complex matmuls that run
    at ~10 GFLOP/s on CPU BLAS -- measured 98 s of a 158 s benchmark-scale
    raw run.  Same associated-Legendre fori recursion as _sph_coeffs_jax,
    contracted against every frame's weights at once on the device."""
    global _PROJECT_KERNEL
    if _PROJECT_KERNEL is not None:
        return _PROJECT_KERNEL
    import jax
    import jax.numpy as jnp
    from jax import lax

    @partial(jax.jit, static_argnames=("L",))
    def kernel(dirs, W, L):
        ct = jnp.clip(dirs[:, 2], -1.0, 1.0)
        st = jnp.sqrt(jnp.maximum(1.0 - ct * ct, 1e-20))
        phi = jnp.arctan2(dirs[:, 1], dirs[:, 0])
        m_idx = jnp.arange(L + 1, dtype=jnp.float32)
        E = jnp.exp(-1j * jnp.outer(m_idx, phi))  # [m, j]
        Wc = W.astype(jnp.complex64)
        signs = jnp.asarray((-1.0) ** np.arange(1, L + 1))

        def coeffs_from_row(P_row):
            # cpos[f, m] = sum_j W[f, j] Pbar_{l m}(j) e^{-i m phi_j}
            cpos = jnp.einsum("mj,fj->fm", (P_row.astype(jnp.complex64) * E), Wc)
            cneg = signs[None, :] * jnp.conj(cpos[:, 1:])
            return jnp.concatenate([jnp.flip(cneg, axis=1), cpos], axis=1)

        n = dirs.shape[0]
        p00 = jnp.full((n,), float(np.sqrt(1.0 / (4.0 * np.pi))), dtype=jnp.float32)
        P0 = jnp.zeros((L + 1, n), dtype=jnp.float32).at[0].set(p00)
        F = jnp.zeros((W.shape[0], L + 1, 2 * L + 1), dtype=jnp.complex64)
        F = F.at[:, 0].set(coeffs_from_row(P0))
        P1 = (
            jnp.zeros((L + 1, n), dtype=jnp.float32)
            .at[0]
            .set(np.sqrt(3.0) * ct * p00)
            .at[1]
            .set(-np.sqrt(3.0 / 2.0) * st * p00)
        )
        F = F.at[:, 1].set(coeffs_from_row(P1))

        def body(l_, state):
            Pm2, Pm1, F = state
            lf = jnp.asarray(l_, dtype=jnp.float32)
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
            diag_prev = Pm1[l_ - 1]
            row = row.at[l_ - 1].set(jnp.sqrt(2.0 * lf + 1.0) * ct * diag_prev)
            row = row.at[l_].set(
                -jnp.sqrt((2.0 * lf + 1.0) / (2.0 * lf)) * st * diag_prev
            )
            row = jnp.where((m_idx <= lf)[:, None], row, 0.0)
            F = F.at[:, l_].set(coeffs_from_row(row))
            return (Pm1, row, F)

        _, _, F = lax.fori_loop(2, L + 1, body, (P0, P1, F))
        return F

    _PROJECT_KERNEL = kernel
    return kernel


@_precise
def project_counts_device(pixel_dirs, W, L, sigma, even_only=True):
    """project_counts on the accelerator: float32, all frames of a shared
    geometry in one fused pass.  Used by the raw-count CLI path; the
    float64 numpy project_counts remains the reference the exactness
    tests pin."""
    import jax.numpy as jnp

    kernel = _get_project_kernel()
    F = np.asarray(
        kernel(
            jnp.asarray(np.asarray(pixel_dirs), dtype=jnp.float32),
            jnp.asarray(np.asarray(W), dtype=jnp.float32),
            int(L),
        )
    ).astype(complex)
    F *= _smoothing(L, sigma)[None, :, None]
    if even_only:
        F[:, 1::2] = 0.0
    return F


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


#: Largest (data x model) overlap array the matching-free objective will
#: materialize at once, in elements.  The objective's cost is one such
#: array; its MEMORY need not be, and a 100 A cell puts 65k reflections on
#: the model side (L1 metallo-beta-lactamase at d_min 2.5), where 0.5M
#: weighted pixels would ask for 117 GiB in one allocation.  Chunking the
#: model axis is exact -- the density is a sum over model directions, so a
#: running sum over chunks is the same number -- and the chunks are
#: rematerialized so the backward pass does not undo the saving.
MAX_OVERLAP_ELEMS = 1 << 28  # ~1.07 GB per float32 temporary


def _overlap_chunk_plan(n_data, n_model, max_elems):
    """How to slice a (data x model) overlap so no temporary exceeds
    max_elems: (n_chunks, chunk_m, pad_idx).  pad_idx is None when one
    pass fits; otherwise it lists model rows with the last chunk padded by
    repeating row 0, and the caller gives the padding rows weight 0 so the
    chunked sum stays exact rather than approximate."""
    if n_data * n_model <= max_elems:
        return 1, n_model, None
    chunk_m = int(max(1, max_elems // max(n_data, 1)))
    n_chunks = int(np.ceil(n_model / chunk_m))
    pad_idx = np.concatenate(
        [np.arange(n_model), np.zeros(n_chunks * chunk_m - n_model, dtype=int)]
    )
    return n_chunks, chunk_m, pad_idx


@_precise
def nearest_line_stats(
    pts, dirs, sigma, want_density=True, max_overlap_elems=MAX_OVERLAP_ELEMS
):
    """Nearest model line and von Mises density for every data direction,
    on the device.  Returns (dev_deg [n], dens [n]); dens is None without
    want_density.

    The same (data x model) overlap the instrument refinement evaluates,
    chunked over the model axis by the same plan, and for the same
    reason: at L1 scale (500k pixels against 65k reflection directions,
    nine passes) the numpy version was a quarter hour of single-threaded
    elementwise work with OpenBLAS spinning 128 threads around each tiny
    chunk matmul while the GPU sat idle.  The device does the whole
    contraction in seconds.  The angle is NOT taken from the float32 dot
    product -- near 1 that quantizes at 0.02 deg, coarser than the medians
    reported -- the device finds WHICH line is nearest and the host
    recomputes that one angle in float64.  The density has no cutoff: it
    is the refinement's objective exactly, every direction summed.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    P = np.asarray(pts, dtype=np.float64)
    D = np.asarray(dirs, dtype=np.float64)
    n, m = len(P), len(D)
    n_chunks, chunk_m, pad_idx = _overlap_chunk_plan(n, m, max_overlap_elems)
    valid = np.ones(n_chunks * chunk_m, dtype=np.float32)
    if pad_idx is not None:
        valid[m:] = 0.0
        D_use = D[pad_idx]
    else:
        D_use = D
    inv2s2 = np.float32(1.0 / (2.0 * sigma * sigma))
    Pj = jnp.asarray(P, dtype=jnp.float32)
    D_r = jnp.asarray(D_use, dtype=jnp.float32).reshape(n_chunks, chunk_m, 3)
    v_r = jnp.asarray(valid).reshape(n_chunks, chunk_m)

    @jax.jit
    def run(Pj, D_r, v_r):
        def body(carry, packed):
            best, arg, dens, off = carry
            d_c, v_c = packed
            dots = jnp.clip(jnp.abs(Pj @ d_c.T), 0.0, 1.0)  # lines: +- collapse
            masked = jnp.where(v_c[None, :] > 0, dots, -1.0)
            loc = jnp.argmax(masked, axis=1)
            val = jnp.take_along_axis(masked, loc[:, None], axis=1)[:, 0]
            better = val > best
            best = jnp.where(better, val, best)
            arg = jnp.where(better, off + loc, arg)
            if want_density:
                ang2 = 2.0 * (1.0 - dots)
                dens = dens + jnp.sum(v_c[None, :] * jnp.exp(-ang2 * inv2s2), axis=1)
            return (best, arg, dens, off + chunk_m), None

        init = (
            jnp.full(n, -1.0, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.int32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.int32(0),
        )
        (best, arg, dens, _), _ = lax.scan(body, init, (D_r, v_r))
        return arg, dens

    arg, dens = run(Pj, D_r, v_r)
    arg = np.asarray(arg, dtype=int)
    if pad_idx is not None:
        arg = pad_idx[arg]
    cosang = np.abs(np.sum(P * D[arg], axis=1))
    dev_deg = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
    return dev_deg, (np.asarray(dens, dtype=np.float64) if want_density else None)


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


def so3_inner_stack(f, g, R):
    """<f, Lambda(R) g> summed over radial channels (stacks [K, L+1, M])."""
    return float(sum(so3_inner(f[k], g[k], R) for k in range(len(f))))


def so3_inner(f, g, R):
    """C(R) = <f, Lambda(R) g> = sum_l f^l dagger D^l(R) g^l.  [real]"""
    rf = rotate_coeffs(g, R)
    return float(np.real(np.sum(np.conj(f) * rf)))


@_precise
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
    f = np.asarray(f)
    g = np.asarray(g)
    L = f.shape[-2] - 1
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
        if f.ndim == 3:
            S = np.einsum("klm,lmnb,kln->bmn", np.conj(f), d, g, optimize=True)
        else:
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


def _coupling_prep(f, g):
    """Numpy prep shared by the fused kernels: coupling and prefactor
    ratios in recursion-depth order (see _correlogram_kernel_jax)."""
    f = np.asarray(f)
    g = np.asarray(g)
    L = f.shape[-2] - 1
    mx, a, b, pref = _wigner_case_tables(L)
    if f.ndim == 3:
        # radial channels (radial_basis): the coupling of a 3D function
        # is the SUM over channels of the per-channel rank-one couplings,
        # because a rotation acts on (l, m) alone -- so the whole radial
        # dimension collapses here, before the SO(3) transform, and the
        # kernel below never sees it
        coef = np.einsum("klm,kln->lmn", np.conj(f), g)
    else:
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
    return a, b, coefK, prefK, rho1, rho2


def _S_terms(a, b, prefK, betas):
    """float64 numpy inputs of the D-recursion for the given betas:
    x = cos(beta) and the O(1) starting rows D0, D1 (see the kernel)."""
    sh, ch = np.sin(betas / 2.0), np.cos(betas / 2.0)
    with np.errstate(divide="ignore"):
        ang = np.exp(
            np.log(np.maximum(sh, 1e-300))[None, None, :] * a[..., None]
            + np.log(np.maximum(ch, 1e-300))[None, None, :] * b[..., None]
        )
    x64 = np.cos(betas)
    D0 = prefK[0][..., None] * ang
    P1_64 = 0.5 * (a - b)[..., None] + (1.0 + 0.5 * (a + b))[..., None] * x64
    D1 = prefK[1][..., None] * ang * P1_64
    return x64, D0, D1


def _S_kernel_jax(coefK, rho1, rho2, a, b, x64, D0, D1):
    """S[m, n, beta] = sum_l conj(f)_lm d^l_mn(beta) g_ln, fused.

    Recurs on the bounded D_k = pref_k ang P_k (|d| <= 1): ang cancels
    from the linear recursion, pref enters via polynomially bounded
    ratios.  Everything float32/complex64 and O(1); no Wigner tensor is
    ever materialized.  One compile per (L, n_beta) shape.
    """
    import jax.numpy as jnp

    kernel = _get_S_kernel()
    return kernel(
        jnp.asarray(coefK, dtype=jnp.complex64),
        jnp.asarray(rho1, dtype=jnp.float32),
        jnp.asarray(rho2, dtype=jnp.float32),
        jnp.asarray(a, dtype=jnp.float32)[..., None],
        jnp.asarray(b, dtype=jnp.float32)[..., None],
        jnp.asarray(x64, dtype=jnp.float32)[None, None, :],
        jnp.asarray(D0, dtype=jnp.float32),
        jnp.asarray(D1, dtype=jnp.float32),
    )


_S_KERNEL = None


def _get_S_kernel():
    """The jitted recursion, created ONCE at module scope.  A jit closure
    built inside the calling function is a fresh cache key every call --
    measured as a full recompile per Nelder-Mead evaluation, slower than
    the numpy path it replaced."""
    global _S_KERNEL
    if _S_KERNEL is not None:
        return _S_KERNEL
    import jax
    import jax.numpy as jnp
    from jax import lax

    @jax.jit
    def kernel(coefK_j, rho1_j, rho2_j, af, bf, x, D0_j, D1_j):
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
        return S

    _S_KERNEL = kernel
    return kernel


def _correlogram_kernel_jax(f, g, betas, Ea):
    """The fused correlogram over the full Euler grid (see _S_kernel_jax
    for the numerics)."""
    import jax.numpy as jnp

    a, b, coefK, prefK, rho1, rho2 = _coupling_prep(f, g)
    x64, D0, D1 = _S_terms(a, b, prefK, betas)
    S = _S_kernel_jax(coefK, rho1, rho2, a, b, x64, D0, D1)
    Ea_j = jnp.asarray(Ea, dtype=jnp.complex64)
    return jnp.real(jnp.einsum("ma,mnb,nc->abc", Ea_j, S, Ea_j))


@_precise
def _so3_inner_fast_factory(f, g):
    """Closure evaluating C(R) = <f, Lambda(R) g> through the fused jax
    kernel -- for optimizers that evaluate hundreds of single rotations
    (refine_local's Nelder-Mead spent 48 s in scalar numpy Wigner builds
    per raw-mode run; this makes each evaluation a milliseconds-scale
    nb = 1 kernel call with the coupling prepared once)."""
    import jax.numpy as jnp

    L = f.shape[-2] - 1  # a channel stack [K, L+1, M] couples summed over K
    a, b, coefK, prefK, rho1, rho2 = _coupling_prep(f, g)
    m = np.arange(-L, L + 1)
    # place the rotation-independent constants on the device ONCE: passing
    # numpy here re-uploaded the 28 MB coupling tensor per evaluation --
    # measured 533 calls x 0.059 s of PCIe traffic per raw-mode run, which
    # was most of the "GPU at 4%" segment
    kernel = _get_S_kernel()
    coefK_d = jnp.asarray(coefK, dtype=jnp.complex64)
    rho1_d = jnp.asarray(rho1, dtype=jnp.float32)
    rho2_d = jnp.asarray(rho2, dtype=jnp.float32)
    af_d = jnp.asarray(a, dtype=jnp.float32)[..., None]
    bf_d = jnp.asarray(b, dtype=jnp.float32)[..., None]

    @_precise
    def inner(R):
        alpha, beta, gamma = euler_zyz(R)
        x64, D0, D1 = _S_terms(a, b, prefK, np.array([beta]))
        S = kernel(
            coefK_d,
            rho1_d,
            rho2_d,
            af_d,
            bf_d,
            jnp.asarray(x64, dtype=jnp.float32)[None, None, :],
            jnp.asarray(D0, dtype=jnp.float32),
            jnp.asarray(D1, dtype=jnp.float32),
        )
        ea = jnp.asarray(np.exp(-1j * m * alpha), dtype=jnp.complex64)
        eg = jnp.asarray(np.exp(-1j * m * gamma), dtype=jnp.complex64)
        return float(jnp.real(jnp.einsum("m,mnb,n->", ea, S, eg)))

    return inner


# ---------------------------------------------------------------------------
# peak extraction, refinement, admission
# ---------------------------------------------------------------------------


def _quat_angle(R1, R2):
    """Rotation angle between two orientations.  [rad]"""
    tr = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))


def top_orientations(C, alphas, betas, gammas, n=4, min_sep_deg=None):
    """Local maxima of the correlogram as rotations, best first.

    Wrap-aware local-maximum test in alpha and gamma, then greedy selection
    with a minimum quaternion separation so one broad peak is not returned
    n times.  Returns [(R, value), ...].

    The separation defaults to twice the grid step (3.7 deg at L = 96): a
    peak is about one kernel wide, and distinct maxima that close are real.
    A fixed 10 deg lost MANDI L1 run 0 outright -- the truth (z 13.6, the
    global maximum of this very correlogram, TOF-validated) sits 4.9 deg
    from a false peak whose grid sample scored higher (11.3), so it was
    never refined; the measured valley between them is 2-3 deg.
    """
    if min_sep_deg is None:
        min_sep_deg = 2.0 * 180.0 / max(len(betas), 1)
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


@_precise
def refine_local(f, g, R, span_deg=3.0):
    """Grid-free polish by maximizing C(R) itself (Nelder--Mead on the
    rotation vector).  Works for any model -- rings included, where Wahba
    has no point pairs to match -- at the cost of being bandlimited: the
    objective is the L-bandlimited correlation, so precision is ~kernel
    width / sqrt(matches), not the Wahba floor."""
    from scipy.optimize import minimize

    inner = _so3_inner_fast_factory(f, g)

    def neg(v):
        return -inner(_rodrigues(v) @ R)

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


@_precise
def nonneg_lasso(f, g, rotations, lam=0.0, n_iter=300, inner_fg=None, inner_gg=None):
    """Sparse nonnegative coefficients over refined orientation candidates.

    Solves min_{c >= 0} 1/2 ||f - sum_r c_r Lambda(R_r) g||^2 + lam sum c
    in the SH domain by projected gradient.  With lam = 0 the nonnegativity
    and the candidates' mutual coherence (the Gram below, generically small
    between distinct orientations) already produce exact zeros for
    candidates the data does not support -- the L1-on-coefficients stage of
    the sparse recovery, run on the few survivors rather than all of SO(3).
    Returns c [n_candidates].
    """
    if inner_fg is None:
        inner_fg = _so3_inner_fast_factory(f, g)
    if inner_gg is None:
        inner_gg = _so3_inner_fast_factory(g, g)
    n = len(rotations)
    G = np.empty((n, n))
    b = np.empty(n)
    for r, Rr in enumerate(rotations):
        b[r] = inner_fg(Rr)
        for s_, Rs in enumerate(rotations):
            if s_ < r:
                G[r, s_] = G[s_, r]
            else:
                G[r, s_] = inner_gg(Rr.T @ Rs)
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
    min_sep_deg=None,
    refine=True,
    lam=0.0,
    refine_method="wahba",
    data_sin_theta=None,
    model_g0=None,
    wavelength_band=None,
    d_min=None,
    n_bands=4,
    n_radial=0,
    model_shells=None,
):
    """Find crystal orientations from measured Q directions.

    data_dirs: measured Ghat unit vectors (use ghat_from_kf).  Provide
    either model_dirs (+ optional weights; see lattice_directions) or
    ring_axes (+ weights; see zone_axes) as the model.  kernel_deg is the
    angular width [deg] given to every direction (mosaic + measurement);
    L defaults to the bandwidth at which that kernel has decayed to ~1%,
    capped at 96 -- candidates only need to be separated here, refinement
    restores full precision off-grid.  refine_method picks how a point
    model's candidates are polished: "wahba" assigns each datum to its
    nearest model direction within 3 deg and solves the orthogonal
    Procrustes problem, right when the model holds every direction the
    data can show; "local" ascends the band-limited correlation itself
    (no assignment), the choice for a partial dictionary such as
    nodal_points, where most data lie on no model direction and an
    assignment to the nearest node would bias the fit.  Ring models
    always use "local".

    With data_sin_theta (|Ghat . ki| per datum), model_g0 (primitive
    spacing per model direction), wavelength_band and d_min, the search
    is the band-consistent one of banded_search: every datum is matched
    only against directions that could have produced it at some
    wavelength in the band.  With n_radial > 0 and model_shells =
    (dirs, |G|, weights) over every reflection, the search is the 3D one
    instead: data as Laue segments and model as shells on the resolution
    ball (radial_basis), the band rule as one inner product.

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

    radial3d = (
        int(n_radial) > 0
        and model_shells is not None
        and data_sin_theta is not None
        and wavelength_band is not None
        and d_min is not None
    )
    banded = (
        not radial3d
        and data_sin_theta is not None
        and model_g0 is not None
        and wavelength_band is not None
        and d_min is not None
        and model_dirs is not None
    )
    if radial3d:
        q_max = 1.0 / float(d_min)
        sd, sq, sw = model_shells
        g3 = radial_model_stack(sd, sq, sw, int(n_radial), q_max, L, sigma)
        f3 = radial_data_stack(
            np.asarray(data_dirs, float),
            data_weights,
            data_sin_theta,
            wavelength_band,
            q_max,
            int(n_radial),
            L,
            sigma,
        )
        C, al, be, ga = correlogram(f3, g3)
    elif banded:
        sth = np.asarray(data_sin_theta, float)
        edges = sin_theta_edges(sth, data_weights, n_bands)
        masks = band_masks(model_g0, edges, wavelength_band, d_min)
        dd = np.asarray(data_dirs, float)
        f_bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sth > lo) & (sth <= hi)
            w_m = None if data_weights is None else np.asarray(data_weights, float)[m]
            f_bands.append(project_points(dd[m], w_m, L, sigma) if m.any() else None)
        C, al, be, ga = banded_search(
            f_bands, model_dirs, model_weights, masks, L, sigma
        )
    else:
        C, al, be, ga = correlogram(f, g)
    cands = top_orientations(C, al, be, ga, n=n_candidates, min_sep_deg=min_sep_deg)
    out = []
    rotations = []
    kept_vals = []
    # single-rotation inners through the fused jax kernel: the numpy path
    # rebuilds a scalar Wigner stack per call (~0.35 s at L = 96), and a
    # per-run offsets fit makes hundreds of these
    inner_fg = _so3_inner_fast_factory(f, g)
    inner_gg = _so3_inner_fast_factory(g, g)
    gnorm = np.sqrt(inner_gg(np.eye(3)))
    for R, val in cands:
        n_matched = 0
        if refine and model_dirs is not None and refine_method == "wahba":
            R, n_matched = refine_wahba(R, data_dirs, model_dirs)
            val = inner_fg(R)
        elif refine:
            R = refine_local(f, g, R)
            val = inner_fg(R)
        # Duplicates: the same peak reached twice (within the grid
        # separation), or a lattice-symmetry copy -- Lambda_S g = g for a
        # symmetry S, so the copy carries the *same* rotated model and,
        # exactly, the same objective value; the sparse stage cannot
        # apportion mass between identical columns, so keep one.  The
        # coherence alone is NOT the test: for a dictionary that is uniform
        # at this bandwidth (MANDI L1, 65k lines at L = 96) it is > 0.99
        # for any rotation of a few degrees, and that deleted the true
        # orientation (z 13.6, TOF-validated) as a "copy" of a false peak
        # 4.9 deg away (z 11.3).  Coherence is recorded for the report.
        mu = max(
            (abs(inner_gg(R.T @ R2)) / gnorm**2 for R2 in rotations),
            default=0.0,
        )
        sep = np.deg2rad(
            min_sep_deg if min_sep_deg is not None else 2.0 * 180.0 / max(len(be), 1)
        )
        if any(
            _quat_angle(R, R2) < sep
            or (mu > 0.99 and abs(val - v2) <= 1e-3 * max(abs(val), abs(v2), 1e-30))
            for R2, v2 in zip(rotations, kept_vals)
        ):
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
        kept_vals.append(val)
    if rotations:
        c = nonneg_lasso(f, g, rotations, lam=lam, inner_fg=inner_fg, inner_gg=inner_gg)
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


@_precise
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
        # Entries that have underflowed carry no gradient.  Near the poles
        # the sectoral chain st^m underflows float32 long before m = L
        # (st = 0.35 at m = 192 is 1e-88), and the backward pass through
        # the recursion's near-diagonal coefficients (~2l per step)
        # amplifies the adjoint of those zeros into inf, then 0 x inf =
        # NaN: at L = 192, 3,984 of L1's 65k directions -- every one
        # within 20 deg of a pole -- and L-BFGS aborted at iteration 0, so
        # refine_matching_free silently returned its input.  The forward
        # values are untouched; a true 1e-90 has no gradient worth keeping.
        row = jnp.where(jnp.abs(row) < 1e-30, lax.stop_gradient(row), row)
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


@_precise
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


@_precise
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
    cylinder_axis=None,
    global_rot_axis=None,
    refine_banks=None,
    bank_ids_present=None,
    refine_beam=False,
    beam_bound_deg=1.0,
    refine_sample=False,
    sample_bound_m=0.005,
    per_run_trans=False,
    per_run_trans_bound_m=0.005,
    max_overlap_elems=MAX_OVERLAP_ELEMS,
    model_weights=None,
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
    # Row membership as one-hot matrices, not index gathers.  The forward
    # cost is the same, the BACKWARD is not: the gradient of a gather of
    # 100k rows from 39 panels (or 11 runs) is a scatter-add into 39 (or
    # 11) rows, which XLA serializes on the GPU -- an nsys trace of the L1
    # refinement put 96% of device time in three such scatters, 0.41 s +
    # 3 x 0.125 s per Adam step, with the GPU at 100 W.  As a matmul the
    # gradient is a GEMM against the transposed membership, microseconds.
    n_banks_m = centers.shape[1]
    M_det_T = jnp.asarray(np.eye(n_banks_m, dtype=np.float32)[det_idx])  # [n, banks]
    widths = jnp.asarray(np.asarray(detector_nominal["widths"], float))[None]
    heights = jnp.asarray(np.asarray(detector_nominal["heights"], float))[None]
    n_banks = centers.shape[1]
    ki_hat = np.asarray(ki, float)
    ki_hat = jnp.asarray(ki_hat / np.linalg.norm(ki_hat))
    s_off = jnp.zeros(3) if sample_offset is None else jnp.asarray(sample_offset)

    from subhkl.instrument.refinables import detector_mode_slices

    # the full classic mode chain, one layout definition for both paths;
    # bounds default to the classic path's defaults
    defaults = {
        "radial": 0.05,
        "area": 0.05,
        "global_rot": np.deg2rad(2.0),
        "global_rot_axis": np.deg2rad(2.0),
        "global_trans": 0.01,  # [m]
        "independent_trans": 0.01,  # [m]
        "independent_rot": np.deg2rad(1.0),  # [rad]
    }
    det_bounds = {**defaults, **(det_bounds or {})}
    param_slices, n_det = detector_mode_slices(modes, n_banks)
    if cylinder_axis is not None:
        cylinder_axis = np.asarray(cylinder_axis, float)
        cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)
    if global_rot_axis is not None:
        global_rot_axis = np.asarray(global_rot_axis, float)
        global_rot_axis = global_rot_axis / np.linalg.norm(global_rot_axis)
    # freeze mask: independent parameters of banks outside refine_banks stay
    # at nominal (0.5); global modes always refine
    det_free = np.ones(n_det, dtype=bool)
    if (
        refine_banks is not None
        and "independent" in modes
        and bank_ids_present is not None
    ):
        keep = np.isin(np.asarray(bank_ids_present), np.asarray(list(refine_banks)))
        sl = param_slices["independent"]
        free6 = np.repeat(keep, 3)
        det_free[sl] = np.concatenate([free6, free6])

    n_tilt = n_prun = n_harm = 0
    if gonio is not None:
        from subhkl.instrument.refinables import axis_tilt_frames

        axes = np.asarray(gonio["axes"], dtype=float)
        angles_deg = np.asarray(gonio["angles_deg"], dtype=float)
        refine_mask = np.asarray(gonio["refine_mask"], dtype=bool)
        gonio_bound = float(gonio.get("bound_deg", 2.0))
        nominal_off = np.asarray(
            gonio.get("nominal_offsets_deg", np.zeros(axes.shape[0])), dtype=float
        )
        run_idx = np.asarray(peaks["run_idx"], dtype=int)
        n_goff = int(refine_mask.sum())
        n_runs_g = angles_deg.shape[0]
        M_run_T = jnp.asarray(np.eye(n_runs_g, dtype=np.float32)[run_idx])  # [n, runs]
        base_dirs = axes[:, :3] / np.linalg.norm(axes[:, :3], axis=1, keepdims=True)
        frames_tilt = axis_tilt_frames(axes)
        # axis-vector tilts: two bounded angles per masked axis, about an
        # orthonormal basis perpendicular to the nominal direction -- the
        # classic --refine-goniometer-axis-vector
        tilt_mask = np.asarray(
            gonio.get("axis_tilt_mask", np.zeros(len(axes), bool)), dtype=bool
        )
        tilt_bound = np.deg2rad(float(gonio.get("tilt_bound_deg", 1.0)))
        tilt_axes = [k for k in range(len(axes)) if tilt_mask[k]]
        n_tilt = 2 * len(tilt_axes)
        # per-run angle corrections on one motor -- the classic
        # --refine-goniometer-per-run (the literal DPHI); the constant part
        # is that motor's zero offset and inherits its gauge status
        per_run_axis = gonio.get("per_run_axis")
        per_run_bound = float(gonio.get("per_run_bound_deg", 1.0))
        n_prun = n_runs_g if per_run_axis is not None else 0
        # harmonic rocking: the per-run correction of one motor constrained
        # to a Fourier series in its own nominal angle -- the classic
        # --refine-goniometer-harmonics, at run resolution
        harmonics_axis = gonio.get("harmonics_axis")
        harm_orders = list(gonio.get("harmonics_orders", []) or [])
        harm_bound = float(gonio.get("harmonics_bound_deg", 0.5))
        n_harm = 2 * len(harm_orders) if harmonics_axis is not None else 0
        if harmonics_axis is not None:
            th_h = np.deg2rad(angles_deg[:, int(harmonics_axis)])
            harm_basis = np.stack(
                sum(([np.cos(o * th_h), np.sin(o * th_h)] for o in harm_orders), []),
                axis=1,
            )  # [n_runs, 2 n_orders]
    else:
        n_goff = 0
    n_beam = 2 if refine_beam else 0
    n_samp = 3 if refine_sample else 0
    n_prt = 3 * angles_deg.shape[0] if (per_run_trans and gonio is not None) else 0

    hkl_j = np.asarray(hkl, dtype=float)
    # Model weights (nodal multiplicities, harmonic counts) scaled to mean
    # 1: the density is unnormalized and the uniform floor is set against
    # a unit-height direction, so the scale must not drift with the
    # dictionary.  Uniform weights are untouched by this.
    if model_weights is None:
        w_model_np = np.ones(len(hkl_j))
    else:
        w_model_np = np.asarray(model_weights, float)
        w_model_np = w_model_np * (len(w_model_np) / np.sum(w_model_np))
    w_model = jnp.asarray(w_model_np)
    # Chunk plan for the (data x model) overlap -- static, so the jitted
    # objective has one shape.  Padding rows carry weight 0 and contribute
    # nothing, which keeps the chunked sum exact rather than approximate.
    n_model_j = len(hkl_j)
    n_chunks, chunk_m, pad_idx = _overlap_chunk_plan(
        n_peaks, n_model_j, max_overlap_elems
    )
    if pad_idx is None:
        w_model_pad = w_model
    else:
        w_model_pad = jnp.asarray(
            np.concatenate([w_model_np, np.zeros(n_chunks * chunk_m - n_model_j)])
        )
        print(
            f"    matching-free refinement: {n_peaks} x {n_model_j} overlap "
            f"chunked into {n_chunks} x ({n_peaks} x {chunk_m}) "
            f"({n_peaks * n_model_j * 4 / 1024**3:.1f} GiB -> "
            f"{n_peaks * chunk_m * 4 / 1024**3:.2f} GiB per temporary)"
        )
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
        if not det_free.all():
            # frozen (out-of-subset) parameters pinned at nominal; skipped
            # entirely when every parameter is free, so the common path is
            # bit-identical to the pre-subset behavior
            det_norm = 0.5 + (det_norm - 0.5) * jnp.asarray(det_free, dtype=jnp.float32)
        i += n_det
        goff = forward_map_param(p[i : i + n_goff], gonio_bound) if n_goff else p[i:i]
        i += n_goff
        extras = {}
        if n_tilt:
            extras["tilt"] = forward_map_param(p[i : i + n_tilt], tilt_bound)
            i += n_tilt
        if n_prun:
            extras["prun"] = forward_map_param(p[i : i + n_prun], per_run_bound)
            i += n_prun
        if n_harm:
            extras["harm"] = forward_map_param(p[i : i + n_harm], harm_bound)
            i += n_harm
        if n_beam:
            extras["beam"] = forward_map_param(
                p[i : i + n_beam], np.deg2rad(beam_bound_deg)
            )
            i += n_beam
        if n_samp:
            extras["samp"] = forward_map_param(p[i : i + n_samp], sample_bound_m)
            i += n_samp
        if n_prt:
            extras["prt"] = forward_map_param(
                p[i : i + n_prt], per_run_trans_bound_m
            ).reshape(-1, 3)
            i += n_prt
        return rotvec, A, det_norm, goff, extras

    def neg_loglik(p):
        rotvec, A, det_norm, goff, extras = unpack(p)
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
            cylinder_axis=cylinder_axis,
            global_rot_axis=global_rot_axis,
        )
        origin = s_off
        if "samp" in extras:
            origin = origin + extras["samp"]
        # peak_lab_xyz by membership matmul (see M_det_T above)
        xyz = (
            M_det_T @ c[0]
            + u_off[:, None] * (M_det_T @ u[0])
            + v_off[:, None] * (M_det_T @ v[0])
            - origin
        )
        if "prt" in extras:
            xyz = xyz - M_run_T @ extras["prt"]
        kf = xyz / jnp.linalg.norm(xyz, axis=1, keepdims=True)
        ki_eff = ki_hat
        if "beam" in extras:
            # the classic refine_beam parameterization: bounded additions to
            # the transverse components, renormalized
            ki_eff = ki_hat + jnp.array([extras["beam"][0], extras["beam"][1], 0.0])
            ki_eff = ki_eff / jnp.linalg.norm(ki_eff)
        g_lab = kf - ki_eff
        d_lab = g_lab / jnp.linalg.norm(g_lab, axis=1, keepdims=True)
        if gonio is not None:
            offs = jnp.asarray(nominal_off).at[np.where(refine_mask)[0]].add(goff)
            axis_dirs = None
            if "tilt" in extras:
                dirs_list = []
                t_i = 0
                for k in range(len(axes)):
                    if k in tilt_axes:
                        e1, e2 = frames_tilt[k]
                        tv = extras["tilt"][2 * t_i] * jnp.asarray(e1) + extras["tilt"][
                            2 * t_i + 1
                        ] * jnp.asarray(e2)
                        dirs_list.append(_rodrigues_jax(tv) @ jnp.asarray(base_dirs[k]))
                        t_i += 1
                    else:
                        dirs_list.append(jnp.asarray(base_dirs[k]))
                axis_dirs = dirs_list

            def _ang_r(r):
                ang = jnp.asarray(angles_deg[r])
                if n_prun:
                    ang = ang.at[int(per_run_axis)].add(extras["prun"][r])
                if n_harm:
                    ang = ang.at[int(harmonics_axis)].add(
                        jnp.asarray(harm_basis[r]) @ extras["harm"]
                    )
                return ang

            R_runs = jnp.stack(
                [
                    gonio_rotation_jax(axes, _ang_r(r), offs, axis_dirs=axis_dirs)
                    for r in range(angles_deg.shape[0])
                ]
            )
            R_rows = (M_run_T @ R_runs.reshape(-1, 9)).reshape(-1, 3, 3)
            d_lab = jnp.einsum("nij,ni->nj", R_rows, d_lab)  # R^T d
        # model side: orientation and (optionally) cell shape
        R = _rodrigues_jax(rotvec) @ jnp.asarray(R0)
        B = (jnp.eye(3) + A) @ jnp.asarray(B0)
        g_vec = hkl_j @ B.T
        dirs = g_vec / jnp.linalg.norm(g_vec, axis=1, keepdims=True)
        if n_chunks == 1:
            # the small case, untouched: one matmul, bit-identical to
            # every result measured before chunking existed
            dots = jnp.abs(d_lab @ (dirs @ R.T).T)  # lines: +- are one direction
            ang2 = 2.0 * (1.0 - jnp.clip(dots, 0.0, 1.0))
            dens = jnp.sum(
                w_model[None, :] * jnp.exp(-ang2 / (2.0 * sigma * sigma)), axis=1
            )
        else:
            dirs_r = (dirs @ R.T)[pad_idx].reshape(n_chunks, chunk_m, 3)
            w_ch = w_model_pad.reshape(n_chunks, chunk_m)

            @jax.checkpoint
            def _acc(carry, packed):
                d_c, w_c = packed
                dots_c = jnp.abs(d_lab @ d_c.T)
                ang2_c = 2.0 * (1.0 - jnp.clip(dots_c, 0.0, 1.0))
                return (
                    carry
                    + jnp.sum(
                        w_c[None, :] * jnp.exp(-ang2_c / (2.0 * sigma * sigma)),
                        axis=1,
                    ),
                    None,
                )

            dens, _ = jax.lax.scan(_acc, jnp.zeros(n_peaks), (dirs_r, w_ch))
        return -jnp.sum(w_data * jnp.log(dens + floor)) / jnp.sum(w_data)

    # Adam with a cosine-decayed step, not a quasi-Newton line search: the
    # objective is float32 (about seven digits), and line searches read
    # that noise floor as convergence long before the minimum.  Bounds by
    # projection; all parameters live in [0, 1] so one rate serves all.
    # The WHOLE loop runs on the device as one lax.scan: the python-loop
    # form synced the loss to the host every step (maxiter round trips) --
    # the same latency pattern measured and removed from refine_local.
    from jax import lax

    n_par = (
        3 + n_cell + n_det + n_goff + n_tilt + n_prun + n_harm + n_beam + n_samp + n_prt
    )
    lr0, beta1, beta2, eps = 0.03, 0.9, 0.999, 1e-8
    lrs = jnp.asarray(
        lr0 * 0.5 * (1.0 + np.cos(np.pi * np.arange(1, int(maxiter) + 1) / maxiter)),
        dtype=jnp.float32,
    )
    ts = jnp.arange(1, int(maxiter) + 1, dtype=jnp.float32)

    @jax.jit
    def run_adam(x0):
        val_grad = jax.value_and_grad(neg_loglik)

        def step(state, sched):
            x, m, v2, best_x, best_v = state
            lr, t = sched
            val, g = val_grad(x)
            better = val < best_v
            best_x = jnp.where(better, x, best_x)
            best_v = jnp.where(better, val, best_v)
            m = beta1 * m + (1 - beta1) * g
            v2 = beta2 * v2 + (1 - beta2) * g * g
            mh = m / (1 - beta1**t)
            vh = v2 / (1 - beta2**t)
            x = jnp.clip(x - lr * mh / (jnp.sqrt(vh) + eps), 0.0, 1.0)
            return (x, m, v2, best_x, best_v), None

        state0 = (
            x0,
            jnp.zeros_like(x0),
            jnp.zeros_like(x0),
            x0,
            jnp.asarray(np.inf, dtype=jnp.float32),
        )
        (x, _, _, best_x, best_v), _ = lax.scan(step, state0, (lrs, ts))
        final_v = neg_loglik(x)
        better = final_v < best_v
        return jnp.where(better, x, best_x), jnp.minimum(final_v, best_v)

    best_x, best_v_j = run_adam(jnp.full(n_par, 0.5, dtype=jnp.float32))
    best_v = float(best_v_j)

    rotvec, A, det_norm, goff, extras = unpack(best_x)
    out = {
        "R": np.asarray(_rodrigues_jax(jnp.asarray(rotvec))) @ np.asarray(R0),
        "B": (np.eye(3) + np.asarray(A)) @ np.asarray(B0),
        "det_params": np.asarray(det_norm),
        "loglik": float(-best_v),
    }
    # The refined geometry itself, not just the normalized parameters that
    # encode it: every consumer downstream (the predictor, the integrator,
    # the overlays, the quality pass) wants panels, not a parameter vector,
    # and regenerating them here -- through the same apply_detector_modes
    # call the objective minimized -- is the only way they cannot drift.
    c_f, u_f, v_f, w_f, h_f, _ = apply_detector_modes(
        jnp.asarray(det_norm)[None, :],
        centers,
        uhats,
        vhats,
        widths,
        heights,
        modes,
        param_slices,
        det_bounds,
        cylinder_axis=cylinder_axis,
        global_rot_axis=global_rot_axis,
    )
    out["panels"] = {
        "centers": np.asarray(c_f[0], dtype=float),
        "uhats": np.asarray(u_f[0], dtype=float),
        "vhats": np.asarray(v_f[0], dtype=float),
        "widths": np.asarray(w_f[0], dtype=float),
        "heights": np.asarray(h_f[0], dtype=float),
    }
    if gonio is not None:
        offs = np.array(nominal_off)
        offs[refine_mask] += np.asarray(goff)
        out["gonio_offsets_deg"] = offs
        if n_tilt:
            tilts = np.rad2deg(np.asarray(extras["tilt"])).reshape(-1, 2)
            out["axis_tilts_deg"] = {k: tilts[i] for i, k in enumerate(tilt_axes)}
            # the tilted axis VECTORS, in the file's own (K, 3 or 4) layout:
            # what goniometer/axes means downstream, with the direction
            # multiplier of column 4 (when present) carried through
            tv_all = np.asarray(extras["tilt"]).reshape(-1, 2)
            dirs_ref = []
            for k in range(len(axes)):
                d_k = np.asarray(base_dirs[k], dtype=float)
                if k in tilt_axes:
                    i_t = tilt_axes.index(k)
                    e1, e2 = frames_tilt[k]
                    tv = tv_all[i_t, 0] * np.asarray(e1) + tv_all[i_t, 1] * np.asarray(
                        e2
                    )
                    d_k = np.asarray(_rodrigues_jax(jnp.asarray(tv)), dtype=float) @ d_k
                dirs_ref.append(d_k)
            axes_ref = np.array(axes, dtype=float)
            axes_ref[:, :3] = np.stack(dirs_ref)
            out["axes_refined"] = axes_ref
        if n_prun:
            out["per_run_deg"] = np.asarray(extras["prun"])
        if n_harm:
            out["harmonics_deg"] = np.asarray(extras["harm"])
            out["harmonics_per_run_deg"] = np.asarray(
                harm_basis @ np.asarray(extras["harm"])
            )
    if n_beam:
        bx, by = np.asarray(extras["beam"])
        ki_eff = np.asarray(ki_hat) + np.array([bx, by, 0.0])
        out["ki_refined"] = ki_eff / np.linalg.norm(ki_eff)
    if n_samp:
        out["sample_offset_m"] = np.asarray(s_off) + np.asarray(extras["samp"])
    if n_prt:
        out["per_run_trans_m"] = np.asarray(extras["prt"])

    # Bound saturation.  Every refinable lives in [0, 1] mapped onto
    # [-bound, +bound], and Adam projects back into the box -- so a
    # parameter that ends AT its bound is a clipped fit, not a converged
    # one: the objective wanted to go further and was not allowed to.
    # Silently, until now.  Measured instance: CG4D's coherent +24 mm
    # radial detector error is 5.4% of the 0.448 m distance, past the 5%
    # default, so the radial mode sat exactly at 0.0500 in every report
    # while the geometry stayed wrong.
    at_bounds = []

    def _check(label, vals, bound, unit, flag):
        if bound is None or not np.isfinite(bound) or bound <= 0:
            return
        v = np.atleast_1d(np.asarray(vals, dtype=float)).reshape(-1)
        for i_v, x in enumerate(v):
            if abs(x) >= 0.98 * bound:
                at_bounds.append(
                    {
                        "what": label,
                        "index": int(i_v),
                        "value": float(x),
                        "bound": float(bound),
                        "unit": unit,
                        "flag": flag,
                    }
                )

    _det_meta = {
        "radial": ("radial", "radial", "--det-radial-bound-frac", ""),
        "cylindrical": ("cylindrical", "radial", "--det-radial-bound-frac", ""),
        "axial_stretch": ("axial_stretch", "radial", "--det-radial-bound-frac", ""),
        "area": ("area", "area", "--det-area-bound-frac", ""),
        "global_rot": ("global_rot", "global_rot", "--det-global-rot-bound-deg", "rad"),
        "global_rot_axis": (
            "global_rot_axis",
            "global_rot_axis",
            "--det-global-rot-bound-deg",
            "rad",
        ),
        "global_trans": (
            "global_trans",
            "global_trans",
            "--det-global-trans-bound",
            "m",
        ),
    }
    dp_out = np.asarray(det_norm)
    for mode_ in modes:
        if mode_ == "independent":
            ip = dp_out[param_slices["independent"]]
            _check(
                "detector independent translation",
                forward_map_param(ip[: n_banks * 3], det_bounds["independent_trans"]),
                det_bounds["independent_trans"],
                "m",
                "--det-trans-bound",
            )
            _check(
                "detector independent tilt",
                forward_map_param(ip[n_banks * 3 :], det_bounds["independent_rot"]),
                det_bounds["independent_rot"],
                "rad",
                "--det-rot-bound-deg",
            )
            continue
        name_, bkey, flag_, unit_ = _det_meta[mode_]
        _check(
            f"detector {name_}",
            forward_map_param(dp_out[param_slices[mode_]], det_bounds[bkey]),
            det_bounds[bkey],
            unit_,
            flag_,
        )
    if gonio is not None:
        _check(
            "goniometer offset",
            np.asarray(goff),
            gonio_bound,
            "deg",
            "--gonio-bound-deg",
        )
        if n_tilt:
            _check(
                "goniometer axis tilt",
                np.asarray(extras["tilt"]),
                tilt_bound,
                "rad",
                "--gonio-axis-vector-bound-deg",
            )
        if n_prun:
            _check(
                "goniometer per-run correction",
                np.asarray(extras["prun"]),
                per_run_bound,
                "deg",
                "--gonio-per-run-bound-deg",
            )
        if n_harm:
            _check(
                "goniometer harmonic coefficient",
                np.asarray(extras["harm"]),
                harm_bound,
                "deg",
                "--gonio-harmonics-bound-deg",
            )
    if n_beam:
        _check(
            "beam tilt",
            np.asarray(extras["beam"]),
            np.deg2rad(beam_bound_deg),
            "rad",
            "--beam-bound-deg",
        )
    if n_samp:
        _check(
            "sample offset",
            np.asarray(extras["samp"]),
            sample_bound_m,
            "m",
            "--sample-bound-m",
        )
    if n_prt:
        _check(
            "per-run sample translation",
            np.asarray(extras["prt"]),
            per_run_trans_bound_m,
            "m",
            "--gonio-per-run-trans-bound-m",
        )
    out["at_bounds"] = at_bounds
    return out
