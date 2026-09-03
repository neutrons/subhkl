"""Orientation search on SO(3): exact identities, then measured recovery.

The identity tests pin every convention (spherical harmonics, Wigner d,
Euler angles, rotation of coefficients) against explicit sums, so the
recovery tests cannot pass by compensating errors.  The recovery tests
measure the claims the module makes: sub-0.5 deg orientations through the
point basis, ring-basis recovery in the small-cell regime where zone
conics dominate, and -- deliberately -- ring-basis *blindness* in the
large-cell regime, so the documented boundary is guarded by a test.
"""

from __future__ import annotations

from itertools import product
from math import factorial

import numpy as np
import pytest

from subhkl.core.crystallography import cartesian_matrix_metric_tensor
from subhkl.search.spherical import (
    _legendre_p0,
    _quat_angle,
    _smoothing,
    correlogram,
    euler_zyz,
    find_orientations,
    ghat_from_kf,
    lattice_directions,
    null_zscore,
    panel_directions,
    project_counts,
    project_points,
    project_rings,
    refine_local,
    refine_matching_free,
    rot_zyz,
    rotate_coeffs,
    so3_inner,
    top_orientations,
    wigner_d_matrix,
    zone_axes,
)


def _rand_rot(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.linalg.det(Q))


def _rand_dirs(rng, n):
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _cubic_rots():
    Rs = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for signs in product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if np.linalg.det(M) > 0.5:
                Rs.append(M)
    return Rs


_D2 = [np.diag(d) for d in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]


def _err_mod(R, R0, group):
    return min(np.rad2deg(_quat_angle(R, R0 @ S)) for S in group)


# ---------------------------------------------------------------------------
# exact identities
# ---------------------------------------------------------------------------


def _d_explicit(j, m, n, beta):
    """Wigner d by the explicit factorial sum -- the definition."""
    tot = 0.0
    for s in range(2 * j + 1):
        if j + n - s < 0 or m - n + s < 0 or j - m - s < 0:
            continue
        tot += (
            (-1) ** (m - n + s)
            / (
                factorial(j + n - s)
                * factorial(s)
                * factorial(m - n + s)
                * factorial(j - m - s)
            )
            * np.cos(beta / 2) ** (2 * j + n - m - 2 * s)
            * np.sin(beta / 2) ** (m - n + 2 * s)
        )
    pref = np.sqrt(
        factorial(j + m) * factorial(j - m) * factorial(j + n) * factorial(j - n)
    )
    return pref * tot


@pytest.mark.parametrize("beta", [0.3, 1.1, 2.7])
def test_wigner_d_matches_the_explicit_sum(beta):
    L = 6
    d = wigner_d_matrix(L, beta)
    for j in range(L + 1):
        for m in range(-j, j + 1):
            for n in range(-j, j + 1):
                assert d[j, m + L, n + L] == pytest.approx(
                    _d_explicit(j, m, n, beta), abs=1e-12
                )


def test_wigner_d_identity_and_orthogonality():
    d0 = wigner_d_matrix(4, 0.0)
    assert np.allclose(d0[3][1:8, 1:8], np.eye(7), atol=1e-12)
    d = wigner_d_matrix(8, 0.77)
    blk = d[5][3:14, 3:14]
    assert np.allclose(blk @ blk.T, np.eye(11), atol=1e-10)


def test_rotation_identity_pins_the_convention():
    """Coefficients of rotated points == rotated coefficients, exactly."""
    rng = np.random.default_rng(0)
    pts = _rand_dirs(rng, 7)
    w = rng.uniform(0.5, 2.0, 7)
    R = rot_zyz(0.7, 1.1, -0.4)
    f = project_points(pts, w, 10, 0.0, even_only=False)
    fd = project_points(pts @ R.T, w, 10, 0.0, even_only=False)
    assert np.max(np.abs(fd - rotate_coeffs(f, R))) < 1e-12


def test_euler_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        R = _rand_rot(rng)
        assert np.allclose(rot_zyz(*euler_zyz(R)), R, atol=1e-10)


def test_correlogram_matches_direct_kernel_sums():
    """SH-domain correlation == direct zonal-kernel sums, for points and
    rings, both at arbitrary rotations and on the readout grid."""
    from numpy.polynomial.legendre import legval

    rng = np.random.default_rng(1)
    L, sig = 16, 0.08
    pts_f, pts_g = _rand_dirs(rng, 5), _rand_dirs(rng, 6)
    wf, wg = rng.uniform(0.5, 2, 5), rng.uniform(0.5, 2, 6)
    f = project_points(pts_f, wf, L, sig, even_only=False)
    g = project_points(pts_g, wg, L, sig, even_only=False)

    def kpoint(t):
        c = _smoothing(L, sig) ** 2 * (2 * np.arange(L + 1) + 1) / (4 * np.pi)
        return legval(np.clip(t, -1, 1), c)

    def kring(t):
        c = (
            _smoothing(L, sig) ** 2
            * _legendre_p0(L)
            * 2
            * np.pi
            * (2 * np.arange(L + 1) + 1)
            / (4 * np.pi)
        )
        return legval(np.clip(t, -1, 1), c)

    R = rot_zyz(0.9, 0.7, -1.3)
    direct = float(np.sum(wf[:, None] * wg[None, :] * kpoint(pts_f @ (pts_g @ R.T).T)))
    assert so3_inner(f, g, R) == pytest.approx(direct, rel=1e-10)

    C, al, be, ga = correlogram(f, g, backend="numpy")
    i, j, k = 3, 2, 5
    assert C[i, j, k] == pytest.approx(
        so3_inner(f, g, rot_zyz(al[i], be[j], ga[k])), rel=1e-10
    )

    axes = _rand_dirs(rng, 3)
    wa = rng.uniform(0.5, 2, 3)
    gr = project_rings(axes, wa, L, sig)
    direct_r = float(np.sum(wf[:, None] * wa[None, :] * kring(pts_f @ (axes @ R.T).T)))
    assert so3_inner(f, gr, R) == pytest.approx(direct_r, rel=1e-10)


def test_ghat_bisector():
    # forward scattering at 2theta about y: Ghat is the unit bisector of
    # kf - ki, and lambda-free -- the Laue collapse this module rests on.
    kf = np.array([[np.sin(0.6), 0.0, np.cos(0.6)]])
    gh = ghat_from_kf(kf)
    expect = kf[0] - [0, 0, 1.0]
    expect /= np.linalg.norm(expect)
    assert np.allclose(gh[0], expect, atol=1e-12)


def test_lattice_directions_are_primitive_and_deduplicated():
    B, _ = cartesian_matrix_metric_tensor(8.0, 8.0, 8.0, *np.deg2rad([90, 90, 90]))
    dirs, w = lattice_directions(B, 1.2)
    # no two directions parallel (primitive + one per +- pair)
    dots = np.abs(dirs @ dirs.T) - np.eye(len(dirs))
    assert np.max(dots) < 1.0 - 1e-9
    assert np.all(w == 1.0)
    # band weights count harmonics: [100] of an 8 A cubic cell has
    # lambda_1 = 2 cos / g0 with g0 = 1/8; a band containing lambda_1/2
    # and lambda_1/3 as well must weight it 3.
    dirs_b, w_b = lattice_directions(
        B, 1.2, wavelength_band=(2.0, 20.0), ki=(0.0, 0.0, 1.0)
    )
    assert len(dirs_b) <= len(dirs)
    assert np.all(w_b >= 1.0)


# ---------------------------------------------------------------------------
# measured recovery
# ---------------------------------------------------------------------------


def _laue_data(rng, dirs, R0, frac, jitter_deg, outlier_frac):
    sel = rng.random(len(dirs)) < frac
    obs = dirs[sel] @ R0.T + rng.normal(
        scale=np.deg2rad(jitter_deg), size=(int(np.sum(sel)), 3)
    )
    obs /= np.linalg.norm(obs, axis=1, keepdims=True)
    out = _rand_dirs(rng, int(outlier_frac * len(obs)))
    return np.vstack([obs, out])


def test_single_crystal_recovery_with_outliers():
    rng = np.random.default_rng(7)
    B, _ = cartesian_matrix_metric_tensor(8.0, 8.0, 8.0, *np.deg2rad([90, 90, 90]))
    dirs, w = lattice_directions(B, 1.2)
    R0 = _rand_rot(rng)
    data = _laue_data(rng, dirs, R0, 0.6, 0.2, 0.3)
    res = find_orientations(
        data, model_dirs=dirs, model_weights=w, kernel_deg=1.5, L=48
    )
    best = res[0]
    assert _err_mod(best["R"], R0, _cubic_rots()) < 0.3
    # lambda = 0 admission: the aggregation over ~300 matched directions is
    # the evidence -- no tuned threshold, yet the peak stands far above the
    # correlogram's own null.  This is the SO(3)-domain form of the
    # implicit-penalty argument, and the domain where it actually holds.
    assert best["z"] > 15.0
    assert best["n_matched"] > 150


def test_two_crystals_are_separated_and_weighted():
    rng = np.random.default_rng(7)
    B, _ = cartesian_matrix_metric_tensor(6.0, 7.5, 9.0, *np.deg2rad([90, 90, 90]))
    dirs, w = lattice_directions(B, 1.3)
    R1, R2 = _rand_rot(rng), _rand_rot(rng)
    d1 = _laue_data(rng, dirs, R1, 0.55, 0.2, 0.1)
    d2 = _laue_data(rng, dirs, R2, 0.40, 0.2, 0.1)
    data = np.vstack([d1, d2])
    res = find_orientations(
        data, model_dirs=dirs, model_weights=w, kernel_deg=1.5, L=64, n_candidates=6
    )
    errs1 = [_err_mod(r["R"], R1, _D2) for r in res]
    errs2 = [_err_mod(r["R"], R2, _D2) for r in res]
    assert min(errs1) < 0.5 and min(errs2) < 0.5
    # the sparse nonnegative stage apportions flux to both, more to the
    # majority crystal
    c1 = res[int(np.argmin(errs1))]["c"]
    c2 = res[int(np.argmin(errs2))]["c"]
    assert c1 > c2 > 0.0
    # symmetry copies were deduplicated by model coherence
    assert all(r["coherence"] <= 0.99 for r in res)


def test_ring_basis_recovers_small_cells():
    """Zone conics dominate when d_min^2/(a c sigma) >~ 1: the ring basis
    alone -- 35 axes, no reflection list -- pins the orientation."""
    rng = np.random.default_rng(11)
    a = 8.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    dirs, _ = lattice_directions(B, 1.2)
    R0 = _rand_rot(rng)
    data = _laue_data(rng, dirs, R0, 0.7, 0.2, 0.25)
    axes, wz = zone_axes(np.diag([a, a, a]), max_index=2)
    L = 64
    sig = np.deg2rad(1.5) / np.sqrt(8 * np.log(2))
    f = project_points(data, None, L, sig)
    g = project_rings(axes, wz, L, sig)
    z0 = so3_inner(f, g, R0) - np.mean(
        [so3_inner(f, g, _rand_rot(rng)) for _ in range(20)]
    )
    z0 /= np.std([so3_inner(f, g, _rand_rot(rng)) for _ in range(20)]) + 1e-12
    assert z0 > 6.0
    C, al, be, ga = correlogram(f, g)
    R, _ = top_orientations(C, al, be, ga, n=1)[0]
    R = refine_local(f, g, R)
    assert _err_mod(R, R0, _cubic_rots()) < 0.5


def test_ring_basis_is_blind_for_large_cells():
    """The documented boundary, guarded: for a large cell the primitive
    directions are uniform to leading order (excess on a zone ~ b/d_min per
    radian vs background ~ a b c/d_min^3 per steradian), so the ring basis
    carries no orientation signal there and the point basis must be used.
    If this test ever starts failing, the boundary in the module docstring
    is wrong and should be re-measured, not the test loosened.
    """
    rng = np.random.default_rng(11)
    a = 120.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    dirs, _ = lattice_directions(B, 2.0)
    R0 = _rand_rot(rng)
    sel = rng.choice(len(dirs), 2000, replace=False)
    obs = dirs[sel] @ R0.T
    axes, wz = zone_axes(np.diag([a, a, a]), max_index=3)
    L = 48
    sig = np.deg2rad(1.0) / np.sqrt(8 * np.log(2))
    f = project_points(obs, None, L, sig)
    g = project_rings(axes, wz, L, sig)
    null = [so3_inner(f, g, _rand_rot(rng)) for _ in range(20)]
    z0 = (so3_inner(f, g, R0) - np.mean(null)) / (np.std(null) + 1e-12)
    assert abs(z0) < 4.0


def test_project_counts_matches_project_points():
    """The chunked raw-count projection is the same operator as the point
    projection -- including negative weights (excess counts go negative
    wherever the frame fluctuates below background)."""
    rng = np.random.default_rng(2)
    d = _rand_dirs(rng, 500)
    w = rng.normal(size=500)
    a = project_counts(d, w, 20, 0.05, even_only=False, chunk=64)
    b = project_points(d, w, 20, 0.05, even_only=False)
    assert np.max(np.abs(a - b)) < 1e-12


def test_orientation_from_raw_counts_below_the_finder_floor():
    """Orientation without any peak finding, from photons a finder cannot use.

    A synthetic detector (theta 40-140 deg, phi 0-180 deg -- deliberately
    partial and anisotropic coverage) records a cubic 8 A pattern where
    every spot carries F = 5 counts against N_bg = 12 background counts per
    footprint: a factor ~3 below the matched-filter floor z sqrt(N_bg), so
    no individual peak is detectable and a find-then-index pipeline has
    nothing to work with.  Projecting the *excess counts* of every pixel
    (background subtracted -- which is also what cancels the anisotropic
    coverage in expectation) still recovers the orientation, because the
    correlogram aggregates every signal photon coherently.  This is the
    bootstrap direction: orientation first, finder second, with the
    predicted directions as its positional prior.
    """
    rng = np.random.default_rng(21)
    nth, nph = 300, 270
    th = np.deg2rad(40 + 100 * (np.arange(nth) + 0.5) / nth)
    ph = np.deg2rad(180 * (np.arange(nph) + 0.5) / nph)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    pix = np.stack(
        [np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1
    ).reshape(-1, 3)
    omega = (np.sin(TH) * np.deg2rad(100 / nth) * np.deg2rad(180 / nph)).ravel()

    B, _ = cartesian_matrix_metric_tensor(8.0, 8.0, 8.0, *np.deg2rad([90, 90, 90]))
    dirs, w = lattice_directions(B, 1.2)
    R0 = _rand_rot(rng)
    spots = np.vstack([dirs, -dirs]) @ R0.T  # both ends of every +- pair
    sig_spot = np.deg2rad(0.5)
    f_peak, b_sky = 5.0, 25000.0  # counts/peak, counts/sr
    n_bg = b_sky * 2 * np.pi * sig_spot**2
    # constructional certificate: individually sub-floor by a wide margin
    assert f_peak < 0.5 * 4.3 * np.sqrt(n_bg)

    rate = b_sky * np.ones(len(pix))
    for shat in spots:
        ct = pix @ shat
        near = ct > np.cos(np.deg2rad(3.0))
        if not np.any(near):
            continue
        ang2 = 2.0 * (1.0 - np.clip(ct[near], -1, 1))
        rate[near] += (
            f_peak / (2 * np.pi * sig_spot**2) * np.exp(-ang2 / (2 * sig_spot**2))
        )
    y = rng.poisson(rate * omega).astype(float)

    excess = y - b_sky * omega
    L = 48
    f = project_counts(pix, excess, L, sig_spot)
    g = project_points(dirs, w, L, sig_spot)
    null = [so3_inner(f, g, _rand_rot(rng)) for _ in range(12)]
    z0 = (so3_inner(f, g, R0) - np.mean(null)) / (np.std(null) + 1e-12)
    assert z0 > 4.0
    C, al, be, ga = correlogram(f, g)
    R, _ = top_orientations(C, al, be, ga, n=1)[0]
    R = refine_local(f, g, R)
    assert _err_mod(R, R0, _cubic_rots()) < 1.5


# ---------------------------------------------------------------------------
# real detector panels and matching-free refinement
# ---------------------------------------------------------------------------


def _flat_panel(center, uhat, vhat, m=96, n=96, width=0.3, height=0.3):
    return {
        "m": m,
        "n": n,
        "width": width,
        "height": height,
        "center": list(center),
        "uhat": list(uhat),
        "vhat": list(vhat),
        "panel": "flat",
    }


def test_panel_directions_match_the_geometry():
    from subhkl.instrument.detector import Detector

    cfg = _flat_panel([0.35, 0.0, 0.35], [0.0, 1.0, 0.0], [-0.7071, 0.0, 0.7071])
    det = Detector(cfg)
    # the peaks path is a subset of the full-frame path, same convention
    all_dirs = panel_directions(det)
    rows = np.array([0, 10, 95])
    cols = np.array([5, 50, 90])
    some = panel_directions(det, rows=rows, cols=cols)
    flat_idx = rows * det.m + cols
    assert np.allclose(some, all_dirs[flat_idx], atol=1e-12)
    # and it is the bisector of the pixel's scattering direction
    xyz = det.pixel_to_lab(rows, cols)
    kf = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    assert np.allclose(some, ghat_from_kf(kf), atol=1e-12)


def test_orientation_from_raw_panel_frames():
    """End to end on real Detector geometry: two flat panels, raw Poisson
    counts, no peak finding -- pooled into one global search."""
    from subhkl.instrument.detector import Detector

    rng = np.random.default_rng(31)
    dets = [
        Detector(
            _flat_panel([0.35, 0.0, 0.35], [0.0, 1.0, 0.0], [-0.7071, 0.0, 0.7071])
        ),
        Detector(
            _flat_panel([-0.35, 0.0, 0.35], [0.0, -1.0, 0.0], [0.7071, 0.0, 0.7071])
        ),
    ]
    B, _ = cartesian_matrix_metric_tensor(8.0, 8.0, 8.0, *np.deg2rad([90, 90, 90]))
    dirs, w = lattice_directions(B, 1.2)
    R0 = _rand_rot(rng)
    spots = np.vstack([dirs, -dirs]) @ R0.T
    sig_spot = np.deg2rad(0.6)

    pix_all, exc_all = [], []
    for det in dets:
        pix = panel_directions(det)
        rate = np.full(len(pix), 0.3)
        for shat in spots:
            ct = pix @ shat
            near = ct > np.cos(np.deg2rad(3.0))
            if np.any(near):
                ang2 = 2.0 * (1.0 - np.clip(ct[near], -1, 1))
                rate[near] += 25.0 * np.exp(-ang2 / (2 * sig_spot**2))
        y = rng.poisson(rate).astype(float)
        pix_all.append(pix)
        exc_all.append(y - 0.3)
    pix = np.vstack(pix_all)
    excess = np.concatenate(exc_all)

    L = 48
    f = project_counts(pix, excess, L, sig_spot)
    g = project_points(dirs, w, L, sig_spot)
    C, al, be, ga = correlogram(f, g)
    R, v = top_orientations(C, al, be, ga, n=1)[0]
    R = refine_local(f, g, R)
    assert _err_mod(R, R0, _cubic_rots()) < 1.0
    assert null_zscore(C, v) > 5.0


def test_matching_free_refinement_recovers_orientation_and_cell():
    """Refinement with no peak list, no indexing labels, no assignment:
    gradient ascent of the correlation itself (jax autodiff through the
    Legendre recursion) recovers a 1.5 deg orientation error to < 0.15 deg
    and a 1% cell-shape strain to a few 1e-3, jointly, in seconds.  Scale
    is structurally invisible (directions are degree-zero homogeneous in B)
    and stays where the parameterization pins it -- traceless strain."""
    rng = np.random.default_rng(9)
    B0, _ = cartesian_matrix_metric_tensor(6.0, 7.5, 9.0, *np.deg2rad([90, 90, 90]))
    A_true = np.array(
        [[0.010, 0.002, 0.0], [0.002, -0.004, 0.001], [0.0, 0.001, -0.006]]
    )
    B_true = (np.eye(3) + A_true) @ B0
    R_true = _rand_rot(rng)

    binv = np.linalg.inv(B0)
    bounds = np.ceil(np.linalg.norm(binv, axis=1) / 1.5).astype(int)
    h, k, l_ = np.meshgrid(*(np.arange(-b, b + 1) for b in bounds), indexing="ij")
    hkl = np.stack([h.ravel(), k.ravel(), l_.ravel()], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    hkl = hkl[np.linalg.norm(hkl @ B0.T, axis=1) <= 1.0 / 1.5]
    gcd = np.gcd.reduce(np.abs(hkl), axis=1)
    hkl = np.unique(hkl // gcd[:, None], axis=0)

    gt = hkl @ B_true.T
    dirs_true = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    sel = rng.random(len(dirs_true)) < 0.65
    obs = dirs_true[sel] @ R_true.T + rng.normal(
        scale=np.deg2rad(0.15), size=(int(np.sum(sel)), 3)
    )
    obs /= np.linalg.norm(obs, axis=1, keepdims=True)
    out = _rand_dirs(rng, int(0.2 * len(obs)))
    data = np.vstack([obs, out])
    L = 48
    f = project_points(data, None, L, np.deg2rad(1.0) / np.sqrt(8 * np.log(2)))

    from subhkl.search.spherical import _rodrigues

    d0 = rng.normal(size=3)
    d0 = d0 / np.linalg.norm(d0) * np.deg2rad(1.5)
    R_start = _rodrigues(d0) @ R_true

    def shape_err(Be, Br):
        G1, G2 = Be @ Be.T, Br @ Br.T
        return np.max(np.abs(G1 / np.trace(G1) - G2 / np.trace(G2))) / np.max(
            np.abs(G2 / np.trace(G2))
        )

    R1, _, c1 = refine_matching_free(f, R_start, hkl, B0, refine_cell=False)
    assert np.rad2deg(_quat_angle(R1, R_true)) < 0.15

    R2, B2, c2 = refine_matching_free(f, R_start, hkl, B0, refine_cell=True)
    assert np.rad2deg(_quat_angle(R2, R_true)) < 0.15
    assert shape_err(B2, B_true) < 0.4 * shape_err(B0, B_true)
    assert c2 > c1  # the cell parameters absorb real signal


def test_wigner_d_batched_equals_scalar():
    """The beta-batched build (the correlogram hot path) is bit-consistent
    with the scalar path it replaced."""
    betas = np.array([0.2, 0.9, 1.7, 2.9])
    batched = wigner_d_matrix(8, betas)
    for j, b in enumerate(betas):
        assert np.allclose(batched[..., j], wigner_d_matrix(8, float(b)), atol=1e-14)


def test_correlogram_jax_matches_numpy_reference():
    """The fused float32 kernel against the float64 reference: same values
    to float32 tolerance, same argmax, at a bandwidth past where the naive
    float32 factorization overflows (the D-recursion keeps every quantity
    O(1); see _correlogram_kernel_jax)."""
    rng = np.random.default_rng(2)
    pf = _rand_dirs(rng, 25)
    pg = _rand_dirs(rng, 35)
    for L, sig in ((16, 0.05), (48, 0.02)):
        f = project_points(pf, rng.uniform(0.5, 2, 25), L, sig, even_only=False)
        g = project_points(pg, rng.uniform(0.5, 2, 35), L, sig, even_only=False)
        C_np, *_ = correlogram(f, g, backend="numpy")
        C_jx, *_ = correlogram(f, g, backend="jax")
        scale = float(np.abs(C_np).max())
        assert np.abs(C_jx - C_np).max() / scale < 2e-3
        assert np.unravel_index(np.argmax(C_np), C_np.shape) == np.unravel_index(
            np.argmax(C_jx), C_jx.shape
        )


def test_project_counts_device_matches_numpy():
    """The float32 device projection against the float64 reference."""
    from subhkl.search.spherical import project_counts_device

    rng = np.random.default_rng(3)
    d = _rand_dirs(rng, 2000)
    W = rng.normal(size=(5, 2000)) * 10
    F_np = project_counts(d, W, 48, 0.02)
    F_dev = project_counts_device(d, W, 48, 0.02)
    scale = np.abs(F_np).max()
    assert np.abs(F_dev - F_np).max() / scale < 2e-3


# ---------------------------------------------------------------------------
# nodal points: the crossings of the zone-ring family
# ---------------------------------------------------------------------------


def test_nodal_points_are_the_zone_crossings():
    """A node is where two zone great circles cross, and that crossing is a
    reciprocal direction: hkl = uvw_1 x uvw_2.  The triples come back
    primitive, sign-canonical and unique, their directions coincide with
    the real-space cross products of the zone axes, the weights are the
    accumulated ring-pair weights, and for a cubic cell the set is its
    own image under the cubic group (which the search relies on when it
    keeps one representative per Laue-equivalent candidate)."""
    from subhkl.search.spherical import nodal_points

    a = 8.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    hkl, w = nodal_points(B, max_index=1)
    assert np.all(np.gcd.reduce(np.abs(hkl), axis=1) == 1)
    assert len(np.unique(hkl, axis=0)) == len(hkl)
    lead = np.where(
        hkl[:, 0] != 0, hkl[:, 0], np.where(hkl[:, 1] != 0, hkl[:, 1], hkl[:, 2])
    )
    assert np.all(lead > 0)
    assert np.all(w > 0)
    for node in ([0, 0, 1], [0, 1, 1], [1, 1, 1]):
        assert np.any(np.all(hkl == node, axis=1))
    # every node lies on the crossing of two zone rings of the family
    axes, wz = zone_axes(np.diag([a, a, a]), max_index=1)
    G = hkl @ B.T
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    crossings = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            c = np.cross(axes[i], axes[j])
            if np.linalg.norm(c) > 1e-9:
                crossings.append(c / np.linalg.norm(c))
    crossings = np.array(crossings)
    assert np.all(np.max(np.abs(dirs @ crossings.T), axis=1) > 1 - 1e-9)
    # [001] collects more ring pairs than [111]: it is the crossing of the
    # <100> zones and of every <110> pair in its plane
    w001 = w[np.all(hkl == [0, 0, 1], axis=1)][0]
    w111 = w[np.all(hkl == [1, 1, 1], axis=1)][0]
    assert w001 > w111
    # cubic invariance of the direction set
    for S in _cubic_rots():
        assert np.all(np.max(np.abs((dirs @ S.T) @ dirs.T), axis=1) > 1 - 1e-9)
    # the dictionary grows with the zone cut, well below the reflection count
    n2 = len(nodal_points(B, max_index=2)[0])
    n3 = len(nodal_points(B, max_index=3)[0])
    assert len(hkl) < n2 < n3
    assert n3 < len(lattice_directions(B, 1.0)[0]) * 4


def test_nodal_dictionary_recovers_the_orientation_matching_free():
    """The nodal points alone -- a few hundred low-index directions, no
    reflection list -- find the orientation from full-resolution data,
    polished without any datum-to-node assignment (most data lie on no
    node), and the matching-free refinement on the nodal TRIPLES carries
    on from there through the same B the reflections use."""
    from subhkl.search.spherical import nodal_points

    rng = np.random.default_rng(17)
    a = 8.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    dirs, _ = lattice_directions(B, 1.0)
    R0 = _rand_rot(rng)
    keep = rng.random(len(dirs)) < 0.7
    obs = dirs[keep] @ R0.T + rng.normal(scale=np.deg2rad(0.2), size=(keep.sum(), 3))
    obs /= np.linalg.norm(obs, axis=1, keepdims=True)
    obs = np.vstack([obs, _rand_dirs(rng, len(obs) // 4)])
    hkl_n, w_n = nodal_points(B, max_index=2)
    G = hkl_n @ B.T
    nodal_dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    assert len(nodal_dirs) < len(dirs) / 2
    res = find_orientations(
        obs,
        model_dirs=nodal_dirs,
        model_weights=w_n,
        kernel_deg=1.0,
        n_candidates=2,
        refine_method="local",
    )
    R = res[0]["R"]
    assert res[0]["n_matched"] == 0  # no assignment was made
    assert res[0]["z"] > 8.0
    assert _err_mod(R, R0, _cubic_rots()) < 0.5
    sig = np.deg2rad(1.0) / np.sqrt(8 * np.log(2))
    f = project_points(obs, None, 64, sig)
    R2, B2, _ = refine_matching_free(f, R, hkl_n, B, weights=w_n, kernel_deg=1.0)
    np.testing.assert_allclose(B2, B)
    assert _err_mod(R2, R0, _cubic_rots()) < 0.3


def test_nearest_line_stats_matches_the_numpy_contraction():
    """The device kernel behind the quality report is the numpy
    (points x model-lines) contraction it replaced, chunked and padded
    or not: the nearest line is found on the device and its angle taken
    in float64 on the host (a float32 dot near 1 quantizes at 0.02 deg),
    and the density is the full von Mises sum with no cutoff -- the
    refinement's objective exactly."""
    from subhkl.search.spherical import nearest_line_stats

    rng = np.random.default_rng(5)
    pts = _rand_dirs(rng, 700)
    dirs = _rand_dirs(rng, 333)
    # put some points very close to a line, where float32 angles fail
    pts[:50] = dirs[:50] + rng.normal(scale=2e-4, size=(50, 3))
    pts[:50] /= np.linalg.norm(pts[:50], axis=1, keepdims=True)
    sigma = np.deg2rad(1.0) / np.sqrt(8 * np.log(2))
    dots = np.clip(np.abs(pts @ dirs.T), 0.0, 1.0)
    dev_ref = np.degrees(np.arccos(dots.max(axis=1)))
    dens_ref = np.sum(np.exp(-2.0 * (1.0 - dots) / (2.0 * sigma * sigma)), axis=1)
    for budget in (1 << 30, 700 * 40, 700 * 37):  # one pass; 9 chunks; padding
        dev, dens = nearest_line_stats(pts, dirs, sigma, max_overlap_elems=budget)
        np.testing.assert_allclose(dev, dev_ref, atol=2e-3)
        # the density is a float32 sum of exp(-ang2 / 2 sigma^2) with
        # 2 sigma^2 = 1e-4: the cancellation in 1 - dot costs ~1e-3
        # relative, the same arithmetic the refinement's objective uses
        np.testing.assert_allclose(dens, dens_ref, rtol=5e-3, atol=1e-5)
        assert np.max(np.abs(dev[:50] - dev_ref[:50])) < 1e-3  # float64 angles
    dev2, none = nearest_line_stats(pts, dirs, sigma, want_density=False)
    assert none is None
    np.testing.assert_allclose(dev2, dev_ref, atol=2e-3)


def test_sph_coeffs_gradient_is_finite_at_high_bandwidth_near_the_poles():
    """The matching-free refinement's objective must have a finite
    gradient at every direction and bandwidth it is run at.  At L = 192
    in float32 the sectoral Legendre chain underflows within ~20 deg of
    the poles, and the recursion's backward pass turned that into NaN --
    L-BFGS then aborted at iteration 0 and the refinement returned its
    input unchanged, silently.  Measured on L1: 3,984 of 65k directions."""
    import jax
    import jax.numpy as jnp

    from subhkl.search.spherical import _sph_coeffs_jax

    rng = np.random.default_rng(2)
    d = _rand_dirs(rng, 400)
    d[:100, :2] *= 0.05  # push a quarter of them to within a few degrees of the pole
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    L = 192
    sigma = np.deg2rad(0.5) / np.sqrt(8 * np.log(2))
    f = jnp.asarray(rng.normal(size=(L + 1, 2 * L + 1)) + 0j, dtype=jnp.complex64)

    def obj(dd):
        g = _sph_coeffs_jax(dd, jnp.ones(len(dd)), L, sigma)
        return -jnp.real(jnp.sum(jnp.conj(f) * g))

    grad = np.asarray(jax.grad(obj)(jnp.asarray(d, dtype=jnp.float32)))
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0)


# ---------------------------------------------------------------------------
# wavelength-band consistency: magnitudes without measuring a wavelength
# ---------------------------------------------------------------------------


def test_band_masks_are_the_laue_harmonic_condition():
    """A spot at |Ghat . ki| = s has |Q| = 2 s / lambda; the lattice offers
    |G| = n g0 along the ray.  For a factor-2 band the condition that some
    harmonic lands in [2s/lambda_max, min(2s/lambda_min, 1/d_min)]
    collapses to g0 <= s below the cut; above 2 s / lambda_max > 1/d_min
    nothing in the model can explain the spot."""
    from subhkl.search.spherical import band_masks, primitive_spacing

    g0 = np.array([0.05, 0.10, 0.20, 0.30, 0.39])
    lam, d_min = (2.0, 4.0), 2.5  # 1/d_min = 0.4
    (m,) = band_masks(g0, [0.15, 0.15 + 1e-9], lam, d_min)  # a single s = 0.15
    np.testing.assert_array_equal(m, g0 <= 0.15)
    (m,) = band_masks(g0, [0.30, 0.30 + 1e-9], lam, d_min)
    np.testing.assert_array_equal(m, g0 <= 0.30)
    # above the cut: s = 0.7 -> window [0.35, 0.40]; g0 = 0.39 (n=1) and
    # 0.20 (n=2 -> 0.40) qualify, 0.30 (0.30, 0.60) does not
    (m,) = band_masks(g0, [0.70, 0.70 + 1e-9], lam, d_min)
    np.testing.assert_array_equal(m, np.array([True, True, True, False, True]))
    # unexplainable: 2 s / lambda_max > 1 / d_min
    (m,) = band_masks(g0, [0.90, 0.90 + 1e-9], lam, d_min)
    assert not m.any()
    # a band is the union over its interior, so the edge case is kept
    (m,) = band_masks(g0, [0.10, 0.31], lam, d_min)
    np.testing.assert_array_equal(m, g0 <= 0.31)
    # primitive spacing reduces harmonics to their primitive vector
    B = np.eye(3) / 10.0
    np.testing.assert_allclose(
        primitive_spacing([[2, 0, 0], [3, 3, 0]], B), [0.1, 0.1 * np.sqrt(2)]
    )


def test_band_consistent_search_recovers_and_sharpens():
    """On a synthetic Laue scene with sin(theta) drawn from the geometry,
    the band-consistent search finds the orientation and its z is no
    lower than the plain search's: restricting every datum to the
    directions that could have produced it removes spurious matches and
    never a true one."""
    from subhkl.search.spherical import primitive_spacing

    rng = np.random.default_rng(3)
    a = 40.0  # a dense direction set, where the plain model starts to blur
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    bounds = int(np.ceil(a / 2.0))
    rng_i = np.arange(-bounds, bounds + 1)
    hkl = np.stack(np.meshgrid(rng_i, rng_i, rng_i, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    G = hkl @ B.T
    gn = np.linalg.norm(G, axis=1)
    keep = gn <= 1.0 / 2.0
    hkl, G, gn = hkl[keep], G[keep], gn[keep]
    dirs = G / gn[:, None]
    key = np.round(dirs * np.where(dirs[:, [0]] < -1e-9, -1, 1), 5)
    _, idx = np.unique(key, axis=0, return_index=True)
    dirs, hkl_u = dirs[idx], hkl[idx]
    g0 = primitive_spacing(hkl_u, B)
    R0 = _rand_rot(rng)
    lam = (2.0, 4.0)
    # observed: reflections excited in band for a beam along +z, with a
    # wavelength drawn per reflection; sin(theta) = |Ghat . ki|
    d_lab = dirs @ R0.T
    s_all = np.abs(d_lab[:, 2])
    lam_r = 2.0 * s_all / np.maximum(gn[idx], 1e-9)  # n = 1 wavelength
    ok = (lam_r >= lam[0]) & (lam_r <= lam[1]) & (d_lab[:, 2] < 0)
    obs = d_lab[ok][rng.random(ok.sum()) < 0.6]
    obs = obs + rng.normal(scale=np.deg2rad(0.15), size=obs.shape)
    obs /= np.linalg.norm(obs, axis=1, keepdims=True)
    noise = _rand_dirs(rng, len(obs) // 2)
    noise[:, 2] = -np.abs(noise[:, 2])
    data = np.vstack([obs, noise])
    sth = np.abs(data[:, 2])
    plain = find_orientations(data, model_dirs=dirs, kernel_deg=1.0, n_candidates=3)
    banded = find_orientations(
        data,
        model_dirs=dirs,
        kernel_deg=1.0,
        n_candidates=3,
        data_sin_theta=sth,
        model_g0=g0,
        wavelength_band=lam,
        d_min=2.0,
    )
    assert (
        _err_mod(banded[0]["R"], R0, _cubic_rots()) < 1.0
    )  # 1 deg kernel on a 40 A cell
    assert banded[0]["z"] >= plain[0]["z"] * 0.9


# ---------------------------------------------------------------------------
# the radial dimension: 3D correlogram through channel stacks
# ---------------------------------------------------------------------------


def test_channel_stacks_collapse_into_the_coupling():
    """A 3D function on the resolution ball is a stack of spherical
    coefficient arrays, one per radial channel; a rotation acts on (l, m)
    alone, so its correlogram is the plain correlogram with the
    per-degree coupling summed over channels.  One channel is the 2D
    case; three channels equal the sum of three 2D correlograms; the
    fused kernel agrees with the numpy reference."""
    from subhkl.search.spherical import so3_inner_stack

    rng = np.random.default_rng(0)
    L, sig = 24, np.deg2rad(2.0)
    f = project_points(_rand_dirs(rng, 50), None, L, sig)
    g = project_points(_rand_dirs(rng, 80), None, L, sig)
    C2, *_ = correlogram(f, g)
    C1, *_ = correlogram(f[None], g[None])
    np.testing.assert_allclose(C1, C2, rtol=1e-4, atol=1e-4 * np.abs(C2).max())
    F = np.stack([project_points(_rand_dirs(rng, 50), None, L, sig) for _ in range(3)])
    G = np.stack([project_points(_rand_dirs(rng, 80), None, L, sig) for _ in range(3)])
    Cj, al, be, ga = correlogram(F, G)
    Cn, *_ = correlogram(F, G, backend="numpy")
    np.testing.assert_allclose(Cj, Cn, rtol=1e-3, atol=1e-3 * np.abs(Cn).max())
    Cs = sum(correlogram(F[k], G[k])[0] for k in range(3))
    np.testing.assert_allclose(Cj, Cs, rtol=1e-3, atol=1e-3 * np.abs(Cn).max())
    R = _rand_rot(rng)
    assert np.isclose(
        so3_inner_stack(F, G, R), sum(so3_inner(F[k], G[k], R) for k in range(3))
    )


def test_radial_basis_is_orthonormal_and_integrates_segments():
    from subhkl.search.spherical import radial_basis

    shell, segment = radial_basis(24, 0.5)
    x, w = np.polynomial.legendre.leggauss(64)
    q, wq = 0.25 * (x + 1), 0.25 * w
    S = shell(q)
    np.testing.assert_allclose((S * wq[:, None]).T @ S, np.eye(24), atol=1e-10)
    # the constant function integrates to its length times its normalization
    assert np.isclose(segment([0.1], [0.3])[0, 0], 0.2 * np.sqrt(1 / 0.5))
    # beyond the ball there is nothing
    assert not segment([0.6], [0.7]).any()
    assert not shell([0.6]).any()


def test_radial_correlogram_recovers_orientation_from_band_segments():
    """Data as Laue segments -- each spot's admissible |Q| range for a
    wavelength band -- and the model as lattice shells at |G|: the 3D
    correlogram must recover the orientation, and its z must not fall
    below the 2D point search's, since the radial overlap can only remove
    spurious matches (the band rule as one inner product)."""
    from subhkl.search.spherical import project_counts_device, radial_basis

    rng = np.random.default_rng(6)
    a = 24.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    bounds = int(np.ceil(a / 2.0))
    r_ = np.arange(-bounds, bounds + 1)
    hkl = np.stack(np.meshgrid(r_, r_, r_, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    G = hkl @ B.T
    gn = np.linalg.norm(G, axis=1)
    keep = gn <= 0.5
    G, gn = G[keep], gn[keep]
    dirs = G / gn[:, None]
    R0 = _rand_rot(rng)
    lam, qmax = (2.0, 4.0), 0.5
    ki = np.array([0.0, 0.0, 1.0])
    d_lab = dirs @ R0.T
    s = np.abs(d_lab @ ki)
    lam_r = 2.0 * s / gn
    ok = (lam_r >= lam[0]) & (lam_r <= lam[1]) & (d_lab @ ki < 0)
    obs = d_lab[ok][rng.random(ok.sum()) < 0.5]
    obs = obs + rng.normal(scale=np.deg2rad(0.2), size=obs.shape)
    obs /= np.linalg.norm(obs, axis=1, keepdims=True)
    noise = _rand_dirs(rng, len(obs))
    noise[:, 2] = -np.abs(noise[:, 2])
    data = np.vstack([obs, noise])
    s_d = np.abs(data @ ki)
    L, sig, N = 48, np.deg2rad(1.5) / np.sqrt(8 * np.log(2)), 16
    shell, segment = radial_basis(N, qmax)
    g3 = project_counts_device(
        np.vstack([dirs, -dirs]), shell(np.concatenate([gn, gn])).T, L, sig
    )
    f3 = project_counts_device(
        data, segment(2 * s_d / lam[1], 2 * s_d / lam[0]).T, L, sig
    )
    g2 = project_points(dirs, None, L, sig)
    f2 = project_points(data, None, L, sig)
    C3, al, be, ga = correlogram(f3, g3)
    C2, *_ = correlogram(f2, g2)
    R3, v3 = top_orientations(C3, al, be, ga, n=1)[0]
    R2, v2 = top_orientations(C2, al, be, ga, n=1)[0]
    assert _err_mod(R3, R0, _cubic_rots()) < 1.5
    assert null_zscore(C3, v3) >= 0.9 * null_zscore(C2, v2)
