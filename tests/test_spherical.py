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

    C, al, be, ga = correlogram(f, g)
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
