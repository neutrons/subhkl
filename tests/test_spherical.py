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
    project_points,
    project_rings,
    refine_local,
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
