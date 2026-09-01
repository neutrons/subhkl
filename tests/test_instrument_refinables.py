"""Shared refinable geometry, and matching-free instrument refinement.

The parameterizations live in subhkl.instrument.refinables and are used by
BOTH the peak-list refinement (VectorizedObjective, verified unchanged by
the indexer test suite) and the spherical matching-free path tested here.

Two identifiability facts are guarded as tests because the first guess is
wrong both ways: a goniometer axis offset is pure gauge with the crystal
orientation whenever every axis inner to it holds constant angles across
the pooled runs (the innermost axis always is -- R_inner constant lets
R_k(delta) commute through and fold into U exactly), and identifiable
precisely when some inner angle varies.  The gauge case must still fit the
*observables* perfectly; only the parameter split is undetermined.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from subhkl.core.crystallography import cartesian_matrix_metric_tensor
from subhkl.instrument.refinables import (
    apply_detector_modes,
    forward_map_param,
    gonio_rotation_jax,
    peak_lab_xyz,
    rodrigues_safe_jax,
)
from subhkl.search.spherical import (
    _quat_angle,
    _rodrigues,
    refine_instrument_matching_free,
)

BOUNDS = {"independent_trans": 0.01, "independent_rot": np.deg2rad(1.0)}
SLICES = {"independent": slice(0, 12)}

CENTERS = np.array([[0.35, 0.0, 0.35], [-0.35, 0.0, 0.35]])
UHATS = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
VHATS = np.array([[-0.7071, 0.0, 0.7071], [0.7071, 0.0, 0.7071]])
WIDTHS = np.array([0.3, 0.3])
HEIGHTS = np.array([0.3, 0.3])
NOMINAL = {
    "centers": CENTERS,
    "uhats": UHATS,
    "vhats": VHATS,
    "widths": WIDTHS,
    "heights": HEIGHTS,
}


def _rand_rot(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.linalg.det(Q))


def _modes(det_norm):
    c, u, v, _w, _h, _a = apply_detector_modes(
        jnp.asarray(det_norm)[None],
        jnp.asarray(CENTERS)[None],
        jnp.asarray(UHATS)[None],
        jnp.asarray(VHATS)[None],
        jnp.asarray(WIDTHS)[None],
        jnp.asarray(HEIGHTS)[None],
        ("independent",),
        SLICES,
        BOUNDS,
    )
    return np.asarray(c[0]), np.asarray(u[0]), np.asarray(v[0])


def _hkl(B, dmin):
    binv = np.linalg.inv(B)
    b_ = np.ceil(np.linalg.norm(binv, axis=1) / dmin).astype(int)
    h, k, l_ = np.meshgrid(*(np.arange(-x, x + 1) for x in b_), indexing="ij")
    hkl = np.stack([h.ravel(), k.ravel(), l_.ravel()], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    hkl = hkl[np.linalg.norm(hkl @ B.T, axis=1) <= 1.0 / dmin]
    g = np.gcd.reduce(np.abs(hkl), axis=1)
    prim = hkl // g[:, None]
    lead = np.where(
        prim[:, 0] != 0, prim[:, 0], np.where(prim[:, 1] != 0, prim[:, 1], prim[:, 2])
    )
    return np.unique(prim * np.where(lead < 0, -1, 1)[:, None], axis=0)


def _scene(rng, ct, ut, vt, axes, angles, off_true, B0, hkl, R_true, jitter=2e-4):
    """Peaks (bank, u_off, v_off, run) from rays through the TRUE geometry."""
    H = np.vstack([hkl, -hkl]) @ B0.T
    Hd = H / np.linalg.norm(H, axis=1, keepdims=True)
    recs = {"det_idx": [], "u_off": [], "v_off": [], "run_idx": []}
    for r in range(len(angles)):
        Rrun = np.asarray(
            gonio_rotation_jax(
                jnp.asarray(axes), jnp.asarray(angles[r]), jnp.asarray(off_true)
            )
        )
        d_lab = Hd @ (Rrun @ R_true).T
        d_lab = d_lab[d_lab[:, 2] < -0.05]
        kf = np.array([0.0, 0.0, 1.0]) - 2 * d_lab[:, 2:3] * d_lab
        kf /= np.linalg.norm(kf, axis=1, keepdims=True)
        for b in range(2):
            n = np.cross(ut[b], vt[b])
            t = (ct[b] @ n) / (kf @ n)
            ok = t > 0
            xyz = t[:, None] * kf
            uo = (xyz - ct[b]) @ ut[b]
            vo = (xyz - ct[b]) @ vt[b]
            ok &= (np.abs(uo) < 0.48 * WIDTHS[b]) & (np.abs(vo) < 0.48 * HEIGHTS[b])
            recs["det_idx"].append(np.full(ok.sum(), b))
            recs["u_off"].append(uo[ok] + rng.normal(scale=jitter, size=ok.sum()))
            recs["v_off"].append(vo[ok] + rng.normal(scale=jitter, size=ok.sum()))
            recs["run_idx"].append(np.full(ok.sum(), r))
    return {k: np.concatenate(v) for k, v in recs.items()}


def test_apply_detector_modes_nominal_is_identity():
    c, u, v = _modes(np.full(12, 0.5))
    assert np.allclose(c, CENTERS, atol=1e-7)
    assert np.allclose(u, UHATS, atol=1e-7)
    assert np.allclose(v, VHATS, atol=1e-7)
    # and the peak reconstruction matches direct arithmetic
    xyz = peak_lab_xyz(
        jnp.asarray(CENTERS)[None],
        jnp.asarray(UHATS)[None],
        jnp.asarray(VHATS)[None],
        np.array([0, 1]),
        jnp.asarray([0.02, -0.05]),
        jnp.asarray([0.01, 0.03]),
    )[0]
    expect = (
        CENTERS
        + np.array([[0.02], [-0.05]]) * UHATS
        + np.array([[0.01], [0.03]]) * VHATS
    )
    assert np.allclose(np.asarray(xyz), expect, atol=1e-7)


def test_panels_recovered_matching_free():
    """Per-bank translation and tilt, through the shared parameterization,
    with no peak list matching anywhere."""
    rng = np.random.default_rng(17)
    det_norm_true = np.full(12, 0.5)
    trans_true = np.array([2.0e-3, -1.0e-3, 1.5e-3])
    tilt_true = np.deg2rad([0.2, -0.3, 0.1])
    det_norm_true[0:3] = 0.5 + trans_true / (2 * BOUNDS["independent_trans"])
    det_norm_true[9:12] = 0.5 + tilt_true / (2 * BOUNDS["independent_rot"])
    ct, ut, vt = _modes(det_norm_true)
    axes = np.array([[0.0, 0.0, 1.0, 1.0]])
    angles = np.array([[0.0]])
    B0, _ = cartesian_matrix_metric_tensor(8.0, 8.0, 8.0, *np.deg2rad([90, 90, 90]))
    hkl = _hkl(B0, 1.2)
    R_true = _rand_rot(rng)
    peaks = _scene(rng, ct, ut, vt, axes, angles, np.zeros(1), B0, hkl, R_true)
    peaks = {k: v for k, v in peaks.items() if k != "run_idx"}
    R_start = R_true @ _rodrigues(np.deg2rad(0.4) * np.array([0.5, 0.5, -0.7]) / 0.99)
    out = refine_instrument_matching_free(
        peaks, NOMINAL, hkl, B0, R_start, det_bounds=BOUNDS, kernel_deg=1.0, maxiter=400
    )
    trans_err = np.abs(
        forward_map_param(out["det_params"][0:3], BOUNDS["independent_trans"])
        - trans_true
    )
    tilt_err = np.abs(
        forward_map_param(out["det_params"][9:12], BOUNDS["independent_rot"])
        - tilt_true
    )
    assert np.all(trans_err < 1.5e-3)
    assert np.all(tilt_err < np.deg2rad(0.25))
    assert np.rad2deg(_quat_angle(out["R"], R_true)) < 0.15


def test_gonio_offset_identifiable_when_inner_angles_vary():
    rng = np.random.default_rng(17)
    ct, ut, vt = _modes(np.full(12, 0.5))
    axes = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    angles = np.array([[15.0, 0.0], [15.0, 30.0], [15.0, 60.0]])
    off_true = np.array([0.7, 0.0])  # on the OUTER axis; inner varies
    B0, _ = cartesian_matrix_metric_tensor(8.0, 8.0, 8.0, *np.deg2rad([90, 90, 90]))
    hkl = _hkl(B0, 1.2)
    R_true = _rand_rot(rng)
    peaks = _scene(rng, ct, ut, vt, axes, angles, off_true, B0, hkl, R_true)
    R_start = R_true @ _rodrigues(np.deg2rad(0.4) * np.array([0.5, 0.5, -0.7]) / 0.99)
    out = refine_instrument_matching_free(
        peaks,
        NOMINAL,
        hkl,
        B0,
        R_start,
        det_bounds=BOUNDS,
        modes=(),
        gonio={
            "axes": axes,
            "angles_deg": angles,
            "refine_mask": np.array([True, False]),
            "bound_deg": 2.0,
        },
        kernel_deg=1.0,
        maxiter=400,
    )
    assert abs(out["gonio_offsets_deg"][0] - 0.7) < 0.1
    assert np.rad2deg(_quat_angle(out["R"], R_true)) < 0.3


def test_innermost_gonio_offset_is_gauge_but_observables_still_fit():
    """The gauge theorem, measured: an innermost-axis offset cannot be
    recovered (it folds into U exactly), yet the refined combination must
    reproduce every observed direction.  If this test ever recovers the
    offset, the composition convention changed -- investigate, don't relax.
    """
    rng = np.random.default_rng(17)
    ct, ut, vt = _modes(np.full(12, 0.5))
    axes = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    angles = np.array([[0.0, 10.0], [25.0, 10.0], [50.0, 10.0]])
    off_true = np.array([0.0, 0.6])  # innermost: pure gauge
    B0, _ = cartesian_matrix_metric_tensor(8.0, 8.0, 8.0, *np.deg2rad([90, 90, 90]))
    hkl = _hkl(B0, 1.2)
    R_true = _rand_rot(rng)
    peaks = _scene(rng, ct, ut, vt, axes, angles, off_true, B0, hkl, R_true)
    R_start = R_true @ _rodrigues(np.deg2rad(0.4) * np.array([0.5, 0.5, -0.7]) / 0.99)
    out = refine_instrument_matching_free(
        peaks,
        NOMINAL,
        hkl,
        B0,
        R_start,
        det_bounds=BOUNDS,
        modes=(),
        gonio={
            "axes": axes,
            "angles_deg": angles,
            "refine_mask": np.array([False, True]),
            "bound_deg": 2.0,
        },
        kernel_deg=1.0,
        maxiter=400,
    )
    # the SPLIT is undetermined, but the gauge relation must hold: the
    # unrecovered offset reappears as a rotation of U about the inner axis
    resid = np.deg2rad(out["gonio_offsets_deg"][1] - 0.6)
    R_gauge = np.asarray(rodrigues_safe_jax(jnp.asarray([-resid, 0.0, 0.0])))
    # 0.3 deg: the gauge relation is exact in the continuum, but the finite
    # Adam run and GPU float32 nondeterminism leave a ~0.15 deg residual
    # that wobbles run to run (measured 0.13-0.17 on identical input).
    assert np.rad2deg(_quat_angle(out["R"], R_gauge @ R_true)) < 0.3
