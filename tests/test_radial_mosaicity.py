"""Radial (scattering-plane) mosaicity model for the RBF integrator.

The isotropic mosaicity tensor eta^2 I projects to an isotropic 2D
blur, and the projected sample ellipsoid's orientation is set by the
goniometer projection -- neither term can represent the radial-along-
2-theta streak that Laue spots actually have (measured 3.7x on
cg4d-t4-lysozyme; the fitted sample-model major axis sat 60 deg from
the streak direction, worse than random).  mosaicity_radial=True
replaces the isotropic term with a rank-1 streak along the projected
per-peak radial direction, eta^2 (D P r)(D P r)^T.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from subhkl.search.sparse_rbf import build_3d_cov, optimize_global_crystal

PITCH_INV = 2000.0  # pixels per meter
DIST = 0.4  # sample-detector distance, meters


def _make_patches(phis, s_px=1.5, eta=0.005, amp=80.0, size=15, seed=3):
    """Streaked Gaussian patches, streak angle phi per patch (in-plane)."""
    rng = np.random.default_rng(seed)
    half = size // 2
    dr, dc = np.meshgrid(
        np.arange(-half, half + 1), np.arange(-half, half + 1), indexing="ij"
    )
    P = PITCH_INV * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    patches, drs, dcs, P_mats, dists, R_mats, streaks = [], [], [], [], [], [], []
    for phi in phis:
        streak3 = np.array([np.cos(phi), np.sin(phi), 0.0])
        e_pix = P @ (DIST * streak3)  # (u=col, v=row) pixels per radian
        C = s_px**2 * np.eye(2) + eta**2 * np.outer(e_pix, e_pix)
        Ci = np.linalg.inv(C)
        x = np.stack([dc, dr])  # index 0 = column = u
        quad = Ci[0, 0] * x[0] ** 2 + 2 * Ci[0, 1] * x[0] * x[1] + Ci[1, 1] * x[1] ** 2
        patch = amp * np.exp(-0.5 * quad) + rng.normal(0, 0.3, size=(size, size))
        patches.append(patch)
        drs.append(dr)
        dcs.append(dc)
        P_mats.append(P)
        dists.append(DIST)
        R_mats.append(np.eye(3))
        streaks.append(streak3)
    return tuple(
        jnp.array(np.array(a))
        for a in (
            patches,
            np.zeros(len(phis)),
            drs,
            dcs,
            P_mats,
            dists,
            R_mats,
            streaks,
        )
    )


def _project(res_x, P, D, streak3, radial):
    Sigma = np.array(build_3d_cov(jnp.array(res_x[:6])))
    if radial:
        S2 = P @ Sigma @ P.T
        e = P @ (D * streak3)
        return S2 + (abs(res_x[6]) + 1e-6) ** 2 * np.outer(e, e)
    S3 = Sigma + D**2 * np.eye(3) * (abs(res_x[6]) + 1e-6) ** 2
    return P @ S3 @ P.T


def test_radial_mode_recovers_the_streak_where_isotropic_cannot():
    phis = np.deg2rad(np.linspace(0, 160, 24))
    args = _make_patches(phis)

    x_iso = optimize_global_crystal(*args, fit_mosaicity=True, mosaicity_radial=False)
    x_rad = optimize_global_crystal(*args, fit_mosaicity=True, mosaicity_radial=True)

    # Recovered mosaicity: true eta = 5 mrad.
    assert abs(abs(x_rad[6]) - 0.005) < 0.0015

    P = PITCH_INV * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mis_rad, mis_iso, ratio_rad = [], [], []
    for phi in phis:
        streak3 = np.array([np.cos(phi), np.sin(phi), 0.0])
        e = P @ (DIST * streak3)
        e = e / np.linalg.norm(e)
        for res_x, radial, sink in ((x_rad, True, mis_rad), (x_iso, False, mis_iso)):
            C = _project(res_x, P, DIST, streak3, radial)
            w, V = np.linalg.eigh(C)
            major = V[:, np.argmax(w)]
            sink.append(np.rad2deg(np.arccos(np.clip(abs(major @ e), 0.0, 1.0))))
            if radial:
                ratio_rad.append(np.sqrt(w.max() / max(w.min(), 1e-12)))

    # The radial model tracks every streak orientation; the isotropic +
    # sample-shape family has ONE orientation for all patches (R = I
    # here), so it must miss most of them.
    assert np.median(mis_rad) < 5.0
    assert np.median(mis_iso) > 20.0
    # True aspect: sqrt(s^2 + (eta*800)^2)/s = sqrt(1.5^2+4^2)/1.5 ~ 2.85
    assert 2.0 < np.median(ratio_rad) < 4.0


def test_spherical_shape_plus_streak_matches_the_full_model():
    """The hypothesis test: if the data are core + streak, constraining
    the sample tensor to a sphere must cost nothing."""
    from subhkl.search.sparse_rbf import val_and_grad_fn
    import jax.numpy as jnp_

    phis = np.deg2rad(np.linspace(10, 170, 16))
    args = _make_patches(phis)

    x_full = optimize_global_crystal(*args, fit_mosaicity=True, mosaicity_radial=True)
    x_sph = optimize_global_crystal(
        *args, fit_mosaicity=True, mosaicity_radial=True, shape_spherical=True
    )

    def mse(x, spherical):
        val, _ = val_and_grad_fn(
            jnp_.array(x),
            *args,
            args[0].shape[-1],
            fit_mosaicity=True,
            mosaicity_radial=True,
            shape_spherical=spherical,
        )
        return float(val)

    m_full, m_sph = mse(x_full, False), mse(x_sph, True)
    assert m_sph < 1.1 * m_full
    # Recovered isotropic core: s = 1.5 px at 2000 px/m -> 7.5e-4 m.
    assert abs(abs(x_sph[0]) - 7.5e-4) < 2.5e-4
    assert abs(abs(x_sph[6]) - 0.005) < 0.0015


def test_normalized_shape_fit_resists_a_bright_outlier():
    """One 100x-brighter patch with a rotated streak must not steer the
    normalized fit away from the 15 typical patches."""
    phis = np.deg2rad(np.linspace(10, 150, 15))
    args = list(_make_patches(phis))
    # append a bright outlier streaked perpendicular to its recorded frame
    out = _make_patches(np.deg2rad([60.0]), amp=8000.0, seed=11)
    wrong_streak = np.array(
        [[np.cos(np.deg2rad(150.0)), np.sin(np.deg2rad(150.0)), 0.0]]
    )
    import jax.numpy as jnp_

    args = [
        jnp_.concatenate([a, (b if i != 7 else jnp_.array(wrong_streak))])
        for i, (a, b) in enumerate(zip(args, out))
    ]
    x_raw = optimize_global_crystal(*args, fit_mosaicity=True, mosaicity_radial=True)
    x_norm = optimize_global_crystal(
        *args, fit_mosaicity=True, mosaicity_radial=True, shape_fit_normalized=True
    )
    # the normalized eta stays near the typical 5 mrad; the raw fit is
    # dragged by the outlier's misfit
    assert abs(abs(x_norm[6]) - 0.005) < abs(abs(x_raw[6]) - 0.005) + 1e-9
    assert abs(abs(x_norm[6]) - 0.005) < 0.002
