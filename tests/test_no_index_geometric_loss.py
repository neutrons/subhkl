"""The --no-index loss must compare predicted and observed directions.

The old geometric loss penalized || q_obs - q_pred || with q_obs built
from the wavelength carried verbatim out of the bootstrap file.  Because
that wavelength was itself fitted under the bootstrap geometry, the loss
could trade detector distance against lattice scale exactly like the
free-wavelength indexing loss (measured on cg4d-t4-lysozyme: +6%
detector radius against -5% lattice at a cost of 0.005 deg of median
angular error).  The rewritten loss takes the wavelength from the
elastic condition under the CURRENT geometry, lam = -2 (ki.G)/|G|^2,
and penalizes the chord |kf_pred - kf_obs| between unit vectors -- the
same observable the metrics report as the angular error.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.optimization import VectorizedObjective


def _make_peaks(n=60, seed=7):
    rng = np.random.default_rng(seed)
    B = np.eye(3) / 6.0
    ki = np.array([0.0, 0.0, 1.0])
    hkls, lams, kfs = [], [], []
    while len(kfs) < n:
        hkl = rng.integers(-4, 5, size=3)
        if not np.any(hkl):
            continue
        G = B @ hkl
        lam = -2.0 * (ki @ G) / (G @ G)
        if not (1.0 <= lam <= 4.0):
            continue
        hkls.append(hkl)
        lams.append(lam)
        kfs.append(ki + lam * G)
    return B, ki, np.array(hkls), np.array(lams), np.array(kfs)


def _objective(B, ki, hkls, lams, kfs):
    return VectorizedObjective(
        B,
        (kfs - ki).T,
        None,
        np.array([1.0, 4.0]),
        beam_nominal=ki,
        kf_lab_fixed_vectors=(kfs - ki).T,
        no_index=True,
        hkl_fixed=hkls.T.astype(float),
        lambda_fixed=lams,
    )


def test_truth_has_zero_residual_and_elastic_wavelengths():
    B, ki, hkls, lams, kfs = _make_peaks()
    obj = _objective(B, ki, hkls, lams, kfs)
    loss, dist, hkl, lam = obj.get_results(np.zeros((1, 3)))
    np.testing.assert_allclose(np.array(dist[0]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.array(loss[0]), 0.0, atol=1e-6)
    # The returned wavelengths come from the elastic condition, not the
    # bootstrap: at the true geometry they coincide with the truth.
    np.testing.assert_allclose(np.array(lam[0]), lams, rtol=1e-5)
    np.testing.assert_allclose(np.array(hkl[0]), hkls, atol=1e-6)


def test_residual_is_the_angular_chord_of_the_observation_error():
    B, ki, hkls, lams, kfs = _make_peaks()
    delta = np.deg2rad(0.5)
    R = Rotation.from_rotvec(delta * np.array([1.0, 0.0, 0.0])).as_matrix()
    kfs_moved = kfs @ R.T  # a rigid error in every observed direction

    obj = _objective(B, ki, hkls, lams, kfs_moved)
    _, dist, _, _ = obj.get_results(np.zeros((1, 3)))
    dist = np.array(dist[0])

    # A rotation by delta moves a unit vector along a chord of
    # 2 sin(delta/2) times its sine with the rotation axis -- exact.
    sin_axis = np.sqrt(np.clip(1.0 - kfs[:, 0] ** 2, 0.0, None))
    expected = 2.0 * np.sin(delta / 2.0) * sin_axis
    np.testing.assert_allclose(dist, expected, atol=1e-6)


def test_lattice_rescale_cannot_absorb_a_detector_error():
    B, ki, hkls, lams, kfs = _make_peaks()
    delta = np.deg2rad(0.5)
    R = Rotation.from_rotvec(delta * np.array([0.0, 1.0, 0.0])).as_matrix()
    obj = _objective(B, ki, hkls, lams, kfs @ R.T)

    kf_ki = np.asarray(obj.kf_lab_fixed) - ki[:, None]  # unit kf minus ki
    kf_ki = kf_ki[None, :, :]
    ki_s = np.broadcast_to(ki[None, :, None], kf_ki.shape)

    loss_1, dist_1, _, lam_1 = obj.geometric_loss_jax(B[None], kf_ki, ki_s)
    scale = 1.06
    loss_s, dist_s, _, lam_s = obj.geometric_loss_jax(scale * B[None], kf_ki, ki_s)

    # G -> sG, lam -> lam/s, kf_pred = ki + lam G invariant: the rescale
    # changes nothing about the residual -- this is the degeneracy the
    # old fixed-lambda q-space loss slid along.
    np.testing.assert_allclose(np.array(dist_s), np.array(dist_1), atol=1e-6)
    np.testing.assert_allclose(np.array(loss_s), np.array(loss_1), atol=1e-7)
    np.testing.assert_allclose(np.array(lam_s), np.array(lam_1) / scale, rtol=1e-5)
    assert np.array(loss_1)[0] > np.deg2rad(0.3)


def test_unassigned_peaks_are_excluded_from_the_loss_and_fail_the_cut():
    B, ki, hkls, lams, kfs = _make_peaks()
    hkls_z = hkls.copy()
    hkls_z[:5] = 0
    obj = _objective(B, ki, hkls_z, lams, kfs)
    loss, dist, _, lam = obj.get_results(np.zeros((1, 3)))
    dist, lam = np.array(dist[0]), np.array(lam[0])

    # hkl = 0 gives lam = 0 and kf_pred = ki, so the reported residual is
    # |q_obs| = 2 sin(theta) -- well above the 1 deg cut -- and because
    # the detector and beam parameters can shrink that quantity, such
    # peaks must not contribute to the loss: at the true geometry the
    # loss is zero despite five peaks with large residuals.
    np.testing.assert_allclose(lam[:5], 0.0, atol=1e-9)
    q_norm = np.linalg.norm(kfs[:5] - ki, axis=1)
    np.testing.assert_allclose(dist[:5], q_norm, atol=1e-6)
    assert (dist[:5] > np.deg2rad(1.0)).all()
    np.testing.assert_allclose(dist[5:], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.array(loss[0]), 0.0, atol=1e-6)


def _rotate_each(kfs, ki, direction, delta):
    """Rotate each kf about its per-peak frame axis: 'radial' displacement
    rotates about t = ki x kf (in-plane, 2-theta change), 'tangential'
    rotates about r = t x kf (out-of-plane)."""
    out = np.empty_like(kfs)
    for i, kf in enumerate(kfs):
        t = np.cross(ki, kf)
        t /= np.linalg.norm(t)
        axis = t if direction == "radial" else np.cross(t, kf)
        out[i] = Rotation.from_rotvec(delta * axis).as_matrix() @ kf
    return out


def test_radial_weight_suppresses_only_the_radial_component():
    B, ki, hkls, lams, kfs = _make_peaks()
    delta = np.deg2rad(0.5)
    chord = 2.0 * np.sin(delta / 2.0)

    # At w = 0 the radial direction is discarded and only the chord's
    # second-order longitudinal component (chord^2 / 2) survives.
    floor = chord**2 / 2.0
    for w, direction, expected in (
        (0.25, "radial", 0.25 * chord),
        (0.25, "tangential", chord),
        (0.0, "radial", floor),
        (0.0, "tangential", chord),
    ):
        moved = _rotate_each(kfs, ki, direction, delta)
        obj = VectorizedObjective(
            B,
            (moved - ki).T,
            None,
            np.array([1.0, 4.0]),
            beam_nominal=ki,
            kf_lab_fixed_vectors=(moved - ki).T,
            no_index=True,
            hkl_fixed=hkls.T.astype(float),
            lambda_fixed=lams,
            radial_weight=w,
        )
        _, dist, _, _ = obj.get_results(np.zeros((1, 3)))
        np.testing.assert_allclose(np.array(dist[0]), expected, rtol=6e-3, atol=1e-6)


def test_constant_polynomial_weight_of_one_is_the_plain_chord():
    B, ki, hkls, lams, kfs = _make_peaks()
    delta = np.deg2rad(0.4)
    moved = _rotate_each(kfs, ki, "radial", delta)

    def build(**kw):
        return VectorizedObjective(
            B,
            (moved - ki).T,
            None,
            np.array([1.0, 4.0]),
            beam_nominal=ki,
            kf_lab_fixed_vectors=(moved - ki).T,
            no_index=True,
            hkl_fixed=hkls.T.astype(float),
            lambda_fixed=lams,
            **kw,
        )

    _, d_plain, _, _ = build().get_results(np.zeros((1, 3)))
    _, d_poly, _, _ = build(radial_weight_poly=[1.0]).get_results(np.zeros((1, 3)))
    # The decomposed weighted form at w = 1 must equal the isotropic
    # chord exactly: t, r, kf form a complete orthonormal frame.
    np.testing.assert_allclose(np.array(d_poly[0]), np.array(d_plain[0]), atol=1e-7)


def test_wavelength_polynomial_applies_the_weight_per_peak():
    B, ki, hkls, lams, kfs = _make_peaks()
    delta = np.deg2rad(0.5)
    chord = 2.0 * np.sin(delta / 2.0)
    moved = _rotate_each(kfs, ki, "radial", delta)
    poly = [0.2, -0.3]  # w(lam) = 0.2 lam - 0.3, clipped to [0, 1]
    obj = VectorizedObjective(
        B,
        (moved - ki).T,
        None,
        np.array([1.0, 4.0]),
        beam_nominal=ki,
        kf_lab_fixed_vectors=(moved - ki).T,
        no_index=True,
        hkl_fixed=hkls.T.astype(float),
        lambda_fixed=lams,
        radial_weight_poly=poly,
    )
    _, dist, _, lam = obj.get_results(np.zeros((1, 3)))
    w = np.clip(np.polyval(poly, np.array(lam[0])), 0.0, 1.0)
    expected = np.sqrt((chord * w) ** 2 + (chord**2 / 2.0) ** 2)
    np.testing.assert_allclose(np.array(dist[0]), expected, rtol=6e-3, atol=1e-6)
