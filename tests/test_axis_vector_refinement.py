"""Goniometer axis-vector refinement: the direction, not just the zero point.

A goniometer mounted at a small angle to the detector frame tilts every
rotation axis -- an error the angular offsets and translations can only
chase degenerately (measured on cg4d-t4-lysozyme: widening their bounds
made the fit slide to the new box edges and index fewer peaks).  These
tests pin the parametrization against an independent rotation
implementation and show the objective separates the true tilt from the
nominal geometry on synthetic peaks it could not index otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.optimization import VectorizedObjective


def _tilted_axis(n, e1, e2, a_deg, b_deg):
    d = n + np.tan(np.deg2rad(a_deg)) * e1 + np.tan(np.deg2rad(b_deg)) * e2
    return d / np.linalg.norm(d)


def _basis(n):
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def _make_fixture(tilt_a=1.2, tilt_b=-0.8, bound_deg=2.0, n_frames=10, seed=3):
    """Synthetic elastic peaks from a crystal rotating about a TILTED omega.

    kf = ki + lambda * G with |kf| = |ki| = 1 (Bragg's condition solved per
    reflection), G = R_true(frame) . UB . hkl -- physically exact, no small-
    angle shortcuts anywhere.
    """
    rng = np.random.default_rng(seed)
    a_cell = 6.0
    B = np.eye(3) / a_cell
    UB = B  # U = I: the orientation part of x is zero
    n_nom = np.array([0.0, 1.0, 0.0])
    e1, e2 = _basis(n_nom)
    n_true = _tilted_axis(n_nom, e1, e2, tilt_a, tilt_b)
    ki = np.array([0.0, 0.0, 1.0])

    angles = np.arange(n_frames) * 14.0
    kfs, runs, hkls = [], [], []
    while len(kfs) < 80:
        hkl = rng.integers(-4, 5, size=3)
        if not np.any(hkl):
            continue
        frame = rng.integers(0, n_frames)
        R_true = Rotation.from_rotvec(np.deg2rad(angles[frame]) * n_true).as_matrix()
        G = R_true @ (UB @ hkl)
        lam = -2.0 * (ki @ G) / (G @ G)
        if not (1.0 <= lam <= 4.0):
            continue
        kfs.append(ki + lam * G)
        runs.append(frame)
        hkls.append(hkl)
    kf = np.array(kfs).T  # (3, N)

    obj = VectorizedObjective(
        B,
        kf - ki[:, None],
        None,
        np.array([1.0, 4.0]),
        goniometer_axes=np.array([[0.0, 1.0, 0.0, 1.0]]),
        goniometer_angles=angles[None, :],
        motor_map=[0],
        goniometer_axis_vector_mask=np.array([True]),
        goniometer_axis_vector_bound_deg=bound_deg,
        beam_nominal=ki,
        kf_lab_fixed_vectors=kf - ki[:, None],
        peak_run_indices=np.array(runs),
    )
    # Normalized truth: tilt = norm * 2b - b  =>  norm = (tilt + b) / (2b)
    x_truth = np.array(
        [0.0, 0.0, 0.0]
        + [
            (tilt_a + bound_deg) / (2 * bound_deg),
            (tilt_b + bound_deg) / (2 * bound_deg),
        ]
    )
    x_nominal = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
    return obj, x_truth, x_nominal, n_true, angles


def test_tilted_axis_rotation_matches_an_independent_implementation():
    obj, x_truth, _, n_true, angles = _make_fixture()
    out = obj._get_physical_params_jax(x_truth[None, :])
    R_cum, axis_dirs, axis_tilts = out[6], out[13], out[14]

    np.testing.assert_allclose(np.array(axis_dirs[0, 0]), n_true, atol=1e-6)
    np.testing.assert_allclose(
        np.rad2deg(np.array(axis_tilts[0, 0])), [1.2, -0.8], atol=1e-6
    )
    for m, ang in enumerate(angles):
        expected = Rotation.from_rotvec(np.deg2rad(ang) * n_true).as_matrix()
        np.testing.assert_allclose(np.array(R_cum[0, m]), expected, atol=1e-6)


def test_true_tilt_indexes_what_the_nominal_axis_cannot():
    obj, x_truth, x_nominal, _, _ = _make_fixture()
    _, dist_truth, _, _ = obj.get_results(x_truth[None, :])
    _, dist_nom, _, _ = obj.get_results(x_nominal[None, :])
    dist_truth = np.array(dist_truth[0])
    dist_nom = np.array(dist_nom[0])

    # At the true tilt every synthetic reflection lands on an integer hkl.
    assert (dist_truth < 0.02).all()
    # The nominal axis misindexes a real fraction of them: this is the
    # residual no offset/translation bound can absorb.
    assert (dist_nom > 0.05).mean() > 0.2
    assert dist_nom.mean() > 5 * dist_truth.mean()


def test_the_refinement_is_off_by_default_and_costs_nothing():
    obj, x_truth, _, _, _ = _make_fixture()
    off = VectorizedObjective(
        np.eye(3) / 6.0,
        np.array(obj.kf_ki_dir_init),
        None,
        np.array([1.0, 4.0]),
        goniometer_axes=np.array([[0.0, 1.0, 0.0, 1.0]]),
        goniometer_angles=np.array(obj.gonio_angles),
        motor_map=[0],
        beam_nominal=np.array([0.0, 0.0, 1.0]),
        kf_lab_fixed_vectors=np.array(obj.kf_lab_fixed),
        peak_run_indices=np.array(obj.peak_run_indices),
    )
    assert off.num_active_axis_vec == 0
    out = off._get_physical_params_jax(np.zeros((1, 3)))
    assert out[13] is None and out[14] is None
    # And the nominal-axis rotations are the plain axis-angle ones.
    expected = Rotation.from_rotvec(
        np.deg2rad(14.0) * np.array([0.0, 1.0, 0.0])
    ).as_matrix()
    np.testing.assert_allclose(np.array(out[6][0, 1]), expected, atol=1e-6)


def test_tilt_bound_is_respected_at_the_box_edge():
    obj, _, _, _, _ = _make_fixture(bound_deg=2.0)
    x_edge = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    out = obj._get_physical_params_jax(x_edge[None, :])
    tilts = np.rad2deg(np.array(out[14][0, 0]))
    assert tilts == pytest.approx([2.0, -2.0], abs=1e-6)
