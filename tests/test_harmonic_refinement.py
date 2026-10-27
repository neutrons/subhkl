"""Fourier-in-phi rocking of the crystal's effective orientation.

The cg4d-t4-lysozyme residual field decomposes into a rocking about
the single lab axis perpendicular to both the scan axis and the beam,
whose signed angle oscillates with phi near the lattice period
(measured 0.29 deg amplitude at ~70 deg period on a trigonal crystal).
refine_goniometer_harmonics models it as bounded Fourier coefficients
delta_k(phi) = sum_m a_km cos(m phi) + b_km sin(m phi) about fixed lab
axes.  The band runs 1..6 by default: crystallographic rotation orders
top out at 6 (cubic included), and a symmetry axis tilted from the
scan axis leaks harmonic n into n +/- 1 sidebands, so restricting to
multiples of n is not enough.  m = 0 stays excluded everywhere -- it
is the motor zero the global goniometer offsets already refine.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.optimization import (
    VectorizedObjective,
    harmonic_axes_from_scan,
    harmonic_rocking_matrices,
    harmonic_rocking_vectors,
)

BOUND = 0.5
ORDERS = [6]
TRUE_A, TRUE_B = 0.3, -0.2  # degrees, cos/sin coefficients of m = 6


def test_rocking_axis_is_perpendicular_to_scan_axis_and_beam():
    axes = harmonic_axes_from_scan([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], "rocking")
    assert axes.shape == (1, 3)
    np.testing.assert_allclose(np.abs(axes[0]), [1.0, 0.0, 0.0], atol=1e-12)

    full = harmonic_axes_from_scan([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], "full")
    assert full.shape == (3, 3)
    np.testing.assert_allclose(full @ full.T, np.eye(3), atol=1e-12)

    # Scan axis along the beam: the cross product degenerates and any
    # perpendicular direction serves.
    degen = harmonic_axes_from_scan([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], "rocking")
    assert abs(degen[0] @ np.array([0.0, 0.0, 1.0])) < 1e-12

    with pytest.raises(ValueError):
        harmonic_axes_from_scan([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], "sideways")


def test_pure_lattice_harmonic_is_periodic_in_360_over_m():
    axes = np.array([[1.0, 0.0, 0.0]])
    coeffs = np.array([[[TRUE_A, TRUE_B]]])  # (K=1, n=1, 2)
    phis = np.array([10.0, 25.0, 40.0])
    R1 = harmonic_rocking_matrices(phis, axes, [6], coeffs)
    R2 = harmonic_rocking_matrices(phis + 60.0, axes, [6], coeffs)
    np.testing.assert_allclose(R1, R2, atol=1e-12)
    # And the rotation vectors point along the requested axis.
    w = harmonic_rocking_vectors(phis, axes, [6], coeffs)
    np.testing.assert_allclose(w[:, 1:], 0.0, atol=1e-15)


def _make_fixture(n_frames=24, seed=7):
    """Elastic peaks from a crystal with a 6th-harmonic rocking about x."""
    rng = np.random.default_rng(seed)
    B = np.eye(3) / 6.0
    n_axis = np.array([0.0, 1.0, 0.0])
    ki = np.array([0.0, 0.0, 1.0])
    angles = np.arange(n_frames) * 11.0
    axes = harmonic_axes_from_scan(n_axis, ki, "rocking")
    coeffs = np.array([[[TRUE_A, TRUE_B]]])
    R_harm = harmonic_rocking_matrices(angles, axes, ORDERS, coeffs)

    kfs, frames = [], []
    while len(kfs) < 90:
        hkl = rng.integers(-4, 5, size=3)
        if not np.any(hkl):
            continue
        frame = rng.integers(0, n_frames)
        R_spin = Rotation.from_rotvec(np.deg2rad(angles[frame]) * n_axis).as_matrix()
        G = R_harm[frame] @ R_spin @ (B @ hkl)
        lam = -2.0 * (ki @ G) / (G @ G)
        if not (2.0 <= lam <= 3.8):
            continue
        kfs.append(ki + lam * G)
        frames.append(frame)
    kf = np.array(kfs).T

    def objective(**kw):
        return VectorizedObjective(
            B,
            kf - ki[:, None],
            None,
            np.array([2.0, 3.8]),
            goniometer_axes=np.array([[0.0, 1.0, 0.0, 1.0]]),
            goniometer_angles=angles[None, :],
            motor_map=[0],
            beam_nominal=ki,
            kf_lab_fixed_vectors=kf - ki[:, None],
            peak_run_indices=np.array(frames),
            **kw,
        )

    obj = objective(
        harmonic_frame_angles_deg=angles,
        harmonic_axes=axes,
        harmonic_orders=ORDERS,
        harmonic_bound_deg=BOUND,
    )
    # layout: [orientation 3][harmonics 2]; norm = (coeff + b) / (2b)
    x_truth = np.concatenate(
        [[0.0, 0.0, 0.0], (np.array([TRUE_A, TRUE_B]) + BOUND) / (2 * BOUND)]
    )
    x_nominal = np.concatenate([[0.0, 0.0, 0.0], [0.5, 0.5]])
    return obj, objective, x_truth, x_nominal


def test_dof_count_follows_axes_mode_and_band():
    _, objective, _, _ = _make_fixture()
    angles = np.arange(24) * 11.0
    ki = np.array([0.0, 0.0, 1.0])
    for mode, k in (("rocking", 1), ("transverse", 2), ("full", 3)):
        obj = objective(
            harmonic_frame_angles_deg=angles,
            harmonic_axes=harmonic_axes_from_scan([0, 1, 0], ki, mode),
            harmonic_orders=list(range(1, 7)),
        )
        assert obj.num_harmonic_params == 2 * 6 * k


def test_m_zero_is_rejected_as_degenerate_with_offsets():
    _, objective, _, _ = _make_fixture()
    with pytest.raises(ValueError, match="motor zero"):
        objective(
            harmonic_frame_angles_deg=np.arange(24) * 11.0,
            harmonic_axes=np.array([[1.0, 0.0, 0.0]]),
            harmonic_orders=[0, 6],
        )


def test_true_harmonic_coefficients_index_what_the_nominal_cannot():
    obj, _, x_truth, x_nominal = _make_fixture()
    _, d_true, _, _ = obj.get_results(x_truth[None, :])
    _, d_nom, _, _ = obj.get_results(x_nominal[None, :])
    d_true, d_nom = np.array(d_true[0]), np.array(d_nom[0])
    assert (d_true < 0.02).all()
    assert d_nom.mean() > 5 * d_true.mean()


def test_objective_rocking_matches_the_numpy_helper():
    """The jitted model and the predictor-side helper agree exactly."""
    obj, _, x_truth, x_nominal = _make_fixture()
    R_with = np.array(obj._get_physical_params_jax(x_truth[None, :])[6][0])
    R_base = np.array(obj._get_physical_params_jax(x_nominal[None, :])[6][0])
    angles = np.arange(24) * 11.0
    axes = harmonic_axes_from_scan([0, 1, 0], [0, 0, 1], "rocking")
    R_harm = harmonic_rocking_matrices(
        angles, axes, ORDERS, np.array([[[TRUE_A, TRUE_B]]])
    )
    np.testing.assert_allclose(R_with, R_harm @ R_base, atol=5e-6)
