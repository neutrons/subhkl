"""Per-run translational offsets: the angular wobble's twin.

One bounded 3-vector per scan run at the innermost goniometer axis, so
it rides with the sample: s_lab(frame) = R_full(frame) @ t_run.  Mount
settling and the translational sphere-of-confusion are per-setting
displacements that no static lever arm can represent, exactly as the
per-run angle errors could not be represented by static offsets.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.optimization import VectorizedObjective

BOUND = 0.0005  # meters
T_TRUE = np.array([[0.0003, -0.0002, 0.0001], [-0.00025, 0.00015, -0.0002]])
DIST = 0.5  # sample-detector distance, meters


def _make_fixture(n_frames=12, seed=9):
    rng = np.random.default_rng(seed)
    B = np.eye(3) / 6.0
    n_axis = np.array([0.0, 1.0, 0.0])
    ki = np.array([0.0, 0.0, 1.0])
    angles = np.arange(n_frames) * 13.0
    frame_map = np.repeat(np.arange(len(T_TRUE)), n_frames // len(T_TRUE))

    xyz, kf_ki, runs = [], [], []
    while len(xyz) < 90:
        hkl = rng.integers(-4, 5, size=3)
        if not np.any(hkl):
            continue
        frame = rng.integers(0, n_frames)
        R = Rotation.from_rotvec(np.deg2rad(angles[frame]) * n_axis).as_matrix()
        G = R @ (B @ hkl)
        lam = -2.0 * (ki @ G) / (G @ G)
        if not (2.0 <= lam <= 3.8):
            continue
        kf = ki + lam * G
        s_lab = R @ T_TRUE[frame_map[frame]]
        p = s_lab + DIST * kf  # lab-fixed detector point
        xyz.append(p)
        kf_ki.append(kf - ki)
        runs.append(frame)
    xyz = np.array(xyz)

    def objective(**kw):
        return VectorizedObjective(
            B,
            np.array(kf_ki).T,
            xyz,
            np.array([2.0, 3.8]),
            goniometer_axes=np.array([[0.0, 1.0, 0.0, 1.0]]),
            goniometer_angles=angles[None, :],
            motor_map=[0],
            beam_nominal=ki,
            peak_run_indices=np.array(runs),
            **kw,
        )

    obj = objective(
        per_run_trans=True,
        per_run_frame_map=frame_map,
        per_run_trans_bound_m=BOUND,
    )
    # layout: [orientation 3][per-run trans 2*3]; norm = (t + b) / (2b)
    x_truth = np.concatenate(
        [[0.0, 0.0, 0.0], ((T_TRUE + BOUND) / (2 * BOUND)).ravel()]
    )
    x_nominal = np.concatenate([[0.0, 0.0, 0.0], np.full(6, 0.5)])
    return obj, objective, x_truth, x_nominal, frame_map, np.array(runs)


def test_true_displacements_index_what_static_geometry_cannot():
    obj, _, x_truth, x_nominal, _, _ = _make_fixture()
    _, d_true, _, _ = obj.get_results(x_truth[None, :])
    _, d_nom, _, _ = obj.get_results(x_nominal[None, :])
    d_true, d_nom = np.array(d_true[0]), np.array(d_nom[0])
    assert (d_true < 0.02).all()
    assert d_nom.mean() > 5 * d_true.mean()


def test_displacements_are_per_run_not_global():
    obj, _, x_truth, _, frame_map, runs = _make_fixture()
    x_partial = x_truth.copy()
    x_partial[3 + 3 : 3 + 6] = 0.5  # zero run 1's displacement
    _, d, _, _ = obj.get_results(x_partial[None, :])
    d = np.array(d[0])
    in_run1 = frame_map[runs] == 1
    assert (d[~in_run1] < 0.02).all()
    assert np.median(d[in_run1]) > 5 * np.median(d[~in_run1])


def test_off_by_default_consumes_no_parameters():
    _, objective, _, _, _, _ = _make_fixture()
    off = objective()
    assert off.num_per_run_trans_params == 0
    loss, _, _, _ = off.get_results(np.zeros((1, 3)))
    assert np.isfinite(float(np.array(loss[0])))


def test_bound_clamps_at_the_box_edge():
    obj, _, _, _, _, _ = _make_fixture()
    x_edge = np.concatenate([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5, 0.5, 1.0, 0.0]])
    out = obj._get_physical_params_jax(x_edge[None, :])
    t = np.array(out[16][0])
    np.testing.assert_allclose(
        t, [[BOUND, -BOUND, 0.0], [0.0, BOUND, -BOUND]], atol=1e-9
    )


def test_frame_map_is_required():
    _, objective, _, _, _, _ = _make_fixture()
    with pytest.raises(ValueError, match="per_run_frame_map"):
        objective(per_run_trans=True)


def test_trans_axes_can_be_selected_independently_of_angle_axes():
    """The phi-stage lever arm must be refinable without freeing the
    pure-gauge phi zero point: the translation mask decouples."""
    axes = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, -1.0],
        ]
    )
    ki = np.array([0.0, 0.0, 1.0])
    obj = VectorizedObjective(
        np.eye(3) / 6.0,
        np.zeros((3, 4)),
        None,
        np.array([2.0, 3.8]),
        goniometer_axes=axes,
        goniometer_angles=np.zeros((5, 1)),
        motor_map=[0, 1, 2, 1, 3],
        refine_sample=True,
        goniometer_refine_mask=np.array([True, False, True, False]),
        goniometer_trans_refine_mask=np.array([True, False, True, True]),
        goniometer_trans_bound_meters=0.005,
        beam_nominal=ki,
        kf_lab_fixed_vectors=np.zeros((3, 4)),
        peak_run_indices=np.zeros(4, dtype=int),
    )
    mask = np.array(obj.gonio_trans_mask).astype(int)
    np.testing.assert_array_equal(mask, [1, 0, 1, 0, 1])  # phi trans active
    assert obj.num_active_gonio == 2  # phi ANGLE offset still inactive
