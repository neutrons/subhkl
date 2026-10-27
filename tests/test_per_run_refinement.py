"""Per-run goniometer angle corrections.

Static per-setting positioning errors (encoder repeatability, mount
settling; measured 0.13 deg rms across the six t4 phi settings, with
random signs) cannot be represented by any static geometry parameter:
offsets, axis vectors and detector modes all slid along a flat valley
trying to average them.  refine_goniometer_per_run gives the scan
motor one bounded angle correction per run, mapped to frames through
the merged file's file_offsets bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.optimization import VectorizedObjective

BOUND = 0.5
TRUE_DELTAS = np.array([0.3, -0.25, 0.15])  # degrees, per run


def _make_fixture(deltas=TRUE_DELTAS, n_frames=12, seed=5):
    """Elastic peaks from a crystal whose per-run angles are offset."""
    rng = np.random.default_rng(seed)
    B = np.eye(3) / 6.0
    n_axis = np.array([0.0, 1.0, 0.0])
    ki = np.array([0.0, 0.0, 1.0])
    angles = np.arange(n_frames) * 11.0
    frame_map = np.repeat(np.arange(len(deltas)), n_frames // len(deltas))

    kfs, runs = [], []
    while len(kfs) < 90:
        hkl = rng.integers(-4, 5, size=3)
        if not np.any(hkl):
            continue
        frame = rng.integers(0, n_frames)
        true_angle = angles[frame] + deltas[frame_map[frame]]
        R_true = Rotation.from_rotvec(np.deg2rad(true_angle) * n_axis).as_matrix()
        G = R_true @ (B @ hkl)
        lam = -2.0 * (ki @ G) / (G @ G)
        if not (2.0 <= lam <= 3.8):
            continue
        kfs.append(ki + lam * G)
        runs.append(frame)
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
            peak_run_indices=np.array(runs),
            **kw,
        )

    obj = objective(
        per_run_motor_index=0,
        per_run_frame_map=frame_map,
        per_run_bound_deg=BOUND,
    )
    # layout: [orientation 3][per-run n_runs]; norm = (delta + b) / (2b)
    x_truth = np.concatenate([[0.0, 0.0, 0.0], (deltas + BOUND) / (2 * BOUND)])
    x_nominal = np.concatenate([[0.0, 0.0, 0.0], np.full(len(deltas), 0.5)])
    return obj, objective, x_truth, x_nominal, frame_map, np.array(runs)


def test_true_per_run_deltas_index_what_the_nominal_angles_cannot():
    obj, _, x_truth, x_nominal, _, _ = _make_fixture()
    _, d_true, _, _ = obj.get_results(x_truth[None, :])
    _, d_nom, _, _ = obj.get_results(x_nominal[None, :])
    d_true, d_nom = np.array(d_true[0]), np.array(d_nom[0])
    assert (d_true < 0.02).all()
    assert d_nom.mean() > 5 * d_true.mean()


def test_corrections_are_per_run_not_global():
    obj, _, x_truth, _, frame_map, runs = _make_fixture()
    # Zero only run 1's correction: exactly run 1's peaks degrade.
    x_partial = x_truth.copy()
    x_partial[3 + 1] = 0.5
    _, d, _, _ = obj.get_results(x_partial[None, :])
    d = np.array(d[0])
    in_run1 = frame_map[runs] == 1
    assert (d[~in_run1] < 0.02).all()
    assert np.median(d[in_run1]) > 5 * np.median(d[~in_run1])


def test_off_by_default_consumes_no_parameters():
    _, objective, _, _, _, _ = _make_fixture()
    off = objective()
    assert off.num_per_run_params == 0
    loss, dist, _, _ = off.get_results(np.zeros((1, 3)))
    assert np.isfinite(float(np.array(loss[0])))


def test_bound_clamps_at_the_box_edge():
    obj, _, _, _, _, _ = _make_fixture()
    x_edge = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.5]])
    out = obj._get_physical_params_jax(x_edge)
    delta = np.array(out[15][0])
    np.testing.assert_allclose(delta, [BOUND, -BOUND, 0.0], atol=1e-6)


def test_frame_map_is_required():
    _, objective, _, _, _, _ = _make_fixture()
    with pytest.raises(ValueError, match="per_run_frame_map"):
        objective(per_run_motor_index=0)


# --- Write-back of the corrections into the output peaks file ---
#
# The optimizer's internal angle array cannot index the file being
# written: identical columns collapse to one and re-tile only up to the
# last peak-bearing image.  A single-run file (force_mode single) makes
# the collapse certain -- every frame shares one angle setting -- and
# any trailing peakless image then truncates the re-tiled array below
# the file's frame count, so a map built from the optimizer's shape
# can neither index the file nor be told apart from an axis-first
# layout (cg4d-l1-mbl: angles (56, 5), optimizer 49-55 frames, all ten
# runs crashed).  _file_frame_run_map rebuilds the map from the file's
# own layout instead.


def test_writeback_map_covers_all_file_frames_not_just_peak_bearing_ones():
    from subhkl.commands import _file_frame_run_map

    ang = np.tile([90.0, 45.0, 0.0, 45.0, 0.0], (56, 1))  # one run, 56 images
    frame_first, frame_to_run = _file_frame_run_map(ang, 5, np.array([0]))
    assert frame_first
    assert len(frame_to_run) == 56
    np.testing.assert_array_equal(frame_to_run, 0)


def test_writeback_map_follows_run_boundaries_in_joint_files():
    from subhkl.commands import _file_frame_run_map

    ang = np.zeros((60, 5))
    frame_first, frame_to_run = _file_frame_run_map(ang, 5, np.array([0, 20, 40]))
    assert frame_first
    np.testing.assert_array_equal(frame_to_run, np.repeat([0, 1, 2], 20))


def test_writeback_detects_axis_first_layout():
    from subhkl.commands import _file_frame_run_map

    ang = np.zeros((5, 56))
    frame_first, frame_to_run = _file_frame_run_map(ang, 5, np.array([0]))
    assert not frame_first
    assert len(frame_to_run) == 56


def test_writeback_square_array_takes_the_frame_first_convention():
    from subhkl.commands import _file_frame_run_map

    ang = np.zeros((5, 5))
    frame_first, frame_to_run = _file_frame_run_map(ang, 5, np.array([0]))
    assert frame_first
    assert len(frame_to_run) == 5


def test_writeback_rejects_angles_that_span_no_axis_dimension():
    from subhkl.commands import _file_frame_run_map

    with pytest.raises(ValueError, match="does not span"):
        _file_frame_run_map(np.zeros((3, 4)), 5, np.array([0]))
    with pytest.raises(ValueError, match="does not span"):
        _file_frame_run_map(np.zeros(5), 5, np.array([0]))
