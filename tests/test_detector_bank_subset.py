"""Refining a SUBSET of detector banks must leave the other banks' peaks
at their measured lab positions.

Before the fix, peaks on banks outside --refine-detector-banks kept the
default bank index 0 and zero panel offsets, so the dynamic-geometry
branch collapsed them all onto the center of the first refined bank.
The subset option therefore only ever worked when the subset covered
every bank that had peaks -- exactly the case where it is redundant.
"""

from __future__ import annotations

import numpy as np
import pytest

from subhkl.optimization import VectorizedObjective


def _make_objectives():
    """Peaks on two panels; only panel A is loaded for refinement.

    Mirrors what commands.py produces for --refine-detector-banks A:
    panel B's peaks keep bank index 0 and zero offsets, and are flagged
    unrefined in peak_pixel_coords["refined_mask"].
    """
    ki = np.array([0.0, 0.0, 1.0])

    c_a = np.array([0.6, 0.0, 0.6])
    u_a = np.array([0.0, 1.0, 0.0])
    v_a = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)

    c_b = np.array([0.0, 0.0, -1.0])
    u_b = np.array([1.0, 0.0, 0.0])
    v_b = np.array([0.0, 1.0, 0.0])

    uv = [(-0.08, 0.05), (0.02, -0.11), (0.09, 0.07), (-0.04, -0.03)]
    xyz, u_off, v_off, bank_idx, refined = [], [], [], [], []
    for u, v in uv:
        xyz.append(c_a + u * u_a + v * v_a)
        u_off.append(u)
        v_off.append(v)
        bank_idx.append(0)
        refined.append(True)
    for u, v in uv:
        xyz.append(c_b + u * u_b + v * v_b)
        u_off.append(0.0)
        v_off.append(0.0)
        bank_idx.append(0)
        refined.append(False)
    xyz = np.array(xyz)

    kf = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    kf_ki = (kf - ki).T  # (3, N)

    common = dict(
        cell_params=None,
        beam_nominal=ki,
        kf_lab_fixed_vectors=kf_ki,
        peak_run_indices=np.zeros(len(xyz), dtype=int),
    )
    B = np.eye(3) / 6.0
    wavelength = np.array([1.0, 4.0])

    detector_params = {
        "centers": [c_a],
        "uhats": [u_a],
        "vhats": [v_a],
        "m": [256],
        "n": [256],
        "pw": [0.5 / 256],
        "ph": [0.5 / 256],
        "modes": ["global_trans"],
        "global_trans_bound_meters": 0.05,
    }
    peak_pixel_coords = {
        "u_offsets": u_off,
        "v_offsets": v_off,
        "bank_indices": bank_idx,
        "refined_mask": refined,
    }

    subset = VectorizedObjective(
        B,
        kf_ki,
        xyz,
        wavelength,
        refine_detector=True,
        detector_params=detector_params,
        peak_pixel_coords=peak_pixel_coords,
        **common,
    )
    reference = VectorizedObjective(B, kf_ki, xyz, wavelength, **common)
    return subset, reference, np.array(refined)


def test_nominal_geometry_reproduces_the_static_path_exactly():
    subset, reference, _ = _make_objectives()
    # Normalized 0.5 maps to zero translation: norm * 2b - b.
    x_zero = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
    x_ref = np.array([[0.0, 0.0, 0.0]])
    _, dist, _, _ = subset.get_results(x_zero)
    _, dist_ref, _, _ = reference.get_results(x_ref)
    # float32 panel reconstruction (c + u*uhat + v*vhat) vs stored xyz
    np.testing.assert_allclose(np.array(dist), np.array(dist_ref), atol=1e-6)


def test_translation_moves_only_the_refined_banks_peaks():
    subset, _, refined = _make_objectives()
    x_zero = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
    x_trans = np.array([[0.0, 0.0, 0.0, 1.0, 0.5, 0.5]])
    _, d0, _, _ = subset.get_results(x_zero)
    _, d1, _, _ = subset.get_results(x_trans)
    d0, d1 = np.array(d0[0]), np.array(d1[0])

    np.testing.assert_allclose(d1[~refined], d0[~refined], atol=1e-12)
    assert np.max(np.abs(d1[refined] - d0[refined])) > 1e-6


def test_subset_without_peak_positions_is_refused():
    subset, _, _ = _make_objectives()
    with pytest.raises(ValueError, match="subset of detector banks"):
        VectorizedObjective(
            np.eye(3) / 6.0,
            np.array(subset.kf_ki_dir_init),
            None,
            np.array([1.0, 4.0]),
            beam_nominal=np.array([0.0, 0.0, 1.0]),
            kf_lab_fixed_vectors=np.array(subset.kf_ki_dir_init),
            refine_detector=True,
            detector_params={
                "centers": [[0.6, 0.0, 0.6]],
                "uhats": [[0.0, 1.0, 0.0]],
                "vhats": [[0.7071, 0.0, -0.7071]],
                "m": [256],
                "n": [256],
                "pw": [0.5 / 256],
                "ph": [0.5 / 256],
                "modes": ["global_trans"],
            },
            peak_pixel_coords={
                "u_offsets": [0.0] * 8,
                "v_offsets": [0.0] * 8,
                "bank_indices": [0] * 8,
                "refined_mask": [True] * 4 + [False] * 4,
            },
        )
