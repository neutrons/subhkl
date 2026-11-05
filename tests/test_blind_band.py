"""Blind-search quadrature and symmetry contracts, without a reference seed."""

import numpy as np
import pytest

from subhkl.search.blind_band import (
    count_search_weights,
    distinct_orientations,
    sample_signed,
)


def test_signed_quadrature_is_unbiased_and_keeps_negative_evidence():
    weights = np.array([-1.0, 3.0, 0.0])
    values = np.array([2.0, -1.0, 7.0])
    estimates = []
    negatives = 0
    for seed in range(3000):
        indices, sampled = sample_signed(weights, 2, seed)
        estimates.append(sampled @ values[indices])
        negatives += np.any(sampled < 0)
        assert 2 not in indices  # identically zero contribution
    assert negatives > 0
    expected = weights @ values / np.abs(weights).sum()
    assert abs(np.mean(estimates) - expected) < 0.03
    first = sample_signed(weights, 2, 77)
    second = sample_signed(weights, 2, 77)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_small_data_are_scored_exactly_including_negative_total():
    weights = np.array([-4.0, 1.0, -2.0])
    indices, sampled = sample_signed(weights, 10)
    np.testing.assert_array_equal(indices, np.arange(3))
    np.testing.assert_allclose(sampled, weights / 7)


@pytest.mark.parametrize("weights,size", [([0, 0], 4), ([1, np.nan], 4), ([1], 0)])
def test_invalid_search_evidence_is_rejected(weights, size):
    with pytest.raises(ValueError):
        sample_signed(weights, size)


def test_symmetry_equivalents_do_not_fill_shortlist():
    from scipy.spatial.transform import Rotation

    B = np.diag([1 / 70, 1 / 98, 1 / 91])
    U = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    equivalent = U @ np.diag([-1, 1, -1])
    other = Rotation.from_rotvec([0.5, -0.4, 0.2]).as_matrix()
    candidates = [(U, 3.0, 8.0), (equivalent, 2.9, 7.9), (other, 2.0, 5.0)]
    chosen = distinct_orientations(candidates, B, "P 1 21 1", 2)
    assert len(chosen) == 2
    np.testing.assert_allclose(chosen[0][0], U)
    np.testing.assert_allclose(chosen[1][0], other)


def test_signed_band_geometry_is_invariant_to_positive_weight_rescaling():
    import jax
    from subhkl.search.spherical import refine_band_geometry

    C = np.array([[0.2, 0.1, 0.2], [-0.2, 0.2, 0.3]])
    with jax.enable_x64():

        def fit(weights):
            return refine_band_geometry(
                C,
                np.zeros_like(C),
                weights,
                np.eye(3) / 12,
                np.eye(3),
                np.eye(3),
                (2.0, 4.0),
                2.0,
                n_zones=13,
                radial_bound=0.0,
                rot_bound_deg=0.0,
                trans_bound_m=0.0,
                orient_bound_deg=0.1,
                maxiter=1,
            )

        first = fit(np.array([-3.0, 1.0]))
        second = fit(np.array([-21.0, 7.0]))
    np.testing.assert_allclose(first["score"], second["score"], atol=1e-12)
    np.testing.assert_allclose(first["U"], second["U"], atol=1e-12)


def test_count_search_weights_are_threshold_free_and_suppress_null_noise():
    dev = np.array([0.0, 0.5, 1.0, 25.0, 100.0, 400.0])
    w = count_search_weights(dev)
    assert w[0] == 0.0
    assert np.all(w[1:] > 0.0)
    assert np.all(np.diff(w) > 0.0)
    # 1-sigma background fluctuation is suppressed by >1e4 relative to a 5-sigma spot
    assert w[2] / w[3] < 2e-4
    # Sublinear sqrt(D) growth at high deviance prevents flare monopolization
    assert np.isclose(w[5] / w[4], 2.0, rtol=1e-3)
    # Quantile cap prevents a single beamstop-flare bin from dominating
    flare = np.append(np.full(1000, 100.0), 1e8)
    w_capped = count_search_weights(flare, max_quantile=0.99, max_factor=3.0)
    assert np.isclose(w_capped[-1] / w_capped[0], 3.0, rtol=1e-3)
