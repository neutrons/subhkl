"""Contracts for the experimental threshold-free count likelihood."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from subhkl.search.poisson_orientation import (
    CountData,
    fit_intensities,
    geometry_vector,
)


def test_zero_counts_and_negative_residuals_are_kept():
    images = np.array([[[0.0, 2.0], [1.0, 5.0]]])
    d = CountData.from_images(images, [3], [np.ones((2, 2)) * 3], bin_px=1)
    np.testing.assert_array_equal(d.counts, images.ravel())
    assert len(d.counts) == 4
    assert np.any(d.counts - d.background < 0)


def test_masked_values_do_not_enter_counts_or_background():
    images = np.array([[[np.nan, 2.0], [1.0, 5.0]]])
    masks = np.array([[[False, True], [True, True]]])
    d = CountData.from_images(images, [3], [np.ones((2, 2))], 1, masks)
    np.testing.assert_array_equal(d.counts, [2.0, 1.0, 5.0])
    assert d.pixel_indices[3][0, 0] == -1


def test_geometry_fixes_the_beam_axis_rotation():
    np.testing.assert_array_equal(
        geometry_vector([1, 2, 3, 4, 5, 6]), [1, 2, 3, 0, 4, 5, 6]
    )


@pytest.mark.parametrize("alpha", [0.0, 0.7, 100.0])
def test_poisson_solve_matches_independent_scalar_lasso_optimizer(alpha):
    # One reflection per group reduces group lasso to ordinary nonnegative L1.
    y = np.array([1.0, 0.0, 8.0, 20.0, 9.0, 3.0, 0.0, 2.0])
    d = CountData.from_images([y.reshape(2, 4)], [1], [np.ones((2, 4)) * 2], 1)
    A = sparse.csc_matrix(
        np.array(
            [
                [0, 0],
                [0, 0.1],
                [0.2, 0.3],
                [0.6, 0.4],
                [0.2, 0.2],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
    )
    fit = fit_intensities(A, d, 2, alpha, weights=np.ones(2), tol=1e-8)
    scale = np.sqrt(np.asarray(A.power(2).T @ (1 / d.background)).ravel())

    def objective(x):
        mu = d.background * x[2] + A @ x[:2]
        return np.sum(mu - y * np.log(mu)) + alpha * (scale @ x[:2])

    ref = minimize(
        objective,
        [10.0, 1.0, 1.0],
        method="L-BFGS-B",
        bounds=[(0, None), (0, None), (1e-8, None)],
        options={"ftol": 1e-13, "gtol": 1e-8},
    )
    assert fit.converged
    np.testing.assert_allclose(
        np.r_[fit.intensities.ravel(), fit.background_scales], ref.x, atol=2e-4
    )


@pytest.mark.parametrize("n_true", [0, 1, 2])
def test_support_is_selected_by_whole_orientation(n_true):
    rng = np.random.default_rng(28)
    # Twenty individually weak reflections per orientation, spatially separate.
    n_groups, size, pixels = 4, 20, 1200
    rows = np.arange(n_groups * size) * 10
    A = sparse.csc_matrix(
        (np.ones(len(rows)), (rows, np.arange(len(rows)))),
        shape=(pixels, n_groups * size),
    )
    mean = np.full(pixels, 100.0)
    signal = np.zeros((n_groups, size))
    signal[:n_true] = 18.0  # only 1.8 background standard deviations per reflection
    y = rng.poisson(mean + A @ signal.ravel())
    data = CountData.from_images([y.reshape(30, 40)], [1], [100.0], 1)
    fit = fit_intensities(A, data, n_groups, 1.2, tol=1e-7)
    assert fit.converged
    np.testing.assert_array_equal(
        np.flatnonzero(fit.group_norms > 0), np.arange(n_true)
    )


def test_empty_data_is_rejected():
    with pytest.raises(ValueError, match="no valid"):
        CountData.from_images(
            [np.zeros((2, 2))], [1], [1.0], 1, [np.zeros((2, 2), bool)]
        )


def test_bright_poisson_fit_preserves_full_pixel_objective():
    # Large flux contrast and many background-only bins reproduce the
    # conditioning of the garnet still without distributing facility data.
    from scipy.special import xlogy

    rng = np.random.default_rng(8)
    pixels = 4000
    columns = np.zeros((pixels, 5))
    for j in range(5):
        columns[j * 10 : j * 10 + 5, j] = [0.05, 0.2, 0.5, 0.2, 0.05]
    A = sparse.csc_matrix(columns)
    bg = np.linspace(0.1, 100, pixels)
    y = rng.poisson(bg * 1.7 + A @ np.array([1e6, 100, 1e4, 0, 5e5]))
    data = CountData.from_images([y.reshape(50, 80)], [1], [bg.reshape(50, 80)], 1)
    fit = fit_intensities(A, data, 1, 1.2, rtol=1e-6)
    assert fit.converged
    assert fit.kkt < fit.kkt_tolerance
    expected = bg * fit.background_scales[0] + A @ fit.intensities.ravel()
    np.testing.assert_allclose(fit.mean, expected)
    objective = np.sum(expected - y + xlogy(y, y / expected))
    objective += 1.2 * (fit.weights @ fit.group_norms)
    np.testing.assert_allclose(fit.objective, objective, rtol=1e-10)
    fisher = np.sqrt(np.asarray(A.power(2).T @ (1 / bg)).ravel())
    coefficients = fit.intensities.ravel() * fisher
    gradient = np.asarray(A.T @ (1 - y / expected)).ravel() / fisher
    gradient += 1.2 * fit.weights[0] * coefficients / np.linalg.norm(coefficients)
    projected = np.where(coefficients > 0, gradient, np.minimum(gradient, 0))
    assert np.max(np.abs(projected)) < 2 * fit.kkt_tolerance
