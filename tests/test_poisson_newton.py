"""Independent checks of the scan-scale Poisson curvature solve."""

import numpy as np
import pytest
from scipy import sparse

from subhkl.search.poisson_newton import schur_direction
from subhkl.search.poisson_orientation import CountData, fit_intensities


@pytest.mark.parametrize("n_background", [3, 600])
def test_schur_step_matches_dense_constrained_hessian(n_background):
    rng = np.random.default_rng(42)
    n_reflection, pixels = 8, n_background * 4
    F = rng.uniform(size=(pixels, n_reflection))
    F[F < 0.8] = 0
    B = sparse.csr_matrix(
        (rng.uniform(0.5, 2, pixels), (np.arange(pixels), np.arange(pixels) // 4)),
        shape=(pixels, n_background),
    )
    M = sparse.hstack([sparse.csr_matrix(F), B], format="csc")
    x = rng.uniform(0.5, 2, M.shape[1])
    gradient = rng.normal(size=len(x))
    x[0], gradient[0] = 0, 1  # constrained reflection
    x[1], gradient[1] = 0, -1  # reflection entering the free set
    curvature = rng.uniform(0.1, 3, pixels)
    penalties = np.array([1.2, 2.3])
    H = (M.T @ M.multiply(curvature[:, None])).toarray()
    for group in range(2):
        sl = slice(group * 4, (group + 1) * 4)
        norm = np.linalg.norm(x[sl])
        u = x[sl] / norm
        H[sl, sl] += penalties[group] / norm * (np.eye(4) - np.outer(u, u))
    free = np.arange(1, len(x))
    ridge = 1e-10 * max(1, np.diag(H)[free].max())
    expected = np.zeros_like(x)
    expected[free] = np.linalg.solve(
        H[np.ix_(free, free)] + ridge * np.eye(len(free)), -gradient[free]
    )
    step, _, info = schur_direction(M, curvature, x, gradient, penalties, n_background)
    assert info == 0
    np.testing.assert_allclose(step, expected, rtol=1e-7, atol=1e-8)


def test_bright_fit_with_more_than_512_panel_backgrounds():
    rng = np.random.default_rng(8)
    panels, pixels = 600, 16
    bg = np.geomspace(0.1, 100, panels * pixels)
    rows = np.arange(5)[:, None] * 16 + np.arange(5)
    A = sparse.csc_matrix(
        (
            np.tile([0.05, 0.2, 0.5, 0.2, 0.05], 5),
            (rows.ravel(), np.repeat(np.arange(5), 5)),
        ),
        shape=(len(bg), 5),
    )
    y = rng.poisson(bg * 1.7 + A @ [1e6, 100, 1e4, 0, 5e5])
    data = CountData.from_images(
        y.reshape(panels, 4, 4), np.arange(panels), bg.reshape(panels, 4, 4), 1
    )
    fit = fit_intensities(A, data, 1, 1.2, tol=1e-8)
    assert fit.converged
    assert fit.newton_iterations > 0
    assert max(fit.reflection_kkt, fit.background_kkt) == fit.kkt
    assert fit.kkt < fit.kkt_tolerance
    # Check the background score directly on unaggregated pixels.
    score = np.bincount(data.bank_index, weights=bg * (1 - y / fit.mean))
    norm = np.sqrt(np.bincount(data.bank_index, weights=bg))
    score /= norm
    coefficients = fit.background_scales * norm
    residual = coefficients - np.maximum(coefficients - score, 1e-10)
    assert np.max(np.abs(residual)) < fit.kkt_tolerance
