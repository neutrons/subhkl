"""The solve reports whether it met its own optimality certificate.

The peak finder solves a convex problem, so a certified iterate is *the*
optimum and the reported peaks are the estimator's answer.  An uncertified one
is wherever the iteration ran out, and nothing else in the output distinguishes
them: same peaks, same deviances, wrong in no way the HDF5 reveals.  The
residual is computed every iteration anyway, so surfacing it costs nothing and
is the difference between "these are the peaks" and "these are the peaks the
solver reached".

The quantity is max_i |G_i| / lam_i, the per-coordinate prox-gradient residual
in units of the local penalty.  For min_{c>=0} f(c) + lam^T c that residual IS
the KKT violation coordinate by coordinate -- max(0, p_i - lam_i) where c_i = 0
(dual feasibility) and |lam_i - p_i| where c_i > 0 (stationarity on the
support) -- so it measures the basis-pursuit dual certificate directly, on the
same scale the solver stops on.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from subhkl.search.matrix_free import (
    _KKT_STOP_TOL,
    MatrixFreeSparseRBFPeakFinder,
)

from .test_auto_max_sigma import frames_with_peaks


def _finder(**kw):
    kw.setdefault("min_sigma", 1.5)
    kw.setdefault("max_sigma", 5.0)
    kw.setdefault("num_sigmas", 5)
    kw.setdefault("show_steps", False)
    return MatrixFreeSparseRBFPeakFinder(**kw)


def test_solve_reports_its_optimality_certificate():
    frames = frames_with_peaks(3.0, n_frames=3)
    finder = _finder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder.find_peaks_batch(frames)

    report = finder.solve_optimality
    assert set(report) == {
        "kkt_rel_max",
        "kkt_rel_median",
        "n_images",
        "n_uncertified",
        "tol",
    }
    assert report["n_images"] == frames.shape[0]
    assert report["tol"] == _KKT_STOP_TOL
    assert np.isfinite(report["kkt_rel_max"])
    assert report["kkt_rel_max"] >= report["kkt_rel_median"] >= 0.0
    assert 0 <= report["n_uncertified"] <= report["n_images"]


def test_uncertified_solves_are_flagged_and_certified_ones_are_not():
    """The warning and the count must agree, whatever the iteration budget.

    Asserted as an invariant rather than as a fixed expectation, so that
    changing max_iter -- which is exactly the knob this reporting exists to
    inform -- moves both together instead of breaking the test.
    """
    frames = frames_with_peaks(3.0, n_frames=2)
    finder = _finder()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder.find_peaks_batch(frames)
    flagged = any("optimality certificate" in str(w.message) for w in caught)
    assert flagged == (finder.solve_optimality["n_uncertified"] > 0)


def test_certificate_responds_to_the_iteration_budget():
    """A real convergence measure must fall when the solver is given more room.

    This is what rules out the reported number being an artifact of where the
    residual is taken rather than a property of the iterate.
    """
    frames = frames_with_peaks(3.0, n_frames=1)
    finder = _finder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder.find_peaks_batch(frames)

    img = jnp.asarray(frames[0])
    bg = jnp.asarray(np.asarray(finder._last_bg_map[0]))
    _, kkt_short = finder._solve_ssn_cg_global(img, bg, max_iter=100)
    _, kkt_long = finder._solve_ssn_cg_global(img, bg, max_iter=400)
    assert float(kkt_long) < float(kkt_short)
