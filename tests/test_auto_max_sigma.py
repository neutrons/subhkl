"""max_sigma=None: the bank ceiling is measured, not guessed.

The estimator is the truncated-moment width census validated in the benchmark
harness (~0.01 px bias on synthetics): unclipped circular moments around a
clipped-weight centroid, the truncation inverted by damped fixed point, two
apertures that must agree.  The important tests use peaks of *known* width,
because a measurement nobody has checked against a known answer is not
obviously better than the hand-set constant it replaces.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from subhkl.search.matrix_free import (
    MatrixFreeSparseRBFPeakFinder,
    _ceiling_from_widths,
    _moment_width_census,
    _sigma_from_truncated_m2,
    _truncated_m2_ratio,
)


def pixel_integrated_gaussian(shape, r0, c0, sigma, amplitude):
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    s2 = sigma * np.sqrt(2.0)
    er = erf((rr - r0 + 0.5) / s2) - erf((rr - r0 - 0.5) / s2)
    ec = erf((cc - c0 + 0.5) / s2) - erf((cc - c0 - 0.5) / s2)
    return amplitude * (np.pi / 2.0) * sigma**2 * er * ec


def frames_with_peaks(sigma, n_frames=3, size=160, amp=400.0, seed=7):
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_frames):
        rate = np.full((size, size), 1.0)
        for r0, c0 in ((40, 40), (40, 120), (120, 40), (120, 120)):
            rate += pixel_integrated_gaussian(
                rate.shape, r0 + 3 * k, c0 - 2 * k, sigma, amp / (2 * np.pi * sigma**2)
            )
        frames.append(rng.poisson(rate).astype(np.float32))
    return np.stack(frames)


def test_truncation_ratio_matches_the_analytic_limits():
    assert _truncated_m2_ratio(1e6) == pytest.approx(2.0)
    assert _truncated_m2_ratio(2.0**2 / 2) == pytest.approx(1.374, abs=0.01)
    assert _truncated_m2_ratio(0.0) == 0.0


def test_truncation_inverts_exactly():
    for sigma in (0.8, 1.5, 3.0, 5.5):
        for k in (2.5, 3.5):
            aperture = k * sigma
            m2 = sigma**2 * _truncated_m2_ratio(aperture**2 / (2 * sigma**2))
            assert _sigma_from_truncated_m2(m2, aperture) == pytest.approx(
                sigma, abs=1e-3
            )


def test_width_census_recovers_known_widths():
    for sigma in (2.0, 3.5):
        frames = frames_with_peaks(sigma, amp=2000.0)
        bg = np.ones_like(frames)
        widths = _moment_width_census(frames, bg, 1.0, window_sigma=5.0)
        assert widths.size >= 8, f"sigma {sigma}: only {widths.size} usable"
        assert np.median(widths) == pytest.approx(sigma, abs=0.25)


def test_ceiling_ladder_matches_the_evidence():
    rng = np.random.default_rng(1)
    many = rng.normal(3.0, 0.2, 1000).clip(2, 4)
    # Plenty of peaks: a p99 with real support, rounded up to the half pixel.
    assert _ceiling_from_widths(many, 1.5) == pytest.approx(4.5, abs=0.5)
    # A handful: the max plus a 20% margin (3.6), then the half-pixel round-up
    # with headroom -- never a runaway percentile.
    few = np.full(10, 3.0)
    assert _ceiling_from_widths(few, 1.5) == pytest.approx(4.5, abs=0.1)
    # Too few to say anything: refused, so the caller can fall back loudly.
    assert _ceiling_from_widths(np.full(3, 3.0), 1.5) is None
    # Never below a usable range above the floor.
    assert _ceiling_from_widths(np.full(50, 0.8), 1.5) >= 1.5 * 1.5


def test_auto_ceiling_measures_the_data_and_finds_the_peaks():
    sigma = 3.0
    frames = frames_with_peaks(sigma, amp=3000.0)
    finder = MatrixFreeSparseRBFPeakFinder(
        min_sigma=1.5,
        num_sigmas=4,
        profile_file="gaussian",
        shape_ratio=1.0,
    )
    assert finder._auto_ceiling
    peaks = finder.find_peaks_batch(frames)
    # The measured ceiling clears the true width with headroom but is not the
    # old hand-set 10: it responded to the data.
    assert 3.2 <= finder.max_sigma <= 6.5, finder.max_sigma
    # And the peaks come back at their true width, not clipped at the edge of
    # a too-small bank.
    widths = np.concatenate([np.asarray(p)[:, 3] for p in peaks if len(p)])
    assert widths.size >= 8
    assert np.median(widths) == pytest.approx(sigma, abs=0.4)
    assert (widths >= 0.98 * finder.max_sigma).mean() < 0.2


def test_auto_ceiling_falls_back_loudly_on_a_peak_free_batch():
    rng = np.random.default_rng(5)
    frames = rng.poisson(np.full((2, 96, 96), 1.0)).astype(np.float32)
    finder = MatrixFreeSparseRBFPeakFinder(
        min_sigma=1.5,
        num_sigmas=3,
        profile_file="gaussian",
        shape_ratio=1.0,
    )
    with pytest.warns(UserWarning, match="measured bank ceiling"):
        finder.find_peaks_batch(frames)
    assert finder.max_sigma == 10.0


def test_explicit_max_sigma_is_never_second_guessed():
    frames = frames_with_peaks(3.0, n_frames=1)
    finder = MatrixFreeSparseRBFPeakFinder(
        min_sigma=1.5,
        max_sigma=4.0,
        num_sigmas=3,
        profile_file="gaussian",
        shape_ratio=1.0,
    )
    assert not finder._auto_ceiling
    finder.find_peaks_batch(frames)
    assert finder.max_sigma == 4.0
