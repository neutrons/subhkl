"""min_sigma=None: the bank floor is measured, not guessed.

The floor gets its own module because it is load-bearing in a way the
constant it replaces was not.  The false-alarm calibration counts *resolution
elements*, ``N_k = area / (2 pi sigma_k**2)``, which is invariant under a
change of pixel pitch: refine the detector at constant flux and both the area
in pixels and sigma_k**2 in pixels scale as 1/Delta**2, so the threshold z
does not move.  Holding that invariance is the point -- past Nyquist a finer
grid adds no information about a band-limited peak, so the detection floor
*should* be flat in Delta, and any drift is an artifact rather than physics.

A bank floor pinned in pixels is exactly such an artifact.  The ceiling is
measured and tracks 1/Delta while the floor stays at one pixel, so refining
grows a tail of sub-PSF channels no real peak can occupy; the finest carries
``N_k = N_pix / 2 pi``, restoring the pixel-count multiplicity penalty and
drifting z up as ``sqrt(2 log N_pix)``.  At gamma = 0 every scale shares one
z, so unoccupied channels tax the occupied ones directly.

test_detection_floor_is_invariant_under_refinement is that statement as a
regression test, with the pixel-pinned floor measured alongside so the test
is demonstrably sensitive to the thing it guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from subhkl.search.matrix_free import (
    _SAMPLING_MIN_SIGMA,
    MatrixFreeSparseRBFPeakFinder,
    _auto_sigma_grid,
    _ceiling_from_widths,
    _floor_from_widths,
    _frag_calibrated_z,
)

from .test_auto_max_sigma import pixel_integrated_gaussian


def dense_frames(sigma, n_frames=3, size=400, amp=400.0, seed=11):
    """A scene with enough peaks to clear the census's 8-peak percentile
    ladder -- frames_with_peaks puts up only four per frame, of which too few
    survive the two-aperture consistency guard to quote a floor."""
    rng = np.random.default_rng(seed)
    centres = [(60 + 55 * i, 60 + 55 * j) for i in range(6) for j in range(6)]
    frames = []
    for k in range(n_frames):
        rate = np.full((size, size), 1.0)
        for r0, c0 in centres:
            rate += pixel_integrated_gaussian(
                rate.shape, r0 + 2 * k, c0 - 2 * k, sigma, amp / (2 * np.pi * sigma**2)
            )
        frames.append(rng.poisson(rate).astype(np.float32))
    return np.stack(frames)


# Detector spans this many PSF widths; held fixed while the pitch changes.
_L_OVER_W = 128.0
_GAMMA, _REF, _M0 = 0.0, 1.0, 0.1


def _z_at(rho, floor_policy, rng):
    """Calibrated z for one rung of the resolution ladder.

    The physical scene is fixed and only the pixel pitch changes, so with
    oversampling rho = w/Delta the frame is (128 rho)^2 pixels, widths scale
    as rho, and the per-pixel background and amplitude fall as 1/rho**2 at
    constant flux.
    """
    n_side = _L_OVER_W * rho
    bg, amp = 0.44 / rho**2, 3.0 / rho**2
    widths = rng.normal(1.5 * rho, 0.2 * rho, 1000).clip(0.5)
    floor = _floor_from_widths(widths) if floor_policy == "census" else 1.0
    ceiling = _ceiling_from_widths(widths, floor)
    grid = np.asarray(
        _auto_sigma_grid(floor, ceiling, _GAMMA, _REF, _M0, bg, amp, n_side, n_side),
        dtype=float,
    )
    return _frag_calibrated_z(grid, _GAMMA, _REF, _M0, n_side, n_side)


def test_detection_floor_is_invariant_under_refinement():
    """256x more pixels at constant flux must not move the threshold.

    The matched-filter floor is F_min = 2 sqrt(pi) z sigma_px sqrt(U), and
    sigma_px sqrt(U) = (w/Delta) sqrt(b Delta**2) = w sqrt(b) is invariant, so
    the whole pitch dependence of the detection floor rides on z alone.
    """
    rungs = (1, 2, 4, 8, 16)
    census = [_z_at(r, "census", np.random.default_rng(1)) for r in rungs]
    pinned = [_z_at(r, "pixel", np.random.default_rng(1)) for r in rungs]

    spread = max(census) / min(census)
    assert spread < 1.05, (
        f"a measured floor must hold z flat across the ladder; got {census} "
        f"(spread {spread:.3f}) over a 256x change in pixel count"
    )

    # The guard: the defect this test exists to catch must actually show up
    # here, or a flat `census` proves nothing about the test's sensitivity.
    assert max(pinned) / min(pinned) > 1.20
    assert pinned == sorted(pinned), "the pixel-pinned penalty is monotone in N"


def test_floor_tracks_the_width_distribution():
    """The floor is proportional to the width scale, which is the invariance.

    Like the ceiling, this is a percentile with a margin, not an extremum: it
    is allowed to sit above the single narrowest survivor (that is what makes
    it robust), so what is asserted is that it sits below the *robust* narrow
    end and scales with the family it measured.
    """
    base = np.random.default_rng(0).normal(1.5, 0.2, 1000).clip(0.5)
    ratios = []
    for rho in (2, 4, 8, 16):
        widths = base * rho  # one peak family, imaged at four pitches
        floor = _floor_from_widths(widths)
        assert floor < np.percentile(widths, 5.0)
        assert floor > widths.min() / 3.0
        ratios.append(floor / rho)
    assert max(ratios) / min(ratios) < 1.05, (
        f"floor/rho must be constant for the calibration to be pitch-"
        f"invariant; got {ratios}"
    )


def test_floor_never_goes_below_the_sampling_limit():
    # A peak family narrower than a pixel is the grid's constraint, not the
    # census's, and the floor must say so rather than extrapolate.
    assert _floor_from_widths(np.full(1000, 0.6)) == _SAMPLING_MIN_SIGMA


def test_floor_declines_to_quote_on_too_few_peaks():
    assert _floor_from_widths(np.array([1.0, 2.0, 3.0])) is None


def test_auto_floor_measures_the_data_and_finds_the_peaks():
    sigma = 3.0
    frames = dense_frames(sigma)
    finder = MatrixFreeSparseRBFPeakFinder(
        min_sigma=None, max_sigma=None, num_sigmas=None, show_steps=False
    )
    results = finder.find_peaks_batch(frames)
    # The floor sits below the true width but not down at the pixel pin.
    assert _SAMPLING_MIN_SIGMA <= finder.min_sigma < sigma
    assert finder.min_sigma > 1.2, (
        f"a 3 px peak family should lift the floor well off the {_SAMPLING_MIN_SIGMA} "
        f"px sampling limit; got {finder.min_sigma}"
    )
    assert sum(len(r) for r in results) >= 36


def test_explicit_min_sigma_is_never_second_guessed():
    frames = dense_frames(3.0)
    finder = MatrixFreeSparseRBFPeakFinder(
        min_sigma=1.25, max_sigma=None, num_sigmas=None, show_steps=False
    )
    finder.find_peaks_batch(frames)
    assert finder.min_sigma == pytest.approx(1.25)
