"""The convex-hull stage as an optional filter.

The hull stage does two jobs: it measures each peak's intensity, and -- because
a candidate it cannot fit a region to is dropped -- it filters the finder's
output.  `hull_filter=False` keeps the measurement and drops the filtering, for
the day the finder's own per-peak metrics are trusted to do that job.
"""

import numpy as np

from subhkl.integration.worker import _aperture_intensity, process_single_image

DET_CONFIG = {
    "m": 64,
    "n": 64,
    "width": 0.1,
    "height": 0.1,
    "center": [0, 0, 0.2],
    "vhat": [0, 1, 0],
    "uhat": [1, 0, 0],
    "panel": "flat",
}

INTEGRATION_PARAMS = {
    "peak_center_box_size": 3,
    "peak_smoothing_window_size": 3,
    "peak_minimum_pixels": 3,
    "peak_minimum_signal_to_noise": 0.0,
    "peak_pixel_outlier_threshold": 4.0,
    "region_growth_distance_threshold": 1.5,
    "region_growth_minimum_intensity": 20.0,
    "region_growth_maximum_pixel_radius": 5.0,
    "region_growth_minimum_sigma": None,
}


def _image_with_one_fittable_peak():
    """A bright compact peak the hull stage can fit, and a dim spot it cannot.

    The region grower only admits pixels above `region_growth_minimum_intensity`
    (20 here), so the second candidate cannot grow to `peak_minimum_pixels` and
    is rejected by the hull stage while still sitting on real counts.
    """
    image = np.full((64, 64), 5.0)
    image[18:23, 18:23] = 400.0  # fittable
    image[44:46, 44:46] = 12.0  # real counts, below the growth threshold
    return image


def _run(hull_filter):
    image = _image_with_one_fittable_peak()
    rows = np.array([20.0, 45.0])
    cols = np.array([20.0, 45.0])
    widths = np.array([2.0, 2.0])
    deviance = np.array([5000.0, 40.0])
    residual = np.array([1.1, 1.3])
    finder_info = ("sparse_rbf", {}, (rows, cols, widths, deviance, residual))

    params = dict(INTEGRATION_PARAMS)
    params["hull_filter"] = hull_filter

    res, log_msg = process_single_image(
        0,
        "test-image",
        1,
        image,
        DET_CONFIG,
        finder_info,
        params,
        (None, None),
        (np.eye(3), None, 0.5, 3.5),
    )
    return res, log_msg


def test_hull_filter_on_is_the_unchanged_default():
    """With the filter on, a candidate the hull stage cannot fit is dropped."""
    res, _ = _run(hull_filter=True)
    assert res is not None, "the fittable peak should survive"
    assert res["count"] == 1, f"expected only the fittable peak, got {res['count']}"
    assert res["intensity"][0] > 0.0


def test_hull_filter_off_keeps_the_rejected_candidate_with_an_aperture_intensity():
    """With the filter off, both candidates are reported and both have numbers."""
    res, log_msg = _run(hull_filter=False)
    assert res is not None
    assert res["count"] == 2, f"expected both candidates, got {res['count']}"

    intensities = np.array(res["intensity"], dtype=float)
    assert np.all(np.isfinite(intensities)), (
        f"switching the filter off must not emit undefined intensities: {intensities}"
    )

    # The aperture-measured peak sits on counts above its local background, so
    # its intensity is positive, and it is quoted with a real angular size
    # rather than the 0.0 that would claim a point-like peak.
    radii = np.array(res["radii"], dtype=float)
    assert np.all(radii > 0.0), f"aperture peak has no angular size: {radii}"

    # The finder's own per-peak metrics travel with it, which is what a
    # downstream filter would use in the hull's place.
    assert len(res["deviance"]) == 2
    assert len(res["residual_deviance"]) == 2

    assert "hull filter off" in log_msg, log_msg
    assert "1 kept by aperture" in log_msg, log_msg


def test_aperture_intensity_recovers_a_known_flux():
    """The fallback estimator is unbiased on a noiseless disc over flat background."""
    image = np.full((60, 60), 7.0)
    mask = np.ones((60, 60), dtype=bool)

    yy, xx = np.mgrid[0:60, 0:60]
    disc = (yy - 30.0) ** 2 + (xx - 30.0) ** 2 <= 9.0
    n_disc = int(disc.sum())
    per_pixel = 25.0
    image[disc] += per_pixel

    intensity, sigma = _aperture_intensity(image, mask, 30.0, 30.0, 4.0)

    expected = n_disc * per_pixel
    assert np.isclose(intensity, expected, rtol=0.02), (
        f"aperture intensity {intensity:.2f} != injected flux {expected:.2f}"
    )
    assert sigma > 0.0

    # An empty field yields no signal, within its own noise.
    flat, flat_sigma = _aperture_intensity(
        np.full((60, 60), 7.0), mask, 30.0, 30.0, 4.0
    )
    assert abs(flat) <= 3.0 * flat_sigma, f"background alone gave {flat:.2f}"
