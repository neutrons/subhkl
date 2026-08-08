"""Redrawing the finder and integrator plots from their HDF5 output.

The point of the commands under test is that a run can be told not to draw
anything -- which is what a benchmark sweep wants, since the plots cost both
rendering time and artifact space -- and still be looked at afterwards. So what
these tests check is that everything the plot needs survives in the two files,
and that the files are read back as what they are: a peak's fitted width is a
size on the detector, and an uncertainty on an intensity is not.
"""

import numpy as np
import h5py
import pytest
import scipy.special

from PIL import Image

from subhkl.io.parser import finder, finder_visualize, integrator_visualize
from subhkl.io.parser import rbf_integrator
from subhkl.viz import detector_assembly, replay

INSTRUMENT = "MANDI"
BANKS = [1, 2]
IMAGE_SHAPE = (256, 256)


def _write_images(path, runs=("frameA", "frameB")):
    """A merged image stack, one bank per run, as `merge-images` writes one."""
    rng = np.random.default_rng(0)
    stack = rng.poisson(2.0, size=(len(runs),) + IMAGE_SHAPE).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = INSTRUMENT
        f.create_dataset("images", data=stack)
        f.create_dataset("bank_ids", data=np.array(BANKS[: len(runs)], dtype=np.int32))
        f.create_dataset("goniometer/angles", data=np.zeros((len(runs), 1)))
        # (M, 4): axis direction plus the sense of rotation.
        f.create_dataset("goniometer/axes", data=np.array([[0.0, 1.0, 0.0, 1.0]]))
        f.create_dataset("instrument/wavelength", data=[2.0, 4.0])
        f["files"] = np.array([f"{name}.h5".encode("utf-8") for name in runs])
        f.create_dataset("file_offsets", data=np.arange(len(runs), dtype=np.int64))

    return path


def _write_finder_peaks(path, sigma_quantity=replay.WIDTH_QUANTITY, n_images=2):
    """A finder output file, as `Peaks.write_hdf5` writes one."""
    image_index = np.repeat(np.arange(n_images), 2)
    n_peaks = len(image_index)

    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = INSTRUMENT
        f["peaks/pixel_r"] = np.tile([100.0, 150.0], n_images)
        f["peaks/pixel_c"] = np.tile([120.0, 60.0], n_images)
        f["peaks/image_index"] = image_index
        f["peaks/run_index"] = image_index
        f["peaks/intensity"] = np.full(n_peaks, 500.0)
        f["peaks/radius"] = np.full(n_peaks, 0.01)
        f["peaks/sigma"] = np.full(n_peaks, 3.0)
        f["bank"] = np.array([BANKS[i % len(BANKS)] for i in image_index])
        if sigma_quantity is not None:
            f["peaks/sigma"].attrs["quantity"] = sigma_quantity

    return path


def _write_integrator_peaks(path, n_images=2):
    """An rbf-integrator output file, with the shape it fitted each peak."""
    image_index = np.repeat(np.arange(n_images), 2)
    n_peaks = len(image_index)

    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = INSTRUMENT
        f["peaks/pixel_r"] = np.tile([100.0, 150.0], n_images)
        f["peaks/pixel_c"] = np.tile([120.0, 60.0], n_images)
        f["peaks/image_index"] = image_index
        f["peaks/run_index"] = image_index
        f["peaks/intensity"] = np.full(n_peaks, 500.0)
        f["peaks/sigma"] = np.full(n_peaks, 20.0)
        f["peaks/var_u"] = np.full(n_peaks, 9.0)
        f["peaks/var_v"] = np.full(n_peaks, 4.0)
        f["peaks/cov_uv"] = np.full(n_peaks, 1.0)

    return path


def test_a_finder_width_is_read_as_a_peak_size(tmp_path):
    """The sparse-RBF finder's per-peak width becomes an isotropic covariance."""
    table = replay.read_peaks_table(_write_finder_peaks(tmp_path / "found.h5"))

    assert table.peaks.var_u == pytest.approx(9.0)  # sigma = 3 px
    assert table.peaks.var_v == pytest.approx(table.peaks.var_u)
    assert table.peaks.cov_uv == pytest.approx(0.0)
    assert table.instrument == INSTRUMENT


def test_an_intensity_sigma_is_not_drawn_as_a_size(tmp_path):
    """`peaks/sigma` from the non-RBF finders is an uncertainty, not a width.

    Drawing it as a radius would put a made-up size on every peak, so a file
    that does not say its sigma is a width gets no shape drawn at all.
    """
    path = _write_finder_peaks(tmp_path / "found.h5", sigma_quantity="intensity_sigma")
    table = replay.read_peaks_table(path)

    assert table.peaks.var_u is None
    assert table.peaks.peak_rows is not None


def test_a_file_predating_the_attribute_gets_no_size(tmp_path):
    """Older finder output carries no marker, and is not guessed about."""
    path = _write_finder_peaks(tmp_path / "found.h5", sigma_quantity=None)

    assert replay.read_peaks_table(path).peaks.var_u is None


def test_the_integrator_covariance_is_read_through(tmp_path):
    table = replay.read_peaks_table(_write_integrator_peaks(tmp_path / "int.h5"))

    assert table.peaks.var_u == pytest.approx(9.0)
    assert table.peaks.var_v == pytest.approx(4.0)
    assert table.peaks.cov_uv == pytest.approx(1.0)


def test_a_file_without_pixel_coordinates_is_refused(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f["peaks/intensity"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="pixel_r"):
        replay.read_peaks_table(path)


def test_peaks_from_a_different_image_file_are_refused(tmp_path):
    """The one mistake that would silently draw a wrong picture."""
    images = _write_images(tmp_path / "images.h5")
    peaks = _write_finder_peaks(tmp_path / "found.h5", n_images=5)

    with pytest.raises(ValueError, match="does not contain"):
        replay.replay_plots(
            images_filename=str(images),
            peaks_filename=str(peaks),
            suffix="-found",
            max_workers=1,
            show_progress=False,
        )


def test_an_unknown_instrument_asks_for_one(tmp_path):
    images = _write_images(tmp_path / "images.h5")
    with h5py.File(images, "a") as f:
        del f.attrs["instrument"]

    peaks = tmp_path / "found.h5"
    _write_finder_peaks(peaks)
    with h5py.File(peaks, "a") as f:
        del f.attrs["instrument"]

    with pytest.raises(ValueError, match="--instrument"):
        replay.replay_plots(
            images_filename=str(images),
            peaks_filename=str(peaks),
            suffix="-found",
            max_workers=1,
            show_progress=False,
        )


def test_one_plot_per_run_named_after_the_run(tmp_path):
    images = _write_images(tmp_path / "images.h5")
    peaks = _write_finder_peaks(tmp_path / "found.h5")

    written = replay.replay_plots(
        images_filename=str(images),
        peaks_filename=str(peaks),
        suffix="-found",
        output_dir=str(tmp_path),
        dpi=50,
        max_workers=1,
        show_progress=False,
    )

    assert [name.rsplit("/", 1)[-1] for name in written] == [
        "frameA-found.png",
        "frameB-found.png",
    ]
    assert all((tmp_path / name).stat().st_size > 0 for name in written)


def test_a_run_with_no_peaks_is_still_drawn(tmp_path):
    """An empty frame is a result, and the one a spot check is looking for."""
    images = _write_images(tmp_path / "images.h5")
    peaks = tmp_path / "found.h5"
    _write_finder_peaks(peaks)
    with h5py.File(peaks, "a") as f:
        # Move every peak into the first run, leaving the second one bare.
        f["peaks/run_index"][...] = 0

    written = replay.replay_plots(
        images_filename=str(images),
        peaks_filename=str(peaks),
        suffix="-found",
        output_dir=str(tmp_path),
        dpi=50,
        max_workers=1,
        show_progress=False,
    )

    assert len(written) == 2


def test_dpi_is_honoured(tmp_path):
    """The reason the option exists: 600 dpi is too heavy to keep per run."""
    images = _write_images(tmp_path / "images.h5", runs=("only",))
    peaks = _write_finder_peaks(tmp_path / "found.h5", n_images=1)

    sizes = {}
    for dpi in (50, 100):
        out_dir = tmp_path / str(dpi)
        out_dir.mkdir()
        (written,) = replay.replay_plots(
            images_filename=str(images),
            peaks_filename=str(peaks),
            suffix="-found",
            output_dir=str(out_dir),
            dpi=dpi,
            max_workers=1,
            show_progress=False,
        )
        with Image.open(written) as rendered:
            sizes[dpi] = rendered.size

    assert sizes[100][0] > sizes[50][0]
    assert sizes[100][1] > sizes[50][1]


def test_the_integrator_plots_carry_their_own_suffix(tmp_path):
    """`-pred.png`, the name `rbf-integrator` gives its own plots."""
    images = _write_images(tmp_path / "images.h5", runs=("only",))
    peaks = _write_integrator_peaks(tmp_path / "int.h5", n_images=1)

    (written,) = replay.replay_plots(
        images_filename=str(images),
        peaks_filename=str(peaks),
        suffix="-pred",
        output_dir=str(tmp_path),
        dpi=50,
        max_workers=1,
        show_progress=False,
    )

    assert written.endswith("only-pred.png")


def test_plots_land_next_to_the_peaks_file_by_default(tmp_path):
    images = _write_images(tmp_path / "images.h5", runs=("only",))
    results = tmp_path / "results"
    results.mkdir()
    peaks = _write_finder_peaks(results / "found.h5", n_images=1)

    (written,) = replay.replay_plots(
        images_filename=str(images),
        peaks_filename=str(peaks),
        suffix="-found",
        dpi=50,
        max_workers=1,
        show_progress=False,
    )

    assert written == str(results / "only-found.png")


def _gaussian_frame(
    centre, shape=IMAGE_SHAPE, sigma=2.0, amplitude=150.0, background=15.0
):
    """One Erf-integrated Gaussian peak on a Poisson background.

    Defaults to the size of the bank it is written against: the plots are drawn
    on the detector's own mesh, so a frame that is not the shape of the bank it
    claims to come from cannot be drawn at all.
    """
    rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    spread = sigma * np.sqrt(2.0) + 1e-6
    erf_r = scipy.special.erf((rows + 0.5 - centre[0]) / spread) - scipy.special.erf(
        (rows - 0.5 - centre[0]) / spread
    )
    erf_c = scipy.special.erf((cols + 0.5 - centre[1]) / spread) - scipy.special.erf(
        (cols - 0.5 - centre[1]) / spread
    )
    peak = amplitude * (np.pi / 2.0) * sigma**2 * erf_r * erf_c

    rng = np.random.default_rng(42)
    return rng.poisson(background + peak).astype(np.float32)


def _run_finder(images, peaks, algorithm, create_visualizations=False, **extra):
    """Run the real finder, told not to draw anything unless asked.

    The integration defaults are tuned for real frames; a single narrow
    synthetic spot needs the same failsafes `test_finder_integration` passes,
    or the hull stage discards it and there is nothing left to plot.
    """
    finder(
        filename=str(images),
        instrument=INSTRUMENT,
        output_filename=str(peaks),
        finder_algorithm=algorithm,
        create_visualizations=create_visualizations,
        show_progress=False,
        peak_local_max_min_relative_intensity=0.5,
        peak_local_max_min_pixel_distance=5,
        region_growth_minimum_intensity=20.0,
        peak_minimum_pixels=1,
        peak_minimum_signal_to_noise=2.0,
        **extra,
    )


def _write_single_frame(path, image):
    """A one-image stack in the form both the finder and the integrator read."""
    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = INSTRUMENT
        f.create_dataset("images", data=image[np.newaxis, ...])
        f.create_dataset("bank_ids", data=np.array([1], dtype=np.int32))
        f.create_dataset("goniometer/angles", data=np.zeros((1, 1)))
        f.create_dataset("goniometer/axes", data=[[0, 1, 0, 1]])
        f.create_dataset("instrument/wavelength", data=[2.0, 4.0])
    return path


def test_a_finder_run_can_be_replayed_afterwards(tmp_path):
    """The whole point: find with no plots, draw them later from the output.

    Runs the real finder, then the real command, and asks only that a plot
    comes out -- which it can only do if every layer the plot draws survived
    into the peaks file.
    """
    images = _write_single_frame(
        tmp_path / "images.h5", _gaussian_frame((128.0, 128.0))
    )
    peaks = tmp_path / "found.h5"

    _run_finder(images, peaks, "peak_local_max")

    with h5py.File(peaks, "r") as f:
        assert len(f["peaks/pixel_r"]) > 0

    finder_visualize(
        images_filename=str(images),
        peaks_filename=str(peaks),
        output_dir=str(tmp_path),
        dpi=50,
        max_workers=1,
        show_progress=False,
    )

    assert (tmp_path / "img0-found.png").stat().st_size > 0


def test_a_finder_run_records_which_algorithm_found_the_peaks(tmp_path):
    """`peaks/sigma` only means a width for one of the finders, so say which."""
    images = _write_single_frame(
        tmp_path / "images.h5", _gaussian_frame((128.0, 128.0))
    )
    peaks = tmp_path / "found.h5"

    _run_finder(images, peaks, "peak_local_max")

    with h5py.File(peaks, "r") as f:
        assert f.attrs["finder_algorithm"] == "peak_local_max"
        assert f["peaks/sigma"].attrs["quantity"] == "intensity_sigma"

    # ... and with no width recorded, no size is invented for the plot.
    assert replay.read_peaks_table(peaks).peaks.var_u is None


@pytest.mark.slow
def test_the_sparse_rbf_finder_records_a_width(tmp_path):
    """The finder that does fit a peak size says so, and the size comes back.

    Kept apart from the round-trip above because the basis-pursuit solve costs
    the better part of a minute, most of it compiling.
    """
    images = _write_single_frame(
        tmp_path / "images.h5", _gaussian_frame((50.0, 50.0), shape=(100, 100))
    )
    peaks = tmp_path / "found.h5"

    _run_finder(
        images,
        peaks,
        "sparse_rbf",
        sparse_rbf_alpha=4.0,
        sparse_rbf_min_sigma=1.0,
        sparse_rbf_max_sigma=5.0,
    )

    with h5py.File(peaks, "r") as f:
        assert f.attrs["finder_algorithm"] == "sparse_rbf"
        assert f["peaks/sigma"].attrs["quantity"] == replay.WIDTH_QUANTITY

    # Read back as a size on the detector: an isotropic covariance, positive,
    # and within the widths the finder was allowed to fit.
    table = replay.read_peaks_table(peaks)
    assert len(table.peaks.var_u) > 0
    assert np.all(table.peaks.var_u > 0)
    assert np.all(table.peaks.var_u <= 5.0**2)
    assert np.all(table.peaks.cov_uv == 0.0)


def test_an_integrator_run_can_be_replayed_afterwards(tmp_path):
    """The integrator keeps the shape it fitted, so the plot can be redrawn."""
    centre = (128.0, 128.0)
    images = _write_single_frame(tmp_path / "images.h5", _gaussian_frame(centre))

    predicted = tmp_path / "predicted.h5"
    with h5py.File(predicted, "w") as f:
        grp = f.create_group("banks/0")
        grp["i"] = np.array([centre[0]])
        grp["j"] = np.array([centre[1]])
        grp["h"], grp["k"], grp["l"] = np.array([1]), np.array([2]), np.array([3])
        grp["wavelength"] = np.array([3.0])

    integrated = tmp_path / "integrated.h5"
    rbf_integrator(
        filename=str(images),
        instrument=INSTRUMENT,
        integration_peaks_filename=str(predicted),
        output_filename=str(integrated),
        sigmas="1.0,2.0,3.0",
        alpha=0.5,
        gamma=2.0,
        create_visualizations=False,
        show_progress=False,
    )

    with h5py.File(integrated, "r") as f:
        assert f.attrs["instrument"] == INSTRUMENT
        for name in ("pixel_r", "pixel_c", "image_index", "var_u", "var_v", "cov_uv"):
            assert f[f"peaks/{name}"].shape == f["peaks/intensity"].shape

    integrator_visualize(
        images_filename=str(images),
        peaks_filename=str(integrated),
        output_dir=str(tmp_path),
        dpi=50,
        max_workers=1,
        show_progress=False,
    )

    assert (tmp_path / "img0-pred.png").stat().st_size > 0


def test_drawing_nothing_at_all_is_an_error(tmp_path):
    """A command whose only job is drawing must not exit quietly having failed.

    A frame that is not the shape of the bank it claims to come from cannot be
    drawn on that bank's mesh; one such run is reported and skipped, but if
    that is every run there is no result to hand back.
    """
    images = tmp_path / "images.h5"
    _write_single_frame(images, _gaussian_frame((50.0, 50.0), shape=(100, 100)))
    peaks = _write_finder_peaks(tmp_path / "found.h5", n_images=1)

    with pytest.raises(RuntimeError, match="could be drawn"):
        replay.replay_plots(
            images_filename=str(images),
            peaks_filename=str(peaks),
            suffix="-found",
            output_dir=str(tmp_path),
            dpi=50,
            max_workers=1,
            show_progress=False,
        )


def test_an_outline_grows_with_the_peak(tmp_path):
    """The point of the whole exercise: radius follows sigma, not a constant."""
    narrow_r, narrow_c = detector_assembly.peak_outline(50.0, 50.0, 4.0, 4.0, 0.0)
    wide_r, wide_c = detector_assembly.peak_outline(50.0, 50.0, 16.0, 16.0, 0.0)

    # sigma 2 -> 4 doubles the radius, and the outline stays centred.  The
    # ratio is exact; the absolute width is a hair under 2 * n_sigma * sigma
    # because 50 sampled points do not land exactly on the extremes.
    assert np.ptp(wide_c) == pytest.approx(2 * np.ptp(narrow_c))
    assert np.ptp(narrow_c) == pytest.approx(
        2 * 2.0 * detector_assembly.DEFAULT_N_SIGMA, rel=0.01
    )
    for points in (narrow_r, narrow_c, wide_r, wide_c):
        assert (np.max(points) + np.min(points)) / 2 == pytest.approx(50.0, abs=0.05)


def test_an_isotropic_peak_is_drawn_as_a_circle():
    rows, cols = detector_assembly.peak_outline(0.0, 0.0, 9.0, 9.0, 0.0)

    radius = np.sqrt(rows**2 + cols**2)
    assert np.allclose(radius, 3.0 * detector_assembly.DEFAULT_N_SIGMA)


def test_an_anisotropic_peak_keeps_its_shape_and_tilt():
    """A correlated covariance is an ellipse turned off the pixel axes."""
    rows, cols = detector_assembly.peak_outline(0.0, 0.0, 16.0, 4.0, 0.0)
    assert np.ptp(cols) > np.ptp(rows)  # wider than it is tall, untilted

    tilted_rows, tilted_cols = detector_assembly.peak_outline(0.0, 0.0, 10.0, 10.0, 6.0)
    # Equal variances with a positive correlation: the long axis runs diagonally,
    # so the extreme point is off both axes rather than on one of them.
    furthest = np.argmax(tilted_rows**2 + tilted_cols**2)
    assert abs(tilted_rows[furthest]) > 1.0
    assert abs(tilted_cols[furthest]) > 1.0


def test_n_sigma_sets_the_contour():
    one, _ = detector_assembly.peak_outline(0.0, 0.0, 4.0, 4.0, 0.0, n_sigma=1.0)
    three, _ = detector_assembly.peak_outline(0.0, 0.0, 4.0, 4.0, 0.0, n_sigma=3.0)

    assert np.ptp(three) == pytest.approx(3 * np.ptp(one))


def test_candidates_are_drawn_at_their_own_radius(tmp_path):
    """A candidate list carrying a width per peak is drawn peak by peak.

    Each outline is a separate `plot` call, and the wide candidate has to come
    out wider than the narrow one -- which is exactly what a constant marker
    size cannot show.
    """
    from subhkl.instrument.detector import Detector
    from subhkl.config import beamlines

    drawn = []

    class _RecordingAxes:
        def plot(self, x, y, **kwargs):
            drawn.append((np.asarray(x), np.asarray(y)))

    detector = Detector(beamlines[INSTRUMENT]["1"])
    # [intensity, row, col, sigma], the finder's candidate layout.
    candidates = [(100.0, 100.0, 1.0), (150.0, 150.0, 4.0)]

    detector_assembly._draw_outlines(
        _RecordingAxes(),
        detector,
        np.zeros(3),
        candidates,
        n_sigma=detector_assembly.DEFAULT_N_SIGMA,
        color="blue",
        label="Finder Candidates",
        wrapped=False,
        compress=lambda r: r,
    )

    assert len(drawn) == 2
    narrow, wide = (np.ptp(y) for _, y in drawn)
    assert wide == pytest.approx(4.0 * narrow, rel=0.05)


def test_a_candidate_with_no_usable_width_is_skipped():
    """A width of zero or NaN describes no peak, so nothing is drawn for it."""
    from subhkl.instrument.detector import Detector
    from subhkl.config import beamlines

    drawn = []

    class _RecordingAxes:
        def plot(self, x, y, **kwargs):
            drawn.append(x)

    detector_assembly._draw_outlines(
        _RecordingAxes(),
        Detector(beamlines[INSTRUMENT]["1"]),
        np.zeros(3),
        [(100.0, 100.0, 0.0), (110.0, 110.0, np.nan), (120.0, 120.0, 2.0)],
        n_sigma=detector_assembly.DEFAULT_N_SIGMA,
        color="blue",
        label="",
        wrapped=False,
        compress=lambda r: r,
    )

    assert len(drawn) == 1


_WORKER_DET = {
    "m": 64,
    "n": 64,
    "width": 0.1,
    "height": 0.1,
    "center": [0, 0, 0.2],
    "vhat": [0, 1, 0],
    "uhat": [1, 0, 0],
    "panel": "flat",
}

_WORKER_PARAMS = {
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


def _harvest(finder_info):
    """One bright peak through the real harvest worker, on a 64x64 panel."""
    from subhkl.integration.worker import process_single_image

    image = np.full((64, 64), 5.0)
    image[18:23, 18:23] = 400.0

    res, _ = process_single_image(
        0,
        "test-image",
        1,
        image,
        _WORKER_DET,
        finder_info,
        dict(_WORKER_PARAMS),
        (None, None),
        (np.eye(3), None, 0.5, 3.5),
    )
    return res


def test_the_finder_reports_widths_under_a_name_that_means_one_thing():
    """`sigma` is a width for one finder and an uncertainty for the others.

    Anything that draws a peak at its true size needs to know which it has, so
    the width also travels under its own key -- absent rather than ambiguous.
    """
    res = _harvest(
        (
            "sparse_rbf",
            {},
            (
                np.array([20.0]),
                np.array([20.0]),
                np.array([2.0]),
                np.array([5000.0]),
                np.array([1.1]),
            ),
        )
    )

    assert res["width"] == pytest.approx([2.0])
    assert res["sigma"] == pytest.approx(res["width"])


def test_a_finder_that_fits_no_width_reports_none():
    res = _harvest(("peak_local_max", {"min_pix": 3, "min_rel_intensity": 0.5}, None))

    assert res["width"] is None
    # `sigma` is still populated -- with an intensity uncertainty, which is why
    # it cannot be used as a size.
    assert len(res["sigma"]) == res["count"]


def test_the_inline_finder_plot_still_works_without_widths(tmp_path):
    """Regression guard: the finder draws its own plot for every algorithm.

    `peak_local_max` fits no width, so the plot falls back to plain markers
    rather than failing to find a size to draw.
    """
    images = _write_single_frame(
        tmp_path / "images.h5", _gaussian_frame((128.0, 128.0))
    )
    peaks = tmp_path / "found.h5"

    _run_finder(images, peaks, "peak_local_max", create_visualizations=True)

    assert (tmp_path / "img0-found.png").stat().st_size > 0
