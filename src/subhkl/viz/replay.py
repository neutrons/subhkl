"""Rebuild the unrolled-detector plots from HDF5 output, without reprocessing.

`finder` and `rbf-integrator` render their plots while they run, from data that
exists only for the duration of the run.  Switching the plots off to save the
rendering time and the disk space therefore also throws away the only way to
look at the result, short of repeating the search.

Nothing in those plots actually requires the search, though.  Every layer is
either in the peaks file the search writes -- the peak centres in pixel
coordinates, and their fitted widths -- or in the reduced (and optionally
merged) image file the search read, which holds the frames themselves and the
bank each one belongs to.  So the plots can be rebuilt afterwards from that
pair of files, and this module does exactly that: it reassembles the same
inputs the inline paths assemble and hands them to the same renderer, so a
replayed plot is the plot the run would have made.

The peaks file addresses images by `peaks/image_index`, which indexes the image
stack of the file the search ran on.  Pointing this at a *different* image file
is the one mistake that would produce a wrong picture rather than an error, so
the indices are checked against the stack before anything is drawn.
"""

import multiprocessing
import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial

import h5py
import numpy as np

from tqdm import tqdm

from subhkl.integration.api import Peaks
from subhkl.viz import detector_assembly

#: Value of the `quantity` attribute on `peaks/sigma` that marks the column as
#: the finder's per-peak Gaussian width in pixels, rather than the integrator's
#: uncertainty on the intensity.  The two share a dataset name but are not the
#: same measurement, and only the first one describes a peak's size on the
#: detector.  Files written before the attribute existed carry neither value,
#: and are drawn as bare centres.
WIDTH_QUANTITY = "gaussian_width_pixels"


@dataclass(frozen=True)
class PlotPeaks:
    """The part of a peaks table that `plot_unrolled_detector` reads.

    Mirrors the objects the finder and the RBF integrator build inline
    (`_RunPeaksFinder` and `RunPeaks` respectively): the renderer accepts any
    object carrying these attributes, and decides between drawing bare centres
    and drawing peak outlines on whether `var_u` is None.
    """

    image_index: np.ndarray
    peak_rows: np.ndarray
    peak_cols: np.ndarray
    var_u: np.ndarray | None = None
    var_v: np.ndarray | None = None
    cov_uv: np.ndarray | None = None


@dataclass(frozen=True)
class PeaksTable:
    """A peaks file's plot inputs, before they are split up per run."""

    peaks: PlotPeaks
    run_index: np.ndarray
    instrument: str | None


def _dataset(f, name):
    return f[name][()] if name in f else None


def _shape_columns(f):
    """Per-peak shape, as a covariance, from whichever form the file stores.

    The RBF integrator fits a full 2x2 covariance per peak and stores its three
    independent components.  The sparse-RBF finder fits one isotropic width,
    which is the same thing with `var_u == var_v` and no correlation -- the
    renderer already special-cases that and draws a circle.  Returns None when
    the file describes no shape at all, which is the signal to fall back to
    plain centres.
    """
    var_u = _dataset(f, "peaks/var_u")
    var_v = _dataset(f, "peaks/var_v")
    cov_uv = _dataset(f, "peaks/cov_uv")
    if var_u is not None and var_v is not None and cov_uv is not None:
        return var_u, var_v, cov_uv

    if "peaks/sigma" not in f:
        return None
    if f["peaks/sigma"].attrs.get("quantity") != WIDTH_QUANTITY:
        # `peaks/sigma` here is an uncertainty on the intensity, which says
        # nothing about how big the peak is.  Drawing it as a radius would
        # invent a size, so draw no size.
        return None

    width = np.asarray(f["peaks/sigma"][()], dtype=float)
    variance = width**2
    return variance, variance, np.zeros_like(variance)


def read_peaks_table(filename: str) -> PeaksTable:
    """Read the plot inputs from a finder or RBF-integrator peaks file."""
    with h5py.File(filename, "r") as f:
        missing = [
            name
            for name in ("peaks/pixel_r", "peaks/pixel_c", "peaks/image_index")
            if name not in f
        ]
        if missing:
            raise ValueError(
                f"{filename} is missing {', '.join(missing)}, so the peaks in it "
                "cannot be placed on a detector image. Peaks files written "
                "before pixel coordinates were recorded cannot be visualized; "
                "re-run the step that produced this file."
            )

        rows = np.asarray(f["peaks/pixel_r"][()], dtype=float)
        cols = np.asarray(f["peaks/pixel_c"][()], dtype=float)
        image_index = np.asarray(f["peaks/image_index"][()], dtype=int)

        run_index = _dataset(f, "peaks/run_index")
        run_index = (
            np.zeros(len(rows), dtype=int)
            if run_index is None
            else np.asarray(run_index, dtype=int)
        )

        shape = _shape_columns(f)
        instrument = f.attrs.get("instrument")
        if isinstance(instrument, bytes):
            instrument = instrument.decode("utf-8")

    var_u, var_v, cov_uv = shape if shape is not None else (None, None, None)

    return PeaksTable(
        peaks=PlotPeaks(
            image_index=image_index,
            peak_rows=rows,
            peak_cols=cols,
            var_u=var_u,
            var_v=var_v,
            cov_uv=cov_uv,
        ),
        run_index=run_index,
        instrument=instrument,
    )


def _resolve_instrument(explicit, table, images_filename):
    if explicit:
        return explicit
    if table.instrument:
        return table.instrument

    with h5py.File(images_filename, "r") as f:
        recorded = f.attrs.get("instrument")
    if isinstance(recorded, bytes):
        recorded = recorded.decode("utf-8")
    if recorded:
        return recorded

    raise ValueError(
        "Neither file records which instrument it belongs to "
        "(merged image files do not carry it); pass --instrument."
    )


def _select(peaks: PlotPeaks, mask) -> PlotPeaks:
    def take(column):
        return None if column is None else np.asarray(column)[mask]

    return PlotPeaks(
        image_index=peaks.image_index[mask],
        peak_rows=peaks.peak_rows[mask],
        peak_cols=peaks.peak_cols[mask],
        var_u=take(peaks.var_u),
        var_v=take(peaks.var_v),
        cov_uv=take(peaks.cov_uv),
    )


def _render(args):
    """Draw one run's plot.  Runs in a worker process, so it takes one tuple.

    Seven elements for the peak replays; an optional eighth carries a binary
    overlay (the static mask) drawn as its own translucent colour layer."""
    out_name, peaks, images, detectors, instrument, dpi, n_sigma, *rest = args
    overlay = rest[0] if rest else None

    import matplotlib.pyplot as plt

    from subhkl.viz.detector_assembly import plot_unrolled_detector

    if plt.get_backend().lower() != "agg":
        plt.switch_backend("Agg")

    plot_unrolled_detector(
        peaks,
        images,
        detectors,
        out_name=out_name,
        instrument=instrument,
        dpi=dpi,
        n_sigma=n_sigma,
        overlay=overlay,
    )
    return out_name


def replay_plots(
    images_filename: str,
    peaks_filename: str,
    suffix: str,
    instrument: str | None = None,
    output_dir: str | None = None,
    dpi: int = 600,
    n_sigma: float = detector_assembly.DEFAULT_N_SIGMA,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> list[str]:
    """Rebuild one unrolled-detector plot per run and return the paths written.

    `suffix` distinguishes the two callers' filenames, matching the names their
    inline plots already use, so a replayed plot lands where the original would
    have.
    """
    table = read_peaks_table(peaks_filename)
    instrument = _resolve_instrument(instrument, table, images_filename)

    images = Peaks(images_filename, instrument)
    if not images.image.ims:
        raise ValueError(f"{images_filename} holds no images to draw on.")

    known = set(images.image.ims)
    unknown = sorted(set(table.peaks.image_index.tolist()) - known)
    if unknown:
        raise ValueError(
            f"{peaks_filename} refers to images {unknown[:5]}"
            f"{'...' if len(unknown) > 5 else ''} that {images_filename} does not "
            f"contain (it has {len(known)}). These peaks were found in a "
            "different image file; pass the one the search actually ran on."
        )

    if output_dir is None:
        output_dir = os.path.dirname(peaks_filename)

    # Group the images by run first, so that a run with no surviving peaks is
    # still drawn -- an empty frame is a result worth seeing, and it is the
    # case a spot check is most likely to be looking for.
    runs = {}
    for img_key in sorted(images.image.ims):
        runs.setdefault(images.get_run_id(img_key), []).append(img_key)

    tasks = []
    for run_id, img_keys in runs.items():
        label = images.get_image_label(img_keys[0])
        out_name = os.path.join(output_dir, f"{label}{suffix}.png")
        tasks.append(
            (
                out_name,
                _select(table.peaks, table.run_index == run_id),
                {key: images.image.ims[key] for key in img_keys},
                {key: images.get_detector_by_img(key) for key in img_keys},
                instrument,
                dpi,
                n_sigma,
            )
        )

    if not tasks:
        return []

    if max_workers is None:
        max_workers = os.cpu_count()
    max_workers = max(1, min(max_workers, len(tasks)))

    written = []

    # Spawning a process costs a fresh import of the whole package, which is
    # not worth paying to draw a single plot.
    if max_workers == 1:
        for task in tasks:
            _attempt(partial(_render, task), task[0], written)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            futures = {executor.submit(_render, task): task[0] for task in tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Rendering unrolled plots",
                disable=not show_progress,
            ):
                _attempt(future.result, futures[future], written)

    # One run failing to draw is worth reporting and carrying on from; every
    # run failing means the command did its whole job and produced nothing,
    # which should not look like success to whatever called it.
    if not written:
        raise RuntimeError(
            f"None of the {len(tasks)} plot(s) could be drawn; see the errors above."
        )

    return sorted(written)


def _attempt(call, out_name, written):
    """Run one render, keeping the path it wrote or reporting why it failed."""
    try:
        written.append(call())
    except Exception:
        import traceback

        print(f"Visualization failed for {out_name}:")
        traceback.print_exc()
    else:
        tqdm.write(f"Saved: {written[-1]}")


def replay_mask_plots(
    images_filename: str,
    mask_filename: str,
    instrument: str | None = None,
    output_dir: str | None = None,
    dpi: int = 600,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> list[str]:
    """Draw the static mask over the frames it applies to, one plot per run.

    The same unrolled-detector rendering as the finder and integrator
    replays, with the mask drawn as its own translucent cyan overlay on the
    data -- an annotation layer, deliberately distinct from both Bragg peaks
    and background density (burning masked pixels into the intensity scale
    made them read as saturated peaks and shifted the shared colour scale).
    Written as ``<label>-mask.png`` next to the mask file (or into
    ``--output-dir``).
    """
    from subhkl.search.static_mask import load_mask_for_banks

    if instrument is None:
        with h5py.File(images_filename, "r") as f:
            recorded = f.attrs.get("instrument")
        if isinstance(recorded, bytes):
            recorded = recorded.decode("utf-8")
        if not recorded:
            raise ValueError(
                "The image file does not record its instrument; pass --instrument."
            )
        instrument = recorded

    images = Peaks(images_filename, instrument)
    if not images.image.ims:
        raise ValueError(f"{images_filename} holds no images to draw on.")

    img_keys = sorted(images.image.ims)
    shape = next(iter(images.image.ims.values())).shape
    banks = [images.image.bank_mapping.get(k, k) for k in img_keys]
    valid = load_mask_for_banks(mask_filename, banks, shape)

    if output_dir is None:
        output_dir = os.path.dirname(mask_filename)

    no_peaks = PlotPeaks(
        image_index=np.zeros(0, dtype=int),
        peak_rows=np.zeros(0),
        peak_cols=np.zeros(0),
    )

    runs = {}
    for img_key in img_keys:
        runs.setdefault(images.get_run_id(img_key), []).append(img_key)

    tasks = []
    for run_id, run_keys in runs.items():
        label = images.get_image_label(run_keys[0])
        out_name = os.path.join(output_dir, f"{label}-mask.png")
        overlay = {key: 1.0 - valid[img_keys.index(key)] for key in run_keys}
        tasks.append(
            (
                out_name,
                no_peaks,
                {key: images.image.ims[key] for key in run_keys},
                {key: images.get_detector_by_img(key) for key in run_keys},
                instrument,
                dpi,
                detector_assembly.DEFAULT_N_SIGMA,
                overlay,
            )
        )

    if max_workers is None:
        max_workers = os.cpu_count()
    max_workers = max(1, min(max_workers, len(tasks)))

    written = []
    if max_workers == 1:
        for task in tasks:
            _attempt(partial(_render, task), task[0], written)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            futures = {executor.submit(_render, task): task[0] for task in tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Rendering mask plots",
                disable=not show_progress,
            ):
                _attempt(future.result, futures[future], written)

    if not written:
        raise RuntimeError(
            f"None of the {len(tasks)} plot(s) could be drawn; see the errors above."
        )

    return sorted(written)
