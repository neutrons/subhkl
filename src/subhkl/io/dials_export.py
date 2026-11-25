"""Export subhkl HDF5 outputs to the DIALS data model.

Every subhkl CLI program can emit its results as a DIALS
``ExperimentList`` (``.expt``) and ``reflection_table`` (``.refl``) pair via a
``--dials`` / ``-dials`` flag.  This module contains the single converter that
turns any subhkl HDF5 output into those two files, plus helpers for deriving the
output filenames.

subhkl and DIALS share a laboratory frame in which ``Z`` is along the incident
beam and ``Y`` points up.  Because every model here (beam, detector, crystal,
goniometer) is copied out of that one subhkl frame, the resulting DIALS
experiment is internally self-consistent and reproduces subhkl's own physics:
``s1 = s0 + S * U * B * h`` where ``S`` is the per-frame goniometer setting
rotation.  Lengths are converted from metres (subhkl) to millimetres (DIALS);
the Busing-Levy ``B`` matrix subhkl stores already follows the crystallographic
(no ``2*pi``) convention DIALS expects, so ``A = U @ B`` maps across directly.

The DIALS/dxtbx/cctbx stack is an optional, heavy (conda-installed) dependency
and is imported lazily so the rest of subhkl runs without it.  See
``docs/source/dials_export.rst`` for the closest DIALS CLI parallel of each
subhkl program.
"""

from __future__ import annotations

import math
import os
import uuid

import h5py
import numpy as np

from subhkl.config import beamlines
from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix
from subhkl.io import dials_imageset

DIALS_IMPORT_ERROR = (
    "The --dials export requires the DIALS/dxtbx/cctbx toolkit, which is not "
    "installed. It is distributed through conda, e.g.\n"
    "    conda install -c conda-forge dials\n"
    "or as part of a full DIALS installation (https://dials.github.io/). Once "
    "DIALS is importable, re-run with --dials."
)

_H5_SUFFIXES = (".nxs.h5", ".h5", ".hdf5", ".mtz", ".expt", ".refl")


# Namespace for the deterministic experiment identifiers (uuid5) below.
_EXPERIMENT_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/neutrons/subhkl"
)


def experiment_identifier(source_path, frame):
    """Globally unique, reproducible DIALS experiment identifier for a frame.

    dials.import gives every experiment a UUID so experiments from separate
    imports can be combined (dials.combine_experiments refuses duplicate
    identifiers).  Deriving a uuid5 from the image stack's absolute path and the
    frame number keeps that uniqueness across runs while staying stable: every
    export backed by the same stack (finder, indexer, predictor, integrator)
    gives a frame the same identifier, as identifiers persist through a native
    DIALS pipeline, and re-exporting reproduces it exactly.
    """
    key = f"{os.path.realpath(os.fspath(source_path))}#frame={int(frame)}"
    return str(uuid.uuid5(_EXPERIMENT_ID_NAMESPACE, key))


def _require_dials():
    """Import the dxtbx/dials symbols we need, or raise a helpful error."""
    try:
        from dials.array_family import flex
        from dxtbx.model import (  # noqa: F401
            BeamFactory,
            Crystal,
            Detector,
            Experiment,
            ExperimentList,
            Goniometer,
        )
    except Exception as exc:  # pragma: no cover - exercised only without DIALS
        raise ImportError(DIALS_IMPORT_ERROR) from exc

    from dxtbx import model

    return model, flex


def strip_known_extension(path: str) -> str:
    """Strip a recognised data-file suffix, falling back to os.path.splitext."""
    for suffix in _H5_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    root, _ = os.path.splitext(path)
    return root


def dials_output_paths(base_path: str, prefix: str | None = None) -> tuple[str, str]:
    """Return ``(expt_path, refl_path)`` derived from a command's output path.

    ``prefix`` overrides the stem entirely; otherwise the stem is ``base_path``
    with any recognised data extension removed.
    """
    stem = prefix if prefix else strip_known_extension(base_path)
    return stem + ".expt", stem + ".refl"


# ---------------------------------------------------------------------------
# HDF5 reading helpers
# ---------------------------------------------------------------------------


def _get(f, key, default=None):
    return f[key][()] if key in f else default


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _floats(seq):
    """Coerce a sequence to a tuple of plain Python floats for dxtbx/flex."""
    return tuple(float(x) for x in np.asarray(seq).ravel())


def _mean_wavelength(f, lam):
    if lam is not None and len(lam) > 0:
        finite = lam[np.isfinite(lam) & (lam > 0)]
        if finite.size:
            return float(np.mean(finite))
    wl = _get(f, "instrument/wavelength")
    if wl is not None and len(wl) >= 2:
        return float((wl[0] + wl[1]) / 2.0)
    return 1.0


# The ``quantity`` attribute that marks ``peaks/sigma`` as the finder's per-peak
# Gaussian width in pixels, as opposed to the integrator's uncertainty on the
# intensity (the two share a dataset name).  Kept equal to
# ``subhkl.viz.replay.WIDTH_QUANTITY``, which is not imported here because that
# module pulls in the integration stack.
_WIDTH_QUANTITY = "gaussian_width_pixels"

# Variance of a still's frame coordinate: a uniform distribution over one
# image, which is what dials.find_spots reports for a single-image centroid.
_STILL_Z_VARIANCE = 1.0 / 12.0


def _collect_flat_peaks(f):
    """Read the flat ``peaks/*`` array schema (finder/indexer/integrator).

    Returns a dict of per-peak arrays or ``None`` if no flat peak arrays exist.
    """
    n = None
    for probe in ("peaks/intensity", "peaks/h", "peaks/pixel_r", "peaks/two_theta"):
        if probe in f:
            n = len(f[probe])
            break
    if n is None:
        return None

    def col(key):
        return f[key][()] if key in f else None

    return {
        "n": n,
        "pixel_r": col("peaks/pixel_r"),
        "pixel_c": col("peaks/pixel_c"),
        "xyz": col("peaks/xyz"),
        "h": col("peaks/h"),
        "k": col("peaks/k"),
        "l": col("peaks/l"),
        "lambda": col("peaks/lambda"),
        "intensity": col("peaks/intensity"),
        "sigma": col("peaks/sigma"),
        # True when peaks/sigma is the finder's spot width, not an intensity
        # sigma: tagged as such by the finder, or carried without any
        # intensity (an intensity sigma never is) -- the indexer copies the
        # finder's peaks without the tag.
        "sigma_is_width": (
            "peaks/sigma" in f
            and (
                _decode(f["peaks/sigma"].attrs.get("quantity", "")) == _WIDTH_QUANTITY
                or "peaks/intensity" not in f
            )
        ),
        "two_theta": col("peaks/two_theta"),
        "azimuthal": col("peaks/azimuthal"),
        "bank": col("bank") if "bank" in f else col("peaks/bank"),
        "run_index": col("peaks/run_index"),
        "image_index": col("peaks/image_index"),
    }


def _collect_predicted_peaks(f):
    """Read the ``banks/<img>/`` group schema written by peak_predictor."""
    if "banks" not in f:
        return None

    bank_ids = _get(f, "bank_ids")
    rows_r, rows_c, hs, ks, ls, wls, banks, images = [], [], [], [], [], [], [], []
    # Each ``banks/<img>`` group is one panel image of the stack, not a frame;
    # hdf5_to_dials maps images to frames from the stack's run boundaries.
    for key in sorted(f["banks"].keys(), key=lambda s: int(s)):
        grp = f[f"banks/{key}"]
        i = grp["i"][()]
        j = grp["j"][()]
        count = len(i)
        # subhkl predictor stores (i=row, j=col)
        rows_r.append(i)
        rows_c.append(j)
        hs.append(grp["h"][()])
        ks.append(grp["k"][()])
        ls.append(grp["l"][()])
        wls.append(grp["wavelength"][()])
        img_key = int(key)
        bank_id = int(bank_ids[img_key]) if bank_ids is not None else img_key
        banks.append(np.full(count, bank_id, dtype=np.int64))
        images.append(np.full(count, img_key, dtype=np.int64))

    if not rows_r:
        return None

    return {
        "n": int(sum(len(r) for r in rows_r)),
        "pixel_r": np.concatenate(rows_r),
        "pixel_c": np.concatenate(rows_c),
        "xyz": None,
        "h": np.concatenate(hs),
        "k": np.concatenate(ks),
        "l": np.concatenate(ls),
        "lambda": np.concatenate(wls),
        "intensity": None,
        "sigma": None,
        "two_theta": None,
        "azimuthal": None,
        "bank": np.concatenate(banks),
        "run_index": None,
        "image_index": np.concatenate(images),
        "predicted": True,
    }


def _read_detector_calibration(f):
    """Read refined per-bank metrology from the ``detector_calibration`` group.

    Returns ``{bank_id: {center, uhat, vhat[, width, height]}}``. subhkl programs
    that refine the detector store the calibrated geometry here rather than in
    ``beamlines.json``, so the export must apply it to reproduce the geometry the
    program actually used.
    """
    grp = f.get("detector_calibration")
    if grp is None:
        return {}

    calibration: dict[int, dict] = {}
    for bank_key in grp:
        try:
            bank_id = int(bank_key.replace("bank_", ""))
        except ValueError:
            continue
        sub = grp[bank_key]
        entry = {}
        for name in ("center", "uhat", "vhat"):
            if name in sub:
                entry[name] = np.asarray(sub[name][()], dtype=float)
        for name in ("width", "height"):
            if name in sub:
                entry[name] = float(sub[name][()])
        if entry:
            calibration[bank_id] = entry
    return calibration


def _read_goniometer_offsets(f, n_axes, names):
    """Read per-axis angular zero-point offsets (degrees), aligned to axis order.

    ``goniometer/offsets`` may be a flat array (positional per axis) or a group
    keyed by motor name. Mirrors how the predictor maps offsets onto axes, so the
    reconstructed rotation matches the geometry subhkl actually used. Returns
    ``None`` when no usable offsets are present.
    """
    off = f.get("goniometer/offsets")
    if off is None:
        return None

    result = np.zeros(n_axes, dtype=float)
    if isinstance(off, h5py.Group):
        if names is None:
            return None
        for i, name in enumerate(names):
            if name in off:
                result[i] = float(off[name][()])
    else:
        raw = np.asarray(off[()], dtype=float).ravel()
        count = min(len(raw), n_axes)
        result[:count] = raw[:count]
    return result


def _read_sample_offset(f):
    """Read the sample position offset (metres) from ``goniometer/translations``.

    subhkl mounts the sample off the rotation origin; this 3-vector is its
    displacement in the sample frame. Per-axis (2-D) translations reduce to the
    innermost stage, which is the sample position. Returns ``None`` when absent or
    zero (i.e. the sample already sits at the origin).
    """
    off = _get(f, "goniometer/translations")
    if off is None:
        return None
    off = np.asarray(off, dtype=float)
    if off.ndim > 1:
        off = off[-1]
    off = off[:3]
    if not np.any(off):
        return None
    return off


def _image_frames(f, layout):
    """Map each stack image index to its frame (goniometer setting).

    Uses the image stack layout when there is one, otherwise the file's own
    ``file_offsets`` run boundaries, otherwise groups images by identical
    per-image ``goniometer/R`` (or ``goniometer/angles``) rows in order of first
    appearance. Returns ``None`` when none of these is available.
    """
    if layout is not None and "frame_of" in layout:
        return np.asarray(layout["frame_of"], dtype=int)

    offsets = _get(f, "file_offsets")
    n_images = None
    for key in ("bank_ids", "goniometer/R", "goniometer/angles"):
        if key in f:
            n_images = int(f[key].shape[0])
            break
    if offsets is not None and len(offsets) and n_images is not None:
        offsets = np.asarray(offsets, dtype=int)
        return np.searchsorted(offsets, np.arange(n_images), side="right") - 1

    for key in ("goniometer/R", "goniometer/angles"):
        rows = _get(f, key)
        if rows is None:
            continue
        rows = np.asarray(rows, dtype=float).reshape(len(rows), -1)
        frames = np.empty(len(rows), dtype=int)
        seen = []
        for i, row in enumerate(rows):
            for fr, ref in enumerate(seen):
                if np.allclose(row, ref, atol=1e-6):
                    frames[i] = fr
                    break
            else:
                frames[i] = len(seen)
                seen.append(row)
        return frames
    return None


def _frame_rotations(f, peaks, frame_ids):
    """Build a ``{frame_id: 3x3 rotation}`` map reproducing subhkl's per-frame R.

    Prefers a stored ``goniometer/R`` (per-peak or per-frame); otherwise rebuilds
    the rotation from ``goniometer/axes`` and ``goniometer/angles``, folding in the
    per-axis ``goniometer/offsets`` zero-points (which a stored ``R`` already
    includes).
    """
    n_peaks = peaks["n"] if peaks else 0
    run_index = peaks["run_index"] if peaks else None
    image_index = peaks["image_index"] if peaks else None

    R_data = _get(f, "goniometer/R")
    axes = _get(f, "goniometer/axes")
    angles = _get(f, "goniometer/angles")

    zero_offsets = None
    if axes is not None:
        names = _get(f, "goniometer/names")
        if names is not None:
            names = [_decode(x) for x in names]
        zero_offsets = _read_goniometer_offsets(f, len(np.asarray(axes)), names)

    rotations: dict[int, np.ndarray] = {}

    R_data = np.asarray(R_data) if R_data is not None else None
    per_peak_R = R_data is not None and R_data.ndim == 3 and R_data.shape[0] == n_peaks
    ang_data = np.asarray(angles) if angles is not None else None

    def first_image(frame_id):
        # The stack image of this frame's first peak, so a per-image R or
        # angles table (one row per panel image) is read at that frame's
        # images rather than at row ``frame_id``.
        if run_index is None or image_index is None:
            return None
        idx = np.where(np.asarray(run_index) == frame_id)[0]
        return int(np.asarray(image_index)[idx[0]]) if idx.size else None

    for frame_id in frame_ids:
        R = None
        img = first_image(frame_id)
        if per_peak_R and run_index is not None:
            idx = np.where(run_index == frame_id)[0]
            if idx.size:
                R = R_data[idx[0]]
        elif (
            R_data is not None
            and R_data.ndim == 3
            and img is not None
            and img < R_data.shape[0]
        ):
            R = R_data[img]
        elif R_data is not None and R_data.ndim == 3 and frame_id < R_data.shape[0]:
            R = R_data[frame_id]
        elif R_data is not None and R_data.ndim == 2 and R_data.shape == (3, 3):
            R = R_data

        if R is None and axes is not None and angles is not None:
            ang = ang_data
            row = None
            if (
                ang.ndim == 2
                and ang.shape[0] not in (1, n_peaks)
                and img is not None
                and img < ang.shape[0]
            ):
                row = ang[img]
            elif ang.ndim == 2 and ang.shape[0] == n_peaks and run_index is not None:
                idx = np.where(np.asarray(run_index) == frame_id)[0]
                row = ang[idx[0]] if idx.size else None
            elif ang.ndim == 2 and frame_id < ang.shape[0]:
                row = ang[frame_id]
            elif ang.ndim == 2 and ang.shape[0] == 1:
                row = ang[0]
            elif ang.ndim == 1:
                row = ang
            if row is not None:
                row = np.asarray(row, dtype=float)
                # Fold in per-axis zero-point offsets, exactly as subhkl does
                # (true_angle = angle + offset, before the axis orientation sign).
                if zero_offsets is not None and zero_offsets.shape == row.shape:
                    row = row + zero_offsets
                try:
                    R = calc_goniometer_rotation_matrix(np.asarray(axes), row)
                except Exception:  # noqa: BLE001 - bad axes/angles row -> identity
                    R = None

        rotations[frame_id] = np.eye(3) if R is None else np.asarray(R, dtype=float)

    return rotations, axes


# ---------------------------------------------------------------------------
# DIALS model builders
# ---------------------------------------------------------------------------


def _build_beam(model, f, lam, *, polychromatic=False, wavelength=None):
    ki = _get(f, "beam/ki_vec")
    ki = np.array([0.0, 0.0, 1.0]) if ki is None else np.asarray(ki, dtype=float)
    ki = ki / np.linalg.norm(ki)

    # subhkl is Laue (polychromatic), so a PolychromaticBeam -- carrying only a
    # direction, with the true per-reflection wavelengths in the reflection
    # table's ``wavelength`` column -- is the most faithful model. But standard
    # DIALS tools (dials.show, dials.image_viewer) assume a monochromatic ``s0``
    # and cannot open such an experiment, so the default is a monochromatic beam
    # at a nominal (peak) wavelength, exactly as laue-dials does; the true
    # per-reflection wavelengths are preserved in the reflection table regardless
    # of this beam wavelength. Opt into the PolychromaticBeam with
    # ``polychromatic=True`` when using Laue-aware tooling.
    beam = None
    if polychromatic and hasattr(model, "PolychromaticBeam"):
        try:
            # ``direction`` is the incident beam direction of travel (ki).
            beam = model.BeamFactory.make_polychromatic_beam(direction=_floats(ki))
        except Exception:  # noqa: BLE001 - API differs across dxtbx versions
            try:
                beam = model.PolychromaticBeam(_floats(ki))
            except Exception:  # noqa: BLE001 - fall back to a monochromatic beam
                beam = None
    if beam is None:
        wl = float(wavelength) if wavelength else float(_mean_wavelength(f, lam))
        # sample_to_source is antiparallel to the direction of beam travel (ki).
        beam = model.BeamFactory.make_beam(sample_to_source=_floats(-ki), wavelength=wl)

    # Mark the probe as neutron. The Probe enum's import location varies across
    # dxtbx builds (``dxtbx.model`` on some, the compiled ext on others), so derive
    # it from the beam itself to stay build-independent; builds without probe
    # support simply skip this.
    try:
        beam.set_probe(type(beam.get_probe()).neutron)
    except Exception:  # noqa: BLE001, S110 - older dxtbx lacks probe support
        pass
    return beam


def _panel_grid_positions(banks):
    """Map each bank id to a ``(row, col)`` montage cell for a flat 2D layout.

    subhkl bank ids commonly encode a physical grid in two digits -- e.g. CG4D
    ``34`` -> column 3, row 4; ``101`` -> column 10, row 1 -- so lay panels out on
    that grid to mirror the instrument's module/panel arrangement. The tens digit
    is the column and the units digit is the row (matching how the banks sit in
    the image viewer). Distinct row and column values are compacted to contiguous
    indices so missing banks leave a small gap rather than a huge empty montage.
    Falls back to a square raster grid (in the given order) when the ids don't
    encode a usable grid.
    """
    banks = [int(b) for b in banks]
    if banks and all(b >= 11 and b % 10 != 0 for b in banks):
        # Rows (units digit) run top-to-bottom in the viewer, so map them in
        # descending order for the units digit to increase downward.
        rows = {
            r: i for i, r in enumerate(sorted({b % 10 for b in banks}, reverse=True))
        }
        cols = {c: i for i, c in enumerate(sorted({b // 10 for b in banks}))}
        return [(rows[b % 10], cols[b // 10]) for b in banks]
    ncols = max(1, math.ceil(math.sqrt(len(banks)))) if banks else 1
    return [(i // ncols, i % ncols) for i in range(len(banks))]


def _apply_flat_projection(detector, banks):
    """Bake a flat 2D montage projection into every panel.

    A detector whose panels wrap around the sample projects onto a best-fit plane
    as tilted, foreshortened tiles in ``dials.image_viewer``. Setting an explicit
    per-panel 2D projection (a rotation + a pixel offset into a grid, as the
    ESS/ISIS neutron formats do) makes each panel display undistorted and
    perpendicular. Each panel is flipped 180 deg (r = -I) so its pixels read in
    the sample-facing sense. This is display-only: the real 3D panel geometry is
    untouched.
    """
    if len(detector) == 0:
        return
    for panel, (row, col) in zip(detector, _panel_grid_positions(banks)):
        fast_px, slow_px = panel.get_image_size()  # (fast, slow), in pixels
        # Flip each panel 180 deg in place (r = -I) so its pixels read in the
        # sample-facing sense, offsetting the translation by the panel extent so
        # the flipped panel stays within its own grid cell.
        t = (
            int(slow_px * row + (slow_px - 1)),
            int(fast_px * col + (fast_px - 1)),
        )
        panel.set_projection_2d((-1, 0, 0, -1), t)


def _build_detector(model, instrument, bank_ids, calibration=None, origin_shift=None):
    """Build a multi-panel dxtbx Detector, returning ``(detector, {bank: panel})``.

    ``calibration`` (from :func:`_read_detector_calibration`) overrides the
    nominal ``beamlines.json`` geometry per bank, so the exported detector matches
    the refined geometry a program actually used.

    ``origin_shift`` (metres, a lab-frame vector) is subtracted from every panel
    origin. This places subhkl's sample at the DIALS origin: subhkl casts the
    diffracted ray from ``s_lab`` (the offset sample position) to the panel, and
    translating the detector by ``-s_lab`` is equivalent to moving the ray origin
    to (0, 0, 0), which is where DIALS assumes the sample sits.
    """
    if instrument is None or instrument not in beamlines:
        return None, {}

    calibration = calibration or {}
    shift = None if origin_shift is None else np.asarray(origin_shift, dtype=float)
    detector = model.Detector()
    bank_to_panel: dict[int, int] = {}
    added_banks: list[int] = []
    config = beamlines[instrument]

    for bank in bank_ids:
        key = str(int(bank))
        if key not in config:
            continue
        cfg = config[key]
        cal = calibration.get(int(bank), {})
        m, n = int(cfg["m"]), int(cfg["n"])  # columns (fast), rows (slow)
        width = float(cal.get("width", cfg["width"]))
        height = float(cal.get("height", cfg["height"]))
        center = np.asarray(cal.get("center", cfg["center"]), dtype=float)
        if shift is not None:
            center = center - shift
        vhat = np.asarray(cal.get("vhat", cfg["vhat"]), dtype=float)

        if "uhat" in cal:
            # Refinement produces a flat panel basis even for curved configs.
            uhat = np.asarray(cal["uhat"], dtype=float)
        elif cfg.get("panel") == "curved":
            # dxtbx panels are planar; approximate the cylinder by its tangent
            # plane at the panel centre. Peak identity is preserved through the
            # stored pixel coordinates; absolute panel geometry is approximate.
            rhat = np.asarray(cfg["rhat"], dtype=float)
            uhat = np.cross(vhat, rhat)
            uhat = uhat / np.linalg.norm(uhat)
        else:
            uhat = np.asarray(cfg["uhat"], dtype=float)

        panel = detector.add_panel()
        panel.set_name(key)
        panel.set_type("SENSOR_PAD")
        # metres -> millimetres; fast=columns=uhat, slow=rows=vhat.
        panel.set_local_frame(_floats(uhat), _floats(vhat), _floats(center * 1000.0))
        pixel_fast = width / max(m - 1, 1) * 1000.0
        pixel_slow = height / max(n - 1, 1) * 1000.0
        # dials.image_viewer (rstbx) asserts each panel's two pixel dimensions are
        # bit-exactly equal and identical across panels. subhkl metrology stores
        # width and height as separate floats that agree only to ~1e-15 mm, so
        # faithful division yields spurious sub-picometre anisotropy that trips
        # that assert. Quantise to 1e-9 mm (1 pm, far finer than any real detector)
        # so genuinely square/uniform panels export as such, while real anisotropy
        # (microns) survives untouched.
        pixel_fast = round(pixel_fast, 9)
        pixel_slow = round(pixel_slow, 9)
        panel.set_pixel_size((float(pixel_fast), float(pixel_slow)))
        panel.set_image_size((int(m), int(n)))
        panel.set_trusted_range((-1.0, 1.0e9))
        # Key on the true dxtbx panel index (banks not in the config are skipped,
        # so the enumerate index would drift from the panel count).
        bank_to_panel[int(bank)] = len(detector) - 1
        added_banks.append(int(bank))

    _apply_flat_projection(detector, added_banks)
    return detector, bank_to_panel


def _build_crystal(model, f):
    U = _get(f, "sample/U")
    B = _get(f, "sample/B")
    if U is None or B is None:
        return None

    U = np.asarray(U, dtype=float).reshape(3, 3)
    B = np.asarray(B, dtype=float).reshape(3, 3)
    A = U @ B  # columns are the reciprocal lattice vectors a*, b*, c* (1/Angstrom)

    try:
        G = np.linalg.inv(A).T  # columns are the real-space vectors a, b, c
    except np.linalg.LinAlgError:
        return None

    sg = _decode(_get(f, "sample/space_group", b"P 1")) or "P 1"
    real_a, real_b, real_c = _floats(G[:, 0]), _floats(G[:, 1]), _floats(G[:, 2])
    try:
        crystal = model.Crystal(real_a, real_b, real_c, space_group_symbol=sg)
    except Exception:  # noqa: BLE001 - cctbx rejects the symbol -> P 1
        crystal = model.Crystal(real_a, real_b, real_c, space_group_symbol="P 1")
    return crystal


def _build_goniometer(model, R, axes):
    axis = (0.0, 1.0, 0.0)
    if axes is not None:
        axes_arr = np.asarray(axes, dtype=float)
        primary = axes_arr[0] if axes_arr.ndim == 2 else axes_arr
        primary = primary[:3]
        norm = np.linalg.norm(primary)
        if norm > 0:
            axis = _floats(primary / norm)
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    gonio = model.Goniometer(axis, identity, _floats(R))
    return gonio


# ---------------------------------------------------------------------------
# Top-level converter
# ---------------------------------------------------------------------------


def _file_has_images(path):
    try:
        with h5py.File(path, "r") as f:
            return "images" in f
    except OSError:
        return False


def hdf5_to_dials(
    h5_path,
    expt_path,
    refl_path,
    instrument=None,
    image_source=None,
    polychromatic=False,
    wavelength=None,
):
    """Convert any subhkl HDF5 output to a DIALS ``.expt`` / ``.refl`` pair.

    Parameters
    ----------
    h5_path : str
        Path to a subhkl HDF5 output (finder, indexer, predictor, integrator,
        or an image stack: reduce/merge, sum-images or a static mask).
    expt_path, refl_path : str
        Destination paths for the ExperimentList and reflection_table. An
        image-only input (no peak data) writes no ``.refl``, as ``dials.import``
        writes experiments only; the returned ``refl_path`` is then ``None``.
    instrument : str, optional
        Instrument name. Falls back to the ``instrument`` attribute stored in the
        HDF5 file. Required to build detector geometry.
    image_source : str, optional
        Path to the image stack (a ``reduce``/``merge`` HDF5 with an ``images``
        dataset) whose pixels back this output. When available, an ``ImageSet`` is
        attached so the ``.expt`` opens in DIALS image tools. Defaults to
        ``h5_path`` itself when that is an image stack.
    polychromatic : bool, optional
        Export a Laue ``PolychromaticBeam`` instead of the default monochromatic
        beam. More faithful to the physics, but standard DIALS tools cannot open
        it (they assume a monochromatic ``s0``). The true per-reflection
        wavelengths live in the reflection table either way.
    wavelength : float, optional
        Nominal (peak) wavelength in Angstroms for the monochromatic beam. Only
        sets the beam model for DIALS tooling; peak identity and per-reflection
        wavelengths are unaffected. Defaults to the data's representative
        wavelength.
    """
    model, flex = _require_dials()

    with h5py.File(h5_path, "r") as f:
        if instrument is None and "instrument" in f.attrs:
            instrument = _decode(f.attrs["instrument"])

        peaks = _collect_flat_peaks(f) or _collect_predicted_peaks(f)

        lam = peaks["lambda"] if peaks else None
        beam = _build_beam(
            model, f, lam, polychromatic=polychromatic, wavelength=wavelength
        )
        crystal = _build_crystal(model, f)
        calibration = _read_detector_calibration(f)
        sample_offset = _read_sample_offset(f)

        # Resolve the pixel-data stack (if any) and its (frame, panel) layout.
        # Without an explicit image_source, a file that records the stack it
        # came from (the ``source`` attribute subhkl steps write) is backed by
        # that stack, so its experiments keep the stack's identifiers and
        # images.  A moved or missing stack falls through to geometry only.
        recorded_source = _decode(f.attrs["source"]) if "source" in f.attrs else None
        stack_path = None
        if image_source and _file_has_images(image_source):
            stack_path = image_source
        elif "images" in f:
            stack_path = h5_path
        elif recorded_source and _file_has_images(recorded_source):
            stack_path = recorded_source

        layout = None
        stack_rotations = None
        if stack_path is not None:
            if stack_path == h5_path:
                layout = dials_imageset.read_image_stack_layout(f)
                angle_rows = (
                    dials_imageset.frame_angle_rows(f, layout) if layout else None
                )
                stack_axes = _get(f, "goniometer/axes")
            else:
                with h5py.File(stack_path, "r") as sf:
                    layout = dials_imageset.read_image_stack_layout(sf)
                    angle_rows = (
                        dials_imageset.frame_angle_rows(sf, layout) if layout else None
                    )
                    stack_axes = _get(sf, "goniometer/axes")
            if layout and angle_rows is not None and stack_axes is not None:
                stack_rotations = {}
                for fr, row in angle_rows.items():
                    try:
                        stack_rotations[fr] = calc_goniometer_rotation_matrix(
                            np.asarray(stack_axes), np.asarray(row)
                        )
                    except Exception:  # noqa: BLE001 - bad row -> identity
                        stack_rotations[fr] = np.eye(3)

        # Outputs that record only each peak's stack image (the predictor) get
        # its frame from the stack's run boundaries.
        if peaks and peaks["run_index"] is None and peaks["image_index"] is not None:
            image_frame = _image_frames(f, layout)
            if image_frame is not None:
                img = np.asarray(peaks["image_index"], dtype=int)
                peaks["run_index"] = image_frame[img]

        # Determine frames and detector banks. Peak-bearing outputs drive these
        # from the reflections; a bare image stack (reduce/merge) drives them from
        # the stack layout so each goniometer setting becomes an experiment.
        if peaks and peaks["run_index"] is not None:
            frame_ids = sorted({int(r) for r in np.asarray(peaks["run_index"])})
        elif layout is not None:
            frame_ids = list(layout["frame_ids"])
        else:
            frame_ids = [0]

        if peaks and peaks["bank"] is not None:
            bank_ids = sorted({int(b) for b in np.asarray(peaks["bank"])})
        elif layout is not None:
            bank_ids = list(layout["panel_bank_ids"])
        else:
            stored = _get(f, "bank_ids")
            if stored is not None:
                bank_ids = sorted({int(b) for b in np.asarray(stored)})
            elif instrument in beamlines:
                bank_ids = sorted(int(k) for k in beamlines[instrument])
            else:
                bank_ids = []

        # Peaks rarely land on every frame and bank, but the ImageSet needs one
        # experiment per stack frame and one panel per stack bank.  When the
        # stack covers everything the peaks refer to, take its frames and banks
        # so the images attach; frames without peaks become experiments with no
        # reflections.  A stack missing some of them is not the one these peaks
        # came from, so it gets geometry only.
        peak_frame_ids = frame_ids
        attach_images = False
        if layout is not None:
            attach_images = set(frame_ids) <= set(layout["frame_ids"]) and set(
                bank_ids
            ) <= set(layout["panel_bank_ids"])
            if attach_images:
                frame_ids = list(layout["frame_ids"])
                bank_ids = list(layout["panel_bank_ids"])
            else:
                print(
                    "Note: the peaks refer to frames or banks missing from the "
                    f"image stack {stack_path}; wrote geometry only."
                )

        detector, bank_to_panel = _build_detector(
            model, instrument, bank_ids, calibration
        )

        if peaks:
            rotations, axes = _frame_rotations(f, peaks, peak_frame_ids)
            # Frames without peaks take the stack's angles, as a bare stack
            # export does.
            for fr in frame_ids:
                if fr not in rotations:
                    rotations[fr] = (
                        stack_rotations.get(fr, np.eye(3))
                        if stack_rotations is not None
                        else np.eye(3)
                    )
        elif stack_rotations is not None:
            rotations = {fr: stack_rotations.get(fr, np.eye(3)) for fr in frame_ids}
            axes = stack_axes
        else:
            rotations, axes = _frame_rotations(f, peaks, frame_ids)

        frame_to_expt = {frame: i for i, frame in enumerate(frame_ids)}
        # Identifiers come from the image stack when there is one, so a frame
        # keeps its identifier across every step's export of that stack.
        id_source = stack_path if stack_path is not None else h5_path

        # subhkl's sample sits at s_lab = R_frame @ sample_offset. Translate each
        # frame's detector by -s_lab so the sample lands at the DIALS origin.
        frame_s_lab = {}
        frame_detectors = []
        experiments = model.ExperimentList()
        for frame in frame_ids:
            R = rotations[frame]
            s_lab = None
            frame_detector = detector
            if sample_offset is not None:
                s_lab = R @ sample_offset
                if detector is not None:
                    frame_detector = _build_detector(
                        model, instrument, bank_ids, calibration, origin_shift=s_lab
                    )[0]
            frame_s_lab[frame] = s_lab
            frame_detectors.append(frame_detector)
            gonio = _build_goniometer(model, R, axes)
            expt = model.Experiment(
                beam=beam,
                detector=frame_detector,
                goniometer=gonio,
                crystal=crystal,
                identifier=experiment_identifier(id_source, frame),
            )
            experiments.append(expt)

        # Per-peak sample position, for the s1 calculation below.
        peak_s_lab = None
        if sample_offset is not None and peaks and peaks["run_index"] is not None:
            peak_s_lab = np.array(
                [frame_s_lab.get(int(r), np.zeros(3)) for r in peaks["run_index"]]
            )

        # Experiments and panels follow the stack layout's order, which is the
        # order _attach_imageset hands out its frames and panels in.
        if attach_images:
            container_path = os.path.splitext(expt_path)[0] + "_images.h5"
            _attach_imageset(
                experiments, stack_path, layout, container_path, instrument
            )

    experiments.as_file(expt_path)
    # An image-only input (reduce/merge stack, sum-images, static mask) has no
    # reflections, so -- like dials.import -- it exports experiments only.  An
    # empty table with no columns would also break DIALS tools that expect at
    # least an ``id`` column (dials.combine_experiments).  A peaks file is a
    # reflection output even when it holds no peaks, so it still gets a .refl.
    if peaks is None:
        return expt_path, None
    reflections = _build_reflection_table(
        flex, peaks, frame_to_expt, bank_to_panel, experiments, peak_s_lab
    )
    reflections.as_file(refl_path)
    return expt_path, refl_path


def _attach_imageset(experiments, stack_path, layout, container_path, instrument):
    """Best-effort: write the pixel container and attach an ImageSet per frame.

    Any failure (missing dxtbx imageset API, unregistered format) leaves the
    experiments without an ImageSet, so the ``.expt`` still loads with
    ``check_format=False``.
    """
    try:
        with h5py.File(stack_path, "r") as sf:
            dials_imageset.write_image_container(
                container_path, sf["images"], layout, instrument
            )

        from subhkl.io.dxtbx_format import make_imageset

        imageset = make_imageset(container_path)
        for i, expt in enumerate(experiments):
            single = imageset[i : i + 1]
            single.set_beam(expt.beam)
            if expt.detector is not None:
                single.set_detector(expt.detector)
            if expt.goniometer is not None:
                single.set_goniometer(expt.goniometer)
            expt.imageset = single
    except Exception as exc:  # noqa: BLE001 - geometry is useful without it
        print(f"Note: could not attach DIALS ImageSet ({exc}); wrote geometry only.")


def _still_image_index(expt):
    """Index of a still experiment's image within its ImageSet (0 if none)."""
    try:
        return int(expt.imageset.indices()[0])
    except Exception:  # noqa: BLE001 - no ImageSet, or one without indices
        return 0


def _laue_s1(expt, hkl):
    """Diffracted wavevector of ``hkl`` at the wavelength where it diffracts.

    Laue: the reflection lands at the wavelength satisfying |s0 + q| = |s0|,
    i.e. lambda = -2 (u . q) / |q|^2 for unit beam direction u and
    q = S F A h. Returns ``None`` when no positive wavelength satisfies it.
    """
    S = np.asarray(expt.goniometer.get_setting_rotation()).reshape(3, 3)
    F = np.asarray(expt.goniometer.get_fixed_rotation()).reshape(3, 3)
    A = np.asarray(expt.crystal.get_A()).reshape(3, 3)
    q = S @ F @ A @ np.asarray(hkl, dtype=float)
    u = np.asarray(expt.beam.get_unit_s0(), dtype=float)
    qq = float(q @ q)
    if qq <= 0:
        return None
    lam = -2.0 * float(u @ q) / qq
    if lam <= 0:
        return None
    return u / lam + q


def _add_derived_geometry(flex, table, experiments):
    """Fill the geometry columns DIALS derives from positions and models.

    Mirrors what dials.index / dials.predict add, computed through the
    exported experiments so it is consistent with them:

    - ``xyzcal.px`` for indexed rows that lack it: where ``S F A h`` diffracts
      at its Laue wavelength, through the row's panel ((0, 0, 0) for unindexed
      rows or a ray missing the panel, as DIALS leaves unpredicted rows);
    - ``xyzobs.mm.value``/``.variance`` from ``xyzobs.px`` (DIALS'
      ``centroid_px_to_mm``) on indexed tables with positional variances;
    - ``xyzcal.mm`` from ``xyzcal.px``;
    - ``s1`` where missing: the ray to the row's position scaled to
      1/``wavelength``.
    """
    n = len(table)
    if n == 0 or not len(experiments):
        return
    ids = list(table["id"])
    panels = list(table["panel"])

    has_hkl = "miller_index" in table and experiments[0].crystal is not None
    if has_hkl and "xyzcal.px" not in table and experiments[0].goniometer:
        cal = []
        for i, hkl in enumerate(table["miller_index"]):
            pos = (0.0, 0.0, 0.0)
            if hkl != (0, 0, 0):
                expt = experiments[ids[i]]
                s1 = _laue_s1(expt, hkl)
                if s1 is not None:
                    try:
                        x, y = expt.detector[panels[i]].get_ray_intersection_px(
                            _floats(s1)
                        )
                        pos = (float(x), float(y), 0.0)
                    except RuntimeError:
                        pass
            cal.append(pos)
        table["xyzcal.px"] = flex.vec3_double(cal)

    # Strong spots alone stay as dials.find_spots writes them (pixels only);
    # dials.index adds the millimetre centroids, so indexed tables carry them.
    if (
        "miller_index" in table
        and "xyzobs.px.value" in table
        and "xyzobs.px.variance" in table
    ):
        table.centroid_px_to_mm(experiments)

    if "xyzcal.px" in table and "xyzcal.mm" not in table:
        mm = []
        for i, (x, y, z) in enumerate(table["xyzcal.px"]):
            panel = experiments[ids[i]].detector[panels[i]]
            mx, my = panel.pixel_to_millimeter((x, y))
            mm.append((float(mx), float(my), float(z)))
        table["xyzcal.mm"] = flex.vec3_double(mm)

    if "s1" not in table and "wavelength" in table:
        key = "xyzobs.px.value" if "xyzobs.px.value" in table else "xyzcal.px"
        if key in table:
            s1 = []
            for i, (x, y, _) in enumerate(table[key]):
                lam = float(table["wavelength"][i])
                lab = np.asarray(
                    experiments[ids[i]].detector[panels[i]].get_pixel_lab_coord((x, y))
                )
                norm = np.linalg.norm(lab)
                if lam > 0 and norm > 0:
                    v = lab / norm / lam
                    s1.append((float(v[0]), float(v[1]), float(v[2])))
                else:
                    s1.append((0.0, 0.0, 0.0))
            table["s1"] = flex.vec3_double(s1)


def _build_reflection_table(
    flex, peaks, frame_to_expt, bank_to_panel, experiments, peak_s_lab=None
):
    table = flex.reflection_table()

    # Keep the reflection identifiers consistent with the experiments.
    for i, expt in enumerate(experiments):
        table.experiment_identifiers()[i] = expt.identifier

    if not peaks or peaks["n"] == 0:
        return table

    n = peaks["n"]
    run_index = peaks["run_index"]
    bank = peaks["bank"]

    if run_index is not None:
        ids = np.array(
            [frame_to_expt.get(int(r), 0) for r in np.asarray(run_index)], dtype=int
        )
    else:
        ids = np.zeros(n, dtype=int)

    if bank is not None:
        panels = np.array(
            [bank_to_panel.get(int(b), 0) for b in np.asarray(bank)], dtype=int
        )
    else:
        panels = np.zeros(n, dtype=int)

    table["id"] = flex.int([int(x) for x in ids])
    table["panel"] = flex.size_t([int(p) for p in panels])

    # Miller indices (indexed / predicted / integrated outputs).
    if peaks["h"] is not None:
        hkl = [
            (int(np.rint(h)), int(np.rint(k)), int(np.rint(l)))
            for h, k, l in zip(peaks["h"], peaks["k"], peaks["l"])
        ]
        table["miller_index"] = flex.miller_index(hkl)

        # Resolution d (Angstroms) from the exported crystal's unit cell, as
        # DIALS computes it; unindexed (0, 0, 0) rows get 0.
        crystal = experiments[0].crystal if len(experiments) else None
        if crystal is not None:
            uc = crystal.get_unit_cell()
            table["d"] = flex.double([uc.d(m) if m != (0, 0, 0) else 0.0 for m in hkl])

    # Per-reflection wavelength (subhkl is polychromatic / Laue).
    if peaks["lambda"] is not None:
        table["wavelength"] = flex.double([float(x) for x in peaks["lambda"]])

    # Observed / calculated pixel positions: DIALS x=fast=column, y=slow=row.
    # The frame coordinate follows DIALS for stills: an observed centroid sits
    # mid-image, at z = (image index) + 0.5, the index being the experiment's
    # image in its ImageSet (what dials.find_spots writes on the same .expt; 0
    # without images), while a calculated position has z = 0, as
    # dials.predict gives a still.
    if peaks["pixel_r"] is not None and peaks["pixel_c"] is not None:
        obs_z = np.array([_still_image_index(e) + 0.5 for e in experiments])
        obs_z = obs_z[ids] if len(obs_z) else np.full(n, 0.5)
        xyz_obs = [
            (float(c), float(r), float(z))
            for r, c, z in zip(peaks["pixel_r"], peaks["pixel_c"], obs_z)
        ]
        xyz_px = [(x, y, 0.0) for x, y, _ in xyz_obs]
        # The integrator integrates at the predicted positions and writes those
        # back as the peak position, so integrated rows (hkl + intensity, no
        # finder width) carry it as dials.integrate's xyzcal.px.  subhkl
        # measures no separate centroid, so xyzobs.px.value -- which
        # dials.export and dials.scale read -- holds the same position.
        integrated_at_prediction = (
            peaks["h"] is not None
            and peaks["intensity"] is not None
            and not peaks.get("sigma_is_width")
        )
        if peaks.get("predicted") or integrated_at_prediction:
            table["xyzcal.px"] = flex.vec3_double(xyz_px)
        if not peaks.get("predicted"):
            table["xyzobs.px.value"] = flex.vec3_double(xyz_obs)

    # Finder detections: peaks/sigma is each spot's fitted Gaussian width in
    # pixels, so its square is the spot's positional variance in x and y (the
    # same isotropic sigma^2 the replay viewer draws).  The frame coordinate of
    # a still gets dials.find_spots' single-image value, 1/12.
    observed_width = (
        peaks.get("sigma_is_width")
        and peaks["sigma"] is not None
        and not peaks.get("predicted")
        and "xyzobs.px.value" in table
    )
    if observed_width:
        var = np.asarray(peaks["sigma"], dtype=float) ** 2
        table["xyzobs.px.variance"] = flex.vec3_double(
            [(float(v), float(v), _STILL_Z_VARIANCE) for v in var]
        )

    # Diffracted beam wavevector s1: direction from the sample to the peak's
    # lab-frame position, scaled to |s1| = 1/wavelength (reciprocal Angstroms).
    # With the sample translated to the origin, the peak position is xyz - s_lab.
    if peaks["xyz"] is not None and peaks["lambda"] is not None:
        xyz = np.asarray(peaks["xyz"], dtype=float)
        if peak_s_lab is not None:
            xyz = xyz - np.asarray(peak_s_lab, dtype=float)
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            directions = np.where(norms > 0, xyz / norms, 0.0)
        lam = np.asarray(peaks["lambda"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_lambda = np.where(lam > 0, 1.0 / lam, 0.0)
        s1 = directions * inv_lambda[:, None]
        table["s1"] = flex.vec3_double(
            [(float(v[0]), float(v[1]), float(v[2])) for v in s1]
        )

    _add_derived_geometry(flex, table, experiments)

    # Intensities.
    if peaks["intensity"] is not None:
        table["intensity.sum.value"] = flex.double(
            [float(x) for x in peaks["intensity"]]
        )
        # A spot width is not an uncertainty on the intensity.
        if peaks["sigma"] is not None and not peaks.get("sigma_is_width"):
            variance = np.asarray(peaks["sigma"], dtype=float) ** 2
            table["intensity.sum.variance"] = flex.double([float(x) for x in variance])

    # Flags describing what was computed.  Finder detections are DIALS' strong
    # spots (what dials.find_spots flags them).
    if observed_width:
        table.set_flags(flex.bool(n, True), table.flags.strong)
    if peaks.get("predicted"):
        table.set_flags(flex.bool(n, True), table.flags.predicted)
    if peaks["h"] is not None:
        # subhkl writes (0, 0, 0) for peaks it could not index.
        has_hkl = (
            (np.asarray(peaks["h"]) != 0)
            | (np.asarray(peaks["k"]) != 0)
            | (np.asarray(peaks["l"]) != 0)
        )
        table.set_flags(flex.bool([bool(x) for x in has_hkl]), table.flags.indexed)
    if peaks["intensity"] is not None:
        table.set_flags(flex.bool(n, True), table.flags.integrated_sum)

    return table
