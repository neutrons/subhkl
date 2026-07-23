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

import os

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


def _require_dials():
    """Import the dxtbx/dials symbols we need, or raise a helpful error."""
    try:
        from dxtbx.model import (  # noqa: F401
            BeamFactory,
            Crystal,
            Detector,
            Experiment,
            ExperimentList,
            Goniometer,
        )
        from dials.array_family import flex  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without DIALS
        raise ImportError(DIALS_IMPORT_ERROR) from exc

    import dxtbx.model as model

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
    rows_r, rows_c, hs, ks, ls, wls, banks, runs = [], [], [], [], [], [], [], []
    for run_idx, key in enumerate(sorted(f["banks"].keys(), key=lambda s: int(s))):
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
        runs.append(np.full(count, run_idx, dtype=np.int64))

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
        "run_index": np.concatenate(runs),
        "image_index": None,
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
    for bank_key in grp.keys():
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


def _frame_rotations(f, peaks, frame_ids):
    """Build a ``{frame_id: 3x3 rotation}`` map reproducing subhkl's per-frame R.

    Prefers a stored ``goniometer/R`` (per-peak or per-frame); otherwise rebuilds
    the rotation from ``goniometer/axes`` and ``goniometer/angles``, folding in the
    per-axis ``goniometer/offsets`` zero-points (which a stored ``R`` already
    includes).
    """
    n_peaks = peaks["n"] if peaks else 0
    run_index = peaks["run_index"] if peaks else None

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

    for frame_id in frame_ids:
        R = None
        if per_peak_R and run_index is not None:
            idx = np.where(run_index == frame_id)[0]
            if idx.size:
                R = R_data[idx[0]]
        elif R_data is not None and R_data.ndim == 3 and frame_id < R_data.shape[0]:
            R = R_data[frame_id]
        elif R_data is not None and R_data.ndim == 2 and R_data.shape == (3, 3):
            R = R_data

        if R is None and axes is not None and angles is not None:
            ang = np.asarray(angles)
            row = None
            if ang.ndim == 2 and frame_id < ang.shape[0]:
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
                except Exception:
                    R = None

        rotations[frame_id] = np.eye(3) if R is None else np.asarray(R, dtype=float)

    return rotations, axes


# ---------------------------------------------------------------------------
# DIALS model builders
# ---------------------------------------------------------------------------


def _build_beam(model, f, lam):
    ki = _get(f, "beam/ki_vec")
    ki = np.array([0.0, 0.0, 1.0]) if ki is None else np.asarray(ki, dtype=float)
    ki = ki / np.linalg.norm(ki)

    # subhkl is polychromatic (Laue): prefer a PolychromaticBeam that carries only
    # a direction, with the true per-reflection wavelengths living in the
    # reflection table's ``wavelength`` column. Fall back to a monochromatic beam
    # at the mean wavelength if this dxtbx build predates PolychromaticBeam.
    beam = None
    if hasattr(model, "PolychromaticBeam"):
        try:
            # ``direction`` is the incident beam direction of travel (ki).
            beam = model.BeamFactory.make_polychromatic_beam(direction=_floats(ki))
        except Exception:
            try:
                beam = model.PolychromaticBeam(_floats(ki))
            except Exception:
                beam = None
    if beam is None:
        # sample_to_source is antiparallel to the direction of beam travel (ki).
        beam = model.BeamFactory.make_beam(
            sample_to_source=_floats(-ki), wavelength=float(_mean_wavelength(f, lam))
        )

    try:
        from dxtbx.model import Probe

        beam.set_probe(Probe.neutron)
    except Exception:
        pass
    return beam


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
    config = beamlines[instrument]

    for panel_index, bank in enumerate(bank_ids):
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
        panel.set_pixel_size((float(pixel_fast), float(pixel_slow)))
        panel.set_image_size((int(m), int(n)))
        panel.set_trusted_range((-1.0, 1.0e9))
        bank_to_panel[int(bank)] = panel_index

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
    except Exception:
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
    except Exception:
        return False


def hdf5_to_dials(h5_path, expt_path, refl_path, instrument=None, image_source=None):
    """Convert any subhkl HDF5 output to a DIALS ``.expt`` / ``.refl`` pair.

    Parameters
    ----------
    h5_path : str
        Path to a subhkl HDF5 output (finder, indexer, predictor, integrator,
        reduce/merge image stack, or zone-axis seed).
    expt_path, refl_path : str
        Destination paths for the ExperimentList and reflection_table.
    instrument : str, optional
        Instrument name. Falls back to the ``instrument`` attribute stored in the
        HDF5 file. Required to build detector geometry.
    image_source : str, optional
        Path to the image stack (a ``reduce``/``merge`` HDF5 with an ``images``
        dataset) whose pixels back this output. When available, an ``ImageSet`` is
        attached so the ``.expt`` opens in DIALS image tools. Defaults to
        ``h5_path`` itself when that is an image stack.
    """
    model, flex = _require_dials()

    with h5py.File(h5_path, "r") as f:
        if instrument is None and "instrument" in f.attrs:
            instrument = _decode(f.attrs["instrument"])

        peaks = _collect_flat_peaks(f) or _collect_predicted_peaks(f)

        lam = peaks["lambda"] if peaks else None
        beam = _build_beam(model, f, lam)
        crystal = _build_crystal(model, f)
        calibration = _read_detector_calibration(f)
        sample_offset = _read_sample_offset(f)

        # Resolve the pixel-data stack (if any) and its (frame, panel) layout.
        stack_path = None
        if image_source and _file_has_images(image_source):
            stack_path = image_source
        elif "images" in f:
            stack_path = h5_path

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
                    except Exception:
                        stack_rotations[fr] = np.eye(3)

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

        detector, bank_to_panel = _build_detector(
            model, instrument, bank_ids, calibration
        )

        if peaks:
            rotations, axes = _frame_rotations(f, peaks, frame_ids)
        elif stack_rotations is not None:
            rotations = {fr: stack_rotations.get(fr, np.eye(3)) for fr in frame_ids}
            axes = stack_axes
        else:
            rotations, axes = _frame_rotations(f, peaks, frame_ids)

        frame_to_expt = {frame: i for i, frame in enumerate(frame_ids)}

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
                identifier=f"subhkl-frame-{frame}",
            )
            experiments.append(expt)

        # Per-peak sample position, for the s1 calculation below.
        peak_s_lab = None
        if sample_offset is not None and peaks and peaks["run_index"] is not None:
            peak_s_lab = np.array(
                [frame_s_lab.get(int(r), np.zeros(3)) for r in peaks["run_index"]]
            )

        # Attach an ImageSet when the frames/panels line up with the stack layout.
        if (
            layout is not None
            and frame_ids == list(layout["frame_ids"])
            and bank_ids == list(layout["panel_bank_ids"])
        ):
            container_path = os.path.splitext(expt_path)[0] + "_images.h5"
            _attach_imageset(
                experiments, stack_path, layout, container_path, instrument
            )

    reflections = _build_reflection_table(
        flex, peaks, frame_to_expt, bank_to_panel, experiments, peak_s_lab
    )

    experiments.as_file(expt_path)
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
    except Exception as exc:
        print(f"Note: could not attach DIALS ImageSet ({exc}); wrote geometry only.")


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
            (int(round(h)), int(round(k)), int(round(l)))
            for h, k, l in zip(peaks["h"], peaks["k"], peaks["l"])
        ]
        table["miller_index"] = flex.miller_index(hkl)

    # Per-reflection wavelength (subhkl is polychromatic / Laue).
    if peaks["lambda"] is not None:
        table["wavelength"] = flex.double([float(x) for x in peaks["lambda"]])

    # Observed / calculated pixel positions: DIALS x=fast=column, y=slow=row.
    if peaks["pixel_r"] is not None and peaks["pixel_c"] is not None:
        xyz_px = [
            (float(c), float(r), 0.0)
            for r, c in zip(peaks["pixel_r"], peaks["pixel_c"])
        ]
        key = "xyzcal.px" if peaks.get("predicted") else "xyzobs.px.value"
        table[key] = flex.vec3_double(xyz_px)

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

    # Intensities.
    if peaks["intensity"] is not None:
        table["intensity.sum.value"] = flex.double(
            [float(x) for x in peaks["intensity"]]
        )
        if peaks["sigma"] is not None:
            variance = np.asarray(peaks["sigma"], dtype=float) ** 2
            table["intensity.sum.variance"] = flex.double([float(x) for x in variance])

    # Flags describing what was computed.
    if peaks.get("predicted"):
        table.set_flags(flex.bool(n, True), table.flags.predicted)
    if peaks["h"] is not None:
        table.set_flags(flex.bool(n, True), table.flags.indexed)
    if peaks["intensity"] is not None:
        table.set_flags(flex.bool(n, True), table.flags.integrated_sum)

    return table
