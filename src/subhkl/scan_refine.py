"""Validated scan inputs and native output for staged band refinement."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from subhkl.config import beamlines
from subhkl.core.spacegroup import get_space_group_object
from subhkl.instrument.detector import Detector, DetectorShape
from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix

MODES = frozenset(
    [
        "cylindrical",
        "axial_stretch",
        "area",
        "global_rot",
        "global_trans",
        "sample",
        "cell",
        "per_run",
        "panels",
    ]
)


def _frame_offsets(f, n_frames):
    if "file_offsets" not in f:
        return np.array([0], dtype=int)
    offsets = np.asarray(f["file_offsets"][()], int)
    if (
        offsets.ndim != 1
        or len(offsets) == 0
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or offsets[-1] >= n_frames
    ):
        raise ValueError("invalid frame file_offsets")
    return offsets


def _frame_rotations_and_scan_axes(f, n_frames):
    """Return nominal per-frame rotations, scan-axis index, and lab scan axes."""
    has_angles = "goniometer/angles" in f
    has_axes = "goniometer/axes" in f
    axes = np.asarray(f["goniometer/axes"][()], float) if has_axes else None
    angles = np.asarray(f["goniometer/angles"][()], float) if has_angles else None
    if angles is not None:
        if axes is None:
            raise ValueError("goniometer angles require axes")
        if (
            axes.ndim != 2
            or axes.shape[1] != 4
            or not np.isfinite(axes).all()
            or np.any(np.linalg.norm(axes[:, :3], axis=1) == 0)
        ):
            raise ValueError("invalid goniometer axes")
        if angles.ndim == 1 and len(angles) == len(axes):
            angles = np.repeat(angles[None, :], n_frames, axis=0)
        if angles.shape != (n_frames, len(axes)) or not np.isfinite(angles).all():
            raise ValueError("goniometer angles must be frame-addressed")
        ptp = np.ptp(angles, axis=0)
        scan_col = int(np.argmax(ptp)) if np.any(ptp > 0) else len(axes) - 1
        actual = np.stack([calc_goniometer_rotation_matrix(axes, a) for a in angles])
        u_k = axes[scan_col, :3] / np.linalg.norm(axes[scan_col, :3])
        signed_u = axes[scan_col, 3] * u_k
        if scan_col == 0:
            scan_axes_lab = np.repeat(signed_u[None, :], n_frames, axis=0)
        else:
            R_outer = np.stack(
                [
                    calc_goniometer_rotation_matrix(axes[:scan_col], a[:scan_col])
                    for a in angles
                ]
            )
            scan_axes_lab = R_outer @ signed_u
    else:
        scan_col = None
        actual = None
        scan_axes_lab = np.repeat(np.array([[0.0, -1.0, 0.0]]), n_frames, axis=0)

    if "goniometer/R" in f:
        R = np.asarray(f["goniometer/R"][()], float)
        if R.ndim == 2 and R.shape == (3, 3):
            R = np.repeat(R[None, :, :], n_frames, axis=0)
        if R.shape != (n_frames, 3, 3) or not np.isfinite(R).all():
            raise ValueError("goniometer R must be frame-addressed")
        if actual is not None and not np.allclose(R, actual, atol=1e-6):
            raise ValueError(
                "stored R disagrees with axes/angles; correct historical metadata"
            )
    elif actual is not None:
        R = actual
    else:
        R = np.repeat(np.eye(3)[None, :, :], n_frames, axis=0)

    if not np.allclose(R @ R.swapaxes(1, 2), np.eye(3), atol=1e-6) or not np.allclose(
        np.linalg.det(R), 1.0, atol=1e-6
    ):
        raise ValueError("goniometer R must contain proper rotations")
    return R, axes, angles, scan_col, scan_axes_lab


def run_scan_refinement(
    frames,
    output,
    *,
    bootstrap=None,
    finder=None,
    instrument="CG4D",
    cell=None,
    space_group=None,
    wavelength=(2.85, 4.35),
    assignment_wavelength=(2.8, 4.5),
    runs=None,
    stable_runs=None,
    modes=None,
    n_zones=400,
    d_min=2.0,
    min_deviance=100.0,
    pixel_polish=False,
    binning=16,
    maxiter=600,
    static_mask_file=None,
    search_samples=16384,
    search_candidates=6,
    search_seed=0,
):
    """Index from finder observations or count bins and refine with band-geometry modes.

    The bootstrap supplies orientation only. Detector geometry starts from
    instrument metrology, matching the staged reference protocol. Counts mode
    keeps signed residuals from every valid bin and cannot use pixel polish.
    """
    if search_samples < 1 or search_candidates < 1 or search_seed < 0:
        raise ValueError("invalid blind search budget or seed")
    if instrument not in beamlines:
        raise ValueError(f"unknown instrument: {instrument}")
    dets = {int(k): Detector(dict(cfg)) for k, cfg in beamlines[instrument].items()}
    if any(det.panel_type != DetectorShape.flat_panel for det in dets.values()):
        raise ValueError("staged band refinement requires flat-panel detectors")
    destinations = {
        Path(output).resolve(),
        Path(output).with_suffix(".stages.h5").resolve(),
    }
    if bootstrap is None:
        destinations.add(Path(output).with_suffix(".blind.h5").resolve())
    for source in (frames, bootstrap, finder, static_mask_file):
        if source and Path(source).resolve() in destinations:
            raise ValueError("output cannot overwrite an input")
    if pixel_polish and not finder:
        raise ValueError("pixel polish needs finder observations; omit it for counts")
    chosen = MODES if modes is None else frozenset(modes)
    if chosen - MODES:
        raise ValueError(f"unknown band refinement modes: {sorted(chosen - MODES)}")
    if min_deviance < 0 or maxiter < 1 or n_zones < 60 or d_min <= 0:
        raise ValueError(
            "invalid band refinement budget, zones, resolution or deviance"
        )
    for band in (wavelength, assignment_wavelength):
        if not 0 < band[0] < band[1]:
            raise ValueError("wavelength bounds must be positive and increasing")
    with h5py.File(frames) as f:
        if "goniometer/translations" in f and np.any(f["goniometer/translations"][()]):
            raise ValueError("band refinement starts from nominal sample translations")
        if "goniometer/per_run" in f and "goniometer/angles_nominal" not in f:
            raise ValueError(
                "fold existing per-run corrections into angles before band refinement"
            )
        if "goniometer/offsets" in f:
            stored = f["goniometer/offsets"]
            values = (
                [v[()] for v in stored.values()]
                if isinstance(stored, h5py.Group)
                else stored[()]
            )
            if np.any(values):
                raise ValueError(
                    "fold global goniometer offsets into angles before band refinement"
                )
        img_shape = f["images"].shape
        if len(img_shape) != 3:
            raise ValueError("images must be a 3D stack (n_frames, n_rows, n_cols)")
        n_frames, n_rows, n_cols = (
            int(img_shape[0]),
            int(img_shape[1]),
            int(img_shape[2]),
        )
        if binning < 1 or n_rows % binning or n_cols % binning:
            raise ValueError(
                f"band binning ({binning}) must divide the detector size ({n_rows}x{n_cols})"
            )
        offsets = _frame_offsets(f, n_frames)
        _frame_rotations_and_scan_axes(f, n_frames)
        if cell is None:
            cell = [
                float(f[f"sample/{k}"][()])
                for k in ("a", "b", "c", "alpha", "beta", "gamma")
            ]
        if space_group is None:
            space_group = f["sample/space_group"][()]
            if isinstance(space_group, bytes):
                space_group = space_group.decode()
    if (
        not np.isfinite(cell).all()
        or len(cell) != 6
        or min(cell[:3]) <= 0
        or min(cell[3:]) <= 0
        or max(cell[3:]) >= 180
    ):
        raise ValueError("invalid cell")
    get_space_group_object(space_group)
    selected = list(range(len(offsets))) if runs is None else list(runs)
    stable = selected if stable_runs is None else list(stable_runs)
    if not selected or not stable or not set(stable) <= set(selected):
        raise ValueError("stable runs must be a nonempty subset of selected runs")
    if min(selected) < 0 or max(selected) >= len(offsets):
        raise ValueError("run index is outside the frame file_offsets")
    args = SimpleNamespace(
        nexus=str(frames),
        output=str(output),
        bootstrap=str(bootstrap) if bootstrap is not None else None,
        finder=finder,
        instrument=instrument,
        cell=cell,
        space_group=space_group,
        runs=selected,
        stable_runs=stable,
        anchor_run=min(stable),
        modes=chosen,
        n_runs=len(offsets),
        wl_min=wavelength[0],
        wl_max=wavelength[1],
        assign_wavelength=assignment_wavelength,
        n_zones=n_zones,
        d_min=d_min,
        min_deviance=min_deviance,
        pixel_polish=pixel_polish,
        binning=binning,
        maxiter=maxiter,
        static_mask_file=static_mask_file,
    )
    from subhkl.search.scan_band import refine_scan
    import jax

    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        if bootstrap is None:
            from subhkl.search.blind_band import discover

            args.bootstrap = discover(
                args, search_samples, search_candidates, search_seed
            )
            args.blind_search = args.bootstrap
        return refine_scan(args)


def load_observations(args, *, return_kinematics=False):
    with h5py.File(args.nexus) as f:
        img_shape = f["images"].shape
        n_frames, n_rows, n_cols = (
            int(img_shape[0]),
            int(img_shape[1]),
            int(img_shape[2]),
        )
        offsets = _frame_offsets(f, n_frames)
        R_frames, _, angles, _, scan_axes_lab = _frame_rotations_and_scan_axes(
            f, n_frames
        )
        if angles is None:
            angles = np.zeros((n_frames, 1), dtype=float)
        frame_banks = np.asarray(f["bank_ids"], int)
        run_ids = np.searchsorted(offsets, np.arange(n_frames), side="right") - 1
        if args.finder:
            with h5py.File(args.finder) as peaks:
                images = np.asarray(peaks["peaks/image_index"], int)
                banks = np.asarray(peaks["bank"], int)
                if np.any(images < 0) or np.any(images >= n_frames):
                    raise ValueError(
                        "finder image_index must address the source frames"
                    )
                if not np.array_equal(frame_banks[images], banks):
                    raise ValueError("finder banks do not match source image_index")
                dev = np.asarray(peaks["peaks/deviance"], float)
                scale_px = 0.5 * ((n_rows - 1) + (n_cols - 1))
                result = (
                    banks,
                    peaks["peaks/pixel_r"][()],
                    peaks["peaks/pixel_c"][()],
                    dev,
                    peaks["peaks/radius"][()] * scale_px,
                    run_ids[images],
                    angles[images],
                    np.sqrt(np.maximum(dev, 0)),
                )
                kinematics = (R_frames[images], scan_axes_lab[images])
        else:
            # No connected components, local maxima or signal qualification.
            # All instrument-valid bins contribute, including negative excess.
            b = int(args.binning)
            if n_rows % b or n_cols % b:
                raise ValueError(
                    f"band binning ({b}) must divide the detector size ({n_rows}x{n_cols})"
                )
            centers_r = np.arange(n_rows // b) * b + (b - 1) / 2
            centers_c = np.arange(n_cols // b) * b + (b - 1) / 2
            rr, cc = np.meshgrid(centers_r, centers_c, indexing="ij")
            collected = [[] for _ in range(8)]
            collected_kin = [[], []]
            mask_file = (
                h5py.File(args.static_mask_file) if args.static_mask_file else None
            )
            try:
                for i in range(n_frames):
                    if run_ids[i] not in args.runs:
                        continue
                    raw = np.asarray(f["images"][i], float)
                    valid = np.isfinite(raw) & (raw >= 0)
                    if mask_file is not None:
                        key = "masks" if "masks" in mask_file else "images"
                        ids = mask_file["bank_ids"][()]
                        matches = np.flatnonzero(ids == frame_banks[i])
                        if len(matches) != 1:
                            raise ValueError("mask must have one entry per bank")
                        valid &= np.asarray(mask_file[key][matches[0]], bool)
                    y = (
                        np.where(valid, raw, 0)
                        .reshape(n_rows // b, b, n_cols // b, b)
                        .sum((1, 3))
                    )
                    good = valid.reshape(n_rows // b, b, n_cols // b, b).all((1, 3))
                    bg = median_filter(y, size=7, mode="nearest")
                    weight = y - bg
                    bg_5 = median_filter(y, size=5, mode="nearest")
                    bg_safe = np.maximum(bg_5, 1.0)
                    y_safe = np.maximum(y, bg_safe)
                    dev_bin = np.where(
                        y > bg_safe,
                        2.0 * (y_safe * np.log(y_safe / bg_safe) - (y_safe - bg_safe)),
                        0.0,
                    )
                    n = int(good.sum())
                    values = (
                        np.full(n, frame_banks[i]),
                        rr[good],
                        cc[good],
                        dev_bin[good],
                        np.zeros(n),
                        np.full(n, run_ids[i]),
                        np.repeat(angles[i : i + 1], n, axis=0),
                        weight[good],
                    )
                    for dest, value in zip(collected, values):
                        dest.append(value)
                    collected_kin[0].append(np.repeat(R_frames[i : i + 1], n, axis=0))
                    collected_kin[1].append(
                        np.repeat(scan_axes_lab[i : i + 1], n, axis=0)
                    )
                result = tuple(np.concatenate(v) for v in collected)
                kinematics = tuple(np.concatenate(v) for v in collected_kin)
            finally:
                if mask_file is not None:
                    mask_file.close()
    selected = np.isin(result[5], args.runs)
    if args.finder:
        selected &= result[3] >= args.min_deviance
    if not np.any(selected) or not np.abs(result[7][selected]).sum() > 0:
        raise ValueError("no weighted observations in selected runs")
    for r in args.stable_runs:
        if not np.any(selected & (result[5] == r)):
            raise ValueError(f"stable run {r} has no observations")
    print(
        f"Band fit: {int(selected.sum())} observations; "
        f"{'finder peaks' if args.finder else 'all signed count residuals'}",
        flush=True,
    )
    if return_kinematics:
        return result + kinematics
    return result


def checkpoint(args, name, parameters, diagnostics):
    path = Path(args.output).with_suffix(".stages.h5")
    with h5py.File(path, "a") as f:
        if name in f:
            del f[name]
        group = f.create_group(name)
        group["parameters"] = parameters
        group.attrs["diagnostics"] = json.dumps(diagnostics[-1])


def write_result(
    args,
    U,
    B,
    cell,
    banks,
    centers,
    uhats,
    vhats,
    widths,
    heights,
    delta,
    sample,
    parameters,
    diagnostics,
):
    temp = str(args.output) + ".tmp"
    status = (
        "converged"
        if diagnostics[-1]["success"] and not len(args.bound_parameters)
        else "not_converged"
    )
    try:
        with h5py.File(args.nexus) as source, h5py.File(temp, "w") as out:
            for key in (
                "instrument",
                "sample",
                "goniometer",
                "files",
                "file_offsets",
                "bank_ids",
            ):
                if key in source:
                    source.copy(key, out)
            out.attrs.update(source.attrs)
            out.attrs["instrument"] = args.instrument
            out.attrs["source"] = str(Path(args.nexus).resolve())

            def put(key, value):
                if key in out:
                    del out[key]
                out[key] = value

            for key, value in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
                put(f"sample/{key}", value)
            put("sample/space_group", args.space_group)
            if status == "converged":
                put("sample/U", U)
            elif "sample/U" in out:
                del out["sample/U"]
            put("sample/B", B)
            put("instrument/wavelength", args.assign_wavelength)
            n_frames = int(source["images"].shape[0])
            offsets = _frame_offsets(source, n_frames)
            if "file_offsets" not in out:
                put("file_offsets", offsets)
            R_nom, axes, angles, col, scan_axes_lab = _frame_rotations_and_scan_axes(
                source, n_frames
            )
            run_ids = np.searchsorted(offsets, np.arange(n_frames), side="right") - 1
            if "goniometer/per_run" in out:
                del out["goniometer/per_run"]
            if angles is not None and axes is not None and col is not None:
                corrected = angles.copy()
                corrected[:, col] += delta[run_ids]
                put("goniometer/angles_nominal", angles)
                put("goniometer/angles", corrected)
                put(
                    "goniometer/R",
                    np.stack(
                        [calc_goniometer_rotation_matrix(axes, a) for a in corrected]
                    ),
                )
                shifts = np.zeros((len(axes), 3))
                shifts[-1] = sample
                put("goniometer/translations", shifts)
                changes = np.zeros((args.n_runs, len(axes)))
                changes[:, col] = delta
                put("goniometer/per_run/delta_deg_all_axes", changes)
                put("goniometer/per_run/frame_to_run", run_ids)
            else:
                d_rot = Rotation.from_rotvec(
                    np.deg2rad(delta[run_ids])[:, None] * scan_axes_lab
                ).as_matrix()
                put("goniometer/R", np.einsum("nij,njk->nik", d_rot, R_nom))
                put("goniometer/translations", np.asarray(sample, float))
                put("goniometer/per_run/frame_to_run", run_ids)
            for i, bank in enumerate(banks):
                for key, value in (
                    ("center", centers[i]),
                    ("uhat", uhats[i]),
                    ("vhat", vhats[i]),
                    ("width", widths[i]),
                    ("height", heights[i]),
                ):
                    put(f"detector_calibration/bank_{bank}/{key}", value)
            report = out.create_group("scan_refinement")
            report["parameters"] = parameters
            report["parameter_bounds"] = args.parameter_bounds
            report["U"] = U
            report["B"] = B
            report["bound_parameters"] = args.bound_parameters
            report["runs"] = args.runs
            report["stable_runs"] = args.stable_runs
            report.attrs["modes"] = ",".join(sorted(args.modes))
            report.attrs["source"] = str(Path(args.nexus).resolve())
            report.attrs["finder"] = str(args.finder or "")
            report.attrs["bootstrap"] = str(Path(args.bootstrap).resolve())
            report.attrs["bootstrap_geometry"] = (
                "orientation only; nominal instrument geometry"
            )
            report.attrs["pixel_polish"] = args.pixel_polish
            report.attrs["stages"] = json.dumps(diagnostics)
            report.attrs["status"] = status
            report.attrs["final_optimizer_success"] = diagnostics[-1]["success"]
            report["band_wavelength"] = [args.wl_min, args.wl_max]
            report.attrs["binning"] = args.binning
            report.attrs["n_zones"] = args.n_zones
            report.attrs["d_min"] = args.d_min
            report.attrs["min_deviance"] = args.min_deviance if args.finder else 0.0
            report.attrs["static_mask"] = str(args.static_mask_file or "")
            if hasattr(args, "pixel_diagnostics"):
                for name, values in args.pixel_diagnostics.items():
                    report[f"pixel_polish/{name}"] = values
            # Shared visualization contract; this is a band objective, not a
            # Poisson fit, so no Poisson deviance or stationarity is invented.
            view = out.create_group("solve")
            view.attrs["status"] = status
            view.attrs["objective"] = "band"
            view["orientations_sample"] = U[None]
            view["active"] = np.array([0], dtype=int)
            if hasattr(args, "blind_search"):
                with h5py.File(args.blind_search) as search:
                    search.copy("search", out, name="blind_search")
        os.replace(temp, args.output)
        print(
            f"Band refinement: {status}; {len(args.bound_parameters)} parameters at bounds",
            flush=True,
        )
        return status
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
