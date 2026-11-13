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
        "per_run_trans",
        "panels",
    ]
)

DEFAULT_MODES = MODES - {"panels"}


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


def _parse_rot_axis(axis):
    if axis is None:
        return None
    if isinstance(axis, str):
        s = axis.strip()
        if s.lower() in ("", "none"):
            return None
        try:
            vals = [float(v.strip()) for v in s.split(",")]
        except ValueError as exc:
            raise ValueError(
                "global_rot_axis must be three comma-separated floats or 'none'"
            ) from exc
    else:
        vals = [float(v) for v in axis]
    arr = np.asarray(vals, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError("global_rot_axis must be a finite 3-vector or None")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("global_rot_axis must be nonzero")
    unit = arr / norm
    dom = int(np.argmax(np.abs(unit)))
    if unit[dom] < 0:
        unit = -unit
    return tuple(float(v) for v in unit)


def run_scan_refinement(
    frames,
    output,
    *,
    bootstrap=None,
    finder=None,
    instrument="CG4D",
    cell=None,
    space_group=None,
    wavelength=None,
    assignment_wavelength=None,
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
    global_rot_axis=(0.0, 1.0, 0.0),
    border_width=4,
):
    """Index from finder observations or count bins and refine with band-geometry modes.

    The bootstrap supplies orientation only. Detector geometry starts from
    instrument metrology, matching the staged reference protocol. Counts mode
    keeps signed residuals from every valid bin and cannot use pixel polish.
    """
    if search_samples < 1 or search_candidates < 1 or search_seed < 0:
        raise ValueError("invalid blind search budget or seed")
    with h5py.File(frames) as f:
        if instrument is None:
            inst_attr = f.attrs.get("instrument")
            if isinstance(inst_attr, bytes):
                inst_attr = inst_attr.decode()
            instrument = str(inst_attr) if inst_attr else "CG4D"
        file_wl = (
            tuple(float(x) for x in f["instrument/wavelength"][()][:2])
            if "instrument/wavelength" in f
            else None
        )
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
    chosen = DEFAULT_MODES if modes is None else frozenset(modes)
    if chosen - MODES:
        raise ValueError(f"unknown band refinement modes: {sorted(chosen - MODES)}")
    if (
        min_deviance < 0
        or maxiter < 1
        or n_zones < 60
        or d_min <= 0
        or border_width < 0
    ):
        raise ValueError(
            "invalid band refinement budget, zones, resolution, deviance or border_width"
        )
    wl_low, wl_high = (None, None) if wavelength is None else wavelength
    wl_low_explicit = wl_low is not None
    wl_high_explicit = wl_high is not None
    if wl_low is None:
        wl_low = file_wl[0] if file_wl is not None else 2.85
    if wl_high is None:
        wl_high = file_wl[1] if file_wl is not None else 4.35
    wavelength = (float(wl_low), float(wl_high))

    awl_low, awl_high = (
        (None, None) if assignment_wavelength is None else assignment_wavelength
    )
    if awl_low is None:
        if wl_low_explicit:
            awl_low = wavelength[0]
        elif file_wl is not None:
            awl_low = file_wl[0]
        else:
            awl_low = 2.8 if wavelength == (2.85, 4.35) else wavelength[0]
    if awl_high is None:
        if wl_high_explicit:
            awl_high = wavelength[1]
        elif file_wl is not None:
            awl_high = file_wl[1]
        else:
            awl_high = 4.5 if wavelength == (2.85, 4.35) else wavelength[1]
    assignment_wavelength = (float(awl_low), float(awl_high))
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
    rot_axis = _parse_rot_axis(global_rot_axis)
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
        global_rot_axis=rot_axis,
        border_width=int(border_width),
    )
    import jax

    from subhkl.search.scan_band import refine_scan

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
            rr_0, cc_0 = np.meshgrid(centers_r, centers_c, indexing="ij")
            rows_full, cols_full = np.meshgrid(
                np.arange(n_rows, dtype=float),
                np.arange(n_cols, dtype=float),
                indexing="ij",
            )
            rows_bin = rows_full.reshape(n_rows // b, b, n_cols // b, b)
            cols_bin = cols_full.reshape(n_rows // b, b, n_cols // b, b)
            collected = [[] for _ in range(8)]
            collected_kin = [[], []]
            mask_file = (
                h5py.File(args.static_mask_file) if args.static_mask_file else None
            )
            bw = int(getattr(args, "border_width", 4))
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
                    if bw > 0 and n_rows > 4 * max(b, bw) and n_cols > 4 * max(b, bw):
                        raw = np.pad(raw[bw:-bw, bw:-bw], bw, mode="edge")
                        if bw >= b:
                            valid[: (bw // b) * b, :] = False
                            valid[-((bw // b) * b) :, :] = False
                            valid[:, : (bw // b) * b] = False
                            valid[:, -((bw // b) * b) :] = False
                    raw_blocks = np.where(valid, raw, 0).reshape(
                        n_rows // b, b, n_cols // b, b
                    )
                    y = raw_blocks.sum((1, 3))
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
                    excess_sq = (
                        np.maximum(raw_blocks - (bg_5 / (b * b))[:, None, :, None], 0.0)
                        ** 2
                    )
                    excess_sum = excess_sq.sum((1, 3))
                    rr = np.where(
                        excess_sum > 0,
                        (excess_sq * rows_bin).sum((1, 3))
                        / np.maximum(excess_sum, 1e-12),
                        rr_0,
                    )
                    cc = np.where(
                        excess_sum > 0,
                        (excess_sq * cols_bin).sum((1, 3))
                        / np.maximum(excess_sum, 1e-12),
                        cc_0,
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


def checkpoint(args, name, parameters, diagnostics, snapshot=None):
    import shutil

    path = Path(args.output).with_suffix(".stages.h5")
    tmp_path = path.with_suffix(".stages.tmp.h5")
    try:
        if path.is_file():
            shutil.copy2(path, tmp_path)
        with h5py.File(tmp_path, "a") as f:
            if name in f:
                del f[name]
            group = f.create_group(name)
            group["parameters"] = parameters
            group.attrs["diagnostics"] = json.dumps(diagnostics[-1])

            if snapshot is not None:

                def _put(key, val):
                    if key in f:
                        del f[key]
                    f[key] = val

                f.attrs["instrument"] = str(args.instrument)
                _put("instrument/name", str(args.instrument))
                _put(
                    "instrument/wavelength",
                    np.asarray(args.assign_wavelength, dtype=float),
                )
                cell_vals = [float(v) for v in snapshot["cell"][:6]]
                _put("sample/cell", np.asarray(cell_vals, dtype=float))
                for k_c, v_c in zip(
                    ("a", "b", "c", "alpha", "beta", "gamma"), cell_vals
                ):
                    _put(f"sample/{k_c}", float(v_c))
                _put("sample/space_group", str(args.space_group))
                _put("sample/U", np.asarray(snapshot["U"], dtype=float))
                _put("sample/B", np.asarray(snapshot["B"], dtype=float))
                if snapshot.get("R") is not None:
                    _put("goniometer/R", np.asarray(snapshot["R"], dtype=float))
                for i_b, b_id in enumerate(snapshot.get("banks", [])):
                    for k_d, v_d in (
                        ("center", snapshot["centers"][i_b]),
                        ("uhat", snapshot["uhats"][i_b]),
                        ("vhat", snapshot["vhats"][i_b]),
                        ("width", float(snapshot["widths"][i_b])),
                        ("height", float(snapshot["heights"][i_b])),
                    ):
                        _put(f"detector_calibration/bank_{b_id}/{k_d}", v_d)
                if "scan_refinement" in f:
                    del f["scan_refinement"]
                rep = f.create_group("scan_refinement")
                rep["parameters"] = np.asarray(parameters, dtype=float)
                rep["U"] = np.asarray(snapshot["U"], dtype=float)
                rep["B"] = np.asarray(snapshot["B"], dtype=float)
                rep["band_wavelength"] = np.asarray(
                    [args.wl_min, args.wl_max], dtype=float
                )
                rep.attrs["stages"] = json.dumps(diagnostics)
                rep.attrs["status"] = "running"
                rep.attrs["n_zones"] = int(
                    snapshot.get("n_zones", getattr(args, "n_zones", 120))
                )
                rep.attrs["d_min"] = float(args.d_min)
                if (
                    hasattr(args, "blind_search")
                    and "blind_search" not in f
                    and os.path.isfile(str(args.blind_search))
                ):
                    import contextlib

                    with (
                        contextlib.suppress(
                            OSError, RuntimeError, KeyError, ValueError
                        ),
                        h5py.File(args.blind_search, "r") as sf,
                    ):
                        if "search" in sf:
                            sf.copy("search", f, name="blind_search")
        os.replace(tmp_path, path)
        if snapshot is not None and (
            os.environ.get("SUBHKL_LIVE_STDIO") == "1"
            or "PYTEST_CURRENT_TEST" not in os.environ
        ):
            import contextlib

            with contextlib.suppress(
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
            ):
                import base64
                import zlib

                live_banks = {}
                for i_b, b_id in enumerate(snapshot.get("banks", [])):
                    live_banks[str(int(b_id))] = {
                        "center": [
                            round(float(x), 6)
                            for x in np.asarray(snapshot["centers"][i_b]).ravel()[:3]
                        ],
                        "uhat": [
                            round(float(x), 6)
                            for x in np.asarray(snapshot["uhats"][i_b]).ravel()[:3]
                        ],
                        "vhat": [
                            round(float(x), 6)
                            for x in np.asarray(snapshot["vhats"][i_b]).ravel()[:3]
                        ],
                        "width": round(float(snapshot["widths"][i_b]), 6),
                        "height": round(float(snapshot["heights"][i_b]), 6),
                    }
                live_doc = {
                    "kind": "stages",
                    "instrument": str(args.instrument),
                    "space_group": str(args.space_group),
                    "wavelength": [
                        float(args.assign_wavelength[0]),
                        float(args.assign_wavelength[1]),
                    ],
                    "band_wavelength": [float(args.wl_min), float(args.wl_max)],
                    "d_min": float(args.d_min),
                    "stage": str(name),
                    "n_zones": int(
                        snapshot.get("n_zones", getattr(args, "n_zones", 120))
                    ),
                    "cell": [round(float(v), 5) for v in snapshot["cell"][:6]],
                    "U": np.round(np.asarray(snapshot["U"], dtype=float), 6).tolist(),
                    "B": np.round(np.asarray(snapshot["B"], dtype=float), 6).tolist(),
                    "R": (
                        np.round(np.asarray(snapshot["R"], dtype=float), 6).tolist()
                        if snapshot.get("R") is not None
                        else None
                    ),
                    "stages": diagnostics,
                    "banks": live_banks,
                }
                b64 = base64.b64encode(
                    zlib.compress(
                        json.dumps(live_doc, separators=(",", ":")).encode("utf-8"),
                        level=9,
                    )
                ).decode("ascii")
                print(f"[subhkl-live-state] {b64}", flush=True)
    finally:
        if tmp_path.exists():
            import contextlib

            with contextlib.suppress(OSError):
                tmp_path.unlink()


def _decompose_R_to_angles(axes, ang_nom, R_target):
    """Decompose a 3D rotation matrix R_target into motor angles on `axes` near `ang_nom`."""
    from scipy.optimize import least_squares

    K = len(axes)
    if K == 3:
        free_cols = [0, 1, 2]
        inits = [np.zeros(3)]
    elif K == 5:
        free_cols = [0, 2, 4]
        inits = [
            np.array([0.0, 2.0, 0.0]),
            np.array([0.0, -2.0, 0.0]),
            np.zeros(3),
        ]
    else:
        free_cols = list(range(min(K, 3)))
        inits = [np.zeros(len(free_cols))]
    best_a, best_err = ang_nom.copy(), 1e9
    for x0 in inits:

        def fn(x):
            a = ang_nom.copy()
            for idx, c in enumerate(free_cols):
                a[c] += x[idx]
            return Rotation.from_matrix(
                calc_goniometer_rotation_matrix(axes, a) @ R_target.T
            ).as_rotvec()

        res = least_squares(fn, x0, ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=200)
        a = ang_nom.copy()
        for idx, c in enumerate(free_cols):
            a[c] += res.x[idx]
        err = float(np.max(np.abs(calc_goniometer_rotation_matrix(axes, a) - R_target)))
        if err < best_err:
            best_err, best_a = err, a
        if best_err < 1e-13:
            break
    return best_a


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
            R_runs_ref = getattr(args, "R_runs_refined", None)
            if angles is not None and axes is not None and col is not None:
                changes = np.zeros((args.n_runs, len(axes)))
                if R_runs_ref is not None and len(axes) >= 3:
                    for r in range(args.n_runs):
                        ang_nom_r = angles[offsets[r]]
                        ang_corr_r = _decompose_R_to_angles(
                            axes, ang_nom_r, R_runs_ref[r]
                        )
                        changes[r] = ang_corr_r - ang_nom_r
                    corrected = angles + changes[run_ids]
                else:
                    changes[:, col] = delta
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
                put("goniometer/per_run/delta_deg", np.asarray(delta, float))
                put("goniometer/per_run/delta_deg_all_axes", changes)
                put("goniometer/per_run/frame_to_run", run_ids)
            else:
                if R_runs_ref is not None:
                    put("goniometer/R", np.asarray(R_runs_ref, float)[run_ids])
                else:
                    d_rot = Rotation.from_rotvec(
                        np.deg2rad(delta[run_ids])[:, None] * scan_axes_lab
                    ).as_matrix()
                    put("goniometer/R", np.einsum("nij,njk->nik", d_rot, R_nom))
                put("goniometer/translations", np.asarray(sample, float))
                put("goniometer/per_run/delta_deg", np.asarray(delta, float))
                put("goniometer/per_run/frame_to_run", run_ids)
            trans_runs_ref = getattr(args, "trans_runs_refined", None)
            if trans_runs_ref is not None:
                put("goniometer/per_run/trans_m", np.asarray(trans_runs_ref, float))
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
            rot_ax = getattr(args, "global_rot_axis", (0.0, 1.0, 0.0))
            report.attrs["global_rot_axis"] = (
                "none" if rot_ax is None else ",".join(f"{v:g}" for v in rot_ax)
            )
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
