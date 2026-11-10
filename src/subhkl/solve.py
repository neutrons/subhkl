"""Shared threshold-free count workflow for orientation and detector geometry.

The workflow accepts stills and frame-addressed rotation scans.
Global lattice proposals seed a Poisson group fit; geometry and orientations
then refine together. Numerical engines from the retired commands remain
available as Python comparison APIs, but are not invoked by this workflow.
"""

import os
import tempfile
from pathlib import Path

import gemmi
import h5py
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from subhkl.calibrate import geometry_detectors, write_detector_calibration
from subhkl.instrument.detector import Detector
from subhkl.search.poisson_orientation import (
    CountData,
    OrientationModel,
    fit_intensities,
    refine,
)
from subhkl.search.profiles import RadialProfile

CELL_KEYS = ("a", "b", "c", "alpha", "beta", "gamma")


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _rotations(value):
    value = np.asarray(value, float).reshape(-1, 3, 3)
    if (
        len(value) == 0
        or not np.isfinite(value).all()
        or not np.allclose(value.swapaxes(1, 2) @ value, np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(value), 1.0, atol=1e-5)
    ):
        raise ValueError("orientations must be proper orthogonal 3x3 matrices")
    return value


def _proposal_data(model, proposal_bin, g=None):
    """Use every valid signed residual in spatial bins; no detection cutoff."""
    from subhkl.search.spherical import panel_directions

    data = model.data
    factor = proposal_bin // data.bin_px
    if factor < 1 or proposal_bin % data.bin_px:
        raise ValueError("proposal binning must be a multiple of count binning")
    dets = model.projection_detectors(np.zeros(6) if g is None else g)
    dirs, residuals = [], []
    for bank in model.banks:
        index = data.pixel_indices[bank]
        h, w = index.shape
        if h % factor or w % factor:
            raise ValueError("proposal binning must divide panel dimensions")
        valid = index >= 0
        residual = np.zeros(index.shape)
        residual[valid] = (data.counts - data.background)[index[valid]]
        shape = (h // factor, factor, w // factor, factor)
        # Partly masked bins retain all valid counts and use their mean pixel
        # position; masks are not counted as zeros or as background exposure.
        count = valid.reshape(shape).sum(axis=(1, 3))
        rr, cc = np.indices(index.shape)
        row = ((rr + 0.5) * data.bin_px - 0.5) * valid
        col = ((cc + 0.5) * data.bin_px - 0.5) * valid
        live = count > 0
        row = row.reshape(shape).sum(axis=(1, 3))[live] / count[live]
        col = col.reshape(shape).sum(axis=(1, 3))[live] / count[live]
        dirs.append(panel_directions(dets[bank], row, col))
        residuals.append(residual.reshape(shape).sum(axis=(1, 3))[live])
    return np.concatenate(dirs), np.concatenate(residuals)


def propose(model, cell, n_candidates=4, proposal_bin=32, g=None):
    """Joint signed count proposals in the sample frame for multiple settings."""
    from subhkl.search.spherical import lattice_ladder
    from subhkl.search.multi_setting import MultiSettingModel

    if isinstance(model, MultiSettingModel):
        directions, residuals, angles = [], [], []
        for child, rotation in zip(model.models, model.rotations):
            D, w = _proposal_data(child, proposal_bin, g)
            directions.append(D @ rotation)
            residuals.append(w)
            angles.append(-D[:, 2])
        D, weights = np.concatenate(directions), np.concatenate(residuals)
        sin_theta = np.concatenate(angles)
    else:
        D, weights = _proposal_data(model, proposal_bin, g)
        sin_theta = -D[:, 2]
    if not np.any(weights):
        raise ValueError(
            "no residual evidence for orientation proposals; supply a bootstrap to test the null"
        )
    # Absolute normalization preserves signs even at negative total excess.
    weights = weights / max(np.abs(weights).sum(), 1e-12)
    pool = max(20, 4 * n_candidates)
    candidates = lattice_ladder(
        D,
        weights,
        model.B,
        grid_deg=4.0,
        n_shortlist=max(300, pool),
        rungs=(
            (13, 0.5, 2.5, max(100, pool), 5),
            (30, 0.5, 1.0, max(60, pool), 5),
            (60, 0.3, 0.5, pool, 5),
            (145, 0.3, 0.25, pool, 5),
        ),
        sin_theta=sin_theta,
        wavelength=model.wavelength,
        d_min=model.d_min,
        cands=None,
    )
    # Quotient by metric symmetries only when they also preserve the allowed
    # reflection set, so systematic absences are not erased.
    tree = cKDTree(model.G)
    sample = model.G[:: max(1, len(model.G) // 300)]
    symmetries = [np.eye(3)]
    for op in gemmi.find_lattice_symmetry(gemmi.UnitCell(*cell), "P", 0.01):
        S = (
            model.B
            @ np.linalg.inv(np.asarray(op.rot) / op.DEN).T
            @ np.linalg.inv(model.B)
        )
        if np.linalg.det(S) < 0:
            S = -S
        if (
            np.allclose(S.T @ S, np.eye(3), atol=1e-6)
            and np.max(tree.query(sample @ S.T)[0]) < 1e-6
        ):
            symmetries.append(S)
    selected, scores = [], []
    for U, score, _ in candidates:
        if any(
            min(Rotation.from_matrix(U @ (V @ S).T).magnitude() for S in symmetries)
            < np.radians(0.5)
            for V in selected
        ):
            continue
        selected.append(U)
        scores.append(score)
        if len(selected) == n_candidates:
            break
    return np.asarray(selected), np.asarray(scores)


def run_solve(
    frames_filename,
    output_filename,
    *,
    metadata=None,
    instrument=None,
    cell=None,
    space_group=None,
    wavelength_min=None,
    wavelength_max=None,
    d_min=3.2,
    binning=8,
    proposal_binning=32,
    n_candidates=4,
    penalty=1.2,
    sigma_px=6.0,
    max_evals=1500,
    bootstrap=None,
    profile_file=None,
    background_file=None,
    static_mask_file=None,
    fixed_geometry=False,
    do_refine=True,
    verbose=True,
):
    """Solve stills or a rotation scan and atomically write frame-addressed results."""
    inputs = [
        frames_filename,
        metadata,
        bootstrap,
        profile_file,
        background_file,
        static_mask_file,
    ]
    if any(p and Path(p).resolve() == Path(output_filename).resolve() for p in inputs):
        raise ValueError("output must not overwrite an input file")
    if (
        not np.isfinite([d_min, penalty, sigma_px]).all()
        or d_min <= 0
        or penalty < 0
        or sigma_px <= 0
        or max_evals < 1
        or n_candidates < 1
        or binning < 1
    ):
        raise ValueError(
            "positive resolution, width, budgets and binning; nonnegative penalty required"
        )
    log = print if verbose else lambda *args: None
    with h5py.File(frames_filename, "r") as f:
        image_shape, banks = f["images"].shape, f["bank_ids"][()].astype(int)
    if len(image_shape) != 3 or len(banks) != image_shape[0] or len(banks) == 0:
        raise ValueError("solve requires a nonempty image stack with matching bank_ids")
    source = metadata or frames_filename
    with h5py.File(source, "r") as f:
        instrument = instrument or f.attrs.get("instrument")
        cell = (
            cell
            if cell is not None
            else (
                [float(f[f"sample/{k}"][()]) for k in CELL_KEYS]
                if "sample/a" in f
                else None
            )
        )
        space_group = space_group or (
            f["sample/space_group"][()] if "sample/space_group" in f else None
        )
        band = (
            f["instrument/wavelength"][()]
            if "instrument/wavelength" in f
            else (None, None)
        )
        from subhkl.solve_frames import frame_kinematics, setting_groups

        with h5py.File(frames_filename, "r") as frames:
            file_offsets = (
                frames["file_offsets"][()] if "file_offsets" in frames else None
            )
        if file_offsets is None and "file_offsets" in f:
            file_offsets = f["file_offsets"][()]
        R, origins, run_ids, corrected_angles, angles_changed = frame_kinematics(
            f, len(banks), file_offsets
        )
        R0 = R[0]
        groups = setting_groups(banks, R, origins, run_ids)
        multi = len(groups) > 1
        if "goniometer/R" not in f and "goniometer/angles" not in f:
            log(
                "No goniometer metadata supplied: defining the sample frame as the lab frame."
            )
        if "beam/ki_vec" in f and not np.allclose(f["beam/ki_vec"][()], [0, 0, 1]):
            raise ValueError("draft solve currently requires incident beam +z")
    if instrument is None or cell is None or space_group is None:
        raise ValueError(
            "provide instrument, six cell values, and space group via metadata or options"
        )
    instrument, space_group = _text(instrument), _text(space_group)
    cell = np.asarray(cell, float)
    if (
        cell.shape != (6,)
        or not np.isfinite(cell).all()
        or np.any(cell[:3] <= 0)
        or np.any((cell[3:] <= 0) | (cell[3:] >= 180))
    ):
        raise ValueError("invalid unit cell")
    band = (
        wavelength_min if wavelength_min is not None else band[0],
        wavelength_max if wavelength_max is not None else band[1],
    )
    if (
        any(v is None for v in band)
        or not np.isfinite(band).all()
        or not 0 < band[0] < band[1]
    ):
        raise ValueError("provide an ordered positive wavelength band")
    dets = geometry_detectors(instrument, np.zeros(7), list(dict.fromkeys(banks)))
    if set(dets) != set(banks):
        raise ValueError("input contains banks absent from instrument geometry")
    for bank in set(banks):
        d = dets[bank]
        if d.config["panel"] != "flat" or image_shape[1:] != (d.n, d.m):
            raise ValueError(
                "draft solve requires full flat-panel images matching instrument geometry"
            )
    # Load fitted absolute panel geometry without mutating global beamlines.
    for filename in dict.fromkeys([source, bootstrap]):
        if not filename:
            continue
        with h5py.File(filename, "r") as f:
            if "detector_calibration" in f:
                for bank, det in list(dets.items()):
                    name = f"detector_calibration/bank_{bank}"
                    if name in f:
                        cfg = dict(det.config)
                        for key in ("center", "uhat", "vhat", "width", "height"):
                            cfg[key] = f[f"{name}/{key}"][()]
                        dets[bank] = Detector(cfg)
    masks = None
    if static_mask_file:
        from subhkl.search.static_mask import load_mask_for_banks

        masks = (
            load_mask_for_banks(static_mask_file, list(banks), image_shape[1:]) > 0.5
        )
    bg = None
    if background_file:
        with h5py.File(background_file, "r") as f:
            ref_banks = f["bank_ids"][()].astype(int)
            if np.array_equal(ref_banks, banks):
                bg = np.arange(len(banks))
            elif len(set(ref_banks)) == len(ref_banks) and set(ref_banks) == set(banks):
                order = {bank: i for i, bank in enumerate(ref_banks)}
                bg = np.asarray([order[bank] for bank in banks])
            else:
                raise ValueError(
                    "background bank ids must match counts or provide one image per bank"
                )
    profile = None
    if profile_file:
        with h5py.File(profile_file, "r") as f:
            profile = RadialProfile(f["profile/u"][()], f["profile/f"][()])
    children = []
    for frames in groups:
        with h5py.File(frames_filename, "r") as raw:
            setting_images = raw["images"][frames]
        setting_background = np.ones(len(frames))
        if bg is not None:
            with h5py.File(background_file, "r") as raw_bg:
                setting_background = np.asarray(
                    [raw_bg["images"][int(i)] for i in bg[frames]]
                )
        data = CountData.from_images(
            setting_images,
            banks[frames],
            setting_background,
            binning,
            None if masks is None else masks[frames],
        )
        if bg is None:
            for bank, index in data.pixel_indices.items():
                live = index >= 0
                if not np.any(live):
                    raise ValueError(f"bank {bank} has no valid count bins")
                frame = np.full(index.shape, np.median(data.counts[index[live]]))
                frame[live] = data.counts[index[live]]
                data.background[index[live]] = np.maximum(
                    median_filter(frame, size=7)[live], 0.1
                )
        children.append(
            OrientationModel(
                data,
                instrument,
                cell,
                space_group,
                band,
                d_min,
                sigma_px,
                profile,
                {bank: dets[bank] for bank in banks[frames]},
                origins[frames[0]],
            )
        )
    if multi:
        from subhkl.search.multi_setting import MultiSettingModel

        model = MultiSettingModel(
            children,
            R[[frames[0] for frames in groups]],
            free_roll=not np.allclose(
                (R @ R[0].T) @ np.array([0.0, 0.0, 1.0]),
                [0.0, 0.0, 1.0],
                atol=1e-8,
                rtol=0,
            ),
        )
    else:
        model = children[0]
    data = model.data
    log(
        f"Reflection dictionary: {children[0].n_hkl} hkl per setting; {len(groups)} setting(s)"
    )
    if bootstrap:
        with h5py.File(bootstrap, "r") as f:
            if "solve/orientations_sample" in f:
                sample = _rotations(f["solve/orientations_sample"][()])
                orientations = sample if multi else R0 @ sample
            elif "solve/orientations_lab" in f:
                lab = _rotations(f["solve/orientations_lab"][()])
                orientations = R0.T @ lab if multi else lab
            elif "orientations" in f:
                lab = _rotations(f["orientations"][()])
                orientations = R0.T @ lab if multi else lab
            else:
                sample = _rotations(f["sample/U"][()])
                orientations = sample if multi else R0 @ sample
        scores = np.full(len(orientations), np.nan)
    else:
        log("Searching orientations from all signed count residuals...")
        orientations, scores = propose(model, cell, n_candidates, proposal_binning)
    log(
        f"Fitting {len(orientations)} orientation candidates on {len(data.counts)} valid count bins"
    )
    g = np.zeros(getattr(model, "geometry_size", 6))
    fit = fit_intensities(
        model.design(orientations, g), data, len(orientations), penalty, rtol=1e-6
    )
    log(
        f"Initial stationarity: reflections {fit.reflection_kkt:g}, "
        f"backgrounds {fit.background_kkt:g}; tolerance {fit.kkt_tolerance:g} "
        f"({fit.newton_iterations} Newton, {fit.cg_iterations} CG iterations)"
    )
    if not fit.converged:
        log(f"initial intensity solve did not converge (residual {fit.kkt:g})")
    initial = fit.objective
    optimizer = None
    if fit.converged and do_refine and np.any(fit.group_norms > 0):
        result = refine(
            model,
            orientations,
            penalty,
            max_evals=max_evals,
            refine_geometry=not fixed_geometry,
            log=log if verbose else None,
        )
        g, orientations, fit, optimizer = (
            result["g"],
            result["orientations"],
            result["fit"],
            result["optimizer"],
        )
    active = np.flatnonzero(fit.group_norms > 0)
    success = fit.converged and (optimizer is None or optimizer.success)
    status = (
        "not_converged"
        if not success
        else ("no_orientation" if len(active) == 0 else "converged")
    )
    output = Path(output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".solve-", suffix=".h5", dir=output.parent)
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as out:
            out.attrs["instrument"] = instrument
            out.attrs["source"] = str(Path(frames_filename).resolve())
            out.attrs["workflow"] = "solve"
            for key, value in zip(CELL_KEYS, cell):
                out[f"sample/{key}"] = value
            out["sample/space_group"] = space_group
            out["sample/B"] = model.B
            out["instrument/wavelength"] = band
            out["beam/ki_vec"] = [0.0, 0.0, 1.0]
            out["goniometer/R"] = R
            # Preserve the setting metadata consumed by the predictor.
            with h5py.File(source, "r") as meta:
                if "goniometer" in meta:
                    for key in meta["goniometer"]:
                        if key != "R" and not (
                            key == "angles" and corrected_angles is not None
                        ):
                            meta.copy(f"goniometer/{key}", out["goniometer"])
            if corrected_angles is not None:
                out["goniometer/angles"] = corrected_angles
                # Predictor displacements ride on the innermost lever arm.
                # Canonicalize a sample-frame vector to that native layout.
                if "goniometer/translations" in out and out[
                    "goniometer/translations"
                ].shape == (3,):
                    lever = np.zeros((corrected_angles.shape[1], 3))
                    lever[-1] = out["goniometer/translations"][()]
                    del out["goniometer/translations"]
                    out["goniometer/translations"] = lever
                if angles_changed and "goniometer/angles_nominal" not in out:
                    with h5py.File(source, "r") as meta:
                        out["goniometer/angles_nominal"] = meta["goniometer/angles"][()]
            if (
                "goniometer/per_run" in out
                and "goniometer/per_run/frame_to_run" not in out
            ):
                out["goniometer/per_run/frame_to_run"] = run_ids
            if (
                "goniometer/per_run/trans_m" in out
                and "goniometer/translations" not in out
                and corrected_angles is not None
            ):
                out["goniometer/translations"] = np.zeros(
                    (corrected_angles.shape[1], 3)
                )
            with h5py.File(frames_filename, "r") as frames:
                for key in ("files", "file_offsets"):
                    if key in frames:
                        frames.copy(key, out)
            with h5py.File(source, "r") as meta:
                for key in ("files", "file_offsets"):
                    if key not in out and key in meta:
                        meta.copy(key, out)
            out["bank_ids"] = banks
            absolute = {}
            for child in children:
                absolute.update(child.detectors(g))
            write_detector_calibration(out, absolute)
            report = out.create_group("solve")
            report.attrs["status"] = status
            report.attrs["orientation_frame"] = "sample" if multi else "lab"
            report.attrs["binning"] = binning
            report.attrs["d_min"] = d_min
            report.attrs["requested_candidates"] = n_candidates
            report.attrs["max_evaluations"] = max_evals
            report.attrs["metadata_source"] = str(Path(source).resolve())
            report.attrs["mask_source"] = str(static_mask_file or "")
            report.attrs["proposal_binning"] = proposal_binning
            report.attrs["penalty"] = penalty
            report.attrs["sigma_px"] = sigma_px
            report.attrs["fixed_geometry"] = fixed_geometry
            report.attrs["refinement_enabled"] = do_refine
            report.attrs["background_source"] = str(
                background_file or "local median of all valid bins"
            )
            report.attrs["profile_source"] = str(profile_file or "Gaussian")
            report.attrs["bootstrap"] = str(bootstrap or "")
            report["g"] = g
            sample = orientations if multi else R0.T @ orientations
            report["orientations_sample"] = sample
            report["orientations_lab"] = (
                np.einsum("sij,njk->snik", model.rotations, sample)
                if multi
                else orientations
            )
            frame_to_setting = np.empty(len(banks), int)
            for setting, frames in enumerate(groups):
                frame_to_setting[frames] = setting
            report["frame_to_setting"] = frame_to_setting
            report["frame_to_run"] = run_ids
            report["sample_origin_lab"] = origins
            report["setting_first_frame"] = [frames[0] for frames in groups]
            report["geometry_parameter_names"] = np.asarray(
                ["scale", "tilt_x", "tilt_y"]
                + (["roll_z"] if len(g) == 7 else [])
                + ["tx", "ty", "tz"],
                dtype=h5py.string_dtype(),
            )
            report["proposal_scores"] = scores
            report["active"] = active
            report["group_norms"] = fit.group_norms
            report["group_weights"] = fit.weights
            report["intensities"] = fit.intensities
            hkl = np.rint(model.G @ np.linalg.inv(model.B).T).astype(int)
            report["hkl"] = np.tile(hkl, (len(groups), 1))
            report["reflection_setting"] = np.repeat(np.arange(len(groups)), len(hkl))
            report["background_scales"] = fit.background_scales
            report["objective_initial"] = initial
            report["objective_final"] = fit.objective
            report["stationarity_residual"] = fit.kkt
            report["stationarity_tolerance"] = fit.kkt_tolerance
            report["reflection_stationarity_residual"] = fit.reflection_kkt
            report["background_stationarity_residual"] = fit.background_kkt
            report["newton_iterations"] = fit.newton_iterations
            report["cg_iterations"] = fit.cg_iterations
            report["inner_converged"] = fit.converged
            if optimizer is not None:
                report.attrs["optimizer_message"] = str(optimizer.message)
                report["n_evaluations"] = optimizer.nfev
            if profile is not None:
                out["profile/u"], out["profile/f"] = profile.u, profile.f
            if len(active) and success:
                primary = int(active[np.argmax(fit.group_norms[active])])
                report["primary_orientation"] = primary
                out["sample/U"] = sample[primary]
        os.replace(tmp, output)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    log(f"solve: {status}, {len(active)} active components; wrote {output}")
    if len(active) > 1:
        log(
            "All components are in solve/; sample/U selects the primary for existing single-crystal consumers."
        )
    return dict(status=status, active=active, g=g, orientations=orientations, fit=fit)
