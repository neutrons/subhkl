"""Shared threshold-free count workflow for orientation and detector geometry.

The draft workflow accepts a single still or frames pooled at one setting.
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


def propose(model, cell, n_candidates=4, proposal_bin=32, g=None):
    """Use every valid signed residual in spatial bins; no detection cutoff."""
    from subhkl.search.spherical import lattice_ladder, panel_directions

    data = model.data
    factor = proposal_bin // data.bin_px
    if factor < 1 or proposal_bin % data.bin_px:
        raise ValueError("proposal binning must be a multiple of count binning")
    dets = model.detectors(np.zeros(6) if g is None else g)
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
    D, weights = np.concatenate(dirs), np.concatenate(residuals)
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
        sin_theta=-D[:, 2],
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
    """Solve one still/one-setting pooled stack and atomically write its result."""
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
        images, banks = f["images"][()], f["bank_ids"][()].astype(int)
    if images.ndim != 3 or len(banks) != len(images) or len(set(banks)) != len(banks):
        raise ValueError(
            "solve requires one image per bank: select one still or pool one setting first"
        )
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
        R = (
            _rotations(f["goniometer/R"][()])
            if "goniometer/R" in f
            else np.eye(3)[None]
        )
        if not np.allclose(R, R[0], atol=1e-6):
            raise ValueError(
                "mixed goniometer settings are not supported; solve one setting at a time"
            )
        R0 = R[0]
        if "goniometer/R" not in f:
            if "goniometer/angles" in f:
                raise ValueError("setting angles need matching goniometer/R metadata")
            log(
                "No goniometer/R supplied: defining the output sample frame as the lab frame."
            )
        if "beam/ki_vec" in f and not np.allclose(f["beam/ki_vec"][()], [0, 0, 1]):
            raise ValueError("draft solve currently requires incident beam +z")
        if "goniometer/translations" in f and not np.allclose(
            f["goniometer/translations"][()], 0
        ):
            raise ValueError(
                "nonzero goniometer translations require a common-origin geometry model"
            )
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
    dets = geometry_detectors(instrument, np.zeros(7), list(banks))
    if set(dets) != set(banks):
        raise ValueError("input contains banks absent from instrument geometry")
    for bank, im in zip(banks, images):
        d = dets[bank]
        if d.config["panel"] != "flat" or im.shape != (d.n, d.m):
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
            load_mask_for_banks(static_mask_file, list(banks), images.shape[1:]) > 0.5
        )
    # Construct binned data before estimating shape, so all validity handling
    # and count conservation lives in CountData, not a spot detector.
    data = CountData.from_images(images, banks, np.ones(len(banks)), binning, masks)
    if background_file:
        with h5py.File(background_file, "r") as f:
            ref_banks = f["bank_ids"][()].astype(int)
            if len(set(ref_banks)) != len(ref_banks) or set(ref_banks) != set(banks):
                raise ValueError("background bank ids must match counts exactly")
            order = {b: i for i, b in enumerate(ref_banks)}
            bg = np.asarray([f["images"][order[b]] for b in banks])
        data.background = CountData.from_images(
            images, banks, bg, binning, masks
        ).background
    else:
        for bank, index in data.pixel_indices.items():
            live = index >= 0
            if not np.any(live):
                raise ValueError(f"bank {bank} has no valid count bins")
            frame = np.full(index.shape, np.median(data.counts[index[live]]))
            frame[live] = data.counts[index[live]]
            data.background[index[live]] = np.maximum(
                median_filter(frame, size=7)[live], 0.1
            )
    profile = None
    if profile_file:
        with h5py.File(profile_file, "r") as f:
            profile = RadialProfile(f["profile/u"][()], f["profile/f"][()])
    model = OrientationModel(
        data, instrument, cell, space_group, band, d_min, sigma_px, profile, dets
    )
    log(f"Reflection dictionary: {model.n_hkl} hkl at d_min={d_min:g} Angstrom")
    if bootstrap:
        with h5py.File(bootstrap, "r") as f:
            if "solve/orientations_lab" in f:
                orientations = _rotations(f["solve/orientations_lab"][()])
            elif "orientations" in f:
                orientations = _rotations(f["orientations"][()])
            else:
                orientations = R0 @ _rotations(f["sample/U"][()])
        scores = np.full(len(orientations), np.nan)
    else:
        log("Searching orientations from all signed count residuals...")
        orientations, scores = propose(model, cell, n_candidates, proposal_binning)
    log(
        f"Fitting {len(orientations)} orientation candidates on {len(data.counts)} valid count bins"
    )
    g = np.zeros(6)
    fit = fit_intensities(
        model.design(orientations, g), data, len(orientations), penalty, rtol=1e-6
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
            out["goniometer/R"] = R0[None]
            # Preserve the setting metadata consumed by the predictor.
            with h5py.File(source, "r") as meta:
                if "goniometer" in meta:
                    for key in meta["goniometer"]:
                        if key != "R":
                            meta.copy(f"goniometer/{key}", out["goniometer"])
            out["bank_ids"] = banks
            write_detector_calibration(out, model.detectors(g))
            report = out.create_group("solve")
            report.attrs["status"] = status
            report.attrs["orientation_frame"] = "lab"
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
            report["orientations_lab"] = orientations
            report["orientations_sample"] = R0.T @ orientations
            report["proposal_scores"] = scores
            report["active"] = active
            report["group_norms"] = fit.group_norms
            report["group_weights"] = fit.weights
            report["intensities"] = fit.intensities
            report["hkl"] = np.rint(model.G @ np.linalg.inv(model.B).T).astype(int)
            report["background_scales"] = fit.background_scales
            report["objective_initial"] = initial
            report["objective_final"] = fit.objective
            report["stationarity_residual"] = fit.kkt
            report["stationarity_tolerance"] = fit.kkt_tolerance
            report["inner_converged"] = fit.converged
            if optimizer is not None:
                report.attrs["optimizer_message"] = str(optimizer.message)
                report["n_evaluations"] = optimizer.nfev
            if profile is not None:
                out["profile/u"], out["profile/f"] = profile.u, profile.f
            if len(active) and success:
                primary = int(active[np.argmax(fit.group_norms[active])])
                report["primary_orientation"] = primary
                out["sample/U"] = R0.T @ orientations[primary]
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
