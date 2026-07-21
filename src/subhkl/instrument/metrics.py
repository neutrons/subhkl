"""
Metrics computation module for indexing quality assessment.
Provides functions to compute and return angular/distance errors from indexed peaks.
"""

import h5py
import numpy as np
import scipy.spatial

from subhkl.config import beamlines
from subhkl.instrument.detector import Detector
from subhkl.instrument.physics import calculate_angular_error
from subhkl.optimization import FindUB


def _get_safe_R_stack(R_file_in, run_indices_in, target_len):
    if R_file_in is None:
        return [np.eye(3)] * target_len

    if R_file_in.ndim == 3 and len(R_file_in) == target_len:
        return R_file_in

    def safe_get_single_R(r_idx):
        ridx = int(r_idx)
        if R_file_in.ndim == 3 and len(R_file_in) > ridx:
            return R_file_in[ridx]
        return R_file_in[0]

    if R_file_in.ndim == 3:
        if len(R_file_in) == target_len:
            return R_file_in
        return [safe_get_single_R(r) for r in run_indices_in]
    return [R_file_in] * target_len


def resolve_indices(f_handle):
    if "peaks/run_index" in f_handle:
        return f_handle["peaks/run_index"][()]
    if "peaks/image_index" in f_handle:
        return f_handle["peaks/image_index"][()]
    if "bank" in f_handle:
        return f_handle["bank"][()]
    return None


def extract_xyz_from_file(file_path, instrument=None):
    """Safely extracts physical XYZ lab coordinates from a file by dynamically converting pixels, applying calibrations if present.

    Returns (xyz, slot_index, panel_index, frame_index):
      - slot_index: the per-peak index used internally to align with per-slot
        arrays like goniometer/angles and goniometer/R. For Predictor "banks"
        output this is a composite frame+panel slot, not a true frame number.
      - panel_index: the physical detector bank/panel each peak was recorded on.
      - frame_index: the true rotation/goniometer-setting each peak belongs to.
        For Predictor "banks" output (which only stores a composite slot index)
        this is reconstructed by grouping slots that share an identical
        goniometer/angles row, since those rows are tiled per-panel from a
        single frame's angles at acquisition time.
    """
    with h5py.File(file_path, "r") as f:
        run_idx = resolve_indices(f)

        bank_array = None
        if "bank" in f:
            bank_array = f["bank"][()]
        elif "peaks/bank" in f:
            bank_array = f["peaks/bank"][()]
        elif "bank_ids" in f and run_idx is not None:
            b_ids = f["bank_ids"][()]
            bank_array = np.array([b_ids[int(r)] for r in run_idx])
        else:
            bank_array = run_idx

        calibration_dict = {}
        if "detector_calibration" in f:
            calib_grp = f["detector_calibration"]
            for b_key in calib_grp.keys():
                calib = {
                    "center": calib_grp[b_key]["center"][()],
                    "uhat": calib_grp[b_key]["uhat"][()],
                    "vhat": calib_grp[b_key]["vhat"][()],
                }
                if "width" in calib_grp[b_key] and "height" in calib_grp[b_key]:
                    calib["width"] = float(calib_grp[b_key]["width"][()])
                    calib["height"] = float(calib_grp[b_key]["height"][()])
                calibration_dict[b_key] = calib

        # 1. Coordinate array reconstruction (Finder, Indexer, Integrator)
        if "peaks/pixel_r" in f and "peaks/pixel_c" in f:
            if instrument is None:
                print(
                    "Warning: Instrument not provided. Cannot compute physical coordinates from pixels."
                )
                return None, None, None, None

            pixel_r = f["peaks/pixel_r"][()]
            pixel_c = f["peaks/pixel_c"][()]
            xyz = np.zeros((len(pixel_r), 3))

            if run_idx is None:
                run_idx = np.zeros(len(xyz))
            if bank_array is None:
                bank_array = run_idx

            if "peaks/run_index" in f:
                frame_idx = f["peaks/run_index"][()]
            elif "peaks/image_index" in f:
                frame_idx = f["peaks/image_index"][()]
            else:
                frame_idx = run_idx

            for phys_bank in np.unique(bank_array):
                bank_str = f"bank_{int(phys_bank)}"
                mask = bank_array == phys_bank
                if not np.any(mask):
                    continue

                try:
                    det_config = beamlines[instrument][str(int(phys_bank))]
                    det = Detector(det_config)

                    if bank_str in calibration_dict:
                        det.center = calibration_dict[bank_str]["center"]
                        det.uhat = calibration_dict[bank_str]["uhat"]
                        det.vhat = calibration_dict[bank_str]["vhat"]
                        if "width" in calibration_dict[bank_str]:
                            det.width = calibration_dict[bank_str]["width"]
                            det.height = calibration_dict[bank_str]["height"]

                    xyz[mask] = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])
                except KeyError:
                    pass

            return xyz, run_idx, bank_array, frame_idx

        # 2. Bank/Pixel format (Predictor output)
        if "banks" in f:
            if instrument is None:
                print(
                    "Warning: Instrument not provided. Cannot compute physical coordinates from prediction pixels."
                )
                return None, None, None, None

            xyz_list, run_list, panel_list, frame_list = [], [], [], []
            bank_ids = f["bank_ids"][()] if "bank_ids" in f else None

            # goniometer/angles has one row per banks/<key> slot (tiled across
            # panels within a frame when the file was written), so slots that
            # share an identical angle row belong to the same true frame.
            gonio_angles_all = (
                f["goniometer/angles"][()] if "goniometer/angles" in f else None
            )
            frame_by_slot = {}
            if gonio_angles_all is not None:
                seen_rows = {}
                for key_str in sorted(f["banks"].keys(), key=int):
                    key = int(key_str)
                    row_bytes = (
                        gonio_angles_all[key].tobytes()
                        if key < len(gonio_angles_all)
                        else None
                    )
                    if row_bytes not in seen_rows:
                        seen_rows[row_bytes] = len(seen_rows)
                    frame_by_slot[key] = seen_rows[row_bytes]

            for img_key_str in f["banks"].keys():
                img_idx = int(img_key_str)
                grp = f[f"banks/{img_key_str}"]
                i_p, j_p = grp["i"][()], grp["j"][()]

                phys_bank = bank_ids[img_idx] if bank_ids is not None else img_idx
                try:
                    det_config = beamlines[instrument][str(phys_bank)]
                    det = Detector(det_config)

                    bank_str = f"bank_{phys_bank}"
                    if bank_str in calibration_dict:
                        det.center = calibration_dict[bank_str]["center"]
                        det.uhat = calibration_dict[bank_str]["uhat"]
                        det.vhat = calibration_dict[bank_str]["vhat"]
                        if "width" in calibration_dict[bank_str]:
                            det.width = calibration_dict[bank_str]["width"]
                            det.height = calibration_dict[bank_str]["height"]

                    xyz = det.pixel_to_lab(i_p, j_p)
                    if xyz.ndim == 1:
                        xyz = xyz[np.newaxis, :]

                    xyz_list.append(xyz)
                    run_list.extend([img_idx] * len(xyz))
                    panel_list.extend([int(phys_bank)] * len(xyz))
                    frame_list.extend(
                        [frame_by_slot.get(img_idx, img_idx)] * len(xyz)
                    )
                except KeyError:
                    continue

            if xyz_list:
                return (
                    np.vstack(xyz_list),
                    np.array(run_list),
                    np.array(panel_list),
                    np.array(frame_list),
                )

    return None, None, None, None


def compute_metrics(
    file1: str,
    file2: str | None = None,
    instrument: str | None = None,
    d_min: float | None = None,
    per_run: bool = False,
    ki_vec_override: np.ndarray | None = None,
    return_per_peak: bool = False,
) -> dict:
    try:
        with h5py.File(file1, "r") as f:
            ub_helper = FindUB()
            ub_helper.a = f["sample/a"][()]
            ub_helper.b = f["sample/b"][()]
            ub_helper.c = f["sample/c"][()]
            ub_helper.alpha = f["sample/alpha"][()]
            ub_helper.beta = f["sample/beta"][()]
            ub_helper.gamma = f["sample/gamma"][()]
            B_mat = ub_helper.reciprocal_lattice_B()
            U = f["sample/U"][()] if "sample/U" in f else np.eye(3)
            if "goniometer/translations" in f:
                sample_offset = f["goniometer/translations"][()]
            else:
                sample_offset = np.zeros(3)

            gonio_axes = f["goniometer/axes"][()] if "goniometer/axes" in f else None
            gonio_angles = (
                f["goniometer/angles"][()] if "goniometer/angles" in f else None
            )

            gonio_offsets = None
            off_data = f.get("goniometer/offsets")
            if off_data is not None:
                gonio_names = (
                    f["goniometer/names"][()] if "goniometer/names" in f else None
                )
                if gonio_names is not None:
                    gonio_names = [
                        n.decode("utf-8") if isinstance(n, bytes) else str(n)
                        for n in gonio_names
                    ]
                if isinstance(off_data, h5py.Group) and gonio_names is not None:
                    gonio_offsets = np.zeros(len(gonio_names), dtype=np.float32)
                    for i, name in enumerate(gonio_names):
                        if name in off_data:
                            gonio_offsets[i] = float(off_data[name][()])
                else:
                    raw_offs = off_data[()]
                    gonio_offsets = np.zeros(len(raw_offs), dtype=np.float32)
                    gonio_offsets[: len(raw_offs)] = raw_offs

            if ki_vec_override is not None:
                ki_vec = ki_vec_override
            else:
                ki_vec = (
                    f["beam/ki_vec"][()]
                    if "beam/ki_vec" in f
                    else np.array([0.0, 0.0, 1.0])
                )

            R_file = f["goniometer/R"][()] if "goniometer/R" in f else None

            if instrument is None:
                instrument = f.attrs.get("instrument")

        (
            matched_h,
            matched_k,
            matched_l,
            matched_lam,
            matched_xyz,
            matched_R,
            matched_run,
            matched_frame,
            matched_panel,
        ) = [], [], [], [], [], [], [], [], []

        # ==========================================
        # TWO FILE COMPARISON
        # ==========================================
        if file2 is not None:
            if instrument is None:
                return {
                    "error_message": "ERROR: --instrument required for matching when not found in file attributes."
                }

            xyz_1, run_1, panel_1, frame_1 = extract_xyz_from_file(file1, instrument)
            xyz_2, run_2, panel_2, frame_2 = extract_xyz_from_file(file2, instrument)

            if xyz_1 is None or xyz_2 is None:
                return {
                    "error_message": "ERROR: Could not extract physical XYZ coordinates from one or both files."
                }

            with h5py.File(file1, "r") as f1:
                if "banks" in f1:
                    for img_key_str in f1["banks"].keys():
                        img_idx = int(img_key_str)
                        grp = f1[f"banks/{img_key_str}"]
                        h_p, k_p, l_p, lam_p = (
                            grp["h"][()],
                            grp["k"][()],
                            grp["l"][()],
                            grp["wavelength"][()],
                        )

                        mask_1 = run_1 == img_idx
                        mask_2 = run_2 == img_idx
                        if not np.any(mask_1) or not np.any(mask_2):
                            continue

                        xyz_1_run = xyz_1[mask_1]
                        xyz_2_run = xyz_2[mask_2]

                        tree = scipy.spatial.KDTree(xyz_1_run)
                        dists, idxs = tree.query(xyz_2_run)
                        valid = dists < 0.01

                        if np.any(valid):
                            num_valid = np.sum(valid)
                            matched_h.extend(h_p[idxs[valid]])
                            matched_k.extend(k_p[idxs[valid]])
                            matched_l.extend(l_p[idxs[valid]])
                            matched_lam.extend(lam_p[idxs[valid]])
                            matched_xyz.extend(xyz_2_run[valid])
                            matched_run.extend([img_idx] * num_valid)
                            matched_frame.extend(frame_1[mask_1][idxs[valid]])
                            matched_panel.extend(panel_1[mask_1][idxs[valid]])
                            matched_R.extend(
                                _get_safe_R_stack(
                                    R_file, [img_idx] * num_valid, num_valid
                                )
                            )
                else:
                    h_p = f1["peaks/h"][()]
                    k_p = f1["peaks/k"][()]
                    l_p = f1["peaks/l"][()]
                    lam_p = f1["peaks/lambda"][()]

                    unique_runs = np.unique(run_1)
                    for r in unique_runs:
                        mask_1 = run_1 == r
                        mask_2 = run_2 == r
                        if not np.any(mask_1) or not np.any(mask_2):
                            continue

                        xyz_1_run = xyz_1[mask_1]
                        xyz_2_run = xyz_2[mask_2]

                        tree = scipy.spatial.KDTree(xyz_1_run)
                        dists, idxs = tree.query(xyz_2_run)
                        valid = dists < 0.01

                        if np.any(valid):
                            num_valid = np.sum(valid)
                            matched_h.extend(h_p[mask_1][idxs[valid]])
                            matched_k.extend(k_p[mask_1][idxs[valid]])
                            matched_l.extend(l_p[mask_1][idxs[valid]])
                            matched_lam.extend(lam_p[mask_1][idxs[valid]])
                            matched_xyz.extend(xyz_2_run[valid])
                            matched_run.extend([r] * num_valid)
                            matched_frame.extend(frame_1[mask_1][idxs[valid]])
                            matched_panel.extend(panel_1[mask_1][idxs[valid]])
                            matched_R.extend(
                                _get_safe_R_stack(R_file, [r] * num_valid, num_valid)
                            )

        # ==========================================
        # SINGLE FILE METRICS
        # ==========================================
        else:
            with h5py.File(file1, "r") as f:
                if "peaks/h" not in f:
                    return {"error_message": "No peaks/h dataset found in file"}
                matched_h = f["peaks/h"][()]
                matched_k = f["peaks/k"][()]
                matched_l = f["peaks/l"][()]
                matched_lam = f["peaks/lambda"][()]

                xyz, r_idx, panel_idx, frame_idx = extract_xyz_from_file(
                    file1, instrument
                )
                if xyz is None:
                    return {"error_message": "Could not extract XYZ coordinates."}

                matched_xyz = xyz
                matched_run = r_idx
                matched_frame = frame_idx
                matched_panel = panel_idx
                matched_R = _get_safe_R_stack(R_file, matched_run, len(matched_h))

        h = np.array(matched_h)
        k = np.array(matched_k)
        l = np.array(matched_l)
        lam = np.array(matched_lam)
        xyz_det = np.array(matched_xyz)
        R_all = np.array(matched_R)
        run_index = np.array(matched_run)
        frame_index = np.array(matched_frame)
        panel_index = np.array(matched_panel)

        mask = (h != 0) | (k != 0) | (l != 0)
        if np.sum(mask) == 0:
            return {
                "median_d_err": 0.0,
                "mean_d_err": 0.0,
                "max_d_err": 0.0,
                "median_ang_err": 0.0,
                "mean_ang_err": 0.0,
                "max_ang_err": 0.0,
                "num_peaks": 0,
            }

        h, k, l = h[mask], k[mask], l[mask]
        lam = lam[mask]
        xyz_det = xyz_det[mask]
        R_all = R_all[mask]
        run_index = run_index[mask]
        frame_index = frame_index[mask]
        panel_index = panel_index[mask]

        d_filter_message = None
        if d_min is not None:
            hkl_vecs = np.stack([h, k, l], axis=1)
            q_cryst = hkl_vecs @ B_mat.T
            q_mag = np.linalg.norm(q_cryst, axis=1)
            with np.errstate(divide="ignore"):
                d_vals = 1.0 / q_mag
            d_mask = d_vals >= d_min
            if np.sum(d_mask) == 0:
                return {"error_message": f"No peaks found with d >= {d_min} A."}
            h, k, l = h[d_mask], k[d_mask], l[d_mask]
            lam = lam[d_mask]
            xyz_det = xyz_det[d_mask]
            R_all = R_all[d_mask]
            run_index = run_index[d_mask]
            frame_index = frame_index[d_mask]
            panel_index = panel_index[d_mask]
            d_filter_message = f"Filtered to {len(h)} peaks with d >= {d_min} A."

        UB = U @ B_mat
        if gonio_angles is not None:
            if gonio_angles.ndim == 2:
                num_axes = len(gonio_axes) if gonio_axes is not None else 1
                if gonio_angles.shape[1] == num_axes:
                    gonio_angles_mapped = gonio_angles[run_index, :]
                else:
                    gonio_angles_mapped = gonio_angles[:, run_index].T
            else:
                gonio_angles_mapped = gonio_angles
        else:
            gonio_angles_mapped = None

        d_err, ang_err = calculate_angular_error(
            xyz_det,
            h,
            k,
            l,
            lam,
            UB,
            sample_offset,
            ki_vec,
            R_all,
            gonio_axes=gonio_axes,
            gonio_angles=gonio_angles_mapped,
            gonio_offsets=gonio_offsets,  # <-- Pass Down
        )

        result = {
            "median_d_err": float(np.median(d_err)),
            "mean_d_err": float(np.mean(d_err)),
            "max_d_err": float(np.max(d_err)),
            "median_ang_err": float(np.median(ang_err)),
            "mean_ang_err": float(np.mean(ang_err)),
            "max_ang_err": float(np.max(ang_err)),
            "num_peaks": len(h),
        }

        if d_filter_message:
            result["filter_message"] = d_filter_message

        if per_run:
            unique_frames = sorted(np.unique(frame_index))
            run_errors = []
            for fr in unique_frames:
                fr_mask = frame_index == fr
                if np.sum(fr_mask) > 0:
                    run_errors.append(
                        (
                            int(fr),
                            float(np.median(ang_err[fr_mask])),
                            int(np.sum(fr_mask)),
                        )
                    )
            run_errors.sort(key=lambda x: x[1], reverse=True)
            result["per_run_errors"] = run_errors

        if return_per_peak:
            result["per_peak"] = {
                "frame_index": frame_index,
                "panel_index": panel_index,
                "h": h,
                "k": k,
                "l": l,
                "d_err": d_err,
                "ang_err": ang_err,
            }

        return result

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error_message": f"Exception during metrics computation: {e!s}"}


def write_per_peak_csv(result: dict, output_path: str) -> None:
    """Write the per-peak frame_index/panel_index/h/k/l/d_err/ang_err arrays
    from a compute_metrics(..., return_per_peak=True) result to a CSV file."""
    per_peak = result.get("per_peak")
    if per_peak is None:
        raise ValueError(
            "result has no 'per_peak' data; call compute_metrics(..., return_per_peak=True)"
        )

    import csv

    fields = ["frame_index", "panel_index", "h", "k", "l", "d_err", "ang_err"]
    columns = [per_peak[field] for field in fields]

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        writer.writerows(zip(*columns))
