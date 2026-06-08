import h5py
import numpy as np
import jax.numpy as jnp

# NOTE(Vivek): deprecate and use Goniometer class to handler rotation calc
from subhkl.instrument.goniometer import (
    get_rotation_data_from_nexus,
)
from subhkl.integration import Peaks
from subhkl.optimization import FindUB
from subhkl.io.export import ImageStackMerger, MTZExporter
from subhkl.search.sparse_hough import SparseHoughIndexer

from typing import List


def apply_detector_calibration(hdf5_filename: str, instrument: str):
    """
    Reads refined detector metrology from an indexer/prediction file (if present)
    and overrides the in-memory beamlines configuration so downstream
    tasks natively use the calibrated geometry.
    """
    from subhkl.config import beamlines
    import os

    if not os.path.exists(hdf5_filename):
        return

    with h5py.File(hdf5_filename, "r") as f:
        if "detector_calibration" in f:
            print(f"Loading calibrated detector geometry from {hdf5_filename}...")
            calib_grp = f["detector_calibration"]
            count = 0
            for bank_key in calib_grp.keys():
                bank_id = bank_key.replace("bank_", "")
                if instrument in beamlines and bank_id in beamlines[instrument]:
                    beamlines[instrument][bank_id]["center"] = calib_grp[bank_key][
                        "center"
                    ][()].tolist()
                    beamlines[instrument][bank_id]["uhat"] = calib_grp[bank_key][
                        "uhat"
                    ][()].tolist()
                    beamlines[instrument][bank_id]["vhat"] = calib_grp[bank_key][
                        "vhat"
                    ][()].tolist()
                    if "width" in calib_grp[bank_key] and "height" in calib_grp[bank_key]:
                        beamlines[instrument][bank_id]["width"] = float(calib_grp[bank_key]["width"][()])
                        beamlines[instrument][bank_id]["height"] = float(calib_grp[bank_key]["height"][()])
                    count += 1
            if count > 0:
                print(f"Successfully applied calibration to {count} detector panels.")


def run_index(
    peaks_h5_filename: str,
    output_peaks_filename: str,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    space_group: str | None = None,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    ki_vec: list[float] | np.ndarray | None = None,
    original_nexus_filename: str | None = None,
    instrument_name: str | None = None,
    strategy_name: str = "DE",
    sigma_init: float | None = None,
    n_runs: int = 1,
    population_size: int = 1000,
    gens: int = 100,
    seed: int = 0,
    tolerance_deg: float = 0.1,
    freeze_orientation: bool = False,
    refine_lattice: bool = False,
    lattice_bound_frac: float = 0.05,
    refine_goniometer: bool = False,
    refine_goniometer_axes: list[str] | None = None,
    goniometer_bound_deg: float | list[float] | np.ndarray = 5.0,
    refine_goniometer_trans: bool = False,
    goniometer_trans_bound_meters: float | list[float] | np.ndarray = 0.005,
    refine_beam: bool = False,
    beam_bound_deg: float = 1.0,
    refine_detector: bool = False,
    refine_detector_banks: list[int] | None = None,
    detector_modes: list[str] | None = None,
    detector_trans_bound_meters: float = 0.005,
    detector_rot_bound_deg: float = 1.0,
    detector_global_rot_bound_deg: float = 2.0,
    detector_global_rot_axis: list[float] | np.ndarray | None = None,
    detector_global_trans_bound_meters: float = 0.01,
    detector_radial_bound_frac: float = 0.05,
    detector_area_bound_frac: float = 0.05,
    cylinder_axis: list[float] | np.ndarray | None = None,
    bootstrap_filename: str | None = None,
    batch_size: int | None = None,
    input_data: dict | None = None,
    num_candidates: int | None = None,
    no_index: bool | None = None,
):
    input_data = input_data or {}

    if detector_modes is None:
        detector_modes = ["independent"]

    if detector_global_rot_axis is not None:
        if "global_rot" in detector_modes:
            print(f"Auto-switching detector mode: 'global_rot' -> 'global_rot_axis' (Axis: {detector_global_rot_axis})")
            detector_modes = [
                "global_rot_axis" if mode == "global_rot" else mode 
                for mode in detector_modes
            ]
    else:
        # Safe default fallback for downstream JAX compilation
        detector_global_rot_axis = [0.0, 1.0, 0.0]

    if cylinder_axis is not None:
        if "radial" in detector_modes:
            print(f"Auto-switching detector mode: 'radial' -> 'cylindrical' (Axis: {cylinder_axis})")
            detector_modes = [
                "cylindrical" if mode == "radial" else mode 
                for mode in detector_modes
            ]
    else:
        cylinder_axis = [0.0, 1.0, 0.0] # Safe default for downstream JAX compilation

    # --- INJECT BOOTSTRAP PHYSICS DIRECTLY ---
    if bootstrap_filename:
        apply_detector_calibration(bootstrap_filename, instrument_name)
        with h5py.File(bootstrap_filename, "r") as b_f:
            if "sample/a" in b_f:
                a = b_f["sample/a"][()]
            if "sample/b" in b_f:
                b = b_f["sample/b"][()]
            if "sample/c" in b_f:
                c = b_f["sample/c"][()]
            if "sample/alpha" in b_f:
                alpha = b_f["sample/alpha"][()]
            if "sample/beta" in b_f:
                beta = b_f["sample/beta"][()]
            if "sample/gamma" in b_f:
                gamma = b_f["sample/gamma"][()]

    print(f"Loading peaks from: {peaks_h5_filename}")
    with h5py.File(peaks_h5_filename, "r") as f:
        if a is None:
            a = f["sample/a"][()] if "sample/a" in f else None
        if b is None:
            b = f["sample/b"][()] if "sample/b" in f else None
        if c is None:
            c = f["sample/c"][()] if "sample/c" in f else None
        if alpha is None:
            alpha = f["sample/alpha"][()] if "sample/alpha" in f else None
        if beta is None:
            beta = f["sample/beta"][()] if "sample/beta" in f else None
        if gamma is None:
            gamma = f["sample/gamma"][()] if "sample/gamma" in f else None

        if space_group is None:
            file_sg = f["sample/space_group"][()] if "sample/space_group" in f else None
            space_group = (
                file_sg.decode("utf-8") if isinstance(file_sg, bytes) else file_sg
            )

        if None in (a, b, c, alpha, beta, gamma, space_group):
            raise ValueError(
                "Unit cell parameters (a,b,c,alpha,beta,gamma) and Space Group must be provided via CLI or exist in the input file."
            )

        from subhkl.core.spacegroup import get_space_group_object

        try:
            get_space_group_object(space_group)
        except ValueError as e:
            raise ValueError(f"Invalid space group '{space_group}': {e}")

        if wavelength_min is None or wavelength_max is None:
            if "instrument/wavelength" in f:
                wl = f["instrument/wavelength"][()]
                if wavelength_min is None:
                    wavelength_min = float(wl[0])
                if wavelength_max is None:
                    wavelength_max = float(wl[1])
            else:
                raise ValueError(
                    "Wavelength min/max not provided and not found in input file."
                )

        keys_to_load = [
            "peaks/intensity",
            "peaks/sigma",
            "peaks/radius",
            "peaks/h",
            "peaks/k",
            "peaks/l",
            "peaks/lambda",
            "goniometer/R",
            "goniometer/axes",
            "goniometer/angles",
            "goniometer/names",
            "goniometer/translations",
            "files",
            "file_offsets",
            "peaks/run_index",
            "peaks/image_index",
            "bank",
            "bank_ids",
            "beam/ki_vec",
            "peaks/pixel_r",
            "peaks/pixel_c",
        ]
        for k in keys_to_load:
            if k in f:
                input_data[k] = f[k][()]

        if ki_vec is not None:
            ki_vec_val = np.array(ki_vec)
        else:
            ki_vec_val = (
                f["beam/ki_vec"][()]
                if "beam/ki_vec" in f
                else np.array([0.0, 0.0, 1.0])
            )

        detector_params = None
        peak_pixel_coords = None
        target_banks = None

        if "peaks/pixel_r" in f and "peaks/pixel_c" in f:
            print("Reconstructing physical geometry from pixels for optimization...")
            if not instrument_name or not original_nexus_filename:
                raise ValueError(
                    "ERROR: Finder file contains pixels. You must provide --instrument and --nexus to rebuild geometry."
                )

            pixel_r = f["peaks/pixel_r"][()]
            pixel_c = f["peaks/pixel_c"][()]

            bank_array = None
            if "bank" in f:
                bank_array = f["bank"][()]
            elif "peaks/bank" in f:
                bank_array = f["peaks/bank"][()]
            elif "bank_ids" in f and "peaks/image_index" in f:
                b_ids = f["bank_ids"][()]
                img_idx = f["peaks/image_index"][()]
                bank_array = np.array([b_ids[int(idx)] for idx in img_idx])
            else:
                bank_array = f["peaks/image_index"][()]

            peaks_obj = Peaks(original_nexus_filename, instrument_name)
            from subhkl.config import beamlines
            from subhkl.instrument.detector import Detector

            if refine_detector:
                all_physical_banks = [int(k) for k in beamlines[instrument_name].keys()]
                target_banks = (
                    refine_detector_banks
                    if refine_detector_banks
                    else sorted(all_physical_banks)
                )

                centers, uhats, vhats, m, n, pw, ph = [], [], [], [], [], [], []
                bank_to_idx = {}
                valid_target_banks = []

                for b_id in target_banks:
                    try:
                        det = peaks_obj.get_detector(b_id)
                        centers.append(det.center)
                        uhats.append(det.uhat)
                        vhats.append(det.vhat)
                        m.append(det.m)
                        n.append(det.n)
                        pw.append(det.width / det.m)
                        ph.append(det.height / det.n)

                        # Map to the true contiguous index
                        bank_to_idx[b_id] = len(valid_target_banks) 
                        valid_target_banks.append(b_id)
                    except Exception as e:
                        print(f"WARNING: Could not load geometry for bank {b_id}: {e}")

                # Overwrite target_banks so the rest of the script is perfectly aligned
                target_banks = valid_target_banks

                detector_params = {
                    "centers": centers,
                    "uhats": uhats,
                    "vhats": vhats,
                    "m": m,
                    "n": n,
                    "pw": pw,
                    "ph": ph,
                    "modes": detector_modes,
                    "radial_bound": detector_radial_bound_frac,
                    "area_bound": detector_area_bound_frac,
                    "global_rot_bound_deg": detector_global_rot_bound_deg,
                    "global_rot_axis": np.array(detector_global_rot_axis),
                    "cylinder_axis": np.array(cylinder_axis),
                    "global_trans_bound_meters": detector_global_trans_bound_meters,
                }

            xyz_out = np.zeros((len(pixel_r), 3))
            tt_out = np.zeros(len(pixel_r))
            az_out = np.zeros(len(pixel_r))

            u_offsets = np.zeros(len(pixel_r))
            v_offsets = np.zeros(len(pixel_r))
            bank_indices = np.zeros(len(pixel_r), dtype=np.int32)

            for phys_bank in np.unique(bank_array):
                mask = bank_array == phys_bank
                if not np.any(mask):
                    continue

                try:
                    det_config = beamlines[instrument_name][str(int(phys_bank))]
                    det = Detector(det_config)

                    xyz_p = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])
                    xyz_out[mask] = xyz_p

                    tt_out[mask], az_out[mask] = det.pixel_to_angles(
                        pixel_r[mask], pixel_c[mask], ki_vec=ki_vec_val
                    )

                    if refine_detector and int(phys_bank) in bank_to_idx:
                        bank_indices[mask] = bank_to_idx[int(phys_bank)]
                        u_offsets[mask] = np.dot(xyz_p - det.center, det.uhat)
                        v_offsets[mask] = np.dot(xyz_p - det.center, det.vhat)

                except KeyError as e:
                    print(
                        f"Warning: Could not rebuild geometry for bank {phys_bank}: {e}"
                    )

            input_data["peaks/xyz"] = xyz_out
            input_data["peaks/two_theta"] = tt_out
            input_data["peaks/azimuthal"] = az_out

            if refine_detector:
                peak_pixel_coords = {
                    "u_offsets": u_offsets.tolist(),
                    "v_offsets": v_offsets.tolist(),
                    "bank_indices": bank_indices.tolist(),
                }
        else:
            raise ValueError(
                "ERROR: Input file does not contain peaks/pixel_r and peaks/pixel_c. Cannot perform physically sound indexing."
            )

    if "peaks/image_index" in input_data:
        input_data["peaks/run_index"] = input_data["peaks/image_index"]

    # --- INJECT SECOND PHASE OF BOOTSTRAP PHYSICS ---
    if bootstrap_filename:
        with h5py.File(bootstrap_filename, "r") as b_f:
            if "goniometer/translations" in b_f:
                input_data["goniometer/translations"] = b_f["goniometer/translations"][()]
            if "beam/ki_vec" in b_f:
                ki_vec_val = b_f["beam/ki_vec"][()]
            if "peaks/h" in b_f:
                input_data["peaks/h"] = b_f["peaks/h"][()]
                input_data["peaks/k"] = b_f["peaks/k"][()]
                input_data["peaks/l"] = b_f["peaks/l"][()]
            if "peaks/lambda" in b_f:
                input_data["peaks/lambda"] = b_f["peaks/lambda"][()]

    input_data["sample/a"], input_data["sample/b"], input_data["sample/c"] = a, b, c
    (
        input_data["sample/alpha"],
        input_data["sample/beta"],
        input_data["sample/gamma"],
    ) = alpha, beta, gamma
    input_data["sample/space_group"] = space_group
    input_data["instrument/wavelength"] = [float(wavelength_min), float(wavelength_max)]
    input_data["beam/ki_vec"] = ki_vec_val

    opt = FindUB(data=input_data)
    opt.wavelength = [float(wavelength_min), float(wavelength_max)]

    if bootstrap_filename:
        with h5py.File(bootstrap_filename, "r") as b_f:
            off_data = b_f.get("goniometer/offsets")
            if off_data is not None:
                if isinstance(off_data, h5py.Group):
                    opt.goniometer_offsets = {
                        k: off_data[k][()] for k in off_data.keys()
                    }
                else:
                    opt.goniometer_offsets = off_data[()]

    print(f"Starting evosax optimization with strategy: {strategy_name}")
    print(f"Running {n_runs} run(s)...")
    print(f"Settings per run: Population Size={population_size}, Generations={gens}")
    if freeze_orientation:
        print("ORIENTATION LOCKED: U Matrix will not be refined.")
    if refine_lattice:
        print(f"Refining lattice parameters with {lattice_bound_frac * 100}% bounds.")
    if refine_goniometer_trans:
        num_axes = len(opt.goniometer_axes) if opt.goniometer_axes is not None else 1
        print(f"Refining per-axis goniometer translations ({num_axes} axes) with bounds: {goniometer_trans_bound_meters} m.")
    if refine_beam:
        print(f"Refining beam tilt with {beam_bound_deg}° bounds.")

    goniometer_names = None

    if original_nexus_filename and instrument_name:
        is_merged = False
        with h5py.File(original_nexus_filename, "r") as f_check:
            if "images" in f_check and "goniometer/axes" in f_check:
                is_merged = True
                axes = f_check["goniometer/axes"][()]
                angles = f_check["goniometer/angles"][()]
                names = (
                    [n.decode("utf-8") for n in f_check["goniometer/names"][()]]
                    if "goniometer/names" in f_check
                    else None
                )

        if not is_merged:
            # This forces the pipeline to read the UPDATED reduction_settings.json
            axes, angles, names = get_rotation_data_from_nexus(
                original_nexus_filename, instrument_name
            )

        if len(axes) > 0:
            opt.goniometer_axes = np.array(axes)
            angles = np.array(angles)
            if angles.ndim == 1:
                angles = angles.reshape(-1, 1)

            opt.goniometer_angles = angles
            goniometer_names = names
            if names is not None:
                opt.goniometer_names = names

            # 1. Overwrite the stale baked axes/angles from finder.h5
            input_data["goniometer/axes"] = opt.goniometer_axes
            input_data["goniometer/angles"] = opt.goniometer_angles
            if names is not None:
                input_data["goniometer/names"] = [
                    n.encode("utf-8") if isinstance(n, str) else n for n in names
                ]

            # This forces JAX VectorizedObjective to build the R matrix dynamically from the new JSON axes.
            opt.R = None
            if "goniometer/R" in input_data:
                del input_data["goniometer/R"]

            # --- RUN/PEAK MAPPING LOGIC ---
            if opt.run_indices is not None:
                max_run_id = int(np.max(opt.run_indices))
                num_peaks = len(opt.run_indices)
                num_axes = len(opt.goniometer_axes)

                # 1. Safely orient angles to (num_axes, N) without blindly transposing square matrices
                if angles.ndim == 2:
                    if angles.shape[0] != num_axes and angles.shape[1] == num_axes:
                        angles = angles.T
                elif angles.ndim == 1:
                    angles = angles.reshape(num_axes, 1)

                num_angles_provided = angles.shape[1]

                # 2. Check for single-frame tiled data from run_reduce
                # If all columns are identical, collapse it back to a single angle.
                if num_angles_provided > 1:
                    if np.allclose(angles, angles[:, 0:1], atol=1e-7):
                        angles = angles[:, 0:1]
                        num_angles_provided = 1

                # 3. Safely map angles to cover the highest requested physical index
                if num_angles_provided == 1:
                    # Single frame: safe to broadcast to cover any max_run_id (e.g., Bank 105)
                    opt.goniometer_angles = np.tile(angles, (1, max_run_id + 1))
                elif num_angles_provided > max_run_id:
                    # Multi-frame: We have enough explicit angles to cover the highest index
                    opt.goniometer_angles = angles
                elif num_angles_provided == num_peaks:
                    # Angles provided explicitly per peak
                    opt.goniometer_angles = angles
                else:
                    # Multi-frame mismatch: run_indices contains physical bank IDs (e.g. 105)
                    # but angles only contains contiguous steps (e.g. 52).
                    print(
                        f"WARNING: Angle shape {angles.shape} does not cover max run index {max_run_id}."
                    )
                    print(
                        "Padding goniometer angles to prevent out-of-bounds lookup..."
                    )
                    padded_angles = np.zeros((num_axes, max_run_id + 1))
                    padded_angles[:, :num_angles_provided] = angles
                    for i in range(num_angles_provided, max_run_id + 1):
                        padded_angles[:, i] = angles[:, -1]
                    opt.goniometer_angles = padded_angles
            else:
                num_peaks = len(opt.two_theta) if opt.two_theta is not None else 1
                num_axes = len(opt.goniometer_axes)

                if angles.ndim == 2 and angles.shape[1] == num_axes:
                    angles = angles.T

                num_angles_provided = (
                    angles.shape[1] if angles.ndim == 2 else len(angles)
                )

                if num_angles_provided == num_peaks:
                    opt.goniometer_angles = angles
                elif num_angles_provided == 1:
                    opt.goniometer_angles = np.tile(angles, (1, num_peaks))
                else:
                    raise ValueError(
                        f"CRITICAL: Angle shape {angles.shape} cannot map to {num_peaks} peaks."
                    )

    # Apply the console messages appropriately
    if refine_goniometer:
        print(
            f"Refining goniometer angles from fresh JSON/Nexus with {goniometer_bound_deg} deg bounds."
        )
    elif opt.goniometer_axes is not None:
        print("Using fresh kinematics from JSON (no refinement).")
    else:
        print("WARNING: No goniometer data found.")

    init_params = None
    if bootstrap_filename:
        init_params = opt.get_bootstrap_params(
            bootstrap_filename=bootstrap_filename,
            refine_lattice=refine_lattice,
            lattice_bound_frac=lattice_bound_frac,
            refine_sample=refine_goniometer_trans,
            sample_bound_meters=goniometer_trans_bound_meters,
            refine_beam=refine_beam,
            beam_bound_deg=beam_bound_deg,
            refine_goniometer=refine_goniometer,
            goniometer_bound_deg=goniometer_bound_deg,
            refine_goniometer_axes=refine_goniometer_axes,
            freeze_orientation=freeze_orientation,
        )

    num, hkl_uvw, lambda_S, U = opt.minimize(
        strategy_name=strategy_name,
        population_size=population_size,
        num_generations=gens,
        n_runs=n_runs,
        sigma_init=sigma_init,
        seed=seed,
        init_params=init_params,
        goniometer_bound_deg=goniometer_bound_deg,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        refine_goniometer=refine_goniometer,
        refine_goniometer_axes=refine_goniometer_axes,
        goniometer_names=goniometer_names,
        refine_sample=refine_goniometer_trans,
        goniometer_trans_bound_meters=goniometer_trans_bound_meters,
        refine_beam=refine_beam,
        beam_bound_deg=beam_bound_deg,
        batch_size=batch_size,
        refine_detector=refine_detector,
        detector_params=detector_params,
        peak_pixel_coords=peak_pixel_coords,
        detector_trans_bound_meters=detector_trans_bound_meters,
        detector_rot_bound_deg=detector_rot_bound_deg,
        freeze_orientation=freeze_orientation,
        num_candidates=num_candidates,
        no_index=no_index,
    )

    print(f"\nOptimization complete. Best solution indexed {num} peaks.")
    opt.reciprocal_lattice_B()

    copy_keys = [
        "sample/space_group",
        "instrument/wavelength",
        "peaks/intensity",
        "peaks/sigma",
        "peaks/radius",
        "goniometer/R",
        "goniometer/axes",
        "goniometer/angles",
        "goniometer/names",
        "files",
        "file_offsets",
        "peaks/run_index",
        "peaks/image_index",
        "bank",
        "goniometer/translations",
        "beam/ki_vec",
        "peaks/pixel_r",
        "peaks/pixel_c",
    ]

    copied_data = {}
    for key in copy_keys:
        if key in input_data:
            copied_data[key] = input_data[key]

    print(f"Saving indexed peaks to {output_peaks_filename}...")
    with h5py.File(output_peaks_filename, "w") as f:
        if instrument_name:
            f.attrs["instrument"] = instrument_name
        elif "instrument" in input_data:
            f.attrs["instrument"] = input_data["instrument"]

        for key, value in copied_data.items():
            f[key] = value

        def safe_write(grp, name, data):
            if name in grp:
                del grp[name]
            grp[name] = data

        safe_write(f, "goniometer/R", opt.R)

        if opt.goniometer_offsets is not None:
            grp_name = "goniometer/offsets"
            if grp_name in f:
                del f[grp_name]

            if isinstance(opt.goniometer_offsets, dict):
                grp = f.create_group(grp_name)
                for k, v in opt.goniometer_offsets.items():
                    grp[k] = v
            else:
                f[grp_name] = opt.goniometer_offsets

        safe_write(f, "sample/a", opt.a)
        safe_write(f, "sample/b", opt.b)
        safe_write(f, "sample/c", opt.c)
        safe_write(f, "sample/alpha", opt.alpha)
        safe_write(f, "sample/beta", opt.beta)
        safe_write(f, "sample/gamma", opt.gamma)
        safe_write(f, "beam/ki_vec", opt.ki_vec)
        safe_write(f, "goniometer/translations", opt.sample_offset)

        B_mat = opt.reciprocal_lattice_B()
        safe_write(f, "sample/B", B_mat)
        f["sample/U"] = U

        if opt.run_indices is not None:
            safe_write(f, "peaks/run_index", opt.run_indices)

        f["peaks/h"] = hkl_uvw[:, 0]
        f["peaks/k"] = hkl_uvw[:, 1]
        f["peaks/l"] = hkl_uvw[:, 2]
        f["peaks/lambda"] = lambda_S

        if opt.x is not None and opt.x.size > 0:
            f["optimization/best_params"] = opt.x

        if bootstrap_filename:
            with h5py.File(bootstrap_filename, "r") as b_f:
                if "detector_calibration" in b_f:
                    b_f.copy("detector_calibration", f)

        import json

        flags = {
            "no_index": opt.no_index,
            "refine_lattice": refine_lattice,
            "refine_goniometer": refine_goniometer,
            "refine_goniometer_trans": refine_goniometer_trans,
            "refine_beam": refine_beam,
            "refine_detector": refine_detector,
            "detector_modes": detector_modes,
            "freeze_orientation": freeze_orientation,
        }
        f.create_dataset("optimization/flags", data=json.dumps(flags).encode("utf-8"))

        if refine_detector and hasattr(opt, "calibrated_centers"):
            for b_idx, b_id in enumerate(target_banks):
                grp_name = f"detector_calibration/bank_{b_id}"

                # Cleanly overwrite the specific bank if it was copied from bootstrap
                if grp_name in f:
                    del f[grp_name]

                f.create_group(grp_name)
                f[f"{grp_name}/center"] = opt.calibrated_centers[b_idx]
                f[f"{grp_name}/uhat"] = opt.calibrated_uhats[b_idx]
                f[f"{grp_name}/vhat"] = opt.calibrated_vhats[b_idx]

                if hasattr(opt, "calibrated_widths"):
                    f[f"{grp_name}/width"] = opt.calibrated_widths[b_idx]
                    f[f"{grp_name}/height"] = opt.calibrated_heights[b_idx]

    print("Done.")


def run_finder(
    filename: str,
    instrument: str,
    output_filename: str = "output.h5",
    finder_algorithm: str = "peak_local_max",
    show_progress: bool = True,
    create_visualizations: bool = False,
    show_steps: bool = False,
    peak_local_max_min_pixel_distance: int = -1,
    peak_local_max_min_relative_intensity: float = -1,
    peak_local_max_normalization: bool = False,
    mask_file: str | None = None,
    mask_rel_erosion_radius: float | None = None,
    thresholding_noise_cutoff_quantile: float = 0.8,
    thresholding_min_peak_dist_pixels: float = 8.0,
    thresholding_blur_kernel_sigma: int = 5,
    thresholding_open_kernel_size_pixels: int = 3,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_sigma: float | None = None,
    region_growth_minimum_intensity: float = 4500.0,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 30,
    peak_minimum_signal_to_noise: float = 1.0,
    peak_pixel_outlier_threshold: float = 2.0,
    sparse_rbf_alpha: float = 0.1,
    sparse_rbf_gamma: float = 1.0,
    sparse_rbf_min_sigma: float = 1.5,
    sparse_rbf_max_sigma: float = 10.0,
    sparse_rbf_chunk_size: int = 512,
    sparse_rbf_tile_rows: int = 2,
    sparse_rbf_tile_cols: int = 2,
    sparse_rbf_loss: str = "poisson",
    sparse_rbf_auto_tune_alpha: bool = False,
    sparse_rbf_candidate_alphas = "500,1000,2500,5000,10000",
    max_workers: int = 16,
):
    print(f"Creating peaks from {filename} for instrument {instrument}")

    wavelength_kwargs = {}
    if wavelength_min:
        wavelength_kwargs["wavelength_min"] = wavelength_min
    if wavelength_max:
        wavelength_kwargs["wavelength_max"] = wavelength_max

    peaks = Peaks(filename, instrument, **wavelength_kwargs)

    peak_kwargs = {"algorithm": finder_algorithm}
    if finder_algorithm == "peak_local_max":
        if peak_local_max_min_pixel_distance > 0:
            peak_kwargs["min_pix"] = peak_local_max_min_pixel_distance
        if peak_local_max_min_relative_intensity > 0:
            peak_kwargs["min_rel_intensity"] = peak_local_max_min_relative_intensity
        peak_kwargs["normalize"] = peak_local_max_normalization
    elif finder_algorithm == "thresholding":
        peak_kwargs.update(
            {
                "noise_cutoff_quantile": thresholding_noise_cutoff_quantile,
                "min_peak_dist_pixels": thresholding_min_peak_dist_pixels,
                "blur_kernel_sigma": thresholding_blur_kernel_sigma,
                "open_kernel_size_pixels": thresholding_open_kernel_size_pixels,
                "show_steps": show_steps,
                "show_scale": "log",
            }
        )
    elif finder_algorithm == "sparse_rbf":
        # Because we separated Typer from core logic, this split is 100% safe
        alpha_list = [float(k.strip()) for k in sparse_rbf_candidate_alphas.split(",")]

        peak_kwargs.update(
            {
                "alpha": sparse_rbf_alpha,
                "gamma": sparse_rbf_gamma,
                "min_sigma": sparse_rbf_min_sigma,
                "max_sigma": sparse_rbf_max_sigma,
                "chunk_size": sparse_rbf_chunk_size,
                "show_steps": show_steps,
                "show_scale": "linear",
                "tiles": (sparse_rbf_tile_rows, sparse_rbf_tile_cols),
                "loss": sparse_rbf_loss,
                "auto_tune_alpha": sparse_rbf_auto_tune_alpha,
                "candidate_alphas": alpha_list,
            }
        )
    else:
        raise ValueError("Invalid finder algorithm")

    peak_kwargs.update(
        {
            "mask_file": mask_file,
            "mask_rel_erosion_radius": mask_rel_erosion_radius,
        }
    )

    integration_params = {
        "region_growth_distance_threshold": region_growth_distance_threshold,
        "region_growth_minimum_sigma": region_growth_minimum_sigma,
        "region_growth_minimum_intensity": region_growth_minimum_intensity,
        "region_growth_maximum_pixel_radius": region_growth_maximum_pixel_radius,
        "peak_center_box_size": peak_center_box_size,
        "peak_smoothing_window_size": peak_smoothing_window_size,
        "peak_minimum_pixels": peak_minimum_pixels,
        "peak_minimum_signal_to_noise": peak_minimum_signal_to_noise,
        "peak_pixel_outlier_threshold": peak_pixel_outlier_threshold,
    }

    detector_peaks = peaks.get_detector_peaks(
        peak_kwargs,
        integration_params,
        visualize=create_visualizations,
        show_progress=show_progress,
        file_prefix=filename,
        max_workers=max_workers,
    )

    peaks.write_hdf5(
        output_filename=output_filename,
        detector_peaks=detector_peaks,
        instrument_wavelength=[peaks.wavelength.min, peaks.wavelength.max],
    )

    # copy over cell params
    copy_keys = [
        "sample/a",
        "sample/b",
        "sample/c",
        "sample/alpha",
        "sample/beta",
        "sample/gamma",
        "sample/space_group",
    ]

    with h5py.File(output_filename, "a") as f:
        with h5py.File(filename, "r") as f_in:
            for key in copy_keys:
                if key in f_in:
                    f_in.copy(f_in[key], f, key)


def run_metrics(
    file1: str,
    file2: str | None = None,
    instrument: str | None = None,
    d_min: float | None = None,
    per_run: bool = False,
    ki_vec: List[float] | np.ndarray = None,
):
    from subhkl.instrument.metrics import compute_metrics

    # No need to call apply_detector_calibration here because metrics.py
    # dynamically shifts coordinates using the detector_calibration group.
    result = compute_metrics(
        file1=file1,
        file2=file2,
        instrument=instrument,
        d_min=d_min,
        per_run=per_run,
        ki_vec_override=ki_vec,
    )

    if "error_message" in result:
        print(result["error_message"])
        if result["error_message"].startswith("Exception"):
            print("METRICS: 9.99 9.99 9.99 9.99 9.99 9.99")
        return

    if "filter_message" in result:
        print(f"METRICS: {result['filter_message']}")

    # Print main metrics
    print(
        f"METRICS: {result['median_d_err']:.5f} {result['mean_d_err']:.5f} {result['max_d_err']:.5f} "
        f"{result['median_ang_err']:.5f} {result['mean_ang_err']:.5f} {result['max_ang_err']:.5f}"
    )

    # Print per-run metrics if requested
    if per_run and "per_run_errors" in result:
        print("\nPER-RUN MEDIAN ANGULAR ERROR (deg) - Sorted by error:")
        for r, err, count in result["per_run_errors"]:
            status = "BAD" if err > 1.0 else "OK"
            print(f"  Run {r:4d}: {err:6.3f} ({count:4d} peaks) [{status}]")


def run_peak_predictor(
    filename: str,
    instrument: str,
    indexed_hdf5_filename: str,
    integration_peaks_filename: str,
    d_min: float = 1.0,
    create_visualizations: bool = False,
    space_group: str | None = None,
    wavel_min: float | None = None,
    wavel_max: float | None = None,
    max_workers: int = 16,
):
    apply_detector_calibration(indexed_hdf5_filename, instrument)

    with h5py.File(indexed_hdf5_filename, "r") as f_idx:
        a = float(f_idx["sample/a"][()])
        b = float(f_idx["sample/b"][()])
        c = float(f_idx["sample/c"][()])
        alpha = float(f_idx["sample/alpha"][()])
        beta = float(f_idx["sample/beta"][()])
        gamma = float(f_idx["sample/gamma"][()])

        if space_group is None:
            space_group = f_idx["sample/space_group"][()].decode("utf-8")

        wavelength = f_idx["instrument/wavelength"][()]
        if wavel_min: wavelength[0] = wavel_min
        if wavel_max: wavelength[1] = wavel_max

        U = f_idx["sample/U"][()]
        B = f_idx["sample/B"][()]

        gonio_offsets = None
        off_data = f_idx.get("goniometer/offsets")
        gonio_names = f_idx["goniometer/names"][()] if "goniometer/names" in f_idx else None

        if gonio_names is not None:
            gonio_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in gonio_names]

        if off_data is not None:
            gonio_offsets = np.zeros(len(gonio_names) if gonio_names else 1, dtype=np.float32)
            if isinstance(off_data, h5py.Group) and gonio_names is not None:
                for i, name in enumerate(gonio_names):
                    if name in off_data:
                        gonio_offsets[i] = float(off_data[name][()])
            else:
                raw_offs = off_data[()]
                gonio_offsets[:len(raw_offs)] = raw_offs

        if "goniometer/translations" in f_idx:
            sample_offset = f_idx["goniometer/translations"][()]
        else:
            sample_offset = np.zeros(3)

        gonio_axes = f_idx["goniometer/axes"][()] if "goniometer/axes" in f_idx else None
        ki_vec = f_idx["beam/ki_vec"][()] if "beam/ki_vec" in f_idx else np.array([0.0, 0.0, 1.0])

    peaks = Peaks(filename, instrument, wavelength_min=wavelength[0], wavelength_max=wavelength[1])
    print(f"Predicting peaks for {len(peaks.image.ims)} images using solution from {indexed_hdf5_filename}")

    if gonio_offsets is not None:
        print(f"Applying refined goniometer offsets from indexer: {gonio_offsets}")

    # Pass the Base UB matrix. The predictor will apply dynamic R_gonio internally!
    UB = U @ B

    results_map = peaks.predict_peaks(
        a, b, c, alpha, beta, gamma, d_min,
        UB=UB,
        space_group=space_group,
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        max_workers=max_workers,

        # --- NEW GROUND TRUTH DELEGATION ---
        R_all=None, # Force dynamic evaluation
        gonio_axes=peaks.goniometer.axes_raw,
        gonio_angles=peaks.goniometer.angles_raw,
        gonio_offsets=gonio_offsets # Pass the pure zero-points
    )

    print(f"Saving predictions to {integration_peaks_filename}")
    with h5py.File(integration_peaks_filename, "w") as f:
        f.attrs["instrument"] = instrument
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = alpha, beta, gamma

        sorted_keys = sorted(peaks.image.ims.keys())
        bank_ids = np.array([peaks.image.bank_mapping.get(k, k) for k in sorted_keys], dtype=np.int32)
        f.create_dataset("bank_ids", data=bank_ids)

        f["sample/space_group"] = space_group
        f["sample/U"], f["sample/B"] = U, B
        f["instrument/wavelength"] = wavelength
        f["beam/ki_vec"] = ki_vec

        # --- SAVE RAW UNCORRECTED METADATA ---
        f["goniometer/angles"] = peaks.goniometer.angles_raw
        f["goniometer/axes"] = peaks.goniometer.axes_raw
        f["goniometer/translations"] = sample_offset
        if gonio_offsets is not None:
            f["goniometer/offsets"] = gonio_offsets

        if peaks.goniometer.names_raw:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("goniometer/names", data=peaks.goniometer.names_raw, dtype=dt)

        for img_key, (i, j, h, k, l, wl) in results_map.items():
            grp = f.create_group(f"banks/{img_key}")
            grp.create_dataset("i", data=i), grp.create_dataset("j", data=j)
            grp.create_dataset("h", data=h), grp.create_dataset("k", data=k), grp.create_dataset("l", data=l)
            grp.create_dataset("wavelength", data=wl)

        with h5py.File(indexed_hdf5_filename, "r") as f_in:
            if "detector_calibration" in f_in:
                f_in.copy("detector_calibration", f)

def run_rbf_integrator(
    filename: str,
    instrument: str,
    integration_peaks_filename: str,
    output_filename: str,
    alpha: float = 1.0,
    gamma: float = 1.0,
    sigmas: str = "1.0,2.0,4.0",
    nominal_sigma: float = 1.0,
    anisotropic: bool = False,
    fit_mosaicity: bool = False,
    rel_border_width: float = 0.0,
    show_progress: bool = True,
    create_visualizations: bool = False,
    chunk_size: int = 256,
    max_workers: int | None = None,
):
    apply_detector_calibration(integration_peaks_filename, instrument)

    import h5py
    from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

    sigma_list = [float(k.strip()) for k in sigmas.split(",")]
    print(f"Starting Dense Sparse RBF Integration on {filename}")
    print(f"Parameters: Alpha={alpha}, Gamma={gamma}, Sigma={sigma_list}")

    peak_dict = {}

    with h5py.File(integration_peaks_filename, "r") as f:
        angles_stack = f["goniometer/angles"][()] if "goniometer/angles" in f else None
        sample_offset = f["goniometer/translations"][()] if "goniometer/translations" in f else np.zeros(3)
        gonio_axes = f["goniometer/axes"][()] if "goniometer/axes" in f else None

        gonio_offsets = f["goniometer/offsets"][()] if "goniometer/offsets" in f else None

        for key in f["banks"].keys():
            img_idx = int(key)
            grp = f[f"banks/{key}"]
            peak_dict[img_idx] = [
                grp["i"][()], grp["j"][()], grp["h"][()],
                grp["k"][()], grp["l"][()], grp["wavelength"][()]
            ]

    peaks = Peaks(filename, instrument)

    if angles_stack is None:
        angles_stack = peaks.goniometer.angles_raw

    one_image = next(iter(peaks.image.ims.values()))
    border_width = int(rel_border_width * min(one_image.shape[0], one_image.shape[1]))

    result = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict,
        peaks_obj=peaks,
        alpha=alpha,
        sigmas=sigma_list,
        gamma=gamma,
        nominal_sigma=nominal_sigma,
        show_progress=show_progress,
        all_R=None,
        sample_offset=sample_offset,
        anisotropic=anisotropic,
        fit_mosaicity=fit_mosaicity,
        border_width=border_width,
        chunk_size=chunk_size,
        create_visualizations=create_visualizations,
        file_prefix=filename,
        max_workers=max_workers,
        gonio_axes=gonio_axes,
        gonio_angles=angles_stack,
        gonio_offsets=gonio_offsets
    )

    print(f"Saving RBF integrated peaks to {output_filename}")
    with h5py.File(output_filename, "w") as f:
        f["peaks/h"], f["peaks/k"], f["peaks/l"] = result.h, result.k, result.l
        f["peaks/lambda"] = result.wavelength
        f["peaks/intensity"], f["peaks/sigma"] = result.intensity, result.sigma
        f["peaks/two_theta"], f["peaks/azimuthal"] = result.tt, result.az
        f["peaks/bank"], f["peaks/run_index"] = result.bank, result.run_id

        # Copy metadata
        copy_keys = [
            "sample/a", "sample/b", "sample/c",
            "sample/alpha", "sample/beta", "sample/gamma",
            "sample/space_group", "sample/U", "sample/B",
            "goniometer/translations", "goniometer/offsets", # <-- INCLUDED
            "beam/ki_vec", "instrument/wavelength",
        ]

        with h5py.File(integration_peaks_filename, "r") as f_in:
            for key in copy_keys:
                if key in f_in: f_in.copy(f_in[key], f, key)
            for k in ["goniometer/axes", "goniometer/names"]:
                if k in f_in: f_in.copy(f_in[k], f, k)

def run_integrator(
    filename: str,
    instrument: str,
    integration_peaks_filename: str,
    output_filename: str,
    integration_method: str = "free_fit",
    integration_mask_file: str | None = None,
    integration_mask_rel_erosion_radius: float | None = 0.05,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 50.0,
    region_growth_minimum_sigma: float | None = None,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 10,
    peak_minimum_signal_to_noise: float = 1.0,
    peak_pixel_outlier_threshold: float = 2.0,
    create_visualizations: bool = False,
    show_progress: bool = True,
    found_peaks_file: str | None = None,
    max_workers: int = 16,
):
    apply_detector_calibration(integration_peaks_filename, instrument)

    peak_dict = {}
    angles_stack = None
    all_R = None
    with h5py.File(integration_peaks_filename, "r") as f:
        U = f["sample/U"][()] if "sample/U" in f else None
        B = f["sample/B"][()] if "sample/B" in f else None
        all_R = f["goniometer/R"][()] if "goniometer/R" in f else None
        angles_stack = f["goniometer/angles"][()] if "goniometer/angles" in f else None
        sample_offset = f["goniometer/translations"][()] if "goniometer/translations" in f else np.zeros(3)
        ki_vec = (
            f["beam/ki_vec"][()] if "beam/ki_vec" in f else np.array([0.0, 0.0, 1.0])
        )

        for key in f["banks"].keys():
            img_idx = int(key)
            grp = f[f"banks/{key}"]
            peak_dict[img_idx] = [
                grp["i"][()],
                grp["j"][()],
                grp["h"][()],
                grp["k"][()],
                grp["l"][()],
                grp["wavelength"][()],
            ]

    peaks = Peaks(filename, instrument)

    integration_params = {
        "region_growth_distance_threshold": region_growth_distance_threshold,
        "region_growth_minimum_intensity": region_growth_minimum_intensity,
        "region_growth_minimum_sigma": region_growth_minimum_sigma,
        "region_growth_maximum_pixel_radius": region_growth_maximum_pixel_radius,
        "peak_center_box_size": peak_center_box_size,
        "peak_smoothing_window_size": peak_smoothing_window_size,
        "peak_minimum_pixels": peak_minimum_pixels,
        "peak_minimum_signal_to_noise": peak_minimum_signal_to_noise,
        "peak_pixel_outlier_threshold": peak_pixel_outlier_threshold,
        "integration_mask_file": integration_mask_file,
        "integration_mask_rel_erosion_radius": integration_mask_rel_erosion_radius,
    }

    if all_R is None:
        print("Warning: Refined R stack not found in prediction file. Using nominal.")
        all_R = peaks.goniometer.rotation

    if angles_stack is None:
        angles_stack = peaks.goniometer.angles_raw

    UB = U @ B if U is not None and B is not None else None
    RUB = None
    if UB is not None:
        RUB = np.matmul(all_R, UB) if all_R.ndim == 3 else all_R @ UB

    result = peaks.integrate(
        peak_dict,
        integration_params,
        RUB=RUB,
        R_stack=all_R,
        angles_stack=angles_stack,
        gonio_axes=gonio_axes,
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        create_visualizations=create_visualizations,
        show_progress=show_progress,
        integration_method=integration_method,
        file_prefix=filename,
        found_peaks_file=found_peaks_file,
        max_workers=max_workers,
    )

    print(f"Saving integrated peaks to {output_filename}")

    copy_keys = [
        "sample/a",
        "sample/b",
        "sample/c",
        "sample/alpha",
        "sample/beta",
        "sample/gamma",
        "sample/space_group",
        "sample/U",
        "sample/B",
        "goniometer/translations",
        "beam/ki_vec",
        "instrument/wavelength",
    ]

    with h5py.File(output_filename, "w") as f:
        f["peaks/h"], f["peaks/k"], f["peaks/l"] = result.h, result.k, result.l
        f["peaks/lambda"] = result.wavelength
        f["peaks/intensity"], f["peaks/sigma"] = result.intensity, result.sigma
        f["peaks/two_theta"], f["peaks/azimuthal"] = result.tt, result.az
        f["peaks/bank"] = result.bank
        f["peaks/run_index"] = result.run_id
        f["peaks/xyz"] = result.xyz

        if result.R and any(r is not None for r in result.R):
            f["goniometer/R"] = np.array(result.R)
        if result.angles and any(a is not None for a in result.angles):
            f["goniometer/angles"] = np.array(result.angles)

        with h5py.File(integration_peaks_filename, "r") as f_in:
            for key in copy_keys:
                if key in f_in:
                    f_in.copy(f_in[key], f, key)
            for k in ["goniometer/axes", "goniometer/names"]:
                if k in f_in:
                    f_in.copy(f_in[k], f, k)


def run_mtz_exporter(
    indexed_h5_filename: str, output_mtz_filename: str, space_group: str = None
):
    algorithm = MTZExporter(indexed_h5_filename, space_group)
    algorithm.write_mtz(output_mtz_filename)


def run_reduce(
    nexus_filename: str,
    output_filename: str,
    instrument: str,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
):
    print(f"Reducing {nexus_filename} -> {output_filename}")
    peaks_handler = Peaks(
        nexus_filename,
        instrument,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )

    if not peaks_handler.image.ims:
        print("Warning: No images found in file.")
        return

    sorted_banks = sorted(peaks_handler.image.ims.keys())
    image_stack = np.stack([peaks_handler.image.ims[b] for b in sorted_banks])
    bank_ids = np.array(sorted_banks, dtype=np.int32)
    n_images = len(sorted_banks)

    if peaks_handler.goniometer.angles_raw is not None:
        angles_repeated = np.tile(peaks_handler.goniometer.angles_raw, (n_images, 1))
    else:
        angles_repeated = np.zeros((n_images, 3))

    axes = (
        np.array(peaks_handler.goniometer.axes_raw)
        if peaks_handler.goniometer.axes_raw is not None
        else np.array([0.0, 1.0, 0.0])
    )

    with h5py.File(output_filename, "w") as f:
        f.create_dataset("images", data=image_stack, compression="lzf")
        f.create_dataset("bank_ids", data=bank_ids)
        f.create_dataset("goniometer/angles", data=angles_repeated)
        f.create_dataset("goniometer/axes", data=axes)

        if peaks_handler.goniometer.names_raw:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "goniometer/names", data=peaks_handler.goniometer.names_raw, dtype=dt
            )

        f.create_dataset(
            "instrument/wavelength",
            data=[peaks_handler.wavelength.min, peaks_handler.wavelength.max],
        )
        f.attrs["instrument"] = instrument

    print(f"Saved {n_images} banks to {output_filename}")


def run_merge_images(
    input_pattern: str,
    output_filename: str,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    space_group: str,
):
    from subhkl.core.spacegroup import get_space_group_object
    import glob

    try:
        get_space_group_object(space_group)
    except ValueError as e:
        raise ValueError(f"ERROR: Invalid space group '{space_group}': {e}")

    if " " in input_pattern:
        h5_files = []
        for p in input_pattern.split():
            h5_files.extend(glob.glob(p))
    else:
        h5_files = glob.glob(input_pattern)

    h5_files = sorted(list(set(h5_files)))

    if not h5_files:
        raise ValueError(f"No files found matching: {input_pattern}")

    print(f"Found {len(h5_files)} files. Merging...")
    merger = ImageStackMerger(h5_files)
    merger.merge(output_filename)

    with h5py.File(output_filename, "a") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/space_group"] = space_group.encode("utf-8")

    print(f"Successfully created {output_filename} with unit cell info embedded.")

def _render_run_unrolled_plot(args):
    """Standalone plotting function for generating unrolled plots per run."""
    out_name, peaks, images, detectors, instrument, zone_axes, pred_zone_axes = args

    import matplotlib.pyplot as plt
    from subhkl.viz.detector_assembly import plot_unrolled_detector

    if plt.get_backend().lower() != "agg":
        plt.switch_backend("Agg")

    plot_unrolled_detector(
        peaks,
        images,
        detectors,
        zone_axes=zone_axes,
        predicted_zone_axes=pred_zone_axes,
        out_name=out_name,
        instrument=instrument
    )
    return out_name

class RunPeaks:
    """Mock object to satisfy the unrolled detector visualizer API."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def real_to_complex_sh_matrix(l):
    """ Host-side matrix construction prevents abstract tracing errors and matches the complex phase basis. """
    dim = int(2 * l + 1)
    U = np.zeros((dim, dim), dtype=np.complex64)
    for m in range(-l, l + 1):
        idx_r = m + l
        if m == 0:
            U[idx_r, l] = 1.0
        elif m > 0:
            U[idx_r, l + m] = (-1.0)**m / np.sqrt(2.0)
            U[idx_r, l - m] = 1.0 / np.sqrt(2.0)
        else:
            U[idx_r, l + abs(m)] = -1j * (-1.0)**abs(m) / np.sqrt(2.0)
            U[idx_r, l - abs(m)] = 1j / np.sqrt(2.0)
    return jnp.array(U)

import jax
@jax.jit(static_argnames=['L_max'])
def compute_legendre_window_weights(q_mags, wl_min, wl_max, L_max):
    """ Computes Legendre polynomial window parameters over reciprocal shell intervals. """
    M = q_mags.shape[0]
    x_max = jnp.clip(-0.5 * q_mags * wl_min, -1.0, 1.0)
    x_min = jnp.clip(-0.5 * q_mags * wl_max, -1.0, 1.0)

    P_min = [jnp.ones(M), x_min]
    P_max = [jnp.ones(M), x_max]

    for l in range(1, L_max + 1):
        P_next_min = ((2 * l + 1) * x_min * P_min[-1] - l * P_min[-2]) / (l + 1)
        P_next_max = ((2 * l + 1) * x_max * P_max[-1] - l * P_max[-2]) / (l + 1)
        P_min.append(P_next_min)
        P_max.append(P_next_max)

    w_list = [0.5 * (x_max - x_min)]
    for l in range(1, L_max + 1):
        int_max = (P_max[l+1] - P_max[l-1]) / (2 * l + 1)
        int_min = (P_min[l+1] - P_min[l-1]) / (2 * l + 1)
        w_l = 0.5 * (2 * l + 1) * (int_max - int_min)
        w_list.append(w_l)

    return jnp.stack(w_list, axis=0)

import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import scipy.special
import jax.scipy.linalg
import e3x
import gemmi
from functools import partial

def skew_symmetric(v):
    """ Computes the 3x3 skew-symmetric matrix cross-product operator. """
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def vector_to_rotation_matrix(omega):
    """ Maps a 3D Lie algebra tangent vector to a proper SO(3) matrix via Rodrigues' formula. """
    theta = jnp.linalg.norm(omega)
    def small_theta():
        return jnp.eye(3) + skew_symmetric(omega)
    def large_theta():
        k = omega / theta
        K = skew_symmetric(k)
        return jnp.eye(3) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * jnp.matmul(K, K)
    return jax.lax.cond(theta < 1e-5, small_theta, large_theta)


@partial(jax.jit, static_argnames=['L_max'])
def precompute_shell_structures(U_base, Y_theo_cryst_jax, w_l_j, ki_batch, L_max, I_weights_jax):
    """ FORWARD MATRIX REDUCTION: Evaluates global structural metrics using physical structure factors. """
    D_full_prev = e3x.so3.rotations.wigner_d(U_base, max_degree=L_max, cartesian_order=False)
    Y_beam = e3x.so3.irreps.spherical_harmonics(ki_batch[0], max_degree=L_max, cartesian_order=False, normalization='orthonormal')

    rotated_beam = jnp.matmul(D_full_prev.T, Y_beam)

    ewald_window = jnp.zeros(Y_theo_cryst_jax.shape[0])
    for l in range(L_max + 1):
        dim_l = 2 * l + 1
        start = l**2
        end = (l+1)**2
        Y_cryst_l = Y_theo_cryst_jax[:, start:end]
        p_l = jnp.matmul(Y_cryst_l, rotated_beam[start:end])
        ewald_window += (w_l_j[l] / float(dim_l)) * p_l

    # NATIVE INTENSITY MODULATION: Multiply window by the true structure factors
    ewald_window = jnp.clip(ewald_window, 0.0, 1.0) * I_weights_jax
    total_window_mass = jnp.maximum(jnp.sum(ewald_window), 1.0)

    G_all = jnp.matmul(Y_theo_cryst_jax.T, Y_theo_cryst_jax * ewald_window[:, None]) / total_window_mass
    T_all = jnp.sum(Y_theo_cryst_jax * ewald_window[:, None], axis=0) / total_window_mass

    return G_all, T_all

def predict_all_shells_pure_tangent(omega, U_base, G_all, T_all, L_max):
    """ LOW-DIMENSIONAL FORWARD PASS: Reconstructs predicted vectors cleanly under a 3D tangent. """
    R_perturb = vector_to_rotation_matrix(omega)
    U_curr = jnp.matmul(U_base, R_perturb)
    D_full = e3x.so3.rotations.wigner_d(U_curr, max_degree=L_max, cartesian_order=False)

    preds = []
    for l in range(1, L_max + 1):
        dim_l = 2 * l + 1
        start = l**2
        end = (l+1)**2
        D_l = D_full[start:end, start:end]

        G_norm = G_all[start:end, start:end] * (4.0 * jnp.pi / float(dim_l))

        A_sig = jnp.matmul(D_l, jnp.matmul(G_norm, D_l.T))
        A_dev_pred = A_sig - (jnp.trace(A_sig) / float(dim_l)) * jnp.eye(dim_l)
        preds.append(A_dev_pred.flatten())

    return jnp.concatenate(preds)

@partial(jax.jit, static_argnames=['L_max'])
def kalman_subspace_update(
    P_prev, Y_events_lab, Y_lab_all_sum, G_all, T_all, U_base,
    process_q_scale, dt, ridge_inflation, meas_noise_1st, meas_weight_2nd, num_events, L_max
):
    """ AUTO-DIFF KERNEL: Updates tangent parameters cleanly using exact forward differentiation. """
    omega_state = jnp.zeros(3)
    P_state = P_prev + jnp.eye(3) * (process_q_scale * dt)

    A_lab_all = jnp.matmul(Y_events_lab.T, Y_events_lab) / num_events

    z_data_list = []
    R_diag_list = []
    for l in range(1, L_max + 1):
        dim_l = 2 * l + 1
        start = l**2
        end = (l+1)**2

        A_lab_norm = A_lab_all[start:end, start:end] * (4.0 * jnp.pi / float(dim_l))
        A_lab_dev = A_lab_norm - jnp.eye(dim_l) / float(dim_l)
        sigma_l = (meas_noise_1st * (l * (l + 1) + 1.0)) / (num_events * float(dim_l)) + ridge_inflation

        z_data_list.append(A_lab_dev.flatten())
        R_diag_list.append(jnp.full(dim_l * dim_l, sigma_l * meas_weight_2nd / float(dim_l)))

    z_data = jnp.concatenate(z_data_list)
    R_diag = jnp.concatenate(R_diag_list)

    z_pred = predict_all_shells_pure_tangent(omega_state, U_base, G_all, T_all, L_max)
    H_global = jax.jacfwd(predict_all_shells_pure_tangent)(omega_state, U_base, G_all, T_all, L_max)

    S_mat = H_global @ P_state @ H_global.T + jnp.diag(R_diag) + ridge_inflation * jnp.eye(H_global.shape[0])
    K_gain = P_state @ H_global.T @ jnp.linalg.pinv(S_mat, rcond=1e-4)

    omega_update = K_gain @ (z_data - z_pred)
    P_new = P_state - jnp.matmul(K_gain, jnp.matmul(H_global, P_state))
    P_new = 0.5 * (P_new + P_new.T)

    # FIXED DUALITY: Negative step vector aligns updates properly with the crystal reference frame
    U_new = jnp.matmul(U_base, vector_to_rotation_matrix(-omega_update))

    return U_new, P_new

def process_chunk_field_kalman(
    P_prev, q_batch, ki_batch, t_batch,
    Y_theo_cryst_jax, w_l_j, I_weights_jax, num_peaks,
    meas_noise_1st, meas_weight_2nd, ridge_inflation, gamma_c, L_max, U_base
):
    dt_chunk = jnp.maximum(1e-4, t_batch[-1] - t_batch[0])
    total_rate = q_batch.shape[0] / dt_chunk

    actual_events = q_batch.shape[0]
    num_events = jnp.maximum(float(actual_events), 1.0)

    Y_events_lab = e3x.so3.irreps.spherical_harmonics(q_batch, max_degree=L_max, cartesian_order=False, normalization='orthonormal')
    Y_lab_all_sum = jnp.sum(Y_events_lab, axis=0)

    # 1. Rotate theoretical peaks into the current lab frame
    D_full_base = e3x.so3.rotations.wigner_d(U_base, max_degree=L_max, cartesian_order=False)
    Y_theo_lab = jnp.matmul(Y_theo_cryst_jax, D_full_base.T)
    
    # 2. Dot product between the total data field and each theoretical peak location
    peak_correlations = jnp.matmul(Y_theo_lab, Y_lab_all_sum)
    
    # 3. Rectify to prevent negative weights from SH ringing, add tiny baseline to prevent zeros
    empirical_weights = jnp.maximum(peak_correlations, 0.0) + 0.01

    G_all, T_all = precompute_shell_structures(U_base, Y_theo_cryst_jax, w_l_j, ki_batch, L_max, I_weights_jax)
    Y_events_lab = e3x.so3.irreps.spherical_harmonics(q_batch, max_degree=L_max, cartesian_order=False, normalization='orthonormal')

    process_q_scale = 1e-7

    U_new, P_new = kalman_subspace_update(
        P_prev, Y_events_lab, Y_lab_all_sum, G_all, T_all, U_base,
        process_q_scale, dt_chunk, ridge_inflation, meas_noise_1st, meas_weight_2nd, num_events, L_max
    )

    U_final = U_new if actual_events > 0 else U_base
    P_final = P_new if actual_events > 0 else (P_prev + jnp.eye(3) * (process_q_scale * dt_chunk))

    D_full_new = e3x.so3.rotations.wigner_d(U_final, max_degree=2, cartesian_order=False)
    signal_mass = jnp.sum(jnp.square(D_full_new[1:4, 1:4]))

    intensive_signal_fraction = jnp.clip(signal_mass / 3.0, 0.0, 1.0)
    sig_rate = intensive_signal_fraction * total_rate
    bg_rate = jnp.maximum(total_rate - sig_rate, 0.0)
    omega_eff = 1.6009
    spectral_nll = -jnp.log(jnp.maximum(signal_mass + (1.0 / (4.0 * jnp.pi)), 1e-9))

    return P_final, U_final, spectral_nll, sig_rate, bg_rate, omega_eff


def run_spectral_holonomic_tracker(
    finder_file: str,
    event_batches,
    structure_factors: gemmi.Mtz = None,
    instrument_name: str | None = None,
    streaming_callback=None,
    sigma_q_start: float = 0.15,
    annealing_rate: float = 1.0,
    gamma_step: float = 100.0,
    h_max: int = 6,
    d_min: float = 2.0,
    d_max: float = 8.0,
    bg_ema_weight: float = 0.99,
    loss_weight_ema: float = 0.05,
    wl_min_tracking: float = 0.5,
    wl_max_tracking: float = 12.0,
    J_max: float = None,
    L_max: int = 8,
    gamma_time: float = 1.0,
    gamma_sig: float = 1e-4,
    gamma_c: float = 0.005,
    init_tangent_blur: float = 0.05,
    prior_ridge: float = 0.15,
    meas_noise_1st: float = 0.5,
    meas_weight_2nd: float = 1.0,
    ridge_inflation: float = 1e-4,
):
    from subhkl.optimization import FindUB

    print(f"[0/3] Preparing Monolithic Lie Algebra Tangent Workspace (SO(3) Dimension=3)...")
    print(f"\n[1/3] Initializing Reciprocal Space from: {finder_file}")
    ub_helper = FindUB()
    U_init = None
    with h5py.File(finder_file, "r") as f:
        ub_helper.a = f["sample/a"][()] if "sample/a" in f else 10.0
        ub_helper.b = f["sample/b"][()] if "sample/b" in f else 10.0
        ub_helper.c = f["sample/c"][()] if "sample/c" in f else 10.0
        ub_helper.alpha = f["sample/alpha"][()] if "sample/alpha" in f else 90.0
        ub_helper.beta = f["sample/beta"][()] if "sample/beta" in f else 90.0
        ub_helper.gamma = f["sample/gamma"][()] if "sample/gamma" in f else 90.0
        sg = f["sample/space_group"][()] if "sample/space_group" in f else b"P 1"
        ub_helper.space_group = sg.decode("utf-8") if isinstance(sg, bytes) else str(sg)

        for key in ["sample/U_init", "sample/initial_U", "sample/U_seed", "orientation/U", "sample/U"]:
            if key in f:
                U_init = f[key][()]
                break

    B_mat = ub_helper.reciprocal_lattice_B()
    h_vals = np.arange(-h_max, h_max + 1)
    hc, kc, lc = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
    hkl_c = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
    mask_hkl_c = ~((hkl_c[0] == 0) & (hkl_c[1] == 0) & (hkl_c[2] == 0))
    theo_hkl = hkl_c[:, mask_hkl_c].astype(np.float32)

    q_theo_cryst = np.array(B_mat @ theo_hkl)
    q_mags_np = np.linalg.norm(q_theo_cryst, axis=0)
    res_mask = (q_mags_np < (1.0 / d_min)) & (q_mags_np > (1.0 / d_max))

    q_theo_cryst = q_theo_cryst[:, res_mask]
    q_mags_jax = jnp.array(q_mags_np[res_mask])
    q_theo_sample_jax = jnp.array(q_theo_cryst / np.where(q_mags_np[res_mask] == 0, 1.0, q_mags_np[res_mask]))
    num_peaks = float(q_theo_sample_jax.shape[1])

    Y_theo_cryst_jax = e3x.so3.irreps.spherical_harmonics(q_theo_sample_jax.T, max_degree=L_max, cartesian_order=False, normalization='orthonormal')

    M_peaks = q_mags_np[res_mask].shape[0]
    I_weights = np.ones(M_peaks, dtype=np.float32)

    if structure_factors is not None:
        print("[*] Gemmi Structure Factors provided. Mapping physical intensities to Ewald model...")

        # Locate Intensity (I) or Amplitude (F) columns
        h_col = structure_factors.column_with_label('H') or structure_factors.columns[0]
        k_col = structure_factors.column_with_label('K') or structure_factors.columns[1]
        l_col = structure_factors.column_with_label('L') or structure_factors.columns[2]

        i_col = next((c for c in structure_factors.columns if c.type == 'J' or c.label == 'I'), None)

        # Fallback to Amplitude if Intensity isn't explicitly provided
        if i_col is None:
            i_col = next((c for c in structure_factors.columns if c.type == 'F' or c.label == 'F'), None)

        if i_col is not None:
            h_arr, k_arr, l_arr, i_arr = np.array(h_col.array), np.array(k_col.array), np.array(l_col.array), np.array(i_col.array)

            # Convert Amplitude to Intensity if necessary
            if i_col.type == 'F':
                i_arr = i_arr ** 2

            hkl_to_intensity = {(int(h), int(k), int(l)): val for h, k, l, val in zip(h_arr, k_arr, l_arr, i_arr)}

            valid_theo_hkl = theo_hkl[:, res_mask]
            for idx in range(M_peaks):
                h, k, l = int(valid_theo_hkl[0, idx]), int(valid_theo_hkl[1, idx]), int(valid_theo_hkl[2, idx])

                # Check for exact match or Friedel opposite
                if (h, k, l) in hkl_to_intensity:
                    I_weights[idx] = hkl_to_intensity[(h, k, l)]
                elif (-h, -k, -l) in hkl_to_intensity:
                    I_weights[idx] = hkl_to_intensity[(-h, -k, -l)]
                else:
                    I_weights[idx] = 0.01  # Small floor for unmeasured peaks

            # Normalize to preserve overall scaling
            if np.sum(I_weights) > 0:
                I_weights /= np.mean(I_weights)
        else:
            print("    Warning: Could not find Intensity (type J) or Amplitude (type F) column in MTZ. Defaulting to uniform.")

    I_weights_jax = jax.device_put(I_weights)

    x_max = np.clip(-0.5 * q_mags_np[res_mask] * wl_min_tracking, -1.0, 1.0)
    x_min = np.clip(-0.5 * q_mags_np[res_mask] * wl_max_tracking, -1.0, 1.0)
    P_min = [np.ones(M_peaks), x_min]
    P_max = [np.ones(M_peaks), x_max]

    for l_idx in range(1, L_max + 1):
        P_min.append(((2 * l_idx + 1) * x_min * P_min[-1] - l_idx * P_min[-2]) / (l_idx + 1))
        P_max.append(((2 * l_idx + 1) * x_max * P_max[-1] - l_idx * P_max[-2]) / (l_idx + 1))

    w_l_j_list = [0.5 * (x_max - x_min)]
    for l_idx in range(1, L_max + 1):
        w_l_j_list.append(0.5 * (P_max[l_idx+1] - P_max[l_idx-1] - (P_min[l_idx+1] - P_min[l_idx-1])))
    w_l_j = jnp.array(np.stack(w_l_j_list, axis=0))

    if U_init is not None:
        U_curr = jnp.array(U_init)
    else:
        U_curr = jnp.eye(3)

    P_spectral_full = jnp.eye(3) * prior_ridge

    print(f"\n[2/3] Executing Field Tracker Pipeline (3 Local Subspace Rotational Degrees Active)...")
    tracking_history = []
    ema_bg_rate = 1.0

    for batch_data in event_batches:
        q_batch_np, t_batch_np, banks_np, pr_np, pc_np, angles_np, slab_np, ki_sample_np, cumulative_count = batch_data
        if len(t_batch_np) == 0: continue

        t_state = float(t_batch_np[-1])
        q_batch = jax.device_put(q_batch_np)
        q_batch = q_batch / (jnp.linalg.norm(q_batch, axis=1, keepdims=True) + 1e-9)
        t_batch = jax.device_put(t_batch_np)
        ki_batch = jax.device_put(ki_sample_np)
        ki_batch = ki_batch / (jnp.linalg.norm(ki_batch, axis=1, keepdims=True) + 1e-9)

        P_spectral_full, U_curr, spectral_nll, sig_rate, bg_rate, omega_eff = process_chunk_field_kalman(
            P_spectral_full, q_batch, ki_batch, t_batch,
            Y_theo_cryst_jax, w_l_j, I_weights_jax, num_peaks,
            meas_noise_1st, meas_weight_2nd, ridge_inflation, gamma_c, L_max, U_curr
        )

        ema_bg_rate = bg_ema_weight * ema_bg_rate + (1.0 - bg_ema_weight) * float(bg_rate)
        U_best = np.array(U_curr)
        tracking_history.append((t_state, U_best))

        norm_gap_metric = float(jnp.sum(jnp.square(U_curr)))

        if cumulative_count % 50000 < len(t_batch_np):
            print(f"    Time {t_state:.2f}s | Sig/Bg: {float(sig_rate):.0f}/{float(bg_rate):.0f} Hz | Coherent-Mass: {norm_gap_metric:8.2f} | Active-Solid-Angle: {float(omega_eff):.4f} sr")

        if streaming_callback is not None:
            new_events = {
                "banks": banks_np, "pixel_r": pr_np, "pixel_c": pc_np,
                "angles": angles_np, "s_lab": slab_np
            }

            streaming_callback(
                time=t_state, U_preds=np.expand_dims(U_best, axis=0),
                losses=np.array([float(spectral_nll)]), best_idx=0,
                neutron_count=cumulative_count, new_events=new_events,
                metrics={
                    "loss": float(spectral_nll), "eigengap": norm_gap_metric,
                    "sig_rate": float(sig_rate), "bg_rate": float(bg_rate), "omega_eff": float(omega_eff)
                }
            )

    print(f"\n[3/3] Global Tracking complete. Saving continuous SO(3) state dataset.")
    with h5py.File(finder_file, "a") as f:
        if "tracking" in f: del f["tracking"]
        group = f.create_group("tracking")
        group.create_dataset("final_u_matrix", data=tracking_history[-1][1])
        group.create_dataset("timestamps", data=np.array([h[0] for h in tracking_history]))

    return tracking_history[-1][1]
