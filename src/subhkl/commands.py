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

def run_bingham_tracker(
    finder_file: str,
    event_batches,
    instrument_name: str | None = None,
    streaming_callback=None,
    sigma_q_start: float = 1.0,
    sigma_q_min: float = 0.05,
    annealing_rate: float = 0.5,
    gamma_step: float = 100.0,   # Kinematic Annihilation: Decay per radian of motor movement
    kappa_init: float = 100.0,
    h_max: int = 6,
    n_ensemble: int = 1,
    b_factor: float = 0.0,
    d_min: float = 2.0,
    d_max: float = 8.0,
    bg_ema_weight: float = 0.99,
    loss_ema_weight: float = 0.05,
    wl_min_tracking: float = 0.0,
    wl_max_tracking: float = 12.0,
    max_rate_hz: float = np.inf,
    max_gpu_batch_size: int = 2048,
    L_max = 8,
    gamma_event: float = 1e-5,   # The baseline Epsilon drift (e.g. 1e-5 = memory of 100k events)
    gamma_c = 0.005,             # Critical Diffusion Gain for Self-Organized Criticality
):
    apply_detector_calibration(finder_file, instrument_name)

    from subhkl.optimization import FindUB, get_lattice_system
    import h5py
    import numpy as np
    import jax
    import jax.numpy as jnp
    import scipy.special

    import e3nn_jax as e3nn

    irreps_str = " + ".join([f"{l}e" if l % 2 == 0 else f"{l}o" for l in range(L_max + 1)])
    sh_irreps = e3nn.Irreps(irreps_str)

    # ====================================================================
    # GLOBAL CONTINUOUS VACUUM INITIALIZATION
    # ====================================================================
    num_coeffs = sum(2 * l + 1 for l in range(L_max + 1))
    C_spectral_state = jnp.zeros(num_coeffs)

    laplacian_eigenvalues = []
    for l in range(L_max + 1):
        laplacian_eigenvalues.extend([l * (l + 1)] * (2 * l + 1))
    laplacian_jax = jnp.array(laplacian_eigenvalues)

    @jax.jit
    def evolve_vacuum_sde(C_prev, q_batch, lambda_short_scalar, dt, tau):
        """
        Gridless Spectral-Galerkin update using fully Implicit sinks and diffusion.
        """
        # --- TERM 2: Low-Frequency Ewald Pull (Explicit Observation) ---
        Y_exp_batch = e3nn.spherical_harmonics(sh_irreps, q_batch, normalize=True).array
        
        # ewald_pull * dt is algebraically just the sum of the incoming discrete events
        batch_sum = jnp.sum(Y_exp_batch, axis=0) 
        C_explicit = C_prev + batch_sum

        # --- TERM 1 & 3: Diagonal Heat Decay & KS Scalar Sink (Implicit) ---
        # By evaluating both the sink and the diffusion implicitly (at t + dt),
        # we guarantee unconditional stability regardless of how massive the neutron flux spikes.
        implicit_denom = 1.0 + (lambda_short_scalar * dt) + (tau * laplacian_jax * dt)
        
        C_new = C_explicit / implicit_denom
        
        return C_new

    print(f"\n[1/3] Initializing Reciprocal Space from: {finder_file}")

    ub_helper = FindUB()
    U_init = None
    with h5py.File(finder_file, "r") as f:
        ub_helper.a = f["sample/a"][()] if "sample/a" in f else None
        ub_helper.b = f["sample/b"][()] if "sample/b" in f else None
        ub_helper.c = f["sample/c"][()] if "sample/c" in f else None
        ub_helper.alpha = f["sample/alpha"][()] if "sample/alpha" in f else None
        ub_helper.beta = f["sample/beta"][()] if "sample/beta" in f else None
        ub_helper.gamma = f["sample/gamma"][()] if "sample/gamma" in f else None

        sg = f["sample/space_group"][()]
        ub_helper.space_group = sg.decode("utf-8") if isinstance(sg, bytes) else str(sg)

        if "sample/U" in f:
            U_init = f["sample/U"][()]
            print("  > Discovered seed U matrix in finder file. Will initialize Bingham Prior.")

    B_mat = ub_helper.reciprocal_lattice_B()
    _, _, centering = get_lattice_system(
        ub_helper.a, ub_helper.b, ub_helper.c,
        ub_helper.alpha, ub_helper.beta, ub_helper.gamma, ub_helper.space_group
    )

    if centering == "I": M_prim = jnp.array([[0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5]])
    elif centering == "F": M_prim = jnp.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    elif centering == "C": M_prim = jnp.array([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.0, 1.0]])
    elif centering == "A": M_prim = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, -0.5]])
    elif centering == "B": M_prim = jnp.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, -0.5]])
    elif centering == "R": M_prim = jnp.array([[2/3, 1/3, 1/3], [-1/3, 1/3, 1/3], [-1/3, -2/3, 1/3]])
    else: M_prim = jnp.eye(3)

    print("  > Pre-computing Forward-Mapping HKL Grid for Continuous Tracking...")
    h_vals = np.arange(-h_max, h_max + 1)
    hc, kc, lc = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
    hkl_c = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)

    mask_hkl_c = ~((hkl_c[0] == 0) & (hkl_c[1] == 0) & (hkl_c[2] == 0))
    theo_hkl = hkl_c[:, mask_hkl_c]

    M_prim_np = np.array(M_prim)
    h_p = M_prim_np[0, 0]*theo_hkl[0] + M_prim_np[0, 1]*theo_hkl[1] + M_prim_np[0, 2]*theo_hkl[2]
    k_p = M_prim_np[1, 0]*theo_hkl[0] + M_prim_np[1, 1]*theo_hkl[1] + M_prim_np[1, 2]*theo_hkl[2]
    l_p = M_prim_np[2, 0]*theo_hkl[0] + M_prim_np[2, 1]*theo_hkl[1] + M_prim_np[2, 2]*theo_hkl[2]

    is_valid = (np.abs(h_p - np.round(h_p)) < 1e-4) & \
               (np.abs(k_p - np.round(k_p)) < 1e-4) & \
               (np.abs(l_p - np.round(l_p)) < 1e-4)
    theo_hkl = theo_hkl[:, is_valid].astype(np.float32)

    q_theo_cryst = np.array(B_mat @ theo_hkl)
    q_mags_np = np.linalg.norm(q_theo_cryst, axis=0)

    q_min_tracking = 1.0 / d_max
    q_max_tracking = 1/max(d_min, 1e-6)

    res_mask = (q_mags_np < q_max_tracking) & (q_mags_np > q_min_tracking)
    q_theo_cryst = q_theo_cryst[:, res_mask]
    q_mags_np = q_mags_np[res_mask]

    q_theo_cryst_jax = jnp.array(q_theo_cryst)
    q_mags_jax = jnp.array(q_mags_np)
    q_theo_sample_jax = q_theo_cryst_jax / jnp.where(q_mags_jax == 0, 1.0, q_mags_jax)

    ub_helper.ki_vec = np.array([0.0, 0.0, 1.0])
    ki_vec_jax = jnp.array(ub_helper.ki_vec)
    ki_vec_jax = ki_vec_jax / (jnp.linalg.norm(ki_vec_jax) + 1e-9)

    lambda_max_jax = 4.0 * jnp.pi / (q_mags_jax + 1e-9)

    expected_noise_ll = float(np.log(1.0 / (4.0 * np.pi)))
    print(f"  > Exact Spherical Background Noise Floor (Log-Likelihood): {expected_noise_ll:.2f}")

    @jax.jit
    def quaternion_to_rotation_matrix(r):
        w, x, y, z = r[0], r[1], r[2], r[3]
        return jnp.array([
            [w*w+x*x-y*y-z*z, 2*(x*y-w*z),     2*(x*z+w*y)],
            [2*(x*y+w*z),     w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
            [2*(x*z-w*y),     2*(y*z+w*x),     w*w-x*x-y*y+z*z]
        ])

    @jax.jit
    def compute_A_from_C(C):
        trC = jnp.trace(C)
        z = jnp.array([C[2, 1] - C[1, 2],
                       C[0, 2] - C[2, 0],
                       C[1, 0] - C[0, 1]])

        A00 = jnp.array([[trC]])
        A01 = z[None, :]
        A10 = z[:, None]
        A11 = C + C.T - trC * jnp.eye(3)

        return jnp.concatenate([
            jnp.concatenate([A00, A01], axis=1),
            jnp.concatenate([A10, A11], axis=1)
        ], axis=0)

    Y_theo_cryst_jax = e3nn.spherical_harmonics(sh_irreps, q_theo_sample_jax.T, normalize=True).array

    @jax.jit
    def process_chunk(A_prev, A_seed, bg_hist_prev, wl_hist_prev, q_batch, ki_batch, t_batch, C_spectral_in, current_sigma_q, delta_angle, current_tau, gamma_event):
        t_curr = t_batch[-1]
        num_events = q_batch.shape[0]

        vals, vecs = jnp.linalg.eigh(A_prev)
        r = vecs[:, -1]
        U = quaternion_to_rotation_matrix(r)

        mu_theo = jnp.matmul(U, q_theo_sample_jax).T

        q_mean = jnp.mean(q_mags_jax)
        q_norm = q_mags_jax / q_mean
        kappa_j = jnp.clip((q_norm ** 2) / (current_sigma_q ** 2), 1e-6, 1e6)

        q_dot_ki_theo = jnp.dot(mu_theo, ki_batch[0])
        lambda_theo = -(2.0 / (q_mags_jax + 1e-9)) * q_dot_ki_theo

        valid_wl_theo = (lambda_theo > 0.0) & (lambda_theo < float(wl_max_tracking))
        safety_penalty_theo = jnp.where(valid_wl_theo, 0.0, -1e9)
        wl_idx_theo = jnp.clip(jnp.floor(lambda_theo * 10.0).astype(jnp.int32), 0, int(num_wl_bins) - 1)

        wl_pdf_spectral = wl_hist_prev / jnp.sum(wl_hist_prev)
        peak_weights_spectral = jnp.exp(jnp.log(wl_pdf_spectral)[wl_idx_theo] + safety_penalty_theo)

        extensive_peak_weights = peak_weights_spectral * kappa_j

        # --------------------------------------------------------------------
        # EXACT CONVOLUTION & INTENSIVE NORMALIZATION
        # --------------------------------------------------------------------
        C_vac_ir = e3nn.IrrepsArray(sh_irreps, C_spectral_in)
        C_cryst = C_vac_ir.transform_by_matrix(U.T).array
        
        # 1. Normalize the continuous vacuum wave into a true Probability Density
        vacuum_mass = jnp.maximum(C_spectral_in[0] * jnp.sqrt(4.0 * jnp.pi), 1e-9)
        C_cryst_pdf = C_cryst / vacuum_mass
        
        # 2. Evaluate the expected spatial density at the theoretical peaks
        normalized_peak_weights = peak_weights_spectral / jnp.maximum(jnp.sum(peak_weights_spectral), 1e-9)
        expected_density = jnp.sum(jnp.dot(Y_theo_cryst_jax, C_cryst_pdf) * normalized_peak_weights)
        
        # 3. Convert continuous density into an exact Log-Likelihood
        spectral_nll = -jnp.log(jnp.maximum(expected_density, 1e-9))

        log_vmf_norm_j = jnp.log(kappa_j / (2.0 * jnp.pi)) - jnp.log(-jnp.expm1(-2.0 * kappa_j))

        # Use dynamic SOC Tau for heat attenuation
        kappa_tilde = kappa_j / (1.0 + 2.0 * current_tau * kappa_j)
        log_vmf_norm_tilde = jnp.log(kappa_tilde / (2.0 * jnp.pi)) - jnp.log(-jnp.expm1(-2.0 * kappa_tilde))

        u_batch = q_batch[:, 2]
        v_batch = jnp.arctan2(q_batch[:, 1], q_batch[:, 0])

        u_idx_batch = jnp.clip(jnp.floor((u_batch + 1.0) * 63.999).astype(jnp.int32), 0, 63)
        v_idx_batch = jnp.clip(jnp.floor((v_batch + jnp.pi) / (2.0 * jnp.pi) * 63.999).astype(jnp.int32), 0, 63)

        flat_idx_batch = u_idx_batch * 64 + v_idx_batch
        raw_hist = jnp.bincount(flat_idx_batch, length=4096)

        bg_hist_new = bg_ema_weight * bg_hist_prev + (1.0 - bg_ema_weight) * raw_hist
        smoothed_hist = bg_hist_new + 1.0

        d_omega = (2.0 / 64.0) * (2.0 * jnp.pi / 64.0)
        bg_pdf = smoothed_hist / (jnp.sum(smoothed_hist) * d_omega)
        bg_log_pdf_flat = jnp.log(bg_pdf)

        wl_max_jax = float(wl_max_tracking)
        num_bins_jax = int(num_wl_bins)

        wl_pdf = wl_hist_prev / jnp.sum(wl_hist_prev)
        wl_log_pdf = jnp.log(wl_pdf)

        def single_event_update(q_exp, ki_exp):
            h_sample = jnp.matmul(U, q_theo_sample_jax)
            cos_theta_err = jnp.dot(q_exp, h_sample)
            q_dot_ki_theo = jnp.dot(ki_exp, h_sample)

            lambda_j = -(2.0 / (q_mags_jax + 1e-9)) * q_dot_ki_theo
            valid_wl = (lambda_j > 0.0) & (lambda_j < wl_max_jax)
            safety_penalty = jnp.where(valid_wl, 0.0, -1e9)

            wl_idx = jnp.clip(jnp.floor(lambda_j * 10.0).astype(jnp.int32), 0, num_bins_jax - 1)
            learned_wl_prior = wl_log_pdf[wl_idx]

            peak_log_lik = kappa_j * (cos_theta_err - 1.0) + log_vmf_norm_j + learned_wl_prior + safety_penalty
            peak_log_lik_tilde = kappa_tilde * (cos_theta_err - 1.0) + log_vmf_norm_tilde + learned_wl_prior + safety_penalty

            u_exp = q_exp[2]
            v_exp = jnp.arctan2(q_exp[1], q_exp[0])
            u_idx_exp = jnp.clip(jnp.floor((u_exp + 1.0) * 63.999).astype(jnp.int32), 0, 63)
            v_idx_exp = jnp.clip(jnp.floor((v_exp + jnp.pi) / (2.0 * jnp.pi) * 63.999).astype(jnp.int32), 0, 63)

            dynamic_bg_log_lik = jnp.array([bg_log_pdf_flat[u_idx_exp * 64 + v_idx_exp]])

            log_Z = jax.scipy.special.logsumexp(jnp.append(peak_log_lik, dynamic_bg_log_lik))
            log_Z_tilde = jax.scipy.special.logsumexp(jnp.append(peak_log_lik_tilde, dynamic_bg_log_lik))

            # --- UNCLIPPED RESIDUAL ---
            # Native statistical mechanics: retains repulsive noise bumps in the Free Energy
            e_short_event = log_Z_tilde - log_Z

            w = jnp.exp(peak_log_lik - log_Z)
            w_sum = jnp.sum(w)

            weighted_h_geom = jnp.sum((w * kappa_j) * q_theo_sample_jax, axis=1)
            F_total = jnp.outer(q_exp, weighted_h_geom)

            best_idx = jnp.argmax(w)
            is_signal = w[best_idx] > 0.5
            actual_winning_lambda = lambda_j[best_idx]

            return compute_A_from_C(F_total), -log_Z, e_short_event, w_sum, is_signal, actual_winning_lambda

        def single_event_wrapper(event_data):
            q_exp, ki_exp = event_data
            return single_event_update(q_exp, ki_exp)

        A_F_batch, nll_batch, e_short_batch, w_sum_batch, is_signal_batch, actual_lambdas_batch = jax.lax.map(
            single_event_wrapper, (q_batch, ki_batch), batch_size=max_gpu_batch_size,
        )

        signal_count = jnp.sum(w_sum_batch)

        wl_idx_batch = jnp.floor(actual_lambdas_batch * 10.0).astype(jnp.int32)
        valid_bin_mask = is_signal_batch & (actual_lambdas_batch > 0.0) & (actual_lambdas_batch < wl_max_jax)

        raw_wl_hist = jnp.bincount(
            jnp.where(valid_bin_mask, wl_idx_batch, num_bins_jax),
            length=num_bins_jax + 1
        )[:num_bins_jax]

        wl_hist_new = bg_ema_weight * wl_hist_prev + (1.0 - bg_ema_weight) * raw_wl_hist + 1e-3

        # The exact implementation of Gamma*(t) = alpha * w(t) + epsilon
        decay_total = jnp.exp(-(gamma_step * delta_angle + gamma_event * num_events))
        
        A_diffused = A_prev * decay_total

        # --- THE INTENSIVE SIGNAL FORCE ---
        # Calculate the geometric precision of the *signal*, ignoring background volume.
        # jnp.maximum(signal_count, 1.0) guarantees that if the batch is 100% noise, 
        # F_pure safely evaluates to near-zero and the tracker correctly decays.
        F_pure = jnp.sum(A_F_batch, axis=0) / jnp.maximum(signal_count, 1.0)
        F_pure = F_pure - (jnp.trace(F_pure) / 4.0) * jnp.eye(4)

        dt_chunk = jnp.maximum(1e-4, t_batch[-1] - t_batch[0])
        total_rate = num_events / dt_chunk
        is_valid_batch = total_rate < max_rate_hz

        # Update using the pure thermodynamic target
        A_updated = A_diffused + (F_pure + A_seed) * (1.0 - decay_total)

        A_new = jnp.where(is_valid_batch, A_updated, A_prev)

        bg_hist_out = jnp.where(is_valid_batch, bg_hist_new, bg_hist_prev)
        wl_hist_out = jnp.where(is_valid_batch, wl_hist_new, wl_hist_prev)

        vals_new = jnp.linalg.eigvalsh(A_new)
        norm_gap = vals_new[-1] - vals_new[-2]

        sig_rate = signal_count / dt_chunk
        bg_rate = (num_events - signal_count) / dt_chunk

        return A_new, bg_hist_out, wl_hist_out, t_curr, U, jnp.mean(nll_batch), jnp.mean(e_short_batch), spectral_nll, norm_gap, sig_rate, bg_rate, bg_pdf

    ensemble_process_chunk = jax.jit(
        jax.vmap(
            process_chunk,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None),
            out_axes=(0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )

    print(f"\n[2/3] Executing Ensemble Bingham Tracking ({n_ensemble} simultaneous trackers)...")

    rng = jax.random.PRNGKey(42)
    r_random = jax.random.normal(rng, (n_ensemble, 4))
    r_random = r_random / jnp.linalg.norm(r_random, axis=1, keepdims=True)
    U_ensemble = jax.vmap(quaternion_to_rotation_matrix)(r_random)

    C_ensemble = kappa_init * U_ensemble
    A_ensemble_state = jax.vmap(compute_A_from_C)(C_ensemble)

    if U_init is not None:
        A_seed = compute_A_from_C(kappa_init * jnp.array(U_init))
        A_ensemble_state = A_ensemble_state.at[0].set(A_seed)

    A_ensemble_state_seeds = jnp.array(A_ensemble_state)

    bg_hist_ensemble = jnp.ones((n_ensemble, 4096)) * 1.0

    num_wl_bins = int(wl_max_tracking * 10.0)
    wl_hist_ensemble = jnp.ones((n_ensemble, num_wl_bins)) * 1.0

    tracking_history = []
    t_start = None
    t_state = 0.0

    effective_annealing_count = 0.0
    angles_prev = None

    smoothed_spectral_ensemble = None
    smoothed_eshort_ensemble = None
    smoothed_loss_ensemble = None
    ema_bg_rate = 1.0

    for batch_data in event_batches:
        q_batch_np, t_batch_np, banks_np, pr_np, pc_np, angles_np, slab_np, ki_sample_np, cumulative_count = batch_data

        if t_start is None and len(t_batch_np) > 0:
            t_state = t_batch_np[0]
            t_start = t_state

        dt_wall = float(t_batch_np[-1] - t_state)

        angles_curr = angles_np[-1]
        if angles_prev is None:
            delta_angle = 0.0
        else:
            delta_angle = float(np.linalg.norm(angles_curr - angles_prev))
        angles_prev = angles_curr

        num_events = len(q_batch_np)
        dt_chunk_py = max(1e-4, float(t_batch_np[-1] - t_batch_np[0]))
        total_rate_py = num_events / dt_chunk_py

        if total_rate_py < max_rate_hz:
            effective_annealing_count += num_events

        current_sigma_q = max(
            sigma_q_min,
            sigma_q_start * np.exp(-annealing_rate * effective_annealing_count)
        )

        q_batch = jax.device_put(q_batch_np)
        q_batch = q_batch / (jnp.linalg.norm(q_batch, axis=1, keepdims=True) + 1e-9)
        t_batch = jax.device_put(t_batch_np)

        ki_batch = jax.device_put(ki_sample_np)
        ki_batch = ki_batch / (jnp.linalg.norm(ki_batch, axis=1, keepdims=True) + 1e-9)

        # --------------------------------------------------------------------
        # SOC: Data-Driven Intensive Scaling via the Background Field
        # --------------------------------------------------------------------
        # The 2D background histogram natively maps the instrument's geometry.
        # Count how many of the 4096 spatial bins are actively receiving neutrons.
        # (Using bg_hist_ensemble[0] since the background is shared/identical across the ensemble)
        active_bins = jnp.sum(bg_hist_ensemble[0] > 1e-3)
        
        # Calculate the fraction of the scattering sphere currently covered by detectors
        coverage_fraction = jnp.maximum(active_bins / 4096.0, 1e-4) # Prevent div-by-zero
        
        # The Intensive Background Rate (Hz per full-sphere equivalent)
        intensive_bg_rate = ema_bg_rate / coverage_fraction

        # Adaptive Tau based purely on the spatial density of the noise
        current_tau = gamma_c * jnp.sqrt(intensive_bg_rate + 1.0)
        
        # Thermodynamic ceiling to prevent the wave from boiling flat during extreme anomalies
        #current_tau = jnp.minimum(current_tau, 2.0)

        A_ensemble_state, bg_hist_ensemble, wl_hist_ensemble, t_state, U_ensemble_curr, loss_ensemble, eshort_ensemble, spectral_losses, eigen_gaps, sig_rates, bg_rates, bg_pdfs = ensemble_process_chunk(
            A_ensemble_state, A_ensemble_state_seeds, bg_hist_ensemble, wl_hist_ensemble,
            q_batch, ki_batch, t_batch, C_spectral_state, current_sigma_q, delta_angle, current_tau, gamma_event
        )

        # Update EMA background rate for next cycle's Tau
        current_mean_bg = float(np.mean(bg_rates))
        ema_bg_rate = bg_ema_weight * ema_bg_rate + (1.0 - bg_ema_weight) * current_mean_bg

        # --------------------------------------------------------------------
        # The KS Scalar Sink: Bayesian Expected Rate
        # --------------------------------------------------------------------
        # The 256 trackers are independent hypotheses. We must drain the vacuum
        # using the expected value of the signal rate across the ensemble.
        ensemble_weights = jax.nn.softmax(-loss_ensemble)
        expected_lambda_short = jnp.sum(sig_rates * ensemble_weights)

        # --------------------------------------------------------------------
        # Evolve the Continuous Vacuum Field (Instantaneous Unitarity)
        # --------------------------------------------------------------------
        C_spectral_state = evolve_vacuum_sde(
            C_spectral_state, q_batch, expected_lambda_short, dt_chunk_py, current_tau
        )

        spectral_losses_np = np.array(spectral_losses)
        eshort_ensemble_np = np.array(eshort_ensemble)
        loss_ensemble_np = np.array(loss_ensemble)

        if smoothed_eshort_ensemble is None:
            smoothed_eshort_ensemble = eshort_ensemble_np
            smoothed_spectral_ensemble = spectral_losses_np
            smoothed_loss_ensemble = loss_ensemble_np
        else:
            smoothed_eshort_ensemble = (1.0 - loss_ema_weight) * smoothed_eshort_ensemble + loss_ema_weight * eshort_ensemble_np
            smoothed_spectral_ensemble = (1.0 - loss_ema_weight) * smoothed_spectral_ensemble + loss_ema_weight * spectral_losses_np
            smoothed_loss_ensemble = (1.0 - loss_ema_weight) * smoothed_loss_ensemble + loss_ema_weight * loss_ensemble_np

        # The LogSumExp (-loss_ensemble) is identically the Log-Partition Function.
        # It natively and perfectly balances Potential Energy and Shannon Entropy.
        best_idx = int(np.argmin(smoothed_loss_ensemble))

        ensemble_weights = jax.nn.softmax(-loss_ensemble)
        wl_hist_weighted = jnp.sum(wl_hist_ensemble * ensemble_weights[:, None], axis=0)
        wl_hist_ensemble = jnp.broadcast_to(wl_hist_weighted, wl_hist_ensemble.shape)

        U_best = np.array(U_ensemble_curr[best_idx])
        best_gap = float(eigen_gaps[best_idx])
        best_loss = float(loss_ensemble[best_idx])

        tracking_history.append((float(t_state), U_best))

        current_sig_rate = float(sig_rates[best_idx])
        current_bg_rate = float(bg_rates[best_idx])

        if cumulative_count % 50000 < len(t_batch_np):
            print(f"    Time {t_state:.2f}s | Sig/Bg: {current_sig_rate:.0f}/{current_bg_rate:.0f} Hz | Step-Rot: {delta_angle:.4f} rad | Norm-Gap: {best_gap:8.2f} | Tau: {float(current_tau):.4f}")

        if streaming_callback is not None:
            new_events = {
                "banks": banks_np,
                "pixel_r": pr_np,
                "pixel_c": pc_np,
                "angles": angles_np,
                "s_lab": slab_np
            }

            metrics_dict = {
                "mean_loss": best_loss,
                "eigengap": best_gap,
                "sig_rate": current_sig_rate,
                "bg_rate": current_bg_rate,
                "sigma_q": current_sigma_q,
                "bg_pdf": np.array(bg_pdfs[best_idx]).reshape((64, 64)),
                "wl_hist": np.array(wl_hist_ensemble[best_idx]),
                "wl_max": wl_max_tracking,
                "tau": float(current_tau),
            }

            streaming_callback(
                time=float(t_state),
                U_preds=np.array(U_ensemble_curr),
                losses=np.array(loss_ensemble),
                mean_loss=best_loss,
                best_idx=best_idx,
                neutron_count=cumulative_count,
                new_events=new_events,
                metrics=metrics_dict,
            )

    if not tracking_history:
        print("\n[3/3] Tracking complete. No events were processed.")
        return None

    print(f"\n[3/3] Global Tracking complete. Extracted {len(tracking_history)} continuous U-matrices.")
    return tracking_history[-1][1]
