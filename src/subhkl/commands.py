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
            off_data = b_f.get("goniometer/offsets") or b_f.get("optimization/goniometer_offsets")
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

        if mode != "zone_axis":
            f["peaks/h"] = hkl_uvw[:, 0]
            f["peaks/k"] = hkl_uvw[:, 1]
            f["peaks/l"] = hkl_uvw[:, 2]
            f["peaks/lambda"] = lambda_S
        else:
            f["zones/u"] = hkl_uvw[:, 0]
            f["zones/v"] = hkl_uvw[:, 1]
            f["zones/w"] = hkl_uvw[:, 2]
            f["zones/S"] = lambda_S

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
        if wavel_min:
            wavelength[0] = wavel_min
        if wavel_max:
            wavelength[1] = wavel_max

        U = f_idx["sample/U"][()]
        B = f_idx["sample/B"][()]

        offsets = None
        off_data = f_idx.get("goniometer/offsets") or f_idx.get("optimization/goniometer_offsets")
        if off_data is not None:
            if isinstance(off_data, h5py.Group):
                offsets = {k: off_data[k][()] for k in off_data.keys()}
            else:
                offsets = off_data[()]

        if "goniometer/translations" in f_idx:
            sample_offset = f_idx["goniometer/translations"][()]
        else:
            sample_offset = np.zeros(3)

        gonio_axes = f_idx["goniometer/axes"][()] if "goniometer/axes" in f_idx else None

        ki_vec = (
            f_idx["beam/ki_vec"][()]
            if "beam/ki_vec" in f_idx
            else np.array([0.0, 0.0, 1.0])
        )

    peaks = Peaks(
        filename, instrument, wavelength_min=wavelength[0], wavelength_max=wavelength[1]
    )
    print(
        f"Predicting peaks for {len(peaks.image.ims)} images using solution from {indexed_hdf5_filename}"
    )

    all_R = peaks.goniometer.rotation

    if offsets is not None:
        from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix

        print(f"Applying refined goniometer offsets from indexer: {offsets}")
        if (
            peaks.goniometer.angles_raw is not None
            and peaks.goniometer.axes_raw is not None
        ):
            # --- SAFE NAMED MAPPING ---
            if isinstance(offsets, dict) and peaks.goniometer.names_raw is not None:
                mapped_offsets = np.array(
                    [offsets.get(name, 0.0) for name in peaks.goniometer.names_raw]
                )
            else:
                # Legacy array fallback
                motor_map = []
                if peaks.goniometer.names_raw is not None:
                    unique_motors = []
                    for name in peaks.goniometer.names_raw:
                        if name not in unique_motors:
                            unique_motors.append(name)
                        motor_map.append(unique_motors.index(name))
                else:
                    motor_map = list(range(len(peaks.goniometer.axes_raw)))
                mapped_offsets = np.array(
                    [offsets[motor_map[i]] for i in range(len(motor_map))]
                )

            angles_refined = peaks.goniometer.angles_raw + mapped_offsets[None, :]

            all_R = np.stack(
                [
                    calc_goniometer_rotation_matrix(peaks.goniometer.axes_raw, ang)
                    for ang in angles_refined
                ]
            )

    UB = U @ B
    if all_R.ndim == 3:
        RUB = np.matmul(all_R, UB)
    else:
        RUB = all_R @ UB

    results_map = peaks.predict_peaks(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        d_min,
        RUB=RUB,
        space_group=space_group,
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        max_workers=max_workers,
        R_all=all_R,
        gonio_axes=gonio_axes,
        gonio_angles=angles_refined,
    )

    print(f"Saving predictions to {integration_peaks_filename}")
    with h5py.File(integration_peaks_filename, "w") as f:
        f.attrs["instrument"] = instrument
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = alpha, beta, gamma

        sorted_keys = sorted(peaks.image.ims.keys())
        bank_ids = np.array(
            [peaks.image.bank_mapping.get(k, k) for k in sorted_keys], dtype=np.int32
        )
        f.create_dataset("bank_ids", data=bank_ids)

        f["sample/space_group"] = space_group
        f["sample/U"], f["sample/B"] = U, B
        f["instrument/wavelength"] = wavelength
        f["goniometer/R"] = all_R

        try:
            goniometer_angles_to_save = angles_refined
        except NameError:
            goniometer_angles_to_save = peaks.goniometer.angles_raw

        f["goniometer/angles"] = goniometer_angles_to_save
        f["goniometer/axes"] = peaks.goniometer.axes_raw
        if peaks.goniometer.names_raw:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "goniometer/names", data=peaks.goniometer.names_raw, dtype=dt
            )

        f["goniometer/translations"] = sample_offset
        f["beam/ki_vec"] = ki_vec

        for img_key, (i, j, h, k, l, wl) in results_map.items():
            grp = f.create_group(f"banks/{img_key}")
            grp.create_dataset("i", data=i)
            grp.create_dataset("j", data=j)
            grp.create_dataset("h", data=h)
            grp.create_dataset("k", data=k)
            grp.create_dataset("l", data=l)
            grp.create_dataset("wavelength", data=wl)

        # Forward the calibration group to the prediction file
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
    print(
        f"Parameters: Alpha={alpha}, Gamma={gamma}, Sigma={sigma_list}"
    )

    peak_dict = {}

    with h5py.File(integration_peaks_filename, "r") as f:
        if "sample/U" in f:
            f["sample/U"][()]
        if "sample/B" in f:
            f["sample/B"][()]
        if "goniometer/R" in f:
            all_R = f["goniometer/R"][()]
        if "goniometer/angles" in f:
            angles_stack = f["goniometer/angles"][()]

        if "goniometer/translations" in f:
            sample_offset = f["goniometer/translations"][()]
        else:
            sample_offset = np.zeros(3)

        gonio_axes = f["goniometer/axes"][()] if "goniometer/axes" in f else None

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

    if all_R is None:
        all_R = peaks.goniometer.rotation
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
        all_R=all_R,
        sample_offset=sample_offset,
        anisotropic=anisotropic,
        fit_mosaicity=fit_mosaicity,
        border_width=border_width,
        chunk_size=chunk_size,
        create_visualizations=create_visualizations,
        file_prefix=filename,
        max_workers=max_workers,
        gonio_axes=gonio_axes,
        gonio_angles=angles_stack
    )

    print(f"Saving RBF integrated peaks to {output_filename}")
    with h5py.File(output_filename, "w") as f:
        f["peaks/h"] = result.h
        f["peaks/k"] = result.k
        f["peaks/l"] = result.l
        f["peaks/lambda"] = result.wavelength
        f["peaks/intensity"] = result.intensity
        f["peaks/sigma"] = result.sigma  # SVD-stabilized Fisher Info UQ
        f["peaks/two_theta"] = result.tt
        f["peaks/azimuthal"] = result.az
        f["peaks/bank"] = result.bank
        f["peaks/run_index"] = result.run_id

        # Copy full metadata context from predictor output
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

        with h5py.File(integration_peaks_filename, "r") as f_in:
            for key in copy_keys:
                if key in f_in:
                    f_in.copy(f_in[key], f, key)

            for k in ["goniometer/axes", "goniometer/names"]:
                if k in f_in:
                    f_in.copy(f_in[k], f, k)


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

def run_egnn(
    finder_file: str,
    output_h5_filename: str,
    instrument_name: str | None = None,
    original_nexus_filename: str | None = None,
    steps: int = 3000,  # Now drives the EGNN Annealing steps
    create_visualizations: bool = False,
    egnn_sigma_end: float = 0.005,
    max_hkl_cons: int = 2,
):
    from subhkl.optimization import FindUB
    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    import h5py
    import numpy as np

    print(f"\n[1/4] Initializing Reciprocal Space from: {finder_file}")

    ub_helper = FindUB()
    with h5py.File(finder_file, "r") as f:
        ub_helper.a = f["sample/a"][()] if "sample/a" in f else None
        ub_helper.b = f["sample/b"][()] if "sample/b" in f else None
        ub_helper.c = f["sample/c"][()] if "sample/c" in f else None
        ub_helper.alpha = f["sample/alpha"][()] if "sample/alpha" in f else None
        ub_helper.beta = f["sample/beta"][()] if "sample/beta" in f else None
        ub_helper.gamma = f["sample/gamma"][()] if "sample/gamma" in f else None

        ub_helper.sample_offset = f["sample/offset"][()] if "sample/offset" in f else np.zeros(3)
        ub_helper.ki_vec = f["beam/ki_vec"][()] if "beam/ki_vec" in f else np.array([0.0, 0.0, 1.0])

        sg = f["sample/space_group"][()]
        ub_helper.space_group = sg.decode("utf-8") if isinstance(sg, bytes) else str(sg)
        ub_helper.wavelength = f["instrument/wavelength"][()]

        # ---------------------------------------------------------
        # NEW: LOAD RAW PIXELS INSTEAD OF PRE-FOUND PEAKS
        # ---------------------------------------------------------
        from subhkl.integration.api import Peaks
        print(f"  > Bypassing peak list. Extracting raw pixels from: {original_nexus_filename}")
        
        peaks_obj = Peaks(original_nexus_filename, instrument_name)
        images_dict = peaks_obj.image.ims

        pixel_r_list, pixel_c_list, bank_list, intensity_list, img_idx_list = [], [], [], [], []

        for logical_key, img_data in images_dict.items():
            # 1. Map logical index to physical bank ID
            bank_id = peaks_obj.image.bank_mapping.get(logical_key, logical_key)
            bank_id = int(bank_id)

            # 2. Extract all pixels with at least 1 count (include the noise!)
            mask = img_data > 0
            if not np.any(mask): 
                continue

            # 3. Get exact pixel coordinates
            cols, rows = np.meshgrid(np.arange(img_data.shape[1]), np.arange(img_data.shape[0]))
            active_r = rows[mask]
            active_c = cols[mask]
            active_intensities = img_data[mask]

            pixel_r_list.append(active_r)
            pixel_c_list.append(active_c)
            intensity_list.append(active_intensities)
            bank_list.append(np.full(len(active_r), bank_id))
            img_idx_list.append(np.full(len(active_r), logical_key))

        # Concatenate all active pixels across the entire detector array
        pixel_r = np.concatenate(pixel_r_list)
        pixel_c = np.concatenate(pixel_c_list)
        peak_intensities = np.concatenate(intensity_list)
        bank_array = np.concatenate(bank_list)
        img_indices = np.concatenate(img_idx_list)
        
        # We need the Goniometer rotations for these specific images
        if "goniometer/R" in f:
            R_all = f["goniometer/R"][()]
            # Ensure img_indices are integers to use as array indices
            r_gonio_obs = R_all[img_indices.astype(int)]
        else:
            r_gonio_obs = np.tile(np.eye(3), (len(pixel_r), 1, 1))

        # ---------------------------------------------------------
        # VRAM PROTECTION: RANDOM SUBSAMPLING
        # ---------------------------------------------------------
        total_pixels = len(pixel_r)
        max_pixels_for_vram = 50000 # Adjust based on your GPU memory limits
        
        if total_pixels > max_pixels_for_vram:
            print(f"  > WARNING: Found {total_pixels:,} active pixels. Randomly subsampling to {max_pixels_for_vram:,} to prevent GPU OOM.")
            # We use random choice to perfectly preserve the ratio of signal-to-noise
            np.random.seed(42)
            sub_idx = np.random.choice(total_pixels, max_pixels_for_vram, replace=False)
            
            pixel_r = pixel_r[sub_idx]
            pixel_c = pixel_c[sub_idx]
            peak_intensities = peak_intensities[sub_idx]
            bank_array = bank_array[sub_idx]
            r_gonio_obs = r_gonio_obs[sub_idx]
            # (If you need img_indices later, subsample it here too)
        else:
            print(f"  > Found {total_pixels:,} active pixels. Fitting in VRAM natively.")
        # ---------------------------------------------------------

    B_mat = ub_helper.reciprocal_lattice_B()

    print("  > Reconstructing sparse geometry for downstream indexing...")
    xyz_out = np.zeros((len(pixel_r), 3))
    for phys_bank in np.unique(bank_array):
        mask = bank_array == phys_bank
        if not np.any(mask): continue
        det_config = beamlines[instrument_name][str(int(phys_bank))]
        det = Detector(det_config)
        xyz_out[mask] = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])

    kf = xyz_out - ub_helper.sample_offset[None, :]
    kf = kf / np.linalg.norm(kf, axis=1, keepdims=True)
    q_lab_obs = kf - ub_helper.ki_vec[None, :]
    q_sample_obs = np.einsum('nij,ni->nj', r_gonio_obs, q_lab_obs)

    # The pure geometric vectors
    q_sample_obs_norm = q_sample_obs / np.linalg.norm(q_sample_obs, axis=1, keepdims=True)

    # Principled Normalization (Log Dampening)
    log_weights = np.log1p(peak_intensities)
    w_min, w_max = np.min(log_weights), np.max(log_weights)
    if w_max > w_min:
        normalized_weights = (log_weights - w_min) / (w_max - w_min) * 0.9 + 0.1
    else:
        normalized_weights = np.ones_like(log_weights)

    # ==========================================
    # PHASE 3A: Independent Brute-Force Search
    # ==========================================
    from subhkl.optimization import get_lattice_system
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    q_sample_unnorm = jnp.array(q_sample_obs).T
    intensities = jnp.array(normalized_weights)
    
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

    wavelengths = np.array(ub_helper.wavelength)
    lam_grid = jnp.logspace(jnp.log10(wavelengths[0]), jnp.log10(wavelengths[1]), 64)
    B_inv = jnp.linalg.inv(B_mat)

    def compute_U(rot_6d):
        a1 = rot_6d[:3]
        a2 = rot_6d[3:]
        b1 = a1 / (jnp.linalg.norm(a1) + 1e-6)
        b2 = a2 - jnp.dot(b1, a2) * b1
        b2 = b2 / (jnp.linalg.norm(b2) + 1e-6)
        b3 = jnp.cross(b1, b2)
        return jnp.stack([b1, b2, b3], axis=-1)

    def single_particle_loss(params, current_c, q_batch_unnorm):
        U_pred = compute_U(params)
        UB_inv = jnp.matmul(B_inv, U_pred.T)
        
        # Shape: (3, N)
        v = jnp.matmul(UB_inv, q_batch_unnorm)

        # 1. Unroll components for XLA fusion
        v_h = v[0, :]
        v_k = v[1, :]
        v_l = v[2, :]

        N_pts = v.shape[1]
        
        # We track the minimum distance and the best lambda
        initial_carry = (
            jnp.inf * jnp.ones(N_pts),          # curr_min_diff_sq
            jnp.zeros(N_pts, dtype=jnp.float32) # curr_best_lamb
        )

        # 2. Sequential scan over lam_grid to avoid the massive VRAM tensor explosion
        def scan_body(carry, i):
            curr_min, curr_best_lamb = carry
            lamda_cand = lam_grid[i]

            h_float = v_h / lamda_cand
            k_float = v_k / lamda_cand
            l_float = v_l / lamda_cand

            h_int = jnp.round(h_float)
            k_int = jnp.round(k_float)
            l_int = jnp.round(l_float)

            diff_sq = (h_float - h_int)**2 + (k_float - k_int)**2 + (l_float - l_int)**2

            update_mask = diff_sq < curr_min
            new_min = jnp.where(update_mask, diff_sq, curr_min)
            new_best_lamb = jnp.where(update_mask, lamda_cand, curr_best_lamb)

            return (new_min, new_best_lamb), None

        # Execute the scan loop (JAX will perfectly fuse this on the GPU)
        (_, best_lam_coarse), _ = jax.lax.scan(scan_body, initial_carry, jnp.arange(len(lam_grid)))

        # 3. TARGET LOCK: Extract nearest integer and STOP GRADIENTS
        hkl_int = jnp.round(v / best_lam_coarse[None, :])
        hkl_int_fixed = jax.lax.stop_gradient(hkl_int)

        # 4. ANALYTICAL PROJECTION
        num = jnp.sum(v**2, axis=0)
        den = jnp.sum(v * hkl_int_fixed, axis=0) + 1e-9
        lam_opt = jnp.clip(num / den, lam_grid[0], lam_grid[-1])

        # 5. EXACT FRACTIONAL COORDINATES
        hkl_float_exact = v / lam_opt[None, :]
        h = hkl_float_exact[0, :]
        k = hkl_float_exact[1, :]
        l = hkl_float_exact[2, :]

        # 6. CENTERING & LOSS
        h_p = M_prim[0, 0]*h + M_prim[0, 1]*k + M_prim[0, 2]*l
        k_p = M_prim[1, 0]*h + M_prim[1, 1]*k + M_prim[1, 2]*l
        l_p = M_prim[2, 0]*h + M_prim[2, 1]*k + M_prim[2, 2]*l

        dist_sq = (jnp.sin(jnp.pi * h_p)**2 + jnp.sin(jnp.pi * k_p)**2 + jnp.sin(jnp.pi * l_p)**2) / (jnp.pi**2)

        robust_penalty = dist_sq / (dist_sq + current_c**2)
        loss = jnp.mean(robust_penalty)

        return loss, U_pred

    J_total = 16384      # Total swarm size you want
    J_batch = 16384       # Maximum particles your H100 can compile at once
    num_batches = J_total // J_batch
    
    print(f"\n[3A/3] Deploying Massive Independent Swarm (J={J_total}) in {num_batches} batches...")
    
    # Compile the training step for the batch size
    single_grad_fn = jax.value_and_grad(single_particle_loss, has_aux=True)
    swarm_grad_fn = jax.vmap(single_grad_fn, in_axes=(0, None))

    steps_phase1 = 2000 # Increased steps for narrower loss landscape
    
    # Define the optimizer (but do not init it yet!)
    opt_1 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=optax.cosine_decay_schedule(0.05, steps_phase1))
    )

    @jax.jit
    def train_step_indep(params, opt_state, current_c):
        (individual_losses, U_preds), individual_grads = swarm_grad_fn(params, current_c)
        mean_grads = individual_grads / J_batch
        updates, opt_state = opt_1.update(mean_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, jnp.mean(individual_losses), individual_losses

    c_start, c_end = 0.25, 0.05
    rng = jax.random.PRNGKey(42)
    
    # Storage for the final results of all batches
    all_final_params = []
    all_final_losses = []

    for b in range(num_batches):
        print(f"\n  -> Processing Batch {b+1}/{num_batches}...")
        
        # 1. Spawn a fresh batch of particles
        rng, subkey = jax.random.split(rng)
        batch_params = jax.random.normal(subkey, (J_batch, 6)) 
        
        # 2. Initialize a fresh optimizer state for this batch
        opt_state_1 = opt_1.init(batch_params)
        
        for i in range(steps_phase1):
            progress = i / steps_phase1
            current_c = c_start * (c_end / c_start) ** progress
            
            batch_params, opt_state_1, mean_loss, batch_losses = train_step_indep(
                batch_params, opt_state_1, current_c
            )
            
            if (i + 1) % (steps_phase1 // 5) == 0:
                best_loss = jnp.min(batch_losses)
                print(f"      Step {i+1:04d}/{steps_phase1} | Batch Mean: {mean_loss:.4f} | Best: {best_loss:.4f} | c: {current_c:.3f}")
        
        # 3. Store the final relaxed state of this batch
        all_final_params.append(batch_params)
        all_final_losses.append(batch_losses)

    # ==========================================
    # POOL AND PRUNE FOR PHASE 3B
    # ==========================================
    # Combine all 131,072 particles back into a single unified array
    swarm_params = jnp.concatenate(all_final_params, axis=0)
    indep_losses = jnp.concatenate(all_final_losses, axis=0)
#
#    J_ensf = 1024
#    print(f"\n[3B/3] Pooling {J_total} particles and pruning to Top {J_ensf} for Gravity Collapse...")
#    
#    # Find the global best 1024 particles out of the massive pool
#    sort_indices = jnp.argsort(indep_losses)
#    top_indices = sort_indices[:J_ensf]
#    
#    ensf_params = swarm_params[top_indices]
#    ensf_losses = indep_losses[top_indices]

    ensf_params = swarm_params
    ensf_losses = indep_losses

    # 2. Define Loss-Weighted KDE Gravity
    @jax.jit
    def single_loss_weighted_vmf(single_params, U_swarm, losses_swarm, current_sigma, gamma):
        U_single = compute_U(single_params)
        sq_chordal_dists = jnp.sum((U_swarm - U_single)**2, axis=(-2, -1))
        
        # Meritocratic Gravity: Penalty based on distance AND target loss
        logits = -sq_chordal_dists / (2.0 * current_sigma**2) - (gamma * losses_swarm)
        return jax.scipy.special.logsumexp(logits)

    prior_score_fn = jax.grad(single_loss_weighted_vmf, argnums=0)
    vmap_prior = jax.vmap(prior_score_fn, in_axes=(0, None, None, None, None))

    # We need a vmapped version of the raw loss to use inside the EnSF train_step
    ensf_loss_fn = jax.vmap(single_particle_loss, in_axes=(0, None))

    steps_phase2 = 5000
    opt_2 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=optax.cosine_decay_schedule(0.01, steps_phase2))
    )
    opt_state_2 = opt_2.init(ensf_params)

    @jax.jit
    def train_step_ensf(params, opt_state, current_c, current_sigma, gamma, current_lambda):
        def total_loss(p):
            losses, Us = ensf_loss_fn(p, current_c)
            return jnp.mean(losses), (Us, losses)
            
        (mean_loss, (U_preds, individual_losses)), raw_grads = jax.value_and_grad(
            total_loss, has_aux=True
        )(params)
        
        # Calculate Loss-Weighted Gravity
        prior_scores = vmap_prior(params, U_preds, individual_losses, current_sigma, gamma)
        
        # Subtract gravity to move UP the density gradient
        total_grads = raw_grads - (current_lambda * prior_scores)
        
        updates, opt_state = opt_2.update(total_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, mean_loss, U_preds, individual_losses

    # Gravity Hyperparameters
    ensf_sigma = 0.10      # Tight chordal radius
    ensf_gamma = 1000.0      # Punish high-loss particles massively
    ensf_lambda = 0.5      # Strength of the gravitational pull

    for i in range(steps_phase2):
        ensf_params, opt_state_2, mean_loss, U_preds, ensf_losses = train_step_ensf(
            ensf_params, opt_state_2, current_c=c_end, current_sigma=ensf_sigma, 
            gamma=ensf_gamma, current_lambda=ensf_lambda
        )
        if (i + 1) % 50 == 0:
            best_loss = jnp.min(ensf_losses)
            print(f"    Step {i+1:03d}/{steps_phase2} | Swarm Mean: {mean_loss:.4f} | Best: {best_loss:.4f}")

    # ==========================================
    # PHASE 4: Localized Differentiable Extraction
    # ==========================================
    print(f"\n[4/4] Extracting differentiable modes via Meritocratic Mean-Shift...")
    
    # 1. Compute the final 3x3 matrices from the EnSF swarm
    U_final_swarm = jax.vmap(compute_U)(ensf_params)
    n_grains = 5  # Set how many orientations/twins you want to extract

    # 2. Heuristic Initialization: Find distinct, low-loss seeds
    def get_initial_seeds(U_swarm, losses, n):
        seeds = []
        first_idx = jnp.argmin(losses)
        seeds.append(U_swarm[first_idx])
        
        penalty_scale = 10.0
        for _ in range(1, n):
            curr_seeds = jnp.stack(seeds)
            dists = jnp.sum((U_swarm[:, None, :, :] - curr_seeds[None, :, :, :])**2, axis=(-2, -1))
            min_dists = jnp.min(dists, axis=1)
            
            scores = losses - penalty_scale * min_dists
            next_idx = jnp.argmin(scores)
            seeds.append(U_swarm[next_idx])
            
        return jnp.stack(seeds)

    initial_centers = jax.lax.stop_gradient(
        get_initial_seeds(U_final_swarm, ensf_losses, n_grains)
    )

    # 3. Differentiable SO(3) Projection
    @jax.vmap
    def project_so3(M):
        U_svd, _, Vh_svd = jnp.linalg.svd(M)
        det = jnp.linalg.det(jnp.matmul(U_svd, Vh_svd))
        correction = jnp.diag(jnp.array([1.0, 1.0, det]))
        return jnp.matmul(U_svd, jnp.matmul(correction, Vh_svd))

    # 4. Meritocratic Mean-Shift Step
    @jax.jit
    def extract_modes_step(centers, U_swarm, losses, tau_dist=0.01, gamma=100.0):
        # Shape: (1024, N)
        dists = jnp.sum((U_swarm[:, None, :, :] - centers[None, :, :, :])**2, axis=(-2, -1))
        
        # THE FIX: Softmax over particles (axis=0), heavily punishing distance and loss
        logits = -dists / tau_dist - gamma * losses[:, None]
        weights = jax.nn.softmax(logits, axis=0)
        
        # Weighted average of matrices for each center
        M_avg = jnp.sum(weights[:, :, None, None] * U_swarm[:, None, :, :], axis=0)
        new_centers = project_so3(M_avg)
        return new_centers

    # 5. Execute Extraction
    mode_centers = initial_centers
    for _ in range(10):
        mode_centers = extract_modes_step(mode_centers, U_final_swarm, ensf_losses)

    # 6. Calculate Relative Phase Fractions
    # Now we post-process to see how many particles actually belong to each extracted grain
    final_dists = jnp.sum((U_final_swarm[:, None, :, :] - mode_centers[None, :, :, :])**2, axis=(-2, -1))
    best_center_idx = jnp.argmin(final_dists, axis=1)
    min_dists = jnp.min(final_dists, axis=1)
    
    # Only count particles that are physically close to a mode (chordal dist < 0.1)
    valid_mask = min_dists < 0.1
    valid_assignments = jnp.where(valid_mask, best_center_idx, n_grains) # assign noise to a dummy bin
    
    cluster_counts = jnp.bincount(valid_assignments, length=n_grains+1)[:-1]
    total_valid = jnp.sum(cluster_counts)
    phase_fractions = cluster_counts / (total_valid + 1e-6)

    for g in range(n_grains):
        print(f"  > Grain {g+1}: Relative Phase Fraction = {phase_fractions[g]:.1%}")

    kmeans_centers = mode_centers
    U_final = kmeans_centers[0]

    # ==========================================
    # FINAL VALIDATION (Fractional Distance)
    # ==========================================
    # We still use the un-normalized lab vectors against the continuous lam_grid 
    # to print out the final crystallographic fractional hit rate for the user!
    q_sample_unnorm = np.array(q_sample_obs).T
    UB_inv_final = np.linalg.inv(B_mat) @ U_final.T
    v_final = UB_inv_final @ q_sample_unnorm
    
    wavelengths = np.array(ub_helper.wavelength)
    lam_grid = np.logspace(np.log10(wavelengths[0]), np.log10(wavelengths[1]), 64)
    hkl_float_final = v_final[np.newaxis, :, :] / lam_grid[:, np.newaxis, np.newaxis]
    
    # Setup the Primitive Matrix (M_prim) based on Space Group
    _, _, centering = get_lattice_system(
        ub_helper.a, ub_helper.b, ub_helper.c,
        ub_helper.alpha, ub_helper.beta, ub_helper.gamma,
        ub_helper.space_group
    )

    if centering == "I": M_prim = jnp.array([[0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5]])
    elif centering == "F": M_prim = jnp.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    elif centering == "C": M_prim = jnp.array([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.0, 1.0]])
    elif centering == "A": M_prim = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, -0.5]])
    elif centering == "B": M_prim = jnp.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, -0.5]])
    elif centering == "R": M_prim = jnp.array([[2/3, 1/3, 1/3], [-1/3, 1/3, 1/3], [-1/3, -2/3, 1/3]])
    else: M_prim = jnp.eye(3)

    h_f, k_f, l_f = hkl_float_final[:, 0, :], hkl_float_final[:, 1, :], hkl_float_final[:, 2, :]
    M_prim_np = np.array(M_prim) # Assumes M_prim was defined earlier in your script
    
    h_p_f = M_prim_np[0, 0]*h_f + M_prim_np[0, 1]*k_f + M_prim_np[0, 2]*l_f
    k_p_f = M_prim_np[1, 0]*h_f + M_prim_np[1, 1]*k_f + M_prim_np[1, 2]*l_f
    l_p_f = M_prim_np[2, 0]*h_f + M_prim_np[2, 1]*k_f + M_prim_np[2, 2]*l_f
    
    dist_final = np.sqrt((h_p_f - np.round(h_p_f))**2 + (k_p_f - np.round(k_p_f))**2 + (l_p_f - np.round(l_p_f))**2)
    best_dist = np.min(dist_final, axis=0)
    
    true_hits = np.sum(best_dist < 0.15)
    print(f"  > Final Matrix correctly indexed {true_hits}/{len(q_sample_obs_norm)} Bragg peaks (Fractional Tol = 0.15).")

    # ==========================================
    # FINAL VALIDATION (Replaces Phase 4)
    # ==========================================
    # Validate the final matrix natively using the continuous fractional distance
    # This correctly catches high-order peaks that were ignored by max_hkl bounds
    UB_inv_final = np.linalg.inv(B_mat) @ U_final.T
    v_final = UB_inv_final @ np.array(q_sample_unnorm)
    hkl_float_final = v_final[np.newaxis, :, :] / np.array(lam_grid)[:, np.newaxis, np.newaxis]
    
    h_f, k_f, l_f = hkl_float_final[:, 0, :], hkl_float_final[:, 1, :], hkl_float_final[:, 2, :]
    M_prim_np = np.array(M_prim)
    
    h_p_f = M_prim_np[0, 0]*h_f + M_prim_np[0, 1]*k_f + M_prim_np[0, 2]*l_f
    k_p_f = M_prim_np[1, 0]*h_f + M_prim_np[1, 1]*k_f + M_prim_np[1, 2]*l_f
    l_p_f = M_prim_np[2, 0]*h_f + M_prim_np[2, 1]*k_f + M_prim_np[2, 2]*l_f
    
    dist_final = np.sqrt((h_p_f - np.round(h_p_f))**2 + (k_p_f - np.round(k_p_f))**2 + (l_p_f - np.round(l_p_f))**2)
    best_dist = np.min(dist_final, axis=0)
    
    # A fractional error of 0.15 represents an excellent topological fit 
    # roughly equivalent to a tight reciprocal angular tolerance.
    true_hits = np.sum(best_dist < 0.15)
    print(f"  > Final Matrix correctly indexed {true_hits}/{len(q_sample_obs_norm)} Bragg peaks (Fractional Tol = 0.15).")

    print(f"\nSaving final U-matrix and experimental geometry to: {output_h5_filename}")
    with h5py.File(output_h5_filename, "w") as f:
        with h5py.File(finder_file, "r") as f_in:
            for key in ["sample/a", "sample/b", "sample/c", "sample/alpha", "sample/beta", "sample/gamma",
                        "sample/space_group", "sample/offset", "instrument/wavelength", "beam/ki_vec",
                        "peaks/run_index", "peaks/image_index", "bank", "bank_ids", "peaks/pixel_r", "peaks/pixel_c",
                        "goniometer/R", "goniometer/axes", "goniometer/angles", "goniometer/names"]:
                if key in f_in:
                    f.create_dataset(key, data=f_in[key][()])
        f.create_dataset("peaks/xyz", data=xyz_out)
        f.create_dataset("sample/U", data=U_final) 
        f.create_dataset("sample/B", data=B_mat)

    print("Export complete.")

    # ==========================================
    # PHASE 5: PARALLEL VISUALIZATION
    # ==========================================
    if create_visualizations:
        import os
        import multiprocessing
        import concurrent.futures
        from tqdm import tqdm
        from collections import defaultdict

        from subhkl.integration.api import Peaks

        print("\n[5/5] Rendering Detector Plots (Parallel)...")
        peaks_obj = Peaks(original_nexus_filename, instrument_name)

        runs_plot_data = defaultdict(lambda: {"images": {}, "detectors": {}})

        for img_key, image_raw in peaks_obj.image.ims.items():
            run_id = peaks_obj.get_run_id(img_key)
            det = peaks_obj.get_detector_by_img(img_key)
            try: match_key = int(img_key)
            except ValueError: match_key = img_key
            runs_plot_data[run_id]["images"][match_key] = np.nan_to_num(image_raw, nan=0.0, posinf=0.0, neginf=0.0)
            runs_plot_data[run_id]["detectors"][match_key] = det

        run_tasks = []
        base_dir = os.path.dirname(output_h5_filename) or "."

        # We no longer extract Empirical Zones, so we generate the Theoretical Predicted Zones for the plot
        viz_hkl = 2
        A_mat = np.linalg.inv(B_mat).T
        hc_vals = np.arange(-viz_hkl, viz_hkl + 1)
        hc, kc, lc = np.meshgrid(hc_vals, hc_vals, hc_vals, indexing="ij")
        hkl_c = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
        mask_hkl_c = ~((hkl_c[0] == 0) & (hkl_c[1] == 0) & (hkl_c[2] == 0))
        theo_indices = hkl_c[:, mask_hkl_c].astype(np.float32).T

        r_theo_zones = (A_mat @ theo_indices.T).T
        r_theo_zones_norm = r_theo_zones / np.linalg.norm(r_theo_zones, axis=1, keepdims=True)

        for r_id, data in runs_plot_data.items():
            mask = [i for i, run in enumerate(run_indices) if run == r_id]
            if len(mask) == 0: continue

            try: image_label = peaks_obj.get_image_label(img_indices[mask[0]])
            except Exception: image_label = f"run_{int(r_id)}"

            out_name = os.path.join(base_dir, f"{image_label}-direct_egnn.png")
            run_peaks = RunPeaks(
                image_index=[img_indices[i] for i in mask],
                peak_rows=[pixel_r[i] for i in mask], peak_cols=[pixel_c[i] for i in mask],
                var_u=None, var_v=None, cov_uv=None,
                sample_offset=ub_helper.sample_offset, ki_vec=ub_helper.ki_vec,
            )

            R_run = r_gonio_obs[mask[0]]

            # Draw the purely predicted zones to visually confirm the U-matrix fit
            pred_lab_zones_for_plot = (R_run @ U_final @ r_theo_zones_norm.T).T

            # Pass an empty array for the "empirical lines" since we don't extract lines anymore
            lab_zones_for_plot = np.empty((0, 3))

            run_tasks.append((
                out_name, run_peaks, data["images"], data["detectors"], instrument_name,
                lab_zones_for_plot, pred_lab_zones_for_plot,
            ))

        if max_workers := min(os.cpu_count() or 4, len(run_tasks)):
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
                futures = {executor.submit(_render_run_unrolled_plot, t): t[0] for t in run_tasks}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Rendering Detector Plots"):
                    try:
                        out_name = future.result()
                        print(f"Saved: {out_name}")
                    except Exception:
                        import traceback
                        print(f"Visualization failed for {out_name}:")
                        traceback.print_exc()

def _process_single_bank(args):
    """Parallel worker to parse and project a single detector bank."""
    import h5py, numpy as np, re
    from subhkl.instrument.detector import Detector
    from subhkl.config import beamlines, reduction_settings

    nexus_filename, key, instrument_name, sample_offset, ki_vec = args

    with h5py.File(nexus_filename, 'r') as f:
        match = re.match(r"bank(\d+)_events", key)
        if not match: return None
        bank_id = int(match.group(1))
        bank_str = str(bank_id)

        folder = f'/entry/{key}'
        if folder+'/event_id' not in f: return None

        event_id = f[folder+'/event_id'][:]
        event_index = f[folder+'/event_index'][:]
        event_time_offset = f[folder+'/event_time_offset'][:]
        event_time_zero = f[folder+'/event_time_zero'][:]

    if len(event_id) == 0: return None

    # Fast Unfold
    counts_per_pulse = np.diff(np.append(event_index, len(event_time_offset))).astype(int)
    absolute_time = np.repeat(event_time_zero, counts_per_pulse) + (event_time_offset * 1e-6)

    # Map Pixels
    det_config = beamlines[instrument_name][bank_str]
    det = Detector(det_config)
    settings = reduction_settings.get(instrument_name, {})

    offset = det_config.get("offset", 0)
    local_id = event_id - offset

    if settings.get("YAxisIsFastVaryingIndex"):
        pixel_c = local_id // det.n
        pixel_r = local_id % det.n
    else:
        pixel_c = local_id % det.m
        pixel_r = local_id // det.m

    # Fast Geometry Projection
    xyz = det.pixel_to_lab(pixel_r, pixel_c)
    kf = xyz - sample_offset[None, :]

    # Vectorized Norm
    kf_norm = np.sqrt(np.sum(kf**2, axis=1, keepdims=True))
    kf /= np.where(kf_norm == 0, 1.0, kf_norm)

    q_lab = kf - ki_vec[None, :]
    banks = np.full(len(absolute_time), bank_id, dtype=np.int16)

    # Cast arrays to smaller memory footprints for massive global sorting!
    return (
        q_lab.astype(np.float32),
        absolute_time,
        banks,
        pixel_r.astype(np.int16),
        pixel_c.astype(np.int16)
    )

def run_score_filter(
    finder_file: str,
    output_h5_filename: str,
    instrument_name: str | None = None,
    event_nexus_filename: str | None = None,
    streaming_callback = None,
    alpha = 0.01,
    J: int = 1024,
    n_lamb: int = 64,
    c: float = 0.15,
    window_size_events:int = 25000,
    step_size_events: int = 10000,
    seed_file: str | None = None,
    learning_rate: float = 0.02,
    eta: float = 0.05,
    sigma: float = 0.1,
    gamma: float = 1000.0,
    lamb: float = 0.8,
):
    from subhkl.optimization import FindUB, get_lattice_system
    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    import h5py
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax

    # ==========================================
    # LOAD GOLDEN SEED (If provided)
    # ==========================================
    true_6d_params = None
    if seed_file is not None:
        print(f"\n[DEBUG] Loading Golden Seed from: {seed_file}")
        with h5py.File(seed_file, "r") as f:
            if "sample/U" in f:
                U_true = f["sample/U"][()]
                # Map 3x3 Matrix to 6D continuous representation (first two columns)
                true_6d_params = np.concatenate([U_true[:, 0], U_true[:, 1]])
                true_6d_params = jnp.array(true_6d_params, dtype=jnp.float32)
                print("  > Seed successfully loaded and converted to 6D.")
            else:
                print("  > WARNING: 'sample/U' not found in seed file.")

    print(f"\n[1/4] Initializing Reciprocal Space from: {finder_file}")

    ub_helper = FindUB()
    with h5py.File(finder_file, "r") as f:
        ub_helper.a = f["sample/a"][()] if "sample/a" in f else None
        ub_helper.b = f["sample/b"][()] if "sample/b" in f else None
        ub_helper.c = f["sample/c"][()] if "sample/c" in f else None
        ub_helper.alpha = f["sample/alpha"][()] if "sample/alpha" in f else None
        ub_helper.beta = f["sample/beta"][()] if "sample/beta" in f else None
        ub_helper.gamma = f["sample/gamma"][()] if "sample/gamma" in f else None

        ub_helper.sample_offset = f["sample/offset"][()] if "sample/offset" in f else np.zeros(3)
        ub_helper.ki_vec = f["beam/ki_vec"][()] if "beam/ki_vec" in f else np.array([0.0, 0.0, 1.0])

        sg = f["sample/space_group"][()]
        ub_helper.space_group = sg.decode("utf-8") if isinstance(sg, bytes) else str(sg)
        ub_helper.wavelength = f["instrument/wavelength"][()]

    B_mat = ub_helper.reciprocal_lattice_B()
    B_inv = jnp.linalg.inv(B_mat)
    wavelengths = np.array(ub_helper.wavelength)
    lam_grid = jnp.logspace(jnp.log10(wavelengths[0]), jnp.log10(wavelengths[1]), n_lamb)

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

    # ==========================================
    # EVENT-MODE DATA ASSIMILATION LOGIC
    # ==========================================
    if event_nexus_filename:
        print(f"\n[2/4] Loading Event-Mode Data from: {event_nexus_filename}")
        import multiprocessing
        import concurrent.futures

        with h5py.File(event_nexus_filename, 'r') as f:
            keys = list(f['entry'].keys())

        args_list = [(event_nexus_filename, k, instrument_name, ub_helper.sample_offset, ub_helper.ki_vec) 
                     for k in keys if k.endswith('_events')]

        print(f"  > Dispatching {len(args_list)} banks to parallel workers...")
        all_q_lab, all_times, all_banks, all_pixels_r, all_pixels_c = [], [], [], [], []

        # Launch physical CPU cores to parse HDF5 in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for result in executor.map(_process_single_bank, args_list):
                if result is not None:
                    q, t, b, pr, pc = result
                    all_q_lab.append(q)
                    all_times.append(t)
                    all_banks.append(b)
                    all_pixels_r.append(pr)
                    all_pixels_c.append(pc)

        print("  > Aggregating and time-sorting global event stream...")
        all_q_lab = np.vstack(all_q_lab)
        all_times = np.concatenate(all_times)
        all_banks = np.concatenate(all_banks)
        all_pixels_r = np.concatenate(all_pixels_r)
        all_pixels_c = np.concatenate(all_pixels_c)
        
        sort_idx = np.argsort(all_times)
        all_q_lab = all_q_lab[sort_idx]
        all_times = all_times[sort_idx]
        all_banks = all_banks[sort_idx]
        all_pixels_r = all_pixels_r[sort_idx]
        all_pixels_c = all_pixels_c[sort_idx]
        
        print(f"  > Ready! Processed {len(all_times):,} total neutron events.")

    # Core JAX Functions
    def compute_U(rot_6d):
        a1 = rot_6d[:3]
        a2 = rot_6d[3:]
        b1 = a1 / (jnp.linalg.norm(a1) + 1e-6)
        b2 = a2 - jnp.dot(b1, a2) * b1
        b2 = b2 / (jnp.linalg.norm(b2) + 1e-6)
        b3 = jnp.cross(b1, b2)
        return jnp.stack([b1, b2, b3], axis=-1)

    def single_particle_loss(params, current_c, q_batch_unnorm):
        U_pred = compute_U(params)
        UB_inv = jnp.matmul(B_inv, U_pred.T)
        
        # Shape: (3, N)
        v = jnp.matmul(UB_inv, q_batch_unnorm)

        # 1. Unroll components for XLA fusion
        v_h = v[0, :]
        v_k = v[1, :]
        v_l = v[2, :]

        N_pts = v.shape[1]
        
        # We track the minimum distance and the best lambda
        initial_carry = jnp.zeros(N_pts)

        # 2. Sequential scan over lam_grid to avoid the massive VRAM tensor explosion
        def scan_body(carry, i):
            curr_best_lamb = carry
            lamda_cand = lam_grid[i]

            h_float = v_h / lamda_cand
            k_float = v_k / lamda_cand
            l_float = v_l / lamda_cand

            mean = lamda_cand * jnp.cos(jnp.pi * (h_float + k_float + l_float)) / n_lamb

            new_best_lamb = curr_best_lamb + mean

            return new_best_lamb, None

        # Execute the scan loop (JAX will perfectly fuse this on the GPU)
        best_lam_coarse, _ = jax.lax.scan(scan_body, initial_carry, jnp.arange(len(lam_grid)))

        # 3. TARGET LOCK: Extract nearest integer and STOP GRADIENTS
        hkl_int = jnp.round(v / best_lam_coarse[None, :])
        hkl_int_fixed = jax.lax.stop_gradient(hkl_int)

        # 4. ANALYTICAL PROJECTION
        num = jnp.sum(v**2, axis=0)
        den = jnp.sum(v * hkl_int_fixed, axis=0) + 1e-9
        lam_opt = jnp.clip(num / den, lam_grid[0], lam_grid[-1])

        # 5. EXACT FRACTIONAL COORDINATES
        hkl_float_exact = v / lam_opt[None, :]
        h = hkl_float_exact[0, :]
        k = hkl_float_exact[1, :]
        l = hkl_float_exact[2, :]

        # 6. CENTERING & LOSS
        h_p = M_prim[0, 0]*h + M_prim[0, 1]*k + M_prim[0, 2]*l
        k_p = M_prim[1, 0]*h + M_prim[1, 1]*k + M_prim[1, 2]*l
        l_p = M_prim[2, 0]*h + M_prim[2, 1]*k + M_prim[2, 2]*l

        dist_sq = (jnp.sin(jnp.pi * h_p)**2 + jnp.sin(jnp.pi * k_p)**2 + jnp.sin(jnp.pi * l_p)**2) / (jnp.pi**2)

        robust_penalty = dist_sq / (dist_sq + current_c**2)

        h_i, k_i, l_i = hkl_int_fixed[0, :], hkl_int_fixed[1, :], hkl_int_fixed[2, :]
        is_zero_hkl = (jnp.abs(h_i) + jnp.abs(k_i) + jnp.abs(l_i)) == 0

        # Force any pixel mapped to the direct beam to have the MAXIMUM possible penalty (1.0)
        robust_penalty = jnp.where(is_zero_hkl, 1.0, robust_penalty)

        loss = jnp.mean(robust_penalty)

        return loss, U_pred

    # ==========================================
    # PHASE 3: STREAMING EnSF LOOP
    # ==========================================
    if event_nexus_filename:
        print("\n[3/4] Executing Streaming Forecast-Analysis Cycle...")

        # Initial Swarm (Could be smaller for fast lock-in)
        J_ensf = J
        rng = jax.random.PRNGKey(42)
        ensf_params = jax.random.normal(rng, (J_ensf, 6))

        # ==========================================
        # INJECT THE SEED
        # ==========================================
        if true_6d_params is not None:
            # Overwrite Particle 0 with the Golden Seed
            ensf_params = ensf_params.at[0].set(true_6d_params)
            print("  > Golden Seed injected into EnSF Swarm Particle 0")

        @jax.jit
        def single_loss_weighted_vmf(single_params, U_swarm, losses_swarm, current_sigma, gamma):
            U_single = compute_U(single_params)
            sq_chordal_dists = jnp.sum((U_swarm - U_single)**2, axis=(-2, -1))
            logits = -sq_chordal_dists / (2.0 * current_sigma**2) - (gamma * losses_swarm)
            return jax.scipy.special.logsumexp(logits)

        prior_score_fn = jax.grad(single_loss_weighted_vmf, argnums=0)
        vmap_prior = jax.vmap(prior_score_fn, in_axes=(0, None, None, None, None))
        ensf_loss_fn = jax.vmap(single_particle_loss, in_axes=(0, None, None))

        opt_ensf = optax.adam(learning_rate)
        opt_state = opt_ensf.init(ensf_params)

        @jax.jit
        def train_step_streaming(params, opt_state, current_c, current_sigma, gamma, current_lambda, q_batch):
            def total_loss(p):
                losses, Us = ensf_loss_fn(p, current_c, q_batch)
                return jnp.mean(losses), (Us, losses)

            (mean_loss, (U_preds, individual_losses)), raw_grads = jax.value_and_grad(
                total_loss, has_aux=True
            )(params)

            prior_scores = vmap_prior(params, U_preds, individual_losses, current_sigma, gamma)
            total_grads = raw_grads - (current_lambda * prior_scores)

            updates, opt_state = opt_ensf.update(total_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, mean_loss, U_preds, individual_losses

        # Sliding Window Setup
        tracking_history = []

        # Fast Initial Lock-in (Phase 3A behavior on first window)
        q_init = jnp.array(all_q_lab[:window_size_events]).T
#        for _ in range(500):
#            ensf_params, opt_state, mean_loss, U_preds, losses = train_step_streaming(
#                ensf_params, opt_state, c, 0.10, 1000.0, 0.0, q_init # No gravity, independent search
#            )
#
#        print("  > Initial global orientation locked. Beginning sliding window tracker.")

        # Initialize memory variables before the loop
        smoothed_losses = None

        # The Streaming Loop
        for start_idx in range(0, len(all_q_lab) - window_size_events, step_size_events):
            end_idx = start_idx + window_size_events
            q_window = jnp.array(all_q_lab[start_idx:end_idx]).T

            # 1. FORECAST (Diffusion)
            rng, subkey = jax.random.split(rng)
            diffusion_noise = jax.random.normal(subkey, ensf_params.shape) * eta
            ensf_params = ensf_params + diffusion_noise

            # 2. ANALYSIS (EnSF Gravity Collapse)
            for _ in range(20): # Very fast update!
                ensf_params, opt_state, mean_loss, U_preds, losses = train_step_streaming(
                    ensf_params, opt_state, c, sigma, gamma, lamb, q_window
                )

            # ==========================================
            # APPLY EMA MEMORY FOR CONVERGENCE
            # ==========================================
            if smoothed_losses is None:
                smoothed_losses = losses
            else:
                smoothed_losses = (1.0 - alpha) * smoothed_losses + alpha * losses

            # Extract current mode based on the SMOOTHED losses
            best_idx = int(jnp.argmin(smoothed_losses))
            
            tracking_history.append((all_times[end_idx], np.array(U_preds[best_idx])))
            
            if start_idx % (step_size_events * 5) == 0:
                print(f"    Assimilation Time {all_times[end_idx]:.2f}s | Current Swarm Loss: {mean_loss:.4f}")

            # 3. TRIGGER CALLBACK
            if streaming_callback is not None:
                # To prevent double-counting in the sliding window, we only pass 
                # the step_size_events that are entirely new to this frame.
                new_start = 0 if start_idx == 0 else end_idx - step_size_events
                
                new_events = {
                    "banks": all_banks[new_start:end_idx],
                    "pixel_r": all_pixels_r[new_start:end_idx],
                    "pixel_c": all_pixels_c[new_start:end_idx]
                }
                
                streaming_callback(
                    time=float(all_times[end_idx]),
                    U_preds=np.array(U_preds),
                    losses=np.array(losses),          # Pass the raw batch losses to the callback
                    mean_loss=float(mean_loss),
                    best_idx=best_idx,                # This now points to the historically stable mode!
                    neutron_count=end_idx,
                    new_events=new_events
                )

        print(f"\n[4/4] Tracking complete. Extracted {len(tracking_history)} continuous U-matrices.")
        U_final = tracking_history[-1][1] # Use the final matrix for downstream saves
