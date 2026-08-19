import h5py
import numpy as np

# NOTE(Vivek): deprecate and use Goniometer class to handler rotation calc
from subhkl.instrument.goniometer import (
    get_rotation_data_from_nexus,
)
from subhkl.integration import Peaks
from subhkl.optimization import FindUB
from subhkl.io.export import ImageStackMerger, MTZExporter
from subhkl.viz import detector_assembly, replay

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
                    if (
                        "width" in calib_grp[bank_key]
                        and "height" in calib_grp[bank_key]
                    ):
                        beamlines[instrument][bank_id]["width"] = float(
                            calib_grp[bank_key]["width"][()]
                        )
                        beamlines[instrument][bank_id]["height"] = float(
                            calib_grp[bank_key]["height"][()]
                        )
                    count += 1
            if count > 0:
                print(f"Successfully applied calibration to {count} detector panels.")


def _file_frame_run_map(
    angles: np.ndarray, n_axes: int, file_offsets: np.ndarray
) -> tuple[bool, np.ndarray]:
    """Frame layout and frame -> run map for a peaks file's own angles.

    The optimizer's internal angle array cannot index the file being
    written: columns that are all identical collapse to one and re-tile
    only up to the last peak-bearing image, so its frame count falls
    short of the file's whenever trailing images carry no peaks
    (guaranteed in single-run files, where every frame shares one angle
    setting).  Corrections therefore address the file's frames
    directly: layout is decided by which dimension spans the goniometer
    axes (square arrays take the writer's frame-first convention), and
    the map is rebuilt from the run boundaries in ``file_offsets``.
    """
    angles = np.asarray(angles)
    if angles.ndim != 2 or n_axes not in angles.shape:
        raise ValueError(
            f"goniometer/angles shape {angles.shape} does not span "
            f"{n_axes} goniometer axes"
        )
    frame_first = angles.shape[1] == n_axes
    n_frames = angles.shape[0] if frame_first else angles.shape[1]
    frame_to_run = (
        np.searchsorted(np.asarray(file_offsets), np.arange(n_frames), side="right") - 1
    )
    return frame_first, frame_to_run


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
    refine_goniometer_axis_vector: list[str] | None = None,
    goniometer_axis_vector_bound_deg: float | list[float] | np.ndarray = 1.0,
    refine_goniometer_per_run: str | None = None,
    goniometer_per_run_bound_deg: float = 0.5,
    refine_goniometer_per_run_trans: bool = False,
    goniometer_per_run_trans_bound_meters: float = 0.002,
    refine_goniometer_harmonics: str | None = None,
    goniometer_harmonics_orders: list[int] | None = None,
    goniometer_harmonics_axes: str = "rocking",
    goniometer_harmonics_bound_deg: float = 0.5,
    refine_goniometer_trans_axes: list[str] | None = None,
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
    radial_weight: float = 1.0,
    radial_weight_poly: list[float] | None = None,
    hkl_metric: str = "isotropic",
    hkl_metric_floor: float = 0.1,
    multi_gpu: bool = False,
):
    input_data = input_data or {}

    if detector_modes is None:
        detector_modes = ["independent"]

    if detector_global_rot_axis is not None:
        if "global_rot" in detector_modes:
            print(
                f"Auto-switching detector mode: 'global_rot' -> 'global_rot_axis' (Axis: {detector_global_rot_axis})"
            )
            detector_modes = [
                "global_rot_axis" if mode == "global_rot" else mode
                for mode in detector_modes
            ]
    else:
        # Safe default fallback for downstream JAX compilation
        detector_global_rot_axis = [0.0, 1.0, 0.0]

    if cylinder_axis is not None:
        if "radial" in detector_modes:
            print(
                f"Auto-switching detector mode: 'radial' -> 'cylindrical' (Axis: {cylinder_axis})"
            )
            detector_modes = [
                "cylindrical" if mode == "radial" else mode for mode in detector_modes
            ]
    else:
        cylinder_axis = [0.0, 1.0, 0.0]  # Safe default for downstream JAX compilation

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
            bank_refined = np.zeros(len(pixel_r), dtype=bool)

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
                        bank_refined[mask] = True
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
                    "refined_mask": bank_refined.tolist(),
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
                input_data["goniometer/translations"] = b_f["goniometer/translations"][
                    ()
                ]
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
        print(
            f"Refining per-axis goniometer translations ({num_axes} axes) with bounds: {goniometer_trans_bound_meters} m."
        )
    if refine_beam:
        print(f"Refining beam tilt with {beam_bound_deg}° bounds.")
    if refine_goniometer_axis_vector:
        print(
            f"Refining goniometer axis vectors for {refine_goniometer_axis_vector} "
            f"with {goniometer_axis_vector_bound_deg} deg tilt bounds."
        )

    goniometer_names = None

    per_run_file_offsets = None
    per_run_files = None
    if original_nexus_filename and instrument_name:
        is_merged = False
        with h5py.File(original_nexus_filename, "r") as f_check:
            if "images" in f_check and "goniometer/axes" in f_check:
                is_merged = True
                axes = f_check["goniometer/axes"][()]
                angles = f_check["goniometer/angles"][()]
                per_run_file_offsets = (
                    f_check["file_offsets"][()] if "file_offsets" in f_check else None
                )
                per_run_files = (
                    [
                        x.decode() if isinstance(x, bytes) else str(x)
                        for x in f_check["files"][()]
                    ]
                    if "files" in f_check
                    else None
                )
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

    # A previous pass's refined axis vectors are this pass's nominal
    # geometry, exactly as offsets and translations already bootstrap.  The
    # nexus block above re-reads the nominal axes, so this override must
    # come after it; unrefined bootstraps carry the nominal axes and this
    # is a no-op.
    if bootstrap_filename and opt.goniometer_axes is not None:
        with h5py.File(bootstrap_filename, "r") as b_f:
            if "goniometer/axes" in b_f:
                boot_axes = np.asarray(b_f["goniometer/axes"][()], dtype=float)
                if boot_axes.shape == np.asarray(opt.goniometer_axes).shape:
                    opt.goniometer_axes = boot_axes
                    input_data["goniometer/axes"] = boot_axes
            # A previous pass's per-run-corrected angles likewise become
            # this pass's nominal angles (goniometer/angles in the
            # bootstrap is written per image, already corrected).
            if "goniometer/per_run" in b_f and "goniometer/angles" in b_f:
                boot_ang = np.asarray(b_f["goniometer/angles"][()], dtype=float)
                cur = np.asarray(opt.goniometer_angles)
                if boot_ang.shape == cur.shape:
                    opt.goniometer_angles = boot_ang
                    input_data["goniometer/angles"] = boot_ang
                elif boot_ang.T.shape == cur.shape:
                    opt.goniometer_angles = boot_ang.T
                    input_data["goniometer/angles"] = boot_ang

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

    per_run_frame_map = None
    if refine_goniometer_per_run or refine_goniometer_per_run_trans:
        if per_run_file_offsets is None:
            raise ValueError(
                "--refine-goniometer-per-run needs a merged --nexus file "
                "with file_offsets (the frame -> run bookkeeping)."
            )
        n_frames = int(np.asarray(opt.goniometer_angles).shape[-1])
        per_run_frame_map = (
            np.searchsorted(
                np.asarray(per_run_file_offsets), np.arange(n_frames), side="right"
            )
            - 1
        )

    num, hkl, lamda, U = opt.minimize(
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
        refine_goniometer_axis_vector=refine_goniometer_axis_vector,
        goniometer_axis_vector_bound_deg=goniometer_axis_vector_bound_deg,
        refine_goniometer_per_run=refine_goniometer_per_run,
        goniometer_per_run_bound_deg=goniometer_per_run_bound_deg,
        refine_goniometer_per_run_trans=refine_goniometer_per_run_trans,
        goniometer_per_run_trans_bound_meters=goniometer_per_run_trans_bound_meters,
        refine_goniometer_harmonics=refine_goniometer_harmonics,
        goniometer_harmonics_orders=goniometer_harmonics_orders,
        goniometer_harmonics_axes=goniometer_harmonics_axes,
        goniometer_harmonics_bound_deg=goniometer_harmonics_bound_deg,
        refine_goniometer_trans_axes=refine_goniometer_trans_axes,
        per_run_frame_map=per_run_frame_map,
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
        radial_weight=radial_weight,
        radial_weight_poly=radial_weight_poly,
        hkl_metric=hkl_metric,
        hkl_metric_floor=hkl_metric_floor,
        multi_gpu=multi_gpu,
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

        if (
            refine_goniometer_axis_vector
            and getattr(opt, "goniometer_axes_refined", None) is not None
        ):
            # Downstream stages (metrics, predictor, integrator) read
            # goniometer/axes from this file, so the refined vectors go
            # there; the nominal axes and the per-motor tilt angles are
            # kept alongside for provenance.
            if "goniometer/axes" in f:
                safe_write(f, "goniometer/axes_nominal", f["goniometer/axes"][()])
            safe_write(f, "goniometer/axes", opt.goniometer_axes_refined)
            grp_name = "goniometer/axis_tilts"
            if grp_name in f:
                del f[grp_name]
            tilts = opt.goniometer_axis_tilts
            if isinstance(tilts, dict):
                grp = f.create_group(grp_name)
                for k, v in tilts.items():
                    grp[k] = v
            elif tilts is not None:
                f[grp_name] = tilts

        if (
            refine_goniometer_per_run
            and getattr(opt, "goniometer_per_run_delta", None) is not None
        ):
            # Downstream stages read per-image goniometer/angles from this
            # file, so the corrected angles go there; nominal angles, the
            # per-run deltas, the image -> run map and the source run
            # files are kept alongside for provenance.
            delta = np.asarray(opt.goniometer_per_run_delta, dtype=float)
            ang = np.asarray(f["goniometer/angles"][()], dtype=float)
            names_axes = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in f["goniometer/names"][()]
            ]
            axis_cols = [
                i
                for i, n in enumerate(names_axes)
                if refine_goniometer_per_run.lower() in n.lower()
            ]
            frame_first, file_frame_map = _file_frame_run_map(
                ang, len(names_axes), per_run_file_offsets
            )
            corr = delta[file_frame_map]
            safe_write(f, "goniometer/angles_nominal", ang)
            for col in axis_cols:
                if frame_first:
                    ang[:, col] += corr
                else:
                    ang[col, :] += corr
            safe_write(f, "goniometer/angles", ang)
            grp_name = "goniometer/per_run"
            if grp_name in f:
                del f[grp_name]
            grp = f.create_group(grp_name)
            grp["motor"] = refine_goniometer_per_run
            grp["delta_deg"] = delta
            grp["frame_to_run"] = np.asarray(file_frame_map, dtype=np.int32)
            if per_run_files is not None:
                grp["run_files"] = np.array(per_run_files, dtype="S")
            if per_run_file_offsets is not None:
                grp["run_file_offsets"] = np.asarray(per_run_file_offsets)

        if (
            refine_goniometer_per_run_trans
            and getattr(opt, "goniometer_per_run_trans", None) is not None
        ):
            # Per-run sample displacements cannot be folded into any static
            # dataset the downstream stages read (unlike the corrected
            # angles above): they are recorded here for provenance, and
            # consumers need explicit support to apply them.
            grp_name = "goniometer/per_run"
            grp = f[grp_name] if grp_name in f else f.create_group(grp_name)
            if "trans_m" in grp:
                del grp["trans_m"]
            grp["trans_m"] = np.asarray(opt.goniometer_per_run_trans)
            if "frame_to_run" not in grp:
                ang = np.asarray(f["goniometer/angles"][()], dtype=float)
                _, file_frame_map = _file_frame_run_map(
                    ang, len(f["goniometer/names"]), per_run_file_offsets
                )
                grp["frame_to_run"] = np.asarray(file_frame_map, dtype=np.int32)
            if "run_files" not in grp and per_run_files is not None:
                grp["run_files"] = np.array(per_run_files, dtype="S")

        if (
            refine_goniometer_harmonics
            and getattr(opt, "goniometer_harmonics", None) is not None
        ):
            # The Fourier rocking has no angle representation (its axes
            # are not motors), so the coefficients travel with the file
            # and the predictor rebuilds the per-frame rotation from
            # them (subhkl.optimization.harmonic_rocking_matrices).
            gh = opt.goniometer_harmonics
            grp_name = "goniometer/harmonics"
            if grp_name in f:
                del f[grp_name]
            grp = f.create_group(grp_name)
            grp["motor"] = gh["motor"]
            grp["orders"] = np.asarray(gh["orders"], dtype=np.int32)
            grp["axes"] = np.asarray(gh["axes"], dtype=float)
            grp["coeffs_deg"] = np.asarray(gh["coeffs_deg"], dtype=float)

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

        f["peaks/h"] = hkl[:, 0]
        f["peaks/k"] = hkl[:, 1]
        f["peaks/l"] = hkl[:, 2]
        f["peaks/lambda"] = lamda

        if opt.x is not None and opt.x.size > 0:
            f["optimization/best_params"] = opt.x

        if bootstrap_filename:
            with h5py.File(bootstrap_filename, "r") as b_f:
                if "detector_calibration" in b_f:
                    b_f.copy("detector_calibration", f)

        import json

        flags = {
            "no_index": opt.no_index,
            "radial_weight": radial_weight,
            "radial_weight_poly": radial_weight_poly,
            "hkl_metric": hkl_metric,
            "hkl_metric_floor": hkl_metric_floor,
            "refine_lattice": refine_lattice,
            "refine_goniometer": refine_goniometer,
            "refine_goniometer_axis_vector": refine_goniometer_axis_vector,
            "refine_goniometer_per_run": refine_goniometer_per_run,
            "goniometer_per_run_bound_deg": goniometer_per_run_bound_deg,
            "refine_goniometer_per_run_trans": refine_goniometer_per_run_trans,
            "goniometer_per_run_trans_bound_meters": goniometer_per_run_trans_bound_meters,
            "refine_goniometer_harmonics": refine_goniometer_harmonics,
            "goniometer_harmonics_orders": goniometer_harmonics_orders,
            "goniometer_harmonics_axes": goniometer_harmonics_axes,
            "goniometer_harmonics_bound_deg": goniometer_harmonics_bound_deg,
            "refine_goniometer_trans_axes": refine_goniometer_trans_axes,
            "refine_goniometer_trans": refine_goniometer_trans,
            "refine_beam": refine_beam,
            "refine_detector": refine_detector,
            "refine_detector_banks": refine_detector_banks,
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
    finder_algorithm: str = "sparse_rbf",
    show_progress: bool = True,
    create_visualizations: bool = False,
    show_steps: bool = False,
    mask_file: str | None = None,
    mask_rel_erosion_radius: float | None = None,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    sparse_rbf_alpha: float | None = None,
    sparse_rbf_gamma: float = 0.0,
    sparse_rbf_min_sigma: float = 1.5,
    sparse_rbf_max_sigma: float | None = None,
    sparse_rbf_num_sigmas: int | None = None,
    sparse_rbf_false_alarms_per_image: float = 1.0,
    sparse_rbf_max_fragmentation_rate: float = 1.0,
    sparse_rbf_profile_file: str | None = "auto",
    sparse_rbf_shape_ratio: float = 1.2,
    sparse_rbf_shape_orientations: int = 4,
    sparse_rbf_chunk_size: int = 64,
    sparse_rbf_tile_rows: int = 2,
    sparse_rbf_tile_cols: int = 2,
    sparse_rbf_loss: str = "poisson",
    max_workers: int = 16,
    multi_gpu: bool = False,
    static_mask_file: str | None = None,
):
    print(f"Creating peaks from {filename} for instrument {instrument}")

    wavelength_kwargs = {}
    if wavelength_min:
        wavelength_kwargs["wavelength_min"] = wavelength_min
    if wavelength_max:
        wavelength_kwargs["wavelength_max"] = wavelength_max

    peaks = Peaks(filename, instrument, **wavelength_kwargs)

    if finder_algorithm != "sparse_rbf":
        raise ValueError(
            f"Unknown finder algorithm: {finder_algorithm!r} (peak_local_max "
            "and thresholding retired with the convex-hull stage)"
        )
    peak_kwargs = {"algorithm": finder_algorithm}
    peak_kwargs.update(
        {
            "alpha": sparse_rbf_alpha,
            "gamma": sparse_rbf_gamma,
            "min_sigma": sparse_rbf_min_sigma,
            "max_sigma": sparse_rbf_max_sigma,
            "num_sigmas": sparse_rbf_num_sigmas,
            "false_alarms_per_image": sparse_rbf_false_alarms_per_image,
            "max_fragmentation_rate": sparse_rbf_max_fragmentation_rate,
            "profile_file": sparse_rbf_profile_file,
            "shape_ratio": sparse_rbf_shape_ratio,
            "shape_orientations": sparse_rbf_shape_orientations,
            "chunk_size": sparse_rbf_chunk_size,
            "multi_gpu": multi_gpu,
            "static_mask_file": static_mask_file,
            "show_steps": show_steps,
            "show_scale": "linear",
            "tiles": (sparse_rbf_tile_rows, sparse_rbf_tile_cols),
            "loss": sparse_rbf_loss,
        }
    )

    peak_kwargs.update(
        {
            "mask_file": mask_file,
            "mask_rel_erosion_radius": mask_rel_erosion_radius,
        }
    )

    detector_peaks = peaks.get_detector_peaks(
        peak_kwargs,
        {},
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
        f.attrs["finder_algorithm"] = finder_algorithm

        # peaks/sigma is the finder's per-peak Gaussian width in pixels
        # (the finder measures no intensity, so no intensity sigma exists
        # to confuse it with); say so explicitly for readers.
        if "peaks/sigma" in f:
            f["peaks/sigma"].attrs["quantity"] = replay.WIDTH_QUANTITY

        with h5py.File(filename, "r") as f_in:
            for key in copy_keys:
                if key in f_in:
                    f_in.copy(f_in[key], f, key)


def run_metrics(
    file1: str,
    file2: str | None = None,
    instrument: str | None = None,
    d_min: float | None = None,
    per_peak: bool = False,
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
        per_peak=per_peak,
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

        gonio_offsets = None
        off_data = f_idx.get("goniometer/offsets")
        gonio_names = (
            f_idx["goniometer/names"][()] if "goniometer/names" in f_idx else None
        )

        if gonio_names is not None:
            gonio_names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in gonio_names
            ]

        if off_data is not None:
            gonio_offsets = np.zeros(
                len(gonio_names) if gonio_names else 1, dtype=np.float32
            )
            if isinstance(off_data, h5py.Group) and gonio_names is not None:
                for i, name in enumerate(gonio_names):
                    if name in off_data:
                        gonio_offsets[i] = float(off_data[name][()])
            else:
                raw_offs = off_data[()]
                gonio_offsets[: len(raw_offs)] = raw_offs

        if "goniometer/translations" in f_idx:
            sample_offset = f_idx["goniometer/translations"][()]
        else:
            sample_offset = np.zeros(3)

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

    if gonio_offsets is not None:
        print(f"Applying refined goniometer offsets from indexer: {gonio_offsets}")

    # The indexer's refined kinematics -- per-run-corrected angles and
    # refined axis vectors -- are written to its output file; the nexus
    # only carries the nominals.  Prefer the refined values whenever the
    # shapes line up, so prediction runs on the same geometry the
    # solution was fitted with.
    with h5py.File(indexed_hdf5_filename, "r") as f_idx:
        for key, attr in (
            ("goniometer/angles", "angles_raw"),
            ("goniometer/axes", "axes_raw"),
        ):
            if key not in f_idx:
                continue
            refined = np.asarray(f_idx[key][()], dtype=float)
            current = np.asarray(getattr(peaks.goniometer, attr))
            if refined.shape == current.shape and not np.allclose(refined, current):
                setattr(peaks.goniometer, attr, refined)
                print(f"Applying refined {key} from indexer.")
            elif refined.T.shape == current.shape and not np.allclose(
                refined.T, current
            ):
                setattr(peaks.goniometer, attr, refined.T)
                print(f"Applying refined {key} from indexer (transposed).")
        per_run_trans = None
        frame_to_run = None
        if "goniometer/per_run/trans_m" in f_idx:
            per_run_trans = f_idx["goniometer/per_run/trans_m"][()]
            frame_to_run = f_idx["goniometer/per_run/frame_to_run"][()]
            print(
                "Applying per-run sample displacements from indexer "
                f"({len(per_run_trans)} runs)."
            )
        harmonic_rot = None
        if "goniometer/harmonics" in f_idx:
            # The Fourier rocking steers q in the lab frame; its axes are
            # not motors, so it cannot ride on the corrected angles and is
            # rebuilt here from the stored coefficients, per frame.
            from subhkl.optimization import harmonic_rocking_matrices

            gh = f_idx["goniometer/harmonics"]
            harm_motor = gh["motor"][()]
            harm_motor = (
                harm_motor.decode("utf-8")
                if isinstance(harm_motor, bytes)
                else str(harm_motor)
            )
            harm_row = None
            if gonio_names is not None:
                for i, name in enumerate(gonio_names):
                    if harm_motor.lower() in name.lower():
                        harm_row = i
                        break
            if harm_row is None:
                raise ValueError(
                    f"Harmonic motor {harm_motor!r} not found in goniometer "
                    f"names {gonio_names}."
                )
            ang = np.asarray(peaks.goniometer.angles_raw, dtype=float)
            n_axes = len(gonio_names)
            frame_first, _ = _file_frame_run_map(ang, n_axes, np.zeros(1, dtype=int))
            harm_angles = ang[:, harm_row] if frame_first else ang[harm_row, :]
            harmonic_rot = harmonic_rocking_matrices(
                harm_angles, gh["axes"][()], gh["orders"][()], gh["coeffs_deg"][()]
            )
            print(
                f"Applying Fourier rocking from indexer ({harm_motor}, "
                f"orders {list(gh['orders'][()])}, "
                f"{harmonic_rot.shape[0]} frames)."
            )

    # Pass the Base UB matrix. The predictor will apply dynamic R_gonio internally!
    UB = U @ B

    results_map = peaks.predict_peaks(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        d_min,
        UB=UB,
        space_group=space_group,
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        max_workers=max_workers,
        # --- NEW GROUND TRUTH DELEGATION ---
        R_all=None,  # Force dynamic evaluation
        gonio_axes=peaks.goniometer.axes_raw,
        gonio_angles=peaks.goniometer.angles_raw,
        gonio_offsets=gonio_offsets,  # Pass the pure zero-points
        per_run_trans=per_run_trans,
        frame_to_run=frame_to_run,
        harmonic_rot=harmonic_rot,
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
        f["beam/ki_vec"] = ki_vec

        # --- SAVE RAW UNCORRECTED METADATA ---
        f["goniometer/angles"] = peaks.goniometer.angles_raw
        f["goniometer/axes"] = peaks.goniometer.axes_raw
        f["goniometer/translations"] = sample_offset
        if gonio_offsets is not None:
            f["goniometer/offsets"] = gonio_offsets

        if peaks.goniometer.names_raw:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "goniometer/names", data=peaks.goniometer.names_raw, dtype=dt
            )

        for img_key, (i, j, h, k, l, wl) in results_map.items():
            grp = f.create_group(f"banks/{img_key}")
            grp.create_dataset("i", data=i), grp.create_dataset("j", data=j)
            (
                grp.create_dataset("h", data=h),
                grp.create_dataset("k", data=k),
                grp.create_dataset("l", data=l),
            )
            grp.create_dataset("wavelength", data=wl)

        with h5py.File(indexed_hdf5_filename, "r") as f_in:
            if "detector_calibration" in f_in:
                f_in.copy("detector_calibration", f)


def run_rbf_integrator(
    filename: str,
    instrument: str,
    integration_peaks_filename: str,
    output_filename: str,
    sigmas: str = "1.0,2.0,4.0",
    nominal_sigma: float = 1.0,
    anisotropic: bool = False,
    fit_mosaicity: bool = False,
    mosaicity_radial: bool = False,
    shape_spherical: bool = False,
    mosaicity_bound_mrad: float = 10.0,
    shape_fit_min_snr: float = 0.0,
    shape_fit_normalized: bool = False,
    matrix_free_profile: str = "gaussian",
    matrix_free_fp_target: float | None = None,
    static_mask_file: str | None = None,
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
    print(f"Parameters: Sigma={sigma_list}")

    peak_dict = {}

    with h5py.File(integration_peaks_filename, "r") as f:
        angles_stack = f["goniometer/angles"][()] if "goniometer/angles" in f else None
        sample_offset = (
            f["goniometer/translations"][()]
            if "goniometer/translations" in f
            else np.zeros(3)
        )
        gonio_axes = f["goniometer/axes"][()] if "goniometer/axes" in f else None

        gonio_offsets = (
            f["goniometer/offsets"][()] if "goniometer/offsets" in f else None
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

    if angles_stack is None:
        angles_stack = peaks.goniometer.angles_raw

    one_image = next(iter(peaks.image.ims.values()))
    border_width = int(rel_border_width * min(one_image.shape[0], one_image.shape[1]))

    result = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict,
        peaks_obj=peaks,  # Pass the full Peaks object
        sigmas=sigma_list,
        nominal_sigma=nominal_sigma,
        show_progress=show_progress,
        all_R=None,
        sample_offset=sample_offset,
        anisotropic=anisotropic,
        fit_mosaicity=fit_mosaicity,
        mosaicity_radial=mosaicity_radial,
        shape_spherical=shape_spherical,
        mosaicity_bound_rad=mosaicity_bound_mrad * 1e-3,
        shape_fit_min_snr=shape_fit_min_snr,
        shape_fit_normalized=shape_fit_normalized,
        matrix_free_profile=matrix_free_profile,
        matrix_free_fp_target=matrix_free_fp_target,
        static_mask_file=static_mask_file,
        border_width=border_width,
        chunk_size=chunk_size,
        create_visualizations=create_visualizations,
        file_prefix=filename,
        max_workers=max_workers,
        gonio_axes=gonio_axes,
        gonio_angles=angles_stack,
        gonio_offsets=gonio_offsets,
    )

    print(f"Saving RBF integrated peaks to {output_filename}")
    with h5py.File(output_filename, "w") as f:
        f.attrs["instrument"] = instrument
        f["peaks/h"], f["peaks/k"], f["peaks/l"] = result.h, result.k, result.l
        f["peaks/lambda"] = result.wavelength
        f["peaks/intensity"], f["peaks/sigma"] = result.intensity, result.sigma
        f["peaks/two_theta"], f["peaks/azimuthal"] = result.tt, result.az
        f["peaks/bank"], f["peaks/run_index"] = result.bank, result.run_id

        # Where each peak sits on its detector image, and the shape the fit
        # gave it.  The integrator computes all of this to draw its own plots
        # and then dropped it on the way out, which left the plots impossible
        # to redraw afterwards; `integrator-visualize` reads it back.
        f["peaks/image_index"] = result.image_index
        f["peaks/pixel_r"], f["peaks/pixel_c"] = result.peak_rows, result.peak_cols
        f["peaks/var_u"], f["peaks/var_v"] = result.var_u, result.var_v
        f["peaks/cov_uv"] = result.cov_uv

        # Copy metadata
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
            "goniometer/offsets",  # <-- INCLUDED
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


def run_mtz_exporter(
    indexed_h5_filename: str,
    output_mtz_filename: str,
    space_group: str = None,
    predictions_file: str | None = None,
    corrections_file: str | None = None,
):
    algorithm = MTZExporter(
        indexed_h5_filename,
        space_group,
        predictions_file=predictions_file,
        corrections_file=corrections_file,
    )
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


def run_finder_visualize(
    images_filename: str,
    peaks_filename: str,
    instrument: str | None = None,
    output_dir: str | None = None,
    dpi: int = 150,
    n_sigma: float = detector_assembly.DEFAULT_N_SIGMA,
    max_workers: int | None = None,
    show_progress: bool = True,
):
    """Redraw the finder's unrolled-detector plots from its output file.

    Takes the same pair of files the finder itself worked from and wrote to,
    and reproduces the `-found.png` it would have produced with
    `--create-visualizations`, without repeating the peak search.
    """
    written = replay.replay_plots(
        images_filename=images_filename,
        peaks_filename=peaks_filename,
        suffix="-found",
        instrument=instrument,
        output_dir=output_dir,
        dpi=dpi,
        n_sigma=n_sigma,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    print(f"Wrote {len(written)} plot(s).")
    return written


def run_integrator_visualize(
    images_filename: str,
    peaks_filename: str,
    instrument: str | None = None,
    output_dir: str | None = None,
    dpi: int = 150,
    n_sigma: float = detector_assembly.DEFAULT_N_SIGMA,
    max_workers: int | None = None,
    show_progress: bool = True,
):
    """Redraw the RBF integrator's unrolled-detector plots from its output file.

    Reproduces the `-pred.png` that `rbf-integrator --create-visualizations`
    would have produced, without repeating the integration.
    """
    written = replay.replay_plots(
        images_filename=images_filename,
        peaks_filename=peaks_filename,
        suffix="-pred",
        instrument=instrument,
        output_dir=output_dir,
        dpi=dpi,
        n_sigma=n_sigma,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    print(f"Wrote {len(written)} plot(s).")
    return written


def run_static_mask(
    output_filename: str,
    input_filenames: list[str],
    peaks_filenames: list[str] | None = None,
    pooled_peaks_filename: str | None = None,
    peak_deviance_min: float = 9.488,
    peak_residual_max: float = 2.0,
    peak_clear_nsigmas: float = 3.5,
    min_frames: int = 5,
    smooth_sigma: float = 2.0,
    grad_nmads: float = 8.0,
    texture_factor: float = 0.15,
    wide_sigma: float = 20.0,
    edge_sigma: float = 25.0,
    dilate_px: int = 8,
    static_quantile: float = 25.0,
    grad_min_frac: float = 0.02,
):
    """Build a static-structure mask from reduced/merged frame stacks.

    See subhkl.search.static_mask for the estimator; the output is itself a
    reduced single-frame stack (1 = valid, 0 = masked, one frame per physical
    bank), so any tool that reads reduced files can display it, and the
    finder maps it onto its input by bank id (--static-mask-file).
    """
    from subhkl.search.static_mask import build_mask_file

    summary = build_mask_file(
        input_filenames,
        output_filename,
        peaks=peaks_filenames,
        pooled_peaks=pooled_peaks_filename,
        peak_deviance_min=peak_deviance_min,
        peak_residual_max=peak_residual_max,
        peak_clear_nsigmas=peak_clear_nsigmas,
        min_frames=min_frames,
        smooth_sigma=smooth_sigma,
        grad_nmads=grad_nmads,
        texture_factor=texture_factor,
        wide_sigma=wide_sigma,
        edge_sigma=edge_sigma,
        dilate_px=dilate_px,
        static_quantile=static_quantile,
        grad_min_frac=grad_min_frac,
    )
    print(
        f"Wrote {output_filename}: {len(summary['banks'])} bank(s), "
        f"{100 * summary['masked_fraction']:.2f}% of pixels masked."
    )
    if summary["thin_banks"]:
        print(
            f"Banks with fewer than {min_frames} distinct frames stay fully "
            "valid: " + ", ".join(str(b) for b in summary["thin_banks"])
        )
    if summary.get("n_exonerated"):
        print(
            f"Exonerated {summary['n_exonerated']} metric-certified peak "
            "footprint(s) from the static evidence."
        )
    if summary.get("n_exonerated_pooled"):
        print(
            f"Exonerated {summary['n_exonerated_pooled']} bank-level peak "
            "footprint(s) certified by the pooled (summed-frame) fit."
        )
    if summary["duplicates_dropped"]:
        n = sum(summary["duplicates_dropped"].values())
        print(
            f"WARNING: dropped {n} duplicate frame(s) (same goniometer "
            "orientation or identical content).  The estimator needs the "
            "sample to move between frames; repeats would promote true "
            "signal into the mask."
        )
    return summary


def run_sum_images(
    output_filename: str,
    input_filenames: list[str],
):
    """Sum each physical bank's deduplicated frames into a one-frame-per-bank
    stack.

    The companion of `static-mask`: a finder run on the summed stack sees the
    pooled evidence exactly as the static-map quantile does, so quasi-static
    reflections too faint for any single frame's certificate compound to
    certification there (deviance is additive across frames).  Feed that
    finder output back to `static-mask` as --pooled-peaks.
    """
    from subhkl.search.static_mask import build_summed_file

    summary = build_summed_file(input_filenames, output_filename)
    print(
        f"Wrote {output_filename}: {len(summary['banks'])} bank(s), "
        f"{sum(summary['n_frames'].values())} frame(s) summed."
    )
    if summary["duplicates_dropped"]:
        n = sum(summary["duplicates_dropped"].values())
        print(f"Dropped {n} duplicate frame(s) before summing.")
    return summary


def run_mask_visualize(
    images_filename: str,
    mask_filename: str,
    instrument: str | None = None,
    output_dir: str | None = None,
    dpi: int = 600,
    max_workers: int | None = None,
    show_progress: bool = True,
):
    """Draw the static mask over the frames it applies to (`<label>-mask.png`)."""
    written = replay.replay_mask_plots(
        images_filename=images_filename,
        mask_filename=mask_filename,
        instrument=instrument,
        output_dir=output_dir,
        dpi=dpi,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    print(f"Wrote {len(written)} plot(s).")
    return written
