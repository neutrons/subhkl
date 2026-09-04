import os

import h5py
import numpy as np

# NOTE(Vivek): deprecate and use Goniometer class to handler rotation calc
from subhkl.instrument.goniometer import (
    get_rotation_data_from_nexus,
)
from subhkl.integration import Peaks
from subhkl.io.export import ImageStackMerger, MTZExporter
from subhkl.optimization import FindUB
from subhkl.viz import detector_assembly, replay


def apply_detector_calibration(hdf5_filename: str, instrument: str):
    """
    Reads refined detector metrology from an indexer/prediction file (if present)
    and overrides the in-memory beamlines configuration so downstream
    tasks natively use the calibrated geometry.
    """
    import os

    from subhkl.config import beamlines

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
    ki_vec: list[float] | np.ndarray = None,
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
            if "goniometer/per_run/frame_to_run" in f_idx:
                per_run_trans = f_idx["goniometer/per_run/trans_m"][()]
                frame_to_run = f_idx["goniometer/per_run/frame_to_run"][()]
                print(
                    "Applying per-run sample displacements from indexer "
                    f"({len(per_run_trans)} runs)."
                )
            else:
                # a displacement per run is meaningless without the frame
                # -> run map; say so and predict without it rather than
                # dying on a KeyError deep in the setup
                print(
                    "WARNING: goniometer/per_run/trans_m without "
                    "frame_to_run; per-run sample displacements not applied."
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
            # The refined panels travel with the peaks they positioned: the
            # predictor already carries the group forward, and without the
            # same copy here integrator-visualize (and anything else reading
            # this file) has no way to know which geometry these pixel
            # coordinates belong to.
            if "detector_calibration" in f_in:
                f_in.copy("detector_calibration", f)


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
    import glob

    from subhkl.core.spacegroup import get_space_group_object

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


def _panel_offsets(dets, bank, pr, pc):
    """Peak-shaped instrument-refinement input from pixel coordinates.

    Returns (det_idx contiguous over the banks present, u_off, v_off [m],
    bank_list, detector_nominal dict) -- u_off/v_off are physical offsets
    from the panel center along uhat/vhat, the convention commands.py
    stores for the classic refinement, so both paths mean the same thing.
    """
    bank = np.asarray(bank, dtype=int)
    banks = sorted(int(b) for b in np.unique(bank))
    b2i = {b: i for i, b in enumerate(banks)}
    det_idx = np.array([b2i[int(b)] for b in bank])
    u_off = np.empty(len(bank))
    v_off = np.empty(len(bank))
    for b in banks:
        det = dets[b]
        m = bank == b
        xyz = det.pixel_to_lab(np.asarray(pr)[m], np.asarray(pc)[m])
        rel = xyz - np.asarray(det.center)
        u_off[m] = rel @ np.asarray(det.uhat)
        v_off[m] = rel @ np.asarray(det.vhat)
    nominal = {
        "centers": np.array([dets[b].center for b in banks], dtype=float),
        "uhats": np.array([dets[b].uhat for b in banks], dtype=float),
        "vhats": np.array([dets[b].vhat for b in banks], dtype=float),
        "widths": np.array([dets[b].width for b in banks], dtype=float),
        "heights": np.array([dets[b].height for b in banks], dtype=float),
    }
    return det_idx, u_off, v_off, banks, nominal


def _frame_angle_table(filenames, n_peaks=None):
    """Frame-shaped goniometer angles and the run boundaries, from the
    first file that carries them.

    The predictor, the integrator and the classic ``--bootstrap`` path all
    address goniometer angles PER FRAME (``goniometer/angles`` with one row
    per image, ``file_offsets`` marking the runs) -- a finder file's
    per-peak angle rows cannot carry a per-run correction to them.  Returns
    (angles, frame_to_run, files, file_offsets) or None.
    """
    for fn in filenames:
        if fn is None or not os.path.exists(fn):
            continue
        with h5py.File(fn, "r") as f:
            if "goniometer/angles" not in f or "file_offsets" not in f:
                continue
            ang = np.asarray(f["goniometer/angles"][()], dtype=float)
            offs = np.asarray(f["file_offsets"][()], dtype=int)
            files = f["files"][()] if "files" in f else None
            if ang.ndim != 2:
                continue
            n_frames = ang.shape[0]
            if n_frames <= offs[-1]:
                continue
            if n_peaks is not None and n_frames == n_peaks:
                # a per-peak angle table alongside run boundaries is
                # ambiguous; refuse rather than mislabel peaks as frames
                continue
            frame_to_run = np.searchsorted(offs, np.arange(n_frames), side="right") - 1
            return ang, frame_to_run, files, offs
    return None


def _calibrated_dets(dets, banks, panels):
    """Detector objects carrying refined panel geometry.

    ``panels`` is the ``panels`` block of refine_instrument_matching_free
    (centers/uhats/vhats/widths/heights, one row per bank in ``banks``).
    Banks the refinement did not see keep their nominal geometry.
    """
    from subhkl.instrument.detector import Detector

    out = dict(dets)
    for i, b in enumerate(banks):
        cfg = dict(dets[int(b)].config)
        cfg["center"] = np.asarray(panels["centers"][i], dtype=float).tolist()
        cfg["uhat"] = np.asarray(panels["uhats"][i], dtype=float).tolist()
        cfg["vhat"] = np.asarray(panels["vhats"][i], dtype=float).tolist()
        cfg["width"] = float(panels["widths"][i])
        cfg["height"] = float(panels["heights"][i])
        out[int(b)] = Detector(cfg)
    return out


def _sample_frame_directions(
    dets_use,
    bank_a,
    row_a,
    col_a,
    ki_use,
    run_a=None,
    R_by_run=None,
    R_rows=None,
    sample_offset=None,
):
    """Sample-frame Q directions for pixel rows, through a given geometry.

    The one place the spherical stage turns (bank, row, col) into a
    direction, so the search, the refinement's report and the quality
    pass cannot disagree about which instrument they mean.  Per-run
    sample displacements enter as a per-row ``sample_offset``.
    """
    import subhkl.search.spherical as sph_

    bank_a = np.asarray(bank_a, dtype=int)
    n = len(bank_a)
    d = np.zeros((n, 3))
    so = (
        None
        if sample_offset is None
        else np.broadcast_to(np.asarray(sample_offset, dtype=float), (n, 3))
    )
    for bk in np.unique(bank_a):
        m = bank_a == bk
        det = dets_use[int(bk)]
        xyz = det.pixel_to_lab(np.asarray(row_a)[m], np.asarray(col_a)[m])
        if so is not None:
            xyz = xyz - so[m]
        kf = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
        d[m] = sph_.ghat_from_kf(kf, ki=ki_use)
    if R_by_run is not None:
        run_a = np.asarray(run_a)
        for r_, R_ in R_by_run.items():
            m = run_a == r_
            if np.any(m):
                d[m] = d[m] @ np.asarray(R_, dtype=float)
    elif R_rows is not None:
        d = np.einsum("nji,nj->ni", np.asarray(R_rows, dtype=float), d)
    return d


def _lattice_proper_rotations(dirs, tol_deg=0.5):
    """Proper rotations of the lattice's Laue group, found numerically:
    candidates from the 24 cubic operations, kept when they map the model
    direction set onto itself.  Orthorhombic and lower keep the subset
    that survives; hexagonal settings fall back to the identity (warned by
    the caller if it matters)."""
    from itertools import product as _product

    cands = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for sg in _product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p_, s_) in enumerate(zip(perm, sg)):
                M[i, p_] = s_
            if np.linalg.det(M) > 0.5:
                cands.append(M)
    keep = []
    ctol = np.cos(np.deg2rad(tol_deg))
    sub = dirs[:: max(1, len(dirs) // 200)]
    for S in cands:
        rot = sub @ S.T
        if np.min(np.max(np.abs(rot @ dirs.T), axis=1)) > ctol:
            keep.append(S)
    return keep if keep else [np.eye(3)]


def _fit_gonio_offsets_from_runs(
    U_runs_lab, R_nominal, axes, angles_by_run, refine_mask, sym_ops, n_iter=4
):
    """Goniometer offsets from per-run orientations: the assignment-free
    analogue of the classic per-run DPHI corrections.

    Each run's lab-frame orientation U_r (from a per-run spherical solve,
    immune to the offsets by construction) should equal R_r(delta) U for a
    single crystal orientation U and per-axis offsets delta.  Linearize:
    the rotation residual rho_r = log(U_r (R_r(0) U)^T) is fit by the
    per-axis lab-frame generators a_kr = d/d delta_k [R_r] (numeric, per
    run) plus a global U correction, as one small least squares over
    3 n_runs equations; per-run lattice-symmetry alignment and the
    linearization are alternated.  Offsets on axes whose inner angles
    never vary are pure gauge with U (see refine_instrument_matching_free)
    and must not be in the mask.

    Returns (offsets_deg full-length, U, rms_residual_deg,
    {axis_index: identifiable}) -- an unidentifiable (gauge) axis keeps a
    ~zero offset and is reported as such, never as a fitted number.
    """
    from scipy.spatial.transform import Rotation as _Rot

    from subhkl.instrument.refinables import gonio_rotation_jax

    run_ids = sorted(U_runs_lab)
    K = len(axes)
    masked = [k for k in range(K) if refine_mask[k]]
    delta = np.zeros(K)

    def R_of(r, d):
        return np.asarray(
            gonio_rotation_jax(np.asarray(axes), np.asarray(angles_by_run[r]), d)
        )

    # initial global U: symmetry-aligned average against run 0's estimate
    U = R_of(run_ids[0], delta).T @ U_runs_lab[run_ids[0]]
    eps = 1e-4
    for _ in range(n_iter):
        rows, rhs = [], []
        for r in run_ids:
            Rr = R_of(r, delta)
            pred = Rr @ U
            # symmetry copy of this run's estimate closest to the prediction
            S = min(
                sym_ops,
                key=lambda S_: -np.trace(pred.T @ U_runs_lab[r] @ np.asarray(S_)),
            )
            rho = _Rot.from_matrix(U_runs_lab[r] @ np.asarray(S) @ pred.T).as_rotvec()
            J = np.zeros((3, len(masked) + 3))
            for j, k in enumerate(masked):
                # generator per radian of offset k at this run's setting
                dp = delta.copy()
                dp[k] += eps
                J[:, j] = _Rot.from_matrix(R_of(r, dp) @ Rr.T).as_rotvec() / np.deg2rad(
                    eps
                )
            J[:, len(masked) :] = Rr  # global U correction generators
            rows.append(J)
            rhs.append(rho)
        A = np.vstack(rows)
        b = np.concatenate(rhs)
        # Truncated SVD, and honesty about flat directions: an offset can
        # be structurally unidentifiable -- e.g. CG4D scans only phi with
        # kappa = 0, which makes the omega and phi axes collinear, so an
        # omega offset is exactly absorbable into U and ANY value for it is
        # a gauge choice (the production pipeline's -3.74 deg included).
        # A plain min-norm solve lets data systematics drag such a
        # parameter to an arbitrary confident-looking number; instead the
        # resolution matrix diagonal flags it and the truncation pins it.
        Um, sv, Vt = np.linalg.svd(A, full_matrices=False)
        keep_sv = sv > 0.05 * sv[0]
        x = (Vt[keep_sv].T / sv[keep_sv]) @ (Um[:, keep_sv].T @ b)
        resolution = np.einsum("ki,ki->i", Vt[keep_sv], Vt[keep_sv])
        delta[masked] += np.rad2deg(x[: len(masked)])
        U = _Rot.from_rotvec(x[len(masked) :]).as_matrix() @ U
        rms = np.rad2deg(np.sqrt(np.mean(b**2)))
    identifiable = resolution[: len(masked)] > 0.5
    return delta, U, rms, {k: bool(i) for k, i in zip(masked, identifiable)}


def _spherical_quality(
    dirs,
    U,
    pts,
    weights,
    run_ids,
    kernel_deg,
    floor=0.01,
    n_null=8,
    chunk=None,  # retired: the device kernel plans its own chunks
    seed=0,
    max_points=500_000,
    max_points_null=200_000,
):
    """Native quality report of a spherical indexing solution.

    Works identically with found peaks (unit weights) and raw counts
    (positive excess as weights) -- no hkl assignment anywhere:

    - deviation: the angle of each data direction to the NEAREST model
      line.  Its weighted median is the assignment-free analogue of the
      benchmark's median_ang_err, and identical to it wherever assignment
      is unambiguous (harmonics share a direction, so they never disagree).
    - null: the same median at random orientations.  Nearest-neighbour
      selection biases the deviation low as the model densifies, so the
      metric is only honest next to its own floor; report both, never one.
    - matched fraction: weight fraction within kernel_FWHM/2 of a line
      (and its null) -- completeness at the resolution actually claimed.
    - loglik: mean log of the von Mises mixture density per unit weight,
      the refinement's own objective -- the proper score for comparing
      candidates or twin fractions, in nats.

    Also returns the per-run median deviation: with the goniometer in the
    data path, a spread across runs is the signature of offset errors
    (measured on MANDI garnet: 0.14 deg per-run vs 1.63 deg multiframe
    through the nominal goniometer).
    """
    rng = np.random.default_rng(seed)
    # Raw mode brings tens of millions of weighted pixels, and each of the
    # 1 + n_null passes is a (points x model-lines) contraction -- minutes
    # at benchmark scale.  A uniform subsample (weights kept, so every
    # estimator below stays unbiased) caps that at seconds; peak-mode
    # inputs are far below the cap and untouched.
    if len(pts) > int(max_points):
        idx = rng.choice(len(pts), int(max_points), replace=False)
        pts = pts[idx]
        weights = weights[idx]
        if run_ids is not None:
            run_ids = run_ids[idx]
    sigma = np.deg2rad(kernel_deg) / np.sqrt(8.0 * np.log(2.0))
    tol = kernel_deg / 2.0

    # the null passes only need the floor's median and matched fraction --
    # a further unbiased subsample makes their 8 full sweeps cheap
    # (measured: 9 x 11.5 s of elementwise work at the 2M-point cap)
    if len(pts) > int(max_points_null):
        nidx = rng.choice(len(pts), int(max_points_null), replace=False)
        pts_null, w_null = pts[nidx], weights[nidx]
    else:
        pts_null, w_null = pts, weights
    # Every pass is the (points x model-lines) overlap the instrument
    # refinement evaluates, on the device through the same chunked
    # kernel (spherical.nearest_line_stats): the numpy form was a quarter
    # hour at L1 scale with the GPU idle.  The log-likelihood is the
    # refinement's objective exactly -- every direction summed, no
    # kernel cutoff.
    from subhkl.search.spherical import nearest_line_stats

    def dev_and_loglik(Umat, want_loglik=False, null=False):
        P, Wt = (pts_null, w_null) if null else (pts, weights)
        dev, dens = nearest_line_stats(
            P, dirs @ Umat.T, sigma, want_density=want_loglik
        )
        ll = 0.0
        if want_loglik:
            ll = float(np.sum(Wt * np.log(dens + floor)))
        return dev, ll / max(np.sum(Wt), 1e-12)

    def wmedian(x, w):
        o = np.argsort(x)
        cw = np.cumsum(w[o])
        return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])

    dev, loglik = dev_and_loglik(U, want_loglik=True)
    med = wmedian(dev, weights)
    matched = float(np.sum(weights[dev < tol]) / max(np.sum(weights), 1e-12))
    # weighted CDFs on a fixed grid, for the null-subtracted statistics
    grid = np.linspace(0.0, 10.0, 1001)
    wsum = max(np.sum(weights), 1e-12)
    F = np.cumsum(np.histogram(dev, bins=grid, weights=weights)[0]) / wsum
    F_null = np.zeros_like(F)
    null_meds, null_matched = [], []
    for _ in range(n_null):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        dv, _ = dev_and_loglik(Q * np.sign(np.linalg.det(Q)), null=True)
        null_meds.append(wmedian(dv, w_null))
        null_matched.append(
            float(np.sum(w_null[dv < tol]) / max(np.sum(w_null), 1e-12))
        )
        F_null += np.cumsum(np.histogram(dv, bins=grid, weights=w_null)[0]) / (
            max(np.sum(w_null), 1e-12) * n_null
        )
    # Null-subtracted aligned component.  In raw mode most of the weight is
    # diffuse scattering with no Bragg alignment, and the plain median then
    # reports the diffuse floor, not the indexing.  The diffuse component is
    # distributed (nearly) like the null, so it cancels in the CDF
    # difference: the height of max(F - F_null) is the weight fraction that
    # is genuinely aligned beyond chance, and the angle where the difference
    # reaches half its maximum is that component's median, undiluted.
    # (Thermal diffuse concentrating near the Bragg lines makes this an
    # upper bound on the aligned fraction, stated rather than hidden.)
    diff = F - F_null
    aligned_frac = float(np.max(diff))
    if aligned_frac > 0.02:
        half_idx = int(np.argmax(diff >= 0.5 * aligned_frac))
        aligned_med = float(grid[1:][half_idx])
    else:
        aligned_med = float("nan")
    # Saturation.  Two ways this median stops being a measurement: the
    # aligned component rises within one grid cell (nothing to read below
    # 0.01 deg), or the model set is so dense that a random orientation
    # already matches most of the weight -- on a 100 A cell at 2.5 A the
    # 65,000 model lines sit 0.32 deg apart, every direction has one within
    # 0.16 deg, and null_matched runs 0.75-0.91.  Reporting the grid step in
    # either case reads as 0.01 deg of accuracy, which is how a saturated
    # diagnostic came to be quoted as a result; say it instead.
    dense = float(np.mean(null_matched)) > 0.5
    unresolved = bool(
        np.isfinite(aligned_med) and aligned_med <= 2.0 * float(grid[1] - grid[0])
    )
    saturated = bool(dense or (unresolved and np.isfinite(aligned_med)))
    sat_why = (
        "the model set is denser than the tolerance"
        if dense
        else ("the aligned component rises inside one grid cell" if saturated else "")
    )
    if saturated:
        aligned_med = float("nan")
    per_run = {}
    if run_ids is not None:
        for r in np.unique(run_ids):
            m = run_ids == r
            per_run[int(r)] = wmedian(dev[m], weights[m])
    return {
        "median_deviation_deg": med,
        "null_median_deviation_deg": float(np.mean(null_meds)),
        "matched_fraction": matched,
        "null_matched_fraction": float(np.mean(null_matched)),
        "matched_tol_deg": tol,
        "loglik_per_weight": loglik,
        "aligned_fraction": aligned_frac,
        "aligned_median_deg": aligned_med,
        "aligned_median_saturated": saturated,
        "aligned_median_saturated_why": sat_why,
        "per_run_median_deg": per_run,
    }


def _read_merged_images(images_filename):
    """merged.h5 -> (images [N, n, m], bank_ids [N], run_of_image [N]).

    The merged file stores one 2D frame per (run, bank); file_offsets mark
    the run boundaries, so image i belongs to run r with
    file_offsets[r] <= i < file_offsets[r + 1].
    """
    with h5py.File(images_filename, "r") as fp:
        images = fp["images"][()]
        bank_ids = (
            fp["bank_ids"][()] if "bank_ids" in fp else np.zeros(len(images), int)
        )
        if "file_offsets" in fp:
            offs = np.asarray(fp["file_offsets"][()], dtype=int)
            run_of_image = (
                np.searchsorted(offs, np.arange(len(images)), side="right") - 1
            )
        else:
            run_of_image = np.zeros(len(images), dtype=int)
    return images, bank_ids, run_of_image


def _ewald_device_factory(G_c, run_ids, R_runs, dets, exc_maps, ki_hat, wl, b2):
    """The exact Ewald objective on the device for flat panels: for a batch
    of orientations, predict every in-band reflection's pixel on every
    bank of every run and gather the binned excess there.  exc_maps:
    {(run, bank): 2D excess}.  Returns (score_batch, polish) or None when a
    panel is not flat (the host path handles those)."""
    import jax
    import jax.numpy as jnp
    from scipy.spatial.transform import Rotation as Rot

    from subhkl.instrument.detector import DetectorShape
    from subhkl.search import spherical as sph

    banks = sorted({bk for _, bk in exc_maps})
    if any(dets[bk].panel_type != DetectorShape.flat_panel for bk in banks):
        return None
    runs = list(run_ids)
    H = max(e.shape[0] for e in exc_maps.values())
    Wd = max(e.shape[1] for e in exc_maps.values())
    E = np.zeros((len(runs), len(banks), H, Wd), np.float32)
    for (r_, bk), e in exc_maps.items():
        E[runs.index(r_), banks.index(bk), : e.shape[0], : e.shape[1]] = e
    cen = np.array([dets[bk].center for bk in banks], np.float32)
    uh = np.array([dets[bk].uhat for bk in banks], np.float32)
    vh = np.array([dets[bk].vhat for bk in banks], np.float32)
    nrm = np.cross(uh, vh).astype(np.float32)
    dw = np.array([dets[bk].width / (dets[bk].m - 1) for bk in banks], np.float32)
    dh = np.array([dets[bk].height / (dets[bk].n - 1) for bk in banks], np.float32)
    nrow = np.array([dets[bk].n for bk in banks], np.float32)
    ncol = np.array([dets[bk].m for bk in banks], np.float32)
    Rr = np.array([R_runs.get(r_, np.eye(3)) for r_ in runs], np.float32)
    Gc = jnp.asarray(np.asarray(G_c, np.float32))
    kih = jnp.asarray(np.asarray(ki_hat, np.float32))
    Ej = jnp.asarray(E)
    consts = tuple(jnp.asarray(x) for x in (cen, uh, vh, nrm, dw, dh, nrow, ncol, Rr))
    lo_l, hi_l = np.float32(wl[0]), np.float32(wl[1])
    b2f = np.float32(b2)

    @jax.jit
    def score_one(U):
        cen_, uh_, vh_, nrm_, dw_, dh_, nrow_, ncol_, Rr_ = consts
        total = 0.0
        for ir in range(len(runs)):
            G = (Gc @ U.T) @ Rr_[ir].T  # lab frame, this run
            qn = jnp.linalg.norm(G, axis=1)
            sth = -(G @ kih) / qn
            lam = 2.0 * sth / qn
            band = (lam >= lo_l) & (lam <= hi_l) & (sth > 0)
            kf = kih[None, :] + lam[:, None] * G
            kf = kf / jnp.linalg.norm(kf, axis=1, keepdims=True)
            dn = kf @ nrm_.T  # [N, Nb]
            t = (cen_ * nrm_).sum(1)[None, :] / jnp.where(jnp.abs(dn) < 1e-9, 1e-9, dn)
            p = t[:, :, None] * kf[:, None, :] - cen_[None, :, :]  # [N, Nb, 3]
            row = (p * vh_[None]).sum(2) / dh_[None]
            col = (p * uh_[None]).sum(2) / dw_[None]
            ok = (
                band[:, None]
                & (t > 0)
                & (row >= 0)
                & (col >= 0)
                & (row < nrow_[None])
                & (col < ncol_[None])
            )
            # bilinear in the binned map: a nearest-bin gather makes the
            # objective piecewise constant, so its maximum is a plateau half
            # a bin wide (0.19 deg at 4x4 on CG4D) and the polish stops
            # there -- measured as a 0.13 deg median residual against the
            # spots where the band-limited refinement reached 0.074.
            u = row / b2f - 0.5
            v = col / b2f - 0.5
            i0 = jnp.floor(u)
            j0 = jnp.floor(v)
            fu = u - i0
            fv = v - j0
            ib = jnp.broadcast_to(jnp.arange(len(banks))[None, :], u.shape)
            i0 = i0.astype(jnp.int32)
            j0 = j0.astype(jnp.int32)

            def gat(di, dj):
                return Ej[ir][
                    ib,
                    jnp.clip(i0 + di, 0, H - 1),
                    jnp.clip(j0 + dj, 0, Wd - 1),
                ]

            vals = (
                (1 - fu) * (1 - fv) * gat(0, 0)
                + (1 - fu) * fv * gat(0, 1)
                + fu * (1 - fv) * gat(1, 0)
                + fu * fv * gat(1, 1)
            )
            total = total + jnp.where(ok, vals, 0.0).sum()
        return total

    score_batch = jax.jit(jax.vmap(score_one))

    def score_np(Us):
        Us = np.asarray(Us, np.float32).reshape(-1, 3, 3)
        out = []
        for i in range(0, len(Us), 32):
            out.append(np.asarray(score_batch(jnp.asarray(Us[i : i + 32]))))
        return np.concatenate(out)

    def polish(U, spans=(0.6, 0.2, 0.07), n=7):
        U = np.asarray(U, np.float32)
        for span in spans:
            rv = (
                np.array(
                    np.meshgrid(
                        *[np.radians(np.linspace(-span, span, n))] * 3, indexing="ij"
                    )
                )
                .reshape(3, -1)
                .T
            )
            Rs = Rot.from_rotvec(rv).as_matrix().astype(np.float32) @ U
            U = Rs[int(np.argmax(score_np(Rs)))]
        return np.asarray(U, float)

    # Same reason the search kernels carry it: a float32 matmul on this
    # class of GPU is TF32 unless asked otherwise, and this objective is a
    # dot product of unit vectors read against a binned image.
    return sph._precise(score_np), sph._precise(polish)


def run_spherical_index(
    peaks_h5_filename: str,
    output_filename: str,
    d_min: float = 1.5,
    kernel_deg: float = 1.0,
    n_candidates: int = 4,
    refine: bool = True,
    refine_cell: bool = False,
    lam: float = 0.0,
    instrument_name: str | None = None,
    ki_vec: list | None = None,
    images_filename: str | None = None,
    binning: int = 4,
    runs: list | None = None,
    bandwidth: int | None = None,
    refine_instrument: bool = False,
    refine_gonio_axes: list | None = None,
    detector_modes: list | None = None,
    refine_detector_banks: list | None = None,
    det_trans_bound: float = 0.005,
    det_rot_bound_deg: float = 0.5,
    det_radial_bound_frac: float = 0.05,
    det_area_bound_frac: float = 0.05,
    det_global_rot_bound_deg: float = 2.0,
    det_global_trans_bound: float = 0.01,
    det_global_rot_axis: list | None = None,
    cylinder_axis: list | None = None,
    refine_gonio_axis_vector: list | None = None,
    gonio_axis_vector_bound_deg: float = 1.0,
    refine_gonio_per_run: str | None = None,
    gonio_per_run_bound_deg: float = 1.0,
    refine_gonio_harmonics: str | None = None,
    gonio_harmonics_orders: list | None = None,
    gonio_harmonics_bound_deg: float = 0.5,
    refine_beam: bool = False,
    beam_bound_deg: float = 1.0,
    refine_sample: bool = False,
    sample_bound_m: float = 0.005,
    refine_gonio_per_run_trans: bool = False,
    gonio_per_run_trans_bound_m: float = 0.005,
    frame_table_filename: str | None = None,
    gonio_bound_deg: float = 5.0,
    refine_maxiter: int = 400,
    refine_max_points: int = 100_000,
    fit_gonio_offsets: bool = False,
    model: str = "auto",
    nodal_max_index: int = 3,
    final_full_refine: bool = False,
    ewald_refine: bool = False,
    search: str = "lattice",
    static_mask_file: str | None = None,
    band_consistency: bool = True,
    radial: int = 24,
):
    """Index by spherical correlation over SO(3) -- the matched-filter dual
    of the sparse orientation-recovery problem (subhkl.search.spherical).

    Reads a finder output file, pools every bank's peaks as lab-frame Q
    directions (the Laue collapse: a spot fixes only the direction of Q),
    rotates them to the sample frame with the stored per-peak goniometer
    rotations, and finds the orientation(s) as peaks of the spherical
    cross-correlation against the cell's reflection directions -- global
    over all panels and all runs at once, multi-crystal capable, seconds on
    a CPU.  Optionally polishes with the matching-free refinement (no peak
    assignment; kernel overlap is the soft assignment).

    Writes a bootstrap file the existing indexer accepts via --bootstrap:
    sample/U, sample/B, the cell, beam/ki_vec -- plus the peaks, bank and
    goniometer groups (so indexer-visualize can draw zone overlays from
    this file alone) and a spherical/ group with every candidate's z-score,
    matched count, model coherence and sparse weight.  Validated on real
    CG4D garnet: U within 0.44 deg of the production indexer in a 4 s
    search (given the same refined goniometer; with nominal goniometer the
    difference is the omega offset, which is exactly the gauge the
    refinement stage owns).

    ``model`` picks the dictionary the search and the refinements see:
    "nodal" (default) is the crossings of the zone-axis great circles to
    |uvw| <= nodal_max_index, weighted by pair multiplicity (see
    spherical.nodal_points) -- a few thousand low-index directions with a
    mutual-coherence floor an order of magnitude below the full set's, so
    its basins separate at the default bandwidth and the (data x model)
    overlap of the instrument refinement shrinks with it; "reflections" is
    every reflection direction at d_min, the dictionary before nodal
    points existed.  ``final_full_refine`` keeps that full-list stage as an
    optional last step after the nodal solve: the instrument refinement
    (or, when it is off, one more orientation/cell pass) then sees every
    reflection, restoring the leverage of the high-index directions the
    nodal set leaves out.

    ``band_consistency`` (default on, needs instrument/wavelength): every
    spot is matched only against directions that could have produced it
    at some wavelength in the band -- the Laue collapse keeps a spot's
    direction and drops its |Q|, but along a ray the lattice only offers
    |G| = n g0, so a spot at |Ghat . ki| = sin(theta) is consistent with a
    direction iff some n g0 lies in [2 sin(theta) / lambda_max,
    min(2 sin(theta) / lambda_min, 1 / d_min)] (spherical.band_masks).  The
    search becomes a Fisher sum of per-band correlograms; measured, it
    lifts garnet's point-model z from 16.8 to 27.4 and shrinks a 100 A
    cell's legal model per spot from 65k to 19k directions.

    ``radial`` (N > 0) replaces the banded search by the 3D one: data as
    Laue segments and the model as shells at |G| on the resolution ball,
    N orthonormal radial functions (spherical.radial_basis), the band
    rule as one inner product and the model's radial position sharp
    however blurred its angular one.  Measured on garnet raw counts at
    L = 96: single stills z 10 -> 16 with rank 0 in every run (the 2D
    nodal default ranks 3-8 per still), 21 runs pooled z 17.3 -> 25.9.
    """
    from subhkl.config import beamlines
    from subhkl.core.crystallography import (
        cartesian_matrix_metric_tensor,
        generate_reflections,
    )
    from subhkl.instrument.detector import Detector
    from subhkl.search import spherical as sph

    # Persistent XLA compilation cache: profiling showed 6.5 s of a 29 s
    # CLI invocation compiling the same matching-free refinement kernels
    # every run.  The cache is keyed by compiler fingerprint, so a jax
    # upgrade simply misses once and refills.
    try:
        import os
        import sys

        if "jax" not in sys.modules:
            # the search loop is CPU-side; do not let the refinement's jax
            # client reserve 75% of every visible GPU for the whole run
            # (respects an explicit user setting)
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        import jax

        if jax.config.jax_compilation_cache_dir is None:
            jax.config.update(
                "jax_compilation_cache_dir",
                os.path.join(
                    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                    "subhkl",
                    "jax",
                ),
            )
    except Exception:
        pass

    with h5py.File(peaks_h5_filename, "r") as fp:
        bank = fp["bank"][()]
        pr = fp["peaks/pixel_r"][()]
        pc = fp["peaks/pixel_c"][()]
        img = fp["peaks/image_index"][()] if "peaks/image_index" in fp else None
        run = fp["peaks/run_index"][()] if "peaks/run_index" in fp else None
        Rg = fp["goniometer/R"][()] if "goniometer/R" in fp else None
        g_axes = fp["goniometer/axes"][()] if "goniometer/axes" in fp else None
        g_angles = fp["goniometer/angles"][()] if "goniometer/angles" in fp else None
        g_names = (
            [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in fp["goniometer/names"][()]
            ]
            if "goniometer/names" in fp
            else None
        )
        cell = tuple(
            float(fp[f"sample/{k}"][()])
            for k in ("a", "b", "c", "alpha", "beta", "gamma")
        )
        sg = fp["sample/space_group"][()]
        sg = sg.decode() if isinstance(sg, bytes) else str(sg)
        wl = fp["instrument/wavelength"][()] if "instrument/wavelength" in fp else None
        # a sample offset already measured (previous solve, either path) is
        # this run's nominal origin -- the refinement fits a delta on top
        in_sample_offset = (
            np.asarray(fp["goniometer/translations"][()], dtype=float)
            if "goniometer/translations" in fp
            else None
        )
        instrument = instrument_name or fp.attrs.get("instrument")

    if instrument is None:
        raise ValueError("no instrument in the peaks file; pass --instrument")
    ki = np.asarray(ki_vec if ki_vec is not None else [0.0, 0.0, 1.0], dtype=float)

    # Single-run (or run-subset) indexing: restrict the data side to the
    # chosen runs.  One run has one goniometer setting, so a per-run solve
    # is immune to goniometer-offset errors -- measured on MANDI garnet,
    # the median nearest-line deviation of run-0 peaks drops from 1.63 deg
    # (multiframe U under the nominal goniometer) to 0.144 deg (run 0
    # alone), and the per-run orientations are exactly the observable a
    # goniometer-offset calibration wants.
    if runs is not None:
        keep = np.isin(run, np.asarray(list(runs), dtype=int))
        if not np.any(keep):
            raise ValueError(f"no peaks in runs {runs}")
        bank, pr, pc = bank[keep], pr[keep], pc[keep]
        if img is not None:
            img = img[keep]
        if Rg is not None and Rg.shape[0] == len(keep):
            Rg = Rg[keep]
        run = run[keep]
        print(f"spherical-index: restricted to runs {sorted(set(int(r) for r in run))}")

    # A calibration already in the peaks file -- from a previous solve on
    # either path -- must reach the search, the refinement's starting point
    # and the overlays; every classic stage calls this, and without it a
    # known geometry is silently discarded and every direction is nominal.
    apply_detector_calibration(peaks_h5_filename, str(instrument))
    dets = {int(k): Detector(v) for k, v in beamlines[str(instrument)].items()}
    d_lab = np.zeros((len(pr), 3))
    for bk in np.unique(bank):
        m = bank == bk
        d_lab[m] = sph.panel_directions(dets[int(bk)], rows=pr[m], cols=pc[m], ki=ki)
    if Rg is not None:
        R_peak = Rg if Rg.shape[0] == len(pr) else Rg[img]
        d_sample = np.einsum("nji,nj->ni", R_peak, d_lab)
    else:
        d_sample = d_lab

    # Raw-count mode: every pixel votes with its excess counts -- no peak
    # finding in the loop (subhkl.search.spherical.project_counts).  The
    # peaks file still supplies the metadata (instrument, cell, per-run
    # goniometer); its peaks are simply not used as the data side.  Pixels
    # are binned (default 4x4) before projection: the direction across a
    # bin changes by far less than the angular kernel, and the cost drops
    # by binning^2.  The per-image background is the scalar zero-fraction
    # estimator mu = -log(P(y = 0)) -- the sparse-regime rate estimator of
    # compute_rate_batch collapsed to one number per frame; subtracting it
    # is what cancels the anisotropic detector acceptance in expectation.
    import time as _time

    _t = {"start": _time.perf_counter()}

    def _mark(name):
        now = _time.perf_counter()
        _t[name] = _t.get(name, 0.0) + (now - _t["start"])
        _t["start"] = now

    ewald_score = None
    ewald_polish = None
    raw_mode = images_filename is not None
    f_raw = None
    f_raw_bands = None
    f_raw_radial = None
    band_edges = None
    ki_hat = ki / np.linalg.norm(ki)
    if images_filename is not None:
        images, bank_ids, run_of_image = _read_merged_images(images_filename)
        static_valid = None
        if static_mask_file is not None:
            # The classic stages all take this mask and the raw path did not,
            # which is how cg4d-l1-mbl came to index on detector structure:
            # its 5 sigma bins repeat between runs at 22-36x the chance rate
            # (fixed features, not scattering) and 62% of them sit on pixels
            # this mask already calls bad.  Carried as a validity stack, not
            # folded into the counts: zeroing a masked pixel would make it a
            # "dead" pixel to the zero-fraction test, and 15% of the panel
            # zeroed flips the dense regime into the sparse one, whose
            # background is then meaningless (measured: 2947 -> 41995 spots).
            from subhkl.search.static_mask import load_mask_for_banks

            static_valid = (
                load_mask_for_banks(
                    static_mask_file, [int(b) for b in bank_ids], images.shape[1:]
                )
                > 0
            )
            print(
                "spherical-index: static mask applied, "
                f"{100.0 * (1.0 - static_valid.mean()):.1f}% of pixels masked"
            )
        _mark("read")
        if runs is not None:
            # the --runs restriction applies to the frames too: without it,
            # frames of unselected runs would be projected with no (or the
            # wrong) goniometer rotation and drown the selected run's signal
            keep_img = np.isin(run_of_image, np.asarray(list(runs), dtype=int))
            images = images[keep_img]
            bank_ids = bank_ids[keep_img]
            run_of_image = run_of_image[keep_img]
        # per-run goniometer rotation from any of the run's peaks
        run = run if run is not None else np.zeros(len(pr), int)
        R_run = {}
        if Rg is not None:
            R_pk = Rg if Rg.shape[0] == len(pr) else Rg[img]
            for r_ in np.unique(run):
                R_run[int(r_)] = R_pk[np.argmax(run == r_)]
        sigma_raw = np.deg2rad(kernel_deg) / np.sqrt(8.0 * np.log(2.0))
        # The auto ceiling of 96 is a cost cap, not a precision floor:
        # measured on MANDI garnet run-0 raw data, the solution moves by
        # < 0.02 deg between L = 96 and L = 192 (at 3x and 9x the cost) --
        # refinement locates the smooth objective's peak far below the
        # bandwidth.  Raise it (--bandwidth) when a dense model needs basin
        # SEPARATION, e.g. large cells with direction spacing under pi/L.
        L_raw = (
            int(bandwidth)
            if bandwidth is not None
            else int(min(max(np.ceil(3.0 / sigma_raw), 16), 96))
        )
        _h, _k, _l = generate_reflections(*cell, space_group=sg, d_min=d_min)
        _Bc, _ = cartesian_matrix_metric_tensor(*cell[:3], *np.deg2rad(cell[3:]))
        _G = np.stack([_h, _k, _l], axis=1) @ _Bc.T
        if bandwidth is None and (search != "lattice" or refine_instrument):
            # ...unless the dictionary is denser than that resolution: then
            # the model is uniform at L and the search blind (MANDI L1).
            L_raw, _spacing = sph.dictionary_bandwidth(L_raw, _G)
            print(
                f"spherical-index: dictionary spacing {_spacing:.2f} deg at "
                f"d_min={d_min:g} -> L = {L_raw}"
            )
        bin_dirs = {}
        bin_rows, bin_cols = {}, {}
        per_bank_rows = {}
        per_bank_sig = {}
        raw_pts, raw_w, raw_run = [], [], []
        raw_bank, raw_prow, raw_pcol = [], [], []
        bin_sth, raw_sth = {}, []
        dense_noted = False
        b2 = int(binning)
        for i_img in range(len(images)):
            bk = int(bank_ids[i_img])
            det = dets.get(bk)
            if det is None:
                continue
            im = images[i_img].astype(float)
            vd = None if static_valid is None else static_valid[i_img]
            n2, m2 = (im.shape[0] // b2) * b2, (im.shape[1] // b2) * b2
            y = im[:n2, :m2].reshape(n2 // b2, b2, m2 // b2, b2).sum(axis=(1, 3))
            zero_frac = float(
                np.mean(im == 0) if vd is None else np.mean(im[vd] == 0)
            )
            if zero_frac >= 0.05:
                # sparse regime: zeros are Poisson draws of the background
                # and their fraction estimates it (P(0) = e^-mu)
                mu = -np.log(max(zero_frac, 1e-6)) * b2 * b2  # [counts/bin]
                excess = (y - mu).ravel()
                sig_bin = np.full(excess.shape, np.sqrt(max(mu, 1.0)))
            else:
                # dense regime (mu > 3 counts/pixel): the only zeros are dead
                # pixels, so the zero fraction reads the panel border, not
                # the background -- on MANDI L1 it gave 56 counts/bin
                # against a true 400 and the projected weight was the
                # acceptance.  A local median over the live bins is the
                # background and the excess is y - bg, zero-mean like the
                # sparse branch's y - mu: the noise averages out in the
                # projection.  (A 2 sigma "floor" subtracted here put every
                # background bin at a systematic -2 sigma -- a biased panel
                # footprint that left the 2D search at z 6.6 on the dense
                # test scene and broke the 3D search outright, whose lowest
                # radial channel sums all of it coherently; unbiased, both
                # index at z ~30.)  Bins whose median window touches a dead
                # region are dropped: a panel edge is a great circle on the
                # sphere and a rim of fake excess along it is a fake zone.
                from scipy.ndimage import binary_erosion, generic_filter

                blk = im[:n2, :m2].reshape(n2 // b2, b2, m2 // b2, b2)
                live = (blk > 0).all(axis=(1, 3))
                if vd is not None:
                    live &= vd[:n2, :m2].reshape(
                        n2 // b2, b2, m2 // b2, b2
                    ).all(axis=(1, 3))
                bg = generic_filter(
                    np.where(live, y, np.nan), np.nanmedian, size=9, mode="nearest"
                )
                keep = binary_erosion(live, iterations=5)
                exc = np.nan_to_num(y - bg)
                excess = np.where(keep, exc, 0.0).ravel()
                sig_bin = np.sqrt(np.clip(np.nan_to_num(bg), 1.0, None)).ravel()
                if not dense_noted:
                    print(
                        "spherical-index: dense background (zero fraction "
                        f"{zero_frac:.3f}); local-median background, panel "
                        "rims masked"
                    )
                    dense_noted = True
            if bk not in bin_dirs:
                rr = (np.arange(n2 // b2) + 0.5) * b2 - 0.5
                cc = (np.arange(m2 // b2) + 0.5) * b2 - 0.5
                RR, CC = np.meshgrid(rr, cc, indexing="ij")
                bin_rows[bk], bin_cols[bk] = RR.ravel(), CC.ravel()
                bin_dirs[bk] = sph.panel_directions(
                    det, rows=RR.ravel(), cols=CC.ravel(), ki=ki
                )
                bin_sth[bk] = np.abs(bin_dirs[bk] @ ki_hat)
            r_ = int(run_of_image[i_img])
            per_bank_rows.setdefault(bk, []).append((r_, excess))
            per_bank_sig.setdefault(bk, []).append((r_, sig_bin))
            d = bin_dirs[bk]
            Rr = R_run.get(r_)
            raw_pts.append(d @ Rr if Rr is not None else d)  # R^T per row
            raw_w.append(np.clip(excess, 0.0, None))
            raw_sth.append(bin_sth[bk])
            raw_run.append(np.full(len(d), r_))
            raw_bank.append(np.full(len(d), bk))
            raw_prow.append(np.tile(bin_rows[bk], 1))
            raw_pcol.append(np.tile(bin_cols[bk], 1))
        _mark("bin")

        # The frame-proportional cost is the associated-Legendre evaluation,
        # and a bank's binned geometry is FIXED in the lab frame -- so the
        # projection runs once per bank with every frame batched into one
        # set of BLAS contractions, and each run's coefficients are rotated
        # afterwards (rotate_coeffs, O(L^3) per run).  The rotated-points
        # form this replaces re-evaluated the basis per frame
        # (O(L^2 n_bins) each): measured ~1.5 s/frame at benchmark scale,
        # half an hour of projection for cg4d-garnet's 1114 frames.  The
        # equivalence rests on the coefficient-rotation identity the
        # convention tests pin at 1e-12.
        from subhkl.search.spherical import project_counts_device

        f_lab_run = {}
        if search != "lattice" or refine_instrument:
            # the lattice ladder works on spots; no harmonic projection,
            # no Wigner pass (a third of a still's wall time, most of a
            # pooled run's) -- unless the matching-free instrument
            # refinement needs the map
            # Band-consistency: the bins are stratified by sin(theta) and each
            # stratum projected on its own, so the search can hold every band
            # to the directions that could have produced it.  The pooled map
            # (every stage downstream) is the sum of the strata.
            use_radial = int(radial) > 0 and wl is not None
            use_band = bool(band_consistency) and wl is not None and not use_radial
            if use_band:
                band_edges = sph.sin_theta_edges(
                    np.concatenate(raw_sth), np.concatenate(raw_w), n_bands=4
                )
            n_bands_raw = len(band_edges) - 1 if use_band else 1
            n_chan = int(radial) if use_radial else 0
            q_max_r = 1.0 / float(d_min)
            f_lab_band = {}
            # pad every bank to the same frame count so the device kernel
            # compiles once (zero rows project to zero and are discarded)
            nf_max = max(len(rows) for rows in per_bank_rows.values())
            for bk, rows in per_bank_rows.items():
                W = np.zeros((nf_max, len(rows[0][1])))
                for i_, (_, e_) in enumerate(rows):
                    W[i_] = e_
                if use_radial:
                    # every frame x channel as one row of a single projection:
                    # the segment weights depend on the bin's sin(theta) only
                    seg = sph.radial_data_weights(bin_sth[bk], wl, q_max_r, n_chan)
                    W3 = (W[:, None, :] * seg.T[None, :, :]).reshape(-1, W.shape[1])
                    F3 = project_counts_device(
                        bin_dirs[bk], W3, L_raw, sigma_raw
                    ).reshape(len(W), n_chan, L_raw + 1, 2 * L_raw + 1)
                    F = project_counts_device(bin_dirs[bk], W, L_raw, sigma_raw)
                    for (r_, _), Fi, F3i in zip(rows, F, F3):
                        f_lab_run[r_] = f_lab_run.get(r_, 0.0) + Fi
                        for c_ in range(n_chan):
                            f_lab_band[(r_, c_)] = (
                                f_lab_band.get((r_, c_), 0.0) + F3i[c_]
                            )
                    continue
                for b_ in range(n_bands_raw):
                    if use_band:
                        m_b = (bin_sth[bk] > band_edges[b_]) & (
                            bin_sth[bk] <= band_edges[b_ + 1]
                        )
                        if not m_b.any():
                            continue
                        F = project_counts_device(
                            bin_dirs[bk], W * m_b[None, :], L_raw, sigma_raw
                        )
                    else:
                        F = project_counts_device(bin_dirs[bk], W, L_raw, sigma_raw)
                    for (r_, _), Fi in zip(rows, F):
                        f_lab_run[r_] = f_lab_run.get(r_, 0.0) + Fi
                        f_lab_band[(r_, b_)] = f_lab_band.get((r_, b_), 0.0) + Fi
            _mark("project")
            # one batched Wigner build for every run's rotation instead of a
            # scalar build per run
            run_keys = sorted(f_lab_run)
            rot_keys = [r_ for r_ in run_keys if R_run.get(r_) is not None]
            if rot_keys:
                eulers = np.array([sph.euler_zyz(R_run[r_].T) for r_ in rot_keys])
                d_all = sph.wigner_d_matrix(L_raw, eulers[:, 1])  # [l, m, n, r]
                m_ax = np.arange(-L_raw, L_raw + 1)
                ea = np.exp(-1j * np.outer(eulers[:, 0], m_ax))  # [r, m]
                eg = np.exp(-1j * np.outer(eulers[:, 2], m_ax))  # [r, n]

            # every map -- the pooled one and each band -- rotated in ONE pass:
            # the Wigner stack is the expensive operand and it is shared
            keys = [None] + (
                list(range(n_chan))
                if use_radial
                else (list(range(n_bands_raw)) if use_band else [])
            )
            zero = np.zeros((L_raw + 1, 2 * L_raw + 1), dtype=complex)

            def _get(r_, b_):
                return (
                    f_lab_run.get(r_, zero)
                    if b_ is None
                    else f_lab_band.get((r_, b_), zero)
                )

            pooled = np.zeros((len(keys), L_raw + 1, 2 * L_raw + 1), dtype=complex)
            for r_ in run_keys:
                if r_ not in rot_keys:
                    for k_, b_ in enumerate(keys):
                        pooled[k_] += _get(r_, b_)
            if rot_keys:
                f_stack = np.stack(
                    [[_get(r_, b_) for b_ in keys] for r_ in rot_keys]
                )  # [r, k, l, n]
                pooled += np.einsum(
                    "rm,lmnr,rn,rkln->klm", ea, d_all, eg, f_stack, optimize=True
                )
            f_raw = pooled[0]
            if use_band:
                f_raw_bands = [pooled[1 + b_] for b_ in range(n_bands_raw)]
            f_raw_radial = pooled[1:] if use_radial else None
            _mark("rotate")
            print(
                f"spherical-index: raw-count mode, {len(images)} frames binned "
                f"{b2}x{b2} -> f at L={L_raw}"
            )

    a, b, c_, al, be, ga = cell
    B, _ = cartesian_matrix_metric_tensor(a, b, c_, *np.deg2rad([al, be, ga]))
    h_, k_, l_ = generate_reflections(a, b, c_, al, be, ga, space_group=sg, d_min=d_min)
    G = np.stack([h_, k_, l_], axis=1) @ B.T
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    key = np.round(dirs * np.where(dirs[:, [0]] < -1e-9, -1, 1), 5)
    _, idx = np.unique(key, axis=0, return_index=True)
    dirs = dirs[idx]
    hkl_all = np.stack([h_, k_, l_], axis=1)
    # The dictionary.  Nodal points enter every stage as INTEGER triples
    # with weights, so the strain a cell refinement applies to B moves them
    # exactly as it moves the reflections; the search itself only ever
    # sees the nominal cell (a coarse stage), and refinement takes it from
    # there.  A partial dictionary is polished matching-free ("local"):
    # assigning every datum to its nearest node would bias the fit.
    #
    # 'auto': with the 3D radial search the shells at |G| already carry the
    # wavelength consistency the nodal points supplied in 2D, and the plain
    # reflection set is the better dictionary (cg4d-garnet, 21 stills:
    # CC(1/2) 0.9999 vs 0.9994 for radial + nodal); without it (no band on
    # the data, or --radial 0) the nodal set is (0.9998 vs 0.9987).
    if model == "auto":
        model = "reflections" if int(radial) > 0 and wl is not None else "nodal"
    if model == "nodal":
        hkl_model, w_model = sph.nodal_points(B, max_index=nodal_max_index)
        G_m = hkl_model @ B.T
        dirs_model = G_m / np.linalg.norm(G_m, axis=1, keepdims=True)
        # Only the nodes the data can show.  A node is a lattice direction,
        # but one whose every in-band harmonic lies beyond d_min or is a
        # systematic absence never receives a peak: it would add mass to
        # the model (and coherence to the dictionary) and no signal.  On
        # garnet at 1.2 A this keeps 577 of the 3,217 nodes to |uvw| <= 3.
        n_nodes = len(hkl_model)
        seen = np.max(np.abs(dirs_model @ dirs.T), axis=1) > np.cos(np.deg2rad(0.01))
        hkl_model, w_model, dirs_model = (
            hkl_model[seen],
            w_model[seen],
            dirs_model[seen],
        )
        if len(hkl_model) < 4:
            raise ValueError(
                f"only {len(hkl_model)} nodal points are observable at "
                f"d_min={d_min:g}; raise --nodal-max-index or use --model reflections"
            )
        refine_method = "local"
    elif model == "reflections":
        hkl_model, w_model, dirs_model = hkl_all, None, dirs
        refine_method = "wahba"
    else:
        raise ValueError(
            f"model must be 'nodal', 'reflections' or 'auto', got {model!r}"
        )
    full_final = bool(final_full_refine) and model != "reflections"
    hkl_inst, w_inst = (hkl_all, None) if full_final else (hkl_model, w_model)
    # primitive spacing along every search direction, for the band rule
    g0_model = sph.primitive_spacing(
        hkl_all[idx] if model == "reflections" else hkl_model, B
    )
    use_radial = int(radial) > 0 and wl is not None
    use_band = bool(band_consistency) and wl is not None and not use_radial
    model_shells = None
    if use_radial:
        # the 3D model: every reflection at its |G|, carrying its ray's
        # dictionary weight (1 for the reflection set; a node's
        # multiplicity, or 0 off the nodes, for the nodal set)
        G_all = hkl_all @ B.T
        q_all = np.linalg.norm(G_all, axis=1)
        d_all = G_all / q_all[:, None]
        if model == "reflections":
            w_all = np.ones(len(hkl_all))
        else:
            gcd_all = np.maximum(np.gcd.reduce(np.abs(hkl_all), axis=1), 1)
            prim_all = sph._canonical_sign(hkl_all // gcd_all[:, None])
            node_w = {
                tuple(int(x) for x in hh): float(ww)
                for hh, ww in zip(hkl_model, w_model)
            }
            w_all = np.array(
                [node_w.get(tuple(int(x) for x in pp), 0.0) for pp in prim_all]
            )
        model_shells = (d_all, q_all, w_all)
        print(
            f"spherical-index: 3D search, {int(radial)} radial channels on "
            f"|q| <= {1.0 / d_min:.3f} 1/A, {int((w_all > 0).sum())} reflections "
            "as shells"
        )
    if bool(band_consistency) and wl is None:
        print(
            "spherical-index: no instrument/wavelength in the peaks file; "
            "band consistency off"
        )
    # Both counts, because they drive different stages: the SEARCH sees
    # unique directions (harmonics point the same way and collapse), the
    # REFINEMENT sees every reflection -- 2.4x more on a 100 A cell, and it
    # is the refinement's model axis that sets its memory.
    print(
        f"spherical-index: {len(pr)} peaks on {len(np.unique(bank))} banks, "
        f"{len(dirs)} model directions ({len(h_)} reflections) "
        f"at d_min={d_min:g}"
    )
    if model == "nodal":
        print(
            f"spherical-index: nodal dictionary, {len(hkl_model)} of {n_nodes} "
            f"zone crossings to |uvw| <= {nodal_max_index} observable at d_min"
            + (
                "; the final stage sees every reflection"
                if full_final
                else " (add --final-full-refine for a last pass on every reflection)"
            )
        )

    if raw_mode:
        if search == "lattice":
            # spots from the excess image: 5 sigma blobs, centroid ->
            # direction in the sample frame, weight sqrt(mass).  The
            # dense-regime excess with this threshold gave 213 blobs on
            # MANDI L1 run 0 of which 207 were real (TOF-indexed).
            from scipy.ndimage import center_of_mass, label

            spot_d, spot_w, spot_s = [], [], []
            for bk, rows in per_bank_rows.items():
                shape_ = (len(np.unique(bin_rows[bk])), len(np.unique(bin_cols[bk])))
                for (r_, e_), (_, s_) in zip(rows, per_bank_sig[bk]):
                    e2, s2 = e_.reshape(shape_), s_.reshape(shape_)
                    lab_, n_ = label(e2 > 5.0 * s2)
                    if n_ == 0:
                        continue
                    cms = center_of_mass(e2, lab_, range(1, n_ + 1))
                    mass = np.array([e2[lab_ == k_].sum() for k_ in range(1, n_ + 1)])
                    pr_ = np.array([cm[0] for cm in cms]) * b2 + (b2 - 1) / 2.0
                    pc_ = np.array([cm[1] for cm in cms]) * b2 + (b2 - 1) / 2.0
                    d_ = sph.panel_directions(dets[bk], rows=pr_, cols=pc_, ki=ki)
                    Rr = R_run.get(r_)
                    spot_d.append(d_ @ Rr if Rr is not None else d_)
                    spot_w.append(np.sqrt(np.clip(mass, 0.0, None)))
                    # sin(theta) is a LAB-frame quantity: the band is set by
                    # the beam, not by where the sample happens to be turned
                    spot_s.append(-(d_ @ np.asarray(ki, float)))
            spot_d, spot_w = np.vstack(spot_d), np.concatenate(spot_w)
            spot_s = np.concatenate(spot_s)
            ladder = sph.lattice_ladder(
                spot_d,
                spot_w,
                B,
                sin_theta=spot_s,
                wavelength=wl,
                d_min=d_min,
            )
            results = [
                {
                    "R": R_,
                    "score": s_,
                    "z": z_,
                    "n_matched": 0,
                    "coherence": 0.0,
                    "c": np.nan,
                }
                for R_, s_, z_ in ladder
            ]
            print(
                f"spherical-index: direct-lattice ladder on {len(spot_d)} spots, "
                f"final-rung z = " + ", ".join(f"{r['z']:.1f}" for r in results)
            )
            ewald_refine = True  # the ladder ends at the exact stage's basin
        else:
            # assemble the search manually around the precomputed raw f
            sigma = np.deg2rad(kernel_deg) / np.sqrt(8.0 * np.log(2.0))
            L = f_raw.shape[0] - 1
            g = sph.project_points(dirs_model, w_model, L, sigma)
            if f_raw_radial is not None:
                g3 = sph.radial_model_stack(
                    model_shells[0],
                    model_shells[1],
                    model_shells[2],
                    int(radial),
                    1.0 / float(d_min),
                    L,
                    sigma,
                )
                C, al_, be_, ga_ = sph.correlogram(f_raw_radial, g3)
            elif f_raw_bands is not None:
                masks_model = sph.band_masks(g0_model, band_edges, wl, d_min)
                print(
                    "spherical-index: band consistency, sin(theta) edges "
                    + " ".join(f"{e:.2f}" for e in band_edges[1:-1])
                    + "; model per band "
                    + " ".join(str(int(m_.sum())) for m_ in masks_model)
                    + f" of {len(g0_model)}"
                )
                C, al_, be_, ga_ = sph.banded_search(
                    f_raw_bands, dirs_model, w_model, masks_model, L, sigma
                )
            else:
                C, al_, be_, ga_ = sph.correlogram(f_raw, g)
            cands = sph.top_orientations(C, al_, be_, ga_, n=n_candidates)
            results = []
            kept_R = []
            kept_vals = []
            gnorm2 = sph.so3_inner(g, g, np.eye(3))
            # refine and rank on the objective that was searched: with the
            # radial channels the 2D map's maximum sits elsewhere (MANDI L1
            # run 0: 2.8 deg off the TOF truth, and a far peak outranked it)
            f_obj, g_obj = (
                (f_raw_radial, g3) if f_raw_radial is not None else (f_raw, g)
            )
            inner_obj = (
                sph.so3_inner_stack if f_raw_radial is not None else sph.so3_inner
            )
            for R_, val_grid in cands:
                R_ = sph.refine_local(f_obj, g_obj, R_)
                val = inner_obj(f_obj, g_obj, R_)
                # duplicates as the peaks path defines them: the same peak
                # within the grid separation, or a lattice-symmetry copy (same
                # rotated model AND the same objective value).  Coherence alone
                # is > 0.99 for any small rotation of a dense dictionary and
                # deleted the true L1 orientation (see find_orientations).
                mu = max(
                    (abs(sph.so3_inner(g, g, R_.T @ R2)) / gnorm2 for R2 in kept_R),
                    default=0.0,
                )
                sep = np.deg2rad(2.0 * 180.0 / max(len(be_), 1))
                if any(
                    sph._quat_angle(R_, R2) < sep
                    or (
                        mu > 0.99
                        and abs(val - v2) <= 1e-3 * max(abs(val), abs(v2), 1e-30)
                    )
                    for R2, v2 in zip(kept_R, kept_vals)
                ):
                    continue
                kept_R.append(R_)
                kept_vals.append(val)
                results.append(
                    {
                        "R": R_,
                        "score": val,
                        # the grid value: C is the band-consistent z-sum when
                        # banding is on, and a pooled inner is not on its scale
                        "z": sph.null_zscore(C, val_grid),
                        "n_matched": 0,
                        "coherence": mu,
                        "c": np.nan,
                    }
                )
            results.sort(key=lambda r: r["score"], reverse=True)
        if ewald_refine:
            # The exact Ewald objective, no band limit: predict every
            # in-band reflection's pixel for U and sum the binned excess
            # there.  Measured on MANDI L1 run 0 against a TOF reference:
            # z 48.7 at the truth over random orientations, half-width
            # 0.2 deg, background by 1 deg -- where the L = 96
            # correlogram is 13.6 and a peak wide.  Sharp enough to rank
            # candidates a few degrees apart and to polish the winner; too
            # narrow to search, so it needs the correlogram's candidates.
            _G_c = hkl_all @ B.T
            _shape = {
                bk: (len(np.unique(bin_rows[bk])), len(np.unique(bin_cols[bk])))
                for bk in per_bank_rows
            }
            _exc2 = {
                bk: [(r_, e_.reshape(_shape[bk])) for r_, e_ in rows]
                for bk, rows in per_bank_rows.items()
            }

            _dev = _ewald_device_factory(
                _G_c,
                sorted(int(x) for x in np.unique(run_of_image)),
                R_run,
                dets,
                {(r_, bk): e_ for bk, rows in _exc2.items() for r_, e_ in rows},
                ki_hat,
                wl,
                b2,
            )

            def ewald_score(Umat):
                total = 0.0
                for r_ in np.unique(run_of_image):
                    Rr = R_run.get(int(r_), np.eye(3))
                    G = (_G_c @ Umat.T) @ Rr.T
                    q = np.linalg.norm(G, axis=1)
                    sth = -(G @ ki_hat) / q
                    lam_ = 2.0 * sth / q
                    m_ = (lam_ >= wl[0]) & (lam_ <= wl[1]) & (sth > 0)
                    kf = ki_hat + lam_[m_, None] * G[m_]
                    kf /= np.linalg.norm(kf, axis=1, keepdims=True)
                    for bk, rows in _exc2.items():
                        e_ = next((e for rr, e in rows if rr == r_), None)
                        if e_ is None:
                            continue
                        mask, pr_, pc_ = dets[bk].reflections_mask(
                            kf[:, 0], kf[:, 1], kf[:, 2]
                        )
                        if not np.any(mask):
                            continue
                        u = pr_[mask] / b2 - 0.5
                        v = pc_[mask] / b2 - 0.5
                        i0 = np.floor(u).astype(int)
                        j0 = np.floor(v).astype(int)
                        fu, fv = u - i0, v - j0

                        def _g(di, dj, i0=i0, j0=j0, e_=e_):
                            return e_[
                                np.clip(i0 + di, 0, e_.shape[0] - 1),
                                np.clip(j0 + dj, 0, e_.shape[1] - 1),
                            ]

                        total += float(
                            (
                                (1 - fu) * (1 - fv) * _g(0, 0)
                                + (1 - fu) * fv * _g(0, 1)
                                + fu * (1 - fv) * _g(1, 0)
                                + fu * fv * _g(1, 1)
                            ).sum()
                        )
                return total

            def ewald_polish(Umat, span_deg=1.5):
                from scipy.optimize import minimize

                res_ = minimize(
                    lambda v: -ewald_score(sph._rodrigues(v) @ Umat),
                    np.zeros(3),
                    method="Nelder-Mead",
                    options={"maxiter": 80, "xatol": np.deg2rad(0.01), "fatol": 1e-9},
                    bounds=[(-np.deg2rad(span_deg), np.deg2rad(span_deg))] * 3,
                )
                return sph._rodrigues(res_.x) @ Umat

            if _dev is not None:
                _score_np, _polish_dev = _dev
                ewald_score = lambda Umat: float(_score_np(Umat)[0])  # noqa: E731
                ewald_polish = _polish_dev
                print("spherical-index: exact Ewald stage on the device (flat panels)")
            rng_ = np.random.default_rng(0)
            null_ = np.array(
                [
                    ewald_score(sph._rodrigues(rng_.normal(size=3) * 3.0))
                    for _ in range(16)
                ]
            )
            for r in results:
                r["ewald_score"] = ewald_score(r["R"])
                r["ewald_z"] = (r["ewald_score"] - null_.mean()) / max(
                    null_.std(), 1e-12
                )
            results.sort(key=lambda r: r["ewald_score"], reverse=True)
            print(
                "spherical-index: exact Ewald ranking of candidates, z = "
                + ", ".join(f"{r['ewald_z']:.1f}" for r in results)
            )
            results[0]["R"] = ewald_polish(results[0]["R"])
            results[0]["ewald_score"] = ewald_score(results[0]["R"])
            results[0]["ewald_z"] = (results[0]["ewald_score"] - null_.mean()) / max(
                null_.std(), 1e-12
            )
            print(
                "spherical-index: exact Ewald polish of the winner, z = "
                f"{results[0]['ewald_z']:.1f}"
            )
    elif search == "lattice":
        # sin(theta) per peak, in the lab frame (see the raw branch)
        s_peak = np.concatenate(
            [
                -(
                    sph.panel_directions(
                        dets[int(bk_)],
                        rows=pr[bank == bk_],
                        cols=pc[bank == bk_],
                        ki=ki,
                    )
                    @ np.asarray(ki, float)
                )
                for bk_ in np.unique(bank)
                if int(bk_) in dets
            ]
        )
        order = np.concatenate(
            [np.where(bank == bk_)[0] for bk_ in np.unique(bank) if int(bk_) in dets]
        )
        sth_peak = np.empty(len(d_sample))
        sth_peak[order] = s_peak
        ladder = sph.lattice_ladder(
            d_sample,
            None,
            B,
            sin_theta=sth_peak,
            wavelength=wl,
            d_min=d_min,
        )
        results = [
            {
                "R": R_,
                "score": s_,
                "z": z_,
                "n_matched": 0,
                "coherence": 0.0,
                "c": np.nan,
            }
            for R_, s_, z_ in ladder
        ]
        print(
            f"spherical-index: direct-lattice ladder on {len(d_sample)} peaks, "
            f"final-rung z = " + ", ".join(f"{r['z']:.1f}" for r in results)
        )
    else:
        results = sph.find_orientations(
            d_sample,
            model_dirs=dirs_model,
            model_weights=w_model,
            kernel_deg=kernel_deg,
            n_candidates=n_candidates,
            lam=lam,
            L=bandwidth,
            refine_method=refine_method,
            data_sin_theta=np.abs(d_lab @ ki_hat) if (use_band or use_radial) else None,
            model_g0=g0_model if use_band else None,
            wavelength_band=wl if (use_band or use_radial) else None,
            d_min=d_min if (use_band or use_radial) else None,
            n_radial=int(radial) if use_radial else 0,
            model_shells=model_shells,
        )
    if not results:
        raise RuntimeError("no orientation candidate found")
    best = results[0]
    U = best["R"]
    B_out = B
    # with the exact Ewald stage on, the band-limited matching-free polish
    # would pull U off the exact maximum again (measured: z 16 -> 9)
    if refine and ewald_polish is None:
        sigma = np.deg2rad(kernel_deg) / np.sqrt(8.0 * np.log(2.0))
        L = (
            int(bandwidth)
            if bandwidth is not None
            else int(min(max(np.ceil(3.0 / sigma), 16), 96))
        )
        f = f_raw if f_raw is not None else sph.project_points(d_sample, None, L, sigma)
        U, B_out, _ = sph.refine_matching_free(
            f,
            U,
            hkl_model,
            B,
            weights=w_model,
            kernel_deg=kernel_deg,
            refine_cell=refine_cell,
        )
        if full_final and not refine_instrument:
            # the optional full-list stage, here the last one: continue
            # from the nodal solution with every reflection in the model
            U, B_out, _ = sph.refine_matching_free(
                f, U, hkl_all, B_out, kernel_deg=kernel_deg, refine_cell=refine_cell
            )
    # per-run goniometer metadata, shared by the two refinement stages
    run_list = sorted(int(x) for x in np.unique(run)) if run is not None else [0]
    angles_by_run = {}
    if g_angles is not None and run is not None:
        # three layouts exist in the wild: one row per peak (a finder
        # file), one per frame (a merged stack, or this stage's own output
        # after per-run corrections), one per run.  Picking the wrong one
        # silently mislabels which angles a run was measured at.
        n_rows = g_angles.shape[0]
        per_peak = n_rows == len(pr)
        per_frame = (
            not per_peak
            and img is not None
            and n_rows > int(np.max(run)) + 1
            and n_rows > int(np.max(img))
        )
        for r_ in run_list:
            i0 = int(np.argmax(run == r_))
            if per_peak:
                row = i0
            elif per_frame:
                row = int(img[i0])
            else:
                row = int(r_)
            angles_by_run[r_] = np.asarray(g_angles[row])

    def _axis_indices(wanted):
        short = [n.split(":")[-1] for n in g_names]
        idx = [
            k
            for k, (n, sn) in enumerate(zip(g_names, short))
            if (n in wanted) or (sn in wanted)
        ]
        if not idx:
            raise ValueError(f"{wanted} matches none of the axes {short}")
        return idx

    gonio_mask = None
    if refine_gonio_axes and g_names is not None:
        short = [n.split(":")[-1] for n in g_names]
        gonio_mask = np.array(
            [
                (n in refine_gonio_axes) or (sn in refine_gonio_axes)
                for n, sn in zip(g_names, short)
            ]
        )
        if not gonio_mask.any():
            raise ValueError(
                f"none of --refine-gonio-axes {refine_gonio_axes} matches "
                f"the file's goniometer axes {short}"
            )

    gonio_offsets_deg = None
    det_report = None
    ref_geom = None
    if refine_instrument:
        # peak-shaped input: found peaks, or the strongest binned pixels of
        # raw mode as weighted peaks
        if raw_mode:
            rb = np.concatenate(raw_bank)
            rr_ = np.concatenate(raw_prow)
            rc_ = np.concatenate(raw_pcol)
            rw = np.concatenate(raw_w)
            rrun = np.concatenate(raw_run)
            keep = np.argsort(-rw)[: int(refine_max_points)]
            keep = keep[rw[keep] > 0]
            det_idx_r, u_off_r, v_off_r, banks_r, nominal_r = _panel_offsets(
                dets, rb[keep], rr_[keep], rc_[keep]
            )
            w_r, run_r = rw[keep], rrun[keep].astype(int)
        else:
            det_idx_r, u_off_r, v_off_r, banks_r, nominal_r = _panel_offsets(
                dets, bank, pr, pc
            )
            w_r, run_r = (
                np.ones(len(pr)),
                (run if run is not None else np.zeros(len(pr), int)),
            )
        peaks_in = {"det_idx": det_idx_r, "u_off": u_off_r, "v_off": v_off_r}
        gonio_in = None
        if gonio_mask is not None and angles_by_run:
            run_pos = {r_: i for i, r_ in enumerate(run_list)}
            peaks_in["run_idx"] = np.array([run_pos[int(r_)] for r_ in run_r])
            gonio_in = {
                "axes": np.asarray(g_axes, dtype=float),
                "angles_deg": np.stack([angles_by_run[r_] for r_ in run_list]),
                "refine_mask": gonio_mask,
                "bound_deg": gonio_bound_deg,
            }
            if refine_gonio_axis_vector:
                tm = np.zeros(len(g_axes), bool)
                tm[_axis_indices(refine_gonio_axis_vector)] = True
                gonio_in["axis_tilt_mask"] = tm
                gonio_in["tilt_bound_deg"] = gonio_axis_vector_bound_deg
            if refine_gonio_per_run:
                gonio_in["per_run_axis"] = _axis_indices([refine_gonio_per_run])[0]
                gonio_in["per_run_bound_deg"] = gonio_per_run_bound_deg
            if refine_gonio_harmonics:
                gonio_in["harmonics_axis"] = _axis_indices([refine_gonio_harmonics])[0]
                gonio_in["harmonics_orders"] = list(gonio_harmonics_orders or [1])
                gonio_in["harmonics_bound_deg"] = gonio_harmonics_bound_deg
        elif run is not None:
            # rotate data into the sample frame beforehand is not possible
            # through refine_instrument without gonio; fold the nominal
            # rotation into the offsets machinery by passing angles with an
            # all-False mask
            if angles_by_run:
                run_pos = {r_: i for i, r_ in enumerate(run_list)}
                peaks_in["run_idx"] = np.array([run_pos[int(r_)] for r_ in run_r])
                gonio_in = {
                    "axes": np.asarray(g_axes, dtype=float),
                    "angles_deg": np.stack([angles_by_run[r_] for r_ in run_list]),
                    "refine_mask": np.zeros(len(g_axes), bool),
                    "bound_deg": gonio_bound_deg,
                }
        modes_r = tuple(detector_modes) if detector_modes else ("independent",)
        # the cylinder modes scale panel centers about an axis; the classic
        # path defaults it to the vertical, so mirror that here (with the
        # value said out loud) rather than dying inside jax as `c * None`
        if cylinder_axis is None and (
            "cylindrical" in modes_r or "axial_stretch" in modes_r
        ):
            cylinder_axis = [0.0, 1.0, 0.0]
            print(
                "  cylinder axis not given; using the classic default "
                "(0, 1, 0) -- override with --cylinder-axis"
            )
        if det_global_rot_axis is None and "global_rot_axis" in modes_r:
            det_global_rot_axis = [0.0, 1.0, 0.0]
            print(
                "  global rotation axis not given; using the classic default "
                "(0, 1, 0) -- override with --det-global-rot-axis"
            )
        out_ref = sph.refine_instrument_matching_free(
            peaks_in,
            nominal_r,
            hkl_inst,
            B,
            U,
            modes=modes_r,
            det_bounds={
                "independent_trans": det_trans_bound,
                "independent_rot": np.deg2rad(det_rot_bound_deg),
                "radial": det_radial_bound_frac,
                "area": det_area_bound_frac,
                "global_rot": np.deg2rad(det_global_rot_bound_deg),
                "global_rot_axis": np.deg2rad(det_global_rot_bound_deg),
                "global_trans": det_global_trans_bound,
            },
            gonio=gonio_in,
            weights=w_r,
            kernel_deg=kernel_deg,
            refine_cell=refine_cell,
            maxiter=refine_maxiter,
            cylinder_axis=cylinder_axis,
            global_rot_axis=det_global_rot_axis,
            refine_banks=refine_detector_banks,
            bank_ids_present=banks_r,
            refine_beam=refine_beam,
            beam_bound_deg=beam_bound_deg,
            sample_offset=in_sample_offset,
            refine_sample=refine_sample,
            sample_bound_m=sample_bound_m,
            per_run_trans=refine_gonio_per_run_trans,
            per_run_trans_bound_m=gonio_per_run_trans_bound_m,
            model_weights=w_inst,
        )
        U, B_out = out_ref["R"], out_ref["B"]
        # The refined instrument, kept as geometry (not as the normalized
        # parameters that encode it): the quality pass scores through it
        # and the writer persists it in the layout the predictor, the
        # integrator and the overlays read.
        ref_geom = {
            "banks": banks_r,
            "panels": out_ref.get("panels"),
            "axes": out_ref.get("axes_refined"),
            "per_run_deg": out_ref.get("per_run_deg"),
            "harmonics_deg": out_ref.get("harmonics_deg"),
            "harmonics_per_run_deg": out_ref.get("harmonics_per_run_deg"),
            "ki": out_ref.get("ki_refined"),
            "sample_offset_m": out_ref.get("sample_offset_m"),
            "per_run_trans_m": out_ref.get("per_run_trans_m"),
            "per_run_axis": (gonio_in or {}).get("per_run_axis"),
            "harmonics_axis": (gonio_in or {}).get("harmonics_axis"),
        }
        from subhkl.instrument.refinables import detector_mode_slices
        from subhkl.instrument.refinables import forward_map_param as _fmp

        nb_ = len(banks_r)
        dp = out_ref["det_params"]
        slices_r, _ = detector_mode_slices(modes_r, nb_)
        det_report = {"banks": banks_r, "det_params": dp, "modes": list(modes_r)}
        msg = [
            f"  instrument refinement ({','.join(modes_r)}): "
            f"loglik {out_ref['loglik']:.3f}"
        ]
        if "independent" in slices_r:
            ip = dp[slices_r["independent"]]
            det_report["trans_m"] = _fmp(ip[: nb_ * 3].reshape(nb_, 3), det_trans_bound)
            det_report["rot_rad"] = _fmp(
                ip[nb_ * 3 :].reshape(nb_, 3), np.deg2rad(det_rot_bound_deg)
            )
            worst = np.abs(det_report["trans_m"]).max(axis=1)
            wb = banks_r[int(np.argmax(worst))]
            msg.append(
                f"largest bank translation {1e3 * worst.max():.2f} mm (bank {wb}), "
                f"largest tilt {np.rad2deg(np.abs(det_report['rot_rad']).max()):.3f} deg"
            )
        mode_bounds = {
            "radial": det_radial_bound_frac,
            "cylindrical": det_radial_bound_frac,
            "axial_stretch": det_radial_bound_frac,
            "area": det_area_bound_frac,
            "global_rot": np.deg2rad(det_global_rot_bound_deg),
            "global_rot_axis": np.deg2rad(det_global_rot_bound_deg),
            "global_trans": det_global_trans_bound,
        }
        for mode_ in modes_r:
            if mode_ == "independent":
                continue
            vals = _fmp(dp[slices_r[mode_]], mode_bounds[mode_])
            det_report[mode_] = vals
            if mode_ in ("global_rot", "global_rot_axis"):
                msg.append(f"{mode_} {np.round(np.rad2deg(vals), 4)} deg")
            elif mode_ == "global_trans":
                msg.append(f"{mode_} {np.round(1e3 * vals, 3)} mm")
            else:
                msg.append(f"{mode_} {np.round(vals, 5)}")
        print("; ".join(msg))
        for key_, fmt_ in (
            (
                "axis_tilts_deg",
                lambda v: {
                    g_names[k2].split(":")[-1]: np.round(t2, 4).tolist()
                    for k2, t2 in v.items()
                },
            ),
            ("per_run_deg", lambda v: np.round(v, 3).tolist()),
            ("harmonics_per_run_deg", lambda v: np.round(v, 3).tolist()),
            ("ki_refined", lambda v: np.round(v, 5).tolist()),
            ("sample_offset_m", lambda v: np.round(1e3 * v, 3).tolist()),
        ):
            if key_ in out_ref:
                print(f"  {key_}: {fmt_(out_ref[key_])}")
                det_report = det_report or {}
                det_report[key_] = out_ref[key_]
        if "per_run_trans_m" in out_ref:
            det_report = det_report or {}
            det_report["per_run_trans_m"] = out_ref["per_run_trans_m"]
            print(
                "  per_run_trans_m: largest "
                f"{1e3 * np.abs(out_ref['per_run_trans_m']).max():.2f} mm"
            )
        if out_ref.get("at_bounds"):
            # A refinable pinned at its bound is a clipped fit, not a
            # converged one -- the objective wanted to go further.  Say so
            # with the flag that lets it: CG4D's +24 mm radial error is
            # 5.4% of the distance and sat silently at the 5% default.
            sat = out_ref["at_bounds"]
            print(
                f"  WARNING: {len(sat)} refined parameter(s) sat AT their "
                "bounds -- the fit was clipped, not converged; raise the "
                "bound and re-run, or the geometry stays wrong:"
            )
            per_flag = {}
            for entry in sat:
                per_flag.setdefault(entry["flag"], []).append(entry)
            for flag_, entries in per_flag.items():
                e0 = max(entries, key=lambda e: abs(e["value"]))
                what = e0["what"]
                unit = e0["unit"]
                val, bnd = e0["value"], e0["bound"]
                if unit == "rad":
                    val, bnd, unit = np.rad2deg(val), np.rad2deg(bnd), "deg"
                elif unit == "m":
                    val, bnd, unit = 1e3 * val, 1e3 * bnd, "mm"
                if what.startswith("goniometer") and g_names is not None:
                    idx = e0["index"]
                    if what.endswith("offset") and idx < len(g_names):
                        what = f"{what} [{g_names[idx].split(':')[-1]}]"
                    elif "per-run" in what:
                        what = f"{what} [run {run_list[idx]}]"
                n_more = len(entries) - 1
                print(
                    f"      {what}: {val:+.4g}{' ' + unit if unit else ''} at the "
                    f"{bnd:.4g}{' ' + unit if unit else ''} bound"
                    + (f" (and {n_more} more)" if n_more else "")
                    + f" -- {flag_}"
                )
        if "gonio_offsets_deg" in out_ref:
            gonio_offsets_deg = out_ref["gonio_offsets_deg"]
            print(
                "  goniometer offsets [deg]: "
                + "  ".join(
                    f"{n.split(':')[-1]} {o:+.3f}"
                    for n, o, mflag in zip(g_names, gonio_offsets_deg, gonio_mask)
                    if mflag
                )
            )

    if fit_gonio_offsets:
        if gonio_mask is None or not angles_by_run or Rg is None:
            raise ValueError(
                "--fit-gonio-offsets needs --refine-gonio-axes and a peaks "
                "file with goniometer axes/angles/R"
            )
        # per-run LAB-frame orientations: each run solved at its own gauge,
        # immune to whatever the offsets are -- the assignment-free
        # analogue of the classic per-run DPHI corrections
        U_runs = {}
        R_nom = {}
        for r_ in run_list:
            m_ = run == r_
            res_r = sph.find_orientations(
                d_lab[m_],
                model_dirs=dirs_model,
                model_weights=w_model,
                kernel_deg=kernel_deg,
                n_candidates=1,
                L=bandwidth,
                refine_method=refine_method,
            )
            U_runs[r_] = res_r[0]["R"]
            R_nom[r_] = (Rg if Rg.shape[0] == len(pr) else Rg[img])[np.argmax(m_)]
        sym_ops = _lattice_proper_rotations(dirs)
        delta, U_fit, rms, ident = _fit_gonio_offsets_from_runs(
            U_runs,
            R_nom,
            np.asarray(g_axes, dtype=float),
            angles_by_run,
            gonio_mask,
            sym_ops,
        )
        gonio_offsets_deg = delta
        U = U_fit
        parts = []
        for k_, (n, mflag) in enumerate(zip(g_names, gonio_mask)):
            if not mflag:
                continue
            if ident.get(k_, True):
                parts.append(f"{n.split(':')[-1]} {delta[k_]:+.3f} deg")
            else:
                parts.append(
                    f"{n.split(':')[-1]} GAUGE (absorbable into U at these "
                    "scan angles; pinned to 0)"
                )
        print(
            "  per-run offsets fit (rms residual "
            f"{rms:.3f} deg over {len(run_list)} runs): " + "  ".join(parts)
        )
        # quality should judge the OFFSET-CORRECTED frame mapping
        from subhkl.instrument.refinables import gonio_rotation_jax as _grj

        d_new = np.array(d_sample)
        for r_ in run_list:
            m_ = run == r_
            Rr = np.asarray(_grj(np.asarray(g_axes, float), angles_by_run[r_], delta))
            d_new[m_] = d_lab[m_] @ Rr
        d_sample = d_new

    for r in results:
        print(
            f"  candidate: z={r['z']:.1f}  matched={r['n_matched']}  "
            f"c={r.get('c', float('nan')):.3f}  coherence={r.get('coherence', 0.0):.3f}"
        )

    # ------------------------------------------------------------------
    # The refined instrument, assembled once: calibrated panels, the beam,
    # the sample origin, and the goniometer rotation per run with the zero
    # offsets, the axis tilts and the per-run (and harmonic) angle
    # corrections all applied.  Everything below -- the quality pass and
    # the file -- speaks through these, so a better fit cannot read worse
    # than the geometry it replaced.
    # ------------------------------------------------------------------
    dets_cal = dets
    ki_out = np.asarray(ki, dtype=float)
    axes_nominal = np.asarray(g_axes, dtype=float) if g_axes is not None else None
    axes_out = axes_nominal
    sample_offset_out = in_sample_offset
    per_run_trans_out = None
    per_run_delta = None
    R_by_run = None
    if ref_geom is not None:
        if ref_geom.get("panels") is not None:
            dets_cal = _calibrated_dets(dets, ref_geom["banks"], ref_geom["panels"])
        if ref_geom.get("ki") is not None:
            ki_out = np.asarray(ref_geom["ki"], dtype=float)
        if ref_geom.get("sample_offset_m") is not None:
            sample_offset_out = np.asarray(ref_geom["sample_offset_m"], dtype=float)
        if ref_geom.get("per_run_trans_m") is not None:
            per_run_trans_out = np.asarray(ref_geom["per_run_trans_m"], dtype=float)
        if ref_geom.get("axes") is not None:
            axes_out = np.asarray(ref_geom["axes"], dtype=float)
        if axes_out is not None:
            # per-run angle corrections ride on their own motor's angle --
            # that is what both the DPHI analogue and our harmonic series
            # are, so both fold exactly into corrected angles (unlike the
            # classic lab-frame rocking, which cannot)
            corr = np.zeros((len(run_list), axes_out.shape[0]))
            for key_, axis_ in (
                ("per_run_deg", ref_geom.get("per_run_axis")),
                ("harmonics_per_run_deg", ref_geom.get("harmonics_axis")),
            ):
                vals = ref_geom.get(key_)
                if vals is not None and axis_ is not None:
                    corr[:, int(axis_)] += np.asarray(vals, dtype=float)
            if np.any(corr):
                per_run_delta = corr
    if (
        axes_out is not None
        and angles_by_run
        and (ref_geom is not None or gonio_offsets_deg is not None)
    ):
        from subhkl.instrument.refinables import gonio_rotation_jax as _grj

        offs_use = (
            np.asarray(gonio_offsets_deg, dtype=float)
            if gonio_offsets_deg is not None
            else np.zeros(axes_out.shape[0])
        )
        R_by_run = {}
        for i_r, r_ in enumerate(run_list):
            ang_r = np.asarray(angles_by_run[r_], dtype=float)
            if per_run_delta is not None:
                ang_r = ang_r + per_run_delta[i_r]
            R_by_run[int(r_)] = np.asarray(_grj(axes_out, ang_r, offs_use))

    if ref_geom is not None or gonio_offsets_deg is not None:
        # re-derive the data directions through the refined instrument
        run_pos = {int(r_): i for i, r_ in enumerate(run_list)}

        def _origin(run_a):
            if sample_offset_out is None and per_run_trans_out is None:
                return None
            o = np.zeros((len(run_a), 3))
            if sample_offset_out is not None:
                o = o + sample_offset_out
            if per_run_trans_out is not None:
                idx = np.array([run_pos.get(int(r_), 0) for r_ in run_a])
                o = o + per_run_trans_out[idx]
            return o

        if raw_mode:
            q_bank = np.concatenate(raw_bank)
            q_row = np.concatenate(raw_prow)
            q_col = np.concatenate(raw_pcol)
            q_runs = np.concatenate(raw_run).astype(int)
            raw_pts = [
                _sample_frame_directions(
                    dets_cal,
                    q_bank,
                    q_row,
                    q_col,
                    ki_out,
                    run_a=q_runs,
                    R_by_run=R_by_run if R_by_run else R_run,
                    sample_offset=_origin(q_runs),
                )
            ]
        else:
            run_q = run if run is not None else np.zeros(len(pr), int)
            d_sample = _sample_frame_directions(
                dets_cal,
                bank,
                pr,
                pc,
                ki_out,
                run_a=run_q,
                R_by_run=R_by_run,
                R_rows=None if R_by_run else (R_peak if Rg is not None else None),
                sample_offset=_origin(run_q),
            )

    # The native quality report, every run -- the solve is seconds, the
    # metrics are one chunked matmul, and a number without its null floor
    # is not a number.  Peaks mode weights each peak once; raw mode weights
    # each binned pixel by its positive excess counts, so the same report
    # exists with no peak finder anywhere.
    if raw_mode:
        q_pts = np.concatenate(raw_pts)
        q_w = np.concatenate(raw_w)
        q_run = np.concatenate(raw_run)
    else:
        q_pts = d_sample
        q_w = np.ones(len(d_sample))
        q_run = run if run is not None else None
    _mark("search+refine")
    quality = _spherical_quality(dirs, U, q_pts, q_w, q_run, kernel_deg)
    _mark("quality")
    phases = {k: v for k, v in _t.items() if k != "start"}
    print("  timings: " + "  ".join(f"{k} {v:.1f}s" for k, v in phases.items()))
    print(
        f"  quality: median nearest-line deviation "
        f"{quality['median_deviation_deg']:.3f} deg "
        f"(null {quality['null_median_deviation_deg']:.2f}); "
        f"matched(<{quality['matched_tol_deg']:.2f} deg) "
        f"{100 * quality['matched_fraction']:.1f}% "
        f"(null {100 * quality['null_matched_fraction']:.1f}%); "
        f"loglik/weight {quality['loglik_per_weight']:.3f} nats; "
        f"null-subtracted: {100 * quality['aligned_fraction']:.1f}% of weight "
        + (
            "aligned, median SATURATED ("
            + quality.get("aligned_median_saturated_why", "")
            + " -- read z and CC(1/2) instead)"
            if quality.get("aligned_median_saturated")
            else f"aligned, at median {quality['aligned_median_deg']:.3f} deg"
        )
    )
    if quality["per_run_median_deg"] and len(quality["per_run_median_deg"]) > 1:
        worst = max(quality["per_run_median_deg"].items(), key=lambda kv: kv[1])
        best = min(quality["per_run_median_deg"].items(), key=lambda kv: kv[1])
        print(
            f"  per-run median deviation: best run {best[0]} at {best[1]:.3f} deg, "
            f"worst run {worst[0]} at {worst[1]:.3f} deg -- a large spread is "
            "the signature of goniometer-offset error (index per run with "
            "--runs to see any run at its own gauge)"
        )

    with h5py.File(output_filename, "w") as out:
        out["sample/U"] = np.asarray(U, dtype=np.float64)
        out["sample/B"] = np.asarray(B_out, dtype=np.float64)
        for kname, val in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            out[f"sample/{kname}"] = val
        out["sample/space_group"] = sg
        out["beam/ki_vec"] = np.asarray(ki_out, dtype=float)
        if not np.allclose(ki_out, ki):
            out["beam/ki_vec_nominal"] = np.asarray(ki, dtype=float)
        if wl is not None:
            out["instrument/wavelength"] = wl
        out["bank"] = bank
        out["peaks/pixel_r"] = pr
        out["peaks/pixel_c"] = pc
        if img is not None:
            out["peaks/image_index"] = img
        if run is not None:
            out["peaks/run_index"] = run
        if Rg is not None:
            R_write = Rg
            if R_by_run is not None and run is not None and Rg.shape[0] == len(pr):
                # the rotation the solution was fitted with, per peak: zero
                # offsets, axis tilts and per-run corrections applied
                R_write = np.stack(
                    [R_by_run.get(int(r_), Rg[i]) for i, r_ in enumerate(run)]
                ).astype(np.float64)
                out["goniometer/R_nominal"] = Rg
            out["goniometer/R"] = R_write
        out["spherical/model"] = model
        out["spherical/model_size"] = len(hkl_model)
        out["spherical/nodal_max_index"] = int(nodal_max_index)
        out["spherical/final_full_refine"] = bool(full_final)
        out["spherical/ewald_refine"] = bool(ewald_refine)
        out["spherical/search"] = str(search)
        out["spherical/U_candidates"] = np.stack([r["R"] for r in results])
        out["spherical/z"] = np.array([r["z"] for r in results])
        out["spherical/n_matched"] = np.array([r["n_matched"] for r in results])
        out["spherical/c"] = np.array([r.get("c", np.nan) for r in results])
        out["spherical/coherence"] = np.array(
            [r.get("coherence", 0.0) for r in results]
        )
        if gonio_offsets_deg is not None and g_names is not None:
            seen = {}
            for n_, o_ in zip(g_names, gonio_offsets_deg):
                seen[n_] = seen.get(n_, 0) + 1
                key = n_ if seen[n_] == 1 else f"{n_}_{seen[n_]}"
                out[f"goniometer/offsets/{key}"] = float(o_)
        # ------------------------------------------------------------------
        # The refined instrument in the layout the CONSUMERS read.  The
        # spherical/* groups below are the report; these are what the
        # predictor, the integrator, the overlays and the MTZ exporter
        # actually apply (apply_detector_calibration, goniometer/angles,
        # goniometer/per_run, goniometer/translations, beam/ki_vec).
        # Without them a refinement improves the fitted U while prediction
        # keeps running on nominal frames -- measured: aligned median
        # 0.69 -> 0.42 deg with per-run corrections, yet CC(1/2)
        # 0.9853 -> 0.8814, because U and the predictor disagreed.
        # ------------------------------------------------------------------
        if ref_geom is not None and ref_geom.get("panels") is not None:
            panels_ = ref_geom["panels"]
            for i_b, b_ in enumerate(ref_geom["banks"]):
                grp = out.create_group(f"detector_calibration/bank_{int(b_)}")
                grp["center"] = np.asarray(panels_["centers"][i_b], dtype=float)
                grp["uhat"] = np.asarray(panels_["uhats"][i_b], dtype=float)
                grp["vhat"] = np.asarray(panels_["vhats"][i_b], dtype=float)
                grp["width"] = float(panels_["widths"][i_b])
                grp["height"] = float(panels_["heights"][i_b])
        if g_names is not None:
            out.create_dataset(
                "goniometer/names",
                data=[str(n) for n in g_names],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        if axes_out is not None:
            out["goniometer/axes"] = np.asarray(axes_out, dtype=float)
            if axes_nominal is not None and not np.allclose(axes_out, axes_nominal):
                out["goniometer/axes_nominal"] = np.asarray(axes_nominal, dtype=float)
        if sample_offset_out is not None:
            out["goniometer/translations"] = np.asarray(sample_offset_out, dtype=float)
        # per-run corrections: fold into the frame-addressed angles the
        # consumers read (with the nominals kept alongside), and record the
        # run map so the integrator's per-run displacements and the MTZ
        # exporter's DPHI/DTX/DTY/DTZ columns find what they need
        frame_tab = _frame_angle_table(
            (frame_table_filename, images_filename, peaks_h5_filename),
            n_peaks=len(pr),
        )
        if per_run_delta is not None or per_run_trans_out is not None:
            # Every per-run array the consumers read is indexed by RUN ID
            # -- the predictor and the MTZ exporter both do
            # per_run[frame_to_run[frame]] -- while the refinement orders
            # its results by position in the refined run list; with --runs
            # those differ, so expand here.  The frame map is written
            # unconditionally: the predictor requires it as soon as
            # trans_m exists.
            n_runs_out = int(max(run_list)) + 1
            frame_to_run = None
            if frame_tab is not None:
                ang_f, frame_to_run, files_, offs_ = frame_tab
                n_runs_out = max(n_runs_out, len(offs_))
            elif img is not None and run is not None:
                # no frame table: the frame -> run map from the peaks
                # themselves -- exact for every frame carrying a peak,
                # with the frames between inheriting the last run seen,
                # which is what a run being a contiguous block of frames
                # means.  Only the frames a peak witnesses are covered, so
                # a trailing peakless tail is absent by construction.
                n_frames_pk = int(np.max(img)) + 1
                f2r = np.full(n_frames_pk, -1, dtype=np.int64)
                f2r[np.asarray(img, dtype=int)] = np.asarray(run, dtype=int)
                valid = np.where(f2r >= 0)[0]
                if len(valid):
                    pos_v = (
                        np.searchsorted(valid, np.arange(n_frames_pk), side="right") - 1
                    )
                    frame_to_run = f2r[valid[np.clip(pos_v, 0, len(valid) - 1)]]
            motor_col = (
                int(ref_geom["per_run_axis"])
                if ref_geom is not None and ref_geom.get("per_run_axis") is not None
                else None
            )

            def _by_run_id(vals):
                out_a = np.zeros((n_runs_out,) + np.asarray(vals).shape[1:])
                for i_r, r_ in enumerate(run_list):
                    if 0 <= int(r_) < n_runs_out:
                        out_a[int(r_)] = np.asarray(vals)[i_r]
                return out_a

            grp = out.create_group("goniometer/per_run")
            grp["runs"] = np.asarray(run_list, dtype=np.int32)
            if motor_col is not None:
                grp["motor"] = str(g_names[motor_col]).split(":")[-1]
            if frame_to_run is not None:
                grp["frame_to_run"] = np.asarray(frame_to_run, dtype=np.int32)
            if per_run_trans_out is not None:
                grp["trans_m"] = _by_run_id(per_run_trans_out)
            delta_full = None
            if per_run_delta is not None:
                delta_full = _by_run_id(per_run_delta)
                # delta_deg is the classic key -- the named motor's per-run
                # correction, what the MTZ exporter writes as DPHI
                col = (
                    motor_col
                    if motor_col is not None
                    else int(np.argmax(np.abs(delta_full).max(axis=0)))
                )
                grp["delta_deg"] = np.asarray(delta_full[:, col], dtype=float)
                grp["delta_deg_all_axes"] = delta_full
            if frame_tab is not None:
                if files_ is not None:
                    grp["run_files"] = files_
                    out["files"] = files_
                grp["run_file_offsets"] = np.asarray(offs_, dtype=np.int64)
                out["file_offsets"] = np.asarray(offs_, dtype=np.int64)
                if delta_full is not None:
                    out["goniometer/angles_nominal"] = ang_f
                    out["goniometer/angles"] = ang_f + delta_full[frame_to_run]
            elif delta_full is not None:
                # no frame table (a peaks file with per-peak angle rows and
                # no run boundaries): correct in the file's own layout and
                # say so, rather than writing a frame array nothing can map
                if g_angles is not None and g_angles.shape[0] == len(pr):
                    out["goniometer/angles_nominal"] = g_angles
                    out["goniometer/angles"] = (
                        g_angles + delta_full[np.asarray(run, dtype=int)]
                    )
                print(
                    "  note: no frame table (goniometer/angles + file_offsets) "
                    "in the inputs -- per-run corrections are recorded and "
                    "applied per peak, but the predictor addresses frames; "
                    "pass --images/--frame-table <merged.h5> to persist "
                    "corrected frames"
                )
        elif frame_tab is not None:
            ang_f, _, files_, offs_ = frame_tab
            out["goniometer/angles"] = ang_f
            out["file_offsets"] = np.asarray(offs_, dtype=np.int64)
            if files_ is not None:
                out["files"] = files_
        if det_report is not None:
            if "banks" in det_report:
                out["spherical/detector/banks"] = np.array(det_report["banks"])
                out["spherical/detector/modes"] = np.array(
                    det_report["modes"], dtype="S"
                )
                out["spherical/detector/det_params"] = det_report["det_params"]
            for kk in (
                "trans_m",
                "rot_rad",
                "radial",
                "cylindrical",
                "axial_stretch",
                "area",
                "global_rot",
                "global_rot_axis",
                "global_trans",
                "per_run_deg",
                "harmonics_per_run_deg",
                "ki_refined",
                "sample_offset_m",
                "per_run_trans_m",
            ):
                if kk in det_report:
                    out[
                        f"spherical/gonio/{kk}"
                        if kk
                        not in (
                            "trans_m",
                            "rot_rad",
                            "radial",
                            "cylindrical",
                            "axial_stretch",
                            "area",
                            "global_rot",
                            "global_rot_axis",
                            "global_trans",
                        )
                        else f"spherical/detector/{kk}"
                    ] = det_report[kk]
            if "axis_tilts_deg" in det_report:
                for k2, t2 in det_report["axis_tilts_deg"].items():
                    out[f"spherical/gonio/axis_tilt_deg/{g_names[k2]}"] = t2
        for kq, vq in quality.items():
            if kq == "per_run_median_deg":
                out["spherical/quality/per_run"] = np.array(sorted(vq))
                out["spherical/quality/per_run_median_deg"] = np.array(
                    [vq[r_] for r_ in sorted(vq)]
                )
            else:
                out[f"spherical/quality/{kq}"] = vq
        out.attrs["instrument"] = str(instrument)
    print(f"wrote {output_filename} (bootstrap-compatible: indexer --bootstrap)")
    return output_filename


def run_indexer_visualize(
    peaks_filename: str,
    instrument: str | None = None,
    output_dir: str | None = None,
    max_index: int = 1,
    dpi: int = 600,
    image_index: int | None = None,
    images_filename: str | None = None,
    zone_alpha: float = 0.5,
):
    """Draw low-index Laue zone conics over the measured peaks, per run.

    Takes an indexer output (or spherical-index output): needs sample/U,
    sample/B, the peaks, and the goniometer rotations.  One figure per run
    (its first frame's goniometer setting; --image-index overrides), named
    '<peaks>-zones-run<r>.png' in --output-dir (default: alongside the
    peaks file).  See subhkl.viz.zones for what is drawn and why the
    wavelength band deliberately is not used to clip the curves.
    """
    import os

    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    from subhkl.viz import zones

    with h5py.File(peaks_filename, "r") as fp:
        bank = fp["bank"][()]
        pr = fp["peaks/pixel_r"][()]
        pc = fp["peaks/pixel_c"][()]
        img = (
            fp["peaks/image_index"][()]
            if "peaks/image_index" in fp
            else np.zeros(len(pr), int)
        )
        run = (
            fp["peaks/run_index"][()]
            if "peaks/run_index" in fp
            else np.zeros(len(pr), int)
        )
        Rg = fp["goniometer/R"][()] if "goniometer/R" in fp else None
        U = fp["sample/U"][()]
        B = fp["sample/B"][()]
        ki = fp["beam/ki_vec"][()] if "beam/ki_vec" in fp else np.array([0.0, 0.0, 1.0])
        inst = instrument or fp.attrs.get("instrument")
    if inst is None:
        raise ValueError("no instrument in the peaks file; pass --instrument")
    # the overlay draws the solution's own geometry: a detector_calibration
    # in the solution file (spherical or classic) beats the nominal panels
    apply_detector_calibration(peaks_filename, str(inst))
    dets = {int(k): Detector(v) for k, v in beamlines[str(inst)].items()}

    base = os.path.splitext(os.path.basename(peaks_filename))[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(peaks_filename))
    os.makedirs(out_dir, exist_ok=True)
    written = []

    if images_filename is not None:
        # With images, adopt the prediction plots' layout and naming
        # exactly: one unrolled-detector figure per run named
        # '<RUNID>-index.png' beside '<RUNID>-pred.png', where RUNID is the
        # source file label the Peaks loader derives -- so the indexing
        # check sits directly next to the prediction check in a directory
        # listing.
        from subhkl.integration.api import Peaks
        from subhkl.viz.detector_assembly import plot_unrolled_detector
        from subhkl.viz.zones import zone_curve_points

        loaded = Peaks(images_filename, str(inst))
        runs_imgs = {}
        for img_key in sorted(loaded.image.ims):
            runs_imgs.setdefault(loaded.get_run_id(img_key), []).append(img_key)

        class _Shim:
            sample_offset = np.zeros(3)
            R = None

        skipped = []
        for r, img_keys in runs_imgs.items():
            m = run == r
            if image_index is not None:
                m = m & (img == int(image_index))
            if not np.any(m):
                # a run the solution file does not cover (e.g. indexed with
                # --runs): there is no goniometer setting to draw it with,
                # and an overlay at the identity is a wrong picture, not a
                # cautious one -- it sits frozen while the spots move
                skipped.append(int(r))
                continue
            if Rg is None:
                R_frame = None
            elif Rg.shape[0] == len(pr):
                R_frame = Rg[np.argmax(m)]
            else:
                R_frame = Rg[img_keys[0]]
            run_dets = {k: loaded.get_detector_by_img(k) for k in img_keys}
            run_ims = {k: loaded.image.ims[k] for k in img_keys}
            curves = zone_curve_points(
                run_dets, U, B, R_gonio=R_frame, ki=ki, max_index=max_index
            )
            # no peak markers here: with the raw counts rendered underneath,
            # the diffraction spots speak for themselves and the s=40
            # circles (an order larger than the spots) only cluttered the
            # check.  The no-images fallback keeps its peak scatter, since
            # there the peaks are the only data on the figure.
            label = loaded.get_image_label(img_keys[0])
            out_name = os.path.join(out_dir, f"{label}-index.png")
            plot_unrolled_detector(
                _Shim(),
                run_ims,
                run_dets,
                out_name=out_name,
                instrument=str(inst),
                dpi=dpi,
                zone_curves=curves,
                zone_alpha=zone_alpha,
            )
            written.append(out_name)
            print(f"wrote {out_name}")
        if skipped:
            print(
                f"indexer-visualize: {len(skipped)} run(s) in the images file have "
                f"no rows in the solution file and were not drawn: {skipped}"
            )
        return written

    for r in np.unique(run):
        # no images: the standalone peaks-plus-conics rendering.  One
        # figure per run with every peak of the run -- the goniometer is
        # constant within a run (verified on CG4D garnet: zero deviation);
        # --image-index restricts to one frame for a scanning-within-run
        # case.
        m = run == r
        if image_index is not None:
            m = m & (img == int(image_index))
            if not np.any(m):
                continue
        if Rg is None:
            R_frame = None
        elif Rg.shape[0] == len(pr):
            R_frame = Rg[np.argmax(m)]
        else:
            R_frame = Rg[int(np.min(img[m]))]
        mm = m
        sel_img = int(np.min(img[m]))
        out_name = os.path.join(out_dir, f"{base}-zones-run{int(r)}.png")
        zones.plot_zone_overlay(
            dets,
            bank[mm],
            pr[mm],
            pc[mm],
            U,
            B,
            R_gonio=R_frame,
            ki=ki,
            max_index=max_index,
            out_name=out_name,
            dpi=dpi,
            zone_alpha=zone_alpha,
            title=f"{base} run {int(r)} frame {sel_img}: zones to index {max_index}",
        )
        written.append(out_name)
        print(f"wrote {out_name}")
    return written
