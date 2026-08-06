import os
from dataclasses import dataclass, astuple
import numpy as np
from typing import List, Any, Optional, Dict, Tuple

from .image_data import ImageData
from subhkl.config import beamlines
from subhkl.instrument.goniometer import Goniometer
from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder
from subhkl.search.sparse_rbf import SparseRBFPeakFinder


@dataclass(frozen=True)
class Wavelength:
    min: float = None
    max: float = None

    def __iter__(self):
        return iter((self.min, self.max))


@dataclass(frozen=True)
class DetectorPeaks:
    R: List[Any]
    two_theta: List[float]
    az_phi: List[float]
    wavelength_mins: List[float]
    wavelength_maxes: List[float]
    intensity: List[float]
    sigma: List[float]
    radii: List[float]
    xyz: List[List[float]]
    bank: List[int]
    image_index: List[int]
    run_id: List[int]
    gonio_axes: Optional[List[List[float]]]
    gonio_angles: List[List[float]]
    gonio_names: Optional[List[str]]
    peak_rows: Optional[List[int]]
    peak_cols: Optional[List[int]]
    # Per-peak quality metrics from the finder, when it reports them.
    # Trailing and defaulted so that positional construction and unpacking of
    # the fields above are unaffected.
    deviance: Optional[List[float]] = None
    residual_deviance: Optional[List[float]] = None

    def __iter__(self):
        """Allows tuple unpacking"""
        return iter(astuple(self))

    def __getitem__(self, index):
        """Allows index access"""
        return astuple(self)[index]


@dataclass(frozen=True)
class IntegrationResult:
    h: List[float]
    k: List[float]
    l: List[float]
    intensity: List[float]
    sigma: List[float]
    tt: List[float]
    az: List[float]
    wavelength: List[float]
    bank: List[int]
    run_id: List[int]
    xyz: List[List[float]]
    R: List[Any]
    angles: List[List[float]]

    def __iter__(self):
        """Allows tuple unpacking"""
        return iter(astuple(self))

    def __getitem__(self, index):
        """Allows index access"""
        return astuple(self)[index]


def prepare_harvest_tasks(
    image_data: ImageData,
    instrument: str,
    goniometer: Goniometer,
    wavelength: Wavelength,
    harvest_peaks_kwargs: Dict[str, Any],
    integration_params: Dict[str, Any],
) -> List[Tuple[Any, ...]]:
    ims = image_data.ims
    bank_mapping = image_data.bank_mapping

    finder_algorithm = harvest_peaks_kwargs.pop("algorithm")

    # --- BATCH PRE-PROCESSING (SparseRBF) ---
    precomputed_peaks = {}
    precomputed_deviance = {}
    precomputed_residual = {}
    if finder_algorithm == "sparse_rbf":
        img_keys = sorted(ims.keys())
        images_list = [ims[k] for k in img_keys]
        img_stack = np.stack(images_list)

        border_width = harvest_peaks_kwargs.get("mask_rel_erosion_radius", 0)
        if border_width is None:
            border_width = 0.0
        border_width *= min(img_stack.shape[1], img_stack.shape[2])

        # The global basis-pursuit finder is the default; the greedy
        # matching-pursuit one is kept behind `legacy` while it is retired.
        # Both return [amplitude, row, column, sigma] per peak.
        if harvest_peaks_kwargs.get("legacy", False):
            legacy_alpha = harvest_peaks_kwargs.get("alpha")
            alg = SparseRBFPeakFinder(
                # The greedy finder has no notion of the false-alarm floor, so
                # it keeps its historical constant when none is given.
                alpha=0.1 if legacy_alpha is None else legacy_alpha,
                gamma=harvest_peaks_kwargs.get("gamma", 2.0),
                loss=harvest_peaks_kwargs.get("loss", "gaussian"),
                min_sigma=harvest_peaks_kwargs.get("min_sigma", 1.0),
                max_sigma=harvest_peaks_kwargs.get("max_sigma", 10.0),
                border_width=int(border_width),
                chunk_size=harvest_peaks_kwargs.get("chunk_size", 128),
                show_steps=harvest_peaks_kwargs.get("show_steps", False),
                auto_tune_alpha=harvest_peaks_kwargs.get("auto_tune_alpha", False),
                candidate_alphas=harvest_peaks_kwargs.get("candidate_alphas", None),
            )
        else:
            alg = MatrixFreeSparseRBFPeakFinder(
                # None means "derive it from the image size"; see
                # MatrixFreeSparseRBFPeakFinder.effective_alpha.
                alpha=harvest_peaks_kwargs.get("alpha"),
                # 0, not the historical 2.0: the flux-matched default; see the
                # class docstring.  The legacy branch above keeps 2.0 so that
                # it still reproduces what it always did.
                gamma=harvest_peaks_kwargs.get("gamma", 0.0),
                loss=harvest_peaks_kwargs.get("loss", "poisson"),
                min_sigma=harvest_peaks_kwargs.get("min_sigma", 1.0),
                max_sigma=harvest_peaks_kwargs.get("max_sigma", 10.0),
                # Bank resolution, independent of the ceiling: without it
                # max_sigma sets both, so a wider range can only be bought by
                # coarsening the spacing.
                num_sigmas=harvest_peaks_kwargs.get("num_sigmas", 5),
                # The m0 of the false-alarm calibration: expected false peaks
                # per image.  The one knob that sets the detection budget.
                false_alarms_per_image=harvest_peaks_kwargs.get(
                    "false_alarms_per_image", 1.0
                ),
                show_steps=harvest_peaks_kwargs.get("show_steps", False),
            )
        batch_coords = alg.find_peaks_batch(img_stack)
        precomputed_peaks = {k: c for k, c in zip(img_keys, batch_coords, strict=False)}
        # Per-peak quality metrics, when the finder reports them (the
        # matrix-free finder does; the legacy greedy one does not).
        batch_deviance = getattr(alg, "peak_deviance", None)
        if batch_deviance is not None:
            precomputed_deviance = {
                k: d for k, d in zip(img_keys, batch_deviance, strict=False)
            }
        batch_residual = getattr(alg, "peak_residual_deviance", None)
        if batch_residual is not None:
            precomputed_residual = {
                k: d for k, d in zip(img_keys, batch_residual, strict=False)
            }

    tasks = []
    for img_key in sorted(ims.keys()):
        physical_bank = bank_mapping.get(img_key, img_key)

        # FIX: Skip banks that are not in beamlines config
        if str(physical_bank) not in beamlines[instrument]:
            print(
                f"WARNING: Bank {physical_bank} not found in beamlines config "
                f"for {instrument}. Skipping..."
            )
            continue

        det_config = beamlines[instrument][str(physical_bank)]

        if goniometer.rotation.ndim == 3:
            current_R = (
                goniometer.rotation[img_key]
                if img_key < len(goniometer.rotation)
                else goniometer.rotation[-1]
            )
        else:
            current_R = goniometer.rotation

        current_angles = None
        if goniometer.angles_raw is not None:
            if goniometer.angles_raw.ndim == 2:
                current_angles = (
                    goniometer.angles_raw[img_key]
                    if img_key < len(goniometer.angles_raw)
                    else goniometer.angles_raw[-1]
                )
            else:
                current_angles = goniometer.angles_raw

        pre_coords = None
        if finder_algorithm == "sparse_rbf":
            coords = precomputed_peaks[img_key]
            # coords shape is [intensity, r, c, sigma].  The third slot of
            # pre_coords carries the finder's per-peak Gaussian width so the
            # harvest output can record it (peaks/sigma) for --max-sigma
            # tuning diagnostics; the fourth carries the per-peak leave-one-out
            # deviance (peaks/deviance), the significance of that atom against
            # chi^2 on its four parameters; the fifth the local residual
            # deviance per degree of freedom (peaks/residual_deviance), which
            # says whether the neighbourhood is actually explained -- a
            # mis-sized sigma scores high on the fourth and badly on the fifth.
            dev = precomputed_deviance.get(img_key)
            res = precomputed_residual.get(img_key)
            if len(coords) > 0:
                if dev is None or len(dev) != len(coords):
                    dev = np.zeros(len(coords))
                if res is None or len(res) != len(coords):
                    res = np.zeros(len(coords))
                pre_coords = (coords[:, 1], coords[:, 2], coords[:, 3], dev, res)
            else:
                empty = np.array([])
                pre_coords = (empty, empty, empty, empty, empty)

        finder_info = (finder_algorithm, harvest_peaks_kwargs, pre_coords)
        mask_info = (
            harvest_peaks_kwargs.get("mask_file"),
            harvest_peaks_kwargs.get("mask_rel_erosion_radius"),
        )
        geo_info = (
            current_R,
            current_angles,
            wavelength.min,
            wavelength.max,
        )

        img_label = image_data.get_label(img_key)

        tasks.append(
            (
                img_key,
                img_label,
                physical_bank,
                ims[img_key],
                det_config,
                finder_info,
                integration_params,
                mask_info,
                geo_info,
            )
        )
    return tasks, precomputed_peaks


def prepare_predict_tasks(
    image_data: ImageData,
    instrument: str,
    wavelength_min: float,
    wavelength_max: float,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    d_min: float,
    UB: np.ndarray,
    space_group: str = "P 1",
    sample_offset: Optional[np.ndarray] = None,
    ki_vec: Optional[np.ndarray] = None,
    R_all: Optional[np.ndarray] = None,
    gonio_axes: Optional[Any] = None,
    gonio_angles: Optional[np.ndarray] = None,
    gonio_offsets: Optional[np.ndarray] = None,
) -> List[Tuple[Any, ...]]:
    bank_mapping = image_data.bank_mapping
    tasks = []

    unit_cell_params = (a, b, c, alpha, beta, gamma, space_group, d_min)

    sorted_keys = sorted(image_data.ims.keys())
    if not sorted_keys:
        return []

    total_images = len(sorted_keys)
    print(f"Predicting peaks for {total_images} banks...")

    for img_index, img_key in enumerate(sorted_keys):
        bank_id = bank_mapping.get(img_key, img_key)
        det_config = beamlines[instrument][str(int(bank_id))]

        run_id = image_data.get_run_id(img_key)

        # 1. Extract single R matrix (Legacy Fallback)
        R_bank = None
        if R_all is not None:
            if R_all.ndim == 3:
                if len(R_all) == total_images:
                    R_bank = R_all[img_index]
                else:
                    R_bank = R_all[run_id] if run_id < len(R_all) else R_all[0]
            else:
                R_bank = R_all

        # 2. Extract single goniometer angle state
        angles_bank = None
        if gonio_angles is not None:
            if gonio_angles.ndim == 2:
                num_axes = len(gonio_axes) if gonio_axes is not None else 1
                if gonio_angles.shape[1] == num_axes:
                    if gonio_angles.shape[0] == total_images:
                        angles_bank = gonio_angles[img_index, :]
                    else:
                        angles_bank = (
                            gonio_angles[run_id, :]
                            if run_id < gonio_angles.shape[0]
                            else gonio_angles[0, :]
                        )
                else:
                    if gonio_angles.shape[1] == total_images:
                        angles_bank = gonio_angles[:, img_index]
                    else:
                        angles_bank = (
                            gonio_angles[:, run_id]
                            if run_id < gonio_angles.shape[1]
                            else gonio_angles[:, 0]
                        )
            else:
                angles_bank = gonio_angles

        tasks.append(
            (
                img_key,
                bank_id,
                det_config,
                unit_cell_params,
                UB,  # <-- Pass constant UB!
                wavelength_min,
                wavelength_max,
                sample_offset,
                ki_vec,
                R_bank,
                gonio_axes,
                angles_bank,
                gonio_offsets,  # <-- NEW
            )
        )
    return tasks


def prepare_integrate_tasks(
    image: ImageData,
    filename: str,
    instrument: str,
    peak_dict: Dict[str, List[Any]],
    integration_params: Dict[str, Any],
    RUB: np.ndarray,
    R_stack: Optional[np.ndarray] = None,
    angles_stack: Optional[np.ndarray] = None,
    sample_offset: Optional[np.ndarray] = None,
    ki_vec: Optional[np.ndarray] = None,
    integration_method: str = "free_fit",
    create_visualizations: bool = False,
    show_progress: bool = False,
    found_peaks_file: Optional[str] = None,
) -> List[Tuple[Any, ...]]:
    found_peaks_xyz = None
    found_peaks_bank = None
    found_peaks_run = None
    if found_peaks_file is not None:
        try:
            import h5py

            print(f"Loading found peaks from: {found_peaks_file}")
            with h5py.File(found_peaks_file, "r") as f:
                if "files" in f and "file_offsets" in f and "peaks/xyz" in f:
                    files_db = f["files"][()]
                    offsets = f["file_offsets"][()]
                    target_name = os.path.basename(filename)
                    match_idxs = []
                    # 1. Direct match
                    for i, fname_bytes in enumerate(files_db):
                        fname_str = (
                            fname_bytes.decode("utf-8")
                            if isinstance(fname_bytes, bytes)
                            else str(fname_bytes)
                        )
                        if target_name in fname_str:
                            match_idxs.append(i)

                    # 2. Match via source files (if is a merged master)
                    if not match_idxs and image.raw_files:
                        for src_file in image.raw_files:
                            src_name = os.path.basename(src_file)
                            for i, fname_bytes in enumerate(files_db):
                                fname_str = (
                                    fname_bytes.decode("utf-8")
                                    if isinstance(fname_bytes, bytes)
                                    else str(fname_bytes)
                                )
                                if src_name == os.path.basename(fname_str):
                                    if i not in match_idxs:
                                        match_idxs.append(i)

                    if match_idxs:
                        # Load and concatenate from all matched indices
                        xyz_list = []
                        bank_list = []
                        run_list = []
                        for idx in match_idxs:
                            start = int(offsets[idx])
                            end = (
                                int(offsets[idx + 1])
                                if idx < len(files_db) - 1
                                else f["peaks/xyz"].shape[0]
                            )
                            xyz_list.append(f["peaks/xyz"][start:end])
                            if "bank" in f:
                                bank_list.append(f["bank"][start:end])
                            elif "peaks/bank" in f:
                                bank_list.append(f["peaks/bank"][start:end])

                            if "peaks/run_index" in f:
                                run_list.append(f["peaks/run_index"][start:end])

                        found_peaks_xyz = (
                            np.concatenate(xyz_list, axis=0) if xyz_list else None
                        )
                        found_peaks_bank = (
                            np.concatenate(bank_list, axis=0) if bank_list else None
                        )
                        found_peaks_run = (
                            np.concatenate(run_list, axis=0) if run_list else None
                        )
                elif "peaks/xyz" in f:
                    found_peaks_xyz = f["peaks/xyz"][()]
                    if "bank" in f:
                        found_peaks_bank = f["bank"][()]
                    elif "peaks/bank" in f:
                        found_peaks_bank = f["peaks/bank"][()]
                    if "peaks/run_index" in f:
                        found_peaks_run = f["peaks/run_index"][()]
        except Exception as e:
            print(f"Failed to load found peaks: {e}")

    tasks = []
    os.path.basename(filename)

    sorted_keys = sorted(peak_dict.keys())
    if not sorted_keys:
        return []

    total_images = len(sorted_keys)

    def _resolve(stack, seq_idx, name):
        if stack is None:
            return None

        is_batch = (stack.ndim == 3) or (stack.ndim == 2 and name == "angles_stack")
        if not is_batch:
            return stack

        n_items = stack.shape[0]
        if n_items == 1:
            return stack[0]

        if n_items == total_images:
            return stack[seq_idx]

        raise ValueError(
            f"CRITICAL: Array dimension mismatch for '{name}'. "
            f"The stack contains {n_items} matrices, but there are {total_images} images scheduled. "
            f"Run index fallback is strictly disabled."
        )

    for _i, bank in enumerate(sorted_keys):
        peaks = peak_dict[bank]
        physical_bank = image.bank_mapping.get(bank, bank)
        det_config = beamlines[instrument][str(physical_bank)]

        current_rub = _resolve(RUB, _i, "RUB")
        current_R_val = _resolve(R_stack, _i, "R_stack")
        current_angles_val = _resolve(angles_stack, _i, "angles_stack")

        # The physical run_id can still be safely fetched for metadata logging
        run_id = image.get_run_id(bank)

        metrics_info = (
            found_peaks_xyz,
            found_peaks_bank,
            found_peaks_run,
            run_id,
            current_rub,
            current_angles_val,
            current_R_val,
            sample_offset,
            ki_vec,
        )

        tasks.append(
            (
                bank,
                physical_bank,
                image.ims[bank],
                peaks,
                det_config,
                integration_params,
                integration_method,
                metrics_info,
            )
        )
    return tasks
