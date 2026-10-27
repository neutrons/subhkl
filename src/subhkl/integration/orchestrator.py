from dataclasses import dataclass, astuple
import numpy as np
from typing import List, Any, Optional, Dict, Tuple

from .image_data import ImageData
from subhkl.config import beamlines
from subhkl.instrument.goniometer import Goniometer
from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder


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
    # No intensity: the finder reports positions, shape and validation
    # metrics; amplitude measurement belongs to the integrator.
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
    # The per-peak Gaussian width, for the finders that fit one.  None when no
    # width was measured, which is the difference from `sigma` above: that one
    # always holds a number, but only sometimes a width.
    width: Optional[List[float]] = None

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

        # The global basis-pursuit finder is the only finder: the greedy
        # matching-pursuit one it superseded is retired.  Returns
        # [amplitude, row, column, sigma] per peak.
        alg = MatrixFreeSparseRBFPeakFinder(
            # None means "derive it from the image size"; see
            # MatrixFreeSparseRBFPeakFinder.effective_alpha.
            alpha=harvest_peaks_kwargs.get("alpha"),
            # 0, not the greedy finder's historical 2.0: the flux-matched
            # default; see the class docstring.
            gamma=harvest_peaks_kwargs.get("gamma", 0.0),
            loss=harvest_peaks_kwargs.get("loss", "poisson"),
            min_sigma=harvest_peaks_kwargs.get("min_sigma", 1.0),
            # None measures the ceiling from the first batch's own width
            # census.
            max_sigma=harvest_peaks_kwargs.get("max_sigma"),
            # Bank resolution, independent of the ceiling.  None lets the
            # finder auto-size the bank against carpet fragmentation; an
            # explicit count keeps the historical uniform grid.
            num_sigmas=harvest_peaks_kwargs.get("num_sigmas"),
            # Tolerable unsupported atoms per image.  Mapped onto the
            # brightness quantile the auto bank protects, via the moment
            # census of each batch -- arithmetic, no extra solves; see
            # _frag_protected_quantile.  Non-positive keeps the fixed
            # p90 census quantile.
            max_fragmentation_rate=harvest_peaks_kwargs.get(
                "max_fragmentation_rate", 1.0
            ),
            # The m0 of the false-alarm calibration: expected false peaks
            # per image.  The one knob that sets the detection budget.
            false_alarms_per_image=harvest_peaks_kwargs.get(
                "false_alarms_per_image", 1.0
            ),
            show_steps=harvest_peaks_kwargs.get("show_steps", False),
            # These four were not forwarded at first, and the omission was
            # invisible from the CLI: the constructor's **kwargs swallows
            # nothing, the class defaults are sensible, and every unit test
            # builds the class directly.  The visible symptoms were that
            # --sparse-rbf-profile-file gaussian (the documented opt-out
            # of the learned family) did nothing, and that a suite tuned
            # to --sparse-rbf-chunk-size 64 was actually running at
            # whatever the class default happened to be.
            profile_file=harvest_peaks_kwargs.get("profile_file", "auto"),
            shape_ratio=harvest_peaks_kwargs.get("shape_ratio", 1.2),
            shape_orientations=harvest_peaks_kwargs.get("shape_orientations", 4),
            chunk_size=harvest_peaks_kwargs.get("chunk_size", 64),
            multi_gpu=harvest_peaks_kwargs.get("multi_gpu", False),
        )
        # A static-structure mask (see subhkl.search.static_mask) is mapped
        # onto the input by *physical* bank, so a mask built from any scans of
        # this instrument -- different sample included -- applies here, and a
        # bank the mask file does not carry runs unmasked.
        valid_stack = None
        static_mask_file = harvest_peaks_kwargs.get("static_mask_file")
        if static_mask_file and isinstance(alg, MatrixFreeSparseRBFPeakFinder):
            from subhkl.search.static_mask import load_mask_for_banks

            valid_stack = load_mask_for_banks(
                static_mask_file,
                [bank_mapping.get(k, k) for k in img_keys],
                img_stack.shape[1:],
            )
        elif static_mask_file:
            print(
                "WARNING: --static-mask-file is only honored by the matrix-free finder."
            )

        if valid_stack is not None:
            batch_coords = alg.find_peaks_batch(img_stack, valid=valid_stack)
        else:
            batch_coords = alg.find_peaks_batch(img_stack)
        precomputed_peaks = {k: c for k, c in zip(img_keys, batch_coords, strict=False)}
        # Per-peak quality metrics, when the finder reports them (the
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
    per_run_trans: Optional[np.ndarray] = None,
    frame_to_run: Optional[np.ndarray] = None,
    harmonic_rot: Optional[np.ndarray] = None,
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

        # Per-run sample displacement rides on the innermost axis; the
        # per-image task granularity makes it a per-task effective offset.
        so_eff = sample_offset
        if (
            per_run_trans is not None
            and frame_to_run is not None
            and sample_offset is not None
            and np.ndim(sample_offset) == 2
            and img_index < len(frame_to_run)
        ):
            so_eff = np.array(sample_offset, dtype=float, copy=True)
            so_eff[-1] = so_eff[-1] + per_run_trans[int(frame_to_run[img_index])]

        # Fourier rocking: a lab-frame q-steering rotation per image.
        extra_rot = None
        if harmonic_rot is not None and img_index < len(harmonic_rot):
            extra_rot = harmonic_rot[img_index]

        tasks.append(
            (
                img_key,
                bank_id,
                det_config,
                unit_cell_params,
                UB,  # <-- Pass constant UB!
                wavelength_min,
                wavelength_max,
                so_eff,
                ki_vec,
                R_bank,
                gonio_axes,
                angles_bank,
                gonio_offsets,  # <-- NEW
                extra_rot,
            )
        )
    return tasks
