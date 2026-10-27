import cv2
import numpy as np
import PIL.Image

from dataclasses import dataclass

from subhkl.instrument.detector import Detector
from subhkl.instrument.physics import (
    predict_reflections_on_panel,
)
from subhkl.core.crystallography import generate_reflections


@dataclass
class _RunPeaksFinder:
    """Lightweight mock of DetectorPeaks for the unrolled plotter."""

    xyz: list
    image_index: list
    peak_rows: list
    peak_cols: list
    intensity: list = None
    var_u: list = None
    var_v: list = None
    cov_uv: list = None


# --- Helper to bypass any hidden worker.py logic and guarantee execution ---
def _safe_plot_wrapper(args):
    run_peaks, images, detectors, finder_peaks, out_name, instrument_label = args
    from subhkl.viz.detector_assembly import plot_unrolled_detector

    plot_unrolled_detector(
        run_peaks,
        images,
        detectors,
        finder_peaks,
        out_name=out_name,
        instrument=instrument_label,
    )
    return out_name


def _render_finder_unrolled_plot(args):
    """Standalone plotting function for generating unrolled plots per run."""
    run_id, peaks, images, detectors, finder_peaks, instrument = args

    import matplotlib.pyplot as plt
    from subhkl.viz.detector_assembly import plot_unrolled_detector

    # Force non-interactive backend for thread safety
    if plt.get_backend().lower() != "agg":
        plt.switch_backend("Agg")

    out_name = f"{run_id}_finder.png"
    plot_unrolled_detector(
        peaks,
        images,
        detectors,
        finder_peaks=finder_peaks,
        out_name=out_name,
        instrument=instrument,
    )


def process_single_image(
    img_key,
    img_label,
    physical_bank,
    image,
    det_config,
    finder_info,
    integration_params,
    mask_info,
    geometry_info,
):
    """Report one image's found peaks: positions, shape, validation metrics.

    The finder measures no amplitude -- intensity belongs to the
    integrator, which solves it jointly per image against the rate-map
    background.  The convex-hull stage that used to sit here did two
    jobs at once (measure an intensity, filter candidates it could not
    fit); both retired: the finder's own per-peak metrics
    (peaks/deviance, peaks/residual_deviance) carry the true-positive
    evidence, and nothing downstream consumes a finder amplitude.
    """
    # Unpack tuple arguments
    algo, harvest_kwargs, pre_coords = finder_info
    mask_file, erosion = mask_info
    gonio_R, gonio_angles, wl_min, wl_max = geometry_info

    det = Detector(det_config)

    # 1. Unpack the batch finder's peaks
    if algo != "sparse_rbf":
        raise ValueError(
            f"Unknown finder algorithm: {algo!r} (the peak_local_max and "
            "thresholding harvesters retired with the convex-hull stage)"
        )
    # (rows, cols[, widths[, deviance[, residual]]]): the optional third
    # slot is the finder's per-peak Gaussian sigma, the fourth the
    # per-peak leave-one-out deviance and the fifth its local residual
    # deviance per degree of freedom.
    i, j = pre_coords[0], pre_coords[1]
    finder_widths = np.asarray(pre_coords[2]) if len(pre_coords) > 2 else None
    finder_deviance = np.asarray(pre_coords[3]) if len(pre_coords) > 3 else None
    finder_residual = np.asarray(pre_coords[4]) if len(pre_coords) > 4 else None

    centers = np.stack([i, j], axis=-1)

    # 2. Setup Mask
    if mask_file is not None:
        mask = np.array(PIL.Image.open(mask_file))
    else:
        mask = np.full(image.shape, 1, dtype=np.uint8)

    if erosion:
        radius = max(1, int(min(mask.shape) * erosion))
        kernel = np.ones((2 * radius, 2 * radius), dtype=np.uint8)
        mask = cv2.erode(
            mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)

    # 2.5 Strictly enforce the mask on candidates
    valid_indices = []
    for idx, (r, c) in enumerate(centers):
        r_int, c_int = int(r), int(c)
        if (
            0 <= r_int < mask.shape[0]
            and 0 <= c_int < mask.shape[1]
            and mask[r_int, c_int]
        ):
            valid_indices.append(idx)

    centers = centers[valid_indices]
    i = i[valid_indices]
    j = j[valid_indices]
    if finder_widths is not None:
        finder_widths = finder_widths[valid_indices]
    if finder_deviance is not None:
        finder_deviance = finder_deviance[valid_indices]
    if finder_residual is not None:
        finder_residual = finder_residual[valid_indices]

    # 3. Gather Results
    if len(i) > 0:
        tt, az = det.pixel_to_angles(i, j)
        lab_coords = det.pixel_to_lab(i, j)
        if lab_coords.ndim == 1:
            lab_coords = lab_coords[np.newaxis, :]
        num = len(tt)

        # Angular radius of each peak's 3-sigma footprint, from an octagon
        # of pixel offsets pushed through the detector geometry (so panel
        # orientation and central-projection distortion are respected the
        # same way the retired hull vertices were).
        FALLBACK_RADIUS_PX = 10.0
        tt_rad, az_rad = np.deg2rad(tt), np.deg2rad(az)
        v_centers = np.stack(
            [
                np.sin(tt_rad) * np.cos(az_rad),
                np.sin(tt_rad) * np.sin(az_rad),
                np.cos(tt_rad),
            ],
            axis=1,
        )
        radii = []
        theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        for k_idx in range(num):
            radius_px = FALLBACK_RADIUS_PX
            if finder_widths is not None and np.isfinite(finder_widths[k_idx]):
                radius_px = 3.0 * float(finder_widths[k_idx])
            v_i = i[k_idx] + radius_px * np.cos(theta)
            v_j = j[k_idx] + radius_px * np.sin(theta)
            v_tt, v_az = det.pixel_to_angles(v_i, v_j, sample_offset=None)
            v_tt_r, v_az_r = np.deg2rad(v_tt), np.deg2rad(v_az)
            v_vecs = np.stack(
                [
                    np.sin(v_tt_r) * np.cos(v_az_r),
                    np.sin(v_tt_r) * np.sin(v_az_r),
                    np.cos(v_tt_r),
                ],
                axis=1,
            )
            dots = np.clip(v_vecs @ v_centers[k_idx], -1.0, 1.0)
            radii.append(np.max(np.arccos(dots)))

        res = {
            "two_theta": tt.tolist(),
            "az_phi": az.tolist(),
            "R": [gonio_R] * num,
            "lamda_min": [wl_min] * num,
            "lamda_max": [wl_max] * num,
            # The finder's per-peak Gaussian width.  No intensity: the
            # finder reports shape and validation metrics only.
            "sigma": (
                finder_widths.tolist() if finder_widths is not None else [0.0] * num
            ),
            # The same widths again, under a name that means only one thing;
            # the plots need a key that is absent rather than ambiguous.
            "width": finder_widths.tolist() if finder_widths is not None else None,
            # Per-peak leave-one-out deviance: the likelihood-ratio statistic
            # for this atom's presence, calibrated against chi^2 with four
            # degrees of freedom (95% point 9.49).
            "deviance": (
                finder_deviance.tolist() if finder_deviance is not None else [0.0] * num
            ),
            # Local residual deviance per degree of freedom over the atom's
            # own 3-sigma footprint: near 1 where the model explains the
            # neighbourhood.
            "residual_deviance": (
                finder_residual.tolist() if finder_residual is not None else [0.0] * num
            ),
            "radii": radii,
            "xyz": lab_coords.tolist(),
            "banks": [physical_bank] * num,
            "image_indices": [img_key] * num,
            "gonio_angles": [gonio_angles] * num if gonio_angles is not None else [],
            "count": num,
            "i": i,
            "j": j,
        }
        log_msg = (
            f"Reported {len(i)}/{len(centers) if len(centers) else len(i)} peaks "
            f"for {img_label} (Bank {physical_bank})"
        )
    else:
        res = None
        log_msg = f"{img_label} (Bank {physical_bank}) had 0 valid peaks"
    return res, log_msg


def predict_single_bank(
    img_key,
    bank_id,
    det_config,
    unit_cell_params,
    UB,
    wavelength_min,
    wavelength_max,
    sample_offset,
    ki_vec,
    R_bank=None,
    gonio_axes=None,
    gonio_angles=None,
    gonio_offsets=None,  # <-- NEW
    extra_rot=None,
):
    """
    Worker function for predicting peaks on a single detector bank.
    Generates HKLs locally (lazy generation) to reduce IPC overhead.
    """
    a, b, c, alpha, beta, gamma, space_group, d_min = unit_cell_params
    h, k, l = generate_reflections(a, b, c, alpha, beta, gamma, space_group, d_min)

    from subhkl.instrument.detector import Detector

    det = Detector(det_config)

    row, col, h_f, k_f, l_f, wl_f = predict_reflections_on_panel(
        detector=det,
        h=h,
        k=k,
        l=l,
        UB=UB,  # <-- Pass UB
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        R_all=R_bank,
        gonio_axes=gonio_axes,
        gonio_angles=gonio_angles,
        gonio_offsets=gonio_offsets,  # <-- Pass Down
        extra_rot=extra_rot,
    )
    if len(row) > 0:
        return bank_id, [row, col, h_f, k_f, l_f, wl_f]
    return img_key, None
