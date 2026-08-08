import cv2
import numpy as np
import PIL.Image
import scipy
import skimage.feature

from dataclasses import dataclass

from subhkl.convex_hull.peak_integrator import PeakIntegrator
from subhkl.instrument.detector import Detector
from subhkl.search.threshold import ThresholdingPeakFinder
from subhkl.instrument.physics import (
    calculate_angular_error,
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


def _run_harvest_local_max(
    im,
    max_peaks=200,
    min_pix=50,
    min_rel_intensity=0.5,
    normalize=False,
    **kwargs,
):
    """
    Worker for finding peak candidates using local maxima search.
    """
    if normalize:
        blur = scipy.ndimage.gaussian_filter(im, 4)
        div = scipy.ndimage.gaussian_filter(im, 60)
        processed = blur / div
    else:
        processed = im

    coords = skimage.feature.peak_local_max(
        processed,
        num_peaks=max_peaks,
        min_distance=min_pix,
        threshold_rel=min_rel_intensity,
        exclude_border=min_pix * 3,
    )
    return coords[:, 0], coords[:, 1]


def _run_harvest_thresholding(im, **kwargs):
    """
    Worker for finding peak candidates using adaptive thresholding.
    """
    valid_keys = [
        "noise_cutoff_quantile",
        "min_peak_dist_pixels",
        "blur_kernel_sigma",
        "open_kernel_size_pixels",
        "mask_file",
        "mask_rel_erosion_radius",
        "show_steps",
        "show_scale",
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    alg = ThresholdingPeakFinder(**filtered_kwargs)
    coords = alg.find_peaks(im)
    return coords[:, 0], coords[:, 1]


def _aperture_intensity(image, mask, row, col, radius_px):
    """Aperture photometry for a peak the convex-hull stage did not fit.

    Used only when the hull filter is switched off, to give a peak the finder
    reported but the hull stage rejected an intensity that is at least defined.
    Counts inside ``radius_px`` of the centre, minus a local background taken
    as the median of the annulus from ``radius_px`` to ``1.5 * radius_px``:

        I = sum_ap (y - b),   sigma_I = sqrt(sum_ap y + n_ap * b * (1 + n_ap/n_bg))

    This is a *different estimator* from the hull integration -- no region
    growing, no shape fit, a fixed circular aperture -- so intensities from the
    two paths are not interchangeable at the precision the hull path offers.
    That is the price of switching the filter off, and the reason the filter is
    on by default.
    """
    h, w = image.shape
    r_out = max(1.0, 1.5 * radius_px)
    r0 = max(0, int(np.floor(row - r_out)))
    r1 = min(h, int(np.ceil(row + r_out)) + 1)
    c0 = max(0, int(np.floor(col - r_out)))
    c1 = min(w, int(np.ceil(col + r_out)) + 1)
    if r1 <= r0 or c1 <= c0:
        return 0.0, 0.0

    yy, xx = np.mgrid[r0:r1, c0:c1]
    dist = np.sqrt((yy - row) ** 2 + (xx - col) ** 2)
    patch = np.asarray(image[r0:r1, c0:c1], dtype=float)
    valid = np.asarray(mask[r0:r1, c0:c1], dtype=bool)

    in_ap = (dist <= max(1.0, radius_px)) & valid
    in_bg = (dist > max(1.0, radius_px)) & (dist <= r_out) & valid
    n_ap = int(in_ap.sum())
    if n_ap == 0:
        return 0.0, 0.0

    n_bg = int(in_bg.sum())
    bg = float(np.median(patch[in_bg])) if n_bg > 0 else 0.0
    total = float(patch[in_ap].sum())
    intensity = total - n_ap * bg
    var = total + (n_ap * bg * (1.0 + n_ap / n_bg) if n_bg > 0 else 0.0)
    return intensity, float(np.sqrt(max(var, 0.0)))


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
    """
    Worker function to process a single image in a separate process.
    Performs peak finding, integration, and optional visualization.
    """
    # Unpack tuple arguments
    algo, harvest_kwargs, pre_coords = finder_info
    mask_file, erosion = mask_info
    gonio_R, gonio_angles, wl_min, wl_max = geometry_info

    det = Detector(det_config)

    # 1. Find Peaks
    finder_widths = None
    finder_deviance = None
    finder_residual = None
    if algo == "sparse_rbf":
        # (rows, cols[, widths[, deviance[, residual]]]): the optional third
        # slot is the finder's per-peak Gaussian sigma, carried through to the
        # harvest output for --max-sigma tuning diagnostics, the fourth is the
        # per-peak leave-one-out deviance and the fifth its local residual
        # deviance per degree of freedom.
        i, j = pre_coords[0], pre_coords[1]
        if len(pre_coords) > 2:
            finder_widths = np.asarray(pre_coords[2])
        if len(pre_coords) > 3:
            finder_deviance = np.asarray(pre_coords[3])
        if len(pre_coords) > 4:
            finder_residual = np.asarray(pre_coords[4])
    elif algo == "peak_local_max":
        i, j = _run_harvest_local_max(image, **harvest_kwargs)
    elif algo == "thresholding":
        i, j = _run_harvest_thresholding(image, **harvest_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

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

    # ==========================================
    # 2.5 STRICTLY ENFORCE MASK ON CANDIDATES
    # ==========================================
    valid_indices = []
    for idx, (r, c) in enumerate(centers):
        r_int, c_int = int(r), int(c)
        # Only keep centers that fall inside the image bounds AND the true mask
        if (
            0 <= r_int < mask.shape[0]
            and 0 <= c_int < mask.shape[1]
            and mask[r_int, c_int]
        ):
            valid_indices.append(idx)

    # Slice the arrays to drop forbidden peaks
    centers = centers[valid_indices]
    i = i[valid_indices]
    j = j[valid_indices]
    if finder_widths is not None:
        finder_widths = finder_widths[valid_indices]
    if finder_deviance is not None:
        finder_deviance = finder_deviance[valid_indices]
    if finder_residual is not None:
        finder_residual = finder_residual[valid_indices]

    # 3. Integration Setup (Sigma Override)
    # Rebuild integrator from params to avoid sharing state
    integrator = PeakIntegrator.build_from_dictionary(integration_params.copy())

    if integration_params.get("region_growth_minimum_sigma") is not None:
        mean = np.mean(image[mask])
        std = np.std(image[mask])
        n_sigma = integration_params["region_growth_minimum_sigma"]
        integrator.region_grower.min_intensity = mean + n_sigma * std

    # 4. Integrate
    int_result, hulls, refined_centers = integrator.integrate_peaks(
        physical_bank, image, centers, return_hulls=True
    )

    bank_intensity = np.array(
        [np.nan if res[3] is None else float(res[3]) for res in int_result], dtype=float
    )
    bank_sigma = np.array(
        [np.nan if res[5] is None else float(res[5]) for res in int_result], dtype=float
    )

    # The convex-hull stage is two things at once: it measures the intensity,
    # and -- because a candidate it cannot fit a region to is dropped here --
    # it acts as a true-positive filter on whatever the finder proposed.  The
    # two roles are separable, and `hull_filter=False` keeps only the first:
    # every candidate that survived the mask is reported, and the ones the hull
    # stage could not fit get an aperture intensity instead of being discarded.
    # It is on by default; switching it off is a statement that the finder's
    # own output is trusted, which is only true once the finder's per-peak
    # metrics (peaks/deviance, peaks/residual_deviance) are doing that job.
    hull_filter = bool(integration_params.get("hull_filter", True))

    keep = []
    aperture_idx = []
    for idx, res in enumerate(int_result):
        has_hull = hulls[idx][1] is not None
        is_valid = res[3] is not None
        hull_ok = bool(is_valid and has_hull)
        keep.append(hull_ok or not hull_filter)
        if not hull_ok and not hull_filter:
            aperture_idx.append(idx)

    # Fill in the peaks the hull stage could not measure, so that switching the
    # filter off never emits an undefined intensity.
    fallback_radius = float(
        integration_params.get("region_growth_maximum_pixel_radius", 10.0) or 10.0
    )
    aperture_px = np.zeros(len(int_result), dtype=float)
    for idx in aperture_idx:
        radius_px = (
            3.0 * float(finder_widths[idx])
            if finder_widths is not None and np.isfinite(finder_widths[idx])
            else fallback_radius
        )
        aperture_px[idx] = radius_px
        bank_intensity[idx], bank_sigma[idx] = _aperture_intensity(
            image, mask, float(i[idx]), float(j[idx]), radius_px
        )

    # Keep integrated centers centers for finder
    i, j = i[keep], j[keep]
    if finder_widths is not None:
        finder_widths = finder_widths[keep]
    if finder_deviance is not None:
        finder_deviance = finder_deviance[keep]
    if finder_residual is not None:
        finder_residual = finder_residual[keep]

    # 6. Gather Results
    if sum(keep) > 0:
        intensities = bank_intensity[keep]
        sigmas = bank_sigma[keep]
        tt, az = det.pixel_to_angles(i, j)
        lab_coords = det.pixel_to_lab(i, j)
        if lab_coords.ndim == 1:
            lab_coords = lab_coords[np.newaxis, :]
        num = len(tt)

        # Radii Calculation
        radii = []
        kept_indices = np.where(keep)[0]
        tt_rad, az_rad = np.deg2rad(tt), np.deg2rad(az)
        v_centers = np.stack(
            [
                np.sin(tt_rad) * np.cos(az_rad),
                np.sin(tt_rad) * np.sin(az_rad),
                np.cos(tt_rad),
            ],
            axis=1,
        )
        for k_idx, orig_idx in enumerate(kept_indices):
            _, hull, _, _ = hulls[orig_idx]
            if hull is None:
                # No hull to take vertices from.  With the filter on this case
                # never reaches here; with it off, quote the angular size of
                # the aperture that was actually integrated rather than 0,
                # which would claim a point-like peak.
                if aperture_px[orig_idx] > 0.0:
                    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
                    verts = np.stack(
                        [
                            i[k_idx] + aperture_px[orig_idx] * np.cos(theta),
                            j[k_idx] + aperture_px[orig_idx] * np.sin(theta),
                        ],
                        axis=1,
                    )
                else:
                    radii.append(0.0)
                    continue
            else:
                verts = hull.points[hull.vertices]
            v_i, v_j = verts[:, 0], verts[:, 1]
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
            "intensity": intensities.tolist(),
            # Redefined at the finding stage: no intensity is measured here,
            # so peaks/sigma carries the finder's per-peak Gaussian width
            # (for --max-sigma tuning diagnostics) instead of the
            # integrator's intensity sigma.  The integration-stage output
            # (the lattice path) keeps intensity sigma unchanged.
            "sigma": (
                finder_widths.tolist() if finder_widths is not None else sigmas.tolist()
            ),
            # Per-peak leave-one-out deviance: the likelihood-ratio statistic
            # for this atom's presence, on the same scale as the global
            # Deviance/DoF report and calibrated against chi^2 with four
            # degrees of freedom (95% point 9.49).
            "deviance": (
                finder_deviance.tolist() if finder_deviance is not None else [0.0] * num
            ),
            # Local residual deviance per degree of freedom over the atom's own
            # 3-sigma footprint: near 1 where the model explains the
            # neighbourhood, well above 1 where it does not -- a mis-sized
            # sigma, a swallowed neighbour, unmodelled background.  The
            # deviance above cannot see any of that: it stays large for a
            # badly shaped atom that still carries real signal.
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
            f"Integrated {len(i)}/{len(centers)} peaks for {img_label} "
            f"(Bank {physical_bank})"
        )
        if not hull_filter:
            # Say how many of those the hull filter would have removed: that
            # count is the whole effect of the switch, and it is what tells you
            # whether the finder is ready to be trusted on its own.
            log_msg += (
                f"; hull filter off, {len(aperture_idx)} kept by aperture "
                f"that the filter would have dropped"
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
    )
    if len(row) > 0:
        return bank_id, [row, col, h_f, k_f, l_f, wl_f]
    return img_key, None


def integrate_single_bank(
    bank_id,
    physical_bank,
    image,
    peaks,
    det_config,
    integration_params,
    integration_method,
    viz_info,
    metrics_info,
):
    # Unpack predicted peaks (i, j, h, k, l, wl)
    bank_i, bank_j, bank_h, bank_k, bank_l, bank_wl = peaks
    centers = np.stack([bank_i, bank_j], axis=-1)

    det = Detector(det_config)

    # --- METRICS: Comparison with found peaks ---
    metrics_str = ""
    (
        found_peaks_xyz,
        found_peaks_bank,
        found_peaks_run,
        run_id,
        RUB,
        current_angles_val,
        current_R_val,
        sample_offset,
        ki_vec,
    ) = metrics_info

    # CORRECTED: Account for Sample-frame offset rotation
    s_lab = (
        current_R_val @ sample_offset if current_R_val is not None else sample_offset
    )
    bank_tt, bank_az = det.pixel_to_angles(
        bank_i, bank_j, sample_offset=s_lab, ki_vec=ki_vec
    )

    # Correctly handle lab coordinate shape (N, 3)
    lab_coords = det.pixel_to_lab(bank_i, bank_j)
    if lab_coords.ndim == 1:
        lab_coords = lab_coords[np.newaxis, :]  # (1, 3) -> (N=1, 3)

    if found_peaks_xyz is not None and len(centers) > 0:
        f_xyz_valid = np.array([])

        # Priority 1: Filter by run index (for merged multi-run files)
        if found_peaks_run is not None:
            mask_run = found_peaks_run == run_id
            f_xyz_valid = found_peaks_xyz[mask_run]
        # Priority 2: Filter by physical bank ID
        elif found_peaks_bank is not None:
            mask_bank = found_peaks_bank == physical_bank
            f_xyz_valid = found_peaks_xyz[mask_bank]
        # Priority 3: Spatial proximity (for single files)
        else:
            s_off = sample_offset if sample_offset is not None else np.zeros(3)
            det_vec = det.center - s_off
            f_vecs = found_peaks_xyz - s_off
            dots = np.dot(f_vecs, det_vec)
            f_xyz_front = found_peaks_xyz[dots > 0]

            if len(f_xyz_front) > 0:
                f_row, f_col = det.lab_to_pixel(
                    f_xyz_front[:, 0],
                    f_xyz_front[:, 1],
                    f_xyz_front[:, 2],
                    clip=False,
                )
                on_sensor = (
                    (f_row >= 0) & (f_row < det.n) & (f_col >= 0) & (f_col < det.m)
                )
                f_xyz_valid = f_xyz_front[on_sensor]

        if len(f_xyz_valid) > 0:
            f_row_valid, f_col_valid = det.lab_to_pixel(
                f_xyz_valid[:, 0],
                f_xyz_valid[:, 1],
                f_xyz_valid[:, 2],
                clip=False,
            )
            on_panel_found = (
                (f_row_valid >= 0)
                & (f_row_valid < det.n)
                & (f_col_valid >= 0)
                & (f_col_valid < det.m)
            )

            if np.sum(on_panel_found) > 0:
                f_row_valid = f_row_valid[on_panel_found]
                f_col_valid = f_col_valid[on_panel_found]
                f_xyz_valid = f_xyz_valid[on_panel_found]

                f_pixels = np.stack([f_row_valid, f_col_valid], axis=1)
                p_pixels = np.stack([bank_i, bank_j], axis=1)

                tree = scipy.spatial.KDTree(p_pixels)
                dists_pix, idxs = tree.query(f_pixels)
                valid_matches = dists_pix < 20.0

                if np.sum(valid_matches) > 0:
                    matched_idxs = idxs[valid_matches]
                    f_xyz_matched = f_xyz_valid[valid_matches]
                    d_err, ang_err = calculate_angular_error(
                        f_xyz_matched,
                        bank_h[matched_idxs],
                        bank_k[matched_idxs],
                        bank_l[matched_idxs],
                        bank_wl[matched_idxs],
                        RUB,
                        sample_offset,
                        ki_vec,
                        current_R_val,
                    )
                    metrics_str = (
                        f" | Med Error: $\\Delta\\theta$={np.median(ang_err):.2f}$^\\circ$, "
                        f"$\\Delta d$={np.median(d_err):.3f}$\\AA$"
                    )

    # --- INTEGRATION ---
    mask_file, mask_erosion = (
        integration_params.get("integration_mask_file"),
        integration_params.get("integration_mask_rel_erosion_radius", None),
    )
    if mask_file is not None:
        mask = np.array(PIL.Image.open(mask_file))
    else:
        mask = np.full(image.shape, 1, dtype=np.uint8)

    if mask_erosion:
        radius = max(1, int(min(mask.shape) * mask_erosion))
        kernel = np.ones((2 * radius, 2 * radius), dtype=np.uint8)
        mask = cv2.erode(
            mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)

    integrator = PeakIntegrator.build_from_dictionary(integration_params.copy())
    if integration_params.get("region_growth_minimum_sigma") is not None:
        mean = np.mean(image[mask])
        std = np.std(image[mask])
        n_sigma = integration_params["region_growth_minimum_sigma"]
        integrator.region_grower.min_intensity = mean + n_sigma * std

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
    bank_tt = bank_tt[valid_indices]
    bank_az = bank_az[valid_indices]
    bank_h = bank_h[valid_indices]
    bank_k = bank_k[valid_indices]
    bank_l = bank_l[valid_indices]
    bank_wl = bank_wl[valid_indices]
    lab_coords = lab_coords[valid_indices]

    int_result, hulls, refined_centers = integrator.integrate_peaks(
        bank_id,
        image,
        centers,
        integration_method=integration_method,
        return_hulls=True,
    )

    bank_intensity = np.array([res[3] for res in int_result])
    bank_sigma = np.array([res[5] for res in int_result])

    keep = []
    for idx, res in enumerate(int_result):
        has_hull = hulls[idx][1] is not None
        is_valid = res[3] is not None
        keep.append(is_valid and has_hull)

    # Re-calculate angles and lab coordinates using refined centers
    # Only for kept peaks
    if not np.any(keep):
        return None

    kept_centers = refined_centers[keep]
    # Re-calculate angles and lab coordinates
    bank_tt, bank_az = det.pixel_to_angles(
        kept_centers[:, 0],
        kept_centers[:, 1],
        sample_offset=s_lab,
        ki_vec=ki_vec,
    )
    lab_coords = det.pixel_to_lab(kept_centers[:, 0], kept_centers[:, 1])
    if lab_coords.ndim == 1:
        lab_coords = lab_coords[np.newaxis, :]

    # --- VISUALIZATION ---
    do_viz, viz_prefix, viz_label = viz_info
    if do_viz:
        import matplotlib.pyplot as plt

        if plt.get_backend().lower() != "agg":
            plt.switch_backend("Agg")
        plt.rc("font", size=8)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- Dynamic Colorscale Compression ---
        # Extract valid pixels using the mask to calculate robust statistics
        valid_pixels = image[mask.astype(bool)] if mask is not None else image

        median_bg = np.median(valid_pixels)
        std_bg = np.std(valid_pixels)

        # Compress lower edge: pin vmin just above the noise floor (e.g., +2 sigma)
        vmin = max(1.0, 1.0 + median_bg + 2.0 * std_bg)

        # Compress upper edge: pin vmax to the brightest signal
        vmax = np.percentile(1.0 + valid_pixels, 99.8)

        # Safety fallback for flat/empty images
        if vmax <= vmin:
            vmax = vmin + 10.0

        im0 = axes[0].imshow(image, cmap="viridis", origin="lower")
        axes[0].set_title(f"{viz_label}")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        label_pred = f"Predicted{metrics_str}"
        if len(centers) > 0:
            axes[0].scatter(
                centers[:, 1],
                centers[:, 0],
                marker="1",
                c="blue",
                label=label_pred,
                s=40,
            )

        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
            shadow=False,
            ncol=1,
        )

        for p_i, p_j, p_h, p_k, p_l in zip(
            centers[:, 0], centers[:, 1], bank_h, bank_k, bank_l, strict=False
        ):
            is_zone = (p_h == 0) or (p_k == 0) or (p_l == 0)
            is_nodal = (abs(p_h) + abs(p_k) + abs(p_l)) < 8
            if is_zone or is_nodal:
                c = "red" if is_nodal else "black"
                w = "bold" if is_nodal else "normal"
                axes[0].text(
                    p_j,
                    p_i,
                    f"({p_h},{p_k},{p_l})",
                    color=c,
                    fontsize=6,
                    fontweight=w,
                    clip_on=True,
                )
        axes[1].imshow(image, cmap="viridis", origin="lower")

        forbidden = ~mask
        overlay = np.zeros((*forbidden.shape, 4))
        overlay[forbidden] = [0, 1, 1, 0.3]
        axes[1].imshow(overlay, origin="lower")
        axes[1].set_title("Integrated peaks")

        for _, hull, _, _ in hulls:
            if hull is not None:
                for simplex in hull.simplices:
                    axes[1].plot(
                        hull.points[simplex, 1],
                        hull.points[simplex, 0],
                        c="red",
                    )

        # New Naming: {prefix}_{label}_int.png
        # viz_label already includes _bank{physical_bank} from Peaks.integrate
        out_name = f"{viz_label}_int.png"
        if viz_prefix:
            out_name = f"{viz_prefix}_{out_name}"
        fig.tight_layout()
        fig.savefig(out_name, bbox_inches="tight")
        plt.close(fig)

    return {
        "h": bank_h[keep],
        "k": bank_k[keep],
        "l": bank_l[keep],
        "intensity": bank_intensity[keep],
        "sigma": bank_sigma[keep],
        "tt": bank_tt,
        "az": bank_az,
        "wavelength": bank_wl[keep],
        "xyz": lab_coords.tolist(),
        "bank": [bank_id] * sum(keep),
        "run_id": [run_id] * sum(keep),
        "R": [current_R_val] * sum(keep) if current_R_val is not None else [],
        "angles": [current_angles_val] * sum(keep)
        if current_angles_val is not None
        else [],
    }
