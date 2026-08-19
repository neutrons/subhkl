import numpy as np
from tqdm import tqdm

from jax import jit
from jax import lax
from jax import vmap
import jax
import jax.numpy as jnp

import jax.scipy.optimize
import jax.scipy.signal
import scipy


from dataclasses import dataclass

from functools import partial
from typing import Dict, List
import multiprocessing
import concurrent.futures
from collections import defaultdict
import os


# Cap on how many samples per window the median is taken over.  Passing this as
# ``max_samples`` makes the filter subsample a window larger than the cap on a
# regular lattice instead of reading every pixel in it; the default of ``None``
# is the exact filter, so nothing changes for a caller that does not ask.
#
# The exact form costs O(window^2) *materialised* values per pixel plus a sort of
# that length.  At ``max_sigma = 25`` the window is 125x125 -- 15625 values per
# pixel, 16 GiB of patches for a single 512x512 frame, and 86 s on an H100
# against 40 ms for the 25x25 window of ``max_sigma = 5``.  That one filter is
# 94% of the finder's per-frame cost at that setting, and it grows as
# ``max_sigma^2 log max_sigma`` while the solver grows linearly.
#
# The subsampled window is the same estimator on a regular subset of the same
# population: the standard error of a median over ``n`` samples goes as
# ``1/sqrt(n)``, so the ~625 samples a 125x125 window reduces to hold it to a few
# percent of the local spread, and the result is fed straight into a sigma=3
# Gaussian blur that averages ~113 further independent pixels on top of that.
# Measured against the exact filter on a 125x125 window, the post-blur difference
# is 0.13 counts rms on a background of 4.4 -- a factor of 16 below the Poisson
# noise on a single pixel of that background (2.1).
#
# That difference is still enough to move the answer, because at large max_sigma
# the solve does not reach its KKT tolerance and the low-amplitude tail of the
# peak list is chaotic in the background estimate.  Measured on a five-peak
# synthetic under deterministic XLA flags, all five peaks keep their positions
# (<= 0.1 px) and amplitudes (<= 2%) while the reported count moves from 39 to
# 14; for scale, the same binary run twice *without* those flags returns 39
# peaks and then 26.  The perturbation is real but smaller than the
# nondeterminism already present, and across three seeds the recovered
# amplitudes came out no worse, and on one seed substantially better.
MEDIAN_MAX_SAMPLES = 31 * 31


@partial(jit, static_argnames=["window_size", "max_samples"])
def jax_median_2d(img, window_size, max_samples=None):
    """
    Args:
        img: [photons/Pixel]
        window_size: [Pixel^0.5]
        max_samples: [-] cap on samples per window; None reads every pixel
    Returns:
        [photons/Pixel]
    """
    if max_samples is None:
        dilation = 1
        n_taps = window_size
    else:
        n_side = int(max_samples**0.5)
        dilation = max(1, -(-window_size // n_side))  # ceil, so n_taps <= n_side
        n_taps = max(1, window_size // dilation)

    span = (n_taps - 1) * dilation + 1  # [Pixel^0.5]
    pad_w = span // 2  # [Pixel^0.5]
    padded = jnp.pad(img, pad_w, mode="reflect")  # [photons/Pixel]
    im_4d = padded[None, None, :, :]  # [photons/Pixel]

    patches = lax.conv_general_dilated_patches(
        im_4d,
        filter_shape=(n_taps, n_taps),
        window_strides=(1, 1),
        padding="VALID",
        rhs_dilation=(dilation, dilation),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    med = jnp.median(patches[0], axis=0)  # [photons/Pixel]
    # An even ``span`` has no centre pixel and leaves one row and column more
    # than the input; trim rather than let a half-pixel shift through.
    return med[: img.shape[0], : img.shape[1]]


@partial(jit, static_argnames=["sigma"])
def jax_gaussian_blur_2d(img, sigma=3.0):
    """
    Args:
        img: [photons/Pixel]
        sigma: [Pixel^0.5]
    Returns:
        [photons/Pixel]
    """
    radius = int(4.0 * sigma + 0.5)  # [Pixel^0.5]
    x = jnp.arange(-radius, radius + 1)  # [Pixel^0.5]
    k_1d = jnp.exp(-0.5 * (x / sigma) ** 2)  # [-]
    k_1d = k_1d / jnp.sum(k_1d)  # [-]

    k_col = k_1d[:, None]  # [-]
    k_row = k_1d[None, :]  # [-]

    padded = jnp.pad(img, radius, mode="reflect")  # [photons/Pixel]
    temp = jax.scipy.signal.correlate2d(padded, k_col, mode="valid")  # [photons/Pixel]
    blurred = jax.scipy.signal.correlate2d(temp, k_row, mode="valid")  # [photons/Pixel]
    return blurred  # [photons/Pixel]


@partial(jit, static_argnames=["filter_size", "max_samples"])
def compute_bg_batch(imgs, filter_size, max_samples=None):
    """
    Args:
        imgs: [photons/Pixel]
        filter_size: [Pixel^0.5]
        max_samples: [-] cap on samples per window; None reads every pixel
    Returns:
        [photons/Pixel]
    """

    def process_one(img):
        med = jax_median_2d(img, filter_size, max_samples)  # [photons/Pixel]
        blur = jax_gaussian_blur_2d(med)  # [photons/Pixel]
        return jnp.maximum(blur, 1e-3)  # [photons/Pixel]

    return lax.map(process_one, imgs)  # [photons/Pixel]


@partial(jit, static_argnames=["window_size"])
def _box_mean_2d(img, window_size):
    """Uniform box average, two separable 1D passes.  [same units as img]"""
    w = window_size | 1  # odd, so the window is centred
    k_1d = jnp.full((w,), 1.0 / w, dtype=img.dtype)
    pad = w // 2
    padded = jnp.pad(img, pad, mode="reflect")
    temp = jax.scipy.signal.correlate2d(padded, k_1d[:, None], mode="valid")
    return jax.scipy.signal.correlate2d(temp, k_1d[None, :], mode="valid")


# Highest count level the quantile inversion resolves directly; brighter
# background falls back to the windowed mean (see compute_rate_batch).  Not a
# tuning constant: it only decides which of two estimators answers, and the
# handover happens where both are accurate.
RATE_K_MAX = 16


@partial(jit, static_argnames=["filter_size"])
def compute_rate_batch(imgs, filter_size, valid=None):
    """Local Poisson rate by exact quantile inversion -- the sparse-regime
    replacement for the median background.

    The median background is identically zero wherever the rate is below
    log 2 ~ 0.69 counts/pixel -- the median of Poisson(mu < log 2) is 0 --
    so on short-exposure frames the whole map collapses to its numerical
    clamp (measured: 100% of pixels at 1e-3 on real MANDI garnet banks whose
    true rate is 0.44).  Every z-score downstream is then measured against a
    background hundreds of times too small, a stray single count clears the
    chi^2 significance level on its own, and no reachable alpha controls the
    peak count.

    This estimator inverts the exact Poisson CDF at an empirical quantile
    instead of reading the median off as if it were the rate:

        F_k = box_mean(y <= k)            (window-fraction at or below k)
        k*  = smallest k with F_k >= 1/2  (per pixel)
        mu  solves  gammaincc(k* + 1, mu) = F_{k*}

    using P(Y <= k) = Q(k+1, mu), the regularized upper incomplete gamma.
    At k* = 0 this is the zero-fraction estimator mu = -log F_0 (measured on
    peak-free MANDI tiles: 0.421 +- 0.056 against a masked-mean truth of
    0.438 +- 0.087 -- tighter than the mean itself); at higher k* it reduces
    to the Poisson-correct reading of the median.  Robustness to peaks is
    the same bulk-quantile argument as the median's: peaks only add counts,
    so they can only lower F_k, by at most the peak-area fraction of the
    window, and the induced rate bias is bounded and positive.  Where even
    F_{RATE_K_MAX} < 1/2 (background brighter than ~16 counts/pixel) the
    windowed mean takes over -- there the Gaussian regime holds and peak
    contamination is a small relative perturbation.

    Cost replaces the median's O(window^2) materialised sort -- the 125x125
    window at max_sigma = 25 sorts 15,625 values per pixel and took ~109 s
    of XLA compilation alone -- with (RATE_K_MAX + 2) separable box filters
    and a fixed 30-step bisection of a scalar special function: no sort, no
    window-sized tensor, compile time in seconds.

    When ``valid`` is given (1 = usable pixel, 0 = excluded), every
    window statistic becomes its conditional version on the usable
    pixels only: F_k = box_mean(valid * (y <= k)) / box_mean(valid),
    and the bright fallback is the masked windowed mean.  A caller that
    knows where the peaks are (the integrator, whose dictionary is the
    predicted reflection list) can mask their footprints and estimate
    the rate from genuine background pixels alone -- the quantile
    argument's "bounded positive bias" from peak contamination is only
    small while peaks fill a small fraction of the window, which a
    focusing instrument's spot density does not guarantee (measured:
    +3.5 photons/pixel on a rate of 3 under a peak covering 22% of the
    window, eating 9% of that peak's flux from a fixed-background
    amplitude solve).

    Args:
        imgs: [photons/Pixel]
        filter_size: [Pixel^0.5]
        valid: [-] optional (B, H, W) mask, 1 where the rate may look
    Returns:
        [photons/Pixel]
    """

    def process_one(inputs):
        img, v = inputs
        v_frac = jnp.maximum(_box_mean_2d(v, filter_size), 1e-3)

        F = jnp.stack(
            [
                _box_mean_2d(v * (img <= k).astype(img.dtype), filter_size) / v_frac
                for k in range(RATE_K_MAX + 1)
            ]
        )  # [K+1, H, W]

        hit = F >= 0.5
        any_hit = jnp.any(hit, axis=0)
        k_star = jnp.argmax(hit, axis=0)  # first k with F_k >= 1/2
        F_sel = jnp.take_along_axis(F, k_star[None], axis=0)[0]
        # Clamp into the open unit interval: F = 1 exactly (a fully empty
        # window) must invert to mu = 0, not to a log of zero.
        F_sel = jnp.clip(F_sel, 1e-7, 1.0 - 1e-7)
        a = (k_star + 1).astype(img.dtype)

        # P(Y <= k; mu) = gammaincc(k+1, mu) is strictly decreasing in mu, so
        # bisection on mu in [0, 4 * RATE_K_MAX] converges unconditionally;
        # 30 halvings resolve the rate to 6e-8 of the bracket.
        lo = jnp.zeros_like(F_sel)
        hi = jnp.full_like(F_sel, 4.0 * RATE_K_MAX)

        def bisect(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            too_low = jax.scipy.special.gammaincc(a, mid) > F_sel
            return jnp.where(too_low, mid, lo), jnp.where(too_low, hi, mid)

        lo, hi = lax.fori_loop(0, 30, bisect, (lo, hi))
        mu = 0.5 * (lo + hi)

        bright = _box_mean_2d(v * img, filter_size) / v_frac
        rate = jnp.where(any_hit, mu, bright)

        blur = jax_gaussian_blur_2d(rate)  # [photons/Pixel]
        # Numerical guard only.  Unlike the median path this never binds on
        # counted data: any region with any exposure has mu >> 1e-3, and
        # regions with none are masked upstream.
        return jnp.maximum(blur, 1e-3)  # [photons/Pixel]

    if valid is None:
        valid = jnp.ones_like(imgs)
    return lax.map(process_one, (imgs, valid.astype(imgs.dtype)))  # [photons/Pixel]


@jit
def build_3d_cov(params):
    # params: [L11, L21, L22, L31, L32, L33]
    L = jnp.array(
        [
            [params[0], 0.0, 0.0],
            [params[1], params[2], 0.0],
            [params[3], params[4], params[5]],
        ]
    )
    # The safety floor is now 1 micrometer (1e-6 meters)
    # instead of 100 millimeters!
    L = L.at[0, 0].set(jnp.abs(L[0, 0]) + 1e-6)
    L = L.at[1, 1].set(jnp.abs(L[1, 1]) + 1e-6)
    L = L.at[2, 2].set(jnp.abs(L[2, 2]) + 1e-6)
    return L @ L.T


@partial(
    jit,
    static_argnames=[
        "patch_size",
        "fit_mosaicity",
        "mosaicity_radial",
        "shape_spherical",
        "shape_fit_normalized",
    ],
)
def global_shape_objective(
    params,
    patches,
    bgs,
    drs,
    dcs,
    P_mats,
    distances,
    R_mats,
    streak_dirs,
    patch_size,
    fit_mosaicity,
    mosaicity_radial=False,
    shape_spherical=False,
    shape_fit_normalized=False,
):
    # 1. Build the crystal shape tensor.  The spherical constraint is the
    # hypothesis test that the anisotropy is streak physics, not sample
    # volume: a sphere is rotation-invariant, so it also removes any
    # sample<->lab convention question from the shape pathway.
    if shape_spherical:
        r_s = jnp.abs(params[0]) + 1e-6
        Sigma_shape_sample = jnp.eye(3) * r_s**2
    else:
        Sigma_shape_sample = build_3d_cov(params[:6])

    # 2. Handle the optional Mosaicity Tensor
    if fit_mosaicity:
        eta = jnp.abs(params[6]) + 1e-6
        Sigma_eta_base = jnp.eye(3) * (eta**2)
    else:
        # If disabled, the tensor is perfectly zeroed out.
        eta = 0.0
        Sigma_eta_base = jnp.zeros((3, 3))

    def fit_one_peak(patch, bg, dr, dc, P_true, D_i, R_gonio, streak3):
        # 3. Rotate Crystal Shape to the Lab Frame
        Sigma_shape_lab = R_gonio @ Sigma_shape_sample @ R_gonio.T

        if mosaicity_radial:
            # A mosaic block rotated within the scattering plane stays
            # reflective at an adjusted wavelength, so the mosaic spread
            # streaks the spot along the 2-theta gradient (measured 3.7x
            # radial-to-tangential on cg4d-t4-lysozyme) -- not the
            # isotropic blur below.  streak3 is the unit radial direction
            # in the lab; the projected displacement per radian of mosaic
            # rotation is D_i * P streak3, which also carries the
            # incidence-angle stretch of the footprint.
            Sigma_2D_physical = P_true @ Sigma_shape_lab @ P_true.T
            streak_pix = P_true @ (D_i * streak3)
            Sigma_2D_physical = Sigma_2D_physical + (eta**2) * jnp.outer(
                streak_pix, streak_pix
            )
        else:
            # 4. Add the tensors. If fit_mosaicity is False, D_i * 0 = 0.
            Sigma_total_3D = Sigma_shape_lab + (D_i**2) * Sigma_eta_base
            Sigma_2D_physical = P_true @ Sigma_total_3D @ P_true.T

        # 5. Exact 2D Projection and Pixel Conversion
        Sigma_2D = Sigma_2D_physical / (1.0**2)  # assuming 1.0mm pitch in P_true

        var_r = jnp.maximum(Sigma_2D[0, 0], 1e-6)
        var_c = jnp.maximum(Sigma_2D[1, 1], 1e-6)

        max_cov = jnp.sqrt(var_r * var_c) * 0.999
        cov_rc = jnp.clip(Sigma_2D[0, 1], -max_cov, max_cov)

        det_sigma = var_r * var_c - cov_rc**2

        a = var_c / det_sigma
        b = -cov_rc / det_sigma
        c = var_r / det_sigma

        dr_grid = dr
        dc_grid = dc

        template = jnp.exp(
            -0.5 * (a * dc_grid**2 + 2.0 * b * dr_grid * dc_grid + c * dr_grid**2)
        )

        y_sub = patch - bg

        # Calculate the exact least-squares amplitude
        amp = jnp.sum(y_sub * template) / jnp.maximum(
            jnp.sum(template * template), 1e-6
        )

        residual = y_sub - amp * template
        sse = jnp.sum(residual**2)
        if shape_fit_normalized:
            # Per-patch power normalization: a patch votes with its misfit
            # FRACTION, not its brightness -- squared-error scales with
            # brightness^2, so an unnormalized mean lets a handful of
            # bright near-beam tails outvote thousands of typical peaks
            # (measured: top-3 patches carried 50% of the widening
            # leverage on cg4d-t4-lysozyme).
            sse = sse / (jnp.sum(y_sub**2) + 1e-6)
        return sse

    mses = vmap(fit_one_peak)(
        patches, bgs, drs, dcs, P_mats, distances, R_mats, streak_dirs
    )
    return jnp.mean(mses)


# Bind the val_and_grad wrapper to recognize the new static argument
val_and_grad_fn = jit(
    jax.value_and_grad(global_shape_objective),
    static_argnames=[
        "patch_size",
        "fit_mosaicity",
        "mosaicity_radial",
        "shape_spherical",
        "shape_fit_normalized",
    ],
)


def optimize_global_crystal(
    patches,
    bgs,
    drs,
    dcs,
    P_mats,
    distances,
    R_mats,
    streak_dirs,
    fit_mosaicity=False,
    mosaicity_radial=False,
    shape_spherical=False,
    mosaicity_bound_rad=0.010,
    shape_fit_normalized=False,
):
    # 1. Dynamically size the optimizer state based on the configuration
    if fit_mosaicity:
        scales = np.array([1e-3] * 7)
        x0_phys = np.array([1e-3, 0.0, 1e-3, 0.0, 0.0, 1e-3, 1e-3])
        print("\n  > Optimizing 3D Crystal Tensor + Distance-Dependent Mosaicity...")
    else:
        scales = np.array([1e-3] * 6)
        x0_phys = np.array([1e-3, 0.0, 1e-3, 0.0, 0.0, 1e-3])
        print("\n  > Optimizing 3D Global Effective Tensor (Constant Distance)...")

    def scipy_objective(x_opt):
        x_phys = x_opt * scales
        val, grad_phys = val_and_grad_fn(
            jnp.array(x_phys),
            patches,
            bgs,
            drs,
            dcs,
            P_mats,
            distances,
            R_mats,
            streak_dirs,
            patches.shape[-1],
            fit_mosaicity=fit_mosaicity,
            mosaicity_radial=mosaicity_radial,
            shape_spherical=shape_spherical,
            shape_fit_normalized=shape_fit_normalized,
        )
        grad_opt = np.array(grad_phys, dtype=np.float64) * scales
        return np.array(val, dtype=np.float64), grad_opt

    bounds = [(None, None)] * (7 if fit_mosaicity else 6)

    # 1. Physical bounds (in METERS). 3.0 mm = 0.003 meters.
    max_radius_meters = 0.003

    # 2. Diagonals (Indices 0, 2, 5) -> Must be positive, capped at max radius
    for idx in [0, 2, 5]:
        bounds[idx] = (1e-6, max_radius_meters / scales[idx])

    # 3. Off-diagonals (Indices 1, 3, 4) -> Symmetric bounds to prevent extreme skew
    for idx in [1, 3, 4]:
        bounds[idx] = (
            -max_radius_meters / scales[idx],
            max_radius_meters / scales[idx],
        )

    if shape_spherical:
        # Only params[0] (the sphere radius) is live; pin the rest of the
        # Cholesky so the optimizer state stays 6+1 wide.
        for idx in [1, 2, 3, 4, 5]:
            bounds[idx] = (0.0, 1e-12)

    # 4. Mosaicity bound (if active; 10 mrad default)
    if fit_mosaicity:
        bounds[6] = (1e-6, mosaicity_bound_rad / scales[6])

    x0_opt = x0_phys / scales
    res = scipy.optimize.minimize(
        scipy_objective,
        x0_opt,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 250, "disp": False},
    )

    x_final_phys = res.x * scales
    print(f"  > Global Optimization Complete. (Final MSE: {res.fun:.2f})")

    # --- EXTRACT PHYSICAL DIMENSIONS ---
    Sigma_shape_opt = np.array(build_3d_cov(jnp.array(x_final_phys[:6])))
    eigvals = np.linalg.eigvalsh(Sigma_shape_opt)
    principal_axes_mm = np.sqrt(np.maximum(eigvals, 0.0)) * 1000.0

    if fit_mosaicity:
        print("  [Separated Sample Properties]")
        print(f"  > Mosaicity (\u03b7): {abs(x_final_phys[6]) * 1000:.3f} mrad")
        print("  > Pure Crystal Shape (1\u03c3 radii):")
    else:
        print("  [Effective Sample Properties]")
        print("  > Combined Footprint (Shape + Blur + Mosaicity):")

    print(f"      Minor: {principal_axes_mm[0]:.4f} mm")
    print(f"      Mid:   {principal_axes_mm[1]:.4f} mm")
    print(f"      Major: {principal_axes_mm[2]:.4f} mm\n")

    return x_final_phys


# =====================================================================
# API WRAPPER FOR BACKWARD COMPATIBILITY
# =====================================================================


@dataclass(frozen=True)
class RunPeaks:
    """Lightweight dataclass to mock DetectorPeaks for the unrolled plotter."""

    image_index: list
    intensity: list
    peak_rows: list
    peak_cols: list
    var_u: list
    var_v: list
    cov_uv: list
    ki_vec: np.ndarray


def _render_run_unrolled_plot(args):
    """Standalone plotting function for generating unrolled plots per run."""
    out_name, peaks, images, detectors, instrument = args

    import matplotlib.pyplot as plt
    from subhkl.viz.detector_assembly import plot_unrolled_detector

    # Force non-interactive backend for thread safety
    if plt.get_backend().lower() != "agg":
        plt.switch_backend("Agg")

    plot_unrolled_detector(
        peaks, images, detectors, out_name=out_name, instrument=instrument
    )


def integrate_peaks_rbf_ssn(
    peak_dict: Dict,
    peaks_obj,
    sigmas: List[float],
    show_progress: bool,
    all_R: np.ndarray = None,
    sample_offset: np.ndarray = None,
    ki_vec: np.ndarray = None,
    nominal_sigma: float = 2.0,
    anisotropic: bool = False,
    fit_mosaicity: bool = False,
    mosaicity_radial: bool = False,
    shape_spherical: bool = False,
    mosaicity_bound_rad: float = 0.010,
    shape_fit_min_snr: float = 0.0,
    shape_fit_normalized: bool = False,
    matrix_free_profile: str = "gaussian",
    matrix_free_fp_target: float | None = None,
    static_mask_file: str | None = None,
    border_width: int = 0,
    chunk_size: int = 1024,
    create_visualizations: bool = False,
    file_prefix: str = None,
    max_workers: int = None,
    gonio_axes: np.ndarray = None,
    gonio_angles: np.ndarray = None,
    gonio_offsets: np.ndarray = None,
):
    """
    Args:
        peak_dict: Dictionary containing peak arrays
        peaks_obj: Instrument mapping object
        sigmas: List of unstretched peak radii
        nominal_sigma: Fallback sigma for crushed peaks
    Returns:
        res: RBFResult containing intensities and sigI
    """

    class RBFResult:
        def __init__(self):
            self.h, self.k, self.l = [], [], []
            self.intensity, self.sigma = [], []
            self.tt, self.az, self.wavelength = [], [], []
            self.run_id, self.bank, self.xyz = [], [], []
            self.var_u, self.var_v, self.cov_uv = [], [], []
            self.peak_rows, self.peak_cols = [], []
            self.image_index = []

    res = RBFResult()
    if sample_offset is None:
        sample_offset = np.zeros(3)
    if ki_vec is None:
        ki_vec = np.array([0, 0, 1.0])

    img_keys_ordered = sorted(peak_dict.keys())
    total_images = len(img_keys_ordered)

    def get_safe_R(img_key, seq_idx, run_id):
        if all_R is not None:
            if all_R.ndim == 3:
                # Safely handle uncompressed (per-image) vs compressed (per-run) arrays
                if len(all_R) == total_images:
                    return all_R[seq_idx]
                else:
                    return all_R[run_id] if run_id < len(all_R) else all_R[0]
            return all_R

        # Dynamic fallback using exact goniometer kinematics
        if gonio_axes is not None and gonio_angles is not None:
            # 1. Extract the specific angle array for this run
            if gonio_angles.ndim == 2:
                num_axes = len(gonio_axes)
                if gonio_angles.shape[1] == num_axes:
                    ang = (
                        gonio_angles[run_id, :]
                        if run_id < gonio_angles.shape[0]
                        else gonio_angles[0, :]
                    )
                else:
                    ang = (
                        gonio_angles[:, run_id]
                        if run_id < gonio_angles.shape[1]
                        else gonio_angles[:, 0]
                    )
            else:
                ang = gonio_angles

            # 2. Compute the exact Rotation matrix
            from scipy.spatial.transform import Rotation

            R_cum = np.eye(3)
            num_axes = len(gonio_axes)

            safe_offsets = (
                gonio_offsets if gonio_offsets is not None else np.zeros(num_axes)
            )

            # Compose rotation from innermost to outermost
            for i in range(num_axes):
                direction = gonio_axes[i][:3]
                direction_mult = gonio_axes[i][3] if len(gonio_axes[i]) > 3 else 1.0

                axis_norm = np.linalg.norm(direction)
                if axis_norm > 0:
                    direction = direction / axis_norm

                true_angle = ang[i] + safe_offsets[i]
                theta_rad = np.radians(true_angle * direction_mult)

                R_i = Rotation.from_rotvec(theta_rad * direction).as_matrix()
                R_cum = R_cum @ R_i

            return R_cum

        return np.eye(3)

    from subhkl.instrument.goniometer import sample_to_lab

    def get_s_lab_for_img(img_key_str, run_id, R_val):
        # 1. Use exact dynamic kinematics if available
        if gonio_axes is not None and gonio_angles is not None:
            if gonio_angles.ndim == 2:
                num_axes = len(gonio_axes)
                if gonio_angles.shape[1] == num_axes:
                    ang = (
                        gonio_angles[run_id, :]
                        if run_id < gonio_angles.shape[0]
                        else gonio_angles[0, :]
                    )
                else:
                    ang = (
                        gonio_angles[:, run_id]
                        if run_id < gonio_angles.shape[1]
                        else gonio_angles[:, 0]
                    )
            else:
                ang = gonio_angles

            offsets = sample_offset
            if offsets is not None and offsets.ndim == 1:
                offsets_full = np.zeros((len(gonio_axes), 3))
                offsets_full[-1] = offsets
            elif offsets is None:
                offsets_full = np.zeros((len(gonio_axes), 3))
            else:
                offsets_full = offsets

            return sample_to_lab(
                np.array([0.0, 0.0, 0.0]),
                gonio_axes,
                ang,
                offsets_full,
                zero_offsets=gonio_offsets,
            )

        # 2. Legacy fallback
        s_off = (
            sample_offset
            if sample_offset is not None and sample_offset.ndim == 1
            else (sample_offset[-1] if sample_offset is not None else np.zeros(3))
        )
        return R_val @ s_off if R_val is not None else s_off

    max_sigma = max(sigmas)

    # --- PHASE 1: GATHER AND BATCH ---
    images_list = []
    batched_banks = []
    all_frames = []
    all_rs, all_cs = [], []
    _meta_h, _meta_k, _meta_l, _meta_wl = [], [], [], []
    meta_keys = []
    meta_harmonics = []

    frame_counter = 0

    for seq_idx, img_key in enumerate(
        tqdm(img_keys_ordered, disable=not show_progress, desc="Batching Images")
    ):
        p_data = peak_dict[img_key]
        i_arr, j_arr, h_arr, k_arr, l_arr, wl_arr = p_data

        initial_peaks_count = len(i_arr)
        if initial_peaks_count == 0:
            continue

        hkl_sq = h_arr**2 + k_arr**2 + l_arr**2
        unique_peaks = {}

        # Exact crystallographic harmonic deduplication for Quasi-Laue
        for idx in range(initial_peaks_count):
            h, k, l = int(h_arr[idx]), int(k_arr[idx]), int(l_arr[idx])

            if h == 0 and k == 0 and l == 0:
                continue

            g = np.gcd.reduce([abs(h), abs(k), abs(l)])
            fund_hkl = (h // g, k // g, l // g)

            if fund_hkl not in unique_peaks:
                unique_peaks[fund_hkl] = {
                    "rep_idx": idx,  # Representative spatial coordinate
                    "hkl_sq": hkl_sq[idx],
                    "harmonic_indices": [idx],  # Track ALL overlapping harmonics
                }
            else:
                unique_peaks[fund_hkl]["harmonic_indices"].append(idx)
                # Keep the lowest harmonic as the spatial representative
                if hkl_sq[idx] < unique_peaks[fund_hkl]["hkl_sq"]:
                    unique_peaks[fund_hkl]["rep_idx"] = idx
                    unique_peaks[fund_hkl]["hkl_sq"] = hkl_sq[idx]

        # Extract the processed data, sorted by original index to maintain determinism
        keep_data = sorted(unique_peaks.values(), key=lambda x: x["rep_idx"])
        actual_peaks_count = len(keep_data)

        det = peaks_obj.get_detector_by_img(img_key)
        run_id = peaks_obj.image.get_run_id(img_key)

        current_R = get_safe_R(img_key, seq_idx, run_id)
        s_lab = get_s_lab_for_img(img_key, run_id, current_R)

        batch_rs = np.array([i_arr[d["rep_idx"]] for d in keep_data])
        batch_cs = np.array([j_arr[d["rep_idx"]] for d in keep_data])

        bank_tt, bank_az = det.pixel_to_angles(
            batch_rs, batch_cs, sample_offset=s_lab, ki_vec=ki_vec
        )

        if show_progress and initial_peaks_count != actual_peaks_count:
            physical_b = peaks_obj.image.bank_mapping.get(img_key, img_key)
            comp_ratio = (1.0 - actual_peaks_count / initial_peaks_count) * 100
            tqdm.write(
                f"Bank {physical_b} [Run {run_id}]: "
                f"Harmonics Filtered {initial_peaks_count} -> {actual_peaks_count} "
                f"({comp_ratio:.1f}% compression)"
            )

        image_raw = np.nan_to_num(
            peaks_obj.image.ims[img_key], nan=0.0, posinf=0.0, neginf=0.0
        )
        images_list.append(image_raw)
        batched_banks.append(peaks_obj.image.bank_mapping.get(img_key, img_key))

        for data in keep_data:
            idx = data["rep_idx"]
            all_frames.append(frame_counter)
            all_rs.append(i_arr[idx])
            all_cs.append(j_arr[idx])

            meta_harmonics.append(data["harmonic_indices"])  # Save the multiplet list!
            meta_keys.append(img_key)

        frame_counter += 1

    if not images_list:
        return res

    images_batch = np.stack(images_list)
    frames = np.array(all_frames, dtype=int)

    # --- PHASE 1.5: GLOBAL EFFECTIVE TENSOR OPTIMIZATION ---
    B, H, W = images_batch.shape
    bw = max(border_width, 5)

    all_P_mats = []
    all_R_mats = []
    all_distances = []
    all_streak_dirs = []
    ki_hat = np.asarray(ki_vec, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)

    for idx, img_key in enumerate(meta_keys):
        det = peaks_obj.get_detector_by_img(img_key)
        seq_idx = frames[idx]  # frames stores the image sequence index
        run_id = peaks_obj.image.get_run_id(img_key)  # Get the real run_id

        current_R = get_safe_R(img_key, seq_idx, run_id)
        s_lab = get_s_lab_for_img(img_key, run_id, current_R)

        pixel_xyz = det.pixel_to_lab(all_rs[idx], all_cs[idx])
        k_f = pixel_xyz - s_lab
        distance = np.linalg.norm(k_f)
        all_distances.append(distance)
        k_f_hat = k_f / distance

        # Unit radial (2-theta gradient) direction in the lab: the axis
        # the mosaic spread streaks the spot along.  Degenerate frames
        # (near-forward peaks) get a zero vector: no streak contribution.
        t3 = np.cross(ki_hat, k_f_hat)
        t3_norm = np.linalg.norm(t3)
        if t3_norm > 1e-6:
            r3 = np.cross(t3 / t3_norm, k_f_hat)
            all_streak_dirs.append(r3 / np.linalg.norm(r3))
        else:
            all_streak_dirs.append(np.zeros(3))

        # Detector Normal & Orthogonal Projection
        n_det = np.cross(det.uhat, det.vhat)
        n_det_hat = n_det / np.linalg.norm(n_det)
        P_ortho = np.vstack([det.uhat, det.vhat])

        # The Central Projection Skew Matrix
        cos_alpha = np.dot(k_f_hat, n_det_hat)
        ca_sign = 1.0 if cos_alpha >= 0 else -1.0
        cos_alpha = ca_sign * max(abs(cos_alpha), 0.01)
        Skew = np.eye(3) - np.outer(k_f_hat, n_det_hat) / cos_alpha

        # Pixel Pitch Scaling Matrix
        pixel_pitch_u = det.width / (det.m - 1)
        pixel_pitch_v = det.height / (det.n - 1)
        S_pix = np.diag([1.0 / pixel_pitch_u, 1.0 / pixel_pitch_v])

        # The Ultimate Projection Matrix
        P_final = S_pix @ P_ortho @ Skew
        all_P_mats.append(P_final)
        all_R_mats.append(current_R if current_R is not None else np.eye(3))

    all_P_mats = np.array(all_P_mats)
    all_R_mats = np.array(all_R_mats)
    all_distances = np.array(all_distances)
    all_streak_dirs = np.array(all_streak_dirs)

    # ==========================================
    # 2. Extract exact patches for global optimization
    # ==========================================
    opt_P = 15
    opt_half = opt_P // 2
    opt_patches, opt_bgs, opt_drs, opt_dcs = [], [], [], []
    opt_Pmats, opt_dists, opt_Rmats, opt_streaks = [], [], [], []

    if show_progress:
        print(f"  > 3D Tensor Optimization: Using ALL {len(frames)} peaks.")

    # Pad images once for the exact patch extraction
    pad_images = np.pad(
        images_batch, ((0, 0), (opt_P, opt_P), (opt_P, opt_P)), mode="reflect"
    )

    for idx in range(len(frames)):
        f, r, c = frames[idx], all_rs[idx], all_cs[idx]

        # Boundary safety check (matches your H, W constraints)
        if not (bw < int(round(r)) < H - bw and bw < int(round(c)) < W - bw):
            continue

        opt_Pmats.append(all_P_mats[idx])
        opt_dists.append(all_distances[idx])
        opt_Rmats.append(all_R_mats[idx])
        opt_streaks.append(all_streak_dirs[idx])

        ri, ci = int(round(r)) + opt_P, int(round(c)) + opt_P

        # Bounding box bounds
        r_min, r_max = ri - opt_half, ri + opt_half + 1
        c_min, c_max = ci - opt_half, ci + opt_half + 1

        patch = pad_images[f, r_min:r_max, c_min:c_max].astype(np.float32)

        # Local background estimation (edges of the 15x15 patch)
        bg_mask = np.ones_like(patch, dtype=bool)
        bg_mask[2:-2, 2:-2] = False
        bg = np.median(patch[bg_mask])

        opt_patches.append(patch)
        opt_bgs.append(bg)

        # Sub-pixel grid shifts
        rr, cc = np.meshgrid(
            np.arange(r_min, r_max) - opt_P,
            np.arange(c_min, c_max) - opt_P,
            indexing="ij",
        )
        opt_drs.append(rr - r)
        opt_dcs.append(cc - c)

    # We require at least 15 valid peaks to mathematically constrain a 6-parameter 3D tensor
    MIN_PEAKS_FOR_GLOBAL_FIT = 15

    # Weak patches carry no shape information -- only background.  Within
    # the fitting window a wide template is a pedestal soaker (its MSE
    # surface is nearly flat in width beyond half the window), so a fit
    # over all peaks lets the many weak ones drag the widths to their
    # bounds.  Restrict the SHAPE fit to significant peaks; integration
    # itself still covers every prediction.
    if shape_fit_min_snr > 0 and len(opt_patches) >= MIN_PEAKS_FOR_GLOBAL_FIT:
        keep = []
        for patch, bg in zip(opt_patches, opt_bgs):
            core = patch[opt_half - 2 : opt_half + 3, opt_half - 2 : opt_half + 3]
            signal = float(core.sum() - 25.0 * bg)
            noise = float(np.sqrt(max(25.0 * bg, 1.0)))
            keep.append(signal / noise >= shape_fit_min_snr)
        keep = np.array(keep)
        if keep.sum() >= MIN_PEAKS_FOR_GLOBAL_FIT:
            if show_progress:
                print(
                    f"  > Shape fit restricted to {int(keep.sum())} of "
                    f"{len(keep)} peaks with core SNR >= {shape_fit_min_snr}."
                )
            sel = np.where(keep)[0]
            opt_patches = [opt_patches[i] for i in sel]
            opt_bgs = [opt_bgs[i] for i in sel]
            opt_drs = [opt_drs[i] for i in sel]
            opt_dcs = [opt_dcs[i] for i in sel]
            opt_Pmats = [opt_Pmats[i] for i in sel]
            opt_dists = [opt_dists[i] for i in sel]
            opt_Rmats = [opt_Rmats[i] for i in sel]
            opt_streaks = [opt_streaks[i] for i in sel]
        elif show_progress:
            print(
                f"  > Only {int(keep.sum())} peaks pass SNR >= "
                f"{shape_fit_min_snr}; shape fit keeps all peaks."
            )

    if not anisotropic or len(opt_patches) < MIN_PEAKS_FOR_GLOBAL_FIT:
        if show_progress:
            if not anisotropic:
                tqdm.write(
                    "  > Anisotropic profile fitting disabled. Using nominal isotropic circles."
                )
            else:
                tqdm.write(
                    f"  > Too few valid peaks ({len(opt_patches)}) for 3D tensor fit. Falling back to nominal isotropic circles."
                )

        all_var_u = np.full(len(all_rs), nominal_sigma**2, dtype=np.float32)
        all_var_v = np.full(len(all_rs), nominal_sigma**2, dtype=np.float32)
        all_cov_uv = np.zeros(len(all_rs), dtype=np.float32)
    else:
        # ==========================================
        # 4. Run the Global Optimizer
        # ==========================================
        res_x = optimize_global_crystal(
            jnp.array(opt_patches),
            jnp.array(opt_bgs),
            jnp.array(opt_drs),
            jnp.array(opt_dcs),
            jnp.array(opt_Pmats),
            jnp.array(opt_dists),
            jnp.array(opt_Rmats),
            jnp.array(opt_streaks),
            fit_mosaicity=fit_mosaicity,
            mosaicity_radial=mosaicity_radial,
            shape_spherical=shape_spherical,
            mosaicity_bound_rad=mosaicity_bound_rad,
            shape_fit_normalized=shape_fit_normalized,
        )

        # 5. Project the EXACT 2D footprints for ALL peaks
        if shape_spherical:
            Sigma_shape_jnp = jnp.eye(3) * (abs(res_x[0]) + 1e-6) ** 2
        else:
            Sigma_shape_jnp = build_3d_cov(jnp.array(res_x[:6]))

        if fit_mosaicity:
            eta_opt = abs(res_x[6]) + 1e-6
            Sigma_eta_jnp = jnp.eye(3) * (eta_opt**2)
        else:
            eta_opt = 0.0
            Sigma_eta_jnp = jnp.zeros((3, 3))

        @jit
        def project_all_shapes(P_mats, dists, R_mats, streaks):
            def project_one(P, D_i, R_gonio, streak3):
                # Rotate to Lab Frame before projecting
                Sigma_shape_lab = R_gonio @ Sigma_shape_jnp @ R_gonio.T
                if mosaicity_radial:
                    Sigma_2D = P @ Sigma_shape_lab @ P.T
                    streak_pix = P @ (D_i * streak3)
                    return Sigma_2D + (eta_opt**2) * jnp.outer(streak_pix, streak_pix)
                Sigma_total = Sigma_shape_lab + (D_i**2) * Sigma_eta_jnp
                return P @ Sigma_total @ P.T

            return vmap(project_one)(P_mats, dists, R_mats, streaks)

        all_Sigma_2D = project_all_shapes(
            jnp.array(all_P_mats),
            jnp.array(all_distances),
            jnp.array(all_R_mats),
            jnp.array(all_streak_dirs),
        )

        all_var_u = np.array(all_Sigma_2D[:, 0, 0])
        all_var_v = np.array(all_Sigma_2D[:, 1, 1])
        all_cov_uv = np.array(all_Sigma_2D[:, 0, 1])

    # --- PHASE 2: GPU INTEGRATION ---
    # Amplitude-only global solve per image on the finder's rate-map noise
    # model.  This is the only integration path: the per-patch fit it
    # replaced compensated for patch locality (force-the-target, Voronoi
    # pixel masks, Huber core protection) that a per-image joint solve
    # removes by construction, and lost to it on every common reflection
    # set measured.  Imported here because matrix_free imports
    # compute_rate_batch from this module.
    from subhkl.search.matrix_free import integrate_reflections_matrix_free

    static_valid = None
    if static_mask_file is not None:
        from subhkl.search.static_mask import load_mask_for_banks

        static_valid = load_mask_for_banks(
            static_mask_file, batched_banks, images_batch.shape[1:]
        )

    integrated_results = integrate_reflections_matrix_free(
        images_batch,
        frames,
        all_rs,
        all_cs,
        var_us=all_var_u,
        var_vs=all_var_v,
        cov_uvs=all_cov_uv,
        ref_sigma=1.0,
        max_sigma=max_sigma,
        static_valid=static_valid,
        profile=matrix_free_profile,
        fp_target=matrix_free_fp_target,
        show_progress=show_progress,
    )

    # --- PHASE 3: GEOMETRY AND METADATA MAPPING ---
    results_by_img = defaultdict(list)
    for i in range(len(meta_keys)):
        results_by_img[meta_keys[i]].append(i)

    # Replaces the old plot_tasks list
    runs_plot_data = defaultdict(lambda: {"images": {}, "detectors": {}})

    # static per-run data
    if create_visualizations:
        for img_key, image_raw in peaks_obj.image.ims.items():
            run_id = peaks_obj.get_run_id(img_key)
            det = peaks_obj.get_detector_by_img(img_key)
            runs_plot_data[run_id]["images"][img_key] = image_raw
            runs_plot_data[run_id]["detectors"][img_key] = det

    for img_key, indices in tqdm(
        results_by_img.items(), disable=not show_progress, desc="Mapping Geometry"
    ):
        physical_bank = peaks_obj.image.bank_mapping.get(img_key, img_key)
        det = peaks_obj.get_detector_by_img(img_key)
        run_id = peaks_obj.image.get_run_id(img_key)  # Safely use .image here

        image_raw = np.nan_to_num(
            peaks_obj.image.ims[img_key], nan=0.0, posinf=0.0, neginf=0.0
        )
        H, W = image_raw.shape
        bw = border_width

        seq_idx = img_keys_ordered.index(img_key)
        current_R = get_safe_R(img_key, seq_idx, run_id)
        s_lab = get_s_lab_for_img(img_key, run_id, current_R)

        img_rs = [all_rs[idx] for idx in indices]
        img_cs = [all_cs[idx] for idx in indices]
        bank_tt, bank_az = det.pixel_to_angles(
            np.array(img_rs), np.array(img_cs), sample_offset=s_lab
        )

        valid_global_indices = []
        valid_local_indices = []

        for local_idx, global_idx in enumerate(indices):
            r = float(integrated_results[global_idx, 1])
            c = float(integrated_results[global_idx, 2])
            if (bw <= r < H - bw) and (bw <= c < W - bw):
                valid_global_indices.append(global_idx)
                valid_local_indices.append(local_idx)

        if not valid_global_indices:
            continue

        for local_idx, global_idx in zip(valid_local_indices, valid_global_indices):
            intensity = float(integrated_results[global_idx, 0])
            sigI = float(integrated_results[global_idx, 4])

            r_center = float(integrated_results[global_idx, 1])
            c_center = float(integrated_results[global_idx, 2])
            var_u = float(all_var_u[global_idx])
            var_v = float(all_var_v[global_idx])
            cov_uv = float(all_cov_uv[global_idx])

            harmonic_indices = meta_harmonics[global_idx]
            p_data = peak_dict[img_key]

            for h_idx in harmonic_indices:
                res.h.append(p_data[2][h_idx])
                res.k.append(p_data[3][h_idx])
                res.l.append(p_data[4][h_idx])
                res.wavelength.append(p_data[5][h_idx])

                res.tt.append(float(bank_tt[local_idx]))
                res.az.append(float(bank_az[local_idx]))
                res.run_id.append(run_id)
                res.bank.append(physical_bank)
                res.intensity.append(intensity)
                res.sigma.append(sigI)

                # Store extended unrolled metadata
                res.image_index.append(img_key)
                res.peak_rows.append(r_center)
                res.peak_cols.append(c_center)
                res.var_u.append(var_u)
                res.var_v.append(var_v)
                res.cov_uv.append(cov_uv)

    # --- PHASE 4: PARALLEL VISUALIZATION (PER RUN) ---
    if create_visualizations and runs_plot_data:
        run_tasks = []

        for r_id, data in runs_plot_data.items():
            # Extract only the peaks belonging to this run_id
            mask = [i for i, run in enumerate(res.run_id) if run == r_id]

            base_dir = os.path.dirname(file_prefix) if file_prefix else ""
            image_label = peaks_obj.get_image_label(res.image_index[mask[0]])
            out_name = os.path.join(base_dir, f"{image_label}-pred.png")

            run_peaks = RunPeaks(
                image_index=[res.image_index[i] for i in mask],
                intensity=[res.intensity[i] for i in mask],
                peak_rows=[res.peak_rows[i] for i in mask],
                peak_cols=[res.peak_cols[i] for i in mask],
                var_u=[res.var_u[i] for i in mask],
                var_v=[res.var_v[i] for i in mask],
                cov_uv=[res.cov_uv[i] for i in mask],
                ki_vec=ki_vec,
            )

            run_tasks.append(
                (
                    out_name,
                    run_peaks,
                    data["images"],
                    data["detectors"],
                    peaks_obj.instrument,
                )
            )

        if max_workers is None:
            max_workers = os.cpu_count()

        max_workers = min(max_workers, len(run_tasks))

        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=ctx, max_workers=max_workers
        ) as executor:
            # Use submit instead of map to guarantee exceptions aren't swallowed
            futures = {
                executor.submit(_render_run_unrolled_plot, t): t[0] for t in run_tasks
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Rendering Detector Plots",
                disable=not show_progress,
            ):
                out_name = futures[future]
                try:
                    # Wait for the worker to finish and catch any exceptions
                    future.result()
                    tqdm.write(f"Saved: {out_name}")
                except Exception:
                    import traceback

                    print(f"Visualization failed for {out_name}:")
                    traceback.print_exc()

    return res
