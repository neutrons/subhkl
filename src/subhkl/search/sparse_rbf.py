import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from jax import jit
from jax import lax
from jax import vmap
import jax
import jax.numpy as jnp

import jax.scipy.optimize
import jax.scipy.signal
import scipy

from subhkl.search.ssn import SparseBasisPursuit

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
def compute_rate_batch(imgs, filter_size):
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

    Args:
        imgs: [photons/Pixel]
        filter_size: [Pixel^0.5]
    Returns:
        [photons/Pixel]
    """

    def process_one(img):
        F = jnp.stack(
            [
                _box_mean_2d((img <= k).astype(img.dtype), filter_size)
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

        bright = _box_mean_2d(img, filter_size)
        rate = jnp.where(any_hit, mu, bright)

        blur = jax_gaussian_blur_2d(rate)  # [photons/Pixel]
        # Numerical guard only.  Unlike the median path this never binds on
        # counted data: any region with any exposure has mu >> 1e-3, and
        # regions with none are masked upstream.
        return jnp.maximum(blur, 1e-3)  # [photons/Pixel]

    return lax.map(process_one, imgs)  # [photons/Pixel]


class SparseRBFPeakFinder(SparseBasisPursuit):
    """
    Hierarchical Sparse RBF Peak Finder with Symmetric V-Cycle Basis Pursuit.

    Units:
        alpha: [-] (Z-score threshold)
        gamma: [-] (Besov weight power)
        min_sigma / max_sigma: [Pixel^0.5]
        ref_sigma: [Pixel^0.5]
    """

    def __init__(
        self,
        alpha: float = 15.0,
        gamma: float = 2.0,
        min_sigma: float = 0.5,
        max_sigma: float = 8.0,
        chunk_size: int = 128,
        loss: str = "gaussian",
        border_width: int = 0,
        num_sigmas: int = 32,
        show_steps: bool = True,
        auto_tune_alpha: bool = False,
        candidate_alphas: list = None,
    ):
        super().__init__(alpha=alpha, gamma=gamma, loss=loss, ref_sigma=1.0)

        self.alpha = alpha  # [-]
        self.gamma = gamma  # [-]
        self.ref_sigma = 1.0  # [Pixel^0.5]
        self.min_sigma = min_sigma  # [Pixel^0.5]
        self.max_sigma = max_sigma  # [Pixel^0.5]
        self.chunk_size = chunk_size  # [-]
        self.loss = loss  # [-]
        self.border_width = border_width  # [Pixel^0.5]
        self.show_steps = show_steps

        self.base_window_size = 64  # [Pixel^0.5]
        self.refine_patch_size = 15  # [Pixel^0.5]
        self.halo = 5  # [Pixel^0.5]
        self.max_local_peaks = 5  # [-]

        self.candidate_sigmas = jnp.linspace(
            min_sigma, max_sigma, num_sigmas
        )  # [Pixel^0.5]

        self.auto_tune_alpha = auto_tune_alpha
        # Reasonable defaults covering low-SNR to high-SNR regimes
        self.candidate_alphas = jnp.array(
            candidate_alphas or [10.0, 15.0, 20.0, 25.0, 30.0], dtype=jnp.float32
        )

    def _compute_background(self, patch, filter_size):
        # 2D Morphological Background
        med = jax_median_2d(patch, filter_size)
        blur = jax_gaussian_blur_2d(med)
        return jnp.maximum(blur, 1e-3)

    def _build_basis_matrix(self, x_grid, params):
        # params: [c_init, r, col, sigma]
        r = params[:, 1]
        col = params[:, 2]
        sigma = params[:, 3]

        def eval_one(ri, ci, si):
            return self._rbf_basis(x_grid, jnp.array([ri, ci]), si).flatten()

        return vmap(eval_one)(r, col, sigma).T

    @staticmethod
    def _rbf_basis(
        x_grid, y, sigma_long, theta=0.0, phi=0.0, anisotropic=False, sigma_short=1.5
    ):
        if not anisotropic:
            # ORIGINAL ISOTROPIC ERF LOGIC
            sig_sq2 = sigma_long * jnp.sqrt(2.0) + 1e-6
            erf_r = jax.scipy.special.erf(
                (x_grid[0] + 0.5 - y[0]) / sig_sq2
            ) - jax.scipy.special.erf((x_grid[0] - 0.5 - y[0]) / sig_sq2)
            erf_c = jax.scipy.special.erf(
                (x_grid[1] + 0.5 - y[1]) / sig_sq2
            ) - jax.scipy.special.erf((x_grid[1] - 0.5 - y[1]) / sig_sq2)
            return (jnp.pi / 2.0) * (sigma_long**2) * erf_r * erf_c
        else:
            # ANISOTROPIC 4x4 QUADRATURE LOGIC
            # 1. Build the Precision Matrix (Sigma^-1)
            cos_p = jnp.cos(phi)
            sin_p = jnp.sin(phi)

            var_l = jnp.maximum(sigma_long**2, 1e-6)
            var_s = jnp.maximum(sigma_short**2, 1e-6)

            a = (cos_p**2) / var_l + (sin_p**2) / var_s
            b = sin_p * cos_p * (1.0 / var_l - 1.0 / var_s)
            c = (sin_p**2) / var_l + (cos_p**2) / var_s

            # Setup 4x4 Sub-pixel Offsets (-0.375, -0.125, 0.125, 0.375)
            sub_offsets = jnp.array([-0.375, -0.125, 0.125, 0.375])

            # meshgrid 'xy' indexing: ox varies across cols (X), oy varies across rows (Y)
            ox, oy = jnp.meshgrid(sub_offsets, sub_offsets)

            # Distance from predicted center to pixel centers
            # x_grid[0] is yy (rows), x_grid[1] is xx (cols)
            dr_center = x_grid[0] - y[0]
            dc_center = x_grid[1] - y[1]

            # 3. Evaluate Gaussian at sub-points
            def eval_subpoint(ox_i, oy_i):
                dr = dr_center + oy_i  # row diff (Y) gets oy_i
                dc = dc_center + ox_i  # col diff (X) gets ox_i

                # 'a' applies to X (cols, dc), 'c' applies to Y (rows, dr)
                return jnp.exp(-0.5 * (a * dc**2 + 2.0 * b * dr * dc + c * dr**2))

            sub_evals = vmap(vmap(eval_subpoint))(ox, oy)

            # 4. Average the 16 evaluations and scale by analytic volume
            area_scalar = 2.0 * jnp.pi * sigma_long * sigma_short
            return jnp.mean(sub_evals, axis=(0, 1)) * area_scalar

    @staticmethod
    def _to_physical(params_raw, H, W, min_s, max_s):
        """
        Returns:
            [c: [photons/Pixel^2], r: [Pixel^0.5], col: [Pixel^0.5], sigma: [Pixel^0.5]]
        """
        params_reshaped = params_raw.reshape((-1, 4))
        c_raw, r_raw, c_col_raw, s_raw = params_reshaped.T
        c = jax.nn.softplus(c_raw)  # [photons/Pixel^2]
        r = jax.nn.sigmoid(r_raw) * H  # [Pixel^0.5]
        col = jax.nn.sigmoid(c_col_raw) * W  # [Pixel^0.5]
        sigma = min_s + jax.nn.sigmoid(s_raw) * (max_s - min_s)  # [Pixel^0.5]
        return jnp.stack([c, r, col, sigma], axis=1)

    @staticmethod
    def _to_unconstrained(params_phys, H, W, min_s, max_s):
        c, r, col, sigma = params_phys.T
        c_safe = jnp.maximum(c, 1e-9)
        c_raw = jnp.where(c_safe > 20.0, c_safe, jnp.log(jnp.expm1(c_safe)))
        r_safe = jnp.clip(r / H, 1e-6, 1.0 - 1e-6)
        r_raw = jax.scipy.special.logit(r_safe)
        c_safe = jnp.clip(col / W, 1e-6, 1.0 - 1e-6)
        c_col_raw = jax.scipy.special.logit(c_safe)
        s_norm = (sigma - min_s) / (max_s - min_s)
        s_safe = jnp.clip(s_norm, 1e-6, 1.0 - 1e-6)
        s_raw = jax.scipy.special.logit(s_safe)
        return jnp.stack([c_raw, r_raw, c_col_raw, s_raw], axis=1).ravel()

    @staticmethod
    def _predict_batch_physical(params_phys, x_grid, mask=None):
        """
        Returns:
            [photons/Pixel]
        """
        c, r, c_col, sigma = params_phys.T
        if mask is not None:
            c = c * mask  # [photons/Pixel^2]

        def eval_one(ci, ri, ci_col, si):
            # [photons/Pixel^2] * [Pixel] = [photons/Pixel]
            return ci * SparseRBFPeakFinder._rbf_basis(
                x_grid, jnp.array([ri, ci_col]), si
            )

        basis_stack = vmap(eval_one)(c, r, c_col, sigma)  # [photons/Pixel]
        return jnp.sum(basis_stack, axis=0)  # [photons/Pixel]

    @staticmethod
    def _predict_batch_scan(params_phys, x_grid):
        """
        Returns: [photons/Pixel]
        """

        def body(carry, param):
            c, r, col, sigma = param
            # [photons/Pixel^2] * [Pixel] = [photons/Pixel]
            term = c * SparseRBFPeakFinder._rbf_basis(
                x_grid, jnp.array([r, col]), sigma
            )
            return carry + term, None

        H, W = x_grid.shape[1], x_grid.shape[2]
        init = jnp.zeros((H, W), dtype=params_phys.dtype)  # [photons/Pixel]
        final_image, _ = lax.scan(body, init, params_phys)
        return final_image  # [photons/Pixel]

    @staticmethod
    def _joint_patch_objective(
        flat_params_unconstrained,
        patch_stat,
        patch_bg,
        x_grid,
        H,
        W,
        min_s,
        max_s,
        loss_code,
    ):
        """
        Evaluates the joint fit of all K candidates on the local patch.
        """
        # 1. Reshape and map back to physical bounds (positive amp, constrained coords)
        params_phys = SparseRBFPeakFinder._to_physical(
            flat_params_unconstrained, H, W, min_s, max_s
        )

        # 2. Render the combined footprint of all K candidates
        recon = SparseRBFPeakFinder._predict_batch_physical(params_phys, x_grid)
        recon_total = jnp.maximum(recon + patch_bg, 1e-9)

        # 3. Compute target loss (Poisson NLL or Gaussian MSE)
        if loss_code == 1:
            # Exact Poisson NLL
            loss = jnp.sum(
                recon_total - jax.scipy.special.xlogy(patch_stat, recon_total)
            )
        else:
            loss = 0.5 * jnp.sum((recon_total - patch_stat) ** 2)

        return loss

    @partial(
        jit,
        static_argnames=["self", "H", "W", "max_peaks_local", "loss_code", "do_merge"],
    )
    def _solve_dense(
        self,
        patch_stat,
        patch_bg,
        alpha_z_score,
        H,
        W,
        max_peaks_local,
        loss_code,
        do_merge,
    ):
        (float(H), float(W), self.min_sigma, self.max_sigma)  # [Pixel^0.5]
        yy, xx = jnp.indices((H, W))  # [Pixel^0.5]
        x_grid = jnp.array([yy, xx])  # [Pixel^0.5]

        max_k_rad = int(3.0 * self.max_sigma)  # [Pixel^0.5]
        k_grid = jnp.arange(-max_k_rad, max_k_rad + 1)  # [Pixel^0.5]

        init_params = jnp.zeros((max_peaks_local, 4))
        init_active = jnp.zeros(max_peaks_local, dtype=bool)
        init_state = (init_params, init_active, 0)

        def step_fn(state, _):
            params, active_mask, idx = state
            recon = self._predict_batch_physical(
                params, x_grid, active_mask
            )  # [photons/Pixel]

            def check_sigma(s):
                sig_sq2 = s * jnp.sqrt(2.0) + 1e-6  # [Pixel^0.5]

                # Separable basis kernels
                k_1d = jax.scipy.special.erf(
                    (k_grid + 0.5) / sig_sq2
                ) - jax.scipy.special.erf((k_grid - 0.5) / sig_sq2)  # [-]

                k_col = k_1d[:, None]  # [-]
                k_row = k_1d[None, :]  # [-]

                # Current expected model U_k
                recon_total = jnp.maximum(recon + patch_bg, 1e-3)  # [photons/Pixel]

                # Exact Poisson Gradient: 1 - (y_k / U_k)
                poisson_grad = (patch_stat / recon_total) - 1.0  # [-]

                # 1st Convolution: Dual Variable (Signal)
                temp_sig = jax.scipy.signal.correlate2d(
                    poisson_grad, k_col, mode="valid"
                )
                dual_var_unscaled = jax.scipy.signal.correlate2d(
                    temp_sig, k_row, mode="valid"
                )

                # Inverse model for spatially varying variance: 1 / U_k
                inv_recon = 1.0 / recon_total  # [Pixel/photons]

                # Square the 1D kernels for the variance convolution
                k_1d_sq = k_1d**2
                k_col_sq = k_1d_sq[:, None]
                k_row_sq = k_1d_sq[None, :]

                # 2nd Convolution: Local Variance Map
                temp_var = jax.scipy.signal.correlate2d(
                    inv_recon, k_col_sq, mode="valid"
                )
                local_var_unscaled = jax.scipy.signal.correlate2d(
                    temp_var, k_row_sq, mode="valid"
                )

                # Apply analytic volume scaling
                area_scalar = (jnp.pi / 2.0) * (s**2)  # [Pixel]

                dual_var_signal = dual_var_unscaled * area_scalar  # [photons]
                dual_var_variance = local_var_unscaled * (area_scalar**2)  # [photons]

                # Compute dimensionless Exact Z-Score Map
                z_score_map = dual_var_signal / jnp.sqrt(
                    jnp.maximum(dual_var_variance, 1e-12)
                )  # [-]

                # Find the maximum Z-score coordinate
                flat_idx = jnp.argmax(z_score_map)
                r_valid, c_valid = jnp.unravel_index(
                    flat_idx, z_score_map.shape
                )  # [Pixel^0.5]

                # --- EXACT LOG-PARABOLIC INTERPOLATION ---
                # We interpolate on the dual_var_signal to find the true physical center of the signal
                padded_dv = jnp.pad(dual_var_signal, 1, mode="edge")
                r_p, c_p = r_valid + 1, c_valid + 1  # [Pixel^0.5]

                safe_dv = jnp.maximum(padded_dv, 1e-6)
                val = jnp.log(safe_dv[r_p, c_p])  # [ln(photons)]
                val_up = jnp.log(safe_dv[r_p - 1, c_p])
                val_dn = jnp.log(safe_dv[r_p + 1, c_p])
                val_lf = jnp.log(safe_dv[r_p, c_p - 1])
                val_rt = jnp.log(safe_dv[r_p, c_p + 1])

                den_r = val_up - 2.0 * val + val_dn  # [-]
                den_r = jnp.minimum(den_r, -1e-6)
                dr = 0.5 * (val_up - val_dn) / den_r  # [Pixel^0.5]

                den_c = val_lf - 2.0 * val + val_rt  # [-]
                den_c = jnp.minimum(den_c, -1e-6)
                dc = 0.5 * (val_lf - val_rt) / den_c  # [Pixel^0.5]

                dr = jnp.clip(dr, -0.5, 0.5)  # [Pixel^0.5]
                dc = jnp.clip(dc, -0.5, 0.5)  # [Pixel^0.5]

                r_idx = r_valid + max_k_rad + dr  # [Pixel^0.5]
                c_idx = c_valid + max_k_rad + dc  # [Pixel^0.5]

                # Dimensional recovery of volumetric density for the SSN warm start
                k_1d_sq_sum = jnp.sum(k_1d_sq)  # [-]
                kernel_sq_norm = (area_scalar**2) * (k_1d_sq_sum**2)  # [Pixel^2]

                c_matched = (
                    dual_var_signal[r_valid, c_valid] / kernel_sq_norm
                )  # [photons / Pixel^2]
                c_init = jnp.maximum(c_matched, 0.0)  # [photons/Pixel^2]

                best_z_score = z_score_map[r_valid, c_valid]

                return best_z_score, jnp.array([c_init, r_idx, c_idx, s])

            vals, candidates = vmap(check_sigma)(self.candidate_sigmas)
            best_idx = jnp.argmax(vals)
            new_peak = candidates[best_idx]

            s_best = new_peak[3]  # [Pixel^0.5]

            # vals array directly holds the exact Z-scores now
            z_score = vals[best_idx]
            weight_best = (s_best / self.ref_sigma) ** self.gamma  # [-]

            is_strong = z_score > (
                alpha_z_score * weight_best
            )  # True dimensionless threshold check

            dummy_peak = jnp.array([0.0, 0.0, 0.0, 1.0])
            new_peak = jnp.where(is_strong, new_peak, dummy_peak)

            params = params.at[idx].set(new_peak)
            active_mask = active_mask.at[idx].set(is_strong)

            def run_opt(operand):
                p, a_mask = operand

                # The engine handles basis construction!
                A_masked = self._build_basis_matrix(x_grid, p) * a_mask

                # The engine handles BIC Auto-Tuning internally!
                c_sparse_stat, _, _, _ = self.tune_and_solve(
                    patch_stat.flatten(), patch_bg.flatten(), A_masked, p
                )

                c_sparse_norm = c_sparse_stat * a_mask
                return jnp.stack([c_sparse_norm, p[:, 1], p[:, 2], p[:, 3]], axis=1)

            def skip_opt(operand):
                p, _ = operand
                return p

            params = lax.cond(is_strong, run_opt, skip_opt, (params, active_mask))

            return (params, active_mask, idx + 1), None

        final_state, _ = lax.scan(step_fn, init_state, None, length=max_peaks_local)
        final_params, final_active, _ = final_state

        # 1. Isolate the active peaks found by the greedy search
        c, r, col, sigma = final_params.T
        active_mask = final_active & (c > 1e-9)
        num_active = jnp.sum(active_mask)

        # 2. Prepare the joint refinement function
        joint_loss_fn = partial(
            self._joint_patch_objective,
            patch_stat=patch_stat,
            patch_bg=patch_bg,
            x_grid=x_grid,
            H=H,
            W=W,
            min_s=self.min_sigma,
            max_s=self.max_sigma,
            loss_code=loss_code,
        )

        # Use JAX's reverse-mode autodiff to get the exact gradients
        grad_fn = jax.value_and_grad(joint_loss_fn)

        # Convert physical starting guesses to unconstrained space for unconstrained optimization
        unconstrained_init = self._to_unconstrained(
            final_params, H, W, self.min_sigma, self.max_sigma
        )

        # 3. Fixed-iteration Gradient Descent using lax.scan
        # This avoids XLA graph unrolling hangs while pushing overlapping peaks apart
        def refinement_step(state, _):
            current_params, current_lr = state
            loss, grads = grad_fn(current_params)

            # Simple Gradient Descent (or upgrade to Adam by tracking momentum in the state)
            # Only update parameters for peaks that were actually active in the greedy phase
            grad_mask = jnp.repeat(active_mask, 4)
            masked_grads = grads * grad_mask

            next_params = current_params - current_lr * masked_grads

            return (next_params, current_lr * 0.95), loss

        # Run for a fixed number of steps (e.g., 25)
        refinement_steps = 25
        initial_lr = 0.1
        (refined_unconstrained, _), loss_history = lax.scan(
            refinement_step,
            (unconstrained_init, initial_lr),
            None,
            length=refinement_steps,
        )

        # 4. Map back to physical parameters
        refined_params_phys = self._to_physical(
            refined_unconstrained, H, W, self.min_sigma, self.max_sigma
        )

        # Enforce the mask again so dead peaks stay dead
        final_params = jnp.where(
            active_mask[:, None], refined_params_phys, final_params
        )

        if do_merge:
            c, r, col, sigma = final_params.T
            active_mask = final_active & (c > 1e-9)
            num_active = jnp.sum(active_mask)

            c_active = jnp.where(active_mask, c, 0.0)  # [photons/Pixel^2]
            total_amp = jnp.sum(c_active) + 1e-12  # [photons/Pixel^2]

            com_r = jnp.sum(c_active * r) / total_amp  # [Pixel^0.5]
            com_c = jnp.sum(c_active * col) / total_amp  # [Pixel^0.5]
            var_r = jnp.sum(c_active * (r - com_r) ** 2) / total_amp  # [Pixel]
            var_c = jnp.sum(c_active * (col - com_c) ** 2) / total_amp  # [Pixel]

            mean_sigma = jnp.sum(jnp.where(active_mask, sigma, 0.0)) / jnp.maximum(
                num_active, 1
            )  # [Pixel^0.5]
            macro_sigma = jnp.sqrt(var_r + var_c) + mean_sigma  # [Pixel^0.5]

            dummy_atom = jnp.array([0.0, -100.0, -100.0, 1.0])
            macro_atom = jnp.stack([total_amp, com_r, com_c, macro_sigma])
            macro_atom = jnp.where(num_active > 1, macro_atom, dummy_atom)

            augmented_dict = jnp.vstack([final_params, macro_atom])
            aug_mask = jnp.append(active_mask, num_active > 1)

            A_aug = self._build_basis_matrix(x_grid, augmented_dict)
            A_aug_masked = A_aug * aug_mask

            c_sparse_stat_aug, _, _, _ = self.tune_and_solve(
                patch_stat.flatten(), patch_bg.flatten(), A_aug_masked, augmented_dict
            )

            # Re-extract the spatial columns for the return stack
            _, r_aug, col_aug, sigma_aug = augmented_dict.T

            return jnp.stack(
                [c_sparse_stat_aug * aug_mask, r_aug, col_aug, sigma_aug], axis=1
            )
        else:
            return final_params

    def compute_metrics(self, images_raw, bg_map, peaks_list, global_max):
        """
        Args:
            images_raw, bg_map: [photons/Pixel]
            global_max: [photons/Pixel]
        """
        B, H, W = images_raw.shape
        yy, xx = np.indices((H, W))  # [Pixel^0.5]
        x_grid = jnp.array([yy, xx])  # [Pixel^0.5]

        if self.show_steps:
            print("\n  [Metrics] Calculating goodness-of-fit...")

        max_k = max([len(p) for p in peaks_list] + [1])
        peaks_padded = np.zeros((B, max_k, 4), dtype=np.float32)
        counts_per_image = np.zeros(B, dtype=np.float32)

        for b in range(B):
            n = len(peaks_list[b])
            if n > 0:
                peaks_padded[b, :n, :] = peaks_list[b]
                if n < max_k:
                    peaks_padded[b, n:, 3] = 1.0
            counts_per_image[b] = n

        if self.loss == "poisson":
            loss_code = 1
        elif self.loss == "gaussian":
            loss_code = 0
        else:
            raise ValueError("Unsupported loss")

        @jit
        def process_one_image(peaks, target_raw, median_val, k_val):
            recon_peaks = self._predict_batch_scan(peaks, x_grid)  # [photons/Pixel]
            recon_total = jnp.maximum(recon_peaks + median_val, 1e-9)  # [photons/Pixel]

            if loss_code == 1:
                # 1. Exact Poisson NLL using xlogy
                nll = jnp.sum(
                    recon_total - jax.scipy.special.xlogy(target_raw, recon_total)
                )  # [photons/Pixel]
                # 2. Exact Poisson Deviance (no more 1e-9 target clamping)
                term = jax.scipy.special.xlogy(target_raw, target_raw / recon_total) - (
                    target_raw - recon_total
                )
                dev = 2 * jnp.sum(
                    term
                )  # [-] (Deviance acts as dimensionless chi-square equivalent)
            else:
                diff = recon_total - target_raw  # [photons/Pixel]
                nll = 0.5 * jnp.sum(diff**2)  # [photons^2 / Pixel^2]
                dev = jnp.sum((diff**2) / recon_total)  # [-]

            n_pix = target_raw.size
            n_params = k_val * 4
            bic = n_params * jnp.log(n_pix) + 2 * nll
            return nll, bic, dev

        nll_total, bic_total, deviance_total = 0.0, 0.0, 0.0

        for b in range(B):
            nll, bic, dev = process_one_image(
                jnp.array(peaks_padded[b]),
                jnp.array(images_raw[b]),
                jnp.array(bg_map[b]),
                jnp.array(counts_per_image[b]),
            )
            nll_total += float(nll)
            bic_total += float(bic)
            deviance_total += float(dev)

        pixels_total = B * H * W
        params_total = float(np.sum(counts_per_image)) * 4
        dof = max(pixels_total - params_total, 1)
        dev_per_dof = deviance_total / dof

        if self.show_steps:
            target_str = (
                "(Target ~ 1.0)" if loss_code == 1 else "(MSE/Variance of noise)"
            )
            print(f"  > Total NLL: {nll_total:.2e}")
            print(f"  > Total BIC: {bic_total:.2e}")
            print(f"  > Deviance/DoF: {dev_per_dof:.4f} {target_str}")

        return {"nll": nll_total, "bic": bic_total, "deviance_nu": dev_per_dof}

    def find_peaks_batch(self, images_batch):
        """
        Args:
            images_batch: [photons/Pixel]
        """
        B, H, W = images_batch.shape

        filter_size = max(15, int(self.max_sigma * 5))  # [Pixel^0.5]
        if filter_size % 2 == 0:
            filter_size += 1

        # --- CHUNKED BACKGROUND EVALUATION ---
        bg_map_list = []
        # Use a reasonable chunk size for the GPU (e.g., 64 or 128 frames at a time)
        bg_chunk_size = min(self.chunk_size, max(1, B // 4))

        bg_pbar = tqdm(
            range(0, B, bg_chunk_size),
            desc="Morphological Bg",
            disable=not self.show_steps,
        )

        for i in bg_pbar:
            chunk = jnp.array(images_batch[i : i + bg_chunk_size], dtype=jnp.float32)
            bg_chunk = compute_bg_batch(chunk, filter_size)
            bg_chunk.block_until_ready()  # Ensure async execution finishes
            bg_map_list.append(np.array(bg_chunk))

        bg_map = np.concatenate(bg_map_list, axis=0)  # [photons/Pixel]
        self._last_bg_map = bg_map

        valid_bg = bg_map[bg_map > 1e-2]
        if valid_bg.size == 0:
            median_bg_level = 1.0  # [photons/Pixel]
        else:
            median_bg_level = float(np.median(valid_bg))  # [photons/Pixel]

        if np.isnan(median_bg_level) or median_bg_level <= 0:
            median_bg_level = 1.0

        poisson_noise_floor = np.maximum(
            np.sqrt(median_bg_level), 1.0
        )  # [photons^0.5 / Pixel^0.5]

        if self.show_steps:
            print("  > Pre-processing: Morphological Bg Evaluated.")
            print(
                f"  > Autotuning: Median BG={median_bg_level:.1f}, Noise Floor=~{poisson_noise_floor:.1f}"
            )

        img_jax_stat_np = np.copy(images_batch)  # [photons/Pixel]
        if self.border_width > 0:
            bw = self.border_width  # [Pixel^0.5]
            valid_interior = np.zeros((H, W), dtype=bool)
            valid_interior[bw:-bw, bw:-bw] = True
            valid_mask_batch = np.broadcast_to(valid_interior, (B, H, W))
            img_jax_stat_np = np.where(valid_mask_batch, img_jax_stat_np, bg_map)

        img_jax_stat = jnp.array(img_jax_stat_np)  # [photons/Pixel]
        img_jax_bg = jnp.array(bg_map)  # [photons/Pixel]

        loss_code_sniper = 1 if self.loss == "poisson" else 0

        max_k_rad = int(3.0 * self.max_sigma)  # [Pixel^0.5]

        # Decide which alpha to use for the SCOUT phase
        # If tuning, use the absolute minimum candidate to cast the widest possible net
        # alpha is strictly interpreted as the Z-score (SNR) threshold
        scout_alpha = (
            jnp.min(self.candidate_alphas) if self.auto_tune_alpha else self.alpha
        )

        w_scout_core = self.base_window_size  # [Pixel^0.5]
        w_ext = w_scout_core + 2 * max_k_rad  # [Pixel^0.5]
        stride = w_scout_core // 2  # [Pixel^0.5]

        min_required_patch = 2 * max_k_rad + 1  # [Pixel^0.5]
        P_core = max(self.refine_patch_size, min_required_patch)  # [Pixel^0.5]
        P_EXT = P_core + 2 * max_k_rad  # [Pixel^0.5]

        pad_size = P_core // 2 + max_k_rad  # [Pixel^0.5]

        img_jax_stat = jnp.pad(
            img_jax_stat,
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode="symmetric",
        )  # [photons/Pixel]
        img_jax_bg = jnp.pad(
            img_jax_bg,
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode="symmetric",
        )  # [photons/Pixel]

        start_h, end_h = pad_size, pad_size + H
        start_w, end_w = pad_size, pad_size + W

        grid_h = list(range(start_h, end_h - w_scout_core + 1, stride))
        if not grid_h or grid_h[-1] + w_scout_core < end_h:
            grid_h.append(max(start_h, end_h - w_scout_core))

        grid_w = list(range(start_w, end_w - w_scout_core + 1, stride))
        if not grid_w or grid_w[-1] + w_scout_core < end_w:
            grid_w.append(max(start_w, end_w - w_scout_core))

        window_coords = [(b, r, c) for b in range(B) for r in grid_h for c in grid_w]
        window_coords_arr = np.array(window_coords, dtype=np.int32)
        total_scout_wins = len(window_coords)

        @jit
        def extract_scout_window(img, b_idx, r_idx, c_idx):
            r_start = r_idx - max_k_rad  # [Pixel^0.5]
            c_start = c_idx - max_k_rad  # [Pixel^0.5]

            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(
                    img[bi], (ri, ci), (w_ext, w_ext)
                )  # [photons/Pixel]

            return vmap(slice_one)(b_idx, r_start, c_start)

        scout_solver = jit(
            vmap(
                lambda ws, wb: self._solve_dense(
                    ws, wb, scout_alpha, w_ext, w_ext, 5, loss_code_sniper, False
                )
            )
        )

        scout_results = []
        scout_pbar = tqdm(
            range(0, total_scout_wins, self.chunk_size),
            desc="Scout Phase",
            disable=not self.show_steps,
        )

        for i in scout_pbar:
            chunk = window_coords_arr[i : i + self.chunk_size]
            wins_stat = extract_scout_window(
                img_jax_stat, chunk[:, 0], chunk[:, 1], chunk[:, 2]
            )
            wins_bg = extract_scout_window(
                img_jax_bg, chunk[:, 0], chunk[:, 1], chunk[:, 2]
            )
            res = scout_solver(wins_stat, wins_bg)
            res.block_until_ready()

            global_res = np.array(res)

            valid_mask = global_res[:, :, 0] > 1e-9
            b_indices, peak_indices = np.where(valid_mask)
            if len(b_indices) > 0:
                valid_peaks = global_res[b_indices, peak_indices]
                valid_banks = chunk[b_indices, 0]

                valid_peaks[:, 1] += chunk[b_indices, 1] - max_k_rad  # [Pixel^0.5]
                valid_peaks[:, 2] += chunk[b_indices, 2] - max_k_rad  # [Pixel^0.5]

                peaks_with_bank = np.column_stack([valid_banks, valid_peaks])
                scout_results.append(peaks_with_bank)

        if not scout_results:
            return [np.empty((0, 4)) for _ in range(B)]

        all_candidates = np.vstack(scout_results)
        unique_candidates = []

        for b in range(B):
            bank_mask = all_candidates[:, 0] == b
            if not np.any(bank_mask):
                continue
            cands = all_candidates[bank_mask, 2:4]  # [Pixel^0.5]
            vals = all_candidates[bank_mask, 1]  # [photons/Pixel^2]
            order = np.argsort(vals)[::-1]
            cands_sorted = cands[order]
            keep = np.ones(len(cands_sorted), dtype=bool)
            if len(cands_sorted) > 1:
                dists = squareform(pdist(cands_sorted))  # [Pixel^0.5]
                np.fill_diagonal(dists, 9999.0)
                radius = 1.5  # [Pixel^0.5]
                for i in range(len(cands_sorted)):
                    if keep[i]:
                        neighbors = np.where(dists[i] < radius)[0]
                        neighbors = neighbors[neighbors > i]
                        keep[neighbors] = False
            valid_seeds = cands_sorted[keep]

            if len(valid_seeds) > 0:
                bank_col = np.full((len(valid_seeds), 1), b)
                unique_candidates.append(np.hstack([bank_col, valid_seeds]))

        if not unique_candidates:
            return [np.empty((0, 4)) for _ in range(B)]

        all_seeds = np.vstack(unique_candidates)
        total_seeds = len(all_seeds)

        @jit
        def extract_patch_with_halo(img, centers):
            b_idx = centers[:, 0].astype(int)
            r_center = centers[:, 1].astype(int)  # [Pixel^0.5]
            c_center = centers[:, 2].astype(int)  # [Pixel^0.5]

            r_start = r_center - pad_size  # [Pixel^0.5]
            c_start = c_center - pad_size  # [Pixel^0.5]

            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (P_EXT, P_EXT))

            return vmap(slice_one)(b_idx, r_start, c_start)

        # Use standard solver with fixed alpha
        sniper_solver = jit(
            vmap(
                lambda ws, wb: self._solve_dense(
                    ws,
                    wb,
                    self.alpha,
                    P_EXT,
                    P_EXT,
                    self.max_local_peaks,
                    loss_code_sniper,
                    do_merge=True,
                )
            )
        )

        refined_peaks_by_bank = [[] for _ in range(B)]

        sniper_pbar = tqdm(
            range(0, total_seeds, self.chunk_size),
            desc="Sniper V-Cycle",
            disable=not self.show_steps,
        )

        for i in sniper_pbar:
            chunk = all_seeds[i : i + self.chunk_size]

            patches_stat = extract_patch_with_halo(img_jax_stat, jnp.array(chunk))
            patches_bg = extract_patch_with_halo(img_jax_bg, jnp.array(chunk))

            res = sniper_solver(patches_stat, patches_bg)
            res.block_until_ready()
            res_cpu = np.array(res)

            valid_mask = res_cpu[:, :, 0] > 1e-9
            b_indices, peak_indices = np.where(valid_mask)

            if len(b_indices) > 0:
                valid_peaks = res_cpu[b_indices, peak_indices]
                valid_b_ids = chunk[b_indices, 0]
                valid_r_centers = chunk[b_indices, 1]
                valid_c_centers = chunk[b_indices, 2]

                global_rs_padded = (
                    valid_r_centers.astype(int) - pad_size + valid_peaks[:, 1]
                )  # [Pixel^0.5]
                global_cs_padded = (
                    valid_c_centers.astype(int) - pad_size + valid_peaks[:, 2]
                )  # [Pixel^0.5]

                global_rs = global_rs_padded - pad_size  # [Pixel^0.5]
                global_cs = global_cs_padded - pad_size  # [Pixel^0.5]

                MARGIN = max(3, self.border_width)  # [Pixel^0.5]
                in_bounds = (
                    (global_rs >= MARGIN)
                    & (global_rs < H - MARGIN)
                    & (global_cs >= MARGIN)
                    & (global_cs < W - MARGIN)
                )

                final_mask = (valid_peaks[:, 0] > 1e-5) & in_bounds

                for k in range(len(final_mask)):
                    if final_mask[k]:
                        b_id = int(valid_b_ids[k])
                        refined_peaks_by_bank[b_id].append(
                            np.array(
                                [
                                    valid_peaks[k, 0],
                                    global_rs[k],
                                    global_cs[k],
                                    valid_peaks[k, 3],
                                ]
                            )
                        )

        final_coords_output = []
        final_peaks_full = []

        for b in range(B):
            peaks = np.array(refined_peaks_by_bank[b])
            if len(peaks) > 0:
                order = np.argsort(peaks[:, 0])[::-1]
                peaks_sorted = peaks[order]
                keep = np.ones(len(peaks_sorted), dtype=bool)
                coords = peaks_sorted[:, 1:3]  # [Pixel^0.5]

                if len(coords) > 1:
                    dists = squareform(pdist(coords))  # [Pixel^0.5]
                    np.fill_diagonal(dists, 9999.0)
                    r = 1.5  # [Pixel^0.5]
                    for i in range(len(coords)):
                        if keep[i]:
                            neighbors = np.where(dists[i] < r)[0]
                            neighbors = neighbors[neighbors > i]
                            keep[neighbors] = False

                unique_peaks = peaks_sorted[keep]
                final_peaks_full.append(unique_peaks)
                final_coords_output.append(unique_peaks)
            else:
                final_peaks_full.append(np.empty((0, 4)))
                final_coords_output.append(np.empty((0, 4)))

        self.compute_metrics(images_batch, bg_map, final_peaks_full, 1.0)

        return final_coords_output


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


@partial(jit, static_argnames=["patch_size", "fit_mosaicity"])
def global_shape_objective(
    params, patches, bgs, drs, dcs, P_mats, distances, R_mats, patch_size, fit_mosaicity
):
    # 1. Build the crystal shape tensor
    Sigma_shape_sample = build_3d_cov(params[:6])

    # 2. Handle the optional Mosaicity Tensor
    if fit_mosaicity:
        eta = jnp.abs(params[6]) + 1e-6
        Sigma_eta_base = jnp.eye(3) * (eta**2)
    else:
        # If disabled, the tensor is perfectly zeroed out.
        Sigma_eta_base = jnp.zeros((3, 3))

    def fit_one_peak(patch, bg, dr, dc, P_true, D_i, R_gonio):
        # 3. Rotate Crystal Shape to the Lab Frame
        Sigma_shape_lab = R_gonio @ Sigma_shape_sample @ R_gonio.T

        # 4. Add the tensors. If fit_mosaicity is False, D_i * 0 = 0.
        Sigma_total_3D = Sigma_shape_lab + (D_i**2) * Sigma_eta_base

        # 5. Exact 2D Projection and Pixel Conversion
        Sigma_2D_physical = P_true @ Sigma_total_3D @ P_true.T
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
        return jnp.sum(residual**2)

    mses = vmap(fit_one_peak)(patches, bgs, drs, dcs, P_mats, distances, R_mats)
    return jnp.mean(mses)


# Bind the val_and_grad wrapper to recognize the new static argument
val_and_grad_fn = jit(
    jax.value_and_grad(global_shape_objective),
    static_argnames=["patch_size", "fit_mosaicity"],
)


def optimize_global_crystal(
    patches, bgs, drs, dcs, P_mats, distances, R_mats, fit_mosaicity=False
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
            patches.shape[-1],
            fit_mosaicity=fit_mosaicity,
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

    # 4. Mosaicity bound (if active, e.g., max 10 mrad = 0.010 rad)
    if fit_mosaicity:
        bounds[6] = (1e-6, 0.010 / scales[6])

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


class SparseLaueIntegrator(SparseRBFPeakFinder):
    """
    Physics-Informed Sniper.
    Takes predicted spot coordinates, extracts patches, and uses Volume-Penalized
    Sparse RBF to accurately integrate intensity using the Preconditioned SSN Engine.

    Units:
        alpha: [-] (Z-score threshold)
        min_sigma / max_sigma / nominal_sigma: [-]
        mosaicity_eta: [Pixel^0.5] (?)
        gamma: [-]
        Returns integrations containing sigI: [photons^0.5 / Pixel^0.5]
    """

    def __init__(
        self,
        alpha=0.05,
        min_sigma=0.1,  # [Pixels] at theta=45
        max_sigma=15.0,  # [Pixels] at theta=45
        gamma=2.0,
        loss="poisson",
        border_width=0,
        num_sigmas=32,
        nominal_sigma=2.0,
        anisotropic=False,
        chunk_size=1024,
        show_steps=False,
    ):
        # 1. Initialize parent with safe dummy pixel values to keep it functional
        # (in case you ever call super().find_peaks_batch)
        super().__init__(
            alpha=alpha,
            gamma=gamma,
            min_sigma=0.5,
            max_sigma=5.0,
            loss=loss,
            border_width=border_width,
            num_sigmas=num_sigmas,
            chunk_size=chunk_size,
            show_steps=show_steps,
        )

        self.nominal_sigma = nominal_sigma
        self.anisotropic = anisotropic

    def integrate_reflections(
        self, images_batch, frames, rs, cs, var_us=None, var_vs=None, cov_uvs=None
    ):
        """
        Args:
            images_batch: [photons/Pixel]
            frames: [-]
            rs, cs: [Pixel^0.5]
            var_us, var_vs, cov_uvs: Pre-computed 2D projection tensors from the global optimizer.
        Returns:
            [intensity: [photons/Pixel], r: [Pixel^0.5], c: [Pixel^0.5], sigma: [Pixel^0.5], sigI: [photons^0.5 / Pixel^0.5]]
        """
        B, H, W = images_batch.shape
        N_spots = len(frames)

        # api backward compatibility for unit tests
        if var_us is None or var_vs is None or cov_uvs is None:
            # Fall back to a perfect isotropic circle using the nominal_sigma property
            var_us = jnp.full(N_spots, self.nominal_sigma**2, dtype=jnp.float32)
            var_vs = jnp.full(N_spots, self.nominal_sigma**2, dtype=jnp.float32)
            cov_uvs = jnp.zeros(N_spots, dtype=jnp.float32)
        else:
            # Ensure they are JAX arrays for the JIT compiler
            var_us = jnp.array(var_us, dtype=jnp.float32)
            var_vs = jnp.array(var_vs, dtype=jnp.float32)
            cov_uvs = jnp.array(cov_uvs, dtype=jnp.float32)

        P = self.refine_patch_size  # [Pixel^0.5]
        half_p = P // 2  # [Pixel^0.5]
        PAD = P  # [Pixel^0.5]

        K_NEIGHBORS = min(4, N_spots) if N_spots > 0 else 1

        filter_size = max(15, int(self.max_sigma * 5))  # [Pixel^0.5]
        if filter_size % 2 == 0:
            filter_size += 1

        # --- CHUNKED BACKGROUND EVALUATION ---
        bg_map_list = []
        bg_chunk_size = min(self.chunk_size, max(1, B // 4))

        # In integrate_reflections, we use self.show_steps to match the parent
        bg_pbar = tqdm(
            range(0, B, bg_chunk_size),
            desc="Integration Bg",
            disable=not self.show_steps,
        )

        for i in bg_pbar:
            chunk = jnp.array(images_batch[i : i + bg_chunk_size], dtype=jnp.float32)
            bg_chunk = compute_bg_batch(chunk, filter_size)
            bg_chunk.block_until_ready()
            bg_map_list.append(
                bg_chunk
            )  # Keep as JAX array to avoid host transfer if possible

        bg_maps_jax = jnp.concatenate(bg_map_list, axis=0)  # [photons/Pixel]
        images_jax = jnp.array(images_batch, dtype=jnp.float32)  # [photons/Pixel]

        img_jax_padded = jnp.pad(
            images_jax, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect"
        )  # [photons/Pixel]
        bg_jax_padded = jnp.pad(
            bg_maps_jax, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect"
        )  # [photons/Pixel]

        (float(P), float(P), self.min_sigma, self.max_sigma)  # [Pixel^0.5]
        yy, xx = jnp.indices((P, P))  # [Pixel^0.5]
        x_grid = jnp.array([yy, xx])  # [Pixel^0.5]

        @jit
        def extract_patches(img_src, bg_src, f_idx, r_idx, c_idx):
            r_start = jnp.clip(
                jnp.int32(jnp.round(r_idx)) - half_p, 0, img_src.shape[1] - P
            )  # [Pixel^0.5]
            c_start = jnp.clip(
                jnp.int32(jnp.round(c_idx)) - half_p, 0, img_src.shape[2] - P
            )  # [Pixel^0.5]

            def slice_img(bi, ri, ci):
                return lax.dynamic_slice(img_src[bi], (ri, ci), (P, P))

            def slice_bg(bi, ri, ci):
                return lax.dynamic_slice(bg_src[bi], (ri, ci), (P, P))

            return (
                vmap(slice_img)(f_idx, r_start, c_start),
                vmap(slice_bg)(f_idx, r_start, c_start),
                r_start,
                c_start,
            )

        @jit
        def solve_patches(
            patches,
            patches_bg,
            fs_chunk,
            rs_global_chunk,
            cs_global_chunk,
            r_starts,
            c_starts,
            all_fs_jnp,
            all_rs_jnp,
            all_cs_jnp,
            var_us_jnp,
            var_vs_jnp,
            cov_uvs_jnp,
        ):
            alpha_z_score = self.alpha

            def process_patch(
                patch, patch_bg, f_global, r_global, c_global, r_start, c_start
            ):
                # 1. Fetch exact pre-computed global geometry
                peak_var_u = var_us_jnp[f_global]
                peak_var_v = var_vs_jnp[f_global]
                peak_cov_uv = cov_uvs_jnp[f_global]

                bg_med = jnp.maximum(jnp.median(patch_bg), 1e-3)
                jnp.sqrt(bg_med)

                dists = (all_rs_jnp - r_global) ** 2 + (all_cs_jnp - c_global) ** 2
                frame_penalty = jnp.where(all_fs_jnp == f_global, 0.0, 1e9)
                _, nbr_idxs = jax.lax.top_k(-(dists + frame_penalty), K_NEIGHBORS)

                nbr_rs = all_rs_jnp[nbr_idxs]
                nbr_cs = all_cs_jnp[nbr_idxs]
                local_rs = nbr_rs - r_start
                local_cs = nbr_cs - c_start

                # SUBPIXEL RELAXATION (Log-Parabolic Target Snapping)
                # We use a static 1.0px blur just to smooth the noise for the center-of-mass finding
                y_sub_raw = patch - patch_bg
                nominal_sig_sq2 = 1.0 * jnp.sqrt(2.0) + 1e-6
                k_grid = jnp.arange(-2, 3)
                k_1d = jax.scipy.special.erf(
                    (k_grid + 0.5) / nominal_sig_sq2
                ) - jax.scipy.special.erf((k_grid - 0.5) / nominal_sig_sq2)

                temp = jax.scipy.signal.correlate2d(
                    y_sub_raw, k_1d[:, None], mode="same"
                )
                dual_var_smooth = jax.scipy.signal.correlate2d(
                    temp, k_1d[None, :], mode="same"
                )

                r_int = jnp.clip(jnp.int32(jnp.round(local_rs[0])), 1, P - 2)
                c_int = jnp.clip(jnp.int32(jnp.round(local_cs[0])), 1, P - 2)

                safe_dv = jnp.maximum(dual_var_smooth, 1e-6)
                val = jnp.log(safe_dv[r_int, c_int])
                val_up = jnp.log(safe_dv[r_int - 1, c_int])
                val_dn = jnp.log(safe_dv[r_int + 1, c_int])
                val_lf = jnp.log(safe_dv[r_int, c_int - 1])
                val_rt = jnp.log(safe_dv[r_int, c_int + 1])

                den_r = jnp.minimum(val_up - 2.0 * val + val_dn, -1e-6)
                dr = 0.5 * (val_up - val_dn) / den_r

                den_c = jnp.minimum(val_lf - 2.0 * val + val_rt, -1e-6)
                dc = 0.5 * (val_lf - val_rt) / den_c

                dr = jnp.clip(dr, -1.5, 1.5)
                dc = jnp.clip(dc, -1.5, 1.5)

                local_rs = local_rs.at[0].add(dr)
                local_cs = local_cs.at[0].add(dc)

                # =====================================================================
                # ANALYTIC GAUSSIAN EVALUATION
                # =====================================================================
                def eval_neighbor(nr, nc):
                    det_sigma = jnp.maximum(
                        peak_var_u * peak_var_v - peak_cov_uv**2, 1e-6
                    )

                    a = peak_var_v / det_sigma
                    b = -peak_cov_uv / det_sigma
                    c = peak_var_u / det_sigma

                    dr_grid = x_grid[0] - nr
                    dc_grid = x_grid[1] - nc

                    gaussian = jnp.exp(
                        -0.5
                        * (
                            a * dc_grid**2
                            + 2.0 * b * dr_grid * dc_grid
                            + c * dr_grid**2
                        )
                    )

                    # 1. The analytic volume of this specific 2D footprint
                    area_scalar = 2.0 * jnp.pi * jnp.sqrt(det_sigma)

                    # 2. Divide by the volume so the basis sums exactly to 1.0.
                    # This guarantees the solver parameter 'c' perfectly equals TOTAL PHOTON FLUX.
                    return (gaussian / area_scalar).flatten()

                A_all = vmap(eval_neighbor)(local_rs, local_cs)
                y_sub = (patch - patch_bg).flatten()

                pixel_dists_k = (yy.flatten()[:, None] - local_rs[None, :]) ** 2 + (
                    xx.flatten()[:, None] - local_cs[None, :]
                ) ** 2
                closest_k = jnp.argmin(pixel_dists_k, axis=1)
                pixel_masks = jax.nn.one_hot(closest_k, K_NEIGHBORS)

                A_k = A_all.T
                A_k_masked = A_k * pixel_masks

                # We use the scalar effective sigma solely for weighting the L1 Z-score threshold
                effective_sigma = jnp.sqrt(
                    jnp.sqrt(
                        jnp.maximum(peak_var_u * peak_var_v - peak_cov_uv**2, 1e-6)
                    )
                )

                c_warm_joint = jnp.zeros(K_NEIGHBORS, dtype=jnp.float32)
                cand_params = jnp.stack(
                    [
                        c_warm_joint,
                        local_rs,
                        local_cs,
                        jnp.full(K_NEIGHBORS, effective_sigma),
                    ],
                    axis=1,
                )

                c_ssn = self.solve_ssn_step(
                    patch.flatten(),
                    patch_bg.flatten(),
                    A_k_masked,
                    cand_params,
                    alpha_override=alpha_z_score,  # Override to bypass tuning during final integration
                )

                surviving_mask_strict = c_ssn > 1e-9
                is_target = jnp.arange(K_NEIGHBORS) == 0
                surviving_mask = surviving_mask_strict | is_target

                A_best_masked = A_k * surviving_mask[None, :]

                # =====================================================================
                # STAGE 2: UNCONSTRAINED OLS
                # =====================================================================
                A_tilde = jnp.hstack([A_best_masked, jnp.ones((P * P, 1))])
                w = 1.0 / jnp.maximum(patch.flatten(), 1.0)

                I_mat = A_tilde.T @ (w[:, None] * A_tilde)
                C_mat = jnp.linalg.inv(I_mat + 1e-6 * jnp.eye(K_NEIGHBORS + 1))
                rhs = A_tilde.T @ (w * y_sub)
                c_ols = C_mat @ rhs

                # Because the basis is normalized, c_ols is literally the total unpenalized photon count!
                intensity = c_ols[0]

                # The variance of the flux parameter from the Fisher Information diagonal
                var_c0 = C_mat[0, 0]
                sigI = jnp.sqrt(jnp.maximum(var_c0, 0.0))

                return jnp.array(
                    [intensity, local_rs[0], local_cs[0], effective_sigma, sigI]
                )

            return vmap(process_patch)(
                patches,
                patches_bg,
                fs_chunk,
                rs_global_chunk,
                cs_global_chunk,
                r_starts,
                c_starts,
            )

        refined_peaks = []
        rs_padded = np.array(rs) + PAD  # [Pixel^0.5]
        cs_padded = np.array(cs) + PAD  # [Pixel^0.5]

        PAD_N = max(N_spots, 4)
        fs_full = np.pad(np.array(frames), (0, PAD_N - N_spots), constant_values=-1)
        rs_full = np.pad(
            rs_padded, (0, PAD_N - N_spots), constant_values=-10000.0
        )  # [Pixel^0.5]
        cs_full = np.pad(
            cs_padded, (0, PAD_N - N_spots), constant_values=-10000.0
        )  # [Pixel^0.5]

        all_fs_jnp = jnp.array(fs_full, dtype=jnp.int32)
        all_rs_jnp = jnp.array(rs_full, dtype=jnp.float32)  # [Pixel^0.5]
        all_cs_jnp = jnp.array(cs_full, dtype=jnp.float32)  # [Pixel^0.5]

        with tqdm(
            total=N_spots, desc="Sparse Laue Integration", disable=not self.show_steps
        ) as pbar:
            for i in range(0, N_spots, self.chunk_size):
                chunk_f = jnp.array(frames[i : i + self.chunk_size])
                chunk_r = jnp.array(rs_padded[i : i + self.chunk_size])  # [Pixel^0.5]
                chunk_c = jnp.array(cs_padded[i : i + self.chunk_size])  # [Pixel^0.5]

                patches, patches_bg, r_starts, c_starts = extract_patches(
                    img_jax_padded, bg_jax_padded, chunk_f, chunk_r, chunk_c
                )

                res = solve_patches(
                    patches,
                    patches_bg,
                    chunk_f,
                    chunk_r,
                    chunk_c,
                    r_starts,
                    c_starts,
                    all_fs_jnp,
                    all_rs_jnp,
                    all_cs_jnp,
                    var_us,
                    var_vs,
                    cov_uvs,
                )
                res.block_until_ready()

                res_cpu = np.array(res)
                res_cpu[:, 1] = res_cpu[:, 1] + r_starts - PAD  # [Pixel^0.5]
                res_cpu[:, 2] = res_cpu[:, 2] + c_starts - PAD  # [Pixel^0.5]

                refined_peaks.append(res_cpu)
                pbar.update(len(chunk_f))

        if len(refined_peaks) == 0:
            return np.empty((0, 4))

        return np.vstack(refined_peaks)


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
    alpha: float,
    gamma: float,
    show_progress: bool,
    all_R: np.ndarray = None,
    sample_offset: np.ndarray = None,
    ki_vec: np.ndarray = None,
    nominal_sigma: float = 2.0,
    anisotropic: bool = False,
    fit_mosaicity: bool = False,
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

    integrator = SparseLaueIntegrator(
        alpha=alpha,
        min_sigma=min(sigmas),
        max_sigma=max(sigmas),
        gamma=gamma,
        loss="poisson",
        border_width=border_width,
        nominal_sigma=nominal_sigma,
        anisotropic=anisotropic,
        chunk_size=chunk_size,
        show_steps=show_progress,
    )
    # Ensure the solver dictionary perfectly matches the provided list
    integrator.candidate_sigmas = jnp.array(sigmas, dtype=jnp.float32)
    integrator.show_steps = show_progress

    # --- PHASE 1: GATHER AND BATCH ---
    images_list = []
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

    # ==========================================
    # 2. Extract exact patches for global optimization
    # ==========================================
    opt_P = 15
    opt_half = opt_P // 2
    opt_patches, opt_bgs, opt_drs, opt_dcs = [], [], [], []
    opt_Pmats, opt_dists, opt_Rmats = [], [], []

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

        all_var_u = np.full(len(all_rs), integrator.nominal_sigma**2, dtype=np.float32)
        all_var_v = np.full(len(all_rs), integrator.nominal_sigma**2, dtype=np.float32)
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
            fit_mosaicity=fit_mosaicity,
        )

        # 5. Project the EXACT 2D footprints for ALL peaks
        Sigma_shape_jnp = build_3d_cov(jnp.array(res_x[:6]))

        if fit_mosaicity:
            Sigma_eta_jnp = jnp.eye(3) * (abs(res_x[6]) ** 2 + 1e-12)
        else:
            Sigma_eta_jnp = jnp.zeros((3, 3))

        @jit
        def project_all_shapes(P_mats, dists, R_mats):
            def project_one(P, D_i, R_gonio):
                # Rotate to Lab Frame before projecting
                Sigma_shape_lab = R_gonio @ Sigma_shape_jnp @ R_gonio.T
                Sigma_total = Sigma_shape_lab + (D_i**2) * Sigma_eta_jnp
                return P @ Sigma_total @ P.T

            return vmap(project_one)(P_mats, dists, R_mats)

        all_Sigma_2D = project_all_shapes(
            jnp.array(all_P_mats), jnp.array(all_distances), jnp.array(all_R_mats)
        )

        all_var_u = np.array(all_Sigma_2D[:, 0, 0])
        all_var_v = np.array(all_Sigma_2D[:, 1, 1])
        all_cov_uv = np.array(all_Sigma_2D[:, 0, 1])

    # --- PHASE 2: GPU INTEGRATION ---
    integrated_results = integrator.integrate_reflections(
        images_batch,
        frames,
        all_rs,
        all_cs,
        var_us=all_var_u,
        var_vs=all_var_v,
        cov_uvs=all_cov_uv,
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
