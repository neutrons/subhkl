"""The parametric peak model shared by peak finding, refinement and integration.

Every stage of the pipeline works with the same object: a 2D Gaussian integrated
over each pixel, written with error functions so a peak that falls between pixel
centres is represented exactly rather than sampled.  Until now each stage
carried its own copy of that algebra -- ``SparseRBFPeakFinder`` has one for its
greedy search, ``MatrixFreeSparseRBFPeakFinder`` a second for its metrics and a
third for the sliding refinement -- and the integrator reached the finder's copy
only by subclassing it.

Keeping one definition here matters beyond tidiness.  Detection and integration
differ in what they hold fixed, not in the model: detection solves for positions
with the profile constrained, while integration knows the positions from the
lattice and solves for an anisotropic profile.  Sharing the forward model is what
lets those be the same estimator viewed two ways, so the profile the integrator
learns is the profile the finder assumed.

Conventions, which the copies agreed on and which are preserved exactly:

* ``amp`` is the *peak amplitude*, not the flux.  A kernel of width ``sigma``
  has unit peak amplitude and total flux ``2 * pi * sigma**2``, so an atom of
  amplitude ``a`` carries flux ``a * 2 * pi * sigma**2``.
* Pixel ``(r, c)`` covers ``[r - 1/2, r + 1/2] x [c - 1/2, c + 1/2]``.
"""

import jax
import jax.numpy as jnp


def pixel_integrated_gaussian_1d(grid, centre, sigma):
    """Integral of a unit-height Gaussian over each pixel along one axis.

    ``grid`` and ``centre`` broadcast against each other, so this serves both the
    dense-image and the per-peak-patch layouts.
    """
    s2 = sigma * jnp.sqrt(2.0) + 1e-6
    d = grid - centre
    return jax.scipy.special.erf((d + 0.5) / s2) - jax.scipy.special.erf(
        (d - 0.5) / s2
    )


def render_peak(y_grid, x_grid, amp, r, c, sigma):
    """One peak on the given coordinate grids, in photons/pixel."""
    ey = pixel_integrated_gaussian_1d(y_grid, r, sigma)
    ex = pixel_integrated_gaussian_1d(x_grid, c, sigma)
    return amp * (jnp.pi / 2.0) * (sigma**2) * ey * ex


def kernel_bank(sigmas, max_k_rad):
    """Dictionary of pixel-integrated kernels, one per scale.

    Returned in ``OIHW`` layout so it can be handed straight to
    ``lax.conv_general_dilated`` as a convolution weight, along with each
    kernel's squared L2 norm, which is what sets the curvature of the
    corresponding coefficient.
    """
    k_grid = jnp.arange(-max_k_rad, max_k_rad + 1)
    yy, xx = jnp.meshgrid(k_grid, k_grid, indexing="ij")

    def build_one(s):
        return render_peak(yy, xx, 1.0, 0.0, 0.0, s)

    kernels = jax.vmap(build_one)(sigmas)
    return kernels[:, None, :, :], jnp.sum(kernels**2, axis=(1, 2))


def render_patches(shape, amps, rows, cols, sigmas, max_k_rad, active=None):
    """Render peaks onto their own bounding boxes and scatter into one image.

    Cost is ``O(n_peaks * patch**2)`` rather than ``O(n_peaks * H * W)``, which is
    what keeps a joint fit over every peak affordable on a full detector image.
    Gradients flow through the continuous ``rows``/``cols``/``sigmas``; only the
    integer anchor of each box is non-differentiable, and that is piecewise
    constant so it does not affect the gradient away from box boundaries.
    """
    H, W = shape
    P = 2 * max_k_rad + 1
    span = jnp.arange(P)
    mask = jnp.ones_like(amps) if active is None else active.astype(amps.dtype)

    r0 = jnp.clip(jnp.round(rows).astype(jnp.int32) - max_k_rad, 0, H - P)
    c0 = jnp.clip(jnp.round(cols).astype(jnp.int32) - max_k_rad, 0, W - P)

    rr = r0[:, None].astype(amps.dtype) + span[None, :].astype(amps.dtype)
    cc = c0[:, None].astype(amps.dtype) + span[None, :].astype(amps.dtype)

    ey = pixel_integrated_gaussian_1d(rr, rows[:, None], sigmas[:, None])
    ex = pixel_integrated_gaussian_1d(cc, cols[:, None], sigmas[:, None])

    scale = amps * (jnp.pi / 2.0) * (sigmas**2) * mask
    patch = scale[:, None, None] * ey[:, :, None] * ex[:, None, :]

    n = amps.shape[0]
    idx_r = jnp.broadcast_to((r0[:, None] + span)[:, :, None], (n, P, P))
    idx_c = jnp.broadcast_to((c0[:, None] + span)[:, None, :], (n, P, P))
    return jnp.zeros((H, W), dtype=patch.dtype).at[idx_r, idx_c].add(patch)


def poisson_nll(model, data, floor=1e-6):
    """Poisson negative log-likelihood, dropping the data-only log(y!) term.

    Convex in ``model``, and the ``-y log u`` barrier keeps the fit strictly
    positive, which is what makes the nonnegative-measure results applicable and
    what gives the reweighting ``W = 1/u`` its Fisher-information meaning.
    """
    u = jnp.maximum(model, floor)
    return jnp.sum(u - data * jnp.log(u))


def gaussian_nll(model, data):
    """Least squares, for the homoscedastic case."""
    return 0.5 * jnp.sum((model - data) ** 2)


def to_unconstrained(amps, sigmas, min_sigma, max_sigma):
    """Map amplitude and width to an unbounded space for free optimisation.

    Amplitude goes through a log so it stays positive; width through a logit onto
    ``[min_sigma, max_sigma]`` so it stays inside the dictionary's range.  This
    removes the need to project after every step.
    """
    lo, hi = min_sigma, max_sigma
    s = jnp.clip(sigmas, lo + 1e-3, hi - 1e-3)
    return (
        jnp.log(jnp.maximum(amps, 1e-3)),
        jnp.log((s - lo) / jnp.maximum(hi - s, 1e-6)),
    )


def to_physical(log_amp, logit_sigma, min_sigma, max_sigma):
    """Inverse of :func:`to_unconstrained`."""
    lo, hi = min_sigma, max_sigma
    return jnp.exp(log_amp), lo + (hi - lo) * jax.nn.sigmoid(logit_sigma)


def flux(amps, sigmas):
    """Total flux of each atom: amplitude times the kernel's area."""
    return amps * 2.0 * jnp.pi * (sigmas**2)


__all__ = [
    "flux",
    "gaussian_nll",
    "kernel_bank",
    "pixel_integrated_gaussian_1d",
    "poisson_nll",
    "render_patches",
    "render_peak",
    "to_physical",
    "to_unconstrained",
]
