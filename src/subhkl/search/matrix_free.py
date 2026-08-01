from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.signal
import jax.scipy.sparse.linalg
import numpy as np
from jax import jit, lax, vmap


class MatrixFreeSparseRBFPeakFinder:
    """
    Matrix-Free Global L1 Peak Finder.
    Replaces greedy Matching Pursuit with Global Convolutional Basis Pursuit.

    A note on ``gamma``, because the default sits on a degenerate value.

    The penalty on an atom of scale sigma works out proportional to
    sigma**(gamma + 1), while that atom's flux is proportional to sigma**2, so
    the penalty *per unit flux* goes as sigma**(gamma - 1).  At ``gamma == 1``
    that is constant -- measured constant to 1% across a 10x range of sigma --
    which is the Radon-measure / total-variation point: the penalty charges an
    atom for its flux and is blind to the scale carrying it.

    That is a genuine degeneracy rather than a mere preference, because the
    Gaussian scale space is closed under mass-preserving superposition
    (G_sigma = G_sigma' * G_sigma'' with sigma^2 = sigma'^2 + sigma''^2).  One
    broad atom and a spread of narrower atoms of equal total flux therefore
    predict the *same* image at the *same* penalty, so the minimiser is not
    unique in the scale coordinate, and since extra atoms always absorb a
    little more noise, the fit breaks the tie towards splitting.  In practice
    this shows up as one broad peak being reported as a cluster of narrower
    ones, plus spurious atoms between genuinely overlapping peaks.

    ``gamma < 1`` breaks the symmetry and makes a single broad atom strictly
    cheaper than any spread of the same flux, which is what a deconvolution
    ought to prefer.  ``gamma = 0`` is the other end: plain unweighted L1 on
    L2-normalised atoms, which over-merges and swallows genuine neighbours.  The
    usable range is interior, and the test-suite runs at ``gamma = 0.5``, which
    recovers a weak peak hidden in a strong peak's tail (``gamma = 1`` misses
    it) and cuts the reported peak count from 36 to 7 on the overlap cases.

    The default is 0.5.  It should still be calibrated like ``alpha`` or a
    regularisation strength, against the merge/split error pair, and that
    calibration is only meaningful away from 1.0 -- the single point where the
    penalty carries no information about model order at all.
    """

    def __init__(
        self,
        alpha: float | None = None,
        gamma: float = 0.5,
        min_sigma: float = 1.0,
        max_sigma: float = 5.0,
        num_sigmas: int = 5,
        loss: str = "poisson",
        show_steps: bool = False,
        ref_sigma: float = 1.0,
        chunk_size: int = 64,
        refine_positions: bool = True,
        reject_boundary_sigma: bool = False,
        boundary_sigma_frac: float = 0.98,
        **kwargs,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigmas = num_sigmas
        self.loss = loss
        self.show_steps = show_steps
        self.ref_sigma = ref_sigma
        self.chunk_size = chunk_size
        self.refine_positions = refine_positions
        self.reject_boundary_sigma = reject_boundary_sigma
        self.boundary_sigma_frac = boundary_sigma_frac

        # 1. Pre-build the Filter Bank
        self.sigmas = jnp.linspace(min_sigma, max_sigma, num_sigmas)
        self.max_k_rad = int(3.0 * max_sigma)

        # Use strictly unnormalized physical bases to preserve flux relationships
        self.K_weights, self.kernel_sq_norms = self._build_kernel_bank()
        self.K_sq = self.K_weights**2

    def effective_alpha(self, height, width):
        """Per-scale significance threshold, in units of coefficient noise.

        Solving globally tests every (pixel, scale) coefficient at once, so a
        bare significance level does not control the false-alarm rate.  The
        maximum of a smooth Gaussian field over ``N_k`` resolution elements sits
        near ``sqrt(2 log N_k)``, and that is the level at which the expected
        number of false detections over the whole image is O(1).

        With ``alpha=None`` the threshold is derived from that floor: the
        smallest level whose weighted profile clears it at every scale.  The
        ``(sigma/sigma_ref)**gamma`` shape is kept, because that shape is what
        sets the merge/split balance -- using the floor alone flattens it into
        the over-merging regime and loses weak peaks in the tails of strong
        ones.  Only the *level* is taken from the data.

        Passing a number instead keeps it as a lower bound on significance,
        still floored, so an explicit request can be stricter but not weaker
        than false-alarm control allows.
        """
        weights = (self.sigmas / self.ref_sigma) ** self.gamma
        n_tests = jnp.maximum(
            (height * width) / (2.0 * jnp.pi * jnp.maximum(self.sigmas**2, 1e-6)),
            2.0,
        )
        floor = jnp.sqrt(2.0 * jnp.log(n_tests))
        if self.alpha is None:
            return jnp.max(floor / weights) * weights
        return jnp.maximum(self.alpha * weights, floor)

    def _build_kernel_bank(self):
        k_grid = jnp.arange(-self.max_k_rad, self.max_k_rad + 1)
        yy, xx = jnp.meshgrid(k_grid, k_grid, indexing="ij")

        def build_one(s):
            sig_sq2 = s * jnp.sqrt(2.0) + 1e-6
            erf_y = jax.scipy.special.erf((yy + 0.5) / sig_sq2) - jax.scipy.special.erf(
                (yy - 0.5) / sig_sq2
            )
            erf_x = jax.scipy.special.erf((xx + 0.5) / sig_sq2) - jax.scipy.special.erf(
                (xx - 0.5) / sig_sq2
            )
            k_2d = (jnp.pi / 2.0) * (s**2) * erf_y * erf_x
            return k_2d

        kernels_2d = vmap(build_one)(self.sigmas)
        sq_norms = jnp.sum(kernels_2d**2, axis=(1, 2))
        return kernels_2d[:, None, :, :], sq_norms

    @staticmethod
    def _forward_op(c, weights):
        weights_fwd = weights.transpose(1, 0, 2, 3)
        return lax.conv_general_dilated(
            c,
            weights_fwd,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

    @staticmethod
    def _adjoint_op(u, weights):
        return lax.conv_general_dilated(
            u,
            weights,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

    @partial(jit, static_argnames=["self", "max_iter"])
    def _solve_ssn_cg_global(self, y_img, bg_img, max_iter=100):
        H, W = y_img.shape
        y = y_img[None, None, :, :]
        bg = bg_img[None, None, :, :]

        K = self.num_sigmas
        c_init = jnp.zeros((1, K, H, W))
        q_init = jnp.zeros((1, K, H, W))

        bg_med = jnp.maximum(jnp.median(bg), 1e-3).astype(jnp.float32)

        # 1. Exact Spatially Varying Variance Map & Lipschitz Bounds
        if self.loss == "gaussian":
            W_ref = jnp.ones_like(bg)
        else:
            W_ref = 1.0 / jnp.maximum(bg, 1e-3)

        H_diag = self._adjoint_op(W_ref, self.K_sq)
        H_diag_safe = jnp.maximum(H_diag, 1e-6)

        # Variance of each channel's coefficient estimate.  This has to be the
        # channel's own curvature: taking the across-channel maximum instead
        # understates the noise on every channel but the broadest -- by a factor
        # of ~11 for the narrowest basis in a typical bank -- so the fine scales
        # get thresholded far too weakly and fit noise into a smear of shifted
        # basis copies rather than one coefficient per peak.
        if self.loss == "gaussian":
            var_c = bg_med / H_diag_safe
        else:
            var_c = 1.0 / H_diag_safe

        alpha_vec = self.effective_alpha(H, W)

        # 2. L1 penalty weight, in units of the objective rather than of the
        # prox step.  Soft-thresholding at lam/H_diag recovers the intended
        # "alpha standard deviations of coefficient noise" cut, so lam must be
        # alpha * weight * H_diag * sqrt(var_c).  Deriving the penalty from the
        # step size instead makes the objective itself move whenever the step
        # does, which leaves the line search minimising a different problem on
        # every iteration.
        lam = alpha_vec[None, :, None, None] * H_diag_safe * jnp.sqrt(var_c)

        # 3. Step size.  A prox-gradient step is only guaranteed to descend for
        # tau <= 1/lambda_max(A^T W A).  The diagonal of that operator is not a
        # bound on its largest eigenvalue: these basis functions overlap heavily,
        # and for a typical bank lambda_max exceeds max(diag) by ~400x, so a step
        # built from the diagonal overshoots by that factor and collapses the
        # line search onto its smallest permitted step every iteration.  A few
        # power iterations give the bound directly; the kernels and weights are
        # positive, so a constant vector is a good starting guess.
        def power_step(_, v):
            Av = self._adjoint_op(
                W_ref * self._forward_op(v, self.K_weights), self.K_weights
            )
            return Av / (jnp.linalg.norm(Av) + 1e-12)

        v0 = jnp.ones((1, K, H, W), dtype=jnp.float32)
        v_top = lax.fori_loop(0, 15, power_step, v0 / jnp.linalg.norm(v0))
        Av_top = self._adjoint_op(
            W_ref * self._forward_op(v_top, self.K_weights), self.K_weights
        )
        L_max = jnp.sum(v_top * Av_top) / jnp.maximum(jnp.sum(v_top * v_top), 1e-12)
        tau_local = 1.0 / (L_max + 1e-4)

        tau_alpha = tau_local * lam

        def get_loss_grad_hess(c_curr):
            u = self._forward_op(c_curr, self.K_weights) + bg
            if self.loss == "gaussian":
                res = u - y
                nll = 0.5 * jnp.sum(res**2)
                grad = self._adjoint_op(res, self.K_weights)
                W_diag = jnp.ones_like(u)
            else:
                u_safe = jnp.maximum(u, 1e-6)
                nll = jnp.sum(u_safe - y * jnp.log(u_safe))
                res_1d = 1.0 - (y / u_safe)
                grad = self._adjoint_op(res_1d, self.K_weights)
                W_diag = 1.0 / jnp.maximum(u_safe, 1e-3)
            return nll, grad, W_diag, u

        def cond_fn(state):
            step, _, _, step_norm = state
            return (step < max_iter) & (step_norm > 1e-3)

        def body_fn(state):
            step, q, c, _ = state
            _, grad, W_diag, u_curr = get_loss_grad_hess(c)
            Gq = (q - c) / tau_local + grad

            # Strict Independent L1 Block Soft-Thresholding
            D_mat = (q > tau_alpha).astype(jnp.float32)

            # Semi-smooth Newton system for G(q) = (q - c(q))/tau + grad(c(q)).
            # With c(q) = max(0, q - tau_alpha), so dc_k/dq_j = D_k delta_kj,
            #
            #     dG_i/dq_j = delta_ij (1 - D_i)/tau + H_ij D_j,   H = A^T W A
            #
            # which in the (active, inactive) ordering is block lower-triangular
            # and *not* symmetric:
            #
            #     [ H_AA      0     ]
            #     [ H_IA    I/tau   ]
            #
            # Projecting H onto the active rows would make the whole operator
            # symmetric, but only by discarding H_IA -- the block carrying how a
            # change in the fitted peaks moves the residual at the coordinates
            # still below threshold, which is exactly what decides the atoms
            # that activate next.  That is an approximation with no error
            # control, and it is not needed: a triangular system does not
            # require symmetry.  Solve the active block, which is symmetric
            # positive definite on its own, then substitute.  One extra operator
            # application per iteration buys the exact Jacobian.
            # H is not dimensionless: A carries [counts]/[c] and W carries
            # 1/[counts], so H = A^T W A carries [counts]/[c]^2 -- both the
            # photon scale of the frame and the px^2 scale of the basis.  A bare
            # additive constant is therefore not a regularisation strength but
            # whatever fraction of diag(H) the data happens to make it.  Summed
            # over the bank, K^2 runs 2.9 to 78.3, so diag(H) differs by 27x
            # between the narrowest and broadest channel on the same frame, and
            # since W = 1/u it moves four more orders of magnitude between a
            # bright frame and a photon-starved one.  The old fixed 1e-4 was
            # 6.4e-4 of the diagonal at 500 counts and 8.2e-7 at the 0.64-count
            # mean of a real MANDI frame: the same line of code regularising
            # 800x differently depending on the data, which is why this bites on
            # real scans and not in the unit tests.
            #
            # Scale it by diag(H) instead, so eps is genuinely dimensionless.
            # Jacobi already divides by diag(H), so a relative ridge becomes
            # exactly eps in the preconditioned system and bounds its condition
            # number at ~1/eps whatever the count rate -- where an absolute
            # ridge became 1e-4 * eta and worked against the preconditioner.
            # eps is set from that bound and the CG budget, not fitted: 1e-3
            # gives a preconditioned condition number of ~1e3, whose sqrt is
            # within the existing maxiter=20.
            RIDGE_REL = 1e-3

            H_diag_local = jnp.maximum(self._adjoint_op(W_diag, self.K_sq), 1e-6)
            eta = 1.0 / H_diag_local

            def apply_active_block(v):
                v_active = v * D_mat
                Av = self._forward_op(v_active, self.K_weights)
                At_W_Av = self._adjoint_op(W_diag * Av, self.K_weights)
                ridge = RIDGE_REL * H_diag_local * v_active
                return (At_W_Av + ridge) * D_mat + (1.0 - D_mat) * v

            def jacobi(v):
                return eta * v * D_mat + (1.0 - D_mat) * v

            # Row 1: H_AA dq_A = -G_A.  The identity carried on the inactive
            # block keeps the operator non-singular for CG; with a right-hand
            # side supported on the active set every iterate stays there too, so
            # this solves the active block alone.
            dq_a, _ = jax.scipy.sparse.linalg.cg(
                apply_active_block, -Gq * D_mat, M=jacobi, tol=1e-3, maxiter=20
            )
            dq_a = jnp.where(jnp.isfinite(dq_a), dq_a, 0.0) * D_mat

            # Row 2: dq_I = -tau (G_I + (H dq_A)_I).  The 1e-4 ridge is diagonal
            # and the two index sets are disjoint, so it plays no part in H_IA.
            H_dq_a = self._adjoint_op(
                W_diag * self._forward_op(dq_a, self.K_weights), self.K_weights
            )
            dq_i = -tau_local * (Gq + H_dq_a) * (1.0 - D_mat)

            dq = jnp.where(jnp.isfinite(dq_a + dq_i), dq_a + dq_i, 0.0)

            # Change in objective from c to c_test, accumulated as a single
            # reduction over per-pixel *differences*.
            #
            # Forming it as j(c_test) - j(c) instead cannot work in float32.
            # Both are sums over the whole image of magnitude |J|, so their
            # difference is unresolvable below one ulp, ~6e-8 * |J| -- on a
            # 126x126 frame that floor is ~9e-3, and on a 542x542 detector
            # frame it is ~0.2.  The true decrease falls under it long before
            # the iterate is converged, at which point no amount of
            # backtracking can show an improvement: every halving looks like an
            # increase, the budget runs out, the step is rejected and the outer
            # loop stops.  Measured on a 6-frame synthetic case, all 12 solves
            # ended that way -- none reached the convergence test or the
            # iteration cap -- and the stopping iteration then moved with
            # anything that perturbed the arithmetic.
            #
            # Differencing per pixel first ties the accuracy to the decrease
            # rather than to |J|.  log1p keeps the Poisson term exact when the
            # trial point is close, which is precisely the regime that matters.
            # This is the same quantity, not an approximation; it also drops the
            # adjoint convolution the old objective computed and threw away on
            # every backtracking trial.
            def obj_delta(c_test):
                u_test = self._forward_op(c_test, self.K_weights) + bg
                if self.loss == "gaussian":
                    d_nll = 0.5 * jnp.sum(
                        (u_test - u_curr) * ((u_test - y) + (u_curr - y))
                    )
                else:
                    u_c = jnp.maximum(u_curr, 1e-6)
                    du = jnp.maximum(u_test, 1e-6) - u_c
                    d_nll = jnp.sum(du - y * jnp.log1p(du / u_c))
                return d_nll + jnp.sum(lam * (c_test - c))

            def bt_cond(bt_state):
                bt_i, _step_size, _, _, d_test = bt_state
                is_valid = jnp.isfinite(d_test)
                return (bt_i < 12) & ((d_test > 0.0) | ~is_valid)

            def bt_body(bt_state):
                bt_i, step_size, _, _, _ = bt_state
                step_size = jnp.float32(step_size * 0.5)

                q_test = q + step_size * dq
                c_test = jnp.maximum(0.0, q_test - tau_alpha)

                return (bt_i + 1, step_size, q_test, c_test, obj_delta(c_test))

            q_test_init = q + dq
            c_test_init = jnp.maximum(0.0, q_test_init - tau_alpha)

            bt_init = (
                0,
                jnp.float32(1.0),
                q_test_init,
                c_test_init,
                obj_delta(c_test_init),
            )
            _, _, q_try, c_try, d_try = lax.while_loop(bt_cond, bt_body, bt_init)

            # If the backtracking budget ran out without finding a decrease the
            # trial point is worse than where we started, so reject it instead
            # of stepping uphill.  A rejected step reports a zero step norm,
            # which stops the outer loop rather than letting it thrash.
            accept = jnp.isfinite(d_try) & (d_try <= 0.0)
            q_final = jnp.where(accept, q_try, q)
            c_final = jnp.where(accept, c_try, c)
            step_norm = jnp.where(accept, jnp.linalg.norm(q_final - q), 0.0)

            return (step + 1, q_final, c_final, step_norm)

        init_state = (0, q_init, c_init, jnp.float32(1e9))
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        _, _, c_l1, _ = final_state

        return c_l1[0]

    @partial(jit, static_argnames=["self", "border"])
    def _extract_peaks(self, c_tensor, border=0):
        c_tot = jnp.sum(c_tensor, axis=0)  # [H, W]

        # Smooth the discrete L1 coefficients to recover true continuous center
        # of mass.  L1 splinters a peak across adjacent pixels, so this only
        # needs to span that one-pixel scale: a wider kernel throws away exactly
        # the separation the fine basis channels were there to resolve, merging
        # neighbouring peaks a few pixels apart into one maximum.  Cap it at the
        # finest scale in the bank.
        smooth_sigma = min(1.0, float(self.min_sigma))
        sig_sq2 = smooth_sigma * jnp.sqrt(2.0) + 1e-6
        k_half = max(1, round(2.0 * smooth_sigma))
        k_grid = jnp.arange(-k_half, k_half + 1)
        k_1d = jax.scipy.special.erf((k_grid + 0.5) / sig_sq2) - jax.scipy.special.erf(
            (k_grid - 0.5) / sig_sq2
        )

        c_smooth_temp = jax.scipy.signal.correlate2d(c_tot, k_1d[:, None], mode="same")
        c_smooth = jax.scipy.signal.correlate2d(
            c_smooth_temp, k_1d[None, :], mode="same"
        )

        window = (3, 3)
        c_max = lax.reduce_window(
            c_smooth, -jnp.inf, jax.lax.max, window, (1, 1), "SAME"
        )
        is_max = (c_smooth == c_max) & (c_smooth > 1e-5)

        # Discard maxima in the replicated border before ranking, not after.
        # The edge padding is a constant strip that the finest basis fits
        # readily, so it carries many strong spurious maxima; leaving them in
        # until after the top-MAX_PEAKS cut lets them consume the whole budget
        # and silently drop the real peaks from the interior.
        if border > 0:
            interior = jnp.zeros_like(is_max)
            interior = interior.at[border:-border, border:-border].set(True)
            is_max = is_max & interior

        c_flat = jnp.where(is_max.flatten(), c_smooth.flatten(), -1.0)

        MAX_PEAKS = 100
        top_indices = jnp.argsort(c_flat)[::-1][:MAX_PEAKS]
        valid_mask = c_flat[top_indices] > 1e-5

        def process_peak(idx):
            r = idx // c_smooth.shape[1]
            c = idx % c_smooth.shape[1]

            r_safe = jnp.clip(r, 1, c_smooth.shape[0] - 2)
            c_safe = jnp.clip(c, 1, c_smooth.shape[1] - 2)

            # Extract 3x3 patch to integrate splintered coefficients
            c_patch = lax.dynamic_slice(
                c_tensor, (0, r_safe - 1, c_safe - 1), (c_tensor.shape[0], 3, 3)
            )
            c_channels = jnp.sum(c_patch, axis=(1, 2))

            # Exact Flux & Variance Preservation
            # Flux of basis k is A_k * sigma_k^2
            flux_k = c_channels * (self.sigmas**2)
            total_flux_scaled = jnp.sum(flux_k) + 1e-9

            # Variance of mixture is sum(Flux_k * sigma_k^2) / sum(Flux_k).
            # Floor it at the finest basis: a slot that matched nothing has zero
            # flux in every channel, and dividing by a zero width below turns its
            # amplitude into an infinity.  Those slots are discarded by the
            # validity mask, but an infinity multiplied by a zero mask is a NaN,
            # which then contaminates anything that consumes the whole array.
            sigma_sq_eff = jnp.maximum(
                jnp.sum(flux_k * (self.sigmas**2)) / total_flux_scaled,
                float(self.min_sigma) ** 2,
            )
            sigma_eff = jnp.sqrt(sigma_sq_eff)

            # Convert flux back to effective central amplitude
            amp_eff = total_flux_scaled / sigma_sq_eff

            # Localise from the raw coefficients, not from the smoothed map used
            # to find the maximum.  Those are two different jobs: smoothing helps
            # decide *whether* there is a peak here, but it also drags the apex
            # toward any neighbouring peak, so reading the position off it makes
            # the centre wander.  L1 splinters a peak across the pixels it
            # straddles, so the coefficient-weighted centroid is the sub-pixel
            # position, and it is exact for a peak sitting on a pixel.
            #
            # The window is kept at the one-pixel splintering scale on purpose.
            # Widening it to follow the fitted width was tried and is worse: the
            # broad channels carry diffuse coefficients, so a wider window pulls
            # the centre off the peak and reaches into close neighbours.
            patch_tot = jnp.sum(c_patch, axis=0)
            patch_mass = jnp.sum(patch_tot) + 1e-9
            offsets = jnp.array([-1.0, 0.0, 1.0])
            dr = jnp.sum(patch_tot * offsets[:, None]) / patch_mass
            dc = jnp.sum(patch_tot * offsets[None, :]) / patch_mass

            r_cont = r_safe + jnp.clip(dr, -1.0, 1.0)
            c_cont = c_safe + jnp.clip(dc, -1.0, 1.0)

            return jnp.array([amp_eff, r_cont, c_cont, sigma_eff])

        extracted = vmap(process_peak)(top_indices)
        return extracted, valid_mask

    @partial(jit, static_argnames=["self", "n_steps"])
    def _refine_peaks(self, y_img, bg_img, peaks, active, n_steps=200):
        """Continuous ("sliding") refinement of the selected support.

        The convex solve picks *which* atoms are present, but it can only place
        them on the integer grid the dictionary is built on, so a peak that sits
        between pixels is split across neighbours and its position is only
        recoverable to whatever the centroid heuristic manages.  Re-fitting the
        selected peaks' continuous (amplitude, row, column, sigma) against the
        same Poisson objective lifts the answer off the grid: this is the
        sliding step of a Frank-Wolfe scheme over measures, and it is what makes
        the recovered positions sub-pixel rather than sub-grid.

        Peaks are rendered on their own bounding box and scattered into the
        model, so cost is O(n_peaks * patch^2) rather than O(n_peaks * H * W)
        and this stays affordable on full detector images.
        """
        H, W = y_img.shape
        P = 2 * self.max_k_rad + 1
        off = jnp.arange(P, dtype=jnp.float32)
        mask = active.astype(jnp.float32)

        # Replace unused slots outright instead of relying on the mask to cancel
        # them.  Masking by multiplication cannot neutralise a non-finite value
        # -- inf * 0 is NaN -- and one such row is enough to make the objective,
        # and therefore every gradient, NaN.  The guard against non-finite
        # gradients further down would then silently turn refinement into a
        # no-op rather than reporting anything.
        safe = jnp.stack(
            [
                jnp.ones_like(peaks[:, 0]),
                jnp.full_like(peaks[:, 1], float(self.max_k_rad)),
                jnp.full_like(peaks[:, 2], float(self.max_k_rad)),
                jnp.full_like(peaks[:, 3], float(self.min_sigma)),
            ],
            axis=1,
        )
        finite = jnp.all(jnp.isfinite(peaks), axis=1) & active
        peaks = jnp.where(finite[:, None], peaks, safe)
        mask = finite.astype(jnp.float32)

        # Unconstrained parameterisation keeps amplitude positive and sigma
        # inside the bank's range without needing a projection each step.
        lo, hi = float(self.min_sigma), float(self.max_sigma)
        sig0 = jnp.clip(peaks[:, 3], lo + 1e-3, hi - 1e-3)
        u_init = jnp.stack(
            [
                jnp.log(jnp.maximum(peaks[:, 0], 1e-3)),
                peaks[:, 1],
                peaks[:, 2],
                jnp.log((sig0 - lo) / jnp.maximum(hi - sig0, 1e-6)),
            ],
            axis=1,
        )

        def physical(u):
            amp = jnp.exp(u[:, 0])
            sig = lo + (hi - lo) * jax.nn.sigmoid(u[:, 3])
            return amp, u[:, 1], u[:, 2], sig

        def render(u):
            amp, r, c, sig = physical(u)
            s2 = sig * jnp.sqrt(2.0) + 1e-6
            r0 = jnp.clip(jnp.round(r).astype(jnp.int32) - self.max_k_rad, 0, H - P)
            c0 = jnp.clip(jnp.round(c).astype(jnp.int32) - self.max_k_rad, 0, W - P)
            rr = r0[:, None].astype(jnp.float32) + off[None, :]
            cc = c0[:, None].astype(jnp.float32) + off[None, :]

            def erf_span(grid, centre):
                d = grid - centre[:, None]
                return jax.scipy.special.erf(
                    (d + 0.5) / s2[:, None]
                ) - jax.scipy.special.erf((d - 0.5) / s2[:, None])

            ey = erf_span(rr, r)
            ex = erf_span(cc, c)
            amp_s = amp * (jnp.pi / 2.0) * (sig**2) * mask
            patch = amp_s[:, None, None] * ey[:, :, None] * ex[:, None, :]

            idx_r = jnp.broadcast_to(
                (r0[:, None] + jnp.arange(P))[:, :, None], (peaks.shape[0], P, P)
            )
            idx_c = jnp.broadcast_to(
                (c0[:, None] + jnp.arange(P))[:, None, :], (peaks.shape[0], P, P)
            )
            return jnp.zeros((H, W)).at[idx_r, idx_c].add(patch)

        def nll(u):
            model = jnp.maximum(render(u) + bg_img, 1e-6)
            return jnp.sum(model - y_img * jnp.log(model))

        grad_fn = jax.value_and_grad(nll)

        def adam_step(state, _):
            u, m, v, t = state
            _, g = grad_fn(u)
            g = jnp.where(jnp.isfinite(g), g, 0.0) * mask[:, None]
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            t = t + 1.0
            upd = (m / (1.0 - 0.9**t)) / (jnp.sqrt(v / (1.0 - 0.999**t)) + 1e-8)
            return (u - 0.05 * upd, m, v, t), None

        (u_fin, _, _, _), _ = lax.scan(
            adam_step,
            (u_init, jnp.zeros_like(u_init), jnp.zeros_like(u_init), 0.0),
            None,
            length=n_steps,
        )
        u_fin = jnp.where(jnp.isfinite(u_fin), u_fin, u_init)

        amp, r, c, sig = physical(u_fin)
        refined = jnp.stack([amp, r, c, sig], axis=1)
        # A refined peak that ran away from its own bounding box is not a
        # refinement of that peak any more, so keep the pre-fit values there.
        moved = jnp.sqrt((r - peaks[:, 1]) ** 2 + (c - peaks[:, 2]) ** 2)
        keep = (
            active
            & (moved < float(self.max_k_rad))
            & jnp.all(jnp.isfinite(refined), axis=1)
        )
        return jnp.where(keep[:, None], refined, peaks)

    def find_peaks_batch(self, images_batch):
        # Detector frames arrive as integer counts.  Everything below is a
        # convolution, and cuDNN will not lower an integer convolution, so the
        # cast is required rather than cosmetic: without it the background
        # estimate fails outright on real data.
        images_batch = np.asarray(images_batch, dtype=np.float32)
        B, H, W = images_batch.shape

        filter_size = max(15, int(self.max_sigma * 5))
        bg_map = np.full_like(images_batch, 10.0)
        try:
            from subhkl.search.sparse_rbf import compute_bg_batch

            # Chunked for the same reason the greedy finder chunks it: a full
            # detector scan is far too much to hold on the device at once.
            bg_chunk = min(self.chunk_size, max(1, B // 4))
            pieces = []
            for start in range(0, B, bg_chunk):
                piece = compute_bg_batch(
                    jnp.asarray(
                        images_batch[start : start + bg_chunk], dtype=jnp.float32
                    ),
                    filter_size,
                )
                piece.block_until_ready()
                pieces.append(np.asarray(piece, dtype=np.float32))
            bg_map = np.concatenate(pieces, axis=0)
            if bg_map.shape != images_batch.shape:
                bg_map_fixed = np.zeros_like(images_batch)
                mh, mw = min(H, bg_map.shape[1]), min(W, bg_map.shape[2])
                bg_map_fixed[:, :mh, :mw] = bg_map[:, :mh, :mw]
                bg_map = bg_map_fixed
        except ImportError:
            pass

        self._last_bg_map = bg_map

        PAD = 2 * self.max_k_rad + 1
        pad_y = PAD // 2
        pad_x = PAD // 2

        images_padded = jnp.pad(
            images_batch, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="edge"
        )
        bg_padded = jnp.pad(
            bg_map, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="edge"
        )

        results = []
        rejected_counts = []

        MARGIN = max(3, self.max_k_rad)

        # vmap multiplies the solver's per-image working set by the batch, so a
        # full detector scan does not fit on the card: on the 1114-frame CG4D
        # garnet stack XLA asks for an 82 GiB program and the allocator gives up
        # at 69.8 GiB on a 96 GB H100.  Chunk it for the same reason the
        # background estimate above is chunked.  Peaks are pulled back to the
        # host inside the chunk loop, which keeps the coefficient tensors, at
        # [chunk, num_sigmas, H, W] the largest thing here, from accumulating.
        #
        # The images are solved independently, so chunking is not an
        # approximation -- but it is not bit-identical either.  XLA compiles a
        # separate kernel per batch shape, and the resulting rounding
        # differences can carry a coefficient across the selection threshold: on
        # a 6-frame synthetic case, solving one image at a time rather than all
        # six moved two frames' peak counts.  A batch of chunk_size or fewer
        # still takes a single chunk and so matches the unchunked path exactly,
        # which at the default of 64 covers every unit test here.
        #
        # Unlike the background rule this does not also divide by four: the
        # solver is the expensive stage, chunk_size is the knob documented to
        # control exactly this, and subdividing a batch that already fits only
        # gives up vmap parallelism.
        solve_chunk = max(1, min(self.chunk_size, B))
        solve_batch = jax.jit(jax.vmap(self._solve_ssn_cg_global))

        for start in range(0, B, solve_chunk):
            stop = min(start + solve_chunk, B)
            c_tensors = solve_batch(images_padded[start:stop], bg_padded[start:stop])

            for i in range(start, stop):
                peaks_padded, valid_mask = self._extract_peaks(
                    c_tensors[i - start], border=pad_y + MARGIN
                )

                # Slide the selected support off the grid before un-padding, while
                # the coordinates still match the image the model is rendered on.
                if self.refine_positions:
                    peaks_padded = self._refine_peaks(
                        images_padded[i], bg_padded[i], peaks_padded, valid_mask
                    )

                peaks_np = np.array(peaks_padded)
                mask_np = np.array(valid_mask)

                valid_peaks = peaks_np[mask_np]

                valid_peaks[:, 1] -= pad_y
                valid_peaks[:, 2] -= pad_x

                keep = (
                    (valid_peaks[:, 1] >= MARGIN)
                    & (valid_peaks[:, 1] < H - MARGIN)
                    & (valid_peaks[:, 2] >= MARGIN)
                    & (valid_peaks[:, 2] < W - MARGIN)
                )

                # An atom whose width has run to the edge of the bank is the solver
                # asking for a wider basis than it was given.  That can mean
                # unmodelled smooth background -- or simply that max_sigma is too
                # small for the data, in which case every real peak saturates the
                # bank too and this discards them.  On a real MANDI scan, where the
                # peaks have a median width of ~34 px against a max_sigma of 5, it
                # removed 87% of genuine detections (466 peaks down to 60), so it is
                # off by default and is a diagnostic to reach for once the bank is
                # known to be wide enough.  The case
                # that motivated this is a diffuse halo whose background estimate
                # falls ~20% short at its centre, leaving a broad positive residual
                # that is then reported as a reflection sitting on the halo.  On real
                # Laue data the structures that trigger it -- thermal diffuse
                # scattering, powder rings, halos around strong reflections -- are
                # exactly the ones that should not be reported as peaks.
                if self.reject_boundary_sigma:
                    at_boundary = (
                        valid_peaks[:, 3] >= self.boundary_sigma_frac * self.max_sigma
                    ) & keep
                    n_rejected = int(np.count_nonzero(at_boundary))
                    keep &= ~at_boundary
                else:
                    n_rejected = 0

                # Record rather than drop silently: a run that discards many atoms
                # here is telling you the background model is leaving structure
                # behind, or that max_sigma is too small for this data, and either
                # is worth knowing.
                rejected_counts.append(n_rejected)
                if self.show_steps and n_rejected:
                    print(
                        f"  > Rejected {n_rejected} atom(s) with sigma at the bank "
                        f"edge (>= {self.boundary_sigma_frac:.2f} * "
                        f"{self.max_sigma:g}); likely unmodelled background."
                    )

                results.append(valid_peaks[keep])

            # Release the chunk before the next one is dispatched, so peak
            # device memory stays at one chunk rather than two.
            del c_tensors

        self.n_boundary_rejected = rejected_counts
        return results

    @partial(jit, static_argnames=["self"])
    def _predict_batch_scan(self, peaks, x_grid):
        def render_peak(p):
            total_I, r, c, sigma = p[0], p[1], p[2], p[3]
            sig_sq2 = sigma * jnp.sqrt(2.0) + 1e-6
            erf_y = jax.scipy.special.erf(
                (x_grid[0] - r + 0.5) / sig_sq2
            ) - jax.scipy.special.erf((x_grid[0] - r - 0.5) / sig_sq2)
            erf_x = jax.scipy.special.erf(
                (x_grid[1] - c + 0.5) / sig_sq2
            ) - jax.scipy.special.erf((x_grid[1] - c - 0.5) / sig_sq2)
            return total_I * (jnp.pi / 2.0) * (sigma**2) * erf_y * erf_x * (total_I > 0)

        def render_image(peaks_img):
            return jnp.sum(vmap(render_peak)(peaks_img), axis=0)

        return render_image(peaks)

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
                # 2. Exact Poisson Deviance
                term = jax.scipy.special.xlogy(target_raw, target_raw / recon_total) - (
                    target_raw - recon_total
                )
                dev = 2 * jnp.sum(term)
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
