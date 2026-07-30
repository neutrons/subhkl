import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
import jax.scipy.signal
import jax.scipy.sparse.linalg
import numpy as np
from tqdm import tqdm
from functools import partial

from subhkl.search import peak_model


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
    ought to prefer.  Measured on the overlap regression cases, ``gamma=0.75``
    both recovers a weak peak hidden in a strong peak's tail (which
    ``gamma=1`` misses) and cuts the reported peak count from 36 to 7.
    ``gamma`` is left at 1.0 here only for backwards compatibility; 0.75 is the
    better default and the callers in the test-suite that pass 1.0 explicitly
    are sitting on the degenerate point.
    """
    def __init__(
        self,
        alpha: float = 4.0,
        gamma: float = 1.0,
        min_sigma: float = 1.0,
        max_sigma: float = 5.0,
        num_sigmas: int = 5,
        loss: str = "poisson",
        show_steps: bool = False,
        ref_sigma: float = 1.0,
        refine_positions: bool = True,
        **kwargs
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigmas = num_sigmas
        self.loss = loss
        self.show_steps = show_steps
        self.ref_sigma = ref_sigma
        self.refine_positions = refine_positions

        # 1. Pre-build the Filter Bank
        self.sigmas = jnp.linspace(min_sigma, max_sigma, num_sigmas)
        self.max_k_rad = int(3.0 * max_sigma)
        
        # Use strictly unnormalized physical bases to preserve flux relationships
        self.K_weights, self.kernel_sq_norms = self._build_kernel_bank()
        self.K_sq = self.K_weights ** 2

    def _build_kernel_bank(self):
        return peak_model.kernel_bank(self.sigmas, self.max_k_rad)

    @staticmethod
    def _forward_op(c, weights):
        weights_fwd = weights.transpose(1, 0, 2, 3) 
        return lax.conv_general_dilated(
            c, weights_fwd, window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )

    @staticmethod
    def _adjoint_op(u, weights):
        return lax.conv_general_dilated(
            u, weights, window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
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

        weights = (self.sigmas / self.ref_sigma) ** self.gamma
        alpha_vec = self.alpha * weights

        # Multiplicity correction.  In the greedy predecessor `alpha` gated one
        # candidate per search window, so it really was a per-peak significance
        # level.  Solving globally tests every (pixel, scale) coefficient at
        # once, and the noise maximum over N independent tests sits at about
        # sqrt(2 log N) standard deviations, so a bare `alpha` of a few sigma no
        # longer controls the false-alarm rate at all -- the narrowest basis is
        # both the most numerous and, under a Besov weight below ref_sigma, the
        # most permissive, so it fits noise spikes everywhere.  Counting
        # resolution elements per scale restores the intended meaning of alpha
        # without changing it where the user has already asked for more.
        n_tests = jnp.maximum(
            (H * W) / (2.0 * jnp.pi * jnp.maximum(self.sigmas**2, 1e-6)), 2.0
        )
        alpha_floor = jnp.sqrt(2.0 * jnp.log(n_tests))
        alpha_vec = jnp.maximum(alpha_vec, alpha_floor)

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
            return nll, grad, W_diag

        def cond_fn(state):
            step, _, _, step_norm = state
            return (step < max_iter) & (step_norm > 1e-3)

        def body_fn(state):
            step, q, c, _ = state
            nll, grad, W_diag = get_loss_grad_hess(c)
            Gq = (q - c) / tau_local + grad

            # Strict Independent L1 Block Soft-Thresholding
            D_mat = (q > tau_alpha).astype(jnp.float32)

            # Semi-smooth Newton system for G(q) = (q - c(q))/tau + grad(c(q)).
            # Since c(q) = max(0, q - tau_alpha), the generalised Jacobian is the
            # Gauss-Newton Hessian A^T W A on the active set (D_mat == 1) and
            # 1/tau_local on the inactive set.  Both blocks are carried in a
            # single operator that is symmetric by construction: CG is only
            # valid for symmetric positive-definite operators, so the Jacobi
            # scaling is supplied through the preconditioner argument rather
            # than multiplied into the operator, which would break symmetry and
            # leave CG returning a direction that is not a descent direction.
            eta = 1.0 / jnp.maximum(self._adjoint_op(W_diag, self.K_sq), 1e-6)

            def apply_jacobian(v):
                v_active = v * D_mat
                Av = self._forward_op(v_active, self.K_weights)
                At_W_Av = self._adjoint_op(W_diag * Av, self.K_weights)
                return (At_W_Av + 1e-4 * v_active) * D_mat + (1.0 - D_mat) * v

            def jacobi(v):
                return eta * v * D_mat + (1.0 - D_mat) * v

            # Active rows solve A^T W A dq = -G; inactive rows reduce to the
            # explicit prox-gradient step dq = -tau_local * G.
            rhs = -Gq * D_mat - tau_local * Gq * (1.0 - D_mat)
            dq, _ = jax.scipy.sparse.linalg.cg(
                apply_jacobian, rhs, M=jacobi, tol=1e-3, maxiter=20
            )
            dq = jnp.where(jnp.isfinite(dq), dq, 0.0)

            def objective(c_test):
                j_test, _, _ = get_loss_grad_hess(c_test)
                return j_test + jnp.sum(lam * c_test)

            def bt_cond(bt_state):
                bt_i, step_size, _, _, j_test, j_curr = bt_state
                is_valid = jnp.isfinite(j_test)
                return (bt_i < 12) & ((j_test > j_curr) | ~is_valid)

            def bt_body(bt_state):
                bt_i, step_size, _, _, _, j_curr = bt_state
                step_size = jnp.float32(step_size * 0.5)

                q_test = q + step_size * dq
                c_test = jnp.maximum(0.0, q_test - tau_alpha)

                return (bt_i + 1, step_size, q_test, c_test, objective(c_test), j_curr)

            q_test_init = q + dq
            c_test_init = jnp.maximum(0.0, q_test_init - tau_alpha)
            obj_val_curr = nll + jnp.sum(lam * c)

            bt_init = (
                0,
                jnp.float32(1.0),
                q_test_init,
                c_test_init,
                objective(c_test_init),
                obj_val_curr,
            )
            _, _, q_try, c_try, j_try, _ = lax.while_loop(bt_cond, bt_body, bt_init)

            # If the backtracking budget ran out without finding a decrease the
            # trial point is worse than where we started, so reject it instead
            # of stepping uphill.  A rejected step reports a zero step norm,
            # which stops the outer loop rather than letting it thrash.
            accept = jnp.isfinite(j_try) & (j_try <= obj_val_curr)
            q_final = jnp.where(accept, q_try, q)
            c_final = jnp.where(accept, c_try, c)
            step_norm = jnp.where(accept, jnp.linalg.norm(q_final - q), 0.0)

            return (step + 1, q_final, c_final, step_norm)

        init_state = (0, q_init, c_init, jnp.float32(1e9))
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        _, _, c_l1, _ = final_state

        # === DEBIASING PHASE ===
        active_mask = (c_l1 > 1e-5).astype(jnp.float32)

        def debias_cond(state):
            step, _, actual_step_norm = state
            return (step < 50) & (actual_step_norm > 1e-4)

        tau_debias = jnp.float32(0.8 if self.loss == "poisson" else 1.0)

        def debias_body(state):
            step, c_deb, _ = state
            _, grad, W_diag = get_loss_grad_hess(c_deb)

            # For the Gaussian loss W_diag is identically one, so this reduces to
            # the H_diag computed above; sharing one expression keeps the two
            # branches from drifting apart.
            eta = 1.0 / jnp.maximum(self._adjoint_op(W_diag, self.K_sq), 1e-6)

            # Same requirement as the SSN solve above: the operator handed to CG
            # must be symmetric, so eta enters as a preconditioner and the
            # right-hand side stays -grad rather than -eta * grad.
            def apply_hessian(v):
                v_active = v * active_mask
                Av = self._forward_op(v_active, self.K_weights)
                At_W_Av = self._adjoint_op(W_diag * Av, self.K_weights)
                return (At_W_Av + 1e-4 * v_active) * active_mask + (
                    1.0 - active_mask
                ) * v

            def jacobi(v):
                return eta * v * active_mask + (1.0 - active_mask) * v

            dc, _ = jax.scipy.sparse.linalg.cg(
                apply_hessian,
                -grad * active_mask,
                M=jacobi,
                tol=1e-4,
                maxiter=50,
            )
            dc = jnp.where(jnp.isfinite(dc), dc, 0.0)

            c_new = jnp.maximum(0.0, c_deb + tau_debias * dc * active_mask)
            # Never let a non-finite iterate propagate: it would wipe out every
            # coefficient and the run would silently report no peaks at all.
            c_new = jnp.where(jnp.isfinite(c_new), c_new, c_deb)

            actual_step = c_new - c_deb
            return (step + 1, c_new, jnp.linalg.norm(actual_step))

        debias_state = lax.while_loop(debias_cond, debias_body, (0, c_l1, jnp.float32(1e9)))
        _, c_final, _ = debias_state

        return c_final[0]

    @partial(jit, static_argnames=["self", "border"])
    def _extract_peaks(self, c_tensor, border=0):
        c_tot = jnp.sum(c_tensor, axis=0) # [H, W]
        
        # Smooth the discrete L1 coefficients to recover true continuous center
        # of mass.  L1 splinters a peak across adjacent pixels, so this only
        # needs to span that one-pixel scale: a wider kernel throws away exactly
        # the separation the fine basis channels were there to resolve, merging
        # neighbouring peaks a few pixels apart into one maximum.  Cap it at the
        # finest scale in the bank.
        smooth_sigma = min(1.0, float(self.min_sigma))
        k_half = max(1, round(2.0 * smooth_sigma))
        k_grid = jnp.arange(-k_half, k_half + 1)
        k_1d = peak_model.pixel_integrated_gaussian_1d(k_grid, 0.0, smooth_sigma)
        
        c_smooth_temp = jax.scipy.signal.correlate2d(c_tot, k_1d[:, None], mode="same")
        c_smooth = jax.scipy.signal.correlate2d(c_smooth_temp, k_1d[None, :], mode="same")
        
        window = (3, 3)
        c_max = lax.reduce_window(c_smooth, -jnp.inf, jax.lax.max, window, (1, 1), 'SAME')
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
            c_patch = lax.dynamic_slice(c_tensor, (0, r_safe - 1, c_safe - 1), (c_tensor.shape[0], 3, 3))
            c_channels = jnp.sum(c_patch, axis=(1, 2))

            # Exact Flux & Variance Preservation
            # Flux of basis k is A_k * sigma_k^2
            flux_k = c_channels * (self.sigmas ** 2)
            total_flux_scaled = jnp.sum(flux_k) + 1e-9

            # Variance of mixture is sum(Flux_k * sigma_k^2) / sum(Flux_k)
            sigma_sq_eff = jnp.sum(flux_k * (self.sigmas ** 2)) / total_flux_scaled
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
        mask = active.astype(jnp.float32)
        lo, hi = float(self.min_sigma), float(self.max_sigma)

        # Unconstrained parameterisation keeps amplitude positive and sigma
        # inside the bank's range without needing a projection each step.
        log_amp, logit_sig = peak_model.to_unconstrained(
            peaks[:, 0], peaks[:, 3], lo, hi
        )
        u_init = jnp.stack([log_amp, peaks[:, 1], peaks[:, 2], logit_sig], axis=1)

        def physical(u):
            amp, sig = peak_model.to_physical(u[:, 0], u[:, 3], lo, hi)
            return amp, u[:, 1], u[:, 2], sig

        def nll(u):
            amp, r, c, sig = physical(u)
            model = peak_model.render_patches(
                (H, W), amp, r, c, sig, self.max_k_rad, active=mask
            )
            return peak_model.poisson_nll(model + bg_img, y_img)

        grad_fn = jax.value_and_grad(nll)

        def adam_step(state, _):
            u, m, v, t = state
            _, g = grad_fn(u)
            g = jnp.where(jnp.isfinite(g), g, 0.0) * mask[:, None]
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            t = t + 1.0
            upd = (m / (1.0 - 0.9**t)) / (
                jnp.sqrt(v / (1.0 - 0.999**t)) + 1e-8
            )
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
        keep = active & (moved < float(self.max_k_rad)) & jnp.all(
            jnp.isfinite(refined), axis=1
        )
        return jnp.where(keep[:, None], refined, peaks)

    def find_peaks_batch(self, images_batch):
        B, H, W = images_batch.shape
        
        filter_size = max(15, int(self.max_sigma * 5))
        bg_map = np.ones_like(images_batch) * 10.0
        try:
            from subhkl.search.sparse_rbf import compute_bg_batch
            bg_map = np.array(compute_bg_batch(jnp.array(images_batch), filter_size))
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

        images_padded = jnp.pad(images_batch, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        bg_padded = jnp.pad(bg_map, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="edge")

        results = []
        c_tensors = jax.jit(jax.vmap(self._solve_ssn_cg_global))(images_padded, bg_padded)

        MARGIN = max(3, self.max_k_rad)

        for i in range(B):
            peaks_padded, valid_mask = self._extract_peaks(
                c_tensors[i], border=pad_y + MARGIN
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

            in_bounds = (
                (valid_peaks[:, 1] >= MARGIN) & (valid_peaks[:, 1] < H - MARGIN) &
                (valid_peaks[:, 2] >= MARGIN) & (valid_peaks[:, 2] < W - MARGIN)
            )
            
            results.append(valid_peaks[in_bounds])

        return results

    @partial(jit, static_argnames=["self"])
    def _predict_batch_scan(self, peaks, x_grid):
        def render_peak(p):
            total_I, r, c, sigma = p[0], p[1], p[2], p[3]
            rendered = peak_model.render_peak(
                x_grid[0], x_grid[1], total_I, r, c, sigma
            )
            return rendered * (total_I > 0)

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
