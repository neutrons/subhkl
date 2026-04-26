"""
Semi-Smooth Newton (SSN) Solver for L1-Regularized Sparse Recovery.
Shared engine for Peak Integration and Sparse Hough Zone Axis Indexing.
"""
import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial

@partial(jit, static_argnames=["max_iter", "loss_type", "force_target"])
def solve_ssn_unified(
    A, y, bg_flat, alpha_vec, loss_type, c_warm, max_iter=20, force_target=False
):
    N_peaks = A.shape[1]
    N_params = N_peaks
    q_init = c_warm.astype(jnp.float32)

    bg_med = jnp.maximum(jnp.median(bg_flat), 1e-3).astype(jnp.float32)

    def get_loss_grad_hess(c):
        u = A @ c + bg_flat

        if loss_type == 1: # Poisson
            u_safe = jnp.maximum(u, 1e-6)
            nll = jnp.sum(u_safe - y * jnp.log(u_safe))
            grad = A.T @ (1.0 - y / u_safe)
            W_diag = 1.0 / jnp.maximum(u_safe, 1e-3)
            hess = A.T @ (W_diag[:, None] * A)
            
        elif loss_type == 2: # Huber
            # Threshold (delta) is 3 standard deviations of the background
            delta = 3.0 * jnp.sqrt(bg_med)
            e = u - y
            abs_e = jnp.abs(e)
            is_inlier = abs_e <= delta
            
            # Loss: 0.5 * e^2 for inliers, delta * |e| - 0.5 * delta^2 for outliers
            nll = jnp.sum(jnp.where(is_inlier, 0.5 * e**2, delta * abs_e - 0.5 * delta**2))
            
            # IRLS Weights for gradient and Hessian
            W_diag = jnp.where(is_inlier, 1.0, delta / jnp.maximum(abs_e, 1e-6))
            
            grad = A.T @ (W_diag * e)
            hess = A.T @ (W_diag[:, None] * A)
            
        else: # Gaussian (MSE)
            nll = 0.5 * jnp.sum((u - y) ** 2)
            grad = A.T @ (u - y)
            hess = A.T @ A

        return nll, grad, hess

    def cond_fn(state):
        step, _, _, dq_norm = state
        return (step < max_iter) & (dq_norm > 1e-3)

    def body_fn(state):
        step, q, c, _ = state
        nll, grad, hess = get_loss_grad_hess(c)

        L = jnp.max(jnp.diag(hess)) + 1e-4
        tau = 1.0 / L

        var_c = jnp.where(loss_type == 1, tau, bg_med * tau)
        tau_alpha = alpha_vec * jnp.sqrt(var_c)

        Gq = (q - c) / tau + grad

        D = (q > tau_alpha).astype(jnp.float32)
        DP_mat = jnp.diag(D)
        I = jnp.eye(N_params, dtype=jnp.float32)

        DG = (I - DP_mat) / tau + hess @ DP_mat + 1e-4 * I
        dq = jnp.linalg.solve(DG, -Gq).astype(jnp.float32)

        def bt_cond(bt_state):
            bt_i, step_size, _, _, j_test, j_curr = bt_state
            is_valid = jnp.isfinite(j_test)
            return (bt_i < 8) & ((j_test > j_curr) | ~is_valid)

        def bt_body(bt_state):
            bt_i, step_size, _, _, _, j_curr = bt_state
            step_size = jnp.float32(step_size * 0.5)

            q_test = (q + step_size * dq).astype(jnp.float32)
            c_test = jnp.maximum(0.0, q_test - tau_alpha).astype(jnp.float32)

            j_test, _, _ = get_loss_grad_hess(c_test)
            reg_penalty = jnp.sum((tau_alpha / tau) * c_test)
            return (bt_i + 1, step_size, q_test, c_test, j_test + reg_penalty, j_curr)

        q_test = (q + dq).astype(jnp.float32)
        c_test = jnp.maximum(0.0, q_test - tau_alpha).astype(jnp.float32)
        j_test, _, _ = get_loss_grad_hess(c_test)

        reg_penalty = jnp.sum((tau_alpha / tau) * c_test)
        obj_val = nll + jnp.sum((tau_alpha / tau) * c)

        bt_init = (0, jnp.float32(1.0), q_test, c_test, j_test + reg_penalty, obj_val)
        bt_final = lax.while_loop(bt_cond, bt_body, bt_init)
        _, _, q_final, c_final, _, _ = bt_final

        return (
            step + 1,
            q_final.astype(jnp.float32),
            c_final.astype(jnp.float32),
            jnp.linalg.norm(dq).astype(jnp.float32),
        )

    init_state = (0, q_init.astype(jnp.float32), c_warm.astype(jnp.float32), jnp.float32(1e9))
    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    _, _, c_l1, _ = final_state

    # DEBIASING PHASE
    active_mask = c_l1 > 1e-5
    if force_target:
        active_mask = active_mask.at[0].set(True)

    def debias_cond(state):
        step, _, actual_step_norm = state
        return (step < 100) & (actual_step_norm > 1e-4)

    def debias_body(state):
        step, c, _ = state
        _, grad, hess = get_loss_grad_hess(c)

        H_diag = jnp.diag(hess)
        eta = 1.0 / jnp.maximum(H_diag, 1e-6)

        I = jnp.eye(N_params, dtype=jnp.float32)
        D_mat = jnp.diag(active_mask.astype(jnp.float32))

        F_c = (1.0 - active_mask) * c + active_mask * (eta * grad)
        DG = (I - D_mat) + (eta[:, None] * hess) @ D_mat + 1e-4 * I

        dc = jnp.linalg.solve(DG, -F_c).astype(jnp.float32)

        tau_debias = jnp.where(loss_type == 1, jnp.float32(0.8), jnp.float32(1.0))

        c_new_raw = c + tau_debias * dc * active_mask
        c_new = jnp.maximum(0.0, c_new_raw) * active_mask

        actual_step = c_new - c
        return (step + 1, c_new.astype(jnp.float32), jnp.linalg.norm(actual_step).astype(jnp.float32))

    debias_state = lax.while_loop(
        debias_cond, debias_body, (0, c_l1.astype(jnp.float32), jnp.float32(1e9))
    )
    _, c_final, _ = debias_state

    return c_final.astype(jnp.float32)

class SparseBasisPursuit:
    """
    The Unified SSN Engine. 
    Now features a Universal BIC Auto-Tuner for unsupervised sparsity tuning!
    """
    def __init__(self, alpha=15.0, gamma=2.0, loss="poisson", ref_sigma=1.0, 
                 auto_tune_alpha=False, candidate_alphas=None):
        self.alpha = alpha
        self.gamma = gamma
        self.loss = loss
        self.ref_sigma = ref_sigma
        if loss == "poisson":
            self.loss_code = 1
        elif loss == "huber":
            self.loss_code = 2
        else:
            self.loss_code = 0
        
        self.auto_tune_alpha = auto_tune_alpha
        if candidate_alphas is None:
            self.candidate_alphas = jnp.array([10.0, 15.0, 20.0, 30.0, 50.0])
        else:
            self.candidate_alphas = jnp.array(candidate_alphas)

    def _compute_background(self, data, filter_size):
        raise NotImplementedError("Child class must define morphology dimensionality.")

    def _build_basis_matrix(self, x_grid, params):
        raise NotImplementedError("Child class must define the geometric basis.")

    @partial(jit, static_argnames=['self'])
    def solve_ssn_step(self, data_flat, bg_flat, A_matrix, params_guess, alpha_override=None):
        c_init = params_guess[:, 0]
        sigmas = params_guess[:, -1]
        
        weights = (sigmas / self.ref_sigma) ** self.gamma
        alpha_val = alpha_override if alpha_override is not None else self.alpha
        alpha_vec = alpha_val * weights
        
        return solve_ssn_unified(
            A_matrix, data_flat, bg_flat, alpha_vec, 
            self.loss_code, c_init, max_iter=20, force_target=False
        )

    @partial(jit, static_argnames=['self'])
    def tune_and_solve(self, data_flat, bg_flat, A_matrix, params_guess):
        """
        Sweeps candidate alphas, evaluates the SSN step, and returns 
        the solution that minimizes the BIC, alongside logging metrics.
        """
        if not self.auto_tune_alpha:
            c = self.solve_ssn_step(data_flat, bg_flat, A_matrix, params_guess)
            return c, self.alpha, 0.0, 0.0

        def evaluate_alpha(alpha_val):
            c_sparse = self.solve_ssn_step(data_flat, bg_flat, A_matrix, params_guess, alpha_override=alpha_val)
            k_active = jnp.sum(c_sparse > 1e-9)

            recon = A_matrix @ c_sparse
            recon_total = jnp.maximum(recon + bg_flat, 1e-9)

            n_pix = data_flat.size

            if self.loss_code == 1: # Poisson
                nll = jnp.sum(recon_total - jax.scipy.special.xlogy(data_flat, recon_total))
                term = jax.scipy.special.xlogy(data_flat, data_flat / recon_total) - (data_flat - recon_total)
                dev = 2 * jnp.sum(term)
            elif self.loss_code == 2: # Huber
                delta = 3.0 * jnp.sqrt(jnp.maximum(jnp.median(bg_flat), 1e-3))
                e = recon_total - data_flat
                abs_e = jnp.abs(e)
                nll = jnp.sum(jnp.where(abs_e <= delta, 0.5 * e**2, delta * abs_e - 0.5 * delta**2))
                
                # Effective Deviance for Huber
                W_diag = jnp.where(abs_e <= delta, 1.0, delta / jnp.maximum(abs_e, 1e-6))
                dev = jnp.sum(W_diag * (e**2) / recon_total)
            else: # Gaussian
                diff = recon_total - data_flat
                nll = 0.5 * jnp.sum(diff**2)
                # THE FIX: True Mean Squared Error for pure continuous signals
                dev = jnp.sum(diff**2) / n_pix

            n_params = k_active * params_guess.shape[1]
            dof = jnp.maximum(n_pix - n_params, 1)
            dev_nu = dev / dof
            
            bic = jnp.where(k_active == 0, 1e9, n_params * jnp.log(n_pix) + 2 * nll)
            return bic, c_sparse, dev_nu

        bics, all_c_sparse, devs = vmap(evaluate_alpha)(self.candidate_alphas)
        best_idx = jnp.argmin(bics)
        
        return all_c_sparse[best_idx], self.candidate_alphas[best_idx], bics[best_idx], devs[best_idx]
