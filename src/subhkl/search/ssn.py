"""Semi-Smooth Newton (SSN) solver for L1-regularized sparse recovery.

The dense shared engine: the matrix-free integrator solves its per-image
amplitude systems here (K x K, direct), and ``calibrated_admission_z``
is the single construction site for L1 admission thresholds -- the
finder's false-alarm calibration parameterized by the caller's own test
count.  The convolution-structured (H*W-scale) solver lives with the
finder in ``matrix_free._solve_ssn_cg_global``.
"""

import jax.numpy as jnp
from jax import lax, jit
from functools import partial

from scipy.special import ndtri


def calibrated_admission_z(n_tests, fp_target):
    """The z threshold whose expected false-admission count is fp_target.

    The L1 penalty rate in this engine is lambda_i = alpha_i / SE_i --
    an atom activates when its whitened residual correlation exceeds
    alpha_i standard errors (the KKT condition IS the admission test).
    Under the null every test is a one-sided Gaussian tail, so the
    calibrated threshold over n_tests candidates is

        z = Phi^-1(1 - fp_target / n_tests).

    This is the same multiple-testing logic as the finder's
    effective_alpha, with the caller's own test count: the finder
    searches ~H*W*K candidate positions per image; the integrator
    tests exactly its predicted reflections.  fp_target >= n_tests
    returns 0 (admit everything).
    """
    q = fp_target / max(float(n_tests), 1.0)
    if q >= 1.0:
        return 0.0
    return float(ndtri(1.0 - q))


@partial(jit, static_argnames=["max_iter", "loss_type", "force_target", "per_atom_var"])
def solve_ssn_unified(
    A,
    y,
    bg_flat,
    alpha_vec,
    loss_type,
    c_warm,
    max_iter=20,
    force_target=False,
    active_override=None,
    per_atom_var=False,
):
    N_peaks = A.shape[1]
    N_params = N_peaks
    q_init = c_warm.astype(jnp.float32)

    bg_med = jnp.maximum(jnp.median(bg_flat), 1e-3).astype(jnp.float32)

    def get_loss_grad_hess(c):
        u = A @ c + bg_flat

        if loss_type == 1:  # Poisson
            u_safe = jnp.maximum(u, 1e-6)
            nll = jnp.sum(u_safe - y * jnp.log(u_safe))
            grad = A.T @ (1.0 - y / u_safe)
            W_diag = 1.0 / jnp.maximum(u_safe, 1e-3)
            hess = A.T @ (W_diag[:, None] * A)

        elif loss_type == 2:  # Huber
            # Threshold (delta) is 3 standard deviations of the background
            delta = 3.0 * jnp.sqrt(bg_med)
            e = u - y
            abs_e = jnp.abs(e)
            is_inlier = abs_e <= delta

            # Loss: 0.5 * e^2 for inliers, delta * |e| - 0.5 * delta^2 for outliers
            nll = jnp.sum(
                jnp.where(is_inlier, 0.5 * e**2, delta * abs_e - 0.5 * delta**2)
            )

            # IRLS Weights for gradient and Hessian
            W_diag = jnp.where(is_inlier, 1.0, delta / jnp.maximum(abs_e, 1e-6))

            grad = A.T @ (W_diag * e)
            hess = A.T @ (W_diag[:, None] * A)

        else:  # Gaussian (MSE)
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

        if per_atom_var:
            # Each atom's own curvature: the global max-diag tau
            # understates the standard error of every atom but the
            # sharpest, so their thresholds come out harsher than the
            # requested z (the CG global solver already thresholds
            # per-coefficient; this is the dense-path equivalent).
            var_c = 1.0 / jnp.maximum(jnp.diag(hess), 1e-6)
        else:
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

    init_state = (
        0,
        q_init.astype(jnp.float32),
        c_warm.astype(jnp.float32),
        jnp.float32(1e9),
    )
    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    _, _, c_l1, _ = final_state

    # DEBIASING PHASE
    active_mask = c_l1 > 1e-5
    if force_target:
        active_mask = active_mask.at[0].set(True)
    if active_override is not None:
        # A caller whose dictionary IS the support (the amplitude-only
        # integrator: every atom is a known reflection, nothing is a
        # candidate) skips support selection entirely -- the debiased
        # Newton phase with its nonnegative projection then computes the
        # constrained MLE for every listed atom, including the ones the
        # L1 phase left at zero.
        active_mask = active_override

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
        return (
            step + 1,
            c_new.astype(jnp.float32),
            jnp.linalg.norm(actual_step).astype(jnp.float32),
        )

    debias_state = lax.while_loop(
        debias_cond, debias_body, (0, c_l1.astype(jnp.float32), jnp.float32(1e9))
    )
    _, c_final, _ = debias_state

    return c_final.astype(jnp.float32)
