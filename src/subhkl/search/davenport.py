import numpy as np
import itertools
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation

@jit
def compute_theo_angles_jax(r_theo_norm_j):
    """Calculates the 9000x9000 pairwise angle matrix instantly on GPU."""
    theo_dots = jnp.clip(jnp.abs(r_theo_norm_j @ r_theo_norm_j.T), 0.0, 1.0)
    return jnp.rad2deg(jnp.arccos(theo_dots))

@jit
def evaluate_triad_chunk_jax(theo_angles_j, va, vb, req_13, req_23, angle_tol):
    """Evaluates millions of geometric triad hypotheses in batched GPU chunks."""
    m13 = jnp.abs(theo_angles_j[va, :] - req_13) < angle_tol
    m23 = jnp.abs(theo_angles_j[vb, :] - req_23) < angle_tol
    return m13 & m23

@jit
def jax_davenport_consensus(e_use, e_lab_obs, r_hyp_batch, r_theo_norm, angle_tol):
    """Pure JAX Batched Davenport Solver + Global Consensus Evaluator."""
    B = jnp.einsum('ax,nay->nxy', e_use, r_hyp_batch)
    S = B + jnp.transpose(B, (0, 2, 1))
    sigma = jnp.trace(B, axis1=1, axis2=2)
    
    Z = jnp.column_stack([
        B[:, 1, 2] - B[:, 2, 1], 
        B[:, 2, 0] - B[:, 0, 2], 
        B[:, 0, 1] - B[:, 1, 0]
    ])
    
    K = jnp.zeros((B.shape[0], 4, 4))
    K = K.at[:, :3, :3].set(S - sigma[:, None, None] * jnp.eye(3))
    K = K.at[:, :3, 3].set(Z)
    K = K.at[:, 3, :3].set(Z)
    K = K.at[:, 3, 3].set(sigma)
    
    evals, evecs = jnp.linalg.eigh(K)
    q = evecs[:, :, -1] 
    
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    U00 = 1.0 - 2.0 * (y**2 + z**2)
    U01 = 2.0 * (x*y - z*w)
    U02 = 2.0 * (x*z + y*w)
    
    U10 = 2.0 * (x*y + z*w)
    U11 = 1.0 - 2.0 * (x**2 + z**2)
    U12 = 2.0 * (y*z - x*w)
    
    U20 = 2.0 * (x*z - y*w)
    U21 = 2.0 * (y*z + x*w)
    U22 = 1.0 - 2.0 * (x**2 + y**2)
    
    U_batch = jnp.stack([
        jnp.stack([U00, U01, U02], axis=1),
        jnp.stack([U10, U11, U12], axis=1),
        jnp.stack([U20, U21, U22], axis=1)
    ], axis=1)
    
    r_lab_batch = jnp.matmul(U_batch, r_theo_norm.T)
    dots = jnp.einsum('ei,nit->net', e_lab_obs, r_lab_batch)
    
    max_dots = jnp.max(jnp.clip(jnp.abs(dots), 0.0, 1.0), axis=2)
    angles = jnp.rad2deg(jnp.arccos(max_dots))
    
    inliers = jnp.sum(angles < angle_tol, axis=1)
    residuals = jnp.sum(jnp.where(angles < angle_tol, angles, 0.0), axis=1) / jnp.maximum(inliers, 1)
    
    return U_batch, inliers, residuals


def align_empirical_zones(
    e_lab_obs, 
    B_mat, 
    max_uvw=10, 
    angle_tol=1.5, 
    hyp_batch_size=50000, 
    eval_batch_size=5000
):
    N_emp = len(e_lab_obs)
    N_use = min(3, N_emp)
    
    if N_use < 3:
        raise ValueError("Need at least 3 empirical zone axes to lock 3D orientation.")
    
    e_use = e_lab_obs[:N_use]

    print(f"  > Generating Theoretical Dictionary (max_uvw={max_uvw})...")
    u_vals = np.arange(-max_uvw, max_uvw + 1)
    u, v, w = np.meshgrid(u_vals, u_vals, u_vals, indexing="ij")
    zones = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=0)
    mask = ~((zones[0] == 0) & (zones[1] == 0) & (zones[2] == 0))
    theo_zones = zones[:, mask].astype(np.float32).T 

    A_mat = np.linalg.inv(B_mat).T
    r_theo = (A_mat @ theo_zones.T).T
    r_norms = np.linalg.norm(r_theo, axis=1, keepdims=True)
    r_theo_norm = r_theo / r_norms
    N_theo = len(r_theo_norm)

    print(f"  > Compiled {N_theo} theoretical axes. Moving to GPU...")

    # Instantly compute the angle matrix on GPU
    r_theo_norm_j = jnp.array(r_theo_norm, dtype=jnp.float32)
    theo_angles_j = compute_theo_angles_jax(r_theo_norm_j)
    theo_angles = np.array(theo_angles_j) # Pull back to CPU just for the baseline np.where

    emp_dots = np.clip(np.abs(e_use @ e_use.T), 0.0, 1.0)
    emp_angles = np.rad2deg(np.arccos(emp_dots))

    req_12, req_13, req_23 = emp_angles[0, 1], emp_angles[0, 2], emp_angles[1, 2]

    print(f"  > Executing JAX-Accelerated Hypothesis Generation (tol={angle_tol} deg)...")

    # Baseline extraction (CPU is fine for a single 1D filter)
    valid_a, valid_b = np.where(np.abs(theo_angles - req_12) < angle_tol)
    valid_mask = valid_a != valid_b
    valid_a, valid_b = valid_a[valid_mask], valid_b[valid_mask]

    a_cand_list, b_cand_list, c_cand_list = [], [], []
    
    # ---------------------------------------------------------
    # HYPOTHESIS BATCHING
    # Pushes the 3D matching boolean logic entirely to the GPU
    # ---------------------------------------------------------
    for i in range(0, len(valid_a), hyp_batch_size):
        va_chunk = jnp.array(valid_a[i:i+hyp_batch_size])
        vb_chunk = jnp.array(valid_b[i:i+hyp_batch_size])
        
        # JAX computes the massive boolean array
        mask_chunk = evaluate_triad_chunk_jax(theo_angles_j, va_chunk, vb_chunk, req_13, req_23, angle_tol)
        
        # Pull the tiny boolean result back to CPU to extract dynamic indices
        mask_chunk_cpu = np.array(mask_chunk)
        p_idx, c_idx = np.where(mask_chunk_cpu)
        
        if len(p_idx) > 0:
            global_p_idx = p_idx + i
            a_cand_list.append(valid_a[global_p_idx])
            b_cand_list.append(valid_b[global_p_idx])
            c_cand_list.append(c_idx)

    if not a_cand_list:
        raise ValueError(f"No theoretical triads match the primary empirical angles within {angle_tol} deg.")

    a_cand = np.concatenate(a_cand_list)
    b_cand = np.concatenate(b_cand_list)
    c_cand = np.concatenate(c_cand_list)

    # Filter degenerates
    triplet_mask = (c_cand != a_cand) & (c_cand != b_cand)
    a_cand, b_cand, c_cand = a_cand[triplet_mask], b_cand[triplet_mask], c_cand[triplet_mask]

    if len(a_cand) == 0:
        raise ValueError(f"No theoretical triads match the primary empirical angles within {angle_tol} deg.")

    # Sort by tightness and take top N
    err12 = np.abs(theo_angles[a_cand, b_cand] - req_12)
    err13 = np.abs(theo_angles[a_cand, c_cand] - req_13)
    err23 = np.abs(theo_angles[b_cand, c_cand] - req_23)
    max_errs = np.maximum(err12, np.maximum(err13, err23))

    max_candidates = 2000
    sort_idx = np.argsort(max_errs)[:max_candidates]
    a_cand, b_cand, c_cand = a_cand[sort_idx], b_cand[sort_idx], c_cand[sort_idx]
    
    r_cand = r_theo_norm[np.column_stack([a_cand, b_cand, c_cand])]
    signs = np.array(list(itertools.product([1, -1], repeat=3)))
    
    W_batch = (r_cand[:, None, :, :] * signs[None, :, :, None]).reshape(-1, 3, 3)
    N_hyp = W_batch.shape[0]
    
    print(f"  > Found {len(a_cand)} geometric triads. Dispatching {N_hyp} U-Matrices to GPU...")

    e_use_j = jnp.array(e_use, dtype=jnp.float32)
    e_lab_obs_j = jnp.array(e_lab_obs, dtype=jnp.float32)

    best_U = None
    best_inliers = -1
    best_residual = np.inf
    
    # ---------------------------------------------------------
    # EVALUATION BATCHING
    # Pushes the Davenport eigenvalue solver and global consensus to the GPU
    # ---------------------------------------------------------
    for i in range(0, N_hyp, eval_batch_size):
        W_chunk = jnp.array(W_batch[i:i+eval_batch_size], dtype=jnp.float32)
        
        U_out, inliers_out, residuals_out = jax_davenport_consensus(
            e_use_j, e_lab_obs_j, W_chunk, r_theo_norm_j, angle_tol
        )
        
        inliers_out = np.array(inliers_out)
        residuals_out = np.array(residuals_out)
        U_out = np.array(U_out)
        
        score = inliers_out - (residuals_out / 1000.0) 
        best_in_chunk = np.argmax(score)
        
        if inliers_out[best_in_chunk] > best_inliers or (
            inliers_out[best_in_chunk] == best_inliers and 
            residuals_out[best_in_chunk] < best_residual
        ):
            best_inliers = inliers_out[best_in_chunk]
            best_residual = residuals_out[best_in_chunk]
            best_U = U_out[best_in_chunk]

    if best_inliers < 3:
        raise ValueError(f"Consensus failed. Best U-matrix only explained {best_inliers} axes.")

    print(f"  > Consensus Achieved! U-Matrix explains {best_inliers}/{N_emp} axes (Mean Error: {best_residual:.3f} deg)")
    
    return best_U
