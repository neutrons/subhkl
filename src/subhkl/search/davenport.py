import numpy as np
import itertools
import jax
import jax.numpy as jnp
from jax import jit

@jit
def evaluate_pca_consensus(U_candidates, q_sample_obs, r_theo_peaks, angle_tol_cons):
    """Evaluates the 4 PCA candidate matrices against the raw peaks on the GPU."""
    # (4, 3, 3) @ (3, N_theo) -> (4, 3, N_theo)
    r_lab_batch = jnp.matmul(U_candidates, r_theo_peaks.T)
    
    # (N_emp, 3) @ (4, 3, N_theo) -> (4, N_emp, N_theo)
    dots = jnp.einsum('ei,nit->net', q_sample_obs, r_lab_batch)
    
    max_dots = jnp.max(jnp.clip(jnp.abs(dots), 0.0, 1.0), axis=2)
    angles = jnp.rad2deg(jnp.arccos(max_dots))
    
    inliers = jnp.sum(angles < angle_tol_cons, axis=1)
    residuals = jnp.sum(jnp.where(angles < angle_tol_cons, angles, 0.0), axis=1) / jnp.maximum(inliers, 1)
    
    return inliers, residuals

def align_inertial_pca(
    e_nodes, 
    q_sample_obs_norm, 
    B_mat, 
    max_hkl_hyp=2, 
    max_hkl_cons=5, 
    angle_tol_cons=0.4
):
    N_emp_peaks = len(q_sample_obs_norm)

    # --- 1. Theoretical Hull Dictionary ---
    print(f"  > Generating Theoretical Nodes for PCA (max_hkl={max_hkl_hyp})...")
    h_vals = np.arange(-max_hkl_hyp, max_hkl_hyp + 1)
    h, k, l = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
    hkl = np.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)
    mask_hkl = ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))
    theo_hkl = hkl[:, mask_hkl].astype(np.float32).T 

    r_theo_nodes = (B_mat @ theo_hkl.T).T
    r_nodes_norm = r_theo_nodes / np.linalg.norm(r_theo_nodes, axis=1, keepdims=True)

    # --- 2. Consensus Dictionary ---
    print(f"  > Generating Theoretical Peaks for Consensus (max_hkl={max_hkl_cons})...")
    hc_vals = np.arange(-max_hkl_cons, max_hkl_cons + 1)
    hc, kc, lc = np.meshgrid(hc_vals, hc_vals, hc_vals, indexing="ij")
    hkl_c = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
    mask_hkl_c = ~((hkl_c[0] == 0) & (hkl_c[1] == 0) & (hkl_c[2] == 0))
    theo_hkl_c = hkl_c[:, mask_hkl_c].astype(np.float32).T 

    r_theo_peaks = (B_mat @ theo_hkl_c.T).T
    r_peaks_norm = r_theo_peaks / np.linalg.norm(r_theo_peaks, axis=1, keepdims=True)

    # --- 3. Compute Principal Axes of Inertia (Covariance) ---
    print("  > Computing Inertial Tensors...")
    
    # Theoretical Covariance
    C_theo = r_nodes_norm.T @ r_nodes_norm
    evals_T, evecs_T = np.linalg.eigh(C_theo)

    # Empirical Covariance (Using the Virtual Hubs)
    C_emp = e_nodes.T @ e_nodes
    evals_E, evecs_E = np.linalg.eigh(C_emp)

    # --- 4. Resolve the Eigenvector Sign Ambiguity ---
    print("  > Extracting Rotation Matrices...")
    # Eigenvectors point along an axis, but have no defined 'head' or 'tail'.
    # We test all 8 sign permutations of the theoretical axes.
    signs = np.array(list(itertools.product([1, -1], repeat=3)))
    
    U_candidates = []
    for s in signs:
        V_T_flipped = evecs_T * s
        # Calculate the rotation that maps Theoretical Axes to Empirical Axes
        U = evecs_E @ V_T_flipped.T
        
        # We only keep proper rotations (Determinant == 1, no reflections)
        if np.linalg.det(U) > 0:
            U_candidates.append(U)
            
    U_candidates = np.array(U_candidates)
    print(f"  > Generated {len(U_candidates)} valid Principal Alignments. Evaluating Consensus...")

    # --- 5. Validate against Raw Peaks ---
    q_sample_j = jnp.array(q_sample_obs_norm, dtype=jnp.float32)
    r_peaks_j = jnp.array(r_peaks_norm, dtype=jnp.float32)
    U_cand_j = jnp.array(U_candidates, dtype=jnp.float32)

    inliers, residuals = evaluate_pca_consensus(
        U_cand_j, q_sample_j, r_peaks_j, angle_tol_cons
    )
    
    inliers = np.array(inliers)
    residuals = np.array(residuals)

    score = inliers - (residuals / 1000.0)
    best_idx = np.argmax(score)

    best_inliers = inliers[best_idx]
    best_residual = residuals[best_idx]
    best_U = U_candidates[best_idx]

    print(f"  > Inertial Alignment Complete! U-Matrix explains {best_inliers}/{N_emp_peaks} raw peaks (Mean Error: {best_residual:.3f} deg)")
    
    return best_U
