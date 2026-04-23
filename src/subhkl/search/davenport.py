import numpy as np
import itertools
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation

@jit
def jax_davenport_consensus(e_use, e_lab_obs, r_hyp_batch, r_theo_norm, angle_tol):
    """Batched Wahba's Problem and RANSAC Evaluator executing strictly on GPU"""
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


def align_empirical_zones(e_lab_obs, B_mat, max_uvw=10, angle_tol=1.5):
    N_emp = len(e_lab_obs)
    N_use = min(4, N_emp)
    
    if N_use < 4:
        raise ValueError("Need at least 4 empirical zone axes to lock the Tetrad orientation.")
    
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

    # Instantly compute the angle matrix natively in CPU RAM
    theo_dots = np.clip(np.abs(r_theo_norm @ r_theo_norm.T), 0.0, 1.0)
    theo_angles = np.rad2deg(np.arccos(theo_dots))

    emp_dots = np.clip(np.abs(e_use @ e_use.T), 0.0, 1.0)
    emp_angles = np.rad2deg(np.arccos(emp_dots))

    req_12, req_13, req_23 = emp_angles[0, 1], emp_angles[0, 2], emp_angles[1, 2]
    req_14, req_24, req_34 = emp_angles[0, 3], emp_angles[1, 3], emp_angles[2, 3]

    print(f"  > Executing Pre-Computed Adjacency Search (tol={angle_tol} deg)...")
    
    # Pre-Compute 85MB Boolean Adjacency Masks (Pure NumPy C-Speed)
    M12 = np.abs(theo_angles - req_12) < angle_tol
    M13 = np.abs(theo_angles - req_13) < angle_tol
    M23 = np.abs(theo_angles - req_23) < angle_tol

    valid_a, valid_b = np.where(M12)
    valid_mask = valid_a != valid_b
    valid_a, valid_b = valid_a[valid_mask], valid_b[valid_mask]

    a_cand_triad, b_cand_triad, c_cand_triad = [], [], []
    chunk_size = 25000 # Safely slices 230MB chunks in RAM
    
    for i in range(0, len(valid_a), chunk_size):
        va = valid_a[i:i+chunk_size]
        vb = valid_b[i:i+chunk_size]
        
        # Slicing pre-computed masks natively avoids all compilation overhead
        valid_c_mask = M13[va, :] & M23[vb, :]
        p_idx, c_idx = np.where(valid_c_mask)
        
        if len(p_idx) > 0:
            global_p_idx = p_idx + i
            a_cand_triad.append(valid_a[global_p_idx])
            b_cand_triad.append(valid_b[global_p_idx])
            c_cand_triad.append(c_idx)

    if not a_cand_triad:
        raise ValueError(f"No theoretical triads match within {angle_tol} deg.")

    a_cand_triad = np.concatenate(a_cand_triad)
    b_cand_triad = np.concatenate(b_cand_triad)
    c_cand_triad = np.concatenate(c_cand_triad)

    triplet_mask = (c_cand_triad != a_cand_triad) & (c_cand_triad != b_cand_triad)
    a_cand_triad = a_cand_triad[triplet_mask]
    b_cand_triad = b_cand_triad[triplet_mask]
    c_cand_triad = c_cand_triad[triplet_mask]

    # Pre-Compute Tetrad Masks
    M14 = np.abs(theo_angles - req_14) < angle_tol
    M24 = np.abs(theo_angles - req_24) < angle_tol
    M34 = np.abs(theo_angles - req_34) < angle_tol
    
    a_cand_list, b_cand_list, c_cand_list, d_cand_list = [], [], [], []
    
    for i in range(0, len(a_cand_triad), chunk_size):
        va = a_cand_triad[i:i+chunk_size]
        vb = b_cand_triad[i:i+chunk_size]
        vc = c_cand_triad[i:i+chunk_size]
        
        valid_d_mask = M14[va, :] & M24[vb, :] & M34[vc, :]
        p_idx, d_idx = np.where(valid_d_mask)
        
        if len(p_idx) > 0:
            global_p_idx = p_idx + i
            a_cand_list.append(a_cand_triad[global_p_idx])
            b_cand_list.append(b_cand_triad[global_p_idx])
            c_cand_list.append(c_cand_triad[global_p_idx])
            d_cand_list.append(d_idx)

    if not a_cand_list:
        raise ValueError(f"No theoretical tetrads match the 6 empirical angles within {angle_tol} deg.")

    a_cand = np.concatenate(a_cand_list)
    b_cand = np.concatenate(b_cand_list)
    c_cand = np.concatenate(c_cand_list)
    d_cand = np.concatenate(d_cand_list)

    tetrad_mask = (d_cand != a_cand) & (d_cand != b_cand) & (d_cand != c_cand)
    a_cand, b_cand, c_cand, d_cand = a_cand[tetrad_mask], b_cand[tetrad_mask], c_cand[tetrad_mask], d_cand[tetrad_mask]

    err12 = np.abs(theo_angles[a_cand, b_cand] - req_12)
    err13 = np.abs(theo_angles[a_cand, c_cand] - req_13)
    err23 = np.abs(theo_angles[b_cand, c_cand] - req_23)
    err14 = np.abs(theo_angles[a_cand, d_cand] - req_14)
    err24 = np.abs(theo_angles[b_cand, d_cand] - req_24)
    err34 = np.abs(theo_angles[c_cand, d_cand] - req_34)
    
    max_errs = np.maximum.reduce([err12, err13, err23, err14, err24, err34])

    max_candidates = 1000
    sort_idx = np.argsort(max_errs)[:max_candidates]
    a_cand, b_cand, c_cand, d_cand = a_cand[sort_idx], b_cand[sort_idx], c_cand[sort_idx], d_cand[sort_idx]
    
    r_cand = r_theo_norm[np.column_stack([a_cand, b_cand, c_cand, d_cand])]
    signs = np.array(list(itertools.product([1, -1], repeat=4)))
    
    W_batch = (r_cand[:, None, :, :] * signs[None, :, :, None]).reshape(-1, 4, 3)
    N_hyp = W_batch.shape[0]
    
    print(f"  > Found {len(a_cand)} pristine geometric tetrads. Dispatching {N_hyp} U-Matrices to GPU...")

    # Upload static geometry to GPU once
    e_use_j = jnp.array(e_use, dtype=jnp.float32)
    e_lab_obs_j = jnp.array(e_lab_obs, dtype=jnp.float32)
    r_theo_norm_j = jnp.array(r_theo_norm, dtype=jnp.float32)

    best_U = None
    best_inliers = -1
    best_residual = np.inf
    
    eval_batch_size = 8000
    
    for i in range(0, N_hyp, eval_batch_size):
        W_chunk = W_batch[i:i+eval_batch_size]
        actual_size = len(W_chunk)
        
        # Static shape padding guarantees XLA compiles exactly ONCE.
        if actual_size < eval_batch_size:
            W_chunk = np.pad(W_chunk, ((0, eval_batch_size - actual_size), (0,0), (0,0)), constant_values=0)
            
        U_out, inliers_out, residuals_out = jax_davenport_consensus(
            e_use_j, e_lab_obs_j, jnp.array(W_chunk, dtype=jnp.float32), r_theo_norm_j, angle_tol
        )
        
        # Strip padding and evaluate
        inliers_out = np.array(inliers_out[:actual_size])
        residuals_out = np.array(residuals_out[:actual_size])
        U_out = np.array(U_out[:actual_size])
        
        score = inliers_out - (residuals_out / 1000.0) 
        best_in_chunk = np.argmax(score)
        
        if inliers_out[best_in_chunk] > best_inliers or (
            inliers_out[best_in_chunk] == best_inliers and 
            residuals_out[best_in_chunk] < best_residual
        ):
            best_inliers = inliers_out[best_in_chunk]
            best_residual = residuals_out[best_in_chunk]
            best_U = U_out[best_in_chunk]

    if best_inliers < 4:
        raise ValueError(f"Consensus failed. Best U-matrix only explained {best_inliers} axes.")

    print(f"  > Consensus Achieved! U-Matrix explains {best_inliers}/{N_emp} axes (Mean Error: {best_residual:.3f} deg)")
    
    return best_U
