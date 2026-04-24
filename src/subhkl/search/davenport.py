import numpy as np
import itertools
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation
import optax

@jit
def compute_theo_angles_jax(r_theo_norm_j):
    theo_dots = jnp.clip(jnp.abs(r_theo_norm_j @ r_theo_norm_j.T), 0.0, 1.0)
    return jnp.rad2deg(jnp.arccos(theo_dots))

@jit
def evaluate_triad_chunk_jax(theo_angles_j, va, vb, req_13, req_23, angle_tol):
    m13 = jnp.abs(theo_angles_j[va, :] - req_13) < angle_tol
    m23 = jnp.abs(theo_angles_j[vb, :] - req_23) < angle_tol
    return m13 & m23

@jit
def evaluate_tetrad_chunk_jax(theo_angles_j, va, vb, vc, req_14, req_24, req_34, angle_tol):
    m14 = jnp.abs(theo_angles_j[va, :] - req_14) < angle_tol
    m24 = jnp.abs(theo_angles_j[vb, :] - req_24) < angle_tol
    m34 = jnp.abs(theo_angles_j[vc, :] - req_34) < angle_tol
    return m14 & m24 & m34

@jit
def jax_davenport_consensus(e_use_nodes, q_sample_obs_norm, r_hyp_nodes_batch, r_theo_rays_norm, angle_tol_cons):
    """Evaluates Consensus using pure Laue Ray Directions."""
    B = jnp.einsum('ax,nay->nxy', e_use_nodes, r_hyp_nodes_batch)
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
    
    # Map theoretical rays to sample frame
    r_lab_batch = jnp.matmul(U_batch, r_theo_rays_norm.T)
    
    # Cosine similarity against normalized empirical rays
    dots = jnp.einsum('ei,nit->net', q_sample_obs_norm, r_lab_batch)
    
    max_dots = jnp.max(jnp.clip(dots, -1.0, 1.0), axis=2) 
    angles = jnp.rad2deg(jnp.arccos(max_dots))
    
    inliers = jnp.sum(angles < angle_tol_cons, axis=1)
    residuals = jnp.sum(jnp.where(angles < angle_tol_cons, angles, 0.0), axis=1) / jnp.maximum(inliers, 1)
    
    return U_batch, inliers, residuals

def align_virtual_nodes(
    e_nodes,
    q_sample_obs_norm,
    B_mat,
    max_hkl_hyp=2,
    max_hkl_cons=5,
    angle_tol_hyp=1.5,
    angle_tol_cons=1.0,
    hyp_batch_size=10000,
    eval_batch_size=500
):
    N_emp_nodes = len(e_nodes)
    if N_emp_nodes < 4:
        raise ValueError("Need at least 4 empirical virtual nodes to lock orientation.")

    N_emp_peaks = len(q_sample_obs_norm)

    print(f"  > Generating Theoretical Nodes for Hypotheses (max_hkl={max_hkl_hyp})...")
    h_vals = np.arange(-max_hkl_hyp, max_hkl_hyp + 1)
    h, k, l = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
    hkl = np.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)
    mask_hkl = ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))
    theo_hkl = hkl[:, mask_hkl].astype(np.float32).T

    r_theo_nodes = (B_mat @ theo_hkl.T).T
    r_nodes_norm = r_theo_nodes / np.linalg.norm(r_theo_nodes, axis=1, keepdims=True)

    print(f"  > Generating Unique Laue Rays for Consensus (max_hkl={max_hkl_cons})...")
    hc_vals = np.arange(-max_hkl_cons, max_hkl_cons + 1)
    hc, kc, lc = np.meshgrid(hc_vals, hc_vals, hc_vals, indexing="ij")
    hkl_c = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
    mask_hkl_c = ~((hkl_c[0] == 0) & (hkl_c[1] == 0) & (hkl_c[2] == 0))
    theo_hkl_c = hkl_c[:, mask_hkl_c].astype(np.float32).T

    r_theo_rays = (B_mat @ theo_hkl_c.T).T
    r_rays_norm = r_theo_rays / np.linalg.norm(r_theo_rays, axis=1, keepdims=True)
    _, unique_idx = np.unique(np.round(r_rays_norm, 4), axis=0, return_index=True)
    r_unique_rays_norm = r_rays_norm[unique_idx]

    print(f"  > Compiled {len(r_nodes_norm)} Hash Nodes and {len(r_unique_rays_norm)} Unique Physical Rays. Moving to GPU...")

    r_nodes_norm_j = jnp.array(r_nodes_norm, dtype=jnp.float32)
    theo_angles_j = compute_theo_angles_jax(r_nodes_norm_j)
    theo_angles = np.array(theo_angles_j)

    q_sample_obs_norm_j = jnp.array(q_sample_obs_norm, dtype=jnp.float32)
    r_unique_rays_norm_j = jnp.array(r_unique_rays_norm, dtype=jnp.float32)

    best_U = None
    best_inliers = -1
    best_residual = np.inf

    print(f"  > Executing Primitive Tetrad Search across Combinations (tol={angle_tol_hyp} deg)...")

    # --- NEW: COMBINATORIAL HUB SEARCH ---
    for comb in itertools.combinations(range(N_emp_nodes), 4):
        e_use = e_nodes[list(comb)]
        e_use_nodes_j = jnp.array(e_use, dtype=jnp.float32)

        emp_dots = np.clip(np.abs(e_use @ e_use.T), 0.0, 1.0)
        emp_angles = np.rad2deg(np.arccos(emp_dots))

        req_12, req_13, req_23 = emp_angles[0, 1], emp_angles[0, 2], emp_angles[1, 2]
        req_14, req_24, req_34 = emp_angles[0, 3], emp_angles[1, 3], emp_angles[2, 3]

        valid_a, valid_b = np.where(np.abs(theo_angles - req_12) < angle_tol_hyp)
        valid_mask = valid_a != valid_b
        valid_a, valid_b = valid_a[valid_mask], valid_b[valid_mask]

        a_cand_triad, b_cand_triad, c_cand_triad = [], [], []

        for i in range(0, len(valid_a), hyp_batch_size):
            va_chunk = valid_a[i:i+hyp_batch_size]
            vb_chunk = valid_b[i:i+hyp_batch_size]
            actual_size = len(va_chunk)
            if actual_size < hyp_batch_size:
                va_chunk = np.pad(va_chunk, (0, hyp_batch_size - actual_size), constant_values=0)
                vb_chunk = np.pad(vb_chunk, (0, hyp_batch_size - actual_size), constant_values=0)

            mask_chunk = evaluate_triad_chunk_jax(theo_angles_j, jnp.array(va_chunk), jnp.array(vb_chunk), req_13, req_23, angle_tol_hyp)
            p_idx_j, c_idx_j = jnp.where(mask_chunk[:actual_size])
            p_idx, c_idx = np.array(p_idx_j), np.array(c_idx_j)
            if len(p_idx) > 0:
                global_p_idx = p_idx + i
                a_cand_triad.append(valid_a[global_p_idx])
                b_cand_triad.append(valid_b[global_p_idx])
                c_cand_triad.append(c_idx)

        if not a_cand_triad:
            continue # Try next combination of 4 empirical nodes

        a_cand_triad = np.concatenate(a_cand_triad)
        b_cand_triad = np.concatenate(b_cand_triad)
        c_cand_triad = np.concatenate(c_cand_triad)

        triplet_mask = (c_cand_triad != a_cand_triad) & (c_cand_triad != b_cand_triad)
        a_cand_triad, b_cand_triad, c_cand_triad = a_cand_triad[triplet_mask], b_cand_triad[triplet_mask], c_cand_triad[triplet_mask]

        a_cand_list, b_cand_list, c_cand_list, d_cand_list = [], [], [], []

        for i in range(0, len(a_cand_triad), hyp_batch_size):
            va_chunk = a_cand_triad[i:i+hyp_batch_size]
            vb_chunk = b_cand_triad[i:i+hyp_batch_size]
            vc_chunk = c_cand_triad[i:i+hyp_batch_size]
            actual_size = len(va_chunk)
            if actual_size < hyp_batch_size:
                va_chunk = np.pad(va_chunk, (0, hyp_batch_size - actual_size), constant_values=0)
                vb_chunk = np.pad(vb_chunk, (0, hyp_batch_size - actual_size), constant_values=0)
                vc_chunk = np.pad(vc_chunk, (0, hyp_batch_size - actual_size), constant_values=0)

            mask_chunk = evaluate_tetrad_chunk_jax(theo_angles_j, jnp.array(va_chunk), jnp.array(vb_chunk), jnp.array(vc_chunk), req_14, req_24, req_34, angle_tol_hyp)
            p_idx_j, d_idx_j = jnp.where(mask_chunk[:actual_size])
            p_idx, d_idx = np.array(p_idx_j), np.array(d_idx_j)
            if len(p_idx) > 0:
                global_p_idx = p_idx + i
                a_cand_list.append(a_cand_triad[global_p_idx])
                b_cand_list.append(b_cand_triad[global_p_idx])
                c_cand_list.append(c_cand_triad[global_p_idx])
                d_cand_list.append(d_idx)

        if not a_cand_list:
            continue # Try next combination

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

        r_cand = r_nodes_norm[np.column_stack([a_cand, b_cand, c_cand, d_cand])]
        signs = np.array(list(itertools.product([1, -1], repeat=4)))

        W_batch = (r_cand[:, None, :, :] * signs[None, :, :, None]).reshape(-1, 4, 3)
        N_hyp = W_batch.shape[0]

        for i in range(0, N_hyp, eval_batch_size):
            W_chunk = W_batch[i:i+eval_batch_size]
            actual_size = len(W_chunk)

            if actual_size < eval_batch_size:
                W_chunk = np.pad(W_chunk, ((0, eval_batch_size - actual_size), (0,0), (0,0)), constant_values=0)

            U_out, inliers_out, residuals_out = jax_davenport_consensus(
                e_use_nodes_j, q_sample_obs_norm_j, jnp.array(W_chunk, dtype=jnp.float32), r_unique_rays_norm_j, angle_tol_cons
            )

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

    if best_inliers < 10:
        raise ValueError(f"Consensus failed completely. Checked all empirical combinations, highest match was {best_inliers} peaks.")

    print(f"  > Consensus Achieved! U-Matrix directionally maps {best_inliers}/{N_emp_peaks} raw peaks to Laue rays (Mean Angle Error: {best_residual:.3f} deg)")

    return best_U

# -------------------------------------------------------------
# CONTINUOUS VORONOI OPTIMIZER (HARD ASSIGNMENT)
# -------------------------------------------------------------

@jit
def quaternion_to_rotation_matrix(q):
    """Converts a quaternion [w, x, y, z] to a 3x3 Rotation Matrix."""
    q = q / jnp.linalg.norm(q) 
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

@jit
def voronoi_ray_loss(q, q_obs_norm, r_unique_rays_norm):
    """
    Hard Voronoi Penalty.
    Calculates the exact distance to the single closest theoretical ray.
    """
    U = quaternion_to_rotation_matrix(q)
    
    # Map theoretical rays to sample frame
    r_lab = jnp.matmul(r_unique_rays_norm, U.T)
    
    # Calculate all pairwise cosine similarities
    sim_matrix = jnp.matmul(q_obs_norm, r_lab.T)
    
    # Hard assignment: Find the absolute maximum dot product for each peak
    max_sims = jnp.max(sim_matrix, axis=1)
    
    # We want to maximize the similarity (drive it to 1.0)
    return -jnp.sum(max_sims)

def optimize_orientation_gradient_descent(q_init, q_obs_norm, r_unique_rays_norm, steps=300):
    """
    Polishes the Davenport seed using micro-steps and tracks the absolute lowest error state.
    """
    # Drop the learning rate by 10x! We are doing microscopic fine-tuning now.
    optimizer = optax.adam(learning_rate=0.0005)
    opt_state = optimizer.init(q_init)

    loss_fn = jax.jit(jax.value_and_grad(voronoi_ray_loss))

    q_current = q_init

    # --- BEST STATE TRACKERS ---
    best_q = q_init
    best_loss = jnp.inf

    print("  > Launching Hard Voronoi Gradient Descent (Micro-stepping)...")

    for step in range(steps):
        loss_val, grads = loss_fn(q_current, q_obs_norm, r_unique_rays_norm)

        # Save the quaternion if this is the deepest point in the crater
        if loss_val < best_loss:
            best_loss = loss_val
            best_q = q_current

        updates, opt_state = optimizer.update(grads, opt_state)
        q_current = optax.apply_updates(q_current, updates)

        if step % 100 == 0:
            print(f"    Step {step:03d} | Voronoi Loss: {loss_val:.3f}")

    # Always return the absolute best matrix found during the loop
    q_final = best_q / jnp.linalg.norm(best_q)
    U_optimal = quaternion_to_rotation_matrix(q_final)

    return U_optimal
