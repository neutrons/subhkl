import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation
import optax

def align_virtual_nodes(
    e_nodes, 
    q_sample_obs_norm,   # <--- BRINGING RAW PEAKS BACK FOR SCORING
    r_eval_rays_norm,    # <--- Massive evaluation dictionary
    B_mat, 
    max_hkl_hyp=2, 
    angle_tol_hyp=1.5,
    angle_tol_cons=0.4
):
    print(f"  > Generating Theoretical Hubs for Hypotheses (max_hkl={max_hkl_hyp})...")
    h_vals = np.arange(-max_hkl_hyp, max_hkl_hyp + 1)
    h, k, l = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
    hkl = np.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)
    mask_hkl = ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))
    theo_hkl = hkl[:, mask_hkl].astype(np.float32).T 
    r_nodes_hyp = (B_mat @ theo_hkl.T).T
    r_nodes_hyp /= np.linalg.norm(r_nodes_hyp, axis=1, keepdims=True)
    
    print(f"  > Executing Exhaustive Pairwise Triad Generation...")
    
    # FIX 1: Polarity-Invariant Angles
    emp_dots = np.clip(np.abs(e_nodes @ e_nodes.T), 0.0, 1.0)
    theo_dots = np.clip(np.abs(r_nodes_hyp @ r_nodes_hyp.T), 0.0, 1.0)
    
    emp_i, emp_j = np.triu_indices(len(e_nodes), k=1)
    theo_i, theo_j = np.triu_indices(len(r_nodes_hyp), k=1)
    
    emp_pair_angles = np.rad2deg(np.arccos(emp_dots[emp_i, emp_j]))
    theo_pair_angles = np.rad2deg(np.arccos(theo_dots[theo_i, theo_j]))
    
    # Filter stable theoretical angles (between 20 and 90 degrees absolute)
    stable_theo = theo_pair_angles > 20.0
    theo_i = theo_i[stable_theo]
    theo_j = theo_j[stable_theo]
    theo_pair_angles = theo_pair_angles[stable_theo]
    
    U_hypotheses = []
    
    for idx, e_angle in enumerate(emp_pair_angles):
        if e_angle < 20.0: continue 
        
        matches = np.where(np.abs(theo_pair_angles - e_angle) < angle_tol_hyp)[0]
        if len(matches) == 0: continue
        
        e1, e2 = e_nodes[emp_i[idx]], e_nodes[emp_j[idx]]
        v1 = e1
        v2 = np.cross(e1, e2); v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        V = np.column_stack([v1, v2, v3]) 
        
        t1_match = r_nodes_hyp[theo_i[matches]]
        t2_match = r_nodes_hyp[theo_j[matches]]
        
        # FIX 2: All 8 Exhaustive Permutations
        signs = [(1,1), (1,-1), (-1,1), (-1,-1)]
        
        for m in range(len(matches)):
            t1, t2 = t1_match[m], t2_match[m]
            
            for s1, s2 in signs:
                # Order 1 (e1 -> t1, e2 -> t2)
                w1 = s1 * t1
                wt2 = s2 * t2
                w2 = np.cross(w1, wt2); w2 /= np.linalg.norm(w2)
                w3 = np.cross(w1, w2)
                U_hypotheses.append(V @ np.column_stack([w1, w2, w3]).T)
                
                # Order 2 (e1 -> t2, e2 -> t1)
                w1_f = s1 * t2
                wt2_f = s2 * t1
                w2_f = np.cross(w1_f, wt2_f); w2_f /= np.linalg.norm(w2_f)
                w3_f = np.cross(w1_f, w2_f)
                U_hypotheses.append(V @ np.column_stack([w1_f, w2_f, w3_f]).T)

    if len(U_hypotheses) == 0:
        raise ValueError("No matching Triads found.")
        
    U_batch = np.array(U_hypotheses)
    print(f"  > Generated {len(U_batch)} orientation hypotheses. Scoring against RAW PEAKS...")

    # FIX 3: Score directly against Ground Truth Raw Peaks
    @jit
    def evaluate_peaks_hard(U_batch_j, q_obs_j, r_eval_j, tol_deg):
        r_samp = jnp.einsum('krc,mc->kmr', U_batch_j, r_eval_j)
        dots = jnp.einsum('pr,kmr->kpm', q_obs_j, r_samp)
        
        max_dots = jnp.max(jnp.abs(dots), axis=2)
        angles = jnp.rad2deg(jnp.arccos(jnp.clip(max_dots, 0.0, 1.0)))
        
        inliers = jnp.sum(angles < tol_deg, axis=1)
        residuals = jnp.sum(jnp.where(angles < tol_deg, angles, 0.0), axis=1) / jnp.maximum(inliers, 1)
        return inliers, residuals
        
    # Chunk size reduced to 1000 to prevent VRAM explosion (P=540, R=1154 matrices are huge)
    chunk_size = 1000 
    best_U = None
    best_inliers = -1
    best_residual = np.inf
    
    q_obs_j = jnp.array(q_sample_obs_norm, dtype=jnp.float32)
    r_eval_j = jnp.array(r_eval_rays_norm, dtype=jnp.float32) 
    
    for i in range(0, len(U_batch), chunk_size):
        chunk = jnp.array(U_batch[i:i+chunk_size], dtype=jnp.float32)
        inliers, residuals = evaluate_peaks_hard(chunk, q_obs_j, r_eval_j, angle_tol_cons)
        
        score = inliers - (residuals / 1000.0)
        max_idx = jnp.argmax(score)
        
        if inliers[max_idx] > best_inliers or (inliers[max_idx] == best_inliers and residuals[max_idx] < best_residual):
            best_inliers = inliers[max_idx]
            best_residual = residuals[max_idx]
            best_U = U_batch[i + int(max_idx)]

    print(f"  > Global Consensus Achieved! Best Matrix mapped {best_inliers}/{len(q_sample_obs_norm)} RAW PEAKS (Mean Error: {best_residual:.3f} deg).")
    return best_U

# -------------------------------------------------------------
# CONTINUOUS VORONOI OPTIMIZER (HARD ASSIGNMENT)
# -------------------------------------------------------------

@jit
def quaternion_to_rotation_matrix(q):
    q = q / jnp.linalg.norm(q) 
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

@jit
def voronoi_ray_loss(q, q_obs_norm, r_unique_rays_norm):
    U = quaternion_to_rotation_matrix(q)
    r_lab = jnp.matmul(r_unique_rays_norm, U.T)
    sim_matrix = jnp.matmul(q_obs_norm, r_lab.T)
    max_sims = jnp.max(sim_matrix, axis=1)
    return -jnp.sum(max_sims)

def optimize_orientation_gradient_descent(q_init, q_obs_norm, r_unique_rays_norm, steps=300):
    optimizer = optax.adam(learning_rate=0.0005) 
    opt_state = optimizer.init(q_init)
    loss_fn = jax.jit(jax.value_and_grad(voronoi_ray_loss))
    
    q_current = q_init
    best_q = q_init
    best_loss = jnp.inf 
    
    print("  > Launching Hard Voronoi Gradient Descent (Micro-stepping)...")
    for step in range(steps):
        loss_val, grads = loss_fn(q_current, q_obs_norm, r_unique_rays_norm)
        if loss_val < best_loss:
            best_loss = loss_val
            best_q = q_current
            
        updates, opt_state = optimizer.update(grads, opt_state)
        q_current = optax.apply_updates(q_current, updates)
        if step % 100 == 0:
            print(f"    Step {step:03d} | Voronoi Loss: {loss_val:.3f}")
            
    q_final = best_q / jnp.linalg.norm(best_q)
    return quaternion_to_rotation_matrix(q_final)
