import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation
import optax

def align_virtual_nodes(
    e_nodes, 
    q_sample_obs_norm,   
    r_eval_rays_norm,    
    B_mat, 
    max_hkl_hyp=4,       # <--- MUST BE HIGH ENOUGH TO CONTAIN TRUE HUBS
    angle_tol_hyp=2.5,   # <--- Wider catch-basin for noisy empirical hubs
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
    
    print(f"  > Executing Gram-Schmidt Triad Generation...")
    
    emp_dots = e_nodes @ e_nodes.T
    theo_dots = r_nodes_hyp @ r_nodes_hyp.T
    
    emp_i, emp_j = np.triu_indices(len(e_nodes), k=1)
    theo_i, theo_j = np.triu_indices(len(r_nodes_hyp), k=1)
    
    emp_pair_dots = emp_dots[emp_i, emp_j]
    theo_pair_dots = theo_dots[theo_i, theo_j]
    
    # Use rigorous absolute angles (Laue rays are unpolarized lines)
    emp_angles = np.rad2deg(np.arccos(np.clip(emp_pair_dots, -1.0, 1.0)))
    theo_angles = np.rad2deg(np.arccos(np.clip(theo_pair_dots, -1.0, 1.0)))
    
    U_hypotheses = []
    
    for idx, e_ang in enumerate(emp_angles):
        # Ignore parallel or orthogonal pairs (too unstable)
        if e_ang < 20.0 or e_ang > 160.0: continue 
        
        e1, e2 = e_nodes[emp_i[idx]], e_nodes[emp_j[idx]]
        
        # Build strict right-handed Empirical Triad (Sample Frame)
        v1 = e1
        v2 = e2 - np.dot(e1, e2) * e1
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        V = np.column_stack([v1, v2, v3]) 
        
        for m in range(len(theo_angles)):
            t_ang = theo_angles[m]
            t1, t2 = r_nodes_hyp[theo_i[m]], r_nodes_hyp[theo_j[m]]
            
            # Case 1: The internal angle matches directly
            if np.abs(t_ang - e_ang) < angle_tol_hyp:
                # Order A
                w1 = t1
                w2 = t2 - np.dot(t1, t2) * t1
                w2 /= np.linalg.norm(w2)
                w3 = np.cross(w1, w2)
                U_hypotheses.append(V @ np.column_stack([w1, w2, w3]).T)
                
                # Order B (Swapped)
                w1_f = t2
                w2_f = t1 - np.dot(t2, t1) * t2
                w2_f /= np.linalg.norm(w2_f)
                w3_f = np.cross(w1_f, w2_f)
                U_hypotheses.append(V @ np.column_stack([w1_f, w2_f, w3_f]).T)
                
            # Case 2: The internal angle matches the supplement (because lines have no polarity)
            if np.abs((180.0 - t_ang) - e_ang) < angle_tol_hyp:
                t2_neg = -t2
                
                # Order C
                w1_n = t1
                w2_n = t2_neg - np.dot(t1, t2_neg) * t1
                w2_n /= np.linalg.norm(w2_n)
                w3_n = np.cross(w1_n, w2_n)
                U_hypotheses.append(V @ np.column_stack([w1_n, w2_n, w3_n]).T)
                
                # Order D (Swapped)
                w1_nf = t2_neg
                w2_nf = t1 - np.dot(t2_neg, t1) * t2_neg
                w2_nf /= np.linalg.norm(w2_nf)
                w3_nf = np.cross(w1_nf, w2_nf)
                U_hypotheses.append(V @ np.column_stack([w1_nf, w2_nf, w3_nf]).T)

    if len(U_hypotheses) == 0:
        raise ValueError("No matching Triads found. Relax angle_tol_hyp.")
        
    U_batch = np.array(U_hypotheses)
    print(f"  > Generated {len(U_batch)} orientation hypotheses. Scoring against RAW PEAKS...")

    # SCORE U-MATRIX DIRECTLY AGAINST RAW PHYSICAL PEAKS
    @jit
    def evaluate_peaks_hard(U_batch_j, q_obs_j, r_eval_j, tol_deg):
        r_samp = jnp.einsum('krc,mc->kmr', U_batch_j, r_eval_j)
        dots = jnp.einsum('pr,kmr->kpm', q_obs_j, r_samp)
        
        max_dots = jnp.max(jnp.abs(dots), axis=2)
        angles = jnp.rad2deg(jnp.arccos(jnp.clip(max_dots, 0.0, 1.0)))
        
        inliers = jnp.sum(angles < tol_deg, axis=1)
        residuals = jnp.sum(jnp.where(angles < tol_deg, angles, 0.0), axis=1) / jnp.maximum(inliers, 1)
        return inliers, residuals
        
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
