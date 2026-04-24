import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation
import optax

def align_virtual_nodes(
    e_nodes, 
    e_weights,           
    r_eval_rays_norm,    
    B_mat, 
    max_hkl_hyp=2,       # <--- 2 keeps pair generation lightning fast
    angle_tol_hyp=2.5
):
    print(f"  > Generating Theoretical Hubs for Hypotheses (max_hkl={max_hkl_hyp})...")
    h_vals = np.arange(-max_hkl_hyp, max_hkl_hyp + 1)
    h, k, l = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
    hkl = np.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)
    mask_hkl = ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))
    theo_hkl = hkl[:, mask_hkl].astype(np.float32).T 
    r_nodes_hyp = (B_mat @ theo_hkl.T).T
    r_nodes_hyp /= np.linalg.norm(r_nodes_hyp, axis=1, keepdims=True)
    
    print(f"  > Executing Weight-Prioritized Symmetric Triad Generation...")
    
    emp_dots = np.clip(np.abs(e_nodes @ e_nodes.T), 0.0, 1.0)
    theo_dots = np.clip(np.abs(r_nodes_hyp @ r_nodes_hyp.T), 0.0, 1.0)
    
    emp_i, emp_j = np.triu_indices(len(e_nodes), k=1)
    theo_i, theo_j = np.triu_indices(len(r_nodes_hyp), k=1)
    
    emp_pair_angles = np.rad2deg(np.arccos(emp_dots[emp_i, emp_j]))
    theo_pair_angles = np.rad2deg(np.arccos(theo_dots[theo_i, theo_j]))
    
    emp_pair_weights = e_weights[emp_i] * e_weights[emp_j]
    pair_order = np.argsort(emp_pair_weights)[::-1]
    
    stable_theo = theo_pair_angles > 20.0
    theo_i = theo_i[stable_theo]
    theo_j = theo_j[stable_theo]
    theo_pair_angles = theo_pair_angles[stable_theo]
    
    U_hypotheses = []
    MAX_HYPOTHESES = 350000 
    
    for idx in pair_order:
        e_angle = emp_pair_angles[idx]
        if e_angle < 20.0 or e_angle > 160.0: continue 
        
        matches = np.where(np.abs(theo_pair_angles - e_angle) < angle_tol_hyp)[0]
        if len(matches) == 0: continue
        
        e1, e2 = e_nodes[emp_i[idx]], e_nodes[emp_j[idx]]
        e3 = np.cross(e1, e2)
        emp_vecs = np.array([e1, e2, e3])
        
        t1_match = r_nodes_hyp[theo_i[matches]]
        t2_match = r_nodes_hyp[theo_j[matches]]
        
        signs = [(1,1), (1,-1), (-1,1), (-1,-1)]
        
        for m in range(len(matches)):
            t1, t2 = t1_match[m], t2_match[m]
            
            for s1, s2 in signs:
                w1, w2 = s1 * t1, s2 * t2
                w3 = np.cross(w1, w2)
                rot, _ = Rotation.align_vectors(emp_vecs, np.array([w1, w2, w3]))
                U_hypotheses.append(rot.as_matrix())
                
                w3_f = np.cross(w2, w1)
                rot_f, _ = Rotation.align_vectors(emp_vecs, np.array([w2, w1, w3_f]))
                U_hypotheses.append(rot_f.as_matrix())
                
        if len(U_hypotheses) > MAX_HYPOTHESES:
            break

    if len(U_hypotheses) == 0:
        raise ValueError("No matching Triads found.")
        
    U_batch = np.array(U_hypotheses)
    print(f"  > Generated {len(U_batch)} hypotheses. Scoring against {len(e_nodes)} WEIGHTED HUBS...")

    @jit
    def evaluate_hubs_weighted(U_batch_j, e_nodes_j, e_weights_j, r_eval_j, tol_deg):
        r_samp = jnp.einsum('krc,mc->kmr', U_batch_j, r_eval_j)
        dots = jnp.einsum('nc,kmc->knm', e_nodes_j, r_samp)
        
        max_dots = jnp.max(jnp.abs(dots), axis=2)
        angles = jnp.rad2deg(jnp.arccos(jnp.clip(max_dots, 0.0, 1.0)))
        
        inliers = angles < tol_deg
        # SCORING FIX: Maximize the SUM OF THE WEIGHTS of the correctly mapped hubs
        scores = jnp.sum(inliers * e_weights_j, axis=1)
        residuals = jnp.sum(jnp.where(inliers, angles, 0.0), axis=1) / jnp.maximum(jnp.sum(inliers, axis=1), 1)
        return scores, residuals, jnp.sum(inliers, axis=1)
        
    chunk_size = 50000 
    best_U = None
    best_score = -1.0
    best_inliers_count = 0
    best_residual = np.inf
    
    e_nodes_j = jnp.array(e_nodes, dtype=jnp.float32)
    e_weights_j = jnp.array(e_weights, dtype=jnp.float32)
    r_eval_j = jnp.array(r_eval_rays_norm, dtype=jnp.float32) 
    
    search_tol = 1.5 
    
    for i in range(0, len(U_batch), chunk_size):
        chunk = jnp.array(U_batch[i:i+chunk_size], dtype=jnp.float32)
        scores, residuals, inliers_count = evaluate_hubs_weighted(chunk, e_nodes_j, e_weights_j, r_eval_j, search_tol)
        
        max_idx = jnp.argmax(scores)
        
        if scores[max_idx] > best_score or (scores[max_idx] == best_score and residuals[max_idx] < best_residual):
            best_score = scores[max_idx]
            best_inliers_count = inliers_count[max_idx]
            best_residual = residuals[max_idx]
            best_U = U_batch[i + int(max_idx)]

    print(f"  > Hub Consensus Achieved! Best Matrix mapped {best_inliers_count}/{len(e_nodes)} WEIGHTED HUBS (Mean Error: {best_residual:.3f} deg).")
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
    # THE FATAL BUG FIX: jnp.abs() added to correctly respect Laue +/- symmetry!
    max_sims = jnp.max(jnp.abs(sim_matrix), axis=1)
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
