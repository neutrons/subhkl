import numpy as np
import itertools
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation
import optax

def align_virtual_nodes(
    e_nodes, 
    B_mat, 
    max_hkl_hyp=3, 
    angle_tol_hyp=1.5
):
    """
    Finds the global orientation by pairing Virtual Hubs and scoring them via 
    Soft Assignment (Nearest Neighbor) against the theoretical dictionary.
    """
    print(f"  > Generating Theoretical Hubs (max_hkl={max_hkl_hyp})...")
    h_vals = np.arange(-max_hkl_hyp, max_hkl_hyp + 1)
    h, k, l = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
    hkl = np.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)
    mask_hkl = ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))
    theo_hkl = hkl[:, mask_hkl].astype(np.float32).T 

    r_theo_nodes = (B_mat @ theo_hkl.T).T
    r_nodes_norm = r_theo_nodes / np.linalg.norm(r_theo_nodes, axis=1, keepdims=True)
    
    print(f"  > Executing Pairwise Triad Generation...")
    
    # 1. Compute internal angles for empirical and theoretical networks
    emp_dots = e_nodes @ e_nodes.T
    theo_dots = r_nodes_norm @ r_nodes_norm.T
    
    emp_i, emp_j = np.triu_indices(len(e_nodes), k=1)
    theo_i, theo_j = np.triu_indices(len(r_nodes_norm), k=1)
    
    emp_pair_dots = emp_dots[emp_i, emp_j]
    theo_pair_dots = theo_dots[theo_i, theo_j]
    
    # We only care about matching highly stable theoretical pairs (between 20 and 160 deg)
    stable_theo = np.abs(theo_pair_dots) < 0.94
    theo_i = theo_i[stable_theo]
    theo_j = theo_j[stable_theo]
    theo_pair_dots = theo_pair_dots[stable_theo]
    
    # Convert angular tolerance to a dot product tolerance
    dot_tol = np.sin(np.deg2rad(90)) * np.deg2rad(angle_tol_hyp)
    
    U_hypotheses = []
    
    # 2. Build 3D Coordinate frames (Triads) for every matching pair
    for idx, e_dot in enumerate(emp_pair_dots):
        if np.abs(e_dot) > 0.94: continue # skip unstable empirical pairs
        
        matches = np.where(np.abs(theo_pair_dots - e_dot) < dot_tol)[0]
        if len(matches) == 0: continue
        
        # Build the Sample-Frame Coordinate System
        e1 = e_nodes[emp_i[idx]]
        e2 = e_nodes[emp_j[idx]]
        
        v1 = e1
        v2 = np.cross(e1, e2)
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        V = np.column_stack([v1, v2, v3]) 
        
        t1_match = r_nodes_norm[theo_i[matches]]
        t2_match = r_nodes_norm[theo_j[matches]]
        
        for m in range(len(matches)):
            t1 = t1_match[m]
            t2 = t2_match[m]
            
            # Permutation 1: e1->t1, e2->t2
            w1 = t1
            w2 = np.cross(t1, t2)
            w2 /= np.linalg.norm(w2)
            w3 = np.cross(w1, w2)
            W = np.column_stack([w1, w2, w3])
            U_hypotheses.append(V @ W.T) # Transforms Crystal -> Sample
            
            # Permutation 2: e1->t2, e2->t1 (Flipped Assignment)
            w1_f = t2
            w2_f = np.cross(t2, t1)
            w2_f /= np.linalg.norm(w2_f)
            w3_f = np.cross(w1_f, w2_f)
            W_f = np.column_stack([w1_f, w2_f, w3_f])
            U_hypotheses.append(V @ W_f.T)
            
            # Permutation 3: e1->-t1, e2->t2 (Centrosymmetric Flip)
            w1_i = -t1
            w2_i = np.cross(-t1, t2)
            w2_i /= np.linalg.norm(w2_i)
            w3_i = np.cross(w1_i, w2_i)
            W_i = np.column_stack([w1_i, w2_i, w3_i])
            U_hypotheses.append(V @ W_i.T)
            
            # Permutation 4: e1->-t2, e2->t1
            w1_if = -t2
            w2_if = np.cross(-t2, t1)
            w2_if /= np.linalg.norm(w2_if)
            w3_if = np.cross(w1_if, w2_if)
            W_if = np.column_stack([w1_if, w2_if, w3_if])
            U_hypotheses.append(V @ W_if.T)

    if len(U_hypotheses) == 0:
        raise ValueError("No matching Triads found. Relax angle_tol_hyp.")
        
    U_batch = np.array(U_hypotheses)
    print(f"  > Generated {len(U_batch)} orientation hypotheses. Evaluating soft assignments...")

    # 3. Soft Assignment Evaluation via JAX Tensor Math
    @jit
    def evaluate_hubs_soft(U_batch_j, e_nodes_j, r_theo_j):
        # Map empirical sample hubs backward into the theoretical crystal frame
        e_cryst = jnp.einsum('krc,nr->knc', U_batch_j, e_nodes_j)
        
        # Calculate cosine similarity against all theoretical poles
        dots = jnp.einsum('kni,mi->knm', e_cryst, r_theo_j)
        
        # Laue symmetry natively ignores polarity
        abs_dots = jnp.abs(dots)
        
        # Soft Assignment: Every empirical hub grabs its Nearest Neighbor distance.
        max_dots = jnp.max(abs_dots, axis=2)
        
        # Final Matrix Score is the average cosine similarity across all empirical hubs
        scores = jnp.mean(max_dots, axis=1)
        return scores
        
    chunk_size = 50000
    best_U = None
    best_score = -1.0
    
    e_nodes_j = jnp.array(e_nodes, dtype=jnp.float32)
    r_nodes_norm_j = jnp.array(r_nodes_norm, dtype=jnp.float32)
    
    for i in range(0, len(U_batch), chunk_size):
        chunk = jnp.array(U_batch[i:i+chunk_size], dtype=jnp.float32)
        scores = evaluate_hubs_soft(chunk, e_nodes_j, r_nodes_norm_j)
        
        max_idx = jnp.argmax(scores)
        max_score = scores[max_idx]
        
        if max_score > best_score:
            best_score = max_score
            best_U = U_batch[i + int(max_idx)]

    # A score > 0.999 is mathematically flawless alignment
    print(f"  > Hub Alignment Complete! Best Matrix mapped {len(e_nodes)} hubs with mean cosine similarity: {best_score:.5f}")
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
    r_lab = jnp.matmul(r_unique_rays_norm, U.T)
    sim_matrix = jnp.matmul(q_obs_norm, r_lab.T)
    max_sims = jnp.max(sim_matrix, axis=1)
    return -jnp.sum(max_sims)

def optimize_orientation_gradient_descent(q_init, q_obs_norm, r_unique_rays_norm, steps=300):
    """
    Polishes the seed using micro-steps and tracks the absolute lowest error state.
    """
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
    U_optimal = quaternion_to_rotation_matrix(q_final)
    
    return U_optimal
