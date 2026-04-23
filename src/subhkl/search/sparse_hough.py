import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from functools import partial
from subhkl.search.ssn import solve_ssn_unified

def get_data_driven_dictionary(q_lab_obs, max_pairs=20000):
    """
    Generates dictionary axes analytically from the empirical data.
    Every pair of peaks defines a candidate normal vector via cross product.
    We subsample to max_pairs to keep memory bounded.
    """
    # Normalize empirical peaks
    q_norms = jnp.linalg.norm(q_lab_obs, axis=1, keepdims=True)
    q_lab = jnp.where(q_norms == 0, 1.0, q_lab_obs / q_norms)
    N_obs = q_lab.shape[0]
    
    # Generate all pairs (i < j)
    # Using jnp.triu_indices is fast but can OOM for N > 5000. N=540 is trivial.
    i_idx, j_idx = jnp.triu_indices(N_obs, k=1)
    
    # Calculate cross products
    cross_prods = jnp.cross(q_lab[i_idx], q_lab[j_idx])
    norms = jnp.linalg.norm(cross_prods, axis=1, keepdims=True)
    
    # Filter out collinear pairs (norm ~ 0)
    valid_mask = jnp.squeeze(norms > 1e-4)
    valid_cross = jnp.where(norms == 0, 1.0, cross_prods / norms)
    
    # Gather valid candidates
    candidates = valid_cross[valid_mask]
    
    # Subsample if necessary (deterministic subset)
    step = max(1, len(candidates) // max_pairs)
    final_dict = candidates[::step][:max_pairs]
    
    return final_dict

def build_hough_dictionary(q_lab_obs, dict_zones, kappa=50.0):
    q_norms = jnp.linalg.norm(q_lab_obs, axis=1, keepdims=True)
    q_lab = q_lab_obs / jnp.where(q_norms == 0, 1.0, q_norms)
    cos_sim = jnp.matmul(q_lab, dict_zones.T)
    # Bingham Kernel
    A_mat = jnp.exp(-kappa * (cos_sim ** 2))
    return A_mat

@jit
def solve_sparse_hough_jax(q_lab_obs, dict_zones, kappa, alpha, bg_val):
    N_obs = q_lab_obs.shape[0]
    N_dict = dict_zones.shape[0]
    
    A_mat = build_hough_dictionary(q_lab_obs, dict_zones, kappa=kappa)
    y_target = jnp.ones(N_obs, dtype=jnp.float32)
    bg_flat = jnp.full(N_obs, bg_val, dtype=jnp.float32)
    c_warm = jnp.zeros(N_dict, dtype=jnp.float32)
    alpha_vec = jnp.full(N_dict, alpha, dtype=jnp.float32)
    
    c_sparse = solve_ssn_unified(
        A_mat, y_target, bg_flat, alpha_vec, 
        loss_type=0, c_warm=c_warm, max_iter=50
    )
    return c_sparse

@jit
def solve_sparse_hough_tuned_jax(q_lab_obs, dict_zones, kappa, candidate_alphas, bg_val):
    N_obs = q_lab_obs.shape[0]
    N_dict = dict_zones.shape[0]
    
    A_mat = build_hough_dictionary(q_lab_obs, dict_zones, kappa=kappa)
    y_target = jnp.ones(N_obs, dtype=jnp.float32)
    bg_flat = jnp.full(N_obs, bg_val, dtype=jnp.float32)
    c_warm = jnp.zeros(N_dict, dtype=jnp.float32)
    
    def evaluate_alpha(alpha_val):
        alpha_vec = jnp.full(N_dict, alpha_val, dtype=jnp.float32)
        c_sparse = solve_ssn_unified(
            A_mat, y_target, bg_flat, alpha_vec, 
            loss_type=0, c_warm=c_warm, max_iter=50
        )
        
        active_mask = c_sparse > 1e-4
        k_active = jnp.sum(active_mask)
        
        u = A_mat @ c_sparse + bg_flat
        nll = 0.5 * jnp.sum((u - y_target) ** 2)
        
        # Calculate BIC
        bic = jnp.where(k_active == 0, 1e9, k_active * jnp.log(N_obs) + 2.0 * nll)
        return bic, c_sparse

    bics, all_c = vmap(evaluate_alpha)(candidate_alphas)
    best_idx = jnp.argmin(bics)
    return all_c[best_idx], bics[best_idx], candidate_alphas[best_idx]

class SparseHoughIndexer:
    def __init__(self, max_dict_size=20000, kappa=50.0, alpha=None, bg_val=0.1, auto_tune=True,
                 candidate_alphas=None):
        self.max_dict_size = max_dict_size
        self.kappa = kappa
        self.alpha = alpha
        self.bg_val = bg_val
        self.auto_tune = auto_tune

        if candidate_alphas is None:
            # A broad sweep of Z-scores from very loose (2.0) to extremely strict (50.0)
            self.candidate_alphas = jnp.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0], dtype=jnp.float32)
        else:
            self.candidate_alphas = jnp.array(candidate_alphas)


    def find_active_zones(self, q_lab_obs, max_dict_size=20000):
        # 1. Build the mathematically guaranteed Dictionary
        print("  > Generating Data-Driven Dictionary from exact empirical pairs...")
        self.dict_zones = get_data_driven_dictionary(jnp.array(q_lab_obs), max_pairs=max_dict_size)
        print(f"  > Compiled {len(self.dict_zones)} analytic candidate axes.")

        if self.auto_tune:
            print(f"  > Auto-tuning Alpha across {self.candidate_alphas.tolist()}...")
            c_sparse, best_bic, best_alpha = solve_sparse_hough_tuned_jax(
                jnp.array(q_lab_obs, dtype=jnp.float32),
                self.dict_zones,
                self.kappa,
                self.candidate_alphas,
                self.bg_val
            )
            print(f"  > Optimal Alpha selected: {best_alpha:.1f} (BIC: {best_bic:.1f})")
        else:
            c_sparse = solve_sparse_hough_jax(
                jnp.array(q_lab_obs, dtype=jnp.float32),
                self.dict_zones,
                self.kappa,
                self.alpha,
                self.bg_val
            )

        c_sparse_cpu = np.array(c_sparse)
        active_mask = c_sparse_cpu > 1e-4

        raw_active_zones = np.array(self.dict_zones)[active_mask]
        raw_weights = c_sparse_cpu[active_mask]

        if len(raw_active_zones) == 0:
            return np.array([]), np.array([])

        # 2. Cluster identical axes (Solver might activate multiple vectors pointing at the same cone)
        merged_zones = []
        merged_weights = []

        # Sort by weight descending
        order = np.argsort(raw_weights)[::-1]
        raw_active_zones = raw_active_zones[order]
        raw_weights = raw_weights[order]

        for z, w in zip(raw_active_zones, raw_weights):
            is_new = True
            for i, mz in enumerate(merged_zones):
                # Check for parallelism (head-tail symmetric)
                dot = np.abs(np.dot(z, mz))
                if dot > 0.99: # ~8 degrees capture radius for clustering
                    merged_weights[i] += w # Accumulate weight
                    is_new = False
                    break
            if is_new:
                merged_zones.append(z)
                merged_weights.append(w)

        # Final sort
        final_zones = np.array(merged_zones)
        final_weights = np.array(merged_weights)
        order = np.argsort(final_weights)[::-1]

        return final_zones[order], final_weights[order]
