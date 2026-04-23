import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from functools import partial
from subhkl.search.ssn import solve_ssn_unified

def get_fibonacci_hemisphere(samples=2000):
    """
    Generates a uniform dictionary of possible Zone Axes covering a hemisphere.
    (We only need a hemisphere because q and -q define the same equator).
    """
    phi = jnp.pi * (3.0 - jnp.sqrt(5.0))
    i = jnp.arange(samples)
    y = 1.0 - (i / float(samples - 1))  # 1.0 to 0.0 (hemisphere)
    radius = jnp.sqrt(1.0 - y * y)
    theta = phi * i
    x = jnp.cos(theta) * radius
    z = jnp.sin(theta) * radius
    return jnp.stack([x, y, z], axis=1)

def build_hough_dictionary(q_lab_obs, dict_zones, kappa=50.0):
    """
    Builds the 'A' matrix. 
    Rows = Empirical Peaks. Columns = Dictionary Zone Axes.
    """
    # Normalize peaks
    q_norms = jnp.linalg.norm(q_lab_obs, axis=1, keepdims=True)
    q_lab = q_lab_obs / jnp.where(q_norms == 0, 1.0, q_norms)
    
    # Cosine similarity matrix
    # q_lab is (N_obs, 3), dict_zones is (N_dict, 3) -> cos_sim is (N_obs, N_dict)
    cos_sim = jnp.matmul(q_lab, dict_zones.T)
    
    # The Bingham "Equator" Kernel. Approaches 1.0 when perfectly orthogonal.
    A_mat = jnp.exp(-kappa * (cos_sim ** 2))
    return A_mat

@jit
def solve_sparse_hough_jax(q_lab_obs, dict_zones, kappa, alpha, bg_val):
    N_obs = q_lab_obs.shape[0]
    N_dict = dict_zones.shape[0]
    
    # 1. Build the Physics Dictionary
    A_mat = build_hough_dictionary(q_lab_obs, dict_zones, kappa=kappa)
    
    # 2. Setup the SSN Targets
    # We want the combined density of active equators to equal 1.0 for every peak
    y_target = jnp.ones(N_obs, dtype=jnp.float32)
    bg_flat = jnp.full(N_obs, bg_val, dtype=jnp.float32)
    
    c_warm = jnp.zeros(N_dict, dtype=jnp.float32)
    alpha_vec = jnp.full(N_dict, alpha, dtype=jnp.float32)
    
    # 3. Run the Unified Semi-Smooth Newton Solver
    # loss_type = 0 corresponds to Gaussian (OLS) loss
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

        # 1. Number of parameters (Active Zone Axes)
        active_mask = c_sparse > 1e-4
        k_active = jnp.sum(active_mask)

        # 2. Negative Log Likelihood (Gaussian/OLS)
        u = A_mat @ c_sparse + bg_flat
        nll = 0.5 * jnp.sum((u - y_target) ** 2)

        # 3. Calculate BIC (Heavily penalize finding absolutely zero axes)
        bic = jnp.where(k_active == 0, 1e9, k_active * jnp.log(N_obs) + 2.0 * nll)

        return bic, c_sparse

    # Vectorize the solver across all candidate alphas in parallel
    bics, all_c = vmap(evaluate_alpha)(candidate_alphas)

    # Select the alpha that yielded the lowest BIC
    best_idx = jnp.argmin(bics)
    return all_c[best_idx], bics[best_idx], candidate_alphas[best_idx]

class SparseHoughIndexer:
    def __init__(self, dict_size=2000, kappa=50.0, alpha=5.0, bg_val=0.1, auto_tune=True,
                 candidate_alphas=None):
        self.dict_size = dict_size
        self.kappa = kappa
        self.alpha = alpha
        self.bg_val = bg_val
        self.auto_tune = auto_tune
        self.dict_zones = get_fibonacci_hemisphere(self.dict_size)

        if candidate_alphas is None:
            # A broad sweep of Z-scores from very loose (2.0) to extremely strict (50.0)
            self.candidate_alphas = jnp.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0], dtype=jnp.float32)
        else:
            self.candidate_alphas = candidate_alphas
        
    def find_active_zones(self, q_lab_obs):
        """
        Returns the absolute lab-frame coordinates of the true crystal zone axes.
        """
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
        
        # Extract the surviving dictionary elements
        c_sparse_cpu = np.array(c_sparse)
        active_mask = c_sparse_cpu > 1e-4
        
        active_zones = np.array(self.dict_zones)[active_mask]
        activation_weights = c_sparse_cpu[active_mask]
        
        # Sort by strength
        order = np.argsort(activation_weights)[::-1]
        
        return active_zones[order], activation_weights[order]
