import jax.numpy as jnp
from jax import jit
import numpy as np

def get_data_driven_dictionary(q_lab_obs, max_pairs=20000):
    """
    The 'Support Vector' Dictionary.
    Generates exact normal vectors by taking the cross product of empirical peak pairs.
    """
    q_norms = jnp.linalg.norm(q_lab_obs, axis=1, keepdims=True)
    q_lab = np.where(q_norms == 0, 1.0, q_lab_obs / q_norms)
    N_obs = q_lab.shape[0]
    
    # Generate upper triangle indices for unique pairs
    i_idx, j_idx = np.triu_indices(N_obs, k=1)
    
    # Cross products define the exact normal of the plane containing both peaks
    cross_prods = np.cross(q_lab[i_idx], q_lab[j_idx])
    norms = np.linalg.norm(cross_prods, axis=1, keepdims=True)
    
    # Filter collinear pairs
    valid_mask = np.squeeze(norms > 1e-4)
    valid_cross = np.where(norms == 0, 1.0, cross_prods / norms)[valid_mask]
    
    # Deterministic subsampling to prevent VRAM overflow
    if len(valid_cross) > max_pairs:
        step = len(valid_cross) // max_pairs
        valid_cross = valid_cross[::step][:max_pairs]
        
    return valid_cross

@jit
def score_dictionary_jax(q_lab_obs, dict_zones, kappa):
    """
    Massively parallel Kernel Density Estimation.
    Evaluates how many Bragg peaks sit perfectly on each candidate Great Circle.
    """
    q_norms = jnp.linalg.norm(q_lab_obs, axis=1, keepdims=True)
    q_lab = jnp.where(q_norms == 0, 1.0, q_lab_obs / q_norms)
    
    # Cosine similarity matrix: (N_obs, 3) @ (3, N_dict) -> (N_obs, N_dict)
    cos_sim = jnp.matmul(q_lab, dict_zones.T)
    
    # Bingham Kernel Density (Summing the peaks that land inside the margin)
    scores = jnp.sum(jnp.exp(-kappa * (cos_sim ** 2)), axis=0)
    return scores

class SparseHoughIndexer:
    def __init__(self, max_dict_size=20000, kappa=400.0, **kwargs):
        self.max_dict_size = max_dict_size
        self.kappa = kappa
        self.dict_zones = None
        
    def find_active_zones(self, q_lab_obs, min_score=3.0, nms_tolerance_deg=8.0):
        print(f"  > Generating Support Vector Dictionary (Max {self.max_dict_size} pairs)...")
        self.dict_zones = get_data_driven_dictionary(np.array(q_lab_obs), max_pairs=self.max_dict_size)
        print(f"  > Compiled {len(self.dict_zones)} analytic candidate axes.")
        
        # 1. Score all candidates simultaneously (~5 milliseconds)
        print(f"  > Evaluating Kernel Density (kappa={self.kappa})...")
        scores = score_dictionary_jax(
            jnp.array(q_lab_obs, dtype=jnp.float32), 
            jnp.array(self.dict_zones, dtype=jnp.float32), 
            self.kappa
        )
        scores = np.array(scores)
        
        # 2. Filter mathematically weak zones
        valid_mask = scores > min_score
        raw_zones = self.dict_zones[valid_mask]
        raw_scores = scores[valid_mask]
        
        if len(raw_zones) == 0:
            return np.array([]), np.array([])
            
        # Sort descending by density score
        order = np.argsort(raw_scores)[::-1]
        raw_zones = raw_zones[order]
        raw_scores = raw_scores[order]
        
        # 3. Non-Maximum Suppression (NMS)
        # Because adjacent pairs on the same Laue cone produce almost identical cross products,
        # the dictionary contains many redundant vectors. We cluster them dynamically.
        merged_zones = []
        merged_scores = []
        
        cos_tol = np.cos(np.deg2rad(nms_tolerance_deg))
        
        for z, s in zip(raw_zones, raw_scores):
            is_new = True
            for i, mz in enumerate(merged_zones):
                # Head-tail symmetric dot product
                if np.abs(np.dot(z, mz)) > cos_tol:
                    # Accumulate score but keep the stronger vector's direction
                    merged_scores[i] += s
                    is_new = False
                    break
            if is_new:
                merged_zones.append(z)
                merged_scores.append(s)
                
        final_zones = np.array(merged_zones)
        final_scores = np.array(merged_scores)
        
        # Final sort to return the absolute strongest Principal Laue Cones
        order = np.argsort(final_scores)[::-1]
        return final_zones[order][:15], final_scores[order][:15]
