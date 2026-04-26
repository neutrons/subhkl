import jax.numpy as jnp
from jax import jit
import numpy as np

def get_data_driven_dictionary(q_lab_obs):
    """
    The Support Vector Dictionary.
    Exhaustively generates exact normal vectors by taking the cross product 
    of all valid empirical peak pairs.
    """
    q_norms = jnp.linalg.norm(q_lab_obs, axis=1, keepdims=True)
    q_lab = np.where(q_norms == 0, 1.0, q_lab_obs / q_norms)
    N_obs = q_lab.shape[0]
    
    # Generate upper triangle indices for unique pairwise combinations
    i_idx, j_idx = np.triu_indices(N_obs, k=1)
    
    # Cross products define the exact normal of the plane containing both peaks
    cross_prods = np.cross(q_lab[i_idx], q_lab[j_idx])
    norms = np.linalg.norm(cross_prods, axis=1, keepdims=True)
    
    # The Lever Rule: Require peaks to be separated by at least ~1 degree
    # to form a mathematically stable geometric plane (sin(1 deg) ~ 0.017)
    valid_mask = np.squeeze(norms > 0.017)
    valid_cross = np.where(norms == 0, 1.0, cross_prods / norms)[valid_mask]
    
    return valid_cross

@jit
def greedy_set_cover_step(unexplained_mask, q_lab, dict_zones, tol_sin):
    """
    JIT-compiled sequential RANSAC step.
    Finds the Great Circle that covers the maximum number of UNEXPLAINED peaks,
    ranked by its true Angular Spread Density.
    """
    # 1. Distance matrix: Angle from every peak to every candidate plane
    cos_sim = jnp.abs(jnp.matmul(dict_zones, q_lab.T))
    
    # 2. Hard threshold inlier masking
    inlier_matrix = (cos_sim < tol_sin).astype(jnp.float32)
    valid_inliers = inlier_matrix * unexplained_mask
    
    # 3. Raw count of explained peaks for each candidate
    inlier_counts = jnp.sum(valid_inliers, axis=1)
    
    # 4. Calculate Vector Sum of all inliers for each plane
    # (N_dict, N_obs) @ (N_obs, 3) -> (N_dict, 3)
    sum_q = jnp.matmul(valid_inliers, q_lab)
    norm_sum = jnp.linalg.norm(sum_q, axis=1)
    
    # 5. Mean Resultant Length (Circular Variance)
    R_mean = norm_sum / jnp.maximum(inlier_counts, 1.0)
    
    # 6. Actual Angular Spread (Arc Length)
    angular_spread = jnp.sqrt(jnp.maximum(1.0 - R_mean, 1e-6))
    
    # 7. Density Score: Chi-Squared divided by observed physical arc length
    # Clamp spread to 0.05 to prevent infinite scores for tiny 2-pixel clumps
    scores = (inlier_counts ** 2) / jnp.maximum(angular_spread, 0.05)
    
    # 8. Extract the undisputed winner
    best_idx = jnp.argmax(scores)
    best_count = inlier_counts[best_idx]
    best_score = scores[best_idx]
    
    # 9. Annihilate the captured points from the search space
    new_explained_points = valid_inliers[best_idx]
    new_unexplained_mask = unexplained_mask * (1.0 - new_explained_points)
    
    return best_idx, best_count, best_score, new_unexplained_mask


class SparseHoughIndexer:
    """
    Data-Driven Hough Transform via Sequential Set Cover.
    """
    def __init__(self, tolerance_deg=0.25, **kwargs):
        self.tolerance_deg = tolerance_deg
        
    def find_active_zones(self, q_lab_obs, max_axes=15, min_peaks=5):
        print("  > Generating Exhaustive Support Vector Dictionary...")
        dict_zones = get_data_driven_dictionary(np.array(q_lab_obs))
        print(f"  > Compiled {len(dict_zones)} analytic candidate axes.")
        
        print(f"  > Running Density-Driven Set Cover (tol={self.tolerance_deg} deg)...")
        
        # Prepare normalized geometry for JAX
        q_norms = np.linalg.norm(q_lab_obs, axis=1, keepdims=True)
        q_lab = np.where(q_norms == 0, 1.0, q_lab_obs / q_norms)
        
        j_q_lab = jnp.array(q_lab, dtype=jnp.float32)
        j_dict_zones = jnp.array(dict_zones, dtype=jnp.float32)
        tol_sin = jnp.sin(jnp.deg2rad(self.tolerance_deg))
        
        # All points start as 1.0 (unexplained)
        unexplained_mask = jnp.ones(len(q_lab), dtype=jnp.float32)
        
        final_zones = []
        final_scores = []
        
        # Sequentially extract the densest Laue Cones
        for step in range(max_axes):
            best_idx, count, chi_score, unexplained_mask = greedy_set_cover_step(
                unexplained_mask, j_q_lab, j_dict_zones, tol_sin
            )
            
            # Terminate if the highest remaining density is just background noise
            if count < min_peaks:
                break
                
            best_zone = np.array(dict_zones[best_idx])
            
            final_zones.append(best_zone)
            final_scores.append(float(chi_score))
            
        final_zones = np.array(final_zones)
        final_scores = np.array(final_scores)

        # The greedy sequence is not strictly monotonic in density!
        order = np.argsort(final_scores)[::-1]
        return final_zones[order], final_scores[order]

import jax
import jax.numpy as jnp
import s2fft
import numpy as np

class ContinuousSphericalHough:
    def __init__(self, L_max=128):
        """
        L_max controls the angular resolution. 
        L_max=128 gives roughly ~1.4 degree resolution.
        """
        self.L = L_max
        
        # Precompute the Spherical Radon Transform operator: 2*pi*P_l(0)
        # P_l(0) is only non-zero for even L.
        self.srt_operator = np.zeros(self.L)
        for l in range(0, self.L, 2):
            # Legendre polynomial at x=0 formula for even l
            val = ((-1)**(l//2)) * np.math.factorial(l) / ((2**l) * (np.math.factorial(l//2)**2))
            self.srt_operator[l] = 2 * jnp.pi * val
            
        self.srt_operator = jnp.array(self.srt_operator)

    def accumulate_panels_to_grid(self, images_dict, detectors_dict, sample_offset):
        """
        Resamples raw detector panels onto a regular Equiangular/MW grid required by s2fft.
        """
        # Create an empty Equiangular grid (McEwen-Wiaux sampling)
        # Shape is (L, 2*L-1)
        grid = jnp.zeros((self.L, 2 * self.L - 1))
        
        for img_key, raw_image in images_dict.items():
            det = detectors_dict[img_key]
            
            # 1. Get pixel coordinates
            row_grid, col_grid = np.indices((det.n, det.m))
            xyz = det.pixel_to_lab(row_grid, col_grid) - sample_offset
            
            # 2. Normalize to unit vectors
            norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
            kf = xyz / np.where(norms == 0, 1.0, norms)
            
            # 3. Convert to Spherical Angles (Theta, Phi)
            theta = np.arccos(kf[..., 2])
            phi = np.arctan2(kf[..., 1], kf[..., 0])
            phi = np.mod(phi, 2 * np.pi)
            
            # 4. Map angles to s2fft Grid Indices
            # Theta maps to [0, L-1], Phi maps to [0, 2L-2]
            theta_idx = jnp.clip(jnp.round((theta / jnp.pi) * (self.L - 1)), 0, self.L - 1).astype(jnp.int32)
            phi_idx = jnp.clip(jnp.round((phi / (2 * jnp.pi)) * (2 * self.L - 2)), 0, 2 * self.L - 2).astype(jnp.int32)
            
            # 5. Scatter Add intensities into the global grid
            # (In reality, use jax.lax.scatter_add for JIT compilation)
            flat_idx = theta_idx * (2 * self.L - 1) + phi_idx
            flat_grid = grid.flatten()
            flat_grid = flat_grid.at[flat_idx.flatten()].add(raw_image.flatten())
            grid = flat_grid.reshape((self.L, 2 * self.L - 1))
            
        return grid

    @jax.jit
    def spherical_radon_transform(self, intensity_grid):
        """
        The O(N^2 log^2 N) Continuous Hough Transform.
        """
        # 1. Forward Spherical Harmonic Transform
        # Output flm shape: (L, 2L-1)
        flm = s2fft.forward(intensity_grid, L=self.L)
        
        # 2. Apply SRT Operator (Multiply every m-mode by the l-th scalar)
        srt_flm = jax.vmap(lambda l: flm[l, :] * self.srt_operator[l])(jnp.arange(self.L))
        
        # 3. Inverse Transform to get the Hough Space
        hough_space = s2fft.inverse(srt_flm, L=self.L)
        
        return jnp.abs(hough_space) # Absolute value to handle floating point ringing
