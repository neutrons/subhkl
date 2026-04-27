import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

from subhkl.search.ssn import SparseBasisPursuit

from functools import partial

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


class AzimuthalJAXHough:
    def __init__(self, N_theta=256, N_phi=512, sigma_deg=3.0):
        self.N_theta = N_theta
        self.N_phi = N_phi
        self.sigma = np.sin(np.deg2rad(sigma_deg))
        
        # A grid of physical widths to test (e.g., 1.5 to 5.0 degrees)
        sigmas_deg = np.array([0.25, 0.5, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        self.candidate_sigmas = jnp.array(np.sin(np.deg2rad(sigmas_deg)))
        
        self.thetas = jnp.linspace(0, jnp.pi, N_theta)
        self.phis = jnp.linspace(0, 2 * jnp.pi, N_phi, endpoint=False)
        
        self.sin_theta = jnp.sin(self.thetas)
        self.cos_theta = jnp.cos(self.thetas)
        self.cos_dphi = jnp.cos(self.phis)

        THETA, PHI = jnp.meshgrid(self.thetas, self.phis, indexing='ij')
        self.Q_x = jnp.sin(THETA) * jnp.cos(PHI)
        self.Q_y = jnp.sin(THETA) * jnp.sin(PHI)
        self.Q_z = jnp.cos(THETA)

    def accumulate_to_grid(self, q_sample_vectors, intensities):
        norms = np.linalg.norm(q_sample_vectors, axis=1, keepdims=True)
        q = np.where(norms == 0, 1.0, q_sample_vectors / norms)

        theta = np.arccos(np.clip(q[:, 2], -1.0, 1.0))
        phi = np.arctan2(q[:, 1], q[:, 0])
        phi = np.mod(phi, 2 * np.pi)

        t_idx = np.clip(np.round((theta / np.pi) * (self.N_theta - 1)), 0, self.N_theta - 1).astype(np.int32)
        p_idx = np.clip(np.round((phi / (2 * np.pi)) * self.N_phi), 0, self.N_phi - 1).astype(np.int32)

        grid = np.zeros((self.N_theta, self.N_phi), dtype=np.float32)
        np.add.at(grid, (t_idx, p_idx), intensities)
        return grid

    @staticmethod
    @jit
    def _hough_scan_step(carry, Theta_k):
        I_fft, thetas, sin_t, cos_t, cos_dphi, sigma = carry
        
        sin_T = jnp.sin(Theta_k)
        cos_T = jnp.cos(Theta_k)
        
        x_ik = cos_t[:, None] * cos_T + (sin_t[:, None] * sin_T) * cos_dphi[None, :]
        W_ik = jnp.exp(-(x_ik**2) / (2 * sigma**2))
        
        W_fft = jnp.fft.rfft(W_ik, axis=1)
        sum_fft = jnp.sum(I_fft * jnp.conj(W_fft), axis=0)
        score_row = jnp.fft.irfft(sum_fft, n=cos_dphi.shape[0])
        
        return carry, score_row

    def transform(self, grid):
        I_fft = jnp.fft.rfft(grid, axis=1)
        carry = (I_fft, self.thetas, self.sin_theta, self.cos_theta, self.cos_dphi, self.sigma)
        _, hough_space = jax.lax.scan(self._hough_scan_step, carry, self.thetas)
        return hough_space

    def find_active_zones(self, grid_raw, max_axes=15, alpha=0.1, gamma=2.0, loss="gaussian", 
                          min_sigma=1.0, max_sigma=5.0, auto_tune_alpha=True, candidate_alphas=None):
        
        print("  > Projecting Raw Photons into Continuous Azimuthal Hough Space...")
        H_raw = self.transform(jnp.array(grid_raw))
        
        print("  > Applying JAX Hessian Topological Filter (Starburst Annihilation)...")
        from subhkl.search.sparse_hough import apply_hessian_starburst_filter
        H_filtered = apply_hessian_starburst_filter(H_raw)
        
        # RESTORE NORMALIZATION: Scale the topological map to [0, 1]
        H_max = jnp.max(H_filtered)
        if H_max > 0:
            H_filtered = H_filtered / H_max
            
        print("  > Engaging 2D Sparse RBF Peak Finder on Hessian Hubs...")
        from subhkl.search.sparse_rbf import SparseRBFPeakFinder
        
        # Force Gaussian loss, as the data is now a smooth mathematical feature map
        peak_finder = SparseRBFPeakFinder(
            alpha=alpha,
            gamma=gamma,
            loss="gaussian", 
            min_sigma=min_sigma, 
            max_sigma=max_sigma, 
            auto_tune_alpha=auto_tune_alpha,
            candidate_alphas=candidate_alphas,
            show_steps=True
        )
        
        peaks = peak_finder.find_peaks_batch(H_filtered[None, :, :])[0]
        
        if len(peaks) == 0:
            print("  > No valid zone axes survived the topological filter.")
            return np.empty((0,3)), np.empty(0)
            
        print(f"  > RBF Solver extracted {len(peaks)} purified Zone Axes.")
        
        order = np.argsort(peaks[:, 0])[::-1]
        top_peaks = peaks[order][:max_axes]
        
        zones = []
        weights = []
        
        for p in top_peaks:
            intensity, r, c, sig = p
            physical_weight = float(intensity * H_max)

            # Interpolate using the true physical axes of the Hough space!
            Theta = float(np.interp(r, np.arange(self.N_theta), self.thetas))

            # We handle Phi with wrapping in case the sub-pixel center pushed it slightly out of bounds
            Phi = float(np.interp(c % self.N_phi, np.arange(self.N_phi), self.phis))

            z_x = np.sin(Theta) * np.cos(Phi)
            z_y = np.sin(Theta) * np.sin(Phi)
            z_z = np.cos(Theta)

            zones.append([z_x, z_y, z_z])
            weights.append(physical_weight)

        return np.array(zones), np.array(weights)

@jax.jit
def apply_hessian_starburst_filter(H_grid):
    from subhkl.search.sparse_rbf import jax_gaussian_blur_2d
    
    pad_w = 15
    
    # 1. AZIMUTHAL WRAPPING: Wrap Phi (axis 1) so the X-Z plane connects perfectly!
    H_pad_phi = jnp.pad(H_grid, ((0, 0), (pad_w, pad_w)), mode='wrap')
    
    # 2. POLAR EDGE PADDING: Pad Theta (axis 0) with 'edge' to prevent vertical derivative artifacts.
    # (We use 'edge' because crossing the North/South pole flips the azimuth, which is too complex for a simple wrap).
    H_padded = jnp.pad(H_pad_phi, ((pad_w, pad_w), (0, 0)), mode='edge')
    
    # 3. Blur the padded, continuous spherical grid
    H_smooth = jax_gaussian_blur_2d(H_padded, sigma=3.0)
    
    # 4. Compute unbroken spatial gradients across the X-Z boundary
    dy, dx = jnp.gradient(H_smooth)
    dyy, dyx = jnp.gradient(dy)
    dxy, dxx = jnp.gradient(dx)
    
    det = (dxx * dyy) - (dxy ** 2)
    trace = dxx + dyy
    
    # 5. The Pure Topological Feature Map
    topo_intensity = jnp.where((trace < 0) & (det > 0), jnp.sqrt(det), 0.0)
    
    # 6. Slice the padding completely off to return to the exact original grid shape!
    return topo_intensity[pad_w:-pad_w, pad_w:-pad_w]
