import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

from subhkl.search.ssn import SparseBasisPursuit

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

class GlobalZoneAxisSniper(SparseBasisPursuit):
    """
    Global Specialization: 1D Gaussian Great Circles in Spherical Angular Space.
    """
    def __init__(self, alpha=5.0, gamma=1.0, loss="gaussian", ref_sigma=0.75, 
                 auto_tune_alpha=True, candidate_alphas=None):
        
        # S/N sweep and Gaussian Loss for "Sparse Necklaces"
        default_alphas = candidate_alphas or [10.0, 15.0,30.0,45.0,50.0,75.0]
        
        super().__init__(
            alpha=alpha, gamma=gamma, loss=loss, ref_sigma=ref_sigma, 
            auto_tune_alpha=auto_tune_alpha, candidate_alphas=default_alphas
        )

    def _compute_background(self, grid, filter_size=31):
        import jax.scipy.signal
        window = jnp.ones((1, filter_size)) / filter_size
        bg = jax.scipy.signal.correlate2d(grid, window, mode='same')
        return jnp.maximum(bg, 1.0)

    def _build_basis_matrix(self, grid_coords, params):
        Q_x, Q_y, Q_z = grid_coords
        z_x = params[:, 1]
        z_y = params[:, 2]
        z_z = params[:, 3]
        sigmas = params[:, 4]

        def eval_great_circle(zx, zy, zz, sig):
            dots = jnp.abs(Q_x * zx + Q_y * zy + Q_z * zz)
            return jnp.exp(-(dots**2) / (2 * sig**2))

        return vmap(eval_great_circle)(z_x, z_y, z_z, sigmas).T

class AzimuthalJAXHough:
    def __init__(self, N_theta=256, N_phi=512, sigma_deg=3.0):
        self.N_theta = N_theta
        self.N_phi = N_phi
        self.sigma = np.sin(np.deg2rad(sigma_deg))
        
        # A grid of physical widths to test (e.g., 1.5 to 5.0 degrees)
        sigmas_deg = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
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

    def find_active_zones(self, grid_raw, grid_bg, max_axes=15):
        print("  > Executing Physics-Informed V-Cycle (Scout + Auto-Tuned Sniper)...")
        
        from subhkl.search.sparse_hough import GlobalZoneAxisSniper
        sniper = GlobalZoneAxisSniper(loss="gaussian", ref_sigma=self.sigma)
        
        # Combine 2D projected BG with 1D azimuthal smoothing to model all noise sources
        azimuthal_bg = sniper._compute_background(grid_bg, filter_size=31)
        
        grid_flat_raw = jnp.array(grid_raw).flatten()
        bg_flat = azimuthal_bg.flatten()
        grid_coords = (self.Q_x.flatten(), self.Q_y.flatten(), self.Q_z.flatten())

        # THE SCOUT'S TARGET: The pure, high-contrast signal
        grid_signal = jnp.maximum(jnp.array(grid_raw) - azimuthal_bg, 0.0)
        current_signal_grid = grid_signal
        
        candidates = []

        @jax.jit
        def test_candidate_widths(c_init_val, zx, zy, zz):
            def eval_sig(sig):
                p = jnp.array([[c_init_val, zx, zy, zz, sig]])
                A = sniper._build_basis_matrix(grid_coords, p)
                c_sparse, best_alpha, bic, dev = sniper.tune_and_solve(grid_flat_raw, bg_flat, A, p)
                return c_sparse[0], best_alpha, bic, dev
                
            c_vals, alphas, bics, devs = jax.vmap(eval_sig)(self.candidate_sigmas)
            best_idx = jnp.argmax(c_vals)
            return c_vals[best_idx], self.candidate_sigmas[best_idx], alphas[best_idx], bics[best_idx], devs[best_idx]
        
        for step in range(max_axes * 2):  
            H = self.transform(current_signal_grid)
            H_np = np.array(H)
            
            max_idx = np.unravel_index(np.argmax(H_np), H_np.shape)
            max_val = float(H_np[max_idx])
            
            t_idx, p_idx = max_idx
            Theta = float(self.thetas[t_idx])
            Phi = float(self.phis[p_idx])
            z_x = np.sin(Theta) * np.cos(Phi)
            z_y = np.sin(Theta) * np.sin(Phi)
            z_z = np.cos(Theta)
            
            c_init = max_val / self.N_phi 
            best_c, best_sig, chosen_alpha, bic, dev = test_candidate_widths(c_init, z_x, z_y, z_z)
            
            if float(best_c) <= 1e-3:
                print(f"    [Step {step+1:02d}] Rejected | SNR fell below noise floor.")
                break
                
            print(f"    [Step {step+1:02d}] Kept | Width: {np.degrees(np.arcsin(best_sig)):.1f} deg | Alpha: {chosen_alpha:4.1f} | BIC: {bic:.2e} | Dev/Nu: {dev:.3f}")
            
            candidates.append([float(best_c), z_x, z_y, z_z, float(best_sig)])
            
            dots = jnp.abs(self.Q_x * z_x + self.Q_y * z_y + self.Q_z * z_z)
            mask_sigma = best_sig * 2.0 
            mask = 1.0 - jnp.exp(-(dots**2) / (2 * mask_sigma**2))
            current_signal_grid = current_signal_grid * mask

        if not candidates:
            print("  > Scout hit the noise floor instantly. No candidates found.")
            return np.empty((0,3)), np.empty(0)

        print(f"  > Scout extracted {len(candidates)} valid lines. Resolving crosstalk...")

        # The Joint Sniper acts on the Raw Grid
        params_guess = jnp.array(candidates)
        A_mat_joint = sniper._build_basis_matrix(grid_coords, params_guess)
        
        c_sparse_joint = sniper.tune_and_solve(grid_flat_raw, bg_flat, A_mat_joint, params_guess)
        
        survivors = c_sparse_joint > 1e-3
        final_zones = np.array(params_guess[survivors, 1:4])
        final_weights = np.array(c_sparse_joint[survivors])
        
        order = np.argsort(final_weights)[::-1]
        return final_zones[order][:max_axes], final_weights[order][:max_axes]
