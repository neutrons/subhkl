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

class GlobalZoneAxisSniper(SparseBasisPursuit):
    def __init__(self, alpha=30.0, gamma=1.0, loss="poisson", ref_sigma=0.026, 
                 auto_tune_alpha=True, candidate_alphas=None):
        
        default_alphas = candidate_alphas or [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
        
        super().__init__(
            alpha=alpha, gamma=gamma, loss=loss, ref_sigma=ref_sigma, 
            auto_tune_alpha=auto_tune_alpha, candidate_alphas=default_alphas
        )

    def _compute_background(self, grid, filter_size=31):
        from subhkl.search.sparse_rbf import jax_median_2d
        bg = jax_median_2d(grid, window_size=filter_size)
        return jnp.maximum(bg, 1.0)

    def _build_basis_matrix(self, grid_coords, params):
        Q_x, Q_y, Q_z = grid_coords
        
        # Unpack the 8 geometric parameters
        z_x, z_y, z_z = params[:, 1], params[:, 2], params[:, 3]
        sigmas = params[:, 4]
        vc_x, vc_y, vc_z = params[:, 5], params[:, 6], params[:, 7]
        kappas = params[:, 8]

        def eval_windowed_circle(zx, zy, zz, sig, vcx, vcy, vcz, kappa):
            # 1. Great Circle Width (Gaussian Cross-Section)
            dots_z = jnp.abs(Q_x * zx + Q_y * zy + Q_z * zz)
            width_mask = jnp.exp(-(dots_z**2) / (2 * sig**2))
            
            # 2. Arc Length Window (Von Mises Azimuthal Distribution)
            # dots_v is the cosine of the angle along the arc from the center point vc
            dots_v = Q_x * vcx + Q_y * vcy + Q_z * vcz
            
            # If kappa == 0, exp(0) == 1.0 (Full 360-degree Circle)
            # If kappa > 0, it decays rapidly away from the arc center!
            arc_mask = jnp.exp(kappa * (dots_v - 1.0))
            
            return width_mask * arc_mask

        return jax.vmap(eval_windowed_circle)(z_x, z_y, z_z, sigmas, vc_x, vc_y, vc_z, kappas).T

    @partial(jax.jit, static_argnames=['self'])
    def tune_and_solve(self, data_flat, bg_flat, A_matrix, params_guess):
        # Explicitly grab sigma from index 4 to calculate the Besov Space Weight!
        sigmas = params_guess[:, 4]
        weights = (sigmas / self.ref_sigma) ** self.gamma
        
        def evaluate_alpha(alpha_val):
            # The penalty scales with the physical width of the Great Circle
            alpha_vec = alpha_val * weights
            
            from subhkl.search.ssn import solve_ssn_unified
            c_sparse = solve_ssn_unified(
                A_matrix, data_flat, bg_flat, alpha_vec,
                1, params_guess[:, 0], max_iter=20, force_target=False
            )
            
            k_active = jnp.sum(c_sparse > 1e-9)
            recon_total = jnp.maximum(A_matrix @ c_sparse + bg_flat, 1e-9)
            
            # Poisson Deviance
            term = jax.scipy.special.xlogy(data_flat, data_flat / recon_total) - (data_flat - recon_total)
            dev = 2 * jnp.sum(term)
            
            # BIC Calculation
            n_pix = data_flat.size
            n_params = k_active * params_guess.shape[1]
            bic = jnp.where(k_active == 0, 1e9, n_params * jnp.log(n_pix) + dev)
            
            return bic, c_sparse, dev
            
        bics, all_c_sparse, devs = jax.vmap(evaluate_alpha)(jnp.array(self.candidate_alphas))
        best_idx = jnp.argmin(bics)
        
        return all_c_sparse[best_idx], self.candidate_alphas[best_idx], bics[best_idx], devs[best_idx]

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

    def find_active_zones(self, grid_raw, max_axes=15):
        print("  > Executing Multi-Scale Joint Dictionary Pursuit (Besov Torus)...")
        from subhkl.search.sparse_hough import GlobalZoneAxisSniper
        from scipy.ndimage import maximum_filter
        
        # We use the narrowest width as the normalization reference for the Besov weights
        ref_sig = float(self.candidate_sigmas[0])
        sniper = GlobalZoneAxisSniper(loss="poisson", gamma=1.0, ref_sigma=ref_sig)
        
        grid_flat_raw = jnp.array(grid_raw).flatten()
        bg_flat = sniper._compute_background(grid_raw, filter_size=31).flatten()
        grid_coords = (self.Q_x.flatten(), self.Q_y.flatten(), self.Q_z.flatten())
        grid_signal_base = jnp.maximum(jnp.array(grid_raw) - bg_flat.reshape(self.N_theta, self.N_phi), 0.0)
        
        H_np = np.array(self.transform(grid_signal_base))
        local_max = (H_np == maximum_filter(H_np, size=11))
        valid_peaks = local_max & (H_np > np.max(H_np) * 0.05)
        t_coords, p_coords = np.where(valid_peaks)
        
        intensities = H_np[t_coords, p_coords]
        order = np.argsort(intensities)[::-1]
        t_coords = t_coords[order][:40] 
        p_coords = p_coords[order][:40]
        
        dictionary_params = []
        
        print(f"  > Scout identified {len(t_coords)} spatial hubs. Generating Multi-Scale frame...")
        
        for t, p in zip(t_coords, p_coords):
            Theta = float(self.thetas[t])
            Phi = float(self.phis[p])
            z_x = np.sin(Theta) * np.cos(Phi)
            z_y = np.sin(Theta) * np.sin(Phi)
            z_z = np.cos(Theta)
            
            # DYNAMIC ARC EXTRACTION
            dots_z = np.abs(self.Q_x * z_x + self.Q_y * z_y + self.Q_z * z_z)
            ribbon_mask = dots_z < np.sin(np.deg2rad(3.0))
            I_ribbon = grid_signal_base * ribbon_mask
            sum_I = np.sum(I_ribbon)
            
            if sum_I > 0:
                mu_x = np.sum(I_ribbon * self.Q_x) / sum_I
                mu_y = np.sum(I_ribbon * self.Q_y) / sum_I
                mu_z = np.sum(I_ribbon * self.Q_z) / sum_I
                R_len = np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
                
                dot_mu_z = mu_x*z_x + mu_y*z_y + mu_z*z_z
                vc_x, vc_y, vc_z = mu_x - dot_mu_z*z_x, mu_y - dot_mu_z*z_y, mu_z - dot_mu_z*z_z
                norm_vc = np.sqrt(vc_x**2 + vc_y**2 + vc_z**2)
                
                if norm_vc > 0:
                    vc_x, vc_y, vc_z = vc_x/norm_vc, vc_y/norm_vc, vc_z/norm_vc
                else:
                    vc_x, vc_y, vc_z = 1.0, 0.0, 0.0
                    
                kappa = 0.0 if R_len < 0.1 else min(20.0, R_len / (1.0 - min(R_len, 0.99)))
            else:
                vc_x, vc_y, vc_z, kappa = 1.0, 0.0, 0.0, 0.0
                
            # MULTI-SCALE DICTIONARY: Append a candidate for every possible physical width!
            for sig in self.candidate_sigmas:
                c_init = float(H_np[t, p] / self.N_phi)
                dictionary_params.append([c_init, z_x, z_y, z_z, float(sig), vc_x, vc_y, vc_z, kappa])
            
        print(f"  > Executing Joint Poisson Basis Pursuit on {len(dictionary_params)} multi-scale vectors...")
        p_guess = jnp.array(dictionary_params)
        A_mat = sniper._build_basis_matrix(grid_coords, p_guess)
        
        c_sparse, best_alpha, bic, dev = sniper.tune_and_solve(grid_flat_raw, bg_flat, A_mat, p_guess)
        
        survivors = c_sparse > 1e-3
        n_survivors = int(np.sum(survivors))
        
        print(f"  > Convergence | Alpha: {best_alpha:4.1f} | BIC: {bic:.2e} | Dev/Nu: {dev:.3f}")
        print(f"  > Besov Regularizer eliminated {len(dictionary_params) - n_survivors} ghost scales. Kept {n_survivors} true axes.")
        
        if n_survivors == 0:
            return np.empty((0,3)), np.empty(0)
            
        active_candidates = np.array(dictionary_params)[survivors]
        final_zones = active_candidates[:, 1:4]
        final_weights = np.array(c_sparse[survivors])
        
        # There might be multiple valid scales surviving for the SAME normal vector. 
        # We find unique spatial normals to hand to the Combinatorial Davenport solver.
        unique_zones, unique_indices = np.unique(np.round(final_zones, 4), axis=0, return_index=True)
        unique_weights = final_weights[unique_indices]
        
        order = np.argsort(unique_weights)[::-1]
        return unique_zones[order][:max_axes], unique_weights[order][:max_axes]
