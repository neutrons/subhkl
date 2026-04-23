import numpy as np
import itertools
from scipy.spatial.transform import Rotation

def align_empirical_zones(e_lab_obs, B_mat, max_uvw=10, angle_tol=1.5):
    """
    Fully Vectorized Hypothesis Consensus (RANSAC) Triad Matcher.
    Generates and evaluates tens of thousands of U-matrices simultaneously 
    via tensor contractions.
    """
    N_emp = len(e_lab_obs)
    N_use = min(3, N_emp)
    
    if N_use < 3:
        raise ValueError("Need at least 3 empirical zone axes to lock 3D orientation.")
    
    e_use = e_lab_obs[:N_use]

    # 1. Generate Theoretical Dictionary
    print(f"  > Generating Theoretical Dictionary (max_uvw={max_uvw})...")
    u_vals = np.arange(-max_uvw, max_uvw + 1)
    u, v, w = np.meshgrid(u_vals, u_vals, u_vals, indexing="ij")
    zones = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=0)
    mask = ~((zones[0] == 0) & (zones[1] == 0) & (zones[2] == 0))
    theo_zones = zones[:, mask].astype(np.float32).T 

    A_mat = np.linalg.inv(B_mat).T
    r_theo = (A_mat @ theo_zones.T).T
    r_norms = np.linalg.norm(r_theo, axis=1, keepdims=True)
    r_theo_norm = r_theo / r_norms
    N_theo = len(r_theo_norm)

    print(f"  > Compiled {N_theo} theoretical axes. Building Tensor Hash Table...")

    theo_dots = np.clip(np.abs(r_theo_norm @ r_theo_norm.T), 0.0, 1.0)
    theo_angles = np.rad2deg(np.arccos(theo_dots))

    emp_dots = np.clip(np.abs(e_use @ e_use.T), 0.0, 1.0)
    emp_angles = np.rad2deg(np.arccos(emp_dots))

    req_12, req_13, req_23 = emp_angles[0, 1], emp_angles[0, 2], emp_angles[1, 2]

    print(f"  > Executing Vectorized Hypothesis Generation (tol={angle_tol} deg)...")

    # 2. Vectorized Triad Generation
    valid_a, valid_b = np.where(np.abs(theo_angles - req_12) < angle_tol)
    valid_mask = valid_a != valid_b
    valid_a, valid_b = valid_a[valid_mask], valid_b[valid_mask]

    # Find all 'c' candidates matching 'a' and 'b' simultaneously
    match_13 = np.abs(theo_angles[valid_a, :] - req_13) < angle_tol
    match_23 = np.abs(theo_angles[valid_b, :] - req_23) < angle_tol
    
    pair_idx, c_idx = np.where(match_13 & match_23)
    a_cand = valid_a[pair_idx]
    b_cand = valid_b[pair_idx]

    # Filter out identical axes
    triplet_mask = (c_idx != a_cand) & (c_idx != b_cand)
    a_cand, b_cand, c_cand = a_cand[triplet_mask], b_cand[triplet_mask], c_idx[triplet_mask]

    if len(a_cand) == 0:
        raise ValueError(f"No theoretical triads match the primary empirical angles within {angle_tol} deg.")

    # Calculate Max Geometric Error for sorting
    err12 = np.abs(theo_angles[a_cand, b_cand] - req_12)
    err13 = np.abs(theo_angles[a_cand, c_cand] - req_13)
    err23 = np.abs(theo_angles[b_cand, c_cand] - req_23)
    max_errs = np.maximum(err12, np.maximum(err13, err23))

    # We safely extract the top 2000 triads without loop bottleneck fears!
    max_candidates = 2000
    sort_idx = np.argsort(max_errs)[:max_candidates]
    a_cand, b_cand, c_cand = a_cand[sort_idx], b_cand[sort_idx], c_cand[sort_idx]
    
    N_cand = len(a_cand)
    
    # 3. Vectorized Sign Permutations
    r_cand = r_theo_norm[np.column_stack([a_cand, b_cand, c_cand])] # Shape: (N_cand, 3, 3)
    signs = np.array(list(itertools.product([1, -1], repeat=3))) # Shape: (8, 3)
    
    # Broadcast multiply: (N_cand, 1, 3_vec, 3_xyz) * (1, 8, 3_vec, 1) -> (N_cand, 8, 3, 3)
    W_batch = (r_cand[:, None, :, :] * signs[None, :, :, None]).reshape(-1, 3, 3)
    N_hyp = W_batch.shape[0] # N_cand * 8
    
    print(f"  > Found {N_cand} geometric triads. Evaluating {N_hyp} total U-Matrix permutations...")

    # 4. Vectorized Davenport Q-Method (Evaluates all N_hyp matrices at once)
    V_obs = e_use # Shape: (3, 3)
    B_batch = np.einsum('ax,nay->nxy', V_obs, W_batch)
    S_batch = B_batch + np.transpose(B_batch, (0, 2, 1))
    sigma_batch = np.trace(B_batch, axis1=1, axis2=2)
    
    Z_batch = np.column_stack([
        B_batch[:, 1, 2] - B_batch[:, 2, 1], 
        B_batch[:, 2, 0] - B_batch[:, 0, 2], 
        B_batch[:, 0, 1] - B_batch[:, 1, 0]
    ])
    
    K_batch = np.zeros((N_hyp, 4, 4))
    K_batch[:, :3, :3] = S_batch - sigma_batch[:, None, None] * np.eye(3)
    K_batch[:, :3, 3] = Z_batch
    K_batch[:, 3, :3] = Z_batch
    K_batch[:, 3, 3] = sigma_batch
    
    # Solve Eigenvalues in parallel (the max eigenvalue is always the last index in eigh)
    evals, evecs = np.linalg.eigh(K_batch)
    q_batch = evecs[:, :, -1] 
    U_batch = Rotation.from_quat(q_batch).as_matrix() # Shape: (N_hyp, 3, 3)

    # 5. Batched Global Consensus Evaluation
    # To prevent multi-GB RAM explosions on 16k permutations, we chunk the dot products.
    best_U = None
    best_inliers = -1
    best_residual = np.inf
    chunk_size = 1000
    
    for i in range(0, N_hyp, chunk_size):
        U_chunk = U_batch[i:i+chunk_size]
        
        # Project: U @ r_theo.T -> (chunk, 3, 3) @ (3, N_theo) -> (chunk, 3, N_theo)
        r_lab_chunk = np.matmul(U_chunk, r_theo_norm.T)
        
        # Evaluate: e_lab @ r_lab -> (N_emp, 3) @ (chunk, 3, N_theo) -> (chunk, N_emp, N_theo)
        dots_chunk = np.einsum('ei,cit->cet', e_lab_obs, r_lab_chunk)
        
        max_dots = np.max(np.clip(np.abs(dots_chunk), 0.0, 1.0), axis=2)
        angles = np.rad2deg(np.arccos(max_dots))
        
        inliers = np.sum(angles < angle_tol, axis=1)
        residuals = np.sum(np.where(angles < angle_tol, angles, 0.0), axis=1) / np.maximum(inliers, 1)
        
        # Rank by Inlier Count, then by Tightest Fit
        score = inliers - (residuals / 1000.0) 
        best_in_chunk = np.argmax(score)
        
        if inliers[best_in_chunk] > best_inliers or (inliers[best_in_chunk] == best_inliers and residuals[best_in_chunk] < best_residual):
            best_inliers = inliers[best_in_chunk]
            best_residual = residuals[best_in_chunk]
            best_U = U_chunk[best_in_chunk]

    if best_inliers < 3:
        raise ValueError(f"Consensus failed. Best U-matrix only explained {best_inliers} axes.")

    print(f"  > Consensus Achieved! U-Matrix explains {best_inliers}/{N_emp} axes (Mean Error: {best_residual:.3f} deg)")
    
    return best_U
