import numpy as np
import itertools
from scipy.spatial.transform import Rotation

def davenport_q_method(V_obs, W_theo):
    """
    Solves Wahba's problem using Davenport's q-method.
    """
    B = np.zeros((3, 3))
    for v, w in zip(V_obs, W_theo):
        B += np.outer(v, w)
        
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1,2] - B[2,1], B[2,0] - B[0,2], B[0,1] - B[1,0]])
    
    K = np.zeros((4, 4))
    K[:3, :3] = S - sigma * np.eye(3)
    K[:3, 3] = Z
    K[3, :3] = Z
    K[3, 3] = sigma
    
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    max_idx = np.argmax(eigenvalues)
    
    q = eigenvectors[:, max_idx] 
    U = Rotation.from_quat(q).as_matrix()
    
    return U, eigenvalues[max_idx]

def align_empirical_zones(e_lab_obs, B_mat, max_uvw=3, angle_tol=2.0):
    """
    Accelerated DFS Triad Matcher. Prunes combinations using pairwise angle constraints.
    """
    # 1. Generate the theoretical zones
    u_vals = np.arange(-max_uvw, max_uvw + 1)
    u, v, w = np.meshgrid(u_vals, u_vals, u_vals, indexing="ij")
    zones = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=0)
    mask = ~((zones[0] == 0) & (zones[1] == 0) & (zones[2] == 0))
    theo_zones = zones[:, mask].astype(np.float32).T  # (N_calc, 3)

    # 2. Map theoretical zones into real space: A = (B^-1)^T
    A_mat = np.linalg.inv(B_mat).T
    r_theo = (A_mat @ theo_zones.T).T
    r_norms = np.linalg.norm(r_theo, axis=1, keepdims=True)
    r_theo_norm = r_theo / r_norms
    N_theo = len(r_theo_norm)

    # 3. Take the strongest empirical axes
    N_use = min(4, len(e_lab_obs))
    if N_use < 2:
        raise ValueError("Need at least 2 empirical zone axes to lock 3D orientation.")
    
    e_use = e_lab_obs[:N_use]

    # Precompute Empirical Pairwise Angles (Abs dot for head-tail symmetry)
    emp_dots = np.clip(np.abs(e_use @ e_use.T), 0.0, 1.0)
    emp_angles = np.rad2deg(np.arccos(emp_dots))

    # Precompute Theoretical Pairwise Angles to enable instant lookup
    theo_dots = np.clip(np.abs(r_theo_norm @ r_theo_norm.T), 0.0, 1.0)
    theo_angles = np.rad2deg(np.arccos(theo_dots))

    best_match_indices = None
    best_error = np.inf

    # --- DEPTH-FIRST SEARCH (DFS) ALGORITHM ---
    def dfs(current_depth, current_assignment):
        nonlocal best_match_indices, best_error
        
        # Base Case: We successfully assigned a theoretical axis to every empirical axis
        if current_depth == N_use:
            # Re-verify the maximum error of the completed set
            max_err = 0.0
            for i in range(N_use):
                for j in range(i+1, N_use):
                    r_i, r_j = current_assignment[i], current_assignment[j]
                    err = np.abs(emp_angles[i, j] - theo_angles[r_i, r_j])
                    max_err = max(max_err, err)
            
            if max_err < best_error:
                best_error = max_err
                best_match_indices = list(current_assignment)
            return

        # Recursive Step: Find candidates for the next empirical axis
        for r_cand in range(N_theo):
            # To avoid degenerate combinations, ensure we don't pick the exact same physical axis twice
            if r_cand in current_assignment:
                continue

            # Verify this candidate against ALL previously assigned axes
            is_valid = True
            for prev_depth, r_prev in enumerate(current_assignment):
                required_angle = emp_angles[prev_depth, current_depth]
                actual_angle = theo_angles[r_prev, r_cand]
                
                # If the candidate violates the angle constraint, prune the entire branch instantly
                if np.abs(required_angle - actual_angle) > angle_tol:
                    is_valid = False
                    break
            
            # If the candidate survived all cross-checks, descend into the next depth
            if is_valid:
                current_assignment.append(r_cand)
                dfs(current_depth + 1, current_assignment)
                current_assignment.pop() # Backtrack

    # Kick off the search (Starting at depth 0 with an empty assignment)
    print(f"  > Starting accelerated DFS Triad Match (max_uvw={max_uvw}, N_theo={N_theo})...")
    dfs(0, [])

    if best_match_indices is None:
        raise ValueError(f"Could not find a theoretical triad matching the empirical angles within {angle_tol} deg.")

    print(f"  > Triad Match Found! Max Angle Error: {best_error:.3f} deg")

    # Extract the matched theoretical vectors
    best_match = r_theo_norm[best_match_indices]

    # --- DAVENPORT SOLVER (Sign Permutations) ---
    best_U = None
    best_score = -np.inf
    
    # Test all 2^N sign combinations to establish the right-handed frame
    for signs in itertools.product([1, -1], repeat=N_use):
        r_flipped = best_match * np.array(signs)[:, None]
        U, score = davenport_q_method(e_use, r_flipped)
        if score > best_score:
            best_score = score
            best_U = U

    return best_U
