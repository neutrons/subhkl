import numpy as np
import itertools
from scipy.spatial.transform import Rotation

def davenport_q_method(V_obs, W_theo):
    """Solves Wahba's problem using Davenport's q-method."""
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

def align_empirical_zones(e_lab_obs, B_mat, max_uvw=6, angle_tol=1.5):
    """
    Hypothesis Consensus (RANSAC) Triad Matcher.
    Immunized against accidental isometries by evaluating global consensus 
    across all extracted empirical axes.
    """
    N_emp = len(e_lab_obs)
    N_use = min(3, N_emp) # We only need 3 axes to generate a Hypothesis U-Matrix
    
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

    # 2. Map theoretical zones into real-space Cartesian frame
    A_mat = np.linalg.inv(B_mat).T
    r_theo = (A_mat @ theo_zones.T).T
    r_norms = np.linalg.norm(r_theo, axis=1, keepdims=True)
    r_theo_norm = r_theo / r_norms
    N_theo = len(r_theo_norm)

    print(f"  > Compiled {N_theo} theoretical axes. Building Hash Table...")

    # 3. Precompute Angle Tables
    theo_dots = np.clip(np.abs(r_theo_norm @ r_theo_norm.T), 0.0, 1.0)
    theo_angles = np.rad2deg(np.arccos(theo_dots))

    emp_dots = np.clip(np.abs(e_use @ e_use.T), 0.0, 1.0)
    emp_angles = np.rad2deg(np.arccos(emp_dots))

    req_12 = emp_angles[0, 1]
    req_13 = emp_angles[0, 2]
    req_23 = emp_angles[1, 2]

    print(f"  > Executing Hypothesis Generation (tol={angle_tol} deg)...")

    # 4. Generate Candidate Triads (The Hypotheses)
    valid_a, valid_b = np.where(np.abs(theo_angles - req_12) < angle_tol)
    valid_mask = valid_a != valid_b
    valid_a = valid_a[valid_mask]
    valid_b = valid_b[valid_mask]

    candidates = []
    for a, b in zip(valid_a, valid_b):
        err_13_all = np.abs(theo_angles[a, :] - req_13)
        err_23_all = np.abs(theo_angles[b, :] - req_23)
        
        valid_c = np.where((err_13_all < angle_tol) & (err_23_all < angle_tol))[0]
        
        for c in valid_c:
            if c == a or c == b: continue
            # Score triad by maximum internal error
            max_err = max(np.abs(theo_angles[a,b]-req_12), err_13_all[c], err_23_all[c])
            candidates.append((max_err, a, b, c))

    if not candidates:
        raise ValueError(f"No theoretical triads match the primary empirical angles within {angle_tol} deg.")

    # Sort candidates by internal geometric tightness, take the top 100 for verification
    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:100]
    
    print(f"  > Found {len(candidates)} plausible isometries. Evaluating Global Consensus...")

    best_U = None
    best_inliers = -1
    best_residual = np.inf

    # 5. Global Consensus Verification
    for max_err, a, b, c in candidates:
        candidate_vectors = r_theo_norm[[a, b, c]]
        
        # Test all head-tail sign combinations for Laue symmetry
        for signs in itertools.product([1, -1], repeat=3):
            r_flipped = candidate_vectors * np.array(signs)[:, None]
            U, _ = davenport_q_method(e_use, r_flipped)
            
            # --- THE CONSENSUS CHECK ---
            # Project ALL theoretical axes into the lab frame using this U-matrix
            r_lab = (U @ r_theo.T).T
            r_lab_norm = r_lab / np.linalg.norm(r_lab, axis=1, keepdims=True)
            
            # Check how many of the FULL set of 15 empirical axes align with any theoretical axis
            dots = np.clip(np.abs(e_lab_obs @ r_lab_norm.T), 0.0, 1.0)
            max_dots = np.max(dots, axis=1) # Best match for each empirical axis
            angles = np.rad2deg(np.arccos(max_dots))
            
            inliers = np.sum(angles < angle_tol)
            
            if inliers > 0:
                residual = np.mean(angles[angles < angle_tol])
            else:
                residual = 999.0
                
            # Keep the U-Matrix that explains the most axes with the tightest fit
            if inliers > best_inliers or (inliers == best_inliers and residual < best_residual):
                best_inliers = inliers
                best_residual = residual
                best_U = U

    print(f"  > Consensus Achieved! U-Matrix explains {best_inliers}/{N_emp} axes (Mean Error: {best_residual:.3f} deg)")
    
    return best_U
