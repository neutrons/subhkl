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
    Ultra-Fast Branch & Bound Vectorized Matcher. 
    Bypasses combinatorics by dynamically shrinking the search radius.
    """
    N_use = min(4, len(e_lab_obs))
    if N_use < 3:
        raise ValueError("Need at least 3 empirical zone axes to lock 3D orientation.")
    
    e_use = e_lab_obs[:N_use]

    # 1. Generate the theoretical zones
    print(f"  > Generating Theoretical Dictionary (max_uvw={max_uvw})...")
    u_vals = np.arange(-max_uvw, max_uvw + 1)
    u, v, w = np.meshgrid(u_vals, u_vals, u_vals, indexing="ij")
    zones = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=0)
    mask = ~((zones[0] == 0) & (zones[1] == 0) & (zones[2] == 0))
    theo_zones = zones[:, mask].astype(np.float32).T 

    # 2. Map theoretical zones into real space: r = (B^-1)^T * z
    A_mat = np.linalg.inv(B_mat).T
    r_theo = (A_mat @ theo_zones.T).T
    r_norms = np.linalg.norm(r_theo, axis=1, keepdims=True)
    r_theo_norm = r_theo / r_norms
    N_theo = len(r_theo_norm)

    print(f"  > Compiled {N_theo} theoretical axes. Building Hash Table...")

    # 3. Precompute the FULL N x N theoretical angle lookup table
    theo_dots = np.clip(np.abs(r_theo_norm @ r_theo_norm.T), 0.0, 1.0)
    theo_angles = np.rad2deg(np.arccos(theo_dots))

    # 4. Extract target empirical angles
    emp_dots = np.clip(np.abs(e_use @ e_use.T), 0.0, 1.0)
    emp_angles = np.rad2deg(np.arccos(emp_dots))

    req_12 = emp_angles[0, 1]
    req_13 = emp_angles[0, 2]
    req_23 = emp_angles[1, 2]
    
    if N_use == 4:
        req_14 = emp_angles[0, 3]
        req_24 = emp_angles[1, 3]
        req_34 = emp_angles[2, 3]

    print(f"  > Executing Branch & Bound Edge Match (tol={angle_tol} deg)...")

    # The Baseline Edge: Extract all pairs matching E1-E2
    valid_a, valid_b = np.where(np.abs(theo_angles - req_12) < angle_tol)
    valid_mask = valid_a != valid_b
    valid_a = valid_a[valid_mask]
    valid_b = valid_b[valid_mask]

    # Sort the baseline pairs by their exactness so the best candidates are tested first
    err_12 = np.abs(theo_angles[valid_a, valid_b] - req_12)
    sort_idx = np.argsort(err_12)
    valid_a = valid_a[sort_idx]
    valid_b = valid_b[sort_idx]
    err_12 = err_12[sort_idx]

    best_error = angle_tol
    best_indices = None

    for a, b, e12 in zip(valid_a, valid_b, err_12):
        # Branch & Bound Cutoff: We can never beat the current best error!
        if e12 >= best_error:
            break
            
        err_13_all = np.abs(theo_angles[a, :] - req_13)
        err_23_all = np.abs(theo_angles[b, :] - req_23)
        
        # Dynamically restrict valid C candidates using the shrinking best_error
        valid_c = np.where((err_13_all < best_error) & (err_23_all < best_error))[0]
        
        for c in valid_c:
            if c == a or c == b: continue
            
            e13 = err_13_all[c]
            e23 = err_23_all[c]
            current_max = max(e12, e13, e23)
            
            if current_max >= best_error:
                continue
                
            if N_use == 3:
                # We found a new best Triad! Shrink the bounds.
                best_error = current_max
                best_indices = [a, b, c]
            else:
                err_14_all = np.abs(theo_angles[a, :] - req_14)
                err_24_all = np.abs(theo_angles[b, :] - req_24)
                err_34_all = np.abs(theo_angles[c, :] - req_34)
                
                valid_d = np.where((err_14_all < best_error) & 
                                   (err_24_all < best_error) & 
                                   (err_34_all < best_error))[0]
                                   
                for d in valid_d:
                    if d in (a, b, c): continue
                    
                    e14 = err_14_all[d]
                    e24 = err_24_all[d]
                    e34 = err_34_all[d]
                    
                    tetrad_max = max(current_max, e14, e24, e34)
                    
                    if tetrad_max < best_error:
                        # We found a new best Tetrad! Shrink the bounds.
                        best_error = tetrad_max
                        best_indices = [a, b, c, d]

    if best_indices is None:
        raise ValueError(f"Could not find a theoretical triad matching within {angle_tol} deg.")

    print(f"  > Match Found! Max Angle Error: {best_error:.3f} deg")

    # 5. Evaluate Davenport just ONCE on the absolute best geometric match
    best_match = r_theo_norm[best_indices]
    best_U = None
    best_score = -np.inf
    
    for signs in itertools.product([1, -1], repeat=N_use):
        r_flipped = best_match * np.array(signs)[:, None]
        U, score = davenport_q_method(e_use, r_flipped)
        if score > best_score:
            best_score = score
            best_U = U

    return best_U
