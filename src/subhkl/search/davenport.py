import itertools
import numpy as np
from scipy.spatial.transform import Rotation

def davenport_q_method(V_obs, W_theo):
    """
    Solves Wahba's problem using Davenport's q-method.
    Finds the optimal rotation matrix U such that U @ W_theo[i] ~ V_obs[i].
    
    V_obs: (N, 3) Empirical lab-frame vectors
    W_theo: (N, 3) Theoretical crystal-frame vectors
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
    
    # The eigenvector corresponding to the largest eigenvalue is the optimal quaternion.
    # SciPy's Rotation class natively expects the [x, y, z, w] convention returned by K.
    q = eigenvectors[:, max_idx] 
    U = Rotation.from_quat(q).as_matrix()
    
    return U, eigenvalues[max_idx]

def align_empirical_zones(e_lab_obs, B_mat, max_uvw=1, angle_tol=2.0):
    """
    Combinatorial Triad Matcher. Assigns empirical zone axes to theoretical 
    integer indices and uses Davenport to return the macroscopic orientation.
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

    # 3. Take the strongest 3 to 4 empirical axes returned by SSN
    N_use = min(4, len(e_lab_obs))
    if N_use < 2:
        raise ValueError("Need at least 2 empirical zone axes to lock 3D orientation.")
    
    e_use = e_lab_obs[:N_use]

    # 4. Compute empirical pairwise angles
    e_angles = []
    for i in range(N_use):
        for j in range(i+1, N_use):
            # Abs limits to [0, 90] for head-tail symmetric zones
            dot = np.clip(np.abs(np.dot(e_use[i], e_use[j])), 0.0, 1.0)
            e_angles.append(np.rad2deg(np.arccos(dot)))
    e_angles = np.array(e_angles)

    best_match = None
    best_error = np.inf

    # 5. Brute force permutations (With max_uvw=1, 13 axes -> ~1716 iterations. Runs in ms.)
    for idxs in itertools.permutations(range(len(r_theo_norm)), N_use):
        r_use = r_theo_norm[list(idxs)]
        r_angles = []
        for i in range(N_use):
            for j in range(i+1, N_use):
                dot = np.clip(np.abs(np.dot(r_use[i], r_use[j])), 0.0, 1.0)
                r_angles.append(np.rad2deg(np.arccos(dot)))
        
        error = np.max(np.abs(e_angles - np.array(r_angles)))
        if error < angle_tol and error < best_error:
            best_error = error
            best_match = r_use

    if best_match is None:
        raise ValueError(f"Could not find a theoretical triad matching the empirical angles within {angle_tol} deg.")

    print(f"  > Triad Match Found! Max Angle Error: {best_error:.3f} deg")

    # 6. Davenport solver with sign flips
    # Since we used abs(dot) to match angles, the assigned theoretical vectors might 
    # be facing backward relative to the empirical vectors. We test all sign combinations 
    # and let Davenport's correlation score pick the right-handed frame.
    best_U = None
    best_score = -np.inf
    for signs in itertools.product([1, -1], repeat=N_use):
        r_flipped = best_match * np.array(signs)[:, None]
        U, score = davenport_q_method(e_use, r_flipped)
        if score > best_score:
            best_score = score
            best_U = U

    return best_U
