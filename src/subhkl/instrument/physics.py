import numpy as np
import numpy.typing as npt
from subhkl.core.crystallography import get_q_lab


def scale_coordinates(xp, yp, scale_x, scale_y, nx, ny):
    """
    Scale pixel coordinates to physical coordinates.

    Parameters
    ----------
    xp : float
        Pixel x-coordinate
    yp : float
        Pixel y-coordinate
    scale_x : float
        Scale factor in x direction (m/pixel)
    scale_y : float
        Scale factor in y direction (m/pixel)
    nx : int
        Number of pixels in x direction
    ny : int
        Number of pixels in y direction

    Returns
    -------
    x, y : tuple of float
        Physical coordinates in meters
    """
    x = (xp - nx / 2) * scale_x
    y = (yp - ny / 2) * scale_y
    return x, y


def predict_reflections_on_panel(
    detector,
    h,
    k,
    l,
    UB,
    wavelength_min,
    wavelength_max,
    sample_offset=None,
    ki_vec=None,
    R_all=None,
    gonio_axes=None,
    gonio_angles=None,
    gonio_offsets=None,   # <-- NEW
):
    """
    Predict reflection positions on a specific detector panel.
    Expects UB to map the crystal to the sample frame.
    Expects h, k, l to be 1D arrays of length M.
    """
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])

    ki_hat = ki_vec / np.linalg.norm(ki_vec)
    hkl = np.stack([h, k, l], axis=0)  # (3, M)

    # 1. Transform Miller indices to Sample frame Q-vectors
    q_sample = UB @ hkl  # (3, M)

    # 2. Transform Sample frame Q-vectors to absolute Lab frame Q-vectors
    if gonio_axes is not None and gonio_angles is not None:
        from subhkl.instrument.goniometer import sample_to_lab
        offsets = sample_offset
        if offsets is not None and offsets.ndim == 1:
            offsets_full = np.zeros((len(gonio_axes), 3))
            offsets_full[-1] = offsets
        elif offsets is None:
            offsets_full = np.zeros((len(gonio_axes), 3))
        else:
            offsets_full = offsets

        # Lab frame Q-vectors (is_vector=True skips translations)
        q_lab_T = sample_to_lab(
            q_sample.T, 
            gonio_axes, 
            gonio_angles, 
            offsets_full, 
            zero_offsets=gonio_offsets, 
            is_vector=True
        )
        q_lab = q_lab_T.T
        
        # True Sample Ray Origin (is_vector=False applies translations)
        s_lab = sample_to_lab(
            np.array([0.0, 0.0, 0.0]), 
            gonio_axes, 
            gonio_angles, 
            offsets_full, 
            zero_offsets=gonio_offsets,
            is_vector=False
        )
    else:
        # Legacy fallback
        s = sample_offset if sample_offset is not None else np.zeros(3)
        s_lab = R_all @ s if R_all is not None else s
        q_lab = R_all @ q_sample if R_all is not None else q_sample

    Q_vec = 2 * np.pi * q_lab  # (3, M)
    Q_sq = np.sum(Q_vec**2, axis=0)  # (M,)
    Q_dot_ki = np.sum(Q_vec * ki_hat[:, None], axis=0)  # (M,)

    # 3. Solve Laue Condition for Wavelength
    with np.errstate(divide="ignore", invalid="ignore"):
        lamda = -4 * np.pi * Q_dot_ki / Q_sq  # (M,)

    mask_wl = (lamda >= wavelength_min) & (lamda <= wavelength_max) & np.isfinite(lamda)

    if not np.any(mask_wl):
        return (
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([])
        )

    lamda_v = lamda[mask_wl]
    Q_vec_v = Q_vec[:, mask_wl]
    h_v, k_v, l_v = h[mask_wl], k[mask_wl], l[mask_wl]

    # 4. Calculate Final Scattered Ray Direction (kf)
    k_mag = 2 * np.pi / lamda_v
    kf_vec = Q_vec_v + k_mag * ki_hat[:, None]  # (3, M_v)
    kf_dir = kf_vec / np.linalg.norm(kf_vec, axis=0, keepdims=True)  # (3, M_v)

    # 5. Intersect with Panel
    mask_panel, row, col = detector.reflections_mask(
        kf_dir[0], kf_dir[1], kf_dir[2], sample_offset=s_lab
    )

    return (
        row[mask_panel], col[mask_panel],
        h_v[mask_panel], k_v[mask_panel], l_v[mask_panel],
        lamda_v[mask_panel],
    )

def calculate_angular_error(
    xyz_det: npt.NDArray,
    h: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,
    lam: npt.NDArray,
    UB: npt.NDArray,            # <-- Changed from RUB
    sample_offset: npt.NDArray = None,
    ki_vec: npt.NDArray = None,
    R_all: npt.NDArray = None,
    gonio_axes: npt.NDArray = None,
    gonio_angles: npt.NDArray = None,
    gonio_offsets: npt.NDArray = None, # <-- NEW
):
    if sample_offset is None:
        sample_offset = np.zeros(3)
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])

    # 1. Transform Miller indices to pristine Sample frame Q-vectors
    hkl = np.stack([h, k, l], axis=0)
    q_sample_calc = (UB @ hkl).T  # Shape: (N, 3)

    if gonio_axes is not None and gonio_angles is not None:
        offsets = sample_offset
        if offsets.ndim == 1:
            offsets_full = np.zeros((len(gonio_axes), 3))
            offsets_full[-1] = offsets
        else:
            offsets_full = offsets
            
        angles = np.tile(gonio_angles, (len(xyz_det), 1)) if gonio_angles.ndim == 1 else gonio_angles

        from subhkl.instrument.goniometer import sample_to_lab
        
        # Q-vectors only rotate (is_vector=True bypasses translations)
        q_lab_calc = sample_to_lab(
            q_sample_calc, 
            gonio_axes, 
            angles.T, 
            offsets_full, 
            zero_offsets=gonio_offsets,
            is_vector=True
        )

        # Sample origin translates and rotates (is_vector=False applies translations)
        s_lab = sample_to_lab(
            np.zeros((len(xyz_det), 3)), 
            gonio_axes, 
            angles.T, 
            offsets_full, 
            zero_offsets=gonio_offsets,
            is_vector=False
        )
        
        v = xyz_det - s_lab
    elif R_all is not None:
        # Legacy static fallback
        if R_all.ndim == 3:
            q_lab_calc = np.einsum("nij,nj->ni", R_all, q_sample_calc)
            s_lab = np.einsum("nij,j->ni", R_all, sample_offset)
        else:
            q_lab_calc = q_sample_calc @ R_all.T
            s_lab = R_all @ sample_offset
        v = xyz_det - s_lab
    else:
        q_lab_calc = q_sample_calc
        v = xyz_det - sample_offset

    # Normalize Q for angular comparison
    q_calc_norm = q_lab_calc / np.linalg.norm(q_lab_calc, axis=1, keepdims=True)

    dist = np.linalg.norm(v, axis=1, keepdims=True)
    kf_dir = v / dist  # Unit vector pointing from sample to pixel
    ki_dir = ki_vec / np.linalg.norm(ki_vec)

    # Scattering vector direction matches delta_k = kf - ki
    delta_k = kf_dir - ki_dir[None, :]
    two_sin_theta = np.linalg.norm(delta_k, axis=1)

    # Q_obs direction
    with np.errstate(divide="ignore", invalid="ignore"):
        q_obs_norm = delta_k / two_sin_theta[:, None]
        q_obs_norm = np.nan_to_num(q_obs_norm, nan=0.0)

    # 3. Angular Error (Angle between Q_calc direction and Q_obs direction)
    dot = np.sum(q_obs_norm * q_calc_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_err = np.rad2deg(np.arccos(dot))

    # 4. D-Spacing Error
    with np.errstate(divide="ignore", invalid="ignore"):
        d_obs = np.divide(lam, two_sin_theta)

    q_lab_mag = np.linalg.norm(q_lab_calc, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_calc = np.divide(1.0, q_lab_mag)

    d_err = np.abs(d_obs - d_calc)

    return d_err, ang_err
