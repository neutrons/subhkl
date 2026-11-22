import numpy as np
import numpy.typing as npt


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
    gonio_offsets=None,  # <-- NEW
    extra_rot=None,
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
            is_vector=True,
        )
        q_lab = q_lab_T.T

        # True Sample Ray Origin (is_vector=False applies translations)
        s_lab = sample_to_lab(
            np.array([0.0, 0.0, 0.0]),
            gonio_axes,
            gonio_angles,
            offsets_full,
            zero_offsets=gonio_offsets,
            is_vector=False,
        )
    else:
        # Legacy fallback
        s = sample_offset if sample_offset is not None else np.zeros(3)
        s_lab = R_all @ s if R_all is not None else s
        q_lab = R_all @ q_sample if R_all is not None else q_sample

    if extra_rot is not None:
        # Fourier rocking: steers the diffraction directions in the lab
        # frame and leaves the sample origin untouched.
        q_lab = np.asarray(extra_rot) @ q_lab

    Q_vec = 2 * np.pi * q_lab  # (3, M)
    Q_sq = np.sum(Q_vec**2, axis=0)  # (M,)
    Q_dot_ki = np.sum(Q_vec * ki_hat[:, None], axis=0)  # (M,)

    # 3. Solve Laue Condition for Wavelength
    with np.errstate(divide="ignore", invalid="ignore"):
        lamda = -4 * np.pi * Q_dot_ki / Q_sq  # (M,)

    mask_wl = (lamda >= wavelength_min) & (lamda <= wavelength_max) & np.isfinite(lamda)

    if not np.any(mask_wl):
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
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
        row[mask_panel],
        col[mask_panel],
        h_v[mask_panel],
        k_v[mask_panel],
        l_v[mask_panel],
        lamda_v[mask_panel],
    )


def calculate_angular_error(
    xyz_det: npt.NDArray,
    h: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,
    lam: npt.NDArray,
    UB: npt.NDArray,  # <-- Changed from RUB
    sample_offset: npt.NDArray = None,
    ki_vec: npt.NDArray = None,
    R_all: npt.NDArray = None,
    gonio_axes: npt.NDArray = None,
    gonio_angles: npt.NDArray = None,
    gonio_offsets: npt.NDArray = None,  # <-- NEW
    per_run_trans: npt.NDArray = None,
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

        angles = (
            np.tile(gonio_angles, (len(xyz_det), 1))
            if gonio_angles.ndim == 1
            else gonio_angles
        )

        from subhkl.instrument.goniometer import sample_to_lab

        # Q-vectors only rotate (is_vector=True bypasses translations)
        q_lab_calc = sample_to_lab(
            q_sample_calc,
            gonio_axes,
            angles.T,
            offsets_full,
            zero_offsets=gonio_offsets,
            is_vector=True,
        )

        # Sample origin translates and rotates (is_vector=False applies translations)
        s_init = (
            np.asarray(per_run_trans, dtype=float)
            if per_run_trans is not None
            else np.zeros((len(xyz_det), 3))
        )
        s_lab = sample_to_lab(
            s_init,
            gonio_axes,
            angles.T,
            offsets_full,
            zero_offsets=gonio_offsets,
            is_vector=False,
        )

        v = xyz_det - s_lab
    elif R_all is not None:
        # Legacy static fallback
        s_eff = sample_offset[-1] if sample_offset.ndim == 2 else sample_offset
        if R_all.ndim == 3:
            q_lab_calc = np.einsum("nij,nj->ni", R_all, q_sample_calc)
            if per_run_trans is not None:
                s_lab = np.einsum("nij,nj->ni", R_all, s_eff[None, :] + per_run_trans)
            else:
                s_lab = np.einsum("nij,j->ni", R_all, s_eff)
        else:
            q_lab_calc = q_sample_calc @ R_all.T
            if per_run_trans is not None:
                s_lab = (s_eff[None, :] + per_run_trans) @ R_all.T
            else:
                s_lab = R_all @ s_eff
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


_PREDICT_HKL_CHUNK = 1 << 20  # 1,048,576 HKLs per static Ewald-mask slice
_PREDICT_RAY_CHUNK = 1 << 17  # 131,072 Ewald-band HKLs per multi-panel slice
_PREDICT_GPU_KERNELS: dict = {}


def _resolve_lab_kinematics(
    UB: np.ndarray,
    sample_offset=None,
    R_all=None,
    gonio_axes=None,
    gonio_angles=None,
    gonio_offsets=None,
    extra_rot=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact float64 (RUB_lab, s_lab) for one image/orientation state."""
    UB64 = np.asarray(UB, dtype=np.float64)
    if gonio_axes is not None and gonio_angles is not None:
        from subhkl.instrument.goniometer import sample_to_lab

        offsets = sample_offset
        if offsets is not None and np.asarray(offsets).ndim == 1:
            offsets_full = np.zeros((len(gonio_axes), 3), dtype=np.float64)
            offsets_full[-1] = offsets
        elif offsets is None:
            offsets_full = np.zeros((len(gonio_axes), 3), dtype=np.float64)
        else:
            offsets_full = np.asarray(offsets, dtype=np.float64)

        R_gonio = np.asarray(
            sample_to_lab(
                np.eye(3, dtype=np.float64),
                gonio_axes,
                gonio_angles,
                offsets_full,
                zero_offsets=gonio_offsets,
                is_vector=True,
            ),
            dtype=np.float64,
        ).T
        s_lab = np.asarray(
            sample_to_lab(
                np.zeros(3, dtype=np.float64),
                gonio_axes,
                gonio_angles,
                offsets_full,
                zero_offsets=gonio_offsets,
                is_vector=False,
            ),
            dtype=np.float64,
        ).reshape(3)
        R_eff = R_gonio
    else:
        s = (
            np.asarray(sample_offset, dtype=np.float64).reshape(3)
            if sample_offset is not None
            else np.zeros(3, dtype=np.float64)
        )
        if R_all is not None:
            R_eff = np.asarray(R_all, dtype=np.float64)
            s_lab = R_eff @ s
        else:
            R_eff = np.eye(3, dtype=np.float64)
            s_lab = s

    if extra_rot is not None:
        R_eff = np.asarray(extra_rot, dtype=np.float64) @ R_eff

    return R_eff @ UB64, s_lab


def _get_ewald_mask_kernel():
    key = ("ewald_mask", _PREDICT_HKL_CHUNK)
    if key not in _PREDICT_GPU_KERNELS:
        import jax
        import jax.numpy as jnp
        from jax import jit

        @jit
        def _ewald_mask(hkl_chunk, rub, ki_hat, wl_lo, wl_hi, n_valid):
            hp = jax.lax.Precision.HIGHEST
            q_lab = jnp.matmul(rub, hkl_chunk, precision=hp)  # (3, C)
            g2 = jnp.sum(q_lab * q_lab, axis=0)
            g_dot_ki = jnp.sum(q_lab * ki_hat[:, None], axis=0)
            lam = -2.0 * g_dot_ki / jnp.maximum(g2, 1e-30)
            valid = jnp.arange(_PREDICT_HKL_CHUNK) < n_valid
            return valid & (g2 > 1e-20) & (lam >= wl_lo) & (lam <= wl_hi)

        _PREDICT_GPU_KERNELS[key] = _ewald_mask
    return _PREDICT_GPU_KERNELS[key]


def _get_flat_panels_kernel(p_bucket: int):
    key = ("flat_panels", p_bucket, _PREDICT_RAY_CHUNK)
    if key not in _PREDICT_GPU_KERNELS:
        import jax
        import jax.numpy as jnp
        from jax import jit

        @jit
        def _flat_hit(
            hkl_chunk,
            rub,
            ki_hat,
            s_lab,
            centers,
            uhats,
            vhats,
            norms,
            dw,
            dh,
            n_rows,
            m_cols,
            p_valid,
            n_rays,
        ):
            hp = jax.lax.Precision.HIGHEST
            q_lab = jnp.matmul(rub, hkl_chunk, precision=hp)  # (3, R)
            g2 = jnp.sum(q_lab * q_lab, axis=0)
            g_dot_ki = jnp.sum(q_lab * ki_hat[:, None], axis=0)
            lam = -2.0 * g_dot_ki / jnp.maximum(g2, 1e-30)
            kf = ki_hat[:, None] + lam[None, :] * q_lab
            kf_dir = (
                kf
                * jax.lax.rsqrt(jnp.maximum(jnp.sum(kf * kf, axis=0), 1e-30))[None, :]
            )

            s_minus_c = s_lab[None, :] - centers  # (P, 3)
            c_minus_s_dot_n = -jnp.sum(s_minus_c * norms, axis=1)[:, None]  # (P, 1)
            d_dot_n = jnp.matmul(norms, kf_dir, precision=hp)  # (P, R)
            safe_den = jnp.where(
                jnp.abs(d_dot_n) > 1e-25,
                d_dot_n,
                jnp.where(d_dot_n >= 0, 1e-25, -1e-25),
            )
            t = c_minus_s_dot_n / safe_den

            sv = jnp.sum(s_minus_c * vhats, axis=1)[:, None]
            su = jnp.sum(s_minus_c * uhats, axis=1)[:, None]
            row = (sv + t * jnp.matmul(vhats, kf_dir, precision=hp)) / dh[:, None]
            col = (su + t * jnp.matmul(uhats, kf_dir, precision=hp)) / dw[:, None]

            valid_ray = (jnp.arange(_PREDICT_RAY_CHUNK) < n_rays)[None, :]
            eps_px = 1.5
            return (
                p_valid[:, None]
                & valid_ray
                & (g2[None, :] > 1e-20)
                & (jnp.abs(d_dot_n) > 1e-18)
                & (t > -1e-5)
                & (row >= -eps_px)
                & (col >= -eps_px)
                & (row <= n_rows[:, None] + eps_px)
                & (col <= m_cols[:, None] + eps_px)
            )

        _PREDICT_GPU_KERNELS[key] = _flat_hit
    return _PREDICT_GPU_KERNELS[key]


def _get_curved_panels_kernel(p_bucket: int):
    key = ("curved_panels", p_bucket, _PREDICT_RAY_CHUNK)
    if key not in _PREDICT_GPU_KERNELS:
        import jax
        import jax.numpy as jnp
        from jax import jit

        @jit
        def _curved_hit(
            hkl_chunk,
            rub,
            ki_hat,
            s_lab,
            centers,
            vhats,
            rhats,
            whats,
            radii,
            dw,
            dh,
            n_rows,
            m_cols,
            p_valid,
            n_rays,
        ):
            hp = jax.lax.Precision.HIGHEST
            q_lab = jnp.matmul(rub, hkl_chunk, precision=hp)  # (3, R)
            g2 = jnp.sum(q_lab * q_lab, axis=0)
            g_dot_ki = jnp.sum(q_lab * ki_hat[:, None], axis=0)
            lam = -2.0 * g_dot_ki / jnp.maximum(g2, 1e-30)
            kf = ki_hat[:, None] + lam[None, :] * q_lab
            kf_dir = (
                kf
                * jax.lax.rsqrt(jnp.maximum(jnp.sum(kf * kf, axis=0), 1e-30))[None, :]
            )  # (3, R)

            # Cylinder intersection per curved panel: (P, 3) x (3, R)
            B_vec = jnp.cross(jnp.broadcast_to(s_lab[None, :], vhats.shape), vhats)
            # D_vec = kf_dir.T x vhat -> shape (P, R, 3)
            kf_T = kf_dir.T[None, :, :]  # (1, R, 3)
            vh_exp = vhats[:, None, :]  # (P, 1, 3)
            D_vec = jnp.cross(kf_T, vh_exp)  # (P, R, 3)

            QA = jnp.sum(D_vec * D_vec, axis=2)  # (P, R)
            QB = 2.0 * jnp.sum(B_vec[:, None, :] * D_vec, axis=2)  # (P, R)
            QC = (jnp.sum(B_vec * B_vec, axis=1) - radii * radii)[:, None]  # (P, 1)
            delta = QB * QB - 4.0 * QA * QC
            sqrt_d = jnp.sqrt(jnp.maximum(delta, 0.0))
            safe_2a = jnp.maximum(2.0 * QA, 1e-25)
            t1 = (-QB + sqrt_d) / safe_2a
            t2 = (-QB - sqrt_d) / safe_2a
            t = jnp.where(t2 > 0.0, t2, t1)
            t = jnp.where(delta < 0.0, -1.0, t)  # (P, R)

            # vec = s_lab + t * kf_dir - center
            s_minus_c = s_lab[None, :] - centers  # (P, 3)
            sv = jnp.sum(s_minus_c * vhats, axis=1)[:, None]
            dot_v = sv + t * jnp.matmul(vhats, kf_dir, precision=hp)  # (P, R)
            row = dot_v / dh[:, None]

            sr = jnp.sum(s_minus_c * rhats, axis=1)[:, None]
            sw = jnp.sum(s_minus_c * whats, axis=1)[:, None]
            # Since vhat _|_ rhat and vhat _|_ what, d_planar . rhat == vec . rhat!
            dot_r = sr + t * jnp.matmul(rhats, kf_dir, precision=hp)
            dot_w = sw + t * jnp.matmul(whats, kf_dir, precision=hp)
            dt = jnp.mod(2.0 * jnp.arctan(-dot_r / dot_w), 2.0 * jnp.pi)
            col = dt * (radii / dw)[:, None]
            col_wrap = (2.0 * jnp.pi * radii / dw)[:, None]

            valid_ray = (jnp.arange(_PREDICT_RAY_CHUNK) < n_rays)[None, :]
            eps_px = 1.5
            in_col = ((col >= -eps_px) & (col <= m_cols[:, None] + eps_px)) | (
                col >= col_wrap - eps_px
            )
            return (
                p_valid[:, None]
                & valid_ray
                & (g2[None, :] > 1e-20)
                & (t > 0.0)
                & (row >= -eps_px)
                & (row <= n_rows[:, None] + eps_px)
                & in_col
            )

        _PREDICT_GPU_KERNELS[key] = _curved_hit
    return _PREDICT_GPU_KERNELS[key]


def _panel_bucket_size(n_panels: int) -> int:
    for b in (4, 16, 64, 256):
        if n_panels <= b:
            return b
    return ((n_panels + 255) // 256) * 256


def predict_peaks_batch_gpu(
    tasks: list[tuple],
    show_progress: bool = False,
) -> dict:
    """Predict peaks for all (image, bank) tasks using JAX/GPU multi-panel batching.

    Generates candidate Miller indices once (using Ewald-shell rod slicing for
    single-orientation datasets), evaluates the Laue wavelength band once per
    unique orientation/run on the accelerator in fixed-shape chunks (zero JIT
    recompilations), filters panel intersections across all active banks of
    each run simultaneously on GPU, and finalizes exact float64 pixel/lambda
    coordinates via ``predict_reflections_on_panel`` on the surviving hits.
    """
    if not tasks:
        return {}

    import jax.numpy as jnp
    from tqdm import tqdm

    from subhkl.core.crystallography import generate_reflections
    from subhkl.instrument.detector import Detector, DetectorShape

    first = tasks[0]
    unit_cell_params = first[3]
    a, b, c, alpha, beta, gamma, space_group, d_min = unit_cell_params
    wl_min = float(first[5])
    wl_max = float(first[6])
    ki_raw = first[8]
    ki_vec = (
        np.asarray(ki_raw, dtype=np.float64)
        if ki_raw is not None
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    ki_hat = ki_vec / np.linalg.norm(ki_vec)
    ki_f32 = jnp.asarray(ki_hat, dtype=jnp.float32)

    # 1. Group tasks by unique lab-frame orientation + sample origin so all
    #    detector banks of a run share one Ewald-shell evaluation and one
    #    multi-panel intersection pass.
    det_cache: dict[int, Detector] = {}
    groups_by_key: dict[tuple, dict] = {}
    for t in tasks:
        (
            img_key,
            bank_id,
            det_config,
            _ucp,
            UB_t,
            _wmin,
            _wmax,
            so_eff,
            _ki,
            R_bank,
            gonio_axes,
            angles_bank,
            gonio_offsets,
            extra_rot,
        ) = t
        b_int = int(bank_id)
        det = det_cache.get(b_int)
        if det is None or det.config is not det_config:
            det = Detector(det_config)
            det_cache[b_int] = det

        rub_lab, s_lab = _resolve_lab_kinematics(
            UB_t,
            sample_offset=so_eff,
            R_all=R_bank,
            gonio_axes=gonio_axes,
            gonio_angles=angles_bank,
            gonio_offsets=gonio_offsets,
            extra_rot=extra_rot,
        )
        g_key = (
            tuple(np.round(rub_lab.ravel(), 11)),
            tuple(np.round(s_lab.ravel(), 11)),
        )
        grp = groups_by_key.get(g_key)
        if grp is None:
            grp = {
                "rub_lab": rub_lab,
                "s_lab": s_lab,
                "items": [],
            }
            groups_by_key[g_key] = grp
        grp["items"].append((img_key, det, t))

    groups = list(groups_by_key.values())

    # 2. Generate candidate (h, k, l) ONCE for the whole dataset, then filter
    #    Ewald wavelength bands and multi-panel intersections on GPU.
    h, k, l = generate_reflections(a, b, c, alpha, beta, gamma, space_group, d_min)

    n_hkl = len(h)
    if n_hkl == 0:
        return {}

    # Pre-build host float32 (3, M) and cache device chunks when compact enough
    # (~192 MB cap for <= 16M HKLs) so multi-run jobs don't re-upload over PCIe.
    ewald_kernel = _get_ewald_mask_kernel()
    n_chunks = (n_hkl + _PREDICT_HKL_CHUNK - 1) // _PREDICT_HKL_CHUNK
    device_hkl_chunks: list | None = None
    if n_hkl <= (1 << 24) and len(groups) > 1:
        device_hkl_chunks = []
        for ci in range(n_chunks):
            lo = ci * _PREDICT_HKL_CHUNK
            hi = min(n_hkl, lo + _PREDICT_HKL_CHUNK)
            m = hi - lo
            buf = np.zeros((3, _PREDICT_HKL_CHUNK), dtype=np.float32)
            buf[0, :m] = h[lo:hi]
            buf[1, :m] = k[lo:hi]
            buf[2, :m] = l[lo:hi]
            device_hkl_chunks.append((jnp.asarray(buf), m, lo))

    wl_pad = max(1e-3, 1e-4 * wl_max)
    wl_lo_f32 = jnp.float32(max(0.0, wl_min - wl_pad))
    wl_hi_f32 = jnp.float32(wl_max + wl_pad)

    results_by_img: dict = {}
    ray_buf = np.zeros((3, _PREDICT_RAY_CHUNK), dtype=np.float32)

    pbar = tqdm(groups, desc="Predicting (GPU)", disable=not show_progress)
    for grp in pbar:
        rub_f32 = jnp.asarray(grp["rub_lab"], dtype=jnp.float32)
        s_f32 = jnp.asarray(grp["s_lab"], dtype=jnp.float32)

        # 3a. Filter full sphere down to active Ewald shell for this orientation
        hit_parts = []
        if device_hkl_chunks is not None:
            for dev_buf, m_valid, lo in device_hkl_chunks:
                mask_dev = ewald_kernel(
                    dev_buf,
                    rub_f32,
                    ki_f32,
                    wl_lo_f32,
                    wl_hi_f32,
                    jnp.int32(m_valid),
                )
                loc = np.flatnonzero(np.asarray(mask_dev)[:m_valid])
                if len(loc):
                    hit_parts.append(loc + lo)
        else:
            host_buf = np.zeros((3, _PREDICT_HKL_CHUNK), dtype=np.float32)
            for ci in range(n_chunks):
                lo = ci * _PREDICT_HKL_CHUNK
                hi = min(n_hkl, lo + _PREDICT_HKL_CHUNK)
                m_valid = hi - lo
                host_buf[0, :m_valid] = h[lo:hi]
                host_buf[1, :m_valid] = k[lo:hi]
                host_buf[2, :m_valid] = l[lo:hi]
                mask_dev = ewald_kernel(
                    jnp.asarray(host_buf),
                    rub_f32,
                    ki_f32,
                    wl_lo_f32,
                    wl_hi_f32,
                    jnp.int32(m_valid),
                )
                loc = np.flatnonzero(np.asarray(mask_dev)[:m_valid])
                if len(loc):
                    hit_parts.append(loc + lo)
        if not hit_parts:
            continue
        wl_idx = hit_parts[0] if len(hit_parts) == 1 else np.concatenate(hit_parts)

        n_wl = len(wl_idx)
        if n_wl == 0:
            continue

        flat_items = [
            it for it in grp["items"] if it[1].panel_type == DetectorShape.flat_panel
        ]
        curved_items = [
            it for it in grp["items"] if it[1].panel_type == DetectorShape.curved_panel
        ]

        # 3b. Multi-panel GPU intersection for flat panels
        cand_by_item: list[tuple[tuple, list[np.ndarray]]] = []
        if flat_items:
            n_p = len(flat_items)
            p_bkt = _panel_bucket_size(n_p)
            flat_kernel = _get_flat_panels_kernel(p_bkt)

            centers = np.zeros((p_bkt, 3), dtype=np.float32)
            uhats = np.zeros((p_bkt, 3), dtype=np.float32)
            vhats = np.zeros((p_bkt, 3), dtype=np.float32)
            norms = np.ones((p_bkt, 3), dtype=np.float32)
            dw = np.ones(p_bkt, dtype=np.float32)
            dh = np.ones(p_bkt, dtype=np.float32)
            n_rows = np.zeros(p_bkt, dtype=np.float32)
            m_cols = np.zeros(p_bkt, dtype=np.float32)
            p_valid = np.zeros(p_bkt, dtype=bool)

            for pi, (_ik, det, _task) in enumerate(flat_items):
                centers[pi] = det.center
                uhats[pi] = det.uhat
                vhats[pi] = det.vhat
                norms[pi] = np.cross(det.uhat, det.vhat)
                dw[pi] = det.width / (det.m - 1)
                dh[pi] = det.height / (det.n - 1)
                n_rows[pi] = float(det.n)
                m_cols[pi] = float(det.m)
                p_valid[pi] = True

            dev_geom = (
                jnp.asarray(centers),
                jnp.asarray(uhats),
                jnp.asarray(vhats),
                jnp.asarray(norms),
                jnp.asarray(dw),
                jnp.asarray(dh),
                jnp.asarray(n_rows),
                jnp.asarray(m_cols),
                jnp.asarray(p_valid),
            )
            per_panel_lists: list[list[np.ndarray]] = [[] for _ in range(n_p)]
            for r0 in range(0, n_wl, _PREDICT_RAY_CHUNK):
                r1 = min(n_wl, r0 + _PREDICT_RAY_CHUNK)
                m_r = r1 - r0
                sub = wl_idx[r0:r1]
                ray_buf[0, :m_r] = h[sub]
                ray_buf[1, :m_r] = k[sub]
                ray_buf[2, :m_r] = l[sub]
                hit_mat = np.asarray(
                    flat_kernel(
                        jnp.asarray(ray_buf),
                        rub_f32,
                        ki_f32,
                        s_f32,
                        *dev_geom,
                        jnp.int32(m_r),
                    )
                )[:n_p, :m_r]
                p_hits, r_hits = np.nonzero(hit_mat)
                if len(p_hits):
                    # Split sorted p_hits in O(nnz) since np.nonzero is row-ordered!
                    splits = np.flatnonzero(np.diff(p_hits)) + 1
                    p_groups = np.split(p_hits, splits)
                    r_groups = np.split(r_hits, splits)
                    for pg, rg in zip(p_groups, r_groups):
                        per_panel_lists[int(pg[0])].append(sub[rg])

            for pi, item in enumerate(flat_items):
                if per_panel_lists[pi]:
                    cand_by_item.append((item, per_panel_lists[pi]))

        # 3c. Multi-panel GPU intersection for curved panels
        if curved_items:
            n_p = len(curved_items)
            p_bkt = _panel_bucket_size(n_p)
            curved_kernel = _get_curved_panels_kernel(p_bkt)

            centers = np.zeros((p_bkt, 3), dtype=np.float32)
            vhats = np.zeros((p_bkt, 3), dtype=np.float32)
            rhats = np.zeros((p_bkt, 3), dtype=np.float32)
            whats = np.zeros((p_bkt, 3), dtype=np.float32)
            radii = np.ones(p_bkt, dtype=np.float32)
            dw = np.ones(p_bkt, dtype=np.float32)
            dh = np.ones(p_bkt, dtype=np.float32)
            n_rows = np.zeros(p_bkt, dtype=np.float32)
            m_cols = np.zeros(p_bkt, dtype=np.float32)
            p_valid = np.zeros(p_bkt, dtype=bool)

            for pi, (_ik, det, _task) in enumerate(curved_items):
                centers[pi] = det.center
                vhats[pi] = det.vhat
                rhats[pi] = det.rhat
                whats[pi] = np.cross(det.vhat, det.rhat)
                radii[pi] = float(det.radius)
                dw[pi] = det.width / (det.m - 1)
                dh[pi] = det.height / (det.n - 1)
                n_rows[pi] = float(det.n)
                m_cols[pi] = float(det.m)
                p_valid[pi] = True

            dev_geom_c = (
                jnp.asarray(centers),
                jnp.asarray(vhats),
                jnp.asarray(rhats),
                jnp.asarray(whats),
                jnp.asarray(radii),
                jnp.asarray(dw),
                jnp.asarray(dh),
                jnp.asarray(n_rows),
                jnp.asarray(m_cols),
                jnp.asarray(p_valid),
            )
            per_panel_lists_c: list[list[np.ndarray]] = [[] for _ in range(n_p)]
            for r0 in range(0, n_wl, _PREDICT_RAY_CHUNK):
                r1 = min(n_wl, r0 + _PREDICT_RAY_CHUNK)
                m_r = r1 - r0
                sub = wl_idx[r0:r1]
                ray_buf[0, :m_r] = h[sub]
                ray_buf[1, :m_r] = k[sub]
                ray_buf[2, :m_r] = l[sub]
                hit_mat = np.asarray(
                    curved_kernel(
                        jnp.asarray(ray_buf),
                        rub_f32,
                        ki_f32,
                        s_f32,
                        *dev_geom_c,
                        jnp.int32(m_r),
                    )
                )[:n_p, :m_r]
                p_hits, r_hits = np.nonzero(hit_mat)
                if len(p_hits):
                    splits = np.flatnonzero(np.diff(p_hits)) + 1
                    p_groups = np.split(p_hits, splits)
                    r_groups = np.split(r_hits, splits)
                    for pg, rg in zip(p_groups, r_groups):
                        per_panel_lists_c[int(pg[0])].append(sub[rg])

            for pi, item in enumerate(curved_items):
                if per_panel_lists_c[pi]:
                    cand_by_item.append((item, per_panel_lists_c[pi]))

        # 3d. Exact float64 coordinate finalization on surviving panel hits
        for (img_key, det, task_tup), idx_parts in cand_by_item:
            cand = idx_parts[0] if len(idx_parts) == 1 else np.concatenate(idx_parts)
            (
                _ik,
                _bid,
                _dcfg,
                _ucp,
                UB_t,
                wmin_t,
                wmax_t,
                so_eff,
                ki_t,
                R_bank,
                gonio_axes,
                angles_bank,
                gonio_offsets,
                extra_rot,
            ) = task_tup
            row, col, h_f, k_f, l_f, wl_f = predict_reflections_on_panel(
                detector=det,
                h=h[cand],
                k=k[cand],
                l=l[cand],
                UB=UB_t,
                wavelength_min=wmin_t,
                wavelength_max=wmax_t,
                sample_offset=so_eff,
                ki_vec=ki_t,
                R_all=R_bank,
                gonio_axes=gonio_axes,
                gonio_angles=angles_bank,
                gonio_offsets=gonio_offsets,
                extra_rot=extra_rot,
            )
            if len(row) > 0:
                results_by_img[img_key] = [row, col, h_f, k_f, l_f, wl_f]

    return {k_id: results_by_img[k_id] for k_id in sorted(results_by_img.keys())}
