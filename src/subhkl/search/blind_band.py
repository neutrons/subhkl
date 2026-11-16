"""Single-crystal discovery by band search, shared geometry, and full-data ranking."""

import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.config import beamlines
from subhkl.core.spacegroup import get_space_group_object
from subhkl.instrument.detector import Detector
from subhkl.scan_refine import load_observations
from subhkl.search.scan_band import band_sum, build_B_jax, direct_lattice_uvw
from subhkl.search.spherical import (
    band_q_range,
    direct_lattice_vectors,
    lattice_ladder,
    refine_band_geometry,
)


def sample_signed(weights, size, seed=0):
    """Unbiased quadrature of a signed sum, sampling by absolute contribution.

    Every nonzero residual can be drawn; duplicates are combined. Returned
    weights estimate the original sum divided by its total absolute weight.
    No intensity cutoff or local-maximum selection is introduced.
    """
    weights = np.asarray(weights, float)
    total = np.abs(weights).sum()
    if size < 1 or not np.isfinite(weights).all() or total == 0:
        raise ValueError(
            "positive sample size and finite nonzero residual evidence required"
        )
    if len(weights) <= size:
        return np.arange(len(weights)), weights / total
    rng = np.random.default_rng(seed)
    drawn = rng.choice(len(weights), size=size, replace=True, p=np.abs(weights) / total)
    indices, counts = np.unique(drawn, return_counts=True)
    return indices, np.sign(weights[indices]) * counts / size


def distinct_orientations(candidates, B, space_group, count, separation_deg=1.0):
    """Remove duplicates under Laue proper crystallographic rotations."""
    operations = []
    B_arr = np.asarray(B, float)
    B_inv = np.linalg.inv(B_arr)
    for op in get_space_group_object(space_group).operations().sym_ops:
        S = B_arr @ np.linalg.inv(np.asarray(op.rot, float) / op.DEN).T @ B_inv
        if np.linalg.det(S) < 0:
            S = -S
        if not any(np.allclose(S, Sp, atol=1e-5) for Sp in operations):
            operations.append(S)
    selected = []
    for candidate in candidates:
        U = candidate[0]
        if all(
            min(Rotation.from_matrix(U @ (V[0] @ S).T).magnitude() for S in operations)
            > np.deg2rad(separation_deg)
            for V in selected
        ):
            selected.append(candidate)
        if len(selected) >= count:
            break
    return selected


def count_search_weights(deviance, scale=25.0, max_quantile=None, max_factor=3.0):
    """Smooth chi-square-null-suppressed Poisson deviance weights for count bins.

    Under a Poisson background null, single-bin deviance follows chi^2_1 (~1),
    whereas genuine Bragg reflections in an 8x8 bin produce deviance >> 25.
    Weighting by sqrt(D) * (1 - exp(-(D / scale)^3)) is strictly threshold-free
    (every positive-deviance bin has positive weight), suppresses 1-sigma background
    fluctuations by ~6e-5 so importance sampling is not flooded by background bins,
    and grows as sqrt(D) at high deviance so bright flare pixels cannot monopolize
    the sample. When `max_quantile` is set, extreme unmasked beamstop-flare bins are
    capped at `max_factor` times that positive-weight quantile.
    """
    dev = np.maximum(np.asarray(deviance, float), 0.0)
    w = np.sqrt(dev) * (1.0 - np.exp(-((dev / float(scale)) ** 3)))
    if max_quantile is not None:
        pos = w[w > 0]
        if len(pos):
            w = np.minimum(w, float(max_factor) * float(np.quantile(pos, max_quantile)))
    return w


def recover_unlogged_outer_goniometer(args, observations, pixel_em=None):
    """Validate and recover unlogged outer goniometer motor angles on multi-run scans.

    When an inner motor (``scan_col > 0``, e.g. ``phi`` on a 5-axis kappa stage)
    is scanned while outer motors (``omega, kappa``) were omitted from raw NeXus
    logs (defaulting to ``0``), the nominal lab scan axis can be tens of degrees
    off, scrambling cross-run de-rotation.  This model-free check tests whether
    observed Bragg spot directions de-rotate into coincidence across run pairs
    ``1 <= |Delta phi| <= 45 deg`` under reachable outer motor angles, without
    requiring a unit cell or orientation.  Activates only when ``scan_col > 0``
    and a distinct outer configuration achieves overwhelming significance
    (``z >= 8``, ``> 5 deg`` from nominal, and ``>= 3x`` nominal coincidence).
    """
    if getattr(args, "_outer_goniometer_checked", False):
        return observations
    args._outer_goniometer_checked = True
    if getattr(args, "finder", None):
        return observations
    nexus_path = getattr(args, "nexus", None)
    active_runs = sorted(set(getattr(args, "runs", [])))
    if not nexus_path or len(active_runs) < 2:
        return observations

    from scipy.optimize import minimize

    from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix
    from subhkl.scan_refine import _frame_offsets, _frame_rotations_and_scan_axes

    with h5py.File(nexus_path, "r") as f:
        n_frames = int(f["images"].shape[0])
        offsets = _frame_offsets(f, n_frames)
        _, axes, angles, scan_col, _ = _frame_rotations_and_scan_axes(f, n_frames)

    if (
        axes is None
        or angles is None
        or scan_col is None
        or int(scan_col) <= 0
        or len(offsets) < 2
    ):
        return observations
    scan_col = int(scan_col)
    n_runs_total = len(offsets)
    valid_runs = [r for r in active_runs if 0 <= r < n_runs_total]
    if len(valid_runs) < 2:
        return observations

    r_ref = valid_runs[0]
    phi_deg = {
        r: float(angles[offsets[r], scan_col] - angles[offsets[r_ref], scan_col])
        for r in valid_runs
    }
    has_pair = any(
        1.0 <= abs(phi_deg[r2] - phi_deg[r1]) <= 45.0
        for i1, r1 in enumerate(valid_runs)
        for r2 in valid_runs[i1 + 1 :]
    )
    if not has_pair:
        return observations

    banks, rows, cols, dev, sig_px, runs, ang_obs, weights, R_obs, scan_axes = (
        observations
    )
    if pixel_em is not None:
        dev_use, _, gv_use = pixel_em.map_to_observations(
            banks, rows, cols, runs, dev, weights
        )
    else:
        dev_use = dev
        gv_use = np.ones(len(dev), dtype=np.float64)
    sw_all = count_search_weights(dev_use, max_quantile=0.999, max_factor=3.0) * gv_use

    # Build lab-frame scattering unit directions only for top spots per run
    run_dirs: dict[int, np.ndarray] = {}
    run_w: dict[int, np.ndarray] = {}
    det_cache: dict[int, Detector] = {}
    for r in valid_runs:
        mr = np.flatnonzero((runs == r) & (dev_use >= 20.0) & (sw_all > 0.0))
        if len(mr) < 3:
            continue
        if len(mr) > 48:
            mr = mr[np.argsort(-sw_all[mr])[:48]]
        xyz = np.empty((len(mr), 3), dtype=np.float64)
        b_sub = banks[mr]
        for bk in np.unique(b_sub):
            bk_i = int(bk)
            if bk_i not in det_cache:
                det_cache[bk_i] = Detector(dict(beamlines[args.instrument][str(bk_i)]))
            det_b = det_cache[bk_i]
            mb = b_sub == bk
            mid = (
                det_b.center
                + 0.5 * det_b.width * det_b.uhat
                + 0.5 * det_b.height * det_b.vhat
            )
            off = ((cols[mr[mb]] / (det_b.m - 1) - 0.5) * det_b.width)[
                :, None
            ] * det_b.uhat + ((rows[mr[mb]] / (det_b.n - 1) - 0.5) * det_b.height)[
                :, None
            ] * det_b.vhat
            xyz[mb] = mid + off
        q = xyz / np.linalg.norm(xyz, axis=1, keepdims=True) - np.array([0.0, 0.0, 1.0])
        run_dirs[r] = q / np.linalg.norm(q, axis=1, keepdims=True)
        wr = sw_all[mr].astype(np.float64)
        run_w[r] = wr / max(float(wr.sum()), 1e-12)

    usable_runs = [r for r in valid_runs if r in run_dirs]
    pairs = [
        (r1, r2)
        for i1, r1 in enumerate(usable_runs)
        for r2 in usable_runs[i1 + 1 :]
        if 1.0 <= abs(phi_deg[r2] - phi_deg[r1]) <= 45.0
    ]
    if not pairs:
        return observations

    # Identify variable outer motor columns outside `scan_col`
    if (
        scan_col == 4
        and np.allclose(axes[1, :3], axes[3, :3], atol=1e-5)
        and abs(float(axes[1, 3]) + float(axes[3, 3])) < 1e-5
    ):
        free_outer = [0, 2]
    else:
        free_outer = list(range(min(scan_col, 2)))

    u_scan = float(axes[scan_col, 3]) * (
        axes[scan_col, :3] / max(float(np.linalg.norm(axes[scan_col, :3])), 1e-12)
    )
    base_outer = angles[offsets[r_ref], :scan_col].astype(np.float64).copy()
    n_nom = calc_goniometer_rotation_matrix(axes[:scan_col], base_outer) @ u_scan
    n_nom = n_nom / max(float(np.linalg.norm(n_nom)), 1e-12)

    axes_list = [n_nom]
    cfg_list = [tuple(float(base_outer[c]) for c in free_outer)]
    cos_dedup = float(np.cos(np.deg2rad(3.5)))
    if len(free_outer) == 1:
        g0 = np.linspace(-180.0, 180.0, 108, endpoint=False)
        raw_cfgs = [(float(v),) for v in g0]
    else:
        g0 = np.linspace(-180.0, 180.0, 60, endpoint=False)
        g1 = np.linspace(-180.0, 180.0, 61)
        raw_cfgs = [
            (float(v0), float(v1))
            for v0, v1 in np.array(np.meshgrid(g0, g1, indexing="ij")).reshape(2, -1).T
        ]
    for cfg in raw_cfgs:
        ao = base_outer.copy()
        for idx_c, col_c in enumerate(free_outer):
            ao[col_c] = cfg[idx_c]
        nv = calc_goniometer_rotation_matrix(axes[:scan_col], ao) @ u_scan
        nv = nv / max(float(np.linalg.norm(nv)), 1e-12)
        if float(np.max(np.asarray(axes_list) @ nv)) < cos_dedup:
            axes_list.append(nv)
            cfg_list.append(cfg)
    axes_s2 = np.asarray(axes_list, dtype=np.float64)

    dphi_rad = {r: np.deg2rad(phi_deg[r]) for r in usable_runs}
    derot = {
        r: np.einsum(
            "mij,nj->mni",
            Rotation.from_rotvec(-dphi_rad[r] * axes_s2).as_matrix(),
            run_dirs[r],
        )
        for r in usable_runs
    }
    inv_s2 = 1.0 / (np.deg2rad(0.9) ** 2)
    cos_static = float(np.cos(np.deg2rad(1.2)))
    pair_masks = {
        (r1, r2): ((run_dirs[r1] @ run_dirs[r2].T) < cos_static).astype(np.float64)
        * (run_w[r1][:, None] * run_w[r2][None, :])
        for r1, r2 in pairs
    }
    scores = np.zeros(len(axes_s2), dtype=np.float64)
    for r1, r2 in pairs:
        dots = np.einsum("mni,mki->mnk", derot[r1], derot[r2])
        scores += np.einsum(
            "mnk,nk->m",
            np.exp(np.clip((dots - 1.0) * inv_s2, -30.0, 0.0)),
            pair_masks[(r1, r2)],
        )

    best_m = int(np.argmax(scores))
    score_nom = float(scores[0])

    def _neg_score_cfg(x_cfg):
        ao = base_outer.copy()
        for idx_c, col_c in enumerate(free_outer):
            ao[col_c] = float(x_cfg[idx_c])
        nh = calc_goniometer_rotation_matrix(axes[:scan_col], ao) @ u_scan
        nh = nh / max(float(np.linalg.norm(nh)), 1e-12)
        dr = {
            r: Rotation.from_rotvec(-dphi_rad[r] * nh).apply(run_dirs[r])
            for r in usable_runs
        }
        tot = 0.0
        for r1, r2 in pairs:
            dots = dr[r1] @ dr[r2].T
            tot += float(
                np.sum(
                    np.exp(np.clip((dots - 1.0) * inv_s2, -30.0, 0.0))
                    * pair_masks[(r1, r2)]
                )
            )
        return -tot

    res_opt = minimize(
        _neg_score_cfg,
        np.asarray(cfg_list[best_m], dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": 60, "xatol": 0.05},
    )
    ao_fit = base_outer.copy()
    for idx_c, col_c in enumerate(free_outer):
        ao_fit[col_c] = float(((res_opt.x[idx_c] + 180.0) % 360.0) - 180.0)
    n_fit = calc_goniometer_rotation_matrix(axes[:scan_col], ao_fit) @ u_scan
    n_fit = n_fit / max(float(np.linalg.norm(n_fit)), 1e-12)
    score_fit = float(-res_opt.fun)

    far_mask = (axes_s2 @ n_fit) < float(np.cos(np.deg2rad(20.0)))
    if not np.any(far_mask):
        return observations
    bg_mean = float(scores[far_mask].mean())
    bg_std = max(float(scores[far_mask].std()), 1e-9)
    z_fit = (score_fit - bg_mean) / bg_std
    z_nom = (score_nom - bg_mean) / bg_std
    ang_diff_deg = float(
        np.degrees(np.arccos(np.clip(float(n_fit @ n_nom), -1.0, 1.0)))
    )

    if not (
        ang_diff_deg > 5.0
        and z_fit >= 8.0
        and z_nom < 3.0
        and score_fit >= 3.0 * max(score_nom, 1e-6)
    ):
        return observations

    outer_desc = ", ".join(f"axis[{c}]={ao_fit[c]:+.2f} deg" for c in free_outer)
    print(
        f"Outer goniometer recovery: nominal lab scan axis {np.round(n_nom, 4).tolist()} "
        f"(z={z_nom:+.2f}) inconsistent across runs; recovered {outer_desc} -> "
        f"lab scan axis {np.round(n_fit, 4).tolist()} (z={z_fit:+.2f}, diff={ang_diff_deg:.1f} deg)",
        flush=True,
    )

    R_runs_fit: dict[int, np.ndarray] = {}
    ang_by_run: dict[int, np.ndarray] = {}
    for r in range(n_runs_total):
        ar = angles[offsets[r]].astype(np.float64).copy()
        for col_c in free_outer:
            ar[col_c] = ao_fit[col_c]
        ang_by_run[r] = ar
        R_runs_fit[r] = calc_goniometer_rotation_matrix(axes, ar)

    R_obs_new = np.array(R_obs, dtype=np.float64, copy=True)
    scan_axes_new = np.broadcast_to(n_fit, scan_axes.shape).copy()
    ang_obs_new = np.array(ang_obs, dtype=np.float64, copy=True)
    for r, R_r in R_runs_fit.items():
        mr = runs == r
        if np.any(mr):
            R_obs_new[mr] = R_r
            if ang_obs_new.ndim == 2 and ang_obs_new.shape[1] == len(ang_by_run[r]):
                ang_obs_new[mr] = ang_by_run[r]

    new_obs = (
        banks,
        rows,
        cols,
        dev,
        sig_px,
        runs,
        ang_obs_new,
        weights,
        R_obs_new,
        scan_axes_new,
    )
    args._recovered_outer_R_runs = R_runs_fit
    if getattr(args, "_cached_obs_kin", None) is not None:
        args._cached_obs_kin = new_obs
    if pixel_em is not None:
        pixel_em.R_by_run.update(R_runs_fit)
    return new_obs


def discover(args, samples=16384, candidates=6, seed=0):
    """Find a seed without reading any reference orientation or calibration.

    Search on the stable-run subset; the subsequent scan fit estimates drift
    on all requested runs. Full-data scores compare every candidate under the
    same winning global geometry, not each candidate's preferred detector.
    """
    start = time.monotonic()
    observations = getattr(args, "_cached_obs_kin", None)
    if observations is None:
        observations = load_observations(args, return_kinematics=True)
        if set(getattr(args, "runs", [])) == set(getattr(args, "stable_runs", [])):
            args._cached_obs_kin = observations
    from subhkl.search.pixel_em import PixelEMClassifier, wilson_search_weights

    pixel_em = getattr(args, "_pixel_em", None)
    if pixel_em is None and not args.finder:
        pixel_em = PixelEMClassifier.from_args(args)
        args._pixel_em = pixel_em
    observations = recover_unlogged_outer_goniometer(args, observations, pixel_em)
    banks, rows, cols, dev, _, runs, _angles, weights, R_obs, _scan_axes = observations
    if pixel_em is not None:
        dev, weights, gv_obs = pixel_em.map_to_observations(
            banks, rows, cols, runs, dev, weights
        )
    else:
        gv_obs = np.ones(len(dev), dtype=np.float64)

    use = np.isin(runs, args.stable_runs)
    if args.finder:
        use &= dev >= args.min_deviance
        search_weights = weights[use]
        weighting_desc = "sqrt finder deviance"
    else:
        search_weights = (
            count_search_weights(dev[use], max_quantile=0.999, max_factor=3.0)
            * gv_obs[use]
        )
        weighting_desc = (
            "EM-masked chi2-suppressed bin Poisson deviance"
            if pixel_em is not None
            else "chi2-suppressed bin Poisson deviance"
        )
    dev_stable = dev[use]
    runs_stable = runs[use]
    banks, rows, cols, Rs = [v[use] for v in (banks, rows, cols, R_obs)]
    weights = search_weights
    rot_axis_only = getattr(args, "global_rot_axis", (0.0, 1.0, 0.0))
    modes = getattr(args, "modes", None)
    r_bound = 0.08 if (modes is None or "cylindrical" in modes) else 0.0
    a_bound = (
        0.08
        if (modes is None or ("cylindrical" in modes and "axial_stretch" in modes))
        else (0.0 if (modes is not None and "axial_stretch" not in modes) else None)
    )
    rot_bound = 2.0 if (modes is None or "global_rot" in modes) else 0.0
    t_bound = 0.02 if (modes is None or "global_trans" in modes) else 0.0
    geom_uncalibrated = r_bound > 0.0 or rot_bound > 0.0 or t_bound > 0.0
    C, O = np.empty((len(weights), 3)), np.empty((len(weights), 3))
    for bank in np.unique(banks):
        m = banks == bank
        det = Detector(dict(beamlines[args.instrument][str(bank)]))
        C[m] = det.center + 0.5 * det.width * det.uhat + 0.5 * det.height * det.vhat
        O[m] = ((cols[m] / (det.m - 1) - 0.5) * det.width)[:, None] * det.uhat + (
            (rows[m] / (det.n - 1) - 0.5) * det.height
        )[:, None] * det.vhat
    B = np.asarray(build_B_jax(*args.cell[:3], *np.deg2rad(args.cell[3:])))
    wavelength = (args.wl_min, args.wl_max)
    pool = max(24, 4 * candidates)
    xyz_all = C + O
    q_all = xyz_all / np.linalg.norm(xyz_all, axis=1, keepdims=True) - [0, 0, 1]
    lab_all = q_all / np.linalg.norm(q_all, axis=1, keepdims=True)
    all_directions = np.einsum("nji,nj->ni", Rs, lab_all)
    sin_th_all = -lab_all[:, 2]
    # Kinematic reachability mask (with 6% slack for uncalibrated geometry scale)
    kin_active = (2.0 * sin_th_all / max(wavelength[1], 1e-6)) < (
        1.06 / max(float(args.d_min), 1e-6)
    )
    if not np.any(kin_active):
        kin_active = np.ones(len(weights), dtype=bool)
    if geom_uncalibrated and not args.finder:
        pos_idx = np.flatnonzero((weights > 0) & kin_active)
        if len(pos_idx) == 0:
            pos_idx = np.flatnonzero(weights > 0)
        if len(pos_idx) > samples:
            top_pos = pos_idx[np.argsort(-weights[pos_idx])[:samples]]
            idx = np.sort(top_pos)
            sampled_weights = weights[idx] / max(np.abs(weights[idx]).sum(), 1e-12)
        elif int(kin_active.sum()) <= samples:
            idx = np.flatnonzero(kin_active)
            sampled_weights = weights[idx] / max(np.abs(weights[idx]).sum(), 1e-12)
        else:
            act_idx = np.flatnonzero(kin_active)
            top_abs = act_idx[np.argsort(-np.abs(weights[act_idx]))[:samples]]
            idx = np.sort(top_abs)
            sampled_weights = weights[idx] / max(np.abs(weights[idx]).sum(), 1e-12)
    else:
        w_kin = np.where(kin_active, weights, 0.0)
        if not np.any(w_kin != 0.0):
            w_kin = weights
        idx, sampled_weights = sample_signed(w_kin, samples, seed)
    Cs, Os, rotations = C[idx], O[idx], Rs[idx]
    xyz = Cs + Os
    q = xyz / np.linalg.norm(xyz, axis=1, keepdims=True) - [0, 0, 1]
    lab = q / np.linalg.norm(q, axis=1, keepdims=True)
    directions = np.einsum("nji,nj->ni", rotations, lab)
    blind_path = Path(args.output).with_suffix(".blind.h5")
    src_nexus = getattr(args, "nexus", getattr(args, "input", ""))

    def _scaled_midpoint(mid_vec: np.ndarray, r_sc: float, a_sc: float) -> np.ndarray:
        return np.stack(
            [
                mid_vec[..., 0] * (1.0 + r_sc),
                mid_vec[..., 1] * (1.0 + a_sc),
                mid_vec[..., 2] * (1.0 + r_sc),
            ],
            axis=-1,
        )

    def _write_blind_rung(
        stage_label: str,
        nz_val: int,
        u_cand: np.ndarray,
        geom_fit: dict | None = None,
    ) -> None:
        tmp_path = blind_path.with_suffix(".blind.tmp.h5")
        try:
            with h5py.File(tmp_path, "w") as bf:
                bf.attrs["instrument"] = str(args.instrument)
                bf["sample/U"] = np.asarray(u_cand, dtype=float)
                bf["sample/B"] = np.asarray(B, dtype=float)
                bf["sample/cell"] = np.asarray(args.cell, dtype=float)
                for k_c, v_c in zip(
                    ("a", "b", "c", "alpha", "beta", "gamma"), args.cell[:6]
                ):
                    bf[f"sample/{k_c}"] = float(v_c)
                bf["sample/space_group"] = str(args.space_group)
                bf["instrument/name"] = str(args.instrument)
                bf["instrument/wavelength"] = np.asarray(
                    [args.wl_min, args.wl_max], dtype=float
                )
                if geom_fit is not None:
                    r_sc = float(geom_fit.get("radial_scale", 0.0))
                    a_sc = float(geom_fit.get("axial_scale", r_sc))
                    r_vec = np.asarray(
                        geom_fit.get("global_rot_vec", np.zeros(3)), dtype=float
                    )
                    t_vec = np.asarray(
                        geom_fit.get("global_trans_m", np.zeros(3)), dtype=float
                    )
                    rot_m = Rotation.from_rotvec(r_vec).as_matrix()
                    cal_g = bf.create_group("detector_calibration")
                    for b_k, b_cfg in beamlines[args.instrument].items():
                        try:
                            b_id = int(b_k)
                        except (ValueError, TypeError):
                            continue
                        det_b = Detector(dict(b_cfg))
                        mid_b = (
                            det_b.center
                            + 0.5 * det_b.width * det_b.uhat
                            + 0.5 * det_b.height * det_b.vhat
                        )
                        mid_cal = _scaled_midpoint(mid_b, r_sc, a_sc) @ rot_m.T + t_vec
                        uh_cal = det_b.uhat @ rot_m.T
                        vh_cal = det_b.vhat @ rot_m.T
                        cen_cal = (
                            mid_cal
                            - 0.5 * det_b.width * uh_cal
                            - 0.5 * det_b.height * vh_cal
                        )
                        bg = cal_g.create_group(f"bank_{b_id}")
                        bg["center"] = np.asarray(cen_cal, dtype=float)
                        bg["uhat"] = np.asarray(uh_cal, dtype=float)
                        bg["vhat"] = np.asarray(vh_cal, dtype=float)
                        bg["width"] = float(det_b.width)
                        bg["height"] = float(det_b.height)
                sg = bf.create_group("search")
                sg.attrs["source"] = str(Path(src_nexus).resolve()) if src_nexus else ""
                sg.attrs["instrument"] = str(args.instrument)
                sg.attrs["d_min"] = float(args.d_min)
                sg.attrs["ladder_stage"] = str(stage_label)
                sg.attrs["ladder_zones"] = int(nz_val)
            tmp_path.replace(blind_path)
            import os as _os

            if (
                _os.environ.get("SUBHKL_LIVE_STDIO") == "1"
                or "PYTEST_CURRENT_TEST" not in _os.environ
            ):
                import base64
                import json as _json
                import zlib

                live_doc: dict = {
                    "kind": "blind",
                    "instrument": str(args.instrument),
                    "space_group": str(args.space_group),
                    "wavelength": [float(args.wl_min), float(args.wl_max)],
                    "d_min": float(args.d_min),
                    "stage": str(stage_label),
                    "n_zones": int(nz_val),
                    "cell": [round(float(x), 5) for x in args.cell[:6]],
                    "U": np.round(np.asarray(u_cand, dtype=float), 6).tolist(),
                    "B": np.round(np.asarray(B, dtype=float), 6).tolist(),
                }
                if geom_fit is not None:
                    live_doc["radial_scale"] = round(
                        float(geom_fit.get("radial_scale", 0.0)), 6
                    )
                    live_doc["axial_scale"] = round(
                        float(
                            geom_fit.get(
                                "axial_scale", geom_fit.get("radial_scale", 0.0)
                            )
                        ),
                        6,
                    )
                    live_doc["global_rot_vec"] = [
                        round(float(x), 6)
                        for x in np.asarray(
                            geom_fit.get("global_rot_vec", np.zeros(3)),
                            dtype=float,
                        ).ravel()[:3]
                    ]
                    live_doc["global_trans_m"] = [
                        round(float(x), 6)
                        for x in np.asarray(
                            geom_fit.get("global_trans_m", np.zeros(3)),
                            dtype=float,
                        ).ravel()[:3]
                    ]
                b64 = base64.b64encode(
                    zlib.compress(
                        _json.dumps(live_doc, separators=(",", ":")).encode("utf-8"),
                        level=9,
                    )
                ).decode("ascii")
                print(f"[subhkl-live-state] {b64}", flush=True)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
        ):
            tmp_path.unlink(missing_ok=True)

    print(
        f"Blind band search: {len(weights)} stable-run observations, {len(idx)} search observations (exhaustive rung init)",
        flush=True,
    )

    env_pool = max(128, 12 * int(candidates))

    if geom_uncalibrated:
        _, xlen_chk, _ = direct_lattice_vectors(B, space_group=args.space_group)
        max_cell_len = float(max(args.cell[:3]))
        coarse_nz = (
            60 if float(xlen_chk[min(29, len(xlen_chk) - 1)]) < max_cell_len else 30
        )
        ladder_rungs = ((coarse_nz, 0.9, 1.0, env_pool * 2, 5),)
        qlo_chk, qhi_chk = band_q_range(sin_th_all[kin_active], wavelength, args.d_min)
        qbar_med = max(float(np.median(0.5 * (qlo_chk + qhi_chk))), 1e-6)
        med_lobe_deg = float(
            np.median(
                np.degrees(
                    1.0 / np.maximum(4.0 * xlen_chk[:coarse_nz] * qbar_med, 1e-9)
                )
            )
        )
        # Macromolecular cells whose Laue carrier lobes are narrower than half
        # the 2 deg SO(3) grid step need spherical-cap max bridging; smaller or
        # anisotropic mineral cells already resolve their carrier lobes on the
        # grid and avoid equatorial belt inflation when band_env_deg == 0.
        band_env_deg = 1.1 if med_lobe_deg < 0.85 else 0.0
    else:
        coarse_nz = 13
        ladder_rungs = (
            (13, 0.5, 1.5, 300, 5),
            (30, 0.5, 1.0, 80, 5),
            (60, 0.3, 0.5, pool, 5),
            (145, 0.3, 0.25, pool, 5),
        )
        band_env_deg = 0.0

    table_limit = 131_072 if args.finder else 16_384
    if args.finder:
        act_pool = np.flatnonzero((weights != 0.0) & kin_active)
    else:
        w_pos_max = float(np.max(weights[kin_active])) if np.any(kin_active) else 0.0
        act_pool = np.flatnonzero((weights > max(1e-4 * w_pos_max, 0.0)) & kin_active)
        if len(act_pool) == 0:
            act_pool = np.flatnonzero((weights != 0.0) & kin_active)
    if len(act_pool) == 0:
        act_pool = np.arange(len(weights))
    if len(act_pool) > table_limit:
        table_cap = min(len(act_pool), table_limit)
        if args.finder:
            t_idx = act_pool[
                np.argpartition(np.abs(weights[act_pool]), -table_cap)[-table_cap:]
            ]
        else:
            pos_all = np.flatnonzero((weights > 0.0) & kin_active)
            if len(pos_all) >= table_cap:
                t_idx = pos_all[
                    np.argpartition(weights[pos_all], -table_cap)[-table_cap:]
                ]
            else:
                t_idx = act_pool[
                    np.argpartition(np.abs(weights[act_pool]), -table_cap)[-table_cap:]
                ]
        t_idx = np.sort(t_idx)
        table_dirs = all_directions[t_idx]
        table_weights = weights[t_idx] * (len(act_pool) / len(t_idx))
        table_sin_theta = -lab_all[t_idx, 2]
    else:
        table_dirs = all_directions[act_pool]
        table_weights = weights[act_pool]
        table_sin_theta = -lab_all[act_pool, 2]

    found = lattice_ladder(
        directions,
        sampled_weights,
        B,
        grid_deg=2.0,
        n_shortlist=1000,
        coarse_zones=coarse_nz,
        rungs=ladder_rungs,
        sin_theta=-lab[:, 2],
        wavelength=wavelength,
        d_min=args.d_min,
        table_dirs=table_dirs,
        table_weights=table_weights,
        table_sin_theta=table_sin_theta,
        n_null=32,
        seed=seed,
        verbose=True,
        space_group=args.space_group,
        dedup_deg=3.0,
        on_rung=_write_blind_rung,
        band_envelope_deg=band_env_deg,
        band_table_min_spacing_deg=0.22 if geom_uncalibrated else 0.0,
        table_tol_deg=0.8 if geom_uncalibrated else 1.0,
    )
    found = distinct_orientations(
        found,
        B,
        args.space_group,
        env_pool if geom_uncalibrated else candidates,
        separation_deg=3.0 if geom_uncalibrated else 1.0,
    )
    if not found:
        raise ValueError("blind band search found no orientation candidates")
    print(
        f"Joint band refinement of {len(found)} distinct blind candidates", flush=True
    )
    fits = []
    if geom_uncalibrated:
        stencil_fit = refine_band_geometry(
            Cs,
            Os,
            sampled_weights,
            B,
            rotations,
            np.asarray([item[0] for item in found], dtype=np.float64),
            wavelength,
            args.d_min,
            n_zones=60,
            radial_bound=r_bound,
            axial_bound=a_bound if (a_bound is not None and a_bound > 0.0) else None,
            rot_bound_deg=rot_bound,
            trans_bound_m=t_bound,
            maxiter=200,
            rot_axis_only=rot_axis_only,
            space_group=args.space_group,
            laue_order_search=True,
            return_top_k=max(int(candidates), 4),
        )
        fits.extend(stencil_fit.get("top_fits", [stencil_fit]))
        _write_blind_rung("joint_stencil", 60, stencil_fit["U"], geom_fit=stencil_fit)
        print(
            f"  laue-order stencil winner: score {stencil_fit['score']:.6g}, "
            f"radial {stencil_fit['radial_scale']:+.4f}, "
            f"axial {stencil_fit.get('axial_scale', stencil_fit['radial_scale']):+.4f}",
            flush=True,
        )
        found = found[:candidates]
        n_backup_target = max(0, min(3, int(candidates)) - len(fits))
        seeded_backups: list[tuple[np.ndarray, float, float]] = [
            (np.asarray(f_it["U"], dtype=np.float64), 1.0, 0.0) for f_it in fits
        ]
        n_backups_done = 0
        for item in found:
            if n_backups_done >= n_backup_target:
                break
            U = item[0]
            trial_b = seeded_backups + [(np.asarray(U, dtype=np.float64), 0.5, 0.0)]
            if len(
                distinct_orientations(
                    trial_b, B, args.space_group, len(trial_b), separation_deg=2.0
                )
            ) < len(trial_b):
                continue
            seeded_backups = trial_b
            fit = refine_band_geometry(
                Cs,
                Os,
                sampled_weights,
                B,
                rotations,
                U,
                wavelength,
                args.d_min,
                radial_bound=r_bound,
                axial_bound=a_bound
                if (a_bound is not None and a_bound > 0.0)
                else None,
                rot_bound_deg=rot_bound,
                trans_bound_m=t_bound,
                maxiter=80,
                rot_axis_only=rot_axis_only,
                space_group=args.space_group,
            )
            fits.append(fit)
            n_backups_done += 1
    else:
        for i, item in enumerate(found):
            U = item[0]
            fit = refine_band_geometry(
                Cs,
                Os,
                sampled_weights,
                B,
                rotations,
                U,
                wavelength,
                args.d_min,
                radial_bound=r_bound,
                rot_bound_deg=rot_bound,
                trans_bound_m=t_bound,
                maxiter=150,
                rot_axis_only=rot_axis_only,
                space_group=args.space_group,
            )
            fits.append(fit)
            best_fit_so_far = max(fits, key=lambda f_it: f_it["score"])
            _write_blind_rung(
                f"joint_{i}", 60, best_fit_so_far["U"], geom_fit=best_fit_so_far
            )
            print(
                f"  candidate {i}: score {fit['score']:.6g}, radial {fit['radial_scale']:+.4f}",
                flush=True,
            )

    R_em_override = getattr(args, "_recovered_outer_R_runs", None)
    if pixel_em is not None and len(fits) > 0:
        # Coarse candidate EM reweighting:
        # E-step evaluates each candidate's (valid, dark, bright) pixel posteriors
        # and null-subtracted Wilson reflection likelihood; M-step re-polishes top
        # finalists with bright/dark artifacts suppressed before selecting winner.
        em_evals = [
            pixel_em.evaluate_candidate(
                fit["U"],
                B,
                R_runs=R_em_override,
                radial_scale=float(fit["radial_scale"]),
                axial_scale=float(fit.get("axial_scale", fit["radial_scale"])),
                global_rot_vec=fit["global_rot_vec"],
                global_trans_m=fit["global_trans_m"],
                match_sigma_px=5.0,
            )
            for fit in fits
        ]
        top_em_order = np.argsort([-ev["em_score"] for ev in em_evals])
        m_step_limit = 1
        if len(top_em_order) >= 2:
            s0_em = float(em_evals[int(top_em_order[0])]["em_score"])
            s1_em = float(em_evals[int(top_em_order[1])]["em_score"])
            if s0_em < 2.0 * max(s1_em, 10.0):
                m_step_limit = 2
        for k_idx in top_em_order[: min(m_step_limit, len(fits))]:
            ev_k = em_evals[int(k_idx)]
            n_bright_k = sum(
                int(np.sum(v > 0.5)) for v in ev_k["gamma_bright"].values()
            )
            n_dark_k = sum(int(np.sum(v > 0.5)) for v in ev_k["gamma_dark"].values())
            if n_bright_k == 0 and n_dark_k == 0:
                continue
            # Only re-run nonlinear band M-step in blind discovery if bright
            # parasitic reflections were newly flagged (static dark panels are
            # already down-weighted in E-step 0 prior).
            if n_bright_k == 0:
                continue
            pixel_em.apply_responsibilities(
                ev_k["gamma_valid"], ev_k["gamma_dark"], ev_k["gamma_bright"]
            )
            dev_k, _, gv_k = pixel_em.map_to_observations(
                banks, rows, cols, runs_stable, dev_stable, weights
            )
            w_k = count_search_weights(dev_k, max_quantile=0.999, max_factor=3.0) * gv_k
            if np.any(w_k[idx] > 0.0):
                w_k_sub = w_k[idx] / max(np.abs(w_k[idx]).sum(), 1e-12)
                fk = fits[int(k_idx)]
                rb_k = r_bound
                ab_k = a_bound if (a_bound is not None and a_bound > 0.0) else None
                ref_k = refine_band_geometry(
                    Cs,
                    Os,
                    w_k_sub,
                    B,
                    rotations,
                    fk["U"],
                    wavelength,
                    args.d_min,
                    n_zones=60,
                    radial_bound=rb_k,
                    axial_bound=ab_k,
                    rot_bound_deg=rot_bound,
                    trans_bound_m=t_bound,
                    orient_bound_deg=2.5,
                    maxiter=80,
                    rot_axis_only=rot_axis_only,
                    space_group=args.space_group,
                )
                ev_ref = pixel_em.evaluate_candidate(
                    ref_k["U"],
                    B,
                    R_runs=R_em_override,
                    radial_scale=float(ref_k["radial_scale"]),
                    axial_scale=float(ref_k.get("axial_scale", ref_k["radial_scale"])),
                    global_rot_vec=ref_k["global_rot_vec"],
                    global_trans_m=ref_k["global_trans_m"],
                    match_sigma_px=4.5,
                )
                if ev_ref["em_score"] >= ev_k["em_score"] - 1.0:
                    fits[int(k_idx)] = ref_k
                    em_evals[int(k_idx)] = ev_ref

        # Deduplicate symmetry-identical / near-duplicate refined candidates,
        # keeping highest EM score first and trimming to `candidates`.
        ranked_idx = list(np.argsort([-ev["em_score"] for ev in em_evals]))
        kept_idx: list[int] = []
        kept_list: list[tuple[np.ndarray, float, float]] = []
        for ki in ranked_idx:
            cand_u = fits[int(ki)]["U"]
            trial = kept_list + [(cand_u, float(em_evals[int(ki)]["em_score"]), 0.0)]
            dedup = distinct_orientations(
                trial, B, args.space_group, len(trial), separation_deg=1.0
            )
            if len(dedup) == len(trial):
                kept_idx.append(int(ki))
                kept_list = trial
            if len(kept_idx) >= int(candidates):
                break
        if not kept_idx:
            kept_idx = ranked_idx[: int(candidates)]
        fits = [fits[i] for i in kept_idx]
        em_evals = [em_evals[i] for i in kept_idx]
        winner = 0
        win_ev = em_evals[winner]
        pixel_em.apply_responsibilities(
            win_ev["gamma_valid"], win_ev["gamma_dark"], win_ev["gamma_bright"]
        )
        dev_win, _, gv_win = pixel_em.map_to_observations(
            banks, rows, cols, runs_stable, dev_stable, weights
        )
        weights = (
            wilson_search_weights(dev_win, gv_win)
            if pixel_em.n_orient_clusters < 2
            else count_search_weights(dev_win, max_quantile=0.999, max_factor=3.0)
        )
        if np.any(weights[idx] > 0.0):
            sampled_weights = weights[idx] / max(np.abs(weights[idx]).sum(), 1e-12)
        n_dk = sum(int(np.sum(v > 0.5)) for v in win_ev["gamma_dark"].values())
        n_br = sum(int(np.sum(v > 0.5)) for v in win_ev["gamma_bright"].values())
        print(
            f"  EM candidate selection: winner=0/{len(fits)} "
            f"(em_score={win_ev['em_score']:.1f}, matched={win_ev['n_matched']}, "
            f"radial={fits[winner]['radial_scale']:+.4f}, "
            f"dark_bins={n_dk}, bright_bins={n_br})",
            flush=True,
        )
    else:
        winner = int(np.argmin([fit.get("loss", -fit["score"]) for fit in fits]))
    geometry = fits[winner]
    r_sc_win = float(geometry["radial_scale"])
    a_sc_win = float(geometry.get("axial_scale", r_sc_win))
    rot = Rotation.from_rotvec(geometry["global_rot_vec"]).as_matrix()
    C_shared = (
        _scaled_midpoint(C, r_sc_win, a_sc_win) @ rot.T + geometry["global_trans_m"]
    )
    O_shared = O @ rot.T
    U_shared = []
    for i_f, fit in enumerate(fits):
        if pixel_em is not None and i_f == winner:
            U_shared.append(np.asarray(fit["U"], dtype=np.float64))
            continue
        local = refine_band_geometry(
            C_shared[idx],
            O_shared[idx],
            sampled_weights,
            B,
            rotations,
            fit["U"],
            wavelength,
            args.d_min,
            n_zones=60,
            radial_bound=0,
            rot_bound_deg=0,
            trans_bound_m=0,
            orient_bound_deg=1.0 if geom_uncalibrated else 2.0,
            maxiter=60 if pixel_em is not None else 120,
            space_group=args.space_group,
        )
        U_shared.append(local["U"])
    xyz = C_shared + O_shared
    q = xyz / np.linalg.norm(xyz, axis=1, keepdims=True) - [0, 0, 1]
    lab = q / np.linalg.norm(q, axis=1, keepdims=True)
    D = np.einsum("nji,nj->ni", Rs, lab)
    lo, hi = band_q_range(-lab[:, 2], wavelength, args.d_min)
    uvw = direct_lattice_uvw(B, space_group=args.space_group)[
        : (145 if geom_uncalibrated else 60)
    ]
    X = uvw @ np.linalg.inv(B)
    w = weights / np.abs(weights).sum()
    import jax.numpy as jnp

    if pixel_em is not None:
        scores = []
        for i_u, U in enumerate(U_shared):
            if i_u == winner:
                scores.append(float(em_evals[winner]["em_score"]))
            else:
                ev_s = pixel_em.evaluate_candidate(
                    U,
                    B,
                    R_runs=R_em_override,
                    radial_scale=r_sc_win,
                    axial_scale=a_sc_win,
                    global_rot_vec=geometry["global_rot_vec"],
                    global_trans_m=geometry["global_trans_m"],
                    match_sigma_px=4.5,
                )
                scores.append(float(ev_s["em_score"]))
        args._em_bootstrapped = True
    else:
        scores = [
            float(
                band_sum(
                    jnp.asarray(X @ U.T),
                    jnp.asarray(D),
                    jnp.asarray(lo),
                    jnp.asarray(hi),
                    jnp.asarray(w),
                    jnp.ones(len(X)),
                )
            )
            for U in U_shared
        ]
    selected = int(np.argmax(scores))
    path = Path(args.output).with_suffix(".blind.h5")
    tmp_final = path.with_suffix(".blind.tmp.h5")
    with h5py.File(tmp_final, "w") as f:
        f.attrs["instrument"] = str(args.instrument)
        f["sample/U"] = U_shared[selected]
        f["sample/B"] = B
        f["sample/cell"] = np.asarray(args.cell, dtype=float)
        for k_c, v_c in zip(("a", "b", "c", "alpha", "beta", "gamma"), args.cell[:6]):
            f[f"sample/{k_c}"] = float(v_c)
        f["sample/space_group"] = str(args.space_group)
        f["instrument/name"] = str(args.instrument)
        f["instrument/wavelength"] = np.asarray([args.wl_min, args.wl_max], dtype=float)
        r_sc = float(geometry.get("radial_scale", 0.0))
        a_sc = float(geometry.get("axial_scale", r_sc))
        t_vec = np.asarray(geometry.get("global_trans_m", np.zeros(3)), dtype=float)
        cal_g = f.create_group("detector_calibration")
        for b_k, b_cfg in beamlines[args.instrument].items():
            try:
                b_id = int(b_k)
            except (ValueError, TypeError):
                continue
            det_b = Detector(dict(b_cfg))
            mid_b = (
                det_b.center
                + 0.5 * det_b.width * det_b.uhat
                + 0.5 * det_b.height * det_b.vhat
            )
            mid_cal = _scaled_midpoint(mid_b, r_sc, a_sc) @ rot.T + t_vec
            uh_cal = det_b.uhat @ rot.T
            vh_cal = det_b.vhat @ rot.T
            cen_cal = mid_cal - 0.5 * det_b.width * uh_cal - 0.5 * det_b.height * vh_cal
            bg = cal_g.create_group(f"bank_{b_id}")
            bg["center"] = np.asarray(cen_cal, dtype=float)
            bg["uhat"] = np.asarray(uh_cal, dtype=float)
            bg["vhat"] = np.asarray(vh_cal, dtype=float)
            bg["width"] = float(det_b.width)
            bg["height"] = float(det_b.height)
        g = f.create_group("search")
        g["orientations"] = U_shared
        g["full_scores"] = scores
        g["proposal_orientations"] = [v[0] for v in found]
        g["proposal_scores"] = [v[1] for v in found]
        g["joint_scores"] = [v["score"] for v in fits]
        g["sample_indices"] = idx
        g["sample_weights"] = sampled_weights
        g["shared_geometry"] = np.r_[
            r_sc,
            geometry["global_rot_vec"],
            geometry["global_trans_m"],
        ]
        g["shared_axial_scale"] = a_sc
        g["candidate_geometries"] = np.array(
            [
                np.r_[
                    v["radial_scale"],
                    v["global_rot_vec"],
                    v["global_trans_m"],
                ]
                for v in fits
            ],
            dtype=np.float64,
        )
        g["candidate_axial_scales"] = np.array(
            [float(v.get("axial_scale", v["radial_scale"])) for v in fits],
            dtype=np.float64,
        )
        g.attrs["selected"] = selected
        g.attrs["samples_requested"] = samples
        g.attrs["seed"] = seed
        g.attrs["d_min"] = float(args.d_min)
        g.attrs["stable_observations"] = len(weights)
        g.attrs["seconds"] = time.monotonic() - start
        g.attrs["source"] = str(Path(src_nexus).resolve()) if src_nexus else ""
        g.attrs["instrument"] = str(args.instrument)
        g.attrs["finder"] = str(args.finder or "")
        g.attrs["weighting"] = weighting_desc
    tmp_final.replace(path)
    print(
        f"Full-data scores under common geometry: {scores}; selected {selected}",
        flush=True,
    )
    return str(path)
