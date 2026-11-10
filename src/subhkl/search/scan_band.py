"""Staged zone-band and fixed-assignment pixel refinement."""

import h5py
import numpy as np
from scipy.optimize import minimize, least_squares, OptimizeResult

from subhkl.config import beamlines
from subhkl.core.crystallography import (
    generate_reflections,
)
from subhkl.core.spacegroup import get_space_group_object
from subhkl.instrument.detector import Detector


def band_sum(X, directions, q_lo, q_hi, weights, zone_weights, chunk_size=2048):
    """Fourier band sum with bounded forward and reverse-mode memory."""
    import jax
    import jax.numpy as jnp

    def score(d, lo, hi, w):
        t = X @ d.T
        valid = (hi > lo)[None, :]
        hi_safe = jnp.where(hi > lo, hi, lo + 1.0)
        den = 2 * jnp.pi * (hi_safe - lo)[None, :] * t
        ok = jnp.abs(den) > 1e-7
        num = jnp.sin(2 * jnp.pi * hi_safe[None, :] * t) - jnp.sin(
            2 * jnp.pi * lo[None, :] * t
        )
        envelope = jnp.tanh(
            (hi_safe - lo)[None, :] / jnp.maximum(0.02 * lo[None, :], 1e-6)
        )
        kernel = jnp.where(
            valid, envelope * jnp.where(ok, num / jnp.where(ok, den, 1.0), 1.0), 0.0
        )
        return jnp.einsum("km,k,m->", kernel, zone_weights, w)

    if len(weights) <= chunk_size:
        return score(directions, q_lo, q_hi, weights)
    pad = (-len(weights)) % chunk_size

    def chunks(a):
        widths = [(0, pad)] + [(0, 0)] * (a.ndim - 1)
        return jnp.pad(a, widths).reshape((-1, chunk_size) + a.shape[1:])

    data = tuple(chunks(a) for a in (directions, q_lo, q_hi, weights))

    @jax.checkpoint
    def add(total, arrays):
        return total + score(*arrays), None

    return jax.lax.scan(add, jnp.array(0.0, dtype=X.dtype), data)[0]


def rodrigues_jax(rv):
    import jax.numpy as jnp

    th = jnp.sqrt(jnp.sum(rv * rv) + 1e-30)
    k = rv / th
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return jnp.eye(3) + jnp.sin(th) * K + (1.0 - jnp.cos(th)) * (K @ K)


def rodrigues_batch_jax(rv):
    import jax.numpy as jnp

    th = jnp.sqrt(jnp.sum(rv * rv, axis=-1, keepdims=True) + 1e-30)
    k = rv / th
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    z = jnp.zeros_like(kx)
    K = jnp.stack(
        [
            jnp.stack([z, -kz, ky], axis=-1),
            jnp.stack([kz, z, -kx], axis=-1),
            jnp.stack([-ky, kx, z], axis=-1),
        ],
        axis=-2,
    )
    th2 = th[..., None]
    return (
        jnp.eye(3)[None, :, :]
        + jnp.sin(th2) * K
        + (1.0 - jnp.cos(th2)) * jnp.matmul(K, K)
    )


def ewald_directions(G_lab):
    """Outgoing rays with finite derivatives for unassigned hkl=0 rows."""
    import jax.numpy as jnp

    G_sq = jnp.sum(G_lab**2, axis=1)
    lam = jnp.where(G_sq > 1e-12, -2 * G_lab[:, 2] / jnp.maximum(G_sq, 1e-12), 3.5)
    ray = jnp.array([0.0, 0.0, 1.0])[None, :] + lam[:, None] * G_lab
    return ray / jnp.linalg.norm(ray, axis=1, keepdims=True)


def parameter_units(n_runs, n_banks):
    return np.r_[
        np.full(3, 0.02),
        [0.05, 0.05, 0.005],
        np.full(3, 0.02),
        np.full(6, 0.005),
        np.full(3, 0.01),
        np.full(n_runs, 0.02),
        np.full(2 * n_banks, 0.001),
    ]


def build_B_jax(a, b, c, alpha_rad, beta_rad=None, gamma_rad=None):
    """Upper-triangular reciprocal-cell matrix matching cartesian_matrix_metric_tensor."""
    import jax.numpy as jnp

    if beta_rad is None:
        beta_rad = alpha_rad
        alpha_rad = gamma_rad = jnp.pi / 2.0
    ca, cb, cg = jnp.cos(alpha_rad), jnp.cos(beta_rad), jnp.cos(gamma_rad)
    G = jnp.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ]
    )
    return jnp.linalg.cholesky(jnp.linalg.inv(G)).T


def _cell_symmetry_spec(cell, space_group):
    """Determine active cell-shape slots in (da, db, dtheta) and angle mapping."""
    a0, b0, c0, al0, be0, ga0 = [float(v) for v in cell]
    sg = get_space_group_object(space_group)
    sys_str = str(sg.crystal_system()).split(".")[-1].lower()
    centering = sg.centring_type()
    if sys_str == "cubic":
        return "cubic", ()
    if sys_str == "tetragonal":
        return "tetragonal", (0,)
    if sys_str == "hexagonal":
        return "hexagonal", (0,)
    if sys_str == "trigonal":
        if (
            centering == "R"
            and np.allclose([a0, b0], [c0, c0], atol=0.05)
            and np.allclose([al0, be0], [ga0, ga0], atol=0.5)
        ):
            return "rhombohedral", (2,)
        return "hexagonal", (0,)
    if sys_str == "orthorhombic":
        return "orthorhombic", (0, 1)
    if sys_str == "monoclinic":
        non90 = int(np.argmax(np.abs(np.array([al0, be0, ga0]) - 90.0)))
        return f"monoclinic_{non90}", (0, 1, 2)
    return "triclinic", (0, 1, 2)


def _apply_cell_jax(cell0, symmetry_kind, da, db, dtheta):
    a0, b0, c0, al0, be0, ga0 = [float(v) for v in cell0]
    al_r, be_r, ga_r = np.deg2rad([al0, be0, ga0])
    if symmetry_kind == "cubic":
        return a0, b0, c0, al_r, be_r, ga_r
    if symmetry_kind in ("tetragonal", "hexagonal"):
        return a0 * (1.0 + da), b0 * (1.0 + da), c0, al_r, be_r, ga_r
    if symmetry_kind == "rhombohedral":
        return a0, b0, c0, al_r + dtheta, be_r + dtheta, ga_r + dtheta
    if symmetry_kind == "orthorhombic":
        return a0 * (1.0 + da), b0 * (1.0 + db), c0, al_r, be_r, ga_r
    if symmetry_kind == "monoclinic_0":
        return a0 * (1.0 + da), b0 * (1.0 + db), c0, al_r + dtheta, be_r, ga_r
    if symmetry_kind == "monoclinic_2":
        return a0 * (1.0 + da), b0 * (1.0 + db), c0, al_r, be_r, ga_r + dtheta
    return a0 * (1.0 + da), b0 * (1.0 + db), c0, al_r, be_r + dtheta, ga_r


def direct_lattice_uvw(B, max_index=6, space_group=None):
    from subhkl.search.spherical import direct_lattice_vectors

    return direct_lattice_vectors(B, max_index=max_index, space_group=space_group)[2]


def _build_ewald_excess_maps(args):
    """Build winsorized fine-binned local-median-subtracted excess maps per (run, bank)."""
    import jax
    import jax.numpy as jnp
    from subhkl.scan_refine import _frame_offsets

    with h5py.File(args.nexus, "r") as f:
        n_frames = int(f["images"].shape[0])
        H, W = int(f["images"].shape[1]), int(f["images"].shape[2])
        b2 = max(1, min(4, H // 4, W // 4))
        hb, wb = H // b2, W // b2
        bw = int(getattr(args, "border_width", 4))
        rim_bins = (bw + b2 - 1) // b2 if bw > 0 else 0
        offsets = _frame_offsets(f, n_frames)
        run_ids = np.searchsorted(offsets, np.arange(n_frames), side="right") - 1
        banks_f = f["bank_ids"][()].astype(int)
        raw_all = np.asarray(f["images"][()], dtype=np.float32)

    @jax.jit
    def _gpu_excess(raw_batch):
        nb = raw_batch.shape[0]
        y = raw_batch[:, : hb * b2, : wb * b2].reshape(nb, hb, b2, wb, b2).sum((2, 4))
        yp = jnp.pad(y, ((0, 0), (2, 2), (2, 2)), mode="edge")
        patches = jax.lax.conv_general_dilated_patches(
            yp[:, None, :, :],
            (5, 5),
            (1, 1),
            "VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        bg = jnp.median(patches, axis=1)
        ex = y - bg
        if rim_bins > 0 and hb > 4 * rim_bins and wb > 4 * rim_bins:
            ex = (
                ex.at[:, :rim_bins, :]
                .set(0.0)
                .at[:, -rim_bins:, :]
                .set(0.0)
                .at[:, :, :rim_bins]
                .set(0.0)
                .at[:, :, -rim_bins:]
                .set(0.0)
            )
        return ex

    ex_all = []
    for i0 in range(0, n_frames, 64):
        ex_all.append(np.asarray(_gpu_excess(jnp.asarray(raw_all[i0 : i0 + 64]))))
    ex_all = np.concatenate(ex_all, axis=0)

    exc_maps = {}
    has_signal = False
    for r in args.runs:
        idx_r = np.where(run_ids == r)[0]
        if len(idx_r) == 0:
            continue
        r_maps = {(int(r), int(banks_f[i])): ex_all[i] for i in idx_r}
        pos = np.concatenate([v[v > 0] for v in r_maps.values()])
        if len(pos) >= 10:
            has_signal = True
            cap = float(max(np.quantile(pos, 0.995) * 3.0, 1e-3))
        else:
            cap = 1e9
        for k, v in r_maps.items():
            exc_maps[k] = np.clip(v, -cap, cap)
    return exc_maps, b2, has_signal


def _make_ewald_polisher(
    G_c,
    run_ids,
    R_runs,
    bank_ids,
    C_corners,
    uhats,
    vhats,
    widths,
    heights,
    M_bank,
    N_bank,
    exc_maps,
    wl,
    b2,
    s_samp=None,
):
    import jax
    import jax.numpy as jnp
    from scipy.spatial.transform import Rotation as Rot

    runs = list(run_ids)
    banks = list(bank_ids)
    H = max(e.shape[0] for e in exc_maps.values())
    Wd = max(e.shape[1] for e in exc_maps.values())
    E = np.zeros((len(runs), len(banks), H, Wd), np.float32)
    r_map = {r_: i for i, r_ in enumerate(runs)}
    b_map = {bk: i for i, bk in enumerate(banks)}
    for (r_, bk), e in exc_maps.items():
        if r_ in r_map and bk in b_map:
            E[r_map[r_], b_map[bk], : e.shape[0], : e.shape[1]] = e

    cen = jnp.asarray(np.asarray(C_corners, np.float32))
    uh = jnp.asarray(np.asarray(uhats, np.float32))
    vh = jnp.asarray(np.asarray(vhats, np.float32))
    nrm = jnp.asarray(np.cross(uhats, vhats).astype(np.float32))
    dw = jnp.asarray(np.asarray(widths / np.maximum(M_bank, 1.0), np.float32))
    dh = jnp.asarray(np.asarray(heights / np.maximum(N_bank, 1.0), np.float32))
    nrow = jnp.asarray(np.asarray(N_bank + 1.0, np.float32))
    ncol = jnp.asarray(np.asarray(M_bank + 1.0, np.float32))
    Rr = jnp.asarray(np.array([R_runs[r_] for r_ in runs], np.float32))
    Gc = jnp.asarray(np.asarray(G_c, np.float32))
    kih = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    s_samp_j = jnp.asarray(
        np.zeros(3, np.float32) if s_samp is None else np.asarray(s_samp, np.float32)
    )
    Ej = jnp.asarray(E)
    lo_l, hi_l = np.float32(wl[0]), np.float32(wl[1])
    b2f = np.float32(b2)

    def score_single_run(Ej_r, R_r, U):
        G = (Gc @ U.T) @ R_r.T
        qn = jnp.linalg.norm(G, axis=1)
        sth = -(G @ kih) / jnp.maximum(qn, 1e-12)
        lam = 2.0 * sth / jnp.maximum(qn, 1e-12)
        band = (lam >= lo_l) & (lam <= hi_l) & (sth > 0)
        kf = kih[None, :] + lam[:, None] * G
        kf = kf / jnp.linalg.norm(kf, axis=1, keepdims=True)
        cen_r = cen - (R_r @ s_samp_j)[None, :]
        dn = kf @ nrm.T
        t = (cen_r * nrm).sum(1)[None, :] / jnp.where(jnp.abs(dn) < 1e-9, 1e-9, dn)
        row = (t * (kf @ vh.T) - (cen_r * vh).sum(1)[None, :]) / dh[None]
        col = (t * (kf @ uh.T) - (cen_r * uh).sum(1)[None, :]) / dw[None]
        ok = (
            band[:, None]
            & (t > 0)
            & (row >= 0)
            & (col >= 0)
            & (row < nrow[None])
            & (col < ncol[None])
        )
        u = row / b2f - 0.5
        v = col / b2f - 0.5
        i0 = jnp.floor(u)
        j0 = jnp.floor(v)
        fu = u - i0
        fv = v - j0
        ib = jnp.broadcast_to(jnp.arange(len(banks))[None, :], u.shape)
        i0 = i0.astype(jnp.int32)
        j0 = j0.astype(jnp.int32)

        def gat(di, dj):
            return Ej_r[ib, jnp.clip(i0 + di, 0, H - 1), jnp.clip(j0 + dj, 0, Wd - 1)]

        vals = (
            (1 - fu) * (1 - fv) * gat(0, 0)
            + (1 - fu) * fv * gat(0, 1)
            + fu * (1 - fv) * gat(1, 0)
            + fu * fv * gat(1, 1)
        )
        return jnp.where(ok, vals, 0.0).sum()

    @jax.jit
    def score_batch_runs(Us):
        return jax.vmap(
            lambda U: jax.vmap(score_single_run, in_axes=(0, 0, None))(Ej, Rr, U)
        )(Us)

    @jax.jit
    def score_per_run_grid(Urs, dRs):
        return jax.vmap(
            lambda Ej_r, R_r, U_r: jax.vmap(
                lambda dR: score_single_run(Ej_r, R_r, dR @ U_r)
            )(dRs),
            in_axes=(0, 0, 0),
        )(Ej, Rr, Urs)

    def eval_runs(Us):
        Us = np.asarray(Us, np.float32).reshape(-1, 3, 3)
        out = []
        for i in range(0, len(Us), 4):
            out.append(np.asarray(score_batch_runs(jnp.asarray(Us[i : i + 4]))))
        return np.concatenate(out, axis=0)

    def polish_joint(U, spans=(1.0, 0.35, 0.12, 0.04), n=7):
        U = np.asarray(U, np.float64)
        s_best = float(eval_runs(U).sum())
        for span in spans:
            ax = np.radians(np.linspace(-span, span, n))
            rv = np.array(np.meshgrid(ax, ax, ax, indexing="ij")).reshape(3, -1).T
            dR = Rot.from_rotvec(rv).as_matrix()
            Us = np.einsum("gij,jk->gik", dR, U)
            s = eval_runs(Us).sum(axis=1)
            idx = int(np.argmax(s))
            if float(s[idx]) > s_best + 1e-6:
                s_best = float(s[idx])
                U = Us[idx]
        return U, s_best

    def polish_per_run(U_joint, spans=(1.2, 0.8, 0.4, 0.15, 0.05), n=7):
        Rr_np = np.array([R_runs[r_] for r_ in runs], np.float64)
        Urs = np.repeat(np.asarray(U_joint, np.float64)[None], len(runs), axis=0)
        cur_s = np.array([float(eval_runs(Urs[ir])[0, ir]) for ir in range(len(runs))])
        for ispan, span in enumerate(spans):
            if ispan >= 1 and len(runs) >= 3:
                r_ref = int(np.argmax(cur_s))
                good = cur_s >= 0.55 * cur_s[r_ref]
                if good.sum() >= 2:
                    R_ref = Rr_np[r_ref]
                    U_ref = Urs[r_ref]

                    def res_fn(x):
                        D = Rot.from_rotvec(x[:3]).as_matrix()
                        U0 = Rot.from_rotvec(x[3:]).as_matrix() @ U_ref
                        errs = []
                        for ir in np.where(good)[0]:
                            U_p = (
                                Rr_np[ir].T @ D @ Rr_np[ir] @ R_ref.T @ D.T @ R_ref @ U0
                            )
                            errs.append(Rot.from_matrix(U_p @ Urs[ir].T).as_rotvec())
                        return np.concatenate(errs)

                    fit = least_squares(
                        res_fn,
                        np.zeros(6),
                        bounds=(-np.radians(4.0), np.radians(4.0)),
                        loss="soft_l1",
                        f_scale=0.005,
                    )
                    D = Rot.from_rotvec(fit.x[:3]).as_matrix()
                    U0 = Rot.from_rotvec(fit.x[3:]).as_matrix() @ U_ref
                    U_preds = np.array(
                        [
                            Rr_np[ir].T @ D @ Rr_np[ir] @ R_ref.T @ D.T @ R_ref @ U0
                            for ir in range(len(runs))
                        ]
                    )
                    pred_s = np.array(
                        [
                            float(eval_runs(U_preds[ir])[0, ir])
                            for ir in range(len(runs))
                        ]
                    )
                    for ir in range(len(runs)):
                        drift_deg = np.degrees(
                            Rot.from_matrix(Urs[ir] @ U_preds[ir].T).magnitude()
                        )
                        if pred_s[ir] > cur_s[ir] or drift_deg > 1.2:
                            Urs[ir] = U_preds[ir]
                            cur_s[ir] = pred_s[ir]
            ax = np.radians(np.linspace(-span, span, n))
            rv = np.array(np.meshgrid(ax, ax, ax, indexing="ij")).reshape(3, -1).T
            dR = Rot.from_rotvec(rv).as_matrix().astype(np.float32)
            scores_list = []
            for i in range(0, len(dR), 4):
                scores_list.append(
                    np.asarray(
                        score_per_run_grid(
                            jnp.asarray(Urs, jnp.float32),
                            jnp.asarray(dR[i : i + 4]),
                        )
                    )
                )
            scores = np.concatenate(scores_list, axis=1)
            best_g = np.argmax(scores, axis=1)
            for ir in range(len(runs)):
                if float(scores[ir, best_g[ir]]) > cur_s[ir] + 1e-6:
                    cur_s[ir] = float(scores[ir, best_g[ir]])
                    Urs[ir] = dR[best_g[ir]].astype(np.float64) @ Urs[ir]
        return Urs, cur_s

    return eval_runs, polish_joint, polish_per_run


def _polish_per_run_translations(
    args,
    U_opt,
    B_opt,
    cell_opt,
    R_runs_init,
    s_samp,
    bank_ids,
    C_final,
    uh_final,
    vh_final,
    W_final,
    H_final,
    M_bank,
    N_bank,
    exc_maps=None,
    b2_ew=4,
    obs=None,
    sel=None,
):
    """Refine per-run 6-DOF (dR_r, ds_lab_r) via 3-round robust ICP on local peak maxima."""
    from scipy.spatial.transform import Rotation as Rot

    bank_to_idx = {b: i for i, b in enumerate(bank_ids)}
    hkl = np.stack(
        generate_reflections(
            *cell_opt,
            space_group=args.space_group,
            d_min=args.d_min,
        ),
        axis=1,
    ).astype(np.float64)
    Gc = hkl @ B_opt.T
    kih = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nh_final = np.cross(uh_final, vh_final)
    dw = W_final / np.maximum(M_bank, 1.0)
    dh = H_final / np.maximum(N_bank, 1.0)
    wmin, wmax = float(args.assign_wavelength[0]), float(args.assign_wavelength[1])
    s_samp_vec = np.asarray(s_samp, dtype=np.float64)

    R_runs_out = [
        np.asarray(R_runs_init[r], dtype=np.float64).copy() for r in range(args.n_runs)
    ]
    trans_runs = np.zeros((args.n_runs, 3), dtype=np.float64)
    med_before_list, med_after_list = [], []

    for r in args.runs:
        if args.finder and obs is not None and sel is not None:
            banks_o, pr_o, pc_o, _dev_o, _sig_o, runs_o, _angs_o, weights_o = obs[:8]
            m_r = sel & (runs_o == r)
            b_r = np.array([bank_to_idx[int(b)] for b in banks_o[m_r]], dtype=np.int32)
            pr_r = np.asarray(pr_o[m_r], dtype=np.float64)
            pc_r = np.asarray(pc_o[m_r], dtype=np.float64)
            w_raw = np.maximum(np.asarray(weights_o[m_r], dtype=np.float64), 1e-6)
            if len(w_raw) > 300:
                order = np.argsort(w_raw)[-300:]
                b_r, pr_r, pc_r, w_raw = (
                    b_r[order],
                    pr_r[order],
                    pc_r[order],
                    w_raw[order],
                )
            w_r = (
                np.clip(
                    np.sqrt(w_raw / np.maximum(np.median(w_raw), 1e-6)),
                    0.5,
                    2.5,
                )
                if len(w_raw)
                else np.array([], dtype=np.float64)
            )
        elif exc_maps is not None:
            pos_r = [
                exc_maps[(r, b)][exc_maps[(r, b)] > 0]
                for b in bank_ids
                if (r, b) in exc_maps and np.any(exc_maps[(r, b)] > 0)
            ]
            if not pos_r:
                continue
            all_pos = np.concatenate(pos_r)
            thr = max(float(np.quantile(all_pos, 0.96)), 15.0)
            b_list, pr_list, pc_list, ex_vals = [], [], [], []
            for bi, b in enumerate(bank_ids):
                ex = exc_maps.get((r, b))
                if ex is None or ex.shape[0] <= 6 or ex.shape[1] <= 6:
                    continue
                H_e, W_e = ex.shape
                center = ex[3:-3, 3:-3]
                is_max = center >= thr
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        if di == 0 and dj == 0:
                            continue
                        is_max &= (
                            center >= ex[3 + di : H_e - 3 + di, 3 + dj : W_e - 3 + dj]
                        )
                ii, jj = np.nonzero(is_max)
                ii = ii + 3
                jj = jj + 3
                for i_m, j_m in zip(ii, jj):
                    patch = (
                        np.maximum(ex[i_m - 1 : i_m + 2, j_m - 1 : j_m + 2], 0.0) ** 2
                    )
                    ps = float(patch.sum())
                    if ps <= 0.0:
                        continue
                    di_c = float(patch[2, :].sum() - patch[0, :].sum()) / ps
                    dj_c = float(patch[:, 2].sum() - patch[:, 0].sum()) / ps
                    row_px = (i_m + di_c + 0.5) * float(b2_ew) - 0.5
                    col_px = (j_m + dj_c + 0.5) * float(b2_ew) - 0.5
                    b_list.append(bi)
                    pr_list.append(row_px)
                    pc_list.append(col_px)
                    ex_vals.append(float(ex[i_m, j_m]))
            if len(ex_vals) > 300:
                order = np.argsort(ex_vals)[-300:]
                b_r = np.array(b_list, dtype=np.int32)[order]
                pr_r = np.array(pr_list, dtype=np.float64)[order]
                pc_r = np.array(pc_list, dtype=np.float64)[order]
                w_raw = np.array(ex_vals, dtype=np.float64)[order]
            else:
                b_r = np.array(b_list, dtype=np.int32)
                pr_r = np.array(pr_list, dtype=np.float64)
                pc_r = np.array(pc_list, dtype=np.float64)
                w_raw = np.array(ex_vals, dtype=np.float64)
            w_r = (
                np.clip(
                    np.sqrt(w_raw / np.maximum(np.median(w_raw), 1.0)),
                    0.5,
                    2.5,
                )
                if len(w_raw)
                else np.array([], dtype=np.float64)
            )
        else:
            continue

        if len(b_r) < 8:
            continue

        R_r0 = R_runs_out[r]
        xs0 = R_r0 @ s_samp_vec

        def project_reflections(x_val):
            rv0, ds0 = x_val[:3], x_val[3:6]
            dR0 = Rot.from_rotvec(rv0).as_matrix()
            G0 = (Gc @ (dR0 @ U_opt).T) @ R_r0.T
            qn = np.linalg.norm(G0, axis=1)
            sth = -(G0 @ kih) / np.maximum(qn, 1e-12)
            lam = 2.0 * sth / np.maximum(qn, 1e-12)
            band = (lam >= wmin) & (lam <= wmax) & (sth > 0)
            idx_band = np.flatnonzero(band)
            if len(idx_band) == 0:
                return (
                    idx_band,
                    np.empty((0, len(bank_ids))),
                    np.empty((0, len(bank_ids))),
                    np.empty((0, len(bank_ids)), dtype=bool),
                )
            G_b, lam_b = G0[band], lam[band]
            kf = kih[None, :] + lam_b[:, None] * G_b
            kf /= np.linalg.norm(kf, axis=1, keepdims=True)
            cen_r = C_final - (xs0 + ds0)[None, :]
            dn = kf @ nh_final.T
            t = (cen_r * nh_final).sum(1)[None, :] / np.where(
                np.abs(dn) < 1e-9, 1e-9, dn
            )
            p = t[:, :, None] * kf[:, None, :] - cen_r[None, :, :]
            row = (p * vh_final[None]).sum(2) / dh[None]
            col = (p * uh_final[None]).sum(2) / dw[None]
            ok = (
                (t > 0)
                & (row >= 0)
                & (col >= 0)
                & (row <= N_bank[None])
                & (col <= M_bank[None])
            )
            return idx_band, row, col, ok

        def get_matches(x_val, radius):
            idx_band, row, col, ok = project_reflections(x_val)
            m_obs, m_hkl = [], []
            if len(idx_band) == 0:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            for i_obs in range(len(b_r)):
                bi = b_r[i_obs]
                valid_h = np.flatnonzero(ok[:, bi])
                if not len(valid_h):
                    continue
                d2 = (row[valid_h, bi] - pr_r[i_obs]) ** 2 + (
                    col[valid_h, bi] - pc_r[i_obs]
                ) ** 2
                k = int(np.argmin(d2))
                if d2[k] < radius**2:
                    m_obs.append(i_obs)
                    m_hkl.append(idx_band[valid_h[k]])
            return np.array(m_obs, dtype=np.int32), np.array(m_hkl, dtype=np.int32)

        def fit_matches(x_init, m_obs, m_hkl, f_sc=2.5, prior_ds=0.45):
            if len(m_obs) < 8:
                return x_init
            b_m = b_r[m_obs]
            rm = pr_r[m_obs]
            cm = pc_r[m_obs]
            wm = w_r[m_obs]
            Gc_m = Gc[m_hkl]
            C_m = C_final[b_m]
            uh_m = uh_final[b_m]
            vh_m = vh_final[b_m]
            nh_m = nh_final[b_m]
            dw_m = dw[b_m]
            dh_m = dh[b_m]

            def res_fn(x):
                rv, ds = x[:3], x[3:6]
                dR = Rot.from_rotvec(rv).as_matrix()
                G = (Gc_m @ (dR @ U_opt).T) @ R_r0.T
                q = np.linalg.norm(G, axis=1)
                st = -(G @ kih) / np.maximum(q, 1e-12)
                lm = 2.0 * st / np.maximum(q, 1e-12)
                kf = kih[None, :] + lm[:, None] * G
                kf /= np.linalg.norm(kf, axis=1, keepdims=True)
                cen_m = C_m - (xs0 + ds)[None, :]
                denom = np.sum(kf * nh_m, axis=1)
                denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
                t = (cen_m * nh_m).sum(1) / denom
                p = t[:, None] * kf - cen_m
                rp = np.sum(p * vh_m, axis=1) / dh_m
                cp = np.sum(p * uh_m, axis=1) / dw_m
                return np.concatenate(
                    [
                        wm * (rm - rp),
                        wm * (cm - cp),
                        (ds * 1000.0) * prior_ds,
                        np.degrees(rv) * 1.5,
                    ]
                )

            sol = least_squares(
                res_fn,
                x_init,
                bounds=(
                    [-0.03, -0.03, -0.03, -0.008, -0.008, -0.008],
                    [0.03, 0.03, 0.03, 0.008, 0.008, 0.008],
                ),
                loss="soft_l1",
                f_scale=f_sc,
            )
            return sol.x

        def eval_med_dist(x_val, m_obs, m_hkl):
            if len(m_obs) == 0:
                return 999.0
            rv, ds = x_val[:3], x_val[3:6]
            dR = Rot.from_rotvec(rv).as_matrix()
            G = (Gc[m_hkl] @ (dR @ U_opt).T) @ R_r0.T
            q = np.linalg.norm(G, axis=1)
            st = -(G @ kih) / np.maximum(q, 1e-12)
            lm = 2.0 * st / np.maximum(q, 1e-12)
            kf = kih[None, :] + lm[:, None] * G
            kf /= np.linalg.norm(kf, axis=1, keepdims=True)
            b_m = b_r[m_obs]
            cen_m = C_final[b_m] - (xs0 + ds)[None, :]
            denom = np.sum(kf * nh_final[b_m], axis=1)
            denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
            t = (cen_m * nh_final[b_m]).sum(1) / denom
            p = t[:, None] * kf - cen_m
            rp = np.sum(p * vh_final[b_m], axis=1) / dh[b_m]
            cp = np.sum(p * uh_final[b_m], axis=1) / dw[b_m]
            return float(np.median(np.hypot(pr_r[m_obs] - rp, pc_r[m_obs] - cp)))

        mo0, mh0 = get_matches(np.zeros(6), 18.0)
        if len(mo0) < 8:
            continue
        d_before = eval_med_dist(np.zeros(6), mo0, mh0)
        x = fit_matches(np.zeros(6), mo0, mh0, f_sc=3.5, prior_ds=0.45)
        mo1, mh1 = get_matches(x, 14.0)
        x = fit_matches(x, mo1, mh1, f_sc=2.0, prior_ds=0.45)
        mo2, mh2 = get_matches(x, 8.0)
        x = fit_matches(x, mo2, mh2, f_sc=1.5, prior_ds=0.45)
        d_after = eval_med_dist(x, mo0, mh0)
        med_before_list.append(d_before)
        if d_after < d_before:
            dR_r = Rot.from_rotvec(x[:3]).as_matrix()
            R_new = R_r0 @ dR_r
            R_runs_out[r] = R_new
            trans_runs[r] = R_new.T @ (xs0 + x[3:6]) - s_samp_vec
            med_after_list.append(d_after)
        else:
            med_after_list.append(d_before)

    return R_runs_out, trans_runs, med_before_list, med_after_list


def refine_scan(args):
    """Staged band refinement; configuration and input validation live in scan_refine."""
    import jax
    import jax.numpy as jnp
    from subhkl.scan_refine import load_observations, write_result, checkpoint

    diagnostics = []
    _snapshot_builder = None

    def record(name, result):
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(
                f"{name} refinement returned nonfinite objective or parameters"
            )
        diagnostics.append(
            dict(
                stage=name,
                objective=float(result.fun),
                iterations=int(result.nit),
                success=bool(result.success),
                message=str(result.message),
            )
        )
        snap = None
        if callable(_snapshot_builder):
            try:
                snap = _snapshot_builder(name, result.x)
            except Exception:
                snap = None
        checkpoint(args, name, result.x, diagnostics, snapshot=snap)

    with h5py.File(args.bootstrap, "r") as f:
        U0 = np.array(f["sample/U"][()], dtype=np.float64)
        if U0.ndim == 3:
            U0 = U0[0]
        cand_Us = [U0]
        if "blind_search/candidate_U" in f:
            for Uc in np.asarray(f["blind_search/candidate_U"][()], dtype=np.float64)[
                :6
            ]:
                if not any(np.allclose(Uc, Uprev, atol=1e-4) for Uprev in cand_Us):
                    cand_Us.append(Uc)
        shared_geom = (
            np.asarray(f["search/shared_geometry"][()], dtype=np.float64)
            if "search/shared_geometry" in f
            else None
        )

    cell0 = [float(v) for v in args.cell]
    symmetry_kind, active_cell_slots = _cell_symmetry_spec(cell0, args.space_group)
    B0 = np.asarray(
        build_B_jax(*cell0[:3], *np.deg2rad(cell0[3:])),
        dtype=np.float64,
    )

    obs = load_observations(args, return_kinematics=True)
    banks, pr, pc, dev, sig_px, runs, _angs, spot_weights, R_obs, scan_axes_obs = obs
    n_runs = args.n_runs
    bank_ids = sorted(int(k) for k in beamlines[args.instrument].keys())
    bank_to_idx = {b: i for i, b in enumerate(bank_ids)}
    n_banks = len(bank_ids)

    dets = {b: Detector(dict(beamlines[args.instrument][str(b)])) for b in bank_ids}
    C_bank_corner = np.array([dets[b].center for b in bank_ids])
    uhat_bank = np.array([dets[b].uhat for b in bank_ids])
    vhat_bank = np.array([dets[b].vhat for b in bank_ids])
    W_bank = np.array([dets[b].width for b in bank_ids])
    H_bank = np.array([dets[b].height for b in bank_ids])
    M_bank = np.array([float(dets[b].m - 1) for b in bank_ids])
    N_bank = np.array([float(dets[b].n - 1) for b in bank_ids])
    # Detector.center is the (row=0, col=0) corner; center panel scaling and
    # offsets on the physical panel midpoint so radial scaling does not introduce
    # a spurious half-panel tangential shift.
    C_bank = (
        C_bank_corner
        + 0.5 * W_bank[:, None] * uhat_bank
        + 0.5 * H_bank[:, None] * vhat_bank
    )

    b_idx_all = np.array([bank_to_idx[b] for b in banks], dtype=np.int32)
    u_off_all = W_bank[b_idx_all] * (pc / M_bank[b_idx_all] - 0.5)
    v_off_all = H_bank[b_idx_all] * (pr / N_bank[b_idx_all] - 0.5)

    sel = np.isin(runs, args.runs)
    if args.finder:
        sel &= dev >= args.min_deviance
        weights_sel = spot_weights[sel]
    else:
        from subhkl.search.blind_band import count_search_weights

        weights_sel = count_search_weights(dev[sel], max_quantile=0.999, max_factor=3.0)
        if not np.any(weights_sel > 0):
            weights_sel = spot_weights[sel]

    is_single_setting = len(args.runs) <= 1 or np.allclose(
        R_obs[sel], R_obs[sel][0:1], atol=1e-6
    )
    per_run_active = "per_run" in args.modes or "per_run_trans" in args.modes

    R_runs_nom = {}
    for r in range(n_runs):
        idx_r = np.flatnonzero(runs == r)
        R_runs_nom[r] = R_obs[idx_r[0]].copy() if len(idx_r) else np.eye(3)

    ewald_ctx = None
    if not args.pixel_polish and args.maxiter > 2:
        exc_maps, b2_ew, has_ew_signal = _build_ewald_excess_maps(args)
        if has_ew_signal:
            hkl_ew = np.stack(
                generate_reflections(
                    *cell0,
                    space_group=args.space_group,
                    d_min=args.d_min,
                ),
                axis=1,
            ).astype(np.float64)
            eval_nom, polish_joint_nom, polish_per_run_nom = _make_ewald_polisher(
                hkl_ew @ B0.T,
                args.runs,
                R_runs_nom,
                bank_ids,
                C_bank_corner,
                uhat_bank,
                vhat_bank,
                W_bank,
                H_bank,
                M_bank,
                N_bank,
                exc_maps,
                args.assign_wavelength,
                b2_ew,
            )
            best_u_nom, best_s_nom = U0, -1e30
            for Uc in cand_Us[:4]:
                Up, sp = polish_joint_nom(Uc, spans=(1.0, 0.35), n=7)
                if sp > best_s_nom:
                    best_s_nom, best_u_nom = sp, Up
            U_nom_joint, ew_nom_joint = polish_joint_nom(
                best_u_nom, spans=(0.12, 0.04), n=7
            )
            if per_run_active and not is_single_setting:
                Urs_nom, s_nom_r = polish_per_run_nom(U_nom_joint)
                ew_nom = float(s_nom_r.sum())
                R_runs_pre = dict(R_runs_nom)
                for ir, r in enumerate(args.runs):
                    R_runs_pre[r] = R_runs_nom[r] @ Urs_nom[ir] @ U_nom_joint.T
                used_pre_polish = (
                    shared_geom is None or abs(float(shared_geom[0])) < 0.015
                )
                if used_pre_polish:
                    for r in args.runs:
                        R_obs[runs == r] = R_runs_pre[r]
                    U0 = U_nom_joint
            else:
                Urs_nom = np.repeat(U_nom_joint[None], len(args.runs), axis=0)
                ew_nom = float(ew_nom_joint)
                R_runs_pre = dict(R_runs_nom)
                used_pre_polish = (
                    shared_geom is None or abs(float(shared_geom[0])) < 0.015
                )
                if used_pre_polish:
                    U0 = U_nom_joint
            ewald_ctx = dict(
                exc_maps=exc_maps,
                b2=b2_ew,
                hkl_ew=hkl_ew,
                U_nom_joint=U_nom_joint,
                Urs_nom=Urs_nom,
                ew_nom=ew_nom,
                R_runs_pre=R_runs_pre,
                R_runs_band_base=R_runs_pre if used_pre_polish else R_runs_nom,
            )

    b_idx = jnp.asarray(b_idx_all[sel], dtype=jnp.int32)
    u_off = jnp.asarray(u_off_all[sel], dtype=jnp.float64)
    v_off = jnp.asarray(v_off_all[sel], dtype=jnp.float64)
    R0_j = jnp.asarray(R_obs[sel], dtype=jnp.float64)
    e_scan_j = jnp.asarray(scan_axes_obs[sel], dtype=jnp.float64)
    r_idx = jnp.asarray(runs[sel], dtype=jnp.int32)
    w_j = jnp.asarray(
        weights_sel / np.maximum(np.abs(weights_sel).sum(), 1e-12), dtype=jnp.float64
    )

    C_j = jnp.asarray(C_bank[b_idx_all[sel]], dtype=jnp.float64)
    uh_j = jnp.asarray(uhat_bank[b_idx_all[sel]], dtype=jnp.float64)
    vh_j = jnp.asarray(vhat_bank[b_idx_all[sel]], dtype=jnp.float64)
    Uj = jnp.asarray(U0, dtype=jnp.float64)
    kij = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)

    uvw = direct_lattice_uvw(B0, max_index=6, space_group=args.space_group)[
        : args.n_zones
    ]
    uvw_j = jnp.asarray(uvw, dtype=jnp.float64)
    w_r29 = jnp.where(jnp.isin(r_idx, jnp.asarray(args.stable_runs)), w_j, 0.0)
    w_r29 = w_r29 / jnp.maximum(jnp.abs(w_r29).sum(), 1e-12)
    rot_axis = getattr(args, "global_rot_axis", (0.0, 1.0, 0.0))
    if rot_axis is not None:
        ax_arr = np.asarray(rot_axis, dtype=np.float64)
        rot_axis_unit = ax_arr / np.linalg.norm(ax_arr)
        rot_dom_idx = int(np.argmax(np.abs(rot_axis_unit)))
        if rot_axis_unit[rot_dom_idx] < 0:
            rot_axis_unit = -rot_axis_unit
        rot_axis_j = jnp.asarray(rot_axis_unit, dtype=jnp.float64)
    else:
        rot_axis_unit = None
        rot_dom_idx = None
        rot_axis_j = None

    def unpack(p):
        rv = p[0:3]
        s_rad = p[3]
        s_ax = s_rad if is_single_setting and "axial_stretch" in args.modes else p[4]
        s_area = p[5]
        rg = p[6 + rot_dom_idx] * rot_axis_j if rot_axis_j is not None else p[6:9]
        T = p[9:12]
        s_samp = p[12:15]
        da = p[15]
        db = p[16]
        dbeta = p[17]
        dom = p[18 : 18 + n_runs].at[args.anchor_run].set(0.0)
        du = p[18 + n_runs : 18 + n_runs + n_banks]
        dv = p[18 + n_runs + n_banks : 18 + n_runs + 2 * n_banks]
        return rv, s_rad, s_ax, s_area, rg, T, s_samp, da, db, dbeta, dom, du, dv

    def _build_stage_snapshot(stage_name, p_vec):
        rv_s, sr_s, sa_s, sar_s, rg_s, T_s, _ss_s, da_s, db_s, dbe_s, dom_s, du_s, dv_s = unpack(
            jnp.asarray(p_vec, dtype=jnp.float64)
        )
        Rg_s = np.asarray(rodrigues_jax(rg_s))
        U_s = np.asarray(rodrigues_jax(rv_s) @ Uj)
        a_s, b_s, c_s, al_sr, be_sr, ga_sr = [
            float(v) for v in _apply_cell_jax(cell0, symmetry_kind, da_s, db_s, dbe_s)
        ]
        al_sd, be_sd, ga_sd = np.degrees([al_sr, be_sr, ga_sr]).tolist()
        B_s = np.asarray(build_B_jax(a_s, b_s, c_s, al_sr, be_sr, ga_sr))
        C_sc_s = np.stack(
            [
                C_bank[:, 0] * (1.0 + float(sr_s)),
                C_bank[:, 1] * (1.0 + float(sa_s)),
                C_bank[:, 2] * (1.0 + float(sr_s)),
            ],
            axis=-1,
        )
        W_s = W_bank * (1.0 + float(sar_s))
        H_s = H_bank * (1.0 + float(sar_s))
        uh_s = uhat_bank @ Rg_s.T
        vh_s = vhat_bank @ Rg_s.T
        C_mid_s = (
            C_sc_s
            + np.asarray(du_s)[:, None] * uhat_bank
            + np.asarray(dv_s)[:, None] * vhat_bank
        ) @ Rg_s.T + np.asarray(T_s)[None, :]
        C_fin_s = (
            C_mid_s - 0.5 * W_s[:, None] * uh_s - 0.5 * H_s[:, None] * vh_s
        )
        R_runs_ref = getattr(args, "R_runs_refined", None)
        if R_runs_ref is not None:
            R_list_s = [np.asarray(R_runs_ref[r], dtype=float) for r in range(n_runs)]
        else:
            R_base_s = (
                ewald_ctx["R_runs_band_base"]
                if (ewald_ctx is not None and "R_runs_band_base" in ewald_ctx)
                else R_runs_nom
            )
            R_list_s = []
            for r in range(n_runs):
                idx_r = np.flatnonzero(runs == r)
                if len(idx_r):
                    dR_r = np.asarray(
                        rodrigues_jax(
                            float(dom_s[r]) * jnp.asarray(scan_axes_obs[idx_r[0]])
                        )
                    )
                    R_list_s.append(dR_r @ R_base_s[r])
                else:
                    R_list_s.append(np.asarray(R_base_s[r], dtype=float))
        nz_stage = (
            60
            if stage_name in ("global", "ewald-pre")
            else (120 if stage_name == "120-zones" else int(args.n_zones))
        )
        return {
            "U": U_s,
            "B": B_s,
            "cell": [a_s, b_s, c_s, al_sd, be_sd, ga_sd],
            "banks": bank_ids,
            "centers": C_fin_s,
            "uhats": uh_s,
            "vhats": vh_s,
            "widths": W_s,
            "heights": H_s,
            "R": np.stack(R_list_s, axis=0),
            "n_zones": nz_stage,
        }

    _snapshot_builder = _build_stage_snapshot
    if ewald_ctx is not None:
        try:
            p_pre = np.zeros(18 + n_runs + 2 * n_banks, dtype=np.float64)
            checkpoint(
                args,
                "ewald-pre",
                p_pre,
                [
                    dict(
                        stage="ewald-pre",
                        objective=float(-ewald_ctx.get("ew_nom", 0.0)),
                        iterations=1,
                        success=True,
                        message="EWALD_PRE_POLISH",
                    )
                ],
                snapshot=_build_stage_snapshot("ewald-pre", p_pre),
            )
        except Exception:
            pass

    def loss_band(p, n_z=400, reg_panel=200.0, use_r29=False, inv_len_weight=False):
        rv, s_rad, s_ax, s_area, rg, T, s_samp, da, db, dbeta, dom, du, dv = unpack(p)
        Rg = rodrigues_jax(rg)
        C_scaled = jnp.stack(
            [
                C_j[:, 0] * (1.0 + s_rad),
                C_j[:, 1] * (1.0 + s_ax),
                C_j[:, 2] * (1.0 + s_rad),
            ],
            axis=-1,
        )
        O_scaled = (u_off * (1.0 + s_area) + du[b_idx])[:, None] * uh_j + (
            v_off * (1.0 + s_area) + dv[b_idx]
        )[:, None] * vh_j
        xyz_det = (C_scaled + O_scaled) @ Rg.T + T[None, :]

        dR = rodrigues_batch_jax(dom[r_idx][:, None] * e_scan_j)
        R_run = jnp.einsum("nij,njk->nik", dR, R0_j)
        x_s = jnp.einsum("nij,j->ni", R_run, s_samp)
        xyz = xyz_det - x_s
        kf = xyz / jnp.linalg.norm(xyz, axis=1, keepdims=True)
        g = kf - kij[None, :]
        d_lab = g / jnp.linalg.norm(g, axis=1, keepdims=True)
        s_ = -(d_lab @ kij)
        q_lo = 2.0 * s_ / args.wl_max
        q_hi = jnp.maximum(jnp.minimum(2.0 * s_ / args.wl_min, 1.0 / args.d_min), q_lo)
        d = jnp.einsum("nji,nj->ni", R_run, d_lab)

        B_cur = build_B_jax(*_apply_cell_jax(cell0, symmetry_kind, da, db, dbeta))
        A_cur = jnp.linalg.inv(B_cur).T
        X0 = uvw_j[:n_z] @ A_cur.T
        Rm = rodrigues_jax(rv) @ Uj
        w_spots = w_r29 if use_r29 else w_j
        wz = 1.0 / jnp.linalg.norm(X0, axis=1) if inv_len_weight else jnp.ones(len(X0))
        score = band_sum(X0 @ Rm.T, d, q_lo, q_hi, w_spots, wz)
        reg_geom = 0.1 if is_single_setting else 5.0
        reg_cell = 1000.0 if is_single_setting else 20.0
        reg = (
            reg_panel * (jnp.sum(du**2) + jnp.sum(dv**2))
            + reg_cell * (da**2 + db**2 + dbeta**2)
            + 500.0 * s_area**2
            + 5.0 * jnp.sum(dom**2)
            + reg_geom
            * (
                jnp.sum(rg**2)
                + jnp.sum((T / 0.15) ** 2)
                + jnp.sum((s_samp / 0.015) ** 2)
            )
        )
        return -score + reg

    bounds_0 = (
        [(-0.2, 0.2)] * 3
        + [(-0.15, 0.15), (-0.15, 0.15), (0.0, 0.0)]
        + [(-0.08, 0.08)] * 3
        + [(-0.03, 0.03)] * 3
        + [(0.0, 0.0)] * 3
        + [(0.0, 0.0)] * 3
        + [(0.0, 0.0)] * n_runs
        + [(0.0, 0.0)] * (2 * n_banks)
    )
    bounds_120 = (
        [(-0.2, 0.2)] * 3
        + [(-0.15, 0.15), (-0.15, 0.15), (0.0, 0.0)]
        + [(-0.08, 0.08)] * 3
        + [(-0.03, 0.03)] * 3
        + [(-0.015, 0.015)] * 3
        + [(0.0, 0.0)] * 3
        + [(-np.deg2rad(6.0), np.deg2rad(6.0))] * n_runs
        + [(0.0, 0.0)] * (2 * n_banks)
    )
    bounds_global = (
        [(-0.2, 0.2)] * 3
        + [(-0.15, 0.15), (-0.15, 0.15), (-0.01, 0.01)]
        + [(-0.08, 0.08)] * 3
        + [(-0.03, 0.03)] * 3
        + [(-0.015, 0.015)] * 3
        + [(-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02)]
        + [(-np.deg2rad(6.0), np.deg2rad(6.0))] * n_runs
        + [(0.0, 0.0)] * (2 * n_banks)
    )
    bounds_panel = bounds_global[: 18 + n_runs] + [(-0.008, 0.008)] * (2 * n_banks)

    # Disable unrequested degrees of freedom in every stage, including polish.
    blocks = {
        "cylindrical": [3],
        "axial_stretch": [4],
        "area": [5],
        "global_rot": list(range(6, 9)),
        "global_trans": list(range(9, 12)),
        "sample": list(range(12, 15)),
        "cell": list(range(15, 18)),
        "per_run": list(range(18, 18 + n_runs)),
        "panels": list(range(18 + n_runs, 18 + n_runs + 2 * n_banks)),
    }
    for bounds in (bounds_0, bounds_120, bounds_global, bounds_panel):
        for mode, indices in blocks.items():
            mode_on = per_run_active if mode == "per_run" else (mode in args.modes)
            if not mode_on:
                for i in indices:
                    bounds[i] = (0.0, 0.0)
        for slot in range(3):
            if slot not in active_cell_slots:
                bounds[15 + slot] = (0.0, 0.0)
        bounds[18 + args.anchor_run] = (0.0, 0.0)
        if rot_axis_unit is not None:
            for i in range(3):
                if i != rot_dom_idx:
                    bounds[6 + i] = (0.0, 0.0)
        # Fix sample-shift gauge: for a single orientation setting, s_samp is
        # indistinguishable from global detector translation; for a multi-run
        # scan, fix the component along the dominant sample-frame rotation axis.
        if is_single_setting:
            bounds[4] = (0.0, 0.0)
            bounds[5] = (0.0, 0.0)
            for idx_s in (12, 13, 14):
                bounds[idx_s] = (0.0, 0.0)
            for idx_p in range(18 + n_runs, 18 + n_runs + 2 * n_banks):
                bounds[idx_p] = (0.0, 0.0)
        else:
            axis_samp = np.einsum("nji,nj->ni", R_obs[sel], scan_axes_obs[sel]).mean(
                axis=0
            )
            gauge_dim = int(np.argmax(np.abs(axis_samp)))
            bounds[12 + gauge_dim] = (0.0, 0.0)
        for r in set(range(n_runs)) - set(args.runs):
            bounds[18 + r] = (0.0, 0.0)

    vg_band_60 = jax.jit(
        jax.value_and_grad(lambda p: loss_band(p, 60, 0.0, True, False))
    )
    vg_band_120 = jax.jit(
        jax.value_and_grad(lambda p: loss_band(p, 120, 0.0, False, False))
    )
    vg_band_400 = jax.jit(
        jax.value_and_grad(lambda p: loss_band(p, args.n_zones, 0.0, False, False))
    )
    vg_band_p = jax.jit(
        jax.value_and_grad(lambda p: loss_band(p, args.n_zones, 200.0, False, False))
    )

    # Stage 0: coarse global geometry on the caller's stable-run subset.
    best_p, best_v, best_result = None, 1e9, None
    s_inits = (
        (-0.05, 0.0, 0.055)
        if ("cylindrical" in args.modes or "axial_stretch" in args.modes)
        else (0.0,)
    )
    for s_init in s_inits:
        p0 = np.zeros(18 + n_runs + 2 * n_banks)
        p0[3] = s_init
        p0[4] = s_init
        if (
            shared_geom is not None
            and len(shared_geom) == 7
            and abs(s_init - float(shared_geom[0])) <= 0.03
        ):
            p0[3] = float(shared_geom[0])
            p0[4] = float(shared_geom[0])
            if "global_rot" in args.modes:
                p0[6:9] = shared_geom[1:4]
            if "global_trans" in args.modes:
                p0[9:12] = shared_geom[4:7]
        p0 = np.clip(p0, *np.asarray(bounds_0).T)
        res = minimize(
            lambda x: (
                float(vg_band_60(jnp.asarray(x))[0]),
                np.asarray(vg_band_60(jnp.asarray(x))[1], np.float64),
            ),
            p0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_0,
            options={"maxiter": args.maxiter, "ftol": 1e-9},
        )
        if res.fun < best_v:
            best_v, best_p, best_result = res.fun, res.x.copy(), res
    record("global", best_result)
    print(
        f"[Stage 0: 60-zone stable runs {args.stable_runs}] band_score = {-best_v:.4f}",
        flush=True,
    )

    # Initialize per-run omega drift via 1D scan on 120-zone band score
    dom_grid = np.deg2rad(np.linspace(-5.0, 5.0, 201))
    for r in args.runs:
        if (
            (
                ewald_ctx is not None
                and ewald_ctx.get("R_runs_band_base") is ewald_ctx.get("R_runs_pre")
            )
            or r == args.anchor_run
            or not per_run_active
        ):
            continue
        if not np.any(runs[sel] == r):
            continue
        take = np.flatnonzero(runs[sel] == r)
        Cr, ur, vr = C_j[take], uh_j[take], vh_j[take]
        uor, vor = u_off[take], v_off[take]
        R0r, er, rid = R0_j[take], e_scan_j[take], r_idx[take]
        w_r = w_j[take]
        w_r = w_r / jnp.maximum(jnp.abs(w_r).sum(), 1e-12)
        p_base_j = jnp.asarray(best_p)

        @jax.jit
        def eval_dom(d_val):
            p_try = p_base_j.at[18 + r].set(d_val)
            rv, s_rad, s_ax, s_area, rg, T, s_samp, da, db, dbeta, dom, du, dv = unpack(
                p_try
            )
            Rg = rodrigues_jax(rg)
            C_scaled = jnp.stack(
                [
                    Cr[:, 0] * (1.0 + s_rad),
                    Cr[:, 1] * (1.0 + s_ax),
                    Cr[:, 2] * (1.0 + s_rad),
                ],
                axis=-1,
            )
            O_scaled = (uor * (1.0 + s_area))[:, None] * ur + (vor * (1.0 + s_area))[
                :, None
            ] * vr
            xyz_det = (C_scaled + O_scaled) @ Rg.T + T[None, :]
            dR = rodrigues_batch_jax(dom[rid][:, None] * er)
            R_run = jnp.einsum("nij,njk->nik", dR, R0r)
            xyz = xyz_det - jnp.einsum("nij,j->ni", R_run, s_samp)
            kf = xyz / jnp.linalg.norm(xyz, axis=1, keepdims=True)
            g = kf - kij[None, :]
            d_lab = g / jnp.linalg.norm(g, axis=1, keepdims=True)
            s_ = -(d_lab @ kij)
            q_lo = 2.0 * s_ / args.wl_max
            q_hi = jnp.maximum(
                jnp.minimum(2.0 * s_ / args.wl_min, 1.0 / args.d_min),
                q_lo,
            )
            d = jnp.einsum("nji,nj->ni", R_run, d_lab)
            A_cur = jnp.linalg.inv(jnp.asarray(B0)).T
            X0 = uvw_j[:120] @ A_cur.T
            wz = jnp.ones(len(X0))
            Rm = rodrigues_jax(rv) @ Uj
            return band_sum(X0 @ Rm.T, d, q_lo, q_hi, w_r, wz)

        scores_r = np.asarray(
            [float(eval_dom(d)) - 5.0 * float(d) ** 2 for d in dom_grid]
        )
        best_p[18 + r] = float(dom_grid[int(np.argmax(scores_r))])
    best_p[18 : 18 + n_runs] -= best_p[18 + args.anchor_run]

    # Stage 1a: 120-zone band across all selected runs, including scan drift.
    res_1a = minimize(
        lambda x: (
            float(vg_band_120(jnp.asarray(x))[0]),
            np.asarray(vg_band_120(jnp.asarray(x))[1], np.float64),
        ),
        best_p,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_120,
        options={"maxiter": args.maxiter, "ftol": 1e-9},
    )
    best_p = res_1a.x.copy()
    record("120-zones", res_1a)
    print(f"[Stage 1a: 120-Zone Band (All Runs)] band_score = {-res_1a.fun:.4f}")

    # Stage 1b: Fine 400-zone band + monoclinic cell ratios
    res_1b = minimize(
        lambda x: (
            float(vg_band_400(jnp.asarray(x))[0]),
            np.asarray(vg_band_400(jnp.asarray(x))[1], np.float64),
        ),
        best_p,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_global,
        options={"maxiter": args.maxiter, "ftol": 1e-9},
    )
    best_p = res_1b.x.copy()
    record("cell", res_1b)
    print(
        f"[Stage 1b: {args.n_zones}-Zone Band+Axial+Cell] band_score = {-res_1b.fun:.4f}"
    )

    # Stage 2: Regularized per-bank (du, dv) shifts on 400-zone band objective
    if "panels" in args.modes and not is_single_setting:
        units = parameter_units(n_runs, n_banks)

        def panel_objective(z):
            value, gradient = vg_band_p(jnp.asarray(z * units))
            return float(value), np.asarray(gradient, float) * units

        res_p = minimize(
            panel_objective,
            best_p / units,
            method="L-BFGS-B",
            jac=True,
            bounds=np.asarray(bounds_panel) / units[:, None],
            options={
                "maxiter": args.maxiter,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxcor": 30,
            },
        )
        res_p.x *= units
        p_cur = res_p.x.copy()
        record("panels", res_p)
        print(f"[Stage 2: +Regularized Per-Bank Shifts] loss = {res_p.fun:.4f}")
    else:
        p_cur = best_p.copy()

    if args.pixel_polish:
        # Stage 3: 2-step median-trimmed pixel residual polish on assigned reflections
        hkl_pool = np.stack(
            generate_reflections(
                *cell0,
                space_group=args.space_group,
                d_min=args.d_min,
            ),
            axis=1,
        ).astype(np.float64)
        sel_pol = sel
        b_p = b_idx_all[sel_pol]
        u_p = u_off_all[sel_pol]
        v_p = v_off_all[sel_pol]
        R0_p = R_obs[sel_pol]
        e_p = scan_axes_obs[sel_pol]
        r_p = runs[sel_pol]

        rv, s_rad, s_ax, s_area, rg, T, s_samp, da, db, dbeta, dom, du, dv = unpack(
            jnp.asarray(p_cur)
        )
        Rg = np.asarray(rodrigues_jax(rg))
        U_c = np.asarray(rodrigues_jax(rv) @ Uj)
        B_c = np.asarray(
            build_B_jax(*_apply_cell_jax(cell0, symmetry_kind, da, db, dbeta))
        )
        UB_c = U_c @ B_c
        G_pool = hkl_pool @ UB_c.T
        G_norm = np.linalg.norm(G_pool, axis=1)
        G_hat = G_pool / G_norm[:, None]

        C_sc = np.stack(
            [
                C_bank[:, 0] * (1.0 + float(s_rad)),
                C_bank[:, 1] * (1.0 + float(s_ax)),
                C_bank[:, 2] * (1.0 + float(s_rad)),
            ],
            axis=-1,
        )
        C_new = (
            C_sc
            + np.asarray(du)[:, None] * uhat_bank
            + np.asarray(dv)[:, None] * vhat_bank
        ) @ Rg.T + np.asarray(T)[None, :]
        uh_new = uhat_bank @ Rg.T
        vh_new = vhat_bank @ Rg.T

        hkl_assigned = np.zeros((len(b_p), 3), dtype=np.float64)
        inlier = np.zeros(len(b_p), dtype=np.float64)
        for i in range(len(b_p)):
            bi = b_p[i]
            ri = r_p[i]
            dR_i = np.asarray(rodrigues_jax(float(dom[ri]) * jnp.asarray(e_p[i])))
            Rr = dR_i @ R0_p[i]
            xs = Rr @ np.asarray(s_samp)
            xyz = (
                C_new[bi]
                + u_p[i] * (1.0 + float(s_area)) * uh_new[bi]
                + v_p[i] * (1.0 + float(s_area)) * vh_new[bi]
                - xs
            )
            kf = xyz / np.linalg.norm(xyz)
            gh = kf - np.array([0.0, 0.0, 1.0])
            d_cryst = Rr.T @ (gh / np.linalg.norm(gh))
            st = -d_cryst @ (Rr.T @ np.array([0.0, 0.0, 1.0]))
            wl_cand = 2.0 * st / G_norm
            m_wl = (wl_cand >= args.assign_wavelength[0]) & (
                wl_cand <= args.assign_wavelength[1]
            )
            idx_m = np.where(m_wl)[0]
            if len(idx_m) == 0:
                continue
            dots = G_hat[idx_m] @ d_cryst
            best_j = idx_m[np.argmax(dots)]
            if np.degrees(np.arccos(np.clip(dots.max(), -1.0, 1.0))) < 0.30:
                hkl_assigned[i] = hkl_pool[best_j]
                inlier[i] = 1.0

        b_pj = jnp.asarray(b_p, dtype=jnp.int32)
        u_pj = jnp.asarray(u_p, dtype=jnp.float64)
        v_pj = jnp.asarray(v_p, dtype=jnp.float64)
        R0_pj = jnp.asarray(R0_p, dtype=jnp.float64)
        e_pj = jnp.asarray(e_p, dtype=jnp.float64)
        r_pj = jnp.asarray(r_p, dtype=jnp.int32)
        hkl_pj = jnp.asarray(hkl_assigned, dtype=jnp.float64)
        C_bj = jnp.asarray(C_bank, dtype=jnp.float64)
        uh_bj = jnp.asarray(uhat_bank, dtype=jnp.float64)
        vh_bj = jnp.asarray(vhat_bank, dtype=jnp.float64)
        W_bj = jnp.asarray(W_bank, dtype=jnp.float64)
        H_bj = jnp.asarray(H_bank, dtype=jnp.float64)
        M_bj = jnp.asarray(M_bank, dtype=jnp.float64)
        N_bj = jnp.asarray(N_bank, dtype=jnp.float64)

        def make_loss_pix(inl_arr):
            inl_j = jnp.asarray(inl_arr, dtype=jnp.float64)

            def loss_pix(p):
                (
                    rv,
                    s_rad,
                    s_ax,
                    s_area,
                    rg,
                    T,
                    s_samp,
                    da,
                    db,
                    dbeta,
                    dom,
                    du,
                    dv,
                ) = unpack(p)
                Rg = rodrigues_jax(rg)
                U_c = rodrigues_jax(rv) @ Uj
                B_c = build_B_jax(*_apply_cell_jax(cell0, symmetry_kind, da, db, dbeta))
                G_c = hkl_pj @ (U_c @ B_c).T
                dR = rodrigues_batch_jax(dom[r_pj][:, None] * e_pj)
                Rr = jnp.einsum("nij,njk->nik", dR, R0_pj)
                G_lab = jnp.einsum("nij,nj->ni", Rr, G_c)
                kf_pred = ewald_directions(G_lab)
                xs = jnp.einsum("nij,j->ni", Rr, s_samp)

                C_sc = jnp.stack(
                    [
                        C_bj[:, 0] * (1.0 + s_rad),
                        C_bj[:, 1] * (1.0 + s_ax),
                        C_bj[:, 2] * (1.0 + s_rad),
                    ],
                    axis=-1,
                )
                C_n = (C_sc + du[:, None] * uh_bj + dv[:, None] * vh_bj) @ Rg.T + T[
                    None, :
                ]
                uh_n = uh_bj @ Rg.T
                vh_n = vh_bj @ Rg.T
                nh_n = jnp.cross(uh_n, vh_n)

                C_spot = C_n[b_pj] - xs
                uh_s = uh_n[b_pj]
                vh_s = vh_n[b_pj]
                nh_s = nh_n[b_pj]
                denominator = jnp.sum(kf_pred * nh_s, axis=1)
                denominator = jnp.where(
                    jnp.abs(denominator) < 1e-12,
                    jnp.where(denominator < 0, -1e-12, 1e-12),
                    denominator,
                )
                t_ray = jnp.sum(C_spot * nh_s, axis=1) / denominator
                P_hit = t_ray[:, None] * kf_pred - C_spot
                u_hit = jnp.sum(P_hit * uh_s, axis=1)
                v_hit = jnp.sum(P_hit * vh_s, axis=1)
                du_px = (u_hit - u_pj * (1.0 + s_area)) * (
                    M_bj[b_pj] / (W_bj[b_pj] * (1.0 + s_area))
                )
                dv_px = (v_hit - v_pj * (1.0 + s_area)) * (
                    N_bj[b_pj] / (H_bj[b_pj] * (1.0 + s_area))
                )
                r2 = jnp.where(inl_j > 0, du_px**2 + dv_px**2, 0.0)
                delta = 2.0
                huber = jnp.where(
                    r2 <= delta**2,
                    0.5 * r2,
                    delta * (jnp.sqrt(r2 + 1e-12) - 0.5 * delta),
                )
                # Residual representation of the same radial Huber objective
                # and priors, for a bounded trust-region least-squares solve.
                # Preserve both pixel coordinates in the Gauss-Newton model.
                # Collapsing each spot to a scalar radius loses its transverse
                # curvature and makes nearly exact spots converge very slowly.
                factor = jnp.sqrt(
                    jnp.where(r2 <= delta**2, 1.0, 2 * huber / jnp.maximum(r2, 1e-20))
                )
                factor *= jnp.sqrt(inl_j / jnp.maximum(jnp.sum(inl_j), 1.0))
                data_residual = (
                    jnp.stack([du_px, dv_px], axis=1) * factor[:, None]
                ).ravel()
                return jnp.concatenate(
                    [
                        data_residual,
                        # Retain the band stage's panel prior during polish.
                        # Dropping it lets sparsely observed panels hit bounds.
                        jnp.sqrt(400.0) * du,
                        jnp.sqrt(400.0) * dv,
                        jnp.sqrt(10.0) * jnp.stack([da, db, dbeta, s_area]),
                    ]
                )

            return loss_pix

        def polish(p, mask):
            residual = make_loss_pix(mask)
            # Metres, radians and fractional scales have very different
            # curvature. Optimize dimensionless steps of comparable size.
            units = np.r_[
                np.full(3, 0.02),
                [0.05, 0.05, 0.005],
                np.full(3, 0.02),
                np.full(6, 0.005),
                np.full(3, 0.01),
                np.full(n_runs, 0.02),
                np.full(2 * n_banks, 0.001),
            ]
            lower, upper = np.asarray(bounds_panel).T
            free = np.flatnonzero(upper > lower)
            base = jnp.asarray(np.clip(p, lower, upper))

            def reduced(x):
                return residual(base.at[free].set(x))

            fun, jac = jax.jit(reduced), jax.jit(jax.jacfwd(reduced))
            result = least_squares(
                lambda x: np.asarray(fun(x), float),
                np.asarray(base)[free],
                jac=lambda x: np.asarray(jac(x), float),
                bounds=(lower[free], upper[free]),
                x_scale=units[free],
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-8,
                max_nfev=args.maxiter,
            )
            solution = np.asarray(base).copy()
            solution[free] = result.x
            return OptimizeResult(
                x=solution,
                fun=result.cost,
                nit=result.nfev,
                success=result.success,
                message=result.message,
                optimality=result.optimality,
            )

        res_pol1 = polish(p_cur, inlier)
        p_cur = res_pol1.x.copy()
        record("pixel-1", res_pol1)

        def eval_residuals_px(p_vec):
            rv, s_rad, s_ax, s_area, rg, T, s_samp, da, db, dbeta, dom, du, dv = unpack(
                jnp.asarray(p_vec)
            )
            Rg = np.asarray(rodrigues_jax(rg))
            U_c = np.asarray(rodrigues_jax(rv) @ Uj)
            B_c = np.asarray(
                build_B_jax(*_apply_cell_jax(cell0, symmetry_kind, da, db, dbeta))
            )
            C_sc = np.stack(
                [
                    C_bank[:, 0] * (1.0 + float(s_rad)),
                    C_bank[:, 1] * (1.0 + float(s_ax)),
                    C_bank[:, 2] * (1.0 + float(s_rad)),
                ],
                axis=-1,
            )
            C_new = (
                C_sc
                + np.asarray(du)[:, None] * uhat_bank
                + np.asarray(dv)[:, None] * vhat_bank
            ) @ Rg.T + np.asarray(T)[None, :]
            uh_new = uhat_bank @ Rg.T
            vh_new = vhat_bank @ Rg.T
            nh_new = np.cross(uh_new, vh_new)
            res_px = np.full(len(b_p), 999.0)
            for i in range(len(b_p)):
                if inlier[i] == 0:
                    continue
                bi, ri = b_p[i], r_p[i]
                dR_i = np.asarray(rodrigues_jax(float(dom[ri]) * jnp.asarray(e_p[i])))
                Rr = dR_i @ R0_p[i]
                xs = Rr @ np.asarray(s_samp)
                G_l = Rr @ (U_c @ B_c @ hkl_assigned[i])
                lam = -2.0 * G_l[2] / np.sum(G_l**2)
                kfp = np.array([0.0, 0.0, 1.0]) + lam * G_l
                kfp = kfp / np.linalg.norm(kfp)
                Cs = C_new[bi] - xs
                tr = np.dot(Cs, nh_new[bi]) / np.dot(kfp, nh_new[bi])
                Ph = tr * kfp - Cs
                du_p = (np.dot(Ph, uh_new[bi]) - u_p[i] * (1.0 + float(s_area))) * (
                    M_bank[bi] / (W_bank[bi] * (1.0 + float(s_area)))
                )
                dv_p = (np.dot(Ph, vh_new[bi]) - v_p[i] * (1.0 + float(s_area))) * (
                    N_bank[bi] / (H_bank[bi] * (1.0 + float(s_area)))
                )
                res_px[i] = np.hypot(du_p, dv_p)
            return res_px

        res_px1 = eval_residuals_px(p_cur)
        n_assigned = int((inlier > 0).sum())
        for bi in np.unique(b_p):
            m_b = (b_p == bi) & (inlier > 0)
            if m_b.sum() >= 3:
                thr = max(1.8 * float(np.median(res_px1[m_b])), 2.2)
                inlier[m_b & (res_px1 > thr)] = 0.0

        res_pol2 = polish(p_cur, inlier)
        p_cur = res_pol2.x.copy()
        record("pixel-2", res_pol2)
        res_px2 = eval_residuals_px(p_cur)
        args.pixel_diagnostics = {
            "peak_index": np.flatnonzero(sel_pol),
            "hkl": hkl_assigned.astype(int),
            "inlier": inlier.astype(bool),
            "residual_px": res_px2,
        }
        m_inl = inlier > 0
        r_inl = res_px2[m_inl]
        s_inl = sig_px[sel_pol][m_inl]
        print(
            f"[Stage 3: Pixel Residual Polish] inliers={int(m_inl.sum())}/{n_assigned}, "
            f"med={np.median(r_inl):.2f} px, "
            f"<2sigma={100.0 * np.mean(r_inl <= 2.0 * s_inl):.1f}%"
        )

    # Export calibrated indexer HDF5
    rv, s_rad, s_ax, s_area, rg, T, s_samp, da, db, dbeta, dom, du, dv = unpack(
        jnp.asarray(p_cur)
    )
    Rg = np.asarray(rodrigues_jax(rg))
    U_opt = np.asarray(rodrigues_jax(rv) @ Uj)
    a_opt, b_opt, c_opt, al_opt_r, be_opt_r, ga_opt_r = [
        float(v) for v in _apply_cell_jax(cell0, symmetry_kind, da, db, dbeta)
    ]
    al_opt, be_opt, ga_opt = np.degrees([al_opt_r, be_opt_r, ga_opt_r]).tolist()
    B_opt = np.asarray(build_B_jax(a_opt, b_opt, c_opt, al_opt_r, be_opt_r, ga_opt_r))
    dom_deg = np.degrees(np.asarray(dom))

    C_sc = np.stack(
        [
            C_bank[:, 0] * (1.0 + float(s_rad)),
            C_bank[:, 1] * (1.0 + float(s_ax)),
            C_bank[:, 2] * (1.0 + float(s_rad)),
        ],
        axis=-1,
    )
    W_final = W_bank * (1.0 + float(s_area))
    H_final = H_bank * (1.0 + float(s_area))
    uh_final = uhat_bank @ Rg.T
    vh_final = vhat_bank @ Rg.T
    C_midpoint_final = (
        C_sc + np.asarray(du)[:, None] * uhat_bank + np.asarray(dv)[:, None] * vhat_bank
    ) @ Rg.T + np.asarray(T)[None, :]
    C_final = (
        C_midpoint_final
        - 0.5 * W_final[:, None] * uh_final
        - 0.5 * H_final[:, None] * vh_final
    )

    if ewald_ctx is not None:
        from scipy.spatial.transform import Rotation as Rot

        R_runs_band = dict(ewald_ctx["R_runs_band_base"])
        for r in args.runs:
            idx_r = np.flatnonzero(runs == r)
            if len(idx_r):
                dR_r = np.asarray(
                    rodrigues_jax(float(dom[r]) * jnp.asarray(scan_axes_obs[idx_r[0]]))
                )
                R_runs_band[r] = dR_r @ R_runs_band[r]
        eval_cal, polish_joint_cal, polish_per_run_cal = _make_ewald_polisher(
            ewald_ctx["hkl_ew"] @ B_opt.T,
            args.runs,
            R_runs_band,
            bank_ids,
            C_final,
            uh_final,
            vh_final,
            W_final,
            H_final,
            M_bank,
            N_bank,
            ewald_ctx["exc_maps"],
            args.assign_wavelength,
            ewald_ctx["b2"],
            s_samp=np.asarray(s_samp),
        )
        U_cal_1, s_cal_1 = polish_joint_cal(U_opt)
        U_cal_2, s_cal_2 = polish_joint_cal(ewald_ctx["U_nom_joint"])
        if s_cal_2 > s_cal_1:
            U_cal_joint, ew_cal_joint = U_cal_2, s_cal_2
        else:
            U_cal_joint, ew_cal_joint = U_cal_1, s_cal_1
        if per_run_active and not is_single_setting:
            Urs_cal, s_cal_r = polish_per_run_cal(U_cal_joint)
            ew_cal = float(s_cal_r.sum())
            used_per_run_cal = True
        else:
            Urs_cal = np.repeat(U_cal_joint[None], len(args.runs), axis=0)
            ew_cal = float(ew_cal_joint)
            used_per_run_cal = False

        ew_nom = float(ewald_ctx["ew_nom"])
        if ew_cal >= ew_nom:
            U_opt = U_cal_joint
            p_cur[0:3] = Rot.from_matrix(U_opt @ np.asarray(Uj).T).as_rotvec()
            if used_per_run_cal:
                R_runs_final = dict(R_runs_band)
                for ir, r in enumerate(args.runs):
                    R_runs_final[r] = R_runs_band[r] @ Urs_cal[ir] @ U_cal_joint.T
                args.R_runs_refined = [R_runs_final[r] for r in range(n_runs)]
            ew_best = ew_cal
        else:
            U_opt = ewald_ctx["U_nom_joint"]
            p_cur[:] = 0.0
            s_rad = jnp.float64(0.0)
            s_ax = jnp.float64(0.0)
            rg = jnp.zeros(3, dtype=jnp.float64)
            s_samp = jnp.zeros(3, dtype=jnp.float64)
            dom_deg = np.zeros(n_runs, dtype=np.float64)
            a_opt, b_opt, c_opt, al_opt, be_opt, ga_opt = cell0
            B_opt = B0.copy()
            C_final = C_bank_corner.copy()
            uh_final = uhat_bank.copy()
            vh_final = vhat_bank.copy()
            W_final = W_bank.copy()
            H_final = H_bank.copy()
            if per_run_active and not is_single_setting:
                args.R_runs_refined = [
                    ewald_ctx["R_runs_pre"][r] for r in range(n_runs)
                ]
            ew_best = ew_nom
        record(
            "ewald-polish",
            OptimizeResult(
                x=p_cur.copy(),
                fun=-ew_best,
                nit=5,
                success=True,
                message="CONVERGENCE: EWALD_REFLECTION_POLISH",
            ),
        )
        print(
            f"[Stage 3: Ewald Reflection Polish] ewald_score = {ew_best:.1f} "
            f"(cal={ew_cal:.1f}, nom={ew_nom:.1f})",
            flush=True,
        )

    if "per_run_trans" in args.modes and not is_single_setting:
        if getattr(args, "R_runs_refined", None) is not None:
            R_runs_init = list(args.R_runs_refined)
        else:
            R_runs_init = []
            for r in range(n_runs):
                idx_r = np.flatnonzero(runs == r)
                if len(idx_r):
                    dR_r = np.asarray(
                        rodrigues_jax(
                            float(dom[r]) * jnp.asarray(scan_axes_obs[idx_r[0]])
                        )
                    )
                    R_runs_init.append(dR_r @ R_obs[idx_r[0]])
                else:
                    R_runs_init.append(R_runs_nom[r])
        exc_maps_pol = ewald_ctx["exc_maps"] if ewald_ctx is not None else None
        b2_pol = ewald_ctx["b2"] if ewald_ctx is not None else 4
        if exc_maps_pol is None and not args.finder:
            exc_maps_pol, b2_pol, _ = _build_ewald_excess_maps(args)
        R_runs_pol, trans_runs_pol, med_b, med_a = _polish_per_run_translations(
            args,
            U_opt,
            B_opt,
            [a_opt, b_opt, c_opt, al_opt, be_opt, ga_opt],
            R_runs_init,
            np.asarray(s_samp),
            bank_ids,
            C_final,
            uh_final,
            vh_final,
            W_final,
            H_final,
            M_bank,
            N_bank,
            exc_maps=exc_maps_pol,
            b2_ew=b2_pol,
            obs=obs,
            sel=sel,
        )
        args.R_runs_refined = R_runs_pol
        args.trans_runs_refined = trans_runs_pol
        max_t_mm = float(1e3 * np.linalg.norm(trans_runs_pol, axis=1).max())
        if med_b:
            print(
                f"[Stage 4: Per-Run Translation Polish] med_px = "
                f"{np.median(med_b):.2f} -> {np.median(med_a):.2f}, "
                f"max_trans = {max_t_mm:.2f} mm",
                flush=True,
            )

    lower, upper = np.asarray(bounds_panel).T
    args.parameter_bounds = np.asarray(bounds_panel)
    free = upper > lower
    if ewald_ctx is not None:
        args.bound_parameters = np.array([], dtype=int)
    else:
        args.bound_parameters = np.flatnonzero(
            free
            & (
                (p_cur - lower < 1e-4 * (upper - lower))
                | (upper - p_cur < 1e-4 * (upper - lower))
            )
        )
    p_out = p_cur.copy()
    p_out[6:9] = np.asarray(rg, dtype=np.float64)
    status = write_result(
        args,
        U_opt,
        B_opt,
        [a_opt, b_opt, c_opt, al_opt, be_opt, ga_opt],
        bank_ids,
        C_final,
        uh_final,
        vh_final,
        W_final,
        H_final,
        dom_deg,
        np.asarray(s_samp),
        p_out,
        diagnostics,
    )
    print(f"Wrote staged band refinement to {args.output}", flush=True)
    print(
        f"  radial={float(s_rad) * 100:+.3f}%, axial={float(s_ax) * 100:+.3f}%",
        flush=True,
    )
    print(f"  per-run offsets (deg): {np.round(dom_deg, 3).tolist()}", flush=True)
    return dict(status=status, stages=diagnostics)
