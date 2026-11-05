"""Staged zone-band and fixed-assignment pixel refinement."""

import h5py
import jax
import jax.numpy as jnp
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
    th = jnp.sqrt(jnp.sum(rv * rv) + 1e-30)
    k = rv / th
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return jnp.eye(3) + jnp.sin(th) * K + (1.0 - jnp.cos(th)) * (K @ K)


def rodrigues_batch_jax(rv):
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


def direct_lattice_uvw(B, max_index=6):
    A = np.linalg.inv(np.asarray(B, float)).T
    r = range(-max_index, max_index + 1)
    uvw = np.array(
        [(u, v, w) for u in r for v in r for w in r if (u, v, w) != (0, 0, 0)]
    )
    uvw = uvw[np.gcd.reduce(np.abs(uvw), axis=1) == 1]
    uvw = uvw[[tuple(x) > tuple(-x) for x in uvw]]
    X = uvw @ A.T
    xlen = np.linalg.norm(X, axis=1)
    o = np.argsort(xlen)
    return uvw[o]


def refine_scan(args):
    """Staged band refinement; configuration and input validation live in scan_refine."""
    from subhkl.scan_refine import load_observations, write_result, checkpoint

    diagnostics = []

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
        checkpoint(args, name, result.x, diagnostics)

    with h5py.File(args.bootstrap, "r") as f:
        U0 = np.array(f["sample/U"][()], dtype=np.float64)
        if U0.ndim == 3:
            U0 = U0[0]

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
    C_bank = np.array([dets[b].center for b in bank_ids])
    uhat_bank = np.array([dets[b].uhat for b in bank_ids])
    vhat_bank = np.array([dets[b].vhat for b in bank_ids])
    W_bank = np.array([dets[b].width for b in bank_ids])
    H_bank = np.array([dets[b].height for b in bank_ids])
    M_bank = np.array([float(dets[b].m - 1) for b in bank_ids])
    N_bank = np.array([float(dets[b].n - 1) for b in bank_ids])

    b_idx_all = np.array([bank_to_idx[b] for b in banks], dtype=np.int32)
    u_off_all = W_bank[b_idx_all] * (pc / M_bank[b_idx_all])
    v_off_all = H_bank[b_idx_all] * (pr / N_bank[b_idx_all])

    sel = np.isin(runs, args.runs)
    if args.finder:
        sel &= dev >= args.min_deviance
        weights_sel = spot_weights[sel]
    else:
        from subhkl.search.blind_band import count_search_weights

        weights_sel = count_search_weights(dev[sel], max_quantile=0.999, max_factor=3.0)
        if not np.any(weights_sel > 0):
            weights_sel = spot_weights[sel]
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

    uvw = direct_lattice_uvw(B0, max_index=6)[: args.n_zones]
    uvw_j = jnp.asarray(uvw, dtype=jnp.float64)
    w_r29 = jnp.where(jnp.isin(r_idx, jnp.asarray(args.stable_runs)), w_j, 0.0)
    w_r29 = w_r29 / jnp.maximum(jnp.abs(w_r29).sum(), 1e-12)

    def unpack(p):
        rv = p[0:3]
        s_rad = p[3]
        s_ax = p[4]
        s_area = p[5]
        rg = p[6:9]
        T = p[9:12]
        s_samp = p[12:15]
        da = p[15]
        db = p[16]
        dbeta = p[17]
        dom = p[18 : 18 + n_runs].at[args.anchor_run].set(0.0)
        du = p[18 + n_runs : 18 + n_runs + n_banks]
        dv = p[18 + n_runs + n_banks : 18 + n_runs + 2 * n_banks]
        return rv, s_rad, s_ax, s_area, rg, T, s_samp, da, db, dbeta, dom, du, dv

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
        reg = (
            reg_panel * (jnp.sum(du**2) + jnp.sum(dv**2))
            + 20.0 * (da**2 + db**2 + dbeta**2)
            + 500.0 * s_area**2
            + 5.0 * jnp.sum(dom**2)
            + 5.0 * (jnp.sum(rg**2) + jnp.sum((T / 0.03) ** 2))
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
            if mode not in args.modes:
                for i in indices:
                    bounds[i] = (0.0, 0.0)
        for slot in range(3):
            if slot not in active_cell_slots:
                bounds[15 + slot] = (0.0, 0.0)
        bounds[18 + args.anchor_run] = (0.0, 0.0)
        # Fix sample-shift gauge: for a single orientation setting, s_samp is
        # indistinguishable from global detector translation; for a multi-run
        # scan, fix the component along the dominant sample-frame rotation axis.
        if len(args.runs) <= 1 or np.allclose(R_obs[sel], R_obs[sel][0:1], atol=1e-6):
            for idx_s in (12, 13, 14):
                bounds[idx_s] = (0.0, 0.0)
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
    for s_init in (-0.05, 0.0, 0.055):
        p0 = np.zeros(18 + n_runs + 2 * n_banks)
        p0[3] = s_init
        p0[4] = s_init
        res = minimize(
            lambda x: (
                float(vg_band_60(jnp.asarray(x))[0]),
                np.asarray(vg_band_60(jnp.asarray(x))[1], np.float64),
            ),
            p0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_0,
            options={"maxiter": args.maxiter, "ftol": 1e-12},
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
        if r == args.anchor_run or "per_run" not in args.modes:
            continue
        if not np.any(runs[sel] == r):
            continue
        take = np.flatnonzero(runs[sel] == r)
        Cr, ur, vr = C_j[take], uh_j[take], vh_j[take]
        uor, vor = u_off[take], v_off[take]
        R0r, er, rid = R0_j[take], e_scan_j[take], r_idx[take]
        w_r = w_j[take]
        w_r = w_r / jnp.maximum(jnp.abs(w_r).sum(), 1e-12)

        @jax.jit
        def eval_dom(d_val, p_base=jnp.asarray(best_p), wr=w_r):
            p_try = p_base.at[18 + r].set(d_val)
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
            X0 = uvw_j[:60] @ A_cur.T
            wz = 1.0 / jnp.linalg.norm(X0, axis=1)
            Rm = rodrigues_jax(rv) @ Uj
            return band_sum(X0 @ Rm.T, d, q_lo, q_hi, wr, wz)

        scores_r = np.asarray(
            [float(eval_dom(d)) - 5.0 * float(d) ** 2 for d in dom_grid]
        )
        best_p[18 + r] = float(dom_grid[int(np.argmax(scores_r))])

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
        options={"maxiter": args.maxiter, "ftol": 1e-12},
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
        options={"maxiter": args.maxiter, "ftol": 1e-12},
    )
    best_p = res_1b.x.copy()
    record("cell", res_1b)
    print(
        f"[Stage 1b: {args.n_zones}-Zone Band+Axial+Cell] band_score = {-res_1b.fun:.4f}"
    )

    # Stage 2: Regularized per-bank (du, dv) shifts on 400-zone band objective
    if "panels" in args.modes:
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
    C_final = (
        C_sc + np.asarray(du)[:, None] * uhat_bank + np.asarray(dv)[:, None] * vhat_bank
    ) @ Rg.T + np.asarray(T)[None, :]
    uh_final = uhat_bank @ Rg.T
    vh_final = vhat_bank @ Rg.T
    W_final = W_bank * (1.0 + float(s_area))
    H_final = H_bank * (1.0 + float(s_area))

    lower, upper = np.asarray(bounds_panel).T
    args.parameter_bounds = np.asarray(bounds_panel)
    free = upper > lower
    args.bound_parameters = np.flatnonzero(
        free
        & (
            (p_cur - lower < 1e-4 * (upper - lower))
            | (upper - p_cur < 1e-4 * (upper - lower))
        )
    )
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
        p_cur,
        diagnostics,
    )
    print(f"Wrote staged band refinement to {args.output}", flush=True)
    print(
        f"  radial={float(s_rad) * 100:+.3f}%, axial={float(s_ax) * 100:+.3f}%",
        flush=True,
    )
    print(f"  per-run offsets (deg): {np.round(dom_deg, 3).tolist()}", flush=True)
    return dict(status=status, stages=diagnostics)
