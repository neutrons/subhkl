"""Staged zone-band and fixed-assignment pixel refinement."""

import h5py
import numpy as np
from scipy.optimize import OptimizeResult, least_squares, minimize

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
        sg_ext = getattr(sg, "ext", "")
        raw_sg = str(space_group or "").strip().upper()
        if sg_ext == "R" or (
            centering == "R"
            and not raw_sg.endswith(":H")
            and np.allclose([a0, b0], [c0, c0], atol=0.05)
            and np.allclose([al0, be0], [ga0, ga0], atol=0.5)
            and not np.isclose(al0, 90.0, atol=0.5)
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
        cell_arg = getattr(args, "cell", None)
        d_min_arg = float(getattr(args, "d_min", 2.0) or 2.0)
        long_axis_ratio = (
            max(float(cell_arg[0]), float(cell_arg[1]), float(cell_arg[2]))
            / max(d_min_arg, 0.5)
            if cell_arg is not None and len(cell_arg) >= 3
            else 0.0
        )
        # Keep >=128x128 Ewald excess bins on 256-px panels (e.g. MANDI/TOPAZ) and
        # long-axis cells so sub-0.4-deg reciprocal rungs are not Nyquist-blurred.
        target_b2 = 2 if (min(H, W) <= 256 or long_axis_ratio >= 85.0) else 4
        b2 = max(1, min(target_b2, H // 4, W // 4))
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
        hb2 = hb // 2
        wb2 = wb // 2
        if hb2 >= 8 and wb2 >= 8:
            # Pool 2x2 to 8-px scale and take a 7x7 median (~56x56 px support) so
            # broad or streaky Bragg spots (~35-45 px) are not self-subtracted,
            # while computing 4x fewer patch medians.
            yc = y[:, : hb2 * 2, : wb2 * 2].reshape(nb, hb2, 2, wb2, 2).mean((2, 4))
            ycp = jnp.pad(yc, ((0, 0), (3, 3), (3, 3)), mode="edge")
            patches = jax.lax.conv_general_dilated_patches(
                ycp[:, None, :, :],
                (7, 7),
                (1, 1),
                "VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            bgc = jnp.median(patches, axis=1)
            bgcp = jnp.pad(bgc, ((0, 0), (1, 1), (1, 1)), mode="edge")
            bgc = (
                bgcp[:, :-2, :-2]
                + bgcp[:, :-2, 1:-1]
                + bgcp[:, :-2, 2:]
                + bgcp[:, 1:-1, :-2]
                + bgcp[:, 1:-1, 1:-1]
                + bgcp[:, 1:-1, 2:]
                + bgcp[:, 2:, :-2]
                + bgcp[:, 2:, 1:-1]
                + bgcp[:, 2:, 2:]
            ) / 9.0
            bg = jnp.repeat(jnp.repeat(bgc, 2, axis=1), 2, axis=2)
            if bg.shape[1] < hb or bg.shape[2] < wb:
                bg = jnp.pad(
                    bg,
                    ((0, 0), (0, hb - bg.shape[1]), (0, wb - bg.shape[2])),
                    mode="edge",
                )
        else:
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


_EWALD_JIT_CACHE = None
_DOM_SCAN_JIT_FN = None


def _get_ewald_jit_fns():
    global _EWALD_JIT_CACHE
    if _EWALD_JIT_CACHE is not None:
        return _EWALD_JIT_CACHE

    import jax
    import jax.numpy as jnp

    def _score_single_run(
        Ej_r,
        R_r,
        U,
        cen,
        uh,
        vh,
        nrm,
        dw,
        dh,
        nrow,
        ncol,
        s_samp_j,
        Gc_chunks,
        inv_qn_chunks,
        lo_l,
        hi_l,
        b2f,
    ):
        kih = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        H = Ej_r.shape[1]
        Wd = Ej_r.shape[2]
        n_banks = cen.shape[0]
        ib_row = jnp.arange(n_banks, dtype=jnp.int32)[None, :]
        RU = R_r @ U
        cen_r = cen - (R_r @ s_samp_j)[None, :]
        cn_dot = (cen_r * nrm).sum(1)
        cv_dot = (cen_r * vh).sum(1)
        cu_dot = (cen_r * uh).sum(1)

        def _body(acc, pair):
            Gc_k, inv_qn_k = pair
            G = Gc_k @ RU.T
            sth = -(G[:, 2]) * inv_qn_k
            lam = 2.0 * sth * inv_qn_k
            band = (lam >= lo_l) & (lam <= hi_l) & (sth > 0) & (inv_qn_k > 0)
            kf = kih[None, :] + lam[:, None] * G
            kf = kf / jnp.linalg.norm(kf, axis=1, keepdims=True)
            dn = kf @ nrm.T
            t = cn_dot[None, :] / jnp.where(jnp.abs(dn) < 1e-9, 1e-9, dn)
            row = (t * (kf @ vh.T) - cv_dot[None, :]) / dh[None]
            col = (t * (kf @ uh.T) - cu_dot[None, :]) / dw[None]
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
            i0i = i0.astype(jnp.int32)
            j0i = j0.astype(jnp.int32)

            def gat(di, dj):
                return Ej_r[
                    ib_row,
                    jnp.clip(i0i + di, 0, H - 1),
                    jnp.clip(j0i + dj, 0, Wd - 1),
                ]

            vals = (
                (1 - fu) * (1 - fv) * gat(0, 0)
                + (1 - fu) * fv * gat(0, 1)
                + fu * (1 - fv) * gat(1, 0)
                + fu * fv * gat(1, 1)
            )
            return acc + jnp.where(ok, vals, 0.0).sum(), None

        tot, _ = jax.lax.scan(_body, jnp.float32(0.0), (Gc_chunks, inv_qn_chunks))
        return tot

    @jax.jit
    def score_batch_runs(
        Us,
        Ej,
        Rr,
        cen,
        uh,
        vh,
        nrm,
        dw,
        dh,
        nrow,
        ncol,
        s_samp_j,
        Gc_chunks,
        inv_qn_chunks,
        lo_l,
        hi_l,
        b2f,
    ):
        return jax.vmap(
            lambda U: jax.vmap(
                lambda Ej_r, R_r, s_r: _score_single_run(
                    Ej_r,
                    R_r,
                    U,
                    cen,
                    uh,
                    vh,
                    nrm,
                    dw,
                    dh,
                    nrow,
                    ncol,
                    s_r,
                    Gc_chunks,
                    inv_qn_chunks,
                    lo_l,
                    hi_l,
                    b2f,
                )
            )(Ej, Rr, s_samp_j)
        )(Us)

    @jax.jit
    def score_per_run_grid(
        Urs,
        dRs,
        Ej,
        Rr,
        cen,
        uh,
        vh,
        nrm,
        dw,
        dh,
        nrow,
        ncol,
        s_samp_j,
        Gc_chunks,
        inv_qn_chunks,
        lo_l,
        hi_l,
        b2f,
    ):
        return jax.vmap(
            lambda Ej_r, R_r, U_r, s_r: jax.vmap(
                lambda dR: _score_single_run(
                    Ej_r,
                    R_r,
                    dR @ U_r,
                    cen,
                    uh,
                    vh,
                    nrm,
                    dw,
                    dh,
                    nrow,
                    ncol,
                    s_r,
                    Gc_chunks,
                    inv_qn_chunks,
                    lo_l,
                    hi_l,
                    b2f,
                )
            )(dRs),
            in_axes=(0, 0, 0, 0),
        )(Ej, Rr, Urs, s_samp_j)

    _EWALD_JIT_CACHE = (score_batch_runs, score_per_run_grid)
    return _EWALD_JIT_CACHE


def _get_dom_scan_jit_fn():
    global _DOM_SCAN_JIT_FN
    if _DOM_SCAN_JIT_FN is not None:
        return _DOM_SCAN_JIT_FN

    import jax
    import jax.numpy as jnp

    @jax.jit
    def _dom_scan_views(X_views, d_lab_pad, q_lo_pad, q_hi_pad, w_pad):
        wz = jnp.ones(X_views.shape[1], dtype=X_views.dtype)
        return jax.vmap(
            lambda Xv: band_sum(Xv, d_lab_pad, q_lo_pad, q_hi_pad, w_pad, wz)
        )(X_views)

    _DOM_SCAN_JIT_FN = _dom_scan_views
    return _DOM_SCAN_JIT_FN


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
    chunk_size=8192,
    max_reflections=200000,
):
    import jax.numpy as jnp
    from scipy.spatial.transform import Rotation as Rot

    score_batch_runs_jit, score_per_run_grid_jit = _get_ewald_jit_fns()
    B_EW = 49

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

    # Kinematic scattering bounds: reflections with |G| < 2*sin(theta_min)/wl_max
    # cannot excite any detector bank, and pure lowest-|G| truncation in Laue
    # mode blinds all 90-deg and backscattering banks (where |G| >= sqrt(2)/wl_max).
    c_arr = np.asarray(C_corners, np.float64)
    u_arr = np.asarray(uhats, np.float64) * np.asarray(widths, np.float64)[:, None]
    v_arr = np.asarray(vhats, np.float64) * np.asarray(heights, np.float64)[:, None]
    det_pts = np.concatenate(
        [c_arr, c_arr + u_arr, c_arr + v_arr, c_arr + u_arr + v_arr], axis=0
    )
    det_r = np.linalg.norm(det_pts, axis=1)
    valid_det = det_r > 1e-6
    if np.any(valid_det):
        cos_2th_max = float(np.max(det_pts[valid_det, 2] / det_r[valid_det]))
        sin_th_min = max(
            0.0, np.sqrt(max(0.0, 0.5 * (1.0 - min(1.0, cos_2th_max)))) - 0.04
        )
    else:
        sin_th_min = 0.0

    G_c_raw = np.asarray(G_c, np.float32)
    qn_raw = np.linalg.norm(G_c_raw, axis=1)
    q_min_kin = np.float32(max(1e-7, 2.0 * sin_th_min / max(float(wl[1]), 1e-6)))
    q_max_kin = np.float32(2.0 / max(float(wl[0]), 1e-6))
    valid_idx = np.flatnonzero((qn_raw >= q_min_kin) & (qn_raw <= q_max_kin))
    if len(valid_idx) > 0:
        if max_reflections and len(valid_idx) > int(max_reflections):
            max_keep = int(max_reflections)
            qn_v = qn_raw[valid_idx]
            order_q = np.argsort(qn_v, kind="mergesort")
            # Retain the lowest-|G| core for wide angular basin capture while
            # striding uniformly across all higher-|G| shells so 90-deg and
            # backscattering Laue banks remain sharply illuminated.
            n_low = max_keep // 3
            low_part = order_q[:n_low]
            rem_part = order_q[n_low:]
            stride_sel = np.linspace(
                0, len(rem_part) - 1, max_keep - n_low, dtype=np.int64
            )
            keep_idx = valid_idx[np.concatenate([low_part, rem_part[stride_sel]])]
        else:
            keep_idx = valid_idx
        G_c_np = G_c_raw[keep_idx]
        qn_np = qn_raw[keep_idx]
    else:
        keep_idx = np.zeros((0,), dtype=np.int64)
        G_c_np = np.zeros((1, 3), np.float32)
        qn_np = np.zeros((1,), np.float32)
    inv_qn_np = np.where(qn_np > 1e-7, 1.0 / np.maximum(qn_np, 1e-7), 0.0).astype(
        np.float32
    )
    n_refl = len(G_c_np)
    chunk_sz = int(max(1024, chunk_size))
    pad = (-max(n_refl, 1)) % chunk_sz
    if pad:
        G_c_np = np.pad(G_c_np, ((0, pad), (0, 0)))
        inv_qn_np = np.pad(inv_qn_np, (0, pad))
    Gc_chunks = jnp.asarray(G_c_np.reshape(-1, chunk_sz, 3))
    inv_qn_chunks = jnp.asarray(inv_qn_np.reshape(-1, chunk_sz))

    cen = jnp.asarray(np.asarray(C_corners, np.float32))
    uh = jnp.asarray(np.asarray(uhats, np.float32))
    vh = jnp.asarray(np.asarray(vhats, np.float32))
    nrm = jnp.asarray(np.cross(uhats, vhats).astype(np.float32))
    dw = jnp.asarray(np.asarray(widths / np.maximum(M_bank, 1.0), np.float32))
    dh = jnp.asarray(np.asarray(heights / np.maximum(N_bank, 1.0), np.float32))
    nrow = jnp.asarray(np.asarray(N_bank + 1.0, np.float32))
    ncol = jnp.asarray(np.asarray(M_bank + 1.0, np.float32))
    Rr = jnp.asarray(np.array([R_runs[r_] for r_ in runs], np.float32))
    if s_samp is None:
        s_arr = np.zeros((len(runs), 3), dtype=np.float32)
    else:
        s_in = np.asarray(s_samp, dtype=np.float32)
        if s_in.ndim == 1:
            s_arr = np.broadcast_to(s_in, (len(runs), 3)).copy()
        else:
            s_arr = s_in
    s_samp_j = jnp.asarray(s_arr)
    Ej = jnp.asarray(E)
    lo_l = jnp.float32(wl[0])
    hi_l = jnp.float32(wl[1])
    b2f = jnp.float32(b2)
    ctx_args = (
        Ej,
        Rr,
        cen,
        uh,
        vh,
        nrm,
        dw,
        dh,
        nrow,
        ncol,
        s_samp_j,
        Gc_chunks,
        inv_qn_chunks,
        lo_l,
        hi_l,
        b2f,
    )

    def eval_runs(Us):
        Us_arr = np.asarray(Us, np.float32).reshape(-1, 3, 3)
        n_u = len(Us_arr)
        pad_u = (-n_u) % B_EW
        if pad_u:
            Us_pad = np.pad(Us_arr, ((0, pad_u), (0, 0), (0, 0)), mode="edge")
        else:
            Us_pad = Us_arr
        out = []
        for i in range(0, len(Us_pad), B_EW):
            out.append(
                np.asarray(
                    score_batch_runs_jit(jnp.asarray(Us_pad[i : i + B_EW]), *ctx_args)
                )
            )
        return np.concatenate(out, axis=0)[:n_u]

    def score_with_Gc(G_c_sub, Urs):
        G_sub_np = np.asarray(G_c_sub, np.float32)
        if len(G_sub_np) == 0:
            G_sub_np = np.zeros((1, 3), np.float32)
        qn_sub = np.linalg.norm(G_sub_np, axis=1)
        inv_qn_sub = np.where(
            qn_sub > 1e-7, 1.0 / np.maximum(qn_sub, 1e-7), 0.0
        ).astype(np.float32)
        pad_s = (-max(len(G_sub_np), 1)) % chunk_sz
        if pad_s:
            G_sub_np = np.pad(G_sub_np, ((0, pad_s), (0, 0)))
            inv_qn_sub = np.pad(inv_qn_sub, (0, pad_s))
        Gc_c = jnp.asarray(G_sub_np.reshape(-1, chunk_sz, 3))
        inv_c = jnp.asarray(inv_qn_sub.reshape(-1, chunk_sz))
        Urs_arr = np.asarray(Urs, np.float32).reshape(-1, 3, 3)
        n_u = len(Urs_arr)
        pad_u = (-n_u) % B_EW
        Us_pad = (
            np.pad(Urs_arr, ((0, pad_u), (0, 0), (0, 0)), mode="edge")
            if pad_u
            else Urs_arr
        )
        alt_ctx = (
            Ej,
            Rr,
            cen,
            uh,
            vh,
            nrm,
            dw,
            dh,
            nrow,
            ncol,
            s_samp_j,
            Gc_c,
            inv_c,
            lo_l,
            hi_l,
            b2f,
        )
        out = []
        for i in range(0, len(Us_pad), B_EW):
            out.append(
                np.asarray(
                    score_batch_runs_jit(jnp.asarray(Us_pad[i : i + B_EW]), *alt_ctx)
                )
            )
        mat = np.concatenate(out, axis=0)[:n_u]
        return np.diag(mat) if n_u == len(runs) else mat

    eval_runs.keep_idx = keep_idx
    eval_runs.score_with_Gc = score_with_Gc

    def _fit_consensus_goniometer(Urs, cur_s, Rr_np):
        pos = cur_s[cur_s > 0]
        if len(pos) < 2:
            return None, None
        med_s = float(np.median(pos))
        r_ref = int(np.argmax(cur_s))
        good = (cur_s >= max(0.45 * med_s, 0.35 * cur_s[r_ref])) & (cur_s > 0)
        if good.sum() < 2:
            top2 = np.argsort(cur_s)[-2:]
            good = np.zeros(len(cur_s), dtype=bool)
            good[top2] = True
        R_ref = Rr_np[r_ref]
        U_ref = Urs[r_ref]

        def res_fn(x, _U_ref=U_ref, _good=good, _R_ref=R_ref):
            D = Rot.from_rotvec(x[:3]).as_matrix()
            U0 = Rot.from_rotvec(x[3:]).as_matrix() @ _U_ref
            errs = []
            for ir in np.where(_good)[0]:
                U_p = Rr_np[ir].T @ D @ Rr_np[ir] @ _R_ref.T @ D.T @ _R_ref @ U0
                errs.append(Rot.from_matrix(U_p @ Urs[ir].T).as_rotvec())
            return np.concatenate(errs)

        fit = least_squares(
            res_fn,
            np.zeros(6),
            bounds=(-np.radians(4.5), np.radians(4.5)),
            loss="soft_l1",
            f_scale=0.004,
        )
        D = Rot.from_rotvec(fit.x[:3]).as_matrix()
        U0 = Rot.from_rotvec(fit.x[3:]).as_matrix() @ U_ref
        U_preds = np.array(
            [
                Rr_np[ir].T @ D @ Rr_np[ir] @ R_ref.T @ D.T @ R_ref @ U0
                for ir in range(len(runs))
            ]
        )
        pred_s = np.diag(eval_runs(U_preds)).astype(np.float64)
        return U_preds, pred_s

    def _grid_step_per_run(Urs, cur_s, span, n_pts):
        ax = np.radians(np.linspace(-span, span, n_pts))
        rv = np.array(np.meshgrid(ax, ax, ax, indexing="ij")).reshape(3, -1).T
        dR = Rot.from_rotvec(rv).as_matrix().astype(np.float32)
        n_g = len(dR)
        pad_g = (-n_g) % B_EW
        dR_pad = np.pad(dR, ((0, pad_g), (0, 0), (0, 0)), mode="edge") if pad_g else dR
        Urs_j = jnp.asarray(Urs, jnp.float32)
        scores_list = []
        for i in range(0, len(dR_pad), B_EW):
            scores_list.append(
                np.asarray(
                    score_per_run_grid_jit(
                        Urs_j,
                        jnp.asarray(dR_pad[i : i + B_EW]),
                        *ctx_args,
                    )
                )
            )
        scores = np.concatenate(scores_list, axis=1)[:, :n_g]
        best_g = np.argmax(scores, axis=1)
        for ir in range(len(runs)):
            if float(scores[ir, best_g[ir]]) > cur_s[ir] + 1e-6:
                cur_s[ir] = float(scores[ir, best_g[ir]])
                Urs[ir] = dR[best_g[ir]].astype(np.float64) @ Urs[ir]

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

    def polish_per_run(
        U_joint,
        spans=(0.85, 0.40, 0.16, 0.05),
        n=7,
        Urs_alt_seed=None,
    ):
        Rr_np = np.array([R_runs[r_] for r_ in runs], np.float64)
        Urs = np.repeat(np.asarray(U_joint, np.float64)[None], len(runs), axis=0)
        cur_s = np.diag(eval_runs(Urs)).astype(np.float64).copy()
        for ispan, span in enumerate(spans):
            if ispan >= 1 and len(runs) >= 3:
                U_preds, pred_s = _fit_consensus_goniometer(Urs, cur_s, Rr_np)
                if U_preds is not None:
                    for ir in range(len(runs)):
                        if pred_s[ir] > cur_s[ir]:
                            Urs[ir] = U_preds[ir]
                            cur_s[ir] = pred_s[ir]
            _grid_step_per_run(Urs, cur_s, span, n)
        if Urs_alt_seed is not None and len(Urs_alt_seed) == len(runs):
            Urs_alt = np.asarray(Urs_alt_seed, dtype=np.float64).copy()
            s_alt = np.diag(eval_runs(Urs_alt)).astype(np.float64).copy()
            for alt_span in (0.45, 0.18, 0.06):
                _grid_step_per_run(Urs_alt, s_alt, alt_span, n)
            for ir in range(len(runs)):
                if s_alt[ir] > cur_s[ir] + 1e-6:
                    Urs[ir] = Urs_alt[ir]
                    cur_s[ir] = s_alt[ir]
        # Outer-goniometer consensus check: rescue runs whose local polish improves
        # strictly above cur_s without clobbering genuine inter-batch crystal slips.
        if len(runs) >= 4:
            U_preds, pred_s = _fit_consensus_goniometer(Urs, cur_s, Rr_np)
            if U_preds is not None:
                outliers = [
                    ir
                    for ir in range(len(runs))
                    if np.degrees(Rot.from_matrix(Urs[ir] @ U_preds[ir].T).magnitude())
                    > 0.75
                ]
                if outliers:
                    Urs_try = Urs.copy()
                    s_try = cur_s.copy()
                    for ir in outliers:
                        Urs_try[ir] = U_preds[ir]
                        s_try[ir] = pred_s[ir]
                    for fine_span in (0.35, 0.14, 0.05):
                        _grid_step_per_run(Urs_try, s_try, fine_span, 5)
                    for ir in outliers:
                        if s_try[ir] > cur_s[ir] + 1e-6:
                            Urs[ir] = Urs_try[ir]
                            cur_s[ir] = s_try[ir]
        return Urs, cur_s

    return eval_runs, polish_joint, polish_per_run


def _zone_funnel_seed_per_run(
    U_joint,
    B_mat,
    run_ids,
    R_runs_dict,
    q_by_run,
    w_by_run,
    space_group=None,
    max_zones=18,
):
    """Coarse-to-fine smooth zone-band pre-alignment per run (+/-3.8 deg capture).

    Long-axis Laue Ewald correlation peaks are only ~0.15-0.25 deg wide, so
    coarse Ewald grids step over multi-degree goniometer wobble or inter-batch
    crystal slips while Gaussian zone bands provide a smooth capture funnel.
    """
    from scipy.spatial.transform import Rotation as Rot

    if not q_by_run:
        return None
    A_mat = np.linalg.inv(np.asarray(B_mat, dtype=np.float64)).T
    uvw_top = direct_lattice_uvw(B_mat, max_index=3, space_group=space_group)[
        :max_zones
    ]
    if len(uvw_top) == 0:
        return None
    X_raw = (A_mat @ np.asarray(uvw_top, dtype=np.float64).T).T
    X_cryst = (
        X_raw / np.maximum(np.linalg.norm(X_raw, axis=1, keepdims=True), 1e-12)
    ).astype(np.float32)
    U0 = np.asarray(U_joint, dtype=np.float64)
    Urs_out = np.repeat(U0[None], len(run_ids), axis=0)
    grids = []
    for span, n_g, sig in ((3.8, 15, 0.38), (0.95, 11, 0.20), (0.24, 9, 0.14)):
        ax = np.radians(np.linspace(-span, span, n_g))
        rv = np.array(np.meshgrid(ax, ax, ax, indexing="ij")).reshape(3, -1).T
        grids.append(
            (Rot.from_rotvec(rv).as_matrix().astype(np.float32), np.float32(sig))
        )

    any_seeded = False
    for ir, r in enumerate(run_ids):
        qr = q_by_run.get(int(r))
        wr = w_by_run.get(int(r))
        if qr is None or wr is None or len(qr) < 8:
            continue
        any_seeded = True
        Rr = np.asarray(R_runs_dict[r], dtype=np.float64)
        L_cur = (Rr @ U0).astype(np.float32)
        qr_f = np.asarray(qr, dtype=np.float32)
        wr_f = np.asarray(wr, dtype=np.float32)
        for dRs, sig in grids:
            best_sc = -1.0
            best_L = L_cur
            for i0 in range(0, len(dRs), 512):
                dR_b = dRs[i0 : i0 + 512]
                X_lab = np.einsum("gij,jk,zk->giz", dR_b, L_cur, X_cryst, optimize=True)
                psi = np.float32(57.2957795) * np.einsum(
                    "ni,giz->gnz", qr_f, X_lab, optimize=True
                )
                sc = np.sum(
                    wr_f[None, :] * np.max(np.exp(-0.5 * (psi / sig) ** 2), axis=2),
                    axis=1,
                )
                k = int(np.argmax(sc))
                if float(sc[k]) > best_sc:
                    best_sc = float(sc[k])
                    best_L = dR_b[k] @ L_cur
            L_cur = best_L
        u_svd, _, vt_svd = np.linalg.svd(Rr.T @ L_cur.astype(np.float64))
        if np.linalg.det(u_svd @ vt_svd) < 0:
            u_svd[:, -1] *= -1.0
        Urs_out[ir] = u_svd @ vt_svd
    return Urs_out if any_seeded else None


def _ewald_d_min(cell, d_min, space_group=None, max_refl=450000):
    a, b, c = float(cell[0]), float(cell[1]), float(cell[2])
    if len(cell) >= 6:
        ca, cb, cg = np.cos(
            np.deg2rad([float(cell[3]), float(cell[4]), float(cell[5])])
        )
        vol_fac = np.sqrt(
            max(1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg, 1e-6)
        )
    else:
        vol_fac = 1.0
    vol_est = max(a * b * c * float(vol_fac), 1.0)
    mult = 1.0
    if space_group:
        sg_lead = str(space_group).strip()[:1].upper()
        if sg_lead == "F":
            mult = 4.0
        elif sg_lead in ("I", "A", "B", "C"):
            mult = 2.0
        elif (
            sg_lead == "R"
            and len(cell) >= 6
            and abs(float(cell[3]) - 90.0) < 1e-2
            and abs(float(cell[5]) - 120.0) < 1e-2
        ):
            mult = 3.0
    d_floor = float(
        (4.0 * np.pi * vol_est / (3.0 * mult * float(max_refl))) ** (1.0 / 3.0)
    )
    return max(float(d_min), d_floor)


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
    d_min_ew = _ewald_d_min(
        cell_opt, args.d_min, space_group=args.space_group, max_refl=450000
    )
    hkl = np.stack(
        generate_reflections(
            *cell_opt,
            space_group=args.space_group,
            d_min=d_min_ew,
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

        def project_reflections(x_val, _R_r0=R_r0, _xs0=xs0):
            rv0, ds0 = x_val[:3], x_val[3:6]
            dR0 = Rot.from_rotvec(rv0).as_matrix()
            G0 = (Gc @ (dR0 @ U_opt).T) @ _R_r0.T
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
            cen_r = C_final - (_xs0 + ds0)[None, :]
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

        def get_matches(
            x_val,
            radius,
            _project_reflections=project_reflections,
            _b_r=b_r,
            _pr_r=pr_r,
            _pc_r=pc_r,
        ):
            idx_band, row, col, ok = _project_reflections(x_val)
            m_obs, m_hkl = [], []
            if len(idx_band) == 0:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            for i_obs in range(len(_b_r)):
                bi = _b_r[i_obs]
                valid_h = np.flatnonzero(ok[:, bi])
                if not len(valid_h):
                    continue
                d2 = (row[valid_h, bi] - _pr_r[i_obs]) ** 2 + (
                    col[valid_h, bi] - _pc_r[i_obs]
                ) ** 2
                if len(valid_h) >= 2:
                    k2 = np.argpartition(d2, 1)[:2]
                    if d2[k2[0]] <= d2[k2[1]]:
                        k, k_sec = int(k2[0]), int(k2[1])
                    else:
                        k, k_sec = int(k2[1]), int(k2[0])
                    if d2[k] < radius**2 and d2[k] < 0.55 * d2[k_sec]:
                        m_obs.append(i_obs)
                        m_hkl.append(idx_band[valid_h[k]])
                elif d2[0] < radius**2:
                    m_obs.append(i_obs)
                    m_hkl.append(idx_band[valid_h[0]])
            return np.array(m_obs, dtype=np.int32), np.array(m_hkl, dtype=np.int32)

        def fit_matches(
            x_init,
            m_obs,
            m_hkl,
            f_sc=2.5,
            prior_ds=0.45,
            _b_r=b_r,
            _pr_r=pr_r,
            _pc_r=pc_r,
            _w_r=w_r,
            _R_r0=R_r0,
            _xs0=xs0,
        ):
            if len(m_obs) < 8:
                return x_init
            b_m = _b_r[m_obs]
            rm = _pr_r[m_obs]
            cm = _pc_r[m_obs]
            wm = _w_r[m_obs]
            Gc_m = Gc[m_hkl]
            C_m = C_final[b_m]
            uh_m = uh_final[b_m]
            vh_m = vh_final[b_m]
            nh_m = nh_final[b_m]
            dw_m = dw[b_m]
            dh_m = dh[b_m]

            def res_fn(x_vec):
                rv, ds = x_vec[:3], x_vec[3:6]
                dR = Rot.from_rotvec(rv).as_matrix()
                G = (Gc_m @ (dR @ U_opt).T) @ _R_r0.T
                q = np.linalg.norm(G, axis=1)
                st = -(G @ kih) / np.maximum(q, 1e-12)
                lm = 2.0 * st / np.maximum(q, 1e-12)
                kf = kih[None, :] + lm[:, None] * G
                kf /= np.linalg.norm(kf, axis=1, keepdims=True)
                cen_m = C_m - (_xs0 + ds)[None, :]
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

        def eval_med_dist(
            x_val,
            m_obs,
            m_hkl,
            _R_r0=R_r0,
            _b_r=b_r,
            _xs0=xs0,
            _pr_r=pr_r,
            _pc_r=pc_r,
        ):
            if len(m_obs) == 0:
                return 999.0
            rv, ds = x_val[:3], x_val[3:6]
            dR = Rot.from_rotvec(rv).as_matrix()
            G = (Gc[m_hkl] @ (dR @ U_opt).T) @ _R_r0.T
            q = np.linalg.norm(G, axis=1)
            st = -(G @ kih) / np.maximum(q, 1e-12)
            lm = 2.0 * st / np.maximum(q, 1e-12)
            kf = kih[None, :] + lm[:, None] * G
            kf /= np.linalg.norm(kf, axis=1, keepdims=True)
            b_m = _b_r[m_obs]
            cen_m = C_final[b_m] - (_xs0 + ds)[None, :]
            denom = np.sum(kf * nh_final[b_m], axis=1)
            denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
            t = (cen_m * nh_final[b_m]).sum(1) / denom
            p = t[:, None] * kf - cen_m
            rp = np.sum(p * vh_final[b_m], axis=1) / dh[b_m]
            cp = np.sum(p * uh_final[b_m], axis=1) / dw[b_m]
            return float(np.median(np.hypot(_pr_r[m_obs] - rp, _pc_r[m_obs] - cp)))

        idx_0, row_0, col_0, ok_0 = project_reflections(np.zeros(6))
        nn_dists = []
        if len(idx_0) >= 2:
            from scipy.spatial import cKDTree

            for bi_u in np.unique(b_r):
                vh_u = np.flatnonzero(ok_0[:, int(bi_u)])
                if len(vh_u) >= 4:
                    pts_u = np.column_stack(
                        [row_0[vh_u, int(bi_u)], col_0[vh_u, int(bi_u)]]
                    )
                    d_nn, _ = cKDTree(pts_u).query(pts_u, k=2)
                    nn_dists.append(d_nn[:, 1])
        nn_med_px = float(np.median(np.concatenate(nn_dists))) if nn_dists else 24.0
        r0_icp = min(18.0, max(2.5, 0.85 * nn_med_px))
        r1_icp = min(14.0, max(2.0, 0.65 * nn_med_px))
        r2_icp = min(8.0, max(1.6, 0.48 * nn_med_px))

        mo0, mh0 = get_matches(np.zeros(6), r0_icp)
        if len(mo0) < 8:
            continue
        d_before = eval_med_dist(np.zeros(6), mo0, mh0)
        x = fit_matches(np.zeros(6), mo0, mh0, f_sc=3.5, prior_ds=0.45)
        mo1, mh1 = get_matches(x, r1_icp)
        x = fit_matches(x, mo1, mh1, f_sc=2.0, prior_ds=0.45)
        mo2, mh2 = get_matches(x, r2_icp)
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

    from subhkl.scan_refine import checkpoint, load_observations, write_result

    diagnostics = []
    _snapshot_builder = None

    def record(name, result):
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(
                f"{name} refinement returned nonfinite objective or parameters"
            )
        diagnostics.append(
            {
                "stage": name,
                "objective": float(result.fun),
                "iterations": int(result.nit),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        snap = None
        if callable(_snapshot_builder):
            try:
                snap = _snapshot_builder(name, result.x)
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
            ):
                snap = None
        checkpoint(args, name, result.x, diagnostics, snapshot=snap)

    with h5py.File(args.bootstrap, "r") as f:
        U0 = np.array(f["sample/U"][()], dtype=np.float64)
        if U0.ndim == 3:
            U0 = U0[0]
        cand_Us = [U0]
        cand_Us_warped = [U0]
        for cand_key in (
            "search/orientations",
            "blind_search/candidate_U",
            "search/proposal_orientations",
        ):
            if cand_key in f:
                for Uc in np.asarray(f[cand_key][()], dtype=np.float64)[:6]:
                    if not any(np.allclose(Uc, Uprev, atol=1e-4) for Uprev in cand_Us):
                        cand_Us.append(Uc)
                    if cand_key != "search/proposal_orientations" and not any(
                        np.allclose(Uc, Uprev, atol=1e-4) for Uprev in cand_Us_warped
                    ):
                        cand_Us_warped.append(Uc)
        cand_geoms_boot = (
            np.asarray(f["search/candidate_geometries"][()], dtype=np.float64)
            if "search/candidate_geometries" in f
            else None
        )
        cand_axials_boot = (
            np.asarray(f["search/candidate_axial_scales"][()], dtype=np.float64)
            if "search/candidate_axial_scales" in f
            else None
        )
        shared_geom = (
            np.asarray(f["search/shared_geometry"][()], dtype=np.float64)
            if "search/shared_geometry" in f
            else None
        )
        shared_axial = (
            float(np.asarray(f["search/shared_axial_scale"][()]))
            if "search/shared_axial_scale" in f
            else (
                float(shared_geom[0])
                if shared_geom is not None and len(shared_geom) >= 1
                else None
            )
        )
        U_boot_orig = U0.copy()
        shared_geom_orig = shared_geom.copy() if shared_geom is not None else None
        shared_axial_orig = shared_axial

    cell0 = [float(v) for v in args.cell]
    symmetry_kind, active_cell_slots = _cell_symmetry_spec(cell0, args.space_group)
    B0 = np.asarray(
        build_B_jax(*cell0[:3], *np.deg2rad(cell0[3:])),
        dtype=np.float64,
    )

    obs = getattr(args, "_cached_obs_kin", None)
    if obs is None:
        obs = load_observations(args, return_kinematics=True)

    from subhkl.search.blind_band import (
        count_search_weights,
        recover_unlogged_outer_goniometer,
    )
    from subhkl.search.pixel_em import PixelEMClassifier

    pixel_em = getattr(args, "_pixel_em", None)
    if pixel_em is None and not args.finder:
        pixel_em = PixelEMClassifier.from_args(args)
        args._pixel_em = pixel_em
    obs = recover_unlogged_outer_goniometer(args, obs, pixel_em)
    banks, pr, pc, dev, sig_px, runs, _angs, spot_weights, R_obs, scan_axes_obs = obs
    raw_dev_obs = dev.copy()
    raw_spot_weights = spot_weights.copy()
    _gv_obs = np.ones(len(dev), dtype=np.float64)
    if pixel_em is not None:
        dev, spot_weights, _gv_obs = pixel_em.map_to_observations(
            banks, pr, pc, runs, raw_dev_obs, raw_spot_weights
        )
    n_runs = args.n_runs
    bank_ids = sorted(int(k) for k in beamlines[args.instrument])
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
    is_single_setting = len(args.runs) <= 1 or np.allclose(
        R_obs[sel], R_obs[sel][0:1], atol=1e-6
    )
    per_run_active = "per_run" in args.modes or "per_run_trans" in args.modes

    rec_R_runs = getattr(args, "_recovered_outer_R_runs", None) or {}
    R_runs_nom = {}
    for r in range(n_runs):
        idx_r = np.flatnonzero(runs == r)
        if len(idx_r):
            R_runs_nom[r] = R_obs[idx_r[0]].copy()
        elif r in rec_R_runs:
            R_runs_nom[r] = np.asarray(rec_R_runs[r], dtype=np.float64).copy()
        else:
            R_runs_nom[r] = np.eye(3)

    # Multi-run EM candidate re-evaluation across bootstrap pool (if present)
    if (
        pixel_em is not None
        and len(cand_Us) >= 1
        and not getattr(args, "_em_bootstrapped", False)
    ):
        best_em_score = -1e30
        best_em_ev = None
        best_em_u = U0
        best_em_geom = shared_geom
        best_em_ax = shared_axial
        for ic, Uc in enumerate(cand_Us):
            geom_trials: list[tuple[np.ndarray | None, float | None]] = [
                (shared_geom, shared_axial)
            ]
            if (
                cand_geoms_boot is not None
                and ic < len(cand_geoms_boot)
                and (
                    shared_geom is None
                    or not np.allclose(cand_geoms_boot[ic], shared_geom, atol=1e-5)
                )
            ):
                ax_ic = (
                    float(cand_axials_boot[ic])
                    if cand_axials_boot is not None and ic < len(cand_axials_boot)
                    else float(cand_geoms_boot[ic, 0])
                )
                geom_trials.append((cand_geoms_boot[ic], ax_ic))
            if shared_geom is not None and abs(float(shared_geom[0])) > 1e-4:
                geom_trials.append((np.zeros(7, dtype=np.float64), 0.0))

            for g_try, ax_try in geom_trials:
                r_try = (
                    float(g_try[0]) if g_try is not None and len(g_try) >= 1 else 0.0
                )
                a_try = float(ax_try) if ax_try is not None else r_try
                rv_try = (
                    g_try[1:4]
                    if g_try is not None and len(g_try) == 7
                    else np.zeros(3, dtype=np.float64)
                )
                tv_try = (
                    g_try[4:7]
                    if g_try is not None and len(g_try) == 7
                    else np.zeros(3, dtype=np.float64)
                )
                ev_c = pixel_em.evaluate_candidate(
                    Uc,
                    B0,
                    R_runs=R_runs_nom,
                    radial_scale=r_try,
                    axial_scale=a_try,
                    global_rot_vec=rv_try,
                    global_trans_m=tv_try,
                    match_sigma_px=5.0,
                )
                if ev_c["em_score"] > best_em_score:
                    best_em_score = float(ev_c["em_score"])
                    best_em_ev = ev_c
                    best_em_u = Uc
                    best_em_geom = g_try
                    best_em_ax = ax_try

        if best_em_ev is not None:
            U0 = best_em_u
            shared_geom = best_em_geom
            shared_axial = best_em_ax
            pixel_em.apply_responsibilities(
                best_em_ev["gamma_valid"],
                best_em_ev["gamma_dark"],
                best_em_ev["gamma_bright"],
            )
            dev, spot_weights, _gv_obs = pixel_em.map_to_observations(
                banks, pr, pc, runs, raw_dev_obs, raw_spot_weights
            )

    if args.finder:
        sel &= dev >= args.min_deviance
    sel_full = sel.copy()
    keep_mask_loc = None
    if not args.finder:
        idx_full = np.flatnonzero(sel_full)
        if len(idx_full) > 4096:
            dev_rank = np.maximum(raw_dev_obs[idx_full], dev[idx_full])
            w_rank = count_search_weights(dev_rank, max_quantile=0.999, max_factor=3.0)
            w_rmax = float(w_rank.max()) if len(w_rank) else 0.0
            sig_loc = np.flatnonzero(w_rank > max(1e-4 * w_rmax, 0.0))
            top_cap = 32_768
            bg_cap = 2_048
            if len(sig_loc) > top_cap:
                sig_loc = sig_loc[np.argpartition(w_rank[sig_loc], -top_cap)[-top_cap:]]
            keep_mask_loc = np.zeros(len(idx_full), dtype=bool)
            keep_mask_loc[sig_loc] = True
            rem_loc = np.flatnonzero(~keep_mask_loc)
            if len(rem_loc) > bg_cap:
                stride_idx = np.linspace(0, len(rem_loc) - 1, bg_cap, dtype=np.int64)
                keep_mask_loc[rem_loc[stride_idx]] = True
            else:
                keep_mask_loc[rem_loc] = True
            if not np.all(keep_mask_loc):
                sel = np.zeros(len(banks), dtype=bool)
                sel[idx_full[keep_mask_loc]] = True
            else:
                keep_mask_loc = None

    def _compute_band_weights(dev_arr, sw_arr, gv_arr=None):
        if args.finder:
            w_s = sw_arr[sel]
        else:
            w_full = count_search_weights(
                dev_arr[sel_full], max_quantile=0.999, max_factor=3.0
            )
            if gv_arr is not None:
                w_full = w_full * gv_arr[sel_full]
            w_s = w_full[keep_mask_loc] if keep_mask_loc is not None else w_full
            if not np.any(w_s > 0):
                w_s = sw_arr[sel]
        wj = jnp.asarray(w_s / np.maximum(np.abs(w_s).sum(), 1e-12), dtype=jnp.float64)
        wr = jnp.where(
            jnp.isin(
                jnp.asarray(runs[sel], dtype=jnp.int32), jnp.asarray(args.stable_runs)
            ),
            wj,
            0.0,
        )
        wr = wr / jnp.maximum(jnp.abs(wr).sum(), 1e-12)
        return wj, wr

    def _build_zone_funnel_obs(C_mid, uh, vh, W_cur, H_cur, s_vec=None):
        clean_mask = sel_full & (_gv_obs > 0.5) & (dev > 20.0)
        if int(clean_mask.sum()) < 24:
            clean_mask = sel_full & (
                dev > max(float(getattr(args, "min_deviance", 6.0)), 6.0)
            )
        idx_cl = np.flatnonzero(clean_mask)
        if len(idx_cl) == 0:
            return {}, {}
        bi_cl = b_idx_all[idx_cl]
        u_cl = (pc[idx_cl] / M_bank[bi_cl] - 0.5) * W_cur[bi_cl]
        v_cl = (pr[idx_cl] / N_bank[bi_cl] - 0.5) * H_cur[bi_cl]
        pos_cl = C_mid[bi_cl] + u_cl[:, None] * uh[bi_cl] + v_cl[:, None] * vh[bi_cl]
        if s_vec is not None:
            pos_cl = pos_cl - np.asarray(s_vec, dtype=np.float64)[None, :]
        kf_cl = pos_cl / np.maximum(
            np.linalg.norm(pos_cl, axis=1, keepdims=True), 1e-12
        )
        q_cl = kf_cl - np.array([0.0, 0.0, 1.0], dtype=np.float64)[None, :]
        q_cl = q_cl / np.maximum(np.linalg.norm(q_cl, axis=1, keepdims=True), 1e-12)
        w_cl = np.clip(dev[idx_cl], 0.0, 400.0) * _gv_obs[idx_cl]
        r_cl = runs[idx_cl]
        q_dict, w_dict = {}, {}
        for r_id in args.runs:
            mr = r_cl == r_id
            if int(mr.sum()) >= 8:
                sub_idx = np.flatnonzero(mr)
                if len(sub_idx) > 400:
                    sub_idx = sub_idx[np.argpartition(w_cl[sub_idx], -400)[-400:]]
                q_dict[int(r_id)] = q_cl[sub_idx]
                w_dict[int(r_id)] = w_cl[sub_idx]
        return q_dict, w_dict

    ewald_ctx = None
    raw_exc_maps = None
    if not args.pixel_polish and args.maxiter > 2:
        raw_exc_maps, b2_ew, has_ew_signal = _build_ewald_excess_maps(args)
        exc_maps = (
            pixel_em.filter_ewald_excess_maps(raw_exc_maps, b2_ew)
            if (pixel_em is not None and has_ew_signal)
            else raw_exc_maps
        )
        if has_ew_signal:
            d_min_ew = _ewald_d_min(
                cell0, args.d_min, space_group=args.space_group, max_refl=600000
            )
            hkl_ew = np.stack(
                generate_reflections(
                    *cell0,
                    space_group=args.space_group,
                    d_min=d_min_ew,
                ),
                axis=1,
            ).astype(np.float64)
            _eval_nom, polish_joint_nom, polish_per_run_nom = _make_ewald_polisher(
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
            used_pre_polish = False
            best_u_nom, best_s_nom = U0, -1e30
            for Uc in cand_Us[:8]:
                Up, sp = polish_joint_nom(Uc, spans=(1.0, 0.35), n=7)
                if sp > best_s_nom:
                    best_s_nom, best_u_nom = sp, Up
            U_nom_joint, ew_nom_joint = polish_joint_nom(
                best_u_nom, spans=(0.12, 0.04), n=7
            )
            if per_run_active and not is_single_setting:
                q_zf_nom, w_zf_nom = _build_zone_funnel_obs(
                    C_bank, uhat_bank, vhat_bank, W_bank, H_bank
                )
                Urs_zf_nom = _zone_funnel_seed_per_run(
                    U_nom_joint,
                    B0,
                    args.runs,
                    R_runs_nom,
                    q_zf_nom,
                    w_zf_nom,
                    space_group=args.space_group,
                )
                Urs_nom, s_nom_r = polish_per_run_nom(
                    U_nom_joint, Urs_alt_seed=Urs_zf_nom
                )
                ew_nom = float(s_nom_r.sum())
                if args.anchor_run in args.runs:
                    U_nom_joint = Urs_nom[list(args.runs).index(args.anchor_run)].copy()
                R_runs_pre = dict(R_runs_nom)
                for ir, r in enumerate(args.runs):
                    R_runs_pre[r] = R_runs_nom[r] @ Urs_nom[ir] @ U_nom_joint.T
            else:
                Urs_nom = np.repeat(U_nom_joint[None], len(args.runs), axis=0)
                ew_nom = float(ew_nom_joint)
                R_runs_pre = dict(R_runs_nom)

            has_nontrivial_shared_geom = (
                shared_geom_orig is not None
                and len(shared_geom_orig) == 7
                and (
                    abs(float(shared_geom_orig[0])) >= 0.01
                    or np.linalg.norm(shared_geom_orig[1:4]) >= np.deg2rad(0.2)
                    or np.linalg.norm(shared_geom_orig[4:7]) >= 0.002
                )
            )
            ew_shared = -1.0
            U_sh_seed = U_boot_orig.copy()
            if has_nontrivial_shared_geom:
                from scipy.spatial.transform import Rotation as _RotSh

                sr_sh = float(shared_geom_orig[0])
                sa_sh = (
                    float(shared_axial_orig) if shared_axial_orig is not None else sr_sh
                )
                Rg_sh = _RotSh.from_rotvec(shared_geom_orig[1:4]).as_matrix()
                T_sh = np.asarray(shared_geom_orig[4:7], dtype=np.float64)
                C_sh_sc = np.stack(
                    [
                        C_bank[:, 0] * (1.0 + sr_sh),
                        C_bank[:, 1] * (1.0 + sa_sh),
                        C_bank[:, 2] * (1.0 + sr_sh),
                    ],
                    axis=-1,
                )
                uh_sh = uhat_bank @ Rg_sh.T
                vh_sh = vhat_bank @ Rg_sh.T
                C_sh_mid = C_sh_sc @ Rg_sh.T + T_sh[None, :]
                C_sh_corner = (
                    C_sh_mid
                    - 0.5 * W_bank[:, None] * uh_sh
                    - 0.5 * H_bank[:, None] * vh_sh
                )
                _, polish_joint_sh, polish_per_run_sh = _make_ewald_polisher(
                    hkl_ew @ B0.T,
                    args.runs,
                    R_runs_nom,
                    bank_ids,
                    C_sh_corner,
                    uh_sh,
                    vh_sh,
                    W_bank,
                    H_bank,
                    M_bank,
                    N_bank,
                    exc_maps,
                    args.assign_wavelength,
                    b2_ew,
                )
                best_u_sh, best_s_sh = U_boot_orig, -1e30
                for Uc_sh in cand_Us_warped[:3]:
                    Up_sh, sp_sh = polish_joint_sh(Uc_sh, spans=(0.6, 0.2), n=5)
                    if sp_sh > best_s_sh:
                        best_s_sh, best_u_sh = sp_sh, Up_sh
                U_sh_joint, ew_sh_joint = polish_joint_sh(
                    best_u_sh, spans=(0.12, 0.04), n=5
                )
                U_sh_seed = U_sh_joint
                if per_run_active and not is_single_setting:
                    q_zf_sh, w_zf_sh = _build_zone_funnel_obs(
                        C_sh_mid, uh_sh, vh_sh, W_bank, H_bank
                    )
                    Urs_zf_sh = _zone_funnel_seed_per_run(
                        U_sh_joint,
                        B0,
                        args.runs,
                        R_runs_nom,
                        q_zf_sh,
                        w_zf_sh,
                        space_group=args.space_group,
                    )
                    _, s_sh_r = polish_per_run_sh(U_sh_joint, Urs_alt_seed=Urs_zf_sh)
                    ew_shared = float(s_sh_r.sum())
                else:
                    ew_shared = float(ew_sh_joint)

            used_pre_polish = (not has_nontrivial_shared_geom) or (
                ew_nom >= 0.90 * ew_shared
            )
            if used_pre_polish:
                U0 = U_nom_joint
                if per_run_active and not is_single_setting:
                    for r in args.runs:
                        R_obs[runs == r] = R_runs_pre[r]
                if pixel_em is not None:
                    ev_pre = pixel_em.evaluate_candidate(
                        U0,
                        B0,
                        R_runs=R_runs_pre,
                        radial_scale=0.0,
                        axial_scale=0.0,
                        match_sigma_px=6.5,
                    )
                    pixel_em.apply_responsibilities(
                        ev_pre["gamma_valid"],
                        ev_pre["gamma_dark"],
                        ev_pre["gamma_bright"],
                    )
                    dev, spot_weights, _gv_obs = pixel_em.map_to_observations(
                        banks, pr, pc, runs, raw_dev_obs, raw_spot_weights
                    )
            else:
                U0 = U_sh_seed
                shared_geom = shared_geom_orig
                shared_axial = shared_axial_orig

            ewald_ctx = {
                "raw_exc_maps": raw_exc_maps,
                "exc_maps": exc_maps,
                "b2": b2_ew,
                "hkl_ew": hkl_ew,
                "U_nom_joint": U_nom_joint,
                "Urs_nom": Urs_nom,
                "ew_nom": ew_nom,
                "ew_shared": ew_shared,
                "U_sh_seed": U_sh_seed,
                "R_runs_pre": R_runs_pre,
                "R_runs_band_base": R_runs_pre if used_pre_polish else R_runs_nom,
            }

    b_idx = jnp.asarray(b_idx_all[sel], dtype=jnp.int32)
    u_off = jnp.asarray(u_off_all[sel], dtype=jnp.float64)
    v_off = jnp.asarray(v_off_all[sel], dtype=jnp.float64)
    R0_j = jnp.asarray(R_obs[sel], dtype=jnp.float64)
    e_scan_j = jnp.asarray(scan_axes_obs[sel], dtype=jnp.float64)
    r_idx = jnp.asarray(runs[sel], dtype=jnp.int32)
    w_j, w_r29 = _compute_band_weights(dev, spot_weights)

    C_j = jnp.asarray(C_bank[b_idx_all[sel]], dtype=jnp.float64)
    uh_j = jnp.asarray(uhat_bank[b_idx_all[sel]], dtype=jnp.float64)
    vh_j = jnp.asarray(vhat_bank[b_idx_all[sel]], dtype=jnp.float64)
    Uj = jnp.asarray(U0, dtype=jnp.float64)
    kij = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)

    uvw = direct_lattice_uvw(B0, max_index=6, space_group=args.space_group)[
        : args.n_zones
    ]
    uvw_j = jnp.asarray(uvw, dtype=jnp.float64)
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
        s_ax = p[4] if "axial_stretch" in args.modes else s_rad
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
        (
            rv_s,
            sr_s,
            sa_s,
            sar_s,
            rg_s,
            T_s,
            _ss_s,
            da_s,
            db_s,
            dbe_s,
            dom_s,
            du_s,
            dv_s,
        ) = unpack(jnp.asarray(p_vec, dtype=jnp.float64))
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
        C_fin_s = C_mid_s - 0.5 * W_s[:, None] * uh_s - 0.5 * H_s[:, None] * vh_s
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
        import contextlib

        with contextlib.suppress(
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
        ):
            p_pre = np.zeros(18 + n_runs + 2 * n_banks, dtype=np.float64)
            checkpoint(
                args,
                "ewald-pre",
                p_pre,
                [
                    {
                        "stage": "ewald-pre",
                        "objective": float(-ewald_ctx.get("ew_nom", 0.0)),
                        "iterations": 1,
                        "success": True,
                        "message": "EWALD_PRE_POLISH",
                    }
                ],
                snapshot=_build_stage_snapshot("ewald-pre", p_pre),
            )

    def loss_band(p, w_spots, n_z=400, reg_panel=200.0, inv_len_weight=False):
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
        wz = 1.0 / jnp.linalg.norm(X0, axis=1) if inv_len_weight else jnp.ones(len(X0))
        score = band_sum(X0 @ Rm.T, d, q_lo, q_hi, w_spots, wz)
        reg_geom = 5.0
        reg_cell = 1000.0 if is_single_setting else 2.0
        reg_aniso = 15.0 * (s_rad - s_ax) ** 2 if "axial_stretch" in args.modes else 0.0
        reg = (
            reg_panel * (jnp.sum(du**2) + jnp.sum(dv**2))
            + reg_cell * (da**2 + db**2 + dbeta**2)
            + 500.0 * s_area**2
            + 5.0 * jnp.sum(dom**2)
            + reg_aniso
            + reg_geom
            * (
                jnp.sum(rg**2)
                + jnp.sum((T / 0.15) ** 2)
                + jnp.sum((s_samp / 0.015) ** 2)
            )
        )
        return -score + reg

    cell_bounds_0 = (
        [(-0.035, 0.035), (-0.035, 0.035), (-0.035, 0.035)]
        if (not is_single_setting and len(args.stable_runs) >= 2)
        else [(0.0, 0.0)] * 3
    )
    bounds_0 = (
        [(-0.2, 0.2)] * 3
        + [(-0.15, 0.15), (-0.15, 0.15), (0.0, 0.0)]
        + [(-0.08, 0.08)] * 3
        + [(-0.03, 0.03)] * 3
        + [(0.0, 0.0)] * 3
        + cell_bounds_0
        + [(0.0, 0.0)] * n_runs
        + [(0.0, 0.0)] * (2 * n_banks)
    )
    bounds_120 = (
        [(-0.2, 0.2)] * 3
        + [(-0.15, 0.15), (-0.15, 0.15), (0.0, 0.0)]
        + [(-0.08, 0.08)] * 3
        + [(-0.03, 0.03)] * 3
        + [(-0.015, 0.015)] * 3
        + [(-0.035, 0.035), (-0.035, 0.035), (-0.035, 0.035)]
        + [(-np.deg2rad(6.0), np.deg2rad(6.0))] * n_runs
        + [(0.0, 0.0)] * (2 * n_banks)
    )
    bounds_global = (
        [(-0.2, 0.2)] * 3
        + [(-0.15, 0.15), (-0.15, 0.15), (-0.01, 0.01)]
        + [(-0.08, 0.08)] * 3
        + [(-0.03, 0.03)] * 3
        + [(-0.015, 0.015)] * 3
        + [(-0.035, 0.035), (-0.035, 0.035), (-0.035, 0.035)]
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
        jax.value_and_grad(lambda p, ws: loss_band(p, ws, 60, 0.0, False))
    )
    vg_band_120 = jax.jit(
        jax.value_and_grad(lambda p, ws: loss_band(p, ws, 120, 0.0, False))
    )
    vg_band_400 = jax.jit(
        jax.value_and_grad(lambda p, ws: loss_band(p, ws, args.n_zones, 0.0, False))
    )
    vg_band_p = jax.jit(
        jax.value_and_grad(lambda p, ws: loss_band(p, ws, args.n_zones, 200.0, False))
    )

    units = parameter_units(n_runs, n_banks)

    def _minimize_band(vg_fn, p_init, stage_bounds, w_spots_arg, maxcor=20):
        def _obj(z):
            value, gradient = vg_fn(jnp.asarray(z * units), w_spots_arg)
            return float(value), np.asarray(gradient, np.float64) * units

        res_s = minimize(
            _obj,
            np.asarray(p_init, np.float64) / units,
            method="L-BFGS-B",
            jac=True,
            bounds=np.asarray(stage_bounds, np.float64) / units[:, None],
            options={
                "maxiter": args.maxiter,
                "ftol": 1e-11,
                "gtol": 1e-8,
                "maxcor": maxcor,
            },
        )
        res_s.x = np.asarray(res_s.x, np.float64) * units
        return res_s

    def _run_em_e_step_from_p(p_vec, match_sigma_px=5.5):
        nonlocal dev, spot_weights, w_j, w_r29
        if pixel_em is None:
            return None
        (
            rv_e,
            sr_e,
            sa_e,
            sar_e,
            rg_e,
            T_e,
            ss_e,
            da_e,
            db_e,
            dbe_e,
            dom_e,
            du_e,
            dv_e,
        ) = unpack(jnp.asarray(p_vec, dtype=jnp.float64))
        Rg_e = np.asarray(rodrigues_jax(rg_e))
        U_e = np.asarray(rodrigues_jax(rv_e) @ Uj)
        B_e = np.asarray(
            build_B_jax(*_apply_cell_jax(cell0, symmetry_kind, da_e, db_e, dbe_e))
        )
        C_sc_e = np.stack(
            [
                C_bank[:, 0] * (1.0 + float(sr_e)),
                C_bank[:, 1] * (1.0 + float(sa_e)),
                C_bank[:, 2] * (1.0 + float(sr_e)),
            ],
            axis=-1,
        )
        W_e = W_bank * (1.0 + float(sar_e))
        H_e = H_bank * (1.0 + float(sar_e))
        uh_e = uhat_bank @ Rg_e.T
        vh_e = vhat_bank @ Rg_e.T
        C_mid_e = (
            C_sc_e
            + np.asarray(du_e)[:, None] * uhat_bank
            + np.asarray(dv_e)[:, None] * vhat_bank
        ) @ Rg_e.T + np.asarray(T_e)[None, :]
        C_fin_e = C_mid_e - 0.5 * W_e[:, None] * uh_e - 0.5 * H_e[:, None] * vh_e
        panel_cal_e = {
            bank_ids[i]: (C_fin_e[i], uh_e[i], vh_e[i], float(W_e[i]), float(H_e[i]))
            for i in range(n_banks)
        }
        R_base_e = (
            ewald_ctx["R_runs_band_base"]
            if (ewald_ctx is not None and "R_runs_band_base" in ewald_ctx)
            else R_runs_nom
        )
        R_runs_e = {}
        for r in args.runs:
            idx_r = np.flatnonzero(runs == r)
            if len(idx_r):
                dR_r = np.asarray(
                    rodrigues_jax(
                        float(dom_e[r]) * jnp.asarray(scan_axes_obs[idx_r[0]])
                    )
                )
                R_runs_e[r] = dR_r @ R_base_e[r]
            else:
                R_runs_e[r] = R_base_e[r]
        stats_e = pixel_em.update_e_step(
            U_e,
            B_e,
            R_runs=R_runs_e,
            panel_cal=panel_cal_e,
            sample_shift_m=np.asarray(ss_e),
            match_sigma_px=match_sigma_px,
        )
        dev, spot_weights, gv_e = pixel_em.map_to_observations(
            banks, pr, pc, runs, raw_dev_obs, raw_spot_weights
        )
        w_j, w_r29 = _compute_band_weights(dev, spot_weights, gv_e)
        return stats_e

    used_nom_pre = ewald_ctx is not None and (
        ewald_ctx.get("R_runs_band_base") is ewald_ctx.get("R_runs_pre")
    )
    has_calibrated_shared_geom = (
        not used_nom_pre
        and shared_geom is not None
        and len(shared_geom) == 7
        and (
            abs(float(shared_geom[0])) >= 0.015
            or np.linalg.norm(shared_geom[1:4]) >= np.deg2rad(0.2)
            or np.linalg.norm(shared_geom[4:7]) >= 0.002
        )
    )
    if has_calibrated_shared_geom:
        ob_tight = np.deg2rad(2.5)
        for b_stage in (bounds_0, bounds_120, bounds_global, bounds_panel):
            for j_rv in range(3):
                b_stage[j_rv] = (-ob_tight, ob_tight)

    # Stage 0: coarse global geometry on the caller's stable-run subset.
    # Always test pure nominal geometry (0.0) without overwriting it with
    # shared_geom, and compete it directly against a shared_geom trial when
    # bootstrap detector geometry was estimated.
    best_p, best_v, best_result = None, 1e9, None
    stage0_trials: list[tuple[float, bool]] = []
    if has_calibrated_shared_geom:
        stage0_trials.append((float(shared_geom[0]), True))
        stage0_trials.append((0.0, False))
    else:
        base_s = (
            (-0.05, 0.0, 0.055)
            if ("cylindrical" in args.modes or "axial_stretch" in args.modes)
            else (0.0,)
        )
        for s_v in base_s:
            stage0_trials.append((float(s_v), False))
        if (
            not used_nom_pre
            and shared_geom_orig is not None
            and len(shared_geom_orig) == 7
            and (
                abs(float(shared_geom_orig[0])) > 1e-4
                or np.linalg.norm(shared_geom_orig[1:4]) > 1e-4
                or np.linalg.norm(shared_geom_orig[4:7]) > 1e-4
            )
        ):
            stage0_trials.append((float(shared_geom_orig[0]), True))
    for s_init, use_sh_geom in stage0_trials:
        p0 = np.zeros(18 + n_runs + 2 * n_banks)
        p0[3] = s_init
        p0[4] = s_init
        g_seed = shared_geom_orig if shared_geom_orig is not None else shared_geom
        ax_seed = shared_axial_orig if shared_geom_orig is not None else shared_axial
        if use_sh_geom and g_seed is not None and len(g_seed) == 7:
            from scipy.spatial.transform import Rotation as _RotS0

            U_sh_ref = (
                ewald_ctx["U_sh_seed"]
                if (ewald_ctx is not None and "U_sh_seed" in ewald_ctx)
                else U_boot_orig
            )
            p0[0:3] = _RotS0.from_matrix(U_sh_ref @ np.asarray(Uj).T).as_rotvec()
            p0[3] = float(g_seed[0])
            p0[4] = float(ax_seed) if ax_seed is not None else float(g_seed[0])
            if "global_rot" in args.modes:
                p0[6:9] = g_seed[1:4]
            if "global_trans" in args.modes:
                p0[9:12] = g_seed[4:7]
        p0 = np.clip(p0, *np.asarray(bounds_0).T)
        res = _minimize_band(vg_band_60, p0, bounds_0, w_r29)
        if res.fun < best_v:
            best_v, best_p, best_result = res.fun, res.x.copy(), res
    record("global", best_result)
    print(
        f"[Stage 0: 60-zone stable runs {args.stable_runs}] band_score = {-best_v:.4f}",
        flush=True,
    )
    _run_em_e_step_from_p(best_p, match_sigma_px=6.0)

    # Initialize per-run omega drift via 1D scan on 120-zone band score
    dom_grid = np.deg2rad(np.linspace(-5.0, 5.0, 201))
    dom_runs = [
        r
        for r in args.runs
        if not (
            (
                ewald_ctx is not None
                and ewald_ctx.get("R_runs_band_base") is ewald_ctx.get("R_runs_pre")
            )
            or r == args.anchor_run
            or not per_run_active
        )
        and np.any(runs[sel] == r)
    ]
    if dom_runs:
        from scipy.spatial.transform import Rotation as _RotDom

        dom_scan_fn = _get_dom_scan_jit_fn()
        (
            rv_d0,
            sr_d0,
            sa_d0,
            sar_d0,
            rg_d0,
            T_d0,
            ss_d0,
            _da_d0,
            _db_d0,
            _dbe_d0,
            _dom_d0,
            _du_d0,
            _dv_d0,
        ) = unpack(jnp.asarray(best_p))
        Rg_d0 = np.asarray(rodrigues_jax(rg_d0), dtype=np.float64)
        Rm_d0 = np.asarray(rodrigues_jax(rv_d0) @ Uj, dtype=np.float64)
        A_d0 = np.linalg.inv(np.asarray(B0, dtype=np.float64)).T
        X_cryst_120 = (np.asarray(uvw_j[:120], dtype=np.float64) @ A_d0.T) @ Rm_d0.T
        T_d0_np = np.asarray(T_d0, dtype=np.float64)
        ss_d0_np = np.asarray(ss_d0, dtype=np.float64)

        C_np_sel = np.asarray(C_j, dtype=np.float64)
        uh_np_sel = np.asarray(uh_j, dtype=np.float64)
        vh_np_sel = np.asarray(vh_j, dtype=np.float64)
        uo_np_sel = np.asarray(u_off, dtype=np.float64)
        vo_np_sel = np.asarray(v_off, dtype=np.float64)
        R0_np_sel = np.asarray(R0_j, dtype=np.float64)
        es_np_sel = np.asarray(e_scan_j, dtype=np.float64)
        w_np_sel = np.asarray(w_j, dtype=np.float64)

        takes_by_r = {r: np.flatnonzero(runs[sel] == r) for r in dom_runs}
        n_pad_dom = max(
            2048, ((max(len(t) for t in takes_by_r.values()) + 2047) // 2048) * 2048
        )
        sr_f, sa_f, sar_f = float(sr_d0), float(sa_d0), float(sar_d0)
        wl_min_f, wl_max_f, inv_dmin_f = (
            float(args.wl_min),
            float(args.wl_max),
            1.0 / float(args.d_min),
        )
        reg_dom_grid = 5.0 * (dom_grid**2)

        for r in dom_runs:
            take = takes_by_r[r]
            n_t = len(take)
            Cr_t = C_np_sel[take]
            C_sc_t = np.stack(
                [
                    Cr_t[:, 0] * (1.0 + sr_f),
                    Cr_t[:, 1] * (1.0 + sa_f),
                    Cr_t[:, 2] * (1.0 + sr_f),
                ],
                axis=-1,
            )
            O_sc_t = (uo_np_sel[take] * (1.0 + sar_f))[:, None] * uh_np_sel[take] + (
                vo_np_sel[take] * (1.0 + sar_f)
            )[:, None] * vh_np_sel[take]
            xyz_det_t = (C_sc_t + O_sc_t) @ Rg_d0.T + T_d0_np[None, :]
            R0_t = R0_np_sel[take]
            es_t = es_np_sel[take]
            xyz_t = xyz_det_t - np.einsum("nij,j->ni", R0_t, ss_d0_np)
            kf_t = xyz_t / np.linalg.norm(xyz_t, axis=1, keepdims=True)
            g_t = kf_t - np.array([[0.0, 0.0, 1.0]])
            d_lab_t = g_t / np.linalg.norm(g_t, axis=1, keepdims=True)
            s_t = -d_lab_t[:, 2]
            q_lo_t = 2.0 * s_t / wl_max_f
            q_hi_t = np.maximum(np.minimum(2.0 * s_t / wl_min_f, inv_dmin_f), q_lo_t)
            w_t = w_np_sel[take]
            w_t = w_t / max(float(np.abs(w_t).sum()), 1e-12)

            dR_grid = _RotDom.from_rotvec(
                dom_grid[:, None] * es_t[0][None, :]
            ).as_matrix()
            X_views = np.einsum("gab,bc,zc->gza", dR_grid, R0_t[0], X_cryst_120).astype(
                np.float32
            )

            d_pad = np.zeros((n_pad_dom, 3), dtype=np.float32)
            d_pad[:n_t] = d_lab_t
            d_pad[n_t:, 2] = 1.0
            qlo_pad = np.zeros(n_pad_dom, dtype=np.float32)
            qlo_pad[:n_t] = q_lo_t
            qhi_pad = np.zeros(n_pad_dom, dtype=np.float32)
            qhi_pad[:n_t] = q_hi_t
            w_pad = np.zeros(n_pad_dom, dtype=np.float32)
            w_pad[:n_t] = w_t

            raw_scores = np.asarray(
                dom_scan_fn(
                    jnp.asarray(X_views),
                    jnp.asarray(d_pad),
                    jnp.asarray(qlo_pad),
                    jnp.asarray(qhi_pad),
                    jnp.asarray(w_pad),
                ),
                dtype=np.float64,
            )
            scores_r = raw_scores - reg_dom_grid
            best_p[18 + r] = float(dom_grid[int(np.argmax(scores_r))])
    best_p[18 : 18 + n_runs] -= best_p[18 + args.anchor_run]

    # Stage 1a: 120-zone band across all selected runs, including scan drift.
    res_1a = _minimize_band(vg_band_120, best_p, bounds_120, w_j)
    best_p = res_1a.x.copy()
    record("120-zones", res_1a)
    print(f"[Stage 1a: 120-Zone Band (All Runs)] band_score = {-res_1a.fun:.4f}")
    _run_em_e_step_from_p(best_p, match_sigma_px=5.0)

    # Stage 1b: Fine 400-zone band + monoclinic cell ratios
    res_1b = _minimize_band(vg_band_400, best_p, bounds_global, w_j)
    best_p = res_1b.x.copy()
    record("cell", res_1b)
    print(
        f"[Stage 1b: {args.n_zones}-Zone Band+Axial+Cell] band_score = {-res_1b.fun:.4f}"
    )

    # Stage 2: Regularized per-bank (du, dv) shifts on 400-zone band objective
    if "panels" in args.modes and not is_single_setting:
        _run_em_e_step_from_p(best_p, match_sigma_px=4.5)
        res_p = _minimize_band(vg_band_p, best_p, bounds_panel, w_j, maxcor=30)
        p_cur = res_p.x.copy()
        record("panels", res_p)
        print(f"[Stage 2: +Regularized Per-Bank Shifts] loss = {res_p.fun:.4f}")
    else:
        p_cur = best_p.copy()
    _run_em_e_step_from_p(p_cur, match_sigma_px=4.5)

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

        _ev_nf = None
        pol_j_nf = None
        pol_pr_nf = None
        if pixel_em is not None and ewald_ctx.get("raw_exc_maps") is not None:
            ewald_ctx["exc_maps"] = pixel_em.filter_ewald_excess_maps(
                ewald_ctx["raw_exc_maps"], ewald_ctx["b2"]
            )
            _ev_nf, pol_j_nf, pol_pr_nf = _make_ewald_polisher(
                ewald_ctx["hkl_ew"] @ B0.T,
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
                ewald_ctx["exc_maps"],
                args.assign_wavelength,
                ewald_ctx["b2"],
            )
            ewald_ctx["ew_nom"] = float(np.diag(_ev_nf(ewald_ctx["Urs_nom"])).sum())

        R_runs_band = dict(ewald_ctx["R_runs_band_base"])
        for r in args.runs:
            idx_r = np.flatnonzero(runs == r)
            if len(idx_r):
                dR_r = np.asarray(
                    rodrigues_jax(float(dom[r]) * jnp.asarray(scan_axes_obs[idx_r[0]]))
                )
                R_runs_band[r] = dR_r @ R_runs_band[r]
        _eval_cal, polish_joint_cal, polish_per_run_cal = _make_ewald_polisher(
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
        if abs(float(s_rad)) < 0.03:
            U_cal_2, s_cal_2 = polish_joint_cal(ewald_ctx["U_nom_joint"])
            if s_cal_2 > s_cal_1:
                U_cal_joint, ew_cal_joint = U_cal_2, s_cal_2
            else:
                U_cal_joint, ew_cal_joint = U_cal_1, s_cal_1
        else:
            U_cal_joint, ew_cal_joint = U_cal_1, s_cal_1
        if per_run_active and not is_single_setting:
            q_zf_cal, w_zf_cal = _build_zone_funnel_obs(
                C_midpoint_final,
                uh_final,
                vh_final,
                W_final,
                H_final,
                s_vec=np.asarray(s_samp),
            )
            Urs_zf_cal = _zone_funnel_seed_per_run(
                U_cal_joint,
                B_opt,
                args.runs,
                R_runs_band,
                q_zf_cal,
                w_zf_cal,
                space_group=args.space_group,
            )
            Urs_cal, s_cal_r = polish_per_run_cal(U_cal_joint, Urs_alt_seed=Urs_zf_cal)
            ew_cal = float(s_cal_r.sum())
            used_per_run_cal = True
        else:
            Urs_cal = np.repeat(U_cal_joint[None], len(args.runs), axis=0)
            ew_cal = float(ew_cal_joint)
            used_per_run_cal = False

        if pol_j_nf is not None and ew_cal < 1.15 * max(
            float(ewald_ctx["ew_nom"]), 1.0
        ):
            U_nf, ew_nf_j = pol_j_nf(ewald_ctx["U_nom_joint"], spans=(0.12, 0.04), n=5)
            ewald_ctx["U_nom_joint"] = U_nf
            if per_run_active and not is_single_setting and pol_pr_nf is not None:
                q_zf_nf, w_zf_nf = _build_zone_funnel_obs(
                    C_bank, uhat_bank, vhat_bank, W_bank, H_bank
                )
                Urs_zf_nf = _zone_funnel_seed_per_run(
                    U_nf,
                    B0,
                    args.runs,
                    R_runs_nom,
                    q_zf_nf,
                    w_zf_nf,
                    space_group=args.space_group,
                )
                Urs_nf, s_nf_r = pol_pr_nf(U_nf, Urs_alt_seed=Urs_zf_nf)
                ewald_ctx["Urs_nom"] = Urs_nf
                ewald_ctx["ew_nom"] = float(s_nf_r.sum())
            else:
                ewald_ctx["ew_nom"] = float(ew_nf_j)

        ew_nom = float(ewald_ctx["ew_nom"])
        if ew_cal >= ew_nom:
            if used_per_run_cal and args.anchor_run in args.runs:
                U_cal_joint = Urs_cal[list(args.runs).index(args.anchor_run)].copy()
            U_opt = U_cal_joint
            p_cur[0:3] = Rot.from_matrix(U_opt @ np.asarray(Uj).T).as_rotvec()
            if used_per_run_cal:
                R_runs_final = dict(R_runs_band)
                for ir, r in enumerate(args.runs):
                    R_runs_final[r] = R_runs_band[r] @ Urs_cal[ir] @ U_cal_joint.T
                args.R_runs_refined = [R_runs_final[r] for r in range(n_runs)]
            Urs_win = Urs_cal
            R_runs_win = R_runs_band
            ew_eval_win = _eval_cal
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
            Urs_win = ewald_ctx["Urs_nom"]
            R_runs_win = R_runs_nom
            ew_eval_win = _ev_nf
            ew_best = ew_nom

        # Direct Ewald unit-cell ratio polish (decoupled from isotropic scale c):
        # preserves unit-cell ratios even when detector radial scale falls back to
        # nominal geometry, and breaks the flat L-BFGS-B band valley on large cells.
        # Alternate (cell <-> wider U + zone-funnel/per-run R) so monoclinic shear
        # and slipped runs decouple cleanly from unit-cell ratio changes.
        if (
            "cell" in args.modes
            and not is_single_setting
            and len(active_cell_slots) > 0
            and len(ewald_ctx["hkl_ew"]) > 0
        ):
            cell_vec = np.array(
                [float(p_cur[15]), float(p_cur[16]), float(p_cur[17])],
                dtype=np.float64,
            )
            for _cell_pass in range(2):
                if ew_eval_win is None:
                    ew_eval_win, _, _ = _make_ewald_polisher(
                        ewald_ctx["hkl_ew"] @ B_opt.T,
                        args.runs,
                        R_runs_win,
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
                keep_c = ew_eval_win.keep_idx
                if len(keep_c) == 0:
                    break
                hkl_sub = ewald_ctx["hkl_ew"][keep_c]

                def _eval_cell_vec(cv, _hkl_sub=hkl_sub, _ew=ew_eval_win, _urs=Urs_win):
                    vals = _apply_cell_jax(
                        cell0, symmetry_kind, float(cv[0]), float(cv[1]), float(cv[2])
                    )
                    B_try = np.asarray(
                        build_B_jax(*[float(v) for v in vals]), np.float64
                    )
                    raw_s = float(_ew.score_with_Gc(_hkl_sub @ B_try.T, _urs).sum())
                    # Gentle tie-breaking prior (~0.01% of score at 1% strain)
                    prior = 1e-4 * max(abs(raw_s), 1.0) * float(np.sum(cv * cv))
                    return raw_s - prior, raw_s

                best_obj_c, best_raw_c = _eval_cell_vec(cell_vec)
                cell_changed = False
                c_spans = (
                    (0.018, 0.005, 0.0014, 0.0004)
                    if _cell_pass == 0
                    else (0.006, 0.0018, 0.0005, 0.00015)
                )
                for c_span in c_spans:
                    for slot in active_cell_slots:
                        c0_s = cell_vec[slot]
                        lo_s = max(-0.035, c0_s - c_span)
                        hi_s = min(0.035, c0_s + c_span)
                        for cand_v in np.linspace(lo_s, hi_s, 9):
                            if abs(cand_v - c0_s) < 1e-9:
                                continue
                            cv_try = cell_vec.copy()
                            cv_try[slot] = float(cand_v)
                            obj_t, raw_t = _eval_cell_vec(cv_try)
                            if obj_t > best_obj_c + 1e-6:
                                best_obj_c = obj_t
                                best_raw_c = raw_t
                                cell_vec = cv_try
                                cell_changed = True
                if not cell_changed:
                    break
                p_cur[15:18] = cell_vec
                a_opt, b_opt, c_opt, al_opt_r, be_opt_r, ga_opt_r = [
                    float(v)
                    for v in _apply_cell_jax(
                        cell0,
                        symmetry_kind,
                        float(cell_vec[0]),
                        float(cell_vec[1]),
                        float(cell_vec[2]),
                    )
                ]
                al_opt, be_opt, ga_opt = np.degrees(
                    [al_opt_r, be_opt_r, ga_opt_r]
                ).tolist()
                B_opt = np.asarray(
                    build_B_jax(a_opt, b_opt, c_opt, al_opt_r, be_opt_r, ga_opt_r),
                    dtype=np.float64,
                )
                ew_best = best_raw_c
                # Fold Urs_win into R_runs_base_c so orientation polish starts at
                # the converged per-run orientations instead of resetting precession.
                R_runs_base_c = dict(R_runs_win)
                if per_run_active and not is_single_setting:
                    for ir, r in enumerate(args.runs):
                        R_runs_base_c[r] = R_runs_win[r] @ Urs_win[ir] @ U_opt.T
                ew_eval_c, pol_j_c, pol_pr_c = _make_ewald_polisher(
                    ewald_ctx["hkl_ew"] @ B_opt.T,
                    args.runs,
                    R_runs_base_c,
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
                base_c_new = float(
                    np.diag(
                        ew_eval_c(np.repeat(U_opt[None], len(args.runs), axis=0))
                    ).sum()
                )
                U_c_try, ew_j_c = pol_j_c(U_opt, spans=(0.60, 0.20, 0.06, 0.02), n=7)
                if per_run_active and not is_single_setting:
                    C_mid_c = (
                        C_final
                        + 0.5 * W_final[:, None] * uh_final
                        + 0.5 * H_final[:, None] * vh_final
                    )
                    q_zf_c, w_zf_c = _build_zone_funnel_obs(
                        C_mid_c,
                        uh_final,
                        vh_final,
                        W_final,
                        H_final,
                        s_vec=np.asarray(s_samp),
                    )
                    Urs_zf_c = _zone_funnel_seed_per_run(
                        U_c_try,
                        B_opt,
                        args.runs,
                        R_runs_base_c,
                        q_zf_c,
                        w_zf_c,
                        space_group=args.space_group,
                    )
                    Urs_c, s_c_r = pol_pr_c(
                        U_c_try,
                        spans=(0.50, 0.18, 0.06, 0.02),
                        n=5,
                        Urs_alt_seed=Urs_zf_c,
                    )
                    ew_c_tot = float(s_c_r.sum())
                    if ew_c_tot >= base_c_new:
                        if args.anchor_run in args.runs:
                            U_c_try = Urs_c[
                                list(args.runs).index(args.anchor_run)
                            ].copy()
                        U_opt = U_c_try
                        p_cur[0:3] = Rot.from_matrix(
                            U_opt @ np.asarray(Uj).T
                        ).as_rotvec()
                        ew_best = ew_c_tot
                        R_runs_final_c = dict(R_runs_base_c)
                        for ir, r in enumerate(args.runs):
                            R_runs_final_c[r] = R_runs_base_c[r] @ Urs_c[ir] @ U_opt.T
                        args.R_runs_refined = [R_runs_final_c[r] for r in range(n_runs)]
                        Urs_win = Urs_c
                        R_runs_win = R_runs_base_c
                        ew_eval_win = ew_eval_c
                elif float(ew_j_c) >= base_c_new:
                    U_opt = U_c_try
                    p_cur[0:3] = Rot.from_matrix(U_opt @ np.asarray(Uj).T).as_rotvec()
                    ew_best = float(ew_j_c)
                    Urs_win = np.repeat(U_opt[None], len(args.runs), axis=0)
                    ew_eval_win = ew_eval_c
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
        if ewald_ctx is not None and len(ewald_ctx["hkl_ew"]) > 0:
            s_base_arr = np.broadcast_to(
                np.asarray(s_samp, dtype=np.float64), (len(args.runs), 3)
            ).copy()
            s_pol_arr = s_base_arr + np.array(
                [trans_runs_pol[r] for r in args.runs], dtype=np.float64
            )
            _ev_st3, _, _ = _make_ewald_polisher(
                ewald_ctx["hkl_ew"] @ B_opt.T,
                args.runs,
                {r: R_runs_init[r] for r in args.runs},
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
                s_samp=s_base_arr,
            )
            _ev_st4, _, _ = _make_ewald_polisher(
                ewald_ctx["hkl_ew"] @ B_opt.T,
                args.runs,
                {r: R_runs_pol[r] for r in args.runs},
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
                s_samp=s_pol_arr,
            )
            U_ident_stack = np.repeat(U_opt[None], len(args.runs), axis=0)
            s_st3_r = np.diag(_ev_st3(U_ident_stack))
            s_st4_r = np.diag(_ev_st4(U_ident_stack))
            for ir, r in enumerate(args.runs):
                if float(s_st4_r[ir]) < 0.99 * float(s_st3_r[ir]):
                    R_runs_pol[r] = R_runs_init[r]
                    trans_runs_pol[r] = 0.0
        if (
            getattr(args, "_recovered_outer_R_runs", None) is None
            and args.anchor_run in args.runs
        ):
            from scipy.spatial.transform import Rotation as _RotAnc

            Q_anc4 = R_runs_nom[args.anchor_run].T @ R_runs_pol[args.anchor_run]
            s_vec_cur = np.asarray(s_samp, dtype=np.float64)
            for r in args.runs:
                R_runs_pol[r] = (
                    R_runs_nom[r].copy()
                    if r == args.anchor_run
                    else R_runs_pol[r] @ Q_anc4.T
                )
                trans_runs_pol[r] = Q_anc4 @ (s_vec_cur + trans_runs_pol[r]) - s_vec_cur
            U_opt = Q_anc4 @ U_opt
            p_cur[0:3] = _RotAnc.from_matrix(U_opt @ np.asarray(Uj).T).as_rotvec()
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

    if (
        getattr(args, "_recovered_outer_R_runs", None) is not None
        and getattr(args, "R_runs_refined", None) is None
    ):
        R_rec_list = []
        for r in range(n_runs):
            idx_r = np.flatnonzero(runs == r)
            if len(idx_r) and r < len(dom):
                dR_r = np.asarray(
                    rodrigues_jax(float(dom[r]) * jnp.asarray(scan_axes_obs[idx_r[0]]))
                )
                R_rec_list.append(dR_r @ R_obs[idx_r[0]])
            else:
                R_rec_list.append(np.asarray(R_runs_nom[r], dtype=np.float64))
        args.R_runs_refined = np.asarray(R_rec_list, dtype=np.float64)

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
    if pixel_em is not None:
        panel_cal_final = {
            bank_ids[i]: (
                C_final[i],
                uh_final[i],
                vh_final[i],
                float(W_final[i]),
                float(H_final[i]),
            )
            for i in range(n_banks)
        }
        R_runs_final_em = {}
        R_ref_list = getattr(args, "R_runs_refined", None)
        for r in args.runs:
            if R_ref_list is not None and r < len(R_ref_list):
                R_runs_final_em[r] = np.asarray(R_ref_list[r], dtype=np.float64)
            else:
                idx_r = np.flatnonzero(runs == r)
                if len(idx_r):
                    dR_r = np.asarray(
                        rodrigues_jax(
                            float(dom[r]) * jnp.asarray(scan_axes_obs[idx_r[0]])
                        )
                    )
                    R_runs_final_em[r] = dR_r @ R_obs[idx_r[0]]
                else:
                    R_runs_final_em[r] = R_runs_nom[r]
        final_em_stats = pixel_em.update_e_step(
            U_opt,
            B_opt,
            R_runs=R_runs_final_em,
            panel_cal=panel_cal_final,
            sample_shift_m=np.asarray(s_samp),
            match_sigma_px=4.5,
        )
        em_export = pixel_em.export_pixel_masks()
        args.em_pixel_result = em_export
        print(
            f"[EM Pixel Classifier] valid={em_export.valid_fraction * 100:.2f}%, "
            f"dark={em_export.dark_fraction * 100:.2f}% "
            f"(dead_banks={em_export.dead_banks}), "
            f"bright={em_export.bright_fraction * 100:.3f}% "
            f"(bright_bins={em_export.n_bright_bins}, tau={final_em_stats['tau_cryst']:.1f})",
            flush=True,
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
    return {"status": status, "stages": diagnostics}
