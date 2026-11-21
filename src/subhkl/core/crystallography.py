import numpy as np
import numpy.typing as npt
import scipy.linalg

from subhkl.core.spacegroup import is_systematically_absent


def cartesian_matrix_metric_tensor(a, b, c, alpha, beta, gamma):
    """
    Calculates the B matrix (orientation matrix) and G* (reciprocal metric tensor).
    """
    G = np.array(
        [
            [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [b * a * np.cos(gamma), b**2, b * c * np.cos(alpha)],
            [c * a * np.cos(beta), c * b * np.cos(alpha), c**2],
        ]
    )
    Gstar = np.linalg.inv(G)
    B = scipy.linalg.cholesky(Gstar, lower=False)
    return B, Gstar


def generate_reflections(a, b, c, alpha, beta, gamma, space_group="P 1", d_min=2.0):
    """
    Generates unique HKL indices for a given unit cell and resolution cutoff.

    Uses exact 2D (h, k) rod quadratic bounds along l in slabs when the 3D bounding
    box is large, avoiding multi-GB 3D meshgrid allocations at high resolution (e.g.
    d_min = 0.5 A on large macromolecular unit cells) while preserving identical output.
    """
    if space_group is None:
        space_group = "P 1"
    constants = a, b, c, *np.deg2rad([alpha, beta, gamma])
    _B, Gstar = cartesian_matrix_metric_tensor(*constants)

    astar, bstar, cstar = np.sqrt(np.diag(Gstar))

    h_max = int(np.floor(1 / d_min / astar))
    k_max = int(np.floor(1 / d_min / bstar))
    l_max = int(np.floor(1 / d_min / cstar))

    box_count = (2 * h_max + 1) * (2 * k_max + 1) * (2 * l_max + 1)
    if box_count <= 1_000_000:
        h, k, l = np.meshgrid(
            np.arange(-h_max, h_max + 1),
            np.arange(-k_max, k_max + 1),
            np.arange(-l_max, l_max + 1),
            indexing="ij",
        )
        h_flat, k_flat, l_flat = h.flatten(), k.flatten(), l.flatten()
        hkl_sq = np.einsum(
            "ij,jl,il->l", Gstar, [h_flat, k_flat, l_flat], [h_flat, k_flat, l_flat]
        )
        with np.errstate(divide="ignore"):
            d = 1 / np.sqrt(hkl_sq)
        res_mask = (d > d_min) & (d < np.inf)
        absent_mask = is_systematically_absent(h_flat, k_flat, l_flat, space_group)
        final_mask = res_mask & (~absent_mask)
        return h_flat[final_mask], k_flat[final_mask], l_flat[final_mask]

    # Exact 2D (h, k) rod bounds along l:
    # G*_{22} l^2 + 2 (G*_{02} h + G*_{12} k) l + (G*_{00} h^2 + 2 G*_{01} h k + G*_{11} k^2 - 1/d_min^2) < 0
    hh, kk = np.meshgrid(
        np.arange(-h_max, h_max + 1, dtype=np.int64),
        np.arange(-k_max, k_max + 1, dtype=np.int64),
        indexing="ij",
    )
    hh_f = hh.ravel()
    kk_f = kk.ravel()
    g00 = float(Gstar[0, 0])
    g01 = float(Gstar[0, 1])
    g02 = float(Gstar[0, 2])
    g11 = float(Gstar[1, 1])
    g12 = float(Gstar[1, 2])
    g22 = float(Gstar[2, 2])
    q_max_sq = 1.0 / (float(d_min) * float(d_min))

    hf = hh_f.astype(np.float64)
    kf = kk_f.astype(np.float64)
    b_rod = g02 * hf + g12 * kf
    c_rod = (g00 * hf * hf + 2.0 * g01 * hf * kf + g11 * kf * kf) - q_max_sq
    disc = b_rod * b_rod - g22 * c_rod
    valid_rods = disc >= 0.0
    if not np.any(valid_rods):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    hh_v = hh_f[valid_rods]
    kk_v = kk_f[valid_rods]
    b_v = b_rod[valid_rods]
    sq_v = np.sqrt(np.maximum(0.0, disc[valid_rods]))
    l_lo = np.maximum(-l_max, np.floor((-b_v - sq_v) / g22).astype(np.int64))
    l_hi = np.minimum(l_max, np.ceil((-b_v + sq_v) / g22).astype(np.int64))
    counts = l_hi - l_lo + 1
    pos = counts > 0
    hh_v = hh_v[pos]
    kk_v = kk_v[pos]
    l_lo = l_lo[pos]
    counts = counts[pos]
    if len(counts) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    slab_target = 2_000_000
    cum = np.cumsum(counts)
    n_rods = len(counts)
    h_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []
    l_parts: list[np.ndarray] = []

    rod_start = 0
    while rod_start < n_rods:
        base = int(cum[rod_start - 1]) if rod_start > 0 else 0
        rod_end = int(np.searchsorted(cum, base + slab_target, side="right"))
        if rod_end <= rod_start:
            rod_end = rod_start + 1
        c_sub = counts[rod_start:rod_end]
        lo_sub = l_lo[rod_start:rod_end]
        h_sub = np.repeat(hh_v[rod_start:rod_end], c_sub)
        k_sub = np.repeat(kk_v[rod_start:rod_end], c_sub)
        offsets = np.cumsum(c_sub) - c_sub
        l_sub = (
            np.arange(int(c_sub.sum()), dtype=np.int64)
            - np.repeat(offsets, c_sub)
            + np.repeat(lo_sub, c_sub)
        )

        hs = h_sub.astype(np.float64)
        ks = k_sub.astype(np.float64)
        ls = l_sub.astype(np.float64)
        hkl_sq = (
            g00 * hs * hs
            + g11 * ks * ks
            + g22 * ls * ls
            + 2.0 * (g01 * hs * ks + g02 * hs * ls + g12 * ks * ls)
        )
        with np.errstate(divide="ignore"):
            d_sub = 1.0 / np.sqrt(hkl_sq)
        res_mask = (d_sub > d_min) & (d_sub < np.inf)
        if np.any(res_mask):
            h_keep = h_sub[res_mask]
            k_keep = k_sub[res_mask]
            l_keep = l_sub[res_mask]
            keep = ~is_systematically_absent(h_keep, k_keep, l_keep, space_group)
            if np.any(keep):
                h_parts.append(h_keep[keep])
                k_parts.append(k_keep[keep])
                l_parts.append(l_keep[keep])
        rod_start = rod_end

    if not h_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    return (
        np.concatenate(h_parts),
        np.concatenate(k_parts),
        np.concatenate(l_parts),
    )


def generate_ewald_reflections(
    a,
    b,
    c,
    alpha,
    beta,
    gamma,
    space_group="P 1",
    d_min=2.0,
    UB=None,
    R_runs=None,
    ki=(0.0, 0.0, 1.0),
    wavelength_range=(1.0, 4.5),
):
    """
    Generates unique HKL indices that satisfy d > d_min AND intersect the polychromatic
    Ewald shell(s) [wl_min, wl_max] for one or more goniometer orientations R_runs.
    """
    wl_min = max(1e-4, float(wavelength_range[0]))
    wl_max = max(wl_min + 1e-4, float(wavelength_range[1]))
    d_eff = max(float(d_min), 0.5 * wl_min - 1e-6)
    if UB is None:
        return generate_reflections(
            a, b, c, alpha, beta, gamma, space_group=space_group, d_min=d_eff
        )

    UB_mat = np.asarray(UB, dtype=np.float64)
    ki_vec = np.asarray(ki, dtype=np.float64)
    ki_vec = ki_vec / max(float(np.linalg.norm(ki_vec)), 1e-12)
    if R_runs is None or len(R_runs) == 0:
        r_mats = [np.eye(3, dtype=np.float64)]
    else:
        r_mats = [np.asarray(r_m, dtype=np.float64) for r_m in R_runs]

    if space_group is None:
        space_group = "P 1"

    b1 = UB_mat[:, 0]
    b2 = UB_mat[:, 1]
    b3 = UB_mat[:, 2]
    c_star2 = max(float(np.dot(b3, b3)), 1e-12)
    inv_ub = np.linalg.inv(UB_mat)
    a_vec, b_vec = inv_ub[0], inv_ub[1]
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))

    q_max = min(1.0 / d_eff, 2.0 / wl_min)
    q_max2 = q_max * q_max
    r_out = 1.0 / wl_min

    # Bounding sphere around outer Ewald sphere(s) c_out,i = -(R_i^T ki) / wl_min
    k_c_all = np.array([r_m.T @ ki_vec for r_m in r_mats], dtype=np.float64)
    c_outs = -k_c_all * r_out
    c_env = np.mean(c_outs, axis=0)
    spread = float(np.max(np.linalg.norm(c_outs - c_env[None, :], axis=1)))
    r_env = r_out + spread
    r_env2 = r_env * r_env

    h_env = float(np.dot(a_vec, c_env))
    k_env = float(np.dot(b_vec, c_env))
    h_lo = max(
        -int(np.ceil(a_len * q_max)), int(np.floor(h_env - a_len * r_env - 1e-5))
    )
    h_hi = min(int(np.ceil(a_len * q_max)), int(np.ceil(h_env + a_len * r_env + 1e-5)))
    k_lo = max(
        -int(np.ceil(b_len * q_max)), int(np.floor(k_env - b_len * r_env - 1e-5))
    )
    k_hi = min(int(np.ceil(b_len * q_max)), int(np.ceil(k_env + b_len * r_env + 1e-5)))
    if h_lo > h_hi or k_lo > k_hi:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    hh, kk = np.meshgrid(
        np.arange(h_lo, h_hi + 1, dtype=np.int64),
        np.arange(k_lo, k_hi + 1, dtype=np.int64),
        indexing="ij",
    )
    hh_f = hh.ravel()
    kk_f = kk.ravel()
    hf = hh_f.astype(np.float64)
    kf = kk_f.astype(np.float64)

    px = hf * b1[0] + kf * b2[0]
    py = hf * b1[1] + kf * b2[1]
    pz = hf * b1[2] + kf * b2[2]

    # Intersect resolution sphere |p + l b3|^2 < q_max^2 and Ewald envelope |p - c_env + l b3|^2 <= r_env^2
    p_dot_b3 = px * b3[0] + py * b3[1] + pz * b3[2]
    p2 = px * px + py * py + pz * pz
    disc_res = p_dot_b3 * p_dot_b3 - c_star2 * (p2 - q_max2)

    pox = px - c_env[0]
    poy = py - c_env[1]
    poz = pz - c_env[2]
    po_dot_b3 = pox * b3[0] + poy * b3[1] + poz * b3[2]
    po2 = pox * pox + poy * poy + poz * poz
    disc_env = po_dot_b3 * po_dot_b3 - c_star2 * (po2 - r_env2)

    valid = (disc_res >= 0.0) & (disc_env >= 0.0)
    if not np.any(valid):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    hh_v = hh_f[valid]
    kk_v = kk_f[valid]
    px_v = px[valid]
    py_v = py[valid]
    pz_v = pz[valid]
    p_b3_v = p_dot_b3[valid]
    po_b3_v = po_dot_b3[valid]
    sq_res = np.sqrt(np.maximum(0.0, disc_res[valid]))
    sq_env = np.sqrt(np.maximum(0.0, disc_env[valid]))

    l_lo = np.maximum(
        np.floor((-p_b3_v - sq_res) / c_star2).astype(np.int64),
        np.floor((-po_b3_v - sq_env) / c_star2).astype(np.int64),
    )
    l_hi = np.minimum(
        np.ceil((-p_b3_v + sq_res) / c_star2).astype(np.int64),
        np.ceil((-po_b3_v + sq_env) / c_star2).astype(np.int64),
    )

    # For single-run orientation, also punch out the inner Ewald sphere (wl > wl_max)
    intervals_h: list[np.ndarray] = []
    intervals_k: list[np.ndarray] = []
    intervals_lo: list[np.ndarray] = []
    intervals_hi: list[np.ndarray] = []

    if len(r_mats) == 1:
        r_in = 1.0 / wl_max
        c_in = -k_c_all[0] * r_in
        pix = px_v - c_in[0]
        piy = py_v - c_in[1]
        piz = pz_v - c_in[2]
        pi_b3 = pix * b3[0] + piy * b3[1] + piz * b3[2]
        pi2 = pix * pix + piy * piy + piz * piz
        disc_in = pi_b3 * pi_b3 - c_star2 * (pi2 - r_in * r_in)
        has_hole = disc_in > 0.0
        sq_in = np.sqrt(np.maximum(0.0, disc_in))
        excl_lo = np.ceil((-pi_b3 - sq_in) / c_star2 + 1e-5).astype(np.int64)
        excl_hi = np.floor((-pi_b3 + sq_in) / c_star2 - 1e-5).astype(np.int64)
        split = has_hole & (excl_lo <= excl_hi)

        no_split = (~split) & (l_lo <= l_hi)
        if np.any(no_split):
            intervals_h.append(hh_v[no_split])
            intervals_k.append(kk_v[no_split])
            intervals_lo.append(l_lo[no_split])
            intervals_hi.append(l_hi[no_split])
        if np.any(split):
            hi_1 = np.minimum(l_hi[split], excl_lo[split] - 1)
            lo_1 = l_lo[split]
            m1 = lo_1 <= hi_1
            if np.any(m1):
                intervals_h.append(hh_v[split][m1])
                intervals_k.append(kk_v[split][m1])
                intervals_lo.append(lo_1[m1])
                intervals_hi.append(hi_1[m1])
            lo_2 = np.maximum(l_lo[split], excl_hi[split] + 1)
            hi_2 = l_hi[split]
            m2 = lo_2 <= hi_2
            if np.any(m2):
                intervals_h.append(hh_v[split][m2])
                intervals_k.append(kk_v[split][m2])
                intervals_lo.append(lo_2[m2])
                intervals_hi.append(hi_2[m2])
    else:
        pos = l_lo <= l_hi
        if np.any(pos):
            intervals_h.append(hh_v[pos])
            intervals_k.append(kk_v[pos])
            intervals_lo.append(l_lo[pos])
            intervals_hi.append(l_hi[pos])

    if not intervals_h:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    hh_i = np.concatenate(intervals_h)
    kk_i = np.concatenate(intervals_k)
    lo_i = np.concatenate(intervals_lo)
    hi_i = np.concatenate(intervals_hi)
    counts = hi_i - lo_i + 1

    q_cut_sq = 1.0 / (float(d_min) * float(d_min))
    slab_target = 2_000_000
    cum = np.cumsum(counts)
    n_int = len(counts)
    h_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []
    l_parts: list[np.ndarray] = []

    idx_start = 0
    while idx_start < n_int:
        base = int(cum[idx_start - 1]) if idx_start > 0 else 0
        idx_end = int(np.searchsorted(cum, base + slab_target, side="right"))
        if idx_end <= idx_start:
            idx_end = idx_start + 1
        c_sub = counts[idx_start:idx_end]
        lo_sub = lo_i[idx_start:idx_end]
        h_sub = np.repeat(hh_i[idx_start:idx_end], c_sub)
        k_sub = np.repeat(kk_i[idx_start:idx_end], c_sub)
        offsets = np.cumsum(c_sub) - c_sub
        l_sub = (
            np.arange(int(c_sub.sum()), dtype=np.int64)
            - np.repeat(offsets, c_sub)
            + np.repeat(lo_sub, c_sub)
        )

        hs = h_sub.astype(np.float64)
        ks = k_sub.astype(np.float64)
        ls = l_sub.astype(np.float64)
        gx = hs * b1[0] + ks * b2[0] + ls * b3[0]
        gy = hs * b1[1] + ks * b2[1] + ls * b3[1]
        gz = hs * b1[2] + ks * b2[2] + ls * b3[2]
        g2 = gx * gx + gy * gy + gz * gz
        res_mask = (g2 > 1e-12) & (g2 < q_cut_sq)
        if np.any(res_mask):
            h_keep = h_sub[res_mask]
            k_keep = k_sub[res_mask]
            l_keep = l_sub[res_mask]
            gx_k = gx[res_mask]
            gy_k = gy[res_mask]
            gz_k = gz[res_mask]
            g2_k = g2[res_mask]
            # Exact check that at least one run diffracts in [wl_min, wl_max]
            ew_ok = np.zeros(len(h_keep), dtype=bool)
            for kc in k_c_all:
                g_dot_k = gx_k * kc[0] + gy_k * kc[1] + gz_k * kc[2]
                lam = (-2.0 * g_dot_k) / g2_k
                ew_ok |= (lam >= wl_min) & (lam <= wl_max)
            if np.any(ew_ok):
                h_keep = h_keep[ew_ok]
                k_keep = k_keep[ew_ok]
                l_keep = l_keep[ew_ok]
                keep = ~is_systematically_absent(h_keep, k_keep, l_keep, space_group)
                if np.any(keep):
                    h_parts.append(h_keep[keep])
                    k_parts.append(k_keep[keep])
                    l_parts.append(l_keep[keep])
        idx_start = idx_end

    if not h_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    return (
        np.concatenate(h_parts),
        np.concatenate(k_parts),
        np.concatenate(l_parts),
    )


def get_q_lab(
    h: npt.ArrayLike,
    k: npt.ArrayLike,
    l: npt.ArrayLike,
    RUB: npt.ArrayLike,
) -> npt.NDArray:
    """
    Calculate Q vectors in the Lab Frame.
    Q_lab = RUB * hkl
    RUB should be the composite matrix (R @ U @ B).
    """
    hkl = np.stack([h, k, l], axis=1)  # (N, 3)

    # Handle RUB shape: (3,3) or (N,3,3)
    if RUB.ndim == 3:
        # Einsum: n=batch, i=row, j=col. RUB[n,i,j] * hkl[n,j] -> out[n,i]
        q_lab = np.einsum("nij,nj->ni", RUB, hkl)
    else:
        # Standard matmul: hkl @ RUB.T
        q_lab = hkl @ RUB.T

    return q_lab
