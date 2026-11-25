"""Complex structure-factor nodal surfaces, 1D phase vortices, and 3D/Wulff/Detector JS renderer.

Provides:
- Space-group-accurate complex structure-factor expansion F(hkl) = |F| exp(i*phi) from phased .mtz files
  (preserving screw/glide translation phase shifts and Friedel Hermiticity F(-h) = F*(h)).
- Band-limited Whittaker-Shannon continuous interpolation Z(G) via centered real-space Fourier synthesis
  and fast separable BLAS tensor contractions on arbitrary 3D regions of interest (ROIs).
- Heterodyne interference modes with a phased reference F_0 .mtz:
    * "single"  : Z(G) = F(G)
    * "delta"   : Z(G) = F_0*(G) * (F(G) - F_0(G))   [two full models, e.g. protonated vs reference]
    * "product" : Z(G) = F_0*(G) * F(G)              [primary .mtz is already a difference map Delta F]
- Scoped 3D ROI Marching Cubes triangulation of Re(Z)=0 and Im(Z)=0 nodal surfaces with analytical
  reciprocal-space normals nabla_G S = (UB)^(-T) nabla_h S.
- Gauge-invariant 3D plaquette phase-winding extraction (Re(Z)=0 intersect Im(Z)=0) with sub-voxel
  Newton refinement and flux-conserving polyline chaining for 1D phase vortices.
- Self-contained JavaScript controller/renderer (`get_nodal_surfaces_js()`) for:
    * 3D Reciprocal Space view (WebGL translucent dual-surface + depth-buffered peaks/planes/vortices
      with automatic Canvas2D painter fallback),
    * Wulff Net <-> 3D Reciprocal Space view toggle,
    * Real-time 1D vortex curve projections onto both the Wulff sphere and 3D detector panels.
"""

from __future__ import annotations

import base64
import math
import os
from collections import OrderedDict
from typing import Any

import numpy as np

_F_LABEL_PRIORITY: tuple[str, ...] = (
    "F",
    "FP",
    "FC",
    "FWT",
    "F_CALC",
    "FCALC",
    "FMODEL",
    "F_MODEL",
    "FC_ALL",
    "DELFWT",
    "FOFCWT",
)

_PHI_LABEL_PRIORITY: tuple[str, ...] = (
    "PHIC",
    "PHIF_CALC",
    "PHIFCALC",
    "PHWT",
    "PHI",
    "PHIFMODEL",
    "PHIF_MODEL",
    "PHFC",
    "PHDELWT",
    "PHFOFCWT",
)

_PHASED_FIELD_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_GLOBAL_VORTEX_CACHE: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()
_MAX_FIELD_CACHE_ENTRIES = 8


def detect_mtz_f_and_phase_columns(mtz: Any) -> tuple[Any | None, Any | None]:
    """Locate the best matching (amplitude 'F', phase 'P') column pair in a gemmi.Mtz object."""
    f_cols = [c for c in mtz.columns if c.type == "F"]
    p_cols = [c for c in mtz.columns if c.type == "P"]
    if not f_cols:
        return None, None

    f_col = None
    by_upper_f = {c.label.upper(): c for c in f_cols}
    for pref in _F_LABEL_PRIORITY:
        if pref in by_upper_f:
            f_col = by_upper_f[pref]
            break
    if f_col is None:
        f_col = f_cols[0]

    if not p_cols:
        return f_col, None

    f_up = f_col.label.upper()
    by_upper_p = {c.label.upper(): c for c in p_cols}
    # Prefer a phase column paired to the chosen amplitude stem (e.g. f_calc -> PHIf_calc, FWT -> PHWT)
    stem_candidates = [
        f"PHI{f_up}",
        f"PH{f_up}",
        f"PHI_{f_up}",
        f_up.replace("F", "PHI", 1) if f_up.startswith("F") else "",
        f_up.replace("F", "PH", 1) if f_up.startswith("F") else "",
    ]
    for cand in stem_candidates:
        if cand and cand in by_upper_p:
            return f_col, by_upper_p[cand]

    for pref in _PHI_LABEL_PRIORITY:
        if pref in by_upper_p:
            return f_col, by_upper_p[pref]

    return f_col, p_cols[0]


def load_phased_structure_factor_field(
    mtz_path: str | os.PathLike[str],
    *,
    space_group_override: str | None = None,
    d_min_floor: float = 1.35,
    max_half_index: int = 56,
) -> dict[str, Any]:
    """Load and symmetry-expand a phased .mtz into a band-limited 3D Whittaker-Shannon field.

    Returns a dictionary containing metadata (`has_phases`, `f_col`, `phi_col`, `q75_f`, `asu_map`, ...)
    and, when phases are present, centered unit-cell Fourier density `rho` on [-1/2, 1/2)^3.
    """
    import gemmi

    resolved = os.path.realpath(str(mtz_path))
    st = os.stat(resolved)
    cache_key = (
        resolved,
        int(st.st_mtime_ns),
        int(st.st_size),
        str(space_group_override or "").strip(),
        round(float(d_min_floor), 3),
        int(max_half_index),
    )
    cached = _PHASED_FIELD_CACHE.get(cache_key)
    if cached is not None:
        _PHASED_FIELD_CACHE.move_to_end(cache_key)
        return cached

    mtz = gemmi.read_mtz_file(resolved)
    f_col, phi_col = detect_mtz_f_and_phase_columns(mtz)
    mtz_sg = mtz.spacegroup or (
        gemmi.find_spacegroup_by_name(space_group_override)
        if space_group_override
        else None
    )
    if mtz_sg is None:
        mtz_sg = gemmi.find_spacegroup_by_name("P 1")

    cell_params = [float(x) for x in mtz.cell.parameters]
    base_info: dict[str, Any] = {
        "path": resolved,
        "name": os.path.basename(resolved),
        "space_group": mtz_sg.hm,
        "is_centrosymmetric": bool(mtz_sg.is_centrosymmetric()),
        "cell": cell_params,
        "f_column": f_col.label if f_col is not None else None,
        "phi_column": phi_col.label if phi_col is not None else None,
        "has_phases": bool(f_col is not None and phi_col is not None),
    }
    if f_col is None or phi_col is None:
        return base_info

    arr = np.array(mtz, copy=False)
    col_labels = [c.label for c in mtz.columns]
    h_idx = col_labels.index("H")
    k_idx = col_labels.index("K")
    l_idx = col_labels.index("L")
    f_idx = col_labels.index(f_col.label)
    p_idx = col_labels.index(phi_col.label)

    hkl_raw = arr[:, [h_idx, k_idx, l_idx]].astype(np.float64)
    f_raw = arr[:, f_idx].astype(np.float64)
    phi_deg_raw = arr[:, p_idx].astype(np.float64)

    valid = (
        np.all(np.isfinite(hkl_raw), axis=1)
        & np.isfinite(f_raw)
        & (f_raw >= 0.0)
        & np.isfinite(phi_deg_raw)
    )
    hkl = np.rint(hkl_raw[valid]).astype(np.int32)
    f_amp = f_raw[valid]
    phi_rad = np.deg2rad(phi_deg_raw[valid])

    # Remove (0,0,0) if present and apply resolution floor + index cap for fast bounded synthesis
    non_origin = np.any(hkl != 0, axis=1)
    hkl = hkl[non_origin]
    f_amp = f_amp[non_origin]
    phi_rad = phi_rad[non_origin]

    if len(hkl) == 0:
        base_info["has_phases"] = False
        return base_info

    # Resolution filter using MTZ unit cell
    frac_mat = np.array(mtz.cell.frac.mat, dtype=np.float64)
    # In gemmi, cell.calculate_1_d2(hkl) = |B * hkl|^2 where B = frac.mat.T
    g_vecs_mtz = hkl @ frac_mat
    inv_d = np.linalg.norm(g_vecs_mtz, axis=1)
    if d_min_floor and d_min_floor > 0.0:
        keep_res = inv_d <= (1.0 / float(d_min_floor)) * 1.0001
        if np.count_nonzero(keep_res) >= 8:
            hkl = hkl[keep_res]
            f_amp = f_amp[keep_res]
            phi_rad = phi_rad[keep_res]

    pos_amps = f_amp[f_amp > 0.0]
    q75_f = float(np.percentile(pos_amps, 75.0)) if len(pos_amps) > 0 else 1.0
    if not np.isfinite(q75_f) or q75_f <= 1e-12:
        q75_f = 1.0

    f_norm = f_amp / q75_f
    z_asu = f_norm * np.exp(1j * phi_rad)

    # Build ASU amplitude map (using active/MTZ reciprocal ASU) for inter-MTZ heterodyne scale matching
    ops = mtz_sg.operations()
    asu_helper = gemmi.ReciprocalAsu(mtz_sg)
    asu_amp_map: dict[tuple[int, int, int], float] = {}
    # Sample up to 25,000 reflections for robust shared-ASU median/Q75 scale alignment
    step_sample = max(1, len(hkl) // 25000)
    for idx_s in range(0, len(hkl), step_sample):
        hk_list = [int(hkl[idx_s, 0]), int(hkl[idx_s, 1]), int(hkl[idx_s, 2])]
        asu_hkl, _ = asu_helper.to_asu(hk_list, ops)
        asu_amp_map[(int(asu_hkl[0]), int(asu_hkl[1]), int(asu_hkl[2]))] = float(
            f_norm[idx_s]
        )

    # Vectorized space-group expansion with exact translation phase shifts:
    # F(h * R) = F(h) * exp(-2*pi*i * h . t)
    rot_h_list: list[np.ndarray] = []
    rot_z_list: list[np.ndarray] = []
    for op in ops:
        rot = np.array(op.rot, dtype=np.int32) // op.DEN
        tran = np.array(op.tran, dtype=np.float64) / float(op.DEN)
        h_eq = hkl @ rot
        phase_shift = -2.0 * np.pi * (hkl @ tran)
        z_eq = z_asu * np.exp(1j * phase_shift)
        rot_h_list.append(h_eq)
        rot_z_list.append(z_eq)

    H_sym = np.concatenate(rot_h_list, axis=0)
    Z_sym = np.concatenate(rot_z_list, axis=0)

    in_cap = np.all(np.abs(H_sym) <= int(max_half_index), axis=1)
    H_sym = H_sym[in_cap]
    Z_sym = Z_sym[in_cap]
    if len(H_sym) == 0:
        base_info["has_phases"] = False
        return base_info

    hmax = np.max(np.abs(H_sym), axis=0).astype(np.int32)
    hmax = np.maximum(hmax, 1)
    dims = (int(2 * hmax[0] + 1), int(2 * hmax[1] + 1), int(2 * hmax[2] + 1))

    grid_sum = np.zeros(dims, dtype=np.complex128)
    grid_cnt = np.zeros(dims, dtype=np.int32)
    idx_pos = (
        H_sym[:, 0] + hmax[0],
        H_sym[:, 1] + hmax[1],
        H_sym[:, 2] + hmax[2],
    )
    np.add.at(grid_sum, idx_pos, Z_sym)
    np.add.at(grid_cnt, idx_pos, 1)

    F_int = np.zeros(dims, dtype=np.complex128)
    occ = grid_cnt > 0
    F_int[occ] = grid_sum[occ] / grid_cnt[occ]

    # Fill missing Friedel mates (-h,-k,-l) via F(-h) = conj(F(h)) while preserving explicit anomalous pairs
    F_friedel = np.conj(F_int[::-1, ::-1, ::-1])
    occ_friedel = occ[::-1, ::-1, ::-1]
    missing_mate = (~occ) & occ_friedel
    F_int[missing_mate] = F_friedel[missing_mate]

    # Centered real-space density on x_d in [-1/2, 1/2):
    # F(h) = sum_j rho[j] * exp(+2*pi*i * h . x[j]), exact at all integer Bragg indices
    rho = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(F_int))) / float(F_int.size)
    x_coords = [
        (np.arange(dims[d], dtype=np.float64) - float(hmax[d])) / float(dims[d])
        for d in range(3)
    ]

    result = {
        **base_info,
        "q75_f": q75_f,
        "n_asu_reflections": len(hkl),
        "n_expanded": int(np.count_nonzero(occ | missing_mate)),
        "hmax": [int(x) for x in hmax],
        "dims": dims,
        "F_int": F_int,
        "rho": rho,
        "x_coords": x_coords,
        "asu_amp_map": asu_amp_map,
    }
    _PHASED_FIELD_CACHE[cache_key] = result
    while len(_PHASED_FIELD_CACHE) > _MAX_FIELD_CACHE_ENTRIES:
        _PHASED_FIELD_CACHE.popitem(last=False)
    return result


def evaluate_field_on_hkl_grid(
    field: dict[str, Any],
    h0: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
) -> np.ndarray:
    """Evaluate Whittaker-Shannon interpolated complex F(h0, h1, h2) on a 3D rectilinear Miller grid."""
    rho = field["rho"]
    x0, x1, x2 = field["x_coords"]
    e0 = np.exp(2j * np.pi * np.outer(np.asarray(h0, dtype=np.float64), x0))
    e1 = np.exp(2j * np.pi * np.outer(np.asarray(h1, dtype=np.float64), x1))
    e2 = np.exp(2j * np.pi * np.outer(np.asarray(h2, dtype=np.float64), x2))
    # Separable 3-mode contraction: O(G*N0*N1*N2 + G^2*N1*N2 + G^3*N2)
    t0 = np.tensordot(e0, rho, axes=([1], [0]))  # (G0, N1, N2)
    t1 = np.tensordot(e1, t0, axes=([1], [1])).transpose(1, 0, 2)  # (G0, G1, N2)
    return np.tensordot(t1, e2, axes=([2], [1]))  # (G0, G1, G2)


def compute_shared_scale_ratio(
    primary_field: dict[str, Any],
    ref_field: dict[str, Any],
) -> float:
    """Compute robust amplitude ratio matching normalized primary |F| onto reference |F_0| on shared ASU HKLs."""
    map_f = primary_field.get("asu_amp_map") or {}
    map_0 = ref_field.get("asu_amp_map") or {}
    if not map_f or not map_0:
        return 1.0
    shared_f: list[float] = []
    shared_0: list[float] = []
    it_small, it_large, flip = (
        (map_f, map_0, False) if len(map_f) <= len(map_0) else (map_0, map_f, True)
    )
    for k, v_s in it_small.items():
        v_l = it_large.get(k)
        if v_l is not None and v_s > 1e-6 and v_l > 1e-6:
            if flip:
                shared_0.append(v_s)
                shared_f.append(v_l)
            else:
                shared_f.append(v_s)
                shared_0.append(v_l)
    if len(shared_f) < 5:
        return 1.0
    arr_f = np.asarray(shared_f, dtype=np.float64)
    arr_0 = np.asarray(shared_0, dtype=np.float64)
    q75_f = float(np.percentile(arr_f, 75.0))
    q75_0 = float(np.percentile(arr_0, 75.0))
    if q75_f > 1e-9 and q75_0 > 1e-9:
        ratio = q75_0 / q75_f
        if np.isfinite(ratio) and 0.05 <= ratio <= 20.0:
            return float(ratio)
    return 1.0


def evaluate_combined_complex_grid(
    primary_field: dict[str, Any],
    h0: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    *,
    ref_field: dict[str, Any] | None = None,
    heterodyne_mode: str = "delta",
    f_to_f0_scale: float = 1.0,
) -> np.ndarray:
    """Evaluate single or heterodyne complex interference field Z(h) on (h0, h1, h2).

    Crucially, F(h) and F_0(h) are band-limited interpolated separately BEFORE forming the
    quadratic interference product so double-bandwidth aliasing is avoided and 1D phase vortices
    sit exactly along the simultaneous zeros of F_0* and Delta F (or F).
    """
    F_grid = evaluate_field_on_hkl_grid(primary_field, h0, h1, h2)
    if (
        ref_field is None
        or not ref_field.get("has_phases")
        or heterodyne_mode == "single"
    ):
        return F_grid
    F0_grid = evaluate_field_on_hkl_grid(ref_field, h0, h1, h2)
    mode = str(heterodyne_mode or "delta").lower()
    if mode == "product":
        return np.conj(F0_grid) * F_grid
    # Default "delta": F_0*(G) * (s * F(G) - F_0(G))
    return np.conj(F0_grid) * (float(f_to_f0_scale) * F_grid - F0_grid)


def _refine_bilinear_zero(
    z00: np.ndarray,
    z10: np.ndarray,
    z11: np.ndarray,
    z01: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized 2-step Newton solver for Z(u,v) = (1-u)(1-v)z00 + u(1-v)z10 + uv*z11 + (1-u)v*z01 = 0."""
    u = np.full(z00.shape, 0.5, dtype=np.float64)
    v = np.full(z00.shape, 0.5, dtype=np.float64)
    a = z00
    b = z10 - z00
    c = z01 - z00
    d = z00 - z10 + z11 - z01
    for _ in range(3):
        f_val = a + b * u + c * v + d * (u * v)
        df_du = b + d * v
        df_dv = c + d * u
        j00, j01 = df_du.real, df_dv.real
        j10, j11 = df_du.imag, df_dv.imag
        det = j00 * j11 - j01 * j10
        ok = np.abs(det) > 1e-18
        du = np.where(
            ok, (f_val.real * j11 - f_val.imag * j01) / np.where(ok, det, 1.0), 0.0
        )
        dv = np.where(
            ok, (j00 * f_val.imag - j10 * f_val.real) / np.where(ok, det, 1.0), 0.0
        )
        u = np.clip(u - du, 0.05, 0.95)
        v = np.clip(v - dv, 0.05, 0.95)
    return u, v


def extract_vortex_polylines(
    Z: np.ndarray,
    h0: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    ub_matrix: np.ndarray,
    *,
    g_max: float | None = None,
    roi_center_g: np.ndarray | None = None,
    roi_radius_g: float | None = None,
    slab_normal_g: np.ndarray | None = None,
    slab_offset_g: float = 0.0,
    slab_half_thickness_g: float | None = None,
    max_polylines: int = 320,
) -> list[dict[str, Any]]:
    """Extract oriented 1D phase-vortex curves (Re Z = 0 intersect Im Z = 0) via 3D plaquette winding.

    Every punctured voxel face carries topological charge w in {-1, +1}; flux conservation
    (div w == 0 on every 3D voxel) chains face punctures into continuous directed 1D curves in
    reciprocal space G = UB * h (in A^-1).
    """
    abs_z = np.abs(Z)
    amp_ref = float(np.percentile(abs_z, 90.0)) if abs_z.size > 0 else 0.0
    if not np.isfinite(amp_ref) or amp_ref <= 1e-12:
        return []
    # Strictly real fields (e.g. centrosymmetric origin with real scattering lengths) have Im === 0
    # everywhere and codimension-1 sheets rather than isolated codimension-2 phase vortices.
    if float(np.max(np.abs(Z.imag))) <= 1e-4 * amp_ref:
        return []
    if float(np.max(np.abs(Z.real))) <= 1e-4 * amp_ref:
        return []

    def _dphi(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
        return np.angle(zb * np.conj(za))

    # Plaquette winding numbers on faces normal to +axis 0, +axis 1, +axis 2
    z0_00, z0_10, z0_11, z0_01 = (
        Z[:, :-1, :-1],
        Z[:, 1:, :-1],
        Z[:, 1:, 1:],
        Z[:, :-1, 1:],
    )
    w0 = (
        _dphi(z0_00, z0_10)
        + _dphi(z0_10, z0_11)
        + _dphi(z0_11, z0_01)
        + _dphi(z0_01, z0_00)
    ) / (2.0 * np.pi)
    max0 = np.maximum(
        np.maximum(np.abs(z0_00), np.abs(z0_10)),
        np.maximum(np.abs(z0_11), np.abs(z0_01)),
    )
    q0 = np.where(max0 > 1e-5 * amp_ref, np.rint(w0).astype(np.int8), np.int8(0))

    z1_00, z1_10, z1_11, z1_01 = (
        Z[:-1, :, :-1],
        Z[:-1, :, 1:],
        Z[1:, :, 1:],
        Z[1:, :, :-1],
    )
    w1 = (
        _dphi(z1_00, z1_10)
        + _dphi(z1_10, z1_11)
        + _dphi(z1_11, z1_01)
        + _dphi(z1_01, z1_00)
    ) / (2.0 * np.pi)
    max1 = np.maximum(
        np.maximum(np.abs(z1_00), np.abs(z1_10)),
        np.maximum(np.abs(z1_11), np.abs(z1_01)),
    )
    q1 = np.where(max1 > 1e-5 * amp_ref, np.rint(w1).astype(np.int8), np.int8(0))

    z2_00, z2_10, z2_11, z2_01 = (
        Z[:-1, :-1, :],
        Z[1:, :-1, :],
        Z[1:, 1:, :],
        Z[:-1, 1:, :],
    )
    w2 = (
        _dphi(z2_00, z2_10)
        + _dphi(z2_10, z2_11)
        + _dphi(z2_11, z2_01)
        + _dphi(z2_01, z2_00)
    ) / (2.0 * np.pi)
    max2 = np.maximum(
        np.maximum(np.abs(z2_00), np.abs(z2_10)),
        np.maximum(np.abs(z2_11), np.abs(z2_01)),
    )
    q2 = np.where(max2 > 1e-5 * amp_ref, np.rint(w2).astype(np.int8), np.int8(0))

    G0, G1, G2 = Z.shape
    dh0 = float(h0[1] - h0[0]) if G0 > 1 else 1.0
    dh1 = float(h1[1] - h1[0]) if G1 > 1 else 1.0
    dh2 = float(h2[1] - h2[0]) if G2 > 1 else 1.0

    # Build punctured face table: face_key -> (puncture_id, hkl_pos, charge_sign)
    puncture_hkl: list[np.ndarray] = []
    # Directed voxel graph: voxel (i,j,k) has incoming puncture_ids and outgoing puncture_ids
    voxel_in: dict[tuple[int, int, int], list[int]] = {}
    voxel_out: dict[tuple[int, int, int], list[int]] = {}
    next_voxel_of_puncture: dict[int, tuple[int, int, int] | None] = {}
    is_boundary_entry: set[int] = set()

    def _register_punctures(
        q_arr: np.ndarray,
        axis: int,
        z_a: np.ndarray,
        z_b: np.ndarray,
        z_c: np.ndarray,
        z_d: np.ndarray,
    ) -> None:
        ii, jj, kk = np.nonzero(q_arr)
        if len(ii) == 0:
            return
        u_sub, v_sub = _refine_bilinear_zero(
            z_a[ii, jj, kk], z_b[ii, jj, kk], z_c[ii, jj, kk], z_d[ii, jj, kk]
        )
        charges = q_arr[ii, jj, kk]
        for idx_p in range(len(ii)):
            i_v = int(ii[idx_p])
            j_v = int(jj[idx_p])
            k_v = int(kk[idx_p])
            ch = int(charges[idx_p])
            u_f = float(u_sub[idx_p])
            v_f = float(v_sub[idx_p])
            if axis == 0:
                pt_h = np.array(
                    [h0[i_v], h1[j_v] + u_f * dh1, h2[k_v] + v_f * dh2],
                    dtype=np.float64,
                )
                vox_neg = (i_v - 1, j_v, k_v) if i_v > 0 else None
                vox_pos = (i_v, j_v, k_v) if i_v < G0 - 1 else None
            elif axis == 1:
                pt_h = np.array(
                    [h0[i_v] + v_f * dh0, h1[j_v], h2[k_v] + u_f * dh2],
                    dtype=np.float64,
                )
                vox_neg = (i_v, j_v - 1, k_v) if j_v > 0 else None
                vox_pos = (i_v, j_v, k_v) if j_v < G1 - 1 else None
            else:
                pt_h = np.array(
                    [h0[i_v] + u_f * dh0, h1[j_v] + v_f * dh1, h2[k_v]],
                    dtype=np.float64,
                )
                vox_neg = (i_v, j_v, k_v - 1) if k_v > 0 else None
                vox_pos = (i_v, j_v, k_v) if k_v < G2 - 1 else None

            n_mult = min(abs(ch), 2)
            sgn = 1 if ch > 0 else -1
            # Positive winding on +axis face means vortex line flows from vox_neg -> vox_pos
            src_vox, dst_vox = (vox_neg, vox_pos) if sgn > 0 else (vox_pos, vox_neg)
            for _ in range(n_mult):
                pid = len(puncture_hkl)
                puncture_hkl.append(pt_h)
                next_voxel_of_puncture[pid] = dst_vox
                if src_vox is not None:
                    voxel_out.setdefault(src_vox, []).append(pid)
                else:
                    if dst_vox is not None:
                        is_boundary_entry.add(pid)
                if dst_vox is not None:
                    voxel_in.setdefault(dst_vox, []).append(pid)

    _register_punctures(q0, 0, z0_00, z0_10, z0_11, z0_01)
    _register_punctures(q1, 1, z1_00, z1_10, z1_11, z1_01)
    _register_punctures(q2, 2, z2_00, z2_10, z2_11, z2_01)

    if not puncture_hkl:
        return []

    # Pair each incoming puncture in a voxel with its unique (or nearest) outgoing puncture
    next_puncture: dict[int, int] = {}
    all_voxels = set(voxel_in.keys()) | set(voxel_out.keys())
    for vox in all_voxels:
        ins = list(voxel_in.get(vox, ()))
        outs = list(voxel_out.get(vox, ()))
        if len(ins) == 1 and len(outs) == 1:
            next_puncture[ins[0]] = outs[0]
        elif ins and outs:
            rem_outs = list(outs)
            for p_in in ins:
                if not rem_outs:
                    break
                pt_in = puncture_hkl[p_in]
                best_j = min(
                    range(len(rem_outs)),
                    key=lambda j: float(
                        np.sum((puncture_hkl[rem_outs[j]] - pt_in) ** 2)
                    ),
                )
                next_puncture[p_in] = rem_outs.pop(best_j)

    visited: set[int] = set()
    raw_chains: list[tuple[list[int], bool]] = []

    # 1. Open curves entering through grid boundary (or orphan starts)
    start_candidates = sorted(is_boundary_entry) + [
        pid for pid in range(len(puncture_hkl)) if pid not in next_puncture.values()
    ]
    seen_starts: set[int] = set()
    for sp in start_candidates:
        if sp in visited or sp in seen_starts:
            continue
        seen_starts.add(sp)
        chain = [sp]
        visited.add(sp)
        cur = sp
        while cur in next_puncture:
            nxt = next_puncture[cur]
            if nxt in visited:
                break
            visited.add(nxt)
            chain.append(nxt)
            cur = nxt
        if len(chain) >= 2:
            raw_chains.append((chain, False))

    # 2. Closed interior vortex loops
    for sp in range(len(puncture_hkl)):
        if sp in visited:
            continue
        chain = [sp]
        visited.add(sp)
        cur = sp
        is_loop = False
        while cur in next_puncture:
            nxt = next_puncture[cur]
            if nxt == sp:
                is_loop = True
                break
            if nxt in visited:
                break
            visited.add(nxt)
            chain.append(nxt)
            cur = nxt
        if len(chain) >= 3:
            raw_chains.append((chain, is_loop))

    ub = np.asarray(ub_matrix, dtype=np.float64)
    polylines: list[dict[str, Any]] = []

    for chain, is_loop in raw_chains:
        pts_h = np.asarray([puncture_hkl[p] for p in chain], dtype=np.float64)
        # Gentle 3-point smoothing in hkl space to suppress lattice staircasing while preserving topology
        if len(pts_h) >= 4:
            sm = pts_h.copy()
            if is_loop:
                sm = (
                    0.2 * np.roll(pts_h, 1, axis=0)
                    + 0.6 * pts_h
                    + 0.2 * np.roll(pts_h, -1, axis=0)
                )
            else:
                sm[1:-1] = 0.2 * pts_h[:-2] + 0.6 * pts_h[1:-1] + 0.2 * pts_h[2:]
            pts_h = sm

        pts_g = pts_h @ ub.T  # (M, 3) in sample reciprocal frame (A^-1)
        if is_loop:
            pts_g = np.vstack([pts_g, pts_g[0:1]])
            pts_h = np.vstack([pts_h, pts_h[0:1]])

        # Clip polyline into valid segments inside resolution sphere & optional ROI / slab mask
        g_norm = np.linalg.norm(pts_g, axis=1)
        valid_pt = g_norm > 1e-5
        if g_max is not None and g_max > 0.0:
            valid_pt &= g_norm <= float(g_max) * 1.02
        if roi_center_g is not None and roi_radius_g is not None and roi_radius_g > 0.0:
            dist_c = np.linalg.norm(pts_g - roi_center_g[None, :], axis=1)
            valid_pt &= dist_c <= float(roi_radius_g) * 1.04
        if slab_normal_g is not None and slab_half_thickness_g is not None:
            slab_d = np.abs(pts_g @ slab_normal_g - float(slab_offset_g))
            valid_pt &= slab_d <= float(slab_half_thickness_g) * 1.05

        # Split contiguous True runs
        idx_true = np.flatnonzero(valid_pt)
        if len(idx_true) < 2:
            continue
        splits = np.where(np.diff(idx_true) > 1)[0] + 1
        runs = np.split(idx_true, splits)
        for run in runs:
            if len(run) < 2:
                continue
            seg_g = pts_g[run]
            seg_h = pts_h[run]
            arc_len = float(np.sum(np.linalg.norm(np.diff(seg_g, axis=0), axis=1)))
            if arc_len <= 1e-6:
                continue
            closed_seg = bool(is_loop and len(run) == len(pts_g))
            polylines.append(
                {
                    "pts_g": np.round(seg_g, 5).tolist(),
                    "pts_hkl": np.round(seg_h, 3).tolist(),
                    "closed": closed_seg,
                    "length_inv_a": round(arc_len, 5),
                    "winding": 1,
                }
            )

    # Keep longest, most prominent vortex curves if capped
    polylines.sort(key=lambda c: c["length_inv_a"], reverse=True)
    return polylines[:max_polylines]


def _pack_f32_b64(arr: np.ndarray) -> str:
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    ).decode("ascii")


def _pack_u32_b64(arr: np.ndarray) -> str:
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype=np.uint32).tobytes()
    ).decode("ascii")


def _extract_single_isosurface_mesh(
    scalar_vol: np.ndarray,
    amp_ref: float,
    h0: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    ub_matrix: np.ndarray,
    ub_inv_t: np.ndarray,
    *,
    g_max: float | None = None,
    roi_center_g: np.ndarray | None = None,
    roi_radius_g: float | None = None,
    slab_normal_g: np.ndarray | None = None,
    slab_offset_g: float = 0.0,
    slab_half_thickness_g: float | None = None,
) -> dict[str, Any]:
    """Run Marching Cubes at level=0 and transform vertices/normals into sample reciprocal space G (A^-1)."""
    from skimage.measure import marching_cubes

    vmin = float(np.min(scalar_vol))
    vmax = float(np.max(scalar_vol))
    tol = 1e-4 * max(amp_ref, 1e-12)
    if not (vmin < -tol and vmax > tol):
        return {
            "n_verts": 0,
            "n_tris": 0,
            "verts_b64": "",
            "norms_b64": "",
            "faces_b64": "",
        }

    dh0 = float(h0[1] - h0[0]) if len(h0) > 1 else 1.0
    dh1 = float(h1[1] - h1[0]) if len(h1) > 1 else 1.0
    dh2 = float(h2[1] - h2[0]) if len(h2) > 1 else 1.0

    verts_idx, faces, norms_hkl, _ = marching_cubes(
        scalar_vol,
        level=0.0,
        spacing=(dh0, dh1, dh2),
        allow_degenerate=False,
    )
    if len(faces) == 0 or len(verts_idx) == 0:
        return {
            "n_verts": 0,
            "n_tris": 0,
            "verts_b64": "",
            "norms_b64": "",
            "faces_b64": "",
        }

    origin_h = np.array([h0[0], h1[0], h2[0]], dtype=np.float64)
    verts_hkl = verts_idx + origin_h[None, :]
    verts_g = verts_hkl @ ub_matrix.T
    norms_g = norms_hkl @ ub_inv_t.T
    n_len = np.linalg.norm(norms_g, axis=1, keepdims=True)
    norms_g = norms_g / np.maximum(n_len, 1e-12)

    # Mask triangles by centroid inside resolution sphere and active ROI sphere / Laue slab
    tri_centroids = (
        verts_g[faces[:, 0]] + verts_g[faces[:, 1]] + verts_g[faces[:, 2]]
    ) / 3.0
    keep_tri = np.ones(len(faces), dtype=bool)
    if g_max is not None and g_max > 0.0:
        keep_tri &= np.linalg.norm(tri_centroids, axis=1) <= float(g_max)
    if roi_center_g is not None and roi_radius_g is not None and roi_radius_g > 0.0:
        keep_tri &= np.linalg.norm(
            tri_centroids - roi_center_g[None, :], axis=1
        ) <= float(roi_radius_g)
    if slab_normal_g is not None and slab_half_thickness_g is not None:
        keep_tri &= np.abs(
            tri_centroids @ slab_normal_g - float(slab_offset_g)
        ) <= float(slab_half_thickness_g)

    faces_kept = faces[keep_tri]
    if len(faces_kept) == 0:
        return {
            "n_verts": 0,
            "n_tris": 0,
            "verts_b64": "",
            "norms_b64": "",
            "faces_b64": "",
        }

    # Compact vertex array to referenced vertices only
    used_v, remap = np.unique(faces_kept.ravel(), return_inverse=True)
    verts_compact = verts_g[used_v]
    norms_compact = norms_g[used_v]
    faces_compact = remap.reshape((-1, 3))

    return {
        "n_verts": len(verts_compact),
        "n_tris": len(faces_compact),
        "verts_b64": _pack_f32_b64(verts_compact),
        "norms_b64": _pack_f32_b64(norms_compact),
        "faces_b64": _pack_u32_b64(faces_compact),
    }


def compute_nodal_surfaces_payload(
    mtz_path: str | os.PathLike[str],
    ub_matrix: np.ndarray | list[list[float]],
    *,
    d_min: float = 1.5,
    space_group_override: str | None = None,
    ref_mtz_path: str | os.PathLike[str] | None = None,
    heterodyne_mode: str = "delta",
    roi_mode: str = "origin",
    roi_center_hkl: list[float] | tuple[float, float, float] | None = None,
    roi_zone_uvw: list[int] | tuple[int, int, int] | None = None,
    roi_zone_order: int = 0,
    roi_radius_mult: float = 2.5,
    roi_grid_size: int = 46,
    vortex_shell_mult: float = 3.6,
    vortex_grid_size: int = 48,
) -> dict[str, Any]:
    """Compute scoped 3D Re=0 & Im=0 nodal meshes and 1D phase-vortex curves in sample reciprocal space.

    Parameters
    ----------
    mtz_path : path to primary phased .mtz (e.g. simulated structure factors F, PHIf).
    ub_matrix : 3x3 active UB matrix mapping [h, k, l]^T -> G_sample (in A^-1, |G| = 1/d).
    ref_mtz_path : optional heterodyne reference F_0 .mtz with phases.
    heterodyne_mode : "delta" [F0* . (F - F0)], "product" [F0* . F], or "single" [F].
    roi_mode : "origin" | "peak" | "zone".
    roi_radius_mult : ROI radius in units of mean reciprocal basis length mean(|a*|, |b*|, |c*|).
    """
    ub = np.asarray(ub_matrix, dtype=np.float64)
    if ub.shape != (3, 3):
        raise ValueError("ub_matrix must be a 3x3 matrix.")
    ub_inv = np.linalg.inv(ub)
    ub_inv_t = ub_inv.T

    d_min_eff = max(0.5, float(d_min or 1.5))
    g_max = 1.0 / d_min_eff

    # Reciprocal basis lengths |a*|, |b*|, |c*| from columns of UB
    astar_norms = np.linalg.norm(ub, axis=0)
    mean_g_star = float(np.mean(astar_norms))
    if not np.isfinite(mean_g_star) or mean_g_star <= 1e-6:
        mean_g_star = 0.05

    primary = load_phased_structure_factor_field(
        mtz_path,
        space_group_override=space_group_override,
        d_min_floor=min(d_min_eff, 1.5),
    )
    if not primary.get("has_phases"):
        return {
            "has_phases": False,
            "primary_name": primary.get("name"),
            "f_column": primary.get("f_column"),
            "phi_column": None,
            "ref_name": None,
            "ref_has_phases": False,
            "heterodyne_active": False,
            "heterodyne_mode": "single",
        }

    ref_field: dict[str, Any] | None = None
    ref_warning: str | None = None
    if ref_mtz_path and os.path.isfile(str(ref_mtz_path)):
        ref_loaded = load_phased_structure_factor_field(
            ref_mtz_path,
            space_group_override=space_group_override,
            d_min_floor=min(d_min_eff, 1.5),
        )
        if ref_loaded.get("has_phases"):
            ref_field = ref_loaded
        else:
            ref_warning = (
                f"Reference MTZ '{ref_loaded.get('name')}' has no phase column (type 'P'); "
                "showing single-MTZ nodal surfaces."
            )

    het_active = bool(ref_field is not None and heterodyne_mode != "single")
    eff_het_mode = str(heterodyne_mode or "delta").lower() if het_active else "single"
    if eff_het_mode not in ("single", "delta", "product"):
        eff_het_mode = "delta" if het_active else "single"

    f_to_f0_scale = (
        compute_shared_scale_ratio(primary, ref_field)
        if (ref_field is not None and eff_het_mode == "delta")
        else 1.0
    )

    # Resolve ROI center & geometry in sample reciprocal space (A^-1)
    r_mult = float(np.clip(roi_radius_mult, 1.0, 8.0))
    roi_radius_g = min(g_max, r_mult * mean_g_star)

    mode_roi = str(roi_mode or "origin").lower()
    center_hkl = np.zeros(3, dtype=np.float64)
    slab_normal_g: np.ndarray | None = None
    slab_offset_g = 0.0
    slab_half_thickness_g: float | None = None

    if mode_roi == "peak" and roi_center_hkl is not None and len(roi_center_hkl) == 3:
        center_hkl = np.asarray(roi_center_hkl, dtype=np.float64)
    elif mode_roi == "zone" and roi_zone_uvw is not None and len(roi_zone_uvw) == 3:
        uvw = np.asarray(roi_zone_uvw, dtype=np.float64)
        # Direct-lattice zone axis in sample frame: t_sample = UB^(-T) * uvw
        t_sample = ub_inv_t @ uvw
        t_norm = float(np.linalg.norm(t_sample))
        if t_norm > 1e-8:
            slab_normal_g = t_sample / t_norm
            slab_offset_g = float(roi_zone_order) / t_norm
            slab_half_thickness_g = max(0.55 * mean_g_star, 0.18 * roi_radius_g)
            roi_radius_g = min(g_max, max(roi_radius_g, 1.45 * r_mult * mean_g_star))
            # Center ROI on the closest point of the HOLZ plane G . n = slab_offset_g to the origin
            center_g_zone = slab_offset_g * slab_normal_g
            center_hkl = ub_inv @ center_g_zone
        else:
            mode_roi = "origin"
    else:
        mode_roi = "origin"

    roi_center_g = ub @ center_hkl

    # Bounding box in fractional Miller space covering the Cartesian sphere |G - roi_center_g| <= roi_radius_g:
    # For each Miller component d, max |delta h_d| = roi_radius_g * ||row_d(UB^-1)||_2
    ub_inv_row_norms = np.linalg.norm(ub_inv, axis=1)
    half_span_hkl = np.maximum(roi_radius_g * ub_inv_row_norms * 1.04, 0.75)

    # Keep Miller step <= 0.22 so Re=0 / Im=0 oscillations on scale Delta h ~ 1 are well-resolved,
    # while capping grid dimension at 64 for sub-100ms response.
    max_span_h = float(np.max(2.0 * half_span_hkl))
    n_grid_roi = int(
        np.clip(max(int(roi_grid_size), math.ceil(max_span_h / 0.22)), 32, 64)
    )

    h0_roi = np.linspace(
        center_hkl[0] - half_span_hkl[0],
        center_hkl[0] + half_span_hkl[0],
        n_grid_roi,
    )
    h1_roi = np.linspace(
        center_hkl[1] - half_span_hkl[1],
        center_hkl[1] + half_span_hkl[1],
        n_grid_roi,
    )
    h2_roi = np.linspace(
        center_hkl[2] - half_span_hkl[2],
        center_hkl[2] + half_span_hkl[2],
        n_grid_roi,
    )

    Z_roi = evaluate_combined_complex_grid(
        primary,
        h0_roi,
        h1_roi,
        h2_roi,
        ref_field=ref_field,
        heterodyne_mode=eff_het_mode,
        f_to_f0_scale=f_to_f0_scale,
    )
    amp_roi = float(np.percentile(np.abs(Z_roi), 90.0)) if Z_roi.size > 0 else 0.0
    is_real_only = bool(float(np.max(np.abs(Z_roi.imag))) <= 1e-4 * max(amp_roi, 1e-12))

    re_mesh = _extract_single_isosurface_mesh(
        Z_roi.real,
        amp_roi,
        h0_roi,
        h1_roi,
        h2_roi,
        ub,
        ub_inv_t,
        g_max=g_max,
        roi_center_g=roi_center_g,
        roi_radius_g=roi_radius_g,
        slab_normal_g=slab_normal_g,
        slab_offset_g=slab_offset_g,
        slab_half_thickness_g=slab_half_thickness_g,
    )
    im_mesh = _extract_single_isosurface_mesh(
        Z_roi.imag,
        amp_roi,
        h0_roi,
        h1_roi,
        h2_roi,
        ub,
        ub_inv_t,
        g_max=g_max,
        roi_center_g=roi_center_g,
        roi_radius_g=roi_radius_g,
        slab_normal_g=slab_normal_g,
        slab_offset_g=slab_offset_g,
        slab_half_thickness_g=slab_half_thickness_g,
    )

    # Extract fine ROI 1D vortices (matching the 3D ROI surfaces)
    roi_vortices = extract_vortex_polylines(
        Z_roi,
        h0_roi,
        h1_roi,
        h2_roi,
        ub,
        g_max=g_max,
        roi_center_g=roi_center_g,
        roi_radius_g=roi_radius_g,
        slab_normal_g=slab_normal_g,
        slab_offset_g=slab_offset_g,
        slab_half_thickness_g=slab_half_thickness_g,
        max_polylines=200,
    )

    # Also extract broader crystal-shell 1D vortices around origin for Wulff & Detector projections
    shell_radius_g = min(
        g_max, max(roi_radius_g, float(vortex_shell_mult) * mean_g_star)
    )
    if (
        mode_roi == "origin"
        and abs(shell_radius_g - roi_radius_g) <= 0.15 * mean_g_star
    ):
        global_vortices = roi_vortices
    else:
        v_cache_key = (
            primary.get("path"),
            ref_field.get("path") if ref_field is not None else None,
            eff_het_mode,
            round(float(shell_radius_g), 4),
            tuple(np.round(ub.ravel(), 5).tolist()),
        )
        cached_gv = _GLOBAL_VORTEX_CACHE.get(v_cache_key)
        if cached_gv is not None:
            _GLOBAL_VORTEX_CACHE.move_to_end(v_cache_key)
            global_vortices = cached_gv
        else:
            shell_half_hkl = np.maximum(shell_radius_g * ub_inv_row_norms * 1.02, 1.0)
            n_grid_v = int(
                np.clip(
                    max(
                        int(vortex_grid_size),
                        math.ceil(float(np.max(2.0 * shell_half_hkl)) / 0.34),
                    ),
                    34,
                    54,
                )
            )
            hv0 = np.linspace(-shell_half_hkl[0], shell_half_hkl[0], n_grid_v)
            hv1 = np.linspace(-shell_half_hkl[1], shell_half_hkl[1], n_grid_v)
            hv2 = np.linspace(-shell_half_hkl[2], shell_half_hkl[2], n_grid_v)
            Z_shell = evaluate_combined_complex_grid(
                primary,
                hv0,
                hv1,
                hv2,
                ref_field=ref_field,
                heterodyne_mode=eff_het_mode,
                f_to_f0_scale=f_to_f0_scale,
            )
            global_vortices = extract_vortex_polylines(
                Z_shell,
                hv0,
                hv1,
                hv2,
                ub,
                g_max=shell_radius_g,
                roi_center_g=np.zeros(3, dtype=np.float64),
                roi_radius_g=shell_radius_g,
                max_polylines=280,
            )
            _GLOBAL_VORTEX_CACHE[v_cache_key] = global_vortices
            while len(_GLOBAL_VORTEX_CACHE) > _MAX_FIELD_CACHE_ENTRIES:
                _GLOBAL_VORTEX_CACHE.popitem(last=False)

    formula_label = (
        "F₀*·(F − F₀)"
        if eff_het_mode == "delta"
        else ("F₀*·F" if eff_het_mode == "product" else "F(G)")
    )

    return {
        "has_phases": True,
        "is_real_only": is_real_only,
        "primary_name": primary.get("name"),
        "f_column": primary.get("f_column"),
        "phi_column": primary.get("phi_column"),
        "ref_name": ref_field.get("name") if ref_field is not None else None,
        "ref_f_column": ref_field.get("f_column") if ref_field is not None else None,
        "ref_phi_column": ref_field.get("phi_column")
        if ref_field is not None
        else None,
        "ref_has_phases": bool(ref_field is not None),
        "ref_warning": ref_warning,
        "heterodyne_active": het_active,
        "heterodyne_mode": eff_het_mode,
        "f_to_f0_scale": round(float(f_to_f0_scale), 5),
        "formula_label": formula_label,
        "mean_g_star": round(mean_g_star, 6),
        "g_max": round(g_max, 5),
        "roi": {
            "mode": mode_roi,
            "center_hkl": [round(float(x), 3) for x in center_hkl],
            "center_g": [round(float(x), 5) for x in roi_center_g],
            "radius_mult": round(r_mult, 2),
            "radius_g": round(float(roi_radius_g), 5),
            "zone_uvw": list(roi_zone_uvw) if roi_zone_uvw is not None else None,
            "zone_order": int(roi_zone_order),
            "slab_normal_g": (
                [round(float(x), 5) for x in slab_normal_g]
                if slab_normal_g is not None
                else None
            ),
            "slab_offset_g": round(float(slab_offset_g), 5),
            "slab_half_thickness_g": (
                round(float(slab_half_thickness_g), 5)
                if slab_half_thickness_g is not None
                else None
            ),
            "grid_size": n_grid_roi,
        },
        "re_surface": re_mesh,
        "im_surface": im_mesh,
        "roi_vortices": roi_vortices,
        "vortices": global_vortices,
        "vortex_shell_radius_g": round(float(shell_radius_g), 5),
    }


def process_nodal_surfaces_api_request(
    req_json: dict[str, Any],
    *,
    mtz_path: str | None,
    ref_mtz_path: str | None,
    heterodyne_mode: str = "delta",
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, str | None, str]:
    """Process a `POST /api/nodal_surfaces` request and return `(response_doc, status_code, next_ref_path, next_het_mode)`."""
    next_ref = ref_mtz_path
    if req_json.get("clear_ref_mtz"):
        next_ref = None
    elif req_json.get("ref_mtz_path"):
        cand_ref = str(req_json["ref_mtz_path"]).strip()
        if not os.path.isfile(cand_ref):
            return (
                {"ok": False, "error": f"Reference .mtz file not found: {cand_ref}"},
                400,
                next_ref,
                heterodyne_mode,
            )
        next_ref = cand_ref

    next_het = str(req_json.get("heterodyne_mode") or heterodyne_mode or "delta")
    if next_het not in ("delta", "product", "single"):
        next_het = "delta"

    if not mtz_path or not os.path.isfile(mtz_path):
        return (
            {
                "ok": False,
                "error": "Load a phased primary .mtz structure factor file first before computing nodal surfaces.",
            },
            400,
            next_ref,
            next_het,
        )

    cur_p = payload or {}
    ub_raw = req_json.get("ub_matrix") or cur_p.get("ub_matrix")
    if ub_raw is None:
        return (
            {
                "ok": False,
                "error": "Active crystal orientation matrix UB is required to map nodal surfaces into reciprocal space.",
            },
            400,
            next_ref,
            next_het,
        )

    d_min_val = float(req_json.get("d_min") or cur_p.get("d_min") or 1.2)
    roi_mode = str(req_json.get("roi_mode") or "origin")
    roi_radius_mult = float(req_json.get("roi_radius_mult") or 2.5)
    roi_center_hkl = req_json.get("roi_center_hkl")
    roi_zone_uvw = req_json.get("roi_zone_uvw")
    roi_zone_order = int(req_json.get("roi_zone_order") or 0)

    nodal_data = compute_nodal_surfaces_payload(
        mtz_path=mtz_path,
        ub_matrix=ub_raw,
        d_min=d_min_val,
        ref_mtz_path=next_ref,
        heterodyne_mode=next_het,
        roi_mode=roi_mode,
        roi_center_hkl=roi_center_hkl,
        roi_zone_uvw=roi_zone_uvw,
        roi_zone_order=roi_zone_order,
        roi_radius_mult=roi_radius_mult,
    )
    if isinstance(payload, dict):
        payload["nodal_data"] = nodal_data
        payload["ref_mtz_overlay_name"] = (
            os.path.basename(next_ref) if next_ref else None
        )
    if not nodal_data.get("has_phases"):
        return (
            {
                "ok": False,
                "error": nodal_data.get("reason")
                or "Primary .mtz has no phase column (type 'P').",
                "nodal_data": nodal_data,
            },
            400,
            next_ref,
            next_het,
        )
    return (
        {"ok": True, "nodal_data": nodal_data},
        200,
        next_ref,
        next_het,
    )


def get_nodal_surfaces_js() -> str:
    """Return self-contained JS defining `createNodalSurfacesController(deps)` for 3D & Vortex rendering.

    Note: Returned as plain JS (not an f-string) so braces require no escaping when interpolated into
    `rainbow_zones.py` via `{NODAL_SURFACES_JS}`.
    """
    return r"""
  function createNodalSurfacesController(deps) {
    let nodalData = deps.initialNodalData || null;
    let dataSeq = nodalData ? 1 : 0;
    let decodedCacheSig = "";
    let decodedMeshes = { re: null, im: null };
    let glCanvas = null;
    let gl = null;
    let glProg = null;
    let glLineProg = null;
    let glBufs = { re: null, im: null };
    let glUnavailable = false;
    let last3DRenderSig = "";
    let screenPeakHits3D = [];
    let fetchInFlight = 0;
    let roiDebounceTimer = null;

    function decodeBase64Float32(b64) {
      if (!b64 || typeof b64 !== "string") return new Float32Array(0);
      const bin = atob(b64);
      const len = bin.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
      return new Float32Array(bytes.buffer);
    }

    function decodeBase64Uint32(b64) {
      if (!b64 || typeof b64 !== "string") return new Uint32Array(0);
      const bin = atob(b64);
      const len = bin.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
      return new Uint32Array(bytes.buffer);
    }

    function decodeSingleMesh(m) {
      if (!m || !m.n_tris || !m.verts_b64 || !m.faces_b64) {
        return { nVerts: 0, nTris: 0, verts: new Float32Array(0), norms: new Float32Array(0), faces: new Uint32Array(0) };
      }
      const verts = decodeBase64Float32(m.verts_b64);
      const norms = m.norms_b64 ? decodeBase64Float32(m.norms_b64) : new Float32Array(verts.length);
      const faces = decodeBase64Uint32(m.faces_b64);
      return {
        nVerts: Number(m.n_verts) || (verts.length / 3),
        nTris: Number(m.n_tris) || (faces.length / 3),
        verts,
        norms,
        faces,
      };
    }

    function ensureDecodedMeshes() {
      if (!nodalData || !nodalData.has_phases) {
        decodedMeshes = { re: null, im: null };
        decodedCacheSig = "";
        return decodedMeshes;
      }
      const reM = nodalData.re_surface || {};
      const imM = nodalData.im_surface || {};
      const sig = [
        nodalData.primary_name || "",
        nodalData.ref_name || "",
        nodalData.heterodyne_mode || "",
        reM.n_tris || 0,
        (reM.verts_b64 || "").slice(0, 24),
        imM.n_tris || 0,
        (imM.verts_b64 || "").slice(0, 24),
      ].join("|");
      if (sig === decodedCacheSig) return decodedMeshes;
      decodedCacheSig = sig;
      decodedMeshes = {
        re: decodeSingleMesh(reM),
        im: decodeSingleMesh(imM),
      };
      uploadGlMeshes();
      last3DRenderSig = "";
      return decodedMeshes;
    }

    function initWebGLOverlay() {
      if (glUnavailable) return null;
      if (gl && glProg) return gl;
      try {
        if (typeof document === "undefined" || !document.createElement) {
          glUnavailable = true;
          return null;
        }
        glCanvas = document.createElement("canvas");
        gl = glCanvas.getContext("webgl2", { alpha: true, antialias: true, premultipliedAlpha: false })
          || glCanvas.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: false });
        if (!gl) {
          glUnavailable = true;
          return null;
        }
        gl.getExtension("OES_element_index_uint");

        const vsSrc = [
          "attribute vec3 aPos;",
          "attribute vec3 aNorm;",
          "uniform mat3 uRot;",
          "uniform vec3 uCenter;",
          "uniform float uInvScale;",
          "uniform float uAspectX;",
          "uniform float uAspectY;",
          "uniform float uCamDist;",
          "varying vec3 vNorm;",
          "varying vec3 vViewPos;",
          "void main() {",
          "  vec3 p = uRot * ((aPos - uCenter) * uInvScale);",
          "  vec3 n = normalize(uRot * aNorm);",
          "  float dist = max(0.35, uCamDist - p.z);",
          "  float kD = sqrt(uCamDist * uCamDist - 1.0);",
          "  float persp = kD / dist;",
          "  float clipX = p.x * persp * uAspectX;",
          "  float clipY = p.y * persp * uAspectY;",
          "  float clipZ = clamp(-p.z * 0.45, -0.98, 0.98);",
          "  vNorm = n;",
          "  vViewPos = p;",
          "  gl_Position = vec4(clipX, clipY, clipZ, 1.0);",
          "}",
        ].join("\n");

        const fsSrc = [
          "precision mediump float;",
          "uniform vec4 uColor;",
          "varying vec3 vNorm;",
          "varying vec3 vViewPos;",
          "void main() {",
          "  vec3 N = normalize(vNorm);",
          "  if (!gl_FrontFacing) N = -N;",
          "  vec3 L1 = normalize(vec3(-0.38, 0.52, 0.76));",
          "  vec3 L2 = normalize(vec3(0.45, -0.35, 0.65));",
          "  float diff = 0.34 + 0.50 * max(dot(N, L1), 0.0) + 0.20 * max(dot(N, L2), 0.0);",
          "  vec3 V = vec3(0.0, 0.0, 1.0);",
          "  vec3 H = normalize(L1 + V);",
          "  float spec = 0.22 * pow(max(dot(N, H), 0.0), 22.0);",
          "  float rim = 0.16 * pow(1.0 - abs(dot(N, V)), 2.0);",
          "  vec3 rgb = clamp(uColor.rgb * diff + vec3(spec + rim * 0.5), 0.0, 1.0);",
          "  gl_FragColor = vec4(rgb, uColor.a);",
          "}",
        ].join("\n");

        function compileShader(type, src) {
          const sh = gl.createShader(type);
          gl.shaderSource(sh, src);
          gl.compileShader(sh);
          if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            gl.deleteShader(sh);
            return null;
          }
          return sh;
        }
        const vs = compileShader(gl.VERTEX_SHADER, vsSrc);
        const fs = compileShader(gl.FRAGMENT_SHADER, fsSrc);
        if (!vs || !fs) {
          glUnavailable = true;
          gl = null;
          return null;
        }
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
          glUnavailable = true;
          gl = null;
          return null;
        }
        glProg = {
          prog,
          aPos: gl.getAttribLocation(prog, "aPos"),
          aNorm: gl.getAttribLocation(prog, "aNorm"),
          uRot: gl.getUniformLocation(prog, "uRot"),
          uCenter: gl.getUniformLocation(prog, "uCenter"),
          uInvScale: gl.getUniformLocation(prog, "uInvScale"),
          uAspectX: gl.getUniformLocation(prog, "uAspectX"),
          uAspectY: gl.getUniformLocation(prog, "uAspectY"),
          uCamDist: gl.getUniformLocation(prog, "uCamDist"),
          uColor: gl.getUniformLocation(prog, "uColor"),
        };
        uploadGlMeshes();
        return gl;
      } catch (_err) {
        glUnavailable = true;
        gl = null;
        return null;
      }
    }

    function uploadGlSingleMesh(m) {
      if (!gl || !m || !m.nTris) return null;
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, m.verts, gl.STATIC_DRAW);
      const nbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, nbo);
      gl.bufferData(gl.ARRAY_BUFFER, m.norms, gl.STATIC_DRAW);
      const ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.faces, gl.STATIC_DRAW);
      return { vbo, nbo, ibo, count: m.faces.length, indexType: gl.UNSIGNED_INT };
    }

    function disposeGlMesh(b) {
      if (!gl || !b) return;
      if (b.vbo) gl.deleteBuffer(b.vbo);
      if (b.nbo) gl.deleteBuffer(b.nbo);
      if (b.ibo) gl.deleteBuffer(b.ibo);
    }

    function uploadGlMeshes() {
      if (!gl || !glProg) return;
      disposeGlMesh(glBufs.re);
      disposeGlMesh(glBufs.im);
      glBufs.re = decodedMeshes.re ? uploadGlSingleMesh(decodedMeshes.re) : null;
      glBufs.im = decodedMeshes.im ? uploadGlSingleMesh(decodedMeshes.im) : null;
    }

    // Camera matrix matching projectToWulff right-handed SO(3) lab camera coupled with R_gonio
    function buildCameraTransform(Rg) {
      const st = deps.state;
      const yawRad = (st.wulffYawDeg || 0.0) * Math.PI / 180.0;
      const pitchRad = (st.wulffPitchDeg || 0.0) * Math.PI / 180.0;
      const cy = Math.cos(yawRad), sy = Math.sin(yawRad);
      const cp = Math.cos(pitchRad), sp = Math.sin(pitchRad);
      // R_cam * vLab:
      // xCam = cy * vx - sy * vz
      // yCam = sp * sy * vx + cp * vy + sp * cy * vz
      // zCam = cp * sy * vx + sp * vy + cp * cy * vz  (note projectToWulff: y1=vy, z1=vx*sy+vz*cy; yCam=y1*cp-z1*sp; zCam=y1*sp+z1*cp)
      const Rc = [
        [cy, 0.0, -sy],
        [-sp * sy, cp, -sp * cy],
        [cp * sy, sp, cp * cy],
      ];
      const Rtot = deps.mat3Mul(Rc, Rg);
      return Rtot;
    }

    function get3DSceneBounds() {
      const st = deps.state;
      const invDmin = 1.0 / Math.max(0.4, Number(st.dMin) || 1.5);
      if (nodalData && nodalData.has_phases && nodalData.roi && st.nodalLockCameraToRoi) {
        const c = nodalData.roi.center_g || [0, 0, 0];
        const r = Math.max(0.01, Number(nodalData.roi.radius_g) || invDmin * 0.35);
        return { center: c, radius: r, invDmin };
      }
      if (nodalData && nodalData.has_phases && nodalData.roi && st.nodalRoiMode === "peak" && nodalData.roi.mode === "peak") {
        const c = nodalData.roi.center_g || [0, 0, 0];
        const r = Math.max(0.01, Number(nodalData.roi.radius_g) || invDmin * 0.35);
        return { center: c, radius: r * 1.15, invDmin };
      }
      // Default centered at origin, scaled comfortably so ROI + Bragg peaks are legible; zoomable up to 20x
      const defaultR = (nodalData && nodalData.has_phases && nodalData.roi && Number(nodalData.roi.radius_g) > 0)
        ? Math.min(invDmin, Math.max(Number(nodalData.roi.radius_g) * 1.35, invDmin * 0.32))
        : invDmin;
      return { center: [0.0, 0.0, 0.0], radius: Math.max(0.01, defaultR), invDmin };
    }

    function projectPoint3D(gSample, Rtot, center, invScale, cx, cy, Rpx, zoom) {
      const dx = (gSample[0] - center[0]) * invScale;
      const dy = (gSample[1] - center[1]) * invScale;
      const dz = (gSample[2] - center[2]) * invScale;
      const px = Rtot[0][0] * dx + Rtot[0][1] * dy + Rtot[0][2] * dz;
      const py = Rtot[1][0] * dx + Rtot[1][1] * dy + Rtot[1][2] * dz;
      const pz = Rtot[2][0] * dx + Rtot[2][1] * dy + Rtot[2][2] * dz;
      const D = 2.8;
      const kD = Math.sqrt(D * D - 1.0);
      const dist = Math.max(0.35, D - pz);
      const persp = kD / dist;
      const scale = Rpx * zoom;
      return [cx + px * persp * scale, cy - py * persp * scale, pz, (D - 0.2) / dist];
    }

    function renderSurfacesWebGLOrFallback(ctx, w, h, dpr, Rtot, center, invScale, Rpx, zoom) {
      const st = deps.state;
      if (!nodalData || !nodalData.has_phases) return;
      if (!st.showNodalSurfacesRe && !st.showNodalSurfacesIm) return;
      ensureDecodedMeshes();
      const hasRe = st.showNodalSurfacesRe && decodedMeshes.re && decodedMeshes.re.nTris > 0;
      const hasIm = st.showNodalSurfacesIm && decodedMeshes.im && decodedMeshes.im.nTris > 0;
      if (!hasRe && !hasIm) return;

      const surfAlpha = Math.max(0.12, Math.min(0.92, Number(st.nodalSurfaceAlpha) || 0.56));
      const glCtx = initWebGLOverlay();
      if (glCtx && glProg && (glBufs.re || glBufs.im)) {
        const tw = Math.max(1, Math.round(w * dpr));
        const th = Math.max(1, Math.round(h * dpr));
        if (glCanvas.width !== tw || glCanvas.height !== th) {
          glCanvas.width = tw;
          glCanvas.height = th;
        }
        glCtx.viewport(0, 0, tw, th);
        glCtx.clearColor(0.0, 0.0, 0.0, 0.0);
        glCtx.clear(glCtx.COLOR_BUFFER_BIT | glCtx.DEPTH_BUFFER_BIT);
        glCtx.enable(glCtx.DEPTH_TEST);
        glCtx.depthFunc(glCtx.LEQUAL);
        glCtx.enable(glCtx.BLEND);
        glCtx.blendFuncSeparate(
          glCtx.SRC_ALPHA, glCtx.ONE_MINUS_SRC_ALPHA,
          glCtx.ONE, glCtx.ONE_MINUS_SRC_ALPHA
        );

        glCtx.useProgram(glProg.prog);
        // Column-major 3x3 rotation matrix for GLSL
        const rotColMajor = new Float32Array([
          Rtot[0][0], Rtot[1][0], Rtot[2][0],
          Rtot[0][1], Rtot[1][1], Rtot[2][1],
          Rtot[0][2], Rtot[1][2], Rtot[2][2],
        ]);
        glCtx.uniformMatrix3fv(glProg.uRot, false, rotColMajor);
        glCtx.uniform3f(glProg.uCenter, center[0], center[1], center[2]);
        glCtx.uniform1f(glProg.uInvScale, invScale);
        const scalePx = Rpx * zoom;
        glCtx.uniform1f(glProg.uAspectX, scalePx / (0.5 * w));
        glCtx.uniform1f(glProg.uAspectY, scalePx / (0.5 * h));
        glCtx.uniform1f(glProg.uCamDist, 2.8);

        function drawPass(buf, rgba, cullFaceMode, depthMask) {
          if (!buf) return;
          glCtx.depthMask(depthMask);
          if (cullFaceMode) {
            glCtx.enable(glCtx.CULL_FACE);
            glCtx.cullFace(cullFaceMode);
          } else {
            glCtx.disable(glCtx.CULL_FACE);
          }
          glCtx.uniform4f(glProg.uColor, rgba[0], rgba[1], rgba[2], rgba[3]);
          glCtx.bindBuffer(glCtx.ARRAY_BUFFER, buf.vbo);
          glCtx.enableVertexAttribArray(glProg.aPos);
          glCtx.vertexAttribPointer(glProg.aPos, 3, glCtx.FLOAT, false, 0, 0);
          glCtx.bindBuffer(glCtx.ARRAY_BUFFER, buf.nbo);
          glCtx.enableVertexAttribArray(glProg.aNorm);
          glCtx.vertexAttribPointer(glProg.aNorm, 3, glCtx.FLOAT, false, 0, 0);
          glCtx.bindBuffer(glCtx.ELEMENT_ARRAY_BUFFER, buf.ibo);
          glCtx.drawElements(glCtx.TRIANGLES, buf.count, buf.indexType, 0);
        }

        // Cyan Re(Z)=0 and Coral-Rose Im(Z)=0:
        // Depth-prepass with subtle alpha + two-sided back-then-front blending for clean sheet intersections
        const colRe = [0.06, 0.78, 0.92, surfAlpha];
        const colIm = [0.96, 0.30, 0.48, surfAlpha];
        if (hasRe && glBufs.re) drawPass(glBufs.re, [colRe[0], colRe[1], colRe[2], surfAlpha * 0.78], glCtx.FRONT, true);
        if (hasIm && glBufs.im) drawPass(glBufs.im, [colIm[0], colIm[1], colIm[2], surfAlpha * 0.78], glCtx.FRONT, true);
        if (hasRe && glBufs.re) drawPass(glBufs.re, colRe, glCtx.BACK, false);
        if (hasIm && glBufs.im) drawPass(glBufs.im, colIm, glCtx.BACK, false);
        glCtx.depthMask(true);
        glCtx.disable(glCtx.CULL_FACE);

        ctx.save();
        ctx.drawImage(glCanvas, 0, 0, w, h);
        ctx.restore();
        return;
      }

      // Software Canvas2D depth-sorted painter's fallback (used in headless / non-WebGL environments)
      const tris = [];
      const cx = w * 0.5, cy = h * 0.5;
      function collectSoftwareTris(mesh, baseRgb) {
        if (!mesh || !mesh.nTris) return;
        const stride = Math.max(1, Math.ceil(mesh.nTris / 3500));
        const v = mesh.verts, n = mesh.norms, f = mesh.faces;
        for (let t = 0; t < mesh.nTris; t += stride) {
          const i0 = f[3 * t] * 3, i1 = f[3 * t + 1] * 3, i2 = f[3 * t + 2] * 3;
          const p0 = projectPoint3D([v[i0], v[i0 + 1], v[i0 + 2]], Rtot, center, invScale, cx, cy, Rpx, zoom);
          const p1 = projectPoint3D([v[i1], v[i1 + 1], v[i1 + 2]], Rtot, center, invScale, cx, cy, Rpx, zoom);
          const p2 = projectPoint3D([v[i2], v[i2 + 1], v[i2 + 2]], Rtot, center, invScale, cx, cy, Rpx, zoom);
          const nx = Rtot[0][0] * n[i0] + Rtot[0][1] * n[i0 + 1] + Rtot[0][2] * n[i0 + 2];
          const ny = Rtot[1][0] * n[i0] + Rtot[1][1] * n[i0 + 1] + Rtot[1][2] * n[i0 + 2];
          const nz = Rtot[2][0] * n[i0] + Rtot[2][1] * n[i0 + 1] + Rtot[2][2] * n[i0 + 2];
          const shade = 0.42 + 0.52 * Math.abs(-0.35 * nx + 0.45 * ny + 0.82 * nz);
          tris.push({
            z: (p0[2] + p1[2] + p2[2]) / 3.0,
            p0, p1, p2,
            fill: `rgba(${Math.round(baseRgb[0] * shade)},${Math.round(baseRgb[1] * shade)},${Math.round(baseRgb[2] * shade)},${surfAlpha.toFixed(2)})`,
          });
        }
      }
      if (hasRe) collectSoftwareTris(decodedMeshes.re, [16, 205, 238]);
      if (hasIm) collectSoftwareTris(decodedMeshes.im, [246, 78, 125]);
      tris.sort((a, b) => a.z - b.z);
      ctx.save();
      for (let i = 0; i < tris.length; i++) {
        const tr = tris[i];
        ctx.beginPath();
        ctx.moveTo(tr.p0[0], tr.p0[1]);
        ctx.lineTo(tr.p1[0], tr.p1[1]);
        ctx.lineTo(tr.p2[0], tr.p2[1]);
        ctx.closePath();
        ctx.fillStyle = tr.fill;
        ctx.fill();
      }
      ctx.restore();
    }

    function render3DReciprocalView(stereoCanvas, forceRedraw) {
      const st = deps.state;
      if (!st.showStereo || st.stereoViewMode !== "recip3d") return false;
      const rect = stereoCanvas.getBoundingClientRect();
      if (rect.width < 20 || rect.height < 20) return true;
      const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
      const targetW = Math.round(rect.width * dpr);
      const targetH = Math.round(rect.height * dpr);
      const Rg = deps.getActiveGoniometerR();
      const hovRef = st.hoveredPeakObj ? `${st.hoveredPeakObj.id}:${(st.hoveredPeakObj.hkl || []).join(",")}` : String(st.hoveredPeakId);
      const selRef = st.selectedPeakObj ? `${st.selectedPeakObj.id}:${(st.selectedPeakObj.hkl || []).join(",")}` : String(st.selectedPeakId);
      const zonesKey = st.showZoneArcs ? Array.from(st.activeZones || []).join(",") : "off";
      const sig = [
        targetW, targetH,
        (st.wulffYawDeg || 0).toFixed(3),
        (st.wulffPitchDeg || 0).toFixed(3),
        (st.wulffZoom || 1).toFixed(4),
        st.wulffExpanded ? 1 : 0,
        st.wulffScope,
        st.wulffColorMode,
        deps.getColorCrossover().toFixed(3),
        st.showWulffFriedel ? 1 : 0,
        st.showNodalPoints ? 1 : 0,
        st.showNodalSurfacesRe ? 1 : 0,
        st.showNodalSurfacesIm ? 1 : 0,
        st.showPhaseVortices ? 1 : 0,
        (st.nodalSurfaceAlpha || 0.56).toFixed(2),
        st.nodalRoiMode || "origin",
        st.dMin, st.lamMin, st.lamMax,
        zonesKey, hovRef, selRef,
        decodedCacheSig,
        nodalData && nodalData.roi ? `${nodalData.roi.radius_g}:${(nodalData.roi.center_g || []).join(",")}` : "noroi",
        Rg[0][0].toFixed(5), Rg[0][2].toFixed(5), Rg[1][1].toFixed(5), Rg[2][0].toFixed(5), Rg[2][2].toFixed(5),
      ].join("|");

      if (!forceRedraw && sig === last3DRenderSig && stereoCanvas.width === targetW && stereoCanvas.height === targetH) {
        return true;
      }
      last3DRenderSig = sig;
      if (stereoCanvas.width !== targetW || stereoCanvas.height !== targetH) {
        stereoCanvas.width = targetW;
        stereoCanvas.height = targetH;
      }

      const ctx = stereoCanvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      // Deep space reciprocal background
      const bg = ctx.createRadialGradient(w * 0.5, h * 0.5, 8, w * 0.5, h * 0.5, Math.max(w, h) * 0.68);
      bg.addColorStop(0, "rgba(13, 20, 40, 0.97)");
      bg.addColorStop(1, "rgba(5, 8, 18, 0.99)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      const cx = w * 0.5;
      const cy = h * 0.5;
      const Rpx = Math.min(w, h) * 0.41;
      const zoom = st.wulffZoom || 1.0;
      const isExp = Boolean(st.wulffExpanded);
      const Rtot = buildCameraTransform(Rg);
      const bounds = get3DSceneBounds();
      const center = bounds.center;
      const sceneR = bounds.radius;
      const invScale = 1.0 / sceneR;
      const ub = deps.getActiveUbMatrix();

      // 1. Wireframe ROI sphere (and outer d_min resolution sphere if visible)
      function drawWireSphere(sphCenter, sphRadius, strokeCol, dashArr) {
        ctx.save();
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = 1.0;
        if (dashArr) ctx.setLineDash(dashArr);
        const axes = [[1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]];
        const nSeg = 56;
        for (let a = 0; a < axes.length; a++) {
          const ax = axes[a];
          ctx.beginPath();
          for (let s = 0; s <= nSeg; s++) {
            const th = (2.0 * Math.PI * s) / nSeg;
            const ct = Math.cos(th), stt = Math.sin(th);
            const gp = [
              sphCenter[0] + sphRadius * (ct * ax[0] + stt * ax[3]),
              sphCenter[1] + sphRadius * (ct * ax[1] + stt * ax[4]),
              sphCenter[2] + sphRadius * (ct * ax[2] + stt * ax[5]),
            ];
            const pr = projectPoint3D(gp, Rtot, center, invScale, cx, cy, Rpx, zoom);
            if (s === 0) ctx.moveTo(pr[0], pr[1]);
            else ctx.lineTo(pr[0], pr[1]);
          }
          ctx.stroke();
        }
        ctx.restore();
      }

      if (nodalData && nodalData.has_phases && nodalData.roi) {
        drawWireSphere(nodalData.roi.center_g || [0, 0, 0], Number(nodalData.roi.radius_g) || sceneR, "rgba(56, 189, 248, 0.22)", [3, 4]);
      } else {
        drawWireSphere([0, 0, 0], sceneR, "rgba(56, 189, 248, 0.20)", [3, 4]);
      }

      // 2. Active Laue Zone reciprocal planes (nodal planes G . t_uvw = 0) & zone-intersection lines
      const actZones = deps.getActiveZoneObjects();
      if (st.showZoneArcs && ub && Array.isArray(actZones) && actZones.length > 0) {
        const invUb = deps.mat3Inv(ub);
        if (invUb) {
          const planeR = (nodalData && nodalData.roi && Number(nodalData.roi.radius_g) > 0)
            ? Number(nodalData.roi.radius_g)
            : sceneR * 0.92;
          const planeCenter = (nodalData && nodalData.roi && nodalData.roi.center_g) || [0, 0, 0];
          const zNormals = [];
          ctx.save();
          for (let zi = 0; zi < Math.min(actZones.length, 6); zi++) {
            const z = actZones[zi];
            if (!Array.isArray(z.uvw) || z.uvw.length !== 3) continue;
            const u = z.uvw[0], v = z.uvw[1], ww = z.uvw[2];
            // Direct lattice axis in sample frame: t = UB^(-T) * uvw
            const tx = invUb[0][0] * u + invUb[1][0] * v + invUb[2][0] * ww;
            const ty = invUb[0][1] * u + invUb[1][1] * v + invUb[2][1] * ww;
            const tz = invUb[0][2] * u + invUb[1][2] * v + invUb[2][2] * ww;
            const tn = Math.hypot(tx, ty, tz);
            if (tn < 1e-8) continue;
            const nHat = [tx / tn, ty / tn, tz / tn];
            zNormals.push({ nHat, color: z.rainbow_color || "#38bdf8", label: z.label });
            // Orthonormal basis (e1, e2) spanning the Laue reciprocal plane G . nHat = 0
            const ref = Math.abs(nHat[0]) < 0.85 ? [1, 0, 0] : [0, 1, 0];
            let e1x = nHat[1] * ref[2] - nHat[2] * ref[1];
            let e1y = nHat[2] * ref[0] - nHat[0] * ref[2];
            let e1z = nHat[0] * ref[1] - nHat[1] * ref[0];
            const e1n = Math.hypot(e1x, e1y, e1z) || 1.0;
            e1x /= e1n; e1y /= e1n; e1z /= e1n;
            const e2x = nHat[1] * e1z - nHat[2] * e1y;
            const e2y = nHat[2] * e1x - nHat[0] * e1z;
            const e2z = nHat[0] * e1y - nHat[1] * e1x;

            // Project plane disc clipped near planeCenter
            const dOff = planeCenter[0] * nHat[0] + planeCenter[1] * nHat[1] + planeCenter[2] * nHat[2];
            if (Math.abs(dOff) < planeR * 0.95) {
              const discR = Math.sqrt(Math.max(0.0, planeR * planeR - dOff * dOff));
              const cx0 = planeCenter[0] - dOff * nHat[0];
              const cy0 = planeCenter[1] - dOff * nHat[1];
              const cz0 = planeCenter[2] - dOff * nHat[2];
              ctx.beginPath();
              const nSeg = 48;
              for (let s = 0; s <= nSeg; s++) {
                const th = (2.0 * Math.PI * s) / nSeg;
                const ct = Math.cos(th), stt = Math.sin(th);
                const gp = [
                  cx0 + discR * (ct * e1x + stt * e2x),
                  cy0 + discR * (ct * e1y + stt * e2y),
                  cz0 + discR * (ct * e1z + stt * e2z),
                ];
                const pr = projectPoint3D(gp, Rtot, center, invScale, cx, cy, Rpx, zoom);
                if (s === 0) ctx.moveTo(pr[0], pr[1]);
                else ctx.lineTo(pr[0], pr[1]);
              }
              ctx.closePath();
              ctx.fillStyle = "rgba(56, 189, 248, 0.045)";
              ctx.fill();
              ctx.strokeStyle = z.rainbow_color || "#38bdf8";
              ctx.globalAlpha = 0.58;
              ctx.lineWidth = st.selectedZoneId === z.zone_id ? 2.0 : 1.15;
              ctx.stroke();
              ctx.globalAlpha = 1.0;
            }
          }

          // 1D nodal axes = pairwise intersections of active Laue planes
          if (st.showNodalPoints && zNormals.length >= 2) {
            ctx.setLineDash([4, 3]);
            for (let i = 0; i < zNormals.length; i++) {
              for (let j = i + 1; j < zNormals.length; j++) {
                const a = zNormals[i].nHat, b = zNormals[j].nHat;
                const lx = a[1] * b[2] - a[2] * b[1];
                const ly = a[2] * b[0] - a[0] * b[2];
                const lz = a[0] * b[1] - a[1] * b[0];
                const ln = Math.hypot(lx, ly, lz);
                if (ln < 1e-4) continue;
                const ux = (lx / ln) * planeR, uy = (ly / ln) * planeR, uz = (lz / ln) * planeR;
                const pA = projectPoint3D([-ux, -uy, -uz], Rtot, center, invScale, cx, cy, Rpx, zoom);
                const pB = projectPoint3D([ux, uy, uz], Rtot, center, invScale, cx, cy, Rpx, zoom);
                ctx.beginPath();
                ctx.moveTo(pA[0], pA[1]);
                ctx.lineTo(pB[0], pB[1]);
                ctx.strokeStyle = "rgba(251, 191, 36, 0.52)";
                ctx.lineWidth = 1.2;
                ctx.stroke();
              }
            }
            ctx.setLineDash([]);
          }
          ctx.restore();
        }
      }

      // 3. Two-color 2D Nodal Surfaces: Re(Z)=0 (Cyan) & Im(Z)=0 (Coral-Rose)
      renderSurfacesWebGLOrFallback(ctx, w, h, dpr, Rtot, center, invScale, Rpx, zoom);

      // 4. Reciprocal lattice basis axes {a*, b*, c*} through origin
      if (ub) {
        const axisCols = ["#f87171", "#4ade80", "#60a5fa"];
        const axisNames = ["a⋆", "b⋆", "c⋆"];
        const pOrig = projectPoint3D([0, 0, 0], Rtot, center, invScale, cx, cy, Rpx, zoom);
        ctx.save();
        for (let d = 0; d < 3; d++) {
          const axVec = [ub[0][d], ub[1][d], ub[2][d]];
          const axNorm = Math.hypot(axVec[0], axVec[1], axVec[2]) || 1.0;
          const lenG = Math.min(sceneR * 0.85, Math.max(axNorm * 2.0, sceneR * 0.35));
          const tip = [(axVec[0] / axNorm) * lenG, (axVec[1] / axNorm) * lenG, (axVec[2] / axNorm) * lenG];
          const pTip = projectPoint3D(tip, Rtot, center, invScale, cx, cy, Rpx, zoom);
          ctx.beginPath();
          ctx.moveTo(pOrig[0], pOrig[1]);
          ctx.lineTo(pTip[0], pTip[1]);
          ctx.strokeStyle = axisCols[d];
          ctx.lineWidth = 1.8;
          ctx.stroke();
          ctx.font = isExp ? "bold 10.5px monospace" : "bold 9px monospace";
          ctx.fillStyle = axisCols[d];
          ctx.fillText(axisNames[d], pTip[0] + 4, pTip[1] - 3);
        }
        ctx.restore();
      }

      // 5. 1D Phase-Vortex Curves (Re=0 intersect Im=0) in 3D Reciprocal Space
      if (nodalData && nodalData.has_phases && st.showPhaseVortices) {
        const vList = Array.isArray(nodalData.roi_vortices) && nodalData.roi_vortices.length > 0
          ? nodalData.roi_vortices
          : (nodalData.vortices || []);
        ctx.save();
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        for (let vi = 0; vi < vList.length; vi++) {
          const pts = vList[vi].pts_g;
          if (!Array.isArray(pts) || pts.length < 2) continue;
          ctx.beginPath();
          for (let pi = 0; pi < pts.length; pi++) {
            const pr = projectPoint3D(pts[pi], Rtot, center, invScale, cx, cy, Rpx, zoom);
            if (pi === 0) ctx.moveTo(pr[0], pr[1]);
            else ctx.lineTo(pr[0], pr[1]);
          }
          ctx.strokeStyle = "rgba(7, 9, 19, 0.88)";
          ctx.lineWidth = isExp ? 4.4 : 3.4;
          ctx.stroke();
          ctx.strokeStyle = "#fde047";
          ctx.lineWidth = isExp ? 2.3 : 1.7;
          ctx.stroke();
        }
        ctx.restore();
      }

      // 6. 3D Bragg Reflections at G = UB * hkl (depth-sorted, interactive hover/click!)
      screenPeakHits3D = [];
      if (ub) {
        const lodState = deps.getWulffVisiblePool(w, h);
        const peaks = lodState.visiblePeaks || [];
        const drawList = [];
        const maxClipR = sceneR * 1.45;
        for (let i = 0; i < peaks.length; i++) {
          const p = peaks[i];
          if (!Array.isArray(p.hkl) || p.hkl.length !== 3) continue;
          const gx = ub[0][0] * p.hkl[0] + ub[0][1] * p.hkl[1] + ub[0][2] * p.hkl[2];
          const gy = ub[1][0] * p.hkl[0] + ub[1][1] * p.hkl[1] + ub[1][2] * p.hkl[2];
          const gz = ub[2][0] * p.hkl[0] + ub[2][1] * p.hkl[1] + ub[2][2] * p.hkl[2];
          const distC = Math.hypot(gx - center[0], gy - center[1], gz - center[2]);
          const isFocus = deps.isSameReflection(p, st.selectedPeakObj) || deps.isSameReflection(p, st.hoveredPeakObj);
          if (distC > maxClipR && !isFocus) continue;
          const pr = projectPoint3D([gx, gy, gz], Rtot, center, invScale, cx, cy, Rpx, zoom);
          drawList.push({ p, x: pr[0], y: pr[1], z: pr[2], sc: pr[3], isFocus });
          if (st.showWulffFriedel) {
            const prF = projectPoint3D([-gx, -gy, -gz], Rtot, center, invScale, cx, cy, Rpx, zoom);
            const distCF = Math.hypot(-gx - center[0], -gy - center[1], -gz - center[2]);
            if (distCF <= maxClipR) {
              drawList.push({ p, x: prF[0], y: prF[1], z: prF[2], sc: prF[3], isFocus: false, isFriedel: true });
            }
          }
        }
        drawList.sort((a, b) => a.z - b.z);
        ctx.save();
        for (let i = 0; i < drawList.length; i++) {
          const it = drawList[i];
          if (it.x < -20 || it.x > w + 20 || it.y < -20 || it.y > h + 20) continue;
          const rBase = (isExp ? 3.5 : 2.7) * Math.max(0.65, Math.min(1.55, it.sc));
          const rad = it.isFocus ? rBase + 2.4 : (it.isFriedel ? rBase * 0.78 : rBase);
          const col = deps.getWulffPeakColor(it.p);
          ctx.beginPath();
          ctx.arc(it.x, it.y, rad, 0, Math.PI * 2);
          ctx.fillStyle = col;
          ctx.globalAlpha = it.isFocus ? 1.0 : (it.isFriedel ? 0.42 : 0.88);
          ctx.fill();
          ctx.lineWidth = it.isFocus ? 1.8 : 0.8;
          ctx.strokeStyle = it.isFocus ? "#ffffff" : "rgba(7, 9, 19, 0.85)";
          ctx.stroke();
          if (!it.isFriedel) {
            screenPeakHits3D.push({ peak: it.p, x: it.x, y: it.y, r: Math.max(8.0, rad + 4.0), z: it.z });
          }
          if (it.isFocus && Array.isArray(it.p.hkl)) {
            ctx.globalAlpha = 1.0;
            const lbl = `(${it.p.hkl.join(",")})`;
            ctx.font = isExp ? "bold 10px monospace" : "bold 8.5px monospace";
            ctx.fillStyle = "#070913";
            ctx.fillText(lbl, it.x + rad + 4, it.y - 3);
            ctx.fillStyle = "#fde047";
            ctx.fillText(lbl, it.x + rad + 3, it.y - 4);
          }
        }
        ctx.globalAlpha = 1.0;
        ctx.restore();
      }

      // 7. Legend & Status Pill Overlay in 3D View
      ctx.save();
      ctx.font = isExp ? "bold 9.5px monospace" : "bold 8px monospace";
      const pills = [];
      if (nodalData && nodalData.has_phases) {
        const fLbl = nodalData.formula_label || "F(G)";
        const nRe = (nodalData.re_surface && nodalData.re_surface.n_tris) || 0;
        const nIm = (nodalData.im_surface && nodalData.im_surface.n_tris) || 0;
        const nVor = Array.isArray(nodalData.roi_vortices) ? nodalData.roi_vortices.length : 0;
        if (st.showNodalSurfacesRe) pills.push({ txt: `Re[${fLbl}]=0 (${nRe}Δ)`, col: "#22d3ee" });
        if (st.showNodalSurfacesIm) {
          pills.push({
            txt: nodalData.is_real_only ? `Im[${fLbl}]≡0 (centric)` : `Im[${fLbl}]=0 (${nIm}Δ)`,
            col: "#fb7185",
          });
        }
        if (st.showPhaseVortices) pills.push({ txt: `1D Vortices (${nVor})`, col: "#fde047" });
      } else {
        pills.push({ txt: "3D Reciprocal Lattice (load phased .mtz for Re/Im surfaces & vortices)", col: "#94a3b8" });
      }
      let py = 8;
      const bh = isExp ? 16 : 13.5;
      for (let i = 0; i < pills.length; i++) {
        const pl = pills[i];
        const tw = ctx.measureText(pl.txt).width + 18;
        ctx.fillStyle = "rgba(7, 9, 19, 0.84)";
        ctx.strokeStyle = "rgba(148, 163, 184, 0.32)";
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") ctx.roundRect(8, py, tw, bh, 4);
        else ctx.rect(8, py, tw, bh);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = pl.col;
        ctx.beginPath();
        ctx.arc(15, py + bh * 0.5, 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(pl.txt, 22, py + (isExp ? 11.5 : 9.8));
        py += bh + 3;
      }
      ctx.restore();
      return true;
    }

    function findNearestPeak3D(mx, my) {
      let best = null;
      let bestScore = 14.0;
      for (let i = screenPeakHits3D.length - 1; i >= 0; i--) {
        const h = screenPeakHits3D[i];
        const d = Math.hypot(mx - h.x, my - h.y);
        if (d <= h.r && d < bestScore) {
          bestScore = d;
          best = h.peak;
        }
      }
      return best;
    }

    // Project 1D phase vortices onto the Convex 3D Wulff Sphere (radial unit-vector projection G / |G|)
    function renderWulffVortices(ctx, cx, cy, R, isExp) {
      const st = deps.state;
      if (!nodalData || !nodalData.has_phases || !st.showPhaseVortices) return;
      const vList = nodalData.vortices || nodalData.roi_vortices || [];
      if (!Array.isArray(vList) || vList.length === 0) return;

      const Rg = deps.getActiveGoniometerR();
      const ki = deps.getActiveIncidentBeam();
      const lMin = Number(st.lamMin) || 0.0;
      const lMax = Number(st.lamMax) || 99.0;
      const signs = st.showWulffFriedel ? [1.0, -1.0] : [1.0];

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let si = 0; si < signs.length; si++) {
        const sgn = signs[si];
        const isMate = sgn < 0;
        for (let vi = 0; vi < vList.length; vi++) {
          const pts = vList[vi].pts_g;
          if (!Array.isArray(pts) || pts.length < 2) continue;

          // Draw front-hemisphere segments brightly, back-hemisphere softly dashed
          for (let pass = 0; pass < 2; pass++) {
            const wantHemi = pass === 0 ? -1 : 1;
            ctx.beginPath();
            let penDown = false;
            for (let pi = 0; pi < pts.length; pi++) {
              const g = [sgn * pts[pi][0], sgn * pts[pi][1], sgn * pts[pi][2]];
              const gn = Math.hypot(g[0], g[1], g[2]);
              if (!(gn > 1e-6)) { penDown = false; continue; }
              // Respect d_min slider
              if (1.0 / gn < (Number(st.dMin) || 0.5) * 0.98) { penDown = false; continue; }
              // In 1-Frame scope, optionally gate by elastic wavelength bandpass [lamMin, lamMax]
              if (st.wulffScope !== "all_runs") {
                const gLab = deps.mat3Vec(Rg, g);
                const sinTh = -(ki[0] * gLab[0] + ki[1] * gLab[1] + ki[2] * gLab[2]) / gn;
                if (!(sinTh > 1e-4)) { penDown = false; continue; }
                const lam = (2.0 * sinTh) / gn;
                if (lam < lMin || lam > lMax) { penDown = false; continue; }
              }
              const pr = deps.projectToWulff(g, cx, cy, R);
              if (pr[2] !== wantHemi) { penDown = false; continue; }
              if (!penDown) {
                ctx.moveTo(pr[0], pr[1]);
                penDown = true;
              } else {
                ctx.lineTo(pr[0], pr[1]);
              }
            }
            if (wantHemi === -1) {
              ctx.setLineDash([2, 3]);
              ctx.strokeStyle = isMate ? "rgba(232, 121, 249, 0.25)" : "rgba(253, 224, 71, 0.28)";
              ctx.lineWidth = isExp ? 1.2 : 0.9;
              ctx.stroke();
              ctx.setLineDash([]);
            } else {
              ctx.strokeStyle = "rgba(7, 9, 19, 0.85)";
              ctx.lineWidth = isExp ? 3.8 : 2.8;
              ctx.stroke();
              ctx.strokeStyle = isMate ? "#e879f9" : "#fde047";
              ctx.lineWidth = isExp ? 2.0 : 1.45;
              ctx.stroke();
            }
          }
        }
      }
      ctx.restore();
    }

    // Project 1D phase vortices onto the 2D Detector Panels via exact Laue Bragg raycasting
    function renderDetectorVortices(ctx, w, h) {
      const st = deps.state;
      if (!nodalData || !nodalData.has_phases || !st.showPhaseVortices) return;
      const vList = nodalData.vortices || nodalData.roi_vortices || [];
      if (!Array.isArray(vList) || vList.length === 0) return;

      const Rg = deps.getActiveGoniometerR();
      const ki = deps.getActiveIncidentBeam();
      const sLab = deps.getActiveSampleOffsetLab();
      const fullPanels = deps.buildFullFaceDetectorRaycastPanels(sLab);
      const isSph = deps.isSphericalDetectorView();
      const lMin = Number(st.lamMin) || 0.0;
      const lMax = Number(st.lamMax) || 99.0;
      const dMinCut = (Number(st.dMin) || 0.5) * 0.98;
      const azShifts = [-360.0, 0.0, 360.0];

      // Both +G(s) and its Friedel mate -G(s) carry the same simultaneous zero |Z(-G)| = |Z(G)| = 0!
      const signs = [1.0, -1.0];
      const projectedSegs = [];

      for (let si = 0; si < signs.length; si++) {
        const sgn = signs[si];
        for (let vi = 0; vi < vList.length; vi++) {
          const pts = vList[vi].pts_g;
          if (!Array.isArray(pts) || pts.length < 2) continue;
          let curSeg = [];
          let prevAz = null;
          let prevBank = null;
          for (let pi = 0; pi < pts.length; pi++) {
            const gx = sgn * pts[pi][0], gy = sgn * pts[pi][1], gz = sgn * pts[pi][2];
            const gn = Math.hypot(gx, gy, gz);
            if (!(gn > 1e-6) || (1.0 / gn) < dMinCut) {
              if (curSeg.length >= 2) projectedSegs.push({ pts: curSeg, isMate: sgn < 0 });
              curSeg = []; prevAz = null; prevBank = null;
              continue;
            }
            const gLab = deps.mat3Vec(Rg, [gx, gy, gz]);
            const sinTh = -(ki[0] * gLab[0] + ki[1] * gLab[1] + ki[2] * gLab[2]) / gn;
            if (!(sinTh > 1e-4 && sinTh < 0.9999)) {
              if (curSeg.length >= 2) projectedSegs.push({ pts: curSeg, isMate: sgn < 0 });
              curSeg = []; prevAz = null; prevBank = null;
              continue;
            }
            const lam = (2.0 * sinTh) / gn;
            if (lam < lMin || lam > lMax) {
              if (curSeg.length >= 2) projectedSegs.push({ pts: curSeg, isMate: sgn < 0 });
              curSeg = []; prevAz = null; prevBank = null;
              continue;
            }
            const invGn = 1.0 / gn;
            const kfx = ki[0] + 2.0 * sinTh * gLab[0] * invGn;
            const kfy = ki[1] + 2.0 * sinTh * gLab[1] * invGn;
            const kfz = ki[2] + 2.0 * sinTh * gLab[2] * invGn;
            const kfn = Math.hypot(kfx, kfy, kfz) || 1.0;
            const kf = [kfx / kfn, kfy / kfn, kfz / kfn];
            if (ki[0] * kf[0] + ki[1] * kf[1] + ki[2] * kf[2] > 0.9995) {
              if (curSeg.length >= 2) projectedSegs.push({ pts: curSeg, isMate: sgn < 0 });
              curSeg = []; prevAz = null; prevBank = null;
              continue;
            }
            const proj = deps.projectRayToDetectorWithGap(kf, sLab, fullPanels);
            if (!proj || !proj.onPanel) {
              if (curSeg.length >= 2) projectedSegs.push({ pts: curSeg, isMate: sgn < 0 });
              curSeg = []; prevAz = null; prevBank = null;
              continue;
            }
            const az = proj.det_x;
            const vert = isSph ? proj.det_el : proj.det_y;
            if (prevAz !== null && (Math.abs(az - prevAz) > 25.0 || (prevBank !== null && proj.bank !== prevBank))) {
              if (curSeg.length >= 2) projectedSegs.push({ pts: curSeg, isMate: sgn < 0 });
              curSeg = [];
            }
            curSeg.push([az, vert]);
            prevAz = az;
            prevBank = proj.bank;
          }
          if (curSeg.length >= 2) projectedSegs.push({ pts: curSeg, isMate: sgn < 0 });
        }
      }

      if (projectedSegs.length === 0) return;

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let si = 0; si < projectedSegs.length; si++) {
        const seg = projectedSegs[si];
        for (let sh = 0; sh < azShifts.length; sh++) {
          const shiftAz = azShifts[sh];
          ctx.beginPath();
          let anyInBounds = false;
          for (let i = 0; i < seg.pts.length; i++) {
            const sc = deps.detToScreen(seg.pts[i][0] + shiftAz, seg.pts[i][1], w, h, 0.35);
            if (sc[0] >= -40 && sc[0] <= w + 40 && sc[1] >= -40 && sc[1] <= h + 40) anyInBounds = true;
            if (i === 0) ctx.moveTo(sc[0], sc[1]);
            else ctx.lineTo(sc[0], sc[1]);
          }
          if (!anyInBounds) continue;
          ctx.strokeStyle = "rgba(7, 9, 19, 0.88)";
          ctx.lineWidth = 3.8;
          ctx.stroke();
          ctx.strokeStyle = seg.isMate ? "#e879f9" : "#fde047";
          ctx.lineWidth = 2.0;
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    async function requestNodalUpdate(extraBody) {
      const st = deps.state;
      const reqSeq = ++fetchInFlight;
      const selPeak = st.selectedPeakObj;
      const actZones = deps.getActiveZoneObjects();
      const selZone = actZones.find(z => z.zone_id === st.selectedZoneId) || actZones[0] || null;
      const body = Object.assign({
        heterodyne_mode: st.heterodyneMode || "delta",
        roi_mode: st.nodalRoiMode || "origin",
        roi_center_hkl: (selPeak && Array.isArray(selPeak.hkl)) ? selPeak.hkl : null,
        roi_zone_uvw: (selZone && Array.isArray(selZone.uvw)) ? selZone.uvw : null,
        roi_zone_order: 0,
        roi_radius_mult: Number(st.nodalRoiRadiusMult) || 2.5,
      }, extraBody || {});
      try {
        const r = await fetch("api/nodal_surfaces", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const doc = await r.json().catch(() => null);
        if (reqSeq !== fetchInFlight) return doc;
        if (r.ok && doc && doc.nodal_data) {
          setNodalData(doc.nodal_data);
          deps.onUpdated(doc.nodal_data);
        } else if (doc && doc.error && typeof deps.setStatusLine === "function") {
          deps.setStatusLine("[NODAL SURFACES]", `<span style="color:#f87171;">${doc.error}</span>`);
        }
        return doc;
      } catch (_err) {
        return null;
      }
    }

    function scheduleRoiRefresh(delayMs) {
      if (!nodalData || !nodalData.has_phases) return;
      if (roiDebounceTimer) clearTimeout(roiDebounceTimer);
      roiDebounceTimer = setTimeout(() => {
        roiDebounceTimer = null;
        requestNodalUpdate();
      }, typeof delayMs === "number" ? delayMs : 140);
    }

    async function uploadAndSetRefMtz(file) {
      if (!file) return;
      if (typeof deps.setStatusLine === "function") {
        deps.setStatusLine("[REF F₀ MTZ]", `Loading heterodyne reference structure factors from <strong>${file.name}</strong>...`);
      }
      let resolvedPath = null;
      try {
        const rr = await fetch("api/pipeline/resolve_local_files", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ files: [{ name: file.name, size: file.size, lastModified: file.lastModified }] }),
        });
        if (rr.ok) {
          const rdoc = await rr.json();
          if (Array.isArray(rdoc.matched_paths) && rdoc.matched_paths.length === 1) {
            resolvedPath = rdoc.matched_paths[0];
          }
        }
      } catch (_e) {}

      if (!resolvedPath) {
        const fd = new FormData();
        fd.append("files", file, file.name);
        const ur = await fetch("api/pipeline/upload_h5", { method: "POST", body: fd });
        if (ur.ok) {
          const udoc = await ur.json();
          const sp = udoc.saved_paths || udoc.uploaded_paths || [];
          if (sp.length > 0) resolvedPath = sp[0];
        }
      }
      if (!resolvedPath) {
        if (typeof deps.setStatusLine === "function") {
          deps.setStatusLine("[REF F₀ MTZ]", `<span style="color:#f87171;">Could not stage ${file.name}</span>`);
        }
        return;
      }
      const doc = await requestNodalUpdate({ ref_mtz_path: resolvedPath });
      if (doc && doc.nodal_data && typeof deps.setStatusLine === "function") {
        const nd = doc.nodal_data;
        if (nd.ref_warning) {
          deps.setStatusLine("[REF F₀ MTZ]", `<span style="color:#fbbf24;">${nd.ref_warning}</span>`);
        } else if (nd.ref_has_phases) {
          deps.setStatusLine(
            "[HETERODYNE F₀]",
            `Loaded <strong>${nd.ref_name}</strong> (${nd.ref_f_column}, ${nd.ref_phi_column}) • Interference: <strong>${nd.formula_label}</strong>`
          );
        }
      }
    }

    function setNodalData(nextData) {
      nodalData = nextData || null;
      dataSeq += 1;
      decodedCacheSig = "";
      last3DRenderSig = "";
      ensureDecodedMeshes();
    }

    return {
      getNodalData: () => nodalData,
      getDataSeq: () => dataSeq,
      setNodalData,
      hasPhases: () => Boolean(nodalData && nodalData.has_phases),
      invalidate3D: () => { last3DRenderSig = ""; },
      render3DReciprocalView,
      findNearestPeak3D,
      renderWulffVortices,
      renderDetectorVortices,
      requestNodalUpdate,
      scheduleRoiRefresh,
      uploadAndSetRefMtz,
    };
  }
"""
