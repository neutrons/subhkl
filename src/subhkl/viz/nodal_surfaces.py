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
import re
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
_BRICK_PAYLOAD_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_MAX_FIELD_CACHE_ENTRIES = 10
_MAX_BRICK_CACHE_ENTRIES = 16
_SUBSTRUCTURE_NAME_RE = re.compile(
    r"(?:only[_-]?h|[_-]h[_-]|substruct|fofc|delfwt|diff[_-]?map)", re.IGNORECASE
)
_VALID_HETERODYNE_MODES = ("delta", "diff", "product", "single")


def infer_default_heterodyne_mode(
    primary_name: str | None,
    *,
    primary_field: dict[str, Any] | None = None,
    ref_field: dict[str, Any] | None = None,
) -> str:
    """Return 'product' when primary .mtz is an isolated substructure map, else 'delta' [F0*(F - F0)]."""
    if primary_field is not None and ref_field is not None:
        q75_f = float(primary_field.get("q75_f") or 0.0)
        q75_0 = float(ref_field.get("q75_f") or 0.0)
        if q75_f > 1e-9 and q75_0 > 1e-9:
            ratio = q75_f / q75_0
            # Two full-structure models on comparable scale (e.g. 5vnq_no_HIS31_delta vs 5vnq)
            # must subtract F_0 first; otherwise |F_0|^2 > 0 dominates Re(F_0* F) everywhere.
            if 0.65 <= ratio <= 1.50:
                return "delta"
    if primary_name and _SUBSTRUCTURE_NAME_RE.search(str(primary_name)):
        return "product"
    return "delta"


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

    # Build raw ASU amplitude map (in native MTZ amplitude units) for physical heterodyne scale matching
    ops = mtz_sg.operations()
    asu_helper = gemmi.ReciprocalAsu(mtz_sg)
    asu_amp_map: dict[tuple[int, int, int], float] = {}
    step_sample = max(1, len(hkl) // 25000)
    for idx_s in range(0, len(hkl), step_sample):
        hk_list = [int(hkl[idx_s, 0]), int(hkl[idx_s, 1]), int(hkl[idx_s, 2])]
        asu_hkl, _ = asu_helper.to_asu(hk_list, ops)
        asu_amp_map[(int(asu_hkl[0]), int(asu_hkl[1]), int(asu_hkl[2]))] = float(
            f_amp[idx_s]
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
    """Compute multiplier `s` in `s * F_norm - F0_norm` so subtraction preserves true physical ΔF = F_raw - F0_raw.

    Since `F_norm = F_raw / q75_f` and `F0_norm = F0_raw / q75_0`:
    - When both MTZs share a compatible physical scale (`0.1 <= q75_shared_raw(F) / q75_shared_raw(F0) <= 10.0`,
      e.g. `f_calc` for all atoms vs heavy atoms), exact unnormalized subtraction `(F_raw - F0_raw) / q75_0`
      requires `s = q75_f / q75_0` (NOT forcing `|F|` and `|F0|` to have equal Q75, which would leak F0 into ΔF).
    - Only when raw scales differ by >10x (arbitrary unscaled experimental amplitudes vs model amplitudes)
      do we align shared-ASU Q75 amplitudes via `s = (q75_f / q75_0) * (q75_shared_0 / q75_shared_f)`.
    """
    q75_f_global = float(primary_field.get("q75_f") or 1.0)
    q75_0_global = float(ref_field.get("q75_f") or 1.0)
    raw_unit_ratio = (
        q75_f_global / q75_0_global
        if (q75_f_global > 1e-12 and q75_0_global > 1e-12)
        else 1.0
    )

    map_f = primary_field.get("asu_amp_map") or {}
    map_0 = ref_field.get("asu_amp_map") or {}
    if not map_f or not map_0:
        return float(raw_unit_ratio) if 0.1 <= raw_unit_ratio <= 10.0 else 1.0

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
        return float(raw_unit_ratio) if 0.1 <= raw_unit_ratio <= 10.0 else 1.0

    arr_f = np.asarray(shared_f, dtype=np.float64)
    arr_0 = np.asarray(shared_0, dtype=np.float64)
    sh_q75_f = float(np.percentile(arr_f, 75.0))
    sh_q75_0 = float(np.percentile(arr_0, 75.0))
    if sh_q75_f > 1e-9 and sh_q75_0 > 1e-9:
        shared_raw_ratio = sh_q75_f / sh_q75_0
        if 0.1 <= shared_raw_ratio <= 10.0:
            # Same physical scale (e.g. F_all vs F_heavy): preserve exact F_raw - F0_raw!
            return float(raw_unit_ratio)
        # Wildly different arbitrary scales: align shared ASU Q75 before subtracting
        aligned = raw_unit_ratio * (sh_q75_0 / sh_q75_f)
        if np.isfinite(aligned) and 0.05 <= aligned <= 20.0:
            return float(aligned)
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
    diff_grid = float(f_to_f0_scale) * F_grid - F0_grid
    if mode == "diff":
        return diff_grid
    # Default "delta": F_0*(G) * (s * F(G) - F_0(G))
    return np.conj(F0_grid) * diff_grid


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

    # 1. Open curves entering through grid boundary (or orphan starts) — O(N) predecessor lookup!
    has_predecessor = set(next_puncture.values())
    start_candidates = sorted(is_boundary_entry) + [
        pid for pid in range(len(puncture_hkl)) if pid not in has_predecessor
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
    roi_center_g_override: list[float] | tuple[float, float, float] | None = None,
    roi_zone_uvw: list[int] | tuple[int, int, int] | None = None,
    roi_zone_order: int = 0,
    roi_radius_mult: float = 2.5,
    roi_grid_size: int = 58,
    vortex_shell_mult: float = 4.2,
    vortex_grid_size: int = 58,
    cache_radius_mult: float = 4.2,
) -> dict[str, Any]:
    """Compute cached 3D Re=0 & Im=0 nodal meshes and 1D phase-vortex curves in sample reciprocal space.

    To enable zero-latency client-side 3D ROI translation (Origin / Peak / Zone / Free Pan) and
    smooth 60-fps ROI radius slider clipping without server round-trips, the server evaluates and
    caches an uncropped spherical 3D brick (`brick_radius_g >= max(roi_radius, cache_radius_mult * a*)`)
    while also returning exact live ROI metadata (`roi`) for initial framing and unit tests.
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
        if os.path.abspath(str(ref_mtz_path)) == os.path.abspath(str(mtz_path)):
            ref_warning = (
                f"Reference MTZ '{os.path.basename(str(ref_mtz_path))}' matches the primary MTZ; "
                "showing single-field F(G)."
            )
        else:
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

    req_het = str(heterodyne_mode or "delta").lower()
    if req_het == "auto":
        req_het = infer_default_heterodyne_mode(
            primary.get("name"),
            primary_field=primary,
            ref_field=ref_field,
        )
    het_active = bool(ref_field is not None and req_het != "single")
    eff_het_mode = req_het if het_active else "single"
    if eff_het_mode not in _VALID_HETERODYNE_MODES:
        eff_het_mode = (
            infer_default_heterodyne_mode(
                primary.get("name"),
                primary_field=primary,
                ref_field=ref_field,
            )
            if het_active
            else "single"
        )

    f_to_f0_scale = (
        compute_shared_scale_ratio(primary, ref_field)
        if (ref_field is not None and eff_het_mode in ("delta", "diff"))
        else 1.0
    )

    # Resolve requested ROI center & geometry in sample reciprocal space (A^-1)
    r_mult = float(np.clip(roi_radius_mult, 0.8, 8.0))
    roi_radius_g = min(g_max, r_mult * mean_g_star)

    mode_roi = str(roi_mode or "origin").lower()
    center_hkl = np.zeros(3, dtype=np.float64)
    slab_normal_g: np.ndarray | None = None
    slab_offset_g = 0.0
    slab_half_thickness_g: float | None = None

    if mode_roi == "peak" and roi_center_hkl is not None and len(roi_center_hkl) == 3:
        center_hkl = np.asarray(roi_center_hkl, dtype=np.float64)
    elif (
        mode_roi == "pan"
        and roi_center_g_override is not None
        and len(roi_center_g_override) == 3
    ):
        center_hkl = ub_inv @ np.asarray(roi_center_g_override, dtype=np.float64)
    elif mode_roi == "pan" and roi_center_hkl is not None and len(roi_center_hkl) == 3:
        center_hkl = np.asarray(roi_center_hkl, dtype=np.float64)
    elif mode_roi == "zone" and roi_zone_uvw is not None and len(roi_zone_uvw) == 3:
        uvw = np.asarray(roi_zone_uvw, dtype=np.float64)
        t_sample = ub_inv_t @ uvw
        t_norm = float(np.linalg.norm(t_sample))
        if t_norm > 1e-8:
            slab_normal_g = t_sample / t_norm
            slab_offset_g = float(roi_zone_order) / t_norm
            slab_half_thickness_g = max(0.48 * mean_g_star, 0.20 * roi_radius_g)
            roi_radius_g = min(g_max, max(roi_radius_g, 1.35 * r_mult * mean_g_star))
            center_g_zone = slab_offset_g * slab_normal_g
            center_hkl = ub_inv @ center_g_zone
        else:
            mode_roi = "origin"
    else:
        mode_roi = "origin"

    roi_center_g = ub @ center_hkl

    # Choose cached uncropped 3D spherical brick covering at least cache_radius_mult * mean_g_star:
    # Keep brick anchored at origin (0,0,0) whenever the requested ROI fits inside the origin brick
    # so switching Origin <-> Zone <-> nearby Peaks or dragging the slider hits the exact same brick!
    brick_mult = max(float(cache_radius_mult), float(vortex_shell_mult), r_mult * 1.15)
    brick_radius_g = min(g_max, brick_mult * mean_g_star)
    dist_origin_g = float(np.linalg.norm(roi_center_g))
    if dist_origin_g + roi_radius_g <= brick_radius_g * 0.96:
        brick_center_hkl = np.zeros(3, dtype=np.float64)
    else:
        # Quantize off-origin brick center to integer Miller lattice so nearby pan/peak clicks share cache
        brick_center_hkl = np.round(center_hkl).astype(np.float64)
        # Ensure brick radius comfortably encloses (roi_center_g, roi_radius_g)
        brick_center_try_g = ub @ brick_center_hkl
        needed_r = (
            float(np.linalg.norm(roi_center_g - brick_center_try_g))
            + roi_radius_g * 1.08
        )
        brick_radius_g = max(brick_radius_g, min(g_max * 1.15, needed_r))

    brick_center_g = ub @ brick_center_hkl
    ub_inv_row_norms = np.linalg.norm(ub_inv, axis=1)
    half_span_hkl = np.maximum(brick_radius_g * ub_inv_row_norms * 1.04, 0.85)
    max_span_h = float(np.max(2.0 * half_span_hkl))
    n_grid_brick = int(
        np.clip(max(int(roi_grid_size), math.ceil(max_span_h / 0.24)), 30, 68)
    )

    brick_cache_key = (
        primary.get("path"),
        primary.get("q75_f"),
        ref_field.get("path") if ref_field is not None else None,
        ref_field.get("q75_f") if ref_field is not None else None,
        eff_het_mode,
        round(float(f_to_f0_scale), 8),
        round(float(g_max), 4),
        tuple(np.round(ub.ravel(), 5).tolist()),
        tuple(np.round(brick_center_hkl, 2).tolist()),
        round(float(brick_radius_g), 4),
        n_grid_brick,
    )

    cached_brick = _BRICK_PAYLOAD_CACHE.get(brick_cache_key)
    if cached_brick is not None:
        _BRICK_PAYLOAD_CACHE.move_to_end(brick_cache_key)
        is_real_only = cached_brick["is_real_only"]
        re_mesh = cached_brick["re_surface"]
        im_mesh = cached_brick["im_surface"]
        brick_vortices = cached_brick["brick_vortices"]
    else:
        h0_b = np.linspace(
            brick_center_hkl[0] - half_span_hkl[0],
            brick_center_hkl[0] + half_span_hkl[0],
            n_grid_brick,
        )
        h1_b = np.linspace(
            brick_center_hkl[1] - half_span_hkl[1],
            brick_center_hkl[1] + half_span_hkl[1],
            n_grid_brick,
        )
        h2_b = np.linspace(
            brick_center_hkl[2] - half_span_hkl[2],
            brick_center_hkl[2] + half_span_hkl[2],
            n_grid_brick,
        )

        Z_brick = evaluate_combined_complex_grid(
            primary,
            h0_b,
            h1_b,
            h2_b,
            ref_field=ref_field,
            heterodyne_mode=eff_het_mode,
            f_to_f0_scale=f_to_f0_scale,
        )
        amp_brick = (
            float(np.percentile(np.abs(Z_brick), 90.0)) if Z_brick.size > 0 else 0.0
        )
        is_real_only = bool(
            float(np.max(np.abs(Z_brick.imag))) <= 1e-4 * max(amp_brick, 1e-12)
        )

        # Extract uncropped spherical brick meshes (slab/ROI radius clipping happens live on client GPU!)
        re_mesh = _extract_single_isosurface_mesh(
            Z_brick.real,
            amp_brick,
            h0_b,
            h1_b,
            h2_b,
            ub,
            ub_inv_t,
            g_max=g_max,
            roi_center_g=brick_center_g,
            roi_radius_g=brick_radius_g,
        )
        im_mesh = _extract_single_isosurface_mesh(
            Z_brick.imag,
            amp_brick,
            h0_b,
            h1_b,
            h2_b,
            ub,
            ub_inv_t,
            g_max=g_max,
            roi_center_g=brick_center_g,
            roi_radius_g=brick_radius_g,
        )
        brick_vortices = extract_vortex_polylines(
            Z_brick,
            h0_b,
            h1_b,
            h2_b,
            ub,
            g_max=g_max,
            roi_center_g=brick_center_g,
            roi_radius_g=brick_radius_g,
            max_polylines=320,
        )
        _BRICK_PAYLOAD_CACHE[brick_cache_key] = {
            "is_real_only": is_real_only,
            "re_surface": re_mesh,
            "im_surface": im_mesh,
            "brick_vortices": brick_vortices,
        }
        while len(_BRICK_PAYLOAD_CACHE) > _MAX_BRICK_CACHE_ENTRIES:
            _BRICK_PAYLOAD_CACHE.popitem(last=False)

    # Also ensure origin-shell vortices are available when viewing an off-origin peak brick
    shell_radius_g = min(
        g_max, max(brick_radius_g, float(vortex_shell_mult) * mean_g_star)
    )
    if float(np.linalg.norm(brick_center_hkl)) < 1e-6:
        global_vortices = brick_vortices
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
                        math.ceil(float(np.max(2.0 * shell_half_hkl)) / 0.30),
                    ),
                    34,
                    58,
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
                max_polylines=320,
            )
            _GLOBAL_VORTEX_CACHE[v_cache_key] = global_vortices
            while len(_GLOBAL_VORTEX_CACHE) > _MAX_FIELD_CACHE_ENTRIES:
                _GLOBAL_VORTEX_CACHE.popitem(last=False)

    if eff_het_mode == "delta":
        formula_label = "F₀*·(F − F₀)"
    elif eff_het_mode == "diff":
        formula_label = "ΔF = F − F₀"
    elif eff_het_mode == "product":
        formula_label = "F₀*·F"
    else:
        formula_label = "F(G)"

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
        "brick": {
            "center_hkl": [round(float(x), 3) for x in brick_center_hkl],
            "center_g": [round(float(x), 5) for x in brick_center_g],
            "radius_g": round(float(brick_radius_g), 5),
            "radius_mult": round(float(brick_radius_g / max(mean_g_star, 1e-6)), 2),
            "grid_size": n_grid_brick,
        },
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
            "grid_size": n_grid_brick,
        },
        "re_surface": re_mesh,
        "im_surface": im_mesh,
        "roi_vortices": brick_vortices,
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
    is_new_ref = False
    if req_json.get("clear_ref_mtz"):
        next_ref = None
        if isinstance(payload, dict):
            payload["ref_mtz_overlay_name"] = None
    elif req_json.get("ref_mtz_path"):
        cand_ref = str(req_json["ref_mtz_path"]).strip()
        if not os.path.isfile(cand_ref):
            return (
                {"ok": False, "error": f"Reference .mtz file not found: {cand_ref}"},
                400,
                next_ref,
                heterodyne_mode,
            )
        is_new_ref = cand_ref != ref_mtz_path
        next_ref = cand_ref

    raw_req_het = req_json.get("heterodyne_mode")
    if req_json.get("clear_ref_mtz") or not next_ref:
        next_het = "single"
    elif raw_req_het == "auto" or (is_new_ref and not raw_req_het):
        next_het = "auto"
    else:
        next_het = str(raw_req_het or heterodyne_mode or "delta").lower()
        if next_het not in _VALID_HETERODYNE_MODES:
            next_het = "delta"

    if not mtz_path or not os.path.isfile(mtz_path):
        return (
            {
                "ok": False,
                "error": "Load a primary .mtz file first before loading Reference F₀ or computing nodal surfaces.",
            },
            400,
            next_ref,
            next_het,
        )

    cur_p = payload or {}
    ref_asu_lu: dict[str, list[float]] | None = None
    ref_ex_lu: dict[str, list[float]] | None = None
    ref_n_refl = 0
    if next_ref and os.path.isfile(next_ref):
        try:
            from subhkl.viz.rainbow_zones import extract_ref_mtz_intensity_lookup

            ref_asu_lu, ref_ex_lu, ref_n_refl = extract_ref_mtz_intensity_lookup(
                next_ref,
                str(cur_p.get("space_group") or "P 1"),
                mtz_global_i_scale=float(cur_p.get("mtz_global_i_scale") or 1.0),
                prefer_i_col=bool(
                    cur_p.get("mtz_has_i_column") and not cur_p.get("mtz_has_f_column")
                ),
            )
            if not ref_asu_lu:
                ref_asu_lu = None
        except (OSError, ValueError, RuntimeError):
            ref_asu_lu = None
            ref_ex_lu = None

    ref_base_name = os.path.basename(next_ref) if next_ref else None
    if isinstance(payload, dict):
        payload["ref_mtz_overlay_name"] = ref_base_name
        payload["ref_mtz_i_asu_lookup"] = ref_asu_lu
        payload["ref_mtz_i_exact_lookup"] = ref_ex_lu

    ub_raw = req_json.get("ub_matrix") or cur_p.get("ub_matrix")
    if ub_raw is None:
        if req_json.get("ref_mtz_path") or req_json.get("clear_ref_mtz"):
            return (
                {
                    "ok": True,
                    "nodal_data": None,
                    "ref_mtz_overlay_name": ref_base_name,
                    "ref_mtz_reflections": ref_n_refl,
                    "ref_mtz_i_asu_lookup": ref_asu_lu,
                    "ref_mtz_i_exact_lookup": ref_ex_lu,
                },
                200,
                next_ref,
                next_het,
            )
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
    roi_center_g = req_json.get("roi_center_g")
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
        roi_center_g_override=roi_center_g,
        roi_zone_uvw=roi_zone_uvw,
        roi_zone_order=roi_zone_order,
        roi_radius_mult=roi_radius_mult,
    )
    if isinstance(payload, dict):
        payload["nodal_data"] = nodal_data
        payload["ref_mtz_overlay_name"] = ref_base_name
    resolved_het = str(nodal_data.get("heterodyne_mode") or next_het)
    if resolved_het not in _VALID_HETERODYNE_MODES:
        resolved_het = "delta"
    if not nodal_data.get("has_phases"):
        if req_json.get("ref_mtz_path") or req_json.get("clear_ref_mtz"):
            return (
                {
                    "ok": True,
                    "nodal_data": nodal_data,
                    "ref_mtz_overlay_name": ref_base_name,
                    "ref_mtz_reflections": ref_n_refl,
                    "ref_mtz_i_asu_lookup": ref_asu_lu,
                    "ref_mtz_i_exact_lookup": ref_ex_lu,
                },
                200,
                next_ref,
                resolved_het,
            )
        return (
            {
                "ok": False,
                "error": nodal_data.get("reason")
                or "Primary .mtz has no phase column (type 'P').",
                "nodal_data": nodal_data,
            },
            400,
            next_ref,
            resolved_het,
        )
    return (
        {
            "ok": True,
            "nodal_data": nodal_data,
            "ref_mtz_overlay_name": ref_base_name,
            "ref_mtz_reflections": ref_n_refl,
            "ref_mtz_i_asu_lookup": ref_asu_lu,
            "ref_mtz_i_exact_lookup": ref_ex_lu,
        },
        200,
        next_ref,
        resolved_het,
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
    let decodedMeshes = { re: null, im: null, vortex: null };
    let glCanvas = null;
    let gl = null;
    let glProg = null;
    let glVortexProg = null;
    let glBufs = { re: null, im: null, vortex: null };
    let glUnavailable = false;
    let last3DRenderSig = "";
    let screenPeakHits3D = [];
    let fetchInFlight = 0;
    let roiDebounceTimer = null;
    let brickFetchInProgress = false;
    const brickCache = new Map();
    let lastSingleOriginData = (nodalData && nodalData.has_phases && !nodalData.ref_has_phases) ? nodalData : null;
    let panCenterG = [0.0, 0.0, 0.0];
    let animCenterG = [0.0, 0.0, 0.0];

    function makeBrickKey(nd) {
      if (!nd || !nd.has_phases) return "";
      const bc = (nd.brick && Array.isArray(nd.brick.center_g))
        ? nd.brick.center_g
        : ((nd.roi && Array.isArray(nd.roi.center_g)) ? nd.roi.center_g : [0, 0, 0]);
      const cKey = bc.map(v => Number(v || 0).toFixed(4)).join(",");
      const refPart = nd.ref_has_phases ? (nd.ref_name || "ref") : "";
      return `${nd.primary_name || ""}|${refPart}|${nd.heterodyne_mode || "single"}|${cKey}`;
    }

    function storeBrickInCache(nd) {
      if (!nd || !nd.has_phases) return;
      const key = makeBrickKey(nd);
      if (key) {
        if (brickCache.has(key)) brickCache.delete(key);
        brickCache.set(key, nd);
        while (brickCache.size > 14) {
          const oldest = brickCache.keys().next().value;
          brickCache.delete(oldest);
        }
      }
      const bc = (nd.brick && Array.isArray(nd.brick.center_g)) ? nd.brick.center_g : [0, 0, 0];
      if (!nd.ref_has_phases && Math.hypot(bc[0] || 0, bc[1] || 0, bc[2] || 0) < 1e-4) {
        lastSingleOriginData = nd;
      }
    }

    if (nodalData && nodalData.has_phases) {
      storeBrickInCache(nodalData);
    }

    function computeMeanGStarFromUb(ub) {
      if (!ub) return 0.04;
      const a = Math.hypot(ub[0][0], ub[1][0], ub[2][0]);
      const b = Math.hypot(ub[0][1], ub[1][1], ub[2][1]);
      const c = Math.hypot(ub[0][2], ub[1][2], ub[2][2]);
      return Math.max(1e-4, (a + b + c) / 3.0);
    }

    function resolveLiveRoi() {
      const st = deps.state;
      const ub = deps.getActiveUbMatrix();
      const meanGStar = (nodalData && nodalData.brick && Number(nodalData.brick.mean_g_star) > 0)
        ? Number(nodalData.brick.mean_g_star)
        : ((nodalData && nodalData.roi && Number(nodalData.roi.mean_g_star) > 0)
            ? Number(nodalData.roi.mean_g_star)
            : computeMeanGStarFromUb(ub));
      const mult = Math.max(0.6, Math.min(8.0, Number(st.nodalRoiRadiusMult) || 2.5));
      const radiusG = Math.max(1e-4, mult * meanGStar);
      const mode = st.nodalRoiMode || "origin";
      let centerG = [0.0, 0.0, 0.0];
      let slabActive = false;
      let slabNormal = [0.0, 0.0, 1.0];
      let slabOffset = 0.0;
      let slabHalfThick = radiusG;
      let zoneColor = "#38bdf8";
      let zoneLabel = "";

      if (mode === "peak") {
        const sp = st.selectedPeakObj;
        if (sp && Array.isArray(sp.hkl) && sp.hkl.length === 3 && ub) {
          centerG = [
            ub[0][0] * sp.hkl[0] + ub[0][1] * sp.hkl[1] + ub[0][2] * sp.hkl[2],
            ub[1][0] * sp.hkl[0] + ub[1][1] * sp.hkl[1] + ub[1][2] * sp.hkl[2],
            ub[2][0] * sp.hkl[0] + ub[2][1] * sp.hkl[1] + ub[2][2] * sp.hkl[2],
          ];
        } else if (nodalData && nodalData.roi && nodalData.roi.mode === "peak" && Array.isArray(nodalData.roi.center_g)) {
          centerG = nodalData.roi.center_g.slice(0, 3);
        }
      } else if (mode === "pan") {
        centerG = [panCenterG[0], panCenterG[1], panCenterG[2]];
      } else if (mode === "zone") {
        const actZones = deps.getActiveZoneObjects() || [];
        const selZone = actZones.find(z => z.zone_id === st.selectedZoneId) || actZones[0] || null;
        if (selZone && Array.isArray(selZone.uvw) && selZone.uvw.length === 3 && ub) {
          const invUb = deps.mat3Inv(ub);
          if (invUb) {
            const u = selZone.uvw[0], v = selZone.uvw[1], ww = selZone.uvw[2];
            const tx = invUb[0][0] * u + invUb[1][0] * v + invUb[2][0] * ww;
            const ty = invUb[0][1] * u + invUb[1][1] * v + invUb[2][1] * ww;
            const tz = invUb[0][2] * u + invUb[1][2] * v + invUb[2][2] * ww;
            const tn = Math.hypot(tx, ty, tz);
            if (tn > 1e-8) {
              slabNormal = [tx / tn, ty / tn, tz / tn];
              slabActive = true;
              slabOffset = 0.0;
              slabHalfThick = Math.max(0.38 * meanGStar, 0.22 * radiusG);
              zoneColor = selZone.rainbow_color || "#38bdf8";
              zoneLabel = selZone.label || `[${u} ${v} ${ww}]`;
            }
          }
        }
      }

      return {
        mode,
        centerG,
        radiusG,
        meanGStar,
        mult,
        slabActive,
        slabNormal,
        slabOffset,
        slabHalfThick,
        zoneColor,
        zoneLabel,
      };
    }

    function pointInsideLiveRoi(pt, liveRoi) {
      if (!liveRoi) return true;
      const dx = pt[0] - liveRoi.centerG[0];
      const dy = pt[1] - liveRoi.centerG[1];
      const dz = pt[2] - liveRoi.centerG[2];
      if (dx * dx + dy * dy + dz * dz > liveRoi.radiusG * liveRoi.radiusG) return false;
      if (liveRoi.slabActive) {
        const d = pt[0] * liveRoi.slabNormal[0] + pt[1] * liveRoi.slabNormal[1] + pt[2] * liveRoi.slabNormal[2] - liveRoi.slabOffset;
        if (Math.abs(d) > liveRoi.slabHalfThick) return false;
      }
      return true;
    }

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

    function decodeVortexRibbonGeometry(vList) {
      if (!Array.isArray(vList) || vList.length === 0) {
        return { nSegs: 0, pos0: new Float32Array(0), pos1: new Float32Array(0), side: new Float32Array(0), indices: new Uint32Array(0) };
      }
      let nSegs = 0;
      for (let vi = 0; vi < vList.length; vi++) {
        const pts = vList[vi] && vList[vi].pts_g;
        if (Array.isArray(pts) && pts.length >= 2) nSegs += (pts.length - 1);
      }
      if (nSegs === 0) {
        return { nSegs: 0, pos0: new Float32Array(0), pos1: new Float32Array(0), side: new Float32Array(0), indices: new Uint32Array(0) };
      }
      const pos0 = new Float32Array(nSegs * 4 * 3);
      const pos1 = new Float32Array(nSegs * 4 * 3);
      const side = new Float32Array(nSegs * 4 * 2);
      const indices = new Uint32Array(nSegs * 6);
      const sideTemplate = [0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0];
      let sIdx = 0;
      for (let vi = 0; vi < vList.length; vi++) {
        const pts = vList[vi] && vList[vi].pts_g;
        if (!Array.isArray(pts) || pts.length < 2) continue;
        for (let i = 0; i < pts.length - 1; i++) {
          const a = pts[i], b = pts[i + 1];
          const vBase = sIdx * 4;
          for (let k = 0; k < 4; k++) {
            const pOff = (vBase + k) * 3;
            pos0[pOff] = a[0]; pos0[pOff + 1] = a[1]; pos0[pOff + 2] = a[2];
            pos1[pOff] = b[0]; pos1[pOff + 1] = b[1]; pos1[pOff + 2] = b[2];
            const sOff = (vBase + k) * 2;
            side[sOff] = sideTemplate[2 * k];
            side[sOff + 1] = sideTemplate[2 * k + 1];
          }
          const iOff = sIdx * 6;
          indices[iOff] = vBase;
          indices[iOff + 1] = vBase + 1;
          indices[iOff + 2] = vBase + 2;
          indices[iOff + 3] = vBase;
          indices[iOff + 4] = vBase + 2;
          indices[iOff + 5] = vBase + 3;
          sIdx++;
        }
      }
      return { nSegs, pos0, pos1, side, indices };
    }

    function getActiveVortexList(nd) {
      if (!nd) return [];
      const roiV = Array.isArray(nd.roi_vortices) ? nd.roi_vortices : null;
      const globV = Array.isArray(nd.vortices) ? nd.vortices : null;
      if (!roiV) return globV || [];
      if (!globV || roiV === globV) return roiV;
      const br = nd.brick;
      const ch = br && Array.isArray(br.center_hkl) ? br.center_hkl : [0, 0, 0];
      const offOrigin = Math.hypot(Number(ch[0]) || 0, Number(ch[1]) || 0, Number(ch[2]) || 0) > 1e-4;
      if (!offOrigin) return roiV.length >= globV.length ? roiV : globV;
      // Off-origin brick: `roi_vortices` covers the active peak/pan brick while `vortices` covers the origin shell.
      // Merge `roi_vortices` with any origin-shell polylines outside the brick so live ROI clipping works everywhere.
      const bCg = br && Array.isArray(br.center_g) ? br.center_g : null;
      const bR2 = br && Number(br.radius_g) > 0 ? Math.pow(Number(br.radius_g) * 0.96, 2) : -1.0;
      if (!bCg || bR2 <= 0) return roiV.length ? roiV : globV;
      const merged = roiV.slice();
      for (let i = 0; i < globV.length; i++) {
        const pts = globV[i] && globV[i].pts_g;
        if (!Array.isArray(pts) || pts.length < 2) continue;
        const mid = pts[pts.length >> 1];
        const dx = mid[0] - bCg[0], dy = mid[1] - bCg[1], dz = mid[2] - bCg[2];
        if (dx * dx + dy * dy + dz * dz > bR2) merged.push(globV[i]);
      }
      return merged;
    }

    function ensureDecodedMeshes() {
      if (!nodalData || !nodalData.has_phases) {
        decodedMeshes = { re: null, im: null, vortex: null };
        decodedCacheSig = "";
        return decodedMeshes;
      }
      const reM = nodalData.re_surface || {};
      const imM = nodalData.im_surface || {};
      const vList = getActiveVortexList(nodalData);
      const sig = [
        makeBrickKey(nodalData),
        reM.n_tris || 0,
        (reM.verts_b64 || "").slice(0, 24),
        imM.n_tris || 0,
        (imM.verts_b64 || "").slice(0, 24),
        (nodalData.roi_vortices || []).length,
        vList.length,
      ].join("|");
      if (sig === decodedCacheSig) return decodedMeshes;
      decodedCacheSig = sig;
      decodedMeshes = {
        re: decodeSingleMesh(reM),
        im: decodeSingleMesh(imM),
        vortex: decodeVortexRibbonGeometry(vList),
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

        function linkProgram(vsSrc, fsSrc) {
          const vs = compileShader(gl.VERTEX_SHADER, vsSrc);
          const fs = compileShader(gl.FRAGMENT_SHADER, fsSrc);
          if (!vs || !fs) return null;
          const p = gl.createProgram();
          gl.attachShader(p, vs);
          gl.attachShader(p, fs);
          gl.linkProgram(p);
          if (!gl.getProgramParameter(p, gl.LINK_STATUS)) return null;
          return p;
        }

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
          "varying vec3 vSamplePos;",
          "varying float vEyeDist;",
          "void main() {",
          "  vec3 p = uRot * ((aPos - uCenter) * uInvScale);",
          "  vec3 n = normalize(uRot * aNorm);",
          "  float eyeDist = uCamDist - p.z;",
          "  float dist = max(0.28, eyeDist);",
          "  float kD = sqrt(uCamDist * uCamDist - 1.0);",
          "  float persp = kD / dist;",
          "  float clipX = p.x * persp * uAspectX;",
          "  float clipY = p.y * persp * uAspectY;",
          "  float clipZ = clamp(-p.z * 0.40, -0.98, 0.98);",
          "  vNorm = n;",
          "  vSamplePos = aPos;",
          "  vEyeDist = eyeDist;",
          "  gl_Position = vec4(clipX, clipY, clipZ, 1.0);",
          "}",
        ].join("\n");

        const fsSrc = [
          "precision mediump float;",
          "uniform vec4 uColor;",
          "uniform vec3 uClipCenter;",
          "uniform float uClipRadius;",
          "uniform float uSlabActive;",
          "uniform vec3 uSlabNormal;",
          "uniform float uSlabOffset;",
          "uniform float uSlabHalfThick;",
          "varying vec3 vNorm;",
          "varying vec3 vSamplePos;",
          "varying float vEyeDist;",
          "void main() {",
          "  if (vEyeDist < 0.28) discard;",
          "  vec3 dRoi = vSamplePos - uClipCenter;",
          "  if (dot(dRoi, dRoi) > uClipRadius * uClipRadius) discard;",
          "  if (uSlabActive > 0.5) {",
          "    if (abs(dot(vSamplePos, uSlabNormal) - uSlabOffset) > uSlabHalfThick) discard;",
          "  }",
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

        const prog = linkProgram(vsSrc, fsSrc);
        if (!prog) {
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
          uClipCenter: gl.getUniformLocation(prog, "uClipCenter"),
          uClipRadius: gl.getUniformLocation(prog, "uClipRadius"),
          uSlabActive: gl.getUniformLocation(prog, "uSlabActive"),
          uSlabNormal: gl.getUniformLocation(prog, "uSlabNormal"),
          uSlabOffset: gl.getUniformLocation(prog, "uSlabOffset"),
          uSlabHalfThick: gl.getUniformLocation(prog, "uSlabHalfThick"),
        };

        const vsVortexSrc = [
          "attribute vec3 aPos0;",
          "attribute vec3 aPos1;",
          "attribute vec2 aSide;",
          "uniform mat3 uRot;",
          "uniform vec3 uCenter;",
          "uniform float uInvScale;",
          "uniform float uAspectX;",
          "uniform float uAspectY;",
          "uniform float uCamDist;",
          "uniform vec2 uViewportPx;",
          "uniform float uHalfWidthPx;",
          "varying vec3 vSamplePos;",
          "varying float vEdge;",
          "varying float vEyeDist;",
          "vec4 projPt(vec3 g) {",
          "  vec3 p = uRot * ((g - uCenter) * uInvScale);",
          "  float eyeDist = uCamDist - p.z;",
          "  float dist = max(0.28, eyeDist);",
          "  float kD = sqrt(uCamDist * uCamDist - 1.0);",
          "  float persp = kD / dist;",
          "  return vec4(p.x * persp * uAspectX, p.y * persp * uAspectY, clamp(-p.z * 0.40 - 0.004, -0.99, 0.99), eyeDist);",
          "}",
          "void main() {",
          "  vec4 c0 = projPt(aPos0);",
          "  vec4 c1 = projPt(aPos1);",
          "  vec2 dPx = (c1.xy - c0.xy) * 0.5 * uViewportPx;",
          "  float dLen = length(dPx);",
          "  vec2 dir = dLen > 1e-3 ? (dPx / dLen) : vec2(1.0, 0.0);",
          "  vec2 perp = vec2(-dir.y, dir.x);",
          "  vec4 baseC = mix(c0, c1, aSide.x);",
          "  vec2 offClip = (perp * (aSide.y * uHalfWidthPx) + dir * ((aSide.x - 0.5) * 1.2)) * (2.0 / uViewportPx);",
          "  vSamplePos = mix(aPos0, aPos1, aSide.x);",
          "  vEdge = aSide.y;",
          "  vEyeDist = baseC.w;",
          "  gl_Position = vec4(baseC.xy + offClip, baseC.z, 1.0);",
          "}",
        ].join("\n");

        const fsVortexSrc = [
          "precision mediump float;",
          "uniform vec3 uClipCenter;",
          "uniform float uClipRadius;",
          "uniform float uSlabActive;",
          "uniform vec3 uSlabNormal;",
          "uniform float uSlabOffset;",
          "uniform float uSlabHalfThick;",
          "varying vec3 vSamplePos;",
          "varying float vEdge;",
          "varying float vEyeDist;",
          "void main() {",
          "  if (vEyeDist < 0.28) discard;",
          "  vec3 dRoi = vSamplePos - uClipCenter;",
          "  if (dot(dRoi, dRoi) > uClipRadius * uClipRadius) discard;",
          "  if (uSlabActive > 0.5) {",
          "    if (abs(dot(vSamplePos, uSlabNormal) - uSlabOffset) > uSlabHalfThick) discard;",
          "  }",
          "  float ae = abs(vEdge);",
          "  if (ae > 0.98) discard;",
          "  vec3 core = vec3(0.992, 0.878, 0.278);",
          "  vec3 halo = vec3(0.027, 0.035, 0.075);",
          "  float t = smoothstep(0.48, 0.72, ae);",
          "  vec3 rgb = mix(core, halo, t);",
          "  float depthCue = clamp(1.28 - 0.20 * vEyeDist, 0.48, 1.0);",
          "  float alpha = (1.0 - smoothstep(0.84, 0.98, ae)) * depthCue;",
          "  gl_FragColor = vec4(rgb, alpha);",
          "}",
        ].join("\n");

        const vProg = linkProgram(vsVortexSrc, fsVortexSrc);
        if (vProg) {
          glVortexProg = {
            prog: vProg,
            aPos0: gl.getAttribLocation(vProg, "aPos0"),
            aPos1: gl.getAttribLocation(vProg, "aPos1"),
            aSide: gl.getAttribLocation(vProg, "aSide"),
            uRot: gl.getUniformLocation(vProg, "uRot"),
            uCenter: gl.getUniformLocation(vProg, "uCenter"),
            uInvScale: gl.getUniformLocation(vProg, "uInvScale"),
            uAspectX: gl.getUniformLocation(vProg, "uAspectX"),
            uAspectY: gl.getUniformLocation(vProg, "uAspectY"),
            uCamDist: gl.getUniformLocation(vProg, "uCamDist"),
            uViewportPx: gl.getUniformLocation(vProg, "uViewportPx"),
            uHalfWidthPx: gl.getUniformLocation(vProg, "uHalfWidthPx"),
            uClipCenter: gl.getUniformLocation(vProg, "uClipCenter"),
            uClipRadius: gl.getUniformLocation(vProg, "uClipRadius"),
            uSlabActive: gl.getUniformLocation(vProg, "uSlabActive"),
            uSlabNormal: gl.getUniformLocation(vProg, "uSlabNormal"),
            uSlabOffset: gl.getUniformLocation(vProg, "uSlabOffset"),
            uSlabHalfThick: gl.getUniformLocation(vProg, "uSlabHalfThick"),
          };
        }

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

    function uploadGlVortexRibbon(vr) {
      if (!gl || !glVortexProg || !vr || !vr.nSegs) return null;
      const b0 = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b0);
      gl.bufferData(gl.ARRAY_BUFFER, vr.pos0, gl.STATIC_DRAW);
      const b1 = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b1);
      gl.bufferData(gl.ARRAY_BUFFER, vr.pos1, gl.STATIC_DRAW);
      const bSide = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, bSide);
      gl.bufferData(gl.ARRAY_BUFFER, vr.side, gl.STATIC_DRAW);
      const ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, vr.indices, gl.STATIC_DRAW);
      return { b0, b1, bSide, ibo, count: vr.indices.length, indexType: gl.UNSIGNED_INT };
    }

    function disposeGlMesh(b) {
      if (!gl || !b) return;
      if (b.vbo) gl.deleteBuffer(b.vbo);
      if (b.nbo) gl.deleteBuffer(b.nbo);
      if (b.b0) gl.deleteBuffer(b.b0);
      if (b.b1) gl.deleteBuffer(b.b1);
      if (b.bSide) gl.deleteBuffer(b.bSide);
      if (b.ibo) gl.deleteBuffer(b.ibo);
    }

    function uploadGlMeshes() {
      if (!gl || !glProg) return;
      disposeGlMesh(glBufs.re);
      disposeGlMesh(glBufs.im);
      disposeGlMesh(glBufs.vortex);
      glBufs.re = decodedMeshes.re ? uploadGlSingleMesh(decodedMeshes.re) : null;
      glBufs.im = decodedMeshes.im ? uploadGlSingleMesh(decodedMeshes.im) : null;
      glBufs.vortex = decodedMeshes.vortex ? uploadGlVortexRibbon(decodedMeshes.vortex) : null;
    }

    // Camera matrix matching projectToWulff right-handed SO(3) lab camera coupled with R_gonio
    function buildCameraTransform(Rg) {
      const st = deps.state;
      const yawRad = (st.wulffYawDeg || 0.0) * Math.PI / 180.0;
      const pitchRad = (st.wulffPitchDeg || 0.0) * Math.PI / 180.0;
      const cy = Math.cos(yawRad), sy = Math.sin(yawRad);
      const cp = Math.cos(pitchRad), sp = Math.sin(pitchRad);
      const Rc = [
        [cy, 0.0, -sy],
        [-sp * sy, cp, -sp * cy],
        [cp * sy, sp, cp * cy],
      ];
      const Rtot = deps.mat3Mul(Rc, Rg);
      return Rtot;
    }

    function get3DZoom() {
      const z = Number(deps.state && deps.state.recip3dZoom);
      if (!Number.isFinite(z) || z <= 0.0) return 1.0;
      return Math.max(0.35, Math.min(3.6, z));
    }

    function get3DSceneBounds(liveRoi) {
      const st = deps.state;
      const invDmin = 1.0 / Math.max(0.4, Number(st.dMin) || 1.5);
      const roi = liveRoi || resolveLiveRoi();
      // Anchor default scene radius to crystal reciprocal lattice scale (~3.5× mean a⋆),
      // independent of the ROI slider so dragging the slider visibly grows/shrinks the ROI!
      const latticeScaleR = Math.min(invDmin, Math.max(0.015, roi.meanGStar * 3.5));
      const radius = nodalData && nodalData.has_phases
        ? Math.max(latticeScaleR, roi.radiusG * 1.12)
        : invDmin;
      const cx = Number.isFinite(animCenterG[0]) ? animCenterG[0] : 0.0;
      const cy = Number.isFinite(animCenterG[1]) ? animCenterG[1] : 0.0;
      const cz = Number.isFinite(animCenterG[2]) ? animCenterG[2] : 0.0;
      return { center: [cx, cy, cz], radius: Math.max(0.008, radius), invDmin };
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
      const eyeDist = D - pz;
      const dist = Math.max(0.28, eyeDist);
      const persp = kD / dist;
      const scale = Rpx * zoom;
      const sx = cx + px * persp * scale;
      const sy = cy - py * persp * scale;
      const vis = eyeDist >= 0.28 && Number.isFinite(sx) && Number.isFinite(sy);
      return [sx, sy, pz, (D - 0.2) / dist, vis];
    }

    function renderSurfacesAndVortices3D(ctx, w, h, dpr, Rtot, center, invScale, Rpx, zoom, liveRoi, isExp) {
      const st = deps.state;
      if (!nodalData || !nodalData.has_phases) return;
      ensureDecodedMeshes();
      const hasRe = st.showNodalSurfacesRe && decodedMeshes.re && decodedMeshes.re.nTris > 0;
      const hasIm = st.showNodalSurfacesIm && decodedMeshes.im && decodedMeshes.im.nTris > 0;
      const hasVor = st.showPhaseVortices && decodedMeshes.vortex && decodedMeshes.vortex.nSegs > 0;
      if (!hasRe && !hasIm && !hasVor) return;

      const surfAlpha = Math.max(0.12, Math.min(0.92, Number(st.nodalSurfaceAlpha) || 0.56));
      const glCtx = initWebGLOverlay();
      if (glCtx && glProg && (glBufs.re || glBufs.im || glBufs.vortex)) {
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

        const rotColMajor = new Float32Array([
          Rtot[0][0], Rtot[1][0], Rtot[2][0],
          Rtot[0][1], Rtot[1][1], Rtot[2][1],
          Rtot[0][2], Rtot[1][2], Rtot[2][2],
        ]);
        const scalePx = Rpx * zoom;
        const aspectX = scalePx / (0.5 * w);
        const aspectY = scalePx / (0.5 * h);

        if ((hasRe && glBufs.re) || (hasIm && glBufs.im)) {
          glCtx.useProgram(glProg.prog);
          glCtx.uniformMatrix3fv(glProg.uRot, false, rotColMajor);
          glCtx.uniform3f(glProg.uCenter, center[0], center[1], center[2]);
          glCtx.uniform1f(glProg.uInvScale, invScale);
          glCtx.uniform1f(glProg.uAspectX, aspectX);
          glCtx.uniform1f(glProg.uAspectY, aspectY);
          glCtx.uniform1f(glProg.uCamDist, 2.8);
          glCtx.uniform3f(glProg.uClipCenter, liveRoi.centerG[0], liveRoi.centerG[1], liveRoi.centerG[2]);
          glCtx.uniform1f(glProg.uClipRadius, liveRoi.radiusG);
          glCtx.uniform1f(glProg.uSlabActive, liveRoi.slabActive ? 1.0 : 0.0);
          glCtx.uniform3f(glProg.uSlabNormal, liveRoi.slabNormal[0], liveRoi.slabNormal[1], liveRoi.slabNormal[2]);
          glCtx.uniform1f(glProg.uSlabOffset, liveRoi.slabOffset);
          glCtx.uniform1f(glProg.uSlabHalfThick, liveRoi.slabHalfThick);

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

          const colRe = [0.06, 0.78, 0.92, surfAlpha];
          const colIm = [0.96, 0.30, 0.48, surfAlpha];
          if (hasRe && glBufs.re) drawPass(glBufs.re, [colRe[0], colRe[1], colRe[2], surfAlpha * 0.78], glCtx.FRONT, true);
          if (hasIm && glBufs.im) drawPass(glBufs.im, [colIm[0], colIm[1], colIm[2], surfAlpha * 0.78], glCtx.FRONT, true);
          if (hasRe && glBufs.re) drawPass(glBufs.re, colRe, glCtx.BACK, false);
          if (hasIm && glBufs.im) drawPass(glBufs.im, colIm, glCtx.BACK, false);
          glCtx.disableVertexAttribArray(glProg.aPos);
          glCtx.disableVertexAttribArray(glProg.aNorm);
        }

        // Clear surface depth buffer so translucent Re=0 / Im=0 sheets never occlude 1D vortices inside the ROI,
        // while keeping depth testing enabled among the 3D vortex filaments themselves!
        if (hasVor && glVortexProg && glBufs.vortex) {
          glCtx.depthMask(true);
          glCtx.clear(glCtx.DEPTH_BUFFER_BIT);
          glCtx.disable(glCtx.CULL_FACE);
          glCtx.useProgram(glVortexProg.prog);
          glCtx.uniformMatrix3fv(glVortexProg.uRot, false, rotColMajor);
          glCtx.uniform3f(glVortexProg.uCenter, center[0], center[1], center[2]);
          glCtx.uniform1f(glVortexProg.uInvScale, invScale);
          glCtx.uniform1f(glVortexProg.uAspectX, aspectX);
          glCtx.uniform1f(glVortexProg.uAspectY, aspectY);
          glCtx.uniform1f(glVortexProg.uCamDist, 2.8);
          glCtx.uniform2f(glVortexProg.uViewportPx, tw, th);
          glCtx.uniform1f(glVortexProg.uHalfWidthPx, (isExp ? 3.1 : 2.45) * dpr);
          glCtx.uniform3f(glVortexProg.uClipCenter, liveRoi.centerG[0], liveRoi.centerG[1], liveRoi.centerG[2]);
          glCtx.uniform1f(glVortexProg.uClipRadius, liveRoi.radiusG);
          glCtx.uniform1f(glVortexProg.uSlabActive, liveRoi.slabActive ? 1.0 : 0.0);
          glCtx.uniform3f(glVortexProg.uSlabNormal, liveRoi.slabNormal[0], liveRoi.slabNormal[1], liveRoi.slabNormal[2]);
          glCtx.uniform1f(glVortexProg.uSlabOffset, liveRoi.slabOffset);
          glCtx.uniform1f(glVortexProg.uSlabHalfThick, liveRoi.slabHalfThick);

          const vb = glBufs.vortex;
          glCtx.bindBuffer(glCtx.ARRAY_BUFFER, vb.b0);
          glCtx.enableVertexAttribArray(glVortexProg.aPos0);
          glCtx.vertexAttribPointer(glVortexProg.aPos0, 3, glCtx.FLOAT, false, 0, 0);
          glCtx.bindBuffer(glCtx.ARRAY_BUFFER, vb.b1);
          glCtx.enableVertexAttribArray(glVortexProg.aPos1);
          glCtx.vertexAttribPointer(glVortexProg.aPos1, 3, glCtx.FLOAT, false, 0, 0);
          glCtx.bindBuffer(glCtx.ARRAY_BUFFER, vb.bSide);
          glCtx.enableVertexAttribArray(glVortexProg.aSide);
          glCtx.vertexAttribPointer(glVortexProg.aSide, 2, glCtx.FLOAT, false, 0, 0);
          glCtx.bindBuffer(glCtx.ELEMENT_ARRAY_BUFFER, vb.ibo);
          glCtx.drawElements(glCtx.TRIANGLES, vb.count, vb.indexType, 0);
          glCtx.disableVertexAttribArray(glVortexProg.aPos0);
          glCtx.disableVertexAttribArray(glVortexProg.aPos1);
          glCtx.disableVertexAttribArray(glVortexProg.aSide);
        }

        glCtx.depthMask(true);
        glCtx.disable(glCtx.CULL_FACE);
        ctx.save();
        ctx.drawImage(glCanvas, 0, 0, w, h);
        ctx.restore();
        return;
      }

      // Software Canvas2D depth-sorted fallback (used in headless / non-WebGL environments)
      const items = [];
      const cx = w * 0.5, cy = h * 0.5;
      function collectSoftwareTris(mesh, baseRgb) {
        if (!mesh || !mesh.nTris) return;
        const stride = Math.max(1, Math.ceil(mesh.nTris / 3500));
        const v = mesh.verts, n = mesh.norms, f = mesh.faces;
        for (let t = 0; t < mesh.nTris; t += stride) {
          const i0 = f[3 * t] * 3, i1 = f[3 * t + 1] * 3, i2 = f[3 * t + 2] * 3;
          const mPt = [(v[i0] + v[i1] + v[i2]) / 3.0, (v[i0 + 1] + v[i1 + 1] + v[i2 + 1]) / 3.0, (v[i0 + 2] + v[i1 + 2] + v[i2 + 2]) / 3.0];
          if (!pointInsideLiveRoi(mPt, liveRoi)) continue;
          const p0 = projectPoint3D([v[i0], v[i0 + 1], v[i0 + 2]], Rtot, center, invScale, cx, cy, Rpx, zoom);
          const p1 = projectPoint3D([v[i1], v[i1 + 1], v[i1 + 2]], Rtot, center, invScale, cx, cy, Rpx, zoom);
          const p2 = projectPoint3D([v[i2], v[i2 + 1], v[i2 + 2]], Rtot, center, invScale, cx, cy, Rpx, zoom);
          const nx = Rtot[0][0] * n[i0] + Rtot[0][1] * n[i0 + 1] + Rtot[0][2] * n[i0 + 2];
          const ny = Rtot[1][0] * n[i0] + Rtot[1][1] * n[i0 + 1] + Rtot[1][2] * n[i0 + 2];
          const nz = Rtot[2][0] * n[i0] + Rtot[2][1] * n[i0 + 1] + Rtot[2][2] * n[i0 + 2];
          const shade = 0.42 + 0.52 * Math.abs(-0.35 * nx + 0.45 * ny + 0.82 * nz);
          items.push({
            kind: "tri",
            z: (p0[2] + p1[2] + p2[2]) / 3.0,
            p0, p1, p2,
            fill: `rgba(${Math.round(baseRgb[0] * shade)},${Math.round(baseRgb[1] * shade)},${Math.round(baseRgb[2] * shade)},${surfAlpha.toFixed(2)})`,
          });
        }
      }
      if (hasRe) collectSoftwareTris(decodedMeshes.re, [16, 205, 238]);
      if (hasIm) collectSoftwareTris(decodedMeshes.im, [246, 78, 125]);
      const vorItems = [];
      if (hasVor) {
        const vList = getActiveVortexList(nodalData);
        for (let vi = 0; vi < vList.length; vi++) {
          const pts = vList[vi] && vList[vi].pts_g;
          if (!Array.isArray(pts) || pts.length < 2) continue;
          for (let pi = 0; pi < pts.length - 1; pi++) {
            const a = pts[pi], b = pts[pi + 1];
            const mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
            if (!pointInsideLiveRoi(mid, liveRoi)) continue;
            const p0 = projectPoint3D(a, Rtot, center, invScale, cx, cy, Rpx, zoom);
            const p1 = projectPoint3D(b, Rtot, center, invScale, cx, cy, Rpx, zoom);
            vorItems.push({ kind: "seg", z: 0.5 * (p0[2] + p1[2]), p0, p1 });
          }
        }
      }
      items.sort((a, b) => a.z - b.z);
      vorItems.sort((a, b) => a.z - b.z);
      ctx.save();
      ctx.lineCap = "round";
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        ctx.beginPath();
        ctx.moveTo(it.p0[0], it.p0[1]);
        ctx.lineTo(it.p1[0], it.p1[1]);
        ctx.lineTo(it.p2[0], it.p2[1]);
        ctx.closePath();
        ctx.fillStyle = it.fill;
        ctx.fill();
      }
      for (let i = 0; i < vorItems.length; i++) {
        const it = vorItems[i];
        ctx.beginPath();
        ctx.moveTo(it.p0[0], it.p0[1]);
        ctx.lineTo(it.p1[0], it.p1[1]);
        ctx.strokeStyle = "rgba(7, 9, 19, 0.88)";
        ctx.lineWidth = isExp ? 4.4 : 3.4;
        ctx.stroke();
        ctx.strokeStyle = "#fde047";
        ctx.lineWidth = isExp ? 2.4 : 1.8;
        ctx.stroke();
      }
      ctx.restore();
    }

    function countLiveVortices(liveRoi) {
      if (!nodalData || !nodalData.has_phases) return 0;
      const vList = getActiveVortexList(nodalData);
      let n = 0;
      for (let vi = 0; vi < vList.length; vi++) {
        const pts = vList[vi] && vList[vi].pts_g;
        if (!Array.isArray(pts) || pts.length < 2) continue;
        for (let pi = 0; pi < pts.length; pi++) {
          if (pointInsideLiveRoi(pts[pi], liveRoi)) {
            n++;
            break;
          }
        }
      }
      return n;
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
      if (!Number.isFinite(panCenterG[0]) || !Number.isFinite(panCenterG[1]) || !Number.isFinite(panCenterG[2])) {
        panCenterG = [0.0, 0.0, 0.0];
      }
      const liveRoi = resolveLiveRoi();
      if (!Number.isFinite(animCenterG[0]) || !Number.isFinite(animCenterG[1]) || !Number.isFinite(animCenterG[2])) {
        animCenterG = [liveRoi.centerG[0], liveRoi.centerG[1], liveRoi.centerG[2]];
      }
      const zoom3d = get3DZoom();

      // Smoothly glide camera center toward liveRoi.centerG when translating between Origin / Peak / Pan
      const tgt = liveRoi.centerG;
      const remDist = Math.hypot(tgt[0] - animCenterG[0], tgt[1] - animCenterG[1], tgt[2] - animCenterG[2]);
      if (remDist > 1e-6) {
        const snapEps = Math.max(1e-5, liveRoi.meanGStar * 0.008);
        if (remDist <= snapEps) {
          animCenterG = [tgt[0], tgt[1], tgt[2]];
        } else {
          const alpha = 0.28;
          animCenterG = [
            animCenterG[0] + (tgt[0] - animCenterG[0]) * alpha,
            animCenterG[1] + (tgt[1] - animCenterG[1]) * alpha,
            animCenterG[2] + (tgt[2] - animCenterG[2]) * alpha,
          ];
        }
      }

      const hovRef = st.hoveredPeakObj ? `${st.hoveredPeakObj.id}:${(st.hoveredPeakObj.hkl || []).join(",")}` : String(st.hoveredPeakId);
      const selRef = st.selectedPeakObj ? `${st.selectedPeakObj.id}:${(st.selectedPeakObj.hkl || []).join(",")}` : String(st.selectedPeakId);
      const zonesKey = st.showZoneArcs ? `${st.selectedZoneId || ""}:${Array.from(st.activeZones || []).join(",")}` : "off";
      const mtzOverlaySig = typeof deps.getMtzOverlaySig === "function" ? deps.getMtzOverlaySig() : "";
      const sig = [
        targetW, targetH,
        (st.wulffYawDeg || 0).toFixed(3),
        (st.wulffPitchDeg || 0).toFixed(3),
        zoom3d.toFixed(4),
        st.wulffExpanded ? 1 : 0,
        st.wulffScope,
        st.wulffColorMode,
        st.colorFullRange ? 1 : 0,
        deps.getColorCrossover().toFixed(3),
        st.showWulffFriedel ? 1 : 0,
        st.showNodalPoints ? 1 : 0,
        st.showNodalSurfacesRe ? 1 : 0,
        st.showNodalSurfacesIm ? 1 : 0,
        st.showPhaseVortices ? 1 : 0,
        (st.nodalSurfaceAlpha || 0.56).toFixed(2),
        liveRoi.mode,
        liveRoi.radiusG.toFixed(5),
        liveRoi.centerG[0].toFixed(5), liveRoi.centerG[1].toFixed(5), liveRoi.centerG[2].toFixed(5),
        animCenterG[0].toFixed(5), animCenterG[1].toFixed(5), animCenterG[2].toFixed(5),
        brickFetchInProgress ? 1 : 0,
        st.dMin, st.lamMin, st.lamMax,
        zonesKey, hovRef, selRef,
        dataSeq,
        mtzOverlaySig,
        decodedCacheSig,
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
      const zoom = zoom3d;
      const isExp = Boolean(st.wulffExpanded);
      const Rtot = buildCameraTransform(Rg);
      const bounds = get3DSceneBounds(liveRoi);
      const center = bounds.center;
      const sceneR = bounds.radius;
      const invScale = 1.0 / sceneR;
      const ub = deps.getActiveUbMatrix();

      // 1. Clean ROI boundary indicator:
      //    - For Origin / Peak / Pan spheres: draw ONLY the outer view-plane silhouette ring (no internal great circles!)
      //    - For Zone Disk: draw the 2 parallel circular rims of the active Laue-zone slab
      function drawCircle3D(c3d, r3d, uVec, vVec, strokeCol, lineW, dashArr) {
        if (!(r3d > 1e-6)) return;
        ctx.save();
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = lineW || 1.1;
        if (dashArr) ctx.setLineDash(dashArr);
        ctx.beginPath();
        const nSeg = 64;
        for (let s = 0; s <= nSeg; s++) {
          const th = (2.0 * Math.PI * s) / nSeg;
          const ct = Math.cos(th), stt = Math.sin(th);
          const gp = [
            c3d[0] + r3d * (ct * uVec[0] + stt * vVec[0]),
            c3d[1] + r3d * (ct * uVec[1] + stt * vVec[1]),
            c3d[2] + r3d * (ct * uVec[2] + stt * vVec[2]),
          ];
          const pr = projectPoint3D(gp, Rtot, center, invScale, cx, cy, Rpx, zoom);
          if (s === 0) ctx.moveTo(pr[0], pr[1]);
          else ctx.lineTo(pr[0], pr[1]);
        }
        ctx.stroke();
        ctx.restore();
      }

      // Camera right & up unit vectors in sample frame (rows 0 and 1 of orthogonal Rtot)
      const camRight = [Rtot[0][0], Rtot[0][1], Rtot[0][2]];
      const camUp = [Rtot[1][0], Rtot[1][1], Rtot[1][2]];
      if (nodalData && nodalData.has_phases) {
        if (liveRoi.slabActive) {
          const nh = liveRoi.slabNormal;
          const ref = Math.abs(nh[0]) < 0.85 ? [1, 0, 0] : [0, 1, 0];
          let e1 = [nh[1] * ref[2] - nh[2] * ref[1], nh[2] * ref[0] - nh[0] * ref[2], nh[0] * ref[1] - nh[1] * ref[0]];
          const e1n = Math.hypot(e1[0], e1[1], e1[2]) || 1.0;
          e1 = [e1[0] / e1n, e1[1] / e1n, e1[2] / e1n];
          const e2 = [nh[1] * e1[2] - nh[2] * e1[1], nh[2] * e1[0] - nh[0] * e1[2], nh[0] * e1[1] - nh[1] * e1[0]];
          const rimR = Math.sqrt(Math.max(0.0, liveRoi.radiusG * liveRoi.radiusG - liveRoi.slabHalfThick * liveRoi.slabHalfThick));
          for (let sgn = -1; sgn <= 1; sgn += 2) {
            const cRim = [
              liveRoi.centerG[0] + sgn * liveRoi.slabHalfThick * nh[0],
              liveRoi.centerG[1] + sgn * liveRoi.slabHalfThick * nh[1],
              liveRoi.centerG[2] + sgn * liveRoi.slabHalfThick * nh[2],
            ];
            drawCircle3D(cRim, rimR, e1, e2, liveRoi.zoneColor || "rgba(56, 189, 248, 0.55)", 1.2, [4, 3]);
          }
        } else {
          drawCircle3D(liveRoi.centerG, liveRoi.radiusG, camRight, camUp, "rgba(56, 189, 248, 0.34)", 1.15, [4, 4]);
          // Subtle center reticle when ROI is translated away from (0,0,0)
          if (Math.hypot(liveRoi.centerG[0], liveRoi.centerG[1], liveRoi.centerG[2]) > 1e-5) {
            const pC = projectPoint3D(liveRoi.centerG, Rtot, center, invScale, cx, cy, Rpx, zoom);
            ctx.save();
            ctx.strokeStyle = "rgba(56, 189, 248, 0.72)";
            ctx.lineWidth = 1.1;
            ctx.beginPath();
            ctx.moveTo(pC[0] - 5, pC[1]); ctx.lineTo(pC[0] + 5, pC[1]);
            ctx.moveTo(pC[0], pC[1] - 5); ctx.lineTo(pC[0], pC[1] + 5);
            ctx.stroke();
            ctx.restore();
          }
        }
      } else {
        drawCircle3D([0, 0, 0], sceneR, camRight, camUp, "rgba(56, 189, 248, 0.24)", 1.0, [4, 4]);
      }

      // 2. Active Laue Zone reciprocal planes (nodal planes G . t_uvw = 0) & zone-intersection lines
      const actZones = deps.getActiveZoneObjects();
      if (st.showZoneArcs && ub && Array.isArray(actZones) && actZones.length > 0) {
        const invUb = deps.mat3Inv(ub);
        if (invUb) {
          const planeR = (nodalData && nodalData.has_phases) ? liveRoi.radiusG : sceneR * 0.92;
          const planeCenter = (nodalData && nodalData.has_phases) ? liveRoi.centerG : [0, 0, 0];
          const zNormals = [];
          ctx.save();
          for (let zi = 0; zi < Math.min(actZones.length, 6); zi++) {
            const z = actZones[zi];
            if (!Array.isArray(z.uvw) || z.uvw.length !== 3) continue;
            const u = z.uvw[0], v = z.uvw[1], ww = z.uvw[2];
            const tx = invUb[0][0] * u + invUb[1][0] * v + invUb[2][0] * ww;
            const ty = invUb[0][1] * u + invUb[1][1] * v + invUb[2][1] * ww;
            const tz = invUb[0][2] * u + invUb[1][2] * v + invUb[2][2] * ww;
            const tn = Math.hypot(tx, ty, tz);
            if (tn < 1e-8) continue;
            const nHat = [tx / tn, ty / tn, tz / tn];
            zNormals.push({ nHat, color: z.rainbow_color || "#38bdf8", label: z.label });
            const ref = Math.abs(nHat[0]) < 0.85 ? [1, 0, 0] : [0, 1, 0];
            let e1x = nHat[1] * ref[2] - nHat[2] * ref[1];
            let e1y = nHat[2] * ref[0] - nHat[0] * ref[2];
            let e1z = nHat[0] * ref[1] - nHat[1] * ref[0];
            const e1n = Math.hypot(e1x, e1y, e1z) || 1.0;
            e1x /= e1n; e1y /= e1n; e1z /= e1n;
            const e2x = nHat[1] * e1z - nHat[2] * e1y;
            const e2y = nHat[2] * e1x - nHat[0] * e1z;
            const e2z = nHat[0] * e1y - nHat[1] * e1x;

            const dOff = planeCenter[0] * nHat[0] + planeCenter[1] * nHat[1] + planeCenter[2] * nHat[2];
            if (Math.abs(dOff) < planeR * 0.96) {
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
                const pA = projectPoint3D([planeCenter[0] - ux, planeCenter[1] - uy, planeCenter[2] - uz], Rtot, center, invScale, cx, cy, Rpx, zoom);
                const pB = projectPoint3D([planeCenter[0] + ux, planeCenter[1] + uy, planeCenter[2] + uz], Rtot, center, invScale, cx, cy, Rpx, zoom);
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

      // 3. Depth-buffered Two-color 2D Nodal Surfaces (Re=0 Cyan & Im=0 Coral-Rose) + 1D Phase Vortices
      renderSurfacesAndVortices3D(ctx, w, h, dpr, Rtot, center, invScale, Rpx, zoom, liveRoi, isExp);

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

      // 5. 3D Bragg Reflections at G = UB * hkl (depth-sorted, interactive hover/click!)
      screenPeakHits3D = [];
      if (ub) {
        const lodState = deps.getWulffVisiblePool(w, h);
        const peaks = lodState.visiblePeaks || [];
        const drawList = [];
        const maxClipR = Math.max(sceneR * 1.45, liveRoi.radiusG * 1.35);
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
          if (pr[4]) {
            drawList.push({ p, x: pr[0], y: pr[1], z: pr[2], sc: pr[3], isFocus });
          }
          if (st.showWulffFriedel) {
            const prF = projectPoint3D([-gx, -gy, -gz], Rtot, center, invScale, cx, cy, Rpx, zoom);
            const distCF = Math.hypot(-gx - center[0], -gy - center[1], -gz - center[2]);
            if (distCF <= maxClipR && prF[4]) {
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

      // 6. Legend & Status Pill Overlay in 3D View
      ctx.save();
      ctx.font = isExp ? "bold 9.5px monospace" : "bold 8px monospace";
      const pills = [];
      if (nodalData && nodalData.has_phases) {
        const fLbl = nodalData.formula_label || "F(G)";
        const nVor = countLiveVortices(liveRoi);
        const roiTag = liveRoi.slabActive
          ? `Zone ${liveRoi.zoneLabel || "disk"}`
          : `${liveRoi.mult.toFixed(1)}×a⋆`;
        const zoomTag = Math.abs(zoom - 1.0) > 0.03 ? ` • ${zoom.toFixed(2)}×` : "";
        if (st.showNodalSurfacesRe) pills.push({ txt: `Re[${fLbl}]=0 • ${roiTag}${zoomTag}`, col: "#22d3ee" });
        if (st.showNodalSurfacesIm) {
          pills.push({
            txt: nodalData.is_real_only ? `Im[${fLbl}]≡0 (centric)` : `Im[${fLbl}]=0`,
            col: "#fb7185",
          });
        }
        if (st.showPhaseVortices) pills.push({ txt: `1D Vortices (${nVor} in ROI)`, col: "#fde047" });
        if (brickFetchInProgress) pills.push({ txt: "Streaming 3D brick…", col: "#94a3b8" });
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

    // Project 1D phase vortices onto the Convex 3D Wulff Sphere (radial unit-vector projection G / |G|),
    // clipped live to the active 3D ROI sphere / Laue-zone disk!
    function renderWulffVortices(ctx, cx, cy, R, isExp) {
      const st = deps.state;
      if (!nodalData || !nodalData.has_phases || !st.showPhaseVortices) return;
      const vList = getActiveVortexList(nodalData);
      if (!Array.isArray(vList) || vList.length === 0) return;

      const liveRoi = resolveLiveRoi();
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

          for (let pass = 0; pass < 2; pass++) {
            const wantHemi = pass === 0 ? -1 : 1;
            ctx.beginPath();
            let penDown = false;
            for (let pi = 0; pi < pts.length; pi++) {
              const rawPt = pts[pi];
              if (!pointInsideLiveRoi(rawPt, liveRoi)) { penDown = false; continue; }
              const g = [sgn * rawPt[0], sgn * rawPt[1], sgn * rawPt[2]];
              const gn = Math.hypot(g[0], g[1], g[2]);
              if (!(gn > 1e-6)) { penDown = false; continue; }
              if (1.0 / gn < (Number(st.dMin) || 0.5) * 0.98) { penDown = false; continue; }
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

    // Project 1D phase vortices onto the 2D Detector Panels via exact Laue Bragg raycasting,
    // clipped live to the active 3D ROI sphere / Laue-zone disk!
    function renderDetectorVortices(ctx, w, h) {
      const st = deps.state;
      if (!nodalData || !nodalData.has_phases || !st.showPhaseVortices) return;
      const vList = getActiveVortexList(nodalData);
      if (!Array.isArray(vList) || vList.length === 0) return;

      const liveRoi = resolveLiveRoi();
      const Rg = deps.getActiveGoniometerR();
      const ki = deps.getActiveIncidentBeam();
      const sLab = deps.getActiveSampleOffsetLab();
      const fullPanels = deps.buildFullFaceDetectorRaycastPanels(sLab);
      const isSph = deps.isSphericalDetectorView();
      const lMin = Number(st.lamMin) || 0.0;
      const lMax = Number(st.lamMax) || 99.0;
      const dMinCut = (Number(st.dMin) || 0.5) * 0.98;
      const azShifts = [-360.0, 0.0, 360.0];

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
            const rawPt = pts[pi];
            if (!pointInsideLiveRoi(rawPt, liveRoi)) {
              if (curSeg.length >= 2) projectedSegs.push({ pts: curSeg, isMate: sgn < 0 });
              curSeg = []; prevAz = null; prevBank = null;
              continue;
            }
            const gx = sgn * rawPt[0], gy = sgn * rawPt[1], gz = sgn * rawPt[2];
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

    function brickCoversRoi(nd, liveRoi) {
      if (!nd || !nd.has_phases) return false;
      const bc = (nd.brick && Array.isArray(nd.brick.center_g))
        ? nd.brick.center_g
        : ((nd.roi && Array.isArray(nd.roi.center_g)) ? nd.roi.center_g : [0, 0, 0]);
      const br = (nd.brick && Number(nd.brick.radius_g) > 0)
        ? Number(nd.brick.radius_g)
        : ((nd.roi && Number(nd.roi.radius_g) > 0) ? Number(nd.roi.radius_g) : 0.0);
      const d = Math.hypot(liveRoi.centerG[0] - bc[0], liveRoi.centerG[1] - bc[1], liveRoi.centerG[2] - bc[2]);
      return (d + liveRoi.radiusG) <= br * 1.03;
    }

    function findCachedBrickCovering(liveRoi, wantRefName, wantHetMode) {
      if (!nodalData || !nodalData.has_phases) return null;
      const pName = nodalData.primary_name || "";
      for (const cand of brickCache.values()) {
        if (!cand || !cand.has_phases || (cand.primary_name || "") !== pName) continue;
        const candRef = cand.ref_has_phases ? (cand.ref_name || "") : "";
        if (candRef !== (wantRefName || "")) continue;
        if ((cand.heterodyne_mode || "single") !== (wantHetMode || "single")) continue;
        if (brickCoversRoi(cand, liveRoi)) return cand;
      }
      return null;
    }

    async function requestNodalUpdate(extraBody) {
      const st = deps.state;
      const reqSeq = ++fetchInFlight;
      brickFetchInProgress = true;
      last3DRenderSig = "";
      const selPeak = st.selectedPeakObj;
      const actZones = deps.getActiveZoneObjects();
      const selZone = actZones.find(z => z.zone_id === st.selectedZoneId) || actZones[0] || null;
      const body = Object.assign({
        heterodyne_mode: st.heterodyneMode || "delta",
        roi_mode: st.nodalRoiMode || "origin",
        roi_center_hkl: (selPeak && Array.isArray(selPeak.hkl)) ? selPeak.hkl : null,
        roi_center_g: st.nodalRoiMode === "pan" ? panCenterG.slice() : null,
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
        brickFetchInProgress = false;
        if (r.ok && doc) {
          if (doc.nodal_data !== undefined) {
            setNodalData(doc.nodal_data);
          } else {
            last3DRenderSig = "";
          }
          deps.onUpdated(doc.nodal_data !== undefined ? doc.nodal_data : nodalData, doc);
        } else if (doc && doc.error && typeof deps.setStatusLine === "function") {
          deps.setStatusLine("[NODAL SURFACES]", `<span style="color:#f87171;">${doc.error}</span>`);
        }
        return doc;
      } catch (_err) {
        if (reqSeq === fetchInFlight) brickFetchInProgress = false;
        return null;
      }
    }

    // Swap immediately to a cached heterodyne brick if available before/during server sync
    function applyCachedHeterodyneMode(wantMode) {
      if (!nodalData || !nodalData.has_phases) return null;
      const curRef = nodalData.ref_has_phases ? (nodalData.ref_name || "") : "";
      const liveRoi = resolveLiveRoi();
      const hit = findCachedBrickCovering(liveRoi, curRef, wantMode);
      if (hit) {
        setNodalData(hit);
        deps.onUpdated(hit);
        return hit;
      }
      return null;
    }
    // and only fetches a new 3D brick in the background if the user moves beyond cached bounds!
    function onLiveRoiChanged() {
      last3DRenderSig = "";
      if (!nodalData || !nodalData.has_phases) return;
      const liveRoi = resolveLiveRoi();
      if (brickCoversRoi(nodalData, liveRoi)) {
        if (roiDebounceTimer) {
          clearTimeout(roiDebounceTimer);
          roiDebounceTimer = null;
        }
        return;
      }
      const curRef = nodalData.ref_has_phases ? (nodalData.ref_name || "") : "";
      const curHet = nodalData.heterodyne_mode || "single";
      const hit = findCachedBrickCovering(liveRoi, curRef, curHet);
      if (hit) {
        if (roiDebounceTimer) {
          clearTimeout(roiDebounceTimer);
          roiDebounceTimer = null;
        }
        setNodalData(hit);
        deps.onUpdated(hit);
        return;
      }
      scheduleRoiRefresh(90);
    }

    function scheduleRoiRefresh(delayMs) {
      if (!nodalData || !nodalData.has_phases) return;
      if (roiDebounceTimer) clearTimeout(roiDebounceTimer);
      roiDebounceTimer = setTimeout(() => {
        roiDebounceTimer = null;
        requestNodalUpdate();
      }, typeof delayMs === "number" ? delayMs : 110);
    }

    // Translate 3D reciprocal-space ROI smoothly via Shift+drag / right-drag on the 3D canvas
    function panByScreenDelta(dxPx, dyPx, canvasW, canvasH) {
      const dx = Math.max(-80.0, Math.min(80.0, Number(dxPx) || 0.0));
      const dy = Math.max(-80.0, Math.min(80.0, Number(dyPx) || 0.0));
      if (dx === 0.0 && dy === 0.0) return;
      const cw = (Number.isFinite(canvasW) && canvasW > 20.0) ? canvasW : 360.0;
      const ch = (Number.isFinite(canvasH) && canvasH > 20.0) ? canvasH : 360.0;
      const st = deps.state;
      if (!Number.isFinite(panCenterG[0]) || !Number.isFinite(panCenterG[1]) || !Number.isFinite(panCenterG[2])) {
        panCenterG = [0.0, 0.0, 0.0];
      }
      const curRoi = resolveLiveRoi();
      if (st.nodalRoiMode !== "pan") {
        panCenterG = [
          Number.isFinite(curRoi.centerG[0]) ? curRoi.centerG[0] : 0.0,
          Number.isFinite(curRoi.centerG[1]) ? curRoi.centerG[1] : 0.0,
          Number.isFinite(curRoi.centerG[2]) ? curRoi.centerG[2] : 0.0,
        ];
        st.nodalRoiMode = "pan";
      }
      const Rg = deps.getActiveGoniometerR();
      const Rtot = buildCameraTransform(Rg);
      const bounds = get3DSceneBounds(curRoi);
      const Rpx = Math.min(cw, ch) * 0.41;
      const zoom = get3DZoom();
      const D = 2.8;
      const persp0 = Math.sqrt(D * D - 1.0) / D;
      const pxPerNorm = Math.max(12.0, Rpx * zoom * persp0);
      const dCamX = -dx / pxPerNorm;
      const dCamY = dy / pxPerNorm;
      const scaleG = bounds.radius;
      let nx = panCenterG[0] + scaleG * (Rtot[0][0] * dCamX + Rtot[1][0] * dCamY);
      let ny = panCenterG[1] + scaleG * (Rtot[0][1] * dCamX + Rtot[1][1] * dCamY);
      let nz = panCenterG[2] + scaleG * (Rtot[0][2] * dCamX + Rtot[1][2] * dCamY);
      const maxPanR = Math.max(0.20, bounds.invDmin * 1.25);
      const nNorm = Math.hypot(nx, ny, nz);
      if (nNorm > maxPanR && nNorm > 1e-8) {
        const k = maxPanR / nNorm;
        nx *= k; ny *= k; nz *= k;
      }
      panCenterG = [nx, ny, nz];
      animCenterG = [nx, ny, nz];
      onLiveRoiChanged();
    }

    function resetPanToOrigin() {
      panCenterG = [0.0, 0.0, 0.0];
      animCenterG = [0.0, 0.0, 0.0];
      if (deps.state.nodalRoiMode === "pan") {
        deps.state.nodalRoiMode = "origin";
      }
      last3DRenderSig = "";
      onLiveRoiChanged();
    }

    function resetView3D() {
      panCenterG = [0.0, 0.0, 0.0];
      animCenterG = [0.0, 0.0, 0.0];
      deps.state.recip3dZoom = 1.0;
      if (deps.state.nodalRoiMode === "pan") {
        deps.state.nodalRoiMode = "origin";
      }
      last3DRenderSig = "";
      onLiveRoiChanged();
    }

    // Instant client-side clear of Ref F₀ + server sync
    async function clearRefMtz() {
      const st = deps.state;
      if (roiDebounceTimer) {
        clearTimeout(roiDebounceTimer);
        roiDebounceTimer = null;
      }
      st.heterodyneMode = "single";
      const clearedMeta = {
        clear_ref_mtz: true,
        ref_mtz_overlay_name: null,
        ref_mtz_i_asu_lookup: null,
        ref_mtz_i_exact_lookup: null,
      };
      // Immediately swap back to cached single-field brick in 0 ms if available!
      if (lastSingleOriginData && nodalData && lastSingleOriginData.primary_name === nodalData.primary_name) {
        setNodalData(lastSingleOriginData);
        deps.onUpdated(lastSingleOriginData, clearedMeta);
      } else if (nodalData) {
        const clearedLocal = Object.assign({}, nodalData, {
          ref_name: null,
          ref_has_phases: false,
          ref_f_column: null,
          ref_phi_column: null,
          ref_warning: null,
          heterodyne_mode: "single",
          formula_label: "F(G)",
        });
        setNodalData(clearedLocal);
        deps.onUpdated(clearedLocal, clearedMeta);
      } else {
        setNodalData(null);
        deps.onUpdated(null, clearedMeta);
      }
      const doc = await requestNodalUpdate({
        heterodyne_mode: "single",
        clear_ref_mtz: true,
      });
      return (doc && doc.nodal_data) ? doc.nodal_data : nodalData;
    }

    async function uploadAndSetRefMtz(file, onLoadingState) {
      if (!file) return;
      if (typeof onLoadingState === "function") onLoadingState(true, file.name);
      if (typeof deps.setStatusLine === "function") {
        deps.setStatusLine("[REF F₀ MTZ]", `Loading reference <strong>${file.name}</strong>...`);
      }
      try {
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
        const doc = await requestNodalUpdate({ ref_mtz_path: resolvedPath, heterodyne_mode: "auto" });
        if (doc && doc.ok && typeof deps.setStatusLine === "function") {
          const nd = doc.nodal_data;
          if (nd && nd.has_phases && nd.ref_has_phases) {
            deps.setStatusLine(
              "[HETERODYNE F₀]",
              `Loaded <strong>${nd.ref_name}</strong> (${nd.ref_f_column}, ${nd.ref_phi_column}) • Interference: <strong>${nd.formula_label}</strong> (Spot Color: <strong>ΔI (F−F₀)</strong> also available)`
            );
          } else if (doc.ref_mtz_i_asu_lookup) {
            const refNm = doc.ref_mtz_overlay_name || file.name;
            const nRefl = Number(doc.ref_mtz_reflections || Object.keys(doc.ref_mtz_i_asu_lookup).length).toLocaleString();
            deps.setStatusLine(
              "[REF F₀ CONTRAST]",
              `Loaded <strong>${refNm}</strong> (${nRefl} ASU reflections) • Coloring spots by <strong>ΔI (F − F₀)</strong>`
            );
          } else if (nd && nd.ref_warning) {
            deps.setStatusLine("[REF F₀ MTZ]", `<span style="color:#fbbf24;">${nd.ref_warning}</span>`);
          }
        }
      } finally {
        if (typeof onLoadingState === "function") onLoadingState(false, file.name);
      }
    }

    function setNodalData(nextData) {
      nodalData = nextData || null;
      if (nodalData && nodalData.has_phases) {
        storeBrickInCache(nodalData);
      }
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
      resolveLiveRoi,
      applyCachedHeterodyneMode,
      onLiveRoiChanged,
      panByScreenDelta,
      resetPanToOrigin,
      resetView3D,
      clearRefMtz,
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
