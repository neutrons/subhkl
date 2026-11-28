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
_GLOBAL_MORSE_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
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


def _trilinear_sample_volume(vol: np.ndarray, pts_idx: np.ndarray) -> np.ndarray:
    """Vectorized trilinear interpolation of 3D array `vol` at fractional index coordinates `(N, 3)`."""
    if len(pts_idx) == 0:
        return np.zeros(0, dtype=np.float64)
    g0, g1, g2 = vol.shape
    fx = np.clip(pts_idx[:, 0], 0.0, max(0.0, g0 - 1.000001))
    fy = np.clip(pts_idx[:, 1], 0.0, max(0.0, g1 - 1.000001))
    fz = np.clip(pts_idx[:, 2], 0.0, max(0.0, g2 - 1.000001))
    i0 = np.floor(fx).astype(np.intp)
    j0 = np.floor(fy).astype(np.intp)
    k0 = np.floor(fz).astype(np.intp)
    i1 = np.minimum(i0 + 1, g0 - 1)
    j1 = np.minimum(j0 + 1, g1 - 1)
    k1 = np.minimum(k0 + 1, g2 - 1)
    tx = fx - i0
    ty = fy - j0
    tz = fz - k0
    c000 = vol[i0, j0, k0]
    c100 = vol[i1, j0, k0]
    c010 = vol[i0, j1, k0]
    c110 = vol[i1, j1, k0]
    c001 = vol[i0, j0, k1]
    c101 = vol[i1, j0, k1]
    c011 = vol[i0, j1, k1]
    c111 = vol[i1, j1, k1]
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return np.asarray(c0 * (1.0 - tz) + c1 * tz, dtype=np.float64)


_NEIGHBOR_INT_OFFSETS_5 = np.stack(
    np.meshgrid(
        np.arange(-2, 3, dtype=np.float64),
        np.arange(-2, 3, dtype=np.float64),
        np.arange(-2, 3, dtype=np.float64),
        indexing="ij",
    ),
    axis=-1,
).reshape(-1, 3)


def extract_morse_maxima_3d(
    Z: np.ndarray,
    h0: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    ub_matrix: np.ndarray,
    *,
    mean_g_star: float = 0.05,
    g_max: float | None = None,
    roi_center_g: np.ndarray | None = None,
    roi_radius_g: float | None = None,
    max_extrema: int = 240,
    bragg_mahal_cap: float | None = None,
) -> list[dict[str, Any]]:
    """Extract sub-voxel 3D Morse maxima of `|Re(W)|` (`∇Re W = 0` with definite Hessian) and 3D capsule geometry.

    Positive antinodal maxima (`Re W > 0`) require a strictly negative-definite 3x3 Hessian (Morse index 3),
    while negative antinodal extrema (`Re W < 0`, local maxima of `-Re W`) require a strictly positive-definite
    Hessian (Morse index 0 of `Re W`), rejecting all saddle points. Principal curvatures of the reciprocal-space
    Hessian `H_G = UB^{-T} H_h UB^{-1}` orient each 3D capsule along the softest-decay ridge eigenvector and
    yield the exact dimensionless Miller-space Mahalanobis precision tensor `P_h = -∇_h^2 Re(W) / Re(W)`
    (satisfying `(h - h*)^T P_h (h - h*) == (G - G*)^T Σ_G^{-1} (G - G*)`).
    """
    u = np.asarray(Z.real, dtype=np.float64)
    v_im = np.asarray(Z.imag, dtype=np.float64)
    if u.ndim != 3 or min(u.shape) < 3:
        return []
    g0, g1, g2 = u.shape
    u_abs_max = float(np.max(np.abs(u)))
    if not np.isfinite(u_abs_max) or u_abs_max <= 1e-12:
        return []

    u_c = u[1:-1, 1:-1, 1:-1]
    max_nb = np.full_like(u_c, -np.inf)
    min_nb = np.full_like(u_c, np.inf)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                nb = u[
                    1 + di : g0 - 1 + di,
                    1 + dj : g1 - 1 + dj,
                    1 + dk : g2 - 1 + dk,
                ]
                np.maximum(max_nb, nb, out=max_nb)
                np.minimum(min_nb, nb, out=min_nb)

    floor_val = 1e-5 * u_abs_max
    cand_mask = ((u_c > floor_val) & (u_c >= max_nb)) | (
        (u_c < -floor_val) & (u_c <= min_nb)
    )
    cand_ijk = np.argwhere(cand_mask) + 1
    if len(cand_ijk) == 0:
        return []

    dh0 = float(h0[1] - h0[0]) if len(h0) > 1 else 1.0
    dh1 = float(h1[1] - h1[0]) if len(h1) > 1 else 1.0
    dh2 = float(h2[1] - h2[0]) if len(h2) > 1 else 1.0
    inv_dh = np.array(
        [
            1.0 / max(abs(dh0), 1e-12),
            1.0 / max(abs(dh1), 1e-12),
            1.0 / max(abs(dh2), 1e-12),
        ],
        dtype=np.float64,
    )

    ub = np.asarray(ub_matrix, dtype=np.float64)
    ub_inv = np.linalg.inv(ub)
    ub_inv_t = ub_inv.T
    g_scale = max(float(mean_g_star), 1e-5)
    vox_step_g = float(
        np.linalg.norm(ub @ np.array([abs(dh0), abs(dh1), abs(dh2)], dtype=np.float64))
    )

    ii = cand_ijk[:, 0]
    jj = cand_ijk[:, 1]
    kk = cand_ijk[:, 2]
    h_coarse = np.column_stack([h0[ii], h1[jj], h2[kk]])
    g_coarse = h_coarse @ ub.T
    keep_pre = np.ones(len(cand_ijk), dtype=bool)
    if g_max is not None and g_max > 0.0:
        keep_pre &= np.linalg.norm(g_coarse, axis=1) <= float(g_max) * 1.03 + vox_step_g
    if roi_center_g is not None and roi_radius_g is not None and roi_radius_g > 0.0:
        keep_pre &= (
            np.linalg.norm(g_coarse - np.asarray(roi_center_g)[None, :], axis=1)
            <= float(roi_radius_g) * 1.03 + vox_step_g
        )
    if not np.any(keep_pre):
        return []
    ii = ii[keep_pre]
    jj = jj[keep_pre]
    kk = kk[keep_pre]

    u0 = u[ii, jj, kk]
    sgn = np.where(u0 > 0.0, 1.0, -1.0)

    gx2 = 0.5 * (u[ii + 1, jj, kk] - u[ii - 1, jj, kk])
    gy2 = 0.5 * (u[ii, jj + 1, kk] - u[ii, jj - 1, kk])
    gz2 = 0.5 * (u[ii, jj, kk + 1] - u[ii, jj, kk - 1])
    hxx2 = u[ii + 1, jj, kk] - 2.0 * u0 + u[ii - 1, jj, kk]
    hyy2 = u[ii, jj + 1, kk] - 2.0 * u0 + u[ii, jj - 1, kk]
    hzz2 = u[ii, jj, kk + 1] - 2.0 * u0 + u[ii, jj, kk - 1]

    in4_x = (ii >= 2) & (ii <= g0 - 3)
    in4_y = (jj >= 2) & (jj <= g1 - 3)
    in4_z = (kk >= 2) & (kk <= g2 - 3)
    if np.any(in4_x):
        ix = ii[in4_x]
        jx = jj[in4_x]
        kx = kk[in4_x]
        gx2[in4_x] = (
            -u[ix + 2, jx, kx]
            + 8.0 * u[ix + 1, jx, kx]
            - 8.0 * u[ix - 1, jx, kx]
            + u[ix - 2, jx, kx]
        ) / 12.0
        hxx2[in4_x] = (
            -u[ix + 2, jx, kx]
            + 16.0 * u[ix + 1, jx, kx]
            - 30.0 * u[ix, jx, kx]
            + 16.0 * u[ix - 1, jx, kx]
            - u[ix - 2, jx, kx]
        ) / 12.0
    if np.any(in4_y):
        iy = ii[in4_y]
        jy = jj[in4_y]
        ky = kk[in4_y]
        gy2[in4_y] = (
            -u[iy, jy + 2, ky]
            + 8.0 * u[iy, jy + 1, ky]
            - 8.0 * u[iy, jy - 1, ky]
            + u[iy, jy - 2, ky]
        ) / 12.0
        hyy2[in4_y] = (
            -u[iy, jy + 2, ky]
            + 16.0 * u[iy, jy + 1, ky]
            - 30.0 * u[iy, jy, ky]
            + 16.0 * u[iy, jy - 1, ky]
            - u[iy, jy - 2, ky]
        ) / 12.0
    if np.any(in4_z):
        iz = ii[in4_z]
        jz = jj[in4_z]
        kz = kk[in4_z]
        gz2[in4_z] = (
            -u[iz, jz, kz + 2]
            + 8.0 * u[iz, jz, kz + 1]
            - 8.0 * u[iz, jz, kz - 1]
            + u[iz, jz, kz - 2]
        ) / 12.0
        hzz2[in4_z] = (
            -u[iz, jz, kz + 2]
            + 16.0 * u[iz, jz, kz + 1]
            - 30.0 * u[iz, jz, kz]
            + 16.0 * u[iz, jz, kz - 1]
            - u[iz, jz, kz - 2]
        ) / 12.0

    hxy = 0.25 * (
        u[ii + 1, jj + 1, kk]
        - u[ii + 1, jj - 1, kk]
        - u[ii - 1, jj + 1, kk]
        + u[ii - 1, jj - 1, kk]
    )
    hxz = 0.25 * (
        u[ii + 1, jj, kk + 1]
        - u[ii + 1, jj, kk - 1]
        - u[ii - 1, jj, kk + 1]
        + u[ii - 1, jj, kk - 1]
    )
    hyz = 0.25 * (
        u[ii, jj + 1, kk + 1]
        - u[ii, jj + 1, kk - 1]
        - u[ii, jj - 1, kk + 1]
        + u[ii, jj - 1, kk - 1]
    )

    g_xi = np.column_stack([gx2, gy2, gz2])
    n_c = len(ii)
    h_xi = np.empty((n_c, 3, 3), dtype=np.float64)
    h_xi[:, 0, 0] = hxx2
    h_xi[:, 1, 1] = hyy2
    h_xi[:, 2, 2] = hzz2
    h_xi[:, 0, 1] = h_xi[:, 1, 0] = hxy
    h_xi[:, 0, 2] = h_xi[:, 2, 0] = hxz
    h_xi[:, 1, 2] = h_xi[:, 2, 1] = hyz

    eig_tol = 1e-8 * u_abs_max
    m_xi = -sgn[:, None, None] * h_xi
    evals_xi = np.linalg.eigvalsh(m_xi)
    def_mask = evals_xi[:, 0] > eig_tol
    if not np.any(def_mask):
        return []

    ii = ii[def_mask]
    jj = jj[def_mask]
    kk = kk[def_mask]
    u0 = u0[def_mask]
    sgn = sgn[def_mask]
    g_xi = g_xi[def_mask]
    h_xi = h_xi[def_mask]
    m_xi = m_xi[def_mask]

    d_xi = -np.linalg.solve(h_xi, g_xi[..., None])[..., 0]
    u_star = u0 + 0.5 * np.sum(g_xi * d_xi, axis=1)
    step_ok = (np.max(np.abs(d_xi), axis=1) <= 0.85) & (sgn * u_star > floor_val)
    if not np.any(step_ok):
        return []

    ii = ii[step_ok]
    jj = jj[step_ok]
    kk = kk[step_ok]
    sgn = sgn[step_ok]
    d_xi = d_xi[step_ok]
    u_star = u_star[step_ok]
    m_xi = m_xi[step_ok]

    vox_pos = np.column_stack([ii + d_xi[:, 0], jj + d_xi[:, 1], kk + d_xi[:, 2]])
    h_star = np.column_stack(
        [
            h0[ii] + d_xi[:, 0] * dh0,
            h1[jj] + d_xi[:, 1] * dh1,
            h2[kk] + d_xi[:, 2] * dh2,
        ]
    )
    g_star = h_star @ ub.T
    g_norm = np.linalg.norm(g_star, axis=1)

    geom_ok = g_norm > 1e-6
    if g_max is not None and g_max > 0.0:
        geom_ok &= g_norm <= float(g_max) * 1.02
    if roi_center_g is not None and roi_radius_g is not None and roi_radius_g > 0.0:
        geom_ok &= (
            np.linalg.norm(g_star - np.asarray(roi_center_g)[None, :], axis=1)
            <= float(roi_radius_g) * 1.02
        )
    if not np.any(geom_ok):
        return []

    ii = ii[geom_ok]
    jj = jj[geom_ok]
    kk = kk[geom_ok]
    sgn = sgn[geom_ok]
    d_xi = d_xi[geom_ok]
    u_star = u_star[geom_ok]
    m_xi = m_xi[geom_ok]
    vox_pos = vox_pos[geom_ok]
    h_star = h_star[geom_ok]
    g_star = g_star[geom_ok]

    abs_u = np.abs(u_star)
    m_h = m_xi * (inv_dh[None, :, None] * inv_dh[None, None, :])
    m_h = 0.5 * (m_h + np.swapaxes(m_h, -1, -2))
    prec_h = m_h / np.maximum(abs_u, 1e-12)[:, None, None]

    base_hkl = np.round(h_star)
    cand_hkl = base_hkl[:, None, :] + _NEIGHBOR_INT_OFFSETS_5[None, :, :]
    delta_h = cand_hkl - h_star[:, None, :]
    d2_all = np.einsum("nki,nij,nkj->nk", delta_h, prec_h, delta_h)
    best_k = np.argmin(d2_all, axis=1)
    min_d2 = np.maximum(0.0, d2_all[np.arange(len(h_star)), best_k])
    min_mahal = np.sqrt(min_d2)
    nearest_hkl_arr = np.round(cand_hkl[np.arange(len(h_star)), best_k]).astype(
        np.int32
    )

    if bragg_mahal_cap is not None and bragg_mahal_cap > 0.0:
        m_ok = min_mahal <= float(bragg_mahal_cap)
        if not np.any(m_ok):
            return []
        ii = ii[m_ok]
        jj = jj[m_ok]
        kk = kk[m_ok]
        sgn = sgn[m_ok]
        d_xi = d_xi[m_ok]
        u_star = u_star[m_ok]
        abs_u = abs_u[m_ok]
        m_h = m_h[m_ok]
        prec_h = prec_h[m_ok]
        vox_pos = vox_pos[m_ok]
        h_star = h_star[m_ok]
        g_star = g_star[m_ok]
        min_mahal = min_mahal[m_ok]
        nearest_hkl_arr = nearest_hkl_arr[m_ok]

    im_grad = np.column_stack(
        [
            0.5 * (v_im[ii + 1, jj, kk] - v_im[ii - 1, jj, kk]),
            0.5 * (v_im[ii, jj + 1, kk] - v_im[ii, jj - 1, kk]),
            0.5 * (v_im[ii, jj, kk + 1] - v_im[ii, jj, kk - 1]),
        ]
    )
    im_star = v_im[ii, jj, kk] + np.sum(im_grad * d_xi, axis=1)

    m_g = np.einsum("ai,nij,bj->nab", ub_inv_t, m_h, ub_inv)
    m_g = 0.5 * (m_g + np.swapaxes(m_g, -1, -2))
    kap, vecs = np.linalg.eigh(m_g)
    axis_g_all = vecs[:, :, 0]
    ax_norms = np.linalg.norm(axis_g_all, axis=1)
    valid_ax = ax_norms > 1e-12
    axis_g_all[valid_ax] /= ax_norms[valid_ax, None]
    axis_g_all[~valid_ax] = np.array([0.0, 0.0, 1.0])

    kap_soft = np.maximum(kap[:, 0], 1e-12)
    kap_perp = np.sqrt(np.maximum(kap[:, 1], 1e-12) * np.maximum(kap[:, 2], 1e-12))
    r_raw = np.sqrt(2.0 * 0.12 * abs_u / kap_perp)
    a_raw = np.sqrt(2.0 * 0.12 * abs_u / kap_soft)
    aspect_all = np.clip(a_raw / np.maximum(r_raw, 1e-9), 1.0, 3.2)
    r_cap_all = np.clip(r_raw, 0.046 * g_scale, 0.112 * g_scale)
    half_len_all = np.clip(r_cap_all * (aspect_all - 1.0), 0.0, 0.21 * g_scale)
    dhkl_all = np.linalg.norm(
        h_star - nearest_hkl_arr.astype(np.float64),
        axis=1,
    )

    order = np.argsort(-abs_u)
    kept: list[dict[str, Any]] = []
    vox_buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for idx in order:
        vp = vox_pos[idx]
        bk = (math.floor(vp[0]), math.floor(vp[1]), math.floor(vp[2]))
        dup = False
        for b0 in (bk[0] - 1, bk[0], bk[0] + 1):
            for b1 in (bk[1] - 1, bk[1], bk[1] + 1):
                for b2 in (bk[2] - 1, bk[2], bk[2] + 1):
                    lst = vox_buckets.get((b0, b1, b2))
                    if lst is not None and any(
                        float(np.linalg.norm(vp - prev)) < 0.48 for prev in lst
                    ):
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if dup:
            continue
        vox_buckets.setdefault(bk, []).append(vp)

        ax_v = axis_g_all[idx].copy()
        for comp in ax_v:
            if abs(float(comp)) > 1e-6:
                if comp < 0.0:
                    ax_v = -ax_v
                break
        ph = prec_h[idx]
        kept.append(
            {
                "pos_g": [round(float(x), 5) for x in g_star[idx]],
                "hkl_cont": [round(float(x), 4) for x in h_star[idx]],
                "nearest_hkl": [int(x) for x in nearest_hkl_arr[idx]],
                "dhkl": round(float(dhkl_all[idx]), 3),
                "min_mahal": round(float(min_mahal[idx]), 3),
                "re_val": round(float(u_star[idx]), 5),
                "im_val": round(float(im_star[idx]), 5),
                "abs_val": round(float(abs_u[idx]), 5),
                "sign": 1 if sgn[idx] > 0 else -1,
                "prec_h": [
                    float(f"{ph[0, 0]:.6g}"),
                    float(f"{ph[1, 1]:.6g}"),
                    float(f"{ph[2, 2]:.6g}"),
                    float(f"{ph[0, 1]:.6g}"),
                    float(f"{ph[0, 2]:.6g}"),
                    float(f"{ph[1, 2]:.6g}"),
                ],
                "axis_g": [round(float(x), 5) for x in ax_v],
                "radius_g": round(float(r_cap_all[idx]), 5),
                "half_len_g": round(float(half_len_all[idx]), 5),
                "aspect": round(float(aspect_all[idx]), 3),
            }
        )
        if len(kept) >= max_extrema:
            break
    return kept


def _upsample_field_zero_padded_fft(
    field: dict[str, Any],
    oversample: int,
) -> tuple[np.ndarray, list[int]]:
    """Evaluate exact Whittaker-Shannon interpolated F(q / K) across [-hmax, +hmax] via zero-padded 3D FFT."""
    rho = field["rho"]
    dims = tuple(int(x) for x in field["dims"])
    hmax = [int(x) for x in field["hmax"]]
    k_up = max(2, int(oversample))
    m_shape = (k_up * dims[0], k_up * dims[1], k_up * dims[2])
    pad = np.zeros(m_shape, dtype=np.complex128)
    s0 = np.arange(-hmax[0], dims[0] - hmax[0]) % m_shape[0]
    s1 = np.arange(-hmax[1], dims[1] - hmax[1]) % m_shape[1]
    s2 = np.arange(-hmax[2], dims[2] - hmax[2]) % m_shape[2]
    pad[np.ix_(s0, s1, s2)] = rho
    f_shifted = np.fft.fftshift(np.fft.ifftn(pad) * float(np.prod(m_shape)))
    centers = [m_shape[0] // 2, m_shape[1] // 2, m_shape[2] // 2]
    return f_shifted, centers


def _pack_detector_morse_b64(records: list[dict[str, Any]]) -> tuple[int, str]:
    """Pack Morse extrema into a compact base64 Float32Array (15 floats/row) for fast full-detector matching."""
    n = len(records)
    if n == 0:
        return 0, ""
    buf = np.empty((n, 15), dtype="<f4")
    for idx, r in enumerate(records):
        hc = r["hkl_cont"]
        ph = r["prec_h"]
        ax = r["axis_g"]
        buf[idx, 0] = hc[0]
        buf[idx, 1] = hc[1]
        buf[idx, 2] = hc[2]
        buf[idx, 3] = r["re_val"]
        buf[idx, 4] = r["im_val"]
        buf[idx, 5] = ph[0]
        buf[idx, 6] = ph[1]
        buf[idx, 7] = ph[2]
        buf[idx, 8] = ph[3]
        buf[idx, 9] = ph[4]
        buf[idx, 10] = ph[5]
        buf[idx, 11] = ax[0]
        buf[idx, 12] = ax[1]
        buf[idx, 13] = ax[2]
        buf[idx, 14] = r["aspect"]
    return n, base64.b64encode(buf.tobytes()).decode("ascii")


def _compute_full_sphere_detector_morse(
    primary_field: dict[str, Any],
    ref_field: dict[str, Any] | None,
    *,
    heterodyne_mode: str,
    f_to_f0_scale: float,
    ub_matrix: np.ndarray,
    mean_g_star: float,
    g_max: float,
) -> dict[str, Any]:
    """Extract Bragg-proximal 3D Morse extrema and Im=0 |Re W| quantiles across the entire d >= d_min sphere."""
    max_dim = max(int(x) for x in primary_field["dims"])
    use_het = (
        ref_field is not None
        and bool(ref_field.get("has_phases"))
        and heterodyne_mode != "single"
    )
    if use_het and ref_field is not None:
        max_dim = max(max_dim, *(int(x) for x in ref_field["dims"]))

    k_up = 4 if max_dim <= 25 else (3 if max_dim <= 47 else 2)
    f_sub, c_f = _upsample_field_zero_padded_fft(primary_field, k_up)
    if not use_het or ref_field is None:
        z_full = f_sub
        c_z = c_f
    else:
        f0_sub, c_0 = _upsample_field_zero_padded_fft(ref_field, k_up)
        q_lo = [min(c_f[d], c_0[d]) for d in range(3)]
        q_hi = [
            min(f_sub.shape[d] - c_f[d], f0_sub.shape[d] - c_0[d]) for d in range(3)
        ]
        f_crop = f_sub[
            c_f[0] - q_lo[0] : c_f[0] + q_hi[0],
            c_f[1] - q_lo[1] : c_f[1] + q_hi[1],
            c_f[2] - q_lo[2] : c_f[2] + q_hi[2],
        ]
        f0_crop = f0_sub[
            c_0[0] - q_lo[0] : c_0[0] + q_hi[0],
            c_0[1] - q_lo[1] : c_0[1] + q_hi[1],
            c_0[2] - q_lo[2] : c_0[2] + q_hi[2],
        ]
        mode = str(heterodyne_mode or "delta").lower()
        if mode == "product":
            z_full = np.conj(f0_crop) * f_crop
        elif mode == "diff":
            z_full = float(f_to_f0_scale) * f_crop - f0_crop
        else:
            z_full = np.conj(f0_crop) * (float(f_to_f0_scale) * f_crop - f0_crop)
        c_z = q_lo

    hg0 = (np.arange(z_full.shape[0], dtype=np.float64) - float(c_z[0])) / float(k_up)
    hg1 = (np.arange(z_full.shape[1], dtype=np.float64) - float(c_z[1])) / float(k_up)
    hg2 = (np.arange(z_full.shape[2], dtype=np.float64) - float(c_z[2])) / float(k_up)

    records = extract_morse_maxima_3d(
        z_full,
        hg0,
        hg1,
        hg2,
        ub_matrix,
        mean_g_star=mean_g_star,
        g_max=g_max,
        max_extrema=12000,
        bragg_mahal_cap=2.75,
    )
    n_det, det_b64 = _pack_detector_morse_b64(records)

    re_v = z_full.real
    im_v = z_full.imag
    amp_90 = float(np.percentile(np.abs(z_full), 90.0)) if z_full.size > 0 else 0.0
    is_centric = float(np.max(np.abs(im_v))) <= 1e-4 * max(amp_90, 1e-12)

    ub = np.asarray(ub_matrix, dtype=np.float64)
    im0_samples: list[np.ndarray] = []
    if not is_centric:
        im_a = im_v[:-1, :, :]
        im_b = im_v[1:, :, :]
        cross = (im_a <= 0.0) != (im_b <= 0.0)
        cx_i, cx_j, cx_k = np.nonzero(cross)
        if len(cx_i) > 0:
            if len(cx_i) > 40000:
                step_s = max(1, len(cx_i) // 40000)
                cx_i = cx_i[::step_s]
                cx_j = cx_j[::step_s]
                cx_k = cx_k[::step_s]
            a_val = np.abs(im_v[cx_i, cx_j, cx_k])
            b_val = np.abs(im_v[cx_i + 1, cx_j, cx_k])
            t_lin = a_val / np.maximum(a_val + b_val, 1e-15)
            h_cross = np.column_stack(
                [hg0[cx_i] + t_lin / float(k_up), hg1[cx_j], hg2[cx_k]]
            )
            gn_c = np.linalg.norm(h_cross @ ub.T, axis=1)
            in_sph = (gn_c > 1e-5) & (gn_c <= g_max)
            if np.any(in_sph):
                re_c = (1.0 - t_lin[in_sph]) * re_v[
                    cx_i[in_sph], cx_j[in_sph], cx_k[in_sph]
                ] + t_lin[in_sph] * re_v[cx_i[in_sph] + 1, cx_j[in_sph], cx_k[in_sph]]
                im0_samples.append(np.abs(re_c))

    if im0_samples:
        q_pool = np.concatenate(im0_samples)
    elif records:
        q_pool = np.array([float(r["abs_val"]) for r in records], dtype=np.float64)
    else:
        q_pool = np.zeros(1, dtype=np.float64)

    det_quantiles = [
        round(float(x), 5) for x in np.quantile(q_pool, np.linspace(0.0, 1.0, 101))
    ]
    return {
        "detector_morse_count": n_det,
        "detector_morse_b64": det_b64,
        "detector_antinode_quantiles": det_quantiles,
    }


def _extract_single_isosurface_mesh(
    scalar_vol: np.ndarray,
    amp_ref: float,
    h0: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    ub_matrix: np.ndarray,
    ub_inv_t: np.ndarray,
    *,
    aux_scalar_vol: np.ndarray | None = None,
    g_max: float | None = None,
    roi_center_g: np.ndarray | None = None,
    roi_radius_g: float | None = None,
    slab_normal_g: np.ndarray | None = None,
    slab_offset_g: float = 0.0,
    slab_half_thickness_g: float | None = None,
) -> dict[str, Any]:
    """Run Marching Cubes at level=0 and transform vertices/normals into sample reciprocal space G (A^-1)."""
    from skimage.measure import marching_cubes

    empty_res: dict[str, Any] = {
        "n_verts": 0,
        "n_tris": 0,
        "verts_b64": "",
        "norms_b64": "",
        "faces_b64": "",
    }
    if aux_scalar_vol is not None:
        empty_res["re_vals_b64"] = ""

    vmin = float(np.min(scalar_vol))
    vmax = float(np.max(scalar_vol))
    tol = 1e-4 * max(amp_ref, 1e-12)
    if not (vmin < -tol and vmax > tol):
        return empty_res

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
        return empty_res

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
        return empty_res

    # Compact vertex array to referenced vertices only
    used_v, remap = np.unique(faces_kept.ravel(), return_inverse=True)
    verts_compact = verts_g[used_v]
    norms_compact = norms_g[used_v]
    faces_compact = remap.reshape((-1, 3))

    res: dict[str, Any] = {
        "n_verts": len(verts_compact),
        "n_tris": len(faces_compact),
        "verts_b64": _pack_f32_b64(verts_compact),
        "norms_b64": _pack_f32_b64(norms_compact),
        "faces_b64": _pack_u32_b64(faces_compact),
    }
    if aux_scalar_vol is not None:
        pts_grid_idx = np.column_stack(
            [
                verts_idx[used_v, 0] / max(abs(dh0), 1e-12),
                verts_idx[used_v, 1] / max(abs(dh1), 1e-12),
                verts_idx[used_v, 2] / max(abs(dh2), 1e-12),
            ]
        )
        re_vals_compact = _trilinear_sample_volume(aux_scalar_vol, pts_grid_idx)
        res["re_vals_b64"] = _pack_f32_b64(re_vals_compact)
        res["abs_re_quantiles"] = [
            round(float(x), 5)
            for x in np.quantile(np.abs(re_vals_compact), np.linspace(0.0, 1.0, 101))
        ]
    return res


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
        morse_maxima = cached_brick["morse_maxima"]
        antinode_quantiles = cached_brick["antinode_quantiles"]
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
            aux_scalar_vol=Z_brick.real,
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
        morse_maxima = extract_morse_maxima_3d(
            Z_brick,
            h0_b,
            h1_b,
            h2_b,
            ub,
            mean_g_star=mean_g_star,
            g_max=g_max,
            roi_center_g=brick_center_g,
            roi_radius_g=brick_radius_g,
            max_extrema=240,
        )
        if im_mesh.get("abs_re_quantiles"):
            antinode_quantiles = list(im_mesh["abs_re_quantiles"])
        else:
            hh0, hh1, hh2 = np.meshgrid(h0_b, h1_b, h2_b, indexing="ij")
            pts_g_all = np.column_stack([hh0.ravel(), hh1.ravel(), hh2.ravel()]) @ ub.T
            gn_all = np.linalg.norm(pts_g_all, axis=1)
            dc_all = np.linalg.norm(pts_g_all - brick_center_g[None, :], axis=1)
            vmask = (gn_all > 1e-5) & (gn_all <= g_max) & (dc_all <= brick_radius_g)
            abs_re_valid = np.abs(Z_brick.real.ravel()[vmask])
            if abs_re_valid.size > 0:
                antinode_quantiles = [
                    round(float(x), 5)
                    for x in np.quantile(abs_re_valid, np.linspace(0.0, 1.0, 101))
                ]
            else:
                antinode_quantiles = [0.0] * 101

        _BRICK_PAYLOAD_CACHE[brick_cache_key] = {
            "is_real_only": is_real_only,
            "re_surface": re_mesh,
            "im_surface": im_mesh,
            "brick_vortices": brick_vortices,
            "morse_maxima": morse_maxima,
            "antinode_quantiles": antinode_quantiles,
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

    # Full-sphere (d >= d_min) Bragg-proximal Morse extrema for 2D detector Mahalanobis mapping
    m_cache_key = (
        primary.get("path"),
        primary.get("q75_f"),
        ref_field.get("path") if ref_field is not None else None,
        ref_field.get("q75_f") if ref_field is not None else None,
        eff_het_mode,
        round(float(f_to_f0_scale), 8),
        round(float(g_max), 4),
        tuple(np.round(ub.ravel(), 5).tolist()),
    )
    cached_gm = _GLOBAL_MORSE_CACHE.get(m_cache_key)
    if cached_gm is not None:
        _GLOBAL_MORSE_CACHE.move_to_end(m_cache_key)
        det_morse_info = cached_gm
    else:
        det_morse_info = _compute_full_sphere_detector_morse(
            primary,
            ref_field,
            heterodyne_mode=eff_het_mode,
            f_to_f0_scale=f_to_f0_scale,
            ub_matrix=ub,
            mean_g_star=mean_g_star,
            g_max=g_max,
        )
        _GLOBAL_MORSE_CACHE[m_cache_key] = det_morse_info
        while len(_GLOBAL_MORSE_CACHE) > _MAX_FIELD_CACHE_ENTRIES:
            _GLOBAL_MORSE_CACHE.popitem(last=False)

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
        "morse_maxima": morse_maxima,
        "antinode_quantiles": antinode_quantiles,
        "detector_morse_count": det_morse_info["detector_morse_count"],
        "detector_morse_b64": det_morse_info["detector_morse_b64"],
        "detector_antinode_quantiles": det_morse_info["detector_antinode_quantiles"],
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
    let decodedMeshes = { re: null, im: null, vortex: null, morsePos: null, morseNeg: null, morseList: [] };
    let glCanvas = null;
    let gl = null;
    let glProg = null;
    let glVortexProg = null;
    let glBufs = { re: null, im: null, vortex: null, morsePos: null, morseNeg: null };
    let glUnavailable = false;
    let last3DRenderSig = "";
    let screenPeakHits3D = [];
    let screenMorseHits3D = [];
    let screenMorseHitsDet = [];
    let detectorMorseMap = new Map();
    let detectorMorseRecords = [];
    let hoveredMorseObj = null;
    let roiQuantileCacheSig = "";
    let roiSortedAbsRe = new Float32Array(0);
    let roiMaxAbsRe = 0.0;
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
        return { nVerts: 0, nTris: 0, verts: new Float32Array(0), norms: new Float32Array(0), scalars: null, faces: new Uint32Array(0) };
      }
      const verts = decodeBase64Float32(m.verts_b64);
      const norms = m.norms_b64 ? decodeBase64Float32(m.norms_b64) : new Float32Array(verts.length);
      const scalars = m.re_vals_b64 ? decodeBase64Float32(m.re_vals_b64) : null;
      const faces = decodeBase64Uint32(m.faces_b64);
      return {
        nVerts: Number(m.n_verts) || (verts.length / 3),
        nTris: Number(m.n_tris) || (faces.length / 3),
        verts,
        norms,
        scalars: (scalars && scalars.length >= (verts.length / 3)) ? scalars : null,
        faces,
      };
    }

    function buildMorseCapsuleMesh(caps) {
      if (!Array.isArray(caps) || caps.length === 0) {
        return { nVerts: 0, nTris: 0, verts: new Float32Array(0), norms: new Float32Array(0), scalars: new Float32Array(0), faces: new Uint32Array(0) };
      }
      const nAz = 14;
      const capSteps = 4;
      let totalVerts = 0;
      let totalTris = 0;
      for (let ci = 0; ci < caps.length; ci++) {
        const L = Math.max(0.0, Number(caps[ci].half_len_g) || 0.0);
        const nRings = (capSteps + 1) + (L > 1e-7 ? (capSteps + 1) : capSteps);
        totalVerts += nRings * nAz;
        totalTris += (nRings - 1) * nAz * 2;
      }
      const verts = new Float32Array(totalVerts * 3);
      const norms = new Float32Array(totalVerts * 3);
      const scalars = new Float32Array(totalVerts);
      const faces = new Uint32Array(totalTris * 3);
      const cosPhi = new Float32Array(nAz);
      const sinPhi = new Float32Array(nAz);
      for (let a = 0; a < nAz; a++) {
        const phi = (2.0 * Math.PI * a) / nAz;
        cosPhi[a] = Math.cos(phi);
        sinPhi[a] = Math.sin(phi);
      }

      let vCursor = 0;
      let fCursor = 0;
      for (let ci = 0; ci < caps.length; ci++) {
        const c = caps[ci];
        const pos = c.pos_g || [0, 0, 0];
        const rawAx = c.axis_g || [0, 0, 1];
        const axNorm = Math.hypot(rawAx[0], rawAx[1], rawAx[2]) || 1.0;
        const ax = rawAx[0] / axNorm, ay = rawAx[1] / axNorm, az = rawAx[2] / axNorm;
        const refX = Math.abs(ax) < 0.85 ? 1.0 : 0.0;
        const refY = Math.abs(ax) < 0.85 ? 0.0 : 1.0;
        let ux = ay * 0.0 - az * refY;
        let uy = az * refX - ax * 0.0;
        let uz = ax * refY - ay * refX;
        const un = Math.hypot(ux, uy, uz) || 1.0;
        ux /= un; uy /= un; uz /= un;
        const vx = ay * uz - az * uy;
        const vy = az * ux - ax * uz;
        const vz = ax * uy - ay * ux;

        const r = Math.max(1e-5, Number(c.radius_g) || 0.003);
        const L = Math.max(0.0, Number(c.half_len_g) || 0.0);
        const reVal = Number(c.re_val) || 0.0;
        const rings = [];
        for (let s = 0; s <= capSteps; s++) {
          const th = -0.5 * Math.PI + (0.5 * Math.PI * s) / capSteps;
          rings.push([Math.sin(th), Math.cos(th), -L]);
        }
        for (let s = (L > 1e-7 ? 0 : 1); s <= capSteps; s++) {
          const th = (0.5 * Math.PI * s) / capSteps;
          rings.push([Math.sin(th), Math.cos(th), L]);
        }

        const baseV = vCursor;
        const nRings = rings.length;
        for (let ri = 0; ri < nRings; ri++) {
          const sinTh = rings[ri][0], cosTh = rings[ri][1], zOff = rings[ri][2];
          const cxR = pos[0] + zOff * ax;
          const cyR = pos[1] + zOff * ay;
          const czR = pos[2] + zOff * az;
          for (let a = 0; a < nAz; a++) {
            const cp = cosPhi[a], sp = sinPhi[a];
            const radX = cp * ux + sp * vx;
            const radY = cp * uy + sp * vy;
            const radZ = cp * uz + sp * vz;
            const nx = cosTh * radX + sinTh * ax;
            const ny = cosTh * radY + sinTh * ay;
            const nz = cosTh * radZ + sinTh * az;
            const o3 = vCursor * 3;
            norms[o3] = nx;
            norms[o3 + 1] = ny;
            norms[o3 + 2] = nz;
            verts[o3] = cxR + r * nx;
            verts[o3 + 1] = cyR + r * ny;
            verts[o3 + 2] = czR + r * nz;
            scalars[vCursor] = reVal;
            vCursor++;
          }
        }
        for (let ri = 0; ri < nRings - 1; ri++) {
          const r0 = baseV + ri * nAz;
          const r1 = baseV + (ri + 1) * nAz;
          for (let a = 0; a < nAz; a++) {
            const aNext = (a + 1) % nAz;
            const v00 = r0 + a, v10 = r0 + aNext, v11 = r1 + aNext, v01 = r1 + a;
            faces[fCursor++] = v00; faces[fCursor++] = v10; faces[fCursor++] = v11;
            faces[fCursor++] = v00; faces[fCursor++] = v11; faces[fCursor++] = v01;
          }
        }
      }
      return { nVerts: totalVerts, nTris: totalTris, verts, norms, scalars, faces };
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

    function buildDetectorMorseIndex(nd, roiMorseList) {
      const hklMap = new Map();
      const allRecords = [];
      if (!nd || !nd.has_phases) {
        detectorMorseMap = hklMap;
        detectorMorseRecords = allRecords;
        return;
      }
      const voxBuckets = new Map();
      function addRecordIfUnique(rec) {
        const hc = rec.hkl_cont;
        if (!Array.isArray(hc) || hc.length !== 3) return;
        const bx = Math.floor(hc[0] * 2.0);
        const by = Math.floor(hc[1] * 2.0);
        const bz = Math.floor(hc[2] * 2.0);
        for (let dx = -1; dx <= 1; dx++) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dz = -1; dz <= 1; dz++) {
              const bucket = voxBuckets.get(`${bx + dx},${by + dy},${bz + dz}`);
              if (bucket) {
                for (let k = 0; k < bucket.length; k++) {
                  const prev = bucket[k].hkl_cont;
                  if (Math.hypot(hc[0] - prev[0], hc[1] - prev[1], hc[2] - prev[2]) < 0.44) {
                    return;
                  }
                }
              }
            }
          }
        }
        const key = `${bx},${by},${bz}`;
        let lst = voxBuckets.get(key);
        if (!lst) {
          lst = [];
          voxBuckets.set(key, lst);
        }
        lst.push(rec);
        allRecords.push(rec);
      }

      if (Array.isArray(roiMorseList)) {
        for (let i = 0; i < roiMorseList.length; i++) {
          const m = roiMorseList[i];
          if (!m || !Array.isArray(m.hkl_cont) || m.hkl_cont.length !== 3) continue;
          const ph = Array.isArray(m.prec_h) && m.prec_h.length === 6
            ? m.prec_h
            : [6.0, 6.0, 6.0, 0.0, 0.0, 0.0];
          addRecordIfUnique({
            hkl_cont: [Number(m.hkl_cont[0]) || 0, Number(m.hkl_cont[1]) || 0, Number(m.hkl_cont[2]) || 0],
            re_val: Number(m.re_val) || 0.0,
            im_val: Number(m.im_val) || 0.0,
            prec_h: ph,
            axis_g: Array.isArray(m.axis_g) && m.axis_g.length === 3 ? m.axis_g : [0.0, 0.0, 1.0],
            aspect: Math.max(1.0, Number(m.aspect) || 1.0),
          });
        }
      }

      if (nd.detector_morse_b64) {
        const raw = decodeBase64Float32(nd.detector_morse_b64);
        const nRows = Math.floor(raw.length / 15);
        for (let i = 0; i < nRows; i++) {
          const o = i * 15;
          addRecordIfUnique({
            hkl_cont: [raw[o], raw[o + 1], raw[o + 2]],
            re_val: raw[o + 3],
            im_val: raw[o + 4],
            prec_h: [raw[o + 5], raw[o + 6], raw[o + 7], raw[o + 8], raw[o + 9], raw[o + 10]],
            axis_g: [raw[o + 11], raw[o + 12], raw[o + 13]],
            aspect: Math.max(1.0, raw[o + 14] || 1.0),
          });
        }
      }

      const maxCap = 2.65;
      const maxCap2 = maxCap * maxCap;
      for (let idx = 0; idx < allRecords.length; idx++) {
        const rec = allRecords[idx];
        const hc0 = rec.hkl_cont[0], hc1 = rec.hkl_cont[1], hc2 = rec.hkl_cont[2];
        const ph = rec.prec_h;
        const phh = Number(ph[0]) || 0.0;
        const pkk = Number(ph[1]) || 0.0;
        const pll = Number(ph[2]) || 0.0;
        const phk = Number(ph[3]) || 0.0;
        const phl = Number(ph[4]) || 0.0;
        const pkl = Number(ph[5]) || 0.0;
        if (!(phh > 1e-9 && pkk > 1e-9 && pll > 1e-9)) continue;

        const c00 = pkk * pll - pkl * pkl;
        const c11 = phh * pll - phl * phl;
        const c22 = phh * pkk - phk * phk;
        const det = phh * c00 - phk * (phk * pll - pkl * phl) + phl * (phk * pkl - pkk * phl);
        let rh = 2, rk = 2, rl = 2;
        if (det > 1e-12 && c00 > 0.0 && c11 > 0.0 && c22 > 0.0) {
          rh = Math.max(1, Math.min(4, Math.ceil(maxCap * Math.sqrt(c00 / det))));
          rk = Math.max(1, Math.min(4, Math.ceil(maxCap * Math.sqrt(c11 / det))));
          rl = Math.max(1, Math.min(4, Math.ceil(maxCap * Math.sqrt(c22 / det))));
        }
        const hRound = Math.round(hc0);
        const kRound = Math.round(hc1);
        const lRound = Math.round(hc2);
        for (let ih = hRound - rh; ih <= hRound + rh; ih++) {
          const dh = ih - hc0;
          for (let ik = kRound - rk; ik <= kRound + rk; ik++) {
            const dk = ik - hc1;
            for (let il = lRound - rl; il <= lRound + rl; il++) {
              const dl = il - hc2;
              const d2 = (
                phh * dh * dh +
                pkk * dk * dk +
                pll * dl * dl +
                2.0 * (phk * dh * dk + phl * dh * dl + pkl * dk * dl)
              );
              if (d2 >= -1e-7 && d2 <= maxCap2) {
                const dMahal = Math.sqrt(Math.max(0.0, d2));
                const hklKey = `${ih},${ik},${il}`;
                let arr = hklMap.get(hklKey);
                if (!arr) {
                  arr = [];
                  hklMap.set(hklKey, arr);
                }
                arr.push({ morse: rec, dMahal });
              }
            }
          }
        }
      }
      for (const arr of hklMap.values()) {
        if (arr.length > 1) {
          arr.sort((a, b) => (a.dMahal - b.dMahal) || (Math.abs(b.morse.re_val) - Math.abs(a.morse.re_val)));
        }
      }
      detectorMorseMap = hklMap;
      detectorMorseRecords = allRecords;
    }

    function ensureDecodedMeshes() {
      if (!nodalData || !nodalData.has_phases) {
        decodedMeshes = { re: null, im: null, vortex: null, morsePos: null, morseNeg: null, morseList: [] };
        detectorMorseMap = new Map();
        detectorMorseRecords = [];
        screenMorseHitsDet = [];
        decodedCacheSig = "";
        roiQuantileCacheSig = "";
        roiSortedAbsRe = new Float32Array(0);
        roiMaxAbsRe = 0.0;
        return decodedMeshes;
      }
      const reM = nodalData.re_surface || {};
      const imM = nodalData.im_surface || {};
      const vList = getActiveVortexList(nodalData);
      const mList = Array.isArray(nodalData.morse_maxima) ? nodalData.morse_maxima : [];
      const sig = [
        makeBrickKey(nodalData),
        reM.n_tris || 0,
        (reM.verts_b64 || "").slice(0, 24),
        imM.n_tris || 0,
        (imM.verts_b64 || "").slice(0, 24),
        (imM.re_vals_b64 || "").slice(0, 24),
        (nodalData.roi_vortices || []).length,
        vList.length,
        mList.length,
        nodalData.detector_morse_count || 0,
        (nodalData.detector_morse_b64 || "").slice(0, 24),
      ].join("|");
      if (sig === decodedCacheSig) return decodedMeshes;
      decodedCacheSig = sig;
      roiQuantileCacheSig = "";
      const posCaps = [];
      const negCaps = [];
      for (let i = 0; i < mList.length; i++) {
        if ((Number(mList[i].re_val) || 0.0) >= 0.0) posCaps.push(mList[i]);
        else negCaps.push(mList[i]);
      }
      decodedMeshes = {
        re: decodeSingleMesh(reM),
        im: decodeSingleMesh(imM),
        vortex: decodeVortexRibbonGeometry(vList),
        morsePos: buildMorseCapsuleMesh(posCaps),
        morseNeg: buildMorseCapsuleMesh(negCaps),
        morseList: mList,
      };
      buildDetectorMorseIndex(nodalData, mList);
      uploadGlMeshes();
      last3DRenderSig = "";
      return decodedMeshes;
    }

    function resolveDetectorMorseThreshold() {
      ensureDecodedMeshes();
      const rawPct = Number(deps.state && deps.state.antinodePercentile);
      const pct = Math.max(0.0, Math.min(1.0, Number.isFinite(rawPct) ? rawPct : 0.75));
      const rawSig = Number(deps.state && deps.state.morseMahalSigma);
      const maxMahal = Math.max(0.10, Math.min(3.0, Number.isFinite(rawSig) ? rawSig : 1.0));
      if (!nodalData || !nodalData.has_phases) {
        return { pct, tau: 0.0, maxMahal };
      }
      const qArr = (Array.isArray(nodalData.detector_antinode_quantiles) && nodalData.detector_antinode_quantiles.length > 0)
        ? nodalData.detector_antinode_quantiles
        : (Array.isArray(nodalData.antinode_quantiles) ? nodalData.antinode_quantiles : []);
      let tau = 0.0;
      if (pct > 0.0 && qArr.length > 0) {
        const effP = Math.min(0.9995, pct);
        const fIdx = effP * (qArr.length - 1);
        const i0 = Math.floor(fIdx);
        const i1 = Math.min(qArr.length - 1, i0 + 1);
        const frac = fIdx - i0;
        const v0 = Math.max(0.0, Number(qArr[i0]) || 0.0);
        const v1 = Math.max(0.0, Number(qArr[i1]) || 0.0);
        tau = v0 * (1.0 - frac) + v1 * frac;
        if (pct >= 0.999) {
          const topV = Math.max(0.0, Number(qArr[qArr.length - 1]) || 0.0);
          tau = Math.min(tau, topV * 0.9995);
        }
      }
      return { pct, tau, maxMahal };
    }

    function getPeakMorseMatch(hkl, preThresh) {
      if (!nodalData || !nodalData.has_phases || !Array.isArray(hkl) || hkl.length !== 3) {
        return null;
      }
      ensureDecodedMeshes();
      const arr = detectorMorseMap.get(`${hkl[0]},${hkl[1]},${hkl[2]}`);
      if (!arr || arr.length === 0) return null;
      const th = preThresh || resolveDetectorMorseThreshold();
      for (let i = 0; i < arr.length; i++) {
        const cand = arr[i];
        if (cand.dMahal <= th.maxMahal + 1e-7 && Math.abs(cand.morse.re_val) >= th.tau - 1e-7) {
          return cand;
        }
      }
      return null;
    }

    function resolveAntinodeThreshold(liveRoi) {
      ensureDecodedMeshes();
      const rawPct = Number(deps.state && deps.state.antinodePercentile);
      const pct = Math.max(0.0, Math.min(1.0, Number.isFinite(rawPct) ? rawPct : 0.75));
      if (!nodalData || !nodalData.has_phases) {
        return { pct, tau: 0.0, maxAbs: 1.0, activeMorseCount: 0 };
      }
      const roiKey = [
        decodedCacheSig,
        liveRoi.mode,
        liveRoi.centerG[0].toFixed(4),
        liveRoi.centerG[1].toFixed(4),
        liveRoi.centerG[2].toFixed(4),
        liveRoi.radiusG.toFixed(4),
        liveRoi.slabActive ? `${liveRoi.slabNormal.map(x => x.toFixed(3)).join(",")}:${liveRoi.slabHalfThick.toFixed(4)}` : "0",
      ].join("|");
      if (roiKey !== roiQuantileCacheSig) {
        roiQuantileCacheSig = roiKey;
        const imMesh = decodedMeshes.im;
        const vals = [];
        let maxV = 0.0;
        if (imMesh && imMesh.scalars && imMesh.nVerts > 0) {
          const v = imMesh.verts;
          const sc = imMesh.scalars;
          const stride = Math.max(1, Math.floor(imMesh.nVerts / 8000));
          for (let i = 0; i < imMesh.nVerts; i += stride) {
            const o3 = i * 3;
            if (pointInsideLiveRoi([v[o3], v[o3 + 1], v[o3 + 2]], liveRoi)) {
              const av = Math.abs(sc[i]);
              vals.push(av);
              if (av > maxV) maxV = av;
            }
          }
        }
        const mList = decodedMeshes.morseList || [];
        for (let i = 0; i < mList.length; i++) {
          const mp = mList[i].pos_g;
          if (mp && pointInsideLiveRoi(mp, liveRoi)) {
            const av = Math.abs(Number(mList[i].re_val) || 0.0);
            if (av > maxV) maxV = av;
            if (!imMesh || !imMesh.scalars || imMesh.nVerts === 0) vals.push(av);
          }
        }
        if (vals.length === 0 && Array.isArray(nodalData.antinode_quantiles) && nodalData.antinode_quantiles.length > 0) {
          for (let i = 0; i < nodalData.antinode_quantiles.length; i++) {
            const av = Math.max(0.0, Number(nodalData.antinode_quantiles[i]) || 0.0);
            vals.push(av);
            if (av > maxV) maxV = av;
          }
        }
        vals.sort((a, b) => a - b);
        roiSortedAbsRe = new Float32Array(vals);
        roiMaxAbsRe = Math.max(maxV, 1e-6);
      }

      let tau = 0.0;
      if (pct > 0.0 && roiSortedAbsRe.length > 0) {
        const effP = Math.min(0.9995, pct);
        const fIdx = effP * (roiSortedAbsRe.length - 1);
        const i0 = Math.floor(fIdx);
        const i1 = Math.min(roiSortedAbsRe.length - 1, i0 + 1);
        const frac = fIdx - i0;
        tau = roiSortedAbsRe[i0] * (1.0 - frac) + roiSortedAbsRe[i1] * frac;
        if (pct >= 0.999) {
          tau = Math.min(tau, roiSortedAbsRe[roiSortedAbsRe.length - 1] * 0.9995);
        }
      }

      let activeMorseCount = 0;
      const mList = decodedMeshes.morseList || [];
      for (let i = 0; i < mList.length; i++) {
        const mp = mList[i].pos_g;
        if (!mp || !pointInsideLiveRoi(mp, liveRoi)) continue;
        if (Math.abs(Number(mList[i].re_val) || 0.0) >= tau) activeMorseCount++;
      }
      return { pct, tau, maxAbs: roiMaxAbsRe, activeMorseCount };
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
          "attribute float aScalar;",
          "uniform mat3 uRot;",
          "uniform vec3 uCenter;",
          "uniform float uInvScale;",
          "uniform float uAspectX;",
          "uniform float uAspectY;",
          "uniform float uCamDist;",
          "varying vec3 vNorm;",
          "varying vec3 vSamplePos;",
          "varying float vEyeDist;",
          "varying float vScalar;",
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
          "  vScalar = aScalar;",
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
          "uniform float uAntinodeMode;",
          "uniform float uAntinodeTau;",
          "uniform float uAntinodeMax;",
          "varying vec3 vNorm;",
          "varying vec3 vSamplePos;",
          "varying float vEyeDist;",
          "varying float vScalar;",
          "void main() {",
          "  if (vEyeDist < 0.28) discard;",
          "  vec3 dRoi = vSamplePos - uClipCenter;",
          "  if (dot(dRoi, dRoi) > uClipRadius * uClipRadius) discard;",
          "  if (uSlabActive > 0.5) {",
          "    if (abs(dot(vSamplePos, uSlabNormal) - uSlabOffset) > uSlabHalfThick) discard;",
          "  }",
          "  float absS = abs(vScalar);",
          "  if (uAntinodeMode > 0.5 && absS < uAntinodeTau) discard;",
          "  vec3 N = normalize(vNorm);",
          "  if (!gl_FrontFacing) N = -N;",
          "  vec3 L1 = normalize(vec3(-0.38, 0.52, 0.76));",
          "  vec3 L2 = normalize(vec3(0.45, -0.35, 0.65));",
          "  float diff = 0.34 + 0.50 * max(dot(N, L1), 0.0) + 0.20 * max(dot(N, L2), 0.0);",
          "  vec3 V = vec3(0.0, 0.0, 1.0);",
          "  vec3 H = normalize(L1 + V);",
          "  vec3 baseCol = uColor.rgb;",
          "  float outAlpha = uColor.a;",
          "  float specGain = 0.22;",
          "  float rimBoost = 0.0;",
          "  if (uAntinodeMode > 0.5 && uAntinodeMode < 1.5) {",
          "    float crest = clamp((absS - uAntinodeTau) / max(1e-6, uAntinodeMax - uAntinodeTau), 0.0, 1.0);",
          "    vec3 posCol = mix(vec3(0.98, 0.62, 0.11), vec3(1.00, 0.88, 0.42), crest * 0.65);",
          "    vec3 negCol = mix(vec3(0.76, 0.34, 0.98), vec3(0.92, 0.68, 1.00), crest * 0.65);",
          "    baseCol = vScalar >= 0.0 ? posCol : negCol;",
          "    rimBoost = 0.34 * (1.0 - smoothstep(0.0, 0.06, crest));",
          "    outAlpha = clamp(uColor.a * (1.12 + 0.22 * crest), 0.18, 0.94);",
          "    specGain = 0.30;",
          "  } else if (uAntinodeMode > 1.5) {",
          "    specGain = 0.46;",
          "    rimBoost = 0.10;",
          "  }",
          "  float spec = specGain * pow(max(dot(N, H), 0.0), 26.0);",
          "  float rim = 0.16 * pow(1.0 - abs(dot(N, V)), 2.0) + rimBoost;",
          "  vec3 rgb = clamp(baseCol * diff + vec3(spec + rim * 0.65), 0.0, 1.0);",
          "  gl_FragColor = vec4(rgb, outAlpha);",
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
          aScalar: gl.getAttribLocation(prog, "aScalar"),
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
          uAntinodeMode: gl.getUniformLocation(prog, "uAntinodeMode"),
          uAntinodeTau: gl.getUniformLocation(prog, "uAntinodeTau"),
          uAntinodeMax: gl.getUniformLocation(prog, "uAntinodeMax"),
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
      let sbo = null;
      if (m.scalars && m.scalars.length >= m.nVerts) {
        sbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, sbo);
        gl.bufferData(gl.ARRAY_BUFFER, m.scalars, gl.STATIC_DRAW);
      }
      const ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.faces, gl.STATIC_DRAW);
      return { vbo, nbo, sbo, ibo, count: m.faces.length, indexType: gl.UNSIGNED_INT };
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
      if (b.sbo) gl.deleteBuffer(b.sbo);
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
      disposeGlMesh(glBufs.morsePos);
      disposeGlMesh(glBufs.morseNeg);
      glBufs.re = decodedMeshes.re ? uploadGlSingleMesh(decodedMeshes.re) : null;
      glBufs.im = decodedMeshes.im ? uploadGlSingleMesh(decodedMeshes.im) : null;
      glBufs.vortex = decodedMeshes.vortex ? uploadGlVortexRibbon(decodedMeshes.vortex) : null;
      glBufs.morsePos = decodedMeshes.morsePos ? uploadGlSingleMesh(decodedMeshes.morsePos) : null;
      glBufs.morseNeg = decodedMeshes.morseNeg ? uploadGlSingleMesh(decodedMeshes.morseNeg) : null;
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

    function renderSurfacesAndVortices3D(ctx, w, h, dpr, Rtot, center, invScale, Rpx, zoom, liveRoi, isExp, antinodeStats) {
      const st = deps.state;
      if (!nodalData || !nodalData.has_phases) return;
      ensureDecodedMeshes();
      const antinodeOn = Boolean(st.showAntinodes);
      const morseOn = Boolean(st.showMorseCapsules);
      const hasRe = st.showNodalSurfacesRe && decodedMeshes.re && decodedMeshes.re.nTris > 0;
      const hasIm = (st.showNodalSurfacesIm || antinodeOn) && decodedMeshes.im && decodedMeshes.im.nTris > 0;
      const hasVor = st.showPhaseVortices && decodedMeshes.vortex && decodedMeshes.vortex.nSegs > 0;
      const hasMorse = morseOn && (
        (decodedMeshes.morsePos && decodedMeshes.morsePos.nTris > 0) ||
        (decodedMeshes.morseNeg && decodedMeshes.morseNeg.nTris > 0)
      );
      if (!hasRe && !hasIm && !hasVor && !hasMorse) return;

      const aStats = antinodeStats || resolveAntinodeThreshold(liveRoi);
      const tau = aStats.tau;
      const maxAbs = aStats.maxAbs;
      const surfAlpha = Math.max(0.12, Math.min(0.92, Number(st.nodalSurfaceAlpha) || 0.56));
      const glCtx = initWebGLOverlay();
      if (glCtx && glProg && (glBufs.re || glBufs.im || glBufs.vortex || glBufs.morsePos || glBufs.morseNeg)) {
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

        function bindProgCommonUniforms() {
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
          glCtx.uniform1f(glProg.uAntinodeTau, tau);
          glCtx.uniform1f(glProg.uAntinodeMax, maxAbs);
        }

        function drawPass(buf, rgba, cullFaceMode, depthMask, antinodeMode) {
          if (!buf) return;
          glCtx.depthMask(depthMask);
          if (cullFaceMode) {
            glCtx.enable(glCtx.CULL_FACE);
            glCtx.cullFace(cullFaceMode);
          } else {
            glCtx.disable(glCtx.CULL_FACE);
          }
          glCtx.uniform1f(glProg.uAntinodeMode, antinodeMode || 0.0);
          glCtx.uniform4f(glProg.uColor, rgba[0], rgba[1], rgba[2], rgba[3]);
          glCtx.bindBuffer(glCtx.ARRAY_BUFFER, buf.vbo);
          glCtx.enableVertexAttribArray(glProg.aPos);
          glCtx.vertexAttribPointer(glProg.aPos, 3, glCtx.FLOAT, false, 0, 0);
          glCtx.bindBuffer(glCtx.ARRAY_BUFFER, buf.nbo);
          glCtx.enableVertexAttribArray(glProg.aNorm);
          glCtx.vertexAttribPointer(glProg.aNorm, 3, glCtx.FLOAT, false, 0, 0);
          if (glProg.aScalar >= 0) {
            if (buf.sbo) {
              glCtx.bindBuffer(glCtx.ARRAY_BUFFER, buf.sbo);
              glCtx.enableVertexAttribArray(glProg.aScalar);
              glCtx.vertexAttribPointer(glProg.aScalar, 1, glCtx.FLOAT, false, 0, 0);
            } else {
              glCtx.disableVertexAttribArray(glProg.aScalar);
              glCtx.vertexAttrib1f(glProg.aScalar, 0.0);
            }
          }
          glCtx.bindBuffer(glCtx.ELEMENT_ARRAY_BUFFER, buf.ibo);
          glCtx.drawElements(glCtx.TRIANGLES, buf.count, buf.indexType, 0);
        }

        if ((hasRe && glBufs.re) || (hasIm && glBufs.im)) {
          bindProgCommonUniforms();
          const colRe = [0.06, 0.78, 0.92, surfAlpha];
          const colIm = [0.96, 0.30, 0.48, surfAlpha];
          const imMode = (antinodeOn && glBufs.im && glBufs.im.sbo) ? 1.0 : 0.0;
          if (hasRe && glBufs.re) drawPass(glBufs.re, [colRe[0], colRe[1], colRe[2], surfAlpha * 0.78], glCtx.FRONT, true, 0.0);
          if (hasIm && glBufs.im) drawPass(glBufs.im, [colIm[0], colIm[1], colIm[2], surfAlpha * 0.78], glCtx.FRONT, true, imMode);
          if (hasRe && glBufs.re) drawPass(glBufs.re, colRe, glCtx.BACK, false, 0.0);
          if (hasIm && glBufs.im) drawPass(glBufs.im, colIm, glCtx.BACK, false, imMode);
          glCtx.disableVertexAttribArray(glProg.aPos);
          glCtx.disableVertexAttribArray(glProg.aNorm);
          if (glProg.aScalar >= 0) glCtx.disableVertexAttribArray(glProg.aScalar);
        }

        // Clear surface depth buffer so translucent Re=0 / Im=0 sheets never occlude 1D vortices or 3D Morse capsules,
        // while keeping hardware depth testing active between vortex filaments and 3D Morse capsules!
        const needForegroundClear = (hasVor && glVortexProg && glBufs.vortex) || (hasMorse && (glBufs.morsePos || glBufs.morseNeg));
        if (needForegroundClear) {
          glCtx.depthMask(true);
          glCtx.clear(glCtx.DEPTH_BUFFER_BIT);
        }

        if (hasVor && glVortexProg && glBufs.vortex) {
          glCtx.depthMask(true);
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

        if (hasMorse && (glBufs.morsePos || glBufs.morseNeg)) {
          bindProgCommonUniforms();
          if (glBufs.morsePos) drawPass(glBufs.morsePos, [1.00, 0.76, 0.16, 0.95], glCtx.BACK, true, 2.0);
          if (glBufs.morseNeg) drawPass(glBufs.morseNeg, [0.85, 0.42, 1.00, 0.95], glCtx.BACK, true, 2.0);
          glCtx.disableVertexAttribArray(glProg.aPos);
          glCtx.disableVertexAttribArray(glProg.aNorm);
          if (glProg.aScalar >= 0) glCtx.disableVertexAttribArray(glProg.aScalar);
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
      function collectSoftwareTris(mesh, baseRgb, useAntinodeFilter) {
        if (!mesh || !mesh.nTris) return;
        const stride = Math.max(1, Math.ceil(mesh.nTris / 3500));
        const v = mesh.verts, n = mesh.norms, f = mesh.faces, sc = mesh.scalars;
        for (let t = 0; t < mesh.nTris; t += stride) {
          const v0 = f[3 * t], v1 = f[3 * t + 1], v2 = f[3 * t + 2];
          let rgb = baseRgb;
          let triAlpha = surfAlpha;
          if (useAntinodeFilter && sc) {
            const sMid = (sc[v0] + sc[v1] + sc[v2]) / 3.0;
            if (Math.abs(sMid) < tau) continue;
            rgb = sMid >= 0.0 ? [251, 174, 36] : [198, 98, 252];
            triAlpha = Math.min(0.92, surfAlpha * 1.18);
          }
          const i0 = v0 * 3, i1 = v1 * 3, i2 = v2 * 3;
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
            fill: `rgba(${Math.round(rgb[0] * shade)},${Math.round(rgb[1] * shade)},${Math.round(rgb[2] * shade)},${triAlpha.toFixed(2)})`,
          });
        }
      }
      if (hasRe) collectSoftwareTris(decodedMeshes.re, [16, 205, 238], false);
      if (hasIm) collectSoftwareTris(decodedMeshes.im, [246, 78, 125], antinodeOn);
      const fgItems = [];
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
            fgItems.push({ kind: "seg", z: 0.5 * (p0[2] + p1[2]), p0, p1 });
          }
        }
      }
      if (hasMorse) {
        const mList = decodedMeshes.morseList || [];
        for (let mi = 0; mi < mList.length; mi++) {
          const mc = mList[mi];
          const reVal = Number(mc.re_val) || 0.0;
          if (Math.abs(reVal) < tau) continue;
          const pos = mc.pos_g;
          if (!pos || !pointInsideLiveRoi(pos, liveRoi)) continue;
          const ax = mc.axis_g || [0, 0, 1];
          const L = Math.max(0.0, Number(mc.half_len_g) || 0.0);
          const rG = Math.max(1e-5, Number(mc.radius_g) || 0.003);
          const pA = projectPoint3D([pos[0] - L * ax[0], pos[1] - L * ax[1], pos[2] - L * ax[2]], Rtot, center, invScale, cx, cy, Rpx, zoom);
          const pB = projectPoint3D([pos[0] + L * ax[0], pos[1] + L * ax[1], pos[2] + L * ax[2]], Rtot, center, invScale, cx, cy, Rpx, zoom);
          const pM = projectPoint3D(pos, Rtot, center, invScale, cx, cy, Rpx, zoom);
          if (!pM[4]) continue;
          const rPx = Math.max(3.2, Math.min(22.0, rG * invScale * Rpx * zoom * pM[3]));
          fgItems.push({ kind: "capsule", z: pM[2], pA, pB, rPx, isPos: reVal >= 0.0 });
        }
      }
      items.sort((a, b) => a.z - b.z);
      fgItems.sort((a, b) => a.z - b.z);
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
      for (let i = 0; i < fgItems.length; i++) {
        const it = fgItems[i];
        if (it.kind === "seg") {
          ctx.beginPath();
          ctx.moveTo(it.p0[0], it.p0[1]);
          ctx.lineTo(it.p1[0], it.p1[1]);
          ctx.strokeStyle = "rgba(7, 9, 19, 0.88)";
          ctx.lineWidth = isExp ? 4.4 : 3.4;
          ctx.stroke();
          ctx.strokeStyle = "#fde047";
          ctx.lineWidth = isExp ? 2.4 : 1.8;
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.moveTo(it.pA[0], it.pA[1]);
          ctx.lineTo(it.pB[0] + (Math.hypot(it.pB[0] - it.pA[0], it.pB[1] - it.pA[1]) < 0.2 ? 0.01 : 0.0), it.pB[1]);
          ctx.strokeStyle = "rgba(7, 9, 19, 0.90)";
          ctx.lineWidth = 2.0 * it.rPx + 2.2;
          ctx.stroke();
          ctx.strokeStyle = it.isPos ? "#f59e0b" : "#c084fc";
          ctx.lineWidth = 2.0 * it.rPx;
          ctx.stroke();
          ctx.strokeStyle = it.isPos ? "#fef08a" : "#f5d0fe";
          ctx.lineWidth = Math.max(1.4, 0.72 * it.rPx);
          ctx.stroke();
        }
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
      const hovMorseKey = hoveredMorseObj && Array.isArray(hoveredMorseObj.pos_g)
        ? hoveredMorseObj.pos_g.join(",")
        : "";
      const sig = [
        targetW, targetH,
        (st.wulffYawDeg || 0).toFixed(3),
        (st.wulffPitchDeg || 0).toFixed(3),
        zoom3d.toFixed(4),
        st.wulffExpanded ? 1 : 0,
        st.wulffScope,
        st.wulffColorMode,
        Number(st.colorPercentile ?? 0.50).toFixed(3),
        deps.getColorCrossover().toFixed(3),
        st.showWulffFriedel ? 1 : 0,
        st.showNodalPoints ? 1 : 0,
        st.showNodalSurfacesRe ? 1 : 0,
        st.showNodalSurfacesIm ? 1 : 0,
        st.showPhaseVortices ? 1 : 0,
        st.showAntinodes ? 1 : 0,
        st.showMorseCapsules ? 1 : 0,
        Number(st.antinodePercentile ?? 0.75).toFixed(3),
        Number(st.morseMahalSigma ?? 1.0).toFixed(3),
        hovMorseKey,
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

      // 3. Depth-buffered Two-color 2D Nodal Surfaces (Re=0 Cyan & Im=0 Coral / Antinodes) + 1D Vortices + 3D Morse Capsules
      const antinodeStats = (nodalData && nodalData.has_phases)
        ? resolveAntinodeThreshold(liveRoi)
        : { pct: 0.75, tau: 0.0, maxAbs: 1.0, activeMorseCount: 0 };
      renderSurfacesAndVortices3D(ctx, w, h, dpr, Rtot, center, invScale, Rpx, zoom, liveRoi, isExp, antinodeStats);

      // Record screen-space 3D Morse capsule hit targets & draw hover callout when active
      screenMorseHits3D = [];
      if (nodalData && nodalData.has_phases && st.showMorseCapsules && decodedMeshes.morseList) {
        const mList = decodedMeshes.morseList;
        for (let mi = 0; mi < mList.length; mi++) {
          const mc = mList[mi];
          if (Math.abs(Number(mc.re_val) || 0.0) < antinodeStats.tau) continue;
          const pos = mc.pos_g;
          if (!pos || !pointInsideLiveRoi(pos, liveRoi)) continue;
          const prM = projectPoint3D(pos, Rtot, center, invScale, cx, cy, Rpx, zoom);
          if (!prM[4]) continue;
          const rG = Math.max(1e-5, Number(mc.radius_g) || 0.003);
          const rPx = Math.max(8.5, Math.min(24.0, rG * invScale * Rpx * zoom * prM[3] + 5.0));
          screenMorseHits3D.push({ morse: mc, x: prM[0], y: prM[1], r: rPx, z: prM[2] });
        }
        screenMorseHits3D.sort((a, b) => a.z - b.z);
      }

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
        const morseThresh3D = (nodalData && nodalData.has_phases && st.showMorseCapsules)
          ? resolveDetectorMorseThreshold()
          : null;
        drawList.sort((a, b) => a.z - b.z);
        ctx.save();
        for (let i = 0; i < drawList.length; i++) {
          const it = drawList[i];
          if (it.x < -20 || it.x > w + 20 || it.y < -20 || it.y > h + 20) continue;
          const rBase = (isExp ? 3.5 : 2.7) * Math.max(0.65, Math.min(1.55, it.sc));
          const rad = it.isFocus ? rBase + 2.4 : (it.isFriedel ? rBase * 0.78 : rBase);
          const col = deps.getWulffPeakColor(it.p);
          const mHit = (!it.isFriedel && morseThresh3D) ? getPeakMorseMatch(it.p.hkl, morseThresh3D) : null;
          if (mHit) {
            ctx.beginPath();
            ctx.arc(it.x, it.y, rad + 2.6, 0, Math.PI * 2);
            ctx.strokeStyle = mHit.morse.re_val >= 0.0 ? "#f59e0b" : "#c084fc";
            ctx.globalAlpha = 0.94;
            ctx.lineWidth = isExp ? 1.8 : 1.4;
            ctx.stroke();
          }
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
        if (st.showAntinodes) {
          const pctTxt = Math.round(antinodeStats.pct * 100);
          pills.push({
            txt: nodalData.is_real_only
              ? `Antinodes |Re[${fLbl}]|≥P${pctTxt} (centric)`
              : `Im[${fLbl}]=0 Antinodes |Re|≥P${pctTxt}`,
            col: "#f59e0b",
          });
        } else if (st.showNodalSurfacesIm) {
          pills.push({
            txt: nodalData.is_real_only ? `Im[${fLbl}]≡0 (centric)` : `Im[${fLbl}]=0`,
            col: "#fb7185",
          });
        }
        if (st.showPhaseVortices) pills.push({ txt: `1D Vortices (${nVor} in ROI)`, col: "#fde047" });
        if (st.showMorseCapsules) {
          const mahalTxt = Number(st.morseMahalSigma ?? 1.0).toFixed(2).replace(/0$/, "");
          pills.push({
            txt: `Morse ∇Re=0 (${antinodeStats.activeMorseCount} capsules • dM≤${mahalTxt}σ)`,
            col: "#e879f9",
          });
        }
        if (brickFetchInProgress) pills.push({ txt: "Streaming 3D brick…", col: "#94a3b8" });
      } else {
        pills.push({ txt: "3D Reciprocal Lattice (load phased .mtz for Re/Im surfaces & vortices)", col: "#94a3b8" });
      }

      if (hoveredMorseObj && st.showMorseCapsules && Array.isArray(hoveredMorseObj.pos_g)) {
        const prH = projectPoint3D(hoveredMorseObj.pos_g, Rtot, center, invScale, cx, cy, Rpx, zoom);
        if (prH[4]) {
          const reV = Number(hoveredMorseObj.re_val) || 0.0;
          const nh = hoveredMorseObj.nearest_hkl || [0, 0, 0];
          const sgnStr = reV >= 0 ? "+" : "";
          const callout = `∇Re=0 (${sgnStr}${reV.toPrecision(3)}) • near (${nh.join(",")})`;
          ctx.save();
          ctx.beginPath();
          ctx.arc(prH[0], prH[1], isExp ? 9.5 : 7.5, 0, Math.PI * 2);
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.8;
          ctx.stroke();
          ctx.font = isExp ? "bold 10px monospace" : "bold 8.5px monospace";
          const cw = ctx.measureText(callout).width + 10;
          const lx = Math.min(w - cw - 6, Math.max(6, prH[0] + 10));
          const ly = Math.max(18, prH[1] - 8);
          ctx.fillStyle = "rgba(7, 9, 19, 0.90)";
          ctx.strokeStyle = reV >= 0 ? "#f59e0b" : "#c084fc";
          ctx.lineWidth = 1.1;
          ctx.beginPath();
          if (typeof ctx.roundRect === "function") ctx.roundRect(lx, ly - 11, cw, 15, 3);
          else ctx.rect(lx, ly - 11, cw, 15);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = reV >= 0 ? "#fde68a" : "#f5d0fe";
          ctx.fillText(callout, lx + 5, ly);
          ctx.restore();
        }
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

    function findNearestMorse3D(mx, my) {
      let best = null;
      let bestScore = 16.0;
      for (let i = screenMorseHits3D.length - 1; i >= 0; i--) {
        const h = screenMorseHits3D[i];
        const d = Math.hypot(mx - h.x, my - h.y);
        if (d <= h.r && d < bestScore) {
          bestScore = d;
          best = h.morse;
        }
      }
      return best;
    }

    function setHoveredMorse(mObj) {
      const next = mObj || null;
      if (hoveredMorseObj !== next) {
        hoveredMorseObj = next;
        last3DRenderSig = "";
        return true;
      }
      return false;
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

    function projectCrystalGToDetector(gVec, Rg, ki, sLab, fullPanels, isSph) {
      const gn = Math.hypot(gVec[0], gVec[1], gVec[2]);
      if (!(gn > 1e-7)) return null;
      const gLab = deps.mat3Vec(Rg, gVec);
      const sinTh = -(ki[0] * gLab[0] + ki[1] * gLab[1] + ki[2] * gLab[2]) / gn;
      if (!(sinTh > 1e-4 && sinTh < 0.9999)) return null;
      const invGn = 1.0 / gn;
      const kfx = ki[0] + 2.0 * sinTh * gLab[0] * invGn;
      const kfy = ki[1] + 2.0 * sinTh * gLab[1] * invGn;
      const kfz = ki[2] + 2.0 * sinTh * gLab[2] * invGn;
      const kfn = Math.hypot(kfx, kfy, kfz) || 1.0;
      const kf = [kfx / kfn, kfy / kfn, kfz / kfn];
      if (ki[0] * kf[0] + ki[1] * kf[1] + ki[2] * kf[2] > 0.9995) return null;
      const proj = deps.projectRayToDetectorWithGap(kf, sLab, fullPanels);
      if (!proj || !proj.onPanel) return null;
      return {
        detX: proj.det_x,
        detV: isSph ? proj.det_el : proj.det_y,
        bank: proj.bank,
      };
    }

    // Map full-sphere 3D Morse extrema (∇Re W = 0) onto the 2D Detector whenever a Bragg peak
    // falls within anisotropic Mahalanobis distance d_M(h_hkl, h_*) <= state.morseMahalSigma
    // and |Re W(h_*)| >= τ(p). Evaluates across ALL active-bandpass detector peaks (immune to zoom LOD).
    function renderDetectorMorseExtrema(ctx, w, h, detOpts) {
      screenMorseHitsDet = [];
      const st = deps.state;
      if (!nodalData || !nodalData.has_phases || !st.showMorseCapsules) return 0;
      ensureDecodedMeshes();
      if (detectorMorseMap.size === 0) return 0;

      const peaks = (detOpts && Array.isArray(detOpts.peaks)) ? detOpts.peaks : [];
      if (peaks.length === 0) return 0;

      const thresh = resolveDetectorMorseThreshold();
      const ub = deps.getActiveUbMatrix();
      const Rg = deps.getActiveGoniometerR();
      const ki = deps.getActiveIncidentBeam();
      const sLab = deps.getActiveSampleOffsetLab();
      const fullPanels = deps.buildFullFaceDetectorRaycastPanels(sLab);
      const isSph = deps.isSphericalDetectorView();

      const hasAffine = Boolean(
        detOpts &&
        Number.isFinite(detOpts.affScaleX) &&
        Number.isFinite(detOpts.affOffsetX) &&
        Number.isFinite(detOpts.affScaleY) &&
        Number.isFinite(detOpts.affOffsetY)
      );
      const affScaleX = hasAffine ? detOpts.affScaleX : 1.0;
      const affOffsetX = hasAffine ? detOpts.affOffsetX : 0.0;
      const affScaleY = hasAffine ? detOpts.affScaleY : 1.0;
      const affOffsetY = hasAffine ? detOpts.affOffsetY : 0.0;
      const baseR = (detOpts && Number.isFinite(detOpts.baseR)) ? detOpts.baseR : 3.2;
      const getVert = (detOpts && typeof detOpts.getPeakVert === "function")
        ? detOpts.getPeakVert
        : ((p) => (isSph ? p.det_el : p.det_y));
      const isHoveredFn = (detOpts && typeof detOpts.isPeakHighlighted === "function")
        ? detOpts.isPeakHighlighted
        : (() => false);

      const shiftList = [0.0, -360.0, 360.0];
      let drawnCount = 0;

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (let pIdx = 0; pIdx < peaks.length; pIdx++) {
        const p = peaks[pIdx];
        if (!p || !Array.isArray(p.hkl) || p.hkl.length !== 3) continue;
        const match = getPeakMorseMatch(p.hkl, thresh);
        if (!match) continue;

        const m = match.morse;
        const isPos = m.re_val >= 0.0;
        const isHov = Boolean(isHoveredFn(p));
        const py = getVert(p);

        // Compute Laue-projected 2D uniaxial ridge direction & foreshortening from 3D axis_g
        let dirAz = 1.0;
        let dirV = 0.0;
        let hasPanelDir = false;
        let effAspect = 1.0;
        let starProj = null;

        if (ub && Array.isArray(m.axis_g) && m.axis_g.length === 3) {
          const gx = ub[0][0] * p.hkl[0] + ub[0][1] * p.hkl[1] + ub[0][2] * p.hkl[2];
          const gy = ub[1][0] * p.hkl[0] + ub[1][1] * p.hkl[1] + ub[1][2] * p.hkl[2];
          const gz = ub[2][0] * p.hkl[0] + ub[2][1] * p.hkl[1] + ub[2][2] * p.hkl[2];
          const gn = Math.hypot(gx, gy, gz);
          if (gn > 1e-6) {
            const ax = m.axis_g;
            const axNorm = Math.hypot(ax[0], ax[1], ax[2]) || 1.0;
            const ax0 = ax[0] / axNorm, ax1 = ax[1] / axNorm, ax2 = ax[2] / axNorm;
            const dotRadial = Math.max(-1.0, Math.min(1.0, (ax0 * gx + ax1 * gy + ax2 * gz) / gn));
            const sinPsi = Math.sqrt(Math.max(0.0, 1.0 - dotRadial * dotRadial));
            const rawAspect = Math.max(1.0, Math.min(3.2, Number(m.aspect) || 1.0));
            effAspect = 1.0 + (rawAspect - 1.0) * sinPsi;

            if (effAspect > 1.04 && fullPanels && fullPanels.length > 0) {
              const stepG = 0.06 * gn;
              const prPlus = projectCrystalGToDetector(
                [gx + stepG * ax0, gy + stepG * ax1, gz + stepG * ax2],
                Rg, ki, sLab, fullPanels, isSph
              );
              const prMinus = projectCrystalGToDetector(
                [gx - stepG * ax0, gy - stepG * ax1, gz - stepG * ax2],
                Rg, ki, sLab, fullPanels, isSph
              );
              if (prPlus && prMinus && prPlus.bank === prMinus.bank && Math.abs(prPlus.detX - prMinus.detX) < 25.0) {
                dirAz = prPlus.detX - prMinus.detX;
                dirV = prPlus.detV - prMinus.detV;
                hasPanelDir = (Math.hypot(dirAz, dirV) > 1e-7);
              }
            }

            if (Array.isArray(m.hkl_cont) && m.hkl_cont.length === 3 && fullPanels && fullPanels.length > 0) {
              const gsx = ub[0][0] * m.hkl_cont[0] + ub[0][1] * m.hkl_cont[1] + ub[0][2] * m.hkl_cont[2];
              const gsy = ub[1][0] * m.hkl_cont[0] + ub[1][1] * m.hkl_cont[1] + ub[1][2] * m.hkl_cont[2];
              const gsz = ub[2][0] * m.hkl_cont[0] + ub[2][1] * m.hkl_cont[1] + ub[2][2] * m.hkl_cont[2];
              starProj = projectCrystalGToDetector([gsx, gsy, gsz], Rg, ki, sLab, fullPanels, isSph);
            }
          }
        }

        const rCap = Math.max(5.5, Math.min(13.5, baseR + (isHov ? 4.8 : 3.6)));
        const halfL = rCap * Math.min(1.75, Math.max(0.0, effAspect - 1.0));
        const proxFrac = Math.max(0.0, Math.min(1.0, match.dMahal / Math.max(0.15, thresh.maxMahal)));
        const rimAlpha = isHov ? 1.0 : (0.96 - 0.24 * proxFrac);

        let anyCopyOnScreen = false;
        for (let sIdx = 0; sIdx < 3; sIdx++) {
          const shiftAz = shiftList[sIdx];
          let x = 0.0, y = 0.0;
          if (hasAffine) {
            x = affScaleX * (p.det_x + shiftAz) + affOffsetX;
            y = affScaleY * py + affOffsetY;
          } else {
            const sc = deps.detToScreen(p.det_x + shiftAz, py, w, h, 0.45);
            x = sc[0];
            y = sc[1];
          }
          if (x < -44 || x > w + 44 || y < -44 || y > h + 44) continue;
          anyCopyOnScreen = true;

          let ux = 1.0, uy = 0.0;
          if (hasPanelDir) {
            const sxDir = hasAffine ? (affScaleX * dirAz) : dirAz;
            const syDir = hasAffine ? (affScaleY * dirV) : -dirV;
            const sNorm = Math.hypot(sxDir, syDir);
            if (sNorm > 1e-7) {
              ux = sxDir / sNorm;
              uy = syDir / sNorm;
            }
          }

          // Subtle tether + crest pip when projected continuous extremum G* lands near the reflection on-panel
          if (starProj && Math.abs(starProj.detX - p.det_x) < 18.0) {
            let xs = 0.0, ys = 0.0;
            if (hasAffine) {
              xs = affScaleX * (starProj.detX + shiftAz) + affOffsetX;
              ys = affScaleY * starProj.detV + affOffsetY;
            } else {
              const scS = deps.detToScreen(starProj.detX + shiftAz, starProj.detV, w, h, 0.45);
              xs = scS[0];
              ys = scS[1];
            }
            const dsPx = Math.hypot(xs - x, ys - y);
            if (dsPx >= 2.4 && dsPx <= 32.0) {
              ctx.save();
              ctx.globalAlpha = 0.72 * rimAlpha;
              ctx.setLineDash([2, 2]);
              ctx.beginPath();
              ctx.moveTo(x, y);
              ctx.lineTo(xs, ys);
              ctx.strokeStyle = isPos ? "#fde047" : "#f0abfc";
              ctx.lineWidth = 1.15;
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.beginPath();
              ctx.arc(xs, ys, 2.3, 0, Math.PI * 2);
              ctx.fillStyle = "rgba(7, 9, 19, 0.92)";
              ctx.fill();
              ctx.beginPath();
              ctx.arc(xs, ys, 1.5, 0, Math.PI * 2);
              ctx.fillStyle = isPos ? "#fef08a" : "#f5d0fe";
              ctx.fill();
              ctx.restore();
            }
          }

          const epsSeg = halfL < 0.08 ? 0.01 : 0.0;
          const axA = x - halfL * ux;
          const ayA = y - halfL * uy;
          const axB = x + halfL * ux + epsSeg;
          const ayB = y + halfL * uy;

          // Stadium translucent interior tint
          ctx.globalAlpha = isHov ? 0.24 : 0.14;
          ctx.beginPath();
          ctx.moveTo(axA, ayA);
          ctx.lineTo(axB, ayB);
          ctx.strokeStyle = isPos ? "#f59e0b" : "#c084fc";
          ctx.lineWidth = 2.0 * rCap;
          ctx.stroke();

          // Crisp uniaxial stadium rim: dark contrast halo + Gold/Violet outer stroke + bright crest highlight
          const ang = Math.atan2(uy, ux);
          ctx.save();
          ctx.translate(x, y);
          ctx.rotate(ang);
          ctx.beginPath();
          ctx.arc(-halfL, 0, rCap, Math.PI * 0.5, Math.PI * 1.5);
          ctx.arc(halfL, 0, rCap, -Math.PI * 0.5, Math.PI * 0.5);
          ctx.closePath();

          ctx.globalAlpha = rimAlpha;
          ctx.strokeStyle = "rgba(7, 9, 19, 0.90)";
          ctx.lineWidth = isHov ? 4.4 : 3.6;
          ctx.stroke();

          ctx.strokeStyle = isPos ? "#f59e0b" : "#c084fc";
          ctx.lineWidth = isHov ? 2.5 : 2.0;
          ctx.stroke();

          ctx.strokeStyle = isPos ? "#fef08a" : "#f5d0fe";
          ctx.lineWidth = isHov ? 1.15 : 0.85;
          ctx.stroke();
          ctx.restore();

          screenMorseHitsDet.push({
            peak: p,
            morse: m,
            dMahal: match.dMahal,
            x,
            y,
            r: rCap + halfL + 3.5,
          });
        }
        if (anyCopyOnScreen) drawnCount++;
      }

      ctx.globalAlpha = 1.0;
      ctx.restore();
      return drawnCount;
    }

    function findNearestDetectorMorsePeak(mx, my) {
      if (!deps.state || !deps.state.showMorseCapsules || screenMorseHitsDet.length === 0) {
        return null;
      }
      let best = null;
      let bestDist = 1e9;
      for (let i = 0; i < screenMorseHitsDet.length; i++) {
        const h = screenMorseHitsDet[i];
        const d = Math.hypot(mx - h.x, my - h.y);
        if (d <= h.r && d < bestDist) {
          bestDist = d;
          best = h.peak;
        }
      }
      return best;
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
      findNearestMorse3D,
      setHoveredMorse,
      getHoveredMorse: () => hoveredMorseObj,
      resolveAntinodeThreshold,
      resolveDetectorMorseThreshold,
      getPeakMorseMatch,
      renderWulffVortices,
      renderDetectorVortices,
      renderDetectorMorseExtrema,
      findNearestDetectorMorsePeak,
      requestNodalUpdate,
      scheduleRoiRefresh,
      uploadAndSetRefMtz,
    };
  }
"""
