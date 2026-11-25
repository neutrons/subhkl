"""Tests for 3D complex structure-factor nodal surfaces and 1D phase vortices."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import gemmi
import numpy as np
import pytest

from subhkl.viz.nodal_surfaces import (
    compute_nodal_surfaces_payload,
    detect_mtz_f_and_phase_columns,
    evaluate_field_on_hkl_grid,
    load_phased_structure_factor_field,
    process_nodal_surfaces_api_request,
)
from subhkl.viz.rainbow_zones import (
    apply_mtz_to_payload,
    build_frame_only_response,
    build_rainbow_zone_payload,
    merge_frame_only_payload,
    render_rainbow_html,
)


def _write_synthetic_phased_mtz(
    path: Path,
    *,
    spacegroup_name: str,
    cell_params: tuple[float, float, float, float, float, float],
    atoms_frac_and_b: list[tuple[tuple[float, float, float], float]],
    d_min: float = 2.0,
    include_phases: bool = True,
) -> tuple[np.ndarray, dict[tuple[int, int, int], complex]]:
    """Write a crystallographic MTZ from exact atomic structure factors across the ASU."""
    sg = gemmi.SpaceGroup(spacegroup_name)
    uc = gemmi.UnitCell(*cell_params)
    ops = list(sg.operations())

    # Expand asymmetric-unit atoms by space-group operations into full unit cell
    unit_cell_atoms: list[tuple[np.ndarray, float]] = []
    for xyz, b_val in atoms_frac_and_b:
        pos = np.asarray(xyz, dtype=np.float64)
        for op in ops:
            rot = np.asarray(op.rot, dtype=np.float64) / float(op.DEN)
            tran = np.asarray(op.tran, dtype=np.float64) / float(op.DEN)
            sym_pos = np.mod(rot @ pos + tran, 1.0)
            unit_cell_atoms.append((sym_pos, float(b_val)))

    # Enumerate reflections inside d >= d_min and reduce to reciprocal ASU
    h_lim = int(np.floor(cell_params[0] / d_min)) + 1
    k_lim = int(np.floor(cell_params[1] / d_min)) + 1
    l_lim = int(np.floor(cell_params[2] / d_min)) + 1

    asu = gemmi.ReciprocalAsu(sg)
    full_f_calc: dict[tuple[int, int, int], complex] = {}
    asu_rows: dict[tuple[int, int, int], tuple[float, float]] = {}

    for h in range(-h_lim, h_lim + 1):
        for k in range(-k_lim, k_lim + 1):
            for l_idx in range(-l_lim, l_lim + 1):
                if h == 0 and k == 0 and l_idx == 0:
                    continue
                d_val = uc.calculate_d([h, k, l_idx])
                if d_val < d_min:
                    continue
                hvec = np.array([h, k, l_idx], dtype=np.float64)
                f_val = 0.0j
                for sym_pos, b_val in unit_cell_atoms:
                    f_val += b_val * np.exp(2j * np.pi * float(np.dot(hvec, sym_pos)))
                if sg.operations().is_systematically_absent([h, k, l_idx]):
                    assert abs(f_val) < 1e-9
                    continue
                full_f_calc[(h, k, l_idx)] = f_val
                if asu.is_in([h, k, l_idx]):
                    amp = float(abs(f_val))
                    phi_deg = float(np.rad2deg(np.angle(f_val)) % 360.0)
                    asu_rows[(h, k, l_idx)] = (amp, phi_deg)

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = sg
    mtz.set_cell_for_all(uc)
    mtz.add_column("F_calc", "F")
    if include_phases:
        mtz.add_column("PHIF_calc", "P")

    rows = []
    for (h, k, l_idx), (amp, phi_deg) in sorted(asu_rows.items()):
        if include_phases:
            rows.append([h, k, l_idx, amp, phi_deg])
        else:
            rows.append([h, k, l_idx, amp])
    mtz.set_data(np.asarray(rows, dtype=np.float32))
    mtz.write_to_file(str(path))

    ub = np.diag([1.0 / cell_params[0], 1.0 / cell_params[1], 1.0 / cell_params[2]])
    return ub, full_f_calc


def test_screw_axis_phase_expansion_and_friedel_hermiticity(tmp_path: Path):
    """4_1 screw-axis ASU phase expansion must match exact structure factors and satisfy F(-h) = conj(F(h))."""
    mtz_path = tmp_path / "chiral_p41.mtz"
    _, full_f = _write_synthetic_phased_mtz(
        mtz_path,
        spacegroup_name="P 41",
        cell_params=(10.0, 10.0, 12.0, 90.0, 90.0, 90.0),
        atoms_frac_and_b=[
            ((0.17, 0.31, 0.09), 6.6),
            ((0.41, 0.12, 0.28), -3.74),
        ],
        d_min=2.4,
    )

    f_col, phi_col = detect_mtz_f_and_phase_columns(gemmi.read_mtz_file(str(mtz_path)))
    assert f_col is not None and f_col.label == "F_calc"
    assert phi_col is not None and phi_col.label == "PHIF_calc"

    field = load_phased_structure_factor_field(str(mtz_path))
    assert field["has_phases"] is True
    q75 = float(field["q75_f"])
    assert q75 > 0.0

    # Check non-ASU screw-related reflections at integer lattice nodes
    hv = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    Z_int = evaluate_field_on_hkl_grid(field, hv, hv, hv)
    checked = 0
    for i, h in enumerate(hv.astype(int)):
        for j, k in enumerate(hv.astype(int)):
            for m, l_idx in enumerate(hv.astype(int)):
                if (h, k, l_idx) in full_f:
                    expected = full_f[(h, k, l_idx)] / q75
                    # Float32 MTZ storage gives ~1e-6 precision
                    assert np.isclose(Z_int[i, j, m], expected, atol=2e-4, rtol=2e-4)
                    checked += 1
    assert checked >= 20

    # Check continuous Friedel Hermiticity Z(-h) == conj(Z(h)) at arbitrary fractional coordinates
    frac = np.linspace(-1.7, 1.7, 9)
    Z_frac = evaluate_field_on_hkl_grid(field, frac, frac, frac)
    assert np.allclose(Z_frac, np.conj(Z_frac[::-1, ::-1, ::-1]), atol=1e-10)


def test_re_im_nodal_surfaces_and_centric_guard(tmp_path: Path):
    """Acentric crystals yield Re=0 and Im=0 meshes; centric crystals flag is_real_only with empty Im mesh."""
    acentric_mtz = tmp_path / "acentric_p212121.mtz"
    ub, _ = _write_synthetic_phased_mtz(
        acentric_mtz,
        spacegroup_name="P 21 21 21",
        cell_params=(9.0, 10.5, 11.5, 90.0, 90.0, 90.0),
        atoms_frac_and_b=[
            ((0.13, 0.27, 0.19), 5.8),
            ((0.36, 0.09, 0.43), -3.74),
        ],
        d_min=2.3,
    )

    nodal = compute_nodal_surfaces_payload(
        mtz_path=str(acentric_mtz),
        ub_matrix=ub,
        d_min=2.3,
        roi_mode="origin",
        roi_radius_mult=2.6,
        roi_grid_size=28,
        vortex_grid_size=28,
    )
    assert nodal["has_phases"] is True
    assert nodal["is_real_only"] is False
    assert nodal["re_surface"]["n_tris"] > 20
    assert nodal["im_surface"]["n_tris"] > 20

    # Verify decoded normal vectors are unit-length
    nrm_bytes = base64.b64decode(nodal["re_surface"]["norms_b64"])
    nrms = np.frombuffer(nrm_bytes, dtype=np.float32).reshape(-1, 3)
    norms = np.linalg.norm(nrms, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

    # Centric P -1 crystal has phases strictly in {0, 180} deg -> Im(F) == 0 everywhere
    centric_mtz = tmp_path / "centric_pbar1.mtz"
    ub_c, _ = _write_synthetic_phased_mtz(
        centric_mtz,
        spacegroup_name="P -1",
        cell_params=(9.0, 10.0, 11.0, 90.0, 90.0, 90.0),
        atoms_frac_and_b=[
            ((0.21, 0.34, 0.15), 6.0),
            ((0.42, 0.11, 0.37), -3.74),
        ],
        d_min=2.4,
    )
    nodal_c = compute_nodal_surfaces_payload(
        mtz_path=str(centric_mtz),
        ub_matrix=ub_c,
        d_min=2.4,
        roi_grid_size=24,
        vortex_grid_size=24,
    )
    assert nodal_c["has_phases"] is True
    assert nodal_c["is_real_only"] is True
    assert nodal_c["re_surface"]["n_tris"] > 10
    assert nodal_c["im_surface"]["n_tris"] == 0
    assert len(nodal_c["vortices"]) == 0


def test_phase_vortices_lie_on_simultaneous_zero_locus_and_roi_modes(tmp_path: Path):
    """Extracted 1D phase vortices must satisfy |Z(G)| << 1 along their polylines and support Peak/Zone ROIs."""
    mtz_path = tmp_path / "vortex_p1.mtz"
    ub, _ = _write_synthetic_phased_mtz(
        mtz_path,
        spacegroup_name="P 1",
        cell_params=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        atoms_frac_and_b=[
            ((0.0, 0.0, 0.0), 1.0),
            ((0.37, 0.19, 0.05), 0.95),
            ((0.11, 0.43, 0.29), 0.90),
        ],
        d_min=2.5,
    )
    field = load_phased_structure_factor_field(str(mtz_path))
    nodal = compute_nodal_surfaces_payload(
        mtz_path=str(mtz_path),
        ub_matrix=ub,
        d_min=2.5,
        roi_mode="origin",
        roi_radius_mult=2.8,
        roi_grid_size=30,
        vortex_grid_size=32,
    )
    vortices = nodal["vortices"]
    assert len(vortices) >= 2

    inv_ub = np.linalg.inv(ub)
    # Sample interior points along extracted vortices and evaluate exact continuous |Z(h)|
    mags = []
    for v in vortices[:8]:
        pts_g = np.asarray(v["pts_g"], dtype=np.float64)
        for g_pt in pts_g[1:-1:2]:
            h_pt = inv_ub @ g_pt
            z_pt = evaluate_field_on_hkl_grid(
                field,
                np.array([h_pt[0]]),
                np.array([h_pt[1]]),
                np.array([h_pt[2]]),
            )[0, 0, 0]
            mags.append(float(abs(z_pt)))
    assert len(mags) > 0
    assert float(np.median(mags)) < 0.03

    # Verify Peak-centered and Zone-slab ROI scoping
    nodal_peak = compute_nodal_surfaces_payload(
        mtz_path=str(mtz_path),
        ub_matrix=ub,
        d_min=2.5,
        roi_mode="peak",
        roi_center_hkl=[1, 1, 0],
        roi_radius_mult=2.0,
        roi_grid_size=22,
        vortex_grid_size=24,
    )
    assert nodal_peak["roi"]["mode"] == "peak"
    assert np.allclose(
        nodal_peak["roi"]["center_g"], ub @ np.array([1.0, 1.0, 0.0]), atol=1e-4
    )

    nodal_zone = compute_nodal_surfaces_payload(
        mtz_path=str(mtz_path),
        ub_matrix=ub,
        d_min=2.5,
        roi_mode="zone",
        roi_zone_uvw=[0, 0, 1],
        roi_radius_mult=2.4,
        roi_grid_size=22,
        vortex_grid_size=24,
    )
    assert nodal_zone["roi"]["mode"] == "zone"
    assert nodal_zone["roi"]["slab_half_thickness_g"] is not None


def test_heterodyne_reference_mtz_protonation_interference(tmp_path: Path):
    """Heterodyne F0* * (s*F - F0) isolates single-site protonation shift and switches cleanly via API helper."""
    f0_mtz = tmp_path / "host_f0.mtz"
    f_dnp_mtz = tmp_path / "protonated_f.mtz"

    # Reference host F0 at origin -> F0(h) is positive real everywhere!
    # Adding a proton site at r_H = (0.25, 0.0, 0.0) gives Delta F(h) = b_H * exp(2pi i * 0.25 * h),
    # so Re(F0* * Delta F) = 0 occurs on planes cos(0.5 * pi * h) = 0 -> h = +/-1, +/-3!
    host_atoms = [((0.0, 0.0, 0.0), 10.0), ((0.12, 0.31, 0.19), 2.5)]
    prot_atoms = [*host_atoms, ((0.25, 0.0, 0.0), 3.5)]

    ub, _ = _write_synthetic_phased_mtz(
        f0_mtz,
        spacegroup_name="P 1",
        cell_params=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        atoms_frac_and_b=host_atoms,
        d_min=2.5,
    )
    _write_synthetic_phased_mtz(
        f_dnp_mtz,
        spacegroup_name="P 1",
        cell_params=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        atoms_frac_and_b=prot_atoms,
        d_min=2.5,
    )

    payload = {"ub_matrix": ub.tolist(), "d_min": 2.5}
    res_doc, status, next_ref, next_het = process_nodal_surfaces_api_request(
        {
            "ref_mtz_path": str(f0_mtz),
            "heterodyne_mode": "delta",
            "roi_radius_mult": 2.4,
        },
        mtz_path=str(f_dnp_mtz),
        ref_mtz_path=None,
        heterodyne_mode="delta",
        payload=payload,
    )
    assert status == 200
    assert res_doc["ok"] is True
    assert next_ref == str(f0_mtz)
    assert next_het == "delta"
    nd = res_doc["nodal_data"]
    assert nd["heterodyne_active"] is True
    assert nd["formula_label"] == "F₀*·(F − F₀)"
    assert nd["re_surface"]["n_tris"] > 10

    # Switch to F0* * F product mode
    res_prod, status_p, _, next_het_p = process_nodal_surfaces_api_request(
        {"heterodyne_mode": "product"},
        mtz_path=str(f_dnp_mtz),
        ref_mtz_path=next_ref,
        heterodyne_mode=next_het,
        payload=payload,
    )
    assert status_p == 200
    assert next_het_p == "product"
    assert res_prod["nodal_data"]["formula_label"] == "F₀*·F"

    # Clear reference F0 back to single-field F(G)
    res_clr, status_c, cleared_ref, _ = process_nodal_surfaces_api_request(
        {"clear_ref_mtz": True, "heterodyne_mode": "single"},
        mtz_path=str(f_dnp_mtz),
        ref_mtz_path=next_ref,
        heterodyne_mode=next_het_p,
        payload=payload,
    )
    assert status_c == 200
    assert cleared_ref is None
    assert res_clr["nodal_data"]["heterodyne_active"] is False
    assert res_clr["nodal_data"]["formula_label"] == "F(G)"


def test_apply_mtz_frame_preservation_and_js_controller_projections(tmp_path: Path):
    """apply_mtz_to_payload attaches nodal_data, preserves it across frame switches, and renders in JS."""
    mtz_path = tmp_path / "sim_phased.mtz"
    ub, _ = _write_synthetic_phased_mtz(
        mtz_path,
        spacegroup_name="P 21 21 21",
        cell_params=(10.0, 12.0, 15.0, 90.0, 90.0, 90.0),
        atoms_frac_and_b=[
            ((0.15, 0.25, 0.11), 6.2),
            ((0.38, 0.14, 0.33), -3.74),
        ],
        d_min=2.6,
    )

    base_payload = build_rainbow_zone_payload()
    base_payload["ub_matrix"] = ub.tolist()
    base_payload["u_matrix"] = np.eye(3).tolist()
    base_payload["b_matrix"] = ub.tolist()
    base_payload["cell"] = [10.0, 12.0, 15.0, 90.0, 90.0, 90.0]
    base_payload["space_group"] = "P 21 21 21"
    base_payload["d_min"] = 2.6
    enriched = apply_mtz_to_payload(
        base_payload, str(mtz_path), {"name": "sim_phased.mtz"}
    )
    assert enriched["mtz_has_phases"] is True
    assert enriched["mtz_phi_column"] == "PHIF_calc"
    assert isinstance(enriched.get("nodal_data"), dict)
    assert enriched["nodal_data"]["has_phases"] is True

    # Frame-only response omits heavy nodal_data on wire while merge_frame_only_payload preserves it
    slim = build_frame_only_response(enriched)
    assert "nodal_data" not in slim
    merged = merge_frame_only_payload(enriched, slim)
    assert merged.get("nodal_data") is enriched["nodal_data"]
    assert merged.get("mtz_has_phases") is True

    html = render_rainbow_html(enriched)
    assert 'id="btn-view-recip3d"' in html
    assert 'id="stereo-nodal-bar"' in html
    assert "createNodalSurfacesController" in html

    if shutil.which("node") is None:
        pytest.skip("node not installed")

    from subhkl.viz.nodal_surfaces import get_nodal_surfaces_js

    nd_json = json.dumps(enriched["nodal_data"])
    ub_json = json.dumps(enriched["ub_matrix"])
    js_driver = f"""
    {get_nodal_surfaces_js()}
    global.atob = (b64) => Buffer.from(b64, "base64").toString("binary");
    let strokeCalls = 0;
    let fillCalls = 0;
    const mockCtx = {{
      setTransform() {{}}, clearRect() {{}}, save() {{}}, restore() {{}},
      beginPath() {{}}, closePath() {{}}, moveTo() {{}}, lineTo() {{}}, arc() {{}},
      fillRect() {{}}, strokeRect() {{}}, roundRect() {{}}, clip() {{}}, setLineDash() {{}},
      fillText() {{}}, drawImage() {{}},
      measureText: (txt) => ({{ width: String(txt).length * 6 }}),
      stroke() {{ strokeCalls++; }},
      fill() {{ fillCalls++; }},
      createRadialGradient() {{ return {{ addColorStop() {{}} }}; }},
    }};
    const mockCanvas = {{
      width: 400, height: 400,
      getBoundingClientRect: () => ({{ width: 400, height: 400 }}),
      getContext: () => mockCtx,
    }};
    const state = {{
      showStereo: true,
      stereoViewMode: "recip3d",
      wulffYawDeg: 20, wulffPitchDeg: -15, wulffZoom: 1.0, wulffExpanded: true,
      wulffScope: "1_frame", wulffColorMode: "mtz_fcalc",
      showWulffFriedel: true, showNodalPoints: true, showZoneArcs: true,
      showNodalSurfacesRe: true, showNodalSurfacesIm: true, showPhaseVortices: true,
      nodalSurfaceAlpha: 0.56, nodalRoiMode: "origin",
      dMin: 1.5, lamMin: 0.8, lamMax: 8.0, activeZones: new Set([0]),
    }};
    const ctrl = createNodalSurfacesController({{
      initialNodalData: {nd_json},
      state,
      mat3Mul: (A, B) => A.map((r, i) => [0,1,2].map(j => r[0]*B[0][j] + r[1]*B[1][j] + r[2]*B[2][j])),
      mat3Inv: () => [[10,0,0],[0,12,0],[0,0,15]],
      mat3Vec: (M, v) => [M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2], M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2], M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]],
      getActiveGoniometerR: () => [[1,0,0],[0,1,0],[0,0,1]],
      getActiveUbMatrix: () => {ub_json},
      getActiveIncidentBeam: () => [0, 0, 1],
      getActiveSampleOffsetLab: () => [0, 0, 0],
      buildFullFaceDetectorRaycastPanels: () => [{{ bank: 1 }}],
      projectRayToDetectorWithGap: (kf) => ({{ onPanel: true, det_x: Math.atan2(kf[0], kf[2]) * 57.2958, det_y: kf[1], det_el: Math.asin(kf[1]) * 57.2958, bank: 1 }}),
      isSphericalDetectorView: () => false,
      detToScreen: (az, y) => [200 + az, 200 - y * 100],
      projectToWulff: (g, cx, cy, R) => {{
        const n = Math.hypot(g[0], g[1], g[2]) || 1;
        return [cx + R * (g[0]/n), cy - R * (g[1]/n), g[2] >= 0 ? 1 : -1, g[2]/n, 1.0];
      }},
      getColorCrossover: () => 0.5,
      getActiveZoneObjects: () => [{{ zone_id: 0, uvw: [1, 0, 0], label: "[100]", rainbow_color: "#38bdf8" }}],
      getWulffVisiblePool: () => ({{ visiblePeaks: [{{ id: 1, hkl: [1, 1, 1], d_spacing: 5.0 }}] }}),
      getWulffPeakColor: () => "#22d3ee",
      isSameReflection: (a, b) => a === b,
      setStatusLine: () => {{}},
      onUpdated: () => {{}},
    }});
    ctrl.render3DReciprocalView(mockCanvas, true);
    const strokesAfter3D = strokeCalls;
    const fillsAfter3D = fillCalls;
    ctrl.renderWulffVortices(mockCtx, 200, 200, 160, true);
    const strokesAfterWulff = strokeCalls;
    ctrl.renderDetectorVortices(mockCtx, 800, 500);
    const strokesAfterDet = strokeCalls;
    console.log(JSON.stringify({{ strokesAfter3D, fillsAfter3D, strokesAfterWulff, strokesAfterDet }}));
    """
    js_file = tmp_path / "run_ctrl.js"
    js_file.write_text(js_driver, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(js_file)], capture_output=True, text=True, check=True
    )
    counts = json.loads(proc.stdout.strip())
    assert counts["fillsAfter3D"] > 20
    assert counts["strokesAfterWulff"] > counts["strokesAfter3D"]
    assert counts["strokesAfterDet"] > counts["strokesAfterWulff"]
