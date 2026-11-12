"""Unit tests for the Rainbow Zones interactive widget & Laue zone helpers."""

from __future__ import annotations

import json
import re

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from subhkl.io.parser import app
from subhkl.viz.nova_rainbow_view import RainbowZonesViewModel
from subhkl.viz.rainbow_zones import (
    build_rainbow_zone_payload,
    default_projection_for_instrument,
    export_rainbow_widget,
)
from subhkl.viz.zones import (
    get_zone_conic_points,
    identify_zones,
    stereographic_projection,
)


def _synthetic_orthorhombic_peaks(
    a: float = 10.0,
    b: float = 12.0,
    c: float = 15.0,
    wl_range: tuple[float, float] = (1.8, 4.5),
):
    """Generate synthetic lab-frame peak positions from a known orthorhombic UB."""
    B = np.diag([1.0 / a, 1.0 / b, 1.0 / c])
    # Slight rotation so multiple zones intersect the diffracting hemisphere
    ang = np.deg2rad(18.0)
    U = np.array(
        [
            [np.cos(ang), 0.0, np.sin(ang)],
            [0.0, 1.0, 0.0],
            [-np.sin(ang), 0.0, np.cos(ang)],
        ]
    )
    UB = U @ B
    ki = np.array([0.0, 0.0, 1.0])

    # Generate reflections in the [100] (h=0) and [010] (k=0) zones plus general (hkl)
    hkls = []
    for k in range(-6, 7):
        for l_idx in range(-6, -1):
            if k != 0 or l_idx != 0:
                hkls.append([0, k, l_idx])
                hkls.append([k, 0, l_idx])
    hkl_arr = np.unique(np.asarray(hkls, dtype=int), axis=0)

    G = hkl_arr @ UB.T
    g2 = np.sum(G * G, axis=1)
    lam = -2.0 * (G @ ki) / np.maximum(g2, 1e-12)
    in_band = (G @ ki < -1e-4) & (lam >= wl_range[0]) & (lam <= wl_range[1])

    hkl_kept = hkl_arr[in_band]
    lam_kept = lam[in_band]
    G_kept = G[in_band]
    kf = ki[None, :] + lam_kept[:, None] * G_kept
    kf /= np.linalg.norm(kf, axis=1, keepdims=True)
    # Place on a 0.4 m radius cylindrical detector
    peaks_xyz = kf * 0.4
    return peaks_xyz, U, B, UB, hkl_kept, lam_kept


def test_stereographic_projection_bounds():
    """Unit vectors on S^2 project into the unit disk r <= 1 with finite values."""
    rng = np.random.default_rng(123)
    vecs = rng.normal(size=(200, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    sx, sy, hemi = stereographic_projection(vecs)
    r = np.hypot(sx, sy)

    assert sx.shape == (200,)
    assert sy.shape == (200,)
    assert hemi.shape == (200,)
    assert np.all(np.isfinite(sx))
    assert np.all(np.isfinite(sy))
    assert np.all(r <= 1.0 + 1e-9)
    assert set(np.unique(hemi)).issubset({-1, 1})

    # North pole (0, 0, 1) maps to origin (0, 0) with hemisphere +1
    px, py, ph = stereographic_projection(np.array([[0.0, 0.0, 1.0]]))
    assert px[0] == pytest.approx(0.0, abs=1e-9)
    assert py[0] == pytest.approx(0.0, abs=1e-9)
    assert ph[0] == 1


def test_zone_conic_orthogonality():
    """Every scattering direction G_hat along a zone arc satisfies G_hat . t_hat = 0."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        t_axis = rng.normal(size=3)
        conic = get_zone_conic_points(t_axis, ki=(0.0, 0.0, 1.0), n_arc=360)
        t_hat = conic["t_hat"]
        g_hat = conic["g_hat"]
        dots = g_hat @ t_hat
        assert np.allclose(dots, 0.0, atol=1e-6)
        assert np.all(np.isfinite(conic["det_x"]))
        assert np.all(np.isfinite(conic["det_y"]))


def test_identify_zones_and_payload_schema():
    """build_rainbow_zone_payload identifies [100] and [010] zones and builds Lattice Ladder."""
    peaks_xyz, _U, _B, UB, _, _ = _synthetic_orthorhombic_peaks()

    zones = identify_zones(peaks_xyz, ub_matrix=UB, max_index=1, tolerance_deg=0.5)
    labels = {z["label"] for z in zones}
    assert "[1 0 0]" in labels or "[0 1 0]" in labels

    payload = build_rainbow_zone_payload(
        peaks_xyz=peaks_xyz,
        ub_matrix=UB,
        wavelength_range=(1.8, 4.5),
        instrument="CG4D",
        space_group="P 21 21 21",
        max_zone_index=2,
    )

    for key in (
        "instrument",
        "space_group",
        "wavelength_range",
        "cell",
        "ub_matrix",
        "peaks",
        "zones",
        "ladder_steps",
        "pipeline_stages",
    ):
        assert key in payload

    assert len(payload["peaks"]) == len(peaks_xyz)
    assert len(payload["zones"]) >= 2
    assert len(payload["ladder_steps"]) == 5
    assert len(payload["pipeline_stages"]) == 4
    assert [s["stage_id"] for s in payload["pipeline_stages"]] == [
        "reduce",
        "solve",
        "integrate",
        "merge",
    ]

    # Verify JSON serializability and ladder progression
    raw = json.dumps(payload)
    parsed = json.loads(raw)
    assert parsed["ladder_steps"][0]["rung"] == 1
    assert parsed["ladder_steps"][-1]["rung"] == 5
    assert parsed["peaks"][0]["parallax_depth"] > 0.0


def test_html_widget_export_contains_unified_mode_and_projections(tmp_path):
    """export_rainbow_widget writes a unified HTML with Cylindrical/Spherical projections, single-arc zone curves, and higher-order zones."""
    peaks_xyz, _, _, UB, _, _ = _synthetic_orthorhombic_peaks()
    payload_cg4d = build_rainbow_zone_payload(
        peaks_xyz=peaks_xyz, ub_matrix=UB, instrument="CG4D"
    )
    assert payload_cg4d["default_projection"] == "cylindrical"
    assert default_projection_for_instrument("CG4D") == "cylindrical"
    assert default_projection_for_instrument("MANDI") == "spherical"
    assert any(z.get("order", 1) >= 2 for z in payload_cg4d["zones"])

    out_html = tmp_path / "rainbow_test.html"
    written = export_rainbow_widget(payload_cg4d, out_html)

    assert written == str(out_html)
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")

    assert "window.RAINBOW_ZONES_DATA =" in content
    assert "canvas-detector" in content
    assert "canvas-stereo" in content
    assert "btn-proj-toggle" in content
    assert "btn-zone-order" not in content
    assert "btn-order-1" in content
    assert 'id="nd-ipts" type="text" value=""' in content
    assert 'id="nd-runs" type="text" value=""' in content
    assert 'id="nd-sg" type="text" value=""' in content
    assert '<option value="IMAGINE">' not in content
    assert '<option value="TOPAZ">TOPAZ (SNS)</option>' in content
    assert "TOPAZ" in payload_cg4d["beamline_detectors"]
    assert len(payload_cg4d["beamline_detectors"]["TOPAZ"]) == 25
    assert "btn-zone-arcs" in content
    assert "showZoneArcs: true" in content
    assert "drawSingleZoneArc" in content
    assert "const OVERLAY_PARALLAX_LAYER = 0.45;" in content
    assert (
        "detToScreen(pt[0] + shiftAz, pt[1], w, h, OVERLAY_PARALLAX_LAYER)" in content
    )
    assert "pipeline-stages" in content
    assert "parseH5BuffersInBrowser" in content
    assert "btn-new-tab" in content
    assert "LATTICE LADDER" in content

    # Extract embedded JSON and validate it parses cleanly
    match = re.search(
        r"window\.RAINBOW_ZONES_DATA\s*=\s*(\{.*?\});\n", content, re.DOTALL
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert len(embedded["peaks"]) == len(peaks_xyz)
    assert "det_el" in embedded["peaks"][0]


def test_nova_view_model_and_cli_rainbow_flag(tmp_path):
    """Verify RainbowZonesViewModel and CLI `rainbow-visualize` / `indexer-visualize --rainbow-html`."""
    peaks_xyz, U, B, _, hkl, lam = _synthetic_orthorhombic_peaks()
    n_peaks = len(peaks_xyz)

    h5_path = tmp_path / "synthetic_indexer.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["instrument"] = "MANDI"
        f["bank"] = np.ones(n_peaks, dtype=np.int64)
        f["peaks/pixel_r"] = np.linspace(40.0, 200.0, n_peaks)
        f["peaks/pixel_c"] = np.linspace(50.0, 210.0, n_peaks)
        f["peaks/xyz"] = peaks_xyz
        f["peaks/intensity"] = np.full(n_peaks, 250.0)
        f["peaks/sigma"] = np.full(n_peaks, 3.0)
        f["peaks/h"] = hkl[:, 0]
        f["peaks/k"] = hkl[:, 1]
        f["peaks/l"] = hkl[:, 2]
        f["peaks/lambda"] = lam
        f["peaks/image_index"] = np.zeros(n_peaks, dtype=np.int64)
        f["peaks/run_index"] = np.zeros(n_peaks, dtype=np.int64)
        f["sample/U"] = U
        f["sample/B"] = B
        f["sample/space_group"] = "P 21 21 21"
        f["instrument/wavelength"] = np.array([1.8, 4.5])
        f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])

    vm = RainbowZonesViewModel(replay_h5=h5_path)
    assert vm.payload["instrument"] == "MANDI"
    assert vm.payload["default_projection"] == "spherical"
    assert len(vm.payload["peaks"]) == n_peaks
    assert any(p.get("texture_b64") for p in vm.payload["panels"])
    assert any(p.get("outline_sph") for p in vm.payload["panels"])
    assert "DIFFRACTION PARALLAX" in vm.to_html()

    # Also verify raw detector image HDF5 ingestion (intensity_source == "raw_h5")
    img_h5 = tmp_path / "synthetic.im.h5"
    with h5py.File(img_h5, "w") as f:
        f.create_dataset("images", data=np.ones((1, 64, 64), dtype=np.float32) * 50.0)
        f.create_dataset("bank_ids", data=np.array([1], dtype=np.int32))

    payload_raw = build_rainbow_zone_payload(replay_h5=h5_path, images_h5=img_h5)
    assert payload_raw["intensity_source"] == "raw_h5"
    assert any(p.get("texture_b64") for p in payload_raw["panels"])

    runner = CliRunner()
    cli_html = tmp_path / "cli_rainbow.html"
    res = runner.invoke(
        app,
        [
            "rainbow-visualize",
            str(h5_path),
            "--output-html",
            str(cli_html),
            "--mode",
            "arcade",
        ],
    )
    assert res.exit_code == 0, res.output
    assert cli_html.exists()
    assert cli_html.stat().st_size > 5000

    # Also test --rainbow-html flag on indexer-visualize
    idx_html = tmp_path / "idx_rainbow.html"
    res2 = runner.invoke(
        app,
        [
            "indexer-visualize",
            str(h5_path),
            "--output-dir",
            str(tmp_path),
            "--dpi",
            "50",
            "--rainbow-html",
            str(idx_html),
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert idx_html.exists()


def test_finderless_solve_h5_forward_projection_and_stages(tmp_path):
    """Finder-less solve.h5 (without peaks/xyz) forward-projects reflections and populates solve stage rungs."""
    _, U, B, _, _, _ = _synthetic_orthorhombic_peaks()

    solve_h5 = tmp_path / "synthetic_solve.h5"
    with h5py.File(solve_h5, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["sample/U"] = U
        f["sample/B"] = B
        f["sample/a"] = 18.0
        f["sample/b"] = 24.0
        f["sample/c"] = 32.0
        f["sample/alpha"] = 90.0
        f["sample/beta"] = 90.0
        f["sample/gamma"] = 90.0
        f["sample/space_group"] = "P 21 21 21"
        f["instrument/wavelength"] = np.array([2.0, 4.5])
        f["goniometer/R"] = np.eye(3)[None, :, :]
        f["bank_ids"] = np.array([101], dtype=np.int32)
        f["file_offsets"] = np.array([0], dtype=np.int64)

        bs = f.create_group("blind_search")
        bs.attrs["stable_observations"] = 791552
        bs.attrs["samples_requested"] = 16384
        bs.attrs["weighting"] = "smooth chi2-suppressed bin Poisson deviance"
        bs["proposal_scores"] = np.array([0.0012, 0.0011], dtype=np.float64)
        bs["full_scores"] = np.array([0.481, 0.102], dtype=np.float64)

        sr = f.create_group("scan_refinement")
        sr.attrs["d_min"] = 3.0
        sr.attrs["n_zones"] = 400
        sr.attrs["status"] = "converged"
        sr.attrs["stages"] = json.dumps(
            [
                {
                    "stage": "global",
                    "objective": -0.513,
                    "iterations": 97,
                    "success": True,
                },
                {
                    "stage": "120-zones",
                    "objective": -1.101,
                    "iterations": 192,
                    "success": True,
                },
                {
                    "stage": "cell",
                    "objective": -2.011,
                    "iterations": 235,
                    "success": True,
                },
            ]
        )

    payload = build_rainbow_zone_payload(replay_h5=solve_h5)
    assert payload["instrument"] == "CG4D"
    assert len(payload["peaks"]) > 0
    assert len(payload["zones"]) > 0
    assert "Binned Count Excess" in payload["ladder_steps"][0]["stage"]
    assert "Zone Ladder" in payload["ladder_steps"][1]["stage"]
    assert "60-Zone Global" in payload["ladder_steps"][2]["stage"]
    assert "120→400 Zone + Cell" in payload["ladder_steps"][3]["stage"]
    assert "Converged Solve UB" in payload["ladder_steps"][4]["stage"]
    assert payload["pipeline_stages"][0]["name"] == "1. REDUCE & MERGE"
    assert "imaginex_subhkl_solve" in payload["pipeline_stages"][1]["tool_ids"]


def test_careless_merge_stamping_and_wulff_symmetry_orbits(tmp_path):
    """Careless merge HDF5 stamping populates ASU orbits, residuals, scales, and multi-run sample_peaks."""
    import gemmi

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import render_rainbow_html

    h5_path = tmp_path / "integrated.h5"
    with h5py.File(h5_path, "w") as hf:
        hf["instrument"] = np.bytes_("MANDI")
        hf["space_group"] = np.bytes_("Ia-3d")
        sgrp = hf.create_group("sample")
        for k, v in [
            ("a", 11.93),
            ("b", 11.93),
            ("c", 11.93),
            ("alpha", 90.0),
            ("beta", 90.0),
            ("gamma", 90.0),
        ]:
            sgrp[k] = v
        hf["wavelength_A"] = np.array([2.0, 4.0])
        hf["B"] = np.eye(3) / 11.93
        hf["U"] = np.eye(3)
        hf["files"] = np.array([b"MANDI_12737.nxs.h5", b"MANDI_12739.nxs.h5"])
        hf["bank_ids"] = np.array([10, 10])
        hf["file_offsets"] = np.array([0, 1, 2])
        ggrp = hf.create_group("goniometer")
        R0 = np.eye(3)
        ang = np.deg2rad(25.0)
        R1 = np.array(
            [
                [np.cos(ang), 0.0, np.sin(ang)],
                [0.0, 1.0, 0.0],
                [-np.sin(ang), 0.0, np.cos(ang)],
            ]
        )
        ggrp["R"] = np.stack([R0, R1], axis=0)
        pgrp = hf.create_group("peaks")
        pgrp["xyz"] = np.array(
            [
                [0.3, 0.1, -0.2],
                [0.3, 0.1, -0.2],
                [-0.25, 0.15, -0.1],
                [0.2, -0.1, 0.25],
            ]
        )
        pgrp["h"] = np.array([1, 2, -1, 1], dtype=np.int32)
        pgrp["k"] = np.array([2, 4, 2, -2], dtype=np.int32)
        pgrp["l"] = np.array([3, 6, 3, 3], dtype=np.int32)
        pgrp["run_index"] = np.array([0, 0, 0, 1], dtype=np.int32)
        pgrp["bank"] = np.array([10, 10, 11, 10], dtype=np.int32)
        pgrp["pixel_r"] = np.array([64.0, 64.0, 80.0, 72.0])
        pgrp["pixel_c"] = np.array([64.0, 64.0, 90.0, 50.0])
        pgrp["intensity"] = np.array([120.0, 85.0, 115.0, 130.0])
        pgrp["sigma"] = np.array([10.0, 9.0, 10.0, 11.0])
        pgrp["wavelength"] = np.array([2.8, 1.4, 2.8, 2.8])
        pgrp["var_u"] = np.array([1.5, 1.5, 1.5, 1.5])
        pgrp["var_v"] = np.array([1.5, 1.5, 1.5, 1.5])
        pgrp["cov_uv"] = np.array([0.0, 0.0, 0.0, 0.0])

    sg = gemmi.find_spacegroup_by_name("Ia-3d")
    asu = gemmi.ReciprocalAsu(sg)
    ops = sg.operations()
    h_asu1, _ = asu.to_asu([1, 2, 3], ops)
    h_asu2, _ = asu.to_asu([2, 4, 6], ops)

    mtz_m = gemmi.Mtz(with_base=True)
    mtz_m.spacegroup = sg
    mtz_m.set_cell_for_all(gemmi.UnitCell(11.93, 11.93, 11.93, 90, 90, 90))
    mtz_m.add_dataset("crystal")
    mtz_m.add_column("I", "J")
    mtz_m.add_column("SigI", "Q")
    mtz_m.set_data(
        np.array(
            [
                [h_asu1[0], h_asu1[1], h_asu1[2], 122.0, 6.0],
                [h_asu2[0], h_asu2[1], h_asu2[2], 84.0, 5.0],
            ],
            dtype=np.float32,
        )
    )
    mtz_m.write_to_file(str(tmp_path / "merged.mtz"))

    mtz_p = gemmi.Mtz(with_base=True)
    mtz_p.spacegroup = sg
    mtz_p.set_cell_for_all(gemmi.UnitCell(11.93, 11.93, 11.93, 90, 90, 90))
    mtz_p.add_dataset("crystal")
    for col, ctype in [
        ("Iobs", "J"),
        ("SigIobs", "Q"),
        ("Ipred", "J"),
        ("SigIpred", "Q"),
        ("Scale", "R"),
    ]:
        mtz_p.add_column(col, ctype)
    mtz_p.set_data(
        np.array(
            [
                [h_asu1[0], h_asu1[1], h_asu1[2], 120.0, 10.0, 122.0, 6.0, 1.02],
                [h_asu2[0], h_asu2[1], h_asu2[2], 85.0, 9.0, 84.0, 5.0, 0.98],
                [h_asu1[0], h_asu1[1], h_asu1[2], 115.0, 10.0, 122.0, 6.0, 1.01],
                [h_asu1[0], h_asu1[1], h_asu1[2], 130.0, 11.0, 122.0, 6.0, 1.06],
            ],
            dtype=np.float32,
        )
    )
    mtz_p.write_to_file(str(tmp_path / "predictions.mtz"))

    (tmp_path / "careless_metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "n_merged_asu": 2,
                "n_observations": 4,
                "mean_isigi": 18.6,
                "mean_multiplicity": 2.0,
                "completeness_pct": 42.5,
                "cc_half": 0.962,
                "cc_half_spearman": 0.951,
                "resolution_bins": [
                    {
                        "bin": i,
                        "d_max": 5.0 - i * 0.3,
                        "d_min": 4.7 - i * 0.3,
                        "n_refl": 10,
                        "cc_half": 0.98 - i * 0.02,
                    }
                    for i in range(10)
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = NovaGalaxyPipelineRunner.__new__(NovaGalaxyPipelineRunner)
    job_rec = {"stdout": ""}
    runner._stamp_careless_merge_into_h5(str(h5_path), str(tmp_path), job_rec)

    payload = build_rainbow_zone_payload(replay_h5=h5_path, frame_index=0)
    assert payload["artifact_kind"] == "merged_careless"
    assert payload["pending_stage"] == "done"
    assert payload["merge_metrics"]["cc_half_overall"] == 0.962
    assert payload["merge_metrics"]["completeness_pct"] == 42.5
    assert len(payload["peaks"]) == 3
    assert len(payload["sample_peaks"]) == 4
    # Symmetry mates (1,2,3), (-1,2,3), (1,-2,3) in Ia-3d share the same canonical ASU orbit
    assert (
        payload["sample_peaks"][0]["asu_orbit_id"]
        == payload["sample_peaks"][2]["asu_orbit_id"]
        == payload["sample_peaks"][3]["asu_orbit_id"]
    )
    # Harmonic pair (1,2,3) and (2,4,6) on the same ray have harmonic_multiplicity == 2
    assert payload["sample_peaks"][0]["harmonic_multiplicity"] == 2
    assert payload["sample_peaks"][1]["harmonic_multiplicity"] == 2
    assert "careless_residual" in payload["sample_peaks"][0]
    assert "careless_scale" in payload["sample_peaks"][0]
    assert "q_hat_lab" in payload["sample_peaks"][0]
    assert "goniometer_R" in payload and len(payload["goniometer_R"]) == 3

    html = render_rainbow_html(payload)
    assert "btn-wulff-scope" in html
    assert "btn-wulff-frame" not in html
    assert "btn-wulff-couple" not in html
    assert "sel-wulff-color" in html
    assert "wulff-merge-bar" in html
    assert "btn-files-toggle" in html
    assert "files-drawer" in html
    assert "--top-bar-scale" in html
    assert "--bottom-bar-scale" in html
    assert "repeat(5, minmax(0, 1fr))" in html
    assert "fitHorizontalBars" in html
    assert "getDetectorDataAspect" in html
    assert "getDetectorPlotBox" in html
    assert "getWarpedPanelCanvas" in html
    assert 'id="detector-pane"' in html
    assert "isPeakHighlightedTarget" in html
    assert "Compl. vs d" in html
    assert "⟨I/σ⟩ vs d" in html
    assert len(payload["merge_metrics"]["cc_half_bins"]) == 10
    assert all(
        "completeness_pct" in b and "mean_isigi" in b
        for b in payload["merge_metrics"]["cc_half_bins"]
    )
    assert payload["pipeline_stages"][3]["name"] == "4. MERGE (CARELESS)"
    assert "imaginex_subhkl_merge" in payload["pipeline_stages"][3]["tool_ids"]

    runner.active_job = None
    step_arts = runner.list_step_artifacts(current_h5=str(h5_path), history_datasets=[])
    merge_names = {a["name"] for a in step_arts["merge"]}
    assert "merged.mtz" in merge_names
    assert "predictions.mtz" in merge_names
    assert "careless_metrics.json" in merge_names
    local_fp, dl_name = runner.resolve_artifact_for_download(
        name="merged.mtz", current_h5=str(h5_path)
    )
    assert local_fp == str(tmp_path / "merged.mtz")
    assert dl_name == "merged.mtz"


def test_nova_mvvm_pydantic_and_view_model(tmp_path):
    """Verify NOVA tutorial MVVM architecture (`SubhklConfig`, `MainModel`, `MainViewModel`, `create_viewmodels`)."""
    import pytest

    pytest.importorskip("pydantic")
    pytest.importorskip("nova.mvvm")
    from pydantic import ValidationError

    from subhkl.viz.nova_app.models.subhkl_model import InstrumentOptions, SubhklConfig

    # 1. Verify Pydantic validation & defaults
    cfg = SubhklConfig(instrument=InstrumentOptions.MANDI, a=70.0)
    assert cfg.instrument == InstrumentOptions.MANDI
    assert cfg.a == 70.0
    assert cfg.wavelength_max > cfg.wavelength_min

    with pytest.raises(ValidationError):
        SubhklConfig(wavelength_min=5.0, wavelength_max=2.0)

    # 2. Verify MainApp(server=None) initializes ThemedApp, TrameBinding, ViewModel & UI cleanly
    from subhkl.viz.logo_svg import SUBHKL_FAVICON_DATA_URI, SUBHKL_FAVICON_SVG
    from subhkl.viz.nova_app.views.main_view import MainApp

    app = MainApp(instrument="CG4D")
    assert app.server is not None
    assert app.server.state.trame__favicon == SUBHKL_FAVICON_DATA_URI
    assert 'class="favicon-laue-zones"' in SUBHKL_FAVICON_SVG
    assert "<rect" not in SUBHKL_FAVICON_SVG
    assert app.view_model.view_state.running is False
    assert app.view_model.model.subhkl.data.instrument == InstrumentOptions.CG4D
    html = app.server.state.parallax_html
    assert "DIFFRACTION PARALLAX" in html
    assert '<img class="subhkl-logo-img" src="data:image/png;base64,' in html
    assert (
        f'<link rel="icon" type="image/svg+xml" href="{SUBHKL_FAVICON_DATA_URI}" />'
        in html
    )
    assert '<span class="logo-sub">sUB</span><span class="logo-hkl">hkl</span>' in html
    assert "executeWorkflowStage" in html
    assert 'id="terminal-box"' in html
    assert "1.85 * state.detZoom" in html
    assert 'id="val-gonio"' in html
    assert 'id="sel-frame"' in html
    payload = app.view_model.model.subhkl.payload or {}
    if payload.get("frames"):
        assert "ω=" in payload.get("goniometer_summary", "")
        p_f2 = app.view_model.model.subhkl.refresh_payload(frame_index=2) or {}
        assert p_f2.get("frame_index") == 2

    # 3. Verify embedded JavaScript has zero syntax errors
    import re
    import shutil
    import subprocess

    if shutil.which("node"):
        scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
        assert scripts
        js_file = tmp_path / "embedded_check.js"
        js_file.write_text(scripts[-1], encoding="utf-8")
        proc = subprocess.run(
            ["node", "--check", str(js_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"JS syntax error: {proc.stderr}"


def test_predictor_and_integrator_ellipses_and_galaxy_dataset_loading(tmp_path):
    """Verify predictor.h5 and integrated.h5 (isotropic circles & anisotropic ellipses) are recognized and rendered."""
    import h5py
    import pytest

    from subhkl.viz.rainbow_zones import render_rainbow_html

    # 1. Create a synthetic predictor.h5 with banks/<img_idx>/(i, j, h, k, l, wavelength)
    pred_h5 = tmp_path / "predictor.h5"
    with h5py.File(pred_h5, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f.create_dataset("bank_ids", data=np.array([21], dtype=int))
        f.create_dataset("sample/U", data=np.eye(3))
        f.create_dataset("sample/B", data=np.diag([1 / 60.0, 1 / 70.0, 1 / 80.0]))
        f.create_dataset("sample/space_group", data=b"P 21 21 21")
        g = f.create_group("banks/0")
        g.create_dataset("i", data=np.array([120.0, 180.0, 240.0]))
        g.create_dataset("j", data=np.array([150.0, 210.0, 260.0]))
        g.create_dataset("h", data=np.array([1, 2, 1], dtype=int))
        g.create_dataset("k", data=np.array([2, 1, 3], dtype=int))
        g.create_dataset("l", data=np.array([3, 4, 2], dtype=int))
        g.create_dataset("wavelength", data=np.array([2.9, 3.2, 3.6]))

    p_pred = build_rainbow_zone_payload(replay_h5=pred_h5)
    assert p_pred["artifact_kind"] == "predictor"
    assert p_pred["peak_shape_mode"] == "predicted_circle"
    assert len(p_pred["peaks"]) == 3

    # 2. Create a synthetic anisotropic integrated.h5 with peaks/(var_u, var_v, cov_uv)
    int_aniso_h5 = tmp_path / "integrated_aniso.h5"
    with h5py.File(int_aniso_h5, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f.create_dataset("sample/U", data=np.eye(3))
        f.create_dataset("sample/B", data=np.diag([1 / 60.0, 1 / 70.0, 1 / 80.0]))
        f.create_dataset("sample/space_group", data=b"P 21 21 21")
        pk = f.create_group("peaks")
        pk.create_dataset("bank", data=np.array([21, 21], dtype=int))
        pk.create_dataset("pixel_r", data=np.array([140.0, 220.0]))
        pk.create_dataset("pixel_c", data=np.array([160.0, 240.0]))
        pk.create_dataset("h", data=np.array([1, 2], dtype=int))
        pk.create_dataset("k", data=np.array([2, 3], dtype=int))
        pk.create_dataset("l", data=np.array([3, 1], dtype=int))
        pk.create_dataset("lambda", data=np.array([3.0, 3.4]))
        pk.create_dataset("intensity", data=np.array([500.0, 800.0]))
        pk.create_dataset("sigma", data=np.array([25.0, 30.0]))
        pk.create_dataset("var_u", data=np.array([12.0, 10.0]))
        pk.create_dataset("var_v", data=np.array([36.0, 42.0]))
        pk.create_dataset("cov_uv", data=np.array([5.5, -6.2]))

    p_aniso = build_rainbow_zone_payload(replay_h5=int_aniso_h5)
    assert p_aniso["artifact_kind"] == "integrator"
    assert p_aniso["peak_shape_mode"] == "anisotropic_ellipse"
    assert len(p_aniso["peaks"]) == 2
    pk0 = p_aniso["peaks"][0]
    assert pk0["shape_kind"] == "ellipse"
    assert len(pk0["ellipse"]) == 16
    assert len(pk0["ellipse_sph"]) == 16
    assert pk0["sigma_major_px"] > pk0["sigma_minor_px"]

    # 3. Create a synthetic isotropic integrated.h5 with var_u == var_v and cov_uv == 0
    int_iso_h5 = tmp_path / "integrated_iso.h5"
    with h5py.File(int_iso_h5, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f.create_dataset("sample/U", data=np.eye(3))
        f.create_dataset("sample/B", data=np.diag([1 / 60.0, 1 / 70.0, 1 / 80.0]))
        pk = f.create_group("peaks")
        pk.create_dataset("bank", data=np.array([21], dtype=int))
        pk.create_dataset("pixel_r", data=np.array([200.0]))
        pk.create_dataset("pixel_c", data=np.array([200.0]))
        pk.create_dataset("h", data=np.array([1], dtype=int))
        pk.create_dataset("k", data=np.array([1], dtype=int))
        pk.create_dataset("l", data=np.array([1], dtype=int))
        pk.create_dataset("lambda", data=np.array([3.1]))
        pk.create_dataset("intensity", data=np.array([400.0]))
        pk.create_dataset("sigma", data=np.array([20.0]))
        pk.create_dataset("var_u", data=np.array([16.0]))
        pk.create_dataset("var_v", data=np.array([16.0]))
        pk.create_dataset("cov_uv", data=np.array([0.0]))

    p_iso = build_rainbow_zone_payload(replay_h5=int_iso_h5)
    assert p_iso["artifact_kind"] == "integrator"
    assert p_iso["peak_shape_mode"] == "isotropic_circle"
    assert p_iso["peaks"][0]["shape_kind"] == "isotropic_circle"
    assert p_iso["peaks"][0]["sigma_major_px"] == pytest.approx(4.0)
    assert p_iso["peaks"][0]["sigma_minor_px"] == pytest.approx(4.0)

    html = render_rainbow_html(p_aniso)
    assert "api/payload?dataset_id=" in html
    assert "INTEGRATED • 3D TENSOR ELLIPSES" in html
    assert "__action_neutron_data__" in html
    assert "__action_local_file__" in html
    assert 'id="modal-neutron-data"' in html


def test_merged_h5_pending_solve_and_cg4d_title_goniometer_harvesting(tmp_path):
    """Verify merged.h5 sets pending_stage='solve' and CG4D /entry/title goniometer angles are harvested."""
    import h5py
    import pytest

    from subhkl.instrument.goniometer import (
        get_rotation_data_from_nexus,
        harvest_goniometer_angles_from_title,
    )

    # 1. Verify regex harvesting for all CG4D /entry/title formats
    assert harvest_goniometer_angles_from_title(
        "Beta-Lac crystal, Omega = 110 degrees"
    ) == {"omega": 110.0}
    assert harvest_goniometer_angles_from_title("beta lac axis2 = 92 degrees") == {
        "omega": 92.0
    }
    assert harvest_goniometer_angles_from_title(
        "Garnet axis0 = 82 degrees, axis1 = -15.5 degrees"
    ) == {"phi": 82.0, "kappa": -15.5}

    # 2. Verify get_rotation_data_from_nexus harvests /entry/title when CG4D:Mot:... is absent from DASlogs
    raw_nxs = tmp_path / "CG4D_2625.nxs.h5"
    with h5py.File(raw_nxs, "w") as f:
        f.create_dataset(
            "entry/title", data=np.array([b"Beta-Lac crystal, Omega = 110 degrees"])
        )
        f.create_group("entry/DASlogs")

    axes, angles, names = get_rotation_data_from_nexus(str(raw_nxs), "CG4D")
    assert names == [
        "CG4D:Mot:omega",
        "CG4D:Mot:alpha",
        "CG4D:Mot:kappa",
        "CG4D:Mot:alpha",
        "CG4D:Mot:phi",
    ]
    assert angles == pytest.approx([110.0, 45.0, 0.0, 45.0, 0.0])
    assert len(axes) == 5

    # 3. Verify merged.h5 (detector images + unit cell + space group + goniometer, no U) -> artifact_kind="merged", pending_stage="solve"
    merged_h5 = tmp_path / "merged.h5"
    with h5py.File(merged_h5, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f.create_dataset("sample/a", data=68.7)
        f.create_dataset("sample/b", data=69.8)
        f.create_dataset("sample/c", data=156.9)
        f.create_dataset("sample/alpha", data=90.0)
        f.create_dataset("sample/beta", data=90.0)
        f.create_dataset("sample/gamma", data=90.0)
        f.create_dataset("sample/space_group", data=b"P 21 21 21")
        f.create_dataset("instrument/wavelength", data=np.array([2.8, 4.5]))
        f.create_dataset("images", data=np.zeros((2, 1, 32, 32), dtype=np.float32))
        f.create_dataset(
            "goniometer/angles",
            data=np.array(
                [
                    [110.0, 45.0, 0.0, 45.0, 0.0],
                    [130.0, 45.0, 0.0, 45.0, 0.0],
                ]
            ),
        )
        f.create_dataset(
            "goniometer/names",
            data=np.array(
                [
                    b"CG4D:Mot:omega",
                    b"CG4D:Mot:alpha",
                    b"CG4D:Mot:kappa",
                    b"CG4D:Mot:alpha",
                    b"CG4D:Mot:phi",
                ]
            ),
        )

    p_merged = build_rainbow_zone_payload(replay_h5=merged_h5)
    assert p_merged["artifact_kind"] == "merged"
    assert p_merged["pending_stage"] == "solve"
    assert len(p_merged["pipeline_stages"]) == 4
    assert [s["stage_id"] for s in p_merged["pipeline_stages"]] == [
        "reduce",
        "solve",
        "integrate",
        "merge",
    ]
    assert p_merged["pipeline_stages"][0]["status"] == "completed"
    assert p_merged["pipeline_stages"][1]["status"] == "pending"
    assert len(p_merged["frames"]) == 2
    assert "ω=110" in p_merged["goniometer_summary"]

    # 4. Verify check_neutron_paths never touches the host filesystem directly and routes strictly through NDIP
    from unittest.mock import MagicMock

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner

    runner_no_ndip = NovaGalaxyPipelineRunner(galaxy_url="")
    chk_unavail = runner_no_ndip.check_neutron_paths(
        {
            "facility": "HFIR",
            "instrument": "CG4D",
            "ipts": "37331",
            "paths": ["/HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5"],
        }
    )
    assert chk_unavail["ok"] is False
    assert chk_unavail["checked_via"] == "ndip_unavailable"

    # Mock NDIP Galaxy connection to verify routing through neutrons_load_data
    runner_ndip = NovaGalaxyPipelineRunner(
        galaxy_url="https://ndip-test.ornl.gov",
        galaxy_api_key="mock_key",
    )
    mock_gi = MagicMock()
    mock_gi.tools.run_tool.return_value = {
        "jobs": [{"id": "job_ndip_1"}],
        "outputs": [{"id": "ds_ndip_1"}],
    }
    mock_store = MagicMock()
    mock_store.history_id = "hist_1"
    mock_store.nova_connection.galaxy_instance = mock_gi
    runner_ndip.get_datastore = MagicMock(return_value=mock_store)
    runner_ndip._has_tool_on_galaxy = MagicMock(return_value=True)
    runner_ndip.browse_oncat_catalog = MagicMock(
        return_value={"ok": False, "runs": [], "auth_status": {"authenticated": False}}
    )
    runner_ndip._api_request = MagicMock(
        return_value={
            "state": "ok",
            "exit_code": 0,
            "tool_stdout": "",
            "tool_stderr": "",
        }
    )
    chk_ok = runner_ndip.check_neutron_paths(
        {
            "ingest_mode": "oncat",
            "facility": "SNS",
            "instrument": "MANDI",
            "ipts": "34720",
            "runs": "12737",
            "paths": ["/SNS/MANDI/IPTS-34720/nexus/MANDI_12737.nxs.h5"],
        }
    )
    assert chk_ok["ok"] is True
    assert chk_ok["checked_via"] == "ndip_neutrons_load_data"
    assert chk_ok["verified_dataset_ids"] == ["ds_ndip_1"]
    mock_gi.tools.run_tool.assert_called_once()
    call_kwargs = mock_gi.tools.run_tool.call_args.kwargs
    assert call_kwargs["tool_id"] == "neutrons_load_data"

    # Verify ONCat-first fast path skips neutrons_load_data file fetch completely
    mock_gi.tools.run_tool.reset_mock()
    runner_ndip.browse_oncat_catalog = MagicMock(
        return_value={
            "ok": True,
            "runs": [
                {
                    "run": 2625,
                    "location": "/HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5",
                    "title": "Beta-Lac Omega=110",
                    "angles": "ω=110°",
                }
            ],
            "auth_status": {"authenticated": True},
        }
    )
    chk_oncat_fast = runner_ndip.check_neutron_paths(
        {
            "ingest_mode": "file",
            "facility": "SNS",
            "instrument": "MANDI",
            "ipts": "34720",
            "paths": ["/HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5"],
        }
    )
    assert chk_oncat_fast["ok"] is True
    assert chk_oncat_fast["verified_dataset_ids"] == ["oncat:HFIR:CG4D:IPTS-37331:1"]
    mock_gi.tools.run_tool.assert_not_called()

    # Mock NDIP failure when wrong IPTS is given
    runner_ndip.browse_oncat_catalog = MagicMock(
        return_value={"ok": False, "runs": [], "auth_status": {"authenticated": False}}
    )
    runner_ndip._api_request = MagicMock(
        return_value={
            "state": "error",
            "exit_code": 1,
            "tool_stdout": "",
            "tool_stderr": "FileNotFoundError: /SNS/MANDI/IPTS-37331/nexus/MANDI_12737.nxs.h5: No such file or directory",
        }
    )
    chk_missing = runner_ndip.check_neutron_paths(
        {
            "ingest_mode": "oncat",
            "facility": "SNS",
            "instrument": "MANDI",
            "ipts": "37331",
            "runs": "12737",
            "paths": ["/SNS/MANDI/IPTS-37331/nexus/MANDI_12737.nxs.h5"],
        }
    )
    assert chk_missing["ok"] is False
    assert chk_missing["checked_via"] == "ndip_neutrons_load_data"

    # 4b. Verify MANDI spherical panel geometry has 4-corner diamond coordinates and 48-point outline
    mandi_merged = tmp_path / "MANDI_12737_merged.h5"
    with h5py.File(mandi_merged, "w") as f:
        f.attrs["instrument"] = "MANDI"
        f.create_dataset("images", data=np.ones((2, 256, 256), dtype=np.float32) * 20.0)
        f.create_dataset("bank_ids", data=np.array([11, 32], dtype=np.int32))
        f.create_dataset("file_offsets", data=np.array([0], dtype=np.int32))
        f.create_dataset("files", data=np.array([b"MANDI_12737.nxs.h5"]))
    p_mandi = build_rainbow_zone_payload(replay_h5=mandi_merged)
    assert p_mandi["instrument"] == "MANDI"
    assert p_mandi["default_projection"] == "spherical"
    assert len(p_mandi["pipeline_stages"]) == 4
    b11 = next(p for p in p_mandi["panels"] if p["bank"] == 11)
    assert len(b11["corners_sph"]) == 4
    assert len(b11["outline_sph"]) == 48
    # Bank 11 on MANDI is a 45-deg diamond: C0=(0,0) and C2=(255,255) share nearly the same azimuth, while C1 and C3 span azimuth
    c0_az, c0_el = b11["corners_sph"][0]
    c1_az, c1_el = b11["corners_sph"][1]
    c2_az, c2_el = b11["corners_sph"][2]
    c3_az, c3_el = b11["corners_sph"][3]
    assert abs(c0_az - c2_az) < 2.0
    assert abs(c1_el - c3_el) < 2.0
    assert abs(c1_az - c3_az) > 15.0
    assert abs(c2_el - c0_el) > 15.0

    # 5. Verify _work_dir_for_reduce isolates reduction outputs from existing directories
    red_dir, red_label = runner_no_ndip._work_dir_for_reduce(
        ["/SNS/MANDI/IPTS-34720/nexus/MANDI_12737.nxs.h5"],
        "/home/xvg/subhkl-examples/betalac-7",
    )
    assert red_dir != "/home/xvg/subhkl-examples/betalac-7"
    assert "MANDI_12737" in red_dir
    assert red_label == "MANDI_12737_merged.h5"

    from subhkl.viz.rainbow_zones import render_rainbow_html

    html_merged = render_rainbow_html(p_merged)
    assert "Number(sub[rowOff + dc])" in html_merged
    assert "api/pipeline/resolve_local_files" in html_merged
    assert "api/pipeline/oncat_catalog" in html_merged
    assert "api/pipeline/oncat_auth" in html_merged
    assert "nd-oncat-browser" in html_merged
    assert "nd-oncat-auth-bar" in html_merged
    assert "formatRunDurationHMS" in html_merged
    assert "+ Neutron data" not in html_merged
    assert "+ From local computer" not in html_merged
    assert ">Neutron data (SNS/HFIR)...</option>" in html_merged
    assert ">From local computer...</option>" in html_merged

    # Verify ONCat browser uses pyoncat (no direct local /SNS or /HFIR header reads)
    import os

    assert not hasattr(runner_no_ndip, "_read_nexus_header_meta")
    assert not hasattr(runner_no_ndip, "_load_inventory_csv")

    tok_file = tmp_path / "test_oncat_token.json"
    monkeypatch_prev = os.environ.get("ONCAT_TOKEN_PATH")
    os.environ["ONCAT_TOKEN_PATH"] = str(tok_file)
    try:
        auth_init = runner_no_ndip.get_oncat_auth_status()
        assert auth_init["ok"] is True
        assert auth_init["authenticated"] is False

        real_build = runner_no_ndip._build_oauth_oncat_client

        def _build_with_mock_user(*args, **kwargs):
            try:
                client = real_build(*args, **kwargs)
            except ImportError:
                client = MagicMock()
                client.has_stored_token.side_effect = lambda: os.path.isfile(
                    runner_no_ndip._get_oncat_token_path()
                )
            client.User = MagicMock()
            client.User.retrieve.return_value = {
                "name": "Test Scientist",
                "id": "xvg",
            }
            return client

        runner_no_ndip._build_oauth_oncat_client = _build_with_mock_user
        auth_seeded = runner_no_ndip.handle_oncat_auth(
            {"action": "token", "access_token": "mock_oidc_jwt_token"}
        )
        assert auth_seeded["ok"] is True
        assert auth_seeded["authenticated"] is True
        assert auth_seeded["user"] == "Test Scientist"
        assert tok_file.is_file()

        auth_out = runner_no_ndip.handle_oncat_auth({"action": "logout"})
        assert auth_out["ok"] is True
        assert auth_out["authenticated"] is False
        assert not tok_file.exists()

        cat_res = runner_no_ndip.browse_oncat_catalog(
            {"facility": "HFIR", "instrument": "CG4D", "ipts": "37331"}
        )
        assert cat_res["ok"] is True
        assert cat_res["multi_run_supported"] is True
        assert isinstance(cat_res["experiments"], list)
        assert isinstance(cat_res["runs"], list)

        runner_no_ndip._oncat_exp_cache = {
            ("HFIR", "CG4D", False): (
                9e12,
                [{"ipts": "37331", "id": "IPTS-37331", "n_runs": 10}],
            )
        }
        cat_empty = runner_no_ndip.browse_oncat_catalog(
            {"facility": "HFIR", "instrument": "CG4D", "ipts": ""}
        )
        assert cat_empty["ipts"] == ""
        assert cat_empty["runs"] == []
    finally:
        runner_no_ndip._build_oauth_oncat_client = real_build
        if monkeypatch_prev is None:
            os.environ.pop("ONCAT_TOKEN_PATH", None)
        else:
            os.environ["ONCAT_TOKEN_PATH"] = monkeypatch_prev


def test_cg4d_load_neutron_data_fallback_and_mandi_curvilinear_panels(tmp_path):
    """Verify CG4D neutrons_load_data file-mode routing, active-solve frame overlay, and MANDI curvilinear panel mesh."""
    import numpy as np

    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    from subhkl.instrument.goniometer import get_rotation_data_from_title
    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import _encode_panel_textures, export_rainbow_widget

    runner = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")

    # 1. HFIR CG4D routes via params|ingest_mode="file" since upstream neutrons_load_data omits CG4D from oncat select
    cg4d_in = runner._build_neutrons_load_data_inputs(
        ingest_mode="oncat",
        fac="HFIR",
        inst="CG4D",
        ipts_raw="36148",
        run_num=192,
        filepath="/HFIR/CG4D/IPTS-36148/nexus/CG4D_192.nxs.h5",
    )
    assert cg4d_in == {
        "params|ingest_mode": "file",
        "params|filepath": "/HFIR/CG4D/IPTS-36148/nexus/CG4D_192.nxs.h5",
    }
    mandi_in = runner._build_neutrons_load_data_inputs(
        ingest_mode="oncat",
        fac="SNS",
        inst="MANDI",
        ipts_raw="34270",
        run_num=12737,
        filepath="/SNS/MANDI/IPTS-34270/nexus/MANDI_12737.nxs.h5",
    )
    assert mandi_in["params|ingest_mode"] == "oncat"
    assert mandi_in["params|instrument"] == "MANDI"

    _axes, angles, names = get_rotation_data_from_title("CaTiO3 phi=-59.0", "CG4D")
    assert len(angles) == 5
    assert angles[-1] == -59.0
    assert "CG4D:Mot:phi" in names

    # 2. Active solve overlay preserves intermediate zones/ladder across frame switches
    runner.active_job = {
        "stage_id": "solve",
        "state": "running",
        "ladder_rung_idx": 2,
        "partial_overlay": {
            "seq": 1,
            "zones": [{"zone_id": "z_1_0_0", "uvw": [1, 0, 0]}],
            "ladder_steps": [{"step": 1, "name": "First Solve"}],
        },
    }
    frame_payload = {"zones": [], "ladder_steps": [], "frame_index": 3}
    runner.apply_active_solve_overlay(frame_payload, frame_index=3)
    assert runner.current_frame_index == 3
    assert len(frame_payload["zones"]) == 1
    assert frame_payload["active_rung_idx"] == 2

    # 3. MANDI bottom-plane panels have 9x9 curvilinear mesh and Bank 5 unwraps across +-180 deg seam
    dets = {int(bk): Detector(cfg) for bk, cfg in beamlines["MANDI"].items()}
    empty = np.array([])
    panels, _ = _encode_panel_textures(dets, empty, empty, empty, empty, empty)
    assert len(panels) == len(dets)
    by_bank = {p["bank"]: p for p in panels}
    for bk in (1, 2, 3, 5, 7, 8):
        assert bk in by_bank
        p = by_bank[bk]
        assert len(p["mesh_sph"]) == 9 and len(p["mesh_sph"][0]) == 9
        el_mesh_min = min(pt[1] for row in p["mesh_sph"] for pt in row)
        el_out_min = min(pt[1] for pt in p["outline_sph"])
        assert abs(el_mesh_min - el_out_min) < 1e-2

    out_html = tmp_path / "mandi_curv.html"
    export_rainbow_widget(
        {"zones": [], "peaks": [], "panels": []},
        output_html=out_html,
    )
    html = out_html.read_text()
    assert "panel.mesh_sph" in html
    assert "isFrameSwitch" in html

    # 4. ImageStackMerger writes LZF-compressed 3D image stacks and multi-run check_neutron_paths returns verified_dataset_ids
    from unittest.mock import MagicMock

    import h5py

    from subhkl.commands import run_merge_images

    r1 = tmp_path / "CG4D_2625.reduce.h5"
    r2 = tmp_path / "CG4D_2626.reduce.h5"
    for rp, ang in ((r1, 10.0), (r2, 20.0)):
        with h5py.File(rp, "w") as hf:
            hf.attrs["instrument"] = "CG4D"
            hf.create_dataset(
                "images",
                data=np.zeros((2, 16, 16), dtype=np.int64),
                chunks=(1, 16, 16),
                compression="lzf",
            )
            hf.create_dataset("bank_ids", data=np.array([1, 2], dtype=np.int32))
            hf.create_dataset(
                "goniometer/angles", data=np.array([[ang, 45.0, 0.0], [ang, 45.0, 0.0]])
            )
            hf.create_dataset("goniometer/axes", data=np.eye(3))
            hf.create_dataset("instrument/wavelength", data=np.array([2.8, 4.5]))
    merged_lzf = tmp_path / "merged_lzf.h5"
    run_merge_images(
        f"{r1} {r2}",
        str(merged_lzf),
        54.0,
        54.0,
        54.0,
        90.0,
        90.0,
        90.0,
        "P 1",
    )
    with h5py.File(merged_lzf, "r") as hf:
        assert hf["images"].compression == "lzf"
        assert hf["images"].chunks == (1, 16, 16)
        assert hf["images"].shape == (4, 16, 16)
    p_lzf = build_rainbow_zone_payload(replay_h5=merged_lzf)
    assert [fr["file"] for fr in p_lzf["frames"]] == ["CG4D_2625", "CG4D_2626"]

    runner_multi = NovaGalaxyPipelineRunner(
        galaxy_url="https://ndip-test.ornl.gov",
        galaxy_api_key="mock_key",
    )
    mock_gi = MagicMock()
    mock_gi.tools.run_tool.return_value = {
        "jobs": [{"id": "job_ndip_m"}],
        "outputs": [{"id": "ds_ndip_m"}],
    }
    mock_store = MagicMock()
    mock_store.history_id = "hist_m"
    mock_store.nova_connection.galaxy_instance = mock_gi
    runner_multi.get_datastore = MagicMock(return_value=mock_store)
    runner_multi._has_tool_on_galaxy = MagicMock(return_value=True)
    runner_multi.browse_oncat_catalog = MagicMock(return_value={"runs": []})
    runner_multi._api_request = MagicMock(return_value={"state": "ok", "exit_code": 0})
    chk_multi = runner_multi.check_neutron_paths(
        {
            "ingest_mode": "oncat",
            "facility": "HFIR",
            "instrument": "CG4D",
            "ipts": "36148",
            "paths": [
                "/HFIR/CG4D/IPTS-36148/nexus/CG4D_2625.nxs.h5",
                "/HFIR/CG4D/IPTS-36148/nexus/CG4D_2626.nxs.h5",
            ],
        }
    )
    assert chk_multi["ok"] is True
    assert chk_multi["verified_dataset_ids"] == ["ds_ndip_m"]

    # 5. Terminal carriage-return (\r) and ANSI CSI emulation for in-place tqdm progress bars
    raw_tqdm = (
        "Starting integration...\n"
        "\rBatching Images:   0%|          | 0/600 [00:00<?, ?it/s]"
        "\r                                                        \r"
        "\rBatching Images:  50%|█████     | 300/600 [00:01<00:01, 250it/s]"
        "\rBatching Images: 100%|██████████| 600/600 [00:02<00:00, 280it/s]\n"
        "Done."
    )
    formatted_tqdm = NovaGalaxyPipelineRunner._format_terminal_stream(raw_tqdm)
    assert formatted_tqdm.splitlines() == [
        "Starting integration...",
        "Batching Images: 100%|██████████| 600/600 [00:02<00:00, 280it/s]",
        "Done.",
    ]
    # Also verify collapsing when Galaxy console_output converts \r to \n (universal newlines)
    formatted_tqdm_univ = NovaGalaxyPipelineRunner._format_terminal_stream(
        raw_tqdm.replace("\r", "\n")
    )
    assert formatted_tqdm_univ.splitlines() == [
        "Starting integration...",
        "Batching Images: 100%|██████████| 600/600 [00:02<00:00, 280it/s]",
        "Done.",
    ]
    assert "formatTerminalStdio" in html
    assert "hasMergedCarelessData()" in html
    assert "opt.disabled = !hasRes" in html
    assert "opt.disabled = !hasScl" in html

    # 6. Verify checkpoint(..., snapshot=...) writes a live Step 4 .stages.h5 loadable with include_panel_images=False
    from types import SimpleNamespace

    from subhkl.scan_refine import checkpoint

    stage_out = tmp_path / "live_solve.h5"
    args_ns = SimpleNamespace(
        output=str(stage_out),
        instrument="CG4D",
        assign_wavelength=(2.8, 4.5),
        wl_min=2.8,
        wl_max=4.5,
        space_group="I a -3 d",
        n_zones=120,
        d_min=1.2,
    )
    cg4d_banks = sorted(int(k) for k in beamlines["CG4D"])
    dets_cg4d = [Detector(dict(beamlines["CG4D"][str(b)])) for b in cg4d_banks]
    snap = {
        "U": np.eye(3),
        "B": np.eye(3) * 0.0838,
        "cell": [11.93, 11.93, 11.93, 90.0, 90.0, 90.0],
        "banks": cg4d_banks,
        "centers": np.asarray(
            [d.center + np.array([0.005, 0.0, 0.0]) for d in dets_cg4d]
        ),
        "uhats": np.asarray([d.uhat for d in dets_cg4d]),
        "vhats": np.asarray([d.vhat for d in dets_cg4d]),
        "widths": np.asarray([d.width for d in dets_cg4d]),
        "heights": np.asarray([d.height for d in dets_cg4d]),
        "R": np.eye(3)[None, :, :],
        "n_zones": 120,
    }
    checkpoint(
        args_ns,
        "120-zones",
        np.zeros(10),
        [
            {
                "stage": "120-zones",
                "objective": -0.42,
                "iterations": 12,
                "success": True,
                "message": "OK",
            }
        ],
        snapshot=snap,
    )
    stages_h5 = tmp_path / "live_solve.stages.h5"
    assert stages_h5.is_file()
    p_live = build_rainbow_zone_payload(
        replay_h5=stages_h5,
        max_zone_index=2,
        max_zones=30,
        include_panels=True,
        include_panel_images=False,
        include_peaks=True,
    )
    assert len(p_live["zones"]) > 0
    assert len(p_live["peaks"]) > 0
    assert len(p_live["panels"]) > 0
    assert all(p.get("texture_b64") is None for p in p_live["panels"])
    assert p_live["ladder_steps"][3]["stage"] == "4. 120→400 Zone + Cell"

    # 7. Verify frame selection persistence during live solve overlay & per-run R_all indexing
    runner.active_job = {
        "stage_id": "solve",
        "state": "running",
        "ladder_rung_idx": 3,
        "partial_h5_path": str(stages_h5),
        "images_h5_path": str(merged_lzf),
        "solve_params": {
            "instrument": "CG4D",
            "space_group": "I a -3 d",
            "wavelength_range": [2.8, 4.5],
        },
        "partial_overlay": {
            "seq": 2,
            "frame_index": 0,
            "zones": [{"zone_id": "z_old", "uvw": [1, 0, 0]}],
            "ladder_steps": [],
        },
    }
    runner.sync_frame_index(1)
    assert runner.current_frame_index == 1
    assert runner.active_job["partial_overlay"]["frame_index"] == 1
    assert runner.active_job["partial_overlay"]["seq"] == 3
    assert "state.selectedFrameIndex = clamped" in html
    assert "api/pipeline/status?frame=${curFrame}" in html

    import os

    old_ep = os.environ.get("EP_PATH")
    try:
        os.environ["EP_PATH"] = "/interactivetool/ep/test/subhkl"
        out_ep_html = tmp_path / "ep_test.html"
        export_rainbow_widget(p_live, output_html=out_ep_html)
        html_ep = out_ep_html.read_text()
        assert '<base href="/interactivetool/ep/test/subhkl/" />' in html_ep

        import asyncio

        from aiohttp.test_utils import make_mocked_request

        from subhkl.viz.nova_app.views.main_view import MainApp

        app_inst = object.__new__(MainApp)
        app_inst.pipeline_runner = MagicMock()
        app_inst.pipeline_runner.browse_oncat_catalog.return_value = {
            "ok": True,
            "ipts": "34720",
            "experiments": [],
            "runs": [{"run": 12737}],
        }
        from aiohttp import web

        aio_app = web.Application()
        app_inst._bind_routes(SimpleNamespace(app=aio_app))
        req = make_mocked_request(
            "GET",
            "/interactivetool/ep/test/subhkl/api/pipeline/oncat_catalog?facility=SNS&instrument=MANDI&ipts=34720",
            app=aio_app,
        )
        resp = asyncio.run(aio_app._handle(req))
        assert resp.status == 200
        app_inst.pipeline_runner.browse_oncat_catalog.assert_called_once_with(
            {"facility": "SNS", "instrument": "MANDI", "ipts": "34720"}
        )
    finally:
        if old_ep is None:
            os.environ.pop("EP_PATH", None)
        else:
            os.environ["EP_PATH"] = old_ep

    # 8. Verify newest-first dataset lookup in _ensure_galaxy_h5_dataset, live-state stdio materialization,
    #    tape-recorder Play/Stop job cancellation, and persistent terminal scroll position
    assert "stage-tape-btn" in html
    assert "cancelWorkflowStage" in html
    assert "api/pipeline/cancel" in html
    assert "savedScrollTop" in html
    assert "box.scrollTop = box.scrollHeight" not in html

    # Verify _ensure_galaxy_h5_dataset picks newer 'ok' dataset over older stuck 'running' dataset
    mock_store_ds = MagicMock()
    mock_store_ds.history_id = "hist_3969"
    mock_gi_ds = MagicMock()
    mock_gi_ds.datasets.get_datasets.return_value = [
        {"id": "50f189cd8eed6da2", "name": "merged_lzf.h5", "state": "running"},
        {"id": "99aabbccddeeff00", "name": "merged_lzf.h5", "state": "ok"},
    ]
    mock_store_ds.nova_connection.galaxy_instance = mock_gi_ds
    ds_job_rec: dict = {"stdout": ""}
    linked_id = runner._ensure_galaxy_h5_dataset(
        mock_store_ds, str(merged_lzf), ds_job_rec
    )
    assert linked_id == "99aabbccddeeff00"
    mock_gi_ds.datasets.wait_for_dataset.assert_not_called()
    assert (
        "Linked existing dataset 'merged_lzf.h5' (id=99aabbccddeeff00, state=ok)"
        in ds_job_rec["stdout"]
    )

    # Verify cancel_active_job deletes active Galaxy jobs and marks job as deleted
    del_calls: list[tuple[str, str]] = []
    runner._api_request = MagicMock(
        side_effect=lambda method, path, **kw: del_calls.append((method, path)) or {}
    )
    runner.active_job = {
        "job_id": "gx_job_42",
        "galaxy_job_ids": ["gx_job_41", "gx_job_42"],
        "stage_id": "solve",
        "mode": "galaxy",
        "state": "running",
        "stdout": "",
        "stderr": "",
    }
    cancel_res = runner.cancel_active_job("solve")
    assert cancel_res["ok"] is True
    assert cancel_res["cancelled_job_id"] == "gx_job_42"
    assert runner.active_job["state"] == "deleted"
    assert runner.active_job["cancel_requested"] is True
    assert ("DELETE", "/api/jobs/gx_job_41") in del_calls
    assert ("DELETE", "/api/jobs/gx_job_42") in del_calls

    # Verify [subhkl-live-state] stdout emission, terminal stripping, and HDF5 materialization
    import base64
    import json
    import zlib

    live_doc = {
        "kind": "stages",
        "stage": "120-zones",
        "instrument": "CG4D",
        "space_group": "I a -3 d",
        "wavelength": [2.8, 4.5],
        "d_min": 1.2,
        "U": np.eye(3).tolist(),
        "B": (np.eye(3) * 0.0838).tolist(),
        "cell": [11.93, 11.93, 11.93, 90.0, 90.0, 90.0],
        "R_all": np.eye(3)[None, :, :].tolist(),
    }
    b64_tok = base64.b64encode(
        zlib.compress(json.dumps(live_doc).encode("utf-8"), level=6)
    ).decode("ascii")
    raw_with_live = f"Step 1...\n[subhkl-live-state] {b64_tok}\nStep 2..."
    stripped = NovaGalaxyPipelineRunner._format_terminal_stream(raw_with_live)
    assert "[subhkl-live-state]" not in stripped
    assert stripped.splitlines() == ["Step 1...", "Step 2..."]

    mat_h5 = tmp_path / "materialized.blind.h5"
    assert NovaGalaxyPipelineRunner._materialize_live_state_h5(b64_tok, str(mat_h5))
    assert mat_h5.is_file()
    p_mat = build_rainbow_zone_payload(
        replay_h5=mat_h5,
        max_zone_index=2,
        max_zones=20,
        include_panels=False,
        include_peaks=True,
    )
    assert len(p_mat["zones"]) > 0
    from subhkl import (
        format_version_badge,
        get_container_image,
        get_container_tag,
        get_version,
    )

    assert p_mat["subhkl_version"] == get_version()
    assert f'id="app-version">{format_version_badge(get_version())}</span>' in html
    assert format_version_badge("1.5.0") == "v1.5.0"
    assert format_version_badge("v1.5.0") == "v1.5.0"
    assert (
        format_version_badge("feat/ui-logo-and-favicon") == "feat/ui-logo-and-favicon"
    )
    assert get_container_tag("feat/ui-logo-and-favicon") == "feat-ui-logo-and-favicon"
    assert get_container_tag("v1.5.0") == "1.5.0"
    assert p_mat["pipeline_stages"][0]["container"] == get_container_image()
    assert p_mat["pipeline_stages"][1]["container"] == get_container_image()
    assert p_mat["pipeline_stages"][2]["container"] == get_container_image()


def test_reopen_history_merged_h5_preserves_dataset_id_and_discovers_images(
    tmp_path, monkeypatch
) -> None:
    import os
    import tempfile as _tempfile
    from types import SimpleNamespace

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import _discover_images_h5

    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(tmp_path))

    mandi_gx_id = "70f27733988683c0"
    cg4d_gx_id = "1805ca028e2eaa6f"
    mandi_cached = (
        tmp_path / f"subhkl_gx_{mandi_gx_id}_MANDI_12737_MANDI_12743_merged.h5"
    )
    with h5py.File(mandi_cached, "w") as hf:
        dt_s = h5py.string_dtype(encoding="utf-8")
        hf.create_dataset(
            "files",
            data=np.array(["MANDI_12737.nxs.h5", "MANDI_12743.nxs.h5"], dtype=object),
            dtype=dt_s,
        )
        hf.create_dataset("images", data=np.zeros((2, 8, 8), dtype=np.float32))
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([2.0, 4.0], dtype=np.float32)
        )
    mandi_size = mandi_cached.stat().st_size

    runner = NovaGalaxyPipelineRunner.__new__(NovaGalaxyPipelineRunner)
    runner.galaxy_url = "https://ndip-test.ornl.gov"
    runner.api_key = "fake_key"
    runner._dataset_id_by_path = {}
    runner._api_request = lambda method, path, **kw: {
        "id": mandi_gx_id,
        "name": "MANDI_12737_MANDI_12743_merged.h5",
        "state": "ok",
        "file_size": mandi_size,
    }

    resolved_local, _ = runner.resolve_dataset_to_local_h5(mandi_gx_id)
    _work_dir, work_merged = runner._resolve_work_dir_and_merged(resolved_local)
    assert os.path.basename(work_merged) == "MANDI_12737_MANDI_12743_merged.h5"
    assert runner._dataset_id_by_path.get(os.path.abspath(work_merged)) == mandi_gx_id

    # Stamp instrument metadata onto merged.h5 that already has an 'instrument' group
    runner._stamp_merged_metadata(
        work_merged, {"instrument": "MANDI", "params": {"space_group": "P 21 21 21"}}
    )
    with h5py.File(work_merged, "r") as hf:
        assert hf.attrs.get("instrument") == "MANDI"
        assert hf["instrument/name"][()].decode("utf-8") == "MANDI"

    # Simulate a Galaxy history containing both an old CG4D 'merged.h5' and the MANDI dataset
    datasets_by_id = {
        cg4d_gx_id: {
            "id": cg4d_gx_id,
            "name": "merged.h5",
            "state": "ok",
            "deleted": False,
            "file_size": mandi_size + 99999,
        },
        mandi_gx_id: {
            "id": mandi_gx_id,
            "name": "MANDI_12737_MANDI_12743_merged.h5",
            "state": "ok",
            "deleted": False,
            "file_size": mandi_size,
        },
    }

    class _FakeDatasetsClient:
        def show_dataset(self, ds_id: str):
            return datasets_by_id[ds_id]

        def get_datasets(self, history_id: str, name: str | None = None):
            all_ds = list(datasets_by_id.values())
            if name is None:
                return all_ds
            return [d for d in all_ds if name in d["name"]]

    fake_store = SimpleNamespace(
        history_id="hist_1",
        nova_connection=SimpleNamespace(
            galaxy_instance=SimpleNamespace(
                datasets=_FakeDatasetsClient(),
                base_url="https://ndip-test.ornl.gov",
            )
        ),
    )

    job_rec = {"stdout": ""}
    linked_id = runner._ensure_galaxy_h5_dataset(fake_store, work_merged, job_rec)
    assert linked_id == mandi_gx_id

    # Verify _ensure_galaxy_h5_dataset for solve.h5 does not link 'betalac-7/solve.h5'
    # or conflate <tag>_merged.h5 candidate paths with solve.h5
    work_solve = os.path.join(_work_dir, "solve.h5")
    with h5py.File(work_solve, "w") as hf:
        hf.create_dataset("hkl", data=np.zeros((4, 3), dtype=np.float32))
    solve_size = os.path.getsize(work_solve)
    datasets_by_id["fc00dcb17c606557"] = {
        "id": "fc00dcb17c606557",
        "name": "betalac-7/solve.h5",
        "state": "ok",
        "deleted": False,
        "file_size": solve_size,
    }
    datasets_by_id["c0662c36e1a24cf3"] = {
        "id": "c0662c36e1a24cf3",
        "name": "solve.h5 (subhkl_solve)",
        "state": "ok",
        "deleted": False,
        "file_size": solve_size,
    }
    linked_solve_id = runner._ensure_galaxy_h5_dataset(fake_store, work_solve, job_rec)
    assert linked_solve_id == "c0662c36e1a24cf3"

    # Verify SubhklGalaxyModel() starts with no preloaded local dataset and pending_stage='reduce'
    from subhkl.viz.nova_app.models.subhkl_model import SubhklGalaxyModel

    empty_model = SubhklGalaxyModel()
    assert empty_model.data.peaks_file == ""
    assert empty_model.data.images_file == ""
    assert empty_model.payload.get("artifact_kind") == "empty"
    assert empty_model.payload.get("pending_stage") == "reduce"
    assert len(empty_model.payload.get("panels") or []) > 0

    # Verify list_history_datasets orders most recently touched first and touch_dataset promotes older items
    from subhkl.viz.rainbow_zones import (
        render_rainbow_html,
        validate_subhkl_ingestible_dataset,
    )

    runner.galaxy_api_key = "fake_key"
    runner.history_id = "hist_1"
    runner._dataset_touch_times = {}
    raw_history = [
        {
            "id": "ds_old",
            "hid": 10,
            "name": "old_merged.h5",
            "state": "ok",
            "extension": "h5",
            "history_content_type": "dataset",
            "deleted": False,
            "update_time": "2026-01-10T10:00:00.000000",
        },
        {
            "id": "ds_new",
            "hid": 20,
            "name": "new_solve.h5",
            "state": "ok",
            "extension": "h5",
            "history_content_type": "dataset",
            "deleted": False,
            "update_time": "2026-01-12T10:00:00.000000",
        },
    ]
    runner._api_request = lambda method, path, **kw: raw_history
    ordered = runner.list_history_datasets()
    assert [d["id"] for d in ordered] == ["ds_new", "ds_old"]

    runner.touch_dataset("ds_old")
    ordered_after_touch = runner.list_history_datasets()
    assert [d["id"] for d in ordered_after_touch] == ["ds_old", "ds_new"]

    # Verify validate_subhkl_ingestible_dataset accepts SubHKL .h5 / raw NeXus and errors on unsupported datasets
    assert (
        validate_subhkl_ingestible_dataset(
            work_solve,
            {
                "name": "integrator.h5",
                "hid": 279,
                "state": "ok",
                "extension": "png",
            },
        )
        == "subhkl_h5"
    )
    raw_nxs = tmp_path / "CG4D_2625.nxs.h5"
    with h5py.File(raw_nxs, "w") as hf:
        hf.create_group("entry")
    assert (
        validate_subhkl_ingestible_dataset(
            raw_nxs,
            {
                "name": "CG4D_2625.nxs.h5",
                "hid": 280,
                "state": "ok",
                "extension": "h5",
            },
        )
        == "raw_nexus"
    )
    from subhkl.viz.rainbow_zones import (
        _extract_from_h5,
        infer_dataset_extension,
    )

    assert infer_dataset_extension("integrator.h5", "png") == "h5"
    assert infer_dataset_extension("reflections.mtz", "png") == "mtz"
    assert infer_dataset_extension("dataset_abc-123.dat", "h5") == "dat"

    bad_mtz = tmp_path / "reflections.mtz"
    bad_mtz.write_bytes(b"MTZ \x00\x00\x00\x00")
    try:
        validate_subhkl_ingestible_dataset(
            bad_mtz,
            {
                "name": "reflections.mtz",
                "hid": 278,
                "state": "ok",
                "extension": "mtz",
            },
            has_active_geometry=False,
        )
        raise AssertionError("Expected ValueError for standalone MTZ file")
    except ValueError as exc:
        assert "Cannot load MTZ dataset" in str(exc)

    # Verify _extract_from_h5 ignores stale sibling careless_metrics.json when peaks/careless_residual is absent
    # and reconstructs peaks_xyz from pixel_r/pixel_c without triggering finder-less solve forward projection
    stale_dir = tmp_path / "stale_integ"
    stale_dir.mkdir(exist_ok=True)
    (stale_dir / "careless_metrics.json").write_text(
        json.dumps({"n_merged_asu": 500, "cc_half_overall": 0.95}),
        encoding="utf-8",
    )
    integ_no_xyz = stale_dir / "integrated.h5"
    with h5py.File(integ_no_xyz, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf["sample/U"] = np.eye(3)
        hf["sample/B"] = np.diag([1.0 / 68.7, 1.0 / 69.8, 1.0 / 156.9])
        hf["peaks/h"] = np.array([1, 2, 3], dtype=int)
        hf["peaks/k"] = np.array([0, 1, 1], dtype=int)
        hf["peaks/l"] = np.array([1, 0, 2], dtype=int)
        hf["peaks/lambda"] = np.array([3.1, 3.4, 3.8], dtype=float)
        hf["peaks/intensity"] = np.array([120.0, 95.0, 80.0], dtype=float)
        hf["peaks/sigma"] = np.array([5.0, 4.0, 4.0], dtype=float)
        hf["peaks/bank"] = np.array([20, 20, 20], dtype=int)
        hf["peaks/pixel_r"] = np.array([120.0, 140.0, 160.0], dtype=float)
        hf["peaks/pixel_c"] = np.array([130.0, 150.0, 170.0], dtype=float)
        hf["peaks/two_theta"] = np.array([0.8, 0.9, 1.0], dtype=float)
        hf["peaks/azimuthal"] = np.array([0.2, 0.3, 0.4], dtype=float)
        hf["peaks/var_u"] = np.array([2.5, 2.5, 2.5], dtype=float)
        hf["peaks/var_v"] = np.array([1.8, 1.8, 1.8], dtype=float)
        hf["peaks/cov_uv"] = np.array([0.1, 0.1, 0.1], dtype=float)

    ext_integ = _extract_from_h5(integ_no_xyz)
    assert ext_integ["artifact_kind"] == "integrator"
    assert ext_integ["merge_metrics"] is None
    assert len(ext_integ["peaks_xyz"]) == 3

    # Verify invalidate_subsequent_stages removes stale careless_metrics.json and hides subsequent stages
    runner.invalidate_subsequent_stages("solve", str(stale_dir))
    assert not (stale_dir / "careless_metrics.json").exists()
    step_arts = runner.list_step_artifacts(
        current_h5=str(integ_no_xyz),
        history_datasets=[
            {"id": "ds_s", "hid": 10, "name": "solve.h5", "state": "ok"},
            {"id": "ds_i", "hid": 11, "name": "integrated.h5", "state": "ok"},
            {"id": "ds_m", "hid": 12, "name": "merged.mtz", "state": "ok"},
        ],
    )
    assert len(step_arts["solve"]) == 1
    assert step_arts["integrate"] == []
    assert step_arts["merge"] == []

    bad_h5 = tmp_path / "unrelated.h5"
    with h5py.File(bad_h5, "w") as hf:
        hf.create_dataset("foo", data=np.array([1, 2, 3]))
    try:
        validate_subhkl_ingestible_dataset(
            bad_h5,
            {"name": "unrelated.h5", "hid": 281, "state": "ok", "extension": "h5"},
        )
        raise AssertionError("Expected ValueError for unrelated HDF5 file")
    except ValueError as exc:
        assert "does not contain SubHKL diffraction data" in str(exc)

    html_doc = render_rainbow_html(empty_model.payload)
    assert "__action_galaxy_dataset__" in html_doc
    assert "From Galaxy..." in html_doc
    assert 'id="modal-galaxy-dataset"' in html_doc
    assert "MAX_RECENT_HISTORY_ITEMS" in html_doc

    # Verify _discover_images_h5 locates <tag>_merged.h5 from a solve.h5 that has 'files'
    # even when attrs['source'] points to an existing Pulsar staging dataset_*.dat file
    staging_dir = tmp_path / "staging_test" / "279969" / "inputs"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_dat = staging_dir / "dataset_c0662c36-e1a2-4cf3-8d04-233deadc86c9.dat"
    with h5py.File(staging_dat, "w") as hf:
        hf.create_dataset("images", data=np.zeros((2, 8, 8), dtype=np.float32))
    solve_h5 = tmp_path / "subhkl_gx_999999_solve.h5"
    with h5py.File(solve_h5, "w") as hf:
        hf.attrs["source"] = str(staging_dat)
        dt_s = h5py.string_dtype(encoding="utf-8")
        hf.create_dataset(
            "files",
            data=np.array(["MANDI_12737.nxs.h5", "MANDI_12743.nxs.h5"], dtype=object),
            dtype=dt_s,
        )
    disc = _discover_images_h5(solve_h5)
    assert disc is not None
    assert os.path.abspath(disc) == os.path.abspath(work_merged)
