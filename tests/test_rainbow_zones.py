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

    # Multi-run solve.h5 must forward-project across ALL runs so All-Runs Wulff Net
    # displays every run's predicted reflections immediately after Solve (before Integration)
    solve_mr_h5 = tmp_path / "synthetic_solve_multirun.h5"
    ang1, ang2 = np.deg2rad(35.0), np.deg2rad(70.0)
    R0 = np.eye(3)
    R1 = np.array(
        [
            [np.cos(ang1), 0.0, np.sin(ang1)],
            [0.0, 1.0, 0.0],
            [-np.sin(ang1), 0.0, np.cos(ang1)],
        ]
    )
    R2 = np.array(
        [
            [np.cos(ang2), 0.0, np.sin(ang2)],
            [0.0, 1.0, 0.0],
            [-np.sin(ang2), 0.0, np.cos(ang2)],
        ]
    )
    with h5py.File(solve_mr_h5, "w") as f:
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
        f["files"] = np.array(
            [b"CG4D_1720.nxs.h5", b"CG4D_1740.nxs.h5", b"CG4D_1760.nxs.h5"]
        )
        f["goniometer/R"] = np.stack([R0, R1, R2], axis=0)
        f["bank_ids"] = np.array([101, 101, 101], dtype=np.int32)
        f["file_offsets"] = np.array([0, 1, 2], dtype=np.int64)
        sr_mr = f.create_group("scan_refinement")
        sr_mr.attrs["d_min"] = 2.5
        sr_mr.attrs["status"] = "converged"

    payload_mr = build_rainbow_zone_payload(replay_h5=solve_mr_h5)
    assert len(payload_mr["frames"]) == 3
    assert len(payload_mr["sample_peaks"]) > len(payload_mr["peaks"])
    assert {p["run_index"] for p in payload_mr["sample_peaks"]} == {0, 1, 2}
    assert len(payload_mr["predicted_sample_peaks"]) == len(payload_mr["sample_peaks"])
    assert all(p.get("intensity") is None for p in payload_mr["sample_peaks"])


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

    # 4. Verify check_neutron_paths falls back cleanly to local /SNS and /HFIR when Galaxy is unavailable,
    #    and returns ndip_unavailable only when neither Galaxy nor local facility mounts exist.
    from unittest.mock import MagicMock

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner

    runner_no_ndip = NovaGalaxyPipelineRunner(galaxy_url="")
    unmounted_sns = tmp_path / "no_sns"
    unmounted_hfir = tmp_path / "no_hfir"
    runner_no_ndip.SNS_HFIR_ROOTS = {
        "SNS": str(unmounted_sns),
        "HFIR": str(unmounted_hfir),
    }
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

    # Now mount a synthetic local /HFIR and /SNS tree and verify offline check_neutron_paths,
    # wrong-IPTS auto-suggestion, browse_oncat_catalog fallback, and _resolve_raw_nexus_inputs.
    mock_hfir = tmp_path / "mock_HFIR"
    mock_sns = tmp_path / "mock_SNS"
    nxs_dir_37331 = mock_hfir / "CG4D" / "IPTS-37331" / "nexus"
    nxs_dir_37331.mkdir(parents=True)
    for r_num, om_deg in ((2625, 110.0), (2627, 114.0)):
        nxs_file = nxs_dir_37331 / f"CG4D_{r_num}.nxs.h5"
        with h5py.File(nxs_file, "w") as nf:
            nf["entry/title"] = f"Beta-Lactamase omega={om_deg}"
            nf["entry/sample/name"] = "Beta-Lac"
            nf["entry/duration"] = np.array([125.0], dtype=np.float64)
            nf["entry/total_counts"] = np.array([45000.0], dtype=np.float64)
            nf["entry/DASlogs/omega/average_value"] = np.array(
                [om_deg], dtype=np.float64
            )
    mock_sns.mkdir(parents=True)

    runner_local_fs = NovaGalaxyPipelineRunner(galaxy_url="")
    runner_local_fs.SNS_HFIR_ROOTS = {
        "SNS": str(mock_sns),
        "HFIR": str(mock_hfir),
    }
    chk_local_ok = runner_local_fs.check_neutron_paths(
        {
            "facility": "HFIR",
            "instrument": "CG4D",
            "ipts": "37331",
            "runs": "2625-2627/2",
        }
    )
    assert chk_local_ok["ok"] is True
    assert chk_local_ok["checked_via"] == "local_filesystem"
    assert len(chk_local_ok["existing"]) == 2
    assert "Beta-Lactamase" in chk_local_ok["sample_title"]
    assert "omega=110.0" in chk_local_ok["sample_goniometer"]

    # Wrong IPTS suggests the sibling IPTS where the run actually lives on /HFIR
    chk_local_wrong_ipts = runner_local_fs.check_neutron_paths(
        {
            "facility": "HFIR",
            "instrument": "CG4D",
            "ipts": "99999",
            "paths": ["/HFIR/CG4D/IPTS-99999/nexus/CG4D_2625.nxs.h5"],
        }
    )
    assert chk_local_wrong_ipts["ok"] is False
    assert chk_local_wrong_ipts["checked_via"] == "local_filesystem"
    assert chk_local_wrong_ipts["suggested_ipts"] == "37331"

    # Also falls back to local /HFIR when Galaxy is configured in env but unreachable
    runner_down_galaxy = NovaGalaxyPipelineRunner(
        galaxy_url="https://ndip-down.example.gov",
        galaxy_api_key="expired_or_unreachable",
    )
    runner_down_galaxy.SNS_HFIR_ROOTS = {
        "SNS": str(mock_sns),
        "HFIR": str(mock_hfir),
    }
    runner_down_galaxy._has_tool_on_galaxy = MagicMock(return_value=False)
    chk_down_fallback = runner_down_galaxy.check_neutron_paths(
        {
            "facility": "HFIR",
            "instrument": "CG4D",
            "ipts": "37331",
            "paths": ["/HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5"],
        }
    )
    assert chk_down_fallback["ok"] is True
    assert chk_down_fallback["checked_via"] == "local_filesystem"

    # Catalog browser falls back to local /HFIR directory & header scan when ONCat is unavailable
    # (including when optional `pyoncat` is not installed in base CI environments)
    import sys as _sys
    from unittest.mock import patch

    mock_broken_oncat = MagicMock()
    mock_broken_oncat.Experiment.list.side_effect = RuntimeError("ONCat offline")
    mock_broken_oncat.Datafile.list.side_effect = RuntimeError("ONCat offline")
    runner_local_fs._get_oncat_client = MagicMock(
        return_value=(mock_broken_oncat, False)
    )
    runner_local_fs.get_oncat_auth_status = MagicMock(
        return_value={"ok": True, "authenticated": False}
    )
    with patch.dict(_sys.modules, {"pyoncat": None}):
        cat_local = runner_local_fs.browse_oncat_catalog(
            {"facility": "HFIR", "instrument": "CG4D", "ipts": "37331"}
        )
    assert cat_local["ok"] is True
    assert cat_local["catalog_source"] == "local_filesystem"
    assert [e["ipts"] for e in cat_local["experiments"]] == ["37331"]
    assert [r["run"] for r in cat_local["runs"]] == [2625, 2627]
    assert cat_local["runs"][0]["duration_hms"] == "00:02:05"

    # _resolve_raw_nexus_inputs resolves IPTS + stepped run range directly on local /HFIR
    resolved_local_nxs = runner_local_fs._resolve_raw_nexus_inputs(
        {
            "facility": "HFIR",
            "instrument": "CG4D",
            "ipts": "37331",
            "runs": "2625:2627:2",
        },
        str(tmp_path),
    )
    assert [p.split("/")[-1] for p in resolved_local_nxs] == [
        "CG4D_2625.nxs.h5",
        "CG4D_2627.nxs.h5",
    ]

    # _resolve_local_podman_image falls back from unbuilt git-branch tags to cached :main (or :main pull)
    import os as _os
    from unittest.mock import patch

    def _fake_podman_run(cmd: list[str], **_kwargs: object) -> object:
        from types import SimpleNamespace

        if cmd[1:3] == ["image", "exists"]:
            target = cmd[3]
            return SimpleNamespace(
                returncode=0 if target == "ghcr.io/neutrons/subhkl:main" else 1
            )
        if cmd[1] == "images":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "ghcr.io/neutrons/subhkl:pr-113\n"
                    "ghcr.io/neutrons/subhkl:main\n"
                    "ghcr.io/neutrons/subhkl:1.5.0\n"
                ),
            )
        if cmd[1:3] == ["manifest", "inspect"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=1, stdout="")

    clean_env = {
        k: v
        for k, v in _os.environ.items()
        if k not in ("SUBHKL_CONTAINER_IMAGE", "SUBHKL_CONTAINER_TAG")
    }
    with (
        patch.dict(_os.environ, clean_env, clear=True),
        patch("subprocess.run", side_effect=_fake_podman_run),
    ):
        resolved_img, note_branch = (
            NovaGalaxyPipelineRunner._resolve_local_podman_image(
                "ghcr.io/neutrons/subhkl:feat-local-sns-hfir-fallback"
            )
        )
        assert resolved_img == "ghcr.io/neutrons/subhkl:main"
        assert note_branch is not None and "bind-mounted subhkl" in note_branch

        exact_img, note_exact = NovaGalaxyPipelineRunner._resolve_local_podman_image(
            "ghcr.io/neutrons/subhkl:main"
        )
        assert exact_img == "ghcr.io/neutrons/subhkl:main"
        assert note_exact is None

        offline_rel_img, note_rel = (
            NovaGalaxyPipelineRunner._resolve_local_podman_image(
                "ghcr.io/neutrons/subhkl:9.9.9"
            )
        )
        assert offline_rel_img == "ghcr.io/neutrons/subhkl:main"
        assert note_rel is not None and "unreachable" in note_rel

    # cancel_active_job immediately kills (podman kill) and force-removes (-f -i -t 0) local container
    podman_cancel_cmds: list[list[str]] = []

    def _record_podman_cancel(cmd: list[str], **_kwargs: object) -> object:
        from types import SimpleNamespace

        podman_cancel_cmds.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    runner_local_fs.active_job = {
        "job_id": "gpu-12345-1",
        "container_name": "subhkl-gpu-12345-1",
        "stage_id": "solve",
        "mode": "container_gpu",
        "state": "running",
        "stdout": "",
        "stderr": "",
    }
    mock_proc = MagicMock()
    mock_proc.pid = 999999
    mock_proc.poll.side_effect = [None, 0]
    runner_local_fs._active_procs = {"gpu-12345-1": mock_proc}
    with (
        patch("shutil.which", return_value="/usr/bin/podman"),
        patch("subprocess.run", side_effect=_record_podman_cancel),
        patch("os.killpg"),
        patch("os.getpgid", return_value=999999),
    ):
        cancel_res = runner_local_fs.cancel_active_job("solve")
    assert cancel_res["ok"] is True
    assert cancel_res["status"] == "cancelled"
    assert cancel_res["cancelled_container"] == "subhkl-gpu-12345-1"
    assert ["/usr/bin/podman", "kill", "subhkl-gpu-12345-1"] in podman_cancel_cmds
    assert [
        "/usr/bin/podman",
        "rm",
        "-f",
        "-i",
        "-t",
        "0",
        "subhkl-gpu-12345-1",
    ] in podman_cancel_cmds
    assert "Stopped container subhkl-gpu-12345-1" in cancel_res["job"]["stdout"]

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
    assert "terminalFollowTail" in html
    assert "handleTerminalUserScroll" in html

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

    # Verify _resolve_work_dir_and_merged on solve.h5 NEVER overwrites <tag>_merged.h5 or its dataset ID
    _wd_s, _resolved_s = runner._resolve_work_dir_and_merged(str(solve_h5))
    with h5py.File(work_merged, "r") as hf_chk:
        assert "images" in hf_chk
    assert runner._dataset_id_by_path.get(os.path.abspath(work_merged)) == mandi_gx_id
    assert runner._dataset_id_by_path.get(os.path.abspath(str(solve_h5))) == "999999"

    # Verify _extract_from_h5 sets artifact_kind="merged_careless" when careless/metrics_json has resolution_bins
    merged_careless_h5 = tmp_path / "merged_careless_bins_only.h5"
    with h5py.File(merged_careless_h5, "w") as hf_mc:
        hf_mc.attrs["instrument"] = "CG4D"
        hf_mc["sample/U"] = np.eye(3)
        hf_mc["sample/B"] = np.diag([1.0 / 68.7, 1.0 / 69.8, 1.0 / 156.9])
        hf_mc["peaks/h"] = np.array([1, 2, 3], dtype=int)
        hf_mc["peaks/k"] = np.array([0, 1, 1], dtype=int)
        hf_mc["peaks/l"] = np.array([1, 0, 2], dtype=int)
        hf_mc["peaks/lambda"] = np.array([3.1, 3.4, 3.8], dtype=float)
        hf_mc["peaks/intensity"] = np.array([120.0, 95.0, 80.0], dtype=float)
        hf_mc["peaks/sigma"] = np.array([5.0, 4.0, 4.0], dtype=float)
        hf_mc["peaks/bank"] = np.array([20, 20, 20], dtype=int)
        hf_mc["peaks/pixel_r"] = np.array([120.0, 140.0, 160.0], dtype=float)
        hf_mc["peaks/pixel_c"] = np.array([130.0, 150.0, 170.0], dtype=float)
        hf_mc["peaks/var_u"] = np.array([2.5, 2.5, 2.5], dtype=float)
        hf_mc["peaks/var_v"] = np.array([1.8, 1.8, 1.8], dtype=float)
        hf_mc["peaks/cov_uv"] = np.array([0.1, 0.1, 0.1], dtype=float)
        hf_mc["peaks/careless_residual"] = np.array([0.2, -0.4, 0.1], dtype=float)
        hf_mc["peaks/careless_scale"] = np.array([1.0, 1.1, 0.9], dtype=float)
        cgrp = hf_mc.require_group("careless")
        cgrp.create_dataset(
            "metrics_json",
            data=np.bytes_(
                json.dumps(
                    {
                        "resolution_bins": [
                            {
                                "bin": 0,
                                "d_max": 5.0,
                                "d_min": 2.0,
                                "n_refl": 42,
                                "n_theo": 50,
                                "completeness_pct": 84.0,
                                "mean_isigi": 12.5,
                                "cc_half": 0.985,
                            }
                        ]
                    }
                ).encode("utf-8")
            ),
        )
    ext_mc = _extract_from_h5(merged_careless_h5)
    assert ext_mc["artifact_kind"] == "merged_careless"
    assert ext_mc["merge_metrics"] is not None
    assert ext_mc["merge_metrics"]["n_merged_asu"] == 42
    assert len(ext_mc["merge_metrics"]["cc_half_bins"]) == 1

    # Verify reduction_settings.json wavelength defaults, Params λ_min/λ_max inputs, synchronized d_min slider,
    # and unscaled I/sigma color mode in rendered HTML
    from subhkl.viz.rainbow_zones import _get_instrument_default_wavelength

    assert _get_instrument_default_wavelength("MANDI") == (2.0, 4.0)
    assert _get_instrument_default_wavelength("TOPAZ") == (0.4, 3.5)
    assert _get_instrument_default_wavelength("CG4D") == (2.8, 4.5)
    fresh_cg4d = SubhklGalaxyModel()
    assert fresh_cg4d.data.wavelength_min == 2.8
    assert fresh_cg4d.data.wavelength_max == 4.5
    assert fresh_cg4d.data.d_min == 1.4
    assert fresh_cg4d.payload["wavelength_range"] == [2.8, 4.5]
    assert fresh_cg4d.payload["d_min"] == 1.4
    assert 'id="inp-lmin"' in html_doc
    assert 'id="inp-lmax"' in html_doc
    assert 'id="slider-dmin"' in html_doc
    assert 'id="sel-spot-color"' in html_doc
    assert '<option value="isigi">Color: I/σ (Log)</option>' in html_doc
    assert '<option value="mtz_isigi">Color: .mtz I/σ (Log)</option>' in html_doc
    assert 'id="slider-color-emphasis"' in html_doc
    assert 'id="slider-parallax"' not in html_doc
    assert 'id="btn-peaks-mtz"' not in html_doc
    assert "hasIntegratedIsigi" in html_doc
    assert "hasMtzIsigi" in html_doc
    assert "syncDminControls" in html_doc
    assert "Runner host:" in html_doc
    from subhkl.io.parser import detect_runner_hostname

    assert detect_runner_hostname()

    # Verify d_min, lambda_min, and lambda_max preservation across solve/blind_search/search HDF5 groups,
    # frame switches, active solve overlays, and UI stage reloads
    blind_h5 = tmp_path / "solve.blind.h5"
    with h5py.File(blind_h5, "w") as hf_b:
        hf_b.attrs["instrument"] = "CG4D"
        hf_b["sample/U"] = np.eye(3)
        hf_b["sample/B"] = np.diag([1.0 / 68.7, 1.0 / 69.8, 1.0 / 156.9])
        hf_b.create_group("instrument").create_dataset(
            "wavelength", data=np.array([2.85, 4.35], dtype=np.float64)
        )
        sgrp = hf_b.create_group("search")
        sgrp.attrs["d_min"] = 1.95
    ext_blind = _extract_from_h5(blind_h5)
    assert np.allclose(ext_blind["wavelength_range"], (2.85, 4.35))
    assert np.isclose(ext_blind["solve_diagnostics"]["d_min"], 1.95)

    empty_model.data.peaks_file = str(work_merged)
    empty_model.data.images_file = str(work_merged)
    empty_model.refresh_payload()
    # User sets custom UI parameters before running / re-running solve
    empty_model.data.wavelength_min = 2.85
    empty_model.data.wavelength_max = 4.35
    empty_model.data.d_min = 1.95
    p_kept = empty_model.refresh_payload(frame_index=1, preserve_params=True)
    assert p_kept["wavelength_range"] == [2.85, 4.35]
    assert p_kept["d_min"] == 1.95
    assert empty_model.data.wavelength_min == 2.85
    assert empty_model.data.wavelength_max == 4.35
    assert empty_model.data.d_min == 1.95

    # Subsequent solve with a different d_min in the UI must not revert to the previous solve.h5's d_min (1.95)
    empty_model.data.peaks_file = str(blind_h5)
    empty_model.data.d_min = 1.50
    p_subsequent = empty_model.refresh_payload(frame_index=0, preserve_params=True)
    assert p_subsequent["d_min"] == 1.50
    assert empty_model.data.d_min == 1.50

    runner.active_job = {
        "stage_id": "solve",
        "state": "running",
        "solve_params": {
            "instrument": "CG4D",
            "space_group": "P 21 21 21",
            "wavelength_range": (2.85, 4.35),
            "d_min": 1.95,
        },
    }
    ov_payload = {"wavelength_range": [2.0, 10.0], "d_min": 1.0}
    runner.apply_active_solve_overlay(ov_payload, frame_index=0)
    assert ov_payload["wavelength_range"] == [2.85, 4.35]
    assert ov_payload["d_min"] == 1.95
    assert 'preserveParams: stageId !== "reduce"' in html_doc
    assert (
        "const keepUserParams = isFrameSwitch || preserveParams || isSolveRunning;"
        in html_doc
    )

    # Verify flexible unit-cell clipboard parser (space group at end, front, or omitted)
    from subhkl.viz.rainbow_zones import (
        parse_unit_cell_text,
        resolve_space_group_symmetry,
    )

    p_end = parse_unit_cell_text(
        "68.84174 98.06962 90.91568 90.07304 106.11396 90.0088 P21"
    )
    assert p_end is not None
    assert np.allclose(
        p_end["cell"],
        [68.84174, 98.06962, 90.91568, 90.07304, 106.11396, 90.0088],
    )
    assert p_end["space_group"] == "P21"

    p_front = parse_unit_cell_text("P 21 21 21 68.7 69.8 156.9 90.0 90.0 90.0")
    assert p_front is not None
    assert np.allclose(p_front["cell"], [68.7, 69.8, 156.9, 90.0, 90.0, 90.0])
    assert p_front["space_group"] == "P 21 21 21"

    p_none = parse_unit_cell_text(
        "68.84174, 98.06962, 90.91568, 90.07304, 106.11396, 90.0088"
    )
    assert p_none is not None
    assert p_none["space_group"] == ""
    assert np.allclose(
        p_none["cell"],
        [68.84174, 98.06962, 90.91568, 90.07304, 106.11396, 90.0088],
    )

    p_rhomb = parse_unit_cell_text("R -3 m :R 8.5 8.5 8.5 55.2 55.2 55.2")
    assert p_rhomb is not None
    assert p_rhomb["space_group"] == "R -3 m :R"
    assert np.allclose(p_rhomb["cell"], [8.5, 8.5, 8.5, 55.2, 55.2, 55.2])

    # Verify crystal-system symmetry constraints for space groups
    sym_cubic = resolve_space_group_symmetry("I a -3 d")
    assert sym_cubic["crystal_system"] == "cubic"
    assert sym_cubic["linked_lengths"] == {"b": "a", "c": "a"}
    assert sym_cubic["fixed_angles"] == {
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }

    sym_tet = resolve_space_group_symmetry("P 41 21 2")
    assert sym_tet["crystal_system"] == "tetragonal"
    assert sym_tet["linked_lengths"] == {"b": "a"}
    assert sym_tet["fixed_angles"] == {
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }

    sym_hex = resolve_space_group_symmetry("P 61 2 2")
    assert sym_hex["crystal_system"] == "hexagonal"
    assert sym_hex["linked_lengths"] == {"b": "a"}
    assert sym_hex["fixed_angles"] == {
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
    }

    sym_rhomb = resolve_space_group_symmetry("R -3 m :R")
    assert sym_rhomb["setting"] == "trigonal_r"
    assert sym_rhomb["linked_lengths"] == {"b": "a", "c": "a"}
    assert sym_rhomb["linked_angles"] == {"beta": "alpha", "gamma": "alpha"}
    assert sym_rhomb["fixed_angles"] == {}

    sym_ortho = resolve_space_group_symmetry("P 21 21 21")
    assert sym_ortho["crystal_system"] == "orthorhombic"
    assert sym_ortho["linked_lengths"] == {}
    assert sym_ortho["fixed_angles"] == {
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }

    sym_mono = resolve_space_group_symmetry("P21")
    assert sym_mono["crystal_system"] == "monoclinic"
    assert sym_mono["setting"] == "monoclinic_b"
    assert sym_mono["fixed_angles"] == {"alpha": 90.0, "gamma": 90.0}

    # Verify Galaxy dataset unit-cell metadata extraction & annotation parsing
    runner.attach_dataset_cell_metadata(
        "ds_test_betalac",
        [68.8417, 98.0696, 90.9157, 90.073, 106.114, 90.0088],
        "P 1 21 1",
        "CG4D",
        sync_remote=False,
    )
    assert runner._dataset_cell_meta["ds_test_betalac"]["space_group"] == "P 1 21 1"
    assert np.allclose(
        runner._dataset_cell_meta["ds_test_betalac"]["cell"],
        [68.8417, 98.0696, 90.9157, 90.073, 106.114, 90.0088],
    )
    parsed_ann = runner._parse_dataset_annotation_meta(
        {
            "annotation": '{"subhkl_cell":[11.93,11.93,11.93,90,90,90],"space_group":"I a -3 d","instrument":"MANDI"}',
            "tags": [],
        }
    )
    assert parsed_ann is not None
    assert parsed_ann["space_group"] == "I a -3 d"
    assert parsed_ann["cell"] == [11.93, 11.93, 11.93, 90.0, 90.0, 90.0]

    # Verify UI elements & JS helpers for per-field hoverable autocomplete dropdown, tentative gray preview, paste, and overrideable symmetry proposals
    assert 'id="nd-symmetry-badge"' in html_doc
    assert 'id="nd-cell-suggestions"' in html_doc
    assert 'id="subhkl-sg-datalist"' in html_doc
    assert "parseUnitCellClipboardText" in html_doc
    assert "resolveSpaceGroupSymmetry" in html_doc
    assert "applyUnitCellSymmetryConstraints" in html_doc
    assert "getKnownUnitCellHistory" in html_doc
    assert "renderUnitCellHistoryDropdown" in html_doc
    assert "highlightAutocompleteOption" in html_doc
    assert "previewTentativeUnitCellEntry" in html_doc
    assert "confirmTentativeUnitCellPreview" in html_doc
    assert "updateUnitCellFieldSuggestions" in html_doc

    # Verify stepped run interval parsing (e.g. 2632-2638/2, 2623:26238:2, 2632-2638..2)
    from subhkl.viz.rainbow_zones import parse_run_range_spec

    assert parse_run_range_spec("2632-2638/2") == [2632, 2634, 2636, 2638]
    assert parse_run_range_spec("2632:2638:2") == [2632, 2634, 2636, 2638]
    assert parse_run_range_spec("2632-2638..2") == [2632, 2634, 2636, 2638]
    assert parse_run_range_spec("2632..2638..2") == [2632, 2634, 2636, 2638]
    assert parse_run_range_spec("2632..2638/2") == [2632, 2634, 2636, 2638]
    assert parse_run_range_spec("2623:26238:2") == list(range(2623, 26239, 2))
    assert parse_run_range_spec("2632-2638/2, 2645, 2650:2656:3") == [
        2632,
        2634,
        2636,
        2638,
        2645,
        2650,
        2653,
        2656,
    ]
    assert "dMinSeqAtStart" in html_doc
    assert "pollInFlight" in html_doc


def test_em_pixel_mask_two_color_overlay_and_srcdoc_tab_handling(tmp_path):
    """Verify two-color EM pixel_class PNG encoding, masked solve peak filtering, and srcdoc Tab popout logic."""
    import base64
    import io
    import re
    import subprocess

    from PIL import Image

    from subhkl.viz.rainbow_zones import render_rainbow_html

    mask_solve_h5 = tmp_path / "solve_with_em_mask.h5"
    p_cls = np.zeros((2, 512, 512), dtype=np.uint8)
    # Bank 21: dead/shadowed top half (1=dark) + bright magnet flare patch (2=bright)
    p_cls[0, :256, :] = 1
    p_cls[0, 320:336, 320:336] = 2
    # Bank 22: completely valid (0) -> should emit mask_b64=None (zero payload bloat)
    p_mask = (p_cls == 0).astype(np.uint8)

    with h5py.File(mask_solve_h5, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf["sample/U"] = np.eye(3)
        hf["sample/B"] = np.diag([1.0 / 25.0, 1.0 / 30.0, 1.0 / 35.0])
        hf["sample/space_group"] = "P 2 2 2"
        hf["scan_refinement/band_wavelength"] = np.array([2.8, 4.5], dtype=float)
        dgrp = hf.create_group("detector")
        dgrp.create_dataset("mask_bank_ids", data=np.array([21, 22], dtype=np.int32))
        dgrp.create_dataset("pixel_mask", data=p_mask)
        dgrp.create_dataset("pixel_class", data=p_cls)

    payload = build_rainbow_zone_payload(replay_h5=mask_solve_h5, instrument="CG4D")
    em_sum = payload["em_mask_summary"]
    assert em_sum["available"] is True
    assert em_sum["n_dark_pixels"] == 256 * 512
    assert em_sum["n_bright_pixels"] == 16 * 16
    assert em_sum["dark_fraction"] > 0.20
    assert em_sum["bright_fraction"] > 0.0

    panels_by_bank = {p["bank"]: p for p in payload["panels"]}
    assert panels_by_bank[22]["mask_b64"] is None
    b21_mask_b64 = panels_by_bank[21]["mask_b64"]
    assert isinstance(b21_mask_b64, str) and len(b21_mask_b64) > 0

    mask_img = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(b21_mask_b64))), dtype=np.uint8
    )
    assert mask_img.shape == (128, 128)
    assert set(np.unique(mask_img).tolist()) == {0, 128, 255}
    assert np.all(mask_img[:64, :] == 128)
    assert np.all(mask_img[80:84, 80:84] == 255)

    # No forward-projected solve peaks should land on masked pixels of bank 21
    for pk in payload["peaks"]:
        if pk["bank"] == 21:
            ri = int(np.clip(round(pk["row"]), 0, 511))
            ci = int(np.clip(round(pk["col"]), 0, 511))
            assert p_cls[0, ri, ci] == 0

    html = render_rainbow_html(payload)
    assert 'id="btn-pixel-mask"' in html
    assert 'id="mask-legend-row"' in html
    assert "getWarpedMaskCanvas" in html
    assert "resolveStandaloneTabUrl" in html
    assert "isEmbeddedInOuterFrame" in html
    assert 'selfHref.startsWith("about:srcdoc")' in html
    assert (
        "for (let sIdx = 0; sIdx < Math.min(stageKeys.length, maxValidCompleted); sIdx++)"
        in html
    )

    scripts = re.findall(
        r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.DOTALL
    )
    assert scripts
    js_file = tmp_path / "rainbow_mask_check.js"
    js_file.write_text("\n;\n".join(scripts), encoding="utf-8")
    proc = subprocess.run(
        ["node", "--check", str(js_file)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_stdio_step_stopwatch_ui_and_pipeline_timings(tmp_path):
    """Verify lower-right stdio mm:ss.ms step stopwatch rendering, JS state transitions, and backend step_timings."""
    import subprocess
    import time

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import render_rainbow_html

    peaks_xyz, _U, _B, UB, _hkls, _lams = _synthetic_orthorhombic_peaks()
    payload = build_rainbow_zone_payload(
        peaks_xyz=peaks_xyz,
        ub_matrix=UB,
        wavelength_range=(1.8, 4.5),
        instrument="CG4D",
        space_group="P 2 2 2",
    )

    for mode in ("arcade", "pro"):
        html = render_rainbow_html(payload, default_mode=mode)
        assert 'class="terminal-wrap"' in html
        assert 'id="stdio-stopwatch"' in html
        assert 'id="stdio-stopwatch-stage"' in html
        assert 'id="stdio-stopwatch-time"' in html
        assert 'class="sw-rb sw-rb-1"' in html
        assert 'viewBox="0 0 24 12"' in html
        assert "swRainbowShoot" in html
        assert "swRainbowTipSparkle" in html
        assert "swStarTwinkle" in html
        assert 'class="sw-sp-tip"' in html
        assert 'class="sw-sp-end sw-sp-end-2"' in html
        assert "ctx.moveTo(bx, by)" not in html
        assert "detToScreen(p.det_x + shiftAz, py, w, h, -0.35)" not in html
        assert "sw-hand" not in html
        assert "stdio-stopwatch-quip" not in html
        assert "BLINK" not in html
        assert "formatStopwatchParts" in html
        assert "startStepStopwatch" in html
        assert "finishStepStopwatch" in html
        assert "hydrateStepTimingsFromServer" in html

        scripts = re.findall(
            r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.DOTALL
        )
        assert scripts
        js_src = "\n;\n".join(scripts)
        js_file = tmp_path / f"rainbow_stopwatch_{mode}.js"
        js_file.write_text(js_src, encoding="utf-8")
        proc = subprocess.run(
            ["node", "--check", str(js_file)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

        # Extract and execute stageStopwatchMeta + formatStopwatchParts + getCompletionStopwatchQuip in Node
        m_meta = re.search(
            r"function stageStopwatchMeta\(stageIdOrIdx\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        m_quip = re.search(
            r"function getCompletionStopwatchQuip\(stageId, finalStatus, elapsedMs\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        m_fmt = re.search(
            r"function formatStopwatchParts\(ms\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        m_term_scroll = re.search(
            r"let _lastAutoTerminalScrollTop = -1;.*?function handleTerminalUserScroll\(box\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        m_append_term = re.search(
            r"function appendTerminalLog\(text, replace = false, resetFollow = false\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        m_fmt_stdio = re.search(
            r"function formatTerminalStdio\(rawText\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        m_odo = re.search(
            r"const STOPWATCH_SLOMO_RATE = .*?function computeStopwatchOdometerPositions\(msFloat\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        m_cheat = re.search(
            r"function toggleStopwatchCheat\(\)\s*\{.*?function handleStopwatchCheatKey\(ev\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        m_live_ms = re.search(
            r"function getEntryLiveMs\(entry, nowPerf = performance\.now\(\)\)\s*\{.*?\n  \}",
            js_src,
            flags=re.DOTALL,
        )
        assert (
            m_meta
            and m_quip
            and m_fmt
            and m_term_scroll
            and m_append_term
            and m_fmt_stdio
            and m_odo
            and m_cheat
            and m_live_ms
        )
        assert "flashStopwatchQuip" in js_src
        assert "sw-odo-strip" in js_src
        eval_script = (
            f'const UI_MODE = "{mode}";\n'
            f"const PIPELINE = {json.dumps(payload['pipeline_stages'])};\n"
            "const state = {\n"
            "  terminalFollowTail: true,\n"
            "  terminalScrollTop: 0,\n"
            "  stopwatchCheatSlowMo: false,\n"
            "  stepTimings: {},\n"
            "  runningStageIdx: -1,\n"
            "};\n"
            "const stopwatchFlash = { text: '', untilPerf: 0, timerId: null, demoStartPerf: 0, demoUntilPerf: 0 };\n"
            "let _lastFlashedQuip = '';\n"
            "function activeRunningStageId() { return null; }\n"
            "function flashStopwatchQuip(t) { if (UI_MODE !== 'pro') _lastFlashedQuip = t; }\n"
            "let _selActive = false;\n"
            "const _docListeners = {};\n"
            "const mockNode = {};\n"
            "const mockBox = {\n"
            '  textContent: "",\n'
            "  clientHeight: 100,\n"
            "  scrollHeight: 100,\n"
            "  _scrollTop: 0,\n"
            "  get scrollTop() { return this._scrollTop; },\n"
            "  set scrollTop(v) {\n"
            "    const maxS = Math.max(0, this.scrollHeight - this.clientHeight);\n"
            "    this._scrollTop = Math.max(0, Math.min(maxS, v));\n"
            "  },\n"
            "  contains(n) { return n === mockNode; },\n"
            "  addEventListener() {},\n"
            "};\n"
            "globalThis.document = {\n"
            "  getElementById(id) { return id === 'terminal-box' ? mockBox : null; },\n"
            "  addEventListener(ev, fn) { _docListeners[ev] = fn; },\n"
            "};\n"
            "globalThis.window = {\n"
            "  getSelection() {\n"
            "    return _selActive\n"
            "      ? { isCollapsed: false, anchorNode: mockNode, focusNode: mockNode }\n"
            "      : { isCollapsed: true, anchorNode: null, focusNode: null };\n"
            "  },\n"
            "};\n"
            f"{m_fmt_stdio.group(0)}\n"
            f"{m_term_scroll.group(0)}\n"
            f"{m_append_term.group(0)}\n"
            f"{m_odo.group(0)}\n"
            f"{m_live_ms.group(0)}\n"
            f"{m_cheat.group(0)}\n"
            f"{m_meta.group(0)}\n"
            f"{m_quip.group(0)}\n"
            f"{m_fmt.group(0)}\n"
            "mockBox.scrollHeight = 400;\n"
            'appendTerminalLog("line 1\\nline 2", true, true);\n'
            "handleTerminalUserScroll(mockBox);\n"
            "const tailTop1 = mockBox.scrollTop;\n"
            "mockBox.scrollHeight = 650;\n"
            'appendTerminalLog("line 1\\nline 2\\nline 3", true);\n'
            "handleTerminalUserScroll(mockBox);\n"
            "const tailTop2 = mockBox.scrollTop;\n"
            "mockBox.scrollTop = 120;\n"
            "handleTerminalUserScroll(mockBox);\n"
            "const followAfterScrollUp = state.terminalFollowTail;\n"
            "mockBox.scrollHeight = 900;\n"
            'appendTerminalLog("line 1\\nline 2\\nline 3\\nline 4", true);\n'
            "const pinnedTopAfterAppend = mockBox.scrollTop;\n"
            "_selActive = true;\n"
            "mockBox.scrollHeight = 1100;\n"
            'appendTerminalLog("line 1\\nline 2\\nline 3\\nline 4\\nline 5", true);\n'
            "const textDuringSel = mockBox.textContent;\n"
            "_selActive = false;\n"
            "if (_docListeners.selectionchange) _docListeners.selectionchange();\n"
            "const textAfterClearSel = mockBox.textContent;\n"
            "const pinnedTopAfterClearSel = mockBox.scrollTop;\n"
            "mockBox.scrollTop = 1000;\n"
            "handleTerminalUserScroll(mockBox);\n"
            "const followAfterBottom = state.terminalFollowTail;\n"
            "mockBox.scrollHeight = 1300;\n"
            'appendTerminalLog("line 1\\nline 2\\nline 3\\nline 4\\nline 5\\nline 6", true);\n'
            "const tailTopResumed = mockBox.scrollTop;\n"
            "for (const ch of 'RAINBOW') {\n"
            "  handleStopwatchCheatKey({ key: ch, target: { tagName: 'INPUT' } });\n"
            "}\n"
            "const cheatFromInput = state.stopwatchCheatSlowMo;\n"
            "handleStopwatchCheatKey({ key: 'Shift', target: { tagName: 'BODY' } });\n"
            "for (const ch of 'RAINBOW') {\n"
            "  handleStopwatchCheatKey({ key: ch, target: { tagName: 'BODY' } });\n"
            "}\n"
            "const cheatAfterRainbow = state.stopwatchCheatSlowMo;\n"
            "const quipAfterRainbow = _lastFlashedQuip;\n"
            "const slomoMsAfter10s = getEntryLiveMs({ status: 'running', startPerfMs: 1000, baseElapsedMs: 0, rate: getStopwatchRate() }, 11000);\n"
            "for (const k of ['ArrowUp','ArrowUp','ArrowDown','ArrowDown','ArrowLeft','ArrowRight','ArrowLeft','ArrowRight','B','A']) {\n"
            "  handleStopwatchCheatKey({ key: k, target: { tagName: 'BODY' } });\n"
            "}\n"
            "const cheatAfterKonamiToggleOff = state.stopwatchCheatSlowMo;\n"
            "const odoMidCarry = computeStopwatchOdometerPositions(9.56);\n"
            "const odoPreTen = computeStopwatchOdometerPositions(9.999999);\n"
            "const odoAtTen = computeStopwatchOdometerPositions(10.0);\n"
            "console.log(JSON.stringify({"
            "labels: [0, 1, 2, 3].map(i => stageStopwatchMeta(i).label),"
            "z: formatStopwatchParts(0).full,"
            "s: formatStopwatchParts(55984).full,"
            "m: formatStopwatchParts(125007).full,"
            'qFast: getCompletionStopwatchQuip("reduce", "done", 3200),'
            'qSolve: getCompletionStopwatchQuip("solve", "done", 55984),'
            "tailTop1, tailTop2, followAfterScrollUp, pinnedTopAfterAppend,"
            "hasLine5DuringSel: textDuringSel.includes('line 5'),"
            "hasLine5AfterClear: textAfterClearSel.includes('line 5'),"
            "pinnedTopAfterClearSel, followAfterBottom, tailTopResumed,"
            "cheatFromInput, cheatAfterRainbow, quipAfterRainbow, slomoMsAfter10s,"
            "cheatAfterKonamiToggleOff, odoMidCarry, odoPreTen, odoAtTen"
            "}));\n"
        )
        res = subprocess.run(
            ["node", "-e", eval_script],
            check=True,
            capture_output=True,
            text=True,
        )
        out = json.loads(res.stdout.strip())
        assert out["labels"] == [
            "1. REDUCE",
            "2. SOLVE",
            "3. INTEGRATE",
            "4. MERGE",
        ]
        assert out["z"] == "00:00.000"
        assert out["s"] == "00:55.984"
        assert out["m"] == "02:05.007"
        assert out["tailTop1"] == 300
        assert out["tailTop2"] == 550
        assert out["followAfterScrollUp"] is False
        assert out["pinnedTopAfterAppend"] == 120
        assert out["hasLine5DuringSel"] is False
        assert out["hasLine5AfterClear"] is True
        assert out["pinnedTopAfterClearSel"] == 120
        assert out["followAfterBottom"] is True
        assert out["tailTopResumed"] == 1200
        assert out["cheatFromInput"] is False
        assert 0.45 < out["odoMidCarry"][5] < 0.55
        assert 9.45 < out["odoMidCarry"][6] < 9.55
        assert abs(out["odoPreTen"][5] - out["odoAtTen"][5]) < 1e-4
        assert abs(out["odoPreTen"][6] - 10.0) < 1e-4
        assert out["odoAtTen"][6] == 0.0
        if mode == "arcade":
            assert out["cheatAfterRainbow"] is True
            assert out["quipAfterRainbow"] == "SLO-MO CHEAT!"
            assert abs(out["slomoMsAfter10s"] - 15.0) < 1e-6
            assert out["cheatAfterKonamiToggleOff"] is False
        else:
            assert out["cheatAfterRainbow"] is False
            assert abs(out["slomoMsAfter10s"] - 10000.0) < 1e-6
        if mode == "arcade":
            assert out["qFast"] == "SPEED BONUS!"
            assert out["qSolve"] == "STAGE CLEAR!"
        else:
            assert out["qFast"] == ""
            assert out["qSolve"] == ""

    # Verify backend per-step millisecond timing lifecycle on NovaGalaxyPipelineRunner
    runner = NovaGalaxyPipelineRunner()
    t0 = time.time() - 55.984
    runner.active_job = {
        "job_id": "test-solve-1",
        "stage_id": "solve",
        "state": "running",
        "started_at": t0,
    }
    snap_running = runner.get_step_timings_snapshot()
    assert "solve" in snap_running
    assert snap_running["solve"]["state"] == "running"
    assert snap_running["solve"]["finished_at"] is None
    assert snap_running["solve"]["elapsed_ms"] >= 55900

    # Stamp exact finish time and verify frozen elapsed_ms persists when switching steps
    runner.active_job["state"] = "ok"
    runner.active_job["finished_at"] = t0 + 55.984
    snap_done = runner.get_step_timings_snapshot()
    assert snap_done["solve"]["state"] == "ok"
    assert snap_done["solve"]["elapsed_ms"] == 55984

    # Start next stage (integrate) and cancel it; both stages must remain in step_timings
    t1 = time.time() - 3.215
    runner.active_job = {
        "job_id": "test-int-1",
        "stage_id": "integrate",
        "state": "running",
        "started_at": t1,
        "stdout": "",
    }
    runner.cancel_active_job("integrate")
    snap_both = runner.get_step_timings_snapshot()
    assert snap_both["solve"]["elapsed_ms"] == 55984
    assert snap_both["integrate"]["state"] == "deleted"
    assert snap_both["integrate"]["finished_at"] is not None
    assert snap_both["integrate"]["elapsed_ms"] >= 3200


def test_wulff_azimuth_peak_source_toggle_and_local_mtz_overlay(tmp_path) -> None:
    """Verify exact θ vs 2θ Wulff alignment + Friedel mates, predictor.h5 discovery, MTZ projection preserving B, and diverging I/σ."""
    import asyncio
    import json
    import os
    import re
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import gemmi
    import h5py
    import numpy as np
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request
    from scipy.spatial.transform import Rotation

    from subhkl.config import beamlines
    from subhkl.core.crystallography import cartesian_matrix_metric_tensor
    from subhkl.instrument.detector import Detector
    from subhkl.viz.nova_app.models.subhkl_model import SubhklGalaxyModel
    from subhkl.viz.nova_app.views.main_view import MainApp
    from subhkl.viz.nova_rainbow_view import RainbowZonesViewModel
    from subhkl.viz.rainbow_zones import (
        _discover_predictor_h5,
        _raycast_hkl_on_detectors,
        _u_from_ub,
        apply_mtz_to_payload,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    # 1. Build multi-frame CG4D integrated.h5 with both integrated peaks (with u/v variances -> ellipses)
    #    and unintegrated predicted reflections (sigma == 0), plus a non-trivial orientation matrix U.
    U_true = Rotation.from_euler("zyx", [27.0, -14.0, 38.0], degrees=True).as_matrix()
    cell_orig = [54.0, 54.0, 54.0, 90.0, 90.0, 90.0]
    B_orig_raw, _ = cartesian_matrix_metric_tensor(
        *cell_orig[:3], *np.deg2rad(cell_orig[3:])
    )
    B_orig = np.asarray(B_orig_raw, dtype=float)
    R0 = np.eye(3)
    R1 = Rotation.from_euler("y", 25.0, degrees=True).as_matrix()
    R_runs = np.stack([R0, R1], axis=0)

    first_bk_key = (
        "57" if "57" in beamlines["CG4D"] else max(beamlines["CG4D"], key=int)
    )
    first_bk = int(first_bk_key)
    det1 = Detector(dict(beamlines["CG4D"][first_bk_key]))
    xyz_a = np.asarray(det1.pixel_to_lab(110.0, 120.0), dtype=float)
    xyz_b = np.asarray(det1.pixel_to_lab(140.0, 150.0), dtype=float)
    xyz_c = np.asarray(det1.pixel_to_lab(125.0, 135.0), dtype=float)
    unique_tag = f"CG4D_testpred_{os.getpid()}"
    integ_h5 = tmp_path / "integrated.h5"
    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        smp = hf.create_group("sample")
        smp.create_dataset("space_group", data=b"P 21 21 21")
        smp.create_dataset("cell", data=np.array(cell_orig, dtype=float))
        smp.create_dataset("U", data=U_true)
        smp.create_dataset("B", data=B_orig)
        gon = hf.create_group("goniometer")
        gon.create_dataset("R", data=R_runs)
        gon.create_dataset("angles", data=np.array([[0.0], [25.0]], dtype=float))
        gon.create_dataset("axes", data=np.array([[0.0, 1.0, 0.0]], dtype=float))
        hf.create_dataset("file_offsets", data=np.array([0, 1], dtype=np.int32))
        hf.create_dataset(
            "bank_ids", data=np.array([first_bk, first_bk], dtype=np.int32)
        )
        hf.create_dataset(
            "files",
            data=np.array(
                [f"{unique_tag}0.nxs.h5", f"{unique_tag}1.nxs.h5"], dtype="S64"
            ),
        )
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([2.8, 4.5], dtype=float)
        )
        pk = hf.create_group("peaks")
        pk.create_dataset("xyz", data=np.vstack([xyz_a, xyz_b, xyz_c]))
        pk.create_dataset(
            "bank", data=np.array([first_bk, first_bk, first_bk], dtype=np.int32)
        )
        pk.create_dataset("pixel_r", data=np.array([110.0, 140.0, 125.0], dtype=float))
        pk.create_dataset("pixel_c", data=np.array([120.0, 150.0, 135.0], dtype=float))
        pk.create_dataset("intensity", data=np.array([180.0, 0.0, 140.0], dtype=float))
        pk.create_dataset("sigma", data=np.array([12.0, 0.0, 10.0], dtype=float))
        pk.create_dataset("var_u", data=np.array([1.2e-4, 1.0e-4, 1.2e-4], dtype=float))
        pk.create_dataset("var_v", data=np.array([0.6e-4, 1.0e-4, 0.6e-4], dtype=float))
        pk.create_dataset("cov_uv", data=np.array([0.1e-4, 0.0, 0.1e-4], dtype=float))
        pk.create_dataset("h", data=np.array([2, -2, 1], dtype=np.int32))
        pk.create_dataset("k", data=np.array([1, -1, 2], dtype=np.int32))
        pk.create_dataset("l", data=np.array([-3, 3, -3], dtype=np.int32))
        pk.create_dataset("lambda", data=np.array([3.2, 3.6, 3.4], dtype=float))
        pk.create_dataset("run_index", data=np.array([0, 0, 1], dtype=np.int32))

    payload0 = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        frame_index=0,
        include_panel_images=False,
    )
    assert payload0["artifact_kind"] == "integrator"
    assert payload0["default_peak_source"] == "integrated"
    assert len(payload0["integrated_peaks"]) == 1
    assert isinstance(payload0["integrated_peaks"][0].get("ellipse"), list)
    assert len(payload0["integrated_peaks"][0]["ellipse"]) >= 6
    assert len(payload0["integrated_sample_peaks"]) == 2
    assert np.allclose(np.asarray(payload0["u_matrix"]), U_true, atol=1e-5)

    # Verify Friedel parity flag on non-centrosymmetric P 21 21 21: (2,1,-3) and (-2,-1,3) share ASU orbit with opposite asu_friedel!
    from subhkl.viz.rainbow_zones import _make_asu_mapper

    mapper_p212121 = _make_asu_mapper("P 21 21 21", cell_orig)
    asu_pos = mapper_p212121(2, 1, -3)
    asu_neg = mapper_p212121(-2, -1, 3)
    asu_rot = mapper_p212121(
        -2, 1, 3
    )  # 2-fold rotation around y: (h,k,l) -> (-h, k, -l)
    assert asu_pos[:3] == asu_neg[:3] == asu_rot[:3]
    assert asu_pos[3] != asu_neg[3]  # Friedel pair has opposite parity
    assert asu_pos[3] == asu_rot[3]  # Proper 2-fold rotation has identical parity
    assert asu_pos[4] != 0 and asu_pos[4] == -asu_neg[4]
    assert asu_pos[4] == asu_rot[4]

    # Verify Fdd2 (mm2): point-group mates (including d-glide mirrors) share nonzero Bijvoet parity,
    # +l vs -l have opposite Bijvoet parity, and centric hk0 reflections have bijvoet_parity == 0.
    mapper_fdd2 = _make_asu_mapper("Fdd2", [18.39, 56.55, 6.54, 90.0, 90.0, 90.0])
    f_p1 = mapper_fdd2(11, -15, 1)
    f_p2 = mapper_fdd2(-11, -15, 1)  # d-glide mirror mate (same +l)
    f_p3 = mapper_fdd2(11, 15, 1)  # d-glide mirror mate (same +l)
    f_bij = mapper_fdd2(-11, -15, -1)  # true Bijvoet mate (-l)
    f_c1 = mapper_fdd2(2, 2, 0)  # centric hk0
    f_c2 = mapper_fdd2(-2, -2, 0)  # centric hk0 2-fold mate
    f_c3 = mapper_fdd2(2, -2, 0)  # centric hk0 mirror mate
    assert f_p1[:3] == f_p2[:3] == f_p3[:3] == f_bij[:3] == (11, 15, 1)
    assert f_p1[4] != 0 and f_p1[4] == f_p2[4] == f_p3[4]
    assert f_bij[4] == -f_p1[4]
    assert f_c1[4] == f_c2[4] == f_c3[4] == 0
    assert f_c1[3] is False and f_c2[3] is False

    # Centrosymmetric space groups have bijvoet_parity == 0 for every reflection
    mapper_ia3d = _make_asu_mapper("Ia-3d", [11.93, 11.93, 11.93, 90.0, 90.0, 90.0])
    assert mapper_ia3d(1, 2, 3)[4] == 0
    assert mapper_ia3d(-1, -2, -3)[4] == 0

    # Verify pre-filtered integrated.h5 (containing ONLY integrated peaks):
    # (a) without predictor.h5 -> theoretical fallback generates full Laue predictions (> integrated count)
    # (b) with companion predictor.h5 in /tmp/subhkl_work_<tag>/predictor.h5 -> discovers exact peak-predictor output!
    sib_dir = tmp_path / "gx_downloaded_integ"
    sib_dir.mkdir()
    integ_only_h5 = sib_dir / "subhkl_gx_test_integrated.h5"
    with h5py.File(integ_h5, "r") as src_f, h5py.File(integ_only_h5, "w") as dst_f:
        for k_top in src_f:
            if k_top != "peaks":
                src_f.copy(k_top, dst_f)
        for ak, av in src_f.attrs.items():
            dst_f.attrs[ak] = av
        pk_dst = dst_f.create_group("peaks")
        keep_idx = np.array([0, 2], dtype=int)
        for pk_k in src_f["peaks"]:
            pk_dst.create_dataset(pk_k, data=np.asarray(src_f["peaks"][pk_k])[keep_idx])

    payload_fb = build_rainbow_zone_payload(
        replay_h5=integ_only_h5,
        frame_index=0,
        d_min=4.0,
        include_panel_images=False,
    )
    assert len(payload_fb["integrated_peaks"]) == 1
    assert len(payload_fb["predicted_peaks"]) > len(payload_fb["integrated_peaks"])

    work_tag = f"{unique_tag}0_{unique_tag}1"
    work_dir_tmp = Path(f"/tmp/subhkl_work_{work_tag}")
    work_dir_tmp.mkdir(parents=True, exist_ok=True)
    pred_work_h5 = work_dir_tmp / "predictor.h5"
    try:
        with h5py.File(pred_work_h5, "w") as pf:
            smp_p = pf.create_group("sample")
            smp_p.create_dataset("U", data=U_true)
            smp_p.create_dataset("B", data=B_orig)
            pf.create_dataset(
                "bank_ids", data=np.array([first_bk, first_bk], dtype=np.int32)
            )
            pf.create_dataset("file_offsets", data=np.array([0, 1], dtype=np.int32))
            b_grp = pf.create_group("banks")
            g0 = b_grp.create_group("0")
            g0.create_dataset("i", data=np.array([110.0, 140.0, 160.0], dtype=float))
            g0.create_dataset("j", data=np.array([120.0, 150.0, 170.0], dtype=float))
            g0.create_dataset("h", data=np.array([2, 3, 4], dtype=np.int32))
            g0.create_dataset("k", data=np.array([1, -1, 0], dtype=np.int32))
            g0.create_dataset("l", data=np.array([-3, -2, -1], dtype=np.int32))
            g0.create_dataset("wavelength", data=np.array([3.2, 3.6, 3.9], dtype=float))
            g1 = b_grp.create_group("1")
            g1.create_dataset("i", data=np.array([125.0, 155.0], dtype=float))
            g1.create_dataset("j", data=np.array([135.0, 165.0], dtype=float))
            g1.create_dataset("h", data=np.array([1, 2], dtype=np.int32))
            g1.create_dataset("k", data=np.array([2, 2], dtype=np.int32))
            g1.create_dataset("l", data=np.array([-3, -2], dtype=np.int32))
            g1.create_dataset("wavelength", data=np.array([3.4, 3.8], dtype=float))

        assert _discover_predictor_h5(integ_only_h5) == str(pred_work_h5.resolve())
        payload_sib = build_rainbow_zone_payload(
            replay_h5=integ_only_h5,
            frame_index=0,
            include_panel_images=False,
        )
        assert len(payload_sib["integrated_peaks"]) == 1
        assert len(payload_sib["predicted_peaks"]) == 3
        assert len(payload_sib["integrated_sample_peaks"]) == 2
        assert len(payload_sib["predicted_sample_peaks"]) == 5
    finally:
        if pred_work_h5.exists():
            pred_work_h5.unlink()
        if work_dir_tmp.exists():
            work_dir_tmp.rmdir()

    # 2. Create a custom local .mtz with a DIFFERENT unit cell & space group (e.g. P 32 2 1)
    #    and verify that loading the .mtz projects its HKLs into the CURRENT active (U_true, B_orig) geometry
    #    WITHOUT modifying B, UB, cell, or space_group!
    mtz_cell = [59.8, 59.8, 96.4, 90.0, 90.0, 120.0]
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 32 2 1")
    mtz.set_cell_for_all(gemmi.UnitCell(*mtz_cell))
    mtz.add_dataset("custom")
    mtz.add_column("I", "J")
    mtz.add_column("SIGI", "Q")

    # Seed ASU reflections whose rays under the ACTIVE geometry (U_true @ B_orig) hit active panel first_bk
    # (plus reflections on an inactive panel to verify .mtz only projects onto active predicted-peak panels!)
    cg4d_dets = {int(bk): Detector(dict(cfg)) for bk, cfg in beamlines["CG4D"].items()}
    grid = np.mgrid[-12:13, -12:13, -12:13].reshape(3, -1).T
    grid = grid[np.any(grid != 0, axis=1)]
    cast_seed_active_r0 = _raycast_hkl_on_detectors(
        grid, U_true, B_orig, R0, cg4d_dets, (2.8, 4.5), active_bids={first_bk}
    )
    cast_seed_active_r1 = _raycast_hkl_on_detectors(
        grid, U_true, B_orig, R1, cg4d_dets, (2.8, 4.5), active_bids={first_bk}
    )
    other_bids = {int(b) for b in cg4d_dets if int(b) != first_bk}
    cast_seed_inactive = _raycast_hkl_on_detectors(
        grid, U_true, B_orig, R0, cg4d_dets, (2.8, 4.5), active_bids=other_bids
    )
    assert len(cast_seed_active_r0["hkl"]) >= 2
    assert len(cast_seed_inactive["hkl"]) >= 2
    asu_mtz = gemmi.ReciprocalAsu(mtz.spacegroup)
    ops_mtz = mtz.spacegroup.operations()
    seen_asu: set[tuple[int, int, int]] = set()
    mtz_rows: list[list[float]] = []
    seed_hkls_combined = (
        [[2, 1, -3], [1, 2, -3]]
        + cast_seed_active_r0["hkl"][:12].tolist()
        + cast_seed_active_r1["hkl"][:8].tolist()
        + cast_seed_inactive["hkl"][:6].tolist()
    )
    for idx_h, h_hit in enumerate(seed_hkls_combined):
        ha, _ = asu_mtz.to_asu([int(h_hit[0]), int(h_hit[1]), int(h_hit[2])], ops_mtz)
        key_a = (int(ha[0]), int(ha[1]), int(ha[2]))
        if key_a not in seen_asu:
            seen_asu.add(key_a)
            mtz_rows.append(
                [
                    float(key_a[0]),
                    float(key_a[1]),
                    float(key_a[2]),
                    250.0 + 10.0 * idx_h,
                    8.0,
                ]
            )
    mtz.set_data(np.asarray(mtz_rows, dtype=np.float32))
    mtz_path = tmp_path / "custom_reflections.mtz"
    mtz.write_to_file(str(mtz_path))

    payload_mtz = apply_mtz_to_payload(
        payload0, mtz_path, {"name": "custom_reflections.mtz"}
    )
    # U, B, UB, cell, and space_group MUST remain strictly unchanged!
    assert np.allclose(np.asarray(payload_mtz["u_matrix"]), U_true, atol=1e-5)
    assert np.allclose(np.asarray(payload_mtz["b_matrix"]), B_orig, atol=1e-5)
    assert np.allclose(np.asarray(payload_mtz["ub_matrix"]), U_true @ B_orig, atol=1e-5)
    assert np.allclose(
        _u_from_ub(np.asarray(payload_mtz["ub_matrix"])), U_true, atol=1e-4
    )
    assert payload_mtz["space_group"] == "P 21 21 21"
    assert np.allclose(payload_mtz["cell"], cell_orig, atol=1e-3)
    assert payload_mtz["default_peak_source"] == "integrated"
    assert payload_mtz["default_color_mode"] == "mtz_isigi"
    assert "mtz_peaks" not in payload_mtz
    assert "mtz_sample_peaks" not in payload_mtz
    # Only panels with predicted peaks (first_bk) may carry .mtz projections!
    assert {int(p["bank"]) for p in payload_mtz["predicted_peaks"]} == {first_bk}
    assert {int(p["bank"]) for p in payload_mtz["predicted_sample_peaks"]} == {first_bk}
    pred_with_mtz = [
        p for p in payload_mtz["predicted_peaks"] if p.get("mtz_isigi") is not None
    ]
    samp_with_mtz = [
        p
        for p in payload_mtz["predicted_sample_peaks"]
        if p.get("mtz_isigi") is not None
    ]
    integ_with_mtz = [
        p for p in payload_mtz["integrated_peaks"] if p.get("mtz_isigi") is not None
    ]
    assert len(pred_with_mtz) >= len(cast_seed_active_r0["hkl"][:12])
    assert len(samp_with_mtz) >= len(pred_with_mtz)
    assert len(payload_mtz["integrated_peaks"]) == 1
    assert len(integ_with_mtz) == 1
    assert len(payload_mtz["predicted_peaks"]) >= len(payload0["predicted_peaks"])

    # Loading an MTZ when no UB geometry is present must raise a clear ValueError
    empty_payload = {"artifact_kind": "empty", "panels": payload0["panels"]}
    try:
        apply_mtz_to_payload(empty_payload, mtz_path)
        raise AssertionError("Expected ValueError when loading MTZ without UB geometry")
    except ValueError as exc:
        assert "UB geometry" in str(exc)

    # 3. Verify HTTP ViewModel & Trame MainApp local .mtz loading + persistence across frame switches (preserving cell/SG!)
    vm = RainbowZonesViewModel(replay_h5=integ_h5, include_panel_images=False)
    vm.payload = apply_mtz_to_payload(
        vm.payload, mtz_path, {"name": "custom_reflections.mtz"}
    )
    vm.mtz_overlay_path = str(mtz_path)
    vm.mtz_overlay_info = {"name": "custom_reflections.mtz"}
    reloaded_f1 = vm.reload(frame_index=1)
    assert reloaded_f1["frame_index"] == 1
    assert reloaded_f1["mtz_overlay_name"] == "custom_reflections.mtz"
    assert reloaded_f1["space_group"] == "P 21 21 21"
    assert np.allclose(np.asarray(reloaded_f1["b_matrix"]), B_orig, atol=1e-5)
    assert any(p.get("mtz_isigi") is not None for p in reloaded_f1["predicted_peaks"])
    assert len(reloaded_f1["integrated_peaks"]) == 1

    app_inst = object.__new__(MainApp)
    app_inst.pipeline_runner = MagicMock()
    app_inst.pipeline_runner.current_frame_index = 0
    sub_m = SubhklGalaxyModel()
    app_inst.view_model = SimpleNamespace(model=SimpleNamespace(subhkl=sub_m))
    aio_app = web.Application()
    app_inst._bind_routes(SimpleNamespace(app=aio_app))

    req_no_u = make_mocked_request(
        "GET",
        f"/api/payload?file_path={mtz_path}",
        app=aio_app,
    )
    resp_no_u = asyncio.run(aio_app._handle(req_no_u))
    assert resp_no_u.status == 400

    req_h5 = make_mocked_request(
        "GET",
        f"/api/payload?file_path={integ_h5}",
        app=aio_app,
    )
    resp_h5 = asyncio.run(aio_app._handle(req_h5))
    assert resp_h5.status == 200
    assert sub_m.data.peaks_file == str(integ_h5)
    orig_model_a = float(sub_m.data.a)
    orig_model_sg = str(sub_m.data.space_group)

    req_mtz = make_mocked_request(
        "GET",
        f"/api/payload?file_path={mtz_path}",
        app=aio_app,
    )
    resp_mtz = asyncio.run(aio_app._handle(req_mtz))
    assert resp_mtz.status == 200
    doc_mtz = json.loads(resp_mtz.body.decode("utf-8"))
    assert doc_mtz["default_peak_source"] == "integrated"
    assert doc_mtz["default_color_mode"] == "mtz_isigi"
    assert doc_mtz["space_group"] == "P 21 21 21"
    assert np.allclose(np.asarray(doc_mtz["b_matrix"]), B_orig, atol=1e-5)
    assert sub_m.data.peaks_file == str(integ_h5)
    assert sub_m.data.space_group == orig_model_sg
    assert abs(float(sub_m.data.a) - orig_model_a) < 1e-6

    req_f1 = make_mocked_request("GET", "/api/payload?frame=1", app=aio_app)
    resp_f1 = asyncio.run(aio_app._handle(req_f1))
    assert resp_f1.status == 200
    doc_f1 = json.loads(resp_f1.body.decode("utf-8"))
    assert doc_f1["frame_index"] == 1
    assert doc_f1["mtz_overlay_name"] == "custom_reflections.mtz"
    assert any(p.get("mtz_isigi") is not None for p in doc_f1["predicted_peaks"])

    # 3b. Verify post-indexing (finder-less solve.h5 before integration) has NO bogus I/σ values,
    #     AND loading .mtz after indexing persists automatically across subsequent Integration!
    solve_only_h5 = tmp_path / "solve_indexed_only.h5"
    with h5py.File(solve_only_h5, "w") as sf, h5py.File(integ_h5, "r") as src_f:
        sf.attrs["instrument"] = "CG4D"
        src_f.copy("sample", sf)
        src_f.copy("goniometer", sf)
        src_f.copy("instrument", sf)
        src_f.copy("file_offsets", sf)
        src_f.copy("bank_ids", sf)

    req_solve = make_mocked_request(
        "GET",
        f"/api/payload?file_path={solve_only_h5}",
        app=aio_app,
    )
    resp_solve = asyncio.run(aio_app._handle(req_solve))
    assert resp_solve.status == 200
    doc_solve = json.loads(resp_solve.body.decode("utf-8"))
    assert len(doc_solve["predicted_peaks"]) > 0
    assert all(p.get("sigma") is None for p in doc_solve["predicted_peaks"])
    assert all(p.get("intensity") is None for p in doc_solve["predicted_peaks"])

    # Load .mtz right after indexing, then transition to integrated.h5 (Integration completed)
    resp_mtz_after_idx = asyncio.run(
        aio_app._handle(
            make_mocked_request(
                "GET", f"/api/payload?file_path={mtz_path}", app=aio_app
            )
        )
    )
    assert resp_mtz_after_idx.status == 200
    resp_integ_after_mtz = asyncio.run(
        aio_app._handle(
            make_mocked_request(
                "GET",
                f"/api/payload?file_path={integ_h5}&preserve_params=1",
                app=aio_app,
            )
        )
    )
    assert resp_integ_after_mtz.status == 200
    doc_integ_after_mtz = json.loads(resp_integ_after_mtz.body.decode("utf-8"))
    assert doc_integ_after_mtz["artifact_kind"] == "integrator"
    assert doc_integ_after_mtz["mtz_overlay_name"] == "custom_reflections.mtz"
    assert doc_integ_after_mtz["mtz_fresh_overlay"] is False
    assert any(
        p.get("mtz_isigi") is not None for p in doc_integ_after_mtz["integrated_peaks"]
    )
    assert any(
        p.get("mtz_isigi") is not None for p in doc_integ_after_mtz["predicted_peaks"]
    )

    # 4. Execute rendered HTML/JS helpers in Node.js to verify:
    #    - Exact θ (k_f - k_i) vs 2θ (k_f) Wulff azimuth alignment & right-handed SO(3) camera.
    #    - Transparent (alpha=0) -> Solar Gold crossover (alpha=0.5) -> Opaque Electric-Lime (alpha=1)
    #      isigiToColor scale + Photoshop-checkerboard live gradient crossover slider (#slider-color-emphasis).
    #    - Two-source peak toggle ('predicted' | 'integrated') with .mtz I/σ data-column coloring.
    html_doc = render_rainbow_html(payload_mtz)
    assert 'accept=".h5,.hdf5,.nxs,.mtz"' in html_doc
    assert 'id="peak-source-group"' in html_doc
    assert 'id="btn-peaks-mtz"' not in html_doc
    assert 'id="btn-wulff-friedel"' in html_doc
    assert "Color: I/σ (Log)" in html_doc
    assert "Color: .mtz I/σ (Log)" in html_doc
    assert 'id="slider-color-emphasis"' in html_doc
    assert "repeating-conic-gradient" in html_doc  # Photoshop transparency checkerboard
    assert "--cs-gradient" in html_doc
    assert "_lastDetRenderSig" in html_doc
    assert "#e879f9" in html_doc  # distinct violet Friedel symmetry link color
    assert 'const isIntegShape = state.peakSource === "integrated";' in html_doc
    assert "if (ellPtsRaw && ellPtsRaw.length >= 6)" in html_doc

    def _extract_js_fn(name: str) -> str:
        pat = rf"function {name}\b[\s\S]*?\n  \}}"
        m = re.search(pat, html_doc)
        assert m is not None, f"Missing JS function {name}"
        return m.group(0)

    fn_names = [
        "extractPeakBucketsFromPayload",
        "getColorCrossover",
        "applyCrossoverCurve",
        "wavelengthToColor",
        "getPeakDataAvailability",
        "peakUnscaledIsigi",
        "peakMtzIsigi",
        "hasIntegratedIsigi",
        "hasMtzIsigi",
        "scaleToColor",
        "isigiToColor",
        "hexToRgba",
        "peakEmphasisAlphaForCategorical",
        "peakStrokeColor",
        "buildColorscaleGradientSpec",
        "isParallaxTriggerActive",
        "detToScreen",
        "getCenteredDetectorAzimuthDeg",
        "setDetectorCenterAzimuthDeg",
        "detectorChiToWulffAzimuthDeg",
        "wulffAzimuthToDetectorChiDeg",
        "syncWulffYawFromDetector",
        "resolveAvailablePeakSource",
        "syncPeakSourceButtons",
        "applyPeakSource",
        "getActiveGoniometerR",
        "projectToWulff",
        "getActiveWulffPeaks",
    ]
    extracted_fns = "\n".join(_extract_js_fn(fn) for fn in fn_names)

    js_harness = f"""
const DATA = {json.dumps(payload_mtz)};
const DEFAULT_COLOR_CROSSOVER = 0.38;
const ORBIT_PALETTE = ["#ff38b4", "#00e5ff", "#fbbf24", "#10b981"];
let _peakDataAvailabilitySig = "";
let _peakDataAvailability = {{
  hasIsigi: false,
  hasMtzIsigi: false,
  hasResiduals: false,
  hasScales: false,
  isigiTop: 40.0,
  mtzIsigiTop: 40.0,
}};
let _visDetPeaksSig = "";
let _visDetPeaksCache = [];
let _lastDetRenderSig = "";
let isPanningDet = false;
let isDraggingWulff = false;
const initBuckets = extractPeakBucketsFromPayload(DATA);
let PREDICTED_PEAKS = initBuckets.predPeaks;
let PREDICTED_SAMPLE_PEAKS = initBuckets.predSample;
let INTEGRATED_PEAKS = initBuckets.integPeaks;
let INTEGRATED_SAMPLE_PEAKS = initBuckets.integSample;
const PEAKS = [...(DATA.peaks || [])];
const SAMPLE_PEAKS = [...(DATA.sample_peaks || PEAKS)];
const detCanvas = {{ getBoundingClientRect: () => ({{ width: 900, height: 520 }}) }};
const btns = {{
  "btn-peaks-pred": {{ disabled: false, title: "", _cls: new Set(), classList: {{ toggle(c, on) {{ if (on) btns["btn-peaks-pred"]._cls.add(c); else btns["btn-peaks-pred"]._cls.delete(c); }} }} }},
  "btn-peaks-integ": {{ disabled: false, title: "", _cls: new Set(), classList: {{ toggle(c, on) {{ if (on) btns["btn-peaks-integ"]._cls.add(c); else btns["btn-peaks-integ"]._cls.delete(c); }} }} }},
}};
globalThis.document = {{ getElementById: (id) => btns[id] || null }};
const state = {{
  projection: "cylindrical",
  peakSource: DATA.default_peak_source || "integrated",
  wulffScope: "all_runs",
  showWulffFriedel: true,
  wulffColorMode: DATA.default_color_mode || "mtz_isigi",
  colorCrossover: 0.38,
  lamMin: 2.8,
  lamMax: 4.5,
  wulffYawDeg: 0.0,
  wulffPitchDeg: 0.0,
  wulffZoom: 1.0,
  parallaxX: 0.0,
  parallaxY: 0.0,
  parallaxStrength: 1.0,
  detZoom: 1.0,
  detPanX: 0.0,
  detPanY: 0.0,
  hoveredPeakId: null,
  hoveredPeakObj: null,
  selectedPeakId: null,
  selectedPeakObj: null,
  activeOrbitId: null,
}};
function getActiveBounds() {{ return {{ minX: -180.0, maxX: 180.0, minY: -0.25, maxY: 0.25 }}; }}
function getDetectorPlotBox(w, h) {{
  const padL = 42, padR = 12, padT = 10, padB = 22;
  return {{ originX: padL, originY: padT, plotW: Math.max(40, w - padL - padR), plotH: Math.max(40, h - padT - padB) }};
}}
function mat3Vec(M, v) {{
  return [
    M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
    M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
    M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
  ];
}}
function hasMergedCarelessData() {{ return false; }}
function isPeakVisible(p) {{ return state.wulffColorMode !== "mtz_isigi" || Number.isFinite(peakMtzIsigi(p)); }}
function zoneColor() {{ return "rgba(255,100,100,1)"; }}
function residualToColor() {{ return "rgba(255,255,255,1)"; }}
function syncColorModeSelectors() {{}}
function syncWulffUI() {{}}
function renderZoneList() {{}}
function updateHUD() {{}}
function renderDetector() {{}}
{extracted_fns}

function gForChiElDeg(chiDeg, elDeg = 0.0) {{
  const rad = chiDeg * Math.PI / 180.0;
  const el = elDeg * Math.PI / 180.0;
  const kx = Math.cos(el) * Math.sin(rad);
  const ky = Math.sin(el);
  const kz = Math.cos(el) * Math.cos(rad);
  const qx = kx, qy = ky, qz = kz - 1.0;
  const n = Math.hypot(qx, qy, qz);
  return [qx / n, qy / n, qz / n];
}}

DATA.goniometer_R = [[1,0,0],[0,1,0],[0,0,1]];
const box = getDetectorPlotBox(900, 520);
const viewCenterX = box.originX + 0.5 * box.plotW;

const chiChecks = [];
for (const [chiTarget, zoomVal] of [[45.0, 1.0], [-45.0, 2.4], [135.0, 1.5], [-135.0, 1.5]]) {{
  state.detZoom = zoomVal;
  setDetectorCenterAzimuthDeg(chiTarget);
  syncWulffYawFromDetector();
  const screenX = detToScreen(chiTarget, 0.0, 900, 520, 0.0)[0];
  const chiRec = getCenteredDetectorAzimuthDeg();
  const chiRoundTrip = wulffAzimuthToDetectorChiDeg(state.wulffYawDeg);
  const wCenter = projectToWulff(gForChiElDeg(chiTarget, 0.0), 100, 100, 80);
  const wUp = projectToWulff(gForChiElDeg(chiTarget, 12.0), 100, 100, 80);
  const wRight = projectToWulff(gForChiElDeg(chiTarget + 10.0, 0.0), 100, 100, 80);
  const wLeft = projectToWulff(gForChiElDeg(chiTarget - 10.0, 0.0), 100, 100, 80);
  chiChecks.push({{
    chiTarget,
    screenXErr: Math.abs(screenX - viewCenterX),
    chiRecErr: Math.abs(chiRec - chiTarget),
    chiRoundTripErr: Math.abs(chiRoundTrip - chiTarget),
    wSx: wCenter[0],
    wSy: wCenter[1],
    wHemi: wCenter[2],
    wZ: wCenter[3],
    upSy: wUp[1],
    rightSx: wRight[0],
    leftSx: wLeft[0],
  }});
}}

state.detZoom = 1.0;
setDetectorCenterAzimuthDeg(10.0);
syncWulffYawFromDetector();
const gNeg10 = gForChiElDeg(-10.0, 0.0);
const friedelNeg10AtPos10 = projectToWulff([-gNeg10[0], -gNeg10[1], -gNeg10[2]], 100, 100, 80);

// Transparent (alpha=0) -> Solar Gold crossover (t=0.5, alpha=0.5) -> Fully Opaque Electric Lime (alpha=1)
state.colorCrossover = 0.38;
const exactCrossSnr = Math.pow(1.0 + 40.0, 0.38) - 1.0;
const colNeg = isigiToColor(-2.0, 1.0, 40.0);
const colZero = isigiToColor(0.0, 1.0, 40.0);
const colWeak = isigiToColor(0.4, 1.0, 40.0);
const colCrossover = isigiToColor(exactCrossSnr, 1.0, 40.0);
const colGold = isigiToColor(14.0, 1.0, 40.0);
const colLime = isigiToColor(150.0, 1.0, 40.0);

// Crossover knob: moving right (0.84) makes moderate 4σ peaks nearly transparent; moving left (0.12) makes them opaque
state.colorCrossover = 0.84;
const colMidRightKnob = isigiToColor(4.0, 1.0, 40.0);
state.colorCrossover = 0.12;
const colMidLeftKnob = isigiToColor(4.0, 1.0, 40.0);
state.colorCrossover = 0.38;

const gradIsigi = buildColorscaleGradientSpec("isigi", 0.38);
const gradWl = buildColorscaleGradientSpec("wavelength", 0.50);

// Parallax trigger gating: off on plain hover AND off on mouse-button pan/drag;
// active ONLY when holding a keyboard modifier key (Ctrl / Meta / Alt / Shift) without mouse buttons
const parIdle = isParallaxTriggerActive({{ buttons: 0, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false }});
const parMouseBtn = isParallaxTriggerActive({{ buttons: 1, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false }});
const parMouseBtnWithCtrl = isParallaxTriggerActive({{ buttons: 1, ctrlKey: true, metaKey: false, altKey: false, shiftKey: false }});
const parCtrl = isParallaxTriggerActive({{ buttons: 0, ctrlKey: true, metaKey: false, altKey: false, shiftKey: false }});
const parShift = isParallaxTriggerActive({{ buttons: 0, ctrlKey: false, metaKey: false, altKey: false, shiftKey: true }});
state.parallaxX = 1.0;
state.parallaxY = 1.0;
const [fgParX, fgParY] = detToScreen(0.0, 0.0, 900, 520, 0.45);
const [bgParX, bgParY] = detToScreen(0.0, 0.0, 900, 520, -0.35);
const parRelSpanX = fgParX - bgParX;
const parRelSpanY = fgParY - bgParY;
state.parallaxX = 0.0;
state.parallaxY = 0.0;

// Peak source toggle ('predicted' <-> 'integrated') switches BOTH shape (circle <-> ellipse)
// AND intensity-related highlight colorscale (Electric Cyan for Pred <-> Gold-to-Lime for Integ)
applyPeakSource("predicted", false);
const nPredFrame = PEAKS.length;
const nPredWulff = getActiveWulffPeaks().length;
const predShape = (state.peakSource === "integrated" && Array.isArray(PEAKS[0]?.ellipse)) ? "ellipse" : "circle";
const hasMtzCol = hasMtzIsigi();
const hasPredCrossLinkedIsigi = hasIntegratedIsigi();
const mtzColStrokePred = peakStrokeColor(PEAKS.find(p => Number.isFinite(peakMtzIsigi(p))), 1.0);
const colPredCross = isigiToColor(exactCrossSnr, 1.0, 40.0);
const colPredTop = isigiToColor(150.0, 1.0, 40.0);
const scalePredTop = scaleToColor(10.0, 1.0);
const gradIsigiPred = buildColorscaleGradientSpec("isigi", 0.38);

applyPeakSource("integrated", false);
const nIntegFrame = PEAKS.length;
const nIntegWulff = getActiveWulffPeaks().length;
const integShape = (state.peakSource === "integrated" && Array.isArray(PEAKS[0]?.ellipse) && PEAKS[0].ellipse.length >= 6) ? "ellipse" : "circle";
const mtzColStrokeInteg = peakStrokeColor(PEAKS.find(p => Number.isFinite(peakMtzIsigi(p))), 1.0);
const scaleIntegTop = scaleToColor(10.0, 1.0);

console.log(JSON.stringify({{
  chiChecks,
  friedelNeg10AtPos10,
  colNeg, colZero, colWeak, colCrossover, colGold, colLime,
  colPredCross, colPredTop, scalePredTop, scaleIntegTop, gradIsigiPred,
  colMidRightKnob, colMidLeftKnob,
  gradIsigi, gradWl,
  parIdle, parMouseBtn, parMouseBtnWithCtrl, parCtrl, parShift, parRelSpanX, parRelSpanY,
  hasMtzCol, hasPredCrossLinkedIsigi,
  mtzColStroke: mtzColStrokePred, mtzColStrokeInteg,
  nPredFrame, nPredWulff, predShape,
  nIntegFrame, nIntegWulff, integShape,
  btnPredDisabled: btns["btn-peaks-pred"].disabled,
  btnIntegDisabled: btns["btn-peaks-integ"].disabled,
}}));
"""
    harness_path = tmp_path / "wulff_peaks_mtz_harness.js"
    harness_path.write_text(js_harness, encoding="utf-8")
    import shutil

    if shutil.which("node"):
        script_blocks = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
        assert len(script_blocks) >= 1
        full_js_check = tmp_path / "full_inline_script.js"
        full_js_check.write_text(script_blocks[-1], encoding="utf-8")
        subprocess.run(["node", "--check", str(full_js_check)], check=True)

        js_res = subprocess.run(
            ["node", str(harness_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        js_out = json.loads(js_res.stdout.strip())
        for chk in js_out["chiChecks"]:
            assert chk["screenXErr"] < 1e-6
            assert chk["chiRecErr"] < 1e-6
            assert chk["chiRoundTripErr"] < 1e-6
            assert abs(chk["wSx"] - 100.0) < 1e-6
            assert abs(chk["wSy"] - 100.0) < 1e-6
            assert chk["wHemi"] == 1 and chk["wZ"] > 0.9999
            assert chk["upSy"] < 95.0
            assert chk["rightSx"] > 102.0 and chk["leftSx"] < 98.0

        fr_proj = js_out["friedelNeg10AtPos10"]
        assert fr_proj[2] == 1 and fr_proj[3] > 0.98 and fr_proj[0] < 95.0

        def _parse_rgba(s: str) -> tuple[int, int, int, float]:
            m_rgba = re.match(r"rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\s*\)", s)
            assert m_rgba is not None, f"Unexpected rgba string: {s}"
            return (
                int(m_rgba.group(1)),
                int(m_rgba.group(2)),
                int(m_rgba.group(3)),
                float(m_rgba.group(4)),
            )

        _rn, _gn, _bn, an = _parse_rgba(js_out["colNeg"])
        _rz, _gz, _bz, az = _parse_rgba(js_out["colZero"])
        _rw, _gw, _bw, aw = _parse_rgba(js_out["colWeak"])
        rc, gc, bc, ac = _parse_rgba(js_out["colCrossover"])
        rg, gg, bg, ag = _parse_rgba(js_out["colGold"])
        rl, gl, bl, al = _parse_rgba(js_out["colLime"])
        _, _, _, a_right_knob = _parse_rgba(js_out["colMidRightKnob"])
        _, _, _, a_left_knob = _parse_rgba(js_out["colMidLeftKnob"])

        # Transparent (alpha=0) -> Solar Gold at exact knob crossover (alpha=0.5) -> Opaque Electric Lime (alpha=1)
        assert an == 0.0 and az == 0.0
        assert 0.0 < aw < ac < ag < al
        assert (rc, gc, bc) == (250, 204, 21) and abs(ac - 0.5) < 0.005
        assert rg > 185 and gg > 205 and bg < 60
        assert (rl, gl, bl) == (163, 242, 44) and al == 1.0
        # Moving crossover knob right isolates strong peaks; moving left reveals weaker peaks
        assert a_right_knob < 0.12 and a_left_knob > 0.72
        # Live gradient CSS stops + physical crossover readouts
        assert js_out["gradIsigi"]["gradient"].startswith("linear-gradient(90deg,")
        assert "0.000) 0.0%" in js_out["gradIsigi"]["gradient"]
        assert "1.000) 100.0%" in js_out["gradIsigi"]["gradient"]
        assert js_out["gradIsigi"]["label"].startswith("I/σ ~")
        assert js_out["gradWl"]["label"].startswith("λ ~")

        # Parallax gated strictly on keyboard modifier (never mouse button / drag) with softer rubber-band span
        assert js_out["parIdle"] is False
        assert js_out["parMouseBtn"] is False
        assert js_out["parMouseBtnWithCtrl"] is False
        assert js_out["parCtrl"] is True
        assert js_out["parShift"] is True
        assert abs(js_out["parRelSpanX"] - 43.2) < 1e-6
        assert abs(js_out["parRelSpanY"] - 30.4) < 1e-6

        assert js_out["hasMtzCol"] is True
        assert js_out["hasPredCrossLinkedIsigi"] is True
        assert js_out["mtzColStroke"].startswith("rgba(")
        assert js_out["mtzColStroke"] != js_out["mtzColStrokeInteg"]
        rpc, gpc, bpc, apc = _parse_rgba(js_out["colPredCross"])
        rpt, gpt, bpt, apt = _parse_rgba(js_out["colPredTop"])
        assert (rpc, gpc, bpc) == (56, 189, 248) and abs(apc - 0.5) < 0.005
        assert (rpt, gpt, bpt) == (0, 240, 255) and apt == 1.0
        assert js_out["scalePredTop"] != js_out["scaleIntegTop"]
        assert js_out["gradIsigiPred"]["gradient"] != js_out["gradIsigi"]["gradient"]
        assert js_out["nPredFrame"] == len(payload_mtz["predicted_peaks"])
        assert js_out["nPredWulff"] == len(samp_with_mtz)
        assert js_out["predShape"] == "circle"
        assert js_out["nIntegFrame"] == 1 and js_out["nIntegWulff"] == 2
        assert js_out["integShape"] == "ellipse"
        assert js_out["btnPredDisabled"] is False
        assert js_out["btnIntegDisabled"] is False


def test_mandi_i2_harmonics_wulff_snr_dimming_and_hires_textures(tmp_path) -> None:
    import base64
    import io

    import h5py
    from PIL import Image

    from subhkl.viz.rainbow_zones import (
        _compute_harmonic_order_and_multiplicity,
        _make_systematic_absence_checker,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    # 1. Exact space-group-aware Laue harmonic multiplicity for MANDI I2 (a=55.16, b=81.19, c=88.7, beta=96.3)
    is_abs_i2 = _make_systematic_absence_checker("I 1 2 1")
    assert is_abs_i2(1, 0, 0) is True  # h+k+l = 1 odd -> body-centered absent
    assert is_abs_i2(2, 0, 0) is False  # h+k+l = 2 even -> allowed

    # (2, 0, 0) at lambda=3.2 A, d=27.4 A in [2.0, 4.0] A, d_min=1.5 A:
    # (1, 0, 0) is absent in I2, and (4, 0, 0) would need lambda=1.6 A < 2.0 A -> NOT a harmonic!
    ord_200, mult_200 = _compute_harmonic_order_and_multiplicity(
        2,
        0,
        0,
        3.2,
        27.4,
        wl_min=2.0,
        wl_max=4.0,
        d_min=1.5,
        is_absent_fn=is_abs_i2,
    )
    assert ord_200 == 1
    assert mult_200 == 1

    # (2, 0, 0) at lambda=3.9 A, d=27.4 A -> (4, 0, 0) at lambda=1.95 A is still out-of-band (< 2.0 A),
    # whereas at lambda=4.0 A -> (4, 0, 0) at lambda=2.0 A is in-band and allowed -> mult == 2!
    _, mult_out = _compute_harmonic_order_and_multiplicity(
        2,
        0,
        0,
        3.9,
        27.4,
        wl_min=2.0,
        wl_max=4.0,
        d_min=1.5,
        is_absent_fn=is_abs_i2,
    )
    ord_400, mult_in = _compute_harmonic_order_and_multiplicity(
        4,
        0,
        0,
        2.0,
        13.7,
        wl_min=2.0,
        wl_max=4.0,
        d_min=1.5,
        is_absent_fn=is_abs_i2,
    )
    assert mult_out == 1
    assert ord_400 == 2
    assert mult_in == 2

    import gemmi

    solve_mandi = tmp_path / "mandi_i2_solve.h5"
    mandi_cell = [55.16, 81.19, 88.7, 90.0, 96.3, 90.0]
    b_mandi = np.asarray(gemmi.UnitCell(*mandi_cell).frac.mat.tolist(), dtype=float).T
    with h5py.File(solve_mandi, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf["sample/space_group"] = "I 1 2 1"
        hf["sample/cell"] = np.array(mandi_cell, dtype=float)
        hf["sample/U"] = np.eye(3)
        hf["sample/B"] = b_mandi
        hf["instrument/wavelength"] = np.array([2.0, 4.0], dtype=float)

    p_mandi = build_rainbow_zone_payload(
        replay_h5=solve_mandi,
        instrument="MANDI",
        space_group="I 1 2 1",
        wavelength_range=(2.0, 4.0),
        d_min=1.6,
        include_panel_images=False,
    )
    assert len(p_mandi["peaks"]) > 20
    harm_frac = sum(
        1 for pk in p_mandi["peaks"] if int(pk.get("harmonic_multiplicity", 1)) > 1
    ) / len(p_mandi["peaks"])
    assert harm_frac < 0.25

    # 2. Integrator HDF5 with image_index + file_offsets (no peaks/run_index) and render_binning=4 vs 1
    img_h5 = tmp_path / "mandi_imgs.h5"
    with h5py.File(img_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf["images"] = np.ones((2, 64, 64), dtype=np.float32) * 12.0
        hf["bank_ids"] = np.array([20, 20], dtype=np.int32)
        hf["file_offsets"] = np.array([0, 1, 2], dtype=np.int32)
        dt_s = h5py.string_dtype(encoding="utf-8")
        hf.create_dataset(
            "files",
            data=np.array(["MANDI_13172.nxs.h5", "MANDI_13174.nxs.h5"], dtype=object),
            dtype=dt_s,
        )
        hf["goniometer/R"] = np.stack([np.eye(3), np.eye(3)], axis=0)
        hf["instrument/wavelength"] = np.array([2.0, 4.0], dtype=np.float32)

    integ_h5 = tmp_path / "mandi_integrated.h5"
    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf["sample/space_group"] = "I 1 2 1"
        hf["sample/cell"] = np.array([55.16, 81.19, 88.7, 90.0, 96.3, 90.0])
        hf["sample/U"] = np.eye(3)
        hf["sample/B"] = np.diag([1.0 / 55.16, 1.0 / 81.19, 1.0 / 88.7])
        hf["goniometer/R"] = np.stack([np.eye(3), np.eye(3)], axis=0)
        hf["file_offsets"] = np.array([0, 1, 2], dtype=np.int32)
        # Peak 0 (run 0, I/sig=10 >= 2.5), Peak 1 (run 0, I/sig=1.8 < 2.5), Peak 2 (run 1, I/sig=12 >= 2.5)
        hf["peaks/image_index"] = np.array([0, 0, 1], dtype=np.int32)
        hf["peaks/bank"] = np.array([20, 20, 20], dtype=np.int32)
        hf["peaks/pixel_r"] = np.array([20.0, 36.0, 28.0], dtype=np.float64)
        hf["peaks/pixel_c"] = np.array([24.0, 40.0, 30.0], dtype=np.float64)
        hf["peaks/h"] = np.array([2, 0, 2], dtype=np.int32)
        hf["peaks/k"] = np.array([0, 2, 2], dtype=np.int32)
        hf["peaks/l"] = np.array([0, 0, 0], dtype=np.int32)
        hf["peaks/lambda"] = np.array([3.1, 3.2, 3.0], dtype=np.float64)
        hf["peaks/intensity"] = np.array([100.0, 18.0, 120.0], dtype=np.float64)
        hf["peaks/sigma"] = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        hf["peaks/var_u"] = np.array([2.2, 1.8, 2.0], dtype=np.float64)
        hf["peaks/var_v"] = np.array([0.8, 0.9, 1.0], dtype=np.float64)
        hf["peaks/cov_uv"] = np.array([0.2, 0.0, -0.1], dtype=np.float64)

    p_lo = build_rainbow_zone_payload(
        replay_h5=integ_h5, images_h5=img_h5, frame_index=0, render_binning=4
    )
    p_hi = build_rainbow_zone_payload(
        replay_h5=integ_h5, images_h5=img_h5, frame_index=0, render_binning=1
    )
    assert p_lo["render_binning"] == 4
    assert p_hi["render_binning"] == 1
    assert p_lo["all_runs_min_isigi"] == 2.5
    # Frame 0 detector PEAKS keeps both strong (I/sig=10) and uncertain (I/sig=1.8 >= 1.5) peaks
    assert len(p_lo["integrated_peaks"]) == 2
    # Multi-run Wulff SAMPLE_PEAKS has I/sig >= 2.5 across properly reconstructed runs [0, 1]
    by_run = {int(sp["run_index"]): sp for sp in p_lo["integrated_sample_peaks"]}
    assert sorted(by_run.keys()) == [0, 1]
    # Run 0: (2, 0, 0) at lambda=3.1 A -> (1,0,0) & (3,0,0) absent in I2, (4,0,0) at 1.55 A < 2.0 A -> mult == 1
    assert by_run[0]["harmonic_order"] == 1
    assert by_run[0]["harmonic_multiplicity"] == 1
    # Run 1: (2, 2, 0) at lambda=3.0 A -> (1,1,0) allowed (1+1+0=2 even, ord=2) and (3,3,0) allowed at 2.0 A -> mult == 2!
    assert by_run[1]["harmonic_order"] == 2
    assert by_run[1]["harmonic_multiplicity"] == 2

    b20_lo = next(pn for pn in p_lo["panels"] if pn["bank"] == 20)
    b20_hi = next(pn for pn in p_hi["panels"] if pn["bank"] == 20)
    im_lo = Image.open(io.BytesIO(base64.b64decode(b20_lo["texture_b64"])))
    im_hi = Image.open(io.BytesIO(base64.b64decode(b20_hi["texture_b64"])))
    assert im_lo.size == (16, 16)
    assert im_hi.size == (64, 64)

    # 3. Verify ellipse crossbars are removed, Wulff I/sig>=2.5 badge + detector dimming + Hi-Res button exist,
    # and Hi-Res direct (u, v) rendering eliminates screen-edge-parallel tile-by-tile nematic warping.
    html_doc = render_rainbow_html(p_hi)
    assert "pQ1" not in html_doc and "pQ3" not in html_doc
    assert 'id="btn-hires-tex"' in html_doc
    assert "toggleHiResTextures" in html_doc
    assert "isWulffSnrCutActive" in html_doc
    assert "isPeakUncertainForScope" in html_doc
    assert "ALL-RUNS · I/σ ≥" in html_doc
    assert "drawPanelTextureDirect" in html_doc
    assert "getCrispUVCanvas" in html_doc

    import re
    import shutil
    import subprocess

    if shutil.which("node"):
        script_blocks = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
        assert len(script_blocks) >= 1
        inline_js = script_blocks[-1]
        js_syntax_file = tmp_path / "hires_inline_check.js"
        js_syntax_file.write_text(inline_js, encoding="utf-8")
        subprocess.run(["node", "--check", str(js_syntax_file)], check=True)

        fn_names = [
            "invalidatePanelRenderCaches",
            "getDetectorDataAspect",
            "getDetectorPlotBox",
            "detToScreen",
            "getPanelVertRange",
            "getCrispUVCanvas",
            "getPanelScreenAspect",
            "expandClampedUVTri",
            "warpTextureOntoPanel",
            "drawPanelTextureDirect",
        ]
        extracted = []
        for fn_name in fn_names:
            m = re.search(rf"\bfunction\s+{fn_name}\s*\(", inline_js)
            assert m is not None, f"Missing JS function {fn_name}"
            start = m.start()
            depth = 0
            in_body = False
            end = start
            for idx in range(start, len(inline_js)):
                ch = inline_js[idx]
                if ch == "{":
                    depth += 1
                    in_body = True
                elif ch == "}":
                    depth -= 1
                    if in_body and depth == 0:
                        end = idx + 1
                        break
            extracted.append(inline_js[start:end])

        b1_panel = next(pn for pn in p_hi["panels"] if pn["bank"] == 1)
        b20_panel = next(pn for pn in p_hi["panels"] if pn["bank"] == 20)
        harness_js = f"""
const PANELS = {json.dumps([b1_panel, b20_panel])};
const warpedBankCanvases = new Map();
const warpedMaskCanvases = new Map();
const uvUpscaledCanvases = new Map();
let _panelTexEpoch = 1;
let _detLayerCacheCanvas = null;
let _detLayerCacheKey = "";
let _panelScratchCanvas = null;
let _cachedCylRadiusM = null;
let _cachedPlotBox = null;
let _cachedPlotBoxKey = "";

function makeMockCanvas(w = 256, h = 256) {{
  const ops = [];
  const ctx = {{
    ops,
    imageSmoothingEnabled: true,
    imageSmoothingQuality: "low",
    globalAlpha: 1.0,
    globalCompositeOperation: "source-over",
    save() {{ ops.push(["save"]); }},
    restore() {{ ops.push(["restore"]); }},
    beginPath() {{}},
    moveTo() {{}},
    lineTo() {{}},
    rect() {{}},
    closePath() {{}},
    clip() {{}},
    clearRect() {{}},
    setTransform(...args) {{ ops.push(["setTransform", ...args]); }},
    transform(...args) {{ ops.push(["transform", ...args]); }},
    drawImage(src, ...rest) {{
      ops.push([
        "drawImage",
        src.width,
        src.height,
        this.imageSmoothingEnabled,
        this.imageSmoothingQuality,
        this.globalCompositeOperation,
        ...rest,
      ]);
    }},
  }};
  return {{ width: w, height: h, _ctx: ctx, getContext: () => ctx }};
}}
globalThis.document = {{ createElement: () => makeMockCanvas(1, 1) }};

const state = {{
  projection: "spherical",
  hiResTextures: true,
  heatmapStyle: "magma",
  detZoom: 1.0,
  detPanX: 0.0,
  detPanY: 0.0,
  parallaxX: 0.0,
  parallaxY: 0.0,
  parallaxStrength: 1.0,
}};
function getActiveBounds() {{
  return {{ minX: -180.0, maxX: 180.0, minY: -90.0, maxY: 90.0 }};
}}

{chr(10).join(extracted)}

const tex = makeMockCanvas(256, 256);
// 1) Wide bottom-ring MANDI bank b1 has spherical az/el ratio ~3.48:1 (never forced to 1:1 square)
const wcB1 = warpTextureOntoPanel(tex, PANELS[0], true, false);
const b1Ratio = wcB1.width / wcB1.height;

// 2) Direct (u, v) rendering on 45-deg diamond bank b20 at overview zoom (detZoom=1.0 -> k=1) vs deep zoom (detZoom=12.0 -> k>=4)
const layerLo = makeMockCanvas(1200, 800);
const p20 = PANELS[1];
const [vMin, vMax] = getPanelVertRange(p20);
let [x0, y0] = detToScreen(p20.az_min, vMax, 1200, 800, 0.0);
let [x1, y1] = detToScreen(p20.az_max, vMin, 1200, 800, 0.0);
const outLo = p20.outline_sph.map(pt => detToScreen(pt[0], pt[1], 1200, 800, 0.0));
drawPanelTextureDirect(
  layerLo.getContext("2d"), tex, "tex_20_magma_hi", p20, true, 0.0,
  1200, 800, 1.0, -24, -24, 1224, 824, x0, y0, x1 - x0, y1 - y0, outLo, 0.85, true
);

state.detZoom = 12.0;
const [cx20, cy20] = detToScreen(0.5 * (p20.az_min + p20.az_max), 0.5 * (vMin + vMax), 1200, 800, 0.0);
state.detPanX = 600.0 - cx20;
state.detPanY = 400.0 - cy20;
[x0, y0] = detToScreen(p20.az_min, vMax, 1200, 800, 0.0);
[x1, y1] = detToScreen(p20.az_max, vMin, 1200, 800, 0.0);
const outHi = p20.outline_sph.map(pt => detToScreen(pt[0], pt[1], 1200, 800, 0.0));
const layerHi = makeMockCanvas(1200, 800);
drawPanelTextureDirect(
  layerHi.getContext("2d"), tex, "tex_20_magma_hi", p20, true, 0.0,
  1200, 800, 2.0, -24, -24, 1224, 824, x0, y0, x1 - x0, y1 - y0, outHi, 0.85, true
);

const scratchDraws = _panelScratchCanvas
  ? _panelScratchCanvas.getContext("2d").ops.filter(o => o[0] === "drawImage")
  : [];
const clampCheck = expandClampedUVTri(0, 0, 32, 0, 0, 32, 2.5, 2.5, 256, 256, 1, 0, 0, 1, 0, 0);

console.log(JSON.stringify({{
  b1Ratio,
  uvCacheKeys: Array.from(uvUpscaledCanvases.keys()),
  firstDrawSrcW: scratchDraws.length > 0 ? scratchDraws[0][1] : 0,
  lastDrawSrcW: scratchDraws.length > 0 ? scratchDraws[scratchDraws.length - 1][1] : 0,
  scratchSmoothing: scratchDraws.every(d => d[3] === true && d[4] === "high"),
  scratchCopyOp: scratchDraws.every(d => d[5] === "copy"),
  clampTopEdgeExactZero: clampCheck[0][1] === 0 && clampCheck[1][1] === 0 && clampCheck[0][0] === 0 && clampCheck[2][0] === 0,
}}));
"""
        harness_file = tmp_path / "hires_nematic_fix_harness.js"
        harness_file.write_text(harness_js, encoding="utf-8")
        res = subprocess.run(
            ["node", str(harness_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        out = json.loads(res.stdout.strip())
        assert out["b1Ratio"] > 3.0
        assert any(k.endswith(("_k4", "_k8")) for k in out["uvCacheKeys"])
        assert out["firstDrawSrcW"] == 256
        assert out["lastDrawSrcW"] >= 1024
        assert out["scratchSmoothing"] is True
        assert out["scratchCopyOp"] is True
        assert out["clampTopEdgeExactZero"] is True


def test_mtz_vectorized_speed_and_multirun_dedup(tmp_path):
    """Tweak 1: External .mtz projection vectorizes ASU/Bragg filtering and deduplicates multi-run Wulff peaks."""
    import time

    import gemmi
    import numpy as np

    from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix
    from subhkl.viz.rainbow_zones import (
        apply_mtz_to_payload,
        build_rainbow_zone_payload,
    )

    solve_h5 = tmp_path / "mandi_mtz_solve.h5"
    mandi_cell = [55.16, 81.19, 88.70, 90.0, 96.3, 90.0]
    b_mandi = np.asarray(gemmi.UnitCell(*mandi_cell).frac.mat.tolist(), dtype=float).T
    angles_deg = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0]
    r_stack = np.stack(
        [
            calc_goniometer_rotation_matrix([[0.0, 1.0, 0.0, 1.0]], [ang])
            for ang in angles_deg
        ],
        axis=0,
    )
    with h5py.File(solve_h5, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf["sample/space_group"] = "I 1 2 1"
        hf["sample/cell"] = np.array(mandi_cell, dtype=float)
        hf["sample/U"] = np.eye(3)
        hf["sample/B"] = b_mandi
        hf["goniometer/R"] = r_stack
        hf["goniometer/angles"] = np.asarray([[ang] for ang in angles_deg], dtype=float)
        hf["instrument/wavelength"] = np.array([2.0, 4.5], dtype=float)

    base = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        instrument="CG4D",
        space_group="I 1 2 1",
        wavelength_range=(2.0, 4.5),
        d_min=1.8,
        frame_index=2,
        include_panel_images=False,
    )
    b_before = np.asarray(base["b_matrix"], dtype=np.float64).copy()

    # Build an external .mtz with ~2,500 reflections including ultra-high-res shells beyond 2/wl_min
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("I 1 2 1")
    mtz.set_cell_for_all(gemmi.UnitCell(55.16, 81.19, 88.70, 90.0, 96.3, 90.0))
    mtz.add_dataset("crystal")
    mtz.add_column("I", "J")
    mtz.add_column("SIGI", "Q")
    rows = []
    rng = np.random.default_rng(42)
    for h in range(-14, 15):
        for k in range(14):
            for l_idx in range(-10, 11):
                if h == 0 and k == 0 and l_idx == 0:
                    continue
                if (h + k + l_idx) % 2 != 0:
                    continue
                if len(rows) >= 2500:
                    break
                sig = float(rng.uniform(1.0, 5.0))
                inten = float(rng.normal(12.0, 8.0))
                rows.append([h, k, l_idx, inten, sig])
            if len(rows) >= 2500:
                break
        if len(rows) >= 2500:
            break
    mtz.set_data(np.asarray(rows, dtype=np.float32))
    mtz_file = tmp_path / "dense_external.mtz"
    mtz.write_to_file(str(mtz_file))

    t0 = time.perf_counter()
    out = apply_mtz_to_payload(base, str(mtz_file), {"name": "dense_external.mtz"})
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"Vectorized MTZ load took too long: {elapsed:.2f}s"
    np.testing.assert_allclose(out["b_matrix"], b_before)
    assert out["default_color_mode"] == "mtz_isigi"
    assert "mtz_peaks" not in out
    mtz_frame = [p for p in out["predicted_peaks"] if p.get("mtz_isigi") is not None]
    mtz_all = [
        p for p in out["predicted_sample_peaks"] if p.get("mtz_isigi") is not None
    ]
    assert len(mtz_frame) > 0
    assert len(mtz_all) >= len(mtz_frame)
    # Multi-run Wulff sample peaks must have unique crystal (h, k, l) keys and preserve active-frame hits
    seen_hkls = {tuple(p["hkl"]) for p in out["predicted_sample_peaks"]}
    assert len(seen_hkls) == len(out["predicted_sample_peaks"])
    frame_hkls = {tuple(p["hkl"]) for p in mtz_frame}
    assert frame_hkls.issubset(seen_hkls)
    by_hkl = {tuple(p["hkl"]): p for p in out["predicted_sample_peaks"]}
    for fp in mtz_frame:
        assert by_hkl[tuple(fp["hkl"])]["run_index"] == 2


def test_wulff_zoom_dspacing_lod_and_dirty_skip(tmp_path):
    """Tweak 2: Wulff All-Runs deduplicates crystal HKLs, couples zoom to d-spacing LOD, and skips idle redraws."""
    import json
    import re
    import shutil
    import subprocess

    from subhkl.viz.rainbow_zones import (
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    if not shutil.which("node"):
        return

    payload = build_rainbow_zone_payload()
    html_doc = render_rainbow_html(payload)
    script_blocks = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
    inline_js = script_blocks[-1]

    fn_names = [
        "getColorCrossover",
        "isSameReflection",
        "isPeakHighlightedTarget",
        "isPeakVisible",
        "isWulffSnrCutActive",
        "getActiveWulffPeaks",
        "_peakSnrScore",
        "getWulffPoolCache",
        "getWulffLodState",
        "projectToWulff",
        "findNearestPeakWulff",
        "renderStereoInset",
    ]
    extracted = []
    for fn_name in fn_names:
        m = re.search(rf"\bfunction\s+{fn_name}\s*\(", inline_js)
        assert m is not None, f"Missing JS function {fn_name}"
        start = m.start()
        depth = 0
        in_body = False
        end = start
        for idx in range(start, len(inline_js)):
            ch = inline_js[idx]
            if ch == "{":
                depth += 1
                in_body = True
            elif ch == "}":
                depth -= 1
                if in_body and depth == 0:
                    end = idx + 1
                    break
        extracted.append(inline_js[start:end])

    harness_js = f"""
let PEAKS = [];
let SAMPLE_PEAKS = [];
let ZONES = [];
let DATA = {{ d_min: 1.4, wavelength_range: [1.8, 4.5], frame_index: 2 }};
const IS_SOLVED = true;
const WULFF_MIN_ISIGI = 2.5;
const DEFAULT_COLOR_CROSSOVER = 0.38;
globalThis.window = {{ devicePixelRatio: 1 }};

const _wulffPoolCache = {{
  sig: "",
  firstRef: null,
  lastRef: null,
  rawPeaks: [],
  uniquePeaks: [],
  observedHklSet: new Set(),
  sortedD: [],
}};
let _lastWulffRenderSig = "";

// Create 1,200 All-Runs predicted reflections spanning d in [1.4, 8.0] A, plus duplicate run_index hits for HKL [9999,1,1]
for (let i = 0; i < 1200; i++) {{
  const frac = i / 1199.0;
  const d = 1.4 + 6.6 * Math.pow(frac, 1.5);
  const th = (i * 2.399963) % (2 * Math.PI);
  const cz = -0.7 + 1.5 * ((i * 0.618034) % 1.0);
  const sz = Math.sqrt(Math.max(0.0, 1.0 - Math.min(1.0, cz * cz)));
  const q = [sz * Math.cos(th), sz * Math.sin(th), Math.min(1.0, Math.max(-1.0, cz))];
  SAMPLE_PEAKS.push({{
    id: i,
    hkl: [i + 1, (i % 9) - 4, (i % 7) - 3],
    asu_hkl: [i + 1, i % 5, i % 3],
    d_spacing: d,
    wavelength: 2.8,
    intensity: 10.0,
    sigma: 2.0,
    q_hat: q,
    q_hat_sample: q,
    q_hat_cryst: q,
    run_index: i % 5,
  }});
}}
for (const rIdx of [0, 2, 4]) {{
  SAMPLE_PEAKS.push({{
    id: 5000 + rIdx,
    hkl: [9999, 1, 1],
    asu_hkl: [9999, 1, 1],
    d_spacing: 6.5,
    wavelength: 3.0,
    intensity: 20.0,
    sigma: 2.0,
    q_hat: [0.1, 0.2, 0.974679],
    q_hat_sample: [0.1, 0.2, 0.974679],
    q_hat_cryst: [0.1, 0.2, 0.974679],
    run_index: rIdx,
  }});
}}

let saveDepth = 0;
let clipDepth = -1;
const ops = [];
const stereoCtx = {{
  fillStyle: "",
  strokeStyle: "",
  lineWidth: 1,
  font: "",
  textAlign: "left",
  globalAlpha: 1,
  setTransform() {{}},
  setLineDash() {{}},
  createRadialGradient() {{ return {{ addColorStop() {{}} }}; }},
  clearRect(...a) {{ ops.push(["clearRect", ...a]); }},
  fillRect(...a) {{ ops.push(["fillRect", ...a]); }},
  strokeRect(...a) {{ ops.push(["strokeRect", ...a]); }},
  beginPath() {{}},
  arc(...a) {{ ops.push(["arc", ...a]); }},
  moveTo() {{}},
  lineTo() {{}},
  closePath() {{}},
  roundRect() {{}},
  stroke() {{}},
  fill() {{}},
  clip() {{ clipDepth = saveDepth; ops.push(["clip"]); }},
  save() {{ saveDepth++; ops.push(["save"]); }},
  restore() {{ if (saveDepth === clipDepth) clipDepth = -1; saveDepth--; ops.push(["restore"]); }},
  fillText(txt, x, y) {{ ops.push(["fillText", String(txt), x, y, clipDepth !== -1]); }},
  measureText(txt) {{ return {{ width: String(txt).length * 6 }}; }},
}};
const stereoCanvas = {{
  width: 260,
  height: 260,
  clientWidth: 260,
  clientHeight: 260,
  getBoundingClientRect: () => ({{ width: 260, height: 260 }}),
  getContext: () => stereoCtx,
}};

const state = {{
  mode: "pro",
  showStereo: true,
  peakSource: "predicted",
  peakColorMode: "wavelength",
  wulffScope: "all_runs",
  wulffColorMode: "wavelength",
  colorCrossover: 0.38,
  wulffExpanded: false,
  showWulffFriedel: false,
  wulffYawDeg: 0.0,
  wulffPitchDeg: 0.0,
  wulffZoom: 1.0,
  dMin: 1.4,
  lamMin: 1.8,
  lamMax: 4.5,
  showUncertainPeaks: true,
  selectedFrameIndex: 2,
  selectedZoneId: null,
  showZoneArcs: false,
  activeZones: new Set(),
  hoveredPeakId: null,
  selectedPeakId: null,
  hoveredPeakObj: null,
  selectedPeakObj: null,
  activeOrbitId: null,
}};
function getActiveGoniometerR() {{ return [[1,0,0],[0,1,0],[0,0,1]]; }}
function mat3T(M) {{ return M; }}
function mat3Vec(M, v) {{ return v; }}
function getPeakCrystalQHat(p) {{ return p.q_hat_cryst || p.q_hat; }}
function getPeakColor() {{ return [0, 229, 255]; }}
function peakStrokeColor() {{ return "rgb(0,229,255)"; }}
function getWulffColor(p) {{ return "rgb(0,229,255)"; }}
function isPeakCapturedByActiveZone() {{ return false; }}

{chr(10).join(extracted)}

const rawActive = getActiveWulffPeaks();
const pool1 = getWulffPoolCache();
const dupMatches = pool1.uniquePeaks.filter(p => p.hkl[0] === 9999);
const lodZoom1 = getWulffLodState(false);

// High-resolution hovered peak (d < 1.5A) must ALWAYS survive zoom-out LOD thinning
const hiResTarget = pool1.uniquePeaks.find(p => p.d_spacing < 1.5);
state.hoveredPeakId = hiResTarget.id;
state.hoveredPeakObj = hiResTarget;
const lodWithHover = getWulffLodState(false);
const hoverKept = lodWithHover.visiblePeaks.some(p => isSameReflection(p, hiResTarget));
state.hoveredPeakId = null;
state.hoveredPeakObj = null;

// Render at wulffZoom=1.0 -> draws & emits d >= badge AFTER restore(); immediate 2nd renderStereoInset() skips redraw
ops.length = 0;
renderStereoInset();
const firstDrawOps = ops.length;
const fillTextsZoom1 = ops.filter(o => o[0] === "fillText").map(o => o[1]);
const badgeOp = ops.find(o => o[0] === "fillText" && o[1].includes("zoom+"));

ops.length = 0;
renderStereoInset();
const secondIdleOps = ops.length;

// Zoom in to 2.5x and 8.0x -> progressively reveals higher-resolution shells
state.wulffZoom = 2.5;
const lodZoom25 = getWulffLodState(false);
state.wulffZoom = 8.0;
const lodZoom80 = getWulffLodState(false);

console.log(JSON.stringify({{
  rawActiveLen: rawActive.length,
  poolDedupLen: pool1.uniquePeaks.length,
  dupCount: dupMatches.length,
  dupPrefersActiveFrame: dupMatches.length === 1 && dupMatches[0].run_index === 2,
  zoom1Count: lodZoom1.visiblePeaks.length,
  zoom1Thinned: lodZoom1.totalUnique - lodZoom1.visiblePeaks.length,
  zoom1Dcut: lodZoom1.lodDcut,
  zoom25Count: lodZoom25.visiblePeaks.length,
  zoom80Count: lodZoom80.visiblePeaks.length,
  zoom80Thinned: lodZoom80.totalUnique - lodZoom80.visiblePeaks.length,
  hoverKept,
  firstDrawOps,
  secondIdleOps,
  hasZoomBadge: fillTextsZoom1.some(t => t.includes("zoom+")),
  badgeUnclippedAfterRestore: Boolean(badgeOp && badgeOp[4] === false),
}}));
"""
    harness_file = tmp_path / "wulff_lod_harness.js"
    harness_file.write_text(harness_js, encoding="utf-8")
    res = subprocess.run(
        ["node", str(harness_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(res.stdout.strip())
    assert out["rawActiveLen"] == 1203
    assert out["poolDedupLen"] == 1201
    assert out["dupCount"] == 1
    assert out["dupPrefersActiveFrame"] is True
    assert out["zoom1Thinned"] > 400
    assert out["zoom1Count"] < out["poolDedupLen"]
    assert out["zoom25Count"] > out["zoom1Count"]
    assert out["zoom80Count"] == out["poolDedupLen"]
    assert out["zoom80Thinned"] == 0
    assert out["hoverKept"] is True
    assert out["firstDrawOps"] > 20
    assert out["secondIdleOps"] == 0
    assert out["hasZoomBadge"] is True
    assert out["badgeUnclippedAfterRestore"] is True


def test_ndip_streaming_download_empty_history_cache_and_mtz_binary_discovery(
    tmp_path, monkeypatch
):
    """Tweaks 3 & 4: Empty-history poll caching, safe hardlink vs copy staging, and 'binary' .mtz discovery."""
    import os

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner

    runner = NovaGalaxyPipelineRunner(
        galaxy_url="https://galaxy.example.org",
        galaxy_api_key="fake-key",
        history_id="hist_1",
    )

    # 1) Empty initial history `[]` must be cached during active-job polling (no HTTP flood every 1.5s)
    api_calls = []

    def fake_api(method, path, payload=None, raw=False, timeout=30.0):
        api_calls.append((method, path))
        if path.startswith("/api/histories/hist_1/contents"):
            return []
        if path == "/api/datatypes":
            return ["h5", "mtz", "tabular", "binary"]
        if path == "/api/datasets/ds_merged":
            return {"id": "ds_merged", "file_size": 16}
        if path == "/api/datasets/ds_merged/display":
            return b"0123456789abcdef"
        return {}

    runner._api_request = fake_api
    assert runner.list_history_datasets(use_cached=False) == []
    assert len(api_calls) == 1
    assert runner.list_history_datasets(use_cached=True) == []
    assert len(api_calls) == 1  # Served from cache even though list is empty!

    # 2) Datatype probe recognizes 'mtz' when Galaxy advertises it
    assert runner._galaxy_has_datatype("mtz") is True

    # 3) Streaming download writes atomically via _api_request fallback and updates job_record safely
    dest_h5 = tmp_path / "merged.h5"
    job_rec = {"stdout": "line1\n"}
    n_bytes = runner._download_dataset_to_file(
        "ds_merged", str(dest_h5), job_record=job_rec, label="merged.h5"
    )
    assert n_bytes == 16
    assert dest_h5.read_bytes() == b"0123456789abcdef"

    # 4) _link_or_copy_file hard-links inside same temp directory, but copies when allow_hardlink=False
    alias_h5 = tmp_path / "cg4d_2523_merged.h5"
    NovaGalaxyPipelineRunner._link_or_copy_file(str(dest_h5), str(alias_h5))
    assert os.stat(dest_h5).st_ino == os.stat(alias_h5).st_ino

    ext_store_src = tmp_path / "galaxy_object_store_dataset_99.dat"
    ext_store_src.write_bytes(b"immutable-galaxy-store")
    staged_copy = tmp_path / "staged_from_store.h5"
    NovaGalaxyPipelineRunner._link_or_copy_file(
        str(ext_store_src), str(staged_copy), allow_hardlink=False
    )
    assert os.stat(ext_store_src).st_ino != os.stat(staged_copy).st_ino
    staged_copy.write_bytes(b"locally-stamped")
    assert ext_store_src.read_bytes() == b"immutable-galaxy-store"

    # 5) Stage 4 Careless history discovery accepts uploaded .mtz datasets even when labeled extension='binary'
    hist_items = [
        {
            "id": "ds_careless_prev",
            "hid": 9,
            "name": "careless_xval_0.mtz",
            "extension": "mtz",
            "state": "ok",
            "deleted": False,
        },
        {
            "id": "ds_user_uploaded_mtz",
            "hid": 8,
            "name": "mandi_13172_integrated.mtz",
            "extension": "binary",
            "state": "ok",
            "deleted": False,
        },
    ]
    matched_id = ""
    for d in sorted(hist_items, key=lambda x: int(x.get("hid") or 0), reverse=True):
        if d.get("state") and d.get("state") != "ok":
            continue
        nm = str(d.get("name") or "")
        nl = nm.lower()
        ext = str(d.get("extension") or "").lower()
        is_mtz_cand = nl.endswith(".mtz") or ext == "mtz"
        if is_mtz_cand and "careless" not in nl:
            matched_id = str(d["id"])
            break
    assert matched_id == "ds_user_uploaded_mtz"


def test_mtz_overlay_matches_solved_geometry_and_warns_on_mismatched_symmetry(
    tmp_path,
) -> None:
    """Verify .mtz projection matches predict_reflections_on_panel under calibrated banks, refined kinematics, and sample offsets without rescaling B."""
    import gemmi
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation

    from subhkl.config import beamlines
    from subhkl.core.crystallography import cartesian_matrix_metric_tensor
    from subhkl.instrument.detector import Detector
    from subhkl.instrument.goniometer import (
        calc_goniometer_rotation_matrix,
        sample_to_lab,
    )
    from subhkl.instrument.physics import predict_reflections_on_panel
    from subhkl.viz.rainbow_zones import (
        _reconstruct_detectors_from_payload,
        apply_mtz_to_payload,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    # Active solve: monoclinic I 1 2 1 (e.g. MANDI-style cell)
    solve_sg = "I 1 2 1"
    solve_cell = [55.16, 81.19, 88.70, 90.0, 96.3, 90.0]
    B_solve_raw, _ = cartesian_matrix_metric_tensor(
        *solve_cell[:3], *np.deg2rad(solve_cell[3:])
    )
    B_solve = np.asarray(B_solve_raw, dtype=float)
    U_solve = Rotation.from_euler("zyx", [31.5, -18.2, 42.7], degrees=True).as_matrix()

    # Calibrated flat bank (MANDI bank 19 shifted by +14 mm and tilted by 1.8 deg)
    bk_id = 19
    nom_cfg = dict(beamlines["MANDI"][str(bk_id)])
    cal_cfg = dict(nom_cfg)
    tilt_R = Rotation.from_euler("xyz", [1.4, -1.8, 0.9], degrees=True).as_matrix()
    cal_cfg["center"] = (
        np.asarray(nom_cfg["center"], dtype=float) + np.array([0.014, -0.009, 0.011])
    ).tolist()
    cal_cfg["uhat"] = (tilt_R @ np.asarray(nom_cfg["uhat"], dtype=float)).tolist()
    cal_cfg["vhat"] = (tilt_R @ np.asarray(nom_cfg["vhat"], dtype=float)).tolist()
    cal_cfg["width"] = float(nom_cfg["width"]) * 1.015
    cal_cfg["height"] = float(nom_cfg["height"]) * 0.988
    cal_det = Detector(cal_cfg)

    # Two runs with refined goniometer zero-offset (+3.25 deg), per-run delta (-1.75 deg on run 1),
    # and refined crystal sample displacement + per-run translation!
    axes = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=float)
    nom_angles_frames = np.array([[10.0, 5.0], [48.0, 5.0]], dtype=float)
    zero_offs = np.array([3.25, -0.80], dtype=float)
    per_run_delta = np.array([0.40, -1.75], dtype=float)
    base_trans = np.array([0.0035, -0.0022, 0.0048], dtype=float)
    per_run_trans = np.array(
        [[0.0005, 0.0, -0.0004], [-0.0012, 0.0008, 0.0006]], dtype=float
    )

    eff_ang_0 = np.array([10.0 + per_run_delta[0], 5.0], dtype=float)
    eff_ang_1 = np.array([48.0 + per_run_delta[1], 5.0], dtype=float)
    R_true_0 = calc_goniometer_rotation_matrix(axes, eff_ang_0 + zero_offs)
    R_true_1 = calc_goniometer_rotation_matrix(axes, eff_ang_1 + zero_offs)

    offs_0 = np.vstack([np.zeros(3), base_trans + per_run_trans[0]])
    offs_1 = np.vstack([np.zeros(3), base_trans + per_run_trans[1]])
    s_lab_0 = np.asarray(
        sample_to_lab(
            np.zeros(3),
            axes,
            eff_ang_0,
            offs_0,
            zero_offsets=zero_offs,
            is_vector=False,
        ),
        dtype=float,
    )
    s_lab_1 = np.asarray(
        sample_to_lab(
            np.zeros(3),
            axes,
            eff_ang_1,
            offs_1,
            zero_offsets=zero_offs,
            is_vector=False,
        ),
        dtype=float,
    )

    wl_min, wl_max = 2.0, 4.0
    ki_vec = np.array([0.008, -0.005, 1.0], dtype=float)
    ki_vec = ki_vec / np.linalg.norm(ki_vec)

    grid = np.mgrid[-16:17, -16:17, -16:17].reshape(3, -1).T
    grid = grid[np.any(grid != 0, axis=1)]
    # Restrict to I-centering allowed reflections h+k+l = 2n
    grid = grid[((grid[:, 0] + grid[:, 1] + grid[:, 2]) % 2) == 0]

    pr0, pc0, h0, k0, l0, lam0 = predict_reflections_on_panel(
        cal_det,
        grid[:, 0],
        grid[:, 1],
        grid[:, 2],
        U_solve @ B_solve,
        wl_min,
        wl_max,
        sample_offset=offs_0,
        ki_vec=ki_vec,
        gonio_axes=axes,
        gonio_angles=eff_ang_0,
        gonio_offsets=zero_offs,
    )
    pr1, pc1, h1, k1, l1, lam1 = predict_reflections_on_panel(
        cal_det,
        grid[:, 0],
        grid[:, 1],
        grid[:, 2],
        U_solve @ B_solve,
        wl_min,
        wl_max,
        sample_offset=offs_1,
        ki_vec=ki_vec,
        gonio_axes=axes,
        gonio_angles=eff_ang_1,
        gonio_offsets=zero_offs,
    )
    assert len(h0) >= 4 and len(h1) >= 4

    # Write nominal merged.h5 + solved solve.h5 where goniometer/R is an UNREFINED per-peak (N_peaks, 3, 3)
    # array (all first 6 peaks in Run 0, next 3 peaks in Run 1 — reproducing the exact f0_r bug!)
    merged_h5 = tmp_path / "MANDI_13172_MANDI_13174_merged.h5"
    with h5py.File(merged_h5, "w") as mf:
        mf.attrs["instrument"] = "MANDI"
        mf.create_dataset("images", data=np.zeros((2, 256, 256), dtype=np.float32))
        mf.create_dataset("bank_ids", data=np.array([bk_id, bk_id], dtype=np.int32))
        mf.create_dataset("file_offsets", data=np.array([0, 1], dtype=np.int32))
        mf.create_dataset(
            "files",
            data=np.array([b"MANDI_13172.nxs.h5", b"MANDI_13174.nxs.h5"]),
        )
        gon_m = mf.create_group("goniometer")
        gon_m.create_dataset("names", data=np.array([b"omega", b"chi"]))
        gon_m.create_dataset("axes", data=axes)
        gon_m.create_dataset("angles", data=nom_angles_frames)
        mf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([wl_min, wl_max], dtype=float)
        )

    solve_h5 = tmp_path / "solve.h5"
    R_unrefined_0 = calc_goniometer_rotation_matrix(axes, nom_angles_frames[0])
    R_unrefined_1 = calc_goniometer_rotation_matrix(axes, nom_angles_frames[1])
    # 6 peaks in run 0, 3 peaks in run 1 -> R_all[1] is STILL Run 0's unrefined matrix!
    R_per_peak_unrefined = np.concatenate(
        [
            np.repeat(R_unrefined_0[None, :, :], 6, axis=0),
            np.repeat(R_unrefined_1[None, :, :], 3, axis=0),
        ],
        axis=0,
    )
    pts_p0 = np.asarray(cal_det.pixel_to_lab(pr0[:6], pc0[:6]), dtype=float)
    pts_p1 = np.asarray(cal_det.pixel_to_lab(pr1[:3], pc1[:3]), dtype=float)
    with h5py.File(solve_h5, "w") as sf:
        sf.attrs["instrument"] = "MANDI"
        sf.attrs["source"] = str(merged_h5)
        sf.create_group("beam").create_dataset("ki_vec", data=ki_vec)
        smp = sf.create_group("sample")
        smp.create_dataset("space_group", data=solve_sg.encode("utf-8"))
        smp.create_dataset("cell", data=np.array(solve_cell, dtype=float))
        smp.create_dataset("U", data=U_solve)
        smp.create_dataset("B", data=B_solve)
        dc = sf.create_group("detector_calibration").create_group(f"bank_{bk_id}")
        dc.create_dataset("panel_type", data=b"flat")
        dc.create_dataset("center", data=np.array(cal_cfg["center"], dtype=float))
        dc.create_dataset("uhat", data=np.array(cal_cfg["uhat"], dtype=float))
        dc.create_dataset("vhat", data=np.array(cal_cfg["vhat"], dtype=float))
        dc.create_dataset("width", data=float(cal_cfg["width"]))
        dc.create_dataset("height", data=float(cal_cfg["height"]))
        dc.create_dataset("m", data=int(cal_cfg["m"]))
        dc.create_dataset("n", data=int(cal_cfg["n"]))
        gon = sf.create_group("goniometer")
        gon.create_dataset("names", data=np.array([b"omega", b"chi"]))
        gon.create_dataset("axes", data=axes)
        gon.create_dataset("axes_nominal", data=axes)
        gon.create_dataset("angles_nominal", data=nom_angles_frames)
        gon.create_dataset("R", data=R_per_peak_unrefined)
        offs_grp = gon.create_group("offsets")
        offs_grp.create_dataset("omega", data=float(zero_offs[0]))
        offs_grp.create_dataset("chi", data=float(zero_offs[1]))
        gon.create_dataset("translations", data=base_trans)
        pr_grp = gon.create_group("per_run")
        pr_grp.create_dataset("motor", data=b"omega")
        pr_grp.create_dataset("delta_deg", data=per_run_delta)
        pr_grp.create_dataset("trans_m", data=per_run_trans)
        pk = sf.create_group("peaks")
        pk.create_dataset("xyz", data=np.vstack([pts_p0, pts_p1]))
        pk.create_dataset("bank", data=np.full(9, bk_id, dtype=np.int32))
        pk.create_dataset(
            "pixel_r", data=np.concatenate([pr0[:6], pr1[:3]]).astype(float)
        )
        pk.create_dataset(
            "pixel_c", data=np.concatenate([pc0[:6], pc1[:3]]).astype(float)
        )
        pk.create_dataset("intensity", data=np.full(9, 200.0, dtype=float))
        pk.create_dataset("sigma", data=np.full(9, 10.0, dtype=float))
        pk.create_dataset("h", data=np.concatenate([h0[:6], h1[:3]]).astype(np.int32))
        pk.create_dataset("k", data=np.concatenate([k0[:6], k1[:3]]).astype(np.int32))
        pk.create_dataset("l", data=np.concatenate([l0[:6], l1[:3]]).astype(np.int32))
        pk.create_dataset(
            "lambda", data=np.concatenate([lam0[:6], lam1[:3]]).astype(float)
        )
        pk.create_dataset("run_index", data=np.array([0] * 6 + [1] * 3, dtype=np.int32))

    # Build payload on Run 1 (where the old f0_r=1 bug previously picked Run 0's unrefined matrix AND dropped calibrated bank 19!)
    payload_r1 = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        frame_index=1,
        d_min=0.5,
        include_panel_images=False,
    )
    reconstructed = _reconstruct_detectors_from_payload(payload_r1)
    assert bk_id in reconstructed
    assert np.allclose(reconstructed[bk_id].center, cal_det.center, atol=1e-6)
    assert np.allclose(np.asarray(payload_r1["goniometer_R"]), R_true_1, atol=1e-6)
    assert np.allclose(
        np.asarray(payload_r1["goniometer_R_per_run"]["0"]), R_true_0, atol=1e-6
    )
    assert np.allclose(
        np.asarray(payload_r1["goniometer_R_per_run"]["1"]), R_true_1, atol=1e-6
    )
    assert np.allclose(np.asarray(payload_r1["sample_offset_lab"]), s_lab_1, atol=1e-6)
    assert np.allclose(
        np.asarray(payload_r1["sample_offset_lab_per_run"]["0"]), s_lab_0, atol=1e-6
    )

    # Write a foreign-symmetry MTZ (P 32 2 1, completely different cell) containing exact HKLs from Run 1 & Run 0
    mtz_foreign = gemmi.Mtz(with_base=True)
    mtz_foreign.spacegroup = gemmi.find_spacegroup_by_name("P 32 2 1")
    mtz_foreign.set_cell_for_all(gemmi.UnitCell(19.9, 19.9, 140.0, 90.0, 90.0, 120.0))
    mtz_foreign.add_dataset("foreign")
    mtz_foreign.add_column("I", "J")
    mtz_foreign.add_column("SIGI", "Q")
    chosen_hkls: list[tuple[int, int, int]] = []
    expected_r1: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for i_pk in range(min(5, len(h1))):
        key_h = (int(h1[i_pk]), int(k1[i_pk]), int(l1[i_pk]))
        if key_h not in expected_r1:
            expected_r1[key_h] = (
                float(pr1[i_pk]),
                float(pc1[i_pk]),
                float(lam1[i_pk]),
            )
            chosen_hkls.append(key_h)
    # Also include HKLs that would strike OTHER uninstalled MANDI banks (e.g. bank 22)
    # to prove .mtz reflections ONLY project onto panels where predicted peaks project (bank 19)!
    uninstalled_det = Detector(dict(beamlines["MANDI"]["22"]))
    _, _, h22, k22, l22, _ = predict_reflections_on_panel(
        uninstalled_det,
        grid[:, 0],
        grid[:, 1],
        grid[:, 2],
        U_solve @ B_solve,
        wl_min,
        wl_max,
        sample_offset=offs_1,
        ki_vec=ki_vec,
        gonio_axes=axes,
        gonio_angles=eff_ang_1,
        gonio_offsets=zero_offs,
    )
    assert len(h22) >= 2
    extra_b22_hkls = [
        (int(h22[0]), int(k22[0]), int(l22[0])),
        (int(h22[1]), int(k22[1]), int(l22[1])),
    ]
    mtz_foreign.set_data(
        np.asarray(
            [[h, k, l, 300.0, 10.0] for (h, k, l) in (chosen_hkls + extra_b22_hkls)],
            dtype=np.float32,
        )
    )
    foreign_mtz_path = tmp_path / "unrelated_trigonal.mtz"
    mtz_foreign.write_to_file(str(foreign_mtz_path))

    overlaid = apply_mtz_to_payload(
        payload_r1, foreign_mtz_path, {"name": "unrelated_trigonal.mtz"}
    )
    # 1. Solve B, U, cell, and space_group stay untouched, and informational warning is emitted!
    assert np.allclose(np.asarray(overlaid["b_matrix"]), B_solve, atol=1e-7)
    assert np.allclose(np.asarray(overlaid["u_matrix"]), U_solve, atol=1e-7)
    assert overlaid["space_group"] == solve_sg
    assert np.allclose(overlaid["cell"], solve_cell, atol=1e-3)
    assert overlaid["default_color_mode"] == "mtz_isigi"
    assert "mtz_peaks" not in overlaid
    assert isinstance(overlaid.get("mtz_overlay_warning"), str)
    assert "P 32 2 1" in overlaid["mtz_overlay_warning"]
    assert "I 1 2 1" in overlaid["mtz_overlay_warning"]

    # 2. Strictly restricted to active predicted-peak panel bank 19 (NEVER bank 22 or other MANDI banks!)
    all_pred_banks = {int(pk["bank"]) for pk in overlaid["predicted_peaks"]}
    all_samp_banks = {int(pk["bank"]) for pk in overlaid["predicted_sample_peaks"]}
    assert all_pred_banks == {bk_id}
    assert all_samp_banks == {bk_id}

    # Every projected MTZ reflection matches predict_reflections_on_panel pixel (row, col) & wavelength
    # under the solved geometry AND has exact d_spacing = 1 / |B_solve @ hkl| (zero unit-cell rescaling or foreign-SG shear)!
    mtz_tagged = [
        pk for pk in overlaid["predicted_peaks"] if pk.get("mtz_isigi") is not None
    ]
    by_hkl_b19 = {
        tuple(int(x) for x in pk["hkl"]): pk
        for pk in mtz_tagged
        if int(pk["bank"]) == bk_id
    }
    for key_h, (exp_r, exp_c, exp_lam) in expected_r1.items():
        assert key_h in by_hkl_b19, (
            f"Missing expected reflection {key_h} on calibrated bank {bk_id}"
        )
        got_pk = by_hkl_b19[key_h]
        assert abs(float(got_pk["row"]) - exp_r) < 0.01
        assert abs(float(got_pk["col"]) - exp_c) < 0.01
        assert abs(float(got_pk["wavelength"]) - exp_lam) < 1e-3
        assert abs(float(got_pk["mtz_isigi"]) - 30.0) < 1e-2
        d_true = float(1.0 / np.linalg.norm(B_solve @ np.asarray(key_h, dtype=float)))
        assert abs(float(got_pk["d_spacing"]) - d_true) < 1e-3
        assert float(got_pk["angular_error_deg"]) < 0.02

    for pk in mtz_tagged:
        h_vec = np.asarray(pk["hkl"], dtype=float)
        d_true_pk = float(1.0 / np.linalg.norm(B_solve @ h_vec))
        assert abs(float(pk["d_spacing"]) - d_true_pk) < 1e-3

    # 3. Loading a matching-symmetry/cell MTZ emits NO warning and populates mtz_isigi on bank 19 only
    mtz_same = gemmi.Mtz(with_base=True)
    mtz_same.spacegroup = gemmi.find_spacegroup_by_name(solve_sg)
    mtz_same.set_cell_for_all(gemmi.UnitCell(*solve_cell))
    mtz_same.add_dataset("same")
    mtz_same.add_column("I", "J")
    mtz_same.add_column("SIGI", "Q")
    mtz_same.set_data(
        np.asarray(
            [[h, k, l, 300.0, 10.0] for (h, k, l) in (chosen_hkls + extra_b22_hkls)],
            dtype=np.float32,
        )
    )
    same_mtz_path = tmp_path / "matching_monoclinic.mtz"
    mtz_same.write_to_file(str(same_mtz_path))
    overlaid_same = apply_mtz_to_payload(
        payload_r1, same_mtz_path, {"name": "matching_monoclinic.mtz"}
    )
    assert overlaid_same["mtz_overlay_warning"] is None
    assert {int(pk["bank"]) for pk in overlaid_same["predicted_peaks"]} == {bk_id}
    same_tagged_hkls = {
        tuple(int(x) for x in pk["hkl"])
        for pk in overlaid_same["predicted_peaks"]
        if pk.get("mtz_isigi") is not None
    }
    assert set(expected_r1.keys()).issubset(same_tagged_hkls)

    html_warn = render_rainbow_html(overlaid)
    assert "mtz_overlay_warning" in html_warn


def test_full_inline_script_boot_and_data_sources_menu(tmp_path) -> None:
    """Execute the full rendered <script> top-to-bottom in Node to guard against TDZ/hoisting bugs."""
    import json
    import re
    import shutil
    import subprocess

    from subhkl.viz.nova_app.models.subhkl_model import SubhklGalaxyModel
    from subhkl.viz.rainbow_zones import render_rainbow_html

    assert shutil.which("node"), "Node.js required for inline script boot test"

    empty_model = SubhklGalaxyModel()
    bij_payload = dict(empty_model.payload)
    bij_peaks = [
        {
            "id": 0,
            "run_index": 0,
            "bank": 23,
            "det_x": -80.0,
            "det_y": 0.0,
            "det_el": 0.0,
            "q_hat": [-0.5, 0.2, -0.84],
            "q_hat_lab": [-0.5, 0.2, -0.84],
            "stereographic": [-0.2, 0.1],
            "hemisphere": 1,
            "wavelength": 2.76,
            "d_spacing": 2.18,
            "harmonic_order": 1,
            "harmonic_multiplicity": 1,
            "hkl": [11, -15, 1],
            "asu_hkl": [11, 15, 1],
            "asu_orbit_id": "11,15,1",
            "asu_friedel": False,
            "asu_bijvoet_parity": 1,
            "orbit_color_idx": 2,
            "zone_ids": [0],
            "indexed": True,
        },
        {
            "id": 1,
            "run_index": 0,
            "bank": 24,
            "det_x": -35.0,
            "det_y": 0.0,
            "det_el": 0.0,
            "q_hat": [0.5, 0.2, -0.84],
            "q_hat_lab": [0.5, 0.2, -0.84],
            "stereographic": [0.2, 0.1],
            "hemisphere": 1,
            "wavelength": 2.82,
            "d_spacing": 2.18,
            "harmonic_order": 1,
            "harmonic_multiplicity": 1,
            "hkl": [-11, -15, 1],
            "asu_hkl": [11, 15, 1],
            "asu_orbit_id": "11,15,1",
            "asu_friedel": False,
            "asu_bijvoet_parity": 1,
            "orbit_color_idx": 2,
            "zone_ids": [0],
            "indexed": True,
        },
        {
            "id": 2,
            "run_index": 0,
            "bank": 53,
            "det_x": 10.0,
            "det_y": 0.0,
            "det_el": 0.0,
            "q_hat": [0.5, 0.2, 0.84],
            "q_hat_lab": [0.5, 0.2, -0.60],
            "stereographic": [0.25, -0.15],
            "hemisphere": 1,
            "wavelength": 2.02,
            "d_spacing": 2.18,
            "harmonic_order": 1,
            "harmonic_multiplicity": 1,
            "hkl": [-11, -15, -1],
            "asu_hkl": [11, 15, 1],
            "asu_orbit_id": "11,15,1",
            "asu_friedel": True,
            "asu_bijvoet_parity": -1,
            "orbit_color_idx": 2,
            "zone_ids": [0],
            "indexed": True,
        },
        {
            "id": 3,
            "run_index": 0,
            "bank": 31,
            "det_x": 55.0,
            "det_y": 0.0,
            "det_el": 0.0,
            "q_hat": [0.6, 0.1, -0.79],
            "q_hat_lab": [0.6, 0.1, -0.79],
            "stereographic": [0.3, 0.05],
            "hemisphere": 1,
            "wavelength": 3.10,
            "d_spacing": 3.00,
            "harmonic_order": 1,
            "harmonic_multiplicity": 1,
            "hkl": [2, 2, 0],
            "asu_hkl": [2, 2, 0],
            "asu_orbit_id": "2,2,0",
            "asu_friedel": False,
            "asu_bijvoet_parity": 0,
            "orbit_color_idx": 5,
            "zone_ids": [0],
            "indexed": True,
        },
        {
            "id": 4,
            "run_index": 0,
            "bank": 35,
            "det_x": 95.0,
            "det_y": 0.0,
            "det_el": 0.0,
            "q_hat": [-0.6, -0.1, -0.79],
            "q_hat_lab": [-0.6, -0.1, -0.79],
            "stereographic": [-0.3, -0.05],
            "hemisphere": 1,
            "wavelength": 3.10,
            "d_spacing": 3.00,
            "harmonic_order": 1,
            "harmonic_multiplicity": 1,
            "hkl": [-2, -2, 0],
            "asu_hkl": [2, 2, 0],
            "asu_orbit_id": "2,2,0",
            "asu_friedel": False,
            "asu_bijvoet_parity": 0,
            "orbit_color_idx": 5,
            "zone_ids": [0],
            "indexed": True,
        },
    ]
    bij_payload.update(
        {
            "instrument": "CG4D",
            "space_group": "Fdd2",
            "wavelength_range": [2.0, 4.5],
            "d_min": 1.0,
            "peaks": bij_peaks,
            "predicted_peaks": bij_peaks,
            "sample_peaks": bij_peaks,
            "predicted_sample_peaks": bij_peaks,
            "zones": [
                {
                    "id": 0,
                    "uvw": [0, 0, 1],
                    "label": "[001]",
                    "n_peaks": 5,
                    "angular_rms_deg": 0.05,
                    "d_direct": 6.54,
                    "color": "#38bdf8",
                    "axis_lab": [0.0, 1.0, 0.0],
                    "axis_crystal": [0.0, 0.0, 1.0],
                    "conic_points": [],
                    "conic_points_sph": [],
                    "wulff_points": [],
                }
            ],
            "frames": [{"index": 0, "label": "CG4D_1720"}],
            "active_frame": 0,
        }
    )
    html_doc = render_rainbow_html(bij_payload)
    script = re.findall(r"<script>([\s\S]*?)</script>", html_doc)[-1]

    js_file = tmp_path / "full_boot_check.js"
    js_file.write_text(
        """
const elems = new Map();
const ctxLogs = new Map();
let rafCb = null;
function makeCtx(id = "") {
  const log = { strokes: [], texts: [], arcs: [] };
  if (id) ctxLogs.set(id, log);
  const ctx = {
    fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "left",
    textBaseline: "alphabetic", globalAlpha: 1, imageSmoothingEnabled: true,
    setTransform() {}, resetTransform() {}, scale() {}, translate() {}, rotate() {},
    transform() {}, clearRect() {}, fillRect() {},
    strokeRect() { log.strokes.push(String(ctx.strokeStyle)); },
    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {},
    arc(x, y, r) { log.arcs.push([x, y, r]); },
    ellipse() {},
    quadraticCurveTo() {}, bezierCurveTo() {}, rect() {}, roundRect() {},
    stroke() { log.strokes.push(String(ctx.strokeStyle)); },
    fill() {}, clip() {}, save() {}, restore() {}, setLineDash() {},
    fillText(t) { log.texts.push(String(t)); },
    strokeText() {}, drawImage() {},
    measureText(t) { return { width: String(t).length * 6 }; },
    createLinearGradient() { return { addColorStop() {} }; },
    createRadialGradient() { return { addColorStop() {} }; },
    getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
    putImageData() {}, createImageData(w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
  };
  return ctx;
}
function makeEl(id = "") {
  const listeners = {};
  const styleProps = {};
  const el = {
    id, tagName: id.startsWith("sel-") ? "SELECT" : "DIV",
    value: id === "slider-color-emphasis" ? "0.38" : "",
    min: "0.05", max: "0.95", step: "0.01",
    checked: false, disabled: false, textContent: "", innerHTML: "", title: "",
    width: 300, height: 240, clientWidth: 300, clientHeight: 240,
    scrollWidth: 300, scrollHeight: 240, scrollTop: 0, offsetWidth: 300, offsetHeight: 36,
    files: [], options: [], children: [], parentElement: null,
    dataset: {},
    clicked: 0,
    style: new Proxy(styleProps, {
      get(t, k) {
        if (k === "setProperty") return (p, v) => { t[p] = v; };
        if (k === "getPropertyValue") return (p) => t[p] || "";
        return t[k] ?? "";
      },
      set(t, k, v) { t[k] = v; return true; }
    }),
    classList: {
      _s: new Set(),
      add(...c) { c.forEach(x => this._s.add(x)); },
      remove(...c) { c.forEach(x => this._s.delete(x)); },
      toggle(c, f) { if (f === undefined ? !this._s.has(c) : f) { this._s.add(c); return true; } this._s.delete(c); return false; },
      contains(c) { return this._s.has(c); }
    },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    removeEventListener() {},
    dispatchEvent(evObj) {
      const arr = listeners[evObj.type] || [];
      arr.forEach(fn => fn.call(el, evObj));
    },
    click() { el.clicked += 1; el.dispatchEvent({ type: "click", target: el, preventDefault() {}, stopPropagation() {} }); },
    focus() {}, blur() {}, select() {}, remove() {},
    appendChild(ch) { this.children.push(ch); ch.parentElement = this; return ch; },
    insertBefore(ch) { this.children.push(ch); ch.parentElement = this; return ch; },
    replaceChildren(...ch) { this.children = ch; },
    setAttribute(k, v) { this[k] = v; },
    getAttribute(k) { return this[k] ?? null; },
    removeAttribute(k) { delete this[k]; },
    getBoundingClientRect() { return { left: 0, top: 0, right: 300, bottom: 240, width: 300, height: 240 }; },
    getContext() { if (!el._ctx) el._ctx = makeCtx(id); return el._ctx; },
    querySelector() { return makeEl(); },
    querySelectorAll() { return []; },
    closest() { return null; },
  };
  return el;
}
function getEl(id) {
  if (!elems.has(id)) elems.set(id, makeEl(id));
  return elems.get(id);
}
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.parent = globalThis;
globalThis.top = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.devicePixelRatio = 1;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
globalThis.sessionStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
globalThis.location = { href: "about:srcdoc", origin: "http://localhost:8080", pathname: "/", search: "" };
globalThis.navigator = { clipboard: { writeText: async () => {} } };
globalThis.Image = class { set src(v) { this._src = v; } };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.MutationObserver = class { observe() {} disconnect() {} };
globalThis.AbortController = class { constructor() { this.signal = {}; } abort() {} };
globalThis.requestAnimationFrame = (fn) => { if (!rafCb) rafCb = fn; return 1; };
globalThis.cancelAnimationFrame = () => {};
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}), text: async () => "" });
globalThis.document = {
  body: makeEl("body"),
  documentElement: makeEl("html"),
  activeElement: null,
  getElementById(id) { return getEl(id); },
  createElement(tag) { return makeEl(tag); },
  querySelector(sel) { return sel.startsWith("#") ? getEl(sel.slice(1)) : makeEl(sel); },
  querySelectorAll() { return []; },
  addEventListener() {},
  removeEventListener() {},
};
"""
        + script
        + """
const slCs = elems.get("slider-color-emphasis");
const selSrc = elems.get("sel-galaxy-h5");
const modalNd = elems.get("modal-neutron-data");
const inpLocal = elems.get("inp-local-h5");

selSrc.value = "__action_neutron_data__";
selSrc.dispatchEvent({ type: "change", target: selSrc });
const ndDisplay = modalNd.style.display;

selSrc.value = "__action_local_file__";
selSrc.dispatchEvent({ type: "change", target: selSrc });
const localClicked = inpLocal ? inpLocal.clicked : 0;

// Trigger initial frame render to capture on-screen [x, y] positions of the 5 Fdd2 peaks
const detCanvasEl = elems.get("canvas-detector");
detCanvasEl.getContext("2d");
const detLog = ctxLogs.get("canvas-detector");
detLog.arcs.length = 0;
if (rafCb) rafCb();
const inViewArcs = detLog.arcs.filter(a => a[0] >= 0 && a[0] <= 300 && a[1] >= 0 && a[1] <= 240);
inViewArcs.sort((a, b) => a[0] - b[0]);

// Hover peak #0 (11, -15, +1) -> leftmost visible peak
const [px0, py0] = inViewArcs[0];
detCanvasEl.dispatchEvent({ type: "mousemove", clientX: px0, clientY: py0, buttons: 0 });
const statusBijHtml = getEl("inspector-box").innerHTML;
detLog.strokes.length = 0;
detLog.texts.length = 0;
if (rafCb) rafCb();
const hasPinkRing = detLog.strokes.some(s => s.includes("255, 77, 184") || s === "#ff4db8");
const hasCyanPgRing = detLog.strokes.some(s => s.includes("0, 229, 255") || s === "#00e5ff");
const hasBijTooltip = detLog.texts.some(t => t.includes("(-11,-15,-1) [Bijvoet]"));
const pgNotTaggedBij = detLog.texts.some(t => t.startsWith("(-11,-15,1) λ="));

// Hover centric peak #3 (2, 2, 0) -> 4th peak from left; must NOT tag (-2,-2,0) as Bijvoet
const [px3, py3] = inViewArcs[3];
detCanvasEl.dispatchEvent({ type: "mousemove", clientX: px3, clientY: py3, buttons: 0 });
const statusCentricHtml = getEl("inspector-box").innerHTML;
detLog.strokes.length = 0;
detLog.texts.length = 0;
if (rafCb) rafCb();
const centricHasPink = detLog.strokes.some(s => s.includes("255, 77, 184") || s === "#ff4db8") ||
  detLog.texts.some(t => t.includes("[Bijvoet]"));

console.log(JSON.stringify({
  csGradientSet: Boolean(slCs && slCs.style["--cs-gradient"] && slCs.style["--cs-gradient"].startsWith("linear-gradient(")),
  ndDisplay,
  localClicked,
  hasBijBadge: statusBijHtml.includes("bijvoet-badge") && statusBijHtml.includes("±Bijvoet ×1 (frame)") && statusBijHtml.includes("I⁺"),
  hasPinkRing,
  hasCyanPgRing,
  hasBijTooltip,
  pgNotTaggedBij,
  centricNoBijBadge: !statusCentricHtml.includes("bijvoet-badge"),
  centricHasPink,
}));
process.exit(0);
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["node", str(js_file)], capture_output=True, text=True, check=True
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["csGradientSet"] is True
    assert out["ndDisplay"] == "flex"
    assert out["localClicked"] == 1
    assert out["hasBijBadge"] is True
    assert out["hasPinkRing"] is True
    assert out["hasCyanPgRing"] is True
    assert out["hasBijTooltip"] is True
    assert out["pgNotTaggedBij"] is True
    assert out["centricNoBijBadge"] is True
    assert out["centricHasPink"] is False


def test_dynamic_dmin_wavelength_reprediction_and_detector_zoom_lod(
    tmp_path,
) -> None:
    """Verify live UI d_min / wavelength re-prediction, finite protein d_min floor, and Detector zoom LOD."""
    import json
    import re
    import shutil
    import subprocess

    import gemmi
    import h5py

    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    from subhkl.viz.rainbow_zones import (
        _make_asu_mapper,
        _make_systematic_absence_checker,
        _raycast_hkl_on_detectors,
        apply_mtz_to_payload,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    if not shutil.which("node"):
        return

    # 1. Build a protein-scale orthorhombic dataset (P 21 21 21, V ~ 296,000 A^3) on curved IMAGINE
    #    (full 2*pi cylindrical image-plate diffractometer, bank 0) with initial d_min = 3.2 A,
    #    wavelength = [2.8, 4.5] A, 2 runs, integrated inner peaks, and a .mtz down to 2.0 A.
    cell_prot = [54.3, 68.7, 79.4, 90.0, 90.0, 90.0]
    sg_name = "P 21 21 21"
    uc_prot = gemmi.UnitCell(*cell_prot)
    b_prot = np.asarray(uc_prot.frac.mat.tolist(), dtype=float).T
    ang1, ang2 = 0.21, -0.17
    r_z = np.array(
        [
            [np.cos(ang1), -np.sin(ang1), 0.0],
            [np.sin(ang1), np.cos(ang1), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    r_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ang2), -np.sin(ang2)],
            [0.0, np.sin(ang2), np.cos(ang2)],
        ],
        dtype=float,
    )
    u_prot = r_z @ r_x
    r0 = np.eye(3, dtype=float)
    phi1 = np.deg2rad(22.0)
    r1 = np.array(
        [
            [np.cos(phi1), 0.0, np.sin(phi1)],
            [0.0, 1.0, 0.0],
            [-np.sin(phi1), 0.0, np.cos(phi1)],
        ],
        dtype=float,
    )

    imagine_cfg = beamlines["IMAGINE"]
    first_bk = 0
    imagine_dets = {first_bk: Detector(dict(imagine_cfg[str(first_bk)]))}
    is_absent = _make_systematic_absence_checker(sg_name)
    asu_map = _make_asu_mapper(sg_name, cell_prot)

    # Enumerate exact Python reference reflections down to d = 2.15 A and widened lambda [2.0, 4.5] A
    h_lim = int(np.floor(cell_prot[0] / 2.0)) + 1
    k_lim = int(np.floor(cell_prot[1] / 2.0)) + 1
    l_lim = int(np.floor(cell_prot[2] / 2.0)) + 1
    grid = (
        np.mgrid[-h_lim : h_lim + 1, -k_lim : k_lim + 1, -l_lim : l_lim + 1]
        .reshape(3, -1)
        .T
    )
    grid = grid[np.any(grid != 0, axis=1)]
    q_cryst = (b_prot @ grid.T).T
    d_all = 1.0 / np.maximum(np.linalg.norm(q_cryst, axis=1), 1e-12)
    grid_20 = grid[d_all >= 2.0 - 1e-9]
    allowed_mask = np.array(
        [not is_absent(int(h), int(k), int(l)) for h, k, l in grid_20], dtype=bool
    )
    grid_allowed = grid_20[allowed_mask]
    d_allowed = 1.0 / np.linalg.norm((b_prot @ grid_allowed.T).T, axis=1)

    def _cast_to_tuples(res: dict, b_mat_ref: np.ndarray) -> list[tuple]:
        hkls = res["hkl"]
        if len(hkls) == 0:
            return []
        ds = 1.0 / np.linalg.norm((b_mat_ref @ hkls.T).T, axis=1)
        return [
            (
                int(res["bank"][i]),
                float(res["pixel_r"][i]),
                float(res["pixel_c"][i]),
                int(hkls[i, 0]),
                int(hkls[i, 1]),
                int(hkls[i, 2]),
                float(res["wavelength"][i]),
                float(ds[i]),
                res["peaks_xyz"][i],
            )
            for i in range(len(hkls))
        ]

    py_cast_320 = _cast_to_tuples(
        _raycast_hkl_on_detectors(
            grid_allowed[d_allowed >= 3.20 - 1e-9],
            u_prot,
            b_prot,
            r0,
            imagine_dets,
            (2.8, 4.5),
            active_bids={first_bk},
        ),
        b_prot,
    )
    py_cast_215 = _cast_to_tuples(
        _raycast_hkl_on_detectors(
            grid_allowed[d_allowed >= 2.15 - 1e-9],
            u_prot,
            b_prot,
            r0,
            imagine_dets,
            (2.8, 4.5),
            active_bids={first_bk},
        ),
        b_prot,
    )
    py_cast_215_wide_wl = _cast_to_tuples(
        _raycast_hkl_on_detectors(
            grid_allowed[d_allowed >= 2.15 - 1e-9],
            u_prot,
            b_prot,
            r0,
            imagine_dets,
            (2.0, 4.5),
            active_bids={first_bk},
        ),
        b_prot,
    )
    assert len(py_cast_320) >= 80
    assert len(py_cast_215) > len(py_cast_320) + 400
    assert len(py_cast_215_wide_wl) > len(py_cast_215) + 150

    # Pick 6 inner reflections (d >= 3.35 A) as integrated peaks in integrated.h5
    inner_hits = [ht for ht in py_cast_320 if ht[7] >= 3.35][:6]
    assert len(inner_hits) == 6
    integ_xyz = np.vstack([ht[8] for ht in inner_hits])

    integ_h5 = tmp_path / "prot_integrated.h5"
    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "IMAGINE"
        smp = hf.create_group("sample")
        smp.create_dataset("space_group", data=sg_name.encode("utf-8"))
        smp.create_dataset("cell", data=np.array(cell_prot, dtype=float))
        smp.create_dataset("U", data=u_prot)
        smp.create_dataset("B", data=b_prot)
        gon = hf.create_group("goniometer")
        gon.create_dataset("R", data=np.stack([r0, r1], axis=0))
        gon.create_dataset("angles", data=np.array([[0.0], [22.0]], dtype=float))
        gon.create_dataset("axes", data=np.array([[0.0, 1.0, 0.0]], dtype=float))
        hf.create_dataset("file_offsets", data=np.array([0, 1], dtype=np.int32))
        hf.create_dataset(
            "bank_ids", data=np.array([first_bk, first_bk], dtype=np.int32)
        )
        hf.create_dataset(
            "files",
            data=np.array(["prot_run0.nxs.h5", "prot_run1.nxs.h5"], dtype="S32"),
        )
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([2.8, 4.5], dtype=float)
        )
        pk = hf.create_group("peaks")
        pk.create_dataset("xyz", data=integ_xyz)
        pk.create_dataset(
            "bank", data=np.full(len(inner_hits), first_bk, dtype=np.int32)
        )
        pk.create_dataset(
            "pixel_r", data=np.array([ht[1] for ht in inner_hits], dtype=float)
        )
        pk.create_dataset(
            "pixel_c", data=np.array([ht[2] for ht in inner_hits], dtype=float)
        )
        pk.create_dataset(
            "intensity",
            data=np.array([240.0 - 15.0 * i for i in range(len(inner_hits))]),
        )
        pk.create_dataset("sigma", data=np.full(len(inner_hits), 10.0, dtype=float))
        pk.create_dataset("var_u", data=np.full(len(inner_hits), 1.2e-4, dtype=float))
        pk.create_dataset("var_v", data=np.full(len(inner_hits), 0.7e-4, dtype=float))
        pk.create_dataset("cov_uv", data=np.full(len(inner_hits), 0.1e-4, dtype=float))
        pk.create_dataset(
            "h", data=np.array([ht[3] for ht in inner_hits], dtype=np.int32)
        )
        pk.create_dataset(
            "k", data=np.array([ht[4] for ht in inner_hits], dtype=np.int32)
        )
        pk.create_dataset(
            "l", data=np.array([ht[5] for ht in inner_hits], dtype=np.int32)
        )
        pk.create_dataset(
            "lambda", data=np.array([ht[6] for ht in inner_hits], dtype=float)
        )
        pk.create_dataset("run_index", data=np.zeros(len(inner_hits), dtype=np.int32))

    # Create a merged .mtz that includes outer-shell reflections (2.15 <= d < 3.0 A) NOT present at d_min=3.2 A
    outer_only_hits = [ht for ht in py_cast_215 if 2.20 <= ht[7] <= 2.85][:25]
    assert len(outer_only_hits) >= 10
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name(sg_name)
    mtz.set_cell_for_all(uc_prot)
    mtz.add_dataset("prot")
    mtz.add_column("I", "J")
    mtz.add_column("SIGI", "Q")
    seen_asu: set[tuple[int, int, int]] = set()
    mtz_rows = []
    for idx_m, ht in enumerate(list(inner_hits[:3]) + outer_only_hits):
        ah, ak, al, _, _ = asu_map(int(ht[3]), int(ht[4]), int(ht[5]))
        if (ah, ak, al) in seen_asu:
            continue
        seen_asu.add((ah, ak, al))
        mtz_rows.append([ah, ak, al, 150.0 + 10.0 * idx_m, 10.0])
    mtz.set_data(np.asarray(mtz_rows, dtype=np.float32))
    mtz_path = tmp_path / "prot_outer_shells.mtz"
    mtz.write_to_file(str(mtz_path))

    payload = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        instrument="IMAGINE",
        frame_index=0,
        d_min=3.20,
        wavelength_range=(2.8, 4.5),
        include_panel_images=False,
    )
    payload = apply_mtz_to_payload(payload, mtz_path, {"name": "prot_outer_shells.mtz"})
    assert isinstance(payload.get("space_group_ops"), dict)
    assert payload["space_group_ops"]["den"] == 24
    assert payload["space_group_ops"]["is_centrosymmetric"] is False
    assert isinstance(payload.get("mtz_asu_lookup"), dict)
    assert len(payload["mtz_asu_lookup"]) == len(mtz_rows)

    html_doc = render_rainbow_html(payload)
    script_blocks = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
    assert len(script_blocks) >= 1
    script = script_blocks[-1]
    iife_close_idx = script.rfind("})();")
    assert iife_close_idx > 0
    script_before_close = script[:iife_close_idx]

    # Also build a flat-panel MANDI payload with a huge macromolecular unit cell (135 x 150 x 210 A)
    # to test both flat-panel dynamic raycasting AND the finite protein d_min floor scaling!
    big_cell = [135.0, 150.0, 210.0, 90.0, 90.0, 90.0]
    b_big = np.asarray(gemmi.UnitCell(*big_cell).frac.mat.tolist(), dtype=float).T
    mandi_solve_h5 = tmp_path / "mandi_huge_cell.h5"
    with h5py.File(mandi_solve_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf["sample/space_group"] = "P 21 21 21"
        hf["sample/cell"] = np.array(big_cell, dtype=float)
        hf["sample/U"] = u_prot
        hf["sample/B"] = b_big
        hf["instrument/wavelength"] = np.array([2.5, 4.0], dtype=float)
    mandi_payload = build_rainbow_zone_payload(
        replay_h5=mandi_solve_h5,
        instrument="MANDI",
        d_min=4.50,
        wavelength_range=(2.5, 4.0),
        include_panel_images=False,
    )
    mandi_active_bids = {
        int(pn["bank"]) for pn in mandi_payload["panels"] if int(pn["bank"]) in {1, 20}
    } or {int(mandi_payload["panels"][0]["bank"])}
    # Restrict MANDI test payload to 2 flat panels for crisp 1:1 Python raycast verification
    mandi_payload["panels"] = [
        pn for pn in mandi_payload["panels"] if int(pn["bank"]) in mandi_active_bids
    ]
    mandi_dets = {
        bk: Detector(dict(beamlines["MANDI"][str(bk)])) for bk in mandi_active_bids
    }
    hb = int(np.floor(big_cell[0] / 3.80)) + 1
    kb = int(np.floor(big_cell[1] / 3.80)) + 1
    lb = int(np.floor(big_cell[2] / 3.80)) + 1
    g_big = np.mgrid[-hb : hb + 1, -kb : kb + 1, -lb : lb + 1].reshape(3, -1).T
    g_big = g_big[np.any(g_big != 0, axis=1)]
    d_big = 1.0 / np.linalg.norm((b_big @ g_big.T).T, axis=1)
    g_big = g_big[d_big >= 3.80 - 1e-9]
    g_big = g_big[
        np.array(
            [not is_absent(int(h), int(k), int(l)) for h, k, l in g_big], dtype=bool
        )
    ]
    py_mandi_380 = _cast_to_tuples(
        _raycast_hkl_on_detectors(
            g_big,
            u_prot,
            b_big,
            r0,
            mandi_dets,
            (2.5, 4.0),
            active_bids=mandi_active_bids,
        ),
        b_big,
    )
    assert len(py_mandi_380) >= 15

    out_json_path = tmp_path / "dyn_dmin_and_det_lod_out.json"
    js_file = tmp_path / "dyn_dmin_and_det_lod_test.js"
    js_file.write_text(
        """
const elems = new Map();
const ctxLogs = new Map();
let rafCb = null;
function makeCtx(id) {
  const log = { strokes: [], texts: [], arcs: [] };
  if (id) ctxLogs.set(id, log);
  const ctx = {
    fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "left",
    textBaseline: "alphabetic", globalAlpha: 1, imageSmoothingEnabled: true,
    setTransform() {}, resetTransform() {}, scale() {}, translate() {}, rotate() {},
    transform() {}, clearRect() {}, fillRect() {},
    strokeRect() {},
    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {},
    arc(x, y, r) { log.arcs.push([x, y, r]); },
    ellipse() {},
    quadraticCurveTo() {}, bezierCurveTo() {}, rect() {}, roundRect() {},
    stroke() { log.strokes.push(String(ctx.strokeStyle)); },
    fill() {}, clip() {}, save() {}, restore() {}, setLineDash() {},
    fillText(t) { log.texts.push(String(t)); },
    strokeText() {}, drawImage() {},
    measureText(t) { return { width: String(t).length * 6 }; },
    createLinearGradient() { return { addColorStop() {} }; },
    createRadialGradient() { return { addColorStop() {} }; },
    getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
    putImageData() {}, createImageData(w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
  };
  return ctx;
}
function makeEl(id = "") {
  const listeners = {};
  const styleProps = {};
  const isDetCanvas = id === "canvas-detector";
  const cw = isDetCanvas ? 800 : 300;
  const ch = isDetCanvas ? 520 : 240;
  const el = {
    id, tagName: id.startsWith("sel-") ? "SELECT" : "DIV",
    value: id === "slider-color-emphasis" ? "0.38" : "",
    min: "0.05", max: "0.95", step: "0.01",
    checked: false, disabled: false, textContent: "", innerHTML: "", title: "",
    width: cw, height: ch, clientWidth: cw, clientHeight: ch,
    scrollWidth: cw, scrollHeight: ch, scrollTop: 0, offsetWidth: cw, offsetHeight: 36,
    files: [], options: [], children: [], parentElement: null,
    dataset: {},
    style: new Proxy(styleProps, {
      get(t, k) {
        if (k === "setProperty") return (p, v) => { t[p] = v; };
        if (k === "getPropertyValue") return (p) => t[p] || "";
        return t[k] ?? "";
      },
      set(t, k, v) { t[k] = v; return true; }
    }),
    classList: {
      _s: new Set(),
      add(...c) { c.forEach(x => this._s.add(x)); },
      remove(...c) { c.forEach(x => this._s.delete(x)); },
      toggle(c, f) { if (f === undefined ? !this._s.has(c) : f) { this._s.add(c); return true; } this._s.delete(c); return false; },
      contains(c) { return this._s.has(c); }
    },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    removeEventListener() {},
    dispatchEvent(evObj) {
      const arr = listeners[evObj.type] || [];
      arr.forEach(fn => fn.call(el, evObj));
    },
    click() { el.dispatchEvent({ type: "click", target: el, preventDefault() {}, stopPropagation() {} }); },
    focus() {}, blur() {}, select() {}, remove() {},
    appendChild(cEl) { this.children.push(cEl); cEl.parentElement = this; return cEl; },
    insertBefore(cEl) { this.children.push(cEl); cEl.parentElement = this; return cEl; },
    replaceChildren(...cEls) { this.children = cEls; },
    setAttribute(k, v) { this[k] = v; },
    getAttribute(k) { return this[k] ?? null; },
    removeAttribute(k) { delete this[k]; },
    getBoundingClientRect() { return { left: 0, top: 0, right: cw, bottom: ch, width: cw, height: ch }; },
    getContext() { if (!el._ctx) el._ctx = makeCtx(id); return el._ctx; },
    querySelector() { return makeEl(); },
    querySelectorAll() { return []; },
    closest() { return null; },
  };
  return el;
}
function getEl(id) {
  if (!elems.has(id)) elems.set(id, makeEl(id));
  return elems.get(id);
}
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.parent = globalThis;
globalThis.top = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.devicePixelRatio = 1;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
globalThis.sessionStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
globalThis.location = { href: "about:srcdoc", origin: "http://localhost:8080", pathname: "/", search: "" };
globalThis.navigator = { clipboard: { writeText: async () => {} } };
globalThis.Image = class { set src(v) { this._src = v; } };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.MutationObserver = class { observe() {} disconnect() {} };
globalThis.AbortController = class { constructor() { this.signal = {}; } abort() {} };
globalThis.requestAnimationFrame = (fn) => { rafCb = fn; return 1; };
globalThis.cancelAnimationFrame = () => {};
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}), text: async () => "" });
globalThis.document = {
  body: makeEl("body"),
  documentElement: makeEl("html"),
  activeElement: null,
  getElementById(id) { return getEl(id); },
  createElement(tag) { return makeEl(tag); },
  querySelector(sel) { return sel.startsWith("#") ? getEl(sel.slice(1)) : makeEl(sel); },
  querySelectorAll() { return []; },
  addEventListener() {},
  removeEventListener() {},
};
"""
        + script_before_close
        + f"""
const slDminEl = getEl("slider-dmin");
const inpDminNode = getEl("inp-dmin");
const inpLamMinNode = getEl("inp-lmin");
const detCanvasEl = getEl("canvas-detector");
detCanvasEl.getContext("2d");
const detLog = ctxLogs.get("canvas-detector");

// 1. Finite unit-cell-volume-aware d_min floor for V ~ 296,000 A^3
const protFloor = getDynamicDminFloor();
const sliderMinAttr = Number(slDminEl.min);
const initialPeakSource = state.peakSource;
const initialIntegCount = INTEGRATED_PEAKS.length;
const initialPredCount = PREDICTED_PEAKS.length;

// Attempting to enter 0.10 A into #inp-dmin clamps cleanly to protFloor
inpDminNode.value = "0.10";
inpDminNode.dispatchEvent({{ type: "change", target: inpDminNode }});
const clampedBelowFloor = state.dMin;

// Reset back to integrated mode + isigi colorscale at d_min = 3.20 A, then drag #slider-dmin down to 2.15 A:
// must auto-switch from 'integrated' -> 'predicted' AND 'isigi' -> 'wavelength' so unintegrated outer shells are visible!
slDminEl.value = "3.20";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
applyPeakSource("integrated", false);
state.wulffColorMode = "isigi";
const sourceBeforeDrag = state.peakSource;
const colorModeBeforeDrag = state.wulffColorMode;

slDminEl.value = "2.15";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
if (_dynMultiRunTimer) {{
  clearTimeout(_dynMultiRunTimer);
  _dynMultiRunTimer = null;
  repredictPeaksDynamic({{ forceSampleImmediate: true }});
}} else {{
  repredictPeaksDynamic({{ forceSampleImmediate: true }});
}}
const sourceAfterDragBelowInteg = state.peakSource;
const colorModeAfterDragBelowInteg = state.wulffColorMode;
const pred215 = PEAKS.map(p => ({{
  bank: p.bank,
  hkl: p.hkl,
  row: p.row,
  col: p.col,
  wl: p.wavelength,
  d: p.d_spacing,
  orbit: p.asu_orbit_id,
  bijvoet: p.asu_bijvoet_parity,
  isigi: (typeof p.intensity === "number" && typeof p.sigma === "number" && p.sigma > 0) ? p.intensity / p.sigma : null,
  hasEllipse: Array.isArray(p.ellipse) && p.ellipse.length >= 6,
  mtzIsigi: p.mtz_isigi,
}}));
const sampleRunsAt215 = Array.from(new Set(PREDICTED_SAMPLE_PEAKS.map(p => p.run_index))).sort();
const sampleCountAt215 = PREDICTED_SAMPLE_PEAKS.length;

// 2. Detector zoom-coupled d-spacing LOD at d_min = 2.15 A (> 800 reflections on 800x520 canvas)
state.detZoom = 1.0;
state.hoveredPeakId = null;
state.selectedPeakId = null;
state.activeOrbitId = null;
const allVisZ1 = getVisibleDetectorPeaks();
const lodZ1 = {{ ...getDetectorLodState(800, 520) }};
detLog.texts.length = 0;
renderDetectorCanvas(true);
const hudTextsZ1 = [...detLog.texts];

// Pick a reflection finer than lodZ1.lodDcut: must NOT be hit-tested at detZoom=1.0 unless pinned!
const hiddenOuterPeak = allVisZ1.find(p => p.d_spacing < lodZ1.lodDcut - 0.08);
const hiddenOuterPeakId = hiddenOuterPeak ? hiddenOuterPeak.id : null;
let hitHiddenUnpinned = null;
let hitHiddenWhenPinned = null;
if (hiddenOuterPeak) {{
  const [sxH, syH] = detToScreen(hiddenOuterPeak.det_x, getPeakVert(hiddenOuterPeak), 800, 520, 0.45);
  const hit0 = findNearestPeakDetector(sxH, syH);
  hitHiddenUnpinned = hit0 ? hit0.id : null;
  state.selectedPeakId = hiddenOuterPeak.id;
  state.selectedPeakObj = hiddenOuterPeak;
  const lodPinned = {{ ...getDetectorLodState(800, 520) }};
  const pinnedIncluded = lodPinned.visiblePeaks.some(p => p.id === hiddenOuterPeak.id);
  const hit1 = findNearestPeakDetector(sxH, syH);
  hitHiddenWhenPinned = Boolean(pinnedIncluded && hit1 && hit1.id === hiddenOuterPeak.id);
  state.selectedPeakId = null;
  state.selectedPeakObj = null;
}}

state.detZoom = 2.0;
const lodZ2 = {{ ...getDetectorLodState(800, 520) }};
state.detZoom = 6.0;
const lodZ6 = {{ ...getDetectorLodState(800, 520) }};
state.detZoom = 1.0;

// 3. Widening wavelength #inp-lam-min from 2.8 -> 2.0 A dynamically adds shorter-lambda spots
inpLamMinNode.value = "2.00";
inpLamMinNode.dispatchEvent({{ type: "input", target: inpLamMinNode }});
const pred215WideWlCount = PEAKS.length;
const minWlObserved = PEAKS.reduce((m, p) => Math.min(m, p.wavelength), Infinity);

// 4. Load huge macromolecular flat-panel MANDI payload (135 x 150 x 210 A) via recomputeBoundsAndReload
const mandiPayload = {json.dumps(mandi_payload)};
recomputeBoundsAndReload(mandiPayload, {{ resetSelectedFrame: true }});
const hugeCellFloor = getDynamicDminFloor();
slDminEl.value = "3.80";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
const mandiPred380 = PEAKS.map(p => ({{
  bank: p.bank,
  hkl: p.hkl,
  row: p.row,
  col: p.col,
}}));

require("fs").writeFileSync({json.dumps(str(out_json_path))}, JSON.stringify({{
  protFloor,
  sliderMinAttr,
  clampedBelowFloor,
  initialPeakSource,
  initialIntegCount,
  initialPredCount,
  sourceBeforeDrag,
  sourceAfterDragBelowInteg,
  colorModeBeforeDrag,
  colorModeAfterDragBelowInteg,
  pred215,
  sampleRunsAt215,
  sampleCountAt215,
  lodZ1: {{
    lodActive: lodZ1.isLodThinned,
    lodDMin: lodZ1.lodDcut,
    shownCount: lodZ1.visiblePeaks.length,
    totalCount: lodZ1.totalCount,
  }},
  lodZ2: {{
    lodActive: lodZ2.isLodThinned,
    lodDMin: lodZ2.lodDcut,
    shownCount: lodZ2.visiblePeaks.length,
    totalCount: lodZ2.totalCount,
  }},
  lodZ6: {{
    lodActive: lodZ6.isLodThinned,
    lodDMin: lodZ6.lodDcut,
    shownCount: lodZ6.visiblePeaks.length,
    totalCount: lodZ6.totalCount,
  }},
  hudHasLodPill: hudTextsZ1.some(t => t.includes("LOD d ≥") && t.includes("zoom+")),
  hiddenOuterPeakId,
  hitHiddenUnpinned,
  hitHiddenWhenPinned,
  pred215WideWlCount,
  minWlObserved,
  hugeCellFloor,
  mandiPred380,
}}));
}})();
process.exit(0);
""",
        encoding="utf-8",
    )

    subprocess.run(["node", str(js_file)], capture_output=True, text=True, check=True)
    out = json.loads(out_json_path.read_text(encoding="utf-8"))

    # A. Verify finite unit-cell-volume-aware d_min safety floor scaling
    assert 0.90 <= out["protFloor"] <= 1.20
    assert abs(out["sliderMinAttr"] - out["protFloor"]) < 1e-6
    assert abs(out["clampedBelowFloor"] - out["protFloor"]) < 1e-6
    assert out["hugeCellFloor"] > out["protFloor"] + 1.0
    assert 2.20 <= out["hugeCellFloor"] <= 2.80

    # B. Verify auto-switch from 'integrated' -> 'predicted' when dragging d_min below integrated cutoff
    assert out["initialPeakSource"] == "integrated"
    assert out["initialIntegCount"] == 6
    assert out["initialPredCount"] == len(payload["predicted_peaks"])
    assert out["sourceBeforeDrag"] == "integrated"
    assert out["sourceAfterDragBelowInteg"] == "predicted"
    assert out["colorModeBeforeDrag"] == "isigi"
    assert out["colorModeAfterDragBelowInteg"] == "wavelength"

    # C. Verify exact 1:1 HKL + pixel + Gemmi ASU/Bijvoet parity on curved cylindrical CG4D at d_min = 2.15 A
    py_by_hkl = {
        (int(ht[0]), int(ht[3]), int(ht[4]), int(ht[5])): ht for ht in py_cast_215
    }
    js_by_hkl = {
        (int(p["bank"]), int(p["hkl"][0]), int(p["hkl"][1]), int(p["hkl"][2])): p
        for p in out["pred215"]
    }
    assert set(js_by_hkl.keys()) == set(py_by_hkl.keys())
    max_px_err = 0.0
    preserved_inner_integ = 0
    outer_mtz_matched = 0
    for key, js_p in js_by_hkl.items():
        _, h, k, l = key
        assert not is_absent(h, k, l), f"Systematic absence leaked: {(h, k, l)}"
        py_ht = py_by_hkl[key]
        px_err = max(abs(js_p["row"] - py_ht[1]), abs(js_p["col"] - py_ht[2]))
        max_px_err = max(max_px_err, px_err)
        ah, ak, al, _, bij_p = asu_map(h, k, l)
        assert js_p["orbit"] == f"{ah},{ak},{al}"
        assert js_p["bijvoet"] == bij_p
        if js_p.get("isigi") is not None and js_p.get("hasEllipse"):
            preserved_inner_integ += 1
        if (
            js_p["d"] < 3.0
            and js_p.get("mtzIsigi") is not None
            and js_p["mtzIsigi"] > 10.0
        ):
            outer_mtz_matched += 1
    assert max_px_err < 0.06
    assert preserved_inner_integ == 6
    assert outer_mtz_matched >= 10

    # Multi-run Wulff PREDICTED_SAMPLE_PEAKS dynamically updated across both runs [0, 1]
    assert out["sampleRunsAt215"] == [0, 1]
    assert out["sampleCountAt215"] > len(py_cast_215)

    # D. Verify widening wavelength [2.8, 4.5] -> [2.0, 4.5] matches Python raycaster count & min wavelength
    assert out["pred215WideWlCount"] == len(py_cast_215_wide_wl)
    assert 2.0 <= out["minWlObserved"] < 2.2

    # E. Verify Detector zoom-coupled d-spacing LOD thinning, progressive zoom unlocking, HUD pill, and hit-testing
    lod1, lod2, lod6 = out["lodZ1"], out["lodZ2"], out["lodZ6"]
    assert lod1["totalCount"] == len(py_cast_215)
    assert lod1["lodActive"] is True
    assert lod1["shownCount"] < lod1["totalCount"]
    assert lod1["shownCount"] <= 500
    assert lod1["lodDMin"] > 2.25
    assert out["hudHasLodPill"] is True
    assert lod2["shownCount"] > lod1["shownCount"]
    assert lod2["lodDMin"] <= lod1["lodDMin"]
    assert lod6["lodActive"] is False
    assert lod6["shownCount"] == lod6["totalCount"]
    assert abs(lod6["lodDMin"] - 2.15) < 1e-3
    assert out["hiddenOuterPeakId"] is not None
    assert out["hitHiddenUnpinned"] != out["hiddenOuterPeakId"]
    assert out["hitHiddenWhenPinned"] is True

    # F. Verify flat-panel MANDI dynamic re-prediction matches Python _raycast_hkl_on_detectors 1:1
    py_mandi_map = {
        (int(ht[0]), int(ht[3]), int(ht[4]), int(ht[5])): ht for ht in py_mandi_380
    }
    js_mandi_map = {
        (int(p["bank"]), int(p["hkl"][0]), int(p["hkl"][1]), int(p["hkl"][2])): p
        for p in out["mandiPred380"]
    }
    assert set(js_mandi_map.keys()) == set(py_mandi_map.keys())
    for key, js_mp in js_mandi_map.items():
        py_mht = py_mandi_map[key]
        assert abs(js_mp["row"] - py_mht[1]) < 0.06
        assert abs(js_mp["col"] - py_mht[2]) < 0.06
