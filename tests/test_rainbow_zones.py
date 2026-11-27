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

    # Verify reduction_settings.json wavelength defaults, dual-knob [ and ] wavelength interval slider,
    # Unit cell / Params button (without λ_min/λ_max/d_min inputs), d_min slider, and I/sigma color mode
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
    assert 'id="wl-interval-slider"' in html_doc
    assert 'id="slider-wl-min"' in html_doc
    assert 'id="slider-wl-max"' in html_doc
    assert 'id="wl-knob-min"' in html_doc and ">[</button>" in html_doc
    assert 'id="wl-knob-max"' in html_doc and ">]</button>" in html_doc
    assert "Unit Cell..." in html_doc
    assert 'id="mtz-status-badge"' in html_doc
    assert "updateMtzStatusBadge" in html_doc
    assert 'id="inp-lmin"' not in html_doc
    assert 'id="inp-lmax"' not in html_doc
    assert 'id="inp-dmin"' not in html_doc
    assert 'id="slider-dmin"' in html_doc
    assert 'id="slider-dmin" min="0.50" max="5.00" step="0.05"' in html_doc
    assert 'id="sel-spot-color"' in html_doc
    assert "max-width: 158px;" in html_doc
    assert '<option value="isigi">Color: I/σ</option>' in html_doc
    assert '<option value="fobs">Color: |F_obs|²</option>' in html_doc
    assert '<option value="mtz_fcalc">Color: |F_calc|²</option>' in html_doc
    assert '<option value="ipred_isigi">Color: I_pred/σ</option>' in html_doc
    assert 'id="slider-color-emphasis"' in html_doc
    assert 'id="slider-parallax"' not in html_doc
    assert 'id="btn-peaks-mtz"' not in html_doc
    assert "hasIntegratedIsigi" in html_doc
    assert "hasMtzFcalcSq" in html_doc
    assert "hasIpredIsigi" in html_doc
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

    sym_trig_h = resolve_space_group_symmetry("R -3 m")
    assert sym_trig_h["setting"] == "trigonal"
    assert sym_trig_h["linked_lengths"] == {"b": "a"}
    assert sym_trig_h["fixed_angles"] == {
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
    }
    from subhkl.viz.rainbow_zones import _build_space_group_symmetry_lookup_json

    sg_groups = json.loads(_build_space_group_symmetry_lookup_json())
    assert "r3" in sg_groups["trigonal"]
    assert "r-3m" in sg_groups["trigonal"]
    assert "r3" not in sg_groups["trigonal_r"]
    assert "r-3m" not in sg_groups["trigonal_r"]
    assert "r3:r" in sg_groups["trigonal_r"]
    assert "r-3m:r" in sg_groups["trigonal_r"]

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
    assert payload_mtz["default_color_mode"] == "mtz_fcalc"
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
    assert doc_mtz["default_color_mode"] == "mtz_fcalc"
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
    #    - Warm Geyser-inspired quantile-normalized divergent isigiToColor scale
    #      (Sea-Teal -> Champagne-Gold median -> Tangerine -> Flame-Vermilion) + live gradient slider.
    #    - Two-source peak toggle ('predicted' | 'integrated') with .mtz I/σ data-column coloring.
    html_doc = render_rainbow_html(payload_mtz)
    assert 'accept=".h5,.hdf5,.nxs,.mtz"' in html_doc
    assert 'id="peak-source-group"' in html_doc
    assert 'id="btn-peaks-mtz"' not in html_doc
    assert 'id="btn-wulff-friedel"' in html_doc
    assert "Color: I/σ</option>" in html_doc
    assert "Color: |F_obs|²</option>" in html_doc
    assert "Color: |F_calc|²</option>" in html_doc
    assert "Color: I_pred/σ</option>" in html_doc
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
        "computeQuantileScaleBounds",
        "hasExternalMtzOverlay",
        "hasCarelessMergedState",
        "peakFobsSqRaw",
        "peakFobsSq",
        "getEffectiveCarelessScaleProfile",
        "interpCarelessProfileSeries",
        "estimateCarelessScaleForPeak",
        "estimateCarelessSnrScaleForPeak",
        "getMtzToCarelessIScaleFactor",
        "peakMtzFcalcSq",
        "peakIpredIsigi",
        "isMtzColorMode",
        "isQuantileSnrColorMode",
        "getPeakDataAvailability",
        "peakUnscaledIsigi",
        "peakMtzIsigi",
        "hasIntegratedIsigi",
        "hasFobsSq",
        "hasMtzIsigi",
        "hasMtzFcalcSq",
        "hasIpredIsigi",
        "preferredFallbackColorMode",
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
const _baselineMeasCache = {{
  carelessIAsuLookup: DATA.careless_i_asu_lookup || null,
  carelessFAsuLookup: DATA.careless_f_asu_lookup || null,
  mtzFAsuLookup: DATA.mtz_f_asu_lookup || null,
}};
let _mtzCarelessAlignCache = {{ sig: "", kI: 1.0 }};
let _jsCarelessProfileCache = {{ sig: "", prof: null }};
let _peakDataAvailabilitySig = "";
let _peakDataAvailability = {{
  hasIsigi: false,
  hasFobsSq: false,
  hasMtzIsigi: false,
  hasMtzFcalcSq: false,
  hasIpredIsigi: false,
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
  wulffColorMode: DATA.default_color_mode || "mtz_fcalc",
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
function isPeakVisible(p) {{ return state.wulffColorMode !== "mtz_fcalc" || Number.isFinite(peakMtzFcalcSq(p)); }}
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

// Warm Geyser-inspired 0–100th percentile (empirical CDF / inverse CDF) divergent I/σ scale:
// Sea-Teal (-3.0) -> Champagne-Gold median (+3.0) -> Solar Tangerine -> Flame-Vermilion (+40.0)
state.colorPercentile = 0.50;
state.colorCrossover = 0.38;
const testBounds = {{ min: -3.0, q25: 0.5, median: 3.0, q75: 9.0, max: 40.0 }};
const colMissing = isigiToColor(NaN, 1.0, testBounds);
const colLowPole = isigiToColor(-3.0, 1.0, testBounds);
const colBelowMed = isigiToColor(0.0, 1.0, testBounds);
const colMedGold = isigiToColor(3.0, 1.0, testBounds);
const colWarmMid = isigiToColor(14.0, 1.0, testBounds);
const colFlameHi = isigiToColor(150.0, 1.0, testBounds);

// Percentile slider knob: dragging right (0.84 = P84%) moves midpoint up so moderate +8σ stays closer to Champagne-Gold;
// dragging left (0.12 = P12%) moves midpoint down so +8σ shifts strongly toward Tangerine/Flame-Vermilion
state.colorPercentile = 0.84;
const colMidRightKnob = isigiToColor(8.0, 1.0, testBounds);
state.colorPercentile = 0.12;
const colMidLeftKnob = isigiToColor(8.0, 1.0, testBounds);
state.colorPercentile = 0.50;

const gradIsigi = buildColorscaleGradientSpec("isigi", 0.50);
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

// Peak source toggle ('predicted' <-> 'integrated') switches shape (circle <-> ellipse)
// while sharing the unified warm divergent I/σ colorscale (and distinct scaleToColor per source)
applyPeakSource("predicted", false);
const nPredFrame = PEAKS.length;
const nPredWulff = getActiveWulffPeaks().length;
const predShape = (state.peakSource === "integrated" && Array.isArray(PEAKS[0]?.ellipse)) ? "ellipse" : "circle";
const hasMtzCol = hasMtzFcalcSq();
const hasPredCrossLinkedIsigi = hasIntegratedIsigi();
const mtzColStrokePred = peakStrokeColor(PEAKS.find(p => Number.isFinite(peakMtzFcalcSq(p))), 1.0);
const scalePredTop = scaleToColor(10.0, 1.0);
const gradMtzIsigi = buildColorscaleGradientSpec("mtz_fcalc", 0.38);

applyPeakSource("integrated", false);
const nIntegFrame = PEAKS.length;
const nIntegWulff = getActiveWulffPeaks().length;
const integShape = (state.peakSource === "integrated" && Array.isArray(PEAKS[0]?.ellipse) && PEAKS[0].ellipse.length >= 6) ? "ellipse" : "circle";
const mtzColStrokeInteg = peakStrokeColor(PEAKS.find(p => Number.isFinite(peakMtzFcalcSq(p))), 1.0);
const scaleIntegTop = scaleToColor(10.0, 1.0);

console.log(JSON.stringify({{
  chiChecks,
  friedelNeg10AtPos10,
  colMissing, colLowPole, colBelowMed, colMedGold, colWarmMid, colFlameHi,
  scalePredTop, scaleIntegTop, gradMtzIsigi,
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

        _, _, _, a_nan = _parse_rgba(js_out["colMissing"])
        r_lo, g_lo, b_lo, a_lo = _parse_rgba(js_out["colLowPole"])
        r_bm, _g_bm, _b_bm, a_bm = _parse_rgba(js_out["colBelowMed"])
        r_md, g_md, b_md, a_md = _parse_rgba(js_out["colMedGold"])
        r_wm, g_wm, b_wm, a_wm = _parse_rgba(js_out["colWarmMid"])
        r_hi, g_hi, b_hi, a_hi = _parse_rgba(js_out["colFlameHi"])
        _, g_right_knob, b_right_knob, _ = _parse_rgba(js_out["colMidRightKnob"])
        _, g_left_knob, b_left_knob, _ = _parse_rgba(js_out["colMidLeftKnob"])

        # Missing I/σ is transparent; all finite I/σ values render opaque along warm Sea-Teal -> Champagne-Gold -> Flame-Vermilion
        assert a_nan == 0.0
        assert (
            a_lo == 1.0 and a_bm == 1.0 and a_md == 1.0 and a_wm == 1.0 and a_hi == 1.0
        )
        assert (r_lo, g_lo, b_lo) == (20, 184, 166)
        assert r_lo < r_bm < r_md
        assert (r_md, g_md, b_md) == (254, 232, 170)
        assert r_wm > 235 and g_hi < g_wm < g_md and b_hi < b_wm < b_md
        assert (r_hi, g_hi, b_hi) == (232, 56, 24)
        # Dragging crossover knob right keeps moderate +8σ closer to warm Champagne-Gold (higher green/blue);
        # dragging left shifts +8σ more strongly toward Tangerine/Flame-Vermilion (lower green/blue)
        assert g_right_knob > g_left_knob + 25 and b_right_knob > b_left_knob + 25
        # Live gradient CSS stops + lo · mid · hi quantile readouts
        assert js_out["gradIsigi"]["gradient"].startswith("linear-gradient(90deg,")
        assert "rgba(20, 184, 166, 1.000) 0.0%" in js_out["gradIsigi"]["gradient"]
        assert "rgba(254, 232, 170, 1.000) 50.0%" in js_out["gradIsigi"]["gradient"]
        assert "rgba(232, 56, 24, 1.000) 100.0%" in js_out["gradIsigi"]["gradient"]
        assert "·" in js_out["gradIsigi"]["label"]
        assert "·" in js_out["gradMtzIsigi"]["label"]
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
        assert js_out["mtzColStrokeInteg"].startswith("rgba(")
        assert js_out["scalePredTop"] != js_out["scaleIntegTop"]
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
    assert out["default_color_mode"] == "mtz_fcalc"
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
        "peakMtzIsigi",
        "isSameReflection",
        "isPeakHighlightedTarget",
        "isPeakVisible",
        "isWulffSnrCutActive",
        "getActiveWulffPeaks",
        "_peakSnrScore",
        "_computePeakLodPriority",
        "getWulffPoolCache",
        "buildWulffSphericalBVH",
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
const _wulffBvhCache = {{
  sig: "",
  firstRef: null,
  bvh: null,
}};
const _wulffLodCache = {{
  sig: "",
  firstRef: null,
  value: null,
}};
let _wulffPrevRenderedPeaks = [];
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
// Add a tight 12-peak high-resolution cluster inside one octant near the front pole (0.08, 0.08, 0.9936)
// so 1x BVH merges it into 1-2 landmarks, 3.5x splits partially, and 14.5x resolves all 12 pop-free!
for (let k = 0; k < 12; k++) {{
  const ang = (k / 12.0) * 2.0 * Math.PI;
  const rad = 0.003 + 0.012 * ((k % 4) / 3.0);
  const qx = 0.08 + rad * Math.cos(ang), qy = 0.08 + rad * Math.sin(ang);
  const qz = Math.sqrt(1.0 - qx * qx - qy * qy);
  const q = [qx, qy, qz];
  SAMPLE_PEAKS.push({{
    id: 6000 + k,
    hkl: [8000 + k, 2, 1],
    asu_hkl: [8000 + k, 2, 1],
    d_spacing: 1.45 + 0.25 * k,
    wavelength: 2.4,
    intensity: 12.0,
    sigma: 2.0,
    q_hat: q,
    q_hat_sample: q,
    q_hat_cryst: q,
    run_index: 2,
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

// Verify lowering dMin monotonically populates more Wulff cells at 1x zoom
state.dMin = 4.2;
const dMin42Count = getWulffLodState(false).visiblePeaks.length;
state.dMin = 2.6;
const dMin26Count = getWulffLodState(false).visiblePeaks.length;
state.dMin = 1.4;
const lodZoom1 = getWulffLodState(false);

// High-resolution BVH-suppressed peak (d < 2.0A) must ALWAYS survive zoom-out LOD thinning when hovered
const shownZoom1Ids = new Set(lodZoom1.visiblePeaks.map(p => p.id));
const hiResTarget = pool1.uniquePeaks.find(p => !shownZoom1Ids.has(p.id) && p.d_spacing < 2.0);
state.hoveredPeakId = hiResTarget.id;
state.hoveredPeakObj = hiResTarget;
const lodWithHover = getWulffLodState(false);
const hoverKept = lodWithHover.visiblePeaks.some(p => isSameReflection(p, hiResTarget));
state.hoveredPeakId = null;
state.hoveredPeakObj = null;

// Render at wulffZoom=1.0 -> draws & emits zoom+ badge AFTER restore(); immediate 2nd renderStereoInset() skips redraw
ops.length = 0;
renderStereoInset();
const firstDrawOps = ops.length;
const fillTextsZoom1 = ops.filter(o => o[0] === "fillText").map(o => o[1]);
const badgeOp = ops.find(o => o[0] === "fillText" && o[1].includes("zoom+"));

ops.length = 0;
renderStereoInset();
const secondIdleOps = ops.length;

// Zoom in to 3.5x and 14.5x on the front-pole cluster (yaw=0, pitch=0) -> pop-free subdivision + full resolution
const coreAt = (peaks) => new Set(
  peaks.filter(p => p.id >= 6000 && p.id < 6012).map(p => p.id)
);
const coreZ1 = coreAt(lodZoom1.visiblePeaks);
state.wulffZoom = 3.5;
const lodZoom25 = getWulffLodState(false);
const coreZ25 = coreAt(lodZoom25.visiblePeaks);
state.wulffZoom = 14.5;
const lodZoom15 = getWulffLodState(false);
const coreZ15 = coreAt(lodZoom15.visiblePeaks);
let polarPopped = 0;
for (const id of coreZ1) if (!coreZ25.has(id)) polarPopped++;
for (const id of coreZ25) if (!coreZ15.has(id)) polarPopped++;

console.log(JSON.stringify({{
  rawActiveLen: rawActive.length,
  poolDedupLen: pool1.uniquePeaks.length,
  dupCount: dupMatches.length,
  dupPrefersActiveFrame: dupMatches.length === 1 && dupMatches[0].run_index === 2,
  dMin42Count,
  dMin26Count,
  zoom1Count: lodZoom1.visiblePeaks.length,
  zoom1Thinned: lodZoom1.totalUnique - lodZoom1.visiblePeaks.length,
  zoom1Dcut: lodZoom1.lodDcut,
  zoom25CoreCount: coreZ25.size,
  zoom1CoreCount: coreZ1.size,
  zoom15CoreCount: coreZ15.size,
  zoom15Unthinned: !lodZoom15.isLodThinned && lodZoom15.visiblePeaks.length === lodZoom15.inCapCount && lodZoom15.inCapCount > 0,
  polarPopped,
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
    assert out["rawActiveLen"] == 1215
    assert out["poolDedupLen"] == 1213
    assert out["dupCount"] == 1
    assert out["dupPrefersActiveFrame"] is True
    assert out["dMin42Count"] < out["dMin26Count"] <= out["zoom1Count"]
    assert out["zoom1Thinned"] > 400
    assert out["zoom1Count"] < out["poolDedupLen"]
    assert 1 <= out["zoom1CoreCount"] <= 4
    assert out["zoom1CoreCount"] < out["zoom25CoreCount"] < out["zoom15CoreCount"]
    assert out["zoom15CoreCount"] == 12
    assert out["zoom15Unthinned"] is True
    assert out["polarPopped"] == 0
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
    bw_mandi = int(0.06 * min(int(cal_cfg["n"]), int(cal_cfg["m"])))
    for i_pk in range(len(h1)):
        if not (
            bw_mandi <= float(pr1[i_pk]) < int(cal_cfg["n"]) - 1 - bw_mandi
            and bw_mandi <= float(pc1[i_pk]) < int(cal_cfg["m"]) - 1 - bw_mandi
        ):
            continue
        key_h = (int(h1[i_pk]), int(k1[i_pk]), int(l1[i_pk]))
        if key_h not in expected_r1:
            expected_r1[key_h] = (
                float(pr1[i_pk]),
                float(pc1[i_pk]),
                float(lam1[i_pk]),
            )
            chosen_hkls.append(key_h)
            if len(chosen_hkls) >= 5:
                break
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
    assert overlaid["default_color_mode"] == "mtz_fcalc"
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
        {
            "id": 5,
            "run_index": 0,
            "bank": 36,
            "det_x": 115.0,
            "det_y": 0.0,
            "det_el": 0.0,
            "q_hat": [0.4, -0.3, -0.86],
            "q_hat_lab": [0.4, -0.3, -0.86],
            "stereographic": [0.22, -0.16],
            "hemisphere": 1,
            "wavelength": 2.95,
            "d_spacing": 2.45,
            "harmonic_order": 1,
            "harmonic_multiplicity": 1,
            "hkl": [7, -9, 1],
            "asu_hkl": [7, 9, 1],
            "asu_orbit_id": "7,9,1",
            "asu_friedel": False,
            "asu_bijvoet_parity": 1,
            "orbit_color_idx": 3,
            "zone_ids": [0],
            "indexed": True,
        },
    ]
    cross_run_bij_mate = {
        "id": 6,
        "run_index": 1,
        "bank": 36,
        "det_x": -120.0,
        "det_y": 0.0,
        "det_el": 0.0,
        "q_hat": [-0.4, 0.3, 0.86],
        "q_hat_lab": [-0.4, 0.3, -0.86],
        "stereographic": [-0.22, 0.16],
        "hemisphere": 1,
        "wavelength": 2.95,
        "d_spacing": 2.45,
        "harmonic_order": 1,
        "harmonic_multiplicity": 1,
        "hkl": [-7, -9, -1],
        "asu_hkl": [7, 9, 1],
        "asu_orbit_id": "7,9,1",
        "asu_friedel": True,
        "asu_bijvoet_parity": -1,
        "orbit_color_idx": 3,
        "zone_ids": [0],
        "indexed": True,
    }
    bij_sample_peaks = [*bij_peaks, cross_run_bij_mate]
    bij_payload.update(
        {
            "instrument": "CG4D",
            "space_group": "Fdd2",
            "wavelength_range": [2.0, 4.5],
            "d_min": 1.0,
            "peaks": bij_peaks,
            "predicted_peaks": bij_peaks,
            "sample_peaks": bij_sample_peaks,
            "predicted_sample_peaks": bij_sample_peaks,
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

// Hover peak #5 (7, -9, +1) -> 6th peak from left; Bijvoet mate exists ONLY in SAMPLE_PEAKS (run 1),
// exercising getActiveWulffSourceArray() inside updatePeakInspector without ReferenceError!
const [px5, py5] = inViewArcs[5];
detCanvasEl.dispatchEvent({ type: "mousemove", clientX: px5, clientY: py5, buttons: 0 });
const statusCrossRunBijHtml = getEl("inspector-box").innerHTML;

console.log(JSON.stringify({
  csGradientSet: Boolean(slCs && slCs.style["--cs-gradient"] && slCs.style["--cs-gradient"].startsWith("linear-gradient(")),
  ndDisplay,
  localClicked,
  hasBijBadge: statusBijHtml.includes("bijvoet-badge") && statusBijHtml.includes("±Bijvoet ×1 (frame)") && statusBijHtml.includes("I⁺"),
  hasCrossRunBijBadge: statusCrossRunBijHtml.includes("bijvoet-badge") && statusCrossRunBijHtml.includes("±Bijvoet ×1 (runs)"),
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
    assert out["hasCrossRunBijBadge"] is True
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
            margin_frac=0.03,
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
            margin_frac=0.03,
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
            margin_frac=0.03,
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
            margin_frac=0.06,
        ),
        b_big,
    )
    assert len(py_mandi_380) >= 15

    # Also build a genuine P 61 2 2 payload (79.91 x 79.91 x 238.0 A, gamma=120, V=1.32M A^3)
    # on MANDI with wavelength [0.65, 3.0] A to test direct Ewald-shell rod prediction down to d_min = 0.50 A!
    from subhkl.core.crystallography import cartesian_matrix_metric_tensor

    p6122_cell = [79.91, 79.91, 238.0, 90.0, 90.0, 120.0]
    b_p6122, _ = cartesian_matrix_metric_tensor(
        79.91, 79.91, 238.0, *np.deg2rad([90.0, 90.0, 120.0])
    )
    p6122_solve_h5 = tmp_path / "mandi_p6122.h5"
    with h5py.File(p6122_solve_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf["sample/space_group"] = "P 61 2 2"
        hf["sample/cell"] = np.array(p6122_cell, dtype=float)
        hf["sample/U"] = u_prot
        hf["sample/B"] = b_p6122
        hf["instrument/wavelength"] = np.array([0.65, 3.0], dtype=float)
    p6122_payload = build_rainbow_zone_payload(
        replay_h5=p6122_solve_h5,
        instrument="MANDI",
        d_min=3.50,
        wavelength_range=(0.65, 3.0),
        include_panel_images=False,
    )
    p6122_payload["panels"] = [
        pn for pn in p6122_payload["panels"] if int(pn["bank"]) in mandi_active_bids
    ]

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
const slWlMinEl = getEl("slider-wl-min");
const slWlMaxEl = getEl("slider-wl-max");
const wlKnobMinEl = getEl("wl-knob-min");
const wlKnobMaxEl = getEl("wl-knob-max");
const detCanvasEl = getEl("canvas-detector");
detCanvasEl.getContext("2d");
const detLog = ctxLogs.get("canvas-detector");

// 1. Unlocked 0.50 A d_min floor + 2M full-sphere pool crossover for V ~ 296,000 A^3
const protFloor = getDynamicDminFloor();
const protPoolFloor = getFullSpherePoolDminFloor();
const sliderMinAttr = Number(slDminEl.min);
const initialPeakSource = state.peakSource;
const initialIntegCount = INTEGRATED_PEAKS.length;
const initialPredCount = PREDICTED_PEAKS.length;

// Attempting to drag #slider-dmin to 0.10 A clamps cleanly to protFloor
slDminEl.value = "0.10";
slDminEl.dispatchEvent({{ type: "change", target: slDminEl }});
const clampedBelowFloor = state.dMin;

// Reset back to integrated mode + isigi colorscale at d_min = 3.20 A, then drag #slider-dmin down to 2.15 A:
// must PRESERVE 'integrated' mode + 'isigi' colorscale without inventing unintegrated peaks;
// then explicitly switching to 'predicted' shows all dynamically predicted 2.15 A reflections!
slDminEl.value = "3.20";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
applyPeakSource("integrated", false);
state.wulffColorMode = "isigi";
const sourceBeforeDrag = state.peakSource;
const colorModeBeforeDrag = state.wulffColorMode;

slDminEl.value = "2.15";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
const sourceAfterDragBelowInteg = state.peakSource;
const colorModeAfterDragBelowInteg = state.wulffColorMode;
const integCountAfterDragBelowInteg = PEAKS.length;

applyPeakSource("predicted", false);
if (_dynMultiRunTimer) {{
  clearTimeout(_dynMultiRunTimer);
  _dynMultiRunTimer = null;
  repredictPeaksDynamic({{ forceSampleImmediate: true }});
}} else {{
  repredictPeaksDynamic({{ forceSampleImmediate: true }});
}}
const sourceAfterExplicitPred = state.peakSource;
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

// 2. Detector gameplay Spherical BVH screen-density LOD at d_min = 2.15 A (> 800 reflections on 800x520 canvas)
state.detZoom = 1.0;
state.hoveredPeakId = null;
state.selectedPeakId = null;
state.activeOrbitId = null;
const allVisZ1 = getVisibleDetectorPeaks();
const lodZ1 = {{ ...getDetectorLodState(800, 520) }};
detLog.texts.length = 0;
renderDetectorCanvas(true);
const hudTextsZ1 = [...detLog.texts];

// Pick a BVH-suppressed reflection at detZoom=1.0: must NOT be hit-tested when unpinned, and MUST be hit-tested when pinned!
const shownSetZ1 = new Set(lodZ1.visiblePeaks.map(p => p.id));
const hiddenOuterPeak = allVisZ1.find(p => !shownSetZ1.has(p.id));
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
state.detZoom = 24.0;
const lodZ24 = {{ ...getDetectorLodState(800, 520) }};
state.detZoom = 1.0;

// 3. Widening wavelength #slider-wl-min ([ knob) from 2.8 -> 2.0 A dynamically adds shorter-lambda spots,
//    and keyboard ArrowLeft / ArrowRight on #wl-knob-max (] knob) adjusts state.lamMax cleanly
slWlMinEl.value = "2.00";
slWlMinEl.dispatchEvent({{ type: "input", target: slWlMinEl }});
const pred215WideWlCount = PEAKS.length;
const minWlObserved = PEAKS.reduce((m, p) => Math.min(m, p.wavelength), Infinity);
const lamMaxBeforeKey = state.lamMax;
wlKnobMaxEl.dispatchEvent({{ type: "keydown", key: "ArrowRight", shiftKey: false, preventDefault() {{}} }});
const lamMaxAfterArrowRight = state.lamMax;
wlKnobMaxEl.dispatchEvent({{ type: "keydown", key: "ArrowLeft", shiftKey: false, preventDefault() {{}} }});
if (Math.abs(lamMaxAfterArrowRight - (lamMaxBeforeKey + 0.05)) > 1e-4) {{
  throw new Error(`Expected ] knob ArrowRight to step lamMax by +0.05, got ${{lamMaxBeforeKey}} -> ${{lamMaxAfterArrowRight}}`);
}}
// Ensure [ knob cannot cross ] knob, and pointer-dragging [ + dblclick reset both work cleanly
slWlMinEl.value = "99.0";
slWlMinEl.dispatchEvent({{ type: "input", target: slWlMinEl }});
if (!(state.lamMin <= state.lamMax - 0.049)) {{
  throw new Error(`Expected [ knob to stop >=0.05 A below ] knob, got [${{state.lamMin}}, ${{state.lamMax}}]`);
}}
const wlWrapEl = getEl("wl-interval-slider");
wlWrapEl.dispatchEvent({{ type: "pointerdown", button: 0, target: wlKnobMinEl, clientX: 60, pointerId: 1, preventDefault() {{}} }});
wlWrapEl.dispatchEvent({{ type: "pointermove", clientX: 75 }});
wlWrapEl.dispatchEvent({{ type: "pointerup", pointerId: 1 }});
const lamMinAfterPointerDrag = state.lamMin;
if (!(lamMinAfterPointerDrag > 1.0 && lamMinAfterPointerDrag < 2.5)) {{
  throw new Error(`Unexpected lamMin after pointer drag on [ knob: ${{lamMinAfterPointerDrag}}`);
}}
wlWrapEl.dispatchEvent({{ type: "dblclick" }});
if (Math.abs(state.lamMin - 2.8) > 1e-4 || Math.abs(state.lamMax - 4.5) > 1e-4) {{
  throw new Error(`Expected dblclick to reset wavelength to [2.8, 4.5], got [${{state.lamMin}}, ${{state.lamMax}}]`);
}}
slWlMinEl.value = "2.00";
slWlMinEl.dispatchEvent({{ type: "input", target: slWlMinEl }});

// 4. Load huge macromolecular flat-panel MANDI payload (135 x 150 x 210 A) via recomputeBoundsAndReload
const mandiPayload = {json.dumps(mandi_payload)};
recomputeBoundsAndReload(mandiPayload, {{ resetSelectedFrame: true }});
const hugeCellFloor = getDynamicDminFloor();
const hugeCellPoolFloor = getFullSpherePoolDminFloor();
slDminEl.value = "3.80";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
const mandiPred380 = PEAKS.map(p => ({{
  bank: p.bank,
  hkl: p.hkl,
  row: p.row,
  col: p.col,
}}));

// 5. Verify P 6_1 22 (a=b=79.91, c=238.0) hexagonal symmetry + unlocked 0.50 A Ewald-shell rod predictor!
const p6122Payload = {json.dumps(p6122_payload)};
recomputeBoundsAndReload(p6122Payload, {{ resetSelectedFrame: true }});
const inpSgNode = getEl("inp-sg");
const inpANode = getEl("inp-a");
const inpCNode = getEl("inp-c");
inpSgNode.value = "P 6_1 22";
inpSgNode.dispatchEvent({{ type: "change", target: inpSgNode }});
const p6122Floor = getDynamicDminFloor();
const p6122PoolFloor = getFullSpherePoolDminFloor();
const p6122Gamma = parseFloat(getEl("inp-gamma").value);
slDminEl.value = "1.45";
slDminEl.dispatchEvent({{ type: "change", target: slDminEl }});
const p6122Allowed145 = state.dMin;

// Push P 6_1 22 all the way down to d_min = 0.50 A (44.2M full-sphere lattice points!).
// (a) Short-wavelength slice [0.74, 0.79] A (~18k reflections down to ~0.58 A): verify exact 1:1 Python/JS Ewald-shell parity.
const slWlMinNode = getEl("slider-wl-min");
const slWlMaxNode = getEl("slider-wl-max");
state.lamMin = 0.74;
state.lamMax = 0.79;
updateWavelengthLabels();
slDminEl.value = "0.50";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
const p6122Allowed050 = state.dMin;
const p6122PoolAt050IsNull = ensureDynamicHklPool(0.50) === null;
const p6122Pred050 = PEAKS.map(p => ({{
  bank: p.bank,
  hkl: p.hkl,
  row: p.row,
  col: p.col,
  d: p.d_spacing,
  wl: p.wavelength,
  det_x: p.det_x,
  det_y: p.det_y,
}}));

// (b) Wide white-beam [0.65, 3.00] A at d_min = 0.50 A (>150k hits on 2 banks!):
// Verify Tier-1 1x overview LOD (~300 spots) AND Tier-2 36x zoomed viewport-frustum streaming (<0.58 A spots!).
state.lamMin = 0.65;
state.lamMax = 3.00;
updateWavelengthLabels();
const t0Ewald = Date.now();
repredictPeaksDynamic({{ userAdjusted: true }});
const p6122EwaldMs = Date.now() - t0Ewald;
state.detZoom = 1.0;
state.detPanX = 0.0;
state.detPanY = 0.0;
const p6122LodZ1 = {{ ...getDetectorLodState(800, 520) }};
const p6122Tier1MinD = PEAKS.reduce((m, p) => Math.min(m, p.d_spacing || 99.0), 99.0);
const refBankPk = PEAKS.find(p => p.bank === 1) || PEAKS[0];
state.detZoom = 36.0;
if (refBankPk) {{
  setDetectorCenterAzimuthDeg(refBankPk.det_x);
  const bAct = getActiveBounds();
  const bxAct = getDetectorPlotBox(800, 520);
  const pyRef = getPeakVert(refBankPk);
  const y0Ref = bxAct.originY + (1.0 - (pyRef - bAct.minY) / Math.max(1e-6, bAct.maxY - bAct.minY)) * bxAct.plotH;
  state.detPanY = 260.0 - (260.0 + (y0Ref - 260.0) * state.detZoom);
}}
const p6122LodZ12Raw = getDetectorLodState(800, 520);
const p6122LodZ12 = {{
  isLodThinned: p6122LodZ12Raw.isLodThinned,
  visiblePeaks: p6122LodZ12Raw.visiblePeaks,
  inViewCount: p6122LodZ12Raw.inViewCount,
  totalCount: p6122LodZ12Raw.totalCount,
  minVisD: p6122LodZ12Raw.visiblePeaks.reduce((m, p) => Math.min(m, p.d_spacing || 99), 99),
}};

// Verify detector wheel zoom-in (cursor-anchored) and zoom-out (zero sideways azimuth drift!)
state.detZoom = 1.0;
setDetectorCenterAzimuthDeg(112.5);
state.detPanY = 0.0;
const azBeforeZoomIn = getCenteredDetectorAzimuthDeg();
// Cursor zoom-in at off-center point (mx=620, my=190): world point under cursor must stay pinned
const [wxBefore, wyBefore] = [620, 190];
const boxDet = getDetectorPlotBox(800, 520);
const boundsDet = getActiveBounds();
function screenXToWorldAz(sx) {{
  const x0 = 400 + (sx - 400 - state.detPanX) / state.detZoom;
  const u = (x0 - boxDet.originX) / boxDet.plotW;
  return boundsDet.minX + u * (boundsDet.maxX - boundsDet.minX);
}}
const cursorWorldAzBefore = screenXToWorldAz(wxBefore);
for (let step = 0; step < 8; step++) {{
  detCanvasEl.dispatchEvent({{
    type: "wheel",
    deltaMode: 0,
    deltaY: -45,
    ctrlKey: false,
    clientX: wxBefore,
    clientY: wyBefore,
    preventDefault() {{}},
  }});
}}
const zoomAfterIn = state.detZoom;
const cursorWorldAzAfterIn = screenXToWorldAz(wxBefore);
// Now zoom all the way out from zoomAfterIn -> 0.7x with cursor way off on the left (clientX = 95):
// centered detector azimuth MUST NOT drift sideways at any step!
const azAtPeakZoom = getCenteredDetectorAzimuthDeg();
let maxZoomOutAzDrift = 0.0;
for (let step = 0; step < 30; step++) {{
  detCanvasEl.dispatchEvent({{
    type: "wheel",
    deltaMode: 0,
    deltaY: 45,
    ctrlKey: false,
    clientX: 95,
    clientY: 410,
    preventDefault() {{}},
  }});
  const curAz = getCenteredDetectorAzimuthDeg();
  let diff = Math.abs(curAz - azAtPeakZoom);
  if (diff > 180.0) diff = 360.0 - diff;
  if (diff > maxZoomOutAzDrift) maxZoomOutAzDrift = diff;
}}
const zoomAfterOut = state.detZoom;
state.detZoom = 1.0;
state.detPanX = 0.0;
state.detPanY = 0.0;

// Restore d_min = 1.45 for workflow param check below
slDminEl.value = "1.45";
slDminEl.dispatchEvent({{ type: "change", target: slDminEl }});

// 6. Verify #inp-binning & slider-driven wavelength/d_min in collectWorkflowParams()
slWlMinEl.value = "2.25";
slWlMinEl.dispatchEvent({{ type: "input", target: slWlMinEl }});
slWlMaxEl.value = "4.35";
slWlMaxEl.dispatchEvent({{ type: "input", target: slWlMaxEl }});
const inpBinNode = getEl("inp-binning");
inpBinNode.value = "";
const wfCheck = collectWorkflowParams();
if (Math.abs(wfCheck.wavelength_min - 2.25) > 1e-4 || Math.abs(wfCheck.wavelength_max - 4.35) > 1e-4 || Math.abs(wfCheck.d_min - 1.45) > 1e-4) {{
  throw new Error(`collectWorkflowParams mismatch: ${{JSON.stringify(wfCheck)}}`);
}}
const paramsAutoBin = wfCheck.binning;
inpBinNode.value = "16";
const paramsBin16 = collectWorkflowParams().binning;
inpBinNode.value = "8";
const paramsBin8 = collectWorkflowParams().binning;
inpBinNode.value = "4";
const paramsBin4 = collectWorkflowParams().binning;

require("fs").writeFileSync({json.dumps(str(out_json_path))}, JSON.stringify({{
  protFloor,
  protPoolFloor,
  sliderMinAttr,
  clampedBelowFloor,
  initialPeakSource,
  initialIntegCount,
  initialPredCount,
  sourceBeforeDrag,
  sourceAfterDragBelowInteg,
  colorModeBeforeDrag,
  colorModeAfterDragBelowInteg,
  integCountAfterDragBelowInteg,
  sourceAfterExplicitPred,
  pred215,
  sampleRunsAt215,
  sampleCountAt215,
  lodZ1: {{
    lodActive: lodZ1.isLodThinned,
    lodDMin: lodZ1.lodDcut,
    shownCount: lodZ1.visiblePeaks.length,
    inViewCount: lodZ1.inViewCount,
    totalCount: lodZ1.totalCount,
  }},
  lodZ2: {{
    lodActive: lodZ2.isLodThinned,
    lodDMin: lodZ2.lodDcut,
    shownCount: lodZ2.visiblePeaks.length,
    inViewCount: lodZ2.inViewCount,
    totalCount: lodZ2.totalCount,
  }},
  lodZ6: {{
    lodActive: lodZ6.isLodThinned,
    lodDMin: lodZ6.lodDcut,
    shownCount: lodZ6.visiblePeaks.length,
    inViewCount: lodZ6.inViewCount,
    totalCount: lodZ6.totalCount,
  }},
  lodZ24: {{
    lodActive: lodZ24.isLodThinned,
    lodDMin: lodZ24.lodDcut,
    shownCount: lodZ24.visiblePeaks.length,
    inViewCount: lodZ24.inViewCount,
    totalCount: lodZ24.totalCount,
  }},
  hudHasLodPill: hudTextsZ1.some(t => t.includes("LOD ") && t.includes("d ≥") && t.includes("zoom+ to resolve")),
  hiddenOuterPeakId,
  hitHiddenUnpinned,
  hitHiddenWhenPinned,
  pred215WideWlCount,
  minWlObserved,
  hugeCellFloor,
  hugeCellPoolFloor,
  mandiPred380,
  p6122Floor,
  p6122PoolFloor,
  p6122Gamma,
  p6122Allowed145,
  p6122Allowed050,
  p6122PoolAt050IsNull,
  p6122EwaldMs,
  p6122Tier1MinD,
  p6122Pred050,
  p6122LodZ1: {{
    lodActive: p6122LodZ1.isLodThinned,
    shownCount: p6122LodZ1.visiblePeaks.length,
    totalCount: p6122LodZ1.totalCount,
  }},
  p6122LodZ12: {{
    lodActive: p6122LodZ12.isLodThinned,
    shownCount: p6122LodZ12.visiblePeaks.length,
    inViewCount: p6122LodZ12.inViewCount,
    totalCount: p6122LodZ12.totalCount,
    minVisD: p6122LodZ12.minVisD,
  }},
  azBeforeZoomIn,
  cursorWorldAzBefore,
  cursorWorldAzAfterIn,
  zoomAfterIn,
  zoomAfterOut,
  maxZoomOutAzDrift,
  paramsAutoBin,
  paramsBin16,
  paramsBin8,
  paramsBin4,
}}));
}})();
process.exit(0);
""",
        encoding="utf-8",
    )

    subprocess.run(["node", str(js_file)], capture_output=True, text=True, check=True)
    out = json.loads(out_json_path.read_text(encoding="utf-8"))

    # A. Verify unlocked 0.50 A d_min floor across all unit cells + 2M full-sphere pool crossover
    assert abs(out["protFloor"] - 0.50) < 1e-6
    assert 0.75 <= out["protPoolFloor"] <= 1.05
    assert abs(out["sliderMinAttr"] - 0.50) < 1e-6
    assert abs(out["clampedBelowFloor"] - 0.50) < 1e-6
    assert abs(out["hugeCellFloor"] - 0.50) < 1e-6
    assert 1.95 <= out["hugeCellPoolFloor"] <= 2.40
    assert out["p6122Gamma"] == 120.0
    assert abs(out["p6122Floor"] - 0.50) < 1e-6
    assert 1.40 <= out["p6122PoolFloor"] <= 1.43
    assert abs(out["p6122Allowed145"] - 1.45) < 1e-6
    assert abs(out["p6122Allowed050"] - 0.50) < 1e-6
    assert out["p6122PoolAt050IsNull"] is True
    assert out["paramsAutoBin"] is None
    assert out["paramsBin16"] == 16
    assert out["paramsBin8"] == 8
    assert out["paramsBin4"] == 4

    # Verify P6122 (79.91 x 79.91 x 238.0 A, gamma=120) direct Ewald-shell 1:1 parity at d_min = 0.50 A!
    from subhkl.core.crystallography import (
        cartesian_matrix_metric_tensor,
        generate_ewald_reflections,
        generate_reflections,
    )

    p6122_cell = [79.91, 79.91, 238.0, 90.0, 90.0, 120.0]
    u_p6122 = np.asarray(p6122_payload["u_matrix"], dtype=float)
    b_p6122, _ = cartesian_matrix_metric_tensor(
        79.91, 79.91, 238.0, *np.deg2rad([90.0, 90.0, 120.0])
    )
    ub_p6122 = u_p6122 @ b_p6122
    he, ke, le = generate_ewald_reflections(
        *p6122_cell,
        space_group="P 61 2 2",
        d_min=0.50,
        UB=ub_p6122,
        R_runs=[r0],
        ki=(0.0, 0.0, 1.0),
        wavelength_range=(0.74, 0.79),
    )
    py_p6122_050 = _cast_to_tuples(
        _raycast_hkl_on_detectors(
            np.column_stack([he, ke, le]),
            u_p6122,
            b_p6122,
            r0,
            mandi_dets,
            (0.74, 0.79),
            active_bids=mandi_active_bids,
            margin_frac=0.06,
        ),
        b_p6122,
    )
    assert len(py_p6122_050) > 1500
    assert min(ht[7] for ht in py_p6122_050) < 0.64
    py_p6122_keys = {
        (int(ht[0]), int(ht[3]), int(ht[4]), int(ht[5])): ht for ht in py_p6122_050
    }
    js_p6122_keys = {
        (int(p["bank"]), int(p["hkl"][0]), int(p["hkl"][1]), int(p["hkl"][2])): p
        for p in out["p6122Pred050"]
    }
    # Allow at most sub-1e-3 pixel float-rounding graze at exact [0, 256) edge
    sym_diff = set(js_p6122_keys.keys()) ^ set(py_p6122_keys.keys())
    assert len(sym_diff) <= 2
    common_keys = set(js_p6122_keys.keys()) & set(py_p6122_keys.keys())
    max_p6122_px_err = max(
        max(
            abs(js_p6122_keys[k]["row"] - py_p6122_keys[k][1]),
            abs(js_p6122_keys[k]["col"] - py_p6122_keys[k][2]),
        )
        for k in common_keys
    )
    assert max_p6122_px_err < 0.06
    assert out["p6122LodZ1"]["lodActive"] is True
    assert out["p6122LodZ1"]["shownCount"] <= 1200
    assert out["p6122LodZ1"]["totalCount"] > 100000
    assert out["p6122Tier1MinD"] < 0.70
    assert out["p6122LodZ12"]["inViewCount"] < out["p6122LodZ12"]["totalCount"]
    assert out["p6122LodZ12"]["shownCount"] > 50
    assert out["p6122LodZ12"]["shownCount"] == out["p6122LodZ12"]["inViewCount"]
    assert out["p6122LodZ12"]["minVisD"] < 0.66

    # Verify slabbed generate_reflections bit-for-bit parity when box_count > 1_000_000
    h_sl, k_sl, l_sl = generate_reflections(
        55.0, 80.0, 88.0, 90.0, 96.0, 90.0, space_group="I 2", d_min=1.35
    )
    assert len(h_sl) > 150_000
    assert np.all((h_sl + k_sl + l_sl) % 2 == 0)

    # Verify detector wheel zoom-in anchors under cursor and zoom-out has zero sideways azimuth drift!
    assert out["zoomAfterIn"] > 1.7
    assert out["zoomAfterOut"] == 0.7
    az_cursor_diff = abs(
        (out["cursorWorldAzAfterIn"] - out["cursorWorldAzBefore"] + 180.0) % 360.0
        - 180.0
    )
    assert az_cursor_diff < 1e-4
    assert out["maxZoomOutAzDrift"] < 1e-5

    # B. Verify 'integrated' mode is preserved when dragging d_min below integrated cutoff (no auto-switch!)
    assert out["initialPeakSource"] == "integrated"
    assert out["initialIntegCount"] == 6
    assert out["initialPredCount"] == len(payload["predicted_peaks"])
    assert out["sourceBeforeDrag"] == "integrated"
    assert out["sourceAfterDragBelowInteg"] == "integrated"
    assert out["integCountAfterDragBelowInteg"] == 6
    assert out["colorModeBeforeDrag"] == "isigi"
    assert out["colorModeAfterDragBelowInteg"] == "isigi"
    assert out["sourceAfterExplicitPred"] == "predicted"

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

    # E. Verify Detector Spherical BVH screen-density LOD thinning, zoom subdivision, HUD pill, and hit-testing
    lod1, lod2, lod6, lod24 = (
        out["lodZ1"],
        out["lodZ2"],
        out["lodZ6"],
        out["lodZ24"],
    )
    assert lod1["totalCount"] == len(py_cast_215)
    assert lod1["lodActive"] is True
    assert lod1["shownCount"] < lod1["totalCount"]
    assert lod1["shownCount"] <= 950
    assert lod1["lodDMin"] < 2.25
    assert lod2["shownCount"] > lod1["shownCount"]
    assert out["hudHasLodPill"] is True
    assert lod6["inViewCount"] < lod6["totalCount"]
    assert lod24["lodActive"] is False
    assert lod24["inViewCount"] < lod24["totalCount"]
    assert lod24["shownCount"] == lod24["inViewCount"]
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


def test_edge_mask_overlay_and_mandi_default_border(tmp_path):
    """MANDI defaults rel_border_width to 0.06 (6%), CG4D to 0.03 (3%), and Mask view renders the live edge frame."""
    import json
    import re
    import shutil
    import subprocess

    from subhkl.viz.rainbow_zones import (
        build_rainbow_zone_payload,
        default_rel_border_for_instrument,
        render_rainbow_html,
    )

    assert default_rel_border_for_instrument("MANDI") == 0.06
    assert default_rel_border_for_instrument("CG4D") == 0.03

    mandi_h5 = tmp_path / "mandi_edge_test.h5"
    with h5py.File(mandi_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf["images"] = np.ones((1, 256, 256), dtype=np.float32) * 10.0
        hf["bank_ids"] = np.array([20], dtype=np.int32)
        hf["instrument/wavelength"] = np.array([2.0, 4.0], dtype=np.float32)

    p_mandi = build_rainbow_zone_payload(
        replay_h5=mandi_h5, instrument="MANDI", include_panel_images=False
    )
    b20 = next(pn for pn in p_mandi["panels"] if pn["bank"] == 20)
    assert b20["n_rows"] == 256 and b20["n_cols"] == 256

    html_mandi = render_rainbow_html(p_mandi)
    assert 'id="lbl-mask-edge"' in html_mandi
    assert "Edge 6.0%" in html_mandi
    assert (
        'id="inp-border" type="number" min="0" max="0.25" step="0.005" value="0.06"'
        in html_mandi
    )

    if shutil.which("node"):
        script_blocks = re.findall(r"<script>([\s\S]*?)</script>", html_mandi)
        inline_js = script_blocks[-1]
        js_check = tmp_path / "edge_mask_inline.js"
        js_check.write_text(inline_js, encoding="utf-8")
        subprocess.run(["node", "--check", str(js_check)], check=True)

        fn_names = [
            "defaultRelBorderForInstrument",
            "getActiveRelBorderWidth",
            "getBorderMaskCanvas",
            "hasPixelMask",
            "syncPixelMaskUI",
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

        harness = f"""
const borderMaskCache = new Map();
const maskBankCanvases = new Map();
const PANELS = [{{ bank: 20, n_rows: 256, n_cols: 256 }}];
const DATA = {{ instrument: "MANDI", em_mask_summary: {{ available: false }} }};
const state = {{ showPixelMask: true }};
const domEls = {{
  "inp-border": {{ value: "0.06" }},
  "btn-pixel-mask": {{ style: {{}}, title: "", classList: {{ toggle() {{}} }} }},
  "mask-legend-row": {{ style: {{}} }},
  "lbl-mask-dark": {{ textContent: "" }},
  "lbl-mask-bright": {{ textContent: "" }},
  "lbl-mask-edge": {{ textContent: "" }},
}};
globalThis.document = {{
  getElementById: (id) => domEls[id] || null,
  createElement: () => {{
    const ops = [];
    const ctx = {{
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 1,
      clearRect() {{}},
      setLineDash() {{}},
      fillRect(...a) {{ ops.push(["fillRect", ...a]); }},
      strokeRect(...a) {{ ops.push(["strokeRect", ...a]); }},
    }};
    return {{ width: 0, height: 0, ops, getContext: () => ctx }};
  }},
}};

{chr(10).join(extracted)}

const c06 = getBorderMaskCanvas(PANELS[0], getActiveRelBorderWidth());
syncPixelMaskUI();
const edgeTxt06 = domEls["lbl-mask-edge"].textContent;
domEls["inp-border"].value = "0.08";
const c08 = getBorderMaskCanvas(PANELS[0], getActiveRelBorderWidth());
syncPixelMaskUI();
const edgeTxt08 = domEls["lbl-mask-edge"].textContent;

console.log(JSON.stringify({{
  w: c06.width,
  h: c06.height,
  topBandH06: c06.ops[0][4],
  topBandH08: c08.ops[0][4],
  edgeTxt06,
  edgeTxt08,
  btnOpacity: domEls["btn-pixel-mask"].style.opacity,
}}));
"""
        hf_js = tmp_path / "edge_mask_harness.js"
        hf_js.write_text(harness, encoding="utf-8")
        res = subprocess.run(
            ["node", str(hf_js)], check=True, capture_output=True, text=True
        )
        out = json.loads(res.stdout.strip())
        assert out["w"] == 256 and out["h"] == 256
        assert out["topBandH06"] == 15  # round(0.06 * 256) == 15 px
        assert out["topBandH08"] == 20  # round(0.08 * 256) == 20 px
        assert out["edgeTxt06"] == "Edge 6.0% (15px)"
        assert out["edgeTxt08"] == "Edge 8.0% (20px)"
        assert out["btnOpacity"] == "1"


def test_textures_only_fast_path_and_vectorized_flat_raycast(tmp_path):
    """Hi-Res textures_only=1 skips HKL/zone rebuilds, vectorized flat-panel raycast matches Detector 1:1, and status poll deduplicates overlay_seq."""
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request
    from scipy.spatial.transform import Rotation

    from subhkl.config import beamlines
    from subhkl.core.crystallography import (
        cartesian_matrix_metric_tensor,
        generate_reflections,
    )
    from subhkl.instrument.detector import Detector
    from subhkl.viz.nova_app.models.subhkl_model import SubhklGalaxyModel
    from subhkl.viz.nova_app.views.main_view import MainApp
    from subhkl.viz.rainbow_zones import (
        _raycast_hkl_on_detectors,
        build_rainbow_panel_textures_only,
        build_rainbow_zone_payload,
    )

    # 1. Vectorized flat-panel raycaster exact parity against Detector.reflections_mask + pixel_to_lab
    mandi_dets = {
        int(bk): Detector(dict(cfg)) for bk, cfg in list(beamlines["MANDI"].items())[:8]
    }
    cell = [68.0, 74.0, 92.0, 90.0, 90.0, 90.0]
    B_mat_raw, _ = cartesian_matrix_metric_tensor(*cell[:3], *np.deg2rad(cell[3:]))
    B_mat = np.asarray(B_mat_raw, dtype=float)
    U_mat = Rotation.from_euler("zyx", [19.0, -31.0, 44.0], degrees=True).as_matrix()
    R_mat = Rotation.from_euler("y", 17.5, degrees=True).as_matrix()
    RUB = R_mat @ U_mat @ B_mat
    sample_off = np.array([0.0012, -0.0008, 0.0021], dtype=float)
    margin = 0.06
    hkls = np.stack(
        generate_reflections(*cell, space_group="P 21 21 21", d_min=2.4),
        axis=1,
    ).astype(int)

    vec_out = _raycast_hkl_on_detectors(
        hkls,
        U_mat,
        B_mat,
        R_mat,
        mandi_dets,
        (2.0, 4.0),
        active_bids=set(mandi_dets.keys()),
        margin_frac=margin,
        sample_offset_lab=sample_off,
    )
    assert len(vec_out["peaks_xyz"]) > 10

    # Reference per-bank Detector calculation
    ki_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    G = hkls.astype(float) @ RUB.T
    qn = np.linalg.norm(G, axis=1)
    sth = -(G @ ki_vec) / np.maximum(qn, 1e-12)
    lam_pred = 2.0 * sth / np.maximum(qn, 1e-12)
    in_wl = (qn > 1e-8) & (sth > 1e-5) & (lam_pred >= 2.0) & (lam_pred <= 4.0)
    cand_idx = np.flatnonzero(in_wl)
    kf_dir = ki_vec[None, :] + lam_pred[cand_idx, None] * G[cand_idx]
    kf_hat = kf_dir / np.maximum(np.linalg.norm(kf_dir, axis=1, keepdims=True), 1e-12)

    ref_by_key = {}
    for bk, det in sorted(mandi_dets.items()):
        hit_m, rows, cols = det.reflections_mask(
            kf_hat[:, 0], kf_hat[:, 1], kf_hat[:, 2], sample_offset=sample_off
        )
        bw = int(margin * min(int(det.n), int(det.m)))
        hi_r = int(det.n) - 1 - bw if bw > 0 else int(det.n)
        hi_c = int(det.m) - 1 - bw if bw > 0 else int(det.m)
        hit_m = hit_m & (rows >= bw) & (rows < hi_r) & (cols >= bw) & (cols < hi_c)
        for loc in np.flatnonzero(hit_m):
            gi = int(cand_idx[loc])
            r_px = float(rows[loc])
            c_px = float(cols[loc])
            xyz_ref = np.asarray(det.pixel_to_lab(r_px, c_px), dtype=float).reshape(3)
            h, k, l = int(hkls[gi, 0]), int(hkls[gi, 1]), int(hkls[gi, 2])
            ref_by_key[(bk, h, k, l)] = (r_px, c_px, xyz_ref)

    vec_by_key = {
        (
            int(vec_out["bank"][i]),
            int(vec_out["hkl"][i, 0]),
            int(vec_out["hkl"][i, 1]),
            int(vec_out["hkl"][i, 2]),
        ): (
            float(vec_out["pixel_r"][i]),
            float(vec_out["pixel_c"][i]),
            np.asarray(vec_out["peaks_xyz"][i], dtype=float),
        )
        for i in range(len(vec_out["peaks_xyz"]))
    }
    assert set(vec_by_key.keys()) == set(ref_by_key.keys())
    for key, (r_v, c_v, xyz_v) in vec_by_key.items():
        r_ref, c_ref, xyz_ref = ref_by_key[key]
        assert abs(r_v - r_ref) < 1e-11
        assert abs(c_v - c_ref) < 1e-11
        assert np.max(np.abs(xyz_v - xyz_ref)) < 1e-13

    # 2. Build multi-frame MANDI HDF5 + verify build_rainbow_panel_textures_only & /api/payload?textures_only=1
    mandi_h5 = tmp_path / "mandi_tex_fast.h5"
    rng = np.random.default_rng(42)
    imgs = rng.poisson(12.0, size=(2, 256, 256)).astype(np.float32)
    imgs[0, 120:128, 130:138] += 250.0
    imgs[1, 60:68, 80:88] += 300.0
    with h5py.File(mandi_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf["images"] = imgs
        hf["bank_ids"] = np.array([20, 20], dtype=np.int32)
        hf["file_offsets"] = np.array([0, 1, 2], dtype=np.int32)
        hf["instrument/wavelength"] = np.array([2.0, 4.0], dtype=np.float32)

    full_hi = build_rainbow_zone_payload(
        replay_h5=mandi_h5,
        images_h5=mandi_h5,
        instrument="MANDI",
        render_binning=1,
        frame_index=1,
    )
    tex_only_hi = build_rainbow_panel_textures_only(
        replay_h5=mandi_h5,
        images_h5=mandi_h5,
        instrument="MANDI",
        render_binning=1,
        frame_index=1,
    )
    assert set(tex_only_hi.keys()) == {
        "render_binning",
        "frame_index",
        "intensity_source",
        "panels",
    }
    assert tex_only_hi["render_binning"] == 1
    assert tex_only_hi["frame_index"] == 1
    b20_full = next(p for p in full_hi["panels"] if p["bank"] == 20)
    b20_fast = next(p for p in tex_only_hi["panels"] if p["bank"] == 20)
    assert "grid_s" not in b20_fast
    assert b20_fast["texture_b64"] == b20_full["texture_b64"]

    # 3. Verify MainApp lock-free /api/payload?textures_only=1 and /api/pipeline/status?poll=1&overlay_seq=...
    app_inst = object.__new__(MainApp)
    app_inst.pipeline_runner = MagicMock()
    app_inst.pipeline_runner.is_connected = False
    app_inst.pipeline_runner.galaxy_url = ""
    app_inst.pipeline_runner.history_id = None
    app_inst.pipeline_runner.list_history_datasets.return_value = []
    app_inst.pipeline_runner.list_step_artifacts.return_value = []
    app_inst.pipeline_runner.get_step_timings_snapshot.return_value = {}
    app_inst.pipeline_runner.active_job = {
        "step": "solve",
        "state": "running",
        "partial_overlay": {
            "seq": 7,
            "frame_index": 1,
            "predicted_peaks": [{"hkl": [1, 2, 3]}],
        },
    }
    sub_m = SubhklGalaxyModel()
    sub_m.data.instrument = "MANDI"
    sub_m.data.images_file = str(mandi_h5)
    sub_m.data.peaks_file = str(mandi_h5)
    sub_m.frame_index = 1
    sub_m.render_binning = 4
    sub_m.payload = build_rainbow_zone_payload(
        replay_h5=mandi_h5,
        images_h5=mandi_h5,
        instrument="MANDI",
        render_binning=4,
        frame_index=1,
    )
    app_inst.view_model = SimpleNamespace(model=SimpleNamespace(subhkl=sub_m))
    aio_app = web.Application()
    app_inst._bind_routes(SimpleNamespace(app=aio_app))

    req_tex = make_mocked_request(
        "GET",
        "/api/payload?frame=1&render_binning=1&textures_only=1",
        app=aio_app,
    )
    resp_tex = asyncio.run(aio_app._handle(req_tex))
    assert resp_tex.status == 200
    doc_tex = json.loads(resp_tex.body.decode("utf-8"))
    assert "predicted_peaks" not in doc_tex
    assert doc_tex["render_binning"] == 1
    assert sub_m.render_binning == 1
    # Cached model payload for frame 1 should also be upgraded in-place to Hi-Res texture
    b20_cached = next(p for p in sub_m.payload["panels"] if p["bank"] == 20)
    assert b20_cached["texture_b64"] == b20_full["texture_b64"]

    # Poll status with matching overlay_seq=7 -> lightweight unchanged stub
    req_st_same = make_mocked_request(
        "GET",
        "/api/pipeline/status?frame=1&poll=1&overlay_seq=7",
        app=aio_app,
    )
    resp_st_same = asyncio.run(aio_app._handle(req_st_same))
    doc_st_same = json.loads(resp_st_same.body.decode("utf-8"))
    assert doc_st_same["active_job"]["partial_overlay"] == {
        "seq": 7,
        "frame_index": 1,
        "unchanged": True,
    }

    # Poll status with older overlay_seq=6 -> full partial_overlay delivered
    req_st_old = make_mocked_request(
        "GET",
        "/api/pipeline/status?frame=1&poll=1&overlay_seq=6",
        app=aio_app,
    )
    resp_st_old = asyncio.run(aio_app._handle(req_st_old))
    doc_st_old = json.loads(resp_st_old.body.decode("utf-8"))
    assert "unchanged" not in doc_st_old["active_job"]["partial_overlay"]
    assert len(doc_st_old["active_job"]["partial_overlay"]["predicted_peaks"]) == 1


def test_sparkline_resolution_bin_hover_reveals_edges_and_statistic(tmp_path):
    """Hovering resolution bins on #canvas-hist highlights the bin and reveals d-edges and active statistic."""
    import json
    import re
    import shutil
    import subprocess

    from subhkl.viz.rainbow_zones import (
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    payload = build_rainbow_zone_payload(instrument="MANDI", include_panel_images=False)
    bins = [
        {
            "bin": 0,
            "d_max": 11.99,
            "d_min": 4.90,
            "n_refl": 180,
            "n_theo": 195,
            "completeness_pct": 92.3,
            "mean_isigi": 14.25,
            "cc_half": 0.9565,
        },
        {
            "bin": 1,
            "d_max": 4.90,
            "d_min": 4.01,
            "n_refl": 172,
            "n_theo": 190,
            "completeness_pct": 90.5,
            "mean_isigi": 11.40,
            "cc_half": 0.9411,
        },
        {
            "bin": 2,
            "d_max": 4.01,
            "d_min": 3.52,
            "n_refl": 165,
            "n_theo": 192,
            "completeness_pct": 85.9,
            "mean_isigi": 8.10,
            "cc_half": 0.8677,
        },
        {
            "bin": 3,
            "d_max": 3.52,
            "d_min": 3.18,
            "n_refl": 150,
            "n_theo": 190,
            "completeness_pct": 78.9,
            "mean_isigi": 5.65,
            "cc_half": 0.7030,
        },
        {
            "bin": 4,
            "d_max": 3.18,
            "d_min": 2.94,
            "n_refl": 141,
            "n_theo": 191,
            "completeness_pct": 73.8,
            "mean_isigi": 4.12,
            "cc_half": 0.5163,
        },
        {
            "bin": 5,
            "d_max": 2.94,
            "d_min": 2.75,
            "n_refl": 128,
            "n_theo": 188,
            "completeness_pct": 68.1,
            "mean_isigi": 2.84,
            "cc_half": 0.3023,
        },
        {
            "bin": 6,
            "d_max": 2.75,
            "d_min": 2.61,
            "n_refl": 112,
            "n_theo": 186,
            "completeness_pct": 60.2,
            "mean_isigi": 1.95,
            "cc_half": 0.1831,
        },
        {
            "bin": 7,
            "d_max": 2.61,
            "d_min": 2.49,
            "n_refl": 98,
            "n_theo": 185,
            "completeness_pct": 53.0,
            "mean_isigi": 1.52,
            "cc_half": 0.1721,
        },
        {
            "bin": 8,
            "d_max": 2.49,
            "d_min": 2.39,
            "n_refl": 84,
            "n_theo": 184,
            "completeness_pct": 45.7,
            "mean_isigi": 1.18,
            "cc_half": 0.1256,
        },
        {
            "bin": 9,
            "d_max": 2.39,
            "d_min": 2.30,
            "n_refl": 62,
            "n_theo": 182,
            "completeness_pct": 34.1,
            "mean_isigi": 0.85,
            "cc_half": -0.0421,
        },
    ]
    payload["artifact_kind"] = "merged_careless"
    payload["merge_metrics"] = {
        "cc_half_overall": 0.812,
        "completeness_pct": 68.3,
        "mean_i_over_sigi": 5.19,
        "n_merged_asu": 1292,
        "cc_half_bins": bins,
        "resolution_bins": bins,
    }
    html = render_rainbow_html(payload)
    assert 'id="hist-bin-tooltip"' in html
    assert 'id="canvas-hist"' in html

    if not shutil.which("node"):
        return

    script_blocks = re.findall(r"<script>([\s\S]*?)</script>", html)
    inline_js = script_blocks[-1]
    js_check = tmp_path / "sparkline_inline.js"
    js_check.write_text(inline_js, encoding="utf-8")
    subprocess.run(["node", "--check", str(js_check)], check=True)

    fn_names = [
        "getMergeResolutionBins",
        "formatHistBinSummary",
        "hideHistBinTooltip",
        "updateHistBinTooltip",
        "resolveHistBinAtClientX",
        "drawHistogram",
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

    # Extract histCanvas listener block as well
    m_blk = re.search(r"if\s*\(\s*histCanvas\s*\)\s*\{", inline_js)
    assert m_blk is not None
    b_start = m_blk.start()
    depth = 0
    in_body = False
    b_end = b_start
    for idx in range(b_start, len(inline_js)):
        ch = inline_js[idx]
        if ch == "{":
            depth += 1
            in_body = True
        elif ch == "}":
            depth -= 1
            if in_body and depth == 0:
                b_end = idx + 1
                break
    listener_block = inline_js[b_start:b_end]

    harness = f"""
const DATA = {{ merge_metrics: {json.dumps(payload["merge_metrics"])} }};
const PEAKS = [];
const isPeakVisible = () => true;
const state = {{ histMetric: "cc_half", hoveredHistBin: null }};
globalThis.window = {{ devicePixelRatio: 1, innerWidth: 1280 }};

const drawnTexts = [];
const fillRects = [];
const strokeRects = [];
const listeners = {{}};
const histCanvas = {{
  width: 0,
  height: 0,
  style: {{}},
  parentElement: {{ clientWidth: 256 }},
  getBoundingClientRect: () => ({{ left: 960, top: 620, width: 256, height: 50 }}),
  addEventListener(type, fn) {{ listeners[type] = fn; }},
  getContext: () => {{
    const ctx = {{
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 1,
      font: "",
      globalAlpha: 1,
      scale() {{}},
      clearRect() {{ drawnTexts.length = 0; fillRects.length = 0; strokeRects.length = 0; }},
      fillRect(...a) {{ fillRects.push({{ fillStyle: ctx.fillStyle, args: a }}); }},
      strokeRect(...a) {{ strokeRects.push(a); }},
      fillText(txt) {{ drawnTexts.push(txt); }},
    }};
    return ctx;
  }},
}};
const histBinTooltip = {{
  innerHTML: "",
  offsetWidth: 210,
  offsetHeight: 34,
  style: {{ display: "none", left: "", top: "", borderColor: "" }},
}};

{chr(10).join(extracted)}
{listener_block}

drawHistogram([]);
const idleTexts = [...drawnTexts];
const idleBarColors = fillRects.map(r => r.fillStyle);
const idleDisplay = histBinTooltip.style.display;
const boundaryColors = {{
  c080: formatHistBinSummary({{ cc_half: 0.80 }}, 0, 10, "cc_half").statColor,
  c050: formatHistBinSummary({{ cc_half: 0.50 }}, 0, 10, "cc_half").statColor,
  c030: formatHistBinSummary({{ cc_half: 0.30 }}, 0, 10, "cc_half").statColor,
  c0299: formatHistBinSummary({{ cc_half: 0.299 }}, 0, 10, "cc_half").statColor,
  cNeg: formatHistBinSummary({{ cc_half: -0.05 }}, 0, 10, "cc_half").statColor,
}};

// Hover over Bin 6/10 (index 5: 2.94 - 2.75 A, CC1/2 = 0.302)
// barW = (256 - 16) / 10 = 24px -> bin 5 center at 960 + 8 + 5.5 * 24 = 1100
listeners.mousemove({{ clientX: 1100 }});
const bin5CcHtml = histBinTooltip.innerHTML;
const bin5CcBorder = histBinTooltip.style.borderColor;
const bin5CcDisplay = histBinTooltip.style.display;
const bin5CcCanvasTexts = [...drawnTexts];
const bin5Strokes = strokeRects.length;

// Click while hovering Bin 6/10 -> cycles to Completeness, then <I/sigma>
listeners.click();
const bin5CompHtml = histBinTooltip.innerHTML;
const bin5CompCanvasTexts = [...drawnTexts];

listeners.click();
const bin5IsigHtml = histBinTooltip.innerHTML;

// Click once more -> back to CC1/2, then hover Bin 10/10 (index 9: negative CC1/2 = -0.042)
listeners.click();
listeners.mousemove({{ clientX: 960 + 8 + 9.5 * 24 }});
const bin9CcHtml = histBinTooltip.innerHTML;
const bin9CcBorder = histBinTooltip.style.borderColor;

// Mouseleave restores summary and hides tooltip
listeners.mouseleave();
const afterLeaveDisplay = histBinTooltip.style.display;
const afterLeaveTexts = [...drawnTexts];

console.log(JSON.stringify({{
  idleTexts,
  idleBarColors,
  boundaryColors,
  idleDisplay,
  bin5CcHtml,
  bin5CcBorder,
  bin5CcDisplay,
  bin5CcCanvasTexts,
  bin5Strokes,
  bin5CompHtml,
  bin5CompCanvasTexts,
  bin5IsigHtml,
  bin9CcHtml,
  bin9CcBorder,
  afterLeaveDisplay,
  afterLeaveTexts,
}}));
"""
    hf_js = tmp_path / "sparkline_hover_harness.js"
    hf_js.write_text(harness, encoding="utf-8")
    res = subprocess.run(
        ["node", str(hf_js)], check=True, capture_output=True, text=True
    )
    out = json.loads(res.stdout.strip())

    assert out["idleDisplay"] == "none"
    assert any("CC½ vs d (0.812)" in t for t in out["idleTexts"])
    # Yellow range extends down to 0.30; everything below 0.30 is red (#f43f5e)
    assert out["idleBarColors"] == [
        "#34d399",  # 0.9565 >= 0.80
        "#34d399",  # 0.9411 >= 0.80
        "#34d399",  # 0.8677 >= 0.80
        "#fbbf24",  # 0.7030 >= 0.30
        "#fbbf24",  # 0.5163 >= 0.30
        "#fbbf24",  # 0.3023 >= 0.30
        "#f43f5e",  # 0.1831 < 0.30
        "#f43f5e",  # 0.1721 < 0.30
        "#f43f5e",  # 0.1256 < 0.30
        "#f43f5e",  # -0.0421 < 0.30
    ]
    assert out["boundaryColors"] == {
        "c080": "#34d399",
        "c050": "#fbbf24",
        "c030": "#fbbf24",
        "c0299": "#f43f5e",
        "cNeg": "#f43f5e",
    }

    assert out["bin5CcDisplay"] == "block"
    assert out["bin5CcBorder"] == "#fbbf24"
    assert out["bin5Strokes"] == 1
    assert "Bin 6/10:" in out["bin5CcHtml"]
    assert "2.94–2.75 Å" in out["bin5CcHtml"]
    assert "CC½ = 0.302" in out["bin5CcHtml"]
    assert "128/188 refl" in out["bin5CcHtml"]
    assert any("2.94–2.75 Å • CC½=0.302" in t for t in out["bin5CcCanvasTexts"])

    assert "Compl. = 68.1%" in out["bin5CompHtml"]
    assert any("2.94–2.75 Å • Compl.=68.1%" in t for t in out["bin5CompCanvasTexts"])
    assert "⟨I/σ⟩ = 2.84" in out["bin5IsigHtml"]

    assert "Bin 10/10:" in out["bin9CcHtml"]
    assert "2.39–2.30 Å" in out["bin9CcHtml"]
    assert "CC½ = -0.042" in out["bin9CcHtml"]
    assert out["bin9CcBorder"] == "#f43f5e"

    assert out["afterLeaveDisplay"] == "none"
    assert any("CC½ vs d (0.812)" in t for t in out["afterLeaveTexts"])


def test_merge_resolution_table_csv_html_popup_and_lazy_download(tmp_path):
    """Publication-ready CC1/2 / Completeness / <I/sigma> resolution table CSV + HTML artifacts, popup, and lazy downloads."""
    import csv
    import io
    import json
    import math
    import re
    import shutil
    import subprocess
    from pathlib import Path

    import gemmi
    import h5py
    import numpy as np

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import (
        build_merge_resolution_table_data,
        build_rainbow_zone_payload,
        render_merge_table_csv,
        render_merge_table_html,
        render_rainbow_html,
    )

    metrics = {
        "space_group": "I 1 2 1",
        "cell": [55.16, 81.19, 88.70, 90.0, 96.3, 90.0],
        "wavelength_range": [2.0, 4.0],
        "d_max": 11.99,
        "d_min": 2.30,
        "n_observations": 3840,
        "n_merged_asu": 1280,
        "n_theoretical_asu": 1873,
        "mean_multiplicity": 3.0,
        "completeness_pct": 68.34,
        "mean_isigi": 5.19,
        "cc_half": 0.8124,
        "resolution_bins": [
            {
                "bin": 0,
                "d_max": 11.99,
                "d_min": 4.90,
                "n_obs": 720,
                "n_refl": 180,
                "n_theo": 195,
                "completeness_pct": 92.3,
                "mean_isigi": 14.25,
                "cc_half": 0.9565,
            },
            {
                "bin": 1,
                "d_max": 4.90,
                "d_min": 2.75,
                "n_obs": 384,
                "n_refl": 128,
                "n_theo": 188,
                "completeness_pct": 68.1,
                "mean_isigi": 2.84,
                "cc_half": 0.3023,
            },
            {
                "bin": 2,
                "d_max": 2.75,
                "d_min": 2.30,
                "n_obs": 124,
                "n_refl": 62,
                "n_theo": 182,
                "completeness_pct": 34.1,
                "mean_isigi": 0.85,
                "cc_half": 0.1003,
            },
        ],
    }

    tdata = build_merge_resolution_table_data(metrics)
    assert tdata["d_cc30"] == 2.75
    assert [r["cc_tier"] for r in tdata["rows"]] == ["green", "yellow", "red"]
    assert [r["rank_badge"] for r in tdata["rows"]] == ["1ST", "2ND", "3RD"]
    assert abs(tdata["rows"][0]["multiplicity"] - 4.0) < 1e-6
    expected_cc_star_0 = math.sqrt(2.0 * 0.9565 / (1.0 + 0.9565))
    assert abs(tdata["rows"][0]["cc_star"] - expected_cc_star_0) < 1e-6

    csv_text = render_merge_table_csv(metrics)
    parsed_csv = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(parsed_csv) == 4  # 3 shells + Overall
    assert parsed_csv[0]["shell"] == "1"
    assert parsed_csv[0]["resolution_range_A"] == "11.99 - 4.90"
    assert parsed_csv[0]["n_obs"] == "720"
    assert parsed_csv[0]["multiplicity"] == "4.00"
    assert parsed_csv[1]["cc_half"] == "0.3023"
    assert parsed_csv[-1]["shell"] == "Overall"
    assert parsed_csv[-1]["n_obs"] == "3840"
    assert parsed_csv[-1]["cc_half"] == "0.8124"

    html_art = render_merge_table_html(metrics)
    assert "BUBBLE" not in html_art.upper()
    assert "1UP" not in html_art.upper()
    assert "Word / Excel" not in html_art
    assert ">Copy</button>" in html_art
    assert "Download .csv" not in html_art
    assert "Georgia" in html_art
    assert "#0a1020" in html_art and "#bae6fd" in html_art
    assert "I 1 2 1" in html_art
    assert "0.302" in html_art
    assert "background:#ffffff;color:#000000" in html_art

    # Verify Rainbow Zones main UI embeds subtle table pictogram button, monochrome blue-on-dark modal, Copy button, and no duplicate download buttons
    payload = build_rainbow_zone_payload(instrument="MANDI", include_panel_images=False)
    payload["artifact_kind"] = "merged_careless"
    payload["space_group"] = "I 1 2 1"
    payload["cell"] = [55.16, 81.19, 88.70, 90.0, 96.3, 90.0]
    payload["wavelength_range"] = [2.0, 4.0]
    payload["merge_metrics"] = metrics
    ui_html = render_rainbow_html(payload)
    assert 'id="modal-merge-table"' in ui_html
    assert 'id="btn-hist-open-table"' in ui_html
    assert 'aria-label="Merging statistics table"' in ui_html
    assert '<path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/>' in ui_html
    assert "☆ Table" not in ui_html
    assert "🏆" not in ui_html
    assert "📋" not in ui_html
    assert "BUBBLE" not in ui_html.upper()
    assert "Word / Excel" not in ui_html
    assert 'id="btn-hist-dl-csv"' not in ui_html
    assert 'id="btn-hist-dl-html"' not in ui_html
    assert 'id="btn-merge-tbl-csv"' not in ui_html
    assert 'id="btn-merge-tbl-html"' not in ui_html
    assert 'id="btn-merge-tbl-style"' not in ui_html
    assert 'id="btn-merge-tbl-copy"' in ui_html
    assert ">Copy</button>" in ui_html

    if shutil.which("node"):
        script_blocks = re.findall(r"<script>([\s\S]*?)</script>", ui_html)
        inline_js = script_blocks[-1]
        fn_names = [
            "getMergeResolutionBins",
            "escapeHtmlText",
            "ordinalShellRank",
            "computeCcStarClient",
            "ccHalfTierClient",
            "buildClientMergeTableData",
            "buildMergeTableClipboardPayloads",
            "renderMergeTableModalContent",
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

        modal_harness = f"""
const DATA = {json.dumps({"space_group": payload["space_group"], "cell": payload["cell"], "wavelength_range": payload["wavelength_range"], "merge_metrics": metrics})};
const els = {{}};
["merge-tbl-card", "merge-tbl-modal-title", "merge-tbl-meta", "merge-tbl-body-wrap"].forEach(id => {{
  els[id] = {{ style: {{}}, innerHTML: "", textContent: "" }};
}});
globalThis.document = {{ getElementById: (id) => els[id] || null }};

{chr(10).join(extracted)}

const clientData = buildClientMergeTableData();
const clip = buildMergeTableClipboardPayloads(clientData);
renderMergeTableModalContent();
const modalHtml = els["merge-tbl-body-wrap"].innerHTML;
const modalBg = els["merge-tbl-card"].style.background;
const modalFg = els["merge-tbl-card"].style.color;

console.log(JSON.stringify({{
  dCc30: clientData.dCc30,
  tiers: clientData.rows.map(r => r.ccTier),
  cleanHtml: clip.cleanHtml,
  tsv: clip.tsv,
  csv: clip.csv,
  modalHtml,
  modalBg,
  modalFg,
}}));
"""
        mh_path = tmp_path / "merge_modal_harness.js"
        mh_path.write_text(modal_harness, encoding="utf-8")
        js_res = subprocess.run(
            ["node", str(mh_path)], check=True, capture_output=True, text=True
        )
        jout = json.loads(js_res.stdout.strip())
        assert jout["dCc30"] == 2.75
        assert jout["tiers"] == ["green", "yellow", "red"]
        assert "background:#ffffff;color:#000000" in jout["cleanHtml"]
        assert "11.99 – 4.90" in jout["cleanHtml"]
        assert "Overall\t11.99 – 2.30\t3840\t1280" in jout["tsv"]
        assert jout["csv"] == csv_text
        assert jout["modalBg"] == "#0a1020"
        assert jout["modalFg"] == "#bae6fd"
        assert "Georgia" in jout["modalHtml"]
        assert "1ST" not in jout["modalHtml"]
        assert "#fbbf24" not in jout["modalHtml"] and "#34d399" not in jout["modalHtml"]

    # Verify _stamp_careless_merge_into_h5 vectorized n_obs/multiplicity + lazy resolve_artifact_for_download
    work_dir = tmp_path / "merge_work"
    work_dir.mkdir()
    uc = gemmi.UnitCell(55.16, 81.19, 88.70, 90.0, 96.3, 90.0)
    sg = gemmi.find_spacegroup_by_name("I 1 2 1")

    merged_mtz = gemmi.Mtz(with_base=True)
    merged_mtz.spacegroup = sg
    merged_mtz.set_cell_for_all(uc)
    merged_mtz.add_dataset("merged")
    merged_mtz.add_column("I", "J")
    merged_mtz.add_column("SigI", "Q")
    asu_hkls = [
        (2, 0, 0),
        (0, 2, 0),
        (2, 2, 0),
        (4, 2, 2),
        (6, 4, 2),
        (8, 6, 4),
        (10, 8, 6),
        (12, 10, 8),
        (14, 12, 10),
        (16, 14, 12),
        (18, 14, 12),
        (20, 14, 12),
    ]
    merged_rows = [
        [h, k, l, 200.0 - 10.0 * i, 12.0] for i, (h, k, l) in enumerate(asu_hkls)
    ]
    merged_mtz.set_data(np.asarray(merged_rows, dtype=np.float32))
    merged_mtz.write_to_file(str(work_dir / "merged.mtz"))

    # Repeat each reflection 3 times in predictions.mtz -> multiplicity == 3.0 in every populated shell
    pred_mtz = gemmi.Mtz(with_base=True)
    pred_mtz.spacegroup = sg
    pred_mtz.set_cell_for_all(uc)
    pred_mtz.add_dataset("pred")
    pred_mtz.add_column("I", "J")
    pred_mtz.add_column("SigI", "Q")
    pred_mtz.add_column("pred", "J")
    pred_mtz.add_column("scale", "R")
    pred_rows = [
        [h, k, l, 180.0, 12.0, 175.0, 1.0]
        for (h, k, l) in asu_hkls
        for _rep in range(3)
    ]
    pred_mtz.set_data(np.asarray(pred_rows, dtype=np.float32))
    pred_mtz.write_to_file(str(work_dir / "predictions.mtz"))

    xval_mtz = gemmi.Mtz(with_base=True)
    xval_mtz.spacegroup = sg
    xval_mtz.set_cell_for_all(uc)
    xval_mtz.add_dataset("xval")
    xval_mtz.add_column("half", "I")
    xval_mtz.add_column("I", "J")
    xval_rows = []
    for i, (h, k, l) in enumerate(asu_hkls):
        xval_rows.append([h, k, l, 0, 200.0 - 10.0 * i])
        xval_rows.append([h, k, l, 1, 198.0 - 9.8 * i])
    xval_mtz.set_data(np.asarray(xval_rows, dtype=np.float32))
    xval_mtz.write_to_file(str(work_dir / "xval.mtz"))

    integ_h5 = work_dir / "integrated.h5"
    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        pk = hf.create_group("peaks")
        pk.create_dataset("h", data=np.array([2, 0], dtype=np.int32))
        pk.create_dataset("k", data=np.array([0, 2], dtype=np.int32))
        pk.create_dataset("l", data=np.array([0, 0], dtype=np.int32))
        pk.create_dataset("intensity", data=np.array([200.0, 190.0], dtype=float))
        pk.create_dataset("sigma", data=np.array([12.0, 12.0], dtype=float))

    runner = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
    runner._local_work_dir = str(work_dir)
    runner._stamp_careless_merge_into_h5(str(integ_h5), str(work_dir), {})

    assert (work_dir / "merge_table.csv").is_file()
    assert (work_dir / "merge_table.html").is_file()
    stamped_metrics = json.loads(
        (work_dir / "careless_metrics.json").read_text(encoding="utf-8")
    )
    populated_bins = [b for b in stamped_metrics["resolution_bins"] if b["n_refl"] > 0]
    assert len(populated_bins) >= 3
    for b in populated_bins:
        assert b["n_obs"] == 3 * b["n_refl"]
        assert abs(b["multiplicity"] - 3.0) < 1e-6

    # Delete merge_table.csv and merge_table.html to simulate a pre-existing workspace,
    # then verify Files (list_step_artifacts) exposes ONLY merge_table.csv (not duplicate merge_table.html)
    (work_dir / "merge_table.csv").unlink()
    (work_dir / "merge_table.html").unlink()
    runner._local_artifacts = {
        "careless_metrics.json": str(work_dir / "careless_metrics.json"),
        "integrated.h5": str(integ_h5),
    }
    arts = runner.list_step_artifacts(current_h5=str(integ_h5))
    merge_art_by_name = {a["name"]: a for a in arts["merge"]}
    assert "merge_table.csv" in merge_art_by_name
    assert "merge_table.html" not in merge_art_by_name
    assert merge_art_by_name["merge_table.csv"]["stage_id"] == "merge"

    # Unlink once more and test direct on-demand resolve_artifact_for_download
    (work_dir / "merge_table.csv").unlink()
    (work_dir / "merge_table.html").unlink()
    runner._local_artifacts.pop("merge_table.csv", None)
    runner._local_artifacts.pop("merge_table.html", None)
    dl_csv = runner.resolve_artifact_for_download(
        name="merge_table.csv", current_h5=str(integ_h5)
    )
    dl_html = runner.resolve_artifact_for_download(
        name="merge_table.html", current_h5=str(integ_h5)
    )
    assert dl_csv is not None and dl_csv[1] == "merge_table.csv"
    assert dl_html is not None and dl_html[1] == "merge_table.html"
    assert "resolution_range_A" in Path(dl_csv[0]).read_text(encoding="utf-8")
    assert ">Copy</button>" in Path(dl_html[0]).read_text(encoding="utf-8")


def test_solve_binning_param_drawer_and_cli_forwarding(tmp_path):
    """Params drawer exposes Solve binning (auto, 16, 8, 4) and forwards --binning only when explicitly chosen."""
    import h5py
    import numpy as np

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import (
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    payload = build_rainbow_zone_payload()
    html_default = render_rainbow_html(payload)
    assert (
        '<select id="inp-binning"><option value="" selected>auto</option>'
        '<option value="16">16</option><option value="8">8</option>'
        '<option value="4">4</option><option value="2">2</option></select>'
    ) in html_default

    payload_b16 = dict(payload, solve_binning=16)
    html_b16 = render_rainbow_html(payload_b16)
    assert '<option value="16" selected>16</option>' in html_b16

    merged_h5 = tmp_path / "merged.h5"
    with h5py.File(merged_h5, "w") as hf:
        hf.create_dataset("images/ims", data=np.zeros((1, 256, 256), dtype=np.float32))

    runner = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
    runner._local_work_dir = str(tmp_path)

    base_params = {
        "instrument": "MANDI",
        "space_group": "P 6_1 22",
        "a": 79.91,
        "b": 79.91,
        "c": 238.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
        "wavelength_min": 2.0,
        "wavelength_max": 4.0,
        "d_min": 1.45,
        "images_file": str(merged_h5),
    }

    rec_auto = {"cancel_requested": True}
    runner._submit_local_subhkl_stage(
        "solve", dict(base_params, binning=None), existing_job_record=rec_auto
    )
    assert "--binning" not in rec_auto["command_line"]
    assert rec_auto["solve_params"]["binning"] is None

    for b_val in (16, 8, 4):
        rec_b = {"cancel_requested": True}
        runner._submit_local_subhkl_stage(
            "solve", dict(base_params, binning=b_val), existing_job_record=rec_b
        )
        assert f"--binning {b_val}" in rec_b["command_line"]
        assert rec_b["solve_params"]["binning"] == b_val


def test_careless_mono_mode_auto_selected_for_tof_without_ui_selector(
    tmp_path, monkeypatch
) -> None:
    """Verify UI has no mono/poly selector and automatically selects careless mono (without --wavelength-key, keeping WAVEL in metadata) whenever TOF data is available."""
    import json
    import re
    import subprocess
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import gemmi
    import h5py
    import numpy as np

    from subhkl.viz.nova_app.models.subhkl_model import SubhklGalaxyModel
    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import (
        _h5_has_tof_data,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    def _write_dummy_mtz(mtz_path: Path) -> None:
        mtz = gemmi.Mtz(with_base=True)
        mtz.spacegroup = gemmi.find_spacegroup_by_name("I a -3 d")
        mtz.set_cell_for_all(gemmi.UnitCell(11.93, 11.93, 11.93, 90.0, 90.0, 90.0))
        mtz.add_dataset("garnet")
        mtz.add_column("M/ISYM", "Y")
        mtz.add_column("BATCH", "B")
        mtz.add_column("I", "J")
        mtz.add_column("SIGI", "Q")
        mtz.add_column("WAVEL", "R")
        mtz.add_column("THETA", "R")
        mtz.add_column("PHI", "R")
        mtz.set_data(
            np.array(
                [
                    [4.0, 0.0, 0.0, 1.0, 1001.0, 150.0, 8.0, 1.49, 0.42, 1.10],
                    [8.0, 0.0, 0.0, 1.0, 1001.0, 90.0, 6.0, 2.98, 0.42, 1.10],
                ],
                dtype=np.float32,
            )
        )
        mtz.write_to_file(str(mtz_path))

    # 1. Non-TOF workdir (e.g. CG4D image-only scan) -> has_tof_data is False -> careless poly + --wavelength-key=WAVEL
    poly_dir = tmp_path / "work_poly"
    poly_dir.mkdir()
    poly_merged = poly_dir / "merged.h5"
    with h5py.File(poly_merged, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf.create_dataset("images", data=np.zeros((1, 32, 32), dtype=np.float32))
        hf.create_dataset("bank_ids", data=np.array([20], dtype=np.int32))
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([2.8, 4.5], dtype=np.float64)
        )
    poly_integ = poly_dir / "integrated.h5"
    with h5py.File(poly_integ, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf.attrs["source"] = str(poly_merged)
        smp = hf.create_group("sample")
        smp.create_dataset("U", data=np.eye(3))
        smp.create_dataset("B", data=np.diag([1.0 / 54.0] * 3))
        pk = hf.create_group("peaks")
        pk.create_dataset("h", data=np.array([1], dtype=np.int32))
        pk.create_dataset("k", data=np.array([0], dtype=np.int32))
        pk.create_dataset("l", data=np.array([0], dtype=np.int32))
        pk.create_dataset("lambda", data=np.array([3.2], dtype=np.float64))
        pk.create_dataset("intensity", data=np.array([100.0], dtype=np.float64))
        pk.create_dataset("sigma", data=np.array([5.0], dtype=np.float64))
        pk.create_dataset("bank", data=np.array([20], dtype=np.int32))
        pk.create_dataset("pixel_r", data=np.array([16.0], dtype=np.float64))
        pk.create_dataset("pixel_c", data=np.array([16.0], dtype=np.float64))
        pk.create_dataset("xyz", data=np.array([[0.1, 0.0, 0.9]], dtype=np.float64))
    _write_dummy_mtz(poly_dir / "reflections.mtz")

    assert _h5_has_tof_data(poly_merged, poly_integ) is False
    payload_poly = build_rainbow_zone_payload(
        replay_h5=poly_integ, images_h5=poly_merged, include_panel_images=False
    )
    assert payload_poly["has_tof_data"] is False

    # 2. TOF workdir (MANDI Ia-3d garnet 1-3 A with pixel-sorted events/tof + tof_gate) -> has_tof_data is True -> careless mono
    mono_dir = tmp_path / "work_mono"
    mono_dir.mkdir()
    mono_merged = mono_dir / "merged.h5"
    with h5py.File(mono_merged, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.create_dataset("images", data=np.zeros((1, 32, 32), dtype=np.float32))
        hf.create_dataset("bank_ids", data=np.array([19], dtype=np.int32))
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([1.0, 3.0], dtype=np.float64)
        )
        ev = hf.create_group("events")
        ev.create_dataset("tof", data=np.array([11500.0, 23000.0], dtype=np.float32))
        ev.create_dataset("image_offsets", data=np.array([0, 2], dtype=np.int64))
    mono_solve = mono_dir / "solve.h5"
    with h5py.File(mono_solve, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.attrs["source"] = str(mono_merged)
        smp = hf.create_group("sample")
        smp.create_dataset("U", data=np.eye(3))
        smp.create_dataset("B", data=np.diag([1.0 / 11.93] * 3))
    mono_integ = mono_dir / "integrated.h5"
    with h5py.File(mono_integ, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.attrs["source"] = str(mono_merged)
        smp = hf.create_group("sample")
        smp.create_dataset("U", data=np.eye(3))
        smp.create_dataset("B", data=np.diag([1.0 / 11.93] * 3))
        tg = hf.create_group("tof_gate")
        tg.attrs["mode"] = "tof_gated_3d"
        pk = hf.create_group("peaks")
        pk.create_dataset("h", data=np.array([4, 8], dtype=np.int32))
        pk.create_dataset("k", data=np.array([0, 0], dtype=np.int32))
        pk.create_dataset("l", data=np.array([0, 0], dtype=np.int32))
        pk.create_dataset("lambda", data=np.array([2.98, 1.49], dtype=np.float64))
        pk.create_dataset("tof", data=np.array([23000.0, 11500.0], dtype=np.float64))
        pk.create_dataset("intensity", data=np.array([150.0, 90.0], dtype=np.float64))
        pk.create_dataset("sigma", data=np.array([8.0, 6.0], dtype=np.float64))
        pk.create_dataset("bank", data=np.array([19, 19], dtype=np.int32))
        pk.create_dataset("pixel_r", data=np.array([16.0, 16.0], dtype=np.float64))
        pk.create_dataset("pixel_c", data=np.array([16.0, 16.0], dtype=np.float64))
        pk.create_dataset(
            "xyz",
            data=np.array([[0.2, 0.0, 0.4], [0.2, 0.0, 0.4]], dtype=np.float64),
        )
    _write_dummy_mtz(mono_dir / "reflections.mtz")

    assert _h5_has_tof_data(mono_merged) is True
    assert _h5_has_tof_data(mono_solve) is True  # resolves via attrs['source']
    assert _h5_has_tof_data(mono_integ) is True
    payload_mono = build_rainbow_zone_payload(
        replay_h5=mono_integ, images_h5=mono_merged, include_panel_images=False
    )
    assert payload_mono["has_tof_data"] is True

    # 3. Verify UI has NO mono/poly selector, and collectWorkflowParams() auto-selects mono iff DATA.has_tof_data
    html_mono = render_rainbow_html(payload_mono)
    assert "inp-careless-mode" not in html_mono
    assert "inp-merge-mode" not in html_mono
    assert 'id="inp-mode"' not in html_mono
    m_collect = re.search(
        r"function collectWorkflowParams\(\)\s*\{[\s\S]*?\n  \}", html_mono
    )
    assert m_collect is not None
    js_eval = f"""
const state = {{ dMin: 1.2, lamMin: 1.0, lamMax: 3.0 }};
const inpDminEl = null;
const slDmin = null;
function getDefaultWavelengthForInstrument() {{ return [2.0, 4.0]; }}
function syncDminControls() {{}}
function defaultRelBorderForInstrument() {{ return 0.0; }}
globalThis.document = {{ getElementById: () => null, activeElement: null }};
let DATA = {{ has_tof_data: false, instrument: "CG4D", space_group: "P 21 21 21", cell: [54,54,54,90,90,90], d_min: 1.4, wavelength_range: [2.8, 4.5] }};
{m_collect.group(0)}
const pPoly = collectWorkflowParams();
DATA = {{ has_tof_data: true, instrument: "MANDI", space_group: "I a -3 d", cell: [11.93,11.93,11.93,90,90,90], d_min: 1.0, wavelength_range: [1.0, 3.0] }};
const pMono = collectWorkflowParams();
console.log(JSON.stringify({{ pPoly, pMono }}));
"""
    proc = subprocess.run(
        ["node", "-e", js_eval], check=True, capture_output=True, text=True
    )
    js_res = json.loads(proc.stdout.strip())
    assert js_res["pPoly"]["has_tof_data"] is False
    assert js_res["pPoly"]["merge_mode"] == "poly"
    assert js_res["pMono"]["has_tof_data"] is True
    assert js_res["pMono"]["merge_mode"] == "mono"

    # 4. Verify SubhklGalaxyModel.build_tool_parameters("merge")
    m_sub = SubhklGalaxyModel()
    m_sub.data.peaks_file = str(poly_integ)
    m_sub.data.images_file = str(poly_merged)
    assert m_sub.build_tool_parameters("merge")["mode"] == "poly"
    m_sub.data.peaks_file = str(mono_integ)
    m_sub.data.images_file = str(mono_merged)
    assert m_sub.build_tool_parameters("merge")["mode"] == "mono"

    # 5. Verify local Careless command line & prep_merge.py metadata keys
    runner_poly = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
    runner_poly._local_work_dir = str(poly_dir)
    rec_poly = {"cancel_requested": True}
    runner_poly._submit_local_subhkl_stage(
        "merge",
        {"images_file": str(poly_merged), "peaks_file": str(poly_integ)},
        existing_job_record=rec_poly,
    )
    assert "careless poly " in rec_poly["command_line"]
    assert "--wavelength-key=WAVEL" in rec_poly["command_line"]
    prep_poly_src = (poly_dir / "careless_work" / "prep_merge.py").read_text()
    assert "'WAVEL'" in prep_poly_src

    runner_mono = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
    runner_mono._local_work_dir = str(mono_dir)
    rec_mono = {"cancel_requested": True}
    runner_mono._submit_local_subhkl_stage(
        "merge",
        {
            "instrument": "MANDI",
            "space_group": "I a -3 d",
            "wavelength_min": 1.0,
            "wavelength_max": 3.0,
            "images_file": str(mono_merged),
            "peaks_file": str(mono_integ),
        },
        existing_job_record=rec_mono,
    )
    assert "careless mono " in rec_mono["command_line"]
    assert "--wavelength-key" not in rec_mono["command_line"]
    prep_mono_src = (mono_dir / "careless_work" / "prep_merge.py").read_text()
    post_mono_src = (mono_dir / "careless_work" / "post_merge.py").read_text()
    assert "'WAVEL'" in prep_mono_src  # WAVEL stays in metadata even in mono mode
    assert "'mode': 'mono'" in post_mono_src

    # 6. Verify Galaxy imaginex_subhkl_merge tool_inputs passes mode="mono" for TOF and mode="poly" for non-TOF
    monkeypatch.setenv("SUBHKL_SHARED_DIR", str(tmp_path / "shared_dir"))
    gx_runner = NovaGalaxyPipelineRunner(
        galaxy_url="http://localhost:8080", galaxy_api_key="fake-key"
    )
    assert gx_runner._resolve_careless_merge_mode({}, work_dir=str(poly_dir)) == "poly"
    assert gx_runner._resolve_careless_merge_mode({}, work_dir=str(mono_dir)) == "mono"

    captured_inputs: list[tuple[str, dict]] = []
    fallback_reasons: list[str] = []

    def _capture_run_tool(history_id, tool_id, tool_inputs):
        captured_inputs.append((tool_id, dict(tool_inputs)))
        return {"jobs": [{"id": f"gx_job_{len(captured_inputs)}"}], "outputs": []}

    fake_store = MagicMock()
    fake_store.history_id = "hist_123"
    fake_store.nova_connection.galaxy_instance.tools.run_tool.side_effect = (
        _capture_run_tool
    )

    with (
        patch.object(gx_runner, "get_datastore", return_value=fake_store),
        patch.object(gx_runner, "_has_tool_on_galaxy", return_value=True),
        patch.object(gx_runner, "_ensure_galaxy_h5_dataset", return_value="hda_mtz_1"),
        patch.object(gx_runner, "_get_tool_version", return_value="0.1.3"),
        patch.object(gx_runner, "_poll_galaxy_job", return_value=None),
        patch.object(
            gx_runner,
            "_submit_local_subhkl_stage",
            side_effect=lambda *a, **k: fallback_reasons.append(
                str(k.get("fallback_reason") or "")
            ),
        ),
    ):
        gx_runner._dispatch_nova_galaxy_thread(
            "merge",
            {"images_file": str(mono_merged), "peaks_file": str(mono_integ)},
            {"job_id": "j1", "stage_id": "merge", "state": "queued", "stdout": ""},
            None,
        )
        gx_runner._dispatch_nova_galaxy_thread(
            "merge",
            {"images_file": str(poly_merged), "peaks_file": str(poly_integ)},
            {"job_id": "j2", "stage_id": "merge", "state": "queued", "stdout": ""},
            None,
        )

    assert fallback_reasons == []
    assert len(captured_inputs) == 2
    assert captured_inputs[0][0] == "imaginex_subhkl_merge"
    assert captured_inputs[0][1]["mode"] == "mono"
    assert captured_inputs[1][0] == "imaginex_subhkl_merge"
    assert captured_inputs[1][1]["mode"] == "poly"

    # Legacy imaginex_subhkl_merge 0.1.2 without mono support must fall back rather than silently run poly on TOF data
    with (
        patch.object(gx_runner, "get_datastore", return_value=fake_store),
        patch.object(gx_runner, "_has_tool_on_galaxy", return_value=True),
        patch.object(gx_runner, "_ensure_galaxy_h5_dataset", return_value="hda_mtz_1"),
        patch.object(gx_runner, "_get_tool_version", return_value="0.1.2"),
        patch.object(gx_runner, "_poll_galaxy_job", return_value=None),
        patch.object(
            gx_runner,
            "_submit_local_subhkl_stage",
            side_effect=lambda *a, **k: fallback_reasons.append(
                str(k.get("fallback_reason") or "")
            ),
        ),
    ):
        gx_runner._dispatch_nova_galaxy_thread(
            "merge",
            {"images_file": str(mono_merged), "peaks_file": str(mono_integ)},
            {"job_id": "j3", "stage_id": "merge", "state": "queued", "stdout": ""},
            None,
        )
    assert len(fallback_reasons) == 1
    assert "lacks monochromatic ('mono') mode" in fallback_reasons[0]


def test_wilson_prior_default_mean_calibrated_residual_colorbar_and_single_table_button(
    tmp_path,
) -> None:
    """Verify wilson_prior_b defaults to None (not overriding careless), residual colorbar covers observed range centered on dataset mean, and Wulff bar omits duplicate table button."""
    import json
    import re
    import shutil
    import subprocess

    import gemmi
    import h5py
    import numpy as np

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import (
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    work_dir = tmp_path / "work_garnet"
    work_dir.mkdir()
    merged_h5 = work_dir / "merged.h5"
    with h5py.File(merged_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.create_dataset("images", data=np.zeros((1, 32, 32), dtype=np.float32))
        hf.create_dataset("bank_ids", data=np.array([19], dtype=np.int32))
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([1.0, 3.0], dtype=np.float64)
        )
        ev = hf.create_group("events")
        ev.create_dataset("tof", data=np.array([11500.0, 23000.0], dtype=np.float32))
        ev.create_dataset("image_offsets", data=np.array([0, 2], dtype=np.int64))
    integ_h5 = work_dir / "integrated.h5"
    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.attrs["source"] = str(merged_h5)
        smp = hf.create_group("sample")
        smp.create_dataset("U", data=np.eye(3))
        smp.create_dataset("B", data=np.diag([1.0 / 11.93] * 3))
        pk = hf.create_group("peaks")
        pk.create_dataset("h", data=np.array([4, 8], dtype=np.int32))
        pk.create_dataset("k", data=np.array([0, 0], dtype=np.int32))
        pk.create_dataset("l", data=np.array([0, 0], dtype=np.int32))
        pk.create_dataset("lambda", data=np.array([2.98, 1.49], dtype=np.float64))
        pk.create_dataset("intensity", data=np.array([150.0, 90.0], dtype=np.float64))
        pk.create_dataset("sigma", data=np.array([8.0, 6.0], dtype=np.float64))
        pk.create_dataset("bank", data=np.array([19, 19], dtype=np.int32))
        pk.create_dataset("pixel_r", data=np.array([16.0, 16.0], dtype=np.float64))
        pk.create_dataset("pixel_c", data=np.array([16.0, 16.0], dtype=np.float64))
        pk.create_dataset(
            "xyz",
            data=np.array([[0.2, 0.0, 0.4], [0.2, 0.0, 0.4]], dtype=np.float64),
        )
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("I a -3 d")
    mtz.set_cell_for_all(gemmi.UnitCell(11.93, 11.93, 11.93, 90.0, 90.0, 90.0))
    mtz.add_dataset("garnet")
    for lbl, ctype in (
        ("M/ISYM", "Y"),
        ("BATCH", "B"),
        ("I", "J"),
        ("SIGI", "Q"),
        ("WAVEL", "R"),
        ("THETA", "R"),
        ("PHI", "R"),
    ):
        mtz.add_column(lbl, ctype)
    mtz.set_data(
        np.array(
            [[4.0, 0.0, 0.0, 1.0, 1001.0, 150.0, 8.0, 1.49, 0.42, 1.10]],
            dtype=np.float32,
        )
    )
    mtz.write_to_file(str(work_dir / "reflections.mtz"))

    # 1. Default wilson_prior_b is None -> --wilson-prior-b omitted; explicit value -> passed
    runner = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
    runner._local_work_dir = str(work_dir)
    rec_default = {"cancel_requested": True}
    runner._submit_local_subhkl_stage(
        "merge",
        {"images_file": str(merged_h5), "peaks_file": str(integ_h5)},
        existing_job_record=rec_default,
    )
    assert "--wilson-prior-b" not in rec_default["command_line"]

    rec_explicit = {"cancel_requested": True}
    runner._submit_local_subhkl_stage(
        "merge",
        {
            "images_file": str(merged_h5),
            "peaks_file": str(integ_h5),
            "wilson_prior_b": 15.0,
        },
        existing_job_record=rec_explicit,
    )
    assert "--wilson-prior-b=15.0" in rec_explicit["command_line"]

    # 2. Only sparkline header has the table button; Wulff merge scoreboard has none
    payload = build_rainbow_zone_payload(
        replay_h5=integ_h5, images_h5=merged_h5, include_panel_images=False
    )
    ui_html = render_rainbow_html(payload)
    assert 'id="btn-hist-open-table"' in ui_html
    assert 'data-merge-action="open-table"' not in ui_html

    # 3. Residual colorbar spans outlier-robust [resMin, resMax] with White midpoint at dataset median
    if shutil.which("node"):
        script_blocks = re.findall(r"<script>([\s\S]*?)</script>", ui_html)
        inline_js = script_blocks[-1]
        fn_names = [
            "getColorCrossover",
            "applyCrossoverCurve",
            "peakUnscaledIsigi",
            "peakMtzIsigi",
            "computeQuantileScaleBounds",
            "getPeakDataAvailability",
            "residualToColor",
            "buildColorscaleGradientSpec",
        ]
        extracted = []
        for fn_name in fn_names:
            m = re.search(rf"(?m)^  function {fn_name}\b[\s\S]*?^  \}}", inline_js)
            assert m is not None, f"Missing JS function {fn_name}"
            extracted.append(m.group(0))

        js_harness = f"""
const DEFAULT_COLOR_CROSSOVER = 0.38;
const ORBIT_PALETTE = ["#ff38b4", "#00e5ff"];
const state = {{ colorPercentile: 0.50, colorCrossover: 0.38, peakSource: "integrated", lamMin: 1.0, lamMax: 3.0 }};
const DATA = {{ mtz_overlay_name: "" }};
let _dynPredEpoch = 0;
let _peakDataAvailabilitySig = "";
let _peakDataAvailability = {{
  hasIsigi: false, hasMtzIsigi: false, hasResiduals: false, hasScales: false,
  isigiTop: 40.0, mtzIsigiTop: 40.0, resMin: -3.0, resMax: 3.0, resMedian: 0.0, resMean: 0.0,
}};
// Include a massive positive outlier (+590000.0) that previously blew up mean (μ=98341.0) and crushed moderate residuals!
const SAMPLE_PEAKS = [
  {{ id: 1, careless_residual: -2.0 }},
  {{ id: 2, careless_residual: 4.0 }},
  {{ id: 3, careless_residual: 8.0 }},
  {{ id: 4, careless_residual: 12.0 }},
  {{ id: 5, careless_residual: 16.0 }},
  {{ id: 6, careless_residual: 590000.0 }},
]; // sorted: [-2, 4, 8, 12, 16, 590000] -> median = 10.0, empirical CDF keeps 16.0 at P80% unaffected by +590000!
const PEAKS = [SAMPLE_PEAKS[0], SAMPLE_PEAKS[2]];
function isigiToColor() {{ return "rgba(0,0,0,1)"; }}
function scaleToColor() {{ return "rgba(0,0,0,1)"; }}
function hexToRgba() {{ return "rgba(0,0,0,1)"; }}
function wavelengthToColor() {{ return "rgba(0,0,0,1)"; }}
{"\n".join(extracted)}
const avail = getPeakDataAvailability();
const spec = buildColorscaleGradientSpec("residual", 0.50);
console.log(JSON.stringify({{
  avail: {{ resMin: avail.resMin, resMax: avail.resMax, resMedian: avail.resMedian, resMean: avail.resMean }},
  colMin: residualToColor(avail.resMin, 1.0),
  colBelowMed: residualToColor(4.0, 1.0),
  colMed: residualToColor(10.0, 1.0),
  colAboveMed: residualToColor(16.0, 1.0),
  colOutlier: residualToColor(590000.0, 1.0),
  spec,
}}));
"""
        proc = subprocess.run(
            ["node", "-e", js_harness], check=True, capture_output=True, text=True
        )
        out = json.loads(proc.stdout.strip())
        assert out["avail"]["resMedian"] == 10.0
        assert out["avail"]["resMean"] == 10.0
        assert out["avail"]["resMin"] == -2.0
        assert out["avail"]["resMax"] == 590000.0
        assert out["colMin"] == "rgba(30, 170, 255, 1.000)"
        assert out["colMed"] == "rgba(248, 250, 252, 1.000)"
        assert out["colOutlier"] == "rgba(244, 63, 94, 1.000)"
        assert out["colAboveMed"] != out["colOutlier"]
        # Positive residual +4.0 is below dataset median (+10.0), so it tints toward blue (b > r)
        r_lo, _, b_lo = [int(x) for x in re.findall(r"\d+", out["colBelowMed"])[:3]]
        assert b_lo > r_lo
        # Residual +16.0 is above dataset median (+10.0) at P80%, so it tints strongly toward red despite +590000 outlier!
        r_hi, _, b_hi = [int(x) for x in re.findall(r"\d+", out["colAboveMed"])[:3]]
        assert r_hi > b_hi + 40
        assert "μ=" not in out["spec"]["label"]
        assert " · 10.0 (P50%) · " in out["spec"]["label"]
        assert "P50%=10.0σ" in out["spec"]["tooltip"]
        assert "rgba(248, 250, 252, 1.000) 50.0%" in out["spec"]["gradient"]


def test_live_solve_dtheta_diagnostic_topbar_tooltips_and_controls_cleanup(
    tmp_path,
) -> None:
    """Verify image-grounded post-Solve & live micro-step Δθ, top-bar tooltips + non-overlapping subtitle CSS, and removal of redundant ALL / Walk buttons."""
    import json
    import os
    import re
    import shutil
    import subprocess

    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation

    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import (
        _raycast_hkl_on_detectors,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    # 1. Synthesize a realistic MANDI Ia-3d garnet detector count map in merged.h5 with true Bragg spots
    mandi_dets = {
        int(bk): Detector(dict(cfg)) for bk, cfg in beamlines["MANDI"].items()
    }
    active_bks = sorted(mandi_dets)[:12]
    sub_dets = {bk: mandi_dets[bk] for bk in active_bks}
    first_det = mandi_dets[active_bks[0]]
    H, W = int(first_det.n), int(first_det.m)

    U_true = Rotation.from_euler("xyz", [14.0, -22.0, 31.0], degrees=True).as_matrix()
    B_mat = np.diag([1.0 / 11.93, 1.0 / 11.93, 1.0 / 11.93])
    R_run = np.eye(3, dtype=float)
    grid = np.mgrid[-6:7, -6:7, -6:7].reshape(3, -1).T
    grid = grid[np.any(grid != 0, axis=1)]
    cast_true = _raycast_hkl_on_detectors(
        grid,
        U_true,
        B_mat,
        R_run,
        sub_dets,
        (1.0, 3.0),
        active_bids=set(active_bks),
        margin_frac=0.06,
    )
    assert len(cast_true["peaks_xyz"]) >= 10

    rng = np.random.default_rng(42)
    imgs = rng.poisson(2.0, size=(len(active_bks), H, W)).astype(np.float32)
    bk_to_slab = {bk: i for i, bk in enumerate(active_bks)}
    yy, xx = np.mgrid[0:H, 0:W]
    # Plant bright 2D Gaussian Bragg spots slightly displaced (~0.35 px -> finite ~0.06° sub-pixel residual even at U_true)
    for bk_p, r_p, c_p in zip(
        cast_true["bank"], cast_true["pixel_r"], cast_true["pixel_c"]
    ):
        si = bk_to_slab.get(int(bk_p))
        if si is not None and 8.0 <= r_p <= H - 9.0 and 8.0 <= c_p <= W - 9.0:
            r_true = float(r_p) + 0.35
            c_true = float(c_p) - 0.25
            imgs[si] += (
                260.0
                * np.exp(
                    -0.5 * (((yy - r_true) / 1.25) ** 2 + ((xx - c_true) / 1.25) ** 2)
                )
            ).astype(np.float32)

    merged_h5 = tmp_path / "merged.h5"
    with h5py.File(merged_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.create_dataset("images", data=imgs)
        hf.create_dataset("banks", data=np.array(active_bks, dtype=np.int32))
        hf.create_dataset("file_offsets", data=np.array([0], dtype=np.int32))
        hf.create_group("goniometer").create_dataset("R", data=R_run[None, :, :])
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([1.0, 3.0], dtype=np.float64)
        )
        sg = hf.create_group("sample")
        sg.attrs["space_group"] = "I a -3 d"
        sg.create_dataset(
            "cell", data=np.array([11.93, 11.93, 11.93, 90.0, 90.0, 90.0])
        )

    # 2. Simulate Solve micro-steps from coarse orientation error (0.75° -> 0.30° -> 0.00°)
    rms_by_step: list[float] = []
    partial_h5 = tmp_path / "solve.stages.h5"
    runner = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
    runner.active_job = {
        "stage_id": "solve",
        "state": "running",
        "partial_h5_path": str(partial_h5),
        "images_h5_path": str(merged_h5),
        "ladder_rung_idx": 1,
        "solve_params": {
            "instrument": "MANDI",
            "space_group": "I a -3 d",
            "wavelength_range": [1.0, 3.0],
            "d_min": 1.0,
        },
    }
    seen_seqs: list[int] = []

    for step_i, mis_deg in enumerate((0.75, 0.30, 0.0)):
        U_step = (
            Rotation.from_rotvec([np.deg2rad(mis_deg), 0.0, 0.0]).as_matrix() @ U_true
        )
        with h5py.File(partial_h5, "w") as sf:
            sf.attrs["instrument"] = "MANDI"
            sf.attrs["source"] = str(merged_h5)
            sg = sf.create_group("sample")
            sg.attrs["space_group"] = "I a -3 d"
            sg.create_dataset("U", data=U_step)
            sg.create_dataset("B", data=B_mat)
            sg.create_dataset(
                "cell", data=np.array([11.93, 11.93, 11.93, 90.0, 90.0, 90.0])
            )
            sf.create_group("goniometer").create_dataset("R", data=R_run[None, :, :])
            sf.create_group("instrument").create_dataset(
                "wavelength", data=np.array([1.0, 3.0], dtype=np.float64)
            )
        # Ensure distinct mtime_ns across rapid micro-step writes
        st_now = os.stat(partial_h5)
        os.utime(
            partial_h5,
            ns=(st_now.st_atime_ns, st_now.st_mtime_ns + (step_i + 1) * 1_000_000),
        )
        runner.active_job["ladder_rung_idx"] = step_i + 1
        ov_payload = runner.apply_active_solve_overlay({}, frame_index=0)
        seen_seqs.append(int(runner.active_job["partial_overlay"]["seq"]))
        err_vals = [
            float(p["angular_error_deg"])
            for p in ov_payload["peaks"]
            if p.get("has_angular_error")
        ]
        assert len(err_vals) >= 4
        rms_step = float(np.sqrt(np.mean(np.square(err_vals))))
        rms_by_step.append(rms_step)

    # Monotonic live partial_overlay sequence numbers and strictly decreasing Δθ across Solve micro-steps,
    # ending at a finite physical >0° value (not an idealized 0.000°)!
    assert seen_seqs == [1, 2, 3]
    assert rms_by_step[0] > rms_by_step[1] > rms_by_step[2] > 0.01

    # Post-Solve payload (solve.h5) also reports finite >0° Δθ and non-zero zone angular_rms_deg
    solve_payload = build_rainbow_zone_payload(
        replay_h5=partial_h5, images_h5=merged_h5, include_panel_images=False
    )
    solve_errs = [
        p["angular_error_deg"]
        for p in solve_payload["peaks"]
        if p.get("has_angular_error")
    ]
    assert len(solve_errs) >= 4
    assert float(np.sqrt(np.mean(np.square(solve_errs)))) > 0.01
    assert any(float(z["angular_rms_deg"]) > 0.0 for z in solve_payload["zones"])

    # 3. Verify blind_band.discover() writes .blind.h5 cleanly without temp-file collision
    import types

    import subhkl.search.blind_band as bb_mod

    orig_ll = bb_mod.lattice_ladder
    orig_rbg = bb_mod.refine_band_geometry
    try:

        def _fake_ladder(*_a, **kw):
            on_rung = kw.get("on_rung")
            if callable(on_rung):
                on_rung("rung_30", 30, U_true)
            return [(U_true, 12.5)]

        def _fake_refine(*_a, **_kw):
            return {
                "U": U_true,
                "score": 15.0,
                "radial_scale": 0.0,
                "axial_scale": 0.0,
                "global_rot_vec": np.zeros(3),
                "global_trans_m": np.zeros(3),
            }

        bb_mod.lattice_ladder = _fake_ladder
        bb_mod.refine_band_geometry = _fake_refine
        fake_obs = (
            np.full(32, active_bks[0], dtype=np.int32),
            np.linspace(16.0, 200.0, 32),
            np.linspace(16.0, 200.0, 32),
            np.linspace(5.0, 40.0, 32),
            np.ones(32),
            np.zeros(32, dtype=np.int32),
            np.zeros((32, 3)),
            np.linspace(1.0, 10.0, 32),
            np.repeat(R_run[None, :, :], 32, axis=0),
            np.repeat(np.array([[0.0, 1.0, 0.0]]), 32, axis=0),
        )
        solve_out = tmp_path / "subhkl_solve_test.h5"
        fake_args = types.SimpleNamespace(
            output=str(solve_out),
            nexus=str(merged_h5),
            instrument="MANDI",
            cell=[11.93, 11.93, 11.93, 90.0, 90.0, 90.0],
            space_group="I a -3 d",
            wl_min=2.0,
            wl_max=4.0,
            d_min=1.5,
            finder="peaks",
            min_deviance=0.0,
            runs=[0],
            stable_runs=[0],
            modes=frozenset({"sample"}),
            _cached_obs_kin=fake_obs,
        )
        blind_written = bb_mod.discover(fake_args, samples=16, candidates=1, seed=0)
        assert os.path.isfile(blind_written)
        assert not list(tmp_path.glob("*.tmp.h5"))
        with h5py.File(blind_written, "r") as bf_chk:
            assert "orientations" in bf_chk["search"]
            assert "full_scores" in bf_chk["search"]
            assert str(bf_chk["search"].attrs["ladder_stage"]) == "blind_selected"
    finally:
        bb_mod.lattice_ladder = orig_ll
        bb_mod.refine_band_geometry = orig_rbg

    # 4. Verify HTML/CSS/JS top-bar tooltips (1.5s custom tooltip), non-overlapping .subtitle-tag CSS, and removed ALL / Walk buttons
    html_doc = render_rainbow_html(solve_payload)
    assert 'id="btn-zones-all"' not in html_doc
    assert 'id="btn-ladder-play"' not in html_doc
    assert "WALK SOLVE STEPS" not in html_doc
    assert "switch steps with &lt; &gt; arrows" in html_doc
    assert 'id="ui-hover-tooltip"' in html_doc
    assert "const UI_TOOLTIP_DELAY_MS = 1500;" in html_doc

    # Every top-bar field, HUD pill, and control button has a descriptive tooltip attribute
    for top_id in (
        "app-title",
        "app-version",
        "subtitle-tag",
        "hud-peaks-pill",
        "hud-rms-pill",
        "hud-zones-pill",
        "hud-gonio-pill",
        "hud-frame-ctrl",
        "btn-frame-prev",
        "sel-frame",
        "btn-frame-next",
        "sel-galaxy-h5",
        "btn-proj-toggle",
        "peak-source-group",
        "btn-peaks-pred",
        "btn-peaks-integ",
        "btn-zone-arcs",
        "btn-pixel-mask",
        "btn-hires-tex",
        "btn-heatmap-cmap",
        "btn-stereo-toggle",
        "btn-reset-view",
        "btn-new-tab",
    ):
        m_tag = re.search(rf'<[^>]*\bid="{top_id}"[^>]*>', html_doc)
        assert m_tag is not None, f"Missing top-bar element #{top_id}"
        assert 'title="' in m_tag.group(0), f"Missing title tooltip on #{top_id}"

    # Verify .brand-block clips overflow and .subtitle-tag shrinks instead of bleeding into PEAKS pill
    brand_css = re.search(r"\.brand-block\s*\{[^}]*\}", html_doc)
    sub_css = re.search(r"\.subtitle-tag\s*\{[^}]*\}", html_doc)
    hud_css = re.search(r"\.hud-metrics\s*\{[^}]*\}", html_doc)
    assert brand_css is not None and "overflow: hidden;" in brand_css.group(0)
    assert sub_css is not None and "flex: 0 1 auto;" in sub_css.group(0)
    assert hud_css is not None and "flex-shrink: 0;" in hud_css.group(0)

    # Verify JS _extractMeasSnapshot + syncTopBarTooltips + updateHUD preserve frame Δθ and format Peaks tooltip
    if shutil.which("node"):
        script_blocks = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
        inline_js = script_blocks[-1]
        fn_extract: list[str] = []
        for fn in ("setElementTooltip", "syncTopBarTooltips", "updateHUD"):
            m_fn = re.search(rf"function {fn}\b[\s\S]*?\n  \}}(?=\n)", inline_js)
            assert m_fn is not None, f"Missing JS function {fn}"
            fn_extract.append(m_fn.group(0))
        node_script = f"""
let _uiTipActiveEl = null;
let _uiTipStashedEls = [];
let _uiTipVisible = false;
const uiHoverTooltip = null;
const els = {{
  "val-indexed": {{ textContent: "", title: "" }},
  "val-rms": {{ textContent: "", title: "" }},
  "val-zones": {{ textContent: "", title: "" }},
  "val-gonio": {{ textContent: "", title: "" }},
  "hud-peaks-pill": {{ title: "" }},
  "hud-rms-pill": {{ title: "" }},
  "hud-zones-pill": {{ title: "" }},
  "hud-gonio-pill": {{ title: "" }},
  "hud-frame-ctrl": {{ title: "" }},
  "sel-frame": {{ selectedIndex: 0, options: [{{ textContent: "MANDI_12737.nxs.h5 (ω=15.0°)" }}], title: "" }},
  "sel-galaxy-h5": {{ selectedIndex: 0, options: [{{ value: "h1", textContent: "solve.h5" }}], title: "" }},
  "app-title": {{ title: "" }},
  "app-version": {{ textContent: "v0.4.0", title: "" }},
  "btn-zone-arcs": {{ title: "" }},
  "btn-heatmap-cmap": {{ title: "" }},
  "btn-stereo-toggle": {{ title: "" }},
}};
const subTag = {{ textContent: "MANDI • I a -3 d • INTEGRATED • 3D TENSOR ELLIPSES", title: "" }};
globalThis.document = {{
  getElementById: (id) => els[id] || null,
  querySelector: (sel) => sel === ".subtitle-tag" ? subTag : null,
}};
const DATA = {{
  cell: [11.93, 11.93, 11.93, 90, 90, 90],
  frames: [{{ motor_summary: "ω=15.0°, χ=45.0°, φ=120.0°" }}],
  frame_index: 0,
}};
const state = {{
  peakSource: "predicted",
  lamMin: 1.0,
  lamMax: 3.0,
  dMin: 1.0,
  activeZones: new Set([1, 2, 3]),
  maxZoneOrder: 1,
  showZoneArcs: true,
  heatmapStyle: "magma",
  showStereo: true,
}};
const PEAKS = [
  {{ indexed: true, peak_source: "predicted", angular_error_deg: 0.08, has_angular_error: true, zone_ids: [1] }},
  {{ indexed: true, peak_source: "predicted", angular_error_deg: 0.0, has_angular_error: false, zone_ids: [1] }},
  {{ indexed: true, peak_source: "predicted", angular_error_deg: 0.06, has_angular_error: true, zone_ids: [] }},
];
function isPeakVisible() {{ return true; }}
function isPeakCapturedByActiveZone(p) {{ return (p.zone_ids || []).length > 0; }}
function drawHistogram() {{}}
function fitHorizontalBars() {{}}
{"\n".join(fn_extract)}
updateHUD();
console.log(JSON.stringify({{
  indexed: els["val-indexed"].textContent,
  rms: els["val-rms"].textContent,
  peaksTip: els["hud-peaks-pill"].title,
  rmsTip: els["hud-rms-pill"].title,
  gonioTip: els["hud-gonio-pill"].title,
  frameTip: els["sel-frame"].title,
  subTip: subTag.title,
}}));
"""
        proc = subprocess.run(
            ["node", "-e", node_script], check=True, capture_output=True, text=True
        )
        res = json.loads(proc.stdout.strip())
        assert res["indexed"] == "2 / 3 (66.7%)"
        # Unobserved predicted peak (has_angular_error: false) does NOT dilute 0.080° down to 0.057°!
        assert res["rms"] == "0.080°"
        assert "2 indexed reflections captured by enabled Laue zones" in res["peaksTip"]
        assert "out of 3 visible predicted reflections" in res["peaksTip"]
        assert "across 1 matched detector spots" in res["rmsTip"]
        assert "ω=15.0°, χ=45.0°, φ=120.0°" in res["gonioTip"]
        assert "MANDI_12737.nxs.h5" in res["frameTip"]
        assert "MANDI • I a -3 d • INTEGRATED" in res["subTip"]


def test_tmpdir_exit_cleanup_removes_app_artifacts_safely(tmp_path) -> None:
    """Verify TMPDIR workspaces/caches/uploads are cleaned at normal exit and SIGTERM without touching external or unregistered files."""
    import os
    import signal
    import subprocess
    import sys
    import tempfile

    from subhkl.viz.nova_rainbow_view import (
        NovaGalaxyPipelineRunner,
        cleanup_registered_temp_artifacts,
        register_temp_artifact,
    )

    isolated_tmp = tmp_path / "app_tmp"
    outside_dir = tmp_path / "user_data"
    isolated_tmp.mkdir()
    outside_dir.mkdir()
    outside_file = outside_dir / "precious.h5"
    outside_file.write_bytes(b"keep")
    unregistered_tmp_file = isolated_tmp / "other_process.txt"
    unregistered_tmp_file.write_text("keep", encoding="utf-8")

    orig_tempdir = tempfile.tempdir
    try:
        tempfile.tempdir = str(isolated_tmp)
        # Never track TMPDIR root itself or paths outside TMPDIR
        assert register_temp_artifact(isolated_tmp) is None
        assert register_temp_artifact(outside_file) is None
        assert register_temp_artifact(outside_dir) is None

        runner = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
        work_d, _ = runner._work_dir_for_reduce(
            ["/SNS/MANDI/MANDI_12737.nxs.h5"], str(isolated_tmp)
        )
        (tmp_path / "dummy").touch()
        with open(os.path.join(work_d, "merged.h5"), "wb") as fh:
            fh.write(b"h5")
        gx_cache = isolated_tmp / "subhkl_gx_deadbeef_solve.h5"
        gx_cache.write_bytes(b"gx")
        assert register_temp_artifact(gx_cache) == str(gx_cache)

        removed = cleanup_registered_temp_artifacts()
        assert os.path.abspath(work_d) in removed
        assert str(gx_cache) in removed
        assert not os.path.exists(work_d)
        assert not gx_cache.exists()
        assert outside_file.exists()
        assert unregistered_tmp_file.exists()
    finally:
        tempfile.tempdir = orig_tempdir

    # Subprocess exit test 1: normal interpreter exit triggers atexit TMPDIR cleanup
    sub_tmp_1 = tmp_path / "sub_tmp_atexit"
    sub_tmp_1.mkdir()
    env_1 = dict(os.environ, TMPDIR=str(sub_tmp_1))
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, pathlib; "
                "from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner, register_temp_artifact; "
                "r = NovaGalaxyPipelineRunner(galaxy_url='', galaxy_api_key=''); "
                "wd, _ = r._work_dir_for_reduce(['MANDI_1.nxs.h5'], os.environ['TMPDIR']); "
                "pathlib.Path(wd, 'merged.h5').write_bytes(b'123'); "
                "up = pathlib.Path(os.environ['TMPDIR']) / 'subhkl_upload_1'; up.mkdir(); "
                "register_temp_artifact(up)"
            ),
        ],
        env=env_1,
        check=True,
    )
    assert list(sub_tmp_1.iterdir()) == []

    # Subprocess exit test 2: SIGTERM triggers signal-handler TMPDIR cleanup
    sub_tmp_2 = tmp_path / "sub_tmp_sigterm"
    sub_tmp_2.mkdir()
    env_2 = dict(os.environ, TMPDIR=str(sub_tmp_2))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import os, pathlib, time; "
                "from subhkl.viz.nova_rainbow_view import install_exit_cleanup_signal_handlers, register_temp_artifact; "
                "install_exit_cleanup_signal_handlers(); "
                "wd = pathlib.Path(os.environ['TMPDIR']) / 'subhkl_work_sigterm'; wd.mkdir(); "
                "(wd / 'solve.h5').write_bytes(b'abc'); "
                "register_temp_artifact(wd); "
                "print('READY', flush=True); "
                "time.sleep(30)"
            ),
        ],
        env=env_2,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "READY"
    assert (sub_tmp_2 / "subhkl_work_sigterm" / "solve.h5").is_file()
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    assert list(sub_tmp_2.iterdir()) == []


def test_remote_galaxy_slim_h5_range_sync_skips_tof_events_and_links_provenance(
    tmp_path, monkeypatch
) -> None:
    """Verify multi-GB-style TOF merged.h5 sync uses HTTP Range to skip events/tof while preserving TOF & Galaxy provenance."""
    import errno
    import http.server
    import shutil
    import socketserver
    import threading
    from types import SimpleNamespace

    import subhkl.viz.nova_rainbow_view as nrv
    from subhkl.io.events import write_events
    from subhkl.viz.rainbow_zones import _h5_has_tof_data

    ds_id = "ed4822a65a7baa02"
    remote_h5 = tmp_path / "remote_scan_master.h5"
    rng = np.random.default_rng(42)
    imgs_true = rng.poisson(3, size=(8, 64, 64)).astype(np.float32)
    with h5py.File(remote_h5, "w") as hf:
        hf.create_dataset(
            "images", data=imgs_true, chunks=(1, 64, 64), compression="lzf"
        )
        hf.create_dataset("bank_ids", data=np.arange(1, 9, dtype=np.int32))
        hf.create_dataset("file_offsets", data=np.array([0, 4], dtype=np.int64))
        hf.create_dataset(
            "files",
            data=["MANDI_11324.nxs.h5", "MANDI_11325.nxs.h5"],
            dtype=h5py.string_dtype(),
        )
        gon = hf.create_group("goniometer")
        gon.create_dataset("angles", data=np.zeros((8, 3), dtype=np.float64))
        gon.create_dataset("axes", data=np.eye(3, dtype=np.float64))
        # Large incompressible random float32 TOF payload (~8 MB >> 2D images ~120 KB)
        tof_chunks = [
            rng.uniform(1000.0, 16666.0, size=260_000).astype(np.float32)
            for _ in range(8)
        ]
        write_events(hf, tof_chunks, l1=30.0)
        # Trailing sample metadata written after events/tof (just like run_merge_images)
        smp = hf.create_group("sample")
        for k, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"),
            (11.93, 11.93, 11.93, 90.0, 90.0, 90.0),
        ):
            smp[k] = v
        smp["space_group"] = b"I a -3 d"

    full_bytes = remote_h5.read_bytes()
    full_size = len(full_bytes)
    assert full_size > 6 * 1024 * 1024

    served_log: list[tuple[str, str, int]] = []

    class _RangeGalaxyHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            path_only = self.path.split("?", 1)[0]
            if path_only == f"/api/datasets/{ds_id}":
                doc = json.dumps(
                    {
                        "id": ds_id,
                        "name": "MANDI_11324_MANDI_11338_merged.h5",
                        "state": "ok",
                        "file_size": full_size,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(doc)))
                self.end_headers()
                self.wfile.write(doc)
                return
            if path_only == f"/api/datasets/{ds_id}/display":
                rng_hdr = self.headers.get("Range") or ""
                accel_hdr = self.headers.get("X-Accel-Buffering") or ""
                m = re.match(r"^bytes=(\d+)-(\d*)$", rng_hdr)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else (full_size - 1)
                    end = min(end, full_size - 1)
                    body = full_bytes[start : end + 1]
                    served_log.append((rng_hdr, accel_hdr, len(body)))
                    self.send_response(206)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end}/{full_size}"
                    )
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                served_log.append(("", accel_hdr, full_size))
                self.send_response(200)
                self.send_header("Content-Length", str(full_size))
                self.end_headers()
                self.wfile.write(full_bytes)
                return
            self.send_response(404)
            self.end_headers()

    class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    monkeypatch.setattr(nrv, "_SLIM_REMOTE_H5_MIN_BYTES", 512 * 1024)
    monkeypatch.setattr(nrv, "_HTTP_RANGE_BLOCK_BYTES", 64 * 1024)

    srv = _ThreadedServer(("127.0.0.1", 0), _RangeGalaxyHandler)
    port = srv.server_address[1]
    thr = threading.Thread(target=srv.serve_forever, daemon=True)
    thr.start()
    try:
        runner = nrv.NovaGalaxyPipelineRunner(
            galaxy_url=f"http://127.0.0.1:{port}",
            galaxy_api_key="test_key",
        )
        runner.history_id = "hist_1"
        dest_h5 = tmp_path / "subhkl_work_MANDI_11324_MANDI_11338" / "merged.h5"
        job_rec: dict[str, object] = {"stdout": ""}
        n_written = runner._download_dataset_to_file(
            ds_id,
            str(dest_h5),
            job_record=job_rec,
            label="MANDI_11324_MANDI_11338_merged.h5",
        )
        assert n_written > 0
        assert dest_h5.is_file()
        # Local slim cache and total network bytes fetched must be a tiny fraction of full_size
        total_http_bytes = sum(sz for _, _, sz in served_log)
        assert dest_h5.stat().st_size < full_size * 0.15
        assert total_http_bytes < full_size * 0.25
        assert all(rng_h.startswith("bytes=") for rng_h, _, _ in served_log)
        assert all(acc_h == "no" for _, acc_h, _ in served_log)
        assert job_rec.get("_last_slim_remote_bytes") == full_size

        with h5py.File(dest_h5, "r") as hf_slim:
            assert "events" not in hf_slim
            assert bool(hf_slim.attrs.get("has_tof_events")) is True
            assert str(hf_slim.attrs.get("galaxy_dataset_id")) == ds_id
            assert int(hf_slim.attrs.get("galaxy_file_size")) == full_size
            assert np.allclose(hf_slim["images"][()], imgs_true)
            assert float(hf_slim["sample/a"][()]) == pytest.approx(11.93)
        assert _h5_has_tof_data(dest_h5) is True

        # Fresh runner (simulating cleared in-memory map) still links back to ds_id via HDF5 provenance without re-upload!
        runner2 = nrv.NovaGalaxyPipelineRunner(
            galaxy_url=f"http://127.0.0.1:{port}",
            galaxy_api_key="test_key",
        )
        runner2.history_id = "hist_1"
        upload_called = False

        class _FakeDatasetsClient:
            def show_dataset(self, q_id: str) -> dict[str, object]:
                assert q_id == ds_id
                return {
                    "id": ds_id,
                    "name": "MANDI_11324_MANDI_11338_merged.h5",
                    "state": "ok",
                    "deleted": False,
                    "file_size": full_size,
                }

            def get_datasets(self, **_kw: object) -> list[dict[str, object]]:
                return []

        class _FakeToolsClient:
            def upload_file(self, *_a: object, **_kw: object) -> dict[str, object]:
                nonlocal upload_called
                upload_called = True
                return {}

        fake_gi = SimpleNamespace(
            datasets=_FakeDatasetsClient(),
            tools=_FakeToolsClient(),
        )
        fake_store = SimpleNamespace(
            history_id="hist_1",
            nova_connection=SimpleNamespace(galaxy_instance=fake_gi),
        )
        monkeypatch.setattr(
            runner2, "attach_dataset_cell_metadata_from_file", lambda *_a, **_k: None
        )
        linked_id = runner2._ensure_galaxy_h5_dataset(
            fake_store, str(dest_h5), {"stdout": "", "params": {}}
        )
        assert linked_id == ds_id
        assert upload_called is False

        # Scratch ENOSPC preflight fails immediately instead of hanging/evicting pod
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda _p: shutil._ntuple_diskusage(100, 90, 10 * 1024 * 1024),
        )
        with pytest.raises(OSError) as exc_info:
            runner._check_scratch_space(
                str(tmp_path), 6400 * 1024 * 1024, "MANDI_merged.h5"
            )
        assert exc_info.value.errno == errno.ENOSPC
    finally:
        srv.shutdown()
        srv.server_close()


def test_local_staging_gpu_mode_and_galaxy_careless_merge_routing(
    tmp_path, monkeypatch
) -> None:
    """Verify local-staging GPU routing, Pulsar staging path handling, and compact-only NDIP uploads."""
    import os
    import stat

    from subhkl.viz.nova_rainbow_view import (
        NovaGalaxyPipelineRunner,
        _get_staging_root,
        _is_readonly_galaxy_dat,
    )

    # Simulate Pulsar NVMe working directory (/data/staging/<job>/working/subhkl_staging)
    pulsar_staging = (
        tmp_path / "data" / "staging" / "282645" / "working" / "subhkl_staging"
    )
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(pulsar_staging))
    resolved_staging = _get_staging_root()
    assert resolved_staging == str(pulsar_staging)
    assert (os.stat(pulsar_staging).st_mode & 0o777) == 0o777

    # Read-only Galaxy dataset_*.dat inside /staging/inputs is detected, whereas our
    # writable subhkl_staging / subhkl_work_* paths under /data/staging/<job>/working are NOT.
    ro_dat = tmp_path / "data" / "staging" / "282645" / "inputs" / "dataset_abc123.dat"
    assert _is_readonly_galaxy_dat(ro_dat) is True
    runner = NovaGalaxyPipelineRunner(
        galaxy_url="https://ndip-test.ornl.gov",
        galaxy_api_key="mock_key",
        history_id="hist_local_stage",
    )
    work_d, merged_lbl = runner._work_dir_for_reduce(
        ["/SNS/MANDI/IPTS-36185/nexus/MANDI_13171.nxs.h5"], str(pulsar_staging)
    )
    assert work_d.startswith(str(pulsar_staging))
    assert stat.S_IMODE(os.stat(work_d).st_mode) == 0o777
    staged_merged = os.path.join(work_d, merged_lbl)
    with h5py.File(staged_merged, "w") as hf:
        hf.create_dataset("images", data=np.zeros((1, 8, 8), dtype=np.float32))
    assert _is_readonly_galaxy_dat(staged_merged) is False
    wd2, mp2 = runner._resolve_work_dir_and_merged(staged_merged)
    assert wd2 == work_d
    assert mp2 == staged_merged

    # Default (no SUBHKL_EXEC_MODE, no EP_PATH) preserves remote Galaxy mode for unit tests
    monkeypatch.delenv("SUBHKL_EXEC_MODE", raising=False)
    monkeypatch.delenv("EP_PATH", raising=False)
    assert runner._is_local_staging_mode() is False

    # Inside NDIP interactive tool container (EP_PATH set + /SNS or /HFIR mounted), local staging activates automatically
    monkeypatch.setenv("EP_PATH", "/interactivetool/ep")
    monkeypatch.setattr(runner, "_has_local_sns_hfir_fs", lambda: True)
    assert runner._is_local_staging_mode() is True

    # Explicit SUBHKL_EXEC_MODE=galaxy overrides EP_PATH
    monkeypatch.setenv("SUBHKL_EXEC_MODE", "galaxy")
    assert runner._is_local_staging_mode() is False

    # Explicit SUBHKL_EXEC_MODE=local routes Reduce, Solve, Integrate locally, while
    # routing Merge through Galaxy whenever careless/podman is absent locally.
    monkeypatch.setenv("SUBHKL_EXEC_MODE", "local")
    monkeypatch.setattr(runner, "_has_local_careless_runner", lambda: False)

    local_dispatched: list[str] = []
    galaxy_dispatched: list[str] = []

    def _fake_local_stage(stage_id, params, on_complete_h5=None, fallback_reason=None):
        local_dispatched.append(stage_id)
        return {
            "stage_id": stage_id,
            "mode": "container_gpu",
            "reason": fallback_reason,
        }

    def _fake_galaxy_thread(stage_id, params, job_record, on_complete_h5=None):
        galaxy_dispatched.append(stage_id)
        job_record["state"] = "ok"

    monkeypatch.setattr(runner, "_submit_local_subhkl_stage", _fake_local_stage)
    monkeypatch.setattr(runner, "_dispatch_nova_galaxy_thread", _fake_galaxy_thread)

    for stg in ("reduce", "solve", "integrate"):
        res_s = runner.submit_galaxy_stage(stg, {"images_file": staged_merged})
        assert res_s["mode"] == "container_gpu"
    res_m = runner.submit_galaxy_stage("merge", {"images_file": staged_merged})
    assert res_m["mode"] == "ndip_galaxy_rest"
    for _ in range(20):
        if "merge" in galaxy_dispatched:
            break
        import time as _t

        _t.sleep(0.01)
    assert local_dispatched == ["reduce", "solve", "integrate"]
    assert galaxy_dispatched == ["merge"]

    # When careless/podman IS available locally, Merge can also run locally
    monkeypatch.setattr(runner, "_has_local_careless_runner", lambda: True)
    res_m_local = runner.submit_galaxy_stage("merge", {"images_file": staged_merged})
    assert res_m_local["mode"] == "container_gpu"
    assert local_dispatched[-1] == "merge"

    # Compact upload helper skips oversized files (> max_bytes) and deduplicates identical (size, mtime_ns)
    upload_calls: list[str] = []

    def _fake_ensure_ds(_store, lpath, _rec, force_upload=False, file_ext="h5"):
        upload_calls.append(f"{os.path.basename(lpath)}:{file_ext}:{force_upload}")
        return f"ds_{len(upload_calls)}"

    monkeypatch.setattr(runner, "_ensure_galaxy_h5_dataset", _fake_ensure_ds)
    mtz_file = os.path.join(work_d, "reflections.mtz")
    with open(mtz_file, "wb") as fm:
        fm.write(b"MTZ " * 256)
    ds1 = runner._upload_compact_ndip_artifact(object(), mtz_file, "mtz")
    ds2 = runner._upload_compact_ndip_artifact(object(), mtz_file, "mtz")
    assert ds1 == "ds_1"
    assert ds2 == "ds_1"
    assert upload_calls == ["reflections.mtz:mtz:True"]

    # Oversized artifact (e.g. merged.h5 exceeding compact ceiling) is never uploaded
    rec_skip: dict[str, str] = {"stdout": ""}
    skipped = runner._upload_compact_ndip_artifact(
        object(), mtz_file, "h5", job_record=rec_skip, max_bytes=128
    )
    assert skipped is None
    assert "Skipping Galaxy upload" in rec_skip["stdout"]
    assert len(upload_calls) == 1

    # Re-running Integrate invalidates cached signature so updated reflections.mtz uploads fresh
    runner.invalidate_subsequent_stages("integrate", work_d)
    with open(mtz_file, "wb") as fm:
        fm.write(b"MTZ " * 300)
    ds3 = runner._upload_compact_ndip_artifact(object(), mtz_file, "mtz")
    assert ds3 == "ds_2"
    assert len(upload_calls) == 2

    # check_neutron_paths in local staging mode verifies directly via local filesystem without calling Galaxy
    local_nxs = tmp_path / "MANDI_13171.nxs.h5"
    with h5py.File(local_nxs, "w") as hf_n:
        hf_n.create_dataset("entry/title", data=np.bytes_("MANDI local staging test"))
    nld_called = False

    def _forbid_has_tool(_tid: str) -> bool:
        nonlocal nld_called
        nld_called = True
        return True

    monkeypatch.setattr(runner, "_has_tool_on_galaxy", _forbid_has_tool)
    chk = runner.check_neutron_paths({"paths": [str(local_nxs)], "ingest_mode": "file"})
    assert chk["ok"] is True
    assert chk["checked_via"] == "local_filesystem"
    assert nld_called is False


def test_cross_instance_galaxy_history_reuse_integrated_restart_and_hydrate_reduce(
    tmp_path, monkeypatch
):
    """Verify cross-instance NDIP History discovery, local cache hit vs on-the-fly Reduce hydration, and Merge restart from integrated.h5."""
    import os
    import threading

    import h5py
    import numpy as np

    from subhkl.viz.nova_rainbow_view import (
        NovaGalaxyPipelineRunner,
        RainbowZonesViewModel,
    )
    from subhkl.viz.rainbow_zones import _discover_images_h5

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    sns_root = tmp_path / "SNS"
    nxs_dir = sns_root / "MANDI" / "IPTS-36185" / "nexus"
    nxs_dir.mkdir(parents=True)
    raw_nxs = nxs_dir / "MANDI_13308.nxs.h5"
    with h5py.File(raw_nxs, "w") as hf_raw:
        hf_raw.create_dataset("entry/title", data=np.bytes_("MANDI 13308"))

    monkeypatch.setenv("SUBHKL_EXEC_MODE", "local")
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(staging_dir))
    monkeypatch.setenv("GALAXY_URL", "https://ndip-test.ornl.gov")
    monkeypatch.setenv("GALAXY_API_KEY", "fake-ndip-key")
    # Simulate NDIP Interactive Tool injecting a fresh per-launch HISTORY_ID
    monkeypatch.setenv("HISTORY_ID", "hist_fresh_it_launch")

    runner = NovaGalaxyPipelineRunner()
    monkeypatch.setitem(runner.SNS_HFIR_ROOTS, "SNS", str(sns_root))
    api_log: list[tuple[str, str]] = []

    def _fake_api(method, path, payload=None, raw=False, timeout=30.0):
        api_log.append((method, path))
        if path == "/api/histories?keys=id,name,deleted":
            return [
                {
                    "id": "hist_fresh_it_launch",
                    "name": "Unnamed history",
                    "deleted": False,
                },
                {
                    "id": "hist_subhkl_shared",
                    "name": "IMAGINE-X SubHKL",
                    "deleted": False,
                },
            ]
        if path.startswith("/api/histories/hist_subhkl_shared/contents?"):
            return [
                {
                    "id": "ds_solve_prev",
                    "hid": 41,
                    "name": "MANDI_13308_solve.h5",
                    "state": "ok",
                    "extension": "h5",
                    "deleted": False,
                    "history_content_type": "dataset",
                    "update_time": "2026-04-20T12:00:00.000000",
                },
                {
                    "id": "ds_int_prev",
                    "hid": 42,
                    "name": "MANDI_13308_integrated.h5",
                    "state": "ok",
                    "extension": "h5",
                    "deleted": False,
                    "history_content_type": "dataset",
                    "update_time": "2026-04-20T12:05:00.000000",
                },
            ]
        if path.startswith("/api/histories/hist_fresh_it_launch/contents?"):
            return [
                {
                    "id": "a1b2c3d4e5f60718",
                    "hid": 1,
                    "name": "user_uploaded.mtz",
                    "state": "ok",
                    "extension": "mtz",
                    "deleted": False,
                    "history_content_type": "dataset",
                    "update_time": "2026-04-20T11:55:00.000000",
                }
            ]
        return {}

    runner._api_request = _fake_api

    # 1) Cross-instance history visibility: sees earlier solve.h5 + integrated.h5 in 'IMAGINE-X SubHKL' AND launch history items
    hist_items = runner.list_history_datasets(use_cached=True)
    names = [d["name"] for d in hist_items]
    assert "MANDI_13308_integrated.h5" in names
    assert "MANDI_13308_solve.h5" in names
    assert "user_uploaded.mtz" in names
    assert runner.history_id == "hist_subhkl_shared"

    # touch_dataset invalidates _cached_history_datasets so newly uploaded datasets appear immediately
    assert runner._cached_history_datasets is not None
    runner.touch_dataset("ds_int_prev")
    assert runner._cached_history_datasets is None

    # Annotation sync targets the dataset's actual owning history
    runner.attach_dataset_cell_metadata(
        "a1b2c3d4e5f60718", [50, 60, 70, 90, 90, 90], "P 21 21 21"
    )
    import time as _time

    for _ in range(20):
        if any(m == "PUT" for m, _ in api_log):
            break
        _time.sleep(0.01)
    put_paths = [p for m, p in api_log if m == "PUT"]
    assert any(
        "/api/histories/hist_fresh_it_launch/contents/a1b2c3d4e5f60718" in p
        for p in put_paths
    )

    # 2) Legacy solve.h5 (only attrs['source'], no 'source_files') + local cache HIT -> finds merged.h5 without re-running Reduce
    work_dir = staging_dir / "subhkl_work_MANDI_13308"
    work_dir.mkdir()
    cached_merged = work_dir / "MANDI_13308_merged.h5"
    with h5py.File(cached_merged, "w") as hf_m:
        hf_m.attrs["instrument"] = "MANDI"
        hf_m.create_dataset("images", data=np.zeros((1, 4, 4), dtype=np.float32))
        dt_s = h5py.string_dtype(encoding="utf-8")
        hf_m.create_dataset("source_files", data=[str(raw_nxs)], dtype=dt_s)
        hf_m.create_dataset("files", data=["MANDI_13308.nxs.h5"], dtype=dt_s)
        hf_m.create_dataset("instrument/wavelength", data=[2.0, 4.0])

    legacy_solve = staging_dir / "subhkl_gx_ds_solve_prev_solve.h5"
    with h5py.File(legacy_solve, "w") as hf_s:
        hf_s.attrs["instrument"] = "MANDI"
        hf_s.attrs["source"] = (
            "/tmp/old_instance/subhkl_work_MANDI_13308/MANDI_13308_merged.h5"
        )
        hf_s.create_dataset("sample/B", data=np.diag([0.02, 0.02, 0.01]))
        hf_s.create_dataset("sample/U", data=np.eye(3))
        hf_s.create_dataset("sample/space_group", data=np.bytes_("P 21 21 21"))

    resolved_wd, resolved_m = runner._resolve_work_dir_and_merged(str(legacy_solve))
    assert os.path.abspath(resolved_wd) == str(work_dir)
    assert os.path.abspath(resolved_m) == str(cached_merged)
    assert (work_dir / "solve.h5").is_file()
    disc_hit = _discover_images_h5(legacy_solve)
    assert disc_hit is not None and os.path.abspath(disc_hit) == str(cached_merged)

    # Provenance helper stamps full `source_files` + `files` + `instrument/wavelength` onto derived HDF5s
    int_h5 = work_dir / "integrated.h5"
    with h5py.File(int_h5, "w") as hf_i:
        hf_i.attrs["instrument"] = "MANDI"
        hf_i.attrs["source"] = str(cached_merged)
        hf_i.create_dataset("sample/a", data=50.0)
        hf_i.create_dataset("sample/b", data=60.0)
        hf_i.create_dataset("sample/c", data=70.0)
        hf_i.create_dataset("sample/alpha", data=90.0)
        hf_i.create_dataset("sample/beta", data=90.0)
        hf_i.create_dataset("sample/gamma", data=90.0)
        hf_i.create_dataset("sample/space_group", data=np.bytes_("P 21 21 21"))
        hf_i.create_dataset("peaks/h", data=np.array([1, 0], dtype=np.int32))
        hf_i.create_dataset("peaks/k", data=np.array([0, 1], dtype=np.int32))
        hf_i.create_dataset("peaks/l", data=np.array([0, 0], dtype=np.int32))
        hf_i.create_dataset(
            "peaks/intensity", data=np.array([120.0, 85.0], dtype=np.float32)
        )
        hf_i.create_dataset("peaks/sigma", data=np.array([11.0, 9.5], dtype=np.float32))
        hf_i.create_dataset("peaks/lambda", data=np.array([2.5, 3.1], dtype=np.float32))
        hf_i.create_dataset(
            "peaks/two_theta", data=np.array([0.4, 0.5], dtype=np.float32)
        )
        hf_i.create_dataset(
            "peaks/azimuthal", data=np.array([0.1, -0.2], dtype=np.float32)
        )
        hf_i.create_dataset(
            "peaks/xyz",
            data=np.array([[0.1, 0.0, 0.4], [0.0, 0.2, 0.4]], dtype=np.float32),
        )
        hf_i.create_dataset("peaks/run_index", data=np.array([0, 0], dtype=np.int32))
        hf_i.create_dataset("peaks/bank", data=np.array([1, 1], dtype=np.int32))
        hf_i.create_dataset("goniometer/R", data=np.eye(3)[None, ...])
    runner._stamp_derived_h5_provenance(
        str(int_h5),
        merged_path=str(cached_merged),
        params={"space_group": "P 21 21 21"},
    )
    with h5py.File(int_h5, "r") as hf_chk:
        assert "source_files" in hf_chk
        assert list(hf_chk["source_files"].asstr()[()]) == [str(raw_nxs)]

    # 3) Restart Stage 4 (Merge) from integrated.h5 even when reflections.mtz and merged.h5 are absent
    os.remove(cached_merged)
    assert not (work_dir / "reflections.mtz").exists()
    rebuilt_mtz = runner._ensure_reflections_mtz_from_integrated(
        str(work_dir), peaks_file=str(int_h5), space_group="P 21 21 21"
    )
    assert rebuilt_mtz is not None and os.path.isfile(rebuilt_mtz)
    assert os.path.getsize(rebuilt_mtz) > 0

    # 4) Local cache MISS -> extracts recipe (both from source_files AND legacy attrs['source']) & triggers non-destructive Reduce hydration
    recipe_from_src = runner._extract_reduce_recipe_from_derived_h5(str(int_h5))
    assert recipe_from_src is not None
    assert recipe_from_src["nexus_files"] == [str(raw_nxs)]
    assert recipe_from_src["instrument"] == "MANDI"
    assert recipe_from_src["wavelength_range"] == [2.0, 4.0]

    recipe_legacy = runner._extract_reduce_recipe_from_derived_h5(
        str(legacy_solve), ds_info={"name": "MANDI_13308_solve.h5"}
    )
    assert recipe_legacy is not None
    assert recipe_legacy["nexus_files"] == [str(raw_nxs)]

    vm = RainbowZonesViewModel(
        replay_h5=None,
        payload={"artifact_kind": "integrator", "panels": [], "peaks": []},
        pipeline_runner=runner,
    )
    vm.replay_h5 = str(int_h5)
    vm_lock = threading.RLock()
    hydrate_calls: list[dict] = []
    reloaded_images: list[str] = []
    monkeypatch.setattr(
        vm,
        "reload",
        lambda images_h5=None, frame_index=None: reloaded_images.append(str(images_h5)),
    )

    def _fake_hydrate_stage(
        stage_id,
        params,
        on_complete_h5=None,
        fallback_reason=None,
        existing_job_record=None,
        preserve_downstream=False,
    ):
        hydrate_calls.append(
            {
                "stage_id": stage_id,
                "params": dict(params),
                "preserve_downstream": preserve_downstream,
            }
        )
        # Ensure downstream files were NOT deleted before/during hydration!
        assert int_h5.is_file()
        assert (work_dir / "solve.h5").is_file()
        with h5py.File(cached_merged, "w") as hf_new:
            hf_new.attrs["instrument"] = "MANDI"
            hf_new.create_dataset("images", data=np.ones((1, 4, 4), dtype=np.float32))
        if callable(on_complete_h5):
            on_complete_h5(str(cached_merged))
        return {"job_id": "gpu-hydrate-1", "state": "running", "hydrate_only": True}

    monkeypatch.setattr(runner, "_submit_local_subhkl_stage", _fake_hydrate_stage)
    h_info = runner._maybe_trigger_hydrate_reduce(vm, vm_lock, str(int_h5))
    assert h_info is not None
    assert h_info["hydrate_only"] is True
    assert h_info["n_runs"] == 1
    assert len(hydrate_calls) == 1
    assert hydrate_calls[0]["stage_id"] == "reduce"
    assert hydrate_calls[0]["preserve_downstream"] is True
    # Completion hot-reloads images_h5 WITHOUT overwriting vm.replay_h5 (still integrated.h5!)
    assert vm.replay_h5 == str(int_h5)
    assert vm.payload_kwargs["images_h5"] == str(cached_merged)
    assert reloaded_images == [str(cached_merged)]

    # 5) Pre-existing solve.h5 containing transient `/.../subhkl_reduce_<ts>/MANDI_13308.reduce.h5` in `files`
    #    (without `source_files`) AND Trame `MainApp` (`/api/payload?dataset_id=...`) end-to-end hydration:
    import asyncio
    from collections.abc import Callable
    from types import SimpleNamespace

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    from subhkl.viz.nova_app.models.subhkl_model import SubhklGalaxyModel
    from subhkl.viz.nova_app.views.main_view import MainApp

    dt_str = h5py.string_dtype(encoding="utf-8")
    transient_reduce_solve = staging_dir / "subhkl_gx_deadbeef_solve.h5"
    with h5py.File(transient_reduce_solve, "w") as hf_tr:
        hf_tr.attrs["instrument"] = "MANDI"
        hf_tr.create_dataset(
            "files",
            data=np.array(
                ["/data/staging_test/282900/subhkl_reduce_99/MANDI_13308.reduce.h5"],
                dtype=object,
            ),
            dtype=dt_str,
        )
        hf_tr.create_dataset("instrument/wavelength", data=np.array([2.0, 4.0]))
        hf_tr.create_dataset(
            "sample/cell", data=np.array([55.2, 81.2, 88.7, 90.0, 96.3, 90.0])
        )
        hf_tr.create_dataset("sample/space_group", data=b"I 1 2 1")
        hf_tr.create_dataset("sample/U", data=np.eye(3))
        hf_tr.create_dataset(
            "sample/B", data=np.diag([1.0 / 55.2, 1.0 / 81.2, 1.0 / 88.7])
        )
        hf_tr.create_dataset("goniometer/R", data=np.eye(3)[None, ...])
        hf_tr.create_dataset("bank_ids", data=np.array([19], dtype=np.int32))
        hf_tr.create_dataset("file_offsets", data=np.array([0], dtype=np.int32))

    recipe_reduce_path = runner._extract_reduce_recipe_from_derived_h5(
        str(transient_reduce_solve), ds_info={"name": "solve.h5"}
    )
    assert recipe_reduce_path is not None
    assert recipe_reduce_path["nexus_files"] == [str(raw_nxs)]
    assert recipe_reduce_path["instrument"] == "MANDI"

    # Also verify _stamp_merged_metadata with local-stage job_record stamps `source_files` and clean `.nxs.h5` `files`
    stamped_m = tmp_path / "local_reduced_merged.h5"
    with h5py.File(stamped_m, "w") as hf_sm:
        hf_sm.create_dataset("images", data=np.ones((1, 16, 16), dtype=np.float32))
        hf_sm.create_dataset(
            "files",
            data=np.array(
                ["/data/staging_test/282900/subhkl_reduce_99/MANDI_13308.reduce.h5"],
                dtype=object,
            ),
            dtype=dt_str,
        )
    runner._stamp_merged_metadata(
        str(stamped_m),
        {
            "params": {
                "instrument": "MANDI",
                "cell": [55.2, 81.2, 88.7, 90.0, 96.3, 90.0],
                "space_group": "I 1 2 1",
            },
            "raw_nxs": [str(raw_nxs)],
            "instrument": "MANDI",
        },
    )
    with h5py.File(stamped_m, "r") as hf_sm_chk:
        assert list(hf_sm_chk["source_files"].asstr()[()]) == [str(raw_nxs)]
        assert list(hf_sm_chk["files"].asstr()[()]) == ["MANDI_13308.nxs.h5"]
        assert str(hf_sm_chk.attrs["instrument"]) == "MANDI"

    # Remove cached_merged so Trame MainApp experiences a fresh instance cache miss on dataset_id reload
    if cached_merged.exists():
        cached_merged.unlink()
    legacy_m_alias = work_dir / "merged.h5"
    if legacy_m_alias.exists():
        legacy_m_alias.unlink()

    monkeypatch.setattr(
        runner,
        "resolve_dataset_to_local_h5",
        lambda ds_id, fallback_dir=None: (
            str(transient_reduce_solve),
            {"id": ds_id, "name": "MANDI_13308_solve.h5", "hid": 14, "state": "ok"},
        ),
    )
    monkeypatch.setattr(runner, "_is_local_staging_mode", lambda: True)

    pending_callbacks: list[Callable[[str], None]] = []

    def _deferred_hydrate_stage(
        stage_id,
        params,
        on_complete_h5=None,
        fallback_reason=None,
        existing_job_record=None,
        preserve_downstream=False,
    ):
        if callable(on_complete_h5):
            pending_callbacks.append(on_complete_h5)
        return {
            "job_id": "gpu-trame-hydrate-1",
            "state": "running",
            "n_runs": len(params.get("nexus_files") or []),
            "hydrate_only": True,
            "work_dir": str(work_dir),
        }

    monkeypatch.setattr(runner, "_submit_local_subhkl_stage", _deferred_hydrate_stage)

    trame_app = object.__new__(MainApp)
    trame_app.pipeline_runner = runner
    trame_sub = SubhklGalaxyModel()
    trame_app.view_model = SimpleNamespace(model=SimpleNamespace(subhkl=trame_sub))
    aio_trame = web.Application()
    trame_app._bind_routes(SimpleNamespace(app=aio_trame))

    resp_ds = asyncio.run(
        aio_trame._handle(
            make_mocked_request(
                "GET", "/api/payload?dataset_id=deadbeef&frame=0", app=aio_trame
            )
        )
    )
    assert resp_ds.status == 200
    doc_ds = json.loads(resp_ds.body.decode("utf-8"))
    # Immediately returns solved geometry + hydration_job while staging solve.h5 into work_dir!
    assert trame_sub.data.peaks_file == str(work_dir / "solve.h5")
    assert doc_ds["intensity_source"] != "raw_h5"
    assert doc_ds.get("hydration_job", {}).get("job_id") == "gpu-trame-hydrate-1"
    assert len(pending_callbacks) == 1

    # Now background Reduce completes and writes MANDI_13308_merged.h5 with bank 19 detector image
    with h5py.File(cached_merged, "w") as hf_hyd:
        hf_hyd.attrs["instrument"] = "MANDI"
        hf_hyd.create_dataset(
            "images",
            data=np.arange(64 * 64, dtype=np.float32).reshape(1, 64, 64) + 5.0,
        )
        hf_hyd.create_dataset("bank_ids", data=np.array([19], dtype=np.int32))
        hf_hyd.create_dataset("file_offsets", data=np.array([0], dtype=np.int32))
        hf_hyd.create_dataset(
            "files",
            data=np.array(["MANDI_13308.nxs.h5"], dtype=object),
            dtype=dt_str,
        )
        hf_hyd.create_dataset("goniometer/R", data=np.eye(3)[None, ...])
        hf_hyd.create_dataset("instrument/wavelength", data=np.array([2.0, 4.0]))

    pending_callbacks[0](str(cached_merged))
    assert trame_sub.data.peaks_file == str(work_dir / "solve.h5")
    assert trame_sub.data.images_file == str(cached_merged)

    resp_after_hyd = asyncio.run(
        aio_trame._handle(
            make_mocked_request(
                "GET", "/api/payload?frame=0&preserve_params=1", app=aio_trame
            )
        )
    )
    assert resp_after_hyd.status == 200
    doc_after_hyd = json.loads(resp_after_hyd.body.decode("utf-8"))
    assert doc_after_hyd["intensity_source"] == "raw_h5"
    assert "hydration_job" not in doc_after_hyd
    b19_after = next(pn for pn in doc_after_hyd["panels"] if int(pn["bank"]) == 19)
    assert b19_after.get("texture_b64")


def test_multiframe_bank_ids_delta_theta_voronoi_and_live_js_recalc(tmp_path) -> None:
    """Verify multi-frame bank_ids DeltaTheta lookup, Voronoi neighbor-rung isolation, and live JS DeltaTheta."""
    import re
    import shutil

    import gemmi
    import h5py

    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    from subhkl.viz.rainbow_zones import (
        _estimate_image_spot_angular_errors_deg,
        _load_cached_frame_bank_images,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    # Non-contiguous bank_ids [19, 32] across 2 runs (file_offsets=[0, 2], NO "banks" dataset!)
    det32 = Detector(dict(beamlines["MANDI"]["32"]))
    img0_19 = np.full((256, 256), 2.0, dtype=np.float32)
    img0_32 = np.full((256, 256), 2.0, dtype=np.float32)
    img1_19 = np.full((256, 256), 2.0, dtype=np.float32)
    img1_32 = np.full((256, 256), 2.0, dtype=np.float32)

    # Plant a single sharp Bragg peak at (row=128.0, col=128.0) ONLY on frame 1, bank 32
    rr, cc = np.ogrid[:256, :256]
    spot = 180.0 * np.exp(-0.5 * (((rr - 128.0) ** 2 + (cc - 128.0) ** 2) / 1.1**2))
    img1_32 += spot.astype(np.float32)

    merged_h5 = tmp_path / "mandi_2frames.h5"
    dt_str = h5py.string_dtype(encoding="utf-8")
    with h5py.File(merged_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.create_dataset(
            "images",
            data=np.stack([img0_19, img0_32, img1_19, img1_32], axis=0),
        )
        hf.create_dataset("bank_ids", data=np.array([19, 32, 19, 32], dtype=np.int32))
        hf.create_dataset("file_offsets", data=np.array([0, 2], dtype=np.int32))
        hf.create_dataset(
            "files",
            data=np.array(["MANDI_11324.nxs.h5", "MANDI_11334.nxs.h5"], dtype=object),
            dtype=dt_str,
        )
        hf.create_dataset("goniometer/R", data=np.repeat(np.eye(3)[None], 4, axis=0))
        hf.create_dataset("instrument/wavelength", data=np.array([2.0, 4.5]))

    # 1) _load_cached_frame_bank_images resolves bank 32 on frame_slice=(2, 4) (would have returned {3, 4} before!)
    bank_imgs_f1 = _load_cached_frame_bank_images(str(merged_h5), (2, 4))
    assert set(bank_imgs_f1.keys()) == {19, 32}
    assert float(bank_imgs_f1[32][128, 128]) > 150.0

    # Construct simple cubic crystal whose (0, 0, 4) reflection at lambda ~ 3.1 A hits (128, 128) on bank 32
    pos_spot = det32.pixel_to_lab(128, 128)
    kf_spot = pos_spot / np.linalg.norm(pos_spot)
    q_dir = kf_spot - np.array([0.0, 0.0, 1.0])
    q_dir /= np.linalg.norm(q_dir)
    sin_th = float(np.linalg.norm(kf_spot - np.array([0.0, 0.0, 1.0])) / 2.0)
    d_target = 3.1 / (2.0 * sin_th)
    a_cubic = 4.0 * d_target
    cell_c = [a_cubic, a_cubic, a_cubic, 90.0, 90.0, 90.0]
    uc_c = gemmi.UnitCell(*cell_c)
    b_c = np.asarray(uc_c.frac.mat.tolist(), dtype=float).T
    z_axis = np.array([0.0, 0.0, 1.0])
    v_cr = np.cross(z_axis, q_dir)
    s_cr = np.linalg.norm(v_cr)
    c_cr = float(np.dot(z_axis, q_dir))
    vx = np.array(
        [[0.0, -v_cr[2], v_cr[1]], [v_cr[2], 0.0, -v_cr[0]], [-v_cr[1], v_cr[0], 0.0]]
    )
    u_c = np.eye(3) + vx + (vx @ vx) * ((1.0 - c_cr) / max(s_cr**2, 1e-12))

    # 2) Voronoi ownership: reflection #0 sits at (128.2, 127.9) (0.22 px from spot),
    #    while adjacent long-axis rung #1 sits 4.2 px away at (132.2, 128.0) (inside 7 px window!).
    #    Only reflection #0 may claim the spot; #1 must remain NaN instead of reporting ~0.5 deg error.
    ang_errs, has_slab = _estimate_image_spot_angular_errors_deg(
        images_h5=str(merged_h5),
        frame_slice=(2, 4),
        hkl_arr=np.array([[0, 0, 4], [1, 0, 4]], dtype=np.float64),
        bank_arr=np.array([32, 32], dtype=np.int32),
        pixel_r_arr=np.array([128.2, 132.2], dtype=np.float64),
        pixel_c_arr=np.array([127.9, 128.0], dtype=np.float64),
        ub_matrix=u_c @ b_c,
        R_run=np.eye(3),
        detectors={32: det32},
        ki_hat=np.array([0.0, 0.0, 1.0]),
    )
    assert has_slab is True
    assert ang_errs is not None
    assert np.isfinite(ang_errs[0]) and float(ang_errs[0]) < 0.08
    assert np.isnan(ang_errs[1])

    # 3) End-to-end build_rainbow_zone_payload on frame_index=1 populates zone angular_rms_deg

    solve_h5 = tmp_path / "solve_2frames.h5"
    with h5py.File(solve_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        for k_n, v_n in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell_c):
            hf.create_dataset(f"sample/{k_n}", data=float(v_n))
        hf.create_dataset("sample/space_group", data=b"P 2 3")
        hf.create_dataset("sample/U", data=u_c)
        hf.create_dataset("sample/B", data=b_c)
        hf.create_dataset("goniometer/R", data=np.repeat(np.eye(3)[None], 4, axis=0))
        hf.create_dataset("bank_ids", data=np.array([19, 32, 19, 32], dtype=np.int32))
        hf.create_dataset("file_offsets", data=np.array([0, 2], dtype=np.int32))
        hf.create_dataset(
            "files",
            data=np.array(["MANDI_11324.nxs.h5", "MANDI_11334.nxs.h5"], dtype=object),
            dtype=dt_str,
        )
        hf.create_dataset("instrument/wavelength", data=np.array([2.0, 4.5]))

    payload_f1 = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        images_h5=merged_h5,
        instrument="MANDI",
        wavelength_range=(2.0, 4.5),
        d_min=d_target * 0.92,
        frame_index=1,
    )
    finite_zone_rms = [
        z["angular_rms_deg"]
        for z in payload_f1["zones"]
        if z.get("angular_rms_deg") is not None
    ]
    assert len(finite_zone_rms) > 0
    assert min(finite_zone_rms) < 0.08

    html_doc = render_rainbow_html(payload_f1)
    assert '<option value="2">2</option>' in html_doc
    assert "estimateActiveFrameAngularErrorsJS" in html_doc
    assert "refreshZonesAngularStatsJS" in html_doc
    if shutil.which("node"):
        script = re.findall(r"<script>([\s\S]*?)</script>", html_doc)[-1]
        assert "predsByBank" in script


def test_stratified_spherical_subsample_and_canonical_bvh_pop_free(tmp_path):
    """Server solid-angle stratification preserves back-scattering banks, and client canonical dyadic BVHs are 100% pop-free across d_min and zoom."""
    import json
    import re
    import shutil
    import subprocess

    import h5py
    import numpy as np

    from subhkl.core.crystallography import cartesian_matrix_metric_tensor
    from subhkl.viz.rainbow_zones import (
        _stratified_spherical_subsample,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    # 1) Server-side _stratified_spherical_subsample: 5,000 forward high-d reflections in 4 narrow patches
    #    must NOT starve 200 back-scattering lower-d reflections at |az| > 150 deg when capping to 400.
    rng = np.random.default_rng(42)
    n_fwd, n_back = 5000, 200
    fwd_az = rng.uniform(-12.0, 12.0, size=n_fwd)
    fwd_el = rng.uniform(-8.0, 8.0, size=n_fwd)
    back_az = rng.choice([-168.0, 168.0], size=n_back) + rng.uniform(
        -10.0, 10.0, size=n_back
    )
    back_el = rng.uniform(-35.0, 35.0, size=n_back)
    az_deg = np.concatenate([fwd_az, back_az])
    el_deg = np.concatenate([fwd_el, back_el])
    az_r = np.deg2rad(az_deg)
    el_r = np.deg2rad(el_deg)
    kf = np.column_stack(
        [np.cos(el_r) * np.sin(az_r), np.sin(el_r), np.cos(el_r) * np.cos(az_r)]
    )
    scores = np.concatenate(
        [rng.uniform(4.0, 12.0, size=n_fwd), rng.uniform(1.0, 2.2, size=n_back)]
    )
    sel = _stratified_spherical_subsample(kf, 400, scores)
    assert len(sel) == 400
    assert int(np.sum(sel >= n_fwd)) >= 80

    if not shutil.which("node"):
        return

    # 2) Client-side 41-bank MANDI P 61 2 2: verify d_min monotonicity at 1x, back-scattering bank coverage,
    #    and 100% pop-free zoom from 1x -> 2x -> 4x (Tier-2 handoff) -> 8x -> 16x -> 26x!
    h5p = tmp_path / "mandi_p6122_bvh.h5"
    b_mat, _ = cartesian_matrix_metric_tensor(
        79.91, 79.91, 238.0, *np.deg2rad([90.0, 90.0, 120.0])
    )
    with h5py.File(h5p, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf["sample/space_group"] = "P 61 2 2"
        hf["sample/cell"] = np.array(
            [79.91, 79.91, 238.0, 90.0, 90.0, 120.0], dtype=float
        )
        hf["sample/U"] = np.eye(3, dtype=float)
        hf["sample/B"] = b_mat
        hf["instrument/wavelength"] = np.array([2.0, 4.0], dtype=float)

    payload = build_rainbow_zone_payload(
        replay_h5=str(h5p),
        instrument="MANDI",
        d_min=3.20,
        wavelength_range=(2.0, 4.0),
        include_panel_images=False,
    )
    html = render_rainbow_html(payload)
    js_code = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.DOTALL)[-1]
    idx_end = js_code.rfind("})();")

    harness = r"""
const stubCtx = new Proxy({
  measureText: (t) => ({ width: String(t).length * 6 }),
  createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
  createLinearGradient: () => ({ addColorStop: () => {} }),
  createRadialGradient: () => ({ addColorStop: () => {} }),
  getImageData: () => ({ data: new Uint8ClampedArray(4) }),
}, { get: (t, p) => (p in t ? t[p] : () => {}) });
function makeElem(w = 900, h = 520) {
  return {
    width: w, height: h, clientWidth: w, clientHeight: h, scrollWidth: w,
    options: [], children: [],
    style: { setProperty(){}, removeProperty(){} },
    classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    dataset: {}, value: "", checked: false, textContent: "", innerHTML: "",
    getContext: () => stubCtx,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: w, height: h }),
    addEventListener: () => {}, removeEventListener: () => {},
    querySelectorAll: () => [], querySelector: () => null,
    appendChild: () => {}, setAttribute: () => {}, getAttribute: () => null,
  };
}
const elemCache = new Map();
globalThis.window = { devicePixelRatio: 1, addEventListener: () => {}, innerWidth: 1280, innerHeight: 800 };
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.ResizeObserver = class { observe(){} disconnect(){} };
globalThis.OffscreenCanvas = undefined;
globalThis.document = {
  documentElement: makeElem(1280, 800),
  getElementById: (id) => {
    if (!elemCache.has(id)) {
      const isStereo = id.includes("stereo");
      elemCache.set(id, makeElem(isStereo ? 220 : 900, isStereo ? 220 : 520));
    }
    return elemCache.get(id);
  },
  createElement: () => makeElem(256, 256),
  querySelectorAll: () => [],
  querySelector: () => makeElem(),
  addEventListener: () => {},
  body: makeElem(1280, 800),
};
"""
    probe = r"""
state.peakSource = "predicted";
state.detZoom = 1.0; state.detPanX = 0; state.detPanY = 0;
state.wulffZoom = 1.0; state.wulffYawDeg = 0; state.wulffPitchDeg = 0;

const dSteps = [3.2, 2.2, 1.4, 1.0];
const detCounts = [];
const wulffCounts = [];
let backCount10 = 0;
for (const dm of dSteps) {
  state.dMin = dm;
  repredictPeaksDynamic({ userAdjusted: true });
  const dl = getDetectorLodState(900, 520);
  const wl = getWulffLodState(false);
  detCounts.push(dl.visiblePeaks.length);
  wulffCounts.push(wl.visiblePeaks.length);
  if (dm === 1.0) {
    backCount10 = dl.visiblePeaks.filter(p => Math.abs(p.det_x) > 130.0).length;
  }
}

state.projection = "spherical";
const bp = PEAKS.filter(p => p.bank === 20);
let sSin = 0, sCos = 0, sEl = 0;
for (const p of bp) {
  const r = p.det_x * Math.PI / 180.0;
  sSin += Math.sin(r); sCos += Math.cos(r); sEl += p.det_el;
}
const cAz = Math.atan2(sSin, sCos) * 180.0 / Math.PI;
const meanEl = sEl / bp.length;
let prevCore = null;
let poppedTotal = 0;
const bankCounts = [];
for (const z of [1.0, 2.0, 4.0, 8.0, 16.0, 26.0]) {
  state.detZoom = z;
  setDetectorCenterAzimuthDeg(cAz);
  const bAct = getActiveBounds();
  const bxAct = getDetectorPlotBox(900, 520);
  const y0Ref = bxAct.originY + (1.0 - (meanEl - bAct.minY) / (bAct.maxY - bAct.minY)) * bxAct.plotH;
  state.detPanY = -(y0Ref - 260.0) * z;
  const lod = getDetectorLodState(900, 520);
  const onBank = lod.visiblePeaks.filter(p => p.bank === 20);
  bankCounts.push(onBank.length);
  const core = new Set(
    onBank
      .filter(p => Math.abs(p.det_x - cAz) < 2.0 && Math.abs(p.det_el - meanEl) < 2.0)
      .map(p => `${p.bank}|${p.hkl.join(",")}`)
  );
  if (prevCore) {
    for (const k of prevCore) if (!core.has(k)) poppedTotal++;
  }
  prevCore = core;
}
console.log(JSON.stringify({ detCounts, wulffCounts, backCount10, poppedTotal, bankCounts, rawTotal: _dynEwaldRawTotal }));
})();
"""
    js_path = tmp_path / "mandi_bvh_test.js"
    js_path.write_text(
        harness + "\n" + js_code[:idx_end] + "\n" + probe, encoding="utf-8"
    )
    res = subprocess.run(
        ["node", str(js_path)], check=True, capture_output=True, text=True
    )
    out = json.loads(res.stdout.strip())
    assert out["rawTotal"] > 150000
    assert (
        out["detCounts"][0]
        < out["detCounts"][1]
        < out["detCounts"][2]
        <= out["detCounts"][3]
    )
    assert out["detCounts"][3] <= 1400
    assert (
        out["wulffCounts"][0]
        < out["wulffCounts"][1]
        < out["wulffCounts"][2]
        <= out["wulffCounts"][3]
    )
    assert out["backCount10"] >= 60
    assert out["poppedTotal"] == 0
    assert out["bankCounts"][0] < out["bankCounts"][2] < out["bankCounts"][3]


def test_predictor_and_rainbow_zones_honor_edge_and_pixel_masks(tmp_path) -> None:
    """Verify peak-predictor and Rainbow Zones (server + client JS pool/Ewald raycasters) honor edge borders and per-pixel masks."""
    import json
    import re
    import shutil
    import subprocess

    from scipy.spatial.transform import Rotation

    from subhkl.commands import run_peak_predictor
    from subhkl.config import beamlines
    from subhkl.core.crystallography import (
        cartesian_matrix_metric_tensor,
        generate_reflections,
    )
    from subhkl.instrument.detector import Detector
    from subhkl.viz.rainbow_zones import (
        _raycast_hkl_on_detectors,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    cell = [68.0, 74.0, 92.0, 90.0, 90.0, 90.0]
    B_raw, _ = cartesian_matrix_metric_tensor(*cell[:3], *np.deg2rad(cell[3:]))
    B_mat = np.asarray(B_raw, dtype=float)
    U_mat = Rotation.from_euler("zyx", [19.0, -31.0, 44.0], degrees=True).as_matrix()
    R_mat = np.eye(3, dtype=float)
    active_bks = [1, 20]
    dets = {bk: Detector(dict(beamlines["MANDI"][str(bk)])) for bk in active_bks}
    H, W = int(dets[1].n), int(dets[1].m)
    bw = int(0.06 * min(H, W))  # 15 px on 256x256 MANDI panel

    hkls = np.stack(
        generate_reflections(*cell, space_group="P 21 21 21", d_min=1.8), axis=1
    ).astype(int)
    raw_cast = _raycast_hkl_on_detectors(
        hkls,
        U_mat,
        B_mat,
        R_mat,
        dets,
        (2.0, 4.0),
        active_bids=set(active_bks),
        margin_frac=0.0,
    )
    assert len(raw_cast["peaks_xyz"]) > 80

    # Confirm there are genuine edge peaks (< bw or >= 255 - bw) AND interior peaks we will mask!
    in_edge = (
        (raw_cast["pixel_r"] < bw)
        | (raw_cast["pixel_r"] >= (H - 1 - bw))
        | (raw_cast["pixel_c"] < bw)
        | (raw_cast["pixel_c"] >= (W - 1 - bw))
    )
    assert int(np.sum(in_edge)) >= 5
    interior_idx = np.flatnonzero(~in_edge)
    assert len(interior_idx) >= 15

    # Mask a 40x40 central patch on bank 1 covering known interior reflections
    b1_int = [
        idx
        for idx in interior_idx
        if int(raw_cast["bank"][idx]) == 1
        and 40 <= raw_cast["pixel_r"][idx] <= 210
        and 40 <= raw_cast["pixel_c"][idx] <= 210
    ]
    assert len(b1_int) >= 1
    target_i = b1_int[0]
    tr = round(float(raw_cast["pixel_r"][target_i]))
    tc = round(float(raw_cast["pixel_c"][target_i]))
    r_lo, r_hi = max(bw + 1, tr - 18), min(H - 2 - bw, tr + 19)
    c_lo, c_hi = max(bw + 1, tc - 18), min(W - 2 - bw, tc + 19)

    cls_masks = np.zeros((2, H, W), dtype=np.uint8)
    cls_masks[0, r_lo:r_hi, c_lo:c_hi] = 1  # dark/shadowed patch on bank 1
    cls_masks[0, max(bw + 1, r_hi) : min(H - 2 - bw, r_hi + 6), c_lo:c_hi] = (
        2  # static-bright strip
    )

    merged_h5 = tmp_path / "mandi_mask_merged.h5"
    with h5py.File(merged_h5, "w") as mf:
        mf.attrs["instrument"] = "MANDI"
        mf.create_dataset("images", data=np.ones((2, H, W), dtype=np.float32) * 5.0)
        mf.create_dataset("bank_ids", data=np.array(active_bks, dtype=np.int32))
        mf.create_dataset("file_offsets", data=np.array([0], dtype=np.int32))
        mf.create_dataset(
            "files", data=np.array(["MANDI_mask_test.nxs.h5"], dtype="S32")
        )
        gon_m = mf.create_group("goniometer")
        gon_m.create_dataset("R", data=R_mat[None, :, :])
        gon_m.create_dataset("angles", data=np.zeros((2, 3), dtype=np.float64))
        gon_m.create_dataset(
            "axes",
            data=np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.float64),
        )
        mf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([2.0, 4.0], dtype=np.float64)
        )

    solve_h5 = tmp_path / "mandi_mask_solve.h5"
    with h5py.File(solve_h5, "w") as sf:
        sf.attrs["instrument"] = "MANDI"
        sf.attrs["source"] = str(merged_h5)
        smp = sf.create_group("sample")
        smp.create_dataset("space_group", data=b"P 21 21 21")
        smp.create_dataset("cell", data=np.array(cell, dtype=float))
        for k_c, v_c in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            smp.create_dataset(k_c, data=float(v_c))
        smp.create_dataset("U", data=U_mat)
        smp.create_dataset("B", data=B_mat)
        sf.create_group("goniometer").create_dataset("R", data=R_mat[None, :, :])
        sf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([2.0, 4.0], dtype=np.float64)
        )
        det_grp = sf.create_group("detector")
        det_grp.create_dataset("pixel_mask", data=(cls_masks == 0).astype(np.uint8))
        det_grp.create_dataset("pixel_class", data=cls_masks)
        det_grp.create_dataset(
            "mask_bank_ids", data=np.array(active_bks, dtype=np.int32)
        )

    # Also create an external static-mask HDF5 blocking an extra patch on bank 20
    static_mask_h5 = tmp_path / "static_mask.h5"
    valid_ext = np.ones((2, H, W), dtype=np.uint8)
    valid_ext[1, 60:140, 60:140] = 0
    with h5py.File(static_mask_h5, "w") as smf:
        smf.attrs["kind"] = "static-mask"
        smf.create_dataset("images", data=valid_ext)
        smf.create_dataset("bank_ids", data=np.array(active_bks, dtype=np.int32))

    # 1. Verify run_peak_predictor honors BOTH rel_border_width AND detector/pixel_mask + static_mask_file
    pred_no_border_h5 = tmp_path / "pred_no_border.h5"
    pred_masked_h5 = tmp_path / "pred_masked.h5"
    run_peak_predictor(
        str(merged_h5),
        "MANDI",
        str(solve_h5),
        str(pred_no_border_h5),
        d_min=1.8,
        wavel_min=2.0,
        wavel_max=4.0,
        space_group="P 21 21 21",
        rel_border_width=0.0,
    )
    run_peak_predictor(
        str(merged_h5),
        "MANDI",
        str(solve_h5),
        str(pred_masked_h5),
        d_min=1.8,
        wavel_min=2.0,
        wavel_max=4.0,
        space_group="P 21 21 21",
        rel_border_width=0.06,
        static_mask_file=str(static_mask_h5),
    )
    with (
        h5py.File(pred_no_border_h5, "r") as f0,
        h5py.File(pred_masked_h5, "r") as f1,
    ):
        n0 = sum(len(f0["banks"][k]["i"]) for k in f0["banks"])
        n1 = sum(len(f1["banks"][k]["i"]) for k in f1["banks"])
        assert n1 < n0
        for img_k, bk_id in enumerate(active_bks):
            if str(img_k) not in f1["banks"]:
                continue
            ri = np.asarray(f1["banks"][str(img_k)]["i"][()], dtype=float)
            ci = np.asarray(f1["banks"][str(img_k)]["j"][()], dtype=float)
            assert np.all((ri >= bw) & (ri < H - 1 - bw))
            assert np.all((ci >= bw) & (ci < W - 1 - bw))
            r_int = np.round(ri).astype(int)
            c_int = np.round(ci).astype(int)
            assert np.all(cls_masks[img_k, r_int, c_int] == 0)
            if bk_id == 20:
                assert np.all(valid_ext[1, r_int, c_int] == 1)

    # 2. Verify Rainbow Zones server payload + client JS live re-prediction honor edge & per-pixel masks
    payload = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        images_h5=merged_h5,
        instrument="MANDI",
        d_min=1.8,
        wavelength_range=(2.0, 4.0),
        include_panel_images=True,
    )
    assert payload["rel_border_width"] == 0.06
    assert isinstance(payload.get("pixel_mask_bits"), dict)
    assert "1" in payload["pixel_mask_bits"]
    for pk in payload["predicted_peaks"]:
        r_pk, c_pk = float(pk["row"]), float(pk["col"])
        assert bw <= r_pk < (H - 1 - bw)
        assert bw <= c_pk < (W - 1 - bw)
        if int(pk["bank"]) == 1:
            assert cls_masks[0, round(r_pk), round(c_pk)] == 0

    if shutil.which("node"):
        html_doc = render_rainbow_html(payload)
        scripts = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
        js_src = scripts[-1]
        iife_end = js_src.rfind("})();")
        assert iife_end > 0
        js_harness = """
const elems = new Map();
function makeEl(id = "") {
  const listeners = {};
  return {
    id, tagName: "DIV", value: "", min: "0.05", max: "0.95", step: "0.01",
    checked: false, disabled: false, textContent: "", innerHTML: "", title: "",
    width: 800, height: 520, clientWidth: 800, clientHeight: 520,
    scrollWidth: 800, scrollHeight: 520, scrollTop: 0, offsetWidth: 800, offsetHeight: 36,
    files: [], options: [], children: [], parentElement: null, dataset: {},
    style: new Proxy({}, {
      get(t, k) {
        if (k === "setProperty") return (p, v) => { t[p] = v; };
        if (k === "getPropertyValue") return (p) => t[p] || "";
        return t[k] ?? "";
      },
      set(t, k, v) { t[k] = v; return true; }
    }),
    classList: { add() {}, remove() {}, toggle() { return false; }, contains() { return false; } },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    removeEventListener() {},
    dispatchEvent(evObj) { (listeners[evObj.type] || []).forEach(fn => fn(evObj)); },
    appendChild(c) { return c; }, insertBefore(c) { return c; }, replaceChildren() {},
    setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
    getBoundingClientRect() { return { left: 0, top: 0, right: 800, bottom: 520, width: 800, height: 520 }; },
    getContext() {
      return {
        fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "left",
        textBaseline: "alphabetic", globalAlpha: 1, imageSmoothingEnabled: true,
        setTransform() {}, resetTransform() {}, scale() {}, translate() {}, rotate() {},
        transform() {}, clearRect() {}, fillRect() {}, strokeRect() {},
        beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, arc() {}, ellipse() {},
        quadraticCurveTo() {}, bezierCurveTo() {}, rect() {}, roundRect() {},
        stroke() {}, fill() {}, clip() {}, save() {}, restore() {}, setLineDash() {},
        fillText() {}, strokeText() {}, drawImage() {},
        measureText(t) { return { width: String(t).length * 6 }; },
        createLinearGradient() { return { addColorStop() {} }; },
        createRadialGradient() { return { addColorStop() {} }; },
        getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
        putImageData() {}, createImageData(w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
      };
    },
    querySelector() { return makeEl(); }, querySelectorAll() { return []; }, closest() { return null; },
  };
}
function getEl(id) { if (!elems.has(id)) elems.set(id, makeEl(id)); return elems.get(id); }
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.parent = globalThis;
globalThis.top = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.devicePixelRatio = 1;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.requestAnimationFrame = () => 1;
globalThis.cancelAnimationFrame = () => {};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.Image = class { set src(_v) {} };
globalThis.document = {
  body: makeEl("body"), documentElement: makeEl("html"),
  getElementById: (id) => getEl(id),
  createElement: (tag) => makeEl(tag),
  querySelector: () => makeEl(), querySelectorAll: () => [],
  addEventListener() {}, removeEventListener() {},
};
getEl("inp-border").value = "0.06";
"""
        js_probe = """
// 1. Trigger dynamic pool reprediction at d_min = 1.80 with border = 0.06
state.dMin = 1.80;
repredictPeaksDynamic({ preserveSource: true });
const poolPeaks06 = PEAKS.map(p => ({ bank: p.bank, hkl: p.hkl, row: p.row, col: p.col }));

// 2. Increase live #inp-border to 0.15 -> strictly fewer peaks, all within [38, 217)
const bInput = getEl("inp-border");
bInput.value = "0.15";
bInput.dispatchEvent({ type: "change", target: bInput });
const poolPeaks15 = PEAKS.map(p => ({ bank: p.bank, hkl: p.hkl, row: p.row, col: p.col }));

// 3. Trigger direct Ewald-shell rod mode at d_min = 0.60 with border = 0.06 -> still honors border & pixel_mask_bits!
bInput.value = "0.06";
state.dMin = 0.60;
repredictPeaksDynamic({ preserveSource: true });
const ewaldPeaks06 = PEAKS.map(p => ({ bank: p.bank, row: p.row, col: p.col }));

console.log(JSON.stringify({ poolPeaks06, poolPeaks15, ewaldCount: ewaldPeaks06.length, ewaldPeaks06 }));
})();
"""
        js_path = tmp_path / "mask_parity_test.js"
        js_path.write_text(
            js_harness + "\n" + js_src[:iife_end] + "\n" + js_probe, encoding="utf-8"
        )
        proc = subprocess.run(
            ["node", str(js_path)], check=True, capture_output=True, text=True
        )
        js_out = json.loads(proc.stdout.strip())
        py_keys_06 = {
            (int(p["bank"]), int(p["hkl"][0]), int(p["hkl"][1]), int(p["hkl"][2]))
            for p in payload["predicted_peaks"]
        }
        js_keys_06 = {
            (int(p["bank"]), int(p["hkl"][0]), int(p["hkl"][1]), int(p["hkl"][2]))
            for p in js_out["poolPeaks06"]
        }
        assert js_keys_06 == py_keys_06
        assert len(js_out["poolPeaks15"]) < len(js_out["poolPeaks06"])
        bw15 = int(0.15 * min(H, W))
        for p15 in js_out["poolPeaks15"]:
            assert bw15 <= p15["row"] < (H - 1 - bw15)
            assert bw15 <= p15["col"] < (W - 1 - bw15)
        assert js_out["ewaldCount"] > 500
        for pe in js_out["ewaldPeaks06"]:
            assert bw <= pe["row"] < (H - 1 - bw)
            assert bw <= pe["col"] < (W - 1 - bw)
            if int(pe["bank"]) == 1:
                assert cls_masks[0, round(pe["row"]), round(pe["col"])] == 0


def test_tof_gated_3d_spot_angular_errors_and_live_ui(tmp_path) -> None:
    """Verify 3D ToF-gated peak placement, full 3D q-space angle, L1+L2 sensitivity, 3D Voronoi ownership, and live UI sync."""
    import json
    import re
    import shutil
    import subprocess

    import gemmi
    import h5py

    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    from subhkl.io.events import ANGSTROM_M_PER_US, write_events
    from subhkl.viz.rainbow_zones import (
        _estimate_image_spot_angular_errors_deg,
        _estimate_tof_gated_spot_angular_errors_deg,
        build_rainbow_zone_payload,
        compute_live_tof_delta_theta_payload,
        render_rainbow_html,
    )

    det32 = Detector(dict(beamlines["MANDI"]["32"]))
    H, W = int(det32.n), int(det32.m)
    l1_true = 30.0
    ki_hat = np.array([0.0, 0.0, 1.0], dtype=float)

    # Geometry at center pixel (128, 128) on MANDI bank 32
    xyz_c = np.asarray(det32.pixel_to_lab(128.0, 128.0), dtype=float).ravel()
    l2_c = float(np.linalg.norm(xyz_c))
    kf_c = xyz_c / l2_c
    q_dir = kf_c - ki_hat
    q_unit_norm = float(np.linalg.norm(q_dir))  # 2 * sin(theta)
    q_hat = q_dir / q_unit_norm

    # Align cubic c* along q_hat so (0, 0, 2) and (0, 0, 3) land on the EXACT SAME pixel (128, 128)
    # at two distinct wavelengths: lam_2 = 3.60 A, lam_3 = 2.40 A!
    lam_2 = 3.60
    d_002 = lam_2 / q_unit_norm
    a_cell = 2.0 * d_002
    cell_c = [a_cell, a_cell, a_cell, 90.0, 90.0, 90.0]
    uc = gemmi.UnitCell(*cell_c)
    b_mat = np.asarray(uc.frac.mat.tolist(), dtype=float).T

    z_ax = np.array([0.0, 0.0, 1.0])
    v_cr = np.cross(z_ax, q_hat)
    s_cr = float(np.linalg.norm(v_cr))
    c_cr = float(np.dot(z_ax, q_hat))
    vx = np.array(
        [[0.0, -v_cr[2], v_cr[1]], [v_cr[2], 0.0, -v_cr[0]], [-v_cr[1], v_cr[0], 0.0]]
    )
    u_mat = np.eye(3) + vx + (vx @ vx) * ((1.0 - c_cr) / max(s_cr**2, 1e-12))
    ub_mat = u_mat @ b_mat

    lam_3 = (2.0 / 3.0) * lam_2  # 2.40 A
    t_pred_2 = lam_2 * (l1_true + l2_c) / ANGSTROM_M_PER_US
    t_pred_3 = lam_3 * (l1_true + l2_c) / ANGSTROM_M_PER_US

    # Synthesize sparse CSR event table on bank 32 (plus an empty/background bank 19):
    # - Uniform broadband Poisson-like background across [14000, 38000] us
    # - Spot A for (0,0,2): exact pixel (128, 128), +0.6% radial ToF shift (moderator tail!) -> directional arccos(q_hat . G_hat) ~ 0.0 deg!
    # - Spot B for (0,0,3): shifted +4 px spatially to (132, 128) at t_pred_3 -> directional angle ~ 0.20 deg!
    #   In 2D broadband images, Spot A and Spot B overlap in the same patch around (128, 128) and cannot be resolved independently.
    rng = np.random.default_rng(12345)
    tof_lists_32: list[list[float]] = []
    for _p_idx in range(H * W):
        bg_ev = rng.uniform(14000.0, 38000.0, size=2).astype(np.float32).tolist()
        tof_lists_32.append(bg_ev)

    rel_shift_2 = 0.006  # +0.6% radial ToF shift within gate, zero pixel shift
    row_shift_3 = 4  # +4 px spatial offset on overlapping harmonic (0, 0, 3)
    xyz_b = np.asarray(
        det32.pixel_to_lab(128.0 + row_shift_3, 128.0), dtype=float
    ).ravel()
    kf_b = xyz_b / float(np.linalg.norm(xyz_b))
    q_hat_b = (kf_b - ki_hat) / float(np.linalg.norm(kf_b - ki_hat))
    expected_dir_deg_3 = float(
        np.rad2deg(np.arccos(np.clip(np.dot(q_hat_b, q_hat), -1.0, 1.0)))
    )

    for dr in range(-2, 3):
        for dc in range(-2, 3):
            w_sp = np.exp(-0.5 * (dr * dr + dc * dc) / (0.85**2))
            n_ev2 = round(65.0 * w_sp)
            n_ev3 = round(85.0 * w_sp)
            if n_ev2 > 0:
                p_flat_a = (128 + dr) * W + (128 + dc)
                ev2 = rng.normal(
                    t_pred_2 * (1.0 + rel_shift_2),
                    t_pred_2 * 0.0006,
                    size=n_ev2,
                ).astype(np.float32)
                tof_lists_32[p_flat_a].extend(ev2.tolist())
            if n_ev3 > 0:
                p_flat_b = (128 + row_shift_3 + dr) * W + (128 + dc)
                ev3 = rng.normal(
                    t_pred_3,
                    t_pred_3 * 0.0006,
                    size=n_ev3,
                ).astype(np.float32)
                tof_lists_32[p_flat_b].extend(ev3.tolist())

    img_19 = np.zeros((H, W), dtype=np.float32)
    tof_19 = np.zeros(0, dtype=np.float32)
    img_32 = np.zeros(H * W, dtype=np.float32)
    flat_32: list[float] = []
    for idx_p, ev_l in enumerate(tof_lists_32):
        if ev_l:
            ev_s = sorted(ev_l)
            flat_32.extend(ev_s)
            img_32[idx_p] = float(len(ev_s))
    tof_32 = np.asarray(flat_32, dtype=np.float32)
    imgs_2 = np.stack([img_19, img_32.reshape(H, W)], axis=0)

    merged_h5 = tmp_path / "mandi_tof_3d_merged.h5"
    with h5py.File(merged_h5, "w") as mf:
        mf.attrs["instrument"] = "MANDI"
        mf.create_dataset("images", data=imgs_2)
        mf.create_dataset("bank_ids", data=np.array([19, 32], dtype=np.int32))
        mf.create_dataset("file_offsets", data=np.array([0], dtype=np.int32))
        mf.create_dataset("files", data=np.array(["MANDI_tof3d.nxs.h5"], dtype="S32"))
        mf.create_group("goniometer").create_dataset(
            "R", data=np.repeat(np.eye(3)[None], 2, axis=0)
        )
        inst_g = mf.create_group("instrument")
        inst_g.create_dataset("wavelength", data=np.array([1.8, 4.2], dtype=float))
        write_events(mf, [tof_19, tof_32], l1=l1_true)

    # 1. Compare 2D image vs 3D ToF on the two same-ray harmonics (0,0,2) and (0,0,3):
    hkl_pair = np.array([[0, 0, 2], [0, 0, 3]], dtype=np.int32)
    bank_pair = np.array([32, 32], dtype=np.int32)
    pr_pair = np.array([128.0, 128.0], dtype=np.float64)
    pc_pair = np.array([128.0, 128.0], dtype=np.float64)
    lam_pair = np.array([lam_2, lam_3], dtype=np.float64)

    err_2d, ok_2d = _estimate_image_spot_angular_errors_deg(
        images_h5=str(merged_h5),
        frame_slice=(0, 2),
        hkl_arr=hkl_pair,
        bank_arr=bank_pair,
        pixel_r_arr=pr_pair,
        pixel_c_arr=pc_pair,
        ub_matrix=ub_mat,
        R_run=np.eye(3),
        detectors={32: det32},
        ki_hat=ki_hat,
    )
    assert ok_2d is True
    assert err_2d is not None
    # In 2D broadband image, Spot A (128,128) and Spot B (132,128) blend into one composite centroid (> 0.08 deg error on (0,0,2)!)
    assert float(err_2d[0]) > 0.08

    solve_h5 = tmp_path / "mandi_tof_3d_solve.h5"
    with h5py.File(solve_h5, "w") as sf:
        sf.attrs["instrument"] = "MANDI"
        sf.attrs["source"] = str(merged_h5)
        smp = sf.create_group("sample")
        smp.create_dataset("space_group", data=b"P 1")
        smp.create_dataset("cell", data=np.array(cell_c, dtype=float))
        for k_n, v_n in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell_c):
            smp.create_dataset(k_n, data=float(v_n))
        smp.create_dataset("U", data=u_mat)
        smp.create_dataset("B", data=b_mat)
        sf.create_group("goniometer").create_dataset("R", data=np.eye(3)[None])
        ig = sf.create_group("instrument")
        ig.create_dataset("wavelength", data=np.array([1.8, 4.2], dtype=float))
        ig.create_dataset("L1", data=float(l1_true))

    err_3d, ok_3d, gate_meta = _estimate_tof_gated_spot_angular_errors_deg(
        images_h5=str(merged_h5),
        replay_h5=str(solve_h5),
        frame_slice=(0, 2),
        hkl_arr=hkl_pair,
        bank_arr=bank_pair,
        pixel_r_arr=pr_pair,
        pixel_c_arr=pc_pair,
        lam_arr=lam_pair,
        ub_matrix=ub_mat,
        R_run=np.eye(3),
        detectors={32: det32},
        ki_hat=ki_hat,
        instrument="MANDI",
        tof_l1=l1_true,
        wavelength_range=(1.8, 4.2),
    )
    assert ok_3d is True
    assert err_3d is not None
    assert gate_meta is not None
    # Both overlapping harmonics are resolved independently in 3D ToF!
    assert np.all(np.isfinite(err_3d))
    # (0, 0, 2) has +0.6% pure radial ToF shift and 0 spatial shift -> directional arccos(q_hat . G_hat) ~ 0.0 deg!
    assert float(err_3d[0]) < 0.04
    # (0, 0, 3) is captured 4 px off-center inside the full circular patch with exact directional unit-vector angle!
    assert abs(float(err_3d[1]) - expected_dir_deg_3) < 0.04

    # 2. Verify L1 + L2 flight-path sensitivity in 3D ToF gating: shifting L1 by +2.5 m (~8%) moves t_pred outside Spot A's gate!
    err_3d_bad_l1, ok_bad_l1, _ = _estimate_tof_gated_spot_angular_errors_deg(
        images_h5=str(merged_h5),
        replay_h5=str(solve_h5),
        frame_slice=(0, 2),
        hkl_arr=hkl_pair,
        bank_arr=bank_pair,
        pixel_r_arr=pr_pair,
        pixel_c_arr=pc_pair,
        lam_arr=lam_pair,
        ub_matrix=ub_mat,
        R_run=np.eye(3),
        detectors={32: det32},
        ki_hat=ki_hat,
        instrument="MANDI",
        tof_l1=l1_true + 2.50,
        wavelength_range=(1.8, 4.2),
    )
    assert ok_bad_l1 is True and err_3d_bad_l1 is not None
    assert np.isnan(err_3d_bad_l1[0]) and np.isnan(err_3d_bad_l1[1])

    # 3. Verify 3D Voronoi ownership & d_min convergence:
    #    Place a higher-index reflection [1, 0, 3] right on Spot B's 3D q_obs = q_hat_b * (q_unit_norm / lam_3)!
    #    When [1, 0, 3] enters the predicted shell (fine d_min), (0, 0, 3) loses its 4-px off-center false match (NaN)
    #    and [1, 0, 3] claims Spot B with ~0.0 deg directional error!
    q_obs_b_cryst = q_hat_b * (q_unit_norm / lam_3)
    ub_crowd = ub_mat.copy()
    ub_crowd[:, 0] = q_obs_b_cryst - 3.0 * ub_mat[:, 2]
    hkl_crowd = np.array([[0, 0, 2], [0, 0, 3], [1, 0, 3]], dtype=np.int32)
    err_crowd, ok_crowd, _ = _estimate_tof_gated_spot_angular_errors_deg(
        images_h5=str(merged_h5),
        replay_h5=str(solve_h5),
        frame_slice=(0, 2),
        hkl_arr=hkl_crowd,
        bank_arr=np.array([32, 32, 32], dtype=np.int32),
        pixel_r_arr=np.array([128.0, 128.0, 128.0 + row_shift_3], dtype=np.float64),
        pixel_c_arr=np.array([128.0, 128.0, 128.0], dtype=np.float64),
        lam_arr=np.array([lam_2, lam_3, lam_3], dtype=np.float64),
        ub_matrix=ub_crowd,
        R_run=np.eye(3),
        detectors={32: det32},
        ki_hat=ki_hat,
        instrument="MANDI",
        tof_l1=l1_true,
        wavelength_range=(1.8, 4.2),
    )
    assert ok_crowd is True and err_crowd is not None
    assert np.isfinite(err_crowd[0]) and float(err_crowd[0]) < 0.04
    assert np.isnan(
        err_crowd[1]
    )  # false 4-px match evicted in 3D q-space Voronoi by [1, 0, 3]!
    assert np.isfinite(err_crowd[2]) and float(err_crowd[2]) < 0.04

    # 4. End-to-end payload + live API helper + client JS sync & 2D/3D guard
    d_cut_coarse = d_002 * 0.95  # admits (0,0,2) only
    d_cut_fine = (a_cell / 3.0) * 0.95  # admits both (0,0,2) and (0,0,3)
    payload_tof = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        images_h5=merged_h5,
        instrument="MANDI",
        wavelength_range=(1.8, 4.2),
        d_min=d_cut_coarse,
        include_panel_images=False,
    )
    assert payload_tof["has_tof_events"] is True
    assert payload_tof["angular_error_mode"] == "tof_3d"
    assert abs(float(payload_tof["tof_l1"]) - l1_true) < 1e-6
    assert isinstance(payload_tof["tof_gate"], dict)

    live_coarse = compute_live_tof_delta_theta_payload(
        images_h5=str(merged_h5),
        replay_h5=str(solve_h5),
        frame_index=0,
        d_min=d_cut_coarse,
        wavelength_range=(1.8, 4.2),
        ub_matrix=payload_tof["ub_matrix"],
        R_run=payload_tof["goniometer_R"],
        sample_offset_lab=payload_tof["sample_offset_lab"],
        ki=payload_tof["ki"],
        instrument="MANDI",
        space_group=payload_tof["space_group"],
        cell=payload_tof["cell"],
        detector_configs=payload_tof["detector_configs"],
        tof_l1=l1_true,
    )
    live_fine = compute_live_tof_delta_theta_payload(
        images_h5=str(merged_h5),
        replay_h5=str(solve_h5),
        frame_index=0,
        d_min=d_cut_fine,
        wavelength_range=(1.8, 4.2),
        ub_matrix=payload_tof["ub_matrix"],
        R_run=payload_tof["goniometer_R"],
        sample_offset_lab=payload_tof["sample_offset_lab"],
        ki=payload_tof["ki"],
        instrument="MANDI",
        space_group=payload_tof["space_group"],
        cell=payload_tof["cell"],
        detector_configs=payload_tof["detector_configs"],
        tof_l1=l1_true,
    )
    assert live_coarse["ok"] is True and live_fine["ok"] is True
    assert "32|0,0,2" in live_coarse["errors"]
    assert "32|0,0,3" not in live_coarse["errors"]
    assert "32|0,0,2" in live_fine["errors"] and "32|0,0,3" in live_fine["errors"]

    # Shifting detector bank 32 laterally by +2.4 mm (~4 px) via live detector_configs aligns (0,0,3) onto Spot B and shifts (0,0,2) by ~0.20 deg!
    shifted_cfgs = json.loads(json.dumps(payload_tof["detector_configs"]))
    v_hat_32 = np.asarray(shifted_cfgs["32"]["vhat"], dtype=float)
    px_h = float(shifted_cfgs["32"]["height"]) / float(shifted_cfgs["32"]["n"])
    shifted_cfgs["32"]["center"] = (
        np.asarray(shifted_cfgs["32"]["center"], dtype=float)
        + row_shift_3 * px_h * v_hat_32
    ).tolist()
    live_shifted_det = compute_live_tof_delta_theta_payload(
        images_h5=str(merged_h5),
        replay_h5=str(solve_h5),
        frame_index=0,
        d_min=d_cut_fine,
        wavelength_range=(1.8, 4.2),
        ub_matrix=payload_tof["ub_matrix"],
        R_run=payload_tof["goniometer_R"],
        sample_offset_lab=payload_tof["sample_offset_lab"],
        ki=payload_tof["ki"],
        instrument="MANDI",
        space_group=payload_tof["space_group"],
        cell=payload_tof["cell"],
        detector_configs=shifted_cfgs,
        tof_l1=l1_true,
    )
    assert (
        live_shifted_det["errors"]["32|0,0,2"]
        > live_coarse["errors"]["32|0,0,2"] + 0.12
    )

    if shutil.which("node"):
        html_doc = render_rainbow_html(payload_tof)
        scripts = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
        js_src = scripts[-1]
        iife_end = js_src.rfind("})();")
        assert iife_end > 0
        live_fine_json = json.dumps(live_fine)
        live_coarse_json = json.dumps(live_coarse)
        js_harness = """
const elems = new Map();
function makeEl(id = "") {
  const listeners = {};
  return {
    id, tagName: "DIV", value: "", min: "0.05", max: "0.95", step: "0.01",
    checked: false, disabled: false, textContent: "", innerHTML: "", title: "",
    width: 800, height: 520, clientWidth: 800, clientHeight: 520,
    scrollWidth: 800, scrollHeight: 520, scrollTop: 0, offsetWidth: 800, offsetHeight: 36,
    files: [], options: [], children: [], parentElement: null, dataset: {},
    style: new Proxy({}, {
      get(t, k) {
        if (k === "setProperty") return (p, v) => { t[p] = v; };
        if (k === "getPropertyValue") return (p) => t[p] || "";
        return t[k] ?? "";
      },
      set(t, k, v) { t[k] = v; return true; }
    }),
    classList: { add() {}, remove() {}, toggle() { return false; }, contains() { return false; } },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    removeEventListener() {},
    dispatchEvent(evObj) { (listeners[evObj.type] || []).forEach(fn => fn(evObj)); },
    appendChild(c) { return c; }, insertBefore(c) { return c; }, replaceChildren() {},
    setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
    getBoundingClientRect() { return { left: 0, top: 0, right: 800, bottom: 520, width: 800, height: 520 }; },
    getContext() {
      return {
        fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "left",
        textBaseline: "alphabetic", globalAlpha: 1, imageSmoothingEnabled: true,
        setTransform() {}, resetTransform() {}, scale() {}, translate() {}, rotate() {},
        transform() {}, clearRect() {}, fillRect() {}, strokeRect() {},
        beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, arc() {}, ellipse() {},
        quadraticCurveTo() {}, bezierCurveTo() {}, rect() {}, roundRect() {},
        stroke() {}, fill() {}, clip() {}, save() {}, restore() {}, setLineDash() {},
        fillText() {}, strokeText() {}, drawImage() {},
        measureText(t) { return { width: String(t).length * 6 }; },
        createLinearGradient() { return { addColorStop() {} }; },
        createRadialGradient() { return { addColorStop() {} }; },
        getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
        putImageData() {}, createImageData(w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
      };
    },
    querySelector() { return makeEl(); }, querySelectorAll() { return []; }, closest() { return null; },
  };
}
function getEl(id) { if (!elems.has(id)) elems.set(id, makeEl(id)); return elems.get(id); }
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.parent = globalThis;
globalThis.top = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.devicePixelRatio = 1;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.requestAnimationFrame = () => 1;
globalThis.cancelAnimationFrame = () => {};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.Image = class { set src(_v) {} };
globalThis.document = {
  body: makeEl("body"), documentElement: makeEl("html"),
  getElementById: (id) => getEl(id),
  createElement: (tag) => makeEl(tag),
  querySelector: () => makeEl(), querySelectorAll: () => [],
  addEventListener() {}, removeEventListener() {},
};
"""
        js_probe = f"""
// 1. Verify 2D estimateActiveFrameAngularErrorsJS is strictly guarded in tof_3d mode!
const guardCount = estimateActiveFrameAngularErrorsJS(PEAKS, true);

// 2. Verify live fetch round-trip when sliding dMin down to d_cut_fine AND back to d_cut_coarse!
const fetchCalls = [];
globalThis.fetch = (url, opts) => {{
  const body = JSON.parse(opts.body);
  fetchCalls.push(body.d_min);
  const respObj = Math.abs(body.d_min - {d_cut_fine:.6f}) < 1e-3 ? {live_fine_json} : {live_coarse_json};
  return Promise.resolve({{ ok: true, json: () => Promise.resolve(respObj) }});
}};

state.dMin = {d_cut_fine:.6f};
repredictPeaksDynamic({{ preserveSource: true }});
scheduleLiveTofDeltaThetaSync(true);

setTimeout(() => {{
  const p002 = PEAKS.find(p => p.bank === 32 && p.hkl[0] === 0 && p.hkl[1] === 0 && p.hkl[2] === 2);
  const p003 = PEAKS.find(p => p.bank === 32 && p.hkl[0] === 0 && p.hkl[1] === 0 && p.hkl[2] === 3);
  const pillTooltip = getEl("hud-rms-pill").title || getEl("hud-rms-pill").dataset.uiTip || "";

  // Slide back to initial payload dMin and ensure it ALSO triggers a fresh sync!
  state.dMin = {d_cut_coarse:.6f};
  repredictPeaksDynamic({{ preserveSource: true }});
  scheduleLiveTofDeltaThetaSync(true);

  setTimeout(() => {{
    console.log(JSON.stringify({{
      guardCount,
      fetchCalls,
      err002: p002 ? p002.angular_error_deg : null,
      err003: p003 ? p003.angular_error_deg : null,
      pillTooltip,
    }}));
  }}, 15);
}}, 15);
}})();
"""
        js_path = tmp_path / "tof_3d_ui_test.js"
        js_path.write_text(
            js_harness + "\n" + js_src[:iife_end] + "\n" + js_probe, encoding="utf-8"
        )
        proc = subprocess.run(
            ["node", str(js_path)], check=True, capture_output=True, text=True
        )
        js_out = json.loads(proc.stdout.strip())
        assert js_out["guardCount"] == 0
        assert len(js_out["fetchCalls"]) == 2
        assert js_out["err002"] is not None and float(js_out["err002"]) < 0.05
        assert (
            js_out["err003"] is not None
            and abs(float(js_out["err003"]) - expected_dir_deg_3) < 0.05
        )
        assert "3D ToF" in js_out["pillTooltip"]
        assert "arccos" in js_out["pillTooltip"]


def test_divergent_quantile_isigi_colorscale_and_signed_mtz(tmp_path) -> None:
    """Verify warm quantile-normalized divergent I/σ & .mtz I/σ colorscale, center-bulk IQR placement, outlier fencing, and signed MTZ overlay."""
    import json
    import re
    import shutil
    import subprocess

    import gemmi
    import h5py
    import numpy as np

    from subhkl.viz.rainbow_zones import (
        apply_mtz_to_payload,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    cell_vals = [55.16, 81.19, 88.70, 90.0, 96.3, 90.0]
    b_mat = np.asarray(gemmi.UnitCell(*cell_vals).frac.mat.tolist(), dtype=float).T
    solve_h5 = tmp_path / "solve_isigi.h5"
    with h5py.File(solve_h5, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf["sample/space_group"] = "I 1 2 1"
        hf["sample/cell"] = np.array(cell_vals, dtype=float)
        hf["sample/U"] = np.eye(3)
        hf["sample/B"] = b_mat
        hf["goniometer/R"] = np.eye(3)[None, :, :]
        hf["goniometer/angles"] = np.zeros((1, 1), dtype=float)
        hf["instrument/wavelength"] = np.array([2.0, 4.5], dtype=float)

    base = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        instrument="CG4D",
        space_group="I 1 2 1",
        wavelength_range=(2.0, 4.5),
        d_min=2.0,
        frame_index=0,
        include_panel_images=False,
    )
    assert len(base["predicted_peaks"]) >= 2

    # 1. Signed negative & positive .mtz I/σ preserved through _overlay_mtz_columns
    h0 = [int(x) for x in base["predicted_peaks"][0]["hkl"]]
    h1 = [int(x) for x in base["predicted_peaks"][1]["hkl"]]
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name(base["space_group"])
    mtz.set_cell_for_all(gemmi.UnitCell(*base["cell"]))
    mtz.add_dataset("unmerged_signed")
    mtz.add_column("I", "J")
    mtz.add_column("SIGI", "Q")
    mtz.set_data(
        np.asarray(
            [
                [h0[0], h0[1], h0[2], -25.0, 10.0],
                [h1[0], h1[1], h1[2], 84.0, 12.0],
            ],
            dtype=np.float32,
        )
    )
    mtz_path = tmp_path / "signed_unmerged.mtz"
    mtz.write_to_file(str(mtz_path))
    overlaid = apply_mtz_to_payload(
        base, str(mtz_path), {"name": "signed_unmerged.mtz"}
    )
    by_hkl = {
        tuple(int(x) for x in p["hkl"]): p
        for p in overlaid["predicted_peaks"]
        if p.get("mtz_isigi") is not None
    }
    assert abs(float(by_hkl[tuple(h0)]["mtz_isigi"]) - (-2.5)) < 1e-3
    assert abs(float(by_hkl[tuple(h1)]["mtz_isigi"]) - 7.0) < 1e-3

    if not shutil.which("node"):
        return

    ui_html = render_rainbow_html(overlaid)
    inline_js = re.findall(r"<script>([\s\S]*?)</script>", ui_html)[-1]
    fn_names = [
        "getColorCrossover",
        "applyCrossoverCurve",
        "peakUnscaledIsigi",
        "peakMtzIsigi",
        "computeQuantileScaleBounds",
        "getPeakDataAvailability",
        "residualToColor",
        "isigiToColor",
        "buildColorscaleGradientSpec",
    ]
    extracted = []
    for fn_name in fn_names:
        m = re.search(rf"\bfunction\s+{fn_name}\s*\(", inline_js)
        assert m is not None, f"Missing JS function {fn_name}"
        start = m.start()
        depth = 0
        in_str = None
        esc = False
        i = m.end() - 1
        while i < len(inline_js):
            ch = inline_js[i]
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif in_str:
                if ch == in_str:
                    in_str = None
            elif ch in ("'", '"', "`"):
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    extracted.append(inline_js[start : i + 1])
                    break
            i += 1

    # 2. Realistic skewed + signed I/σ distribution with extreme outliers AND top-k biased SAMPLE_PEAKS:
    #    - Active-frame PEAKS (n=200): signed background-dominated + Bragg reflections + [-999, +76e6] outliers
    #    - SAMPLE_PEAKS: artificially inflated top-k pool (SNR ~ 500..4000) that must NOT skew frame I/σ bounds!
    rng = np.random.default_rng(12345)
    core_snr = np.concatenate(
        [
            rng.normal(0.6, 1.4, size=140),
            rng.lognormal(mean=1.5, sigma=0.7, size=58),
            np.array([-999.0, 76_682_336.0]),
        ]
    )
    peaks_js = [
        {
            "id": idx + 1,
            "intensity": float(val * 10.0),
            "sigma": 10.0,
            "mtz_i": float((val - 0.5) * 8.0),
            "mtz_sigi": 8.0,
            "careless_residual": float(np.clip(val * 0.4, -4.0, 4.0)),
        }
        for idx, val in enumerate(core_snr)
    ]
    inflated_sample_js = [
        {
            "id": 1000 + idx,
            "intensity": float((600.0 + idx * 25.0) * 10.0),
            "sigma": 10.0,
            "mtz_i": float(val * 8.0),
            "mtz_sigi": 8.0,
        }
        for idx, val in enumerate(core_snr[:80])
    ]

    js_probe = f"""
const DEFAULT_COLOR_CROSSOVER = 0.38;
const ORBIT_PALETTE = ["#ff38b4", "#00e5ff"];
const state = {{
  colorPercentile: 0.50,
  colorCrossover: 0.38,
  peakSource: "integrated",
  wulffColorMode: "isigi",
  lamMin: 1.0,
  lamMax: 4.5,
}};
const DATA = {{ mtz_overlay_name: "signed_unmerged.mtz" }};
let _dynPredEpoch = 0;
let _peakDataAvailabilitySig = "";
let _peakDataAvailability = {{}};
const PEAKS = {json.dumps(peaks_js)};
const SAMPLE_PEAKS = {json.dumps(inflated_sample_js)};
function scaleToColor() {{ return "rgba(0,0,0,1)"; }}
function hexToRgba() {{ return "rgba(0,0,0,1)"; }}
function wavelengthToColor() {{ return "rgba(0,0,0,1)"; }}
{"\n".join(extracted)}

const avail = getPeakDataAvailability();
const b = avail.isigiBounds;
const colLo = isigiToColor(b.min, 1.0, b);
const colQ25 = isigiToColor(b.q25, 1.0, b);
const colMed = isigiToColor(b.median, 1.0, b);
const colQ75 = isigiToColor(b.q75, 1.0, b);
const colHi = isigiToColor(b.max, 1.0, b);
const colExtremePos = isigiToColor(76682336.0, 1.0, b);
const colExtremeNeg = isigiToColor(-999.0, 1.0, b);
const resLo = residualToColor(avail.resMin, 1.0, avail);
const resMed = residualToColor(avail.resMedian, 1.0, avail);
const resHi = residualToColor(avail.resMax, 1.0, avail);
const specIsigi = buildColorscaleGradientSpec("isigi", 0.50);

// Also check truncated positive-only frame I/σ >= 2.5 (e.g. single-frame peak finder threshold)
const truncBounds = computeQuantileScaleBounds(
  Array.from({{ length: 120 }}, (_, k) => 2.5 + Math.pow(k / 119, 2.4) * 28.0)
);
const truncQ25Col = isigiToColor(truncBounds.q25, 1.0, truncBounds);
const truncMedCol = isigiToColor(truncBounds.median, 1.0, truncBounds);
const truncQ75Col = isigiToColor(truncBounds.q75, 1.0, truncBounds);

console.log(JSON.stringify({{
  signedNegPeak: peakUnscaledIsigi({{ intensity: -32.0, sigma: 8.0 }}),
  signedNegMtz: peakMtzIsigi({{ mtz_i: -18.0, mtz_sigi: 6.0 }}),
  bounds: b,
  colLo, colQ25, colMed, colQ75, colHi, colExtremePos, colExtremeNeg,
  resLo, resMed, resHi,
  specIsigi,
  truncBounds, truncQ25Col, truncMedCol, truncQ75Col,
}}));
"""
    proc = subprocess.run(
        ["node", "-e", js_probe], check=True, capture_output=True, text=True
    )
    out = json.loads(proc.stdout.strip())

    def _rgb(s: str) -> tuple[int, int, int]:
        nums = [int(x) for x in re.findall(r"\d+", s)[:3]]
        return nums[0], nums[1], nums[2]

    def _dist(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
        return float(np.hypot(np.hypot(c1[0] - c2[0], c1[1] - c2[1]), c1[2] - c2[2]))

    # Signed negative I/σ preserved without clamping at 0
    assert abs(out["signedNegPeak"] - (-4.0)) < 1e-6
    assert abs(out["signedNegMtz"] - (-3.0)) < 1e-6

    b = out["bounds"]
    # Unbiased PEAKS median (~0.8..2.0) & IQR used instead of top-k SAMPLE_PEAKS (>600), spanning full 0-100th percentile
    assert b["min"] == -999.0
    assert 0.0 < b["median"] < 4.0
    assert 1.5 < b["q75"] < 10.0
    assert b["max"] == 76_682_336.0
    assert out["colExtremePos"] == out["colHi"] == "rgba(232, 56, 24, 1.000)"
    assert out["colExtremeNeg"] == out["colLo"] == "rgba(20, 184, 166, 1.000)"
    assert out["colMed"] == "rgba(254, 232, 170, 1.000)"

    # Central 50% bulk [Q25, Q75] sits much closer to warm Champagne-Gold center than to either extreme pole!
    c_lo = _rgb(out["colLo"])
    c_q25 = _rgb(out["colQ25"])
    c_med = _rgb(out["colMed"])
    c_q75 = _rgb(out["colQ75"])
    c_hi = _rgb(out["colHi"])
    assert _dist(c_q25, c_med) < _dist(c_q25, c_lo)
    assert _dist(c_q75, c_med) < _dist(c_q75, c_hi)

    # Same center-bulk guarantee holds even on lower-truncated I/σ >= 2.5 frame distributions!
    ct_q25 = _rgb(out["truncQ25Col"])
    ct_med = _rgb(out["truncMedCol"])
    ct_q75 = _rgb(out["truncQ75Col"])
    assert ct_med == (254, 232, 170)
    assert _dist(ct_q25, ct_med) < _dist(ct_q25, c_lo)
    assert _dist(ct_q75, ct_med) < _dist(ct_q75, c_hi)

    # Distinct from ΔI/σ (Blue-White-Rose) at lower pole, midpoint (warmer gold vs cool white), and upper pole
    assert _dist(c_lo, _rgb(out["resLo"])) > 50.0
    assert _dist(c_med, _rgb(out["resMed"])) > 70.0
    assert c_med[0] > c_med[1] > c_med[2]  # warm sunlit gold midpoint
    assert _dist(c_hi, _rgb(out["resHi"])) > 65.0

    assert "·" in out["specIsigi"]["label"]
    assert "76682" not in out["specIsigi"]["label"]
    assert "quantile-normalized range" in out["specIsigi"]["tooltip"]
    assert "bulk IQR" in out["specIsigi"]["tooltip"]


def test_slider_direct_numerical_input_integ_button_and_stationary_scope_btn(
    tmp_path,
) -> None:
    """Verify click-to-edit λ_min/λ_max & d_min numerical inputs, warm I/σ Integ button styling, and stationary 1-Frame/All-Runs button."""
    import json
    import re
    import shutil
    import subprocess

    import gemmi
    import h5py
    import numpy as np

    from subhkl.viz.rainbow_zones import (
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    cell_vals = [55.16, 81.19, 88.70, 90.0, 96.3, 90.0]
    b_mat = np.asarray(gemmi.UnitCell(*cell_vals).frac.mat.tolist(), dtype=float).T
    solve_h5 = tmp_path / "solve_ui_polish.h5"
    with h5py.File(solve_h5, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf["sample/space_group"] = "I 1 2 1"
        hf["sample/cell"] = np.array(cell_vals, dtype=float)
        hf["sample/U"] = np.eye(3)
        hf["sample/B"] = b_mat
        hf["goniometer/R"] = np.eye(3)[None, :, :]
        hf["goniometer/angles"] = np.zeros((1, 1), dtype=float)
        hf["instrument/wavelength"] = np.array([2.0, 4.5], dtype=float)

    payload = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        instrument="CG4D",
        space_group="I 1 2 1",
        wavelength_range=(2.0, 4.5),
        d_min=2.0,
        frame_index=0,
        include_panel_images=False,
    )
    html = render_rainbow_html(payload)

    # 1. HTML/CSS assertions: clickable λ_min/λ_max & d_min spans, static LOD-gold Integ button, fixed-width 1-Frame/All-Runs & corner caret button
    assert 'id="wl-val-min"' in html and 'id="wl-val-max"' in html
    assert 'id="lbl-dmin-val" class="editable-slider-val"' in html
    assert "#bef264" not in html
    assert "#a3f22c" not in html
    assert "rgba(163, 242, 44" not in html
    assert (
        re.search(
            r'\.seg-toggle button\[data-peak-src="integrated"\]\.active-btn\s*\{[^}]*color:\s*#fbbf24',
            html,
        )
        is not None
    )
    assert re.search(r"#btn-wulff-scope\s*\{[^}]*width:\s*54px", html) is not None
    assert re.search(r"#stereo-title\s*\{[^}]*flex:\s*1\s+1\s+0", html) is not None
    assert (
        re.search(r"\.stereo-win-controls\s*\{[^}]*margin-left:\s*auto", html)
        is not None
    )
    assert 'class="stereo-win-controls"' in html
    assert 'id="btn-wulff-expand" type="button" data-caret="up"' in html

    if not shutil.which("node"):
        return

    inline_js = re.findall(r"<script>([\s\S]*?)</script>", html)[-1]
    fn_names = [
        "getDefaultWavelengthForInstrument",
        "getColorCrossover",
        "applyCrossoverCurve",
        "getDynamicDminFloor",
        "getWavelengthSliderBounds",
        "updateWavelengthLabels",
        "syncWavelengthControls",
        "syncDminControls",
        "updateColorscaleSliderGradient",
        "syncPeakSourceButtons",
        "syncWulffUI",
        "applyDirectWavelengthInput",
        "applyDirectDminInput",
        "openWavelengthDirectInput",
        "openDminDirectInput",
    ]
    extracted = []
    for fn_name in fn_names:
        m = re.search(rf"(?m)^  function {fn_name}\b[\s\S]*?^  \}}", inline_js)
        assert m is not None, f"Missing JS function {fn_name}"
        extracted.append(m.group(0))

    js_probe = f"""
const MIN_WL_SPAN = 0.05;
const ABS_DMIN_FLOOR = 0.25;
const DYNAMIC_DMIN_HARD_FLOOR = 0.25;
const DEFAULT_COLOR_CROSSOVER = 0.38;
const WULFF_ALL_RUNS_MIN_ISIGI = 2.5;
const INSTRUMENT_WAVELENGTH_DEFAULTS = {{ CG4D: [2.0, 4.5] }};
const _payloadBaseParams = {{ lamMin: 2.0, lamMax: 4.5, dMin: 2.0 }};
const DATA = {{
  wavelength_range: [2.0, 4.5],
  d_min: 2.0,
  artifact_kind: "integrator",
  mtz_overlay_name: "mandi_lysozyme_very_long_name.mtz",
}};
const state = {{
  lamMin: 2.0,
  lamMax: 4.5,
  dMin: 2.0,
  dMinUserSeq: 0,
  colorCrossover: 0.38,
  wulffColorMode: "isigi",
  peakSource: "integrated",
  showStereo: true,
  wulffExpanded: true,
  wulffScope: "single_frame",
  showWulffFriedel: true,
}};
const PREDICTED_PEAKS = [{{ id: 1 }}];
const INTEGRATED_PEAKS = [{{ id: 1, intensity: 100, sigma: 10 }}];
let repredictCount = 0;
function handleDynamicPredictionControlChange() {{ repredictCount += 1; updateColorscaleSliderGradient(); }}
function buildColorscaleGradientSpec(_mode, c) {{ return {{ gradient: `grad(${{c}})`, label: "0 · 3 · 15", tooltip: "tip" }}; }}
function hasIntegratedIsigi() {{ return true; }}
function hasMtzIsigi() {{ return true; }}
function hasCarelessResiduals() {{ return false; }}
function hasCarelessScales() {{ return false; }}
function isWulffSnrCutActive() {{ return state.wulffScope === "all_runs"; }}
function updateMergeScoreboard() {{}}

function makeEl(id, tag = "SPAN", initText = "") {{
  const listeners = {{}};
  const props = {{}};
  const cls = new Set();
  const children = [];
  let _text = initText;
  const el = {{
    id,
    tagName: tag,
    value: "",
    min: "0.20",
    max: "6.50",
    title: "",
    disabled: false,
    options: [
      {{ value: "wavelength", textContent: "Color: λ/Zone", style: {{}} }},
      {{ value: "isigi", textContent: "Color: I/σ", style: {{}} }},
      {{ value: "fobs", textContent: "Color: |F_obs|²", style: {{}} }},
      {{ value: "mtz_fcalc", textContent: "Color: |F_calc|²", style: {{}} }},
      {{ value: "ipred_isigi", textContent: "Color: I_pred/σ", style: {{}} }},
      {{ value: "residual", textContent: "Color: ΔI/σ Residual", style: {{}} }},
      {{ value: "scale", textContent: "Color: Scale", style: {{}} }},
    ],
    style: {{
      left: "",
      width: "",
      height: "",
      borderColor: "",
      setProperty(k, v) {{ props[k] = v; }},
      getPropertyValue(k) {{ return props[k] || ""; }},
    }},
    classList: {{
      toggle(c, on) {{ if (on) cls.add(c); else cls.delete(c); }},
      contains(c) {{ return cls.has(c); }},
    }},
    get textContent() {{
      if (children.length > 0) return children.map((c) => c.textContent ?? "").join("");
      return _text;
    }},
    set textContent(v) {{
      children.length = 0;
      _text = String(v);
    }},
    appendChild(child) {{ children.push(child); return child; }},
    querySelector(sel) {{
      if (sel.startsWith("input")) return children.find((c) => c.tagName === "INPUT") || null;
      return null;
    }},
    setAttribute(k, v) {{ el[k] = v; }},
    addEventListener(type, fn) {{ (listeners[type] = listeners[type] || []).push(fn); }},
    dispatch(type, ev = {{}}) {{
      const eventObj = Object.assign({{
        preventDefault() {{}},
        stopPropagation() {{}},
      }}, ev);
      (listeners[type] || []).slice().forEach((fn) => fn(eventObj));
    }},
    focus() {{}},
    select() {{}},
    _children: children,
  }};
  return el;
}}

const els = {{
  "wl-readout": makeEl("wl-readout", "SPAN"),
  "wl-val-min": makeEl("wl-val-min", "SPAN", "2.00"),
  "wl-val-max": makeEl("wl-val-max", "SPAN", "4.50"),
  "lbl-dmin-val": makeEl("lbl-dmin-val", "SPAN", "2.00 Å"),
  "slider-wl-min": makeEl("slider-wl-min", "INPUT"),
  "slider-wl-max": makeEl("slider-wl-max", "INPUT"),
  "slider-dmin": makeEl("slider-dmin", "INPUT"),
  "wl-track-fill": makeEl("wl-track-fill", "DIV"),
  "wl-knob-min": makeEl("wl-knob-min", "BUTTON"),
  "wl-knob-max": makeEl("wl-knob-max", "BUTTON"),
  "slider-color-emphasis": makeEl("slider-color-emphasis", "INPUT"),
  "lbl-color-emphasis": makeEl("lbl-color-emphasis", "SPAN"),
  "btn-peaks-pred": makeEl("btn-peaks-pred", "BUTTON"),
  "btn-peaks-integ": makeEl("btn-peaks-integ", "BUTTON"),
  "legend-peak-shape-icon": makeEl("legend-peak-shape-icon", "SPAN"),
  "legend-peak-shape-label": makeEl("legend-peak-shape-label", "SPAN"),
  "stereo-inset": makeEl("stereo-inset", "DIV"),
  "btn-stereo-toggle": makeEl("btn-stereo-toggle", "BUTTON"),
  "btn-wulff-scope": makeEl("btn-wulff-scope", "BUTTON", "1-Frame"),
  "btn-wulff-friedel": makeEl("btn-wulff-friedel", "BUTTON"),
  "btn-wulff-expand": makeEl("btn-wulff-expand", "BUTTON"),
  "sel-wulff-color": makeEl("sel-wulff-color", "SELECT"),
  "sel-spot-color": makeEl("sel-spot-color", "SELECT"),
  "stereo-title": makeEl("stereo-title", "SPAN"),
}};
els["wl-readout"].appendChild({{ textContent: "[" }});
els["wl-readout"].appendChild(els["wl-val-min"]);
els["wl-readout"].appendChild({{ textContent: ", " }});
els["wl-readout"].appendChild(els["wl-val-max"]);
els["wl-readout"].appendChild({{ textContent: "] Å" }});

const slWlMin = els["slider-wl-min"];
const slWlMax = els["slider-wl-max"];
const slDmin = els["slider-dmin"];
const wlTrackFill = els["wl-track-fill"];
const wlKnobMin = els["wl-knob-min"];
const wlKnobMax = els["wl-knob-max"];
const slColorEmphasis = els["slider-color-emphasis"];
let _cachedPlotBox = null;
let _cachedPlotBoxKey = "";

const document = {{
  getElementById(id) {{ return els[id] || null; }},
  createElement(tag) {{ return makeEl("", String(tag).toUpperCase()); }},
  createTextNode(txt) {{ return {{ textContent: String(txt) }}; }},
}};

{"\n".join(extracted)}

// A. Click λ_min -> type 1.35 -> Tab jumps directly to λ_max -> type 8.25 (beyond initial 6.50 slider max!) -> Enter
openWavelengthDirectInput("min");
const inpMin = els["wl-val-min"].querySelector("input.inline-slider-input");
inpMin.value = "1.35";
inpMin.dispatch("keydown", {{ key: "Tab", shiftKey: false }});
const inpMax = els["wl-val-max"].querySelector("input.inline-slider-input");
const tabOpenedMax = Boolean(inpMax);
inpMax.value = "8.25";
inpMax.dispatch("keydown", {{ key: "Enter" }});

const afterWl = {{
  lamMin: state.lamMin,
  lamMax: state.lamMax,
  readout: els["wl-readout"].textContent,
  slMaxBound: parseFloat(slWlMax.max),
  slMinVal: slWlMin.value,
  slMaxVal: slWlMax.value,
}};

// Escape cancels without modifying λ_min
openWavelengthDirectInput("min");
const inpEsc = els["wl-val-min"].querySelector("input.inline-slider-input");
inpEsc.value = "3.99";
inpEsc.dispatch("keydown", {{ key: "Escape" }});
const lamMinAfterEsc = state.lamMin;

// B. Click d_min -> type 1.12 -> Enter updates d_min in 0.01 A steps; dragging slider snaps to 0.05 A steps; typing 0.02 clamps to dynamic floor
openDminDirectInput();
const inpD = els["lbl-dmin-val"].querySelector("input.inline-slider-input");
inpD.value = "1.12";
inpD.dispatch("keydown", {{ key: "Enter" }});
const dMinAfterCommit = state.dMin;
const lblDAfterCommit = els["lbl-dmin-val"].textContent;
const slDValAfterCommit = slDmin.value;
const slDStepAttr = slDmin.step;

slDmin.value = "1.73";
syncDminControls(slDmin.value, slDmin);
const dMinAfterSliderSnap = state.dMin;
const lblDAfterSliderSnap = els["lbl-dmin-val"].textContent;

openDminDirectInput();
const inpDFloor = els["lbl-dmin-val"].querySelector("input.inline-slider-input");
inpDFloor.value = "0.02";
inpDFloor.dispatch("keydown", {{ key: "Enter" }});
const dFloorExpected = getDynamicDminFloor();
const dMinAfterFloor = state.dMin;

// C. Static LOD-gold (#fbbf24) Integ button & legend ellipse icon
syncPeakSourceButtons();
const legBorderInteg = els["legend-peak-shape-icon"].style.borderColor;
const btnIntegActive = els["btn-peaks-integ"].classList.contains("active-btn");

// D. Wulff scope button label stays concise ("1-Frame" vs "All-Runs") & corner expand button switches ^ (up) <-> v (down) caret
state.wulffExpanded = false;
state.wulffScope = "single_frame";
syncWulffUI();
const scopeSingleLbl = els["btn-wulff-scope"].textContent;
const expandCornerCaret = els["btn-wulff-expand"]["data-caret"];
const expandCornerHtml = els["btn-wulff-expand"].innerHTML;
const expandCornerTip = els["btn-wulff-expand"].title;

state.wulffExpanded = true;
state.wulffScope = "all_runs";
syncWulffUI();
const scopeAllLbl = els["btn-wulff-scope"].textContent;
const scopeAllTip = els["btn-wulff-scope"].title;
const stereoTitleAll = els["stereo-title"].textContent;
const expandSplitCaret = els["btn-wulff-expand"]["data-caret"];
const expandSplitHtml = els["btn-wulff-expand"].innerHTML;
const expandSplitTip = els["btn-wulff-expand"].title;

console.log(JSON.stringify({{
  tabOpenedMax,
  afterWl,
  lamMinAfterEsc,
  dMinAfterCommit,
  lblDAfterCommit,
  slDValAfterCommit,
  dFloorExpected,
  dMinAfterFloor,
  repredictCount,
  slDStepAttr,
  dMinAfterSliderSnap,
  lblDAfterSliderSnap,
  legBorderInteg,
  btnIntegActive,
  scopeSingleLbl,
  scopeAllLbl,
  scopeAllTip,
  stereoTitleAll,
  expandCornerCaret,
  expandCornerHtml,
  expandCornerTip,
  expandSplitCaret,
  expandSplitHtml,
  expandSplitTip,
}}));
"""
    js_file = tmp_path / "ui_polish_probe.js"
    js_file.write_text(js_probe, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(js_file)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip())

    assert out["tabOpenedMax"] is True
    assert abs(out["afterWl"]["lamMin"] - 1.35) < 1e-6
    assert abs(out["afterWl"]["lamMax"] - 8.25) < 1e-6
    assert out["afterWl"]["readout"] == "[1.35, 8.25] Å"
    assert out["afterWl"]["slMaxBound"] >= 8.25
    assert out["afterWl"]["slMinVal"] == "1.35"
    assert out["afterWl"]["slMaxVal"] == "8.25"
    assert abs(out["lamMinAfterEsc"] - 1.35) < 1e-6

    assert abs(out["dMinAfterCommit"] - 1.12) < 1e-6
    assert out["lblDAfterCommit"] == "1.12 Å"
    assert out["slDValAfterCommit"] == "1.12"
    assert out["slDStepAttr"] == "0.05"
    assert abs(out["dMinAfterSliderSnap"] - 1.75) < 1e-6
    assert out["lblDAfterSliderSnap"] == "1.75 Å"
    assert abs(out["dMinAfterFloor"] - out["dFloorExpected"]) < 0.02
    assert out["repredictCount"] >= 4

    assert out["btnIntegActive"] is True
    assert out["legBorderInteg"] == "#fbbf24"

    assert out["scopeSingleLbl"] == "1-Frame"
    assert out["scopeAllLbl"] == "All-Runs"
    assert "I/σ ≥ 2.5" in out["scopeAllTip"]
    assert "ALL-RUNS · I/σ ≥ 2.5" in out["stereoTitleAll"]

    assert out["expandCornerCaret"] == "up"
    assert "3.5 10.5 8 6 12.5 10.5" in out["expandCornerHtml"]
    assert "Magnify" in out["expandCornerTip"]
    assert out["expandSplitCaret"] == "down"
    assert "3.5 6 8 10.5 12.5 6" in out["expandSplitHtml"]
    assert "Reduce" in out["expandSplitTip"]


def test_nodal_points_gap_bridge_zone_arc_hover_and_holz(tmp_path) -> None:
    """Verify {a*, b*, c*} + intersection nodal bull's-eyes, inter-panel gap overlap & hover badge, [uvw] arc hover badge, and ±1/±2 HOLZ."""
    import json
    import re
    import shutil
    import subprocess

    import h5py
    from scipy.spatial.transform import Rotation

    from subhkl.instrument.detector import Detector
    from subhkl.viz.rainbow_zones import (
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    assert shutil.which("node"), "Node.js required for nodal/HOLZ behavioral test"

    # Align cubic [111] along +Z so (-1,0,0), (0,-1,0), (0,0,-1) all scatter into forward 2theta = 70.53 deg,
    # 120 deg apart around the beam:
    # - Bank 1 & Bank 2 straddle a* with a 5-deg azimuthal gap right across a* (onPanel == False!)
    # - Bank 3 covers b* directly (onPanel == True)
    # - Bank 4 covers c* directly (onPanel == True)
    v_from = np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
    v_to = np.array([0.0, 0.0, 1.0], dtype=float)
    rot_axis = np.cross(v_from, v_to)
    rot_axis /= np.linalg.norm(rot_axis)
    rot_ang = float(np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0)))
    u_mat = Rotation.from_rotvec(rot_axis * rot_ang).as_matrix()
    a_cell = 12.0
    b_mat = np.eye(3, dtype=float) / a_cell
    ub_mat = u_mat @ b_mat
    ki = np.array([0.0, 0.0, 1.0], dtype=float)

    def _kf_for_hkl(hkl_vec: list[int]) -> np.ndarray:
        g = ub_mat @ np.asarray(hkl_vec, dtype=float)
        gh = g / np.linalg.norm(g)
        if np.dot(ki, gh) > 0:
            gh = -gh
        sth = -float(np.dot(ki, gh))
        kf = ki + 2.0 * sth * gh
        return kf / np.linalg.norm(kf)

    kf_a = _kf_for_hkl([1, 0, 0])
    kf_b = _kf_for_hkl([0, 1, 0])
    kf_c = _kf_for_hkl([0, 0, 1])

    def _make_flat_bank_at_dir(
        kf_dir: np.ndarray, az_offset_deg: float, width_m: float, height_m: float
    ) -> dict[str, object]:
        rot_y = Rotation.from_euler("y", az_offset_deg, degrees=True).as_matrix()
        d_center = (rot_y @ kf_dir) * 0.45
        n_hat = -d_center / np.linalg.norm(d_center)
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        u_hat = np.cross(up, n_hat)
        u_hat /= np.linalg.norm(u_hat)
        v_hat = np.cross(n_hat, u_hat)
        v_hat /= np.linalg.norm(v_hat)
        corner00 = d_center - 0.5 * width_m * u_hat - 0.5 * height_m * v_hat
        return {
            "panel": "flat",
            "panel_type": "flat",
            "center": corner00.tolist(),
            "uhat": u_hat.tolist(),
            "vhat": v_hat.tolist(),
            "width": float(width_m),
            "height": float(height_m),
            "m": 128,
            "n": 128,
        }

    kf_bc = _kf_for_hkl([0, 1, 1])
    # Bank 1 & Bank 2 sit +/-13 deg around kf_a with angular half-width ~10.1 deg -> ~5.8 deg gap containing kf_a!
    # Bank 5 sits on (0,1,1) = [1,0,0] x [0,1,-1] so a secondary zone-intersection hub is also on-panel at order <= 1.
    bank_cfgs = {
        1: _make_flat_bank_at_dir(kf_a, -13.0, 0.16, 0.22),
        2: _make_flat_bank_at_dir(kf_a, +13.0, 0.16, 0.22),
        3: _make_flat_bank_at_dir(kf_b, 0.0, 0.22, 0.22),
        4: _make_flat_bank_at_dir(kf_c, 0.0, 0.22, 0.22),
        5: _make_flat_bank_at_dir(kf_bc, 0.0, 0.22, 0.22),
    }
    det_objs = {bid: Detector(dict(cfg)) for bid, cfg in bank_cfgs.items()}
    # Confirm kf_a misses every bank while kf_b hits bank 3 and kf_c hits bank 4
    for bid, det in det_objs.items():
        hm_a, _, _ = det.reflections_mask(
            np.array([kf_a[0]]), np.array([kf_a[1]]), np.array([kf_a[2]])
        )
        assert not bool(hm_a[0]), f"Expected a* ray to lie in gap outside bank {bid}"
    assert bool(
        det_objs[3].reflections_mask(
            np.array([kf_b[0]]), np.array([kf_b[1]]), np.array([kf_b[2]])
        )[0][0]
    )
    assert bool(
        det_objs[4].reflections_mask(
            np.array([kf_c[0]]), np.array([kf_c[1]]), np.array([kf_c[2]])
        )[0][0]
    )

    solve_h5 = tmp_path / "nodal_holz_solve.h5"
    with h5py.File(solve_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.create_group("beam").create_dataset("ki_vec", data=ki)
        smp = hf.create_group("sample")
        smp.create_dataset("space_group", data=b"P 1")
        smp.create_dataset(
            "cell", data=np.array([a_cell, a_cell, a_cell, 90.0, 90.0, 90.0])
        )
        smp.create_dataset("U", data=u_mat)
        smp.create_dataset("B", data=b_mat)
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([1.2, 3.2], dtype=float)
        )
        dc_root = hf.create_group("detector_calibration")
        for bid, cfg in bank_cfgs.items():
            g = dc_root.create_group(f"bank_{bid}")
            g.create_dataset("panel_type", data=b"flat")
            g.create_dataset("center", data=np.array(cfg["center"], dtype=float))
            g.create_dataset("uhat", data=np.array(cfg["uhat"], dtype=float))
            g.create_dataset("vhat", data=np.array(cfg["vhat"], dtype=float))
            g.create_dataset("width", data=float(cfg["width"]))
            g.create_dataset("height", data=float(cfg["height"]))
            g.create_dataset("m", data=int(cfg["m"]))
            g.create_dataset("n", data=int(cfg["n"]))

    payload = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        instrument="MANDI",
        d_min=1.5,
        wavelength_range=(1.2, 3.2),
        include_panel_images=False,
    )
    # Restrict detector_configs & panels strictly to our 4 custom banks so Bank 1/2 gap around a* is preserved
    payload["detector_configs"] = {
        str(bid): bank_cfgs[bid] for bid in sorted(bank_cfgs)
    }
    payload["panels"] = [pn for pn in payload["panels"] if int(pn["bank"]) in bank_cfgs]
    if len(payload["panels"]) < len(bank_cfgs):
        # Populate 2D panel envelopes from the 4 custom Detector objects
        panels_custom = []
        for bid, det in sorted(det_objs.items()):
            corners_cyl = []
            corners_sph = []
            for rc, cc in [(0, 0), (127, 0), (127, 127), (0, 127)]:
                pt = np.asarray(det.pixel_to_lab(rc, cc), dtype=float).reshape(3)
                az = float(np.degrees(np.arctan2(pt[0], pt[2])))
                r3 = float(np.linalg.norm(pt))
                el = float(np.degrees(np.arcsin(np.clip(pt[1] / r3, -1.0, 1.0))))
                corners_cyl.append([az, float(pt[1])])
                corners_sph.append([az, el])
            azs = [c[0] for c in corners_cyl]
            ys = [c[1] for c in corners_cyl]
            els_v = [c[1] for c in corners_sph]
            panels_custom.append(
                {
                    "id": bid,
                    "bank": bid,
                    "name": f"Bank {bid}",
                    "n_rows": 128,
                    "n_cols": 128,
                    "az_min": min(azs),
                    "az_max": max(azs),
                    "y_min": min(ys),
                    "y_max": max(ys),
                    "el_min": min(els_v),
                    "el_max": max(els_v),
                    "corners": corners_cyl,
                    "corners_sph": corners_sph,
                }
            )
        payload["panels"] = panels_custom

    html_doc = render_rainbow_html(payload)
    assert 'id="btn-nodal-points"' in html_doc
    assert 'id="btn-holz-1"' in html_doc
    assert 'id="btn-holz-2"' in html_doc
    script = re.findall(r"<script>([\s\S]*?)</script>", html_doc)[-1]
    script_open = script[: script.rfind("})();")]

    js_file = tmp_path / "nodal_holz_check.js"
    js_file.write_text(
        """
const elems = new Map();
const ctxLogs = new Map();
let rafCb = null;
function makeCtx(id = "") {
  const log = { strokes: [], texts: [], textCalls: [], arcs: [], fills: [] };
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
    fill() { log.fills.push(String(ctx.fillStyle)); },
    clip() {}, save() {}, restore() {}, setLineDash() {},
    fillText(t, x, y) { log.texts.push(String(t)); log.textCalls.push({ t: String(t), x: Number(x), y: Number(y) }); },
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
  const cw = isDetCanvas ? 900 : 300;
  const ch = isDetCanvas ? 540 : 240;
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
    click() { el.dispatchEvent({ type: "click", target: el, currentTarget: el, preventDefault() {}, stopPropagation() {} }); },
    focus() {}, blur() {}, select() {}, remove() {},
    appendChild(chEl) { this.children.push(chEl); chEl.parentElement = this; return chEl; },
    insertBefore(chEl) { this.children.push(chEl); chEl.parentElement = this; return chEl; },
    replaceChildren(...chEls) { this.children = chEls; },
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
        + script_open
        + """
const detCanvasEl = getEl("canvas-detector");
detCanvasEl.getContext("2d");
const detLog = ctxLogs.get("canvas-detector");

// 1. Even when NONE is clicked (activeZones empty), {a⋆, b⋆, c⋆} principal nodal points MUST be visible & labeled with superscript ⋆!
getEl("btn-zones-none").click();
detLog.texts.length = 0;
detLog.textCalls.length = 0;
renderDetectorCanvas(true);
const nodalsWhenNone = getVisibleNodalPoints();
const principalSymsWhenNone = nodalsWhenNone.filter(n => n.isPrincipal).map(n => n.principalSymbol).sort();
const canvasLabelsWhenNone = ["a", "b", "c"].every(baseLbl => {
  const idx = detLog.textCalls.findIndex((c, i) => (
    c.t === baseLbl &&
    detLog.textCalls[i + 1] &&
    detLog.textCalls[i + 1].t === "⋆" &&
    detLog.textCalls[i + 1].y <= c.y - 3.5
  ));
  return idx >= 0;
});

// Verify a⋆ lies in the inter-panel gap (onPanel === false), while b⋆ and c⋆ lie on panels (onPanel === true)
const nodalA = nodalsWhenNone.find(n => n.principalSymbol === "a⋆");
const nodalB = nodalsWhenNone.find(n => n.principalSymbol === "b⋆");
const nodalC = nodalsWhenNone.find(n => n.principalSymbol === "c⋆");

// Check screen-space circle radius of gap-straddling a⋆: must exceed pixel distance to nearest panel edge so it overlaps the panel!
const hitRecA = _detNodalScreenHits.find(h => h.nodal.principalSymbol === "a⋆" && h.x >= 0 && h.x <= 900);
const [edgeAx, edgeAy] = detToScreen(nodalA.nearAz, state.projection === "spherical" ? nodalA.nearEl : nodalA.nearY, 900, 540, 0.45);
const distCenterToPanelPx = Math.hypot(hitRecA.x - edgeAx, hitRecA.y - edgeAy);
const panelOverlapPx = hitRecA.r - distCenterToPanelPx;

// Hover ON THE ADJACENT PANEL inside the overlapping rim of a* (3 px past the panel boundary onto the panel!)
const dirToEdgeX = (edgeAx - hitRecA.x) / Math.max(1e-6, distCenterToPanelPx);
const dirToEdgeY = (edgeAy - hitRecA.y) / Math.max(1e-6, distCenterToPanelPx);
const hoverOnPanelX = edgeAx + dirToEdgeX * 2.5;
const hoverOnPanelY = edgeAy + dirToEdgeY * 2.5;
detCanvasEl.dispatchEvent({ type: "mousemove", clientX: hoverOnPanelX, clientY: hoverOnPanelY, buttons: 0 });
detLog.texts.length = 0;
renderDetectorCanvas(true);
const gapHoverBadgeTexts = [...detLog.texts];
const gapInspectorHtml = getEl("inspector-box").innerHTML;

// 2. Re-enable <=1 Laue zones: secondary zone intersections appear, and hovering a [uvw] zone arc shows the zone badge!
detCanvasEl.dispatchEvent({ type: "mouseleave" });
setMaxZoneOrder(1);
renderDetectorCanvas(true);
const nodalsOrder1 = getVisibleNodalPoints();
const secondaryCountOrder1 = nodalsOrder1.filter(n => !n.isPrincipal).length;
const secRadii = _detNodalScreenHits.filter(h => !h.nodal.isPrincipal).map(h => h.r);
const maxSecondaryR = secRadii.length > 0 ? Math.max(...secRadii) : 0;
const minSecondaryR = secRadii.length > 0 ? Math.min(...secRadii) : 0;

// Pick a rendered zone-arc segment midpoint well away from peaks and nodal points
let chosenArcPt = null;
let chosenArcLabel = null;
for (const seg of _detZoneArcScreenSegs) {
  for (let i = 0; i + 1 < seg.pts.length; i++) {
    const mx = 0.5 * (seg.pts[i][0] + seg.pts[i + 1][0]);
    const my = 0.5 * (seg.pts[i][1] + seg.pts[i + 1][1]);
    if (mx < 30 || mx > 870 || my < 30 || my > 510) continue;
    if (!findNearestPeakDetector(mx, my) && !findNearestNodalDetector(mx, my)) {
      chosenArcPt = [mx, my];
      chosenArcLabel = seg.zone.label;
      break;
    }
  }
  if (chosenArcPt) break;
}
let zoneArcBadgeTexts = [];
let zoneArcInspectorHtml = "";
if (chosenArcPt) {
  detCanvasEl.dispatchEvent({ type: "mousemove", clientX: chosenArcPt[0], clientY: chosenArcPt[1], buttons: 0 });
  detLog.texts.length = 0;
  renderDetectorCanvas(true);
  zoneArcBadgeTexts = [...detLog.texts];
  zoneArcInspectorHtml = getEl("inspector-box").innerHTML;
}

// 3. Toggle ±1 and ±2 Higher-Order Laue Zones (#btn-holz-1, #btn-holz-2):
//    verify small-circle cone equation kf·t_lab == ki·t_lab + N*lam/d_direct and reflection capture!
detCanvasEl.dispatchEvent({ type: "mouseleave" });
getEl("btn-holz-1").click();
getEl("btn-holz-2").click();
const holzBtn1Active = getEl("btn-holz-1").classList.contains("active-btn");
const holzBtn2Active = getEl("btn-holz-2").classList.contains("active-btn");
const holzSegs = getHolzSegmentsForActiveZones();
const holzOrdersSeen = Array.from(new Set(holzSegs.map(h => Math.abs(h.holzN)))).sort();

// Verify exact Ewald small-circle kinematics on a sampled HOLZ point via inverse ray direction
const sampleZone = ZONES.find(z => state.activeZones.has(z.zone_id));
const synthHolzPeak1 = { hkl: [1, 0, 0], zone_ids: [] }; // for uvw=[1,0,0] or [1,1,0], |h·u| in {1, 2}
// Keep only one known active zone [1, 0, 0] to test exact N=1 and N=2 capture vs N=3 rejection
const z100 = ZONES.find(z => Array.isArray(z.uvw) && z.uvw[0] === 1 && z.uvw[1] === 0 && z.uvw[2] === 0) || ZONES[0];
state.activeZones = new Set([z100.zone_id]);
const u0 = z100.uvw;
// Construct HKLs with h·u = 1, 2, 3 (since u0 has a +/-1 entry)
const idxOne = u0.findIndex(v => Math.abs(v) === 1);
function makeLayerPeak(nVal) {
  const hkl = [0, 0, 0];
  hkl[idxOne] = nVal * Math.sign(u0[idxOne]);
  return { hkl, zone_ids: [] };
}
const capN1 = isPeakCapturedByActiveZone(makeLayerPeak(1));
const capN2 = isPeakCapturedByActiveZone(makeLayerPeak(2));
const capN3 = isPeakCapturedByActiveZone(makeLayerPeak(3));

// Hover a HOLZ segment (with order-1 zones active) and verify floating badge reports HOLZ N=
setMaxZoneOrder(1);
renderDetectorCanvas(true);
let holzHoverTexts = [];
let hPt = null;
for (const seg of _detZoneArcScreenSegs) {
  if (!seg.holzN || seg.pts.length < 2) continue;
  for (let i = 0; i + 1 < seg.pts.length; i++) {
    const mx = 0.5 * (seg.pts[i][0] + seg.pts[i + 1][0]);
    const my = 0.5 * (seg.pts[i][1] + seg.pts[i + 1][1]);
    if (
      mx >= 20 && mx <= 880 && my >= 20 && my <= 520 &&
      !findNearestPeakDetector(mx, my) &&
      !findNearestNodalDetector(mx, my)
    ) {
      const zHit = findNearestZoneArcDetector(mx, my);
      if (zHit && zHit.holzN !== 0) {
        hPt = [mx, my];
        break;
      }
    }
  }
  if (hPt) break;
}
if (hPt) {
  detCanvasEl.dispatchEvent({ type: "mousemove", clientX: hPt[0], clientY: hPt[1], buttons: 0 });
  detLog.texts.length = 0;
  renderDetectorCanvas(true);
  holzHoverTexts = [...detLog.texts];
}

// Verify that when a* is > 25% of a panel width away from every panel (removing Banks 1 & 2), a* is NOT drawn
const savedPanels = [...PANELS];
const savedCfgs = Object.assign({}, DATA.detector_configs);
PANELS.splice(0, PANELS.length, ...savedPanels.filter(p => (p.bank ?? p.id) >= 3));
delete DATA.detector_configs["1"];
delete DATA.detector_configs["2"];
_fullFacePanelsCacheKey = "";
_nodalPointsCacheSig = "";
const symsWithoutBank12 = getVisibleNodalPoints().filter(n => n.isPrincipal).map(n => n.principalSymbol).sort();
PANELS.splice(0, PANELS.length, ...savedPanels);
DATA.detector_configs = savedCfgs;
_fullFacePanelsCacheKey = "";
_nodalPointsCacheSig = "";

// Toggle #btn-nodal-points off -> nodals hidden
getEl("btn-nodal-points").click();
const nodalsWhenOff = getVisibleNodalPoints().length;

console.log(JSON.stringify({
  principalSymsWhenNone,
  canvasLabelsWhenNone,
  aOnPanel: nodalA ? nodalA.onPanel : null,
  aGapPanelFrac: nodalA ? nodalA.gapPanelFrac : null,
  bOnPanel: nodalB ? nodalB.onPanel : null,
  cOnPanel: nodalC ? nodalC.onPanel : null,
  symsWithoutBank12,
  distCenterToPanelPx,
  nodalARadiusPx: hitRecA ? hitRecA.r : 0,
  panelOverlapPx,
  gapHoverHasBadge: (
    gapHoverBadgeTexts.includes("Nodal ") &&
    gapHoverBadgeTexts.includes("a") &&
    gapHoverBadgeTexts.includes("⋆") &&
    gapHoverBadgeTexts.some(t => t.includes("[gap]"))
  ),
  gapHoverUsesZonesLabel: gapHoverBadgeTexts.some(t => t.includes("zones)")),
  gapInspectorHasAStar: gapInspectorHtml.includes("a<sup") && gapInspectorHtml.includes(">⋆</sup>") && gapInspectorHtml.includes("% panel width"),
  secondaryCountOrder1,
  maxSecondaryR,
  minSecondaryR,
  chosenArcLabel,
  zoneArcHasBadge: Boolean(chosenArcLabel && zoneArcBadgeTexts.some(t => t.includes(`Zone ${chosenArcLabel}`))),
  zoneArcInspectorHasLabel: Boolean(chosenArcLabel && zoneArcInspectorHtml.includes(`Zone ${chosenArcLabel}`)),
  holzBtn1Active,
  holzBtn2Active,
  holzSegCount: holzSegs.length,
  holzOrdersSeen,
  capN1,
  capN2,
  capN3,
  holzHoverHasBadge: holzHoverTexts.some(t => t.includes("HOLZ N=")),
  nodalsWhenOff,
}));
})();
process.exit(0);
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["node", str(js_file)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["principalSymsWhenNone"] == ["a⋆", "b⋆", "c⋆"]
    assert out["canvasLabelsWhenNone"] is True
    assert out["aOnPanel"] is False
    assert 0.0 < out["aGapPanelFrac"] <= 0.25
    assert out["bOnPanel"] is True
    assert out["cOnPanel"] is True
    assert out["symsWithoutBank12"] == ["b⋆", "c⋆"]
    assert out["distCenterToPanelPx"] > 2.0
    assert out["panelOverlapPx"] >= 2.5
    assert out["maxSecondaryR"] <= 12.0
    assert abs(out["maxSecondaryR"] - out["minSecondaryR"]) < 1e-6
    assert out["gapHoverHasBadge"] is True
    assert out["gapHoverUsesZonesLabel"] is True
    assert out["gapInspectorHasAStar"] is True
    assert out["secondaryCountOrder1"] >= 1
    assert out["zoneArcHasBadge"] is True
    assert out["zoneArcInspectorHasLabel"] is True
    assert out["holzBtn1Active"] is True
    assert out["holzBtn2Active"] is True
    assert out["holzSegCount"] >= 2
    assert out["holzOrdersSeen"] == [1, 2]
    assert out["capN1"] is True
    assert out["capN2"] is True
    assert out["capN3"] is False
    assert out["holzHoverHasBadge"] is True
    assert out["nodalsWhenOff"] == 0


def test_cg4d_calibrated_flat_panels_nodal_alignment_and_holz_spot_counts(tmp_path):
    """Verify calibrated CG4D flat-panel post-indexing reload keeps nodal hubs aligned to <0.03 deg and counts HOLZ spots."""
    import json
    import re
    import shutil
    import subprocess

    import numpy as np

    from subhkl.instrument.detector import Detector
    from subhkl.viz.rainbow_zones import (
        _compute_zones_payload_for_geometry,
        _encode_panel_textures,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    if shutil.which("node") is None:
        pytest.skip("node not installed")

    # 1. Start from uncalibrated CG4D (80 flat panels arranged around a cylinder)
    nominal_payload = build_rainbow_zone_payload(
        replay_h5=None, images_h5=None, instrument="CG4D"
    )
    html = render_rainbow_html(nominal_payload)

    # 2. Build a calibrated CG4D solve payload with shifted flat panels (~25 mm / ~20 px)
    # and active frame_index=2 with distinct per-run sample_offset_lab.
    cal_det_cfgs: dict[str, dict] = {}
    cal_detectors: dict[int, Detector] = {}
    shift_vec = np.array([0.018, -0.012, 0.014], dtype=float)
    for k_str, cfg in nominal_payload["detector_configs"].items():
        bid = int(k_str)
        new_center = (np.asarray(cfg["center"], dtype=float) + shift_vec).tolist()
        cal_det_cfgs[str(bid)] = {
            "panel_type": cfg.get("panel_type", "flat"),
            "m": int(cfg["m"]),
            "n": int(cfg["n"]),
            "width": float(cfg["width"]),
            "height": float(cfg["height"]),
            "center": list(new_center),
            "uhat": list(cfg["uhat"]),
            "vhat": list(cfg["vhat"]),
        }
        cal_detectors[bid] = Detector(
            {
                "panel": cfg.get("panel_type", "flat"),
                "m": int(cfg["m"]),
                "n": int(cfg["n"]),
                "width": float(cfg["width"]),
                "height": float(cfg["height"]),
                "center": list(new_center),
                "uhat": list(cfg["uhat"]),
                "vhat": list(cfg["vhat"]),
            }
        )

    ang = np.deg2rad(17.0)
    R_active = np.array(
        [
            [np.cos(ang), 0.0, np.sin(ang)],
            [0.0, 1.0, 0.0],
            [-np.sin(ang), 0.0, np.cos(ang)],
        ],
        dtype=float,
    )
    ub = np.array(
        [
            [0.082, 0.014, -0.009],
            [-0.011, 0.076, 0.018],
            [0.012, -0.015, 0.069],
        ],
        dtype=float,
    )
    ki_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    s_lab_run0 = np.array([-0.004, -0.005, 0.003], dtype=float)
    s_lab_run2 = np.array([0.0035, 0.0045, -0.0025], dtype=float)

    _, zones_payload = _compute_zones_payload_for_geometry(
        np.zeros((0, 3), dtype=float),
        ub_matrix=ub,
        R_gonio=R_active,
        R_single=R_active,
        ki_hat=ki_hat,
        detectors=cal_detectors,
        sample_offset_lab=s_lab_run2,
        max_zone_index=2,
    )
    panels_payload, _ = _encode_panel_textures(
        cal_detectors, None, None, None, None, None, encode_images=False
    )

    # Synthesize predicted peaks spanning ZOLZ (N=0) and HOLZ (N=±1, ±2) for zone 0
    u0, v0, w0 = zones_payload[0]["uvw"]
    synth_peaks = []
    pid = 0
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l_idx in range(-3, 4):
                if h == 0 and k == 0 and l_idx == 0:
                    continue
                layer_n = h * u0 + k * v0 + l_idx * w0
                if abs(layer_n) > 2:
                    continue
                synth_peaks.append(
                    {
                        "id": pid,
                        "hkl": [h, k, l_idx],
                        "det_x": float((pid % 40) - 20.0),
                        "det_y": 0.02,
                        "det_el": 3.0,
                        "wavelength": 3.2,
                        "d_spacing": 2.5,
                        "zone_ids": [0] if layer_n == 0 else [],
                    }
                )
                pid += 1

    cal_payload = dict(nominal_payload)
    cal_payload.update(
        {
            "artifact_kind": "solve",
            "instrument": "CG4D",
            "ki": [0.0, 0.0, 1.0],
            "wavelength_range": [2.8, 4.5],
            "d_min": 1.8,
            "frame_index": 2,
            "frames": [
                {"index": 0, "label": "Run 0"},
                {"index": 1, "label": "Run 1"},
                {"index": 2, "label": "Run 2"},
            ],
            "goniometer_R": R_active.tolist(),
            "goniometer_R_per_run": {
                "0": np.eye(3).tolist(),
                "1": np.eye(3).tolist(),
                "2": R_active.tolist(),
            },
            "sample_offset_lab": s_lab_run2.tolist(),
            "sample_offset_lab_per_run": {
                "0": s_lab_run0.tolist(),
                "1": [0.0, 0.0, 0.0],
                "2": s_lab_run2.tolist(),
            },
            "ub_matrix": ub.tolist(),
            "u_matrix": (ub / np.linalg.norm(ub, axis=0, keepdims=True)).tolist(),
            "detector_configs": cal_det_cfgs,
            "panels": panels_payload,
            "zones": zones_payload,
            "peaks": synth_peaks,
            "predicted_peaks": synth_peaks,
            "sample_peaks": synth_peaks,
            "predicted_sample_peaks": synth_peaks,
        }
    )

    cal_json = tmp_path / "cal_cg4d.json"
    cal_json.write_text(json.dumps(cal_payload), encoding="utf-8")

    scripts = re.findall(r"<script[^>]*>([\s\S]*?)</script>", html)
    main_js = max(scripts, key=len)
    idx_close = main_js.rfind("})();")
    assert idx_close > 0

    probe = f"""
const fs = require("fs");
// Prime pre-indexing flat-panel raycast cache on nominal CG4D (80 banks)
const _preNodals = getVisibleNodalPoints();
const calPay = JSON.parse(fs.readFileSync({json.dumps(str(cal_json))}, "utf8"));
recomputeBoundsAndReload(calPay, {{}});
state.holzOrders = new Set([1, 2]);
const nodalsAfter = getVisibleNodalPoints().filter(n => n.onPanel);

function segDistDeg(px, py, segs) {{
  let best = 1e9;
  for (const seg of (segs || [])) {{
    for (let i = 0; i + 1 < seg.length; i++) {{
      const [x1, y1] = seg[i], [x2, y2] = seg[i + 1];
      const sy = 161.4;
      const dx = x2 - x1, dy = (y2 - y1) * sy;
      const l2 = dx * dx + dy * dy;
      const t = l2 > 1e-12 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * sy * dy) / l2)) : 0;
      const qx = x1 + t * dx, qy = y1 + t * (y2 - y1);
      const d = Math.hypot(px - qx, (py - qy) * sy);
      if (d < best) best = d;
    }}
  }}
  return best;
}}

let maxErrDeg = 0.0;
let checkedPairs = 0;
for (const nd of nodalsAfter) {{
  for (const zid of nd.zoneIds) {{
    const z = ZONES.find(zz => zz.zone_id === zid);
    if (!z || !Array.isArray(z.detector_segments) || z.detector_segments.length === 0) continue;
    const d = segDistDeg(nd.det_x, nd.det_y, z.detector_segments);
    if (d < 5.0) {{
      if (d > maxErrDeg) maxErrDeg = d;
      checkedPairs++;
    }}
  }}
}}

const holzCounts = getZoneHolzLayerCountMap(0);
console.log(JSON.stringify({{
  frameIdx: DATA.frame_index,
  selFrameIdx: state.selectedFrameIndex,
  onPanelCount: nodalsAfter.length,
  checkedPairs,
  maxErrDeg,
  holzPos1: holzCounts[1] || 0,
  holzNeg1: holzCounts[-1] || 0,
  holzPos2: holzCounts[2] || 0,
}}));
"""
    stub = """
const noop = () => ({
  style: { setProperty(){}, removeProperty(){} },
  dataset: {}, options: [], children: [], childNodes: [],
  classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
  addEventListener(){}, removeEventListener(){}, setAttribute(){}, getAttribute(){ return null; },
  getContext(){ return {
    save(){}, restore(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, fill(){}, arc(){}, rect(){}, roundRect(){},
    clearRect(){}, fillRect(){}, strokeRect(){}, fillText(){}, setLineDash(){}, clip(){}, closePath(){},
    measureText(t){ return { width: String(t).length * 6 }; },
    createLinearGradient(){ return { addColorStop(){} }; },
    drawImage(){}, setTransform(){}, scale(){}, translate(){}
  }; },
  getBoundingClientRect(){ return { left: 0, top: 0, width: 1400, height: 800 }; },
  appendChild(){}, removeChild(){}, innerHTML: "", textContent: "", value: "0", checked: false,
  width: 1400, height: 800, clientWidth: 1400, clientHeight: 800, offsetWidth: 1400, offsetHeight: 800,
  querySelector(){ return noop(); }, querySelectorAll(){ return []; }
});
global.requestAnimationFrame = () => 1;
global.cancelAnimationFrame = () => {};
global.window = {
  innerWidth: 1600, innerHeight: 900, devicePixelRatio: 1,
  addEventListener(){}, removeEventListener(){},
  requestAnimationFrame: global.requestAnimationFrame, cancelAnimationFrame: global.cancelAnimationFrame,
  location: { search: "", href: "http://localhost:8080/", pathname: "/", protocol: "http:", host: "localhost:8080" },
  parent: null, localStorage: { getItem(){ return null; }, setItem(){} },
};
global.window.parent = global.window;
global.self = global.window;
global.document = {
  getElementById(){ return noop(); }, createElement(){ return noop(); },
  querySelector(){ return noop(); }, querySelectorAll(){ return []; },
  body: noop(), documentElement: noop(), addEventListener(){},
};
global.navigator = { hardwareConcurrency: 4 };
global.ResizeObserver = class { observe(){} disconnect(){} };
global.Image = class { set src(_v){} };
global.fetch = async () => ({ ok: false, status: 404, json: async () => ({}) });
global.performance = { now: () => 0 };
"""
    js_file = tmp_path / "cg4d_nodal_probe.js"
    js_file.write_text(
        stub + "\n" + main_js[:idx_close] + probe + main_js[idx_close:],
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["node", str(js_file)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["frameIdx"] == 2
    assert out["selFrameIdx"] == 2
    assert out["onPanelCount"] >= 4
    assert out["checkedPairs"] >= 8
    assert out["maxErrDeg"] < 0.03
    assert out["holzPos1"] > 0
    assert out["holzNeg1"] > 0
    assert out["holzPos2"] > 0


def test_nova_app_tof_route_and_fast_frame_switch(tmp_path):
    """Verify MainApp POST /api/tof_delta_theta (no 405) and fast frame_only=1 switching parity."""
    import asyncio
    import importlib.util
    import json
    import re
    import shutil
    import subprocess
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import h5py
    import numpy as np

    from subhkl.config import beamlines
    from subhkl.core.crystallography import cartesian_matrix_metric_tensor
    from subhkl.instrument.detector import Detector
    from subhkl.io.events import ANGSTROM_M_PER_US, write_events
    from subhkl.viz.nova_rainbow_view import RainbowZonesViewModel
    from subhkl.viz.rainbow_zones import (
        build_frame_only_response,
        build_rainbow_zone_payload,
        merge_frame_only_payload,
        render_rainbow_html,
    )

    cfg32 = dict(beamlines["MANDI"]["32"])
    det32 = Detector(cfg32)
    H, W = int(det32.n), int(det32.m)
    ki_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    l1_true = 30.0
    xyz_c = np.asarray(det32.pixel_to_lab(128.0, 128.0), dtype=float).ravel()
    l2_c = float(np.linalg.norm(xyz_c))
    kf_c = xyz_c / l2_c
    q_vec = kf_c - ki_hat
    q_norm = float(np.linalg.norm(q_vec))
    q_hat = q_vec / q_norm
    z_ax = q_hat
    x_ax = np.cross(np.array([0.0, 1.0, 0.0]), z_ax)
    x_ax /= float(np.linalg.norm(x_ax))
    y_ax = np.cross(z_ax, x_ax)
    u_mat = np.column_stack([x_ax, y_ax, z_ax])
    a_cell = 18.0
    cell_c = [a_cell, a_cell, a_cell, 90.0, 90.0, 90.0]
    b_mat, _ = cartesian_matrix_metric_tensor(*cell_c[:3], *np.deg2rad(cell_c[3:]))
    lam_2 = (a_cell / 2.0) * q_norm
    t_pred_2 = lam_2 * (l1_true + l2_c) / ANGSTROM_M_PER_US

    def _make_bank_events(center_r: int, center_c: int):
        lists = [[] for _ in range(H * W)]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                p_flat = (center_r + dr) * W + (center_c + dc)
                lists[p_flat] = [
                    float(t_pred_2 - 2.0),
                    float(t_pred_2),
                    float(t_pred_2 + 2.0),
                ]
        img = np.zeros(H * W, dtype=np.float32)
        flat: list[float] = []
        for idx_p, ev_l in enumerate(lists):
            if ev_l:
                flat.extend(ev_l)
                img[idx_p] = float(len(ev_l))
        return img.reshape(H, W), np.asarray(flat, dtype=np.float32)

    img_f0, tof_f0 = _make_bank_events(128, 128)
    img_f1, tof_f1 = _make_bank_events(132, 124)
    rot_f1 = np.array(
        [
            [np.cos(0.08), -np.sin(0.08), 0.0],
            [np.sin(0.08), np.cos(0.08), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    R_two = np.stack([np.eye(3), rot_f1], axis=0)

    merged_h5 = tmp_path / "mandi_2run_merged.h5"
    with h5py.File(merged_h5, "w") as mf:
        mf.attrs["instrument"] = "MANDI"
        mf.create_dataset("images", data=np.stack([img_f0, img_f1], axis=0))
        mf.create_dataset("bank_ids", data=np.array([32, 32], dtype=np.int32))
        mf.create_dataset("file_offsets", data=np.array([0, 1], dtype=np.int32))
        mf.create_dataset(
            "files",
            data=np.array(["MANDI_run0.nxs.h5", "MANDI_run1.nxs.h5"], dtype="S32"),
        )
        mf.create_group("goniometer").create_dataset("R", data=R_two)
        inst_g = mf.create_group("instrument")
        inst_g.create_dataset("wavelength", data=np.array([1.8, 4.2], dtype=float))
        write_events(mf, [tof_f0, tof_f1], l1=l1_true)

    integ_h5 = tmp_path / "mandi_2run_integrated.h5"
    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.attrs["source"] = str(merged_h5)
        hf.create_dataset("file_offsets", data=np.array([0, 1], dtype=np.int32))
        hf.create_dataset(
            "files",
            data=np.array(["MANDI_run0.nxs.h5", "MANDI_run1.nxs.h5"], dtype="S32"),
        )
        smp = hf.create_group("sample")
        smp.create_dataset("space_group", data=b"P 1")
        smp.create_dataset("cell", data=np.array(cell_c, dtype=float))
        for k_n, v_n in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell_c):
            smp.create_dataset(k_n, data=float(v_n))
        smp.create_dataset("U", data=u_mat)
        smp.create_dataset("B", data=b_mat)
        hf.create_group("goniometer").create_dataset("R", data=R_two)
        ig = hf.create_group("instrument")
        ig.create_dataset("wavelength", data=np.array([1.8, 4.2], dtype=float))
        ig.create_dataset("L1", data=float(l1_true))
        pk = hf.create_group("peaks")
        pk.create_dataset("run_index", data=np.array([0, 0, 1, 1], dtype=np.int32))
        pk.create_dataset("bank", data=np.array([32, 32, 32, 32], dtype=np.int32))
        pk.create_dataset(
            "pixel_r", data=np.array([128.0, 140.0, 132.0, 144.0], dtype=np.float64)
        )
        pk.create_dataset(
            "pixel_c", data=np.array([128.0, 140.0, 124.0, 136.0], dtype=np.float64)
        )
        pk.create_dataset("h", data=np.array([0, 0, 0, 1], dtype=np.int32))
        pk.create_dataset("k", data=np.array([0, 0, 0, 0], dtype=np.int32))
        pk.create_dataset("l", data=np.array([2, 2, 2, 2], dtype=np.int32))
        pk.create_dataset(
            "intensity", data=np.array([220.0, 90.0, 185.0, 110.0], dtype=np.float64)
        )
        pk.create_dataset(
            "sigma", data=np.array([12.0, 9.0, 11.0, 8.5], dtype=np.float64)
        )
        pk.create_dataset(
            "wavelength",
            data=np.array([2.8, 3.1, 2.8, 3.3], dtype=np.float64),
        )
        pk.create_dataset(
            "var_u", data=np.array([1.4, 1.2, 1.5, 1.1], dtype=np.float64)
        )
        pk.create_dataset(
            "var_v", data=np.array([0.8, 0.9, 0.7, 0.85], dtype=np.float64)
        )
        pk.create_dataset(
            "cov_uv", data=np.array([0.1, -0.1, 0.2, 0.0], dtype=np.float64)
        )
        pk.create_dataset(
            "careless_residual", data=np.array([0.2, -0.4, 0.3, -0.1], dtype=np.float64)
        )
        pk.create_dataset(
            "merged_i", data=np.array([200.0, 95.0, 200.0, 112.0], dtype=np.float64)
        )
        cg = hf.create_group("careless")
        cg.create_dataset(
            "metrics_json",
            data=json.dumps(
                {"cc_half": 0.985, "multiplicity": 2.0, "n_merged_asu": 3}
            ).encode("utf-8"),
        )

    # 1. Bit-exact parity between include_all_runs_peaks=False and True on frame 1
    p_init = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        images_h5=merged_h5,
        instrument="MANDI",
        frame_index=0,
        d_min=4.0,
        include_all_runs_peaks=True,
    )
    p_f1_fast = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        images_h5=merged_h5,
        instrument="MANDI",
        frame_index=1,
        d_min=4.0,
        include_all_runs_peaks=False,
    )
    p_f1_full = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        images_h5=merged_h5,
        instrument="MANDI",
        frame_index=1,
        d_min=4.0,
        include_all_runs_peaks=True,
    )
    assert p_f1_fast["integrated_peaks"] == p_f1_full["integrated_peaks"]
    assert p_f1_fast["predicted_peaks"] == p_f1_full["predicted_peaks"]
    assert p_f1_fast["zones"] == p_f1_full["zones"]
    assert p_f1_fast["tof_gate"] == p_f1_full["tof_gate"]
    assert p_f1_fast["sample_peaks"] == []
    merged_srv = merge_frame_only_payload(p_init, dict(p_f1_fast))
    assert merged_srv["sample_peaks"] == p_init["sample_peaks"]
    assert merged_srv["merge_metrics"] == p_init["merge_metrics"]
    assert merged_srv["pixel_mask_bits"] == p_init["pixel_mask_bits"]

    wire_f1 = build_frame_only_response(merged_srv)
    assert wire_f1["frame_only"] is True
    assert "sample_peaks" not in wire_f1
    assert "merge_metrics" not in wire_f1
    assert wire_f1["peaks"] is None
    assert len(wire_f1["integrated_peaks"]) == 2

    # Standalone HTTP ViewModel fast frame reload preserves sample_peaks & merge_metrics
    vm = RainbowZonesViewModel(
        replay_h5=integ_h5,
        images_h5=merged_h5,
        instrument="MANDI",
        d_min=4.0,
    )
    assert len(vm.payload["sample_peaks"]) == 4
    vm_f1 = vm.reload(frame_index=1, include_all_runs_peaks=False)
    assert vm_f1["frame_index"] == 1
    assert len(vm_f1["sample_peaks"]) == 4
    assert vm_f1["merge_metrics"] == p_init["merge_metrics"]

    # 2. MainApp HTTP routes: POST /api/tof_delta_theta (200, not 405!) & GET /api/payload?frame=1&frame_only=1
    if importlib.util.find_spec("aiohttp") is not None:
        from aiohttp import web
        from aiohttp.test_utils import make_mocked_request

        from subhkl.viz.nova_app.models.subhkl_model import (
            InstrumentOptions,
            SubhklGalaxyModel,
        )
        from subhkl.viz.nova_app.views.main_view import MainApp

        sub_m = SubhklGalaxyModel()
        sub_m.data.peaks_file = str(integ_h5)
        sub_m.data.images_file = str(merged_h5)
        sub_m.data.instrument = InstrumentOptions.MANDI
        sub_m.data.d_min = 4.0
        sub_m.refresh_payload(frame_index=0, preserve_params=True)
        assert len(sub_m.payload["sample_peaks"]) == 4
        init_sub_mm = dict(sub_m.payload["merge_metrics"])

        app_inst = object.__new__(MainApp)
        app_inst.pipeline_runner = MagicMock()
        app_inst.pipeline_runner.current_frame_index = 0
        app_inst.view_model = SimpleNamespace(model=SimpleNamespace(subhkl=sub_m))
        aio_app = web.Application()
        app_inst._bind_routes(SimpleNamespace(app=aio_app))

        tof_req_body = json.dumps(
            {
                "frame_index": 0,
                "d_min": 5.5,
                "wavelength_range": [1.8, 4.2],
                "ub_matrix": p_init["ub_matrix"],
                "R_run": p_init["goniometer_R"],
                "instrument": "MANDI",
                "space_group": "P 1",
                "cell": cell_c,
                "tof_l1": l1_true,
            }
        ).encode("utf-8")

        async def _exercise_routes():
            req_tof = make_mocked_request(
                "POST",
                "/api/tof_delta_theta",
                headers={"Content-Type": "application/json"},
                app=aio_app,
            )
            req_tof._read_bytes = tof_req_body
            resp_tof = await aio_app._handle(req_tof)
            req_fo = make_mocked_request(
                "GET",
                "/api/payload?frame=1&frame_only=1",
                app=aio_app,
            )
            resp_fo = await aio_app._handle(req_fo)
            return resp_tof, resp_fo

        resp_tof, resp_fo = asyncio.run(_exercise_routes())
        assert resp_tof.status == 200
        tof_json = json.loads(resp_tof.text)
        assert tof_json["ok"] is True
        assert tof_json["angular_error_mode"] == "tof_3d"

        assert resp_fo.status == 200
        fo_json = json.loads(resp_fo.text)
        assert fo_json["frame_only"] is True
        assert fo_json["frame_index"] == 1
        assert "sample_peaks" not in fo_json
        assert len(sub_m.payload["sample_peaks"]) == 4
        assert sub_m.payload["merge_metrics"] == init_sub_mm

    # 3. Client JS frame switch with frame_only=1 preserves SAMPLE_PEAKS & merge_metrics
    if shutil.which("node"):
        html_doc = render_rainbow_html(p_init)
        scripts = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
        js_src = scripts[-1]
        iife_end = js_src.rfind("})();")
        assert iife_end > 0
        wire_f1_json = json.dumps(wire_f1)
        js_harness = """
const elems = new Map();
function makeEl(id = "") {
  const listeners = {};
  return {
    id, tagName: "DIV", value: "", min: "0.05", max: "0.95", step: "0.01",
    checked: false, disabled: false, textContent: "", innerHTML: "", title: "",
    width: 800, height: 520, clientWidth: 800, clientHeight: 520,
    scrollWidth: 800, scrollHeight: 520, scrollTop: 0, offsetWidth: 800, offsetHeight: 36,
    files: [], options: [], children: [], parentElement: null, dataset: {},
    style: new Proxy({}, {
      get(t, k) {
        if (k === "setProperty") return (p, v) => { t[p] = v; };
        if (k === "getPropertyValue") return (p) => t[p] || "";
        return t[k] ?? "";
      },
      set(t, k, v) { t[k] = v; return true; }
    }),
    classList: { add() {}, remove() {}, toggle() { return false; }, contains() { return false; } },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    removeEventListener() {},
    dispatchEvent(evObj) { (listeners[evObj.type] || []).forEach(fn => fn(evObj)); },
    appendChild(c) { return c; }, insertBefore(c) { return c; }, replaceChildren() {},
    setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
    getBoundingClientRect() { return { left: 0, top: 0, right: 800, bottom: 520, width: 800, height: 520 }; },
    getContext() {
      return {
        fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "left",
        textBaseline: "alphabetic", globalAlpha: 1, imageSmoothingEnabled: true,
        setTransform() {}, resetTransform() {}, scale() {}, translate() {}, rotate() {},
        transform() {}, clearRect() {}, fillRect() {}, strokeRect() {},
        beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, arc() {}, ellipse() {},
        quadraticCurveTo() {}, bezierCurveTo() {}, rect() {}, roundRect() {},
        stroke() {}, fill() {}, clip() {}, save() {}, restore() {}, setLineDash() {},
        fillText() {}, strokeText() {}, drawImage() {},
        measureText(t) { return { width: String(t).length * 6 }; },
        createLinearGradient() { return { addColorStop() {} }; },
        createRadialGradient() { return { addColorStop() {} }; },
        getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
        putImageData() {}, createImageData(w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
      };
    },
    querySelector() { return makeEl(); }, querySelectorAll() { return []; }, closest() { return null; },
  };
}
function getEl(id) { if (!elems.has(id)) elems.set(id, makeEl(id)); return elems.get(id); }
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.parent = globalThis;
globalThis.top = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.devicePixelRatio = 1;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.requestAnimationFrame = () => 1;
globalThis.cancelAnimationFrame = () => {};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.Image = class { set src(_v) {} };
globalThis.document = {
  body: makeEl("body"), documentElement: makeEl("html"),
  getElementById: (id) => getEl(id),
  createElement: (tag) => makeEl(tag),
  querySelector: () => makeEl(), querySelectorAll: () => [],
  addEventListener() {}, removeEventListener() {},
};
"""
        js_probe = f"""
const seenUrls = [];
globalThis.fetch = (url, _opts) => {{
  seenUrls.push(String(url));
  if (String(url).includes("frame_only=1")) {{
    return Promise.resolve({{ ok: true, json: () => Promise.resolve({wire_f1_json}) }});
  }}
  return Promise.resolve({{ ok: true, json: () => Promise.resolve({{ ok: false }}) }});
}};
selectFrame(1);
setTimeout(() => {{
  console.log(JSON.stringify({{
    seenUrls,
    stateFrame: state.selectedFrameIndex,
    dataFrame: DATA.frame_index,
    samplePeaksLen: SAMPLE_PEAKS.length,
    integSampleLen: INTEGRATED_SAMPLE_PEAKS.length,
    peaksRows: PEAKS.filter(p => p.peak_source === "integrated").map(p => p.row),
    hasMergeMetrics: !!(DATA.merge_metrics && DATA.merge_metrics.cc_half),
    hasPixelMaskBits: !!DATA.pixel_mask_bits,
  }}));
}}, 20);
}})();
"""
        js_path = tmp_path / "fast_frame_switch_ui_test.js"
        js_path.write_text(
            js_harness + "\n" + js_src[:iife_end] + "\n" + js_probe, encoding="utf-8"
        )
        proc = subprocess.run(
            ["node", str(js_path)], check=True, capture_output=True, text=True
        )
        js_out = json.loads(proc.stdout.strip())
        assert any("frame=1" in u and "frame_only=1" in u for u in js_out["seenUrls"])
        assert js_out["stateFrame"] == 1
        assert js_out["dataFrame"] == 1
        assert js_out["samplePeaksLen"] == 4
        assert js_out["integSampleLen"] == 4
        assert js_out["peaksRows"] == [132.0, 144.0]
        assert js_out["hasMergeMetrics"] is True
        assert js_out["hasPixelMaskBits"] is True


def test_post_integration_wulff_dmin_filter_preserves_integ_and_all_runs(
    tmp_path,
) -> None:
    """Verify changing d_min after integration preserves Integ/Pred mode and filters all runs on the Wulff map without collapsing to 1 frame or inventing unintegrated peaks."""
    import json
    import re
    import shutil
    import subprocess

    from scipy.spatial.transform import Rotation

    from subhkl.config import beamlines
    from subhkl.core.crystallography import (
        cartesian_matrix_metric_tensor,
        generate_reflections,
    )
    from subhkl.instrument.detector import Detector
    from subhkl.viz.rainbow_zones import (
        _raycast_hkl_on_detectors,
        build_rainbow_zone_payload,
        render_rainbow_html,
    )

    cell = [42.0, 48.0, 56.0, 90.0, 90.0, 90.0]
    B_raw, _ = cartesian_matrix_metric_tensor(*cell[:3], *np.deg2rad(cell[3:]))
    B_mat = np.asarray(B_raw, dtype=float)
    U_mat = Rotation.from_euler("zyx", [14.0, -23.0, 31.0], degrees=True).as_matrix()
    active_bks = [1, 20]
    dets = {bk: Detector(dict(beamlines["MANDI"][str(bk)])) for bk in active_bks}
    H, W = int(dets[1].n), int(dets[1].m)
    wl_range = (2.0, 4.2)
    d_integ_cut = 2.0

    # Build 3 distinct goniometer orientations (runs 0, 1, 2), 2 banks per run = 6 images.
    # Intentionally omit `file_offsets` on `integrated.h5` so standalone run inference from
    # `(peaks/run_index, peaks/image_index)` is also verified!
    run_angles_deg = [0.0, 35.0, 70.0]
    R_runs = [
        Rotation.from_euler("y", ang, degrees=True).as_matrix()
        for ang in run_angles_deg
    ]
    R_per_image = np.stack([R_r for R_r in R_runs for _ in active_bks], axis=0)
    angles_per_image = np.array(
        [[ang, 0.0, 0.0] for ang in run_angles_deg for _ in active_bks],
        dtype=np.float64,
    )
    bank_ids_per_image = np.array(active_bks * len(run_angles_deg), dtype=np.int32)

    hkls_int = np.stack(
        generate_reflections(*cell, space_group="P 21 21 21", d_min=d_integ_cut),
        axis=1,
    ).astype(int)

    all_xyz, all_r, all_c, all_img, all_run = [], [], [], [], []
    all_h, all_k, all_l, all_wl, all_I, all_sig = [], [], [], [], [], []
    bw = int(0.06 * min(H, W))
    for r_idx, R_r in enumerate(R_runs):
        cast_r = _raycast_hkl_on_detectors(
            hkls_int,
            U_mat,
            B_mat,
            R_r,
            dets,
            wl_range,
            active_bids=set(active_bks),
            margin_frac=0.0,
        )
        assert len(cast_r["bank"]) >= 12
        for j in range(len(cast_r["bank"])):
            pr_j = float(cast_r["pixel_r"][j])
            pc_j = float(cast_r["pixel_c"][j])
            if pr_j < bw or pr_j >= (H - 1 - bw) or pc_j < bw or pc_j >= (W - 1 - bw):
                continue
            bk = int(cast_r["bank"][j])
            local_b = active_bks.index(bk)
            img_idx = r_idx * len(active_bks) + local_b
            all_xyz.append(cast_r["peaks_xyz"][j])
            all_r.append(pr_j)
            all_c.append(pc_j)
            all_img.append(img_idx)
            all_run.append(r_idx)
            all_h.append(int(cast_r["hkl"][j, 0]))
            all_k.append(int(cast_r["hkl"][j, 1]))
            all_l.append(int(cast_r["hkl"][j, 2]))
            all_wl.append(float(cast_r["wavelength"][j]))
            all_I.append(120.0)
            all_sig.append(8.0)  # I/sigma = 15.0 >= 2.5 SNR cut

    integ_h5 = tmp_path / "standalone_multirun_integrated.h5"
    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "MANDI"
        hf.create_dataset("bank_ids", data=bank_ids_per_image)
        hf.create_dataset(
            "files",
            data=np.array(
                [f"MANDI_run_{i}.nxs.h5" for i in range(len(run_angles_deg))],
                dtype="S32",
            ),
        )
        gon = hf.create_group("goniometer")
        gon.create_dataset("R", data=R_per_image)
        gon.create_dataset("angles", data=angles_per_image)
        gon.create_dataset(
            "axes",
            data=np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.float64),
        )
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array(wl_range, dtype=np.float64)
        )
        smp = hf.create_group("sample")
        smp.create_dataset("space_group", data=b"P 21 21 21")
        smp.create_dataset("cell", data=np.array(cell, dtype=np.float64))
        for k_c, v_c in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            smp.create_dataset(k_c, data=float(v_c))
        smp.create_dataset("U", data=U_mat)
        smp.create_dataset("B", data=B_mat)
        pgrp = hf.create_group("peaks")
        pgrp.create_dataset("xyz", data=np.asarray(all_xyz, dtype=np.float64))
        pgrp.create_dataset("pixel_r", data=np.asarray(all_r, dtype=np.float64))
        pgrp.create_dataset("pixel_c", data=np.asarray(all_c, dtype=np.float64))
        pgrp.create_dataset("image_index", data=np.asarray(all_img, dtype=np.int32))
        pgrp.create_dataset("run_index", data=np.asarray(all_run, dtype=np.int32))
        pgrp.create_dataset("h", data=np.asarray(all_h, dtype=np.int32))
        pgrp.create_dataset("k", data=np.asarray(all_k, dtype=np.int32))
        pgrp.create_dataset("l", data=np.asarray(all_l, dtype=np.int32))
        pgrp.create_dataset("wavelength", data=np.asarray(all_wl, dtype=np.float64))
        pgrp.create_dataset("intensity", data=np.asarray(all_I, dtype=np.float64))
        pgrp.create_dataset("sigma", data=np.asarray(all_sig, dtype=np.float64))
        n_tot = len(all_h)
        pgrp.create_dataset("var_u", data=np.full(n_tot, 2.5, dtype=np.float64))
        pgrp.create_dataset("var_v", data=np.full(n_tot, 2.5, dtype=np.float64))
        pgrp.create_dataset("cov_uv", data=np.zeros(n_tot, dtype=np.float64))

    payload = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        instrument="MANDI",
        d_min=d_integ_cut,
        wavelength_range=wl_range,
        frame_index=0,
        include_panel_images=False,
    )
    assert payload["default_peak_source"] == "integrated"
    assert len(payload["frames"]) == 3
    assert {int(p["run_index"]) for p in payload["integrated_sample_peaks"]} == {
        0,
        1,
        2,
    }
    assert {int(p["run_index"]) for p in payload["predicted_sample_peaks"]} == {0, 1, 2}
    assert all(
        float(p["d_spacing"]) >= d_integ_cut - 1e-3
        for p in payload["integrated_sample_peaks"]
    )

    # Frame-only payload for run 1 (mirrors /api/payload?frame=1&frame_only=1)
    payload_f1 = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        instrument="MANDI",
        d_min=d_integ_cut,
        wavelength_range=wl_range,
        frame_index=1,
        include_panel_images=True,
        include_all_runs_peaks=False,
    )
    payload_f1["frame_only"] = True
    assert payload_f1["integrated_sample_peaks"] == []
    assert payload_f1["predicted_sample_peaks"] == []

    if not shutil.which("node"):
        return

    html_doc = render_rainbow_html(payload)
    js_src = re.findall(r"<script>([\s\S]*?)</script>", html_doc)[-1]
    iife_end = js_src.rfind("})();")
    assert iife_end > 0

    js_harness = """
const elems = new Map();
function makeEl(id = "") {
  const listeners = {};
  return {
    id, tagName: "DIV", value: "", min: "0.25", max: "6.50", step: "0.05",
    checked: false, disabled: false, textContent: "", innerHTML: "", title: "",
    width: 800, height: 520, clientWidth: 800, clientHeight: 520,
    scrollWidth: 800, scrollHeight: 520, scrollTop: 0, offsetWidth: 800, offsetHeight: 36,
    files: [], options: [], children: [], parentElement: null, dataset: {},
    style: new Proxy({}, {
      get(t, k) {
        if (k === "setProperty") return (p, v) => { t[p] = v; };
        if (k === "getPropertyValue") return (p) => t[p] || "";
        return t[k] ?? "";
      },
      set(t, k, v) { t[k] = v; return true; }
    }),
    classList: { add() {}, remove() {}, toggle() { return false; }, contains() { return false; } },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    removeEventListener() {},
    dispatchEvent(evObj) { (listeners[evObj.type] || []).forEach(fn => fn(evObj)); },
    appendChild(c) { return c; }, insertBefore(c) { return c; }, replaceChildren() {},
    setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
    getBoundingClientRect() { return { left: 0, top: 0, right: 800, bottom: 520, width: 800, height: 520 }; },
    getContext() {
      return {
        fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "left",
        textBaseline: "alphabetic", globalAlpha: 1, imageSmoothingEnabled: true,
        setTransform() {}, resetTransform() {}, scale() {}, translate() {}, rotate() {},
        transform() {}, clearRect() {}, fillRect() {}, strokeRect() {},
        beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, arc() {}, ellipse() {},
        quadraticCurveTo() {}, bezierCurveTo() {}, rect() {}, roundRect() {},
        stroke() {}, fill() {}, clip() {}, save() {}, restore() {}, setLineDash() {},
        fillText() {}, strokeText() {}, drawImage() {},
        measureText(t) { return { width: String(t).length * 6 }; },
        createLinearGradient() { return { addColorStop() {} }; },
        createRadialGradient() { return { addColorStop() {} }; },
        getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
        putImageData() {}, createImageData(w, h) { return { data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }; },
      };
    },
    querySelector() { return makeEl(); }, querySelectorAll() { return []; }, closest() { return null; },
  };
}
function getEl(id) { if (!elems.has(id)) elems.set(id, makeEl(id)); return elems.get(id); }
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.parent = globalThis;
globalThis.top = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.devicePixelRatio = 1;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.requestAnimationFrame = () => 1;
globalThis.cancelAnimationFrame = () => {};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.Image = class { set src(_v) {} };
globalThis.document = {
  body: makeEl("body"), documentElement: makeEl("html"),
  getElementById: (id) => getEl(id),
  createElement: (tag) => makeEl(tag),
  querySelector: () => makeEl(), querySelectorAll: () => [],
  addEventListener() {}, removeEventListener() {},
};
"""
    js_probe = f"""
const slDminEl = getEl("slider-dmin");
state.wulffScope = "all_runs";
applyPeakSource("integrated", false);
const initVis = getActiveWulffPeaks();
const initState = {{
  peakSource: state.peakSource,
  wulffScope: state.wulffScope,
  count: initVis.length,
  runs: Array.from(new Set(initVis.map(p => p.run_index))).sort(),
  minD: initVis.reduce((m, p) => Math.min(m, p.d_spacing), Infinity),
}};

// 1. Raise d_min above integrated cutoff (2.00 -> 2.80 A) in Integ + All-Runs:
// must stay in 'integrated' + 'all_runs', filter all runs to d >= 2.80 A, and keep all 3 runs!
slDminEl.value = "2.80";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
const highDVis = getActiveWulffPeaks();
const highDState = {{
  peakSource: state.peakSource,
  wulffScope: state.wulffScope,
  count: highDVis.length,
  runs: Array.from(new Set(highDVis.map(p => p.run_index))).sort(),
  minD: highDVis.reduce((m, p) => Math.min(m, p.d_spacing), Infinity),
}};

// 2. Lower d_min below integrated cutoff (2.80 -> 1.45 A) in Integ + All-Runs:
// must NEVER switch to 'predicted', must NEVER add unintegrated (< 2.0 A) peaks, and must keep all 3 runs!
slDminEl.value = "1.45";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
if (_dynMultiRunTimer) {{
  clearTimeout(_dynMultiRunTimer);
  _dynMultiRunTimer = null;
  if (typeof _dynPendingMultiRunFn === "function") _dynPendingMultiRunFn();
}}
const lowDIntegVis = getActiveWulffPeaks();
const lowDIntegState = {{
  peakSource: state.peakSource,
  wulffScope: state.wulffScope,
  count: lowDIntegVis.length,
  runs: Array.from(new Set(lowDIntegVis.map(p => p.run_index))).sort(),
  minD: lowDIntegVis.reduce((m, p) => Math.min(m, p.d_spacing), Infinity),
}};

// 3. Switch active detector frame via frame_only payload (empty sample arrays) and re-filter d_min:
// must preserve multi-run INTEGRATED_SAMPLE_PEAKS & PREDICTED_SAMPLE_PEAKS across all 3 runs!
const frameOnlyPayload = {json.dumps(payload_f1)};
recomputeBoundsAndReload(frameOnlyPayload, {{ preserveViewState: true }});
slDminEl.value = "2.40";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
const afterFrameSwitchIntegVis = getActiveWulffPeaks();
const afterFrameSwitchInteg = {{
  peakSource: state.peakSource,
  wulffScope: state.wulffScope,
  count: afterFrameSwitchIntegVis.length,
  runs: Array.from(new Set(afterFrameSwitchIntegVis.map(p => p.run_index))).sort(),
  minD: afterFrameSwitchIntegVis.reduce((m, p) => Math.min(m, p.d_spacing), Infinity),
}};

// 4. Explicitly switch to Pred + All-Runs at d_min = 1.45 A:
// must dynamically predict outer reflections (< 2.0 A) across ALL 3 runs without active frame starving others!
slDminEl.value = "1.45";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
applyPeakSource("predicted", false);
if (_dynMultiRunTimer) {{
  clearTimeout(_dynMultiRunTimer);
  _dynMultiRunTimer = null;
  if (typeof _dynPendingMultiRunFn === "function") _dynPendingMultiRunFn();
}}
const predAllRunsVis = getActiveWulffPeaks();
const predAllRunsState = {{
  peakSource: state.peakSource,
  wulffScope: state.wulffScope,
  count: predAllRunsVis.length,
  runs: Array.from(new Set(predAllRunsVis.map(p => p.run_index))).sort(),
  minD: predAllRunsVis.reduce((m, p) => Math.min(m, p.d_spacing), Infinity),
}};

// Also verify Pred + All-Runs filters cleanly when raising d_min to 2.80 A
slDminEl.value = "2.80";
slDminEl.dispatchEvent({{ type: "input", target: slDminEl }});
if (_dynMultiRunTimer) {{
  clearTimeout(_dynMultiRunTimer);
  _dynMultiRunTimer = null;
  if (typeof _dynPendingMultiRunFn === "function") _dynPendingMultiRunFn();
}}
const predHighDVis = getActiveWulffPeaks();
const predHighDState = {{
  peakSource: state.peakSource,
  count: predHighDVis.length,
  runs: Array.from(new Set(predHighDVis.map(p => p.run_index))).sort(),
  minD: predHighDVis.reduce((m, p) => Math.min(m, p.d_spacing), Infinity),
}};

console.log(JSON.stringify({{
  initState,
  highDState,
  lowDIntegState,
  afterFrameSwitchInteg,
  predAllRunsState,
  predHighDState,
}}));
}})();
"""
    js_path = tmp_path / "post_integ_wulff_dmin_test.js"
    js_path.write_text(
        js_harness + "\n" + js_src[:iife_end] + "\n" + js_probe, encoding="utf-8"
    )
    proc = subprocess.run(
        ["node", str(js_path)], check=True, capture_output=True, text=True
    )
    out = json.loads(proc.stdout.strip())

    assert out["initState"]["peakSource"] == "integrated"
    assert out["initState"]["runs"] == [0, 1, 2]
    assert out["initState"]["minD"] >= 2.0 - 1e-3

    assert out["highDState"]["peakSource"] == "integrated"
    assert out["highDState"]["wulffScope"] == "all_runs"
    assert out["highDState"]["runs"] == [0, 1, 2]
    assert 0 < out["highDState"]["count"] < out["initState"]["count"]
    assert out["highDState"]["minD"] >= 2.80 - 1e-3

    assert out["lowDIntegState"]["peakSource"] == "integrated"
    assert out["lowDIntegState"]["wulffScope"] == "all_runs"
    assert out["lowDIntegState"]["runs"] == [0, 1, 2]
    assert out["lowDIntegState"]["count"] == out["initState"]["count"]
    assert out["lowDIntegState"]["minD"] >= 2.0 - 1e-3

    assert out["afterFrameSwitchInteg"]["peakSource"] == "integrated"
    assert out["afterFrameSwitchInteg"]["runs"] == [0, 1, 2]
    assert out["afterFrameSwitchInteg"]["minD"] >= 2.40 - 1e-3
    assert (
        out["highDState"]["count"]
        < out["afterFrameSwitchInteg"]["count"]
        < out["initState"]["count"]
    )

    assert out["predAllRunsState"]["peakSource"] == "predicted"
    assert out["predAllRunsState"]["runs"] == [0, 1, 2]
    assert out["predAllRunsState"]["count"] > out["initState"]["count"]
    assert out["predAllRunsState"]["minD"] < 1.65

    assert out["predHighDState"]["peakSource"] == "predicted"
    assert out["predHighDState"]["runs"] == [0, 1, 2]
    assert out["predHighDState"]["minD"] >= 2.80 - 1e-3


def test_careless_merge_respects_dmin_and_stamps_extrapolated_asu_lookup(
    tmp_path, monkeypatch
) -> None:
    """Verify Careless merge forwards d_min (subset & Bayesian harmonic extrapolation) in container + Galaxy modes and stamps full merged.mtz ASU lookup."""
    import json
    from unittest.mock import MagicMock, patch

    import gemmi

    from subhkl.core.crystallography import cartesian_matrix_metric_tensor
    from subhkl.viz.nova_app.models.subhkl_model import SubhklGalaxyModel
    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import build_rainbow_zone_payload

    cell = [20.0, 24.0, 28.0, 90.0, 90.0, 90.0]
    B_raw, _ = cartesian_matrix_metric_tensor(*cell[:3], *np.deg2rad(cell[3:]))
    B_mat = np.asarray(B_raw, dtype=float)

    merged_h5 = tmp_path / "merged.h5"
    integ_h5 = tmp_path / "integrated.h5"
    with h5py.File(merged_h5, "w") as mf:
        mf.attrs["instrument"] = "CG4D"
        mf.create_dataset("bank_ids", data=np.array([1], dtype=np.int32))
        mf.create_group("goniometer").create_dataset("R", data=np.eye(3)[None, :, :])
        mf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([1.0, 4.5], dtype=np.float64)
        )

    # Fundamental (2, 0, 0) has d = 10.0 A; harmonic (4, 0, 0) has d = 5.0 A (not in /peaks/*)
    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf.create_dataset("bank_ids", data=np.array([1], dtype=np.int32))
        hf.create_group("goniometer").create_dataset("R", data=np.eye(3)[None, :, :])
        hf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([1.0, 4.5], dtype=np.float64)
        )
        smp = hf.create_group("sample")
        smp.create_dataset("space_group", data=b"P 21 21 21")
        smp.create_dataset("cell", data=np.array(cell, dtype=np.float64))
        for k_c, v_c in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            smp.create_dataset(k_c, data=float(v_c))
        smp.create_dataset("U", data=np.eye(3, dtype=np.float64))
        smp.create_dataset("B", data=B_mat)
        pgrp = hf.create_group("peaks")
        pgrp.create_dataset(
            "xyz", data=np.array([[-0.35, 0.0, 0.25]], dtype=np.float64)
        )
        pgrp.create_dataset("pixel_r", data=np.array([128.0], dtype=np.float64))
        pgrp.create_dataset("pixel_c", data=np.array([128.0], dtype=np.float64))
        pgrp.create_dataset("image_index", data=np.array([0], dtype=np.int32))
        pgrp.create_dataset("run_index", data=np.array([0], dtype=np.int32))
        pgrp.create_dataset("h", data=np.array([2], dtype=np.int32))
        pgrp.create_dataset("k", data=np.array([0], dtype=np.int32))
        pgrp.create_dataset("l", data=np.array([0], dtype=np.int32))
        pgrp.create_dataset("wavelength", data=np.array([3.2], dtype=np.float64))
        pgrp.create_dataset("intensity", data=np.array([240.0], dtype=np.float64))
        pgrp.create_dataset("sigma", data=np.array([12.0], dtype=np.float64))
        pgrp.create_dataset("var_u", data=np.array([2.0], dtype=np.float64))
        pgrp.create_dataset("var_v", data=np.array([2.0], dtype=np.float64))
        pgrp.create_dataset("cov_uv", data=np.array([0.0], dtype=np.float64))

    # Write a minimal unmerged.mtz so merge stage skips Exporter
    mtz_in = gemmi.Mtz(with_base=True)
    mtz_in.spacegroup = gemmi.find_spacegroup_by_name("P 21 21 21")
    mtz_in.set_cell_for_all(gemmi.UnitCell(*cell))
    mtz_in.add_dataset("unmerged")
    mtz_in.add_column("M/ISYM", "Y")
    mtz_in.add_column("BATCH", "B")
    mtz_in.add_column("I", "J")
    mtz_in.add_column("SIGI", "Q")
    mtz_in.add_column("WAVEL", "W")
    mtz_in.set_data(np.array([[2, 0, 0, 1, 1000, 240.0, 12.0, 3.2]], dtype=np.float32))
    mtz_in.write_to_file(str(tmp_path / "integrated.mtz"))

    # 1. Local Podman container stage: verify --dmin=<val> for both subset (d_min=2.8) and extrapolation (d_min=1.45)
    mtz_in.write_to_file(str(tmp_path / "reflections.mtz"))
    for dmin_val in (2.8, 1.45):
        runner = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
        runner._local_work_dir = str(tmp_path)
        rec = {"cancel_requested": True}
        runner._submit_local_subhkl_stage(
            "merge",
            {
                "images_file": str(merged_h5),
                "peaks_file": str(integ_h5),
                "d_min": dmin_val,
            },
            existing_job_record=rec,
        )
        assert f"--dmin={dmin_val:g}" in rec["command_line"]
        prep_src = (tmp_path / "careless_work" / "prep_merge.py").read_text()
        post_src = (tmp_path / "careless_work" / "post_merge.py").read_text()
        assert f"_req_dmin = {dmin_val!r}" in prep_src
        assert "requested_d_min" in post_src

    # 2. SubhklGalaxyModel + Galaxy imaginex_subhkl_merge tool_inputs["dmin"] forwarding
    m_sub = SubhklGalaxyModel()
    m_sub.data.peaks_file = str(integ_h5)
    m_sub.data.images_file = str(merged_h5)
    m_sub.data.d_min = 1.45
    assert m_sub.build_tool_parameters("merge")["dmin"] == 1.45

    shared_dir = tmp_path / "shared_dir"
    shared_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SUBHKL_SHARED_DIR", str(shared_dir))
    monkeypatch.setenv("SUBHKL_EXEC_MODE", "galaxy")
    gx_runner = NovaGalaxyPipelineRunner(
        galaxy_url="http://localhost:8080", galaxy_api_key="fake-key"
    )
    captured_gx: list[tuple[str, dict]] = []
    fake_store = MagicMock()
    fake_store.history_id = "hist_dmin"
    fake_store.nova_connection.galaxy_instance.histories.show_history.return_value = []
    fake_store.nova_connection.galaxy_instance.tools.run_tool.side_effect = (
        lambda history_id, tool_id, tool_inputs: (
            captured_gx.append((tool_id, dict(tool_inputs)))
            or {"jobs": [{"id": "gx_1"}], "outputs": []}
        )
    )
    mtz_in.write_to_file(str(tmp_path / "reflections.mtz"))
    gx_job_rec = {
        "job_id": "j_dmin",
        "stage_id": "merge",
        "state": "queued",
        "stdout": "",
    }
    with (
        patch.object(gx_runner, "get_datastore", return_value=fake_store),
        patch.object(gx_runner, "_has_tool_on_galaxy", return_value=True),
        patch.object(
            gx_runner, "_upload_compact_ndip_artifact", return_value="hda_mtz"
        ),
        patch.object(gx_runner, "_ensure_galaxy_h5_dataset", return_value="hda_mtz"),
        patch.object(gx_runner, "_get_tool_version", return_value="0.1.3"),
        patch.object(gx_runner, "_poll_galaxy_job", return_value=None),
        patch.object(gx_runner, "_submit_local_subhkl_stage", return_value=None),
    ):
        gx_runner._dispatch_nova_galaxy_thread(
            "merge",
            {
                "images_file": str(merged_h5),
                "peaks_file": str(integ_h5),
                "d_min": 1.45,
            },
            gx_job_rec,
            None,
        )
    assert len(captured_gx) == 1, f"Galaxy dispatch state: {gx_job_rec}"
    assert captured_gx[0][0] == "imaginex_subhkl_merge"
    assert captured_gx[0][1]["dmin"] == 1.45

    # 3. _stamp_careless_merge_into_h5 stamps uppercase SIGI + Bayesian-extrapolated outer harmonic (4, 0, 0)
    stamp_dir = tmp_path / "stamp_extrap"
    stamp_dir.mkdir()
    mtz_merged = gemmi.Mtz(with_base=True)
    mtz_merged.spacegroup = gemmi.find_spacegroup_by_name("P 21 21 21")
    mtz_merged.set_cell_for_all(gemmi.UnitCell(*cell))
    mtz_merged.add_dataset("merged")
    mtz_merged.add_column("I", "J")
    mtz_merged.add_column("SIGI", "Q")
    mtz_merged.add_column("N", "I")
    mtz_merged.set_data(
        np.array(
            [
                [2, 0, 0, 100.0, 5.0, 4.0],  # I/SIGI = 20.0
                [4, 0, 0, 36.0, 6.0, 2.0],  # Extrapolated harmonic: I/SIGI = 6.0
            ],
            dtype=np.float32,
        )
    )
    mtz_merged.write_to_file(str(stamp_dir / "merged.mtz"))

    mtz_pred = gemmi.Mtz(with_base=True)
    mtz_pred.spacegroup = gemmi.find_spacegroup_by_name("P 21 21 21")
    mtz_pred.set_cell_for_all(gemmi.UnitCell(*cell))
    mtz_pred.add_dataset("predictions")
    for cname, ctype in (
        ("Iobs", "J"),
        ("SigIobs", "Q"),
        ("Ipred", "J"),
        ("SigIpred", "Q"),
        ("Scale", "R"),
    ):
        mtz_pred.add_column(cname, ctype)
    mtz_pred.set_data(
        np.array([[2, 0, 0, 240.0, 12.0, 235.0, 8.0, 2.35]], dtype=np.float32)
    )
    mtz_pred.write_to_file(str(stamp_dir / "predictions.mtz"))
    (stamp_dir / "careless_metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "mode": "poly",
                "requested_d_min": 4.5,
                "n_merged_asu": 2,
                "n_observations": 1,
                "mean_isigi": 13.0,
                "mean_multiplicity": 3.0,
            }
        ),
        encoding="utf-8",
    )

    runner_stamp = NovaGalaxyPipelineRunner(galaxy_url="", galaxy_api_key="")
    runner_stamp._stamp_careless_merge_into_h5(str(integ_h5), str(stamp_dir), {})

    merged_payload = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        instrument="CG4D",
        d_min=4.5,
        wavelength_range=(1.0, 4.5),
        include_panel_images=False,
    )
    assert merged_payload["artifact_kind"] == "merged_careless"
    assert merged_payload["mtz_asu_lookup"]["2,0,0"] == [100.0, 5.0]
    assert merged_payload["mtz_asu_lookup"]["4,0,0"] == [36.0, 6.0]
    # Real SIGI=5.0 used (mtz_isigi = 100 / 5 = 20.0, not fallback SIGI=1.0 -> 100.0!)
    assert abs(float(merged_payload["integrated_peaks"][0]["mtz_isigi"]) - 20.0) < 1e-3
    extrap_hits = [
        p for p in merged_payload["predicted_peaks"] if p.get("asu_orbit_id") == "4,0,0"
    ]
    for p_ex in extrap_hits:
        assert abs(float(p_ex["mtz_isigi"]) - 6.0) < 1e-3


def test_cg4d_bandpass_single_slider_blue_alpha_overlap_and_snap(tmp_path):
    """Verify CG4D [2.8, 4.5] Å wide default, active Colorscale Crossover in Color: λ/Zone,
    quantile wavelength histogram + overlap visualization on #wl-interval-slider,
    and magnetic [ ] edge snapping onto empirical [3.1, 4.3] Å bandpass."""
    import importlib.resources
    import math
    import re
    import shutil
    import subprocess

    from subhkl.viz.rainbow_zones import (
        _compute_wavelength_bandpass_profile,
        _get_instrument_default_wavelength,
        render_rainbow_html,
    )

    settings_data = json.loads(
        importlib.resources.files("subhkl.resources")
        .joinpath("reduction_settings.json")
        .read_text(encoding="utf-8")
    )
    assert settings_data["CG4D"]["Wavelength"] == [2.8, 4.5]
    assert _get_instrument_default_wavelength("CG4D") == (2.8, 4.5)

    # Synthetic wide-band [2.50, 5.00] Å predicted-peak integration on CG4D where
    # physical neutron bandpass is [3.10, 4.30] Å, contaminated in off-band wings by:
    # - realistic ~5% Gaussian/background positive tails (I/σ ~ 2.5..3.2),
    # - a Laue harmonic overlap (hkl=[2, 4, -6], gcd=2) with I/σ = 185 at λ = 2.62 Å,
    # - a single cosmic-ray outlier with I/σ = 260 at λ = 4.88 Å,
    # AND `integrated_sample_peaks` pre-filtered to I/σ >= 2.5 (matching `all_runs_data`).
    rng = np.random.default_rng(42)
    integ_peaks = []
    lam_list = []
    int_list = []
    sig_list = []
    hkl_list = []
    for k in range(400):
        lam = 2.503 + (k + 0.5) * (2.494 / 400.0)
        in_band = 3.10 <= lam <= 4.30
        hkl = [1, (k % 7) - 3, (k // 7) - 25]
        if hkl[0] == 0 and hkl[1] == 0 and hkl[2] == 0:
            hkl = [1, 1, 1]
        # Ensure fundamental gcd == 1 except for injected harmonic overlaps
        if math.gcd(math.gcd(abs(hkl[0]), abs(hkl[1])), abs(hkl[2])) > 1:
            hkl[0] = 1
        if in_band:
            # Realistic quasi-Laue mixture: mostly weak high-res peaks + strong dome peaking near 3.55 Å
            dome = math.sin(math.pi * (lam - 3.05) / 1.30)
            snr = (
                float(rng.uniform(5.5, 14.0 + 16.0 * max(0.0, dome)))
                if (k % 4 == 0)
                else float(rng.normal(0.45, 1.05))
            )
        else:
            # Off-band wing: unit-normal noise + ~5% positive tail around 2.8σ
            snr = 2.85 if (k % 20 == 3) else float(rng.normal(0.15, 0.95))
        if abs(lam - 2.62) < 0.008:
            # Harmonic multiple copying strong in-band spot onto 2.62 Å
            hkl = [2, 4, -6]
            snr = 185.0
        elif abs(lam - 4.88) < 0.005:
            # Single isolated cosmic-ray outlier at 4.88 Å
            snr = 260.0
        sigma = 10.0
        intensity = snr * sigma
        lam_list.append(lam)
        int_list.append(intensity)
        sig_list.append(sigma)
        hkl_list.append(hkl)
        az = -60.0 + (k % 25) * 4.8
        el = -14.0 + (k // 25) * 1.8
        gcd_ord = math.gcd(math.gcd(abs(hkl[0]), abs(hkl[1])), abs(hkl[2]))
        integ_peaks.append(
            {
                "hkl": hkl,
                "harmonic_order": max(1, gcd_ord),
                "asu_orbit_id": f"{hkl[0]},{hkl[1]},{hkl[2]}",
                "orbit_color_idx": k % 8,
                "wavelength": round(lam, 4),
                "d_spacing": round(max(1.25, 0.55 * lam), 4),
                "az_deg": round(az, 3),
                "el_deg": round(el, 3),
                "det_x": round(az, 3),
                "det_y": round(math.tan(math.radians(el)), 4),
                "zone_ids": ["z0"],
                "g_sample": [0.0, 0.6, 0.8],
                "intensity": round(intensity, 3),
                "sigma": sigma,
                "isigi": round(snr, 3),
                "unscaled_isigi": round(snr, 3),
                "observed": bool(snr >= 3.0),
                "uncertain": False,
                "ellipse": None,
            }
        )

    wl_prof = _compute_wavelength_bandpass_profile(
        np.asarray(lam_list, dtype=float),
        np.asarray(int_list, dtype=float),
        np.asarray(sig_list, dtype=float),
        hkl=np.asarray(hkl_list, dtype=np.int64),
    )
    assert wl_prof is not None
    # Simulate `all_runs_data` SNR>=2.5 truncation on `integrated_sample_peaks`
    truncated_sample_peaks = [p for p in integ_peaks if p["unscaled_isigi"] >= 2.5]

    payload = {
        "instrument": "CG4D",
        "artifact_kind": "integrated",
        "space_group": "P 21 21 21",
        "wavelength_range": [2.5, 5.0],
        "d_min": 1.25,
        "cell": [50.0, 60.0, 70.0, 90.0, 90.0, 90.0],
        "default_projection": "cylindrical",
        "default_peak_source": "integrated",
        "panels": [
            {
                "bank_id": 1,
                "name": "det1",
                "az_min": -170.0,
                "az_max": 170.0,
                "el_min": -25.0,
                "el_max": 25.0,
                "y_min": -0.5,
                "y_max": 0.5,
            }
        ],
        "zones": [
            {
                "zone_id": "z0",
                "label": "[100]",
                "rainbow_color": "#38bdf8",
                "angular_rms_deg": 0.12,
                "n_peaks": 250,
                "d_direct": 50.0,
                "axis_lab": [0.0, 1.0, 0.0],
                "axis_crystal": [1.0, 0.0, 0.0],
                "continuous_segments": [],
                "detector_segments": [],
                "wulff_points": [],
            }
        ],
        "predicted_peaks": integ_peaks,
        "predicted_sample_peaks": integ_peaks,
        "integrated_peaks": integ_peaks,
        "integrated_sample_peaks": truncated_sample_peaks,
        "wavelength_bandpass_profile": wl_prof,
        "ladder_steps": [],
        "pipeline_stages": [],
    }
    html_doc = render_rainbow_html(payload=payload)
    assert 'id="cs-wl-interval-slider"' not in html_doc
    assert 'id="btn-cs-wl-auto"' not in html_doc
    assert 'id="btn-cs-wl-apply"' not in html_doc
    assert 'id="wl-interval-slider"' in html_doc
    assert 'id="canvas-wl-carpet"' in html_doc
    assert 'class="controls-section-divider"' in html_doc

    # Visualization controls [Detector Contrast, Spot Color, Colorscale Crossover]
    # are grouped above the divider, before physical parameters [Wavelength λ, Resolution d_min]
    pos_spot = html_doc.index('id="sel-spot-color"')
    pos_emph = html_doc.index('id="slider-color-emphasis"')
    pos_div = html_doc.index('class="controls-section-divider"')
    pos_wl = html_doc.index('id="wl-interval-slider"')
    pos_dmin = html_doc.index('id="slider-dmin"')
    assert pos_spot < pos_emph < pos_div < pos_wl < pos_dmin

    # Physical .wl-track-fill has no rainbow gradient
    m_wl_css = re.search(r"\.wl-track-fill\s*\{([^}]*)\}", html_doc)
    assert m_wl_css is not None
    assert "hsla(265" not in m_wl_css.group(1)
    assert "rgba(56, 189, 248" in m_wl_css.group(1)

    if not shutil.which("node"):
        return

    script_blocks = re.findall(r"<script>([\s\S]*?)</script>", html_doc)
    assert len(script_blocks) >= 1
    inline_js = script_blocks[-1]

    harness_js = (
        """
const elems = new Map();
const ctxLogs = new Map();
function makeCtx(id = "") {
  const log = { fills: [], strokes: [], rects: [], rectCount: 0, fillCalls: 0 };
  if (id) ctxLogs.set(id, log);
  const ctx = {
    fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "left",
    textBaseline: "alphabetic", globalAlpha: 1, imageSmoothingEnabled: true,
    setTransform() {}, resetTransform() {}, scale() {}, translate() {}, rotate() {},
    transform() {},
    clearRect() { log.fills.length = 0; log.strokes.length = 0; log.rects.length = 0; log.rectCount = 0; },
    fillRect(x, y, w, h) { log.rectCount += 1; const fs = String(ctx.fillStyle); log.fills.push(fs); log.rects.push({ x, y, w, h, fill: fs }); },
    strokeRect() { log.strokes.push(String(ctx.strokeStyle)); },
    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {},
    arc() {}, ellipse() {}, quadraticCurveTo() {}, bezierCurveTo() {}, rect() {}, roundRect() {},
    stroke() { log.strokes.push(String(ctx.strokeStyle)); },
    fill() { log.fillCalls += 1; log.fills.push(String(ctx.fillStyle)); },
    clip() {}, save() {}, restore() {}, setLineDash() {},
    fillText() {}, strokeText() {}, drawImage() {},
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
  const isSliderOrCarpet = id === "wl-interval-slider" || id === "canvas-wl-carpet";
  const el = {
    id, tagName: id.startsWith("sel-") ? "SELECT" : "DIV",
    value: id === "slider-color-emphasis" ? "0.38" : (id === "sel-spot-color" ? "wavelength" : ""),
    min: "0.05", max: "0.95", step: "0.01",
    checked: false, disabled: false, textContent: "", innerHTML: "", title: "",
    width: isSliderOrCarpet ? 300 : 1400, height: isSliderOrCarpet ? 11 : 700,
    clientWidth: isSliderOrCarpet ? 300 : 1400, clientHeight: isSliderOrCarpet ? 11 : 700,
    scrollWidth: 300, scrollHeight: 240, scrollTop: 0, offsetWidth: 300, offsetHeight: 36,
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
    dispatchEvent(evObj) { (listeners[evObj.type] || []).forEach(fn => fn.call(el, evObj)); },
    click() { el.dispatchEvent({ type: "click", target: el, preventDefault() {}, stopPropagation() {} }); },
    focus() {}, blur() {}, select() {}, remove() {},
    appendChild(ch) { this.children.push(ch); ch.parentElement = this; return ch; },
    insertBefore(ch) { this.children.push(ch); ch.parentElement = this; return ch; },
    replaceChildren(...ch) { this.children = ch; },
    setAttribute(k, v) { this[k] = v; },
    getAttribute(k) { return this[k] ?? null; },
    removeAttribute(k) { delete this[k]; },
    contains(target) { return target === el; },
    getBoundingClientRect() {
      return {
        left: 0, top: 0,
        right: isSliderOrCarpet ? 300 : 1400,
        bottom: isSliderOrCarpet ? 11 : 700,
        width: isSliderOrCarpet ? 300 : 1400,
        height: isSliderOrCarpet ? 11 : 700,
      };
    },
    getContext() { if (!el._ctx) el._ctx = makeCtx(id); return el._ctx; },
    querySelector() { return null; },
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
let rafCb = null;
globalThis.requestAnimationFrame = (fn) => { rafCb = fn; return 1; };
globalThis.cancelAnimationFrame = () => {};
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}), text: async () => "" });
globalThis.document = {
  body: makeEl("body"),
  documentElement: makeEl("html"),
  baseURI: "http://localhost:8080/",
  getElementById: (id) => getEl(id),
  createElement: (tag) => makeEl(tag),
  createTextNode: (txt) => ({ textContent: txt }),
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: () => {},
  removeEventListener: () => {},
};
"""
        + inline_js
        + """
;
// Trigger initial animation frame so Detector & Wulff canvases paint
getEl("canvas-detector").getContext("2d");
getEl("canvas-wulff").getContext("2d");
if (rafCb) { const cb = rafCb; rafCb = null; cb(); }

// 1. Color: λ/Zone initial state -> Colorscale Crossover is ACTIVE (not disabled),
// shows the [λ_min, λ_max] rainbow gradient, and adjusts spectral hue midpoint λ ~ X.XX Å
const slEmph = getEl("slider-color-emphasis");
const lblEmph = getEl("lbl-color-emphasis");
const initEmphDisabled = Boolean(slEmph.disabled);
const initWlGrad = slEmph.style.getPropertyValue("--cs-gradient");
const initLblEmph = lblEmph.textContent;
const initWlHasCarpet = getEl("wl-track-fill").classList.contains("has-carpet");

// Drag single knob on Colorscale Crossover -> shifts spectral hue midpoint λ ~ X.XX Å
slEmph.value = "0.70";
slEmph.dispatchEvent({ type: "input", target: slEmph });
const shiftedLblEmph = lblEmph.textContent;

// Wide [2.50, 5.00] Å selection vs [3.10, 4.30] Å integrated band on #canvas-wl-carpet:
// Pure monochrome/sky-blue (zero rainbow hsla(..., 95%, ...) fills on physical Wavelength slider!)
const carpetLog = ctxLogs.get("canvas-wl-carpet");
const wideAnyRainbowOnWl = carpetLog.fills.some(s => s.startsWith("hsla(") && s.includes("95%"));
const wideOverlapCyanCount = carpetLog.fills.filter(s => s.startsWith("rgba(56, 189, 248,")).length;
const wideNoiseSlateCount = carpetLog.fills.filter(s => s === "rgba(148, 163, 184, 0.14)").length;

// 2. Narrow [λ_min, λ_max] to [3.40, 4.00] Å -> unselected integrated WL band [3.10..3.40) and (4.00..4.30]
// renders in cerulean blue outside [ ], while Colorscale Crossover updates its λ midpoint to [3.40, 4.00] Å
const slWlMin = getEl("slider-wl-min");
const slWlMax = getEl("slider-wl-max");
slWlMin.value = "3.40";
slWlMin.dispatchEvent({ type: "input", target: slWlMin });
slWlMax.value = "4.00";
slWlMax.dispatchEvent({ type: "input", target: slWlMax });

const narrowBlueBarsCount = carpetLog.fills.filter(s => s.startsWith("hsla(204.0, 94%, 52%,")).length;
const narrowSkyStrokes = carpetLog.strokes.filter(s => s.startsWith("rgba(56, 189, 248,")).length;
const narrowOverlapCyanCount = carpetLog.fills.filter(s => s.startsWith("rgba(56, 189, 248,")).length;
const narrowAnyRainbowOnWl = carpetLog.fills.some(s => s.startsWith("hsla(") && s.includes("95%"));
const narrowLblEmph = lblEmph.textContent;

// Verify quantile outlier rejection directly on rendered #canvas-wl-carpet bars:
// off-band wings [2.50, 2.98] & [4.42, 5.00] Å draw 0 signal bars despite harmonic/cosmic-ray outliers
const domLo = parseFloat(slWlMin.min);
const domHi = parseFloat(slWlMax.max);
const sigRects = carpetLog.rects.filter(
  r => r.fill.startsWith("rgba(56, 189, 248,") || r.fill.startsWith("hsla(204.0, 94%, 52%,")
);
const rectLam = (r) => domLo + ((r.x + 0.5 * r.w) / 300.0) * (domHi - domLo);
const offBandRectCount = sigRects.filter(r => {
  const l = rectLam(r);
  return l < 2.98 || l > 4.42;
}).length;
const inBandMaxBarH = sigRects
  .filter(r => { const l = rectLam(r); return l >= 3.25 && l <= 3.95; })
  .reduce((m, r) => Math.max(m, r.h), 0.0);

// 3. Magnetic edge snapping on #wl-interval-slider (width = 300px, rail domain = [domLo, domHi]):
const wlToClientX = (lam) => ((lam - domLo) / Math.max(0.05, domHi - domLo)) * 300;
const wlSlider = getEl("wl-interval-slider");
const knobMin = getEl("wl-knob-min");
const knobMax = getEl("wl-knob-max");
wlSlider.dispatchEvent({ type: "pointerdown", button: 0, target: knobMin, clientX: wlToClientX(3.40), pointerId: 1, preventDefault() {} });
wlSlider.dispatchEvent({ type: "pointermove", target: wlSlider, clientX: wlToClientX(3.17), shiftKey: false, altKey: false });
const snappedLo = parseFloat(getEl("wl-val-min").textContent);
const knobMinSnappedCls = knobMin.classList.contains("snapped-band-edge");

// Holding Shift bypasses magnetic snap -> 0.05 Å grid (3.15 Å)
wlSlider.dispatchEvent({ type: "pointermove", target: wlSlider, clientX: wlToClientX(3.15), shiftKey: true, altKey: false });
const shiftBypassLo = parseFloat(getEl("wl-val-min").textContent);
// Release back onto magnetic snap
wlSlider.dispatchEvent({ type: "pointermove", target: wlSlider, clientX: wlToClientX(3.17), shiftKey: false, altKey: false });
wlSlider.dispatchEvent({ type: "pointerup", pointerId: 1 });

// Drag ] knob near upper band edge ~4.30 Å (raw λ = 4.22 Å)
wlSlider.dispatchEvent({ type: "pointerdown", button: 0, target: knobMax, clientX: wlToClientX(4.00), pointerId: 2, preventDefault() {} });
wlSlider.dispatchEvent({ type: "pointermove", target: wlSlider, clientX: wlToClientX(4.22), shiftKey: false, altKey: false });
wlSlider.dispatchEvent({ type: "pointerup", pointerId: 2 });
const snappedHi = parseFloat(getEl("wl-val-max").textContent);
const knobMaxSnappedCls = knobMax.classList.contains("snapped-band-edge");

// Hover tooltip on #wl-interval-slider reports integrated band
wlSlider.dispatchEvent({ type: "pointermove", target: wlSlider, clientX: wlToClientX(3.70) });
const hoverSliderTitle = wlSlider.title;

// 4. Switch to Color: I/σ -> λ rainbow disappears from Colorscale Crossover (shows I/σ gradient instead)
const selColor = getEl("sel-spot-color");
selColor.value = "isigi";
selColor.dispatchEvent({ type: "change", target: selColor });
const isigiEmphDisabled = Boolean(slEmph.disabled);
const isigiGrad = slEmph.style.getPropertyValue("--cs-gradient");
const isigiLblEmph = lblEmph.textContent;

console.log(JSON.stringify({
  initEmphDisabled,
  initWlGrad,
  initLblEmph,
  shiftedLblEmph,
  narrowLblEmph,
  initWlHasCarpet,
  wideAnyRainbowOnWl,
  wideOverlapCyanCount,
  wideNoiseSlateCount,
  narrowBlueBarsCount,
  narrowSkyStrokes,
  narrowOverlapCyanCount,
  narrowAnyRainbowOnWl,
  offBandRectCount,
  inBandMaxBarH,
  snappedLo,
  knobMinSnappedCls,
  shiftBypassLo,
  snappedHi,
  knobMaxSnappedCls,
  hoverSliderTitle,
  isigiEmphDisabled,
  isigiGrad,
  isigiLblEmph,
}));
"""
    )
    harness_file = tmp_path / "bandpass_viz_harness.js"
    harness_file.write_text(harness_js, encoding="utf-8")
    res = subprocess.run(
        ["node", str(harness_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(res.stdout.strip().splitlines()[-1])

    # 1. Colorscale Crossover is active in Color: λ/Zone, displays λ rainbow gradient, and shifts spectral midpoint
    assert out["initEmphDisabled"] is False
    assert "hsla(265.0, 95%, 60%," in out["initWlGrad"]
    assert "hsla(0.0, 95%, 60%," in out["initWlGrad"]
    assert out["initLblEmph"] == "λ ~ 3.45 Å"
    assert out["shiftedLblEmph"] == "λ ~ 4.25 Å"
    assert out["narrowLblEmph"] == "λ ~ 3.82 Å"

    # 2. Physical Wavelength λ [Å] rail is monochrome/sky-blue (zero rainbow hues), showing overlap vs noise vs outside-bracket blue
    assert out["initWlHasCarpet"] is True
    assert out["wideAnyRainbowOnWl"] is False
    assert out["narrowAnyRainbowOnWl"] is False
    assert out["wideOverlapCyanCount"] >= 15
    assert out["wideNoiseSlateCount"] >= 20
    assert out["narrowBlueBarsCount"] >= 8
    assert out["narrowSkyStrokes"] >= 2
    assert out["narrowOverlapCyanCount"] >= 8
    # Off-band wings [2.50, 2.98] & [4.42, 5.00] Å render zero signal bars despite harmonic/cosmic-ray outliers
    assert out["offBandRectCount"] == 0
    assert out["inBandMaxBarH"] >= 8.0

    # 3. Switching Spot Color away from Color: λ/Zone removes λ rainbow from Colorscale Crossover
    assert out["isigiEmphDisabled"] is False
    assert "hsla(265.0, 95%, 60%," not in out["isigiGrad"]
    assert "λ ~" not in out["isigiLblEmph"]

    # 4. Magnetic snapping snaps [ and ] onto [3.10, 4.30] Å quantile edges (with Shift bypass)
    assert abs(out["snappedLo"] - 3.10) <= 0.06
    assert out["knobMinSnappedCls"] is True
    assert abs(out["shiftBypassLo"] - 3.15) < 1e-6
    assert abs(out["snappedHi"] - 4.30) <= 0.10
    assert out["knobMaxSnappedCls"] is True
    assert "Q[1..99%] band ≈" in out["hoverSliderTitle"]


def test_mtz_f_global_scaling_careless_fobs_and_instrument_units(tmp_path) -> None:
    """Verify .mtz F global normalization, Careless F_obs/σ_F colorscale, f_calc phase immunity, and Σ(λ)·F̃² instrument units."""
    import json
    import re
    import shutil
    import subprocess

    import gemmi
    import h5py
    import numpy as np

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import (
        apply_mtz_to_payload,
        build_frame_only_response,
        build_rainbow_zone_payload,
        merge_frame_only_payload,
        render_rainbow_html,
    )

    cell_vals = [54.0, 62.0, 74.0, 90.0, 90.0, 90.0]
    sg_name = "P 21 21 21"
    uc = gemmi.UnitCell(*cell_vals)
    sg = gemmi.find_spacegroup_by_name(sg_name)
    asu = gemmi.ReciprocalAsu(sg)
    ops = sg.operations()
    b_mat = np.asarray(uc.frac.mat.tolist(), dtype=float).T

    # 0. Build a baseline geometry payload to sample realistic in-band reflections on CG4D
    solve_h5 = tmp_path / "cg4d_geom_seed.h5"
    with h5py.File(solve_h5, "w") as shf:
        shf.attrs["instrument"] = "CG4D"
        shf["sample/space_group"] = sg_name
        shf["sample/cell"] = np.array(cell_vals, dtype=float)
        shf["sample/U"] = np.eye(3)
        shf["sample/B"] = b_mat
        shf["goniometer/R"] = np.eye(3)[None, :, :]
        shf["goniometer/angles"] = np.zeros((1, 1), dtype=float)
        shf["instrument/wavelength"] = np.array([2.8, 4.5], dtype=float)

    geom_payload = build_rainbow_zone_payload(
        replay_h5=solve_h5,
        instrument="CG4D",
        space_group=sg_name,
        wavelength_range=(2.8, 4.5),
        d_min=2.4,
        frame_index=0,
        include_panel_images=False,
    )
    pred_pks = geom_payload["predicted_peaks"]
    assert len(pred_pks) >= 24

    # Deduplicate ASU reflections and pick 16 measured observations + unmeasured predicted reflections
    seen_asu: dict[tuple[int, int, int], dict] = {}
    for pk in pred_pks:
        h_raw = [int(x) for x in pk["hkl"]]
        ha, _ = asu.to_asu(h_raw, ops)
        akey = (int(ha[0]), int(ha[1]), int(ha[2]))
        if akey not in seen_asu:
            seen_asu[akey] = pk
    asu_items = list(seen_asu.items())
    assert len(asu_items) >= 16
    meas_items = asu_items[:12]
    unmeas_items = asu_items[12:18]

    # True normalized Careless F_obs, SigF_obs, and wavelength-dependent Careless scale Σ(λ)
    # Peak flux around λ=3.35 Å (Σ ≈ 220), weaker wing around λ=4.35 Å (Σ ≈ 45)
    def true_sigma_of_lambda(lam_val: float) -> float:
        return float(35.0 + 195.0 * np.exp(-0.5 * ((lam_val - 3.35) / 0.38) ** 2))

    rng = np.random.default_rng(202604)
    true_f_map: dict[tuple[int, int, int], tuple[float, float, float, float]] = {}
    for idx_a, (akey, _pk) in enumerate(asu_items[:24]):
        f_obs = float(0.45 + 0.18 * (idx_a % 9) + 0.05 * rng.uniform())
        sigf_obs = float(0.08 + 0.02 * (idx_a % 4))
        # Careless truncated-normal posterior second moment: I = F^2 + SigF^2
        i_obs_m = float(f_obs * f_obs + sigf_obs * sigf_obs)
        sigi_obs_m = float(2.0 * f_obs * sigf_obs)
        true_f_map[akey] = (i_obs_m, sigi_obs_m, f_obs, sigf_obs)

    # Write integrated.h5 with 2 frames and Careless work_dir (merged.mtz + predictions.mtz)
    integ_h5 = tmp_path / "cg4d_integ_fobs.h5"
    work_dir = tmp_path / "careless_work"
    work_dir.mkdir()
    n_obs = len(meas_items)
    obs_lams = np.linspace(2.90, 4.40, n_obs, dtype=np.float64)
    obs_scales = np.array([true_sigma_of_lambda(w) for w in obs_lams], dtype=np.float64)
    obs_h = np.array([it[1]["hkl"][0] for it in meas_items], dtype=np.int32)
    obs_k = np.array([it[1]["hkl"][1] for it in meas_items], dtype=np.int32)
    obs_l = np.array([it[1]["hkl"][2] for it in meas_items], dtype=np.int32)
    obs_ipred = np.array(
        [obs_scales[i] * true_f_map[meas_items[i][0]][0] for i in range(n_obs)],
        dtype=np.float64,
    )
    # Detector Poisson + background counting noise: σ_{I_obs} = sqrt(σ_bg^2 + I_pred)
    obs_sigi = np.sqrt(100.0 + np.maximum(obs_ipred, 0.0))
    obs_iobs = obs_ipred + np.linspace(-0.6, 0.7, n_obs) * obs_sigi
    obs_runs = np.array(
        [0 if i < n_obs // 2 else 1 for i in range(n_obs)], dtype=np.int32
    )

    with h5py.File(integ_h5, "w") as hf:
        hf.attrs["instrument"] = "CG4D"
        hf.create_dataset("file_offsets", data=np.array([0, 1], dtype=np.int32))
        hf.create_dataset(
            "files", data=np.array(["run0.nxs.h5", "run1.nxs.h5"], dtype="S16")
        )
        hf["sample/space_group"] = sg_name
        hf["sample/cell"] = np.array(cell_vals, dtype=float)
        hf["sample/U"] = np.eye(3)
        hf["sample/B"] = b_mat
        hf["goniometer/R"] = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        hf["instrument/wavelength"] = np.array([2.8, 4.5], dtype=float)
        pk = hf.create_group("peaks")
        pk["h"] = obs_h
        pk["k"] = obs_k
        pk["l"] = obs_l
        pk["run_index"] = obs_runs
        pk["bank"] = np.array(
            [int(it[1].get("bank") or 1) for it in meas_items], dtype=np.int32
        )
        pk["pixel_r"] = np.full(n_obs, 128.0, dtype=np.float64)
        pk["pixel_c"] = np.full(n_obs, 128.0, dtype=np.float64)
        pk["intensity"] = obs_iobs
        pk["sigma"] = obs_sigi
        pk["lambda"] = obs_lams

    # Write Careless merged.mtz (with explicit F, SigF, I, SigI) and predictions.mtz
    mtz_m = gemmi.Mtz(with_base=True)
    mtz_m.spacegroup = sg
    mtz_m.set_cell_for_all(uc)
    mtz_m.add_dataset("careless")
    mtz_m.add_column("F", "F")
    mtz_m.add_column("SigF", "Q")
    mtz_m.add_column("I", "J")
    mtz_m.add_column("SigI", "Q")
    m_rows = [
        [ak[0], ak[1], ak[2], vals[2], vals[3], vals[0], vals[1]]
        for ak, vals in true_f_map.items()
    ]
    mtz_m.set_data(np.asarray(m_rows, dtype=np.float32))
    mtz_m.write_to_file(str(work_dir / "merged.mtz"))

    mtz_p = gemmi.Mtz(with_base=True)
    mtz_p.spacegroup = sg
    mtz_p.set_cell_for_all(uc)
    mtz_p.add_dataset("predictions")
    for cname, ctype in (
        ("Iobs", "J"),
        ("SigIobs", "Q"),
        ("Ipred", "J"),
        ("SigIpred", "Q"),
        ("Scale", "R"),
    ):
        mtz_p.add_column(cname, ctype)
    p_rows = []
    for i in range(n_obs):
        ak = meas_items[i][0]
        p_rows.append(
            [
                ak[0],
                ak[1],
                ak[2],
                obs_iobs[i],
                obs_sigi[i],
                obs_ipred[i],
                0.15 * obs_sigi[i],
                obs_scales[i],
            ]
        )
    mtz_p.set_data(np.asarray(p_rows, dtype=np.float32))
    mtz_p.write_to_file(str(work_dir / "predictions.mtz"))
    (work_dir / "careless_metrics.json").write_text(
        json.dumps({"cc_half": 0.991, "n_merged_asu": len(true_f_map)}),
        encoding="utf-8",
    )

    runner = NovaGalaxyPipelineRunner.__new__(NovaGalaxyPipelineRunner)
    runner._stamp_careless_merge_into_h5(str(integ_h5), str(work_dir), {})

    with h5py.File(integ_h5, "r") as hf_chk:
        assert "merged_f" in hf_chk["careless"]
        assert "merged_sigf" in hf_chk["careless"]
        assert "merged_f" in hf_chk["peaks"]
        assert "merged_sigf" in hf_chk["peaks"]

    careless_payload = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        instrument="CG4D",
        frame_index=0,
        d_min=2.4,
        include_panel_images=False,
        include_all_runs_peaks=True,
    )
    assert careless_payload["artifact_kind"] == "merged_careless"
    assert isinstance(careless_payload.get("careless_f_asu_lookup"), dict)
    assert len(careless_payload["careless_f_asu_lookup"]) == len(true_f_map)
    assert isinstance(careless_payload.get("careless_scale_profile"), dict)
    assert careless_payload["careless_scale_profile"]["global_median"] > 30.0

    # Verify F_obs / σ_F is populated on both integrated observations and unmeasured predicted HKLs
    ak0, (_, _, f0_true, sf0_true) = next(iter(true_f_map.items()))
    lu_f0 = careless_payload["careless_f_asu_lookup"][f"{ak0[0]},{ak0[1]},{ak0[2]}"]
    assert abs(lu_f0[0] - f0_true) < 1e-3
    assert abs(lu_f0[1] - sf0_true) < 1e-3
    integ0 = careless_payload["integrated_peaks"][0]
    ak_i0 = tuple(int(x) for x in integ0["asu_hkl"])
    assert abs(integ0["merged_f"] - true_f_map[ak_i0][2]) < 1e-3
    assert (
        abs(integ0["merged_fsigf"] - (true_f_map[ak_i0][2] / true_f_map[ak_i0][3]))
        < 1e-2
    )

    # 1. Arbitrary-unit external .mtz with (F, SIGF, I, SIGI) scaled by 420x:
    #    - Divides out single global scale (~420.0) to match Careless F_obs via careless_shared_q75
    #    - Preserves Careless merged_i / merged_f intact (does NOT clobber!)
    #    - Maps normalized F̃(hkl)^2 back into instrument detector units via Σ(λ) · F̃^2
    arb_scale = 420.0
    ext_mtz = gemmi.Mtz(with_base=True)
    ext_mtz.spacegroup = sg
    ext_mtz.set_cell_for_all(uc)
    ext_mtz.add_dataset("external_ref")
    ext_mtz.add_column("F", "F")
    ext_mtz.add_column("SIGF", "Q")
    ext_mtz.add_column("I", "J")
    ext_mtz.add_column("SIGI", "Q")
    ext_rows = [
        [
            ak[0],
            ak[1],
            ak[2],
            arb_scale * vals[2],
            arb_scale * vals[3],
            (arb_scale**2) * (vals[2] ** 2),
            (arb_scale**2) * vals[1],
        ]
        for ak, vals in true_f_map.items()
    ]
    ext_mtz.set_data(np.asarray(ext_rows, dtype=np.float32))
    ext_mtz_path = tmp_path / "external_420x.mtz"
    ext_mtz.write_to_file(str(ext_mtz_path))

    overlaid_f = apply_mtz_to_payload(
        careless_payload,
        ext_mtz_path,
        {"name": "very_long_refinement_structure_factors_5vnq_3.74A_final_model.mtz"},
    )
    assert overlaid_f["mtz_has_f_column"] is True
    assert overlaid_f["mtz_has_sigf_column"] is True
    assert overlaid_f["mtz_f_column"] == "F"
    assert overlaid_f["mtz_sigf_column"] == "SIGF"
    assert overlaid_f["mtz_scale_basis"] == "careless_shared_q75"
    assert overlaid_f["default_color_mode"] == "mtz_fcalc"

    ov_integ0 = overlaid_f["integrated_peaks"][0]
    # Careless merged_i and merged_f were NOT clobbered by the arbitrary 420^2-scaled external MTZ!
    assert ov_integ0["merged_i"] == integ0["merged_i"]
    assert ov_integ0["merged_f"] == integ0["merged_f"]
    assert abs(ov_integ0["merged_i"] - true_f_map[ak_i0][0]) < 1e-2
    assert abs(ov_integ0["merged_f"] - true_f_map[ak_i0][2]) < 1e-3
    # Normalized |F̃_calc|² sits on the exact same Q75 scale as Careless |F_obs|² (`merged_i`)!
    shared_fcalc_sq = [
        float(overlaid_f["mtz_f_asu_lookup"][k][3])
        for k in overlaid_f["careless_i_asu_lookup"]
        if k in overlaid_f["mtz_f_asu_lookup"]
    ]
    shared_fobs_sq = [
        float(overlaid_f["careless_i_asu_lookup"][k][0])
        for k in overlaid_f["careless_i_asu_lookup"]
        if k in overlaid_f["mtz_f_asu_lookup"]
    ]
    assert (
        abs(np.quantile(shared_fcalc_sq, 0.75) - np.quantile(shared_fobs_sq, 0.75))
        < 1e-3
    )
    assert abs(ov_integ0["mtz_i_norm"] - (ov_integ0["mtz_f"] ** 2)) < 1e-3
    # Post-merge I_pred/σ uses Σ_obs * |F̃_calc|² / σ_{I_obs} on integrated peaks and Σ(λ)/σ(λ) * |F̃_calc|² on predicted peaks
    expected_inst0 = float(ov_integ0["careless_scale"]) * float(ov_integ0["mtz_i_norm"])
    expected_ipred_snr0 = expected_inst0 / float(ov_integ0["sigma"])
    assert (
        abs(float(ov_integ0["mtz_inst_i"]) - expected_inst0) / max(expected_inst0, 1e-6)
        < 1e-2
    )
    assert (
        abs(float(ov_integ0["mtz_ipred_isigi"]) - expected_ipred_snr0)
        / max(expected_ipred_snr0, 1e-6)
        < 1e-2
    )

    unmeas_key_set = {it[0] for it in unmeas_items}
    unmeas_preds = [
        p
        for p in overlaid_f["predicted_peaks"]
        if tuple(int(x) for x in p["asu_hkl"]) in unmeas_key_set
        and p.get("mtz_ipred_isigi") is not None
    ]
    assert len(unmeas_preds) >= 2
    for upk in unmeas_preds:
        assert upk.get("careless_scale") is None
        assert float(upk["mtz_inst_i"]) > 0.0
        assert float(upk["mtz_ipred_isigi"]) > 0.0
        # Σ(λ) > 30 across [2.8, 4.5] Å -> instrument intensity is dramatically scaled above unit |F̃_calc|^2
        assert float(upk["mtz_inst_i"]) > 20.0 * float(upk["mtz_i_norm"])

    # Frame-only fast switch preserves careless_scale_profile, careless_i_asu_lookup, careless_f_asu_lookup, and mtz_f_* lookups
    fast_f1 = build_rainbow_zone_payload(
        replay_h5=integ_h5,
        instrument="CG4D",
        frame_index=1,
        d_min=2.4,
        include_panel_images=False,
        include_all_runs_peaks=False,
    )
    merged_f1 = merge_frame_only_payload(overlaid_f, dict(fast_f1))
    assert merged_f1["careless_scale_profile"] == overlaid_f["careless_scale_profile"]
    assert merged_f1["careless_i_asu_lookup"] == overlaid_f["careless_i_asu_lookup"]
    assert merged_f1["careless_f_asu_lookup"] == overlaid_f["careless_f_asu_lookup"]
    assert merged_f1["mtz_f_asu_lookup"] == overlaid_f["mtz_f_asu_lookup"]
    assert merged_f1["mtz_has_f_column"] is True
    resp_f1 = build_frame_only_response(merged_f1)
    assert "careless_scale_profile" not in resp_f1
    assert "careless_i_asu_lookup" not in resp_f1
    assert "careless_f_asu_lookup" not in resp_f1
    assert "mtz_f_asu_lookup" not in resp_f1

    # 2. Calculated model .mtz with ('f_calc', 'F') and ('PHIf_calc', 'P') BEFORE merging (like 5vnq-3.74.mtz):
    #    - PHIf_calc (type 'P') is NEVER treated as SIGF
    #    - Before Careless merge, `fobs` (|F_obs|²) and `ipred_isigi` (I_pred/σ) MUST be disabled!
    fcalc_mtz = gemmi.Mtz(with_base=True)
    fcalc_mtz.spacegroup = sg
    fcalc_mtz.set_cell_for_all(uc)
    fcalc_mtz.add_dataset("model_fcalc")
    fcalc_mtz.add_column("f_calc", "F")
    fcalc_mtz.add_column("PHIf_calc", "P")
    fcalc_rows = [
        [
            ak[0],
            ak[1],
            ak[2],
            800.0 * vals[2],
            float(
                ((idx_r * 73) % 360) - 180
            ),  # signed phase angle in degrees, including 0° and negatives!
        ]
        for idx_r, (ak, vals) in enumerate(true_f_map.items())
    ]
    fcalc_mtz.set_data(np.asarray(fcalc_rows, dtype=np.float32))
    fcalc_path = tmp_path / "model_5vnq_like.mtz"
    fcalc_mtz.write_to_file(str(fcalc_path))

    standalone_fcalc = apply_mtz_to_payload(
        geom_payload, fcalc_path, {"name": "model_5vnq_like.mtz"}
    )
    assert standalone_fcalc["mtz_has_f_column"] is True
    assert standalone_fcalc["mtz_has_sigf_column"] is False
    assert standalone_fcalc["mtz_f_column"] == "f_calc"
    assert standalone_fcalc["mtz_sigf_column"] is None
    assert standalone_fcalc["mtz_scale_basis"] == "unit_wilson_rms"
    assert standalone_fcalc["default_color_mode"] == "mtz_fcalc"
    fcalc_matched = [
        p
        for p in standalone_fcalc["predicted_peaks"]
        if p.get("mtz_i_norm") is not None
    ]
    assert len(fcalc_matched) >= 12
    assert all(
        p["mtz_i_norm"] >= 0.0
        and p.get("mtz_sigf") is None
        and p.get("mtz_ipred_isigi") is None
        and p.get("merged_i") is None
        for p in fcalc_matched
    )

    if not shutil.which("node"):
        return

    # 3. Full Node VM check on rendered HTML/JS for:
    #    - Concise non-overflowing #sel-spot-color labels even with a 62-char .mtz filename
    #    - Paired |F_obs|² <-> |F_calc|² shared colorbar bounds and I/σ <-> I_pred/σ shared colorbar bounds
    #    - Pre-merge gating (`fobs` and `ipred_isigi` strictly disabled before Careless merge!)
    #    - λ-dependent I_pred/σ across goniometer frames + hard [λ_min, λ_max] slider bandpass cutoff
    html_doc = render_rainbow_html(payload=overlaid_f)
    for opt_val in ("isigi", "fobs", "mtz_fcalc", "ipred_isigi"):
        assert f'value="{opt_val}"' in html_doc

    inline_js = re.findall(r"<script>([\s\S]*?)</script>", html_doc)[-1]
    assert inline_js.rstrip().endswith("})();")
    instrumented_js = (
        inline_js.rstrip()[: -len("})();")]
        + "\nglobalThis.__rz = { state, DATA, PEAKS, getPeakDataAvailability, peakStrokeColor, peakFobsSq, peakMtzFcalcSq, peakIpredIsigi, isPeakVisible, recomputeBoundsAndReload, updatePeakInspector, syncWulffUI };\n})();"
    )
    opt_defs = [
        ("wavelength", "Color: λ/Zone"),
        ("orbit", "Color: ASU Orbit"),
        ("isigi", "Color: I/σ"),
        ("fobs", "Color: |F_obs|²"),
        ("mtz_fcalc", "Color: |F_calc|²"),
        ("ipred_isigi", "Color: I_pred/σ"),
        ("residual", "Color: ΔI/σ Residual"),
        ("scale", "Color: Scale"),
    ]
    js_opts_literal = json.dumps(
        [{"value": k, "textContent": lbl, "disabled": False} for k, lbl in opt_defs]
    )

    vm_js = (
        f"""
const elems = new Map();
function makeSelectOptions() {{
  return {js_opts_literal}.map(o => ({{ ...o, title: "", style: {{}} }}));
}}
function makeCtx() {{
  return {{
    fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "left",
    textBaseline: "alphabetic", globalAlpha: 1, imageSmoothingEnabled: true,
    setTransform() {{}}, resetTransform() {{}}, scale() {{}}, translate() {{}}, rotate() {{}}, transform() {{}},
    clearRect() {{}}, fillRect() {{}}, strokeRect() {{}}, beginPath() {{}}, closePath() {{}},
    moveTo() {{}}, lineTo() {{}}, arc() {{}}, ellipse() {{}}, quadraticCurveTo() {{}}, bezierCurveTo() {{}},
    rect() {{}}, roundRect() {{}}, stroke() {{}}, fill() {{}}, clip() {{}}, save() {{}}, restore() {{}},
    setLineDash() {{}}, fillText() {{}}, strokeText() {{}}, drawImage() {{}},
    measureText(t) {{ return {{ width: String(t).length * 6 }}; }},
    createLinearGradient() {{ return {{ addColorStop() {{}} }}; }},
    createRadialGradient() {{ return {{ addColorStop() {{}} }}; }},
    getImageData(x, y, w, h) {{ return {{ data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }}; }},
    putImageData() {{}}, createImageData(w, h) {{ return {{ data: new Uint8ClampedArray(Math.max(4, w * h * 4)) }}; }},
  }};
}}
function makeEl(id = "") {{
  const listeners = {{}};
  const styleProps = {{}};
  const isSelColor = id === "sel-spot-color" || id === "sel-wulff-color";
  const el = {{
    id, tagName: isSelColor ? "SELECT" : "DIV",
    value: id === "slider-color-emphasis" ? "0.38" : (isSelColor ? "mtz_fcalc" : ""),
    min: "0.05", max: "0.95", step: "0.01",
    checked: false, disabled: false, textContent: "", innerHTML: "", title: "",
    width: 1200, height: 640, clientWidth: 1200, clientHeight: 640,
    scrollWidth: 300, scrollHeight: 240, scrollTop: 0, offsetWidth: 300, offsetHeight: 36,
    files: [], options: isSelColor ? makeSelectOptions() : [], children: [], parentElement: null,
    dataset: {{}},
    style: new Proxy(styleProps, {{
      get(t, k) {{
        if (k === "setProperty") return (p, v) => {{ t[p] = v; }};
        if (k === "getPropertyValue") return (p) => t[p] || "";
        return t[k] ?? "";
      }},
      set(t, k, v) {{ t[k] = v; return true; }}
    }}),
    classList: {{
      _s: new Set(),
      add(...c) {{ c.forEach(x => this._s.add(x)); }},
      remove(...c) {{ c.forEach(x => this._s.delete(x)); }},
      toggle(c, f) {{ if (f === undefined ? !this._s.has(c) : f) {{ this._s.add(c); return true; }} this._s.delete(c); return false; }},
      contains(c) {{ return this._s.has(c); }}
    }},
    addEventListener(ev, fn) {{ (listeners[ev] = listeners[ev] || []).push(fn); }},
    removeEventListener() {{}},
    dispatchEvent(evObj) {{ (listeners[evObj.type] || []).forEach(fn => fn.call(el, evObj)); }},
    click() {{ el.dispatchEvent({{ type: "click", target: el, preventDefault() {{}}, stopPropagation() {{}} }}); }},
    focus() {{}}, blur() {{}}, select() {{}}, remove() {{}},
    appendChild(ch) {{ this.children.push(ch); ch.parentElement = this; return ch; }},
    insertBefore(ch) {{ this.children.push(ch); ch.parentElement = this; return ch; }},
    replaceChildren(...ch) {{ this.children = ch; }},
    setAttribute(k, v) {{ this[k] = v; }},
    getAttribute(k) {{ return this[k] ?? null; }},
    removeAttribute(k) {{ delete this[k]; }},
    contains(target) {{ return target === el; }},
    getBoundingClientRect() {{ return {{ left: 0, top: 0, right: 1200, bottom: 640, width: 1200, height: 640 }}; }},
    getContext() {{ if (!el._ctx) el._ctx = makeCtx(); return el._ctx; }},
    querySelector() {{ return null; }},
    querySelectorAll() {{ return []; }},
    closest() {{ return null; }},
  }};
  return el;
}}
function getEl(id) {{
  if (!elems.has(id)) elems.set(id, makeEl(id));
  return elems.get(id);
}}
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.parent = globalThis;
globalThis.top = globalThis;
globalThis.addEventListener = () => {{}};
globalThis.removeEventListener = () => {{}};
globalThis.devicePixelRatio = 1;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.localStorage = {{ getItem() {{ return null; }}, setItem() {{}}, removeItem() {{}} }};
globalThis.sessionStorage = {{ getItem() {{ return null; }}, setItem() {{}}, removeItem() {{}} }};
globalThis.location = {{ href: "about:srcdoc", origin: "http://localhost:8080", pathname: "/", search: "" }};
globalThis.navigator = {{ clipboard: {{ writeText: async () => {{}} }} }};
globalThis.Image = class {{ set src(v) {{ this._src = v; }} }};
globalThis.ResizeObserver = class {{ observe() {{}} disconnect() {{}} }};
globalThis.MutationObserver = class {{ observe() {{}} disconnect() {{}} }};
globalThis.AbortController = class {{ constructor() {{ this.signal = {{}}; }} abort() {{}} }};
let rafCb = null;
globalThis.requestAnimationFrame = (fn) => {{ rafCb = fn; return 1; }};
globalThis.cancelAnimationFrame = () => {{}};
globalThis.fetch = async () => ({{ ok: false, status: 404, json: async () => ({{}}), text: async () => "" }});
globalThis.document = {{
  body: makeEl("body"),
  documentElement: makeEl("html"),
  baseURI: "http://localhost:8080/",
  getElementById: (id) => getEl(id),
  createElement: (tag) => makeEl(tag),
  createTextNode: (txt) => ({{ textContent: txt }}),
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: () => {{}},
  removeEventListener: () => {{}},
}};
"""
        + instrumented_js
        + f"""
;
const {{ state, DATA, PEAKS, getPeakDataAvailability, peakStrokeColor, peakFobsSq, peakMtzFcalcSq, peakIpredIsigi, isPeakVisible, recomputeBoundsAndReload, updatePeakInspector, syncWulffUI }} = globalThis.__rz;
getEl("canvas-detector").getContext("2d");
getEl("canvas-wulff").getContext("2d");
if (rafCb) {{ const cb = rafCb; rafCb = null; cb(); }}

const selSpot = getEl("sel-spot-color");
const slEmph = getEl("slider-color-emphasis");
const initMode = state.wulffColorMode;
const optStatesPostMerge = Object.fromEntries(
  selSpot.options.map(o => [o.value, {{ disabled: o.disabled, text: o.textContent, title: o.title }}])
);
const initTooltip = slEmph.title;
const availPost = getPeakDataAvailability();
const boundsMatchF = JSON.stringify(availPost.mtzFcalcBounds) === JSON.stringify(availPost.fobsBounds);
const boundsMatchSnr = JSON.stringify(availPost.ipredIsigiBounds) === JSON.stringify(availPost.isigiBounds);

// Switch to |F_obs|² (`fobs`)
selSpot.value = "fobs";
selSpot.dispatchEvent({{ type: "change", target: selSpot }});
const fobsTooltip = slEmph.title;
const fobsStroke = peakStrokeColor(PEAKS[0], 1.0);

// Switch to I_pred/σ (`ipred_isigi`) and check λ-dependent illumination + hard [λ_min, λ_max] bandpass!
selSpot.value = "ipred_isigi";
selSpot.dispatchEvent({{ type: "change", target: selSpot }});
const ipredTooltip = slEmph.title;
state.lamMin = 3.10;
state.lamMax = 4.45;
const hiFluxProbe = {{ asu_hkl: PEAKS[0].asu_hkl, hkl: PEAKS[0].hkl, wavelength: 3.35, d_spacing: 2.6, mtz_f: 1.25, mtz_i_norm: 1.5625 }};
const loFluxProbe = {{ asu_hkl: PEAKS[0].asu_hkl, hkl: PEAKS[0].hkl, wavelength: 4.40, d_spacing: 2.6, mtz_f: 1.25, mtz_i_norm: 1.5625 }};
const outOfBandProbe = {{ asu_hkl: PEAKS[0].asu_hkl, hkl: PEAKS[0].hkl, wavelength: 2.95, d_spacing: 2.6, mtz_f: 1.25, mtz_i_norm: 1.5625 }};
const ipredHiFlux = peakIpredIsigi(hiFluxProbe);
const ipredLoFlux = peakIpredIsigi(loFluxProbe);
const ipredOutBand = peakIpredIsigi(outOfBandProbe);
const visInBand = isPeakVisible(hiFluxProbe);
const visOutBand = isPeakVisible(outOfBandProbe);

// Inspector readout includes |F_obs|²=..., |F_calc|²=..., and I_pred/σ=... for in-band peaks (and omits I_pred/σ outside [λ_min, λ_max])
const inBandPeak = PEAKS.find(p => p.wavelength >= state.lamMin && p.wavelength <= state.lamMax) || PEAKS[PEAKS.length - 1];
updatePeakInspector(inBandPeak);
const inspectorHtml = getEl("inspector-box").innerHTML;

// Reload pre-merge standalone f_calc payload -> `fobs` and `ipred_isigi` MUST be disabled, while `mtz_fcalc` is enabled!
const preMergeDoc = {json.dumps(standalone_fcalc)};
recomputeBoundsAndReload(preMergeDoc, {{ resetSelectedFrame: true }});
syncWulffUI();
const preMergeMode = state.wulffColorMode;
const optStatesPreMerge = Object.fromEntries(
  selSpot.options.map(o => [o.value, {{ disabled: o.disabled, text: o.textContent }}])
);
selSpot.value = "ipred_isigi";
selSpot.dispatchEvent({{ type: "change", target: selSpot }});
const blockedIpredMode = state.wulffColorMode;

console.log(JSON.stringify({{
  initMode,
  optStatesPostMerge,
  initTooltip,
  boundsMatchF,
  boundsMatchSnr,
  fobsTooltip,
  fobsStroke,
  ipredTooltip,
  ipredHiFlux,
  ipredLoFlux,
  ipredOutBandIsNaN: !Number.isFinite(ipredOutBand),
  visInBand,
  visOutBand,
  inspectorHtml,
  preMergeMode,
  optStatesPreMerge,
  blockedIpredMode,
}}));
"""
    )
    vm_file = tmp_path / "mtz_f_scaling_vm.js"
    vm_file.write_text(vm_js, encoding="utf-8")
    vm_res = subprocess.run(
        ["node", str(vm_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    vm_out = json.loads(vm_res.stdout.strip().splitlines()[-1])

    assert vm_out["initMode"] == "mtz_fcalc"
    assert vm_out["optStatesPostMerge"]["fobs"]["disabled"] is False
    assert vm_out["optStatesPostMerge"]["mtz_fcalc"]["disabled"] is False
    assert vm_out["optStatesPostMerge"]["ipred_isigi"]["disabled"] is False
    # Concise option labels never append long .mtz filenames (filename lives in .title tooltip instead)
    assert vm_out["optStatesPostMerge"]["mtz_fcalc"]["text"] == "Color: |F_calc|²"
    assert vm_out["optStatesPostMerge"]["ipred_isigi"]["text"] == "Color: I_pred/σ"
    assert (
        "very_long_refinement_structure_factors"
        in vm_out["optStatesPostMerge"]["mtz_fcalc"]["title"]
    )
    # Paired colorbars share identical quantile bounds for direct 1-to-1 comparison
    assert vm_out["boundsMatchF"] is True
    assert vm_out["boundsMatchSnr"] is True
    assert "|F_calc|²" in vm_out["initTooltip"]
    assert "|F_obs|²" in vm_out["fobsTooltip"]
    assert vm_out["fobsStroke"].startswith("rgba(") and not vm_out[
        "fobsStroke"
    ].endswith("0.000)")
    assert "I_pred/σ" in vm_out["ipredTooltip"]
    # Illumination across frames + hard [λ_min, λ_max] slider bandpass cutoff
    assert vm_out["ipredHiFlux"] > 1.2 * vm_out["ipredLoFlux"]
    assert vm_out["ipredOutBandIsNaN"] is True
    assert vm_out["visInBand"] is True
    assert vm_out["visOutBand"] is False
    assert "|F_obs|²=" in vm_out["inspectorHtml"]
    assert "|F_calc|²=" in vm_out["inspectorHtml"]
    assert "I_pred/σ=" in vm_out["inspectorHtml"]
    # Pre-merge gating: `fobs` and `ipred_isigi` strictly disabled before Careless merge!
    assert vm_out["preMergeMode"] == "mtz_fcalc"
    assert vm_out["optStatesPreMerge"]["mtz_fcalc"]["disabled"] is False
    assert vm_out["optStatesPreMerge"]["fobs"]["disabled"] is True
    assert vm_out["optStatesPreMerge"]["ipred_isigi"]["disabled"] is True
    assert vm_out["blockedIpredMode"] == "mtz_fcalc"
