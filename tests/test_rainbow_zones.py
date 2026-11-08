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
    peaks_xyz, U, B, UB, _, _ = _synthetic_orthorhombic_peaks()

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
    assert "btn-zone-order" in content
    assert "btn-zone-arcs" in content
    assert "showZoneArcs: true" in content
    assert "drawSingleZoneArc" in content
    assert "pipeline-stages" in content
    assert "parseH5BuffersInBrowser" in content
    assert "btn-new-tab" in content
    assert "LATTICE LADDER" in content

    # Extract embedded JSON and validate it parses cleanly
    match = re.search(r"window\.RAINBOW_ZONES_DATA\s*=\s*(\{.*?\});\n", content, re.S)
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
                {"stage": "global", "objective": -0.513, "iterations": 97, "success": True},
                {"stage": "120-zones", "objective": -1.101, "iterations": 192, "success": True},
                {"stage": "cell", "objective": -2.011, "iterations": 235, "success": True},
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


def test_nova_mvvm_pydantic_and_view_model(tmp_path):
    """Verify NOVA tutorial MVVM architecture (`SubhklConfig`, `MainModel`, `MainViewModel`, `create_viewmodels`)."""
    import pytest

    pytest.importorskip("pydantic")
    pytest.importorskip("nova.mvvm")
    from pydantic import ValidationError
    from nova.mvvm.trame_binding import TrameBinding
    from trame.app import get_server

    from subhkl.viz.nova_app.models.subhkl_model import InstrumentOptions, SubhklConfig
    from subhkl.viz.nova_app.mvvm_factory import create_viewmodels

    # 1. Verify Pydantic validation & defaults
    cfg = SubhklConfig(instrument=InstrumentOptions.MANDI, a=70.0)
    assert cfg.instrument == InstrumentOptions.MANDI
    assert cfg.a == 70.0
    assert cfg.wavelength_max > cfg.wavelength_min

    with pytest.raises(ValidationError):
        SubhklConfig(wavelength_min=5.0, wavelength_max=2.0)

    # 2. Verify MainApp(server=None) initializes ThemedApp, TrameBinding, ViewModel & UI cleanly
    from subhkl.viz.nova_app.views.main_view import MainApp

    app = MainApp(instrument="CG4D")
    assert app.server is not None
    assert app.view_model.view_state.running is False
    assert app.view_model.model.subhkl.data.instrument == InstrumentOptions.CG4D
    assert "DIFFRACTION PARALLAX" in app.server.state.parallax_html


