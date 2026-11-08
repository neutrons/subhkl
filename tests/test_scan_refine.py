"""Staged refinement modes, count observations, and rotation metadata."""

import json
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from subhkl.config import reduction_settings
from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix
from subhkl.scan_refine import load_observations, run_scan_refinement
from subhkl.search.scan_band import band_sum, ewald_directions
from subhkl.solve_frames import frame_kinematics


def test_chunked_band_value_and_gradient_match_dense_signed_sum():
    rng = np.random.default_rng(21)
    with jax.enable_x64():
        X = jnp.asarray(rng.normal(size=(9, 3)))
        d = jnp.asarray(rng.normal(size=(37, 3)))
        lo = jnp.asarray(rng.uniform(0.1, 0.2, 37))
        hi = lo + 0.3
        weights = jnp.asarray(rng.normal(size=37))
        zw = jnp.asarray(rng.uniform(size=9))
        dense = jax.value_and_grad(lambda x: band_sum(x, d, lo, hi, weights, zw, 100))(
            X
        )
        chunked = jax.value_and_grad(lambda x: band_sum(x, d, lo, hi, weights, zw, 8))(
            X
        )
        for a, b in zip(dense, chunked):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
        # Empty resolution intervals (hi <= lo) contribute zero value and zero gradient
        empty_val, empty_grad = jax.value_and_grad(
            lambda x: band_sum(x, d, hi, lo, weights, zw, 8)
        )(X)
        assert float(empty_val) == 0.0
        np.testing.assert_array_equal(empty_grad, 0.0)


def test_unassigned_reflection_has_finite_masked_gradient():
    g = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.2, -0.3]])

    def objective(x):
        rays = ewald_directions(x)
        return jnp.sum(jnp.array([0.0, 1.0]) * jnp.sum((rays - 0.2) ** 2, axis=1))

    value, gradient = jax.value_and_grad(objective)(g)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[0], 0)


@pytest.fixture
def inputs(tmp_path):
    frames, bootstrap = tmp_path / "counts.h5", tmp_path / "seed.h5"
    axes = np.array([[0, 1, 0, -1]], float)
    angles = np.array([[10.0], [40.0]])
    counts = np.full((2, 512, 512), 3.0)
    counts[0, :32, :32] = 1
    counts[1, :32, :32] = 5
    with h5py.File(frames, "w") as f:
        f["images"], f["bank_ids"] = counts, [62, 62]
        f["file_offsets"] = [0, 1]
        f["goniometer/axes"], f["goniometer/angles"] = axes, angles
        f["goniometer/names"] = [b"omega"]
        for k, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"), [12, 13, 14, 90, 105, 90]
        ):
            f[f"sample/{k}"] = v
        f["sample/space_group"] = "P 1 21 1"
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = np.eye(3)
    return frames, bootstrap


def test_count_observations_keep_both_signs_and_use_detector_pixel_origin(inputs):
    frames, _ = inputs
    obs = load_observations(
        SimpleNamespace(
            nexus=frames,
            finder=None,
            runs=[0, 1],
            stable_runs=[0],
            binning=32,
            static_mask_file=None,
        )
    )
    assert len(obs[0]) == 512
    assert obs[1].min() == obs[2].min() == 15.5
    assert obs[1].max() == obs[2].max() == 495.5
    assert np.any(obs[-1] < 0) and np.any(obs[-1] > 0)
    assert np.count_nonzero(obs[-1] == 0) == 510


def test_mode_bounds_and_native_metadata_roundtrip(inputs, tmp_path):
    frames, bootstrap = inputs
    output = tmp_path / "band.h5"
    result = run_scan_refinement(
        frames,
        output,
        bootstrap=bootstrap,
        runs=[0, 1],
        stable_runs=[0],
        modes=[],
        maxiter=1,
        n_zones=60,
        binning=32,
    )
    assert result["status"] in ("converged", "not_converged")
    with h5py.File(output) as f:
        np.testing.assert_array_equal(f["scan_refinement/parameters"][3:], 0)
        assert "peaks" not in f
        assert f["scan_refinement"].attrs["finder"] == ""
        assert ("sample/U" in f) == (result["status"] == "converged")
        diagnostics = json.loads(f["scan_refinement"].attrs["stages"])
        assert all("success" in stage for stage in diagnostics)
        R, _, runs, angles, changed = frame_kinematics(f, 2, f["file_offsets"][()])
        assert not changed  # folded deltas must not be applied a second time
        np.testing.assert_array_equal(runs, [0, 1])
        expected = [
            calc_goniometer_rotation_matrix(f["goniometer/axes"][()], a) for a in angles
        ]
        np.testing.assert_allclose(R, expected, atol=1e-12)


def test_bad_modes_and_finderless_pixel_polish_rejected(inputs, tmp_path):
    frames, bootstrap = inputs
    with pytest.raises(ValueError, match="unknown"):
        run_scan_refinement(
            frames, tmp_path / "bad.h5", bootstrap=bootstrap, modes=["wrong"]
        )
    with pytest.raises(ValueError, match="pixel polish"):
        run_scan_refinement(
            frames, tmp_path / "bad.h5", bootstrap=bootstrap, pixel_polish=True
        )


def test_solve_cli_routes_band_options_and_failure_status(monkeypatch):
    from typer.testing import CliRunner
    from subhkl.io.parser import app

    captured = {}

    def run(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "not_converged"}

    monkeypatch.setattr("subhkl.scan_refine.run_scan_refinement", run)
    result = CliRunner().invoke(
        app,
        [
            "solve",
            "frames.h5",
            "out.h5",
            "--refinement",
            "band",
            "--bootstrap",
            "seed.h5",
            "--runs",
            "2,3",
            "--stable-runs",
            "2",
            "--refine-modes",
            "cylindrical,per_run",
        ],
    )
    assert result.exit_code == 1
    assert "solve failed: 1" not in result.output
    assert captured["runs"] == [2, 3]
    assert captured["modes"] == ["cylindrical", "per_run"]
    assert captured["finder"] is None and not captured["pixel_polish"]


def test_cg4d_omega_future_metadata_polarity():
    settings = reduction_settings["CG4D"]["Goniometer"]
    np.testing.assert_array_equal(settings["CG4D:Mot:omega"], [0, 1, 0, -1])
    # At kappa=0 the tilt/antitilt cancel and all remaining rotations are vertical.
    axes = np.array(list(settings.values()))
    angles = [20, 50, 0, 50, 0]
    expected = calc_goniometer_rotation_matrix([[0, 1, 0, -1]], [20])
    np.testing.assert_allclose(
        calc_goniometer_rotation_matrix(axes, angles), expected, atol=1e-12
    )


def test_unseeded_count_path_runs_blind_search_then_refinement(
    inputs, tmp_path, monkeypatch
):
    frames, _ = inputs
    output = tmp_path / "blind_counts.h5"
    seed = tmp_path / "blind_counts.blind.h5"
    seen = []

    def discover(args, *budgets):
        assert args.bootstrap is None
        assert args.finder is None
        seen.append("search_counts")
        with h5py.File(seed, "w") as f:
            f["sample/U"] = np.eye(3)
        return str(seed)

    def refine(args):
        seen.append("refine_counts")
        assert args.bootstrap == args.blind_search == str(seed)
        return {"status": "converged"}

    monkeypatch.setattr("subhkl.search.blind_band.discover", discover)
    monkeypatch.setattr("subhkl.search.scan_band.refine_scan", refine)
    result = run_scan_refinement(frames, output)
    assert result["status"] == "converged"
    assert seen == ["search_counts", "refine_counts"]


def test_missing_bootstrap_runs_blind_search_then_same_refinement(
    inputs, tmp_path, monkeypatch
):
    frames, _ = inputs
    finder = tmp_path / "finder.h5"
    finder.touch()
    output = tmp_path / "blind.h5"
    seed = tmp_path / "blind.blind.h5"
    seen = []

    def discover(args, *budgets):
        assert args.bootstrap is None
        assert args.finder == finder
        seen.append("search")
        with h5py.File(seed, "w") as f:
            f["sample/U"] = np.eye(3)
        return str(seed)

    def refine(args):
        seen.append("refine")
        assert args.bootstrap == args.blind_search == str(seed)
        return {"status": "converged"}

    monkeypatch.setattr("subhkl.search.blind_band.discover", discover)
    monkeypatch.setattr("subhkl.search.scan_band.refine_scan", refine)
    result = run_scan_refinement(frames, output, finder=finder)
    assert result["status"] == "converged"
    assert seen == ["search", "refine"]


def test_generated_search_file_cannot_overwrite_finder(inputs, tmp_path):
    frames, _ = inputs
    finder = tmp_path / "result.blind.h5"
    finder.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="overwrite"):
        run_scan_refinement(frames, tmp_path / "result.h5", finder=finder)
    assert finder.read_bytes() == b"do not overwrite"


def test_mandi_256_panel_multi_axis_goniometer_and_trigonal_cell(tmp_path):
    frames, bootstrap = tmp_path / "mandi_counts.h5", tmp_path / "mandi_seed.h5"
    axes = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    angles = np.array([[26.0, 90.0, 0.0], [26.0, 90.0, 20.0]])
    counts = np.full((2, 256, 256), 3.0)
    counts[0, :32, :32] = 1.0
    counts[1, :32, :32] = 5.0
    with h5py.File(frames, "w") as f:
        f["images"] = counts
        f["bank_ids"] = [20, 20]
        f["file_offsets"] = [0, 1]
        f["goniometer/axes"] = axes
        f["goniometer/angles"] = angles
        f["goniometer/names"] = [b"omega", b"chi", b"phi"]
        for k, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"),
            [61.5, 61.5, 95.9, 90.0, 90.0, 120.0],
        ):
            f[f"sample/{k}"] = v
        f["sample/space_group"] = "P3_221"
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = np.eye(3)

    obs = load_observations(
        SimpleNamespace(
            nexus=frames,
            finder=None,
            runs=[0, 1],
            stable_runs=[0],
            binning=32,
            static_mask_file=None,
        )
    )
    assert len(obs[0]) == 128
    assert obs[1].min() == obs[2].min() == 15.5
    assert obs[1].max() == obs[2].max() == 239.5

    output = tmp_path / "mandi_band.h5"
    result = run_scan_refinement(
        frames,
        output,
        bootstrap=bootstrap,
        instrument="MANDI",
        wavelength=(2.0, 4.0),
        assignment_wavelength=(2.0, 4.0),
        runs=[0, 1],
        stable_runs=[0],
        modes=["cell", "per_run"],
        maxiter=1,
        n_zones=60,
        binning=32,
    )
    assert result["status"] in ("converged", "not_converged")
    with h5py.File(output) as f:
        assert f.attrs["instrument"] == "MANDI"
        np.testing.assert_allclose(
            f["sample/a"][()], f["sample/b"][()], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            [f["sample/alpha"][()], f["sample/beta"][()], f["sample/gamma"][()]],
            [90.0, 90.0, 120.0],
            atol=1e-12,
        )
        bounds = f["scan_refinement/parameter_bounds"][()]
        # Hexagonal/trigonal cell: only da (slot 15) is free; db (16) and dtheta (17) are locked
        assert bounds[15, 0] < bounds[15, 1]
        np.testing.assert_array_equal(bounds[16], [0.0, 0.0])
        np.testing.assert_array_equal(bounds[17], [0.0, 0.0])
        R, _, runs_out, angles_out, changed = frame_kinematics(
            f, 2, f["file_offsets"][()]
        )
        assert not changed
        np.testing.assert_array_equal(runs_out, [0, 1])
        expected = [
            calc_goniometer_rotation_matrix(f["goniometer/axes"][()], a)
            for a in angles_out
        ]
        np.testing.assert_allclose(R, expected, atol=1e-12)


@pytest.mark.parametrize(
    "cell,sg,expected_free",
    [
        ([11.93, 11.93, 11.93, 90.0, 90.0, 90.0], "Ia-3d", ()),
        ([79.1, 79.1, 37.9, 90.0, 90.0, 90.0], "P4_32_12", (0,)),
        ([18.39, 56.55, 6.54, 90.0, 90.0, 90.0], "Fdd2", (0, 1)),
        ([68.84, 98.07, 90.92, 90.0, 106.11, 90.0], "P21", (0, 1, 2)),
    ],
)
def test_cell_symmetry_constraints(cell, sg, expected_free):
    from subhkl.search.scan_band import _cell_symmetry_spec

    _, active = _cell_symmetry_spec(cell, sg)
    assert active == expected_free


def test_single_run_support_without_file_offsets(tmp_path):
    frames, bootstrap = tmp_path / "single.h5", tmp_path / "seed.h5"
    counts = np.full((1, 256, 256), 3.0)
    counts[0, :32, :32] = 6.0
    cell = [18.39, 56.55, 6.54, 90.0, 90.0, 90.0]
    with h5py.File(frames, "w") as f:
        f["images"] = counts
        f["bank_ids"] = [20]
        f["goniometer/axes"] = np.array([[0.0, 1.0, 0.0, -1.0]])
        f["goniometer/angles"] = np.array([[35.0]])
        for k, v in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            f[f"sample/{k}"] = v
        f["sample/space_group"] = "Fdd2"
        f["instrument/wavelength"] = [1.1, 8.5]
        f.attrs["instrument"] = "MANDI"
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = np.eye(3)

    output = tmp_path / "single_out.h5"
    result = run_scan_refinement(
        frames,
        output,
        bootstrap=bootstrap,
        instrument=None,
        modes=["cell", "per_run", "sample"],
        maxiter=1,
        n_zones=60,
        binning=32,
    )
    assert result["status"] in ("converged", "not_converged")
    with h5py.File(output) as f:
        assert f.attrs["instrument"] == "MANDI"
        np.testing.assert_allclose(f["instrument/wavelength"][()], [1.1, 8.5])
        np.testing.assert_allclose(f["scan_refinement/band_wavelength"][()], [1.1, 8.5])
        np.testing.assert_array_equal(f["file_offsets"][()], [0])
        np.testing.assert_array_equal(f["goniometer/per_run/delta_deg"][()], [0.0])
        bounds = f["scan_refinement/parameter_bounds"][()]
        # Single run: per_run drift (index 18) and sample shift (indices 12..14) are gauge-fixed
        np.testing.assert_array_equal(bounds[12:15], np.zeros((3, 2)))
        np.testing.assert_array_equal(bounds[18], [0.0, 0.0])
        assert bounds[15, 0] < bounds[15, 1]
        assert bounds[16, 0] < bounds[16, 1]
        np.testing.assert_array_equal(bounds[17], [0.0, 0.0])


def test_global_rot_axis_modes(tmp_path):
    frames, bootstrap = tmp_path / "rot_axis.h5", tmp_path / "seed_rot.h5"
    counts = np.full((1, 256, 256), 3.0)
    counts[0, :32, :32] = 6.0
    cell = [18.39, 56.55, 6.54, 90.0, 90.0, 90.0]
    with h5py.File(frames, "w") as f:
        f["images"] = counts
        f["bank_ids"] = [20]
        f["goniometer/axes"] = np.array([[0.0, 1.0, 0.0, -1.0]])
        f["goniometer/angles"] = np.array([[35.0]])
        for k, v in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            f[f"sample/{k}"] = v
        f["sample/space_group"] = "Fdd2"
        f["instrument/wavelength"] = [1.1, 8.5]
        f.attrs["instrument"] = "MANDI"
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = np.eye(3)

    # 1. Default vertical axis (0,1,0): only slot 7 is free; slots 6 and 8 are locked
    out_y = tmp_path / "rot_y.h5"
    run_scan_refinement(
        frames,
        out_y,
        bootstrap=bootstrap,
        instrument="MANDI",
        modes=["global_rot"],
        maxiter=1,
        n_zones=60,
        binning=32,
    )
    with h5py.File(out_y) as f:
        assert f["scan_refinement"].attrs["global_rot_axis"] == "0,1,0"
        bounds = f["scan_refinement/parameter_bounds"][()]
        np.testing.assert_array_equal(bounds[6], [0.0, 0.0])
        assert bounds[7, 0] < bounds[7, 1]
        np.testing.assert_array_equal(bounds[8], [0.0, 0.0])

    # 2. Unconstrained 3D rotation ('none'): all 3 slots 6..8 are free
    out_none = tmp_path / "rot_none.h5"
    run_scan_refinement(
        frames,
        out_none,
        bootstrap=bootstrap,
        instrument="MANDI",
        modes=["global_rot"],
        global_rot_axis="none",
        maxiter=1,
        n_zones=60,
        binning=32,
    )
    with h5py.File(out_none) as f:
        assert f["scan_refinement"].attrs["global_rot_axis"] == "none"
        bounds = f["scan_refinement/parameter_bounds"][()]
        for idx in (6, 7, 8):
            assert bounds[idx, 0] < bounds[idx, 1]

    # 3. Arbitrary diagonal axis ('1,1,0'): 1 free slot (slot 6), rg parallel to (1,1,0)
    out_diag = tmp_path / "rot_diag.h5"
    run_scan_refinement(
        frames,
        out_diag,
        bootstrap=bootstrap,
        instrument="MANDI",
        modes=["global_rot"],
        global_rot_axis="1,1,0",
        maxiter=1,
        n_zones=60,
        binning=32,
    )
    with h5py.File(out_diag) as f:
        bounds = f["scan_refinement/parameter_bounds"][()]
        assert bounds[6, 0] < bounds[6, 1]
        np.testing.assert_array_equal(bounds[7], [0.0, 0.0])
        np.testing.assert_array_equal(bounds[8], [0.0, 0.0])
        rg = f["scan_refinement/parameters"][()][6:9]
        np.testing.assert_allclose(rg[0], rg[1], atol=1e-12)
        assert rg[2] == 0.0


def test_per_run_trans_polish_recovers_shift_and_writes_trans_m(tmp_path):
    from subhkl.config import beamlines
    from subhkl.core.crystallography import generate_reflections
    from subhkl.instrument.detector import Detector
    from subhkl.instrument.physics import predict_reflections_on_panel
    from subhkl.search.scan_band import _polish_per_run_translations, build_B_jax

    cell = [30.0, 35.0, 40.0, 90.0, 90.0, 90.0]
    sg = "P 21 21 21"
    U_true = np.eye(3)
    B_true = np.asarray(build_B_jax(*cell[:3], *np.deg2rad(cell[3:])), dtype=float)
    UB_true = U_true @ B_true
    h, k, l = generate_reflections(*cell, space_group=sg, d_min=2.5)
    axes = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, -1.0]])
    angles_run = [np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 130.0])]
    R_runs_init = [calc_goniometer_rotation_matrix(axes, a) for a in angles_run]
    # Run 0 has zero shift; Run 1 has a +3 mm z / -2 mm y lab-frame sample displacement
    ds_lab_true = [np.zeros(3), np.array([0.0, -0.002, 0.003])]
    bank_ids = sorted(int(k_) for k_ in beamlines["CG4D"].keys())
    dets = {b: Detector(dict(beamlines["CG4D"][str(b)])) for b in bank_ids}
    C_final = np.array([dets[b].center for b in bank_ids])
    uh_final = np.array([dets[b].uhat for b in bank_ids])
    vh_final = np.array([dets[b].vhat for b in bank_ids])
    W_final = np.array([dets[b].width for b in bank_ids])
    H_final = np.array([dets[b].height for b in bank_ids])
    M_bank = np.array([float(dets[b].m - 1) for b in bank_ids])
    N_bank = np.array([float(dets[b].n - 1) for b in bank_ids])

    banks_l, pr_l, pc_l, dev_l, sig_l, runs_l, angs_l, w_l = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for r in (0, 1):
        s_samp_r = R_runs_init[r].T @ ds_lab_true[r]
        for b in bank_ids:
            row, col, _, _, _, _ = predict_reflections_on_panel(
                dets[b],
                h,
                k,
                l,
                UB_true,
                2.8,
                4.5,
                sample_offset=s_samp_r,
                R_all=R_runs_init[r],
            )
            if len(row):
                n = len(row)
                banks_l.append(np.full(n, b, int))
                pr_l.append(row)
                pc_l.append(col)
                dev_l.append(np.full(n, 200.0))
                sig_l.append(np.full(n, 1.5))
                runs_l.append(np.full(n, r, int))
                angs_l.append(np.repeat(angles_run[r][None, :], n, axis=0))
                w_l.append(np.ones(n))
    obs = tuple(
        np.concatenate(v)
        for v in (banks_l, pr_l, pc_l, dev_l, sig_l, runs_l, angs_l, w_l)
    )
    sel = np.ones(len(obs[0]), dtype=bool)
    args = SimpleNamespace(
        runs=[0, 1],
        n_runs=2,
        space_group=sg,
        d_min=2.5,
        assign_wavelength=(2.8, 4.5),
        finder="finder.h5",
    )
    R_out, trans_out, med_b, med_a = _polish_per_run_translations(
        args,
        U_true,
        B_true,
        cell,
        R_runs_init,
        np.zeros(3),
        bank_ids,
        C_final,
        uh_final,
        vh_final,
        W_final,
        H_final,
        M_bank,
        N_bank,
        obs=obs,
        sel=sel,
    )
    assert med_a[1] < 0.35 * med_b[1]
    recovered_lab_1 = R_out[1] @ trans_out[1]
    np.testing.assert_allclose(recovered_lab_1, ds_lab_true[1], atol=4e-4)

    # Verify run_scan_refinement with per_run_trans writes goniometer/per_run/trans_m
    frames, bootstrap = tmp_path / "prt_counts.h5", tmp_path / "prt_seed.h5"
    counts = np.full((2, 512, 512), 3.0)
    counts[0, :32, :32] = 1.0
    counts[1, :32, :32] = 5.0
    with h5py.File(frames, "w") as f:
        f["images"] = counts
        f["bank_ids"] = [62, 62]
        f["file_offsets"] = [0, 1]
        f["goniometer/axes"] = axes
        f["goniometer/angles"] = np.stack(angles_run)
        f["goniometer/names"] = [b"omega", b"chi", b"phi"]
        for k_, v_ in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            f[f"sample/{k_}"] = v_
        f["sample/space_group"] = sg
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = U_true
    out_h5 = tmp_path / "prt_out.h5"
    run_scan_refinement(
        frames,
        out_h5,
        bootstrap=bootstrap,
        modes=["per_run_trans"],
        maxiter=1,
        n_zones=60,
        binning=32,
    )
    with h5py.File(out_h5) as f:
        assert "goniometer/per_run/trans_m" in f
        assert f["goniometer/per_run/trans_m"].shape == (2, 3)
        R_kin, origins_kin, _, _, _ = frame_kinematics(f, 2, f["file_offsets"][()])
        assert R_kin.shape == (2, 3, 3)
        assert origins_kin.shape == (2, 3)
