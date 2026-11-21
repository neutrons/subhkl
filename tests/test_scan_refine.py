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
    bank_ids = sorted(int(k_) for k_ in beamlines["CG4D"])
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


def test_default_band_binning_preserves_at_least_32x32_bins_regardless_of_frame_count(
    tmp_path, monkeypatch
):
    from typer.testing import CliRunner

    from subhkl.io.parser import app
    from subhkl.scan_refine import default_band_binning

    for n_rows, n_cols, expected in [
        (512, 512, 8),
        (256, 256, 8),
        (128, 128, 4),
        (64, 64, 2),
        (32, 32, 1),
        (5000, 1800, 8),
    ]:
        b = default_band_binning(n_rows, n_cols)
        assert b == expected
        assert n_rows % b == 0 and n_cols % b == 0
        assert n_rows // b >= 32 and n_cols // b >= 32

    # >240 256x256 MANDI panels must still resolve binning=8 (32x32 bins/panel), never 16
    frames = tmp_path / "mandi_many_frames.h5"
    bootstrap = tmp_path / "seed.h5"
    n_frames = 250
    with h5py.File(frames, "w") as f:
        f.attrs["instrument"] = "MANDI"
        f.create_dataset("images", shape=(n_frames, 256, 256), dtype=np.float32)
        f["bank_ids"] = np.full(n_frames, 20, dtype=np.int32)
        f["file_offsets"] = np.array([0], dtype=np.int64)
        f["goniometer/axes"] = np.array([[0.0, 1.0, 0.0, 1.0]])
        f["goniometer/angles"] = np.zeros((n_frames, 1), dtype=np.float64)
        for k, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"),
            [69.12, 98.43, 90.92, 90.07, 105.93, 90.01],
        ):
            f[f"sample/{k}"] = v
        f["sample/space_group"] = "P 21"
        f["instrument/wavelength"] = [2.0, 4.0]
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = np.eye(3)

    seen_binning = []

    def fake_refine_scan(args):
        seen_binning.append(int(args.binning))
        return {"status": "converged"}

    monkeypatch.setattr("subhkl.search.scan_band.refine_scan", fake_refine_scan)
    out = tmp_path / "out.h5"
    res = CliRunner().invoke(
        app,
        [
            "solve",
            str(frames),
            str(out),
            "--instrument",
            "MANDI",
            "--bootstrap",
            str(bootstrap),
            "--refinement",
            "band",
        ],
    )
    assert res.exit_code == 0, res.output
    assert seen_binning == [8]


def test_long_axis_auto_binning_ewald_stratification_and_gimbal_lock(
    tmp_path, monkeypatch
):
    from scipy.spatial.transform import Rotation as Rot

    from subhkl.config import beamlines
    from subhkl.core.crystallography import generate_reflections
    from subhkl.instrument.detector import Detector
    from subhkl.scan_refine import _decompose_R_to_angles, default_band_binning
    from subhkl.search.scan_band import (
        _ewald_d_min,
        _make_ewald_polisher,
        _zone_funnel_seed_per_run,
        build_B_jax,
    )

    # 1. Long-axis P 61 2 2 cell (a=b=80, c=238, d_min=1.41) selects binning=4 on 256x256 panels
    cell_hex = [80.0, 80.0, 238.0, 90.0, 90.0, 120.0]
    assert default_band_binning(256, 256, cell=cell_hex, d_min=1.41) == 4
    assert default_band_binning(512, 512, cell=cell_hex, d_min=1.41) == 4

    frames = tmp_path / "mandi_p6122.h5"
    bootstrap = tmp_path / "seed_p6122.h5"
    with h5py.File(frames, "w") as f:
        f.attrs["instrument"] = "MANDI"
        f.create_dataset("images", shape=(2, 256, 256), dtype=np.float32)
        f["bank_ids"] = np.array([19, 19], dtype=np.int32)
        f["file_offsets"] = np.array([0, 1], dtype=np.int64)
        f["goniometer/axes"] = np.array([[0.0, 1.0, 0.0, 1.0]])
        f["goniometer/angles"] = np.array([[0.0], [15.0]], dtype=np.float64)
        for k, v in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell_hex):
            f[f"sample/{k}"] = v
        f["sample/space_group"] = "P 61 2 2"
        f["instrument/wavelength"] = [2.0, 4.5]
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = np.eye(3)

    seen_binning = []
    monkeypatch.setattr(
        "subhkl.search.scan_band.refine_scan",
        lambda args: seen_binning.append(int(args.binning)) or {"status": "converged"},
    )
    run_scan_refinement(
        frames, tmp_path / "out_p6122.h5", bootstrap=bootstrap, instrument="MANDI"
    )
    assert seen_binning == [4]

    # 2. _ewald_d_min accounts for hexagonal metric determinant and Bravais centering
    d_p = _ewald_d_min(cell_hex, 1.41, space_group="P 61 2 2", max_refl=450000)
    d_ortho = _ewald_d_min(
        [80.0, 80.0, 238.0, 90.0, 90.0, 90.0],
        1.41,
        space_group="P 2 2 2",
        max_refl=450000,
    )
    assert 1.41 <= d_p < d_ortho
    d_f = _ewald_d_min(
        [120.0, 120.0, 240.0, 90.0, 90.0, 90.0],
        1.2,
        space_group="F d d 2",
        max_refl=200000,
    )
    d_prim = _ewald_d_min(
        [120.0, 120.0, 240.0, 90.0, 90.0, 90.0],
        1.2,
        space_group="P 2 2 2",
        max_refl=200000,
    )
    assert d_f < d_prim

    # 3. _make_ewald_polisher retains high-|G| shells for backscattering banks
    B_hex = np.asarray(build_B_jax(*cell_hex[:3], *np.deg2rad(cell_hex[3:])), float)
    hkl = np.stack(
        generate_reflections(*cell_hex, space_group="P 61 2 2", d_min=2.0), axis=1
    ).astype(np.float64)
    bank_ids = [19, 32]
    dets = {b: Detector(dict(beamlines["MANDI"][str(b)])) for b in bank_ids}
    exc_maps = {(0, b): np.zeros((128, 128), dtype=np.float32) for b in bank_ids}
    eval_runs, _, _ = _make_ewald_polisher(
        hkl @ B_hex.T,
        [0],
        {0: np.eye(3)},
        bank_ids,
        np.array([dets[b].center for b in bank_ids]),
        np.array([dets[b].uhat for b in bank_ids]),
        np.array([dets[b].vhat for b in bank_ids]),
        np.array([dets[b].width for b in bank_ids]),
        np.array([dets[b].height for b in bank_ids]),
        np.array([float(dets[b].m - 1) for b in bank_ids]),
        np.array([float(dets[b].n - 1) for b in bank_ids]),
        exc_maps,
        (2.0, 4.5),
        b2=2,
        max_reflections=15000,
    )
    kept_q = np.linalg.norm((hkl[eval_runs.keep_idx] @ B_hex.T), axis=1)
    assert len(eval_runs.keep_idx) == 15000
    assert float(kept_q.max()) > 0.45

    # 4. _zone_funnel_seed_per_run captures a >2 deg inter-batch crystal slip
    U_true_slip = Rot.from_rotvec(np.radians([2.1, -0.8, 1.2])).as_matrix()
    A_hex = np.linalg.inv(B_hex).T
    q_synth = []
    for uvw in ([1, 0, 0], [0, 1, 0], [1, 1, 0]):
        axis_lab = U_true_slip @ (A_hex @ np.array(uvw, float))
        axis_lab /= np.linalg.norm(axis_lab)
        # Build 6 scattering unit vectors orthogonal to each zone axis
        ref_v = (
            np.array([0.0, 0.0, 1.0])
            if abs(axis_lab[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        e1 = np.cross(axis_lab, ref_v)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis_lab, e1)
        for ph in np.linspace(-0.8, 0.8, 6):
            q_synth.append(np.cos(ph) * e1 + np.sin(ph) * e2)
    q_arr = np.asarray(q_synth, dtype=np.float64)
    Urs_seeded = _zone_funnel_seed_per_run(
        np.eye(3),
        B_hex,
        [0],
        {0: np.eye(3)},
        {0: q_arr},
        {0: np.ones(len(q_arr))},
        space_group="P 61 2 2",
    )
    assert Urs_seeded is not None
    err_deg = np.degrees(Rot.from_matrix(Urs_seeded[0] @ U_true_slip.T).magnitude())
    assert err_deg < 0.25

    # 5. 3-axis Y-Z-Y goniometer decomposition escapes chi=0 gimbal lock exactly
    axes_yzy = np.array(
        [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    )
    ang_nom = np.array([0.0, 0.0, 10.0])
    R_nom = calc_goniometer_rotation_matrix(axes_yzy, ang_nom)
    R_slipped = Rot.from_rotvec(np.radians([3.5, -1.2, 2.0])).as_matrix() @ R_nom
    ang_fit = _decompose_R_to_angles(axes_yzy, ang_nom, R_slipped)
    R_fit = calc_goniometer_rotation_matrix(axes_yzy, ang_fit)
    np.testing.assert_allclose(R_fit, R_slipped, atol=1e-10)


def test_cell_symmetry_spec_rhombohedral_h_vs_r():
    from subhkl.search.scan_band import _cell_symmetry_spec

    assert _cell_symmetry_spec([12.0, 12.0, 28.0, 90.0, 90.0, 120.0], "R 3") == (
        "hexagonal",
        (0,),
    )
    assert _cell_symmetry_spec([12.0, 12.0, 28.0, 90.0, 90.0, 120.0], "R -3 m :H") == (
        "hexagonal",
        (0,),
    )
    assert _cell_symmetry_spec([8.5, 8.5, 8.5, 55.2, 55.2, 55.2], "R 3:R") == (
        "rhombohedral",
        (2,),
    )
    assert _cell_symmetry_spec([8.5, 8.5, 8.5, 55.2, 55.2, 55.2], "R -3 m :R") == (
        "rhombohedral",
        (2,),
    )


def _write_synthetic_cg4d_counts(
    frames_path,
    cell_true,
    space_group,
    U_true,
    angles_per_run,
    axes,
    *,
    R_true_per_run=None,
    radial_scale_true=0.0,
    wl=(2.2, 4.2),
    d_min=2.4,
):
    from subhkl.config import beamlines
    from subhkl.core.crystallography import generate_reflections
    from subhkl.instrument.detector import Detector
    from subhkl.instrument.physics import predict_reflections_on_panel
    from subhkl.search.scan_band import build_B_jax

    bank_ids = sorted(int(k) for k in beamlines["CG4D"])
    dets = {}
    for b in bank_ids:
        cfg = dict(beamlines["CG4D"][str(b)])
        det0 = Detector(cfg)
        if abs(radial_scale_true) > 0.0:
            c_mid = (
                det0.center
                + 0.5 * det0.width * det0.uhat
                + 0.5 * det0.height * det0.vhat
            )
            c_mid_sc = np.array(
                [
                    c_mid[0] * (1.0 + radial_scale_true),
                    c_mid[1] * (1.0 + radial_scale_true),
                    c_mid[2] * (1.0 + radial_scale_true),
                ]
            )
            cfg["center"] = (
                c_mid_sc - 0.5 * det0.width * det0.uhat - 0.5 * det0.height * det0.vhat
            ).tolist()
        dets[b] = Detector(cfg)

    B_true = np.asarray(
        build_B_jax(*cell_true[:3], *np.deg2rad(cell_true[3:])), dtype=np.float64
    )
    UB_true = U_true @ B_true
    h, k, l = generate_reflections(*cell_true, space_group=space_group, d_min=d_min)

    rng = np.random.default_rng(42)
    n_runs = len(angles_per_run)
    n_b = len(bank_ids)
    m_px, n_px = dets[bank_ids[0]].m, dets[bank_ids[0]].n
    images = rng.poisson(2.0, size=(n_runs * n_b, n_px, m_px)).astype(np.float32)
    yy, xx = np.ogrid[:n_px, :m_px]

    for r in range(n_runs):
        R_r = (
            R_true_per_run[r]
            if R_true_per_run is not None
            else calc_goniometer_rotation_matrix(axes, angles_per_run[r])
        )
        for ib, b in enumerate(bank_ids):
            row, col, _, _, _, _ = predict_reflections_on_panel(
                dets[b], h, k, l, UB_true, wl[0], wl[1], R_all=R_r
            )
            img = images[r * n_b + ib]
            for pr_i, pc_i in zip(row, col):
                ri, ci = round(float(pr_i)), round(float(pc_i))
                r0, r1 = max(0, ri - 4), min(n_px, ri + 5)
                c0, c1 = max(0, ci - 4), min(m_px, ci + 5)
                if r1 > r0 and c1 > c0:
                    d2 = (yy[r0:r1, :] - pr_i) ** 2 + (xx[:, c0:c1] - pc_i) ** 2
                    img[r0:r1, c0:c1] += (180.0 * np.exp(-0.5 * d2 / (1.3**2))).astype(
                        np.float32
                    )

    with h5py.File(frames_path, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["images"] = images
        f["bank_ids"] = np.tile(np.array(bank_ids, dtype=np.int32), n_runs)
        f["file_offsets"] = np.arange(n_runs, dtype=np.int64) * n_b
        f["goniometer/axes"] = axes
        f["goniometer/angles"] = np.repeat(
            np.asarray(angles_per_run, dtype=np.float64), n_b, axis=0
        )
        f["goniometer/names"] = [b"omega", b"chi", b"phi"]
        for k_, v_ in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell_true):
            f[f"sample/{k_}"] = v_
        f["sample/space_group"] = space_group
        f["instrument/wavelength"] = list(wl)
    return bank_ids


def test_perturbed_monoclinic_cell_with_slip_and_spurious_shared_geom(tmp_path):
    from scipy.spatial.transform import Rotation as Rot

    cell_true = [32.0, 46.0, 50.0, 90.0, 96.4, 90.0]
    cell_init = [31.55, 45.6, 50.0, 90.0, 95.9, 90.0]
    sg = "I 2"
    U_true = Rot.from_rotvec(np.radians([7.0, -11.0, 5.0])).as_matrix()
    axes = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, -1.0]])
    angles_nom = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 35.0]),
        np.array([0.0, 0.0, 75.0]),
    ]
    R_nom = [calc_goniometer_rotation_matrix(axes, a) for a in angles_nom]
    # Run 0 is anchor; Run 1 slips +0.45 deg, Run 2 slips +1.15 deg
    R_true = [
        R_nom[0],
        Rot.from_rotvec(np.radians([0.0, 0.45, 0.0])).as_matrix() @ R_nom[1],
        Rot.from_rotvec(np.radians([0.15, 1.15, -0.1])).as_matrix() @ R_nom[2],
    ]
    frames = tmp_path / "mono_slip_counts.h5"
    _write_synthetic_cg4d_counts(
        frames, cell_true, sg, U_true, angles_nom, axes, R_true_per_run=R_true
    )

    # Bootstrap mimics blind_band absorbing cell+slip error into fake shared_geometry
    # (>1.5% radial cliff and >1.5 deg U warp), while proposal_orientations stays close.
    U_warped = Rot.from_rotvec(np.radians([1.15, -1.25, 0.45])).as_matrix() @ U_true
    U_prop = Rot.from_rotvec(np.radians([-0.45, 0.55, -0.20])).as_matrix() @ U_true
    bootstrap = tmp_path / "mono_slip_boot.h5"
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = U_warped
        f["search/orientations"] = np.stack([U_warped])
        f["search/proposal_orientations"] = np.stack([U_prop])
        f["search/shared_geometry"] = np.array(
            [0.019, 0.0, -0.009, 0.0, 0.003, 0.0, -0.004], dtype=np.float64
        )
        f["search/shared_axial_scale"] = -0.005

    output = tmp_path / "mono_slip_out.h5"
    res = run_scan_refinement(
        frames,
        output,
        bootstrap=bootstrap,
        instrument="CG4D",
        cell=cell_init,
        space_group=sg,
        wavelength=(2.2, 4.2),
        assignment_wavelength=(2.2, 4.2),
        d_min=2.4,
        maxiter=120,
    )
    assert res["status"] == "converged"
    with h5py.File(output, "r") as f:
        U_sol = f["sample/U"][()]
        c_sol = [
            float(f[f"sample/{k}"][()])
            for k in ("a", "b", "c", "alpha", "beta", "gamma")
        ]
        params = f["scan_refinement/parameters"][()]
        offs = f["file_offsets"][()]
        ang_nom_f = f["goniometer/angles_nominal"][()]
        R_sol, _, _, ang_corr, _ = frame_kinematics(f, len(f["bank_ids"]), offs)

    u_err = min(
        np.degrees(Rot.from_matrix(U_sol @ S @ U_true.T).magnitude())
        for S in (np.eye(3), np.diag([-1.0, 1.0, -1.0]))
    )
    assert u_err < 0.18
    # Anchor run (run 0) goniometer angles remain unshifted
    np.testing.assert_allclose(ang_corr[offs[0]], ang_nom_f[offs[0]], atol=1e-4)
    # Spurious +1.9% radial warp was rejected
    assert abs(float(params[3])) < 0.004
    # Monoclinic cell ratios and beta recovered accurately from perturbed cell_init
    assert abs(c_sol[0] - cell_true[0]) < 0.15
    assert abs(c_sol[1] - cell_true[1]) < 0.20
    assert abs(c_sol[4] - cell_true[4]) < 0.18
    # Slipped run 2 crystal orientation in lab frame is recovered
    S_op = (
        np.eye(3)
        if np.degrees(Rot.from_matrix(U_sol @ U_true.T).magnitude()) < 90.0
        else np.diag([-1.0, 1.0, -1.0])
    )
    err_run2 = np.degrees(
        Rot.from_matrix(
            (R_sol[offs[2]] @ U_sol @ S_op) @ (R_true[2] @ U_true).T
        ).magnitude()
    )
    assert err_run2 < 0.18


def test_genuine_uncalibrated_shared_geom_preserved(tmp_path):
    from scipy.spatial.transform import Rotation as Rot

    cell_true = [28.0, 34.0, 40.0, 90.0, 90.0, 90.0]
    sg = "P 21 21 21"
    U_true = Rot.from_rotvec(np.radians([-6.0, 9.0, -4.0])).as_matrix()
    axes = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, -1.0]])
    angles_nom = [np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 65.0])]
    frames = tmp_path / "uncal_counts.h5"
    _write_synthetic_cg4d_counts(
        frames, cell_true, sg, U_true, angles_nom, axes, radial_scale_true=0.045
    )

    U_seed = Rot.from_rotvec(np.radians([0.3, -0.35, 0.2])).as_matrix() @ U_true
    bootstrap = tmp_path / "uncal_boot.h5"
    with h5py.File(bootstrap, "w") as f:
        f["sample/U"] = U_seed
        f["search/orientations"] = np.stack([U_seed])
        f["search/proposal_orientations"] = np.stack([U_seed])
        f["search/shared_geometry"] = np.array(
            [0.044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        f["search/shared_axial_scale"] = 0.044

    output = tmp_path / "uncal_out.h5"
    res = run_scan_refinement(
        frames,
        output,
        bootstrap=bootstrap,
        instrument="CG4D",
        cell=cell_true,
        space_group=sg,
        wavelength=(2.2, 4.2),
        assignment_wavelength=(2.2, 4.2),
        d_min=2.4,
        maxiter=120,
    )
    assert res["status"] == "converged"
    with h5py.File(output, "r") as f:
        params = f["scan_refinement/parameters"][()]
        U_sol = f["sample/U"][()]
    # Genuine +4.5% radial shift is preserved rather than discarded
    assert abs(float(params[3]) - 0.045) < 0.008
    sym_ops = [
        np.eye(3),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ]
    u_err = min(
        np.degrees(Rot.from_matrix(U_sol @ S @ U_true.T).magnitude()) for S in sym_ops
    )
    assert u_err < 0.20
