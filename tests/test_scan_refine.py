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
