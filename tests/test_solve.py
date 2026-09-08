"""Unified command contracts and real-panel count recovery without a finder."""

import copy

import h5py
import numpy as np
import pytest
from scipy.special import ndtr
from scipy.spatial.transform import Rotation
from typer.testing import CliRunner

from subhkl.calibrate import geometry_detectors, write_detector_calibration
from subhkl.config import beamlines
from subhkl.io.parser import app
from subhkl.search.poisson_orientation import (
    CountData,
    OrientationModel,
    geometry_vector,
)
from subhkl.search.profiles import RadialProfile
from subhkl.solve import run_solve


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    path = tmp_path_factory.mktemp("solve-counts")
    rng = np.random.default_rng(17)
    banks = [62, 55, 43, 84, 104, 94, 95, 86]
    images = np.zeros((len(banks), 512, 512))
    data = CountData.from_images(images, banks, [0.5] * len(banks), 8)
    model = OrientationModel(
        data, "CG4D", [12] * 3 + [90] * 3, "P 1", (2.0, 10.0), 2.0, 6.0
    )
    U = Rotation.random(3, random_state=rng).as_matrix()
    g = np.array([0.025, 0.008, -0.006, 0.001, -0.0015, 0.002])
    A = model.design(U, g)
    flux = np.zeros((len(U), model.n_hkl))
    flux[:2] = rng.lognormal(0.0, 0.5, (2, model.n_hkl)) * 600.0
    y = rng.poisson(data.background + A @ flux.ravel())
    # Count-conserving embedding: only the 8x8 bin sums are consumed here.
    for b, im in zip(banks, images):
        im[::8, ::8] = y[data.pixel_indices[b]]
    frame = path / "counts.h5"
    R = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    with h5py.File(frame, "w") as f:
        f["images"], f["bank_ids"] = images, banks
        f.attrs["instrument"] = "CG4D"
        for k, v in zip(("a", "b", "c", "alpha", "beta", "gamma"), [12] * 3 + [90] * 3):
            f[f"sample/{k}"] = v
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [2.0, 10.0]
        f["goniometer/R"] = R[None]
    bootstrap = path / "bootstrap.h5"
    with h5py.File(bootstrap, "w") as f:
        f["orientations"] = U
        write_detector_calibration(
            f, geometry_detectors("CG4D", geometry_vector(g), banks)
        )
    bg = path / "background.h5"
    with h5py.File(bg, "w") as f:
        f["bank_ids"] = banks[::-1]
        f["images"] = np.full_like(images, 0.5)
    return frame, bootstrap, bg, U, R, g


def test_cli_has_only_unified_entry_point():
    runner = CliRunner()
    out = runner.invoke(app, ["--help"])
    assert out.exit_code == 0, out.output
    names = [
        c.name or c.callback.__name__.replace("_", "-") for c in app.registered_commands
    ]
    assert "solve" in names
    assert "calibrate" not in names and "spherical-index" not in names
    out = runner.invoke(app, ["solve", "--help"])
    assert out.exit_code == 0, out.output
    assert "--profile-file" in out.output


def test_solve_writes_all_components_and_sample_frame_without_global_mutation(
    scene, tmp_path
):
    frame, bootstrap, bg, U, R, g = scene
    nominal = copy.deepcopy(beamlines["CG4D"])
    out = tmp_path / "result.h5"
    result = run_solve(
        frame,
        out,
        bootstrap=bootstrap,
        background_file=bg,
        d_min=2.0,
        do_refine=False,
        verbose=False,
    )
    assert result["status"] == "converged"
    np.testing.assert_array_equal(result["active"], [0, 1])
    assert beamlines["CG4D"] == nominal
    with h5py.File(out) as f:
        np.testing.assert_allclose(f["solve/orientations_sample"][()], R.T @ U)
        primary = int(f["solve/primary_orientation"][()])
        np.testing.assert_allclose(f["sample/U"][()], R.T @ U[primary])
        assert f["solve/intensities"].shape[0] == 3
        assert f["solve/hkl"].shape[1] == 3
        expected = geometry_detectors("CG4D", geometry_vector(g))
        np.testing.assert_allclose(
            f["detector_calibration/bank_62/center"][()], expected[62].center
        )
    # Saved absolute geometry is a reusable baseline; applying it twice must
    # not double the correction or change the lab-frame orientations.
    again = tmp_path / "again.h5"
    run_solve(
        frame,
        again,
        bootstrap=out,
        background_file=bg,
        d_min=2.0,
        do_refine=False,
        verbose=False,
    )
    with h5py.File(out) as a, h5py.File(again) as b:
        np.testing.assert_allclose(a["sample/U"][()], b["sample/U"][()])
        np.testing.assert_allclose(
            a["detector_calibration/bank_62/center"][()],
            b["detector_calibration/bank_62/center"][()],
        )


def test_cli_executes_count_workflow(scene, tmp_path):
    frame, bootstrap, bg, *_ = scene
    out = tmp_path / "cli.h5"
    result = CliRunner().invoke(
        app,
        [
            "solve",
            str(frame),
            str(out),
            "--bootstrap",
            str(bootstrap),
            "--background-file",
            str(bg),
            "--d-min",
            "2",
            "--no-refine",
        ],
    )
    assert result.exit_code == 0, result.output
    with h5py.File(out) as f:
        assert f["solve"].attrs["status"] == "converged"


def test_unaddressed_rotations_are_rejected(scene, tmp_path):
    frame, *_ = scene
    metadata = tmp_path / "mixed.h5"
    with h5py.File(frame) as src, h5py.File(metadata, "w") as f:
        src.copy("sample", f)
        src.copy("instrument", f)
        f.attrs["instrument"] = "CG4D"
        f["goniometer/R"] = [np.eye(3), Rotation.from_rotvec([0.1, 0, 0]).as_matrix()]
    with pytest.raises(ValueError, match="frame-addressed"):
        run_solve(frame, tmp_path / "bad.h5", metadata=metadata, do_refine=False)


def test_output_cannot_overwrite_input(scene):
    frame, *_ = scene
    with pytest.raises(ValueError, match="overwrite"):
        run_solve(frame, frame)


def test_empty_support_has_diagnostics_but_no_bootstrap_orientation(scene, tmp_path):
    frame, bootstrap, bg, *_ = scene
    empty = tmp_path / "empty.h5"
    with h5py.File(frame) as src, h5py.File(empty, "w") as dst:
        for key in src:
            src.copy(key, dst)
        dst.attrs["instrument"] = "CG4D"
        dst["images"][...] = 0
    out = tmp_path / "null-result.h5"
    result = CliRunner().invoke(
        app,
        [
            "solve",
            str(empty),
            str(out),
            "--bootstrap",
            str(bootstrap),
            "--background-file",
            str(bg),
            "--d-min",
            "2",
            "--no-refine",
        ],
    )
    assert result.exit_code == 1, result.output
    with h5py.File(out) as f:
        assert f["solve"].attrs["status"] == "no_orientation"
        assert len(f["solve/active"]) == 0
        assert "sample/U" not in f


def test_unconverged_refinement_does_not_export_a_valid_orientation(scene, tmp_path):
    frame, bootstrap, bg, *_ = scene
    out = tmp_path / "unfinished.h5"
    result = run_solve(
        frame,
        out,
        bootstrap=bootstrap,
        background_file=bg,
        d_min=2.0,
        max_evals=1,
        verbose=False,
    )
    assert result["status"] == "not_converged"
    with h5py.File(out) as f:
        assert "sample/U" not in f
        assert len(f["solve/active"]) > 0


def test_empirical_gaussian_prior_matches_integrated_gaussian():
    u = np.linspace(0, 6, 601)
    profile = RadialProfile(u, np.exp(-0.5 * u * u))
    r = c = np.arange(-10, 11)
    actual = profile.integrate_bins(r, c, 0.23, -0.37, 1.3)
    vr = ndtr((r + 1 - 0.23) / 1.3) - ndtr((r - 0.23) / 1.3)
    vc = ndtr((c + 1 + 0.37) / 1.3) - ndtr((c + 0.37) / 1.3)
    np.testing.assert_allclose(actual, np.outer(vr, vc), atol=2e-6)
    assert abs(actual.sum() - 1) < 1e-4
    assert profile.integrate_bins(r[r >= 0], c, 0.23, -0.37, 1.3).sum() < 1


@pytest.mark.parametrize("sigma", [0.04, 0.2])
def test_narrow_profile_conserves_flux_in_coarse_bins(sigma):
    u = np.linspace(0, 6, 601)
    profile = RadialProfile(u, np.exp(-0.5 * u * u))
    bins = np.arange(-2, 3)
    actual = profile.integrate_bins(bins, bins, 0.03, 0.43, sigma)
    vr = ndtr((bins + 1 - 0.03) / sigma) - ndtr((bins - 0.03) / sigma)
    vc = ndtr((bins + 1 - 0.43) / sigma) - ndtr((bins - 0.43) / sigma)
    np.testing.assert_allclose(actual, np.outer(vr, vc), atol=1e-5)
    assert abs(actual.sum() - 1) < 1e-5


def test_profile_is_used_and_persisted(scene, tmp_path):
    frame, bootstrap, bg, *_ = scene
    prior = tmp_path / "profile.h5"
    u = np.linspace(0, 6, 301)
    with h5py.File(prior, "w") as f:
        f["profile/u"], f["profile/f"] = u, np.exp(-0.5 * u * u)
    out = tmp_path / "profile-result.h5"
    run_solve(
        frame,
        out,
        bootstrap=bootstrap,
        background_file=bg,
        profile_file=prior,
        d_min=2.0,
        do_refine=False,
        verbose=False,
    )
    with h5py.File(out) as f:
        np.testing.assert_array_equal(f["profile/u"][()], u)
        np.testing.assert_array_equal(f["solve/active"][()], [0, 1])


@pytest.mark.parametrize("u,f", [([0, 0], [1, 0]), ([0, 1], [1, -1]), ([0, 1], [0, 0])])
def test_invalid_profile_rejected(u, f):
    with pytest.raises(ValueError, match="profile"):
        RadialProfile(u, f)


@pytest.mark.parametrize("inner_failure", [False, True])
def test_failed_intensity_fit_retains_diagnostics(
    scene, tmp_path, monkeypatch, inner_failure
):
    import subhkl.solve as workflow
    import subhkl.search.poisson_orientation as engine
    from dataclasses import replace

    original = engine.fit_intensities

    def fail(*args, **kwargs):
        return replace(original(*args, **kwargs), converged=False, kkt=123.0)

    monkeypatch.setattr(engine if inner_failure else workflow, "fit_intensities", fail)
    frame, bootstrap, bg, *_ = scene
    output = tmp_path / "failed.h5"
    result = run_solve(
        frame, output, bootstrap=bootstrap, background_file=bg, d_min=2, verbose=False
    )
    assert result["status"] == "not_converged"
    with h5py.File(output) as f:
        assert "sample/U" not in f
        assert f["solve/orientations_lab"].shape == (3, 3, 3)
        assert f["solve/stationarity_residual"][()] == 123
        assert not f["solve/inner_converged"][()]
