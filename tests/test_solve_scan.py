"""CLI scan, corrected frame kinematics, and finder-free visualizer contracts."""

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from typer.testing import CliRunner

from subhkl.calibrate import geometry_detectors, write_detector_calibration
from subhkl.io.parser import app
from subhkl.search.poisson_orientation import CountData, OrientationModel
from subhkl.solve import run_solve
from subhkl.solve_frames import frame_kinematics


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    path = tmp_path_factory.mktemp("count-scan")
    frame = path / "frames.h5"
    boot = path / "boot.h5"
    bg = path / "bg.h5"
    banks = np.array([62, 55, 43, 84, 104, 94, 95, 86])
    U = Rotation.random(random_state=17).as_matrix()
    g = np.array([0.025, 0.008, -0.006, 0.012, 0.001, -0.0015, 0.002])
    R = Rotation.from_rotvec(np.radians([[0, 2.5, 0], [0, 36.7, 0]])).as_matrix()
    shifts = np.array([[0.001, 0.002, 0], [0.002, 0.002, 0.001]])
    origins = np.einsum("nij,nj->ni", R, shifts)
    rng = np.random.default_rng(4)
    images = []
    dets = geometry_detectors("CG4D", g, banks)
    for r, origin in zip(R, origins):
        im = np.zeros((len(banks), 512, 512))
        data = CountData.from_images(im, banks, np.full(len(banks), 0.5), 16)
        model = OrientationModel(
            data,
            "CG4D",
            [12] * 3 + [90] * 3,
            "P 1",
            (2, 10),
            2,
            6,
            detectors=dets,
            sample_origin=origin,
        )
        y = rng.poisson(
            data.background
            + model.design((r @ U)[None], np.zeros(6)) @ np.full(model.n_hkl, 3000.0)
        )
        for bank, panel in zip(banks, im):
            panel[::16, ::16] = y[data.pixel_indices[bank]]
        images.append(im)
    with h5py.File(frame, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["images"] = np.concatenate(images)
        f["bank_ids"] = np.tile(banks, 2)
        f["file_offsets"] = [0, len(banks)]
        f["files"] = np.array(["run-a.h5", "run-b.h5"], dtype=h5py.string_dtype())
        for key, val in zip(
            ["a", "b", "c", "alpha", "beta", "gamma"], [12] * 3 + [90] * 3
        ):
            f["sample/" + key] = val
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [2, 10]
        f["goniometer/axes"] = [[0, 1, 0, 1]]
        f["goniometer/names"] = np.array(["phi"], dtype=h5py.string_dtype())
        f["goniometer/angles"] = np.repeat([[0.0], [35.0]], len(banks), axis=0)
        f["goniometer/offsets/phi"] = 2.0
        f["goniometer/translations"] = [0.001, 0.002, 0]
        f["goniometer/per_run/frame_to_run"] = np.repeat([0, 1], len(banks))
        f["goniometer/per_run/motor"] = "phi"
        f["goniometer/per_run/delta_deg"] = [0.5, -0.3]
        f["goniometer/per_run/trans_m"] = [[0, 0, 0], [0.001, 0, 0.001]]
    with h5py.File(boot, "w") as f:
        f["sample/U"] = U
        write_detector_calibration(f, dets)
    with h5py.File(bg, "w") as f:
        f["bank_ids"] = banks
        f["images"] = np.full((len(banks), 512, 512), 0.5)
    return frame, boot, bg, U, R, origins


def test_scan_cli_and_corrected_metadata_round_trip(scan, tmp_path):
    frame, boot, bg, U, R, origins = scan
    output = tmp_path / "solve.h5"
    args = [
        "solve",
        str(frame),
        str(output),
        "--bootstrap",
        str(boot),
        "--background-file",
        str(bg),
        "--binning",
        "16",
        "--d-min",
        "2",
        "--no-refine",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    with h5py.File(output) as f:
        assert f["solve/g"].shape == (7,)
        assert f["solve/orientations_lab"].shape == (2, 1, 3, 3)
        np.testing.assert_allclose(f["sample/U"][()], U)
        np.testing.assert_allclose(
            f["goniometer/R"][()], np.repeat(R, 8, axis=0), atol=1e-12
        )
        np.testing.assert_allclose(
            f["solve/sample_origin_lab"][()], np.repeat(origins, 8, axis=0)
        )
        np.testing.assert_array_equal(
            f["solve/reflection_setting"][()],
            np.repeat([0, 1], len(f["solve/hkl"]) // 2),
        )
        reread = frame_kinematics(f, 16, f["file_offsets"][()])
        np.testing.assert_allclose(reread[0], f["goniometer/R"][()])
        np.testing.assert_allclose(reread[1], f["solve/sample_origin_lab"][()])
    again = tmp_path / "again.h5"
    run_solve(
        frame,
        again,
        metadata=output,
        bootstrap=output,
        background_file=bg,
        binning=16,
        d_min=2,
        do_refine=False,
        verbose=False,
    )
    with h5py.File(output) as a, h5py.File(again) as b:
        for name in [
            "goniometer/R",
            "goniometer/angles",
            "solve/sample_origin_lab",
            "sample/U",
            "solve/objective_final",
        ]:
            np.testing.assert_allclose(a[name][()], b[name][()])


def test_scan_visualizer_without_peak_table(scan, tmp_path, monkeypatch):
    from subhkl.commands import run_indexer_visualize
    import subhkl.viz.solve as viz

    frame, boot, bg, *_ = scan
    output = tmp_path / "solve.h5"
    run_solve(
        frame,
        output,
        bootstrap=boot,
        background_file=bg,
        binning=16,
        d_min=2,
        do_refine=False,
        verbose=False,
    )
    calls = []
    monkeypatch.setattr(
        viz,
        "plot_unrolled_detector",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    paths = run_indexer_visualize(
        str(output), images_filename=str(frame), output_dir=str(tmp_path), dpi=50
    )
    assert len(paths) == len(calls) == 2
    assert paths[0].endswith("run-a-index.png") and paths[1].endswith("run-b-index.png")
    assert list(calls[0][0][1]) == list(range(8))
    assert list(calls[1][0][1]) == list(range(8, 16))
    assert any(curve["points"] for curve in calls[0][1]["zone_curves"])
    calls.clear()
    paths = run_indexer_visualize(str(output), image_index=9, output_dir=str(tmp_path))
    assert len(paths) == 1 and list(calls[0][0][1]) == [9]


def test_scan_joint_proposals_transform_q_but_preserve_bragg_angle(
    scan, monkeypatch, tmp_path
):
    import subhkl.search.spherical as spherical

    frame, boot, bg, U, R, _ = scan
    captured = []

    def ladder(D, w, B, **kwargs):
        captured.append((D, w, kwargs["sin_theta"]))
        return [(U, 1.0, None)]

    monkeypatch.setattr(spherical, "lattice_ladder", ladder)
    output = tmp_path / "blind.h5"
    # Proposal data uses nominal geometry, while candidate intensity fitting
    # remains the same checked solver. No bootstrap orientation is supplied.
    run_solve(
        frame,
        output,
        metadata=frame,
        background_file=bg,
        binning=16,
        d_min=2,
        do_refine=False,
        verbose=False,
    )
    D, w, sin = captured[0]
    assert len(D) == len(w) == len(sin)
    assert np.any(w < 0)
    assert not np.allclose(sin, -D[:, 2])


def test_roll_changes_predictions_in_a_scan(scan):
    from subhkl.search.multi_setting import MultiSettingModel

    frame, _, _, U, R, _ = scan
    with h5py.File(frame) as f:
        models = []
        for lo in [0, 8]:
            d = CountData.from_images(
                f["images"][lo : lo + 8], f["bank_ids"][lo : lo + 8], np.ones(8), 16
            )
            models.append(
                OrientationModel(d, "CG4D", [12] * 3 + [90] * 3, "P 1", (2, 10), 2)
            )
    model = MultiSettingModel(models, R, free_roll=True)
    baseline = model.design(U[None], np.zeros(7))
    rolled = model.design(U[None], np.array([0, 0, 0, 0.01, 0, 0, 0]))
    assert model.geometry_size == 7
    assert np.linalg.norm((baseline - rolled).data) > 1e-3


def test_seven_parameter_refinement_retains_budget_diagnostics(scan, tmp_path):
    frame, boot, bg, *_ = scan
    output = tmp_path / "refine.h5"
    result = run_solve(
        frame,
        output,
        bootstrap=boot,
        background_file=bg,
        binning=16,
        d_min=2,
        max_evals=1,
        verbose=False,
    )
    assert result["g"].shape == (7,)
    with h5py.File(output) as f:
        assert f["solve/n_evaluations"][()] >= 11
        assert f["solve/geometry_parameter_names"][3] == b"roll_z"
        if result["status"] != "converged":
            assert "sample/U" not in f


def test_unconverged_visualizations_are_labeled(scan, tmp_path, monkeypatch):
    from subhkl.commands import run_indexer_visualize
    import subhkl.viz.solve as viz

    frame, boot, bg, *_ = scan
    output = tmp_path / "failed.h5"
    run_solve(
        frame,
        output,
        bootstrap=boot,
        background_file=bg,
        binning=16,
        d_min=2,
        do_refine=False,
        verbose=False,
    )
    with h5py.File(output, "r+") as f:
        del f["sample/U"]
        f["solve"].attrs["status"] = "not_converged"
    calls = []
    monkeypatch.setattr(viz, "plot_unrolled_detector", lambda *a, **k: calls.append(a))
    paths = run_indexer_visualize(str(output), image_index=0)
    assert "diagnostic-not_converged" in paths[0]
    assert calls[0][0].diagnostic_status == "not_converged"


def test_zone_curves_use_corrected_sample_origin(scan):
    from subhkl.instrument.detector import Detector
    from subhkl.viz.zones import zone_curve_points

    _, _, _, U, R, origins = scan
    dets = geometry_detectors("CG4D", np.zeros(7), [62, 55, 43, 84])
    origin = origins[1]
    shifted = {
        bank: Detector(dict(det.config, center=det.center - origin))
        for bank, det in dets.items()
    }
    B = np.eye(3) / 12
    absolute = zone_curve_points(dets, U, B, R_gonio=R[1], sample_origin=origin)
    relative = zone_curve_points(shifted, U, B, R_gonio=R[1])
    count = 0
    for a, b in zip(absolute, relative):
        for bank, points in a["points"].items():
            np.testing.assert_allclose(points, b["points"][bank] + origin, atol=1e-12)
            count += len(points)
    assert count > 0
