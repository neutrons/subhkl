"""Tests for subhkl -> DIALS ExperimentList/reflection_table export.

The reader, filename, and error-path tests run everywhere. The full round-trip
tests that actually build and read back ``.expt``/``.refl`` files are skipped
unless the DIALS/dxtbx toolkit is importable.
"""

import h5py
import numpy as np
import pytest

from subhkl.io import dials_export as dx
from subhkl.io import dials_imageset as di


# ---------------------------------------------------------------------------
# Filename derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base,expected",
    [
        ("output.h5", ("output.expt", "output.refl")),
        ("dir/run.nxs.h5", ("dir/run.expt", "dir/run.refl")),
        ("indexed.hdf5", ("indexed.expt", "indexed.refl")),
        ("peaks.mtz", ("peaks.expt", "peaks.refl")),
        ("noext", ("noext.expt", "noext.refl")),
    ],
)
def test_dials_output_paths(base, expected):
    assert dx.dials_output_paths(base) == expected


def test_dials_output_paths_prefix_override():
    assert dx.dials_output_paths("output.h5", prefix="custom") == (
        "custom.expt",
        "custom.refl",
    )


# ---------------------------------------------------------------------------
# Missing-DIALS error path
# ---------------------------------------------------------------------------


def test_require_dials_message():
    try:
        import dxtbx  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("DIALS is installed; error path not exercised")
    with pytest.raises(ImportError) as exc:
        dx._require_dials()
    assert "conda" in str(exc.value)


# ---------------------------------------------------------------------------
# HDF5 readers
# ---------------------------------------------------------------------------


def _write_finder(path):
    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["instrument/wavelength"] = [2.0, 10.0]
        f["peaks/intensity"] = np.array([10.0, 20.0, 30.0])
        f["peaks/sigma"] = np.array([1.0, 2.0, 3.0])
        f["peaks/pixel_r"] = np.array([5.0, 6.0, 7.0])
        f["peaks/pixel_c"] = np.array([1.0, 2.0, 3.0])
        f["bank"] = np.array([11, 12, 11])
        f["peaks/run_index"] = np.array([0, 0, 1])
        f["goniometer/axes"] = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        f["goniometer/angles"] = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])


def _write_indexed(path):
    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["instrument/wavelength"] = [2.0, 10.0]
        f["sample/a"] = 10.0
        f["sample/b"] = 11.0
        f["sample/c"] = 12.0
        f["sample/alpha"] = 90.0
        f["sample/beta"] = 90.0
        f["sample/gamma"] = 90.0
        f["sample/space_group"] = b"P 1"
        f["sample/U"] = np.eye(3)
        # B for an orthorhombic 10x11x12 cell (1/Angstrom, upper triangular).
        f["sample/B"] = np.diag([1 / 10.0, 1 / 11.0, 1 / 12.0])
        f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
        f["peaks/h"] = np.array([1, 2, -1])
        f["peaks/k"] = np.array([0, 1, 1])
        f["peaks/l"] = np.array([1, 1, 2])
        f["peaks/lambda"] = np.array([3.0, 4.0, 5.0])
        f["peaks/intensity"] = np.array([100.0, 200.0, 300.0])
        f["peaks/sigma"] = np.array([10.0, 20.0, 30.0])
        f["peaks/pixel_r"] = np.array([5.0, 6.0, 7.0])
        f["peaks/pixel_c"] = np.array([1.0, 2.0, 3.0])
        f["bank"] = np.array([11, 12, 11])
        f["peaks/run_index"] = np.array([0, 0, 1])
        f["goniometer/axes"] = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        f["goniometer/angles"] = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])


def _write_integrated(path):
    """Integrator-style output: carries peaks/xyz and peaks/lambda for s1."""
    _write_indexed(path)
    with h5py.File(path, "a") as f:
        # Lab-frame positions in metres (roughly +Z detector).
        f["peaks/xyz"] = np.array(
            [[0.01, 0.02, 0.30], [-0.02, 0.01, 0.28], [0.00, -0.03, 0.31]]
        )


def _write_integrated_with_offset(path, offset):
    """Integrator output with a nonzero sample position offset (metres)."""
    _write_integrated(path)
    with h5py.File(path, "a") as f:
        f["goniometer/translations"] = np.asarray(offset, dtype=float)


def _write_calibrated(path):
    """Indexed output that also carries a refined detector_calibration group."""
    _write_indexed(path)
    with h5py.File(path, "a") as f:
        g = f.create_group("detector_calibration/bank_11")
        g["center"] = np.array([0.1, 0.2, -0.3])
        g["uhat"] = np.array([1.0, 0.0, 0.0])
        g["vhat"] = np.array([0.0, 1.0, 0.0])
        g["width"] = 0.2
        g["height"] = 0.2


def _write_reduce_stack(path):
    """Single goniometer setting, two banks (11, 12) as separate images."""
    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["instrument/wavelength"] = [2.0, 10.0]
        f["images"] = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        f["bank_ids"] = np.array([11, 12], dtype=np.int32)
        f["goniometer/angles"] = np.array([[0.0], [0.0]])
        f["goniometer/axes"] = np.array([[0, 1, 0, 1]], dtype=float)


def _write_merged_stack(path):
    """Two goniometer settings (frames) x two banks, via file_offsets."""
    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["instrument/wavelength"] = [2.0, 10.0]
        f["images"] = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
        f["bank_ids"] = np.array([11, 12, 11, 12], dtype=np.int32)
        # First two images belong to frame 0, last two to frame 1.
        f["file_offsets"] = np.array([0, 2], dtype=np.int64)
        f["goniometer/angles"] = np.array([[0.0], [0.0], [10.0], [10.0]])
        f["goniometer/axes"] = np.array([[0, 1, 0, 1]], dtype=float)


def _write_predictor(path):
    with h5py.File(path, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["instrument/wavelength"] = [2.0, 10.0]
        f["bank_ids"] = np.array([11, 12])
        g = f.create_group("banks/0")
        g["i"] = np.array([1.0, 2.0])
        g["j"] = np.array([3.0, 4.0])
        g["h"] = np.array([1, 2])
        g["k"] = np.array([0, 1])
        g["l"] = np.array([1, 1])
        g["wavelength"] = np.array([3.0, 4.0])
        g2 = f.create_group("banks/1")
        g2["i"] = np.array([5.0])
        g2["j"] = np.array([6.0])
        g2["h"] = np.array([2])
        g2["k"] = np.array([2])
        g2["l"] = np.array([2])
        g2["wavelength"] = np.array([5.0])


def test_collect_flat_peaks(tmp_path):
    p = tmp_path / "finder.h5"
    _write_finder(str(p))
    with h5py.File(p, "r") as f:
        peaks = dx._collect_flat_peaks(f)
    assert peaks["n"] == 3
    assert peaks["bank"].tolist() == [11, 12, 11]
    assert peaks["run_index"].tolist() == [0, 0, 1]
    assert peaks["h"] is None


def test_collect_predicted_peaks(tmp_path):
    p = tmp_path / "pred.h5"
    _write_predictor(str(p))
    with h5py.File(p, "r") as f:
        peaks = dx._collect_predicted_peaks(f)
    assert peaks["n"] == 3
    assert peaks["bank"].tolist() == [11, 11, 12]
    assert peaks["run_index"].tolist() == [0, 0, 1]
    assert peaks["predicted"] is True


def test_frame_rotations_from_axes_angles(tmp_path):
    p = tmp_path / "finder.h5"
    _write_finder(str(p))
    with h5py.File(p, "r") as f:
        peaks = dx._collect_flat_peaks(f)
        rot, axes = dx._frame_rotations(f, peaks, [0, 1])
    assert set(rot) == {0, 1}
    # Both are valid rotation matrices (det ~ 1).
    for R in rot.values():
        assert R.shape == (3, 3)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6
    # Frame 0 has zero angles -> identity.
    assert np.allclose(rot[0], np.eye(3))


def test_read_detector_calibration(tmp_path):
    p = tmp_path / "calibrated.h5"
    _write_calibrated(str(p))
    with h5py.File(p, "r") as f:
        calib = dx._read_detector_calibration(f)
    assert set(calib) == {11}
    assert np.allclose(calib[11]["center"], [0.1, 0.2, -0.3])
    assert np.allclose(calib[11]["uhat"], [1.0, 0.0, 0.0])
    assert calib[11]["width"] == 0.2


def test_read_detector_calibration_absent(tmp_path):
    p = tmp_path / "finder.h5"
    _write_finder(str(p))
    with h5py.File(p, "r") as f:
        assert dx._read_detector_calibration(f) == {}


def test_read_goniometer_offsets_flat(tmp_path):
    p = tmp_path / "off.h5"
    with h5py.File(p, "w") as f:
        f["goniometer/offsets"] = np.array([1.0, 2.0, 3.0])
    with h5py.File(p, "r") as f:
        off = dx._read_goniometer_offsets(f, 3, None)
    assert np.allclose(off, [1.0, 2.0, 3.0])


def test_read_goniometer_offsets_group(tmp_path):
    p = tmp_path / "off.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("goniometer/offsets")
        g["omega"] = 1.5
        g["phi"] = -0.5
    with h5py.File(p, "r") as f:
        off = dx._read_goniometer_offsets(f, 3, ["omega", "chi", "phi"])
    # chi has no stored offset -> stays zero; omega/phi map by name.
    assert np.allclose(off, [1.5, 0.0, -0.5])


def test_read_goniometer_offsets_absent(tmp_path):
    p = tmp_path / "off.h5"
    with h5py.File(p, "w") as f:
        f["goniometer/axes"] = np.eye(3)
    with h5py.File(p, "r") as f:
        assert dx._read_goniometer_offsets(f, 3, None) is None


def test_read_sample_offset(tmp_path):
    p = tmp_path / "off.h5"
    with h5py.File(p, "w") as f:
        f["goniometer/translations"] = np.array([0.001, -0.002, 0.0005])
    with h5py.File(p, "r") as f:
        off = dx._read_sample_offset(f)
    assert np.allclose(off, [0.001, -0.002, 0.0005])


def test_read_sample_offset_zero_is_none(tmp_path):
    p = tmp_path / "off.h5"
    with h5py.File(p, "w") as f:
        f["goniometer/translations"] = np.zeros(3)
    with h5py.File(p, "r") as f:
        assert dx._read_sample_offset(f) is None


def test_read_sample_offset_per_axis_uses_innermost(tmp_path):
    p = tmp_path / "off.h5"
    with h5py.File(p, "w") as f:
        f["goniometer/translations"] = np.array([[0.0, 0.0, 0.0], [0.01, 0.02, 0.03]])
    with h5py.File(p, "r") as f:
        off = dx._read_sample_offset(f)
    assert np.allclose(off, [0.01, 0.02, 0.03])


def test_read_sample_offset_absent(tmp_path):
    p = tmp_path / "finder.h5"
    _write_finder(str(p))
    with h5py.File(p, "r") as f:
        assert dx._read_sample_offset(f) is None


def test_image_stack_layout_single_frame(tmp_path):
    p = tmp_path / "reduced.h5"
    _write_reduce_stack(str(p))
    with h5py.File(p, "r") as f:
        layout = di.read_image_stack_layout(f)
    assert layout["frame_ids"] == [0]
    assert layout["panel_bank_ids"] == [11, 12]
    assert layout["row_of"] == {(0, 0): 0, (0, 1): 1}


def test_image_stack_layout_multi_frame(tmp_path):
    p = tmp_path / "merged.h5"
    _write_merged_stack(str(p))
    with h5py.File(p, "r") as f:
        layout = di.read_image_stack_layout(f)
        angle_rows = di.frame_angle_rows(f, layout)
    assert layout["frame_ids"] == [0, 1]
    assert layout["panel_bank_ids"] == [11, 12]
    # frame 1's images (rows 2,3) carry the 10-degree setting.
    assert layout["row_of"][(1, 0)] == 2
    assert np.allclose(angle_rows[0], [0.0])
    assert np.allclose(angle_rows[1], [10.0])


def test_write_and_read_image_container(tmp_path):
    stack = tmp_path / "merged.h5"
    _write_merged_stack(str(stack))
    container = tmp_path / "container.h5"
    with h5py.File(stack, "r") as f:
        layout = di.read_image_stack_layout(f)
        di.write_image_container(str(container), f["images"], layout, "CG4D")

    assert di.is_image_container(str(container))
    assert di.container_num_frames(str(container)) == 2
    assert di.container_panel_ids(str(container)) == [11, 12]

    # Frame 1, panel 0 (bank 11) came from image row 2 of the stack.
    with h5py.File(stack, "r") as f:
        expected = np.asarray(f["images"][2])
    panels = di.container_frame(str(container), 1)
    assert len(panels) == 2
    assert np.allclose(panels[0], expected)


def test_is_image_container_false(tmp_path):
    p = tmp_path / "finder.h5"
    _write_finder(str(p))
    assert di.is_image_container(str(p)) is False


def test_frame_rotations_apply_offsets(tmp_path):
    """A goniometer/offsets zero-point shifts the reconstructed rotation."""
    from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix

    p = tmp_path / "pred.h5"
    with h5py.File(p, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["instrument/wavelength"] = [2.0, 10.0]
        # Predictor-style: raw angles + separate offsets, no goniometer/R.
        f["bank_ids"] = np.array([11])
        g = f.create_group("banks/0")
        g["i"] = np.array([1.0])
        g["j"] = np.array([2.0])
        g["h"] = np.array([1])
        g["k"] = np.array([0])
        g["l"] = np.array([1])
        g["wavelength"] = np.array([3.0])
        f["goniometer/axes"] = np.array([[0, 1, 0, 1]], dtype=float)
        f["goniometer/angles"] = np.array([[10.0]])
        f["goniometer/offsets"] = np.array([5.0])
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("goniometer/names", data=["omega"], dtype=dt)

    with h5py.File(p, "r") as f:
        peaks = dx._collect_predicted_peaks(f)
        rot, _ = dx._frame_rotations(f, peaks, [0])

    # Rotation must match angle 10 + offset 5 = 15 degrees, not 10.
    expected = calc_goniometer_rotation_matrix(
        np.array([[0, 1, 0, 1]], dtype=float), np.array([15.0])
    )
    wrong = calc_goniometer_rotation_matrix(
        np.array([[0, 1, 0, 1]], dtype=float), np.array([10.0])
    )
    assert np.allclose(rot[0], expected)
    assert not np.allclose(rot[0], wrong)


# ---------------------------------------------------------------------------
# Full round-trip (requires DIALS)
# ---------------------------------------------------------------------------


def test_roundtrip_indexed(tmp_path):
    pytest.importorskip("dxtbx")
    from dxtbx.model.experiment_list import ExperimentListFactory
    from dials.array_family import flex

    h5 = tmp_path / "indexed.h5"
    _write_indexed(str(h5))
    expt_path = str(tmp_path / "indexed.expt")
    refl_path = str(tmp_path / "indexed.refl")

    dx.hdf5_to_dials(str(h5), expt_path, refl_path, instrument="CG4D")

    experiments = ExperimentListFactory.from_json_file(expt_path, check_format=False)
    # One experiment per frame (run_index 0 and 1).
    assert len(experiments) == 2
    for expt in experiments:
        assert expt.beam is not None
        assert expt.detector is not None
        assert expt.crystal is not None
        assert expt.goniometer is not None

    refl = flex.reflection_table.from_file(refl_path)
    assert len(refl) == 3
    assert "miller_index" in refl
    assert "wavelength" in refl
    assert "intensity.sum.value" in refl
    assert "panel" in refl
    # Reflection ids map onto the two experiments.
    assert set(refl["id"]) == {0, 1}


def test_roundtrip_predictor(tmp_path):
    pytest.importorskip("dxtbx")
    from dials.array_family import flex

    h5 = tmp_path / "pred.h5"
    _write_predictor(str(h5))
    expt_path = str(tmp_path / "pred.expt")
    refl_path = str(tmp_path / "pred.refl")

    dx.hdf5_to_dials(str(h5), expt_path, refl_path, instrument="CG4D")

    refl = flex.reflection_table.from_file(refl_path)
    assert len(refl) == 3
    assert "xyzcal.px" in refl
    assert refl.get_flags(refl.flags.predicted).count(True) == 3


def test_roundtrip_s1_magnitude(tmp_path):
    """s1 must be a wavevector with |s1| = 1/wavelength, not a lab position."""
    pytest.importorskip("dxtbx")
    from dials.array_family import flex

    h5 = tmp_path / "integrated.h5"
    _write_integrated(str(h5))
    expt_path = str(tmp_path / "integrated.expt")
    refl_path = str(tmp_path / "integrated.refl")

    dx.hdf5_to_dials(str(h5), expt_path, refl_path, instrument="CG4D")

    refl = flex.reflection_table.from_file(refl_path)
    assert "s1" in refl
    wavelengths = [3.0, 4.0, 5.0]
    for s1, wl in zip(refl["s1"], wavelengths):
        assert abs((s1[0] ** 2 + s1[1] ** 2 + s1[2] ** 2) ** 0.5 - 1.0 / wl) < 1e-6


def test_roundtrip_sample_offset_shifts_detector(tmp_path):
    """A sample offset shifts each frame's detector by -R @ offset (millimetres)."""
    pytest.importorskip("dxtbx")
    from dxtbx.model.experiment_list import ExperimentListFactory

    offset = np.array([0.001, -0.002, 0.003])  # metres

    base_h5 = tmp_path / "integrated.h5"
    _write_integrated(str(base_h5))
    base_expt = str(tmp_path / "integrated.expt")
    dx.hdf5_to_dials(
        str(base_h5), base_expt, str(tmp_path / "integrated.refl"), instrument="CG4D"
    )
    base = ExperimentListFactory.from_json_file(base_expt, check_format=False)

    off_h5 = tmp_path / "offset.h5"
    _write_integrated_with_offset(str(off_h5), offset)
    off_expt = str(tmp_path / "offset.expt")
    dx.hdf5_to_dials(
        str(off_h5), off_expt, str(tmp_path / "offset.refl"), instrument="CG4D"
    )
    shifted = ExperimentListFactory.from_json_file(off_expt, check_format=False)

    # Frame 0 has identity rotation, so s_lab = offset; origins shift by -offset mm.
    base_origin = np.array(base[0].detector[0].get_origin())
    shifted_origin = np.array(shifted[0].detector[0].get_origin())
    assert np.allclose(shifted_origin, base_origin - offset * 1000.0)


def test_roundtrip_reduce_has_imageset(tmp_path):
    """A reduce/merge stack exports one experiment per frame with an ImageSet."""
    pytest.importorskip("dxtbx")
    from dxtbx.model.experiment_list import ExperimentListFactory

    stack = tmp_path / "merged.h5"
    _write_merged_stack(str(stack))
    expt_path = str(tmp_path / "merged.expt")
    dx.hdf5_to_dials(
        str(stack), expt_path, str(tmp_path / "merged.refl"), instrument="CG4D"
    )

    experiments = ExperimentListFactory.from_json_file(expt_path, check_format=True)
    assert len(experiments) == 2  # two goniometer settings
    for expt in experiments:
        assert expt.imageset is not None
        assert expt.detector is not None
        # Two banks -> two panels of raw data.
        raw = expt.imageset.get_raw_data(0)
        assert len(raw) == 2


def test_roundtrip_detector_calibration(tmp_path):
    """The refined detector_calibration group overrides the nominal geometry."""
    pytest.importorskip("dxtbx")
    from dxtbx.model.experiment_list import ExperimentListFactory

    h5 = tmp_path / "calibrated.h5"
    _write_calibrated(str(h5))
    expt_path = str(tmp_path / "calibrated.expt")
    refl_path = str(tmp_path / "calibrated.refl")

    dx.hdf5_to_dials(str(h5), expt_path, refl_path, instrument="CG4D")

    experiments = ExperimentListFactory.from_json_file(expt_path, check_format=False)
    panel = experiments[0].detector[0]  # bank 11 is the first panel
    origin = panel.get_origin()  # millimetres
    assert np.allclose(origin, [100.0, 200.0, -300.0])
