"""Systematics metadata columns in the unmerged MTZ export.

The scaling model downstream (careless) can only absorb systematics it
can see.  With companion files the exporter adds per-observation
proxies -- SNAPD (geometry miss for this peak), SIGEFF (projected
radius), DPHI/DTX/DTY/DTZ (the run's fitted goniometer corrections) --
and without them the export is unchanged.
"""

from __future__ import annotations

import gemmi
import h5py
import numpy as np
import pytest

from subhkl.io.export import MTZExporter

N = 6


@pytest.fixture
def files(tmp_path):
    peaks = tmp_path / "integrator.h5"
    with h5py.File(peaks, "w") as f:
        for key, val in (
            ("a", 61.0),
            ("b", 61.0),
            ("c", 96.0),
            ("alpha", 90.0),
            ("beta", 90.0),
            ("gamma", 120.0),
        ):
            f[f"sample/{key}"] = val
        f["sample/space_group"] = b"P 32 2 1"
        f["peaks/h"] = np.array([1, 2, 3, 1, 2, 0])
        f["peaks/k"] = np.array([0, 1, 1, 2, 0, 0])
        f["peaks/l"] = np.array([2, 3, 1, 1, 4, 0])  # last row dropped (000)
        f["peaks/lambda"] = np.full(N, 3.2)
        f["peaks/two_theta"] = np.linspace(0.5, 1.5, N)
        f["peaks/azimuthal"] = np.linspace(-1, 1, N)
        f["peaks/intensity"] = np.arange(N, dtype=float) * 100 + 10
        f["peaks/sigma"] = np.full(N, 5.0)
        f["peaks/bank"] = np.full(N, 61)
        f["peaks/run_index"] = np.array([0, 0, 0, 1, 1, 1])
        f["peaks/image_index"] = np.array([0, 0, 0, 1, 1, 1])
        f["peaks/pixel_r"] = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
        f["peaks/pixel_c"] = np.array([15.0, 25.0, 35.0, 15.0, 25.0, 35.0])
        f["peaks/var_u"] = np.full(N, 16.0)
        f["peaks/var_v"] = np.full(N, 9.0)
        f["peaks/cov_uv"] = np.zeros(N)

    predictions = tmp_path / "predictor.h5"
    with h5py.File(predictions, "w") as f:
        # image 0: predictions offset by (0.6, 0.8) -> SNAPD = 1.0
        f["banks/0/i"] = np.array([10.6, 20.6, 30.6])
        f["banks/0/j"] = np.array([15.8, 25.8, 35.8])
        # image 1: exact -> SNAPD = 0
        f["banks/1/i"] = np.array([10.0, 20.0, 30.0])
        f["banks/1/j"] = np.array([15.0, 25.0, 35.0])

    corrections = tmp_path / "calib.h5"
    with h5py.File(corrections, "w") as f:
        f["goniometer/per_run/delta_deg"] = np.array([-0.1, 0.3])
        f["goniometer/per_run/trans_m"] = np.array(
            [[1e-3, -2e-3, 0.5e-3], [0.0, 4e-4, -1e-3]]
        )

    return peaks, predictions, corrections


def _columns(path):
    mtz = gemmi.read_mtz_file(str(path))
    return {c.label: np.array(c) for c in mtz.columns}


def test_plain_export_has_no_extra_columns(files, tmp_path):
    peaks, _, _ = files
    out = tmp_path / "plain.mtz"
    MTZExporter(str(peaks)).write_mtz(str(out))
    cols = _columns(out)
    assert "SNAPD" not in cols and "DPHI" not in cols and "SIGEFF" not in cols
    assert len(cols["I"]) == N - 1  # hkl=000 row dropped


def test_boundary_marked_rows_are_not_exported(files, tmp_path):
    """sigI = 0 marks a censored (nonneg-boundary) amplitude, not a
    measurement; the exporter must drop it."""
    import h5py

    peaks, _, _ = files
    with h5py.File(peaks, "r+") as f:
        sig = f["peaks/sigma"][()]
        sig[1] = 0.0  # mark the second (valid-hkl) row as censored
        f["peaks/sigma"][...] = sig
    out = tmp_path / "censored.mtz"
    MTZExporter(str(peaks)).write_mtz(str(out))
    cols = _columns(out)
    assert len(cols["I"]) == N - 2  # hkl=000 row AND censored row dropped
    np.testing.assert_allclose(sorted(cols["I"]), [10.0, 210.0, 310.0, 410.0])


def test_metadata_columns_carry_the_right_values(files, tmp_path):
    peaks, predictions, corrections = files
    out = tmp_path / "meta.mtz"
    MTZExporter(
        str(peaks),
        predictions_file=str(predictions),
        corrections_file=str(corrections),
    ).write_mtz(str(out))
    cols = _columns(out)
    # SIGEFF = (16*9)^(1/4)
    np.testing.assert_allclose(cols["SIGEFF"], (16.0 * 9.0) ** 0.25, rtol=1e-5)
    # image 0 rows miss by 1 px, image 1 rows are exact
    np.testing.assert_allclose(cols["SNAPD"][:3], 1.0, atol=1e-5)
    np.testing.assert_allclose(cols["SNAPD"][3:], 0.0, atol=1e-5)
    # run 0 -> -0.1 deg, run 1 -> +0.3 deg; trans in mm
    np.testing.assert_allclose(cols["DPHI"][:3], -0.1, rtol=1e-5)
    np.testing.assert_allclose(cols["DPHI"][3:], 0.3, rtol=1e-5)
    np.testing.assert_allclose(cols["DTX"][:3], 1.0, rtol=1e-5)
    np.testing.assert_allclose(cols["DTZ"][3:], -1.0, rtol=1e-5)
