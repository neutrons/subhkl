"""Multi-run indexing against a merged image stack whose angles are per image.

``merge-images`` stores ``goniometer/angles`` once per panel image (runs are
contiguous blocks marked by ``file_offsets``), while the finder's
``peaks/run_index`` is the run.  Each peak must be fitted with the angles of
its own image: looking the per-image table up by run number gives every run
after the first the angles of an image of run 0.
"""

from unittest.mock import MagicMock, patch

import h5py
import numpy as np

from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix
from subhkl.io.parser import indexer

AXES = np.array([[0.0, 1.0, 0.0, 1.0]])
RUN_ANGLES = [0.0, 30.0]
IMAGES_PER_RUN = 3
HKLS = np.array(
    [
        [1, 0, -2],
        [0, 1, -2],
        [1, 1, -2],
        [2, 0, -2],
        [0, 2, -2],
        [1, 0, -3],
        [0, 1, -3],
        [1, 1, -3],
        [2, 1, -3],
        [1, 2, -3],
    ]
)
CELL = (4.5, 5.0, 5.5, 90.0, 90.0, 90.0)


def _b_matrix():
    from subhkl.optimization import FindUB

    fu = FindUB()
    fu.a, fu.b, fu.c, fu.alpha, fu.beta, fu.gamma = CELL
    return fu.reciprocal_lattice_B()


def _write_inputs(tmp_path):
    """A merged stack (per-image angles) and finder peaks spread over its images."""
    n_images = IMAGES_PER_RUN * len(RUN_ANGLES)
    image_angles = np.repeat(RUN_ANGLES, IMAGES_PER_RUN).reshape(-1, 1)

    stack = tmp_path / "merged.h5"
    with h5py.File(stack, "w") as f:
        f["images"] = np.zeros((n_images, 2, 2), dtype=np.float32)
        f["bank_ids"] = np.ones(n_images, dtype=np.int32)
        f["goniometer/axes"] = AXES
        f["goniometer/angles"] = image_angles
        f["file_offsets"] = np.arange(len(RUN_ANGLES)) * IMAGES_PER_RUN

    B = _b_matrix()
    ki = np.array([0.0, 0.0, 1.0])
    xyz, runs, images, kf_dirs = [], [], [], []
    for run, angle in enumerate(RUN_ANGLES):
        R = calc_goniometer_rotation_matrix(AXES, [angle])
        for j, hkl in enumerate(HKLS):
            Q = 2.0 * np.pi * (R @ B @ hkl)
            lam = -4.0 * np.pi * (ki @ Q) / (Q @ Q)
            kf = (lam / (2.0 * np.pi)) * Q + ki
            xyz.append(kf)
            kf_dirs.append(kf / np.linalg.norm(kf))
            runs.append(run)
            images.append(run * IMAGES_PER_RUN + j % IMAGES_PER_RUN)
    xyz = np.array(xyz)
    n = len(xyz)

    peaks = tmp_path / "finder.h5"
    with h5py.File(peaks, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = CELL[:3]
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = CELL[3:]
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.1, 10.0]
        f["peaks/sigma"] = np.full(n, 2.0)
        f["peaks/radius"] = np.zeros(n)
        f["peaks/pixel_r"] = np.zeros(n)
        f["peaks/pixel_c"] = np.zeros(n)
        f["peaks/run_index"] = np.array(runs, dtype=np.int64)
        f["peaks/image_index"] = np.array(images, dtype=np.int64)
        f["bank"] = np.ones(n, dtype=np.int32)
        f["bank_ids"] = np.ones(n_images, dtype=np.int32)
    return stack, peaks, xyz, np.array(kf_dirs)


def test_multi_run_merged_stack_fits_each_peak_with_its_image_angles(tmp_path):
    stack, peaks, xyz, kf_dirs = _write_inputs(tmp_path)
    out = tmp_path / "indexed.h5"

    with (
        patch("subhkl.instrument.detector.Detector") as mock_detector,
        patch("subhkl.commands.Peaks"),
        patch.dict("subhkl.config.config.beamlines", {"DUMMY": {"1": {}}}),
    ):
        det = MagicMock()
        det.pixel_to_lab.return_value = xyz
        det.pixel_to_angles.return_value = (np.zeros(len(xyz)), np.zeros(len(xyz)))
        mock_detector.return_value = det

        indexer(
            peaks_h5_filename=str(peaks),
            output_peaks_filename=str(out),
            a=CELL[0],
            b=CELL[1],
            c=CELL[2],
            alpha=CELL[3],
            beta=CELL[4],
            gamma=CELL[5],
            space_group="P 1",
            strategy_name="DE",
            population_size=1000,
            gens=1000,
            n_runs=1,
            seed=42,
            sigma_init=3.14,
            instrument_name="DUMMY",
            original_nexus_filename=str(stack),
        )

    with h5py.File(out, "r") as f:
        hkl = np.stack([f["peaks/h"][()], f["peaks/k"][()], f["peaks/l"][()]], 1)
        image = f["peaks/image_index"][()]
        run = f["peaks/run_index"][()]
        R = f["goniometer/R"][()]
        UB = f["sample/U"][()] @ f["sample/B"][()]

    # The output keeps the finder's run numbering and one R per stack image.
    assert run.tolist() == np.repeat([0, 1], len(HKLS)).tolist()
    assert R.shape[0] == IMAGES_PER_RUN * len(RUN_ANGLES)

    # Every peak, in both runs, is indexed and explained by its own image's R.
    assert np.all(np.any(hkl != 0, axis=1))
    ki = np.array([0.0, 0.0, 1.0])
    for r in range(len(RUN_ANGLES)):
        sel = run == r
        q_obs = kf_dirs[sel] - ki
        q_fit = np.einsum("nij,jk,nk->ni", R[image[sel]], UB, hkl[sel])
        cos = np.sum(q_obs * q_fit, 1) / (
            np.linalg.norm(q_obs, axis=1) * np.linalg.norm(q_fit, axis=1)
        )
        err = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        assert np.max(err) < 0.05, f"run {r}: max angular error {np.max(err):.3f} deg"


def _write_metrics_input(path, rotation_rows):
    with h5py.File(path, "w") as f:
        f["peaks/run_index"] = np.array([0, 0, 1, 1])
        f["peaks/image_index"] = np.array([0, 2, 3, 5])
        f["goniometer/R"] = np.repeat(np.eye(3)[None], rotation_rows, axis=0)


def test_metrics_reads_per_image_rotations_at_each_peaks_image(tmp_path):
    from subhkl.instrument.metrics import per_image_geometry_index

    p = tmp_path / "indexed.h5"
    _write_metrics_input(p, rotation_rows=6)  # one R per stack image
    with h5py.File(p, "r") as f:
        geom = per_image_geometry_index(f, f["peaks/run_index"][()], 4)
    assert geom.tolist() == [0, 2, 3, 5]


def test_metrics_keeps_run_lookup_for_per_run_and_per_peak_tables(tmp_path):
    from subhkl.instrument.metrics import per_image_geometry_index

    for rows in (2, 4):  # one R per run; one R per peak
        p = tmp_path / f"indexed_{rows}.h5"
        _write_metrics_input(p, rotation_rows=rows)
        with h5py.File(p, "r") as f:
            geom = per_image_geometry_index(f, f["peaks/run_index"][()], 4)
        assert geom.tolist() == [0, 0, 1, 1]
