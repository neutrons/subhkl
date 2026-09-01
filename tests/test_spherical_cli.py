"""spherical-index and indexer-visualize, end to end on real CG4D geometry.

A synthetic crystal is ray-traced onto the actual CG4D panel bank (the 80
flat panels of beamlines.json), written as a finder-style peaks file, and
recovered through the same command functions the CLI drives.  No data
downloads: the instrument definition ships with the package.
"""

from __future__ import annotations

import os
from itertools import product

import h5py
import numpy as np

from subhkl.commands import run_indexer_visualize, run_spherical_index
from subhkl.config import beamlines
from subhkl.core.crystallography import (
    cartesian_matrix_metric_tensor,
    generate_reflections,
)
from subhkl.instrument.detector import Detector
from subhkl.search.spherical import _quat_angle


def _cubic_rots():
    Rs = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for sg in product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s_) in enumerate(zip(perm, sg)):
                M[i, p] = s_
            if np.linalg.det(M) > 0.5:
                Rs.append(M)
    return Rs


def _synthetic_finder_file(path, rng):
    a = 8.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    h, k, l_ = generate_reflections(a, a, a, 90, 90, 90, space_group="P 1", d_min=1.3)
    G = np.stack([h, k, l_], axis=1) @ B.T
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    U_true = Q * np.sign(np.linalg.det(Q))

    d_lab = dirs @ U_true.T
    d_lab = d_lab[d_lab[:, 2] < -0.05]
    kf = np.array([0.0, 0.0, 1.0]) - 2.0 * d_lab[:, 2:3] * d_lab
    kf /= np.linalg.norm(kf, axis=1, keepdims=True)

    banks, rows, cols = [], [], []
    dets = {int(kk): Detector(v) for kk, v in beamlines["CG4D"].items()}
    for bk, det in dets.items():
        mask, r_, c_ = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
        if not np.any(mask):
            continue
        banks.append(np.full(int(np.sum(mask)), bk))
        rows.append(r_[mask] + rng.normal(scale=0.5, size=int(np.sum(mask))))
        cols.append(c_[mask] + rng.normal(scale=0.5, size=int(np.sum(mask))))
    bank = np.concatenate(banks)
    n = len(bank)
    with h5py.File(path, "w") as fp:
        fp["bank"] = bank
        fp["peaks/pixel_r"] = np.concatenate(rows)
        fp["peaks/pixel_c"] = np.concatenate(cols)
        fp["peaks/image_index"] = np.zeros(n, dtype=int)
        fp["peaks/run_index"] = np.zeros(n, dtype=int)
        fp["goniometer/R"] = np.tile(np.eye(3), (n, 1, 1))
        for kk, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"), (a, a, a, 90.0, 90.0, 90.0)
        ):
            fp[f"sample/{kk}"] = v
        fp["sample/space_group"] = "P 1"
        fp["instrument/wavelength"] = np.array([2.0, 10.0])
        fp.attrs["instrument"] = "CG4D"
    return U_true, n


def test_spherical_index_and_zone_overlay(tmp_path):
    rng = np.random.default_rng(23)
    peaks_file = str(tmp_path / "finder.h5")
    U_true, n = _synthetic_finder_file(peaks_file, rng)
    assert n > 100  # the scene actually covers the instrument

    out_file = str(tmp_path / "spherical.h5")
    run_spherical_index(peaks_file, out_file, d_min=1.3, kernel_deg=1.0)
    with h5py.File(out_file) as fp:
        U = fp["sample/U"][()]
        z = fp["spherical/z"][()]
        assert "sample/B" in fp and "beam/ki_vec" in fp  # bootstrap contract
        q_med = float(fp["spherical/quality/median_deviation_deg"][()])
        q_null = float(fp["spherical/quality/null_median_deviation_deg"][()])
        q_frac = float(fp["spherical/quality/matched_fraction"][()])
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 0.5
    assert z[0] > 10.0
    # the native quality report: accuracy far inside its own null floor
    assert q_med < 0.3
    assert q_null > 3.0 * q_med
    assert q_frac > 0.8

    written = run_indexer_visualize(out_file, output_dir=str(tmp_path), max_index=1)
    assert len(written) == 1
    assert os.path.exists(written[0])
    assert os.path.getsize(written[0]) > 10_000


def test_raw_count_indexing_and_image_backed_overlay(tmp_path):
    """Raw-count mode end to end: synthetic Poisson frames on real CG4D
    panels, no usable peak list (the metadata file carries three dummy
    peaks), orientation recovered from the counts alone; then the zone
    overlay rendered on top of the raw frames."""
    rng = np.random.default_rng(29)
    peaks_file = str(tmp_path / "finder.h5")
    U_true, _ = _synthetic_finder_file(peaks_file, rng)

    # rebuild the spot list from the truth and paint frames for the six
    # busiest banks
    a = 8.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    h, k, l_ = generate_reflections(a, a, a, 90, 90, 90, space_group="P 1", d_min=1.3)
    G = np.stack([h, k, l_], axis=1) @ B.T
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    d_lab = dirs @ U_true.T
    d_lab = d_lab[d_lab[:, 2] < -0.05]
    kf = np.array([0.0, 0.0, 1.0]) - 2.0 * d_lab[:, 2:3] * d_lab
    kf /= np.linalg.norm(kf, axis=1, keepdims=True)
    dets = {int(kk): Detector(v) for kk, v in beamlines["CG4D"].items()}
    hits = {}
    for bk, det in dets.items():
        mask, r_, c_ = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
        if np.any(mask):
            hits[bk] = (r_[mask].astype(int), c_[mask].astype(int), det)
    busiest = sorted(hits, key=lambda b: -len(hits[b][0]))[:6]

    frames, bank_ids = [], []
    for bk in busiest:
        r_, c_, det = hits[bk]
        im = rng.poisson(0.2, size=(det.n, det.m)).astype(np.int32)
        ok = (r_ >= 0) & (r_ < det.n) & (c_ >= 0) & (c_ < det.m)
        np.add.at(im, (r_[ok], c_[ok]), 30)
        frames.append(im)
        bank_ids.append(bk)
    images_file = str(tmp_path / "merged.h5")
    with h5py.File(images_file, "w") as fp:
        fp["images"] = np.stack(frames)
        fp["bank_ids"] = np.array(bank_ids)
        fp["file_offsets"] = np.array([0])

    # overwrite the peaks with three dummies: raw mode must not need them
    with h5py.File(peaks_file, "a") as fp:
        for key in (
            "bank",
            "peaks/pixel_r",
            "peaks/pixel_c",
            "peaks/image_index",
            "peaks/run_index",
            "goniometer/R",
        ):
            del fp[key]
        fp["bank"] = np.array([busiest[0]] * 3)
        fp["peaks/pixel_r"] = np.array([10.0, 20.0, 30.0])
        fp["peaks/pixel_c"] = np.array([10.0, 20.0, 30.0])
        fp["peaks/image_index"] = np.zeros(3, dtype=int)
        fp["peaks/run_index"] = np.zeros(3, dtype=int)
        fp["goniometer/R"] = np.tile(np.eye(3), (3, 1, 1))

    out_file = str(tmp_path / "spherical_raw.h5")
    run_spherical_index(
        peaks_file,
        out_file,
        d_min=1.3,
        kernel_deg=1.0,
        images_filename=images_file,
        binning=8,
        runs=[0],  # exercises the frame-side run filter too
    )
    with h5py.File(out_file) as fp:
        U = fp["sample/U"][()]
        z = fp["spherical/z"][()]
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 1.0
    assert z[0] > 5.0
    with h5py.File(out_file) as fp:
        aligned = float(fp["spherical/quality/aligned_fraction"][()])
        aligned_med = float(fp["spherical/quality/aligned_median_deg"][()])
    # The null-subtracted statistic sees through the noise background.  In
    # this scene the spots carry only ~2% of the positive-excess weight
    # (6k spot counts against ~270k Poisson upward fluctuations), so a
    # small aligned fraction IS the truth -- what matters is that the
    # aligned component is detected at all and located sharply, while the
    # plain median (dominated by the null-distributed noise weight) would
    # say nothing.
    assert 0.02 < aligned < 0.2
    assert aligned_med < 0.8

    written = run_indexer_visualize(
        out_file, output_dir=str(tmp_path), max_index=1, images_filename=images_file
    )
    assert os.path.exists(written[0])
    assert os.path.getsize(written[0]) > 30_000  # images actually rendered


def test_fit_gonio_offsets_and_gauge_detection(tmp_path):
    """The per-run DPHI analogue end to end: a true offset on an axis with
    varying inner angles is recovered; an axis whose inner angles never
    vary is reported as gauge, never as a fitted number.  (Validated on
    real data too: MANDI garnet recovers the production pipeline's refined
    omega to 0.08 deg; CG4D's omega is structurally gauge because its
    kappa = 0 scan makes the omega and phi axes collinear.)"""
    from subhkl.instrument.refinables import gonio_rotation_jax

    rng = np.random.default_rng(41)
    a = 8.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    h, k, l_ = generate_reflections(a, a, a, 90, 90, 90, space_group="P 1", d_min=1.3)
    G = np.stack([h, k, l_], axis=1) @ B.T
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    U_true = Q * np.sign(np.linalg.det(Q))

    # two axes: outer z (offset +0.8 deg, inner varies -> identifiable),
    # inner x (offset never identifiable: nothing inner to it varies)
    axes = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    off_true = np.array([0.8, 0.0])
    run_angles = np.array([[10.0, 0.0], [10.0, 35.0], [10.0, 70.0]])

    dets = {int(kk): Detector(v) for kk, v in beamlines["CG4D"].items()}
    recs = {k2: [] for k2 in ("bank", "pr", "pc", "img", "run", "R", "ang")}
    for r in range(3):
        Rr_true = np.asarray(gonio_rotation_jax(axes, run_angles[r], off_true))
        Rr_nom = np.asarray(gonio_rotation_jax(axes, run_angles[r], np.zeros(2)))
        d_lab = np.vstack([dirs, -dirs]) @ (Rr_true @ U_true).T
        d_lab = d_lab[d_lab[:, 2] < -0.05]
        kf = np.array([0.0, 0.0, 1.0]) - 2.0 * d_lab[:, 2:3] * d_lab
        kf /= np.linalg.norm(kf, axis=1, keepdims=True)
        for bk, det in dets.items():
            mask, r_, c_ = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
            n_ = int(np.sum(mask))
            if n_ == 0:
                continue
            recs["bank"].append(np.full(n_, bk))
            recs["pr"].append(r_[mask] + rng.normal(scale=0.5, size=n_))
            recs["pc"].append(c_[mask] + rng.normal(scale=0.5, size=n_))
            recs["img"].append(np.full(n_, r))
            recs["run"].append(np.full(n_, r))
            recs["R"].append(np.tile(Rr_nom, (n_, 1, 1)))
            recs["ang"].append(np.tile(run_angles[r], (n_, 1)))

    peaks_file = str(tmp_path / "finder.h5")
    with h5py.File(peaks_file, "w") as fp:
        fp["bank"] = np.concatenate(recs["bank"])
        fp["peaks/pixel_r"] = np.concatenate(recs["pr"])
        fp["peaks/pixel_c"] = np.concatenate(recs["pc"])
        fp["peaks/image_index"] = np.concatenate(recs["img"])
        fp["peaks/run_index"] = np.concatenate(recs["run"])
        fp["goniometer/R"] = np.concatenate(recs["R"])
        fp["goniometer/angles"] = np.concatenate(recs["ang"])
        fp["goniometer/axes"] = axes
        fp["goniometer/names"] = np.array([b"outer_z", b"inner_x"])
        for kk, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"), (a, a, a, 90.0, 90.0, 90.0)
        ):
            fp[f"sample/{kk}"] = v
        fp["sample/space_group"] = "P 1"
        fp["instrument/wavelength"] = np.array([2.0, 10.0])
        fp.attrs["instrument"] = "CG4D"

    out_file = str(tmp_path / "spherical.h5")
    run_spherical_index(
        peaks_file,
        out_file,
        d_min=1.3,
        kernel_deg=1.0,
        fit_gonio_offsets=True,
        refine_gonio_axes=["outer_z", "inner_x"],
    )
    with h5py.File(out_file) as fp:
        off_z = float(fp["goniometer/offsets/outer_z"][()])
        off_x = float(fp["goniometer/offsets/inner_x"][()])
    assert abs(off_z - 0.8) < 0.15  # identifiable, recovered
    assert abs(off_x) < 0.15  # gauge: pinned near zero, not a wild number


def test_refined_instrument_reaches_the_consumers_layout(tmp_path):
    """What the refinement fits must land where the consumers look.

    The refinement can improve the fitted U while the predictor keeps
    running on nominal frames -- measured on cg4d-garnet, aligned median
    0.69 -> 0.42 deg yet CC(1/2) 0.9853 -> 0.8814, purely because the
    corrections lived in the report group and nowhere else.  This pins the
    round trip: refined panels appear as a detector_calibration group that
    apply_detector_calibration can consume, and per-run angle corrections
    appear folded into the FRAME-addressed goniometer/angles the predictor
    reads, with the nominal angles and the run map kept alongside.
    """
    from subhkl.commands import apply_detector_calibration
    from subhkl.config import beamlines as _beamlines

    rng = np.random.default_rng(53)
    a = 8.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    h, k, l_ = generate_reflections(a, a, a, 90, 90, 90, space_group="P 1", d_min=1.3)
    G = np.stack([h, k, l_], axis=1) @ B.T
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    U_true = Q * np.sign(np.linalg.det(Q))

    axes = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    run_angles = np.array([[0.0, 0.0], [0.0, 40.0], [0.0, 80.0]])
    frames_per_run = 2
    dets = {int(kk): Detector(v) for kk, v in beamlines["CG4D"].items()}

    from subhkl.instrument.refinables import gonio_rotation_jax

    recs = {k2: [] for k2 in ("bank", "pr", "pc", "img", "run", "R", "ang")}
    for r in range(len(run_angles)):
        Rr = np.asarray(gonio_rotation_jax(axes, run_angles[r], np.zeros(2)))
        d_lab = np.vstack([dirs, -dirs]) @ (Rr @ U_true).T
        d_lab = d_lab[d_lab[:, 2] < -0.05]
        kf = np.array([0.0, 0.0, 1.0]) - 2.0 * d_lab[:, 2:3] * d_lab
        kf /= np.linalg.norm(kf, axis=1, keepdims=True)
        for bk, det in dets.items():
            mask, r_, c_ = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
            n_ = int(np.sum(mask))
            if n_ == 0:
                continue
            recs["bank"].append(np.full(n_, bk))
            recs["pr"].append(r_[mask] + rng.normal(scale=0.5, size=n_))
            recs["pc"].append(c_[mask] + rng.normal(scale=0.5, size=n_))
            recs["img"].append(np.full(n_, r * frames_per_run))
            recs["run"].append(np.full(n_, r))
            recs["R"].append(np.tile(Rr, (n_, 1, 1)))
            recs["ang"].append(np.tile(run_angles[r], (n_, 1)))

    peaks_file = str(tmp_path / "finder.h5")
    with h5py.File(peaks_file, "w") as fp:
        fp["bank"] = np.concatenate(recs["bank"])
        fp["peaks/pixel_r"] = np.concatenate(recs["pr"])
        fp["peaks/pixel_c"] = np.concatenate(recs["pc"])
        fp["peaks/image_index"] = np.concatenate(recs["img"])
        fp["peaks/run_index"] = np.concatenate(recs["run"])
        fp["goniometer/R"] = np.concatenate(recs["R"])
        fp["goniometer/angles"] = np.concatenate(recs["ang"])
        fp["goniometer/axes"] = axes
        fp["goniometer/names"] = np.array([b"outer_z", b"phi"])
        for kk, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"), (a, a, a, 90.0, 90.0, 90.0)
        ):
            fp[f"sample/{kk}"] = v
        fp["sample/space_group"] = "P 1"
        fp["instrument/wavelength"] = np.array([2.0, 10.0])
        fp.attrs["instrument"] = "CG4D"

    # the frame table the predictor addresses: one row per image, run
    # boundaries in file_offsets (a merged stack's metadata)
    n_frames = len(run_angles) * frames_per_run
    frame_angles = np.repeat(run_angles, frames_per_run, axis=0)
    images_file = str(tmp_path / "merged.h5")
    with h5py.File(images_file, "w") as fp:
        fp["goniometer/angles"] = frame_angles
        fp["goniometer/axes"] = axes
        fp["file_offsets"] = np.arange(len(run_angles)) * frames_per_run
        fp["files"] = np.array([f"run{r}.nxs".encode() for r in range(len(run_angles))])

    out_file = str(tmp_path / "spherical.h5")
    run_spherical_index(
        peaks_file,
        out_file,
        d_min=1.3,
        kernel_deg=1.0,
        refine_instrument=True,
        detector_modes=["global_trans"],
        refine_gonio_axes=["outer_z"],
        refine_gonio_per_run="phi",
        refine_maxiter=150,
        frame_table_filename=images_file,
    )

    with h5py.File(out_file) as fp:
        assert "detector_calibration" in fp
        banks = [int(b.split("_")[1]) for b in fp["detector_calibration"]]
        assert set(banks) == set(
            int(b) for b in np.unique(np.concatenate(recs["bank"]))
        )
        ang = fp["goniometer/angles"][()]
        ang_nom = fp["goniometer/angles_nominal"][()]
        delta = fp["goniometer/per_run/delta_deg"][()]
        f2r = fp["goniometer/per_run/frame_to_run"][()]
        per_run = fp["spherical/gonio/per_run_deg"][()]
        motor = fp["goniometer/per_run/motor"][()]
        centers = {
            int(b.split("_")[1]): fp[f"detector_calibration/{b}/center"][()]
            for b in fp["detector_calibration"]
        }

    # frame-addressed, nominal preserved, correction on the named motor only
    assert ang.shape == (n_frames, 2)
    assert np.allclose(ang_nom, frame_angles)
    assert (motor.decode() if isinstance(motor, bytes) else str(motor)) == "phi"
    assert np.allclose(ang[:, 0], ang_nom[:, 0])  # untouched axis
    # every frame carries exactly its run's fitted correction: the fit and
    # the file cannot disagree
    assert np.allclose(ang[:, 1] - ang_nom[:, 1], delta[f2r], atol=1e-9)
    assert np.allclose(delta[: len(per_run)], per_run, atol=1e-9)

    # and the panels are consumable by the very function downstream calls
    nominal_centers = {b: np.array(dets[b].center) for b in centers}
    apply_detector_calibration(out_file, "CG4D")
    for b, c_ref in centers.items():
        assert np.allclose(_beamlines["CG4D"][str(b)]["center"], c_ref)
    # global_trans moved every panel by the same vector
    shifts = np.stack([centers[b] - nominal_centers[b] for b in centers])
    assert np.allclose(shifts, shifts[0], atol=1e-6)
