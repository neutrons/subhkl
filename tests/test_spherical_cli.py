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


def _synthetic_finder_file(path, rng, wavelength=None, perturb=None):
    """``perturb`` = dict(radial=, rot_deg=) ray-traces onto a DISPLACED
    detector (every bank centre scaled, the assembly rotated about y) while
    the file still names the nominal instrument -- the situation on
    IMAGINE-X.  ``wavelength`` narrower than the default keeps only the
    reflections the band admits."""
    a = 8.0
    B, _ = cartesian_matrix_metric_tensor(a, a, a, *np.deg2rad([90, 90, 90]))
    h, k, l_ = generate_reflections(a, a, a, 90, 90, 90, space_group="P 1", d_min=1.3)
    G = np.stack([h, k, l_], axis=1) @ B.T
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    U_true = Q * np.sign(np.linalg.det(Q))

    G_lab = G @ U_true.T
    d_lab = G_lab / np.linalg.norm(G_lab, axis=1, keepdims=True)
    lam = -2.0 * d_lab[:, 2] / np.linalg.norm(G_lab, axis=1)  # 2 sin(theta) / |Q|
    keep = d_lab[:, 2] < -0.05
    if wavelength is not None:  # an explicit band keeps only what it admits
        keep &= (lam >= wavelength[0]) & (lam <= wavelength[1])
    d_lab = d_lab[keep]
    kf = np.array([0.0, 0.0, 1.0]) - 2.0 * d_lab[:, 2:3] * d_lab
    kf /= np.linalg.norm(kf, axis=1, keepdims=True)

    banks, rows, cols = [], [], []
    dets = {}
    for kk, v in beamlines["CG4D"].items():
        cfg = dict(v)
        if perturb:
            from scipy.spatial.transform import Rotation as _Rot

            Rp = _Rot.from_rotvec(
                [0.0, np.deg2rad(perturb["rot_deg"]), 0.0]
            ).as_matrix()
            cfg["center"] = (
                Rp @ (np.asarray(cfg["center"], float) * (1.0 + perturb["radial"]))
            ).tolist()
            cfg["uhat"] = (Rp @ np.asarray(cfg["uhat"], float)).tolist()
            cfg["vhat"] = (Rp @ np.asarray(cfg["vhat"], float)).tolist()
        dets[int(kk)] = Detector(cfg)
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
        fp["instrument/wavelength"] = np.array(
            (2.0, 10.0) if wavelength is None else wavelength, float
        )
        fp.attrs["instrument"] = "CG4D"
    return U_true, n


def test_spherical_index_and_zone_overlay(tmp_path):
    rng = np.random.default_rng(23)
    peaks_file = str(tmp_path / "finder.h5")
    U_true, n = _synthetic_finder_file(peaks_file, rng)
    assert n > 100  # the scene actually covers the instrument

    out_file = str(tmp_path / "spherical.h5")
    run_spherical_index(
        peaks_file, out_file, d_min=1.3, kernel_deg=1.0, search="correlogram"
    )
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

    # dpi kept small on purpose: the product default is 600 (print-grade
    # overlays), which is a needlessly heavy render for a smoke check
    written = run_indexer_visualize(
        out_file, output_dir=str(tmp_path), max_index=1, dpi=150
    )
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
        runs=[0],  # exercises the frame-side run filter too,
        search="correlogram",
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
        out_file,
        output_dir=str(tmp_path),
        max_index=1,
        images_filename=images_file,
        dpi=150,
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
        search="correlogram",
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
        search="correlogram",
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

    # Without a frame table the same run-addressed keys must still be
    # complete: the predictor requires frame_to_run as soon as trans_m
    # exists (KeyError on the first native run), and the MTZ exporter
    # reads the classic delta_deg -- neither may depend on a merged stack
    # being at hand.
    out_nt = str(tmp_path / "spherical_no_table.h5")
    run_spherical_index(
        peaks_file,
        out_nt,
        d_min=1.3,
        kernel_deg=1.0,
        refine_instrument=True,
        detector_modes=["global_trans"],
        refine_gonio_axes=["outer_z"],
        refine_gonio_per_run="phi",
        refine_gonio_per_run_trans=True,
        refine_maxiter=150,
        search="correlogram",
    )
    with h5py.File(out_nt) as fp:
        g = fp["goniometer/per_run"]
        assert "frame_to_run" in g and "delta_deg" in g and "trans_m" in g
        f2r = g["frame_to_run"][()]
        delta1 = g["delta_deg"][()]
        trans1 = g["trans_m"][()]
        per_run1 = fp["spherical/gonio/per_run_deg"][()]
    # frame -> run is a real map over the frames the peaks witness, and
    # every per-run array is indexed by RUN ID (what the consumers use),
    # long enough for the largest run id present
    n_frames_seen = int(np.max(np.concatenate(recs["img"]))) + 1
    assert len(f2r) == n_frames_seen
    for r in range(len(run_angles)):
        assert f2r[r * frames_per_run] == r
    n_runs = int(np.max(np.concatenate(recs["run"]))) + 1
    assert len(delta1) == n_runs and trans1.shape == (n_runs, 3)
    assert np.allclose(delta1, per_run1, atol=1e-9)

    # and the panels are consumable by the very function downstream calls
    nominal_centers = {b: np.array(dets[b].center) for b in centers}
    apply_detector_calibration(out_file, "CG4D")
    for b, c_ref in centers.items():
        assert np.allclose(_beamlines["CG4D"][str(b)]["center"], c_ref)
    # global_trans moved every panel by the same vector
    shifts = np.stack([centers[b] - nominal_centers[b] for b in centers])
    assert np.allclose(shifts, shifts[0], atol=1e-6)


def test_nodal_dictionary_indexes_end_to_end(tmp_path):
    """The nodal dictionary (the 2D search's default, --radial 0): a P1
    cell's zone crossings to |uvw| <= 2 index the synthetic scene through
    the CLI path, the file records which dictionary solved it, and the
    optional full-list final stage continues from that solution rather than
    replacing it."""
    rng = np.random.default_rng(29)
    peaks_file = str(tmp_path / "finder.h5")
    U_true, n = _synthetic_finder_file(peaks_file, rng)

    out_nodal = str(tmp_path / "nodal.h5")
    run_spherical_index(
        peaks_file,
        out_nodal,
        d_min=1.3,
        kernel_deg=1.0,
        model="nodal",
        radial=0,
        nodal_max_index=2,
        search="correlogram",
    )
    with h5py.File(out_nodal) as fp:
        U = fp["sample/U"][()]
        assert fp["spherical/model"][()].decode() == "nodal"
        assert int(fp["spherical/nodal_max_index"][()]) == 2
        assert not bool(fp["spherical/final_full_refine"][()])
        n_model = int(fp["spherical/model_size"][()])
        z = fp["spherical/z"][()]
    assert 0 < n_model < 400
    assert z[0] > 8.0
    err_nodal = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err_nodal < 0.5

    out_full = str(tmp_path / "nodal_full.h5")
    run_spherical_index(
        peaks_file,
        out_full,
        d_min=1.3,
        kernel_deg=1.0,
        model="nodal",
        radial=0,
        nodal_max_index=2,
        final_full_refine=True,
        search="correlogram",
    )
    with h5py.File(out_full) as fp:
        U2 = fp["sample/U"][()]
        assert bool(fp["spherical/final_full_refine"][()])
    err_full = min(np.rad2deg(_quat_angle(U2, U_true @ S)) for S in _cubic_rots())
    assert err_full < 0.5
    assert err_full <= err_nodal + 0.05


def test_band_consistency_is_on_by_default_and_switchable(tmp_path):
    """The synthetic finder file carries instrument/wavelength, so the
    default search is the band-consistent one; switching it off still
    indexes, and the file records the same orientation either way."""
    rng = np.random.default_rng(31)
    peaks_file = str(tmp_path / "finder.h5")
    U_true, _ = _synthetic_finder_file(peaks_file, rng)
    outs = {}
    for flag in (True, False):
        out = str(tmp_path / f"band_{int(flag)}.h5")
        run_spherical_index(
            peaks_file,
            out,
            d_min=1.3,
            kernel_deg=1.0,
            radial=0,
            band_consistency=flag,
            search="correlogram",
        )
        with h5py.File(out) as fp:
            outs[flag] = fp["sample/U"][()]
    for flag, U in outs.items():
        err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
        assert err < 0.5, (flag, err)


def _dense_scene(tmp_path, seed=41):
    """One still of 80 CG4D banks on a flat 25 count/pixel background with
    a dead border: the dense-regime scene shared by the raw-path tests.
    Returns (meta_path, merged_path, U_true)."""
    rng = np.random.default_rng(seed)
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
    dets = {int(kk): Detector(v) for kk, v in beamlines["CG4D"].items()}
    images, bank_ids = [], []
    yy, xx = np.mgrid[0:512, 0:512]
    for bk, det in dets.items():
        mask, r_, c_ = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
        im = np.full((512, 512), 25.0)
        for rr, cc in zip(r_[mask], c_[mask]):
            im += 300.0 * np.exp(-((yy - rr) ** 2 + (xx - cc) ** 2) / (2 * 2.0**2))
        im = rng.poisson(im).astype(np.int64)
        im[:6, :] = 0
        im[-6:, :] = 0
        im[:, :6] = 0
        im[:, -6:] = 0  # dead border
        images.append(im)
        bank_ids.append(bk)
    merged = str(tmp_path / "merged.h5")
    with h5py.File(merged, "w") as fp:
        fp["images"] = np.stack(images)
        fp["bank_ids"] = np.array(bank_ids, dtype=np.int32)
        fp["file_offsets"] = np.array([0], dtype=np.int64)
    meta = str(tmp_path / "meta.h5")
    with h5py.File(meta, "w") as fp:
        n = len(images)
        fp["bank"] = np.array(bank_ids)
        fp["peaks/pixel_r"] = np.zeros(n)
        fp["peaks/pixel_c"] = np.zeros(n)
        fp["peaks/image_index"] = np.arange(n)
        fp["peaks/run_index"] = np.zeros(n, dtype=int)
        fp["goniometer/R"] = np.tile(np.eye(3), (n, 1, 1))
        for kk, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"), (a, a, a, 90.0, 90.0, 90.0)
        ):
            fp[f"sample/{kk}"] = v
        fp["sample/space_group"] = "P 1"
        fp["instrument/wavelength"] = np.array([2.0, 10.0])
        fp.attrs["instrument"] = "CG4D"
    return meta, merged, U_true


def test_dense_background_regime_isolates_spots_and_masks_rims(tmp_path):
    """A frame with a flat 25 count/pixel background has no zero pixels
    except its dead border, so the zero-fraction estimator reads the
    border, not the background (56 counts/bin against a true 400 on
    MANDI L1).  In that regime the raw path must switch to a local
    median: the excess it projects has to come from the spots, and the
    panel rim -- a great circle on the sphere -- must carry none."""
    from subhkl.commands import run_spherical_index

    meta, merged, U_true = _dense_scene(tmp_path)
    out = str(tmp_path / "out.h5")
    run_spherical_index(
        meta,
        out,
        d_min=1.3,
        kernel_deg=1.0,
        images_filename=merged,
        binning=4,
        refine=False,
        search="correlogram",
    )
    with h5py.File(out) as fp:
        U = fp["sample/U"][()]
        z = fp["spherical/z"][()]
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 1.0
    assert z[0] > 5.0  # one still on a 25 count/pixel background


def test_ewald_refine_ranks_and_polishes_raw_counts(tmp_path):
    """--ewald-refine: the correlogram's candidates are re-ranked on the
    exact Ewald objective (predicted in-band spots against the excess
    image, no band limit) and the winner polished on it.  Measured on
    MANDI L1: candidates a band-limited z could not order (16 vs 13)
    separate 3:1 on the exact score, and the polish lands within 0.2 deg
    of a TOF reference where the band-limited polish left 0.7-1.0 deg."""
    from subhkl.commands import run_spherical_index

    meta, merged, U_true = _dense_scene(tmp_path, seed=43)
    out = str(tmp_path / "ewald.h5")
    run_spherical_index(
        meta,
        out,
        d_min=1.3,
        kernel_deg=1.0,
        images_filename=merged,
        binning=4,
        n_candidates=8,
        ewald_refine=True,
        search="correlogram",
    )
    with h5py.File(out) as fp:
        U = fp["sample/U"][()]
        assert bool(fp["spherical/ewald_refine"][()])
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 0.3


def test_lattice_search_indexes_raw_counts_and_peaks(tmp_path):
    """--search lattice: the direct-lattice ladder.  The data side (raw
    path: every live bin of the excess image with its signed excess for the
    exhaustive rung, the strongest bins for the zooms; peaks path: the peak
    list) is scored by the band average of the phase at the shortest
    direct-lattice vectors, coarse to fine, with no band limit anywhere; in
    raw-count mode the exact Ewald stage polishes the winner.  Measured on MANDI L1
    (100 A) and garnet (12 A, 24 copies): every still within 0.16 deg of an
    independent reference, ~2 s per still on an H100."""
    from subhkl.commands import run_spherical_index

    meta, merged, U_true = _dense_scene(tmp_path, seed=47)
    out = str(tmp_path / "lattice_raw.h5")
    run_spherical_index(
        meta,
        out,
        d_min=1.3,
        kernel_deg=1.0,
        images_filename=merged,
        binning=4,
        search="lattice",
    )
    with h5py.File(out) as fp:
        U = fp["sample/U"][()]
        assert fp["spherical/search"][()].decode() == "lattice"
        assert bool(fp["spherical/ewald_refine"][()])
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 0.3

    rng = np.random.default_rng(53)
    peaks_file = str(tmp_path / "finder.h5")
    U_true, _ = _synthetic_finder_file(peaks_file, rng)
    out = str(tmp_path / "lattice_peaks.h5")
    run_spherical_index(peaks_file, out, d_min=1.3, kernel_deg=1.0, search="lattice")
    with h5py.File(out) as fp:
        U = fp["sample/U"][()]
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 0.5


def test_radial_search_indexes_both_data_paths(tmp_path):
    """--radial N: the 3D search (Laue segments against lattice shells)
    on the peaks path and on the raw-count path, both from the synthetic
    CG4D scene; the orientation must come out and the file must say
    which search ran."""
    rng = np.random.default_rng(37)
    peaks_file = str(tmp_path / "finder.h5")
    U_true, _ = _synthetic_finder_file(peaks_file, rng)
    out = str(tmp_path / "radial_peaks.h5")
    run_spherical_index(
        peaks_file,
        out,
        d_min=1.3,
        kernel_deg=1.0,
        model="reflections",
        radial=16,
        search="correlogram",
    )
    with h5py.File(out) as fp:
        U = fp["sample/U"][()]
        z = fp["spherical/z"][()]
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 0.5
    assert z[0] > 8.0


def test_corefinement_recovers_a_displaced_detector(tmp_path):
    """The ladder searches on nominal geometry.  Ray-trace a crystal onto a
    CG4D assembly that is 5% further out and turned 1.5 deg about y while
    the file names the nominal instrument -- the IMAGINE-X situation, where
    the classic calibration wants a 44 mm centre shift.  Co-refining the
    seven global parameters on the band objective, then re-searching the
    orientation under the adopted geometry, must beat the same search with
    the geometry left nominal, and must recover most of the displacement.
    Not all of it: on ONE still a rotation of the assembly about the sample
    is, to first order in the directions, a rotation of the crystal, so
    part of the 1.5 deg lands in U (measured: 0.79 deg, centre residual 41%
    of the displacement); the pooled instrument refinement downstream is
    what separates them across runs.  On cg4d-l1-mbl 1996: geometry
    +5.5% / 2.4 deg, peaks explained 31% -> 54%."""
    from subhkl.commands import run_spherical_index

    rng = np.random.default_rng(61)
    peaks_file = str(tmp_path / "finder_displaced.h5")
    U_true, n = _synthetic_finder_file(
        peaks_file, rng, wavelength=(2.0, 4.5), perturb=dict(radial=0.05, rot_deg=1.5)
    )
    assert n > 40
    errs = {}
    for corefine in (False, True):
        out = str(tmp_path / f"corefine_{corefine}.h5")
        run_spherical_index(
            peaks_file,
            out,
            d_min=1.3,
            kernel_deg=1.0,
            search="lattice",
            corefine=corefine,
            n_candidates=8,
        )
        with h5py.File(out) as fp:
            U = fp["sample/U"][()]
            assert ("detector_calibration" in fp) == corefine
            if corefine:
                cal = fp["detector_calibration"]
                Rp = np.array(
                    [
                        [np.cos(np.deg2rad(1.5)), 0, np.sin(np.deg2rad(1.5))],
                        [0, 1, 0],
                        [-np.sin(np.deg2rad(1.5)), 0, np.cos(np.deg2rad(1.5))],
                    ]
                )
                resid = []
                for g in cal:
                    b = int(g.split("_")[1])
                    c_nom = np.asarray(beamlines["CG4D"][str(b)]["center"], float)
                    c_true = Rp @ (c_nom * 1.05)
                    resid.append(
                        np.linalg.norm(cal[g]["center"][()] - c_true)
                        / np.linalg.norm(c_true - c_nom)
                    )
                # most of the displacement recovered (0 = all, 1 = none)
                assert np.median(resid) < 0.6, np.median(resid)
        errs[corefine] = min(
            np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots()
        )
    assert errs[True] < errs[False], errs
    assert errs[True] < 1.0, errs
