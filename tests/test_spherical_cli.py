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
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 0.5
    assert z[0] > 10.0

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
    )
    with h5py.File(out_file) as fp:
        U = fp["sample/U"][()]
        z = fp["spherical/z"][()]
    err = min(np.rad2deg(_quat_angle(U, U_true @ S)) for S in _cubic_rots())
    assert err < 1.0
    assert z[0] > 5.0

    written = run_indexer_visualize(
        out_file, output_dir=str(tmp_path), max_index=1, images_filename=images_file
    )
    assert os.path.exists(written[0])
    assert os.path.getsize(written[0]) > 30_000  # images actually rendered
