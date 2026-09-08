"""Finder-free geometry calibration, on synthetic frames ray-traced onto real
CG4D panels.

A crystal is imaged onto a subset of the actual CG4D panel bank under a
DISPLACED geometry (radial scale, assembly rotation, sample offset) while the
instrument still names the nominal one -- the situation the calibration is
for.  The objectives are checked to prefer the displacing geometry, the
round trip through ``apply_detector_calibration`` is pinned, and the slow
tests run the blind search and the whole optimiser.
"""

from __future__ import annotations

import copy
from itertools import product

import h5py
import numpy as np
import pytest

from subhkl.calibrate import (
    Calibrator,
    Evaluation,
    Stage,
    geometry_detectors,
    prepare_banks,
    write_detector_calibration,
)
from subhkl.config import beamlines
from subhkl.core.crystallography import (
    cartesian_matrix_metric_tensor,
    generate_reflections,
)

A_CELL = 12.0
BAND = (2.0, 10.0)
D_MIN = 1.2
N_BANKS = 12  # the panels the crystal actually hits, chosen per orientation
SOME_BANKS = [int(k) for k in sorted(beamlines["CG4D"], key=int)[:12]]


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


def _angle_to_lattice(U, U_true):
    """Smallest rotation between U and U_true up to the cubic symmetry."""
    best = 180.0
    for S in _cubic_rots():
        c = (np.trace(U @ (U_true @ S).T) - 1.0) / 2.0
        best = min(best, np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return best


def _synthetic_frames(rng, g_true, background=6.0, amplitude=80.0, sigma_px=2.0):
    """Frames of one crystal orientation on the first CG4D panels, imaged
    through the displaced geometry ``g_true``: Poisson background plus a
    Gaussian spot at every in-band, on-panel reflection."""
    B, _ = cartesian_matrix_metric_tensor(
        A_CELL, A_CELL, A_CELL, *np.deg2rad([90, 90, 90])
    )
    h, k, l_ = generate_reflections(
        A_CELL, A_CELL, A_CELL, 90, 90, 90, space_group="P 1", d_min=D_MIN
    )
    G = np.stack([h, k, l_], axis=1) @ B.T
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    U_true = Q * np.sign(np.linalg.det(Q))
    G_lab = G @ U_true.T
    qn = np.linalg.norm(G_lab, axis=1)
    st = -G_lab[:, 2] / qn
    lam = 2.0 * st / qn
    keep = (st > 0) & (lam >= BAND[0]) & (lam <= BAND[1])
    kf = np.array([0.0, 0.0, 1.0]) + lam[keep, None] * G_lab[keep]
    kf /= np.linalg.norm(kf, axis=1, keepdims=True)

    dets = geometry_detectors("CG4D", g_true)
    hits = {
        b: int(np.sum(det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])[0]))
        for b, det in dets.items()
    }
    banks = sorted(sorted(hits, key=lambda b: -hits[b])[:N_BANKS])
    images = []
    n_spots = 0
    r = int(np.ceil(4 * sigma_px))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    blob = amplitude * np.exp(-0.5 * (xx**2 + yy**2) / sigma_px**2)
    for b in banks:
        det = dets[b]
        H, W = det.n, det.m
        im = np.full((H, W), background, float)
        mask, pr, pc = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
        for row, col in zip(np.asarray(pr)[mask], np.asarray(pc)[mask]):
            i0, j0 = int(round(row)), int(round(col))
            r0, r1 = max(i0 - r, 0), min(i0 + r + 1, H)
            c0, c1 = max(j0 - r, 0), min(j0 + r + 1, W)
            if r1 <= r0 or c1 <= c0:
                continue
            im[r0:r1, c0:c1] += blob[
                r0 - i0 + r : r1 - i0 + r, c0 - j0 + r : c1 - j0 + r
            ]
            n_spots += 1
        images.append(rng.poisson(im).astype(np.int32))
    return np.stack(images), U_true, n_spots, banks


G_TRUE = np.array([0.04, 0.0, np.radians(1.0), 0.0, 0.0, 0.004, 0.006])


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(7)
    images, U_true, n_spots, banks = _synthetic_frames(rng, G_TRUE)
    assert n_spots > 60, n_spots
    return images, U_true, banks


@pytest.fixture(scope="module")
def calibrator(frames):
    images, _, banks = frames
    data = prepare_banks(images, banks, bin_px=4)
    assert len(data) == len(banks)
    return Calibrator(
        data, "CG4D", [A_CELL] * 3 + [90.0] * 3, "P 1", BAND, D_MIN, n_random=60, seed=1
    )


def test_nominal_geometry_is_the_identity():
    dets = geometry_detectors("CG4D", np.zeros(7), banks=SOME_BANKS)
    for b in SOME_BANKS:
        cfg = beamlines["CG4D"][str(b)]
        assert np.allclose(dets[b].center, cfg["center"])
        assert np.allclose(dets[b].uhat, cfg["uhat"])
        assert np.allclose(dets[b].vhat, cfg["vhat"])


def test_geometry_moves_the_assembly_rigidly():
    g = np.array([0.05, 0.0, 0.02, 0.0, 0.001, 0.002, 0.003])
    dets = geometry_detectors("CG4D", g, banks=SOME_BANKS)
    for b in SOME_BANKS:
        cfg = beamlines["CG4D"][str(b)]
        c0 = np.asarray(cfg["center"], float)
        # rotated, scaled and shifted centre; rotated, still orthonormal frame
        assert np.isclose(
            np.linalg.norm(np.asarray(dets[b].center) - g[4:7]),
            1.05 * np.linalg.norm(c0),
        )
        assert np.isclose(np.dot(dets[b].uhat, dets[b].vhat), 0.0, atol=1e-12)
        assert np.isclose(np.dot(dets[b].uhat, cfg["uhat"]), np.cos(0.02), atol=1e-3)


def test_detector_calibration_round_trip(tmp_path):
    """What calibrate writes, the consumers must read back unchanged."""
    from subhkl.commands import apply_detector_calibration

    snapshot = copy.deepcopy(beamlines["CG4D"])
    try:
        dets = geometry_detectors("CG4D", G_TRUE, banks=SOME_BANKS)
        out_file = str(tmp_path / "calib.h5")
        with h5py.File(out_file, "w") as fp:
            write_detector_calibration(fp, dets)
        with h5py.File(out_file) as fp:
            groups = sorted(
                fp["detector_calibration"], key=lambda s: int(s.split("_")[1])
            )
            assert [int(s.split("_")[1]) for s in groups] == SOME_BANKS
            for b in SOME_BANKS:
                grp = fp[f"detector_calibration/bank_{b}"]
                assert grp["center"].shape == (3,) and grp["center"].dtype == np.float64
                assert float(grp["width"][()]) == pytest.approx(dets[b].width)
        apply_detector_calibration(out_file, "CG4D")
        for b in SOME_BANKS:
            assert np.allclose(beamlines["CG4D"][str(b)]["center"], dets[b].center)
            assert np.allclose(beamlines["CG4D"][str(b)]["uhat"], dets[b].uhat)
    finally:
        beamlines["CG4D"].clear()
        beamlines["CG4D"].update(snapshot)


def test_objectives_prefer_the_displacing_geometry(frames, calibrator):
    """With the orientation given, both objectives score the true geometry
    above the nominal one, and the continuous one explains most detections."""
    _, U_true, _ = frames
    stage = Stage(1.0, "cell", 8.0, 1.0, 1)
    true = calibrator.evaluate(G_TRUE, stage, orientation=U_true)
    nominal = calibrator.evaluate(np.zeros(7), stage, orientation=U_true)
    assert true.n_detections > 30
    assert true.raw_explained > 0.6
    assert true.raw_explained > nominal.raw_explained + 0.15
    assert true.cell_correlation > nominal.cell_correlation


def test_cell_dictionary_is_optional(frames, calibrator):
    """The cell dictionary is the default objective but optional: the
    correlation is computed only for the 'cell' objective (or when asked
    for), and the ladder's own 'band' score can be selected instead."""
    from subhkl.calibrate import DEFAULT_STAGES, QUICK_STAGES, with_objective

    assert all(s.objective == "cell" for s in DEFAULT_STAGES + QUICK_STAGES)
    assert [s.objective for s in with_objective(QUICK_STAGES, "band")] == ["band"] * 3
    with pytest.raises(ValueError):
        with_objective(QUICK_STAGES, "zone")
    _, U_true, _ = frames
    band = calibrator.evaluate(
        G_TRUE, Stage(1.0, "band", 8.0, 1.0, 1), orientation=U_true
    )
    assert np.isnan(band.cell_correlation)
    cell = calibrator.evaluate(
        G_TRUE, Stage(1.0, "cell", 8.0, 1.0, 1), orientation=U_true
    )
    assert np.isfinite(cell.cell_correlation)
    assert (
        Calibrator.objective_value(cell, Stage(1.0, "cell", 8.0, 1.0, 1))
        == cell.cell_correlation
    )
    ev = Evaluation(G_TRUE, U_true, 0.0123, 1, float("nan"), 0.5, 1)
    assert Calibrator.objective_value(
        ev, Stage(1.0, "band", 8.0, 1.0, 1)
    ) == pytest.approx(12.3)
    assert Calibrator.objective_value(
        ev, Stage(1.0, "raw", 8.0, 1.0, 1)
    ) == pytest.approx(50.0)


def test_detection_map_is_sparse_and_precise(calibrator, frames):
    """Detections are a small fraction of the covered cells, and each carries
    the exact direction of its argmax pixel (unit, inside its own cell)."""
    stage = Stage(1.0, "cell", 8.0, 1.0, 1)
    dmap, _ = calibrator.detection_map(G_TRUE, stage)
    det = dmap.detected
    assert 0 < det.sum() < 0.1 * dmap.covered.sum()
    d = dmap.direction[det]
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    from subhkl.calibrate import SphereGrid

    assert np.array_equal(SphereGrid(1.0).to_cell(d), np.where(det)[0])


@pytest.mark.slow
def test_blind_orientation_recovers_the_crystal(frames, calibrator):
    _, U_true, _ = frames
    ev = calibrator.evaluate(G_TRUE, Stage(1.0, "cell", 8.0, 1.0, 1))
    assert _angle_to_lattice(ev.U, U_true) < 3.0
    assert ev.raw_explained > 0.5


@pytest.mark.slow
def test_calibrate_climbs_toward_the_displacing_geometry(frames, calibrator):
    """The staged optimiser moves the geometry the right way.

    It starts half-way to the true displacement: on this small synthetic
    frame (119 spots on 12 panels) the blind ladder locks there (72 %
    explained) but not at nominal (7.7 deg off, chance level), so from
    nominal the objective is flat.  The real pooled data locks at nominal;
    the basin's reach is a matter of spot count, not of the optimiser.
    """
    g0 = 0.5 * G_TRUE
    stages = (Stage(1.0, "band", 8.0, 1.0, 30), Stage(0.5, "raw", 8.0, 0.5, 20))
    start = calibrator.evaluate(g0, stages[-1])
    res = calibrator.calibrate(stages=stages, g0=g0)
    assert res.final.raw_explained > start.raw_explained + 0.03
    assert abs(res.radial_scale - G_TRUE[0]) < abs(g0[0] - G_TRUE[0])
