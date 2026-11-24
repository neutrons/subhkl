"""The Laue chain on a synthetic instrument: cell, per-run orientation,
scale, detector translation, and the hand-off into the band refinement."""

import copy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.config import beamlines
from subhkl.instrument.goniometer import Goniometer
from subhkl.search import blind_cell
from subhkl.search.blind_cell import (
    cell_of,
    determine_cell,
    find_cell,
    has_events,
    indexed_share,
    laue_observations,
    reduce_basis,
    refine_run_rotations,
    scaled_detectors,
    search_detector_shift,
)

INSTRUMENT = "TESTLAUE"
BAND = (3.0, 4.3)
D_MIN = 4.0
CELL = (48.0, 62.0, 80.0, 90.0, 104.0, 90.0)
PIXELS = 256


def _basis(cell):
    a, b, c = cell[:3]
    al, be, ga = np.radians(cell[3:])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array(
        [
            [a, b * np.cos(ga), cx],
            [0.0, b * np.sin(ga), cy],
            [0.0, 0.0, np.sqrt(c * c - cx * cx - cy * cy)],
        ]
    )


def _panels():
    """Three rings of flat panels, 0.12 m wide at 0.4 m: the angular pixel
    size of a real Laue camera."""
    out, bank = {}, 1
    for y in (-0.13, 0.0, 0.13):
        for phi in np.radians(np.arange(18.0, 360.0, 18.0)):
            mid = np.array([0.4 * np.sin(phi), y, 0.4 * np.cos(phi)])
            uhat = np.array([np.cos(phi), 0.0, -np.sin(phi)])
            vhat = np.array([0.0, 1.0, 0.0])
            out[str(bank)] = {
                "panel": "flat",
                "center": (mid - 0.06 * uhat - 0.06 * vhat).tolist(),
                "uhat": uhat.tolist(),
                "vhat": vhat.tolist(),
                "width": 0.12,
                "height": 0.12,
                "m": PIXELS,
                "n": PIXELS,
                "offset": 0,
            }
            bank += 1
    return out


@pytest.fixture
def instrument(monkeypatch):
    monkeypatch.setitem(beamlines, INSTRUMENT, _panels())
    return INSTRUMENT


def _write_images(
    path, instrument, true_basis, errors_deg, seed=0, shift=(0.0, 0.0, 0.0)
):
    """A Laue images file: one image per panel and run, spots where the
    lattice puts them.  ``errors_deg`` are rotation vectors [degree], one per
    run, by which the true rotation differs from the logged one.  ``shift``
    [m] moves the true panels against the nominal ones.

    Returns:
        true rotations (n_images, 3, 3)
    """
    rng = np.random.default_rng(seed)
    cfg = beamlines[instrument]
    banks = sorted(cfg, key=int)
    n_runs = len(errors_deg)
    omega = 25.0 * np.arange(n_runs)
    with h5py.File(path, "w") as f:
        f["goniometer/axes"] = np.array([[0.0, 1.0, 0.0, 1.0]])
        f["goniometer/angles"] = np.repeat(omega, len(banks))[:, None]
        f["goniometer/names"] = [b"omega"]
        f["bank_ids"] = np.tile([int(b) for b in banks], n_runs)
        f["file_offsets"] = np.arange(n_runs) * len(banks)
        f.attrs["instrument"] = instrument
        f["instrument/wavelength"] = np.array(BAND)
    with h5py.File(path, "r") as f:
        logged = np.array(Goniometer.from_h5_file(f).rotation, dtype=float)
    true = logged.copy()
    recip = np.linalg.inv(true_basis).T
    r = np.arange(-24, 25)
    hkl = np.array(
        [(h, k, l) for h in r for k in r for l in r if (h, k, l) != (0, 0, 0)]
    )
    q_s = hkl @ recip.T
    q_s = q_s[np.linalg.norm(q_s, axis=1) < 1.0 / D_MIN]
    images = rng.poisson(1.0, (n_runs * len(banks), PIXELS, PIXELS)).astype(np.float32)
    for run, err in enumerate(errors_deg):
        first = run * len(banks)
        R = logged[first] @ Rotation.from_rotvec(np.radians(err)).as_matrix()
        true[first : first + len(banks)] = R
        q = q_s @ R.T
        lam = -2.0 * q[:, 2] / (q * q).sum(1)
        ok = (lam > BAND[0]) & (lam < BAND[1])
        d = np.array([0.0, 0.0, 1.0]) + lam[ok, None] * q[ok]
        for j, bank in enumerate(banks):
            c = cfg[bank]
            u, v = np.array(c["uhat"]), np.array(c["vhat"])
            corner = np.array(c["center"]) + np.asarray(shift)
            normal = np.cross(u, v)
            t = (corner @ normal) / (d @ normal)
            p = d * t[:, None] - corner
            col = np.rint((p @ u) / c["width"] * (PIXELS - 1)).astype(int)
            row = np.rint((p @ v) / c["height"] * (PIXELS - 1)).astype(int)
            hit = (
                (t > 0)
                & (col >= 2)
                & (col < PIXELS - 2)
                & (row >= 2)
                & (row < PIXELS - 2)
            )
            images[first + j, row[hit], col[hit]] += rng.uniform(
                200.0, 600.0, int(hit.sum())
            )
    with h5py.File(path, "a") as f:
        f["images"] = images
    return true


def _angle(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)))


@pytest.fixture
def lattice():
    A = Rotation.from_euler("xyz", [17, -31, 52], degrees=True).as_matrix() @ _basis(
        CELL
    )
    return reduce_basis(A)


def test_scaled_detectors_moves_the_panel_midpoints_and_restores(instrument):
    before = copy.deepcopy(beamlines[instrument])
    bank = "7"
    c = before[bank]
    half = 0.06 * np.array(c["uhat"]) + 0.06 * np.array(c["vhat"])
    mid = np.array(c["center"]) + half
    with scaled_detectors(instrument, 0.05, 0.02, shift=(0.001, -0.002, 0.003)):
        moved = np.array(beamlines[instrument][bank]["center"]) + half
        assert np.allclose(moved, mid * [1.05, 1.02, 1.05] + [0.001, -0.002, 0.003])
        assert beamlines[instrument][bank]["uhat"] == c["uhat"]
    assert beamlines[instrument] == before
    with pytest.raises(RuntimeError), scaled_detectors(instrument, 0.05):
        raise RuntimeError
    assert beamlines[instrument] == before


def test_has_events_is_false_for_an_images_only_file(instrument, lattice, tmp_path):
    path = tmp_path / "laue.h5"
    _write_images(path, instrument, lattice, [(0, 0, 0)])
    assert not has_events(path)
    assert not has_events(tmp_path / "missing.h5")


def test_run_rotations_are_recovered_against_a_shared_lattice(
    instrument, lattice, tmp_path
):
    """Two of three runs are logged 1.5 and 2 degrees off; with the lattice
    fixed, every run comes back to its true orientation, and the bins the
    lattice indexes say so."""
    path = tmp_path / "laue.h5"
    errors = [(0.0, 0.0, 0.0), (1.2, -0.6, 0.7), (-0.9, 1.5, 0.9)]
    true = _write_images(path, instrument, lattice, errors)
    n = len(true) // 3
    rotations, angles, stats = refine_run_rotations(
        path, instrument, lattice, BAND, d_min=D_MIN
    )
    for run, err in enumerate(errors):
        assert _angle(rotations[run * n], true[run * n]) < 0.25
        assert abs(angles[run] - np.linalg.norm(err)) < 0.3
    assert stats[1, 3] > stats[1, 2] + 0.2
    assert stats[:, 3].min() > 0.6
    # runs left out are left alone
    only, turned, _ = refine_run_rotations(
        path, instrument, lattice, BAND, d_min=D_MIN, runs=[2]
    )
    assert turned[1] == 0.0 and turned[2] > 1.0
    assert _angle(only[n], true[n]) > 1.0


def test_find_cell_from_laue_images_and_a_given_scale(instrument, lattice, tmp_path):
    """The lattice comes out of the images with the right shape; a known
    volume or edge then fixes its size."""
    path = tmp_path / "laue.h5"
    _write_images(path, instrument, lattice, [(0, 0, 0)] * 3)
    kw = {"d_min": D_MIN, "grid": 192, "x_min": 30.0, "wavelength": BAND}
    found = find_cell(path, instrument, **kw)
    want = np.array(cell_of(lattice))
    got = np.array(found.cell)
    assert np.allclose(got[:3] / got[0], want[:3] / want[0], atol=0.03)
    assert np.allclose(got[3:], want[3:], atol=1.5)
    assert found.diagnostics["ladder_z"] > 8
    obs = laue_observations(path, instrument, BAND, D_MIN)
    assert indexed_share(found.basis, obs) > 0.5

    volume = abs(
        np.linalg.det(lattice)
    )  # primitive monoclinic: conventional = primitive
    sized = find_cell(path, instrument, volume=volume, **kw)
    assert sized.diagnostics["scale_source"] == "given"
    assert abs(abs(np.linalg.det(sized.basis)) / volume - 1.0) < 1e-6
    edged = find_cell(path, instrument, length=80.0, **kw)
    assert np.isclose(np.array(edged.conventional_cell[:3]), 80.0, atol=1e-6).any()
    assert np.allclose(np.sort(edged.cell[:3]), np.sort(want[:3]), rtol=0.03)


def test_determine_cell_restricts_the_chain_to_the_selected_runs(
    instrument, lattice, tmp_path, monkeypatch
):
    path = tmp_path / "laue.h5"
    true = _write_images(path, instrument, lattice, [(0, 0, 0)] * 3)
    n = len(true) // 3
    seen = []

    def fake_find_cell(merged, inst, rotations=None, frames=None, **kw):
        seen.append(None if frames is None else list(frames))
        diag = {
            "ladder_score": 0.05,
            "ladder_z": 30.0,
            "scale_source": "band limits",
            "scale_determined": True,
        }
        return blind_cell.BlindCell(
            diagnostics=diag, **blind_cell.describe_lattice(lattice)
        )

    def no_refinement(*args, **kwargs):
        raise AssertionError("one run has nothing to align")

    monkeypatch.setattr(blind_cell, "find_cell", fake_find_cell)
    monkeypatch.setattr(blind_cell, "refine_run_rotations", no_refinement)
    found, rotations = determine_cell(path, instrument, wavelength=BAND, runs=[1])
    assert seen == [list(range(n, 2 * n))]
    assert rotations.shape == (3 * n, 3, 3)
    assert np.allclose(found.basis, lattice)


def test_detector_shift_search_finds_the_best_scoring_translation(
    instrument, monkeypatch
):
    """The search walks to the translation with the best lattice score and
    leaves the instrument as it found it."""
    before = copy.deepcopy(beamlines[instrument])
    nominal = np.array(before["1"]["center"])
    best = np.array([-0.0075, 0.005, 0.0025])

    def fake_find_cell(merged, inst, **kw):
        assert kw["grid"] == 384 and kw["x_min"] == 35.0  # the Laue settings
        shift = np.array(beamlines[inst]["1"]["center"]) - nominal
        miss = np.linalg.norm((shift - best) * [1.0, 1.0, 0.2])
        return SimpleNamespace(
            diagnostics={
                "ladder_score": 0.06 * np.exp(-((miss / 0.008) ** 2)),
                "ladder_z": 30.0,
            }
        )

    monkeypatch.setattr(blind_cell, "find_cell", fake_find_cell)
    lines = []
    shift, score, tried = search_detector_shift(
        "unused.h5", instrument, BAND, log=lines.append
    )
    assert np.allclose(shift[:2], best[:2], atol=1e-9)
    assert abs(shift[2] - best[2]) <= 0.0026
    assert score == max(tried.values())
    assert len(tried) == len(lines) and 18 < len(tried) <= 30
    assert beamlines[instrument] == before

    def lost(merged, inst, **kw):
        raise ValueError("no lattice")

    monkeypatch.setattr(blind_cell, "find_cell", lost)
    _shift, score, _tried = search_detector_shift("unused.h5", instrument, BAND)
    assert score == 0.0
