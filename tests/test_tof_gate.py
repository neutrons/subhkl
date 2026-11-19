"""Pixel-sorted event storage and time-of-flight gated integration."""

import h5py
import numpy as np
import pytest

from subhkl.io.events import (
    EventStore,
    box_events,
    has_events,
    sort_bank_events,
    write_events,
)


def _synthetic_image_events(rng, shape, band, bg_per_pixel, peaks, sigma_px, sigma_rel):
    """Events of one image: flat-in-time background plus Bragg peaks.

    peaks: list of (row, col, tof [microsecond], flux [photons])
    Returns the image and its events sorted by (flat pixel, tof).
    """
    H, W = shape
    n_bg = rng.poisson(bg_per_pixel * H * W)
    flat = [rng.integers(0, H * W, n_bg)]
    tof = [rng.uniform(band[0], band[1], n_bg)]
    for r, c, t, flux in peaks:
        n = rng.poisson(flux)
        rr = np.rint(rng.normal(r, sigma_px, n)).astype(int)
        cc = np.rint(rng.normal(c, sigma_px, n)).astype(int)
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        flat.append(rr[ok] * W + cc[ok])
        tof.append(rng.normal(t, sigma_rel * t, n)[ok])
    flat = np.concatenate(flat)
    tof = np.concatenate(tof).astype(np.float32)
    order = np.lexsort((tof, flat))
    image = np.bincount(flat, minlength=H * W).reshape(H, W)
    return image, tof[order]


def test_sort_bank_events_matches_loader_image():
    """The sorted events index the same image the loader has always built."""
    rng = np.random.default_rng(0)
    m, n = 6, 5
    local = rng.integers(0, m * n, 400)
    tof = rng.uniform(1.0e4, 3.0e4, 400).astype(np.float32)

    image, sorted_tof = sort_bank_events(local, tof, m, n, y_fast=True)
    legacy = np.bincount(local, minlength=m * n).reshape(m, n).T
    np.testing.assert_array_equal(image, legacy)

    offsets = np.concatenate([[0], np.cumsum(image.ravel())])
    for r in range(n):
        for c in range(m):
            mine = np.sort(tof[local == c * n + r])
            flat = r * m + c
            np.testing.assert_array_equal(
                sorted_tof[offsets[flat] : offsets[flat + 1]], mine
            )

    image, sorted_tof = sort_bank_events(local, tof, m, n, y_fast=False)
    np.testing.assert_array_equal(
        image, np.bincount(local, minlength=m * n).reshape(n, m)
    )
    assert len(sorted_tof) == 400


def test_box_events_returns_the_pixels_of_the_box():
    rng = np.random.default_rng(1)
    H, W = 12, 9
    image, tof = _synthetic_image_events(rng, (H, W), (1e4, 3e4), 5.0, [], 1.0, 0.0)
    offsets = np.concatenate([[0], np.cumsum(image.ravel())])
    t, pix = box_events(tof, offsets, W, 3, 8, 2, 6)
    counts = np.bincount(pix, minlength=5 * 4).reshape(5, 4)
    np.testing.assert_array_equal(counts, image[3:8, 2:6])
    assert len(t) == image[3:8, 2:6].sum()


def _write_fake_nexus(path, rng, banks, n_events=3000):
    from subhkl.config import beamlines

    with h5py.File(path, "w") as f:
        for bank in banks:
            det = beamlines["MANDI"][str(bank)]
            grp = f.create_group(f"entry/bank{bank}_events")
            ids = rng.integers(0, det["m"] * det["n"], n_events) + det["offset"]
            grp["event_id"] = ids.astype(np.uint32)
            grp["event_time_offset"] = rng.uniform(1.5e4, 3.1e4, n_events).astype(
                np.float32
            )


def test_reduce_and_merge_carry_events(tmp_path):
    """reduce stores the events, merge-images concatenates them, and the
    merged images stay the histograms of their own events."""
    from subhkl.commands import run_merge_images, run_reduce

    rng = np.random.default_rng(2)
    reduced = []
    for i, banks in enumerate([(1, 2), (1, 2)]):
        nxs = tmp_path / f"MANDI_{i}.nxs.h5"
        _write_fake_nexus(nxs, rng, banks)
        out = tmp_path / f"MANDI_{i}.reduce.h5"
        run_reduce(str(nxs), str(out), "MANDI")
        assert has_events(str(out))
        reduced.append(out)

    no_events = tmp_path / "plain.reduce.h5"
    run_reduce(
        str(tmp_path / "MANDI_0.nxs.h5"), str(no_events), "MANDI", store_events=False
    )
    assert not has_events(str(no_events))
    with h5py.File(no_events, "r") as f_plain, h5py.File(reduced[0], "r") as f_ev:
        np.testing.assert_array_equal(f_plain["images"][()], f_ev["images"][()])

    merged = tmp_path / "merged.h5"
    run_merge_images(
        str(tmp_path / "MANDI_*.reduce.h5"), str(merged), 10, 10, 10, 90, 90, 90, "P 1"
    )
    with h5py.File(merged, "r") as f:
        images = f["images"][()]
        assert images.shape[0] == 4
        assert f["instrument/L1"][()] == 30.0
    with EventStore(str(merged)) as store:
        assert store.l1 == 30.0
        for i in range(4):
            tof, offsets = store.image_events(i, images[i])
            assert len(tof) == images[i].sum() == 3000
            # Sorted in time within every pixel.
            for p in np.flatnonzero(images[i].ravel() > 1)[:50]:
                assert np.all(np.diff(tof[offsets[p] : offsets[p + 1]]) >= 0)
        with pytest.raises(ValueError):
            store.image_events(0, images[0] + 1)


def test_gate_from_stack_finds_offset_and_width():
    from subhkl.search.tof_gate import gate_from_stack

    rng = np.random.default_rng(3)
    edges = np.linspace(-0.05, 0.05, 401)
    x = 0.5 * (edges[:-1] + edges[1:])
    base = 2000.0 * (1.0 + 2.0 * x)
    excess = 800.0 * np.exp(-0.5 * ((x - 0.004) / 0.002) ** 2)
    hist = rng.poisson(base + excess).astype(float)

    gate = gate_from_stack(edges, hist, coverage=0.95)
    assert gate is not None
    assert gate.lo < 0.004 < gate.hi
    # 95% of a Gaussian lies within +-1.96 sigma.
    assert 0.006 < gate.width < 0.011
    assert abs(0.5 * (gate.lo + gate.hi) - 0.004) < 0.001

    # No excess: nothing to read a gate from.
    assert gate_from_stack(edges, rng.poisson(base).astype(float)) is None


def test_gated_integration_separates_time_and_cuts_background():
    """Reflections are recovered inside their own time windows: two on the
    same pixels at different times come back with their own fluxes, and the
    uncertainty reflects the background of the gate, not of the band."""
    from subhkl.search.matrix_free import integrate_reflections_matrix_free
    from subhkl.search.tof_gate import integrate_reflections_tof_gated

    rng = np.random.default_rng(4)
    H = W = 96
    band = (15000.0, 31000.0)
    sigma_px, sigma_rel = 1.5, 0.002
    # (row, col, tof, flux); the last two share their pixels.
    truth = [
        (20.0, 22.0, 17000.0, 400.0),
        (25.0, 70.0, 21000.0, 150.0),
        (60.0, 30.0, 26000.0, 800.0),
        (70.0, 72.0, 18000.0, 300.0),
        (70.0, 72.0, 27000.0, 900.0),
        (45.0, 50.0, 23000.0, 0.0),
    ]
    image, tof = _synthetic_image_events(
        rng, (H, W), band, 40.0, truth, sigma_px, sigma_rel
    )
    offsets = np.concatenate([[0], np.cumsum(image.ravel())])
    n = len(truth)
    rs = np.array([p[0] for p in truth])
    cs = np.array([p[1] for p in truth])
    # Predictions are off by a common +0.3%, as a cell-scale error would be.
    tof_pred = np.array([p[2] for p in truth]) / 1.003
    flux = np.array([p[3] for p in truth])
    var = np.full(n, sigma_px**2 + 1.0 / 12.0, dtype=np.float32)

    rows, info = integrate_reflections_tof_gated(
        image[None].astype(np.float32),
        lambda f: (tof, offsets),
        np.zeros(n, dtype=int),
        rs,
        cs,
        tof_pred,
        np.full(n, band[0]),
        np.full(n, band[1]),
        var_us=var,
        var_vs=var,
        cov_uvs=np.zeros(n, dtype=np.float32),
        allow_negative=True,
    )
    gate = info["gate"]
    assert gate.source == "stack"
    assert gate.lo < 0.003 < gate.hi

    got, sig = rows[:, 0], rows[:, 4]
    assert np.all(sig > 0)
    # The gate holds ~95% of each peak in time.
    np.testing.assert_array_less(np.abs(got - 0.95 * flux), 5.0 * sig + 0.05 * flux)
    # The two reflections sharing pixels keep their own intensities.
    assert got[4] > 2.0 * got[3]
    # The empty prediction is consistent with zero.
    assert abs(got[5]) < 4.0 * sig[5]

    # On the whole band the same empty prediction sits on ~90x the
    # background, so its uncertainty is several times larger.  (Bright
    # peaks gain less: their own counting noise dominates either way.)
    full = integrate_reflections_matrix_free(
        image[None].astype(np.float32),
        np.zeros(1, dtype=int),
        rs[5:],
        cs[5:],
        var_us=var[5:],
        var_vs=var[5:],
        cov_uvs=np.zeros(1, dtype=np.float32),
        ref_sigma=1.0,
        max_sigma=4.0,
        allow_negative=True,
    )
    assert full[0, 4] > 3.0 * sig[5]


def test_gated_integration_reports_misaligned_predictions():
    """Without any Bragg excess near the predicted times the gate cannot be
    measured, and the integrator says so instead of integrating noise."""
    from subhkl.search.tof_gate import integrate_reflections_tof_gated

    rng = np.random.default_rng(5)
    H = W = 48
    image, tof = _synthetic_image_events(
        rng, (H, W), (15000.0, 31000.0), 20.0, [], 1.5, 0.002
    )
    offsets = np.concatenate([[0], np.cumsum(image.ravel())])
    n = 4
    var = np.full(n, 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="do not line up"):
        integrate_reflections_tof_gated(
            image[None].astype(np.float32),
            lambda f: (tof, offsets),
            np.zeros(n, dtype=int),
            np.array([12.0, 20.0, 30.0, 36.0]),
            np.array([14.0, 30.0, 18.0, 34.0]),
            np.full(n, 22000.0),
            np.full(n, 15000.0),
            np.full(n, 31000.0),
            var_us=var,
            var_vs=var,
            cov_uvs=np.zeros(n, dtype=np.float32),
        )


def test_write_events_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    images, tofs = [], []
    for _ in range(3):
        image, tof = _synthetic_image_events(rng, (8, 8), (1e4, 3e4), 3.0, [], 1.0, 0.0)
        images.append(image)
        tofs.append(tof)
    path = tmp_path / "ev.h5"
    with h5py.File(path, "w") as f:
        f["images"] = np.stack(images)
        write_events(f, tofs, l1=30.0)
    with EventStore(str(path)) as store:
        for i in range(3):
            tof, offsets = store.image_events(i, images[i])
            np.testing.assert_array_equal(tof, tofs[i])
            assert offsets[-1] == len(tofs[i])
