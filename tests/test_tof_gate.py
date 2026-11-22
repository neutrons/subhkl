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


def test_wavelength_band_follows_the_predictions():
    """The band comes from the predictions, not from the images file, and
    always holds every predicted wavelength."""
    from subhkl.search.tof_gate import wavelength_band

    predicted = np.array([1.05, 1.8, 2.9])
    # Predicted in 1-3 A while the images file still says 2-4 A.
    assert wavelength_band(predicted, (1.0, 3.0), (2.0, 4.0)) == (1.0, 3.0)
    # No band with the predictions: the images band, widened to hold them.
    assert wavelength_band(predicted, None, (2.0, 4.0)) == (1.05, 4.0)
    assert wavelength_band(predicted, None, None) == (1.05, 2.9)


def test_neighbor_plan_and_masked_median_match_dense_reference():
    """cKDTree sparse neighbor lookup and masked row median match dense O(n^2)
    and np.nanmedian references on crowded panels, including exact tie-breaking."""
    import warnings

    from subhkl.search import tof_gate

    rng = np.random.default_rng(7)

    # 1. _masked_row_median vs np.nanmedian across odd, even, and empty rows
    vals = rng.normal(10.0, 3.0, size=(120, 32))
    mask = rng.uniform(0.0, 1.0, size=(120, 32)) > 0.35
    mask[0] = False  # all-masked row
    mask[1, :3] = True
    mask[1, 3:] = False  # odd count
    mask[2, :4] = True
    mask[2, 4:] = False  # even count
    got_med = tof_gate._masked_row_median(vals, mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref_med = np.nanmedian(np.where(mask, vals, np.nan), axis=1)
    np.testing.assert_allclose(got_med, ref_med, rtol=0.0, atol=1e-14, equal_nan=True)

    # 2. _build_neighbor_plan vs dense O(n^2) reference (including exact distance ties)
    n = 600
    rs_f = rng.uniform(8.0, 120.0, n)
    cs_f = rng.uniform(8.0, 120.0, n)
    # Inject exact lattice distance ties around peak 0
    rs_f[0], cs_f[0] = 64.0, 64.0
    rs_f[1:5] = np.array([66.0, 62.0, 64.0, 64.0])
    cs_f[1:5] = np.array([64.0, 64.0, 66.0, 62.0])
    reach_f = rng.uniform(3.0, 6.5, n)
    in_band_f = rng.uniform(0.0, 1.0, n) > 0.08
    in_band_f[:5] = True
    b_lo_f = rng.integers(30, 180, size=n, dtype=np.int64)
    b_hi_f = b_lo_f + rng.integers(10, 30, size=n, dtype=np.int64)
    b_lo_f[1:5] = b_lo_f[0]
    b_hi_f[1:5] = b_hi_f[0]
    sb_edges = np.sort(
        rng.integers(10, 220, size=(n, 2, tof_gate.SIDEBAND_BINS + 1)), axis=2
    )
    keep0 = rng.uniform(0.0, 1.0, size=(n, 2, tof_gate.SIDEBAND_BINS)) > 0.1
    half = 11
    K = tof_gate.ATOM_PAD

    keep_kd, local_kd, has_kd, drop_kd = tof_gate._build_neighbor_plan(
        rs_f, cs_f, reach_f, in_band_f, b_lo_f, b_hi_f, sb_edges, keep0, half, K
    )

    dr = np.abs(rs_f[:, None] - rs_f[None, :])
    dc = np.abs(cs_f[:, None] - cs_f[None, :])
    near = (dr <= half + reach_f[None, :]) & (dc <= half + reach_f[None, :])
    np.fill_diagonal(near, False)
    near &= in_band_f[None, :]
    keep_dn = keep0.copy()
    pa, pj = np.nonzero(near)
    if len(pa):
        hit = (sb_edges[pa, :, :-1] < b_hi_f[pj, None, None]) & (
            sb_edges[pa, :, 1:] > b_lo_f[pj, None, None]
        )
        blocked = np.zeros(keep_dn.shape, dtype=bool)
        np.logical_or.at(blocked, pa, hit)
        keep_dn &= ~blocked

    touch = reach_f[:, None] + reach_f[None, :]
    joint = (
        near
        & (dr <= touch)
        & (dc <= touch)
        & (b_lo_f[:, None] < b_hi_f[None, :])
        & (b_hi_f[:, None] > b_lo_f[None, :])
    )
    d2 = np.where(joint, dr**2 + dc**2, np.inf)
    order = np.argsort(d2, axis=1, kind="stable")[:, : K - 1]
    picked = np.take_along_axis(d2, order, axis=1) < np.inf
    drop_dn = int((joint.sum(axis=1) > K - 1).sum())
    local_dn = np.concatenate([np.arange(n)[:, None], order], axis=1)
    has_dn = np.concatenate([np.ones((n, 1), dtype=bool), picked], axis=1)

    np.testing.assert_array_equal(keep_kd, keep_dn)
    np.testing.assert_array_equal(has_kd, has_dn)
    np.testing.assert_array_equal(
        np.where(has_kd, local_kd, -1), np.where(has_dn, local_dn, -1)
    )
    assert drop_kd == drop_dn


def test_gated_integration_fixed_chunks_avoid_per_bank_recompilation(monkeypatch):
    """Multi-chunk event scatter and tile solvers recover signal and do not
    trigger new XLA compilations when subsequent banks have different peak counts."""
    from subhkl.search import tof_gate

    monkeypatch.setattr(tof_gate, "TILE_CHUNK_SMALL", 32)
    monkeypatch.setattr(tof_gate, "TILE_CHUNK_LARGE", 64)
    monkeypatch.setattr(tof_gate, "EVENT_CHUNK_SMALL", 2048)
    monkeypatch.setattr(tof_gate, "EVENT_CHUNK", 8192)

    rng = np.random.default_rng(8)
    H = W = 64
    band = (15000.0, 31000.0)
    sigma_px, sigma_rel = 1.3, 0.002

    def _run_batch(counts_per_frame: list[int]):
        frames_l, rs_l, cs_l, tp_l, truth_flux_l = [], [], [], [], []
        ev_frames = []
        for f_idx, npk in enumerate(counts_per_frame):
            r_pk = rng.uniform(10.0, H - 10.0, npk)
            c_pk = rng.uniform(10.0, W - 10.0, npk)
            tp = rng.uniform(17000.0, 28000.0, npk)
            fl = np.where(np.arange(npk) < 6, 450.0, 0.0)
            peaks = [(r_pk[i], c_pk[i], tp[i], fl[i]) for i in range(npk)]
            img, tof = _synthetic_image_events(
                rng, (H, W), band, 3.0, peaks, sigma_px, sigma_rel
            )
            offsets = np.concatenate([[0], np.cumsum(img.ravel())])
            ev_frames.append((img, tof, offsets))
            frames_l.append(np.full(npk, f_idx, dtype=int))
            rs_l.append(r_pk)
            cs_l.append(c_pk)
            tp_l.append(tp)
            truth_flux_l.append(fl)

        images = np.stack([e[0] for e in ev_frames]).astype(np.float32)
        frames = np.concatenate(frames_l)
        rs = np.concatenate(rs_l)
        cs = np.concatenate(cs_l)
        tp = np.concatenate(tp_l)
        truth = np.concatenate(truth_flux_l)
        n_tot = len(frames)
        var = np.full(n_tot, sigma_px**2 + 1.0 / 12.0, dtype=np.float32)
        rows, _ = tof_gate.integrate_reflections_tof_gated(
            images,
            lambda f: (ev_frames[f][1], ev_frames[f][2]),
            frames,
            rs,
            cs,
            tp,
            np.full(n_tot, band[0]),
            np.full(n_tot, band[1]),
            var_us=var,
            var_vs=var,
            cov_uvs=np.zeros(n_tot, dtype=np.float32),
            allow_negative=True,
        )
        return rows, truth

    solve_fn = tof_gate._solve_tiles(False, False, True)
    tiles_fn = tof_gate._jit("tiles")
    snap_fn = tof_gate._jit("snap")
    box_fn = tof_gate._jit("box")
    cache_before = {
        "tiles": tiles_fn._cache_size(),
        "snap": snap_fn._cache_size(),
        "solve": solve_fn._cache_size(),
    }

    # Three banks with distinct peak counts [9, 75, 145] compile tile/snap/solve
    # at most once for this (H, W, chunk_size=64) configuration.
    rows_1, truth_1 = _run_batch([9, 75, 145])
    bright_1 = truth_1 > 0
    assert np.median(rows_1[bright_1, 0]) > 300.0

    cache_after_first = {
        "tiles": tiles_fn._cache_size(),
        "snap": snap_fn._cache_size(),
        "solve": solve_fn._cache_size(),
        "box": box_fn._cache_size(),
    }
    assert cache_after_first["tiles"] - cache_before["tiles"] <= 1
    assert cache_after_first["snap"] - cache_before["snap"] <= 1
    assert cache_after_first["solve"] - cache_before["solve"] <= 1

    # Second call with 4 completely different multi-chunk peak counts per bank:
    # zero additional XLA compilations may occur.
    rows_2, truth_2 = _run_batch([17, 110, 41, 180])
    bright_2 = truth_2 > 0
    assert np.median(rows_2[bright_2, 0]) > 300.0

    assert tiles_fn._cache_size() == cache_after_first["tiles"]
    assert snap_fn._cache_size() == cache_after_first["snap"]
    assert solve_fn._cache_size() == cache_after_first["solve"]
    assert box_fn._cache_size() == cache_after_first["box"]


def test_event_store_read_slab_and_multi_image_dense_neighbor_parity(
    tmp_path, monkeypatch, capsys
):
    """Coalesced EventStore.read_slab matches per-image reads (including empty
    and non-contiguous images), and multi-block GPU neighbor merge matches CPU
    cKDTree + plain-reader integration when > ATOM_PAD-1 peaks overlap."""
    from subhkl.search import tof_gate

    monkeypatch.setattr(tof_gate, "TILE_CHUNK_SMALL", 16)
    monkeypatch.setattr(tof_gate, "TILE_CHUNK_LARGE", 16)

    rng = np.random.default_rng(42)
    H = W = 48
    band = (15000.0, 31000.0)
    sigma_px, sigma_rel = 1.25, 0.002

    # Four HDF5 images: image 1 is deliberately empty (0 events); frames select [0, 1, 3].
    all_imgs, all_tofs = [], []
    for k in range(4):
        if k == 1:
            all_imgs.append(np.zeros((H, W), dtype=np.int64))
            all_tofs.append(np.zeros(0, dtype=np.float32))
        else:
            pks = [
                (24.0 + (i % 3) * 1.5, 24.0 + (i // 3) * 1.5, 22000.0, 320.0)
                for i in range(10)
            ]
            im, tf = _synthetic_image_events(
                rng, (H, W), band, 2.5, pks, sigma_px, sigma_rel
            )
            all_imgs.append(im)
            all_tofs.append(tf)

    h5_path = tmp_path / "slab_events.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("images", data=np.stack(all_imgs).astype(np.int32))
        write_events(h5, all_tofs)

    sel_keys = [0, 1, 3]
    sel_images = np.stack([all_imgs[k] for k in sel_keys]).astype(np.float32)

    with EventStore(h5_path) as store:
        tof_slab, off_slab, n_ev = store.read_slab(sel_keys, sel_images)
        ref_tofs, ref_off_rows = [], []
        base_ev = 0
        for slot, k in enumerate(sel_keys):
            tf_k, off_k = store.image_events(k, sel_images[slot])
            ref_tofs.append(tf_k.astype(np.float32))
            ref_off_rows.append((off_k + base_ev).astype(np.int32))
            base_ev += len(tf_k)
        ref_tof_cat = np.concatenate(ref_tofs)
        ref_off_mat = np.stack(ref_off_rows, axis=0)
        assert n_ev == len(ref_tof_cat)
        np.testing.assert_array_equal(tof_slab[:n_ev], ref_tof_cat)
        np.testing.assert_array_equal(off_slab, ref_off_mat)

        # Build >16 peaks on frame 0 & frame 2 (spanning multiple 16-peak GPU blocks)
        # including a 10-peak tight overlapping cluster (> ATOM_PAD - 1 = 7).
        frames_l, rs_l, cs_l, tp_l = [], [], [], []
        for f_slot in (0, 2):
            cluster_r = np.array(
                [24.0 + (i % 3) * 1.5 for i in range(10)]
                + list(rng.uniform(8.0, 40.0, 12))
            )
            cluster_c = np.array(
                [24.0 + (i // 3) * 1.5 for i in range(10)]
                + list(rng.uniform(8.0, 40.0, 12))
            )
            tp_f = np.array([22000.0] * 10 + list(rng.uniform(17000.0, 29000.0, 12)))
            frames_l.append(np.full(len(cluster_r), f_slot, dtype=np.int64))
            rs_l.append(cluster_r)
            cs_l.append(cluster_c)
            tp_l.append(tp_f)
        frames = np.concatenate(frames_l)
        rs = np.concatenate(rs_l)
        cs = np.concatenate(cs_l)
        tp = np.concatenate(tp_l)
        n_tot = len(frames)
        var = np.full(n_tot, sigma_px**2 + 1.0 / 12.0, dtype=np.float32)
        gate = tof_gate.TofGate(0.015, 0.030)

        def plain_reader(f_slot: int):
            return store.image_events(sel_keys[f_slot], sel_images[f_slot])

        def slab_reader(f_slot: int):
            return store.image_events(sel_keys[f_slot], sel_images[f_slot])

        slab_reader.event_store = store
        slab_reader.frame_img_keys = sel_keys

        rows_plain, info_plain = tof_gate.integrate_reflections_tof_gated(
            sel_images,
            plain_reader,
            frames,
            rs,
            cs,
            tp,
            np.full(n_tot, band[0]),
            np.full(n_tot, band[1]),
            var_us=var,
            var_vs=var,
            cov_uvs=np.zeros(n_tot, dtype=np.float32),
            gate=gate,
            allow_negative=True,
            show_progress=True,
        )
        out_plain = capsys.readouterr().out
        rows_slab, info_slab = tof_gate.integrate_reflections_tof_gated(
            sel_images,
            slab_reader,
            frames,
            rs,
            cs,
            tp,
            np.full(n_tot, band[0]),
            np.full(n_tot, band[1]),
            var_us=var,
            var_vs=var,
            cov_uvs=np.zeros(n_tot, dtype=np.float32),
            gate=gate,
            allow_negative=True,
            show_progress=True,
        )
        out_slab = capsys.readouterr().out

    assert "overlapping neighbours" in out_plain
    assert out_slab == out_plain
    np.testing.assert_allclose(info_slab["bg"], info_plain["bg"], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(rows_slab, rows_plain, rtol=1e-5, atol=1e-4)


def _curved_spectrum_events(rng, shape, band, bg_per_pixel, tau):
    """Background only, with a time spectrum falling off as exp(-t / tau)."""
    H, W = shape
    n = rng.poisson(bg_per_pixel * H * W)
    u = rng.uniform(0.0, 1.0, n)
    span = band[1] - band[0]
    tof = band[0] - tau * np.log(1.0 - u * (1.0 - np.exp(-span / tau)))
    flat = rng.integers(0, H * W, n)
    tof = tof.astype(np.float32)
    order = np.lexsort((tof, flat))
    image = np.bincount(flat, minlength=H * W).reshape(H, W)
    return image, tof[order]


def test_sideband_background_follows_a_curved_spectrum():
    """Empty predictions on a background whose spectrum is curved in time
    integrate to zero when the sidebands are scaled by the image's own
    spectrum, and to a systematic offset when the rate is taken as flat."""
    from subhkl.search.tof_gate import integrate_reflections_tof_gated, parse_gate

    rng = np.random.default_rng(7)
    H = W = 96
    band = (15000.0, 31000.0)
    image, tof = _curved_spectrum_events(rng, (H, W), band, 150.0, 2000.0)
    offsets = np.concatenate([[0], np.cumsum(image.ravel())])

    grid_rc = np.arange(14, 96 - 13, 14, dtype=float)
    rs, cs = (a.ravel() for a in np.meshgrid(grid_rc, grid_rc, indexing="ij"))
    n = len(rs)
    tof_pred = np.linspace(16000.0, 19000.0, n)
    var = np.full(n, 1.5**2, dtype=np.float32)

    def mean_z(measure):
        rows, _ = integrate_reflections_tof_gated(
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
            gate=parse_gate("-0.008,0.008"),
            sideband_measure=measure,
            allow_negative=True,
        )
        assert np.all(rows[:, 4] > 0)
        return float(np.mean(rows[:, 0] / rows[:, 4]))

    z_time = mean_z("time")
    z_spectrum = mean_z("spectrum")
    print(
        f"mean I/sigma of empty predictions: time {z_time:+.2f}, spectrum {z_spectrum:+.2f}"
    )
    # 36 independent empty predictions: the mean of unbiased I/sigma has a
    # standard error of 1/6.
    assert abs(z_spectrum) < 0.5
    assert abs(z_time) > 1.0


def test_spectrum_measure_falls_back_on_sparse_images():
    from subhkl.search.tof_gate import TimeGrid, _spectrum_measure

    grid = TimeGrid.spanning(15000.0, 31000.0, 1.0e-3)
    flat = grid.edge_time(np.arange(grid.nt + 1)) - grid.edge_time(0)
    rng = np.random.default_rng(8)
    sparse = rng.uniform(15000.0, 31000.0, 100).astype(np.float32)
    assert _spectrum_measure(sparse, grid, flat) is flat

    dense = rng.uniform(15000.0, 23000.0, 200000).astype(np.float32)
    measure = _spectrum_measure(dense, grid, flat)
    assert np.all(np.diff(measure) >= 0)
    assert measure[-1] == pytest.approx(flat[-1])
    # No events beyond 23000: those intervals have no length.
    late = int(np.ceil(grid.edge_index(24000.0)))
    assert measure[-1] - measure[late] == pytest.approx(0.0, abs=1e-6 * flat[-1])


def test_gpu_slab_spectrum_matches_cpu_spectrum_measure():
    import jax
    import jax.numpy as jnp

    from subhkl.search.tof_gate import TimeGrid, _jit, _spectrum_measure

    rng = np.random.default_rng(2026)
    grid = TimeGrid.spanning(15000.0, 31000.0, 1.0e-3)  # non-power-of-two nt
    assert (grid.nt & (grid.nt - 1)) != 0
    flat = grid.edge_time(np.arange(grid.nt + 1)) - grid.edge_time(0)
    nt_cap = max(2048, 1 << int(np.ceil(np.log2(max(grid.nt, 1)))))
    flat_pad = np.empty(nt_cap + 1, dtype=np.float32)
    flat_pad[: grid.nt + 1] = flat.astype(np.float32)
    flat_pad[grid.nt + 1 :] = np.float32(flat[-1])

    m_cap, n_m, n_pix = 8, 5, 64
    counts = [
        180_000,
        0,
        120,
        210_000,
        160_000,
    ]  # dense, empty, sparse, band-cut, dense
    cap = 600_000
    tof_buf = np.full(cap, 19000.0, dtype=np.float32)  # non-zero tail garbage
    off_buf = np.full(
        (m_cap, n_pix + 1), 599_999, dtype=np.int32
    )  # unused-slot garbage
    cursor = 0
    raw_tofs = []
    for m, ev in enumerate(counts):
        if m == 3:
            t_m = rng.uniform(15500.0, 22500.0, size=ev).astype(np.float32)
        else:
            t_m = rng.uniform(15000.0, 31000.0, size=ev).astype(np.float32)
        raw_tofs.append(t_m)
        if ev > 0:
            tof_buf[cursor : cursor + ev] = t_m
        off_buf[m, :] = np.linspace(cursor, cursor + ev, n_pix + 1, dtype=np.int32)
        cursor += ev

    gpu_out = np.asarray(
        _jit("slab_spectrum")(
            jax.device_put(tof_buf),
            jax.device_put(off_buf),
            jax.device_put(np.int32(n_m)),
            jax.device_put(np.int32(grid.nt)),
            jax.device_put(flat_pad),
            jnp.float32(1.0 / grid.t0),
            jnp.float32(1.0 / grid.dl),
        )
    )
    for m in range(n_m):
        ref = _spectrum_measure(raw_tofs[m], grid, flat)
        np.testing.assert_allclose(
            gpu_out[m, : grid.nt + 1], ref, rtol=1e-5, atol=2e-4 * flat[-1]
        )


def test_unresolvable_overlaps_are_censored():
    """Reflections packed closer than the spot width in one time window
    cannot be told apart: the joint solve censors them instead of reporting
    a value with a sigma that is only the regularisation constant."""
    import jax.numpy as jnp

    from subhkl.search.matrix_free import _solve_image_amplitudes_impl

    rng = np.random.default_rng(9)
    T = 41
    yy, xx = np.mgrid[0:T, 0:T].astype(np.float64)
    sigma_px, bg = 2.1, 1.5

    def solve(cols, flux, max_inflation):
        img = np.full((T, T), bg)
        for c, f in zip(cols, flux):
            img += (
                f
                * np.exp(-0.5 * ((yy - 20.0) ** 2 + (xx - c) ** 2) / sigma_px**2)
                / (2 * np.pi * sigma_px**2)
            )
        k = len(cols)
        var = jnp.full(k, sigma_px**2, dtype=jnp.float32)
        c, s = _solve_image_amplitudes_impl(
            jnp.array(rng.poisson(img).astype(np.float32)),
            jnp.full((T, T), bg, dtype=jnp.float32),
            jnp.ones((T, T), dtype=jnp.float32),
            jnp.full(k, 20.0, dtype=jnp.float32),
            jnp.array(cols, dtype=jnp.float32),
            var,
            var,
            jnp.zeros(k, dtype=jnp.float32),
            jnp.ones(k, dtype=jnp.float32),
            jnp.float32(0.0),
            allow_negative=True,
            max_variance_inflation=max_inflation,
        )
        return np.array(c), np.array(s)

    # Two reflections six pixels apart are resolved: nothing changes.
    cols, flux = [17.0, 23.0], [400.0, 250.0]
    rng = np.random.default_rng(9)
    c_all, s_all = solve(cols, flux, None)
    rng = np.random.default_rng(9)
    c_cut, s_cut = solve(cols, flux, 1000.0)
    np.testing.assert_array_equal(c_all, c_cut)
    np.testing.assert_array_equal(s_all, s_cut)
    assert np.all(s_cut > 0) and np.all(s_cut < 60)

    # Eight reflections half a pixel apart are not.
    cols = list(20.0 + 0.5 * (np.arange(8) - 3.5))
    flux = [300.0] * 8
    _c_free, s_free = solve(cols, flux, None)
    # Without the cut the sigma sits near the ridge's ceiling of 1000.
    assert np.median(s_free) > 500
    c_cut, s_cut = solve(cols, flux, 1000.0)
    assert np.all(s_cut == 0) and np.all(c_cut == 0)


def test_counts_scaled_ridge_keeps_sigma_meaningful_for_unresolved_overlaps():
    """With the ridge on the scale of the counts under each reflection, a
    reflection that cannot be told apart from its neighbours comes back
    within its own sigma, and isolated reflections are unchanged."""
    import jax.numpy as jnp

    from subhkl.search.matrix_free import _solve_image_amplitudes_impl

    T = 41
    yy, xx = np.mgrid[0:T, 0:T].astype(np.float64)
    bg, true_sigma = 1.5, 2.1

    def solve(rng, cols, flux, model_cols, model_sigma, ridge):
        img = np.full((T, T), bg)
        for c, f in zip(cols, flux):
            img += (
                f
                * np.exp(-0.5 * ((yy - 20.0) ** 2 + (xx - c) ** 2) / true_sigma**2)
                / (2 * np.pi * true_sigma**2)
            )
        k = len(model_cols)
        var = jnp.full(k, model_sigma**2, dtype=jnp.float32)
        c, s = _solve_image_amplitudes_impl(
            jnp.array(rng.poisson(img).astype(np.float32)),
            jnp.full((T, T), bg, dtype=jnp.float32),
            jnp.ones((T, T), dtype=jnp.float32),
            jnp.full(k, 20.0, dtype=jnp.float32),
            jnp.array(model_cols, dtype=jnp.float32),
            var,
            var,
            jnp.zeros(k, dtype=jnp.float32),
            jnp.ones(k, dtype=jnp.float32),
            jnp.float32(0.0),
            allow_negative=True,
            ridge=ridge,
        )
        return np.array(c, dtype=np.float64), np.array(s, dtype=np.float64)

    # Isolated reflection, exact model: the two ridges agree.
    c_f, s_f = solve(np.random.default_rng(1), [20.0], [400.0], [20.0], 2.1, "fixed")
    c_c, s_c = solve(np.random.default_rng(1), [20.0], [400.0], [20.0], 2.1, "counts")
    assert c_c[0] == pytest.approx(c_f[0], rel=0.02)
    assert s_c[0] == pytest.approx(s_f[0], rel=0.02)

    # A row of ten half a pixel apart, fitted as the integrator would: a
    # spot model that is too wide, positions off, two neighbours left out.
    rng = np.random.default_rng(2)
    z = {"fixed": [], "counts": []}
    for _ in range(25):
        cols = 20.0 + 0.5 * (np.arange(10) - 4.5)
        flux = rng.exponential(300.0, 10)
        keep = np.argsort(np.abs(np.arange(10) - 5))[:8]
        model_cols = cols[keep] + rng.normal(0.0, 0.5, 8)
        seed = int(rng.integers(1 << 30))
        for ridge, zs in z.items():
            c, s = solve(
                np.random.default_rng(seed), cols, flux, model_cols, 2.7, ridge
            )
            zs.append((c[0] - flux[5]) / s[0])
    z_fixed = np.abs(np.array(z["fixed"]))
    z_counts = np.abs(np.array(z["counts"]))
    print(
        f"|z| median / 90th pct: fixed {np.median(z_fixed):.2f} / "
        f"{np.percentile(z_fixed, 90):.2f}, counts {np.median(z_counts):.2f} / "
        f"{np.percentile(z_counts, 90):.2f}"
    )
    assert np.median(z_counts) < 1.2
    assert np.percentile(z_counts, 90) < 3.0
    assert np.percentile(z_fixed, 90) > 1.5 * np.percentile(z_counts, 90)
