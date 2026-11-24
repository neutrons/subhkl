"""Time-resolved joint fit of all reflections on a detector image."""

import numpy as np
import pytest

from subhkl.search.tof_gate import TofGate
from subhkl.search.tof_joint import integrate_reflections_tof_joint, stack_profile


def _events(rng, shape, band, bg_per_pixel, peaks, sigma_px, sigma_rel):
    """Events of one image sorted by (flat pixel, tof): a background flat
    in time plus Bragg peaks given as (row, col, tof [microsecond], flux)."""
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


def _integrate(image, tof, truth, band, sigma_px, scale=1.0, **kwargs):
    offsets = np.concatenate([[0], np.cumsum(image.ravel())])
    n = len(truth)
    var = np.full(n, sigma_px**2 + 1.0 / 12.0, dtype=np.float32)
    return integrate_reflections_tof_joint(
        image[None].astype(np.float32),
        lambda f: (tof, offsets),
        np.zeros(n, dtype=int),
        np.array([p[0] for p in truth]),
        np.array([p[1] for p in truth]),
        np.array([p[2] for p in truth]) / scale,
        np.full(n, band[0]),
        np.full(n, band[1]),
        var_us=var,
        var_vs=var,
        cov_uvs=np.zeros(n, dtype=np.float32),
        allow_negative=True,
        **kwargs,
    )


def test_stack_profile_is_a_cumulative_share():
    edges = np.linspace(-0.05, 0.05, 101)
    x = 0.5 * (edges[:-1] + edges[1:])
    hist = 1000.0 + 5000.0 * np.exp(-0.5 * ((x - 0.004) / 0.003) ** 2)
    px, cdf = stack_profile(edges, hist)
    assert px[0] < 0.004 < px[-1]
    assert cdf[0] == 0.0 and cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cdf) >= 0)
    assert stack_profile(edges, np.full(100, 1000.0)) is None


def test_joint_fit_recovers_total_flux_and_separates_time():
    """Each reflection comes back with its whole flux; two on the same
    pixels at different times keep their own, and an empty prediction is
    consistent with zero."""
    rng = np.random.default_rng(4)
    band = (15000.0, 31000.0)
    sigma_px, sigma_rel = 1.5, 0.002
    truth = [
        (20.0, 22.0, 17000.0, 400.0),
        (25.0, 70.0, 21000.0, 150.0),
        (60.0, 30.0, 26000.0, 800.0),
        (70.0, 72.0, 18000.0, 300.0),
        (70.0, 72.0, 27000.0, 900.0),
        (45.0, 50.0, 23000.0, 0.0),
    ]
    image, tof = _events(rng, (96, 96), band, 40.0, truth, sigma_px, sigma_rel)
    # Predictions are off by a common +0.3%, as a cell-scale error would be.
    rows, info = _integrate(image, tof, truth, band, sigma_px, scale=1.003)
    assert info["gate"].source == "stack"
    assert info["gate"].lo < 0.003 < info["gate"].hi
    flux = np.array([p[3] for p in truth])
    got, sig = rows[:, 0], rows[:, 4]
    assert np.all(sig > 0)
    np.testing.assert_array_less(np.abs(got - flux), 5.0 * sig + 0.05 * flux)
    assert got[4] > 2.0 * got[3]
    assert abs(got[5]) < 4.0 * sig[5]
    assert info["dropped"] == 0


def test_joint_fit_resolves_neighbours_a_gate_would_merge():
    """Reflections on the same pixels and 1.2% apart in time lie inside one
    another's gate, yet the time profile tells them apart."""
    rng = np.random.default_rng(7)
    band = (15000.0, 31000.0)
    sigma_px, sigma_rel = 1.5, 0.002
    chain = [
        (48.0, 48.0, 20000.0 * 1.012**i, f)
        for i, f in enumerate((600.0, 0.0, 900.0, 200.0))
    ]
    # A second image of isolated reflections sets the measured time profile.
    lone = [
        (12.0 + 14.0 * (i // 6), 12.0 + 14.0 * (i % 6), 16000.0 + 400.0 * i, 3000.0)
        for i in range(36)
    ]
    data = [
        _events(rng, (96, 96), band, 40.0, peaks, sigma_px, sigma_rel)
        for peaks in (chain, lone)
    ]
    offsets = [np.concatenate([[0], np.cumsum(im.ravel())]) for im, _ in data]
    truth = chain + lone
    n = len(truth)
    var = np.full(n, sigma_px**2 + 1.0 / 12.0, dtype=np.float32)
    rows, info = integrate_reflections_tof_joint(
        np.stack([im for im, _ in data]).astype(np.float32),
        lambda f: (data[f][1], offsets[f]),
        np.array([0] * len(chain) + [1] * len(lone)),
        np.array([p[0] for p in truth]),
        np.array([p[1] for p in truth]),
        np.array([p[2] for p in truth]),
        np.full(n, band[0]),
        np.full(n, band[1]),
        var_us=var,
        var_vs=var,
        cov_uvs=np.zeros(n, dtype=np.float32),
        allow_negative=True,
    )
    flux = np.array([p[3] for p in truth])
    got, sig = rows[:, 0], rows[:, 4]
    assert np.all(sig > 0)
    np.testing.assert_array_less(np.abs(got - flux), 5.0 * sig + 0.05 * flux)
    assert np.all(info["inflation"][:4] < 10.0)
    assert info["dropped"] == 0


def test_joint_fit_censors_what_no_data_can_separate():
    """Two predictions at the same place and time: only their sum is
    determined.  They are censored; a separate reflection is unaffected."""
    rng = np.random.default_rng(9)
    band = (15000.0, 31000.0)
    sigma_px, sigma_rel = 1.5, 0.002
    truth = [
        (40.0, 40.0, 20000.0, 500.0),
        (40.0, 40.0, 20000.0, 500.0),
        (70.0, 70.0, 25000.0, 700.0),
        (20.0, 70.0, 18000.0, 400.0),
    ]
    image, tof = _events(rng, (96, 96), band, 40.0, truth, sigma_px, sigma_rel)
    rows, info = _integrate(image, tof, truth, band, sigma_px)
    sig = rows[:, 4]
    assert sig[0] == 0.0 and sig[1] == 0.0
    assert info["dropped"] == 1
    assert sig[2] > 0 and abs(rows[2, 0] - 700.0) < 5.0 * sig[2] + 35.0
    assert sig[3] > 0 and abs(rows[3, 0] - 400.0) < 5.0 * sig[3] + 20.0


def test_joint_fit_with_a_given_gate_and_band_edge():
    """A given gate is a flat time profile; a reflection whose profile lies
    mostly outside the band is censored."""
    rng = np.random.default_rng(11)
    band = (15000.0, 31000.0)
    sigma_px, sigma_rel = 1.5, 0.002
    truth = [
        (30.0, 30.0, 20000.0, 600.0),
        (60.0, 60.0, 26000.0, 300.0),
        (30.0, 70.0, 14950.0, 300.0),
    ]
    image, tof = _events(rng, (96, 96), band, 40.0, truth, sigma_px, sigma_rel)
    rows, info = _integrate(
        image, tof, truth, band, sigma_px, gate=TofGate(lo=-0.008, hi=0.008)
    )
    assert info["gate"].lo == -0.008
    got, sig = rows[:, 0], rows[:, 4]
    assert abs(got[0] - 600.0) < 5.0 * sig[0] + 30.0
    assert abs(got[1] - 300.0) < 5.0 * sig[1] + 15.0
    assert sig[2] == 0.0


def test_joint_fit_multi_image_event_store_matches_callable(tmp_path):
    """Multi-image EventStore slab batching (spanning full + partial tail
    batches and an empty image) matches the callable reader path."""
    import h5py

    from subhkl.io.events import EventStore, write_events

    rng = np.random.default_rng(21)
    n_imgs = 11  # exceeds MAX_BATCH_IMAGES=8 -> full batch + partial tail batch
    shape = (64, 64)
    band = (15000.0, 30000.0)
    sigma_px, sigma_rel = 1.4, 0.002

    images, tofs = [], []
    frames, rs, cs, tpreds, truth_flux = [], [], [], [], []
    for f in range(n_imgs):
        peaks_f = []
        if f != 5:  # leave one image without predicted reflections
            for p in range(6):
                r = 12.0 + 7.5 * p
                c = 14.0 + 7.0 * ((p + f) % 6)
                t = 16500.0 + 1800.0 * p
                flx = float(300 + 80 * p)
                peaks_f.append((r, c, t, flx))
                frames.append(f)
                rs.append(r)
                cs.append(c)
                tpreds.append(t)
                truth_flux.append(flx)
        im, tf = _events(rng, shape, band, 20.0, peaks_f, sigma_px, sigma_rel)
        images.append(im)
        tofs.append(tf)

    images_arr = np.stack(images).astype(np.float32)
    offsets = [np.concatenate([[0], np.cumsum(im.ravel())]) for im in images]
    h5_path = tmp_path / "multi_events.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("images", data=images_arr)
        write_events(hf, tofs, l1=30.0)

    n = len(frames)
    var = np.full(n, sigma_px**2 + 1.0 / 12.0, dtype=np.float32)
    common = {
        "var_us": var,
        "var_vs": var,
        "cov_uvs": np.zeros(n, dtype=np.float32),
        "gate": TofGate(lo=-0.01, hi=0.01),
        "allow_negative": True,
    }
    args = (
        images_arr,
        np.array(frames, dtype=int),
        np.array(rs),
        np.array(cs),
        np.array(tpreds),
        np.full(n, band[0]),
        np.full(n, band[1]),
    )

    rows_call, info_call = integrate_reflections_tof_joint(
        args[0], lambda f: (tofs[f], offsets[f]), *args[1:], **common
    )
    with EventStore(h5_path) as store:
        # Also verify non-contiguous + tail chunk reads match image_events bit-for-bit.
        pick = [0, 3, 4, 10]
        tof_slab, off_slab, total_ev = store.read_slab(pick, images_arr[pick])
        cur = 0
        for k, idx in enumerate(pick):
            t_ref, o_ref = store.image_events(idx, images_arr[idx])
            assert np.array_equal(tof_slab[cur : cur + len(t_ref)], t_ref)
            assert np.array_equal(off_slab[k], o_ref + cur)
            cur += len(t_ref)
        assert cur == total_ev

        def store_reader(f):
            return store.image_events(int(f), images_arr[int(f)])

        store_reader.event_store = store
        rows_store, info_store = integrate_reflections_tof_joint(
            args[0], store_reader, *args[1:], **common
        )

    np.testing.assert_allclose(rows_store, rows_call, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(info_store["bg"], info_call["bg"], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_less(
        np.abs(rows_store[:, 0] - np.array(truth_flux)),
        5.0 * rows_store[:, 4] + 0.05 * np.array(truth_flux),
    )


def test_joint_fit_escalates_when_cluster_exceeds_default_nbr_pad():
    """When more than DEFAULT_NBR_PAD reflections mutually overlap on one
    image, the GPU neighbour cap escalates cleanly without dropping active
    neighbours."""
    from subhkl.search.tof_joint import DEFAULT_NBR_PAD

    rng = np.random.default_rng(33)
    band = (15000.0, 31000.0)
    sigma_px, sigma_rel = 1.6, 0.002
    n_cluster = DEFAULT_NBR_PAD + 6
    cluster = [
        (
            32.0 + (i % 6 - 2.5) * 1.1,
            32.0 + (i // 6 - 2.5) * 1.1,
            21000.0 * (1.0 + 0.0025 * ((i % 7) - 3)),
            250.0,
        )
        for i in range(n_cluster)
    ]
    isolated = [(12.0, 12.0, 17000.0, 700.0), (52.0, 52.0, 27000.0, 500.0)]
    truth = cluster + isolated
    image, tof = _events(rng, (64, 64), band, 25.0, truth, sigma_px, sigma_rel)
    rows, _info = _integrate(
        image,
        tof,
        truth,
        band,
        sigma_px,
        gate=TofGate(lo=-0.008, hi=0.008),
    )
    assert np.all(np.isfinite(rows[:, 0]))
    assert np.all(np.isfinite(rows[:, 4]))
    # Total recovered cluster flux and isolated fluxes match truth.
    cluster_sum = rows[:n_cluster, 0].sum()
    assert abs(cluster_sum - 250.0 * n_cluster) < 0.15 * (250.0 * n_cluster)
    assert abs(rows[-2, 0] - 700.0) < 5.0 * rows[-2, 4] + 35.0
    assert abs(rows[-1, 0] - 500.0) < 5.0 * rows[-1, 4] + 25.0
