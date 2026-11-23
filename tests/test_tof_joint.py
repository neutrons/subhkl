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
