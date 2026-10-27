"""Matrix-free amplitude-only integration: the global-solve contract.

One Poisson-likelihood solve per image over a fixed dictionary of
flux-normalized Gaussians, background pinned to the finder's rate map,
amplitudes strictly nonnegative.  These tests pin the properties that
motivated the design: flux recovery through overlap, no negative
amplitudes, no background flux leaking into empty atoms, and a Fisher
sigma that matches the actual estimator scatter.
"""

from __future__ import annotations

import numpy as np

from subhkl.search.matrix_free import integrate_reflections_matrix_free

H = W = 64
BG_RATE = 3.0  # [photons/Pixel]
VAR = 4.0  # [Pixel^2] isotropic peak variance


def _render(positions, fluxes, rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rate = np.full((H, W), BG_RATE)
    for (r, c), flux in zip(positions, fluxes):
        g = np.exp(-0.5 * ((yy - r) ** 2 + (xx - c) ** 2) / VAR)
        rate += flux * g / (2.0 * np.pi * VAR)
    return rng.poisson(rate).astype(np.float64)


def _solve(image, positions):
    n = len(positions)
    return integrate_reflections_matrix_free(
        image[None],
        np.zeros(n, dtype=int),
        np.array([p[0] for p in positions], dtype=float),
        np.array([p[1] for p in positions], dtype=float),
        np.full(n, VAR),
        np.full(n, VAR),
        np.zeros(n),
        ref_sigma=1.0,
        max_sigma=3.0,
    )


def test_overlapping_fluxes_recovered():
    """Two peaks 3 px apart share pixels; the joint solve must split the
    flux by shape, not by patch ownership."""
    positions = [(30.0, 28.0), (30.0, 31.0), (14.0, 48.0)]
    truth = np.array([4000.0, 1500.0, 800.0])
    errs = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        image = _render(positions, truth, rng)
        out = _solve(image, positions)
        errs.append((out[:, 0] - truth) / out[:, 4])
    errs = np.array(errs)
    # Mean pull per peak stays within ~2 standard errors of zero.
    pulls = errs.mean(axis=0) * np.sqrt(len(errs))
    assert np.all(np.abs(pulls) < 3.0), pulls


def test_amplitudes_never_negative():
    """37.9% of unconstrained-WLS amplitudes were negative on real data;
    the constrained solve returns none, even for atoms placed on pure
    background."""
    positions = [(20.0, 20.0), (45.0, 45.0), (10.0, 50.0), (50.0, 10.0)]
    truth = np.array([2000.0, 0.0, 0.0, 0.0])
    saw_boundary = False
    for seed in range(4):
        rng = np.random.default_rng(100 + seed)
        image = _render(positions, truth, rng)
        out = _solve(image, positions)
        assert np.all(out[:, 0] >= 0.0)
        # Boundary-pinned rows are censored, not measured: sigI = 0 marks
        # them for the exporter to drop; measured rows keep sigI > 0.
        at_boundary = out[:, 0] <= 0.0
        assert np.all(out[at_boundary, 4] == 0.0)
        assert np.all(out[~at_boundary, 4] > 0.0)
        saw_boundary |= bool(at_boundary.any())
    assert saw_boundary  # the scenario must actually exercise the marker


def test_background_does_not_leak_into_empty_atoms():
    """An atom over pure background must report (nearly) zero flux, not
    a share of the pedestal: the faint-half inflation of the loose
    global WLS measured 33x on cg4d-t4-lysozyme."""
    positions = [(20.0, 20.0), (45.0, 45.0)]
    truth = np.array([3000.0, 0.0])
    recovered = []
    for seed in range(6):
        rng = np.random.default_rng(200 + seed)
        image = _render(positions, truth, rng)
        out = _solve(image, positions)
        recovered.append(out[1, 0])
    # The empty atom's flux stays below its own 2 sigma (sigma ~ sqrt of
    # background photons under the footprint, ~60 here).
    assert np.mean(recovered) < 2.0 * np.sqrt(2 * np.pi * VAR * BG_RATE) * 2.0


def test_fisher_sigma_matches_estimator_scatter():
    """The reported sigma is the Fisher information at the optimum; the
    empirical scatter of the flux estimate over noise realizations must
    match it, not just rank it."""
    positions = [(32.0, 32.0)]
    truth = np.array([2500.0])
    fluxes, sigmas = [], []
    for seed in range(12):
        rng = np.random.default_rng(300 + seed)
        image = _render(positions, truth, rng)
        out = _solve(image, positions)
        fluxes.append(out[0, 0])
        sigmas.append(out[0, 4])
    ratio = np.std(fluxes, ddof=1) / np.mean(sigmas)
    assert 0.5 < ratio < 2.0, ratio


def test_positions_snap_to_the_true_center():
    """A prediction 1 px off must recover the true flux through the
    log-parabolic snap, and report the snapped position."""
    true_pos = [(30.0, 30.0)]
    offset_pos = [(31.0, 29.0)]
    truth = np.array([5000.0])
    rng = np.random.default_rng(42)
    image = _render(true_pos, truth, rng)
    out = _solve(image, offset_pos)
    assert abs(out[0, 1] - 30.0) < 0.5
    assert abs(out[0, 2] - 30.0) < 0.5
    assert abs(out[0, 0] - truth[0]) < 5.0 * out[0, 4]


def test_weak_peaks_are_measured_not_gated():
    """With a permissive fp_target (>= the test count) the admission
    threshold is z = 0 and every reflection is measured: a peak far
    below any detection gate still comes back with its flux.  The
    first cut transplanted the finder's width-taxed threshold (~48
    sigma) and returned exact zeros for 97.8% of cg4d-t4-lysozyme;
    CC(1/2) collapsed to 0.02."""
    positions = [(30.0, 30.0), (14.0, 48.0)]
    truth = np.array([30.0, 3000.0])  # weak peak ~ noise scale
    weak = []
    for seed in range(10):
        rng = np.random.default_rng(600 + seed)
        image = _render(positions, truth, rng)
        out = _solve(image, positions)
        weak.append(out[0, 0])
    weak = np.array(weak)
    # Not gated: the weak peak is nonzero in most realizations (the
    # nonnegative MLE only pins genuinely negative-fluctuation draws).
    assert np.mean(weak > 0) > 0.5, weak
    # And measured: mean recovery within a few standard errors of truth
    # (nonnegative truncation biases a 1.5-sigma peak up by ~2%, far
    # below this tolerance).
    se = np.std(weak, ddof=1) / np.sqrt(len(weak))
    assert abs(np.mean(weak) - truth[0]) < 4.0 * se, (np.mean(weak), se)


def test_calibrated_admission_censors_subthreshold_only():
    """The L1 gate is the finder's admission logic at the integrator's
    test count: with a strict fp_target the sub-threshold peak is
    censored (flux 0, sigI 0 -> dropped by the exporter), while the
    bright peak is admitted AND debiased -- its flux must be unbiased,
    not shrunk by z*SE (~2.6 sigma here, far above the tolerance)."""
    positions = [(20.0, 20.0), (45.0, 45.0)]
    truth = np.array([20000.0, 12.0])  # z >> gate, z << gate
    n = len(positions)
    bright, weak_censored = [], 0
    for seed in range(6):
        rng = np.random.default_rng(800 + seed)
        image = _render(positions, truth, rng)
        out = integrate_reflections_matrix_free(
            image[None],
            np.zeros(n, dtype=int),
            np.array([p[0] for p in positions]),
            np.array([p[1] for p in positions]),
            np.full(n, VAR),
            np.full(n, VAR),
            np.zeros(n),
            ref_sigma=1.0,
            max_sigma=3.0,
            fp_target=0.01,  # z = Phi^-1(1 - 0.005) ~ 2.58
        )
        bright.append((out[0, 0] - truth[0]) / out[0, 4])
        if out[1, 0] == 0.0:
            assert out[1, 4] == 0.0  # censored, not "0 +- sigma"
            weak_censored += 1
    # the weak peak (z ~ 0.5) is censored in every realization
    assert weak_censored == 6
    # the admitted peak is debiased: mean pull consistent with zero
    assert abs(np.mean(bright)) * np.sqrt(len(bright)) < 3.0, np.mean(bright)


def test_calibrated_admission_z_values():
    from subhkl.search.ssn import calibrated_admission_z

    # finder-scale test count -> the familiar ~5-sigma regime
    assert 4.5 < calibrated_admission_z(2.6e5, 0.05) < 5.5
    # integrator-scale
    assert 3.0 < calibrated_admission_z(16600, 1.0) < 4.5
    # permissive target admits everything
    assert calibrated_admission_z(10, 10) == 0.0
    assert calibrated_admission_z(10, 50) == 0.0


def test_masked_rate_map_is_unbiased_under_peaks():
    """The unmasked quantile estimator reads +1 photon/pixel or more
    under a bright peak (its 'bounded positive bias'); with the
    footprint masked it must recover the true rate from the ring."""
    import jax.numpy as jnp

    from subhkl.search.matrix_free import _footprint_mask
    from subhkl.search.sparse_rbf import compute_rate_batch

    positions = [(30.0, 28.0), (30.0, 31.0)]
    truth = np.array([4000.0, 1500.0])
    under_masked, under_raw = [], []
    for seed in range(6):
        rng = np.random.default_rng(400 + seed)
        image = _render(positions, truth, rng).astype(np.float32)
        valid = _footprint_mask(
            1,
            H,
            W,
            np.zeros(2, dtype=int),
            np.array([p[0] for p in positions]),
            np.array([p[1] for p in positions]),
            np.full(2, VAR),
            np.full(2, VAR),
            np.zeros(2),
        )
        rm_masked = np.array(
            compute_rate_batch(jnp.array(image[None]), 15, valid=jnp.array(valid))
        )[0]
        rm_raw = np.array(compute_rate_batch(jnp.array(image[None]), 15))[0]
        under_masked.append(rm_masked[30, 28])
        under_raw.append(rm_raw[30, 28])
    assert abs(np.mean(under_masked) - BG_RATE) < 0.5, np.mean(under_masked)
    # and the raw estimator really was biased here, else this test is vacuous
    assert np.mean(under_raw) - BG_RATE > 1.0, np.mean(under_raw)


def test_static_mask_removes_ridge_bias():
    """A bright static ridge crossing near a peak biases its amplitude
    (the rate map cannot follow a structure narrower than its window);
    masking the ridge as missing data must restore the flux, exactly the
    contract of the static-mask files."""
    positions = [(30.0, 30.0)]
    truth = np.array([2000.0])
    biased, masked = [], []
    for seed in range(6):
        rng = np.random.default_rng(500 + seed)
        image = _render(positions, truth, rng)
        # a 2-px ridge 4 px from the peak center, 12 photons/px on bg 3
        image[:, 34:36] += rng.poisson(12.0, size=(H, 2))
        valid = np.ones((1, H, W), dtype=np.float32)
        valid[:, :, 33:37] = 0.0

        out_b = _solve(image, positions)
        n = len(positions)
        out_m = integrate_reflections_matrix_free(
            image[None],
            np.zeros(n, dtype=int),
            np.array([p[0] for p in positions]),
            np.array([p[1] for p in positions]),
            np.full(n, VAR),
            np.full(n, VAR),
            np.zeros(n),
            ref_sigma=1.0,
            max_sigma=3.0,
            static_valid=valid,
        )
        biased.append(out_b[0, 0] - truth[0])
        masked.append((out_m[0, 0] - truth[0]) / out_m[0, 4])
    # masked: mean pull consistent with zero
    assert abs(np.mean(masked)) * np.sqrt(len(masked)) < 3.0, np.mean(masked)
    # and the ridge really biases the unmasked solve, else vacuous
    assert abs(np.mean(biased)) > 30.0, np.mean(biased)


def _render_profile(positions, fluxes, rng, trunk):
    """Scene whose peaks follow an arbitrary radial trunk f(m), unit flux."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    mm = np.linspace(0, 6, 601)
    norm = np.trapezoid(trunk(mm) * mm, mm) * 2 * np.pi * VAR
    rate = np.full((H, W), BG_RATE)
    for (r, c), flux in zip(positions, fluxes):
        m = np.sqrt(((yy - r) ** 2 + (xx - c) ** 2) / VAR)
        rate += flux * trunk(m) / norm
    return rng.poisson(rate).astype(np.float64)


def _solve_profile(image, positions, profile):
    n = len(positions)
    return integrate_reflections_matrix_free(
        image[None],
        np.zeros(n, dtype=int),
        np.array([p[0] for p in positions], dtype=float),
        np.array([p[1] for p in positions], dtype=float),
        np.full(n, VAR),
        np.full(n, VAR),
        np.zeros(n),
        ref_sigma=1.0,
        max_sigma=3.0,
        profile=profile,
        profile_min_peaks=5,
    )


def test_measured_trunk_removes_template_mismatch_bias():
    """Truth peaks are Gaussian-cored with TRUNCATED tails -- the shape
    the t4 stack measured (no detectable flux beyond m ~ 2.5 where the
    Gaussian still claims 2%).  Gaussian atoms mis-apportion that
    phantom tail mass under the Poisson loss; the measured trunk must
    not.  (A flat-topped trunk would also defeat the log-parabolic
    position snap, whose curvature floor turns a curvature-free core
    into noise-driven +-1.5 px shifts -- a separate, known limitation
    inherited from the patch integrator; this test isolates the
    template shape.)  The census needs many peaks, so the scene
    carries a grid of isolated bright ones."""

    def truncated_gauss(m):
        # 30% wider than the model tensor claims, tails cut at m = 2.8:
        # the two deviations the t4 stack measured (there: 15% and 2.5).
        m = np.asarray(m, dtype=float)
        return np.exp(-0.5 * (m / 1.3) ** 2) * (m < 2.8)

    # sub-pixel dither as on a real detector: in-phase integer positions
    # alias the census against the pixel lattice
    positions = [
        (r + 0.31 * ((r + c) % 3), c + 0.47 * ((r - c) % 3 - 1))
        for r in (10.0, 30.0, 50.0)
        for c in (10.0, 30.0, 50.0)
    ]
    truth = np.full(len(positions), 4000.0)
    pulls_g, pulls_a = [], []
    for seed in range(6):
        rng = np.random.default_rng(700 + seed)
        image = _render_profile(positions, truth, rng, truncated_gauss)
        out_g = _solve_profile(image, positions, "gaussian")
        out_a = _solve_profile(image, positions, "auto")
        pulls_g.append((out_g[:, 0] - truth) / out_g[:, 4])
        pulls_a.append((out_a[:, 0] - truth) / out_a[:, 4])
    bias_g = np.mean(pulls_g)
    bias_a = np.mean(pulls_a)
    # Gaussian atoms on a truncated truth are measurably biased...
    assert abs(bias_g) > 1.0, bias_g
    # ...and the measured trunk removes most of it.
    assert abs(bias_a) < abs(bias_g) / 2.0, (bias_g, bias_a)


def test_auto_profile_falls_back_to_gaussian_when_starved():
    """One faint peak cannot seed a trunk; auto must silently equal the
    Gaussian solve rather than fit a profile to nothing."""
    positions = [(30.0, 30.0)]
    rng = np.random.default_rng(42)
    image = _render(positions, [200.0], rng)
    out_g = _solve_profile(image, positions, "gaussian")
    out_a = _solve_profile(image, positions, "auto")
    np.testing.assert_allclose(out_a[:, 0], out_g[:, 0], rtol=1e-6)


def test_empty_input_returns_empty():
    out = integrate_reflections_matrix_free(
        np.zeros((1, H, W)),
        np.array([], dtype=int),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        ref_sigma=1.0,
        max_sigma=3.0,
    )
    assert out.shape == (0, 5)


def test_multi_image_scatter_back():
    """Peaks interleaved across two images land back on their own rows."""
    rng = np.random.default_rng(7)
    img0 = _render([(20.0, 20.0)], [3000.0], rng)
    img1 = _render([(40.0, 40.0)], [1200.0], rng)
    out = integrate_reflections_matrix_free(
        np.stack([img0, img1]),
        np.array([1, 0]),
        np.array([40.0, 20.0]),
        np.array([40.0, 20.0]),
        np.full(2, VAR),
        np.full(2, VAR),
        np.zeros(2),
        ref_sigma=1.0,
        max_sigma=3.0,
    )
    assert abs(out[0, 0] - 1200.0) < 5.0 * out[0, 4]
    assert abs(out[1, 0] - 3000.0) < 5.0 * out[1, 4]
