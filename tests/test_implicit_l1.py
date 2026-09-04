"""Does the zero-count pixels' implicit L1 penalty replace the threshold?

The Poisson loss terms of zero-count pixels are exactly linear in the
intensity (see the endgame note in matrix_free: lam_aug = lam + A^T
1_{y=0}), so the maximum-likelihood nonnegative fit -- alpha = 0, no
explicit threshold -- is already an implicitly L1-penalized problem.  The
question is whether that penalty *suffices*.  These are the measurements;
the mechanism is the size of an atom's footprint relative to what it
explains.

Image domain: one atom explains one bump.  Activation at c = 0 happens
whenever the matched-filter correlation of the counts exceeds its own
background expectation -- a mean-zero test that any upward fluctuation
passes -- so the ML fit assigns spurious flux everywhere the noise
fluctuates up, at every background level, and a single stray count becomes
a full atom as soon as the background falls below ~1 count per PSF
footprint (2 pi sigma^2 b < 1).  Measured below with Richardson-Lucy,
which converges to exactly this ML fit.  This is why the finder's explicit
false-alarm calibration (effective_alpha) exists and cannot be dropped.

SO(3) domain (subhkl.search.spherical): one atom explains *hundreds* of
directions at once.  The separation between a true orientation and the
null grows like sqrt(matches), so admission needs no tuned threshold
there; measured in test_spherical.py as null z-scores of 15+ at 30%
outliers with lam = 0.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

SIG = 1.5  # PSF width [px]
N = 64
PEAKS = [(16, 16), (16, 48), (48, 16)]
AMP = 12.0  # peak amplitude [counts/px]


def _psf():
    yy, xx = np.mgrid[-8:9, -8:9]
    return np.exp(-(xx**2 + yy**2) / (2 * SIG**2))


def _ml_fit(y, b, n_iter=2000):
    """Richardson-Lucy with flat background: the ML nonneg Poisson fit,
    i.e. the alpha = 0 limit where the implicit penalty is the only one."""
    psf = _psf()
    c = np.full_like(y, 0.01, dtype=float)
    a_t1 = psf.sum()
    for _ in range(n_iter):
        u = b + fftconvolve(c, psf, mode="same")
        c = c * fftconvolve(y / u, psf[::-1, ::-1], mode="same") / a_t1
    return fftconvolve(c, psf, mode="same")  # fitted signal intensity


def _frame(rng, b):
    rate = np.full((N, N), b)
    for r0, c0 in PEAKS:
        rate += AMP * np.exp(
            -((np.arange(N)[:, None] - r0) ** 2 + (np.arange(N)[None, :] - c0) ** 2)
            / (2 * SIG**2)
        )
    y = rng.poisson(rate).astype(float)
    y[48, 48] = 1.0  # planted isolated stray count in an empty corner
    return y


def _fluxes(m):
    stray = m[45:52, 45:52].sum()
    pk = sum(m[r0 - 3 : r0 + 4, c0 - 3 : c0 + 4].sum() for r0, c0 in PEAKS)
    mask = np.ones((N, N), bool)
    for r0, c0 in PEAKS:
        mask[r0 - 5 : r0 + 6, c0 - 5 : c0 + 6] = False
    mask[43:54, 43:54] = False
    return stray, pk, m[mask].sum()


def test_peaks_are_recovered_without_a_threshold():
    rng = np.random.default_rng(5)
    true_flux = 3 * AMP * 2 * np.pi * SIG**2
    for b in (0.44, 0.02):
        _, pk, _ = _fluxes(_ml_fit(_frame(rng, b), b))
        assert abs(pk - true_flux) / true_flux < 0.15


def test_no_false_alarm_control_at_moderate_background():
    """At b = 0.44 the ML fit assigns hundreds of counts of spurious flux
    to background fluctuations (measured: ~270 on a 64^2 frame).  The
    implicit penalty bounds nothing here -- explicit calibration is what
    controls the false-alarm rate."""
    rng = np.random.default_rng(5)
    _, _, offpeak = _fluxes(_ml_fit(_frame(rng, 0.44), 0.44))
    assert offpeak > 100.0


def test_stray_count_becomes_an_atom_below_one_count_per_footprint():
    """At 2 pi sigma^2 b < 1 a single count activates its own atom: the
    fitted flux at the planted stray approaches the full count.  At
    2 pi sigma^2 b > 1 the same count is (mostly) absorbed as background.
    The boundary is the activation condition Phi(0)/b > <Phi, 1> of the
    alpha = 0 KKT system."""
    rng = np.random.default_rng(5)
    stray_sparse, _, _ = _fluxes(_ml_fit(_frame(rng, 0.02), 0.02))
    assert stray_sparse > 0.5
