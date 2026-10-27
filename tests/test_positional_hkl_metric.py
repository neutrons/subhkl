"""The soft indexer's washboard, warped by the positional metric.

The sin^2 fractional-hkl loss is a positional criterion measured in the
wrong (hkl-isotropic) metric: the fractional defect maps to a detector
displacement through a per-peak Jacobian.  hkl_metric="positional"
scores each basin by that displacement -- radial component weighted by
radial_weight in the streak frame, plus an isotropic floor for the
wavelength-tube null direction -- while keeping the periodic, smooth,
assignment-free structure that gives DE its global convergence.

What is pinned here: exact truth recovery (modulo collinear harmonics,
which no positional criterion can separate -- both metrics report them
with a vanishing residual), the anisotropic cost ordering, and that the
default metric is bit-for-bit the previous loss.  Whether the warped
selection improves real assignments is a measurement, not a theorem:
on cg4d-t4-lysozyme the equivalent point-matcher scored 94.0% vs 91.6%.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.optimization import VectorizedObjective

BAND = (2.0, 3.8)  # ratio < 2: no exact second harmonics in band


def _make_peaks(n=60, seed=7, a_cell=6.0):
    rng = np.random.default_rng(seed)
    B = np.eye(3) / a_cell
    ki = np.array([0.0, 0.0, 1.0])
    hkls, lams, kfs = [], [], []
    while len(kfs) < n:
        hkl = rng.integers(-4, 5, size=3)
        if not np.any(hkl):
            continue
        G = B @ hkl
        lam = -2.0 * (ki @ G) / (G @ G)
        if not (BAND[0] <= lam <= BAND[1]):
            continue
        hkls.append(hkl)
        lams.append(lam)
        kfs.append(ki + lam * G)
    return B, ki, np.array(hkls), np.array(lams), np.array(kfs)


def _collinear_free(B, hkls, lams):
    """Peaks whose reflection has no in-band collinear alternative.

    Collinear multiples k*g predict the SAME detector position at a
    rescaled wavelength; they are indistinguishable to any positional
    criterion and are excluded from exact-assignment checks.
    """
    keep = []
    for H, lam in zip(hkls, lams):
        g = H // np.gcd.reduce(np.abs(H[H != 0]).astype(int))
        base = np.linalg.norm(B @ H)
        ok = True
        for k in range(1, 9):
            Hk = k * g
            if np.array_equal(Hk, H):
                continue
            lam_k = lam * base / np.linalg.norm(B @ Hk)
            # The optimizer clips lambda to the band edges, so an
            # alternative AT an edge is reachable: pad the check.
            if BAND[0] - 1e-6 <= lam_k <= BAND[1] + 1e-6:
                ok = False
                break
        keep.append(ok)
    return np.array(keep)


def _objective(B, ki, kfs, **kw):
    return VectorizedObjective(
        B,
        (kfs - ki).T,
        None,
        np.array(BAND),
        beam_nominal=ki,
        kf_lab_fixed_vectors=(kfs - ki).T,
        **kw,
    )


def _rotate_each(kfs, ki, direction, delta):
    out = np.empty_like(kfs)
    for i, kf in enumerate(kfs):
        t = np.cross(ki, kf)
        nt = np.linalg.norm(t)
        if nt < 1e-9:  # forward peak: frame undefined, leave in place
            out[i] = kf
            continue
        t /= nt
        axis = t if direction == "radial" else np.cross(t, kf)
        out[i] = Rotation.from_rotvec(delta * axis).as_matrix() @ kf
    return out


def test_positional_metric_recovers_the_truth():
    B, ki, hkls, lams, kfs = _make_peaks()
    unique = _collinear_free(B, hkls, lams)
    assert unique.sum() > 40  # the fixture is dominated by unambiguous peaks

    obj = _objective(B, ki, kfs, hkl_metric="positional", radial_weight=0.27)
    loss, dist, hkl, lam = obj.get_results(np.zeros((1, 3)))
    # dist keeps fractional-hkl units for the downstream indexed cut, and
    # vanishes for every peak -- collinear alternatives are exact too.
    np.testing.assert_allclose(np.array(dist[0]), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.array(loss[0]), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.array(hkl[0])[unique], hkls[unique], atol=1e-6)
    np.testing.assert_allclose(np.array(lam[0])[unique], lams[unique], rtol=1e-4)


def test_radial_errors_cost_less_than_tangential_under_the_metric():
    B, ki, hkls, lams, kfs = _make_peaks()
    delta = np.deg2rad(0.4)
    losses = {}
    for metric in ("isotropic", "positional"):
        for direction in ("radial", "tangential"):
            moved = _rotate_each(kfs, ki, direction, delta)
            obj = _objective(B, ki, moved, hkl_metric=metric, radial_weight=0.27)
            loss, _, _, _ = obj.get_results(np.zeros((1, 3)))
            losses[metric, direction] = float(np.array(loss[0]))

    # The free wavelength already makes the isotropic washboard radially
    # soft; the positional metric makes the preference explicit and
    # stronger, by a calibrated rather than accidental amount.
    iso_ratio = losses["isotropic", "radial"] / losses["isotropic", "tangential"]
    pos_ratio = losses["positional", "radial"] / losses["positional", "tangential"]
    assert losses["positional", "radial"] < losses["positional", "tangential"]
    assert pos_ratio < 0.8 * iso_ratio


def test_the_default_metric_is_the_previous_loss_bit_for_bit():
    B, ki, hkls, lams, kfs = _make_peaks()
    moved = _rotate_each(kfs, ki, "radial", np.deg2rad(0.7))
    ref = _objective(B, ki, moved)
    iso = _objective(B, ki, moved, hkl_metric="isotropic", hkl_metric_floor=0.5)
    x = np.zeros((1, 3))
    for a, b in zip(ref.get_results(x), iso.get_results(x)):
        np.testing.assert_array_equal(np.array(a), np.array(b))
