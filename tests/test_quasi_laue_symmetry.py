"""Experimental symmetry tests on a merge without point symmetry (quasi-Laue)."""

import gemmi
import numpy as np
import pytest

from subhkl.search.blind_cell import lattice_operations
from subhkl.search.space_group import (
    Observations,
    best_candidates,
    laue_class,
    laue_class_merged,
    merged_node_counts,
    space_group_candidates,
)


def _hkl_ops(name):
    """Rotations of a space group as integer matrices on Miller indices."""
    out = []
    for op in gemmi.SpaceGroup(name).operations().sym_ops:
        w = np.array(op.rot, dtype=float) / gemmi.Op.DEN
        out.append(np.rint(np.linalg.inv(w).T).astype(int))
    return out


def _basis(cell):
    return np.array(gemmi.UnitCell(*cell).orth.mat.tolist())


def _observations(point_group, cell, rng, hmax=6):
    """Two noisy measurements of every reflection, with intensities that
    have exactly the symmetry of ``point_group`` plus Friedel's law."""
    ops = [s * m for m in _hkl_ops(point_group) for s in (1, -1)]
    r = np.arange(-hmax, hmax + 1)
    hkl = np.array(
        [(h, k, l) for h in r for k in r for l in r if (h, k, l) != (0, 0, 0)]
    )
    orbit = np.stack([hkl @ m.T for m in ops], 1)  # (N, n_ops, 3)
    code = (
        (orbit[..., 0] + 50) * 10000 + (orbit[..., 1] + 50) * 100 + orbit[..., 2] + 50
    )
    rep = code.min(axis=1)
    _, inv = np.unique(rep, return_inverse=True)
    truth = rng.exponential(1000.0, inv.max() + 1)[inv.ravel()]
    d = 1.0 / np.linalg.norm(hkl @ np.linalg.inv(_basis(cell)), axis=1)
    sigma = 0.03 * truth + 5.0
    both = [truth + sigma * rng.normal(size=len(truth)) for _ in range(2)]
    return Observations(
        np.concatenate([hkl, hkl]),
        np.concatenate(both),
        np.concatenate([sigma, sigma]),
        np.full(2 * len(hkl), 2.5),
        np.concatenate([d, d]),
        np.repeat([1, 2], len(hkl)),
    )


def _merged(point_group, cell, rng, hmax=7, noise=0.3):
    """Friedel-unique merged amplitudes with the symmetry of ``point_group``."""
    ops = [s * m for m in _hkl_ops(point_group) for s in (1, -1)]
    r = np.arange(-hmax, hmax + 1)
    hkl = np.array(
        [
            (h, k, l)
            for h in r
            for k in r
            for l in r
            if (l > 0) or (l == 0 and (k > 0 or (k == 0 and h > 0)))
        ]
    )
    orbit = np.stack([hkl @ m.T for m in ops], 1)
    code = (
        (orbit[..., 0] + 50) * 10000 + (orbit[..., 1] + 50) * 100 + orbit[..., 2] + 50
    )
    _, inv = np.unique(code.min(axis=1), return_inverse=True)
    truth = rng.exponential(10.0, inv.max() + 1)[inv.ravel()]
    d = 1.0 / np.linalg.norm(hkl @ np.linalg.inv(_basis(cell)), axis=1)
    return hkl, truth * (1 + noise * rng.normal(size=len(truth))), d


def test_merged_laue_test_for_data_without_matched_wavelengths():
    """After a merge in P 1 the mates of a symmetry correlate, the others do not."""
    cell = (62.0, 62.0, 97.0, 90.0, 90.0, 120.0)
    mats, _name = lattice_operations(cell)
    hkl, amp, d = _merged("P 3 2 1", cell, np.random.default_rng(21))
    res = laue_class_merged(hkl, amp, d, mats, d_min=3.0)
    assert res.point_group == "32" and res.symmorphic == "P 3 2 1"
    assert len(res.rotations) == 6 and not res.conflicts
    assert abs(res.unrelated) < 0.1
    with_ref = laue_class_merged(hkl, amp, d, mats, cc_half=0.8, d_min=3.0)
    assert with_ref.point_group == "32"

    # no symmetry at all: nothing stands above unrelated reflections
    hkl, amp, d = _merged("P 1", cell, np.random.default_rng(22))
    none = laue_class_merged(hkl, amp, d, mats, d_min=3.0)
    assert len(none.rotations) == 1 and none.point_group == "1"


def test_merged_intensities_as_node_counts():
    cell = (62.0, 62.0, 97.0, 90.0, 90.0, 120.0)
    r = np.arange(-5, 6)
    hkl = np.array(
        [(h, k, l) for h in r for k in r for l in range(13) if (h, k, l) != (0, 0, 0)]
    )
    absent = np.asarray(
        gemmi.SpaceGroup("P 31 2 1")
        .operations()
        .systematic_absences(hkl.astype(np.int32)),
        dtype=bool,
    )
    rng = np.random.default_rng(23)
    inten = np.where(absent, 0.0, 150.0) + rng.normal(0, 3, len(hkl))
    nodes = merged_node_counts(
        hkl, inten, np.full(len(hkl), 3.0), _basis(cell), d_min=3.0
    )
    assert nodes.covered[
        tuple(nodes.hmax + np.array([1, 0, -4]))
    ]  # Friedel mate filled in
    best = best_candidates(
        space_group_candidates(
            nodes, rotations=_hkl_ops("P 3 2 1"), sohncke=True, rel_res=0.0
        )
    )
    assert set(best[0]["space_groups"]) == {"P 31 2 1", "P 32 2 1"}


def test_scaled_observations_are_compared_at_any_wavelength():
    """Observations with different scales (wavelengths) agree once divided
    by the scale; the indices are only known up to inversion."""
    rng = np.random.default_rng(31)
    cell = (62.0, 62.0, 97.0, 90.0, 90.0, 120.0)
    mats, _name = lattice_operations(cell)
    hkl, truth, d = _merged("P 3 2 1", cell, rng, hmax=6, noise=0.0)
    truth = truth**2
    rows, inten, sig, scale, dd, image = [], [], [], [], [], []
    for rep in range(3):
        k = rng.uniform(0.2, 5.0, len(hkl))  # e.g. the spectrum at each wavelength
        s = 0.03 * k * truth + 2.0
        flip = rng.random(len(hkl)) < 0.5  # either Friedel mate
        rows.append(np.where(flip[:, None], -hkl, hkl))
        inten.append(k * truth + s * rng.normal(size=len(hkl)))
        sig.append(s)
        scale.append(k)
        dd.append(d)
        image.append(np.full(len(hkl), rep))
    obs = Observations.scaled(
        np.concatenate(rows),
        np.concatenate(inten),
        np.concatenate(sig),
        np.concatenate(scale),
        np.concatenate(dd),
        np.concatenate(image),
        d_min=3.0,
    )
    res = laue_class(obs, mats)
    assert res.point_group == "32" and len(res.rotations) == 6 and not res.conflicts
    assert res.same < 2.0 < res.unrelated


def test_symmetry_tests_refuse_data_that_cannot_decide():
    cell = (62.0, 62.0, 97.0, 90.0, 90.0, 120.0)
    mats, _name = lattice_operations(cell)
    # a merge without signal: the half-dataset correlation says so
    hkl, amp, d = _merged("P 3 2 1", cell, np.random.default_rng(41))
    with pytest.raises(ValueError, match="too little signal"):
        laue_class_merged(hkl, amp, d, mats, cc_half=0.01, d_min=3.0)
    # repeats that disagree almost as much as unrelated reflections
    obs = _observations("P 3 2 1", cell, np.random.default_rng(42))
    obs.I = obs.I * np.random.default_rng(43).lognormal(0.0, 1.5, len(obs.I))
    with pytest.raises(ValueError, match="cannot decide"):
        laue_class(obs, mats)
