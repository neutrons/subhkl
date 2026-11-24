"""Point group and space-group candidates without a known space group."""

import gemmi
import numpy as np
import pytest

from subhkl.search.blind_cell import lattice_operations
from subhkl.search.space_group import (
    NodeCounts,
    Observations,
    best_candidates,
    laue_class,
    laue_class_presence,
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


def test_laue_test_finds_a_point_group_below_the_lattice_symmetry():
    """A trigonal crystal in a hexagonal lattice: six of the twelve lattice
    rotations are symmetries, and the two-folds fix the 321 setting."""
    cell = (60.0, 60.0, 95.0, 90.0, 90.0, 120.0)
    mats, _name = lattice_operations(cell)
    assert len(mats) == 12
    obs = _observations("P 3 2 1", cell, np.random.default_rng(5))
    res = laue_class(obs, mats)
    assert len(res.rotations) == 6
    assert res.point_group == "32"
    assert res.symmorphic == "P 3 2 1"
    assert not res.conflicts
    assert res.friedel is not None and res.friedel < res.cut
    assert {tuple(m.ravel()) for m in res.rotations} == {
        tuple(m.ravel()) for m in _hkl_ops("P 3 2 1")
    }

    full = laue_class(_observations("P 6 2 2", cell, np.random.default_rng(6)), mats)
    assert len(full.rotations) == 12 and full.point_group == "622"


def test_laue_test_needs_repeated_reflections():
    cell = (60.0, 60.0, 95.0, 90.0, 90.0, 120.0)
    mats, _name = lattice_operations(cell)
    obs = _observations("P 3 2 1", cell, np.random.default_rng(5))
    obs.group[:] = 1  # one measurement only: nothing to compare
    with pytest.raises(ValueError, match="redundancy"):
        laue_class(obs, mats)


def _nodes(space_group, cell, hmax, rng):
    """Counts at every node: present where ``space_group`` allows, empty elsewhere."""
    hmax = np.array(hmax)
    grid = np.stack(
        np.meshgrid(*[np.arange(-m, m + 1) for m in hmax], indexing="ij"), -1
    ).reshape(-1, 3)
    absent = np.asarray(
        gemmi.SpaceGroup(space_group)
        .operations()
        .systematic_absences(grid.astype(np.int32)),
        dtype=bool,
    )
    shape = tuple(2 * hmax + 1)
    net = np.where(absent, 0.0, 200.0) + rng.normal(0.0, 3.0, len(grid))
    covered = np.ones(len(grid), dtype=bool)
    covered[np.all(grid == 0, axis=1)] = False
    return NodeCounts(
        hmax,
        net.reshape(shape),
        np.full(shape, 3.0),
        covered.reshape(shape),
        _basis(cell),
        d_min=1.0,
    )


def test_candidates_follow_the_point_group_and_its_setting():
    cell = (60.0, 60.0, 95.0, 90.0, 90.0, 120.0)
    nodes = _nodes("P 31 2 1", cell, (5, 5, 14), np.random.default_rng(8))
    results = space_group_candidates(
        nodes, rotations=_hkl_ops("P 3 2 1"), sohncke=True, rel_res=0.0
    )
    names = {n for r in results for n in r["space_groups"]}
    # only the 321 setting is offered, not 312; rhombohedral centring (both
    # ways round) is offered too and rejected by the counts
    assert {n for n in names if n.startswith("P")} == {
        "P 3 2 1",
        "P 31 2 1",
        "P 32 2 1",
    }
    assert {n.split(" (")[0] for n in names if not n.startswith("P")} == {"R 3 2:H"}
    assert all(
        not r["consistent"] for r in results if r["space_groups"][0].startswith("R")
    )
    best = best_candidates(results)
    assert len(best) == 1
    assert set(best[0]["space_groups"]) == {"P 31 2 1", "P 32 2 1"}
    plain = next(r for r in results if r["space_groups"] == ["P 3 2 1"])
    assert plain["consistent"] and plain["forbidden_nodes"] == 0

    # without the screw axis the screw-axis groups are rejected
    no_screw = _nodes("P 3 2 1", cell, (5, 5, 14), np.random.default_rng(9))
    best = best_candidates(
        space_group_candidates(
            no_screw, rotations=_hkl_ops("P 3 2 1"), sohncke=True, rel_res=0.0
        )
    )
    assert [r["space_groups"] for r in best] == [["P 3 2 1"]]


def test_candidates_from_the_name_of_the_laue_class():
    cell = (11.93, 11.93, 11.93, 90.0, 90.0, 90.0)
    nodes = _nodes("I a -3 d", cell, (8, 8, 8), np.random.default_rng(10))
    results = space_group_candidates(nodes, laue="m-3m", rel_res=0.0)
    best = best_candidates(results)
    assert [r["space_groups"] for r in best] == [["I a -3 d"]]
    rejected = {n for r in results if not r["consistent"] for n in r["space_groups"]}
    assert any(n.startswith("F ") for n in rejected)


def test_a_forbidden_node_next_to_an_allowed_one_is_not_evidence():
    """With finite resolution along a long axis, 00l absences that sit
    beside allowed reflections cannot be tested and do not count."""
    cell = (60.0, 60.0, 95.0, 90.0, 90.0, 120.0)
    nodes = _nodes("P 31 2 1", cell, (3, 3, 60), np.random.default_rng(11))
    results = space_group_candidates(
        nodes, rotations=_hkl_ops("P 3 2 1"), sohncke=True, rel_res=0.012
    )
    screw = next(r for r in results if "P 31 2 1" in r["space_groups"])
    assert screw["consistent"]
    assert 0 < screw["tested_nodes"] < screw["forbidden_nodes"]
    assert screw["untestable_nodes"] == 0 or screw["tested_nodes"] > 0


def test_monoclinic_centred_setting_is_found_among_the_candidates():
    """Space group 5 in its I 1 2 1 setting: the candidates include
    non-reference settings, and the counts pick the centring."""
    cell = (54.75, 80.55, 88.0, 90.0, 96.45, 90.0)
    nodes = _nodes("I 1 2 1", cell, (6, 8, 9), np.random.default_rng(12))
    results = space_group_candidates(
        nodes, rotations=_hkl_ops("P 1 2 1"), sohncke=True, rel_res=0.0
    )
    best = best_candidates(results)
    assert len(best) == 1
    assert "I 1 2 1" in best[0]["space_groups"]
    assert not any(n.startswith(("C", "A", "P")) for n in best[0]["space_groups"])
    rejected = {n for r in results if not r["consistent"] for n in r["space_groups"]}
    assert "C 1 2 1" in rejected and "A 1 2 1" in rejected


def _unscaled(point_group, cell, rng, n_meas=3):
    """Measurements of every reflection with an arbitrary scale each (up to
    a factor of three either way), as unnormalised Laue data have."""
    obs = _observations(point_group, cell, rng)
    half = len(obs.I) // 2
    hkl, truth, d = obs.hkl[:half], obs.I[:half], obs.d[:half]
    truth = np.abs(truth) + 1.0
    rows, inten, sig, dd, group = [], [], [], [], []
    for g in range(n_meas):
        k = np.exp(rng.uniform(-1.1, 1.1, len(hkl)))
        s = 0.05 * k * truth + 2.0
        rows.append(hkl)
        inten.append(k * truth + s * rng.normal(size=len(hkl)))
        sig.append(s)
        dd.append(d)
        group.append(np.full(len(hkl), g))
    return Observations(
        np.concatenate(rows),
        np.concatenate(inten),
        np.concatenate(sig),
        np.full(n_meas * len(hkl), 2.5),
        np.concatenate(dd),
        np.concatenate(group),
        match=None,
    )


def test_presence_test_needs_no_scaling():
    """With every measurement on its own scale, comparing intensities fails
    but asking whether the mate of a strong reflection is present works."""
    cell = (60.0, 60.0, 95.0, 90.0, 90.0, 120.0)
    mats, _name = lattice_operations(cell)
    obs = _unscaled("P 3 2 1", cell, np.random.default_rng(51))
    res = laue_class_presence(obs, mats)
    assert res.point_group == "32" and res.symmorphic == "P 3 2 1"
    assert len(res.rotations) == 6 and not res.conflicts
    assert res.same < 0.05
    assert res.unrelated > 0.2
    assert res.friedel is not None and res.friedel < res.cut
    with pytest.raises(ValueError, match="cannot decide"):
        laue_class(obs, mats)


def test_presence_test_refuses_unbounded_scales():
    """If repeats of one reflection differ a hundredfold, absence proves nothing."""
    cell = (60.0, 60.0, 95.0, 90.0, 90.0, 120.0)
    mats, _name = lattice_operations(cell)
    obs = _unscaled("P 3 2 1", cell, np.random.default_rng(52))
    scale = np.exp(np.random.default_rng(53).uniform(-4.0, 4.0, len(obs.I)))
    obs.I, obs.sig = obs.I * scale, obs.sig * scale
    with pytest.raises(ValueError, match="not bounded"):
        laue_class_presence(obs, mats)
