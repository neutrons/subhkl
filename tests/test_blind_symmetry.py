"""Blind cell search and symmetry determination (no known space group)."""

import gemmi
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.search.blind_cell import (
    cell_of,
    conventional_basis,
    direct_space_map,
    find_peaks,
    lattice_basis,
    lattice_operations,
    operations_in_basis,
    reduce_basis,
    refine_basis,
    symmetrized_cell,
)
from subhkl.search.space_group import (
    NodeCounts,
    Observations,
    best_candidates,
    close_group,
    laue_class,
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


def test_body_centred_cubic_lattice_is_recognised():
    a = 11.93
    primitive = 0.5 * a * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]]).T
    turned = (
        Rotation.from_euler("xyz", [17, -31, 52], degrees=True).as_matrix() @ primitive
    )
    A = reduce_basis(turned @ np.array([[1, 1, 0], [0, 1, 2], [0, 0, 1]]))
    cell = cell_of(A)
    assert np.allclose(cell[:3], a * np.sqrt(3) / 2, atol=1e-6)
    mats, _name = lattice_operations(cell)
    assert len(mats) == 24
    C, T, mult = conventional_basis(A)
    assert mult == 2
    assert np.allclose(cell_of(C), (a, a, a, 90, 90, 90), atol=1e-6)
    # the rotations stay integer in the conventional indexing
    conv = operations_in_basis(mats, T)
    assert len(close_group(conv)) == 24


def test_symmetrized_cell_imposes_the_lattice_symmetry():
    rough = (60.47, 60.56, 97.15, 90.10, 90.04, 119.24)
    mats, name = lattice_operations(rough)
    assert name == "P 6 2 2"
    a, b, c, al, be, ga = symmetrized_cell(_basis(rough), mats)
    assert a == pytest.approx(b, abs=1e-6) and a == pytest.approx(60.5, abs=0.3)
    assert c == pytest.approx(97.15, abs=0.05)
    assert (al, be, ga) == pytest.approx((90.0, 90.0, 120.0), abs=1e-6)


def test_direct_space_map_gives_the_lattice():
    """A set of reciprocal-lattice points transforms into peaks on the
    direct lattice; the basis read off them reproduces the cell."""
    rng = np.random.default_rng(3)
    cell = (11.0, 13.0, 17.0, 84.0, 97.0, 105.0)
    A_true = Rotation.from_euler(
        "xyz", [10, 25, -40], degrees=True
    ).as_matrix() @ _basis(cell)
    recip = np.linalg.inv(A_true).T  # columns are reciprocal vectors
    n, q_max = 128, 0.5
    voxel = 1.0 / (2.0 * q_max)
    r = np.arange(-9, 10)
    hkl = np.array(
        [(h, k, l) for h in r for k in r for l in r if (h, k, l) != (0, 0, 0)]
    )
    q = hkl @ recip.T
    inside = np.all(np.abs(q) < q_max - 2.0 * q_max / n, axis=1)
    q, w = q[inside], rng.exponential(100.0, inside.sum())
    grid = np.zeros((n, n, n), dtype=np.float32)
    idx = np.floor((q + q_max) * (n / (2.0 * q_max))).astype(int)
    np.add.at(grid, tuple(idx.T), w)

    g = direct_space_map(grid, voxel, half_extent=60.0)
    pos, height = find_peaks(g, voxel, x_min=6.0)
    A, _good, diag = lattice_basis(pos, height, g, voxel)
    assert diag["indexed_weight"] > 0.8
    A, before, after = refine_basis(A, q, w, extent=60.0)
    assert after >= before
    found = np.array(cell_of(reduce_basis(A)))
    want = np.array(cell_of(reduce_basis(A_true)))
    assert np.allclose(found[:3], want[:3], atol=0.15)
    assert np.allclose(found[3:], want[3:], atol=1.0)


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
    assert {n.split(" (")[0] for n in names if not n.startswith("P")} == {"R 3 2"}
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
