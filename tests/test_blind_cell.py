"""Blind cell search: lattice from a direct-space map, conventional cells."""

import gemmi
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.search.blind_cell import (
    cell_of,
    centring_of,
    conventional_basis,
    describe_lattice,
    direct_space_map,
    find_peaks,
    lattice_basis,
    lattice_operations,
    operations_in_basis,
    reduce_basis,
    refine_basis,
    symmetrized_cell,
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
    assert len({tuple(m.ravel()) for m in conv}) == 24


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


_CENTRING = {
    "P": [],
    "I": [(0.5, 0.5, 0.5)],
    "C": [(0.5, 0.5, 0)],
    "F": [(0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)],
    "R": [(2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)],
}


def _primitive(cell, centring, seed):
    """A randomly oriented primitive basis of a centred lattice."""
    conv = _basis(cell)
    vecs = [conv[:, i] for i in range(3)] + [
        conv @ np.array(c) for c in _CENTRING[centring]
    ]
    pts = np.array(
        [sum(n * v for n, v in zip(ns, vecs)) for ns in np.ndindex(*(3,) * len(vecs))]
    )
    pts = np.unique(np.round(pts - pts.mean(0), 6), axis=0)
    origin = pts[np.argmin(np.linalg.norm(pts, axis=1))]
    prim = []
    for i in np.argsort(np.linalg.norm(pts - origin, axis=1))[1:]:
        trial = [*prim, pts[i] - origin]
        if np.linalg.matrix_rank(np.array(trial), tol=1e-6) == len(trial):
            prim = trial
        if len(prim) == 3:
            break
    return Rotation.random(random_state=seed).as_matrix() @ np.array(prim).T


@pytest.mark.parametrize(
    ("cell", "centring", "want_cell", "want_centring", "group", "p1"),
    [
        (
            (54.75, 80.55, 88.0, 90, 96.45, 90),
            "I",
            (54.75, 80.55, 88.0, 90, 96.45, 90),
            "I",
            "I 1 2 1",
            "I 1",
        ),
        (
            (40.0, 60.0, 50.0, 90, 101.0, 90),
            "P",
            (40.0, 60.0, 50.0, 90, 101.0, 90),
            "P",
            "P 1 2 1",
            "P 1",
        ),
        (
            (18.39, 56.55, 6.54, 90, 90, 90),
            "F",
            (6.54, 18.39, 56.55, 90, 90, 90),
            "F",
            "F 2 2 2",
            "F 1",
        ),
        (
            (40.0, 40.0, 70.0, 90, 90, 90),
            "I",
            (40.0, 40.0, 70.0, 90, 90, 90),
            "I",
            "I 4 2 2",
            "I 1",
        ),
        (
            (60.0, 60.0, 95.0, 90, 90, 120),
            "R",
            (60.0, 60.0, 95.0, 90, 90, 120),
            "R",
            "R 3 2:H",
            "P 1",
        ),
        (
            (25.0, 25.0, 25.0, 90, 90, 90),
            "F",
            (25.0, 25.0, 25.0, 90, 90, 90),
            "F",
            "F 4 3 2",
            "F 1",
        ),
    ],
)
def test_conventional_cell_of_centred_lattices(
    cell, centring, want_cell, want_centring, group, p1
):
    info = describe_lattice(_primitive(cell, centring, seed=7))
    assert info["centring"] == want_centring
    assert info["multiplicity"] == 1 + len(_CENTRING[centring])
    assert np.allclose(info["conventional_cell"], want_cell, atol=1e-4)
    assert info["lattice_group"] == group
    assert info["p1_symbol"] == p1
    assert centring_of(info["to_conventional"])[0] == want_centring
