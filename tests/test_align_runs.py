"""Relating the lattices of single runs (blind goniometer correction)."""

import gemmi
import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.search.blind_cell import relate_lattices


def _basis(cell):
    return np.array(gemmi.UnitCell(*cell).orth.mat.tolist())


def test_relate_lattices_recovers_a_small_rotation_between_equivalent_bases():
    """The same lattice, turned by 2.5 degrees and written in another
    basis: the rotation comes back; a different lattice is refused."""
    A = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix() @ _basis(
        (60.9, 60.9, 97.1, 90.0, 90.0, 120.0)
    )
    turn = Rotation.from_rotvec(
        np.radians(2.5) * np.array([0.3, -0.5, 0.81])
    ).as_matrix()
    rebase = np.array(
        [[0, -1, 0], [1, -1, 0], [0, 0, 1]]
    )  # an equivalent hexagonal basis
    M, ok = relate_lattices(turn @ A @ rebase, A)
    assert ok
    assert np.allclose(M, turn, atol=1e-6)
    other = _basis((60.9, 60.9, 110.0, 90.0, 90.0, 120.0))
    assert not relate_lattices(turn @ other, A)[1]
