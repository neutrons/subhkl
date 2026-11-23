"""Prototype: from a blind lattice basis to an indexing solution in P1.

Takes the basis found by ``blind_cell.py`` (direct-lattice vectors in the
sample frame), reduces it, finds the symmetry of the lattice itself (the
holohedry, which bounds the Laue class), and writes a solve-style file with
the cell, the orientation and space group P 1 that ``peak-predictor`` and
``integrator`` accept.  Goniometer and detector geometry are nominal.

Usage:
    python blind_index.py blind_cell.npz merged.h5 MANDI out_solve.h5 [--reference solve.h5]
"""

import argparse
import json

import gemmi
import h5py
import numpy as np

from subhkl.config import reduction_settings
from subhkl.io.events import ANGSTROM_M_PER_US


def reduce_basis(A):
    """Shortest-vector (Minkowski) reduction of a 3D basis, right-handed,
    with the Niggli sign convention (angles all obtuse or all acute)."""
    A = A.copy()
    changed = True
    while changed:
        changed = False
        order = np.argsort(np.linalg.norm(A, axis=0))
        A = A[:, order]
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                m = np.rint(A[:, i] @ A[:, j] / (A[:, j] @ A[:, j]))
                if m != 0:
                    cand = A[:, i] - m * A[:, j]
                    if np.linalg.norm(cand) < np.linalg.norm(A[:, i]) - 1e-9:
                        A[:, i] = cand
                        changed = True
    a, b, c = A[:, 0], A[:, 1], A[:, 2]
    # all-obtuse where possible: flip b and c so that the angles to a are >= 90
    if a @ b > 0:
        b = -b
    if a @ c > 0:
        c = -c
    A = np.stack([a, b, c], 1)
    if np.linalg.det(A) < 0:
        A = -A
    return A


def cell_of(A):
    lengths = np.linalg.norm(A, axis=0)

    def ang(u, v):
        return np.degrees(
            np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
        )

    return (
        *lengths,
        ang(A[:, 1], A[:, 2]),
        ang(A[:, 0], A[:, 2]),
        ang(A[:, 0], A[:, 1]),
    )


def lattice_operations(cell, max_obliquity=3.0):
    """Rotations of the lattice itself, as integer matrices acting on Miller
    indices (h' = M h), and the corresponding rotation group's name."""
    ops = gemmi.find_lattice_symmetry(gemmi.UnitCell(*cell), "P", max_obliquity)
    mats = []
    for op in ops.sym_ops:
        w = (
            np.array(op.rot, dtype=float) / gemmi.Op.DEN
        )  # acts on fractional coordinates
        mats.append(np.rint(np.linalg.inv(w).T).astype(int))
    return mats, gemmi.find_spacegroup_by_ops(ops)


def wavelength_band(merged, instrument, n_sample=4_000_000):
    """[A] wavelength range the events cover, to two significant steps."""
    l1 = float(reduction_settings[instrument]["L1"])
    with h5py.File(merged, "r") as f:
        tof = f["events/tof"]
        step = max(1, tof.shape[0] // n_sample)
        t = tof[::step][:n_sample]
    lam = np.percentile(t, [0.5, 99.5]) * ANGSTROM_M_PER_US / (l1 + 0.42)
    return float(np.round(lam[0], 1)), float(np.round(lam[1], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("basis")
    ap.add_argument("merged")
    ap.add_argument("instrument")
    ap.add_argument("out")
    ap.add_argument("--reference", default=None, help="a solve file to compare against")
    args = ap.parse_args()

    A = reduce_basis(np.load(args.basis)["A"])
    cell = cell_of(A)
    print(
        "reduced primitive cell: "
        + "a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *cell
        )
    )
    mats, lattice_group = lattice_operations(cell)
    print(
        f"lattice symmetry within 3 deg obliquity: {len(mats)} rotations, group {lattice_group.hm}"
    )

    # q_sample = UB h with UB = A^-T; split as U (orthogonal) times B (upper triangular)
    UB = np.linalg.inv(A).T
    U, B = np.linalg.qr(UB)
    sign = np.sign(np.diag(B))
    U, B = U * sign, (B.T * sign).T
    assert np.linalg.det(U) > 0
    band = wavelength_band(args.merged, args.instrument)
    print(f"wavelength band from the events: {band[0]}-{band[1]} A")

    with h5py.File(args.merged, "r") as fm, h5py.File(args.out, "w") as f:
        f.attrs["instrument"] = args.instrument
        f.attrs["source"] = args.merged
        for name, value in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            f[f"sample/{name}"] = float(value)
        f["sample/space_group"] = np.bytes_(b"P 1")
        f["sample/U"] = U
        f["sample/B"] = B
        f["instrument/wavelength"] = np.array(band)
        for key in (
            "goniometer/axes",
            "goniometer/angles",
            "goniometer/names",
            "bank_ids",
            "files",
            "file_offsets",
        ):
            if key in fm:
                fm.copy(fm[key], f, key)
        g = f.create_group("blind")
        g["direct_basis"] = A
        g["lattice_operations"] = np.array(mats)
        g.attrs["lattice_group"] = lattice_group.hm
    print("wrote", args.out)

    if args.reference:
        with h5py.File(args.reference, "r") as f:
            ref_UB = f["sample/U"][()] @ f["sample/B"][()]
            ref_cell = [
                float(f[f"sample/{k}"][()])
                for k in ("a", "b", "c", "alpha", "beta", "gamma")
            ]
        print(
            "reference cell:         "
            + "a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
                *ref_cell
            )
        )
        # the two indexings differ by an integer change of basis: h_ref = T h
        T = np.linalg.solve(ref_UB, UB)
        print("change of basis to the reference indexing (should be integers):")
        print(np.round(T, 3))
        Ti = np.rint(T)
        # misorientation after removing the best integer relation and any scale
        R = UB @ np.linalg.inv(ref_UB @ Ti)
        scale = np.cbrt(abs(np.linalg.det(R)))
        u, _s, vt = np.linalg.svd(R / scale)
        rot = u @ vt
        angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
        print(
            f"|det T| = {abs(np.linalg.det(Ti)):.0f}; reciprocal scale ratio (blind / reference) {scale:.4f}; misorientation {angle:.2f} deg"
        )
        print(
            json.dumps(
                {
                    "cell": [round(float(v), 3) for v in cell],
                    "lattice_group": lattice_group.hm,
                }
            )
        )


if __name__ == "__main__":
    main()
