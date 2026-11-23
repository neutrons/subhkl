"""Prototype: centring, glide planes and screw axes from raw events.

Every event is converted to fractional Miller indices of the conventional
cell.  Counts at the integer nodes, above the level between nodes, are then
summed by reflection class:

- all nodes, by a general condition (lattice centring);
- nodes in a zone (a central plane of reciprocal space), by a zonal
  condition (glide plane: the structure projected onto the plane has a
  shorter period);
- nodes on an axis, by an axial condition (screw axis).

Nothing is integrated and no intensity is normalised: a forbidden class
simply carries no counts above background.

Usage:
    python zone_absences.py reduced_or_merged.h5 MANDI cell.npz
"""

import argparse

import gemmi
import h5py
import numpy as np
from blind_cell import pixel_positions
from blind_index import cell_of, reduce_basis

from subhkl.config import reduction_settings
from subhkl.instrument.goniometer import Goniometer
from subhkl.io.events import ANGSTROM_M_PER_US, EventStore


def conventional_basis(A, max_obliquity=3.0):
    """Conventional basis (columns [A]) of a reduced primitive basis: its
    three shortest independent lattice vectors along symmetry axes of
    highest order.  Returns it and the centring multiplicity."""
    cell = cell_of(A)
    ops = gemmi.find_lattice_symmetry(gemmi.UnitCell(*cell), "P", max_obliquity)
    axes = []
    for op in ops.sym_ops:
        w = np.array(op.rot, dtype=float) / gemmi.Op.DEN
        order = next(
            (
                n
                for n in (1, 2, 3, 4, 6)
                if np.allclose(np.linalg.matrix_power(w, n), np.eye(3), atol=1e-6)
            ),
            None,
        )
        if order in (None, 1):
            continue
        vals, vecs = np.linalg.eig(w)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
        v = v / np.abs(v[np.abs(v) > 1e-6]).min()
        uvw = np.rint(v).astype(int)
        if tuple(uvw) < tuple(-uvw):
            uvw = -uvw
        axes.append((order, tuple(uvw)))
    best = max(o for o, _ in axes)
    cand = sorted(
        {a for o, a in axes if o == best}, key=lambda u: np.linalg.norm(A @ np.array(u))
    )
    chosen = []
    for u in cand:
        trial = [*chosen, u]
        m = np.array(trial, dtype=float).T
        if np.linalg.matrix_rank(m) == len(trial):
            chosen = trial
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        return A, 1, len(ops.sym_ops)
    M = np.array(chosen, dtype=float).T
    if np.linalg.det(M) < 0:
        M[:, 2] *= -1
    return A @ M, round(abs(np.linalg.det(M))), len(ops.sym_ops)


def event_indices(filename, instrument, C):
    """Fractional conventional Miller indices of every event, (N, 3)."""
    l1 = float(reduction_settings[instrument]["L1"])
    with h5py.File(filename, "r") as f:
        images = f["images"][()]
        bank_ids = f["bank_ids"][()]
        rot = Goniometer.from_h5_file(f).rotation
    if rot.ndim == 2:
        rot = np.broadcast_to(rot, (len(images), 3, 3))
    out = []
    with EventStore(filename) as store:
        for i in range(len(images)):
            tof, off = store.image_events(i, images[i])
            if len(tof) == 0:
                continue
            pos = pixel_positions(instrument, bank_ids[i], images[i].shape)
            xyz = pos[np.repeat(np.arange(images[i].size), np.diff(off))].astype(
                np.float64
            )
            l2 = np.linalg.norm(xyz, axis=1)
            lam = ANGSTROM_M_PER_US * tof / (l1 + l2)
            q_lab = (xyz / l2[:, None] - np.array([0.0, 0.0, 1.0])) / lam[:, None]
            out.append((q_lab @ rot[i]) @ C)  # h = c_j . q_sample
    return np.concatenate(out)


def node_counts(hkl, select, r_in=0.18, r_out=(0.32, 0.5)):
    """Net counts at integer nodes for the events picked by ``select``.

    Signal: events within r_in of a node.  Background: events between
    r_out[0] and r_out[1] of the same node, scaled by volume.  Returns
    (nodes [n, 3] int, net, sigma) for nodes that the data cover at all.
    """
    h = hkl[select]
    node = np.rint(h).astype(np.int64)
    dist = np.linalg.norm(h - node, axis=1)
    key, inv = np.unique(node, axis=0, return_inverse=True)
    inv = inv.ravel()
    sig = np.bincount(inv, weights=(dist < r_in).astype(float), minlength=len(key))
    bkg = np.bincount(
        inv,
        weights=((dist > r_out[0]) & (dist < r_out[1])).astype(float),
        minlength=len(key),
    )
    ratio = r_in**3 / (r_out[1] ** 3 - r_out[0] ** 3)
    covered = (sig + bkg) >= 5
    net = sig - ratio * bkg
    err = np.sqrt(sig + ratio**2 * bkg + 1.0)
    return key[covered], net[covered], err[covered]


def report(title, nodes, net, err, classes):
    print(f"\n{title}")
    print(
        "   class                          nodes   net counts    sigma   net/sigma   per node"
    )
    for name, mask in classes(nodes):
        if mask.sum() == 0:
            continue
        tot, s = net[mask].sum(), np.sqrt((err[mask] ** 2).sum())
        print(
            f"   {name:28s} {mask.sum():7d}   {tot:10.0f}  {s:7.0f}   {tot / s:8.1f}   {tot / mask.sum():8.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("instrument")
    ap.add_argument("basis")
    ap.add_argument(
        "--slab",
        type=float,
        default=0.12,
        help="half-thickness of a zone or axis in fractional indices",
    )
    args = ap.parse_args()

    A = reduce_basis(np.load(args.basis)["A"])
    C, mult, n_rot = conventional_basis(A)
    cell = cell_of(C)
    print(
        "primitive cell:    "
        + "a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *cell_of(A)
        )
    )
    print(
        "conventional cell: "
        + "a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *cell
        )
    )
    print(
        f"lattice rotations: {n_rot}; the conventional cell holds {mult} lattice point(s)"
    )

    hkl = event_indices(args.events, args.instrument, C)
    print(f"events: {len(hkl)}")
    everything = np.ones(len(hkl), dtype=bool)

    nodes, net, err = node_counts(hkl, everything)
    nz = np.any(nodes != 0, axis=1)
    nodes, net, err = nodes[nz], net[nz], err[nz]
    s = nodes.sum(axis=1)
    report(
        "all reflections, by lattice centring",
        nodes,
        net,
        err,
        lambda n: [
            ("h+k+l even (I allows)", s % 2 == 0),
            ("h+k+l odd  (I forbids)", s % 2 == 1),
            ("all even or all odd (F allows)", np.all(n % 2 == n[:, :1] % 2, axis=1)),
            ("mixed parity (F forbids)", ~np.all(n % 2 == n[:, :1] % 2, axis=1)),
        ],
    )

    # zones perpendicular to each cell axis: 0kl, h0l, hk0
    acc = []
    for ax in range(3):
        sel = np.abs(hkl[:, ax]) < args.slab
        n_, net_, err_ = node_counts(hkl, sel)
        keep = (n_[:, ax] == 0) & np.any(n_ != 0, axis=1)
        others = [i for i in range(3) if i != ax]
        acc.append((n_[keep][:, others], net_[keep], err_[keep]))
    n2 = np.concatenate([a[0] for a in acc])
    net2 = np.concatenate([a[1] for a in acc])
    err2 = np.concatenate([a[2] for a in acc])
    ev = n2 % 2 == 0
    report(
        "zones 0kl, h0l, hk0 together (indices are the two non-zero ones)",
        n2,
        net2,
        err2,
        lambda n: [
            ("both even", ev[:, 0] & ev[:, 1]),
            ("both odd", ~ev[:, 0] & ~ev[:, 1]),
            ("one even, one odd", ev[:, 0] ^ ev[:, 1]),
        ],
    )

    # diagonal zones hhl and permutations: events with two indices equal
    acc = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        k_ = 3 - i - j
        sel = np.abs(hkl[:, i] - hkl[:, j]) < args.slab * np.sqrt(2)
        n_, net_, err_ = node_counts(hkl, sel)
        keep = (n_[:, i] == n_[:, j]) & np.any(n_ != 0, axis=1) & (n_[:, i] != 0)
        acc.append(
            (np.stack([n_[keep][:, i], n_[keep][:, k_]], 1), net_[keep], err_[keep])
        )
    n3 = np.concatenate([a[0] for a in acc])
    net3 = np.concatenate([a[1] for a in acc])
    err3 = np.concatenate([a[2] for a in acc])
    v = (2 * n3[:, 0] + n3[:, 1]) % 4
    report(
        "zones hhl, hkk, hkh together, by 2h + l mod 4",
        n3,
        net3,
        err3,
        lambda n: [(f"2h + l = 4n + {r}", v == r) for r in range(4)],
    )

    # axes h00, 0k0, 00l
    acc = []
    for ax in range(3):
        others = [i for i in range(3) if i != ax]
        sel = np.all(np.abs(hkl[:, others]) < args.slab, axis=1)
        n_, net_, err_ = node_counts(hkl, sel)
        keep = np.all(n_[:, others] == 0, axis=1) & (n_[:, ax] != 0)
        acc.append((np.abs(n_[keep][:, ax]), net_[keep], err_[keep]))
    n1 = np.concatenate([a[0] for a in acc])
    net1 = np.concatenate([a[1] for a in acc])
    err1 = np.concatenate([a[2] for a in acc])
    report(
        "axes h00, 0k0, 00l together, by h mod 4",
        n1,
        net1,
        err1,
        lambda n: [(f"h = 4n + {r}", n % 4 == r) for r in range(4)],
    )


if __name__ == "__main__":
    main()
