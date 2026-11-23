"""Prototype: space-group candidates from the Laue class and raw events.

Given the conventional cell, the orientation and the Laue class, every
space group of that Laue class is a hypothesis about which reflections are
systematically absent (centring, glide planes, screw axes).  The hypotheses
are tested on raw events: each event is converted to fractional Miller
indices, and the counts at integer nodes above the level between nodes are
compared.  Nothing is integrated or normalised.

How a hypothesis G is judged
- The nodes the candidates forbid are split into elementary classes: sets
  of nodes that the same candidates forbid.  A class is one reflection
  condition (or an intersection of a few).
- A reflection's scattering vector is known to a relative precision
  ``rel_res``, so a node of index n is blurred by about rel_res * n.  A
  forbidden node is only "testable" under G if every node inside that blur
  is forbidden under G too; otherwise an allowed neighbour leaks into it.
- For each class G forbids, the mean net counts of its testable nodes are
  compared with allowed nodes of the same resolution.  G is consistent if
  no class carries more than ``threshold`` of that.
- Among consistent hypotheses the one forbidding the most is reported,
  with every other candidate that forbids exactly the same nodes.

Usage:
    python absences.py events.h5 MANDI --laue m-3m --basis cell.npz
    python absences.py events.h5 MANDI --laue 6/mmm --solve refined.h5 --sohncke
"""

import argparse
import itertools
import json

import gemmi
import h5py
import numpy as np
from blind_cell import pixel_positions
from blind_index import cell_of, reduce_basis
from scipy.ndimage import maximum_filter
from zone_absences import conventional_basis

from subhkl.config import reduction_settings
from subhkl.instrument.goniometer import Goniometer
from subhkl.io.events import ANGSTROM_M_PER_US, EventStore

R_IN = 0.18  # a node's signal sphere, fractional indices
R_OUT = (0.32, 0.5)  # its background shell


def node_counts(filename, instrument, to_hkl_of_frame, C, d_min):
    """Net counts at every integer node with d >= d_min.

    Returns (hmax [3], net, err, covered) as dense arrays over the node box
    (2 hmax + 1 per axis), indexed by h + hmax.
    """
    l1 = float(reduction_settings[instrument]["L1"])
    hmax = np.ceil(np.linalg.norm(C, axis=0) / d_min).astype(int) + 1
    shape = tuple(2 * hmax + 1)
    size = int(np.prod(shape))
    sig = np.zeros(size)
    bkg = np.zeros(size)
    with h5py.File(filename, "r") as f:
        images = f["images"][()]
        bank_ids = f["bank_ids"][()]
    pos_cache = {}
    n_events = 0
    with EventStore(filename) as store:
        for i in range(len(images)):
            tof, off = store.image_events(i, images[i])
            if len(tof) == 0:
                continue
            bank = int(bank_ids[i])
            if bank not in pos_cache:
                pos_cache[bank] = pixel_positions(
                    instrument, bank, images[i].shape
                ).astype(np.float64)
            xyz = pos_cache[bank][np.repeat(np.arange(images[i].size), np.diff(off))]
            l2 = np.linalg.norm(xyz, axis=1)
            lam = ANGSTROM_M_PER_US * tof / (l1 + l2)
            q_lab = (xyz / l2[:, None] - np.array([0.0, 0.0, 1.0])) / lam[:, None]
            q_lab = q_lab[np.linalg.norm(q_lab, axis=1) <= 1.0 / d_min]
            h = q_lab @ to_hkl_of_frame(i).T
            node = np.rint(h).astype(np.int64)
            dist = np.linalg.norm(h - node, axis=1)
            ok = np.all(np.abs(node) <= hmax, axis=1)
            flat = np.ravel_multi_index((node[ok] + hmax).T, shape)
            sig += np.bincount(flat[dist[ok] < R_IN], minlength=size)
            in_shell = (dist[ok] > R_OUT[0]) & (dist[ok] < R_OUT[1])
            bkg += np.bincount(flat[in_shell], minlength=size)
            n_events += len(h)
    ratio = R_IN**3 / (R_OUT[1] ** 3 - R_OUT[0] ** 3)
    net = (sig - ratio * bkg).reshape(shape)
    err = np.sqrt(sig + ratio**2 * bkg + 1.0).reshape(shape)
    covered = ((sig + bkg) >= 5).reshape(shape)
    covered[tuple(hmax)] = False  # the origin
    return hmax, net, err, covered, n_events


def candidates(laue, sohncke):
    out = []
    for sg in gemmi.spacegroup_table():
        if sg.laue_str() != laue or not sg.is_reference_setting():
            continue
        if sohncke and not sg.is_sohncke():
            continue
        out.append(sg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("instrument")
    ap.add_argument("--laue", required=True, help="Laue class, e.g. 6/mmm or m-3m")
    ap.add_argument("--basis", help="blind_cell.py output (nominal goniometer)")
    ap.add_argument(
        "--solve",
        help="refined solve file (its cell, orientation and per-frame rotations)",
    )
    ap.add_argument("--d-min", type=float, default=None)
    ap.add_argument(
        "--rel-res",
        type=float,
        default=0.012,
        help="relative precision of a scattering vector",
    )
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument(
        "--sohncke", action="store_true", help="only groups a chiral molecule can take"
    )
    ap.add_argument(
        "--out", help="write every hypothesis with its per-class evidence as JSON"
    )
    args = ap.parse_args()

    if args.solve:
        with h5py.File(args.solve, "r") as f:
            ub = f["sample/U"][()] @ f["sample/B"][()]
            rot = f["goniometer/R"][()]
        C = np.linalg.inv(ub).T  # direct basis, columns [A], sample frame

        def to_hkl(i):
            return np.linalg.inv(rot[i] @ ub)  # h = (R UB)^-1 q_lab

    else:
        C, _mult, _n = conventional_basis(reduce_basis(np.load(args.basis)["A"]))
        with h5py.File(args.events, "r") as f:
            rot = Goniometer.from_h5_file(f).rotation
            n_img = f["images"].shape[0]
        if rot.ndim == 2:
            rot = np.broadcast_to(rot, (n_img, 3, 3))

        def to_hkl(i):
            return C.T @ rot[i].T  # h_j = c_j . (R^T q_lab)

    cell = cell_of(C)
    d_min = args.d_min or 0.7
    print(
        "conventional cell: "
        + "a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *cell
        )
    )

    hmax, net, err, covered, n_events = node_counts(
        args.events, args.instrument, to_hkl, C, d_min
    )
    shape = net.shape
    grid = np.stack(
        np.meshgrid(*[np.arange(-m, m + 1) for m in hmax], indexing="ij"), -1
    ).reshape(-1, 3)
    d_all = 1.0 / np.maximum(np.linalg.norm(grid @ np.linalg.inv(C), axis=1), 1e-9)
    cov = covered.ravel() & (d_all >= d_min)
    netf, errf = net.ravel(), err.ravel()
    print(
        f"events used: {n_events}; nodes with data to {d_min} A: {cov.sum()}; net counts on them: {netf[cov].sum():.0f}"
    )

    # blur of each node in index units, and the distinct neighbourhood sizes
    blur = np.rint(2.5 * args.rel_res * np.abs(grid)).astype(int)
    blur_keys, blur_inv = np.unique(blur, axis=0, return_inverse=True)
    blur_inv = blur_inv.ravel()

    # hypotheses: each candidate group in each axis assignment, by what it forbids
    cands = candidates(args.laue, args.sohncke)
    hyp = {}
    # Axis assignments worth trying: those that leave the cell unchanged (all
    # six for a cubic cell, a <-> b for a hexagonal or tetragonal one).
    base = np.array(cell)
    perms = []
    for perm in itertools.permutations(range(3)):
        permuted = np.array(cell_of(C[:, list(perm)]))
        same_len = np.allclose(permuted[:3], base[:3], rtol=0.01)
        same_ang = np.allclose(
            np.cos(np.radians(permuted[3:])), np.cos(np.radians(base[3:])), atol=0.02
        )
        if same_len and same_ang:
            perms.append(perm)
    for sg in cands:
        ops = sg.operations()
        for perm in perms:
            forb = np.asarray(
                ops.systematic_absences(grid[:, list(perm)].astype(np.int32)),
                dtype=bool,
            )
            key = forb[cov].tobytes()
            label = (
                sg.hm
                if perm == (0, 1, 2)
                else f"{sg.hm} (axes {''.join('abc'[p] for p in perm)})"
            )
            if key not in hyp:
                hyp[key] = {"forb": forb, "names": [label]}
            elif label.split(" (")[0] not in [
                n.split(" (")[0] for n in hyp[key]["names"]
            ]:
                hyp[key]["names"].append(label)
    hyps = list(hyp.values())
    print(
        f"space groups of Laue class {args.laue}{' (Sohncke only)' if args.sohncke else ''}: {len(cands)}; distinct absence patterns on these nodes: {len(hyps)}"
    )

    # elementary classes: nodes forbidden by the same set of hypotheses
    F = np.stack([h["forb"][cov] for h in hyps], 1)
    sig_keys, sig_inv = np.unique(F, axis=0, return_inverse=True)
    sig_inv = sig_inv.ravel()
    cov_idx = np.flatnonzero(cov)

    # reference: mean net counts of never-forbidden nodes, by resolution
    never = ~F.any(axis=1)
    edges = np.quantile(d_all[cov_idx[never]], np.linspace(0, 1, 9))
    edges[0], edges[-1] = 0.0, np.inf
    which = np.clip(np.searchsorted(edges, d_all[cov_idx], side="right") - 1, 0, 7)
    ref_bin = np.array(
        [
            netf[cov_idx[never & (which == b)]].mean()
            if (never & (which == b)).any()
            else np.nan
            for b in range(8)
        ]
    )
    expect = ref_bin[which]

    results = []
    for h in hyps:
        forb3 = h["forb"].reshape(shape)
        allowed3 = ~forb3
        # testable: no allowed node inside the node's blur
        testable = np.zeros(len(grid), dtype=bool)
        for j, b in enumerate(blur_keys):
            m = (blur_inv == j) & h["forb"]
            if not m.any():
                continue
            if not b.any():
                testable[m] = True
                continue
            near_allowed = maximum_filter(
                allowed3.astype(np.uint8),
                size=tuple(2 * b + 1),
                mode="constant",
                cval=1,
            )
            testable[m] = near_allowed.ravel()[m] == 0
        t_cov = testable[cov_idx]
        f_cov = h["forb"][cov_idx]
        worst, worst_desc, n_tested, n_untested = 0.0, "", 0, 0
        classes = []
        for s in np.unique(sig_inv[f_cov]):
            cls = f_cov & (sig_inv == s)
            use = cls & t_cov
            top = grid[cov_idx[cls]][np.argsort(-netf[cov_idx[cls]])[:3]]
            entry = {
                "nodes": int(cls.sum()),
                "testable": int(use.sum()),
                "examples": [[int(v) for v in e] for e in top],
                "also_forbidden_by": int(sig_keys[s].sum()),
            }
            classes.append(entry)
            if use.sum() < 3:
                n_untested += int(cls.sum())
                continue
            n_tested += int(use.sum())
            ratio = netf[cov_idx[use]].sum() / max(expect[use].sum(), 1e-9)
            z = netf[cov_idx[use]].sum() / np.sqrt((errf[cov_idx[use]] ** 2).sum())
            entry.update(
                net_counts=float(netf[cov_idx[use]].sum()),
                sigma=float(np.sqrt((errf[cov_idx[use]] ** 2).sum())),
                expected_if_present=float(expect[use].sum()),
                fraction_of_allowed=float(ratio),
            )
            if ratio > worst:
                ex = grid[cov_idx[use]][np.argsort(-netf[cov_idx[use]])[:3]]
                worst, worst_desc = (
                    ratio,
                    f"{use.sum()} nodes, {z:.0f} sigma, e.g. {' '.join(str(tuple(int(v) for v in e)) for e in ex)}",
                )
        results.append(
            {
                "names": h["names"],
                "n_forb": int(f_cov.sum()),
                "n_tested": n_tested,
                "n_untested": n_untested,
                "worst": worst,
                "desc": worst_desc,
                "ok": worst < args.threshold,
                "classes": classes,
            }
        )

    results.sort(key=lambda r: (not r["ok"], -r["n_tested"]))
    print(
        f"\nhypotheses, consistent first (a class counts against one if it carries more than {args.threshold:.0%} of comparable allowed nodes):"
    )
    print("   verdict      forbidden  tested  untestable  worst class   groups")
    for r in results[:14]:
        names = ", ".join(r["names"][:4]) + (" ..." if len(r["names"]) > 4 else "")
        print(
            f"   {'consistent' if r['ok'] else 'rejected  '}   {r['n_forb']:8d} {r['n_tested']:7d} {r['n_untested']:9d}     {r['worst']:6.1%}      {names}"
        )
        if not r["ok"]:
            print(f"                {'':40s} offending class: {r['desc']}")
        elif r["worst"] > 0.05:
            print(f"                {'':40s} marginal class: {r['desc']}")
    good = [r for r in results if r["ok"]]
    if good:
        best = max(good, key=lambda r: r["n_tested"])
        ties = [r for r in good if r["n_tested"] == best["n_tested"]]
        print("\nmost restrictive consistent hypothesis:")
        for r in ties:
            print(
                "   "
                + ", ".join(r["names"])
                + (
                    f"   ({r['n_untested']} forbidden nodes could not be tested)"
                    if r["n_untested"]
                    else ""
                )
            )

    if args.out:
        doc = {
            "laue_class": args.laue,
            "cell": [float(v) for v in cell],
            "d_min": float(d_min),
            "threshold": args.threshold,
            "relative_precision": args.rel_res,
            "nodes_with_data": int(cov.sum()),
            "hypotheses": [
                {
                    "space_groups": r["names"],
                    "consistent": bool(r["ok"]),
                    "forbidden_nodes": r["n_forb"],
                    "tested_nodes": r["n_tested"],
                    "untestable_nodes": r["n_untested"],
                    "worst_class_fraction": float(r["worst"]),
                    "classes": r["classes"],
                }
                for r in results
            ],
        }
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=1)
        print(f"\nwrote every hypothesis and its evidence to {args.out}")


if __name__ == "__main__":
    main()
