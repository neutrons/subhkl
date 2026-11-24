"""Experimental: point group for quasi-Laue data from a merge in P 1.

Quasi-Laue images carry no event times, so the cell cannot be read off the
events, and a rotation scan hardly ever measures symmetry mates at the same
wavelength.  With the cell known, one can still try to find the symmetry
without assuming it:

1. solve the orientation with the cell and the symmetry of the lattice only
   (solve --refinement band --space-group <lattice group>)
2. predict and integrate once in P 1      (peak-predictor, integrator)
3. merge in P 1 so that the merge does the wavelength normalisation
   (mtz-exporter, careless poly)
4. test every lattice rotation in two ways:
   - correlate each merged reflection with its mate  (laue_class_merged)
   - divide each observation by the scale the merge assigned to it and
     compare mates pairwise              (Observations.scaled, laue_class)
5. score the space groups of that point symmetry by their absences on the
   merged intensities       (merged_node_counts, space_group_candidates)

This script does steps 4 and 5 on the output of a Careless merge and only
names a point group when both tests give a verdict and agree.

Status.  Tried on one dataset (IMAGINE-X T4 lysozyme, eleven runs).  The
outcome depends on how the merge was set up:

- With the scale model given all metadata (wavelength, angles, run), both
  tests found the right point group.  But with fewer than two observations
  per reflection the scale model fitted each observation individually, so
  the "scale" was not a wavelength normalisation and the second test was
  not independent of the first.
- With the scale model restricted to wavelength and run, the merge had no
  signal (half-dataset correlation near zero) and both tests are refused.

So treat a result from this script as a hypothesis to check, not as a
determination.  Screw axes can only be judged if axial reflections were
measured.

Usage:
    python blind_symmetry_quasi_laue.py careless_0.mtz --xval careless_xval_0.mtz \
        [--predictions careless_predictions_0.mtz] [--d-min 3.0] [--sohncke]
"""

import argparse

import gemmi
import numpy as np

from subhkl.search.blind_cell import lattice_operations
from subhkl.search.space_group import (
    Observations,
    best_candidates,
    laue_class,
    laue_class_merged,
    merged_node_counts,
    space_group_candidates,
)


def columns(path):
    mtz = gemmi.read_mtz_file(path)
    col = {c.label: np.array(c, dtype=float) for c in mtz.columns}
    hkl = np.stack([col["H"], col["K"], col["L"]], 1).astype(int)
    return mtz, col, hkl, mtz.cell.calculate_d_array(hkl)


def half_dataset_cc(path, d_min):
    """Correlation of the two half-dataset merges Careless writes."""
    _mtz, col, hkl, d = columns(path)
    half = col["half"].astype(int)
    ok = (d > d_min) & np.isfinite(col["F"])
    first = {
        tuple(h): f
        for h, f in zip(hkl[ok & (half == 0)].tolist(), col["F"][ok & (half == 0)])
    }
    second = {
        tuple(h): f
        for h, f in zip(hkl[ok & (half == 1)].tolist(), col["F"][ok & (half == 1)])
    }
    common = [k for k in first if k in second]
    if len(common) < 30:
        return None
    return float(
        np.corrcoef([first[k] for k in common], [second[k] for k in common])[0, 1]
    )


def show(res, n_ops, unit):
    for row in res.rows:
        print(
            f"    {row['name']:28s} {row['strong']:6d} pairs  {unit} {row['spread']:6.3f}  {row['verdict']}"
        )
    print(
        f"    point group {res.point_group}, Laue class {res.laue} ({len(res.rotations)} of {n_ops} lattice rotations)"
    )
    if res.conflicts:
        print(
            "    INCONSISTENT: rejected operations that the accepted ones imply: "
            + ", ".join(res.conflicts)
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("merged", help="P 1 merge (MTZ with H, K, L, F, SigF)")
    ap.add_argument(
        "--xval",
        required=True,
        help="half-dataset merges of the same run (MTZ with a 'half' column)",
    )
    ap.add_argument(
        "--predictions",
        help="per-observation output of the same merge (MTZ with Iobs, SigIobs, Scale, image_id)",
    )
    ap.add_argument(
        "--d-min", type=float, default=3.0, help="resolution used by the tests [A]"
    )
    ap.add_argument(
        "--sohncke",
        action="store_true",
        help="only space groups a chiral molecule can take",
    )
    args = ap.parse_args()

    mtz, col, hkl, d = columns(args.merged)
    cell = list(mtz.cell.parameters)
    mats, group = lattice_operations(cell)
    print(
        "cell: "
        + " ".join(f"{v:.2f}" for v in cell)
        + f"; lattice rotations: {len(mats)} ({group})"
    )
    cc_half = half_dataset_cc(args.xval, args.d_min)
    print(
        f"half-dataset correlation of the P 1 merge to {args.d_min} A: "
        + ("too few common reflections" if cc_half is None else f"{cc_half:.3f}")
    )

    verdicts = {}
    print("test 1, merged amplitudes of symmetry mates:")
    try:
        if cc_half is None:
            raise ValueError("no half-dataset correlation to set the scale of the test")
        res = laue_class_merged(
            hkl, col["F"], d, mats, cc_half=cc_half, d_min=args.d_min
        )
        print(
            f"    unrelated reflections correlate {res.unrelated:.3f}; a symmetry should reach {res.same:.3f}; accept above {res.cut:.3f}"
        )
        show(res, len(mats), "CC")
        if not res.conflicts:
            verdicts["merged"] = res
    except ValueError as exc:
        print(f"    no verdict: {exc}")

    if args.predictions:
        print("test 2, observations divided by the scale the merge gave them:")
        try:
            _pmtz, p, phkl, pd = columns(args.predictions)
            z = p["Iobs"] / p["SigIobs"]
            weak, strong = np.median(p["Scale"][z < 1]), np.median(p["Scale"][z > 5])
            note = ""
            if strong > 5 * abs(weak):
                note = "  <- the scale follows the observations themselves; test 2 is then not independent of test 1"
            print(
                f"    median scale of observations below 1 sigma {weak:.3g}, above 5 sigma {strong:.3g}{note}"
            )
            obs = Observations.scaled(
                phkl,
                p["Iobs"],
                p["SigIobs"],
                p["Scale"],
                pd,
                p["image_id"].astype(int),
                d_min=args.d_min,
            )
            res = laue_class(obs, mats)
            print(
                f"    same reflection twice {res.same:.2f}, unrelated {res.unrelated:.2f}, accept below {res.cut:.2f}"
            )
            show(res, len(mats), "spread")
            if not res.conflicts:
                verdicts["scaled"] = res
        except ValueError as exc:
            print(f"    no verdict: {exc}")

    groups = {
        name: frozenset(tuple(m.ravel()) for m in r.rotations)
        for name, r in verdicts.items()
    }
    wanted = 2 if args.predictions else 1
    if len(verdicts) < wanted or len(set(groups.values())) != 1:
        why = (
            "the tests disagree"
            if len(verdicts) == wanted
            else "a test gave no verdict"
        )
        print(f"point group: INCONCLUSIVE ({why})")
        return
    res = next(iter(verdicts.values()))
    print(f"point group (hypothesis): {res.point_group}, {res.symmorphic} setting")

    basis = np.array(mtz.cell.orth.mat.tolist())
    nodes = merged_node_counts(
        hkl, col["F"] ** 2, 2.0 * col["F"] * col["SigF"], basis, args.d_min
    )
    results = space_group_candidates(
        nodes, rotations=res.rotations, sohncke=args.sohncke, rel_res=0.0
    )
    print("space group:")
    print("  verdict      forbidden  tested  untestable  worst class   space groups")
    for r in results:
        print(
            f"  {'consistent' if r['consistent'] else 'rejected  '} {r['forbidden_nodes']:9d} {r['tested_nodes']:7d} {r['untestable_nodes']:9d}"
            f"     {r['worst_class_fraction']:6.1%}      {', '.join(r['space_groups'])}"
        )
    best = best_candidates(results)
    print(
        "  most restrictive consistent: "
        + ("; ".join(", ".join(r["space_groups"]) for r in best) or "none")
    )
    if best and best[0]["tested_nodes"] == 0 and len(best[0]["space_groups"]) > 1:
        print("  (no reflection that would tell these apart was measured)")


if __name__ == "__main__":
    main()
