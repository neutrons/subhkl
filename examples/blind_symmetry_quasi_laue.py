"""Point group and space-group candidates for quasi-Laue data, from a merge in P 1.

Quasi-Laue images carry no event times, so the cell cannot be read off the
events, and a rotation scan hardly ever measures symmetry mates at the same
wavelength.  With the cell known, the symmetry can still be found without
assuming it:

1. solve the orientation with the cell and the symmetry of the lattice only
   (solve --refinement band --space-group <lattice group>)
2. predict and integrate once in P 1      (peak-predictor, integrator)
3. merge in P 1, which does the wavelength normalisation
   (mtz-exporter, careless poly)
4. divide every observation by the scale the merge learned for it and
   compare symmetry mates pairwise    (Observations.scaled, laue_class);
   as a cross-check, correlate every merged reflection with its mate
   under each lattice rotation                    (laue_class_merged)
5. score the space groups of that point symmetry by their absences on the
   merged intensities       (merged_node_counts, space_group_candidates)

This script does steps 4 and 5 on the output of a Careless merge.  Screw
axes can only be judged if axial reflections were measured; where they
were not, the groups that differ only by a screw axis are reported together.

Usage:
    python blind_symmetry_quasi_laue.py careless_0.mtz --predictions careless_predictions_0.mtz \
        [--xval careless_xval_0.mtz] [--d-min 3.0] [--sohncke]

The pairwise test on scaled observations decides (it involves no merged
amplitudes, so the merge's prior cannot bias it); without --predictions the
merged-amplitude test is used alone.
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("merged", help="P 1 merge (MTZ with H, K, L, F, SigF)")
    ap.add_argument(
        "--predictions",
        help="per-observation output of the same merge (MTZ with Iobs, SigIobs, Scale, image_id)",
    )
    ap.add_argument(
        "--xval", help="half-dataset merges of the same run (MTZ with a 'half' column)"
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
    cc_half = half_dataset_cc(args.xval, args.d_min) if args.xval else None
    if cc_half is not None:
        print(f"half-dataset CC of the P 1 merge to {args.d_min} A: {cc_half:.3f}")

    merged = laue_class_merged(
        hkl, col["F"], d, mats, cc_half=cc_half, d_min=args.d_min
    )
    res = merged
    if args.predictions:
        _pmtz, p, phkl, pd = columns(args.predictions)
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
            f"point group, from {len(obs.I)} observations divided by their learned scale: "
            f"same reflection twice {res.same:.2f}, unrelated {res.unrelated:.2f}, accept below {res.cut:.2f}"
        )
        for row in res.rows:
            print(
                f"  {row['name']:28s} {row['strong']:6d} strong pairs  spread {row['spread']:6.2f}  {row['verdict']}"
            )
        print(
            f"  point group {res.point_group}, Laue class {res.laue} ({len(res.rotations)} of {len(mats)} lattice rotations)"
        )
    print(
        f"{'cross-check' if args.predictions else 'point group'}, from merged amplitudes: unrelated reflections correlate "
        f"{merged.unrelated:.3f}; a symmetry should reach {merged.same:.3f}; accept above {merged.cut:.3f}"
    )
    for row in merged.rows:
        print(
            f"  {row['name']:28s} {row['pairs']:6d} pairs  CC {row['spread']:6.3f}  {row['verdict']}"
        )
    print(
        f"  point group {merged.point_group}, Laue class {merged.laue} ({len(merged.rotations)} of {len(mats)} lattice rotations)"
    )
    same_group = {tuple(m.ravel()) for m in res.rotations} == {
        tuple(m.ravel()) for m in merged.rotations
    }
    if not same_group:
        print(
            "  WARNING: the two tests disagree; the pairwise test on scaled observations is used below"
        )
    if res.conflicts:
        print(
            "  WARNING: rejected operations that the accepted ones imply: "
            + ", ".join(res.conflicts)
        )

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
