"""Blind reduction of a time-of-flight Laue dataset: cell, point group, space group.

Starts from a merged images file with events (``reduce`` then
``merge-images``; the cell given to ``merge-images`` is a placeholder) and
needs no cell, orientation or space group:

1. the lattice of each run on its own, and from their differences the
   rotation errors of the goniometer     (subhkl.search.blind_cell.align_runs);
   then cell, absolute scale and orientation from the 3D Fourier transform
   of the raw events of all runs                                (find_cell)
2. detector geometry and per-run orientation from the existing refinement,
   seeded by step 1                      (solve --refinement band --bootstrap)
3. one prediction and integration in P 1 (peak-predictor, integrator)
4. which lattice rotations are symmetries of the intensities: a strong
   reflection's mate must be present (laue_class_presence, no scaling and
   no wavelength matching), cross-checked by comparing mates at matched
   wavelength                           (subhkl.search.space_group.laue_class)
5. which space groups of that point symmetry fit the raw events
                                         (space_group_candidates)

The result is a ranked list of space-group hypotheses with their evidence
(``space_group.json``).  Enantiomorphic pairs such as P 31 2 1 / P 32 2 1
cannot be told apart from intensities and are reported together.  Merging
is left to the usual commands, run with the chosen group.

Usage:
    python blind_reduction_tof.py merged.h5 MANDI workdir [--d-min 2.0] [--sohncke]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from subhkl.search.blind_cell import (
    align_runs,
    cell_of,
    conventional_basis,
    find_cell,
    operations_in_basis,
    write_seed,
)
from subhkl.search.space_group import (
    Observations,
    best_candidates,
    event_node_counts,
    laue_class,
    laue_class_presence,
    space_group_candidates,
)


def cli(log, *args):
    """Run one subhkl command, output to a log file."""
    cmd = [sys.executable, "-u", "-m", "subhkl.io.parser", *[str(a) for a in args]]
    print("  $ subhkl " + " ".join(str(a) for a in args), flush=True)
    with open(log, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("merged", help="merged images file with events")
    ap.add_argument("instrument")
    ap.add_argument("workdir")
    ap.add_argument(
        "--d-min", type=float, default=2.0, help="resolution of the P 1 integration [A]"
    )
    ap.add_argument(
        "--test-d-min",
        type=float,
        default=2.5,
        help="resolution used by the symmetry tests [A]",
    )
    ap.add_argument(
        "--cell-d-min",
        type=float,
        default=3.0,
        help="resolution of the cell search [A]; lower it for small cells",
    )
    ap.add_argument("--grid", type=int, default=768, help="grid of the cell search")
    ap.add_argument(
        "--x-min",
        type=float,
        default=12.0,
        help="shortest lattice vector considered [A]",
    )
    ap.add_argument(
        "--sohncke",
        action="store_true",
        help="only space groups a chiral molecule can take",
    )
    ap.add_argument(
        "--reuse", action="store_true", help="skip steps whose output already exists"
    )
    args = ap.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    seed, refined = work / "blind_seed.h5", work / "blind_refined.h5"
    pred, integrated = work / "blind_pred.h5", work / "integrated_p1.h5"

    print("1. cell and orientation from raw events")
    say = lambda m: print("  " + m)
    runs = align_runs(
        args.merged,
        args.instrument,
        d_min=args.cell_d_min,
        grid=args.grid,
        x_min=args.x_min,
        log=say,
    )
    found = find_cell(
        args.merged,
        args.instrument,
        d_min=args.cell_d_min,
        grid=args.grid,
        x_min=args.x_min,
        log=say,
        rotations=runs.rotations,
    )
    band = write_seed(found, args.merged, args.instrument, seed)
    # The refinement keeps the metric symmetry of the lattice (not of the
    # crystal, which is not known yet): the lattice group constrains the cell.
    cell_arg = ",".join(f"{v:.4f}" for v in found.setting_cell)

    print("2. geometry and per-run orientation (existing refinement, seeded)")
    if not (args.reuse and refined.exists()):
        cli(
            work / "solve.log",
            "solve",
            args.merged,
            refined,
            "--refinement",
            "band",
            "--bootstrap",
            seed,
            "--instrument",
            args.instrument,
            "--cell",
            cell_arg,
            "--space-group",
            found.setting_group,
            "--wavelength-min",
            band[0],
            "--wavelength-max",
            band[1],
            "--d-min",
            args.test_d_min,
        )

    print("3. one integration in P 1")
    if not (args.reuse and pred.exists()):
        cli(
            work / "predict.log",
            "peak-predictor",
            args.merged,
            args.instrument,
            refined,
            pred,
            "--d-min",
            args.d_min,
            "--wavel-min",
            band[0],
            "--wavel-max",
            band[1],
            "--space-group",
            found.p1_symbol,
            "--rel-border-width",
            0.03,
            "--static-mask-file",
            refined,
        )
    if not (args.reuse and integrated.exists()):
        cli(
            work / "integrate.log",
            "integrator",
            args.merged,
            args.instrument,
            pred,
            integrated,
            "--anisotropic",
            "--fit-mosaicity",
            "--shape-fit-normalized",
            "--shape-fit-min-snr",
            5.0,
            "--allow-negative",
            "--matrix-free",
            "--rel-border-width",
            0.03,
            "--static-mask-file",
            refined,
        )

    print("4. point group: is the mate of a strong reflection present?")
    n_ops = len(found.setting_operations)
    laue = laue_class_presence(
        Observations.from_file(integrated, d_min=args.test_d_min, match=None),
        found.setting_operations,
    )
    print(
        f"  mate absent: same reflection twice {laue.same:.1%}, unrelated reflections {laue.unrelated:.1%}, accept below {laue.cut:.1%}"
    )
    for row in laue.rows:
        print(
            f"  {row['name']:28s} {row['pairs']:6d} usable pairs  mate absent {row['spread']:6.1%}  {row['verdict']}"
        )
    print(
        f"  point group {laue.point_group}, Laue class {laue.laue} ({len(laue.rotations)} of {n_ops} lattice rotations)"
    )
    if laue.conflicts:
        print(
            "  WARNING: rejected operations that the accepted ones imply: "
            + ", ".join(laue.conflicts)
        )
    try:
        check = laue_class(
            Observations.from_file(integrated, d_min=args.test_d_min),
            found.setting_operations,
        )
        same_group = {tuple(m.ravel()) for m in check.rotations} == {
            tuple(m.ravel()) for m in laue.rotations
        }
        print(
            f"  cross-check at matched wavelength (same {check.same:.2f}, unrelated {check.unrelated:.2f}): "
            f"point group {check.point_group}"
            + ("" if same_group else "  <- DISAGREES with the presence test")
        )
    except ValueError as exc:
        print(f"  cross-check at matched wavelength: no verdict ({exc})")

    print("5. space group: absences tested on raw events")
    with h5py.File(refined, "r") as f:
        ub = f["sample/U"][()] @ f["sample/B"][()]
        frames = f["goniometer/R"][()]
    basis, rotations, mult = np.linalg.inv(ub).T, laue.rotations, found.multiplicity
    if found.centring != "P" and found.p1_symbol == "P 1":
        # A centred lattice that was integrated in its primitive cell
        # (rhombohedral): the absences are tested in the conventional one.
        basis, T, mult = conventional_basis(basis)
        rotations = operations_in_basis(rotations, T)
    nodes = event_node_counts(
        args.merged, args.instrument, basis, frames, args.test_d_min
    )
    results = space_group_candidates(nodes, rotations=rotations, sohncke=args.sohncke)
    print(
        "  conventional cell: a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *cell_of(basis)
        )
        + f" ({mult} lattice point(s))"
    )
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

    report = {
        "cell": [round(float(v), 4) for v in cell_of(basis)],
        "lattice_points_per_cell": int(mult),
        "lattice_group": found.lattice_group,
        "cell_search": found.diagnostics,
        "run_corrections_deg": runs.angles,
        "rotation_axis_lab": None if runs.axis is None else runs.axis.tolist(),
        "rotation_axis_tilt_deg": runs.axis_tilt,
        "point_group": laue.point_group,
        "laue_class": laue.laue,
        "operations": [
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in row.items()
            }
            for row in laue.rows
        ],
        "test": "presence",
        "yardsticks": {
            "same": laue.same,
            "unrelated": laue.unrelated,
            "cut": laue.cut,
            "friedel": laue.friedel,
        },
        "hypotheses": results,
    }
    with open(work / "space_group.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(f"wrote {work / 'space_group.json'}")


if __name__ == "__main__":
    main()
