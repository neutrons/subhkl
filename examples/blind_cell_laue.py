"""Cell and per-run orientation from Laue images, with nothing given.

The chain alternates two library calls on the same count data:

1. ``find_cell(wavelength=...)``: one lattice for all runs.  The
   direct-space map proposes candidate lattices, the orientation search's
   band objective decides (``ladder_scores``), and where the band limits
   hold it the scale is read off them (``laue_scale``).
2. ``refine_run_rotations``: with that lattice fixed, each run has only its
   orientation left.  The band objective proposes a small turn of the run;
   the share of the run's count bins that the full lattice indexes
   (``indexed_share``) accepts or rejects it.

Goniometer errors blur the pooled map of step 1, and a blurred cell limits
step 2, so the two are repeated: three rounds suffice when the logged
angles are within a degree, five when they are 3 degrees off.  The turns
are then written as goniometer angles (``write_aligned_images``), so every
later step starts from them; projecting them onto a goniometer model is
left to ``solve``.

Blind against a full refinement, starting from the logged angles
(median / largest orientation error per run, before -> after):

- IMAGINE-X L1 MBL, 14 runs: 1.04 / 2.67 -> 0.17 / 0.35 degrees;
  cell 69.6 98.1 91.5, beta 106.0 against 69.0 97.9 90.9, 106.1.
- IMAGINE-X T4 lysozyme, 11 runs: 0.37 / 0.55 -> 0.32 / 0.51 degrees;
  cell 62.1 62.2 98.3, gamma 120.0 against 62.2 62.2 97.1, 120.  Here the
  chain indexes more of the late runs' count bins than the reference does
  (87 to 89% against 59 to 80%), so the remaining difference is not all
  ours.
- MANDI Mpro complex with the time of flight ignored, 17 runs, 5 rounds:
  1.77 / 3.05 -> 0.12 / 0.26 degrees; I-centred, beta 96.7 against 96.4,
  edges 7% short (50.8 74.9 81.5 against 54.7 80.5 88.0).
- MANDI T4 lysozyme with the time of flight ignored, 7 runs:
  0.99 / 1.45 -> 0.11 / 0.17 degrees; 59.8 59.9 95.6 against 61.0 61.0
  97.2.

What the data cannot give: the overall scale when the band spans an octave
(MANDI's 2 to 4 Angstrom).  Directions fix the shape; the size is in the
band limits only, and with an octave every ray has a lattice point in
every interval.  The script then reports the scale as undetermined and
holds the median volume over the rounds (lengths good to about 10%),
unless the scale is given: ``--volume`` or ``--length`` (one edge of the
conventional cell), or ``--scale-from-events``, which takes the volume from
the events of a single run of time-of-flight data (one run needs no
goniometer).  With
IMAGINE-X's band of 3.0 to 4.3 Angstrom the edges come out within about
1%.

IMAGINE-X needs a detector calibration (``--detectors``, a solve of any
crystal on the instrument).  Its nominal panel positions are 15 to 33 mm
off at 355 to 600 mm from the sample, and not by a rotation: with them the
chain finds a distorted lattice in the first round at best and then loses
it.  The calibration carries over between crystals: L1 MBL with the
detectors from the T4 lysozyme solve gives 70.3 99.1 92.4, beta 106.1 and
orientations to 0.35 degrees (0.17 with its own).

With the scale from the events of the first run, MANDI as Laue gives
60.7 60.9 97.1 for T4 lysozyme and 55.2 81.4 88.5, beta 96.6 for the Mpro
complex.

Usage:
    python examples/blind_cell_laue.py merged.h5 CG4D 3.0 4.3 \\
        --detectors earlier_solve.h5 --out aligned.h5
"""

import argparse

import numpy as np

from subhkl.search.blind_cell import (
    describe_lattice,
    find_cell,
    refine_run_rotations,
    write_aligned_images,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("merged", help="images file (all runs)")
    ap.add_argument("instrument")
    ap.add_argument("lam_min", type=float, help="[Angstrom]")
    ap.add_argument("lam_max", type=float, help="[Angstrom]")
    ap.add_argument(
        "--detectors", help="solve file whose detector calibration to apply"
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--d-min", type=float, default=3.0, help="[Angstrom]")
    ap.add_argument("--grid", type=int, default=384)
    ap.add_argument(
        "--x-min", type=float, default=35.0, help="[Angstrom] shortest cell edge"
    )
    ap.add_argument(
        "--volume", type=float, help="[Angstrom^3] conventional cell volume, if known"
    )
    ap.add_argument(
        "--length", type=float, help="[Angstrom] one conventional cell edge, if known"
    )
    ap.add_argument(
        "--scale-from-events",
        action="store_true",
        help="take the cell volume from the events of the first run (time-of-flight data)",
    )
    ap.add_argument("--out", help="images file with the refined goniometer angles")
    args = ap.parse_args()

    if args.detectors:
        from subhkl.commands import apply_detector_calibration

        apply_detector_calibration(args.detectors, args.instrument)
    band = (args.lam_min, args.lam_max)
    kw = {
        "d_min": args.d_min,
        "grid": args.grid,
        "x_min": args.x_min,
        "wavelength": band,
    }

    if args.scale_from_events:
        # One run needs no goniometer: its events give a lattice at the true
        # scale, whatever the rotation errors between runs.
        import h5py

        with h5py.File(args.merged, "r") as f:
            offsets = f["file_offsets"][()].tolist() if "file_offsets" in f else [0]
            n_first = (
                offsets[1] if len(offsets) > 1 else f["images"].shape[0]
            ) - offsets[0]
        tof = find_cell(
            args.merged, args.instrument, d_min=args.d_min, frames=range(n_first)
        )
        kw["volume"] = abs(np.linalg.det(tof.setting_basis))
        print(
            "from the events of the first run: "
            + " ".join(f"{v:.2f}" for v in tof.conventional_cell)
            + f"; conventional cell volume {kw['volume']:.0f} A^3"
        )
    elif args.volume or args.length:
        kw.update(volume=args.volume, length=args.length)

    rotations = None  # the logged goniometer angles
    volumes, found = [], None
    for k in range(args.rounds + 1):
        found = find_cell(args.merged, args.instrument, rotations=rotations, **kw)
        d = found.diagnostics
        volumes.append(abs(np.linalg.det(found.basis)))
        print(
            f"round {k}: "
            + " ".join(f"{v:.2f}" for v in found.conventional_cell)
            + f"  {found.centring}, lattice group {found.lattice_group}; score {1e3 * d['ladder_score']:.1f}"
            f" (z {d['ladder_z']:.0f}); scale "
            + (
                f"from the {d['scale_source']}"
                if d["scale_determined"]
                else "undetermined"
            )
        )
        if k == args.rounds:
            break
        rotations, angles, stats = refine_run_rotations(
            args.merged,
            args.instrument,
            found.basis,
            band,
            rotations=rotations,
            d_min=args.d_min,
        )
        print(
            "   runs turned by [deg]: "
            + " ".join(f"{a:.2f}" for a in angles)
            + f";  bins indexed {100 * np.nanmedian(stats[:, 2]):.0f}% -> {100 * np.nanmedian(stats[:, 3]):.0f}%"
        )

    basis = found.basis
    if not found.diagnostics["scale_determined"]:
        later = volumes[1:] or volumes
        basis = basis * (np.median(later) / volumes[-1]) ** (1 / 3)
        spread = (max(later) / min(later)) ** (1 / 3) - 1
        print(
            f"the band limits do not fix the scale: median volume of the rounds taken, "
            f"lengths spread {100 * spread:.0f}% between rounds"
        )
    info = describe_lattice(basis)
    print(
        "cell: "
        + " ".join(f"{v:.2f}" for v in info["conventional_cell"])
        + f"  centring {info['centring']}, lattice group {info['lattice_group']}"
    )
    if args.out and rotations is not None:
        miss = write_aligned_images(args.merged, rotations, args.out)
        print(
            f"wrote {args.out}; goniometer angles reproduce the rotations to {miss:.3f} deg"
        )


if __name__ == "__main__":
    main()
