"""Prototype: unit-cell shape from quasi-Laue images, without a cell.

Without event times a spot gives the direction of its scattering vector but
not its length.  Directions are enough for the shape of the cell:

1. Spots (local maxima above the background) -> scattering-vector
   directions in the sample frame, all runs together.
2. Funk transform on the sphere (subhkl.search.spherical.funk_table): the
   spots of a zone h.x = 0 lie on the great circle perpendicular to the
   direct-lattice vector x, so the transform peaks at the directions of
   short direct-lattice vectors.
3. Three of those directions are taken as axes and a fourth as their body
   diagonal, which fixes the axial ratios; the choice that indexes the
   strongest poles with the smallest integers wins.  The result is reduced.

The shape (axial ratios and angles) comes out; the absolute scale does not.
Give one length (--scale-a) to fix it.  Directions alone cannot tell a
lattice from one commensurate with it (an axis doubled or halved); the
smallest-integer rule decides, which worked on the one dataset tried.

Tried on IMAGINE-X T4 lysozyme (five runs, 2,470 spots): axial ratios
1 : 1.006 : 1.582 and angles 89.8, 90.1, 119.0 degrees against
1 : 1 : 1.562 and 90, 90, 120; the basis found was the true primitive one.
The detector geometry used there came from an earlier solve (the nominal
IMAGINE-X geometry is off by several percent), so this was not a fully
blind run.

On MANDI data with the time of flight ignored (band-summed images) and
the refined geometry of a solve, it found the true primitive lattice of T4
lysozyme (ratios 1 : 1.002 : 1.604, angles 89.4, 90.4, 119.9 against
1 : 1 : 1.594 and 90, 90, 120) and of an I-centred monoclinic Mpro complex
(15 of 16 poles are lattice directions in both).  With the nominal MANDI
geometry it failed on both, for all runs together and for single runs: the
zone axes need the geometry and the rotations to a few tenths of a degree.

It failed on the second dataset tried, IMAGINE-X L1 MBL (fourteen runs,
3,050 spots, a larger monoclinic cell and a weaker crystal): only 4 of the
16 strongest poles were lattice directions, so step 2 already did not
deliver zone axes and the shape came out wrong.  With --reference the
script reports how many poles are true lattice directions.

Usage:
    python quasi_laue_cell_shape.py merged.h5 CG4D [--geometry solve.h5] [--scale-a 62.2]
"""

import argparse
import itertools

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from subhkl.commands import apply_detector_calibration
from subhkl.instrument.goniometer import Goniometer
from subhkl.search.blind_cell import cell_of, pixel_positions, reduce_basis
from subhkl.search.spherical import funk_table


def spot_directions(merged, instrument, rotations, images=None, min_z=8.0):
    """Scattering-vector directions of the spots, sample frame, with weights."""
    dirs, wts = [], []
    cache = {}
    with h5py.File(merged, "r") as f:
        bank_ids = f["bank_ids"][()]
        for i in range(f["images"].shape[0]) if images is None else images:
            img = f["images"][i].astype(np.float64)
            fine, coarse = gaussian_filter(img, 1.5), gaussian_filter(img, 10.0)
            z = (fine - coarse) / np.sqrt(
                np.maximum(coarse, 0.05) / (4 * np.pi * 1.5**2)
            )
            peak = (fine == maximum_filter(fine, size=9)) & (z > min_z)
            peak[:16], peak[-16:], peak[:, :16], peak[:, -16:] = (
                False,
                False,
                False,
                False,
            )
            rr, cc = np.nonzero(peak)
            if len(rr) == 0:
                continue
            bank = int(bank_ids[i])
            if bank not in cache:
                cache[bank] = pixel_positions(instrument, bank, img.shape).astype(
                    np.float64
                )
            xyz = cache[bank][rr * img.shape[1] + cc]
            q = xyz / np.linalg.norm(xyz, axis=1)[:, None] - np.array([0.0, 0.0, 1.0])
            dirs.append((q / np.linalg.norm(q, axis=1)[:, None]) @ rotations[i])
            wts.append((fine - coarse)[rr, cc])
    dirs, wts = np.concatenate(dirs), np.concatenate(wts)
    return dirs, np.minimum(wts, np.quantile(wts, 0.95))


def zone_poles(
    dirs, weights, n_poles=16, tol_deg=0.35, spacing_deg=0.4, min_sep_deg=2.5
):
    """Directions of the strongest peaks of the Funk transform."""
    table, off, nph, nth = funk_table(
        dirs, weights, tol_deg=tol_deg, spacing_deg=spacing_deg
    )
    table, nph, off = np.asarray(table), np.asarray(nph), np.asarray(off)
    th = (np.arange(nth) + 0.5) * np.pi / nth
    it = np.repeat(np.arange(nth), nph)
    ph = (np.arange(len(table)) - np.repeat(off, nph) + 0.5) * (2 * np.pi / nph[it])
    P = np.stack(
        [np.sin(th[it]) * np.cos(ph), np.sin(th[it]) * np.sin(ph), np.cos(th[it])], 1
    )
    spread = 1.4826 * np.median(np.abs(table - np.median(table)))
    poles, heights = [], []
    for j in np.argsort(-table)[:20000]:
        if all(abs(P[j] @ p) < np.cos(np.radians(min_sep_deg)) for p in poles):
            poles.append(P[j])
            heights.append((table[j] - np.median(table)) / spread)
        if len(poles) == n_poles:
            break
    return np.array(poles), np.array(heights)


def shape_from_poles(poles, n_axes=8, max_index=3, tol_deg=0.6):
    """Basis (arbitrary scale) whose lattice directions index the poles
    with the smallest integers."""
    r = range(-max_index, max_index + 1)
    small = np.array(
        [
            (a, b, c)
            for a in r
            for b in r
            for c in r
            if (a, b, c) != (0, 0, 0) and np.gcd.reduce([abs(a), abs(b), abs(c)]) == 1
        ]
    )
    weight = np.abs(small).sum(1)
    best = None
    for i, j, k in itertools.combinations(range(min(n_axes, len(poles))), 3):
        U = np.stack([poles[i], poles[j], poles[k]], 1)
        if abs(np.linalg.det(U)) < 0.15:
            continue
        C = np.linalg.solve(U, poles.T).T
        for m in range(len(poles)):
            if m in (i, j, k) or np.abs(C[m]).min() < 0.1:
                continue
            for unit in (
                (1, 1, 1),
                (1, 1, 2),
                (1, 2, 1),
                (2, 1, 1),
                (1, 2, 2),
                (2, 1, 2),
                (2, 2, 1),
            ):
                basis = U * (np.abs(C[m]) / np.array(unit))[None, :]
                cand = small @ basis.T
                cand = cand / np.linalg.norm(cand, axis=1)[:, None]
                cosang = np.abs(poles @ cand.T)
                which = np.argmax(
                    cosang - 1e-3 * weight[None, :] / weight.max(), axis=1
                )
                ang = np.degrees(
                    np.arccos(np.clip(cosang[np.arange(len(poles)), which], -1, 1))
                )
                hit = ang < tol_deg
                score = hit.sum() * 100 - weight[which][hit].sum()
                if best is None or score > best[0]:
                    best = (score, basis, int(hit.sum()))
    if best is None:
        raise ValueError("No three independent zone axes among the strongest poles.")
    return reduce_basis(best[1]), best[2]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("merged", help="merged images file")
    ap.add_argument("instrument")
    ap.add_argument(
        "--geometry",
        help="a solve file whose detector calibration and per-image rotations to use (not its cell)",
    )
    ap.add_argument(
        "--images",
        type=int,
        nargs=2,
        action="append",
        metavar=("FIRST", "LAST"),
        help="use images FIRST..LAST-1 (repeatable)",
    )
    ap.add_argument(
        "--scale-a",
        type=float,
        default=None,
        help="known length of the shortest axis [A]",
    )
    ap.add_argument("--reference", help="a solve file to compare the result with")
    args = ap.parse_args()

    if args.geometry:
        apply_detector_calibration(args.geometry, args.instrument)
        with h5py.File(args.geometry, "r") as f:
            rotations = f["goniometer/R"][()]
    else:
        with h5py.File(args.merged, "r") as f:
            rotations = Goniometer.from_h5_file(f).rotation
            if rotations.ndim == 2:
                rotations = np.broadcast_to(rotations, (f["images"].shape[0], 3, 3))
    images = None
    if args.images:
        images = [i for a, b in args.images for i in range(a, b)]
    dirs, wts = spot_directions(args.merged, args.instrument, rotations, images)
    print(f"{len(dirs)} spots")
    poles, heights = zone_poles(dirs, wts)
    print(
        f"{len(poles)} zone-axis poles, heights {heights.max():.1f} to {heights.min():.1f} sd above the transform's median"
    )
    basis, n_hit = shape_from_poles(poles)
    cell = np.array(cell_of(basis))
    print(f"{n_hit} of {len(poles)} poles indexed")
    print(
        "axial ratios: "
        + " : ".join(f"{v:.3f}" for v in cell[:3] / cell[0])
        + "; angles: "
        + ", ".join(f"{v:.2f}" for v in cell[3:])
    )
    if args.scale_a:
        scaled = cell[:3] * args.scale_a / cell[0]
        print(
            "cell with the given scale: "
            + " ".join(f"{v:.2f}" for v in scaled)
            + " "
            + " ".join(f"{v:.2f}" for v in cell[3:])
        )
    if args.reference:
        with h5py.File(args.reference, "r") as f:
            true = reduce_basis(np.linalg.inv(f["sample/U"][()] @ f["sample/B"][()]).T)
        tc = np.array(cell_of(true))
        r = range(-4, 5)
        uvw = np.array(
            [
                (a, b, c)
                for a in r
                for b in r
                for c in r
                if np.gcd.reduce([abs(a), abs(b), abs(c)]) == 1
            ]
        )
        X = uvw @ true.T
        X = X / np.linalg.norm(X, axis=1)[:, None]
        off = np.degrees(np.arccos(np.clip(np.abs(poles @ X.T).max(axis=1), -1, 1)))
        print(
            f"poles within 0.5 degrees of a reference lattice direction (indices up to 4): {int((off < 0.5).sum())} of {len(poles)}"
        )
        print(
            "reference:    "
            + " : ".join(f"{v:.3f}" for v in tc[:3] / tc[0])
            + "; angles: "
            + ", ".join(f"{v:.2f}" for v in tc[3:])
        )
        T = np.linalg.solve(true, basis * (tc[0] / cell[0]))
        print(
            "basis in terms of the reference (integers with determinant +-1 = the same lattice):"
        )
        print(
            "  "
            + str(np.round(T, 2).tolist())
            + f", determinant {np.linalg.det(T):.2f}"
        )


if __name__ == "__main__":
    main()
