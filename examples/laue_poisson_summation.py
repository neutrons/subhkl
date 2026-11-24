"""Unit cell from Laue images without a cell: find_cell in Laue mode.

A time-of-flight event is a point in reciprocal space.  A Laue reflection
is not: its direction is known, its length only to within the wavelength
band.  ``find_cell(wavelength=(lam_min, lam_max))`` spreads each reflection
along its radial streak over the band and takes the Fourier transform; the
map, the peak search, the lattice basis and its refinement are the same
code as for time-of-flight events.  A lattice vector does not need its own
zone (h.x = 0) to have been measured, because the first-order planes
contribute too.

Status, all runs of each dataset, blind cell against the known one:

- IMAGINE-X T4 lysozyme: 62.6 62.7 98.2, 89.9 90.1 119.9 against
  62.2 62.2 97.1, 90 90 120.
- MANDI T4 lysozyme with the time of flight ignored: 54.3 54.4 86.6,
  89.6 90.3 119.9 against 61.0 61.0 97.2, 90 90 120: the right lattice and
  shape (c/a 1.594 against 1.594), all lengths 11% short.
- MANDI Mpro complex as Laue: 51.5 59.7 59.8, 79.0 110.3 110.7 against the
  primitive cell 55.1 63.1 64.2, 78.0 110.7 111.8 (lengths 6% short).
- IMAGINE-X L1 MBL: 35.5 90.4 100.4, 90.1 90.1 96.1 against 69.0 90.9 97.9,
  89.9 90.0 106.1: the basis picked half of [-1 1 0], so the cell has half
  the volume.

The overall scale follows the assumed spectrum (a smooth bump over the
band unless one is given); the shape does not.  Counting every significant
pixel instead of one entry per reflection lost the lattice on IMAGINE-X.
The IMAGINE-X cases used detector geometry and rotations from an earlier
solve (--geometry), the MANDI cases nominal detectors and rotations from
``align_runs`` (--rotations).

Usage:
    python laue_poisson_summation.py merged.h5 INSTRUMENT lam_min lam_max [--geometry solve.h5]
        [--rotations rotations.npy] [--reference solve.h5] [--images FIRST LAST]...
"""

import argparse
import time

import h5py
import numpy as np

from subhkl.commands import apply_detector_calibration
from subhkl.search.blind_cell import cell_of, find_cell, reduce_basis

ap = argparse.ArgumentParser()
ap.add_argument("merged")
ap.add_argument("instrument")
ap.add_argument("lam_min", type=float)
ap.add_argument("lam_max", type=float)
ap.add_argument("--geometry")
ap.add_argument("--rotations")
ap.add_argument("--reference")
ap.add_argument("--images", type=int, nargs=2, action="append")
ap.add_argument("--grid", type=int, default=384)
ap.add_argument("--x-min", type=float, default=35.0)
args = ap.parse_args()
rot = None
if args.geometry:
    apply_detector_calibration(args.geometry, args.instrument)
    with h5py.File(args.geometry, "r") as f:
        rot = f["goniometer/R"][()]
if args.rotations:
    rot = np.load(args.rotations)
frames = [i for a, b in args.images for i in range(a, b)] if args.images else None
t0 = time.time()
try:
    found = find_cell(
        args.merged,
        args.instrument,
        grid=args.grid,
        x_min=args.x_min,
        frames=frames,
        rotations=rot,
        wavelength=(args.lam_min, args.lam_max),
        log=print,
    )
except (ValueError, StopIteration) as exc:
    print("no lattice:", exc)
    raise SystemExit(0)
print(
    f"{time.time() - t0:.0f} s; lattice Fourier sum {found.diagnostics['fourier_sum_after']:.3f}"
)
if args.reference:
    with h5py.File(args.reference, "r") as f:
        true = reduce_basis(np.linalg.inv(f["sample/U"][()] @ f["sample/B"][()]).T)
    print("reference (as stored): " + " ".join(f"{v:.2f}" for v in cell_of(true)))
    T = np.linalg.solve(true, found.basis)
    det = np.linalg.det(T)
    scale = abs(det) ** (1 / 3)
    print(
        f"   blind primitive basis in terms of the reference: {np.round(T, 2).tolist()}; volume ratio {det:.3f}"
    )
