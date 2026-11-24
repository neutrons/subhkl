"""Prototype: Poisson summation for Laue data, the same map as for time of flight.

A time-of-flight event is a point in reciprocal space.  A Laue spot is not:
its direction is known, its length only to within the wavelength band.  So
each spot is spread along its radial streak between 2 sin(theta) / lam_max
and 2 sin(theta) / lam_min, weighted by a smooth spectrum.  The Fourier
transform of that streaked density is the direct-space map of
``subhkl.search.blind_cell``, and the rest is the same code: peaks,
``lattice_basis``, ``refine_basis``.

Compared with reading zone axes off the Funk transform, a lattice vector
does not need its own zone (h.x = 0) to have been measured: the first-order
planes contribute too.  Direction and length come out together.

Status, all runs of each dataset, blind cell against the known one:

- IMAGINE-X T4 lysozyme: 62.9 63.1 98.9, 89.9 90.1 119.8 against
  62.2 62.2 97.1, 90 90 120.
- MANDI T4 lysozyme with the time of flight ignored: 55.6 55.7 88.8,
  89.6 90.2 119.9 against 61.0 61.0 97.2, 90 90 120: the right lattice and
  shape (c/a 1.593 against 1.594), all lengths 9% short.  The zone-axis
  route failed here because the c axis is not among the Funk peaks.
- MANDI Mpro complex as Laue: 51.9 60.3 60.3, 79.0 110.3 110.7 against the
  primitive cell 55.1 63.1 64.2, 78.0 110.7 111.8 (lengths 5% short).
- IMAGINE-X L1 MBL: 35.4 90.3 100.2, 90.1 90.1 96.1 against 69.0 90.9 97.9,
  89.9 90.0 106.1: the map's strongest peaks are lattice vectors (38 of
  40), but the basis picked half of [-1 1 0], so the cell has half the
  volume.

The overall scale follows the assumed spectrum (here a smooth bump over
the band), from 9% short to 2% long; the shape does not.  Spots are still
found by a threshold; the next step is to streak the binned counts.  The
IMAGINE-X cases used detector geometry and rotations from an earlier
solve, the MANDI cases nominal detectors and rotations from align_runs.

Usage:
    python laue_poisson_summation.py merged.h5 INSTRUMENT lam_min lam_max [--geometry solve.h5]
        [--rotations rotations.npy] [--reference solve.h5] [--images FIRST LAST]...
"""

import argparse
import sys

import h5py
import numpy as np
from free_cell_ladder import spots

from subhkl.commands import apply_detector_calibration
from subhkl.instrument.goniometer import Goniometer
from subhkl.search.blind_cell import (
    cell_of,
    direct_space_map,
    find_peaks,
    lattice_basis,
    reduce_basis,
    refine_basis,
)

ap = argparse.ArgumentParser()
ap.add_argument("merged")
ap.add_argument("instrument")
ap.add_argument("lam_min", type=float)
ap.add_argument("lam_max", type=float)
ap.add_argument("--geometry")
ap.add_argument("--rotations")
ap.add_argument("--reference")
ap.add_argument("--images", type=int, nargs=2, action="append")
ap.add_argument("--d-min", type=float, default=3.0)
ap.add_argument("--grid", type=int, default=384)
ap.add_argument("--samples", type=int, default=96)
ap.add_argument("--x-min", type=float, default=35.0)
ap.add_argument("--spectrum", choices=("flat", "peaked"), default="peaked")
args = ap.parse_args()

if args.geometry:
    apply_detector_calibration(args.geometry, args.instrument)
    with h5py.File(args.geometry, "r") as f:
        rot = f["goniometer/R"][()]
else:
    with h5py.File(args.merged, "r") as f:
        rot = Goniometer.from_h5_file(f).rotation
if args.rotations:
    rot = np.load(args.rotations)
images = [i for a, b in args.images for i in range(a, b)] if args.images else None
dirs, wts, sth = spots(args.merged, args.instrument, rot, images)

# streak every spot along its direction between the band limits (uniform in wavelength)
n, q_max = args.grid, 1.0 / args.d_min
lam = np.linspace(args.lam_min, args.lam_max, args.samples)
q = 2.0 * sth[:, None] / lam[None, :]  # (spots, samples) [1/A]
pts = dirs[:, None, :] * q[:, :, None]
shape = np.ones(args.samples)
if args.spectrum == "peaked":
    # a smooth spectrum instead of a box: no sharp band edges, so the map does not ring
    shape = np.sin(np.pi * (np.arange(args.samples) + 0.5) / args.samples) ** 2
shape = shape / shape.sum()
w = wts[:, None] * shape[None, :]
pts, w = pts.reshape(-1, 3), w.ravel()
idx = np.floor((pts + q_max) * (n / (2.0 * q_max))).astype(int)
ok = np.all((idx >= 0) & (idx < n), axis=1)
grid = np.zeros((n, n, n), dtype=np.float32)
np.add.at(grid, tuple(idx[ok].T), w[ok])
# Friedel: the pattern is centrosymmetric; adding the inverse makes the map real
grid = grid + grid[::-1, ::-1, ::-1]
voxel = 1.0 / (2.0 * q_max)
print(
    f"{len(dirs)} spots streaked over {args.lam_min}-{args.lam_max} A; {int(ok.sum())} of {len(ok)} samples inside |q_i| <= {q_max:.3f}; direct voxel {voxel:.2f} A, extent {0.5 * n * voxel:.0f} A"
)

g = direct_space_map(grid, voxel, half_extent=0.5 * n * voxel - 2 * voxel)
pos, height = find_peaks(g, voxel, x_min=args.x_min, n_keep=300)
# A lattice vector shows as a few maxima along its own direction (fringes of
# the band average): keep one per direction and length, at their weighted mean.
keep_p, keep_h = [], []
used = np.zeros(len(pos), bool)
unit = pos / np.linalg.norm(pos, axis=1)[:, None]
length = np.linalg.norm(pos, axis=1)
for i in range(len(pos)):
    if used[i]:
        continue
    same = (
        (~used)
        & (unit @ unit[i] > np.cos(np.radians(2.0)))
        & (np.abs(length / length[i] - 1.0) < 0.12)
    )
    used |= same
    wgt = height[same]
    keep_p.append((pos[same] * wgt[:, None]).sum(0) / wgt.sum())
    keep_h.append(height[same].max())
pos, height = np.array(keep_p), np.array(keep_h)
order_ = np.argsort(-height)
pos, height = pos[order_], height[order_]
true = None
if args.reference:
    with h5py.File(args.reference, "r") as f:
        true = reduce_basis(np.linalg.inv(f["sample/U"][()] @ f["sample/B"][()]).T)
    coords = np.linalg.solve(true, pos.T).T
    near = np.abs(coords - np.rint(coords)).max(axis=1)
    r = range(-3, 4)
    print(
        "strongest peaks: |x| [A], height relative to the strongest, coordinates in the reference lattice"
    )
    for p_, h_, c_, e_ in list(zip(pos, height, coords, near))[:14]:
        print(
            f"   {np.linalg.norm(p_):7.2f}  {h_ / height[0]:5.2f}   [{c_[0]:+6.2f} {c_[1]:+6.2f} {c_[2]:+6.2f}]  {'lattice vector' if e_ < 0.12 else ''}"
        )
    k = min(40, len(pos))
    print(
        f"   of the {k} strongest peaks, {int((near[:k] < 0.12).sum())} are lattice vectors of the reference (within 0.12 in each coordinate)"
    )
try:
    A, good, diag = lattice_basis(pos, height, g, voxel)
except (ValueError, StopIteration) as exc:
    print("no lattice:", exc)
    sys.exit(0)
print(
    f"lattice: {int(good.sum())} of {len(pos)} peaks indexed ({100 * diag['indexed_weight']:.0f}% of the weight), rms misfit {diag['rms_misfit']:.2f} A"
)
print(
    "cell before refinement: " + " ".join(f"{v:.2f}" for v in cell_of(reduce_basis(A)))
)
sel = np.flatnonzero(ok)
take = sel[np.argsort(-w[sel])[: min(len(sel), 200000)]]
A2, before, after = refine_basis(A, pts[take], w[take], extent=0.45 * n * voxel)
A2 = reduce_basis(A2)
print(
    "cell after refinement:  "
    + " ".join(f"{v:.2f}" for v in cell_of(A2))
    + f"   (lattice Fourier sum {before:.3f} -> {after:.3f})"
)
if true is not None:
    print("reference cell:         " + " ".join(f"{v:.2f}" for v in cell_of(true)))
    T = np.linalg.solve(true, A2)
    print(
        f"   basis in terms of the reference: {np.round(T, 2).tolist()}; volume ratio {np.linalg.det(T):.3f}"
    )
