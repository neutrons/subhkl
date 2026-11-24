"""Prototype: the ladder with a free cell, for Laue data (no wavelengths).

Observations: spot directions d_m (sample frame), sin(theta_m), weights.

1. Directions of lattice vectors: peaks of the Funk transform (the spots of
   the zone h.x = 0 lie on the great circle perpendicular to x).
2. Length of each: for a lattice vector x = L n, every spot satisfies
   x.q = N (integer) with |q| = 2 sin(theta) / lambda, so
       y_m := 2 sin(theta_m) (n . d_m) = N lambda_m / L .
   Spots of order N = 0 sit at y = 0; order 1 fills [lam_min, lam_max] / L.
   Between them nothing is allowed: the gap ends at lam_min / L.  L is the
   value for which the first-order band is populated and the gap is empty.
3. Lattice: three of the vectors (direction x length) that index the rest;
   least squares on all indexed vectors; reduction.

Status (prototype, first step towards estimating the cell inside the
ladder so that Laue and time-of-flight data share one path).  Blind cell
against the known one:

- IMAGINE-X T4 lysozyme, five runs: 62.1 62.5 99.6, 89.4 90.8 118.9
  against 62.2 62.2 97.1, 90 90 120 (the true primitive lattice).
- IMAGINE-X L1 MBL, fourteen runs: 72.8 95.5 101.5, 89.4 90.3 105.8
  against 69.0 90.9 97.9, 89.9 90.0 106.1.  This needed the spots that stay
  on one detector pixel across runs to be removed: they gave a family of
  Funk peaks on a cone about the rotation axis, as strong as the true ones.
- MANDI with the time of flight ignored: Mpro complex 54.3 63.0 64.7,
  80.5 106.9 112.7 against the primitive cell 55.1 63.1 64.2, 78.0 110.7
  111.8; T4 lysozyme 54.3 59.5 92.8, 86.8 92.3 115.4 against 61.0 61.0
  97.2, 90 90 120 (the right lattice, lengths 5-11% short).
- Single runs: roughly right for IMAGINE-X T4 lysozyme and MANDI Mpro,
  wrong for MANDI T4 lysozyme.

The IMAGINE-X cases used detector geometry and rotations from an earlier
solve, the MANDI cases nominal detectors and rotations from align_runs.
Lengths carry a bias of a few percent that follows the assumed lower band
edge, and ladders longer than about 180 A are not resolved.

Usage:
    python free_cell_ladder.py merged.h5 INSTRUMENT lam_min lam_max [--geometry solve.h5]
        [--rotations rotations.npy] [--reference solve.h5] [--images FIRST LAST]...
"""

import argparse
import itertools

import h5py
import numpy as np
from quasi_laue_cell_shape import zone_poles
from scipy.ndimage import gaussian_filter, maximum_filter

from subhkl.commands import apply_detector_calibration
from subhkl.instrument.goniometer import Goniometer
from subhkl.search.blind_cell import cell_of, pixel_positions, reduce_basis


def spots(merged, instrument, rotations, images=None, min_z=8.0, drop_fixed=True):
    """Spot directions (sample frame), weights and sin(theta).

    With ``drop_fixed``, spots that recur at the same detector pixel in three
    or more runs are left out: a Bragg spot moves when the crystal turns, a
    detector artefact or a feature fixed in the laboratory does not, and such
    features produce Funk peaks on a cone about the rotation axis.
    """
    dirs, wts, sth = [], [], []
    where = []
    cache = {}
    with h5py.File(merged, "r") as f:
        bank_ids = f["bank_ids"][()]
        offsets = f["file_offsets"][()] if "file_offsets" in f else np.array([0])
        for i in range(f["images"].shape[0]) if images is None else images:
            img = f["images"][i].astype(np.float64)
            fine, coarse = gaussian_filter(img, 1.5), gaussian_filter(img, 10.0)
            z = (fine - coarse) / np.sqrt(
                np.maximum(coarse, 0.05) / (4 * np.pi * 1.5**2)
            )
            peak = (fine == maximum_filter(fine, size=9)) & (z > min_z)
            edge = max(8, img.shape[0] // 32)
            peak[:edge], peak[-edge:], peak[:, :edge], peak[:, -edge:] = (
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
            norm = np.linalg.norm(q, axis=1)
            dirs.append((q / norm[:, None]) @ rotations[i])
            sth.append(0.5 * norm)
            wts.append((fine - coarse)[rr, cc])
            run = int(np.searchsorted(offsets, i, side="right") - 1)
            where.append(
                np.stack([np.full(len(rr), bank), rr, cc, np.full(len(rr), run)], 1)
            )
    dirs, wts, sth = np.concatenate(dirs), np.concatenate(wts), np.concatenate(sth)
    where = np.concatenate(where)
    if drop_fixed and len(np.unique(where[:, 3])) >= 3:
        cell = (where[:, 0] * 4096 + where[:, 1] // 4) * 4096 + where[:, 2] // 4
        key = np.unique(np.stack([cell, where[:, 3]], 1), axis=0)
        cells, n_runs = np.unique(key[:, 0], return_counts=True)
        fixed = np.isin(cell, cells[n_runs >= 3])
        dirs, wts, sth = dirs[~fixed], wts[~fixed], sth[~fixed]
    return dirs, np.minimum(wts, np.quantile(wts, 0.95)), sth


def ladder_length(pole, dirs, sth, wts, lam_lo, lam_hi, lengths, zone_tol_deg=0.35):
    """Length of the lattice vector along a pole, and how clear it is."""
    c = dirs @ pole
    y = np.abs(2.0 * sth * c)
    off_zone = np.abs(c) > np.sin(np.radians(zone_tol_deg))
    y, w, s = y[off_zone], wts[off_zone], sth[off_zone]
    # resolution of y from the zone tolerance
    tol = 2.0 * s * np.sin(np.radians(zone_tol_deg))
    order = np.argsort(y)
    y, w, tol = y[order], w[order], tol[order]
    cum = np.concatenate([[0.0], np.cumsum(w)])
    density = cum[-1] / max(y[-1], 1e-9)  # mean weight per unit y, as a chance level
    best = (-np.inf, np.nan, 0.0, 0.0)
    scores = np.zeros(len(lengths))
    for j, L in enumerate(lengths):
        lo, hi = lam_lo / L, lam_hi / L
        margin = np.median(tol)
        if lo < 3 * margin:
            continue
        gap = (
            cum[np.searchsorted(y, lo - margin)] - cum[np.searchsorted(y, 2.0 * margin)]
        )
        band = cum[np.searchsorted(y, hi)] - cum[np.searchsorted(y, lo)]
        e_gap, e_band = density * max(lo - 3 * margin, 1e-9), density * (hi - lo)
        # populated first order, empty gap; both relative to chance
        scores[j] = band / e_band - gap / e_gap
        if scores[j] > best[0]:
            best = (scores[j], L, band / e_band, gap / e_gap)
    return best, scores


def assemble(vecs, weight, tol=0.2):
    """Lattice basis from a set of lattice vectors with a few percent of error."""
    order = np.argsort(np.linalg.norm(vecs, axis=1))
    short = order[:10]
    best = None
    for i, j, k in itertools.combinations(short, 3):
        A = np.stack([vecs[i], vecs[j], vecs[k]], 1)
        vol = abs(np.linalg.det(A))
        if vol < 0.2 * np.prod(np.linalg.norm(A, axis=0)):
            continue
        c = np.linalg.solve(A, vecs.T).T
        hit = (np.abs(c - np.rint(c)).max(axis=1) < tol) & (
            np.abs(np.rint(c)).sum(axis=1) > 0
        )
        key = (round(float(weight[hit].sum()), 6), -vol)
        if best is None or key > best[0]:
            best = (key, A, hit, np.rint(c))
    if best is None:
        return None
    _key, A, hit, n = best
    for _ in range(3):
        N, X, W = n[hit].T, vecs[hit].T, weight[hit]
        A = (X * W) @ N.T @ np.linalg.inv((N * W) @ N.T)
        c = np.linalg.solve(A, vecs.T).T
        n = np.rint(c)
        hit = (np.abs(c - n).max(axis=1) < tol) & (np.abs(n).sum(axis=1) > 0)
    return reduce_basis(A), int(hit.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("instrument")
    ap.add_argument("lam_min", type=float)
    ap.add_argument("lam_max", type=float)
    ap.add_argument("--geometry")
    ap.add_argument("--rotations", help="npy file of per-image rotations")
    ap.add_argument("--reference")
    ap.add_argument("--images", type=int, nargs=2, action="append")
    ap.add_argument("--poles", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--max-length",
        type=float,
        default=180.0,
        help="ladders longer than this are not resolved [A]",
    )
    ap.add_argument(
        "--keep-fixed",
        action="store_true",
        help="keep spots that stay on one pixel across runs",
    )
    args = ap.parse_args()

    if args.geometry:
        apply_detector_calibration(args.geometry, args.instrument)
        with h5py.File(args.geometry, "r") as f:
            rot = f["goniometer/R"][()]
    else:
        with h5py.File(args.merged, "r") as f:
            rot = Goniometer.from_h5_file(f).rotation
            if rot.ndim == 2:
                rot = np.broadcast_to(rot, (f["images"].shape[0], 3, 3))
    if args.rotations:
        rot = np.load(args.rotations)
    images = [i for a, b in args.images for i in range(a, b)] if args.images else None
    dirs, wts, sth = spots(
        args.merged, args.instrument, rot, images, drop_fixed=not args.keep_fixed
    )
    poles, heights = zone_poles(dirs, wts, n_poles=args.poles)
    true = None
    if args.reference:
        with h5py.File(args.reference, "r") as f:
            true = reduce_basis(np.linalg.inv(f["sample/U"][()] @ f["sample/B"][()]).T)
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
        Xl = np.linalg.norm(X, axis=1)
        Xd = X / Xl[:, None]
    lengths = np.exp(np.linspace(np.log(8.0), np.log(400.0), 1200))
    print(
        f"{len(dirs)} spots, {len(poles)} poles, band {args.lam_min}-{args.lam_max} A"
    )
    if not args.quiet:
        print(
            " pole  height   ladder length [A]  band/chance  gap/chance"
            + (
                "   true direction (angle) [uvw] length [A]   ratio"
                if true is not None
                else ""
            )
        )
    vecs, score = [], []
    ratios = []
    for n, (p, h) in enumerate(zip(poles, heights)):
        (_s, L, band, gap), _ = ladder_length(
            p, dirs, sth, wts, args.lam_min, args.lam_max, lengths
        )
        line = f" {n:3d}  {h:6.1f}   {L:10.2f}        {band:6.2f}      {gap:6.2f}"
        if true is not None:
            cosang = np.abs(Xd @ p)
            k = int(np.argmax(cosang - 1e-4 * Xl / Xl.max()))
            ang = np.degrees(np.arccos(min(cosang[k], 1.0)))
            line += f"       {ang:5.2f} deg  {tuple(int(t) for t in uvw[k])!s:14s} {Xl[k]:7.2f}   {L / Xl[k]:6.3f}"
            if ang < 0.5 and band > 1.5 and gap < 1.0 and L < args.max_length:
                ratios.append(L / Xl[k])
        if not args.quiet:
            print(line)
        if np.isfinite(L) and band > 1.5 and gap < 1.0 and L < args.max_length:
            vecs.append(p * L)
            score.append(band - gap)
    if ratios:
        r_ = np.array(ratios)
        print(
            f"clear ladders on true directions: {len(r_)}; length / true length: median {np.median(r_):.3f}, within 5% of 1, 2, 3 or 1/2: {np.mean(np.min(np.abs(r_[:, None] / np.array([1, 2, 3, 0.5]) - 1), axis=1) < 0.05) * 100:.0f}%, within 5% of 1: {np.mean(np.abs(r_ - 1) < 0.05) * 100:.0f}%"
        )
    if len(vecs) < 3:
        print("fewer than three clear ladders")
        return
    out = assemble(np.array(vecs), np.array(score))
    if out is None:
        print("no basis")
        return
    A, n_hit = out
    print(f"{len(vecs)} vectors with a clear ladder; {n_hit} indexed by the basis")
    print("blind cell:     " + " ".join(f"{v:.2f}" for v in cell_of(A)))
    if true is not None:
        print("reference cell: " + " ".join(f"{v:.2f}" for v in cell_of(true)))
        T = np.linalg.solve(true, A)
        scale = abs(np.linalg.det(T)) ** (1 / 3)
        print(
            f"   basis in terms of the reference: {np.round(T / scale, 2).tolist()}; volume ratio {np.linalg.det(T):.3f} (scale {scale:.3f})"
        )


if __name__ == "__main__":
    main()
