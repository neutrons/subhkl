"""Diagnostic: where in wavelength is the Bragg signal, and does it last through the scan?

Predict and integrate over a band wider than the nominal one (for example
2 to 10 A with the existing solve), then run this on the integrator output.
It prints, from the data alone:

- the share of predicted reflections above 5 sigma in each wavelength bin.
  Outside the real band this sits on a floor of a few percent (a predicted
  position that happens to overlap another reflection's spot); the band is
  where it rises above that floor.
- the same share per run inside the band, split by resolution.  A crystal
  that decays loses the high-resolution part first while the spots stay
  where they are predicted.

Usage:
    python diagnose_band_and_decay.py integrated_wide.h5 [--band 3.2 4.0] [--split-d 4.0]
"""

import argparse

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "integrated", help="integrator output with predictions over a wide band"
    )
    ap.add_argument("--step", type=float, default=0.1, help="wavelength bin [A]")
    ap.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=None,
        help="band for the per-run table [A]; default: where the share exceeds half its maximum",
    )
    ap.add_argument(
        "--split-d",
        type=float,
        default=4.0,
        help="resolution separating low from high [A]",
    )
    ap.add_argument("--min-isigi", type=float, default=5.0)
    args = ap.parse_args()

    with h5py.File(args.integrated, "r") as f:
        p = {
            k: f["peaks"][k][()]
            for k in ("h", "k", "l", "intensity", "sigma", "lambda", "run_index")
        }
        B = f["sample/B"][()]
    ok = p["sigma"] > 0
    lam, run = p["lambda"][ok], p["run_index"][ok].astype(int)
    strong = (p["intensity"][ok] / p["sigma"][ok]) > args.min_isigi
    hkl = np.stack([p["h"], p["k"], p["l"]], 1)[ok]
    d = 1.0 / np.linalg.norm(hkl @ B.T, axis=1)

    edges = np.arange(np.floor(lam.min() * 10) / 10, lam.max() + args.step, args.step)
    which = np.clip(np.digitize(lam, edges) - 1, 0, len(edges) - 2)
    n = np.bincount(which, minlength=len(edges) - 1)
    share = np.bincount(which, strong, minlength=len(edges) - 1) / np.maximum(n, 1)
    print(f"{ok.sum()} reflections; share above {args.min_isigi:g} sigma by wavelength")
    print(" wavelength [A]   reflections   share")
    for a, cnt, s in zip(edges[:-1], n, share):
        if cnt >= 50:
            print(
                f"   {a:5.2f}-{a + args.step:5.2f}    {cnt:8d}    {100 * s:5.1f}%  "
                + "#" * round(60 * s / max(share.max(), 1e-9))
            )
    busy = n >= 50
    floor = float(np.median(np.sort(share[busy])[: max(3, busy.sum() // 4)]))
    half = floor + 0.5 * (share[busy].max() - floor)
    above = np.flatnonzero(busy & (share > half))
    lo, hi = edges[above[0]], edges[above[-1]] + args.step
    print(
        f"floor {100 * floor:.1f}%, maximum {100 * share[busy].max():.1f}%; above half height from {lo:.1f} to {hi:.1f} A"
    )

    lo, hi = args.band if args.band else (lo, hi)
    inside = (lam >= lo) & (lam < hi)
    print(
        f"\nper run, inside {lo:.1f}-{hi:.1f} A: reflections, share above {args.min_isigi:g} sigma (all, d > {args.split_d:g} A, d <= {args.split_d:g} A)"
    )
    for r in np.unique(run):
        m = inside & (run == r)
        if m.sum() < 20:
            continue
        low, high = m & (d > args.split_d), m & (d <= args.split_d)
        fmt = lambda sel: (
            f"{100 * strong[sel].mean():5.1f}%" if sel.sum() >= 20 else "   n/a"
        )
        print(f"   run {r:3d}: {m.sum():6d}   {fmt(m)}  {fmt(low)}  {fmt(high)}")


if __name__ == "__main__":
    main()
