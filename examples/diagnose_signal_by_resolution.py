"""Diagnostic: is there Bragg signal in the raw images, shell by shell?

Stacks raw image patches around every predicted position, by resolution
shell, and compares with the same stack displaced diagonally (where no
reflection is predicted).  No integration, scaling or merge is involved,
so it separates "the crystal does not diffract there" from "the pipeline
loses it".

Per shell it prints the net counts per spot above the displaced control,
the noise of one spot (set by the background under it), and their ratio:
the signal-to-noise of a single observation.  Merging m observations of a
reflection gains about sqrt(m), so the last column is what a merge can
reach at best.

Usage:
    python diagnose_signal_by_resolution.py merged.h5 predictions.h5 [--band 3.1 4.0]
"""

import argparse

import h5py
import numpy as np

SHELLS = [
    (8.0, 99.0),
    (5.0, 8.0),
    (4.0, 5.0),
    (3.5, 4.0),
    (3.3, 3.5),
    (3.1, 3.3),
    (2.9, 3.1),
    (2.7, 2.9),
    (2.5, 2.7),
    (2.3, 2.5),
    (2.0, 2.3),
    (1.7, 2.0),
    (1.4, 1.7),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("merged", help="images file")
    ap.add_argument("predictions", help="peak-predictor output for the same images")
    ap.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=None,
        help="use predictions in this wavelength range only [A]",
    )
    ap.add_argument(
        "--radius", type=float, default=5.0, help="radius of the summed disc [pixel]"
    )
    ap.add_argument(
        "--shift", type=int, default=25, help="displacement of the control [pixel]"
    )
    args = ap.parse_args()
    half = int(args.radius) + 7

    parts = {k: [] for k in ("h", "k", "l", "lam", "img", "r", "c")}
    with h5py.File(args.predictions, "r") as f:
        B = f["sample/B"][()]
        for key, g in f["banks"].items():
            for name, src in (
                ("h", "h"),
                ("k", "k"),
                ("l", "l"),
                ("lam", "wavelength"),
                ("r", "i"),
                ("c", "j"),
            ):
                parts[name].append(g[src][()])
            parts["img"].append(np.full(g["h"].shape[0], int(key)))
    h, k, l, lam, img, r, c = (
        np.concatenate(parts[x]) for x in ("h", "k", "l", "lam", "img", "r", "c")
    )
    hkl = np.stack([h, k, l], 1)
    d = 1.0 / np.linalg.norm(hkl @ B.T, axis=1)
    use = (
        np.ones(len(d), bool)
        if args.band is None
        else (lam >= args.band[0]) & (lam <= args.band[1])
    )
    # observations per unique reflection (Friedel mates taken as one)
    flip = (
        (hkl[:, 2] < 0)
        | ((hkl[:, 2] == 0) & (hkl[:, 1] < 0))
        | ((hkl[:, 2] == 0) & (hkl[:, 1] == 0) & (hkl[:, 0] < 0))
    )
    code = (np.where(flip[:, None], -hkl, hkl) + 512) @ np.array([1048576, 1024, 1])

    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    rad = np.hypot(yy, xx)
    core, ring = rad <= args.radius, rad >= half - 2
    shell_of = np.full(len(d), -1)
    for i, (lo, hi) in enumerate(SHELLS):
        shell_of[(d >= lo) & (d < hi)] = i
    sel = np.flatnonzero(use & (shell_of >= 0))
    acc = np.zeros((len(SHELLS), 2, 3))  # shell, on/off, (net, variance, count)
    with h5py.File(args.merged, "r") as f:
        images = f["images"]
        H, W = images.shape[1:]
        cur, im = -1, None
        for i in sel[np.argsort(img[sel], kind="stable")]:
            if img[i] != cur:
                cur = img[i]
                im = images[cur].astype(np.float64)
            for which, shift in enumerate((0, args.shift)):
                rr, cc = round(r[i]) + shift, round(c[i]) + shift
                if not (half <= rr < H - half and half <= cc < W - half):
                    continue
                patch = im[rr - half : rr + half + 1, cc - half : cc + half + 1]
                bg = np.median(patch[ring])
                acc[shell_of[i], which] += (
                    patch[core].sum() - bg * core.sum(),
                    patch[core].sum() + core.sum() * max(bg, 0.0),
                    1,
                )

    print(
        " shell [A]     spots   net per spot   control   above control   noise per spot   signal/noise   obs. per reflection   best merged signal/noise"
    )
    for i, (lo, hi) in enumerate(SHELLS):
        on, off = acc[i]
        if on[2] < 20:
            continue
        net, ctl = on[0] / on[2], off[0] / max(off[2], 1)
        noise = np.sqrt(on[1] / on[2])
        m = use & (shell_of == i)
        mult = m.sum() / max(np.unique(code[m]).size, 1)
        sn = (net - ctl) / noise
        print(
            f" {lo:4.1f}-{min(hi, 30.0):4.1f}  {int(on[2]):7d}   {net:9.1f}   {ctl:7.1f}   {net - ctl:10.1f}   {noise:11.1f}   {sn:10.2f}   {mult:14.1f}   {sn * np.sqrt(mult):16.2f}"
        )


if __name__ == "__main__":
    main()
