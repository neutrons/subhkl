"""Threshold-free real pooled-count score, with externally supplied candidates.

This is a seeded model check, NOT evidence of blind indexing. Background shape
is a local median on a separately thinned training exposure. A validation
exposure tests predictive likelihood without re-fitting reflection intensities.
No pixels are selected by their count or residual; no static mask is used.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from scipy.special import xlogy

from subhkl.search.poisson_orientation import (
    CountData,
    OrientationModel,
    fit_intensities,
)


def run(args):
    rng = np.random.default_rng(42)
    with h5py.File(args.frames) as f:
        images, banks = f["images"][()], f["bank_ids"][()]
    with h5py.File(args.calibration) as f:
        U, g7 = f["calibrate/U"][()], f["calibrate/g"][()]
        cell = [
            float(f[f"sample/{k}"][()])
            for k in ["a", "b", "c", "alpha", "beta", "gamma"]
        ]
        sg = f["sample/space_group"][()].decode()
        band = f["instrument/wavelength"][()]
    # Apply the same beam-axis gauge to geometry and orientation.
    R = Rotation.from_rotvec(g7[1:4]).as_matrix()

    def zrot(angle):
        return Rotation.from_rotvec([0, 0, angle]).as_matrix()

    angle = brentq(
        lambda a: Rotation.from_matrix(zrot(a) @ R).as_rotvec()[2], -0.2, 0.2
    )
    Z = zrot(angle)
    v = Rotation.from_matrix(Z @ R).as_rotvec()
    g = np.r_[g7[0], v[:2], Z @ g7[4:]]
    U = Z @ U
    candidates = np.concatenate(
        [U[None], Rotation.random(3, random_state=rng).as_matrix()]
    )
    if not np.allclose(images, np.rint(images)):
        raise ValueError("Poisson thinning requires integer raw counts")
    train = rng.binomial(np.rint(images).astype(np.int64), 0.5)
    validation = images - train
    b = args.bin_px
    bg = []
    for im in train:
        h, w = im.shape
        counts = im.reshape(h // b, b, w // b, b).sum(axis=(1, 3))
        rate = np.maximum(median_filter(counts.astype(float), size=7), 0.1)
        bg.append(np.repeat(np.repeat(rate / (b * b), b, axis=0), b, axis=1))
    data = CountData.from_images(train, banks, bg, b)
    val = CountData.from_images(validation, banks, bg, b).counts
    records = []
    for sigma in args.sigmas:
        model = OrientationModel(data, "CG4D", cell, sg, band, 3.2, sigma)
        for name, geom in [("nominal", np.zeros(6)), ("seed_calibration", g)]:
            A = model.design(candidates, geom)
            for penalty in args.penalties:
                try:
                    fit = fit_intensities(
                        A, data, len(candidates), penalty, max_iter=5000, tol=1e-4
                    )
                except RuntimeError as exc:
                    record = dict(
                        geometry=name,
                        sigma_px=sigma,
                        penalty=penalty,
                        converged=False,
                        error=str(exc),
                    )
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    continue
                mu = fit.mean
                null_scale = np.bincount(
                    data.bank_index, weights=data.counts
                ) / np.bincount(data.bank_index, weights=data.background)
                null = data.background * null_scale[data.bank_index]
                val_gain = float(np.sum(val * np.log(mu / null) - (mu - null)))
                record = dict(
                    geometry=name,
                    sigma_px=sigma,
                    penalty=penalty,
                    groups=fit.group_norms.tolist(),
                    objective=fit.objective,
                    validation_loglik_gain=val_gain,
                    converged=fit.converged,
                    kkt=fit.kkt,
                    iterations=fit.iterations,
                    validation_half_deviance=float(
                        np.sum(mu - val + xlogy(val, val / mu))
                    ),
                )
                records.append(record)
                print(json.dumps(record), flush=True)
    Path(args.output).write_text(
        json.dumps(
            dict(
                frames=args.frames,
                calibration=args.calibration,
                bin_px=b,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--bin-px", type=int, default=16)
    p.add_argument("--sigmas", type=float, nargs="+", default=[6.0, 10.0])
    p.add_argument("--penalties", type=float, nargs="+", default=[1.2, 2.0, 4.0])
    p.add_argument("--output", default="/tmp/poisson-orientation-real.json")
    run(p.parse_args())
