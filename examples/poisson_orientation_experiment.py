"""Reproducible finite-dictionary experiment; see docs/poisson-orientations.md.

Run with the repository's src on PYTHONPATH. Outputs JSON, never rewrites data.
The planted orientations are deliberately included in the candidate list:
support recovery here tests the likelihood, not blind orientation discovery.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.calibrate import geometry_detectors
from subhkl.search.poisson_orientation import (
    CountData,
    OrientationModel,
    fit_intensities,
    refine,
)


def experiment(seed=17, n_banks=8, bin_px=8):
    rng = np.random.default_rng(seed)
    U = Rotation.random(6, random_state=rng).as_matrix()
    g = np.array([0.025, 0.008, -0.006, 0.001, -0.0015, 0.002])
    dets = geometry_detectors("CG4D", np.zeros(7))
    banks = list(dets)
    shapes = [(dets[b].n, dets[b].m) for b in banks]
    data = CountData.from_images(
        [np.zeros(s) for s in shapes], banks, [np.ones(s) * 0.5 for s in shapes], bin_px
    )
    model = OrientationModel(
        data, "CG4D", [12] * 3 + [90] * 3, "P 1", (2.0, 10.0), 2.0, 6.0
    )
    A = model.design(U[:2], g)
    hits = np.asarray(A @ np.ones(A.shape[1]))
    scores = np.bincount(data.bank_index, weights=hits)
    banks = [banks[i] for i in np.argsort(scores)[-n_banks:]]
    shapes = [(dets[b].n, dets[b].m) for b in banks]
    data = CountData.from_images(
        [np.zeros(s) for s in shapes], banks, [np.ones(s) * 0.5 for s in shapes], bin_px
    )
    model = OrientationModel(
        data, "CG4D", [12] * 3 + [90] * 3, "P 1", (2.0, 10.0), 2.0, 6.0
    )
    A = model.design(U, g)
    # Nonuniform, unknown reflection intensities; two distinct orientations.
    flux = rng.lognormal(0.0, 0.5, (len(U), model.n_hkl))
    return rng, U, g, data, model, A, flux


def run(output, geometry=False, joint=False):
    started = time.monotonic()
    rng, U, g, data, model, A, flux = experiment()
    records = []
    bg_scale = np.linspace(0.85, 1.15, len(model.banks))
    bg = data.background * bg_scale[data.bank_index]
    for n_true, brightness in [(0, 0.0), (1, 120.0), (2, 120.0), (2, 35.0)]:
        I = np.zeros_like(flux)
        I[:n_true] = flux[:n_true] * brightness
        data.counts = rng.poisson(bg + A @ I.ravel()).astype(float)
        for penalty in [0.8, 1.2, 2.0]:
            fit = fit_intensities(A, data, len(U), penalty)
            row = dict(
                n_true=n_true,
                brightness=brightness,
                penalty=penalty,
                active=np.flatnonzero(fit.group_norms > 0).tolist(),
                group_norms=fit.group_norms.tolist(),
                objective=fit.objective,
                background_scale_error=float(
                    np.max(abs(fit.background_scales - bg_scale))
                ),
                iterations=fit.iterations,
                kkt=fit.kkt,
                converged=fit.converged,
            )
            records.append(row)
            print(json.dumps(row), flush=True)
    out = {"support": records, "banks": model.banks, "g_true": g.tolist()}
    if geometry:
        I = np.zeros_like(flux)
        I[0] = flux[0] * 600.0
        data.counts = rng.poisson(bg + A @ I.ravel()).astype(float)
        # Start with a slightly imperfect orientation and nominal geometry.
        start_U = Rotation.from_rotvec([0.001, -0.001, 0.0015]).as_matrix() @ U[:1]
        start = fit_intensities(model.design(start_U, np.zeros(6)), data, 1, 1.2)
        result = refine(model, start_U, penalty=1.2, max_evals=500, log=print)
        err = Rotation.from_matrix(result["orientations"][0] @ U[0].T).magnitude()
        out["geometry"] = dict(
            g=result["g"].tolist(),
            objective_start=start.objective,
            objective_final=result["value"],
            angle_error_deg=float(np.degrees(err)),
            success=bool(result["optimizer"].success),
            message=str(result["optimizer"].message),
            nfev=int(result["optimizer"].nfev),
        )
    if joint:
        rng, U, g, data, model, A, flux = experiment(seed=23)
        I = np.zeros_like(flux)
        I[:2] = flux[:2] * 600.0
        data.counts = rng.poisson(data.background + A @ I.ravel())
        start = fit_intensities(model.design(U, np.zeros(6)), data, len(U), 1.2)
        result = refine(
            model, U, 1.2, max_evals=700, refine_orientations=False, log=print
        )
        out["joint"] = dict(
            seed=23,
            banks=model.banks,
            g_true=g.tolist(),
            g_fit=result["g"].tolist(),
            nominal_groups=start.group_norms.tolist(),
            final_groups=result["fit"].group_norms.tolist(),
            objective_start=start.objective,
            objective_final=result["value"],
            success=bool(result["optimizer"].success),
            message=str(result["optimizer"].message),
            nfev=int(result["optimizer"].nfev),
        )
    out["seconds"] = time.monotonic() - started
    Path(output).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="/tmp/poisson-orientation-results.json")
    p.add_argument("--geometry", action="store_true")
    p.add_argument("--joint", action="store_true")
    args = p.parse_args()
    run(args.output, args.geometry, args.joint)
