"""Attempt a global proposal from all signed, spatially binned residuals.

This tests the existing lattice ladder as a proposal mechanism. It is not the
Poisson group objective; proposals would subsequently need likelihood scoring.
The default starts from nominal geometry; --known-geometry is a control.
No orientation candidates or count thresholds are supplied. Spatially summed
bins supply proposals; the original bins supply the final count likelihood.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from poisson_orientation_experiment import experiment
from subhkl.calibrate import geometry_detectors
from subhkl.search.poisson_orientation import geometry_vector, refine
from subhkl.search.spherical import lattice_ladder, panel_directions


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--known-geometry", action="store_true")
p.add_argument("--refine", action="store_true")
p.add_argument("--output", default="/tmp/poisson-orientation-blind.json")
args = p.parse_args()
rng, U, g, data, model, A, flux = experiment(seed=17, bin_px=8)
intensities = np.zeros_like(flux)
intensities[0] = flux[0] * 600.0
data.counts = rng.poisson(data.background + A @ intensities.ravel())
g0 = g if args.known_geometry else np.zeros(6)
dets = geometry_detectors("CG4D", geometry_vector(g0), model.banks)
directions, residuals = [], []
for bank in model.banks:
    det = dets[bank]
    h, w = data.shapes[bank]
    residual = (data.counts - data.background)[data.pixel_indices[bank]]
    residuals.append(residual.reshape(h // 4, 4, w // 4, 4).sum(axis=(1, 3)).ravel())
    h, w = h // 4, w // 4
    rr, cc = np.meshgrid(
        (np.arange(h) + 0.5) * data.bin_px * 4 - 0.5,
        (np.arange(w) + 0.5) * data.bin_px * 4 - 0.5,
        indexing="ij",
    )
    directions.append(panel_directions(det, rr.ravel(), cc.ravel()))
D = np.concatenate(directions)
weights = np.concatenate(residuals)
result = lattice_ladder(
    D,
    weights,
    model.B,
    grid_deg=4.0,
    n_shortlist=300,
    rungs=(
        (13, 0.5, 2.5, 100, 5),
        (30, 0.5, 1.0, 30, 5),
        (60, 0.3, 0.5, 10, 5),
        (145, 0.3, 0.25, 5, 5),
    ),
    sin_theta=-D[:, 2],
    wavelength=(2.0, 10.0),
    d_min=2.0,
    cands=None,
    verbose=True,
)
symmetries = Rotation.create_group("O").as_matrix()
errors = [
    min(Rotation.from_matrix(r[0] @ (U[0] @ s).T).magnitude() for s in symmetries)
    for r in result
]
row = dict(
    angle_errors_deg=np.degrees(errors).tolist(),
    scores=[r[1] for r in result],
    n_bins=len(D),
    n_negative_residuals=int((weights < 0).sum()),
    true_geometry_supplied=args.known_geometry,
    orientations_supplied=False,
)
print(json.dumps(row), flush=True)
if args.refine:
    fit = refine(model, np.array([result[0][0]]), 1.2, g0=g0, max_evals=1500, log=print)
    row["refinement"] = dict(
        g_true=g.tolist(),
        g_fit=fit["g"].tolist(),
        angle_error_deg=float(
            np.degrees(
                min(
                    Rotation.from_matrix(
                        fit["orientations"][0] @ (U[0] @ s).T
                    ).magnitude()
                    for s in symmetries
                )
            )
        ),
        objective=fit["value"],
        success=bool(fit["optimizer"].success),
        message=str(fit["optimizer"].message),
        nfev=int(fit["optimizer"].nfev),
    )
Path(args.output).write_text(json.dumps(row, indent=2) + "\n")
