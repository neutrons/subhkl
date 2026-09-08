"""Stress the finite dictionary with neighboring atoms and a mismatched PSF."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from poisson_orientation_experiment import experiment
from subhkl.search.poisson_orientation import fit_intensities


rng, U, g, data, model, _, flux = experiment(seed=19)
rotations = np.array(
    [
        Rotation.from_rotvec([0.0, np.radians(d), 0.0]).as_matrix() @ U[0]
        for d in [0.0, -0.1, 0.1, -0.3, 0.3]
    ]
)
candidates = np.concatenate([rotations, U[1:3]])
A = model.design(candidates, g)
rows = []
for psf in [6.0, 7.2]:
    model.sigma_px = psf
    generation = model.design(U[:1], g)
    data.counts = rng.poisson(data.background + generation @ (flux[0] * 120.0))
    model.sigma_px = 6.0
    for penalty in [1.2, 2.0]:
        fit = fit_intensities(A, data, len(candidates), penalty, max_iter=10000)
        row = dict(
            psf_true=psf,
            psf_fit=6.0,
            penalty=penalty,
            offsets_deg=[0.0, -0.1, 0.1, -0.3, 0.3, None, None],
            groups=fit.group_norms.tolist(),
            converged=fit.converged,
            kkt=fit.kkt,
            iterations=fit.iterations,
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
Path("/tmp/poisson-orientation-stress.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
