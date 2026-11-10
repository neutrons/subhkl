"""Controlled garnet one-versus-many-setting experiment, using a shared seed.

Inputs: benchmark merged.h5, first-setting count file with goniometer/R,
a first-setting solve result, and reference lab orientations. Reference U is
used for scoring and fixed-orientation profiles, not the joint-refinement seed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import gemmi
import h5py
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix
from subhkl.search.multi_setting import MultiSettingModel
from subhkl.search.poisson_orientation import (
    CountData,
    OrientationModel,
    fit_intensities,
    orientation_weights,
    refine,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--merged", required=True)
p.add_argument("--first-setting", required=True)
p.add_argument("--seed", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--settings", type=int, nargs="+", default=[0, 10, 20])
p.add_argument("--binning", type=int, default=16)
p.add_argument("--max-evals", type=int, default=1500)
p.add_argument("--output", required=True, type=Path)
p.add_argument(
    "--evaluate",
    type=Path,
    nargs="+",
    help="Score saved fits on these settings, refitting only intensities/backgrounds",
)
a = p.parse_args()
start = time.monotonic()
with h5py.File(a.first_setting) as f:
    R0 = f["goniometer/R"][()].reshape(-1, 3, 3)[0]
with h5py.File(a.seed) as f:
    seed = R0.T @ f["solve/orientations_lab"][()]
if seed.shape != (1, 3, 3):
    raise ValueError("this controlled comparison requires one common seed orientation")
with h5py.File(a.reference) as f:
    truth = R0.T @ f["orientations"][()].reshape(-1, 3, 3)[0]
models, rotations, angles = [], [], []
with h5py.File(a.merged) as f:
    offsets = np.r_[f["file_offsets"][()], len(f["images"])]
    axes = f["goniometer/axes"][()]
    cell = [
        float(f["sample/" + k][()]) for k in ["a", "b", "c", "alpha", "beta", "gamma"]
    ]
    sg = f["sample/space_group"][()]
    if isinstance(sg, bytes):
        sg = sg.decode()
    band = f["instrument/wavelength"][()]
    for setting in a.settings:
        lo, hi = offsets[setting : setting + 2]
        setting_angles = f["goniometer/angles"][lo:hi]
        if not np.allclose(setting_angles, setting_angles[0]):
            raise ValueError("mixed angles within setting")
        r = calc_goniometer_rotation_matrix(axes, setting_angles[0])
        if setting == 0:
            np.testing.assert_allclose(r, R0, atol=1e-10)
        rotations.append(r)
        angles.append(setting_angles[0].tolist())
        banks = f["bank_ids"][lo:hi]
        data = CountData.from_images(
            f["images"][lo:hi], banks, np.ones(len(banks)), a.binning
        )
        for index in data.pixel_indices.values():
            data.background[index.ravel()] = np.maximum(
                median_filter(data.counts[index], size=7), 0.1
            ).ravel()
        models.append(OrientationModel(data, "CG4D", cell, sg, band, 1.5, 6))
model = MultiSettingModel(models, rotations)
ops = []
for op in gemmi.find_lattice_symmetry(gemmi.UnitCell(*cell), "P", 0.01):
    s = np.array(op.rot) / op.DEN
    if np.linalg.det(s) > 0:
        ops.append(s)


def error(U):
    return min(
        float(np.degrees(Rotation.from_matrix(U @ s @ truth.T).magnitude()))
        for s in ops
    )


summary = dict(
    settings=a.settings,
    angles_deg=angles,
    binning=a.binning,
    seed_error_deg=error(seed[0]),
    count_bins=len(model.data.counts),
    reference_radial_profile=[],
    max_evals=a.max_evals,
)
if a.evaluate:
    weights = orientation_weights(model.design(seed, np.zeros(6)), model.data, 1)
    summary["conditional_evaluation"] = []
    for filename in a.evaluate:
        saved = json.loads(filename.read_text())
        design = model.design(
            np.asarray(saved["orientations_sample"]), np.asarray(saved["g"])
        )
        fit = fit_intensities(
            design, model.data, 1, 1.2, weights, max_iter=2000, rtol=1e-6
        )
        summary["conditional_evaluation"].append(
            dict(
                source=str(filename),
                source_outer_converged=saved["outer_converged"],
                objective=fit.objective,
                converged=bool(fit.converged),
                kkt=fit.kkt,
                kkt_tolerance=fit.kkt_tolerance,
            )
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    sys.exit(0)
weights = orientation_weights(model.design(truth[None], np.zeros(6)), model.data, 1)
for radial in [0, 0.02, 0.04, 0.06, 0.08]:
    g = np.array([radial, 0, 0, 0, 0, 0])
    fit = fit_intensities(
        model.design(truth[None], g),
        model.data,
        1,
        1.2,
        weights,
        max_iter=300,
        rtol=1e-6,
    )
    record = dict(
        radial=radial,
        objective=fit.objective,
        converged=bool(fit.converged),
        kkt=fit.kkt,
    )
    summary["reference_radial_profile"].append(record)
    print("profile", record, flush=True)
result = refine(
    model,
    seed,
    penalty=1.2,
    max_evals=a.max_evals,
    inner_max_iter=300,
    log=lambda s: print(s, flush=True),
)
fit = result["fit"]
optimizer = result["optimizer"]
summary.update(
    g=result["g"].tolist(),
    orientation_error_deg=error(result["orientations"][0]),
    orientations_sample=result["orientations"].tolist(),
    objective=fit.objective,
    inner_converged=bool(fit.converged),
    kkt=fit.kkt,
    kkt_tolerance=fit.kkt_tolerance,
    outer_converged=bool(optimizer.success),
    message=str(optimizer.message),
    n_evaluations=int(optimizer.nfev),
    seconds=time.monotonic() - start,
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
