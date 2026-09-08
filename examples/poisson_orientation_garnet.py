"""Reproduce the single-still CG4D garnet probes from PR #80.

Use run 2022 counts with computed goniometer/R and a reference HDF5 containing
`orientations` in lab coordinates. The reference is used only for comparison
in blind cases. Facility data are not included in this repository.
"""

import argparse
import json
from pathlib import Path

import gemmi
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.solve import run_solve

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frames", required=True)
parser.add_argument("--reference", required=True)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument(
    "--case", choices=["default", "blind", "seeded", "blind-refined"], required=True
)
parser.add_argument("--max-evals", type=int, default=600)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
with h5py.File(args.reference) as source:
    truth = source["orientations"][()].reshape(-1, 3, 3)[0]
result = run_solve(
    args.frames,
    args.output_dir / f"{args.case}.h5",
    bootstrap=args.reference if args.case == "seeded" else None,
    d_min=3.2 if args.case in ("default", "seeded") else 1.5,
    do_refine=args.case in ("seeded", "blind-refined"),
    max_evals=args.max_evals,
)
rotations = []
for op in gemmi.find_lattice_symmetry(
    gemmi.UnitCell(11.93, 11.93, 11.93, 90, 90, 90), "P", 0.01
):
    rotation = np.asarray(op.rot) / op.DEN
    if np.linalg.det(rotation) > 0:
        rotations.append(rotation)
angles = [
    min(
        np.degrees(Rotation.from_matrix(u @ s @ truth.T).magnitude()) for s in rotations
    )
    for u in result["orientations"]
]
fit = result["fit"]
summary = dict(
    case=args.case,
    status=result["status"],
    g=result["g"].tolist(),
    orientation_error_deg=angles,
    active=result["active"].tolist(),
    objective=fit.objective,
    stationarity_residual=fit.kkt,
    stationarity_tolerance=fit.kkt_tolerance,
)
(args.output_dir / f"{args.case}.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
