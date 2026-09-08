# Experiment: one count likelihood for calibration and indexing

The synthetic experiment supports sharing the objective. A global proposal
from all signed residuals followed by the Poisson fit recovers one orientation
and detector geometry from nominal geometry, with no supplied orientation or
spot threshold. A separate two-crystal experiment selects the two orientations
while fitting the same shared geometry. This is an experimental module, not a
replacement for the existing CLI or evidence of blind recovery on real data.

Implementation: `src/subhkl/search/poisson_orientation.py`. Based on
`feat/calibrate` commit `9b35f5ba`, in branch `experiment/poisson-orientations`.

## Objective and implementation

For valid count bins p, orientation candidates j, and reflections h:

\[
\mu_p = b_p s_{\operatorname{bank}(p)} +
        \sum_{jh} P_{pjh}(g,R_j) I_{jh},\qquad I_{jh}\geq0,
\]

\[
\min_{g,R,I,s\geq0}\; \sum_p[\mu_p-y_p+y_p\log(y_p/\mu_p)]
    +\lambda\sum_j w_j\|c_{j,:}\|_2.
\]

Here `c_jh = I_jh sqrt(sum_p P_pjh**2 / b_p)` measures intensity in
approximate background-noise units. Group weights are the square root of the
number of geometrically visible reflections; outer refinement freezes those
weights at its starting geometry. Reflection normalization changes with the
forward model, so the penalty is explicitly on normalized intensities, not
unweighted physical flux. These weights are a prototype convention, not a
calibrated false-positive guarantee over SO(3).

The forward model uses flat CG4D panels, the known unit cell and wavelength
band, free reflection intensities, and pixel-integrated isotropic Gaussian
footprints. Five-sigma footprint truncation is geometric and independent of
counts. Counts are summed into bins; all valid bins, including zero counts and
negative background residuals, contribute. No maximum map, per-spot threshold,
argmax location, or observed peak list is used. Masks can express instrument
validity, but the real-data experiment uses no mask.

The inner solver uses nonnegative group proximal steps, Poisson backtracking,
acceleration with monotone/feasibility restarts, and fitted panel background
scales. It reports a proximal-gradient stationarity residual and convergence.
The outer reference implementation uses finite differences of the profiled
objective to fit six geometry parameters and optionally local orientation
increments. Assembly roll about the incident beam is fixed; the remaining
parameters are radial scale, two tilts, and three translations. There are no
per-panel calibration degrees of freedom.

## Synthetic results

These tests use actual CG4D panel geometry, a 12 A cubic cell, a 2–10 A band,
eight panels selected by predicted reflection coverage, Poisson counts,
lognormally varying unknown reflection intensities, and a known Gaussian PSF.
They intentionally begin with a matched forward model and informative panel
coverage. They do not establish general performance across instruments, cells,
spot shapes, or backgrounds. Counts use 8-pixel binning. Seeds and full numerical
results are in `experiments/poisson-orientations/`.

| Experiment | Result | Information supplied to fit |
|---|---|---|
| Background only | No active orientation at lambda 1.2 or 2 | Six candidate orientations |
| One crystal | Only planted orientation active | Candidate set includes truth |
| Two crystals, including weaker exposure | Both planted orientations active | Candidate set includes both truths |
| Shared geometry plus two-crystal support | Five active groups at nominal become exactly two | Six fixed orientation candidates; nominal geometry |
| Global proposal plus shared geometry/orientation fit | Final orientation error 0.0096 degrees | No orientation; nominal geometry |

In the two-crystal joint fit (seed 23), the radial scale is **2.5286%** versus
**2.5000%** planted. Fitted translation is `(0.989, -1.471, 1.976)` mm versus
`(1.000, -1.500, 2.000)` mm. The objective falls from 41638.33 to 19160.44;
the optimizer converges in 154 objective evaluations. The true orientations
are in the six-member candidate set and remain fixed during this test: this
isolates simultaneous geometry and support recovery, not blind multi-crystal
discovery.

In the global single-crystal test, the existing lattice ladder searches 364,500
coarse orientations using all 2,048 spatially summed residual bins; **980
residuals are negative and retained**. Geometry is nominal. Its best proposal
is 0.5364 degrees from the crystal, up to cubic symmetry. The Poisson objective
then jointly refines that orientation and geometry on the original count bins:

| Quantity | Planted | Recovered |
|---|---:|---:|
| Radial correction | 2.5000% | 2.4454% |
| Tilt x (radians) | 0.008000 | 0.007674 |
| Tilt y (radians) | -0.006000 | -0.005906 |
| Translation x (mm) | 1.000 | 1.002 |
| Translation y (mm) | -1.500 | -1.578 |
| Translation z (mm) | 2.000 | 2.012 |
| Orientation error | 0 | 0.0096 degrees |

The outer optimizer reports convergence after 860 evaluations. The proposal
stage uses the existing band-ladder score, not the Poisson group objective.
This is therefore a working threshold-free proposal/refinement chain, not a
global optimizer of one likelihood. It selects the top proposal for this
single-crystal test; blind multi-crystal proposal discovery is still untested.

A separate near-orientation initialization reaches 0.00745 degrees and a
2.5090% radial correction but exhausts its evaluation budget. Its artifact
preserves that failure-to-converge status; it is not counted as a converged
recovery result.

## What the stress tests expose

Neighboring dictionary atoms are not independent crystals. At lambda 1.2,
one correctly modeled crystal splits slightly between the true orientation
and a candidate 0.1 degrees away. With a PSF 20% wider than modeled, it splits
over four nearby candidates; even lambda 2 leaves two. These coefficients
cannot be interpreted as a crystal count without continuous orientation and
profile refinement, symmetry handling, and a test for genuinely distinct
orientations.

At lambda 0.8, even the background-only synthetic control admits weak false
groups. Eliminating individual detection thresholds does not eliminate
regularization selection or the need to calibrate it against an appropriate
null model and candidate multiplicity.

## Real pooled-count check

Data: `pool/merged_sum.h5` for CG4D L1 MBL, 56 panels. Candidates are an
orientation from the existing `calibrate_real.h5` result plus three seeded
random orientations. This is explicitly a seeded check. A beam-axis gauge
transform is applied consistently to the supplied geometry and orientation.

Integer counts are split by binomial thinning into training and validation
exposures. The background shape is a 7-bin median of the training data, with
one fitted scale per panel. All count bins are retained, using 16-pixel bins.
Validation uses frozen fitted reflection intensities and background scales.
However, the supplied orientation/calibration was itself obtained from the
full dataset: these validation scores are conditional predictive checks, not
an independent end-to-end validation of indexing or calibration.

At a 6-pixel PSF width:

| Geometry | lambda | Training penalized objective (lower better) | Validation log-likelihood gain over background (higher better) |
|---|---:|---:|---:|
| Nominal | 1.2 | 408582.49 | 80605.43 |
| Supplied calibrated geometry | 1.2 | 362186.59 | 139530.61 |
| Nominal | 4 | 453211.89 | 23552.59 |
| Supplied calibrated geometry | 4 | 433525.31 | 75241.48 |

Compare geometries **within** a penalty setting. The objective favors the
supplied calibration, which is useful evidence that it contains a geometry
signal. This experiment does not optimize real-data geometry from nominal.
Random orientations also remain active at these penalties. Their predictive
gain means an ordinary count split alone does not distinguish crystal signal
from repeatable background/profile errors that flexible reflection templates
can absorb. The stronger-penalty sweep records numerical nonconvergence or
line-search failure explicitly rather than claiming successful support
recovery. This reference solver and background model are not production ready.

## Consequence for unification

Keep one forward model and one profiled likelihood for calibration, orientation
refinement, and orientation support. The two-crystal experiment directly shows
why geometry and support belong together: correcting geometry removes false
extra orientations. Keep global proposal generation as a separate acceleration
mechanism until it has been tested on real counts without planted candidates.

A principled next step is to search for violations of the inactive-orientation
KKT condition, using normalized template columns F and the current fitted mean:

\[
\|[F_j^T(y/\mu-1)]_+\|_2 \leq \lambda w_j.
\]

An orientation violating this inequality can enter the active dictionary;
jointly refit intensities, orientations, geometry, and background afterward.
This performs selection on aggregate orientation evidence and never requires
detecting individual spots. Candidate search must also account for Laue
symmetry and duplicate/neighboring atoms. Real-data PSF and background modeling,
regularization calibration, and stable boundary solves need further work
before replacing either existing command.

## Reproduce

From this worktree, with the existing subhkl environment:

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
PYTHON=/home/xvg/subhkl/.venv/bin/python

$PYTHON examples/poisson_orientation_experiment.py --geometry --joint
$PYTHON examples/poisson_orientation_blind.py --refine
$PYTHON examples/poisson_orientation_stress.py
$PYTHON examples/poisson_orientation_real.py \
  --frames /home/xvg/.claude/jobs/cf39132c/tmp/pool/merged_sum.h5 \
  --calibration /home/xvg/.claude/jobs/cf39132c/tmp/calibrate_real.h5 \
  --sigmas 6 --penalties 1.2 4 6 8 12
$PYTHON -m pytest tests/test_poisson_orientation.py tests/test_calibrate.py \
  -m 'not slow' -q
```

The numerical tests include comparison with an independent scalar-lasso
optimizer, zero-count and negative-residual retention, masked-value exclusion,
the geometry convention, and null/one/two-group recovery from individually
weak reflections. The expensive geometric/global cases are reproducible
experiment scripts rather than part of the default unit-test suite.
