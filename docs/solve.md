# Joint count-based solve (draft)

`solve` is the sole CLI entry point for the experimental count-based indexing
and detector calibration workflow. `calibrate` and `spherical-index` are
removed from the CLI, with no aliases. Their numerical Python runners remain
available for regression comparisons; `solve` does not call them or the finder.
The conventional finder/indexer/integrator workflow is unaffected.

```bash
python -m subhkl.io.parser solve pooled.h5 solution.h5 \
  --metadata setting.h5 --d-min 3.2 --penalty 1.2
```

`pooled.h5` contains `images` shaped `(banks, rows, columns)` and distinct
`bank_ids`. Use one still or pool frames recorded at the same setting. Multiple
images per bank and mixed `goniometer/R` settings are rejected; summing a
rotation scan is not supported. The metadata file supplies the instrument
attribute, `sample/{a,b,c,alpha,beta,gamma,space_group}`, the wavelength band at
`instrument/wavelength`, and the common setting at `goniometer/R`. These can
live in the counts file instead. Cell/instrument/band values have CLI overrides.

Without `goniometer/R`, the sample frame is explicitly defined as the lab
frame. Supply setting metadata before using the result with a predictor that
reads nonidentity goniometer angles from the original acquisition. Angles
without their rotation matrix, nonzero goniometer translations, non-+z beams,
and curved panels are currently rejected rather than silently approximated.

## What is unified

The global lattice ladder receives spatial sums of all valid signed count
residuals. It proposes distinct orientations, quotienting by lattice
symmetries that preserve the allowed reflection set. One nonnegative Poisson
group-sparse fit then estimates reflection intensities and panel background
scales. A local outer refinement profiles that same objective over orientations
and one six-parameter geometry (radial scale, two tilts, three translations).
The group weights are frozen at the starting geometry of the fit/refinement.

The proposal score is an acceleration mechanism, not the likelihood itself.
Proposals can miss an orientation; `--n-candidates` is a maximum, not a promise
of that many distinct basins or an exhaustive multi-crystal search. The
regularization `--penalty` selects orientation groups, never individual observed
spot locations. Nearby groups can still represent one mismodeled crystal.
The default value is experimental, not calibrated false-alarm control over SO(3).

`--fixed-geometry` holds the instrument geometry fixed while fitting orientation
and intensities. `--no-refine` performs only the sparse intensity/background fit
on the proposed or supplied orientation dictionary. Neither option invokes the
retired calibration/indexing objectives.

## Background and shape priors

The default background shape is a local median of all valid count bins, with
a free scale per panel. This simple model is inadequate for some real static
structures. `--background-file background.h5` supplies positive expected counts
per original pixel in `images`, with `bank_ids` aligned by ID; its panel scales
are still fitted. `--static-mask-file` carries instrument validity, and partially
masked proposal bins retain their valid counts with their mean valid location.

The default reflection footprint is a pixel-integrated Gaussian, with
`--sigma-px` in original detector pixels. An informed non-Gaussian shape can be
supplied using `--profile-file prior.h5`, containing:

| Dataset | Meaning |
|---|---|
| `profile/u` | Increasing nonnegative radius in model sigma units |
| `profile/f` | Finite nonnegative radial shape at those radii |

The forward model integrates that prior over count bins and normalizes it on
the entire unmasked plane. Cropping and masks do not renormalize reflection
flux. The shape is persisted in the result so it can be reused.

The routines formerly in `search/matrix_free.py` are now public, shared APIs:

```python
from subhkl.search.profiles import (
    measure_radial_profile,
    measure_whitened_profile,
    RadialProfile,
)
```

Finder/integrator call those same implementations through their existing local
names. Both measurements qualify bright windows and may return `None` when
there is insufficient information. Those rules estimate a family prior; they
are **not** called automatically by `solve` and never gate its count bins.
For example, a profile measured from a separate reference exposure can be saved
and supplied without introducing a peak-detection stage into indexing:

```python
shape = measure_radial_profile(reference_images, reference_background,
                               bg_hi=background_level, max_sigma=6.0)
if shape is not None:
    with h5py.File("prior.h5", "w") as out:
        out["profile/u"], out["profile/f"] = shape
```

## Output and restart

`solution.h5` contains the usual cell, reciprocal matrix `sample/B`, wavelength
band, and absolute `detector_calibration/bank_*` geometry. That geometry is
loaded as a fresh baseline when restarting with `--bootstrap solution.h5`;
global instrument configuration is not mutated and corrections are not doubled.
A supplied metadata file's detector calibration is also honored.

| Dataset/group | Meaning |
|---|---|
| `solve/orientations_lab`, `solve/orientations_sample` | Every fitted candidate, in the named frame |
| `solve/active`, `solve/group_norms`, `solve/group_weights` | Sparse support and its normalization |
| `solve/intensities`, `solve/hkl` | Physical reflection flux estimates and column identities |
| `solve/g`, `solve/background_scales` | Shared geometry increment and background scales |
| `solve/objective_*`, `solve/stationarity_residual` | Fit diagnostics |
| `solve` attributes | Status, model settings and source provenance |
| `sample/U` | Primary active orientation, only for a converged nonempty solution |

The existing predictor consumes the primary `sample/U`. Multiple components
are saved in `solve/` and announced; exporting every component is future work.
These are group-sparse, shrunken intensity estimates, not final integrated
intensities for merging. The output is written atomically. An unconverged outer refinement or a fitted
empty support produces a diagnostic result without `sample/U` and a nonzero
CLI exit status. Invalid inputs, absent proposal evidence, and failures of an
inner intensity solve stop before replacing the output. Input files cannot
be overwritten by the output.

For an explicit dictionary, `--bootstrap candidates.h5` can read an
`orientations` dataset `(N,3,3)` in lab coordinates. It can also read the
`sample/U` of an existing single-crystal result, converted using the current
setting, or all `solve/orientations_lab` from a previous solve. The entire
provided dictionary is fitted; `--n-candidates` limits global proposals only.

## Migration and evidence

The actual `solve` CLI also recovers the single-crystal synthetic case from
nominal geometry without a bootstrap: **0.00943 degrees** orientation error,
**2.44599%** radial correction versus 2.50000% planted, and convergence in 510
objective evaluations. The machine-readable [CLI result](experiments/poisson-orientations/solve-cli.json)
records the fit diagnostics. The separate experiment scripts below are earlier
measurements of the numerical prototype.

Replace `calibrate pooled.h5 calibration.h5` followed by `spherical-index ...`
with one `solve pooled.h5 solution.h5 --metadata setting.h5`. There is no
one-for-one migration for the old spherical CLI's per-run/harmonic goniometer,
cell-shape, per-panel, beam, or curved-detector refinements. Those features
have not been ported to this draft model.

The [experiment report](poisson-orientations.md) records a threshold-free global
synthetic chain reaching 0.0096 degrees from nominal geometry, and a two-crystal
candidate test in which fitting geometry removes false extra orientations.
The real pooled-count objective favors a supplied calibrated geometry, but
background/profile mismatch activates unrelated candidates and strong penalties
expose boundary-solver failures. These are draft limitations, not evidence of
production-ready real-data calibration or crystal counting.
