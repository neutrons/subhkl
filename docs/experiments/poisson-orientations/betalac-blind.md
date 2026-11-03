# Blind band indexing on betalac

`solve --refinement band --finder-peaks finder.h5` now works without an
orientation bootstrap. It reuses the existing lattice ladder and band geometry
refinement, then passes the discovered orientation to the staged scan fit.
This is a single-crystal, peak-based path. It does not eliminate detection
thresholds or demonstrate blind indexing from Poisson counts.

## Search and refinement

The known cell, space group, wavelength band, nominal detector metrology and
frame-addressed goniometry are supplied. Runs 2–9 are identified as stable;
the final fit uses runs 2–15. No reference orientation or fitted calibration
is read during discovery or refinement.

1. Search the orientation grid with the band lattice ladder on stable runs.
2. Remove symmetry-equivalent proposals and jointly refine each remaining
   orientation with global detector geometry.
3. Hold the best candidate's geometry common, locally refit all candidate
   orientations, and rank them on all stable-run observations.
4. Refine the selected orientation through the existing geometry, cell,
   per-run drift and panel stages, with optional pixel polish.

The default search budget is 16,384 observations. Larger lists use absolute-
weight importance sampling with a reproducible seed; final ranking uses the
full stable-run list. Betalac has only 918 eligible stable-run peaks, so this
benchmark uses every one exactly. Finder weights are square-root deviance,
with the existing deviance >=100 selection. Candidate scores are optimization
diagnostics, not calibrated probabilities of correct indexing.

The discovered seed and search diagnostics are saved as `solution.blind.h5`;
completed refinement stages go to `solution.stages.h5`. The native result also
contains the search diagnostics under `blind_search`.

## Measured results

| Fit | Search time | Median residual | Within 2 sigma | Final status |
| --- | ---: | ---: | ---: | --- |
| Prior seeded peak fit | — | 1.447 px | 96.95% | Converged |
| Blind peak fit | 47.1 s | 1.447 px | 96.95% | Converged |
| Blind peak fit, phi origin shifted +37 degrees | 55.0 s | 1.447 px | 96.95% | Converged |

Search times are on one H100 and exclude the subsequent staged fit. Residuals
use all 1,572 finder observations at deviance >=100, without the fit's trimming,
against the nearest forward prediction on the observed panel. This measures
agreement with the same observations, not held-out predictive accuracy. Both
blind fits reach the seeded solution to numerical precision, with zero bound
hits and approximately 3.7 degrees of late-run scan correction.

The shifted-origin case changes the actual varying encoder, phi, and recomputes
frame rotations. Its recovered sample orientation transforms as expected. This
checks coordinate handling on the same images; it is not a second independent
dataset. Results do not establish general reliability or multi-crystal support.

## Raw-count discovery experiments

Four finder-less proposal experiments used signed raw-count evidence and no
orientation seed. They did **not** recover the validated orientation:

| Proposal evidence | Selected orientation difference from reference |
| --- | ---: |
| Signed excess over spatial median background | 147.2 degrees |
| Signed Poisson deviance over spatial median | 103.9 degrees |
| Same, with five coarse radial geometry starts | 48.4 degrees |
| Signed deviance over exposure-normalized scan median | 78.8 degrees |

Each used 1,594,944 stable-run count bins with a 16,384-draw importance budget.
These are band proposals weighted by count evidence, not optimization of the
full Poisson likelihood. Reference comparisons were performed only after
discovery. Remaining local refinement was interrupted once the selected
orientation lay beyond its recovery bounds; the solver did not automatically
reject these candidates. Failed experimental variants are preserved separately
and are not exposed as supported band options.

This isolates an unresolved global discovery problem. The earlier seeded
count-only band fit still works (2.04 px, 92.3% within 2 sigma), but it does not
show that the count score can identify the correct basin globally. These tests
do not distinguish background mismatch, sampling error and an insufficiently
selective proposal objective. A matched count model remains a research step.

## Reproduction

Copied source data, experimental code snapshots, intermediate files, logs and
evaluation scripts are in `/home/xvg/subhkl-examples/betalac-6`, with exact
commands in `REPRODUCE_GUIDE.txt`. From that directory:

```bash
python -m subhkl.io.parser solve source/merged.h5 work/blind-peaks.h5 \
  --refinement band --finder-peaks comparison/finder.h5 --pixel-polish \
  --runs 2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --stable-runs 2,3,4,5,6,7,8,9 --d-min 2.0 \
  --band-maxiter 1200 --n-candidates 6
```

Machine-readable measurements are in [betalac-blind-results.json](betalac-blind-results.json).
The default Poisson/group-L1 solver is unchanged by this experiment.
