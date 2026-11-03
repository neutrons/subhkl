# Blind band indexing on betalac

`solve --refinement band` now works without an orientation bootstrap both from
finder observations (`--finder-peaks finder.h5`) and directly from raw detector
counts (`--finder-peaks` omitted). It reuses the existing lattice ladder and
band geometry refinement, then passes the discovered orientation to the staged
scan fit.

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
full stable-run list. Finder weights are square-root deviance, with the
existing deviance >=100 selection. For raw detector counts, `count_search_weights`
assigns smooth $\chi^2$-null-suppressed Poisson deviance weights
$w_p = \sqrt{D_p}(1 - \exp(-(D_p / 25)^3))$ to positive-deviance count bins
($y_p > b_p$) without a hard cutoff, suppressing $1\sigma$ background noise by
$6.4\times 10^{-5}$ while remaining sublinear ($\sqrt{D_p}$) on bright flares.
Subsequent staged count refinement uses the unmodified signed count residuals
$y_p - b_p$ across all 2,796,272 valid bins.

The discovered seed and search diagnostics are saved as `solution.blind.h5`;
completed refinement stages go to `solution.stages.h5`. The native result also
contains the search diagnostics under `blind_search`.

## Measured results

| Fit | Search time | Seed / Refined diff | Median residual | Within 2 sigma | Within 3 sigma | Final status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior seeded peak fit | — | — | 1.447 px | 96.95% | 99.05% | Converged |
| Blind peak fit | 47.1 s | 0.453° / 0.000° | 1.447 px | 96.95% | 99.05% | Converged |
| Blind peak fit, phi origin shifted +37 degrees | 55.0 s | 0.471° / 0.000° | 1.447 px | 96.95% | 99.05% | Converged |
| Prior seeded count-only fit | — | — / 0.060° | 2.230 px | 91.92% | 98.22% | Converged |
| **Blind count-only fit (no finder)** | **78.3 s** | **0.478° / 0.057°** | **2.251 px** | **91.98%** | **98.22%** | **Converged** |

Search times are on one GPU and exclude the subsequent staged fit. Residuals
use all 1,572 finder observations at deviance >=100, without trimming,
against the nearest forward prediction on the observed panel. Both blind peak
fits reach the seeded peak solution, and the blind count-only fit reaches the
seeded count-only solution (`0.0567°` from the peak reference, radial `+6.713%`,
axial `+5.520%`), with zero bound hits and approximately 3.7 degrees of late-run
scan correction.

## Why naive raw-count weights failed and $\chi^2$-suppressed deviance succeeds

Four initial finder-less proposal experiments used naive count weights and did
**not** recover the validated orientation (`48.4°–147.2°` error):

| Proposal evidence | Selected orientation difference from reference |
| --- | ---: |
| Signed excess over spatial median background ($y_p - b_p$) | 147.2 degrees |
| Square-root Poisson deviance ($\sqrt{D_p}$) | 103.9 degrees |
| Same, with five coarse radial geometry starts | 48.4 degrees |
| Square-root deviance over exposure-normalized scan median | 78.8 degrees |
| **Smooth $\chi^2$-null-suppressed deviance $\sqrt{D_p}(1 - \exp(-(D_p/25)^3))$** | **0.478 degrees (refined to 0.0567 degrees)** |

Under a Poisson background null, single-bin deviance $D_p \sim \chi^2_1$ has
mean 1 across all 1,594,944 count bins, while Bragg spots occupy `< 0.5%` of
bins ($D_p \gtrsim 25$). Weighting by $y_p - b_p$ or $\sqrt{D_p}$ caused `~98%`
of the 16,384 importance-sampling draws to land on $1\sigma$ background
fluctuations, whereas steep powers ($D_p^2$) caused unmasked direct-beam flare
bins to monopolize the sample. The smooth $\chi^2$-null-suppressed weight
eliminates the background flood (`7,919` informative bins drawn) without a hard
detection threshold or flare monopolization, producing a `3.5x` joint band-score
separation (`0.7619` vs `0.2179`) and `8.1x` common-geometry separation
(`0.7388` vs `0.0907`).

## Reproduction

Copied source data, experimental code snapshots, intermediate files, logs and
evaluation scripts are in `/home/xvg/subhkl-examples/betalac-6`, with exact
commands in `REPRODUCE_GUIDE.txt`. From that directory:

```bash
# Blind raw-count solve (no finder):
python -m subhkl.io.parser solve source/merged.h5 work/blind-counts.h5 \
  --refinement band \
  --runs 2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --stable-runs 2,3,4,5,6,7,8,9 --d-min 2.0 \
  --band-maxiter 1200 --n-candidates 6

# Blind peak-assisted solve:
python -m subhkl.io.parser solve source/merged.h5 work/blind-peaks.h5 \
  --refinement band --finder-peaks comparison/finder.h5 --pixel-polish \
  --runs 2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --stable-runs 2,3,4,5,6,7,8,9 --d-min 2.0 \
  --band-maxiter 1200 --n-candidates 6
```

Machine-readable measurements are in [betalac-blind-results.json](betalac-blind-results.json).
The default Poisson/group-L1 solver is unchanged by this experiment.
