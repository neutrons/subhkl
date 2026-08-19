# Release Notes

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0]

The experimental paths that 1.3 kept alongside the new ones are gone: one
finder, one integrator, one indexing strategy. If your scripts used the
retired paths, the flags below tell you what replaced them. The calibration
grew per-run and periodic goniometer corrections, and the integrator's
intensities now come from a global amplitude solve with censoring instead of
per-patch fits.

### Removed

- **The zone-axis indexer is retired**: the `zone-axis-search` command and
  all its options are gone. The DE indexer is the one indexing strategy.
- **The per-patch integrator is retired**, and with it the 1.3 `integrator`
  command and its options (`--integration-method`, `--integration-mask-file`,
  `--integration-mask-rel-erosion-radius`, `--min-intensity`, `--ki-vec`).
  Integration is the matrix-free amplitude solve (below). Note the name
  reuse: `integrator` in this release is the *matrix-free* command (the
  renamed `rbf-integrator`), not the retired per-patch one.
- **The legacy finders are retired**: `--finder-algorithm` accepts only
  `sparse_rbf`; the `peak_local_max` and `thresholding` harvesters and their
  options (`--thresholding-*`, `--peak-local-max-*`,
  `--region-growth-minimum-intensity`, `--peak-minimum-pixels`) are gone,
  as are `--sparse-rbf-legacy`, `--sparse-rbf-auto-tune-alpha` and
  `--sparse-rbf-candidate-alphas`.
- **The convex-hull stage is retired** (`--hull-filter/--no-hull-filter`):
  the finder's own per-peak statistics (`peaks/deviance`,
  `peaks/residual_deviance`) carry the true-positive evidence. With it,
  **the finder no longer writes `peaks/intensity`** — the finder reports
  positions, shape and validation metrics only; intensity belongs to the
  integrator. Anything reading a finder amplitude needs a look
  (`static-mask` was the one in-tree consumer and now measures what it
  needs itself).
- **The integrator's `--alpha`/`--gamma` are removed.** They were tuned for
  the retired per-patch path and were no-ops in the matrix-free path
  (measured: bit-identical output with and without them). Admission is
  governed by `--matrix-free-fp-target` alone.

### Changed

- **`rbf-integrator` is renamed to `integrator`.** The old name remains as
  a deprecated alias that warns at runtime; move scripts to `integrator`.
- **Integrated intensities come from one nonnegative Poisson amplitude
  solve per image** on the finder-footprint-masked rate map
  (`--matrix-free`, now the only path). Amplitudes that end on the
  nonnegativity boundary are censored — dropped at export instead of
  merging as "0 ± sigma". An optional admission gate prices expected false
  admissions (`--matrix-free-fp-target`); it is **off by default**
  (censoring only), since it trades completeness for purity.
  `--matrix-free-profile` swaps the Gaussian atom for the finder's
  measured low-rank profile. `--static-mask-file` on the integrator feeds
  masked pixels into the solve as missing data.
- **Indexer restarts are seeded from one split key** rather than
  consecutive integer seeds, and the multi-GPU padding no longer competes
  in the best-of-N, so the answer is a function of `(seed, n_runs)` alone,
  independent of batch size and device count. **The same seed draws
  differently than in 1.3** — a pinned-seed workflow will reproduce itself,
  but not its 1.3 result.
- The per-run goniometer angle write-back is correct for files whose runs
  contain peakless images.
- The predictor inherits the indexer's refined angles, axes and per-run
  sample displacements instead of re-reading nominal geometry.

### Added

- **Per-run goniometer corrections** in the calibration: one scan-motor
  angle delta per run (`--refine-goniometer-per-run`,
  `--goniometer-per-run-bound-deg`) and one sample displacement per run
  (`--refine-goniometer-per-run-trans`,
  `--goniometer-per-run-trans-bound-meters`).
- **A Fourier-in-phi rocking of the effective crystal orientation**
  (`--refine-goniometer-harmonics`, with `--goniometer-harmonics-orders`,
  `--goniometer-harmonics-axes`, `--goniometer-harmonics-bound-deg`):
  periodic misorientation sampled by the scan, e.g. crystal-fixed
  anisotropic mosaicity. Crystallographic rotation orders cap at 6, so
  `--goniometer-harmonics-orders` never needs to exceed 1–6.
- **Goniometer axis-direction refinement** (`--refine-goniometer-axis-vector`,
  `--goniometer-axis-vector-bound-deg`) — the axis's tilt, not just its
  zero point — and per-axis translation masks
  (`--refine-goniometer-trans-axes`).
- **A positional indexing metric** (`--hkl-metric positional`, with
  `--radial-weight`/`--radial-weight-poly`/`--hkl-metric-floor`): candidate
  assignment warped by the per-peak Jacobian, for instruments whose
  tangential and radial residuals differ.
- **Anisotropic peak-shape modelling in the integrator**: a global crystal
  tensor with optional mosaicity (`--fit-mosaicity`,
  `--mosaicity-bound-mrad`), a radial-streak mosaic model
  (`--mosaicity-radial/--mosaicity-isotropic`), a spherical-sample
  hypothesis test (`--shape-spherical/--shape-ellipsoidal`), and shape-fit
  controls (`--shape-fit-min-snr`, `--shape-fit-normalized`).
- **Per-observation systematics columns in the MTZ export**
  (`mtz-exporter --predictions-file`/`--corrections-file`): SIGEFF and
  SNAPD from the predictor, DPHI/DTX/DTY/DTZ from the per-run corrections,
  for the scaling model to learn from.
- **Per-peak error tables**: `metrics --per_peak` writes
  `metrics/per_peak` (h, k, l, run, lambda, d_err, ang_err) into the
  indexed file, consumed by the `error_analysis` script.
- **The sigma bank can be sized for the data**
  (`--sparse-rbf-expected-peak-amplitude`,
  `--sparse-rbf-expected-background`): the values the finder's
  fragmentation warning prescribes are now settable.  An undersized
  bank reports bright peaks as clusters of narrower atoms, and since
  fragmentation sits on floating-point rounding, it also makes the
  peak set platform-dependent (measured: the same file gave 770 peaks
  on a CPU runner and 471 on a GPU until the bank was sized for it).
- The finder warns when one image's refinement support exceeds its exact
  single-chunk regime (more than 256 atoms), instead of silently switching
  to a one-sweep block-coordinate refinement.

### Known issues

- Peak finding is still not bit-reproducible on GPU (#13); the indexer,
  given the same peaks, now is — across repeat runs, batch sizes and
  device counts on one machine. Bit-identity across *different* GPU models
  or dependency builds is not promised anywhere: kernels round differently.
- `metrics` currently writes the `metrics/per_peak` table unconditionally
  and prints a stray `error per_peak` line (#63); the fix makes the table
  opt-in via `--per-peak`.

## [1.3.0]

The peak finder is rebuilt around a global convex solve: it fits all peaks at
once instead of subtracting them one at a time. Most of that is invisible, but
the defaults changed and one output column changed meaning, so scripts that pin
finder settings or read its HDF5 need a look.

- **`--finder-algorithm sparse_rbf` now runs the new basis-pursuit finder**,
  with sub-pixel position refinement. `--sparse-rbf-legacy` restores the old
  greedy one.
- **The basis now sizes itself**: `--sparse-rbf-max-sigma` and
  `--sparse-rbf-num-sigmas` default to unset and are measured from your first
  batch. A hand-set ceiling that is too small splits one peak into several, so
  prefer the default unless you know the width.
- **Sensitivity is now one number**: `--sparse-rbf-false-alarms-per-image`
  (default 0.1), the expected count of spurious peaks per image — lower it to
  demand more evidence. Related defaults changed: `--sparse-rbf-gamma`
  1.0 → 0.0, and `--sparse-rbf-loss` gaussian → poisson, since detector frames
  are photon counts.
- **`peaks/sigma` in the finder's output is now the fitted peak width in
  pixels, not the intensity sigma** — same name, different quantity, so check
  anything reading it. Two per-peak fit statistics join it: `peaks/deviance`
  (is this peak real?) and `peaks/residual_deviance` (is it fitted well?).
  `--no-hull-filter` reports every peak the finder proposes, for diagnosis.
- **Plots can be redrawn after the fact**: `finder-visualize` and
  `integrator-visualize` rebuild the unrolled-detector plots from existing
  HDF5, so a run can skip plotting and still be inspected later, at any `--dpi`.
- **Detector artifacts can be masked**: `static-mask` builds a mask of static
  structure (dead panels, glow, shadows) from your frames, applied with
  `finder --static-mask-file`; `sum-images` and `mask-visualize` support it.
- **`--multi-gpu` shards the finder and indexer across all visible GPUs.**
  Opt-in, because JAX claims memory on every visible device.

### Known issues

- Peak finding is not bit-reproducible on GPU: identical inputs can return
  slightly different peak sets. `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
  makes it reproducible at about 2.1x wall clock (#13).
