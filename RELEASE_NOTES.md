# Release Notes

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Global basis-pursuit peak finder (`MatrixFreeSparseRBFPeakFinder`) is now
  reachable from the command line and is the default for
  `--finder-algorithm sparse_rbf`.
- Continuous ("sliding") refinement of the selected support: after the convex
  solve chooses which atoms are present, their amplitude, position and width are
  re-fitted against the same Poisson objective, so recovered positions are
  sub-pixel rather than sub-grid. Reaches 0.03–0.17 px on an isolated synthetic
  peak. Controlled by `refine_positions`, default on.
- Optional rejection of atoms whose fitted width reaches the edge of the sigma
  bank, via `reject_boundary_sigma` (default **off**) and `boundary_sigma_frac`.
  Such an atom is the solver asking for a wider basis than it was given, which
  can mean unmodelled smooth background — or simply that `max_sigma` is too
  small, in which case real peaks saturate the bank too. On a real MANDI scan,
  where peak widths are ~34 px against a `max_sigma` of 5, it removed 87% of
  genuine detections, so it is a diagnostic to enable once the bank is known to
  be wide enough rather than a default.
- The count of atoms rejected that way is recorded on `n_boundary_rejected` and
  reported under `show_steps`, rather than being discarded silently.
- Multiplicity correction to the significance threshold. Solving globally tests
  every (pixel, scale) coefficient at once, so `alpha` is now floored at
  `sqrt(2 log N_k)` with `N_k` the resolution-element count for that scale.
- `docs/matrix_free_theory.md`, recording the results behind these changes with
  proofs and the measurements that establish them.
- `--sparse-rbf-legacy`, to opt back out to the greedy finder.
- `alpha` may now be left as `None`, which is the new default. The significance
  threshold is then derived from the data: the level at which the expected
  number of false detections over the whole image is O(1), from the maximum of a
  smooth Gaussian field over the resolution elements at each scale. Passing a
  number keeps it as a lower bound on significance, so an explicit request can be
  stricter than false-alarm control requires but not weaker. Exposed as
  `effective_alpha(height, width)`.

### Changed

- **Default `gamma` is now 0.5, was 1.0** (2.0 in the orchestrator). `gamma=1`
  is the value at which the penalty per unit flux becomes independent of scale,
  so one broad atom and a mass-preserving spread of narrower ones have the same
  cost and the same predicted image; the minimiser is then not unique in the
  scale coordinate and the fit breaks the tie towards splitting. See
  `docs/matrix_free_theory.md` Theorem 1. The legacy finder keeps its historical
  default of 2.0 so that it still reproduces what it always did.
- **The debiasing phase is removed from the finder** (99 lines). It existed to
  strip L1 shrinkage from amplitudes, and nothing downstream reads the finder's
  amplitude: the orchestrator reduces the peak list to `(row, column)` before
  handing it to the workers, and intensity is measured later by the integrator at
  known positions, where it is a well-posed problem and where its own debiasing
  still lives. It was also not free — dropping the penalty drops the only thing
  suppressing what the model cannot explain, so an unpenalised refit absorbed a
  mis-estimated background into the peaks, in proportion to how sparse the
  support was. Removing it settled two flaky tests.
- **Default `--sparse-rbf-loss` is now `poisson`, was `gaussian`.** Detector
  frames are photon counts, so Poisson is the matching noise model; the Gaussian
  likelihood assumes one constant variance across the frame, which is wrong
  wherever the background varies — exactly the regime the spatially varying
  threshold exists to handle. The legacy finder's API-level fallback is
  unchanged so direct callers still get what they always did.
- **Default `alpha` is now `None`, was 4.0.** The right threshold depends on how
  many coefficients are being tested, so it depends on image size, which a
  constant cannot express. The derived values run from 3.60 on a 64x64 padded
  crop to 5.44 on a 4096x4096 frame, so the old constant was about right for a
  mid-size image but ~26% below the floor on a full detector — under-thresholded
  exactly where it matters. The `sigma**gamma` shape of the threshold is kept,
  since that is what sets the merge/split balance; deriving the level from the
  floor *alone* flattens that shape into the over-merging regime and loses weak
  peaks in the tails of strong ones, which was measured and rejected.
- Peak positions are now reported from the coefficient-weighted centroid of the
  raw coefficients rather than a log-parabola on the smoothed map, which was
  dragging centres toward neighbouring peaks.
- Detection smoothing in peak extraction is capped at the finest scale in the
  bank, so it no longer merges peaks a few pixels apart.
- Peaks are ranked and truncated to the reporting limit *after* out-of-bounds
  ones are discarded, not before, so maxima in the replicated edge padding can
  no longer consume the budget and displace real interior peaks.

### Fixed

- Conjugate gradients was being given a non-symmetric operator in both the
  semi-smooth Newton solve and the debiasing solve, because the Jacobi scaling
  was multiplied into the operator instead of passed as a preconditioner.
  Measured asymmetry of the operators as written was ~50%. The debiasing solve
  diverged to NaN as a result, and because a non-finite iterate was allowed to
  propagate, whole images returned no peaks at all.
- The L1 threshold used the across-channel maximum of the Hessian diagonal as
  the coefficient variance. The noise on a channel's coefficient is set by that
  channel's own curvature; using the maximum understated it by ~11x for the
  narrowest basis in a typical bank, so fine scales were thresholded far too
  weakly and fitted noise.
- The prox-gradient step size was derived from the diagonal of `A^T W A` rather
  than a bound on its largest eigenvalue. For a typical bank the two differ by a
  factor of 419, so every step overshot and the line search collapsed onto its
  smallest permitted step, leaving the solve stalled far from the optimum.
- The line search accepted a step that increased the objective when its
  backtracking budget ran out, instead of rejecting it.
- The debiasing phase diverged on large, heavily overlapping supports. It drops
  the L1 term, which is the only thing holding the near-null-space directions of
  the active set in check, so on a near-singular support CG returned a direction
  it had not solved for and the unguarded Newton step ran away. Measured on two
  overlapping broad peaks, the likelihood got *worse* on every iteration and the
  rms residual went from 3.3 to 60 — an order of magnitude worse than reporting
  no peaks at all, while the L1 phase that preceded it had fitted the data well.
  Debiasing now backtracks and refuses a step that does not improve the
  likelihood, so it can only ever improve on the L1 solution it starts from.
- The outer convergence test measured the raw Newton direction rather than the
  step actually taken, so it never triggered.
- The sliding refinement was a silent no-op. A reporting slot that matched
  nothing has zero flux in every channel, so its fitted width came out as zero
  and its amplitude as an infinity; the refinement neutralised unused slots by
  multiplying them by a zero mask, and `inf * 0` is `NaN`. One such row made the
  objective and every gradient `NaN`, and the guard against non-finite gradients
  then turned the whole step into a no-op instead of reporting anything. Whether
  it happened at all depended on whether a slot's flux was *exactly* zero, so
  refinement silently ran on some images and not others. Widths are now floored
  at the finest basis and unused slots are replaced rather than multiplied out.
  On the two-peak multiscale case the narrow peak's fitted width goes from 1.587
  to 0.924 against a true 1.0, its position to within 0.03 px, and its amplitude
  from 49.2 to 134.9 against a true 120.
- `subhkl finder --finder-algorithm sparse_rbf` returned no peaks on its own
  integration test. It now passes.
- Two heavily overlapping broad peaks at 2.67 sigma of separation were
  reconstructed with a spurious atom at the composite centre and up to 2.6x the
  true flux (`test_overlapping_ghost_center_shift_failure`). This was the
  debiasing divergence above, not a resolution limit: the condition number of
  the true pair is 1.40. The test now passes.
- Removed `src/subhkl/:q`, a stray editor buffer saved under the wrong name, and
  added editor swap files to `.gitignore` so that cannot recur.

### Notes

- `test_poisson_vs_gaussian_sparse_flux` is replaced by
  `test_gaussian_loss_path_finds_peaks`. The old test compared *recovered flux*
  between the two losses, which is a quantity the finder no longer promises and
  which no consumer reads; it was also the only thing keeping the debiasing phase
  alive. The Gaussian likelihood still needs covering, since it is the CLI
  default for `--sparse-rbf-loss`, so the replacement asserts that both losses
  detect and localise rather than that either recovers a flux.
- The finder suite is currently stable: 20 of 20 on three consecutive whole-file
  runs, and 6 of 6 in isolation for each of the three tests that used to flap.
  That is after the refinement no-op above was fixed; before it, a clean run was
  roughly 55% likely.

### Deprecated

- `SparseRBFPeakFinder`, the greedy matching-pursuit finder, is superseded by
  the basis-pursuit finder and reachable only via `--sparse-rbf-legacy`. It
  cannot be removed yet: `SparseLaueIntegrator` still inherits from it to obtain
  the peak model. Breaking that inheritance, by extracting the shared model into
  its own module, is the prerequisite.

### Known issues

- Peak finding is not reproducible run to run on GPU. Identical input and flags
  return one of a small number of distinct peak sets, because reductions are
  not deterministic; `XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes it
  bit-reproducible at about 2.1x wall clock. Tracked in #13. Tests whose
  assertions sit near a threshold will flap accordingly.
- `test_poisson_local_variance_suppression` and
  `test_poisson_subpatch_variance_suppression` each pass in isolation (six runs
  out of six) but fail roughly one run in three when the whole file is run, so
  their outcome depends on what executed before them. That is the
  non-determinism above expressing itself through JIT and GPU state rather than
  through the test inputs.
- The global `Deviance/DoF` statistic is too diluted to detect a locally bad
  fit: on a case where the reported model was worse than reporting no peaks at
  all it read 1.21, against 1.12 for a well-behaved control. Evaluated over a
  peak's own footprint the same statistic gave 26.9 against 1.04. A per-peak
  goodness-of-fit check would therefore catch what the global one misses, and is
  not yet wired to anything.
- The morphological background estimator under-fits smooth extended structure at
  its centre — by about 21% on a diffuse halo — leaving a broad positive
  residual that is reported as a broad peak.
- On a real MANDI scan (160 frames, 256×256) the finder returns 466 peaks
  against the greedy finder's 675, with comparable widths. Whether that shortfall
  matters is a question for the benchmark suite, not the unit tests: indexing
  yield and angular residual are the measures that decide it, and this branch has
  not been run through them.
- The unit tests do not resemble the data. They build float32 images, densely
  populated, with backgrounds of 10–50 counts; a real frame is int64, 37% zeros,
  with a mean of 0.64 counts. An integer-convolution failure that killed every
  real run was invisible to all 20 of them.
