# CG4D garnet, run 2022

Follow-up to [the PR #80 reproduction](https://github.com/neutrons/subhkl/pull/80#issuecomment-5585755998), using the supplied `solve-frames-2022.h5` (53 banks, 217088 valid 8-pixel count bins) and lab-frame `boot-truth.h5`. The source dataset and multi-setting reference recipe are described in [subhkl-benchmarks](https://github.com/jglaser/subhkl-benchmarks/blob/main/benchmarks/cg4d/garnet.yaml). The reference orientation is external to these single-still fits. The approximate +24 mm / 5.3% radial correction is a comparison target from the existing calibration, not a planted exact solution for this six-parameter model.

All probes use CPU, penalty 1.2, Gaussian sigma 6 pixels, local median background, and no peak finder or count cutoff. Angles are minimized over the 24 proper cubic symmetries. Numerical convergence alone does not certify correct orientation or detector geometry.

The old solver reproduced a failure even with the true orientation supplied at d_min 1.5: residual 0.389 after 3000 iterations, objective 2672399.67. An intermediate projected-acceleration experiment incorrectly reported zero residual at a worse objective when backtracking made the step tiny. This exposed a second defect in measuring stationarity through the accepted line-search displacement. The final implementation checks a unit-step proximal residual at the accepted coefficients, compresses only exact background sufficient statistics, and polishes difficult active supports with L-BFGS and reduced Newton steps. Full inactive-group stationarity remains required.

The updated probes establish:

- **Blind default, d_min 3.2:** still selects the wrong orientation and correctly returns `no_orientation`. Solver changes cannot repair this insufficient dictionary.
- **Blind, d_min 1.5, support fit only:** converges; the single proposed orientation is **0.57119 degrees** from the reference. Objective **1920532.96577**, stationarity residual **4.86e-11**. The earlier hard failure had prevented inspecting this candidate.
- **Truth-seeded, d_min 3.2, joint refinement:** now converges instead of aborting on inner tolerance. Objective **2766728.94988 → 2634274.01805**, residual **3.92e-11**. However, the result moves **0.49204 degrees** from the reference and recovers only **2.89355%** radial correction. This is evidence against treating convergence with the sparse default dictionary as successful calibration.

- **Blind candidate, d_min 1.5, joint refinement:** objective **1920532.96577 → 669248.60937**, radial correction **6.30120%**, but orientation error increases from **0.57119 to 1.31360 degrees**. The inner fit passes its stationarity check (2.46e-5 against 2.95e-5); the outer solve exhausts its requested 600-evaluation budget after 790 evaluations (SciPy completes finite-difference/line-search batches). Status is `not_converged`, and no `sample/U` is exported. A better likelihood and roughly plausible radial scale do not establish accurate calibration.

The full [machine-readable results](garnet.json) include input SHA-256 hashes. The larger dictionary has 782 allowed hkl. Remaining work is to test background/profile adequacy and geometry/orientation identifiability against independent evidence; simply relaxing tolerances further would not address the orientation drift.

Initial and inner-refinement failures now produce diagnostic HDF5 output with candidates, objective, stationarity residual, and stopping threshold. Failed results still omit `sample/U` and exit nonzero.

Reproduce an individual case with:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu python \
  examples/poisson_orientation_garnet.py \
  --frames solve-frames-2022.h5 --reference boot-truth.h5 \
  --output-dir garnet-probes --case blind
```

Cases are `default`, `blind`, `seeded`, and `blind-refined`; the latter two use a 600-evaluation outer budget by default. The blind-refined probe recorded here reused the saved blind proposal as a bootstrap to avoid repeating the deterministic global search; no reference orientation or calibrated geometry was supplied to that fit. HDF5/JSON diagnostics are retained even when a case fails to converge. Raw facility counts are not committed.
