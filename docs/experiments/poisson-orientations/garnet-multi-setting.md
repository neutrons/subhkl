# Adding goniometer settings to the garnet count fit

This controlled comparison uses the same raw stack as the [single-still probes](garnet.md). The first 53 images of the merged stack were checked for exact equality with `solve-frames-2022.h5`. The available 21-setting scan runs from phi 70 to 129 degrees.

`MultiSettingModel` sums the pixel likelihoods across settings with one sample orientation U and one shared six-parameter detector correction. Setting s projects R_s U. Each setting has independent reflection intensities and each panel/exposure its own background scale. Orientation groups span all setting/hkl coefficients; these are not independent orientation fits or summed images from different angles. All valid counts, including zeros, enter the likelihood.

The comparison holds the model and initialization fixed: d_min 1.5, Gaussian sigma 6 pixels, penalty 1.2, 16-pixel sum bins, seven-bin median background, nominal geometry, and the first still's blind seed (0.57119 degrees from the reference). These results should not be compared directly with the earlier 8-pixel runs. The common outer budget is 1500 evaluations. A 300-iteration inner first pass is polished and checked as usual; unsuccessful evaluations retry the standard 2000-iteration budget before the outer fit is stopped. The stopping tolerance is unchanged.

| Settings | Runs | Orientation error | Radial correction | Outer status |
|---|---|---:|---:|---|
| 1 | 2022 | 0.90803 deg | 6.12642% | converged |
| 3 | 2022, 2042, 2062 | 0.61921 deg | 6.91719% | converged |
| 5 | 2022, 2032, 2042, 2052, 2062 | 0.46163 deg | 4.89906% | converged |

The five-setting fit improves on both the control and the initial seed; its distance correction is also closer to the approximate reference. The three-setting fit reduces the orientation drift relative to the one-setting control, but does not improve on the initial 0.57119-degree seed. The radial target (~5.3%) comes from a richer calibration family and is approximate. Numerical convergence still does not establish an accurate calibration.

The fixed-reference-orientation radial scans are diagnostic slices, not joint minima: they hold the other detector parameters at zero, and in three settings they favor zero distance correction over the positive values tested. Raw objective totals across different numbers of settings are not comparable.

## Unused-setting comparison

The saved geometry and orientation from each fit were evaluated on runs **2026, 2038, 2058**, none of which were used in any training subset. Only reflection intensities and panel background scales were refitted, using the same regularization and common group weights for all three evaluations. All nuisance fits converged.

| Training settings | Conditional penalized objective on unused settings | Reduction versus one setting |
|---|---:|---:|
| 1 | 5,976,993.89 | — |
| 3 | 4,786,648.15 | 19.92% |
| 5 | 4,381,140.20 | 26.70% |

This is conditional validation with nuisance refitting, not prediction of held-out intensities. It supports improved orientation/geometry transfer as more settings are fitted, without requiring spot detection. There is still measurable orientation error and substantial model residual; the experiment does not establish full calibration accuracy.

Machine-readable results: [one](garnet-multi-one.json), [three](garnet-multi-three.json), [five](garnet-multi-five.json), and [unused settings](garnet-multi-heldout.json).

## Restrictions

Important restrictions:

- This experiment uses nominal encoder rotations; it does not fit goniometer offsets, per-run translations, or per-panel geometry. More frames can expose inconsistency in those fixed assumptions.
- Detector roll about the beam remains fixed to match the one-setting model. The single-still gauge argument does **not** generally extend to settings whose rotations do not commute with rotation about the beam: R_s U must hold simultaneously, so a common lab beam-axis rotation cannot generally be absorbed in one U. A complete joint calibration should revisit this parameter.
- The reference orientation is used for scoring and the explicitly labeled fixed-orientation profiles. It is not used to initialize joint refinement. These probes do not test blind multi-setting proposal generation or multi-crystal discovery.
- This is an experimental numerical API and comparison script. The production `solve` CLI still rejects mixed settings; no unsupported output contract has been introduced.

Reproduce with:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu python \
  examples/poisson_orientation_multi_setting.py \
  --merged merged.h5 --first-setting solve-frames-2022.h5 \
  --seed blind.h5 --reference boot-truth.h5 \
  --settings 0 10 20 --max-evals 1500 --output three.json
```

`blind.h5` is the first-setting d_min 1.5 `--no-refine` result from the earlier probe. Use `--settings 0` for the control. JSON output includes all fitted parameters, encoder angles, orientation matrices, stationarity checks, and optimizer status.

For the five-setting fit use `--settings 0 5 10 15 20`. To score saved JSON fits on unused settings, use `--settings 2 8 18 --evaluate one.json three.json five.json`; the other input arguments remain the same.
