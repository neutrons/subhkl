# Release Notes

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.3]

### Changed

- **Time-of-flight integration fits each detector image once, in position and time (`tof_joint`, `integrator --tof-fit`)**:
  - With events present, all reflections of an image are now fitted together
    as spot profile × measured time profile on a background taken from the
    voxels no reflection touches. Reflections that share pixels are told apart
    by when their neutrons arrive. Previously each reflection was fitted on its
    own tile, summed over the time gate, with up to seven neighbours; two
    reflections on the same pixels and inside one another's gate were then
    indistinguishable and were censored.
  - The reported intensity is the reflection's whole flux, not the 95% inside
    a gate.
  - `--tof-max-variance-inflation` keeps its meaning. Where neighbours inflate
    a variance beyond it, the image is refitted without the directions the data
    do not determine, and only reflections with a share in those directions are
    censored.
  - `--tof-fit gated` selects the earlier method. `--tof-sideband-factor`,
    `--tof-sideband-measure` and `--tof-amplitude-ridge` apply only to it.
  - On MANDI IPTS-30763 runs 11324–11338 (1.89 million predictions to 1.41 Å):
    half-dataset CC of the `careless mono` merge rises from 0.949 to 0.975 at
    4.0–3.0 Å, 0.803 to 0.884 at 3.0–2.5 Å, 0.628 to 0.774 at 2.5–2.2 Å and
    0.352 to 0.447 at 2.2–2.0 Å; it is 0.968 against 0.982 below 4 Å. 1.85
    million reflections are fitted (1.74 million before) and none is censored
    as unresolved. The overall CC½ of the merge to 1.41 Å falls from 0.900 to
    0.759, because the added observations lie beyond 2 Å where this crystal
    does not diffract; cut the merge at the resolution the shells support. The
    integrator takes about three minutes on one GPU.
  - The fit relies on the spot shape: with the nominal one-pixel spot
    (no `--anisotropic` shape fit) the 2.5–2.2 Å shell drops to 0.19.
### Fixed

- **Galaxy History derived-HDF5 (`...integrated.h5` / `...solve.h5`) background frame hydration completion (`rainbow_zones`, `nova_rainbow_view`, `nova_app`)**:
  - Preserved a stable `client_job_id` across `submit_galaxy_stage`'s
    placeholder `nova-<timestamp>` ID -> Galaxy/Pulsar `job_id` transition so
    `startHydrationJobPoll` no longer aborts polling early while leaving the
    Step 1 stopwatch running indefinitely.
  - Kept hydration `work_dir` aligned with staged derived datasets (including
    dot-suffixed `.integrated.h5` / `.solve.h5` names), surfaced live runner
    host status during hydration, and cleared completed `hydration_job`
    descriptors from cached `/api/payload` responses.

## [1.7.2]

### Fixed

- **NDIP/Galaxy Stage 4 (`imaginex_subhkl_merge`) Careless output handling and UI completion (`nova_rainbow_view`, `rainbow_zones`, `nova_app`)**:
  - Fixed a post-job `h5py.File` failure (`Unable to synchronously open file
    (file signature not found)`) when `imaginex_subhkl_merge` completed on
    Galaxy/Pulsar: `h5_outs` is now skipped completely on `stage_id == "merge"`,
    and Galaxy output descriptors are normalized and reconciled against
    `/api/jobs/{id}` without mislabeling non-HDF5 datasets (`metrics_json`,
    `merge_table_csv`, `merge_table_html`, `*_mtz`) as `h5`.
  - Materialized a local writable `integrated.h5` before stamping Careless
    merge metrics and per-observation residuals so read-only shared `/scratch`
    symlinks are never modified in place, supported both separate
    `peaks/{h,k,l}` and packed `peaks/hkl` arrays during residual stamping, and
    added state rollback guards if payload reload fails.

## [1.7.1]

### Added

- **Reversible two-phase session Exit (`⏻`) with deterministic shared/staging cleanup and 80s pixel-font exit modal (`rainbow_zones`, `arcade_exit_quips`, `nova_rainbow_view`, `nova_app`)**:
  - Added a compact top-bar `⏻` exit button that opens a minimalist gold-on-dark
    `8×8` pixel-font (`SubhklArcade`, offline embedded TrueType) modal with 230
    cheerful arcade/beamline sign-off messages, paired with reversible **`GO
    BACK`** and crimson **`CLOSE TAB`** actions (and clean sans-serif typography
    in Pro mode).
  - Implemented a two-phase `/api/pipeline/exit` lifecycle (`prepare` on modal
    open, `confirm` on **`CLOSE TAB`** / tab unload):
  - In **Galaxy/NDIP** mode, `prepare` immediately sweeps Tier 1 staging and
    Tier 2 shared intermediates (locally or via `imaginex_subhkl_cleanup_shared`)
    while keeping the Trame/HTTP server alive so **`GO BACK`** restores the
    previous step and hydrates reduced detector frames on the fly from Galaxy
    History.
  - In **Local** mode, intermediate file deletion is deferred until `confirm`
    or process finalization so **`GO BACK`** preserves local state intact.
- **Execution host reporting for Careless variational merge (`nova_rainbow_view`, `rainbow_zones`)**:
  - Surfaced the active execution host (`runner_host`, resolved from local
    container headers, `[pulsar] Runner host:` console output, or Galaxy
    `job_metrics`) in `#cell-summary`, the live `[4. MERGE • <HOST>]` HUD status
    bar, terminal logs, and `careless_metrics.json`.

### Fixed

- **Post-integration Wulff $d_{\min}$ filtering in `Integ` / `All-Runs` mode and Careless `--dmin` cutoff forwarding (`rainbow_zones`, `nova_rainbow_view`, `nova_app`)**:
  - Preserved the selected peak source (`Integ` vs. `Pred`) and multi-run
    `All-Runs` peak pools when adjusting $d_{\min}$ or $[\lambda_{\min},
    \lambda_{\max}]$ after integration: `Integ` mode filters integrated
    reflections in place by $d \ge d_{\min}$ without switching to predicted
    peaks, starving runs `1..N-1`, or dropping multi-run `frame_only` pools.
  - Forwarded the active UI $d_{\min}$ as `--dmin` to both local Careless
    containers (with pre-flight empty-cutoff validation) and Galaxy
    `imaginex_subhkl_merge`, and stamped full `/careless/{asu_hkl,merged_i,merged_sigi}`
    datasets so Bayesian-extrapolated harmonics ($d_{\min} <
    d_{\min,\text{integrated}}$) retain merged structure factors.
- **Read-only UI `SUBHKL_SHARED_DIR` mounts, worker-delegated shared cleanup, and non-ORNL Galaxy hydration (`shared_storage`, `nova_rainbow_view`)**:
  - Relaxed UI `require_shared_dir_for_galaxy_mode()` to require read-only
    `R_OK | X_OK` permissions (retaining `W_OK | X_OK` on worker writers),
    dispatching `imaginex_subhkl_cleanup_shared` on a read-write worker whenever
    the UI container mounts `SUBHKL_SHARED_DIR` read-only.
  - Allowed canonical `/SNS` and `/HFIR` NeXus `source_files` paths to trigger
    remote Galaxy Reduce hydration even when `/SNS` and `/HFIR` are not mounted
    on the UI or CI host.

## [1.7.0]

### Added

- **Three-tier shared storage orchestration for Galaxy/Pulsar runner mode with NFS race-free UUID sideband publication (`shared_storage`, `parser`, `nova_rainbow_view`, `galaxy`)**:
  - Separated intermediate and persistent data across three storage tiers so
    interactive `subhkl-nova-app` sessions release GPUs to job runners without
    moving multi-GB HDF5 files through Galaxy/Pulsar object-storage caches:
  - **Tier 1 (`SUBHKL_STAGING_DIR` / `TMPDIR`)**: fast node-local NVMe scratch
    for step-local work (`reduce-batch` per-run files, local HDF5 assembly,
    and Careless scratch).
  - **Tier 2 (`SUBHKL_SHARED_DIR`, mandatory when `SUBHKL_EXEC_MODE=galaxy`)**:
    network filesystem exchanging compact `subhkl_shared_pointer/v1` JSON
    sideband pointers (`--shared-pointer-out`) for large cross-step HDF5
    artifacts (`merged.h5`, `predictor.h5`, `integrated.h5`), with automatic
    `/proc/self/mountinfo` canonical network-path translation across
    heterogeneous runner mounts (e.g. `/scratch` on `imaginex-{1,2}` vs.
    `/gpfs/neutronsfs/scratch` on `analysis.sns.gov`) and deterministic
    cleanup (`subhkl cleanup-shared` and app exit hooks).
  - **Tier 3 (Galaxy Object Storage / History)**: persists thin crystallography
    artifacts (`solve.h5`, `reflections.mtz`, `merged.mtz`, `xval.mtz`,
    `careless_metrics.json`, `merge_table.{csv,html}`), reconstructing reduced
    detector data on the fly when restarting from a History `solve.h5`.
  - Hardened Tier 2 sideband publication for lockless/non-atomic NFS mounts:
    allocates fresh 128-bit UUIDv4 filenames (`subhkl_shared_<32hex>_<artifact>`)
    opened via `O_WRONLY | O_CREAT | O_EXCL` with zero pre-creation `stat`
    probes to guarantee cold client dcache/attribute/page-cache misses, flushes
    and `os.fsync`s + closes the HDF5 file (and parent directory) strictly
    **before** writing and `os.fsync`ing the JSON pointer, exports
    `HDF5_USE_FILE_LOCKING=FALSE`, and keeps published shared files
    write-once/read-only.
- **Laue nodal bullseyes (`a⋆`, `b⋆`, `c⋆` & zone hubs), `[uvw]` arc hover badges, and `±1`/`±2` Higher-Order Laue Zones (`rainbow_zones`)**:
  - Rendered uniform-radius Laue nodal bullseye markers at zone-axis
    intersections (with superscript `a⋆`, `b⋆`, `c⋆` labels on principal
    reciprocal axes, dimmed markers in inter-panel gaps within 25% of a panel
    width, and intersection-count hover badges), calibrated flat-panel hub
    raycasting on cylindrical bank assemblies (e.g. CG4D), interactive `[uvw]`
    zone-axis hover badges, and optional `±1` / `±2` HOLZ curves with live
    predicted reflection counts.
- **3D ToF reciprocal-space `Δθ`, warm divergent `I/σ` colorscale, and direct numerical slider inputs (`rainbow_zones`)**:
  - Added directional 3D ToF-gated reciprocal-space `Δθ` (`arccos` unit-vector
    placement) with a dedicated `/api/tof_delta_theta` route and $10\times$
    faster frame switching.
  - Added a warm quantile-normalized divergent `I/σ` colorscale (with matching
    gold `Integ` button styling), median/outlier-robust `ΔI/σ` colorbar bounds,
    click-to-edit `0.01 Å` numerical inputs on the $[\lambda_{\min},
    \lambda_{\max}]$ and $d_{\min}$ sliders (paired with `0.05 Å` slider drag
    steps on `#slider-dmin`), fixed-width `1-Frame`/`All-Runs` toggle geometry,
    and corner `^`/`v` magnification carets on the Wulff pane.
- **Unlocked `0.50 Å` high-resolution $d_{\min}$ floor on large macromolecular unit cells via Ewald-shell rod slicing & viewport-frustum LOD (`rainbow_zones`, `crystallography`)**:
  - Removed the `2,000,000` full-sphere lattice-point UI $d_{\min}$ ceiling
    (which clamped `P 61 2 2`, $80\times 80\times 238\text{ Å}$, at `1.41 Å`),
    allowing `#slider-dmin` to reach `0.50 Å` across all unit-cell volumes.
  - Added closed-form Ewald-shell $(h, k)$ rod slicing in both the browser
    (`raycastRunDirectEwaldShell`) and Python (`generate_ewald_reflections`) plus
    slabbed quadratic rod bounds in `generate_reflections` so high-resolution
    macromolecular shells ($40\text{M}+$ reciprocal lattice points) evaluate
    directly inside the active $[\lambda_{\min}, \lambda_{\max}]$ Ewald band
    without multi-GB 3D `meshgrid` allocations.
  - Upgraded Detector and Wulff-sphere LOD thinning with two-tier
    viewport-frustum high-resolution streaming (up to `48×` detector zoom) and
    gameplay-style canonical dyadic Spherical BVHs (`buildDetectorSphericalBVH`,
    `buildWulffSphericalBVH`) paired with solid-angle stratified Ewald sampling
    (`_stratified_spherical_subsample` on server, `selectSpatiallyStratifiedOrder`
    on client): outer resolution shells and back-scattering banks populate
    immediately at `1×` zoom as `#slider-dmin` decreases while screen peak
    density stays bounded (~18 px cells), and zooming in subdivides cells
    100% pop-free down to full doublet/harmonic separation (`0.50 Å`).
- **GPU multi-panel peak prediction & fixed-chunk $O(n)$ TOF-gated integration for large macromolecular unit cells (`physics`, `tof_gate`, `detector`)**:
  - Replaced per-bank `ProcessPoolExecutor` peak prediction (which regenerated
    millions of `(h, k, l)` tuples inside every worker process, taking ~3–4 min
    on `P 61 2 2` at $d_{\min} = 1.0\text{ Å}$) with `predict_peaks_batch_gpu`:
    generates candidate `(h, k, l)` once, filters Ewald shells and batches flat
    or curved panel intersections on GPU using static-shape `Precision.HIGHEST`
    kernels (~1.6–2.1 s across 41–164 banks), and finalizes surviving hits in
    float64 for exact parity with `predict_single_bank`.
  - Eliminated per-bank XLA JIT recompilation stalls and $O(n^2)$ host overhead
    in `integrate_reflections_tof_gated` by evaluating `_scatter_events_impl`,
    `_box_sums_impl`, `_tiles_impl`, `_snap_positions_impl`, and
    `_solve_tiles` in fixed-size chunks, freeing intermediate 3D prefix cubes
    immediately after sideband box sums, replacing dense $n \times n$ neighbor
    broadcasts and `np.nanmedian` with `scipy.spatial.cKDTree` sparse Chebyshev
    queries + vectorized masked row medians, and prefetching next-bank events in
    a background thread.

### Fixed

- **Time-of-flight sideband background biased by the shape of the spectrum (`tof_gate`, `detector`)**:
  - The gated integrator scaled each reflection's sideband background to its
    gate by the length of the intervals in time, which assumes a background
    rate flat in time. The incident spectrum is curved and falls off steeply at
    the band edges, so weak reflections carried an additive offset of up to
    about ±1.5σ depending on wavelength, scattering angle and bank. The offset
    repeats between measurements and showed up as CC½ where there is no Bragg
    signal.
  - Intervals are now measured in each image's own cumulative time spectrum
    (`--tof-sideband-measure spectrum`, the default, folded directly into the
    fused on-GPU 4D slab prefix sums alongside consistent detector border/pixel
    mask enforcement across prediction, gating, and visualization), so the
    background is taken as proportional to what the image shows at each time;
    images with too few events for a spectrum fall back to plain time.
    `--tof-sideband-measure time` restores the previous behaviour.
  - Check for this kind of bias with an off-peak control: integrate with the
    gate moved off the reflections (e.g. `--tof-gate=0.06,0.11`) and merge; any
    CC½ that survives is not Bragg signal.

- **Unresolvable overlaps reported with meaningless intensities (`tof_gate`, `matrix_free`)**:
  - Where reflections sit closer together than the spot width inside one time
    window (around the direction of a long cell axis, e.g. within ~10° of c*
    for c = 238 Å), the joint amplitude fit is singular to working precision.
    The reported σ then saturated at the fixed regularisation constant (1000
    photons) while the signed estimate left it by orders of magnitude, giving
    rows with |I/σ| in the hundreds; merged, their reproducible values added
    CC½ of 0.2–0.4 in shells without signal.
  - A gated reflection is now censored (σ_I = 0, dropped from the export) when
    overlapping neighbours inflate the variance of its intensity more than
    1000-fold relative to an isolated reflection
    (`--tof-max-variance-inflation`; 0 keeps every row). Only the sum of such a
    group is determined, and a per-reflection table cannot carry it.
  - `--tof-amplitude-ridge counts` sets the regularisation of the joint fit to
    the gross counts under each reflection's own profile instead of a fixed
    1000 photons. It gives unresolved reflections a meaningful σ, but on a
    crowded lattice it also shrinks weak, partly overlapped reflections, so
    `fixed` stays the default.

- **Detector view sideways drift when zooming out (`rainbow_zones`)**:
  - Anchored `detCanvas` wheel zoom-in at the cursor and zoom-out horizontally at
    `viewCenterX` (with periodic $\pm 360^\circ$ pan wrapping and smooth vertical
    recentering toward `1×`), eliminating unexpected sideways azimuth shifts and
    coupled Wulff rotations when zooming out.
- **Rhombohedral hexagonal (`:H`) vs. primitive rhombohedral (`:R`) space-group settings in cell refinement and UI**:
  - `gemmi.SpaceGroup` reports `ext='H', centring_type()='R'` for
    hexagonal-axes `R` space groups (`R 3`, `R 3:H`, `R -3 m`) and
    `ext='R', centring_type()='P'` for primitive rhombohedral-axes settings
    (`R 3:R`, `R -3 m :R`).
  - Fixed `optimization.get_lattice_system` and
    `scan_band._cell_symmetry_spec` to inspect `sg.ext == 'R'` rather than
    `centering == 'R'`, preserving hexagonal cell constraints
    ($a=b\neq c,\,90^\circ,90^\circ,120^\circ$, 3-point `R` centering) on
    default/`:H` `R` groups and primitive rhombohedral angle/length constraints
    ($a=b=c,\,\alpha=\beta=\gamma$, `P` centering) on `:R` groups.
  - Prevented `trigonal_r` from overwriting bare `R` symbols (`"R 3"`,
    `"R -3 m"`, `"R 3 2"`) in Rainbow Zones' client-side `SG_SYMMETRY_LOOKUP`.
- **Generalized multi-run unit-cell, detector-geometry, and per-run crystal-slip refinement (`scan_band`, `scan_refine`)**:
  - Included unwarped `search/proposal_orientations` in the bootstrap candidate
    pool and replaced the `1.5%` radial `shared_geometry` cliff with a direct
    Ewald + per-run zone-funnel competition against nominal detector geometry
    (refreshing EM pixel responsibilities when nominal wins while preserving
    genuine uncalibrated detector shifts).
  - Kept pure nominal geometry (`s_init = 0`) as a distinct Stage 0 trial,
    stopped re-injecting rejected `shared_geometry`, and freed unit-cell
    ratios/angles ($\pm 3.5\%$) from Stage 0 onward on multi-setting scans so
    anisotropic cell errors (`0.5–1.0 Å`) are not absorbed into fake detector
    warps.
  - Replaced the single-pass $0.15^\circ$-clamped Stage 3 post-cell orientation
    polish with a 2-pass alternating `(cell <-> 0.6° U + zone-funnel + per-run R)`
    polish so monoclinic/anisotropic cell updates unwind coupled shear
    rotations.
  - Re-anchored `(R_r, U)` to `anchor_run` (`ΔΘ_anchor = 0°`) across Ewald and
    Stage 4 ICP polishes and regularized $\chi = 0$ gimbal-lock 3-axis
    goniometer decomposition across both axis handedness conventions.

## [1.6.6]

### Added

- Cell- and $d_{\min}$-aware automatic `Solve` band binning (`binning=4` on
  $\le 256\times 256$ panels for dense long-axis cells), `b2=2` (`128×128`)
  Ewald excess maps, and a `<option value="2">2</option>` binning option in the
  Rainbow Zones `Params` drawer.
- Live client-side HUD `Δθ` (`angular_rms_deg`) recomputation in Rainbow Zones
  when switching scan frames or dragging wavelength, $d_{\min}$, or manual
  orientation controls.

### Fixed

- **Long-axis joint scan refinement, per-run crystal slip, and Stage 4 ICP aliasing (`scan_refine`, `scan_band`)**:
  - Computed exact metric-tensor unit-cell volume and Bravais centering
    multiplicity in `_ewald_d_min`, applied the Laue kinematic lower bound
    $q_{\min} = 2\sin\theta_{\min}/\lambda_{\max}$, and stratified large
    Ewald reflection pools across $|G|$ shells instead of truncating to
    overlapping inner rings.
  - Added per-run Gaussian zone-band pre-alignment (`_zone_funnel_seed_per_run`,
    $\pm 3.8^\circ$) plus fine-stepped `polish_per_run` to capture genuine
    inter-batch crystal slips into `R_refined[r]` without rigid-orientation
    overwrites, direct Stage 3 Ewald unit-cell axis-ratio polish, and
    off-singularity `Y-Z-Y` gimbal-lock seeds in `_decompose_R_to_angles`.
  - Adapted Stage 4 per-run ICP matching radii to median nearest-neighbor
    predicted spot spacing with unambiguous second-neighbor filtering
    (`d2_best < 0.55 * d2_second`) and an Ewald non-degradation guard.
- **Missing (`"-"`) and neighbor-contaminated HUD `Δθ` on multi-frame datasets (`rainbow_zones`)**:
  - Resolved `bank_ids` before `banks` in `_load_cached_frame_bank_images` (fixing
    `Δθ = "-"` on frames $\ge 1$ and bank-scrambling on non-contiguous detector
    bank lists) and enforced Voronoi nearest-predicted-reflection ownership in
    both Python and client-side JS spot-centroid error estimation.

## [1.6.5]

### Added

- Promoted the wavelength interval ($\lambda_{\min}$–$\lambda_{\max}$) to a
  dedicated dual-knob (`[` and `]`) sidebar range slider above $d_{\min}$ in
  Rainbow Zones and simplified the `Params` drawer to `Unit cell / Params`.

### Fixed

- Fixed `ReferenceError: getActiveWulffSourceArray is not defined` in
  `updatePeakInspector` when hovering detector panels.
- Wired `stage_derived_h5_in_work_dir` and background `Reduce` frame hydration
  into the Trame `MainApp` / `SubhklGalaxyModel.refresh_payload` path (plus
  `.reduce.h5` → `.nxs.h5` provenance recovery) so reloading `solve.h5` or
  `integrated.h5` from NDIP Galaxy History regenerates detector images.

## [1.6.4]

### Added

- **Cross-instance NDIP Galaxy History reuse, `integrated.h5` persistence, and on-the-fly `Reduce` frame hydration**:
  - Unifies Galaxy History discovery across the shared `"IMAGINE-X SubHKL"`
    datastore history and the interactive tool's launch `HISTORY_ID` (with
    automatic cache invalidation on upload/touch and descriptive
    `<tag>_solve.h5` / `<tag>_integrated.h5` dataset names) so analysis
    artifacts from previous app instances are immediately visible in local
    execution mode.
  - Persists compact `integrated.h5` and `predictor.h5` artifacts to NDIP
    Galaxy History during Stage 3 (`Integrate`) and re-uploads Careless-stamped
    `integrated.h5` after Stage 4 (`Merge`), automatically reconstructing
    `reflections.mtz` from `integrated.h5` when restarting Stage 4 in a fresh
    instance.
  - Embeds raw NeXus provenance (`source_files`, `files`, `file_offsets`,
    wavelength, and cell metadata) in `solve.h5` and `integrated.h5` (with
    `/SNS` and `/HFIR` run-range fallback for pre-existing NDIP datasets) and
    automatically triggers a non-destructive background `Reduce` hydration job
    whenever derived data is loaded without a hit in the instance's local
    staging cache, hot-reloading detector images and TOF histograms on
    completion.

## [1.6.3]

### Changed

- **Local-staging GPU execution in `subhkl-nova-app` with Galaxy Careless merge**:
  - When running in local-staging mode (`SUBHKL_EXEC_MODE=local`, or
    automatically inside an NDIP interactive GPU tool session with `EP_PATH`
    and `/SNS` / `/HFIR` mounted), `Reduce`, `Solve`, and `Integrate` execute
    directly inside the app's local NVMe staging directory
    (`SUBHKL_STAGING_DIR` / `TMPDIR`) instead of round-tripping multi-GB
    `merged.h5` / `integrated.h5` datasets through NDIP Rucio/Ceph object
    storage.
  - Uploads only compact (`<= 50 MiB`) metadata and crystallography
    deliverables (`solve.h5`, `reflections.mtz`, `merged.mtz`, `xval.mtz`,
    `careless_metrics.json`, `merge_table.csv`, `merge_table.html`) to NDIP
    Galaxy History, deduplicating `reflections.mtz` by `(size, mtime_ns)`.
  - Routes Stage 4 (`Merge` / Careless) through NDIP Galaxy
    (`imaginex_subhkl_merge`) whenever `careless` is not installed locally in
    the `subhkl` container, exchanging only compact `.mtz` files.
  - Ensures Pulsar-friendly world-writable permissions (`0777` directories,
    `0666` files) inside `SUBHKL_STAGING_DIR` so host `pulsar` job cleanup
    never encounters `PermissionError`.

## [1.6.2]

### Fixed

- Fixed a ~29-minute post-upload hang on remote Galaxy instances (e.g.
  `ndip-test.ornl.gov`) after `reduce` produces multi-GB TOF `merged.h5`
  datasets (`6 × 240 s` silent proxy-buffering read timeouts when downloading
  embedded `events/tof`): `subhkl-nova-app` now syncs only the 2D `images`
  stack and metadata (`~1–2%` of file size) via chunked HTTP `Range` requests
  (`X-Accel-Buffering: no`) while keeping the full `events/tof` dataset in
  Galaxy for Pulsar GPU `solve` / `integrate` jobs, emits immediate connection
  and retry status lines, and fails fast on scratch `ENOSPC`.

## [1.6.1]

### Added

- Monochromatic (`careless mono`) vs. polychromatic (`careless poly`) merge
  routing for time-of-flight gated data:
  - Added an explicit `merge_mode` (`mono` / `poly`) selector to the NDIP
    Galaxy `imaginex_subhkl_merge` tool wrapper, omitting `--wavelength-key` in
    `mono` mode while keeping `WAVEL` in the neural-scaler metadata so
    time-resolved Laue harmonics are scaled and merged independently.
  - Automatically selects `careless mono` in `subhkl-nova-app` whenever
    time-of-flight data is present (`peaks/tof`, `tof_gate`, or `events/tof`),
    falling back to `careless poly` for steady-state quasi-Laue datasets.
- Live `Δθ` spot-centroid angular residual diagnostic and top-bar tooltips in
  Rainbow Zones:
  - Measures local detector-spot centroids directly on `merged.h5` for
    predicted/solved reflections and streams `blind_band` / `scan_band`
    micro-step rungs so `Δθ` updates live during `Solve` and reports physical
    angular RMS residuals immediately after finder-less `Solve` (preserved
    across dynamic client-side re-predictions).
  - Added descriptive static and live dynamic tooltips across all top-bar HUD
    pills (`PEAKS`, `Δθ`, `ZONES`, `GONIO`, `FRAME`), selectors, and buttons
    via a `1.5 s`-delay hover overlay (`#ui-hover-tooltip`).
- Automatic `TMPDIR` cleanup in `subhkl-nova-app` and `serve_rainbow_zones`:
  removes per-job staging files immediately upon stage completion and removes
  tracked `subhkl_work_*`, `subhkl_gx_*`, `subhkl_upload_*`,
  `subhkl_merge_tables`, and `subhkl_nova_*` artifacts on normal exit,
  `KeyboardInterrupt`, or `SIGTERM`.

### Changed

- Calibrated the post-merge `ΔI/σ` signed-residual colormap and colorbar to
  span the observed `[min, max]` range with its neutral white midpoint anchored
  at the dataset mean.
- Removed redundant UI controls: the duplicate merging-table button on the
  Wulff bar, the `ALL` Laue-zone button (superseded by `≤1` / `≤2` / `≤3`), and
  the `Walk` solve-step button.
- Reverted container `/dev/shm` and `podman --tmpfs /tmp` scratch overrides so
  large multi-run datasets use standard `TMPDIR` storage paired with exit
  cleanup.

### Fixed

- Time-of-flight gated integration took the wavelength band from the images
  file, which only records the instrument default written at reduce time. When
  the choppers ran a different band (e.g. MANDI at 1–3 Å instead of 2–4 Å),
  every predicted reflection outside the default band was dropped as
  unobserved. The band now comes from the predictions (`instrument/wavelength`
  of the predictor file) and is always widened to hold every predicted
  wavelength; the integrator prints the band it uses.
- Left Careless's built-in Wilson prior intact by defaulting `wilson_prior_b`
  to `None` instead of forcing `--wilson-prior-b=40.0`, which underflowed
  `float32` (`loss=nan`) at high resolution ($d \approx 1\,\text{\AA}$).
- Prevented long top-bar `.subtitle-tag` labels from overlapping the `PEAKS`
  HUD pill via flex overflow clipping and bounding-box overlap scaling.
- Separated staging filenames in `blind_band` (`.blind.rung.tmp.h5` vs.
  `.blind.final.tmp.h5`) so live rung updates cannot collide with the final
  `solve.blind.h5` atomic rename.


## [1.6.0]

### Added

- Time-of-flight gated integration for pulsed-source instruments (MANDI,
  TOPAZ). `reduce` / `reduce-batch` now keep the event times of flight next to
  the detector images (`events/tof`, sorted by pixel so the image itself is the
  index; `--no-events` opts out), and `merge-images` carries them into the
  merged stack. When the images file has events, `integrator` integrates every
  predicted reflection from the events inside a time window around its
  predicted time of flight, with the background measured in time sidebands on
  the same pixels, using the unchanged 2D Poisson amplitude solve on a small
  per-reflection tile. The window is measured from the data by stacking the
  events near all predictions in `t / t_predicted - 1` (`--tof-gate auto`), or
  set with `--tof-gate lo,hi`; `--tof-gate off` restores the whole-band 2D
  integration. Harmonics become separate rows with their own intensities, and
  the output records the gate (`tof_gate`), each row's gate centre
  (`peaks/tof`) and in-gate background (`peaks/tof_background`).
- Resolution-shell hover on the post-merge sparkline in Rainbow Zones
  (`#canvas-hist`): the hovered bar is highlighted and its exact
  `d_max`–`d_min` Å edges, reflection count and active statistic
  ($\mathrm{CC}_{1/2}$, completeness or $\langle I/\sigma \rangle$) are shown
  in the sparkline footer and a floating tooltip (`#hist-bin-tooltip`). Signed
  $\mathrm{CC}_{1/2}$ values are preserved, including negative outer-shell
  correlations, and the readout updates when clicking to cycle metrics.

### Changed

- Time-of-flight gating runs on the accelerator one image at a time: each
  image's events are histogrammed into a (row, column, log-time) cube whose
  cumulative sums give every gated tile and sideband count by lookup, so gate
  and sideband edges are time-bin edges at most a thirtieth of the gate wide.
  A 22-run MANDI job at `d_min` 1.5 (1.12M predictions) integrates in about
  3 minutes on one GPU.
- Vectorized the integrator's per-reflection peak geometry (projection
  matrices, flight distances, streak directions) and shape-fit patch
  extraction per image, with progress output; only the patches the shape fit
  keeps are held in memory.

### Fixed

- Removed a quadratic loop from the app's post-Careless stamping of
  per-observation residuals and scales into `integrated.h5`: each row
  allocated five full-length arrays as lookup defaults (about 3 ms per row,
  most of an hour for the ~950k observations of a `d_min` 1.5 merge).

## [1.5.9]

### Added

- Segmented `Pred` / `Integ` peak overlay modes and direct `.mtz` reflection
  inspection in Rainbow Zones (`subhkl-nova-app` and `rainbow-visualize`):
  - Switch between unfiltered predicted Bragg circles (`Pred`, loaded from
    companion `predictor.h5`, unfiltered `integrated.h5`, or live `UB`
    raycast) and SNR-filtered integrated $2\sigma$ ellipses (`Integ`), with
    mode-distinct intensity highlight palettes (Sky-Cyan/Aqua for `Pred` vs.
    Solar Gold/Chartreuse-Lime for `Integ`) and a live transparency-checkerboard
    `Colorscale` crossover slider.
  - Load external `.mtz` files from disk or Galaxy History once `U` is solved,
    forward-projecting `.mtz` $(h,k,l)$ reflections through the full calibrated
    detector and goniometer geometry (keeping solved `B` and space group
    authoritative with mismatch warnings), coloring by `.mtz I/σ`, and
    persisting `.mtz` overlays across frame switches and `Solve` $\to$
    `Integrate` $\to$ `Merge` transitions.
- Dynamic client-side $d_{\min}$ and $[\lambda_{\min}, \lambda_{\max}]$ peak
  re-prediction with zoom-coupled level-of-detail (LOD) thinning:
  - Re-predicts reflections interactively in the browser past the initial
    indexing/integration cutoff using exact Gemmi space-group
    systematic-absence, ASU, and Bijvoet rules on flat and curved cylindrical
    detectors, guarded by a unit-cell-volume-aware lower $d_{\min}$ floor.
  - Couples zoom level to a progressive $d$-spacing LOD cutoff on both the 3D
    Wulff stereogram and 2D Detector view with live HUD badges and LOD-aware
    hover/click hit-testing.
- Acentric Bijvoet and Friedel mate highlighting across Detector and Wulff
  views, computing signed acentric Bijvoet hand ($\pm 1$ for $I^+/I^-$, $0$ for
  centric zones and centrosymmetric space groups) with pink (`#ff4db8`) hover
  rings, `[Bijvoet]` labels, status-bar `±Bijvoet` badges, and antipodal
  Friedel diamond markers (`-h,-k,-l`).
- `Hi-Res` detector microscope toggle switching between $4\times$-binned
  preview and native $1\times$ pixel textures via a lock-free
  `/api/payload?textures_only=1` fast path (`build_rainbow_panel_textures_only`)
  and `h5wasm` client fallback, plus live coral-amber detector border frames in
  the `Mask` view (`6%` default `rel_border_width` on `MANDI`/`TOPAZ`/`SNAP`,
  `3%` elsewhere).
- Physical vanadium spectral calibration (`--spectral-file` with
  `--trainable-spectral-scale`) and `--lorentz-correction` support in the NOVA
  Careless merge runner alongside `--disable-image-scales`,
  `--wilson-prior-b=40.0`, and clean `WAVEL,THETA,PHI,RUN` metadata columns.

### Changed

- Improved weak-reflection integration, backscattering-shadow masking, and
  high-resolution merging:
  - Added cross-bank same-$2\theta$ peer-ring dark-floor normalization in
    `PixelEMClassifier` to mask whole- and partial-panel shadows, and wired
    `detector/pixel_mask` loading directly from `solve.h5`.
  - Gated sub-pixel centroid snapping by matched-filter SNR, excluded detector
    border pixels from background and Poisson likelihoods, rejected heavily
    occluded footprints ($<65\%$ valid weight), raised default
    `shape_fit_min_snr` to `5.0`, and added `--allow-negative` 1-step
    Fisher-scoring signed intensities for unbiased Careless merging.
- Accelerated multi-run visualization, NDIP staging, and detector rendering:
  - Vectorized flat-panel `_raycast_hkl_on_detectors` across all active banks,
    projected multi-run `All-Runs` Wulff peaks immediately after finder-less
    `Solve`, cached HKL generation and multi-run projections across frame
    switches, deduplicated unchanged `partial_overlay` status polls via
    `overlay_seq`, and defaulted BLAS/OMP thread counts to `1` in
    `subhkl-nova-app`.
  - Streamed multi-GB Galaxy `merged.h5` downloads with live progress,
    eliminated redundant temp copies via hardlinks, batched 3D NumPy panel
    binning, and ran blocking HTTP payload/status handlers via
    `asyncio.to_thread`.
  - Gated parallax strictly on held modifier keys (`Ctrl`/`Shift`/`Alt`/`Meta`
    without mouse buttons) with a wider rubber-band excursion, and rendered
    detector bank textures directly from native $(u,v)$ pixel coordinates with
    adaptive supersampling to eliminate tile aliasing.

### Fixed

- Preserved $\ge 32 \times 32$ bins per panel in `solve --refinement band`
  default binning (`default_band_binning()`, capped at `8`), removing the
  $>240$-panel switch to `binning=16` that collapsed $256 \times 256$
  `MANDI`/`TOPAZ` panels to $16 \times 16$ bins and aliased macromolecular Laue
  zones.
- Fixed right-handed $SO(3)$ Wulff camera projection and centered detector
  azimuth ($\chi$) coupling, and computed exact space-group-aware in-band Laue
  harmonic multiplicity without forbidden centered half-indices.
- Propagated refined `goniometer/{angles,R,harmonics}` into `integrated.h5`,
  accepted uploaded `.mtz` datasets typed as `binary` in Galaxy Stage 4
  Careless discovery, hoisted early colorscale constants above initial script
  boot so `Data Sources` binds cleanly, and reclaimed JAX XLA caches and glibc
  arenas between `test_sparse_rbf_integrator.py` runs to prevent 16 GiB CI OOM
  kills.

## [1.5.8]

### Added

- Per-step live `mm:ss.ms` stopwatch overlay in the lower-right corner of the
  `Terminal // Stdio` pane (`#stdio-stopwatch`) across `subhkl-nova-app` and
  `rainbow-visualize`, featuring an animated skinny 5-stripe Rainbow Islands
  arch with 8-bit sparkles, transient stage-milestone/completion arcade quips,
  server-synchronized `step_timings` persistence, click-to-cycle step split
  inspection (`1. Reduce`, `2. Solve`, `3. Integrate`, `4. Merge`, and
  `Σ Total`), and an Arcade Mode slow-motion rolling-odometer cheat code
  (`↑↑↓↓←→←→BA` or `RAINBOW`).

### Changed

- Accelerated `solve --refinement band` startup, blind band discovery, and EM
  scoring on GPU and eliminated redundant per-run XLA recompilations:
  - Enabled persistent on-disk JAX compilation caching (`~/.cache/subhkl/jax`)
    in `solve` (`enable_jax_compilation_cache`) and hoisted nested
    `refine_band_geometry` Adam/L-BFGS-B closures to module-level bucket-padded
    `@jax.jit` kernels so observation arrays are passed dynamically rather than
    baked as HLO constants.
  - Shared single-pass HDF5 slab loading (`load_binned_nexus_panels`) and
    batched JAX GPU 2D `median`/`75th`-percentile background filters,
    mask-cached EDT fill, and `jax.vmap` IEEE FP32 (`Precision.HIGHEST`)
    multi-run reflection splatting across `load_observations` and
    `PixelEMClassifier`, plus a `@jax.jit` $S^2$ outer-goniometer scan-axis
    recovery sweep.
- Auto-scrolled the `Terminal // Stdio` log pane as live output appends
  (`tail -f` behavior), automatically pausing auto-scroll to keep selected or
  scrolled-up log content steady until the user scrolls back to the bottom.
- Removed parallax tether lines connecting predicted Bragg peak rings to
  detector spots in Rainbow Zones so shifting the parallax view leaves raw
  detector pixels completely unoccluded.

### Fixed

- Fixed CPU CI out-of-memory failures (`exit code 143`) and long runtimes on
  dense raw-pixel `band_table` searches by decomposing `band_table` into fixed
  2D `[2048, <=16384]` dot/zone kernels with $S^2 \times (q_{\text{lo}},
  q_{\text{hi}})$ solid-angle quadrature coalescing when pixel clouds exceed
  $65,536$ bins and releasing temporary `run_spherical_index` arrays early.

## [1.5.7]

### Added

- 3-state Expectation-Maximization (`valid=0`, `dark=1`, `bright=2`) static
  detector pixel classifier (`PixelEMClassifier`) integrated across blind band
  discovery, staged band/Ewald refinement, and peak prediction:
  - Identifies static dark regions (dead detector banks, cryostat/panel
    shadows) via ambient-deficit and missing-predicted-reflection likelihoods
    with nearest-valid EDT background fill, and static bright artifacts
    (parasitic Al magnet/sample-environment flares, hot pixels) via a
    crystallographic Wilson Bragg exponential model and multi-orientation
    persistence across $N \ge 1$ goniometer frames.
  - Reweights and M-step re-polishes coarse orientation candidates in
    `blind_band`, updates per-stage observation weights and Ewald excess maps
    in `scan_band` without JAX recompilation, persists
    `detector/{mask_bank_ids,pixel_mask,pixel_class}` in `solve.h5`, and
    filters masked reflections in `peak-predictor` while propagating the
    `detector` mask group downstream.
- Two-color translucent `Mask` overlay in Rainbow Zones (`subhkl-nova-app` and
  `rainbow-visualize`) rendering classified `Dark` and `Bright` detector
  regions on curvilinear warped panels (including browser-side `h5wasm` loading)
  with sidebar pixel class percentages.

### Changed

- Auto-selected CLI `solve --refinement band` default binning (`binning=8` for
  single/few-frame runs with $\le 240$ panel images, `binning=16` for larger
  multi-run scans) and gated `blind_band` quadrature/table bins to
  kinematically active $2\theta$ angles so coarse $d_{\min}$ single-frame
  solves resolve closely spaced zones.
  - Vectorized multi-bank EM reflection projection and 2D grid-stencil splatting,
  cached clean median backgrounds and module-level `@jax.jit` Ewald/omega scan
  kernels, and concatenated active-bin `band_table` chunks on-device.
- Added model-free outer-goniometer motor validation and recovery
  (`recover_unlogged_outer_goniometer`) for multi-run inner-axis scans where
  outer goniometer motors (for example `omega`, `kappa`) were omitted from raw
  NeXus DASlogs, together with multi-branch `kappa` angle decomposition when
  writing `goniometer/angles`.
- Improved 3D band indexing on strongly anisotropic and streaky crystals
  (such as Mesolite, $b = 56.55\,\text{\AA}$ vs. $c = 6.54\,\text{\AA}$):
  - Fixed `laue_order_stencil` rotation step sizes to use perpendicular
    direct-space lever arms $\max_{j \ne i} |\hat{\mathbf{a}}_i \times
    \mathbf{a}_j|$, added `float32` coarse-zone Adam stencil screening and
    Laue-proper symmetry deduplication across `refine_band_geometry` and
    `distinct_orientations`, and widened coarse zone coverage and Ewald
    background pooling ($56 \times 56\,\text{px}$) so streaky reflections are
    not self-subtracted.

### Fixed

- Fixed post-solve `Files` drawer and `Mask` reload state in `subhkl-nova-app`
  where `runningStageIdx` remained set during completion reload and hid newly
  produced stage artifacts.
- Fixed the upper-right `Tab` button inside `<iframe srcdoc>` views by hiding
  it in standalone top-level tabs and resolving `window.parent.location.href`
  when embedded in a host page.

## [1.5.6]

### Added

- Laue-order-aware $S^2$ spherical-cap max-envelope `lattice_ladder`
  (`score_band_table_env`) and crystallographic $\Delta\mathbf{m} \in
  \{-1,0,+1\}^3$ Laue-order stencil co-refinement (`laue_order_stencil`,
  `refine_band_geometry`) to solve single-frame stills and multi-frame scans
  under uncalibrated detector geometry scissor.
- Automatic local `/SNS` and `/HFIR` filesystem fallback in `subhkl-nova-app`
  when NDIP Galaxy or ONCat is unconfigured or unreachable:
  - Direct NeXus/HDF5 path verification, lightweight HDF5 header metadata
    harvesting (`title`, `DASlogs` goniometer angles, `duration`), and sibling
    `IPTS-*` auto-suggestion on typos.
  - Offline `/SNS/<INST>/IPTS-*/nexus` and `/HFIR/<INST>/IPTS-*/nexus`
    experiment and run catalog browsing.
- Stepped run-interval syntax in the Load Neutron Data run selector (for
  example `2632-2638/2`, `2632:2638:2`, or `2632-2638..2`).

### Changed

- Unified $N \ge 1$ `scan_band` preconditioning and geometry regularization so
  single-frame solves retain anisotropic axial scale (`s_ax`), `parameter_units`
  preconditioning across Stages `0`–`1b`, and `reg_geom = 5.0`.
- Switched uncalibrated count-mode `blind_band` search to deterministic
  top-positive deviance bin selection and capped multi-frame `band_table`
  construction at `131,072` bins.

### Fixed

- Fixed an async polling race in `subhkl-nova-app` where finishing `solve` runs
  could overwrite a newly entered UI $d_{\min}$ with an earlier solve's
  $d_{\min}$.
- Automatic Podman base-image fallback to cached `:main` / `:latest` (with
  bind-mounted `subhkl`) when launching local GPU stages from an unbuilt git
  branch checkout or offline registry.
- Immediate local Podman GPU container termination on Stop (`podman kill` +
  `podman rm -f -i -t 0`) instead of timing out against Podman's 10-second
  `SIGTERM` grace period.

## [1.5.5]

### Added

- Unit cell metadata history on Galaxy datasets in `subhkl-nova-app`:
  - Unit cell parameters ($a, b, c, \alpha, \beta, \gamma$) and space group are
    persisted as dataset annotations/tags when datasets are loaded or produced
    and retrieved alongside Galaxy history.
  - Per-field hoverable autocomplete dropdown in the Load Neutron Data dialog
    that tentatively previews matching sibling unit cell and space group values
    in gray on hover/arrow navigation and commits them in white on selection.
  - Flexible clipboard paste parser in all unit cell and space group inputs
    (accepts `a b c alpha beta gamma` with space group at the front, at the end,
    or omitted).
  - Crystal-system symmetry proposals when selecting a space group (automatically
    linking/proposing constrained lengths and angles such as $a=b=c$ and
    $\alpha=\beta=\gamma=90^\circ$ for cubic, while remaining manually
    overrideable).

### Changed

- Condensed Space Group and unit cell inputs ($a, b, c, \alpha, \beta, \gamma$)
  in the Load Neutron Data dialog into a compact single row.
- Updated HFIR `CG4D` default wavelength range in `reduction_settings.json` to
  `[2.8, 4.5]` Å (with initial $d_{\min} = 1.4$ Å).

### Fixed

- Preserved UI-configured $d_{\min}$, $\lambda_{\min}$, and $\lambda_{\max}$
  when running or re-running `solve` instead of resetting $d_{\min}$ to
  $\lambda_{\min} / 2$.
- Guarded optional `nova.mvvm` and `trame` imports in `MainApp` so base unit
  tests pass when the `nova` extra is not installed.

## [1.5.4]

### Added

- Emitted runner hostname from CLI commands and displayed `[hostname] RUNNING`
  in the `subhkl-nova-app` pipeline status badge to distinguish load-balanced
  NDIP Pulsar workers.
- Added a synchronized $d_{\min}$ resolution slider in the sidebar coupled to
  the Params bar, auto-populated $\lambda_{\min}$ / $\lambda_{\max}$ in the
  Params bar with instrument defaults from `reduction_settings.json`, and added
  unscaled $I/\sigma$ reflection coloring after integration.

### Fixed

- Fixed Galaxy `solve.h5` $\rightarrow$ `integrate` image dataset resolution so
  `peak_predictor` receives the underlying `merged.h5` detector image stack
  rather than `solve.h5`.
- Restored post-merge Careless resolution-bin sparklines ($\mathrm{CC}_{1/2}$,
  $\langle I/\sigma \rangle$, completeness) and downloadable step artifacts.

## [1.5.3]

### Added

- "From Galaxy..." dataset selector modal in `subhkl-nova-app`, sorting Galaxy
  history by most-recently-touched first and truncating the recent dropdown list.

### Fixed

- Refined Galaxy dataset validation and classification, loading cues, Careless
  residual extraction, and downstream stage invalidation when re-running earlier
  stages.
- Avoided resolving Pulsar staging `dataset_*.dat` paths as local `images_file`
  and started `subhkl-nova-app` without a preloaded local dataset.

## [1.5.2]

### Fixed

- Prevented the Load Neutron Data IPTS selector from flipping back to the first
  entry while typing or selecting an experiment, and preserved query parameters
  when rewriting `EP_PATH` in `MainApp`.

## [1.5.1]

### Added

- Calibrated TOPAZ detector geometry (`instruments/topaz_detector.json`, 24 panels)
  and ISAW `.DetCal` converter script (`scripts/detcal_to_json.py`).
- Per-peak angular and $d$-spacing error export in `metrics` (`--per-peak` written
  to a separate HDF5 file, distinct `frame` and `panel` columns, `--bins`, and
  default `metrics.csv` / `metrics.png` filenames for bare `--csv` / `--plot`).
- Finder-free `solve --refinement band` integration test for multi-run mesolite.

### Changed

- `subhkl-nova-app` UI improvements:
  - Side-by-side split view for the Wulff stereogram and detector assembly after
    merge, with coupled reflection target reticles and parallax across both views.
  - Window-independent detector panel aspect ratio, pre-warped offscreen panel
    texture caching for smooth pan/zoom, and dynamic horizontal scaling of the top
    and bottom bars.
  - Per-step Play/Stop job cancellation, Galaxy scheduler `misc_info` error
    reporting, blank-by-default Load Neutron Data dialog (with retired IMAGINE
    image-plate entry removed), new `sUBhkl` logo/favicon, and dynamic version
    and container tag resolution.

### Fixed

- Blind zone-pair orientation proposal in `solve` now includes primitive lattice
  generators so centered cells ($F$, $I$, $C$, $R$) with long axes are not missed
  by the conventional $|u|,|v|,|w| \le 4$ cap.
- `FindUB.minimize` wraps DE Rodrigues vectors into the principal ball before
  L-BFGS-B polish and uses a two-stage wide-to-tight Gaussian funnel for
  long-axis cells.
- Radial peak-profile width estimation filters outlier widths with a monotonic
  $2\theta$ prior.
- Indexer preserves `peaks/run_index` instead of overwriting it with
  `peaks/image_index`.

## [1.5.0]

### Changed

- new UI based on nova-trame, launch with
```
   GALAXY_URL=https://ndip-test.ornl.gov/ GALAXY_API_KEY=<your api key> \
   uv run --extra nova subhkl-nova-app
```
- Band mode can discover a single-crystal orientation without `--bootstrap`
  either from finder observations or directly from raw detector counts using
  smooth $\chi^2$-null-suppressed Poisson bin deviance weights, then jointly
  refine geometry and scan drift.
- CG4D omega polarity is now -1 for newly ingested metadata. Existing HDF5
  axes/rotations are preserved and must be corrected explicitly when needed.
- `solve --refinement band` adds staged CG4D scan refinement with selectable
  axial, cell, per-run, sample and panel modes, optional fixed-assignment pixel
  polish, and a finder-less signed-count variant. See the betalac experiment
  report for scope, measured results and remaining blind-indexing limitations.

- The experimental `calibrate` and `spherical-index` CLI entry points are
  replaced by `solve`: global count-based orientation proposals followed by
  one Poisson group-sparse fit of intensities, orientations and shared detector
  geometry. The standalone numerical Python runners remain reference APIs for
  regression comparisons, not alternate `solve` modes.
- `solve` accepts rotation scans, enables identifiable beam-axis detector roll,
  and honors existing global/per-run goniometer corrections. Its outputs can
  be inspected with finder-free `indexer-visualize` overlays, including labeled
  diagnostics for unsuccessful fits.
- Reflection-family profile measurement is shared in `subhkl.search.profiles`.
  Finder and integrator retain their existing measurements; `solve` can use
  an explicit measured radial prior through `--profile-file`.

The new workflow is draft functionality: frame-addressed stills/scans, flat panels,
fixed cell and wavelength band, beam +z. See [migration and limitations](docs/solve.md).

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
