# src/subhkl/io/command_line_parser.py
from typing import Annotated
from typing import Optional

import sys
import typer
import h5py
import os

from subhkl.utils.devices import restrict_to_first_device
from subhkl.viz.detector_assembly import DEFAULT_N_SIGMA

from subhkl.commands import (
    run_index,
    run_rbf_integrator,
    run_finder,
    run_metrics,
    run_peak_predictor,
    run_mtz_exporter,
    run_reduce,
    run_merge_images,
    run_finder_visualize,
    run_integrator_visualize,
    run_static_mask,
    run_sum_images,
    run_mask_visualize,
)


app = typer.Typer()


def apply_detector_calibration(hdf5_filename: str, instrument: str):
    """
    Reads refined detector metrology from an indexer/prediction file (if present)
    and overrides the in-memory beamlines configuration so downstream
    tasks natively use the calibrated geometry.
    """
    from subhkl.config import beamlines

    if not os.path.exists(hdf5_filename):
        return

    with h5py.File(hdf5_filename, "r") as f:
        if "detector_calibration" in f:
            print(f"Loading calibrated detector geometry from {hdf5_filename}...")
            calib_grp = f["detector_calibration"]
            count = 0
            for bank_key in calib_grp.keys():
                bank_id = bank_key.replace("bank_", "")
                if instrument in beamlines and bank_id in beamlines[instrument]:
                    beamlines[instrument][bank_id]["center"] = calib_grp[bank_key][
                        "center"
                    ][()].tolist()
                    beamlines[instrument][bank_id]["uhat"] = calib_grp[bank_key][
                        "uhat"
                    ][()].tolist()
                    beamlines[instrument][bank_id]["vhat"] = calib_grp[bank_key][
                        "vhat"
                    ][()].tolist()
                    count += 1
            if count > 0:
                print(f"Successfully applied calibration to {count} detector panels.")


app = typer.Typer()


@app.command()
def finder(
    filename: Annotated[str, typer.Argument(help="Input raw/event Nexus file")],
    instrument: Annotated[str, typer.Argument(help="Instrument name")],
    output_filename: str = "output.h5",
    finder_algorithm: Annotated[
        str,
        typer.Option(
            help="Only 'sparse_rbf' remains: the peak_local_max and "
            "thresholding harvesters retired with the convex-hull stage, "
            "whose region-growing intensity they existed to seed."
        ),
    ] = "sparse_rbf",
    show_progress: bool = True,
    create_visualizations: bool = False,
    show_steps: bool = False,
    mask_file: str | None = None,
    mask_rel_erosion_radius: float | None = None,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    sparse_rbf_alpha: Annotated[
        float | None,
        typer.Option(
            help="Significance threshold in units of coefficient noise. Left "
            "unset it is derived so the expected number of false detections "
            "over the image is O(1), which depends on the image size; set it "
            "to demand more evidence than that."
        ),
    ] = None,
    sparse_rbf_gamma: float = 0.0,
    sparse_rbf_min_sigma: Annotated[
        float | None,
        typer.Option(
            help="Bank floor in pixels.  The default (unset) measures it from "
            "the first batch's own peaks -- the same width census that sets "
            "the ceiling, read at a low percentile -- and places the floor one "
            "safe scale-gap below the narrowest peak actually present, never "
            "below the ~1 px sampling limit.  Leaving it unset is what keeps "
            "the detection threshold independent of the pixel pitch: the "
            "false-alarm calibration counts resolution elements, so a floor "
            "pinned in pixels grows a tail of sub-PSF channels as the detector "
            "is refined and drifts the threshold up as sqrt(2 log N_pixels)."
        ),
    ] = None,
    sparse_rbf_max_sigma: Annotated[
        float | None,
        typer.Option(
            help="Bank ceiling in pixels.  The default (unset) measures it "
            "from the first batch's own bright peaks -- a truncated-moment "
            "width census with a two-aperture consistency guard -- and sets "
            "the ceiling at a high percentile of the measured widths; too "
            "few usable peaks falls back to 10 with a warning.  Set it to "
            "pin the ceiling by hand."
        ),
    ] = None,
    sparse_rbf_false_alarms_per_image: Annotated[
        float,
        typer.Option(
            help="Expected number of false peaks per image (the m0 of the "
            "false-alarm calibration). This is the parameter that sets the "
            "detection budget: the significance threshold is solved from "
            "E[false peaks] = m0 over every (position, scale) tested, so "
            "lowering it demands more evidence everywhere at once. gamma "
            "reshapes the threshold across scales at constant budget."
        ),
    ] = 0.1,
    sparse_rbf_max_fragmentation_rate: Annotated[
        float,
        typer.Option(
            help="Tolerable unsupported atoms per image -- the bank-sizing "
            "analogue of --sparse-rbf-false-alarms-per-image.  An unsupported "
            "atom is one whose leave-one-out deviance falls below the chi^2_4 "
            "95% point: the signature of one peak reported as a cluster of "
            "fragments.  Costs nothing extra: the rate maps onto the "
            "brightness quantile of the measured peak census the bank "
            "protects (peaks above it may fragment, ~2 unsupported atoms "
            "each), so it is arithmetic on window moments, not extra solves.  "
            "The realised rate is reported with the final statistics.  Set "
            "to 0 (or negative) to keep the fixed p90 census quantile."
        ),
    ] = 1.0,
    sparse_rbf_profile_file: Annotated[
        str | None,
        typer.Option(
            help="Peak profile replacing the Gaussian atom.  'auto' (default) "
            "measures the radial profile f(u), u = r/sigma, from the first "
            "batch's own bright peaks and rebuilds the bank; a path reads a "
            "measured profile as JSON {u: [...], f: [...]}; 'gaussian' keeps "
            "the analytic Gaussian."
        ),
    ] = "auto",
    sparse_rbf_shape_ratio: Annotated[
        float,
        typer.Option(
            help="Axis ratio of elliptical shape variants added per scale, "
            "area-preserving.  The default 1.2 is the measured anisotropy of "
            "CG4D/MANDI reflections; 1.0 restores the isotropic basis (and "
            "cuts the bank convolution cost 5x)."
        ),
    ] = 1.2,
    sparse_rbf_shape_orientations: Annotated[
        int,
        typer.Option(
            help="Number of position angles for the elliptical variants "
            "(uniform over 180 deg)."
        ),
    ] = 4,
    sparse_rbf_num_sigmas: Annotated[
        int | None,
        typer.Option(
            help="Number of widths in the basis bank, spaced linearly from "
            "--sparse-rbf-min-sigma to --sparse-rbf-max-sigma. Controls the "
            "bank's resolution independently of its ceiling: raising max-sigma "
            "alone widens the spacing, which approximates a peak whose true "
            "width falls between two available scales with several atoms "
            "instead of one. The default (unset) auto-sizes an adaptive bank "
            "just dense enough to prevent exactly that fragmentation, and an "
            "explicit value below that density warns at startup."
        ),
    ] = None,
    # 64, matching MatrixFreeSparseRBFPeakFinder's own default.  This said 512
    # while the orchestrator was not forwarding it, so 512 was never what
    # actually ran -- aligning the defaults keeps "no flag given" meaning what
    # it always meant now that the flag is honored.
    sparse_rbf_chunk_size: int = 64,
    sparse_rbf_loss: Annotated[
        str,
        typer.Option(
            help="Likelihood for the peak finder. Detector frames are photon "
            "counts, so 'poisson' is the matching noise model; 'gaussian' "
            "assumes a single constant variance across the frame."
        ),
    ] = "poisson",
    max_workers: int = 16,
    multi_gpu: Annotated[
        bool,
        typer.Option(
            "--multi-gpu/--no-multi-gpu",
            help="Shard the solve across every visible GPU (the image axis "
            "of each chunk).  Off by default so one process never claims "
            "every device in the machine; scope visibility with "
            "CUDA_VISIBLE_DEVICES.",
        ),
    ] = False,
    static_mask_file: Annotated[
        str | None,
        typer.Option(
            "--static-mask-file",
            help="Static-structure mask built by `static-mask`, mapped onto "
            "the input by physical bank.  Masked pixels enter the solve as "
            "missing data (the counts are never modified) and no peak is "
            "reported from inside the mask.",
        ),
    ] = None,
):
    if not multi_gpu:
        restrict_to_first_device()
    # Pass everything straight into the core logic function
    run_finder(
        filename=filename,
        instrument=instrument,
        output_filename=output_filename,
        finder_algorithm=finder_algorithm,
        show_progress=show_progress,
        create_visualizations=create_visualizations,
        show_steps=show_steps,
        mask_file=mask_file,
        mask_rel_erosion_radius=mask_rel_erosion_radius,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        sparse_rbf_alpha=sparse_rbf_alpha,
        sparse_rbf_gamma=sparse_rbf_gamma,
        sparse_rbf_min_sigma=sparse_rbf_min_sigma,
        sparse_rbf_max_sigma=sparse_rbf_max_sigma,
        sparse_rbf_num_sigmas=sparse_rbf_num_sigmas,
        sparse_rbf_max_fragmentation_rate=sparse_rbf_max_fragmentation_rate,
        sparse_rbf_profile_file=sparse_rbf_profile_file,
        sparse_rbf_shape_ratio=sparse_rbf_shape_ratio,
        sparse_rbf_shape_orientations=sparse_rbf_shape_orientations,
        sparse_rbf_false_alarms_per_image=sparse_rbf_false_alarms_per_image,
        sparse_rbf_chunk_size=sparse_rbf_chunk_size,
        sparse_rbf_loss=sparse_rbf_loss,
        max_workers=max_workers,
        multi_gpu=multi_gpu,
        static_mask_file=static_mask_file,
    )


@app.command()
def indexer(
    peaks_h5_filename: str,
    output_peaks_filename: str,
    a: Annotated[float | None, typer.Option(help="Unit cell parameter a")] = None,
    b: Annotated[float | None, typer.Option(help="Unit cell parameter b")] = None,
    c: Annotated[float | None, typer.Option(help="Unit cell parameter c")] = None,
    alpha: Annotated[
        float | None, typer.Option(help="Unit cell parameter alpha")
    ] = None,
    beta: Annotated[float | None, typer.Option(help="Unit cell parameter beta")] = None,
    gamma: Annotated[
        float | None, typer.Option(help="Unit cell parameter gamma")
    ] = None,
    space_group: Annotated[
        str | None, typer.Option(help="Space group (e.g. 'P 1')")
    ] = None,
    wavelength_min: Annotated[float | None, typer.Option("--wavelength-min")] = None,
    wavelength_max: Annotated[float | None, typer.Option("--wavelength-max")] = None,
    ki_vec: Annotated[
        str | None,
        typer.Option(
            "--ki-vec", help="Override incident beam vector (e.g., '0,0,1' or '0,0,-1')"
        ),
    ] = None,
    original_nexus_filename: Annotated[
        str | None,
        typer.Option("--nexus", help="Original nexus file for instrument definitions"),
    ] = None,
    instrument_name: Annotated[str | None, typer.Option("--instrument")] = None,
    strategy_name: Annotated[str, typer.Option("--strategy")] = "DE",
    sigma_init: Annotated[float | None, typer.Option("--sigma-init")] = None,
    n_runs: Annotated[int, typer.Option("--n-runs", "-n")] = 1,
    population_size: Annotated[
        int, typer.Option("--population-size", "--popsize")
    ] = 1000,
    gens: Annotated[int, typer.Option("--gens")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    tolerance_deg: Annotated[float, typer.Option("--tolerance-deg")] = 0.1,
    freeze_orientation: Annotated[
        bool,
        typer.Option(
            "--freeze-orientation", help="Lock the U matrix to its initial state."
        ),
    ] = False,
    refine_lattice: Annotated[bool, typer.Option("--refine-lattice")] = False,
    lattice_bound_frac: Annotated[float, typer.Option("--lattice-bound-frac")] = 0.05,
    refine_goniometer: Annotated[bool, typer.Option("--refine-goniometer")] = False,
    refine_goniometer_axes: Annotated[
        str | None, typer.Option("--refine-goniometer-axes")
    ] = None,
    goniometer_bound_deg: Annotated[
        str,
        typer.Option(
            "--goniometer-bound-deg",
            help="Comma-separated bounds per axis or a single float",
        ),
    ] = "5.0",
    refine_goniometer_axis_vector: Annotated[
        str | None,
        typer.Option(
            "--refine-goniometer-axis-vector",
            help="Comma-separated motor names whose axis *direction* is "
            "refined (two tilt angles each about an orthonormal basis "
            "perpendicular to the nominal axis).  A goniometer mounted at "
            "a small angle to the detector frame tilts its axes -- an "
            "error the angular offsets and translations can only chase "
            "degenerately.  The refined vectors are written to "
            "goniometer/axes (nominals kept as goniometer/axes_nominal, "
            "tilt angles as goniometer/axis_tilts), so downstream stages "
            "pick them up transparently, and a calibration pass "
            "bootstraps them as its nominal geometry.",
        ),
    ] = None,
    goniometer_axis_vector_bound_deg: Annotated[
        str,
        typer.Option(
            "--goniometer-axis-vector-bound-deg",
            help="Comma-separated tilt bounds (deg) per refined axis "
            "vector, or a single float.",
        ),
    ] = "1.0",
    refine_goniometer_per_run: Annotated[
        Optional[str],
        typer.Option(
            "--refine-goniometer-per-run",
            help="Motor name (e.g. 'phi') to refine one bounded angle "
            "correction per scan run: per-setting positioning errors "
            "(encoder repeatability, mount settling) cannot be "
            "represented by any static geometry parameter.  Needs a "
            "merged --nexus file with file_offsets for the frame -> run "
            "bookkeeping.",
        ),
    ] = None,
    goniometer_per_run_bound_deg: Annotated[
        float,
        typer.Option(
            "--goniometer-per-run-bound-deg",
            help="Bound (deg) for each per-run angle correction.",
        ),
    ] = 0.5,
    refine_goniometer_per_run_trans: Annotated[
        bool,
        typer.Option(
            "--refine-goniometer-per-run-trans/--no-refine-goniometer-per-run-trans",
            help="Refine one bounded sample displacement (3-vector) per "
            "scan run, attached at the innermost goniometer axis: the "
            "translational twin of the per-run angle corrections (mount "
            "settling, sphere of confusion).  Needs a merged --nexus "
            "file with file_offsets.",
        ),
    ] = False,
    goniometer_per_run_trans_bound_meters: Annotated[
        float,
        typer.Option(
            "--goniometer-per-run-trans-bound-meters",
            help="Bound (meters) per component of each per-run sample displacement.",
        ),
    ] = 0.002,
    refine_goniometer_harmonics: Annotated[
        Optional[str],
        typer.Option(
            "--refine-goniometer-harmonics",
            help="Scan motor name (e.g. 'phi') to refine a Fourier-in-phi "
            "rocking of the crystal's effective orientation: bounded "
            "cos/sin coefficients per harmonic about fixed lab axes "
            "built from the scan axis and the beam.  Captures the "
            "phi-periodic steering that crystal-fixed anisotropic "
            "mosaicity imprints on spot positions (grows as 2 sin "
            "theta, so it dominates the high-2theta banks).",
        ),
    ] = None,
    goniometer_harmonics_orders: Annotated[
        Optional[str],
        typer.Option(
            "--goniometer-harmonics-orders",
            help="Comma-separated harmonic orders m (default '1,2,3,4,5,6'). "
            "Crystallographic rotation orders top out at 6, and a "
            "symmetry axis tilted from the scan axis leaks harmonic n "
            "into the n +/- 1 sidebands, so the full band is the safe "
            "default; m = 0 is always excluded (it is the motor zero "
            "the global offsets refine).  For sparse phi coverage "
            "restrict the band to keep the fit determined.",
        ),
    ] = None,
    goniometer_harmonics_axes: Annotated[
        str,
        typer.Option(
            "--goniometer-harmonics-axes",
            help="Which rocking axes get a harmonic series: 'rocking' "
            "(scan-axis x beam, the rocking-curve axis; 2M DoF), "
            "'transverse' (both axes perpendicular to the scan axis; "
            "4M), or 'full' (adds the scan axis itself for periodic "
            "drive error; 6M).",
        ),
    ] = "rocking",
    goniometer_harmonics_bound_deg: Annotated[
        float,
        typer.Option(
            "--goniometer-harmonics-bound-deg",
            help="Bound (deg) per Fourier coefficient of the rocking.",
        ),
    ] = 0.5,
    refine_goniometer_trans_axes: Annotated[
        Optional[str],
        typer.Option(
            "--refine-goniometer-trans-axes",
            help="Comma-separated motor names whose lever-arm translations "
            "to refine.  Defaults to the --refine-goniometer-axes list; "
            "set it separately to refine e.g. the phi-stage (sample) "
            "translation without freeing the pure-gauge phi zero point.",
        ),
    ] = None,
    refine_goniometer_trans: Annotated[
        bool, typer.Option("--refine-goniometer-trans")
    ] = False,
    goniometer_trans_bound_meters: Annotated[
        str,
        typer.Option(
            "--goniometer-trans-bound-meters",
            help="Comma-separated translation bounds per axis (meters) or a single float",
        ),
    ] = "0.005",
    refine_beam: Annotated[bool, typer.Option("--refine-beam")] = False,
    beam_bound_deg: Annotated[float, typer.Option("--beam-bound-deg")] = 1.0,
    refine_detector: Annotated[bool, typer.Option("--refine-detector")] = False,
    refine_detector_banks: Annotated[
        str | None,
        typer.Option(
            "--refine-detector-banks", help="Comma-separated bank IDs to refine"
        ),
    ] = None,
    detector_modes: Annotated[
        str,
        typer.Option(
            "--detector-modes",
            help="Comma-separated list of refinement modes (e.g. radial,global_rot,area,independent,axial_stretch)",
        ),
    ] = "independent",
    detector_trans_bound_meters: Annotated[
        float, typer.Option("--detector-trans-bound-meters")
    ] = 0.005,
    detector_rot_bound_deg: Annotated[
        float, typer.Option("--detector-rot-bound-deg")
    ] = 1.0,
    detector_global_rot_bound_deg: Annotated[
        float, typer.Option("--detector-global-rot-bound-deg")
    ] = 2.0,
    detector_global_rot_axis: Annotated[
        str,
        typer.Option(
            "--detector-global-rot-axis",
            help="Axis vector for global_rot_axis mode (e.g. 0,1,0)",
        ),
    ] = None,
    cylinder_axis: Annotated[
        str,
        typer.Option(
            "--cylinder-axis",
            help="Axis vector for global_rot_axis mode (e.g. 0,1,0)",
        ),
    ] = None,
    detector_area_bound_frac: Annotated[
        float, typer.Option("--detector-area-bound-frac")
    ] = 0.05,
    detector_global_trans_bound_meters: Annotated[
        float, typer.Option("--detector-global-trans-bound-meters")
    ] = 0.01,
    detector_radial_bound_frac: Annotated[
        float, typer.Option("--detector-radial-bound-frac")
    ] = 0.05,
    bootstrap_filename: Annotated[str | None, typer.Option("--bootstrap")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
    num_candidates: Annotated[
        int | None, typer.Option(help="Number of lambda candidates (default: 64)")
    ] = None,
    index: Annotated[Optional[bool], typer.Option("--index/--no-index")] = None,
    radial_weight: Annotated[
        float,
        typer.Option(
            help="Dimensionless weight in [0, 1] multiplying the radial "
            "(2-theta gradient) component of the --no-index positional "
            "residual: the tangential-to-radial streak scale ratio.  1 "
            "keeps the isotropic chord, 0 fits tangential-only (measured "
            "0.27 on cg4d-t4-lysozyme)."
        ),
    ] = 1.0,
    radial_weight_poly: Annotated[
        Optional[str],
        typer.Option(
            help="Comma-separated polynomial coefficients for w(lambda) "
            "in Angstrom, highest degree first; overrides --radial-weight."
        ),
    ] = None,
    hkl_metric: Annotated[
        str,
        typer.Option(
            help="Basin metric for the soft indexing loss: 'isotropic' "
            "(plain fractional-hkl distance) or 'positional' (each basin "
            "warped by the Jacobian to detector displacement, radial "
            "component weighted by --radial-weight; candidate selection "
            "becomes the anisotropic assignment)."
        ),
    ] = "isotropic",
    hkl_metric_floor: Annotated[
        float,
        typer.Option(
            help="Dimensionless isotropic floor added to the positional "
            "metric; keeps the wavelength-tube null direction from "
            "becoming gauge."
        ),
    ] = 0.1,
    multi_gpu: Annotated[
        bool,
        typer.Option(
            "--multi-gpu/--no-multi-gpu",
            help="Shard the independent optimization runs across every "
            "visible GPU.  Off by default so one process never claims every "
            "device in the machine; scope visibility with "
            "CUDA_VISIBLE_DEVICES.",
        ),
    ] = False,
) -> None:
    if not multi_gpu:
        restrict_to_first_device()
    # 1. Safely Parse Comma-Separated Strings into Python Lists
    ki_vec_parsed = [float(x.strip()) for x in ki_vec.split(",")] if ki_vec else None
    gonio_axes_parsed = (
        [x.strip() for x in refine_goniometer_axes.split(",")]
        if refine_goniometer_axes
        else None
    )
    gonio_bounds_parsed = (
        [float(x.strip()) for x in goniometer_bound_deg.split(",")]
        if goniometer_bound_deg
        else [5.0]
    )
    gonio_trans_bounds_parsed = (
        [float(x.strip()) for x in goniometer_trans_bound_meters.split(",")]
        if goniometer_trans_bound_meters
        else [0.005]
    )
    gonio_axis_vec_parsed = (
        [x.strip() for x in refine_goniometer_axis_vector.split(",")]
        if refine_goniometer_axis_vector
        else None
    )
    gonio_axis_vec_bounds_parsed = (
        [float(x.strip()) for x in goniometer_axis_vector_bound_deg.split(",")]
        if goniometer_axis_vector_bound_deg
        else [1.0]
    )
    det_banks_parsed = (
        [int(x.strip()) for x in refine_detector_banks.split(",")]
        if refine_detector_banks
        else None
    )
    det_modes_parsed = (
        [x.strip().lower() for x in detector_modes.split(",")]
        if detector_modes
        else ["independent"]
    )
    global_rot_axis_parsed = (
        [float(x.strip()) for x in detector_global_rot_axis.split(",")]
        if detector_global_rot_axis
        else None
    )
    cylinder_axis = (
        [float(x.strip()) for x in cylinder_axis.split(",")] if cylinder_axis else None
    )

    # 2. Hand off to Core Logic
    run_index(
        peaks_h5_filename=peaks_h5_filename,
        output_peaks_filename=output_peaks_filename,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        space_group=space_group,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        ki_vec=ki_vec_parsed,
        original_nexus_filename=original_nexus_filename,
        instrument_name=instrument_name,
        strategy_name=strategy_name,
        sigma_init=sigma_init,
        n_runs=n_runs,
        population_size=population_size,
        gens=gens,
        seed=seed,
        tolerance_deg=tolerance_deg,
        freeze_orientation=freeze_orientation,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        refine_goniometer=refine_goniometer,
        refine_goniometer_axes=gonio_axes_parsed,
        goniometer_bound_deg=gonio_bounds_parsed,
        refine_goniometer_axis_vector=gonio_axis_vec_parsed,
        refine_goniometer_per_run=refine_goniometer_per_run,
        goniometer_per_run_bound_deg=goniometer_per_run_bound_deg,
        refine_goniometer_per_run_trans=refine_goniometer_per_run_trans,
        goniometer_per_run_trans_bound_meters=goniometer_per_run_trans_bound_meters,
        refine_goniometer_harmonics=refine_goniometer_harmonics,
        goniometer_harmonics_orders=(
            [int(x.strip()) for x in goniometer_harmonics_orders.split(",")]
            if goniometer_harmonics_orders
            else None
        ),
        goniometer_harmonics_axes=goniometer_harmonics_axes,
        goniometer_harmonics_bound_deg=goniometer_harmonics_bound_deg,
        refine_goniometer_trans_axes=(
            [x.strip() for x in refine_goniometer_trans_axes.split(",")]
            if refine_goniometer_trans_axes
            else None
        ),
        goniometer_axis_vector_bound_deg=gonio_axis_vec_bounds_parsed,
        refine_goniometer_trans=refine_goniometer_trans,
        goniometer_trans_bound_meters=gonio_trans_bounds_parsed,
        refine_beam=refine_beam,
        beam_bound_deg=beam_bound_deg,
        refine_detector=refine_detector,
        refine_detector_banks=det_banks_parsed,
        detector_modes=det_modes_parsed,
        detector_trans_bound_meters=detector_trans_bound_meters,
        detector_rot_bound_deg=detector_rot_bound_deg,
        detector_global_rot_bound_deg=detector_global_rot_bound_deg,
        detector_global_rot_axis=global_rot_axis_parsed,
        detector_global_trans_bound_meters=detector_global_trans_bound_meters,
        detector_radial_bound_frac=detector_radial_bound_frac,
        detector_area_bound_frac=detector_area_bound_frac,
        cylinder_axis=cylinder_axis,
        bootstrap_filename=bootstrap_filename,
        batch_size=batch_size,
        num_candidates=num_candidates,
        no_index=not index if index is not None else None,
        radial_weight=radial_weight,
        radial_weight_poly=(
            [float(x.strip()) for x in radial_weight_poly.split(",")]
            if radial_weight_poly
            else None
        ),
        hkl_metric=hkl_metric,
        hkl_metric_floor=hkl_metric_floor,
        multi_gpu=multi_gpu,
    )


@app.command()
def integrator(
    filename: Annotated[str, typer.Argument(help="Merged HDF5 image stack")],
    instrument: Annotated[str, typer.Argument(help="Instrument name")],
    integration_peaks_filename: Annotated[
        str, typer.Argument(help="Predicted peaks HDF5 file")
    ],
    output_filename: Annotated[
        str, typer.Argument(help="Output integrated peaks HDF5 file")
    ],
    sigmas: Annotated[str, typer.Option(help="Unstretched peak radii")] = "1.0,2.0,4.0",
    nominal_sigma: Annotated[
        float,
        typer.Option(
            help="The typical peak radius, used as a fallback for weak reflections"
        ),
    ] = 1.0,
    anisotropic: Annotated[
        bool, typer.Option(help="Integrate anisotropic quasi-Laue peaks")
    ] = False,
    fit_mosaicity: Annotated[
        bool,
        typer.Option(
            help="Whether to fit the mosaicity separately from sample dimensions to explain peak shape. Only use in non-spherical detector geometries."
        ),
    ] = False,
    mosaicity_radial: Annotated[
        bool,
        typer.Option(
            "--mosaicity-radial/--mosaicity-isotropic",
            help="Model the mosaic spread as a streak along the per-peak "
            "2-theta gradient (a mosaic block rotated within the "
            "scattering plane stays reflective at an adjusted wavelength) "
            "instead of an isotropic 3D blur.  Requires --fit-mosaicity.",
        ),
    ] = False,
    shape_spherical: Annotated[
        bool,
        typer.Option(
            "--shape-spherical/--shape-ellipsoidal",
            help="Constrain the sample tensor to a sphere (one radius).  "
            "With --mosaicity-radial this is the hypothesis test that the "
            "spot anisotropy is streak physics rather than the parallel "
            "projection of the sample volume; a sphere is also "
            "rotation-invariant, removing any sample<->lab convention "
            "question from the shape pathway.",
        ),
    ] = False,
    mosaicity_bound_mrad: Annotated[
        float,
        typer.Option(help="Upper bound for the fitted mosaicity, in mrad."),
    ] = 10.0,
    shape_fit_min_snr: Annotated[
        float,
        typer.Option(
            help="Restrict the global shape fit to peaks whose 5x5 core "
            "exceeds this SNR over the local background.  Weak patches "
            "carry no shape information and bias the fitted widths "
            "upward (broad templates soak background pedestals).  "
            "0 keeps every peak."
        ),
    ] = 0.0,
    shape_fit_normalized: Annotated[
        bool,
        typer.Option(
            "--shape-fit-normalized/--no-shape-fit-normalized",
            help="Normalize each patch's shape-fit error by its own power "
            "so patches vote with their misfit fraction, not their "
            "brightness (a handful of bright near-beam tails otherwise "
            "dominate the global template).",
        ),
    ] = False,
    matrix_free: Annotated[
        bool,
        typer.Option(
            "--matrix-free/--no-matrix-free",
            help="Deprecated no-op: the matrix-free amplitude solve is the "
            "only integration path (one nonnegative Poisson solve per "
            "image on the finder's rate-map noise model).  The per-patch "
            "fit it replaced was retired after losing on every common "
            "reflection set; --no-matrix-free is an error.",
        ),
    ] = True,
    matrix_free_profile: Annotated[
        str,
        typer.Option(
            help="Radial atom profile for --matrix-free: 'gaussian' "
            "(analytic), 'auto' (measure the family's trunk from "
            "isolated bright peaks in Mahalanobis coordinates of the "
            "projected shape model -- the finder's low-rank census with "
            "known centroids and covariances -- falling back to the "
            "Gaussian if too few qualify), or a path to a finder-style "
            "profile JSON."
        ),
    ] = "gaussian",
    matrix_free_fp_target: Annotated[
        float | None,
        typer.Option(
            help="Expected number of FALSE admissions over the whole "
            "dataset for the matrix-free L1 gate; the admission "
            "threshold is z = Phi^-1(1 - fp_target/n_predictions) -- "
            "the finder's false-alarm calibration with the "
            "integrator's own test count.  Unset (the default) applies "
            "no gate: elimination happens only at the nonnegativity "
            "boundary.  Eliminated reflections are censored (sigI = 0, "
            "dropped from the export); the gate is a purity/"
            "completeness dial, measured -31 points of completeness at "
            "z = 2.17 on cg4d-t4-lysozyme."
        ),
    ] = None,
    static_mask_file: Annotated[
        str | None,
        typer.Option(
            help="Static-structure mask HDF5 (from `static-mask`), mapped "
            "onto the input by bank id.  With --matrix-free, masked "
            "pixels are excluded from both the rate-map background and "
            "the amplitude likelihood as missing data."
        ),
    ] = None,
    rel_border_width: Annotated[
        float, typer.Option(help="Border width in fraction of image size")
    ] = 0.0,
    show_progress: Annotated[bool, typer.Option("--show-progress")] = True,
    create_visualizations: bool = False,
    chunk_size: int = 256,
    max_workers: Annotated[
        int | None, typer.Option(help="Maximum number of CPU tasks for visualization.")
    ] = None,
):
    """
    Integrates predicted peaks using the Dense Sparse RBF network approach on GPU.
    Calculates intensities and rigorous I/SIGI via Fisher Information matrix SVD.
    """
    # click's context exists only on CLI invocation (silent=True returns None
    # for direct function calls, where no deprecation question arises), and it
    # is what knows which of the two registered names was typed.
    import click

    _ctx = click.get_current_context(silent=True)
    if _ctx is not None and _ctx.command.name == "rbf-integrator":
        print(
            "WARNING: 'rbf-integrator' is a deprecated alias; use 'integrator'.",
            file=sys.stderr,
        )
    if not matrix_free:
        raise typer.BadParameter(
            "the per-patch integrator was retired; --no-matrix-free has no "
            "implementation (see the matrix-free integration notes in "
            "subhkl.search.matrix_free)"
        )
    run_rbf_integrator(
        filename=filename,
        instrument=instrument,
        integration_peaks_filename=integration_peaks_filename,
        output_filename=output_filename,
        sigmas=sigmas,
        nominal_sigma=nominal_sigma,
        anisotropic=anisotropic,
        fit_mosaicity=fit_mosaicity,
        mosaicity_radial=mosaicity_radial,
        shape_spherical=shape_spherical,
        mosaicity_bound_mrad=mosaicity_bound_mrad,
        shape_fit_min_snr=shape_fit_min_snr,
        shape_fit_normalized=shape_fit_normalized,
        matrix_free_profile=matrix_free_profile,
        matrix_free_fp_target=matrix_free_fp_target,
        static_mask_file=static_mask_file,
        rel_border_width=rel_border_width,
        show_progress=show_progress,
        create_visualizations=create_visualizations,
        chunk_size=chunk_size,
        max_workers=max_workers,
    )


# The command's historical name; same callback, flagged deprecated in --help
# and warned about at runtime (via ctx.command.name above).
app.command("rbf-integrator", deprecated=True, hidden=True)(integrator)


@app.command()
def metrics(
    file1: Annotated[
        str, typer.Argument(help="Primary file (e.g., indexer.h5 or predictor.h5)")
    ],
    file2: Annotated[
        str | None,
        typer.Option(
            "--file2",
            help="Optional secondary file to match against (e.g., finder.h5).",
        ),
    ] = None,
    instrument: Annotated[
        str | None,
        typer.Option(
            "--instrument",
            help="Instrument name (required if using file2 or predictor outputs).",
        ),
    ] = None,
    d_min: Annotated[
        float | None,
        typer.Option(
            "--d-min", help="Optional minimum d-spacing filter for metrics calculation."
        ),
    ] = None,
    per_run: Annotated[
        bool,
        typer.Option(
            "--per-run", help="Calculate and display metrics for each run/image."
        ),
    ] = False,
    per_peak: Annotated[
        bool,
        typer.Option(
            "--per-peak",
            help="Write the metrics/per_peak table (h, k, l, run, lambda,"
            "d_err, ang_err) into FILE1, for scripts/error_analysis.py.",
        ),
    ] = False,
    ki_vec: Annotated[
        str | None,
        typer.Option(
            "--ki-vec", help="Override incident beam vector (e.g., '0,0,1' or '0,0,-1')"
        ),
    ] = None,
):
    """
    CLI command to compute and display indexing quality metrics.
    Compares HKL accuracy internally (file1), or spatial matching between file1 (predicted) and file2 (observed).
    """
    ki_vec_parsed = [float(x.strip()) for x in ki_vec.split(",")] if ki_vec else None

    run_metrics(
        file1=file1,
        file2=file2,
        instrument=instrument,
        d_min=d_min,
        per_run=per_run,
        per_peak=per_peak,
        ki_vec=ki_vec_parsed,
    )


@app.command()
def peak_predictor(
    filename: str,
    instrument: str,
    indexed_hdf5_filename: str,
    integration_peaks_filename: str,
    d_min: float = 1.0,
    create_visualizations: bool = False,
    space_group: str | None = None,
    wavel_min: float | None = None,
    wavel_max: float | None = None,
    max_workers: int = 16,
):
    run_peak_predictor(
        filename,
        instrument,
        indexed_hdf5_filename,
        integration_peaks_filename,
        d_min=d_min,
        wavel_min=wavel_min,
        wavel_max=wavel_max,
        space_group=space_group,
        max_workers=max_workers,
        create_visualizations=create_visualizations,
    )


@app.command()
def mtz_exporter(
    indexed_h5_filename: str,
    output_mtz_filename: str,
    space_group: str = typer.Option(
        None, help="Optional. Loaded from indexer h5 if missing."
    ),
    predictions_file: Annotated[
        str | None,
        typer.Option(
            help="Predictor HDF5; adds SNAPD (distance from the "
            "integrated to the nearest predicted position, px) and "
            "SIGEFF (projected peak radius, px) columns -- per-peak "
            "systematics proxies a scaling model can learn from."
        ),
    ] = None,
    corrections_file: Annotated[
        str | None,
        typer.Option(
            help="Indexer HDF5 carrying goniometer/per_run; adds DPHI "
            "(deg) and DTX/DTY/DTZ (mm) columns with the peak's run's "
            "fitted goniometer corrections."
        ),
    ] = None,
):
    run_mtz_exporter(
        indexed_h5_filename,
        output_mtz_filename,
        space_group,
        predictions_file=predictions_file,
        corrections_file=corrections_file,
    )


@app.command()
def reduce(
    nexus_filename: str,
    output_filename: str,
    instrument: str,
    wavelength_min: Annotated[
        float | None, typer.Option(help="Override min wavelength")
    ] = None,
    wavelength_max: Annotated[
        float | None, typer.Option(help="Override max wavelength")
    ] = None,
):
    run_reduce(
        nexus_filename, output_filename, instrument, wavelength_min, wavelength_max
    )


@app.command()
def merge_images(
    input_pattern: Annotated[
        str,
        typer.Argument(help="Glob pattern for reduced .h5 files (e.g. 'reduced/*.h5')"),
    ],
    output_filename: Annotated[str, typer.Argument(help="Output master .h5 file")],
    a: float = typer.Argument(..., help="Unit cell parameter a"),
    b: float = typer.Argument(..., help="Unit cell parameter b"),
    c: float = typer.Argument(..., help="Unit cell parameter c"),
    alpha: float = typer.Argument(..., help="Unit cell parameter alpha"),
    beta: float = typer.Argument(..., help="Unit cell parameter beta"),
    gamma: float = typer.Argument(..., help="Unit cell parameter gamma"),
    space_group: str = typer.Argument(..., help="Space group (e.g. 'P 1')"),
):
    try:
        run_merge_images(
            input_pattern, output_filename, a, b, c, alpha, beta, gamma, space_group
        )

    except ValueError as e:
        print(str(e))
        raise typer.Exit(code=1)


_IMAGES_HELP = "Reduced (or merged) HDF5 file holding the image stack the search ran on"
_INSTRUMENT_HELP = (
    "Instrument name. Read from the peaks file when not given; merged image "
    "files do not record it, so pass it if the peaks file predates the change."
)
_OUTPUT_DIR_HELP = "Where to write the plots (default: next to the peaks file)"
_N_SIGMA_HELP = (
    "How many standard deviations out to draw each peak's outline. A peak has "
    "no edge, only a width, so the circle is a choice of contour; 2 sigma "
    "holds about 86% of a 2D Gaussian's flux."
)
_DPI_HELP = (
    "Resolution of the saved plots. The inline plots are written at 600, which "
    "is too heavy to keep for every run of a benchmark sweep."
)


@app.command()
def finder_visualize(
    images_filename: Annotated[str, typer.Argument(help=_IMAGES_HELP)],
    peaks_filename: Annotated[str, typer.Argument(help="Finder output HDF5 file")],
    instrument: Annotated[Optional[str], typer.Option(help=_INSTRUMENT_HELP)] = None,
    output_dir: Annotated[Optional[str], typer.Option(help=_OUTPUT_DIR_HELP)] = None,
    dpi: Annotated[int, typer.Option(help=_DPI_HELP)] = 150,
    n_sigma: Annotated[float, typer.Option(help=_N_SIGMA_HELP)] = DEFAULT_N_SIGMA,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
):
    """
    Redraw the finder's unrolled-detector plots from an existing output file.

    Produces the same '-found.png' per run that 'finder
    --create-visualizations' would have, reading the peak centres and widths
    back out of the peaks file instead of searching the images again. Run the
    finder without visualizations, keep the two HDF5 files, and come back to
    the pictures later.
    """
    run_finder_visualize(
        images_filename=images_filename,
        peaks_filename=peaks_filename,
        instrument=instrument,
        output_dir=output_dir,
        dpi=dpi,
        n_sigma=n_sigma,
        max_workers=max_workers,
        show_progress=show_progress,
    )


@app.command()
def integrator_visualize(
    images_filename: Annotated[str, typer.Argument(help=_IMAGES_HELP)],
    peaks_filename: Annotated[str, typer.Argument(help="integrator output HDF5 file")],
    instrument: Annotated[Optional[str], typer.Option(help=_INSTRUMENT_HELP)] = None,
    output_dir: Annotated[Optional[str], typer.Option(help=_OUTPUT_DIR_HELP)] = None,
    dpi: Annotated[int, typer.Option(help=_DPI_HELP)] = 150,
    n_sigma: Annotated[float, typer.Option(help=_N_SIGMA_HELP)] = DEFAULT_N_SIGMA,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
):
    """
    Redraw the RBF integrator's unrolled-detector plots from an existing output file.

    Produces the same '-pred.png' per run that 'integrator
    --create-visualizations' would have, drawing each peak at the shape the
    integrator fitted it, without repeating the integration.
    """
    run_integrator_visualize(
        images_filename=images_filename,
        peaks_filename=peaks_filename,
        instrument=instrument,
        output_dir=output_dir,
        dpi=dpi,
        n_sigma=n_sigma,
        max_workers=max_workers,
        show_progress=show_progress,
    )


@app.command()
def static_mask(
    output_filename: Annotated[str, typer.Argument(help="Mask HDF5 to write")],
    input_filenames: Annotated[
        list[str],
        typer.Argument(
            help="Reduced/merged HDF5 stacks (images + bank_ids).  Any "
            "number of files, any samples -- what matters is that they come "
            "from the same instrument configuration, because the estimator "
            "keeps what never moves."
        ),
    ],
    peaks: Annotated[
        list[str] | None,
        typer.Option(
            "--peaks",
            help="Finder outputs from an unmasked run, one per input file in "
            "the same order.  Optional: unnecessary when the inputs are a "
            "control experiment without a sample.  Detections whose fit "
            "metrics certify them as genuine (see --peak-deviance-min / "
            "--peak-residual-max) are exonerated: their footprints leave "
            "the static evidence, so a reflection cannot be declared "
            "static however many frames it persists through.",
        ),
    ] = None,
    pooled_peaks: Annotated[
        Optional[str],
        typer.Option(
            "--pooled-peaks",
            help="Finder output from a run on the per-bank *summed* stack "
            "(see `sum-images`).  Certified detections are exonerated in "
            "every frame of their bank: significance compounds across "
            "frames in the pooled fit, rescuing quasi-static reflections "
            "too faint for any single frame's certificate -- which the "
            "static map, pooling the same frames, would otherwise mask.",
        ),
    ] = None,
    peak_deviance_min: Annotated[
        float,
        typer.Option(
            help="Exoneration needs per-peak deviance above this.  The "
            "default is the chi^2_4 admission level (9.49): the finder's "
            "calibrated false-alarm control already governs evidence above "
            "it, and a higher bar leaves faint genuine peaks -- bright "
            "enough to mask, too faint to certify -- with no route to "
            "exoneration."
        ),
    ] = 9.488,
    peak_residual_max: Annotated[
        float,
        typer.Option(
            help="Exoneration needs residual deviance per DoF below this "
            "(a shape the atom family explains; artifacts fail it)."
        ),
    ] = 2.0,
    peak_clear_nsigmas: Annotated[
        float,
        typer.Option(
            help="Minimum protected radius around an exonerated peak, in "
            "units of its fitted width; brightness extends it further (to "
            "where the peak's own tail falls below the texture threshold)."
        ),
    ] = 3.5,
    min_frames: Annotated[
        int,
        typer.Option(
            help="Banks with fewer frames than this across all inputs stay "
            "fully valid (and are reported): statistics too thin to tell a "
            "peak from a shadow must not silently mask either."
        ),
    ] = 5,
    smooth_sigma: Annotated[
        float, typer.Option(help="Gaussian smoothing of the per-bank median, px.")
    ] = 2.0,
    grad_nmads: Annotated[
        float,
        typer.Option(
            help="Boundary criterion: mask where the smoothed median's "
            "gradient exceeds this many MADs of the panel-wide gradient "
            "(illumination edges, beam-stop shadows)."
        ),
    ] = 8.0,
    texture_factor: Annotated[
        float,
        typer.Option(
            help="Band-pass level criterion, in units of the ambient rate: "
            "mask static structure at scales between the atom footprint and "
            "the background window.  A wide smooth halo vanishes from the "
            "band-pass (the background model follows it, and real peaks "
            "live there); the plume texture and illumination steps remain."
        ),
    ] = 0.15,
    wide_sigma: Annotated[
        float,
        typer.Option(
            help="Long end of the *level* band, px (the glow-texture "
            "scale).  Longer lets plateaus inflate the noise floor and "
            "re-admits dense-diffraction texture."
        ),
    ] = 20.0,
    edge_sigma: Annotated[
        float,
        typer.Option(
            help="Long end of the *contrast* band, px: the scale the "
            "finder's background model can actually follow (its window is "
            "max(15, 5 * max_sigma) px).  Shorter opens a blind gap -- "
            "edges too diffuse for the band yet too sharp for the finder "
            "produced false atoms at 10; their capture saturates at 25."
        ),
    ] = 25.0,
    dilate_px: Annotated[
        int,
        typer.Option(
            help="Dilation of the masked region, px.  Size it to an atom "
            "footprint (~2x the finder's max sigma) so atoms whose tails "
            "rest on the structure are covered."
        ),
    ] = 8,
    static_quantile: Annotated[
        float,
        typer.Option(
            help="Per-pixel quantile across frames that defines the static "
            "map.  Low (default p25) so a dense diffraction pattern cannot "
            "leak into it: a static feature is in every frame and survives "
            "any quantile, a reflection would have to sit still through "
            "more than (100 - q)%% of the scan."
        ),
    ] = 25.0,
    grad_min_frac: Annotated[
        float,
        typer.Option(
            help="Effect-size floor for the boundary criterion, as gradient "
            "per pixel in units of the ambient rate.  Significance (MADs) "
            "alone masks soft genuine variation on flat panels; a real "
            "illumination edge runs percents of ambient per pixel."
        ),
    ] = 0.02,
):
    """Build a static-structure mask from frame stacks of one instrument.

    Beam-stop shadows, illumination boundaries and instrument glow are fixed
    in the detector frame while Bragg peaks move with the sample, so the
    per-bank median across enough frames contains the artifacts and none of
    the crystal.  The output is itself a reduced single-frame stack (uint8,
    1 = valid, one frame per physical bank); feed it to the finder as
    --static-mask-file, which maps it onto its input by bank id.
    """
    run_static_mask(
        output_filename=output_filename,
        input_filenames=list(input_filenames),
        peaks_filenames=list(peaks) if peaks else None,
        pooled_peaks_filename=pooled_peaks,
        peak_deviance_min=peak_deviance_min,
        peak_residual_max=peak_residual_max,
        peak_clear_nsigmas=peak_clear_nsigmas,
        min_frames=min_frames,
        smooth_sigma=smooth_sigma,
        grad_nmads=grad_nmads,
        texture_factor=texture_factor,
        wide_sigma=wide_sigma,
        edge_sigma=edge_sigma,
        dilate_px=dilate_px,
        static_quantile=static_quantile,
        grad_min_frac=grad_min_frac,
    )


@app.command()
def sum_images(
    output_filename: Annotated[str, typer.Argument(help="Summed HDF5 to write")],
    input_filenames: Annotated[
        list[str],
        typer.Argument(
            help="Reduced/merged HDF5 stacks (images + bank_ids), the same "
            "inputs `static-mask` will see."
        ),
    ],
):
    """Sum each bank's deduplicated frames into a one-frame-per-bank stack.

    The companion of `static-mask`: run the finder on the summed stack and
    pass its output back as --pooled-peaks.  Deviance is additive across
    frames, so a quasi-static reflection sitting just below every single
    frame's admission level -- bright enough for the static map to mask,
    too faint for any per-frame certificate -- compounds to certification
    in the pooled fit.  Goniometer angles in the output are placeholders;
    the file exists for peak metrics only, never for indexing.
    """
    run_sum_images(
        output_filename=output_filename,
        input_filenames=list(input_filenames),
    )


@app.command()
def mask_visualize(
    images_filename: Annotated[str, typer.Argument(help=_IMAGES_HELP)],
    mask_filename: Annotated[
        str, typer.Argument(help="Static mask HDF5 (from `static-mask`)")
    ],
    instrument: Annotated[Optional[str], typer.Option(help=_INSTRUMENT_HELP)] = None,
    output_dir: Annotated[Optional[str], typer.Option(help=_OUTPUT_DIR_HELP)] = None,
    dpi: Annotated[int, typer.Option(help=_DPI_HELP)] = 600,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
):
    """
    Draw the static mask over the frames it applies to.

    The same unrolled-detector rendering as finder-visualize, with masked
    pixels burnt to the top of the intensity scale so they read as solid
    regions against the data.  One `<label>-mask.png` per run.
    """
    run_mask_visualize(
        images_filename=images_filename,
        mask_filename=mask_filename,
        instrument=instrument,
        output_dir=output_dir,
        dpi=dpi,
        max_workers=max_workers,
        show_progress=show_progress,
    )


if __name__ == "__main__":
    app()
