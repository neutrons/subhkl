"""
Integration tests for MANDI mesolite data processing workflow.

This test suite validates the complete single-run crystallography workflow:
1. Peak finding (finder)
2. Indexing (indexer)
3. Peak prediction (peak_predictor)
4. Integration (integrator)
5. MTZ export (mtz_exporter)

Test data is automatically downloaded from Zenodo (DOI: 10.5281/zenodo.18475332)
"""

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.ndimage import median_filter

from subhkl.commands import (
    run_finder as finder,
)
from subhkl.commands import (
    run_mtz_exporter as mtz_exporter,
)
from subhkl.commands import (
    run_peak_predictor as peak_predictor,
)
from subhkl.commands import (
    run_rbf_integrator as integrator,
)
from subhkl.commands import (
    run_reduce as reduce,
)
from subhkl.config import beamlines
from subhkl.instrument.detector import Detector
from subhkl.instrument.metrics import compute_metrics
from subhkl.instrument.physics import calculate_angular_error
from subhkl.io.parser import indexer
from subhkl.optimization import FindUB
from subhkl.solve import run_solve
from subhkl.solve_frames import frame_kinematics

# Test data configuration
MESOLITE_FILE = "MANDI_11613.nxs.h5"  # Use the first file from the Zenodo dataset
INSTRUMENT = "MANDI"

# Expected lattice parameters for mesolite (from the bash script)
LATTICE_PARAMS = {
    "a": 18.39,
    "b": 56.55,
    "c": 6.54,
    "alpha": 90.0,
    "beta": 90.0,
    "gamma": 90.0,
}
SPACE_GROUP = "F d d 2"

# Integration parameters (from bash script)
FINDER_PARAMS = {
    "finder_algorithm": "sparse_rbf",
}

# Indexer parameters with explicit values for all typer.Option parameters
# to avoid OptionInfo type errors when calling functions directly
INDEXER_DEFAULTS = {
    "instrument_name": "MANDI",
    "strategy_name": "DE",
    "n_runs": 10,
    "population_size": 1000,
    "gens": 200,
    "seed": 12345,
    "sigma_init": None,
    "refine_lattice": False,
    "lattice_bound_frac": 0.05,
    "refine_goniometer": False,
    "refine_goniometer_axes": None,
    "goniometer_bound_deg": str(5.0),
    "refine_goniometer_trans": False,
    "goniometer_trans_bound_meters": str(2.0),
    "refine_beam": False,
    "beam_bound_deg": 1.0,
    "bootstrap_filename": None,
    "batch_size": 1,
}


@pytest.fixture(name="mesolite_input_file")
def fixture__mesolite_input_file(mesolite_data):
    """Provide path to mesolite test data file."""
    filepath = Path(mesolite_data) / MESOLITE_FILE

    if not filepath.exists():
        pytest.skip(f"Mesolite test file not found: {filepath}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_reduce = os.path.join(tmpdir, "mesolite.im.h5")

        reduce(
            nexus_filename=filepath,
            output_filename=output_reduce,
            instrument=INSTRUMENT,
            wavelength_min=2.0,
            wavelength_max=4.5,
        )

        yield str(output_reduce)


@pytest.fixture(name="temp_output_dir")
def fixture__temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestMandiMesoliteSingleRun:
    """Test suite for single-run MANDI mesolite workflow."""

    @pytest.mark.slow
    def test_full_workflow(self, mesolite_input_file, temp_output_dir):
        """
        Integration test for complete single-run workflow.
        Runs all steps sequentially and halts immediately if any step fails.
        """
        # Define output filenames
        finder_output = os.path.join(temp_output_dir, "mesolite.finder.h5")
        indexer_output = os.path.join(temp_output_dir, "mesolite.indexer.h5")
        predictor_output = os.path.join(temp_output_dir, "mesolite.peak_predictor.h5")
        integrator_output = os.path.join(temp_output_dir, "mesolite.integrator.h5")
        mtz_output = os.path.join(temp_output_dir, "mesolite.mtz")

        print("\n[1/5] Running peak finder...")
        finder(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            output_filename=finder_output,
            **FINDER_PARAMS,
        )
        assert os.path.exists(finder_output), "Finder failed to create output file"

        print("[2/5] Running indexer...")
        indexer(
            peaks_h5_filename=finder_output,
            output_peaks_filename=indexer_output,
            original_nexus_filename=mesolite_input_file,
            a=LATTICE_PARAMS["a"],
            b=LATTICE_PARAMS["b"],
            c=LATTICE_PARAMS["c"],
            alpha=LATTICE_PARAMS["alpha"],
            beta=LATTICE_PARAMS["beta"],
            gamma=LATTICE_PARAMS["gamma"],
            space_group=SPACE_GROUP,
            **INDEXER_DEFAULTS,
        )
        assert os.path.exists(indexer_output), "Indexer failed to create output file"

        # Check the accuracy
        print("[3/5] Validating metrics...")
        metrics = compute_metrics(indexer_output)
        median_ang_err_deg = metrics["median_ang_err"]
        assert median_ang_err_deg < 0.3, (
            f"Indexing accuracy too low: {median_ang_err_deg} deg"
        )

        print("[4/5] Running peak predictor...")
        peak_predictor(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            indexed_hdf5_filename=indexer_output,
            integration_peaks_filename=predictor_output,
            d_min=1.35,
        )
        assert os.path.exists(predictor_output), (
            "Predictor failed to create output file"
        )

        print("[5/5] Running integrator...")
        integrator(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            integration_peaks_filename=predictor_output,
            output_filename=integrator_output,
        )
        assert os.path.exists(integrator_output), (
            "Integrator failed to create output file"
        )

        print("[6/6] Exporting to MTZ...")
        mtz_exporter(
            indexed_h5_filename=integrator_output,
            output_mtz_filename=mtz_output,
            space_group=SPACE_GROUP,
        )
        assert os.path.exists(mtz_output), "MTZ Exporter failed to create output file"

        # Final validation: Check we have reflections in peaks group
        with h5py.File(integrator_output, "r") as f:
            assert "peaks" in f, "Missing peaks group in final integrator output"
            peaks = f["peaks"]
            total_reflections = len(peaks["h"]) if "h" in peaks else 0

        print("\n✓ Complete workflow finished successfully")
        print(f"  Total reflections: {total_reflections}")
        print(f"  Output files in: {temp_output_dir}")

    @pytest.mark.slow
    def test_solve_workflow(self, mesolite_input_file, temp_output_dir):
        """
        Integration test for the finder-free single-run workflow using `solve`:
        reduce -> solve -> peak_predictor -> integrator -> mtz_exporter.

        Without `finder` peaks, indexing quality is validated via:
        1. `solve` internal convergence, Poisson objective reduction, and sparse
           group-norm support (`solve/group_norms`).
        2. Direct geometric angular error (`calculate_angular_error`) between `U @ B`
           and local detector spot centroids measured in small windows around the
           predicted reflection positions on the raw reduced images.
        3. Downstream integrated reflection signal-to-noise (`I / sigma`) from the
           RBF integrator on unbinned detector panels.
        """
        solve_output = os.path.join(temp_output_dir, "mesolite.solve.h5")
        predictor_output = os.path.join(
            temp_output_dir, "mesolite.solve.peak_predictor.h5"
        )
        integrator_output = os.path.join(
            temp_output_dir, "mesolite.solve.integrator.h5"
        )
        mtz_output = os.path.join(temp_output_dir, "mesolite.solve.mtz")

        print("\n[1/4] Running finder-free solve...")
        solve_result = run_solve(
            mesolite_input_file,
            solve_output,
            instrument=INSTRUMENT,
            cell=[
                LATTICE_PARAMS["a"],
                LATTICE_PARAMS["b"],
                LATTICE_PARAMS["c"],
                LATTICE_PARAMS["alpha"],
                LATTICE_PARAMS["beta"],
                LATTICE_PARAMS["gamma"],
            ],
            space_group=SPACE_GROUP,
            wavelength_min=2.0,
            wavelength_max=4.5,
            d_min=2.0,
            n_candidates=1,
            fixed_geometry=True,
        )
        assert os.path.exists(solve_output), "Solve failed to create output file"
        assert solve_result["status"] == "converged", (
            f"Solve did not converge: {solve_result['status']}"
        )

        with h5py.File(solve_output, "r") as f:
            assert "sample/U" in f, "Missing sample/U in solve output"
            U = f["sample/U"][()]
            assert np.allclose(U.T @ U, np.eye(3), atol=1e-5)
            assert np.isclose(np.linalg.det(U), 1.0, atol=1e-5)
            assert bool(f["solve/inner_converged"][()]), "Inner solve did not converge"
            obj_init = float(f["solve/objective_initial"][()])
            obj_final = float(f["solve/objective_final"][()])
            group_norms = f["solve/group_norms"][()]
            assert obj_final < obj_init, (
                f"Expected Poisson refinement to reduce objective ({obj_init} -> {obj_final})"
            )
            assert float(np.max(group_norms)) > 500.0, (
                f"Expected strong reflection group norm support, got {group_norms}"
            )

        print("[2/4] Running peak predictor...")
        peak_predictor(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            indexed_hdf5_filename=solve_output,
            integration_peaks_filename=predictor_output,
            d_min=1.35,
        )
        assert os.path.exists(predictor_output), (
            "Predictor failed to create output file"
        )

        # Finder-free geometric validation: compare U @ B against local Bragg spot
        # centroids on the raw detector panels around predicted reflections.
        matched_spots, median_ang_err_deg = _compute_finder_free_angular_error(
            mesolite_input_file, solve_output, predictor_output, INSTRUMENT
        )
        print(
            f"  Finder-free spot centroid check: {matched_spots} spots, "
            f"median angular error = {median_ang_err_deg:.4f} deg"
        )
        assert matched_spots >= 100, (
            f"Too few predicted reflections matched detector spots: {matched_spots}"
        )
        assert median_ang_err_deg < 0.25, (
            f"Finder-free geometric angular error too high: {median_ang_err_deg:.4f} deg"
        )

        print("[3/4] Running integrator...")
        integrator(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            integration_peaks_filename=predictor_output,
            output_filename=integrator_output,
        )
        assert os.path.exists(integrator_output), (
            "Integrator failed to create output file"
        )

        print("[4/4] Exporting to MTZ...")
        mtz_exporter(
            indexed_h5_filename=integrator_output,
            output_mtz_filename=mtz_output,
            space_group=SPACE_GROUP,
        )
        assert os.path.exists(mtz_output), "MTZ Exporter failed to create output file"

        # Validate integrated reflection quality (a misindexed orientation yields
        # median I/sigma < 1.0 and < 5 reflections with I/sigma > 3.0).
        with h5py.File(integrator_output, "r") as f:
            assert "peaks" in f, "Missing peaks group in final integrator output"
            peaks = f["peaks"]
            total_reflections = len(peaks["h"]) if "h" in peaks else 0
            intensities = peaks["intensity"][()]
            sigmas = peaks["sigma"][()]

        valid = (sigmas > 0) & np.isfinite(intensities) & np.isfinite(sigmas)
        snr = intensities[valid] / sigmas[valid]
        n_strong_3sig = int(np.sum(snr > 3.0))
        n_strong_5sig = int(np.sum(snr > 5.0))
        median_snr = float(np.median(snr))

        assert total_reflections >= 150, (
            f"Expected >= 150 integrated reflections, got {total_reflections}"
        )
        assert n_strong_3sig >= 100, (
            f"Expected >= 100 reflections with I/sigma > 3, got {n_strong_3sig}"
        )
        assert n_strong_5sig >= 80, (
            f"Expected >= 80 reflections with I/sigma > 5, got {n_strong_5sig}"
        )
        assert median_snr > 5.0, (
            f"Expected median integrated I/sigma > 5.0, got {median_snr:.2f}"
        )

        print("\n✓ Finder-free solve workflow finished successfully")
        print(
            f"  Total reflections: {total_reflections} "
            f"(I/sigma > 3: {n_strong_3sig}, median I/sigma: {median_snr:.2f})"
        )


def _compute_finder_free_angular_error(
    images_h5: str,
    solve_h5: str,
    predictor_h5: str,
    instrument: str,
    window_radius_px: int = 10,
    min_local_snr: float = 6.0,
) -> tuple[int, float]:
    """Measure geometric angular error against local image spot centroids without finder."""
    with h5py.File(images_h5, "r") as f:
        images = f["images"][()].astype(float)
        bank_ids = f["bank_ids"][()].astype(int)
        R0 = frame_kinematics(f, len(bank_ids), None)[0][0]

    bank_to_img = {int(b): images[i] for i, b in enumerate(bank_ids)}
    bank_to_bg = {
        int(b): np.maximum(median_filter(images[i], size=15), 1.0)
        for i, b in enumerate(bank_ids)
    }

    with h5py.File(solve_h5, "r") as f:
        U = f["sample/U"][()]
        ub = FindUB()
        ub.a, ub.b, ub.c = f["sample/a"][()], f["sample/b"][()], f["sample/c"][()]
        ub.alpha, ub.beta, ub.gamma = (
            f["sample/alpha"][()],
            f["sample/beta"][()],
            f["sample/gamma"][()],
        )
        B = ub.reciprocal_lattice_B()

    xyz_obs, h_list, k_list, l_list, lam_list = [], [], [], [], []
    rad = window_radius_px
    with h5py.File(predictor_h5, "r") as f:
        pred_bank_ids = f["bank_ids"][()].astype(int)
        for key_str in f["banks"]:
            grp = f[f"banks/{key_str}"]
            bank = int(pred_bank_ids[int(key_str)])
            im = bank_to_img[bank]
            bg = bank_to_bg[bank]
            z_map = (im - bg) / np.sqrt(bg)
            i_arr, j_arr = grp["i"][()], grp["j"][()]
            h_arr, k_arr, l_arr, lam_arr = (
                grp["h"][()],
                grp["k"][()],
                grp["l"][()],
                grp["wavelength"][()],
            )
            det = Detector(dict(beamlines[instrument][str(bank)]))
            for idx_p in range(len(i_arr)):
                r0 = round(float(i_arr[idx_p]))
                c0 = round(float(j_arr[idx_p]))
                if (
                    r0 < rad
                    or r0 >= im.shape[0] - rad
                    or c0 < rad
                    or c0 >= im.shape[1] - rad
                ):
                    continue
                z_patch = z_map[r0 - rad : r0 + rad + 1, c0 - rad : c0 + rad + 1]
                if z_patch.max() < min_local_snr:
                    continue
                mr, mc = np.unravel_index(np.argmax(z_patch), z_patch.shape)
                rp, cp = r0 - rad + mr, c0 - rad + mc
                sub = np.maximum(
                    im[rp - 1 : rp + 2, cp - 1 : cp + 2]
                    - bg[rp - 1 : rp + 2, cp - 1 : cp + 2],
                    0.0,
                )
                rr, cc = np.mgrid[rp - 1 : rp + 2, cp - 1 : cp + 2]
                denom = max(float(sub.sum()), 1e-9)
                rc = float((rr * sub).sum() / denom)
                cc_ = float((cc * sub).sum() / denom)
                xyz_obs.append(det.pixel_to_lab(np.array([rc]), np.array([cc_]))[0])
                h_list.append(h_arr[idx_p])
                k_list.append(k_arr[idx_p])
                l_list.append(l_arr[idx_p])
                lam_list.append(lam_arr[idx_p])

    if not xyz_obs:
        return 0, float("inf")

    _, ang_errs = calculate_angular_error(
        np.asarray(xyz_obs),
        np.asarray(h_list),
        np.asarray(k_list),
        np.asarray(l_list),
        np.asarray(lam_list),
        U @ B,
        R_all=R0,
    )
    return len(xyz_obs), float(np.median(ang_errs))


# Mark the test class for pytest markers. `mesolite` keeps these tests, and the
# hours-long dataset download they need, out of a default test run.
pytestmark = [pytest.mark.integration, pytest.mark.mesolite]
