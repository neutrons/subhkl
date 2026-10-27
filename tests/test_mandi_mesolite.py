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
import pytest

from subhkl.io.parser import indexer
from subhkl.commands import (
    run_reduce as reduce,
    run_finder as finder,
    run_rbf_integrator as integrator,
    run_mtz_exporter as mtz_exporter,
    run_peak_predictor as peak_predictor,
)
from subhkl.instrument.metrics import compute_metrics

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
    "batch_size": None,
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


# Mark the test class for pytest markers. `mesolite` keeps these tests, and the
# hours-long dataset download they need, out of a default test run.
pytestmark = [pytest.mark.integration, pytest.mark.mesolite]
