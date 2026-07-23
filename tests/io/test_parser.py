import pytest
from typer.testing import CliRunner

from subhkl.io.parser import app, finder, _maybe_export_dials

# Every subhkl CLI command should expose the DIALS-export options.
DIALS_COMMANDS = [
    "finder",
    "indexer",
    "rbf-integrator",
    "metrics",
    "peak-predictor",
    "integrator",
    "mtz-exporter",
    "reduce",
    "merge-images",
    "zone-axis-search",
]


@pytest.mark.parametrize("command", DIALS_COMMANDS)
def test_dials_flag_is_wired(command):
    """Every CLI command exposes the --dials and --dials-prefix options.

    A wide terminal keeps Rich from wrapping the option names apart.
    """
    result = CliRunner().invoke(app, [command, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--dials" in result.stdout
    assert "--dials-prefix" in result.stdout


def test_maybe_export_dials_without_dials_warns(tmp_path, capsys):
    """Without DIALS installed, --dials warns and leaves native output intact.

    The program's own output is written before the export runs, so a missing
    toolkit must degrade to a clear warning for the --dials step only rather than
    crashing the command.
    """
    try:
        import dxtbx  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("DIALS is installed; missing-DIALS path not exercised")

    native = tmp_path / "indexed.h5"
    native.write_bytes(b"native output")  # stand-in for the program's own output

    # Must not raise even though dxtbx is unavailable.
    _maybe_export_dials(True, str(native), str(native), instrument="CG4D")

    captured = capsys.readouterr()
    assert "Skipping --dials export" in captured.err
    # No .expt/.refl were produced, and the native output is untouched.
    assert not (tmp_path / "indexed.expt").exists()
    assert not (tmp_path / "indexed.refl").exists()
    assert native.read_bytes() == b"native output"


def test_maybe_export_dials_disabled_is_noop(tmp_path, capsys):
    """With the flag off, nothing is written and nothing is printed."""
    native = tmp_path / "indexed.h5"
    native.write_bytes(b"native output")

    _maybe_export_dials(False, str(native), str(native), instrument="CG4D")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert not (tmp_path / "indexed.expt").exists()


@pytest.mark.skip(reason="finder function returns None, not 0 - needs fixing")
def test_finder_function_tiff(meso_tiff):
    instrument = "IMAGINE"
    result = finder(
        filename=meso_tiff,
        instrument=instrument,
    )
    assert result == 0


@pytest.mark.skip(reason="CliRunner usage issue - needs fixing")
def test_find_args_tiff(meso_tiff):
    runner = CliRunner(meso_tiff)
    test_args = ["finder", meso_tiff, "IMAGINE", "output.h5"]
    result = runner.invoke(app, test_args)
    assert result.exit_code == 0

    output = result.stdout.rstrip()
    expected_output = ""
    assert expected_output in output
