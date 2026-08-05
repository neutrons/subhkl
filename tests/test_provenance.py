"""Tests for the run and build information recorded by every subhkl step."""

import json
import re
import sys

import h5py
import pytest
import typer

from typer.testing import CliRunner

from subhkl.utils import provenance


@pytest.fixture(autouse=True)
def _clear_build_info_cache():
    provenance.build_info.cache_clear()
    yield
    provenance.build_info.cache_clear()


def test_build_info_reports_version_and_commit():
    info = provenance.build_info()

    assert info["subhkl_version"]
    assert info["git_commit"]
    # subhkl is developed and tested from a git checkout, so the commit the
    # code was built from must be resolvable here.
    assert re.fullmatch(r"[0-9a-f]{40}", info["git_commit"])
    assert info["git_commit_source"] == "git"


def test_build_info_prefers_the_environment(monkeypatch):
    monkeypatch.setenv(provenance.COMMIT_ENV_VAR, "0123456789abcdef")

    info = provenance.build_info()

    assert info["git_commit"] == "0123456789abcdef"
    assert info["git_commit_source"] == "environment"


def test_record_captures_command_line_and_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["parser.py", "finder", "--max-workers", "4"])

    record = provenance.new_record("finder", {"max_workers": 4, "mask_file": None})

    assert record["command"] == "finder"
    assert record["command_line"].endswith("finder --max-workers 4")
    assert record["parameters"] == {"max_workers": 4, "mask_file": None}
    assert record["subhkl_version"] == provenance.build_info()["subhkl_version"]
    assert record["timestamp_utc"]
    assert record["record_id"]


def test_command_line_is_recorded_the_way_it_is_typed(monkeypatch):
    """subhkl is started as ``python -m subhkl.io.parser``, not as a script."""

    class _Main:
        __spec__ = type("spec", (), {"name": "subhkl.io.parser"})

    monkeypatch.setitem(sys.modules, "__main__", _Main())
    monkeypatch.setattr(sys, "argv", ["/some/where/parser.py", "reduce", "in.nxs"])

    assert provenance.command_line().endswith(
        "-m subhkl.io.parser reduce in.nxs"
    )


def test_record_renders_values_that_hdf5_cannot_hold():
    record = provenance.new_record("indexer", {"ki_vec": [0.0, 0.0, 1.0], "obj": object()})

    assert record["parameters"]["ki_vec"] == [0.0, 0.0, 1.0]
    assert isinstance(record["parameters"]["obj"], str)


def test_format_record_lists_the_options():
    record = provenance.new_record("reduce", {"instrument": "MANDI"})

    text = provenance.format_record(record)

    assert "subhkl" in text
    assert "commit" in text
    assert "instrument" in text
    assert "MANDI" in text


def _make_app(tmp_path):
    """A miniature two step pipeline built like the real CLI commands."""
    app = typer.Typer()

    @app.command()
    @provenance.track(outputs=("output_filename",), inputs=("filename",))
    def first(filename: str, output_filename: str, threshold: float = 2.5):
        with h5py.File(output_filename, "w") as handle:
            handle["result"] = [1, 2, 3]

    @app.command()
    @provenance.track(outputs=("output_mtz_filename",), inputs=("input_filename",))
    def second(input_filename: str, output_mtz_filename: str):
        with open(output_mtz_filename, "w") as handle:
            handle.write("not really an mtz")

    @app.command()
    @provenance.track(outputs=("output_filename",), inputs=("filename",))
    def crashing(filename: str, output_filename: str):
        with h5py.File(output_filename, "w") as handle:
            handle["partial"] = [1]
        raise RuntimeError("no peaks found")

    return app


def test_step_prints_and_stores_its_invocation(tmp_path):
    output = tmp_path / "first.h5"
    result = CliRunner().invoke(
        _make_app(tmp_path), ["first", "input.nxs", str(output), "--threshold", "9.5"]
    )
    assert result.exit_code == 0, result.output

    # printed for the user running the step
    assert "commit" in result.output
    assert "threshold" in result.output

    # and stored alongside the data it produced
    records = provenance.read_records(output)
    assert len(records) == 1
    assert records[0]["command"] == "first"
    assert records[0]["parameters"]["threshold"] == 9.5
    assert records[0]["parameters"]["filename"] == "input.nxs"
    assert records[0]["git_commit"] == provenance.build_info()["git_commit"]
    assert records[0]["duration_seconds"] >= 0


def test_stored_options_include_untouched_defaults(tmp_path):
    output = tmp_path / "first.h5"
    CliRunner().invoke(_make_app(tmp_path), ["first", "input.nxs", str(output)])

    records = provenance.read_records(output)

    assert records[0]["parameters"]["threshold"] == 2.5


def test_provenance_group_is_readable_without_subhkl(tmp_path):
    output = tmp_path / "first.h5"
    CliRunner().invoke(_make_app(tmp_path), ["first", "input.nxs", str(output)])

    with h5py.File(output, "r") as handle:
        group = handle["provenance/0000_first"]
        assert group.attrs["command"] == "first"
        assert group.attrs["command_line"]
        assert group.attrs["git_commit"]
        assert group.parent.name == "/provenance"
        assert group["parameters"].attrs["threshold"] == 2.5
        assert json.loads(group.attrs["record_json"])["command"] == "first"


def test_step_does_not_disturb_the_data_it_stamps(tmp_path):
    output = tmp_path / "first.h5"
    CliRunner().invoke(_make_app(tmp_path), ["first", "input.nxs", str(output)])

    with h5py.File(output, "r") as handle:
        assert list(handle["result"][()]) == [1, 2, 3]


def test_a_chain_of_steps_accumulates_provenance(tmp_path):
    app = _make_app(tmp_path)
    first_output = tmp_path / "first.h5"
    second_output = tmp_path / "second.mtz"

    CliRunner().invoke(app, ["first", "input.nxs", str(first_output)])
    result = CliRunner().invoke(
        app, ["second", str(first_output), str(second_output)]
    )
    assert result.exit_code == 0, result.output

    # the mtz is not an HDF5 file, so its provenance lands in a sidecar
    sidecar = second_output.with_name(second_output.name + provenance.SIDECAR_SUFFIX)
    assert sidecar.is_file()

    records = provenance.read_records(second_output)
    assert [record["command"] for record in records] == ["first", "second"]


def test_rerunning_a_step_does_not_duplicate_records(tmp_path):
    app = _make_app(tmp_path)
    output = tmp_path / "first.h5"

    CliRunner().invoke(app, ["first", "input.nxs", str(output)])
    CliRunner().invoke(app, ["first", "input.nxs", str(output)])

    assert len(provenance.read_records(output)) == 1


def test_a_failed_step_records_why_it_failed(tmp_path):
    output = tmp_path / "partial.h5"
    result = CliRunner().invoke(
        _make_app(tmp_path), ["crashing", "input.nxs", str(output)]
    )
    assert result.exit_code != 0

    records = provenance.read_records(output)

    assert records[0]["status"] == "failed"
    assert "no peaks found" in records[0]["error"]
    assert "failed" in provenance.format_record(records[0])


def test_a_completed_step_is_recorded_as_such(tmp_path):
    output = tmp_path / "first.h5"
    CliRunner().invoke(_make_app(tmp_path), ["first", "input.nxs", str(output)])

    assert provenance.read_records(output)[0]["status"] == "completed"


def test_an_input_that_is_not_hdf5_is_not_a_problem(tmp_path):
    """Steps also read raw images, such as the .tif files of IMAGINE."""
    image = tmp_path / "image.tif"
    image.write_bytes(b"II*\x00 not an hdf5 file")
    output = tmp_path / "first.h5"

    result = CliRunner().invoke(
        _make_app(tmp_path), ["first", str(image), str(output)]
    )

    assert result.exit_code == 0, result.output
    assert len(provenance.read_records(output)) == 1


def test_reading_a_file_without_provenance(tmp_path):
    plain = tmp_path / "plain.h5"
    with h5py.File(plain, "w") as handle:
        handle["data"] = [1]

    assert provenance.read_records(plain) == []
    assert provenance.read_records(tmp_path / "missing.h5") == []


def test_a_failing_stamp_does_not_lose_the_result(tmp_path, capsys):
    record = provenance.new_record("first", {})

    provenance.stamp(tmp_path, record)  # a directory, not a file

    assert provenance.read_records(tmp_path / "missing.h5") == []


def test_cli_reports_version_and_stored_provenance(tmp_path):
    from subhkl.io.parser import app as cli

    runner = CliRunner()

    version = runner.invoke(cli, ["version"])
    assert version.exit_code == 0
    assert provenance.build_info()["git_commit"][:12] in version.output

    output = tmp_path / "first.h5"
    CliRunner().invoke(_make_app(tmp_path), ["first", "input.nxs", str(output)])

    shown = runner.invoke(cli, ["provenance", str(output)])
    assert shown.exit_code == 0
    assert "step 0" in shown.output
    assert "first" in shown.output

    as_json = runner.invoke(cli, ["provenance", str(output), "--json"])
    assert as_json.exit_code == 0
    assert json.loads(as_json.output)[0]["command"] == "first"

    missing = runner.invoke(cli, ["provenance", str(tmp_path / "nope.h5")])
    assert missing.exit_code == 1
