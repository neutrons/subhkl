"""Tests that a plain `pytest` run does not download the mesolite dataset.

Downloading it takes hours, which is more than a first `pytest` after a fresh
install should ever do, so the tests that need the dataset are marked
``mesolite`` and deselected by default.
"""

import subprocess
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_TEST_FILES = ["tests/test_mandi_mesolite.py", "tests/test_mandi_multi_run.py"]


def _pytest(*arguments):
    """Run pytest in a subprocess, so it reads the settings of this repository."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_tests_needing_the_dataset_are_deselected_by_default():
    result = _pytest("--collect-only", "-q", *DATA_TEST_FILES)

    assert "no tests collected" in result.stdout, result.stdout
    assert "3 deselected" in result.stdout, result.stdout


def test_tests_needing_the_dataset_can_be_asked_for():
    result = _pytest("--collect-only", "-q", "-m", "mesolite", *DATA_TEST_FILES)

    assert "test_full_workflow" in result.stdout, result.stdout
    assert "test_mandi_multi_run_finder_merger" in result.stdout, result.stdout


def test_a_default_run_never_sets_up_the_download():
    """The download fixture must not be requested by tests that do not need it."""
    result = _pytest("--setup-plan", "-q", "tests/test_utils.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "mesolite" not in result.stdout, result.stdout
