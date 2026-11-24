import os
import urllib.request
from pathlib import Path

import matplotlib
import pytest

# Use non-interactive backend for tests to prevent plot windows from popping up
matplotlib.use("Agg")


# Zenodo dataset configuration
ZENODO_RECORD_ID = "18475332"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
DATA_DIR = Path(__file__).parent / "data" / "MANDI" / "mesolite"

# List of all files on Zenodo (generated from the actual upload)
ZENODO_FILES = [f"MANDI_{i}.nxs.h5" for i in range(11613, 11685)]


def download_mesolite_data() -> Path:
    """Download mesolite test data from Zenodo if not already present.

    Set MESOLITE_MAX_FILES environment variable to limit number of files downloaded (for testing).
    Example: MESOLITE_MAX_FILES=5 uv run pytest -m mesolite
    """
    max_files = os.environ.get("MESOLITE_MAX_FILES")
    max_files = int(max_files) if max_files is not None else len(ZENODO_FILES)

    files_to_download = ZENODO_FILES[:max_files]

    # Check which files we are missing
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = {f.name for f in DATA_DIR.glob("*.h5")}
    needed_files = [f for f in files_to_download if f not in existing_files]

    if not needed_files:
        print(
            f"All {len(files_to_download)} requested mesolite test files already exist at {DATA_DIR}"
        )
        return DATA_DIR

    print(
        f"Downloading {len(needed_files)} missing mesolite files from Zenodo (record {ZENODO_RECORD_ID})..."
    )

    # Download each file
    for i, filename in enumerate(needed_files, 1):
        file_url = f"{ZENODO_BASE_URL}/{filename}"
        file_path = DATA_DIR / filename

        try:
            print(f"  [{i}/{len(needed_files)}] Downloading {filename}...")
            urllib.request.urlretrieve(file_url, file_path)
        except (OSError, urllib.error.URLError) as e:
            print(f"  Warning: Failed to download {filename}: {e}")
            continue

    downloaded_count = len(list(DATA_DIR.glob("*.h5")))
    print(f"✓ Downloaded {downloaded_count} files to {DATA_DIR}")
    return DATA_DIR


@pytest.fixture(scope="session", name="mesolite_data")
def fixture__mesolite_data() -> Path:
    """Directory holding the mesolite dataset, downloading it on first use.

    Downloading the dataset takes hours, so it happens only for the tests that
    ask for this fixture. Those tests carry the ``mesolite`` marker, which is
    deselected by default (see ``addopts`` in pyproject.toml); run them with
    ``pytest -m mesolite``.
    """
    return download_mesolite_data()


@pytest.fixture(name="test_data_dir")
def fixture__test_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """Reclaim JAX XLA CPU compilation caches and glibc arenas so 16 GiB CI runners do not OOM.

    ``MatrixFreeSparseRBFPeakFinder`` jits bound methods per instance, so each test in
    ``test_sparse_rbf_integrator.py`` otherwise pins un-reusable XLA executables and
    multi-GB LLVM/Eigen buffers in process RSS until ``test_large_sensor_*`` triggers a
    silent Linux OOM SIGKILL right after ``test_real_neutron_structured_background``.
    """
    import contextlib
    import ctypes
    import gc
    import sys

    boundary = nextitem is None or item.path != nextitem.path
    rbf_suite = item.path.name == "test_sparse_rbf_integrator.py"
    heavy_sensor = "large_sensor" in item.name or (
        nextitem is not None and "large_sensor" in nextitem.name
    )
    if boundary or rbf_suite or heavy_sensor:
        jax_mod = sys.modules.get("jax")
        if jax_mod is not None and hasattr(jax_mod, "clear_caches"):
            jax_mod.clear_caches()
        gc.collect()
        if sys.platform.startswith("linux"):
            with contextlib.suppress(OSError, AttributeError):
                ctypes.CDLL("libc.so.6").malloc_trim(0)


@pytest.fixture(name="meso_tiff")
def fixture__meso_tiff(test_data_dir) -> str:
    return os.path.join(test_data_dir, "meso_2_15min_2-0_4-5_050.tif")
