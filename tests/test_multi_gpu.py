"""Multi-GPU sharding: opt-in, and equivalent to the single-device answer.

The solve tests force the CPU backend to present several devices (XLA_FLAGS
--xla_force_host_platform_device_count) -- the sharding machinery is the same
GSPMD path a GPU mesh takes, and forcing it keeps the test independent of the
hardware.  That flag must be set before JAX initializes, which is why the
tests run in a subprocess rather than in this process, where other tests have
long since created arrays.

The forced-CPU solve is too slow for a starved shared CPU-only CI runner,
though, so the subprocess tests skip wherever JAX's default backend is the
CPU and run only on machines with an accelerator attached (see
``_requires_accelerator``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from subhkl.utils import devices as device_util


def test_multi_gpu_is_opt_in():
    """Without the flag exactly one device is used, whatever is visible."""
    assert len(device_util.batch_devices(False)) == 1
    assert len(device_util.batch_devices(True)) >= 1


def test_pad_to_multiple_repeats_the_last_element():
    import jax.numpy as jnp

    array = jnp.asarray(np.arange(12.0).reshape(3, 2, 2))
    padded = device_util.pad_to_multiple(array, 2)
    assert padded.shape == (4, 2, 2)
    np.testing.assert_array_equal(padded[3], padded[2])
    # Already divisible: returned untouched, no copy of the batch axis.
    assert device_util.pad_to_multiple(array, 3) is array
    assert device_util.pad_to_multiple(array, 1) is array


#: Both subprocess tests run the forced-two-device CPU solve inside a fixed
#: 20-minute budget.  On an idle machine they finish in about a minute, but a
#: starved shared CPU-only CI runner can burn the whole budget with no defect
#: present -- and waiting out the timeout just to record an xfail costs those
#: 20 minutes on every CI run.  So the tests skip up front wherever no
#: accelerator is attached: a visible GPU/TPU marks a machine with the
#: headroom to finish in minutes.  (The solve itself still runs on the
#: forced-two-device CPU backend either way; the accelerator is only the
#: capability signal, not the device under test.)
def _cpu_only_machine() -> bool:
    import jax

    return jax.default_backend() == "cpu"


_requires_accelerator = pytest.mark.skipif(
    _cpu_only_machine(),
    reason=(
        "no accelerator attached: the forced-two-device CPU solve overruns "
        "its 20-minute budget on starved CPU-only CI runners (it finishes in "
        "~1 minute on an idle machine)"
    ),
)


_EQUIVALENCE_SCRIPT = textwrap.dedent(
    """
    import json
    import numpy as np
    from scipy.special import erf

    def gaussian(shape, r0, c0, sigma, amp):
        rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
        s2 = sigma * np.sqrt(2.0)
        er = erf((rr - r0 + 0.5) / s2) - erf((rr - r0 - 0.5) / s2)
        ec = erf((cc - c0 + 0.5) / s2) - erf((cc - c0 - 0.5) / s2)
        return amp * (np.pi / 2.0) * sigma**2 * er * ec

    rng = np.random.default_rng(3)
    frames = []
    # Three images: an odd count, so two devices force the padding path.
    for k in range(3):
        frame = np.full((72, 72), 1.0)
        frame += gaussian(frame.shape, 20.0 + k, 24.0, 2.0, 60.0)
        frame += gaussian(frame.shape, 50.0, 46.0 - k, 2.5, 60.0)
        frames.append(rng.poisson(frame).astype(float))
    stack = np.stack(frames)

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    def run(multi_gpu):
        finder = MatrixFreeSparseRBFPeakFinder(
            min_sigma=1.5,
            max_sigma=3.0,
            num_sigmas=3,
            profile_file="gaussian",
            shape_ratio=1.0,
            # chunk 2 over 3 images: the single-device run takes two chunks,
            # the two-device run one padded, sharded chunk.
            chunk_size=2,
            multi_gpu=multi_gpu,
        )
        peaks = finder.find_peaks_batch(stack)
        return [np.asarray(p)[:, :4].tolist() for p in peaks]

    import jax

    n_dev = len(jax.devices())
    result = {
        "n_devices": n_dev,
        "single": run(False),
        "sharded": run(True),
    }
    print("RESULT " + json.dumps(result))
    """
)


_INDEXER_SCRIPT = textwrap.dedent(
    """
    import sys

    import jax
    import numpy as np

    from subhkl.optimization import FindUB

    opt = FindUB(filename=sys.argv[1])
    # 3 runs in batches of 2 on 2 devices: both the batch size and the run
    # count are indivisible, so this exercises the round-up path (4 runs in
    # batches of 2) as well as the sharded init/step.
    num, hkl, lamda, U = opt.minimize(
        "DE",
        population_size=64,
        num_generations=3,
        n_runs=3,
        batch_size=2,
        multi_gpu=True,
    )
    print(f"RESULT devices={len(jax.devices())} num={int(num)} U={np.shape(U)}")
    """
)


@_requires_accelerator
@pytest.mark.filterwarnings("ignore")
def test_sharded_indexer_runs_and_returns_a_solution(tmp_path):
    """Smoke, not equivalence: DE is stochastic across launch geometries by
    design (the round-up launches extra runs), so the claim under test is that
    the sharded path runs end to end and produces a well-formed solution."""

    from tests.test_indexer_synthetic import create_synthetic_finder

    finder_h5 = str(tmp_path / "synthetic_finder.h5")
    create_synthetic_finder(finder_h5)

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    ).strip()
    proc = subprocess.run(
        [sys.executable, "-c", _INDEXER_SCRIPT, finder_h5],
        capture_output=True,
        text=True,
        timeout=1200,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, proc.stdout[-2000:]
    assert "devices=2" in line[0]
    assert "U=(3, 3)" in line[0]


@_requires_accelerator
@pytest.mark.filterwarnings("ignore")
def test_sharded_solve_matches_the_single_device_answer():
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    ).strip()
    proc = subprocess.run(
        [sys.executable, "-c", _EQUIVALENCE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=1200,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, proc.stdout[-2000:]
    result = json.loads(line[0][len("RESULT ") :])
    assert result["n_devices"] == 2

    for single, sharded in zip(result["single"], result["sharded"], strict=True):
        single = np.asarray(single)
        sharded = np.asarray(sharded)
        # Two bright, well-separated peaks per image: the chunk shapes differ
        # between the two runs (that is the point), so demand agreement to
        # comfortably sub-pixel/sub-percent rather than bit-identity -- the
        # same tolerance chunking itself is documented to need.
        assert single.shape == sharded.shape
        order_a = np.lexsort((single[:, 2], single[:, 1]))
        order_b = np.lexsort((sharded[:, 2], sharded[:, 1]))
        np.testing.assert_allclose(
            single[order_a], sharded[order_b], rtol=0.02, atol=0.05
        )
