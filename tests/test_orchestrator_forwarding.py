"""The orchestrator is the only place the CLI's finder options become a finder.

Every unit test that exercises an option builds MatrixFreeSparseRBFPeakFinder
directly, so an option the orchestrator drops stays green in all of them while
being dead on every command line.  That is exactly what happened: the class
defaults are sensible and the constructor ends in **kwargs, so nothing crashed
when --sparse-rbf-profile-file gaussian (the documented opt-out of the learned
atom family) and --sparse-rbf-chunk-size (the documented memory knob) were
silently ignored.  This test goes through prepare_harvest_tasks, the path the
CLI takes.
"""

import inspect

import numpy as np

from subhkl.integration import orchestrator
from subhkl.integration.image_data import ImageData
from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder


class _RecordingFinder:
    received: dict = {}

    def __init__(self, **kwargs):
        _RecordingFinder.received = kwargs

    def find_peaks_batch(self, stack):
        return [np.zeros((0, 4)) for _ in range(stack.shape[0])]


def test_every_forwarded_finder_option_reaches_the_constructor(monkeypatch):
    monkeypatch.setattr(orchestrator, "MatrixFreeSparseRBFPeakFinder", _RecordingFinder)
    # Values deliberately different from every default, so a dropped key
    # cannot pass by coincidence.
    options = {
        "alpha": 3.5,
        "gamma": 0.25,
        "loss": "poisson",
        "min_sigma": 1.25,
        "max_sigma": 4.5,
        "num_sigmas": 7,
        "false_alarms_per_image": 0.5,
        "max_fragmentation_rate": 2.0,
        "show_steps": True,
        "profile_file": "gaussian",
        "shape_ratio": 1.0,
        "shape_orientations": 2,
        "chunk_size": 7,
        "multi_gpu": True,
    }
    image_data = ImageData(ims={0: np.zeros((8, 8))})
    tasks, _ = orchestrator.prepare_harvest_tasks(
        image_data,
        "CG4D",
        None,  # goniometer: untouched, bank 0 is not a CG4D bank
        None,  # wavelength: likewise
        harvest_peaks_kwargs={"algorithm": "sparse_rbf", **options},
        integration_params={},
    )
    assert tasks == []  # the finder ran; no bank survived to become a task

    received = _RecordingFinder.received
    for key, value in options.items():
        assert received.get(key) == value, f"{key} did not reach the finder"

    # Nothing forwarded may fall into the constructor's **kwargs, where it
    # would be swallowed without effect -- the failure mode this file exists
    # to prevent.
    params = inspect.signature(MatrixFreeSparseRBFPeakFinder.__init__).parameters
    for key in received:
        assert key in params, f"{key} is not a real constructor parameter"
