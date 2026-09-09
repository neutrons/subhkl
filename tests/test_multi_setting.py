import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.search.multi_setting import MultiSettingModel
from subhkl.search.poisson_orientation import (
    CountData,
    OrientationModel,
    fit_intensities,
)


def test_joint_design_frames_grouping_and_background_statistics():
    models = []
    for i in range(2):
        data = CountData.from_images(np.zeros((2, 512, 512)), [62, 55], [0.5, 0.7], 16)
        models.append(
            OrientationModel(data, "CG4D", [12] * 3 + [90] * 3, "P 1", (2, 10), 2, 6)
        )
    rotations = Rotation.from_rotvec(np.radians([[0, 0, 0], [0, 35, 0]])).as_matrix()
    joint = MultiSettingModel(models, rotations)
    U = Rotation.random(2, random_state=8).as_matrix()
    g = np.array([0.025, 0.008, -0.006, 0.001, -0.0015, 0.002])
    A = joint.design(U, g)
    rng = np.random.default_rng(4)
    flux = rng.uniform(0, 1000, (2, 2, models[0].n_hkl))
    expected = np.concatenate(
        [
            m.design(r @ U, g) @ flux[:, i].ravel()
            for i, (m, r) in enumerate(zip(models, rotations))
        ]
    )
    np.testing.assert_allclose(A @ flux.ravel(), expected)
    assert len(joint.data.pixel_indices) == 4
    for index in joint.data.pixel_indices.values():
        assert np.all(index >= 0)
    joint.data.counts = joint.data.background.copy()
    fit = fit_intensities(A, joint.data, 2, 1.2)
    assert fit.converged
    np.testing.assert_allclose(fit.background_scales, 1)
    np.testing.assert_array_equal(fit.group_norms, 0)


def test_invalid_setting_rotations_rejected():
    with pytest.raises(ValueError):
        MultiSettingModel([], [])


def test_short_inner_budget_retries_without_accepting_failed_fit(monkeypatch):
    from dataclasses import replace
    from scipy import sparse
    import subhkl.search.poisson_orientation as engine

    class Model:
        data = CountData.from_images([np.ones((2, 2))], [1], [1.0], 1)

        def design(self, orientations, geometry):
            return sparse.csc_matrix(np.ones((4, len(orientations))))

    original = engine.fit_intensities
    budgets = []

    def fit(*args, **kwargs):
        budgets.append(kwargs["max_iter"])
        result = original(*args, **kwargs)
        return replace(result, converged=False) if kwargs["max_iter"] < 2000 else result

    monkeypatch.setattr(engine, "fit_intensities", fit)
    result = engine.refine(Model(), np.eye(3)[None], inner_max_iter=100, max_evals=20)
    assert result["fit"].converged
    assert budgets[:2] == [100, 2000]
