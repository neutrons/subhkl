"""Stack still-count likelihoods with shared sample orientations and geometry."""

import numpy as np
from scipy import sparse

from subhkl.search.poisson_orientation import CountData


class MultiSettingModel:
    """One group per sample orientation, containing all setting/hkl fluxes.

    Each setting retains its own reflection intensities and panel backgrounds.
    Goniometer rotations are fixed, supplied sample-to-lab matrices; detector
    corrections are shared. No peak matching or intensity threshold is used.

    The constituent models currently fix detector roll about the beam. For
    settings that do not commute with beam-axis rotation this is a model
    restriction, not generally a gauge freedom; a complete joint calibration should revisit that choice.
    """

    def __init__(self, models, rotations):
        self.models = list(models)
        self.rotations = np.asarray(rotations, float)
        if not self.models or self.rotations.shape != (len(self.models), 3, 3):
            raise ValueError("one rotation per nonempty setting required")
        if not np.allclose(
            self.rotations @ self.rotations.swapaxes(1, 2), np.eye(3)
        ) or not np.allclose(np.linalg.det(self.rotations), 1):
            raise ValueError("setting rotations must be proper orthogonal matrices")
        first = self.models[0]
        for model in self.models:
            if (
                not np.array_equal(model.G, first.G)
                or model.data.bin_px != first.data.bin_px
            ):
                raise ValueError(
                    "settings must share reflection identities and binning"
                )
        self.n_hkl = first.n_hkl * len(self.models)
        counts, background, bank_indices = [], [], []
        indices, shapes = {}, {}
        offset, panel = 0, 0
        for model in self.models:
            data = model.data
            counts.append(data.counts)
            background.append(data.background)
            bank_indices.append(data.bank_index + panel)
            for bank, index in data.pixel_indices.items():
                indices[panel] = np.where(index >= 0, index + offset, -1)
                shapes[panel] = data.shapes[bank]
                panel += 1
            offset += len(data.counts)
        self.data = CountData(
            np.concatenate(counts),
            np.concatenate(background),
            np.concatenate(bank_indices),
            indices,
            shapes,
            first.data.bin_px,
        )

    def design(self, orientations, g):
        orientations = np.asarray(orientations).reshape(-1, 3, 3)
        designs = [
            model.design(rotation @ orientations, g)
            for model, rotation in zip(self.models, self.rotations)
        ]
        # block_diag orders columns setting/orientation/hkl; the group penalty
        # requires orientation/setting/hkl, with no sharing of physical fluxes.
        order = (
            np.arange(sum(a.shape[1] for a in designs))
            .reshape(len(designs), len(orientations), self.models[0].n_hkl)
            .transpose(1, 0, 2)
            .ravel()
        )
        return sparse.block_diag(designs, format="csc")[:, order]
