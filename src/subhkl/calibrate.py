"""Finder-free detector-geometry calibration from raw (pooled) counts.

The nominal detector geometry of an instrument can be wrong enough that an
orientation search under it explains only a fraction of the Bragg peaks.
This module calibrates the rigid geometry -- a radial scale of the detector
assembly, a rotation of the assembly and an offset of the sample -- from the
raw counts alone: no peak finder, no prior orientation, no goniometer.

The construction treats crystal *orientations* as the atoms of a dictionary.
For a geometry ``g`` and an orientation ``R`` the predicted pattern is the set
of scattering-vector directions of the in-band, on-panel reflections, binned
on a sphere of angular cells; a single crystal is a 1-sparse code in that
dictionary and its matched filter ``D^T y`` is the orientation-coherent score
the spherical indexer maximises.  Two things make the raw problem tractable:

* the data map ``y`` is made sparse like the atoms -- per sphere cell the
  *maximum* of a profiled Poisson log-likelihood ratio, passed through a
  saturating detection readout, so a spot counts once however bright it is;
* the dictionary is *centered*: every orientation lights the same band-and-
  panel cone, a rank-one common component that makes the raw dictionary
  fatally coherent and that subtracting the mean atom removes exactly.

With the exact scattering direction of each detected cell's argmax pixel as
its peak list, the lattice ladder finds an orientation blind.  Two scalar
objectives are then read off that orientation and maximised over the seven
geometry parameters jointly, coarse to fine:

* ``cell``: the matched-filter correlation of the found atom against the
  centered data map.  Wide basin, deterministic, but quantised to cells --
  at 1 deg three geometries indexing at 44 / 57 / 87 % are indistinguishable
  and the ridge is only resolved by finer cells (0.5, then 0.25 deg).
* ``raw``: the fraction of raw detections within an angular tolerance of a
  predicted reflection -- the indexing criterion itself with detections in
  place of finder peaks, continuous in the geometry.  It takes the last step
  the cell correlation cannot see (sub-cell tilt and offset).

The objectives are invariant to a global rotation of the lattice, so the
goniometer setting of the frames is irrelevant: the found orientation absorbs
it.  What is required is enough exposure for real spots to clear the null
tail of the detection statistic; on CG4D L1 MBL that means the frames of one
orientation pooled (10x), not a single still.

Measured on cg4d-l1-mbl (runs 1996-2014/2 pooled): nominal 24.8 % of the
finder's peaks explained under a full orientation search, the four stages
43.6 -> 64.4 -> 67.1 -> 73.2 %, the finder-based band-fit ceiling 86.6 %.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, generic_filter
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation as Rot

from subhkl.core.crystallography import (
    cartesian_matrix_metric_tensor,
    generate_reflections,
)
from subhkl.instrument.detector import Detector

KI = np.array([0.0, 0.0, 1.0])

# geometry parameter vector g = (radial scale s, assembly rotation vector
# [rad] x3, sample offset [m] x3); bounds as physical box constraints
G_LOWER = np.array([-0.08] + [-0.05] * 3 + [-0.02] * 3)
G_UPPER = -G_LOWER
# initial Nelder-Mead simplex step per parameter, at unit step scale:
# the expected size of a coupled geometry shift
G_STEP = np.array([0.03] + [np.radians(1.0)] * 3 + [0.005] * 3)


@dataclass(frozen=True)
class Stage:
    """One rung of the coarse-to-fine optimisation.

    cell_deg     sphere cell size for the detection map and the atoms
    objective    'cell' (matched-filter correlation) or 'raw' (fraction of
                 detections explained within ``tol_deg``)
    theta        detection threshold on the profiled LLR ``t``; ``t`` is
                 standardised so the null false-positive rate it fixes is
                 exposure-independent (8 permissive, 12 for the continuous
                 objective, 20 strict for validation)
    step_scale   multiplier on ``G_STEP`` for the initial simplex
    max_evals    Nelder-Mead evaluation budget
    """

    cell_deg: float
    objective: str
    theta: float
    step_scale: float
    max_evals: int
    tol_deg: float = 0.3
    width: float = 3.0


DEFAULT_STAGES = (
    Stage(1.0, "cell", 8.0, 2.0, 160),
    Stage(0.5, "cell", 8.0, 0.7, 100),
    Stage(0.25, "cell", 8.0, 0.7, 100),
    Stage(0.25, "raw", 12.0, 0.4, 100),
)

QUICK_STAGES = (
    Stage(1.0, "cell", 8.0, 2.0, 60),
    Stage(0.5, "cell", 8.0, 0.7, 40),
    Stage(0.25, "raw", 12.0, 0.4, 40),
)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def geometry_detectors(
    instrument: str, g: np.ndarray, banks: list[int] | None = None
) -> dict[int, Detector]:
    """The instrument's panels under the rigid geometry ``g``.

    Every panel centre is scaled by ``1 + s`` and the whole assembly rotated
    by the rotation vector, then shifted by the offset (equivalently the
    sample is moved the other way).  Panel orientations rotate with the
    assembly; sizes are untouched.
    """
    from subhkl.config import beamlines

    g = np.asarray(g, float)
    s, Rg, T = g[0], Rot.from_rotvec(g[1:4]).as_matrix(), g[4:7]
    out = {}
    for key, cfg0 in beamlines[instrument].items():
        b = int(key)
        if banks is not None and b not in banks:
            continue
        cfg = dict(cfg0)
        cfg["center"] = (
            Rg @ ((1.0 + s) * np.asarray(cfg["center"], float)) + T
        ).tolist()
        for name in ("uhat", "vhat", "rhat"):
            if name in cfg:
                cfg[name] = (Rg @ np.asarray(cfg[name], float)).tolist()
        out[b] = Detector(cfg)
    return out


def panel_table(dets: dict[int, Detector]) -> dict:
    """Panels in the layout the ``detector_calibration`` writer takes."""
    banks = sorted(dets)
    return {
        "banks": banks,
        "centers": np.array([dets[b].center for b in banks], float),
        "uhats": np.array([dets[b].uhat for b in banks], float),
        "vhats": np.array([dets[b].vhat for b in banks], float),
        "widths": np.array([dets[b].width for b in banks], float),
        "heights": np.array([dets[b].height for b in banks], float),
    }


def write_detector_calibration(out: h5py.File | h5py.Group, dets: dict[int, Detector]):
    """Write the bootstrap-compatible ``detector_calibration`` group.

    Same layout the spherical indexer writes and ``apply_detector_calibration``
    reads: ``detector_calibration/bank_<id>/{center,uhat,vhat,width,height}``.
    """
    if "detector_calibration" in out:
        del out["detector_calibration"]
    pan = panel_table(dets)
    for i, b in enumerate(pan["banks"]):
        grp = out.create_group(f"detector_calibration/bank_{int(b)}")
        grp["center"] = np.asarray(pan["centers"][i], dtype=float)
        grp["uhat"] = np.asarray(pan["uhats"][i], dtype=float)
        grp["vhat"] = np.asarray(pan["vhats"][i], dtype=float)
        grp["width"] = float(pan["widths"][i])
        grp["height"] = float(pan["heights"][i])


# --------------------------------------------------------------------------
# reciprocal lattice
# --------------------------------------------------------------------------


def reciprocal_lattice(cell, space_group: str, d_min: float):
    """``(B, G_c)``: the reciprocal-cell matrix and the reflections' scattering
    vectors in the crystal frame (rows), to ``d_min``."""
    a, b, c, al, be, ga = [float(v) for v in cell]
    B, _ = cartesian_matrix_metric_tensor(a, b, c, *np.deg2rad([al, be, ga]))
    h, k, l_ = generate_reflections(
        a, b, c, al, be, ga, space_group=space_group, d_min=d_min
    )
    G_c = np.stack([h, k, l_], axis=1) @ B.T
    return B, G_c


def predicted_reflections(
    dets: dict[int, Detector], U: np.ndarray, G_c: np.ndarray, wavelength
):
    """Scattering-vector directions and scattered-beam panel directions of the
    in-band, on-panel reflections of orientation ``U`` (lab frame, unit)."""
    from subhkl.search.spherical import panel_directions

    Gq = G_c @ U.T
    qn = np.linalg.norm(Gq, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        st = -(Gq @ KI) / qn
        la = 2.0 * st / qn
    m = (la >= wavelength[0]) & (la <= wavelength[1]) & (st > 0)
    kf = KI + la[m, None] * Gq[m]
    kf /= np.linalg.norm(kf, axis=1, keepdims=True)
    Gm = Gq[m] / qn[m, None]
    ghat, pdir = [], []
    for det in dets.values():
        mk, pr, pc = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
        mk = np.asarray(mk, bool)
        if not mk.any():
            continue
        ghat.append(Gm[mk])
        p = panel_directions(
            det, rows=np.asarray(pr)[mk], cols=np.asarray(pc)[mk], ki=KI
        )
        pdir.append(p / np.linalg.norm(p, axis=1, keepdims=True))
    if not ghat:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(ghat), np.concatenate(pdir)


# --------------------------------------------------------------------------
# data: per-bank residual maps
# --------------------------------------------------------------------------


def prepare_banks(
    images: np.ndarray,
    bank_ids,
    masks: dict[int, np.ndarray] | None = None,
    bin_px: int = 4,
    median_window: int = 9,
    zero_fraction_max: float = 0.05,
    background_reference: dict | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per bank: binned counts ``y``, background ``b`` and a validity map.

    Frames are binned ``bin_px`` x ``bin_px``.  Masks (1 = valid) are carried
    as validity, never as zero counts -- zeros would push a bank into the
    sparse regime below.  A frame whose valid pixels are mostly non-zero gets
    a local median background over the bins with no zero pixel; a sparse
    frame gets a flat Poisson estimate from its zero fraction.

    ``background_reference`` (the output of this function on a pooled stack
    of the same instrument setting) replaces both: background is linear in
    exposure, so the reference background is scaled per bank by the median
    ratio of counts over its valid bins -- the fitted ratio, never a nominal
    one (a 16 % bias in that ratio produced thousands of false detections).
    """
    data = {}
    for im, bk in zip(images, bank_ids):
        bk = int(bk)
        im = np.asarray(im, float)
        n2, m2 = (im.shape[0] // bin_px) * bin_px, (im.shape[1] // bin_px) * bin_px
        blk = im[:n2, :m2].reshape(n2 // bin_px, bin_px, m2 // bin_px, bin_px)
        y = blk.sum(axis=(1, 3))
        valid = (
            np.asarray(masks[bk], bool)
            if masks is not None and bk in masks
            else np.ones(im.shape, bool)
        )
        vb = (
            valid[:n2, :m2]
            .reshape(n2 // bin_px, bin_px, m2 // bin_px, bin_px)
            .all(axis=(1, 3))
        )
        if background_reference is not None:
            if bk not in background_reference:
                continue
            _, bs, ks = background_reference[bk]
            ks = ks & vb
            if ks.sum() < 50:
                continue
            scale = float(np.median(y[ks] / bs[ks]))
            data[bk] = (y, np.clip(scale * bs, 0.5, None), ks)
            continue
        zf = float(np.mean(im[valid] == 0)) if valid.any() else 1.0
        if zf >= zero_fraction_max:
            bg = -np.log(max(zf, 1e-6)) * bin_px * bin_px * np.ones_like(y)
            keep = vb.copy()
        else:
            lv = (blk > 0).all(axis=(1, 3)) & vb
            if lv.sum() < 100:
                continue
            bg = np.nan_to_num(
                generic_filter(
                    np.where(lv, y, np.nan),
                    np.nanmedian,
                    size=median_window,
                    mode="nearest",
                ),
                nan=1.0,
            )
            keep = binary_erosion(lv, iterations=5)
        if keep.sum() < 50:
            continue
        data[bk] = (y, np.clip(bg, 0.5, None), keep)
    return data


def _gauss(sigma: float) -> np.ndarray:
    r = int(np.ceil(4 * sigma))
    x = np.arange(-r, r + 1)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    return np.outer(g, g)


def spot_maps(
    data: dict, radial_scale: float, sigma_div: float = 1.6, sigma_psf: float = 0.8
) -> dict[int, np.ndarray]:
    """Per bank, the profiled Poisson log-likelihood ratio of a spot at every
    bin: ``t = L1^2 / (2 L2)`` for ``L1 > 0`` with ``L1 = P * (y-b)/b`` and
    ``L2 = P^2 * y/b^2``.  The profile's divergence term scales with the
    sample-to-detector distance, ``sigma^2 = (sigma_div (1+s))^2 + sigma_psf^2``
    (bins), so a trial radial scale carries its own spot size.
    """
    P = _gauss(np.sqrt((sigma_div * (1.0 + radial_scale)) ** 2 + sigma_psf**2))
    out = {}
    for bk, (y, bg, keep) in data.items():
        w = np.where(keep, (y - bg) / bg, 0.0)
        inf = np.where(keep, y / bg**2, 0.0)
        L1 = fftconvolve(w, P, mode="same")
        L2 = np.maximum(fftconvolve(inf, P**2, mode="same"), 1e-9)
        out[bk] = np.where(keep & (L1 > 0), L1**2 / (2.0 * L2), 0.0)
    return out


# --------------------------------------------------------------------------
# the sphere of scattering directions
# --------------------------------------------------------------------------


class SphereGrid:
    """Equiangular cells on the unit sphere of scattering-vector directions."""

    def __init__(self, cell_deg: float):
        self.cell_deg = float(cell_deg)
        self.nth = int(round(180.0 / cell_deg))
        self.nph = 2 * self.nth
        self.n_cells = self.nth * self.nph

    def to_cell(self, ghat: np.ndarray) -> np.ndarray:
        th = np.arccos(np.clip(ghat[:, 2], -1.0, 1.0))
        ph = np.arctan2(ghat[:, 1], ghat[:, 0]) % (2.0 * np.pi)
        it = np.clip((th / np.pi * self.nth).astype(int), 0, self.nth - 1)
        ip = np.clip((ph / (2.0 * np.pi) * self.nph).astype(int), 0, self.nph - 1)
        return it * self.nph + ip


@dataclass
class DetectionMap:
    """The sparse data map on the sphere for one geometry.

    y          soft detection per cell (0..1), 0 where the detector does not
               look
    covered    cells the detector covers
    direction  exact scattering direction of the argmax pixel of each covered
               cell -- finder-grade precision for the orientation search;
               the cell centre (+-half a cell) is not enough
    """

    y: np.ndarray
    covered: np.ndarray
    direction: np.ndarray

    @property
    def detected(self) -> np.ndarray:
        return self.covered & (self.y > 0.5)


def detect_on_sphere(
    dets: dict[int, Detector],
    tmaps: dict[int, np.ndarray],
    data: dict,
    grid: SphereGrid,
    theta: float,
    width: float,
    bin_px: int,
) -> DetectionMap:
    """Push every valid bin's ``t`` to its sphere cell (max per cell) and read
    it through the saturating detection ``sigmoid((t_max - theta) / width)``."""
    from subhkl.search.spherical import panel_directions

    acc = np.zeros(grid.n_cells)
    cnt = np.zeros(grid.n_cells)
    cells, tvals, gvecs = [], [], []
    for bk, tm in tmaps.items():
        det = dets[bk]
        H, W = tm.shape
        rr, cc = np.meshgrid(
            (np.arange(H) + 0.5) * bin_px, (np.arange(W) + 0.5) * bin_px, indexing="ij"
        )
        gh = panel_directions(det, rows=rr.ravel(), cols=cc.ravel(), ki=KI)
        gh = gh / np.linalg.norm(gh, axis=1, keepdims=True)
        c = grid.to_cell(gh)
        keep = data[bk][2].ravel()
        np.maximum.at(acc, c[keep], tm.ravel()[keep])
        np.add.at(cnt, c[keep], 1)
        cells.append(c[keep])
        tvals.append(tm.ravel()[keep])
        gvecs.append(gh[keep])
    direction = np.zeros((grid.n_cells, 3))
    if cells:
        cells = np.concatenate(cells)
        tvals = np.concatenate(tvals)
        gvecs = np.concatenate(gvecs)
        order = np.lexsort((-tvals, cells))
        cells, gvecs = cells[order], gvecs[order]
        first = np.r_[True, cells[1:] != cells[:-1]]
        direction[cells[first]] = gvecs[first]
    covered = cnt > 0
    y = np.where(covered, 1.0 / (1.0 + np.exp(-(acc - theta) / width)), 0.0)
    return DetectionMap(y=y, covered=covered, direction=direction)


# --------------------------------------------------------------------------
# orientation dictionary and the two objectives
# --------------------------------------------------------------------------


def orientation_atom(
    dets: dict[int, Detector],
    U: np.ndarray,
    G_c: np.ndarray,
    wavelength,
    grid: SphereGrid,
) -> np.ndarray:
    """Cells lit by orientation ``U``: its in-band, on-panel reflections."""
    ghat, _ = predicted_reflections(dets, U, G_c, wavelength)
    if len(ghat) == 0:
        return np.zeros(0, int)
    return np.unique(grid.to_cell(ghat))


def _enable_compilation_cache():
    """Persist XLA compilations across runs (the same cache spherical-index
    uses); a calibration is hundreds of ladder calls."""
    try:
        import os

        import jax

        if jax.config.jax_compilation_cache_dir is None:
            jax.config.update(
                "jax_compilation_cache_dir",
                os.path.join(
                    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                    "subhkl",
                    "jax",
                ),
            )
    except Exception:
        pass


def blind_orientation(
    dmap: DetectionMap,
    B: np.ndarray,
    wavelength,
    d_min: float,
    seed: int = 0,
    n_pad: int | None = None,
):
    """The lattice ladder on the detected cells' exact directions -- no seed
    orientation, no finder.  Returns ``(U, band_score, n_detections)``.

    ``n_pad`` pads the detection list (a repeated real direction at zero
    weight, which every ladder kernel ignores) so its shape does not change
    from one geometry to the next: the ladder compiles per size class, and
    an optimiser whose detection count straddles a class boundary would
    otherwise recompile on almost every evaluation (9 s of a 20 s call).
    """
    from subhkl.search.spherical import _orthonormalize, lattice_ladder

    sel = np.where(dmap.detected)[0]
    d = dmap.direction[sel]
    w = dmap.y[sel]
    if n_pad is not None and len(d) < n_pad and len(d) > 0:
        extra = n_pad - len(d)
        d = np.vstack([d, np.tile(d[:1], (extra, 1))])
        w = np.concatenate([w, np.zeros(extra)])
    res = lattice_ladder(
        d,
        w,
        B,
        sin_theta=-(d @ KI),
        wavelength=tuple(wavelength),
        d_min=d_min,
        cands=None,
        seed=seed,
    )
    return (
        _orthonormalize(np.asarray(res[0][0], float)),
        float(res[0][1]),
        int(len(sel)),
    )


def cell_correlation(
    dmap: DetectionMap,
    dets: dict[int, Detector],
    U: np.ndarray,
    G_c: np.ndarray,
    wavelength,
    grid: SphereGrid,
    random_orientations,
) -> float:
    """Matched-filter correlation of orientation ``U``'s atom with the data
    map, both centered over the covered cells.  The random orientations
    supply the mean atom -- the coverage component -- that centering removes;
    50 of them reproduce 200 (and 200 reproduce 2500) to a few hundredths."""
    atoms = [
        orientation_atom(dets, R, G_c, wavelength, grid)
        for R in [U, *random_orientations]
    ]
    cov = dmap.covered
    D = np.zeros((grid.n_cells, len(atoms)))
    for j, cells in enumerate(atoms):
        D[cells, j] = 1.0
    D = D[cov]
    Dc = D - D.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Dc[:, 0])
    if n == 0:
        return 0.0
    yc = dmap.y[cov] - dmap.y[cov].mean()
    return float(Dc[:, 0] @ yc / n)


def raw_explained(
    dmap: DetectionMap,
    dets: dict[int, Detector],
    U: np.ndarray,
    G_c: np.ndarray,
    wavelength,
    tol_deg: float = 0.3,
) -> tuple[float, int]:
    """Fraction of detections within ``tol_deg`` of a reflection predicted
    for ``U`` under this geometry -- the indexing criterion, finder-free and
    continuous in the geometry.  Returns ``(fraction, n_predicted)``."""
    _, pdir = predicted_reflections(dets, U, G_c, wavelength)
    D = dmap.direction[dmap.detected]
    if len(pdir) == 0 or len(D) == 0:
        return 0.0, int(len(pdir))
    ang = np.degrees(np.arccos(np.clip(D @ pdir.T, -1.0, 1.0))).min(axis=1)
    return float((ang < tol_deg).mean()), int(len(pdir))


# --------------------------------------------------------------------------
# the calibration
# --------------------------------------------------------------------------


@dataclass
class Evaluation:
    g: np.ndarray
    U: np.ndarray
    band_score: float
    n_detections: int
    cell_correlation: float
    raw_explained: float
    n_predicted: int


@dataclass
class StageResult:
    stage: Stage
    n_evals: int
    best_value: float
    g: np.ndarray


@dataclass
class CalibrationResult:
    g: np.ndarray
    dets: dict[int, Detector]
    U: np.ndarray
    stages: list[StageResult] = field(default_factory=list)
    nominal: Evaluation | None = None
    final: Evaluation | None = None

    @property
    def radial_scale(self) -> float:
        return float(self.g[0])

    @property
    def rotation_vector(self) -> np.ndarray:
        return np.asarray(self.g[1:4], float)

    @property
    def translation(self) -> np.ndarray:
        return np.asarray(self.g[4:7], float)


class Calibrator:
    """Everything one geometry evaluation needs, held once."""

    def __init__(
        self,
        data: dict,
        instrument: str,
        cell,
        space_group: str,
        wavelength,
        d_min: float,
        bin_px: int = 4,
        n_random: int = 50,
        seed: int = 0,
        sigma_div: float = 1.6,
        sigma_psf: float = 0.8,
    ):
        self.data = data
        self.instrument = instrument
        self.banks = sorted(data)
        self.wavelength = (float(wavelength[0]), float(wavelength[1]))
        self.d_min = float(d_min)
        self.bin_px = int(bin_px)
        self.seed = int(seed)
        self.sigma_div, self.sigma_psf = float(sigma_div), float(sigma_psf)
        self.B, self.G_c = reciprocal_lattice(cell, space_group, d_min)
        rng = np.random.default_rng(seed)
        self._n_pad: dict[tuple, int] = {}  # stable ladder input size per stage
        _enable_compilation_cache()
        self.random_orientations = [
            Rot.random(random_state=rng).as_matrix() for _ in range(n_random)
        ]

    def detectors(self, g) -> dict[int, Detector]:
        return geometry_detectors(self.instrument, g, banks=self.banks)

    def detection_map(
        self, g, stage: Stage
    ) -> tuple[DetectionMap, dict[int, Detector]]:
        g = np.asarray(g, float)
        dets = self.detectors(g)
        tmaps = spot_maps(self.data, g[0], self.sigma_div, self.sigma_psf)
        grid = SphereGrid(stage.cell_deg)
        return detect_on_sphere(
            dets, tmaps, self.data, grid, stage.theta, stage.width, self.bin_px
        ), dets

    def evaluate(
        self, g, stage: Stage, orientation: np.ndarray | None = None
    ) -> Evaluation:
        """Both objectives at geometry ``g``.  ``orientation`` skips the blind
        search (a known orientation, for validation and tests)."""
        g = np.asarray(g, float)
        dmap, dets = self.detection_map(g, stage)
        grid = SphereGrid(stage.cell_deg)
        if orientation is None:
            # one size class per stage, with headroom; bumped only if exceeded
            key = (stage.cell_deg, stage.theta)
            n_det = int(dmap.detected.sum())
            n_pad = self._n_pad.get(key)
            if n_pad is None or n_det > n_pad:
                n_pad = 1 << int(np.ceil(np.log2(max(int(1.25 * n_det), 2))))
                self._n_pad[key] = n_pad
            U, band, n_det = blind_orientation(
                dmap, self.B, self.wavelength, self.d_min, self.seed, n_pad=n_pad
            )
        else:
            U, band, n_det = (
                np.asarray(orientation, float),
                float("nan"),
                int(dmap.detected.sum()),
            )
        corr = cell_correlation(
            dmap, dets, U, self.G_c, self.wavelength, grid, self.random_orientations
        )
        frac, n_pred = raw_explained(
            dmap, dets, U, self.G_c, self.wavelength, stage.tol_deg
        )
        return Evaluation(g, U, band, n_det, corr, frac, n_pred)

    @staticmethod
    def objective_value(ev: Evaluation, stage: Stage) -> float:
        return (
            100.0 * ev.raw_explained
            if stage.objective == "raw"
            else ev.cell_correlation
        )

    def optimise_stage(
        self, g0, stage: Stage, log=None
    ) -> tuple[StageResult, Evaluation]:
        """Nelder-Mead over all seven parameters jointly.

        Jointly is essential: the parameters are coupled, and along the radial
        scale alone the nominal geometry is a local maximum (a +6 % radial
        change without its tilt and offset loses the lattice lock entirely).
        The initial simplex spans the expected coupled shift so the first
        reflections through the centroid already probe the diagonal.
        """
        g0 = np.clip(np.asarray(g0, float), G_LOWER, G_UPPER)
        best = {"value": -np.inf, "g": g0.copy(), "ev": None}
        n_ev = [0]

        def f(gp):
            pen = 1e4 * float(
                np.sum(
                    np.maximum(0.0, G_LOWER - gp) ** 2
                    + np.maximum(0.0, gp - G_UPPER) ** 2
                )
            )
            g = np.clip(gp, G_LOWER, G_UPPER)
            ev = self.evaluate(g, stage)
            v = self.objective_value(ev, stage)
            n_ev[0] += 1
            if v > best["value"]:
                best.update(value=v, g=g.copy(), ev=ev)
            if log is not None:
                log(
                    f"  [{stage.objective} {stage.cell_deg:g} deg] eval {n_ev[0]:3d}  "
                    f"value {v:8.3f}  radial {100 * g[0]:+5.2f}%  "
                    f"rot {np.degrees(np.linalg.norm(g[1:4])):4.2f} deg  "
                    f"|T| {1e3 * np.linalg.norm(g[4:7]):4.1f} mm  best {best['value']:.3f}"
                )
            return -v + pen

        step = stage.step_scale * G_STEP
        simplex = np.vstack([g0] + [g0 + np.eye(7)[i] * step[i] for i in range(7)])
        minimize(
            f,
            g0,
            method="Nelder-Mead",
            options={
                "initial_simplex": simplex,
                "maxfev": int(stage.max_evals),
                "xatol": 1e-4,
                "fatol": 1e-3,
                "adaptive": True,
            },
        )
        return StageResult(stage, n_ev[0], best["value"], best["g"]), best["ev"]

    def calibrate(self, stages=DEFAULT_STAGES, g0=None, log=None) -> CalibrationResult:
        g = np.zeros(7) if g0 is None else np.asarray(g0, float)
        nominal = self.evaluate(np.zeros(7), stages[0])
        if log is not None:
            log(
                f"nominal geometry: {nominal.n_detections} detections, cell corr "
                f"{nominal.cell_correlation:.3f}, raw explained {100 * nominal.raw_explained:.1f}%"
            )
        results, ev = [], None
        for stage in stages:
            if log is not None:
                log(
                    f"stage: {stage.objective} objective, {stage.cell_deg:g} deg cells, "
                    f"theta {stage.theta:g}, {stage.max_evals} evaluations"
                )
            sr, ev = self.optimise_stage(g, stage, log=log)
            results.append(sr)
            g = sr.g
            if log is not None:
                log(
                    f"  -> best {sr.best_value:.3f} at radial {100 * g[0]:+.2f}%  "
                    f"rot {np.degrees(np.linalg.norm(g[1:4])):.2f} deg  "
                    f"T {np.round(1e3 * g[4:7], 1)} mm"
                )
        final = self.evaluate(g, stages[-1]) if ev is None else ev
        return CalibrationResult(
            g=g,
            dets=self.detectors(g),
            U=final.U,
            stages=results,
            nominal=nominal,
            final=final,
        )


def load_masks(
    mask_filename: str | None, bank_ids, shape
) -> dict[int, np.ndarray] | None:
    """Static mask file (``images`` 1 = valid, ``bank_ids``) as per-bank
    validity, aligned to ``bank_ids``."""
    if not mask_filename:
        return None
    from subhkl.search.static_mask import load_mask_for_banks

    stack = load_mask_for_banks(mask_filename, [int(b) for b in bank_ids], tuple(shape))
    return {int(b): np.asarray(m, float) > 0.5 for b, m in zip(bank_ids, stack)}


def write_report(
    out: h5py.File | h5py.Group, result: CalibrationResult, instrument: str
):
    """The ``calibrate`` group: the fitted parameters, the stage table and
    the orientation the calibrated geometry implies."""
    if "calibrate" in out:
        del out["calibrate"]
    grp = out.create_group("calibrate")
    grp.attrs["instrument"] = instrument
    grp["radial_scale"] = float(result.g[0])
    grp["rotation_vector"] = np.asarray(result.g[1:4], float)
    grp["rotation_deg"] = float(np.degrees(np.linalg.norm(result.g[1:4])))
    grp["translation_m"] = np.asarray(result.g[4:7], float)
    grp["g"] = np.asarray(result.g, float)
    grp["U"] = np.asarray(result.U, float)
    for name, ev in (("nominal", result.nominal), ("final", result.final)):
        if ev is None:
            continue
        e = grp.create_group(name)
        e["cell_correlation"] = float(ev.cell_correlation)
        e["raw_explained"] = float(ev.raw_explained)
        e["n_detections"] = int(ev.n_detections)
        e["n_predicted"] = int(ev.n_predicted)
        e["band_score"] = float(ev.band_score)
    st = grp.create_group("stages")
    st["cell_deg"] = np.array([s.stage.cell_deg for s in result.stages], float)
    st["objective"] = np.array(
        [s.stage.objective for s in result.stages], dtype=h5py.string_dtype()
    )
    st["theta"] = np.array([s.stage.theta for s in result.stages], float)
    st["n_evals"] = np.array([s.n_evals for s in result.stages], int)
    st["best_value"] = np.array([s.best_value for s in result.stages], float)
    st["g"] = np.array([s.g for s in result.stages], float).reshape(-1, 7)
