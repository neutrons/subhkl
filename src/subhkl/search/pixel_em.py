"""Expectation-Maximization per-pixel/bin classifier (valid, dark, bright).

Classifies static detector artifacts simultaneously with orientation and
geometry refinement across N >= 1 goniometer frames:

* ``dark`` (class 1): dead detector panels, unilluminated edge/tube strips,
  and beamstop/cryostat/magnet shadows. Dark pixels stay depressed relative to
  expected ambient illumination and never light up even when crystal
  reflections are predicted to land on them.
* ``bright`` (class 2): parasitic static reflections (e.g. aluminium DNP
  magnet/cryostat enclosure spots), hot columns/wires, and epithermal scatter.
  Bright pixels light up strongly even outside the crystal's predicted
  (hkl, wavelength) reflections, and persist across distinct goniometer
  frames when N >= 2.
* ``valid`` (class 0): normal active detector pixels used for background
  estimation, band/Ewald refinement, and downstream reflection prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    median_filter,
    percentile_filter,
    uniform_filter,
)
from scipy.spatial.transform import Rotation

from subhkl.config import beamlines
from subhkl.core.crystallography import generate_reflections
from subhkl.instrument.detector import Detector


def masked_median_bg(
    y2d: np.ndarray, valid_bg_mask: np.ndarray, size: int
) -> np.ndarray:
    """2D median background after nearest-valid EDT fill of masked regions.

    On clean panels where ``valid_bg_mask`` is all-True, returns the exact
    ``scipy.ndimage.median_filter`` result without modification. Near shadow
    boundaries or bright parasitic blobs, filling masked bins from the nearest
    unmasked bin prevents step-edge under/overshoot ridges.
    """
    if np.all(valid_bg_mask):
        return median_filter(y2d, size=size, mode="nearest")
    if not np.any(valid_bg_mask):
        return np.zeros_like(y2d)
    _, indices = distance_transform_edt(~valid_bg_mask, return_indices=True)
    filled = y2d[indices[0], indices[1]]
    return median_filter(filled, size=size, mode="nearest")


def inpainted_median_bg(
    y2d: np.ndarray,
    dark_mask: np.ndarray,
    bright_mask: np.ndarray | None,
    size: int,
    bg_raw: np.ndarray | None = None,
) -> np.ndarray:
    """2D median background with monotone dark fill and EDT bright removal.

    Masking depressed/dark bins can only *increase* the local ambient estimate
    relative to an unmasked median filter, preventing edge-glow strips next to
    lower-background interiors from ever receiving an artificially low
    background. Masking localized static-bright bins then prevents parasitic
    enclosure reflections from inflating neighboring background.
    """
    if bg_raw is None:
        bg_raw = median_filter(y2d, size=size, mode="nearest")
    if np.any(dark_mask):
        bg_dark = np.maximum(bg_raw, masked_median_bg(y2d, ~dark_mask, size=size))
    else:
        bg_dark = bg_raw
    if bright_mask is None or not np.any(bright_mask):
        return bg_dark
    y_clean_dark = np.where(dark_mask, bg_dark, y2d)
    valid_both = (~dark_mask) & (~bright_mask)
    return masked_median_bg(y_clean_dark, valid_both, size=size)


def poisson_upward_deviance(y: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """One-sided upward Poisson deviance D+(y || bg) >= 0."""
    bg_safe = np.maximum(bg, 1.0)
    y_safe = np.maximum(y, bg_safe)
    return np.where(
        y > bg_safe,
        2.0 * (y_safe * np.log(y_safe / bg_safe) - (y_safe - bg_safe)),
        0.0,
    )


def overdispersed_upward_deviance(
    y: np.ndarray, bg: np.ndarray, rel_sigma: float = 0.09
) -> np.ndarray:
    """Upward deviance normalized by Poisson + smooth diffuse spatial variance.

    At low background (``bg << 1 / rel_sigma^2``) this reduces to exact Poisson
    deviance; at high diffuse scattering (e.g. ``bg = 6000`` counts per 16x16
    bin) smooth 8-12% spatial background gradients produce ``D_disp ~ 1-2``
    instead of mimicking static bright defects.
    """
    d_pois = poisson_upward_deviance(y, bg)
    disp = 1.0 + (float(rel_sigma) ** 2) * np.maximum(bg, 0.0)
    return d_pois / disp


def wilson_search_weights(
    dev: np.ndarray,
    gamma_valid: np.ndarray | None = None,
    *,
    null_scale: float = 9.0,
    wilson_tau: float = 120.0,
) -> np.ndarray:
    """Smooth chi2-gated, Wilson log-deviance weights for coarse search.

    Suppresses 1-sigma Poisson noise via the same smooth gate ``(D / (D + s))^4``
    as ``count_search_weights`` while replacing unbounded ``sqrt(D)`` growth
    with ``log(1 + D / tau)`` (the marginal reflection log-likelihood under an
    exponential Wilson intensity prior) so extreme parasitic enclosure spots
    cannot outweigh dozens of genuine protein reflections before ``U`` is known.
    """
    d = np.maximum(np.asarray(dev, dtype=np.float64), 0.0)
    s = float(null_scale)
    gate = (d / (d + s)) ** 4
    w = gate * np.log1p(d / max(float(wilson_tau), 1.0))
    if gamma_valid is not None:
        w = w * np.clip(np.asarray(gamma_valid, dtype=np.float64), 0.0, 1.0)
    return w


def _distinct_rotation_clusters(
    run_ids: list[int], R_by_run: dict[int, np.ndarray], tol_deg: float = 1.5
) -> list[list[int]]:
    """Group runs whose goniometer orientations agree within ``tol_deg``."""
    clusters: list[list[int]] = []
    reps: list[np.ndarray] = []
    cos_thr = float(np.cos(np.deg2rad(tol_deg)))
    for r in run_ids:
        R = np.asarray(R_by_run.get(r, np.eye(3)), dtype=np.float64)
        placed = False
        for cid, R_rep in enumerate(reps):
            tr = float(np.trace(R_rep.T @ R))
            cos_ang = np.clip(0.5 * (tr - 1.0), -1.0, 1.0)
            if cos_ang >= cos_thr:
                clusters[cid].append(r)
                placed = True
                break
        if not placed:
            reps.append(R)
            clusters.append([r])
    return clusters


@dataclass
class EMPixelMaskExport:
    """Full-resolution per-bank detector pixel masks and summary diagnostics."""

    bank_ids: np.ndarray  # (n_banks,) int32
    pixel_mask: np.ndarray  # (n_banks, H, W) uint8, 1 = valid, 0 = ignored
    pixel_class: np.ndarray  # (n_banks, H, W) uint8, 0 = valid, 1 = dark, 2 = bright
    valid_fraction: float
    dark_fraction: float
    bright_fraction: float
    dead_banks: list[int]
    n_bright_bins: int
    n_dark_bins: int


class PixelEMClassifier:
    """Closed-form 3-state (valid, dark, bright) per-bin EM classifier."""

    def __init__(
        self,
        *,
        instrument: str,
        n_rows: int,
        n_cols: int,
        binning: int,
        border_width: int,
        sel_runs: list[int],
        obs_banks: list[int],
        y_rb: dict[tuple[int, int], np.ndarray],
        good_rb: dict[tuple[int, int], np.ndarray],
        rr_rb: dict[tuple[int, int], np.ndarray],
        cc_rb: dict[tuple[int, int], np.ndarray],
        bad_px_bank: dict[int, np.ndarray],
        R_by_run: dict[int, np.ndarray],
        cell: list[float],
        space_group: str,
        wavelength: tuple[float, float],
        d_min: float,
        sub_peak_dev_rb: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> None:
        self.instrument = str(instrument)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.b = int(binning)
        self.bw = int(border_width)
        self.hb = self.n_rows // self.b
        self.wb = self.n_cols // self.b
        self.sel_runs = [int(r) for r in sel_runs]
        self.obs_banks = sorted(int(bk) for bk in obs_banks)
        self.y_rb = y_rb
        self.good_rb = good_rb
        self.rr_rb = rr_rb
        self.cc_rb = cc_rb
        self.bad_px_bank = bad_px_bank
        self.R_by_run = R_by_run
        self.sub_peak_dev_rb = sub_peak_dev_rb or {}
        self.cell = [float(x) for x in cell[:6]]
        self.space_group = str(space_group)
        self.wavelength = (float(wavelength[0]), float(wavelength[1]))
        self.d_min = float(d_min)

        self.rot_clusters = _distinct_rotation_clusters(self.sel_runs, self.R_by_run)
        self.n_orient_clusters = len(self.rot_clusters)

        # Protect higher-resolution crystal reflections down to the kinematic
        # wavelength floor (bounded by a 280k reflection budget on large cells)
        # so genuine high-Q Bragg peaks are never misclassified as bright when
        # the user specifies a coarser solver d_min (e.g. d_min=2.8).
        ca, cb, cg = np.cos(np.deg2rad(self.cell[3:]))
        vol_fac = max(1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg, 1e-6)
        v_cell = float(self.cell[0] * self.cell[1] * self.cell[2] * np.sqrt(vol_fac))
        d_kin = max(0.54 * self.wavelength[0], 1.50)
        d_cap = float(((2.0 * np.pi * v_cell) / (3.0 * 280_000.0)) ** (1.0 / 3.0))
        d_min_em = min(self.d_min, max(d_kin, d_cap))
        hkl_em = np.stack(
            generate_reflections(
                *self.cell, space_group=self.space_group, d_min=d_min_em
            ),
            axis=1,
        ).astype(np.float64)
        self._hkl_em = hkl_em
        self._d_min_em = d_min_em

        # Pre-index bank positions and per-run centroid grids for fast stencil splat
        self._bank_to_bi = {bk: i for i, bk in enumerate(self.obs_banks)}
        n_b = len(self.obs_banks)
        centers_r = np.arange(self.hb) * self.b + (self.b - 1) / 2.0
        centers_c = np.arange(self.wb) * self.b + (self.b - 1) / 2.0
        rr_def, cc_def = np.meshgrid(centers_r, centers_c, indexing="ij")
        self._rr_run: dict[int, np.ndarray] = {}
        self._cc_run: dict[int, np.ndarray] = {}
        self._present_run: dict[int, np.ndarray] = {}
        for r in self.sel_runs:
            rr_s = np.empty((n_b, self.hb, self.wb), dtype=np.float64)
            cc_s = np.empty((n_b, self.hb, self.wb), dtype=np.float64)
            pres = np.zeros(n_b, dtype=bool)
            for bi, bk in enumerate(self.obs_banks):
                if (r, bk) in self.rr_rb:
                    rr_s[bi] = self.rr_rb[(r, bk)]
                    cc_s[bi] = self.cc_rb[(r, bk)]
                    pres[bi] = True
                else:
                    rr_s[bi] = rr_def
                    cc_s[bi] = cc_def
            self._rr_run[r] = rr_s
            self._cc_run[r] = cc_s
            self._present_run[r] = pres

        inst_cfg = beamlines[self.instrument]
        self._det_nom = [Detector(dict(inst_cfg[str(bk)])) for bk in self.obs_banks]

        # Exposure scale per run (median = 1.0 across active panels)
        run_totals = {}
        for r in self.sel_runs:
            tot = sum(
                float(self.y_rb[(r, bk)].sum())
                for bk in self.obs_banks
                if (r, bk) in self.y_rb
            )
            run_totals[r] = tot
        pos_tots = [v for v in run_totals.values() if v > 0.0]
        med_tot = float(np.median(pos_tots)) if pos_tots else 1.0
        self.exp_scale = {
            r: max(run_totals[r] / max(med_tot, 1.0), 0.05) for r in self.sel_runs
        }

        # Precompute cross-run high envelope per bank for dark baseline
        self.y_hi: dict[int, np.ndarray] = {}
        q70_list: list[float] = []
        for bk in self.obs_banks:
            arrs = [
                self.y_rb[(r, bk)] / self.exp_scale[r]
                for r in self.sel_runs
                if (r, bk) in self.y_rb
            ]
            st = np.stack(arrs, axis=0)
            yh = (
                np.max(st, axis=0) if len(arrs) < 5 else np.percentile(st, 90.0, axis=0)
            )
            self.y_hi[bk] = yh
            q70 = float(np.percentile(yh, 70.0))
            if q70 > 5.0:
                q70_list.append(q70)
        self.A_det = float(np.median(q70_list)) if q70_list else 0.0

        # Per-bank latent state responsibilities: shape (hb, wb)
        self.gamma_dark_prior: dict[int, np.ndarray] = {}
        self.log_odds_dark_prior: dict[int, np.ndarray] = {}
        self.log_odds_bright_persist: dict[int, np.ndarray] = {}

        self.gamma_dark: dict[int, np.ndarray] = {
            bk: np.zeros((self.hb, self.wb), dtype=np.float64) for bk in self.obs_banks
        }
        self.gamma_bright: dict[int, np.ndarray] = {
            bk: np.zeros((self.hb, self.wb), dtype=np.float64) for bk in self.obs_banks
        }
        self.gamma_valid: dict[int, np.ndarray] = {
            bk: np.ones((self.hb, self.wb), dtype=np.float64) for bk in self.obs_banks
        }

        # Masked background residuals & cached clean median filters per (r, bk)
        self._bg5_clean: dict[tuple[int, int], np.ndarray] = {
            k: median_filter(y, size=5, mode="nearest") for k, y in self.y_rb.items()
        }
        self._bg7_clean: dict[tuple[int, int], np.ndarray] = {
            k: median_filter(y, size=7, mode="nearest") for k, y in self.y_rb.items()
        }
        self._bg7_cur: dict[tuple[int, int], np.ndarray] = {}
        self._last_masks_bank: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.raw_dev_rb: dict[tuple[int, int], np.ndarray] = {}
        self.bright_dev_rb: dict[tuple[int, int], np.ndarray] = {}
        self.dev_rb: dict[tuple[int, int], np.ndarray] = {}
        self.weight_rb: dict[tuple[int, int], np.ndarray] = {}

        self._compute_prior()

    @classmethod
    def from_args(cls, args: Any) -> PixelEMClassifier | None:
        """Build classifier from ``run_scan_refinement`` / ``discover`` args.

        Returns ``None`` in peak-finder mode or if image frames are absent.
        """
        if getattr(args, "finder", None):
            return None
        nexus_path = getattr(args, "nexus", None)
        if not nexus_path:
            return None

        from subhkl.scan_refine import (
            _frame_offsets,
            _frame_rotations_and_scan_axes,
        )

        b = int(getattr(args, "binning", 8))
        bw = int(getattr(args, "border_width", 4))
        sel_runs = [int(r) for r in getattr(args, "runs", [0])]
        wl_raw = getattr(
            args,
            "assign_wavelength",
            (getattr(args, "wl_min", 1.0), getattr(args, "wl_max", 5.0)),
        )
        wavelength = (float(wl_raw[0]), float(wl_raw[1]))

        with h5py.File(nexus_path, "r") as f:
            if "images" not in f or "bank_ids" not in f:
                return None
            img_shape = f["images"].shape
            n_frames, n_rows, n_cols = (
                int(img_shape[0]),
                int(img_shape[1]),
                int(img_shape[2]),
            )
            if n_rows % b or n_cols % b:
                return None
            offsets = _frame_offsets(f, n_frames)
            R_frames, _, _, _, _ = _frame_rotations_and_scan_axes(f, n_frames)
            frame_banks = np.asarray(f["bank_ids"], int)
            run_ids = np.searchsorted(offsets, np.arange(n_frames), side="right") - 1

            hb, wb = n_rows // b, n_cols // b
            centers_r = np.arange(hb) * b + (b - 1) / 2.0
            centers_c = np.arange(wb) * b + (b - 1) / 2.0
            rr_0, cc_0 = np.meshgrid(centers_r, centers_c, indexing="ij")
            rows_full, cols_full = np.meshgrid(
                np.arange(n_rows, dtype=np.float64),
                np.arange(n_cols, dtype=np.float64),
                indexing="ij",
            )
            rows_bin = rows_full.reshape(hb, b, wb, b)
            cols_bin = cols_full.reshape(hb, b, wb, b)

            y_rb: dict[tuple[int, int], np.ndarray] = {}
            good_rb: dict[tuple[int, int], np.ndarray] = {}
            rr_rb: dict[tuple[int, int], np.ndarray] = {}
            cc_rb: dict[tuple[int, int], np.ndarray] = {}
            sub_peak_dev_rb: dict[tuple[int, int], np.ndarray] = {}
            bad_px_bank: dict[int, np.ndarray] = {}
            R_by_run: dict[int, np.ndarray] = {}
            obs_banks_set: set[int] = set()
            b_sub = (
                4 if (b >= 4 and b % 4 == 0) else (2 if (b >= 2 and b % 2 == 0) else 1)
            )
            k_sub = b // b_sub
            hs, ws = n_rows // b_sub, n_cols // b_sub

            mask_path = getattr(args, "static_mask_file", None)
            mask_file = h5py.File(mask_path, "r") if mask_path else None
            try:
                sel_set = set(sel_runs)
                mask_ids = None
                mask_key = None
                if mask_file is not None:
                    mask_key = "masks" if "masks" in mask_file else "images"
                    mask_ids = mask_file["bank_ids"][()]
                slab_sz = 64
                for s0 in range(0, n_frames, slab_sz):
                    s1 = min(s0 + slab_sz, n_frames)
                    if not any(int(run_ids[j]) in sel_set for j in range(s0, s1)):
                        continue
                    slab = np.asarray(f["images"][s0:s1], dtype=np.float32)
                    for i in range(s0, s1):
                        r = int(run_ids[i])
                        if r not in sel_set:
                            continue
                        bk = int(frame_banks[i])
                        obs_banks_set.add(bk)
                        if r not in R_by_run:
                            R_by_run[r] = np.asarray(R_frames[i], dtype=np.float64)
                        raw_f32 = slab[i - s0]
                        valid = np.isfinite(raw_f32) & (raw_f32 >= 0.0)
                        if mask_file is not None and mask_ids is not None:
                            matches = np.flatnonzero(mask_ids == bk)
                            if len(matches) == 1:
                                valid &= np.asarray(
                                    mask_file[mask_key][matches[0]], dtype=bool
                                )
                        if bk not in bad_px_bank:
                            bad_px_bank[bk] = ~valid
                        else:
                            bad_px_bank[bk] |= ~valid
                        if (
                            bw > 0
                            and n_rows > 4 * max(b, bw)
                            and n_cols > 4 * max(b, bw)
                        ):
                            raw_f32 = np.pad(raw_f32[bw:-bw, bw:-bw], bw, mode="edge")
                            if bw >= b:
                                cut = (bw // b) * b
                                valid[:cut, :] = False
                                valid[-cut:, :] = False
                                valid[:, :cut] = False
                                valid[:, -cut:] = False
                        clean_f32 = np.where(valid, raw_f32, np.float32(0.0))
                        raw_blocks = clean_f32.reshape(hb, b, wb, b)
                        y = raw_blocks.sum((1, 3), dtype=np.float64)
                        good = valid.reshape(hb, b, wb, b).all((1, 3))
                        if float(y.max()) > 0.0:
                            bg_5_init = median_filter(
                                y.astype(np.float32), size=5, mode="nearest"
                            )
                            excess_sq = (
                                np.maximum(
                                    raw_blocks
                                    - (bg_5_init / np.float32(b * b))[:, None, :, None],
                                    0.0,
                                ).astype(np.float64)
                                ** 2
                            )
                            excess_sum = excess_sq.sum((1, 3))
                            rr = np.where(
                                excess_sum > 0.0,
                                (excess_sq * rows_bin).sum((1, 3))
                                / np.maximum(excess_sum, 1e-12),
                                rr_0,
                            )
                            cc = np.where(
                                excess_sum > 0.0,
                                (excess_sq * cols_bin).sum((1, 3))
                                / np.maximum(excess_sum, 1e-12),
                                cc_0,
                            )
                        else:
                            rr = rr_0.copy()
                            cc = cc_0.copy()
                        y_rb[(r, bk)] = y
                        good_rb[(r, bk)] = good
                        rr_rb[(r, bk)] = rr
                        cc_rb[(r, bk)] = cc
                        if k_sub > 1:
                            if float(y.max()) > 0.0:
                                y_sub = clean_f32.reshape(hs, b_sub, ws, b_sub).sum(
                                    (1, 3), dtype=np.float32
                                )
                                bg_sub = median_filter(y_sub, size=7, mode="nearest")
                                d_sub = overdispersed_upward_deviance(
                                    y_sub.astype(np.float64),
                                    bg_sub.astype(np.float64),
                                    rel_sigma=0.09,
                                )
                                sub_peak_dev_rb[(r, bk)] = d_sub.reshape(
                                    hb, k_sub, wb, k_sub
                                ).max((1, 3))
                            else:
                                sub_peak_dev_rb[(r, bk)] = np.zeros(
                                    (hb, wb), dtype=np.float64
                                )
            finally:
                if mask_file is not None:
                    mask_file.close()

        if not obs_banks_set:
            return None

        return cls(
            instrument=args.instrument,
            n_rows=n_rows,
            n_cols=n_cols,
            binning=b,
            border_width=bw,
            sel_runs=sel_runs,
            obs_banks=sorted(obs_banks_set),
            y_rb=y_rb,
            good_rb=good_rb,
            rr_rb=rr_rb,
            cc_rb=cc_rb,
            bad_px_bank=bad_px_bank,
            R_by_run=R_by_run,
            cell=list(args.cell),
            space_group=args.space_group,
            wavelength=wavelength,
            d_min=float(args.d_min),
            sub_peak_dev_rb=sub_peak_dev_rb,
        )

    def _compute_prior(self) -> None:
        """Model-free E-step 0: static dark deficits + multi-frame static bright."""
        # Minimum ambient counts per bin (~0.4 counts/px on an 8x8 bin) required
        # before pure ambient deficit alone can flag shadows/dead panels.
        min_ambient_bin = max(25.0, 0.35 * (self.b * self.b))

        for bk in self.obs_banks:
            yh = self.y_hi[bk]
            if self.A_det < min_ambient_bin:
                self.log_odds_dark_prior[bk] = np.full(
                    (self.hb, self.wb), -9.0, dtype=np.float64
                )
                self.gamma_dark_prior[bk] = np.zeros(
                    (self.hb, self.wb), dtype=np.float64
                )
                continue

            q75 = float(np.percentile(yh, 75.0))
            if q75 < 0.08 * self.A_det:
                # Entire panel is unilluminated / dead
                self.log_odds_dark_prior[bk] = np.full(
                    (self.hb, self.wb), +25.0, dtype=np.float64
                )
                self.gamma_dark_prior[bk] = np.ones(
                    (self.hb, self.wb), dtype=np.float64
                )
                continue

            # Suppress 1-bin outer perimeter tube-edge glow before estimating
            # smooth panel ambient envelope, and respect naturally lower
            # diffuse background on high-angle backscattering panels.
            yh_int = yh.copy()
            if self.hb >= 4 and self.wb >= 4:
                yh_int[0, :] = np.minimum(yh_int[0, :], yh_int[1, :])
                yh_int[-1, :] = np.minimum(yh_int[-1, :], yh_int[-2, :])
                yh_int[:, 0] = np.minimum(yh_int[:, 0], yh_int[:, 1])
                yh_int[:, -1] = np.minimum(yh_int[:, -1], yh_int[:, -2])
            q75_int = float(np.percentile(yh_int, 75.0))
            A_pan = max(q75_int, 0.25 * self.A_det)
            win = max(5, min(21, (min(self.hb, self.wb) // 3) | 1))
            env = np.maximum(
                percentile_filter(yh_int, percentile=75.0, size=win, mode="nearest"),
                0.80 * A_pan,
            )
            floor_0 = 0.50 * env
            def_dev = np.zeros((self.hb, self.wb), dtype=np.float64)
            for r in self.sel_runs:
                if (r, bk) not in self.y_rb:
                    continue
                yt = self.y_rb[(r, bk)]
                mu = np.maximum(floor_0 * self.exp_scale[r], 1.0)
                ys = np.maximum(yt, 1e-6)
                def_dev += np.where(
                    yt < mu, 2.0 * (mu - yt + yt * np.log(ys / mu)), 0.0
                )
            def_cap = np.minimum(def_dev, 80.0)
            def_smooth = uniform_filter(def_cap, size=3, mode="nearest")
            eff_def = np.where(
                def_dev > 1.0,
                np.maximum(def_dev, 0.5 * def_dev + 0.5 * def_smooth),
                0.0,
            )
            lo_dark = np.clip(0.25 * eff_def - 9.0, -25.0, 25.0)
            self.log_odds_dark_prior[bk] = lo_dark
            self.gamma_dark_prior[bk] = 1.0 / (1.0 + np.exp(-lo_dark))

        self.gamma_dark = {bk: v.copy() for bk, v in self.gamma_dark_prior.items()}

        # Precompute immutable static-bright outlier deviance per (r, bk) once
        for (r, bk), y in self.y_rb.items():
            gd0 = self.gamma_dark_prior[bk]
            if np.all(gd0 > 0.85):
                self.bright_dev_rb[(r, bk)] = np.zeros_like(y)
                continue
            dark_mask0 = (gd0 > 0.5) | (~self.good_rb[(r, bk)])
            bg_5_0 = inpainted_median_bg(
                y, dark_mask0, None, size=5, bg_raw=self._bg5_clean[(r, bk)]
            )
            d_bin0 = overdispersed_upward_deviance(y, bg_5_0, rel_sigma=0.09)
            if (r, bk) in self.sub_peak_dev_rb:
                d_sub0 = np.where(dark_mask0, 0.0, self.sub_peak_dev_rb[(r, bk)])
                self.bright_dev_rb[(r, bk)] = np.maximum(d_bin0, 0.65 * d_sub0)
            else:
                self.bright_dev_rb[(r, bk)] = d_bin0

        # First pass of dark-masked residuals so shadow cliffs cannot mimic static bright
        self._recompute_residuals()

        # Multi-orientation static bright persistence (when >= 2 distinct goniometer R)
        for bk in self.obs_banks:
            if self.n_orient_clusters < 2:
                self.log_odds_bright_persist[bk] = np.zeros(
                    (self.hb, self.wb), dtype=np.float64
                )
                continue
            cluster_devs = []
            for cl in self.rot_clusters:
                arrs = [
                    self.bright_outlier_deviance(r, bk)
                    for r in cl
                    if (r, bk) in self.y_rb
                ]
                if arrs:
                    cluster_devs.append(np.max(np.stack(arrs, axis=0), axis=0))
            if len(cluster_devs) < 2:
                self.log_odds_bright_persist[bk] = np.zeros(
                    (self.hb, self.wb), dtype=np.float64
                )
                continue
            st_d = np.sort(np.stack(cluster_devs, axis=0), axis=0)
            # Require the bin to remain sharply bright across >= 75% of distinct
            # goniometer orientations (min across all clusters when K <= 3).
            k_ord = 0 if len(cluster_devs) <= 3 else int(0.25 * (len(cluster_devs) - 1))
            d_persist = st_d[k_ord]
            lo_persist = np.where(
                d_persist > 28.0,
                np.clip(0.35 * (d_persist - 28.0), 0.0, 30.0),
                0.0,
            )
            self.log_odds_bright_persist[bk] = lo_persist
            if np.any(lo_persist > 0.0):
                gb = (1.0 - self.gamma_dark[bk]) / (
                    1.0 + np.exp(-np.clip(lo_persist - 4.0, -30.0, 30.0))
                )
                self.gamma_bright[bk] = gb

        for bk in self.obs_banks:
            self.gamma_valid[bk] = np.clip(
                1.0 - self.gamma_dark[bk] - self.gamma_bright[bk], 0.0, 1.0
            )
        self._recompute_residuals()

    def _recompute_residuals(self) -> None:
        """Recompute local background, signed excess, and upward deviance."""
        for bk in self.obs_banks:
            gd = self.gamma_dark[bk]
            gb = self.gamma_bright[bk]
            gv = self.gamma_valid[bk]
            all_dead = bool(np.all(gd > 0.85))
            d_mask_b = gd > 0.5
            b_mask_b = gb >= 0.5
            prev = self._last_masks_bank.get(bk)
            masks_same = (
                prev is not None
                and np.array_equal(d_mask_b, prev[0])
                and np.array_equal(b_mask_b, prev[1])
            )
            if not masks_same:
                self._last_masks_bank[bk] = (d_mask_b.copy(), b_mask_b.copy())

            for r in self.sel_runs:
                key = (r, bk)
                if key not in self.y_rb:
                    continue
                y = self.y_rb[key]
                if all_dead:
                    z = np.zeros_like(y)
                    self.raw_dev_rb[key] = z
                    self._bg7_cur[key] = z
                    self.dev_rb[key] = z
                    self.weight_rb[key] = z
                    continue
                if not masks_same or key not in self.raw_dev_rb:
                    dark_mask = d_mask_b | (~self.good_rb[key])
                    bg_7 = inpainted_median_bg(
                        y,
                        dark_mask,
                        b_mask_b,
                        size=7,
                        bg_raw=self._bg7_clean[key],
                    )
                    bg_5 = inpainted_median_bg(
                        y,
                        dark_mask,
                        b_mask_b,
                        size=5,
                        bg_raw=self._bg5_clean[key],
                    )
                    self._bg7_cur[key] = bg_7
                    self.raw_dev_rb[key] = poisson_upward_deviance(y, bg_5)
                dev_raw = self.raw_dev_rb[key]
                self.dev_rb[key] = np.where(gd > 0.85, 0.0, dev_raw)
                self.weight_rb[key] = (y - self._bg7_cur[key]) * gv

    def raw_unweighted_deviance(self, r: int, bk: int) -> np.ndarray:
        """Dark/bright-inpainted upward deviance before multiplying by gamma_valid."""
        key = (r, bk)
        if key in self.raw_dev_rb:
            prev = self._last_masks_bank.get(bk)
            if (
                prev is not None
                and np.array_equal(self.gamma_dark[bk] > 0.5, prev[0])
                and np.array_equal(self.gamma_bright[bk] >= 0.5, prev[1])
            ):
                return self.raw_dev_rb[key]
        y = self.y_rb[key]
        gd = self.gamma_dark[bk]
        gb = self.gamma_bright[bk]
        if np.all(gd > 0.85):
            return np.zeros_like(y)
        dark_mask = (gd > 0.5) | (~self.good_rb[key])
        bright_mask = gb >= 0.5
        bg_5 = inpainted_median_bg(
            y, dark_mask, bright_mask, size=5, bg_raw=self._bg5_clean.get(key)
        )
        return poisson_upward_deviance(y, bg_5)

    def bright_outlier_deviance(self, r: int, bk: int) -> np.ndarray:
        """Localized overdispersed upward deviance for static-bright testing."""
        key = (r, bk)
        if key in self.bright_dev_rb:
            return self.bright_dev_rb[key]
        y = self.y_rb[key]
        gd = self.gamma_dark_prior[bk]
        if np.all(gd > 0.85):
            return np.zeros_like(y)
        dark_mask = (gd > 0.5) | (~self.good_rb[key])
        bg_5 = inpainted_median_bg(
            y, dark_mask, None, size=5, bg_raw=self._bg5_clean.get(key)
        )
        d_bin = overdispersed_upward_deviance(y, bg_5, rel_sigma=0.09)
        if key in self.sub_peak_dev_rb:
            d_sub = np.where(dark_mask, 0.0, self.sub_peak_dev_rb[key])
            return np.maximum(d_bin, 0.65 * d_sub)
        return d_bin

    def map_to_observations(
        self,
        obs_banks: np.ndarray,
        obs_pr: np.ndarray,
        obs_pc: np.ndarray,
        obs_runs: np.ndarray,
        fallback_dev: np.ndarray,
        fallback_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project current EM ``(dev, signed_weight, gamma_valid)`` onto flat obs."""
        n = len(obs_banks)
        out_dev = np.array(fallback_dev, dtype=np.float64, copy=True)
        out_w = np.array(fallback_weights, dtype=np.float64, copy=True)
        out_gv = np.ones(n, dtype=np.float64)

        br_all = np.clip((np.asarray(obs_pr) // self.b).astype(int), 0, self.hb - 1)
        bc_all = np.clip((np.asarray(obs_pc) // self.b).astype(int), 0, self.wb - 1)

        for (r, bk), dev_grid in self.dev_rb.items():
            m = (obs_runs == r) & (obs_banks == bk)
            if not np.any(m):
                continue
            rr_m = br_all[m]
            cc_m = bc_all[m]
            out_dev[m] = dev_grid[rr_m, cc_m]
            out_w[m] = self.weight_rb[(r, bk)][rr_m, cc_m]
            out_gv[m] = self.gamma_valid[bk][rr_m, cc_m]
        return out_dev, out_w, out_gv

    def _splat_hits_to_grid(
        self,
        r: int,
        bi_hit: np.ndarray,
        pr_hit: np.ndarray,
        pc_hit: np.ndarray,
        sigma_hit: np.ndarray | float,
    ) -> np.ndarray:
        """Vectorized 2D stencil Gaussian splat matching exact cKDTree nearest-spot M."""
        n_b = len(self.obs_banks)
        M_run = np.zeros((n_b, self.hb, self.wb), dtype=np.float64)
        if len(bi_hit) == 0:
            return M_run
        rr_run = self._rr_run[r]
        cc_run = self._cc_run[r]
        if np.ndim(sigma_hit) == 0:
            sig = float(sigma_hit)
            max_dist = 3.5 * sig
            max_d2_arr = max_dist * max_dist
            inv_2s2_arr = 0.5 / max(sig * sig, 1e-6)
        else:
            sig_arr = np.asarray(sigma_hit, dtype=np.float64)
            max_dist = float(3.5 * np.max(sig_arr))
            max_d2_arr = (3.5 * sig_arr) ** 2
            inv_2s2_arr = 0.5 / np.maximum(sig_arr * sig_arr, 1e-6)

        r_sten = int(np.ceil(max_dist / self.b)) + 1
        br0 = np.floor(pr_hit / self.b).astype(np.int32)
        bc0 = np.floor(pc_hit / self.b).astype(np.int32)

        for dr in range(-r_sten, r_sten + 1):
            br_t = br0 + dr
            m_r = (br_t >= 0) & (br_t < self.hb)
            if not np.any(m_r):
                continue
            idx_r = np.flatnonzero(m_r)
            bi_r = bi_hit[idx_r]
            br_r = br_t[idx_r]
            bc0_r = bc0[idx_r]
            pr_r = pr_hit[idx_r]
            pc_r = pc_hit[idx_r]
            md2_r = max_d2_arr if np.ndim(max_d2_arr) == 0 else max_d2_arr[idx_r]
            i2s_r = inv_2s2_arr if np.ndim(inv_2s2_arr) == 0 else inv_2s2_arr[idx_r]
            for dc in range(-r_sten, r_sten + 1):
                bc_t = bc0_r + dc
                m_c = (bc_t >= 0) & (bc_t < self.wb)
                if not np.any(m_c):
                    continue
                bi_g = bi_r[m_c]
                br_g = br_r[m_c]
                bc_g = bc_t[m_c]
                dr_px = pr_r[m_c] - rr_run[bi_g, br_g, bc_g]
                dc_px = pc_r[m_c] - cc_run[bi_g, br_g, bc_g]
                d2 = dr_px * dr_px + dc_px * dc_px
                md2_c = md2_r if np.ndim(md2_r) == 0 else md2_r[m_c]
                within = d2 <= md2_c
                if not np.any(within):
                    continue
                i2s_c = i2s_r if np.ndim(i2s_r) == 0 else i2s_r[m_c][within]
                val = np.exp(-d2[within] * i2s_c)
                np.maximum.at(M_run, (bi_g[within], br_g[within], bc_g[within]), val)
        return M_run

    def _build_bank_arrays(
        self,
        panel_cal: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]
        | None,
        radial_scale: float,
        axial_scale: float | None,
        global_rot_vec: np.ndarray | None,
        global_trans_m: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        a_sc = float(radial_scale if axial_scale is None else axial_scale)
        r_sc = float(radial_scale)
        rot_m = (
            Rotation.from_rotvec(
                np.asarray(global_rot_vec, dtype=np.float64)
            ).as_matrix()
            if global_rot_vec is not None
            else np.eye(3)
        )
        t_vec = (
            np.asarray(global_trans_m, dtype=np.float64)
            if global_trans_m is not None
            else np.zeros(3, dtype=np.float64)
        )
        n_b = len(self.obs_banks)
        cen_b = np.empty((n_b, 3), dtype=np.float64)
        uh_b = np.empty((n_b, 3), dtype=np.float64)
        vh_b = np.empty((n_b, 3), dtype=np.float64)
        nh_b = np.empty((n_b, 3), dtype=np.float64)
        mid_b = np.empty((n_b, 3), dtype=np.float64)
        half_diag_b = np.empty(n_b, dtype=np.float64)
        scale_r = np.empty(n_b, dtype=np.float64)
        scale_c = np.empty(n_b, dtype=np.float64)
        n_max_b = np.empty(n_b, dtype=np.float64)
        m_max_b = np.empty(n_b, dtype=np.float64)

        for bi, bk in enumerate(self.obs_banks):
            det_b = self._det_nom[bi]
            m_max = float(det_b.m - 1)
            n_max = float(det_b.n - 1)
            if panel_cal is not None and bk in panel_cal:
                cen, uh, vh, W, H = panel_cal[bk]
                mid_sc = cen + 0.5 * W * uh + 0.5 * H * vh
            else:
                mid = (
                    det_b.center
                    + 0.5 * det_b.width * det_b.uhat
                    + 0.5 * det_b.height * det_b.vhat
                )
                mid_sc = (
                    np.array(
                        [
                            mid[0] * (1.0 + r_sc),
                            mid[1] * (1.0 + a_sc),
                            mid[2] * (1.0 + r_sc),
                        ],
                        dtype=np.float64,
                    )
                    @ rot_m.T
                    + t_vec
                )
                uh = det_b.uhat @ rot_m.T
                vh = det_b.vhat @ rot_m.T
                W, H = float(det_b.width), float(det_b.height)
                cen = mid_sc - 0.5 * W * uh - 0.5 * H * vh
            cen_b[bi] = cen
            uh_b[bi] = uh
            vh_b[bi] = vh
            nh_b[bi] = np.cross(uh, vh)
            mid_b[bi] = mid_sc
            half_diag_b[bi] = 0.58 * np.hypot(W, H)
            scale_r[bi] = n_max / max(float(H), 1e-12)
            scale_c[bi] = m_max / max(float(W), 1e-12)
            n_max_b[bi] = n_max
            m_max_b[bi] = m_max
        return (
            cen_b,
            uh_b,
            vh_b,
            nh_b,
            mid_b,
            half_diag_b,
            scale_r,
            scale_c,
            n_max_b,
            m_max_b,
        )

    def _project_and_splat(
        self,
        U: np.ndarray,
        B: np.ndarray,
        *,
        R_runs: dict[int, np.ndarray] | None = None,
        panel_cal: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]
        | None = None,
        radial_scale: float = 0.0,
        axial_scale: float | None = None,
        global_rot_vec: np.ndarray | None = None,
        global_trans_m: np.ndarray | None = None,
        sample_shift_m: np.ndarray | None = None,
        match_sigma_px: float = 5.0,
        only_d_min: bool = False,
        dual_tight_sigma_px: float | None = None,
    ) -> (
        dict[tuple[int, int], np.ndarray]
        | tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]
    ):
        UB = np.asarray(U, dtype=np.float64) @ np.asarray(B, dtype=np.float64)
        G_cryst = self._hkl_em @ UB.T
        gn = np.linalg.norm(G_cryst, axis=1)
        in_core = (gn > 1e-7) & (gn <= 1.0 / self.d_min + 1e-7)
        if only_d_min:
            G_cryst = G_cryst[in_core]
            in_core = np.ones(len(G_cryst), dtype=bool)

        wl_pad = 0.05 if not only_d_min else 0.0
        wl_lo_em = max(0.1, self.wavelength[0] - wl_pad)
        wl_hi_em = self.wavelength[1] + wl_pad
        wl_lo_core = self.wavelength[0]
        wl_hi_core = self.wavelength[1]

        s_samp = (
            np.asarray(sample_shift_m, dtype=np.float64)
            if sample_shift_m is not None
            else np.zeros(3, dtype=np.float64)
        )
        (
            cen_b,
            uh_b,
            vh_b,
            nh_b,
            mid_b,
            half_diag_b,
            scale_r,
            scale_c,
            n_max_b,
            m_max_b,
        ) = self._build_bank_arrays(
            panel_cal, radial_scale, axial_scale, global_rot_vec, global_trans_m
        )

        M_out: dict[tuple[int, int], np.ndarray] = {}
        M_tight_out: dict[tuple[int, int], np.ndarray] = {}
        ki = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        sig_core = float(match_sigma_px)
        sig_hi = min(sig_core, 3.2)

        for r in self.sel_runs:
            pres = self._present_run[r]
            if not np.any(pres):
                continue
            Rr = np.asarray(
                (
                    R_runs[r]
                    if R_runs is not None and r in R_runs
                    else self.R_by_run.get(r, np.eye(3))
                ),
                dtype=np.float64,
            )
            xs = Rr @ s_samp
            G_lab = G_cryst @ Rr.T
            g2 = np.sum(G_lab * G_lab, axis=1)
            lam = -2.0 * G_lab[:, 2] / np.maximum(g2, 1e-12)
            m_wl = (g2 > 1e-12) & (lam >= wl_lo_em) & (lam <= wl_hi_em)
            if not np.any(m_wl):
                for bi, bk in enumerate(self.obs_banks):
                    if pres[bi]:
                        z = np.zeros((self.hb, self.wb), dtype=np.float64)
                        M_out[(r, bk)] = z
                        if dual_tight_sigma_px is not None:
                            M_tight_out[(r, bk)] = z
                continue

            lam_w = lam[m_wl]
            core_w = in_core[m_wl]
            kfp = ki[None, :] + lam_w[:, None] * G_lab[m_wl]
            kfp /= np.linalg.norm(kfp, axis=1, keepdims=True)

            c_eff = cen_b - xs[None, :]
            mid_eff = mid_b - xs[None, :]
            mid_dist = np.maximum(np.linalg.norm(mid_eff, axis=1), 1e-6)
            mid_hat = mid_eff / mid_dist[:, None]
            cos_cone = np.cos(
                np.minimum(np.arctan2(half_diag_b, mid_dist) + 0.08, np.pi)
            )

            num_b = np.einsum("bi,bi->b", c_eff, nh_b)
            dn = kfp @ nh_b.T
            cone_ok = (
                ((dn * num_b[None, :]) > 1e-9)
                & ((kfp @ mid_hat.T) >= cos_cone[None, :])
                & pres[None, :]
            )
            ray_i, bi_cand = np.nonzero(cone_ok)
            if len(ray_i) == 0:
                for bi, bk in enumerate(self.obs_banks):
                    if pres[bi]:
                        z = np.zeros((self.hb, self.wb), dtype=np.float64)
                        M_out[(r, bk)] = z
                        if dual_tight_sigma_px is not None:
                            M_tight_out[(r, bk)] = z
                continue

            t_ray = num_b[bi_cand] / dn[ray_i, bi_cand]
            ph = t_ray[:, None] * kfp[ray_i] - c_eff[bi_cand]
            pr = np.einsum("ij,ij->i", ph, vh_b[bi_cand]) * scale_r[bi_cand]
            pc = np.einsum("ij,ij->i", ph, uh_b[bi_cand]) * scale_c[bi_cand]
            ok = (
                (t_ray > 0.0)
                & (pr >= -self.b)
                & (pr <= n_max_b[bi_cand] + self.b)
                & (pc >= -self.b)
                & (pc <= m_max_b[bi_cand] + self.b)
            )
            bi_h = bi_cand[ok]
            pr_h = pr[ok]
            pc_h = pc[ok]
            ray_h = ray_i[ok]

            if only_d_min:
                M_run = self._splat_hits_to_grid(r, bi_h, pr_h, pc_h, sig_core)
            else:
                sig_h = np.where(core_w[ray_h], sig_core, sig_hi)
                M_run = self._splat_hits_to_grid(r, bi_h, pr_h, pc_h, sig_h)
            for bi, bk in enumerate(self.obs_banks):
                if pres[bi]:
                    M_out[(r, bk)] = M_run[bi]

            if dual_tight_sigma_px is not None:
                m_t = (
                    core_w[ray_h]
                    & (lam_w[ray_h] >= wl_lo_core)
                    & (lam_w[ray_h] <= wl_hi_core)
                )
                M_t_run = self._splat_hits_to_grid(
                    r,
                    bi_h[m_t],
                    pr_h[m_t],
                    pc_h[m_t],
                    float(dual_tight_sigma_px),
                )
                for bi, bk in enumerate(self.obs_banks):
                    if pres[bi]:
                        M_tight_out[(r, bk)] = M_t_run[bi]

        if dual_tight_sigma_px is not None:
            return M_out, M_tight_out
        return M_out

    def _predict_occupancy(
        self,
        U: np.ndarray,
        B: np.ndarray,
        *,
        R_runs: dict[int, np.ndarray] | None = None,
        panel_cal: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]
        | None = None,
        radial_scale: float = 0.0,
        axial_scale: float | None = None,
        global_rot_vec: np.ndarray | None = None,
        global_trans_m: np.ndarray | None = None,
        sample_shift_m: np.ndarray | None = None,
        match_sigma_px: float = 5.0,
        only_d_min: bool = False,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Compute predicted reflection proximity M_{r,bk} in [0, 1] per bin."""
        res = self._project_and_splat(
            U,
            B,
            R_runs=R_runs,
            panel_cal=panel_cal,
            radial_scale=radial_scale,
            axial_scale=axial_scale,
            global_rot_vec=global_rot_vec,
            global_trans_m=global_trans_m,
            sample_shift_m=sample_shift_m,
            match_sigma_px=match_sigma_px,
            only_d_min=only_d_min,
            dual_tight_sigma_px=None,
        )
        assert isinstance(res, dict)
        return res

    def _posteriors_from_occupancy(
        self, M_occ: dict[tuple[int, int], np.ndarray]
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        float,
    ]:
        """Closed-form posterior responsibilities given reflection occupancy M."""
        # Raw deviance without previous gamma_valid attenuation so EM is reversible
        raw_dev: dict[tuple[int, int], np.ndarray] = {
            (r, bk): self.raw_unweighted_deviance(r, bk) for (r, bk) in self.y_rb
        }
        bright_dev: dict[tuple[int, int], np.ndarray] = {
            (r, bk): self.bright_outlier_deviance(r, bk) for (r, bk) in self.y_rb
        }

        matched_devs = []
        matched_bdevs = []
        total_matched_hits = 0
        for (r, bk), M in M_occ.items():
            gd0 = self.gamma_dark_prior[bk]
            dv = raw_dev[(r, bk)]
            bdv = bright_dev[(r, bk)]
            hit = (M > 0.35) & (dv > 25.0) & (gd0 < 0.5)
            if np.any(hit):
                matched_devs.append(dv[hit])
                matched_bdevs.append(bdv[hit])
                total_matched_hits += int(hit.sum())

        if total_matched_hits >= 5:
            all_md = np.concatenate(matched_devs)
            all_mbd = np.concatenate(matched_bdevs)
            tau_cryst = float(np.clip(1.5 * np.median(all_md), 50.0, 600.0))
            tau_cryst_b = float(np.clip(1.5 * np.median(all_mbd), 20.0, 300.0))
        else:
            tau_cryst = 100.0
            tau_cryst_b = 45.0

        tau_bright_b = max(8.0 * tau_cryst_b, 400.0)
        scale_pred_b = 2.0 * tau_cryst_b
        scale_unmod_b = 1.4 * tau_cryst_b
        # On a single goniometer setting (N=1), goniometer persistence cannot
        # distinguish moving Bragg spots from static bright defects; include a
        # small Wilson-exponential unmodeled-reflection component and require a
        # clear outlier floor so real crystal Bragg peaks are never over-masked.
        single_setting_floor = (
            max(5.5 * tau_cryst_b, 220.0) if self.n_orient_clusters < 2 else 22.0
        )

        new_dark: dict[int, np.ndarray] = {}
        new_bright: dict[int, np.ndarray] = {}
        new_valid: dict[int, np.ndarray] = {}

        for bk in self.obs_banks:
            lo_dark = self.log_odds_dark_prior[bk].copy()

            # Model-informed dark extinction:
            # 1) Whole-panel extinction (catches dead panels even at zero diffuse bg)
            pred_sum_panel = 0.0
            obs_hit_panel = 0
            M_sum_grid = np.zeros((self.hb, self.wb), dtype=np.float64)
            max_dev_grid = np.zeros((self.hb, self.wb), dtype=np.float64)
            for r in self.sel_runs:
                if (r, bk) not in M_occ:
                    continue
                Mr = M_occ[(r, bk)]
                dvr = raw_dev[(r, bk)]
                yr = self.y_rb[(r, bk)]
                pred_sum_panel += float(np.sum(Mr > 0.5))
                obs_hit_panel += int(np.sum((Mr > 0.35) & ((dvr > 12.0) | (yr > 2.0))))
                M_sum_grid += Mr
                max_dev_grid = np.maximum(max_dev_grid, dvr)

            if (
                total_matched_hits >= 15
                and pred_sum_panel >= 6.0
                and obs_hit_panel == 0
            ):
                lo_dark = np.maximum(lo_dark, +20.0)
            elif total_matched_hits >= 15:
                # 2) Patch-level extinction (3x3 bin neighborhood with >= 2.5 predicted
                # reflections and zero observed signal + depressed local counts)
                patch_M = uniform_filter(M_sum_grid, size=3, mode="nearest") * 9.0
                patch_max_d = uniform_filter(max_dev_grid, size=3, mode="nearest")
                pan_med = max(float(np.percentile(self.y_hi[bk], 70.0)), 1.0)
                depressed = self.y_hi[bk] < 0.70 * pan_med
                ext_patch = (patch_M >= 2.5) & (patch_max_d < 4.0) & depressed
                lo_dark = np.where(ext_patch, lo_dark + 1.8 * (patch_M - 2.0), lo_dark)

            # Generative bright vs valid log-likelihood summed across distinct
            # goniometer orientation clusters (no double-counting of persistence).
            ll_valid_sum = np.zeros((self.hb, self.wb), dtype=np.float64)
            ll_bright_sum = np.zeros((self.hb, self.wb), dtype=np.float64)
            max_bdev_grid = np.zeros((self.hb, self.wb), dtype=np.float64)
            n_active_cl = 0
            for cl in self.rot_clusters:
                ms_cl = [M_occ[(r, bk)] for r in cl if (r, bk) in M_occ]
                ds_cl = [bright_dev[(r, bk)] for r in cl if (r, bk) in bright_dev]
                if not ms_cl:
                    continue
                n_active_cl += 1
                Mc = np.clip(np.max(np.stack(ms_cl, axis=0), axis=0), 1e-5, 1.0 - 1e-5)
                dvc = np.mean(np.stack(ds_cl, axis=0), axis=0)
                max_bdev_grid = np.maximum(max_bdev_grid, dvc)
                # Crystallographic Wilson exponential intensity law for Bragg spots
                ll_bragg = np.log(Mc) - np.log(scale_pred_b) - dvc / scale_pred_b
                if self.n_orient_clusters < 2:
                    eps_u = 0.015
                    ll_unmod = (
                        np.log((1.0 - Mc) * eps_u)
                        - np.log(scale_unmod_b)
                        - dvc / scale_unmod_b
                    )
                    ll_bg = np.log((1.0 - Mc) * (1.0 - eps_u)) - 0.5 * dvc
                    ll_valid_sum += np.logaddexp(
                        ll_bragg, np.logaddexp(ll_unmod, ll_bg)
                    )
                else:
                    ll_bg = np.log(1.0 - Mc) - 0.5 * dvc
                    ll_valid_sum += np.logaddexp(ll_bragg, ll_bg)
                # Heavy Cauchy tail for parasitic static bright artifacts;
                # penalize near-zero deviance in frames where pixel turns off.
                ll_bright_sum += (
                    -np.log(tau_bright_b)
                    - np.log1p(dvc / tau_bright_b)
                    - (4.0 * np.exp(-dvc / 6.0) if self.n_orient_clusters >= 2 else 0.0)
                )

            lo_bright_model = np.log(5e-4) + (ll_bright_sum - ll_valid_sum)
            eligible_bright = max_bdev_grid >= single_setting_floor
            if n_active_cl >= 2:
                eligible_bright &= self.log_odds_bright_persist[bk] > 0.0
            lo_bright = np.where(
                eligible_bright, np.clip(lo_bright_model, -25.0, 25.0), -25.0
            )
            lo_dark = np.clip(lo_dark, -25.0, 25.0)

            # Softmax over (0.0, lo_dark, lo_bright)
            m_max = np.maximum(0.0, np.maximum(lo_dark, lo_bright))
            ev = np.exp(-m_max)
            ed = np.exp(lo_dark - m_max)
            eb = np.exp(lo_bright - m_max)
            denom = ev + ed + eb
            new_valid[bk] = ev / denom
            new_dark[bk] = ed / denom
            new_bright[bk] = eb / denom

        return new_valid, new_dark, new_bright, tau_cryst

    def evaluate_candidate(
        self,
        U: np.ndarray,
        B: np.ndarray,
        *,
        R_runs: dict[int, np.ndarray] | None = None,
        radial_scale: float = 0.0,
        axial_scale: float | None = None,
        global_rot_vec: np.ndarray | None = None,
        global_trans_m: np.ndarray | None = None,
        match_sigma_px: float = 4.5,
        geom_prior_weight: float = 2.0,
    ) -> dict[str, Any]:
        """Evaluate candidate (U, geom) under the 3-state EM Wilson objective.

        Uses null-subtracted occupancy ``(M - mean_panel(M))`` weighted by each
        bin's learned ``gamma_valid`` and Wilson log-deviance ``log(1 + D+/80)``,
        eliminating panel solid-angle magnification bias and Al-flare hijacking.
        """
        dual = self._project_and_splat(
            U,
            B,
            R_runs=R_runs,
            radial_scale=radial_scale,
            axial_scale=axial_scale,
            global_rot_vec=global_rot_vec,
            global_trans_m=global_trans_m,
            match_sigma_px=max(float(match_sigma_px) * 1.15, 5.0),
            only_d_min=False,
            dual_tight_sigma_px=float(match_sigma_px),
        )
        assert isinstance(dual, tuple)
        M_em, M_tight = dual
        gv_map, gd_map, gb_map, tau_cryst = self._posteriors_from_occupancy(M_em)

        llr_excess = 0.0
        n_matched = 0
        for (r, bk), Mt in M_tight.items():
            gv = gv_map[bk]
            valid_mask = self.good_rb[(r, bk)] & (gd_map[bk] < 0.5)
            if not np.any(valid_mask):
                continue
            dv = self.raw_unweighted_deviance(r, bk)
            w_log = gv * np.log1p(dv / 80.0)
            m_bar = float(np.mean(Mt[valid_mask]))
            llr_excess += float(np.sum((Mt[valid_mask] - m_bar) * w_log[valid_mask]))
            n_matched += int(np.sum((Mt > 0.5) & (dv > 36.0) & (gv > 0.5)))

        a_sc = float(radial_scale if axial_scale is None else axial_scale)
        r_sc = float(radial_scale)
        rvec = (
            np.asarray(global_rot_vec, dtype=np.float64)
            if global_rot_vec is not None
            else np.zeros(3)
        )
        tvec = (
            np.asarray(global_trans_m, dtype=np.float64)
            if global_trans_m is not None
            else np.zeros(3)
        )
        geom_pen = float(geom_prior_weight) * (
            ((r_sc - a_sc) / 0.03) ** 2
            + (r_sc / 0.12) ** 2
            + (a_sc / 0.12) ** 2
            + float(np.sum((rvec / np.deg2rad(2.5)) ** 2))
            + float(np.sum((tvec / 0.02) ** 2))
        )
        return {
            "em_score": llr_excess - geom_pen,
            "llr_excess": llr_excess,
            "geom_penalty": geom_pen,
            "n_matched": n_matched,
            "tau_cryst": tau_cryst,
            "gamma_valid": gv_map,
            "gamma_dark": gd_map,
            "gamma_bright": gb_map,
        }

    def update_e_step(
        self,
        U: np.ndarray,
        B: np.ndarray,
        *,
        R_runs: dict[int, np.ndarray] | None = None,
        panel_cal: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]
        | None = None,
        radial_scale: float = 0.0,
        axial_scale: float | None = None,
        global_rot_vec: np.ndarray | None = None,
        global_trans_m: np.ndarray | None = None,
        sample_shift_m: np.ndarray | None = None,
        match_sigma_px: float = 6.0,
    ) -> dict[str, float]:
        """Run one closed-form E-step and refresh masked residuals in-place."""
        M_occ = self._predict_occupancy(
            U,
            B,
            R_runs=R_runs,
            panel_cal=panel_cal,
            radial_scale=radial_scale,
            axial_scale=axial_scale,
            global_rot_vec=global_rot_vec,
            global_trans_m=global_trans_m,
            sample_shift_m=sample_shift_m,
            match_sigma_px=match_sigma_px,
            only_d_min=False,
        )
        gv, gd, gb, tau_cryst = self._posteriors_from_occupancy(M_occ)
        self.gamma_valid = gv
        self.gamma_dark = gd
        self.gamma_bright = gb
        self._recompute_residuals()

        n_tot = max(len(self.obs_banks) * self.hb * self.wb, 1)
        n_dark = sum(int(np.sum(v > 0.5)) for v in self.gamma_dark.values())
        n_bright = sum(int(np.sum(v > 0.5)) for v in self.gamma_bright.values())
        return {
            "tau_cryst": tau_cryst,
            "dark_bins": n_dark,
            "bright_bins": n_bright,
            "valid_fraction": float(n_tot - n_dark - n_bright) / n_tot,
        }

    def apply_responsibilities(
        self,
        gamma_valid: dict[int, np.ndarray],
        gamma_dark: dict[int, np.ndarray],
        gamma_bright: dict[int, np.ndarray],
    ) -> None:
        """Install precomputed candidate responsibilities and refresh residuals."""
        self.gamma_valid = {bk: v.copy() for bk, v in gamma_valid.items()}
        self.gamma_dark = {bk: v.copy() for bk, v in gamma_dark.items()}
        self.gamma_bright = {bk: v.copy() for bk, v in gamma_bright.items()}
        self._recompute_residuals()

    def filter_ewald_excess_maps(
        self, exc_maps: dict[tuple[int, int], np.ndarray], b2: int
    ) -> dict[tuple[int, int], np.ndarray]:
        """Mask dark/bright bins and apply Wilson log-compression to Ewald maps.

        Suppresses dead panels, shadow-edge steps, and parasitic Al flares while
        preserving sub-bin reflection peaks on valid detector regions.
        """
        if not exc_maps:
            return exc_maps
        hb2 = self.n_rows // int(b2)
        wb2 = self.n_cols // int(b2)
        out: dict[tuple[int, int], np.ndarray] = {}

        # Upsample/downsample bin masks from (hb, wb) to (hb2, wb2)
        gv_b2: dict[int, np.ndarray] = {}
        valid_bg_b2: dict[int, np.ndarray] = {}
        r_idx = np.clip(
            ((np.arange(hb2) + 0.5) * b2 / self.b).astype(int), 0, self.hb - 1
        )
        c_idx = np.clip(
            ((np.arange(wb2) + 0.5) * b2 / self.b).astype(int), 0, self.wb - 1
        )
        for bk in self.obs_banks:
            gd = self.gamma_dark[bk]
            gb = self.gamma_bright[bk]
            dark_dil = (
                binary_dilation(gd > 0.4, iterations=1)
                if np.any(gd > 0.4)
                else np.zeros_like(gd, dtype=bool)
            )
            bright_dil = (
                binary_dilation(gb > 0.5, iterations=1)
                if np.any(gb > 0.5)
                else np.zeros_like(gb, dtype=bool)
            )
            gv_sub = np.where(dark_dil | bright_dil, 0.0, self.gamma_valid[bk])
            gv_b2[bk] = gv_sub[np.ix_(r_idx, c_idx)].astype(np.float32)
            valid_bg_b2[bk] = (~(dark_dil | bright_dil))[np.ix_(r_idx, c_idx)]

        # Recompute clean 4x4 excess where dark/bright artifacts exist, then
        # apply smooth Wilson log-compression so flares cannot dominate Ewald score.
        pos_vals = []
        raw_masked: dict[tuple[int, int], np.ndarray] = {}
        for (r, bk), ex in exc_maps.items():
            if bk in gv_b2 and ex.shape == (hb2, wb2):
                vbg = valid_bg_b2[bk]
                if not np.all(vbg):
                    # Inpaint masked bins on existing excess map & re-center locally
                    ex_clean = ex * gv_b2[bk]
                    # Zero a 1-bin collar on b2 grid around dark cliffs to kill step residue
                    v_erode = ~binary_dilation(~vbg, iterations=1)
                    ex_clean = np.where(v_erode, ex_clean, 0.0).astype(np.float32)
                else:
                    ex_clean = (ex * gv_b2[bk]).astype(np.float32)
            else:
                ex_clean = ex
            raw_masked[(r, bk)] = ex_clean
            p = ex_clean[ex_clean > 0.0]
            if len(p):
                pos_vals.append(p)

        if pos_vals:
            all_pos = np.concatenate(pos_vals)
            tau_ew = float(max(np.quantile(all_pos, 0.95), 10.0))
        else:
            tau_ew = 50.0

        for k, ex_clean in raw_masked.items():
            out[k] = (
                tau_ew * np.sign(ex_clean) * np.log1p(np.abs(ex_clean) / tau_ew)
            ).astype(np.float32)
        return out

    def export_pixel_masks(
        self,
        *,
        dilate_bright_bins: int = 1,
        dilate_dark_px: int = 2,
    ) -> EMPixelMaskExport:
        """Export full-resolution ``(n_banks, H, W)`` pixel masks and classes."""
        n_b = len(self.obs_banks)
        pixel_mask = np.ones((n_b, self.n_rows, self.n_cols), dtype=np.uint8)
        pixel_class = np.zeros((n_b, self.n_rows, self.n_cols), dtype=np.uint8)
        dead_banks: list[int] = []
        n_bright_bins = 0
        n_dark_bins = 0

        for i, bk in enumerate(self.obs_banks):
            gd = self.gamma_dark[bk]
            gb = self.gamma_bright[bk]
            dark_b = gd > 0.5
            bright_b = (gb > 0.5) & (~dark_b)
            n_dark_bins += int(dark_b.sum())
            n_bright_bins += int(bright_b.sum())

            if float(dark_b.mean()) > 0.90:
                dead_banks.append(int(bk))
                pixel_mask[i, :, :] = 0
                pixel_class[i, :, :] = 1
                continue

            if dilate_bright_bins > 0 and np.any(bright_b):
                max_bdev_bk = np.zeros((self.hb, self.wb), dtype=np.float64)
                for r in self.sel_runs:
                    if (r, bk) in self.bright_dev_rb:
                        max_bdev_bk = np.maximum(
                            max_bdev_bk, self.bright_dev_rb[(r, bk)]
                        )
                halo_cand = binary_dilation(
                    bright_b, iterations=int(dilate_bright_bins)
                ) & (~dark_b)
                bright_b_dil = bright_b | (halo_cand & (max_bdev_bk >= 15.0))
            else:
                bright_b_dil = bright_b

            dark_px = np.repeat(np.repeat(dark_b, self.b, axis=0), self.b, axis=1)
            if bk in self.bad_px_bank:
                dark_px |= self.bad_px_bank[bk]
            if dilate_dark_px > 0 and np.any(dark_px):
                dark_px = binary_dilation(dark_px, iterations=int(dilate_dark_px))

            bright_px = np.repeat(
                np.repeat(bright_b_dil, self.b, axis=0), self.b, axis=1
            ) & (~dark_px)

            cls_i = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)
            cls_i[dark_px] = 1
            cls_i[bright_px] = 2
            pixel_class[i] = cls_i
            pixel_mask[i] = (cls_i == 0).astype(np.uint8)

        tot_px = max(pixel_mask.size, 1)
        valid_frac = float(np.sum(pixel_class == 0)) / tot_px
        dark_frac = float(np.sum(pixel_class == 1)) / tot_px
        bright_frac = float(np.sum(pixel_class == 2)) / tot_px
        return EMPixelMaskExport(
            bank_ids=np.asarray(self.obs_banks, dtype=np.int32),
            pixel_mask=pixel_mask,
            pixel_class=pixel_class,
            valid_fraction=valid_frac,
            dark_fraction=dark_frac,
            bright_fraction=bright_frac,
            dead_banks=dead_banks,
            n_bright_bins=n_bright_bins,
            n_dark_bins=n_dark_bins,
        )
