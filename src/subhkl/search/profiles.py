"""Shared empirical reflection-family profiles for finding, integration and solving.

The measurement routines retain their bright-window qualification rules.
They estimate a shape prior; they do not select pixels for the count solver.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadialProfile:
    """Shared (u, f) shape prior; u is radius in model sigma units."""

    u: np.ndarray
    f: np.ndarray

    def __post_init__(self):
        u, f = np.asarray(self.u, float), np.asarray(self.f, float)
        if (
            u.ndim != 1
            or len(u) < 2
            or f.shape != u.shape
            or not np.isfinite(u).all()
            or not np.isfinite(f).all()
            or u[0] < 0
            or np.any(np.diff(u) <= 0)
            or np.any(f < 0)
            or not np.any(f > 0)
        ):
            raise ValueError(
                "profile requires increasing nonnegative u and finite nonnegative f"
            )
        object.__setattr__(self, "u", u.copy())
        object.__setattr__(self, "f", f.copy())

    def integrate_bins(self, rows, cols, center_r, center_c, sigma):
        """Integrate over unit bins, normalized on the entire unmasked plane."""
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("profile sigma must be finite and positive")
        # Clip integration intervals to the footprint before quadrature.
        # A very narrow profile inside a coarse bin must not fall between
        # quadrature nodes and disappear. Work per model sigma, bounding the
        # node count even when sigma is tiny relative to a count bin.
        bound = self.u[-1] * sigma
        order = max(4, int(np.ceil(4 * min(1 / sigma, 2 * self.u[-1]))))
        nodes, weights = np.polynomial.legendre.leggauss(order)

        def axis_bins(bins, center):
            lo = np.clip(bins - center, -bound, bound)
            hi = np.clip(bins + 1 - center, -bound, bound)
            half = (hi - lo) / 2
            points = ((hi + lo)[:, None] / 2 + half[:, None] * nodes) / sigma
            return points, half[:, None] * weights

        dr, wr = axis_bins(np.asarray(rows), center_r)
        dc, wc = axis_bins(np.asarray(cols), center_c)
        radius = np.hypot(dr[:, None, :, None], dc[None, :, None, :])
        values = np.interp(radius, self.u, self.f, left=self.f[0], right=0.0)
        u = np.r_[0.0, self.u] if self.u[0] > 0 else self.u
        f = np.r_[self.f[0], self.f] if self.u[0] > 0 else self.f
        slope = np.diff(f) / np.diff(u)
        intercept = f[:-1] - slope * u[:-1]
        integral = np.sum(slope * np.diff(u**3) / 3 + intercept * np.diff(u**2) / 2)
        return np.einsum("rcij,ri,cj->rc", values, wr, wc) / (
            2 * np.pi * integral * sigma**2
        )


#: Two-aperture consistency (see the width census): a neighbour strong enough
#: to inflate the wide aperture inflates the tight one differently, and 6%
#: keeps every clean synthetic peak while removing the runaway tail.
_WIDTH_TIGHT_APERTURE = 2.5
_WIDTH_WIDE_APERTURE = 3.5
_WIDTH_CONSISTENCY_TOL = 0.06


def _truncated_m2_ratio(x):
    """``m2(R)/sigma**2`` for a 2D Gaussian truncated at R, ``x = R**2/(2 sigma**2)``."""
    if x <= 0:
        return 0.0
    if x > 60:
        return 2.0
    ex = np.exp(-x)
    return 2.0 * (1.0 - (1.0 + x) * ex) / (1.0 - ex)


def _sigma_from_truncated_m2(m2, aperture):
    """Invert the truncation: the sigma whose truncated m2 matches."""
    from scipy.optimize import brentq

    if m2 <= 0 or aperture <= 0:
        return float("nan")

    def resid(sigma):
        return sigma**2 * _truncated_m2_ratio(aperture**2 / (2 * sigma**2)) - m2

    try:
        if resid(0.05) * resid(60.0) > 0:
            return float("nan")
        return float(brentq(resid, 0.05, 60.0, xtol=1e-6))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _width_at_aperture(frame, bg_patch_src, row, col, guess, max_half, k):
    """Truncation-corrected sigma at an aperture of ``k`` sigma, iterated
    with damping so it cannot oscillate.  Centroid from clipped weights
    (stability), moments from *unclipped* values inside a circular aperture:
    clipping rectifies background noise into a pedestal that inflates m2."""
    height, width = frame.shape
    r0, c0 = round(float(row)), round(float(col))
    outer = max_half
    if not (outer < r0 < height - outer and outer < c0 < width - outer):
        return float("nan")
    sl_out = (slice(r0 - outer, r0 + outer + 1), slice(c0 - outer, c0 + outer + 1))
    raw_out = frame[sl_out]
    bg_out = bg_patch_src[sl_out]
    rr_out, cc_out = np.mgrid[sl_out].astype(float)
    d2_0 = (rr_out - row) ** 2 + (cc_out - col) ** 2
    ring = (d2_0 >= (outer - 3.0) ** 2) & (d2_0 <= outer**2)
    bg_offset = (
        float(np.median(raw_out[ring] - bg_out[ring])) if ring.sum() >= 8 else 0.0
    )
    # In the bright-background regime (bg >= 16 counts/px), compute_rate_batch
    # falls back to a windowed box mean, which a bright peak's own flux
    # inflates by +3..6 counts/px inside the filter window.  That positive
    # background bias makes the peak skirt go negative inside the wide
    # aperture while leaving the tight aperture positive, so genuine bright
    # peaks fail two-aperture consistency while broad pedestals pass.  Adding
    # min(bg_offset, 0.0) removes the peak-induced over-subtraction without
    # ever raising the background on a clean frame.
    bg_corr = min(bg_offset, 0.0)

    sigma = guess
    for _ in range(12):
        aperture = min(k * sigma, max_half)
        half = int(np.ceil(aperture)) + 1
        if not (half < r0 < height - half and half < c0 < width - half):
            return float("nan")
        sl = (slice(r0 - half, r0 + half + 1), slice(c0 - half, c0 + half + 1))
        patch = frame[sl] - (bg_patch_src[sl] + bg_corr)
        rr, cc = np.mgrid[sl].astype(float)
        pos = np.maximum(patch, 0.0)
        if pos.sum() <= 0:
            return float("nan")
        cr = (pos * rr).sum() / pos.sum()
        cq = (pos * cc).sum() / pos.sum()
        d2 = (rr - cr) ** 2 + (cc - cq) ** 2
        inside = d2 <= aperture**2
        flux = patch[inside].sum()
        if flux <= 0:
            return float("nan")
        m2 = (patch[inside] * d2[inside]).sum() / flux
        if not np.isfinite(m2) or m2 <= 0:
            return float("nan")
        new = _sigma_from_truncated_m2(m2, aperture)
        if not np.isfinite(new):
            return float("nan")
        if abs(new - sigma) < 0.01:
            sigma = float(new)
            break
        sigma = 0.5 * (sigma + new)
    # If the tight aperture itself clips at max_half, both tight and wide
    # apertures evaluate the identical disk R = max_half and agree trivially
    # (|wide - tight|/tight = 0) on any window-filling shadow or flare.
    if _WIDTH_TIGHT_APERTURE * sigma > max_half - 0.5:
        return float("nan")
    return float(sigma)


def _filter_widths_by_prior(widths, two_theta=None):
    """Filter candidate peak widths against a smooth monotonic prior vs.
    scattering angle ``two_theta`` (degrees) or resolution ``sin(theta)``.

    Diffraction spot widths vary smoothly and monotonically with ``sin(theta)``
    (wavelength spread, divergence, mosaicity, and oblique incidence), whereas
    detector artifacts (direct-beam halos at low ``2theta``, panel shadows, or
    unmasked edge steps) appear as isolated large-radius outliers or
    non-monotonic spikes.  Fitting a robust non-decreasing quadratic prior
    ``sigma(s) = b0 + b1 * s + b2 * s**2`` in ``s = sin(theta)`` with
    ``b1, b2 >= 0`` and asymmetric upward Cauchy loss rejects both halo
    outliers and non-monotonic artifact clusters.
    """
    from scipy.optimize import least_squares

    w = np.asarray(widths, dtype=float)
    if w.size < 8:
        return w, np.ones(w.size, dtype=bool), w.copy()

    q25 = float(np.percentile(w, 25.0))
    q50 = float(np.percentile(w, 50.0))
    scale = max(1.4826 * (q50 - q25), 0.05 * q50, 0.08)

    if two_theta is None:
        pred = np.full_like(w, q50)
        diff = w - pred
        inlier_mad = max(
            float(np.median(np.abs(diff[diff <= scale]))) * 1.4826
            if np.any(diff <= scale)
            else scale,
            0.05 * q50,
            0.08,
        )
        keep = (w >= 0.5 * pred) & (w <= pred + 2.5 * inlier_mad)
        return w[keep], keep, pred

    tt = np.asarray(two_theta, dtype=float)
    s = np.sin(0.5 * np.radians(np.clip(tt, 0.0, 179.0)))

    def resid(p):
        pred_s = p[0] + p[1] * s + p[2] * (s**2)
        r_data = (w - pred_s) / scale
        r_asym = np.where(r_data > 0, 2.5 * r_data, r_data)
        w_reg = np.sqrt(max(w.size, 1))
        r_reg = np.array([w_reg * p[1] / (0.15 * q50), w_reg * p[2] / (0.10 * q50)])
        return np.concatenate([r_asym, r_reg])

    try:
        res = least_squares(
            resid,
            x0=[q25, 0.05 * q25, 0.0],
            bounds=([0.3, 0.0, 0.0], [np.inf, 0.35 * q50, 0.20 * q50]),
            loss="cauchy",
            f_scale=1.0,
        )
        pred = res.x[0] + res.x[1] * s + res.x[2] * (s**2)
    except (ValueError, RuntimeError):
        pred = np.full_like(w, q50)

    diff = w - pred
    inlier_sel = diff <= scale
    inlier_mad = max(
        float(np.median(np.abs(diff[inlier_sel]))) * 1.4826
        if np.any(inlier_sel)
        else scale,
        0.04 * float(np.median(pred)),
        0.08,
    )
    keep = (w >= 0.5 * pred) & (w <= pred + 2.0 * inlier_mad)
    return w[keep], keep, pred


def _lookup_two_theta(two_theta, fi, row, col, H, W):
    """Evaluate scattering angle ``two_theta`` (degrees) at ``(fi, row, col)``,
    or fall back to radial distance from the frame centre mapped onto
    ``[0, 90]`` degrees when no detector geometry is supplied."""
    if two_theta is not None:
        if callable(two_theta):
            return float(two_theta(fi, float(row), float(col)))
        tt_arr = np.asarray(two_theta)
        if tt_arr.ndim == 1:
            return float(tt_arr[fi])
        r_idx = int(np.clip(round(float(row)), 0, tt_arr.shape[-2] - 1))
        c_idx = int(np.clip(round(float(col)), 0, tt_arr.shape[-1] - 1))
        if tt_arr.ndim == 2:
            return float(tt_arr[r_idx, c_idx])
        return float(tt_arr[fi, r_idx, c_idx])
    dr_c = (float(row) - 0.5 * (H - 1)) / max(0.5 * H, 1.0)
    dc_c = (float(col) - 0.5 * (W - 1)) / max(0.5 * W, 1.0)
    return float(np.clip(np.hypot(dr_c, dc_c) * 60.0, 0.0, 90.0))


def measure_radial_profile(
    images,
    bg_map,
    bg_hi,
    max_sigma,
    min_windows=8,
    u_max=4.0,
    du=0.1,
    valid=None,
    two_theta=None,
):
    """The peak family's radial profile, measured from the frames themselves.

    The low-rank result that motivates this: stacking bright isolated peaks
    (recentred on moment centroids, rescaled by moment widths) leaves the mean
    profile carrying ~95% of the family's energy, so one measured trunk is the
    family up to anisotropy.  Measuring it is the same box arithmetic as the
    amplitude census -- no solve, no fit, no functional form:

      per qualifying window: float centroid, m2 -> sigma_w; bin the
      background-subtracted counts by u = r/sigma_w; normalise to unit flux;
      average windows weighted by flux.

    Binning around the *float* centroid handles sub-pixel centring without
    interpolation.  Windows are the Voronoi (counting) census, so each peak
    contributes once; the flux floor already clears 8 sigma of window-sum
    noise, so a peak-free frame contributes nothing rather than a noise
    profile.  Returns ``(u, f)`` with ``f[0] = 1``, or ``None`` when fewer
    than ``min_windows`` qualify -- the caller then stays on the Gaussian,
    which is the safe default rather than a profile made of too few peaks.
    """
    frames = np.asarray(images, dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None, ...]
    bgs = np.asarray(bg_map, dtype=np.float32)
    if bgs.ndim == 2:
        bgs = bgs[None, ...]
    excess = np.asarray(images, dtype=np.float64) - np.asarray(bg_map, dtype=np.float64)
    if excess.ndim == 2:
        excess = excess[None, ...]
    if valid is not None:
        valid = np.asarray(valid)
        if valid.ndim == 2:
            valid = valid[None, ...]
    edges = np.arange(0.0, u_max + du, du)
    centres = 0.5 * (edges[1:] + edges[:-1])
    acc = np.zeros(centres.size)
    wgt = np.zeros(centres.size)
    candidates = []
    for fi, frame in enumerate(excess):
        H, W = frame.shape
        # The window must hold a max_sigma peak out to ~2.5 sigma, or its
        # second moment is so truncated that every measured width -- and with
        # it the u = r/sigma axis -- is wrong.  Measured cost of the census
        # default (17 px on a 512 frame, sigma up to 6.5): a pure-Gaussian
        # synthetic read 43% too broad at u = 1.5.
        step = max(8, int(np.ceil(2.5 * float(max_sigma))))
        yy, xx = np.mgrid[-step : step + 1, -step : step + 1].astype(float)
        for r0 in range(step, H - step, step):
            for c0 in range(step, W - step, step):
                # See _moment_census: masked structure must not become the
                # measured peak profile.
                if (
                    valid is not None
                    and valid[
                        fi, r0 - step : r0 + step + 1, c0 - step : c0 + step + 1
                    ].min()
                    < 1.0
                ):
                    continue
                win = frame[r0 - step : r0 + step + 1, c0 - step : c0 + step + 1]
                flux = float(win.sum())
                area = win.size
                if flux < max(
                    20.0 * bg_hi, 50.0, 8.0 * np.sqrt(max(bg_hi, 0.05) * area)
                ):
                    continue
                # Centroid from clipped weights (stability); everything after
                # from *unclipped* values inside a *circular* mask.  Clipping
                # rectifies background noise into a pedestal that inflates m2,
                # and a square window's corners do the same relative to the
                # circular truncation model inverted below -- together they
                # read a pure-Gaussian synthetic 20-40% wrong in width.
                w = np.maximum(win, 0.0)
                tot = w.sum()
                if tot <= 0:
                    continue
                dr = (w * yy).sum() / tot
                dc = (w * xx).sum() / tot
                # Voronoi ownership: only the window that owns its centroid.
                if abs(dr) > step / 2 or abs(dc) > step / 2:
                    continue
                row, col = r0 + dr, c0 + dc
                tight = _width_at_aperture(
                    frames[fi], bgs[fi], row, col, 3.0, step, _WIDTH_TIGHT_APERTURE
                )
                wide = _width_at_aperture(
                    frames[fi], bgs[fi], row, col, 3.0, step, _WIDTH_WIDE_APERTURE
                )
                if not (np.isfinite(tight) and np.isfinite(wide)) or tight <= 0:
                    continue
                if abs(wide - tight) / tight > _WIDTH_CONSISTENCY_TOL:
                    continue
                sigma_w = float(wide)
                # A width at the window scale was never bracketed.
                if sigma_w < 0.5 or sigma_w > step / 2.5:
                    continue
                d2 = (yy - dr) ** 2 + (xx - dc) ** 2
                ring = (d2 >= (step - 3.0) ** 2) & (d2 <= step**2)
                bg_corr = min(
                    float(np.median(win[ring])) if ring.sum() >= 8 else 0.0, 0.0
                )
                win_c = win - bg_corr
                disk = d2 <= step**2
                flux_d = float(win_c[disk].sum())
                if flux_d <= 0:
                    continue
                tt = _lookup_two_theta(two_theta, fi, row, col, H, W)
                candidates.append(
                    (sigma_w, tt, flux, d2[disk].ravel(), win_c[disk].ravel(), flux_d)
                )

    if len(candidates) < min_windows:
        return None
    cand_widths = np.asarray([c[0] for c in candidates], dtype=float)
    cand_tts = (
        np.asarray([c[1] for c in candidates], dtype=float)
        if two_theta is not None
        else None
    )
    _, keep, _ = _filter_widths_by_prior(cand_widths, cand_tts)
    n_used = 0
    for idx, is_kept in enumerate(keep):
        if not is_kept:
            continue
        sigma_w, _, flux, d2_disk, win_disk, flux_d = candidates[idx]
        u = np.sqrt(d2_disk) / sigma_w
        v = win_disk / flux_d
        which = np.clip((u / du).astype(int), 0, centres.size - 1)
        inside = u < u_max
        # Flux-weighted mean of the unit-flux pixel values per u bin:
        # acc/wgt is then the average profile, bright peaks weighted by
        # their better statistics.
        np.add.at(acc, which[inside], flux * v[inside])
        counts = np.bincount(which[inside], minlength=centres.size).astype(float)
        wgt += flux * counts
        n_used += 1
    if n_used < min_windows:
        return None
    ok = wgt > 0
    f = np.zeros(centres.size)
    f[ok] = acc[ok] / wgt[ok]
    # Per-bin averages of per-pixel values: convert back to a profile by
    # normalising the centre to 1; the kernel builder only needs the shape.
    if f[0] <= 0:
        return None
    f = np.maximum(f / f[0], 0.0)
    return centres[ok], f[ok]


def measure_whitened_profile(
    images,
    rate_maps,
    valid,
    frames,
    rs,
    cs,
    var_us,
    var_vs,
    cov_uvs,
    min_peaks=50,
    u_max=4.0,
    du=0.1,
):
    """The reflection family's radial profile, measured in model sigmas.

    The finder's ``_measure_radial_profile`` census, upgraded with what
    the integrator knows and the finder had to estimate: centroids come
    from the predicted (snapped) positions instead of window moments,
    the u axis is the MAHALANOBIS radius of each peak's projected
    covariance instead of an isotropic moment width -- so anisotropy is
    divided out exactly, and the rank-1-after-scale result the finder
    measured on cg4d-garnet (mean profile = 95.5% of family energy)
    holds a fortiori: on cg4d-t4-lysozyme the whitened mean peak is
    round to the eye at 3 sigma -- and the background under each window
    is the footprint-masked rate map instead of a local ring, so the
    profile's own tails cannot subtract themselves.

    Same guards as the finder: a window flux floor at 8 sigma of the
    background noise, one vote per peak, windows overlapping another
    footprint or a masked pixel are skipped, flux-weighted averaging,
    and ``None`` (caller stays on the Gaussian) when fewer than
    ``min_peaks`` qualify.  No functional form is fitted anywhere.

    Returns:
        (u, f) with f[0] = 1, or None.
    """
    edges = np.arange(0.0, u_max + du, du)
    centres = 0.5 * (edges[1:] + edges[:-1])
    acc = np.zeros(centres.size)
    wgt = np.zeros(centres.size)
    n_used = 0

    sig_maj = np.sqrt(
        0.5 * (var_us + var_vs)
        + np.sqrt(np.maximum(0.25 * (var_us - var_vs) ** 2 + cov_uvs**2, 0.0))
    )
    B, H, W = images.shape
    by_frame = [np.where(frames == f)[0] for f in range(B)]

    for f in range(B):
        idx = by_frame[f]
        if len(idx) == 0:
            continue
        img = np.asarray(images[f], dtype=np.float64)
        bg = np.asarray(rate_maps[f], dtype=np.float64)
        v = np.asarray(valid[f], dtype=np.float64)
        for i in idx:
            half = int(np.ceil(u_max * sig_maj[i])) + 1
            r0, c0 = round(rs[i]), round(cs[i])
            if not (half <= r0 < H - half and half <= c0 < W - half):
                continue
            # one vote per peak: skip windows contaminated by a
            # neighbouring footprint (3 sigma_major each)
            others = idx[idx != i]
            if len(others):
                d = np.hypot(rs[others] - rs[i], cs[others] - cs[i])
                if np.any(d < 3.0 * (sig_maj[others] + sig_maj[i])):
                    continue
            win_v = v[r0 - half : r0 + half + 1, c0 - half : c0 + half + 1]
            if win_v.min() < 1.0:
                continue
            win = (
                img[r0 - half : r0 + half + 1, c0 - half : c0 + half + 1]
                - bg[r0 - half : r0 + half + 1, c0 - half : c0 + half + 1]
            )
            dv, dr = np.mgrid[-half : half + 1, -half : half + 1].astype(float)
            dv += r0 - rs[i]
            dr += c0 - cs[i]
            det = max(var_us[i] * var_vs[i] - cov_uvs[i] ** 2, 1e-6)
            m = np.sqrt(
                (var_vs[i] * dr**2 - 2.0 * cov_uvs[i] * dr * dv + var_us[i] * dv**2)
                / det
            )
            disk = m < u_max
            flux = float(win[disk].sum())
            bg_hi = float(np.median(bg[r0, c0 - half : c0 + half + 1]))
            area = int(disk.sum())
            if flux < max(50.0, 8.0 * np.sqrt(max(bg_hi, 0.05) * area)):
                continue
            u = m[disk].ravel()
            val = win[disk].ravel() / flux
            # Linear bin sharing: each pixel splits its vote between the
            # two nearest bin centres.  A hard assignment aliases against
            # the pixel lattice -- peaks near-integer positions sample m
            # at discrete radii, and when several peaks are in phase the
            # binned profile turns into a comb (measured on a synthetic
            # integer grid: non-monotone by 40%).  Sharing is the same
            # census with a triangular kernel one bin wide, not a fit.
            t = np.clip(u / du - 0.5, 0.0, centres.size - 1.0)
            lo = np.floor(t).astype(int)
            hi = np.minimum(lo + 1, centres.size - 1)
            w_hi = t - lo
            np.add.at(acc, lo, flux * val * (1.0 - w_hi))
            np.add.at(acc, hi, flux * val * w_hi)
            np.add.at(wgt, lo, flux * (1.0 - w_hi))
            np.add.at(wgt, hi, flux * w_hi)
            n_used += 1

    if n_used < min_peaks:
        return None
    ok = wgt > 0
    f_prof = np.zeros(centres.size)
    f_prof[ok] = acc[ok] / wgt[ok]
    if f_prof[0] <= 0:
        return None
    f_prof = np.maximum(f_prof / f_prof[0], 0.0)
    return centres, f_prof
