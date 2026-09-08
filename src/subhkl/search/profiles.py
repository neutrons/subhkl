"""Shared empirical reflection-family profiles for finding, integration and solving.

The measurement routines retain their bright-window qualification rules.
They estimate a shape prior; they do not select pixels for the count solver.
"""

import numpy as np
from dataclasses import dataclass


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


def measure_radial_profile(
    images, bg_map, bg_hi, max_sigma, min_windows=8, u_max=4.0, du=0.1, valid=None
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
    n_used = 0
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
                d2 = (yy - dr) ** 2 + (xx - dc) ** 2
                disk = d2 <= step**2
                flux_d = float(win[disk].sum())
                if flux_d <= 0:
                    continue
                m2 = float((win[disk] * d2[disk]).sum()) / flux_d
                # Invert the truncation: inside radius R a Gaussian returns
                # m2 = sigma^2 * 2(1 - (1+x)e^-x)/(1 - e^-x), x = R^2/2sigma^2.
                # A damped fixed point converges in a few steps and needs no
                # bracketing solver.
                sigma_w = np.sqrt(max(m2, 1e-9) / 2.0)
                for _ in range(8):
                    x = step**2 / (2.0 * sigma_w**2)
                    ex = np.exp(-min(x, 60.0))
                    ratio = 2.0 * (1.0 - (1.0 + x) * ex) / max(1.0 - ex, 1e-9)
                    new_sigma = np.sqrt(max(m2, 1e-9) / max(ratio, 1e-9))
                    if abs(new_sigma - sigma_w) < 0.005:
                        sigma_w = new_sigma
                        break
                    sigma_w = 0.5 * (sigma_w + new_sigma)
                # A width at the window scale was never bracketed.
                if sigma_w < 0.5 or sigma_w > step / 2.5:
                    continue
                u = np.sqrt(d2[disk]).ravel() / sigma_w
                v = win[disk].ravel() / flux_d
                which = np.clip((u / du).astype(int), 0, centres.size - 1)
                inside = u < u_max
                # Flux-weighted mean of the unit-flux pixel values per u bin:
                # acc/wgt is then the average profile, bright peaks weighted by
                # their better statistics.
                np.add.at(acc, which[inside], flux * v[inside])
                counts = np.bincount(which[inside], minlength=centres.size).astype(
                    float
                )
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
            r0, c0 = int(round(rs[i])), int(round(cs[i]))
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
