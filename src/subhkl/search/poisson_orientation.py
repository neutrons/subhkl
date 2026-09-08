"""Experimental count likelihood shared by indexing and geometry refinement.

No count threshold, peak list, maximum map, or clipped background residual is
used. Each orientation owns a group of nonnegative reflection intensities.
The convex inner problem has a Poisson likelihood and a sum of group norms;
the outer problem changes orientations and one shared six-parameter geometry.

This is a finite-candidate/local-refinement experiment, not a global SO(3)
search. Flat panels, a known cell/band, and an isotropic Gaussian PSF are the
current forward-model assumptions. Binning is a sum of independent counts.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, minimize
from scipy.special import ndtr, xlogy
from scipy.spatial.transform import Rotation

from subhkl.calibrate import geometry_detectors, reciprocal_lattice


@dataclass
class CountData:
    counts: np.ndarray
    background: np.ndarray
    bank_index: np.ndarray
    pixel_indices: dict
    shapes: dict
    bin_px: int

    @classmethod
    def from_images(cls, images, bank_ids, background, bin_px=4, masks=None):
        """Keep all valid bins, including zeros; masks describe instrument validity.

        ``background`` is a positive expected-count image per panel, before
        binning. Its spatial shape is fixed; its panel scales are fitted.
        """
        if bin_px < 1 or int(bin_px) != bin_px:
            raise ValueError("bin_px must be a positive integer")
        if len(images) != len(bank_ids) or len(set(bank_ids)) != len(bank_ids):
            raise ValueError("one image per distinct bank is required")
        ys, bs, banks, indices, shapes = [], [], [], {}, {}
        offset = 0
        for i, (bank, im) in enumerate(zip(bank_ids, images)):
            bank = int(bank)
            im = np.asarray(im, float)
            bg = np.broadcast_to(np.asarray(background[i], float), im.shape)
            valid = (
                np.ones(im.shape, bool) if masks is None else np.asarray(masks[i], bool)
            )
            if im.ndim != 2 or valid.shape != im.shape:
                raise ValueError("images and masks must be matching 2D arrays")
            if not np.isfinite(im[valid]).all() or np.any(im[valid] < 0):
                raise ValueError("counts must be finite and nonnegative")
            if not np.isfinite(bg[valid]).all() or np.any(bg[valid] <= 0):
                raise ValueError("background must be finite and positive")
            h, w = im.shape
            if h % bin_px or w % bin_px:
                raise ValueError("image dimensions must be divisible by bin_px")
            shape = (h // bin_px, w // bin_px)

            def blocks(a):
                return a.reshape(shape[0], bin_px, shape[1], bin_px)

            keep = blocks(valid).all(axis=(1, 3))
            y = blocks(np.where(valid, im, 0)).sum(axis=(1, 3))[keep]
            b = blocks(np.where(valid, bg, 0)).sum(axis=(1, 3))[keep]
            index = np.full(shape, -1, int)
            index[keep] = offset + np.arange(len(y))
            offset += len(y)
            ys.append(y)
            bs.append(b)
            banks.append(np.full(len(y), i, int))
            indices[bank], shapes[bank] = index, shape
        if offset == 0:
            raise ValueError("no valid pixels")
        return cls(
            np.concatenate(ys),
            np.concatenate(bs),
            np.concatenate(banks),
            indices,
            shapes,
            bin_px,
        )


def geometry_vector(g):
    """Fix assembly roll about +z: (scale, tilt_x, tilt_y, tx, ty, tz)."""
    g = np.asarray(g, float)
    if g.shape != (6,):
        raise ValueError("geometry must contain six parameters")
    return np.r_[g[:3], 0.0, g[3:]]


class OrientationModel:
    def __init__(
        self,
        data,
        instrument,
        cell,
        space_group,
        wavelength,
        d_min,
        sigma_px=4.0,
        profile=None,
        detectors=None,
        sample_origin=None,
    ):
        if sigma_px <= 0 or not 0 < wavelength[0] < wavelength[1]:
            raise ValueError("positive PSF width and ordered positive band required")
        self.data, self.instrument = data, instrument
        self.wavelength = wavelength
        self.d_min = float(d_min)
        self.sigma_px = float(sigma_px)
        self.profile, self.base_detectors = profile, detectors
        self.sample_origin = (
            np.zeros(3) if sample_origin is None else np.asarray(sample_origin, float)
        )
        self.B, self.G = reciprocal_lattice(cell, space_group, d_min)
        self.n_hkl = len(self.G)
        self.banks = list(data.pixel_indices)

    def design(self, orientations, g):
        """Sparse, pixel-integrated Gaussian reflection footprints (unit flux).

        Support truncation at five sigma is geometric, never count-dependent.
        Invisible reflections retain zero columns, preserving intensity IDs
        as the geometry and orientations change. Masked flux is not renormalized.
        """
        dets = self.projection_detectors(g)
        rows, cols, values = [], [], []
        sig = self.sigma_px / self.data.bin_px
        support = 5.0 if self.profile is None else self.profile.u[-1]
        for j, U in enumerate(orientations):
            q = self.G @ np.asarray(U).T
            lam = -2 * q[:, 2] / np.sum(q * q, axis=1)
            hk = np.flatnonzero(
                (lam >= self.wavelength[0]) & (lam <= self.wavelength[1])
            )
            kf = np.array([0.0, 0.0, 1.0]) + lam[hk, None] * q[hk]
            for bank, det in dets.items():
                if det.config["panel"] != "flat":
                    raise NotImplementedError(
                        "experimental forward model supports flat panels"
                    )
                _, pr, pc = det.reflections_mask(*kf.T)
                normal = np.cross(det.uhat, det.vhat)
                forward = (kf @ normal) * (det.center @ normal) > 0
                # Original pixel i covers [i-.5,i+.5]; binned center convention.
                rr = (np.asarray(pr) + 0.5) / self.data.bin_px
                cc = (np.asarray(pc) + 0.5) / self.data.bin_px
                h, w = self.data.shapes[bank]
                keep = forward & np.isfinite(rr) & np.isfinite(cc)
                keep &= (
                    (rr > -support * sig)
                    & (rr < h + support * sig)
                    & (cc > -support * sig)
                    & (cc < w + support * sig)
                )
                for k in np.flatnonzero(keep):
                    r = np.arange(
                        max(0, int(np.floor(rr[k] - support * sig))),
                        min(h, int(np.ceil(rr[k] + support * sig))),
                    )
                    c = np.arange(
                        max(0, int(np.floor(cc[k] - support * sig))),
                        min(w, int(np.ceil(cc[k] + support * sig))),
                    )
                    vr = ndtr((r + 1 - rr[k]) / sig) - ndtr((r - rr[k]) / sig)
                    vc = ndtr((c + 1 - cc[k]) / sig) - ndtr((c - cc[k]) / sig)
                    ix = self.data.pixel_indices[bank][r[:, None], c[None, :]].ravel()
                    v = (
                        np.outer(vr, vc)
                        if self.profile is None
                        else self.profile.integrate_bins(r, c, rr[k], cc[k], sig)
                    ).ravel()
                    valid = ix >= 0
                    rows.extend(ix[valid])
                    cols.extend(np.full(valid.sum(), j * self.n_hkl + hk[k]))
                    values.extend(v[valid])
        return sparse.csc_matrix(
            (values, (rows, cols)),
            shape=(len(self.data.counts), len(orientations) * self.n_hkl),
        )

    def detectors(self, g):
        g7 = np.asarray(g, float) if len(g) == 7 else geometry_vector(g)
        if self.base_detectors is None:
            return geometry_detectors(self.instrument, g7, self.banks)
        from subhkl.instrument.detector import Detector

        rotation = Rotation.from_rotvec(g7[1:4]).as_matrix()
        out = {}
        for bank, det in self.base_detectors.items():
            cfg = dict(det.config)
            cfg["center"] = rotation @ ((1 + g7[0]) * det.center) + g7[4:]
            cfg["uhat"], cfg["vhat"] = rotation @ det.uhat, rotation @ det.vhat
            out[bank] = Detector(cfg)
        return out

    def projection_detectors(self, g):
        from subhkl.instrument.detector import Detector

        return {
            bank: Detector(dict(det.config, center=det.center - self.sample_origin))
            for bank, det in self.detectors(g).items()
        }


@dataclass
class SparseFit:
    intensities: np.ndarray
    group_norms: np.ndarray
    background_scales: np.ndarray
    mean: np.ndarray
    objective: float
    iterations: int
    kkt: float
    converged: bool
    weights: np.ndarray
    kkt_tolerance: float


def orientation_weights(A, data, n_groups):
    """One construction of the geometry-frozen group weights for both stages."""
    fisher = np.sqrt(np.asarray(A.power(2).T @ (1 / data.background)).ravel())
    return np.sqrt((fisher.reshape(n_groups, -1) > 1e-12).sum(axis=1))


def fit_intensities(
    A, data, n_groups, penalty=3.0, weights=None, max_iter=3000, tol=1e-5, rtol=0.0
):
    """Nonnegative Poisson group lasso, including unpenalized panel backgrounds.

    Reflection columns are scaled to unit Fisher norm under the supplied
    background. Coefficients therefore measure approximate SNR; physical
    fluxes are returned. Group weights default to sqrt(visible reflections).
    They should be fixed across an outer refinement, not adapted to its fit.
    An accelerated proximal-gradient solve with monotone restarts reports its
    stationarity residual at the accepted iterate. The stopping threshold is
    tol + rtol * max(1, largest group penalty), in Fisher-normalized units.
    Acceleration matters near disappearing groups.
    """
    y, bg = data.counts, data.background
    if penalty < 0 or n_groups < 1 or A.shape[1] % n_groups:
        raise ValueError("invalid penalty or orientation groups")
    size = A.shape[1] // n_groups
    fisher = np.sqrt(np.asarray(A.power(2).T @ (1 / bg)).ravel())
    scale = np.where(fisher > 1e-12, fisher, 1.0)
    F = A @ sparse.diags(1 / scale)
    nb = len(data.pixel_indices)
    bg_norm = np.sqrt(np.bincount(data.bank_index, weights=bg, minlength=nb))
    if np.any(bg_norm == 0):
        raise ValueError("each panel needs valid pixels")
    B = sparse.csc_matrix(
        (bg / bg_norm[data.bank_index], (np.arange(len(y)), data.bank_index)),
        shape=(len(y), nb),
    )
    M = sparse.hstack([F, B], format="csc")
    full_M = M
    # Pixels outside every reflection footprint constrain only their panel's
    # background scale. Their totals are exact Poisson sufficient statistics.
    live = np.asarray(A.getnnz(axis=1)).ravel() > 0
    empty_y = np.bincount(data.bank_index[~live], weights=y[~live], minlength=nb)
    empty_bg = np.bincount(data.bank_index[~live], weights=bg[~live], minlength=nb)
    panels = np.flatnonzero(empty_bg > 0)
    collapsed = sparse.csc_matrix(
        (
            empty_bg[panels] / bg_norm[panels],
            (np.arange(len(panels)), A.shape[1] + panels),
        ),
        shape=(len(panels), M.shape[1]),
    )
    constant = float(
        np.sum(xlogy(y[~live], y[~live] / bg[~live]))
        - np.sum(xlogy(empty_y[panels], empty_y[panels] / empty_bg[panels]))
    )
    M = sparse.vstack([M[live], collapsed], format="csc")
    y = np.r_[y[live], empty_y[panels]]
    if weights is None:
        weights = orientation_weights(A, data, n_groups)
    weights = np.asarray(weights, float)
    if weights.shape != (n_groups,) or np.any(weights < 0):
        raise ValueError("one nonnegative weight per orientation required")
    if tol <= 0 or rtol < 0:
        raise ValueError(
            "positive absolute and nonnegative relative tolerances required"
        )
    tolerance = tol + rtol * max(1.0, penalty * np.max(weights))
    x = np.r_[np.zeros(A.shape[1]), bg_norm]

    def smooth(z):
        mu = np.asarray(M @ z)
        if np.any(mu <= 0):
            return np.inf, mu
        # Deviance / 2, including constants, improves numerical conditioning.
        return float(np.sum(mu - y + xlogy(y, y / mu))), mu

    def prox(z, step):
        z = np.maximum(z, 0.0)
        groups = z[:-nb].reshape(n_groups, size)
        norms = np.linalg.norm(groups, axis=1)
        factors = np.maximum(
            0.0, 1 - step * penalty * weights / np.maximum(norms, 1e-300)
        )
        groups *= factors[:, None]
        z[-nb:] = np.maximum(z[-nb:], 1e-10)
        return z

    def regularizer(z):
        return penalty * float(
            weights @ np.linalg.norm(z[:-nb].reshape(n_groups, size), axis=1)
        )

    step, kkt = 1.0, np.inf
    f, mu = smooth(x)
    extrapolated, momentum = x.copy(), 1.0
    for iteration in range(1, max_iter + 1):
        # Extrapolation can make a vanishing reflection negative. Near a
        # low-background pixel that cancellation makes Poisson curvature
        # arbitrarily large. Project the extrapolation to preserve feasibility
        # without discarding all momentum whenever one coefficient hits zero.
        if np.any(extrapolated[:-nb] < 0) or np.any(extrapolated[-nb:] <= 0):
            extrapolated = np.maximum(extrapolated, 0.0)
            extrapolated[-nb:] = np.maximum(extrapolated[-nb:], 1e-10)
        fv, mv = smooth(extrapolated)
        if not np.isfinite(fv):
            extrapolated, momentum = x.copy(), 1.0
            fv, mv = f, mu
        grad = np.asarray(M.T @ (1 - y / mv))
        step = min(step * 1.2, 10.0)
        for _ in range(80):
            trial = prox(extrapolated - step * grad, step)
            delta = trial - extrapolated
            ft, mt = smooth(trial)
            if ft <= fv + grad @ delta + delta @ delta / (2 * step) + 1e-12 * max(
                1, abs(fv)
            ):
                break
            step *= 0.5
        else:
            # Preserve the last feasible fit for the caller's diagnostics.
            break
        if ft + regularizer(trial) > f + regularizer(x) + 1e-12 * max(1, abs(f)):
            extrapolated, momentum = x.copy(), 1.0
            continue
        # Check stationarity at the accepted point with a unit proximal step,
        # independent of backtracking and acceleration. Tiny line-search
        # steps must not round the reported residual down to zero.
        trial_grad = np.asarray(M.T @ (1 - y / mt))
        kkt = float(np.max(np.abs(trial - prox(trial - trial_grad, 1.0))))
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = trial + ((momentum - 1) / next_momentum) * (trial - x)
        momentum = next_momentum
        x, f, mu = trial, ft, mt
        if kkt < tolerance:
            break
    if kkt >= tolerance and max_iter >= 100:
        # Once proximal iterations identify support, curvature-aware polishing
        # avoids a single bright reflection setting the step for every panel.
        # Inactive groups remain fixed; the full proximal KKT check below
        # still rejects a solution that should activate another group.
        active_columns = np.repeat(
            np.linalg.norm(x[:-nb].reshape(n_groups, size), axis=1) > 0, size
        )

        def value_gradient(z):
            value, mean = smooth(z)
            groups = z[:-nb].reshape(n_groups, size)
            norms = np.linalg.norm(groups, axis=1)
            gradient = np.asarray(M.T @ (1 - y / mean))
            gradient[:-nb] += (
                penalty * weights[:, None] * groups / np.maximum(norms[:, None], 1e-300)
            ).ravel()
            return value + regularizer(z), gradient

        polished = minimize(
            value_gradient,
            x,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, None) if active else (0, 0) for active in active_columns]
            + [(1e-10, None)] * nb,
            options={
                "maxiter": max_iter,
                "ftol": 0.0,
                "gtol": tol * 0.1,
                "maxls": 50,
                "maxcor": 30,
            },
        )
        if polished.fun <= f + regularizer(x):
            x = polished.x
            f, mu = smooth(x)
            grad = np.asarray(M.T @ (1 - y / mu))
            kkt = float(np.max(np.abs(x - prox(x - grad, 1.0))))
            iteration += polished.nit
        # A few reduced Newton steps remove the objective-roundoff floor of
        # L-BFGS on bright data. Only free coefficients enter this small dense
        # system; large problems retain the checked proximal/L-BFGS result.
        for _ in range(12):
            value, gradient = value_gradient(x)
            free = (x > 1e-9) | (gradient < 0)
            free[:-nb] &= active_columns
            indices = np.flatnonzero(free)
            if kkt < tolerance or len(indices) > 512:
                break
            reduced = M[:, indices]
            hessian = (reduced.T @ reduced.multiply((y / mu**2)[:, None])).toarray()
            for group in range(n_groups):
                start, end = group * size, (group + 1) * size
                pos = np.flatnonzero((indices >= start) & (indices < end))
                norm = np.linalg.norm(x[start:end])
                if len(pos) and norm > 0:
                    z = x[indices[pos]] / norm
                    hessian[np.ix_(pos, pos)] += (
                        penalty
                        * weights[group]
                        / norm
                        * (np.eye(len(pos)) - np.outer(z, z))
                    )
            direction = np.linalg.lstsq(hessian, -gradient[indices], rcond=1e-12)[0]
            descent = gradient[indices] @ direction
            if descent >= 0:
                break
            negative = direction < 0
            length = (
                min(1.0, float(np.min(-x[indices[negative]] / direction[negative])))
                if np.any(negative)
                else 1.0
            )
            for _ in range(30):
                trial = x.copy()
                trial[indices] = np.maximum(x[indices] + length * direction, 0)
                trial[-nb:] = np.maximum(trial[-nb:], 1e-10)
                ft, mt = smooth(trial)
                if ft + regularizer(
                    trial
                ) <= value + 1e-4 * length * descent + 1e-12 * max(1, abs(value)):
                    x, f, mu = trial, ft, mt
                    break
                length *= 0.5
            else:
                break
            grad = np.asarray(M.T @ (1 - y / mu))
            kkt = float(np.max(np.abs(x - prox(x - grad, 1.0))))
            iteration += 1
    norms = np.linalg.norm(x[:-nb].reshape(n_groups, size), axis=1)
    return SparseFit(
        (x[:-nb] / scale).reshape(n_groups, size),
        norms,
        x[-nb:] / bg_norm,
        np.asarray(full_M @ x),
        f + regularizer(x) + constant,
        iteration,
        kkt,
        kkt < tolerance,
        weights.copy(),
        tolerance,
    )


def refine(
    model,
    orientations,
    penalty=3.0,
    g0=None,
    max_evals=250,
    refine_orientations=True,
    refine_geometry=True,
    log=None,
    inner_max_iter=2000,
):
    """Profile the same objective over a local geometry/orientation search.

    Finite differences include reprojection and intensity refitting. This
    intentionally simple reference implementation prioritizes an inspectable
    objective over speed; it is not a replacement for a global indexer.
    """
    ng = getattr(model, "geometry_size", 6)
    g0 = np.zeros(ng) if g0 is None else np.asarray(g0, float)
    orientations = np.asarray(orientations, float)
    n = len(orientations)
    A0 = model.design(orientations, g0)
    weights = orientation_weights(A0, model.data, n)
    units = np.array([0.03, 0.02, 0.02, 0.005, 0.005, 0.005])
    limits = np.array([0.08, 0.05, 0.05, 0.02, 0.02, 0.02])
    if ng == 7:
        units = np.insert(units, 3, 0.02)
        limits = np.insert(limits, 3, 0.05)
    extra = 3 * n if refine_orientations else 0
    x0 = np.r_[g0 / units, np.zeros(extra)]
    bounds = (
        list(
            zip(
                -limits / units,
                limits / units,
            )
        )
        + [(-3.0, 3.0)] * extra
    )
    if not refine_geometry:
        bounds[:ng] = [(v, v) for v in x0[:ng]]
    best = {"value": np.inf}
    evaluations = 0

    class InnerNotConverged(RuntimeError):
        pass

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        g = x[:ng] * units
        U = (
            orientations
            if not extra
            else Rotation.from_rotvec(x[ng:].reshape(n, 3) * 0.02).as_matrix()
            @ orientations
        )
        design = model.design(U, g)
        fit = fit_intensities(
            design,
            model.data,
            n,
            penalty,
            weights,
            max_iter=inner_max_iter,
            tol=2e-5,
            rtol=1e-6,
        )
        if not fit.converged and inner_max_iter < 2000:
            # A short first pass is useful for large setting stacks. Retry
            # difficult evaluations with the standard budget before failing.
            fit = fit_intensities(
                design,
                model.data,
                n,
                penalty,
                weights,
                max_iter=2000,
                tol=2e-5,
                rtol=1e-6,
            )
        if fit.objective < best["value"]:
            best.update(value=fit.objective, g=g.copy(), orientations=U.copy(), fit=fit)
            if log:
                log(f"objective {fit.objective:.5f}; g={g}; groups={fit.group_norms}")
        if not fit.converged:
            raise InnerNotConverged(f"inner solve did not converge: KKT={fit.kkt:g}")
        return fit.objective

    try:
        result = minimize(
            evaluate,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxfun": max_evals,
                "maxiter": max_evals,
                "ftol": 1e-9,
                "gtol": 1e-4,
                "eps": 1e-4,
            },
        )
    except InnerNotConverged as error:
        result = OptimizeResult(success=False, message=str(error), nfev=evaluations)
    best["optimizer"] = result
    return best
