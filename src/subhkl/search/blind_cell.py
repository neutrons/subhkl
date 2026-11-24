"""Unit cell and orientation from raw time-of-flight events.

Every event of a time-of-flight Laue run has a full scattering vector
q = (k_f - k_i) / lambda.  Histogram all events in the sample frame, take
the 3D Fourier transform, and the direct lattice appears: with the
reciprocal-space density rho(q) = sum_h w_h delta(q - h), the transform
G(x) = sum_h w_h exp(2 pi i h.x) adds coherently exactly when x is a
direct-lattice vector (Poisson summation).  This is the lattice ladder's
objective with |q| kept instead of integrated out, so it yields absolute
cell lengths as well as directions.  No reflection is located and no cell
is assumed.

``find_cell`` runs the whole step; ``write_seed`` stores its result as a
solve-style file in space group P 1 that seeds ``solve --bootstrap``.
"""

import contextlib
from dataclasses import dataclass, field

import gemmi
import h5py
import numpy as np

from subhkl.config import beamlines, reduction_settings
from subhkl.instrument.detector import Detector
from subhkl.instrument.goniometer import Goniometer
from subhkl.io.events import ANGSTROM_M_PER_US, EventStore

EVENT_PAD = 1 << 21


def pixel_positions(instrument, bank, shape):
    """[m] lab position of every pixel of a bank, flat row-major, nominal geometry."""
    det = Detector(beamlines[instrument][str(int(bank))])
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    return det.pixel_to_lab(rows.ravel(), cols.ravel()).astype(np.float32)


def _accumulate_impl(grid, tof, pix, pos, rot, l1, q_max, n):
    """Add one image's events to the reciprocal-space histogram (sample frame)."""
    import jax.numpy as jnp

    xyz = pos[pix]
    l2 = jnp.linalg.norm(xyz, axis=1)
    lam = ANGSTROM_M_PER_US * tof / (l1 + l2)
    k_hat = xyz / l2[:, None]
    q_lab = (k_hat - jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)) / jnp.maximum(
        lam, 1e-6
    )[:, None]
    q = q_lab @ rot  # R^T q_lab: lab -> sample
    idx = jnp.floor((q + q_max) * (n / (2.0 * q_max))).astype(jnp.int32)
    ok = (tof > 0) & jnp.all((idx >= 0) & (idx < n), axis=1)
    flat = jnp.where(ok, (idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2], n * n * n)
    # Out-of-range events get an out-of-bounds index and are dropped: sending
    # them all to one spare slot makes the scatter collide on that slot and
    # run orders of magnitude slower.
    return grid.at[flat].add(1.0, mode="drop")


def reciprocal_histogram(merged, instrument, d_min, n, frames=None, rotations=None):
    """Event density on an n^3 grid over |q_i| <= 1/d_min, sample frame.

    Args:
        merged: images file with pixel-sorted events and goniometer angles
        instrument: instrument name (nominal detector geometry and L1)
        d_min: [Angstrom] the grid spans |q_i| <= 1 / d_min
        n: [-] grid points per axis
        frames: optional image indices to use (for example one run)
        rotations: optional (n_images, 3, 3) goniometer rotations to use
            instead of the ones in the file
    Returns:
        (grid [counts] (n, n, n), q_max [1/Angstrom], events read)
    """
    import jax
    import jax.numpy as jnp

    l1 = float(reduction_settings[instrument]["L1"])
    q_max = 1.0 / d_min
    acc = jax.jit(_accumulate_impl, static_argnames=["n"])
    with h5py.File(merged, "r") as f:
        images = f["images"][()]
        bank_ids = f["bank_ids"][()]
        rot = Goniometer.from_h5_file(f).rotation
    if rot.ndim == 2:
        rot = np.broadcast_to(rot, (len(images), 3, 3))
    if rotations is not None:
        rot = np.asarray(rotations)
    pos_cache = {}
    grid = jnp.zeros(n * n * n, dtype=jnp.float32)
    n_events = 0
    with EventStore(merged) as store:
        for i in range(len(images)) if frames is None else frames:
            bank = int(bank_ids[i])
            if bank not in pos_cache:
                pos_cache[bank] = jnp.array(
                    pixel_positions(instrument, bank, images[i].shape)
                )
            tof, off = store.image_events(i, images[i])
            pix = np.repeat(np.arange(images[i].size, dtype=np.int32), np.diff(off))
            n_pad = max(EVENT_PAD, -(-len(tof) // EVENT_PAD) * EVENT_PAD)
            tof_p = np.zeros(n_pad, dtype=np.float32)
            pix_p = np.zeros(n_pad, dtype=np.int32)
            tof_p[: len(tof)] = tof
            pix_p[: len(tof)] = pix
            grid = acc(
                grid,
                jnp.array(tof_p),
                jnp.array(pix_p),
                pos_cache[bank],
                jnp.array(rot[i], dtype=jnp.float32),
                jnp.float32(l1),
                jnp.float32(q_max),
                n=n,
            )
            n_events += len(tof)
    return grid.reshape(n, n, n), q_max, n_events


LAUE_PIXEL_PAD = 1 << 13
LAUE_SAMPLES = 64


def _streak_impl(grid, weight, pos, rot, lam, shape, q_max, n):
    """Add one image's significant pixels, each spread along its radial
    streak over the wavelength samples, to the reciprocal-space histogram."""
    import jax.numpy as jnp

    l2 = jnp.linalg.norm(pos, axis=1)
    q_dir = pos / l2[:, None] - jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    q_dir = q_dir @ rot  # lab -> sample; |q| lambda = 2 sin(theta) is its length
    q = q_dir[:, None, :] / lam[None, :, None]
    w = weight[:, None] * shape[None, :]
    both = jnp.concatenate([q, -q], axis=0).reshape(-1, 3)  # Friedel mate too
    w = jnp.concatenate([w, w], axis=0).ravel()
    idx = jnp.floor((both + q_max) * (n / (2.0 * q_max))).astype(jnp.int32)
    ok = (w > 0) & jnp.all((idx >= 0) & (idx < n), axis=1)
    flat = jnp.where(ok, (idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2], n * n * n)
    return grid.at[flat].add(w, mode="drop")


def laue_histogram(
    merged,
    instrument,
    wavelength,
    d_min,
    n,
    frames=None,
    rotations=None,
    spectrum=None,
    min_z=4.0,
    weighting="maxima",
    binning=None,
):
    """Reciprocal-space density for data without wavelengths.

    A count on the detector fixes the direction of its scattering vector but
    not its length, which lies between 2 sin(theta) / lam_max and
    2 sin(theta) / lam_min.  Each significant pixel is therefore spread along
    that radial streak, weighted by the spectrum.  The Fourier transform of
    the result is the same direct-space map as for time-of-flight events
    (``reciprocal_histogram``): Poisson summation with the unknown wavelength
    integrated over the band.

    A pixel counts when it is the local maximum of a patch standing
    ``min_z`` standard errors above a smooth local background.  Maxima that
    recur at the same place in three or more runs are left out: a reflection moves when
    the crystal turns, a feature fixed in the laboratory does not.

    Args:
        merged: images file with goniometer angles (no events needed)
        instrument: instrument name (pixel positions)
        wavelength: [Angstrom] (lam_min, lam_max)
        d_min: [Angstrom] the grid spans |q_i| <= 1 / d_min
        n: [-] grid points per axis
        frames: optional image indices to use
        rotations: optional (n_images, 3, 3) goniometer rotations
        spectrum: optional (wavelengths [Angstrom], relative weights) of the
            scattered intensity; default a smooth bump over the band, which
            avoids the ringing that sharp band edges cause in the map
        min_z: [-] significance of a pixel (``weighting="maxima"``)
        weighting: "deviance" uses every count bin with the weight of the
            orientation search (``blind_band.count_search_weights``): the
            images are binned so that a reflection fills about one bin, each
            bin's Poisson deviance D against the local background is
            computed, and the weight grows as sqrt(D) with background
            fluctuations suppressed.  That is a significance, not an
            intensity, so strong reflections do not outweigh the lattice.
            "maxima" uses one entry per significant local maximum with the
            excess counts, capped, as weight.
            On IMAGINE-X only "maxima" has found the lattice so far (there
            are about six times more significant bins than reflections);
            on MANDI both do.
        binning: [Pixel] bin size for "deviance"; default 8 for panels of
            512 pixels and more, else 4
    Returns:
        (grid (n, n, n), q_max [1/Angstrom], number of bins or pixels used)
    """
    import jax
    import jax.numpy as jnp
    from scipy.ndimage import gaussian_filter, maximum_filter

    q_max = 1.0 / d_min
    lam_lo, lam_hi = float(wavelength[0]), float(wavelength[1])
    lam = lam_lo + (np.arange(LAUE_SAMPLES) + 0.5) * (lam_hi - lam_lo) / LAUE_SAMPLES
    if spectrum is None:
        shape = np.sin(np.pi * (np.arange(LAUE_SAMPLES) + 0.5) / LAUE_SAMPLES) ** 2
    else:
        shape = np.interp(
            lam, np.asarray(spectrum[0]), np.asarray(spectrum[1]), left=0.0, right=0.0
        )
    shape = np.maximum(shape, 0.0) / shape.sum()
    with h5py.File(merged, "r") as f:
        n_img = f["images"].shape[0]
        bank_ids = f["bank_ids"][()]
        offsets = f["file_offsets"][()] if "file_offsets" in f else np.array([0])
        rot = Goniometer.from_h5_file(f).rotation
        if rot.ndim == 2:
            rot = np.broadcast_to(rot, (n_img, 3, 3))
        if rotations is not None:
            rot = np.asarray(rotations)
        use = list(range(n_img)) if frames is None else list(frames)
        run_of = np.searchsorted(offsets, np.arange(n_img), side="right") - 1
        # first pass: significant pixels of every image, and how often each
        # detector position is significant across runs
        picked = {}
        seen = {}
        for i in use:
            img = f["images"][i].astype(np.float64)
            if weighting == "deviance":
                from scipy.ndimage import median_filter

                from subhkl.search.blind_band import count_search_weights

                b = binning or (8 if min(img.shape) >= 512 else 4)
                hb, wb = img.shape[0] // b, img.shape[1] // b
                y = img[: hb * b, : wb * b].reshape(hb, b, wb, b).sum(axis=(1, 3))
                mu = np.maximum(median_filter(y, size=7, mode="nearest"), 0.5)
                with np.errstate(divide="ignore", invalid="ignore"):
                    dev = 2.0 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
                wbin = count_search_weights(np.where(y > mu, dev, 0.0))
                wbin[0], wbin[-1], wbin[:, 0], wbin[:, -1] = 0.0, 0.0, 0.0, 0.0
                br, bc = np.nonzero(wbin > 1.0)
                rr, cc = br * b + b // 2, bc * b + b // 2
                picked[i] = (rr, cc, wbin[br, bc], img.shape)
                key = (int(bank_ids[i]), img.shape)
                runs = seen.setdefault(key, {})
                near = np.zeros(img.shape, dtype=bool)
                near[rr, cc] = True
                prev = runs.setdefault(int(run_of[i]), np.zeros(img.shape, dtype=bool))
                prev |= near
                continue
            fine, coarse = gaussian_filter(img, 1.5), gaussian_filter(img, 10.0)
            excess = fine - coarse
            z = excess / np.sqrt(np.maximum(coarse, 0.05) / (4.0 * np.pi * 1.5**2))
            # one entry per reflection (the local maximum of a significant
            # patch): with every pixel of a patch counted, broad and strong
            # reflections outweigh the rest and the map loses the lattice
            sig = (z > min_z) & (fine == maximum_filter(fine, size=9))
            edge = max(4, img.shape[0] // 32)
            sig[:edge], sig[-edge:], sig[:, :edge], sig[:, -edge:] = (
                False,
                False,
                False,
                False,
            )
            rr, cc = np.nonzero(sig)
            picked[i] = (rr, cc, excess[rr, cc], img.shape)
            key = (int(bank_ids[i]), img.shape)
            runs = seen.setdefault(key, {})
            near = np.zeros(img.shape, dtype=bool)
            near[rr, cc] = True
            near = maximum_filter(
                near, size=5
            )  # the same spot may peak a pixel or two away
            prev = runs.setdefault(int(run_of[i]), np.zeros(img.shape, dtype=bool))
            prev |= near
    fixed = {}
    for key, runs in seen.items():
        if len(runs) >= 3:
            fixed[key] = np.sum(list(runs.values()), axis=0) >= 3
    # A few very strong reflections must not dominate the map: every lattice
    # point counts, so the weights are capped at their 95th percentile.
    every = np.concatenate([p_[2] for p_ in picked.values()] or [np.zeros(1)])
    cap = float(np.quantile(every, 0.95)) if len(every) else 0.0
    if weighting == "deviance":
        cap = np.inf  # already a significance
    acc = jax.jit(_streak_impl, static_argnames=["n"])
    grid = jnp.zeros(n * n * n, dtype=jnp.float32)
    pos_cache = {}
    n_used = 0
    lam_j, shape_j = (
        jnp.asarray(lam, dtype=jnp.float32),
        jnp.asarray(shape, dtype=jnp.float32),
    )
    for i in use:
        rr, cc, wgt, shp = picked[i]
        wgt = np.minimum(wgt, cap)
        bank = int(bank_ids[i])
        mask = fixed.get((bank, shp))
        if mask is not None and len(rr):
            keep = ~mask[rr, cc]
            rr, cc, wgt = rr[keep], cc[keep], wgt[keep]
        if len(rr) == 0:
            continue
        if bank not in pos_cache:
            pos_cache[bank] = pixel_positions(instrument, bank, shp)
        xyz = pos_cache[bank][rr * shp[1] + cc]
        n_used += len(rr)
        for a in range(0, len(rr), LAUE_PIXEL_PAD):
            w_p = np.zeros(LAUE_PIXEL_PAD, dtype=np.float32)
            x_p = np.ones((LAUE_PIXEL_PAD, 3), dtype=np.float32)
            m = min(LAUE_PIXEL_PAD, len(rr) - a)
            w_p[:m] = wgt[a : a + m]
            x_p[:m] = xyz[a : a + m]
            grid = acc(
                grid,
                jnp.asarray(w_p),
                jnp.asarray(x_p),
                jnp.asarray(rot[i], dtype=jnp.float32),
                lam_j,
                shape_j,
                jnp.float32(q_max),
                n=n,
            )
    return grid.reshape(n, n, n), q_max, n_used


def merge_radial_peaks(pos, height, angle_deg=2.0, rel_length=0.12):
    """One peak per lattice vector.

    In the map of a Laue pattern a lattice vector shows as two or three
    maxima along its own direction (fringes of the average over the band).
    Peaks within ``angle_deg`` in direction and ``rel_length`` in length are
    replaced by their height-weighted mean.
    """
    if len(pos) == 0:
        return pos, height
    length = np.linalg.norm(pos, axis=1)
    unit = pos / length[:, None]
    used = np.zeros(len(pos), dtype=bool)
    out_p, out_h = [], []
    for i in range(len(pos)):
        if used[i]:
            continue
        same = (
            (~used)
            & (unit @ unit[i] > np.cos(np.radians(angle_deg)))
            & (np.abs(length / length[i] - 1.0) < rel_length)
        )
        used |= same
        w = height[same]
        out_p.append((pos[same] * w[:, None]).sum(0) / w.sum())
        out_h.append(height[same].max())
    out_p, out_h = np.array(out_p), np.array(out_h)
    order = np.argsort(-out_h)
    return out_p[order], out_h[order]


@contextlib.contextmanager
def scaled_detectors(instrument, radial=0.0, axial=None, shift=(0.0, 0.0, 0.0)):
    """Temporarily move every panel of an instrument outwards.

    The panel midpoints are scaled by ``1 + radial`` across the beam axis
    plane (x, z) and by ``1 + axial`` along y, the panels keep their size and
    orientation: the two leading terms of an uncalibrated geometry, in the
    parametrisation of the blind orientation search
    (``blind_band.discover``).

    Args:
        radial: [-] relative change of the panel distances in x and z
        axial: [-] relative change in y; default the same as ``radial``
        shift: [m] translation of all panels, i.e. minus the displacement of
            the sample from the nominal centre
    """
    from subhkl.config import beamlines

    axial = radial if axial is None else axial
    saved = {}
    scale = np.array([1.0 + radial, 1.0 + axial, 1.0 + radial])
    for bank, cfg in beamlines[instrument].items():
        if not isinstance(cfg, dict) or "center" not in cfg:
            continue
        saved[bank] = list(cfg["center"])
        half = 0.5 * cfg["width"] * np.array(cfg["uhat"]) + 0.5 * cfg[
            "height"
        ] * np.array(cfg["vhat"])
        cfg["center"] = (
            (np.array(cfg["center"]) + half) * scale
            - half
            + np.asarray(shift, dtype=float)
        ).tolist()
    try:
        yield
    finally:
        for bank, center in saved.items():
            beamlines[instrument][bank]["center"] = center


def laue_observations(
    merged,
    instrument,
    wavelength,
    d_min,
    frames=None,
    rotations=None,
    binning=None,
    max_bins=65536,
):
    """Count bins as the orientation search sees them, for scoring lattices.

    The images are binned so that a reflection fills about one bin; each bin
    gets the weight of ``blind_band.count_search_weights`` (the square root
    of its Poisson deviance against the local background, with background
    fluctuations suppressed), its scattering-vector direction in the sample
    frame, and the interval of |q| the wavelength band allows.

    Returns:
        dict with ``dirs`` (M, 3), ``q_lo``, ``q_hi`` [1/Angstrom] and
        ``weights`` (sum 1)
    """
    from scipy.ndimage import median_filter

    from subhkl.search.blind_band import count_search_weights

    lam_lo, lam_hi = float(wavelength[0]), float(wavelength[1])
    dirs, wts, qlo, qhi = [], [], [], []
    cache = {}
    with h5py.File(merged, "r") as f:
        n_img = f["images"].shape[0]
        bank_ids = f["bank_ids"][()]
        rot = Goniometer.from_h5_file(f).rotation
        if rot.ndim == 2:
            rot = np.broadcast_to(rot, (n_img, 3, 3))
        if rotations is not None:
            rot = np.asarray(rotations)
        for i in range(n_img) if frames is None else frames:
            img = f["images"][i].astype(np.float64)
            b = binning or (8 if min(img.shape) >= 512 else 4)
            hb, wb = img.shape[0] // b, img.shape[1] // b
            y = img[: hb * b, : wb * b].reshape(hb, b, wb, b).sum(axis=(1, 3))
            mu = np.maximum(median_filter(y, size=7, mode="nearest"), 0.5)
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = 2.0 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
            w = count_search_weights(np.where(y > mu, dev, 0.0))
            w[0], w[-1], w[:, 0], w[:, -1] = 0.0, 0.0, 0.0, 0.0
            br, bc = np.nonzero(w > 0.5)
            if len(br) == 0:
                continue
            bank = int(bank_ids[i])
            if bank not in cache:
                cache[bank] = pixel_positions(instrument, bank, img.shape).astype(
                    np.float64
                )
            xyz = cache[bank][(br * b + b // 2) * img.shape[1] + bc * b + b // 2]
            q = xyz / np.linalg.norm(xyz, axis=1)[:, None] - np.array([0.0, 0.0, 1.0])
            two_sin = np.linalg.norm(q, axis=1)
            dirs.append((q / two_sin[:, None]) @ rot[i])
            wts.append(w[br, bc])
            qlo.append(two_sin / lam_hi)
            qhi.append(np.minimum(two_sin / lam_lo, 1.0 / d_min))
    if not dirs:
        raise ValueError("No count bin stands above its background.")
    dirs, wts, qlo, qhi = (np.concatenate(v) for v in (dirs, wts, qlo, qhi))
    ok = qhi > 1.02 * qlo
    dirs, wts, qlo, qhi = dirs[ok], wts[ok], qlo[ok], qhi[ok]
    if len(wts) > max_bins:
        top = np.argsort(-wts)[:max_bins]
        dirs, wts, qlo, qhi = dirs[top], wts[top], qlo[top], qhi[top]
    return {"dirs": dirs, "q_lo": qlo, "q_hi": qhi, "weights": wts / wts.sum()}


def candidate_lattices(
    pos, height, n_peaks=30, n_fit=80, tol=0.15, max_candidates=300, min_length=0.0
):
    """Lattices that triples of the strongest map peaks span.

    Each triple of non-coplanar peaks is taken as a basis, refined by least
    squares on the peaks it indexes, and reduced.  Duplicates (the same
    lattice within 3% and 2 degrees) are dropped.  The map alone cannot say
    which of them is the lattice (a strong peak at half a lattice vector, or
    a missing short one, gives sub- and superlattices that index the peaks
    almost as well); ``ladder_scores`` decides.

    ``min_length`` [Angstrom] drops lattices with a shorter basis vector.

    Returns:
        list of (basis, share of the peak weight it indexes), best share first
    """
    import itertools

    keep = [i for i in range(len(pos)) if tuple(pos[i]) > tuple(-pos[i])]
    pos, height = pos[keep], height[keep]
    top = pos[:n_peaks]
    fit_p, fit_h = pos[:n_fit], height[:n_fit]
    found = []
    for i, j, k in itertools.combinations(range(len(top)), 3):
        A = np.stack([top[i], top[j], top[k]], 1)
        if abs(np.linalg.det(A)) < 0.15 * np.prod(np.linalg.norm(A, axis=0)):
            continue
        good = None
        for _ in range(2):
            c = np.linalg.solve(A, fit_p.T).T
            nint = np.rint(c)
            good = (np.abs(c - nint).max(axis=1) < tol) & (np.abs(nint).sum(axis=1) > 0)
            if good.sum() < 4:
                break
            N, X, W = nint[good].T, fit_p[good].T, fit_h[good]
            M = (N * W) @ N.T
            if abs(np.linalg.det(M)) < 1e-9:
                break
            A = (X * W) @ N.T @ np.linalg.inv(M)
        if good is None or good.sum() < 4:
            continue
        A = reduce_basis(A)
        # The peaks were searched beyond ``min_length`` only; a basis with a
        # shorter vector is a combination of errors, and the score would
        # favour it, because every bin satisfies the Laue condition of a
        # vector that is short enough.
        if np.linalg.norm(A, axis=0).min() < min_length:
            continue
        found.append((A, float(fit_h[good].sum() / fit_h.sum())))
    found.sort(key=lambda t: -t[1])
    unique = []
    for A, share in found:
        cell = np.array(cell_of(A))
        same = False
        for B, _s in unique:
            other = np.array(cell_of(B))
            if np.allclose(cell[:3], other[:3], rtol=0.03) and np.allclose(
                cell[3:], other[3:], atol=2.0
            ):
                T = np.linalg.solve(B, A)
                if np.abs(T - np.rint(T)).max() < 0.15:
                    same = True
                    break
        if not same:
            unique.append((A, share))
        if len(unique) >= max_candidates:
            break
    return unique


_LADDER_UVW = None


def _ladder_vectors(A, n_zones):
    """The ``n_zones`` shortest primitive lattice vectors of a basis and
    their weights 1/|x| (normalised)."""
    global _LADDER_UVW
    if _LADDER_UVW is None:
        r = range(-4, 5)
        _LADDER_UVW = np.array(
            [
                (a, b, c)
                for a in r
                for b in r
                for c in r
                if np.gcd.reduce([abs(a), abs(b), abs(c)]) == 1
                and (a, b, c) > (0, 0, 0)
            ]
        )
    X = _LADDER_UVW @ np.asarray(A).T
    length = np.linalg.norm(X, axis=1)
    order = np.argsort(length)[:n_zones]
    w = 1.0 / length[order]
    return X[order], w / w.sum(), _LADDER_UVW[order]


def ladder_scores(bases, obs, n_zones=60, n_null=24, polish=True):
    """The orientation search's band objective for a list of lattices.

    For a lattice vector x and a count bin with direction d and |q| between
    q_lo and q_hi, the Laue condition x.q = integer is averaged over that
    interval (``scan_band.band_sum``); the score sums it over the bins and
    over the ``n_zones`` shortest lattice vectors with weights 1/|x|.  A
    lattice with a spurious short vector, or with a true one missing, scores
    lower than the lattice itself, and so does one at the wrong scale.

    Args:
        bases: list of (3, 3) direct bases, columns [Angstrom], sample frame
        obs: output of ``laue_observations``
        n_null: random orientations of the best lattice for its z-score
        polish: refine the best lattice on the objective
    Returns:
        (scores (len(bases),), index of the best, its basis after the
        polish, its score, its z-score against random orientations)
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    from subhkl.search.scan_band import band_sum

    D, LO, HI, W = (
        jnp.asarray(obs[k], dtype=jnp.float32)
        for k in ("dirs", "q_lo", "q_hi", "weights")
    )
    score = jax.jit(lambda X, zw: band_sum(X, D, LO, HI, W, zw))
    scores = np.zeros(len(bases))
    for n, A in enumerate(bases):
        X, zw, _uvw = _ladder_vectors(A, n_zones)
        scores[n] = float(
            score(jnp.asarray(X, jnp.float32), jnp.asarray(zw, jnp.float32))
        )
    best = int(np.argmax(scores))
    A = np.array(bases[best], dtype=float)
    top = float(scores[best])
    if polish:
        _X, zw, uvw = _ladder_vectors(A, n_zones)
        uvw_j, zw_j = jnp.asarray(uvw, jnp.float32), jnp.asarray(zw, jnp.float32)
        A0 = jnp.asarray(A, jnp.float32)

        def neg(delta):
            return -band_sum(uvw_j @ (A0 + delta.reshape(3, 3)).T, D, LO, HI, W, zw_j)

        fun = jax.jit(jax.value_and_grad(neg))

        def f(delta):
            v, g = fun(jnp.asarray(delta, jnp.float32))
            return float(v), np.asarray(g, dtype=np.float64)

        res = minimize(
            f, np.zeros(9), jac=True, method="L-BFGS-B", options={"maxiter": 60}
        )
        if -res.fun > top:
            # The polish is kept for shape and orientation only.  The objective
            # assumes a flat spectrum between the band limits and gains a few
            # percent by shrinking the lattice when the real spectrum is
            # narrower, so the volume is put back to what the map gave.
            moved = A + res.x.reshape(3, 3)
            moved = moved * (abs(np.linalg.det(A)) / abs(np.linalg.det(moved))) ** (
                1 / 3
            )
            A, top = reduce_basis(moved), float(-res.fun)
    X, zw, _uvw = _ladder_vectors(A, n_zones)
    null = np.array(
        [
            float(
                score(jnp.asarray(X @ Rn.T, jnp.float32), jnp.asarray(zw, jnp.float32))
            )
            for Rn in Rotation.random(n_null, random_state=1).as_matrix()
        ]
    )
    z = (top - null.mean()) / max(float(null.std()), 1e-12)
    return scores, best, A, top, float(z)


def bragg_excess(grid, sigma=4.0, k=5.0):
    """Counts above the local background, zero elsewhere.

    The background is most of the events, and the detector gaps and band
    edges cut it into a pattern whose own Fourier transform has strong
    peaks (along the goniometer axis for a rotation scan).  Keep only what
    stands ``k`` standard deviations above a Gaussian-smoothed copy of the
    density: Bragg peaks are a few voxels wide and survive, the background
    and most of its edges do not.
    """
    import jax.numpy as jnp

    n = grid.shape[0]
    f = jnp.fft.fftfreq(n)
    k2 = f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2
    smooth = jnp.fft.ifftn(
        jnp.fft.fftn(grid) * jnp.exp(-2.0 * jnp.pi**2 * sigma**2 * k2)
    ).real
    smooth = jnp.maximum(smooth, 0.0)
    excess = grid - smooth
    return jnp.where(excess > k * jnp.sqrt(smooth + 1.0), excess, 0.0)


def direct_space_map(density, voxel, half_extent=330.0):
    """|G(x)| on the conjugate grid (voxel size 1 / (2 q_max)), cropped to
    +-half_extent [Angstrom] around the origin."""
    import jax.numpy as jnp

    g = jnp.fft.fftshift(jnp.abs(jnp.fft.fftn(density)))
    n = g.shape[0]
    half = min(n // 2, int(half_extent / voxel))
    c = n // 2
    return np.array(g[c - half : c + half, c - half : c + half, c - half : c + half])


def find_peaks(g, voxel, x_min=12.0, x_max=None, n_keep=400, size=5):
    """Local maxima of |G| away from the origin, strongest first.

    Returns positions [Angstrom] (refined to a fraction of a voxel) and heights.
    """
    from scipy.ndimage import maximum_filter

    n = g.shape[0]
    c = n // 2
    ax = (np.arange(n) - c) * voxel
    r2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
    x_max = x_max or (0.5 * n * voxel - 3 * voxel)
    mask = (r2 > x_min**2) & (r2 < x_max**2)
    is_max = (g == maximum_filter(g, size=size, mode="wrap")) & mask
    idx = np.argwhere(is_max)
    h = g[is_max]
    order = np.argsort(h)[::-1][:n_keep]
    idx, h = idx[order], h[order]
    pos = np.zeros((len(idx), 3))
    for j, (i0, i1, i2) in enumerate(idx):
        p = np.array([i0, i1, i2], dtype=float)
        for a in range(3):
            lo = [i0, i1, i2]
            hi = [i0, i1, i2]
            lo[a] = (lo[a] - 1) % n
            hi[a] = (hi[a] + 1) % n
            ym, y0, yp = (
                np.log(g[tuple(lo)] + 1e-9),
                np.log(g[i0, i1, i2] + 1e-9),
                np.log(g[tuple(hi)] + 1e-9),
            )
            den = ym - 2 * y0 + yp
            if den < 0:
                p[a] += np.clip(0.5 * (ym - yp) / den, -0.5, 0.5)
        pos[j] = (p - c) * voxel
    return pos, h


def _map_at(g, voxel, x):
    """Largest |G| within one voxel of each position x [Angstrom] (0 outside the map)."""
    n = g.shape[0]
    c = n // 2
    idx = np.rint(np.atleast_2d(x) / voxel).astype(int) + c
    out = np.zeros(len(idx))
    for j, (i0, i1, i2) in enumerate(idx):
        if 1 <= i0 < n - 1 and 1 <= i1 < n - 1 and 1 <= i2 < n - 1:
            out[j] = g[i0 - 1 : i0 + 2, i1 - 1 : i1 + 2, i2 - 1 : i2 + 2].max()
    return out


def lattice_basis(pos, height, g, voxel, tol=0.15):
    """A basis of the lattice the peaks lie on, columns [Angstrom].

    G is the Patterson function of the crystal: besides its origin peak
    repeated on every lattice vector it has weaker interatomic peaks, so a
    peak alone is not evidence of a lattice vector.  A lattice vector is
    one whose whole comb is occupied.

    - The first two axes are the two shortest independent peaks among the
      dominant ones (within a factor 2.5 of the strongest).
    - The third is the out-of-plane peak whose translates by the first two
      and whose multiples carry the most weight, scored on the map itself.
    - Every peak the basis then indexes refines it by least squares.

    Returns:
        (A, indexed mask, diagnostics): diagnostics holds the comb score of
        the third axis relative to the strongest peak, the share of peak
        weight indexed, and the rms misfit [Angstrom]
    """
    order = np.argsort(np.linalg.norm(pos, axis=1))
    dominant = [i for i in order if height[i] >= 0.4 * height.max()]
    a1 = pos[dominant[0]]
    a2 = next(
        pos[i]
        for i in dominant
        if np.linalg.norm(np.cross(a1, pos[i]))
        > 0.3 * np.linalg.norm(a1) * np.linalg.norm(pos[i])
    )
    normal = np.cross(a1, a2)
    normal /= np.linalg.norm(normal)
    in_plane = [a1 * i + a2 * j for i in (-1, 0, 1) for j in (-1, 0, 1)]
    best, best_score = None, -1.0
    for i in order:
        v = pos[i]
        if abs(v @ normal) < 0.25 * np.linalg.norm(a1):
            continue
        pts = np.array([m * v + t for m in (1, 2) for t in in_plane])
        vals = _map_at(g, voxel, pts)
        inside = vals > 0
        if inside.sum() < 9:
            continue
        score = vals[inside].mean()
        if score > 1.15 * best_score:  # ties go to the shorter vector
            best, best_score = v, score
    if best is None:
        raise ValueError(
            "No third lattice axis found in the direct-space map: the events "
            "show two-dimensional order at most. Check d_min and the grid."
        )
    A = np.stack([a1, a2, best], 1)
    for _ in range(3):
        coords = np.linalg.solve(A, pos.T).T
        nint = np.rint(coords)
        good = (np.abs(coords - nint).max(axis=1) < tol) & (
            np.abs(nint).sum(axis=1) > 0
        )
        N, X, W = nint[good].T, pos[good].T, height[good]
        A = (X * W) @ N.T @ np.linalg.inv((N * W) @ N.T)
    diagnostics = {
        "third_axis_score": float(best_score / height.max()),
        "indexed_weight": float(height[good].sum() / height.sum()),
        "rms_misfit": float(np.sqrt(np.mean(np.sum((X - A @ N) ** 2, axis=0)))),
    }
    return A, good, diagnostics


def refine_basis(A, q, w, extent=330.0):
    """Refine the basis on the data itself, off the voxel grid.

    The lattice Fourier sum F(x) = sum_i w_i cos(2 pi q_i . x) over the
    Bragg-excess voxels peaks at every direct-lattice vector.  The basis is
    moved to maximise the sum of F over all its lattice points within
    ``extent``: the many combinations pin down each axis far better than
    its own peak does, which matters for a long axis whose peak is weak.

    Returns:
        (A, score before, score after), scores as a share of the maximum
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    r = range(-4, 5)
    n_all = np.array(
        [(i, j, k) for i in r for j in r for k in r if (i, j, k) > (0, 0, 0)]
    )
    n_use = n_all[np.linalg.norm(n_all @ A.T, axis=1) < extent]
    qj = jnp.array(q, dtype=jnp.float32)
    wj = jnp.array(w, dtype=jnp.float32)
    nj = jnp.array(n_use, dtype=jnp.float32)
    norm = float(np.sum(w)) * len(n_use)
    A0 = jnp.array(A, dtype=jnp.float32)

    def neg_score(delta):
        # The phase is evaluated for the starting basis plus a small change,
        # and reduced to a turn before the cosine, to stay accurate in
        # single precision (q.x reaches a hundred turns).
        x = nj @ (A0 + delta.reshape(3, 3)).T  # lattice points [Angstrom]
        turns = qj @ x.T
        return (
            -jnp.sum(wj[:, None] * jnp.cos(2.0 * jnp.pi * (turns - jnp.round(turns))))
            / norm
        )

    fun = jax.jit(jax.value_and_grad(neg_score))

    def f(delta):
        v, g = fun(jnp.array(delta, dtype=jnp.float32))
        return float(v), np.array(g, dtype=np.float64)

    before = -f(np.zeros(9))[0]
    res = minimize(
        f, np.zeros(9), jac=True, method="L-BFGS-B", options={"maxiter": 200}
    )
    after = -float(res.fun)
    if after < before:
        return A, before, before
    return A + res.x.reshape(3, 3), before, after


def reduce_basis(A):
    """Shortest-vector (Minkowski) reduction of a 3D basis, right-handed,
    with the angles to the first axis obtuse."""
    A = np.array(A, dtype=float)
    changed = True
    while changed:
        changed = False
        order = np.argsort(np.linalg.norm(A, axis=0))
        A = A[:, order]
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                m = np.rint(A[:, i] @ A[:, j] / (A[:, j] @ A[:, j]))
                if m != 0:
                    cand = A[:, i] - m * A[:, j]
                    if np.linalg.norm(cand) < np.linalg.norm(A[:, i]) - 1e-9:
                        A[:, i] = cand
                        changed = True
    a, b, c = A[:, 0], A[:, 1], A[:, 2]
    if a @ b > 0:
        b = -b
    if a @ c > 0:
        c = -c
    A = np.stack([a, b, c], 1)
    if np.linalg.det(A) < 0:
        A = -A
    return A


def cell_of(A):
    """(a, b, c [Angstrom], alpha, beta, gamma [degree]) of a basis (columns)."""
    lengths = np.linalg.norm(A, axis=0)

    def ang(u, v):
        return np.degrees(
            np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
        )

    return (
        *(float(v) for v in lengths),
        float(ang(A[:, 1], A[:, 2])),
        float(ang(A[:, 0], A[:, 2])),
        float(ang(A[:, 0], A[:, 1])),
    )


def lattice_operations(cell, max_obliquity=3.0):
    """Rotations of the lattice itself.

    Returns:
        (list of integer matrices M acting on Miller indices, h' = M h;
        name of the rotation group)
    """
    ops = gemmi.find_lattice_symmetry(gemmi.UnitCell(*cell), "P", max_obliquity)
    mats = []
    for op in ops.sym_ops:
        w = np.array(op.rot, dtype=float) / gemmi.Op.DEN  # on fractional coordinates
        mats.append(np.rint(np.linalg.inv(w).T).astype(int))
    group = gemmi.find_spacegroup_by_ops(ops)
    return mats, (group.hm if group is not None else "unknown")


def symmetrized_cell(A, mats):
    """Cell of the basis with its metric averaged over the lattice
    rotations: the nearest cell that has the lattice symmetry exactly
    (for example a = b and gamma = 120 for a hexagonal lattice)."""
    G = A.T @ A
    acc = np.zeros((3, 3))
    for M in mats:
        W = np.linalg.inv(np.asarray(M, dtype=float)).T  # on direct coordinates
        acc += W.T @ G @ W
    G = acc / len(mats)
    a, b, c = np.sqrt(np.diag(G))
    ang = lambda i, j: float(
        np.degrees(np.arccos(np.clip(G[i, j] / np.sqrt(G[i, i] * G[j, j]), -1, 1)))
    )
    return float(a), float(b), float(c), ang(1, 2), ang(0, 2), ang(0, 1)


def _axis_vector(W):
    """Shortest lattice vector (integer direct coordinates) along the axis
    of a rotation W acting on direct coordinates."""
    vals, vecs = np.linalg.eig(W)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    v = v / np.abs(v).max()
    for m in range(1, 13):
        u = v * m
        if np.allclose(u, np.rint(u), atol=1e-6):
            u = np.rint(u).astype(int)
            break
    else:
        return None
    u = u // np.gcd.reduce(np.abs(u))
    return max(tuple(-u), tuple(u))


def conventional_basis(A, max_obliquity=3.0):
    """Conventional basis of a primitive basis, from the lattice's own rotations.

    The axes are the shortest lattice vectors along rotation axes:

    - monoclinic: b along the two-fold; a and c the two shortest vectors of
      the lattice plane perpendicular to it, beta obtuse (the cell is then
      P, C, A or I, whichever the lattice gives)
    - orthorhombic: along the three two-folds, shortest first
    - tetragonal: c along the four-fold, a and b along the shortest
      perpendicular two-folds (P or I)
    - trigonal and hexagonal: c along the three- or six-fold, a and b along
      the shortest perpendicular two-folds, 120 degrees apart (P, or R in
      the obverse setting)
    - cubic: along the three four-folds (P, I or F)
    - triclinic: unchanged

    Returns:
        (C, T, multiplicity): C = A @ T (columns [Angstrom]); Miller indices
        transform as h_conventional = T^T h_primitive
    """
    cell = cell_of(A)
    ops = gemmi.find_lattice_symmetry(gemmi.UnitCell(*cell), "P", max_obliquity)
    axes = {}
    twofold = None
    for op in ops.sym_ops:
        w = np.array(op.rot, dtype=float) / gemmi.Op.DEN
        order = next(
            (
                n
                for n in (1, 2, 3, 4, 6)
                if np.allclose(np.linalg.matrix_power(w, n), np.eye(3), atol=1e-6)
            ),
            None,
        )
        if order in (None, 1):
            continue
        u = _axis_vector(w)
        if u is None:
            continue
        axes[u] = max(order, axes.get(u, 0))
        if order == 2:
            twofold = w
    n_rot = len(ops.sym_ops)
    length = lambda u: float(np.linalg.norm(A @ np.array(u, dtype=float)))
    cosine = lambda u, v: float(
        (A @ np.array(u, float)) @ (A @ np.array(v, float)) / (length(u) * length(v))
    )
    by_len = lambda us: sorted(us, key=length)
    chosen = None
    if n_rot == 2 and twofold is not None:
        (bvec,) = axes
        r = range(-3, 4)
        plane = [
            (i, j, k)
            for i in r
            for j in r
            for k in r
            if (i, j, k) != (0, 0, 0)
            and not np.any(np.rint((twofold + np.eye(3)) @ np.array([i, j, k])))
        ]
        plane = by_len(plane)
        avec = plane[0]
        cvec = next(v for v in plane if abs(cosine(avec, v)) < 0.999)
        if cosine(avec, cvec) > 0:
            cvec = tuple(-np.array(cvec))
        chosen = [avec, bvec, cvec]
    elif n_rot == 4:
        chosen = by_len(axes)
    elif n_rot in (6, 8, 12):
        main = next(
            u for u, o in axes.items() if o == (4 if n_rot == 8 else max(axes.values()))
        )
        perp = by_len(
            [u for u, o in axes.items() if u != main and abs(cosine(u, main)) < 0.05]
        )
        avec = perp[0]
        want = 0.0 if n_rot == 8 else -0.5
        cands = [v for u in perp for v in (u, tuple(-np.array(u)))]
        bvec = next(
            (
                v
                for v in cands
                if abs(cosine(avec, v) - want) < 0.05
                and abs(length(v) / length(avec) - 1) < 0.03
            ),
            None,
        )
        if bvec is not None:
            chosen = [avec, bvec, main]
    elif n_rot == 24:
        four = by_len([u for u, o in axes.items() if o == 4])
        if len(four) == 3:
            chosen = four
    identity = np.eye(3)
    if chosen is None:
        return A, identity, 1
    T = np.array(chosen, dtype=float).T
    if abs(np.linalg.det(T)) < 0.5:
        return A, identity, 1
    if np.linalg.det(T) < 0:
        T[:, 1 if n_rot == 2 else 2] *= -1
    # rhombohedral lattice: use the obverse setting
    if n_rot == 6 and centring_of(T)[0] == "R(reverse)":
        T[:, :2] *= -1
    return A @ T, T, round(abs(np.linalg.det(T)))


def centring_of(T):
    """Centring of the conventional cell A @ T: (letter, list of fractional
    lattice points inside the cell besides the origin)."""
    Ti = np.linalg.inv(np.asarray(T, dtype=float))
    r = range(-3, 4)
    pts = {
        tuple(np.round((Ti @ np.array([i, j, k])) % 1.0, 6) % 1.0)
        for i in r
        for j in r
        for k in r
    }
    pts.discard((0.0, 0.0, 0.0))
    key = frozenset(tuple(round(float(x) * 6) for x in p) for p in pts)
    names = {
        frozenset(): "P",
        frozenset({(3, 3, 0)}): "C",
        frozenset({(0, 3, 3)}): "A",
        frozenset({(3, 0, 3)}): "B",
        frozenset({(3, 3, 3)}): "I",
        frozenset({(3, 3, 0), (0, 3, 3), (3, 0, 3)}): "F",
        frozenset({(4, 2, 2), (2, 4, 4)}): "R",
        frozenset({(2, 4, 2), (4, 2, 4)}): "R(reverse)",
    }
    return names.get(key, "?"), sorted(pts)


def lattice_group_of(A, T, max_obliquity=3.0):
    """Symbol of the lattice's rotation group in the conventional setting,
    with its centring (for example "I 1 2 1"), or "unknown"."""
    ops = gemmi.find_lattice_symmetry(gemmi.UnitCell(*cell_of(A)), "P", max_obliquity)
    Ti = np.linalg.inv(T)
    cen = [[0, 0, 0]] + [
        [round(x * gemmi.Op.DEN) for x in p] for p in centring_of(T)[1]
    ]
    full = []
    for op in ops.sym_ops:
        w = Ti @ (np.array(op.rot, dtype=float) / gemmi.Op.DEN) @ T
        for shift in cen:
            o = gemmi.Op()
            o.rot = np.rint(w * gemmi.Op.DEN).astype(int).tolist()
            o.tran = shift
            full.append(o)
    try:
        group = gemmi.find_spacegroup_by_ops(gemmi.GroupOps(full))
    except (RuntimeError, ValueError):
        group = None
    return group.xhm() if group is not None else "unknown"


P1_SYMBOLS = {"P": "P 1", "C": "C 1", "A": "A 1", "B": "B 1", "I": "I 1", "F": "F 1"}


def operations_in_basis(mats, T):
    """Miller-index operations after a change of basis h' = T^T h."""
    Tt = np.asarray(T, dtype=float).T
    Ti = np.linalg.inv(Tt)
    return [np.rint(Tt @ np.asarray(M, dtype=float) @ Ti).astype(int) for M in mats]


def wavelength_band(merged, instrument, n_sample=4_000_000):
    """[Angstrom] wavelength range the events cover, to one decimal."""
    l1 = float(reduction_settings[instrument]["L1"])
    with h5py.File(merged, "r") as f:
        tof = f["events/tof"]
        step = max(1, tof.shape[0] // n_sample)
        t = tof[::step][:n_sample]
    lam = np.percentile(t, [0.5, 99.5]) * ANGSTROM_M_PER_US / (l1 + 0.42)
    return float(np.round(lam[0], 1)), float(np.round(lam[1], 1))


@dataclass
class BlindCell:
    """Result of ``find_cell``.

    ``basis`` is the reduced primitive basis.  ``setting_basis`` is the
    basis the rest of the pipeline should index in: the conventional cell
    where the lattice is centred and a centred P 1 symbol exists
    (``p1_symbol``, e.g. "I 1"), otherwise the primitive one.
    """

    basis: np.ndarray  # reduced primitive basis, columns [Angstrom], sample frame
    cell: tuple  # a, b, c [Angstrom], alpha, beta, gamma [degree] of ``basis``
    conventional_basis: np.ndarray
    conventional_cell: tuple
    to_conventional: np.ndarray  # T: h_conventional = T^T h_primitive
    multiplicity: int  # lattice points per conventional cell
    centring: str  # P, C, A, B, I, F or R
    operations: list  # lattice rotations on primitive Miller indices
    lattice_group: str  # rotation group of the lattice, conventional setting
    setting_basis: np.ndarray
    setting_cell: tuple  # with the lattice symmetry imposed on the metric
    setting_operations: list  # lattice rotations on the setting's Miller indices
    p1_symbol: str  # space group for the one integration without symmetry
    setting_group: str  # lattice rotation group valid for ``setting_cell`` (constrains the refinement)
    diagnostics: dict = field(default_factory=dict)


def describe_lattice(A):
    """Everything ``BlindCell`` holds besides the diagnostics, from a primitive basis."""
    A = reduce_basis(A)
    cell = cell_of(A)
    mats, primitive_group = lattice_operations(cell)
    C, T, mult = conventional_basis(A)
    letter = centring_of(T)[0]
    conv_ops = operations_in_basis(mats, T)
    if mult == 1:
        group = lattice_group_of(A, T)
        if group == "unknown":
            group = primitive_group
        setting, ops, p1 = C, conv_ops, "P 1"
    else:
        group = lattice_group_of(A, T)
        if letter in P1_SYMBOLS and gemmi.find_spacegroup_by_name(P1_SYMBOLS[letter]):
            setting, ops, p1 = C, conv_ops, P1_SYMBOLS[letter]
        else:  # e.g. rhombohedral: integrate in the primitive cell
            setting, ops, p1 = A, mats, "P 1"
    return {
        "basis": A,
        "cell": cell,
        "conventional_basis": C,
        "conventional_cell": cell_of(C),
        "to_conventional": T,
        "multiplicity": mult,
        "centring": letter,
        "operations": mats,
        "lattice_group": group,
        "setting_basis": setting,
        "setting_cell": symmetrized_cell(setting, ops),
        "setting_operations": ops,
        "p1_symbol": p1,
        "setting_group": group
        if setting is C
        else (primitive_group if primitive_group != "unknown" else "P 1"),
    }


def find_cell(
    merged,
    instrument,
    d_min=3.0,
    grid=768,
    x_min=12.0,
    log=None,
    frames=None,
    rotations=None,
    wavelength=None,
    spectrum=None,
    min_z=8.0,
    weighting="maxima",
    volume=None,
    length=None,
):
    """Unit cell, scale and orientation from raw data, with nothing given.

    With events (time of flight) every count is a point in reciprocal
    space.  Without them (``wavelength`` given) every significant pixel is
    a radial streak over the band (``laue_histogram``).  From there the two
    are the same: Fourier transform, peaks on the direct lattice, a basis
    that indexes them, refinement on the data.

    For Laue data the overall scale is only as good as the band and the
    spectrum; the shape of the cell does not depend on them.

    Args:
        merged: images file with goniometer angles, and pixel-sorted events
            unless ``wavelength`` is given
        instrument: instrument name
        d_min: [Angstrom] resolution of the reciprocal-space histogram; the
            direct-space map then has voxels of d_min / 2
        grid: [-] grid points per axis; the map reaches grid * d_min / 4
            from the origin and must hold a few cell lengths
        x_min: [Angstrom] shortest lattice vector considered
        log: optional callable for progress lines
        frames: optional image indices to use (for example one run)
        rotations: optional (n_images, 3, 3) goniometer rotations to use
            instead of the ones in the file
        wavelength: [Angstrom] (lam_min, lam_max) of a Laue measurement; the
            events are then not used
        spectrum: optional (wavelengths, relative weights) for Laue data
        min_z: [-] significance of a pixel for Laue data (weighting "maxima")
        weighting: "deviance" or "maxima" for Laue data, see ``laue_histogram``
        volume: [Angstrom^3] for Laue data, the volume of the conventional
            cell if known (for example from one time-of-flight run of the
            same crystal form); sets the overall scale, which Laue data hold
            only in their band limits (``laue_scale``)
        length: [Angstrom] for Laue data, one edge of the conventional cell
            if known; the edge closest to it is set to it
    Returns:
        BlindCell
    """
    say = log or (lambda *_: None)
    if wavelength is None:
        counts, q_max, n_events = reciprocal_histogram(
            merged, instrument, d_min, grid, frames=frames, rotations=rotations
        )
        voxel = 1.0 / (2.0 * q_max)
        density = bragg_excess(counts)
        say(
            f"{float(counts.sum()):.3g} of {n_events:.3g} events inside |q_i| <= {q_max:.3f} 1/A; "
            f"{float(density.sum()):.3g} counts above the local background"
        )
        g = direct_space_map(density, voxel)
        pos, height = find_peaks(g, voxel, x_min=x_min)
    else:
        density, q_max, n_events = laue_histogram(
            merged,
            instrument,
            wavelength,
            d_min,
            grid,
            frames=frames,
            rotations=rotations,
            spectrum=spectrum,
            min_z=min_z,
            weighting=weighting,
        )
        voxel = 1.0 / (2.0 * q_max)
        say(
            f"{n_events} significant pixels spread over {wavelength[0]:g}-{wavelength[1]:g} A"
        )
        g = direct_space_map(density, voxel, half_extent=0.5 * grid * voxel - 2 * voxel)
        pos, height = merge_radial_peaks(*find_peaks(g, voxel, x_min=x_min, n_keep=300))
    if len(pos) < 3:
        raise ValueError(
            "Fewer than three peaks in the direct-space map: no lattice in the events."
        )
    if wavelength is not None:
        # The map proposes, the ladder objective decides: a linear map takes
        # every structure in the data for a peak, whereas a score of a whole
        # lattice on the count bins averages out what does not belong to it.
        cands = candidate_lattices(pos, height, min_length=0.8 * x_min)
        if not cands:
            raise ValueError(
                "No three independent peaks in the direct-space map: no lattice in the images."
            )
        obs = laue_observations(
            merged, instrument, wavelength, d_min, frames=frames, rotations=rotations
        )
        scores, best, A, top, z = ladder_scores([c[0] for c in cands], obs)
        factor, contrast, scale_found = laue_scale(A, obs)
        scale_source = "band limits" if scale_found else "map"
        if volume is not None or length is not None:
            conv = np.array(describe_lattice(A)["conventional_cell"])
            if volume is not None:
                al, be, ga = np.radians(conv[3:])
                v = np.prod(conv[:3]) * np.sqrt(
                    1
                    - np.cos(al) ** 2
                    - np.cos(be) ** 2
                    - np.cos(ga) ** 2
                    + 2 * np.cos(al) * np.cos(be) * np.cos(ga)
                )
                factor = float((volume / v) ** (1.0 / 3.0))
            else:
                edge = conv[:3][np.argmin(np.abs(np.log(conv[:3] / length)))]
                factor = float(length / edge)
            scale_found, scale_source = True, "given"
        A = factor * A
        order = np.argsort(-scores)
        other = next(
            (
                float(scores[i])
                for i in order[1:]
                if abs(
                    abs(np.linalg.det(np.linalg.solve(cands[best][0], cands[i][0])))
                    - 1.0
                )
                > 0.2
            ),
            float("nan"),
        )
        diag = {
            "candidates": len(cands),
            "ladder_score": top,
            "ladder_z": z,
            "next_other_lattice_score": other,
            "map_share_of_winner": cands[best][1],
            "map_rank_of_winner": int(best),
            "bins": len(obs["weights"]),
            "scale_factor": factor,
            "scale_contrast": contrast,
            "scale_determined": scale_found,
            "scale_source": scale_source,
            "events": n_events,
        }
        say(
            f"{len(cands)} candidate lattices from the map, scored on {len(obs['weights'])} count bins: "
            f"best {1e3 * top:.2f} (z {z:.0f} against random orientations), next different lattice "
            f"{1e3 * other:.2f}; the map alone ranked it {best + 1}"
        )
        say(
            f"scale from the {scale_source}: factor {factor:.3f} (contrast of the band limits {contrast:.2f})"
            if scale_found
            else f"the band limits do not fix the scale (contrast {contrast:.2f}): the scale is the map's, good to about 10%"
        )
    else:
        A, good, diag = lattice_basis(pos, height, g, voxel)
        say(
            f"lattice: {int(good.sum())} of {len(pos)} map peaks indexed "
            f"({100 * diag['indexed_weight']:.0f}% of the peak weight), "
            f"rms misfit {diag['rms_misfit']:.2f} A, third-axis comb {diag['third_axis_score']:.2f}"
        )
        dens = np.array(density)
        nz = np.argwhere(dens > 0)
        wts = dens[dens > 0]
        if (
            len(wts) > 400000
        ):  # streaks fill many voxels: the heaviest carry the lattice
            top = np.argsort(-wts)[:400000]
            nz, wts = nz[top], wts[top]
        q_pts = (nz + 0.5) * (2.0 * q_max / grid) - q_max
        extent = 330.0 if wavelength is None else min(330.0, 0.45 * grid * voxel)
        A, before, after = refine_basis(A, q_pts, wts, extent=extent)
        diag.update(fourier_sum_before=before, fourier_sum_after=after, events=n_events)
    info = describe_lattice(A)
    say(
        "primitive cell: a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *info["cell"]
        )
    )
    say(
        "conventional cell: a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *info["conventional_cell"]
        )
        + f"; centring {info['centring']}; lattice rotations: {len(info['operations'])} ({info['lattice_group']})"
    )
    return BlindCell(**info, diagnostics=diag)


def _nearest_rotation(M):
    u, _s, vt = np.linalg.svd(M)
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def _rotation_angle(Rm):
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))))


def relate_lattices(basis, reference, tol=0.05):
    """Rotation M with basis ~ M @ reference, for two bases of one lattice
    that may differ by an integer change of basis and a small rotation.

    Returns:
        (M, ok): ok is False when no integer change of basis of determinant
        +-1 relates them, or the lattices do not coincide after the rotation
        to within ``tol`` of the shortest axis.
    """
    T = np.rint(np.linalg.solve(basis, reference))
    if abs(round(float(np.linalg.det(T)))) != 1:
        return np.eye(3), False
    M = _nearest_rotation(basis @ T @ np.linalg.inv(reference))
    resid = np.abs(M.T @ basis @ T - reference).max()
    if resid > tol * np.linalg.norm(reference, axis=0).min():
        return np.eye(3), False
    return M, True


@dataclass
class RunAlignment:
    """Result of ``align_runs``."""

    rotations: np.ndarray  # (n_images, 3, 3) corrected goniometer rotations
    corrections: list  # per run: rotation applied in the nominal sample frame
    angles: list  # [degree] size of each correction
    aligned: list  # per run: False where the run's lattice could not be related
    cells: list  # per run: reduced cell found from that run alone
    axis: (
        np.ndarray | None
    )  # lab-frame axis of the run-to-run rotations, if they share one
    axis_nominal: np.ndarray | None
    axis_tilt: float | None  # [degree] between the two


def align_runs(
    merged,
    instrument,
    d_min=3.0,
    grid=768,
    x_min=12.0,
    log=None,
    wavelength=None,
    min_z=8.0,
):
    """Goniometer corrections from the data, with no cell and no model of
    the goniometer.

    Combining runs needs the rotation between them, and a goniometer whose
    axes are a few degrees off spoils that (the lattices of the runs no
    longer coincide and the pooled cell search blurs).  But each run on its
    own gives the lattice in its own nominal sample frame.  The same crystal
    was measured throughout, so the rotation that carries the lattice of the
    first run onto the lattice of run r is the error of run r's rotation
    relative to the first.  The lattices may come out in equivalent bases;
    the integer change of basis is found first, which works while the error
    is small against the angles between lattice vectors.

    Args:
        merged: images file with pixel-sorted events, goniometer angles and
            ``file_offsets`` (first image of each run)
        instrument: instrument name
        d_min, grid, x_min: as for ``find_cell``, applied to single runs
        log: optional callable for progress lines
        wavelength: [Angstrom] (lam_min, lam_max) for Laue data; each run's
            lattice then comes from ``find_cell`` in Laue mode
        min_z: [-] for Laue data, a run's lattice is only used if its ladder
            score stands this many standard deviations above random
            orientations
    Returns:
        RunAlignment.  ``rotations`` goes to ``find_cell(rotations=...)``.
        When the run-to-run rotations share one axis, it is reported with
        its tilt from the nominal axis: a goniometer calibration.
    """
    say = log or (lambda *_: None)
    with h5py.File(merged, "r") as f:
        n_img = f["images"].shape[0]
        offsets = f["file_offsets"][()].tolist() if "file_offsets" in f else [0]
        r_nom = Goniometer.from_h5_file(f).rotation
    if r_nom.ndim == 2:
        r_nom = np.broadcast_to(r_nom, (n_img, 3, 3))
    r_nom = np.array(r_nom, dtype=float)
    bounds = [*offsets, n_img]
    runs = [range(bounds[r], bounds[r + 1]) for r in range(len(offsets))]

    bases, cells = [], []
    for r, frames in enumerate(runs):
        try:
            found = find_cell(
                merged,
                instrument,
                d_min=d_min,
                grid=grid,
                x_min=x_min,
                frames=frames,
                wavelength=wavelength,
            )
            if (
                wavelength is not None
                and found.diagnostics.get("ladder_z", 0.0) < min_z
            ):
                raise ValueError("no lattice stands out in this run")
            bases.append(found.basis)
            cells.append(found.cell)
        except ValueError:
            bases.append(None)
            cells.append(None)
    first = next((r for r, b in enumerate(bases) if b is not None), None)
    if first is None:
        raise ValueError("No run gave a lattice on its own.")
    corrections, aligned = [], []
    for r, basis in enumerate(bases):
        ok, M = basis is not None, np.eye(3)
        if ok and r != first:
            if wavelength is None:
                M, ok = relate_lattices(basis, bases[first])
            else:
                # In Laue mode the scale of a single run is only good to a
                # few percent: compare the lattices at equal volume.
                same = basis * (
                    abs(np.linalg.det(bases[first])) / abs(np.linalg.det(basis))
                ) ** (1.0 / 3.0)
                M, ok = relate_lattices(same, bases[first], tol=0.1)
        corrections.append(M)
        aligned.append(bool(ok))
        say(
            f"run {r}: "
            + (
                f"correction {_rotation_angle(M):.2f} deg"
                if ok
                else "not aligned (its lattice does not match the first run's)"
            )
        )
    rotations = r_nom.copy()
    for r, frames in enumerate(runs):
        for i in frames:
            rotations[i] = r_nom[i] @ corrections[r]

    # Do the corrected run-to-run rotations share one axis?  (lab frame)
    def rel_axes(rot):
        out = []
        for r in range(len(runs)):
            if r == first or not aligned[r]:
                continue
            rel = rot[runs[r][0]] @ rot[runs[first][0]].T
            v = np.array(
                [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
            )
            if np.linalg.norm(v) > 0.03:  # more than about a degree apart
                out.append(v / np.linalg.norm(v))
        return np.array(out)

    axis = axis_nom = tilt = None
    ax, ax_n = rel_axes(rotations), rel_axes(r_nom)
    if len(ax) >= 2 and len(ax) == len(ax_n):
        ax = ax * np.sign(np.sum(ax * ax_n, axis=1))[:, None]  # same sense as nominal
        mean, mean_n = ax.mean(0), ax_n.mean(0)
        mean, mean_n = mean / np.linalg.norm(mean), mean_n / np.linalg.norm(mean_n)
        scatter = np.degrees(np.arccos(np.clip(ax @ mean, -1, 1)))
        scatter_n = np.degrees(np.arccos(np.clip(ax_n @ mean_n, -1, 1)))
        if scatter.max() < 0.5 and scatter_n.max() < 0.5:
            axis, axis_nom = mean, mean_n
            tilt = float(np.degrees(np.arccos(np.clip(mean @ mean_n, -1, 1))))
            say(
                f"the runs rotate about one axis, [{mean[0]:+.3f} {mean[1]:+.3f} {mean[2]:+.3f}] "
                f"in the lab frame, {tilt:.2f} deg from the nominal one"
            )
    if axis is not None and not all(aligned):
        # A run that gave no lattice of its own is placed with the axis the
        # other runs share, turned by its nominal angle from the first run.
        from scipy.spatial.transform import Rotation

        i0 = runs[first][0]
        for r, frames in enumerate(runs):
            if aligned[r]:
                continue
            rel = r_nom[frames[0]] @ r_nom[i0].T
            v = np.array(
                [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
            )
            theta = np.radians(_rotation_angle(rel)) * np.sign(v @ axis_nom)
            fixed = Rotation.from_rotvec(theta * axis).as_matrix() @ r_nom[i0]
            corrections[r] = r_nom[frames[0]].T @ fixed
            for i in frames:
                rotations[i] = r_nom[i] @ corrections[r]
            say(
                f"run {r}: placed with the common axis ({_rotation_angle(corrections[r]):.2f} deg)"
            )
    return RunAlignment(
        rotations=rotations,
        corrections=corrections,
        angles=[_rotation_angle(M) for M in corrections],
        aligned=aligned,
        cells=cells,
        axis=axis,
        axis_nominal=axis_nom,
        axis_tilt=tilt,
    )


def ladder_scale(basis, obs, n_zones=60, span=0.25, steps=101):
    """The ladder objective as a function of the lattice's overall scale.

    Returns:
        (scale factors (steps,), scores (steps,)); the lattice is
        ``factor * basis``
    """
    import jax
    import jax.numpy as jnp

    from subhkl.search.scan_band import band_sum

    D, LO, HI, W = (
        jnp.asarray(obs[k], dtype=jnp.float32)
        for k in ("dirs", "q_lo", "q_hi", "weights")
    )
    X, zw, _uvw = _ladder_vectors(basis, n_zones)
    Xj, zwj = jnp.asarray(X, jnp.float32), jnp.asarray(zw, jnp.float32)
    factors = np.linspace(1.0 - span, 1.0 + span, steps)
    scores = jax.jit(
        lambda F: jax.lax.map(lambda f: band_sum(f * Xj, D, LO, HI, W, zwj), F)
    )(jnp.asarray(factors, jnp.float32))
    return factors, np.asarray(scores)


def indexed_share(basis, obs, turn=None, tol=0.08, samples=81):
    """Share of the count-bin weight that the full lattice indexes.

    A bin is indexed if, for some |q| in its interval, the scattering vector
    lies within ``tol`` of a reciprocal lattice point in every index.  The
    ladder objective looks at low-order lattice vectors only and has false
    maxima a few degrees from the truth; this test uses the whole lattice,
    and a wrong orientation falls to the chance level.

    Args:
        turn: optional (3, 3) rotation applied to the lattice
    Returns:
        share [-] between 0 and 1
    """
    A = np.asarray(basis, dtype=float) if turn is None else np.asarray(turn) @ basis
    w = obs["weights"]
    t = np.linspace(0.0, 1.0, samples)[None, :]
    q = obs["q_lo"][:, None] + t * (obs["q_hi"] - obs["q_lo"])[:, None]
    h = (obs["dirs"] @ A)[:, None, :] * q[:, :, None]
    miss = np.abs(h - np.rint(h)).max(axis=2).min(axis=1)
    return float((w * (miss < tol)).sum() / w.sum())


def laue_scale(
    basis, obs, span=0.15, steps=61, tol=0.05, max_bins=6000, min_contrast=0.25
):
    """Overall scale of a lattice from Laue count bins, where they hold it.

    Directions fix the shape of a lattice but not its size; the size is in
    the band limits only.  A bin whose |q| interval is shorter than the
    spacing of lattice points along its ray is indexed at some scales and
    not at others, so the share of bins the full lattice indexes
    (``indexed_share``) has a maximum at the right scale.  How clear that
    maximum is depends on the band: clear for a band of 3.0 to 4.3 Angstrom,
    absent for one spanning an octave, where every ray has a lattice point
    inside every interval.

    Returns:
        (factor, contrast, determined): the lattice is ``factor * basis``;
        ``contrast`` is (max - min) / max of the share over the scan;
        ``determined`` says whether it reaches ``min_contrast`` (otherwise
        the factor is 1)
    """
    top = np.argsort(-obs["weights"])[:max_bins]
    sub = {k: v[top] for k, v in obs.items()}
    factors = np.linspace(1.0 - span, 1.0 + span, steps)
    share = np.array(
        [indexed_share(f * np.asarray(basis), sub, tol=tol) for f in factors]
    )
    contrast = float((share.max() - share.min()) / max(share.max(), 1e-12))
    if contrast < min_contrast:
        return 1.0, contrast, False
    # middle of the plateau: steadier than the highest point of a flat top
    high = factors[share >= share.max() - 0.25 * (share.max() - share.min())]
    return float(0.5 * (high.min() + high.max())), contrast, True


def refine_run_rotations(
    merged,
    instrument,
    basis,
    wavelength,
    rotations=None,
    d_min=3.0,
    n_zones=60,
    span_deg=3.0,
    max_bins=8192,
    log=None,
):
    """Orientation of every run against one shared lattice, for Laue data.

    The lattices of single runs (``align_runs``) give the rotation between
    runs to about a degree.  With the cell fixed, a run has only its
    orientation left to determine, and the ladder objective pins that down
    much better: for every run the lattice is turned by a small rotation
    that maximises the band objective on that run's count bins (a grid over
    +-``span_deg`` and a gradient polish propose; the share of bins the full
    lattice indexes, ``indexed_share``, accepts or rejects).  The result
    is a correction of each run's rotation, in the same sense as
    ``align_runs``, which ``goniometer_angles_for`` turns into motor angles.

    Args:
        merged: images file with goniometer angles and ``file_offsets``
        basis: (3, 3) shared direct basis, columns [Angstrom], sample frame
        wavelength: [Angstrom] (lam_min, lam_max)
        rotations: (n_images, 3, 3) current rotations; default the nominal ones
        span_deg: [degree] half-width of the search around each run's
            current orientation
        max_bins: count bins used per run (the most significant)
    Returns:
        (rotations (n_images, 3, 3), angles [degree] per run, per run the
        score before and after and the share of bins indexed before and
        after (n_runs, 4))
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    from subhkl.search.scan_band import band_sum, rodrigues_jax

    say = log or (lambda *_: None)
    with h5py.File(merged, "r") as f:
        n_img = f["images"].shape[0]
        offsets = f["file_offsets"][()].tolist() if "file_offsets" in f else [0]
        r_nom = Goniometer.from_h5_file(f).rotation
    if r_nom.ndim == 2:
        r_nom = np.broadcast_to(r_nom, (n_img, 3, 3))
    rot = np.array(r_nom if rotations is None else rotations, dtype=float)
    bounds = [*offsets, n_img]
    _X, zw, uvw = _ladder_vectors(basis, n_zones)
    uvw_j, zw_j = jnp.asarray(uvw, jnp.float32), jnp.asarray(zw, jnp.float32)
    A0 = jnp.asarray(basis, jnp.float32)

    def score(delta, D, LO, HI, W):
        return band_sum(uvw_j @ (rodrigues_jax(delta) @ A0).T, D, LO, HI, W, zw_j)

    score_j = jax.jit(score)
    grid_j = jax.jit(
        lambda G, D, LO, HI, W: jax.lax.map(lambda d: score(d, D, LO, HI, W), G)
    )
    grad_j = jax.jit(
        jax.value_and_grad(lambda d, D, LO, HI, W: -score(d, D, LO, HI, W))
    )
    steps = np.arange(-span_deg, span_deg + 0.01, 1.0)
    G = jnp.asarray(
        np.radians([(a, b, c) for a in steps for b in steps for c in steps]),
        jnp.float32,
    )
    out = rot.copy()
    angles, gains = [], []
    for r in range(len(offsets)):
        frames = range(bounds[r], bounds[r + 1])
        try:
            obs = laue_observations(
                merged,
                instrument,
                wavelength,
                d_min,
                frames=frames,
                rotations=rot,
                max_bins=max_bins,
            )
        except ValueError:
            angles.append(0.0)
            gains.append((np.nan, np.nan, np.nan, np.nan))
            continue
        m = len(obs["weights"])
        pad = max_bins - m
        data = tuple(
            jnp.asarray(
                np.pad(obs[k], [(0, pad)] + [(0, 0)] * (obs[k].ndim - 1)), jnp.float32
            )
            for k in ("dirs", "q_lo", "q_hi", "weights")
        )
        before = float(score_j(jnp.zeros(3, jnp.float32), *data))
        coarse = np.asarray(grid_j(G, *data))

        def f(d, data=data):
            v, g = grad_j(jnp.asarray(d, jnp.float32), *data)
            return float(v), np.asarray(g, dtype=np.float64)

        # Two proposals, the nearest maximum and the best of the grid; the
        # share of bins the full lattice indexes decides between them and
        # staying put (the objective alone has false maxima).
        delta, after = np.zeros(3), before
        share = share0 = indexed_share(basis, obs)
        for start in (
            np.zeros(3),
            np.asarray(G[int(np.argmax(coarse))], dtype=np.float64),
        ):
            res = minimize(
                f, start, jac=True, method="L-BFGS-B", options={"maxiter": 60}
            )
            got = indexed_share(
                basis, obs, turn=Rotation.from_rotvec(res.x).as_matrix()
            )
            if got > share:
                delta, after, share = res.x, float(-res.fun), got
        M = Rotation.from_rotvec(delta).as_matrix()
        for i in frames:
            out[i] = rot[i] @ M
        angles.append(float(np.degrees(np.linalg.norm(delta))))
        gains.append((before, after, share0, share))
        say(
            f"run {r}: turned by {angles[-1]:.2f} deg, score {1e3 * before:.2f} -> {1e3 * after:.2f}, "
            f"bins indexed {100 * share0:.0f}% -> {100 * share:.0f}%"
        )
    return out, angles, np.array(gains)


def has_events(merged):
    """Whether an images file carries time-of-flight events."""
    try:
        with h5py.File(merged, "r") as f:
            first = f["images"][0]
        with EventStore(merged) as store:
            store.image_events(0, first)
    except (KeyError, OSError, ValueError, IndexError):
        return False
    return True


def search_detector_shift(
    merged, instrument, wavelength, radial=0.0, axial=None, step=0.010, log=None, **kw
):
    """Translation of the detector against the sample from the lattice score.

    The cell search needs pixel directions good to well under a degree.  A
    sample 10 mm from the nominal centre turns them by 1.5 degrees at 0.4 m,
    and the lattice is lost.  The score of the best lattice
    (``find_cell(...).diagnostics["ladder_score"]``) falls off with the
    error, so it is scanned: a grid of ``step`` in x and z and half of it in
    y, then two rounds of half steps around the best.

    Args:
        radial, axial: [-] as for ``scaled_detectors``
        step: [m] grid step
        kw: passed to ``find_cell``
    Returns:
        (shift [m] (3,), score, {shift: score})
    """
    say = log or (lambda *_: None)

    def score(shift):
        with scaled_detectors(instrument, radial, axial, shift=shift):
            try:
                d = find_cell(
                    merged, instrument, wavelength=wavelength, **kw
                ).diagnostics
            except ValueError:
                return 0.0
        say(
            f"shift [mm] {np.round(1e3 * np.array(shift), 1).tolist()}: "
            f"score {1e3 * d['ladder_score']:.1f} (z {d['ladder_z']:.0f})"
        )
        return d["ladder_score"] if d["ladder_z"] >= 8 else 0.0

    tried = {}
    for x in (-step, 0.0, step):
        for y in (0.0, 0.5 * step):
            for z in (-step, 0.0, step):
                tried[(x, y, z)] = score((x, y, z))
    best = max(tried, key=tried.get)
    for h in (0.5 * step, 0.25 * step):
        for axis in range(3):
            for sign in (-1.0, 1.0):
                t = list(best)
                t[axis] += sign * h
                t = tuple(round(v, 5) for v in t)
                if t not in tried:
                    tried[t] = score(t)
        best = max(tried, key=tried.get)
    return np.array(best), tried[best], tried


def determine_cell(
    merged,
    instrument,
    wavelength=None,
    rounds=3,
    d_min=3.0,
    grid=None,
    x_min=None,
    volume=None,
    length=None,
    log=None,
):
    """Cell and per-run rotations with nothing given: the whole chain.

    With events (``wavelength`` None): ``align_runs`` for the rotation
    errors between runs, then ``find_cell`` on all runs.  Without
    (``wavelength`` the band): ``find_cell`` in Laue mode alternated with
    ``refine_run_rotations`` for ``rounds`` rounds, starting from the logged
    goniometer angles.

    Returns:
        (BlindCell, rotations (n_images, 3, 3))
    """
    say = log or (lambda *_: None)
    if wavelength is None:
        kw = {"d_min": d_min, "grid": grid or 768, "x_min": x_min or 12.0}
        runs = align_runs(merged, instrument, log=log, **kw)
        return find_cell(
            merged, instrument, rotations=runs.rotations, log=log, **kw
        ), runs.rotations
    kw = {
        "d_min": d_min,
        "grid": grid or 384,
        "x_min": x_min or 35.0,
        "wavelength": wavelength,
        "volume": volume,
        "length": length,
    }
    rotations, found, volumes = None, None, []
    for k in range(rounds + 1):
        found = find_cell(merged, instrument, rotations=rotations, **kw)
        d = found.diagnostics
        volumes.append(abs(np.linalg.det(found.basis)))
        say(
            f"round {k}: "
            + " ".join(f"{v:.2f}" for v in found.conventional_cell)
            + f"  {found.centring}, {found.lattice_group}; score {1e3 * d['ladder_score']:.1f}"
            f" (z {d['ladder_z']:.0f}); scale from the {d['scale_source']}"
        )
        if k == rounds:
            break
        rotations, angles, stats = refine_run_rotations(
            merged,
            instrument,
            found.basis,
            wavelength,
            rotations=rotations,
            d_min=d_min,
        )
        say(
            "   runs turned by [deg]: "
            + " ".join(f"{a:.2f}" for a in angles)
            + f";  bins indexed {100 * np.nanmedian(stats[:, 2]):.0f}% -> "
            f"{100 * np.nanmedian(stats[:, 3]):.0f}%"
        )
    if not found.diagnostics["scale_determined"] and len(volumes) > 2:
        # No scale in the data: hold the median volume of the rounds.
        later = volumes[1:]
        factor = (np.median(later) / volumes[-1]) ** (1.0 / 3.0)
        spread = (max(later) / min(later)) ** (1.0 / 3.0) - 1.0
        say(
            f"the band limits do not fix the scale: median volume of the rounds taken, "
            f"lengths spread {100 * spread:.0f}% between rounds"
        )
        diagnostics = dict(found.diagnostics, scale_spread=float(spread))
        found = BlindCell(
            diagnostics=diagnostics, **describe_lattice(factor * found.basis)
        )
    if rotations is None:
        with h5py.File(merged, "r") as f:
            rotations = np.array(Goniometer.from_h5_file(f).rotation, dtype=float)
    return found, rotations


def goniometer_angles_for(rotations, axes, start):
    """Motor angles that reproduce the given rotations.

    The goniometer has three axes, so any rotation can be written as three
    angles about them; they are found numerically from the nominal angles.
    This is how a correction measured as a rotation is handed to code that
    reads goniometer angles.

    Args:
        rotations: (n, 3, 3) target rotations
        axes: goniometer axes as stored with the images (direction and sense)
        start: (n, 3) nominal angles [degree]
    Returns:
        (angles [degree] (n, 3), largest remaining misfit [degree])
    """
    from scipy.optimize import least_squares

    from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix

    rotations, start = (
        np.asarray(rotations, dtype=float),
        np.asarray(start, dtype=float),
    )
    out = np.array(start, dtype=float)
    worst = 0.0
    cache = {}
    for i in range(len(rotations)):
        key = (rotations[i].round(9).tobytes(), start[i].round(9).tobytes())
        if key not in cache:

            def resid(a, target=rotations[i]):
                return (calc_goniometer_rotation_matrix(axes, a) - target).ravel()

            fit = least_squares(resid, start[i], xtol=1e-12, ftol=1e-14, gtol=1e-14)
            got = calc_goniometer_rotation_matrix(axes, fit.x)
            cache[key] = (fit.x, _rotation_angle(got @ rotations[i].T))
        out[i], miss = cache[key]
        worst = max(worst, miss)
    return out, worst


def write_aligned_images(merged, rotations, out):
    """An images file that links all data of ``merged`` and carries
    goniometer angles reproducing ``rotations`` (from ``align_runs``), so
    that every later step (cell search, refinement, prediction,
    integration) starts from the corrected rotations.

    Returns:
        largest misfit [degree] between the rotations asked for and those
        the written angles give
    """
    import os

    merged = os.path.abspath(merged)
    with h5py.File(merged, "r") as src:
        axes = src["goniometer/axes"][()]
        nominal = np.atleast_2d(src["goniometer/angles"][()])
        angles, worst = goniometer_angles_for(rotations, axes, nominal)
        with h5py.File(out, "w") as dst:
            for key in src:
                if key != "goniometer":
                    dst[key] = h5py.ExternalLink(merged, "/" + key)
            g = dst.create_group("goniometer")
            for key in src["goniometer"]:
                g[key] = angles if key == "angles" else src["goniometer"][key][()]
            g["angles_nominal"] = nominal
            for k, v in src.attrs.items():
                dst.attrs[k] = v
    return worst


def write_seed(found, merged, instrument, out):
    """Store a ``BlindCell`` as a solve-style file without point symmetry.

    The file carries the cell and orientation of ``found.setting_basis``
    (``sample/U``, ``sample/B``), the space group ``found.p1_symbol``, the
    wavelength band read off the events, the goniometer of the images
    file, and under ``blind/`` the basis and the lattice rotations in that
    indexing.  It seeds ``solve --refinement band --bootstrap``.
    """
    UB = np.linalg.inv(found.setting_basis).T  # q_sample = UB h
    U, B = np.linalg.qr(UB)
    sign = np.sign(np.diag(B))
    U, B = U * sign, (B.T * sign).T
    band = wavelength_band(merged, instrument)
    with h5py.File(merged, "r") as fm, h5py.File(out, "w") as f:
        f.attrs["instrument"] = instrument
        f.attrs["source"] = str(merged)
        for name, value in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"), cell_of(found.setting_basis)
        ):
            f[f"sample/{name}"] = float(value)
        f["sample/space_group"] = np.bytes_(found.p1_symbol.encode())
        f["sample/U"] = U
        f["sample/B"] = B
        f["instrument/wavelength"] = np.array(band)
        for key in (
            "goniometer/axes",
            "goniometer/angles",
            "goniometer/names",
            "bank_ids",
            "files",
            "file_offsets",
        ):
            if key in fm:
                fm.copy(fm[key], f, key)
        g = f.create_group("blind")
        g["direct_basis"] = found.setting_basis
        g["primitive_basis"] = found.basis
        g["lattice_operations"] = np.array(found.setting_operations)
        g.attrs["lattice_group"] = found.lattice_group
        g.attrs["centring"] = found.centring
    return band
