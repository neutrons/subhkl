"""Prototype: unit cell and orientation from raw events, no peaks and no cell.

Every event of a time-of-flight Laue run has a full scattering vector
q = (k_f - k_i) / lambda.  Histogram all events in the sample frame, take
the 3D Fourier transform, and the direct lattice appears: with the
reciprocal-space density rho(q) = sum_h w_h delta(q - h), the transform
G(x) = sum_h w_h exp(2 pi i h.x) adds coherently exactly when x is a
direct-lattice vector (Poisson summation).  This is the lattice ladder's
objective with |q| kept instead of integrated out, so it yields absolute
cell lengths as well as directions.

The smooth incoherent background only contributes near x = 0 and needs no
subtraction; nothing is fitted and no reflection is located.

Usage:
    python blind_cell.py merged.h5 MANDI out.npz [--d-min 3.0] [--grid 384]
"""

import argparse
import time

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


def reciprocal_histogram(merged, instrument, d_min, n, frames=None, verbose=True):
    """Event density on an n^3 grid over |q_i| <= 1/d_min, sample frame."""
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
    pos_cache = {}
    grid = jnp.zeros(n * n * n, dtype=jnp.float32)
    n_events = 0
    use = range(len(images)) if frames is None else frames
    t0 = time.time()
    with EventStore(merged) as store:
        for i in use:
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
    grid = grid.reshape(n, n, n)
    if verbose:
        print(
            f"histogrammed {float(grid.sum()):.3g} of {n_events:.3g} events inside "
            f"|q_i| <= {q_max:.3f} 1/A in {time.time() - t0:.0f} s"
        )
    return grid, q_max


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
    +-half_extent [A] around the origin."""
    import jax.numpy as jnp

    g = jnp.fft.fftshift(jnp.abs(jnp.fft.fftn(density)))
    n = g.shape[0]
    half = min(n // 2, int(half_extent / voxel))
    c = n // 2
    return np.array(g[c - half : c + half, c - half : c + half, c - half : c + half])


def find_peaks(g, voxel, x_min=12.0, x_max=None, n_keep=400, size=5):
    """Local maxima of |G| away from the origin, strongest first.

    Returns positions [A] (refined to a fraction of a voxel) and heights.
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
    """Largest |G| within one voxel of each position x [A] (0 outside the map)."""
    n = g.shape[0]
    c = n // 2
    idx = np.rint(np.atleast_2d(x) / voxel).astype(int) + c
    out = np.zeros(len(idx))
    for j, (i0, i1, i2) in enumerate(idx):
        if 1 <= i0 < n - 1 and 1 <= i1 < n - 1 and 1 <= i2 < n - 1:
            out[j] = g[i0 - 1 : i0 + 2, i1 - 1 : i1 + 2, i2 - 1 : i2 + 2].max()
    return out


def lattice_basis(pos, height, g, voxel, tol=0.15, verbose=True):
    """A basis of the lattice the peaks lie on, columns [A].

    G is the Patterson function of the crystal: besides its origin peak
    repeated on every lattice vector it has weaker interatomic peaks, so a
    peak alone is not evidence of a lattice vector.  A lattice vector is
    one whose whole comb is occupied.

    - The first two axes are the two shortest independent peaks among the
      dominant ones (within a factor 2.5 of the strongest).
    - The third is the out-of-plane peak whose translates by the first two
      and whose multiples carry the most weight, scored on the map itself.
    - Every peak the basis then indexes refines it by least squares.
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
    A = np.stack([a1, a2, best], 1)
    for _ in range(3):
        coords = np.linalg.solve(A, pos.T).T
        nint = np.rint(coords)
        good = (np.abs(coords - nint).max(axis=1) < tol) & (
            np.abs(nint).sum(axis=1) > 0
        )
        N, X, W = nint[good].T, pos[good].T, height[good]
        A = (X * W) @ N.T @ np.linalg.inv((N * W) @ N.T)
    if verbose:
        rms = np.sqrt(np.mean(np.sum((X - A @ N) ** 2, axis=0)))
        print(
            f"lattice: third axis comb score {best_score / height.max():.3f} of the "
            f"strongest peak; {int(good.sum())} of {len(pos)} peaks indexed "
            f"({100 * height[good].sum() / height.sum():.0f}% of the peak weight), "
            f"rms misfit {rms:.2f} A"
        )
    return A, good


def refine_basis(A, q, w, extent=330.0, verbose=True):
    """Refine the basis on the data itself, off the voxel grid.

    The lattice Fourier sum F(x) = sum_i w_i cos(2 pi q_i . x) over the
    Bragg-excess voxels peaks at every direct-lattice vector.  The basis is
    moved to maximise the sum of F over all its lattice points within
    ``extent``: the many combinations pin down each axis far better than
    its own peak does, which matters for a long axis whose peak is weak.
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    jax.config.update("jax_enable_x64", True)
    r = range(-4, 5)
    n_all = np.array(
        [(i, j, k) for i in r for j in r for k in r if (i, j, k) > (0, 0, 0)]
    )
    n_use = n_all[np.linalg.norm(n_all @ A.T, axis=1) < extent]
    qj = jnp.array(q, dtype=jnp.float64)
    wj = jnp.array(w, dtype=jnp.float64)
    nj = jnp.array(n_use, dtype=jnp.float64)
    norm = float(np.sum(w)) * len(n_use)

    def neg_score(a_flat):
        x = nj @ a_flat.reshape(3, 3).T  # lattice points [A]
        phase = 2.0 * jnp.pi * (qj @ x.T)
        return -jnp.sum(wj[:, None] * jnp.cos(phase)) / norm

    fun = jax.jit(jax.value_and_grad(neg_score))

    def f(a_flat):
        v, g = fun(jnp.array(a_flat))
        return float(v), np.array(g, dtype=float)

    before = -f(A.ravel())[0]
    res = minimize(f, A.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    A_new = res.x.reshape(3, 3)
    if verbose:
        shift = np.linalg.norm(A_new - A, axis=0)
        print(
            f"refinement on {len(n_use)} lattice points: mean lattice Fourier sum "
            f"{before:.4f} -> {-res.fun:.4f} of its maximum; axes moved "
            f"{shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f} A"
        )
    return A_new


def cell_from_basis(A):
    a, b, c = (np.linalg.norm(A[:, i]) for i in range(3))
    ang = lambda u, v: np.degrees(
        np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
    )
    return a, b, c, ang(A[:, 1], A[:, 2]), ang(A[:, 0], A[:, 2]), ang(A[:, 0], A[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("instrument")
    ap.add_argument("out")
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--grid", type=int, default=768)
    args = ap.parse_args()

    grid, q_max = reciprocal_histogram(
        args.merged, args.instrument, args.d_min, args.grid
    )
    voxel = 1.0 / (2.0 * q_max)
    density = bragg_excess(grid)
    print(
        f"kept {float(density.sum()):.3g} counts above the local background "
        f"in {int((density > 0).sum())} voxels"
    )
    g = direct_space_map(density, voxel)
    print(
        f"direct-space map: voxel {voxel:.2f} A, half-extent {0.5 * g.shape[0] * voxel:.0f} A"
    )
    pos, height = find_peaks(g, voxel)
    print("strongest peaks (|x| [A], relative height):")
    for p, hgt in list(zip(pos, height))[:12]:
        print(f"   {np.linalg.norm(p):7.2f}  {hgt / height[0]:.3f}   {np.round(p, 1)}")
    A, good = lattice_basis(pos, height, g, voxel)
    print(
        "before refinement:  "
        + "a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *cell_from_basis(A)
        )
    )
    dens = np.array(density)
    nz = np.argwhere(dens > 0)
    q_pts = (nz + 0.5) * (2.0 * q_max / args.grid) - q_max
    A = refine_basis(A, q_pts, dens[dens > 0])
    cell = cell_from_basis(A)
    print(
        "primitive cell: "
        + "a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *cell
        )
        + f"  volume {abs(np.linalg.det(A)):.0f} A^3"
    )
    np.savez(
        args.out,
        A=A,
        peaks=pos,
        heights=height,
        indexed=good,
        cell=np.array(cell),
        q_max=q_max,
        voxel=voxel,
    )


if __name__ == "__main__":
    main()
