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
):
    """Unit cell, absolute scale and orientation from raw events.

    Args:
        merged: images file with pixel-sorted events and goniometer angles
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
    Returns:
        BlindCell
    """
    say = log or (lambda *_: None)
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
    if len(pos) < 3:
        raise ValueError(
            "Fewer than three peaks in the direct-space map: no lattice in the events."
        )
    A, good, diag = lattice_basis(pos, height, g, voxel)
    say(
        f"lattice: {int(good.sum())} of {len(pos)} map peaks indexed "
        f"({100 * diag['indexed_weight']:.0f}% of the peak weight), "
        f"rms misfit {diag['rms_misfit']:.2f} A, third-axis comb {diag['third_axis_score']:.2f}"
    )
    dens = np.array(density)
    nz = np.argwhere(dens > 0)
    q_pts = (nz + 0.5) * (2.0 * q_max / grid) - q_max
    A, before, after = refine_basis(A, q_pts, dens[dens > 0])
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


def align_runs(merged, instrument, d_min=3.0, grid=768, x_min=12.0, log=None):
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
                merged, instrument, d_min=d_min, grid=grid, x_min=x_min, frames=frames
            )
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
            M, ok = relate_lattices(basis, bases[first])
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
