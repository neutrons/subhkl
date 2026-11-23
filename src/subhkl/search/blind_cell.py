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


def reciprocal_histogram(merged, instrument, d_min, n):
    """Event density on an n^3 grid over |q_i| <= 1/d_min, sample frame.

    Args:
        merged: images file with pixel-sorted events and goniometer angles
        instrument: instrument name (nominal detector geometry and L1)
        d_min: [Angstrom] the grid spans |q_i| <= 1 / d_min
        n: [-] grid points per axis
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
    pos_cache = {}
    grid = jnp.zeros(n * n * n, dtype=jnp.float32)
    n_events = 0
    with EventStore(merged) as store:
        for i in range(len(images)):
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


def conventional_basis(A, max_obliquity=3.0):
    """Conventional basis of a reduced primitive basis, where one exists
    with three axes along symmetry axes of the highest order (cubic
    lattices: P, I, F).  Other lattices are returned unchanged.

    Returns:
        (C, T, multiplicity): C = A @ T (columns [Angstrom]); Miller indices
        transform as h_conventional = T^T h_primitive
    """
    cell = cell_of(A)
    ops = gemmi.find_lattice_symmetry(gemmi.UnitCell(*cell), "P", max_obliquity)
    axes = []
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
        vals, vecs = np.linalg.eig(w)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
        v = v / np.abs(v[np.abs(v) > 1e-6]).min()
        uvw = np.rint(v).astype(int)
        if tuple(uvw) < tuple(-uvw):
            uvw = -uvw
        axes.append((order, tuple(uvw)))
    identity = np.eye(3)
    if not axes:
        return A, identity, 1
    best = max(o for o, _ in axes)
    cand = sorted(
        {a for o, a in axes if o == best}, key=lambda u: np.linalg.norm(A @ np.array(u))
    )
    chosen = []
    for u in cand:
        trial = [*chosen, u]
        if np.linalg.matrix_rank(np.array(trial, dtype=float).T) == len(trial):
            chosen = trial
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        return A, identity, 1
    T = np.array(chosen, dtype=float).T
    if np.linalg.det(T) < 0:
        T[:, 2] *= -1
    return A @ T, T, round(abs(np.linalg.det(T)))


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
    """Result of ``find_cell``."""

    basis: np.ndarray  # reduced primitive basis, columns [Angstrom], sample frame
    cell: tuple  # a, b, c [Angstrom], alpha, beta, gamma [degree] of ``basis``
    symmetric_cell: tuple  # the same with the lattice symmetry imposed on the metric
    conventional_basis: np.ndarray
    conventional_cell: tuple
    to_conventional: np.ndarray  # T: h_conventional = T^T h_primitive
    multiplicity: int  # lattice points per conventional cell
    operations: list  # lattice rotations on primitive Miller indices
    lattice_group: str
    diagnostics: dict = field(default_factory=dict)


def find_cell(merged, instrument, d_min=3.0, grid=768, x_min=12.0, log=None):
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
    Returns:
        BlindCell
    """
    say = log or (lambda *_: None)
    counts, q_max, n_events = reciprocal_histogram(merged, instrument, d_min, grid)
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
    A = reduce_basis(A)
    cell = cell_of(A)
    mats, group = lattice_operations(cell)
    C, T, mult = conventional_basis(A)
    say(
        "primitive cell: a={:.2f} b={:.2f} c={:.2f} alpha={:.2f} beta={:.2f} gamma={:.2f}".format(
            *cell
        )
        + f"; lattice rotations: {len(mats)} ({group}); lattice points per conventional cell: {mult}"
    )
    return BlindCell(
        A, cell, symmetrized_cell(A, mats), C, cell_of(C), T, mult, mats, group, diag
    )


def write_seed(found, merged, instrument, out):
    """Store a ``BlindCell`` as a solve-style file in space group P 1.

    The file carries the cell and orientation (``sample/U``, ``sample/B``),
    the wavelength band read off the events, the goniometer of the images
    file, and under ``blind/`` the basis and the lattice rotations.  It
    seeds ``solve --refinement band --bootstrap``.
    """
    UB = np.linalg.inv(found.basis).T  # q_sample = UB h
    U, B = np.linalg.qr(UB)
    sign = np.sign(np.diag(B))
    U, B = U * sign, (B.T * sign).T
    band = wavelength_band(merged, instrument)
    with h5py.File(merged, "r") as fm, h5py.File(out, "w") as f:
        f.attrs["instrument"] = instrument
        f.attrs["source"] = str(merged)
        for name, value in zip(("a", "b", "c", "alpha", "beta", "gamma"), found.cell):
            f[f"sample/{name}"] = float(value)
        f["sample/space_group"] = np.bytes_(b"P 1")
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
        g["direct_basis"] = found.basis
        g["lattice_operations"] = np.array(found.operations)
        g["to_conventional"] = found.to_conventional
        g.attrs["lattice_group"] = found.lattice_group
        g.attrs["multiplicity"] = found.multiplicity
    return band
