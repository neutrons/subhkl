"""Point group and space-group candidates without a known space group.

Two tests, both free of wavelength normalisation:

``laue_class``
    Symmetry mates h and M.h have the same d-spacing, so two observations
    of them share the scattering angle whenever they share the wavelength;
    the incident spectrum and the Lorentz factor are then the same for both
    and intensities integrated once in P 1 compare directly.  Every rotation
    of the lattice is tested on such matched pairs against two yardsticks
    from the same data: the same reflection measured twice, and unrelated
    reflections of the same d.

``space_group_candidates``
    Every space group compatible with the point group is a hypothesis about
    which reciprocal-lattice nodes are empty (centring, glide planes, screw
    axes).  The hypotheses are scored on counts at the nodes: raw events
    converted to fractional Miller indices (``event_node_counts``), nothing
    integrated or normalised.
"""

import itertools
from dataclasses import dataclass, field

import gemmi
import h5py
import numpy as np

from subhkl.config import reduction_settings
from subhkl.io.events import ANGSTROM_M_PER_US, EventStore

IDENTITY = np.eye(3, dtype=int)
_KEY_OFFSET = 512


def _key(hkl):
    hkl = np.asarray(hkl, dtype=np.int64)
    o = _KEY_OFFSET
    return ((hkl[:, 0] + o) * 2048 + (hkl[:, 1] + o)) * 2048 + (hkl[:, 2] + o)


def _robust_spread(z):
    """Robust standard deviation (from the interquartile range)."""
    if len(z) <= 30:
        return float("nan")
    return 0.7413 * float(np.diff(np.percentile(z, [25, 75]))[0])


def describe_operation(M):
    """Rotation order and axis (direct-lattice indices) of an integer matrix on hkl."""
    M = np.asarray(M)
    if np.array_equal(M, -IDENTITY):
        return "inversion (Friedel)"
    order = next(
        (
            n
            for n in (1, 2, 3, 4, 6)
            if np.array_equal(np.linalg.matrix_power(M, n), IDENTITY)
        ),
        None,
    )
    if order is None:
        return "not a rotation"
    if order == 1:
        return "identity"
    W = np.linalg.inv(M).T  # acts on direct-lattice coordinates
    w, v = np.linalg.eig(W)
    axis = np.real(v[:, np.argmin(np.abs(w - 1))])
    axis = axis / np.abs(axis[np.abs(axis) > 1e-6]).min()
    return f"{order}-fold about [{' '.join(str(round(x)) for x in axis)}]"


def close_group(mats, limit=48):
    """All products of the given integer matrices (the group they generate)."""
    group = {tuple(IDENTITY.ravel())}
    frontier = [IDENTITY]
    gens = [np.asarray(m, dtype=int) for m in mats]
    while frontier:
        new = []
        for a in frontier:
            for g in gens:
                p = a @ g
                k = tuple(p.ravel())
                if k not in group:
                    group.add(k)
                    new.append(p)
        if len(group) > limit:
            raise ValueError(
                "The operations do not generate a crystallographic point group."
            )
        frontier = new
    return [np.array(k).reshape(3, 3) for k in sorted(group)]


def _group_ops(mats):
    ops = []
    for M in mats:
        W = np.rint(np.linalg.inv(np.asarray(M, dtype=float)).T).astype(int)
        op = gemmi.Op()
        op.rot = (W * gemmi.Op.DEN).tolist()
        ops.append(op)
    return gemmi.find_spacegroup_by_ops(gemmi.GroupOps(ops))


class Observations:
    """Reflections integrated once in P 1, unscaled.

    Args:
        hkl: (N, 3) integer Miller indices
        intensity, sigma: (N,) intensity and its standard error
        wavelength: [Angstrom] (N,)
        d: [Angstrom] (N,) d-spacing
        group: (N,) integer label of the measurement (run and bank); a pair
            from the same label is the same spot and is not used
        match: [-] largest |log wavelength ratio| of a matched pair; None
            compares pairs at any wavelength, which is only valid for
            intensities that have been divided by a scale function
        min_isigi: [-] a pair counts as strong if either member exceeds this
        friedel_merged: the indices are only known up to inversion (as
            after mapping to the asymmetric unit of P 1); mates are then
            looked up up to inversion as well
    """

    def __init__(
        self,
        hkl,
        intensity,
        sigma,
        wavelength,
        d,
        group,
        match=0.02,
        min_isigi=5.0,
        friedel_merged=False,
    ):
        ok = np.asarray(sigma) > 0
        self.friedel_merged = friedel_merged
        self.hkl = np.asarray(hkl, dtype=np.int64)[ok]
        if friedel_merged:
            self.hkl = _friedel_unique(self.hkl)
        self.I = np.asarray(intensity, dtype=float)[ok]
        self.sig = np.asarray(sigma, dtype=float)[ok]
        self.lam = np.asarray(wavelength, dtype=float)[ok]
        self.d = np.asarray(d, dtype=float)[ok]
        self.group = np.asarray(group)[ok]
        self.match, self.min_isigi = match, min_isigi
        self.order = np.argsort(_key(self.hkl), kind="stable")
        self.keys = _key(self.hkl)[self.order]

    @classmethod
    def from_file(cls, filename, d_min=2.5, **kwargs):
        """Read an integrator output file; keep reflections with d > d_min."""
        with h5py.File(filename, "r") as f:
            p = {
                k: f["peaks"][k][()]
                for k in (
                    "h",
                    "k",
                    "l",
                    "intensity",
                    "sigma",
                    "lambda",
                    "run_index",
                    "bank",
                )
            }
            B = f["sample/B"][()]
        hkl = np.stack([p["h"], p["k"], p["l"]], 1).astype(np.int64)
        d = 1.0 / np.maximum(np.linalg.norm(hkl @ B.T, axis=1), 1e-12)
        keep = d > d_min
        group = p["run_index"].astype(np.int64) * 100000 + p["bank"].astype(np.int64)
        return cls(
            hkl[keep],
            p["intensity"][keep],
            p["sigma"][keep],
            p["lambda"][keep],
            d[keep],
            group[keep],
            **kwargs,
        )

    def mate_pairs(self, M, max_pairs=300000):
        """Pairs (i, j): observation j is of M.h_i, at matched wavelength,
        from a different measurement."""
        mate = self.hkl @ np.asarray(M).T
        want = _key(_friedel_unique(mate) if self.friedel_merged else mate)
        lo = np.searchsorted(self.keys, want, side="left")
        hi = np.searchsorted(self.keys, want, side="right")
        cnt = hi - lo
        i_all, j_all = [], []
        for off in range(int(cnt.max()) if len(cnt) else 0):
            m = cnt > off
            i_all.append(np.flatnonzero(m))
            j_all.append(self.order[lo[m] + off])
        if not i_all:
            return np.empty(0, int), np.empty(0, int)
        i, j = np.concatenate(i_all), np.concatenate(j_all)
        keep = (i != j) & (self.group[i] != self.group[j])
        if self.match is not None:
            keep &= np.abs(np.log(self.lam[i] / self.lam[j])) < self.match
        i, j = i[keep], j[keep]
        if len(i) > max_pairs:
            s = np.random.default_rng(0).choice(len(i), max_pairs, replace=False)
            i, j = i[s], j[s]
        return i, j

    def random_pairs(self, n=600000):
        """Unrelated reflections of the same d at matched wavelength."""
        rng = np.random.default_rng(1)
        i = rng.choice(len(self.I), n)
        shell = np.argsort(self.d)
        pos = np.searchsorted(self.d[shell], self.d[i])
        j = shell[np.clip(pos + rng.integers(-50, 51, n), 0, len(self.I) - 1)]
        keep = np.any(self.hkl[i] != self.hkl[j], axis=1)
        if self.match is not None:
            keep &= np.abs(np.log(self.lam[i] / self.lam[j])) < self.match
        return i[keep], j[keep]

    @classmethod
    def scaled(cls, hkl, intensity, sigma, scale, d, group, d_min=3.0, **kwargs):
        """Experimental.  Observations divided by a learned scale function.

        Only as good as the scale: a scale model with many inputs and few
        observations per reflection can follow the observations themselves,
        and the comparison is then not independent of the merge.

        For data without matched-wavelength repeats: a merge without point
        symmetry learns the scale of every observation (incident spectrum,
        Lorentz factor, absorption, exposure) as a function of its
        metadata.  Dividing each observed intensity by its scale makes
        observations at different wavelengths comparable while keeping
        them independent measurements; no merged amplitude enters, so the
        prior of the merge does not bias the comparison.

        Args:
            hkl: (N, 3) Miller indices, Friedel mates not distinguished
            intensity, sigma: (N,) observed intensity and standard error
            scale: (N,) scale of each observation (intensity = scale x F^2)
            d: [Angstrom] (N,) d-spacings
            group: (N,) label of the image the observation is from
            d_min: [Angstrom] observations beyond this resolution are left out
        """
        scale = np.asarray(scale, dtype=float)
        ok = np.isfinite(scale) & (scale > 0.05 * np.nanmedian(scale[scale > 0]))
        ok &= (np.asarray(sigma) > 0) & (np.asarray(d) > d_min)
        return cls(
            np.asarray(hkl)[ok],
            np.asarray(intensity, dtype=float)[ok] / scale[ok],
            np.asarray(sigma, dtype=float)[ok] / scale[ok],
            np.ones(int(ok.sum())),
            np.asarray(d)[ok],
            np.asarray(group)[ok],
            match=None,
            friedel_merged=True,
            **kwargs,
        )

    def statistic(self, i, j):
        """(pairs, strong pairs, spread of (I1 - I2) / sigma over the strong pairs)."""
        strong = (self.I[i] / self.sig[i] > self.min_isigi) | (
            self.I[j] / self.sig[j] > self.min_isigi
        )
        z = (self.I[i] - self.I[j]) / np.sqrt(self.sig[i] ** 2 + self.sig[j] ** 2)
        return len(i), int(strong.sum()), _robust_spread(z[strong])


@dataclass
class LaueResult:
    """Result of ``laue_class``."""

    rows: list  # one dict per tested operation: name, matrix, pairs, strong, spread, verdict
    same: float  # spread for the same reflection measured twice
    unrelated: float  # spread for unrelated reflections of the same d
    cut: float  # acceptance level
    rotations: list  # accepted rotations and everything they generate (on hkl)
    point_group: str  # of the rotations, e.g. "32"
    laue: str  # e.g. "-3m"
    symmorphic: str  # space group of the rotations without translations, e.g. "P 3 2 1"
    conflicts: list = field(
        default_factory=list
    )  # rejected operations the accepted ones imply
    friedel: float | None = None  # spread for inversion mates (a check, not an input)


def laue_class(obs, operations, min_strong=100, level=0.5, min_contrast=2.0):
    """Which rotations of the lattice are symmetries of the intensities.

    Args:
        obs: Observations
        operations: integer matrices M on Miller indices (h' = M h), the
            rotations of the lattice (see ``blind_cell.lattice_operations``)
        min_strong: [-] fewest strong pairs for a verdict
        level: [-] acceptance level between the two yardsticks (0 = same
            reflection measured twice, 1 = unrelated reflections)
        min_contrast: [-] smallest ratio of the two yardsticks for a
            verdict.  If repeated measurements of one reflection disagree
            almost as much as unrelated reflections do, the intensities
            (or their scaling) cannot carry a symmetry test.
    Returns:
        LaueResult.  Rotations that could not be tested but follow from the
        accepted ones are included; a rejected rotation that follows from
        accepted ones is listed under ``conflicts``.
    """
    _n, _s, same = obs.statistic(*obs.mate_pairs(IDENTITY))
    _n, _s, unrelated = obs.statistic(*obs.random_pairs())
    if not (np.isfinite(same) and np.isfinite(unrelated)) or unrelated <= same:
        raise ValueError(
            "Too few repeated strong reflections at matched wavelength to set "
            "the scale of the symmetry test (need redundancy across runs or banks)."
        )
    if unrelated < min_contrast * same:
        raise ValueError(
            f"Repeated measurements of one reflection disagree almost as much "
            f"as unrelated reflections do (spread {same:.2f} against "
            f"{unrelated:.2f}): these intensities cannot decide the symmetry."
        )
    cut = same + level * (unrelated - same)
    rows, accepted, rejected = [], [], []
    for M in operations:
        M = np.asarray(M, dtype=int)
        if np.array_equal(M, IDENTITY):
            continue
        n, s, val = obs.statistic(*obs.mate_pairs(M))
        if s < min_strong or not np.isfinite(val):
            verdict = "too few pairs"
        elif val < cut:
            verdict = "symmetry"
            accepted.append(M)
        else:
            verdict = "not a symmetry"
            rejected.append(M)
        rows.append(
            {
                "name": describe_operation(M),
                "matrix": M,
                "pairs": n,
                "strong": s,
                "spread": val,
                "verdict": verdict,
            }
        )
    rotations = close_group(accepted)
    have = {tuple(m.ravel()) for m in rotations}
    conflicts = [describe_operation(M) for M in rejected if tuple(M.ravel()) in have]
    for row in rows:
        if row["verdict"] == "too few pairs" and tuple(row["matrix"].ravel()) in have:
            row["verdict"] = "symmetry (implied)"
    _n, s_f, friedel = obs.statistic(*obs.mate_pairs(-IDENTITY))
    sg = _group_ops(rotations)
    return LaueResult(
        rows=rows,
        same=same,
        unrelated=unrelated,
        cut=cut,
        rotations=rotations,
        point_group=sg.point_group_hm() if sg else "unknown",
        laue=sg.laue_str() if sg else "unknown",
        symmorphic=sg.hm if sg else "unknown",
        conflicts=conflicts,
        friedel=friedel if s_f >= min_strong else None,
    )


def laue_class_presence(
    obs,
    operations,
    ratio=0.1,
    min_isigi=10.0,
    min_pairs=100,
    level=0.5,
    min_contrast=3.0,
):
    """Which lattice rotations are symmetries, by presence and absence alone.

    Null hypothesis: M is a symmetry.  Then the mate of a strong reflection
    is present too, whatever positive factors (incident spectrum, Lorentz
    factor, exposure, absorption) separate the two measurements, as long as
    those factors stay well within 1 / ``ratio``.  The only evidence counted
    against a rotation is a strong reflection whose mate is absent: below
    ``ratio`` of it, although it was measured well enough to see that much.

    The test needs no wavelength matching and no normalisation, and it does
    not rely on the standard errors being right, only on their telling
    detectable from undetectable.  It is the counterpart, for the point
    group, of testing systematic absences for the space group.

    Two yardsticks come from the same data: the same reflection measured
    twice (what a symmetry looks like) and unrelated reflections of the
    same d (what a non-symmetry looks like).  If repeats of one reflection
    are themselves often "absent", the scale factors are not bounded as
    assumed (a quasi-Laue band whose spectrum varies tenfold, a decaying
    crystal) and no verdict is given.

    Args:
        obs: Observations of an integration without point symmetry; pass
            ``match=None`` when building them so that pairs at any
            wavelength are used
        operations: integer matrices M on Miller indices, the lattice rotations
        ratio: [-] a mate below this fraction of the strong member is absent
        min_isigi: [-] significance of the strong member
        min_pairs: [-] fewest usable pairs for a verdict on a rotation
        level: [-] acceptance level between the two yardsticks
        min_contrast: [-] smallest ratio of the yardsticks for any verdict
    Returns:
        LaueResult; ``spread`` in its rows, ``same``, ``unrelated`` and
        ``cut`` are shares of absent mates here (lower = more alike), and
        ``friedel`` is that share for inversion mates
    """

    def share(i, j):
        z_i, z_j = obs.I[i] / obs.sig[i], obs.I[j] / obs.sig[j]
        first = z_i >= z_j
        s = np.where(first, i, j)
        w = np.where(first, j, i)
        usable = (obs.I[s] / obs.sig[s] > min_isigi) & (
            obs.sig[w] < ratio * obs.I[s] / 3.0
        )
        s, w = s[usable], w[usable]
        if len(s) == 0:
            return 0, float("nan")
        return len(s), float(np.mean(obs.I[w] < ratio * obs.I[s]))

    n_same, same = share(*obs.mate_pairs(IDENTITY))
    n_unrel, unrelated = share(*obs.random_pairs())
    if n_same < min_pairs or n_unrel < min_pairs:
        raise ValueError(
            "Too few repeated strong reflections to set the scale of the "
            "presence test (need redundancy across runs or banks)."
        )
    if unrelated < min_contrast * max(same, 0.01):
        raise ValueError(
            f"Repeated measurements of one reflection are absent almost as often "
            f"as unrelated reflections ({100 * same:.0f}% against {100 * unrelated:.0f}%): "
            f"the scale between measurements is not bounded, and presence cannot "
            f"decide the symmetry."
        )
    cut = same + level * (unrelated - same)
    rows, accepted, rejected = [], [], []
    for M in operations:
        M = np.asarray(M, dtype=int)
        if np.array_equal(M, IDENTITY):
            continue
        n, val = share(*obs.mate_pairs(M))
        if n < min_pairs:
            verdict = "too few pairs"
        elif val < cut:
            verdict = "symmetry"
            accepted.append(M)
        else:
            verdict = "not a symmetry"
            rejected.append(M)
        rows.append(
            {
                "name": describe_operation(M),
                "matrix": M,
                "pairs": n,
                "strong": n,
                "spread": val,
                "verdict": verdict,
            }
        )
    rotations = close_group(accepted)
    have = {tuple(m.ravel()) for m in rotations}
    conflicts = [describe_operation(M) for M in rejected if tuple(M.ravel()) in have]
    for row in rows:
        if row["verdict"] == "too few pairs" and tuple(row["matrix"].ravel()) in have:
            row["verdict"] = "symmetry (implied)"
    n_f, friedel = share(*obs.mate_pairs(-IDENTITY))
    sg = _group_ops(rotations)
    return LaueResult(
        rows=rows,
        same=same,
        unrelated=unrelated,
        cut=cut,
        rotations=rotations,
        point_group=sg.point_group_hm() if sg else "unknown",
        laue=sg.laue_str() if sg else "unknown",
        symmorphic=sg.hm if sg else "unknown",
        conflicts=conflicts,
        friedel=friedel if n_f >= min_pairs else None,
    )


# ---------------------------------------------------------------------------
# Counts at reciprocal-lattice nodes, from raw events
# ---------------------------------------------------------------------------

R_IN = 0.18  # [-] a node's signal sphere, fractional indices
R_OUT = (0.32, 0.5)  # [-] its background shell


@dataclass
class NodeCounts:
    """Net counts at integer nodes of reciprocal space.

    ``net``, ``err`` and ``covered`` are dense arrays over the box
    |h_i| <= hmax_i, indexed by h + hmax.
    """

    hmax: np.ndarray
    net: np.ndarray
    err: np.ndarray
    covered: np.ndarray
    basis: np.ndarray  # direct basis the indices refer to, columns [Angstrom]
    d_min: float
    events: int = 0

    @property
    def grid(self):
        return np.stack(
            np.meshgrid(*[np.arange(-m, m + 1) for m in self.hmax], indexing="ij"), -1
        ).reshape(-1, 3)


def event_node_counts(
    events_file, instrument, basis, rotations, d_min, pixel_positions=None
):
    """Counts near every reciprocal-lattice node, above the level between nodes.

    Args:
        events_file: images file with pixel-sorted events
        instrument: instrument name (L1; nominal pixel positions unless given)
        basis: (3, 3) direct basis, columns [Angstrom], in the sample frame
        rotations: (n_images, 3, 3) goniometer rotation of each image
        d_min: [Angstrom] nodes and events beyond this resolution are left out
        pixel_positions: optional callable (bank, shape) -> (H*W, 3) [m]
    Returns:
        NodeCounts
    """
    if pixel_positions is None:
        from subhkl.search.blind_cell import pixel_positions as nominal

        def pixel_positions(bank, shape):
            return nominal(instrument, bank, shape)

    l1 = float(reduction_settings[instrument]["L1"])
    basis = np.asarray(basis, dtype=float)
    hmax = np.ceil(np.linalg.norm(basis, axis=0) / d_min).astype(int) + 1
    shape = tuple(2 * hmax + 1)
    size = int(np.prod(shape))
    sig = np.zeros(size)
    bkg = np.zeros(size)
    with h5py.File(events_file, "r") as f:
        images = f["images"][()]
        bank_ids = f["bank_ids"][()]
    pos_cache = {}
    n_events = 0
    with EventStore(events_file) as store:
        for i in range(len(images)):
            tof, off = store.image_events(i, images[i])
            if len(tof) == 0:
                continue
            bank = int(bank_ids[i])
            if bank not in pos_cache:
                pos_cache[bank] = np.asarray(
                    pixel_positions(bank, images[i].shape), dtype=np.float64
                )
            xyz = pos_cache[bank][np.repeat(np.arange(images[i].size), np.diff(off))]
            l2 = np.linalg.norm(xyz, axis=1)
            lam = ANGSTROM_M_PER_US * tof / (l1 + l2)
            q_lab = (xyz / l2[:, None] - np.array([0.0, 0.0, 1.0])) / lam[:, None]
            q_lab = q_lab[np.linalg.norm(q_lab, axis=1) <= 1.0 / d_min]
            h = q_lab @ (rotations[i] @ basis)  # h_j = a_j . (R^T q_lab)
            node = np.rint(h).astype(np.int64)
            dist = np.linalg.norm(h - node, axis=1)
            ok = np.all(np.abs(node) <= hmax, axis=1)
            flat = np.ravel_multi_index((node[ok] + hmax).T, shape)
            sig += np.bincount(flat[dist[ok] < R_IN], minlength=size)
            in_shell = (dist[ok] > R_OUT[0]) & (dist[ok] < R_OUT[1])
            bkg += np.bincount(flat[in_shell], minlength=size)
            n_events += len(h)
    ratio = R_IN**3 / (R_OUT[1] ** 3 - R_OUT[0] ** 3)
    net = (sig - ratio * bkg).reshape(shape)
    err = np.sqrt(sig + ratio**2 * bkg + 1.0).reshape(shape)
    covered = ((sig + bkg) >= 5).reshape(shape)
    covered[tuple(hmax)] = False  # the origin
    return NodeCounts(hmax, net, err, covered, basis, float(d_min), n_events)


# ---------------------------------------------------------------------------
# Space-group hypotheses
# ---------------------------------------------------------------------------


def _signed_permutations():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3), dtype=int)
            for row, (col, s) in enumerate(zip(perm, signs)):
                P[row, col] = s
            if round(np.linalg.det(P)) == 1:
                out.append(P)
    return out


def _laue_set(mats):
    return frozenset(
        tuple((s * np.asarray(m, dtype=int)).ravel()) for m in mats for s in (1, -1)
    )


def _settings(sohncke):
    """Every tabulated space-group setting (reference and others, such as
    I 1 2 1 or P 21 21 2), one entry per distinct symbol."""
    seen = set()
    for sg in gemmi.spacegroup_table():
        name = sg.xhm()
        if name in seen or (sohncke and not sg.is_sohncke()):
            continue
        seen.add(name)
        yield sg


def _hkl_rotations(sg):
    out = []
    for op in sg.operations().sym_ops:
        w = np.array(op.rot, dtype=float) / gemmi.Op.DEN
        out.append(np.rint(np.linalg.inv(w).T).astype(int))
    return out


def _settings_from_rotations(rotations, sohncke):
    """(space group, P): tabulated settings whose Laue group, after the
    change of indices h_setting = P h, is the one the rotations generate."""
    target = _laue_set(close_group(rotations))
    perms = _signed_permutations()
    for sg in _settings(sohncke):
        ref = _hkl_rotations(sg)
        if len(_laue_set(ref)) != len(target):
            continue
        for P in perms:
            Pi = np.rint(np.linalg.inv(P)).astype(int)
            if _laue_set([Pi @ m @ P for m in ref]) == target:
                yield sg, P


def _settings_from_laue(laue, basis, sohncke):
    """Fallback when only the name of the Laue class is known: reference
    settings of that class, in the axis orders that leave the cell unchanged."""
    from subhkl.search.blind_cell import cell_of

    base = np.array(cell_of(basis))
    keep = []
    for perm in itertools.permutations(range(3)):
        permuted = np.array(cell_of(basis[:, list(perm)]))
        if np.allclose(permuted[:3], base[:3], rtol=0.01) and np.allclose(
            np.cos(np.radians(permuted[3:])), np.cos(np.radians(base[3:])), atol=0.02
        ):
            P = np.zeros((3, 3), dtype=int)
            for row, col in enumerate(perm):
                P[row, col] = 1
            keep.append(P)
    for sg in _settings(sohncke):
        if sg.is_reference_setting() and sg.laue_str() == laue:
            for P in keep:
                yield sg, P


def _axes_label(P):
    names = []
    for row in np.asarray(P):
        j = int(np.argmax(np.abs(row)))
        names.append(("-" if row[j] < 0 else "") + "abc"[j])
    return ",".join(names)


def space_group_candidates(
    nodes,
    rotations=None,
    laue=None,
    sohncke=False,
    rel_res=0.012,
    threshold=0.15,
    min_nodes=3,
):
    """Score every space group compatible with the point symmetry.

    How a hypothesis G is judged:

    - The nodes the candidates forbid are split into elementary classes:
      sets of nodes that the same candidates forbid.  A class is one
      reflection condition, or the intersection of a few.
    - A scattering vector is known to a relative precision ``rel_res``, so
      a node of index n is blurred by about rel_res * n.  A forbidden node
      is "testable" under G only if every node inside that blur is
      forbidden under G too; otherwise an allowed neighbour leaks into it.
    - For each class G forbids, the net counts of its testable nodes are
      compared with never-forbidden nodes of the same resolution.  G is
      consistent if no class carries more than ``threshold`` of that.

    Args:
        nodes: NodeCounts
        rotations: integer matrices on the Miller indices of ``nodes`` that
            are symmetries (``LaueResult.rotations``).  Candidates are the
            tabulated space-group settings (reference and others, such as
            I 1 2 1) whose Laue group equals the one these generate, in
            every axis assignment that makes it so.
        laue: name of the Laue class (e.g. "m-3m"), used when ``rotations``
            is None; axis orders that leave the cell unchanged are tried
        sohncke: only groups a chiral molecule can crystallise in
        rel_res: [-] relative precision of a scattering vector
        threshold: [-] largest share of the allowed level a forbidden class
            may carry
        min_nodes: [-] fewest testable nodes for a class to be judged
    Returns:
        list of dicts, consistent hypotheses first and among them the most
        restrictive first: ``space_groups`` (names sharing the absence
        pattern), ``consistent``, ``forbidden_nodes``, ``tested_nodes``,
        ``untestable_nodes``, ``worst_class_fraction``, ``classes``
        (per-class evidence)
    """
    from scipy.ndimage import maximum_filter

    if rotations is None and laue is None:
        raise ValueError("Give the accepted rotations or the name of the Laue class.")
    grid = nodes.grid
    shape = nodes.net.shape
    recip = np.linalg.inv(nodes.basis)  # rows are reciprocal vectors
    d_all = 1.0 / np.maximum(np.linalg.norm(grid @ recip, axis=1), 1e-9)
    cov = nodes.covered.ravel() & (d_all >= nodes.d_min)
    netf, errf = nodes.net.ravel(), nodes.err.ravel()
    cov_idx = np.flatnonzero(cov)
    if len(cov_idx) == 0:
        raise ValueError("No reciprocal-lattice node has data.")

    if rotations is not None:
        settings = _settings_from_rotations(rotations, sohncke)
    else:
        settings = _settings_from_laue(laue, nodes.basis, sohncke)
    hyp = {}
    for sg, P in settings:
        forb = np.asarray(
            sg.operations().systematic_absences((grid @ P.T).astype(np.int32)),
            dtype=bool,
        )
        entry = hyp.setdefault(
            forb[cov].tobytes(), {"forb": forb, "direct": [], "permuted": []}
        )
        name = sg.xhm()
        if np.array_equal(P, IDENTITY):
            if name not in entry["direct"]:
                entry["direct"].append(name)
        elif not any(n.startswith(name + " (") for n in entry["permuted"]):
            entry["permuted"].append(f"{name} (axes {_axes_label(P)})")
    for entry in hyp.values():
        # A symbol that applies to the indices as they are is preferred; the
        # same group with its axes renamed is listed only when there is none.
        entry["names"] = entry["direct"] or entry["permuted"][:4]
    hyps = list(hyp.values())
    if not hyps:
        raise ValueError(
            "No space group matches the given point symmetry in this cell."
        )

    # elementary classes: nodes forbidden by the same set of hypotheses
    F = np.stack([h["forb"][cov] for h in hyps], 1)
    sig_keys, sig_inv = np.unique(F, axis=0, return_inverse=True)
    sig_inv = sig_inv.ravel()

    # reference level: never-forbidden nodes, by resolution
    never = ~F.any(axis=1)
    if not never.any():
        raise ValueError("Every node with data is forbidden by some candidate.")
    edges = np.quantile(d_all[cov_idx[never]], np.linspace(0, 1, 9))
    edges[0], edges[-1] = 0.0, np.inf
    which = np.clip(np.searchsorted(edges, d_all[cov_idx], side="right") - 1, 0, 7)
    overall = netf[cov_idx[never]].mean()
    ref_bin = np.array(
        [
            netf[cov_idx[never & (which == b)]].mean()
            if (never & (which == b)).any()
            else overall
            for b in range(8)
        ]
    )
    expect = ref_bin[which]

    blur = np.rint(2.5 * rel_res * np.abs(grid)).astype(int)
    blur_keys, blur_inv = np.unique(blur, axis=0, return_inverse=True)
    blur_inv = blur_inv.ravel()

    results = []
    for h in hyps:
        allowed3 = ~h["forb"].reshape(shape)
        testable = np.zeros(len(grid), dtype=bool)
        for j, b in enumerate(blur_keys):
            m = (blur_inv == j) & h["forb"]
            if not m.any():
                continue
            if not b.any():
                testable[m] = True
                continue
            near_allowed = maximum_filter(
                allowed3.astype(np.uint8),
                size=tuple(2 * b + 1),
                mode="constant",
                cval=1,
            )
            testable[m] = near_allowed.ravel()[m] == 0
        t_cov = testable[cov_idx]
        f_cov = h["forb"][cov_idx]
        worst, n_tested, n_untested, classes = 0.0, 0, 0, []
        for s in np.unique(sig_inv[f_cov]):
            cls = f_cov & (sig_inv == s)
            use = cls & t_cov
            top = grid[cov_idx[cls]][np.argsort(-netf[cov_idx[cls]])[:3]]
            entry = {
                "nodes": int(cls.sum()),
                "testable": int(use.sum()),
                "examples": [[int(v) for v in e] for e in top],
                "also_forbidden_by": int(sig_keys[s].sum()),
            }
            classes.append(entry)
            if use.sum() < min_nodes:
                n_untested += int(cls.sum())
                continue
            n_tested += int(use.sum())
            total = float(netf[cov_idx[use]].sum())
            ratio = total / max(float(expect[use].sum()), 1e-9)
            entry.update(
                net_counts=total,
                sigma=float(np.sqrt((errf[cov_idx[use]] ** 2).sum())),
                expected_if_present=float(expect[use].sum()),
                fraction_of_allowed=float(ratio),
            )
            worst = max(worst, ratio)
        results.append(
            {
                "space_groups": h["names"],
                "consistent": bool(worst < threshold),
                "forbidden_nodes": int(f_cov.sum()),
                "tested_nodes": n_tested,
                "untestable_nodes": n_untested,
                "worst_class_fraction": float(worst),
                "classes": classes,
            }
        )
    results.sort(
        key=lambda r: (not r["consistent"], -r["tested_nodes"], -r["forbidden_nodes"])
    )
    return results


def best_candidates(results):
    """The most restrictive consistent hypotheses (ties included)."""
    good = [r for r in results if r["consistent"]]
    if not good:
        return []
    top = max(r["tested_nodes"] for r in good)
    return [r for r in good if r["tested_nodes"] == top]


# ---------------------------------------------------------------------------
# The same two tests on intensities merged without point symmetry
# ---------------------------------------------------------------------------


def _friedel_unique(hkl):
    hkl = np.asarray(hkl, dtype=np.int64)
    flip = (
        (hkl[:, 2] < 0)
        | ((hkl[:, 2] == 0) & (hkl[:, 1] < 0))
        | ((hkl[:, 2] == 0) & (hkl[:, 1] == 0) & (hkl[:, 0] < 0))
    )
    return np.where(flip[:, None], -hkl, hkl)


def laue_class_merged(
    hkl,
    amplitude,
    d,
    operations,
    cc_half=None,
    d_min=3.0,
    level=0.5,
    min_reference=0.15,
):
    """Experimental.  Which lattice rotations are symmetries, from a merge in P 1.

    On the quasi-Laue data tried so far the outcome depended on how the
    merge was set up (see examples/blind_symmetry_quasi_laue.py); treat a
    result as a hypothesis.

    For data without matched-wavelength repeats (a quasi-Laue rotation
    scan), merge once without point symmetry so that the merge does the
    wavelength normalisation, then correlate every reflection with its
    mate under each rotation.  A symmetry correlates as well as two
    independent measurements of one reflection; a non-symmetry correlates
    like unrelated reflections of the same d, that is, not at all.

    Args:
        hkl: (N, 3) Miller indices of the merged reflections (Friedel mates
            merged, either hemisphere)
        amplitude: (N,) merged structure-factor amplitudes or intensities
        d: [Angstrom] (N,) d-spacings
        operations: integer matrices M on Miller indices, the lattice rotations
        cc_half: [-] half-dataset correlation of the merge in the same
            resolution range, if known; it sets what a true symmetry should
            reach.  Without it the best rotation takes that role, provided
            it stands clearly above unrelated reflections.
        d_min: [Angstrom] reflections beyond this resolution are left out
        level: [-] acceptance level between unrelated (0) and the reference (1)
        min_reference: [-] smallest correlation a true symmetry must be
            expected to reach, above the unrelated level, for a verdict
    Returns:
        LaueResult; ``spread`` in its rows, ``same``, ``unrelated`` and
        ``cut`` are correlation coefficients here (higher = more alike)
    """
    keep = np.isfinite(amplitude) & (np.asarray(d) > d_min)
    hkl = _friedel_unique(np.asarray(hkl)[keep])
    f = np.asarray(amplitude, dtype=float)[keep]
    d = np.asarray(d, dtype=float)[keep]
    if len(f) < 100:
        raise ValueError("Too few merged reflections for the symmetry test.")
    keys = _key(hkl)
    order = np.argsort(keys)
    ks = keys[order]
    rng = np.random.default_rng(0)
    shell = np.argsort(d)
    pos = np.searchsorted(d[shell], d)
    other = shell[np.clip(pos + rng.integers(5, 40, len(d)), 0, len(d) - 1)]
    unrelated = float(np.corrcoef(f, f[other])[0, 1])
    rows = []
    for M in operations:
        M = np.asarray(M, dtype=int)
        if np.array_equal(M, IDENTITY):
            continue
        want = _key(_friedel_unique(hkl @ M.T))
        p = np.clip(np.searchsorted(ks, want), 0, len(ks) - 1)
        hit = (ks[p] == want) & (order[p] != np.arange(len(f)))
        n = int(hit.sum())
        cc = (
            float(np.corrcoef(f[hit], f[order[p[hit]]])[0, 1])
            if n > 30
            else float("nan")
        )
        rows.append(
            {
                "name": describe_operation(M),
                "matrix": M,
                "pairs": n,
                "strong": n,
                "spread": cc,
            }
        )
    tested = [r for r in rows if np.isfinite(r["spread"])]
    if cc_half is not None:
        reference = 2.0 * cc_half / (1.0 + cc_half)
    else:
        best = max(tested, key=lambda r: r["spread"], default=None)
        clear = (
            best is not None
            and (best["spread"] - unrelated) * np.sqrt(best["pairs"]) > 8.0
        )
        reference = best["spread"] if clear else float("inf")
    if cc_half is not None and reference - unrelated < min_reference:
        raise ValueError(
            f"The merge carries too little signal for the symmetry test "
            f"(half-dataset correlation {cc_half:.3f})."
        )
    cut = unrelated + level * (reference - unrelated)
    accepted, rejected = [], []
    for r in rows:
        if not np.isfinite(r["spread"]):
            r["verdict"] = "too few pairs"
        elif r["spread"] > cut:
            r["verdict"] = "symmetry"
            accepted.append(r["matrix"])
        else:
            r["verdict"] = "not a symmetry"
            rejected.append(r["matrix"])
    rotations = close_group(accepted)
    have = {tuple(m.ravel()) for m in rotations}
    conflicts = [describe_operation(M) for M in rejected if tuple(M.ravel()) in have]
    for r in rows:
        if r["verdict"] == "too few pairs" and tuple(r["matrix"].ravel()) in have:
            r["verdict"] = "symmetry (implied)"
    sg = _group_ops(rotations)
    return LaueResult(
        rows=rows,
        same=float(reference),
        unrelated=unrelated,
        cut=float(cut),
        rotations=rotations,
        point_group=sg.point_group_hm() if sg else "unknown",
        laue=sg.laue_str() if sg else "unknown",
        symmorphic=sg.hm if sg else "unknown",
        conflicts=conflicts,
    )


def merged_node_counts(hkl, intensity, sigma, basis, d_min):
    """``NodeCounts`` from intensities merged in P 1, for
    ``space_group_candidates`` (use ``rel_res=0``: the reflections are
    already separated).  Friedel mates are filled in.

    Args:
        hkl: (N, 3) Miller indices
        intensity, sigma: (N,) merged intensity and its standard error
        basis: (3, 3) direct basis the indices refer to, columns [Angstrom]
        d_min: [Angstrom] resolution limit of the test
    """
    hkl = np.asarray(hkl, dtype=np.int64)
    ok = np.isfinite(intensity) & np.isfinite(sigma) & (np.asarray(sigma) > 0)
    hkl = np.concatenate([hkl[ok], -hkl[ok]])
    val = np.tile(np.asarray(intensity, dtype=float)[ok], 2)
    err = np.tile(np.asarray(sigma, dtype=float)[ok], 2)
    hmax = np.abs(hkl).max(axis=0)
    shape = tuple(2 * hmax + 1)
    net, errs, cov = np.zeros(shape), np.ones(shape), np.zeros(shape, dtype=bool)
    idx = tuple((hkl + hmax).T)
    net[idx], errs[idx], cov[idx] = val, err, True
    cov[tuple(hmax)] = False
    return NodeCounts(
        hmax, net, errs, cov, np.asarray(basis, dtype=float), float(d_min), 0
    )
