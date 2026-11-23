"""Prototype: point-group symmetry from intensities integrated once in P1.

Symmetry mates h and M.h have the same d-spacing, so two observations of
them share the scattering angle whenever they share the wavelength; the
incident spectrum and the Lorentz factor are then the same for both and
their intensities compare directly, with no wavelength normalisation.

For every rotation M of the lattice (and for inversion) the spread of
(I1 - I2) / sigma over such matched pairs is compared with two yardsticks
taken from the same data: the same reflection measured twice (what a true
symmetry looks like) and unrelated reflections of the same d (what a false
one looks like).

Usage:
    python laue_class.py integrated_p1.h5 blind_solve.h5 [--d-min 2.5]
"""

import argparse

import gemmi
import h5py
import numpy as np

OFF = 512


def key(hkl):
    hkl = np.asarray(hkl, dtype=np.int64)
    return ((hkl[:, 0] + OFF) * 2048 + (hkl[:, 1] + OFF)) * 2048 + (hkl[:, 2] + OFF)


def spread(z):
    """Robust standard deviation."""
    return (
        0.7413 * float(np.diff(np.percentile(z, [25, 75]))[0])
        if len(z) > 30
        else np.nan
    )


class Observations:
    def __init__(self, filename, d_min, match=0.02, min_isigi=5.0):
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
        d = 1.0 / np.linalg.norm(hkl @ B.T, axis=1)
        ok = (p["sigma"] > 0) & (d > d_min)
        self.hkl, self.d = hkl[ok], d[ok]
        self.I, self.sig, self.lam = p["intensity"][ok], p["sigma"][ok], p["lambda"][ok]
        self.run, self.bank = p["run_index"][ok], p["bank"][ok]
        self.match, self.min_isigi = match, min_isigi
        self.order = np.argsort(key(self.hkl), kind="stable")
        self.keys = key(self.hkl)[self.order]

    def mate_pairs(self, M, max_pairs=300000):
        """Index pairs (i, j): observation j is of M.h_i, at matched wavelength,
        from a different run or bank."""
        want = key(self.hkl @ np.asarray(M).T)
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
        keep = (i != j) & (
            (self.run[i] != self.run[j]) | (self.bank[i] != self.bank[j])
        )
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
        keep = (np.abs(np.log(self.lam[i] / self.lam[j])) < self.match) & np.any(
            self.hkl[i] != self.hkl[j], axis=1
        )
        return i[keep], j[keep]

    def statistic(self, i, j):
        """(pairs, strong pairs, spread of (I1 - I2)/sigma over the strong pairs)."""
        strong = (self.I[i] / self.sig[i] > self.min_isigi) | (
            self.I[j] / self.sig[j] > self.min_isigi
        )
        z = (self.I[i] - self.I[j]) / np.sqrt(self.sig[i] ** 2 + self.sig[j] ** 2)
        return len(i), int(strong.sum()), spread(z[strong])


def describe(M):
    """Rotation order and axis (in direct-lattice indices) of an integer matrix on hkl."""
    M = np.asarray(M)
    order = next(
        n
        for n in (1, 2, 3, 4, 6)
        if np.array_equal(np.linalg.matrix_power(M, n), np.eye(3, dtype=int))
    )
    if order == 1:
        return "identity"
    W = np.linalg.inv(M).T  # acts on direct-lattice coordinates
    w, v = np.linalg.eig(W)
    axis = np.real(v[:, np.argmin(np.abs(w - 1))])
    axis = axis / np.abs(axis[np.abs(axis) > 1e-6]).min()
    return f"{order}-fold about [{' '.join(str(round(x)) for x in axis)}]"


def group_from(mats):
    ops = []
    for M in mats:
        W = np.rint(np.linalg.inv(np.asarray(M, dtype=float)).T).astype(int)
        op = gemmi.Op()
        op.rot = (W * gemmi.Op.DEN).tolist()
        ops.append(op)
    return gemmi.find_spacegroup_by_ops(gemmi.GroupOps(ops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("integrated")
    ap.add_argument("solve")
    ap.add_argument("--d-min", type=float, default=2.5)
    args = ap.parse_args()

    with h5py.File(args.solve, "r") as f:
        mats = f["blind/lattice_operations"][()]
        lattice_group = f["blind"].attrs["lattice_group"]
    obs = Observations(args.integrated, args.d_min)
    print(
        f"{len(obs.I)} observations with d > {args.d_min} A, integrated in P1, unscaled"
    )
    print(f"lattice rotation group: {lattice_group} ({len(mats)} rotations)\n")

    _n0, s0, same = obs.statistic(*obs.mate_pairs(np.eye(3, dtype=int)))
    _nr, sr, unrelated = obs.statistic(*obs.random_pairs())
    cut = same + 0.5 * (unrelated - same)
    print(
        f"yardsticks: same reflection re-measured {same:.2f} ({s0} strong pairs); unrelated reflections {unrelated:.2f} ({sr}); accept below {cut:.2f}\n"
    )
    print(f"  {'operation':34s} {'pairs':>8s} {'strong':>7s} {'spread':>7s}  verdict")
    accepted = [np.eye(3, dtype=int)]
    for M in list(mats) + [-np.eye(3, dtype=int)]:
        if np.array_equal(M, np.eye(3, dtype=int)):
            continue
        name = (
            "inversion (Friedel)"
            if np.array_equal(M, -np.eye(3, dtype=int))
            else describe(M)
        )
        n, s, val = obs.statistic(*obs.mate_pairs(M))
        if s < 100 or not np.isfinite(val):
            verdict = "too few pairs"
        elif val < cut:
            verdict = "symmetry"
            if not np.array_equal(M, -np.eye(3, dtype=int)):
                accepted.append(np.asarray(M))
        else:
            verdict = "NOT a symmetry"
        print(f"  {name:34s} {n:8d} {s:7d} {val:7.2f}  {verdict}")
    sg = group_from(accepted)
    print(f"\nrotations accepted: {len(accepted)} of {len(mats)}")
    print(
        f"point group of the rotations: {sg.point_group_hm()}   Laue class: {sg.laue_str()}   (highest-symmetry candidate space group without screw axes: {sg.hm})"
    )


if __name__ == "__main__":
    main()
