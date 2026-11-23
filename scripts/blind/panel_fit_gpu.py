"""Prototype: the per-image joint fit of panel_modes.py in JAX, for the GPU.

Same model: counts(pixel, time bin) ~ Poisson(background + sum_k I_k P_k T_k),
all reflections of a detector image at once.  Nothing the size of the
voxel grid is factorised:

- each reflection carries its own window (pixels x time bins) and the
  profile on it; the model cube is a scatter-add of those windows;
- the information matrix H (reflections x reflections) is assembled pair
  by pair from a neighbour list.  The profile is separable, so a pair's
  overlap is a (pixels, bins) sum on one reflection's window with the
  other's spot and time profile evaluated there;
- H scaled to unit diagonal is solved by Cholesky.  The diagonal of its
  inverse is each reflection's variance inflation.  Only if that exceeds
  1 / threshold somewhere is the image redone with an eigendecomposition
  that leaves out the directions below the threshold.

Usage:
    python panel_fit_gpu.py merged.h5 integrated_all.h5 out.npz [--images ...]
"""

import argparse
import time
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import cho_solve
from panel_modes import DL, HALF, time_profile
from scipy.spatial import cKDTree

from subhkl.io.events import EventStore

jax.config.update("jax_enable_x64", True)

NPIX = (2 * HALF + 1) ** 2
PAIR_BATCH = 64
N_ITER = 4


def _blur(x, sigma=6.0):
    """Separable Gaussian blur of a 2D map (edges reflected)."""
    half = int(4 * sigma)
    k = jnp.exp(-0.5 * (jnp.arange(-half, half + 1) / sigma) ** 2)
    k = k / k.sum()
    xp = jnp.pad(x, half, mode="symmetric")
    xp = jax.vmap(lambda row: jnp.convolve(row, k, mode="valid"))(xp)
    return jax.vmap(
        lambda col: jnp.convolve(col, k, mode="valid"), in_axes=1, out_axes=1
    )(xp)


@partial(jax.jit, static_argnames=("size",))
def _histogram(ev, size):
    return jnp.zeros(size, jnp.int32).at[ev].add(1, mode="drop")


def _fit(N, valid_px, pflat, P, T, b0, active, nbr, geo, thr, truncate, nt):
    """One image.  Shapes: P, pflat (n, NPIX); T (n, NB); nbr (n, K)."""
    H_, W_ = valid_px.shape
    n, nb = T.shape
    size = H_ * W_ * nt
    pr, pc, vu, vv, cuv, r0, c0 = geo
    tbk = b0[:, None] + jnp.arange(nb)[None, :]
    tbc = jnp.clip(tbk, 0, nt - 1)
    vox = pflat[:, :, None] * nt + tbc[:, None, :]
    a = P[:, :, None] * T[:, None, :]
    N = N.astype(jnp.float64)
    valid_vox = jnp.repeat(valid_px.ravel(), nt)

    # background: pixel map times time spectrum, from voxels no reflection touches
    cover = jnp.zeros(size).at[vox].add((a > 0).astype(jnp.float64)) > 0
    free = (~cover & valid_vox).reshape(H_ * W_, nt)
    Nf = N.reshape(H_ * W_, nt) * free
    s = Nf.sum(0) / jnp.maximum(free.sum(0), 1)
    s = jnp.convolve(s, jnp.ones(5) / 5, mode="same")
    wgt = (free * s[None, :]).sum(1).reshape(H_, W_)
    b = _blur(Nf.sum(1).reshape(H_, W_)) / jnp.maximum(_blur(wgt), 1e-9)
    mu_bg = jnp.maximum((b.reshape(-1, 1) * s[None, :]).ravel(), 1e-4)

    rr, cc = pflat // W_, pflat % W_
    act = active.astype(jnp.float64)
    nbr_c = jnp.clip(nbr, 0, n - 1)
    nbr_ok = nbr < n

    def expected(amp):
        model = jnp.zeros(size).at[vox].add((amp * act)[:, None, None] * a)
        return jnp.maximum(mu_bg + model, 0.3 * mu_bg)

    def information(mu):
        w2 = (valid_vox / mu).reshape(H_ * W_, nt)

        def row(k):
            wk = w2[pflat[k]][:, tbc[k]]  # (NPIX, NB)
            j = nbr_c[k]
            dv = rr[k][None, :] - pr[j][:, None]
            du = cc[k][None, :] - pc[j][:, None]
            det = jnp.maximum(vu[j] * vv[j] - cuv[j] ** 2, 1e-6)[:, None]
            m2 = (
                vv[j][:, None] * du**2
                - 2 * cuv[j][:, None] * du * dv
                + vu[j][:, None] * dv**2
            ) / det
            inwin = (jnp.abs(rr[k][None, :] - r0[j][:, None]) <= HALF) & (
                jnp.abs(cc[k][None, :] - c0[j][:, None]) <= HALF
            )
            pj = jnp.exp(-0.5 * m2) / (2 * jnp.pi * jnp.sqrt(det)) * inwin
            rel = tbk[k][None, :] - b0[j][:, None]
            tj = jnp.take_along_axis(T[j], jnp.clip(rel, 0, nb - 1), axis=1) * (
                (rel >= 0) & (rel < nb)
            )
            x = jnp.einsum("jp,pt->jt", pj * P[k][None, :], wk)
            return jnp.sum(x * tj * T[k][None, :], axis=1) * nbr_ok[k] * act[k] * act[j]

        rows = lax.map(row, jnp.arange(n), batch_size=PAIR_BATCH)
        Hm = jnp.zeros((n, n)).at[jnp.arange(n)[:, None], nbr].add(rows, mode="drop")
        Hm = 0.5 * (Hm + Hm.T)
        return Hm + jnp.diag(1.0 - act)

    def score(mu):
        ratio = (N - mu) / mu * valid_vox
        return jnp.sum(a * ratio[vox], axis=(1, 2)) * act

    def solver(Hm):
        d = jnp.sqrt(jnp.diag(Hm))
        R = Hm / jnp.outer(d, d)
        if truncate:
            lam, V = jnp.linalg.eigh(R)
            kept = lam > thr

            def solve(rhs):
                return V @ jnp.where(kept, (V.T @ rhs) / jnp.where(kept, lam, 1.0), 0.0)

            infl = jnp.sum(
                jnp.where(kept, V**2 / jnp.where(kept, lam, 1.0), 0.0), axis=1
            )
            lev = jnp.sum(jnp.where(kept, 0.0, V**2), axis=1)
            return d, solve, infl, lev, jnp.sum(~kept)
        L = jnp.linalg.cholesky(R)

        def solve(rhs):
            return cho_solve((L, True), rhs)

        return d, solve, None, None, 0

    def step(_, amp):
        mu = expected(amp)
        d, solve, *_ = solver(information(mu))
        return amp + solve(score(mu) / d) / d

    amp = lax.fori_loop(0, N_ITER, step, jnp.zeros(n))
    mu = expected(amp)
    d, solve, infl, lev, dropped = solver(information(mu))
    if not truncate:
        infl = jnp.diag(solve(jnp.eye(n)))
        lev = jnp.zeros(n)
    return amp * act, jnp.sqrt(jnp.maximum(infl, 0.0)) / d * act, infl, lev, dropped


_fit_jit = jax.jit(_fit, static_argnames=("truncate", "nt"))


def _bucket(v, quantum):
    return int(np.ceil(max(v, 1) / quantum) * quantum)


def prepare(pk, valid_px, prof_x, prof_cdf, l0, nt):
    """Host side: windows, profiles and neighbour list of one image's reflections."""
    H_, W_ = valid_px.shape
    n = len(pk["tof"])
    lt = np.log(pk["tof"])
    r0, c0 = (
        np.rint(pk["pixel_r"]).astype(np.int64),
        np.rint(pk["pixel_c"]).astype(np.int64),
    )
    o = np.arange(-HALF, HALF + 1)
    rr = r0[:, None, None] + o[None, :, None]
    cc = c0[:, None, None] + o[None, None, :]
    dv, du = rr - pk["pixel_r"][:, None, None], cc - pk["pixel_c"][:, None, None]
    vu, vv, cuv = (
        pk["var_u"][:, None, None],
        pk["var_v"][:, None, None],
        pk["cov_uv"][:, None, None],
    )
    det = np.maximum(vu * vv - cuv**2, 1e-6)
    P = np.exp(-0.5 * (vv * du**2 - 2 * cuv * du * dv + vu * dv**2) / det) / (
        2 * np.pi * np.sqrt(det)
    )
    inb = (rr >= 0) & (rr < H_) & (cc >= 0) & (cc < W_)
    pflat = np.where(inb, rr * W_ + cc, 0)
    P = (P * inb * valid_px.ravel()[pflat]).reshape(n, -1)
    pflat = pflat.reshape(n, -1)
    nb = int(np.ceil((prof_x[-1] - prof_x[0]) / DL)) + 1
    b0 = np.floor((lt + prof_x[0] - l0) / DL).astype(np.int64)
    be = b0[:, None] + np.arange(nb + 1)[None, :]
    T = np.diff(
        np.interp(l0 + be * DL - lt[:, None], prof_x, prof_cdf, left=0.0, right=1.0),
        axis=1,
    )
    T = T * ((be[:, :-1] >= 0) & (be[:, :-1] < nt))
    well = (P.sum(1) >= 0.65) & (T.sum(1) >= 0.9)

    tree = cKDTree(np.stack([r0, c0], 1).astype(float))
    pairs = tree.query_pairs(2 * HALF + 0.5, p=np.inf, output_type="ndarray")
    ok = (
        (np.abs(b0[pairs[:, 0]] - b0[pairs[:, 1]]) < nb)
        & well[pairs[:, 0]]
        & well[pairs[:, 1]]
    )
    pairs = pairs[ok]
    src = np.concatenate([np.arange(n), pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([np.arange(n), pairs[:, 1], pairs[:, 0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    count = np.bincount(src, minlength=n)
    n_pad, k_pad = _bucket(n, 1024), max(16, int(2 ** np.ceil(np.log2(count.max()))))
    nbr = np.full((n_pad, k_pad), n_pad, dtype=np.int32)
    nbr[src, np.arange(len(src)) - np.repeat(np.cumsum(count) - count, count)] = dst

    def pad(x, fill=0):
        out = np.full((n_pad, *x.shape[1:]), fill, dtype=x.dtype)
        out[:n] = x
        return out

    geo = tuple(
        pad(np.asarray(v, dtype=np.float64))
        for v in (
            pk["pixel_r"],
            pk["pixel_c"],
            pk["var_u"],
            pk["var_v"],
            pk["cov_uv"],
            r0,
            c0,
        )
    )
    geo = (
        *geo[:2],
        pad(pk["var_u"].astype(np.float64), 1.0),
        pad(pk["var_v"].astype(np.float64), 1.0),
        *geo[4:],
    )
    return (
        pad(pflat.astype(np.int32)),
        pad(P),
        pad(T),
        pad(b0.astype(np.int32)),
        pad(well),
        nbr,
        geo,
        well,
        count.max(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument(
        "integrated",
        help="gated integration with nothing censored (geometry, stacked time profile)",
    )
    ap.add_argument("out")
    ap.add_argument("--images", type=int, nargs="*", default=None)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument(
        "--time-shift",
        type=float,
        default=0.0,
        help="null control: fit at predicted time x (1 + shift)",
    )
    args = ap.parse_args()

    with h5py.File(args.integrated, "r") as f:
        pk_all = {
            k: f["peaks"][k][()]
            for k in (
                "image_index",
                "pixel_r",
                "pixel_c",
                "tof",
                "var_u",
                "var_v",
                "cov_uv",
            )
        }
        prof_x, prof_cdf = time_profile(
            f["tof_gate/stack_edges"][()], f["tof_gate/stack_counts"][()]
        )
        mask = f["detector/pixel_mask"][()].astype(bool)
        mask_ids = f["detector/mask_bank_ids"][()]
    with h5py.File(args.merged, "r") as f:
        images = f["images"][()]
        bank_ids = f["bank_ids"][()]
    pk_all["tof"] = pk_all["tof"] * (1.0 + args.time_shift)
    img = pk_all["image_index"].astype(int)
    order = np.argsort(img, kind="stable")
    bounds = np.searchsorted(img[order], np.arange(len(images) + 1))
    # one time grid for the whole dataset, so every image compiles to the same cube
    lt_all = np.log(pk_all["tof"])
    l0 = lt_all.min() + prof_x[0] - 0.03
    nt = int(np.ceil((lt_all.max() + prof_x[-1] + 0.03 - l0) / DL))
    H_, W_ = images[0].shape
    size = H_ * W_ * nt
    pick = list(args.images) if args.images else list(range(len(images)))
    print(
        f"time grid: {nt} bins of {DL}; reflection window {2 * HALF + 1}x{2 * HALF + 1} pixels x {int(np.ceil((prof_x[-1] - prof_x[0]) / DL)) + 1} bins",
        flush=True,
    )

    rows, flux, sigma, plain, infl_all = [], [], [], [], []
    stats = []
    seen = set()
    t_start = time.time()
    with EventStore(args.merged) as store:
        for i in pick:
            ix = order[bounds[i] : bounds[i + 1]]
            if len(ix) == 0:
                continue
            t0 = time.time()
            pk = {k: v[ix] for k, v in pk_all.items()}
            valid_px = mask[int(np.flatnonzero(mask_ids == bank_ids[i])[0])]
            tof, off = store.image_events(i, images[i])
            tb = np.floor((np.log(tof) - l0) / DL).astype(np.int64)
            pix = np.repeat(np.arange(H_ * W_), np.diff(off))
            ev = np.where((tb >= 0) & (tb < nt), pix * nt + tb, size).astype(np.int32)
            ev_pad = np.full(_bucket(len(ev), 1 << 20), size, dtype=np.int32)
            ev_pad[: len(ev)] = ev
            pflat, P, T, b0, active, nbr, geo, well, kmax = prepare(
                pk, valid_px, prof_x, prof_cdf, l0, nt
            )
            t1 = time.time()
            N = _histogram(jnp.asarray(ev_pad), size)
            dev = (
                N,
                jnp.asarray(valid_px),
                jnp.asarray(pflat),
                jnp.asarray(P),
                jnp.asarray(T),
                jnp.asarray(b0),
                jnp.asarray(active),
                jnp.asarray(nbr),
                tuple(jnp.asarray(g) for g in geo),
            )
            key = (P.shape[0], nbr.shape[1])
            fresh = key not in seen
            seen.add(key)
            out = _fit_jit(*dev, args.threshold, truncate=False, nt=nt)
            amp, sig, infl, lev, dropped = (np.asarray(o) for o in out)
            redo = (not np.all(np.isfinite(amp))) or (
                infl[: len(ix)][well].max(initial=0.0) > 1.0 / args.threshold
            )
            if redo:
                out = _fit_jit(*dev, args.threshold, truncate=True, nt=nt)
                amp, sig, infl, lev, dropped = (np.asarray(o) for o in out)
            n = len(ix)
            rows.append(ix[well])
            flux.append(amp[:n][well])
            sigma.append(sig[:n][well])
            plain.append(lev[:n][well] < 0.01)
            infl_all.append(infl[:n][well])
            t2 = time.time()
            stats.append(
                (
                    i,
                    n,
                    int(well.sum()),
                    kmax,
                    int(dropped),
                    redo,
                    t1 - t0,
                    t2 - t1,
                    fresh,
                )
            )
            print(
                f"image {i:4d}: {n:5d} reflections, {len(tof) / 1e6:5.2f}M events, up to {kmax:3d} neighbours, host {t1 - t0:5.2f} s, device {t2 - t1:6.2f} s{' (compiled)' if fresh else ''}{f', truncated: {int(dropped)} directions dropped' if redo else ''}",
                flush=True,
            )

    st = np.array([s[:8] for s in stats], dtype=float)
    warm = np.array([not s[8] and not s[5] for s in stats])
    total = time.time() - t_start
    print(
        f"\n{len(stats)} images in {total / 60:.1f} min; {int(st[:, 2].sum())} reflections fitted"
    )
    if warm.any():
        print(
            f"per image without compilation: host {np.median(st[warm, 6]):.2f} s, device median {np.median(st[warm, 7]):.2f} s, largest {st[warm, 7].max():.2f} s"
        )
    print(
        f"compiled shapes: {len(seen)}; images redone with the eigendecomposition: {int(st[:, 5].sum())}; directions dropped: {int(st[:, 4].sum())}"
    )
    np.savez(
        args.out,
        rows=np.concatenate(rows),
        flux=np.concatenate(flux),
        sigma=np.concatenate(sigma),
        plain=np.concatenate(plain),
        inflation=np.concatenate(infl_all),
        dropped=st[:, 4],
        seconds=st[:, 6] + st[:, 7],
        image=st[:, 0],
    )


if __name__ == "__main__":
    main()
