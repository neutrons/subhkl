"""Prototype: one joint fit per detector image, and its determined directions.

Today's time-gated integrator fits one small tile per reflection (the
reflection and up to seven neighbours) on counts summed over a time gate,
and censors a reflection whose variance is inflated more than 1000-fold by
its neighbours.  This script instead fits every reflection of an image at
once, in position and time:

    counts(pixel, time bin) ~ Poisson(background + sum_k I_k P_k(pixel) T_k(time))

with P_k the predicted Gaussian spot and T_k the measured stacked time
profile, and looks at the information matrix H = A^T diag(1/mu) A of that
fit, scaled to unit diagonal (R).  An eigenvalue of R is the inverse of the
variance inflation along its eigenvector.

- Directions with eigenvalue below ``--threshold`` are dropped (never
  solved for); the rest is a pseudo-inverse solve.  No ridge, no censoring
  of individual reflections.
- A reflection's leverage is its squared length inside the dropped
  subspace.  Reflections with negligible leverage are ordinary
  observations.  The others are grouped, and each group yields
  (members - dropped directions) uncorrelated combinations.
- Signal check: events are split at random into two halves and the same
  linear estimator is applied to each.

Usage:
    python panel_modes.py merged.h5 integrated_all.h5 --censored integrated.h5
"""

import argparse
import time

import h5py
import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from subhkl.io.events import EventStore

DL = 0.003  # [-] log-time bin
HALF = 9  # [pixel] half-size of a reflection's window (3 sigma of the widest spots)


def time_profile(edges, counts):
    """Cumulative stacked time profile above its baseline: (x, cdf)."""
    x = 0.5 * (edges[:-1] + edges[1:])
    outer = np.abs(x) > 0.6 * edges[-1]
    base = np.polyval(np.polyfit(x[outer], counts[outer], 1), x)
    excess = counts - base
    smooth = np.convolve(excess, np.ones(3) / 3, mode="same")
    noise = np.sqrt(np.maximum(base, 1.0) / 3)
    peak = int(np.argmax(np.where(~outer, smooth, -np.inf)))
    lo = peak
    while lo > 0 and not outer[lo - 1] and smooth[lo - 1] > noise[lo - 1]:
        lo -= 1
    hi = peak
    while hi < len(x) - 1 and not outer[hi + 1] and smooth[hi + 1] > noise[hi + 1]:
        hi += 1
    mass = np.maximum(excess[lo : hi + 1], 0.0)
    cdf = np.concatenate([[0.0], np.cumsum(mass) / mass.sum()])
    return edges[lo : hi + 2], cdf


def build(image_idx, store, images, valid_px, pk, prof_x, prof_cdf):
    """Counts cube, design matrix and bookkeeping for one image."""
    H_, W_ = images[image_idx].shape
    tof, off = store.image_events(image_idx, images[image_idx])
    lt_pred = np.log(pk["tof"])
    l0 = lt_pred.min() + prof_x[0] - 0.03
    nt = int(np.ceil((lt_pred.max() + prof_x[-1] + 0.03 - l0) / DL))
    pix = np.repeat(np.arange(H_ * W_), np.diff(off))
    tb = np.floor((np.log(tof) - l0) / DL).astype(np.int64)
    ok = (tb >= 0) & (tb < nt)
    flat_ev = pix[ok] * nt + tb[ok]

    n = len(lt_pred)
    r0, c0 = np.rint(pk["pixel_r"]).astype(int), np.rint(pk["pixel_c"]).astype(int)
    o = np.arange(-HALF, HALF + 1)
    rr = r0[:, None, None] + o[None, :, None]
    cc = c0[:, None, None] + o[None, None, :]
    dv = rr - pk["pixel_r"][:, None, None]
    du = cc - pk["pixel_c"][:, None, None]
    vu, vv, cuv = (
        pk["var_u"][:, None, None],
        pk["var_v"][:, None, None],
        pk["cov_uv"][:, None, None],
    )
    det = np.maximum(vu * vv - cuv**2, 1e-6)
    m2 = (vv * du**2 - 2 * cuv * du * dv + vu * dv**2) / det
    P = np.exp(-0.5 * m2) / (2 * np.pi * np.sqrt(det))
    inb = (rr >= 0) & (rr < H_) & (cc >= 0) & (cc < W_)
    pflat = np.where(inb, rr * W_ + cc, 0)
    P = P * inb * valid_px.ravel()[pflat]
    P = P.reshape(n, -1)
    pflat = pflat.reshape(n, -1)

    nb = int(np.ceil((prof_x[-1] - prof_x[0]) / DL)) + 1
    b0 = np.floor((lt_pred + prof_x[0] - l0) / DL).astype(int)
    be = b0[:, None] + np.arange(nb + 1)[None, :]
    T = np.diff(
        np.interp(
            l0 + be * DL - lt_pred[:, None], prof_x, prof_cdf, left=0.0, right=1.0
        ),
        axis=1,
    )
    tbk = be[:, :-1]
    T = T * ((tbk >= 0) & (tbk < nt))

    vals = (P[:, :, None] * T[:, None, :]).ravel()
    rows = (pflat[:, :, None] * nt + np.clip(tbk, 0, nt - 1)[:, None, :]).ravel()
    cols = np.repeat(np.arange(n), P.shape[1] * nb)
    nz = vals > 0
    A = coo_matrix((vals[nz], (rows[nz], cols[nz])), shape=(H_ * W_ * nt, n)).tocsr()
    well = (P.sum(1) >= 0.65) & (T.sum(1) >= 0.9)
    return flat_ev, A, well, nt


def background(N, A, valid_vox, shape):
    """Separable background (pixel map times time spectrum) from voxels no reflection touches."""
    H_, W_, nt = shape
    free = (np.diff(A.indptr) == 0) & valid_vox
    F = free.reshape(H_ * W_, nt)
    Nf = (N * free).reshape(H_ * W_, nt)
    s = Nf.sum(0) / np.maximum(F.sum(0), 1)
    s = np.convolve(s, np.ones(5) / 5, mode="same")
    w = (F * s[None, :]).sum(1).reshape(H_, W_)
    num = Nf.sum(1).reshape(H_, W_)
    b = gaussian_filter(num, 6.0) / np.maximum(gaussian_filter(w, 6.0), 1e-9)
    mu = (b.reshape(-1, 1) * s[None, :]).ravel()
    chi2 = float(
        (((N - mu) ** 2 / np.maximum(mu, 1e-3))[free]).sum() / max(free.sum(), 1)
    )
    return np.maximum(mu, 1e-4), float(free.mean()), chi2


def weighted(A, w):
    Aw = A.multiply(w[:, None]).tocsr()
    return np.asarray((A.T @ Aw).todense())


def analyse(image_idx, store, images, valid_px, pk, prof, thr, rng, pipe):
    t0 = time.time()
    flat_ev, A_all, well, nt = build(image_idx, store, images, valid_px, pk, *prof)
    shape = (*images[image_idx].shape, nt)
    size = int(np.prod(shape))
    N = np.bincount(flat_ev, minlength=size).astype(np.float64)
    valid_vox = np.repeat(valid_px.ravel(), nt)
    mu_bg, free_frac, chi2_bg = background(N, A_all, valid_vox, shape)
    keep = np.flatnonzero(well)
    A = A_all[:, keep]
    touched = np.diff(A.indptr) > 0
    sel = np.flatnonzero(
        touched & valid_vox
    )  # only voxels some reflection touches matter
    A = A[sel]
    Ns, bg = N[sel], mu_bg[sel]
    n = len(keep)

    amp = np.zeros(n)
    for _ in range(4):
        mu = np.maximum(bg + A @ amp, 0.3 * bg)
        Hm = weighted(A, 1.0 / mu)
        g = A.T @ ((Ns - mu) / mu)
        d = np.sqrt(np.diag(Hm))
        lam, V = eigh(Hm / np.outer(d, d))
        kept = lam > thr
        amp = amp + (V[:, kept] @ ((V[:, kept].T @ (g / d)) / lam[kept])) / d
    # final linearisation point
    mu = np.maximum(bg + A @ amp, 0.3 * bg)
    Hm = weighted(A, 1.0 / mu)
    d = np.sqrt(np.diag(Hm))
    lam, V = eigh(Hm / np.outer(d, d))
    kept = lam > thr
    Vk, Vd = V[:, kept], V[:, ~kept]
    chi2 = float(((Ns - mu) ** 2 / mu).sum() / max(len(Ns) - kept.sum(), 1))

    def estimate(counts, bg_part):
        """Whitened amplitudes u = d * I from the fixed linear estimator."""
        rhs = (A.T @ ((counts - bg_part) / mu)) / d
        return Vk @ ((Vk.T @ rhs) / lam[kept])

    u = estimate(Ns, bg)
    N1 = rng.binomial(Ns.astype(np.int64), 0.5).astype(np.float64)
    u1, u2 = estimate(2 * N1, bg), estimate(2 * (Ns - N1), bg)
    var_u = np.einsum(
        "ij,j,ij->i", Vk, 1.0 / lam[kept], Vk
    )  # variance inflation of each reflection

    lev = np.einsum("ij,ij->i", Vd, Vd) if Vd.shape[1] else np.zeros(n)
    plain = lev < 0.01
    S = np.flatnonzero(~plain)
    combos = []  # (z, sigma, z_half1, z_half2, group size)
    group_sizes, part = [], []
    if Vd.shape[1]:
        part = 1.0 / (Vd**4).sum(0)
    if len(S):
        Pd = Vd[S] @ Vd[S].T
        _, lab = connected_components(csr_matrix(np.abs(Pd) > 0.01), directed=False)
        Cs = (Vk[S] / lam[kept]) @ Vk[S].T
        for gi in range(lab.max() + 1):
            m = np.flatnonzero(lab == gi)
            group_sizes.append(len(m))
            ev, W = np.linalg.eigh(Pd[np.ix_(m, m)])
            Wn = W[:, ev < 0.05]  # directions inside the group that the data determine
            if Wn.shape[1] == 0:
                continue
            cv, Q = np.linalg.eigh(Wn.T @ Cs[np.ix_(m, m)] @ Wn)
            B = Wn @ Q
            for j in range(B.shape[1]):
                combos.append(
                    (
                        B[:, j] @ u[S[m]],
                        np.sqrt(cv[j]),
                        B[:, j] @ u1[S[m]],
                        B[:, j] @ u2[S[m]],
                        len(m),
                    )
                )
    combos = np.array(combos).reshape(-1, 5)

    cens_today = pipe["censored"][keep]
    out = {
        "image": image_idx,
        "keep": keep,
        "n": n,
        "n_events": len(flat_ev),
        "free": free_frac,
        "chi2_bg": chi2_bg,
        "chi2": chi2,
        "lam": lam,
        "dropped": int((~kept).sum()),
        "part": np.asarray(part),
        "n_S": len(S),
        "group_sizes": np.array(group_sizes),
        "combos": combos,
        "cens_today": int(cens_today.sum()),
        "plain": plain,
        "cens": cens_today,
        "u": u,
        "u1": u1,
        "u2": u2,
        "sig_u": np.sqrt(var_u),
        "flux": u / d,
        "sig_flux": np.sqrt(var_u) / d,
        "pipe_I": pipe["intensity"][keep],
        "pipe_sig": pipe["sigma"][keep],
        "d": pk["d"][keep],
        "seconds": time.time() - t0,
    }
    return out


def cc(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument(
        "integrated",
        help="gated integration with nothing censored (geometry, stacked time profile)",
    )
    ap.add_argument(
        "--censored", required=True, help="the same integration with today's censoring"
    )
    ap.add_argument("--images", type=int, nargs="*", default=None)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--n-worst", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--shard",
        default=None,
        help="k/n: every n-th image starting at k, instead of a sample",
    )
    ap.add_argument(
        "--write",
        default=None,
        help="save row index, flux and sigma of every fitted reflection",
    )
    args = ap.parse_args()

    with h5py.File(args.integrated, "r") as f, h5py.File(args.censored, "r") as g:
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
                "sigma",
                "intensity",
                "lambda",
                "two_theta",
            )
        }
        s5, i5 = g["peaks/sigma"][()], g["peaks/intensity"][()]
        prof = time_profile(
            f["tof_gate/stack_edges"][()], f["tof_gate/stack_counts"][()]
        )
        mask = f["detector/pixel_mask"][()].astype(bool)
        mask_ids = f["detector/mask_bank_ids"][()]
    with h5py.File(args.merged, "r") as f:
        images = f["images"][()]
        bank_ids = f["bank_ids"][()]
    img = pk_all["image_index"].astype(int)
    tth = (
        pk_all["two_theta"]
        if pk_all["two_theta"].max() < 3.2
        else np.radians(pk_all["two_theta"])
    )
    pk_all["d"] = pk_all["lambda"] / (2 * np.sin(tth / 2))
    censored = (s5 <= 0) & (pk_all["sigma"] > 0)
    rng = np.random.default_rng(1)
    if args.shard:
        k_, n_ = (int(v) for v in args.shard.split("/"))
        pick = list(range(len(images)))[k_::n_]
    elif args.images:
        pick = list(args.images)
    else:
        worst = np.argsort(-np.bincount(img[censored], minlength=len(images)))[
            : args.n_worst
        ]
        rest = np.setdiff1d(np.arange(len(images)), worst)
        pick = [*rng.choice(rest, args.n_random, replace=False), *worst]
    width = prof[0][-1] - prof[0][0]
    print(
        f"time profile: support {prof[0][0]:+.3f} to {prof[0][-1]:+.3f} of the predicted time ({width / DL:.0f} bins of {DL})"
    )
    print(
        f"dropping directions with eigenvalue below {args.threshold:g} (variance inflation above {1 / args.threshold:g})\n"
    )
    print(
        " image  refl  events  bg chi2  fit chi2 | censored today | dropped dirs | with leverage  groups  combos | new obs lost | s"
    )

    res = []
    with EventStore(args.merged) as store:
        for i in pick:
            ix = np.flatnonzero(img == i)
            pk = {k: v[ix] for k, v in pk_all.items()}
            slot = int(np.flatnonzero(mask_ids == bank_ids[i])[0])
            pipe = {"censored": censored[ix], "intensity": i5[ix], "sigma": s5[ix]}
            r = analyse(
                int(i), store, images, mask[slot], pk, prof, args.threshold, rng, pipe
            )
            r["rows"] = ix[r["keep"]]
            res.append(r)
            lost = r["n_S"] - len(r["combos"])
            print(
                f" {r['image']:5d} {r['n']:5d} {r['n_events'] / 1e6:6.2f}M {r['chi2_bg']:7.2f} {r['chi2']:8.2f} | {r['cens_today']:6d} ({100 * r['cens_today'] / r['n']:4.1f}%) |"
                f" {r['dropped']:5d} ({100 * r['dropped'] / r['n']:4.1f}%) | {r['n_S']:6d} {len(r['group_sizes']):7d} {len(r['combos']):7d} | {lost:5d} ({100 * lost / r['n']:4.1f}%) | {r['seconds']:.0f}",
                flush=True,
            )

    if args.write:
        np.savez(
            args.write,
            rows=np.concatenate([r["rows"] for r in res]),
            flux=np.concatenate([r["flux"] for r in res]),
            sigma=np.concatenate([r["sig_flux"] for r in res]),
            plain=np.concatenate([r["plain"] for r in res]),
            dropped=np.array([r["dropped"] for r in res]),
            seconds=np.array([r["seconds"] for r in res]),
            image=np.array([r["image"] for r in res]),
        )
    n = sum(r["n"] for r in res)
    cat = lambda k: np.concatenate([r[k] for r in res])
    lam, plain, cens = cat("lam"), cat("plain"), cat("cens")
    u, u1, u2, su = cat("u"), cat("u1"), cat("u2"), cat("sig_u")
    combos = np.concatenate([r["combos"] for r in res])
    part = cat("part")
    gs = cat("group_sizes")
    print(
        f"\n{len(res)} images, {n} reflections with enough of their profile on the detector"
    )
    print("eigenvalues of the real unit-diagonal information matrix:")
    for t in (1e-1, 1e-2, 1e-3, 1e-4):
        print(f"   below {t:g}: {100 * (lam < t).mean():5.2f}% of directions")
    print(f"censored today: {100 * cens.mean():.2f}% of reflections")
    if len(part):
        q = np.quantile(part, [0.5, 0.9, 1.0])
        print(
            f"dropped directions: {len(part)}; reflections each one spreads over (participation): median {q[0]:.1f}, 90% {q[1]:.1f}, largest {q[2]:.0f}"
        )
    print(
        f"reflections with leverage above 1% on the dropped subspace: {(~plain).sum()} ({100 * (~plain).mean():.2f}%)"
    )
    if len(gs):
        print(
            f"   they form {len(gs)} groups; sizes: median {np.median(gs):.0f}, 90% {np.quantile(gs, 0.9):.0f}, largest {gs.max()}"
        )
        print(f"   determined combinations exported from the groups: {len(combos)}")
    new_obs = plain.sum() + len(combos)
    print(
        f"observations: today {(~cens).sum()} ({100 * (~cens).mean():.1f}%), per-image fit {new_obs} ({100 * new_obs / n:.1f}%)"
    )

    print(
        "\nvariance inflation of ordinary reflections in the per-image fit (1 = isolated):"
    )
    for t in (1.5, 3, 10, 100):
        print(f"   above {t:g}: {100 * (su[plain] ** 2 > t).mean():5.2f}%")

    print("\nsignal check: the same estimator on two random halves of the events")
    print(
        "   set                                           number   CC halves  Spearman   mean (x/sigma)^2   share above 3 sigma"
    )

    def line(name, a, b, full, sig):
        if len(a) < 3:
            print(f"   {name:44s} {len(a):7d}")
            return
        print(
            f"   {name:44s} {len(a):7d}   {cc(a, b):8.3f}  {spearmanr(a, b)[0]:8.3f}   {np.mean((full / sig) ** 2):12.2f}   {100 * np.mean(full / sig > 3):12.1f}%"
        )

    line(
        "ordinary, kept today",
        u1[plain & ~cens],
        u2[plain & ~cens],
        u[plain & ~cens],
        su[plain & ~cens],
    )
    line(
        "ordinary, censored today",
        u1[plain & cens],
        u2[plain & cens],
        u[plain & cens],
        su[plain & cens],
    )
    if len(combos):
        line(
            "combinations from the groups",
            combos[:, 2],
            combos[:, 3],
            combos[:, 0],
            combos[:, 1],
        )

    dd = cat("d")
    print(
        "\nby resolution (ordinary reflections): number, CC between halves, mean (x/sigma)^2"
    )
    print("   shell [A]        kept today                 censored today")
    for lo, hi in (
        (4.0, 99),
        (3.0, 4.0),
        (2.5, 3.0),
        (2.2, 2.5),
        (2.0, 2.2),
        (1.7, 2.0),
        (0, 1.7),
    ):
        sh = (dd >= lo) & (dd < hi) & plain
        a, b = sh & ~cens, sh & cens
        fmt = lambda m: (
            f"{m.sum():6d}  {cc(u1[m], u2[m]):6.3f}  {np.mean((u[m] / su[m]) ** 2) if m.any() else float('nan'):6.2f}"
        )
        print(f"   {lo:3.1f} - {hi:4.1f}   {fmt(a)}      {fmt(b)}")

    both = plain & ~cens
    flux, pI, sf, ps = (
        cat("flux")[both],
        cat("pipe_I")[both],
        cat("sig_flux")[both],
        cat("pipe_sig")[both],
    )
    sane = np.abs(pI) < 50 * np.maximum(ps, 1e-9)
    print(f"   (today's rows more than 50 sigma from zero left out: {(~sane).sum()})")
    flux, pI, sf, ps = flux[sane], pI[sane], sf[sane], ps[sane]
    slope = float(np.median(pI[flux > 3 * sf] / flux[flux > 3 * sf]))
    print(
        f"\nagainst today's intensities (reflections kept by both): CC {cc(flux, pI):.3f}, Spearman {spearmanr(flux, pI)[0]:.3f}, today = {slope:.2f} x per-image fit"
    )
    print(
        f"   median sigma, today / ({slope:.2f} x per-image fit): {np.median(ps / (slope * sf)):.2f}"
    )
    if args.out:
        np.savez_compressed(
            args.out,
            lam=lam,
            plain=plain,
            cens=cens,
            u=u,
            u1=u1,
            u2=u2,
            sig_u=su,
            combos=combos,
            part=part,
            groups=gs,
            d=dd,
        )


if __name__ == "__main__":
    main()
