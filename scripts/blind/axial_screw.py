"""Prototype: screw axis along a cell axis from raw events.

The reflections 00l all lie on one ray, so they share a detector position
and differ only in wavelength: at that spot the time-of-flight spectrum is
the one-dimensional diffraction pattern along c*.  Converting every event
near the c* axis to a fractional index l = c . q gives that pattern
directly.  A screw axis shows as the period of the pattern in l: every l
for a pure rotation, every 2nd, 3rd or 6th for 6_3, 6_2/6_4, 6_1/6_5.

No reflection is integrated.  A parallel cylinder between the lattice rows
(h = k = 1/2) gives the background.

Usage:
    python axial_screw.py merged.h5 MANDI refined_solve.h5 [--axis 2] [--radius 0.12]
"""

import argparse

import h5py
import numpy as np
from blind_cell import EVENT_PAD, pixel_positions

from subhkl.config import reduction_settings
from subhkl.io.events import ANGSTROM_M_PER_US, EventStore

L_BINS_PER_UNIT = 20
L_MAX = 160


def _axial_impl(hist, tof, pix, pos, to_hkl, l1, axis, radius, centre):
    """Add one image's events within ``radius`` of the row through ``centre``
    (fractional indices of the two other axes) to the histogram of the
    fractional index along ``axis``."""
    import jax.numpy as jnp

    xyz = pos[pix]
    l2 = jnp.linalg.norm(xyz, axis=1)
    lam = ANGSTROM_M_PER_US * tof / (l1 + l2)
    q_lab = (xyz / l2[:, None] - jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)) / (
        jnp.maximum(lam, 1e-6)[:, None]
    )
    hkl = q_lab @ to_hkl.T  # fractional Miller indices
    other = jnp.array([i for i in range(3) if i != axis])
    off = hkl[:, other] - centre[None, :]
    near = jnp.sum(off**2, axis=1) < radius**2
    along = jnp.abs(hkl[:, axis])
    b = jnp.floor(along * L_BINS_PER_UNIT).astype(jnp.int32)
    ok = (tof > 0) & near & (b >= 0) & (b < L_MAX * L_BINS_PER_UNIT)
    return hist.at[jnp.where(ok, b, L_MAX * L_BINS_PER_UNIT)].add(1, mode="drop")


def axial_patterns(merged, instrument, solve, axis, radius):
    import jax
    import jax.numpy as jnp

    l1 = float(reduction_settings[instrument]["L1"])
    with h5py.File(solve, "r") as f:
        ub = f["sample/U"][()] @ f["sample/B"][()]
        rot = f["goniometer/R"][()]
    with h5py.File(merged, "r") as f:
        images = f["images"][()]
        bank_ids = f["bank_ids"][()]
    fn = jax.jit(_axial_impl, static_argnames=["axis"])
    n_bins = L_MAX * L_BINS_PER_UNIT
    on = jnp.zeros(n_bins, dtype=jnp.int32)
    offp = jnp.zeros(n_bins, dtype=jnp.int32)
    pos_cache = {}
    with EventStore(merged) as store:
        for i in range(len(images)):
            bank = int(bank_ids[i])
            if bank not in pos_cache:
                pos_cache[bank] = jnp.array(
                    pixel_positions(instrument, bank, images[i].shape)
                )
            tof, offsets = store.image_events(i, images[i])
            pix = np.repeat(np.arange(images[i].size, dtype=np.int32), np.diff(offsets))
            n_pad = max(EVENT_PAD, -(-len(tof) // EVENT_PAD) * EVENT_PAD)
            tof_p = np.zeros(n_pad, dtype=np.float32)
            pix_p = np.zeros(n_pad, dtype=np.int32)
            tof_p[: len(tof)] = tof
            pix_p[: len(tof)] = pix
            # h = (R UB)^-1 q_lab for this frame
            to_hkl = jnp.array(np.linalg.inv(rot[i] @ ub), dtype=jnp.float32)
            args = (
                jnp.array(tof_p),
                jnp.array(pix_p),
                pos_cache[bank],
                to_hkl,
                jnp.float32(l1),
            )
            on = fn(
                on,
                *args,
                axis=axis,
                radius=jnp.float32(radius),
                centre=jnp.zeros(2, jnp.float32),
            )
            offp = fn(
                offp,
                *args,
                axis=axis,
                radius=jnp.float32(radius),
                centre=jnp.full(2, 0.5, jnp.float32),
            )
    return np.array(on, dtype=float), np.array(offp, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("instrument")
    ap.add_argument("solve")
    ap.add_argument("--axis", type=int, default=2)
    ap.add_argument("--radius", type=float, default=0.12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    on, off = axial_patterns(
        args.merged, args.instrument, args.solve, args.axis, args.radius
    )
    if args.out:
        np.savez(args.out, on=on, off=off, bins_per_unit=L_BINS_PER_UNIT)
    lcoord = (np.arange(len(on)) + 0.5) / L_BINS_PER_UNIT
    name = "hkl"[args.axis]
    used = on > 0
    print(
        f"events within {args.radius} of the {name} axis: {on.sum():.0f}; on the parallel background row: {off.sum():.0f}; {name} from {lcoord[used].min():.1f} to {lcoord[used].max():.1f}"
    )

    # Integer positions: net counts within +-0.25 of each integer index.
    n_max = int(lcoord[used].max())
    idx = np.arange(1, n_max + 1)
    net, err = np.zeros(len(idx)), np.zeros(len(idx))
    for j, n in enumerate(idx):
        m = np.abs(lcoord - n) < 0.25
        net[j] = on[m].sum() - off[m].sum()
        err[j] = np.sqrt(on[m].sum() + off[m].sum())
    covered = err > 0
    print(
        f"\nnet counts at integer {name}, grouped by {name} mod 6 (a 6_1 or 6_5 axis leaves only {name} = 6n):"
    )
    print(
        f"   {name} mod 6   reflections   net counts      sigma   net / sigma   mean per reflection"
    )
    for r in range(6):
        m = covered & (idx % 6 == r)
        tot, sig = net[m].sum(), np.sqrt((err[m] ** 2).sum())
        print(
            f"      {r}         {m.sum():5d}     {tot:10.0f}  {sig:9.0f}    {tot / sig:8.1f}       {tot / max(m.sum(), 1):10.1f}"
        )
    for period, label in ((2, "6_3"), (3, "6_2 or 6_4"), (6, "6_1 or 6_5")):
        m_abs = covered & (idx % period != 0)
        tot, sig = net[m_abs].sum(), np.sqrt((err[m_abs] ** 2).sum())
        m_pre = covered & (idx % period == 0)
        tot_p, sig_p = net[m_pre].sum(), np.sqrt((err[m_pre] ** 2).sum())
        print(
            f"   {label:11s}: reflections it forbids carry {tot:9.0f} +- {sig:6.0f} ({tot / sig:6.1f} sigma); those it allows {tot_p:9.0f} +- {sig_p:6.0f}"
        )

    # Period of the pattern: Fourier amplitudes of the net pattern at 1/period.
    netpat = on - off
    print(
        f"\nFourier amplitude of the net pattern along {name}, relative to its mean level:"
    )
    for period in (1, 1.5, 2, 3, 6):
        amp = np.abs(np.sum(netpat * np.exp(2j * np.pi * lcoord / period))) / max(
            netpat.sum(), 1.0
        )
        print(f"   period {period:>3} in {name}: {amp:.3f}")


if __name__ == "__main__":
    main()
