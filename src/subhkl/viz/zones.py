"""Low-index Laue zone overlays on the unrolled detector.

A crystallographic zone [uvw] collects the reflections G with G . t = 0,
t = u a + v b + w c: a great circle of scattering-vector directions, which
the Laue geometry maps to a conic of spot positions per panel.  Overlaying
the low-index zone conics predicted by a refined (U, B) on the measured
peak positions is the classic visual check of an indexing solution -- every
strong row of spots should ride a drawn curve, and a systematic offset
shows *which* refinable (orientation, a panel, the goniometer) is wrong,
not just that one is.

Rendering uses the same unrolled coordinates as the finder/integrator
plots: horizontal is the room-frame azimuth atan2(x, z) [deg], vertical is
the lab y of the pixel [m].  No images are needed -- peaks are drawn as
points, zones as curves -- so this works from an indexer output file alone.

The wavelength band is deliberately NOT used to clip the curves: a zone
circle fixes only the direction of G, every direction carries its harmonic
ladder, and which harmonic is in band varies along the arc.  The drawn
curve is where zone reflections CAN fall; the data shows where they do.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def zone_axes_uvw(max_index=1):
    """Primitive [uvw] up to max_index, one per +- pair, sorted low first."""
    rng = np.arange(-max_index, max_index + 1)
    u, v, w = np.meshgrid(rng, rng, rng, indexing="ij")
    uvw = np.stack([u.ravel(), v.ravel(), w.ravel()], axis=1)
    uvw = uvw[np.any(uvw != 0, axis=1)]
    g = np.gcd.reduce(np.abs(uvw), axis=1)
    uvw = uvw // g[:, None]
    lead = np.where(
        uvw[:, 0] != 0, uvw[:, 0], np.where(uvw[:, 1] != 0, uvw[:, 1], uvw[:, 2])
    )
    uvw = np.unique(uvw * np.where(lead < 0, -1, 1)[:, None], axis=0)
    order = np.argsort(np.sum(uvw * uvw, axis=1), kind="stable")
    return uvw[order]


def _unrolled(xyz):
    """Lab position -> (azimuth [deg], height [m]) of the unrolled view."""
    roty = np.rad2deg(np.arctan2(xyz[..., 0], xyz[..., 2]))
    return roty, xyz[..., 1]


def plot_zone_overlay(
    detectors,
    peak_bank,
    peak_row,
    peak_col,
    U,
    B,
    R_gonio=None,
    ki=(0.0, 0.0, 1.0),
    max_index=1,
    n_arc=1440,
    out_name="zones.png",
    dpi=150,
    title=None,
):
    """One unrolled-detector figure: measured peaks + low-index zone conics.

    detectors: {bank: Detector}; peaks as parallel bank/row/col arrays for
    the frame being drawn; U, B the orientation and cell matrices; R_gonio
    the frame's goniometer rotation (identity if None).
    """
    R = np.eye(3) if R_gonio is None else np.asarray(R_gonio, dtype=float)
    ki_hat = np.asarray(ki, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)
    A_real = np.linalg.inv(np.asarray(B, dtype=float)).T  # rows a, b, c [Angstrom]

    fig, ax = plt.subplots(figsize=(16, 6))

    # panel outlines and measured peaks
    for bk, det in detectors.items():
        rr = np.array([0, det.n - 1, det.n - 1, 0, 0])
        cc = np.array([0, 0, det.m - 1, det.m - 1, 0])
        x, y = _unrolled(det.pixel_to_lab(rr, cc))
        ax.plot(x, y, color="0.85", lw=0.5, zorder=1)
    pk = np.asarray(peak_bank)
    for bk in np.unique(pk):
        det = detectors.get(int(bk))
        if det is None:
            continue
        m = pk == bk
        x, y = _unrolled(
            det.pixel_to_lab(np.asarray(peak_row)[m], np.asarray(peak_col)[m])
        )
        ax.scatter(x, y, s=4, c="0.3", zorder=2)

    # zone conics
    uvw = zone_axes_uvw(max_index)
    cmap = plt.get_cmap("tab10")
    theta = np.linspace(0.0, 2.0 * np.pi, n_arc, endpoint=False)
    for zi, z in enumerate(uvw):
        t = z @ A_real
        t_lab = R @ (np.asarray(U, dtype=float) @ (t / np.linalg.norm(t)))
        # orthonormal frame of the zone plane
        e1 = np.cross(t_lab, ki_hat)
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(t_lab, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(t_lab, e1)
        G = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
        # diffracting branch only, and away from the direct beam
        keep = G @ ki_hat < -1e-3
        G = G[keep]
        kf = ki_hat[None, :] - 2.0 * (G @ ki_hat)[:, None] * G
        color = cmap(zi % 10)
        first = True
        for bk, det in detectors.items():
            mask, row, col = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
            if not np.any(mask):
                continue
            x, y = _unrolled(det.pixel_to_lab(row[mask], col[mask]))
            ax.scatter(
                x,
                y,
                s=0.5,
                color=color,
                zorder=3,
                label=f"[{z[0]}{z[1]}{z[2]}]" if first else None,
            )
            first = False

    ax.set_xlabel("detector azimuth atan2(x, z) [deg]")
    ax.set_ylabel("lab y [m]")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, markerscale=8, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_name) or ".", exist_ok=True)
    fig.savefig(out_name, dpi=dpi)
    plt.close(fig)
    return out_name
