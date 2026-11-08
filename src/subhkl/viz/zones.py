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
    """Lab position -> (azimuth [deg], height [m]) of the cylindrical unrolled view."""
    roty = np.rad2deg(np.arctan2(xyz[..., 0], xyz[..., 2]))
    return roty, xyz[..., 1]


def _spherical_angles(xyz):
    """Lab position or direction -> (azimuth [deg], elevation [deg]) of the spherical view."""
    v = np.asarray(xyz, dtype=float)
    r = np.linalg.norm(v, axis=-1)
    roty = np.rad2deg(np.arctan2(v[..., 0], v[..., 2]))
    el = np.rad2deg(
        np.arcsin(np.clip(v[..., 1] / np.maximum(r, 1e-12), -1.0, 1.0))
    )
    return roty, el


def stereographic_projection(dirs, pole=(0.0, 0.0, 1.0)):
    """Project unit directions on S^2 onto the stereographic disk (sx, sy, hemisphere).

    Parameters
    ----------
    dirs : array-like of shape (..., 3)
        3D vectors (automatically normalized along the last axis).
    pole : length-3 sequence
        Reference pole direction (default +z, i.e., incident beam axis).
        Points in the +pole hemisphere map inside the unit disk r <= 1;
        antipodal points are projected from the opposite pole with hemisphere = -1.

    Returns
    -------
    sx, sy : ndarray of shape (...)
        Stereographic coordinates in [-1, 1] inside the unit disk.
    hemisphere : ndarray of shape (...), int8
        +1 for points with dir . pole >= 0, -1 for dir . pole < 0.
    """
    v = np.asarray(dirs, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_hat = np.divide(v, np.maximum(norm, 1e-12))

    p = np.asarray(pole, dtype=float)
    p = p / np.linalg.norm(p)

    # Build a right-handed orthonormal frame (e1, e2, p)
    if np.allclose(p, [0.0, 0.0, 1.0]):
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
    elif np.allclose(p, [0.0, 0.0, -1.0]):
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, -1.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0]) if abs(p[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(up, p)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(p, e1)

    x = v_hat @ e1
    y = v_hat @ e2
    z = v_hat @ p

    hemi = np.where(z >= 0.0, 1, -1).astype(np.int8)
    denom = 1.0 + np.abs(z)
    sx = x / np.maximum(denom, 1e-12)
    sy = y / np.maximum(denom, 1e-12)
    return sx, sy, hemi


def get_zone_conic_points(
    t_lab,
    ki=(0.0, 0.0, 1.0),
    n_arc=360,
):
    """Compute great-circle scattering directions G_hat and Laue diffracted rays kf_hat for a zone axis.

    Parameters
    ----------
    t_lab : array-like of shape (3,)
        Zone axis vector in lab frame.
    ki : array-like of shape (3,)
        Incident beam direction.
    n_arc : int
        Number of angular samples along the great circle.

    Returns
    -------
    dict with keys:
        "t_hat": (3,) unit zone axis in lab frame
        "g_hat": (M, 3) unit scattering vectors in the diffracting hemisphere (G . ki < 0)
        "kf_hat": (M, 3) unit diffracted ray directions
        "sx", "sy", "hemisphere": stereographic projection of g_hat
        "det_x", "det_y": unrolled detector coordinates (azimuth [deg], y [m] at unit cylinder R=1m)
    """
    t_arr = np.asarray(t_lab, dtype=float)
    t_norm = np.linalg.norm(t_arr)
    if t_norm < 1e-12:
        raise ValueError("Zone axis vector t_lab must be non-zero")
    t_hat = t_arr / t_norm

    ki_hat = np.asarray(ki, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)

    e1 = np.cross(t_hat, ki_hat)
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(t_hat, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(t_hat, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t_hat, e1)

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_arc), endpoint=False)
    G = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
    keep = (G @ ki_hat) < -1e-3
    G = G[keep]
    kf = ki_hat[None, :] - 2.0 * (G @ ki_hat)[:, None] * G
    kf_norm = np.linalg.norm(kf, axis=1, keepdims=True)
    kf = kf / np.maximum(kf_norm, 1e-12)

    sx, sy, hemi = stereographic_projection(G)
    # Cylindrical projection at unit radius R_cyl = 1.0 m for detector-agnostic conic
    horiz = np.hypot(kf[:, 0], kf[:, 2])
    det_x = np.rad2deg(np.arctan2(kf[:, 0], kf[:, 2]))
    det_y = kf[:, 1] / np.maximum(horiz, 0.15)
    det_el = np.rad2deg(np.arcsin(np.clip(kf[:, 1], -1.0, 1.0)))

    return {
        "t_hat": t_hat,
        "g_hat": G,
        "kf_hat": kf,
        "sx": sx,
        "sy": sy,
        "hemisphere": hemi,
        "det_x": det_x,
        "det_y": det_y,
        "det_el": det_el,
    }


def identify_zones(
    peaks_xyz,
    U=None,
    B=None,
    ub_matrix=None,
    R_gonio=None,
    ki=(0.0, 0.0, 1.0),
    max_index=2,
    tolerance_deg=0.75,
    min_peaks=3,
    n_arc=360,
):
    """Identify populated Laue zones from observed peak positions or scattering vectors.

    Parameters
    ----------
    peaks_xyz : array-like of shape (N, 3)
        Lab-frame detector positions (or scattering directions if already unit vectors)
        of observed peaks.
    U, B : (3, 3) arrays, optional
        Orientation and B matrices. Alternatively pass `ub_matrix = U @ B`.
    ub_matrix : (3, 3) array, optional
        Combined UB matrix. If neither (U, B) nor ub_matrix is supplied, candidate
        zone axes are discovered blind via cross-product great-circle clustering on S^2.
    R_gonio : (3, 3) or (N, 3, 3) array, optional
        Goniometer rotation matrix.
    ki : length-3 sequence
        Incident beam direction (default (0, 0, 1)).
    max_index : int
        Maximum Miller zone index |u|, |v|, |w| <= max_index when UB is provided.
    tolerance_deg : float
        Angular half-width [deg] for assigning a peak to a zone plane |g_hat . t_hat| <= sin(tol).
    min_peaks : int
        Minimum number of captured peaks required to include a zone.
    n_arc : int
        Number of points along each zone arc.

    Returns
    -------
    list of dict
        Sorted by captured peak count (descending), then zone norm (ascending).
    """
    pts = np.asarray(peaks_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("peaks_xyz must have shape (N, 3)")

    ki_hat = np.asarray(ki, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)

    # Convert lab positions to unit diffracted rays kf_hat and scattering directions g_hat
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    kf_hat = pts / np.maximum(norms, 1e-12)
    q_vec = kf_hat - ki_hat[None, :]
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
    # If points were already passed as reciprocal vectors G with G . ki < 0 and |pts| ~ 1,
    # check whether pts are lab detector positions or direct G vectors:
    g_hat = np.where(q_norm > 1e-6, q_vec / np.maximum(q_norm, 1e-12), kf_hat)

    if ub_matrix is None and U is not None and B is not None:
        ub_matrix = np.asarray(U, dtype=float) @ np.asarray(B, dtype=float)

    R_single = np.eye(3)
    if R_gonio is not None:
        R_arr = np.asarray(R_gonio, dtype=float)
        if R_arr.ndim == 2:
            R_single = R_arr
            g_crystal = g_hat @ R_single
        elif R_arr.ndim == 3 and len(R_arr) == len(g_hat):
            g_crystal = np.einsum("ni,nij->nj", g_hat, R_arr)
            R_single = R_arr[0]
        else:
            g_crystal = g_hat
    else:
        g_crystal = g_hat

    sin_tol = np.sin(np.deg2rad(float(tolerance_deg)))
    zones = []

    if ub_matrix is not None:
        UB = np.asarray(ub_matrix, dtype=float)
        # Since G_crystal = UB @ hkl, the dual direct-lattice basis vectors in crystal
        # frame (satisfying G_crystal . t_crystal = hkl . uvw) are the rows of inv(UB),
        # which equals (U @ inv(B).T).T when U is orthogonal.
        A_real = np.linalg.inv(UB)
        uvw_list = zone_axes_uvw(max_index=max_index)

        for zi, z in enumerate(uvw_list):
            t_cryst = z @ A_real
            d_direct = float(np.linalg.norm(t_cryst))
            if d_direct < 1e-12:
                continue
            t_cryst_hat = t_cryst / d_direct
            t_lab_hat = R_single @ t_cryst_hat

            dots = np.abs(g_crystal @ t_cryst_hat)
            ang_deg = np.rad2deg(np.arcsin(np.clip(dots, 0.0, 1.0)))
            member_mask = dots <= sin_tol
            member_ids = np.where(member_mask)[0]
            if len(member_ids) < min_peaks:
                continue

            rms_deg = float(np.sqrt(np.mean(ang_deg[member_ids] ** 2))) if len(member_ids) else 0.0
            score = float(np.sum(np.exp(-0.5 * (ang_deg[member_ids] / max(tolerance_deg, 1e-3)) ** 2)))
            conic = get_zone_conic_points(t_lab_hat, ki=ki_hat, n_arc=n_arc)

            zones.append(
                {
                    "zone_id": int(zi),
                    "uvw": [int(z[0]), int(z[1]), int(z[2])],
                    "label": f"[{int(z[0])} {int(z[1])} {int(z[2])}]",
                    "axis_xyz": [float(x) for x in t_lab_hat],
                    "axis_crystal_xyz": [float(x) for x in t_cryst_hat],
                    "d_direct": round(d_direct, 4),
                    "score": round(score, 3),
                    "n_peaks": int(len(member_ids)),
                    "angular_rms_deg": round(rms_deg, 4),
                    "member_peak_ids": [int(i) for i in member_ids],
                    "conic": conic,
                }
            )
    else:
        # Blind discovery of great-circle zones from pairwise cross products of g_crystal
        n_pts = len(g_crystal)
        if n_pts >= max(2, min_peaks):
            max_pairs = min(n_pts, 64)
            idx_a, idx_b = np.triu_indices(max_pairs, k=1)
            cross = np.cross(g_crystal[idx_a], g_crystal[idx_b])
            cnorm = np.linalg.norm(cross, axis=1)
            valid = cnorm > 0.15
            cand = cross[valid] / cnorm[valid, None]
            # Canonicalize sign
            lead = np.where(
                np.abs(cand[:, 0]) > 1e-6,
                cand[:, 0],
                np.where(np.abs(cand[:, 1]) > 1e-6, cand[:, 1], cand[:, 2]),
            )
            cand = cand * np.where(lead < 0, -1.0, 1.0)[:, None]

            used_axes = []
            dots_all = np.abs(cand @ g_crystal.T)
            counts = np.sum(dots_all <= sin_tol, axis=1)
            order = np.argsort(-counts)
            for idx in order:
                if counts[idx] < min_peaks:
                    break
                t_cand = cand[idx]
                if any(abs(np.dot(t_cand, u)) > np.cos(np.deg2rad(3.0)) for u in used_axes):
                    continue
                # Refine normal via SVD of inlier g_crystal vectors
                inliers = dots_all[idx] <= sin_tol
                if np.sum(inliers) >= 3:
                    _, _, vh = np.linalg.svd(g_crystal[inliers], full_matrices=False)
                    t_cand = vh[-1]
                    t_cand /= np.linalg.norm(t_cand)
                used_axes.append(t_cand)
                dots = np.abs(g_crystal @ t_cand)
                ang_deg = np.rad2deg(np.arcsin(np.clip(dots, 0.0, 1.0)))
                member_ids = np.where(dots <= sin_tol)[0]
                if len(member_ids) < min_peaks:
                    continue
                t_lab_hat = R_single @ t_cand
                rms_deg = float(np.sqrt(np.mean(ang_deg[member_ids] ** 2)))
                score = float(
                    np.sum(np.exp(-0.5 * (ang_deg[member_ids] / max(tolerance_deg, 1e-3)) ** 2))
                )
                conic = get_zone_conic_points(t_lab_hat, ki=ki_hat, n_arc=n_arc)
                zi = len(zones)
                zones.append(
                    {
                        "zone_id": int(zi),
                        "uvw": [0, 0, zi + 1],
                        "label": f"Zone #{zi + 1}",
                        "axis_xyz": [float(x) for x in t_lab_hat],
                        "axis_crystal_xyz": [float(x) for x in t_cand],
                        "d_direct": 0.0,
                        "score": round(score, 3),
                        "n_peaks": int(len(member_ids)),
                        "angular_rms_deg": round(rms_deg, 4),
                        "member_peak_ids": [int(i) for i in member_ids],
                        "conic": conic,
                    }
                )
                if len(zones) >= 20:
                    break

    zones.sort(key=lambda z: (-z["n_peaks"], -z["score"], sum(x * x for x in z["uvw"])))
    for new_id, z in enumerate(zones):
        z["zone_id"] = new_id
    return zones


def _cell_from_B(B):
    G_star = np.asarray(B, dtype=float).T @ np.asarray(B, dtype=float)
    G_real = np.linalg.inv(G_star)
    a = float(np.sqrt(G_real[0, 0]))
    b = float(np.sqrt(G_real[1, 1]))
    c = float(np.sqrt(G_real[2, 2]))
    alpha = float(np.degrees(np.arccos(np.clip(G_real[1, 2] / (b * c), -1.0, 1.0))))
    beta = float(np.degrees(np.arccos(np.clip(G_real[0, 2] / (a * c), -1.0, 1.0))))
    gamma = float(np.degrees(np.arccos(np.clip(G_real[0, 1] / (a * b), -1.0, 1.0))))
    return a, b, c, alpha, beta, gamma


def zone_curve_points(
    detectors,
    U,
    B,
    R_gonio=None,
    ki=(0.0, 0.0, 1.0),
    max_index=1,
    n_arc=1440,
    label_max=13,
    sample_origin=None,
    wavelength=None,
    d_min=None,
    space_group="P 1",
    sigma_px=4.0,
):
    """Per-panel lab positions and Bragg reflection circles of low-index zones.

    detectors: {key: Detector} -- the returned entries reuse the same keys,
    so the caller can feed panels keyed by bank (the standalone plot) or by
    image index (the unrolled replay layout) alike.  Returns a list of
    {"label", "color", "points": {key: (Ni, 3) lab xyz},
     "peaks": {key: (Mi, 3) (row, col, sigma_px)}} per zone [uvw].
    """
    from subhkl.core.crystallography import generate_reflections
    from subhkl.instrument.detector import Detector

    origin = np.zeros(3) if sample_origin is None else np.asarray(sample_origin, float)
    projected = {
        key: Detector(dict(det.config, center=det.center - origin))
        for key, det in detectors.items()
    }
    R = np.eye(3) if R_gonio is None else np.asarray(R_gonio, dtype=float)
    ki_hat = np.asarray(ki, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)
    B_arr = np.asarray(B, dtype=float)
    U_arr = np.asarray(U, dtype=float)
    A_real = np.linalg.inv(B_arr).T  # rows a, b, c

    # Also compute discrete in-band Bragg reflections per zone [uvw]
    wl = (
        (2.0, 4.5)
        if wavelength is None
        else (float(wavelength[0]), float(wavelength[1]))
    )
    a, b, c, alpha, beta, gamma = _cell_from_B(B_arr)
    min_cell = min(a, b, c)
    # Cap visualization resolution so ultra-fine d_min never creates thousands of overlapping circles
    d_cut = float(d_min) if d_min is not None else max(2.0, min_cell / 25.0)
    d_cut = max(d_cut, min_cell / 35.0)
    h, k, l_ = generate_reflections(
        a, b, c, alpha, beta, gamma, space_group=str(space_group), d_min=d_cut
    )
    hkl = np.stack([h, k, l_], axis=1)
    G_lab = (hkl @ (U_arr @ B_arr).T) @ R.T
    g2 = np.sum(G_lab * G_lab, axis=1)
    valid_g = (g2 > 1e-12) & ((G_lab @ ki_hat) < -1e-6)
    lam = np.where(valid_g, -2.0 * (G_lab @ ki_hat) / np.maximum(g2, 1e-12), -1.0)
    in_band = valid_g & (lam >= wl[0]) & (lam <= wl[1])
    hkl_band = hkl[in_band]
    kf_band = ki_hat[None, :] + lam[in_band, None] * G_lab[in_band]
    kf_band = kf_band / np.linalg.norm(kf_band, axis=1, keepdims=True)
    # Assign each reflection to its lowest-index zone [uvw] so circles do not
    # overplot multiple colors on the same spot
    uvw = zone_axes_uvw(max_index)
    zone_dot = (
        (hkl_band @ uvw.T == 0) if len(hkl_band) else np.zeros((0, len(uvw)), bool)
    )
    assigned_zone = np.where(np.any(zone_dot, axis=1), np.argmax(zone_dot, axis=1), -1)

    cmap = plt.get_cmap("tab10")
    theta = np.linspace(0.0, 2.0 * np.pi, n_arc, endpoint=False)
    out = []
    for zi, z in enumerate(uvw):
        t = z @ A_real
        t_lab = R @ (U_arr @ (t / np.linalg.norm(t)))
        e1 = np.cross(t_lab, ki_hat)
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(t_lab, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(t_lab, e1)
        G = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
        keep = G @ ki_hat < -1e-3  # diffracting branch, off the direct beam
        G = G[keep]
        kf = ki_hat[None, :] - 2.0 * (G @ ki_hat)[:, None] * G
        points = {}
        peaks = {}
        m_z = assigned_zone == zi
        kf_z = kf_band[m_z]
        for key, det in detectors.items():
            mask, row, col = projected[key].reflections_mask(
                kf[:, 0], kf[:, 1], kf[:, 2]
            )
            if np.any(mask):
                points[key] = det.pixel_to_lab(row[mask], col[mask])
            if len(kf_z):
                pmask, prow, pcol = projected[key].reflections_mask(
                    kf_z[:, 0], kf_z[:, 1], kf_z[:, 2]
                )
                if np.any(pmask):
                    r_hit = np.asarray(prow[pmask], float)
                    c_hit = np.asarray(pcol[pmask], float)
                    s_hit = np.full(len(r_hit), float(sigma_px), dtype=float)
                    peaks[key] = np.column_stack([r_hit, c_hit, s_hit])
        out.append(
            {
                # a full index-2 legend is 43 entries and overflows the
                # canvas; label the low-index families (sorted first) and
                # let the rest share colors unlabeled
                "label": f"[{z[0]}{z[1]}{z[2]}]" if zi < label_max else None,
                "color": cmap(zi % 10),
                "points": points,
                "peaks": peaks,
            }
        )
    return out


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
    images=None,
    render_binning=2,
    zone_alpha=0.5,
):
    """One unrolled-detector figure: measured peaks + low-index zone conics.

    detectors: {bank: Detector}; peaks as parallel bank/row/col arrays for
    the frame being drawn; U, B the orientation and cell matrices; R_gonio
    the frame's goniometer rotation (identity if None).

    ``zone_alpha`` draws the conics semi-transparent: once a solution is
    good the curves land ON the spots, and an opaque curve then hides the
    very agreement it is drawn to show.

    With ``images`` ({bank: 2D counts array}), the raw detector data is
    rendered underneath in the same unrolled coordinates (log1p grayscale,
    pcolormesh on the true pixel corners, downsampled by render_binning for
    figure weight) -- the strongest visual check there is: the conics must
    ride the raw diffraction streaks, with no peak finder in the loop.
    """
    R = np.eye(3) if R_gonio is None else np.asarray(R_gonio, dtype=float)
    ki_hat = np.asarray(ki, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)
    A_real = np.linalg.inv(np.asarray(B, dtype=float)).T  # rows a, b, c [Angstrom]

    fig, ax = plt.subplots(figsize=(16, 6))

    if images is not None:
        vmax = max(
            (np.percentile(im, 99.8) for im in images.values() if im is not None),
            default=1.0,
        )
        for bk, im in images.items():
            det = detectors.get(int(bk))
            if det is None or im is None:
                continue
            b = int(render_binning)
            n2, m2 = (im.shape[0] // b) * b, (im.shape[1] // b) * b
            imb = im[:n2, :m2].reshape(n2 // b, b, m2 // b, b).sum(axis=(1, 3))
            rr = np.arange(0, n2 + 1, b) - 0.5
            cc = np.arange(0, m2 + 1, b) - 0.5
            RR, CC = np.meshgrid(rr, cc, indexing="ij")
            x, y = _unrolled(det.pixel_to_lab(RR, CC))
            # a panel that straddles the +-180 deg seam would smear across
            # the figure; draw it on one side
            if np.ptp(x) > 180:
                x = np.where(x < 0, x + 360.0, x)
            ax.pcolormesh(
                x,
                y,
                np.log1p(imb),
                cmap="gray_r",
                vmin=0.0,
                vmax=np.log1p(vmax * b * b),
                zorder=0,
                shading="flat",
                rasterized=True,
            )

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
                alpha=zone_alpha,
                zorder=3,
                label=f"[{z[0]}{z[1]}{z[2]}]" if first else None,
            )
            first = False

    ax.set_xlabel("detector azimuth atan2(x, z) [deg]")
    ax.set_ylabel("lab y [m]")
    if title:
        ax.set_title(title)
    leg = ax.legend(loc="upper right", fontsize=7, markerscale=8, ncol=2)
    for lh in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        lh.set_alpha(1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_name) or ".", exist_ok=True)
    fig.savefig(out_name, dpi=dpi)
    plt.close(fig)
    return out_name
