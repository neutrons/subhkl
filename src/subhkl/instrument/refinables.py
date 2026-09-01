"""Shared refinable-geometry parameterizations.

The peak-list refinement (subhkl.optimization.VectorizedObjective) and the
matching-free spherical refinement (subhkl.search.spherical) refine the same
physical quantities -- detector panel placement, goniometer offsets, crystal
orientation and cell -- against different objectives.  What must not differ
between them is what a parameter *means*: a bank translation here is a bank
translation there, same modes, same bounds convention, same composition
order.  This module holds those parameterizations as pure jax functions so
both code paths share one definition.

Conventions (inherited from VectorizedObjective, unchanged):

- normalized parameters live in [0, 1]; ``forward_map_param`` maps them to
  physical units as ``norm * 2 * bound - bound`` (0.5 is nominal);
- detector modes compose in the fixed order radial, cylindrical, area,
  axial_stretch, global_rot, global_rot_axis, global_trans, independent;
- peak positions on a refined panel are ``center + u_off * uhat +
  v_off * vhat`` with u_off/v_off the *physical* offsets from the panel
  center in meters (see commands.py), so pixel logic never enters the
  refinement loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def forward_map_param(norm, bound):
    """[0, 1] -> [-bound, +bound], 0.5 -> 0.  [units of bound]"""
    return norm * 2.0 * bound - bound


def rotation_matrix_from_axis_angle_jax(axis, angle_rad):
    u = axis / jnp.linalg.norm(axis)
    ux, uy, uz = u
    K = jnp.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    eye = jnp.eye(3)
    return eye + s[..., None, None] * K + (1.0 - c)[..., None, None] * (K @ K)


def rotation_matrix_from_rodrigues_jax(w):
    """exp([w]x) as VectorizedObjective has always computed it.

    NOTE the gradient of jnp.linalg.norm is NaN at w = 0 exactly; the
    population optimizer never lands there, but any gradient-based caller
    starting from the nominal geometry does.  Use rodrigues_safe_jax for
    gradient-based refinement.
    """
    theta = jnp.linalg.norm(w) + 1e-9
    k = w / theta
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    eye = jnp.eye(3)
    return eye + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)


def rodrigues_safe_jax(v):
    """exp([v]x), autodiff-safe at v = 0.

    The naive form has NaN gradients at the origin -- which is exactly
    where every refinement starts -- through 0/0 in sin(th)/th and the
    norm's own derivative.  The standard fix: Taylor branches near zero,
    with *safe inputs to the exact branch* so the untaken branch cannot
    poison the gradient through jnp.where.
    """
    th2 = v @ v
    small = th2 < 1e-8
    th = jnp.sqrt(jnp.where(small, 1.0, th2))
    A = jnp.where(small, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    B = jnp.where(
        small, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / jnp.where(small, 1.0, th2)
    )
    K = jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return jnp.eye(3) + A * K + B * (K @ K)


def detector_mode_slices(modes, n_banks):
    """Parameter layout of the detector mode chain -- one definition for
    both refinement paths (extracted verbatim from VectorizedObjective, so
    a --detector-modes string means the same parameters to the peak-list
    and the spherical refinement alike).

    Returns ({mode: slice}, total_parameter_count)."""
    slices = {}
    n = 0
    for mode in modes:
        if mode in ("radial", "cylindrical", "axial_stretch", "area"):
            size = 1
        elif mode == "global_rot":
            size = 3
        elif mode == "global_rot_axis":
            size = 1
        elif mode == "global_trans":
            size = 3
        elif mode == "independent":
            size = n_banks * 6
        else:
            raise ValueError(f"Unknown detector refinement mode: {mode}")
        slices[mode] = slice(n, n + size)
        n += size
    return slices, n


def apply_detector_modes(
    det_params,
    centers,
    uhats,
    vhats,
    widths,
    heights,
    modes,
    param_slices,
    bounds,
    cylinder_axis=None,
    global_rot_axis=None,
):
    """Perturbed panel geometry from normalized parameters, batched over S.

    Extracted verbatim from VectorizedObjective._get_physical_params_jax so
    that the spherical path applies *exactly* the modes the peak-list path
    refines.  ``det_params`` is (S, P) normalized; centers/uhats/vhats are
    (S, n_banks, 3); widths/heights (S, n_banks).  Returns
    (centers, uhats, vhats, widths, heights, area_scale).
    """
    if cylinder_axis is None and ("cylindrical" in modes or "axial_stretch" in modes):
        # without this the mode dies deep inside jax as `c * None`
        raise ValueError(
            "the 'cylindrical'/'axial_stretch' detector modes scale panel "
            "centers perpendicular/parallel to a cylinder axis, so they need "
            "one: pass cylinder_axis=(x, y, z) (the classic path's default is "
            "the vertical (0, 1, 0); spherical-index mirrors it, "
            "--cylinder-axis overrides)."
        )
    if global_rot_axis is None and "global_rot_axis" in modes:
        raise ValueError(
            "the 'global_rot_axis' detector mode rotates the whole assembly "
            "about a named axis: pass global_rot_axis=(x, y, z) "
            "(--det-global-rot-axis; the classic default is (0, 1, 0))."
        )
    c, u, v, w, h = centers, uhats, vhats, widths, heights
    S = det_params.shape[0]
    area_scale = jnp.zeros((S, 1))

    if "radial" in modes:
        scale = forward_map_param(
            det_params[:, param_slices["radial"]], bounds["radial"]
        )
        c = c * (1.0 + scale[:, :, None])

    if "cylindrical" in modes:
        scale = forward_map_param(
            det_params[:, param_slices["cylindrical"]], bounds["radial"]
        )
        c_dot_a = jnp.sum(c * cylinder_axis, axis=-1, keepdims=True)
        c_parallel = c_dot_a * cylinder_axis
        c_perp = c - c_parallel
        c = c_parallel + c_perp * (1.0 + scale[:, :, None])

    if "area" in modes:
        scale = forward_map_param(det_params[:, param_slices["area"]], bounds["area"])
        area_scale = scale
        # Pin the physical center of the panel: shift the corner backwards
        # to counteract the area expansion.
        c = (
            c
            - (w / 2.0)[:, :, None] * scale[:, :, None] * u
            - (h / 2.0)[:, :, None] * scale[:, :, None] * v
        )
        w = w * (1.0 + scale)
        h = h * (1.0 + scale)

    if "axial_stretch" in modes:
        scale = forward_map_param(
            det_params[:, param_slices["axial_stretch"]], bounds["radial"]
        )
        c_dot_a = jnp.sum(c * cylinder_axis, axis=-1, keepdims=True)
        c_parallel = c_dot_a * cylinder_axis
        c_perp = c - c_parallel
        c = c_parallel * (1.0 + scale[:, :, None]) + c_perp

    if "global_rot" in modes:
        rot_vec = forward_map_param(
            det_params[:, param_slices["global_rot"]], bounds["global_rot"]
        )
        # rodrigues_safe_jax, not the historical variant: identical values
        # (the historical form biases theta by +1e-9), but with gradients
        # defined at zero rotation -- which is where every gradient-based
        # refinement starts.  The population optimizer never differentiates,
        # so this is behavior-identical for the peak-list path.
        R_global = jax.vmap(rodrigues_safe_jax)(rot_vec)
        c = jnp.einsum("sij,snj->sni", R_global, c)
        u = jnp.einsum("sij,snj->sni", R_global, u)
        v = jnp.einsum("sij,snj->sni", R_global, v)

    if "global_rot_axis" in modes:
        angle_rad = forward_map_param(
            det_params[:, param_slices["global_rot_axis"]].reshape(-1),
            bounds["global_rot_axis"],
        )
        R_global = jax.vmap(rotation_matrix_from_axis_angle_jax, in_axes=(None, 0))(
            global_rot_axis, angle_rad
        )
        c = jnp.einsum("sij,snj->sni", R_global, c)
        u = jnp.einsum("sij,snj->sni", R_global, u)
        v = jnp.einsum("sij,snj->sni", R_global, v)

    if "global_trans" in modes:
        trans_vec = forward_map_param(
            det_params[:, param_slices["global_trans"]], bounds["global_trans"]
        )
        c = c + trans_vec[:, None, :]

    if "independent" in modes:
        indep = det_params[:, param_slices["independent"]]
        n_banks = centers.shape[1]
        t_vec = forward_map_param(
            indep[:, : n_banks * 3].reshape(-1, n_banks, 3),
            bounds["independent_trans"],
        )
        c = c + t_vec
        r_vec = forward_map_param(
            indep[:, n_banks * 3 :].reshape(-1, n_banks, 3),
            bounds["independent_rot"],
        )
        R_local = jax.vmap(jax.vmap(rodrigues_safe_jax))(r_vec)
        u = jnp.einsum("snij,snj->sni", R_local, u)
        v = jnp.einsum("snij,snj->sni", R_local, v)

    return c, u, v, w, h, area_scale


def peak_lab_xyz(centers, uhats, vhats, det_idx, u_off, v_off):
    """Peak lab positions on (possibly perturbed) panels, batched over S.

    xyz = center[bank] + u_off * uhat[bank] + v_off * vhat[bank], with
    u_off/v_off physical offsets from the panel center [m] (the convention
    commands.py stores: (xyz - center) . uhat at nominal geometry).
    Returns (S, n_peaks, 3).
    """
    c = centers[:, det_idx, :]
    u = uhats[:, det_idx, :]
    v = vhats[:, det_idx, :]
    return c + u_off[None, :, None] * u + v_off[None, :, None] * v


def gonio_rotation_jax(axes, angles_deg, offsets_deg, axis_dirs=None):
    """Composed goniometer rotation with per-axis zero offsets.

    R = prod_k R(axis_k, angle_k + offset_k), outermost axis first --
    matching subhkl.instrument.goniometer.sample_to_lab's composition.
    ``axes`` (K, 3 or 4; direction multiplier honored),
    ``angles_deg``/``offsets_deg`` (K,).  ``axis_dirs`` (K, 3), when
    given, overrides the axis DIRECTIONS (e.g. tilted by an axis-vector
    calibration) while the multiplier still comes from ``axes``.

    Identifiability reminders, all measured elsewhere in this codebase: a
    zero offset on axis k is pure gauge with the crystal orientation
    whenever every axis inner to k holds constant angles across the
    pooled runs (the innermost always is), and a tilt of an axis whose
    own angle never varies folds into the constant composition the same
    way.
    """
    R = jnp.eye(3)
    for k in range(axes.shape[0]):
        ax = (
            jnp.asarray(axis_dirs[k])
            if axis_dirs is not None
            else jnp.asarray(axes[k][:3])
        )
        mult = axes[k][3] if len(axes[k]) > 3 else 1.0
        Rk = rotation_matrix_from_axis_angle_jax(
            ax, jnp.deg2rad((angles_deg[k] + offsets_deg[k]) * mult)
        )
        R = R @ Rk
    return R


def axis_tilt_frames(axes):
    """Per-axis orthonormal (e1, e2) perpendicular to each nominal axis --
    the tilt basis of the classic axis-vector refinement: the refined
    direction is R(t1 e1 + t2 e2) axis, two bounded angles per axis."""
    frames = []
    for k in range(axes.shape[0]):
        a = np.asarray(axes[k][:3], dtype=float)
        a = a / np.linalg.norm(a)
        h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(a, h)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        frames.append((np.asarray(e1), np.asarray(e2)))
    return frames
