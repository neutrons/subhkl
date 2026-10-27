import os
import warnings
from functools import partial


import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.scipy.linalg as jscipy_linalg
from evosax.algorithms import CMA_ES, PSO, DifferentialEvolution, GuidedES

import numpy as np
import h5py
import scipy.linalg

from subhkl.instrument.detector import scattering_vector_from_angles
from subhkl.utils import devices as device_util
from subhkl.core.spacegroup import get_space_group_object

try:
    from tqdm import trange
except ImportError:
    trange = None


def _forward_map_param(norm, bound):
    return norm * 2.0 * bound - bound


def _forward_map_lattice(norm, nominal, frac_bound):
    delta = np.abs(nominal) * frac_bound
    min_val = nominal - delta
    max_val = nominal + delta
    return min_val + norm * (max_val - min_val)


HARMONIC_AXES_MODES = ("rocking", "transverse", "full")


def harmonic_axes_from_scan(scan_dir, ki_vec, mode="rocking"):
    """Lab-frame rotation axes for the Fourier-in-phi rocking model.

    The measured steering on cg4d-t4-lysozyme puts 96% of its power on
    the single axis perpendicular to both the scan axis and the beam
    (the rocking-curve axis), so that axis alone is the default.
    "transverse" adds the second axis perpendicular to the scan axis;
    "full" adds the scan axis itself, whose harmonics represent a
    periodic drive error (its constant term stays excluded everywhere:
    that is the motor zero the global offsets already refine).
    """
    if mode not in HARMONIC_AXES_MODES:
        raise ValueError(
            f"Unknown harmonic axes mode {mode!r}; pick from {HARMONIC_AXES_MODES}."
        )
    n = np.asarray(scan_dir, dtype=float)
    n = n / np.linalg.norm(n)
    b = np.asarray(ki_vec, dtype=float)
    b = b / np.linalg.norm(b)
    e1 = np.cross(n, b)
    if np.linalg.norm(e1) < 1e-6:
        # Scan axis along the beam: any perpendicular direction serves.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(n @ ref) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, ref)
    e1 = e1 / np.linalg.norm(e1)
    if mode == "rocking":
        return np.array([e1])
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    if mode == "transverse":
        return np.array([e1, e2])
    return np.array([e1, e2, n])


def harmonic_rocking_vectors(angles_deg, axes, orders, coeffs_deg):
    """Per-frame Rodrigues vectors (radians) of the Fourier rocking.

    delta_k(phi) = sum_m a_km cos(m phi) + b_km sin(m phi) about lab
    axis e_k, with coeffs_deg shaped (n_axes, n_orders, 2) as (a, b).
    """
    ang = np.asarray(angles_deg, dtype=float)
    orders_np = np.asarray(list(orders), dtype=float)
    phase = np.deg2rad(orders_np[None, :] * ang[:, None])  # (F, n)
    c = np.asarray(coeffs_deg, dtype=float)  # (K, n, 2)
    delta = np.einsum("kn,fn->kf", c[..., 0], np.cos(phase)) + np.einsum(
        "kn,fn->kf", c[..., 1], np.sin(phase)
    )
    return np.deg2rad(np.einsum("kf,kd->fd", delta, np.asarray(axes, dtype=float)))


def harmonic_rocking_matrices(angles_deg, axes, orders, coeffs_deg):
    """Per-frame 3x3 rocking rotations, applied to lab-frame q vectors."""
    from scipy.spatial.transform import Rotation

    w = harmonic_rocking_vectors(angles_deg, axes, orders, coeffs_deg)
    return Rotation.from_rotvec(w).as_matrix()


def _get_active_lattice_indices(lattice_system):
    if lattice_system == "Cubic":
        return [0]
    if lattice_system in ("Hexagonal", "Tetragonal"):
        return [0, 2]
    if lattice_system == "Rhombohedral":
        return [0, 3]
    if lattice_system == "Orthorhombic":
        return [0, 1, 2]
    if lattice_system == "Monoclinic":
        return [0, 1, 2, 4]
    return [0, 1, 2, 3, 4, 5]


def get_lattice_system(
    a, b, c, alpha, beta, gamma, space_group_name, atol_len=0.05, atol_ang=0.5
):
    try:
        sg = get_space_group_object(space_group_name)
        sys_str = str(sg.crystal_system()).split(".")[-1].lower()
        centering = sg.centring_type()
    except Exception:
        sys_str = "triclinic"
        centering = "P"

    expected = "Triclinic"
    if sys_str == "cubic":
        expected = "Cubic"
    elif sys_str == "hexagonal":
        expected = "Hexagonal"
    elif sys_str == "trigonal":
        expected = "Rhombohedral" if centering == "R" else "Hexagonal"
    elif sys_str == "tetragonal":
        expected = "Tetragonal"
    elif sys_str == "orthorhombic":
        expected = "Orthorhombic"
    elif sys_str == "monoclinic":
        expected = "Monoclinic"

    def is_90(x):
        return np.isclose(x, 90.0, atol=atol_ang)

    def is_120(x):
        return np.isclose(x, 120.0, atol=atol_ang)

    def eq(x, y):
        return np.isclose(x, y, atol=atol_len)

    violation_msg = []
    if expected == "Cubic":
        if not (eq(a, b) and eq(b, c)):
            violation_msg.append("a=b=c")
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)):
            violation_msg.append("angles=90")
    elif expected == "Hexagonal":
        if not eq(a, b):
            violation_msg.append("a=b")
        if not (is_90(alpha) and is_90(beta) and is_120(gamma)):
            violation_msg.append("angles=90,90,120")
    elif expected == "Rhombohedral":
        if not (eq(a, b) and eq(b, c)):
            violation_msg.append("a=b=c")
        if not (eq(alpha, beta) and eq(beta, gamma)):
            violation_msg.append("alpha=beta=gamma")
    elif expected == "Tetragonal":
        if not eq(a, b):
            violation_msg.append("a=b")
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)):
            violation_msg.append("angles=90")
    elif expected == "Orthorhombic":
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)):
            violation_msg.append("angles=90")
    elif expected == "Monoclinic":
        count90 = sum([is_90(alpha), is_90(beta), is_90(gamma)])
        if count90 < 2:
            violation_msg.append("at least two angles=90")

    if violation_msg:
        warnings.warn(
            f"\n[Lattice System] Input parameters violate {space_group_name} ({expected}) constraints: {', '.join(violation_msg)}.\n"
            f"optimization will enforce {expected} constraints, which may cause a jump in parameters.",
            stacklevel=2,
        )

    geometric = "Triclinic"
    if is_90(alpha) and is_90(beta) and is_90(gamma):
        if eq(a, b) and eq(b, c):
            geometric = "Cubic"
        elif eq(a, b):
            geometric = "Tetragonal"
        else:
            geometric = "Orthorhombic"
    elif is_90(alpha) and is_90(beta) and is_120(gamma):
        if eq(a, b):
            geometric = "Hexagonal"
    elif (
        centering == "R"
        and eq(a, b)
        and eq(b, c)
        and eq(alpha, beta)
        and eq(beta, gamma)
    ):
        geometric = "Rhombohedral"
    elif sum([is_90(alpha), is_90(beta), is_90(gamma)]) >= 2:
        geometric = "Monoclinic"

    ranks = {
        "Triclinic": 0,
        "Monoclinic": 1,
        "Orthorhombic": 2,
        "Tetragonal": 3,
        "Rhombohedral": 4,
        "Hexagonal": 4,
        "Cubic": 5,
    }
    rank_exp = ranks.get(expected, 0)
    rank_geo = ranks.get(geometric, 0)

    final_system = expected
    num = {
        "Triclinic": 6,
        "Monoclinic": 4,
        "Orthorhombic": 3,
        "Tetragonal": 2,
        "Hexagonal": 2,
        "Rhombohedral": 2,
        "Cubic": 1,
    }.get(final_system, 6)

    if rank_exp < rank_geo:
        print(
            f"Lattice System Override: Geometry suggests {geometric}, but Space Group {space_group_name} requires {expected}. Enforcing {expected}."
        )

    return final_system, num, centering


def rotation_matrix_from_axis_angle_jax(axis, angle_rad):
    u = axis / jnp.linalg.norm(axis)
    ux, uy, uz = u
    K = jnp.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    eye = jnp.eye(3)
    return eye + s[..., None, None] * K + (1.0 - c)[..., None, None] * (K @ K)


def rotation_matrix_from_rodrigues_jax(w):
    theta = jnp.linalg.norm(w) + 1e-9
    k = w / theta
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    eye = jnp.eye(3)
    return eye + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)


class VectorizedObjective:
    def __init__(
        self,
        B,
        kf_ki_dir,
        peak_xyz_lab,
        wavelength,
        cell_params=None,
        refine_lattice=False,
        lattice_bound_frac=0.05,
        lattice_system="Triclinic",
        centering="P",
        motor_map=None,
        goniometer_axes=None,
        goniometer_angles=None,
        refine_goniometer=False,
        goniometer_bound_deg=5.0,
        goniometer_refine_mask=None,
        goniometer_trans_refine_mask=None,
        goniometer_nominal_offsets=None,
        goniometer_axis_vector_mask=None,
        goniometer_axis_vector_bound_deg=1.0,
        per_run_motor_index=None,
        per_run_frame_map=None,
        per_run_bound_deg=0.5,
        per_run_trans=False,
        per_run_trans_bound_m=0.002,
        harmonic_frame_angles_deg=None,
        harmonic_axes=None,
        harmonic_orders=None,
        harmonic_bound_deg=0.5,
        refine_sample=False,
        goniometer_trans_bound_meters=0.005,
        sample_nominal=None,
        refine_beam=False,
        beam_bound_deg=1.0,
        beam_nominal=None,
        static_R=None,
        kf_lab_fixed_vectors=None,
        peak_run_indices=None,
        refine_detector=False,
        detector_params=None,
        peak_pixel_coords=None,
        detector_trans_bound_meters=0.005,
        detector_rot_bound_deg=2.0,
        freeze_orientation=False,
        fixed_rot_params=None,
        num_candidates=None,
        no_index=False,
        hkl_fixed=None,
        lambda_fixed=None,
        radial_weight=1.0,
        radial_weight_poly=None,
        hkl_metric="isotropic",
        hkl_metric_floor=0.1,
    ):
        self.no_index = no_index
        if self.no_index:
            self.hkl_fixed = jnp.array(hkl_fixed)  # Shape: (3, N)
            self.lambda_fixed = jnp.array(lambda_fixed)  # Shape: (N,)
        # w in [0, 1]: dimensionless weight multiplying the radial
        # residual component, w = tangential scale / radial scale.
        # w = 1 keeps the isotropic chord, w = 0 discards the radial
        # direction entirely (the measured streak ratio 3.7 on
        # cg4d-t4-lysozyme corresponds to w = 0.27).  A polynomial in
        # the elastic wavelength (Angstrom, highest-degree coefficient
        # first) overrides the scalar.
        self.radial_weight = float(radial_weight)
        self.radial_weight_poly = (
            jnp.array(radial_weight_poly) if radial_weight_poly is not None else None
        )
        self.radial_weight_active = bool(
            radial_weight_poly is not None or self.radial_weight < 1.0
        )
        # Soft-indexer basin metric.  "isotropic" is the plain fractional-hkl
        # washboard; "positional" warps each basin by the per-peak Jacobian
        # that maps fractional defects to detector displacement, weighted by
        # radial_weight in the streak frame, with hkl_metric_floor as the
        # dimensionless isotropic floor that keeps the Jacobian's null
        # direction (the wavelength tube) from becoming gauge.
        if hkl_metric not in ("isotropic", "positional"):
            raise ValueError(f"Unknown hkl_metric: {hkl_metric!r}")
        self.hkl_metric_positional = hkl_metric == "positional"
        self.hkl_metric_floor = float(hkl_metric_floor)

        self.B = jnp.array(B)
        self.kf_ki_dir_init = jnp.array(kf_ki_dir)
        if self.kf_ki_dir_init.ndim == 2 and self.kf_ki_dir_init.shape[0] != 3:
            self.kf_ki_dir_init = self.kf_ki_dir_init.T

        self.k_sq_init = jnp.sum(self.kf_ki_dir_init**2, axis=0)
        num_peaks = self.kf_ki_dir_init.shape[1]
        self.freeze_orientation = freeze_orientation
        self.fixed_rot_params = (
            jnp.array(fixed_rot_params)
            if fixed_rot_params is not None
            else jnp.zeros(3)
        )

        self.static_R = jnp.array(static_R) if static_R is not None else jnp.eye(3)

        if peak_run_indices is not None:
            self.peak_run_indices = jnp.array(peak_run_indices, dtype=jnp.int32)
            if self.static_R.ndim == 3:
                max_run = jnp.max(self.peak_run_indices)
                if max_run >= self.static_R.shape[0] and self.static_R.shape[0] == 1:
                    self.static_R = jnp.tile(self.static_R, (max_run + 1, 1, 1))
        elif self.static_R.ndim == 3 and self.static_R.shape[0] == num_peaks:
            self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)
        else:
            self.peak_run_indices = jnp.zeros(num_peaks, dtype=jnp.int32)

        if self.static_R.ndim == 3:
            self.peak_run_indices = jnp.clip(
                self.peak_run_indices, 0, self.static_R.shape[0] - 1
            )

        if peak_xyz_lab is not None:
            p_xyz = jnp.array(peak_xyz_lab)
            self.peak_xyz = p_xyz.T if p_xyz.shape[0] != 3 else p_xyz
        else:
            self.peak_xyz = None

        if goniometer_axes is not None:
            axes = jnp.array(goniometer_axes)
            if axes.ndim == 2 and axes.shape[1] == 3:
                axes = jnp.concatenate([axes, jnp.ones((axes.shape[0], 1))], axis=1)
            self.gonio_axes = axes
            self.num_gonio_axes = self.gonio_axes.shape[0]

            # Register the 1:n mapping
            self.motor_map = (
                jnp.array(motor_map, dtype=jnp.int32)
                if motor_map is not None
                else jnp.arange(self.num_gonio_axes)
            )
            self.num_motors = (
                jnp.max(self.motor_map) + 1 if self.num_gonio_axes > 0 else 0
            )

            angles = jnp.array(goniometer_angles)
            if angles.ndim == 2 and angles.shape[0] != self.num_gonio_axes:
                angles = angles.T
            self.gonio_angles = angles

            if self.gonio_angles.shape[1] == num_peaks:
                self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)

            self.gonio_mask = (
                np.array(goniometer_refine_mask, dtype=bool)
                if goniometer_refine_mask is not None
                else np.ones(self.num_motors, dtype=bool)
            )
            self.num_active_gonio = np.sum(self.gonio_mask)

            # The translation mask follows the angle mask unless a
            # separate per-motor selection is given: the phi-stage lever
            # arm (a sample-on-pin eccentricity rotates WITH phi) is only
            # refinable through this mask, and slaving it to the angle
            # list would also free the pure-gauge phi zero point.
            trans_motor_mask = (
                np.array(goniometer_trans_refine_mask, dtype=bool)
                if goniometer_trans_refine_mask is not None
                else np.asarray(self.gonio_mask, dtype=bool)
            )
            self.gonio_trans_mask = trans_motor_mask[self.motor_map]
            self.num_active_trans = np.sum(self.gonio_trans_mask)

            raw_trans_bounds = jnp.array(goniometer_trans_bound_meters)
            if raw_trans_bounds.ndim == 1 and raw_trans_bounds.size == self.num_motors:
                # 1. Expand the motor bounds to match the physical axes
                mapped_bounds = raw_trans_bounds[self.motor_map]
                # 2. Mask down to active axes and add (:, None) to broadcast over XYZ
                self.gonio_trans_bound = mapped_bounds[self.gonio_trans_mask][:, None]
            else:
                self.gonio_trans_bound = raw_trans_bounds

            self.gonio_nominal_offsets = (
                jnp.array(goniometer_nominal_offsets)
                if goniometer_nominal_offsets is not None
                else jnp.zeros(self.num_motors)
            )

            # --- Goniometer axis-vector refinement (mount tilt) ---
            # The direction of each rotation axis, not just the zero point
            # about it: a goniometer mounted at a small angle to the
            # detector frame tilts every axis, an error the angular offsets
            # and translations can only chase degenerately.  Each selected
            # motor gets two tilt parameters about an orthonormal basis
            # perpendicular to its nominal direction; every physical axis
            # of that motor tilts identically (one mount, one error).
            self.axis_vec_mask = (
                np.array(goniometer_axis_vector_mask, dtype=bool)
                if goniometer_axis_vector_mask is not None
                else np.zeros(int(self.num_motors), dtype=bool)
            )
            self.num_active_axis_vec = int(np.sum(self.axis_vec_mask))
            raw_av = np.atleast_1d(
                np.asarray(goniometer_axis_vector_bound_deg, dtype=float)
            )
            av_bounds = (
                raw_av
                if raw_av.size == int(self.num_motors)
                else np.full(int(self.num_motors), raw_av.ravel()[0])
            )
            self.axis_vec_bound_rad = jnp.deg2rad(
                jnp.array(av_bounds[self.axis_vec_mask])
            )
            self.axis_vec_active_motors = np.nonzero(self.axis_vec_mask)[0]
            # Static (python-level) per-axis decision: the objective is
            # jitted with self in the pytree, so jnp attributes are traced
            # and cannot steer python control flow.
            motor_map_np = np.asarray(
                motor_map if motor_map is not None else np.arange(self.num_gonio_axes)
            ).astype(int)
            self._num_motors_static = (
                int(motor_map_np.max()) + 1 if motor_map_np.size else 0
            )
            self.axis_vec_refined_per_axis = [
                bool(self.axis_vec_mask[m]) for m in motor_map_np
            ]
            self._motor_map_list = motor_map_np.tolist()
            dirs = np.asarray(self.gonio_axes[:, 0:3], dtype=float)
            dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            ref = np.where(
                np.abs(dirs[:, 2:3]) < 0.9,
                np.array([[0.0, 0.0, 1.0]]),
                np.array([[1.0, 0.0, 0.0]]),
            )
            e1 = np.cross(dirs, ref)
            e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
            e2 = np.cross(dirs, e1)
            self.axis_dirs_nominal = jnp.array(dirs)
            self.axis_e1 = jnp.array(e1)
            self.axis_e2 = jnp.array(e2)

            # Per-run angle corrections: one bounded delta per scan run for
            # a single motor, applied to every frame of that run.  Static
            # per-setting positioning errors (encoder repeatability, mount
            # settling; measured 0.13 deg rms across the six t4 phi
            # settings, random signs) cannot be represented by any static
            # geometry parameter -- offsets, axis vectors and detector
            # modes all slid along a flat valley trying to average them.
            self.per_run_motor_idx = (
                int(per_run_motor_index) if per_run_motor_index is not None else None
            )
            self.per_run_trans = bool(per_run_trans)
            if self.per_run_motor_idx is not None or self.per_run_trans:
                if per_run_frame_map is None:
                    raise ValueError(
                        "Per-run refinement needs per_run_frame_map "
                        "(frame index -> run ordinal)."
                    )
                frame_map_np = np.asarray(per_run_frame_map, dtype=int)
                self.per_run_frame_map = jnp.array(frame_map_np, dtype=jnp.int32)
                self.num_runs_static = int(frame_map_np.max()) + 1
            else:
                self.per_run_frame_map = None
                self.num_runs_static = 0
            self.num_per_run_params = (
                self.num_runs_static if self.per_run_motor_idx is not None else 0
            )
            self.per_run_bound_deg = float(per_run_bound_deg)
            # Per-run translational offsets: the angular wobble's twin.
            # One 3-vector per scan run at the INNERMOST axis, so it rides
            # with the sample (mount settling, sphere-of-confusion).
            self.num_per_run_trans_params = (
                3 * self.num_runs_static if self.per_run_trans else 0
            )
            self.per_run_trans_bound_m = float(per_run_trans_bound_m)
            # Fourier-in-phi rocking: the crystal's effective orientation
            # oscillates about fixed lab axes as the scan motor turns
            # (crystal-fixed anisotropic mosaicity sampled by the
            # rotation).  Bounded Fourier coefficients per axis over the
            # requested harmonic band; m = 0 is excluded by construction
            # (degenerate with the global goniometer offsets).
            self.num_harmonic_params = 0
            if harmonic_axes is not None and harmonic_orders is not None:
                if harmonic_frame_angles_deg is None:
                    raise ValueError(
                        "Harmonic rocking needs harmonic_frame_angles_deg "
                        "(scan-motor angle per frame)."
                    )
                ang = np.asarray(harmonic_frame_angles_deg, dtype=float)
                n_frames = np.asarray(goniometer_angles).shape[1]
                if ang.shape != (n_frames,):
                    raise ValueError(
                        f"harmonic_frame_angles_deg shape {ang.shape} does "
                        f"not match the {n_frames} goniometer frames."
                    )
                orders_np = np.asarray(list(harmonic_orders), dtype=float)
                if orders_np.size == 0 or np.any(orders_np < 1):
                    raise ValueError(
                        "Harmonic orders must be positive integers; the "
                        "m = 0 term is the motor zero the global offsets "
                        "already refine."
                    )
                phase = np.deg2rad(orders_np[:, None] * ang[None, :])
                self.harmonic_cos = jnp.array(np.cos(phase))
                self.harmonic_sin = jnp.array(np.sin(phase))
                axes_np = np.asarray(harmonic_axes, dtype=float)
                self.harmonic_axes_mat = jnp.array(
                    axes_np / np.linalg.norm(axes_np, axis=1, keepdims=True)
                )
                self.num_harmonic_axes = axes_np.shape[0]
                self.num_harmonic_orders = orders_np.shape[0]
                self.num_harmonic_params = (
                    2 * self.num_harmonic_axes * self.num_harmonic_orders
                )
            self.harmonic_bound_deg = float(harmonic_bound_deg)
        else:
            self.gonio_axes = None
            self.num_gonio_axes = 0
            self.num_active_trans = 1
            self.gonio_trans_mask = np.ones(1, dtype=bool)
            self.num_motors = 0
            self.axis_vec_mask = np.zeros(0, dtype=bool)
            self.num_active_axis_vec = 0
            self.per_run_motor_idx = None
            self.per_run_frame_map = None
            self.num_per_run_params = 0
            self.per_run_trans = False
            self.num_per_run_trans_params = 0
            self.num_runs_static = 0
            self.num_harmonic_params = 0

        self.refine_gonio_trans = refine_sample
        num_trans = max(1, self.num_gonio_axes)

        if sample_nominal is not None:
            base_offset = jnp.array(sample_nominal)
            if base_offset.ndim == 1:
                t_base = jnp.zeros((num_trans, 3))
                self.gonio_trans_nominal = t_base.at[-1].set(base_offset)
            else:
                self.gonio_trans_nominal = base_offset
        else:
            self.gonio_trans_nominal = jnp.zeros((num_trans, 3))

        self.refine_beam = refine_beam
        self.beam_bound_deg = beam_bound_deg
        self.beam_nominal = (
            jnp.array(beam_nominal)
            if beam_nominal is not None
            else jnp.array([0.0, 0.0, 1.0])
        )

        self.kf_lab_fixed = None
        if self.peak_xyz is not None:
            v = self.peak_xyz - self.gonio_trans_nominal[-1, :][:, None]
            dist = jnp.linalg.norm(v, axis=0)
            self.kf_lab_fixed = v / jnp.where(dist == 0, 1.0, dist[None, :])

        if kf_lab_fixed_vectors is not None and self.kf_lab_fixed is None:
            q_vecs = jnp.array(kf_lab_fixed_vectors)
            q_vecs = q_vecs.T if q_vecs.shape[0] != 3 else q_vecs
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(
                self.kf_lab_fixed, axis=0
            )

        if self.kf_lab_fixed is None:
            q_vecs = self.kf_ki_dir_init
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(
                self.kf_lab_fixed, axis=0
            )

        self.refine_lattice = refine_lattice
        self.lattice_system = lattice_system
        self.lattice_bound_frac = lattice_bound_frac
        self.refine_goniometer = refine_goniometer
        self.goniometer_bound_deg = jnp.array(goniometer_bound_deg)

        if self.refine_lattice:
            self.cell_init = jnp.array(cell_params)
            if self.lattice_system == "Cubic":
                self.free_params_init = self.cell_init[0:1]
            elif self.lattice_system in ("Hexagonal", "Tetragonal"):
                self.free_params_init = jnp.array(
                    [self.cell_init[0], self.cell_init[2]]
                )
            elif self.lattice_system == "Rhombohedral":
                self.free_params_init = jnp.array(
                    [self.cell_init[0], self.cell_init[3]]
                )
            elif self.lattice_system == "Orthorhombic":
                self.free_params_init = self.cell_init[0:3]
            elif self.lattice_system == "Monoclinic":
                self.free_params_init = jnp.array(
                    [
                        self.cell_init[0],
                        self.cell_init[1],
                        self.cell_init[2],
                        self.cell_init[4],
                    ]
                )
            else:
                self.free_params_init = self.cell_init

        wavelength = jnp.array(wavelength)
        self.wl_min_val = wavelength[0]
        self.wl_max_val = wavelength[1]

        if num_candidates is not None:
            self.num_candidates = num_candidates
        else:
            self.num_candidates = 64

        self.refine_detector = refine_detector
        if self.refine_detector:
            self.det_centers = jnp.array(detector_params["centers"])
            self.det_uhats = jnp.array(detector_params["uhats"])
            self.det_vhats = jnp.array(detector_params["vhats"])

            # Exact physical offsets in meters bypass pixel logic completely
            self.peak_u_offsets = jnp.array(peak_pixel_coords["u_offsets"])
            self.peak_v_offsets = jnp.array(peak_pixel_coords["v_offsets"])
            self.peak_det_idx = jnp.array(
                peak_pixel_coords["bank_indices"], dtype=jnp.int32
            )

            # Peaks on banks outside the refined subset keep their static
            # lab positions; without this they would all collapse onto
            # refined bank 0 (bank_indices defaults to zero for them).
            refined_mask = np.asarray(
                peak_pixel_coords.get(
                    "refined_mask",
                    np.ones(len(peak_pixel_coords["bank_indices"]), dtype=bool),
                ),
                dtype=bool,
            )
            self.peak_det_refined = jnp.array(refined_mask)
            self.all_peaks_on_refined_banks = bool(np.all(refined_mask))
            if not self.all_peaks_on_refined_banks and self.peak_xyz is None:
                raise ValueError(
                    "Refining a subset of detector banks requires peaks/xyz "
                    "for the peaks on the unrefined banks."
                )

            self.num_banks = self.det_centers.shape[0]

            self.det_modes = detector_params.get("modes", ["independent"])
            self.det_param_slices = {}
            self.num_det_params = 0

            rot_axis = jnp.array(
                detector_params.get("global_rot_axis", [0.0, 1.0, 0.0])
            )
            self.det_global_rot_axis = rot_axis / (jnp.linalg.norm(rot_axis) + 1e-9)

            # Normalize the cylinder axis
            cyl_axis = jnp.array(detector_params.get("cylinder_axis", [0.0, 1.0, 0.0]))
            self.cylinder_axis = cyl_axis / (jnp.linalg.norm(cyl_axis) + 1e-9)

            self.bounds = {
                "radial": detector_params.get("radial_bound", 0.05),
                "area": detector_params.get("area_bound", 0.05),
                "global_rot": jnp.deg2rad(
                    detector_params.get("global_rot_bound_deg", 2.0)
                ),
                "global_rot_axis": jnp.deg2rad(
                    detector_params.get("global_rot_bound_deg", 2.0)
                ),
                "global_trans": detector_params.get("global_trans_bound_meters", 0.01),
                "independent_trans": detector_trans_bound_meters,
                "independent_rot": jnp.deg2rad(detector_rot_bound_deg),
            }

            for mode in self.det_modes:
                if mode in ("radial", "cylindrical", "axial_stretch", "area"):
                    size = 1
                elif mode == "global_rot":
                    size = 3
                elif mode == "global_rot_axis":
                    size = 1
                elif mode == "global_trans":
                    size = 3
                elif mode == "independent":
                    size = self.num_banks * 6
                else:
                    raise ValueError(f"Unknown detector refinement mode: {mode}")

                self.det_param_slices[mode] = slice(
                    self.num_det_params, self.num_det_params + size
                )
                self.num_det_params += size

            self.det_widths = jnp.array(
                [pw * m for pw, m in zip(detector_params["pw"], detector_params["m"])]
            )
            self.det_heights = jnp.array(
                [ph * n for ph, n in zip(detector_params["ph"], detector_params["n"])]
            )

        # primitive cell
        self.centering = centering
        if self.centering == "I":
            self.M_prim = jnp.array(
                [[0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]
            )
        elif self.centering == "F":
            self.M_prim = jnp.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
        elif self.centering == "C":
            self.M_prim = jnp.array(
                [[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.0, 1.0]]
            )
        elif self.centering == "A":
            self.M_prim = jnp.array(
                [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, -0.5]]
            )
        elif self.centering == "B":
            self.M_prim = jnp.array(
                [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, -0.5]]
            )
        elif self.centering == "R":
            self.M_prim = jnp.array(
                [[2 / 3, 1 / 3, 1 / 3], [-1 / 3, 1 / 3, 1 / 3], [-1 / 3, -2 / 3, 1 / 3]]
            )
        else:  # Default to P
            self.M_prim = jnp.eye(3)
        self.M_prim_inv = jnp.linalg.inv(self.M_prim)

    def orientation_U_jax(self, param):
        U = jax.vmap(rotation_matrix_from_rodrigues_jax)(param)
        return U

    def reconstruct_cell_params(self, params_norm):
        p_free = _forward_map_lattice(
            params_norm, self.free_params_init, self.lattice_bound_frac
        )

        S = params_norm.shape[0]
        deg90, deg120 = jnp.full((S,), 90.0), jnp.full((S,), 120.0)

        if self.lattice_system == "Cubic":
            a = p_free[:, 0]
            return jnp.stack([a, a, a, deg90, deg90, deg90], axis=1)
        if self.lattice_system == "Hexagonal":
            a, c = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, c, deg90, deg90, deg120], axis=1)
        if self.lattice_system == "Tetragonal":
            a, c = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, c, deg90, deg90, deg90], axis=1)
        if self.lattice_system == "Rhombohedral":
            a, alpha = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, a, alpha, alpha, alpha], axis=1)
        if self.lattice_system == "Orthorhombic":
            a, b, c = p_free[:, 0], p_free[:, 1], p_free[:, 2]
            return jnp.stack([a, b, c, deg90, deg90, deg90], axis=1)
        if self.lattice_system == "Monoclinic":
            a, b, c, beta = p_free[:, 0], p_free[:, 1], p_free[:, 2], p_free[:, 3]
            return jnp.stack([a, b, c, deg90, beta, deg90], axis=1)

        return p_free

    def _get_physical_params_jax(self, x):
        idx = 0
        if self.freeze_orientation:
            rot_params = self.fixed_rot_params[None, :].repeat(x.shape[0], axis=0)
        else:
            rot_params = x[:, idx : idx + 3]
            idx += 3
        U = self.orientation_U_jax(rot_params)

        if self.refine_lattice:
            n_lat = self.free_params_init.size
            cell_params_norm = x[:, idx : idx + n_lat]
            p = self.reconstruct_cell_params(cell_params_norm)

            deg2rad = jnp.pi / 180.0
            a, b, c = p[:, 0], p[:, 1], p[:, 2]
            alpha, beta, gamma = p[:, 3] * deg2rad, p[:, 4] * deg2rad, p[:, 5] * deg2rad

            g11, g22, g33 = a**2, b**2, c**2
            g12, g13, g23 = (
                a * b * jnp.cos(gamma),
                a * c * jnp.cos(beta),
                b * c * jnp.cos(alpha),
            )
            G = jnp.stack(
                [
                    jnp.stack([g11, g12, g13], axis=-1),
                    jnp.stack([g12, g22, g23], axis=-1),
                    jnp.stack([g13, g23, g33], axis=-1),
                ],
                axis=-2,
            )

            B = jscipy_linalg.cholesky(jnp.linalg.inv(G), lower=False)
            idx += n_lat
            UB = jnp.matmul(U, B)
        else:
            B = self.B
            UB = jnp.matmul(U, B[None, ...])

        num_trans = max(1, self.num_gonio_axes)
        if self.refine_gonio_trans:
            # Only extract active translation parameters
            num_active = self.num_active_trans
            t_norm_active = x[:, idx : idx + num_active * 3].reshape(-1, num_active, 3)
            idx += num_active * 3

            t_delta_active = _forward_map_param(t_norm_active, self.gonio_trans_bound)
            # Scatter active parameters into a full (S, num_trans, 3) zero array
            t_delta = jnp.zeros((x.shape[0], num_trans, 3))
            t_delta = t_delta.at[:, self.gonio_trans_mask, :].set(t_delta_active)
            t_axes = self.gonio_trans_nominal[None, :, :] + t_delta
        else:
            t_axes = self.gonio_trans_nominal[None, :, :].repeat(x.shape[0], axis=0)

        if self.refine_beam:
            bound_rad = jnp.deg2rad(self.beam_bound_deg)
            tx = _forward_map_param(x[:, idx], bound_rad)
            ty = _forward_map_param(x[:, idx + 1], bound_rad)
            idx += 2
            ki_vec = jnp.tile(self.beam_nominal[None, :], (x.shape[0], 1))
            ki_vec = ki_vec.at[(slice(None), 0)].add(tx)
            ki_vec = ki_vec.at[(slice(None), 1)].add(ty)
            ki_vec = ki_vec / jnp.linalg.norm(ki_vec, axis=1, keepdims=True)
        else:
            ki_vec = self.beam_nominal[None, :].repeat(x.shape[0], axis=0)

        offsets_total = None
        R_cum = None
        axis_dirs = None
        axis_tilts = None
        per_run_delta = None
        per_run_trans_delta = None
        harmonic_coeffs = None
        per_frame_trans = None
        sample_origin_lab = jnp.zeros((x.shape[0], 1, 3))

        if self.gonio_axes is not None:
            if self.refine_goniometer:
                gonio_norm = jnp.full((x.shape[0], self.num_motors), 0.5)
                if self.num_active_gonio > 0:
                    gonio_norm = gonio_norm.at[(slice(None), self.gonio_mask)].set(
                        x[:, idx : idx + self.num_active_gonio]
                    )
                    idx += self.num_active_gonio

                offsets_delta = _forward_map_param(
                    gonio_norm, self.goniometer_bound_deg
                )
                offsets_total = self.gonio_nominal_offsets[None, :] + offsets_delta
            else:
                offsets_total = self.gonio_nominal_offsets[None, :].repeat(
                    x.shape[0], axis=0
                )

            S, M = offsets_total.shape[0], self.gonio_angles.shape[1]

            # Axis-vector tilts: two bounded angles per selected motor,
            # applied as a gnomonic perturbation of the nominal direction
            # (d = n + tan(a) e1 + tan(b) e2, renormalised) -- exact over
            # the bounded cap and trivially invertible for reporting.
            axis_dirs = None
            if self.num_active_axis_vec > 0:
                av_norm = x[:, idx : idx + 2 * self.num_active_axis_vec].reshape(
                    -1, self.num_active_axis_vec, 2
                )
                idx += 2 * self.num_active_axis_vec
                tilts_active = _forward_map_param(
                    av_norm, self.axis_vec_bound_rad[None, :, None]
                )
                axis_tilts = jnp.zeros((x.shape[0], self._num_motors_static, 2))
                axis_tilts = axis_tilts.at[
                    :, jnp.array(self.axis_vec_active_motors), :
                ].set(tilts_active)
                tilt_per_axis = axis_tilts[
                    :, jnp.array([m for m in self._motor_map_list]), :
                ]
                d = (
                    self.axis_dirs_nominal[None, :, :]
                    + jnp.tan(tilt_per_axis[..., 0:1]) * self.axis_e1[None, :, :]
                    + jnp.tan(tilt_per_axis[..., 1:2]) * self.axis_e2[None, :, :]
                )
                axis_dirs = d / jnp.linalg.norm(d, axis=-1, keepdims=True)

            per_run_delta = None
            per_frame_corr = None
            if self.num_per_run_params > 0:
                pr_norm = x[:, idx : idx + self.num_per_run_params]
                idx += self.num_per_run_params
                per_run_delta = _forward_map_param(pr_norm, self.per_run_bound_deg)
                per_frame_corr = per_run_delta[:, self.per_run_frame_map]  # (S, M)

            per_run_trans_delta = None
            per_frame_trans = None
            if self.num_per_run_trans_params > 0:
                prt_norm = x[:, idx : idx + self.num_per_run_trans_params].reshape(
                    -1, self.num_runs_static, 3
                )
                idx += self.num_per_run_trans_params
                per_run_trans_delta = _forward_map_param(
                    prt_norm, self.per_run_trans_bound_m
                )
                per_frame_trans = per_run_trans_delta[
                    :, self.per_run_frame_map, :
                ]  # (S, M, 3)

            if self.num_harmonic_params > 0:
                hc_norm = x[:, idx : idx + self.num_harmonic_params].reshape(
                    -1, self.num_harmonic_axes, self.num_harmonic_orders, 2
                )
                idx += self.num_harmonic_params
                harmonic_coeffs = _forward_map_param(
                    hc_norm, self.harmonic_bound_deg
                )  # (S, K, n, 2) in degrees

            R_list = []
            deg2rad = jnp.pi / 180.0

            for i in range(self.num_gonio_axes):
                motor_idx = self.motor_map[i]
                direction = self.gonio_axes[i][0:3]

                # Extract direction multiplier to maintain geometry parity
                direction_mult = (
                    self.gonio_axes[i][3] if len(self.gonio_axes[i]) > 3 else 1.0
                )

                current_axis_angle = (
                    self.gonio_angles[i, :][None, :]
                    + offsets_total[:, motor_idx][:, None]
                )
                if (
                    per_frame_corr is not None
                    and self._motor_map_list[i] == self.per_run_motor_idx
                ):
                    current_axis_angle = current_axis_angle + per_frame_corr

                theta = direction_mult * current_axis_angle * deg2rad
                if axis_dirs is not None and self.axis_vec_refined_per_axis[i]:
                    # Per-sample tilted axis: batch the rotation over S.
                    Ri = jax.vmap(rotation_matrix_from_axis_angle_jax)(
                        axis_dirs[:, i, :], theta
                    )
                else:
                    Ri = rotation_matrix_from_axis_angle_jax(direction, theta)
                R_list.append(Ri)

            R_cum = jnp.eye(3)[None, None, ...].repeat(S, axis=0).repeat(M, axis=1)
            for i in range(self.num_gonio_axes):
                R_cum = jnp.matmul(R_cum, R_list[i])

            if harmonic_coeffs is not None:
                # The rocking multiplies from the lab side: it steers the
                # diffraction directions (q -> R_harm q) and leaves the
                # sample origin untouched, which is why it is folded into
                # R_cum after the translation stack is assembled.
                delta_deg = jnp.einsum(
                    "skn,nm->skm", harmonic_coeffs[..., 0], self.harmonic_cos
                ) + jnp.einsum(
                    "skn,nm->skm", harmonic_coeffs[..., 1], self.harmonic_sin
                )
                w = jnp.deg2rad(
                    jnp.einsum("skm,kd->smd", delta_deg, self.harmonic_axes_mat)
                )
                R_harm = jax.vmap(jax.vmap(rotation_matrix_from_rodrigues_jax))(w)
                R_cum = jnp.matmul(R_harm, R_cum)

            sample_origin_lab = jnp.zeros((S, M, 3))
            for i in reversed(range(self.num_gonio_axes)):
                t_i = t_axes[:, i, :][:, None, :]
                if per_frame_trans is not None and i == self.num_gonio_axes - 1:
                    # Per-run sample displacement rides on the innermost
                    # axis: s_lab gains R_full(frame) @ t_run(frame).
                    t_i = t_i + per_frame_trans
                # --- NEW KINEMATICS: Translate in local frame, THEN Rotate ---
                sample_origin_lab = jnp.einsum(
                    "smij,smj->smi", R_list[i], sample_origin_lab + t_i
                )
        else:
            sample_origin_lab = t_axes[:, 0, :][:, None, :]

        if self.refine_detector:
            det_params = x[:, idx : idx + self.num_det_params]
            idx += self.num_det_params

            S = x.shape[0]
            c = self.det_centers[None, :, :].repeat(S, axis=0)
            u = self.det_uhats[None, :, :].repeat(S, axis=0)
            v = self.det_vhats[None, :, :].repeat(S, axis=0)

            w = self.det_widths[None, :].repeat(S, axis=0)
            h = self.det_heights[None, :].repeat(S, axis=0)
            area_scale = jnp.zeros((S, 1))

            if "radial" in self.det_modes:
                slc = self.det_param_slices["radial"]
                scale_norm = det_params[:, slc]
                scale = _forward_map_param(scale_norm, self.bounds["radial"])
                c = c * (1.0 + scale[:, :, None])

            if "cylindrical" in self.det_modes:
                slc = self.det_param_slices["cylindrical"]
                scale_norm = det_params[:, slc]
                scale = _forward_map_param(scale_norm, self.bounds["radial"])

                c_dot_a = jnp.sum(c * self.cylinder_axis, axis=-1, keepdims=True)

                c_parallel = c_dot_a * self.cylinder_axis
                c_perp = c - c_parallel

                c = c_parallel + c_perp * (1.0 + scale[:, :, None])

            if "area" in self.det_modes:
                slc = self.det_param_slices["area"]
                scale_norm = det_params[:, slc]
                scale = _forward_map_param(scale_norm, self.bounds["area"])
                area_scale = scale

                # Pin the physical center of the panel ---
                # Shift the corner backwards to counteract the area expansion
                c = (
                    c
                    - (w / 2.0)[:, :, None] * scale[:, :, None] * u
                    - (h / 2.0)[:, :, None] * scale[:, :, None] * v
                )

                w = w * (1.0 + scale)
                h = h * (1.0 + scale)

            if "axial_stretch" in self.det_modes:
                slc = self.det_param_slices["axial_stretch"]
                scale_norm = det_params[:, slc]
                scale = _forward_map_param(scale_norm, self.bounds["radial"])

                c_dot_a = jnp.sum(c * self.cylinder_axis, axis=-1, keepdims=True)
                c_parallel = c_dot_a * self.cylinder_axis
                c_perp = c - c_parallel

                c = c_parallel * (1.0 + scale[:, :, None]) + c_perp

            if "global_rot" in self.det_modes:
                slc = self.det_param_slices["global_rot"]
                rot_norm = det_params[:, slc]
                rot_vec = _forward_map_param(rot_norm, self.bounds["global_rot"])
                R_global = jax.vmap(rotation_matrix_from_rodrigues_jax)(rot_vec)

                c = jnp.einsum("sij,snj->sni", R_global, c)
                u = jnp.einsum("sij,snj->sni", R_global, u)
                v = jnp.einsum("sij,snj->sni", R_global, v)

            if "global_rot_axis" in self.det_modes:
                slc = self.det_param_slices["global_rot_axis"]
                angle_norm = det_params[:, slc].reshape(-1)
                angle_rad = _forward_map_param(
                    angle_norm, self.bounds["global_rot_axis"]
                )

                R_global = jax.vmap(
                    rotation_matrix_from_axis_angle_jax, in_axes=(None, 0)
                )(self.det_global_rot_axis, angle_rad)

                c = jnp.einsum("sij,snj->sni", R_global, c)
                u = jnp.einsum("sij,snj->sni", R_global, u)
                v = jnp.einsum("sij,snj->sni", R_global, v)

            if "global_trans" in self.det_modes:
                slc = self.det_param_slices["global_trans"]
                trans_norm = det_params[:, slc]
                trans_vec = _forward_map_param(trans_norm, self.bounds["global_trans"])
                c = c + trans_vec[:, None, :]

            if "independent" in self.det_modes:
                slc = self.det_param_slices["independent"]
                indep_params = det_params[:, slc]

                t_norm = indep_params[:, : self.num_banks * 3].reshape(
                    -1, self.num_banks, 3
                )
                t_vec = _forward_map_param(t_norm, self.bounds["independent_trans"])
                c = c + t_vec

                r_norm = indep_params[:, self.num_banks * 3 :].reshape(
                    -1, self.num_banks, 3
                )
                r_vec = _forward_map_param(r_norm, self.bounds["independent_rot"])
                R_local = jax.vmap(jax.vmap(rotation_matrix_from_rodrigues_jax))(r_vec)

                u = jnp.einsum("snij,snj->sni", R_local, u)
                v = jnp.einsum("snij,snj->sni", R_local, v)

            dyn_centers, dyn_uhats, dyn_vhats = c, u, v
            dyn_widths, dyn_heights = w, h
        else:
            dyn_centers, dyn_uhats, dyn_vhats = None, None, None
            dyn_widths, dyn_heights, area_scale = None, None, None

        return (
            UB,
            B,
            t_axes,
            sample_origin_lab,
            ki_vec,
            offsets_total,
            R_cum,
            dyn_centers,
            dyn_uhats,
            dyn_vhats,
            dyn_widths,
            dyn_heights,
            area_scale,
            axis_dirs,
            axis_tilts,
            per_run_delta,
            per_run_trans_delta,
            harmonic_coeffs,
        )

    def _positional_metric_vectors(self, ub_mat, kf_ki_sample, ki_sample, k_sq):
        """Per-peak rows of the Jacobian mapping fractional-hkl defects to
        detector displacement, in the streak frame.

        kf_pred depends on hkl only through q_hat = normalize(UB hkl), so
        d(kf) = -2 [q_hat ki^T + (ki.q_hat) I] (I - q_hat q_hat^T) UB dhkl
        times lambda/|q| (applied per candidate in the scan, where the
        analytic lambda lives).  Projected on the per-peak frame
        t = ki x kf (tangential; t.q_hat = 0 collapses that row) and
        r = t x kf (radial, streak-elongated, weighted by radial_weight).
        Centering is folded in: the scan's fractional defects are primitive,
        so UB is post-multiplied by M_prim^-1.

        Returns (p_t, p_r, floor): two (S, 3, N) row vectors and the
        per-peak isotropic floor, set to 1 where the frame is degenerate
        (near-forward peaks) so those keep the plain fractional distance.
        """
        k_norm = jnp.sqrt(jnp.maximum(k_sq, 1e-12))
        q_hat = kf_ki_sample / k_norm[:, None, :]
        kf_s = kf_ki_sample + ki_sample

        t_vec = jnp.cross(ki_sample, kf_s, axis=1)
        t_norm = jnp.linalg.norm(t_vec, axis=1, keepdims=True)
        degenerate = t_norm[:, 0, :] < 1e-6
        t_hat = t_vec / jnp.where(t_norm == 0.0, 1.0, t_norm)
        r_vec = jnp.cross(t_hat, kf_s, axis=1)
        r_hat = r_vec / jnp.maximum(
            jnp.linalg.norm(r_vec, axis=1, keepdims=True), 1e-12
        )

        c = jnp.sum(ki_sample * q_hat, axis=1)  # (S, N)
        rq = jnp.sum(r_hat * q_hat, axis=1)
        ki_perp = ki_sample - c[:, None, :] * q_hat
        r_perp = r_hat - rq[:, None, :] * q_hat

        b_t = -2.0 * c[:, None, :] * t_hat
        b_r = -2.0 * (rq[:, None, :] * ki_perp + c[:, None, :] * r_perp)

        W = jnp.matmul(ub_mat, self.M_prim_inv[None, :, :])  # (S, 3, 3) tiny

        def project(b):
            # W^T b as explicit multiply-adds: an einsum here lowers to a
            # batched 3x3 cuBLAS GEMM (S tiny matrices), which neither
            # fuses into the elementwise graph nor rematerializes cheaply
            # inside the candidate scan.
            return jnp.stack(
                [
                    W[:, 0, i, None] * b[:, 0, :]
                    + W[:, 1, i, None] * b[:, 1, :]
                    + W[:, 2, i, None] * b[:, 2, :]
                    for i in range(3)
                ],
                axis=1,
            )

        p_t = project(b_t)
        p_r = project(b_r)

        zero = jnp.zeros_like(p_t)
        p_t = jnp.where(degenerate[:, None, :], zero, p_t)
        p_r = jnp.where(degenerate[:, None, :], zero, p_r)
        floor = jnp.where(degenerate, 1.0, self.hkl_metric_floor)
        # The candidate scan reads these seven (S, N)-planes once per
        # wavelength candidate, and that traffic dominates the metric's
        # cost.  A weight needs ~1% accuracy; bfloat16 (0.4% relative)
        # halves the traffic -- measured 1.3x per generation with 99.97%
        # identical assignments and <0.1% loss difference.
        return (
            p_t.astype(jnp.bfloat16),
            p_r.astype(jnp.bfloat16),
            floor.astype(jnp.bfloat16),
        )

    def indexer_dynamic_soft_jax(
        self, ub_mat, kf_ki_sample, k_sq_override=None, pos_metric=None
    ):
        ub_inv = jnp.linalg.inv(ub_mat)
        v = jnp.matmul(ub_inv, kf_ki_sample)

        k_sq = k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]

        lam_grid = jnp.logspace(
            jnp.log10(self.wl_min_val), jnp.log10(self.wl_max_val), self.num_candidates
        )

        S, _, N = v.shape
        if pos_metric is not None:
            p_t, p_r, floor = pos_metric
            k_norm = jnp.sqrt(jnp.maximum(k_sq, 1e-12))
            initial_carry = (
                jnp.inf * jnp.ones((S, N)),
                jnp.zeros((S, N)),
                jnp.zeros((S, 3, N), dtype=jnp.int32),
                jnp.zeros((S, N)),
            )
        else:
            initial_carry = (
                jnp.inf * jnp.ones((S, N)),
                jnp.zeros((S, 3, N), dtype=jnp.int32),
                jnp.zeros((S, N)),
            )

        # 1. Unpack v
        v_h = v[:, 0, :]
        v_k = v[:, 1, :]
        v_l = v[:, 2, :]

        # 2. Pre-compute the projection (UB^T @ kf_ki)
        ub_T_kf_ki = jnp.matmul(ub_mat.transpose((0, 2, 1)), kf_ki_sample)
        ub_T_kf_ki_h = ub_T_kf_ki[:, 0, :]
        ub_T_kf_ki_k = ub_T_kf_ki[:, 1, :]
        ub_T_kf_ki_l = ub_T_kf_ki[:, 2, :]

        def scan_body(carry, i):
            if pos_metric is not None:
                curr_min, curr_best_frac, curr_best_hkl, curr_best_lamb = carry
            else:
                curr_min, curr_best_hkl, curr_best_lamb = carry

            lamda_cand = lam_grid[i]

            # 1. Float HKL from coarse grid search (Fully Unrolled)
            h_float = v_h / lamda_cand
            k_float = v_k / lamda_cand
            l_float = v_l / lamda_cand

            h_int = jnp.round(h_float)
            k_int = jnp.round(k_float)
            l_int = jnp.round(l_float)

            h_i = jax.lax.stop_gradient(h_int)
            k_i = jax.lax.stop_gradient(k_int)
            l_i = jax.lax.stop_gradient(l_int)

            # Reconstruct the integer array for the carry state
            hkl_int = jnp.stack([h_i, k_i, l_i], axis=1).astype(jnp.int32)

            # 2. Exact Analytical Lambda (Using Hoisted Projection)
            k_dot_q = ub_T_kf_ki_h * h_i + ub_T_kf_ki_k * k_i + ub_T_kf_ki_l * l_i
            safe_dot = jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
            lambda_opt = jnp.clip(k_sq / safe_dot, self.wl_min_val, self.wl_max_val)

            # 3. Exact Analytical Fractional HKL (Unrolled)
            h = v_h / lambda_opt
            k = v_k / lambda_opt
            l = v_l / lambda_opt

            # 4. Apply Lattice Centering
            h_p = self.M_prim[0, 0] * h + self.M_prim[0, 1] * k + self.M_prim[0, 2] * l
            k_p = self.M_prim[1, 0] * h + self.M_prim[1, 1] * k + self.M_prim[1, 2] * l
            l_p = self.M_prim[2, 0] * h + self.M_prim[2, 1] * k + self.M_prim[2, 2] * l

            # 5. The 3-term loss
            dh = jnp.sin(jnp.pi * h_p)
            dk = jnp.sin(jnp.pi * k_p)
            dl = jnp.sin(jnp.pi * l_p)
            dist = jnp.sqrt(dh**2 + dk**2 + dl**2) / jnp.pi

            if pos_metric is not None:
                # Basin metric: the positional displacement this fractional
                # defect implies, weighted in the streak frame, plus the
                # isotropic floor for the Jacobian's null direction.  The
                # candidate is SELECTED by this metric (anisotropic
                # assignment); `dist` keeps the fractional units downstream.
                c_t = (p_t[:, 0] * dh + p_t[:, 1] * dk + p_t[:, 2] * dl) / jnp.pi
                c_r = (p_r[:, 0] * dh + p_r[:, 1] * dk + p_r[:, 2] * dl) / jnp.pi
                scale = lambda_opt / k_norm
                pos_sq = scale**2 * (c_t**2 + (self.radial_weight * c_r) ** 2)
                metric = jnp.sqrt(pos_sq + (floor * dist) ** 2)

                update_mask = metric < curr_min
                new_min = jnp.where(update_mask, metric, curr_min)
                new_frac = jnp.where(update_mask, dist, curr_best_frac)
                new_best_hkl = jnp.where(
                    update_mask[:, None, :], hkl_int, curr_best_hkl
                )
                new_best_lamb = jnp.where(update_mask, lambda_opt, curr_best_lamb)
                return (new_min, new_frac, new_best_hkl, new_best_lamb), None

            # 6. Update states
            update_mask = dist < curr_min
            new_min = jnp.where(update_mask, dist, curr_min)
            new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)
            new_best_lamb = jnp.where(update_mask, lambda_opt, curr_best_lamb)

            return (new_min, new_best_hkl, new_best_lamb), None

        final_carry, _ = lax.scan(
            scan_body, initial_carry, jnp.arange(self.num_candidates)
        )
        if pos_metric is not None:
            metric_min, dist_min, best_hkl, best_lamb = final_carry
            loss = jnp.mean(metric_min, axis=1)
        else:
            dist_min, best_hkl, best_lamb = final_carry
            loss = jnp.mean(dist_min, axis=1)
        return loss, dist_min, best_hkl.transpose((0, 2, 1)), best_lamb

    def geometric_loss_jax(self, ub_mat, kf_ki_sample, ki_sample):
        """Predicted-vs-observed scattering-direction residual at fixed hkl.

        The wavelength is not a free parameter here: the elastic condition
        fixes it from the assigned integer hkl and the current geometry,
        lam = -2 (ki . G) / |G|^2 with G = UB hkl, so a geometry error must
        surface as a direction mismatch instead of being absorbed into a
        per-peak wavelength.  In particular kf_pred = ki + lam G is exactly
        invariant under an isotropic lattice rescale (G -> sG, lam -> lam/s),
        which removes the detector-distance <-> lattice-scale valley the
        free-wavelength indexing loss slides along.

        The residual is the chord |kf_pred - kf_obs| between unit vectors:
        equal to the angular error in radians to third order, and smooth at
        zero where arccos is not.

        Laue spots are streaks along the 2-theta gradient (mosaic blocks
        rotated within the scattering plane stay reflective at an adjusted
        wavelength), so the observed centroid scatters radially far more
        than tangentially (measured 3.7x on cg4d-t4-lysozyme).  The chord
        is therefore decomposed in the per-peak frame t = ki x kf (out of
        the scattering plane), r = t x kf (2-theta gradient), and the
        radial component is multiplied by the dimensionless weight
        w = radial_weight in [0, 1] (tangential-to-radial scale ratio),
        optionally wavelength-dependent through a polynomial w(lam) --
        streak-centroid noise then cannot steer the geometry, while the
        narrow tangential direction keeps full weight.  w = 1 reproduces
        the plain chord exactly; w = 0 fits tangential-only, at which
        point purely radial parameter directions become gauge.
        """
        G = jnp.matmul(ub_mat, self.hkl_fixed)  # (S, 3, N)
        G_sq = jnp.sum(G * G, axis=1)
        lam = -2.0 * jnp.sum(ki_sample * G, axis=1) / jnp.where(G_sq == 0.0, 1.0, G_sq)
        kf_pred = ki_sample + lam[:, None, :] * G
        kf_obs = kf_ki_sample + ki_sample

        delta = kf_pred - kf_obs
        if self.radial_weight_active:
            t_vec = jnp.cross(ki_sample, kf_obs, axis=1)
            t_norm = jnp.linalg.norm(t_vec, axis=1, keepdims=True)
            t_hat = t_vec / jnp.where(t_norm == 0.0, 1.0, t_norm)
            r_vec = jnp.cross(t_hat, kf_obs, axis=1)
            r_norm = jnp.linalg.norm(r_vec, axis=1, keepdims=True)
            r_hat = r_vec / jnp.where(r_norm == 0.0, 1.0, r_norm)

            c_t = jnp.sum(delta * t_hat, axis=1)
            c_r = jnp.sum(delta * r_hat, axis=1)
            c_l = jnp.sum(delta * kf_obs, axis=1)

            if self.radial_weight_poly is not None:
                w = jnp.polyval(self.radial_weight_poly, lam)
            else:
                w = self.radial_weight
            w = jnp.clip(w, 0.0, 1.0)

            weighted = jnp.sqrt(c_t**2 + c_l**2 + (w * c_r) ** 2)
            # Near-forward peaks (ki x kf -> 0) have no defined frame:
            # keep the isotropic chord there.
            degenerate = t_norm[:, 0, :] < 1e-6
            dist = jnp.where(degenerate, jnp.linalg.norm(delta, axis=1), weighted)
        else:
            dist = jnp.linalg.norm(delta, axis=1)
        # hkl = 0 rows (peaks the bootstrap left unassigned) carry no
        # assignment information and must not enter the loss: their
        # kf_pred = ki makes the residual |kf_obs - ki| = 2 sin(theta),
        # which the detector and beam parameters CAN shrink -- measured on
        # t4, 1800 such peaks dragged the fit to its bounds.  They still
        # get lam = 0 and a reported residual, and always fail the cut.
        assigned = jnp.any(self.hkl_fixed != 0.0, axis=0)
        n_assigned = jnp.maximum(jnp.sum(assigned), 1.0)
        loss = jnp.sum(dist * assigned[None, :], axis=1) / n_assigned

        S = kf_ki_sample.shape[0]
        hkl_ret = jnp.tile(self.hkl_fixed.T[None, :, :], (S, 1, 1))
        return loss, dist, hkl_ret, lam

    @partial(jax.jit, static_argnames="self")
    def get_results(self, x):
        original_S = x.shape[0]
        pad_size = max(0, 2 - original_S)
        x_pad = jnp.pad(x, ((0, pad_size), (0, 0)), mode="edge") if pad_size > 0 else x

        (
            UB,
            _,
            _,
            sample_origin_lab,
            ki_vec,
            _,
            R_cum,
            dyn_centers,
            dyn_uhats,
            dyn_vhats,
            dyn_widths,
            dyn_heights,
            area_scale,
            _,
            _,
            _,
            _,
            _,
        ) = self._get_physical_params_jax(x_pad)

        R_curr = R_cum if R_cum is not None else self.static_R

        if sample_origin_lab.shape[1] > 1:
            # Multi-frame: Extract the specific origin for each peak's run
            s_lab = sample_origin_lab[:, self.peak_run_indices, :]
        else:
            # Single-frame: Tile the origin to match all peaks
            s_lab = jnp.tile(sample_origin_lab, (1, self.peak_run_indices.shape[0], 1))

        s = s_lab.transpose(0, 2, 1)  # Shape: (S, 3, N_peaks)

        if self.refine_detector:
            c = dyn_centers[:, self.peak_det_idx, :]
            u_vec = dyn_uhats[:, self.peak_det_idx, :]
            v_vec = dyn_vhats[:, self.peak_det_idx, :]

            if area_scale is not None:
                scale_factor = 1.0 + area_scale[:, 0][:, None]
                u_offset = self.peak_u_offsets[None, :] * scale_factor
                v_offset = self.peak_v_offsets[None, :] * scale_factor
            else:
                u_offset = self.peak_u_offsets[None, :]
                v_offset = self.peak_v_offsets[None, :]

            u_offset = self.peak_u_offsets
            v_offset = self.peak_v_offsets

            dynamic_xyz = (
                c + u_offset[None, :, None] * u_vec + v_offset[None, :, None] * v_vec
            )
            p = dynamic_xyz.transpose(0, 2, 1)
            if not self.all_peaks_on_refined_banks:
                p = jnp.where(
                    self.peak_det_refined[None, None, :],
                    p,
                    self.peak_xyz[None, :, :],
                )
        else:
            p = self.peak_xyz[None, :, :] if self.peak_xyz is not None else None

        if p is not None:
            v = p - s
            dist = jnp.sqrt(jnp.sum(v**2, axis=1, keepdims=True))
            kf = v / jnp.where(dist == 0, 1.0, dist)
            ki = ki_vec[:, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)
        else:
            kf = self.kf_lab_fixed[None, :, :].repeat(x_pad.shape[0], axis=0)
            ki = ki_vec[:, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)

        # q_lab is a pure momentum vector, so it strictly requires the
        # inverse goniometer rotation (R^T) to reach the sample frame
        if R_curr is not None:
            if R_curr.ndim == 4:
                RT = R_curr[:, self.peak_run_indices, :, :].transpose(0, 1, 3, 2)
            elif R_curr.ndim == 3:
                RT = R_curr[self.peak_run_indices, :, :].transpose(0, 2, 1)[None, ...]
            else:
                RT = R_curr.T[None, None, ...]

            def to_sample(vec):
                vec_T = jnp.matmul(RT, vec.transpose(0, 2, 1)[..., None]).squeeze(-1)
                return vec_T.transpose(0, 2, 1)

            kf_ki_vec = to_sample(q_lab)
        else:
            kf_ki_vec = q_lab

        if self.no_index or self.hkl_metric_positional:
            ki_full = jnp.broadcast_to(ki, q_lab.shape)
            ki_sample = to_sample(ki_full) if R_curr is not None else ki_full
        if self.no_index:
            res = self.geometric_loss_jax(UB, kf_ki_vec, ki_sample)
        else:
            pos_metric = None
            if self.hkl_metric_positional:
                pos_metric = self._positional_metric_vectors(
                    UB, kf_ki_vec, ki_sample, k_sq_dyn
                )
            res = self.indexer_dynamic_soft_jax(
                UB,
                kf_ki_vec,
                k_sq_override=k_sq_dyn,
                pos_metric=pos_metric,
            )

        return jax.tree.map(
            lambda arr: (
                arr[:original_S] if hasattr(arr, "shape") and arr.ndim > 0 else arr
            ),
            res,
        )

    @partial(jax.jit, static_argnames="self")
    def __call__(self, x):
        score, _, _, _ = self.get_results(x)
        return score


class FindUB:
    def __init__(self, filename=None, data=None):
        self.goniometer_axes = None
        self.goniometer_angles = None
        self.goniometer_offsets = None
        self.goniometer_axes_refined = None
        self.goniometer_axis_tilts = None
        self.goniometer_per_run_delta = None
        self.goniometer_per_run_motor = None
        self.goniometer_per_run_trans = None
        self.goniometer_harmonics = None
        self.goniometer_names = None
        self.sample_offset = None
        self.peak_xyz = None
        self.ki_vec = None
        self.base_sample_offset = np.zeros(3)
        self.base_gonio_offset = None
        self.fixed_rot_params = np.zeros(3)

        if filename is not None:
            self.load_peaks(filename)
        elif data is not None:
            self.load_from_dict(data)

    def load_from_dict(self, data):
        self.a = data["sample/a"]
        self.b = data["sample/b"]
        self.c = data["sample/c"]
        self.alpha = data["sample/alpha"]
        self.beta = data["sample/beta"]
        self.gamma = data["sample/gamma"]
        self.wavelength = data["instrument/wavelength"]
        self.R = data.get("goniometer/R")

        self.two_theta = data.get("peaks/two_theta", np.array([]))
        self.az_phi = data.get("peaks/azimuthal", np.array([]))

        r_stack = data.get("goniometer/R")
        idx_run = data.get("peaks/run_index")
        idx_img = data.get("peaks/image_index")
        idx_bank = data.get("bank")
        if idx_bank is None:
            idx_bank = data.get("bank_ids")

        if r_stack is not None and r_stack.ndim == 3:
            num_rot = r_stack.shape[0]
            if idx_run is not None and int(np.max(idx_run)) + 1 == num_rot:
                self.run_indices = idx_run
            elif idx_img is not None and int(np.max(idx_img)) + 1 == num_rot:
                self.run_indices = idx_img
            elif idx_bank is not None and int(np.max(idx_bank)) + 1 == num_rot:
                self.run_indices = idx_bank
            else:
                self.run_indices = idx_run if idx_run is not None else idx_img
        else:
            self.run_indices = idx_run if idx_run is not None else idx_img

        if self.run_indices is None:
            self.run_indices = idx_bank

        if self.run_indices is None:
            num_peaks = len(data.get("peaks/pixel_r", data.get("peaks/two_theta", [])))
            self.run_indices = np.zeros(num_peaks, dtype=int)

        sg = data["sample/space_group"]
        self.space_group = sg.decode("utf-8") if isinstance(sg, bytes) else str(sg)

        if "goniometer/translations" in data:
            self.base_sample_offset = data["goniometer/translations"]
        if "peaks/xyz" in data:
            self.peak_xyz = data["peaks/xyz"]
        if "goniometer/axes" in data:
            self.goniometer_axes = data["goniometer/axes"]
        if "goniometer/angles" in data:
            self.goniometer_angles = data["goniometer/angles"]
        if "goniometer/names" in data:
            self.goniometer_names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in data["goniometer/names"]
            ]
        self.ki_vec = (
            data["beam/ki_vec"] if "beam/ki_vec" in data else np.array([0.0, 0.0, 1.0])
        )

        # Generalize bootstrap loader for existing hkl and lambda
        if "peaks/h" in data and "peaks/k" in data and "peaks/l" in data:
            self.hkl = np.vstack([data["peaks/h"], data["peaks/k"], data["peaks/l"]])
        else:
            self.hkl = None

        if "peaks/lambda" in data:
            self.lambdas = data["peaks/lambda"]
        else:
            self.lambdas = None

    def load_peaks(self, filename):
        with h5py.File(os.path.abspath(filename), "r") as f:
            data = {
                "sample/a": f["sample/a"][()],
                "sample/b": f["sample/b"][()],
                "sample/c": f["sample/c"][()],
                "sample/alpha": f["sample/alpha"][()],
                "sample/beta": f["sample/beta"][()],
                "sample/gamma": f["sample/gamma"][()],
                "instrument/wavelength": f["instrument/wavelength"][()],
                "goniometer/R": f["goniometer/R"][()],
                "sample/space_group": f["sample/space_group"][()],
            }
            if "peaks/two_theta" in f:
                data["peaks/two_theta"] = f["peaks/two_theta"][()]
            if "peaks/azimuthal" in f:
                data["peaks/azimuthal"] = f["peaks/azimuthal"][()]
            if "peaks/run_index" in f:
                data["peaks/run_index"] = f["peaks/run_index"][()]
            if "peaks/image_index" in f:
                data["peaks/image_index"] = f["peaks/image_index"][()]
            if "bank" in f:
                data["bank"] = f["bank"][()]
            if "bank_ids" in f:
                data["bank_ids"] = f["bank_ids"][()]
            if "peaks/xyz" in f:
                data["peaks/xyz"] = f["peaks/xyz"][()]
            if "goniometer/axes" in f:
                data["goniometer/axes"] = f["goniometer/axes"][()]
            if "goniometer/angles" in f:
                data["goniometer/angles"] = f["goniometer/angles"][()]
            if "goniometer/names" in f:
                data["goniometer/names"] = f["goniometer/names"][()]
            if "goniometer/translations" in f:
                data["goniometer/translations"] = f["goniometer/translations"][()]
            if "beam/ki_vec" in f:
                data["beam/ki_vec"] = f["beam/ki_vec"][()]

            # Add bootstrap keys
            if "peaks/h" in f:
                data["peaks/h"] = f["peaks/h"][()]
                data["peaks/k"] = f["peaks/k"][()]
                data["peaks/l"] = f["peaks/l"][()]
            if "peaks/lambda" in f:
                data["peaks/lambda"] = f["peaks/lambda"][()]

            self.load_from_dict(data)

    def reciprocal_lattice_B(self):
        alpha, beta, gamma = np.deg2rad([self.alpha, self.beta, self.gamma])
        g11, g22, g33 = self.a**2, self.b**2, self.c**2
        g12 = self.a * self.b * np.cos(gamma)
        g13 = self.c * self.a * np.cos(beta)
        g23 = self.b * self.c * np.cos(alpha)
        G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])
        return scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    def get_bootstrap_params(
        self,
        bootstrap_filename,
        refine_lattice=False,
        lattice_bound_frac=0.05,
        refine_sample=False,
        sample_bound_meters=0.002,
        refine_beam=False,
        beam_bound_deg=1.0,
        refine_goniometer=False,
        goniometer_bound_deg=5.0,
        refine_goniometer_axes=None,
        freeze_orientation=False,
    ):
        print(f"Bootstrapping from physical solution: {bootstrap_filename}")
        with h5py.File(bootstrap_filename, "r") as f:
            raw_x = (
                f["optimization/best_params"][()]
                if "optimization/best_params" in f
                else None
            )
            b_a, b_b, b_c = f["sample/a"][()], f["sample/b"][()], f["sample/c"][()]
            b_alpha, b_beta, b_gamma = (
                f["sample/alpha"][()],
                f["sample/beta"][()],
                f["sample/gamma"][()],
            )
            if "goniometer/translations" in f:
                b_offset = f["goniometer/translations"][()]
            elif "sample/offset" in f:
                b_offset = f["sample/offset"][()]
            else:
                b_offset = np.zeros(3)

            b_ki = (
                f["beam/ki_vec"][()]
                if "beam/ki_vec" in f
                else np.array([0.0, 0.0, 1.0])
            )
            b_gonio_offsets = None
            if "goniometer/offsets" in f:
                off_data = f["goniometer/offsets"]
                if isinstance(off_data, h5py.Group):
                    b_gonio_offsets = {k: off_data[k][()] for k in off_data.keys()}
                else:
                    b_gonio_offsets = off_data[()]

            if "sample/U" in f:
                U_initial = f["sample/U"][()]
                from scipy.spatial.transform import Rotation as R

                rodrigues_vec = R.from_matrix(U_initial).as_rotvec()
                if raw_x is None:
                    raw_x = rodrigues_vec
                else:
                    raw_x[:3] = rodrigues_vec

        if freeze_orientation:
            self.fixed_rot_params = raw_x[:3] if raw_x is not None else np.zeros(3)
            new_params = []
        else:
            new_params = [raw_x[:3] if raw_x is not None else np.zeros(3)]

        if refine_lattice:
            self.a, self.b, self.c = b_a, b_b, b_c
            self.alpha, self.beta, self.gamma = b_alpha, b_beta, b_gamma
            lat_sys, _, _ = get_lattice_system(
                self.a,
                self.b,
                self.c,
                self.alpha,
                self.beta,
                self.gamma,
                self.space_group,
            )
            new_params.append(np.full(len(_get_active_lattice_indices(lat_sys)), 0.5))

        if b_offset is not None:
            self.base_sample_offset = b_offset

        # Build the 1:N motor-to-axis map exactly like minimize()
        motor_map = []
        unique_motors = []
        if self.goniometer_names is not None:
            for name in self.goniometer_names:
                if name not in unique_motors:
                    unique_motors.append(name)
                motor_map.append(unique_motors.index(name))
        elif self.goniometer_axes is not None:
            motor_map = list(range(len(self.goniometer_axes)))
            unique_motors = [f"axis_{i}" for i in range(len(self.goniometer_axes))]

        if refine_sample:
            if refine_goniometer_axes is not None and self.goniometer_names is not None:
                # 1. Match requested axes against unique motors
                motor_mask = [
                    any(req.lower() in name.lower() for req in refine_goniometer_axes)
                    for name in unique_motors
                ]
                # 2. Expand motor mask to physical axis mask
                axis_mask = [motor_mask[m_idx] for m_idx in motor_map]
                new_params.append(np.full(sum(axis_mask) * 3, 0.5))
            else:
                num_trans = (
                    max(1, len(self.goniometer_axes))
                    if self.goniometer_axes is not None
                    else 1
                )
                new_params.append(np.full(num_trans * 3, 0.5))

        if b_ki is not None:
            self.ki_vec = b_ki
        if refine_beam:
            new_params.append(np.full(2, 0.5))

        if b_gonio_offsets is not None:
            if isinstance(b_gonio_offsets, dict) and self.goniometer_names is not None:
                # Re-use unique_motors built above
                self.base_gonio_offset = np.array(
                    [b_gonio_offsets.get(name, 0.0) for name in unique_motors]
                )
            else:
                self.base_gonio_offset = b_gonio_offsets

        if refine_goniometer:
            if refine_goniometer_axes is not None and self.goniometer_names is not None:
                # Rotations are optimized per-MOTOR, so we just use the motor_mask
                motor_mask = [
                    any(req.lower() in name.lower() for req in refine_goniometer_axes)
                    for name in unique_motors
                ]
                new_params.append(np.full(sum(motor_mask), 0.5))
            else:
                new_params.append(np.full(len(unique_motors), 0.5))

        return (
            np.concatenate([np.atleast_1d(p) for p in new_params])
            if new_params
            else np.array([])
        )

    def minimize(
        self,
        strategy_name: str,
        population_size: int = 1000,
        num_generations: int = 100,
        n_runs: int = 1,
        seed: int = 0,
        init_params: np.ndarray | None = None,
        refine_lattice: bool = False,
        lattice_bound_frac: float = 0.05,
        goniometer_axes: list | None = None,
        goniometer_angles: np.ndarray | None = None,
        refine_goniometer: bool = False,
        goniometer_bound_deg: float | list | np.ndarray = 5.0,
        goniometer_names: list | None = None,
        refine_goniometer_axes: list | None = None,
        refine_goniometer_axis_vector: list | None = None,
        goniometer_axis_vector_bound_deg: float | list | np.ndarray = 1.0,
        refine_goniometer_per_run: str | None = None,
        goniometer_per_run_bound_deg: float = 0.5,
        refine_goniometer_per_run_trans: bool = False,
        goniometer_per_run_trans_bound_meters: float = 0.002,
        per_run_frame_map: np.ndarray | None = None,
        refine_goniometer_harmonics: str | None = None,
        goniometer_harmonics_orders: list[int] | None = None,
        goniometer_harmonics_axes: str = "rocking",
        goniometer_harmonics_bound_deg: float = 0.5,
        refine_sample: bool = False,
        refine_goniometer_trans_axes: list | None = None,
        goniometer_trans_bound_meters: float | list | np.ndarray = 0.005,
        refine_beam: bool = False,
        beam_bound_deg: float = 1.0,
        batch_size: int | None = None,
        sigma_init: float | None = None,
        refine_detector: bool = False,
        detector_params: dict | None = None,
        peak_pixel_coords: dict | None = None,
        detector_trans_bound_meters: float = 0.005,
        detector_rot_bound_deg: float = 1.0,
        freeze_orientation: bool = False,
        no_index: bool | None = None,
        radial_weight: float = 1.0,
        radial_weight_poly: list | None = None,
        hkl_metric: str = "isotropic",
        hkl_metric_floor: float = 0.1,
        multi_gpu: bool = False,
        **kwargs,
    ):
        if goniometer_axes is None and self.goniometer_axes is not None:
            goniometer_axes = self.goniometer_axes
        if goniometer_angles is None and self.goniometer_angles is not None:
            goniometer_angles = self.goniometer_angles
        if goniometer_names is None and self.goniometer_names is not None:
            goniometer_names = self.goniometer_names

        # Determine if we should bypass indexing based on boolean or presence of hkls
        if no_index is None:
            no_index = self.hkl is not None and self.lambdas is not None

        self.no_index = no_index

        if self.no_index:
            print(
                "Bootstrapped solution detected. Bypassing integer search and minimizing via geometric vector displacement."
            )
            print("You can enable indexing with --index.")

        # Provide a dummy lab vector for metric initialization if angles were removed
        if (
            self.two_theta.size == 0
            and self.az_phi.size == 0
            and self.peak_xyz is not None
        ):
            v_norm = self.peak_xyz / np.linalg.norm(
                self.peak_xyz, axis=1, keepdims=True
            )
            kf_ki_dir_lab = v_norm.T
            num_obs = self.peak_xyz.shape[0]
        else:
            kf_ki_dir_lab = scattering_vector_from_angles(self.two_theta, self.az_phi)
            num_obs = kf_ki_dir_lab.shape[1]

        static_R_input = self.R if self.R is not None else np.eye(3)
        if self.run_indices is not None:
            max_run_id = int(np.max(self.run_indices))
            num_runs_range = max_run_id + 1
            unique_runs, first_indices = np.unique(self.run_indices, return_index=True)

            def has_variation(data, indices):
                if data is None:
                    return False
                for r in unique_runs:
                    mask = indices == r
                    if np.sum(mask) <= 1:
                        continue
                    subset = data[mask] if data.ndim == 2 else data[mask, ...]
                    if not np.allclose(subset, subset[0:1], atol=1e-7):
                        return True
                return False

            can_reduce_angles = (
                goniometer_angles is not None
                and goniometer_angles.shape[1] == num_obs
                and not has_variation(goniometer_angles.T, self.run_indices)
            )
            can_reduce_R = (
                self.R is not None
                and self.R.ndim == 3
                and self.R.shape[0] == num_obs
                and not has_variation(self.R, self.run_indices)
            )

            if can_reduce_angles:
                new_angles = np.zeros((goniometer_angles.shape[0], num_runs_range))
                new_angles[:] = goniometer_angles[:, first_indices[0:1]]
                new_angles[:, unique_runs] = goniometer_angles[:, first_indices]
                goniometer_angles = new_angles

            if can_reduce_R:
                new_R = np.zeros((num_runs_range, 3, 3))
                new_R[:] = self.R[first_indices[0:1]]
                new_R[unique_runs] = self.R[first_indices]
                static_R_input = new_R
            elif self.R is not None and self.R.ndim == 3 and self.R.shape[0] == num_obs:
                static_R_input = self.R
                self.run_indices = np.arange(num_obs, dtype=np.int32)
            elif (
                goniometer_angles is not None and goniometer_angles.shape[1] == num_obs
            ):
                self.run_indices = np.arange(num_obs, dtype=np.int32)

        # 1. Build the motor map to support 1:n axis-to-motor relationships
        motor_map = None
        unique_motors = []
        if self.goniometer_names is not None:
            motor_map = []
            for name in self.goniometer_names:
                if name not in unique_motors:
                    unique_motors.append(name)
                motor_map.append(unique_motors.index(name))
        else:
            # Fallback to 1:1 mapping if names are missing
            if goniometer_axes is not None:
                motor_map = list(range(len(goniometer_axes)))
                unique_motors = [f"axis_{i}" for i in range(len(goniometer_axes))]

        # Ensure bounds is a structured list
        if isinstance(goniometer_bound_deg, (int, float)):
            gonio_bounds_list = [float(goniometer_bound_deg)]
        else:
            gonio_bounds_list = list(goniometer_bound_deg)

        # Initialize the global bounds array using the first value as a default fallback
        bounds_array = np.full(
            len(unique_motors), gonio_bounds_list[0] if gonio_bounds_list else 5.0
        )

        goniometer_refine_mask = None
        if refine_goniometer and refine_goniometer_axes is not None:
            mask = [False] * len(unique_motors)
            for i, name in enumerate(unique_motors):
                for req_idx, req in enumerate(refine_goniometer_axes):
                    # --- FIX: Case-insensitive match ---
                    if req.lower() in name.lower():
                        mask[i] = True
                        if len(gonio_bounds_list) == len(refine_goniometer_axes):
                            bounds_array[i] = gonio_bounds_list[req_idx]
            goniometer_refine_mask = np.array(mask, dtype=bool)
        elif refine_goniometer:
            goniometer_refine_mask = np.ones(len(unique_motors), dtype=bool)
            if len(gonio_bounds_list) == len(unique_motors):
                bounds_array = np.array(gonio_bounds_list)

        # Axis-vector refinement: per-motor mask and tilt bounds, matched
        # by the same case-insensitive name rule as the angular offsets.
        axis_vector_mask = None
        if refine_goniometer_axis_vector:
            av_bounds_raw = (
                [float(goniometer_axis_vector_bound_deg)]
                if isinstance(goniometer_axis_vector_bound_deg, (int, float))
                else list(goniometer_axis_vector_bound_deg)
            )
            axis_vector_mask = np.zeros(len(unique_motors), dtype=bool)
            bounds_array_axis_vec = np.full(
                len(unique_motors), av_bounds_raw[0] if av_bounds_raw else 1.0
            )
            for i, name in enumerate(unique_motors):
                for req_idx, req in enumerate(refine_goniometer_axis_vector):
                    if req.lower() in name.lower():
                        axis_vector_mask[i] = True
                        if len(av_bounds_raw) == len(refine_goniometer_axis_vector):
                            bounds_array_axis_vec[i] = av_bounds_raw[req_idx]
        else:
            bounds_array_axis_vec = np.full(len(unique_motors), 1.0)

        # Per-run corrections: resolve the single target motor by the same
        # case-insensitive name rule.
        per_run_motor_index = None
        if refine_goniometer_per_run:
            for i, name in enumerate(unique_motors):
                if refine_goniometer_per_run.lower() in name.lower():
                    per_run_motor_index = i
                    break
            if per_run_motor_index is None:
                raise ValueError(
                    f"Motor {refine_goniometer_per_run!r} not found in "
                    f"{unique_motors} for per-run refinement."
                )
            if per_run_frame_map is None:
                raise ValueError(
                    "Per-run refinement needs per_run_frame_map "
                    "(frame index -> run ordinal)."
                )
        trans_refine_mask = None
        if refine_goniometer_trans_axes:
            trans_refine_mask = np.array(
                [
                    any(
                        req.lower() in name.lower()
                        for req in refine_goniometer_trans_axes
                    )
                    for name in unique_motors
                ],
                dtype=bool,
            )
            if not trans_refine_mask.any():
                raise ValueError(
                    f"None of {refine_goniometer_trans_axes} matched motors "
                    f"{unique_motors} for translation refinement."
                )

        if refine_goniometer_per_run_trans and per_run_frame_map is None:
            raise ValueError(
                "Per-run translation refinement needs per_run_frame_map "
                "(frame index -> run ordinal)."
            )

        # Fourier-in-phi rocking: resolve the scan motor, take its axis
        # direction and the beam to build the rocking axes, and hand the
        # scan angle per frame to the objective as the harmonic phase.
        harmonic_axes_mat = None
        harmonic_orders_list = None
        harmonic_frame_angles = None
        if refine_goniometer_harmonics:
            if goniometer_axes is None or goniometer_angles is None:
                raise ValueError(
                    "Harmonic rocking refinement needs goniometer axes and angles."
                )
            harm_motor_index = None
            for i, name in enumerate(unique_motors):
                if refine_goniometer_harmonics.lower() in name.lower():
                    harm_motor_index = i
                    break
            if harm_motor_index is None:
                raise ValueError(
                    f"Motor {refine_goniometer_harmonics!r} not found in "
                    f"{unique_motors} for harmonic refinement."
                )
            harm_axis_row = motor_map.index(harm_motor_index)
            axis_arr = np.asarray(goniometer_axes[harm_axis_row], dtype=float)
            scan_dir = axis_arr[0:3] * (axis_arr[3] if axis_arr.shape[0] > 3 else 1.0)
            ki_nominal = np.asarray(
                self.ki_vec if self.ki_vec is not None else [0.0, 0.0, 1.0],
                dtype=float,
            )
            harmonic_axes_mat = harmonic_axes_from_scan(
                scan_dir, ki_nominal, goniometer_harmonics_axes
            )
            # The full band 1..6 by default: crystallographic rotation
            # orders top out at 6 (cubic included), and a symmetry axis
            # tilted from the scan axis leaks its harmonic n into the
            # n +/- 1 sidebands, so multiples of n alone are not enough.
            harmonic_orders_list = (
                [int(m) for m in goniometer_harmonics_orders]
                if goniometer_harmonics_orders
                else list(range(1, 7))
            )
            harmonic_frame_angles = np.asarray(goniometer_angles, dtype=float)[
                harm_axis_row, :
            ]

        # Map translation bounds to active axes
        if isinstance(goniometer_trans_bound_meters, (int, float)):
            gonio_trans_bounds_list = [float(goniometer_trans_bound_meters)]
        else:
            gonio_trans_bounds_list = list(goniometer_trans_bound_meters)

        bounds_array_trans = np.full(
            len(unique_motors),
            gonio_trans_bounds_list[0] if gonio_trans_bounds_list else 0.005,
        )

        if refine_sample and refine_goniometer_axes is not None:
            for i, name in enumerate(unique_motors):
                for req_idx, req in enumerate(refine_goniometer_axes):
                    # --- FIX: Case-insensitive match ---
                    if req.lower() in name.lower():
                        if len(gonio_trans_bounds_list) == len(refine_goniometer_axes):
                            bounds_array_trans[i] = gonio_trans_bounds_list[req_idx]
        elif refine_sample:
            if len(gonio_trans_bounds_list) == len(unique_motors):
                bounds_array_trans = np.array(gonio_trans_bounds_list)

        cell_params_init = np.array(
            [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]
        )
        lattice_system, num_lattice_params, centering = get_lattice_system(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.space_group
        )

        if refine_detector:
            if detector_params is None or peak_pixel_coords is None:
                raise ValueError(
                    "To use --refine-detector, detector_params and peak_pixel_coords must be passed from parser.py"
                )

        objective = VectorizedObjective(
            self.reciprocal_lattice_B(),
            kf_ki_dir_lab,
            self.peak_xyz,
            np.array(self.wavelength),
            cell_params=cell_params_init,
            refine_lattice=refine_lattice,
            lattice_bound_frac=lattice_bound_frac,
            lattice_system=lattice_system,
            centering=centering,
            motor_map=motor_map,
            goniometer_axes=goniometer_axes,
            goniometer_angles=goniometer_angles,
            refine_goniometer=refine_goniometer,
            goniometer_refine_mask=goniometer_refine_mask,
            goniometer_trans_refine_mask=trans_refine_mask,
            goniometer_nominal_offsets=self.base_gonio_offset,
            goniometer_bound_deg=bounds_array,
            goniometer_axis_vector_mask=axis_vector_mask,
            goniometer_axis_vector_bound_deg=bounds_array_axis_vec,
            per_run_motor_index=per_run_motor_index,
            per_run_frame_map=per_run_frame_map,
            per_run_bound_deg=goniometer_per_run_bound_deg,
            per_run_trans=refine_goniometer_per_run_trans,
            per_run_trans_bound_m=goniometer_per_run_trans_bound_meters,
            harmonic_frame_angles_deg=harmonic_frame_angles,
            harmonic_axes=harmonic_axes_mat,
            harmonic_orders=harmonic_orders_list,
            harmonic_bound_deg=goniometer_harmonics_bound_deg,
            goniometer_trans_bound_meters=bounds_array_trans,
            sample_nominal=self.base_sample_offset,
            refine_beam=refine_beam,
            refine_sample=refine_sample,
            beam_bound_deg=beam_bound_deg,
            beam_nominal=self.ki_vec,
            static_R=static_R_input,
            kf_lab_fixed_vectors=kf_ki_dir_lab,
            peak_run_indices=self.run_indices,
            refine_detector=refine_detector,
            detector_params=detector_params,
            peak_pixel_coords=peak_pixel_coords,
            detector_trans_bound_meters=detector_trans_bound_meters,
            detector_rot_bound_deg=detector_rot_bound_deg,
            freeze_orientation=freeze_orientation,
            fixed_rot_params=self.fixed_rot_params,
            no_index=self.no_index,
            hkl_fixed=self.hkl,
            lambda_fixed=self.lambdas,
            radial_weight=radial_weight,
            radial_weight_poly=radial_weight_poly,
            hkl_metric=hkl_metric,
            hkl_metric_floor=hkl_metric_floor,
        )

        num_dims = 0 if freeze_orientation else 3
        if refine_lattice:
            num_dims += num_lattice_params
        if refine_sample and self.peak_xyz is not None:
            trans_motor_mask = (
                trans_refine_mask
                if trans_refine_mask is not None
                else goniometer_refine_mask
            )
            if trans_motor_mask is not None and goniometer_axes is not None:
                # motor_map exists here, map mask to axes
                axis_mask = trans_motor_mask[motor_map]
                num_dims += np.sum(axis_mask) * 3
            else:
                num_trans = (
                    max(1, len(goniometer_axes)) if goniometer_axes is not None else 1
                )
                num_dims += num_trans * 3
        if refine_beam and self.peak_xyz is not None:
            num_dims += 2
        if refine_goniometer:
            num_dims += (
                np.sum(goniometer_refine_mask)
                if goniometer_refine_mask is not None
                else len(goniometer_axes)
            )
        if axis_vector_mask is not None:
            num_dims += 2 * int(np.sum(axis_vector_mask))
        if per_run_motor_index is not None:
            num_dims += objective.num_per_run_params
        if refine_goniometer_per_run_trans:
            num_dims += objective.num_per_run_trans_params
        if harmonic_axes_mat is not None:
            num_dims += objective.num_harmonic_params
        if refine_detector:
            num_dims += objective.num_det_params

        if num_dims == 0:
            print(
                "No parameters selected for refinement. Evaluating static physical model..."
            )
            self.x = np.array([])
            x_batch = jnp.zeros((1, 0))
            (
                UB_final_batch,
                B_new_batch,
                t_axes_batch,
                sample_origin_lab_batch,
                ki_vec_batch,
                offsets_total_batch,
                R_batch,
                dyn_centers,
                dyn_uhats,
                dyn_vhats,
                dyn_widths,
                dyn_heights,
                area_scale,
                axes_refined_batch,
                axis_tilts_batch,
                per_run_delta_batch,
                per_run_trans_batch,
                harmonic_coeffs_batch,
            ) = objective._get_physical_params_jax(x_batch)
            self.sample_offset = np.array(t_axes_batch[0])
            self.ki_vec = np.array(ki_vec_batch[0]).flatten()
            if offsets_total_batch is not None:
                self.goniometer_offsets = np.array(offsets_total_batch[0])
            if R_batch is not None:
                self.R = np.array(R_batch[0])

            rot_params = self.fixed_rot_params if freeze_orientation else np.zeros(3)
            U = objective.orientation_U_jax(rot_params[None])[0]

            loss_score, dist_min, hkl, lamb = objective.get_results(x_batch)
            dist_min_final = np.array(dist_min[0])
            # Soft indexing measures fractional-hkl distance; the fixed-hkl
            # geometric loss measures an angular chord, cut at the 1 degree
            # line the metrics report already calls BAD.
            threshold = np.deg2rad(1.0) if objective.no_index else 0.15
            mask = dist_min_final < threshold
            num_indexed = int(np.sum(mask))
            hkl_final = np.array(hkl[0])
            hkl_final[~mask] = 0

            return num_indexed, hkl_final, np.array(lamb[0]), np.array(U)

        has_start_sol = False
        if init_params is not None and len(init_params) > 0:
            start_sol = jnp.array(init_params)
            if start_sol.shape[0] < num_dims:
                start_sol_processed = jnp.concatenate(
                    [start_sol, jnp.full((num_dims - start_sol.shape[0],), 0.5)]
                )
            elif start_sol.shape[0] > num_dims:
                start_sol_processed = start_sol[:num_dims]
            else:
                start_sol_processed = start_sol
            has_start_sol = True
            target_sigma = sigma_init or 0.01
        else:
            # Create a dummy array to satisfy JAX typing compilation,
            # but we won't use its values because we branch on has_start_sol
            start_sol_processed = jnp.full((num_dims,), 0.5)
            target_sigma = sigma_init or 3.14

        sample_solution = jnp.zeros(num_dims)

        if strategy_name.lower() == "de":
            strategy = DifferentialEvolution(
                solution=sample_solution, population_size=population_size
            )
            strategy_type = "population_based"
        elif strategy_name.lower() == "pso":
            strategy = PSO(solution=sample_solution, population_size=population_size)
            strategy_type = "population_based"
        elif strategy_name.lower() == "cma_es":
            strategy = CMA_ES(solution=sample_solution, population_size=population_size)
            strategy_type = "distribution_based"
        elif strategy_name.lower() == "guided_es":
            strategy = GuidedES(
                solution=sample_solution, population_size=population_size
            )
            strategy_type = "distribution_based"
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        es_params = strategy.default_params

        def init_single_run(rng, start_sol):
            rng, rng_pop, rng_init = jax.random.split(rng, 3)

            if has_start_sol:
                if strategy_type == "population_based":
                    noise = (
                        jax.random.normal(rng_pop, (population_size, num_dims))
                        * target_sigma
                    )
                    if freeze_orientation:
                        population_init = jnp.clip(start_sol + noise, 0.0, 1.0)
                    else:
                        population_init = jnp.concatenate(
                            [
                                start_sol[:3] + noise[:, :3],
                                jnp.clip(start_sol[3:] + noise[:, 3:], 0.0, 1.0),
                            ],
                            axis=1,
                        )
                    state = strategy.init(
                        rng_init, population_init, objective(population_init), es_params
                    )
                else:
                    state = strategy.init(rng_init, start_sol, es_params).replace(
                        std=target_sigma
                    )
            else:
                if strategy_type == "population_based":
                    if freeze_orientation:
                        population_init = jax.random.uniform(
                            rng_pop, (population_size, num_dims)
                        )
                    else:
                        pop_orient = (
                            jax.random.normal(rng_pop, (population_size, 3))
                            * target_sigma
                        )
                        pop_rest = jax.random.uniform(
                            jax.random.split(rng_pop)[0],
                            (population_size, max(0, num_dims - 3)),
                        )
                        population_init = jnp.concatenate(
                            [pop_orient, pop_rest], axis=1
                        )
                    state = strategy.init(
                        rng_init, population_init, objective(population_init), es_params
                    )
                else:
                    if freeze_orientation:
                        solution_init = jnp.full((num_dims,), 0.5)
                    else:
                        solution_init = jnp.concatenate(
                            [jnp.zeros(3), jnp.full((max(0, num_dims - 3),), 0.5)]
                        )
                    state = strategy.init(rng_init, solution_init, es_params).replace(
                        std=target_sigma
                    )
            return state

        def step_single_run(rng, state):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)

            if strategy_name.lower() == "guided_es":
                # Ensure the mean respects your clipping logic before evaluating gradient
                if freeze_orientation:
                    mean_valid = jnp.clip(state.mean, 0.0, 1.0)
                else:
                    mean_valid = jnp.concatenate(
                        [state.mean[:3], jnp.clip(state.mean[3:], 0.0, 1.0)]
                    )

                # Compute gradient (matches your existing BFGS logic)
                grad_fn = jax.grad(lambda x_flat: objective(x_flat[None, :])[0])
                g = grad_fn(mean_valid)

                # Feed the gradient into the GuidedES state
                state = state.replace(grad=g)

            x, state_ask = strategy.ask(rng_ask, state, es_params)
            if freeze_orientation:
                x_valid = jnp.clip(x, 0.0, 1.0)
            else:
                x_valid = jnp.concatenate(
                    [x[:, :3], jnp.clip(x[:, 3:], 0.0, 1.0)], axis=1
                )

            state_tell, metrics = strategy.tell(
                rng_tell, x_valid, objective(x_valid), state_ask, es_params
            )
            return rng, state_tell, metrics

        init_batch_jit = jax.jit(jax.vmap(init_single_run, in_axes=(0, None)))
        step_batch_jit = jax.jit(jax.vmap(step_single_run, in_axes=(0, 0)))

        exec_batch_size = batch_size if batch_size is not None else n_runs

        # Opted in via multi_gpu, the vmapped run axis is sharded across the
        # visible devices.  Runs is the outer, embarrassingly parallel axis
        # here -- the population lives *inside* each run's strategy state, so
        # sharding it would reach into evosax internals, while sharding runs
        # only touches the batch dimension the code already vmaps over.  Both
        # the batch size and the run count are rounded up to a multiple of the
        # device count; the rounding launches extra *real* runs with fresh
        # seeds rather than padding with dummies, so it can only improve the
        # best-of-N result it feeds.
        devices = device_util.batch_devices(multi_gpu)
        n_dev = len(devices)
        n_runs_launched = n_runs
        run_sharding = None
        if n_dev > 1:
            exec_batch_size = -(-exec_batch_size // n_dev) * n_dev
            n_runs_launched = -(-n_runs // n_dev) * n_dev
            run_sharding = device_util.batch_sharding(devices)
            print(
                f"Sharding {n_runs_launched} optimization runs "
                f"(batches of {exec_batch_size}) across {n_dev} devices"
            )

        seeds = jnp.arange(seed, seed + n_runs_launched)
        all_keys = jax.vmap(jax.random.PRNGKey)(seeds)
        batch_keys_list, batch_states_list = [], []

        for b_i in range(int(np.ceil(n_runs_launched / exec_batch_size))):
            start_idx, end_idx = (
                b_i * exec_batch_size,
                min((b_i + 1) * exec_batch_size, n_runs_launched),
            )
            batch_keys = all_keys[start_idx:end_idx]
            if run_sharding is not None:
                # Every batch length is a multiple of n_dev (both operands
                # were rounded up), so the shards are equal-sized.
                batch_keys = jax.device_put(batch_keys, run_sharding)
            batch_keys_list.append(batch_keys)
            batch_states_list.append(
                init_batch_jit(batch_keys_list[-1], start_sol_processed)
            )

        pbar = (
            trange(num_generations, desc="Optimizing")
            if trange
            else range(num_generations)
        )
        for gen in pbar:
            current_gen_best = np.inf
            for b_i in range(len(batch_keys_list)):
                batch_keys_list[b_i], batch_states_list[b_i], _ = step_batch_jit(
                    batch_keys_list[b_i], batch_states_list[b_i]
                )
                current_gen_best = min(
                    current_gen_best, jnp.min(batch_states_list[b_i].best_fitness)
                )
            if trange:
                pbar.set_description(
                    f"Gen {gen + 1} | Best Loss: {current_gen_best:.5f}"
                )

        all_loss = jnp.concatenate([b.best_fitness for b in batch_states_list], axis=0)
        all_solutions = jnp.concatenate(
            [b.best_solution for b in batch_states_list], axis=0
        )

        best_idx = np.argmin(all_loss)
        best_overall_loss, best_overall_member = (
            all_loss[best_idx],
            all_solutions[best_idx],
        )

        print("Polishing solution with BFGS refinement...")
        from scipy.optimize import minimize as scipy_minimize

        res_ref = scipy_minimize(
            lambda x_flat: float(objective(x_flat[None, :])[0]),
            np.array(best_overall_member),
            jac=lambda x_flat: np.array(
                jax.grad(lambda x: objective(x[None, :])[0])(x_flat)
            ),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0) for _ in range(num_dims)]
            if freeze_orientation
            else [(0.0, 1.0) if i >= 3 else (None, None) for i in range(num_dims)],
            options={"maxiter": 50},
        )

        if res_ref.success and res_ref.fun < best_overall_loss:
            best_overall_member, best_overall_loss = res_ref.x, res_ref.fun

        self.x = np.array(best_overall_member)
        x_batch = jnp.array(self.x[None, :])
        (
            UB_final_batch,
            B_new_batch,
            t_axes_batch,
            sample_origin_lab_batch,
            ki_vec_batch,
            offsets_total_batch,
            R_batch,
            dyn_centers,
            dyn_uhats,
            dyn_vhats,
            dyn_widths,
            dyn_heights,
            area_scale,
            axes_refined_batch,
            axis_tilts_batch,
            per_run_delta_batch,
            per_run_trans_batch,
            harmonic_coeffs_batch,
        ) = objective._get_physical_params_jax(x_batch)

        self.sample_offset = np.array(t_axes_batch[0])
        self.ki_vec = np.array(ki_vec_batch[0]).flatten()
        if offsets_total_batch is not None:
            raw_offsets = np.array(offsets_total_batch[0])
            if self.goniometer_names is not None:
                unique_motors = []
                for name in self.goniometer_names:
                    if name not in unique_motors:
                        unique_motors.append(name)
                self.goniometer_offsets = {
                    name: float(val) for name, val in zip(unique_motors, raw_offsets)
                }
            else:
                self.goniometer_offsets = raw_offsets
        if R_batch is not None:
            self.R = np.array(R_batch[0])

        if axis_vector_mask is not None and axes_refined_batch is not None:
            axes_full = np.array(objective.gonio_axes)
            axes_full[:, 0:3] = np.array(axes_refined_batch[0])
            self.goniometer_axes_refined = axes_full
            tilts_deg = np.rad2deg(np.array(axis_tilts_batch[0]))
            if goniometer_names is not None:
                unique = []
                for name in goniometer_names:
                    if name not in unique:
                        unique.append(name)
                self.goniometer_axis_tilts = {
                    name: tilts_deg[i].tolist() for i, name in enumerate(unique)
                }
            else:
                self.goniometer_axis_tilts = tilts_deg
            for name, (ta, tb) in (
                self.goniometer_axis_tilts.items()
                if isinstance(self.goniometer_axis_tilts, dict)
                else enumerate(self.goniometer_axis_tilts)
            ):
                if abs(ta) > 1e-6 or abs(tb) > 1e-6:
                    print(f"Refined axis tilt {name}: ({ta:+.4f}, {tb:+.4f}) deg")

        if per_run_motor_index is not None and per_run_delta_batch is not None:
            self.goniometer_per_run_delta = np.array(per_run_delta_batch[0])
            self.goniometer_per_run_motor = refine_goniometer_per_run
            print(
                f"Per-run corrections for {refine_goniometer_per_run}: "
                + " ".join(f"{d:+.4f}" for d in self.goniometer_per_run_delta)
                + " deg"
            )
        if refine_goniometer_per_run_trans and per_run_trans_batch is not None:
            self.goniometer_per_run_trans = np.array(per_run_trans_batch[0])
            norms = np.linalg.norm(self.goniometer_per_run_trans, axis=1) * 1e3
            print(
                "Per-run sample displacements (|t| mm): "
                + " ".join(f"{v:.3f}" for v in norms)
            )

        if harmonic_axes_mat is not None and harmonic_coeffs_batch is not None:
            coeffs = np.array(harmonic_coeffs_batch[0])
            self.goniometer_harmonics = {
                "motor": refine_goniometer_harmonics,
                "orders": np.asarray(harmonic_orders_list, dtype=np.int32),
                "axes": np.asarray(harmonic_axes_mat, dtype=float),
                "coeffs_deg": coeffs,
            }
            amp = np.hypot(coeffs[..., 0], coeffs[..., 1])
            for k in range(amp.shape[0]):
                print(
                    f"Harmonic rocking axis {np.round(harmonic_axes_mat[k], 3)}: "
                    + " ".join(
                        f"m={m}: {a:.4f}" for m, a in zip(harmonic_orders_list, amp[k])
                    )
                    + " deg"
                )

        if freeze_orientation:
            rot_params = self.fixed_rot_params
        else:
            rot_params = self.x[:3]
        U = objective.orientation_U_jax(rot_params[None])[0]

        if refine_lattice:
            cell_norm = jnp.array(
                self.x[
                    None,
                    (0 if freeze_orientation else 3) : (0 if freeze_orientation else 3)
                    + num_lattice_params,
                ]
            )
            p_full = np.array(objective.reconstruct_cell_params(cell_norm)[0])
            self.a, self.b, self.c = p_full[:3]
            self.alpha, self.beta, self.gamma = p_full[3:]

        if refine_detector:
            self.calibrated_centers = np.array(dyn_centers[0])
            self.calibrated_uhats = np.array(dyn_uhats[0])
            self.calibrated_vhats = np.array(dyn_vhats[0])
            self.calibrated_widths = np.array(dyn_widths[0])
            self.calibrated_heights = np.array(dyn_heights[0])
            max_drift = (
                np.max(
                    np.linalg.norm(
                        self.calibrated_centers - detector_params["centers"], axis=1
                    )
                )
                * 1000
            )
            print("--- Refined Detector Geometry ---")
            print(f"Max Center Translation: {max_drift:.3f} mm")

        loss_score, dist_min, hkl, lamb = objective.get_results(x_batch)
        dist_min_final = np.array(dist_min[0])

        threshold = np.deg2rad(1.0) if objective.no_index else 0.15
        mask = dist_min_final < threshold
        num_indexed = int(np.sum(mask))

        hkl_final = np.array(hkl[0])
        hkl_final[~mask] = 0

        print(f"Final Solution indexed {num_indexed}/{num_obs} peaks.")

        return num_indexed, hkl_final, np.array(lamb[0]), np.array(U)
