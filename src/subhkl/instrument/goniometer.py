"""
Compute goniometer rotation matrix from Euler angles.

Euler specification is loaded `reduction_settings.json`, and
angle values are loaded from .nxs.h5.

---

Notes
- Mantid `SetGoniometer` assumes that angles are *intrinsic* and composed
  in the same order as pushed onto m_motors, i.e., if the axis order is
  Axis0, Axis1, Axis2, with corresponding local rotation matrices R0, R1, R2,
  then the global rotation matrix is R = R0 * R1 * R2.
  (see https://github.com/mantidproject/mantid/blob/main/Framework/Geometry/src/Instrument/Goniometer.cpp#L346,
  which is called by `SetGoniometer` https://github.com/mantidproject/mantid/blob/main/Framework/Crystal/src/SetGoniometer.cpp#L181)
- Mantid `SetGoniometer` expects rotation axes to be specified by the axis
  local x,y,z coordinates (since angles are intrinsic) and an orientation
  o in {-1, 1}, which indicates whether the angle is taken in clockwise (-1)
  or counter-clockwise (1) sense about the axis. Data are packed in an array
  as [x, y, z, o] (see `reduction_settings.json` for examples).
- ??? The order of the axes in `reduction_settings.json` corresponds to the
  order they would be input into Mantid `SetGoniometer` ???
- Based on experimentation, `scipy.spatial.transform.Rotation.from_rotvec`
  constructs a rotation from a rotation vector according to counter-clockwise
  orientation in a right-handed coordinate system (you can achieve clockwise by
  negating the rotation vector and using `from_rotvec`). A rotation vector is
  easily constructed from the axis-angle obtained by combining the Euler angle
  specification [x, y, z, o] and the corresponding angle read from a .nxs.h5
  file.
- Mantid converts the axis-angle representation directly into a quaternion using
  the standard method (see https://github.com/mantidproject/mantid/blob/main/Framework/Kernel/src/Quat.cpp#L114)
  which results in the same sense of the rotation as the interpretation as a
  rotation counter-clockwise about the axis in a right-handed coordinate system
  (that is, it is consistent with constructing the rotation with from_rotvec)
  See here if you are interested: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Rotation_identity
"""

import re
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.config import reduction_settings

_TITLE_ANGLE_PATTERNS = {
    "phi": re.compile(r"(?:axis0|phi)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
    "kappa": re.compile(r"(?:axis1|kappa)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
    "omega": re.compile(r"(?:axis2|omega)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
    "chi": re.compile(r"(?:chi)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
    "alpha": re.compile(r"(?:alpha)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
}

# CG4D Attocube goniometer PV mapping (outer XYZ translation + Omega/Kappa/Phi rotation;
# underlying controllers: CG4D:SE:AMC300One and CG4D:SE:AMC300Two).
# Note: Until these PVs are synced in CG4D NeXus DASlogs, rotation angles are harvested
# from /entry/title first, with nominal 0° (and alpha=45° kappa-arm inclination) defaults.
_ATTOCUBE_TRANS_PVS = {
    "x": "CG4D:Mot:AttocubeGoni:Trans:ReadPosition0",
    "y": "CG4D:Mot:AttocubeGoni:Trans:ReadPosition1",
    "z": "CG4D:Mot:AttocubeGoni:Trans:ReadPosition2",
}

_ATTOCUBE_ROT_PVS = {
    "phi": "CG4D:Mot:AttocubeGoni:Rot:ReadPosition0",
    "kappa": "CG4D:Mot:AttocubeGoni:Rot:ReadPosition1",
    "omega": "CG4D:Mot:AttocubeGoni:Rot:ReadPosition2",
}

_CG4D_NOMINAL_DEFAULTS = {
    "omega": 0.0,
    "alpha": 45.0,
    "kappa": 0.0,
    "phi": 0.0,
    "chi": 0.0,
}


def harvest_goniometer_angles_from_title(title: str) -> dict[str, float]:
    """Harvest goniometer angles (phi/axis0, kappa/axis1, omega/axis2, chi, alpha) from a NeXus run title."""
    if not title:
        return {}
    parsed: dict[str, float] = {}
    for key, pat in _TITLE_ANGLE_PATTERNS.items():
        m = pat.search(str(title))
        if m:
            try:
                parsed[key] = float(m.group(1))
            except ValueError:
                pass
    return parsed


def _read_nexus_title(f: h5py.File) -> str:
    if "entry/title" not in f:
        return ""
    raw = f["entry/title"][()]
    if isinstance(raw, np.ndarray) and raw.size > 0:
        raw = raw.flat[0]
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def get_rotation_data_from_title(title: str, instrument: str):
    """Build (axes, angles, names) from a run title/angles string using reduction_settings[instrument]."""
    settings = reduction_settings[instrument]
    axes, angles, names = [], [], []
    title_angles = harvest_goniometer_angles_from_title(title or "")
    for axis_name, axis_spec in settings["Goniometer"].items():
        log_name = axis_name.split("#")[0]
        short_key = log_name.split(":")[-1].lower()
        angle_deg = None
        if short_key in title_angles:
            angle_deg = float(title_angles[short_key])
        elif (
            str(instrument).upper() in ("CG4D", "IMAGINE")
            and short_key in _CG4D_NOMINAL_DEFAULTS
        ):
            angle_deg = float(_CG4D_NOMINAL_DEFAULTS[short_key])
        if angle_deg is not None:
            axes.append(np.array(axis_spec, dtype=float))
            angles.append(angle_deg)
            names.append(log_name)
    return axes, angles, names


def get_rotation_data_from_nexus(filename, instrument):
    settings = reduction_settings[instrument]
    axes, angles, names = [], [], []
    with h5py.File(filename, "r") as f:
        das_logs = f.get("entry/DASlogs", {})
        title_str = _read_nexus_title(f)
        title_angles = harvest_goniometer_angles_from_title(title_str)

        for axis_name, axis_spec in settings["Goniometer"].items():
            # Strip the suffix to get the real NeXus log name
            log_name = axis_name.split("#")[0]
            short_key = log_name.split(":")[-1].lower()
            angle_deg = None

            if log_name in das_logs and "average_value" in das_logs[log_name]:
                try:
                    angle_deg = float(das_logs[log_name]["average_value"][0])
                except (ValueError, TypeError, KeyError, IndexError):
                    angle_deg = None

            if angle_deg is None and short_key in title_angles:
                angle_deg = float(title_angles[short_key])

            if (
                angle_deg is None
                and title_angles
                and str(instrument).upper() in ("CG4D", "IMAGINE")
                and short_key in _CG4D_NOMINAL_DEFAULTS
            ):
                angle_deg = float(_CG4D_NOMINAL_DEFAULTS[short_key])

            if angle_deg is None and short_key in _ATTOCUBE_ROT_PVS:
                atto_pv = _ATTOCUBE_ROT_PVS[short_key]
                if atto_pv in das_logs and "average_value" in das_logs[atto_pv]:
                    try:
                        angle_deg = float(das_logs[atto_pv]["average_value"][0])
                    except (ValueError, TypeError, KeyError, IndexError):
                        angle_deg = None

            if (
                angle_deg is None
                and str(instrument).upper() in ("CG4D", "IMAGINE")
                and short_key in _CG4D_NOMINAL_DEFAULTS
            ):
                angle_deg = float(_CG4D_NOMINAL_DEFAULTS[short_key])

            if angle_deg is not None:
                axis = np.array(axis_spec, dtype=float)
                angles.append(angle_deg)
                axes.append(axis)
                names.append(log_name)
            else:
                print(
                    f"Warning: Could not load goniometer axis {axis_name} from {filename} (title={title_str!r})"
                )

    return axes, angles, names


def calc_goniometer_rotation_matrix(axes, angles):
    """
    Computes the cumulative rotation matrix R for reciprocal space transformations.
    (Translations are ignored in momentum space).
    """
    R_cum = np.eye(3)
    deg2rad = np.pi / 180.0

    for i in range(len(axes)):
        direction = axes[i][:3]
        direction = direction / np.linalg.norm(direction)
        theta = axes[i][3] * angles[i] * deg2rad

        K = np.array(
            [
                [0, -direction[2], direction[1]],
                [direction[2], 0, -direction[0]],
                [-direction[1], direction[0], 0],
            ]
        )
        R_i = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        R_cum = R_cum @ R_i

    return R_cum


def sample_to_lab(p_sample, axes, angles, offsets, zero_offsets=None, is_vector=False):
    """
    Convert points or vectors from the Sample frame to the Lab frame.

    p_sample: (3,) or (N, 3) array of coordinates or vectors.
    axes: (M, 4) array [x, y, z, direction_multiplier].
    angles: (M,) or (M, N) array of raw encoder angles in degrees.
    offsets: (M, 3) array of translation lever arms.
    zero_offsets: (M,) array of zero-point angle calibrations in degrees.
    is_vector: If True, skips translations (used for directional rays like Q or ki).
    """
    p = np.array(p_sample, dtype=np.float32)
    num_axes = len(axes)

    if zero_offsets is None:
        zero_offsets = np.zeros(num_axes)

    angles = np.array(angles)
    is_batched_p = p.ndim == 2
    is_batched_a = angles.ndim == 2

    # Broadcast points to match the number of temporal angles if necessary
    if is_batched_a and not is_batched_p:
        p = np.tile(p, (angles.shape[1], 1))

    # Traverse from innermost (Sample, index M-1) to outermost (Lab, index 0)
    for i in reversed(range(num_axes)):
        direction = axes[i][:3]
        direction_mult = axes[i][3] if len(axes[i]) > 3 else 1.0

        axis_norm = np.linalg.norm(direction)
        if axis_norm > 0:
            direction = direction / axis_norm

        true_angle = angles[i] + zero_offsets[i]
        theta_rad = np.radians(true_angle * direction_mult)

        # Build SciPy Rotation (handles both scalar and N-dimensional arrays natively)
        if np.isscalar(theta_rad) or theta_rad.ndim == 0:
            R_i = Rotation.from_rotvec(theta_rad * direction)
        else:
            rotvecs = theta_rad[:, None] * direction[None, :]
            R_i = Rotation.from_rotvec(rotvecs)

        # Translate in the local frame (lever arm), THEN Rotate
        if not is_vector:
            p = p + offsets[i][:3]
        p = R_i.apply(p)

    return p


def lab_to_sample(p_lab, axes, angles, offsets, zero_offsets=None, is_vector=False):
    """
    Convert points or vectors from the Lab frame back to the Sample frame.
    """
    p = np.array(p_lab, dtype=np.float32)
    num_axes = len(axes)

    if zero_offsets is None:
        zero_offsets = np.zeros(num_axes)

    angles = np.array(angles)
    is_batched_p = p.ndim == 2
    is_batched_a = angles.ndim == 2

    if is_batched_a and not is_batched_p:
        p = np.tile(p, (angles.shape[1], 1))

    # Traverse from outermost (Lab, index 0) to innermost (Sample, index M-1)
    for i in range(num_axes):
        direction = axes[i][:3]
        direction_mult = axes[i][3] if len(axes[i]) > 3 else 1.0

        axis_norm = np.linalg.norm(direction)
        if axis_norm > 0:
            direction = direction / axis_norm

        true_angle = angles[i] + zero_offsets[i]
        theta_rad = np.radians(true_angle * direction_mult)

        if np.isscalar(theta_rad) or theta_rad.ndim == 0:
            R_i = Rotation.from_rotvec(theta_rad * direction)
        else:
            rotvecs = theta_rad[:, None] * direction[None, :]
            R_i = Rotation.from_rotvec(rotvecs)

        # INVERSE KINEMATICS: Reverse Rotate, THEN subtract local translation
        p = R_i.inv().apply(p)
        if not is_vector:
            p = p - offsets[i][:3]

    return p


@dataclass
class Goniometer:
    axes_raw: list[np.ndarray] = None
    angles_raw: list[float] | np.ndarray = None
    names_raw: list[str] = None
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    @classmethod
    def from_nexus(cls, filename, instrument):
        """
        Get goniometer axes and rotation angles from Nexus file

        Parameters
        ----------
        filename : str
            Name of nexus file to load angles from

        instrument : str
            Name of instrument used to collect data

        Returns
        -------
        axes : list[length 4 numpy array]
            List of axes in format used by Mantid `SetGoniometer`
        angles : list[float]
            List of angles in degrees about the axes
        names : list[str]
            List of axis names
        """
        axes, angles, names = get_rotation_data_from_nexus(filename, instrument)

        rotation = cls.get_rotation(axes, angles)

        return cls(
            axes_raw=axes,
            angles_raw=np.array(angles),
            names_raw=names,
            rotation=rotation,
        )

    @classmethod
    def from_h5_file(cls, h5_file):
        axes = h5_file["goniometer/axes"][()]
        angles = h5_file["goniometer/angles"][()]
        names = None
        if "goniometer/names" in h5_file:
            names = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in h5_file["goniometer/names"][()]
            ]

        # Handle 2D angles (stack of matrices) vs 1D angles
        if angles.ndim == 2:
            rotation = np.stack([cls.get_rotation(axes, ang) for ang in angles])
        else:
            rotation = cls.get_rotation(axes, angles)

        return cls(axes_raw=axes, angles_raw=angles, names_raw=names, rotation=rotation)

    @staticmethod
    def get_rotation(axes, angles):
        """
        Calculate the goniometer rotation matrix.

        Parameters
        ----------
        axes : list[list[float]]
            Parallel list of axes corresponding to the angles; each list is packed
            as in Mantid `SetGoniometer`.
        angles : list[float]
            List of the angles in degrees (in the same order as Mantid
            `SetGoniometer`)

        Returns
        -------
        matrix : 3x3 numpy array
            The goniometer rotation matrix
        """
        return calc_goniometer_rotation_matrix(axes, angles)
