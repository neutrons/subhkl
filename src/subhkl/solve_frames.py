"""Frame-addressed kinematics for count solving and its consumers."""

import numpy as np

from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix, sample_to_lab


def frame_kinematics(f, n_frames, file_offsets=None):
    """Read corrected R and sample origins, applying unfurled corrections once.

    Canonical angles with angles_nominal already include per-run deltas.
    Per-run translations are sample-frame displacements (the innermost lever
    arm in the existing predictor), not lab-frame detector translations.
    """

    def text(v):
        return v.decode() if isinstance(v, bytes) else str(v)

    def frames(value, tail):
        value = np.asarray(value, float).reshape((-1,) + tail)
        if len(value) == 1:
            value = np.repeat(value, n_frames, axis=0)
        if len(value) != n_frames or not np.isfinite(value).all():
            raise ValueError("goniometer metadata must be finite and frame-addressed")
        return value

    offsets = np.asarray(file_offsets if file_offsets is not None else [0], int)
    if (
        offsets.ndim != 1
        or len(offsets) == 0
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or offsets[-1] >= n_frames
    ):
        raise ValueError("invalid frame file_offsets")
    runs = np.searchsorted(offsets, np.arange(n_frames), side="right") - 1
    per_run = f.get("goniometer/per_run")
    if per_run is not None and "frame_to_run" in per_run:
        runs = np.asarray(per_run["frame_to_run"][()], int)
        if runs.shape != (n_frames,) or np.any(runs < 0):
            raise ValueError("invalid goniometer/per_run/frame_to_run")
    elif per_run is not None and file_offsets is None:
        raise ValueError(
            "per-run corrections require a frame_to_run map or file_offsets"
        )

    axes = f["goniometer/axes"][()] if "goniometer/axes" in f else None
    angles = None
    zeros = None
    if "goniometer/angles" in f:
        if axes is None:
            raise ValueError("goniometer angles require axes")
        axes = np.asarray(axes, float)
        if (
            axes.ndim != 2
            or axes.shape[1] != 4
            or not np.isfinite(axes).all()
            or np.any(np.linalg.norm(axes[:, :3], axis=1) == 0)
        ):
            raise ValueError("invalid goniometer axes")
        angles = frames(f["goniometer/angles"][()], (len(axes),))
        zeros = np.zeros(len(axes))
        names = (
            [text(v) for v in f["goniometer/names"][()]]
            if "goniometer/names" in f
            else []
        )
        if "goniometer/offsets" in f:
            stored = f["goniometer/offsets"]
            if hasattr(stored, "keys"):
                if len(names) != len(axes):
                    raise ValueError("named offsets require goniometer names")
                seen = {}
                for i, name in enumerate(names):
                    seen[name] = seen.get(name, 0) + 1
                    key = name if seen[name] == 1 else f"{name}_{seen[name]}"
                    if key in stored:
                        zeros[i] = float(stored[key][()])
            else:
                zeros = np.asarray(stored[()], float)
                if zeros.shape != (len(axes),):
                    raise ValueError("invalid goniometer offsets")
    folded = "goniometer/angles_nominal" in f
    changed = False
    if per_run is not None:
        delta = None
        if "delta_deg_all_axes" in per_run:
            delta = np.asarray(per_run["delta_deg_all_axes"][()], float)
        elif "delta_deg" in per_run:
            if angles is None or "motor" not in per_run or not names:
                raise ValueError(
                    "per-run angular corrections require axes, angles and motor names"
                )
            motor = text(per_run["motor"][()])
            matches = [
                i for i, name in enumerate(names) if name.split(":")[-1] == motor
            ]
            if len(matches) != 1:
                raise ValueError("per-run motor must identify one axis")
            vals = np.asarray(per_run["delta_deg"][()], float)
            delta = np.zeros((len(vals), len(axes)))
            delta[:, matches[0]] = vals
        if delta is not None and not folded:
            if (
                angles is None
                or delta.ndim != 2
                or delta.shape[1] != len(axes)
                or len(delta) <= runs.max()
                or not np.isfinite(delta).all()
            ):
                raise ValueError("invalid per-run angular corrections")
            angles = angles + delta[runs]
            changed = True
    if "goniometer/R" in f and not changed:
        R = frames(f["goniometer/R"][()], (3, 3))
    elif angles is not None:
        R = np.stack([calc_goniometer_rotation_matrix(axes, a + zeros) for a in angles])
    else:
        R = np.repeat(np.eye(3)[None], n_frames, axis=0)
    if not np.allclose(R @ R.swapaxes(1, 2), np.eye(3), atol=1e-6) or not np.allclose(
        np.linalg.det(R), 1, atol=1e-6
    ):
        raise ValueError("goniometer R must contain proper rotations")
    origins = np.zeros((n_frames, 3))
    if "goniometer/translations" in f:
        shift = np.asarray(f["goniometer/translations"][()], float)
        if not np.isfinite(shift).all():
            raise ValueError("invalid sample translations")
        if shift.shape == (3,):
            origins = np.einsum("nij,j->ni", R, shift)
        elif axes is not None and angles is not None and shift.shape == (len(axes), 3):
            origins = np.stack(
                [sample_to_lab(np.zeros(3), axes, a, shift, zeros) for a in angles]
            )
        else:
            raise ValueError(
                "sample translations require a sample vector or one lever arm per axis"
            )
    if per_run is not None and "trans_m" in per_run:
        shift = np.asarray(per_run["trans_m"][()], float)
        if (
            shift.ndim != 2
            or shift.shape[1] != 3
            or len(shift) <= runs.max()
            or not np.isfinite(shift).all()
        ):
            raise ValueError("invalid per-run sample translations")
        origins += np.einsum("nij,nj->ni", R, shift[runs])
    return R, origins, runs, angles, changed


def setting_groups(banks, rotations, origins, runs):
    """Separate exposures; never combine different rotations or sample origins."""
    groups = []
    for frame, bank in enumerate(banks):
        if (
            not groups
            or runs[frame] != runs[groups[-1][0]]
            or bank in banks[groups[-1]]
            or not np.allclose(
                rotations[frame], rotations[groups[-1][0]], atol=1e-8, rtol=0
            )
            or not np.allclose(
                origins[frame], origins[groups[-1][0]], atol=1e-10, rtol=0
            )
        ):
            groups.append([])
        groups[-1].append(frame)
    return [np.asarray(group, int) for group in groups]
