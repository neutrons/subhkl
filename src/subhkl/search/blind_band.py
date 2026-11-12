"""Single-crystal discovery by band search, shared geometry, and full-data ranking."""

import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.config import beamlines
from subhkl.core.spacegroup import get_space_group_object
from subhkl.instrument.detector import Detector
from subhkl.scan_refine import load_observations
from subhkl.search.scan_band import band_sum, build_B_jax, direct_lattice_uvw
from subhkl.search.spherical import band_q_range, lattice_ladder, refine_band_geometry


def sample_signed(weights, size, seed=0):
    """Unbiased quadrature of a signed sum, sampling by absolute contribution.

    Every nonzero residual can be drawn; duplicates are combined. Returned
    weights estimate the original sum divided by its total absolute weight.
    No intensity cutoff or local-maximum selection is introduced.
    """
    weights = np.asarray(weights, float)
    total = np.abs(weights).sum()
    if size < 1 or not np.isfinite(weights).all() or total == 0:
        raise ValueError(
            "positive sample size and finite nonzero residual evidence required"
        )
    if len(weights) <= size:
        return np.arange(len(weights)), weights / total
    rng = np.random.default_rng(seed)
    drawn = rng.choice(len(weights), size=size, replace=True, p=np.abs(weights) / total)
    indices, counts = np.unique(drawn, return_counts=True)
    return indices, np.sign(weights[indices]) * counts / size


def distinct_orientations(candidates, B, space_group, count, separation_deg=1.0):
    """Remove duplicates under proper crystallographic rotations."""
    operations = []
    for op in get_space_group_object(space_group).operations().sym_ops:
        S = B @ np.linalg.inv(np.asarray(op.rot) / op.DEN).T @ np.linalg.inv(B)
        if np.linalg.det(S) > 0:
            operations.append(S)
    selected = []
    for candidate in candidates:
        U = candidate[0]
        if all(
            min(Rotation.from_matrix(U @ (V[0] @ S).T).magnitude() for S in operations)
            > np.deg2rad(separation_deg)
            for V in selected
        ):
            selected.append(candidate)
        if len(selected) >= count:
            break
    return selected


def count_search_weights(deviance, scale=25.0, max_quantile=None, max_factor=3.0):
    """Smooth chi-square-null-suppressed Poisson deviance weights for count bins.

    Under a Poisson background null, single-bin deviance follows chi^2_1 (~1),
    whereas genuine Bragg reflections in an 8x8 bin produce deviance >> 25.
    Weighting by sqrt(D) * (1 - exp(-(D / scale)^3)) is strictly threshold-free
    (every positive-deviance bin has positive weight), suppresses 1-sigma background
    fluctuations by ~6e-5 so importance sampling is not flooded by background bins,
    and grows as sqrt(D) at high deviance so bright flare pixels cannot monopolize
    the sample. When `max_quantile` is set, extreme unmasked beamstop-flare bins are
    capped at `max_factor` times that positive-weight quantile.
    """
    dev = np.maximum(np.asarray(deviance, float), 0.0)
    w = np.sqrt(dev) * (1.0 - np.exp(-((dev / float(scale)) ** 3)))
    if max_quantile is not None:
        pos = w[w > 0]
        if len(pos):
            w = np.minimum(w, float(max_factor) * float(np.quantile(pos, max_quantile)))
    return w


def discover(args, samples=16384, candidates=6, seed=0):
    """Find a seed without reading any reference orientation or calibration.

    Search on the stable-run subset; the subsequent scan fit estimates drift
    on all requested runs. Full-data scores compare every candidate under the
    same winning global geometry, not each candidate's preferred detector.
    """
    start = time.monotonic()
    observations = load_observations(args, return_kinematics=True)
    banks, rows, cols, dev, _, runs, _angles, weights, R_obs, _scan_axes = observations
    use = np.isin(runs, args.stable_runs)
    if args.finder:
        use &= dev >= args.min_deviance
        search_weights = weights[use]
        weighting_desc = "sqrt finder deviance"
    else:
        search_weights = count_search_weights(
            dev[use], max_quantile=0.999, max_factor=3.0
        )
        weighting_desc = "smooth chi2-suppressed bin Poisson deviance"
    banks, rows, cols, Rs = [v[use] for v in (banks, rows, cols, R_obs)]
    weights = search_weights
    idx, sampled_weights = sample_signed(weights, samples, seed)
    C, O = np.empty((len(weights), 3)), np.empty((len(weights), 3))
    for bank in np.unique(banks):
        m = banks == bank
        det = Detector(dict(beamlines[args.instrument][str(bank)]))
        C[m] = det.center + 0.5 * det.width * det.uhat + 0.5 * det.height * det.vhat
        O[m] = ((cols[m] / (det.m - 1) - 0.5) * det.width)[:, None] * det.uhat + (
            (rows[m] / (det.n - 1) - 0.5) * det.height
        )[:, None] * det.vhat
    Cs, Os, rotations = C[idx], O[idx], Rs[idx]
    B = np.asarray(build_B_jax(*args.cell[:3], *np.deg2rad(args.cell[3:])))
    wavelength = (args.wl_min, args.wl_max)
    pool = max(24, 4 * candidates)
    xyz = Cs + Os
    q = xyz / np.linalg.norm(xyz, axis=1, keepdims=True) - [0, 0, 1]
    lab = q / np.linalg.norm(q, axis=1, keepdims=True)
    directions = np.einsum("nji,nj->ni", rotations, lab)
    blind_path = Path(args.output).with_suffix(".blind.h5")
    src_nexus = getattr(args, "nexus", getattr(args, "input", ""))

    def _write_blind_rung(
        stage_label: str,
        nz_val: int,
        u_cand: np.ndarray,
        geom_fit: dict | None = None,
    ) -> None:
        tmp_path = blind_path.with_suffix(".blind.tmp.h5")
        try:
            with h5py.File(tmp_path, "w") as bf:
                bf.attrs["instrument"] = str(args.instrument)
                bf["sample/U"] = np.asarray(u_cand, dtype=float)
                bf["sample/B"] = np.asarray(B, dtype=float)
                bf["sample/cell"] = np.asarray(args.cell, dtype=float)
                for k_c, v_c in zip(
                    ("a", "b", "c", "alpha", "beta", "gamma"), args.cell[:6]
                ):
                    bf[f"sample/{k_c}"] = float(v_c)
                bf["sample/space_group"] = str(args.space_group)
                bf["instrument/name"] = str(args.instrument)
                bf["instrument/wavelength"] = np.asarray(
                    [args.wl_min, args.wl_max], dtype=float
                )
                if geom_fit is not None:
                    r_sc = float(geom_fit.get("radial_scale", 0.0))
                    r_vec = np.asarray(
                        geom_fit.get("global_rot_vec", np.zeros(3)), dtype=float
                    )
                    t_vec = np.asarray(
                        geom_fit.get("global_trans_m", np.zeros(3)), dtype=float
                    )
                    rot_m = Rotation.from_rotvec(r_vec).as_matrix()
                    cal_g = bf.create_group("detector_calibration")
                    for b_k, b_cfg in beamlines[args.instrument].items():
                        try:
                            b_id = int(b_k)
                        except (ValueError, TypeError):
                            continue
                        det_b = Detector(dict(b_cfg))
                        mid_b = (
                            det_b.center
                            + 0.5 * det_b.width * det_b.uhat
                            + 0.5 * det_b.height * det_b.vhat
                        )
                        mid_cal = mid_b * (1.0 + r_sc) @ rot_m.T + t_vec
                        uh_cal = det_b.uhat @ rot_m.T
                        vh_cal = det_b.vhat @ rot_m.T
                        cen_cal = (
                            mid_cal
                            - 0.5 * det_b.width * uh_cal
                            - 0.5 * det_b.height * vh_cal
                        )
                        bg = cal_g.create_group(f"bank_{b_id}")
                        bg["center"] = np.asarray(cen_cal, dtype=float)
                        bg["uhat"] = np.asarray(uh_cal, dtype=float)
                        bg["vhat"] = np.asarray(vh_cal, dtype=float)
                        bg["width"] = float(det_b.width)
                        bg["height"] = float(det_b.height)
                sg = bf.create_group("search")
                sg.attrs["source"] = str(Path(src_nexus).resolve()) if src_nexus else ""
                sg.attrs["instrument"] = str(args.instrument)
                sg.attrs["d_min"] = float(args.d_min)
                sg.attrs["ladder_stage"] = str(stage_label)
                sg.attrs["ladder_zones"] = int(nz_val)
            tmp_path.replace(blind_path)
            import os as _os

            if (
                _os.environ.get("SUBHKL_LIVE_STDIO") == "1"
                or "PYTEST_CURRENT_TEST" not in _os.environ
            ):
                import base64
                import json as _json
                import zlib

                live_doc: dict = {
                    "kind": "blind",
                    "instrument": str(args.instrument),
                    "space_group": str(args.space_group),
                    "wavelength": [float(args.wl_min), float(args.wl_max)],
                    "d_min": float(args.d_min),
                    "stage": str(stage_label),
                    "n_zones": int(nz_val),
                    "cell": [round(float(x), 5) for x in args.cell[:6]],
                    "U": np.round(np.asarray(u_cand, dtype=float), 6).tolist(),
                    "B": np.round(np.asarray(B, dtype=float), 6).tolist(),
                }
                if geom_fit is not None:
                    live_doc["radial_scale"] = round(
                        float(geom_fit.get("radial_scale", 0.0)), 6
                    )
                    live_doc["global_rot_vec"] = [
                        round(float(x), 6)
                        for x in np.asarray(
                            geom_fit.get("global_rot_vec", np.zeros(3)),
                            dtype=float,
                        ).ravel()[:3]
                    ]
                    live_doc["global_trans_m"] = [
                        round(float(x), 6)
                        for x in np.asarray(
                            geom_fit.get("global_trans_m", np.zeros(3)),
                            dtype=float,
                        ).ravel()[:3]
                    ]
                b64 = base64.b64encode(
                    zlib.compress(
                        _json.dumps(live_doc, separators=(",", ":")).encode("utf-8"),
                        level=9,
                    )
                ).decode("ascii")
                print(f"[subhkl-live-state] {b64}", flush=True)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
        ):
            tmp_path.unlink(missing_ok=True)

    print(
        f"Blind band search: {len(weights)} stable-run observations, {len(idx)} search observations (exhaustive rung init)",
        flush=True,
    )

    found = lattice_ladder(
        directions,
        sampled_weights,
        B,
        grid_deg=2.0,
        n_shortlist=1000,
        rungs=(
            (13, 0.5, 1.5, 300, 5),
            (30, 0.5, 1.0, 80, 5),
            (60, 0.3, 0.5, pool, 5),
            (145, 0.3, 0.25, pool, 5),
        ),
        sin_theta=-lab[:, 2],
        wavelength=wavelength,
        d_min=args.d_min,
        n_null=32,
        seed=seed,
        verbose=True,
        space_group=args.space_group,
        on_rung=_write_blind_rung,
    )
    found = distinct_orientations(found, B, args.space_group, candidates)
    if not found:
        raise ValueError("blind band search found no orientation candidates")
    print(
        f"Joint band refinement of {len(found)} distinct blind candidates", flush=True
    )
    rot_axis_only = getattr(args, "global_rot_axis", (0.0, 1.0, 0.0))
    modes = getattr(args, "modes", None)
    r_bound = 0.08 if (modes is None or "cylindrical" in modes) else 0.0
    rot_bound = 2.0 if (modes is None or "global_rot" in modes) else 0.0
    t_bound = 0.02 if (modes is None or "global_trans" in modes) else 0.0
    fits = []
    for i, item in enumerate(found):
        U = item[0]
        fit = refine_band_geometry(
            Cs,
            Os,
            sampled_weights,
            B,
            rotations,
            U,
            wavelength,
            args.d_min,
            radial_bound=r_bound,
            rot_bound_deg=rot_bound,
            trans_bound_m=t_bound,
            maxiter=150,
            rot_axis_only=rot_axis_only,
            space_group=args.space_group,
        )
        fits.append(fit)
        best_fit_so_far = max(fits, key=lambda f_it: f_it["score"])
        _write_blind_rung(
            f"joint_{i}", 60, best_fit_so_far["U"], geom_fit=best_fit_so_far
        )
        print(
            f"  candidate {i}: score {fit['score']:.6g}, radial {fit['radial_scale']:+.4f}",
            flush=True,
        )
    winner = int(np.argmax([fit["score"] for fit in fits]))
    geometry = fits[winner]
    rot = Rotation.from_rotvec(geometry["global_rot_vec"]).as_matrix()
    C_shared = C * (1 + geometry["radial_scale"]) @ rot.T + geometry["global_trans_m"]
    O_shared = O @ rot.T
    U_shared = []
    for fit in fits:
        local = refine_band_geometry(
            C_shared[idx],
            O_shared[idx],
            sampled_weights,
            B,
            rotations,
            fit["U"],
            wavelength,
            args.d_min,
            radial_bound=0,
            rot_bound_deg=0,
            trans_bound_m=0,
            orient_bound_deg=2.0,
            maxiter=100,
            space_group=args.space_group,
        )
        U_shared.append(local["U"])
    xyz = C_shared + O_shared
    q = xyz / np.linalg.norm(xyz, axis=1, keepdims=True) - [0, 0, 1]
    lab = q / np.linalg.norm(q, axis=1, keepdims=True)
    D = np.einsum("nji,nj->ni", Rs, lab)
    lo, hi = band_q_range(-lab[:, 2], wavelength, args.d_min)
    uvw = direct_lattice_uvw(B, space_group=args.space_group)[:60]
    X = uvw @ np.linalg.inv(B)
    w = weights / np.abs(weights).sum()
    import jax.numpy as jnp

    scores = [
        float(
            band_sum(
                jnp.asarray(X @ U.T),
                jnp.asarray(D),
                jnp.asarray(lo),
                jnp.asarray(hi),
                jnp.asarray(w),
                jnp.ones(len(X)),
            )
        )
        for U in U_shared
    ]
    selected = int(np.argmax(scores))
    path = Path(args.output).with_suffix(".blind.h5")
    tmp_final = path.with_suffix(".blind.tmp.h5")
    with h5py.File(tmp_final, "w") as f:
        f.attrs["instrument"] = str(args.instrument)
        f["sample/U"] = U_shared[selected]
        f["sample/B"] = B
        f["sample/cell"] = np.asarray(args.cell, dtype=float)
        for k_c, v_c in zip(("a", "b", "c", "alpha", "beta", "gamma"), args.cell[:6]):
            f[f"sample/{k_c}"] = float(v_c)
        f["sample/space_group"] = str(args.space_group)
        f["instrument/name"] = str(args.instrument)
        f["instrument/wavelength"] = np.asarray([args.wl_min, args.wl_max], dtype=float)
        r_sc = float(geometry.get("radial_scale", 0.0))
        t_vec = np.asarray(geometry.get("global_trans_m", np.zeros(3)), dtype=float)
        cal_g = f.create_group("detector_calibration")
        for b_k, b_cfg in beamlines[args.instrument].items():
            try:
                b_id = int(b_k)
            except (ValueError, TypeError):
                continue
            det_b = Detector(dict(b_cfg))
            mid_b = (
                det_b.center
                + 0.5 * det_b.width * det_b.uhat
                + 0.5 * det_b.height * det_b.vhat
            )
            mid_cal = mid_b * (1.0 + r_sc) @ rot.T + t_vec
            uh_cal = det_b.uhat @ rot.T
            vh_cal = det_b.vhat @ rot.T
            cen_cal = mid_cal - 0.5 * det_b.width * uh_cal - 0.5 * det_b.height * vh_cal
            bg = cal_g.create_group(f"bank_{b_id}")
            bg["center"] = np.asarray(cen_cal, dtype=float)
            bg["uhat"] = np.asarray(uh_cal, dtype=float)
            bg["vhat"] = np.asarray(vh_cal, dtype=float)
            bg["width"] = float(det_b.width)
            bg["height"] = float(det_b.height)
        g = f.create_group("search")
        g["orientations"] = U_shared
        g["full_scores"] = scores
        g["proposal_orientations"] = [v[0] for v in found]
        g["proposal_scores"] = [v[1] for v in found]
        g["joint_scores"] = [v["score"] for v in fits]
        g["sample_indices"] = idx
        g["sample_weights"] = sampled_weights
        g["shared_geometry"] = np.r_[
            geometry["radial_scale"],
            geometry["global_rot_vec"],
            geometry["global_trans_m"],
        ]
        g["candidate_geometries"] = np.array(
            [
                np.r_[
                    v["radial_scale"],
                    v["global_rot_vec"],
                    v["global_trans_m"],
                ]
                for v in fits
            ],
            dtype=np.float64,
        )
        g.attrs["selected"] = selected
        g.attrs["samples_requested"] = samples
        g.attrs["seed"] = seed
        g.attrs["stable_observations"] = len(weights)
        g.attrs["seconds"] = time.monotonic() - start
        g.attrs["source"] = str(Path(src_nexus).resolve()) if src_nexus else ""
        g.attrs["instrument"] = str(args.instrument)
        g.attrs["finder"] = str(args.finder or "")
        g.attrs["weighting"] = weighting_desc
    tmp_final.replace(path)
    print(
        f"Full-data scores under common geometry: {scores}; selected {selected}",
        flush=True,
    )
    return str(path)
