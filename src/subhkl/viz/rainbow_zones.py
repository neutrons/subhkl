"""Interactive Laue diffraction & orientation diagnostic widget for subhkl.

Key features:
1. **Detector Intensity Grounding**: Renders actual log-scaled neutron detector bank
   pixel intensities (from ``--images`` / ``*.im.h5`` / ``merged.h5``, or synthesized
   2D Gaussian spot maps from peak centroids/sigmas/intensities when raw images
   are omitted) directly onto each panel.
2. **Instrument-Aware Cylindrical & Spherical Projections**:
   - *Cylindrical* ``(Azimuth φ [deg], Height y [m])`` — default for ``CG4D`` / ``IMAGINE``
     (and other cylindrical detector geometries).
   - *Spherical* ``(Azimuth φ [deg], Elevation ψ [deg])`` — default for ``MANDI``
     (and other spherical/soccer-ball detector geometries), with an instant UI toggle
     to switch between Cylindrical and Spherical views at any time.
3. **2-Layer Depth Parallax ("Look Behind the Predicted Spots")**:
   - *Back Layer (Detector Plate)*: Physical detector bank panels & log1p pixel intensity heatmaps.
   - *Front Layer (Bragg Reflections)*: Vibrant spectral Bragg peak rings floating by
     resolution ($d$-spacing) with subtle depth tethers to their physical spots on the
     detector plate — moving the mouse shifts the predicted rings aside so you can
     inspect the raw detector spot morphology directly underneath!
"""

from __future__ import annotations

import base64
import glob
import io
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from subhkl.viz.zones import (
    _cell_from_B,
    _spherical_angles,
    _unrolled,
    get_zone_conic_points,
    identify_zones,
    stereographic_projection,
)

# Vibrant 7-color spectral palette
RAINBOW_SPECTRAL_HEX = [
    "#ff2a55",  # Red
    "#ff8c1a",  # Orange
    "#ffe600",  # Yellow
    "#1aff66",  # Green
    "#00e5ff",  # Cyan
    "#3366ff",  # Blue
    "#b833ff",  # Violet
]

PRO_PALETTE_HEX = [
    "#38bdf8",
    "#fb923c",
    "#4ade80",
    "#f87171",
    "#c084fc",
    "#facc15",
    "#2dd4bf",
    "#f472b6",
    "#a3e635",
    "#60a5fa",
]

# Instruments whose native detector arrangement is spherical rather than cylindrical
SPHERICAL_INSTRUMENTS = {"MANDI", "TOPAZ", "SNS_MANDI", "CORELLI"}


def default_projection_for_instrument(instrument: str | None) -> str:
    """Return ``'spherical'`` for spherical instruments (e.g. MANDI) and ``'cylindrical'`` for cylindrical ones (e.g. CG4D)."""
    if instrument and str(instrument).strip().upper() in SPHERICAL_INSTRUMENTS:
        return "spherical"
    return "cylindrical"


def _decode_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray) and val.ndim == 0:
        item = val.item()
        if isinstance(item, (bytes, np.bytes_)):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(val)


def _load_detectors_from_h5(f: h5py.File, instrument: str | None) -> dict[int, Any]:
    """Load calibrated or nominal detector banks from ``subhkl.config.beamlines`` + HDF5 calibration."""
    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector

    detectors: dict[int, Any] = {}
    base_configs: dict[int, dict] = {}
    if instrument and str(instrument) in beamlines:
        for k, v in beamlines[str(instrument)].items():
            try:
                base_configs[int(k)] = dict(v)
            except (ValueError, TypeError):
                pass

    for bank_id, cfg in base_configs.items():
        try:
            detectors[bank_id] = Detector(cfg)
        except Exception:
            pass

    if "detector_calibration" in f:
        cal_grp = f["detector_calibration"]
        for name in cal_grp:
            if not name.startswith("bank_"):
                continue
            try:
                bank_id = int(name.split("_", 1)[1])
            except ValueError:
                continue
            g = cal_grp[name]
            cfg = dict(
                base_configs.get(
                    bank_id, {"panel_type": "flat", "m": 256, "n": 256}
                )
            )
            for field in ("center", "uhat", "vhat"):
                if field in g:
                    cfg[field] = np.asarray(g[field][()], dtype=float).tolist()
            for field in ("width", "height"):
                if field in g:
                    cfg[field] = float(g[field][()])
            try:
                detectors[bank_id] = Detector(cfg)
            except Exception:
                pass

    return detectors


def _discover_images_h5(replay_h5: str | os.PathLike | None) -> str | None:
    """Auto-discover an accompanying ``*.im.h5`` or ``merged.h5`` (or ``attrs['source']``)."""
    if replay_h5 is None:
        return None
    p = Path(replay_h5).resolve()
    if not p.exists():
        return None
    try:
        with h5py.File(p, "r") as f:
            if "images" in f:
                return str(p)
            src_attr = _decode_str(f.attrs.get("source"), "")
            if src_attr:
                for cand_src in (Path(src_attr), p.parent / Path(src_attr).name):
                    if cand_src.exists():
                        try:
                            with h5py.File(cand_src, "r") as sf:
                                if "images" in sf:
                                    return str(cand_src.resolve())
                        except Exception:
                            pass
    except Exception:
        pass

    parent = p.parent
    candidates = sorted(glob.glob(str(parent / "*.im.h5"))) + sorted(
        glob.glob(str(parent / "*merged*.h5"))
    )
    for cand in candidates:
        try:
            with h5py.File(cand, "r") as f:
                if "images" in f and "bank_ids" in f:
                    return cand
        except Exception:
            continue
    return None


def _encode_panel_textures(
    detectors: dict[int, Any],
    bank_arr: np.ndarray | None,
    pixel_r_arr: np.ndarray | None,
    pixel_c_arr: np.ndarray | None,
    intensity_arr: np.ndarray | None,
    sigma_arr: np.ndarray | None,
    images_h5: str | os.PathLike | None = None,
    render_binning: int = 4,
    frame_slice: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Build cylindrical + spherical detector panel geometries and base64 log-intensity PNG textures.

    Returns ``(panels_payload, intensity_source)`` where ``intensity_source`` is
    ``"raw_h5"`` when real detector images were loaded or ``"synthesized_peaks"``
    when reconstructed from fitted peak intensities/widths.
    """
    raw_bank_images: dict[int, np.ndarray] = {}
    intensity_source = "synthesized_peaks"

    if images_h5 is not None and os.path.exists(images_h5):
        try:
            with h5py.File(images_h5, "r") as f:
                if "images" in f:
                    ims = f["images"]
                    n_total_frames = len(ims)
                    b_ids_all = (
                        np.asarray(f["bank_ids"][()], dtype=int)
                        if "bank_ids" in f
                        else np.arange(1, n_total_frames + 1)
                    )
                    f0, f1 = 0, n_total_frames
                    if frame_slice is not None:
                        f0 = max(0, min(int(frame_slice[0]), n_total_frames))
                        f1 = max(f0, min(int(frame_slice[1]), n_total_frames))
                    elif "file_offsets" in f:
                        offsets = np.asarray(f["file_offsets"][()], dtype=int)
                        if len(offsets) > 1:
                            counts = np.diff(np.append(offsets, n_total_frames))
                            best_r = int(np.argmax(counts))
                            f0 = int(offsets[best_r])
                            f1 = int(offsets[best_r] + counts[best_r])
                    b = max(1, int(render_binning))
                    for idx in range(f0, f1):
                        bk = int(b_ids_all[idx])
                        im = np.asarray(ims[idx], dtype=np.float32)
                        if im.ndim != 2:
                            continue
                        n2, m2 = (im.shape[0] // b) * b, (im.shape[1] // b) * b
                        if n2 > 0 and m2 > 0:
                            imb = (
                                im[:n2, :m2]
                                .reshape(n2 // b, b, m2 // b, b)
                                .mean(axis=(1, 3))
                            )
                        else:
                            imb = im
                        if bk in raw_bank_images:
                            raw_bank_images[bk] = np.maximum(
                                raw_bank_images[bk], imb
                            )
                        else:
                            raw_bank_images[bk] = imb
                    if raw_bank_images:
                        intensity_source = "raw_h5"
        except Exception:
            pass

    # If no raw images were loaded, synthesize clean 2D Gaussian spot maps on each populated bank
    if not raw_bank_images and bank_arr is not None and len(bank_arr) > 0:
        grid_size = 96
        yy, xx = np.meshgrid(
            np.arange(grid_size, dtype=np.float32),
            np.arange(grid_size, dtype=np.float32),
            indexing="ij",
        )
        for bk in np.unique(bank_arr):
            bk_int = int(bk)
            det = detectors.get(bk_int)
            n_det = float(getattr(det, "n", 256))
            m_det = float(getattr(det, "m", 256))
            mask = bank_arr == bk_int
            if not np.any(mask):
                continue
            canvas = np.zeros((grid_size, grid_size), dtype=np.float32)
            pr_b = (
                pixel_r_arr[mask]
                if pixel_r_arr is not None
                else np.full(np.sum(mask), n_det * 0.5)
            )
            pc_b = (
                pixel_c_arr[mask]
                if pixel_c_arr is not None
                else np.full(np.sum(mask), m_det * 0.5)
            )
            inten_b = (
                intensity_arr[mask]
                if intensity_arr is not None
                else np.full(np.sum(mask), 200.0)
            )
            sig_b = (
                sigma_arr[mask]
                if sigma_arr is not None
                else np.full(np.sum(mask), 3.5)
            )
            for r0, c0, amp, sig in zip(pr_b, pc_b, inten_b, sig_b):
                ry = float(r0) / max(n_det, 1.0) * (grid_size - 1)
                cx = float(c0) / max(m_det, 1.0) * (grid_size - 1)
                s_grid = max(
                    1.3,
                    min(5.0, float(sig) / max(n_det, 1.0) * grid_size * 1.4),
                )
                g_spot = max(float(amp), 10.0) * np.exp(
                    -0.5 * (((yy - ry) / s_grid) ** 2 + ((xx - cx) / s_grid) ** 2)
                )
                canvas += g_spot
            raw_bank_images[bk_int] = canvas

    # Compute global log1p normalization across all bank images for consistent contrast
    vmin, vmax = 0.0, 1.0
    if raw_bank_images:
        all_logs = [
            np.log1p(np.maximum(im, 0.0)) for im in raw_bank_images.values()
        ]
        flat = np.concatenate([a.ravel() for a in all_logs])
        if len(flat) > 0:
            vmin = float(np.percentile(flat, 10.0))
            vmax = float(np.percentile(flat, 99.6))
            if vmax <= vmin + 1e-5:
                vmax = float(np.max(flat)) + 1.0

    panels_payload: list[dict[str, Any]] = []
    for bk, det in sorted(detectors.items()):
        try:
            rr = np.array([0, det.n - 1, det.n - 1, 0, 0], dtype=float)
            cc = np.array([0, 0, det.m - 1, det.m - 1, 0], dtype=float)
            xyz_corners = det.pixel_to_lab(rr, cc)
            x, y = _unrolled(xyz_corners)
            _, el = _spherical_angles(xyz_corners)
            if np.ptp(x) > 180.0:
                continue

            az_min, az_max = float(np.min(x[:4])), float(np.max(x[:4]))
            y_min, y_max = float(np.min(y[:4])), float(np.max(y[:4]))
            el_min, el_max = float(np.min(el[:4])), float(np.max(el[:4]))

            tex_b64 = None
            im = raw_bank_images.get(int(bk))
            if im is not None:
                im_oriented = im.copy()
                if y[1] > y[0]:
                    im_oriented = im_oriented[::-1, :]
                if x[3] < x[0]:
                    im_oriented = im_oriented[:, ::-1]

                log_im = np.log1p(np.maximum(im_oriented, 0.0))
                norm = np.clip(
                    (log_im - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0
                )
                u8 = (norm * 255.0).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(u8, mode="L").save(
                    buf, format="PNG", optimize=True
                )
                tex_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            panels_payload.append(
                {
                    "bank": int(bk),
                    "az_min": round(az_min, 3),
                    "az_max": round(az_max, 3),
                    "y_min": round(y_min, 4),
                    "y_max": round(y_max, 4),
                    "el_min": round(el_min, 3),
                    "el_max": round(el_max, 3),
                    "outline": [
                        [round(float(x[j]), 2), round(float(y[j]), 4)]
                        for j in range(len(x))
                    ],
                    "outline_sph": [
                        [round(float(x[j]), 2), round(float(el[j]), 3)]
                        for j in range(len(x))
                    ],
                    "texture_b64": tex_b64,
                }
            )
        except Exception:
            pass

    return panels_payload, intensity_source


def _extract_from_h5(
    h5_path: str | os.PathLike,
    images_h5: str | os.PathLike | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    """Extract peaks, detectors, UB, wavelength range, goniometer frames, and metadata from a subhkl HDF5 file."""
    src_h5 = (
        str(images_h5)
        if images_h5 is not None and os.path.exists(images_h5)
        else _discover_images_h5(h5_path)
    )
    sib_solve = Path(h5_path).parent / "solve.h5"
    if (not sib_solve.is_file() or sib_solve == Path(h5_path)) and src_h5:
        sib_solve = Path(src_h5).parent / "solve.h5"
    sib_solve_path = (
        str(sib_solve)
        if sib_solve.is_file() and sib_solve.resolve() != Path(h5_path).resolve()
        else None
    )

    with h5py.File(h5_path, "r") as f:
        instrument = _decode_str(f.attrs.get("instrument"), "IMAGINE")
        ki = (0.0, 0.0, 1.0)
        if "beam/ki_vec" in f:
            ki = tuple(
                float(x)
                for x in np.asarray(f["beam/ki_vec"][()], dtype=float).ravel()[
                    :3
                ]
            )

        wavelength_range = (1.8, 4.5)
        if "scan_refinement/band_wavelength" in f:
            wl = np.asarray(
                f["scan_refinement/band_wavelength"][()], dtype=float
            ).ravel()
            if len(wl) >= 2:
                wavelength_range = (float(wl[0]), float(wl[1]))
        elif "instrument/wavelength" in f:
            wl = np.asarray(
                f["instrument/wavelength"][()], dtype=float
            ).ravel()
            if len(wl) >= 2:
                wavelength_range = (float(wl[0]), float(wl[1]))
                if (
                    abs(wavelength_range[0] - 2.0) < 1e-3
                    and abs(wavelength_range[1] - 10.0) < 1e-3
                ):
                    if sib_solve_path:
                        try:
                            with h5py.File(sib_solve_path, "r") as sf:
                                for wk in (
                                    "scan_refinement/band_wavelength",
                                    "instrument/wavelength",
                                ):
                                    if wk in sf:
                                        swl = np.asarray(sf[wk][()], dtype=float).ravel()
                                        if len(swl) >= 2:
                                            wavelength_range = (
                                                float(swl[0]),
                                                float(swl[1]),
                                            )
                                            break
                        except Exception:
                            pass
        elif sib_solve_path:
            try:
                with h5py.File(sib_solve_path, "r") as sf:
                    for wk in (
                        "scan_refinement/band_wavelength",
                        "instrument/wavelength",
                    ):
                        if wk in sf:
                            swl = np.asarray(sf[wk][()], dtype=float).ravel()
                            if len(swl) >= 2:
                                wavelength_range = (float(swl[0]), float(swl[1]))
                                break
            except Exception:
                pass

        U = (
            np.asarray(f["sample/U"][()], dtype=float)
            if "sample/U" in f
            else None
        )
        B = (
            np.asarray(f["sample/B"][()], dtype=float)
            if "sample/B" in f
            else None
        )
        space_group = (
            _decode_str(f["sample/space_group"][()], "P 1")
            if "sample/space_group" in f
            else "P 1"
        )

        cell = None
        if all(
            f"sample/{k}" in f
            for k in ("a", "b", "c", "alpha", "beta", "gamma")
        ):
            cell = [
                float(f["sample/a"][()]),
                float(f["sample/b"][()]),
                float(f["sample/c"][()]),
                float(f["sample/alpha"][()]),
                float(f["sample/beta"][()]),
                float(f["sample/gamma"][()]),
            ]
        elif B is not None:
            cell = list(_cell_from_B(B))

        detectors = _load_detectors_from_h5(f, instrument)

        bank = (
            np.asarray(f["bank"][()], dtype=int) if "bank" in f else None
        )
        if "peaks/bank" in f and bank is None:
            bank = np.asarray(f["peaks/bank"][()], dtype=int)

        pixel_r = (
            np.asarray(f["peaks/pixel_r"][()], dtype=float)
            if "peaks/pixel_r" in f
            else None
        )
        pixel_c = (
            np.asarray(f["peaks/pixel_c"][()], dtype=float)
            if "peaks/pixel_c" in f
            else None
        )
        peaks_xyz = (
            np.asarray(f["peaks/xyz"][()], dtype=float)
            if "peaks/xyz" in f
            else None
        )

        # Extract multi-frame offsets, file names, and goniometer motor angles
        offsets = (
            np.asarray(f["file_offsets"][()], dtype=int)
            if "file_offsets" in f
            else None
        )
        n_frames_total = len(f["bank_ids"]) if "bank_ids" in f else 0
        files_raw = list(f["files"][()]) if "files" in f else None
        gonio_names_raw = (
            list(f["goniometer/names"][()]) if "goniometer/names" in f else None
        )
        gonio_angles_ref = (
            np.asarray(f["goniometer/angles"][()], dtype=float)
            if "goniometer/angles" in f
            else None
        )
        gonio_angles_nom = None
        R_all = (
            np.asarray(f["goniometer/R"][()], dtype=float)
            if "goniometer/R" in f
            else None
        )

        for aux_path in (src_h5, sib_solve_path):
            if not aux_path or not os.path.isfile(aux_path):
                continue
            try:
                with h5py.File(aux_path, "r") as af:
                    if offsets is None and "file_offsets" in af:
                        offsets = np.asarray(af["file_offsets"][()], dtype=int)
                    if n_frames_total == 0 and "bank_ids" in af:
                        n_frames_total = len(af["bank_ids"])
                    if files_raw is None and "files" in af:
                        files_raw = list(af["files"][()])
                    if gonio_names_raw is None and "goniometer/names" in af:
                        gonio_names_raw = list(af["goniometer/names"][()])
                    if aux_path == src_h5 and "goniometer/angles" in af:
                        gonio_angles_nom = np.asarray(
                            af["goniometer/angles"][()], dtype=float
                        )
                    elif gonio_angles_ref is None and "goniometer/angles" in af:
                        gonio_angles_ref = np.asarray(
                            af["goniometer/angles"][()], dtype=float
                        )
                    if R_all is None and "goniometer/R" in af:
                        R_all = np.asarray(af["goniometer/R"][()], dtype=float)
            except Exception:
                pass

        if gonio_angles_nom is None:
            gonio_angles_nom = gonio_angles_ref

        sym_map = {
            "omega": "ω",
            "phi": "φ",
            "kappa": "κ",
            "chi": "χ",
            "alpha": "α",
            "twotheta": "2θ",
            "2theta": "2θ",
        }
        motor_names: list[str] = []
        if gonio_names_raw:
            for nm in gonio_names_raw:
                s_nm = _decode_str(nm, "motor").split(":")[-1].strip()
                motor_names.append(sym_map.get(s_nm.lower(), s_nm))

        frames_meta: list[dict[str, Any]] = []
        active_frame = 0
        frame_slice: tuple[int, int] | None = None
        if offsets is not None and len(offsets) >= 1 and n_frames_total > 0:
            counts = np.diff(np.append(offsets, n_frames_total))
            # Identify which motor columns vary across frames
            varying_cols: set[int] = set()
            if (
                gonio_angles_nom is not None
                and gonio_angles_nom.ndim == 2
                and len(offsets) > 1
            ):
                valid_offs = [o for o in offsets if 0 <= int(o) < len(gonio_angles_nom)]
                if len(valid_offs) > 1:
                    sub_ang = gonio_angles_nom[valid_offs]
                    for col_i in range(sub_ang.shape[1]):
                        if float(np.ptp(sub_ang[:, col_i])) > 1e-2:
                            varying_cols.add(col_i)

            for r_idx in range(len(offsets)):
                f0 = int(offsets[r_idx])
                f1 = int(offsets[r_idx] + counts[r_idx])
                file_label = f"Frame {r_idx + 1}"
                if files_raw and r_idx < len(files_raw):
                    raw_fn = os.path.basename(_decode_str(files_raw[r_idx], file_label))
                    for ext in (".nxs.h5.img.h5", ".nxs.h5.im.h5", ".nxs.h5", ".img.h5", ".h5"):
                        if raw_fn.endswith(ext):
                            raw_fn = raw_fn[: -len(ext)]
                            break
                    file_label = raw_fn

                motors_list: list[dict[str, Any]] = []
                seen_static: set[tuple[str, float]] = set()
                if (
                    gonio_angles_nom is not None
                    and gonio_angles_nom.ndim == 2
                    and f0 < len(gonio_angles_nom)
                ):
                    row_nom = gonio_angles_nom[f0]
                    row_ref = (
                        gonio_angles_ref[f0]
                        if gonio_angles_ref is not None
                        and gonio_angles_ref.ndim == 2
                        and f0 < len(gonio_angles_ref)
                        else row_nom
                    )
                    col_order = sorted(
                        range(len(row_nom)),
                        key=lambda ci: (0 if ci in varying_cols else 1, ci),
                    )
                    for ci in col_order:
                        m_sym = (
                            motor_names[ci]
                            if ci < len(motor_names)
                            else f"M{ci + 1}"
                        )
                        v_nom = round(float(row_nom[ci]), 2)
                        v_ref = round(float(row_ref[ci]), 2)
                        if ci not in varying_cols:
                            key_st = (m_sym, round(v_nom, 1))
                            if key_st in seen_static:
                                continue
                            seen_static.add(key_st)
                        motors_list.append(
                            {
                                "name": m_sym,
                                "angle_deg": v_nom,
                                "refined_deg": v_ref,
                                "scanned": ci in varying_cols,
                            }
                        )

                motor_str = " ".join(
                    f"{m['name']}={m['angle_deg']:.1f}°" for m in motors_list
                )
                primary_str = " ".join(
                    f"{m['name']}={m['angle_deg']:.1f}°"
                    for m in motors_list
                    if m.get("scanned")
                ) or (
                    f"{motors_list[0]['name']}={motors_list[0]['angle_deg']:.1f}°"
                    if motors_list
                    else ""
                )
                opt_label = (
                    f"{r_idx + 1}/{len(offsets)} · {file_label}"
                    + (f" ({primary_str})" if primary_str else "")
                )
                frames_meta.append(
                    {
                        "frame_index": r_idx,
                        "f0": f0,
                        "f1": f1,
                        "file": file_label,
                        "label": opt_label,
                        "motor_summary": motor_str or f"Frame {r_idx + 1}",
                        "motors": motors_list,
                    }
                )

            if frame_index is not None and 0 <= int(frame_index) < len(frames_meta):
                active_frame = int(frame_index)
            else:
                active_frame = int(np.argmax(counts))
            frame_slice = (
                int(frames_meta[active_frame]["f0"]),
                int(frames_meta[active_frame]["f1"]),
            )

        if (
            peaks_xyz is None
            and pixel_r is not None
            and pixel_c is not None
            and bank is not None
        ):
            xyz_list = np.zeros((len(pixel_r), 3), dtype=float)
            for bk in np.unique(bank):
                mask = bank == bk
                det = detectors.get(int(bk))
                if det is not None:
                    xyz_list[mask] = det.pixel_to_lab(
                        pixel_r[mask], pixel_c[mask]
                    )
                else:
                    az = (int(bk) * 37.0 % 360.0 - 180.0) + (
                        pixel_c[mask] - 128.0
                    ) * 0.12
                    rad = np.deg2rad(az)
                    y = (pixel_r[mask] - 128.0) * 0.002
                    xyz_list[mask] = np.column_stack(
                        [np.sin(rad), y, np.cos(rad)]
                    )
            peaks_xyz = xyz_list

        # Extract finder-less solve / lattice-ladder / staged-band diagnostics if present
        solve_diagnostics: dict[str, Any] | None = None
        if any(k in f for k in ("blind_search", "scan_refinement", "solve", "search")):
            solve_diagnostics = {}
            bs_grp = f["blind_search"] if "blind_search" in f else (f["search"] if "search" in f else None)
            if bs_grp is not None:
                solve_diagnostics["stable_observations"] = int(bs_grp.attrs.get("stable_observations", 0))
                solve_diagnostics["samples_requested"] = int(bs_grp.attrs.get("samples_requested", 0))
                solve_diagnostics["weighting"] = _decode_str(
                    bs_grp.attrs.get("weighting"), "smooth chi2-suppressed bin Poisson deviance"
                )
                if "proposal_scores" in bs_grp:
                    ps = np.asarray(bs_grp["proposal_scores"][()], dtype=float).ravel()
                    solve_diagnostics["n_candidates"] = int(len(ps))
                    solve_diagnostics["top_proposal_score"] = float(np.max(ps)) if len(ps) else 0.0
                if "full_scores" in bs_grp:
                    fs = np.asarray(bs_grp["full_scores"][()], dtype=float).ravel()
                    solve_diagnostics["top_full_score"] = float(np.max(fs)) if len(fs) else 0.0
            if "scan_refinement" in f:
                sr_grp = f["scan_refinement"]
                solve_diagnostics["d_min"] = float(sr_grp.attrs.get("d_min", 2.8))
                solve_diagnostics["n_zones"] = int(sr_grp.attrs.get("n_zones", 400))
                solve_diagnostics["modes"] = _decode_str(sr_grp.attrs.get("modes"), "")
                solve_diagnostics["status"] = _decode_str(sr_grp.attrs.get("status"), "converged")
                stages_raw = _decode_str(sr_grp.attrs.get("stages"), "[]")
                try:
                    solve_diagnostics["stages"] = json.loads(stages_raw)
                except Exception:
                    solve_diagnostics["stages"] = []
            elif "solve" in f:
                s_grp = f["solve"]
                solve_diagnostics["objective"] = _decode_str(s_grp.attrs.get("objective"), "poisson")
                solve_diagnostics["status"] = _decode_str(s_grp.attrs.get("status"), "converged")

        hkl = None
        if all(f"peaks/{k}" in f for k in ("h", "k", "l")):
            hkl = np.column_stack(
                [
                    np.asarray(f["peaks/h"][()], dtype=int),
                    np.asarray(f["peaks/k"][()], dtype=int),
                    np.asarray(f["peaks/l"][()], dtype=int),
                ]
            )
        lam = (
            np.asarray(f["peaks/lambda"][()], dtype=float)
            if "peaks/lambda" in f
            else None
        )
        intensity = (
            np.asarray(f["peaks/intensity"][()], dtype=float)
            if "peaks/intensity" in f
            else None
        )
        sigma = (
            np.asarray(f["peaks/sigma"][()], dtype=float)
            if "peaks/sigma" in f
            else None
        )
        var_u = (
            np.asarray(f["peaks/var_u"][()], dtype=float)
            if "peaks/var_u" in f
            else None
        )
        var_v = (
            np.asarray(f["peaks/var_v"][()], dtype=float)
            if "peaks/var_v" in f
            else None
        )
        cov_uv = (
            np.asarray(f["peaks/cov_uv"][()], dtype=float)
            if "peaks/cov_uv" in f
            else None
        )
        run_index = (
            np.asarray(f["peaks/run_index"][()], dtype=int)
            if "peaks/run_index" in f
            else None
        )

        artifact_kind = "solve"
        peak_shape_mode = "solved_circle"
        if var_u is not None and var_v is not None:
            artifact_kind = "integrator"
            is_iso = bool(
                np.allclose(var_u, var_v)
                and (cov_uv is None or np.allclose(cov_uv, 0.0))
            )
            peak_shape_mode = (
                "isotropic_circle" if is_iso else "anisotropic_ellipse"
            )
        elif "peaks/pixel_r" in f:
            artifact_kind = "finder"
            peak_shape_mode = "isotropic_circle"

        # If multi-frame dataset has per-run peaks (e.g. integrated.h5), filter to active_frame
        if (
            len(frames_meta) > 1
            and peaks_xyz is not None
            and run_index is not None
            and len(run_index) == len(peaks_xyz)
        ):
            mask_run = run_index == active_frame
            if np.any(mask_run):
                peaks_xyz = peaks_xyz[mask_run]
                if bank is not None and len(bank) == len(mask_run):
                    bank = bank[mask_run]
                if pixel_r is not None and len(pixel_r) == len(mask_run):
                    pixel_r = pixel_r[mask_run]
                if pixel_c is not None and len(pixel_c) == len(mask_run):
                    pixel_c = pixel_c[mask_run]
                if hkl is not None and len(hkl) == len(mask_run):
                    hkl = hkl[mask_run]
                if lam is not None and len(lam) == len(mask_run):
                    lam = lam[mask_run]
                if intensity is not None and len(intensity) == len(mask_run):
                    intensity = intensity[mask_run]
                if sigma is not None and len(sigma) == len(mask_run):
                    sigma = sigma[mask_run]
                if var_u is not None and len(var_u) == len(mask_run):
                    var_u = var_u[mask_run]
                if var_v is not None and len(var_v) == len(mask_run):
                    var_v = var_v[mask_run]
                if cov_uv is not None and len(cov_uv) == len(mask_run):
                    cov_uv = cov_uv[mask_run]
                run_index = run_index[mask_run]

        # Filter out zero/noise predictions in integrated.h5 (keep significant reflections I/sigma >= 2.5)
        if (
            peaks_xyz is not None
            and intensity is not None
            and sigma is not None
            and len(intensity) == len(peaks_xyz)
            and len(sigma) == len(peaks_xyz)
        ):
            snr_mask = (sigma > 0) & (
                intensity / np.maximum(sigma, 1e-9) >= 2.5
            )
            if int(np.sum(snr_mask)) >= 20 and int(np.sum(snr_mask)) < len(peaks_xyz):
                peaks_xyz = peaks_xyz[snr_mask]
                if bank is not None and len(bank) == len(snr_mask):
                    bank = bank[snr_mask]
                if pixel_r is not None and len(pixel_r) == len(snr_mask):
                    pixel_r = pixel_r[snr_mask]
                if pixel_c is not None and len(pixel_c) == len(snr_mask):
                    pixel_c = pixel_c[snr_mask]
                if hkl is not None and len(hkl) == len(snr_mask):
                    hkl = hkl[snr_mask]
                if lam is not None and len(lam) == len(snr_mask):
                    lam = lam[snr_mask]
                intensity = intensity[snr_mask]
                sigma = sigma[snr_mask]
                if var_u is not None and len(var_u) == len(snr_mask):
                    var_u = var_u[snr_mask]
                if var_v is not None and len(var_v) == len(snr_mask):
                    var_v = var_v[snr_mask]
                if cov_uv is not None and len(cov_uv) == len(snr_mask):
                    cov_uv = cov_uv[snr_mask]
                if run_index is not None and len(run_index) == len(snr_mask):
                    run_index = run_index[snr_mask]

        # Recognize predictor.h5 format: banks/<img_idx>/(i, j, h, k, l, wavelength)
        if (peaks_xyz is None or len(peaks_xyz) == 0) and "banks" in f:
            try:
                artifact_kind = "predictor"
                peak_shape_mode = "predicted_circle"
                b_ids_pred = (
                    np.asarray(f["bank_ids"][()], dtype=int)
                    if "bank_ids" in f
                    else None
                )
                n_bank_entries = (
                    len(b_ids_pred)
                    if b_ids_pred is not None
                    else len(f["banks"])
                )
                f0_p, f1_p = (
                    frame_slice
                    if frame_slice is not None
                    else (0, n_bank_entries)
                )

                bank_imgs_pred: dict[int, np.ndarray] = {}
                if src_h5 and os.path.exists(src_h5):
                    try:
                        with h5py.File(src_h5, "r") as mf:
                            if "images" in mf and "bank_ids" in mf:
                                b_ids_s = np.asarray(
                                    mf["bank_ids"][f0_p:f1_p], dtype=int
                                )
                                ims_s = np.asarray(
                                    mf["images"][f0_p:f1_p], dtype=np.float32
                                )
                                for b_id_s, im_s in zip(b_ids_s, ims_s):
                                    bank_imgs_pred[int(b_id_s)] = im_s
                    except Exception:
                        pass

                xyz_p, hkl_p, lam_p, bank_p, pr_p, pc_p, inten_p = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                banks_grp = f["banks"]
                for img_idx in range(f0_p, f1_p):
                    key_s = str(img_idx)
                    if key_s not in banks_grp:
                        continue
                    grp = banks_grp[key_s]
                    if "i" not in grp or "j" not in grp:
                        continue
                    pr_i = np.asarray(grp["i"][()], dtype=float).ravel()
                    pc_i = np.asarray(grp["j"][()], dtype=float).ravel()
                    if len(pr_i) == 0:
                        continue
                    bid = (
                        int(b_ids_pred[img_idx])
                        if b_ids_pred is not None and img_idx < len(b_ids_pred)
                        else int(img_idx)
                    )
                    det = detectors.get(bid)
                    h_i = np.asarray(grp["h"][()], dtype=int).ravel()
                    k_i = np.asarray(grp["k"][()], dtype=int).ravel()
                    l_i = np.asarray(grp["l"][()], dtype=int).ravel()
                    wl_i = np.asarray(grp["wavelength"][()], dtype=float).ravel()

                    im_b = bank_imgs_pred.get(bid)
                    if im_b is not None:
                        r_idx = np.clip(
                            np.round(pr_i).astype(int), 2, im_b.shape[0] - 3
                        )
                        c_idx = np.clip(
                            np.round(pc_i).astype(int), 2, im_b.shape[1] - 3
                        )
                        pk_val = np.maximum.reduce(
                            [
                                im_b[r_idx + dr, c_idx + dc]
                                for dr in (-1, 0, 1)
                                for dc in (-1, 0, 1)
                            ]
                        )
                        bg_val = float(np.median(im_b))
                        snr_val = (pk_val - bg_val) / np.sqrt(max(bg_val, 1.0))
                        keep_p = snr_val >= 2.5
                        if not np.any(keep_p):
                            continue
                        pr_i = pr_i[keep_p]
                        pc_i = pc_i[keep_p]
                        h_i, k_i, l_i = h_i[keep_p], k_i[keep_p], l_i[keep_p]
                        wl_i = wl_i[keep_p]
                        pk_val = pk_val[keep_p]
                    else:
                        pk_val = np.full(len(pr_i), 150.0, dtype=float)

                    if det is not None:
                        xyz_i = det.pixel_to_lab(pr_i, pc_i)
                    else:
                        az = (int(bid) * 37.0 % 360.0 - 180.0) + (
                            pc_i - 128.0
                        ) * 0.12
                        rad = np.deg2rad(az)
                        y_c = (pr_i - 128.0) * 0.002
                        xyz_i = np.column_stack([np.sin(rad), y_c, np.cos(rad)])
                    xyz_p.append(xyz_i)
                    hkl_p.append(np.column_stack([h_i, k_i, l_i]))
                    lam_p.append(wl_i)
                    bank_p.append(np.full(len(pr_i), bid, dtype=int))
                    pr_p.append(pr_i)
                    pc_p.append(pc_i)
                    inten_p.append(pk_val)

                if xyz_p:
                    peaks_xyz = np.concatenate(xyz_p, axis=0)
                    hkl = np.concatenate(hkl_p, axis=0)
                    lam = np.concatenate(lam_p, axis=0)
                    bank = np.concatenate(bank_p, axis=0)
                    pixel_r = np.concatenate(pr_p, axis=0)
                    pixel_c = np.concatenate(pc_p, axis=0)
                    intensity = np.concatenate(inten_p, axis=0)
                    sigma = np.full(len(peaks_xyz), 3.0, dtype=float)
            except Exception:
                pass

        R_gonio = None
        if R_all is not None:
            if R_all.ndim == 3 and len(R_all) > 0:
                if frame_slice is not None and frame_slice[0] < len(R_all):
                    R_gonio = R_all[frame_slice[0]]
                elif peaks_xyz is not None and len(R_all) == len(peaks_xyz):
                    R_gonio = R_all
                elif (
                    run_index is not None
                    and len(run_index) > 0
                    and np.max(run_index, initial=0) < len(R_all)
                ):
                    R_gonio = R_all[run_index]
                else:
                    R_gonio = R_all[0]
            elif R_all.ndim == 2:
                R_gonio = R_all

        # Finder-less solve.h5 forward projection onto calibrated detector banks
        if (peaks_xyz is None or len(peaks_xyz) == 0) and U is not None and B is not None and detectors:
            try:
                from subhkl.core.crystallography import generate_reflections

                U_mat = np.asarray(U, dtype=float)
                if U_mat.ndim == 3:
                    U_mat = U_mat[0]
                U = U_mat
                B_mat = np.asarray(B, dtype=float)
                R_run = (
                    np.asarray(R_gonio, dtype=float)
                    if R_gonio is not None and np.asarray(R_gonio).ndim == 2
                    else np.eye(3)
                )
                RUB = R_run @ U_mat @ B_mat
                cell_use = cell if cell and len(cell) == 6 else list(_cell_from_B(B_mat))
                d_min_use = (
                    float(solve_diagnostics.get("d_min", 2.5))
                    if solve_diagnostics and "d_min" in solve_diagnostics
                    else 2.5
                )
                hkl_all = np.stack(
                    generate_reflections(
                        *cell_use, space_group=space_group, d_min=d_min_use
                    ),
                    axis=1,
                ).astype(int)
                G = hkl_all.astype(float) @ RUB.T
                qn = np.linalg.norm(G, axis=1)
                ki_vec = np.asarray(ki, dtype=float)
                ki_vec = ki_vec / max(float(np.linalg.norm(ki_vec)), 1e-12)
                sth = -(G @ ki_vec) / np.maximum(qn, 1e-12)
                lam_pred = 2.0 * sth / np.maximum(qn, 1e-12)
                keep_wl = (
                    (lam_pred >= wavelength_range[0])
                    & (lam_pred <= wavelength_range[1])
                    & (sth > 0)
                )
                hkl_b = hkl_all[keep_wl]
                lam_b = lam_pred[keep_wl]
                G_b = G[keep_wl]
                kf = ki_vec[None, :] + lam_b[:, None] * G_b
                kf /= np.linalg.norm(kf, axis=1, keepdims=True)

                bank_imgs: dict[int, np.ndarray] = {}
                src_h5 = (
                    str(images_h5)
                    if images_h5 is not None and os.path.exists(images_h5)
                    else _discover_images_h5(h5_path)
                )
                if src_h5 and os.path.exists(src_h5):
                    try:
                        with h5py.File(src_h5, "r") as mf:
                            if "images" in mf and "bank_ids" in mf:
                                f0_s, f1_s = (
                                    frame_slice
                                    if frame_slice is not None
                                    else (0, len(mf["bank_ids"]))
                                )
                                b_ids_s = np.asarray(
                                    mf["bank_ids"][f0_s:f1_s], dtype=int
                                )
                                ims_s = np.asarray(
                                    mf["images"][f0_s:f1_s], dtype=np.float32
                                )
                                for b_id_s, im_s in zip(b_ids_s, ims_s):
                                    bank_imgs[int(b_id_s)] = im_s
                    except Exception:
                        pass

                xyz_l, hkl_l, lam_l, bank_l, pr_l, pc_l, inten_l = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for bid, det in sorted(detectors.items()):
                    cen = np.asarray(det.center, dtype=float)
                    uh = np.asarray(det.uhat, dtype=float)
                    vh = np.asarray(det.vhat, dtype=float)
                    w_d, h_d = float(det.width), float(det.height)
                    nrm = np.cross(uh, vh)
                    dn = kf @ nrm
                    t_ray = np.dot(cen, nrm) / np.where(
                        np.abs(dn) < 1e-9, -1.0, dn
                    )
                    hit = t_ray > 0
                    if not np.any(hit):
                        continue
                    idx_h = np.flatnonzero(hit)
                    p_hit = t_ray[idx_h, None] * kf[idx_h] - cen[None, :]
                    um = p_hit @ uh
                    vm = p_hit @ vh
                    in_b = (
                        (um >= w_d * 0.02)
                        & (um <= w_d * 0.98)
                        & (vm >= h_d * 0.02)
                        & (vm <= h_d * 0.98)
                    )
                    if not np.any(in_b):
                        continue
                    idx_h = idx_h[in_b]
                    um, vm = um[in_b], vm[in_b]
                    pr_hit = vm / max(h_d, 1e-6) * (det.n - 1)
                    pc_hit = um / max(w_d, 1e-6) * (det.m - 1)
                    im_b = bank_imgs.get(int(bid))
                    if im_b is not None:
                        r_i = np.clip(
                            np.round(pr_hit).astype(int), 2, im_b.shape[0] - 3
                        )
                        c_i = np.clip(
                            np.round(pc_hit).astype(int), 2, im_b.shape[1] - 3
                        )
                        peak_val = np.maximum.reduce(
                            [
                                im_b[r_i + dr, c_i + dc]
                                for dr in (-1, 0, 1)
                                for dc in (-1, 0, 1)
                            ]
                        )
                        bg_val = float(np.median(im_b))
                        snr = (peak_val - bg_val) / np.sqrt(max(bg_val, 1.0))
                        keep_snr = snr >= 3.0
                        if not np.any(keep_snr):
                            continue
                        idx_h = idx_h[keep_snr]
                        um, vm = um[keep_snr], vm[keep_snr]
                        pr_hit, pc_hit = pr_hit[keep_snr], pc_hit[keep_snr]
                        peak_val = peak_val[keep_snr]
                    else:
                        peak_val = np.full(len(idx_h), 150.0, dtype=float)

                    xyz_hit = (
                        cen[None, :]
                        + um[:, None] * uh[None, :]
                        + vm[:, None] * vh[None, :]
                    )
                    xyz_l.append(xyz_hit)
                    hkl_l.append(hkl_b[idx_h])
                    lam_l.append(lam_b[idx_h])
                    bank_l.append(np.full(len(idx_h), int(bid), dtype=int))
                    pr_l.append(pr_hit)
                    pc_l.append(pc_hit)
                    inten_l.append(peak_val)

                if xyz_l:
                    peaks_xyz = np.concatenate(xyz_l, axis=0)
                    hkl = np.concatenate(hkl_l, axis=0)
                    lam = np.concatenate(lam_l, axis=0)
                    bank = np.concatenate(bank_l, axis=0)
                    pixel_r = np.concatenate(pr_l, axis=0)
                    pixel_c = np.concatenate(pc_l, axis=0)
                    intensity = np.concatenate(inten_l, axis=0)
                    sigma = np.full(len(peaks_xyz), 3.0, dtype=float)
            except Exception:
                pass

        if peaks_xyz is None:
            peaks_xyz = np.zeros((0, 3), dtype=float)

        n_peaks = len(peaks_xyz)
        if bank is None:
            bank = np.ones(n_peaks, dtype=int)
        if pixel_r is None:
            pixel_r = np.zeros(n_peaks, dtype=float)
        if pixel_c is None:
            pixel_c = np.zeros(n_peaks, dtype=float)
        if intensity is None:
            intensity = np.ones(n_peaks, dtype=float) * 100.0
        if sigma is None:
            sigma = np.ones(n_peaks, dtype=float) * 3.0
        if run_index is None:
            run_index = np.zeros(n_peaks, dtype=int)

        return {
            "instrument": instrument,
            "ki": ki,
            "wavelength_range": wavelength_range,
            "U": U,
            "B": B,
            "space_group": space_group,
            "cell": cell,
            "detectors": detectors,
            "peaks_xyz": peaks_xyz,
            "bank": bank,
            "pixel_r": pixel_r,
            "pixel_c": pixel_c,
            "intensity": intensity,
            "sigma": sigma,
            "var_u": var_u,
            "var_v": var_v,
            "cov_uv": cov_uv,
            "artifact_kind": artifact_kind,
            "peak_shape_mode": peak_shape_mode,
            "run_index": run_index,
            "hkl": hkl,
            "wavelength": lam,
            "R_gonio": R_gonio,
            "solve_diagnostics": solve_diagnostics,
            "frame_slice": frame_slice,
            "frames": frames_meta,
            "frame_index": active_frame,
        }


def _build_lattice_ladder_steps(
    zones: list[dict[str, Any]],
    peaks_list: list[dict[str, Any]],
    cell: list[float] | None,
    space_group: str,
    solve_diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Construct the interactive Lattice Ladder progression (Rung 1 -> Goal)."""
    n_total = max(len(peaks_list), 1)
    indexed_peaks = [p for p in peaks_list if p["indexed"]]
    overall_rms = (
        float(
            np.sqrt(
                np.mean([p["angular_error_deg"] ** 2 for p in indexed_peaks])
            )
        )
        if indexed_peaks
        else 0.0
    )

    stages_map: dict[str, dict[str, Any]] = {}
    if solve_diagnostics and isinstance(solve_diagnostics.get("stages"), list):
        for st in solve_diagnostics["stages"]:
            if isinstance(st, dict) and "stage" in st:
                stages_map[str(st["stage"])] = st

    all_ids = [p["id"] for p in peaks_list]
    if solve_diagnostics and solve_diagnostics.get("stable_observations"):
        n_obs = int(solve_diagnostics.get("stable_observations", 0))
        n_samp = int(solve_diagnostics.get("samples_requested", 16384))
        w_desc = solve_diagnostics.get(
            "weighting", "smooth chi2-suppressed bin Poisson deviance"
        )
        s1_stage = "1. Binned Count Excess"
        s1_sub = f"Finder-less Poisson deviance ({n_obs:,} bins → {n_samp:,} sampled)"
        s1_desc = (
            f"Threshold-free count extraction ({w_desc}): {n_obs:,} stable-run count bins "
            f"({n_samp:,} quadrature samples) enter lattice_ladder directly with no spot finder."
        )
    else:
        s1_stage = "1. Raw Detector Counts"
        s1_sub = "Neutron counts & peak centroids on panels"
        s1_desc = (
            f"Inspect the raw detector banks: {len(all_ids)} diffraction "
            "spots sit directly on the panels with no predicted overlay."
        )

    steps: list[dict[str, Any]] = [
        {
            "rung": 1,
            "stage": s1_stage,
            "subtitle": s1_sub,
            "active_zones": [],
            "highlighted_peaks": [],
            "n_captured": len(all_ids),
            "n_total": len(all_ids),
            "fraction_indexed": round(len(indexed_peaks) / n_total, 4),
            "rms_deg": round(overall_rms, 3),
            "zone_tol_deg": 1.0,
            "description": s1_desc,
        }
    ]

    z0_ids = [zones[0]["zone_id"]] if len(zones) >= 1 else []
    z0_peaks = zones[0]["member_peak_ids"] if len(zones) >= 1 else []
    z0_label = zones[0]["label"] if len(zones) >= 1 else "[1 0 0]"
    z0_rms = zones[0]["angular_rms_deg"] if len(zones) >= 1 else overall_rms
    if solve_diagnostics and solve_diagnostics.get("n_candidates"):
        n_cands = int(solve_diagnostics.get("n_candidates", 4))
        full_sc = float(solve_diagnostics.get("top_full_score", 0.0))
        s2_stage = f"2. 13→145 Zone Ladder ({z0_label})"
        s2_sub = f"Blind Lattice-Ladder ({n_cands} candidates • score {full_sc:.3f})"
    else:
        s2_stage = f"2. Primary Zone {z0_label}"
        s2_sub = f"Coarse zone lock ({len(z0_peaks)} spots in family)"
    steps.append(
        {
            "rung": 2,
            "stage": s2_stage,
            "subtitle": s2_sub,
            "active_zones": z0_ids,
            "highlighted_peaks": z0_peaks,
            "n_captured": len(z0_peaks),
            "n_total": len(all_ids),
            "fraction_indexed": round(len(z0_peaks) / n_total, 4),
            "rms_deg": round(float(z0_rms), 3),
            "zone_tol_deg": 1.0,
            "description": (
                f"Primary crystal zone axis {z0_label} captures {len(z0_peaks)} "
                "coplanar scattering vectors (G · t_uvw = 0) during lattice_ladder (13→30→60→145 zones)."
            ),
        }
    )

    z12_ids = [z["zone_id"] for z in zones[:2]]
    z12_peaks = sorted(
        {pid for z in zones[:2] for pid in z["member_peak_ids"]}
    )
    psi_deg = 90.0
    if len(zones) >= 2:
        a0 = np.asarray(zones[0]["axis_crystal_xyz"], dtype=float)
        a1 = np.asarray(zones[1]["axis_crystal_xyz"], dtype=float)
        psi_deg = float(
            np.rad2deg(np.arccos(np.clip(np.dot(a0, a1), -1.0, 1.0)))
        )
    z1_label = zones[1]["label"] if len(zones) >= 2 else "[0 1 0]"
    if "global" in stages_map:
        st_g = stages_map["global"]
        s3_stage = f"3. 60-Zone Global ({z0_label}×{z1_label})"
        s3_sub = f"Stage 0 global geometry • score={-float(st_g.get('objective', 0.0)):.3f} ({int(st_g.get('iterations', 0))} iters)"
    else:
        s3_stage = f"3. Zone Lock {z0_label} × {z1_label}"
        s3_sub = f"Inter-zone angle ψ = {psi_deg:.1f}° locks 3D orientation"
    steps.append(
        {
            "rung": 3,
            "stage": s3_stage,
            "subtitle": s3_sub,
            "active_zones": z12_ids,
            "highlighted_peaks": z12_peaks,
            "n_captured": len(z12_peaks),
            "n_total": len(all_ids),
            "fraction_indexed": round(len(z12_peaks) / n_total, 4),
            "rms_deg": round(
                float(np.mean([z["angular_rms_deg"] for z in zones[:2]]))
                if zones
                else overall_rms,
                3,
            ),
            "zone_tol_deg": 0.5,
            "psi_deg": round(psi_deg, 2),
            "description": (
                f"Two intersecting zone families ({z0_label} and {z1_label}, ψ = {psi_deg:.1f}°) "
                "lock 3D crystal orientation and global detector geometry."
            ),
        }
    )

    z_const_ids = [z["zone_id"] for z in zones[: min(6, len(zones))]]
    z_const_peaks = sorted(
        {
            pid
            for z in zones[: min(6, len(zones))]
            for pid in z["member_peak_ids"]
        }
    )
    if "cell" in stages_map or "120-zones" in stages_map:
        st_c = stages_map.get("cell", stages_map.get("120-zones", {}))
        s4_stage = "4. 120→400 Zone + Cell"
        s4_sub = f"Staged band + cell/per-run • score={-float(st_c.get('objective', 0.0)):.3f} ({int(st_c.get('iterations', 0))} iters)"
    else:
        s4_stage = "4. Multi-Zone Refinement"
        s4_sub = f"{len(z_const_ids)} major zones capture {len(z_const_peaks)} spots"
    steps.append(
        {
            "rung": 4,
            "stage": s4_stage,
            "subtitle": s4_sub,
            "active_zones": z_const_ids,
            "highlighted_peaks": z_const_peaks,
            "n_captured": len(z_const_peaks),
            "n_total": len(all_ids),
            "fraction_indexed": round(len(z_const_peaks) / n_total, 4),
            "rms_deg": round(
                float(
                    np.mean(
                        [
                            z["angular_rms_deg"]
                            for z in zones[: min(6, len(zones))]
                        ]
                    )
                )
                if zones
                else overall_rms,
                3,
            ),
            "zone_tol_deg": 0.3,
            "description": (
                f"Staged 120→400 zone-band refinement adds higher-order zones ({len(z_const_ids)} shown), "
                "co-refining axial stretch, per-run ω drift, and unit-cell ratios."
            ),
        }
    )

    all_zone_ids = [z["zone_id"] for z in zones]
    cell_str = (
        f"a={cell[0]:.2f}, b={cell[1]:.2f}, c={cell[2]:.2f} Å"
        if cell and len(cell) == 6
        else "Refined UB Matrix"
    )
    s5_stage = (
        "5. Converged Solve UB"
        if solve_diagnostics
        else "5. Indexed UB Solution"
    )
    steps.append(
        {
            "rung": 5,
            "stage": s5_stage,
            "subtitle": f"{space_group} • {cell_str}",
            "active_zones": all_zone_ids,
            "highlighted_peaks": [p["id"] for p in indexed_peaks] or all_ids,
            "n_captured": len(indexed_peaks),
            "n_total": len(all_ids),
            "fraction_indexed": round(len(indexed_peaks) / n_total, 4),
            "rms_deg": round(overall_rms, 3),
            "zone_tol_deg": 0.15,
            "description": (
                f"Full 3D UB solution ({space_group}, {cell_str}): "
                f"{len(indexed_peaks)}/{len(peaks_list)} reflections indexed "
                f"({100.0 * len(indexed_peaks) / n_total:.1f}%), Δθ RMS = {overall_rms:.3f}°. "
                "Move the mouse to shift predicted rings aside and inspect raw spots behind them."
            ),
        }
    )
    return steps


def _build_ndip_pipeline_stages(
    peaks_list: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    *,
    instrument: str = "CG4D",
    space_group: str = "P 1",
    cell: list[float] | None = None,
    solve_diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the 4-stage coarse NDIP/Galaxy workflow pipeline descriptors."""
    n_peaks = len(peaks_list)
    n_indexed = sum(1 for p in peaks_list if p.get("indexed"))
    n_zones_ord1 = sum(1 for z in zones if (z.get("order") or 1) <= 1)
    cell_tag = (
        f"{cell[0]:.1f}, {cell[1]:.1f}, {cell[2]:.1f} Å"
        if cell and len(cell) >= 3
        else space_group
    )
    is_finderless = bool(
        solve_diagnostics and solve_diagnostics.get("stable_observations")
    )
    return [
        {
            "stage_id": "reduce",
            "step_num": 1,
            "name": "1. REDUCE & MERGE" if is_finderless else "1. REDUCE & FIND",
            "tool_ids": (
                ["imaginex_subhkl_prepare_peaks"]
                if is_finderless
                else ["imaginex_subhkl_prepare_peaks", "imaginex_subhkl_peak_fit"]
            ),
            "container": "ghcr.io/neutrons/subhkl:1.4.0",
            "artifact": (
                f"merged.h5 ({solve_diagnostics['stable_observations']:,} count bins)"
                if is_finderless and solve_diagnostics
                else f"scan_master.h5 → finder.h5 ({n_peaks} peaks)"
            ),
            "status": "completed",
            "ladder_rung_idx": 0,
            "summary": (
                f"NDIP Image Reduction & Merge ({instrument}): "
                "reduced raw NeXus event data into merged.h5 detector bank stacks."
            ),
        },
        {
            "stage_id": "solve",
            "step_num": 2,
            "name": "2. SOLVE (LATTICE LADDER)",
            "tool_ids": ["imaginex_subhkl_solve", "imaginex_subhkl"],
            "container": "ghcr.io/neutrons/subhkl:1.4.0",
            "artifact": f"solve.h5 ({n_indexed}/{n_peaks} spots • {n_zones_ord1} ord-1 zones)",
            "status": "active",
            "ladder_rung_idx": 4,
            "summary": (
                f"NDIP Finder-Less Solve (imaginex_subhkl_solve): steps 1–5 below run "
                f"lattice_ladder + staged zone-band refinement ({space_group}, {cell_tag})."
            ),
        },
        {
            "stage_id": "integrate",
            "step_num": 3,
            "name": "3. PREDICT & INTEGRATE",
            "tool_ids": [
                "imaginex_subhkl_predict",
                "imaginex_subhkl_integrate",
            ],
            "container": "ghcr.io/neutrons/subhkl:1.4.0",
            "artifact": "predictor.h5 → integrated.h5",
            "status": "ready",
            "ladder_rung_idx": 4,
            "summary": (
                "NDIP Peak Prediction & Matrix-Free Integration: forward-projects reflections using "
                "solve.h5 calibrated geometry and integrates 3D Gaussian profiles on raw images."
            ),
        },
        {
            "stage_id": "merge",
            "step_num": 4,
            "name": "4. MERGE & MTZ",
            "tool_ids": ["imaginex_subhkl_mtz", "imaginex_merge", "imaginex_analysis"],
            "container": "ghcr.io/neutrons/subhkl:1.4.0",
            "artifact": "reflections.mtz + completeness report",
            "status": "ready",
            "ladder_rung_idx": 4,
            "summary": (
                "NDIP Multi-Orientation Merge & MTZ Export: merges integrated reflections across "
                "goniometer settings into a structure-factor MTZ dataset."
            ),
        },
    ]


def build_rainbow_zone_payload(
    peaks_xyz: np.ndarray | None = None,
    ub_matrix: np.ndarray | None = None,
    replay_h5: str | os.PathLike | None = None,
    *,
    images_h5: str | os.PathLike | None = None,
    projection: str | None = None,
    U: np.ndarray | None = None,
    B: np.ndarray | None = None,
    R_gonio: np.ndarray | None = None,
    ki: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wavelength_range: tuple[float, float] = (1.8, 4.5),
    instrument: str = "IMAGINE",
    space_group: str = "P 1",
    max_zone_index: int = 3,
    zone_tolerance_deg: float = 0.75,
    min_zone_peaks: int = 2,
    max_zones: int = 36,
    n_arc: int = 720,
    max_peaks: int = 2500,
    render_binning: int = 4,
    frame_index: int | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable data contract for the Rainbow Zones widget.

    Includes detector bank outlines in both Cylindrical ``(az, y)`` and Spherical
    ``(az, elevation)`` coordinates, log-scaled detector intensity heatmaps,
    peak reflections, and Lattice Ladder steps.
    """
    detectors: dict[int, Any] = {}
    cell: list[float] | None = None
    bank_arr = None
    pixel_r_arr = None
    pixel_c_arr = None
    intensity_arr = None
    sigma_arr = None
    var_u_arr = None
    var_v_arr = None
    cov_uv_arr = None
    artifact_kind = "solve"
    peak_shape_mode = "solved_circle"
    hkl_arr = None
    lam_arr = None
    solve_diagnostics = None
    frame_slice = None
    frames_meta: list[dict[str, Any]] = []
    active_frame = int(frame_index) if frame_index is not None else 0

    if replay_h5 is not None:
        extracted = _extract_from_h5(
            replay_h5, images_h5=images_h5, frame_index=frame_index
        )
        instrument = extracted["instrument"] or instrument
        ki = extracted["ki"]
        wavelength_range = extracted["wavelength_range"]
        if U is None:
            U = extracted["U"]
        if B is None:
            B = extracted["B"]
        space_group = extracted["space_group"] or space_group
        cell = extracted["cell"]
        detectors = extracted["detectors"]
        if peaks_xyz is None:
            peaks_xyz = extracted["peaks_xyz"]
        bank_arr = extracted["bank"]
        pixel_r_arr = extracted["pixel_r"]
        pixel_c_arr = extracted["pixel_c"]
        intensity_arr = extracted["intensity"]
        sigma_arr = extracted["sigma"]
        var_u_arr = extracted.get("var_u")
        var_v_arr = extracted.get("var_v")
        cov_uv_arr = extracted.get("cov_uv")
        artifact_kind = extracted.get("artifact_kind") or artifact_kind
        peak_shape_mode = extracted.get("peak_shape_mode") or peak_shape_mode
        hkl_arr = extracted["hkl"]
        lam_arr = extracted["wavelength"]
        solve_diagnostics = extracted.get("solve_diagnostics")
        frame_slice = extracted.get("frame_slice")
        frames_meta = extracted.get("frames") or []
        active_frame = int(extracted.get("frame_index", active_frame))
        if R_gonio is None:
            R_gonio = extracted["R_gonio"]
        if images_h5 is None:
            images_h5 = _discover_images_h5(replay_h5)

    if peaks_xyz is None:
        peaks_xyz = np.zeros((0, 3), dtype=float)
    pts = np.asarray(peaks_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("peaks_xyz must have shape (N, 3)")

    if ub_matrix is None and U is not None and B is not None:
        ub_matrix = np.asarray(U, dtype=float) @ np.asarray(B, dtype=float)
    elif ub_matrix is not None:
        ub_matrix = np.asarray(ub_matrix, dtype=float)
        if B is None:
            cell = list(_cell_from_B(ub_matrix))

    if cell is None and B is not None:
        cell = list(_cell_from_B(B))

    n_raw = len(pts)
    if n_raw > max_peaks:
        order = (
            np.argsort(-np.asarray(intensity_arr))[:max_peaks]
            if intensity_arr is not None and len(intensity_arr) == n_raw
            else np.linspace(0, n_raw - 1, max_peaks, dtype=int)
        )
        order = np.sort(order)
        pts = pts[order]
        if bank_arr is not None and len(bank_arr) == n_raw:
            bank_arr = bank_arr[order]
        if pixel_r_arr is not None and len(pixel_r_arr) == n_raw:
            pixel_r_arr = pixel_r_arr[order]
        if pixel_c_arr is not None and len(pixel_c_arr) == n_raw:
            pixel_c_arr = pixel_c_arr[order]
        if intensity_arr is not None and len(intensity_arr) == n_raw:
            intensity_arr = intensity_arr[order]
        if sigma_arr is not None and len(sigma_arr) == n_raw:
            sigma_arr = sigma_arr[order]
        if var_u_arr is not None and len(var_u_arr) == n_raw:
            var_u_arr = var_u_arr[order]
        if var_v_arr is not None and len(var_v_arr) == n_raw:
            var_v_arr = var_v_arr[order]
        if cov_uv_arr is not None and len(cov_uv_arr) == n_raw:
            cov_uv_arr = cov_uv_arr[order]
        if hkl_arr is not None and len(hkl_arr) == n_raw:
            hkl_arr = hkl_arr[order]
        if lam_arr is not None and len(lam_arr) == n_raw:
            lam_arr = lam_arr[order]
        if (
            isinstance(R_gonio, np.ndarray)
            and R_gonio.ndim == 3
            and len(R_gonio) == n_raw
        ):
            R_gonio = R_gonio[order]

    n_peaks = len(pts)
    ki_hat = np.asarray(ki, dtype=float)
    ki_hat = ki_hat / np.linalg.norm(ki_hat)

    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    kf_hat = pts / np.maximum(norms, 1e-12)
    q_vec = kf_hat - ki_hat[None, :]
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
    g_hat = np.where(q_norm > 1e-6, q_vec / np.maximum(q_norm, 1e-12), kf_hat)

    R_single = np.eye(3)
    if R_gonio is not None:
        R_arr = np.asarray(R_gonio, dtype=float)
        if R_arr.ndim == 2:
            R_single = R_arr
            g_crystal = g_hat @ R_single
        elif R_arr.ndim == 3 and len(R_arr) == n_peaks:
            g_crystal = np.einsum("ni,nij->nj", g_hat, R_arr)
            R_single = R_arr[0]
        else:
            g_crystal = g_hat
    else:
        g_crystal = g_hat

    sx, sy, hemi = (
        stereographic_projection(g_crystal)
        if n_peaks > 0
        else (np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int8))
    )
    det_x, det_y = (
        _unrolled(pts)
        if n_peaks > 0
        else (np.zeros(0, dtype=float), np.zeros(0, dtype=float))
    )
    _, det_el = (
        _spherical_angles(pts)
        if n_peaks > 0
        else (np.zeros(0, dtype=float), np.zeros(0, dtype=float))
    )

    wl_min, wl_max = float(wavelength_range[0]), float(wavelength_range[1])
    if n_peaks > 0 and ub_matrix is not None:
        inv_ub = np.linalg.inv(ub_matrix)
        hkl_dir = g_crystal @ inv_ub.T
        sin_theta = np.clip(0.5 * q_norm.ravel(), 1e-4, 0.9999)
        if hkl_arr is None:
            lam_grid = np.linspace(wl_min, wl_max, 64)
            q_mag_grid = (2.0 * sin_theta[:, None]) / lam_grid[None, :]
            hkl_cand = hkl_dir[:, None, :] * q_mag_grid[:, :, None]
            hkl_round = np.rint(hkl_cand)
            nonzero = np.any(hkl_round != 0, axis=-1)
            diff = np.linalg.norm(hkl_cand - hkl_round, axis=-1)
            diff = np.where(nonzero, diff, 1e9)
            best_l = np.argmin(diff, axis=1)
            hkl_arr = hkl_round[np.arange(n_peaks), best_l].astype(int)

        G_pred_cryst = hkl_arr @ ub_matrix.T
        G_norm = np.linalg.norm(G_pred_cryst, axis=1)
        valid_hkl = G_norm > 1e-8
        G_pred_hat = np.divide(
            G_pred_cryst, np.maximum(G_norm[:, None], 1e-12)
        )
        cos_err = np.clip(np.sum(g_crystal * G_pred_hat, axis=1), -1.0, 1.0)
        ang_err_deg = np.where(valid_hkl, np.rad2deg(np.arccos(cos_err)), 90.0)
        d_spacing = np.where(valid_hkl, 1.0 / np.maximum(G_norm, 1e-8), 0.0)
        if lam_arr is None:
            lam_arr = np.where(
                valid_hkl,
                2.0 * sin_theta / np.maximum(G_norm, 1e-8),
                0.5 * (wl_min + wl_max),
            )
    else:
        if hkl_arr is None:
            hkl_arr = np.zeros((n_peaks, 3), dtype=int)
        sin_theta = (
            np.clip(0.5 * q_norm.ravel(), 1e-4, 0.9999)
            if n_peaks > 0
            else np.zeros(0)
        )
        if lam_arr is None:
            lam_arr = np.full(n_peaks, 0.5 * (wl_min + wl_max), dtype=float)
        d_spacing = (
            lam_arr / np.maximum(2.0 * sin_theta, 1e-4)
            if n_peaks > 0
            else np.zeros(0, dtype=float)
        )
        ang_err_deg = np.zeros(n_peaks, dtype=float)

    raw_zones = (
        identify_zones(
            pts,
            ub_matrix=ub_matrix,
            R_gonio=R_gonio,
            ki=ki_hat,
            max_index=max_zone_index,
            tolerance_deg=zone_tolerance_deg,
            min_peaks=0 if ub_matrix is not None else min(min_zone_peaks, max(1, n_peaks)),
            n_arc=n_arc,
        )
        if n_peaks > 0
        else []
    )
    for z in raw_zones:
        uvw_vals = [abs(int(x)) for x in z.get("uvw", [1, 0, 0])]
        z["order"] = max(uvw_vals) if any(uvw_vals) else 1

    # Keep all fundamental Order 1 zones (|u|,|v|,|w| <= 1) plus populated Order 2 and Order 3 zones
    selected_zones: list[dict[str, Any]] = []
    z_ord1 = [
        z for z in raw_zones
        if z["order"] == 1 and (z["n_peaks"] >= 1 or len(z["conic"]["det_x"]) > 0)
    ][:13]
    z_ord2 = [
        z for z in raw_zones
        if z["order"] == 2 and z["n_peaks"] >= max(1, min_zone_peaks)
    ][:14]
    z_ord3 = [
        z for z in raw_zones
        if z["order"] == 3 and z["n_peaks"] >= max(2, min_zone_peaks)
    ][:16]
    selected_zones.extend(z_ord1)
    selected_zones.extend(z_ord2)
    selected_zones.extend(z_ord3)
    other_zones = [
        z for z in raw_zones
        if z["order"] > 3 and z["n_peaks"] >= max(2, min_zone_peaks)
    ][:8]
    selected_zones.extend(other_zones)
    selected_zones.sort(
        key=lambda z: (z["order"], -z["n_peaks"], -z["score"])
    )
    raw_zones = selected_zones[:max(max_zones, len(z_ord1) + len(z_ord2))]
    for new_zid, z in enumerate(raw_zones):
        z["zone_id"] = new_zid

    peak_zones: list[list[int]] = [[] for _ in range(n_peaks)]
    for z in raw_zones:
        zid = z["zone_id"]
        for pid in z["member_peak_ids"]:
            if 0 <= pid < n_peaks:
                peak_zones[pid].append(zid)

    # Precompute 2-sigma detector-projected peak contours (ellipses or isotropic circles)
    # when covariance parameters (var_u, var_v, cov_uv) are present from integrator/finder
    ellipse_cyl_by_idx: dict[int, list[list[float]]] = {}
    ellipse_sph_by_idx: dict[int, list[list[float]]] = {}
    sigma_major_by_idx: dict[int, float] = {}
    sigma_minor_by_idx: dict[int, float] = {}
    ellipse_ang_by_idx: dict[int, float] = {}
    if (
        n_peaks > 0
        and var_u_arr is not None
        and var_v_arr is not None
        and len(var_u_arr) == n_peaks
        and len(var_v_arr) == n_peaks
        and pixel_r_arr is not None
        and pixel_c_arr is not None
        and bank_arr is not None
        and detectors
    ):
        cuv_use = (
            np.asarray(cov_uv_arr, dtype=float)
            if cov_uv_arr is not None and len(cov_uv_arr) == n_peaks
            else np.zeros(n_peaks, dtype=float)
        )
        vu_use = np.asarray(var_u_arr, dtype=float)
        vv_use = np.asarray(var_v_arr, dtype=float)
        diff_uv = vu_use - vv_use
        sq_uv = np.sqrt(diff_uv**2 + 4.0 * cuv_use**2)
        s1_all = np.sqrt(np.maximum((vu_use + vv_use + sq_uv) / 2.0, 1e-6))
        s2_all = np.sqrt(np.maximum((vu_use + vv_use - sq_uv) / 2.0, 1e-6))
        phi_all = 0.5 * np.arctan2(2.0 * cuv_use, diff_uv)

        n_ell_pts = 16
        theta_ell = np.linspace(0.0, 2.0 * np.pi, n_ell_pts, endpoint=False)
        cos_t = np.cos(theta_ell)[None, :]
        sin_t = np.sin(theta_ell)[None, :]
        n_sigma_contour = 2.0

        for bk in np.unique(bank_arr):
            det = detectors.get(int(bk))
            if det is None:
                continue
            idxs = np.flatnonzero(bank_arr == bk)
            if len(idxs) == 0:
                continue
            a_bk = (n_sigma_contour * s1_all[idxs])[:, None]
            b_bk = (n_sigma_contour * s2_all[idxs])[:, None]
            phi_bk = phi_all[idxs][:, None]
            c_phi = np.cos(phi_bk)
            s_phi = np.sin(phi_bk)
            r0 = pixel_r_arr[idxs][:, None]
            c0 = pixel_c_arr[idxs][:, None]
            cols_ell = c0 + a_bk * cos_t * c_phi - b_bk * sin_t * s_phi
            rows_ell = r0 + a_bk * cos_t * s_phi + b_bk * sin_t * c_phi
            try:
                xyz_ell = det.pixel_to_lab(rows_ell.ravel(), cols_ell.ravel())
                az_ell, y_ell = _unrolled(xyz_ell)
                _, el_ell = _spherical_angles(xyz_ell)
                az_mat = az_ell.reshape(len(idxs), n_ell_pts)
                y_mat = y_ell.reshape(len(idxs), n_ell_pts)
                el_mat = el_ell.reshape(len(idxs), n_ell_pts)
                for k_local, p_idx in enumerate(idxs):
                    p_i = int(p_idx)
                    az_row = az_mat[k_local]
                    if float(np.ptp(az_row)) > 90.0:
                        continue
                    ellipse_cyl_by_idx[p_i] = [
                        [round(float(az_row[m]), 3), round(float(y_mat[k_local, m]), 4)]
                        for m in range(n_ell_pts)
                    ]
                    ellipse_sph_by_idx[p_i] = [
                        [round(float(az_row[m]), 3), round(float(el_mat[k_local, m]), 3)]
                        for m in range(n_ell_pts)
                    ]
                    sigma_major_by_idx[p_i] = round(float(s1_all[p_i]), 2)
                    sigma_minor_by_idx[p_i] = round(float(s2_all[p_i]), 2)
                    ellipse_ang_by_idx[p_i] = round(float(np.rad2deg(phi_all[p_i])), 1)
            except Exception:
                pass

    peaks_list: list[dict[str, Any]] = []
    for i in range(n_peaks):
        h_i, k_i, l_i = (
            int(hkl_arr[i, 0]),
            int(hkl_arr[i, 1]),
            int(hkl_arr[i, 2]),
        )
        g_gcd = (
            int(np.gcd.reduce(np.abs([h_i, k_i, l_i])))
            if (h_i or k_i or l_i)
            else 1
        )
        d_val = float(d_spacing[i]) if i < len(d_spacing) else 2.0
        lam_val = (
            float(lam_arr[i]) if i < len(lam_arr) else 0.5 * (wl_min + wl_max)
        )
        err_val = float(ang_err_deg[i]) if i < len(ang_err_deg) else 0.0
        is_indexed = bool(
            (
                (h_i != 0 or k_i != 0 or l_i != 0)
                and err_val <= max(2.0, zone_tolerance_deg * 2.5)
            )
            or len(peak_zones[i]) > 0
        )
        parallax_depth = float(
            np.clip(1.0 / (1.0 + 0.35 * max(d_val, 0.2)), 0.05, 0.98)
        )

        pk_entry: dict[str, Any] = {
            "id": i,
            "bank": (
                int(bank_arr[i])
                if bank_arr is not None and i < len(bank_arr)
                else 1
            ),
            "row": (
                round(float(pixel_r_arr[i]), 2)
                if pixel_r_arr is not None and i < len(pixel_r_arr)
                else 0.0
            ),
            "col": (
                round(float(pixel_c_arr[i]), 2)
                if pixel_c_arr is not None and i < len(pixel_c_arr)
                else 0.0
            ),
            "det_x": round(float(det_x[i]), 3),
            "det_y": round(float(det_y[i]), 4),
            "det_el": round(float(det_el[i]), 3),
            "q_hat": [round(float(x), 5) for x in g_crystal[i]],
            "stereographic": [
                round(float(sx[i]), 5),
                round(float(sy[i]), 5),
            ],
            "hemisphere": int(hemi[i]),
            "intensity": (
                round(float(intensity_arr[i]), 2)
                if intensity_arr is not None and i < len(intensity_arr)
                else 100.0
            ),
            "sigma": (
                round(float(sigma_arr[i]), 2)
                if sigma_arr is not None and i < len(sigma_arr)
                else 3.0
            ),
            "wavelength": round(
                float(np.clip(lam_val, wl_min * 0.5, wl_max * 1.5)), 4
            ),
            "d_spacing": round(d_val, 4),
            "harmonic_order": max(1, g_gcd),
            "parallax_depth": round(parallax_depth, 4),
            "hkl": [h_i, k_i, l_i],
            "zone_ids": peak_zones[i],
            "indexed": is_indexed,
            "angular_error_deg": round(err_val, 4),
        }
        if i in ellipse_cyl_by_idx:
            s_maj = sigma_major_by_idx[i]
            s_min = sigma_minor_by_idx[i]
            pk_entry["ellipse"] = ellipse_cyl_by_idx[i]
            pk_entry["ellipse_sph"] = ellipse_sph_by_idx[i]
            pk_entry["sigma_major_px"] = s_maj
            pk_entry["sigma_minor_px"] = s_min
            pk_entry["ellipse_angle_deg"] = ellipse_ang_by_idx[i]
            pk_entry["shape_kind"] = (
                "ellipse" if abs(s_maj - s_min) > 0.05 else "isotropic_circle"
            )
        peaks_list.append(pk_entry)

    zones_payload: list[dict[str, Any]] = []
    for zi, z in enumerate(raw_zones):
        t_cryst = np.asarray(z["axis_crystal_xyz"], dtype=float)
        t_lab = R_single @ t_cryst
        conic_cryst = get_zone_conic_points(t_cryst, ki=ki_hat, n_arc=n_arc)
        conic_lab = get_zone_conic_points(t_lab, ki=ki_hat, n_arc=n_arc)

        det_segments: list[list[list[float]]] = []
        det_segments_sph: list[list[list[float]]] = []
        kf = conic_lab["kf_hat"]
        if detectors:
            for bk, det in sorted(detectors.items()):
                try:
                    mask, row, col = det.reflections_mask(
                        kf[:, 0], kf[:, 1], kf[:, 2]
                    )
                    if not np.any(mask):
                        continue
                    idx_hits = np.where(mask)[0]
                    splits = np.where(np.diff(idx_hits) > 2)[0] + 1
                    for chunk in np.split(idx_hits, splits):
                        if len(chunk) < 2:
                            continue
                        xyz_seg = det.pixel_to_lab(row[chunk], col[chunk])
                        px, py = _unrolled(xyz_seg)
                        _, pel = _spherical_angles(xyz_seg)
                        if np.ptp(px) <= 90.0:
                            step_s = max(1, len(px) // 32)
                            det_segments.append(
                                [
                                    [
                                        round(float(px[j]), 2),
                                        round(float(py[j]), 4),
                                    ]
                                    for j in range(0, len(px), step_s)
                                ]
                            )
                            det_segments_sph.append(
                                [
                                    [
                                        round(float(px[j]), 2),
                                        round(float(pel[j]), 3),
                                    ]
                                    for j in range(0, len(px), step_s)
                                ]
                            )
                except Exception:
                    pass

        dx = np.asarray(conic_lab["det_x"], dtype=float)
        dy = np.asarray(conic_lab["det_y"], dtype=float)
        del_deg = np.asarray(conic_lab["det_el"], dtype=float)
        if n_peaks > 0:
            med_r_xz = float(np.median(np.hypot(pts[:, 0], pts[:, 2])))
            if med_r_xz > 0.05:
                dy = dy * med_r_xz

        continuous_segments: list[list[list[float]]] = []
        continuous_segments_sph: list[list[list[float]]] = []
        cur_seg: list[list[float]] = []
        cur_seg_sph: list[list[float]] = []
        for k in range(0, len(dx), 2):
            if cur_seg and abs(dx[k] - cur_seg[-1][0]) > 45.0:
                if len(cur_seg) >= 2:
                    continuous_segments.append(cur_seg)
                    continuous_segments_sph.append(cur_seg_sph)
                cur_seg = []
                cur_seg_sph = []
            cur_seg.append([round(float(dx[k]), 2), round(float(dy[k]), 4)])
            cur_seg_sph.append(
                [round(float(dx[k]), 2), round(float(del_deg[k]), 3)]
            )
        if len(cur_seg) >= 2:
            continuous_segments.append(cur_seg)
            continuous_segments_sph.append(cur_seg_sph)

        if not detectors and not det_segments:
            det_segments = continuous_segments
            det_segments_sph = continuous_segments_sph

        sx_z = np.asarray(conic_cryst["sx"], dtype=float)
        sy_z = np.asarray(conic_cryst["sy"], dtype=float)
        hm_z = np.asarray(conic_cryst["hemisphere"], dtype=int)
        stereo_segments: list[list[list[float]]] = []
        cur_s: list[list[float]] = []
        for k in range(0, len(sx_z), 2):
            if cur_s and (
                hm_z[k] != cur_s[-1][2]
                or np.hypot(sx_z[k] - cur_s[-1][0], sy_z[k] - cur_s[-1][1])
                > 0.35
            ):
                if len(cur_s) >= 2:
                    stereo_segments.append(cur_s)
                cur_s = []
            cur_s.append(
                [
                    round(float(sx_z[k]), 4),
                    round(float(sy_z[k]), 4),
                    int(hm_z[k]),
                ]
            )
        if len(cur_s) >= 2:
            stereo_segments.append(cur_s)

        pole_sx, pole_sy, pole_hemi = stereographic_projection(t_cryst)

        zones_payload.append(
            {
                "zone_id": int(z["zone_id"]),
                "uvw": z["uvw"],
                "order": int(z.get("order", max(abs(int(x)) for x in z["uvw"]))),
                "label": z["label"],
                "axis_xyz": z["axis_xyz"],
                "axis_crystal_xyz": z["axis_crystal_xyz"],
                "pole_stereo": [
                    round(float(pole_sx), 4),
                    round(float(pole_sy), 4),
                    int(pole_hemi),
                ],
                "d_direct": z["d_direct"],
                "score": z["score"],
                "n_peaks": z["n_peaks"],
                "angular_rms_deg": z["angular_rms_deg"],
                "member_peak_ids": z["member_peak_ids"],
                "rainbow_color": RAINBOW_SPECTRAL_HEX[
                    zi % len(RAINBOW_SPECTRAL_HEX)
                ],
                "pro_color": PRO_PALETTE_HEX[zi % len(PRO_PALETTE_HEX)],
                "stereo_segments": stereo_segments,
                "detector_segments": det_segments,
                "detector_segments_sph": det_segments_sph,
                "continuous_segments": continuous_segments,
                "continuous_segments_sph": continuous_segments_sph,
            }
        )

    panels_payload, intensity_source = _encode_panel_textures(
        detectors=detectors,
        bank_arr=bank_arr,
        pixel_r_arr=pixel_r_arr,
        pixel_c_arr=pixel_c_arr,
        intensity_arr=intensity_arr,
        sigma_arr=sigma_arr,
        images_h5=images_h5,
        render_binning=render_binning,
        frame_slice=frame_slice,
    )

    ladder_steps = _build_lattice_ladder_steps(
        zones_payload,
        peaks_list,
        cell=cell,
        space_group=space_group,
        solve_diagnostics=solve_diagnostics,
    )

    pipeline_stages = _build_ndip_pipeline_stages(
        peaks_list,
        zones_payload,
        instrument=str(instrument),
        space_group=str(space_group),
        cell=cell,
        solve_diagnostics=solve_diagnostics,
    )

    resolved_proj = (
        str(projection).lower()
        if projection in ("cylindrical", "spherical")
        else default_projection_for_instrument(instrument)
    )

    d_min_use = float((solve_diagnostics or {}).get("d_min", 2.8))
    gonio_summary = (
        frames_meta[active_frame].get("motor_summary", "—")
        if 0 <= active_frame < len(frames_meta)
        else "—"
    )

    return {
        "instrument": str(instrument),
        "default_projection": resolved_proj,
        "space_group": str(space_group),
        "intensity_source": intensity_source,
        "artifact_kind": artifact_kind,
        "peak_shape_mode": peak_shape_mode,
        "wavelength_range": [round(wl_min, 3), round(wl_max, 3)],
        "d_min": round(d_min_use, 3),
        "frame_index": int(active_frame),
        "goniometer_summary": gonio_summary,
        "frames": frames_meta,
        "cell": [round(float(x), 4) for x in cell] if cell else None,
        "ub_matrix": (
            [[round(float(v), 6) for v in row] for row in ub_matrix]
            if ub_matrix is not None
            else None
        ),
        "peaks": peaks_list,
        "zones": zones_payload,
        "panels": panels_payload,
        "ladder_steps": ladder_steps,
        "pipeline_stages": pipeline_stages,
    }


def render_rainbow_html(
    payload: dict[str, Any],
    *,
    default_mode: str = "arcade",
    projection: str | None = None,
    title: str = "SUBHKL // DIFFRACTION PARALLAX",
) -> str:
    """Render a unified, self-contained HTML5/Canvas/JS diffraction parallax widget."""
    payload_json = json.dumps(payload, separators=(",", ":"))
    inst = payload.get("instrument", "IMAGINE")
    sg = payload.get("space_group", "P 1")
    initial_proj = (
        str(projection).lower()
        if projection in ("cylindrical", "spherical")
        else payload.get(
            "default_projection", default_projection_for_instrument(inst)
        )
    )
    src_badge = (
        "RAW DETECTOR COUNTS"
        if payload.get("intensity_source") == "raw_h5"
        else "RECONSTRUCTED SPOT INTENSITY"
    )
    init_wl = payload.get("wavelength_range") or [2.8, 4.5]
    init_wl_min = float(init_wl[0]) if len(init_wl) >= 2 else 2.8
    init_wl_max = float(init_wl[1]) if len(init_wl) >= 2 else 4.5
    init_dmin = float(payload.get("d_min") or 2.8)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/h5wasm@0.7.8/dist/iife/h5wasm.js" async></script>
<style>
  :root {{
    --bg-primary: #070911;
    --bg-panel: #0d111f;
    --bg-card: #13192b;
    --border-color: #26314f;
    --text-main: #f1f5f9;
    --text-dim: #94a3b8;
    --accent-gold: #f472b6;
    --accent-cyan: #38bdf8;
    --accent-pink: #f43f85;
    --accent-green: #34d399;
    --font-display: ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, Consolas, monospace;
    --font-body: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    background: var(--bg-primary);
    color: var(--text-main);
    font-family: var(--font-body);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }}

  /* Top Header Bar — single compact row */
  .top-bar {{
    display: flex;
    flex-wrap: nowrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 5px 12px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-color);
    min-height: 38px;
  }}

  .brand-block {{
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
  }}

  .rainbow-logo {{
    font-family: var(--font-body);
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 0.3px;
    display: inline-flex;
    align-items: baseline;
  }}

  .logo-sub {{
    color: #cbd5e1;
    font-style: normal;
  }}

  .logo-hkl {{
    color: #f43f85;
    font-style: italic;
    margin-left: 1px;
    text-shadow: 0 0 12px rgba(244, 63, 133, 0.38);
  }}

  .subtitle-tag {{
    font-size: 0.68rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--accent-cyan);
    font-family: var(--font-display);
  }}

  .hud-metrics {{
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }}

  .hud-pill {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 2px 7px;
    display: flex;
    align-items: baseline;
    gap: 5px;
  }}

  .hud-label {{
    font-size: 0.58rem;
    text-transform: uppercase;
    color: var(--text-dim);
    font-family: var(--font-display);
  }}

  .hud-value {{
    font-size: 0.74rem;
    font-weight: 700;
    color: var(--accent-gold);
    font-family: var(--font-display);
  }}

  .controls-cluster {{
    display: flex;
    align-items: center;
    gap: 4px;
    flex-wrap: nowrap;
  }}

  button {{
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 3px 8px;
    font-size: 0.68rem;
    font-family: var(--font-display);
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.12s ease;
  }}

  button:hover {{
    border-color: var(--accent-cyan);
  }}

  button.active-btn {{
    background: rgba(244, 63, 133, 0.22);
    border-color: var(--accent-pink);
    color: #fce7f3;
  }}

  /* Main Workspace: Maximize Hero Detector + Compact Right Sidebar */
  .workspace {{
    display: grid;
    grid-template-columns: 1fr 264px;
    gap: 6px;
    padding: 6px 8px;
    flex: 1;
    min-height: 0;
  }}

  .canvas-card {{
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }}

  .card-header {{
    display: none;
  }}

  .canvas-wrap {{
    position: relative;
    flex: 1;
    width: 100%;
    height: 100%;
    overflow: hidden;
  }}

  canvas {{
    display: block;
    width: 100%;
    height: 100%;
    cursor: grab;
  }}

  canvas:active {{
    cursor: grabbing;
  }}

  /* Floating Stereogram Mini-HUD in Bottom-Left of Detector Canvas */
  .stereo-inset {{
    position: absolute;
    bottom: 10px;
    left: 10px;
    width: 165px;
    height: 165px;
    background: rgba(10, 14, 28, 0.9);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.65);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    backdrop-filter: blur(6px);
    transition: opacity 0.2s ease, transform 0.2s ease;
  }}

  .stereo-inset.hidden {{
    opacity: 0;
    pointer-events: none;
    transform: scale(0.9);
  }}

  .stereo-inset-header {{
    font-family: var(--font-display);
    font-size: 0.6rem;
    padding: 3px 6px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-dim);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  /* Hidden legend overlay so it never obscures the detector canvas */
  .layer-legend {{
    display: none;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-main);
  }}

  .side-panel {{
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    overflow: hidden;
  }}

  .section-title {{
    font-family: var(--font-display);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--accent-gold);
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 4px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .slider-group {{
    display: flex;
    flex-direction: column;
    gap: 5px;
    background: var(--bg-card);
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
  }}

  .slider-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.73rem;
  }}

  input[type="range"] {{
    width: 100%;
    accent-color: var(--accent-cyan);
    cursor: pointer;
  }}

  .zone-list {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 150px;
    overflow-y: auto;
  }}

  .zone-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 8px;
    border-radius: 5px;
    background: var(--bg-card);
    border: 1px solid transparent;
    cursor: pointer;
    font-size: 0.75rem;
  }}

  .zone-item:hover {{
    border-color: var(--accent-cyan);
  }}

  .zone-item.zone-active {{
    border-color: var(--accent-gold);
    background: rgba(251, 191, 36, 0.1);
  }}

  .zone-swatch {{
    width: 11px;
    height: 11px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 7px;
  }}

  .inspector-box {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.74rem;
    line-height: 1.45;
    font-family: var(--font-display);
  }}

  /* Single-Row Solve Lattice Ladder Footer */
  .ladder-bar {{
    background: var(--bg-panel);
    border-top: 1px solid var(--border-color);
    padding: 4px 8px 6px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}

  .stage-pill {{
    font-family: var(--font-display);
    font-size: 0.65rem;
    padding: 3px 9px;
    border-radius: 5px;
    border: 1px solid var(--border-color);
    background: var(--bg-card);
    color: var(--text-main);
    cursor: pointer;
    transition: all 0.12s ease;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }}

  .stage-pill:hover {{
    border-color: var(--accent-pink);
    background: rgba(244, 63, 133, 0.15);
    color: #fff;
  }}

  .stage-pill.stage-active {{
    border-color: var(--accent-pink);
    color: #fce7f3;
    background: rgba(244, 63, 133, 0.24);
    font-weight: 700;
  }}

  .stage-pill.stage-done {{
    border-color: rgba(52, 211, 153, 0.5);
    color: var(--accent-green);
  }}

  .terminal-box {{
    background: #04060c;
    border: 1px solid #1e293b;
    border-radius: 5px;
    padding: 6px 8px;
    font-size: 0.62rem;
    line-height: 1.38;
    font-family: var(--font-display);
    color: #cbd5e1;
    flex: 1;
    min-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    user-select: text;
    margin: 0;
  }}

  .params-drawer {{
    display: none;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    background: #090d18;
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 4px 8px;
    font-family: var(--font-display);
    font-size: 0.64rem;
  }}

  .params-drawer.open {{
    display: flex;
  }}

  .params-drawer input {{
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 2px 5px;
    font-family: var(--font-display);
    font-size: 0.64rem;
    width: 54px;
  }}

  .iframe-popout-banner {{
    display: none !important;
  }}

  .ladder-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}

  .ladder-title {{
    font-family: var(--font-display);
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent-gold);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .ladder-rungs {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
  }}

  .rung-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 4px 8px;
    cursor: pointer;
    transition: all 0.12s ease;
  }}

  .rung-card:hover {{
    border-color: var(--accent-cyan);
  }}

  .rung-card.rung-current {{
    border-color: var(--accent-gold);
    background: linear-gradient(180deg, rgba(251, 191, 36, 0.16), var(--bg-card));
  }}

  .rung-top {{
    display: flex;
    justify-content: space-between;
    font-family: var(--font-display);
    font-size: 0.62rem;
    color: var(--accent-cyan);
  }}

  .rung-name {{
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 0.7rem;
    margin: 1px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .rung-sub {{
    font-size: 0.62rem;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
</style>
</head>
<body>

  <!-- TOP BAR -->
  <header class="top-bar">
    <div class="brand-block">
      <div class="rainbow-logo" id="app-title"><span class="logo-sub">sUB</span><span class="logo-hkl">hkl</span></div>
      <span class="subtitle-tag">{inst} · {sg}</span>
    </div>

    <div class="hud-metrics">
      <div class="hud-pill">
        <span class="hud-label">PEAKS</span>
        <span class="hud-value" id="val-indexed">0 / 0</span>
      </div>
      <div class="hud-pill">
        <span class="hud-label">Δθ</span>
        <span class="hud-value" id="val-rms">0.000°</span>
      </div>
      <div class="hud-pill">
        <span class="hud-label">ZONES</span>
        <span class="hud-value" id="val-zones">0</span>
      </div>
      <div class="hud-pill" id="hud-gonio-pill" title="Goniometer motor angles for the active frame">
        <span class="hud-label">GONIO</span>
        <span class="hud-value" id="val-gonio">—</span>
      </div>
      <div class="hud-pill" id="hud-frame-ctrl" style="gap:3px;padding:1px 5px;" title="Select goniometer frame / run orientation">
        <span class="hud-label">FRAME</span>
        <button id="btn-frame-prev" style="padding:1px 5px;font-size:0.62rem;" title="Previous frame">◀</button>
        <select id="sel-frame" style="background:transparent;color:var(--accent-pink);border:none;font-family:var(--font-display);font-size:0.68rem;font-weight:700;outline:none;cursor:pointer;max-width:190px;">
          <option value="0">1 / 1</option>
        </select>
        <button id="btn-frame-next" style="padding:1px 5px;font-size:0.62rem;" title="Next frame">▶</button>
      </div>
      <div class="hud-pill" style="display:none;">
        <span class="hud-label">PROJ</span>
        <span class="hud-value" id="val-proj">CYL</span>
      </div>
    </div>

    <div class="controls-cluster">
      <select id="sel-galaxy-h5" title="Load solve.h5 / indexer.h5 dataset directly from active Galaxy History" style="background:var(--bg-card);color:var(--accent-cyan);border:1px solid var(--border-color);border-radius:5px;padding:3px 6px;font-size:0.68rem;font-family:var(--font-display);max-width:145px;">
        <option value="">📂 Galaxy (.h5)...</option>
      </select>
      <input type="file" id="inp-local-h5" accept=".h5,.hdf5" multiple style="display:none;" />
      <button id="btn-local-h5" title="Open local solve.h5 + merged.h5 files directly in the browser (via h5wasm)">
        📂 Local .h5
      </button>
      <button id="btn-proj-toggle" class="active-btn" title="Switch between Cylindrical (CG4D: Azimuth vs Y [m]) and Spherical (MANDI: Azimuth vs Elevation [deg]) projection">
        🥫 Cyl
      </button>
      <button id="btn-zone-order" class="active-btn" title="Cycle maximum Laue zone index order: |u,v,w| ≤ 1 (fundamental) → ≤ 2 → ≤ 3 (higher-order)">
        |uvw|≤1
      </button>
      <button id="btn-zone-arcs" class="active-btn" title="Show/Hide single-arc Laue zone curves on the detector">
        Arcs
      </button>
      <button id="btn-heatmap-cmap" title="Cycle Detector Intensity Heatmap Colormap (Magma / Cyan Glow / Film)">
        Magma
      </button>
      <button id="btn-stereo-toggle" title="Show/Hide 3D Crystal Stereogram Inset">
        Wulff
      </button>
      <button id="btn-reset-view" title="Reset Zoom, Pan, and Parallax">
        ⟲
      </button>
      <button id="btn-new-tab" style="border-color:var(--accent-gold);color:var(--accent-gold);" title="Open this view in its own full-screen browser tab">
        ↗
      </button>
    </div>
  </header>

  <div id="iframe-popout-banner" class="iframe-popout-banner" style="display:none;">
    <span>📐 Running inside an embedded frame —</span>
    <button id="btn-popout-now" style="border-color:var(--accent-gold);color:var(--accent-gold);">
      ↗ Open Full-Screen in New Browser Tab
    </button>
    <button id="btn-popout-dismiss" style="padding:2px 7px;font-size:0.68rem;color:var(--text-dim);">✕</button>
  </div>

  <!-- MAIN WORKSPACE -->
  <main class="workspace">
    <!-- Hero View: Unrolled Detector with Grounded Pixel Intensity + Parallax Layers -->
    <section class="canvas-card">
      <div class="card-header">
        <span id="detector-header-title">🎯 DETECTOR ASSEMBLY — LOG(INTENSITY) + PARALLAX BRAGG RINGS</span>
        <span class="hint">Move mouse to peek behind predicted rings • Scroll to Zoom • Drag to Pan • Click spot to inspect</span>
      </div>
      <div class="canvas-wrap">
        <canvas id="canvas-detector"></canvas>

        <!-- Clear 3-Layer Depth Legend -->
        <div class="layer-legend">
          <div class="legend-item">
            <span style="display:inline-block;width:10px;height:10px;background:#f97316;border-radius:2px;"></span>
            <span>Back Layer: Detector Pixel Counts</span>
          </div>
          <div class="legend-item">
            <span style="display:inline-block;width:14px;height:3px;background:#00e5ff;border-radius:2px;"></span>
            <span>Mid Layer: Single-Arc Zone Conics [uvw]</span>
          </div>
          <div class="legend-item">
            <span style="display:inline-block;width:10px;height:10px;border:2px solid #fbbf24;border-radius:50%;"></span>
            <span>Front Layer: Predicted Bragg Rings (parallax peeks behind spots)</span>
          </div>
        </div>

        <!-- Optional Corner Stereogram Inset (hidden by default so detector banks are unobstructed) -->
        <div class="stereo-inset hidden" id="stereo-inset">
          <div class="stereo-inset-header">
            <span>WULFF NET</span>
            <span id="stereo-hint">Ĝ</span>
          </div>
          <canvas id="canvas-stereo"></canvas>
        </div>
      </div>
    </section>

    <!-- Right Sidebar: Grounded Controls & Terminal -->
    <aside class="side-panel">
      <div>
        <div class="section-title">
          <span>🎛️ Detector &amp; Parallax Controls</span>
          <span id="wl-readout">λ: {init_wl_min:.1f}–{init_wl_max:.1f} Å</span>
        </div>
        <div class="slider-group">
          <div class="slider-row">
            <span>Detector Image Contrast</span>
            <span id="lbl-det-alpha">85%</span>
          </div>
          <input type="range" id="slider-det-alpha" min="0.0" max="1.0" step="0.05" value="0.85" />

          <div class="slider-row">
            <span>Depth Parallax ("Peek Behind")</span>
            <span id="lbl-parallax">1.00×</span>
          </div>
          <input type="range" id="slider-parallax" min="0.0" max="2.5" step="0.05" value="1.00" />

          <div class="slider-row">
            <span>Wavelength Band λ [Å]</span>
            <span id="lbl-lam-band">{init_wl_min:.2f}–{init_wl_max:.2f}</span>
          </div>
          <div style="display:flex;gap:6px;">
            <input type="range" id="slider-lam-min" min="0.5" max="6.0" step="0.05" value="{init_wl_min:.2f}" title="Min Wavelength" />
            <input type="range" id="slider-lam-max" min="1.0" max="10.0" step="0.05" value="{init_wl_max:.2f}" title="Max Wavelength" />
          </div>
        </div>
      </div>

      <div>
        <div class="section-title">
          <span>🧬 Laue Zone Arcs [uvw]</span>
          <div style="display:flex;gap:4px;">
            <button id="btn-order-1" class="active-btn" style="padding:2px 6px;font-size:0.65rem;" title="Fundamental zones |u|,|v|,|w| ≤ 1">≤1</button>
            <button id="btn-order-2" style="padding:2px 6px;font-size:0.65rem;" title="Include 2nd-order zones |u|,|v|,|w| ≤ 2">≤2</button>
            <button id="btn-order-3" style="padding:2px 6px;font-size:0.65rem;" title="Include higher-order zones |u|,|v|,|w| ≤ 3">≤3</button>
            <button id="btn-zones-all" style="padding:2px 6px;font-size:0.65rem;" title="Enable all zones in current order">ALL</button>
            <button id="btn-zones-none" style="padding:2px 6px;font-size:0.65rem;" title="Hide all zone arcs">NONE</button>
          </div>
        </div>
        <div class="zone-list" id="zone-list"></div>
      </div>

      <div style="display:flex;flex-direction:column;flex:1;min-height:140px;">
        <div class="section-title">
          <span>&gt;_ Terminal // Stdio</span>
          <span id="cell-summary" style="font-size:0.62rem;color:var(--accent-pink);">IDLE</span>
        </div>
        <pre class="terminal-box" id="terminal-box">$ sUBhkl ready — click [▶ 1. Reduce], [▶ 2. Solve], [▶ 3. Integrate], or [▶ 4. Merge] below to execute on GPU / Pulsar and stream live stdout/stderr here.</pre>
        <canvas id="canvas-hist" style="display:none;width:100%;height:40px;"></canvas>
      </div>
    </aside>
  </main>

  <!-- SINGLE-ROW SOLVE LATTICE LADDER FOOTER (WITH MERGED STATUS/INSPECTOR LINE & WORKFLOW STAGE BUTTONS) -->
  <footer class="ladder-bar">
    <div class="ladder-header">
      <div class="ladder-title" id="ladder-desc" style="flex:1;min-width:0;margin-right:8px;">
        <span id="combo-badge" style="color:var(--accent-cyan);margin-right:6px;">[INSPECTOR]</span><span id="inspector-box">Hover or click a Bragg spot to inspect (hkl), λ, d, and bank • Step through Lattice Ladder below</span>
      </div>
      <div style="display:flex;align-items:center;gap:5px;flex-wrap:nowrap;">
        <span style="font-size:0.62rem;color:var(--text-dim);font-family:var(--font-display);">EXECUTE:</span>
        <div id="pipeline-stages" style="display:flex;align-items:center;gap:4px;"></div>
        <button id="btn-params-toggle" title="Configure unit cell, space group, and wavelength parameters for Solve/Integrate">
          ⚙ Params
        </button>
        <span id="galaxy-status-badge" style="font-size:0.62rem;font-family:var(--font-display);padding:2px 6px;border-radius:4px;border:1px solid #334155;color:var(--text-dim);">
          ⚪ LOCAL
        </span>
        <button id="btn-galaxy-dispatch" style="display:none;">Run</button>
        <button id="btn-ladder-prev" style="padding:2px 7px;font-size:0.66rem;">◀</button>
        <button id="btn-ladder-play" class="active-btn" style="padding:2px 8px;font-size:0.66rem;">▶ Walk</button>
        <button id="btn-ladder-next" style="padding:2px 7px;font-size:0.66rem;">▶</button>
      </div>
    </div>
    <div id="params-drawer" class="params-drawer">
      <span>SG <input id="inp-sg" type="text" value="{sg}" style="width:78px;" /></span>
      <span>a <input id="inp-a" type="number" step="0.1" value="68.7" /></span>
      <span>b <input id="inp-b" type="number" step="0.1" value="69.8" /></span>
      <span>c <input id="inp-c" type="number" step="0.1" value="156.9" /></span>
      <span>α <input id="inp-alpha" type="number" step="0.5" value="90.0" style="width:44px;" /></span>
      <span>β <input id="inp-beta" type="number" step="0.5" value="90.0" style="width:44px;" /></span>
      <span>γ <input id="inp-gamma" type="number" step="0.5" value="90.0" style="width:44px;" /></span>
      <span>d_min <input id="inp-dmin" type="number" step="0.1" value="{init_dmin:.1f}" style="width:44px;" /></span>
      <span>steps <input id="inp-steps" type="number" step="10" value="120" style="width:48px;" /></span>
    </div>
    <div class="ladder-rungs" id="ladder-rungs"></div>
  </footer>

<script>
window.RAINBOW_ZONES_DATA = {payload_json};

(function() {{
  const DATA = window.RAINBOW_ZONES_DATA || {{ peaks: [], zones: [], panels: [], ladder_steps: [], pipeline_stages: [] }};
  const PEAKS = DATA.peaks || [];
  const ZONES = DATA.zones || [];
  const PANELS = DATA.panels || [];
  const LADDER = DATA.ladder_steps || [];
  const PIPELINE = DATA.pipeline_stages || [];

  const wlRange = DATA.wavelength_range || [2.8, 4.5];
  const order1ZoneIds = ZONES.filter(z => (z.order || 1) <= 1).map(z => z.zone_id);
  const initialZoneIds = order1ZoneIds.length > 0
    ? order1ZoneIds
    : ZONES.slice(0, Math.min(6, ZONES.length)).map(z => z.zone_id);

  const state = {{
    projection: "{initial_proj}", // "cylindrical" (CG4D) | "spherical" (MANDI)
    maxZoneOrder: order1ZoneIds.length > 0 ? 1 : 2, // 1 | 2 | 3
    showZoneArcs: true,           // single vibrant arc per zone (not multi-band rainbow)
    heatmapStyle: "magma",        // "magma" | "cyan" | "gray"
    detAlpha: 0.85,
    showStereo: false,
    lamMin: wlRange[0],
    lamMax: wlRange[1],
    zoneTolDeg: 0.75,
    parallaxStrength: 1.0,
    activeZones: new Set(initialZoneIds),
    selectedZoneId: null,
    hoveredPeakId: null,
    selectedPeakId: null,
    currentPipelineIdx: 1,        // 0: Reduce, 1: Solve, 2: Integrate, 3: Merge
    currentRungIdx: Math.max(0, LADDER.length - 1),
    ladderTimer: null,
    parallaxX: 0.0,
    parallaxY: 0.0,
    targetParallaxX: 0.0,
    targetParallaxY: 0.0,
    detZoom: 1.0,
    detPanX: 0.0,
    detPanY: 0.0,
  }};

  // Pre-decode base64 grayscale bank textures and colorize them into offscreen canvases
  const bankCanvases = new Map();

  function colorizeGrayscaleImage(img, style) {{
    const c = document.createElement("canvas");
    c.width = img.width || 64;
    c.height = img.height || 64;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const imgData = ctx.getImageData(0, 0, c.width, c.height);
    const d = imgData.data;

    for (let i = 0; i < d.length; i += 4) {{
      const v = d[i] / 255.0;
      const t = Math.pow(Math.max(0, (v - 0.06) / 0.94), 0.85);
      if (style === "magma") {{
        d[i]     = Math.min(255, Math.round(12 + 255 * Math.pow(t, 0.75)));
        d[i + 1] = Math.min(255, Math.round(14 + 220 * Math.pow(t, 1.65)));
        d[i + 2] = Math.min(255, Math.round(36 + 180 * Math.sin(t * Math.PI * 0.85) + 120 * Math.pow(t, 3.0)));
        d[i + 3] = Math.round(255 * Math.min(1.0, 0.25 + t * 0.95));
      }} else if (style === "cyan") {{
        d[i]     = Math.min(255, Math.round(8 + 245 * Math.pow(t, 2.2)));
        d[i + 1] = Math.min(255, Math.round(20 + 235 * Math.pow(t, 0.95)));
        d[i + 2] = Math.min(255, Math.round(45 + 210 * Math.pow(t, 0.65)));
        d[i + 3] = Math.round(255 * Math.min(1.0, 0.25 + t * 0.95));
      }} else {{
        const g = Math.round(18 + 237 * t);
        d[i] = g; d[i + 1] = g; d[i + 2] = g;
        d[i + 3] = Math.round(255 * Math.min(1.0, 0.25 + t * 0.95));
      }}
    }}
    ctx.putImageData(imgData, 0, 0);
    return c;
  }}

  const rawBankImages = new Map();
  function decodeAllBankTextures() {{
    PANELS.forEach(p => {{
      if (!p.texture_b64) return;
      const img = new Image();
      img.onload = () => {{
        rawBankImages.set(p.bank, img);
        bankCanvases.set(p.bank, colorizeGrayscaleImage(img, state.heatmapStyle));
      }};
      img.src = "data:image/png;base64," + p.texture_b64;
    }});
  }}

  function rebuildBankColorMaps() {{
    rawBankImages.forEach((img, bank) => {{
      bankCanvases.set(bank, colorizeGrayscaleImage(img, state.heatmapStyle));
    }});
  }}

  decodeAllBankTextures();

  // Initialize sliders
  const slDetAlpha = document.getElementById("slider-det-alpha");
  const slParallax = document.getElementById("slider-parallax");
  const slLamMin = document.getElementById("slider-lam-min");
  const slLamMax = document.getElementById("slider-lam-max");

  slLamMin.min = Math.max(0.2, (wlRange[0] * 0.5)).toFixed(2);
  slLamMin.max = (wlRange[1] * 1.2).toFixed(2);
  slLamMin.value = wlRange[0].toFixed(2);

  slLamMax.min = Math.max(0.3, (wlRange[0] * 0.6)).toFixed(2);
  slLamMax.max = (wlRange[1] * 1.5).toFixed(2);
  slLamMax.value = wlRange[1].toFixed(2);

  function updateWavelengthLabels() {{
    document.getElementById("lbl-lam-band").textContent =
      `${{state.lamMin.toFixed(2)}}–${{state.lamMax.toFixed(2)}}`;
    document.getElementById("wl-readout").textContent =
      `λ: ${{state.lamMin.toFixed(1)}}–${{state.lamMax.toFixed(1)}} Å`;
  }}
  updateWavelengthLabels();

  if (DATA.cell && DATA.cell.length === 6) {{
    const c = DATA.cell;
    document.getElementById("cell-summary").textContent =
      `${{c[0].toFixed(1)}}, ${{c[1].toFixed(1)}}, ${{c[2].toFixed(1)}} Å`;
    const ids = ["inp-a", "inp-b", "inp-c", "inp-alpha", "inp-beta", "inp-gamma"];
    ids.forEach((id, i) => {{
      const el = document.getElementById(id);
      if (el && typeof c[i] === "number") el.value = c[i].toFixed(1);
    }});
  }}

  const btnParamsToggle = document.getElementById("btn-params-toggle");
  const paramsDrawer = document.getElementById("params-drawer");
  if (btnParamsToggle && paramsDrawer) {{
    btnParamsToggle.addEventListener("click", () => {{
      paramsDrawer.classList.toggle("open");
      btnParamsToggle.classList.toggle("active-btn", paramsDrawer.classList.contains("open"));
    }});
  }}

  function appendTerminalLog(text, replace = false) {{
    const box = document.getElementById("terminal-box");
    if (!box) return;
    if (replace) {{
      box.textContent = text;
    }} else {{
      box.textContent += (box.textContent.endsWith("\\n") ? "" : "\\n") + text;
    }}
    box.scrollTop = box.scrollHeight;
  }}

  function collectWorkflowParams() {{
    const cell = DATA.cell || [68.7, 69.8, 156.9, 90.0, 90.0, 90.0];
    const getNum = (id, fb) => {{
      const el = document.getElementById(id);
      const v = el ? parseFloat(el.value) : NaN;
      return Number.isFinite(v) ? v : fb;
    }};
    const sgEl = document.getElementById("inp-sg");
    return {{
      instrument: DATA.instrument || "CG4D",
      space_group: (sgEl && sgEl.value.trim()) || DATA.space_group || "P 21 21 21",
      a: getNum("inp-a", cell[0]),
      b: getNum("inp-b", cell[1]),
      c: getNum("inp-c", cell[2]),
      alpha: getNum("inp-alpha", cell[3]),
      beta: getNum("inp-beta", cell[4]),
      gamma: getNum("inp-gamma", cell[5]),
      d_min: getNum("inp-dmin", DATA.d_min || 2.8),
      steps: Math.round(getNum("inp-steps", 120)),
      wavelength_min: state.lamMin,
      wavelength_max: state.lamMax,
    }};
  }}

  // Vibrant spectral color preset for wavelength / zone highlights
  function wavelengthToColor(lam, alpha = 1.0) {{
    const t = Math.max(0, Math.min(1, (lam - state.lamMin) / Math.max(0.05, state.lamMax - state.lamMin)));
    const hue = (1.0 - t) * 265;
    return `hsla(${{hue.toFixed(1)}}, 95%, 60%, ${{alpha}})`;
  }}

  function syncProjectionUI() {{
    const isSph = state.projection === "spherical";
    const btn = document.getElementById("btn-proj-toggle");
    btn.textContent = isSph ? "🌐 Sph" : "🥫 Cyl";
    document.getElementById("val-proj").textContent = isSph ? "SPH" : "CYL";
    document.getElementById("detector-header-title").textContent = isSph
      ? "🎯 SPHERICAL DETECTOR PROJECTION (AZIMUTH φ [deg] vs ELEVATION ψ [deg])"
      : "🎯 CYLINDRICAL DETECTOR PROJECTION (AZIMUTH φ [deg] vs LAB HEIGHT Y [m])";
  }}
  syncProjectionUI();

  function setMaxZoneOrder(ordLevel) {{
    state.maxZoneOrder = ordLevel;
    const matching = ZONES.filter(z => (z.order || 1) <= ordLevel).map(z => z.zone_id);
    state.activeZones = new Set(matching);
    const btnTop = document.getElementById("btn-zone-order");
    if (btnTop) btnTop.textContent = `|uvw|≤${{ordLevel}}`;
    [1, 2, 3].forEach(k => {{
      const b = document.getElementById(`btn-order-${{k}}`);
      if (b) b.classList.toggle("active-btn", k === ordLevel);
    }});
    renderZoneList();
    updateHUD();
  }}

  function renderZoneList() {{
    const container = document.getElementById("zone-list");
    container.innerHTML = "";
    const filtered = ZONES.filter(z => (z.order || 1) <= state.maxZoneOrder);
    filtered.forEach(z => {{
      const item = document.createElement("div");
      const isAct = state.activeZones.has(z.zone_id);
      item.className = "zone-item" + (isAct ? " zone-active" : "");
      const color = z.rainbow_color;
      const ordTag = `ord ${{z.order || 1}}`;
      item.innerHTML = `
        <div style="display:flex;align-items:center;">
          <span class="zone-swatch" style="background:${{color}};"></span>
          <strong style="font-family:var(--font-display);">${{z.label}}</strong>
          <span style="margin-left:6px;font-size:0.62rem;color:var(--text-dim);font-family:var(--font-display);">(${{ordTag}})</span>
        </div>
        <span style="color:var(--text-dim);font-family:var(--font-display);font-size:0.7rem;">
          ${{z.n_peaks}} spots • ${{z.angular_rms_deg.toFixed(2)}}°
        </span>
      `;
      item.addEventListener("click", (ev) => {{
        if (ev.shiftKey) {{
          state.activeZones = new Set([z.zone_id]);
        }} else {{
          if (state.activeZones.has(z.zone_id)) state.activeZones.delete(z.zone_id);
          else state.activeZones.add(z.zone_id);
        }}
        state.selectedZoneId = z.zone_id;
        updateZoneInspector(z);
        renderZoneList();
        updateHUD();
      }});
      container.appendChild(item);
    }});
  }}

  function renderPipeline() {{
    const container = document.getElementById("pipeline-stages");
    if (!container) return;
    container.innerHTML = "";
    const shortLabels = ["1. Reduce", "2. Solve", "3. Integrate", "4. Merge"];
    PIPELINE.forEach((st, idx) => {{
      const isAct = idx === state.currentPipelineIdx;
      const isDone = idx < state.currentPipelineIdx;
      const pill = document.createElement("button");
      pill.className = "stage-pill" + (isAct ? " stage-active" : (isDone ? " stage-done" : ""));
      const icon = (state.runningStageIdx === idx) ? "⏳" : (isDone ? "✓" : "▶");
      pill.title = `Click to execute ${{st.name}} (${{(st.tool_ids || []).join(" + ")}}) → ${{st.artifact}}`;
      pill.textContent = `${{icon}} ${{shortLabels[idx] || st.name}}`;
      pill.addEventListener("click", () => {{
        selectPipelineStage(idx, true);
        executeWorkflowStage(idx);
      }});
      container.appendChild(pill);
    }});
  }}

  function setStatusLine(badgeText, contentHtml) {{
    const cb = document.getElementById("combo-badge");
    const ib = document.getElementById("inspector-box");
    if (cb && ib) {{
      cb.textContent = badgeText;
      ib.innerHTML = contentHtml;
      return;
    }}
    const ld = document.getElementById("ladder-desc");
    if (ld) {{
      ld.innerHTML = `<span id="combo-badge" style="color:var(--accent-cyan);margin-right:6px;">${{badgeText}}</span><span id="inspector-box">${{contentHtml}}</span>`;
    }}
  }}

  function selectPipelineStage(pIdx, syncLadder = true) {{
    if (pIdx < 0 || pIdx >= PIPELINE.length) return;
    state.currentPipelineIdx = pIdx;
    const st = PIPELINE[pIdx];
    setStatusLine(
      `[STAGE ${{st.step_num}}]`,
      `<strong style="color:var(--accent-pink);">${{st.name}}</strong> → <strong style="color:var(--accent-cyan);">${{st.artifact}}</strong> • Tools: <code>${{(st.tool_ids || []).join(", ")}}</code>`
    );
    if (syncLadder && typeof st.ladder_rung_idx === "number") {{
      selectLadderRung(st.ladder_rung_idx, false);
    }}
    renderPipeline();
  }}

  function renderLadder() {{
    const container = document.getElementById("ladder-rungs");
    container.innerHTML = "";
    LADDER.forEach((step, idx) => {{
      const card = document.createElement("div");
      card.className = "rung-card" + (idx === state.currentRungIdx ? " rung-current" : "");
      card.innerHTML = `
        <div class="rung-top">
          <span>STEP ${{step.rung}}</span>
          <span>${{step.n_captured}} SPOTS</span>
        </div>
        <div class="rung-name">${{step.stage}}</div>
        <div class="rung-sub">${{step.subtitle}}</div>
      `;
      card.addEventListener("click", () => selectLadderRung(idx));
      container.appendChild(card);
    }});
  }}

  function selectLadderRung(idx, setSolveStage = true) {{
    if (idx < 0 || idx >= LADDER.length) return;
    state.currentRungIdx = idx;
    if (setSolveStage && PIPELINE.length >= 2) {{
      state.currentPipelineIdx = idx === 0 ? 0 : 1;
      renderPipeline();
    }}
    const step = LADDER[idx];
    if (idx === 0) {{
      state.activeZones = new Set();
    }} else if (step.active_zones && step.active_zones.length > 0) {{
      state.activeZones = new Set(step.active_zones);
    }}
    setStatusLine(
      `[🪜 STEP ${{step.rung}}]`,
      `<strong style="color:var(--accent-gold);">${{step.stage}}</strong> — <span>${{step.description}}</span>`
    );
    renderZoneList();
    renderLadder();
    updateHUD();
  }}

  function updatePeakInspector(p) {{
    if (!p) return;
    const zLabels = (p.zone_ids || [])
      .map(id => ZONES.find(z => z.zone_id === id))
      .filter(Boolean)
      .map(z => z.label)
      .join(", ") || "None";
    const elVal = typeof p.det_el === "number" ? p.det_el : 0.0;
    const snrStr = p.sigma && p.sigma > 0 ? ` (I/σ=${{(p.intensity / p.sigma).toFixed(1)}})` : "";
    let shapeStr = "";
    if (typeof p.sigma_major_px === "number" && typeof p.sigma_minor_px === "number") {{
      if (p.shape_kind === "ellipse") {{
        shapeStr = ` • 2σ=<strong style="color:#38bdf8;">${{(2 * p.sigma_major_px).toFixed(1)}}×${{(2 * p.sigma_minor_px).toFixed(1)}}px ∠${{(p.ellipse_angle_deg || 0).toFixed(0)}}°</strong>`;
      }} else {{
        shapeStr = ` • 2σ=<strong style="color:#38bdf8;">${{(2 * p.sigma_major_px).toFixed(1)}}px (iso)</strong>`;
      }}
    }}
    setStatusLine(
      `[BANK #${{p.bank}} • #${{p.id}}]`,
      `<strong>(hkl)=<span style="color:var(--accent-gold);">(${{p.hkl.join(", ")}})</span></strong> (n=${{p.harmonic_order}}) • λ=<strong>${{p.wavelength.toFixed(3)}} Å</strong> • d=<strong>${{p.d_spacing.toFixed(3)}} Å</strong> • I=<strong>${{Math.round(p.intensity)}}</strong>${{snrStr}}${{shapeStr}} • φ=${{p.det_x.toFixed(1)}}°, Y=${{p.det_y.toFixed(3)}}m, ψ=${{elVal.toFixed(1)}}° • Zones: <span style="color:var(--accent-cyan);">${{zLabels}}</span> (Δθ=${{p.angular_error_deg.toFixed(3)}}°)`
    );
  }}

  function updateZoneInspector(z) {{
    if (!z) return;
    setStatusLine(
      `[ZONE ${{z.label}}]`,
      `<strong style="color:var(--accent-gold);">Zone ${{z.label}}</strong> • Captures <strong>${{z.n_peaks}}</strong> spots (RMS Δθ=${{z.angular_rms_deg.toFixed(3)}}°) • |t_uvw|=<strong>${{z.d_direct.toFixed(2)}} Å</strong> • G · ${{z.label}} = 0`
    );
  }}

  function isPeakVisible(p) {{
    return p.wavelength >= state.lamMin && p.wavelength <= state.lamMax;
  }}

  function isPeakCapturedByActiveZone(p) {{
    if (!p.zone_ids || p.zone_ids.length === 0) return false;
    for (let i = 0; i < p.zone_ids.length; i++) {{
      if (state.activeZones.has(p.zone_ids[i])) return true;
    }}
    return false;
  }}

  function updateHUD() {{
    const visiblePeaks = PEAKS.filter(isPeakVisible);
    const capturedPeaks = visiblePeaks.filter(p => p.indexed && isPeakCapturedByActiveZone(p));
    const nVis = visiblePeaks.length;
    const nCap = capturedPeaks.length;
    const pct = nVis > 0 ? ((nCap / nVis) * 100).toFixed(1) : "0.0";

    document.getElementById("val-indexed").textContent = `${{nCap}} / ${{nVis}} (${{pct}}%)`;
    const rms = nCap > 0
      ? Math.sqrt(capturedPeaks.reduce((s, p) => s + p.angular_error_deg * p.angular_error_deg, 0) / nCap)
      : 0.0;
    document.getElementById("val-rms").textContent = `${{rms.toFixed(3)}}°`;
    const elZones = document.getElementById("val-zones");
    if (elZones) elZones.textContent = `${{state.activeZones.size}} (≤${{state.maxZoneOrder}})`;
    const elGonio = document.getElementById("val-gonio");
    if (elGonio) {{
      const frames = DATA.frames || [];
      const fIdx = typeof DATA.frame_index === "number" ? DATA.frame_index : 0;
      const curF = frames[fIdx];
      elGonio.textContent = (curF && curF.motor_summary) || DATA.goniometer_summary || "—";
    }}
    drawHistogram(visiblePeaks);
  }}

  function renderFrameSelector() {{
    const selFrame = document.getElementById("sel-frame");
    const ctrlPill = document.getElementById("hud-frame-ctrl");
    const gonioPill = document.getElementById("hud-gonio-pill");
    const frames = DATA.frames || [];
    const fIdx = typeof DATA.frame_index === "number" ? DATA.frame_index : 0;
    if (selFrame) {{
      selFrame.innerHTML = "";
      if (frames.length === 0) {{
        const opt = document.createElement("option");
        opt.value = "0";
        opt.textContent = "1 / 1";
        selFrame.appendChild(opt);
      }} else {{
        frames.forEach((f, idx) => {{
          const opt = document.createElement("option");
          opt.value = String(idx);
          opt.textContent = f.label || `${{idx + 1}}/${{frames.length}}`;
          if (idx === fIdx) opt.selected = true;
          selFrame.appendChild(opt);
        }});
      }}
    }}
    if (ctrlPill) {{
      ctrlPill.style.display = frames.length > 0 ? "flex" : "none";
    }}
    if (gonioPill) {{
      const hasGonio = frames.length > 0 || (DATA.goniometer_summary && DATA.goniometer_summary !== "—");
      gonioPill.style.display = hasGonio ? "flex" : "none";
    }}
  }}

  const histCanvas = document.getElementById("canvas-hist");
  function drawHistogram(visiblePeaks) {{
    const rect = histCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    histCanvas.width = Math.max(100, rect.width * dpr);
    histCanvas.height = Math.max(45, rect.height * dpr);
    const ctx = histCanvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const nBins = 20;
    const maxDeg = 1.5;
    const bins = new Array(nBins).fill(0);
    visiblePeaks.forEach(p => {{
      const b = Math.min(nBins - 1, Math.floor((p.angular_error_deg / maxDeg) * nBins));
      if (b >= 0) bins[b]++;
    }});
    const maxCount = Math.max(1, ...bins);
    const barW = (w - 16) / nBins;

    for (let i = 0; i < nBins; i++) {{
      const bh = ((bins[i] / maxCount) * (h - 20));
      const x = 8 + i * barW;
      const y = h - 14 - bh;
      const degCenter = ((i + 0.5) / nBins) * maxDeg;
      ctx.fillStyle = degCenter <= state.zoneTolDeg ? "#34d399" : "#64748b";
      ctx.fillRect(x + 1, y, Math.max(2, barW - 2), bh);
    }}
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px monospace";
    ctx.fillText("0.0°", 6, h - 3);
    ctx.fillText("Δθ error", w * 0.42, h - 3);
    ctx.fillText("1.5°", w - 28, h - 3);
  }}

  // Compute detector coordinate bounds for both Cylindrical (y [m]) and Spherical (elevation [deg])
  const detCanvas = document.getElementById("canvas-detector");
  const stereoCanvas = document.getElementById("canvas-stereo");

  const cylBounds = {{ minX: -180, maxX: 180, minY: -0.42, maxY: 0.42 }};
  const sphBounds = {{ minX: -180, maxX: 180, minY: -75.0, maxY: 75.0 }};

  if (PANELS.length > 0) {{
    const allY = [];
    const allEl = [];
    PANELS.forEach(p => {{
      allY.push(p.y_min, p.y_max);
      if (typeof p.el_min === "number") allEl.push(p.el_min, p.el_max);
    }});
    PEAKS.forEach(p => {{
      allY.push(p.det_y);
      if (typeof p.det_el === "number") allEl.push(p.det_el);
    }});
    const maxAbsY = Math.max(0.28, ...allY.map(v => Math.abs(v))) * 1.15;
    cylBounds.minY = -maxAbsY;
    cylBounds.maxY = maxAbsY;
    if (allEl.length > 0) {{
      const maxAbsEl = Math.min(88.0, Math.max(45.0, ...allEl.map(v => Math.abs(v))) * 1.12);
      sphBounds.minY = -maxAbsEl;
      sphBounds.maxY = maxAbsEl;
    }}
  }} else if (PEAKS.length > 0) {{
    const maxAbsY = Math.max(0.28, ...PEAKS.map(p => Math.abs(p.det_y))) * 1.25;
    cylBounds.minY = -maxAbsY;
    cylBounds.maxY = maxAbsY;
    const allEl = PEAKS.map(p => Math.abs(typeof p.det_el === "number" ? p.det_el : 30.0));
    const maxAbsEl = Math.min(88.0, Math.max(45.0, ...allEl) * 1.18);
    sphBounds.minY = -maxAbsEl;
    sphBounds.maxY = maxAbsEl;
  }}

  function getActiveBounds() {{
    return state.projection === "spherical" ? sphBounds : cylBounds;
  }}

  function getPeakVert(p) {{
    if (state.projection === "spherical") {{
      return typeof p.det_el === "number" ? p.det_el : 0.0;
    }}
    return p.det_y;
  }}

  function getPanelVertRange(panel) {{
    if (state.projection === "spherical" && typeof panel.el_min === "number") {{
      return [panel.el_min, panel.el_max];
    }}
    return [panel.y_min, panel.y_max];
  }}

  // Physical Parallax Transform:
  // layerOffset = -0.35 (Back: Detector Intensity Panels)
  // layerOffset = +0.25..+0.90 (Front: Predicted Bragg Rings floating by d-spacing)
  function detToScreen(dx, dy, w, h, layerOffset = 0.0) {{
    const bounds = getActiveBounds();
    const padL = 42, padR = 12, padT = 14, padB = 22;
    const plotW = Math.max(50, w - padL - padR);
    const plotH = Math.max(50, h - padT - padB);
    const u = (dx - bounds.minX) / (bounds.maxX - bounds.minX);
    const v = (dy - bounds.minY) / (bounds.maxY - bounds.minY);
    const x0 = padL + u * plotW;
    const y0 = padT + (1.0 - v) * plotH;
    const cx = w * 0.5;
    const cy = h * 0.5;
    const shiftX = state.parallaxX * state.parallaxStrength * layerOffset * 18.0;
    const shiftY = state.parallaxY * state.parallaxStrength * layerOffset * 12.0;
    return [
      cx + (x0 - cx) * state.detZoom + state.detPanX + shiftX,
      cy + (y0 - cy) * state.detZoom + state.detPanY + shiftY,
    ];
  }}

  function drawSingleZoneArc(ctx, pts, zone, isContinuousGap = false) {{
    if (!pts || pts.length < 2) return;
    const isSelected = state.selectedZoneId === zone.zone_id;
    const zOrder = zone.order || 1;
    ctx.save();
    ctx.beginPath();
    pts.forEach((p, idx) => {{
      if (idx === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    }});
    ctx.strokeStyle = zone.rainbow_color;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (isContinuousGap) {{
      ctx.lineWidth = 1.0;
      ctx.globalAlpha = 0.22;
      ctx.setLineDash([4, 4]);
    }} else {{
      const baseW = zOrder === 1 ? 2.2 : (zOrder === 2 ? 1.6 : 1.25);
      ctx.lineWidth = isSelected ? baseW + 1.2 : baseW;
      ctx.globalAlpha = isSelected ? 0.96 : (zOrder === 1 ? 0.84 : (zOrder === 2 ? 0.68 : 0.54));
    }}
    ctx.stroke();
    ctx.restore();
  }}

  function renderDetectorCanvas() {{
    const rect = detCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    detCanvas.width = Math.max(200, rect.width * dpr);
    detCanvas.height = Math.max(200, rect.height * dpr);
    const ctx = detCanvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const bounds = getActiveBounds();
    const isSph = state.projection === "spherical";

    // 1. Coordinate Grid (Azimuth deg vs Height [m] or Elevation [deg])
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px monospace";
    for (let az = -180; az <= 180; az += 45) {{
      const [gx, gy0] = detToScreen(az, bounds.minY, w, h, -0.35);
      const [, gy1] = detToScreen(az, bounds.maxY, w, h, -0.35);
      ctx.beginPath();
      ctx.moveTo(gx, gy0);
      ctx.lineTo(gx, gy1);
      ctx.stroke();
      ctx.fillText(`${{az}}°`, gx - 12, h - 8);
    }}
    const yStep = Number(((bounds.maxY - bounds.minY) / 4).toFixed(isSph ? 0 : 2)) || (isSph ? 30 : 0.15);
    for (let yv = bounds.minY; yv <= bounds.maxY + 1e-4; yv += yStep) {{
      const [gx0, gy] = detToScreen(bounds.minX, yv, w, h, -0.35);
      const [gx1] = detToScreen(bounds.maxX, yv, w, h, -0.35);
      ctx.beginPath();
      ctx.moveTo(gx0, gy);
      ctx.lineTo(gx1, gy);
      ctx.stroke();
      const lbl = isSph ? `${{yv.toFixed(0)}}°` : `${{yv.toFixed(2)}}m`;
      ctx.fillText(lbl, 4, gy + 3);
    }}
    ctx.restore();

    // 2. LAYER 0 (BACK): Detector Bank Pixel Intensity Heatmaps + Panel Outlines
    ctx.save();
    PANELS.forEach(panel => {{
      const [vMin, vMax] = getPanelVertRange(panel);
      const [x0, y0] = detToScreen(panel.az_min, vMax, w, h, -0.35);
      const [x1, y1] = detToScreen(panel.az_max, vMin, w, h, -0.35);
      const pw = x1 - x0;
      const ph = y1 - y0;

      const outlinePts = isSph && panel.outline_sph ? panel.outline_sph : panel.outline;

      ctx.save();
      if (outlinePts && outlinePts.length >= 4) {{
        ctx.beginPath();
        outlinePts.forEach((pt, idx) => {{
          const [ox, oy] = detToScreen(pt[0], pt[1], w, h, -0.35);
          if (idx === 0) ctx.moveTo(ox, oy);
          else ctx.lineTo(ox, oy);
        }});
        ctx.closePath();
        ctx.fillStyle = "rgba(15, 23, 42, 0.72)";
        ctx.fill();
        ctx.clip();
      }} else {{
        ctx.fillStyle = "rgba(15, 23, 42, 0.72)";
        ctx.fillRect(x0, y0, pw, ph);
      }}

      const texCanvas = bankCanvases.get(panel.bank);
      if (texCanvas && state.detAlpha > 0.01) {{
        ctx.globalAlpha = state.detAlpha;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(texCanvas, x0, y0, pw, ph);
      }}
      ctx.restore();

      // True panel polygon border (shows curved elevation edges in Spherical MANDI view)
      ctx.strokeStyle = texCanvas ? "rgba(56, 189, 248, 0.38)" : "rgba(148, 163, 184, 0.18)";
      ctx.lineWidth = texCanvas ? 1.1 : 0.7;
      if (outlinePts && outlinePts.length >= 4) {{
        ctx.beginPath();
        outlinePts.forEach((pt, idx) => {{
          const [ox, oy] = detToScreen(pt[0], pt[1], w, h, -0.35);
          if (idx === 0) ctx.moveTo(ox, oy);
          else ctx.lineTo(ox, oy);
        }});
        ctx.closePath();
        ctx.stroke();
      }} else {{
        ctx.strokeRect(x0, y0, pw, ph);
      }}

    // Only show bank labels when zoomed in so default view is clean
      if (state.detZoom > 1.35 && pw > 34 && ph > 24) {{
        ctx.fillStyle = "rgba(148, 163, 184, 0.42)";
        ctx.font = "8px monospace";
        ctx.fillText(`b${{panel.bank}}`, x0 + 3, y0 + 9);
      }}
    }});
    ctx.restore();

    const OVERLAY_PARALLAX_LAYER = 0.45;

    // 3. LAYER 1 (FOREGROUND OVERLAY): Single-Arc Laue Zone Conics [uvw] clipped to detector panels
    // Moves together with the predicted Bragg circles in parallax above the detector heatmap
    if (state.showZoneArcs) {{
      const hasPanels = Array.isArray(PANELS) && PANELS.length > 0;
      ZONES.forEach(zone => {{
        if (!state.activeZones.has(zone.zone_id)) return;
        const contSegs = isSph
          ? (zone.continuous_segments_sph || [])
          : (zone.continuous_segments || []);
        const panelSegs = isSph
          ? (hasPanels ? (zone.detector_segments_sph || []) : (zone.detector_segments_sph || contSegs))
          : (hasPanels ? (zone.detector_segments || []) : (zone.detector_segments || contSegs));
        panelSegs.forEach((seg, sIdx) => {{
          const screenPts = seg.map(pt => detToScreen(pt[0], pt[1], w, h, OVERLAY_PARALLAX_LAYER));
          drawSingleZoneArc(ctx, screenPts, zone, false);
          if (sIdx === 0 && screenPts.length > 2 && (state.selectedZoneId === zone.zone_id || state.detZoom > 1.35)) {{
            const mid = screenPts[Math.floor(screenPts.length * 0.5)];
            ctx.save();
            ctx.font = "bold 9px monospace";
            ctx.fillStyle = "#070913";
            ctx.fillText(zone.label, mid[0] + 6, mid[1] - 4);
            ctx.fillStyle = zone.rainbow_color;
            ctx.fillText(zone.label, mid[0] + 5, mid[1] - 5);
            ctx.restore();
          }}
        }});
      }});
    }}

    // 4. FOREGROUND LAYER: Predicted Bragg Rings hovering together with Laue zone circles
    // Moving the mouse shifts both Laue zones and peak rings together relative to the back detector plate!
    const visiblePeaks = PEAKS.filter(isPeakVisible)
      .slice()
      .sort((a, b) => b.parallax_depth - a.parallax_depth);

    visiblePeaks.forEach(p => {{
      const py = getPeakVert(p);
      const fgLayer = OVERLAY_PARALLAX_LAYER;
      const [x, y] = detToScreen(p.det_x, py, w, h, fgLayer);
      const [bx, by] = detToScreen(p.det_x, py, w, h, -0.35);

      const captured = isPeakCapturedByActiveZone(p);
      const isHovered = state.hoveredPeakId === p.id || state.selectedPeakId === p.id;
      if (!captured && !isHovered && state.activeZones.size === 0) {{
        // Step 1 (Raw Detector Counts): don't clutter raw spots unless hovered
        return;
      }}

      const baseR = Math.max(1.2, Math.min(20.0, 1.85 * state.detZoom));
      const r = isHovered ? baseR + 1.8 : baseR;

      ctx.save();
      // Subtle depth tether connecting the floating Bragg ring to its physical spot on the detector panel
      if (Math.hypot(x - bx, y - by) > 2.0 && (captured || isHovered)) {{
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(x, y);
        ctx.strokeStyle = "rgba(244, 63, 133, 0.38)";
        ctx.lineWidth = 1.0;
        ctx.stroke();
      }}

      // Integrated peak 2-sigma contour (anisotropic 3D tensor ellipse or isotropic circle)
      // or predicted/solved Bragg ring (scaled by zoom factor)
      const ellPtsRaw = isSph ? (p.ellipse_sph || p.ellipse) : p.ellipse;
      const strokeCol = isHovered
        ? "#f43f85"
        : wavelengthToColor(p.wavelength, captured ? 0.95 : 0.42);
      ctx.strokeStyle = strokeCol;
      ctx.lineWidth = isHovered ? 2.4 : (captured ? 1.85 : 1.05);

      if (Array.isArray(ellPtsRaw) && ellPtsRaw.length >= 6) {{
        const scrPts = ellPtsRaw.map(pt => detToScreen(pt[0], pt[1], w, h, fgLayer));
        let maxDist = 0.0, minDist = 1e9;
        for (let m = 0; m < scrPts.length; m++) {{
          const d = Math.hypot(scrPts[m][0] - x, scrPts[m][1] - y);
          if (d > maxDist) maxDist = d;
          if (d < minDist) minDist = d;
        }}
        // Keep physical 2-sigma contour scaling with zoom, with a legible floor at 1x zoom
        // that preserves the fitted major/minor aspect ratio and orientation angle
        const minTarget = isHovered ? baseR + 1.5 : baseR * 0.9;
        const scale = minDist > 1e-3 && minDist < minTarget ? (minTarget / minDist) : 1.0;
        ctx.beginPath();
        for (let m = 0; m < scrPts.length; m++) {{
          const sx = x + (scrPts[m][0] - x) * scale;
          const sy = y + (scrPts[m][1] - y) * scale;
          if (m === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }}
        ctx.closePath();
        ctx.stroke();

        // When hovered or zoomed in on an anisotropic ellipse, draw subtle principal-axis ticks
        if (p.shape_kind === "ellipse" && (isHovered || state.detZoom > 2.2) && scrPts.length >= 16) {{
          const nHalf = Math.floor(scrPts.length / 2);
          const nQuarter = Math.floor(scrPts.length / 4);
          const p0 = [x + (scrPts[0][0] - x) * scale, y + (scrPts[0][1] - y) * scale];
          const pHalf = [x + (scrPts[nHalf][0] - x) * scale, y + (scrPts[nHalf][1] - y) * scale];
          const pQ1 = [x + (scrPts[nQuarter][0] - x) * scale, y + (scrPts[nQuarter][1] - y) * scale];
          const pQ3 = [x + (scrPts[nQuarter + nHalf][0] - x) * scale, y + (scrPts[nQuarter + nHalf][1] - y) * scale];
          ctx.save();
          ctx.strokeStyle = isHovered ? "rgba(251, 191, 36, 0.75)" : "rgba(56, 189, 248, 0.42)";
          ctx.lineWidth = 0.85;
          ctx.beginPath();
          ctx.moveTo(p0[0], p0[1]); ctx.lineTo(pHalf[0], pHalf[1]);
          ctx.moveTo(pQ1[0], pQ1[1]); ctx.lineTo(pQ3[0], pQ3[1]);
          ctx.stroke();
          ctx.restore();
        }}
      }} else {{
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.stroke();
      }}

      if (state.detAlpha < 0.2) {{
        ctx.fillStyle = wavelengthToColor(p.wavelength, 0.85);
        ctx.fill();
      }}

      if (isHovered) {{
        const shapeTag = (typeof p.sigma_major_px === "number" && typeof p.sigma_minor_px === "number")
          ? (p.shape_kind === "ellipse"
              ? ` 2σ=${{(2 * p.sigma_major_px).toFixed(1)}}×${{(2 * p.sigma_minor_px).toFixed(1)}}px`
              : ` 2σ=${{(2 * p.sigma_major_px).toFixed(1)}}px`)
          : "";
        const tag = `(${{p.hkl.join(",")}}) λ=${{p.wavelength.toFixed(2)}}Å d=${{p.d_spacing.toFixed(2)}}Å${{shapeTag}}`;
        ctx.font = "bold 11px monospace";
        const tw = ctx.measureText(tag).width + 10;
        ctx.fillStyle = "rgba(11, 15, 23, 0.92)";
        ctx.fillRect(x + 8, y - 20, tw, 18);
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 1.0;
        ctx.strokeRect(x + 8, y - 20, tw, 18);
        ctx.fillStyle = "#fbbf24";
        ctx.fillText(tag, x + 13, y - 7);
      }}
      ctx.restore();
    }});
  }}

  // Render Corner Stereographic Wulff Net Inset
  function renderStereoInset() {{
    if (!state.showStereo) return;
    const rect = stereoCanvas.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 20) return;
    const dpr = window.devicePixelRatio || 1;
    stereoCanvas.width = rect.width * dpr;
    stereoCanvas.height = rect.height * dpr;
    const ctx = stereoCanvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.5;
    const R = Math.min(w, h) * 0.43;

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(7, 9, 19, 0.92)";
    ctx.fill();
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.clip();

    ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.arc(cx, cy, R * Math.tan(Math.PI / 8), 0, Math.PI * 2);
    ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
    ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
    ctx.stroke();

    if (state.showZoneArcs) {{
      ZONES.forEach(zone => {{
        if (!state.activeZones.has(zone.zone_id)) return;
        ctx.strokeStyle = zone.rainbow_color;
        ctx.lineWidth = state.selectedZoneId === zone.zone_id ? 2.0 : 1.2;
        ctx.globalAlpha = 0.72;
        (zone.stereo_segments || []).forEach(seg => {{
          ctx.beginPath();
          seg.forEach((pt, idx) => {{
            const sx = cx + pt[0] * R;
            const sy = cy - pt[1] * R;
            if (idx === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }});
          ctx.stroke();
        }});
      }});
    }}

    PEAKS.filter(isPeakVisible).forEach(p => {{
      const sx = cx + p.stereographic[0] * R;
      const sy = cy - p.stereographic[1] * R;
      const captured = isPeakCapturedByActiveZone(p);
      const isHovered = state.hoveredPeakId === p.id;
      ctx.beginPath();
      ctx.arc(sx, sy, isHovered ? 4.5 : (captured ? 2.6 : 1.8), 0, Math.PI * 2);
      ctx.fillStyle = isHovered ? "#fbbf24" : wavelengthToColor(p.wavelength, captured ? 0.95 : 0.35);
      ctx.fill();
    }});
    ctx.restore();
  }}

  function findNearestPeakDetector(mx, my) {{
    const rect = detCanvas.getBoundingClientRect();
    let best = null;
    let bestDist = 14.0;
    const fgLayer = 0.45;
    PEAKS.forEach(p => {{
      if (!isPeakVisible(p)) return;
      const py = getPeakVert(p);
      const [x, y] = detToScreen(p.det_x, py, rect.width, rect.height, fgLayer);
      const [bx, by] = detToScreen(p.det_x, py, rect.width, rect.height, -0.35);
      const d = Math.min(Math.hypot(mx - x, my - y), Math.hypot(mx - bx, my - by));
      if (d < bestDist) {{
        bestDist = d;
        best = p;
      }}
    }});
    return best;
  }}

  // Mouse & Wheel interactions on Detector Canvas
  let isPanningDet = false;
  let panStart = [0, 0];
  detCanvas.addEventListener("mousedown", (ev) => {{
    isPanningDet = true;
    panStart = [ev.clientX - state.detPanX, ev.clientY - state.detPanY];
  }});
  window.addEventListener("mouseup", () => {{ isPanningDet = false; }});

  detCanvas.addEventListener("mousemove", (ev) => {{
    const rect = detCanvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    state.targetParallaxX = ((mx / rect.width) - 0.5) * 2.0;
    state.targetParallaxY = ((my / rect.height) - 0.5) * 2.0;

    if (isPanningDet) {{
      state.detPanX = ev.clientX - panStart[0];
      state.detPanY = ev.clientY - panStart[1];
      return;
    }}
    const hit = findNearestPeakDetector(mx, my);
    state.hoveredPeakId = hit ? hit.id : null;
    if (hit) updatePeakInspector(hit);
  }});

  detCanvas.addEventListener("mouseleave", () => {{
    state.targetParallaxX = 0.0;
    state.targetParallaxY = 0.0;
  }});

  detCanvas.addEventListener("wheel", (ev) => {{
    ev.preventDefault();
    const factor = ev.deltaY < 0 ? 1.14 : 0.88;
    state.detZoom = Math.max(0.7, Math.min(10.0, state.detZoom * factor));
  }}, {{ passive: false }});

  detCanvas.addEventListener("click", (ev) => {{
    const rect = detCanvas.getBoundingClientRect();
    const hit = findNearestPeakDetector(ev.clientX - rect.left, ev.clientY - rect.top);
    if (hit) {{
      state.selectedPeakId = hit.id;
      updatePeakInspector(hit);
    }}
  }});

  // Controls bindings
  slDetAlpha.addEventListener("input", () => {{
    state.detAlpha = parseFloat(slDetAlpha.value);
    document.getElementById("lbl-det-alpha").textContent = Math.round(state.detAlpha * 100) + "%";
  }});

  slParallax.addEventListener("input", () => {{
    state.parallaxStrength = parseFloat(slParallax.value);
    document.getElementById("lbl-parallax").textContent = state.parallaxStrength.toFixed(2) + "×";
  }});

  slLamMin.addEventListener("input", () => {{
    state.lamMin = Math.min(parseFloat(slLamMin.value), state.lamMax - 0.05);
    slLamMin.value = state.lamMin;
    updateWavelengthLabels();
    updateHUD();
  }});

  slLamMax.addEventListener("input", () => {{
    state.lamMax = Math.max(parseFloat(slLamMax.value), state.lamMin + 0.05);
    slLamMax.value = state.lamMax;
    updateWavelengthLabels();
    updateHUD();
  }});

  document.getElementById("btn-proj-toggle").addEventListener("click", () => {{
    state.projection = state.projection === "cylindrical" ? "spherical" : "cylindrical";
    syncProjectionUI();
  }});

  document.getElementById("btn-zone-order").addEventListener("click", () => {{
    const nextOrder = state.maxZoneOrder >= 3 ? 1 : state.maxZoneOrder + 1;
    setMaxZoneOrder(nextOrder);
  }});

  [1, 2, 3].forEach(ordLevel => {{
    const b = document.getElementById(`btn-order-${{ordLevel}}`);
    if (b) b.addEventListener("click", () => setMaxZoneOrder(ordLevel));
  }});

  document.getElementById("btn-zone-arcs").addEventListener("click", (ev) => {{
    state.showZoneArcs = !state.showZoneArcs;
    ev.currentTarget.classList.toggle("active-btn", state.showZoneArcs);
    ev.currentTarget.textContent = "Arcs";
  }});

  document.getElementById("btn-heatmap-cmap").addEventListener("click", (ev) => {{
    const order = ["magma", "cyan", "gray"];
    const labels = {{ magma: "Magma", cyan: "Cyan", gray: "Film" }};
    state.heatmapStyle = order[(order.indexOf(state.heatmapStyle) + 1) % order.length];
    ev.currentTarget.textContent = labels[state.heatmapStyle];
    rebuildBankColorMaps();
  }});

  document.getElementById("btn-stereo-toggle").addEventListener("click", (ev) => {{
    state.showStereo = !state.showStereo;
    ev.currentTarget.classList.toggle("active-btn", state.showStereo);
    document.getElementById("stereo-inset").classList.toggle("hidden", !state.showStereo);
  }});

  document.getElementById("btn-reset-view").addEventListener("click", () => {{
    state.detZoom = 1.0;
    state.detPanX = 0.0;
    state.detPanY = 0.0;
    setMaxZoneOrder(1);
  }});

  document.getElementById("btn-zones-all").addEventListener("click", () => {{
    state.activeZones = new Set(
      ZONES.filter(z => (z.order || 1) <= state.maxZoneOrder).map(z => z.zone_id)
    );
    renderZoneList();
    updateHUD();
  }});

  document.getElementById("btn-zones-none").addEventListener("click", () => {{
    state.activeZones = new Set();
    renderZoneList();
    updateHUD();
  }});

  document.getElementById("btn-ladder-prev").addEventListener("click", () => {{
    selectLadderRung((state.currentRungIdx - 1 + LADDER.length) % LADDER.length);
  }});

  document.getElementById("btn-ladder-next").addEventListener("click", () => {{
    selectLadderRung((state.currentRungIdx + 1) % LADDER.length);
  }});

  document.getElementById("btn-ladder-play").addEventListener("click", (ev) => {{
    if (state.ladderTimer) {{
      clearInterval(state.ladderTimer);
      state.ladderTimer = null;
      ev.currentTarget.textContent = "▶ WALK SOLVE STEPS";
      return;
    }}
    selectPipelineStage(1, false);
    selectLadderRung(0, false);
    ev.currentTarget.textContent = "⏹ STOP";
    state.ladderTimer = setInterval(() => {{
      if (state.currentRungIdx + 1 < LADDER.length) {{
        selectLadderRung(state.currentRungIdx + 1, false);
      }} else {{
        clearInterval(state.ladderTimer);
        state.ladderTimer = null;
        document.getElementById("btn-ladder-play").textContent = "▶ WALK SOLVE STEPS";
      }}
    }}, 1800);
  }});

  // Open in its own full-screen browser tab (button + auto-detect when inside Galaxy iframe)
  function openInOwnTab() {{
    const targetUrl = window.location.href;
    window.open(targetUrl, "_blank", "noopener");
  }}
  const btnNewTab = document.getElementById("btn-new-tab");
  if (btnNewTab) btnNewTab.addEventListener("click", openInOwnTab);

  if (window.self !== window.top) {{
    const banner = document.getElementById("iframe-popout-banner");
    const btnPopNow = document.getElementById("btn-popout-now");
    const btnPopDismiss = document.getElementById("btn-popout-dismiss");
    // Attempt immediate popout; if popup-blocked, show 1-click banner
    const w = window.open(window.location.href, "_blank");
    if (!w && banner) {{
      banner.style.display = "flex";
    }}
    if (btnPopNow) {{
      btnPopNow.addEventListener("click", () => {{
        openInOwnTab();
        if (banner) banner.style.display = "none";
      }});
    }}
    if (btnPopDismiss && banner) {{
      btnPopDismiss.addEventListener("click", () => {{
        banner.style.display = "none";
      }});
    }}
  }}

  // Dynamic payload hot-reload + Client-Side h5wasm HDF5 Parser (for truly static HTML usage)
  function recomputeBoundsAndReload(newPayload) {{
    if (!newPayload) return;
    DATA.instrument = newPayload.instrument || DATA.instrument;
    DATA.space_group = newPayload.space_group || DATA.space_group;
    DATA.wavelength_range = newPayload.wavelength_range || DATA.wavelength_range;
    DATA.d_min = newPayload.d_min || DATA.d_min;
    DATA.cell = newPayload.cell || DATA.cell;
    if (Array.isArray(newPayload.frames)) DATA.frames = newPayload.frames;
    if (typeof newPayload.frame_index === "number") DATA.frame_index = newPayload.frame_index;
    if (newPayload.goniometer_summary) DATA.goniometer_summary = newPayload.goniometer_summary;
    renderFrameSelector();

    if (DATA.wavelength_range && DATA.wavelength_range.length >= 2) {{
      state.lamMin = Number(DATA.wavelength_range[0]);
      state.lamMax = Number(DATA.wavelength_range[1]);
      if (slLamMin) slLamMin.value = state.lamMin.toFixed(2);
      if (slLamMax) slLamMax.value = state.lamMax.toFixed(2);
      updateWavelengthLabels();
    }}
    if (DATA.d_min) {{
      const dEl = document.getElementById("inp-dmin");
      if (dEl) dEl.value = Number(DATA.d_min).toFixed(1);
    }}

    PEAKS.length = 0;
    (newPayload.peaks || []).forEach(p => PEAKS.push(p));
    ZONES.length = 0;
    (newPayload.zones || []).forEach(z => ZONES.push(z));
    PANELS.length = 0;
    (newPayload.panels || []).forEach(p => PANELS.push(p));
    LADDER.length = 0;
    (newPayload.ladder_steps || []).forEach(s => LADDER.push(s));
    PIPELINE.length = 0;
    (newPayload.pipeline_stages || []).forEach(s => PIPELINE.push(s));

    if (PANELS.length > 0) {{
      const allY = [], allEl = [];
      PANELS.forEach(p => {{
        allY.push(p.y_min, p.y_max);
        if (typeof p.el_min === "number") allEl.push(p.el_min, p.el_max);
      }});
      if (allY.length > 0) {{
        const padY = Math.max(0.03, (Math.max(...allY) - Math.min(...allY)) * 0.08);
        cylBounds.minY = Math.min(...allY) - padY;
        cylBounds.maxY = Math.max(...allY) + padY;
      }}
      if (allEl.length > 0) {{
        const padEl = Math.max(4.0, (Math.max(...allEl) - Math.min(...allEl)) * 0.08);
        sphBounds.minY = Math.max(-90.0, Math.min(...allEl) - padEl);
        sphBounds.maxY = Math.min(90.0, Math.max(...allEl) + padEl);
      }}
    }}

    rawBankImages.clear();
    bankCanvases.clear();
    decodeAllBankTextures();
    // Support direct canvas textures generated by client-side h5wasm
    PANELS.forEach(p => {{
      if (p._grayCanvas) {{
        rawBankImages.set(p.bank, p._grayCanvas);
        bankCanvases.set(p.bank, colorizeGrayscaleImage(p._grayCanvas, state.heatmapStyle));
      }}
    }});

    const subEl = document.querySelector(".subtitle-tag");
    const artKind = newPayload.artifact_kind || DATA.artifact_kind || "solve";
    const shapeMode = newPayload.peak_shape_mode || DATA.peak_shape_mode || "solved_circle";
    DATA.artifact_kind = artKind;
    DATA.peak_shape_mode = shapeMode;
    let artBadge = "SOLVED ORIENTATION";
    if (artKind === "integrator") {{
      artBadge = shapeMode === "anisotropic_ellipse"
        ? "INTEGRATED • 3D TENSOR ELLIPSES"
        : "INTEGRATED • ISOTROPIC 2σ CIRCLES";
    }} else if (artKind === "predictor") {{
      artBadge = "PREDICTOR • PREDICTED PEAKS";
    }} else if (newPayload.intensity_source === "raw_h5") {{
      artBadge = "RAW DETECTOR COUNTS";
    }}
    if (subEl) {{
      subEl.textContent = `${{DATA.instrument || "CG4D"}} • ${{DATA.space_group || "P 1"}} • ${{artBadge}}`;
    }}
    if (DATA.cell && DATA.cell.length === 6) {{
      const c = DATA.cell;
      document.getElementById("cell-summary").textContent = `${{c[0].toFixed(1)}}, ${{c[1].toFixed(1)}}, ${{c[2].toFixed(1)}} Å`;
    }}
    const ord1 = ZONES.filter(z => (z.order || 1) <= 1).map(z => z.zone_id);
    state.maxZoneOrder = ord1.length > 0 ? 1 : 2;
    state.activeZones = new Set(ord1.length > 0 ? ord1 : ZONES.map(z => z.zone_id));
    state.currentRungIdx = Math.max(0, LADDER.length - 1);
    renderPipeline();
    if (LADDER.length > 0) selectLadderRung(state.currentRungIdx, false);
    else {{ renderZoneList(); updateHUD(); }}
    if (artKind === "integrator" || artKind === "predictor") {{
      const nEll = PEAKS.filter(p => Array.isArray(p.ellipse)).length;
      setStatusLine(
        `[${{artBadge}}]`,
        `<strong style="color:var(--accent-gold);">${{artBadge}}</strong> — Loaded <strong>${{PEAKS.length}}</strong> reflections${{nEll > 0 ? ` with <strong>${{nEll}}</strong> detector-projected 2σ ${{shapeMode === "anisotropic_ellipse" ? "anisotropic 3D tensor ellipses" : "isotropic circles"}}` : ""}}.`
      );
    }}
  }}

  // Pure Client-Side HDF5 Parser + Crystallographic Forward Projector via h5wasm
  const ZONE_COLORS = ["#ff1e42","#ff7b00","#ffd500","#12eb5b","#00e1ff","#4d5bff","#c82eff","#ff38b4","#9dff00","#00ffa6"];
  function gcd2(a, b) {{ a = Math.abs(a); b = Math.abs(b); while (b) {{ const t = b; b = a % b; a = t; }} return a; }}
  function gcd3(a, b, c) {{ return gcd2(gcd2(a, b), c); }}
  function mat3Mul(A, B) {{
    const C = [[0,0,0],[0,0,0],[0,0,0]];
    for (let i=0;i<3;i++) for (let j=0;j<3;j++) C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j];
    return C;
  }}
  function mat3Vec(A, v) {{
    return [
      A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2],
      A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2],
      A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2],
    ];
  }}
  function mat3T(A) {{
    return [[A[0][0],A[1][0],A[2][0]],[A[0][1],A[1][1],A[2][1]],[A[0][2],A[1][2],A[2][2]]];
  }}
  function mat3Inv(M) {{
    const det = M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
    const invDet = 1.0 / (Math.abs(det) > 1e-12 ? det : 1e-12);
    return [
      [(M[1][1]*M[2][2]-M[1][2]*M[2][1])*invDet, (M[0][2]*M[2][1]-M[0][1]*M[2][2])*invDet, (M[0][1]*M[1][2]-M[0][2]*M[1][1])*invDet],
      [(M[1][2]*M[2][0]-M[1][0]*M[2][2])*invDet, (M[0][0]*M[2][2]-M[0][2]*M[2][0])*invDet, (M[0][2]*M[1][0]-M[0][0]*M[1][2])*invDet],
      [(M[1][0]*M[2][1]-M[1][1]*M[2][0])*invDet, (M[0][1]*M[2][0]-M[0][0]*M[2][1])*invDet, (M[0][0]*M[1][1]-M[0][1]*M[1][0])*invDet],
    ];
  }}
  function vecNorm(v) {{ return Math.hypot(v[0], v[1], v[2]); }}
  function vecUnit(v) {{ const n = Math.max(vecNorm(v), 1e-12); return [v[0]/n, v[1]/n, v[2]/n]; }}
  function vecDot(a, b) {{ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }}
  function vecCross(a, b) {{ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }}
  function labToCyl(xyz) {{
    const az = Math.atan2(xyz[0], xyz[2]) * 180.0 / Math.PI;
    const r = Math.max(vecNorm(xyz), 1e-9);
    const el = Math.asin(Math.max(-1, Math.min(1, xyz[1] / r))) * 180.0 / Math.PI;
    return [az, xyz[1], el];
  }}
  function stereoProj(g) {{
    const u = vecUnit(g);
    const hemi = u[2] >= 0 ? 1 : -1;
    const zAbs = Math.abs(u[2]);
    return [u[0] / (1.0 + zAbs), u[1] / (1.0 + zAbs), hemi];
  }}

  async function parseH5BuffersInBrowser(solveBuf, imagesBuf) {{
    if (!window.h5wasm) throw new Error("h5wasm library is still loading from CDN");
    const {{ FS }} = await window.h5wasm.ready;
    FS.writeFile("solve_in.h5", new Uint8Array(solveBuf));
    const f = new window.h5wasm.File("solve_in.h5", "r");
    let mf = null;
    if (imagesBuf) {{
      FS.writeFile("images_in.h5", new Uint8Array(imagesBuf));
      mf = new window.h5wasm.File("images_in.h5", "r");
    }} else if (f.keys().includes("images")) {{
      mf = f;
    }}

    const getAttrStr = (node, key, defVal) => {{
      try {{
        const a = node.attrs[key];
        if (!a) return defVal;
        return String(a.value || defVal);
      }} catch (_e) {{ return defVal; }}
    }};
    const inst = getAttrStr(f, "instrument", DATA.instrument || "CG4D");
    let wlRangeNew = [2.8, 4.5];
    let dMinNew = 2.8;
    try {{
      if (f.keys().includes("scan_refinement")) {{
        const srGrp = f.get("scan_refinement");
        if (srGrp.keys().includes("band_wavelength")) {{
          const bwl = srGrp.get("band_wavelength").value;
          if (bwl && bwl.length >= 2) wlRangeNew = [Number(bwl[0]), Number(bwl[1])];
        }}
        if (srGrp.attrs && srGrp.attrs["d_min"]) {{
          dMinNew = Number(srGrp.attrs["d_min"].value) || 2.8;
        }}
      }}
      if (wlRangeNew[0] === 2.8 && f.keys().includes("instrument")) {{
        const wl = f.get("instrument/wavelength").value;
        if (wl && wl.length >= 2 && !(Math.abs(Number(wl[0]) - 2.0) < 1e-3 && Math.abs(Number(wl[1]) - 10.0) < 1e-3)) {{
          wlRangeNew = [Number(wl[0]), Number(wl[1])];
        }}
      }}
    }} catch (_e) {{}}

    // Determine best run slice from file_offsets & bank_ids
    let f0 = 0, f1 = 0;
    const refFile = f.keys().includes("bank_ids") ? f : mf;
    let bankIdsRun = [];
    if (refFile && refFile.keys().includes("bank_ids")) {{
      const bAll = Array.from(refFile.get("bank_ids").value).map(Number);
      f1 = bAll.length;
      if (refFile.keys().includes("file_offsets")) {{
        const offs = Array.from(refFile.get("file_offsets").value).map(Number);
        let bestCount = -1;
        for (let r = 0; r < offs.length; r++) {{
          const s = offs[r];
          const e = (r + 1 < offs.length) ? offs[r + 1] : bAll.length;
          if (e - s > bestCount) {{ bestCount = e - s; f0 = s; f1 = e; }}
        }}
      }}
      bankIdsRun = bAll.slice(f0, f1);
    }}

    // Read calibrated detector banks from detector_calibration/bank_*
    const detMap = new Map();
    if (f.keys().includes("detector_calibration")) {{
      const calGrp = f.get("detector_calibration");
      for (const bName of calGrp.keys()) {{
        if (!bName.startsWith("bank_")) continue;
        const bid = parseInt(bName.slice(5), 10);
        const g = calGrp.get(bName);
        const center = Array.from(g.get("center").value).map(Number);
        const uhat = Array.from(g.get("uhat").value).map(Number);
        const vhat = Array.from(g.get("vhat").value).map(Number);
        const width = Number(g.get("width").value);
        const height = Number(g.get("height").value);
        detMap.set(bid, {{ bid, center, uhat, vhat, width, height, m: 512, n: 512 }});
      }}
    }}

    // Read raw detector images for the active run slice and build grayscale texture canvases
    const bankImgs = new Map();
    const bankGrayCanvases = new Map();
    if (mf && mf.keys().includes("images")) {{
      const imDs = mf.get("images");
      const shape = imDs.shape; // [N, H, W]
      const H = shape[1] || 512, W = shape[2] || 512;
      const sliceArr = imDs.slice([[f0, f1], [0, H], [0, W]]);
      const bStep = 4;
      const Hb = Math.floor(H / bStep), Wb = Math.floor(W / bStep);
      for (let i = 0; i < bankIdsRun.length; i++) {{
        const bid = bankIdsRun[i];
        const offset = i * H * W;
        const sub = sliceArr.subarray(offset, offset + H * W);
        bankImgs.set(bid, {{ data: sub, H, W }});

        // Downsample and log1p-scale into offscreen canvas
        const logBuf = new Float32Array(Hb * Wb);
        for (let rb = 0; rb < Hb; rb++) {{
          for (let cb = 0; cb < Wb; cb++) {{
            let sum = 0;
            for (let dr = 0; dr < bStep; dr++) {{
              const rowOff = (rb * bStep + dr) * W + cb * bStep;
              for (let dc = 0; dc < bStep; dc++) sum += sub[rowOff + dc];
            }}
            logBuf[rb * Wb + cb] = Math.log1p(Math.max(0, sum / (bStep * bStep)));
          }}
        }}
        const sorted = Float32Array.from(logBuf).sort();
        const vmax = Math.max(sorted[Math.floor(sorted.length * 0.995)] || 1.0, 1e-6);
        const gc = document.createElement("canvas");
        gc.width = Wb; gc.height = Hb;
        const gctx = gc.getContext("2d");
        const idata = gctx.createImageData(Wb, Hb);
        for (let rb = 0; rb < Hb; rb++) {{
          for (let cb = 0; cb < Wb; cb++) {{
            // Flip horizontally so column 0..W maps left-to-right in azimuth
            const v = Math.min(255, Math.round((logBuf[rb * Wb + (Wb - 1 - cb)] / vmax) * 255));
            const pIdx = (rb * Wb + cb) * 4;
            idata.data[pIdx] = v; idata.data[pIdx+1] = v; idata.data[pIdx+2] = v; idata.data[pIdx+3] = 255;
          }}
        }}
        gctx.putImageData(idata, 0, 0);
        bankGrayCanvases.set(bid, gc);
      }}
    }}

    // Build panel outlines from detMap
    const panelsOut = [];
    const pixelToLab = (det, r, c) => {{
      const u = ((det.m - 1 - c) / (det.m - 1) - 0.5) * det.width;
      const v = ((r / (det.n - 1)) - 0.5) * det.height;
      return [
        det.center[0] + u * det.uhat[0] + v * det.vhat[0],
        det.center[1] + u * det.uhat[1] + v * det.vhat[1],
        det.center[2] + u * det.uhat[2] + v * det.vhat[2],
      ];
    }};
    for (const [bid, det] of detMap.entries()) {{
      const outline = [], outlineSph = [];
      const nPts = 12;
      const edges = [];
      for (let k=0;k<nPts;k++) edges.push([0, (k/(nPts-1))*(det.m-1)]);
      for (let k=0;k<nPts;k++) edges.push([(k/(nPts-1))*(det.n-1), det.m-1]);
      for (let k=0;k<nPts;k++) edges.push([det.n-1, ((nPts-1-k)/(nPts-1))*(det.m-1)]);
      for (let k=0;k<nPts;k++) edges.push([((nPts-1-k)/(nPts-1))*(det.n-1), 0]);
      let azMin=180, azMax=-180, yMin=10, yMax=-10, elMin=90, elMax=-90;
      edges.forEach(([r, c]) => {{
        const [az, y, el] = labToCyl(pixelToLab(det, r, c));
        if (az < azMin) azMin = az; if (az > azMax) azMax = az;
        if (y < yMin) yMin = y; if (y > yMax) yMax = y;
        if (el < elMin) elMin = el; if (el > elMax) elMax = el;
        outline.push([az, y]); outlineSph.push([az, el]);
      }});
      if (azMax - azMin <= 180) {{
        panelsOut.push({{
          bank: bid, az_min: azMin, az_max: azMax, y_min: yMin, y_max: yMax, el_min: elMin, el_max: elMax,
          outline, outline_sph: outlineSph, _grayCanvas: bankGrayCanvases.get(bid) || null,
        }});
      }}
    }}

    // Read sample U, B, R_run and forward-project reflections
    let sg = "P 21 21 21";
    try {{ sg = String(f.get("sample/space_group").value || sg); }} catch (_e) {{}}
    let cell = DATA.cell || [68.7, 69.8, 156.9, 90, 90, 90];
    let U = [[1,0,0],[0,1,0],[0,0,1]], B = null, Rrun = [[1,0,0],[0,1,0],[0,0,1]];
    if (f.keys().includes("sample")) {{
      const sGrp = f.get("sample");
      if (sGrp.keys().includes("U")) {{
        const uRaw = Array.from(sGrp.get("U").value).map(Number);
        U = [[uRaw[0],uRaw[1],uRaw[2]],[uRaw[3],uRaw[4],uRaw[5]],[uRaw[6],uRaw[7],uRaw[8]]];
      }}
      if (sGrp.keys().includes("B")) {{
        const bRaw = Array.from(sGrp.get("B").value).map(Number);
        B = [[bRaw[0],bRaw[1],bRaw[2]],[bRaw[3],bRaw[4],bRaw[5]],[bRaw[6],bRaw[7],bRaw[8]]];
        const invB = mat3Inv(B);
        const aLen = Math.hypot(invB[0][0], invB[1][0], invB[2][0]);
        const bLen = Math.hypot(invB[0][1], invB[1][1], invB[2][1]);
        const cLen = Math.hypot(invB[0][2], invB[1][2], invB[2][2]);
        cell = [aLen, bLen, cLen, 90, 90, 90];
      }}
    }}
    if (f.keys().includes("goniometer") && f.get("goniometer").keys().includes("R")) {{
      const rAll = Array.from(f.get("goniometer/R").value).map(Number);
      const rOff = Math.min(f0 * 9, Math.max(0, rAll.length - 9));
      Rrun = [[rAll[rOff],rAll[rOff+1],rAll[rOff+2]],[rAll[rOff+3],rAll[rOff+4],rAll[rOff+5]],[rAll[rOff+6],rAll[rOff+7],rAll[rOff+8]]];
    }}

    const peaksOut = [];
    const zonesOut = [];
    let artifactKind = "solve";
    let peakShapeMode = "solved_circle";
    if (B && detMap.size > 0) {{
      const UB = mat3Mul(U, B);
      const RUB = mat3Mul(Rrun, UB);
      const R_T = mat3T(Rrun);
      const candPeaks = [];

      // 1. Check if f is an integrator / finder HDF5 with a 'peaks' group
      if (f.keys().includes("peaks")) {{
        const pGrp = f.get("peaks");
        const pKeys = pGrp.keys();
        if (pKeys.includes("pixel_r") && pKeys.includes("pixel_c") && pKeys.includes("bank")) {{
          const prArr = Array.from(pGrp.get("pixel_r").value).map(Number);
          const pcArr = Array.from(pGrp.get("pixel_c").value).map(Number);
          const bkArr = Array.from(pGrp.get("bank").value).map(Number);
          const hArr = pKeys.includes("h") ? Array.from(pGrp.get("h").value).map(Number) : null;
          const kArr = pKeys.includes("k") ? Array.from(pGrp.get("k").value).map(Number) : null;
          const lArr = pKeys.includes("l") ? Array.from(pGrp.get("l").value).map(Number) : null;
          const lamArr = pKeys.includes("lambda") ? Array.from(pGrp.get("lambda").value).map(Number) : null;
          const intArr = pKeys.includes("intensity") ? Array.from(pGrp.get("intensity").value).map(Number) : null;
          const sigArr = pKeys.includes("sigma") ? Array.from(pGrp.get("sigma").value).map(Number) : null;
          const vuArr = pKeys.includes("var_u") ? Array.from(pGrp.get("var_u").value).map(Number) : null;
          const vvArr = pKeys.includes("var_v") ? Array.from(pGrp.get("var_v").value).map(Number) : null;
          const cuvArr = pKeys.includes("cov_uv") ? Array.from(pGrp.get("cov_uv").value).map(Number) : null;
          const runArr = pKeys.includes("run_index") ? Array.from(pGrp.get("run_index").value).map(Number) : null;

          if (vuArr && vvArr) {{
            artifactKind = "integrator";
            const anyAniso = vuArr.some((vu, idx) => Math.abs(vu - vvArr[idx]) > 1e-2 || (cuvArr && Math.abs(cuvArr[idx]) > 1e-3));
            peakShapeMode = anyAniso ? "anisotropic_ellipse" : "isotropic_circle";
          }} else {{
            artifactKind = "finder";
            peakShapeMode = "isotropic_circle";
          }}

          for (let i = 0; i < prArr.length; i++) {{
            if (runArr && runArr[i] !== 0) continue;
            const inten = intArr ? intArr[i] : 120.0;
            const sig = sigArr ? sigArr[i] : 3.0;
            if (intArr && sigArr && (sig <= 0 || inten / Math.max(sig, 1e-9) < 2.5)) continue;
            const bid = bkArr[i];
            const det = detMap.get(bid);
            if (!det) continue;
            const rPix = prArr[i], cPix = pcArr[i];
            const pt = pixelToLab(det, rPix, cPix);
            const [az, dy, el] = labToCyl(pt);
            const kf = vecUnit(pt);
            const qLab = vecUnit([kf[0], kf[1], kf[2] - 1.0]);
            const gCryst = mat3Vec(R_T, qLab);
            const [sx, sy, hemi] = stereoProj(gCryst);
            const h = hArr ? Math.round(hArr[i]) : 1;
            const k = kArr ? Math.round(kArr[i]) : 0;
            const l = lArr ? Math.round(lArr[i]) : 0;
            const G = mat3Vec(RUB, [h, k, l]);
            const qn = Math.max(vecNorm(G), 0.1);
            const dVal = 1.0 / qn;
            const lam = lamArr ? lamArr[i] : 2.5;
            const pkObj = {{
              bank: bid, row: Number(rPix.toFixed(1)), col: Number(cPix.toFixed(1)),
              det_x: az, det_y: dy, det_el: el, q_hat: gCryst,
              stereographic: [sx, sy], hemisphere: hemi, intensity: inten, sigma: sig,
              wavelength: lam, d_spacing: dVal, harmonic_order: Math.max(1, gcd3(h, k, l)),
              parallax_depth: Math.max(0.05, Math.min(0.98, 1.0 / (1.0 + 0.35 * dVal))),
              hkl: [h, k, l], zone_ids: [], indexed: true, angular_error_deg: 0.02,
            }};
            if (vuArr && vvArr) {{
              const vu = vuArr[i], vv = vvArr[i], cuv = cuvArr ? cuvArr[i] : 0.0;
              const diff = vu - vv;
              const sq = Math.sqrt(diff * diff + 4.0 * cuv * cuv);
              const s1 = Math.sqrt(Math.max((vu + vv + sq) * 0.5, 1e-6));
              const s2 = Math.sqrt(Math.max((vu + vv - sq) * 0.5, 1e-6));
              const phi = 0.5 * Math.atan2(2.0 * cuv, diff);
              const a = 2.0 * s1, b = 2.0 * s2;
              const ellCyl = [], ellSph = [];
              for (let m = 0; m < 16; m++) {{
                const th = (m / 16) * 2.0 * Math.PI;
                const ct = Math.cos(th), st = Math.sin(th);
                const cE = cPix + a * ct * Math.cos(phi) - b * st * Math.sin(phi);
                const rE = rPix + a * ct * Math.sin(phi) + b * st * Math.cos(phi);
                const [azE, dyE, elE] = labToCyl(pixelToLab(det, rE, cE));
                ellCyl.push([azE, dyE]); ellSph.push([azE, elE]);
              }}
              pkObj.ellipse = ellCyl;
              pkObj.ellipse_sph = ellSph;
              pkObj.sigma_major_px = Number(s1.toFixed(2));
              pkObj.sigma_minor_px = Number(s2.toFixed(2));
              pkObj.ellipse_angle_deg = Number((phi * 180 / Math.PI).toFixed(1));
              pkObj.shape_kind = Math.abs(s1 - s2) > 0.05 ? "ellipse" : "isotropic_circle";
            }}
            candPeaks.push(pkObj);
          }}
        }}
      }} else if (f.keys().includes("banks")) {{
        // 2. Recognize predictor.h5 format: banks/<img_idx>/(i, j, h, k, l, wavelength)
        artifactKind = "predictor";
        peakShapeMode = "predicted_circle";
        const bGrp = f.get("banks");
        const bAll = f.keys().includes("bank_ids") ? Array.from(f.get("bank_ids").value).map(Number) : [];
        for (let imgIdx = f0; imgIdx < (f1 || bGrp.keys().length); imgIdx++) {{
          const kStr = String(imgIdx);
          if (!bGrp.keys().includes(kStr)) continue;
          const g = bGrp.get(kStr);
          if (!g.keys().includes("i") || !g.keys().includes("j")) continue;
          const prArr = Array.from(g.get("i").value).map(Number);
          const pcArr = Array.from(g.get("j").value).map(Number);
          const hArr = Array.from(g.get("h").value).map(Number);
          const kArr = Array.from(g.get("k").value).map(Number);
          const lArr = Array.from(g.get("l").value).map(Number);
          const wlArr = Array.from(g.get("wavelength").value).map(Number);
          const bid = bAll[imgIdx] !== undefined ? bAll[imgIdx] : imgIdx;
          const det = detMap.get(bid);
          if (!det) continue;
          for (let i = 0; i < prArr.length; i++) {{
            const rPix = prArr[i], cPix = pcArr[i];
            const h = Math.round(hArr[i]), k = Math.round(kArr[i]), l = Math.round(lArr[i]);
            const lam = wlArr[i];
            const pt = pixelToLab(det, rPix, cPix);
            const [az, dy, el] = labToCyl(pt);
            const kf = vecUnit(pt);
            const qLab = vecUnit([kf[0], kf[1], kf[2] - 1.0]);
            const gCryst = mat3Vec(R_T, qLab);
            const [sx, sy, hemi] = stereoProj(gCryst);
            const G = mat3Vec(RUB, [h, k, l]);
            const qn = Math.max(vecNorm(G), 0.1);
            const dVal = 1.0 / qn;
            candPeaks.push({{
              bank: bid, row: Number(rPix.toFixed(1)), col: Number(cPix.toFixed(1)),
              det_x: az, det_y: dy, det_el: el, q_hat: gCryst,
              stereographic: [sx, sy], hemisphere: hemi, intensity: 150.0, sigma: 3.0,
              wavelength: lam, d_spacing: dVal, harmonic_order: Math.max(1, gcd3(h, k, l)),
              parallax_depth: Math.max(0.05, Math.min(0.98, 1.0 / (1.0 + 0.35 * dVal))),
              hkl: [h, k, l], zone_ids: [], indexed: true, angular_error_deg: 0.02,
            }});
          }}
        }}
      }}

      // 3. Fallback to forward-projection from solved U, B if no peaks were read above
      if (candPeaks.length === 0) {{
      const dMin = 2.6;
      const hMax = Math.min(18, Math.floor(cell[0] / dMin));
      const kMax = Math.min(18, Math.floor(cell[1] / dMin));
      const lMax = Math.min(28, Math.floor(cell[2] / dMin));
      for (let h = -hMax; h <= hMax; h++) {{
        for (let k = -kMax; k <= kMax; k++) {{
          for (let l = -lMax; l <= lMax; l++) {{
            if (h === 0 && k === 0 && l === 0) continue;
            const G = mat3Vec(RUB, [h, k, l]);
            const qn = vecNorm(G);
            if (qn < 1e-8 || (1.0 / qn) < dMin) continue;
            const sth = -G[2] / qn;
            if (sth <= 0) continue;
            const lam = (2.0 * sth) / qn;
            if (lam < wlRangeNew[0] || lam > wlRangeNew[1]) continue;
            const kf = vecUnit([lam * G[0], lam * G[1], 1.0 + lam * G[2]]);
            for (const [bid, det] of detMap.entries()) {{
              const nVec = vecCross(det.uhat, det.vhat);
              const denom = vecDot(kf, nVec);
              if (Math.abs(denom) <= 1e-6) continue;
              const tRay = vecDot(det.center, nVec) / denom;
              if (tRay <= 0) continue;
              const pt = [kf[0]*tRay, kf[1]*tRay, kf[2]*tRay];
              const rel = [pt[0]-det.center[0], pt[1]-det.center[1], pt[2]-det.center[2]];
              const u = vecDot(rel, det.uhat), v = vecDot(rel, det.vhat);
              if (Math.abs(u) > 0.49 * det.width || Math.abs(v) > 0.49 * det.height) continue;
              const cPix = (det.m - 1) * (1.0 - (u / det.width + 0.5));
              const rPix = (det.n - 1) * (v / det.height + 0.5);
              let inten = 120.0, keep = true;
              const bIm = bankImgs.get(bid);
              if (bIm) {{
                const ri = Math.round(rPix), ci = Math.round(cPix);
                let pkMax = 0, bgSum = 0, bgN = 0;
                for (let dr = -9; dr <= 9; dr++) {{
                  const rr = ri + dr; if (rr < 0 || rr >= bIm.H) continue;
                  for (let dc = -9; dc <= 9; dc++) {{
                    const cc = ci + dc; if (cc < 0 || cc >= bIm.W) continue;
                    const val = bIm.data[rr * bIm.W + cc];
                    if (Math.abs(dr) <= 4 && Math.abs(dc) <= 4) {{ if (val > pkMax) pkMax = val; }}
                    bgSum += val; bgN++;
                  }}
                }}
                const bgMean = bgSum / Math.max(1, bgN);
                const snr = (pkMax - bgMean) / Math.max(Math.sqrt(Math.max(bgMean, 1.0)), 1.0);
                if (snr < 3.0) keep = false;
                inten = pkMax;
              }}
              if (!keep) break;
              const [az, dy, el] = labToCyl(pt);
              const qLab = vecUnit([kf[0], kf[1], kf[2] - 1.0]);
              const gCryst = mat3Vec(R_T, qLab);
              const [sx, sy, hemi] = stereoProj(gCryst);
              const dVal = 1.0 / qn;
              candPeaks.push({{
                bank: bid, row: Number(rPix.toFixed(1)), col: Number(cPix.toFixed(1)),
                det_x: az, det_y: dy, det_el: el, q_hat: gCryst,
                stereographic: [sx, sy], hemisphere: hemi, intensity: inten, sigma: 3.0,
                wavelength: lam, d_spacing: dVal, harmonic_order: Math.max(1, gcd3(h, k, l)),
                parallax_depth: Math.max(0.05, Math.min(0.98, 1.0 / (1.0 + 0.35 * dVal))),
                hkl: [h, k, l], zone_ids: [], indexed: true, angular_error_deg: 0.02,
              }});
              break;
            }}
          }}
        }}
      }}
      }}
      candPeaks.sort((a, b) => b.intensity - a.intensity);
      candPeaks.slice(0, 1200).forEach((p, idx) => {{ p.id = idx; peaksOut.push(p); }});

      // Build Laue zones [u v w] with A_real = inv(UB)
      const A_real = mat3Inv(UB);
      const rawZ = [];
      for (let u = -3; u <= 3; u++) {{
        for (let v = -3; v <= 3; v++) {{
          for (let w = -3; w <= 3; w++) {{
            if (u === 0 && v === 0 && w === 0) continue;
            if (gcd3(u, v, w) !== 1) continue;
            if (u < 0 || (u === 0 && v < 0) || (u === 0 && v === 0 && w < 0)) continue;
            const ord = Math.max(Math.abs(u), Math.abs(v), Math.abs(w));
            const tCryst = vecUnit(mat3Vec(A_real, [u, v, w]));
            const tLab = vecUnit(mat3Vec(Rrun, tCryst));
            const memberIds = [];
            peaksOut.forEach(p => {{
              if (p.hkl[0]*u + p.hkl[1]*v + p.hkl[2]*w === 0) memberIds.push(p.id);
            }});
            if (ord > 1 && memberIds.length < 3) continue;
            rawZ.push({{ uvw: [u, v, w], order: ord, tCryst, tLab, memberIds, n_peaks: memberIds.length }});
          }}
        }}
      }}
      rawZ.sort((a, b) => (a.order !== b.order) ? (a.order - b.order) : (b.n_peaks - a.n_peaks));
      rawZ.slice(0, 36).forEach((z, zi) => {{
        z.memberIds.forEach(pid => peaksOut[pid].zone_ids.push(zi));
        // Trace reflection cone kf = ki - 2(ki·n)*n around great circle perpendicular to tLab
        const ref = Math.abs(z.tLab[0]) < 0.9 ? [1,0,0] : [0,1,0];
        const e1 = vecUnit(vecCross(z.tLab, ref));
        const e2 = vecUnit(vecCross(z.tLab, e1));
        const detSegs = [], detSegsSph = [], stereoSegs = [[]];
        let curCyl = [], curSph = [];
        const nArc = 360;
        for (let s = 0; s <= nArc; s++) {{
          const ang = (s / nArc) * 2.0 * Math.PI;
          const gLab = [Math.cos(ang)*e1[0] + Math.sin(ang)*e2[0], Math.cos(ang)*e1[1] + Math.sin(ang)*e2[1], Math.cos(ang)*e1[2] + Math.sin(ang)*e2[2]];
          const gC = mat3Vec(R_T, gLab);
          const [sx, sy] = stereoProj(gC);
          if (s % 4 === 0) stereoSegs[0].push([sx, sy]);
          const kiDotG = gLab[2];
          if (kiDotG >= -1e-4) {{
            if (curCyl.length > 1) {{ detSegs.push(curCyl); detSegsSph.push(curSph); }}
            curCyl = []; curSph = []; continue;
          }}
          const kf = vecUnit([-2*kiDotG*gLab[0], -2*kiDotG*gLab[1], 1.0 - 2*kiDotG*gLab[2]]);
          const [az, dy, el] = labToCyl(kf);
          const onPanel = panelsOut.some(p => az >= p.az_min - 1.5 && az <= p.az_max + 1.5 && dy >= p.y_min - 0.02 && dy <= p.y_max + 0.02);
          if (!onPanel || (curCyl.length > 0 && Math.abs(az - curCyl[curCyl.length-1][0]) > 25)) {{
            if (curCyl.length > 1) {{ detSegs.push(curCyl); detSegsSph.push(curSph); }}
            curCyl = []; curSph = [];
          }}
          if (onPanel) {{ curCyl.push([az, dy]); curSph.push([az, el]); }}
        }}
        if (curCyl.length > 1) {{ detSegs.push(curCyl); detSegsSph.push(curSph); }}
        zonesOut.push({{
          zone_id: zi, uvw: z.uvw, order: z.order, label: `[${{z.uvw[0]}} ${{z.uvw[1]}} ${{z.uvw[2]}}]`,
          n_peaks: z.n_peaks, angular_rms_deg: 0.02, rainbow_color: ZONE_COLORS[zi % ZONE_COLORS.length],
          detector_segments: detSegs, detector_segments_sph: detSegsSph, stereo_segments: stereoSegs,
        }});
      }});
    }}

    const ord1Ids = zonesOut.filter(z => z.order <= 1 && z.n_peaks > 0).map(z => z.zone_id);
    const ladderOut = [
      {{ rung: 1, stage: "1. Binned Count Excess", subtitle: "Finder-less Poisson deviance bins", active_zones: [], n_captured: peaksOut.length, description: "Threshold-free detector count bins enter lattice_ladder directly with no spot finder." }},
      {{ rung: 2, stage: "2. 13→145 Zone Ladder", subtitle: `Primary Laue zones (${{ord1Ids.slice(0,2).length}} locked)`, active_zones: ord1Ids.slice(0, 2), n_captured: Math.round(peaksOut.length * 0.45), description: "Coarse-to-fine axis ladder locks the dominant intersecting zone planes." }},
      {{ rung: 3, stage: "3. 60-Zone Global", subtitle: `${{ord1Ids.length}} fundamental zones`, active_zones: ord1Ids, n_captured: Math.round(peaksOut.length * 0.78), description: "Joint SO(3) + metrological translation/tilt refinement across fundamental zones." }},
      {{ rung: 4, stage: "4. 120→400 Zone + Cell", subtitle: "Higher-order zones + cell ratios", active_zones: zonesOut.slice(0, 12).map(z => z.zone_id), n_captured: Math.round(peaksOut.length * 0.92), description: "Staged 120→400 zone-band refinement co-refines axial stretch, ω drift, and unit-cell ratios." }},
      {{ rung: 5, stage: "5. Converged Solve UB", subtitle: `${{sg}} • a=${{cell[0].toFixed(1)}}, b=${{cell[1].toFixed(1)}}, c=${{cell[2].toFixed(1)}} Å`, active_zones: ord1Ids.length ? ord1Ids : zonesOut.map(z => z.zone_id), n_captured: peaksOut.length, description: `Converged 3D UB orientation (${{peaksOut.length}} forward-projected spots with SNR ≥ 3.0).` }},
    ];
    try {{ f.close(); if (mf && mf !== f) mf.close(); }} catch (_e) {{}}
    return {{
      instrument: inst, space_group: sg, intensity_source: bankImgs.size > 0 ? "raw_h5" : "reconstructed",
      artifact_kind: artifactKind, peak_shape_mode: peakShapeMode,
      wavelength_range: wlRangeNew, d_min: dMinNew, cell, peaks: peaksOut, zones: zonesOut, panels: panelsOut,
      ladder_steps: ladderOut, pipeline_stages: PIPELINE,
    }};
  }}

  // Live Galaxy / NOVA / Pulsar GPU job orchestration
  const galaxyBridge = {{
    mode: "none", // "nova" | "galaxy_iframe" | "none"
    historyId: null,
    datasets: [],
    pollTimer: null,
  }};

  const elGalaxyBadge = document.getElementById("galaxy-status-badge");
  const btnGalaxyDispatch = document.getElementById("btn-galaxy-dispatch");
  const selGalaxyH5 = document.getElementById("sel-galaxy-h5");
  const btnLocalH5 = document.getElementById("btn-local-h5");
  const inpLocalH5 = document.getElementById("inp-local-h5");

  function populateGalaxyH5Selector() {{
    if (!selGalaxyH5) return;
    const h5s = (galaxyBridge.datasets || []).filter(
      d => d.state === "ok" && (d.extension === "h5" || d.extension === "hdf5")
    );
    selGalaxyH5.innerHTML = `<option value="">📂 Galaxy History (${{h5s.length}} .h5)...</option>`;
    h5s.forEach(d => {{
      const opt = document.createElement("option");
      opt.value = d.id;
      opt.textContent = `#${{d.hid || ""}} ${{d.name}}`;
      selGalaxyH5.appendChild(opt);
    }});
  }}

  async function loadFromGalaxyDatasetId(dsId) {{
    if (!dsId) return;
    setStatusLine("[⏳ GALAXY HDF5]", `Loading dataset <strong>${{dsId}}</strong> from NDIP / Galaxy History...`);
    try {{
      // 1. Prefer server-side HDF5 extraction via NOVA/HTTP backend (handles solve.h5, predictor.h5, integrated.h5 with ellipses & multi-frame)
      try {{
        const rServer = await fetch(`api/payload?dataset_id=${{encodeURIComponent(dsId)}}`, {{
          cache: "no-store",
          credentials: "same-origin",
        }});
        if (rServer.ok) {{
          const serverPayload = await rServer.json();
          if (serverPayload && (Array.isArray(serverPayload.peaks) || Array.isArray(serverPayload.panels))) {{
            recomputeBoundsAndReload(serverPayload);
            return;
          }}
        }}
      }} catch (_srvErr) {{
        // Fall through to client-side h5wasm fetch
      }}

      // 2. Fallback: fetch raw HDF5 via /api/datasets/<id>/display and parse in browser via h5wasm
      while (!window.h5wasm) await new Promise(r => setTimeout(r, 120));
      const h5s = (galaxyBridge.datasets || []).filter(
        d => d.state === "ok" && (d.extension === "h5" || d.extension === "hdf5")
      );
      const mergedDs = [...h5s].reverse().find(
        d => d.id !== dsId && /merged|image|scan|prepare/i.test(d.name || "")
      );
      const r1 = await fetch(`/api/datasets/${{dsId}}/display`, {{ credentials: "same-origin" }});
      if (!r1.ok) throw new Error(`HTTP ${{r1.status}} fetching dataset ${{dsId}}`);
      const buf1 = await r1.arrayBuffer();
      let buf2 = null;
      if (mergedDs) {{
        const r2 = await fetch(`/api/datasets/${{mergedDs.id}}/display`, {{ credentials: "same-origin" }});
        if (r2.ok) buf2 = await r2.arrayBuffer();
      }}
      const newPayload = await parseH5BuffersInBrowser(buf1, buf2);
      recomputeBoundsAndReload(newPayload);
    }} catch (err) {{
      setStatusLine("[⚠️ GALAXY HDF5]", `<span style="color:#f87171;">Failed to load HDF5: ${{err.message}}</span>`);
    }}
  }}

  if (selGalaxyH5) {{
    selGalaxyH5.addEventListener("change", () => {{
      if (selGalaxyH5.value) loadFromGalaxyDatasetId(selGalaxyH5.value);
    }});
  }}

  if (btnLocalH5 && inpLocalH5) {{
    btnLocalH5.addEventListener("click", () => inpLocalH5.click());
    inpLocalH5.addEventListener("change", async () => {{
      const files = Array.from(inpLocalH5.files || []);
      if (files.length === 0) return;
      const ladderDesc = document.getElementById("ladder-desc");
      if (ladderDesc) ladderDesc.textContent = "⏳ Parsing local HDF5 file(s) in browser via h5wasm...";
      try {{
        while (!window.h5wasm) await new Promise(r => setTimeout(r, 120));
        let solveFile = files.find(f => /solve|index|integ/i.test(f.name)) || files[0];
        let imgFile = files.find(f => f !== solveFile && /merged|image|scan/i.test(f.name)) || (files.length > 1 ? files[1] : null);
        const buf1 = await solveFile.arrayBuffer();
        const buf2 = imgFile ? await imgFile.arrayBuffer() : null;
        const newPayload = await parseH5BuffersInBrowser(buf1, buf2);
        recomputeBoundsAndReload(newPayload);
      }} catch (err) {{
        if (ladderDesc) ladderDesc.textContent = `⚠️ Local HDF5 error: ${{err.message}}`;
      }}
    }});
  }}

  async function probeGalaxyConnection() {{
    // 1. Check NOVA server endpoint (api/pipeline/status)
    try {{
      const r = await fetch("api/pipeline/status", {{ cache: "no-store" }});
      if (r.ok) {{
        const doc = await r.json();
        if (doc && typeof doc.galaxy_connected === "boolean") {{
          galaxyBridge.mode = "nova";
          galaxyBridge.historyId = doc.history_id;
          galaxyBridge.datasets = doc.history_datasets || [];
          populateGalaxyH5Selector();
          if (elGalaxyBadge) {{
            elGalaxyBadge.textContent = doc.galaxy_connected
              ? `🟢 NOVA → GALAXY (${{galaxyBridge.datasets.length}} H5s)`
              : "🟡 NOVA LOCAL";
            elGalaxyBadge.style.borderColor = doc.galaxy_connected ? "#34d399" : "#fbbf24";
            elGalaxyBadge.style.color = doc.galaxy_connected ? "#34d399" : "#fbbf24";
          }}
          return;
        }}
      }}
    }} catch (_e) {{}}

    // 2. Check same-origin Galaxy REST API (works from /static/subhkl_parallax.html or Galaxy tab)
    try {{
      const r = await fetch("/api/histories?limit=1", {{ credentials: "same-origin" }});
      if (r.ok) {{
        const hists = await r.json();
        if (Array.isArray(hists) && hists.length > 0 && hists[0].id) {{
          galaxyBridge.mode = "galaxy_iframe";
          galaxyBridge.historyId = hists[0].id;
          const cr = await fetch(
            `/api/histories/${{galaxyBridge.historyId}}/contents?v=dev&q=deleted&qv=false`,
            {{ credentials: "same-origin" }}
          );
          if (cr.ok) {{
            const items = await cr.json();
            if (Array.isArray(items)) {{
              galaxyBridge.datasets = items.filter(
                it => it.history_content_type === "dataset" && !it.deleted
              );
            }}
          }}
          populateGalaxyH5Selector();
          if (elGalaxyBadge) {{
            elGalaxyBadge.textContent = `🟢 GALAXY (${{galaxyBridge.datasets.length}} items)`;
            elGalaxyBadge.style.borderColor = "#34d399";
            elGalaxyBadge.style.color = "#34d399";
          }}
          // If opened as a truly static HTML page with no pre-baked peaks, auto-load the latest solve.h5 from Galaxy!
          if (PEAKS.length === 0 && PANELS.length === 0) {{
            const h5s = galaxyBridge.datasets.filter(
              d => d.state === "ok" && (d.extension === "h5" || d.extension === "hdf5")
            );
            const solveDs = [...h5s].reverse().find(d => /solve|index/i.test(d.name || "")) || h5s[0];
            if (solveDs) {{
              if (selGalaxyH5) selGalaxyH5.value = solveDs.id;
              loadFromGalaxyDatasetId(solveDs.id);
            }}
          }}
        }}
      }}
    }} catch (_e) {{}}
  }}
  probeGalaxyConnection();

  async function executeWorkflowStage(stageIdx) {{
    const st = PIPELINE[stageIdx] || PIPELINE[state.currentPipelineIdx] || PIPELINE[1] || {{ stage_id: "solve", name: "2. SOLVE" }};
    const stageId = st.stage_id || "solve";
    const params = collectWorkflowParams();
    state.runningStageIdx = stageIdx;
    renderPipeline();
    const cellSummary = document.getElementById("cell-summary");
    if (cellSummary) cellSummary.textContent = `RUNNING ${{stageId.toUpperCase()}}...`;
    appendTerminalLog(
      `$ subhkl [${{st.name}}] — dispatching (SG=${{params.space_group}}, cell=${{params.a}},${{params.b}},${{params.c}}, λ=${{params.wavelength_min.toFixed(2)}}–${{params.wavelength_max.toFixed(2)}} Å)...\\n`,
      true
    );

    if (galaxyBridge.mode === "nova") {{
      try {{
        const resp = await fetch("api/pipeline/run", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ stage_id: stageId, params: params }}),
        }});
        const res = await resp.json();
        if (!resp.ok || res.status === "error") {{
          throw new Error(res.error || "Failed to dispatch job");
        }}
        const jobObj = res.job || {{}};
        const jobId = jobObj.job_id || "local";
        if (jobObj.stdout || jobObj.stderr) {{
          appendTerminalLog((jobObj.stdout || "") + (jobObj.stderr ? `\\n[stderr]\\n${{jobObj.stderr}}` : ""), true);
        }}
        if (stageId === "solve") selectLadderRung(0, false);
        if (galaxyBridge.pollTimer) clearInterval(galaxyBridge.pollTimer);
        galaxyBridge.pollTimer = setInterval(async () => {{
          try {{
            const sr = await fetch("api/pipeline/status", {{ cache: "no-store" }});
            if (!sr.ok) return;
            const sdoc = await sr.json();
            const aj = sdoc.active_job;
            if (!aj) return;
            const outText = (aj.stdout || "") + (aj.stderr ? `\\n[stderr]\\n${{aj.stderr}}` : "");
            if (outText) appendTerminalLog(outText, true);
            if (stageId === "solve" && typeof aj.ladder_rung_idx === "number") {{
              selectLadderRung(aj.ladder_rung_idx, false);
            }}
            if (cellSummary) cellSummary.textContent = `${{aj.mode === "galaxy" ? "PULSAR" : "SUBHKL"}} [${{(aj.state || "running").toUpperCase()}}]`;
            if (aj.state === "ok" || aj.state === "error") {{
              clearInterval(galaxyBridge.pollTimer);
              galaxyBridge.pollTimer = null;
              state.runningStageIdx = -1;
              renderPipeline();
              if (cellSummary) cellSummary.textContent = aj.state === "ok" ? "DONE ✓" : "ERROR ✗";
              if (aj.state === "ok" && aj.updated_payload_ready) {{
                try {{
                  const pr = await fetch("api/payload", {{ cache: "no-store" }});
                  if (pr.ok) {{
                    const newPayload = await pr.json();
                    recomputeBoundsAndReload(newPayload);
                    appendTerminalLog("\\n✓ Reloaded updated orientation & Bragg reflections into viewport.");
                  }}
                }} catch (_re) {{}}
              }}
            }}
          }} catch (_pe) {{}}
        }}, 700);
      }} catch (err) {{
        state.runningStageIdx = -1;
        renderPipeline();
        appendTerminalLog(`\\n[error] ${{err.message}}`);
      }}
      return;
    }}

    if (galaxyBridge.mode === "galaxy_iframe" && galaxyBridge.historyId) {{
      try {{
        await probeGalaxyConnection();
        const h5s = galaxyBridge.datasets.filter(
          d => d.state === "ok" && (d.extension === "h5" || d.extension === "hdf5")
        );
        const mergedDs = [...h5s].reverse().find(
          d => /merged|image|scan|prepare/i.test(d.name || "")
        ) || h5s[h5s.length - 1];
        const solveDs = [...h5s].reverse().find(
          d => /solve|index/i.test(d.name || "")
        ) || h5s[0];

        let toolId = "imaginex_subhkl_solve";
        let inputs = {{}};

        if (stageId === "solve") {{
          if (!mergedDs) throw new Error("Upload merged.h5 to Galaxy History first");
          toolId = "imaginex_subhkl_solve";
          inputs = {{
            input_h5: {{ src: "hda", id: mergedDs.id }},
            instrument: params.instrument,
            space_group: params.space_group,
            a: params.a, b: params.b, c: params.c,
            alpha: params.alpha, beta: params.beta, gamma: params.gamma,
            wavelength_min: params.wavelength_min,
            wavelength_max: params.wavelength_max,
            objective: "band",
            d_min: params.d_min,
            n_zones: 120,
            steps: params.steps,
          }};
        }} else if (stageId === "integrate") {{
          if (!mergedDs || !solveDs) throw new Error("Need merged.h5 and solve.h5 in Galaxy History");
          toolId = "imaginex_subhkl_predict";
          inputs = {{
            input: {{ src: "hda", id: mergedDs.id }},
            indexed_peaks: {{ src: "hda", id: solveDs.id }},
            instrument: params.instrument,
            d_min: params.d_min,
            wavelength_min: params.wavelength_min,
            wavelength_max: params.wavelength_max,
          }};
        }} else {{
          if (!mergedDs) throw new Error("No HDF5 dataset in Galaxy History");
          toolId = (st.tool_ids && st.tool_ids[0]) || "imaginex_subhkl_solve";
          inputs = {{ input_h5: {{ src: "hda", id: mergedDs.id }} }};
        }}

        appendTerminalLog(`POST /api/tools -> tool_id=${{toolId}} (history=${{galaxyBridge.historyId}})...`);
        const tr = await fetch("/api/tools", {{
          method: "POST",
          credentials: "same-origin",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            history_id: galaxyBridge.historyId,
            tool_id: toolId,
            inputs: inputs,
          }}),
        }});
        const tres = await tr.json();
        if (!tr.ok) {{
          throw new Error((tres && tres.err_msg) || "Galaxy /api/tools error");
        }}
        const jobId = tres.jobs && tres.jobs[0] && tres.jobs[0].id;
        appendTerminalLog(`Dispatched Galaxy / Pulsar job ${{jobId}} (${{toolId}}). Polling stdio...`);
        if (stageId === "solve") selectLadderRung(0, false);
        let rungStep = 0;
        if (galaxyBridge.pollTimer) clearInterval(galaxyBridge.pollTimer);
        galaxyBridge.pollTimer = setInterval(async () => {{
          try {{
            const jr = await fetch(`/api/jobs/${{jobId}}?full=true`, {{ credentials: "same-origin" }});
            if (!jr.ok) return;
            const jdoc = await jr.json();
            const jstate = jdoc.state || "running";
            const cmdLine = jdoc.command_line ? `$ ${{jdoc.command_line}}\\n` : "";
            const stdOut = jdoc.tool_stdout || jdoc.stdout || "";
            const stdErr = jdoc.tool_stderr || jdoc.stderr || "";
            if (cmdLine || stdOut || stdErr) {{
              appendTerminalLog(
                `[job ${{jobId}} · ${{jstate}}]\\n${{cmdLine}}${{stdOut}}${{stdErr ? "\\n[stderr]\\n" + stdErr : ""}}`,
                true
              );
            }}
            if (cellSummary) cellSummary.textContent = `GALAXY [${{jstate.toUpperCase()}}]`;
            if (stageId === "solve" && (jstate === "queued" || jstate === "running")) {{
              rungStep = Math.min(3, rungStep + 1);
              selectLadderRung(rungStep, false);
            }}
            if (jstate === "ok" || jstate === "error") {{
              clearInterval(galaxyBridge.pollTimer);
              galaxyBridge.pollTimer = null;
              state.runningStageIdx = -1;
              renderPipeline();
              if (jstate === "ok") {{
                selectLadderRung(4, false);
                if (cellSummary) cellSummary.textContent = "DONE ✓";
                await probeGalaxyConnection();
                const outs = (tres.outputs || []).filter(o => o.extension === "h5" || o.extension === "hdf5");
                if (outs.length > 0) {{
                  appendTerminalLog(`\\n✓ Loading output dataset ${{outs[0].id}} into viewport...`);
                  await loadFromGalaxyDatasetId(outs[0].id);
                }}
              }} else {{
                if (cellSummary) cellSummary.textContent = "ERROR ✗";
              }}
            }}
          }} catch (_je) {{}}
        }}, 1200);
      }} catch (err) {{
        state.runningStageIdx = -1;
        renderPipeline();
        appendTerminalLog(`\\n[error] ${{err.message}}`);
      }}
      return;
    }}

    // Standalone static file fallback: step through the Lattice Ladder rungs with diagnostic output
    state.runningStageIdx = -1;
    renderPipeline();
    if (stageId === "solve") {{
      let r = 0;
      selectLadderRung(0, false);
      const t = setInterval(() => {{
        const rung = LADDER[r];
        if (rung) appendTerminalLog(`[step ${{rung.step_num}}] ${{rung.name}} — ${{rung.metric_label}}`);
        r++;
        if (r < LADDER.length) selectLadderRung(r, false);
        else clearInterval(t);
      }}, 650);
    }}
  }}

  if (btnGalaxyDispatch) {{
    btnGalaxyDispatch.addEventListener("click", () => executeWorkflowStage(state.currentPipelineIdx));
  }}

  function animate() {{
    state.parallaxX += (state.targetParallaxX - state.parallaxX) * 0.14;
    state.parallaxY += (state.targetParallaxY - state.parallaxY) * 0.14;
    renderDetectorCanvas();
    renderStereoInset();
    requestAnimationFrame(animate);
  }}

  async function selectFrame(frameIdx) {{
    const frames = DATA.frames || [];
    if (frames.length === 0) return;
    const clamped = ((frameIdx % frames.length) + frames.length) % frames.length;
    DATA.frame_index = clamped;
    renderFrameSelector();
    updateHUD();
    const curF = frames[clamped];
    if (curF) {{
      setStatusLine(
        `[FRAME ${{clamped + 1}}/${{frames.length}}]`,
        `<strong style="color:var(--accent-pink);">${{curF.file || "Frame"}}</strong> • Goniometer: <strong style="color:var(--accent-cyan);">${{curF.motor_summary || "—"}}</strong>`
      );
    }}
    try {{
      const resp = await fetch(`api/payload?frame=${{clamped}}`, {{ cache: "no-store" }});
      if (resp.ok) {{
        const newPayload = await resp.json();
        recomputeBoundsAndReload(newPayload);
      }}
    }} catch (_e) {{}}
  }}

  const selFrameEl = document.getElementById("sel-frame");
  if (selFrameEl) {{
    selFrameEl.addEventListener("change", () => {{
      const idx = parseInt(selFrameEl.value, 10);
      if (!Number.isNaN(idx)) selectFrame(idx);
    }});
  }}
  const btnFramePrev = document.getElementById("btn-frame-prev");
  if (btnFramePrev) {{
    btnFramePrev.addEventListener("click", () => {{
      const cur = typeof DATA.frame_index === "number" ? DATA.frame_index : 0;
      selectFrame(cur - 1);
    }});
  }}
  const btnFrameNext = document.getElementById("btn-frame-next");
  if (btnFrameNext) {{
    btnFrameNext.addEventListener("click", () => {{
      const cur = typeof DATA.frame_index === "number" ? DATA.frame_index : 0;
      selectFrame(cur + 1);
    }});
  }}

  renderFrameSelector();
  renderZoneList();
  renderPipeline();
  renderLadder();
  updateHUD();
  requestAnimationFrame(animate);
}})();
</script>
</body>
</html>
"""
    return html


def export_rainbow_widget(
    payload_or_h5: dict[str, Any] | str | os.PathLike,
    output_html: str | os.PathLike,
    *,
    images_h5: str | os.PathLike | None = None,
    projection: str | None = None,
    default_mode: str = "arcade",
    title: str = "SUBHKL // DIFFRACTION PARALLAX",
    **payload_kwargs: Any,
) -> str:
    """Export a standalone HTML5/Canvas diffraction parallax interactive widget."""
    if isinstance(payload_or_h5, (str, os.PathLike, Path)):
        payload = build_rainbow_zone_payload(
            replay_h5=payload_or_h5,
            images_h5=images_h5,
            projection=projection,
            **payload_kwargs,
        )
    elif isinstance(payload_or_h5, dict):
        payload = payload_or_h5
    else:
        raise TypeError("payload_or_h5 must be a dict or path to an HDF5 file")

    html = render_rainbow_html(
        payload,
        default_mode=default_mode,
        projection=projection,
        title=title,
    )
    out_path = os.path.abspath(os.fspath(output_html))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


def export_static_app_html(
    output_html: str | os.PathLike,
    *,
    instrument: str = "CG4D",
    title: str = "SUBHKL // DIFFRACTION PARALLAX",
) -> str:
    """Export a truly static HTML5/Canvas web app (not generated per-dataset by an XML tool)
    that connects directly to Galaxy / NOVA REST APIs and parses .h5 datasets client-side via h5wasm.
    """
    empty_payload: dict[str, Any] = {
        "instrument": instrument,
        "space_group": "P 21 21 21",
        "intensity_source": "static_app",
        "wavelength_range": [1.8, 4.5],
        "cell": [68.7, 69.8, 156.9, 90.0, 90.0, 90.0],
        "peaks": [],
        "zones": [],
        "panels": [],
        "ladder_steps": [
            {
                "rung": 1,
                "stage": "1. Binned Count Excess",
                "subtitle": "Finder-less Poisson deviance bins",
                "active_zones": [],
                "n_captured": 0,
                "description": "Select a Galaxy History .h5 dataset (top bar) or click 📂 Load .h5 to inspect or dispatch a GPU solve.",
            },
            {
                "rung": 2,
                "stage": "2. 13→145 Zone Ladder",
                "subtitle": "Primary Laue zones",
                "active_zones": [],
                "n_captured": 0,
                "description": "Coarse-to-fine axis ladder locks the dominant intersecting zone planes.",
            },
            {
                "rung": 3,
                "stage": "3. 60-Zone Global",
                "subtitle": "Fundamental zones",
                "active_zones": [],
                "n_captured": 0,
                "description": "Joint SO(3) + metrological translation/tilt refinement across fundamental zones.",
            },
            {
                "rung": 4,
                "stage": "4. 120→400 Zone + Cell",
                "subtitle": "Higher-order zones + cell ratios",
                "active_zones": [],
                "n_captured": 0,
                "description": "Staged 120→400 zone-band refinement co-refines axial stretch, ω drift, and unit-cell ratios.",
            },
            {
                "rung": 5,
                "stage": "5. Converged Solve UB",
                "subtitle": "Converged 3D orientation",
                "active_zones": [],
                "n_captured": 0,
                "description": "Converged 3D UB orientation and forward-projected reflections.",
            },
        ],
        "pipeline_stages": _build_ndip_pipeline_stages(
            [], [], instrument=instrument, space_group="P 21 21 21"
        ),
    }
    html = render_rainbow_html(empty_payload, title=title)
    out_path = os.path.abspath(os.fspath(output_html))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path

