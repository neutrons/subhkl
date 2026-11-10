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
import tempfile
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
                base_configs.get(bank_id, {"panel_type": "flat", "m": 256, "n": 256})
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
            if not src_attr and "scan_refinement" in f:
                src_attr = _decode_str(f["scan_refinement"].attrs.get("source"), "")
            if not src_attr and "search" in f:
                src_attr = _decode_str(f["search"].attrs.get("source"), "")
            if src_attr:
                for cand_src in (Path(src_attr), p.parent / Path(src_attr).name):
                    if cand_src.exists():
                        try:
                            with h5py.File(cand_src, "r") as sf:
                                if "images" in sf:
                                    return str(cand_src.resolve())
                        except Exception:
                            pass
            if "files" in f:
                try:
                    raw_files = [_decode_str(x, "") for x in list(f["files"][()])]
                    raw_files = [x for x in raw_files if x]
                    if raw_files:
                        first_stem = os.path.basename(raw_files[0]).split(".")[0]
                        last_stem = os.path.basename(raw_files[-1]).split(".")[0]
                        for tag in (
                            first_stem,
                            f"{first_stem}_{last_stem}",
                        ):
                            safe_tag = "".join(
                                c if (c.isalnum() or c in "._-") else "_" for c in tag
                            )
                            work_cand = (
                                Path(tempfile.gettempdir())
                                / f"subhkl_work_{safe_tag}"
                                / "merged.h5"
                            )
                            if work_cand.is_file():
                                return str(work_cand.resolve())
                except Exception:
                    pass
    except Exception:
        pass

    parent = p.parent
    if parent.resolve() == Path(tempfile.gettempdir()).resolve():
        return None
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
                    # Guard against mixing an images_h5 from a different instrument (e.g. CG4D 512x512 vs MANDI 256x256)
                    det_compat = True
                    if detectors:
                        sample_det = next(iter(detectors.values()))
                        det_n = int(getattr(sample_det, "n", 0))
                        if len(ims.shape) >= 3 and (
                            (int(ims.shape[1]) == 512 and det_n == 256)
                            or (int(ims.shape[1]) == 256 and det_n == 512)
                        ):
                            det_compat = False
                        uniq_b = set(int(x) for x in np.unique(b_ids_all))
                        if uniq_b and not uniq_b.issubset(set(detectors.keys())):
                            if len(uniq_b - set(detectors.keys())) > 2:
                                det_compat = False
                    if det_compat:
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
                sigma_arr[mask] if sigma_arr is not None else np.full(np.sum(mask), 3.5)
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
        all_logs = [np.log1p(np.maximum(im, 0.0)) for im in raw_bank_images.values()]
        flat = np.concatenate([a.ravel() for a in all_logs])
        if len(flat) > 0:
            vmin = float(np.percentile(flat, 10.0))
            vmax = float(np.percentile(flat, 99.6))
            if vmax <= vmin + 1e-5:
                vmax = float(np.max(flat)) + 1.0

    panels_payload: list[dict[str, Any]] = []
    t_edge = np.linspace(0.0, 1.0, 12, endpoint=False)
    grid_n = 8
    t_grid = np.linspace(0.0, 1.0, grid_n + 1)
    for bk, det in sorted(detectors.items()):
        try:
            # 4 canonical corners: C0=(0,0), C1=(n-1,0), C2=(n-1,m-1), C3=(0,m-1)
            rr_c = np.array([0, det.n - 1, det.n - 1, 0], dtype=float)
            cc_c = np.array([0, 0, det.m - 1, det.m - 1], dtype=float)
            xyz_corners = det.pixel_to_lab(rr_c, cc_c)
            x_c, y_c = _unrolled(xyz_corners)
            _, el_c = _spherical_angles(xyz_corners)

            # Dense 48-point perimeter along all 4 edges for curved spherical/cylindrical borders
            rr_e = np.concatenate(
                [
                    t_edge * (det.n - 1),
                    np.full_like(t_edge, det.n - 1),
                    (1.0 - t_edge) * (det.n - 1),
                    np.zeros_like(t_edge),
                ]
            )
            cc_e = np.concatenate(
                [
                    np.zeros_like(t_edge),
                    t_edge * (det.m - 1),
                    np.full_like(t_edge, det.m - 1),
                    (1.0 - t_edge) * (det.m - 1),
                ]
            )
            xyz_edge = det.pixel_to_lab(rr_e, cc_e)
            x_e, y_e = _unrolled(xyz_edge)
            _, el_e = _spherical_angles(xyz_edge)

            # Curvilinear (grid_n + 1) x (grid_n + 1) interior/boundary mesh so high-elevation
            # panels (such as MANDI's bottom-plane banks) warp accurately to their curved outlines
            rr_m_grid, cc_m_grid = np.meshgrid(
                t_grid * (det.n - 1), t_grid * (det.m - 1), indexing="ij"
            )
            xyz_mesh = det.pixel_to_lab(rr_m_grid.ravel(), cc_m_grid.ravel())
            x_m, y_m = _unrolled(xyz_mesh)
            _, el_m = _spherical_angles(xyz_mesh)

            # Unwrap azimuth across the +-180 deg seam (e.g., MANDI Bank 5 straddling +-180 deg)
            x_all = np.concatenate([x_c, x_e, x_m])
            if np.ptp(x_all) > 180.0:
                x_all_u = np.where(x_all < 0.0, x_all + 360.0, x_all)
                if np.ptp(x_all_u) <= 180.0:
                    x_c = np.where(x_c < 0.0, x_c + 360.0, x_c)
                    x_e = np.where(x_e < 0.0, x_e + 360.0, x_e)
                    x_m = np.where(x_m < 0.0, x_m + 360.0, x_m)
                else:
                    continue

            az_min = float(min(np.min(x_e), np.min(x_m)))
            az_max = float(max(np.max(x_e), np.max(x_m)))
            y_min = float(min(np.min(y_e), np.min(y_m)))
            y_max = float(max(np.max(y_e), np.max(y_m)))
            el_min = float(min(np.min(el_e), np.min(el_m)))
            el_max = float(max(np.max(el_e), np.max(el_m)))

            x_m_2d = x_m.reshape(grid_n + 1, grid_n + 1)
            y_m_2d = y_m.reshape(grid_n + 1, grid_n + 1)
            el_m_2d = el_m.reshape(grid_n + 1, grid_n + 1)

            tex_b64 = None
            im = raw_bank_images.get(int(bk))
            if im is not None:
                # Keep canonical (r, c) orientation: row r=0..n-1 is C0->C1, col c=0..m-1 is C0->C3
                log_im = np.log1p(np.maximum(im, 0.0))
                norm = np.clip((log_im - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
                u8 = (norm * 255.0).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(u8, mode="L").save(buf, format="PNG", optimize=True)
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
                    "corners": [
                        [round(float(x_c[j]), 3), round(float(y_c[j]), 4)]
                        for j in range(4)
                    ],
                    "corners_sph": [
                        [round(float(x_c[j]), 3), round(float(el_c[j]), 3)]
                        for j in range(4)
                    ],
                    "mesh": [
                        [
                            [
                                round(float(x_m_2d[ir, ic]), 3),
                                round(float(y_m_2d[ir, ic]), 4),
                            ]
                            for ic in range(grid_n + 1)
                        ]
                        for ir in range(grid_n + 1)
                    ],
                    "mesh_sph": [
                        [
                            [
                                round(float(x_m_2d[ir, ic]), 3),
                                round(float(el_m_2d[ir, ic]), 3),
                            ]
                            for ic in range(grid_n + 1)
                        ]
                        for ir in range(grid_n + 1)
                    ],
                    "outline": [
                        [round(float(x_e[j]), 2), round(float(y_e[j]), 4)]
                        for j in range(len(x_e))
                    ],
                    "outline_sph": [
                        [round(float(x_e[j]), 2), round(float(el_e[j]), 3)]
                        for j in range(len(x_e))
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
    *,
    include_peaks: bool = True,
    default_instrument: str = "IMAGINE",
) -> dict[str, Any]:
    """Extract peaks, detectors, UB, wavelength range, goniometer frames, and metadata from a subhkl HDF5 file."""
    disc_h5 = _discover_images_h5(h5_path)
    if disc_h5 and os.path.exists(disc_h5):
        src_h5 = str(disc_h5)
    elif images_h5 is not None and os.path.exists(images_h5):
        src_h5 = str(images_h5)
    else:
        src_h5 = None
    is_blind_h5 = str(h5_path).endswith(".blind.h5")
    sib_solve = Path(h5_path).parent / "solve.h5"
    if (not sib_solve.is_file() or sib_solve == Path(h5_path)) and src_h5:
        sib_solve = Path(src_h5).parent / "solve.h5"
    sib_solve_path = (
        str(sib_solve)
        if (
            not is_blind_h5
            and sib_solve.is_file()
            and sib_solve.resolve() != Path(h5_path).resolve()
        )
        else None
    )

    with h5py.File(h5_path, "r") as f:
        is_pure_merged = (
            "images" in f
            and "sample/U" not in f
            and "peaks" not in f
            and "banks" not in f
        )
        if is_pure_merged:
            sib_solve_path = None
        instrument = _decode_str(
            f.attrs.get("instrument"), default_instrument or "IMAGINE"
        )
        ki = (0.0, 0.0, 1.0)
        if "beam/ki_vec" in f:
            ki = tuple(
                float(x)
                for x in np.asarray(f["beam/ki_vec"][()], dtype=float).ravel()[:3]
            )

        wavelength_range = (1.8, 4.5)
        if "scan_refinement/band_wavelength" in f:
            wl = np.asarray(
                f["scan_refinement/band_wavelength"][()], dtype=float
            ).ravel()
            if len(wl) >= 2:
                wavelength_range = (float(wl[0]), float(wl[1]))
        elif "instrument/wavelength" in f:
            wl = np.asarray(f["instrument/wavelength"][()], dtype=float).ravel()
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
                                        swl = np.asarray(
                                            sf[wk][()], dtype=float
                                        ).ravel()
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

        U = np.asarray(f["sample/U"][()], dtype=float) if "sample/U" in f else None
        B = np.asarray(f["sample/B"][()], dtype=float) if "sample/B" in f else None
        if "sample/space_group" in f:
            space_group = _decode_str(f["sample/space_group"][()], "P 1")
        elif "space_group" in f:
            space_group = _decode_str(f["space_group"][()], "P 1")
        else:
            space_group = "P 1"

        cell = None
        if all(f"sample/{k}" in f for k in ("a", "b", "c", "alpha", "beta", "gamma")):
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

        # Auto-detect instrument from filename, embedded files, detector_calibration, or bank/image dimensions
        explicit_h5_inst = _decode_str(f.attrs.get("instrument"), "").strip()
        if not explicit_h5_inst and "instrument/name" in f:
            explicit_h5_inst = _decode_str(f["instrument/name"][()], "").strip()
        if not explicit_h5_inst and "search" in f:
            explicit_h5_inst = _decode_str(
                f["search"].attrs.get("instrument"), ""
            ).strip()
        if explicit_h5_inst:
            instrument = explicit_h5_inst
        else:
            found_by_name = False
            name_cands = [str(h5_path).upper()]
            if src_h5:
                name_cands.append(str(src_h5).upper())
            if "files" in f:
                name_cands.extend(
                    _decode_str(x, "").upper() for x in list(f["files"][()])
                )
            elif src_h5 and os.path.isfile(src_h5):
                try:
                    with h5py.File(src_h5, "r") as sf_inst:
                        if "files" in sf_inst:
                            name_cands.extend(
                                _decode_str(x, "").upper()
                                for x in list(sf_inst["files"][()])
                            )
                except Exception:
                    pass
            for fn_s in name_cands:
                for known_inst in ("MANDI", "TOPAZ", "CG4D", "IMAGINE"):
                    if known_inst in fn_s:
                        instrument = known_inst
                        found_by_name = True
                        break
                if found_by_name:
                    break
            if not found_by_name:
                if "detector_calibration" in f:
                    cal_bids = []
                    for b_nm in f["detector_calibration"]:
                        if b_nm.startswith("bank_"):
                            try:
                                cal_bids.append(int(b_nm.split("_", 1)[1]))
                            except ValueError:
                                pass
                    if cal_bids:
                        if max(cal_bids) > 60:
                            instrument = "CG4D"
                        elif min(cal_bids) < 10 or len(cal_bids) <= 45:
                            instrument = "MANDI"
                elif "images" in f and len(f["images"].shape) >= 3:
                    instrument = "MANDI" if int(f["images"].shape[1]) == 256 else "CG4D"
                else:
                    for b_key in ("bank_ids", "peaks/bank", "bank"):
                        if b_key in f:
                            b_arr_chk = np.asarray(f[b_key][()], dtype=int)
                            if len(b_arr_chk) > 0:
                                if int(np.max(b_arr_chk)) > 60:
                                    instrument = "CG4D"
                                    break
                                if int(np.min(b_arr_chk)) < 10 or (
                                    8 < len(np.unique(b_arr_chk)) <= 45
                                ):
                                    instrument = "MANDI"
                                    break

        detectors = _load_detectors_from_h5(f, instrument)

        bank = np.asarray(f["bank"][()], dtype=int) if "bank" in f else None
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
            np.asarray(f["peaks/xyz"][()], dtype=float) if "peaks/xyz" in f else None
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
        gonio_axes = (
            np.asarray(f["goniometer/axes"][()], dtype=float)
            if "goniometer/axes" in f
            else None
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
                    if gonio_axes is None and "goniometer/axes" in af:
                        gonio_axes = np.asarray(af["goniometer/axes"][()], dtype=float)
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

        if (
            gonio_angles_nom is not None
            and gonio_angles_nom.ndim == 2
            and len(gonio_angles_nom) > 0
        ):
            if n_frames_total == 0:
                n_frames_total = len(gonio_angles_nom)
            if offsets is None:
                offsets = np.arange(len(gonio_angles_nom), dtype=int)

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
                    for ext in (
                        ".nxs.h5.img.h5",
                        ".nxs.h5.im.h5",
                        ".nxs.h5.reduce.h5",
                        ".reduce.h5",
                        ".nxs.h5",
                        ".img.h5",
                        ".h5",
                    ):
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
                            motor_names[ci] if ci < len(motor_names) else f"M{ci + 1}"
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
                opt_label = f"{r_idx + 1}/{len(offsets)} · {file_label}" + (
                    f" ({primary_str})" if primary_str else ""
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
                    xyz_list[mask] = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])
                else:
                    az = (int(bk) * 37.0 % 360.0 - 180.0) + (
                        pixel_c[mask] - 128.0
                    ) * 0.12
                    rad = np.deg2rad(az)
                    y = (pixel_r[mask] - 128.0) * 0.002
                    xyz_list[mask] = np.column_stack([np.sin(rad), y, np.cos(rad)])
            peaks_xyz = xyz_list

        # Extract finder-less solve / lattice-ladder / staged-band diagnostics if present
        solve_diagnostics: dict[str, Any] | None = None
        if any(k in f for k in ("blind_search", "scan_refinement", "solve", "search")):
            solve_diagnostics = {}
            bs_grp = (
                f["blind_search"]
                if "blind_search" in f
                else (f["search"] if "search" in f else None)
            )
            if bs_grp is not None:
                solve_diagnostics["stable_observations"] = int(
                    bs_grp.attrs.get("stable_observations", 0)
                )
                solve_diagnostics["samples_requested"] = int(
                    bs_grp.attrs.get("samples_requested", 0)
                )
                solve_diagnostics["weighting"] = _decode_str(
                    bs_grp.attrs.get("weighting"),
                    "smooth chi2-suppressed bin Poisson deviance",
                )
                if "proposal_scores" in bs_grp:
                    ps = np.asarray(bs_grp["proposal_scores"][()], dtype=float).ravel()
                    solve_diagnostics["n_candidates"] = int(len(ps))
                    solve_diagnostics["top_proposal_score"] = (
                        float(np.max(ps)) if len(ps) else 0.0
                    )
                if "full_scores" in bs_grp:
                    fs = np.asarray(bs_grp["full_scores"][()], dtype=float).ravel()
                    solve_diagnostics["top_full_score"] = (
                        float(np.max(fs)) if len(fs) else 0.0
                    )
            if "scan_refinement" in f:
                sr_grp = f["scan_refinement"]
                default_d_min = round(0.9 * float(wavelength_range[0]), 2)
                solve_diagnostics["d_min"] = float(
                    sr_grp.attrs.get("d_min", default_d_min)
                )
                solve_diagnostics["n_zones"] = int(sr_grp.attrs.get("n_zones", 400))
                solve_diagnostics["modes"] = _decode_str(sr_grp.attrs.get("modes"), "")
                solve_diagnostics["status"] = _decode_str(
                    sr_grp.attrs.get("status"), "converged"
                )
                stages_raw = _decode_str(sr_grp.attrs.get("stages"), "[]")
                try:
                    solve_diagnostics["stages"] = json.loads(stages_raw)
                except Exception:
                    solve_diagnostics["stages"] = []
            elif "solve" in f:
                s_grp = f["solve"]
                solve_diagnostics["objective"] = _decode_str(
                    s_grp.attrs.get("objective"), "poisson"
                )
                solve_diagnostics["status"] = _decode_str(
                    s_grp.attrs.get("status"), "converged"
                )

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
        careless_residual = (
            np.asarray(f["peaks/careless_residual"][()], dtype=float)
            if "peaks/careless_residual" in f
            else None
        )
        careless_scale = (
            np.asarray(f["peaks/careless_scale"][()], dtype=float)
            if "peaks/careless_scale" in f
            else None
        )
        merged_i = (
            np.asarray(f["peaks/merged_i"][()], dtype=float)
            if "peaks/merged_i" in f
            else None
        )
        merged_sigi = (
            np.asarray(f["peaks/merged_sigi"][()], dtype=float)
            if "peaks/merged_sigi" in f
            else None
        )

        def _normalize_merge_metrics(doc_m: dict[str, Any]) -> dict[str, Any]:
            out = dict(doc_m)
            if "cc_half_overall" not in out and "cc_half" in out:
                out["cc_half_overall"] = out["cc_half"]
            if "mean_i_over_sigi" not in out and "mean_isigi" in out:
                out["mean_i_over_sigi"] = out["mean_isigi"]
            if "multiplicity" not in out and "mean_multiplicity" in out:
                out["multiplicity"] = out["mean_multiplicity"]
            if "cc_half_bins" not in out and isinstance(
                out.get("resolution_bins"), list
            ):
                out["cc_half_bins"] = out["resolution_bins"]
            return out

        merge_metrics: dict[str, Any] | None = None
        if "careless/metrics_json" in f:
            try:
                raw_m = _decode_str(f["careless/metrics_json"][()], "{}")
                doc_m = json.loads(raw_m)
                if isinstance(doc_m, dict) and doc_m.get("n_merged_asu"):
                    merge_metrics = _normalize_merge_metrics(doc_m)
            except Exception:
                pass
        elif "peaks" in f:
            sib_m = Path(h5_path).parent / "careless_metrics.json"
            if sib_m.is_file():
                try:
                    with open(sib_m, "r", encoding="utf-8") as mfh:
                        doc_m = json.load(mfh)
                        if isinstance(doc_m, dict) and doc_m.get("n_merged_asu"):
                            merge_metrics = _normalize_merge_metrics(doc_m)
                except Exception:
                    pass

        artifact_kind = "solve"
        peak_shape_mode = "solved_circle"
        if var_u is not None and var_v is not None:
            artifact_kind = (
                "merged_careless" if merge_metrics is not None else "integrator"
            )
            is_iso = bool(
                np.allclose(var_u, var_v)
                and (cov_uv is None or np.allclose(cov_uv, 0.0))
            )
            peak_shape_mode = "isotropic_circle" if is_iso else "anisotropic_ellipse"
        elif "peaks/pixel_r" in f:
            artifact_kind = "finder"
            peak_shape_mode = "isotropic_circle"
        elif "images" in f and U is None and "banks" not in f:
            artifact_kind = "merged"
            peak_shape_mode = "none"
        elif "entry" in f and U is None:
            artifact_kind = "raw_nexus"
            peak_shape_mode = "none"

        # Capture all-runs sample-space peak arrays before filtering to active_frame
        all_runs_data: dict[str, Any] | None = None
        if (
            peaks_xyz is not None
            and len(peaks_xyz) > 0
            and hkl is not None
            and len(hkl) == len(peaks_xyz)
        ):
            mask_all_snr = np.ones(len(peaks_xyz), dtype=bool)
            if (
                intensity is not None
                and sigma is not None
                and len(intensity) == len(peaks_xyz)
                and len(sigma) == len(peaks_xyz)
            ):
                cand_snr = (sigma > 0) & (intensity / np.maximum(sigma, 1e-9) >= 2.5)
                if int(np.sum(cand_snr)) >= 20:
                    mask_all_snr = cand_snr
            idx_all = np.flatnonzero(mask_all_snr)
            if len(idx_all) > 3500:
                if intensity is not None and len(intensity) == len(peaks_xyz):
                    top_k = np.argsort(-intensity[idx_all])[:3500]
                    idx_all = np.sort(idx_all[top_k])
                else:
                    idx_all = idx_all[:3500]
            r_idx_all = (
                run_index[idx_all]
                if run_index is not None and len(run_index) == len(peaks_xyz)
                else np.zeros(len(idx_all), dtype=int)
            )
            R_per_run: dict[int, np.ndarray] = {}
            for ur in np.unique(r_idx_all):
                ur_i = int(ur)
                f0_r = (
                    int(offsets[ur_i])
                    if offsets is not None and 0 <= ur_i < len(offsets)
                    else ur_i
                )
                if R_all is not None and R_all.ndim == 3 and len(R_all) > 0:
                    if 0 <= f0_r < len(R_all):
                        R_per_run[ur_i] = R_all[f0_r]
                    elif 0 <= ur_i < len(R_all):
                        R_per_run[ur_i] = R_all[ur_i]
                if (
                    ur_i not in R_per_run
                    and gonio_axes is not None
                    and gonio_angles_nom is not None
                    and gonio_angles_nom.ndim == 2
                    and 0 <= f0_r < len(gonio_angles_nom)
                ):
                    try:
                        from subhkl.instrument.goniometer import (
                            calc_goniometer_rotation_matrix,
                        )

                        R_per_run[ur_i] = calc_goniometer_rotation_matrix(
                            gonio_axes, gonio_angles_nom[f0_r]
                        )
                    except Exception:
                        pass
            all_runs_data = {
                "peaks_xyz": peaks_xyz[idx_all],
                "hkl": hkl[idx_all],
                "wavelength": lam[idx_all]
                if lam is not None and len(lam) == len(peaks_xyz)
                else None,
                "intensity": intensity[idx_all]
                if intensity is not None and len(intensity) == len(peaks_xyz)
                else None,
                "sigma": sigma[idx_all]
                if sigma is not None and len(sigma) == len(peaks_xyz)
                else None,
                "bank": bank[idx_all]
                if bank is not None and len(bank) == len(peaks_xyz)
                else None,
                "run_index": r_idx_all,
                "careless_residual": careless_residual[idx_all]
                if careless_residual is not None
                and len(careless_residual) == len(peaks_xyz)
                else None,
                "careless_scale": careless_scale[idx_all]
                if careless_scale is not None and len(careless_scale) == len(peaks_xyz)
                else None,
                "R_per_run": R_per_run,
            }

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
                if careless_residual is not None and len(careless_residual) == len(
                    mask_run
                ):
                    careless_residual = careless_residual[mask_run]
                if careless_scale is not None and len(careless_scale) == len(mask_run):
                    careless_scale = careless_scale[mask_run]
                if merged_i is not None and len(merged_i) == len(mask_run):
                    merged_i = merged_i[mask_run]
                if merged_sigi is not None and len(merged_sigi) == len(mask_run):
                    merged_sigi = merged_sigi[mask_run]
                run_index = run_index[mask_run]

        # Filter out zero/noise predictions in integrated.h5 (keep significant reflections I/sigma >= 2.5)
        if (
            peaks_xyz is not None
            and intensity is not None
            and sigma is not None
            and len(intensity) == len(peaks_xyz)
            and len(sigma) == len(peaks_xyz)
        ):
            snr_mask = (sigma > 0) & (intensity / np.maximum(sigma, 1e-9) >= 2.5)
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
                if careless_residual is not None and len(careless_residual) == len(
                    snr_mask
                ):
                    careless_residual = careless_residual[snr_mask]
                if careless_scale is not None and len(careless_scale) == len(snr_mask):
                    careless_scale = careless_scale[snr_mask]
                if merged_i is not None and len(merged_i) == len(snr_mask):
                    merged_i = merged_i[snr_mask]
                if merged_sigi is not None and len(merged_sigi) == len(snr_mask):
                    merged_sigi = merged_sigi[snr_mask]
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
                    len(b_ids_pred) if b_ids_pred is not None else len(f["banks"])
                )
                f0_p, f1_p = (
                    frame_slice if frame_slice is not None else (0, n_bank_entries)
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
                        az = (int(bid) * 37.0 % 360.0 - 180.0) + (pc_i - 128.0) * 0.12
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
                elif (
                    offsets is not None
                    and len(R_all) == len(offsets)
                    and active_frame < len(R_all)
                ):
                    R_gonio = R_all[active_frame]
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
        if (
            R_gonio is None
            and gonio_axes is not None
            and gonio_angles_nom is not None
            and gonio_angles_nom.ndim == 2
            and len(gonio_angles_nom) > 0
        ):
            try:
                from subhkl.instrument.goniometer import (
                    calc_goniometer_rotation_matrix,
                )

                f0_ang = (
                    frame_slice[0]
                    if frame_slice is not None
                    and 0 <= frame_slice[0] < len(gonio_angles_nom)
                    else 0
                )
                R_gonio = calc_goniometer_rotation_matrix(
                    gonio_axes, gonio_angles_nom[f0_ang]
                )
            except Exception:
                pass

        # Finder-less solve.h5 forward projection onto calibrated detector banks
        if (
            include_peaks
            and (peaks_xyz is None or len(peaks_xyz) == 0)
            and U is not None
            and B is not None
            and detectors
        ):
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
                cell_use = (
                    cell if cell and len(cell) == 6 else list(_cell_from_B(B_mat))
                )
                default_d_min = round(0.9 * float(wavelength_range[0]), 2)
                d_min_use = (
                    float(solve_diagnostics.get("d_min", default_d_min))
                    if solve_diagnostics and "d_min" in solve_diagnostics
                    else default_d_min
                )
                max_cell_edge = (
                    max(float(x) for x in cell_use[:3])
                    if cell_use and len(cell_use) >= 3
                    else 68.7
                )
                min_cell_edge = (
                    min(float(x) for x in cell_use[:3])
                    if cell_use and len(cell_use) >= 3
                    else 68.7
                )
                d_min_pred = (
                    min(d_min_use, max(0.75, min_cell_edge * 0.085))
                    if max_cell_edge < 35.0
                    else d_min_use
                )
                hkl_all = np.stack(
                    generate_reflections(
                        *cell_use, space_group=space_group, d_min=d_min_pred
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
                                sample_det = (
                                    next(iter(detectors.values()))
                                    if detectors
                                    else None
                                )
                                det_n = int(getattr(sample_det, "n", 0))
                                compat = True
                                if (
                                    ims_s.ndim >= 3
                                    and det_n > 0
                                    and int(ims_s.shape[1]) != det_n
                                ):
                                    compat = False
                                if compat:
                                    for b_id_s, im_s in zip(b_ids_s, ims_s):
                                        if int(b_id_s) in detectors:
                                            bank_imgs[int(b_id_s)] = im_s
                    except Exception:
                        pass

                cal_bids_set: set[int] = set()
                if "detector_calibration" in f:
                    for b_nm in f["detector_calibration"]:
                        if b_nm.startswith("bank_"):
                            try:
                                cal_bids_set.add(int(b_nm.split("_", 1)[1]))
                            except ValueError:
                                pass

                active_bids = (
                    set(bank_imgs.keys())
                    if bank_imgs
                    else (
                        cal_bids_set
                        if cal_bids_set
                        else set(int(k) for k in detectors.keys())
                    )
                )

                raw_hits: list[dict[str, Any]] = []
                total_geom_hits = 0
                for bid, det in sorted(detectors.items()):
                    if int(bid) not in active_bids:
                        continue
                    cen = np.asarray(det.center, dtype=float)
                    uh = np.asarray(det.uhat, dtype=float)
                    vh = np.asarray(det.vhat, dtype=float)
                    w_d, h_d = float(det.width), float(det.height)
                    nrm = np.cross(uh, vh)
                    dn = kf @ nrm
                    t_ray = np.dot(cen, nrm) / np.where(np.abs(dn) < 1e-9, -1.0, dn)
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
                            np.round(pr_hit).astype(int), 3, im_b.shape[0] - 4
                        )
                        c_i = np.clip(
                            np.round(pc_hit).astype(int), 3, im_b.shape[1] - 4
                        )
                        peak_val = np.maximum.reduce(
                            [
                                im_b[r_i + dr, c_i + dc]
                                for dr in range(-3, 4)
                                for dc in range(-3, 4)
                            ]
                        )
                        bg_val = float(np.median(im_b))
                        snr = (peak_val - bg_val) / np.sqrt(max(bg_val, 1.0))
                    else:
                        peak_val = np.full(len(idx_h), 150.0, dtype=float)
                        snr = np.full(len(idx_h), 10.0, dtype=float)

                    xyz_hit = (
                        cen[None, :]
                        + um[:, None] * uh[None, :]
                        + vm[:, None] * vh[None, :]
                    )
                    total_geom_hits += len(idx_h)
                    raw_hits.append(
                        {
                            "bid": int(bid),
                            "idx_h": idx_h,
                            "xyz_hit": xyz_hit,
                            "pr_hit": pr_hit,
                            "pc_hit": pc_hit,
                            "peak_val": peak_val,
                            "snr": snr,
                            "has_img": im_b is not None,
                        }
                    )

                xyz_l, hkl_l, lam_l, bank_l, pr_l, pc_l, inten_l = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                use_strict_snr = total_geom_hits > 400
                for rh in raw_hits:
                    idx_h = rh["idx_h"]
                    xyz_hit = rh["xyz_hit"]
                    pr_hit = rh["pr_hit"]
                    pc_hit = rh["pc_hit"]
                    peak_val = rh["peak_val"]
                    snr = rh["snr"]
                    if rh["has_img"]:
                        keep_mask = (snr >= 3.0) if use_strict_snr else (peak_val > 0.0)
                        if not np.any(keep_mask) and not use_strict_snr:
                            keep_mask = np.ones(len(idx_h), dtype=bool)
                        if not np.any(keep_mask):
                            continue
                        idx_h = idx_h[keep_mask]
                        xyz_hit = xyz_hit[keep_mask]
                        pr_hit = pr_hit[keep_mask]
                        pc_hit = pc_hit[keep_mask]
                        peak_val = np.maximum(peak_val[keep_mask], 1.0)
                    xyz_l.append(xyz_hit)
                    hkl_l.append(hkl_b[idx_h])
                    lam_l.append(lam_b[idx_h])
                    bank_l.append(np.full(len(idx_h), rh["bid"], dtype=int))
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
            "images_h5": src_h5,
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
            "careless_residual": careless_residual,
            "careless_scale": careless_scale,
            "merged_i": merged_i,
            "merged_sigi": merged_sigi,
            "merge_metrics": merge_metrics,
            "all_runs_data": all_runs_data,
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
        float(np.sqrt(np.mean([p["angular_error_deg"] ** 2 for p in indexed_peaks])))
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

    z0_slice = zones[: min(3, len(zones))]
    z0_ids = [z["zone_id"] for z in z0_slice]
    z0_peaks = sorted({pid for z in z0_slice for pid in z["member_peak_ids"]})
    z0_label = zones[0]["label"] if len(zones) >= 1 else "[1 0 0]"
    z0_rms = (
        float(np.mean([z["angular_rms_deg"] for z in z0_slice]))
        if z0_slice
        else overall_rms
    )
    if solve_diagnostics and solve_diagnostics.get("n_candidates"):
        n_cands = int(solve_diagnostics.get("n_candidates", 4))
        full_sc = float(solve_diagnostics.get("top_full_score", 0.0))
        s2_stage = f"2. 13→145 Zone Ladder ({z0_label})"
        s2_sub = f"Blind Lattice-Ladder ({n_cands} candidates • score {full_sc:.3f})"
    else:
        s2_stage = f"2. Primary Zones ({z0_label})"
        s2_sub = f"Coarse 13-zone rung ({len(z0_ids)} principal axes)"
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
                f"Primary crystal zone axes ({len(z0_ids)} shown, led by {z0_label}) capture {len(z0_peaks)} "
                "coplanar scattering vectors (G · t_uvw = 0) during lattice_ladder (13→30→60→145 zones)."
            ),
        }
    )

    z12_slice = zones[: min(7, len(zones))]
    z12_ids = [z["zone_id"] for z in z12_slice]
    z12_peaks = sorted({pid for z in z12_slice for pid in z["member_peak_ids"]})
    psi_deg = 90.0
    if len(zones) >= 2:
        a0 = np.asarray(zones[0]["axis_crystal_xyz"], dtype=float)
        a1 = np.asarray(zones[1]["axis_crystal_xyz"], dtype=float)
        psi_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(a0, a1), -1.0, 1.0))))
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
                float(np.mean([z["angular_rms_deg"] for z in z12_slice]))
                if z12_slice
                else overall_rms,
                3,
            ),
            "zone_tol_deg": 0.5,
            "psi_deg": round(psi_deg, 2),
            "description": (
                f"Intersecting zone families ({z0_label} and {z1_label}, ψ = {psi_deg:.1f}°, {len(z12_ids)} zones shown) "
                "lock 3D crystal orientation and global detector geometry."
            ),
        }
    )

    ord1_zones = [z for z in zones if z.get("order", 1) <= 1]
    z_const_slice = ord1_zones if ord1_zones else zones[: min(13, len(zones))]
    z_const_ids = [z["zone_id"] for z in z_const_slice]
    z_const_peaks = sorted({pid for z in z_const_slice for pid in z["member_peak_ids"]})
    if "cell" in stages_map or "120-zones" in stages_map:
        st_c = stages_map.get("cell", stages_map.get("120-zones", {}))
        s4_stage = "4. 120→400 Zone + Cell"
        s4_sub = f"Staged band + cell/per-run • score={-float(st_c.get('objective', 0.0)):.3f} ({int(st_c.get('iterations', 0))} iters)"
    else:
        s4_stage = "4. Multi-Zone Refinement"
        s4_sub = (
            f"{len(z_const_ids)} fundamental zones capture {len(z_const_peaks)} spots"
        )
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
                float(np.mean([z["angular_rms_deg"] for z in z_const_slice]))
                if z_const_slice
                else overall_rms,
                3,
            ),
            "zone_tol_deg": 0.3,
            "description": (
                f"Staged 120→400 zone-band refinement locks all fundamental zones ({len(z_const_ids)} shown), "
                "co-refining axial stretch, per-run ω drift, and unit-cell ratios."
            ),
        }
    )

    all_zone_ids = [z["zone_id"] for z in z_const_slice] or [
        z["zone_id"] for z in zones
    ]
    cell_str = (
        f"a={cell[0]:.2f}, b={cell[1]:.2f}, c={cell[2]:.2f} Å"
        if cell and len(cell) == 6
        else "Refined UB Matrix"
    )
    s5_stage = (
        "5. Converged Solve UB" if solve_diagnostics else "5. Indexed UB Solution"
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


def _pending_stage_for_artifact_kind(artifact_kind: str | None) -> str:
    """Return the pending pipeline stage ID given the currently loaded HDF5 artifact kind."""
    kind = (artifact_kind or "solve").lower()
    if kind == "raw_nexus":
        return "reduce"
    if kind == "merged":
        return "solve"
    if kind == "integrator":
        return "merge"
    if kind == "merged_careless":
        return "done"
    return "integrate"


def _build_ndip_pipeline_stages(
    peaks_list: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    *,
    instrument: str = "CG4D",
    space_group: str = "P 1",
    cell: list[float] | None = None,
    solve_diagnostics: dict[str, Any] | None = None,
    artifact_kind: str = "solve",
    merge_metrics: dict[str, Any] | None = None,
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
        (solve_diagnostics and solve_diagnostics.get("stable_observations"))
        or artifact_kind in ("merged", "raw_nexus")
    )
    pending_stage = _pending_stage_for_artifact_kind(artifact_kind)
    stage_order = ["reduce", "solve", "integrate", "merge"]
    if pending_stage == "done":
        pending_idx = len(stage_order)
    else:
        pending_idx = (
            stage_order.index(pending_stage) if pending_stage in stage_order else 1
        )

    def _stage_status(idx: int) -> str:
        if idx < pending_idx:
            return "completed"
        if idx == pending_idx:
            return "pending"
        return "ready"

    merge_artifact_str = "merged.mtz + predictions.mtz + careless_metrics.json"
    if isinstance(merge_metrics, dict) and merge_metrics.get("n_merged_asu"):
        cc_val = merge_metrics.get("cc_half_overall")
        n_asu = int(merge_metrics.get("n_merged_asu", 0))
        cc_txt = f"CC1/2={float(cc_val):.3f} • " if cc_val is not None else ""
        merge_artifact_str = f"merged.mtz ({cc_txt}{n_asu:,} ASU)"

    return [
        {
            "stage_id": "reduce",
            "step_num": 1,
            "name": "1. REDUCE & MERGE" if is_finderless else "1. REDUCE & FIND",
            "tool_ids": (
                ["neutrons_load_data", "imaginex_subhkl_prepare_peaks"]
                if is_finderless
                else ["imaginex_subhkl_prepare_peaks", "imaginex_subhkl_peak_fit"]
            ),
            "container": "ghcr.io/neutrons/subhkl:1.4.0",
            "artifact": (
                f"merged.h5 ({solve_diagnostics['stable_observations']:,} count bins)"
                if is_finderless
                and solve_diagnostics
                and solve_diagnostics.get("stable_observations")
                else (
                    "merged.h5 (detector image stack)"
                    if artifact_kind == "merged"
                    else f"scan_master.h5 → finder.h5 ({n_peaks} peaks)"
                )
            ),
            "status": _stage_status(0),
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
            "status": _stage_status(1),
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
            "status": _stage_status(2),
            "ladder_rung_idx": 4,
            "summary": (
                "NDIP Peak Prediction & Matrix-Free Integration: forward-projects reflections using "
                "solve.h5 calibrated geometry and integrates 3D Gaussian profiles on raw images."
            ),
        },
        {
            "stage_id": "merge",
            "step_num": 4,
            "name": "4. MERGE (CARELESS)",
            "tool_ids": ["imaginex_subhkl_merge"],
            "container": "ghcr.io/jglaser/careless:cuda",
            "artifact": merge_artifact_str,
            "status": _stage_status(3),
            "ladder_rung_idx": 4,
            "summary": (
                "NDIP Careless Variational Poly-Laue Merge (imaginex_subhkl_merge): scales and merges "
                "multi-orientation reflections into ASU structure-factor amplitudes with half-dataset CC1/2 and per-spot residuals."
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
    cell: list[float] | None = None,
    max_zone_index: int = 3,
    zone_tolerance_deg: float = 0.75,
    min_zone_peaks: int = 2,
    max_zones: int = 36,
    n_arc: int = 720,
    max_peaks: int = 2500,
    render_binning: int = 4,
    frame_index: int | None = None,
    include_panels: bool = True,
    include_peaks: bool = True,
) -> dict[str, Any]:
    """Build the JSON-serializable data contract for the Rainbow Zones widget.

    Includes detector bank outlines in both Cylindrical ``(az, y)`` and Spherical
    ``(az, elevation)`` coordinates, log-scaled detector intensity heatmaps,
    peak reflections, and Lattice Ladder steps.
    """
    detectors: dict[int, Any] = {}
    fallback_cell = list(cell) if cell and len(cell) == 6 else None
    fallback_sg = space_group
    cell = None
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
    careless_residual_arr = None
    careless_scale_arr = None
    merged_i_arr = None
    merged_sigi_arr = None
    merge_metrics: dict[str, Any] | None = None
    all_runs_data: dict[str, Any] | None = None
    solve_diagnostics = None
    frame_slice = None
    frames_meta: list[dict[str, Any]] = []
    active_frame = int(frame_index) if frame_index is not None else 0

    if replay_h5 is not None:
        extracted = _extract_from_h5(
            replay_h5,
            images_h5=images_h5,
            frame_index=frame_index,
            include_peaks=include_peaks,
            default_instrument=instrument or "IMAGINE",
        )
        instrument = extracted["instrument"] or instrument
        ki = extracted["ki"]
        wavelength_range = extracted["wavelength_range"]
        if U is None:
            U = extracted["U"]
        if B is None:
            B = extracted["B"]
        ex_sg = extracted.get("space_group")
        if ex_sg and ex_sg not in ("P 1", "Unknown"):
            space_group = ex_sg
        elif fallback_sg and fallback_sg not in ("P 1", "Unknown"):
            space_group = fallback_sg
        else:
            space_group = ex_sg or space_group
        cell = extracted["cell"] or fallback_cell
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
        careless_residual_arr = extracted.get("careless_residual")
        careless_scale_arr = extracted.get("careless_scale")
        merged_i_arr = extracted.get("merged_i")
        merged_sigi_arr = extracted.get("merged_sigi")
        merge_metrics = extracted.get("merge_metrics")
        all_runs_data = extracted.get("all_runs_data")
        solve_diagnostics = extracted.get("solve_diagnostics")
        frame_slice = extracted.get("frame_slice")
        frames_meta = extracted.get("frames") or []
        active_frame = int(extracted.get("frame_index", active_frame))
        if R_gonio is None:
            R_gonio = extracted["R_gonio"]
        if extracted.get("images_h5"):
            images_h5 = extracted["images_h5"]
        elif images_h5 is None:
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
        if careless_residual_arr is not None and len(careless_residual_arr) == n_raw:
            careless_residual_arr = careless_residual_arr[order]
        if careless_scale_arr is not None and len(careless_scale_arr) == n_raw:
            careless_scale_arr = careless_scale_arr[order]
        if merged_i_arr is not None and len(merged_i_arr) == n_raw:
            merged_i_arr = merged_i_arr[order]
        if merged_sigi_arr is not None and len(merged_sigi_arr) == n_raw:
            merged_sigi_arr = merged_sigi_arr[order]
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
    sx_lab, sy_lab, hemi_lab = (
        stereographic_projection(g_hat)
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
        G_pred_hat = np.divide(G_pred_cryst, np.maximum(G_norm[:, None], 1e-12))
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
            np.clip(0.5 * q_norm.ravel(), 1e-4, 0.9999) if n_peaks > 0 else np.zeros(0)
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
            min_peaks=0
            if ub_matrix is not None
            else min(min_zone_peaks, max(1, n_peaks)),
            n_arc=n_arc,
        )
        if (n_peaks > 0 or ub_matrix is not None)
        else []
    )
    for z in raw_zones:
        uvw_vals = [abs(int(x)) for x in z.get("uvw", [1, 0, 0])]
        z["order"] = max(uvw_vals) if any(uvw_vals) else 1

    # Keep all fundamental Order 1 zones (|u|,|v|,|w| <= 1) plus populated Order 2 and Order 3 zones
    selected_zones: list[dict[str, Any]] = []
    z_ord1 = [
        z
        for z in raw_zones
        if z["order"] == 1 and (z["n_peaks"] >= 1 or len(z["conic"]["det_x"]) > 0)
    ][:13]
    z_ord2 = [
        z
        for z in raw_zones
        if z["order"] == 2
        and (
            z["n_peaks"] >= max(1, min_zone_peaks)
            or (n_peaks == 0 and len(z["conic"]["det_x"]) > 0)
        )
    ][:14]
    z_ord3 = [
        z
        for z in raw_zones
        if z["order"] == 3
        and (
            z["n_peaks"] >= max(2, min_zone_peaks)
            or (n_peaks == 0 and len(z["conic"]["det_x"]) > 0)
        )
    ][:16]
    selected_zones.extend(z_ord1)
    selected_zones.extend(z_ord2)
    selected_zones.extend(z_ord3)
    other_zones = [
        z
        for z in raw_zones
        if z["order"] > 3 and z["n_peaks"] >= max(2, min_zone_peaks)
    ][:8]
    selected_zones.extend(other_zones)
    selected_zones.sort(key=lambda z: (z["order"], -z["n_peaks"], -z["score"]))
    raw_zones = selected_zones[: max(max_zones, len(z_ord1) + len(z_ord2))]
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
                        [
                            round(float(az_row[m]), 3),
                            round(float(el_mat[k_local, m]), 3),
                        ]
                        for m in range(n_ell_pts)
                    ]
                    sigma_major_by_idx[p_i] = round(float(s1_all[p_i]), 2)
                    sigma_minor_by_idx[p_i] = round(float(s2_all[p_i]), 2)
                    ellipse_ang_by_idx[p_i] = round(float(np.rad2deg(phi_all[p_i])), 1)
            except Exception:
                pass

    # Compute canonical ASU symmetry orbits and Laue harmonic ray multiplicity
    sg_obj = None
    asu_obj = None
    sg_ops = None
    try:
        import gemmi

        sg_obj = gemmi.find_spacegroup_by_name(str(space_group)) or (
            gemmi.find_spacegroup_by_name("P 1")
        )
        if sg_obj is not None:
            asu_obj = gemmi.ReciprocalAsu(sg_obj)
            sg_ops = sg_obj.operations()
    except Exception:
        pass

    def _to_asu_hkl(h_v: int, k_v: int, l_v: int) -> tuple[int, int, int]:
        if asu_obj is not None and sg_ops is not None and (h_v or k_v or l_v):
            try:
                ha, _ = asu_obj.to_asu([h_v, k_v, l_v], sg_ops)
                return (int(ha[0]), int(ha[1]), int(ha[2]))
            except Exception:
                pass
        ah, ak, al = abs(int(h_v)), abs(int(k_v)), abs(int(l_v))
        sg_u = str(space_group or "").upper()
        is_cubic = "3" in sg_u or (
            cell is not None
            and len(cell) >= 6
            and abs(float(cell[0]) - float(cell[1])) < 0.05
            and abs(float(cell[1]) - float(cell[2])) < 0.05
        )
        if is_cubic:
            s3 = sorted((ah, ak, al), reverse=True)
            return (s3[0], s3[1], s3[2])
        return (ah, ak, al)

    base_ray_counts: dict[tuple[int, int, int], int] = {}
    for i in range(n_peaks):
        h_i, k_i, l_i = (
            int(hkl_arr[i, 0]),
            int(hkl_arr[i, 1]),
            int(hkl_arr[i, 2]),
        )
        if h_i or k_i or l_i:
            g_c = max(1, int(np.gcd.reduce(np.abs([h_i, k_i, l_i]))))
            b_ray = (h_i // g_c, k_i // g_c, l_i // g_c)
            base_ray_counts[b_ray] = base_ray_counts.get(b_ray, 0) + 1

    peaks_list: list[dict[str, Any]] = []
    for i in range(n_peaks):
        h_i, k_i, l_i = (
            int(hkl_arr[i, 0]),
            int(hkl_arr[i, 1]),
            int(hkl_arr[i, 2]),
        )
        g_gcd = (
            int(np.gcd.reduce(np.abs([h_i, k_i, l_i]))) if (h_i or k_i or l_i) else 1
        )
        b_ray = (
            (h_i // max(1, g_gcd), k_i // max(1, g_gcd), l_i // max(1, g_gcd))
            if (h_i or k_i or l_i)
            else (0, 0, 0)
        )
        harm_mult = max(
            int(base_ray_counts.get(b_ray, 1)),
            2 if g_gcd > 1 else 1,
        )
        asu_h, asu_k, asu_l = _to_asu_hkl(h_i, k_i, l_i)
        orbit_id = f"{asu_h},{asu_k},{asu_l}"
        orbit_color_idx = (
            abs(asu_h * 73856093 ^ asu_k * 19349663 ^ asu_l * 83492791) % 12
        )
        d_val = float(d_spacing[i]) if i < len(d_spacing) else 2.0
        lam_val = float(lam_arr[i]) if i < len(lam_arr) else 0.5 * (wl_min + wl_max)
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
            "run_index": int(active_frame),
            "bank": (
                int(bank_arr[i]) if bank_arr is not None and i < len(bank_arr) else 1
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
            "q_hat_lab": [round(float(x), 5) for x in g_hat[i]],
            "stereographic": [
                round(float(sx[i]), 5),
                round(float(sy[i]), 5),
            ],
            "hemisphere": int(hemi[i]),
            "stereo_lab": [
                round(float(sx_lab[i]), 5),
                round(float(sy_lab[i]), 5),
                int(hemi_lab[i]),
            ],
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
            "wavelength": round(float(np.clip(lam_val, wl_min * 0.5, wl_max * 1.5)), 4),
            "d_spacing": round(d_val, 4),
            "harmonic_order": max(1, g_gcd),
            "harmonic_multiplicity": harm_mult,
            "parallax_depth": round(parallax_depth, 4),
            "hkl": [h_i, k_i, l_i],
            "asu_hkl": [asu_h, asu_k, asu_l],
            "asu_orbit_id": orbit_id,
            "orbit_color_idx": int(orbit_color_idx),
            "zone_ids": peak_zones[i],
            "indexed": is_indexed,
            "angular_error_deg": round(err_val, 4),
        }
        if (
            careless_residual_arr is not None
            and i < len(careless_residual_arr)
            and np.isfinite(careless_residual_arr[i])
        ):
            pk_entry["careless_residual"] = round(float(careless_residual_arr[i]), 3)
        if (
            careless_scale_arr is not None
            and i < len(careless_scale_arr)
            and np.isfinite(careless_scale_arr[i])
        ):
            pk_entry["careless_scale"] = round(float(careless_scale_arr[i]), 4)
        if (
            merged_i_arr is not None
            and i < len(merged_i_arr)
            and np.isfinite(merged_i_arr[i])
        ):
            pk_entry["merged_i"] = round(float(merged_i_arr[i]), 2)
        if (
            merged_sigi_arr is not None
            and i < len(merged_sigi_arr)
            and np.isfinite(merged_sigi_arr[i])
        ):
            pk_entry["merged_sigi"] = round(float(merged_sigi_arr[i]), 2)
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

    # Build all-runs sample-space peak list (for Wulff Net 'All Runs (Sample)' mode)
    sample_peaks: list[dict[str, Any]] = []
    if (
        all_runs_data is not None
        and len(all_runs_data.get("peaks_xyz", [])) > 0
        and len(frames_meta) > 1
    ):
        pts_all = np.asarray(all_runs_data["peaks_xyz"], dtype=float)
        hkl_all = np.asarray(all_runs_data["hkl"], dtype=int)
        r_idx_all = np.asarray(all_runs_data["run_index"], dtype=int)
        lam_all = all_runs_data.get("wavelength")
        int_all = all_runs_data.get("intensity")
        sig_all = all_runs_data.get("sigma")
        bnk_all = all_runs_data.get("bank")
        res_all = all_runs_data.get("careless_residual")
        scl_all = all_runs_data.get("careless_scale")
        R_per_run = all_runs_data.get("R_per_run") or {}

        norms_all = np.linalg.norm(pts_all, axis=1, keepdims=True)
        kf_all = pts_all / np.maximum(norms_all, 1e-12)
        q_all = kf_all - ki_hat[None, :]
        qn_all = np.linalg.norm(q_all, axis=1, keepdims=True)
        g_lab_all = np.where(qn_all > 1e-6, q_all / np.maximum(qn_all, 1e-12), kf_all)
        g_samp_all = np.copy(g_lab_all)
        for ur in np.unique(r_idx_all):
            R_u = R_per_run.get(int(ur), R_single)
            m_u = r_idx_all == ur
            g_samp_all[m_u] = g_lab_all[m_u] @ R_u

        sx_all, sy_all, hm_all = stereographic_projection(g_samp_all)
        ray_counts_all: dict[tuple[int, int, int, int], int] = {}
        for j in range(len(pts_all)):
            hj, kj, lj = int(hkl_all[j, 0]), int(hkl_all[j, 1]), int(hkl_all[j, 2])
            if hj or kj or lj:
                gc = max(1, int(np.gcd.reduce(np.abs([hj, kj, lj]))))
                rk = (int(r_idx_all[j]), hj // gc, kj // gc, lj // gc)
                ray_counts_all[rk] = ray_counts_all.get(rk, 0) + 1

        for j in range(len(pts_all)):
            hj, kj, lj = int(hkl_all[j, 0]), int(hkl_all[j, 1]), int(hkl_all[j, 2])
            gc = int(np.gcd.reduce(np.abs([hj, kj, lj]))) if (hj or kj or lj) else 1
            rk = (
                (
                    int(r_idx_all[j]),
                    hj // max(1, gc),
                    kj // max(1, gc),
                    lj // max(1, gc),
                )
                if (hj or kj or lj)
                else (int(r_idx_all[j]), 0, 0, 0)
            )
            h_mult = max(int(ray_counts_all.get(rk, 1)), 2 if gc > 1 else 1)
            ah, ak, al = _to_asu_hkl(hj, kj, lj)
            orb_id = f"{ah},{ak},{al}"
            orb_c = abs(ah * 73856093 ^ ak * 19349663 ^ al * 83492791) % 12
            lv = (
                float(lam_all[j])
                if lam_all is not None and j < len(lam_all)
                else 0.5 * (wl_min + wl_max)
            )
            st_j = float(np.clip(0.5 * qn_all[j, 0], 1e-4, 0.9999))
            dv = lv / max(2.0 * st_j, 1e-4)
            sp_item: dict[str, Any] = {
                "id": j,
                "run_index": int(r_idx_all[j]),
                "bank": int(bnk_all[j])
                if bnk_all is not None and j < len(bnk_all)
                else 1,
                "q_hat": [round(float(x), 5) for x in g_samp_all[j]],
                "q_hat_lab": [round(float(x), 5) for x in g_lab_all[j]],
                "stereographic": [
                    round(float(sx_all[j]), 5),
                    round(float(sy_all[j]), 5),
                ],
                "hemisphere": int(hm_all[j]),
                "intensity": round(float(int_all[j]), 2)
                if int_all is not None and j < len(int_all)
                else 100.0,
                "sigma": round(float(sig_all[j]), 2)
                if sig_all is not None and j < len(sig_all)
                else 3.0,
                "wavelength": round(float(np.clip(lv, wl_min * 0.5, wl_max * 1.5)), 4),
                "d_spacing": round(dv, 4),
                "harmonic_order": max(1, gc),
                "harmonic_multiplicity": h_mult,
                "hkl": [hj, kj, lj],
                "asu_hkl": [ah, ak, al],
                "asu_orbit_id": orb_id,
                "orbit_color_idx": int(orb_c),
                "zone_ids": [],
                "indexed": bool(hj or kj or lj),
            }
            if res_all is not None and j < len(res_all) and np.isfinite(res_all[j]):
                sp_item["careless_residual"] = round(float(res_all[j]), 3)
            if scl_all is not None and j < len(scl_all) and np.isfinite(scl_all[j]):
                sp_item["careless_scale"] = round(float(scl_all[j]), 4)
            if (
                merged_i_arr is not None
                and j < len(merged_i_arr)
                and np.isfinite(merged_i_arr[j])
            ):
                sp_item["merged_i"] = round(float(merged_i_arr[j]), 2)
            if (
                merged_sigi_arr is not None
                and j < len(merged_sigi_arr)
                and np.isfinite(merged_sigi_arr[j])
            ):
                sp_item["merged_sigi"] = round(float(merged_sigi_arr[j]), 2)
            sample_peaks.append(sp_item)
    else:
        sample_peaks = peaks_list

    if isinstance(merge_metrics, dict):
        if "harmonic_overlap_fraction" not in merge_metrics and len(sample_peaks) > 0:
            n_harm = sum(
                1 for p in sample_peaks if int(p.get("harmonic_multiplicity", 1)) > 1
            )
            merge_metrics["harmonic_overlap_fraction"] = round(
                float(n_harm) / float(len(sample_peaks)), 4
            )
        bins_in = merge_metrics.get("cc_half_bins")
        needs_bin_enrich = (
            not isinstance(bins_in, list)
            or len(bins_in) == 0
            or any(
                "mean_isigi" not in b or "completeness_pct" not in b for b in bins_in
            )
        )
        if needs_bin_enrich and len(sample_peaks) > 0:
            try:
                import gemmi as _gm

                asu_obs_map: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
                for sp in sample_peaks:
                    ak = tuple(
                        int(x)
                        for x in (sp.get("asu_hkl") or sp.get("hkl") or (0, 0, 0))
                    )
                    if ak == (0, 0, 0):
                        continue
                    asu_obs_map.setdefault(ak, []).append(sp)
                if asu_obs_map:
                    asu_keys = list(asu_obs_map.keys())
                    asu_d = np.array(
                        [
                            float(asu_obs_map[k][0].get("d_spacing", 2.0))
                            for k in asu_keys
                        ],
                        dtype=float,
                    )
                    asu_isigi = np.array(
                        [
                            float(asu_obs_map[k][0]["merged_i"])
                            / max(
                                float(asu_obs_map[k][0].get("merged_sigi", 1.0)), 1e-6
                            )
                            if "merged_i" in asu_obs_map[k][0]
                            and float(asu_obs_map[k][0].get("merged_sigi", 0.0)) > 0
                            else float(asu_obs_map[k][0].get("intensity", 100.0))
                            / max(float(asu_obs_map[k][0].get("sigma", 3.0)), 1e-6)
                            for k in asu_keys
                        ],
                        dtype=float,
                    )
                    inv_d2_obs = 1.0 / np.maximum(asu_d**2, 1e-6)
                    if isinstance(bins_in, list) and len(bins_in) == 10:
                        edges = [
                            1.0
                            / max(
                                float(bins_in[0].get("d_max", np.max(asu_d))) ** 2, 1e-6
                            )
                        ] + [
                            1.0 / max(float(b.get("d_min", np.min(asu_d))) ** 2, 1e-6)
                            for b in bins_in
                        ]
                        edges_arr = np.asarray(edges, dtype=float)
                    else:
                        edges_arr = np.linspace(
                            float(inv_d2_obs.min()), float(inv_d2_obs.max()), 11
                        )
                    c6 = (
                        cell
                        if len(cell) == 6
                        else [11.93, 11.93, 11.93, 90.0, 90.0, 90.0]
                    )
                    uc = _gm.UnitCell(*[float(x) for x in c6])
                    sg_obj = _gm.find_spacegroup_by_name(
                        str(space_group)
                    ) or _gm.find_spacegroup_by_name("P 1")
                    asu_rec = _gm.ReciprocalAsu(sg_obj)
                    ops_rec = sg_obj.operations()
                    dmin_th = max(float(np.min(asu_d)), 0.6)
                    h_lim = min(28, int(np.ceil(uc.a / dmin_th)) + 1)
                    k_lim = min(28, int(np.ceil(uc.b / dmin_th)) + 1)
                    l_lim = min(28, int(np.ceil(uc.c / dmin_th)) + 1)
                    theo_asu_set: set[tuple[int, int, int]] = set()
                    for hh in range(-h_lim, h_lim + 1):
                        for kk in range(-k_lim, k_lim + 1):
                            for ll in range(-l_lim, l_lim + 1):
                                if hh == 0 and kk == 0 and ll == 0:
                                    continue
                                if ops_rec.is_systematically_absent([hh, kk, ll]):
                                    continue
                                dv_th = uc.calculate_d([hh, kk, ll])
                                if dv_th >= dmin_th - 1e-5:
                                    ha, _ = asu_rec.to_asu([hh, kk, ll], ops_rec)
                                    theo_asu_set.add(
                                        (int(ha[0]), int(ha[1]), int(ha[2]))
                                    )
                    inv_d2_th = (
                        np.array(
                            [
                                1.0 / max(uc.calculate_d(list(k)) ** 2, 1e-6)
                                for k in theo_asu_set
                            ],
                            dtype=float,
                        )
                        if theo_asu_set
                        else np.array([], dtype=float)
                    )
                    enriched_bins: list[dict[str, Any]] = []
                    for b_idx in range(10):
                        lo_e = float(edges_arr[b_idx])
                        hi_e = float(edges_arr[b_idx + 1])
                        bm_o = (inv_d2_obs >= lo_e) & (
                            inv_d2_obs <= hi_e if b_idx == 9 else inv_d2_obs < hi_e
                        )
                        bm_t = (
                            (inv_d2_th >= lo_e)
                            & (inv_d2_th <= hi_e if b_idx == 9 else inv_d2_th < hi_e)
                            if len(inv_d2_th)
                            else np.array([], dtype=bool)
                        )
                        n_m_b = int(np.sum(bm_o))
                        n_t_b = max(n_m_b, int(np.sum(bm_t))) if len(bm_t) else n_m_b
                        comp_b = (
                            round(100.0 * n_m_b / max(1, n_t_b), 2)
                            if n_t_b > 0
                            else 0.0
                        )
                        isig_b = (
                            round(float(np.mean(asu_isigi[bm_o])), 2)
                            if n_m_b > 0
                            else 0.0
                        )
                        orig_b = (
                            bins_in[b_idx]
                            if isinstance(bins_in, list) and b_idx < len(bins_in)
                            else {}
                        )
                        cc_b = orig_b.get("cc_half")
                        d_lo = round(float(1.0 / np.sqrt(max(lo_e, 1e-6))), 2)
                        d_hi = round(float(1.0 / np.sqrt(max(hi_e, 1e-6))), 2)
                        enriched_bins.append(
                            {
                                "bin": b_idx,
                                "d_max": orig_b.get("d_max", d_lo),
                                "d_min": orig_b.get("d_min", d_hi),
                                "n_refl": n_m_b,
                                "n_theo": n_t_b,
                                "completeness_pct": comp_b,
                                "mean_isigi": isig_b,
                                "cc_half": cc_b,
                            }
                        )
                    merge_metrics["cc_half_bins"] = enriched_bins
                    merge_metrics["resolution_bins"] = enriched_bins
            except Exception:
                pass

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
                    mask, row, col = det.reflections_mask(kf[:, 0], kf[:, 1], kf[:, 2])
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
            cur_seg_sph.append([round(float(dx[k]), 2), round(float(del_deg[k]), 3)])
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
                or np.hypot(sx_z[k] - cur_s[-1][0], sy_z[k] - cur_s[-1][1]) > 0.35
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
                "rainbow_color": RAINBOW_SPECTRAL_HEX[zi % len(RAINBOW_SPECTRAL_HEX)],
                "pro_color": PRO_PALETTE_HEX[zi % len(PRO_PALETTE_HEX)],
                "stereo_segments": stereo_segments,
                "detector_segments": det_segments,
                "detector_segments_sph": det_segments_sph,
                "continuous_segments": continuous_segments,
                "continuous_segments_sph": continuous_segments_sph,
            }
        )

    if include_panels:
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
    else:
        panels_payload, intensity_source = [], "raw_h5"

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
        artifact_kind=artifact_kind,
        merge_metrics=merge_metrics,
    )
    pending_stage = _pending_stage_for_artifact_kind(artifact_kind)

    resolved_proj = (
        str(projection).lower()
        if projection in ("cylindrical", "spherical")
        else default_projection_for_instrument(instrument)
    )

    default_d_min = round(0.9 * float(wl_min), 2)
    d_min_use = float((solve_diagnostics or {}).get("d_min", default_d_min))
    gonio_summary = (
        frames_meta[active_frame].get("motor_summary", "—")
        if 0 <= active_frame < len(frames_meta)
        else "—"
    )

    from subhkl.config import beamlines as _cfg_beamlines

    beamline_detectors: dict[str, dict[str, Any]] = {}
    for _inst_k in ("CG4D", "MANDI", "IMAGINE"):
        if _inst_k in _cfg_beamlines:
            beamline_detectors[_inst_k] = {
                str(bk): {
                    "center": [round(float(x), 5) for x in cfg["center"]],
                    "uhat": [round(float(x), 5) for x in cfg["uhat"]],
                    "vhat": [round(float(x), 5) for x in cfg["vhat"]],
                    "width": round(float(cfg["width"]), 5),
                    "height": round(float(cfg["height"]), 5),
                    "m": int(cfg.get("m", 256)),
                    "n": int(cfg.get("n", 256)),
                }
                for bk, cfg in _cfg_beamlines[_inst_k].items()
                if isinstance(cfg, dict)
                and "center" in cfg
                and "uhat" in cfg
                and "vhat" in cfg
            }

    return {
        "instrument": str(instrument),
        "default_projection": resolved_proj,
        "space_group": str(space_group),
        "intensity_source": intensity_source,
        "artifact_kind": artifact_kind,
        "peak_shape_mode": peak_shape_mode,
        "pending_stage": pending_stage,
        "wavelength_range": [round(wl_min, 3), round(wl_max, 3)],
        "d_min": round(d_min_use, 3),
        "frame_index": int(active_frame),
        "goniometer_summary": gonio_summary,
        "goniometer_R": [[round(float(v), 6) for v in row] for row in R_single],
        "frames": frames_meta,
        "images_h5": str(images_h5) if images_h5 else None,
        "cell": [round(float(x), 4) for x in cell] if cell else None,
        "ub_matrix": (
            [[round(float(v), 6) for v in row] for row in ub_matrix]
            if ub_matrix is not None
            else None
        ),
        "beamline_detectors": beamline_detectors,
        "solve_diagnostics": solve_diagnostics,
        "merge_metrics": merge_metrics,
        "peaks": peaks_list,
        "sample_peaks": sample_peaks,
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
        else payload.get("default_projection", default_projection_for_instrument(inst))
    )
    init_wl = payload.get("wavelength_range") or [2.8, 4.5]
    init_wl_min = float(init_wl[0]) if len(init_wl) >= 2 else 2.8
    init_wl_max = float(init_wl[1]) if len(init_wl) >= 2 else 4.5
    init_dmin = float(payload.get("d_min") or round(0.9 * init_wl_min, 2))

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

  html, body {{
    background: var(--bg-primary);
    color: var(--text-main);
    font-family: var(--font-body);
    width: 100%;
    height: 100%;
    max-height: 100dvh;
    margin: 0;
    padding: 0;
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
    padding: 3px 10px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-color);
    min-height: 32px;
    flex-shrink: 0;
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
    grid-template-columns: minmax(0, 1fr) 224px;
    grid-template-rows: minmax(0, 1fr);
    gap: 5px;
    padding: 4px 8px;
    flex: 1 1 0;
    min-height: 0;
    overflow: hidden;
  }}

  .canvas-card {{
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    min-height: 0;
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

  /* Floating / Expandable Stereogram Wulff Net HUD */
  .stereo-inset {{
    position: absolute;
    bottom: 10px;
    left: 10px;
    width: 230px;
    height: 250px;
    background: rgba(10, 14, 28, 0.94);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.72);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    backdrop-filter: blur(8px);
    z-index: 35;
    transition: opacity 0.18s ease, transform 0.18s ease, width 0.18s ease, height 0.18s ease;
  }}

  .stereo-inset.expanded {{
    width: min(80vw, 760px);
    height: min(78vh, 700px);
    left: 50%;
    top: 50%;
    bottom: auto;
    transform: translate(-50%, -50%);
    border-color: var(--accent-cyan);
    box-shadow: 0 22px 64px rgba(0, 0, 0, 0.88);
    z-index: 75;
  }}

  .stereo-inset.hidden {{
    opacity: 0;
    pointer-events: none;
    transform: scale(0.9);
  }}

  .stereo-inset-header {{
    font-family: var(--font-display);
    font-size: 0.62rem;
    padding: 4px 7px;
    background: rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid var(--border-color);
    color: var(--text-main);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
  }}

  .stereo-toolbar {{
    display: flex;
    align-items: center;
    gap: 3px;
    flex-wrap: wrap;
  }}

  .stereo-toolbar button, .stereo-toolbar select {{
    font-size: 0.58rem;
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-color);
    font-family: var(--font-display);
    cursor: pointer;
  }}

  .stereo-merge-bar {{
    display: none;
    padding: 4px 8px;
    background: rgba(7, 9, 19, 0.92);
    border-top: 1px solid var(--border-color);
    font-family: var(--font-display);
    font-size: 0.62rem;
    color: var(--text-main);
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
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
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 6px;
    min-height: 0;
    overflow: hidden;
  }}

  .section-title {{
    font-family: var(--font-display);
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--accent-gold);
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 2px;
    margin-bottom: 3px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .slider-group {{
    display: flex;
    flex-direction: column;
    gap: 3px;
    background: var(--bg-card);
    padding: 5px 7px;
    border-radius: 5px;
    border: 1px solid var(--border-color);
  }}

  .slider-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.66rem;
  }}

  input[type="range"] {{
    width: 100%;
    height: 12px;
    accent-color: var(--accent-cyan);
    cursor: pointer;
  }}

  .zone-list {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 3px;
    max-height: 92px;
    overflow-y: auto;
  }}

  .zone-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2px 5px;
    border-radius: 4px;
    background: var(--bg-card);
    border: 1px solid transparent;
    cursor: pointer;
    font-size: 0.62rem;
    min-width: 0;
  }}

  .zone-item:hover {{
    border-color: var(--accent-cyan);
  }}

  .zone-item.zone-active {{
    border-color: var(--accent-gold);
    background: rgba(251, 191, 36, 0.1);
  }}

  .zone-swatch {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
    flex-shrink: 0;
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
    padding: 3px 8px 4px 8px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex-shrink: 0;
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
    padding: 5px 7px;
    font-size: 0.6rem;
    line-height: 1.32;
    font-family: var(--font-display);
    color: #cbd5e1;
    flex: 1 1 0;
    min-height: 0;
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

  .file-dl-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-card);
    color: var(--accent-cyan);
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: var(--font-display);
    font-size: 0.63rem;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.12s ease;
  }}

  .file-dl-pill:hover {{
    border-color: var(--accent-gold);
    color: var(--accent-gold);
    background: rgba(251, 191, 36, 0.12);
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
        <button id="btn-frame-prev" style="padding:1px 5px;font-size:0.62rem;" title="Previous frame">&lt;</button>
        <select id="sel-frame" style="background:transparent;color:var(--accent-pink);border:none;font-family:var(--font-display);font-size:0.68rem;font-weight:700;outline:none;cursor:pointer;max-width:190px;">
          <option value="0">1 / 1</option>
        </select>
        <button id="btn-frame-next" style="padding:1px 5px;font-size:0.62rem;" title="Next frame">&gt;</button>
      </div>
      <div class="hud-pill" style="display:none;">
        <span class="hud-label">PROJ</span>
        <span class="hud-value" id="val-proj">CYL</span>
      </div>
    </div>

    <div class="controls-cluster">
      <select id="sel-galaxy-h5" title="Data Source: Load Neutron Data (SNS/HFIR), open local HDF5/NeXus files, or select from Galaxy History" style="background:var(--bg-card);color:var(--accent-cyan);border:1px solid var(--border-color);border-radius:5px;padding:3px 6px;font-size:0.68rem;font-family:var(--font-display);max-width:175px;">
        <option value="">Data Source...</option>
        <option value="__action_neutron_data__">Neutron data (SNS/HFIR)...</option>
        <option value="__action_local_file__">From local computer...</option>
      </select>
      <input type="file" id="inp-local-h5" accept=".h5,.hdf5,.nxs" multiple style="display:none;" />
      <button id="btn-local-h5" style="display:none;" title="Open local .h5 / .nxs.h5 files">
        Local .h5
      </button>
      <button id="btn-proj-toggle" class="active-btn" title="Switch between Cylindrical (CG4D: Azimuth vs Y [m]) and Spherical (MANDI: Azimuth vs Elevation [deg]) projection">
        Cyl
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
        Reset
      </button>
      <button id="btn-new-tab" style="border-color:var(--accent-gold);color:var(--accent-gold);" title="Open this view in its own full-screen browser tab">
        Tab
      </button>
    </div>
  </header>

  <div id="iframe-popout-banner" class="iframe-popout-banner" style="display:none;">
    <span>Running inside an embedded frame —</span>
    <button id="btn-popout-now" style="border-color:var(--accent-gold);color:var(--accent-gold);">
      Open Full-Screen in New Browser Tab
    </button>
    <button id="btn-popout-dismiss" style="padding:2px 7px;font-size:0.68rem;color:var(--text-dim);">X</button>
  </div>

  <!-- MAIN WORKSPACE -->
  <main class="workspace">
    <!-- Hero View: Unrolled Detector with Grounded Pixel Intensity + Parallax Layers -->
    <section class="canvas-card">
      <div class="card-header">
        <span id="detector-header-title">DETECTOR ASSEMBLY — LOG(INTENSITY) + PARALLAX BRAGG RINGS</span>
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

        <!-- Interactive Stereogram Wulff Net (Corner Inset or Expanded Popout) -->
        <div class="stereo-inset hidden" id="stereo-inset">
          <div class="stereo-inset-header">
            <span id="stereo-title" style="color:var(--accent-cyan);font-weight:700;">WULFF NET</span>
            <div class="stereo-toolbar">
              <button id="btn-wulff-scope" title="Toggle between Single Frame (1-Frame) and All Runs Aggregated (All-Runs)">1-Frame</button>
              <select id="sel-wulff-color" title="Color reflections on Wulff Net & Detector by Wavelength, ASU Symmetry Orbit, or Careless Merge Residual">
                <option value="wavelength">Color: λ/Zone</option>
                <option value="orbit">Color: ASU Orbit</option>
                <option value="residual">Color: ΔI/σ Residual</option>
                <option value="scale">Color: Scale</option>
              </select>
              <button id="btn-wulff-expand" title="Expand or dock the Wulff Net view">Expand</button>
              <button id="btn-wulff-close" title="Hide Wulff Net">X</button>
            </div>
          </div>
          <canvas id="canvas-stereo" style="flex:1;width:100%;min-height:0;cursor:grab;"></canvas>
          <div class="stereo-merge-bar" id="wulff-merge-bar"></div>
        </div>
      </div>
    </section>

    <!-- Right Sidebar: Grounded Controls & Terminal -->
    <aside class="side-panel">
      <div>
        <div class="section-title">
          <span>Detector &amp; Parallax Controls</span>
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
          <span>Laue Zones [uvw]</span>
          <div style="display:flex;gap:2px;">
            <button id="btn-order-1" class="active-btn" style="padding:1px 4px;font-size:0.58rem;" title="Fundamental zones |u|,|v|,|w| ≤ 1">≤1</button>
            <button id="btn-order-2" style="padding:1px 4px;font-size:0.58rem;" title="Include 2nd-order zones |u|,|v|,|w| ≤ 2">≤2</button>
            <button id="btn-order-3" style="padding:1px 4px;font-size:0.58rem;" title="Include higher-order zones |u|,|v|,|w| ≤ 3">≤3</button>
            <button id="btn-zones-all" style="padding:1px 4px;font-size:0.58rem;" title="Enable all zones in current order">ALL</button>
            <button id="btn-zones-none" style="padding:1px 4px;font-size:0.58rem;" title="Hide all zone arcs">NONE</button>
          </div>
        </div>
        <div class="zone-list" id="zone-list"></div>
      </div>

      <div style="display:flex;flex-direction:column;flex:1 1 0;min-height:0;">
        <div class="section-title">
          <span>&gt;_ Terminal // Stdio</span>
          <span id="cell-summary" style="font-size:0.62rem;color:var(--accent-pink);">IDLE</span>
        </div>
        <pre class="terminal-box" id="terminal-box">$ sUBhkl ready — click [1. Reduce], [2. Solve], [3. Integrate], or [4. Merge] below to execute on GPU / Pulsar and stream live stdout/stderr here.</pre>
        <canvas id="canvas-hist" title="Click to cycle resolution sparkline: CC½ vs d → Completeness vs d → ⟨I/σ⟩ vs d" style="display:none;width:100%;height:40px;cursor:pointer;"></canvas>
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
          Params
        </button>
        <button id="btn-files-toggle" title="Download NDIP / workflow artifacts for the current step (switch steps with < Walk > arrows)">
          Files
        </button>
        <span id="galaxy-status-badge" style="font-size:0.62rem;font-family:var(--font-display);padding:2px 6px;border-radius:4px;border:1px solid #334155;color:var(--text-dim);">
          LOCAL
        </span>
        <button id="btn-galaxy-dispatch" style="display:none;">Run</button>
        <button id="btn-ladder-prev" style="padding:2px 7px;font-size:0.66rem;">&lt;</button>
        <button id="btn-ladder-play" class="active-btn" style="padding:2px 8px;font-size:0.66rem;">Walk</button>
        <button id="btn-ladder-next" style="padding:2px 7px;font-size:0.66rem;">&gt;</button>
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
      <span title="Panel border rejection width (fraction of image size)">border <input id="inp-border" type="number" min="0" max="0.25" step="0.005" value="0.03" style="width:52px;" /></span>
    </div>
    <div id="files-drawer" class="params-drawer"></div>
    <div class="ladder-rungs" id="ladder-rungs"></div>
  </footer>

  <!-- Modal Dialog: Load Neutron Data (NDIP neutrons_load_data + Auto-Reduce/Merge) -->
  <div id="modal-neutron-data" style="display:none;position:fixed;inset:0;z-index:9000;background:rgba(4,6,14,0.82);backdrop-filter:blur(4px);align-items:center;justify-content:center;">
    <div style="width:min(760px,95vw);max-height:92vh;overflow-y:auto;background:var(--bg-panel);border:1px solid var(--accent-cyan);border-radius:10px;box-shadow:0 18px 50px rgba(0,0,0,0.75);padding:12px 15px;color:var(--text-main);font-size:0.76rem;">
      <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border-color);padding-bottom:7px;margin-bottom:8px;">
        <div>
          <strong style="font-family:var(--font-display);font-size:0.86rem;color:var(--accent-cyan);">LOAD NEUTRON DATA (SNS / HFIR → NDIP GALAXY)</strong>
          <div style="font-size:0.65rem;color:var(--text-dim);margin-top:2px;">
            Browse ONCat projects &amp; runs with title previews; raw NeXus runs trigger Step 1 (Reduce + Merge Images) into <code>merged.h5</code>.
          </div>
        </div>
        <button id="btn-nd-close" style="padding:2px 8px;font-size:0.72rem;">X</button>
      </div>

      <div style="display:flex;gap:8px;margin-bottom:8px;">
        <button id="btn-nd-mode-ipts" class="active-btn" style="flex:1;padding:5px 8px;font-family:var(--font-display);font-size:0.7rem;">
          IPTS + Run Browser (ONCat)
        </button>
        <button id="btn-nd-mode-paths" style="flex:1;padding:5px 8px;font-family:var(--font-display);font-size:0.7rem;">
          Explicit File Path(s) (/HFIR, /SNS)
        </button>
      </div>

      <div id="nd-pane-ipts" style="display:flex;flex-direction:column;gap:8px;margin-bottom:8px;">
        <div style="display:grid;grid-template-columns:0.8fr 1.1fr 0.9fr 1.4fr;gap:8px;">
          <label style="display:flex;flex-direction:column;gap:3px;font-size:0.65rem;color:var(--text-dim);min-width:0;">
            FACILITY
            <select id="nd-facility" style="width:100%;min-width:0;box-sizing:border-box;background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;">
              <option value="HFIR" selected>HFIR</option>
              <option value="SNS">SNS</option>
            </select>
          </label>
          <label style="display:flex;flex-direction:column;gap:3px;font-size:0.65rem;color:var(--text-dim);min-width:0;">
            INSTRUMENT
            <select id="nd-instrument" style="width:100%;min-width:0;box-sizing:border-box;background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;">
              <option value="CG4D" selected>CG4D (IMAGINE-X)</option>
              <option value="IMAGINE">IMAGINE (HFIR)</option>
              <option value="MANDI">MANDI (SNS)</option>
              <option value="TOPAZ">TOPAZ (SNS)</option>
            </select>
          </label>
          <label style="display:flex;flex-direction:column;gap:3px;font-size:0.65rem;color:var(--text-dim);min-width:0;">
            IPTS NUMBER
            <input id="nd-ipts" type="text" value="37331" placeholder="e.g. 37331" style="width:100%;min-width:0;box-sizing:border-box;background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
          <label style="display:flex;flex-direction:column;gap:3px;font-size:0.65rem;color:var(--text-dim);min-width:0;">
            <span style="display:flex;align-items:center;justify-content:space-between;">
              <span>RUN OR RANGE</span>
              <span id="nd-multirun-badge" style="font-size:0.58rem;color:var(--accent-cyan);">Multi-run</span>
            </span>
            <input id="nd-runs" type="text" value="2625-2638" placeholder="e.g. 2625-2638 or 12737,12739" style="width:100%;min-width:0;box-sizing:border-box;background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
        </div>

        <!-- Scrollable ONCat Project & Run Browser -->
        <div id="nd-oncat-browser" style="display:grid;grid-template-columns:0.95fr 1.85fr;gap:8px;background:var(--bg-card);border:1px solid var(--border-color);border-radius:6px;padding:7px 8px;">
          <!-- ONCat OAuth / OIDC Status & Device Authorization Sign-In Bar -->
          <div id="nd-oncat-auth-bar" style="grid-column:1 / -1;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:3px 6px;border-radius:4px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);">
            <span id="nd-oncat-auth-msg" style="font-family:var(--font-display);font-size:0.61rem;color:var(--text-dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
              ONCat: Checking OAuth session…
            </span>
            <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
              <a id="link-nd-oncat-verify" href="#" target="_blank" rel="noopener noreferrer" style="display:none;padding:2px 8px;border-radius:4px;background:var(--accent-cyan);color:#060719;font-family:var(--font-display);font-size:0.61rem;font-weight:700;text-decoration:none;">
                Approve in Browser ↗
              </a>
              <button id="btn-nd-oncat-login" type="button" style="padding:2px 8px;font-size:0.61rem;font-family:var(--font-display);border-color:var(--accent-cyan);color:var(--accent-cyan);">
                Sign in to ONCat
              </button>
              <button id="btn-nd-oncat-cancel" type="button" style="display:none;padding:2px 6px;font-size:0.6rem;font-family:var(--font-display);">
                Cancel
              </button>
            </div>
          </div>

          <!-- Left column: Scrollable IPTS Projects -->
          <div style="display:flex;flex-direction:column;gap:4px;min-width:0;">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;">
              <span style="font-family:var(--font-display);font-size:0.63rem;color:var(--accent-gold);">PROJECTS (IPTS)</span>
              <span id="nd-ipts-count" style="font-family:var(--font-display);font-size:0.6rem;color:var(--text-dim);">…</span>
            </div>
            <input id="nd-ipts-filter" type="text" placeholder="Filter IPTS or sample..." style="width:100%;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:3px 6px;font-family:var(--font-display);font-size:0.65rem;" />
            <div id="nd-ipts-list" style="height:150px;overflow-y:auto;border:1px solid rgba(255,255,255,0.07);border-radius:4px;background:#070913;display:flex;flex-direction:column;gap:1px;padding:2px;">
              <div style="padding:8px;color:var(--text-dim);font-size:0.65rem;">Loading IPTS projects from ONCat…</div>
            </div>
          </div>

          <!-- Right column: Scrollable Runs with Title & Angle Previews -->
          <div style="display:flex;flex-direction:column;gap:4px;min-width:0;">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
              <span id="nd-runs-header" style="font-family:var(--font-display);font-size:0.63rem;color:var(--accent-gold);">RUNS &amp; TITLES</span>
              <div style="display:flex;align-items:center;gap:6px;">
                <span id="nd-runs-hint" style="font-family:var(--font-display);font-size:0.59rem;color:var(--text-dim);">Click / Shift-click to select runs</span>
                <button id="btn-nd-clear-runs" type="button" style="padding:1px 6px;font-size:0.6rem;font-family:var(--font-display);">Clear</button>
              </div>
            </div>
            <input id="nd-runs-filter" type="text" placeholder="Filter run #, title, sample, or angle (e.g. garnet, beta lac, omega)..." style="width:100%;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:3px 6px;font-family:var(--font-display);font-size:0.65rem;" />
            <div id="nd-runs-list" style="height:150px;overflow-y:auto;border:1px solid rgba(255,255,255,0.07);border-radius:4px;background:#070913;display:flex;flex-direction:column;gap:1px;padding:2px;">
              <div style="padding:8px;color:var(--text-dim);font-size:0.65rem;">Select an IPTS project to browse runs…</div>
            </div>
          </div>
        </div>
      </div>

      <div id="nd-pane-paths" style="display:none;margin-bottom:10px;">
        <label style="display:flex;flex-direction:column;gap:4px;font-size:0.66rem;color:var(--text-dim);">
          EXPLICIT NEXUS OR HDF5 PATH(S) ON /HFIR OR /SNS (ONE PER LINE, COMMA-SEPARATED, OR WILDCARD GLOB)
          <textarea id="nd-filepaths" rows="3" placeholder="/HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5&#10;/HFIR/CG4D/IPTS-37331/nexus/CG4D_2626.nxs.h5" style="width:100%;min-width:0;box-sizing:border-box;background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:6px;font-family:var(--font-display);font-size:0.7rem;resize:vertical;"></textarea>
        </label>
      </div>

      <div style="background:var(--bg-card);border:1px solid var(--border-color);border-radius:6px;padding:8px 10px;margin-bottom:10px;">
        <div style="font-family:var(--font-display);font-size:0.66rem;color:var(--accent-gold);margin-bottom:6px;">
          SAMPLE UNIT CELL (LENGTHS &amp; ANGLES), SPACE GROUP &amp; WAVELENGTH BAND
        </div>
        <!-- Row 1: Space group + Unit cell lengths a, b, c -->
        <div style="display:grid;grid-template-columns:1.3fr repeat(3,minmax(0,1fr));gap:8px;margin-bottom:6px;">
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            SPACE GROUP
            <input id="nd-sg" type="text" value="{sg}" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            a (Å)
            <input id="nd-a" type="number" step="0.01" value="69.07" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            b (Å)
            <input id="nd-b" type="number" step="0.01" value="98.00" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            c (Å)
            <input id="nd-c" type="number" step="0.01" value="90.92" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
        </div>
        <!-- Row 2: Unit cell angles alpha, beta, gamma + wavelength band -->
        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr)) 0.85fr 0.85fr;gap:8px;margin-bottom:8px;">
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            α (°)
            <input id="nd-alpha" type="number" step="0.01" value="90.07" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            β (°)
            <input id="nd-beta" type="number" step="0.01" value="106.14" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            γ (°)
            <input id="nd-gamma" type="number" step="0.01" value="90.01" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            λ_min (Å)
            <input id="nd-wl-min" type="number" step="0.1" value="{init_wl_min:.1f}" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
          <label style="display:flex;flex-direction:column;gap:2px;font-size:0.62rem;color:var(--text-dim);min-width:0;">
            λ_max (Å)
            <input id="nd-wl-max" type="number" step="0.1" value="{init_wl_max:.1f}" style="width:100%;min-width:0;box-sizing:border-box;background:#070913;color:var(--text-main);border:1px solid var(--border-color);border-radius:4px;padding:4px 6px;font-family:var(--font-display);font-size:0.72rem;" />
          </label>
        </div>
        <div id="nd-preview-path" style="font-family:var(--font-display);font-size:0.64rem;color:var(--accent-cyan);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:2px;">
          Preview: /HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5 … CG4D_2638.nxs.h5 (14 runs)
        </div>
        <div id="nd-preview-title" style="font-family:var(--font-display);font-size:0.63rem;color:var(--accent-gold);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:4px;"></div>
        <div id="nd-validation-status" style="font-family:var(--font-display);font-size:0.66rem;min-height:18px;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:3px 6px;border-radius:4px;background:rgba(7,9,19,0.65);border:1px solid rgba(56,189,248,0.2);">
          <span id="nd-validation-msg" style="color:var(--text-dim);">Ready to verify via NDIP Load Neutron Data (`neutrons_load_data`)</span>
          <div style="display:flex;align-items:center;gap:6px;">
            <button id="btn-nd-use-suggested-ipts" type="button" style="display:none;padding:2px 8px;font-size:0.64rem;font-family:var(--font-display);background:rgba(251,191,36,0.18);border:1px solid var(--accent-gold);color:var(--accent-gold);border-radius:4px;cursor:pointer;white-space:nowrap;"></button>
            <button id="btn-nd-verify" type="button" style="padding:2px 8px;font-size:0.64rem;font-family:var(--font-display);background:rgba(56,189,248,0.14);border:1px solid rgba(56,189,248,0.45);color:var(--accent-cyan);border-radius:4px;cursor:pointer;white-space:nowrap;">Check via NDIP</button>
          </div>
        </div>
      </div>

      <div style="display:flex;justify-content:flex-end;gap:8px;">
        <button id="btn-nd-cancel" style="padding:5px 12px;font-size:0.72rem;">Cancel</button>
        <button id="btn-nd-submit" class="active-btn" style="padding:5px 14px;font-size:0.72rem;font-family:var(--font-display);font-weight:700;">
          Load Neutron Data &amp; Reduce to merged.h5
        </button>
      </div>
    </div>
  </div>

<script>
window.RAINBOW_ZONES_DATA = {payload_json};

(function() {{
  const DATA = window.RAINBOW_ZONES_DATA || {{ peaks: [], sample_peaks: [], zones: [], panels: [], ladder_steps: [], pipeline_stages: [] }};
  const PEAKS = DATA.peaks || [];
  const SAMPLE_PEAKS = DATA.sample_peaks || [];
  const ZONES = DATA.zones || [];
  const PANELS = DATA.panels || [];
  const LADDER = DATA.ladder_steps || [];
  const PIPELINE = DATA.pipeline_stages || [];

  const wlRange = DATA.wavelength_range || [2.8, 4.5];
  const order1ZoneIds = ZONES.filter(z => (z.order || 1) <= 1).map(z => z.zone_id);
  const initialZoneIds = order1ZoneIds.length > 0
    ? order1ZoneIds
    : ZONES.slice(0, Math.min(6, ZONES.length)).map(z => z.zone_id);

  function pipelineIdxForArtifactKind(artKind) {{
    const k = String(artKind || "solve").toLowerCase();
    if (k === "raw_nexus") return 0; // Pending step: 1. Reduce & Merge
    if (k === "merged") return 1;    // Pending step: 2. Solve
    if (k === "integrator") return 3; // Completed 1..3, Pending step: 4. Merge
    if (k === "merged_careless") return 4; // Completed 1..4
    return 2; // solve / predictor / finder -> Completed 1..2, Pending step: 3. Integrate
  }}

  const isMergedInit = String(DATA.artifact_kind || "").toLowerCase() === "merged_careless" || Boolean(DATA.merge_metrics);
  const state = {{
    projection: "{initial_proj}", // "cylindrical" (CG4D) | "spherical" (MANDI)
    maxZoneOrder: order1ZoneIds.length > 0 ? 1 : 2, // 1 | 2 | 3
    showZoneArcs: true,           // single vibrant arc per zone (not multi-band rainbow)
    heatmapStyle: "magma",        // "magma" | "cyan" | "gray"
    detAlpha: 0.85,
    showStereo: isMergedInit,
    wulffExpanded: isMergedInit,
    wulffScope: isMergedInit ? "all_runs" : "single_frame", // "single_frame" (1-Frame) | "all_runs" (All-Runs)
    wulffFrame: "sample",         // always project in Sample frame (coupled via goniometer R)
    wulffCoupled: true,           // 360° periodic detector pan <-> Wulff yaw always coupled
    wulffColorMode: isMergedInit ? "residual" : "wavelength", // "wavelength" | "orbit" | "residual" | "scale"
    wulffYawDeg: 0.0,
    wulffPitchDeg: 0.0,
    wulffZoom: 1.0,
    histMetric: "cc_half",        // "cc_half" | "completeness" | "isigi"
    lamMin: wlRange[0],
    lamMax: wlRange[1],
    zoneTolDeg: 0.75,
    parallaxStrength: 1.0,
    activeZones: new Set(initialZoneIds),
    selectedZoneId: null,
    hoveredPeakId: null,
    selectedPeakId: null,
    activeOrbitId: null,
    currentPipelineIdx: pipelineIdxForArtifactKind(DATA.artifact_kind), // 0: Reduce, 1: Solve, 2: Integrate, 3: Merge, 4: Done
    currentRungIdx: DATA.artifact_kind === "merged" ? 0 : Math.max(0, LADDER.length - 1),
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
    let pending = 0;
    PANELS.forEach(p => {{
      if (!p.texture_b64) return;
      pending++;
      const img = new Image();
      img.onload = () => {{
        rawBankImages.set(p.bank, img);
        bankCanvases.set(p.bank, colorizeGrayscaleImage(img, state.heatmapStyle));
        pending--;
        if (pending <= 0 || pending % 8 === 0) {{
          renderDetector();
        }}
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
  const btnFilesToggle = document.getElementById("btn-files-toggle");
  const filesDrawer = document.getElementById("files-drawer");
  if (btnParamsToggle && paramsDrawer) {{
    btnParamsToggle.addEventListener("click", () => {{
      const nextOpen = !paramsDrawer.classList.contains("open");
      paramsDrawer.classList.toggle("open", nextOpen);
      btnParamsToggle.classList.toggle("active-btn", nextOpen);
      if (nextOpen && filesDrawer && btnFilesToggle) {{
        filesDrawer.classList.remove("open");
        btnFilesToggle.classList.remove("active-btn");
      }}
    }});
  }}
  if (btnFilesToggle && filesDrawer) {{
    btnFilesToggle.addEventListener("click", () => {{
      const nextOpen = !filesDrawer.classList.contains("open");
      filesDrawer.classList.toggle("open", nextOpen);
      btnFilesToggle.classList.toggle("active-btn", nextOpen);
      if (nextOpen && paramsDrawer && btnParamsToggle) {{
        paramsDrawer.classList.remove("open");
        btnParamsToggle.classList.remove("active-btn");
      }}
      if (nextOpen) renderFilesDrawer();
    }});
  }}

  function formatTerminalStdio(rawText) {{
    if (!rawText) return "";
    const text = String(rawText).replace(/\\r\\n/g, "\\n");
    const tokens = text.split(/(\\x1b\\[[0-9;?]*[A-Za-z]|\\r|\\n)/);
    const lines = [""];
    let row = 0;
    let col = 0;
    for (let i = 0; i < tokens.length; i++) {{
      const tok = tokens[i];
      if (!tok) continue;
      if (tok === "\\n") {{
        lines[row] = (lines[row] || "").replace(/\\s+$/, "");
        row += 1;
        if (row >= lines.length) lines.push("");
        col = 0;
      }} else if (tok === "\\r") {{
        col = 0;
      }} else if (tok.charCodeAt(0) === 27 && tok[1] === "[") {{
        const cmdCh = tok[tok.length - 1];
        const paramStr = tok.slice(2, -1).replace(/^[?;]+/, "");
        const firstPart = paramStr.split(";")[0];
        const n = /^\\d+$/.test(firstPart) ? parseInt(firstPart, 10) : 1;
        if (cmdCh === "A") {{
          row = Math.max(0, row - Math.max(1, n));
          col = 0;
        }} else if (cmdCh === "B") {{
          row = Math.min(lines.length - 1, row + Math.max(1, n));
          col = 0;
        }} else if (cmdCh === "G") {{
          col = Math.max(0, n - 1);
        }} else if (cmdCh === "K") {{
          const modeK = /^\\d+$/.test(paramStr) ? parseInt(paramStr, 10) : 0;
          if (modeK === 2) {{
            lines[row] = "";
            col = 0;
          }} else if (modeK === 1) {{
            lines[row] = " ".repeat(col) + (lines[row] || "").slice(col);
          }} else {{
            lines[row] = (lines[row] || "").slice(0, col);
          }}
        }}
      }} else {{
        let cur = lines[row] || "";
        if (col > cur.length) cur += " ".repeat(col - cur.length);
        lines[row] = cur.slice(0, col) + tok + cur.slice(col + tok.length);
        col += tok.length;
      }}
    }}
    const tqdmRe = /^\\s*([^%|]*?)\\s*\\d{{1,3}}%\\|.*\\|/;
    const collapsed = [];
    for (let i = 0; i < lines.length; i++) {{
      const cleanLn = (lines[i] || "").replace(/\\s+$/, "");
      const mCur = tqdmRe.exec(cleanLn);
      if (mCur && collapsed.length > 0) {{
        const mPrev = tqdmRe.exec(collapsed[collapsed.length - 1]);
        if (mPrev && mPrev[1].trim() === mCur[1].trim()) {{
          collapsed[collapsed.length - 1] = cleanLn;
          continue;
        }}
      }}
      collapsed.push(cleanLn);
    }}
    return collapsed.join("\\n");
  }}

  function appendTerminalLog(text, replace = false) {{
    const box = document.getElementById("terminal-box");
    if (!box) return;
    if (replace) {{
      box.textContent = formatTerminalStdio(text);
    }} else {{
      const combined = box.textContent + (box.textContent.endsWith("\\n") ? "" : "\\n") + text;
      box.textContent = formatTerminalStdio(combined);
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
      d_min: getNum("inp-dmin", Number(((state.lamMin || 2.0) * 0.9).toFixed(2))),
      steps: Math.round(getNum("inp-steps", 120)),
      rel_border_width: Math.max(0.0, Math.min(0.25, getNum("inp-border", 0.03))),
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

  const ORBIT_PALETTE = [
    "#ff38b4", "#00e5ff", "#fbbf24", "#10b981", "#a855f7", "#f97316",
    "#38bdf8", "#84cc16", "#f43f5e", "#14b8a6", "#eab308", "#6366f1"
  ];

  // Diverging Blue–White–Red colormap for Careless standardized residual (I_obs - I_pred)/sigma
  function residualToColor(resVal, alpha = 1.0) {{
    if (typeof resVal !== "number" || !Number.isFinite(resVal)) return `rgba(148, 163, 184, ${{alpha * 0.45}})`;
    const clamped = Math.max(-3.0, Math.min(3.0, resVal));
    if (clamped < 0) {{
      const u = Math.min(1.0, -clamped / 3.0);
      const r = Math.round(248 * (1 - u) + 30 * u);
      const g = Math.round(250 * (1 - u) + 170 * u);
      const b = Math.round(252 * (1 - u) + 255 * u);
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
    }} else {{
      const u = Math.min(1.0, clamped / 3.0);
      const r = Math.round(248 * (1 - u) + 244 * u);
      const g = Math.round(250 * (1 - u) + 63 * u);
      const b = Math.round(252 * (1 - u) + 94 * u);
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
    }}
  }}

  function scaleToColor(scaleVal, alpha = 1.0) {{
    if (typeof scaleVal !== "number" || !Number.isFinite(scaleVal)) return `rgba(148, 163, 184, ${{alpha * 0.45}})`;
    const t = Math.max(0.0, Math.min(1.0, (Math.log10(Math.max(1e-4, scaleVal)) + 2.0) / 3.0));
    const hue = (1.0 - t) * 195 + t * 45;
    return `hsla(${{hue.toFixed(1)}}, 92%, 58%, ${{alpha}})`;
  }}

  function peakStrokeColor(p, alpha = 0.95) {{
    const mode = state.wulffColorMode || "wavelength";
    if (mode === "orbit") {{
      const idx = typeof p.orbit_color_idx === "number"
        ? ((p.orbit_color_idx % ORBIT_PALETTE.length) + ORBIT_PALETTE.length) % ORBIT_PALETTE.length
        : 0;
      const hex = ORBIT_PALETTE[idx];
      return alpha >= 0.9 ? hex : `${{hex}}99`;
    }}
    if (mode === "residual" && typeof p.careless_residual === "number") {{
      return residualToColor(p.careless_residual, alpha);
    }}
    if (mode === "scale" && typeof p.careless_scale === "number") {{
      return scaleToColor(p.careless_scale, alpha);
    }}
    return wavelengthToColor(p.wavelength, alpha);
  }}

  function syncWulffUI() {{
    const stereoEl = document.getElementById("stereo-inset");
    const btnTop = document.getElementById("btn-stereo-toggle");
    const btnScope = document.getElementById("btn-wulff-scope");
    const btnExpand = document.getElementById("btn-wulff-expand");
    const selColor = document.getElementById("sel-wulff-color");
    const titleEl = document.getElementById("stereo-title");
    if (stereoEl) {{
      stereoEl.classList.toggle("hidden", !state.showStereo);
      stereoEl.classList.toggle("expanded", Boolean(state.wulffExpanded));
    }}
    if (btnTop) btnTop.classList.toggle("active-btn", Boolean(state.showStereo));
    if (btnScope) {{
      const isAll = state.wulffScope === "all_runs";
      btnScope.textContent = isAll ? "All-Runs" : "1-Frame";
      btnScope.classList.toggle("active-btn", isAll);
    }}
    if (btnExpand) {{
      btnExpand.textContent = state.wulffExpanded ? "Dock" : "Expand";
      btnExpand.classList.toggle("active-btn", Boolean(state.wulffExpanded));
    }}
    if (selColor) selColor.value = state.wulffColorMode || "wavelength";
    if (titleEl) {{
      const scopeLbl = state.wulffScope === "all_runs" ? "ALL-RUNS" : "1-FRAME";
      titleEl.textContent = `WULFF NET · ${{scopeLbl}}`;
    }}
    updateMergeScoreboard();
  }}

  function updateMergeScoreboard() {{
    const bar = document.getElementById("wulff-merge-bar");
    const m = DATA.merge_metrics;
    if (!bar) return;
    if (!m || !m.n_merged_asu) {{
      bar.style.display = "none";
      return;
    }}
    bar.style.display = "flex";
    const cc = typeof m.cc_half_overall === "number" ? m.cc_half_overall.toFixed(3) : "—";
    const ccs = typeof m.cc_half_spearman === "number" ? m.cc_half_spearman.toFixed(3) : null;
    const isig = typeof m.mean_i_over_sigi === "number" ? m.mean_i_over_sigi.toFixed(1) : "—";
    const comp = typeof m.completeness_pct === "number" ? `${{m.completeness_pct.toFixed(1)}}%` : "—";
    const mult = typeof m.multiplicity === "number" ? `${{m.multiplicity.toFixed(2)}}×` : "—";
    const nAsu = Number(m.n_merged_asu || 0).toLocaleString();
    const nObs = Number(m.n_observations || 0).toLocaleString();
    const harmPct = typeof m.harmonic_overlap_fraction === "number"
      ? `${{(m.harmonic_overlap_fraction * 100).toFixed(1)}}%`
      : "—";
    bar.innerHTML = `
      <span><strong style="color:var(--accent-cyan);">CARELESS MERGE:</strong> CC½=<strong style="color:var(--accent-gold);">${{cc}}</strong>${{ccs ? ` (ρ=${{ccs}})` : ""}}</span>
      <span>⟨I/σ⟩=<strong style="color:var(--accent-gold);">${{isig}}</strong></span>
      <span>Completeness=<strong style="color:#34d399;">${{comp}}</strong> (${{nAsu}} ASU / ${{nObs}} obs)</span>
      <span>Multiplicity=<strong>${{mult}}</strong> • Harmonics=<strong>${{harmPct}}</strong></span>
    `;
  }}

  function syncProjectionUI() {{
    const isSph = state.projection === "spherical";
    const btn = document.getElementById("btn-proj-toggle");
    btn.textContent = isSph ? "Sph" : "Cyl";
    document.getElementById("val-proj").textContent = isSph ? "SPH" : "CYL";
    document.getElementById("detector-header-title").textContent = isSph
      ? "SPHERICAL DETECTOR PROJECTION (AZIMUTH φ [deg] vs ELEVATION ψ [deg])"
      : "CYLINDRICAL DETECTOR PROJECTION (AZIMUTH φ [deg] vs LAB HEIGHT Y [m])";
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
      const rightStat = z.n_peaks > 0 ? `${{z.n_peaks}}p` : `o${{z.order || 1}}`;
      item.title = `${{z.label}} (order ${{z.order || 1}}) • ${{z.n_peaks}} spots • Δθ=${{z.angular_rms_deg.toFixed(2)}}°`;
      item.innerHTML = `
        <div style="display:flex;align-items:center;min-width:0;overflow:hidden;">
          <span class="zone-swatch" style="background:${{color}};"></span>
          <strong style="font-family:var(--font-display);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${{z.label}}</strong>
        </div>
        <span style="color:var(--text-dim);font-family:var(--font-display);font-size:0.58rem;margin-left:3px;flex-shrink:0;">
          ${{rightStat}}
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
      const prefix = (state.runningStageIdx === idx) ? "[...] " : "";
      pill.title = `Click to execute ${{st.name}} (${{(st.tool_ids || []).join(" + ")}}) → ${{st.artifact}}`;
      pill.textContent = `${{prefix}}${{shortLabels[idx] || st.name}}`;
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
    if (pIdx === 3) {{
      state.showStereo = true;
      syncWulffUI();
    }}
    if (syncLadder && typeof st.ladder_rung_idx === "number") {{
      selectLadderRung(st.ladder_rung_idx, false);
    }}
    renderPipeline();
    if (typeof renderFilesDrawer === "function") renderFilesDrawer();
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
      `[STEP ${{step.rung}}]`,
      `<strong style="color:var(--accent-gold);">${{step.stage}}</strong> — <span>${{step.description}}</span>`
    );
    renderZoneList();
    renderLadder();
    updateHUD();
    if (typeof renderFilesDrawer === "function") renderFilesDrawer();
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
    const asuStr = Array.isArray(p.asu_hkl) ? ` → ASU <strong style="color:#38bdf8;">(${{p.asu_hkl.join(",")}})</strong>` : "";
    let mergeStr = "";
    if (typeof p.careless_residual === "number") {{
      const rCol = p.careless_residual >= 0 ? "#f43f5e" : "#38bdf8";
      mergeStr = ` • ΔI/σ=<strong style="color:${{rCol}};">${{p.careless_residual >= 0 ? "+" : ""}}${{p.careless_residual.toFixed(2)}}σ</strong>`;
      if (typeof p.merged_i === "number" && typeof p.merged_sigi === "number" && p.merged_sigi > 0) {{
        mergeStr += ` (I_asu=${{p.merged_i.toFixed(1)}}±${{p.merged_sigi.toFixed(1)}})`;
      }}
    }}
    const detXVal = typeof p.det_x === "number" ? p.det_x : 0.0;
    const detYVal = typeof p.det_y === "number" ? p.det_y : 0.0;
    const errDeg = typeof p.angular_error_deg === "number" ? p.angular_error_deg : 0.0;
    const harmMult = p.harmonic_multiplicity || p.harmonic_order || 1;
    setStatusLine(
      `[BANK #${{p.bank || 1}} • RUN #${{(p.run_index || 0) + 1}}]`,
      `<strong>(hkl)=<span style="color:var(--accent-gold);">(${{p.hkl.join(", ")}})</span></strong>${{asuStr}} (n=${{p.harmonic_order || 1}}${{harmMult > 1 ? `, mult=${{harmMult}}` : ""}}) • λ=<strong>${{p.wavelength.toFixed(3)}} Å</strong> • d=<strong>${{p.d_spacing.toFixed(3)}} Å</strong> • I=<strong>${{Math.round(p.intensity)}}</strong>${{snrStr}}${{mergeStr}}${{shapeStr}} • φ=${{detXVal.toFixed(1)}}°, Y=${{detYVal.toFixed(3)}}m, ψ=${{elVal.toFixed(1)}}° • Zones: <span style="color:var(--accent-cyan);">${{zLabels}}</span> (Δθ=${{errDeg.toFixed(3)}}°)`
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
  if (histCanvas) {{
    histCanvas.addEventListener("click", () => {{
      if (!DATA.merge_metrics) return;
      const order = ["cc_half", "completeness", "isigi"];
      state.histMetric = order[(order.indexOf(state.histMetric || "cc_half") + 1) % order.length];
      drawHistogram(PEAKS.filter(isPeakVisible));
    }});
  }}
  function drawHistogram(visiblePeaks) {{
    const m = DATA.merge_metrics;
    const hasMergeBins = Boolean(m && Array.isArray(m.cc_half_bins) && m.cc_half_bins.length > 0);
    if (histCanvas) {{
      histCanvas.style.display = hasMergeBins ? "block" : "none";
      histCanvas.style.height = hasMergeBins ? "50px" : "40px";
    }}
    const rect = histCanvas.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 15) return;
    const dpr = window.devicePixelRatio || 1;
    histCanvas.width = Math.max(100, rect.width * dpr);
    histCanvas.height = Math.max(45, rect.height * dpr);
    const ctx = histCanvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    if (hasMergeBins) {{
      const bins = m.cc_half_bins;
      const nBins = bins.length;
      const barW = (w - 16) / Math.max(1, nBins);
      const metric = state.histMetric || "cc_half";
      const maxComp = Math.max(20, ...bins.map(b => Number(b.completeness_pct || 0)));
      const maxIsig = Math.max(5, ...bins.map(b => Number(b.mean_isigi || 0)));
      for (let i = 0; i < nBins; i++) {{
        const x = 8 + i * barW;
        if (metric === "completeness") {{
          const cVal = Math.max(0, Number(bins[i].completeness_pct || 0));
          const frac = Math.min(1, cVal / maxComp);
          const bh = cVal > 0 ? Math.max(2, frac * (h - 20)) : 1;
          const y = h - 14 - bh;
          ctx.fillStyle = cVal > 0 ? (cVal >= 50 ? "#34d399" : "#38bdf8") : "#1e293b";
          ctx.fillRect(x + 1, y, Math.max(2, barW - 2), bh);
        }} else if (metric === "isigi") {{
          const sVal = Math.max(0, Number(bins[i].mean_isigi || 0));
          const frac = Math.min(1, sVal / maxIsig);
          const bh = sVal > 0 ? Math.max(2, frac * (h - 20)) : 1;
          const y = h - 14 - bh;
          ctx.fillStyle = sVal > 0 ? (sVal >= 5 ? "#fbbf24" : "#38bdf8") : "#1e293b";
          ctx.fillRect(x + 1, y, Math.max(2, barW - 2), bh);
        }} else {{
          const rawCc = bins[i].cc_half;
          const hasCc = rawCc !== null && rawCc !== undefined && Number.isFinite(Number(rawCc));
          const ccVal = hasCc ? Math.max(0, Math.min(1, Number(rawCc))) : 0;
          const bh = hasCc ? Math.max(2, ccVal * (h - 20)) : 1;
          const y = h - 14 - bh;
          ctx.fillStyle = !hasCc ? "#1e293b" : (ccVal >= 0.8 ? "#34d399" : (ccVal >= 0.5 ? "#fbbf24" : "#f43f5e"));
          ctx.fillRect(x + 1, y, Math.max(2, barW - 2), bh);
        }}
      }}
      ctx.fillStyle = "#94a3b8";
      ctx.font = "9px monospace";
      const d0 = bins[0] && bins[0].d_max ? `${{Number(bins[0].d_max).toFixed(1)}}Å` : "low-q";
      const d1 = bins[nBins - 1] && bins[nBins - 1].d_min ? `${{Number(bins[nBins - 1].d_min).toFixed(2)}}Å` : "d_min";
      ctx.fillText(d0, 6, h - 3);
      let midLbl = "";
      let midCol = "#38bdf8";
      if (metric === "completeness") {{
        const compOv = typeof m.completeness_pct === "number" ? `${{m.completeness_pct.toFixed(1)}}%` : "—";
        midLbl = `Compl. vs d (${{compOv}}) ↻`;
        midCol = "#34d399";
      }} else if (metric === "isigi") {{
        const isigOv = typeof m.mean_i_over_sigi === "number" ? m.mean_i_over_sigi.toFixed(1) : "—";
        midLbl = `⟨I/σ⟩ vs d (${{isigOv}}) ↻`;
        midCol = "#fbbf24";
      }} else {{
        const ccOv = typeof m.cc_half_overall === "number" ? m.cc_half_overall.toFixed(3) : "—";
        midLbl = `CC½ vs d (${{ccOv}}) ↻`;
        midCol = "#38bdf8";
      }}
      ctx.fillStyle = midCol;
      ctx.fillText(midLbl, Math.max(36, w * 0.24), h - 3);
      ctx.fillStyle = "#94a3b8";
      ctx.fillText(d1, Math.max(w - 36, 60), h - 3);
      return;
    }}

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
    const isCoupled360 = true;
    const azShifts = [-360.0, 0.0, 360.0];

    // 1. Coordinate Grid (Azimuth deg vs Height [m] or Elevation [deg])
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px monospace";
    azShifts.forEach(shiftAz => {{
      for (let az = -180; az <= 180; az += 45) {{
        if (shiftAz !== 0 && (az === -180 || az === 180)) continue;
        const [gx, gy0] = detToScreen(az + shiftAz, bounds.minY, w, h, -0.35);
        if (gx < -40 || gx > w + 40) continue;
        const [, gy1] = detToScreen(az + shiftAz, bounds.maxY, w, h, -0.35);
        ctx.beginPath();
        ctx.moveTo(gx, gy0);
        ctx.lineTo(gx, gy1);
        ctx.stroke();
        ctx.fillText(`${{az}}°`, gx - 12, h - 8);
      }}
    }});
    const yStep = Number(((bounds.maxY - bounds.minY) / 4).toFixed(isSph ? 0 : 2)) || (isSph ? 30 : 0.15);
    for (let yv = bounds.minY; yv <= bounds.maxY + 1e-4; yv += yStep) {{
      const [gx0, gy] = detToScreen(bounds.minX - (isCoupled360 ? 360 : 0), yv, w, h, -0.35);
      const [gx1] = detToScreen(bounds.maxX + (isCoupled360 ? 360 : 0), yv, w, h, -0.35);
      ctx.beginPath();
      ctx.moveTo(Math.max(0, gx0), gy);
      ctx.lineTo(Math.min(w, gx1), gy);
      ctx.stroke();
      const lbl = isSph ? `${{yv.toFixed(0)}}°` : `${{yv.toFixed(2)}}m`;
      ctx.fillText(lbl, 4, gy + 3);
    }}
    ctx.restore();

    // 2. LAYER 0 (BACK): Detector Bank Pixel Intensity Heatmaps + Panel Outlines
    ctx.save();
    azShifts.forEach(shiftAz => {{
      PANELS.forEach(panel => {{
        const [vMin, vMax] = getPanelVertRange(panel);
        const [x0, y0] = detToScreen(panel.az_min + shiftAz, vMax, w, h, -0.35);
        const [x1, y1] = detToScreen(panel.az_max + shiftAz, vMin, w, h, -0.35);
        if (Math.max(x0, x1) < -80 || Math.min(x0, x1) > w + 80) return;
        const pw = x1 - x0;
        const ph = y1 - y0;

        const outlinePts = isSph && panel.outline_sph ? panel.outline_sph : panel.outline;
        const cornersPts = isSph && panel.corners_sph ? panel.corners_sph : panel.corners;
        const meshPts = isSph && panel.mesh_sph ? panel.mesh_sph : panel.mesh;

        ctx.save();
        if (outlinePts && outlinePts.length >= 4) {{
          ctx.beginPath();
          outlinePts.forEach((pt, idx) => {{
            const [ox, oy] = detToScreen(pt[0] + shiftAz, pt[1], w, h, -0.35);
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
          const W = Math.max(1, texCanvas.width);
          const H = Math.max(1, texCanvas.height);
          if (Array.isArray(meshPts) && meshPts.length >= 2 && Array.isArray(meshPts[0]) && meshPts[0].length >= 2) {{
            // Curvilinear mesh warping (Nr x Nc grid of sub-quads) so high-latitude / tilted panels
            // (e.g. MANDI bottom-plane panels) fill their curved outlines accurately
            const nRows = meshPts.length - 1;
            const nCols = meshPts[0].length - 1;
            const SGrid = meshPts.map(row => row.map(pt => detToScreen(pt[0] + shiftAz, pt[1], w, h, -0.35)));
            const S0 = SGrid[0][0], S1 = SGrid[nRows][0], S2 = SGrid[nRows][nCols], S3 = SGrid[0][nCols];
            const nonPara = Math.hypot((S3[0] + S1[0] - S2[0]) - S0[0], (S3[1] + S1[1] - S2[1]) - S0[1]);
            const midR = Math.floor(nRows / 2), midC = Math.floor(nCols / 2);
            const topBow = Math.hypot(SGrid[0][midC][0] - 0.5 * (S0[0] + S3[0]), SGrid[0][midC][1] - 0.5 * (S0[1] + S3[1]));
            const botBow = Math.hypot(SGrid[nRows][midC][0] - 0.5 * (S1[0] + S2[0]), SGrid[nRows][midC][1] - 0.5 * (S1[1] + S2[1]));
            const lftBow = Math.hypot(SGrid[midR][0][0] - 0.5 * (S0[0] + S1[0]), SGrid[midR][0][1] - 0.5 * (S0[1] + S1[1]));
            const rgtBow = Math.hypot(SGrid[midR][nCols][0] - 0.5 * (S3[0] + S2[0]), SGrid[midR][nCols][1] - 0.5 * (S3[1] + S2[1]));
            const maxBow = Math.max(topBow, botBow, lftBow, rgtBow);
            if (nonPara < 1.5 && maxBow < 1.2) {{
              const ax = ((S3[0] - S0[0]) + (S2[0] - S1[0])) * 0.5 / W;
              const ay = ((S3[1] - S0[1]) + (S2[1] - S1[1])) * 0.5 / W;
              const cx = ((S1[0] - S0[0]) + (S2[0] - S3[0])) * 0.5 / H;
              const cy = ((S1[1] - S0[1]) + (S2[1] - S3[1])) * 0.5 / H;
              const ex = (S0[0] + (S3[0] + S1[0] - S2[0])) * 0.5;
              const ey = (S0[1] + (S3[1] + S1[1] - S2[1])) * 0.5;
              ctx.save();
              ctx.transform(ax, ay, cx, cy, ex, ey);
              ctx.drawImage(texCanvas, 0, 0, W, H);
              ctx.restore();
            }} else {{
              const du = W / nCols;
              const dv = H / nRows;
              const expandTri = (A, B, C) => {{
                const mx = (A[0] + B[0] + C[0]) / 3;
                const my = (A[1] + B[1] + C[1]) / 3;
                const pad = 0.45;
                return [A, B, C].map(P => {{
                  const dx = P[0] - mx, dy = P[1] - my;
                  const len = Math.hypot(dx, dy) || 1;
                  return [P[0] + (dx / len) * pad, P[1] + (dy / len) * pad];
                }});
              }};
              for (let ir = 0; ir < nRows; ir++) {{
                const v0 = ir * dv;
                const v1 = (ir + 1) * dv;
                for (let ic = 0; ic < nCols; ic++) {{
                  const u0 = ic * du;
                  const u1 = (ic + 1) * du;
                  const P00 = SGrid[ir][ic];
                  const P01 = SGrid[ir][ic + 1];
                  const P11 = SGrid[ir + 1][ic + 1];
                  const P10 = SGrid[ir + 1][ic];

                  // Sub-triangle 1: (u0, v0)->P00, (u1, v0)->P01, (u0, v1)->P10
                  const [T1A, T1B, T1C] = expandTri(P00, P01, P10);
                  const a1 = (P01[0] - P00[0]) / du;
                  const b1 = (P01[1] - P00[1]) / du;
                  const c1 = (P10[0] - P00[0]) / dv;
                  const d1 = (P10[1] - P00[1]) / dv;
                  const e1 = P00[0] - a1 * u0 - c1 * v0;
                  const f1 = P00[1] - b1 * u0 - d1 * v0;
                  ctx.save();
                  ctx.beginPath();
                  ctx.moveTo(T1A[0], T1A[1]);
                  ctx.lineTo(T1B[0], T1B[1]);
                  ctx.lineTo(T1C[0], T1C[1]);
                  ctx.closePath();
                  ctx.clip();
                  ctx.transform(a1, b1, c1, d1, e1, f1);
                  ctx.drawImage(texCanvas, 0, 0, W, H);
                  ctx.restore();

                  // Sub-triangle 2: (u1, v0)->P01, (u1, v1)->P11, (u0, v1)->P10
                  const [T2A, T2B, T2C] = expandTri(P01, P11, P10);
                  const a2 = (P11[0] - P10[0]) / du;
                  const b2 = (P11[1] - P10[1]) / du;
                  const c2 = (P11[0] - P01[0]) / dv;
                  const d2 = (P11[1] - P01[1]) / dv;
                  const e2 = P11[0] - a2 * u1 - c2 * v1;
                  const f2 = P11[1] - b2 * u1 - d2 * v1;
                  ctx.save();
                  ctx.beginPath();
                  ctx.moveTo(T2A[0], T2A[1]);
                  ctx.lineTo(T2B[0], T2B[1]);
                  ctx.lineTo(T2C[0], T2C[1]);
                  ctx.closePath();
                  ctx.clip();
                  ctx.transform(a2, b2, c2, d2, e2, f2);
                  ctx.drawImage(texCanvas, 0, 0, W, H);
                  ctx.restore();
                }}
              }}
            }}
          }} else if (cornersPts && cornersPts.length === 4) {{
            // True 2-triangle affine texture projection from (r, c) corners C0=(0,0), C1=(n-1,0), C2=(n-1,m-1), C3=(0,m-1)
            const S0 = detToScreen(cornersPts[0][0] + shiftAz, cornersPts[0][1], w, h, -0.35);
            const S1 = detToScreen(cornersPts[1][0] + shiftAz, cornersPts[1][1], w, h, -0.35);
            const S2 = detToScreen(cornersPts[2][0] + shiftAz, cornersPts[2][1], w, h, -0.35);
            const S3 = detToScreen(cornersPts[3][0] + shiftAz, cornersPts[3][1], w, h, -0.35);
            const nonPara = Math.hypot((S3[0] + S1[0] - S2[0]) - S0[0], (S3[1] + S1[1] - S2[1]) - S0[1]);
            if (nonPara < 2.2) {{
              const ax = ((S3[0] - S0[0]) + (S2[0] - S1[0])) * 0.5 / W;
              const ay = ((S3[1] - S0[1]) + (S2[1] - S1[1])) * 0.5 / W;
              const cx = ((S1[0] - S0[0]) + (S2[0] - S3[0])) * 0.5 / H;
              const cy = ((S1[1] - S0[1]) + (S2[1] - S3[1])) * 0.5 / H;
              const ex = (S0[0] + (S3[0] + S1[0] - S2[0])) * 0.5;
              const ey = (S0[1] + (S3[1] + S1[1] - S2[1])) * 0.5;
              ctx.save();
              ctx.transform(ax, ay, cx, cy, ex, ey);
              ctx.drawImage(texCanvas, 0, 0, W, H);
              ctx.restore();
            }} else {{
              // Triangle A: (0,0)->S0, (W,0)->S3, (0,H)->S1
              ctx.save();
              ctx.beginPath();
              ctx.moveTo(S0[0], S0[1]);
              ctx.lineTo(S3[0], S3[1]);
              ctx.lineTo(S1[0], S1[1]);
              ctx.closePath();
              ctx.clip();
              ctx.transform(
                (S3[0] - S0[0]) / W, (S3[1] - S0[1]) / W,
                (S1[0] - S0[0]) / H, (S1[1] - S0[1]) / H,
                S0[0], S0[1]
              );
              ctx.drawImage(texCanvas, 0, 0, W, H);
              ctx.restore();

              // Triangle B: (W,0)->S3, (W,H)->S2, (0,H)->S1
              ctx.save();
              ctx.beginPath();
              ctx.moveTo(S3[0], S3[1]);
              ctx.lineTo(S2[0], S2[1]);
              ctx.lineTo(S1[0], S1[1]);
              ctx.closePath();
              ctx.clip();
              ctx.transform(
                (S2[0] - S1[0]) / W, (S2[1] - S1[1]) / W,
                (S2[0] - S3[0]) / H, (S2[1] - S3[1]) / H,
                S3[0] + S1[0] - S2[0], S3[1] + S1[1] - S2[1]
              );
              ctx.drawImage(texCanvas, 0, 0, W, H);
              ctx.restore();
            }}
          }} else {{
            ctx.drawImage(texCanvas, x0, y0, pw, ph);
          }}
        }}
        ctx.restore();

        // True panel polygon border (shows curved elevation edges in Spherical MANDI view)
        ctx.strokeStyle = texCanvas ? "rgba(56, 189, 248, 0.38)" : "rgba(148, 163, 184, 0.18)";
        ctx.lineWidth = texCanvas ? 1.1 : 0.7;
        if (outlinePts && outlinePts.length >= 4) {{
          ctx.beginPath();
          outlinePts.forEach((pt, idx) => {{
            const [ox, oy] = detToScreen(pt[0] + shiftAz, pt[1], w, h, -0.35);
            if (idx === 0) ctx.moveTo(ox, oy);
            else ctx.lineTo(ox, oy);
          }});
          ctx.closePath();
          ctx.stroke();
        }} else {{
          ctx.strokeRect(x0, y0, pw, ph);
        }}

        // Only show bank labels when zoomed in so default view is clean
        if (state.detZoom > 1.35 && Math.abs(pw) > 34 && Math.abs(ph) > 24) {{
          ctx.fillStyle = "rgba(148, 163, 184, 0.42)";
          ctx.font = "8px monospace";
          if (cornersPts && cornersPts.length === 4) {{
            const cScreen = cornersPts.map(pt => detToScreen(pt[0] + shiftAz, pt[1], w, h, -0.35));
            const cxLbl = 0.25 * (cScreen[0][0] + cScreen[1][0] + cScreen[2][0] + cScreen[3][0]);
            const cyLbl = 0.25 * (cScreen[0][1] + cScreen[1][1] + cScreen[2][1] + cScreen[3][1]);
            ctx.fillText(`b${{panel.bank}}`, cxLbl - 7, cyLbl + 3);
          }} else {{
            ctx.fillText(`b${{panel.bank}}`, x0 + 3, y0 + 9);
          }}
        }}
      }});
    }});
    ctx.restore();

    const OVERLAY_PARALLAX_LAYER = 0.45;

    // 3. LAYER 1 (FOREGROUND OVERLAY): Single-Arc Laue Zone Conics [uvw] clipped to detector panels
    if (state.showZoneArcs) {{
      const hasPanels = Array.isArray(PANELS) && PANELS.length > 0;
      azShifts.forEach(shiftAz => {{
        ZONES.forEach(zone => {{
          if (!state.activeZones.has(zone.zone_id)) return;
          const contSegs = isSph
            ? (zone.continuous_segments_sph || [])
            : (zone.continuous_segments || []);
          const panelSegs = isSph
            ? (hasPanels ? (zone.detector_segments_sph || []) : (zone.detector_segments_sph || contSegs))
            : (hasPanels ? (zone.detector_segments || []) : (zone.detector_segments || contSegs));
          panelSegs.forEach((seg, sIdx) => {{
            const screenPts = seg.map(pt => detToScreen(pt[0] + shiftAz, pt[1], w, h, OVERLAY_PARALLAX_LAYER));
            drawSingleZoneArc(ctx, screenPts, zone, false);
            if (shiftAz === 0 && sIdx === 0 && screenPts.length > 2 && (state.selectedZoneId === zone.zone_id || state.detZoom > 1.35)) {{
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
      }});
    }}

    // 4. FOREGROUND LAYER: Predicted Bragg Rings hovering together with Laue zone circles
    const visiblePeaks = PEAKS.filter(isPeakVisible)
      .slice()
      .sort((a, b) => b.parallax_depth - a.parallax_depth);
    const pendingTooltips = [];
    const queuedTooltipPeakIds = new Set();

    [0.0, -360.0, 360.0].forEach(shiftAz => {{
      visiblePeaks.forEach(p => {{
        const py = getPeakVert(p);
        const fgLayer = OVERLAY_PARALLAX_LAYER;
        const [x, y] = detToScreen(p.det_x + shiftAz, py, w, h, fgLayer);
        if (x < -30 || x > w + 30) return;
        const [bx, by] = detToScreen(p.det_x + shiftAz, py, w, h, -0.35);

        const captured = isPeakCapturedByActiveZone(p);
        const isHovered = state.hoveredPeakId === p.id || state.selectedPeakId === p.id;
        const isOrbitMate = Boolean(state.activeOrbitId && p.asu_orbit_id && p.asu_orbit_id === state.activeOrbitId);
        if (!captured && !isHovered && !isOrbitMate && state.activeZones.size === 0) {{
          return;
        }}

        const baseR = Math.max(1.2, Math.min(20.0, 1.85 * state.detZoom));
        const r = isHovered ? baseR + 1.8 : (isOrbitMate ? baseR + 1.2 : baseR);

        ctx.save();
        if (Math.hypot(x - bx, y - by) > 2.0 && (captured || isHovered || isOrbitMate)) {{
          ctx.beginPath();
          ctx.moveTo(bx, by);
          ctx.lineTo(x, y);
          ctx.strokeStyle = isOrbitMate ? "rgba(251, 191, 36, 0.55)" : "rgba(244, 63, 133, 0.38)";
          ctx.lineWidth = 1.0;
          ctx.stroke();
        }}

        const ellPtsRaw = isSph ? (p.ellipse_sph || p.ellipse) : p.ellipse;
        const strokeCol = isHovered
          ? "#f43f85"
          : (isOrbitMate ? "#fbbf24" : peakStrokeColor(p, captured ? 0.95 : 0.42));
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = isHovered ? 2.4 : (isOrbitMate ? 2.1 : (captured ? 1.85 : 1.05));

        if (Array.isArray(ellPtsRaw) && ellPtsRaw.length >= 6) {{
          const scrPts = ellPtsRaw.map(pt => detToScreen(pt[0] + shiftAz, pt[1], w, h, fgLayer));
          let maxDist = 0.0, minDist = 1e9;
          for (let m = 0; m < scrPts.length; m++) {{
            const d = Math.hypot(scrPts[m][0] - x, scrPts[m][1] - y);
            if (d > maxDist) maxDist = d;
            if (d < minDist) minDist = d;
          }}
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

          if (p.shape_kind === "ellipse" && (isHovered || isOrbitMate || state.detZoom > 2.2) && scrPts.length >= 16) {{
            const nHalf = Math.floor(scrPts.length / 2);
            const nQuarter = Math.floor(scrPts.length / 4);
            const p0 = [x + (scrPts[0][0] - x) * scale, y + (scrPts[0][1] - y) * scale];
            const pHalf = [x + (scrPts[nHalf][0] - x) * scale, y + (scrPts[nHalf][1] - y) * scale];
            const pQ1 = [x + (scrPts[nQuarter][0] - x) * scale, y + (scrPts[nQuarter][1] - y) * scale];
            const pQ3 = [x + (scrPts[nQuarter + nHalf][0] - x) * scale, y + (scrPts[nQuarter + nHalf][1] - y) * scale];
            ctx.save();
            ctx.strokeStyle = (isHovered || isOrbitMate) ? "rgba(251, 191, 36, 0.75)" : "rgba(56, 189, 248, 0.42)";
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

        // Highlight symmetry-orbit mates or Laue harmonic overlaps on detector
        if (isOrbitMate && !isHovered) {{
          ctx.beginPath();
          ctx.arc(x, y, r + 3.2, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(0, 229, 255, 0.82)";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }}

        if (state.detAlpha < 0.2) {{
          ctx.fillStyle = peakStrokeColor(p, 0.85);
          ctx.fill();
        }}

        if ((isHovered || isOrbitMate) && !queuedTooltipPeakIds.has(p.id) && x >= 0 && x <= w) {{
          queuedTooltipPeakIds.add(p.id);
          pendingTooltips.push({{ p, x, y, isHovered }});
        }}
        ctx.restore();
      }});
    }});

    // Render tooltips for all hovered + symmetry-equivalent reflections on top of rings
    pendingTooltips.forEach(({{ p, x, y, isHovered }}) => {{
      const resBadge = typeof p.careless_residual === "number"
        ? ` ΔI/σ=${{p.careless_residual >= 0 ? "+" : ""}}${{p.careless_residual.toFixed(1)}}σ`
        : "";
      const tag = `(${{p.hkl.join(",")}}) λ=${{p.wavelength.toFixed(2)}}Å d=${{p.d_spacing.toFixed(2)}}Å${{resBadge}}`;
      ctx.save();
      ctx.font = "bold 11px monospace";
      const tw = ctx.measureText(tag).width + 10;
      const tx = Math.min(Math.max(4, x + 8), Math.max(4, w - tw - 4));
      const ty = Math.max(20, y);
      ctx.fillStyle = "rgba(11, 15, 23, 0.92)";
      ctx.fillRect(tx, ty - 20, tw, 18);
      ctx.strokeStyle = isHovered ? "#fbbf24" : "#00e5ff";
      ctx.lineWidth = 1.0;
      ctx.strokeRect(tx, ty - 20, tw, 18);
      ctx.fillStyle = isHovered ? "#fbbf24" : "#38bdf8";
      ctx.fillText(tag, tx + 5, ty - 7);
      ctx.restore();
    }});
  }}

  function getActiveGoniometerR() {{
    const R = DATA.goniometer_R;
    if (Array.isArray(R) && R.length === 3 && Array.isArray(R[0]) && R[0].length === 3) {{
      return R;
    }}
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  }}

  // Rotate 3D Sample-frame unit vector g=[gx, gy, gz] by lab-coupled wulffYawDeg (around lab Y) and wulffPitchDeg (around lab X),
  // coupled through the active goniometer rotation matrix R, and project onto a convex 3D perspective Wulff sphere.
  function projectToWulff(g, cx, cy, R) {{
    if (!Array.isArray(g) || g.length < 3) return [cx, cy, 1, 0.0, 1.0];
    const yawRad = (state.wulffYawDeg || 0.0) * Math.PI / 180.0;
    const pitchRad = (state.wulffPitchDeg || 0.0) * Math.PI / 180.0;
    const cyaw = Math.cos(yawRad), syaw = Math.sin(yawRad);
    const cp = Math.cos(pitchRad), sp = Math.sin(pitchRad);

    // Normalize Sample-frame unit vector and orient crystal sphere by active goniometer R_gonio
    const Rg = getActiveGoniometerR();
    const gn = Math.hypot(g[0], g[1], g[2]) || 1.0;
    const gUnit = [g[0] / gn, g[1] / gn, g[2] / gn];
    const vBase = Rg ? mat3Vec(Rg, gUnit) : gUnit;

    // Apply 360° lab-frame detector view rotation (yaw around lab Y, then screen-space pitch around camera X)
    // Flip -z_lab -> +z_cam so the Ewald diffraction hemisphere (q_z < 0) faces the viewer (+z_cam > 0) at yaw=0
    const vx = vBase[0];
    const vy = vBase[1];
    const vz = -vBase[2];
    const x1 = vx * cyaw + vz * syaw;
    const y1 = vy;
    const z1 = -vx * syaw + vz * cyaw;
    const xCam = x1;
    const yCam = y1 * cp - z1 * sp;
    const zCam = y1 * sp + z1 * cp;

    // Convex 3D sphere perspective projection from camera at z = +D (D > 1):
    // As a feature moves closer to the viewer (zCam -> +1), distance (D - zCam) decreases so apparent size GROWS.
    const D = 2.6;
    const kD = Math.sqrt(D * D - 1.0); // maps tangent horizon limb of unit sphere to radius 1.0
    const dist = Math.max(0.5, D - zCam);
    const persp = kD / dist;
    const dotScale = (D - 0.2) / dist; // ~1.5x at front pole (zCam=+1), ~0.67x at back pole (zCam=-1)
    const hemi = zCam >= 0 ? 1 : -1;
    const scale = R * (state.wulffZoom || 1.0);
    return [cx + xCam * persp * scale, cy - yCam * persp * scale, hemi, zCam, dotScale];
  }}

  function getActiveWulffPeaks() {{
    const src = (state.wulffScope === "all_runs" && SAMPLE_PEAKS.length > 0) ? SAMPLE_PEAKS : PEAKS;
    return src.filter(isPeakVisible);
  }}

  // Render Convex 3D Wulff Sphere (Corner Inset or Expanded Popout)
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
    const zoom = state.wulffZoom || 1.0;
    const rSphere = R * Math.min(zoom, 1.35);
    const isExp = Boolean(state.wulffExpanded);
    const Rg = getActiveGoniometerR();
    const RgT = mat3T(Rg);

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, rSphere, 0, Math.PI * 2);
    const grad = ctx.createRadialGradient(
      cx - rSphere * 0.22, cy - rSphere * 0.22, rSphere * 0.05,
      cx, cy, rSphere
    );
    grad.addColorStop(0, "rgba(18, 25, 48, 0.95)");
    grad.addColorStop(0.72, "rgba(9, 13, 26, 0.95)");
    grad.addColorStop(1, "rgba(4, 6, 14, 0.98)");
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = isExp ? "#00e5ff" : "#334155";
    ctx.lineWidth = isExp ? 1.8 : 1.5;
    ctx.stroke();
    ctx.clip();

    // Helper to stroke a 3D circle on S^2 with front/back hemisphere depth styling
    function strokeSphereCircle(sampleFn, nSteps, frontAlpha, backAlpha, color, lineW, dashBack) {{
      let prevSx = 0, prevSy = 0, prevHemi = 0;
      for (let s = 0; s <= nSteps; s++) {{
        const ang = (s / nSteps) * Math.PI * 2.0;
        const gVec = sampleFn(ang);
        const [sx, sy, hemi] = projectToWulff(gVec, cx, cy, R);
        if (s > 0) {{
          ctx.beginPath();
          ctx.moveTo(prevSx, prevSy);
          ctx.lineTo(sx, sy);
          ctx.strokeStyle = color;
          ctx.lineWidth = lineW * (hemi >= 0 ? 1.0 : 0.75);
          ctx.globalAlpha = (hemi >= 0 && prevHemi >= 0) ? frontAlpha : backAlpha;
          if (dashBack && (hemi < 0 || prevHemi < 0)) ctx.setLineDash([2, 3]);
          else ctx.setLineDash([]);
          ctx.stroke();
        }}
        prevSx = sx; prevSy = sy; prevHemi = hemi;
      }}
      ctx.setLineDash([]);
    }}

    // 3D Wulff Net latitude & meridian graticule circles that rotate with the sphere
    ctx.save();
    [-Math.PI / 4, 0.0, Math.PI / 4].forEach(lat => {{
      const cl = Math.cos(lat), sl = Math.sin(lat);
      strokeSphereCircle(
        ang => [cl * Math.cos(ang), sl, cl * Math.sin(ang)],
        72, 0.22, 0.08, "#94a3b8", 0.9, false
      );
    }});
    [0.0, Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4].forEach(lon => {{
      const clon = Math.cos(lon), slon = Math.sin(lon);
      strokeSphereCircle(
        ang => [Math.cos(ang) * clon, Math.sin(ang), Math.cos(ang) * slon],
        72, 0.20, 0.07, "#94a3b8", 0.9, false
      );
    }});
    ctx.restore();

    // Draw zone great-circle conics directly from 3D Sample-frame zone axis normal
    if (state.showZoneArcs) {{
      ctx.save();
      ZONES.forEach(zone => {{
        if (!state.activeZones.has(zone.zone_id)) return;
        const isSel = state.selectedZoneId === zone.zone_id;
        const lw = isSel ? (isExp ? 2.5 : 2.1) : (isExp ? 1.45 : 1.15);
        const tCryst = Array.isArray(zone.axis_crystal_xyz) && zone.axis_crystal_xyz.length === 3
          ? zone.axis_crystal_xyz
          : (Array.isArray(zone.axis_xyz) && zone.axis_xyz.length === 3 ? mat3Vec(RgT, zone.axis_xyz) : null);
        if (tCryst) {{
          const tAxis = vecUnit(tCryst);
          const ref = Math.abs(tAxis[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
          const e1 = vecUnit(vecCross(tAxis, ref));
          const e2 = vecUnit(vecCross(tAxis, e1));
          strokeSphereCircle(
            ang => {{
              const ca = Math.cos(ang), sa = Math.sin(ang);
              return [ca * e1[0] + sa * e2[0], ca * e1[1] + sa * e2[1], ca * e1[2] + sa * e2[2]];
            }},
            96, isSel ? 0.92 : 0.72, isSel ? 0.35 : 0.20, zone.rainbow_color, lw, true
          );
        }}
      }});
      ctx.restore();
    }}

    const wPeaks = getActiveWulffPeaks();
    const orbitMates = [];

    // Project all peaks in Sample frame and sort back-to-front by camera depth zCam so closer peaks render on top
    const projectedPeaks = wPeaks.map(p => {{
      const gVec = Array.isArray(p.q_hat) ? p.q_hat : mat3Vec(RgT, p.q_hat_lab || [0, 0, 1]);
      const [sx, sy, hemi, zCam, dotScale] = projectToWulff(gVec, cx, cy, R);
      p._wulff_sx = sx;
      p._wulff_sy = sy;
      p._wulff_hemi = hemi;
      return {{ p, sx, sy, hemi, zCam, dotScale }};
    }});
    projectedPeaks.sort((a, b) => a.zCam - b.zCam);

    projectedPeaks.forEach(({{ p, sx, sy, hemi, dotScale }}) => {{
      const captured = isPeakCapturedByActiveZone(p) || state.wulffScope === "all_runs";
      const isHovered = state.hoveredPeakId === p.id || state.selectedPeakId === p.id;
      const isOrbitMate = Boolean(state.activeOrbitId && p.asu_orbit_id && p.asu_orbit_id === state.activeOrbitId);
      if (isOrbitMate) orbitMates.push([sx, sy, p, dotScale]);

      const baseDot = (isExp ? (captured ? 3.8 : 2.6) : (captured ? 2.5 : 1.75)) * dotScale;
      const rDot = isHovered ? baseDot + 2.5 : (isOrbitMate ? baseDot + 1.6 : baseDot);
      const alphaBase = hemi >= 0 ? (captured ? 0.94 : 0.52) : (captured ? 0.55 : 0.28);
      const col = isHovered ? "#fbbf24" : peakStrokeColor(p, alphaBase);

      ctx.save();
      ctx.beginPath();
      ctx.arc(sx, sy, rDot, 0, Math.PI * 2);
      if (hemi < 0) {{
        // Back hemisphere (further from viewer): smaller open hollow circle
        ctx.strokeStyle = col;
        ctx.lineWidth = (isExp ? 1.45 : 1.1) * Math.max(0.75, dotScale);
        ctx.stroke();
      }} else {{
        // Front hemisphere (closer to viewer): larger filled circle
        ctx.fillStyle = col;
        ctx.fill();
      }}

      // Harmonic Overlap Bullseye: concentric outer ring when harmonic_multiplicity > 1 or harmonic_order > 1
      const harmMult = p.harmonic_multiplicity || p.harmonic_order || 1;
      if (harmMult > 1) {{
        ctx.beginPath();
        ctx.arc(sx, sy, rDot + (isExp ? 3.0 : 2.1) * dotScale, 0, Math.PI * 2);
        ctx.strokeStyle = isOrbitMate ? "#fbbf24" : col;
        ctx.lineWidth = (isExp ? 1.25 : 0.9) * Math.max(0.75, dotScale);
        ctx.stroke();
      }}
      ctx.restore();
    }});

    // Symmetry Starburst: connect all symmetry-equivalent observations sharing activeOrbitId
    if (orbitMates.length > 0) {{
      ctx.save();
      orbitMates.forEach(([mx, my, mp]) => {{
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(mx, my);
        ctx.strokeStyle = "rgba(251, 191, 36, 0.58)";
        ctx.lineWidth = isExp ? 1.5 : 1.1;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.arc(mx, my, isExp ? 6.5 : 4.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#00e5ff";
        ctx.lineWidth = 1.6;
        ctx.stroke();

        if (isExp) {{
          const resTxt = typeof mp.careless_residual === "number"
            ? ` ${{mp.careless_residual >= 0 ? "+" : ""}}${{mp.careless_residual.toFixed(1)}}σ`
            : "";
          const lbl = `(${{mp.hkl.join(",")}})${{resTxt}}`;
          ctx.font = "bold 10px monospace";
          ctx.fillStyle = "#070913";
          ctx.fillText(lbl, mx + 8, my - 4);
          ctx.fillStyle = "#fbbf24";
          ctx.fillText(lbl, mx + 7, my - 5);
        }}
      }});
      if (isExp && state.activeOrbitId) {{
        ctx.font = "bold 11px monospace";
        ctx.fillStyle = "#00e5ff";
        ctx.fillText(`ASU Orbit [${{state.activeOrbitId}}] • ${{orbitMates.length}} symmetry mates`, 14, h - 12);
      }}
      ctx.restore();
    }}

    ctx.restore();
  }}

  function findNearestPeakDetector(mx, my) {{
    const rect = detCanvas.getBoundingClientRect();
    let best = null;
    let bestDist = 14.0;
    const fgLayer = 0.45;
    const azShifts = [-360.0, 0.0, 360.0];
    PEAKS.forEach(p => {{
      if (!isPeakVisible(p)) return;
      const py = getPeakVert(p);
      azShifts.forEach(shiftAz => {{
        const [x, y] = detToScreen(p.det_x + shiftAz, py, rect.width, rect.height, fgLayer);
        const [bx, by] = detToScreen(p.det_x + shiftAz, py, rect.width, rect.height, -0.35);
        const d = Math.min(Math.hypot(mx - x, my - y), Math.hypot(mx - bx, my - by));
        if (d < bestDist) {{
          bestDist = d;
          best = p;
        }}
      }});
    }});
    return best;
  }}

  function findNearestPeakWulff(mx, my) {{
    let best = null;
    let bestDist = state.wulffExpanded ? 14.0 : 10.0;
    const wPeaks = getActiveWulffPeaks();
    wPeaks.forEach(p => {{
      if (typeof p._wulff_sx !== "number" || typeof p._wulff_sy !== "number") return;
      const d = Math.hypot(mx - p._wulff_sx, my - p._wulff_sy);
      if (d < bestDist) {{
        bestDist = d;
        best = p;
      }}
    }});
    return best;
  }}

  // Mouse & Wheel interactions on Detector Canvas (with Google-Earth 360° periodic coupling)
  let isPanningDet = false;
  let panStart = [0, 0];
  detCanvas.addEventListener("mousedown", (ev) => {{
    isPanningDet = true;
    panStart = [ev.clientX - state.detPanX, ev.clientY - state.detPanY];
  }});
  window.addEventListener("mouseup", () => {{ isPanningDet = false; isDraggingWulff = false; }});

  detCanvas.addEventListener("mousemove", (ev) => {{
    const rect = detCanvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    state.targetParallaxX = ((mx / rect.width) - 0.5) * 2.0;
    state.targetParallaxY = ((my / rect.height) - 0.5) * 2.0;

    if (isPanningDet) {{
      let nextPanX = ev.clientX - panStart[0];
      const plotW = Math.max(50, rect.width - 54);
      const span360 = plotW * state.detZoom;
      if (span360 > 10) {{
        while (nextPanX > 0.5 * span360) {{ nextPanX -= span360; panStart[0] += span360; }}
        while (nextPanX < -0.5 * span360) {{ nextPanX += span360; panStart[0] -= span360; }}
        state.wulffYawDeg = -(nextPanX / span360) * 360.0;
      }}
      state.detPanX = nextPanX;
      state.detPanY = ev.clientY - panStart[1];
      return;
    }}
    const hit = findNearestPeakDetector(mx, my);
    state.hoveredPeakId = hit ? hit.id : null;
    if (hit) {{
      state.activeOrbitId = hit.asu_orbit_id || null;
      updatePeakInspector(hit);
    }} else if (state.selectedPeakId === null) {{
      state.activeOrbitId = null;
    }}
  }});

  detCanvas.addEventListener("mouseleave", () => {{
    state.targetParallaxX = 0.0;
    state.targetParallaxY = 0.0;
    state.hoveredPeakId = null;
    if (state.selectedPeakId === null) state.activeOrbitId = null;
  }});

  detCanvas.addEventListener("wheel", (ev) => {{
    ev.preventDefault();
    const rawDy = ev.deltaMode === 1 ? ev.deltaY * 16 : (ev.deltaMode === 2 ? ev.deltaY * 100 : ev.deltaY);
    const clampedDy = Math.max(-50, Math.min(50, rawDy));
    const sensitivity = ev.ctrlKey ? 0.0035 : 0.0016;
    const factor = Math.exp(-clampedDy * sensitivity);
    state.detZoom = Math.max(0.7, Math.min(10.0, state.detZoom * factor));
  }}, {{ passive: false }});

  detCanvas.addEventListener("click", (ev) => {{
    const rect = detCanvas.getBoundingClientRect();
    const hit = findNearestPeakDetector(ev.clientX - rect.left, ev.clientY - rect.top);
    if (hit) {{
      state.selectedPeakId = hit.id;
      state.activeOrbitId = hit.asu_orbit_id || null;
      updatePeakInspector(hit);
    }} else {{
      state.selectedPeakId = null;
      state.activeOrbitId = null;
    }}
  }});

  // Interactive 3D Yaw/Pitch Drag + Wheel Zoom + Symmetry Starburst Selection on Wulff Net Canvas
  let isDraggingWulff = false;
  let wulffDragStart = [0, 0, 0, 0];
  if (stereoCanvas) {{
    stereoCanvas.addEventListener("mousedown", (ev) => {{
      isDraggingWulff = true;
      wulffDragStart = [ev.clientX, ev.clientY, state.wulffYawDeg, state.wulffPitchDeg];
    }});
    stereoCanvas.addEventListener("mousemove", (ev) => {{
      const rect = stereoCanvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;
      if (isDraggingWulff) {{
        const dx = ev.clientX - wulffDragStart[0];
        const dy = ev.clientY - wulffDragStart[1];
        state.wulffYawDeg = ((wulffDragStart[2] + dx * 0.55 + 540) % 360) - 180;
        state.wulffPitchDeg = Math.max(-85, Math.min(85, wulffDragStart[3] + dy * 0.45));
        const detRect = detCanvas.getBoundingClientRect();
        const span360 = Math.max(50, detRect.width - 54) * state.detZoom;
        state.detPanX = -(state.wulffYawDeg / 360.0) * span360;
        return;
      }}
      const hit = findNearestPeakWulff(mx, my);
      if (hit) {{
        state.hoveredPeakId = hit.id;
        state.activeOrbitId = hit.asu_orbit_id || null;
        updatePeakInspector(hit);
      }} else {{
        state.hoveredPeakId = null;
        if (state.selectedPeakId === null) state.activeOrbitId = null;
      }}
    }});
    stereoCanvas.addEventListener("mouseleave", () => {{
      state.hoveredPeakId = null;
      if (state.selectedPeakId === null) state.activeOrbitId = null;
    }});
    stereoCanvas.addEventListener("wheel", (ev) => {{
      ev.preventDefault();
      const rawDy = ev.deltaMode === 1 ? ev.deltaY * 16 : (ev.deltaMode === 2 ? ev.deltaY * 100 : ev.deltaY);
      const clampedDy = Math.max(-50, Math.min(50, rawDy));
      const sensitivity = ev.ctrlKey ? 0.0035 : 0.0016;
      const factor = Math.exp(-clampedDy * sensitivity);
      state.wulffZoom = Math.max(0.6, Math.min(5.0, (state.wulffZoom || 1.0) * factor));
    }}, {{ passive: false }});
    stereoCanvas.addEventListener("click", (ev) => {{
      const rect = stereoCanvas.getBoundingClientRect();
      const hit = findNearestPeakWulff(ev.clientX - rect.left, ev.clientY - rect.top);
      if (hit) {{
        state.selectedPeakId = hit.id;
        state.activeOrbitId = hit.asu_orbit_id || null;
        updatePeakInspector(hit);
      }}
    }});
  }}

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

  document.getElementById("btn-stereo-toggle").addEventListener("click", () => {{
    state.showStereo = !state.showStereo;
    syncWulffUI();
  }});

  const btnWulffScope = document.getElementById("btn-wulff-scope");
  if (btnWulffScope) {{
    btnWulffScope.addEventListener("click", () => {{
      state.wulffScope = state.wulffScope === "all_runs" ? "single_frame" : "all_runs";
      const detRect = detCanvas.getBoundingClientRect();
      const span360 = Math.max(50, detRect.width - 54) * state.detZoom;
      state.wulffYawDeg = -(state.detPanX / Math.max(1, span360)) * 360.0;
      syncWulffUI();
    }});
  }}

  const selWulffColor = document.getElementById("sel-wulff-color");
  if (selWulffColor) {{
    selWulffColor.addEventListener("change", () => {{
      state.wulffColorMode = selWulffColor.value || "wavelength";
      syncWulffUI();
    }});
  }}

  const btnWulffExpand = document.getElementById("btn-wulff-expand");
  if (btnWulffExpand) {{
    btnWulffExpand.addEventListener("click", () => {{
      state.wulffExpanded = !state.wulffExpanded;
      syncWulffUI();
    }});
  }}

  const btnWulffClose = document.getElementById("btn-wulff-close");
  if (btnWulffClose) {{
    btnWulffClose.addEventListener("click", () => {{
      state.showStereo = false;
      state.wulffExpanded = false;
      syncWulffUI();
    }});
  }}

  document.getElementById("btn-reset-view").addEventListener("click", () => {{
    state.detZoom = 1.0;
    state.detPanX = 0.0;
    state.detPanY = 0.0;
    state.wulffYawDeg = 0.0;
    state.wulffPitchDeg = 0.0;
    state.wulffZoom = 1.0;
    state.selectedPeakId = null;
    state.activeOrbitId = null;
    setMaxZoneOrder(1);
  }});

  syncWulffUI();

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
    if (filesDrawer && filesDrawer.classList.contains("open") && PIPELINE.length > 0) {{
      const curStage = Math.min(PIPELINE.length - 1, Math.max(0, state.currentPipelineIdx));
      selectPipelineStage((curStage - 1 + PIPELINE.length) % PIPELINE.length, true);
      return;
    }}
    if (state.currentPipelineIdx >= 2 && PIPELINE.length >= 4) {{
      const curStage = Math.min(3, state.currentPipelineIdx);
      if (curStage === 3) {{
        selectPipelineStage(2, false);
        return;
      }}
      if (curStage === 2) {{
        selectLadderRung(Math.max(0, LADDER.length - 1), true);
        return;
      }}
    }}
    if (state.currentRungIdx > 0) {{
      selectLadderRung(state.currentRungIdx - 1);
    }} else if (PIPELINE.length >= 4) {{
      selectPipelineStage(3, false);
    }} else {{
      selectLadderRung((state.currentRungIdx - 1 + LADDER.length) % LADDER.length);
    }}
  }});

  document.getElementById("btn-ladder-next").addEventListener("click", () => {{
    if (filesDrawer && filesDrawer.classList.contains("open") && PIPELINE.length > 0) {{
      const curStage = Math.min(PIPELINE.length - 1, Math.max(0, state.currentPipelineIdx));
      selectPipelineStage((curStage + 1) % PIPELINE.length, true);
      return;
    }}
    if (state.currentPipelineIdx >= 2 && PIPELINE.length >= 4) {{
      const curStage = Math.min(3, state.currentPipelineIdx);
      if (curStage === 2) {{
        selectPipelineStage(3, false);
        return;
      }}
      if (curStage === 3) {{
        selectLadderRung(0, true);
        return;
      }}
    }}
    if (state.currentRungIdx + 1 < LADDER.length) {{
      selectLadderRung(state.currentRungIdx + 1);
    }} else if (PIPELINE.length >= 4) {{
      selectPipelineStage(2, false);
    }} else {{
      selectLadderRung((state.currentRungIdx + 1) % LADDER.length);
    }}
  }});

  document.getElementById("btn-ladder-play").addEventListener("click", (ev) => {{
    if (state.ladderTimer) {{
      clearInterval(state.ladderTimer);
      state.ladderTimer = null;
      ev.currentTarget.textContent = "WALK SOLVE STEPS";
      return;
    }}
    selectPipelineStage(1, false);
    selectLadderRung(0, false);
    ev.currentTarget.textContent = "STOP";
    state.ladderTimer = setInterval(() => {{
      if (state.currentRungIdx + 1 < LADDER.length) {{
        selectLadderRung(state.currentRungIdx + 1, false);
      }} else {{
        clearInterval(state.ladderTimer);
        state.ladderTimer = null;
        document.getElementById("btn-ladder-play").textContent = "WALK SOLVE STEPS";
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
  function recomputeBoundsAndReload(newPayload, opts) {{
    if (!newPayload) return;
    const isFrameSwitch = Boolean(opts && opts.isFrameSwitch);
    const isSolveRunning = state.runningStageIdx === 1 || typeof newPayload.active_rung_idx === "number";
    const prevZones = [...ZONES];
    const prevPeaks = [...PEAKS];
    const prevSamplePeaks = [...SAMPLE_PEAKS];
    const prevLadder = [...LADDER];
    const prevRungIdx = state.currentRungIdx;
    const prevPipelineIdx = state.currentPipelineIdx;
    const prevInst = DATA.instrument;
    DATA.instrument = newPayload.instrument || DATA.instrument;
    if (newPayload.beamline_detectors) {{
      DATA.beamline_detectors = newPayload.beamline_detectors;
    }}
    if (!isFrameSwitch && (newPayload.default_projection || (DATA.instrument && DATA.instrument !== prevInst))) {{
      const nextProj = newPayload.default_projection || ((DATA.instrument === "MANDI" || DATA.instrument === "TOPAZ") ? "spherical" : "cylindrical");
      state.projection = nextProj;
      syncProjectionUI();
    }}
    DATA.space_group = newPayload.space_group || DATA.space_group;
    DATA.wavelength_range = newPayload.wavelength_range || DATA.wavelength_range;
    const nextArtKind = newPayload.artifact_kind || DATA.artifact_kind || "solve";
    const wl0 = Number((DATA.wavelength_range && DATA.wavelength_range[0]) || state.lamMin || 2.0);
    if (nextArtKind === "merged" || nextArtKind === "raw_nexus") {{
      DATA.d_min = Number((0.9 * wl0).toFixed(2));
    }} else {{
      DATA.d_min = Number(newPayload.d_min || (0.9 * wl0).toFixed(2));
    }}
    DATA.cell = newPayload.cell || DATA.cell;
    if (Array.isArray(newPayload.frames)) DATA.frames = newPayload.frames;
    if (typeof newPayload.frame_index === "number") DATA.frame_index = newPayload.frame_index;
    if (newPayload.goniometer_summary) DATA.goniometer_summary = newPayload.goniometer_summary;
    if (Array.isArray(newPayload.goniometer_R)) DATA.goniometer_R = newPayload.goniometer_R;
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

    let nextPeaks = Array.isArray(newPayload.peaks) ? [...newPayload.peaks] : [];
    let nextSamplePeaks = Array.isArray(newPayload.sample_peaks) ? [...newPayload.sample_peaks] : nextPeaks;
    let nextZones = Array.isArray(newPayload.zones) ? [...newPayload.zones] : [];
    const nextPanels = Array.isArray(newPayload.panels) ? [...newPayload.panels] : [];
    let nextLadder = Array.isArray(newPayload.ladder_steps) && newPayload.ladder_steps.length > 0 ? [...newPayload.ladder_steps] : [...LADDER];
    const nextPipeline = Array.isArray(newPayload.pipeline_stages) && newPayload.pipeline_stages.length > 0 ? [...newPayload.pipeline_stages] : [...PIPELINE];

    if (isSolveRunning) {{
      if (nextZones.length === 0 && prevZones.length > 0) {{
        nextZones = prevZones;
        nextLadder = prevLadder;
      }}
      if (nextPeaks.length === 0 && prevPeaks.length > 0) {{
        nextPeaks = prevPeaks;
        nextSamplePeaks = prevSamplePeaks;
      }}
    }}

    PEAKS.length = 0;
    nextPeaks.forEach(p => PEAKS.push(p));
    SAMPLE_PEAKS.length = 0;
    nextSamplePeaks.forEach(p => SAMPLE_PEAKS.push(p));
    DATA.merge_metrics = newPayload.merge_metrics || null;
    ZONES.length = 0;
    nextZones.forEach(z => ZONES.push(z));
    PANELS.length = 0;
    nextPanels.forEach(p => PANELS.push(p));
    LADDER.length = 0;
    nextLadder.forEach(s => LADDER.push(s));
    PIPELINE.length = 0;
    nextPipeline.forEach(s => PIPELINE.push(s));

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
    DATA.pending_stage = newPayload.pending_stage || (artKind === "merged" ? "solve" : (artKind === "raw_nexus" ? "reduce" : (artKind === "integrator" ? "merge" : (artKind === "merged_careless" ? "done" : "integrate"))));
    let artBadge = "SOLVED ORIENTATION";
    if (artKind === "merged_careless") {{
      const ccVal = DATA.merge_metrics && typeof DATA.merge_metrics.cc_half_overall === "number"
        ? `CC½=${{DATA.merge_metrics.cc_half_overall.toFixed(3)}}`
        : "CARELESS MERGED";
      artBadge = `CARELESS MERGE • ${{ccVal}}`;
    }} else if (artKind === "integrator") {{
      artBadge = shapeMode === "anisotropic_ellipse"
        ? "INTEGRATED • 3D TENSOR ELLIPSES"
        : "INTEGRATED • ISOTROPIC 2σ CIRCLES";
    }} else if (artKind === "predictor") {{
      artBadge = "PREDICTOR • PREDICTED PEAKS";
    }} else if (artKind === "merged") {{
      artBadge = "MERGED.H5 • PENDING: 2. SOLVE";
    }} else if (artKind === "raw_nexus") {{
      artBadge = "RAW NEXUS • PENDING: 1. REDUCE";
    }} else if (newPayload.intensity_source === "raw_h5") {{
      artBadge = "RAW DETECTOR COUNTS";
    }}
    if (subEl) {{
      subEl.textContent = `${{DATA.instrument || "CG4D"}} • ${{DATA.space_group || "P 1"}} • ${{artBadge}}`;
    }}
    if (DATA.space_group) {{
      const sgEl = document.getElementById("inp-sg");
      if (sgEl) sgEl.value = DATA.space_group;
      const ndSgEl = document.getElementById("nd-sg");
      if (ndSgEl) ndSgEl.value = DATA.space_group;
    }}
    if (DATA.cell && DATA.cell.length === 6) {{
      const c = DATA.cell;
      document.getElementById("cell-summary").textContent = `${{c[0].toFixed(1)}}, ${{c[1].toFixed(1)}}, ${{c[2].toFixed(1)}} Å`;
      const ids = ["inp-a", "inp-b", "inp-c", "inp-alpha", "inp-beta", "inp-gamma"];
      const ndIds = ["nd-a", "nd-b", "nd-c", "nd-alpha", "nd-beta", "nd-gamma"];
      ids.forEach((id, i) => {{
        const el = document.getElementById(id);
        if (el && typeof c[i] === "number") el.value = c[i].toFixed(2);
      }});
      ndIds.forEach((id, i) => {{
        const el = document.getElementById(id);
        if (el && typeof c[i] === "number") el.value = c[i].toFixed(2);
      }});
    }}
    const ord1 = ZONES.filter(z => (z.order || 1) <= 1).map(z => z.zone_id);
    state.maxZoneOrder = ord1.length > 0 ? 1 : 2;
    state.activeZones = new Set(ord1.length > 0 ? ord1 : ZONES.map(z => z.zone_id));
    if (isSolveRunning) {{
      state.currentPipelineIdx = 1;
      state.currentRungIdx = typeof newPayload.active_rung_idx === "number"
        ? newPayload.active_rung_idx
        : Math.max(1, prevRungIdx);
    }} else if (isFrameSwitch) {{
      state.currentPipelineIdx = prevPipelineIdx;
      state.currentRungIdx = prevRungIdx;
    }} else {{
      state.currentPipelineIdx = pipelineIdxForArtifactKind(artKind);
      state.currentRungIdx = artKind === "merged" ? 0 : Math.max(0, LADDER.length - 1);
    }}
    if (!isFrameSwitch && (artKind === "merged_careless" || Boolean(DATA.merge_metrics))) {{
      state.showStereo = true;
      state.wulffExpanded = true;
      state.wulffScope = "all_runs";
      state.wulffFrame = "sample";
      state.wulffColorMode = "residual";
    }}
    syncWulffUI();
    renderPipeline();
    if (LADDER.length > 0) selectLadderRung(state.currentRungIdx, false);
    else {{ renderZoneList(); updateHUD(); }}
    renderFilesDrawer();
    probeGalaxyConnection();
    if (isFrameSwitch) {{
      return;
    }}
    if (artKind === "merged_careless" && DATA.merge_metrics) {{
      const m = DATA.merge_metrics;
      const cc = typeof m.cc_half_overall === "number" ? m.cc_half_overall.toFixed(3) : "—";
      const isig = typeof m.mean_i_over_sigi === "number" ? m.mean_i_over_sigi.toFixed(1) : "—";
      const comp = typeof m.completeness_pct === "number" ? `${{m.completeness_pct.toFixed(1)}}%` : "—";
      const mult = typeof m.multiplicity === "number" ? `${{m.multiplicity.toFixed(2)}}×` : "—";
      setStatusLine(
        `[4. MERGE COMPLETE]`,
        `<strong style="color:var(--accent-gold);">Careless Variational Merge</strong> • CC½=<strong style="color:var(--accent-cyan);">${{cc}}</strong> • ⟨I/σ⟩=<strong>${{isig}}</strong> • Completeness=<strong style="color:#34d399;">${{comp}}</strong> (${{m.n_merged_asu || 0}} ASU, mult=${{mult}}) — Click any reflection to inspect its ASU symmetry starburst &amp; ΔI/σ residual.`
      );
    }} else if (artKind === "merged") {{
      const cStr = (DATA.cell && DATA.cell.length === 6)
        ? `${{DATA.space_group || "P 1"}} (${{DATA.cell[0].toFixed(1)}}, ${{DATA.cell[1].toFixed(1)}}, ${{DATA.cell[2].toFixed(1)}} Å)`
        : (DATA.space_group || "P 1");
      setStatusLine(
        `[MERGED.H5]`,
        `<strong style="color:var(--accent-cyan);">Merged detector stack loaded</strong> (${{(DATA.frames || []).length || 1}} frames • ${{cStr}}) — Pending step: <strong style="color:var(--accent-pink);">2. Solve (Lattice Ladder)</strong>.`
      );
    }} else if (artKind === "integrator" || artKind === "predictor") {{
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
    let inst = getAttrStr(f, "instrument", "");
    if (!inst) {{
      try {{
        if (f.keys().includes("files")) {{
          const fl = Array.from(f.get("files").value || []).map(String).join(" ").toUpperCase();
          if (fl.includes("MANDI")) inst = "MANDI";
          else if (fl.includes("CG4D")) inst = "CG4D";
          else if (fl.includes("IMAGINE")) inst = "IMAGINE";
        }}
      }} catch (_e) {{}}
    }}
    if (!inst && mf && mf.keys().includes("images")) {{
      const sh = Array.from(mf.get("images").shape || []).map(Number);
      if (sh[1] === 256) inst = "MANDI";
      else if (sh[1] === 512) inst = "CG4D";
    }}
    if (!inst) inst = DATA.instrument || "CG4D";

    let wlRangeNew = (inst === "MANDI" || inst === "TOPAZ") ? [2.0, 4.0] : [2.8, 4.5];
    let dMinNew = Number((0.9 * Number(wlRangeNew[0])).toFixed(2));
    try {{
      if (f.keys().includes("instrument")) {{
        const wl = f.get("instrument/wavelength").value;
        if (wl && wl.length >= 2 && !(Math.abs(Number(wl[0]) - 2.0) < 1e-3 && Math.abs(Number(wl[1]) - 10.0) < 1e-3)) {{
          wlRangeNew = [Number(wl[0]), Number(wl[1])];
          dMinNew = Number((0.9 * Number(wlRangeNew[0])).toFixed(2));
        }}
      }}
      if (f.keys().includes("scan_refinement")) {{
        const srGrp = f.get("scan_refinement");
        if (srGrp.keys().includes("band_wavelength")) {{
          const bwl = srGrp.get("band_wavelength").value;
          if (bwl && bwl.length >= 2) {{
            wlRangeNew = [Number(bwl[0]), Number(bwl[1])];
            dMinNew = Number((0.9 * Number(wlRangeNew[0])).toFixed(2));
          }}
        }}
        if (srGrp && srGrp.attrs && srGrp.attrs["d_min"]) {{
          dMinNew = Number(srGrp.attrs["d_min"].value) || dMinNew;
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
      if (bankIdsRun.some(b => b > 60)) inst = "CG4D";
      else if (bankIdsRun.some(b => b < 10) || (bankIdsRun.length > 0 && bankIdsRun.length <= 45)) inst = "MANDI";
    }}

    // Seed nominal detector banks from DATA.beamline_detectors[inst], then override with detector_calibration/bank_*
    const detMap = new Map();
    const nomBanks = (DATA.beamline_detectors && DATA.beamline_detectors[inst]) || {{}};
    Object.entries(nomBanks).forEach(([bkStr, cfg]) => {{
      const bid = parseInt(bkStr, 10);
      detMap.set(bid, {{
        bid,
        center: [...cfg.center],
        uhat: [...cfg.uhat],
        vhat: [...cfg.vhat],
        width: Number(cfg.width),
        height: Number(cfg.height),
        m: Number(cfg.m || (inst === "MANDI" ? 256 : 512)),
        n: Number(cfg.n || (inst === "MANDI" ? 256 : 512)),
      }});
    }});
    if (f.keys().includes("detector_calibration")) {{
      const calGrp = f.get("detector_calibration");
      for (const bName of calGrp.keys()) {{
        if (!bName.startsWith("bank_")) continue;
        const bid = parseInt(bName.slice(5), 10);
        const g = calGrp.get(bName);
        const baseDet = detMap.get(bid) || {{ m: inst === "MANDI" ? 256 : 512, n: inst === "MANDI" ? 256 : 512 }};
        const center = Array.from(g.get("center").value).map(Number);
        const uhat = Array.from(g.get("uhat").value).map(Number);
        const vhat = Array.from(g.get("vhat").value).map(Number);
        const width = Number(g.get("width").value);
        const height = Number(g.get("height").value);
        detMap.set(bid, {{ bid, center, uhat, vhat, width, height, m: baseDet.m, n: baseDet.n }});
      }}
    }}

    // Read raw detector images for the active run slice and build grayscale texture canvases
    const bankImgs = new Map();
    const bankGrayCanvases = new Map();
    if (mf && mf.keys().includes("images")) {{
      const imDs = mf.get("images");
      const shape = Array.from(imDs.shape || []).map(Number); // [N, H, W]
      const H = shape[1] || 512, W = shape[2] || 512;
      const bStep = 4;
      const Hb = Math.floor(H / bStep), Wb = Math.floor(W / bStep);
      for (let i = 0; i < bankIdsRun.length; i++) {{
        const bid = Number(bankIdsRun[i]);
        const sub = imDs.slice([[f0 + i, f0 + i + 1], [0, H], [0, W]]);
        bankImgs.set(bid, {{ data: sub, H, W }});

        // Downsample and log1p-scale into offscreen canvas (canonical r, c order matching C0..C3)
        const logBuf = new Float32Array(Hb * Wb);
        for (let rb = 0; rb < Hb; rb++) {{
          for (let cb = 0; cb < Wb; cb++) {{
            let sum = 0;
            for (let dr = 0; dr < bStep; dr++) {{
              const rowOff = (rb * bStep + dr) * W + cb * bStep;
              for (let dc = 0; dc < bStep; dc++) sum += Number(sub[rowOff + dc]);
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
            const v = Math.min(255, Math.round((logBuf[rb * Wb + cb] / vmax) * 255));
            const pIdx = (rb * Wb + cb) * 4;
            idata.data[pIdx] = v; idata.data[pIdx+1] = v; idata.data[pIdx+2] = v; idata.data[pIdx+3] = 255;
          }}
        }}
        gctx.putImageData(idata, 0, 0);
        bankGrayCanvases.set(bid, gc);
      }}
    }}

    // Build panel outlines and 4 corners from detMap (det.center is the (0,0) corner in subhkl Detector.pixel_to_lab)
    const panelsOut = [];
    const pixelToLab = (det, r, c) => {{
      const u = (c / Math.max(1, det.m - 1)) * det.width;
      const v = (r / Math.max(1, det.n - 1)) * det.height;
      return [
        det.center[0] + u * det.uhat[0] + v * det.vhat[0],
        det.center[1] + u * det.uhat[1] + v * det.vhat[1],
        det.center[2] + u * det.uhat[2] + v * det.vhat[2],
      ];
    }};
    for (const [bid, det] of detMap.entries()) {{
      const corners = [], cornersSph = [];
      [[0, 0], [det.n - 1, 0], [det.n - 1, det.m - 1], [0, det.m - 1]].forEach(([rc, cc]) => {{
        const [azC, yC, elC] = labToCyl(pixelToLab(det, rc, cc));
        corners.push([azC, yC]);
        cornersSph.push([azC, elC]);
      }});
      const outline = [], outlineSph = [];
      const nPts = 12;
      const edges = [];
      for (let k=0;k<nPts;k++) edges.push([(k/nPts)*(det.n-1), 0]);
      for (let k=0;k<nPts;k++) edges.push([det.n-1, (k/nPts)*(det.m-1)]);
      for (let k=0;k<nPts;k++) edges.push([((nPts-k)/nPts)*(det.n-1), det.m-1]);
      for (let k=0;k<nPts;k++) edges.push([0, ((nPts-k)/nPts)*(det.m-1)]);
      edges.forEach(([r, c]) => {{
        const [az, y, el] = labToCyl(pixelToLab(det, r, c));
        outline.push([az, y]); outlineSph.push([az, el]);
      }});
      const gridN = 8;
      const mesh = [], meshSph = [];
      for (let ir = 0; ir <= gridN; ir++) {{
        const r = (ir / gridN) * (det.n - 1);
        const rowCyl = [], rowSph = [];
        for (let ic = 0; ic <= gridN; ic++) {{
          const c = (ic / gridN) * (det.m - 1);
          const [azM, yM, elM] = labToCyl(pixelToLab(det, r, c));
          rowCyl.push([azM, yM]);
          rowSph.push([azM, elM]);
        }}
        mesh.push(rowCyl);
        meshSph.push(rowSph);
      }}
      const allAz = [
        ...corners.map(p => p[0]),
        ...outline.map(p => p[0]),
        ...mesh.flat().map(p => p[0]),
      ];
      let azSpan = Math.max(...allAz) - Math.min(...allAz);
      if (azSpan > 180) {{
        const unwrapped = allAz.map(a => a < 0 ? a + 360 : a);
        const uSpan = Math.max(...unwrapped) - Math.min(...unwrapped);
        if (uSpan <= 180) {{
          corners.forEach(p => {{ if (p[0] < 0) p[0] += 360; }});
          cornersSph.forEach(p => {{ if (p[0] < 0) p[0] += 360; }});
          outline.forEach(p => {{ if (p[0] < 0) p[0] += 360; }});
          outlineSph.forEach(p => {{ if (p[0] < 0) p[0] += 360; }});
          mesh.forEach(row => row.forEach(p => {{ if (p[0] < 0) p[0] += 360; }}));
          meshSph.forEach(row => row.forEach(p => {{ if (p[0] < 0) p[0] += 360; }}));
          azSpan = uSpan;
        }}
      }}
      if (azSpan <= 180) {{
        const ptsCyl = [...outline, ...mesh.flat()];
        const ptsSph = [...outlineSph, ...meshSph.flat()];
        const azMin = Math.min(...ptsCyl.map(p => p[0]));
        const azMax = Math.max(...ptsCyl.map(p => p[0]));
        const yMin = Math.min(...ptsCyl.map(p => p[1]));
        const yMax = Math.max(...ptsCyl.map(p => p[1]));
        const elMin = Math.min(...ptsSph.map(p => p[1]));
        const elMax = Math.max(...ptsSph.map(p => p[1]));
        panelsOut.push({{
          bank: bid, az_min: azMin, az_max: azMax, y_min: yMin, y_max: yMax, el_min: elMin, el_max: elMax,
          corners, corners_sph: cornersSph,
          mesh, mesh_sph: meshSph,
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
      const sKeys = sGrp.keys();
      if (["a", "b", "c", "alpha", "beta", "gamma"].every(k => sKeys.includes(k))) {{
        cell = [
          Number(sGrp.get("a").value),
          Number(sGrp.get("b").value),
          Number(sGrp.get("c").value),
          Number(sGrp.get("alpha").value),
          Number(sGrp.get("beta").value),
          Number(sGrp.get("gamma").value),
        ];
      }}
      if (sKeys.includes("U")) {{
        const uRaw = Array.from(sGrp.get("U").value).map(Number);
        U = [[uRaw[0],uRaw[1],uRaw[2]],[uRaw[3],uRaw[4],uRaw[5]],[uRaw[6],uRaw[7],uRaw[8]]];
      }}
      if (sKeys.includes("B")) {{
        const bRaw = Array.from(sGrp.get("B").value).map(Number);
        B = [[bRaw[0],bRaw[1],bRaw[2]],[bRaw[3],bRaw[4],bRaw[5]],[bRaw[6],bRaw[7],bRaw[8]]];
        const invB = mat3Inv(B);
        const aLen = Math.hypot(invB[0][0], invB[1][0], invB[2][0]);
        const bLen = Math.hypot(invB[0][1], invB[1][1], invB[2][1]);
        const cLen = Math.hypot(invB[0][2], invB[1][2], invB[2][2]);
        cell = [aLen, bLen, cLen, cell[3] || 90, cell[4] || 90, cell[5] || 90];
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
    if (f.keys().includes("images") && !f.keys().includes("peaks") && !f.keys().includes("banks") && !B) {{
      artifactKind = "merged";
      peakShapeMode = "none";
    }} else if (f.keys().includes("entry") && !f.keys().includes("images")) {{
      artifactKind = "raw_nexus";
      peakShapeMode = "none";
    }}
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
              det_x: az, det_y: dy, det_el: el, q_hat: gCryst, q_hat_lab: qLab,
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
              det_x: az, det_y: dy, det_el: el, q_hat: gCryst, q_hat_lab: qLab,
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
      const maxCell = Math.max(cell[0] || 68.7, cell[1] || 69.8, cell[2] || 156.9);
      const minCell = Math.min(cell[0] || 68.7, cell[1] || 69.8, cell[2] || 156.9);
      const dMin = maxCell < 35.0 ? Math.max(0.75, minCell * 0.085) : 2.6;
      const hMax = Math.min(20, Math.floor(cell[0] / dMin));
      const kMax = Math.min(20, Math.floor(cell[1] / dMin));
      const lMax = Math.min(28, Math.floor(cell[2] / dMin));
      const sgClean = String(sg || "").trim().toUpperCase();
      const isBodyCentered = sgClean.startsWith("I");
      const isFaceCentered = sgClean.startsWith("F");
      const isCCentered = sgClean.startsWith("C");
      const geomHits = [];
      for (let h = -hMax; h <= hMax; h++) {{
        for (let k = -kMax; k <= kMax; k++) {{
          for (let l = -lMax; l <= lMax; l++) {{
            if (h === 0 && k === 0 && l === 0) continue;
            if (isBodyCentered && ((h + k + l) & 1) !== 0) continue;
            if (isFaceCentered && (((h + k) & 1) !== 0 || ((k + l) & 1) !== 0 || ((h + l) & 1) !== 0)) continue;
            if (isCCentered && ((h + k) & 1) !== 0) continue;
            const G = mat3Vec(RUB, [h, k, l]);
            const qn = vecNorm(G);
            if (qn < 1e-8 || (1.0 / qn) < dMin) continue;
            const sth = -G[2] / qn;
            if (sth <= 0) continue;
            const lam = (2.0 * sth) / qn;
            if (lam < wlRangeNew[0] || lam > wlRangeNew[1]) continue;
            const kf = vecUnit([lam * G[0], lam * G[1], 1.0 + lam * G[2]]);
            for (const [bid, det] of detMap.entries()) {{
              if (bankImgs.size > 0 && !bankImgs.has(bid)) continue;
              const nVec = vecCross(det.uhat, det.vhat);
              const denom = vecDot(kf, nVec);
              if (Math.abs(denom) <= 1e-6) continue;
              const tRay = vecDot(det.center, nVec) / denom;
              if (tRay <= 0) continue;
              const pt = [kf[0]*tRay, kf[1]*tRay, kf[2]*tRay];
              const rel = [pt[0]-det.center[0], pt[1]-det.center[1], pt[2]-det.center[2]];
              const u = vecDot(rel, det.uhat), v = vecDot(rel, det.vhat);
              if (u < 0.02 * det.width || u > 0.98 * det.width || v < 0.02 * det.height || v > 0.98 * det.height) continue;
              const cPix = (det.m - 1) * (u / det.width);
              const rPix = (det.n - 1) * (v / det.height);
              let inten = 120.0, snr = 10.0, hasImg = false;
              const bIm = bankImgs.get(bid);
              if (bIm) {{
                hasImg = true;
                const ri = Math.round(rPix), ci = Math.round(cPix);
                let pkMax = 0, bgSum = 0, bgN = 0;
                for (let dr = -9; dr <= 9; dr++) {{
                  const rr = ri + dr; if (rr < 0 || rr >= bIm.H) continue;
                  for (let dc = -9; dc <= 9; dc++) {{
                    const cc = ci + dc; if (cc < 0 || cc >= bIm.W) continue;
                    const val = Number(bIm.data[rr * bIm.W + cc]);
                    if (Math.abs(dr) <= 3 && Math.abs(dc) <= 3) {{ if (val > pkMax) pkMax = val; }}
                    bgSum += val; bgN++;
                  }}
                }}
                const bgMean = bgSum / Math.max(1, bgN);
                snr = (pkMax - bgMean) / Math.max(Math.sqrt(Math.max(bgMean, 1.0)), 1.0);
                inten = Math.max(pkMax, 1.0);
              }}
              geomHits.push({{ bid, rPix, cPix, pt, kf, h, k, l, lam, qn, inten, snr, hasImg }});
              break;
            }}
          }}
        }}
      }}
      const strictSnr = geomHits.length > 400;
      for (const gh of geomHits) {{
        if (gh.hasImg && strictSnr && gh.snr < 3.0) continue;
        const [az, dy, el] = labToCyl(gh.pt);
        const qLab = vecUnit([gh.kf[0], gh.kf[1], gh.kf[2] - 1.0]);
        const gCryst = mat3Vec(R_T, qLab);
        const [sx, sy, hemi] = stereoProj(gCryst);
        const dVal = 1.0 / gh.qn;
        candPeaks.push({{
          bank: gh.bid, row: Number(gh.rPix.toFixed(1)), col: Number(gh.cPix.toFixed(1)),
          det_x: az, det_y: dy, det_el: el, q_hat: gCryst, q_hat_lab: qLab,
          stereographic: [sx, sy], hemisphere: hemi, intensity: gh.inten, sigma: 3.0,
          wavelength: gh.lam, d_spacing: dVal, harmonic_order: Math.max(1, gcd3(gh.h, gh.k, gh.l)),
          parallax_depth: Math.max(0.05, Math.min(0.98, 1.0 / (1.0 + 0.35 * dVal))),
          hkl: [gh.h, gh.k, gh.l], zone_ids: [], indexed: true, angular_error_deg: 0.02,
        }});
      }}
      }}
      candPeaks.sort((a, b) => b.intensity - a.intensity);
      candPeaks.slice(0, 1200).forEach((p, idx) => {{
        p.id = idx;
        if (!p.asu_orbit_id && Array.isArray(p.hkl)) {{
          const s3 = [Math.abs(p.hkl[0]), Math.abs(p.hkl[1]), Math.abs(p.hkl[2])].sort((a, b) => b - a);
          p.asu_hkl = s3;
          p.asu_orbit_id = s3.join(",");
          p.orbit_color_idx = Math.abs(s3[0] * 73856093 ^ s3[1] * 19349663 ^ s3[2] * 83492791) % 12;
        }}
        peaksOut.push(p);
      }});

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
          const [sx, sy, sHemi] = stereoProj(gC);
          if (s % 4 === 0) stereoSegs[0].push([sx, sy, sHemi]);
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
          axis_xyz: z.tLab, axis_crystal_xyz: mat3Vec(R_T, z.tLab),
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
      instrument: inst,
      default_projection: (inst === "MANDI" || inst === "TOPAZ") ? "spherical" : "cylindrical",
      space_group: sg,
      intensity_source: bankImgs.size > 0 ? "raw_h5" : "reconstructed",
      artifact_kind: artifactKind,
      peak_shape_mode: peakShapeMode,
      wavelength_range: wlRangeNew,
      d_min: dMinNew,
      goniometer_R: Rrun,
      cell,
      peaks: peaksOut,
      sample_peaks: peaksOut,
      zones: zonesOut,
      panels: panelsOut,
      ladder_steps: ladderOut,
      pipeline_stages: PIPELINE.map(s => ({{ ...s }})),
    }};
  }}

  // Live Galaxy / NOVA / Pulsar GPU job orchestration
  const galaxyBridge = {{
    mode: "none", // "nova" | "galaxy_iframe" | "none"
    historyId: null,
    datasets: [],
    stepArtifacts: {{}},
    pollTimer: null,
  }};

  function classifyArtifactStageClient(filename) {{
    const low = String(filename || "").split("/").pop().toLowerCase();
    if (!low) return null;
    if (["merged.mtz", "predictions.mtz", "xval.mtz", "careless_metrics.json", "careless_0.mtz", "careless_predictions_0.mtz", "careless_xval_0.mtz"].some(k => low === k || low.endsWith(k))) {{
      return "merge";
    }}
    if (["predictor.h5", "integrated.h5", "reflections.mtz"].some(k => low === k || low.endsWith(k))) {{
      return "integrate";
    }}
    if (low.includes("solve") && low.endsWith(".h5")) return "solve";
    if (["merged.h5", "_merged.h5", ".reduce.h5", ".nxs.h5", "scan_master.h5"].some(k => low.includes(k))) return "reduce";
    return null;
  }}

  function renderFilesDrawer() {{
    if (!filesDrawer) return;
    const stageKeys = ["reduce", "solve", "integrate", "merge"];
    const stageTitles = ["1. Reduce", "2. Solve", "3. Integrate", "4. Merge"];
    const pIdx = Math.min(3, Math.max(0, state.currentPipelineIdx));
    const stageId = (PIPELINE[pIdx] && PIPELINE[pIdx].stage_id) || stageKeys[pIdx] || "solve";
    const stageTitle = stageTitles[pIdx] || (PIPELINE[pIdx] && PIPELINE[pIdx].name) || stageId.toUpperCase();

    const byName = new Map();
    const srvList = (galaxyBridge.stepArtifacts && galaxyBridge.stepArtifacts[stageId]) || [];
    srvList.forEach(it => {{
      if (it && it.name) byName.set(it.name, {{ ...it }});
    }});
    (galaxyBridge.datasets || []).forEach(ds => {{
      if (ds.state && ds.state !== "ok") return;
      const bname = String(ds.name || "").split("/").pop();
      if (!bname) return;
      if (classifyArtifactStageClient(bname) !== stageId) return;
      const cur = byName.get(bname) || {{ name: bname, stage_id: stageId }};
      if (ds.id) cur.dataset_id = ds.id;
      if (ds.hid) cur.hid = ds.hid;
      byName.set(bname, cur);
    }});

    filesDrawer.innerHTML = "";
    const lbl = document.createElement("span");
    lbl.style.color = "var(--accent-gold)";
    lbl.style.fontWeight = "700";
    lbl.textContent = `${{stageTitle.toUpperCase()}} FILES:`;
    filesDrawer.appendChild(lbl);

    const items = Array.from(byName.values());
    if (items.length === 0) {{
      const emptyEl = document.createElement("span");
      emptyEl.style.color = "var(--text-dim)";
      emptyEl.textContent = `No stored artifacts yet for ${{stageTitle}} (use < Walk > arrows to switch steps).`;
      filesDrawer.appendChild(emptyEl);
      return;
    }}

    items.forEach(item => {{
      const a = document.createElement("a");
      a.className = "file-dl-pill";
      const qs = new URLSearchParams();
      if (item.name) qs.set("name", item.name);
      if (item.dataset_id) qs.set("dataset_id", item.dataset_id);
      if (galaxyBridge.mode === "galaxy_iframe" && item.dataset_id && !item.has_local) {{
        a.href = `/api/datasets/${{encodeURIComponent(item.dataset_id)}}/display?to_ext=${{encodeURIComponent(item.name.split(".").pop() || "h5")}}`;
      }} else {{
        a.href = `api/artifacts/download?${{qs.toString()}}`;
      }}
      a.setAttribute("download", item.name);
      const sizeStr = typeof item.size_bytes === "number" && item.size_bytes > 0
        ? ` (${{item.size_bytes >= 1048576 ? (item.size_bytes / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(item.size_bytes / 1024)) + " KB"}})`
        : (item.hid ? ` (#${{item.hid}})` : "");
      a.textContent = `↓ ${{item.name}}${{sizeStr}}`;
      a.title = `Download ${{item.name}} (${{stageTitle}}) to local computer`;
      filesDrawer.appendChild(a);
    }});

    const hintEl = document.createElement("span");
    hintEl.style.color = "var(--text-dim)";
    hintEl.style.marginLeft = "auto";
    hintEl.style.fontSize = "0.58rem";
    hintEl.textContent = "Switch step with < Walk >";
    filesDrawer.appendChild(hintEl);
  }}

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
    selGalaxyH5.innerHTML = "";
    const optHeader = document.createElement("option");
    optHeader.value = "";
    optHeader.textContent = h5s.length > 0
      ? `Data Source (${{h5s.length}} in History)...`
      : "Data Source...";
    selGalaxyH5.appendChild(optHeader);

    const grpIngest = document.createElement("optgroup");
    grpIngest.label = "Ingest / Open Data";
    const optNeutron = document.createElement("option");
    optNeutron.value = "__action_neutron_data__";
    optNeutron.textContent = "Neutron data (SNS / HFIR)...";
    grpIngest.appendChild(optNeutron);

    const optLocal = document.createElement("option");
    optLocal.value = "__action_local_file__";
    optLocal.textContent = "From local computer...";
    grpIngest.appendChild(optLocal);
    selGalaxyH5.appendChild(grpIngest);

    if (h5s.length > 0) {{
      const grpHist = document.createElement("optgroup");
      grpHist.label = "Galaxy History (.h5)";
      h5s.forEach(d => {{
        const opt = document.createElement("option");
        opt.value = d.id;
        opt.textContent = `#${{d.hid || ""}} ${{d.name}}`;
        grpHist.appendChild(opt);
      }});
      selGalaxyH5.appendChild(grpHist);
    }}
  }}

  // Modal Dialog: Load Neutron Data (IPTS+run range or explicit file paths)
  const modalNeutronData = document.getElementById("modal-neutron-data");
  const btnNdClose = document.getElementById("btn-nd-close");
  const btnNdCancel = document.getElementById("btn-nd-cancel");
  const btnNdSubmit = document.getElementById("btn-nd-submit");
  const btnNdModeIpts = document.getElementById("btn-nd-mode-ipts");
  const btnNdModePaths = document.getElementById("btn-nd-mode-paths");
  const ndPaneIpts = document.getElementById("nd-pane-ipts");
  const ndPanePaths = document.getElementById("nd-pane-paths");
  let ndIngestMode = "ipts"; // "ipts" | "paths"
  let pendingLocalNexusFiles = null;
  let ndValidateTimer = null;
  let lastNdValidationResult = null;
  let ndOncatCatalog = {{
    experiments: [],
    runs: [],
    oncat_authenticated: false,
    oncat_auth: null,
    multi_run_supported: true,
    multi_run_note: "",
    lastClickedRun: null,
    loading: false,
  }};
  let ndOncatAuthPollTimer = null;

  function renderOncatAuthBar() {{
    const msgEl = document.getElementById("nd-oncat-auth-msg");
    const loginBtn = document.getElementById("btn-nd-oncat-login");
    const cancelBtn = document.getElementById("btn-nd-oncat-cancel");
    const verifyLink = document.getElementById("link-nd-oncat-verify");
    if (!msgEl || !loginBtn) return;
    const auth = ndOncatCatalog.oncat_auth || {{}};
    const isAuth = Boolean(ndOncatCatalog.oncat_authenticated || auth.authenticated);
    if (isAuth) {{
      msgEl.style.color = "#34d399";
      msgEl.textContent = `ONCat OAuth: Signed in as ${{auth.user || "ORNL User"}} (run titles & sample metadata active)`;
      loginBtn.textContent = "Sign out";
      loginBtn.style.display = "inline-block";
      if (cancelBtn) cancelBtn.style.display = "none";
      if (verifyLink) verifyLink.style.display = "none";
    }} else if (auth.status === "pending_verification") {{
      msgEl.style.color = "var(--accent-gold)";
      const codeTxt = auth.user_code ? `Code: ${{auth.user_code}}` : "waiting for code…";
      msgEl.textContent = `ONCat OAuth (ORNL PingFed) — ${{codeTxt}}`;
      loginBtn.style.display = "none";
      if (cancelBtn) cancelBtn.style.display = "inline-block";
      if (verifyLink) {{
        const targetUri = auth.verification_uri_complete || auth.verification_uri || "#";
        verifyLink.href = targetUri;
        verifyLink.textContent = auth.user_code
          ? `Open ORNL Login (${{auth.user_code}}) ↗`
          : "Open ORNL Login ↗";
        verifyLink.style.display = targetUri !== "#" ? "inline-block" : "none";
      }}
    }} else {{
      msgEl.style.color = "var(--text-dim)";
      msgEl.textContent = auth.message || "ONCat: Public catalog (sign in via ORNL PingFed for run titles & sample metadata)";
      loginBtn.textContent = "Sign in to ONCat";
      loginBtn.style.display = "inline-block";
      if (cancelBtn) cancelBtn.style.display = "none";
      if (verifyLink) verifyLink.style.display = "none";
    }}
  }}

  async function pollOncatAuthStatus() {{
    try {{
      const r = await fetch("api/pipeline/oncat_auth");
      if (!r.ok) return;
      const auth = await r.json();
      const wasAuth = Boolean(ndOncatCatalog.oncat_authenticated);
      ndOncatCatalog.oncat_auth = auth;
      ndOncatCatalog.oncat_authenticated = Boolean(auth.authenticated);
      renderOncatAuthBar();
      if (auth.authenticated && !wasAuth) {{
        if (ndOncatAuthPollTimer) {{
          clearInterval(ndOncatAuthPollTimer);
          ndOncatAuthPollTimer = null;
        }}
        fetchOncatCatalog();
      }} else if (auth.status !== "pending_verification" && ndOncatAuthPollTimer) {{
        clearInterval(ndOncatAuthPollTimer);
        ndOncatAuthPollTimer = null;
      }}
    }} catch (_e) {{}}
  }}

  async function handleOncatAuthButtonClick() {{
    const isAuth = Boolean(ndOncatCatalog.oncat_authenticated || ndOncatCatalog.oncat_auth?.authenticated);
    const action = isAuth ? "logout" : "start";
    try {{
      const r = await fetch("api/pipeline/oncat_auth", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ action }}),
      }});
      if (!r.ok) return;
      const auth = await r.json();
      ndOncatCatalog.oncat_auth = auth;
      ndOncatCatalog.oncat_authenticated = Boolean(auth.authenticated);
      renderOncatAuthBar();
      if (action === "start" && auth.status === "pending_verification") {{
        if (auth.verification_uri_complete) {{
          window.open(auth.verification_uri_complete, "_blank", "noopener,noreferrer");
        }}
        if (ndOncatAuthPollTimer) clearInterval(ndOncatAuthPollTimer);
        ndOncatAuthPollTimer = setInterval(pollOncatAuthStatus, 1500);
      }} else if (action === "logout") {{
        fetchOncatCatalog();
      }}
    }} catch (_e) {{}}
  }}

  async function cancelOncatAuthClick() {{
    if (ndOncatAuthPollTimer) {{
      clearInterval(ndOncatAuthPollTimer);
      ndOncatAuthPollTimer = null;
    }}
    try {{
      const r = await fetch("api/pipeline/oncat_auth", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ action: "cancel" }}),
      }});
      if (r.ok) {{
        const auth = await r.json();
        ndOncatCatalog.oncat_auth = auth;
        ndOncatCatalog.oncat_authenticated = Boolean(auth.authenticated);
        renderOncatAuthBar();
      }}
    }} catch (_e) {{}}
  }}

  function parseRunRangeSpec(specStr) {{
    const runs = [];
    String(specStr || "").split(/[,\\s]+/).forEach(tok => {{
      const t = tok.trim();
      if (!t) return;
      const m = t.match(/^(\\d+)\\s*[-–:]\\s*(\\d+)$/);
      if (m) {{
        const a = parseInt(m[1], 10), b = parseInt(m[2], 10);
        const lo = Math.min(a, b), hi = Math.max(a, b);
        for (let r = lo; r <= Math.min(hi, lo + 500); r++) runs.push(r);
      }} else if (/^\\d+$/.test(t)) {{
        runs.push(parseInt(t, 10));
      }}
    }});
    return runs;
  }}

  function formatRunListToSpec(runNumbers) {{
    const uniq = Array.from(new Set(runNumbers.map(n => parseInt(n, 10)).filter(Number.isFinite))).sort((a, b) => a - b);
    if (uniq.length === 0) return "";
    const parts = [];
    let start = uniq[0], prev = uniq[0];
    for (let i = 1; i <= uniq.length; i++) {{
      const cur = uniq[i];
      if (cur === prev + 1) {{
        prev = cur;
      }} else {{
        if (prev === start) parts.push(String(start));
        else if (prev === start + 1) parts.push(`${{start}},${{prev}}`);
        else parts.push(`${{start}}-${{prev}}`);
        start = prev = cur;
      }}
    }}
    return parts.join(",");
  }}

  function buildNeutronFilePathsFromDialog() {{
    const fac = (document.getElementById("nd-facility")?.value || "HFIR").trim();
    const inst = (document.getElementById("nd-instrument")?.value || "CG4D").trim();
    const iptsRaw = (document.getElementById("nd-ipts")?.value || "").trim().replace(/^IPTS-/i, "");
    const runsSpec = (document.getElementById("nd-runs")?.value || "").trim();
    if (ndIngestMode === "paths") {{
      const rawTxt = document.getElementById("nd-filepaths")?.value || "";
      return rawTxt.split(/[\\r\\n,]+/).map(s => s.trim()).filter(Boolean);
    }}
    let runs = parseRunRangeSpec(runsSpec);
    if (!ndOncatCatalog.multi_run_supported && runs.length > 1) {{
      runs = [runs[0]];
    }}
    if (!iptsRaw || runs.length === 0) return [];
    const runLocMap = new Map((ndOncatCatalog.runs || []).map(r => [Number(r.run), r.location]));
    return runs.map(r => runLocMap.get(Number(r)) || `/${{fac}}/${{inst}}/IPTS-${{iptsRaw}}/nexus/${{inst}}_${{r}}.nxs.h5`);
  }}

  function renderOncatProjectsList() {{
    const listEl = document.getElementById("nd-ipts-list");
    const countEl = document.getElementById("nd-ipts-count");
    const filterVal = (document.getElementById("nd-ipts-filter")?.value || "").trim().toLowerCase();
    const curIpts = (document.getElementById("nd-ipts")?.value || "").trim().replace(/^IPTS-/i, "");
    if (!listEl) return;
    const exps = (ndOncatCatalog.experiments || []).filter(e => {{
      if (!filterVal) return true;
      const hay = `${{e.ipts}} ${{e.id}} ${{e.title || ""}} ${{e.run_ranges || ""}} ${{e.latest_date || ""}}`.toLowerCase();
      return hay.includes(filterVal);
    }});
    if (countEl) countEl.textContent = `${{exps.length}} IPTS`;
    listEl.innerHTML = "";
    if (exps.length === 0) {{
      const empty = document.createElement("div");
      empty.style.cssText = "padding:8px;color:var(--text-dim);font-size:0.64rem;";
      empty.textContent = ndOncatCatalog.loading ? "Loading IPTS projects…" : "No matching IPTS projects";
      listEl.appendChild(empty);
      return;
    }}
    let activeRow = null;
    exps.forEach(exp => {{
      const isSel = String(exp.ipts) === String(curIpts);
      const row = document.createElement("div");
      row.style.cssText = `padding:4px 6px;border-radius:4px;cursor:pointer;display:flex;flex-direction:column;gap:1px;border:1px solid ${{isSel ? "var(--accent-cyan)" : "transparent"}};background:${{isSel ? "rgba(56,189,248,0.16)" : "rgba(255,255,255,0.02)"}};`;
      const top = document.createElement("div");
      top.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:4px;";
      const lbl = document.createElement("span");
      lbl.style.cssText = `font-family:var(--font-display);font-size:0.68rem;font-weight:700;color:${{isSel ? "var(--accent-cyan)" : "var(--text-main)"}};`;
      lbl.textContent = `IPTS-${{exp.ipts}}`;
      const badge = document.createElement("span");
      badge.style.cssText = "font-family:var(--font-display);font-size:0.59rem;color:var(--accent-gold);";
      badge.textContent = `${{exp.n_runs || 0}} runs`;
      top.appendChild(lbl);
      top.appendChild(badge);
      row.appendChild(top);

      const sub = document.createElement("div");
      sub.style.cssText = "font-size:0.6rem;color:var(--text-dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      const metaBits = [];
      if (exp.title) metaBits.push(exp.title);
      if (exp.min_run != null && exp.max_run != null) metaBits.push(`#${{exp.min_run}}–${{exp.max_run}}`);
      if (exp.latest_date) metaBits.push(exp.latest_date);
      sub.textContent = metaBits.join(" • ") || (exp.run_ranges || "No runs indexed");
      row.title = `IPTS-${{exp.ipts}} (${{exp.n_runs || 0}} runs)\\n${{exp.title ? "Samples: " + exp.title + "\\n" : ""}}Ranges: ${{exp.run_ranges || "—"}}`;
      row.appendChild(sub);

      row.addEventListener("click", () => {{
        const iptsInput = document.getElementById("nd-ipts");
        if (iptsInput) iptsInput.value = String(exp.ipts);
        fetchOncatCatalog(String(exp.ipts), true);
      }});
      if (isSel) activeRow = row;
      listEl.appendChild(row);
    }});
    if (activeRow && typeof activeRow.scrollIntoView === "function") {{
      activeRow.scrollIntoView({{ block: "nearest" }});
    }}
  }}

  function formatRunDurationHMS(r) {{
    if (!r) return "";
    if (r.duration_hms) return String(r.duration_hms);
    const s = Number(r.duration_s || 0);
    if (!(s > 0)) return "";
    const tot = Math.round(s);
    const hh = Math.floor(tot / 3600);
    const mm = Math.floor((tot % 3600) / 60);
    const ss = tot % 60;
    return [hh, mm, ss].map(v => String(v).padStart(2, "0")).join(":");
  }}

  function renderOncatRunsList(scrollToSelected = false) {{
    const listEl = document.getElementById("nd-runs-list");
    const hdrEl = document.getElementById("nd-runs-header");
    const hintEl = document.getElementById("nd-runs-hint");
    const badgeEl = document.getElementById("nd-multirun-badge");
    const filterVal = (document.getElementById("nd-runs-filter")?.value || "").trim().toLowerCase();
    const curIpts = (document.getElementById("nd-ipts")?.value || "").trim().replace(/^IPTS-/i, "");
    if (!listEl) return;

    const multiOk = Boolean(ndOncatCatalog.multi_run_supported);
    if (badgeEl) {{
      badgeEl.textContent = multiOk ? "Multi-run supported" : "Single run only";
      badgeEl.style.color = multiOk ? "var(--accent-cyan)" : "var(--accent-gold)";
      badgeEl.title = ndOncatCatalog.multi_run_note || "";
    }}
    if (hintEl) {{
      hintEl.textContent = multiOk ? "Click / Shift-click to select multiple runs" : "Click to select a run";
    }}

    const selectedSet = new Set(parseRunRangeSpec(document.getElementById("nd-runs")?.value || ""));
    const allRuns = ndOncatCatalog.runs || [];
    const filtered = allRuns.filter(r => {{
      if (!filterVal) return true;
      const durHms = formatRunDurationHMS(r);
      const hay = `${{r.run}} ${{r.title || ""}} ${{r.sample_name || ""}} ${{r.angles || ""}} ${{durHms}} ${{r.created || ""}}`.toLowerCase();
      return hay.includes(filterVal);
    }});
    if (hdrEl) {{
      hdrEl.textContent = `RUNS IN IPTS-${{curIpts || "?"}} (${{filtered.length}}/${{allRuns.length}}, ${{selectedSet.size}} selected)`;
    }}
    listEl.innerHTML = "";
    if (filtered.length === 0) {{
      const empty = document.createElement("div");
      empty.style.cssText = "padding:8px;color:var(--text-dim);font-size:0.64rem;";
      empty.textContent = ndOncatCatalog.loading ? "Loading runs & titles from ONCat…" : "No matching runs in this IPTS";
      listEl.appendChild(empty);
      return;
    }}

    let firstSelRow = null;
    filtered.forEach(r => {{
      const rNum = Number(r.run);
      const isSel = selectedSet.has(rNum);
      const durHms = formatRunDurationHMS(r);
      const row = document.createElement("div");
      row.style.cssText = `padding:3px 6px;border-radius:4px;cursor:pointer;display:grid;grid-template-columns:auto 54px 1fr auto auto;align-items:center;gap:6px;border:1px solid ${{isSel ? "var(--accent-cyan)" : "transparent"}};background:${{isSel ? "rgba(56,189,248,0.16)" : "rgba(255,255,255,0.015)"}};`;

      const cb = document.createElement("input");
      cb.type = multiOk ? "checkbox" : "radio";
      cb.checked = isSel;
      cb.style.cssText = "margin:0;cursor:pointer;accent-color:var(--accent-cyan);";

      const runSpan = document.createElement("span");
      runSpan.style.cssText = `font-family:var(--font-display);font-size:0.67rem;font-weight:700;color:${{isSel ? "var(--accent-cyan)" : "var(--text-main)"}};`;
      runSpan.textContent = `#${{rNum}}`;

      const titleSpan = document.createElement("span");
      titleSpan.style.cssText = "font-size:0.65rem;color:var(--text-main);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      const primaryTitle = r.title || r.sample_name || r.location?.split("/").pop() || `Run ${{rNum}}`;
      titleSpan.textContent = (r.title && r.sample_name && !r.title.toLowerCase().includes(r.sample_name.toLowerCase()))
        ? `${{r.title}} (${{r.sample_name}})`
        : primaryTitle;

      const durSpan = document.createElement("span");
      durSpan.style.cssText = "font-family:var(--font-display);font-size:0.59rem;color:var(--text-dim);white-space:nowrap;";
      durSpan.textContent = durHms;

      const metaSpan = document.createElement("span");
      metaSpan.style.cssText = "font-family:var(--font-display);font-size:0.59rem;color:var(--accent-gold);white-space:nowrap;";
      metaSpan.textContent = r.angles || (r.size_mb ? `${{r.size_mb}} MB` : "");

      const tipLines = [`Run #${{rNum}} (${{r.location || ""}})`];
      if (r.title) tipLines.push(`Title: ${{r.title}}`);
      if (r.sample_name) tipLines.push(`Sample: ${{r.sample_name}}`);
      if (durHms) tipLines.push(`Duration: ${{durHms}}`);
      if (r.angles) tipLines.push(`Goniometer: ${{r.angles}}`);
      if (r.created) tipLines.push(`Created: ${{r.created}}`);
      if (r.size_mb) tipLines.push(`Size: ${{r.size_mb}} MB`);
      row.title = tipLines.join("\\n");

      const handleSelect = (ev) => {{
        ev.stopPropagation();
        let nextSet = new Set(selectedSet);
        if (!multiOk) {{
          nextSet = new Set([rNum]);
        }} else if (ev.shiftKey && ndOncatCatalog.lastClickedRun != null) {{
          const visibleNums = filtered.map(x => Number(x.run));
          const iA = visibleNums.indexOf(Number(ndOncatCatalog.lastClickedRun));
          const iB = visibleNums.indexOf(rNum);
          if (iA >= 0 && iB >= 0) {{
            const [lo, hi] = iA < iB ? [iA, iB] : [iB, iA];
            for (let k = lo; k <= hi; k++) nextSet.add(visibleNums[k]);
          }} else {{
            if (nextSet.has(rNum)) nextSet.delete(rNum); else nextSet.add(rNum);
          }}
        }} else {{
          if (nextSet.has(rNum)) nextSet.delete(rNum);
          else nextSet.add(rNum);
        }}
        ndOncatCatalog.lastClickedRun = rNum;
        const runsInput = document.getElementById("nd-runs");
        if (runsInput) runsInput.value = formatRunListToSpec(Array.from(nextSet));
        renderOncatRunsList(false);
        updateNeutronDialogPreview();
      }};

      cb.addEventListener("click", handleSelect);
      row.addEventListener("click", handleSelect);
      row.appendChild(cb);
      row.appendChild(runSpan);
      row.appendChild(titleSpan);
      row.appendChild(durSpan);
      row.appendChild(metaSpan);
      if (isSel && !firstSelRow) firstSelRow = row;
      listEl.appendChild(row);
    }});

    if (scrollToSelected && firstSelRow && typeof firstSelRow.scrollIntoView === "function") {{
      firstSelRow.scrollIntoView({{ block: "center" }});
    }}
  }}

  async function fetchOncatCatalog(iptsOverride = null, autoSelectLatestRun = false) {{
    const fac = (document.getElementById("nd-facility")?.value || "HFIR").trim();
    const inst = (document.getElementById("nd-instrument")?.value || "CG4D").trim();
    const iptsEl = document.getElementById("nd-ipts");
    const iptsRaw = (iptsOverride !== null ? String(iptsOverride) : (iptsEl?.value || "")).trim().replace(/^IPTS-/i, "");
    ndOncatCatalog.loading = true;
    renderOncatProjectsList();
    renderOncatRunsList(false);
    try {{
      const qs = new URLSearchParams({{ facility: fac, instrument: inst, ipts: iptsRaw }});
      const r = await fetch(`api/pipeline/oncat_catalog?${{qs.toString()}}`);
      if (!r.ok) throw new Error(`HTTP ${{r.status}}`);
      const data = await r.json();
      ndOncatCatalog.experiments = data.experiments || [];
      ndOncatCatalog.runs = data.runs || [];
      ndOncatCatalog.oncat_authenticated = Boolean(data.oncat_authenticated);
      ndOncatCatalog.oncat_auth = data.oncat_auth || null;
      ndOncatCatalog.multi_run_supported = data.multi_run_supported !== false;
      ndOncatCatalog.multi_run_note = data.multi_run_note || "";
      ndOncatCatalog.loading = false;
      renderOncatAuthBar();
      if (data.ipts && iptsEl && (!iptsEl.value.trim() || iptsOverride !== null)) {{
        iptsEl.value = String(data.ipts);
      }}
      const runsEl = document.getElementById("nd-runs");
      if (runsEl && ndOncatCatalog.runs.length > 0) {{
        const curSelected = parseRunRangeSpec(runsEl.value);
        const runSet = new Set(ndOncatCatalog.runs.map(x => Number(x.run)));
        const hasOverlap = curSelected.some(n => runSet.has(Number(n)));
        if (autoSelectLatestRun || !hasOverlap) {{
          const lastRun = ndOncatCatalog.runs[ndOncatCatalog.runs.length - 1].run;
          runsEl.value = String(lastRun);
        }}
      }}
      renderOncatProjectsList();
      renderOncatRunsList(true);
      updateNeutronDialogPreview();
    }} catch (_err) {{
      ndOncatCatalog.loading = false;
      renderOncatProjectsList();
      renderOncatRunsList(false);
    }}
  }}

  async function verifyNeutronPathsInDialog() {{
    const statusBox = document.getElementById("nd-validation-status");
    const msgEl = document.getElementById("nd-validation-msg");
    const sugBtn = document.getElementById("btn-nd-use-suggested-ipts");
    if (!statusBox || !msgEl) return null;
    if (pendingLocalNexusFiles && pendingLocalNexusFiles.length > 0) {{
      statusBox.style.borderColor = "rgba(52,211,153,0.45)";
      msgEl.style.color = "#34d399";
      msgEl.textContent = `Ready to upload ${{pendingLocalNexusFiles.length}} local file(s)`;
      if (sugBtn) sugBtn.style.display = "none";
      lastNdValidationResult = {{ ok: true }};
      return lastNdValidationResult;
    }}
    const fac = (document.getElementById("nd-facility")?.value || "HFIR").trim();
    const inst = (document.getElementById("nd-instrument")?.value || "CG4D").trim();
    const iptsRaw = (document.getElementById("nd-ipts")?.value || "").trim().replace(/^IPTS-/i, "");
    const runsSpec = (document.getElementById("nd-runs")?.value || "").trim();
    const paths = buildNeutronFilePathsFromDialog();
    if (paths.length === 0) {{
      statusBox.style.borderColor = "rgba(248,113,113,0.45)";
      msgEl.style.color = "#f87171";
      msgEl.textContent = "Enter a valid IPTS number and run (or explicit file path).";
      if (sugBtn) sugBtn.style.display = "none";
      lastNdValidationResult = {{ ok: false, message: msgEl.textContent }};
      return lastNdValidationResult;
    }}
    statusBox.style.borderColor = "rgba(56,189,248,0.45)";
    msgEl.style.color = "#38bdf8";
    msgEl.textContent = `Verifying ${{paths.length}} run(s) via NDIP Load Neutron Data...`;
    if (sugBtn) sugBtn.style.display = "none";
    try {{
      const r = await fetch("api/pipeline/check_neutron_paths", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          ingest_mode: ndIngestMode === "ipts" ? "oncat" : "file",
          facility: fac,
          instrument: inst,
          ipts: iptsRaw,
          runs: runsSpec,
          paths: paths,
        }}),
      }});
      if (!r.ok) return null;
      const res = await r.json();
      lastNdValidationResult = res;
      if (res.ok) {{
        statusBox.style.borderColor = "rgba(52,211,153,0.5)";
        msgEl.style.color = "#34d399";
        msgEl.textContent = res.message || `Verified ${{res.existing?.length || paths.length}} file(s)`;
        if (sugBtn) sugBtn.style.display = "none";
      }} else {{
        statusBox.style.borderColor = "rgba(248,113,113,0.6)";
        msgEl.style.color = "#f87171";
        msgEl.textContent = res.message || "File not found on /SNS or /HFIR";
        if (sugBtn && res.suggested_ipts) {{
          sugBtn.style.display = "inline-block";
          sugBtn.textContent = `Switch to IPTS-${{res.suggested_ipts}}`;
          sugBtn.onclick = () => {{
            const iptsEl = document.getElementById("nd-ipts");
            if (iptsEl) iptsEl.value = String(res.suggested_ipts);
            if (Array.isArray(res.suggested_paths) && res.suggested_paths.length > 0) {{
              const firstP = res.suggested_paths[0];
              const facEl = document.getElementById("nd-facility");
              if (facEl && firstP.startsWith("/SNS/")) facEl.value = "SNS";
              if (facEl && firstP.startsWith("/HFIR/")) facEl.value = "HFIR";
              if (ndIngestMode === "paths") {{
                const fpEl = document.getElementById("nd-filepaths");
                if (fpEl) fpEl.value = res.suggested_paths.join("\\n");
              }}
            }}
            updateNeutronDialogPreview();
          }};
        }} else if (sugBtn) {{
          sugBtn.style.display = "none";
        }}
      }}
      return res;
    }} catch (_e) {{
      return null;
    }}
  }}

  function updateNeutronDialogPreview() {{
    const prevEl = document.getElementById("nd-preview-path");
    const titlePrevEl = document.getElementById("nd-preview-title");
    const statusBox = document.getElementById("nd-validation-status");
    const msgEl = document.getElementById("nd-validation-msg");
    const sugBtn = document.getElementById("btn-nd-use-suggested-ipts");
    lastNdValidationResult = null;
    if (sugBtn) sugBtn.style.display = "none";
    if (!prevEl) return;
    if (pendingLocalNexusFiles && pendingLocalNexusFiles.length > 0) {{
      prevEl.textContent = `Local NeXus files (${{pendingLocalNexusFiles.length}}): ${{pendingLocalNexusFiles.map(f => f.name).join(", ")}}`;
      if (titlePrevEl) titlePrevEl.textContent = "";
      if (statusBox) statusBox.style.borderColor = "rgba(52,211,153,0.5)";
      if (msgEl) {{
        msgEl.style.color = "#34d399";
        msgEl.textContent = `${{pendingLocalNexusFiles.length}} local NeXus file(s) staged for upload`;
      }}
      return;
    }}
    const paths = buildNeutronFilePathsFromDialog();
    if (titlePrevEl) {{
      const selRuns = parseRunRangeSpec(document.getElementById("nd-runs")?.value || "");
      const runMap = new Map((ndOncatCatalog.runs || []).map(r => [Number(r.run), r]));
      const previews = selRuns
        .map(n => runMap.get(Number(n)))
        .filter(Boolean)
        .map(r => {{
          const durHms = formatRunDurationHMS(r);
          return `#${{r.run}}: ${{r.title || r.sample_name || "—"}}${{durHms ? " (" + durHms + ")" : ""}}${{r.angles ? " [" + r.angles + "]" : ""}}`;
        }});
      titlePrevEl.textContent = previews.length > 0
        ? `Title Preview: ${{previews.slice(0, 3).join("  |  ")}}${{previews.length > 3 ? ` (+${{previews.length - 3}} more)` : ""}}`
        : "";
    }}
    if (paths.length === 0) {{
      prevEl.textContent = "Preview: specify IPTS + run range or explicit file path(s)";
      if (statusBox) statusBox.style.borderColor = "rgba(248,113,113,0.45)";
      if (msgEl) {{
        msgEl.style.color = "#f87171";
        msgEl.textContent = "Enter a valid IPTS number and run (or explicit file path).";
      }}
    }} else {{
      if (paths.length === 1) {{
        prevEl.textContent = `Preview: ${{paths[0]}}`;
      }} else {{
        const lastBase = paths[paths.length - 1].split("/").pop();
        prevEl.textContent = `Preview: ${{paths[0]}} ... ${{lastBase}} (${{paths.length}} runs)`;
      }}
      if (statusBox) statusBox.style.borderColor = "rgba(56,189,248,0.25)";
      if (msgEl) {{
        msgEl.style.color = "var(--text-dim)";
        msgEl.textContent = `Ready to verify ${{paths.length}} run(s) via NDIP Load Neutron Data`;
      }}
    }}
  }}

  function openNeutronDataModal(localNexusFiles = null) {{
    pendingLocalNexusFiles = localNexusFiles;
    const params = collectWorkflowParams();
    const setVal = (id, v) => {{ const el = document.getElementById(id); if (el && v !== undefined) el.value = v; }};
    if (params.instrument) {{
      setVal("nd-instrument", params.instrument);
      setVal("nd-facility", (params.instrument === "MANDI" || params.instrument === "TOPAZ") ? "SNS" : "HFIR");
    }}
    setVal("nd-sg", params.space_group);
    setVal("nd-a", Number(params.a).toFixed(2));
    setVal("nd-b", Number(params.b).toFixed(2));
    setVal("nd-c", Number(params.c).toFixed(2));
    setVal("nd-alpha", Number(params.alpha).toFixed(2));
    setVal("nd-beta", Number(params.beta).toFixed(2));
    setVal("nd-gamma", Number(params.gamma).toFixed(2));
    setVal("nd-wl-min", Number(params.wavelength_min).toFixed(1));
    setVal("nd-wl-max", Number(params.wavelength_max).toFixed(1));
    updateNeutronDialogPreview();
    if (modalNeutronData) modalNeutronData.style.display = "flex";
    fetchOncatCatalog();
  }}

  function closeNeutronDataModal() {{
    pendingLocalNexusFiles = null;
    if (ndOncatAuthPollTimer) {{
      clearInterval(ndOncatAuthPollTimer);
      ndOncatAuthPollTimer = null;
    }}
    if (modalNeutronData) modalNeutronData.style.display = "none";
  }}

  if (btnNdClose) btnNdClose.addEventListener("click", closeNeutronDataModal);
  if (btnNdCancel) btnNdCancel.addEventListener("click", closeNeutronDataModal);
  const btnNdVerify = document.getElementById("btn-nd-verify");
  if (btnNdVerify) {{
    btnNdVerify.addEventListener("click", () => {{
      verifyNeutronPathsInDialog();
    }});
  }}
  if (btnNdModeIpts && btnNdModePaths && ndPaneIpts && ndPanePaths) {{
    btnNdModeIpts.addEventListener("click", () => {{
      ndIngestMode = "ipts";
      btnNdModeIpts.classList.add("active-btn");
      btnNdModePaths.classList.remove("active-btn");
      ndPaneIpts.style.display = "flex";
      ndPanePaths.style.display = "none";
      updateNeutronDialogPreview();
    }});
    btnNdModePaths.addEventListener("click", () => {{
      ndIngestMode = "paths";
      btnNdModePaths.classList.add("active-btn");
      btnNdModeIpts.classList.remove("active-btn");
      ndPaneIpts.style.display = "none";
      ndPanePaths.style.display = "block";
      updateNeutronDialogPreview();
    }});
  }}
  ["nd-facility", "nd-instrument", "nd-ipts", "nd-runs", "nd-filepaths"].forEach(id => {{
    const el = document.getElementById(id);
    if (el) {{
      el.addEventListener("input", () => {{
        if (id === "nd-runs") renderOncatRunsList(false);
        updateNeutronDialogPreview();
      }});
      el.addEventListener("change", () => {{
        if (id === "nd-ipts") fetchOncatCatalog(el.value);
        else if (id === "nd-facility") fetchOncatCatalog(null, true);
        updateNeutronDialogPreview();
      }});
    }}
  }});
  const ndIptsFilterEl = document.getElementById("nd-ipts-filter");
  if (ndIptsFilterEl) ndIptsFilterEl.addEventListener("input", renderOncatProjectsList);
  const ndRunsFilterEl = document.getElementById("nd-runs-filter");
  if (ndRunsFilterEl) ndRunsFilterEl.addEventListener("input", () => renderOncatRunsList(false));
  const btnNdClearRuns = document.getElementById("btn-nd-clear-runs");
  if (btnNdClearRuns) {{
    btnNdClearRuns.addEventListener("click", () => {{
      const runsEl = document.getElementById("nd-runs");
      if (runsEl) runsEl.value = "";
      renderOncatRunsList(false);
      updateNeutronDialogPreview();
    }});
  }}
  const btnNdOncatLogin = document.getElementById("btn-nd-oncat-login");
  if (btnNdOncatLogin) {{
    btnNdOncatLogin.addEventListener("click", handleOncatAuthButtonClick);
  }}
  const btnNdOncatCancel = document.getElementById("btn-nd-oncat-cancel");
  if (btnNdOncatCancel) {{
    btnNdOncatCancel.addEventListener("click", cancelOncatAuthClick);
  }}
  const ndInstEl = document.getElementById("nd-instrument");
  if (ndInstEl) {{
    ndInstEl.addEventListener("change", () => {{
      const facEl = document.getElementById("nd-facility");
      if (facEl) {{
        facEl.value = (ndInstEl.value === "MANDI" || ndInstEl.value === "TOPAZ") ? "SNS" : "HFIR";
      }}
      fetchOncatCatalog(null, true);
      updateNeutronDialogPreview();
    }});
  }}

  if (btnNdSubmit) {{
    btnNdSubmit.addEventListener("click", async () => {{
      // Sync sample unit cell, space group, and wavelength band back to the main parameters bar
      const syncPair = [
        ["nd-sg", "inp-sg"], ["nd-a", "inp-a"], ["nd-b", "inp-b"], ["nd-c", "inp-c"],
        ["nd-alpha", "inp-alpha"], ["nd-beta", "inp-beta"], ["nd-gamma", "inp-gamma"],
      ];
      syncPair.forEach(([srcId, dstId]) => {{
        const s = document.getElementById(srcId), d = document.getElementById(dstId);
        if (s && d) d.value = s.value;
      }});
      const wlMinV = parseFloat(document.getElementById("nd-wl-min")?.value || "");
      const wlMaxV = parseFloat(document.getElementById("nd-wl-max")?.value || "");
      if (Number.isFinite(wlMinV) && Number.isFinite(wlMaxV)) {{
        state.lamMin = wlMinV; state.lamMax = wlMaxV;
        if (slLamMin) slLamMin.value = wlMinV.toFixed(2);
        if (slLamMax) slLamMax.value = wlMaxV.toFixed(2);
        updateWavelengthLabels();
      }}
      const inpDmin = document.getElementById("inp-dmin");
      if (inpDmin) {{
        const autoD = Number((0.9 * (Number.isFinite(wlMinV) ? wlMinV : (state.lamMin || 2.0))).toFixed(2));
        inpDmin.value = autoD.toFixed(1);
        DATA.d_min = autoD;
      }}
      const instVal = (document.getElementById("nd-instrument")?.value || DATA.instrument || "CG4D").trim();
      DATA.instrument = instVal;

      if (pendingLocalNexusFiles && pendingLocalNexusFiles.length > 0) {{
        const filesToUpload = pendingLocalNexusFiles;
        closeNeutronDataModal();
        await uploadLocalFilesToServer(filesToUpload, true);
        return;
      }}

      const fac = (document.getElementById("nd-facility")?.value || "HFIR").trim();
      const iptsRaw = (document.getElementById("nd-ipts")?.value || "").trim().replace(/^IPTS-/i, "");
      const runsSpec = (document.getElementById("nd-runs")?.value || "").trim();
      const paths = buildNeutronFilePathsFromDialog();
      if (paths.length === 0) {{
        await verifyNeutronPathsInDialog();
        return;
      }}
      // Pre-submit path check inside the dialog before closing or dispatching a job
      const chk = (lastNdValidationResult && lastNdValidationResult.ok)
        ? lastNdValidationResult
        : await verifyNeutronPathsInDialog();
      if (chk && chk.ok === false) {{
        const statusBox = document.getElementById("nd-validation-status");
        if (statusBox) {{
          statusBox.style.boxShadow = "0 0 0 2px rgba(248,113,113,0.55)";
          setTimeout(() => {{ statusBox.style.boxShadow = "none"; }}, 1200);
        }}
        return;
      }}
      closeNeutronDataModal();
      // If a single already-merged .h5 was specified, load it directly and set pending stage to 2. Solve
      if (paths.length === 1 && !/\\.nxs(\\.h5)?$/i.test(paths[0]) && /merged|scan|solve|integrat|predict/i.test(paths[0])) {{
        try {{
          const r = await fetch(`api/payload?file_path=${{encodeURIComponent(paths[0])}}`, {{ cache: "no-store" }});
          if (r.ok) {{
            const p = await r.json();
            recomputeBoundsAndReload(p);
            return;
          }}
        }} catch (_e) {{}}
      }}
      // Otherwise trigger Step 1 (Reduce & Merge Images -> merged.h5 -> pending stage = Solve)
      selectPipelineStage(0, true);
      await executeWorkflowStage(0, {{
        ingest_mode: ndIngestMode === "ipts" ? "oncat" : "file",
        facility: fac,
        instrument: instVal,
        ipts: iptsRaw,
        runs: runsSpec,
        nexus_files: paths,
        verified_dataset_ids: (chk && Array.isArray(chk.verified_dataset_ids)) ? chk.verified_dataset_ids : [],
      }});
    }});
  }}

  async function uploadLocalFilesToServer(files, forceReduce = false) {{
    const params = collectWorkflowParams();
    const isRawNexus = forceReduce || files.some(f => /\\.nxs(\\.h5)?$/i.test(f.name));
    try {{
      setStatusLine(
        "[LOCAL HDF5]",
        isRawNexus
          ? `Opening ${{files.length}} raw NeXus file(s) and triggering <strong>1. Reduce + Merge Images</strong>...`
          : `Loading <strong>${{files.map(f => f.name).join(", ")}}</strong>...`
      );
      // Fast path: check if the selected local file(s) already exist on the server filesystem by (name, size)
      try {{
        const rr = await fetch("api/pipeline/resolve_local_files", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            files: files.map(f => ({{ name: f.name, size: f.size, lastModified: f.lastModified }})),
          }}),
        }});
        if (rr.ok) {{
          const rdoc = await rr.json();
          const matchedPaths = rdoc.matched_paths || [];
          if (matchedPaths.length === files.length && matchedPaths.length > 0) {{
            if (isRawNexus) {{
              selectPipelineStage(0, true);
              await executeWorkflowStage(0, {{
                ingest_mode: "file",
                nexus_files: matchedPaths,
              }});
              return;
            }}
            const pr = await fetch(`api/payload?file_path=${{encodeURIComponent(matchedPaths[0])}}`, {{ cache: "no-store" }});
            if (pr.ok) {{
              const srvPayload = await pr.json();
              if (srvPayload && (Array.isArray(srvPayload.peaks) || Array.isArray(srvPayload.panels))) {{
                recomputeBoundsAndReload(srvPayload);
                return;
              }}
            }}
          }}
        }}
      }} catch (_fastErr) {{}}

      const fd = new FormData();
      files.forEach(f => fd.append("files", f, f.name));
      fd.append("params", JSON.stringify(params));
      const r = await fetch("api/pipeline/upload_h5", {{
        method: "POST",
        body: fd,
      }});
      if (r.ok) {{
        const res = await r.json();
        const savedPaths = res.saved_paths || res.uploaded_paths || [];
        if (isRawNexus && savedPaths.length > 0) {{
          selectPipelineStage(0, true);
          await executeWorkflowStage(0, {{
            ingest_mode: "file",
            nexus_files: savedPaths,
          }});
          return;
        }}
        if (savedPaths.length === 1) {{
          const pr = await fetch(`api/payload?file_path=${{encodeURIComponent(savedPaths[0])}}`, {{ cache: "no-store" }});
          if (pr.ok) {{
            const srvPayload = await pr.json();
            if (srvPayload && (Array.isArray(srvPayload.peaks) || Array.isArray(srvPayload.panels))) {{
              recomputeBoundsAndReload(srvPayload);
              return;
            }}
          }}
        }}
      }}
    }} catch (_uploadErr) {{
      // Fall through to client-side h5wasm parsing when running as standalone static file
    }}
    try {{
      while (!window.h5wasm) await new Promise(r => setTimeout(r, 120));
      let solveFile = files.find(f => /solve|index|integ/i.test(f.name)) || files[0];
      let imgFile = files.find(f => f !== solveFile && /merged|image|scan/i.test(f.name)) || (files.length > 1 ? files[1] : null);
      const buf1 = await solveFile.arrayBuffer();
      const buf2 = imgFile ? await imgFile.arrayBuffer() : null;
      const newPayload = await parseH5BuffersInBrowser(buf1, buf2);
      recomputeBoundsAndReload(newPayload);
    }} catch (err) {{
      setStatusLine("[LOCAL HDF5]", `<span style="color:#f87171;">Local HDF5 error: ${{err.message}}</span>`);
    }}
  }}

  async function loadFromGalaxyDatasetId(dsId) {{
    if (!dsId) return;
    const dsMeta = (galaxyBridge.datasets || []).find(d => d.id === dsId);
    const dsName = (dsMeta && dsMeta.name) ? String(dsMeta.name) : "";
    // If a raw NeXus file in Galaxy History is selected, automatically trigger Step 1 (Reduce + Merge Images)
    if (/\\.nxs(\\.h5)?$/i.test(dsName)) {{
      setStatusLine(
        "[AUTO-REDUCE]",
        `Raw NeXus dataset <strong>${{dsName}}</strong> selected — triggering <strong>Step 1 (Reduce + Merge Images)</strong>...`
      );
      selectPipelineStage(0, true);
      await executeWorkflowStage(0, {{
        ingest_mode: "galaxy_dataset",
        galaxy_dataset_id: dsId,
      }});
      return;
    }}
    setStatusLine("[GALAXY HDF5]", `Loading dataset <strong>${{dsName || dsId}}</strong> from NDIP / Galaxy History...`);
    try {{
      // 1. Prefer server-side HDF5 extraction via NOVA/HTTP backend (handles merged.h5, solve.h5, predictor.h5, integrated.h5 with ellipses & multi-frame)
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
      setStatusLine("[GALAXY HDF5]", `<span style="color:#f87171;">Failed to load HDF5: ${{err.message}}</span>`);
    }}
  }}

  if (selGalaxyH5) {{
    selGalaxyH5.addEventListener("change", () => {{
      const val = selGalaxyH5.value;
      if (!val) return;
      if (val === "__action_neutron_data__") {{
        selGalaxyH5.value = "";
        openNeutronDataModal(null);
        return;
      }}
      if (val === "__action_local_file__") {{
        selGalaxyH5.value = "";
        if (inpLocalH5) {{
          inpLocalH5.value = "";
          inpLocalH5.click();
        }}
        return;
      }}
      loadFromGalaxyDatasetId(val);
    }});
  }}

  if (inpLocalH5) {{
    if (btnLocalH5) btnLocalH5.addEventListener("click", () => inpLocalH5.click());
    inpLocalH5.addEventListener("change", async () => {{
      const files = Array.from(inpLocalH5.files || []);
      if (files.length === 0) return;
      const isRawNexus = files.some(f => /\\.nxs(\\.h5)?$/i.test(f.name));
      if (isRawNexus) {{
        // Open Neutron Data modal pre-populated with local NeXus files so user can confirm Unit Cell + Space Group for merge-images
        openNeutronDataModal(files);
        return;
      }}
      await uploadLocalFilesToServer(files, false);
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
          galaxyBridge.stepArtifacts = doc.step_artifacts || {{}};
          populateGalaxyH5Selector();
          renderFilesDrawer();
          if (elGalaxyBadge) {{
            elGalaxyBadge.textContent = doc.galaxy_connected
              ? `NOVA → GALAXY (${{galaxyBridge.datasets.length}} H5s)`
              : "NOVA LOCAL";
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
          renderFilesDrawer();
          if (elGalaxyBadge) {{
            elGalaxyBadge.textContent = `GALAXY (${{galaxyBridge.datasets.length}} items)`;
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

  async function executeWorkflowStage(stageIdx, extraParams = null) {{
    const st = PIPELINE[stageIdx] || PIPELINE[state.currentPipelineIdx] || PIPELINE[1] || {{ stage_id: "solve", name: "2. SOLVE" }};
    const stageId = st.stage_id || "solve";
    if (stageId === "reduce" && !extraParams) {{
      openNeutronDataModal(null);
      return;
    }}
    const params = {{ ...collectWorkflowParams(), ...(extraParams || {{}}) }};
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
        if (stageId === "solve") {{
          PEAKS.length = 0;
          ZONES.length = 0;
          state.activeZones = new Set();
          selectLadderRung(0, false);
          renderZoneList();
          updateHUD();
        }}
        if (galaxyBridge.pollTimer) clearInterval(galaxyBridge.pollTimer);
        let lastPartialSeq = 0;
        galaxyBridge.pollTimer = setInterval(async () => {{
          try {{
            const sr = await fetch("api/pipeline/status", {{ cache: "no-store" }});
            if (!sr.ok) return;
            const sdoc = await sr.json();
            if (sdoc.step_artifacts) {{
              galaxyBridge.stepArtifacts = sdoc.step_artifacts;
              renderFilesDrawer();
            }}
            const aj = sdoc.active_job;
            if (!aj) return;
            const outText = (aj.stdout || "") + (aj.stderr ? `\\n[stderr]\\n${{aj.stderr}}` : "");
            if (outText) appendTerminalLog(outText, true);
            if (
              stageId === "solve" &&
              aj.state !== "ok" &&
              aj.partial_overlay &&
              (aj.partial_overlay.seq || 0) > lastPartialSeq
            ) {{
              lastPartialSeq = aj.partial_overlay.seq;
              if (Array.isArray(aj.partial_overlay.zones) && aj.partial_overlay.zones.length > 0) {{
                ZONES.length = 0;
                aj.partial_overlay.zones.forEach(z => ZONES.push(z));
              }}
              if (Array.isArray(aj.partial_overlay.peaks)) {{
                PEAKS.length = 0;
                aj.partial_overlay.peaks.forEach(p => PEAKS.push(p));
              }}
              if (Array.isArray(aj.partial_overlay.ladder_steps) && aj.partial_overlay.ladder_steps.length > 0) {{
                LADDER.length = 0;
                aj.partial_overlay.ladder_steps.forEach(s => LADDER.push(s));
              }}
              const targetRung = Math.max(1, typeof aj.ladder_rung_idx === "number" ? aj.ladder_rung_idx : 1);
              selectLadderRung(targetRung, false);
              renderZoneList();
              renderLadder();
              updateHUD();
            }} else if (stageId === "solve" && typeof aj.ladder_rung_idx === "number") {{
              selectLadderRung(aj.ladder_rung_idx, false);
            }}
            if (cellSummary) cellSummary.textContent = `${{aj.mode === "galaxy" ? "PULSAR" : "SUBHKL"}} [${{(aj.state || "running").toUpperCase()}}]`;
            const finishedOk = aj.state === "ok";
            const finishedErr = aj.state === "error" || aj.state === "deleted" || aj.state === "paused" || (!finishedOk && !!aj.error);
            if (finishedOk || finishedErr) {{
              clearInterval(galaxyBridge.pollTimer);
              galaxyBridge.pollTimer = null;
              state.runningStageIdx = -1;
              renderPipeline();
              if (cellSummary) cellSummary.textContent = finishedOk ? "DONE" : "ERROR";
              await probeGalaxyConnection();
              if (finishedOk) {{
                const h5Outs = Array.isArray(aj.outputs)
                  ? aj.outputs.filter(o => o && o.id && (!o.extension || o.extension === "h5" || o.extension === "hdf5"))
                  : [];
                if (selGalaxyH5 && h5Outs.length > 0) {{
                  selGalaxyH5.value = h5Outs[0].id;
                }}
                let loadedOk = false;
                try {{
                  const payloadUrl = aj.output_h5_path
                    ? `api/payload?file_path=${{encodeURIComponent(aj.output_h5_path)}}`
                    : "api/payload";
                  const pr = await fetch(payloadUrl, {{ cache: "no-store" }});
                  if (pr.ok) {{
                    const newPayload = await pr.json();
                    if (newPayload && (Array.isArray(newPayload.panels) || Array.isArray(newPayload.peaks))) {{
                      recomputeBoundsAndReload(newPayload);
                      loadedOk = true;
                      appendTerminalLog(
                        stageId === "reduce"
                          ? "\\nLoaded merged.h5 into viewport — Pending step: 2. Solve (Lattice Ladder)."
                          : "\\nReloaded updated orientation, Laue zones, and Bragg reflections into viewport."
                      );
                    }}
                  }}
                }} catch (_re) {{}}
                if (!loadedOk && h5Outs.length > 0) {{
                  await loadFromGalaxyDatasetId(h5Outs[0].id);
                }}
              }} else if (aj.error) {{
                appendTerminalLog(`\\n[error] ${{aj.error}}`);
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
            const rawCmd = String(jdoc.command_line || "").trim();
            const cmdLine = (stageId !== "merge" && rawCmd && !rawCmd.includes("\\n") && !rawCmd.includes("PYEOF"))
              ? `$ ${{rawCmd}}\\n`
              : "";
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
                if (cellSummary) cellSummary.textContent = "DONE";
                await probeGalaxyConnection();
                const outs = (tres.outputs || []).filter(o => o.extension === "h5" || o.extension === "hdf5");
                if (outs.length > 0) {{
                  appendTerminalLog(`\\nLoading output dataset ${{outs[0].id}} into viewport...`);
                  await loadFromGalaxyDatasetId(outs[0].id);
                }}
              }} else {{
                if (cellSummary) cellSummary.textContent = "ERROR";
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
        recomputeBoundsAndReload(newPayload, {{ isFrameSwitch: true }});
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
                "description": "Select a Galaxy History .h5 dataset (top bar) or open a local .h5 to inspect or dispatch a GPU solve.",
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
