"""Finder-free zone overlays for frame-addressed solve results."""

from pathlib import Path

import h5py
import numpy as np

from subhkl.calibrate import geometry_detectors
from subhkl.instrument.detector import Detector
from subhkl.solve_frames import frame_kinematics, setting_groups
from subhkl.viz.detector_assembly import plot_unrolled_detector
from subhkl.viz.zones import zone_curve_points


def plot_solve(
    filename,
    instrument=None,
    output_dir=None,
    max_index=1,
    dpi=600,
    image_index=None,
    images_filename=None,
    zone_alpha=0.5,
):
    with h5py.File(filename) as f:
        status = f["solve"].attrs.get("status", "unknown")
        if isinstance(status, bytes):
            status = status.decode()
        diagnostic = status if "sample/U" not in f else None
        instrument = instrument or f.attrs.get("instrument")
        if isinstance(instrument, bytes):
            instrument = instrument.decode()
        if instrument is None:
            raise ValueError("solve output has no instrument")
        images_filename = images_filename or f.attrs.get("source")
        if not images_filename or not Path(images_filename).is_file():
            raise ValueError("solve visualization requires raw counts via --images")
        with h5py.File(images_filename) as raw:
            banks = raw["bank_ids"][()].astype(int)
            if not np.array_equal(banks, f["bank_ids"][()]):
                raise ValueError(
                    "visualizer images must match the solve frame/bank layout"
                )
            if raw["images"].shape[0] != len(banks):
                raise ValueError("image stack and bank ids have different lengths")
            offsets = raw["file_offsets"][()] if "file_offsets" in raw else None
            files = raw["files"][()] if "files" in raw else None
        R, origins, runs, _, _ = frame_kinematics(f, len(banks), offsets)
        if "solve/sample_origin_lab" in f:
            origins = f["solve/sample_origin_lab"][()]
        groups = setting_groups(banks, R, origins, runs)
        B = f["sample/B"][()]
        U = f["sample/U"][()][None] if "sample/U" in f else None
        if "solve/orientations_sample" in f and "solve/active" in f:
            candidates = f["solve/orientations_sample"][()]
            active = f["solve/active"][()].astype(int)
            U = candidates[active] if len(active) else candidates
        if U is None or len(U) == 0:
            raise ValueError(
                "solve output contains no orientation candidates to visualize"
            )
        dets = geometry_detectors(
            str(instrument), np.zeros(7), list(dict.fromkeys(banks))
        )
        for bank, det in list(dets.items()):
            name = f"detector_calibration/bank_{bank}"
            if name in f:
                cfg = dict(det.config)
                for key in ("center", "uhat", "vhat", "width", "height"):
                    cfg[key] = f[name + "/" + key][()]
                dets[bank] = Detector(cfg)
    if image_index is not None and not 0 <= image_index < len(banks):
        raise ValueError("image index outside solve frame range")
    output = Path(output_dir or Path(filename).resolve().parent)
    output.mkdir(parents=True, exist_ok=True)
    written = []

    class Shim:
        sample_offset = np.zeros(3)
        R = None
        diagnostic_status = diagnostic

    for setting, frames in enumerate(groups):
        if image_index is not None:
            frames = frames[frames == image_index]
            if not len(frames):
                continue
        first = frames[0]
        detectors = {int(i): dets[banks[i]] for i in frames}
        curves = []
        for component, u in enumerate(U):
            component_curves = zone_curve_points(
                detectors,
                u,
                B,
                R_gonio=R[first],
                max_index=max_index,
                sample_origin=origins[first],
            )
            if len(U) > 1:
                for curve in component_curves:
                    curve["label"] = f"{component + 1}: {curve['label']}"
            curves.extend(component_curves)
        run = int(runs[first])
        if files is not None and run < len(files):
            name = (
                files[run].decode()
                if isinstance(files[run], bytes)
                else str(files[run])
            )
            label = Path(name).stem
        else:
            label = f"{Path(filename).stem}-run{run}"
        if sum(runs[g[0]] == run for g in groups) > 1:
            label += f"-setting{setting}"
        if image_index is not None:
            label += f"-frame{image_index}"
        if diagnostic:
            label += f"-diagnostic-{diagnostic}"
        target = output / f"{label}-index.png"
        with h5py.File(images_filename) as raw:
            images = {int(i): raw["images"][i] for i in frames}
        plot_unrolled_detector(
            Shim(),
            images,
            detectors,
            out_name=str(target),
            instrument=str(instrument),
            dpi=dpi,
            zone_curves=curves,
            zone_alpha=zone_alpha,
        )
        written.append(str(target))
        print(f"wrote {target}")
    return written
