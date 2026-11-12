"""Convert an ISAW ``.DetCal`` detector calibration file to ``beamlines.json`` format.

This script reproduces the flat-panel geometry produced by Mantid's
``LoadIsawDetCal`` + ``PreprocessDetectorsToMD`` directly in pure Python/NumPy:

- ``CenterX, CenterY, CenterZ`` (cm) is the panel midpoint :math:`\\mathbf{m}`.
- ``Up = (UpX, UpY, UpZ)`` defines :math:`\\hat{\\mathbf{v}}` (row / Y axis,
  lower-left to upper-left).
- ``Base = (BaseX, BaseY, BaseZ)`` and :math:`\\hat{\\mathbf{v}}` define the panel
  normal :math:`\\hat{\\mathbf{n}} = (\\mathbf{b} \\times \\hat{\\mathbf{v}}) / \\|\\mathbf{b} \\times \\hat{\\mathbf{v}}\\|`
  and :math:`\\hat{\\mathbf{u}} = \\hat{\\mathbf{v}} \\times \\hat{\\mathbf{n}}` (column / X axis,
  lower-left to lower-right), matching Mantid's ``LoadIsawDetCal`` orthonormalization.
- Pixel ``(0, 0)`` origin is
  :math:`\\mathbf{c} = \\mathbf{m} - \\tfrac{1}{2} w \\hat{\\mathbf{u}} - \\tfrac{1}{2} h \\hat{\\mathbf{v}}`
  where :math:`w = (\\text{WIDTH}/100)\\frac{N_{\\text{cols}}-1}{N_{\\text{cols}}}` and
  :math:`h = (\\text{HEIGHT}/100)\\frac{N_{\\text{rows}}-1}{N_{\\text{rows}}}`.

Example
-------
Update the ``TOPAZ`` entry in ``src/subhkl/resources/beamlines.json``::

    uv run python scripts/detcal_to_beamlines.py \\
        --detcal /SNS/TOPAZ/shared/calibration/2026B_AG/calibration.DetCal \\
        --instrument TOPAZ \\
        --nexus /SNS/TOPAZ/IPTS-37250/nexus/TOPAZ_55023.nxs.h5 \\
        --update-beamlines
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEAMLINES_JSON = REPO_ROOT / "src" / "subhkl" / "resources" / "beamlines.json"


def _load_idf_offsets_from_nexus(nexus_path: str | Path) -> dict[int, int]:
    """Extract ``bank_id -> idstart`` pixel offsets from ``/entry/instrument/instrument_xml``."""
    import h5py

    offsets: dict[int, int] = {}
    with h5py.File(nexus_path, "r") as h5:
        if "entry/instrument/instrument_xml/data" not in h5:
            return offsets
        xml_raw = h5["entry/instrument/instrument_xml/data"][()]
        xml_str = (
            xml_raw[0].decode("utf-8")
            if hasattr(xml_raw, "__len__")
            and isinstance(xml_raw[0], (bytes, np.bytes_))
            else (
                xml_raw.decode("utf-8")
                if isinstance(xml_raw, (bytes, np.bytes_))
                else str(xml_raw)
            )
        )
    root = ET.fromstring(xml_str)
    ns = {"idf": "http://www.mantidproject.org/IDF/1.0"}
    for comp in root.findall("idf:component[@type='panel']", ns):
        idstart_str = comp.attrib.get("idstart")
        loc = comp.find("idf:location", ns)
        if idstart_str is None or loc is None:
            continue
        bname = loc.attrib.get("name", "")
        if bname.startswith("bank"):
            try:
                bid = int(bname[len("bank") :])
                offsets[bid] = int(idstart_str)
            except ValueError:
                pass
    return offsets


def parse_detcal(
    detcal_path: str | Path,
    idf_offsets: dict[int, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse an ISAW ``.DetCal`` file into a ``beamlines.json`` bank dictionary."""
    banks: dict[str, dict[str, Any]] = {}
    with open(detcal_path, encoding="utf-8") as fh:
        for raw_line in fh:
            parts = raw_line.split()
            if not parts or parts[0] != "5":
                continue
            detnum = int(parts[1])
            nrows = int(parts[2])
            ncols = int(parts[3])
            width_cm = float(parts[4])
            height_cm = float(parts[5])
            cx, cy, cz = float(parts[8]), float(parts[9]), float(parts[10])
            bx, by, bz = float(parts[11]), float(parts[12]), float(parts[13])
            ux, uy, uz = float(parts[14]), float(parts[15]), float(parts[16])

            w = float((width_cm / 100.0) * (ncols - 1) / ncols)
            h = float((height_cm / 100.0) * (nrows - 1) / nrows)

            vhat = np.array([ux, uy, uz], dtype=np.float64)
            vhat /= np.linalg.norm(vhat)

            base = np.array([bx, by, bz], dtype=np.float64)
            base /= np.linalg.norm(base)

            normal = np.cross(base, vhat)
            normal /= np.linalg.norm(normal)

            uhat = np.cross(vhat, normal)
            uhat /= np.linalg.norm(uhat)

            mid = np.array([cx, cy, cz], dtype=np.float64) / 100.0
            center = mid - 0.5 * w * uhat - 0.5 * h * vhat

            offset = (
                int(idf_offsets[detnum])
                if idf_offsets and detnum in idf_offsets
                else int(detnum * ncols * nrows)
            )

            banks[str(detnum)] = {
                "panel": "flat",
                "center": center.tolist(),
                "uhat": uhat.tolist(),
                "vhat": vhat.tolist(),
                "width": w,
                "height": h,
                "m": ncols,
                "n": nrows,
                "offset": offset,
            }
    return banks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an ISAW .DetCal file into subhkl beamlines.json format."
    )
    parser.add_argument(
        "--detcal",
        required=True,
        type=Path,
        help="Path to the ISAW .DetCal calibration file.",
    )
    parser.add_argument(
        "--instrument",
        required=True,
        type=str,
        help="Instrument key in beamlines.json (e.g. TOPAZ or MANDI).",
    )
    parser.add_argument(
        "--nexus",
        type=Path,
        default=None,
        help="Optional NeXus .nxs.h5 file to read idstart offsets from /entry/instrument/instrument_xml.",
    )
    parser.add_argument(
        "--beamlines-json",
        type=Path,
        default=DEFAULT_BEAMLINES_JSON,
        help=f"Path to beamlines.json (default: {DEFAULT_BEAMLINES_JSON}).",
    )
    parser.add_argument(
        "--update-beamlines",
        action="store_true",
        help="Update --beamlines-json in place with the parsed instrument banks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional standalone JSON output file for the parsed banks.",
    )
    args = parser.parse_args()

    idf_offsets = _load_idf_offsets_from_nexus(args.nexus) if args.nexus else None
    banks = parse_detcal(args.detcal, idf_offsets=idf_offsets)
    print(
        f"Parsed {len(banks)} detector banks for {args.instrument} from {args.detcal}"
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(banks, fh, indent=4)
        print(f"Wrote {args.output}")

    if args.update_beamlines:
        with open(args.beamlines_json, encoding="utf-8") as fh:
            all_beamlines = json.load(fh)
        all_beamlines[args.instrument] = banks
        with open(args.beamlines_json, "w", encoding="utf-8") as fh:
            json.dump(all_beamlines, fh, indent=4)
        print(f"Updated '{args.instrument}' in {args.beamlines_json}")


if __name__ == "__main__":
    main()
