import re

import h5py
import numpy as np

from subhkl.config import beamlines, reduction_settings
from subhkl.integration.image_data import ImageData
from subhkl.io.events import sort_bank_events


class ImageLoader:
    def load_nexus(filename: str, instrument, load_events: bool = False) -> ImageData:
        """Detector images of a NeXus event file, one per bank.

        With ``load_events`` the event times of flight are kept as well,
        sorted by image pixel (see ``subhkl.io.events``).
        """
        detectors = beamlines[instrument]
        settings = reduction_settings[instrument]
        y_fast = bool(settings.get("YAxisIsFastVaryingIndex"))
        ims = {}
        events = {} if load_events else None
        with h5py.File(filename, "r") as f:
            keys = []
            banks = []
            for key in f["/entry/"]:
                match = re.match(r"bank(\d+).*", key)
                if match is not None:
                    keys.append(key)
                    banks.append(int(match.groups()[0]))
            for rel_key, bank in zip(keys, banks, strict=False):
                key = "/entry/" + rel_key + "/event_id"
                array = f[key][()]
                det = detectors.get(str(bank))
                if det is not None:
                    m, n, offset = det["m"], det["n"], det["offset"]
                    tof_key = "/entry/" + rel_key + "/event_time_offset"
                    if load_events and tof_key in f:
                        image, tof = sort_bank_events(
                            array.astype(np.int64) - offset,
                            f[tof_key][()],
                            m,
                            n,
                            y_fast,
                        )
                        if image.sum() > 0:
                            ims[bank] = image
                            events[bank] = tof
                        continue
                    bc = np.bincount(array - offset, minlength=m * n)
                    if np.sum(bc) > 0:
                        if y_fast:
                            ims[bank] = bc.reshape(m, n).T
                        else:
                            ims[bank] = bc.reshape(n, m)

        if events is not None and set(events) != set(ims):
            # A bank without event times cannot be gated; keep images only.
            events = None
        return ImageData(ims=ims, events=events)

    def load_merged_h5(filename: str) -> ImageData:
        ims = {}
        bank_mapping = {}
        with h5py.File(filename, "r") as f:
            images = f["images"]
            N = images.shape[0]
            if "bank_ids" in f:
                bank_ids = f["bank_ids"][()]
            else:
                bank_ids = np.zeros(N, dtype=int)
            if "files" in f and "file_offsets" in f:
                image_files_raw = [
                    n.decode("utf-8") if isinstance(n, bytes) else str(n)
                    for n in f["files"][()]
                ]
                file_offsets = f["file_offsets"][()]
            else:
                image_files_raw = None
                file_offsets = None
            data = images[()]
            for i in range(N):
                ims[i] = data[i]
                bank_mapping[i] = int(bank_ids[i])

        return ImageData(
            ims=ims,
            raw_files=image_files_raw,
            file_offsets=file_offsets,
            bank_mapping=bank_mapping,
        )
