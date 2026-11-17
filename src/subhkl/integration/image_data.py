import bisect
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ImageData:
    ims: dict[int, np.ndarray]
    file_offsets: np.ndarray | None = None
    raw_files: list[str] | None = None
    bank_mapping: dict[int, int] = field(default_factory=dict)

    def get_label(self, img_key: int) -> str:
        files = self.raw_files
        offsets = self.file_offsets
        if files and offsets is not None:
            file_idx = bisect.bisect_right(self.file_offsets, img_key) - 1
            if 0 <= file_idx < len(self.raw_files):
                orig_name = os.path.basename(self.raw_files[file_idx])
                clean_name = os.path.splitext(orig_name)[0]
                clean_name = clean_name.replace(".nxs.h5", "").replace(".h5", "")
                return clean_name
        return f"img{img_key}"

    def get_run_id(self, img_key: int) -> int:
        if self.file_offsets is None:
            return 0
        return int(np.searchsorted(self.file_offsets, img_key, side="right") - 1)
