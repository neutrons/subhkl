import h5py
import numpy as np
import re
import multiprocessing
import concurrent.futures

from subhkl.instrument.detector import Detector
from subhkl.config import beamlines, reduction_settings
from subhkl.instrument.goniometer import sample_to_lab, lab_to_sample

def _process_single_bank(args):
    """Parallel worker to parse and project a single detector bank."""
    nexus_filename, key, instrument_name, ki_vec, gonio_axes, gonio_continuous_logs, gonio_translations, gonio_offsets = args

    with h5py.File(nexus_filename, 'r') as f:
        match = re.match(r"bank(\d+)_events", key)
        if not match: return None
        bank_id = int(match.group(1))
        bank_str = str(bank_id)

        folder = f'/entry/{key}'
        if folder+'/event_id' not in f: return None

        event_id = f[folder+'/event_id'][:]
        event_index = f[folder+'/event_index'][:]
        event_time_offset = f[folder+'/event_time_offset'][:]
        event_time_zero = f[folder+'/event_time_zero'][:]

    if len(event_id) == 0: return None

    counts_per_pulse = np.diff(np.append(event_index, len(event_time_offset))).astype(int)
    absolute_time = np.repeat(event_time_zero, counts_per_pulse) + (event_time_offset * 1e-6)

    det_config = beamlines[instrument_name][bank_str]
    det = Detector(det_config)
    settings = reduction_settings.get(instrument_name, {})

    offset = det_config.get("offset", 0)
    local_id = event_id - offset

    if settings.get("YAxisIsFastVaryingIndex"):
        pixel_c = local_id // det.n
        pixel_r = local_id % det.n
    else:
        pixel_c = local_id % det.m
        pixel_r = local_id // det.m

    xyz = det.pixel_to_lab(pixel_r, pixel_c)
    num_events = len(absolute_time)
    num_axes = len(gonio_axes) if gonio_axes is not None else 1

    if gonio_axes is not None and gonio_continuous_logs is not None:
        interpolated_angles = np.zeros((num_axes, num_events), dtype=np.float32)
        
        for i in range(num_axes):
            g_times, g_vals = gonio_continuous_logs[i]
            if len(g_times) <= 1:
                interpolated_angles[i, :] = g_vals[0] if len(g_vals) > 0 else 0.0
            else:
                interpolated_angles[i, :] = np.interp(absolute_time, g_times, g_vals)

        s_lab_dynamic = sample_to_lab(
            np.zeros((num_events, 3)), 
            gonio_axes, 
            interpolated_angles, 
            gonio_translations, 
            zero_offsets=gonio_offsets
        )
        kf_lab = xyz - s_lab_dynamic
    else:
        interpolated_angles = np.zeros((num_axes, num_events), dtype=np.float32)
        s_lab_static = gonio_translations[-1][:3] if gonio_translations is not None else np.zeros(3)
        s_lab_dynamic = np.tile(s_lab_static, (num_events, 1)).astype(np.float32)
        kf_lab = xyz - s_lab_dynamic

    kf_norm = np.sqrt(np.sum(kf_lab**2, axis=1, keepdims=True))
    kf_lab /= np.where(kf_norm == 0, 1.0, kf_norm)
    q_lab = kf_lab - ki_vec[None, :]

    if gonio_axes is not None:
        # Pass is_vector=True so translations are ignored for directional momentum rays!
        q_sample = lab_to_sample(
            q_lab, gonio_axes, interpolated_angles, gonio_translations, gonio_offsets, is_vector=True
        )
        ki_sample = lab_to_sample(
            np.tile(ki_vec, (num_events, 1)), gonio_axes, interpolated_angles, gonio_translations, gonio_offsets, is_vector=True
        )
    else:
        q_sample = q_lab
        ki_sample = np.tile(ki_vec, (num_events, 1))

    banks = np.full(num_events, bank_id, dtype=np.int16)

    return (
        q_sample.astype(np.float32), 
        absolute_time,
        banks,
        pixel_r.astype(np.int16),
        pixel_c.astype(np.int16),
        interpolated_angles.T.astype(np.float32),
        s_lab_dynamic,
        ki_sample.astype(np.float32) 
    )

class EventStreamLoader:
    def __init__(
        self,
        event_nexus_filename: str,
        instrument_name: str,
        ki_vec: np.ndarray,
        sample_offset: np.ndarray,
        gonio_axes=None,
        gonio_names=None,
        gonio_offsets=None,
    ):
        self.event_nexus_filename = event_nexus_filename
        self.instrument_name = instrument_name
        self.ki_vec = ki_vec
        self.sample_offset = sample_offset
        self.gonio_axes = gonio_axes
        self.gonio_names = gonio_names
        self.gonio_offsets = gonio_offsets
        
        print(f"  > Initializing Event Stream Loader from: {event_nexus_filename}")
        self._load_and_sort_events()

    def _load_and_sort_events(self):
        gonio_continuous_logs = []
        with h5py.File(self.event_nexus_filename, 'r') as f:
            keys = [k for k in f['entry'].keys() if k.endswith('_events')]
            
            if self.gonio_names is not None and self.gonio_axes is not None:
                for name in self.gonio_names:
                    try:
                        log_path = f'entry/DASlogs/{name}'
                        if log_path in f and 'time' in f[log_path] and 'value' in f[log_path]:
                            t = f[f'{log_path}/time'][:]
                            v = f[f'{log_path}/value'][:]
                            gonio_continuous_logs.append((t, v))
                        else:
                            gonio_continuous_logs.append((np.array([0.0]), np.array([0.0])))
                    except Exception as e:
                        gonio_continuous_logs.append((np.array([0.0]), np.array([0.0])))
            else:
                gonio_continuous_logs = None

        args_list = [
            (
                self.event_nexus_filename, 
                k, 
                self.instrument_name, 
                self.ki_vec, 
                self.gonio_axes, 
                gonio_continuous_logs, 
                self.sample_offset,
                self.gonio_offsets
            )
            for k in keys
        ]

        all_q_lab, all_times, all_banks, all_pixels_r, all_pixels_c = [], [], [], [], []
        all_angles, all_slab, all_ki_sample = [], [], []
        
        print(f"  > Extracting and projecting {len(keys)} detector banks via Multiprocessing...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for result in executor.map(_process_single_bank, args_list):
                if result is not None:
                    q, t, b, pr, pc, ang, slab, ki_s = result
                    all_q_lab.append(q)
                    all_times.append(t)
                    all_banks.append(b)
                    all_pixels_r.append(pr)
                    all_pixels_c.append(pc)
                    all_angles.append(ang)
                    all_slab.append(slab)
                    all_ki_sample.append(ki_s)

        if not all_q_lab:
            self.total_events = 0
            return

        all_q_lab = np.vstack(all_q_lab)
        all_times = np.concatenate(all_times)
        all_banks = np.concatenate(all_banks)
        all_pixels_r = np.concatenate(all_pixels_r)
        all_pixels_c = np.concatenate(all_pixels_c)
        all_angles = np.vstack(all_angles)
        all_slab = np.vstack(all_slab)
        all_ki_sample = np.vstack(all_ki_sample)

        print("  > Performing global chronological sort...")
        
        # 1. Use 'stable' (Timsort) instead of the default quicksort. 
        # Since 'all_times' is a concatenation of mostly sorted sub-arrays, this is much faster.
        sort_idx = np.argsort(all_times, kind='stable')
        
        # 2. Parallelize the memory-intensive reordering step.
        # NumPy releases the GIL for array indexing, so ThreadPoolExecutor works perfectly.
        def apply_sort(arr):
            return arr[sort_idx]

        arrays_to_sort = [
            all_q_lab, all_times, all_banks, all_pixels_r, 
            all_pixels_c, all_angles, all_slab, all_ki_sample
        ]

        print("  > Applying sorted indices to arrays...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(arrays_to_sort)) as executor:
            (
                self.all_q_lab, 
                self.all_times, 
                self.all_banks, 
                self.all_pixels_r, 
                self.all_pixels_c, 
                self.all_angles, 
                self.all_slab, 
                self.all_ki_sample
            ) = list(executor.map(apply_sort, arrays_to_sort))

        self.total_events = len(self.all_q_lab)
        print(f"  > EventStreamLoader Ready. Cached {self.total_events:,} events.")

    # Add min_batch_size parameter
    def get_batches(self, batch_size_events: int = 10000, min_batch_size: int = 1):
        if self.total_events == 0:
            return

        for start_idx in range(0, self.total_events, batch_size_events):
            end_idx = min(start_idx + batch_size_events, self.total_events)
            
            # Use the parameter instead of the hardcoded 100
            if end_idx - start_idx < min_batch_size:
                break
                
            yield (
                self.all_q_lab[start_idx:end_idx],
                self.all_times[start_idx:end_idx],
                self.all_banks[start_idx:end_idx],
                self.all_pixels_r[start_idx:end_idx],
                self.all_pixels_c[start_idx:end_idx],
                self.all_angles[start_idx:end_idx],
                self.all_slab[start_idx:end_idx],
                self.all_ki_sample[start_idx:end_idx],
                end_idx
            )
