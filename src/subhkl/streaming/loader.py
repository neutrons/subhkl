import h5py
import numpy as np
import re
import multiprocessing
import concurrent.futures

from subhkl.instrument.detector import Detector
from subhkl.config import beamlines, reduction_settings

def _process_single_bank(args):
    """Parallel worker to parse and project a single detector bank."""
    nexus_filename, key, instrument_name, ki_vec, gonio_axes, gonio_continuous_logs, gonio_translations = args

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

    # Fast Unfold
    counts_per_pulse = np.diff(np.append(event_index, len(event_time_offset))).astype(int)
    absolute_time = np.repeat(event_time_zero, counts_per_pulse) + (event_time_offset * 1e-6)

    # Map Pixels
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

    if gonio_axes is not None and gonio_continuous_logs is not None:
        num_events = len(absolute_time)
        interpolated_angles = np.zeros((len(gonio_axes), num_events))
        
        for i in range(len(gonio_axes)):
            g_times, g_vals = gonio_continuous_logs[i]
            if len(g_times) <= 1:
                interpolated_angles[i, :] = g_vals[0] if len(g_vals) > 0 else 0.0
            else:
                interpolated_angles[i, :] = np.interp(absolute_time, g_times, g_vals)
            
        from scipy.spatial.transform import Rotation
       
        from scipy.spatial.transform import Rotation
        
        s_lab_dynamic = np.zeros((num_events, 3), dtype=np.float32)
        
        # Apply kinematic chain in reverse (Sample -> Base)
        for i in range(len(gonio_axes) - 1, -1, -1):
            axis = gonio_axes[i][:3]
            
            direction_multiplier = gonio_axes[i][3] if len(gonio_axes[i]) > 3 else 1.0
            
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis = axis / axis_norm
                
            # Multiply the angle by the direction flag before converting to radians
            theta_rad = np.radians(interpolated_angles[i, :] * direction_multiplier)
            rotvecs = theta_rad[:, None] * axis[None, :]
            
            # C-optimized massive batch rotation
            R_i = Rotation.from_rotvec(rotvecs)
            s_lab_dynamic = R_i.apply(s_lab_dynamic)
            
            if gonio_translations is not None:
                s_lab_dynamic += gonio_translations[i][:3]

        kf = xyz - s_lab_dynamic
    else:
        s_lab_static = gonio_translations[-1] if gonio_translations is not None else np.zeros(3)
        kf = xyz - s_lab_static[None, :]

    kf_norm = np.sqrt(np.sum(kf**2, axis=1, keepdims=True))
    kf /= np.where(kf_norm == 0, 1.0, kf_norm)

    q_lab = kf - ki_vec[None, :]
    banks = np.full(len(absolute_time), bank_id, dtype=np.int16)

    return (
        q_lab.astype(np.float32),
        absolute_time,
        banks,
        pixel_r.astype(np.int16),
        pixel_c.astype(np.int16)
    )

class EventStreamLoader:
    """
    Stateful event loader. Performs heavy I/O, kinematic projection, and global 
    chronological sorting exactly once during instantiation.
    """
    def __init__(
        self,
        event_nexus_filename: str,
        instrument_name: str,
        ki_vec: np.ndarray,
        sample_offset: np.ndarray,
        gonio_axes=None,
        gonio_names=None, # <-- Changed to accept NAMES instead of discrete angles
    ):
        self.event_nexus_filename = event_nexus_filename
        self.instrument_name = instrument_name
        self.ki_vec = ki_vec
        self.sample_offset = sample_offset
        self.gonio_axes = gonio_axes
        self.gonio_names = gonio_names
        
        print(f"  > Initializing Event Stream Loader from: {event_nexus_filename}")
        self._load_and_sort_events()

    def _load_and_sort_events(self):
        # --- NEW: Extract the true continuous NeXus logs ---
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
                            # Fallback for missing logs
                            gonio_continuous_logs.append((np.array([0.0]), np.array([0.0])))
                    except Exception as e:
                        print(f"Warning: Failed to load continuous log for {name}: {e}")
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
                gonio_continuous_logs, # <-- Pass the full tuples
                self.sample_offset
            )
            for k in keys
        ]

        all_q_lab, all_times, all_banks, all_pixels_r, all_pixels_c = [], [], [], [], []
        
        print(f"  > Extracting and projecting {len(keys)} detector banks via Multiprocessing...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for result in executor.map(_process_single_bank, args_list):
                if result is not None:
                    q, t, b, pr, pc = result
                    all_q_lab.append(q)
                    all_times.append(t)
                    all_banks.append(b)
                    all_pixels_r.append(pr)
                    all_pixels_c.append(pc)

        if not all_q_lab:
            self.total_events = 0
            return

        all_q_lab = np.vstack(all_q_lab)
        all_times = np.concatenate(all_times)
        all_banks = np.concatenate(all_banks)
        all_pixels_r = np.concatenate(all_pixels_r)
        all_pixels_c = np.concatenate(all_pixels_c)

        print("  > Performing global chronological sort...")
        sort_idx = np.argsort(all_times)
        
        self.all_q_lab = all_q_lab[sort_idx]
        self.all_times = all_times[sort_idx]
        self.all_banks = all_banks[sort_idx]
        self.all_pixels_r = all_pixels_r[sort_idx]
        self.all_pixels_c = all_pixels_c[sort_idx]

        self.total_events = len(self.all_q_lab)
        print(f"  > EventStreamLoader Ready. Cached {self.total_events:,} chronologically sorted events.")

    def get_batches(self, batch_size_events: int = 10000):
        """
        Yields lightweight slices of the pre-loaded, pre-sorted memory arrays.
        Can be called indefinitely without triggering disk I/O.
        """
        if self.total_events == 0:
            return

        for start_idx in range(0, self.total_events, batch_size_events):
            end_idx = min(start_idx + batch_size_events, self.total_events)
            if end_idx - start_idx < 100:
                break
                
            yield (
                self.all_q_lab[start_idx:end_idx],
                self.all_times[start_idx:end_idx],
                self.all_banks[start_idx:end_idx],
                self.all_pixels_r[start_idx:end_idx],
                self.all_pixels_c[start_idx:end_idx],
                end_idx
            )
