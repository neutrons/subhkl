import unittest
import tempfile
import os
import h5py
import numpy as np
import itertools
from scipy.spatial.transform import Rotation

from subhkl.commands import run_bingham_tracker

def get_cubic_symmetries():
    """Generates the 24 valid rotation matrices for a Cubic point group."""
    syms = []
    I = np.eye(3)
    for p in itertools.permutations([0, 1, 2]):
        P = I[list(p), :]
        for signs in itertools.product([1, -1], repeat=3):
            S = np.diag(signs)
            M = S @ P
            if np.isclose(np.linalg.det(M), 1.0):
                syms.append(M)
    return syms

class TestBinghamTracker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.finder_file = os.path.join(self.test_dir.name, "mock_finder.h5")
        
        # Ensure deterministic testing environment
        np.random.seed(42)

    def tearDown(self):
        self.test_dir.cleanup()

    def generate_poissonian_events(self, U_true, num_events=1000000, duration=5.0, sigma_q=0.05, bg_fraction=0.0, b_factor=0.0):
        B_mat = np.array([
            [2*np.pi/10.0, 0, 0],
            [0, 2*np.pi/10.0, 0],
            [0, 0, 2*np.pi/10.0]
        ])
        ki_vec = np.array([0.0, 0.0, 1.0])

        h_vals = np.arange(-3, 4)
        hc, kc, lc = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
        hkl = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
        mask = ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))
        hkl = hkl[:, mask]

        q_theo = B_mat @ hkl
        q_norms = np.linalg.norm(q_theo, axis=0)
        q_theo_hat = q_theo / q_norms

        kinematic_proj = ki_vec.T @ (U_true @ q_theo_hat)
        
        # True Laue equation (2.0 instead of 4*pi) to match new tracker physics
        wavelengths = -(2.0 / q_norms) * kinematic_proj

        valid_mask = (wavelengths > 0.5) & (wavelengths < 10.0)
        valid_q_hat = q_theo_hat[:, valid_mask]
        valid_norms = q_norms[valid_mask]
        num_valid = valid_q_hat.shape[1]

        num_bg = int(num_events * bg_fraction)
        num_sig = num_events - num_bg

        # ==========================================================
        # --- THE WILSON PRIOR (Intensity Decay) ---
        # ==========================================================
        if b_factor > 0.0:
            # I(q) ~ exp(-B * |q|^2)
            # This makes low-q peaks overwhelmingly more frequent
            intensities = np.exp(-b_factor * (valid_norms**2))
            p_dist = intensities / np.sum(intensities)
        else:
            p_dist = None # Uniform distribution for old tests

        peak_indices = np.random.choice(num_valid, size=num_sig, p=p_dist)
        
        q_exp_list = []
        # 1. Generate Physical Signal Events
        for idx in peak_indices:
            q_hat_lab = U_true @ valid_q_hat[:, idx]
            angular_std = sigma_q / valid_norms[idx]
            noise_vec = np.random.normal(0, angular_std, 3)
            q_exp = q_hat_lab + noise_vec
            q_exp /= np.linalg.norm(q_exp)
            q_exp_list.append(q_exp)

        # 2. Generate Uniform Background Noise Events
        if num_bg > 0:
            bg_vecs = np.random.normal(0, 1, (num_bg, 3))
            bg_vecs /= np.linalg.norm(bg_vecs, axis=1, keepdims=True)
            q_exp_list.extend(bg_vecs)

        q_lab = np.array(q_exp_list)

        # Shuffle to mix background and signal evenly across time
        shuffle_idx = np.random.permutation(num_events)
        q_lab = q_lab[shuffle_idx]

        times = np.sort(np.random.uniform(0, duration, num_events)) 
        banks = np.ones(num_events, dtype=int)
        pixels_r = np.zeros(num_events, dtype=int)
        pixels_c = np.zeros(num_events, dtype=int)

        return q_lab, times, banks, pixels_r, pixels_c

    def test_wilson_intensity_modulation(self):
        print(f"\n{'='*60}\nExecuting Regression: WILSON INTENSITY MODULATION (Low-Q Preference)\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # b_factor=0.5 causes extreme exponential decay of intensity at high-q
        sim_data = self.generate_poissonian_events(
            U_true, num_events=1000000, duration=5.0, b_factor=0.5
        )
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)

        def streaming_callback(time, U_preds, losses, mean_loss, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,
            sigma_q_min=0.02,
            annealing_rate=1.0,
            gamma_event=1e-4,
            gamma_step=100.0,
            kappa_init=100.0,
            n_ensemble=1,
            streaming_callback=streaming_callback
        )

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Wilson Modulation failed to converge: Final Error {final_err:.2f}° >= 2.0°")

    def get_fake_batches(self, sim_data, batch_size=10000):
        """Yields streaming tuples exactly matching the EventStreamLoader signature."""
        q_lab, times, banks, pixels_r, pixels_c = sim_data
        num_events = len(times)
        
        for start_idx in range(0, num_events, batch_size):
            end_idx = min(start_idx + batch_size, num_events)
            N = end_idx - start_idx
            
            yield (
                q_lab[start_idx:end_idx].astype(np.float32),
                times[start_idx:end_idx].astype(np.float32),
                banks[start_idx:end_idx].astype(np.int16),
                pixels_r[start_idx:end_idx].astype(np.int16),
                pixels_c[start_idx:end_idx].astype(np.int16),
                np.zeros((N, 1), dtype=np.float32),  # angles
                np.zeros((N, 3), dtype=np.float32),  # s_lab
                np.tile([0.0, 0.0, 1.0], (N, 1)).astype(np.float32), # ki_sample
                end_idx # cumulative count
            )

    def _evaluate_cubic_symmetric_error(self, U_true, U_pred):
        min_err_deg = 180.0
        for sym in get_cubic_symmetries():
            U_mate = U_true @ sym
            trace_val = np.clip(np.trace(U_mate.T @ U_pred), -1.0, 3.0)
            err_deg = np.degrees(np.arccos((trace_val - 1.0) / 2.0))
            min_err_deg = min(min_err_deg, err_deg)
        return min_err_deg

    def test_local_capture(self):
        print(f"\n{'='*60}\nExecuting Regression: LOCAL CAPTURE (Seed Err: 5.0°)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)
        
        def streaming_callback(time, U_preds, losses, mean_loss, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,   
            sigma_q_min=0.02,    # Match simulated peak width closer
            annealing_rate=1.0,  
            gamma_event=1e-4,    # Fast learning rate for clean local capture
            gamma_step=100.0,
            kappa_init=100.0,
            n_ensemble=1, 
            streaming_callback=streaming_callback
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Local Capture failed to converge: Final Error {final_err:.2f}° >= 2.0°")

    def test_global_aliasing(self):
        print(f"\n{'='*60}\nExecuting Regression: GLOBAL ALIASING (Seed Err: 30.0°, Ens: 128)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 15.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)
        
        def streaming_callback(time, U_preds, losses, mean_loss, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,   
            sigma_q_min=0.05,    # Keep capture funnel wide enough for physical peaks!
            annealing_rate=0.5,  # Slower annealing to let the ensemble explore
            gamma_event=1e-4,    # Fast learning rate to let the ensemble collapse and move!
            gamma_step=100.0,
            kappa_init=100.0,
            n_ensemble=128, 
            streaming_callback=streaming_callback
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Global Aliasing failed to escape trap: Final Error {final_err:.2f}° >= 2.0°")

    def test_background_robustness(self):
        print(f"\n{'='*60}\nExecuting Regression: BACKGROUND ROBUSTNESS (80% Noise, Seed Err: 5.0°)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # Generates a massive 80% uniform random spherical noise!
        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0, bg_fraction=0.95)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)
        
        def streaming_callback(time, U_preds, losses, mean_loss, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,   
            sigma_q_min=0.05,    # Don't starve SDE by shrinking smaller than physics
            annealing_rate=0.5,
            gamma_event=1e-4,    # Agile enough to rotate 5 degrees in 5 seconds
            gamma_step=100.0,
            kappa_init=100.0,
            n_ensemble=1,        
            streaming_callback=streaming_callback
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Background test failed: Tracker derailed by noise (Final Error {final_err:.2f}° >= 2.0°)")

if __name__ == '__main__':
    unittest.main()
