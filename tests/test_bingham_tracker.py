import unittest
import tempfile
import os
import h5py
import numpy as np
import itertools
from unittest.mock import patch
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
        self.nexus_file = os.path.join(self.test_dir.name, "mock_nexus.h5")
        self.output_file = os.path.join(self.test_dir.name, "output.h5")
        
        with h5py.File(self.nexus_file, "w") as f:
            f.create_group("entry/bank1_events")
            
        # Ensure deterministic testing environment
        np.random.seed(42)

    def tearDown(self):
        self.test_dir.cleanup()

    def generate_poissonian_events(self, U_true, num_events=200000, duration=5.0, sigma_q=0.05):
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
        wavelengths = -(4 * np.pi / q_norms) * kinematic_proj

        valid_mask = (wavelengths > 0.5) & (wavelengths < 10.0)
        valid_q_hat = q_theo_hat[:, valid_mask]
        valid_norms = q_norms[valid_mask]
        num_valid = valid_q_hat.shape[1]

        peak_indices = np.random.choice(num_valid, size=num_events)
        
        q_exp_list = []
        for idx in peak_indices:
            q_hat_lab = U_true @ valid_q_hat[:, idx]
            angular_std = sigma_q / valid_norms[idx]
            noise_vec = np.random.normal(0, angular_std, 3)
            q_exp = q_hat_lab + noise_vec
            q_exp /= np.linalg.norm(q_exp)
            q_exp_list.append(q_exp)

        times = np.sort(np.random.uniform(0, duration, num_events)) 
        banks = np.ones(num_events, dtype=int)
        pixels_r = np.zeros(num_events, dtype=int)
        pixels_c = np.zeros(num_events, dtype=int)

        return np.array(q_exp_list), times, banks, pixels_r, pixels_c

    def _evaluate_cubic_symmetric_error(self, U_true, U_pred):
        min_err_deg = 180.0
        for sym in get_cubic_symmetries():
            U_mate = U_true @ sym
            trace_val = np.clip(np.trace(U_mate.T @ U_pred), -1.0, 3.0)
            err_deg = np.degrees(np.arccos((trace_val - 1.0) / 2.0))
            min_err_deg = min(min_err_deg, err_deg)
        return min_err_deg

    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_local_capture(self, mock_executor_class):
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
        
        mock_executor_instance = mock_executor_class.return_value
        mock_executor_instance.__enter__.return_value.map.return_value = [sim_data]
        
        def streaming_callback(time, U_preds, losses, mean_loss, best_idx, neutron_count, new_events, eigengap=0.0):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[0])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={eigengap:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            output_h5_filename=self.output_file,
            event_nexus_filename=self.nexus_file,
            sigma_q_start=1.0,   
            sigma_q_min=0.05,    
            annealing_rate=1.0,  # Rapid decay for 5-second simulated run
            lambda_alpha=0.5,
            gamma_diffusion=1.0,
            kappa_init=100.0,
            batch_size_events=10000, 
            n_ensemble=1, 
            streaming_callback=streaming_callback
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Local Capture failed to converge: Final Error {final_err:.2f}° >= 2.0°")

    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_global_aliasing(self, mock_executor_class):
        print(f"\n{'='*60}\nExecuting Regression: GLOBAL ALIASING (Seed Err: 30.0°, Ens: 256)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 15.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        sim_data = self.generate_poissonian_events(U_true, num_events=200000, duration=5.0)
        
        mock_executor_instance = mock_executor_class.return_value
        mock_executor_instance.__enter__.return_value.map.return_value = [sim_data]
        
        def streaming_callback(time, U_preds, losses, mean_loss, best_idx, neutron_count, new_events, eigengap=0.0):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[0])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={eigengap:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            output_h5_filename=self.output_file,
            event_nexus_filename=self.nexus_file,
            sigma_q_start=1.0,   
            sigma_q_min=0.05,    
            annealing_rate=1.0,
            lambda_alpha=0.5,
            gamma_diffusion=1.0,
            kappa_init=100.0,
            batch_size_events=10000, 
            n_ensemble=256, 
            streaming_callback=streaming_callback
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Global Aliasing failed to escape trap: Final Error {final_err:.2f}° >= 2.0°")

if __name__ == '__main__':
    unittest.main()
