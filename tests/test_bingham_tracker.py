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

    def generate_poissonian_events(self, U_true, num_events=1000000, duration=5.0, sigma_q=0.008, bg_fraction=0.0, b_factor=0.0):
        # Busing-Levy convention (1/d) to match the tracker's geometry exactly
        B_mat = np.array([
            [1.0/10.0, 0, 0],
            [0, 1.0/10.0, 0],
            [0, 0, 1.0/10.0]
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
        
        # True kinematics in Busing-Levy space
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
            # Scale b_factor so the exponential drop is identical to the old 2*pi space
            # (2 * pi)^2 approx 39.47
            intensities = np.exp(-(b_factor * 39.47) * (valid_norms**2))
            p_dist = intensities / np.sum(intensities)
        else:
            p_dist = None 

        peak_indices = np.random.choice(num_valid, size=num_sig, p=p_dist)
        
        q_exp_list = []
        # 1. Generate Physical Signal Events
        for idx in peak_indices:
            q_hat_lab = U_true @ valid_q_hat[:, idx]
            
            # Since q_norms is now 2*pi smaller, sigma_q must also be smaller 
            # to maintain the exact same angular variance as the previous tests.
            # 0.05 / (2*pi) approx 0.008
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

        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0, b_factor=0.5)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,
            sigma_q_min=0.02,
            gamma_step=100.0,
            kappa_init=1.0,
            n_ensemble=1,
            streaming_callback=streaming_callback,
            gamma_c=0.05  # bg=0.0 -> Tau = 0.05 * sqrt(1) = 0.05
        )

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Wilson Modulation failed to converge: Final Error {final_err:.2f}° >= 2.0°")

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
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,   
            sigma_q_min=0.02,    
            gamma_step=100.0,
            kappa_init=1.0,
            n_ensemble=1, 
            streaming_callback=streaming_callback,
            gamma_c=0.05  # bg=0.0 -> Tau = 0.05 * sqrt(1) = 0.05
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
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,   
            sigma_q_min=0.05,    
            gamma_step=100.0,
            kappa_init=1.0,
            n_ensemble=256, 
            streaming_callback=streaming_callback,
            L_max=8,
            gamma_c=0.05
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
        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0, bg_fraction=0.80)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,   
            sigma_q_min=0.05,    
            gamma_step=100.0,
            kappa_init=1.0,
            n_ensemble=1,        
            streaming_callback=streaming_callback,
            gamma_c=1e-4  # bg=160kHz -> Tau = 1e-4 * sqrt(160000) = 0.04
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Background test failed: Tracker derailed by noise (Final Error {final_err:.2f}° >= 2.0°)")

    def test_spectral_basin_selection(self):
        print(f"\n{'='*60}\nExecuting Regression: SPECTRAL BASIN SELECTION (Blind M-Step)\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 15.0, degrees=True).as_matrix()

        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        sim_data = self.generate_poissonian_events(U_true, num_events=200000, duration=1.0)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Loss={metrics['loss']:.2f} | Spectral-NLL={metrics['spectral_nll']:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=0.005,  # Razor thin!
            sigma_q_min=0.005,
            annealing_rate=0.0,   # No annealing needed
            gamma_step=0.0,
            gamma_time=0.0, # disable SDE diffusion
            gamma_sig=0.0,
            kappa_init=100.0,
            n_ensemble=128,
            streaming_callback=streaming_callback,
            gamma_c=0.0,  # tau = 0 for this unit test to cancel short-range contribution
        )

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        
        self.assertLess(final_err, 31.0, f"Spectral E-Step failed to select Tracker 0. Error {final_err:.2f}°")

    def test_soc_background_flash(self):
        print(f"\n{'='*60}\nExecuting Regression: SELF-ORGANIZED CRITICALITY (Dynamic Flash)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # --- GENERATE A MULTI-PHASE DYNAMIC EVENT STREAM ---
        # Phase 1: Calm Approach (1.5s, 20% Noise)
        data_p1 = self.generate_poissonian_events(U_true, num_events=300000, duration=1.5, bg_fraction=0.20)
        
        # Phase 2: The Flash (2.0s, 98% Noise - Would shatter a static tracker)
        data_p2 = self.generate_poissonian_events(U_true, num_events=400000, duration=2.0, bg_fraction=0.98)
        
        # Phase 3: Recovery (1.5s, 20% Noise)
        data_p3 = self.generate_poissonian_events(U_true, num_events=300000, duration=1.5, bg_fraction=0.20)

        # Concatenate the streams and shift times to be continuous
        q_lab = np.concatenate([data_p1[0], data_p2[0], data_p3[0]])
        
        times_p2 = data_p2[1] + data_p1[1][-1]
        times_p3 = data_p3[1] + times_p2[-1]
        times = np.concatenate([data_p1[1], times_p2, times_p3])
        
        banks = np.concatenate([data_p1[2], data_p2[2], data_p3[2]])
        pixels_r = np.concatenate([data_p1[3], data_p2[3], data_p3[3]])
        pixels_c = np.concatenate([data_p1[4], data_p2[4], data_p3[4]])
        
        sim_data_flash = (q_lab, times, banks, pixels_r, pixels_c)
        event_stream = self.get_fake_batches(sim_data_flash, batch_size=10000)

        # Telemetry storage for assertions
        recorded_taus = []
        recorded_errors = []

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            current_tau = metrics.get('tau', 0.0) # Ensure "tau": float(current_tau) is in metrics_dict!
            
            recorded_taus.append((time, current_tau))
            recorded_errors.append((time, err))
            
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Tau={current_tau:.4f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,
            sigma_q_min=0.05,
            gamma_step=100.0,
            kappa_init=1.0,
            n_ensemble=1,
            streaming_callback=streaming_callback,
            gamma_c=1e-4,     # Enable SOC
            bg_ema_weight=0.85  # <--- Decrease thermal inertia for rapid recovery!
        )

        # --- THERMODYNAMIC ASSERTIONS ---
        times_arr = np.array([t[0] for t in recorded_taus])
        taus_arr = np.array([t[1] for t in recorded_taus])
        errs_arr = np.array([e[1] for e in recorded_errors])

        # Extract phase slices
        phase1_mask = times_arr <= 1.5
        phase2_mask = (times_arr > 1.5) & (times_arr <= 3.5)
        phase3_mask = times_arr > 3.5

        tau_p1_mean = np.mean(taus_arr[phase1_mask][-5:]) # End of P1
        tau_p2_peak = np.max(taus_arr[phase2_mask])       # Peak of Flash
        tau_p3_mean = np.mean(taus_arr[phase3_mask][-5:]) # End of P3

        max_err_during_flash = np.max(errs_arr[phase2_mask])
        final_err = errs_arr[-1]

        # 1. Did the tracker heat up to survive the flash?
        self.assertGreater(tau_p2_peak, tau_p1_mean * 1.3,
                           f"SOC Failure: Tau did not surge during flash. P1: {tau_p1_mean:.4f}, Flash Peak: {tau_p2_peak:.4f}")

        # 2. Did the tracker cool down after the flash?
        self.assertLess(tau_p3_mean, tau_p2_peak * 0.8, 
                        f"SOC Failure: Tau did not cool down after flash. Flash Peak: {tau_p2_peak:.4f}, P3: {tau_p3_mean:.4f}")

        # 3. Did the tracker maintain topological lock during the flash? (Didn't shatter)
        self.assertLess(max_err_during_flash, 15.0, 
                        f"Tracking Failure: The flash shattered the tracker (Max Error {max_err_during_flash:.2f}° >= 15.0°)")

        # 4. Did it recover absolute precision?
        self.assertLess(final_err, 2.0, 
                        f"Tracking Failure: Failed to regain precision after flash (Final Error {final_err:.2f}° >= 2.0°)")

    def test_thermodynamic_entropy_stabilization(self):
        print(f"\n{'='*60}\nExecuting Regression: THERMODYNAMIC ENTROPY STABILIZATION\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # Simulate the "4-panel" scenario: Moderate background, but plenty of time to overfit.
        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0, bg_fraction=0.98)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            
            # If you added entropy to your metrics_dict, we can print it for telemetry!
            entropy = metrics.get('entropy', 0.0) 
            
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Best-Idx={best_idx:3d} | Sym-Err={err:6.2f}° | Free-Energy={metrics['loss']:.2f} | Entropy={entropy:.2f}")

        final_U = run_bingham_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            sigma_q_start=1.0,   
            sigma_q_min=0.05,    
            gamma_step=100.0,
            kappa_init=1.0,
            # MASSIVE ENSEMBLE: 63 trackers will actively hunt for narrow noise traps to trick the argmin
            n_ensemble=64,       
            streaming_callback=streaming_callback,
            gamma_c=0.05
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        
        self.assertLess(
            final_err, 
            2.0, 
            f"Thermodynamic Collapse: The tracker overfit to a noise trap. (Final Error {final_err:.2f}° >= 2.0°)"
        )

if __name__ == '__main__':
    unittest.main()
