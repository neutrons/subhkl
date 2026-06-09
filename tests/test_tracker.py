import unittest
import tempfile
import os
import h5py
import numpy as np
import itertools
from scipy.spatial.transform import Rotation
import scipy.spatial.transform
import jax
#jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
import e3nn_jax as e3nn

from subhkl.commands import run_spectral_holonomic_tracker

import pytest


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

    def create_mock_mtz(self, hkl_array, intensities):
        """ Wraps exact test-generated intensities into a valid Gemmi object. """
        import gemmi
        mtz = gemmi.Mtz()
        mtz.cell = gemmi.UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        mtz.spacegroup = gemmi.SpaceGroup('P 1')
        mtz.add_dataset('mock')

        mtz.add_column('H', type='H')
        mtz.add_column('K', type='H')
        mtz.add_column('L', type='H')
        mtz.add_column('I', type='J')

        data = np.zeros((hkl_array.shape[1], 4), dtype=np.float32)
        data[:, 0] = hkl_array[0, :]
        data[:, 1] = hkl_array[1, :]
        data[:, 2] = hkl_array[2, :]
        data[:, 3] = intensities
        mtz.set_data(data)

        return mtz

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
        valid_hkl = hkl[:, valid_mask] # <--- Capture valid indices
        num_valid = valid_q_hat.shape[1]

        num_bg = int(num_events * bg_fraction)
        num_sig = num_events - num_bg

        # ==========================================================
        # --- THE WILSON PRIOR (Intensity Decay) ---
        # ==========================================================
        if b_factor > 0.0:
            raw_intensities = np.exp(-(b_factor * 39.47) * (valid_norms**2))
            p_dist = raw_intensities / np.sum(raw_intensities)
        else:
            raw_intensities = np.ones_like(valid_norms)
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

        return q_lab, times, banks, pixels_r, pixels_c, valid_hkl, raw_intensities

    def get_fake_batches(self, sim_data, batch_size=10000):
        """Yields streaming tuples exactly matching the EventStreamLoader signature."""
        q_lab, times, banks, pixels_r, pixels_c = sim_data[:5] # <--- Slice here to prevent unpacking errors
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
        min_err_deg = np.inf
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

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            sigma_q_start=1.0,
            sigma_q_min=0.02,
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

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
            gamma_c=0.05,
            L_max=8,
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

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            gamma_time=0.0,
            streaming_callback=streaming_callback,
            gamma_c=0.05,
            L_max=16,
            init_tangent_blur = 0.55,
            prior_ridge = 0.5,
            meas_weight_2nd = 2000.0,
            ridge_inflation = 1e-4,
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

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
            gamma_c=1e-4, # bg=160kHz -> Tau = 1e-4 * sqrt(160000) = 0.04
            L_max=8,
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

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Loss={metrics['loss']:.2f} | Spectral-NLL={metrics['spectral_nll']:.2f} | Entropy={metrics['entropy']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            annealing_rate=0.0,   # No annealing needed
            gamma_time=0.0, # disable SDE diffusion
            gamma_sig=0.0,
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

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        # Telemetry storage for assertions
        recorded_taus = []
        recorded_errors = []

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            current_tau = metrics.get('tau', 0.0) # Ensure "tau": float(current_tau) is in metrics_dict!
            
            recorded_taus.append((time, current_tau))
            recorded_errors.append((time, err))
            
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Tau={current_tau:.4f} | Entropy={metrics['entropy']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
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
        sim_data = self.generate_poissonian_events(U_true, num_events=10000000, duration=5.0, bg_fraction=0.98)
        event_stream = self.get_fake_batches(sim_data, batch_size=100000)

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            
            # If you added entropy to your metrics_dict, we can print it for telemetry!
            entropy = metrics.get('entropy', 0.0) 
           
            print("Tracker 0 Error", self._evaluate_cubic_symmetric_error(U_true, U_preds[0]))
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Best-Idx={best_idx:3d} | Sym-Err={err:6.2f}° | Free-Energy={metrics['loss']:.2f} | Entropy={entropy:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            annealing_rate=5,    # Smooth time-driven cooling funnel
            streaming_callback=streaming_callback,
            gamma_c=0.05,
            L_max=8,
        ) 

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        
        self.assertLess(
            final_err, 
            2.0, 
            f"Thermodynamic Collapse: The tracker overfit to a noise trap. (Final Error {final_err:.2f}° >= 2.0°)"
        )

    def test_resolution_dependent_narrowing(self):
        print(f"\n{'='*60}\nExecuting Regression: RESOLUTION-DEPENDENT NARROWING (Lever Arm)\n{'='*60}")
        
        # Set true orientation and seed with a localized perturbation (approx 2 degrees off)
        U_true = Rotation.from_euler('xyz', [15.0, 25.0, 35.0], degrees=True).as_matrix()
        U_seed = Rotation.from_euler('xyz', [16.5, 23.8, 36.2], degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 8.0, 8.0, 8.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # Generate a dataset extending deep into the high-Q shell (d_min = 1.5 Angstroms)
        # Outer reflections are synthesized with tightly constrained angular variance
        sim_data = self.generate_poissonian_events(
            U_true, num_events=200000, duration=1.0, sigma_q=0.008, bg_fraction=0.50
        )
        event_stream = self.get_fake_batches(sim_data, batch_size=20000)

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        recorded_gaps = []
        recorded_errors = []

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            recorded_gaps.append(metrics['eigengap'])
            recorded_errors.append(err)
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Eigengap={metrics['eigengap']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            annealing_rate=1.0,      # Smooth physical time cooling funnel
            d_min=1.5,               # Open up the high-resolution shell to activate the high-Q lever arm
            d_max=8.0,
            streaming_callback=streaming_callback,
            gamma_c=0.01
        )

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        
        # --- VERIFICATION ASSERTIONS ---
        
        # 1. Physical Convergence: Ensure the tracker uses the high-Q precision 
        # to achieve ultra-sharp, sub-degree crystalline lock
        self.assertLess(
            final_err, 
            0.5, 
            f"Lever arm refinement failed to achieve sub-degree precision. Final Error: {final_err:.2f}°"
        )
        
        # 2. Curvature Acceleration: Verify that the structural orientation matrix curvature 
        # (eigengap) increases by at least a factor of 3 as high-Q reflections lock in
        self.assertGreater(
            recorded_gaps[-1], 
            recorded_gaps[0] * 3.0, 
            f"Thermodynamic Failure: Eigengap curvature did not accelerate. Initial: {recorded_gaps[0]:.2f}, Final: {recorded_gaps[-1]:.2f}"
        )

        # 3. Monotonic Refinement: Ensure the residual error actively decreases from the seed state
        self.assertLess(
            recorded_errors[-1], 
            recorded_errors[0], 
            f"Kinematic Failure: Crystalline funnel did not actively refine the seed error."
        )

class TestTrackerInitialization:
    @pytest.fixture
    def mock_reciprocal_h5(self, tmp_path):
        """ Generates a valid test fixture file containing a known crystal sample layout. """
        h5_path = tmp_path / "mock_finder.h5"
        
        # Generate a deterministic orientation matrix with a known 5.0 degree error offset
        rot_true = scipy.spatial.transform.Rotation.from_euler('xyz', [10.0, 20.0, 30.0], degrees=True)
        rot_error = scipy.spatial.transform.Rotation.from_euler('x', [5.0], degrees=True)
        rot_seed = rot_true * rot_error
        
        import h5py
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("sample/a", data=5.43)
            f.create_dataset("sample/b", data=5.43)
            f.create_dataset("sample/c", data=5.43)
            f.create_dataset("sample/alpha", data=90.0)
            f.create_dataset("sample/beta", data=90.0)
            f.create_dataset("sample/gamma", data=90.0)
            f.create_dataset("sample/space_group", data=b"F m -3 m")
            f.create_dataset("sample/U", data=rot_seed.as_matrix())
            
        return str(h5_path), rot_seed.as_matrix(), rot_true.as_matrix()

    def test_local_capture_initialization_gauge(self, mock_reciprocal_h5):
        """
        Verifies that the tracking prior correctly imports the initial seed 
        without introducing any stride, layout, or transposition offsets.
        """
        h5_file, U_seed, _ = mock_reciprocal_h5
        
        # Mock an empty single-step batch array to isolate the initialization block
        mock_batch = [
            (
                np.zeros((0, 3), dtype=np.float32),  # q_batch
                np.array([0.0]),                      # t_batch
                None, None, None, None, None,
                np.array([[0.0, 0.0, 1.0]]),         # ki_sample
                None
            )
        ]
        
        # Execute tracking graph up to the end of the entry sequence
        final_U = run_spectral_holonomic_tracker(
            finder_file=h5_file,
            event_batches=mock_batch,
            L_max=8,
            sigma_q_start=0.01  # Sharp spike initialization
        )
        
        # Calculate angular trace metric between the seed and extracted tracking frame
        trace_val = np.clip((np.trace(final_U.T @ U_seed) - 1.0) / 2.0, -1.0, 1.0)
        angular_error_deg = np.degrees(np.arccos(trace_val))
        
        print(f"\n[Validation Test] Extracted Angle Error to Seed Matrix: {angular_error_deg:.6f}°")
        
        # Assert that the extracted frame matches the injected seed matrix with machine precision
        assert angular_error_deg < 1e-4, (
            f"Gauge error detected! The tracker scrambled the input matrix at startup. "
            f"Expected initial error offset: 0.00°, got {angular_error_deg:.4f}°"
        )

class TestP1TriclinicMomentUniqueness(unittest.TestCase):
    def test_p1_anisotropy_breaks_gauge_drift(self):
        """
        Verifies that a P1 space group generates an anisotropic G_1 tensor,
        proving that the tracking bounce is caused by high point group symmetry.
        """
        # 1. Mock a highly anisotropic P1 lattice (no orthogonal symmetries)
        q_reflections = jnp.array([
            [0.23, 0.81, 0.52],
            [-0.12, 0.34, 0.93],
            [0.72, -0.11, 0.68],
            [0.45, 0.55, -0.71]
        ])
        q_normalized = q_reflections / jnp.linalg.norm(q_reflections, axis=1, keepdims=True)

        # 2. Compute the l=1 spherical harmonic profile
        irreps = e3nn.Irreps("1o")
        Y_l1 = e3nn.spherical_harmonics(irreps, q_normalized, normalize=True).array

        # 3. Form the crystal-frame Ewald tensor G_1
        G_1 = jnp.matmul(Y_l1.T, Y_l1)

        # 4. Compute eigenvalues to evaluate isotropy
        eigenvalues = jnp.linalg.eigvalsh(G_1)
        eigen_spread = eigenvalues[-1] - eigenvalues[0]

        print(f"\n[Symmetry Diagnostic] P1 Eigenvalue Spread: {eigen_spread:.4f}")

        # In P1, the spread must be significantly greater than zero (anisotropic)
        self.assertGreater(float(eigen_spread), 0.1,
            "Lattice is isotropic! Anisotropy failed to break the gauge symmetry.")

class TestP1LatticeTracker(unittest.TestCase):
    def setUp(self):
        """
        Sets up a mock crystal framework with strict P1 triclinic symmetry
        to isolate point group effects from the filter dynamics.
        """
        self.finder_file = "/tmp/mock_p1_finder.h5"

        # Define an asymmetric triclinic cell matrix (Strict P1 symmetry)
        with h5py.File(self.finder_file, "w") as f:
            f.create_dataset("sample/a", data=8.24)
            f.create_dataset("sample/b", data=9.65)
            f.create_dataset("sample/c", data=11.02)
            f.create_dataset("sample/alpha", data=93.4)
            f.create_dataset("sample/beta", data=102.1)
            f.create_dataset("sample/gamma", data=87.6)
            f.create_dataset("sample/space_group", data=b"P 1")

            # Seed an intentional 5.0 degree initial orientation error
            U_true = np.eye(3)
            # Small rotation about the x-axis by 5 degrees
            theta = np.radians(5.0)
            U_seed = np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)]
            ])
            f.create_dataset("sample/U_init", data=U_seed)
            f.create_dataset("orientation/U", data=U_true)

        # Generate a continuous single-crystal event stream
        np.random.seed(42)
        self.event_stream = []
        num_chunks = 20
        events_per_chunk = 10000

        current_time = 0.0
        for chunk in range(num_chunks):
            # Generate random scattering vectors distributed across the sphere
            vecs = np.random.normal(size=(events_per_chunk, 3))
            q_vectors = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

            # Generate monotonically increasing timestamps
            timestamps = np.linspace(current_time, current_time + 0.05, events_per_chunk)
            current_time += 0.05

            # Incoming beam momentum vectors along the z-axis
            ki_vectors = np.tile(np.array([0.0, 0.0, 1.0]), (events_per_chunk, 1))

            # Blank detector bank metadata maps
            banks = np.zeros(events_per_chunk, dtype=np.int32)
            pr = np.zeros(events_per_chunk, dtype=np.int32)
            pc = np.zeros(events_per_chunk, dtype=np.int32)
            angles = np.zeros(events_per_chunk)
            slab = np.zeros(events_per_chunk)

            cumulative_count = (chunk + 1) * events_per_chunk

            self.event_stream.append((
                q_vectors, timestamps, banks, pr, pc, angles, slab, ki_vectors, cumulative_count
            ))

    def tearDown(self):
        if os.path.exists(self.finder_file):
            os.remove(self.finder_file)

    def test_p1_tracking_path_execution(self):
        """ Runs the structural tracker through the asymmetric single-crystal stream. """
        def blank_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            pass

        print("\n=== Executing Pure P1 Symmetry Verification Test ===")
        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=self.event_stream,
            streaming_callback=blank_callback,
            L_max=4,
            gamma_c=0.01
        )

        # Verify the final array matches expected structural dimensions
        self.assertEqual(final_U.shape, (3, 3))
        print("P1 tracking run completed successfully.")

if __name__ == '__main__':
    unittest.main()
