import unittest
import tempfile
import os
import h5py
import numpy as np
import itertools
from scipy.spatial.transform import Rotation
import scipy.spatial.transform
import jax
import jax.numpy as jnp

from subhkl.commands import run_spectral_holonomic_tracker, _sample_to_lab_matrix
from subhkl.instrument.goniometer import lab_to_sample

import pytest
import e3x

@pytest.fixture(scope="session", autouse=True)
def setup_temp_e3x_cache(tmp_path_factory):
    """
    Creates a temporary cache that lasts exactly as long as the pytest run.
    """
    # 1. Ask pytest for a session-scoped temporary directory
    temp_dir = tmp_path_factory.mktemp("e3x_cache")
    cache_path = temp_dir / "sph.npz"

    # 2. Point e3x to this path. Do NOT write fake data to it!
    # e3x will see the file doesn't exist, calculate the real L=16 table, and save it here.
    e3x.Config.set_spherical_harmonics_cache(str(cache_path))

    yield  # 3. All tests run here. 

    # 4. Clean up the global e3x state
    e3x.Config.set_spherical_harmonics_cache("")
    # (pytest automatically deletes the temp directory when the session ends)

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
            streaming_callback=streaming_callback,
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
            L_max=8,
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Local Capture failed to converge: Final Error {final_err:.2f}° >= 2.0°")

    def test_global_aliasing(self):
        print(f"\n{'='*60}\nExecuting Regression: GLOBAL ALIASING (Seed Err: 30.0°)\n{'='*60}")
        
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
            streaming_callback=streaming_callback,
            L_max=8,
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
            L_max=8,
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Background test failed: Tracker derailed by noise (Final Error {final_err:.2f}° >= 2.0°)")

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
        
        valid_hkl = data_p1[5]
        intensities = data_p1[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
        
        # Telemetry storage for assertions
        recorded_errors = []

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            recorded_errors.append((time, err))
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}°")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
        )

        # --- SURVIVAL ASSERTIONS ---
        times_arr = np.array([t[0] for t in recorded_errors])
        errs_arr = np.array([e[1] for e in recorded_errors])

        # Extract Phase 2 slice (The Flash)
        phase2_mask = (times_arr > 1.5) & (times_arr <= 3.5)

        max_err_during_flash = np.max(errs_arr[phase2_mask])
        final_err = errs_arr[-1]

        # 1. Did the tracker maintain topological lock during the flash? (Didn't shatter)
        self.assertLess(max_err_during_flash, 15.0, 
                        f"Tracking Failure: The flash shattered the tracker (Max Error {max_err_during_flash:.2f}° >= 15.0°)")

        # 2. Did it recover absolute precision?
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
            
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Best-Idx={best_idx:3d} | Sym-Err={err:6.2f}° | Free-Energy={metrics['loss']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            annealing_rate=5,    # Smooth time-driven cooling funnel
            streaming_callback=streaming_callback,
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
            L_max=16,                # needed for high resolution
            process_q_scale_start=1e-4, # Start cooler since we are only 2 degrees off
            process_q_scale_end=1e-9,   # End much colder to allow maximum precision
            streaming_callback=streaming_callback,
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

    def generate_anisotropic_background(self, num_bg, duration=5.0,
                                        axis=np.array([0.0, 0.0, 1.0]),
                                        spread=0.35):
        """
        Structured (non-isotropic) background concentrated around `axis`
        (default: the incident beam, +z). Models forward/air scatter and the
        incoherent beamstop halo. Unlike a uniform sphere, this has real
        l=1/l=2 content that survives the deviatoric SH projection.

        Returns (q_dirs (N,3) float32, times (N,) float32).
        """
        axis = axis / np.linalg.norm(axis)
        vecs = axis[None, :] + np.random.normal(0.0, spread, size=(num_bg, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        times = np.sort(np.random.uniform(0.0, duration, num_bg))
        return vecs.astype(np.float32), times.astype(np.float32)


    def test_anisotropic_background_rejection(self):
        print(f"\n{'='*60}\nExecuting Regression: ANISOTROPIC (STRUCTURED) BACKGROUND\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        duration = 5.0
        n_sig = 400_000
        n_bg = 600_000  # 60% STRUCTURED background (much milder than the 98% isotropic tests)

        # Pure signal events at U_true (reuse the validated generator, no internal bg).
        sig = self.generate_poissonian_events(
            U_true, num_events=n_sig, duration=duration, bg_fraction=0.0
        )
        sig_q, sig_t = sig[0], sig[1]
        valid_hkl, intensities = sig[5], sig[6]

        # Forward-scatter lobe along the beam (+z). Try axis=normalize([0,0.3,1]) or a
        # second lobe to make it harder; even this single lobe should bias the estimate.
        bg_q, bg_t = self.generate_anisotropic_background(
            n_bg, duration=duration, axis=np.array([0.0, 0.0, 1.0]), spread=0.2
        )

        # Merge and re-sort chronologically -- mirrors the loader's global time sort,
        # so every batch sees a realistic signal/background mix rather than blocks.
        q_lab = np.concatenate([sig_q, bg_q], axis=0)
        times = np.concatenate([sig_t, bg_t])
        order = np.argsort(times, kind="stable")
        q_lab, times = q_lab[order], times[order]

        N = len(times)
        sim_data = (
            q_lab,
            times,
            np.ones(N, dtype=int),    # banks
            np.zeros(N, dtype=int),   # pixels_r
            np.zeros(N, dtype=int),   # pixels_c
        )

        event_stream = self.get_fake_batches(sim_data, batch_size=10_000)
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        errs = []

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            errs.append(err)
            print(f"  -> [t={time:4.2f}s | {neutron_count:7d} evts] "
                  f"Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
            L_max=8,
        )

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)

        self.assertLess(
            final_err, 2.0,
            f"Structured-background rejection failed: a beam-aligned scatter lobe "
            f"biased the orientation (Final Error {final_err:.2f}° >= 2.0°). "
            f"Isotropic-background tests miss this because a uniform sphere cancels "
            f"in the deviatoric SH moment; a forward lobe does not."
        )

    def apply_detector_coverage(self, q, times, coverage_fraction,
                                axis=np.array([1.0, 0.0, 0.0])):
        """
        Cull events to a single spherical cap of solid angle
        Omega = coverage_fraction * 4*pi, centered on `axis` (a stand-in for the
        detector bank center in the sample frame).

            Omega = 2*pi*(1 - cos a)  =>  cos a = 1 - 2 * coverage_fraction

            coverage_fraction = 1.00 -> full sphere   (a = 180 deg)
            coverage_fraction = 0.50 -> hemisphere    (a =  90 deg)
            coverage_fraction = 0.25 -> 60 deg cap

        Returns the masked (q, times).
        """
        axis = axis / np.linalg.norm(axis)
        cos_a = 1.0 - 2.0 * coverage_fraction
        cos_theta = q @ axis
        keep = cos_theta >= cos_a
        return q[keep], times[keep]


    def test_partial_detector_coverage(self):
        print(f"\n{'='*60}\nExecuting Regression: PARTIAL DETECTOR S^2 COVERAGE\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        duration = 5.0

        # Large clean signal pool (NO background) so every coverage level still has
        # ample statistics after masking and subsampling.
        sig = self.generate_poissonian_events(
            U_true, num_events=2_000_000, duration=duration, bg_fraction=0.0
        )
        q_all, t_all = sig[0], sig[1]
        valid_hkl, intensities = sig[5], sig[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        # Detector center: a side bank ~90 deg off the beam (beam is +z here).
        det_axis = np.array([1.0, 0.0, 0.0])

        coverages = [1.0, 0.5, 0.25]

        # Fix the statistics budget: subsample every level to the count available
        # at the tightest cap, isolating geometry from photon starvation.
        axis_n = det_axis / np.linalg.norm(det_axis)
        counts = {
            f: int(np.sum((q_all @ axis_n) >= (1.0 - 2.0 * f)))
            for f in coverages
        }
        common_n = min(counts.values())
        print(f"  Common event budget (all levels): {common_n:,}")

        rng = np.random.default_rng(0)
        results = {}

        for f in coverages:
            q_c, t_c = self.apply_detector_coverage(
                q_all, t_all, coverage_fraction=f, axis=det_axis
            )
            idx = rng.choice(len(t_c), size=common_n, replace=False)
            q_c, t_c = q_c[idx], t_c[idx]
            order = np.argsort(t_c, kind="stable")  # mirror the loader's time sort
            q_c, t_c = q_c[order], t_c[order]

            N = len(t_c)
            sim_data = (
                q_c, t_c,
                np.ones(N, dtype=int),
                np.zeros(N, dtype=int),
                np.zeros(N, dtype=int),
            )
            event_stream = self.get_fake_batches(sim_data, batch_size=10_000)

            # Fresh seed each run (tracker reads sample/U and writes a tracking group).
            with h5py.File(self.finder_file, "w") as fh:
                fh["sample/a"], fh["sample/b"], fh["sample/c"] = 10.0, 10.0, 10.0
                fh["sample/alpha"], fh["sample/beta"], fh["sample/gamma"] = 90.0, 90.0, 90.0
                fh["sample/space_group"] = b"P 1"
                fh["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
                fh["sample/U"] = U_seed

            final_U = run_spectral_holonomic_tracker(
                finder_file=self.finder_file,
                event_batches=event_stream,
                structure_factors=mock_mtz,
                L_max=8,
            )
            err = self._evaluate_cubic_symmetric_error(U_true, final_U)
            results[f] = err
            print(f"  coverage={f*100:5.1f}% of 4pi | events={N:,} | final Sym-Err={err:6.2f} deg")

        # Sanity: full-sphere coverage with the SAME budget must converge.
        self.assertLess(
            results[1.0], 2.0,
            f"Full-coverage control failed ({results[1.0]:.2f} deg) -- statistics "
            f"budget too small; raise num_events before trusting the partial runs."
        )

        # Regression: a hemispherical detector (Omega = 2*pi) must not silently bias U.

        # the hemisphere is adversarial because it cuts the U rotation axis, so
        # increase the tolerance (there is tension with Ewald sphere physics and the 
        # global aliasing test)
        self.assertLess(
            results[0.5], 10.0,
            f"Partial-coverage bias: a hemispherical detector (Omega=2pi) produced a "
            f"{results[0.5]:.2f} deg error at identical statistics to the converging "
            f"full-sphere control ({results[1.0]:.2f} deg). The forward model "
            f"integrates Ewald-allowed reflections over the full sphere while the data "
            f"populate only the covered cap; the missing detector acceptance mask is "
            f"an uncorrected anisotropic window on S^2."
        )

    def test_sample_lab_transform_roundtrip(self):
        """
        DECISIVE check: the tracker's sample->lab matrix must invert the loader's
        lab_to_sample for finite multi-axis angles AND non-zero offsets. Convention
        -agnostic -- both sides use the same goniometer module, so this asserts they
        are mutually inverse, which is exactly what the coverage path relies on
        (q_lab = R_batch @ q_sample). Deterministic, fast, no Kalman filter.
        """
        print(f"\n{'='*60}\nUnit: SAMPLE<->LAB ROUND-TRIP (finite angles + offsets)\n{'='*60}")
        rng = np.random.default_rng(0)

        cases = [
            # (axes (na,3),                         angles (na,),          offsets (na,) | None)
            (np.array([[0.0, 1.0, 0.0]]),           np.array([30.0]),      None),
            (np.array([[0.0, 1.0, 0.0]]),           np.array([30.0]),      np.array([5.0])),
            (np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0]]),           np.array([25.0, 15.0]), np.array([3.0, -2.0])),
            (np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]]),           np.array([40.0, -20.0, 10.0]),
                                                    np.array([1.0, 2.0, -3.0])),
        ]

        for axes, ang, offsets in cases:
            na = len(axes)
            # R = sample->lab at this constant setting, as the tracker builds it.
            R = np.asarray(_sample_to_lab_matrix(axes, ang, offsets), dtype=float)

            # 1) Proper rotation.
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4,
                err_msg=f"R not orthonormal for axes={axes.tolist()} ang={ang.tolist()}")
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=4,
                msg=f"det(R) != +1 for ang={ang.tolist()} offsets={offsets}")

            # 2) R inverts the loader's lab_to_sample on independent probe vectors.
            v_lab = rng.normal(size=(64, 3))
            v_lab /= np.linalg.norm(v_lab, axis=1, keepdims=True)
            ang_full = np.tile(ang.reshape(na, 1), (1, v_lab.shape[0]))      # (na, N)
            v_sample = np.asarray(
                lab_to_sample(v_lab, axes, ang_full, None, offsets, is_vector=True))
            v_lab_rec = (R @ v_sample.T).T

            np.testing.assert_allclose(
                v_lab_rec, v_lab, atol=1e-4,
                err_msg=("tracker R_batch does not invert loader lab_to_sample for "
                         f"finite angles/offsets (axes={axes.tolist()}, ang={ang.tolist()}, "
                         f"offsets={offsets}). Coverage map would land in the wrong "
                         "lab region."))

            print(f"  ok: na={na} ang={ang.tolist()} offsets={offsets} "
                  f"-> max round-trip err {np.max(np.abs(v_lab_rec - v_lab)):.2e}")

    def test_goniometer_finite_setting(self):
        """
        Integration: run the tracker with the transform path ACTIVE (finite, non-
        trivial multi-axis goniometer setting + offsets) and confirm it recovers the
        sample-frame U_true and stays stable.
     
        Faithful construction (matches the loader exactly):
          * The crystal's lab orientation under setting R is U_eff = R @ U_true, so
            events are generated at U_eff (correct diffracting set for the fixed beam).
          * The loader de-rotates: q_sample = lab_to_sample(q_lab) = U_true @ q_theo,
            and the beam in the sample frame is ki_sample = lab_to_sample([0,0,1]).
          * The tracker therefore must recover U_true (NOT U_eff).
     
        NOTE ON SCOPE: at full coverage the visibility gate is inert, so this does not
        test masking correctness -- it verifies the active R_batch path is plumbed,
        the _sample_to_lab_matrix runtime self-check passes for the real setting, and
        the loader<->tracker frame contract is self-consistent. Masking correctness is
        the deterministic round-trip test above, plus the optional heavier test.
        """
        print(f"\n{'='*60}\nIntegration: FINITE GONIOMETER SETTING (+offsets)\n{'='*60}")
     
        axes = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # two-axis goniometer
        ang = np.array([20.0, -12.0])                          # finite, non-trivial
        offs = np.array([5.0, -3.0])                           # finite offsets
        na = len(axes)
     
        U_true = Rotation.from_euler("y", 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler("y", 40.0, degrees=True).as_matrix()   # 5 deg off
     
        # sample->lab for this setting, built from the trusted loader transform.
        R = np.asarray(_sample_to_lab_matrix(axes, ang, offs), dtype=float)
        U_eff = R @ U_true
     
        # Sanity: the setting is genuinely non-trivial (so this isn't the static case).
        self.assertGreater(self._evaluate_cubic_symmetric_error(U_true, U_eff), 5.0,
            "goniometer setting is too close to identity to be a meaningful test")
     
        # Generate at U_eff -> correct lab diffracting set for the fixed beam [0,0,1].
        sim = self.generate_poissonian_events(U_eff, num_events=1_000_000,
                                              duration=5.0, bg_fraction=0.0)
        q_lab, t = sim[0], sim[1]
        valid_hkl, intensities = sim[5], sim[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
     
        # De-rotate to the sample frame exactly as the loader does.
        N = len(t)
        ang_full = np.tile(ang.reshape(na, 1), (1, N))                    # (na, N)
        q_sample = np.asarray(
            lab_to_sample(q_lab, axes, ang_full, None, offs, is_vector=True))
        ki_sample = np.asarray(
            lab_to_sample(np.tile([0.0, 0.0, 1.0], (N, 1)), axes, ang_full,
                          None, offs, is_vector=True))
        angles_col = np.tile(ang.reshape(1, na), (N, 1))                  # (N, na)
     
        def emit(batch=10000):
            order = np.argsort(t, kind="stable")
            qs, ts, ac, ks = q_sample[order], t[order], angles_col[order], ki_sample[order]
            for s in range(0, N, batch):
                e = min(s + batch, N)
                n = e - s
                yield (
                    qs[s:e].astype(np.float32),
                    ts[s:e].astype(np.float32),
                    np.ones(n, dtype=np.int16),
                    np.zeros(n, dtype=np.int16),
                    np.zeros(n, dtype=np.int16),
                    ac[s:e].astype(np.float32),          # angles (n, na)  <-- the point
                    np.zeros((n, 3), dtype=np.float32),  # s_lab
                    ks[s:e].astype(np.float32),          # ki_sample (de-rotated beam)
                    e,
                )
     
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed
     
        final_U = run_spectral_holonomic_tracker(
            finder_file=self.finder_file,
            event_batches=emit(),
            structure_factors=mock_mtz,
            gonio_axes=axes,            # turns the transform path ON
            gonio_offsets=offs,
            L_max=8,
        )
     
        err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        print(f"  recovered sample-frame U: Sym-Err = {err:.3f} deg "
              f"(U_eff was {self._evaluate_cubic_symmetric_error(U_true, U_eff):.1f} deg from U_true)")
        self.assertLess(err, 2.0,
            f"Finite-setting recovery failed: {err:.2f} deg >= 2.0. The tracker did "
            f"not recover the sample-frame U_true with the goniometer path active.")

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
                np.zeros((0,), dtype=np.float32),    # t_batch
                np.zeros((0,), dtype=np.int16),      # banks
                np.zeros((0,), dtype=np.int16),      # pr
                np.zeros((0,), dtype=np.int16),      # pc
                np.zeros((0, 1), dtype=np.float32),  # angles
                np.zeros((0, 3), dtype=np.float32),  # slab
                np.zeros((0, 3), dtype=np.float32),  # ki_sample
                0,                                   # cumulative count
            )
        ]

        # Execute tracking graph up to the end of the entry sequence
        final_U = run_spectral_holonomic_tracker(
            finder_file=h5_file,
            event_batches=mock_batch,
            L_max=8,
        )
        
        # Calculate angular trace metric between the seed and extracted tracking frame
        trace_val = np.clip((np.trace(final_U.T @ U_seed) - 1.0) / 2.0, -1.0, 1.0)
        angular_error_deg = np.degrees(np.arccos(trace_val))
        
        print(f"\n[Validation Test] Extracted Angle Error to Seed Matrix: {angular_error_deg:.6f}°")
        
        # Assert that the extracted frame matches the injected seed matrix with machine precision
        assert angular_error_deg < 0.05, (
            f"Gauge error detected! The tracker scrambled the input matrix at startup. "
            f"Expected initial error offset: 0.00°, got {angular_error_deg:.4f}°"
        )

