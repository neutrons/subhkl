import unittest
import tempfile
import os
import h5py
import numpy as np

# Adjust this import depending on exactly where EventStreamLoader lives in your codebase
from subhkl.config import beamlines
from subhkl.streaming.loader import EventStreamLoader  

class TestEventStreamLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.nexus_file = os.path.join(self.test_dir.name, "mock_non_contiguous.nxs")
        
        # 1. Dynamically select two non-contiguous banks that ACTUALLY exist in your config.
        # This bypasses multiprocessing pickling/mocking issues with the worker pool.
        self.instrument = "MANDI" # You can swap this to CG4D
        
        available_banks = list(beamlines.get(self.instrument, {}).keys())
        
        if len(available_banks) < 2:
            self.skipTest(f"Not enough banks configured for {self.instrument} to test non-contiguous loading.")
            
        # Pick the first and last bank strings (e.g., '17' and '42')
        self.bank_A_str = available_banks[0]
        self.bank_B_str = available_banks[-1]
        
        self.bank_A_id = int(self.bank_A_str)
        self.bank_B_id = int(self.bank_B_str)
        
        # 2. Create a mock NeXus file with ONLY these two specific banks
        with h5py.File(self.nexus_file, 'w') as f:
            entry = f.create_group('entry')
            
            # Setup Bank A (Inject exactly 3 events)
            bA = entry.create_group(f'bank{self.bank_A_id}_events')
            bA.create_dataset('event_id', data=np.array([1000, 1001, 1002]))
            bA.create_dataset('event_index', data=np.array([0, 2]))
            bA.create_dataset('event_time_offset', data=np.array([10.0, 20.0, 30.0]))
            bA.create_dataset('event_time_zero', data=np.array([1.0, 2.0]))
            
            # Setup Bank B (Inject exactly 2 events)
            bB = entry.create_group(f'bank{self.bank_B_id}_events')
            bB.create_dataset('event_id', data=np.array([2000, 2001]))
            bB.create_dataset('event_index', data=np.array([0, 1]))
            bB.create_dataset('event_time_offset', data=np.array([40.0, 50.0]))
            bB.create_dataset('event_time_zero', data=np.array([3.0, 4.0]))

    def tearDown(self):
        self.test_dir.cleanup()

    def test_non_contiguous_bank_assignment(self):
        print(f"\n{'='*60}\nExecuting Regression: NON-CONTIGUOUS BANK ASSIGNMENT\n{'='*60}")
        print(f"Testing {self.instrument} with Bank {self.bank_A_id} and Bank {self.bank_B_id}")
        
        # Load the mock file through the actual EventStreamLoader
        # Load the mock file through the actual EventStreamLoader
        loader = EventStreamLoader(
            event_nexus_filename=self.nexus_file,
            instrument_name=self.instrument,
            ki_vec=np.array([0.0, 0.0, 1.0]),
            sample_offset=np.zeros((1, 3))
        )
        
        # Total events: 3 from Bank A + 2 from Bank B = 5
        self.assertEqual(loader.total_events, 5, "Total events loaded does not match the generated mock data.")
        
        # Count the occurrences of each bank_id mapped to the events
        unique_banks, counts = np.unique(loader.all_banks, return_counts=True)
        bank_counts = dict(zip(unique_banks, counts))
        
        # Assert Bank A was mapped perfectly
        self.assertIn(self.bank_A_id, bank_counts, f"Bank {self.bank_A_id} events were lost.")
        self.assertEqual(bank_counts[self.bank_A_id], 3, f"Bank {self.bank_A_id} should have exactly 3 events.")
        
        # Assert Bank B was mapped perfectly
        self.assertIn(self.bank_B_id, bank_counts, f"Bank {self.bank_B_id} events were lost.")
        self.assertEqual(bank_counts[self.bank_B_id], 2, f"Bank {self.bank_B_id} should have exactly 2 events.")
        
        # Ensure the regex didn't invent or pull in phantom banks
        self.assertEqual(len(unique_banks), 2, f"Found unexpected bank IDs in the loaded data: {unique_banks}")
        
        print(f"  -> Success: Mapped {loader.total_events} events perfectly to banks {self.bank_A_id} and {self.bank_B_id}.")

    def test_sample_frame_transformation(self):
        print(f"\n{'='*60}\nExecuting Regression: SAMPLE FRAME TRANSFORMATION\n{'='*60}")
        
        self.instrument = "MANDI"
        available_banks = list(beamlines.get(self.instrument, {}).keys())
        if not available_banks:
            self.skipTest("No banks found for MANDI.")
        bank_id = int(available_banks[0])
        
        with h5py.File(self.nexus_file, 'w') as f:
            entry = f.create_group('entry')
            
            # 1. Inject a single event at absolute time = 5.0 seconds
            bA = entry.create_group(f'bank{bank_id}_events')
            bA.create_dataset('event_id', data=np.array([1000]))
            bA.create_dataset('event_index', data=np.array([0]))
            bA.create_dataset('event_time_offset', data=np.array([0.0])) 
            bA.create_dataset('event_time_zero', data=np.array([5.0]))
            
            # 2. Inject a mocked Goniometer Log: Constant 90 degrees on the Y-axis (omega)
            omega = entry.create_group('DASlogs/omega')
            omega.create_dataset('time', data=np.array([0.0, 10.0]))
            omega.create_dataset('value', data=np.array([90.0, 90.0]))
            
        loader = EventStreamLoader(
            event_nexus_filename=self.nexus_file,
            instrument_name=self.instrument,
            ki_vec=np.array([0.0, 0.0, 1.0]),         # Beam travels down Lab Z
            sample_offset=np.zeros((1, 3)),
            gonio_axes=np.array([[0.0, 1.0, 0.0]]),   # Omega rotates around Lab Y
            gonio_names=['omega']
        )
        
        batches = list(loader.get_batches(10000))
        self.assertEqual(len(batches), 1, "Loader failed to yield the event batch.")
        
        # Unpack the tuple exactly as the Bingham Tracker expects it
        q_sample, times, banks, pr, pc, angles, slab, ki_sample, count = batches[0]
        
        # 1. Verify Time Interpolation
        self.assertAlmostEqual(
            angles[0, 0], 90.0, 
            msg="Goniometer angle was not correctly interpolated from the DASlogs."
        )
        
        # 2. Verify Frame Kinematics
        # At 90 degrees around +Y, the mapping from Lab -> Sample is a -90 deg rotation around Y.
        # Lab Z (0, 0, 1) should physically map to Sample -X (-1, 0, 0)
        expected_ki_sample = np.array([-1.0, 0.0, 0.0])
        
        np.testing.assert_allclose(
            ki_sample[0], 
            expected_ki_sample, 
            atol=1e-6, 
            err_msg="ki_sample did not properly rotate from the Lab frame into the Sample frame."
        )
        
        # 3. Verify q_sample norm consistency 
        # (q is a directional momentum vector, its length must remain invariant under rotation)
        q_norm_sample = np.linalg.norm(q_sample[0])
        self.assertAlmostEqual(
            q_norm_sample, 1.0, places=5,
            msg="The q_sample momentum vector was corrupted during the frame transformation."
        )
        
        print("  -> Success: EventStreamLoader correctly interpolated logs and mapped vectors into the rotating Sample Frame.")

if __name__ == '__main__':
    unittest.main()
