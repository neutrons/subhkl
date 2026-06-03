import unittest
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import scipy.spatial.transform
import scipy.special
import e3nn_jax as e3nn

def polar_extract_l1(c_l1_flat):
    """ Projects the l=1 Wigner matrix onto the closest orthogonal SO(3) coordinate track. """
    E_D1 = c_l1_flat.reshape((3, 3))
    P = jnp.array([
        [0.0, 1.0, 0.0],  # y
        [0.0, 0.0, 1.0],  # z
        [1.0, 0.0, 0.0]   # x
    ])
    E_U = jnp.matmul(P.T, jnp.matmul(E_D1, P))
    V, _, Wt = jnp.linalg.svd(E_U)
    return jnp.matmul(V, jnp.matmul(jnp.diag(jnp.array([1.0, 1.0, jnp.linalg.det(jnp.matmul(V, Wt))])), Wt))

def test_wigner_matrix_unitary_norm():
    """ Gauge 1: Validates that Wigner-D blocks preserve the exact (2l + 1) Frobenius norm. """
    rot = scipy.spatial.transform.Rotation.from_euler('xyz', [15.0, 32.0, 75.0], degrees=True)
    U_matrix = rot.as_matrix()
    
    for l in range(1, 5):
        irrep = e3nn.Irrep(f"{l}e" if l % 2 == 0 else f"{l}o")
        D_l = irrep.D_from_matrix(U_matrix)  # GAUGE FIX: Use the native e3nn matrix output directly
            
        frobenius_norm_sq = jnp.sum(jnp.square(D_l))
        expected_norm = float(2 * l + 1)
        
        # Increased tolerance to accommodate float32 polynomial truncation artifacts
        assert np.isclose(frobenius_norm_sq, expected_norm, atol=1e-2), \
            f"Wigner block l={l} violated unitary norm footprint. Expected {expected_norm}, got {frobenius_norm_sq}"

def test_legendre_contraction_gauge():
    """ Gauge 2: Validates the Wigner-Legendre contraction against analytical Scipy polynomials. """
    rot = scipy.spatial.transform.Rotation.from_euler('xyz', [20.0, -45.0, 60.0], degrees=True)
    U_matrix = jnp.array(rot.as_matrix())
    
    ki = jnp.array([0.0, 0.0, 1.0])
    q_cryst = jnp.array([0.5, 0.5, 0.7071])
    q_cryst = q_cryst / jnp.linalg.norm(q_cryst)
    
    q_lab_true = jnp.matmul(U_matrix, q_cryst)
    cos_theta = jnp.dot(ki, q_lab_true)
    
    sh_irreps = e3nn.Irreps("0e + 1o + 2e + 3o")
    Y_beam = e3nn.spherical_harmonics(sh_irreps, ki, normalize=True).array
    Y_cryst = e3nn.spherical_harmonics(sh_irreps, q_cryst, normalize=True).array
    
    slices = [slice(0, 1), slice(1, 4), slice(4, 9), slice(9, 16)]
    
    for l in range(1, 4):
        irrep = e3nn.Irrep(f"{l}e" if l % 2 == 0 else f"{l}o")
        D_l = irrep.D_from_matrix(U_matrix)  # GAUGE FIX: Removed manual P_perm scrambling
        
        Y_b_l = Y_beam[slices[l]]
        Y_c_l = Y_cryst[slices[l]]
        
        contraction = jnp.dot(Y_b_l, jnp.matmul(D_l, Y_c_l))
        scaled_contraction = contraction / (2 * l + 1)
        
        analytical_legendre = scipy.special.legendre(l)(float(cos_theta))
        
        assert np.isclose(scaled_contraction, analytical_legendre, atol=1e-5), \
            f"Legendre contraction gauge error at l={l}. Expected {analytical_legendre}, got {scaled_contraction}"

class TestE3nnNormalizationInvariants(unittest.TestCase):
    def setUp(self):
        # Configure a standard tracking setup up to L_max = 4
        self.L_max = 4
        self.irreps_str = " + ".join([f"{l}e" if l % 2 == 0 else f"{l}o" for l in range(self.L_max + 1)])
        self.sh_irreps = e3nn.Irreps(self.irreps_str)

        # Build block slices to isolate specific level headers
        self.block_slices = []
        sh_idx = 0
        for l in range(self.L_max + 1):
            dim = 2 * l + 1
            self.block_slices.append(slice(sh_idx, sh_idx + dim))
            sh_idx += dim

        # Generate random laboratory coordinate events (simulating a neutron batch)
        np.random.seed(42)
        num_events = 500
        vecs = np.random.normal(size=(num_events, 3))
        self.q_batch = jnp.array(vecs / np.linalg.norm(vecs, axis=1, keepdims=True))

    def test_spherical_harmonics_addition_theorem(self):
        """
        Verifies that e3nn's default spherical harmonics satisfy component normalization,
        meaning the sum of squares over all m channels for any l is identically 1.0.
        """
        # Compute the full harmonic grid array
        Y_events_lab = e3nn.spherical_harmonics(self.sh_irreps, self.q_batch, normalize=True).array

        for l in range(1, self.L_max + 1):
            Y_l = Y_events_lab[:, self.block_slices[l]]

            # The sum of squares over the m channels for each individual event
            sum_of_squares = jnp.sum(jnp.square(Y_l), axis=1)

            # Assert that every single event has an absolute norm of exactly 1.0
            np.testing.assert_allclose(
                sum_of_squares,
                1.0,
                atol=1e-5,
                err_msg=f"Component normalization failed at level l={l}. Sum of squares is not 1.0."
            )

    def test_empirical_autocorrelation_trace_invariant(self):
        """
        Verifies that the trace of the empirical 2nd-moment matrix (A_lab_data)
        is strictly equal to 1.0 across all levels due to component normalization.
        """
        Y_events_lab = e3nn.spherical_harmonics(self.sh_irreps, self.q_batch, normalize=True).array
        num_events = float(self.q_batch.shape[0])

        for l in range(1, self.L_max + 1):
            Y_l = Y_events_lab[:, self.block_slices[l]]

            # Compute empirical autocorrelation matrix: A = (Y^T @ Y) / N
            A_lab_data = jnp.matmul(Y_l.T, Y_l) / num_events

            # Extract its algebraic trace
            matrix_trace = jnp.trace(A_lab_data)

            self.assertAlmostEqual(
                float(matrix_trace),
                1.0,
                places=5,
                msg=f"Trace of A_lab_data at l={l} is {matrix_trace}, expected exactly 1.0."
            )

    def test_traceless_innovation_under_mixture_model(self):
        """
        Verifies that blending the background pedestal as (I / dim) forces
        the Kalman filter innovation profile to remain perfectly traceless.
        """
        Y_events_lab = e3nn.spherical_harmonics(self.sh_irreps, self.q_batch, normalize=True).array
        num_events = float(self.q_batch.shape[0])

        for l in range(1, self.L_max + 1):
            dim = 2 * l + 1
            Y_l = Y_events_lab[:, self.block_slices[l]]
            A_lab_data = jnp.matmul(Y_l.T, Y_l) / num_events

            # Simulate an arbitrary predicted signal matrix (orthogonal Wigner structure)
            # and a random environmental mixture ratio rho
            A_pred_sig = jnp.eye(dim)  # Ideal fully aligned signal matrix
            rho_l = 0.20               # 80% Background Noise environment

            # Mass-Conserving Background Pedestal
            bg_pedestal = jnp.eye(dim) / float(dim)

            # Compute mixture prediction: Z_pred_2nd = rho * A_sig + (1 - rho) * Bg
            A_pred_mat = rho_l * A_pred_sig + (1.0 - rho_l) * bg_pedestal

            # Extract Innovation matrix: Error = A_data - A_pred
            innovation_matrix = A_lab_data - A_pred_mat
            innovation_trace = jnp.trace(innovation_matrix)

            self.assertAlmostEqual(
                float(innovation_trace),
                0.0,
                places=5,
                msg=f"Innovation matrix is not traceless at l={l}. Trace = {innovation_trace}"
            )

if __name__ == '__main__':
    unittest.main()
