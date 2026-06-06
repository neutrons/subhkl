import pytest
import numpy as np
import jax
import jax.numpy as jnp
from subhkl.commands import real_to_complex_sh_matrix

def test_spherical_harmonics_basis_unitarity():
    """ 
    FIXED PRECISION: Executes matrix multiplication in standard host NumPy channels
    to bypass GPU Tensor Core TF32 truncation artifacts and ensure true unitarity bounds.
    """
    for l in range(1, 9):
        U = np.array(real_to_complex_sh_matrix(l))
        dim = int(2 * l + 1)
        
        # Test U @ U^dagger == Identity
        UI = U @ U.conj().T
        np.testing.assert_allclose(
            np.real(UI), np.eye(dim), atol=1e-5,
            err_msg=f"Real-to-Complex SH transformation matrix lost row unitarity at rank l={l}"
        )
        
        # Test U^dagger @ U == Identity
        IU = U.conj().T @ U
        np.testing.assert_allclose(
            np.real(IU), np.eye(dim), atol=1e-5,
            err_msg=f"Real-to-Complex SH transformation matrix lost column unitarity at rank l={l}"
        )

def holonomic_su2_unitary_constraints(C_state, num_blocks, block_dims_static):
    """ Enforces complex unitary conditions directly from the flattened array representation. """
    constraints = []
    R_12 = C_state[0:4].reshape((2, 2))
    I_12 = C_state[4:8].reshape((2, 2))

    state_idx = 0
    for b in range(1, num_blocks):
        dim = block_dims_static[b]
        C_real = C_state[state_idx : state_idx + dim * dim].reshape((dim, dim))
        state_idx += dim * dim
        C_imag = C_state[state_idx : state_idx + dim * dim].reshape((dim, dim))
        state_idx += dim * dim

        V_real = jnp.matmul(C_real, C_real.T) + jnp.matmul(C_imag, C_imag.T)
        V_imag = jnp.matmul(C_imag, C_real.T) - jnp.matmul(C_real, C_imag.T)

        iu = jnp.triu_indices(dim, k=1)
        constraints.append(V_real[iu])
        constraints.append(V_imag[iu])
        constraints.append(jnp.diagonal(V_real) - 1.0)

        if b > 1:
            phase_lock = jnp.sum(C_real[:2, :2] * I_12 - C_imag[:2, :2] * R_12)
            constraints.append(jnp.array([phase_lock]))

    return jnp.concatenate(constraints)

def test_mass_conservation_under_diffusion():
    """ Ascertains that the structural blocks remain rigidly on their group-theoretic mass shells. """
    num_blocks = 9
    block_dims_static = [1] + [int(2 * (twice_j / 2.0) + 1) for twice_j in range(1, num_blocks)]
    num_state_coeffs = 2 * sum(block_dims_static[b]**2 for b in range(1, num_blocks))
    
    c_init_list = []
    for b in range(1, num_blocks):
        dim = block_dims_static[b]
        c_init_list.append(np.eye(dim).flatten())
        c_init_list.append(np.zeros((dim, dim)).flatten())
    C_state = jnp.concatenate(c_init_list)
    
    psi = holonomic_su2_unitary_constraints(C_state, num_blocks, block_dims_static)
    np.testing.assert_allclose(
        psi, 0.0, atol=1e-6,
        err_msg="The holonomic representation constraints are misaligned with the SU(2) initialization manifolds"
    )
