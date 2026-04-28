import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

class EGNNLayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, h, edge_indices):
        senders, receivers = edge_indices

        x_diff = x[receivers] - x[senders]
        sq_dist = jnp.sum(x_diff ** 2, axis=-1, keepdims=True)

        m_input = jnp.concatenate([h[receivers], h[senders], sq_dist], axis=-1)
        m_ij = nn.Dense(self.hidden_dim)(m_input)
        m_ij = nn.silu(m_ij)
        m_ij = nn.Dense(self.hidden_dim)(m_ij)

        coord_weight = nn.Dense(1, use_bias=False)(m_ij)
        coord_weight = jnp.tanh(coord_weight) * 0.1
        coord_update = x_diff * coord_weight
        
        x_new = x + jax.ops.segment_sum(coord_update, receivers, x.shape[0])
        
        # Stable spherical projection to prevent floating point overflow
        x_new = x_new / jnp.sqrt(jnp.sum(x_new**2, axis=-1, keepdims=True) + 1e-6)

        m_i = jax.ops.segment_sum(m_ij, receivers, h.shape[0])
        h_input = jnp.concatenate([h, m_i], axis=-1)
        h_new = nn.Dense(self.hidden_dim)(h_input)
        h_new = nn.silu(h_new)
        h_new = nn.Dense(self.hidden_dim)(h_new) + h

        return x_new, h_new

class EquivariantIndexer(nn.Module):
    hidden_dim: int = 64
    num_layers: int = 3
    k_neighbors: int = 20

    @nn.compact
    def __call__(self, q_vectors, intensities):
        N = q_vectors.shape[0]
        
        h = nn.Dense(self.hidden_dim)(intensities[:, None])
        x = q_vectors
        
        x_np = np.array(q_vectors)
        k = min(self.k_neighbors, N - 1)
        
        diff = x_np[:, None, :] - x_np[None, :, :]
        dist = np.sum(diff**2, axis=-1)
        
        idx = np.argsort(dist, axis=1)[:, 1:k+1] 
        senders = np.repeat(np.arange(N), k)
        receivers = idx.flatten()
        
        edge_indices = jnp.array(np.stack([senders, receivers]))

        for _ in range(self.num_layers):
            x, h = EGNNLayer(self.hidden_dim)(x, h, edge_indices)

        # ==========================================
        # THE FIX: Manifest Covariance Extraction
        # ==========================================
        # Predict strictly positive attention weights for the tensor
        weights = jax.nn.softplus(nn.Dense(1)(h)) + 1e-4 
        
        # Construct the 3x3 Weighted Covariance Matrix (Outer Product).
        # x is (N, 3), x.T is (3, N). The result is strictly a 3x3 matrix.
        M = jnp.dot(x.T, x * weights)
        
        # Break eigenvalue degeneracy with micro-noise to guarantee stable SVD gradients
        M = M + jnp.diag(jnp.array([1e-5, 2e-5, 3e-5]))
        
        # Extract Principal Axes mathematically
        U, S, Vh = jnp.linalg.svd(M)
        
        # SVD inherently returns orthogonal unit vectors. 
        # We just verify the handedness to ensure it is a valid SO(3) rotation.
        det = jnp.linalg.det(U)
        U_pred = jnp.where(det < 0, U.at[:, 2].multiply(-1), U)
        
        return U_pred
