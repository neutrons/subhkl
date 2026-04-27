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

        norms = jnp.maximum(jnp.linalg.norm(x_new, axis=-1, keepdims=True), 1e-8)
        x_new = x_new / norms

        m_i = jax.ops.segment_sum(m_ij, receivers, h.shape[0])
        h_input = jnp.concatenate([h, m_i], axis=-1)
        h_new = nn.Dense(self.hidden_dim)(h_input)
        h_new = nn.silu(h_new)
        h_new = nn.Dense(self.hidden_dim)(h_new) + h

        return x_new, h_new

class EquivariantIndexer(nn.Module):
    hidden_dim: int = 64
    num_layers: int = 3
    k_neighbors: int = 20  # NEW: Enforce local geometric message passing

    @nn.compact
    def __call__(self, q_vectors, intensities):
        N = q_vectors.shape[0]
        
        h = nn.Dense(self.hidden_dim)(intensities[:, None])
        x = q_vectors
        
        # ---------------------------------------------------------
        # K-Nearest Neighbors Graph (Static Trace via NumPy)
        # ---------------------------------------------------------
        x_np = np.array(q_vectors)
        k = min(self.k_neighbors, N - 1)
        
        # Compute pairwise distance matrix
        diff = x_np[:, None, :] - x_np[None, :, :]
        dist = np.sum(diff**2, axis=-1)
        
        # Extract top k nearest neighbors for each node
        idx = np.argsort(dist, axis=1)[:, 1:k+1] 
        senders = np.repeat(np.arange(N), k)
        receivers = idx.flatten()
        
        edge_indices = jnp.array(np.stack([senders, receivers]))
        # ---------------------------------------------------------

        for _ in range(self.num_layers):
            x, h = EGNNLayer(self.hidden_dim)(x, h, edge_indices)

        h_global = jnp.mean(h, axis=0)
        frame_weights = nn.Dense(2)(h_global) 
        
        v1 = jnp.sum(x * frame_weights[0], axis=0)
        v2 = jnp.sum(x * frame_weights[1], axis=0)
        
        v1 = v1 / jnp.maximum(jnp.linalg.norm(v1), 1e-8)
        v2 = v2 - jnp.dot(v1, v2) * v1
        v2 = v2 / jnp.maximum(jnp.linalg.norm(v2), 1e-8)
        v3 = jnp.cross(v1, v2)
        
        U_pred = jnp.stack([v1, v2, v3], axis=1)
        return U_pred
