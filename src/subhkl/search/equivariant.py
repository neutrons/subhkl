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
        
        # FIX 1: Bound the spatial update magnitude to prevent violent explosions
        coord_weight = jnp.tanh(coord_weight) * 0.1
        coord_update = x_diff * coord_weight
        
        x_new = x + jax.ops.segment_sum(coord_update, receivers, x.shape[0])

        # FIX 2: Project coordinates back to the unit sphere (Directions only!)
        # The jnp.maximum prevents division by zero if a vector collapses.
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

    @nn.compact
    def __call__(self, q_vectors, intensities):
        N = q_vectors.shape[0]
        
        h = nn.Dense(self.hidden_dim)(intensities[:, None])
        x = q_vectors
        
        # Use standard NumPy to bypass the JIT dynamic shape error
        idx = np.arange(N)
        senders, receivers = np.meshgrid(idx, idx)
        edge_indices_np = np.stack([senders.flatten(), receivers.flatten()])
        
        mask = edge_indices_np[0] != edge_indices_np[1]
        edge_indices = jnp.array(edge_indices_np[:, mask])

        for _ in range(self.num_layers):
            x, h = EGNNLayer(self.hidden_dim)(x, h, edge_indices)

        h_global = jnp.mean(h, axis=0)
        frame_weights = nn.Dense(2)(h_global) 
        
        v1 = jnp.sum(x * frame_weights[0], axis=0)
        v2 = jnp.sum(x * frame_weights[1], axis=0)
        
        # FIX 3: Safe Gram-Schmidt Orthogonalization
        v1 = v1 / jnp.maximum(jnp.linalg.norm(v1), 1e-8)
        v2 = v2 - jnp.dot(v1, v2) * v1
        v2 = v2 / jnp.maximum(jnp.linalg.norm(v2), 1e-8)
        v3 = jnp.cross(v1, v2)
        
        U_pred = jnp.stack([v1, v2, v3], axis=1)
        return U_pred
