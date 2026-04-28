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
        
        # STRICT SPHERICAL PROJECTION: Prevent points from shrinking
        x_new = x_new + jnp.array([1e-6, 0.0, 0.0]) # Micro-drift prevents dead zero
        x_norms = jnp.sqrt(jnp.maximum(jnp.sum(x_new**2, axis=-1, keepdims=True), 1e-12))
        x_new = x_new / x_norms

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

        node_weights = nn.Dense(2)(h) 
        
        v1 = jnp.sum(x * node_weights[:, 0:1], axis=0)
        v2 = jnp.sum(x * node_weights[:, 1:2], axis=0)
        
        # ==========================================
        # THE FIX: Prevent Trivial Zero Collapse
        # ==========================================
        # Inject an orthogonal seed. If the network outputs 0, U becomes Identity.
        v1 = v1 + jnp.array([1e-6, 0.0, 0.0])
        v2 = v2 + jnp.array([0.0, 1e-6, 0.0])
        
        # Strict Gram-Schmidt with hard minimum bounds
        v1_norm = jnp.sqrt(jnp.maximum(jnp.sum(v1**2), 1e-12))
        b1 = v1 / v1_norm
        
        v2_ortho = v2 - jnp.dot(v2, b1) * b1
        v2_norm = jnp.sqrt(jnp.maximum(jnp.sum(v2_ortho**2), 1e-12))
        b2 = v2_ortho / v2_norm
        
        b3 = jnp.cross(b1, b2)
        
        U_pred = jnp.stack([b1, b2, b3], axis=1)
        return U_pred
