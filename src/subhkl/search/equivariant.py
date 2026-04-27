import jax
import jax.numpy as jnp
import flax.linen as nn

class EGNNLayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, h, edge_indices):
        # x: Equivariant 3D coordinates -> Shape (N, 3)
        # h: Invariant node features -> Shape (N, hidden_dim)
        # edge_indices: (2, num_edges)
        
        senders, receivers = edge_indices

        # 1. Compute relative vectors (Target Node - Source Node)
        # x_diff is EQUIVARIANT. sq_dist is INVARIANT.
        x_diff = x[receivers] - x[senders]
        sq_dist = jnp.sum(x_diff ** 2, axis=-1, keepdims=True)

        # 2. Invariant Message Passing (The MLP only sees scalars!)
        m_input = jnp.concatenate([h[receivers], h[senders], sq_dist], axis=-1)
        m_ij = nn.Dense(self.hidden_dim)(m_input)
        m_ij = nn.silu(m_ij)
        m_ij = nn.Dense(self.hidden_dim)(m_ij)

        # 3. Equivariant Coordinate Update (The Magic)
        coord_weight = nn.Dense(1, use_bias=False)(m_ij)
        coord_update = x_diff * coord_weight
        
        # THE FIX: segment_sum reduces the 210 edge messages down to 15 node updates.
        # Since the shape is now (N, 3), we add it directly to x!
        x_new = x + jax.ops.segment_sum(coord_update, receivers, x.shape[0])

        # 4. Invariant Feature Update
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
        
        # Initialize node features
        h = nn.Dense(self.hidden_dim)(intensities[:, None])
        x = q_vectors
        
        # ---------------------------------------------------------
        # THE FIX: Use standard NumPy to build the static graph.
        # This forces the shape to become concrete during JIT tracing.
        # ---------------------------------------------------------
        import numpy as np
        idx = np.arange(N)
        senders, receivers = np.meshgrid(idx, idx)
        edge_indices_np = np.stack([senders.flatten(), receivers.flatten()])
        
        # NumPy resolves the boolean mask instantly
        mask = edge_indices_np[0] != edge_indices_np[1]
        
        # Wrap the concrete, static-shaped array back into JAX!
        edge_indices = jnp.array(edge_indices_np[:, mask])
        # ---------------------------------------------------------

        # Pass through EGNN layers
        for _ in range(self.num_layers):
            x, h = EGNNLayer(self.hidden_dim)(x, h, edge_indices)

        # --- THE SO(3) OUTPUT POOLING ---
        h_global = jnp.mean(h, axis=0)
        
        frame_weights = nn.Dense(2)(h_global) 
        
        v1 = jnp.sum(x * frame_weights[0], axis=0)
        v2 = jnp.sum(x * frame_weights[1], axis=0)
        
        # Gram-Schmidt Orthogonalization
        v1 = v1 / jnp.linalg.norm(v1)
        v2 = v2 - jnp.dot(v1, v2) * v1
        v2 = v2 / jnp.linalg.norm(v2)
        v3 = jnp.cross(v1, v2)
        
        U_pred = jnp.stack([v1, v2, v3], axis=1)
        return U_pred
