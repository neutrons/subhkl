import jax
import jax.numpy as jnp
import flax.linen as nn

class EGNNLayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, h, edge_indices):
        # x: Equivariant 3D coordinates (Zone Axes) -> Shape (N, 3)
        # h: Invariant node features (e.g., peak intensities) -> Shape (N, hidden_dim)
        # edge_indices: (2, num_edges) defining the fully connected graph
        
        senders, receivers = edge_indices

        # 1. Compute relative vectors and invariant distances
        # x_diff is EQUIVARIANT. sq_dist is INVARIANT.
        x_diff = x[senders] - x[receivers]
        sq_dist = jnp.sum(x_diff ** 2, axis=-1, keepdims=True)

        # 2. Invariant Message Passing (The MLP only sees scalars!)
        # Concat: Sender feature, Receiver feature, and the Invariant Distance
        m_input = jnp.concatenate([h[senders], h[receivers], sq_dist], axis=-1)
        m_ij = nn.Dense(self.hidden_dim)(m_input)
        m_ij = nn.silu(m_ij)
        m_ij = nn.Dense(self.hidden_dim)(m_ij)

        # 3. Equivariant Coordinate Update (The Magic)
        # We compute a scalar weight for the vector x_diff. 
        # A scalar multiplied by a vector remains a perfectly equivariant vector!
        coord_weight = nn.Dense(1, use_bias=False)(m_ij)
        coord_update = x_diff * coord_weight
        
        # Sum the vector messages to update the 3D node positions
        x_new = x.at[senders].add(jax.ops.segment_sum(coord_update, senders, x.shape[0]))

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
        
        # Initialize node features (h) with intensities, mapped to hidden dim
        h = nn.Dense(self.hidden_dim)(intensities[:, None])
        x = q_vectors
        
        # Build a fully connected graph for the zone axes
        idx = jnp.arange(N)
        senders, receivers = jnp.meshgrid(idx, idx)
        edge_indices = jnp.stack([senders.flatten(), receivers.flatten()])
        
        # Remove self-loops
        mask = edge_indices[0] != edge_indices[1]
        edge_indices = edge_indices[:, mask]

        # Pass through EGNN layers
        for _ in range(self.num_layers):
            x, h = EGNNLayer(self.hidden_dim)(x, h, edge_indices)

        # --- THE SO(3) OUTPUT POOLING ---
        # Pool the invariant features to get a global graph embedding
        h_global = jnp.mean(h, axis=0)
        
        # Use the global embedding to weight the final 3D coordinates, 
        # projecting the point cloud down to exactly 2 orthogonal L=1 vectors.
        # (2 vectors perfectly define a 3D rotation frame).
        frame_weights = nn.Dense(2)(h_global) 
        
        v1 = jnp.sum(x * frame_weights[0], axis=0)
        v2 = jnp.sum(x * frame_weights[1], axis=0)
        
        # Gram-Schmidt Orthogonalization to ensure a pure Rotation Matrix
        v1 = v1 / jnp.linalg.norm(v1)
        v2 = v2 - jnp.dot(v1, v2) * v1
        v2 = v2 / jnp.linalg.norm(v2)
        v3 = jnp.cross(v1, v2)
        
        U_pred = jnp.stack([v1, v2, v3], axis=1)
        return U_pred
