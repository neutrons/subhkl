import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

class SO3_EGNNLayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, h, edge_indices):
        senders, receivers = edge_indices

        # ==========================================
        # 1. The SO(3) Invariant Scalars
        # ==========================================
        # Instead of relative Euclidean distance, we compute the pure inner product space.
        # This anchors the physics to the Direct Beam (Origin) preserving d-spacing magnitudes!
        norm_sq_i = jnp.sum(x[receivers] ** 2, axis=-1, keepdims=True)
        norm_sq_j = jnp.sum(x[senders] ** 2, axis=-1, keepdims=True)
        dot_ij = jnp.sum(x[receivers] * x[senders], axis=-1, keepdims=True)

        # 2. Invariant Message Passing
        m_input = jnp.concatenate([
            h[receivers], 
            h[senders], 
            norm_sq_i, 
            norm_sq_j, 
            dot_ij
        ], axis=-1)
        
        m_ij = nn.Dense(self.hidden_dim)(m_input)
        m_ij = nn.silu(m_ij)
        m_ij = nn.Dense(self.hidden_dim)(m_ij)

        # ==========================================
        # 3. The SO(3) Equivariant Coordinate Update
        # ==========================================
        # Absolute positions are valid equivariant vectors in SO(3). 
        # We predict independent scalars to weight x_i and x_j directly.
        weight_i = nn.Dense(1, use_bias=False)(m_ij)
        weight_j = nn.Dense(1, use_bias=False)(m_ij)
        
        # Apply tanh to prevent exploding spatial coordinates
        weight_i = jnp.tanh(weight_i) * 0.1
        weight_j = jnp.tanh(weight_j) * 0.1
        
        # The physical update: Node A is pulled by Node B, but also scales itself
        coord_update = (x[receivers] * weight_i) + (x[senders] * weight_j)
        
        # Sum the vector messages
        x_new = x + jax.ops.segment_sum(coord_update, receivers, x.shape[0])
        
        # Note: We intentionally DO NOT project back to the unit sphere here. 
        # In reciprocal space, |q| represents the scattering resolution (1/d). 
        # We must allow the vectors to stretch and breathe to retain their physics.

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
    k_neighbors: int = 20

    @nn.compact
    def __call__(self, q_vectors, intensities):
        N = q_vectors.shape[0]
        
        # Initialize node features
        h = nn.Dense(self.hidden_dim)(intensities[:, None])
        x = q_vectors
        
        # ---------------------------------------------------------
        # K-Nearest Neighbors Graph (Static Trace via NumPy)
        # ---------------------------------------------------------
        x_np = np.array(q_vectors)
        k = min(self.k_neighbors, N - 1)
        
        diff = x_np[:, None, :] - x_np[None, :, :]
        dist = np.sum(diff**2, axis=-1)
        
        idx = np.argsort(dist, axis=1)[:, 1:k+1] 
        senders = np.repeat(np.arange(N), k)
        receivers = idx.flatten()
        
        edge_indices = jnp.array(np.stack([senders, receivers]))

        # Pass through SO(3) EGNN layers
        for _ in range(self.num_layers):
            x, h = SO3_EGNNLayer(self.hidden_dim)(x, h, edge_indices)

        # ==========================================
        # THE FIX: Manifest Covariance Extraction
        # ==========================================
        # Predict strictly positive attention weights for every Bragg peak
        weights = jax.nn.softplus(nn.Dense(1)(h)) + 1e-4 
        
        # Construct the 3x3 Weighted Covariance Tensor (Outer Product).
        # Squaring the vectors inherently solves the Friedel pair (+q, -q) cancellation.
        # x is (N, 3), x.T is (3, N). The result is strictly a 3x3 orientation matrix.
        M = jnp.dot(x.T, x * weights)
        
        # Break eigenvalue degeneracy with micro-noise to guarantee stable JAX SVD gradients
        M = M + jnp.diag(jnp.array([1e-5, 2e-5, 3e-5]))
        
        # Extract Principal Axes mathematically
        U, S, Vh = jnp.linalg.svd(M)
        
        # SVD inherently returns orthogonal unit vectors. 
        # We must verify the handedness (determinant) to ensure it is a valid SO(3) spatial rotation, 
        # rather than a mirror reflection.
        det = jnp.linalg.det(U)
        U_pred = jnp.where(det < 0, U.at[:, 2].multiply(-1), U)
        
        return U_pred
