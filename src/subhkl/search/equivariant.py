import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import jraph
import e3nn_jax as e3nn

def build_laue_graph(q_vectors_norm, k_neighbors=20):
    # (Remains exactly the same as before)
    N = q_vectors_norm.shape[0]
    x_np = np.array(q_vectors_norm)
    diff = x_np[:, None, :] - x_np[None, :, :]
    dist = np.sum(diff**2, axis=-1)
    k = min(k_neighbors, N - 1)
    idx = np.argsort(dist, axis=1)[:, 1:k+1] 
    senders = np.repeat(np.arange(N), k)
    receivers = idx.flatten()
    
    return jraph.GraphsTuple(
        nodes=jnp.array(q_vectors_norm),           
        edges=jnp.array(q_vectors_norm[senders]),  
        receivers=jnp.array(receivers),
        senders=jnp.array(senders),
        n_node=jnp.array([N]),
        n_edge=jnp.array([len(senders)]),
        globals=None
    )

class E3NN_Indexer(nn.Module):
    # We still use a mix of scalars, vectors, and tensors in the hidden layers
    hidden_irreps: e3nn.Irreps = e3nn.Irreps("16x0e + 16x1o + 16x2e")
    num_layers: int = 3

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, intensities: jnp.ndarray):
        # 1. Lift spatial coordinates
        node_vecs = e3nn.IrrepsArray("1o", graph.nodes)
        edge_vecs = e3nn.IrrepsArray("1o", graph.edges)
        
        node_sh = e3nn.spherical_harmonics("0e + 1o + 2e", node_vecs, normalize=True)
        edge_sh = e3nn.spherical_harmonics("0e + 1o + 2e", edge_vecs, normalize=True)
        
        h_init = e3nn.IrrepsArray("1x0e", intensities[:, None])
        nodes = e3nn.flax.Linear(self.hidden_irreps)(e3nn.tensor_product(h_init, node_sh))

        # 2. Message Passing
        def update_edge_fn(edges, senders, receivers, globals_):
            tp = e3nn.tensor_product(senders, edge_sh)
            return e3nn.flax.Linear(self.hidden_irreps)(tp)

        def update_node_fn(nodes, senders, receivers, globals_):
            tp = e3nn.tensor_product(nodes, receivers)
            return e3nn.flax.Linear(self.hidden_irreps)(tp)

        gn = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn
        )
        
        for _ in range(self.num_layers):
            graph = graph._replace(nodes=nodes)
            graph = gn(graph)
            nodes = graph.nodes

        # ==========================================
        # THE FIX: Direct Equivariant Frame Regression
        # ==========================================
        # Instead of 1x0e scalars for SVD, we ask the network to output two 3D vectors!
        # Because of e3nn math, these vectors will rotate perfectly with the input graph.
        node_vectors = e3nn.flax.Linear("2x1o")(nodes).array  # Shape: (N, 2, 3)

        # Global Average Pooling: The whole crystal votes on the lattice orientation
        global_vectors = jnp.mean(node_vectors, axis=0)       # Shape: (2, 3)
        
        v1 = global_vectors[0]
        v2 = global_vectors[1]

        # Gram-Schmidt Orthogonalization to guarantee a rigid 3x3 SO(3) matrix
        b1 = v1 / (jnp.linalg.norm(v1) + 1e-6)
        b2 = v2 - jnp.dot(b1, v2) * b1
        b2 = b2 / (jnp.linalg.norm(b2) + 1e-6)
        b3 = jnp.cross(b1, b2)

        U_pred = jnp.stack([b1, b2, b3], axis=-1)
        
        return U_pred
