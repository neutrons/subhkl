import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import jraph
import e3nn_jax as e3nn

def build_laue_graph(q_vectors_norm, k_neighbors=20):
    """
    Constructs a strict SO(3) k-NN graph. 
    Crucially, edge features are the absolute positions of the sender nodes,
    locking the entire graph physically to the Ewald sphere origin.
    """
    N = q_vectors_norm.shape[0]
    
    # 1. Compute distances for k-NN
    x_np = np.array(q_vectors_norm)
    diff = x_np[:, None, :] - x_np[None, :, :]
    dist = np.sum(diff**2, axis=-1)
    
    # 2. Extract k-nearest neighbors
    k = min(k_neighbors, N - 1)
    idx = np.argsort(dist, axis=1)[:, 1:k+1] 
    
    senders = np.repeat(np.arange(N), k)
    receivers = idx.flatten()
    
    # 3. Build pristine jraph object
    return jraph.GraphsTuple(
        nodes=jnp.array(q_vectors_norm),           # (N, 3) 
        edges=jnp.array(q_vectors_norm[senders]),  # (E, 3) Absolute sender positions
        receivers=jnp.array(receivers),
        senders=jnp.array(senders),
        n_node=jnp.array([N]),
        n_edge=jnp.array([len(senders)]),
        globals=None
    )



class E3NN_Indexer(nn.Module):
    # The hidden state lives in an abstract space of Scalars (0e), Vectors (1o), and Tensors (2e)
    hidden_irreps: e3nn.Irreps = e3nn.Irreps("16x0e + 16x1o + 16x2e")
    num_layers: int = 3

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, intensities: jnp.ndarray):
        # 1. Lift spatial coordinates into Spherical Harmonics
        node_vecs = e3nn.IrrepsArray("1o", graph.nodes)
        edge_vecs = e3nn.IrrepsArray("1o", graph.edges)
        
        node_sh = e3nn.spherical_harmonics("0e + 1o + 2e", node_vecs, normalize=True)
        edge_sh = e3nn.spherical_harmonics("0e + 1o + 2e", edge_vecs, normalize=True)
        
        h_init = e3nn.IrrepsArray("1x0e", intensities[:, None])
        
        # FIX 1: Use e3nn.flax.Linear
        nodes = e3nn.flax.Linear(self.hidden_irreps)(e3nn.tensor_product(h_init, node_sh))

        # 2. Define the e3nn Message Passing Layer
        def update_edge_fn(edges, senders, receivers, globals_):
            tp = e3nn.tensor_product(senders, edge_sh)
            # FIX 2: Use e3nn.flax.Linear
            return e3nn.flax.Linear(self.hidden_irreps)(tp)

        def update_node_fn(nodes, senders, receivers, globals_):
            tp = e3nn.tensor_product(nodes, receivers)
            # FIX 3: Use e3nn.flax.Linear
            return e3nn.flax.Linear(self.hidden_irreps)(tp)

        # 3. Execute Graph Network
        gn = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn
        )
        
        for _ in range(self.num_layers):
            graph = graph._replace(nodes=nodes)
            graph = gn(graph)
            nodes = graph.nodes

        # 4. Invariant Readout (0e)
        # FIX 4: Use e3nn.flax.Linear
        invariant_scalars = e3nn.flax.Linear("1x0e")(nodes).array
        weights = jax.nn.softplus(invariant_scalars) + 1e-4

        # 5. Manifest Covariance Extraction
        x = graph.nodes 
        
        M = jnp.dot(x.T, x * weights)
        M = M + jnp.diag(jnp.array([1e-5, 2e-5, 3e-5])) 
        
        U, S, Vh = jnp.linalg.svd(M)
        
        det = jnp.linalg.det(U)
        U_pred = jnp.where(det < 0, U.at[:, 2].multiply(-1), U)
        
        return U_pred
