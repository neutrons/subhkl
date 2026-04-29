import jax
import jax.numpy as jnp
import numpy as np
import e3nn_jax as e3nn
import optax
from scipy.spatial.transform import Rotation as R
from ott.geometry import pointcloud, costs
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# Import your actual network architecture
from subhkl.search.equivariant import build_laue_graph, E3NN_Indexer

def generate_synthetic_crystal(max_hkl=2):
    """Generates a perfect Simple Cubic theoretical dictionary."""
    hc_vals = np.arange(-max_hkl, max_hkl + 1)
    hc, kc, lc = np.meshgrid(hc_vals, hc_vals, hc_vals, indexing="ij")
    hkl_c = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
    
    mask = ~((hkl_c[0] == 0) & (hkl_c[1] == 0) & (hkl_c[2] == 0))
    theo_hkl = hkl_c[:, mask].astype(np.float32).T 
    
    # B_mat for Simple Cubic (Identity)
    B_mat = np.eye(3)
    r_theo = (B_mat @ theo_hkl.T).T
    
    # Normalize to unit sphere
    z_theo = r_theo / np.linalg.norm(r_theo, axis=1, keepdims=True)
    return jnp.array(z_theo)

def test_strict_equivariance():
    print("\n--- RUNNING EQUIVARIANCE TEST ---")
    
    # 1. Setup Data
    z_theo = generate_synthetic_crystal()
    N = z_theo.shape[0]
    intensities = jnp.ones(N)
    
    # 2. Init Model
    model = E3NN_Indexer(num_layers=2)
    rng = jax.random.PRNGKey(42)
    graph_base = build_laue_graph(z_theo, k_neighbors=10)
    params = model.init(rng, graph_base, intensities)
    
    # 3. Baseline Prediction
    U_base = model.apply(params, graph_base, intensities)
    
    # 4. Rotate Crystal Arbitrarily
    rot_matrix = jnp.array(R.from_euler('xyz', [45, -30, 75], degrees=True).as_matrix())
    z_rotated = jnp.matmul(rot_matrix, z_theo.T).T
    graph_rot = build_laue_graph(z_rotated, k_neighbors=10)
    
    # 5. Prediction on Rotated Crystal
    U_rot = model.apply(params, graph_rot, intensities)
    
    # 6. Verify Mathematical Equivariance (U_rot MUST equal rot_matrix @ U_base)
    U_expected = jnp.matmul(rot_matrix, U_base)
    
    error = jnp.max(jnp.abs(U_rot - U_expected))
    print(f"Max Equivariance Error: {error:.2e}")
    if error < 1e-5:
        print("✅ SUCCESS: The Neural Network is perfectly SO(3) Equivariant!")
    else:
        print("❌ FAILURE: The geometry is broken. Check e3nn layers.")

def test_sinkhorn_convergence():
    print("\n--- RUNNING CONVERGENCE TEST ---")
    
    # 1. Setup Perfect Data
    z_theo = generate_synthetic_crystal(max_hkl=2)
    
    # Create an empirical point cloud rotated by a known amount
    true_rot = jnp.array(R.from_euler('xyz', [15, 25, -10], degrees=True).as_matrix())
    q_empirical = jnp.matmul(true_rot, z_theo.T).T
    
    N = q_empirical.shape[0]
    intensities = jnp.ones(N)
    graph = build_laue_graph(q_empirical, k_neighbors=10)
    
    # 2. Setup Model & Optimizer
    model = E3NN_Indexer(num_layers=2)
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, graph, intensities)
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    
    # 3. Sinkhorn Loss Loop
    @jax.jit
    def train_step(params, opt_state, epsilon):
        def loss_fn(p):
            U_pred = model.apply(p, graph, intensities)
            empirical_rays_rotated = jnp.matmul(U_pred, q_empirical.T).T
            
            geom = pointcloud.PointCloud(
                x=empirical_rays_rotated, y=z_theo,
                cost_fn=costs.Cosine(), epsilon=epsilon
            )
            prob = linear_problem.LinearProblem(geom, a=jnp.ones(N)/N, b=jnp.ones(N)/N)
            return sinkhorn.Sinkhorn()(prob).reg_ot_cost, U_pred
            
        (loss, U_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, U_pred

    print("Starting optimization on perfect synthetic crystal...")
    epsilon = 0.05
    for i in range(150):
        # Anneal epsilon
        if i > 50: epsilon = 0.01
        if i > 100: epsilon = 0.001
        
        params, opt_state, loss, U_pred = train_step(params, opt_state, epsilon)
        if (i+1) % 50 == 0:
            print(f"  Step {i+1:03d}/150 | Transport Cost: {loss:.4f}")
            
    # Check if U_pred perfectly inverts true_rot
    # In a perfect match, U_pred @ true_rot = Identity (or a symmetry equivalent)
    final_alignment = jnp.matmul(U_pred, true_rot)
    trace = jnp.trace(final_alignment)
    angle_error = jnp.arccos(jnp.clip((trace - 1) / 2, -1.0, 1.0)) * (180 / jnp.pi)
    
    print(f"\nFinal Angular Error from Ground Truth: {angle_error:.2f} degrees")
    if angle_error < 1.0 or abs(angle_error - 90.0) < 1.0 or abs(angle_error - 180.0) < 1.0:
        print("✅ SUCCESS: Sinkhorn perfectly recovered the crystal frame (or a valid symmetry alias)!")
    else:
        print("❌ FAILURE: Optimization got stuck.")

if __name__ == "__main__":
    test_strict_equivariance()
    test_sinkhorn_convergence()
