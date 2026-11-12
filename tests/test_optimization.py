import numpy as np
import scipy.linalg

from subhkl import optimization


def test_get_lattice_system_simple_cubic():
    final, num, centering = optimization.get_lattice_system(
        10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P 4 3 2"
    )
    assert final == "Cubic"
    assert num == 1
    assert centering == "P"


def test_findub_load_from_dict_and_reciprocal_B():
    data = {}
    data["sample/a"] = 10.0
    data["sample/b"] = 10.0
    data["sample/c"] = 10.0
    data["sample/alpha"] = 90.0
    data["sample/beta"] = 90.0
    data["sample/gamma"] = 90.0
    data["instrument/wavelength"] = np.array([1.0, 2.0])
    data["goniometer/R"] = np.eye(3)
    data["peaks/two_theta"] = np.array([30.0])
    data["peaks/azimuthal"] = np.array([10.0])
    data["peaks/intensity"] = np.array([1.0])
    data["peaks/sigma"] = np.array([0.1])
    data["peaks/radius"] = np.array([0.0])
    data["sample/space_group"] = "P 1"

    fu = optimization.FindUB(data=data)
    B = fu.reciprocal_lattice_B()
    assert B.shape == (3, 3)

    # Recompute expected B via metric tensor
    a = data["sample/a"]
    b = data["sample/b"]
    c = data["sample/c"]
    alpha = np.deg2rad(data["sample/alpha"])
    beta = np.deg2rad(data["sample/beta"])
    gamma = np.deg2rad(data["sample/gamma"])

    g11 = a**2
    g22 = b**2
    g33 = c**2
    g12 = a * b * np.cos(gamma)
    g13 = c * a * np.cos(beta)
    g23 = b * c * np.cos(alpha)
    G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])
    B_expected = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    assert np.allclose(B, B_expected)


def test_findub_minimize_wraps_rodrigues_into_principal_ball():
    rng = np.random.default_rng(7)
    w_true = np.array([0.6, -1.1, 0.8], dtype=np.float64)
    theta = np.linalg.norm(w_true)
    # Equivalent rotation 3 full turns outside the principal ball (||w|| > 20 rad)
    w_unwrapped = w_true * (1.0 + 6.0 * np.pi / theta)
    U_true = np.array(optimization.rotation_matrix_from_rodrigues_jax(w_true))

    a, b, c = 10.0, 20.0, 8.0
    B = np.diag([1.0 / a, 1.0 / b, 1.0 / c])
    hkls = np.array(
        [[1, 2, 1], [2, -3, 1], [-1, 4, 2], [3, 1, -2], [1, -5, 1], [2, 3, -1]],
        dtype=np.float64,
    ).T
    q_sample = U_true @ (B @ hkls)
    ki_hat = np.array([0.0, 0.0, 1.0])
    # Choose wavelengths in [1.5, 3.5] satisfying Bragg condition |q| = -2 (ki_hat . q_hat) / wl
    # Perturb orientations slightly so peaks scatter into detector directions
    kf_ki_dirs = []
    two_thetas = []
    azimuthals = []
    for j in range(hkls.shape[1]):
        q_vec = q_sample[:, j]
        if q_vec[2] >= -0.05:
            q_vec = -q_vec
        wl = -2.0 * q_vec[2] / np.dot(q_vec, q_vec)
        kf_hat = ki_hat + wl * q_vec
        d = kf_hat - ki_hat
        kf_ki_dirs.append(d)
        two_thetas.append(np.rad2deg(np.arccos(np.clip(kf_hat[2], -1.0, 1.0))))
        azimuthals.append(np.rad2deg(np.arctan2(kf_hat[1], kf_hat[0])))

    data = {
        "sample/a": a,
        "sample/b": b,
        "sample/c": c,
        "sample/alpha": 90.0,
        "sample/beta": 90.0,
        "sample/gamma": 90.0,
        "instrument/wavelength": np.array([0.5, 10.0]),
        "goniometer/R": np.eye(3),
        "peaks/two_theta": np.array(two_thetas),
        "peaks/azimuthal": np.array(azimuthals),
        "peaks/intensity": np.ones(len(two_thetas)),
        "peaks/sigma": np.full(len(two_thetas), 0.1),
        "peaks/radius": np.zeros(len(two_thetas)),
        "sample/space_group": "P 1",
    }
    fu = optimization.FindUB(data=data)
    num_indexed, _, _, U_est = fu.minimize(
        strategy_name="de",
        population_size=32,
        num_generations=10,
        n_runs=2,
        seed=42,
        init_params=w_unwrapped + rng.normal(scale=0.02, size=3),
        sigma_init=0.02,
    )
    assert np.linalg.norm(fu.x[:3]) <= np.pi + 1e-5
    assert num_indexed == len(two_thetas)
    assert np.allclose(U_est, U_true, atol=1e-2)
