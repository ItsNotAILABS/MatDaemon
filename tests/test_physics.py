import json

import numpy as np

from matdaemon.mcp_server import handle_request
from matdaemon.physics import (
    PHYSICS_ALGORITHMS,
    boltzmann_distribution,
    coulomb_matrix,
    coulomb_potential_energy,
    density_matrix_expectation,
    get_physics_algorithm,
    gravitational_potential_energy,
    heat_equation_step,
    ising_energy,
    kalman_predict,
    laplacian_1d,
    lennard_jones_energy,
    markov_diffusion_step,
    pairwise_distances,
    pairwise_squared_distances,
    partition_function,
    rbf_kernel,
    rotate_points,
    rotation_matrix_3d,
    state_space_step,
    stress_tensor_rotation,
    structure_factor,
    transfer_matrix_ising_1d,
    verlet_step,
    wave_equation_step,
)


def _tool_payload(response):
    return json.loads(response["result"]["content"][0]["text"])


# --- registry -----------------------------------------------------------------

def test_registry_has_at_least_twenty_algorithms():
    assert len(PHYSICS_ALGORITHMS) >= 20
    for entry in PHYSICS_ALGORITHMS:
        assert {"id", "name", "domain", "formula", "matrix_form", "matmul_role"} <= set(entry)


def test_get_physics_algorithm_by_id():
    assert get_physics_algorithm("ising-energy")["domain"].startswith("statistical")
    assert get_physics_algorithm("does-not-exist") is None


def test_runnable_entries_point_at_real_callables():
    import matdaemon.physics as physics
    for entry in PHYSICS_ALGORITHMS:
        fn = entry.get("function")
        if fn is not None:
            assert callable(getattr(physics, fn)), fn


# --- many-body (Gram trick) ---------------------------------------------------

def test_pairwise_distances_against_hand_computed():
    pts = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
    d = pairwise_distances(pts)
    assert np.isclose(d[0, 1], 3.0)
    assert np.isclose(d[0, 2], 4.0)
    assert np.isclose(d[1, 2], 5.0)  # 3-4-5 triangle
    assert np.allclose(np.diag(d), 0.0)


def test_pairwise_squared_distances_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 4))
    d2 = pairwise_squared_distances(X)
    brute = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    assert np.allclose(d2, brute, atol=1e-8)


def test_rbf_kernel_diagonal_is_one_and_symmetric():
    K = rbf_kernel([[0.0], [1.0], [2.0]], gamma=0.5)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)


def test_coulomb_matrix_diagonal_and_offdiagonal():
    Z = [1.0, 1.0]
    R = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    M = coulomb_matrix(Z, R)
    assert np.isclose(M[0, 0], 0.5 * 1.0 ** 2.4)
    assert np.isclose(M[0, 1], 1.0)  # Z1*Z2 / r = 1*1/1


def test_gravitational_energy_two_body():
    # U = -G m1 m2 / r ; unit masses, unit distance, G=1 -> -1
    u = gravitational_potential_energy([1.0, 1.0], [[0.0], [1.0]], g_const=1.0)
    assert np.isclose(u, -1.0)


def test_coulomb_energy_two_like_charges_is_positive():
    u = coulomb_potential_energy([1.0, 1.0], [[0.0], [1.0]], k_e=1.0)
    assert np.isclose(u, 1.0)


def test_lennard_jones_minimum_depth():
    # V(r) = 4eps[(s/r)^12 - (s/r)^6]; at r = 2^(1/6) sigma the pair energy = -eps
    r_min = 2.0 ** (1.0 / 6.0)
    e = lennard_jones_energy([[0.0], [r_min]], epsilon=1.0, sigma=1.0)
    assert np.isclose(e, -1.0, atol=1e-9)


def test_structure_factor_forward_scattering_equals_n_squared():
    # At q = 0, all phases are 1, so S(0) = |N|^2
    positions = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    s = structure_factor(positions, [[0.0, 0.0]])
    assert np.isclose(s[0], 9.0)  # N=3 -> 9


# --- statistical mechanics ----------------------------------------------------

def test_boltzmann_sums_to_one_and_orders_correctly():
    p = boltzmann_distribution([0.0, 1.0, 2.0], kt=1.0)
    assert np.isclose(p.sum(), 1.0)
    assert p[0] > p[1] > p[2]


def test_boltzmann_two_level_analytic():
    # p1/p0 = exp(-(E1-E0)/kT)
    p = boltzmann_distribution([0.0, 1.0], kt=1.0)
    assert np.isclose(p[1] / p[0], np.exp(-1.0))


def test_partition_function_value():
    z = partition_function([0.0, 0.0], kt=1.0)
    assert np.isclose(z, 2.0)  # two zero-energy states


def test_ising_energy_aligned_lower_than_antialigned():
    A = [[0.0, 1.0], [1.0, 0.0]]
    aligned = ising_energy([1.0, 1.0], A, coupling=1.0, field=0.0)
    anti = ising_energy([1.0, -1.0], A, coupling=1.0, field=0.0)
    assert np.isclose(aligned, -1.0)
    assert np.isclose(anti, 1.0)
    assert aligned < anti


def test_transfer_matrix_shape_and_positivity():
    T = transfer_matrix_ising_1d(coupling=1.0, field=0.0, kt=1.0)
    assert T.shape == (2, 2)
    assert np.all(T > 0)


# --- PDE stepping -------------------------------------------------------------

def test_laplacian_stencil():
    L = laplacian_1d(3, dx=1.0)
    assert np.allclose(L[1], [1.0, -2.0, 1.0])


def test_heat_step_smooths_a_spike():
    u = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    u1 = heat_equation_step(u, alpha=1.0, dt=0.1, dx=1.0)
    assert u1[2] < u[2]  # peak decays
    assert u1[1] > u[1] and u1[3] > u[3]  # spreads to neighbours


def test_wave_step_runs_and_preserves_shape():
    u = np.array([0.0, 1.0, 0.0])
    u1 = wave_equation_step(u, u, speed=1.0, dt=0.1, dx=1.0)
    assert u1.shape == (3,)


def test_markov_step_conserves_probability():
    P = np.array([[0.9, 0.2], [0.1, 0.8]])  # column-stochastic
    p1 = markov_diffusion_step([1.0, 0.0], P)
    assert np.isclose(p1.sum(), 1.0)
    assert np.allclose(p1, [0.9, 0.1])


# --- quantum / dynamics -------------------------------------------------------

def test_density_matrix_expectation_trace():
    rho = [[0.5, 0.0], [0.0, 0.5]]  # maximally mixed
    Z = [[1.0, 0.0], [0.0, -1.0]]   # Pauli-Z
    assert np.isclose(density_matrix_expectation(rho, Z), 0.0)


def test_rotation_90_degrees_about_z():
    R = rotation_matrix_3d([0, 0, 1], np.pi / 2)
    out = rotate_points([[1.0, 0.0, 0.0]], R)
    assert np.allclose(out[0], [0.0, 1.0, 0.0], atol=1e-9)


def test_rotation_is_orthonormal():
    R = rotation_matrix_3d([1, 1, 0], 0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_stress_tensor_rotation_preserves_trace():
    sigma = [[2.0, 0.0], [0.0, 1.0]]
    theta = 0.6
    R = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    rotated = stress_tensor_rotation(sigma, R)
    assert np.isclose(np.trace(rotated), 3.0)  # trace invariant


def test_state_space_step_with_control():
    A = [[1.0, 1.0], [0.0, 1.0]]  # constant-velocity kinematics
    s = state_space_step([0.0, 2.0], A)
    assert np.allclose(s, [2.0, 2.0])  # position advances by velocity
    s2 = state_space_step([0.0, 0.0], A, control_matrix=[[0.0], [1.0]], control=[3.0])
    assert np.allclose(s2, [0.0, 3.0])


def test_verlet_constant_velocity():
    # zero acceleration -> next = 2*now - prev keeps constant spacing
    nxt = verlet_step([1.0], [0.0], [0.0], dt=1.0)
    assert np.allclose(nxt, [2.0])


def test_kalman_predict_shapes_and_covariance_growth():
    x, P = kalman_predict([0.0, 1.0], np.eye(2), [[1.0, 1.0], [0.0, 1.0]], np.eye(2) * 0.1)
    assert x.shape == (2,) and P.shape == (2, 2)
    assert P[0, 0] > 1.0  # uncertainty grows through the predict step


# --- MCP tool surface ---------------------------------------------------------

def test_mcp_physics_algorithms_list():
    resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "matdaemon_physics_algorithms", "arguments": {}}})
    payload = _tool_payload(resp)
    assert payload["count"] >= 20


def test_mcp_pairwise_distances_tool():
    resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                           "params": {"name": "matdaemon_pairwise_distances",
                                      "arguments": {"points": [[0, 0], [3, 4]]}}})
    payload = _tool_payload(resp)
    assert np.isclose(payload["distances"][0][1], 5.0)


def test_mcp_ising_and_boltzmann_tools():
    r1 = handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                         "params": {"name": "matdaemon_ising_energy",
                                    "arguments": {"spins": [1, 1], "adjacency": [[0, 1], [1, 0]]}}})
    assert np.isclose(_tool_payload(r1)["energy"], -1.0)
    r2 = handle_request({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                         "params": {"name": "matdaemon_boltzmann_distribution",
                                    "arguments": {"energies": [0, 0], "kt": 1.0}}})
    assert np.allclose(_tool_payload(r2)["probabilities"], [0.5, 0.5])
