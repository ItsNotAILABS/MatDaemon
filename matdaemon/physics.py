"""Physics algorithm registry and matmul-backed reference implementations.

Physics simulation is, computationally, repeated dense linear algebra:
many-body interactions are pairwise sums that reduce to a Gram matrix
(``X @ X.T``), PDE time-stepping is a stencil operator applied as a
matrix-vector product, and quantum/spectral problems are eigenproblems of a
Hamiltonian matrix. That is exactly MatDaemon's compute surface.

This module ships two things:

1. ``PHYSICS_ALGORITHMS`` -- a registry of real physics formulas, each with its
   closed-form equation, the matrix form it maps to, and the role matmul plays.
   AIs and coding agents can enumerate these (like ``use_cases``) to pick the
   right primitive for a simulation, ML-for-science feature, or agent tool.
2. Verified NumPy/matmul reference implementations for the subset that reduces
   cleanly to linear algebra. Every implementation is covered by a correctness
   test against a hand-computed or analytic value.

The core acceleration trick, used across the many-body entries::

    ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 (X @ X.T)_ij

so an O(N^2) pairwise sum becomes one matmul plus two norm broadcasts.

Safety boundary is unchanged: pure NumPy, no network, no file reads, no shell.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .matdaemon import matmul

# ---------------------------------------------------------------------------
# Core linear-algebra primitives shared by many entries
# ---------------------------------------------------------------------------


def pairwise_squared_distances(points: Sequence[Sequence[float]], backend: str = "auto") -> np.ndarray:
    """Squared Euclidean distance matrix via the Gram-matrix identity.

    ``D2_ij = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j`` -- the ``x_i . x_j`` term is
    a single ``X @ X.T`` matmul, which is why this scales to large point clouds.
    """
    X = np.asarray(points, dtype=np.float64)
    gram = matmul(X, X.T, backend=backend)
    sq_norms = (X * X).sum(axis=1)
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    return np.maximum(d2, 0.0)  # clamp tiny negative values from float error


def pairwise_distances(points: Sequence[Sequence[float]], backend: str = "auto") -> np.ndarray:
    """Euclidean distance matrix (sqrt of :func:`pairwise_squared_distances`)."""
    return np.sqrt(pairwise_squared_distances(points, backend=backend))


def laplacian_1d(n: int, dx: float = 1.0) -> np.ndarray:
    """1D discrete Laplacian (second-difference stencil ``[1, -2, 1] / dx^2``)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    L = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n)
    L[idx, idx] = -2.0
    L[idx[:-1], idx[:-1] + 1] = 1.0
    L[idx[1:], idx[1:] - 1] = 1.0
    return L / (dx * dx)


def _upper_pair_sum(matrix: np.ndarray) -> float:
    """Sum of the strict upper triangle -- i.e. each unordered pair once."""
    return float(np.triu(matrix, k=1).sum())


# ---------------------------------------------------------------------------
# Many-body interactions (Gram-trick -> matmul)
# ---------------------------------------------------------------------------


def rbf_kernel(points: Sequence[Sequence[float]], gamma: float = 1.0, backend: str = "auto") -> np.ndarray:
    """Gaussian radial basis kernel ``K_ij = exp(-gamma ||x_i - x_j||^2)``."""
    return np.exp(-gamma * pairwise_squared_distances(points, backend=backend))


def coulomb_matrix(atomic_numbers: Sequence[float], positions: Sequence[Sequence[float]]) -> np.ndarray:
    """Coulomb matrix molecular descriptor (Rupp et al., PRL 2012).

    ``M_ii = 0.5 Z_i^2.4``; ``M_ij = Z_i Z_j / ||R_i - R_j||`` for ``i != j``.
    A standard ML-for-chemistry feature: an invariant matrix encoding of a
    molecule built entirely from nuclear charges and interatomic distances.
    """
    Z = np.asarray(atomic_numbers, dtype=np.float64)
    dist = pairwise_distances(positions)
    with np.errstate(divide="ignore"):
        inv = np.where(dist > 0, 1.0 / dist, 0.0)
    M = np.outer(Z, Z) * inv
    np.fill_diagonal(M, 0.5 * Z ** 2.4)
    return M


def gravitational_potential_energy(
    masses: Sequence[float], positions: Sequence[Sequence[float]], g_const: float = 6.674e-11
) -> float:
    """Total Newtonian gravitational PE ``U = -G sum_{i<j} m_i m_j / r_ij``."""
    m = np.asarray(masses, dtype=np.float64)
    dist = pairwise_distances(positions)
    with np.errstate(divide="ignore"):
        inv = np.where(dist > 0, 1.0 / dist, 0.0)
    return -g_const * _upper_pair_sum(np.outer(m, m) * inv)


def coulomb_potential_energy(
    charges: Sequence[float], positions: Sequence[Sequence[float]], k_e: float = 8.9875e9
) -> float:
    """Total electrostatic PE ``U = k_e sum_{i<j} q_i q_j / r_ij``."""
    q = np.asarray(charges, dtype=np.float64)
    dist = pairwise_distances(positions)
    with np.errstate(divide="ignore"):
        inv = np.where(dist > 0, 1.0 / dist, 0.0)
    return k_e * _upper_pair_sum(np.outer(q, q) * inv)


def lennard_jones_energy(
    positions: Sequence[Sequence[float]], epsilon: float = 1.0, sigma: float = 1.0
) -> float:
    """Total Lennard-Jones PE ``sum_{i<j} 4 eps [(sig/r)^12 - (sig/r)^6]``."""
    dist = pairwise_distances(positions)
    with np.errstate(divide="ignore"):
        inv = np.where(dist > 0, sigma / dist, 0.0)
    pair = 4.0 * epsilon * (inv ** 12 - inv ** 6)
    return _upper_pair_sum(pair)


def structure_factor(
    positions: Sequence[Sequence[float]], q_vectors: Sequence[Sequence[float]], backend: str = "auto"
) -> np.ndarray:
    """Static structure factor ``S(q) = |sum_j exp(i q . r_j)|^2`` (diffraction).

    The phase argument ``q . r`` is a ``Q @ R.T`` matmul over all (q, atom) pairs.
    """
    R = np.asarray(positions, dtype=np.float64)
    Q = np.asarray(q_vectors, dtype=np.float64)
    phase = matmul(Q, R.T, backend=backend)  # (n_q, n_atoms)
    amplitude = np.exp(1j * phase).sum(axis=1)
    return np.abs(amplitude) ** 2


# ---------------------------------------------------------------------------
# Statistical mechanics
# ---------------------------------------------------------------------------


def boltzmann_distribution(energies: Sequence[float], kt: float = 1.0) -> np.ndarray:
    """Boltzmann occupation ``p_i = exp(-E_i/kT) / Z`` (numerically stable)."""
    E = np.asarray(energies, dtype=np.float64)
    if kt <= 0:
        raise ValueError("kt must be > 0")
    shifted = -(E - E.min()) / kt
    weights = np.exp(shifted)
    return weights / weights.sum()


def partition_function(energies: Sequence[float], kt: float = 1.0) -> float:
    """Canonical partition function ``Z = sum_i exp(-E_i/kT)``."""
    E = np.asarray(energies, dtype=np.float64)
    if kt <= 0:
        raise ValueError("kt must be > 0")
    return float(np.exp(-E / kt).sum())


def ising_energy(
    spins: Sequence[float], adjacency: Sequence[Sequence[float]], coupling: float = 1.0, field: float = 0.0
) -> float:
    """Ising Hamiltonian ``E = -J/2 s^T A s - h sum_i s_i``.

    The interaction term is a quadratic form ``s^T A s`` (two matmuls), summing
    each bond once via the 1/2 factor on a symmetric adjacency matrix.
    """
    s = np.asarray(spins, dtype=np.float64)
    A = np.asarray(adjacency, dtype=np.float64)
    interaction = -0.5 * coupling * float(s @ matmul(A, s.reshape(-1, 1)).ravel())
    return interaction - field * float(s.sum())


def transfer_matrix_ising_1d(coupling: float, field: float, kt: float) -> np.ndarray:
    """Transfer matrix for the 1D Ising chain; ``Z = Tr(T^N)``.

    ``T = [[e^{b(J+h)}, e^{-bJ}], [e^{-bJ}, e^{b(J-h)}]]`` with ``b = 1/kT``.
    """
    if kt <= 0:
        raise ValueError("kt must be > 0")
    b = 1.0 / kt
    return np.array(
        [
            [np.exp(b * (coupling + field)), np.exp(-b * coupling)],
            [np.exp(-b * coupling), np.exp(b * (coupling - field))],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# PDE / field time-stepping (stencil operator -> matmul)
# ---------------------------------------------------------------------------


def heat_equation_step(field: Sequence[float], alpha: float, dt: float, dx: float = 1.0) -> np.ndarray:
    """One explicit (FTCS) heat-equation step ``u <- u + alpha*dt * L u``."""
    u = np.asarray(field, dtype=np.float64)
    L = laplacian_1d(len(u), dx)
    return u + alpha * dt * matmul(L, u.reshape(-1, 1)).ravel()


def wave_equation_step(
    field_now: Sequence[float], field_prev: Sequence[float], speed: float, dt: float, dx: float = 1.0
) -> np.ndarray:
    """One explicit wave-equation step ``u+ = 2u - u- + (c*dt)^2 L u``."""
    u = np.asarray(field_now, dtype=np.float64)
    u_prev = np.asarray(field_prev, dtype=np.float64)
    L = laplacian_1d(len(u), dx)
    return 2.0 * u - u_prev + (speed * dt) ** 2 * matmul(L, u.reshape(-1, 1)).ravel()


def markov_diffusion_step(probabilities: Sequence[float], transition: Sequence[Sequence[float]]) -> np.ndarray:
    """One discrete diffusion / Markov step ``p_{n+1} = P p_n`` (column-stochastic P)."""
    p = np.asarray(probabilities, dtype=np.float64)
    P = np.asarray(transition, dtype=np.float64)
    return matmul(P, p.reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------------
# Quantum / spectral
# ---------------------------------------------------------------------------


def hamiltonian_1d(
    potential: Sequence[float], dx: float = 1.0, hbar: float = 1.0, mass: float = 1.0
) -> np.ndarray:
    """Discrete 1D time-independent Hamiltonian ``H = -(hbar^2/2m) L + diag(V)``.

    Eigenvalues of H are the energy levels; solve ``H psi = E psi``.
    """
    V = np.asarray(potential, dtype=np.float64)
    kinetic = -(hbar ** 2) / (2.0 * mass) * laplacian_1d(len(V), dx)
    return kinetic + np.diag(V)


def density_matrix_expectation(
    density: Sequence[Sequence[float]], observable: Sequence[Sequence[float]]
) -> float:
    """Quantum expectation value ``<A> = Tr(rho A)`` (two matmuls + trace)."""
    rho = np.asarray(density, dtype=np.float64)
    A = np.asarray(observable, dtype=np.float64)
    return float(np.trace(matmul(rho, A)))


# ---------------------------------------------------------------------------
# Classical dynamics (state-space / rotation -> matmul)
# ---------------------------------------------------------------------------


def rotation_matrix_3d(axis: Sequence[float], angle: float) -> np.ndarray:
    """3D rotation matrix about ``axis`` by ``angle`` radians (Rodrigues formula)."""
    a = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm == 0:
        raise ValueError("axis must be non-zero")
    a = a / norm
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * matmul(K, K)


def rotate_points(points: Sequence[Sequence[float]], rotation: Sequence[Sequence[float]], backend: str = "auto") -> np.ndarray:
    """Apply a rotation (or any linear map) to a point cloud: ``X R^T``."""
    X = np.asarray(points, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    return matmul(X, R.T, backend=backend)


def stress_tensor_rotation(stress: Sequence[Sequence[float]], rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Rotate a rank-2 tensor into a new frame: ``sigma' = R sigma R^T``."""
    sigma = np.asarray(stress, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    return matmul(matmul(R, sigma), R.T)


def state_space_step(
    state: Sequence[float],
    system: Sequence[Sequence[float]],
    control_matrix: Optional[Sequence[Sequence[float]]] = None,
    control: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Linear dynamical step ``s_{n+1} = A s_n + B u`` (kinematics, control, RL)."""
    s = np.asarray(state, dtype=np.float64)
    A = np.asarray(system, dtype=np.float64)
    nxt = matmul(A, s.reshape(-1, 1)).ravel()
    if control_matrix is not None and control is not None:
        B = np.asarray(control_matrix, dtype=np.float64)
        u = np.asarray(control, dtype=np.float64)
        nxt = nxt + matmul(B, u.reshape(-1, 1)).ravel()
    return nxt


def verlet_step(pos_now: Sequence[float], pos_prev: Sequence[float], accel: Sequence[float], dt: float) -> np.ndarray:
    """One position-Verlet MD step ``x_{n+1} = 2 x_n - x_{n-1} + a dt^2``."""
    x = np.asarray(pos_now, dtype=np.float64)
    x_prev = np.asarray(pos_prev, dtype=np.float64)
    a = np.asarray(accel, dtype=np.float64)
    return 2.0 * x - x_prev + a * dt * dt


def kalman_predict(
    state: Sequence[float],
    covariance: Sequence[Sequence[float]],
    transition: Sequence[Sequence[float]],
    process_noise: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman predict step ``x = F x``, ``P = F P F^T + Q`` (state estimation)."""
    x = np.asarray(state, dtype=np.float64)
    P = np.asarray(covariance, dtype=np.float64)
    F = np.asarray(transition, dtype=np.float64)
    Q = np.asarray(process_noise, dtype=np.float64)
    x_pred = matmul(F, x.reshape(-1, 1)).ravel()
    P_pred = matmul(matmul(F, P), F.T) + Q
    return x_pred, P_pred


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PHYSICS_ALGORITHMS: list[dict] = [
    {"id": "pairwise-distances", "name": "Pairwise Euclidean distances", "domain": "many-body / geometry",
     "formula": "D2_ij = ||x_i - x_j||^2", "matrix_form": "||x_i||^2 + ||x_j||^2 - 2 (X @ X.T)_ij",
     "matmul_role": "Gram matrix X @ X.T turns the O(N^2) pairwise sum into one matmul",
     "recommended_backend": "numpy", "function": "pairwise_squared_distances",
     "reference": "Standard kernel/N-body distance identity"},
    {"id": "nbody-gravity", "name": "Newtonian gravitational potential energy", "domain": "classical gravity",
     "formula": "U = -G sum_{i<j} m_i m_j / r_ij", "matrix_form": "outer(m,m) * (1/D) summed over upper triangle",
     "matmul_role": "distance matrix D from the Gram trick", "recommended_backend": "numpy",
     "function": "gravitational_potential_energy", "reference": "Newton's law of universal gravitation"},
    {"id": "coulomb-energy", "name": "Electrostatic (Coulomb) potential energy", "domain": "electromagnetism",
     "formula": "U = k_e sum_{i<j} q_i q_j / r_ij", "matrix_form": "outer(q,q) * (1/D) upper triangle",
     "matmul_role": "distance matrix from the Gram trick", "recommended_backend": "numpy",
     "function": "coulomb_potential_energy", "reference": "Coulomb's law"},
    {"id": "lennard-jones", "name": "Lennard-Jones potential", "domain": "molecular dynamics",
     "formula": "V(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6]", "matrix_form": "elementwise on the distance matrix",
     "matmul_role": "pairwise distances via Gram trick", "recommended_backend": "numpy",
     "function": "lennard_jones_energy", "reference": "Lennard-Jones (1924)"},
    {"id": "coulomb-matrix", "name": "Coulomb matrix descriptor", "domain": "ML for chemistry",
     "formula": "M_ii = 0.5 Z_i^2.4 ; M_ij = Z_i Z_j / r_ij", "matrix_form": "outer(Z,Z) / D with diagonal override",
     "matmul_role": "distance matrix from atomic coordinates", "recommended_backend": "numpy",
     "function": "coulomb_matrix", "reference": "Rupp, Tkatchenko, Muller, von Lilienfeld (PRL 2012)"},
    {"id": "rbf-kernel", "name": "Gaussian RBF kernel", "domain": "kernel methods / GPs",
     "formula": "K_ij = exp(-gamma ||x_i - x_j||^2)", "matrix_form": "exp(-gamma * D2)",
     "matmul_role": "squared-distance matrix via Gram trick", "recommended_backend": "numpy",
     "function": "rbf_kernel", "reference": "Radial basis function kernel"},
    {"id": "structure-factor", "name": "Static structure factor", "domain": "condensed matter / diffraction",
     "formula": "S(q) = |sum_j exp(i q.r_j)|^2", "matrix_form": "phases = Q @ R.T then |sum exp(i*phase)|^2",
     "matmul_role": "Q @ R.T computes all q.r phase products", "recommended_backend": "numpy",
     "function": "structure_factor", "reference": "Scattering / X-ray diffraction theory"},
    {"id": "boltzmann-distribution", "name": "Boltzmann distribution", "domain": "statistical mechanics",
     "formula": "p_i = exp(-E_i/kT) / Z", "matrix_form": "softmax over -E/kT",
     "matmul_role": "vectorized exponential normalization", "recommended_backend": "numpy",
     "function": "boltzmann_distribution", "reference": "Boltzmann / Gibbs canonical ensemble"},
    {"id": "partition-function", "name": "Canonical partition function", "domain": "statistical mechanics",
     "formula": "Z = sum_i exp(-E_i/kT)", "matrix_form": "reduction of the Boltzmann weight vector",
     "matmul_role": "vector reduction", "recommended_backend": "numpy",
     "function": "partition_function", "reference": "Canonical ensemble"},
    {"id": "ising-energy", "name": "Ising model Hamiltonian", "domain": "statistical mechanics / spin systems",
     "formula": "E = -J sum_<ij> s_i s_j - h sum_i s_i", "matrix_form": "-J/2 s^T A s - h (1.s)",
     "matmul_role": "quadratic form s^T A s is two matmuls", "recommended_backend": "numpy",
     "function": "ising_energy", "reference": "Ising (1925)"},
    {"id": "transfer-matrix", "name": "1D Ising transfer matrix", "domain": "statistical mechanics",
     "formula": "Z = Tr(T^N)", "matrix_form": "matrix power of the 2x2 transfer matrix",
     "matmul_role": "repeated matmul (matrix power)", "recommended_backend": "numpy",
     "function": "transfer_matrix_ising_1d", "reference": "Kramers-Wannier transfer matrix (1941)"},
    {"id": "heat-equation", "name": "Heat equation (explicit FTCS)", "domain": "PDE / thermal",
     "formula": "du/dt = alpha d2u/dx2", "matrix_form": "u <- u + alpha*dt * L u, L = discrete Laplacian",
     "matmul_role": "Laplacian stencil applied as matrix-vector product", "recommended_backend": "tiled",
     "function": "heat_equation_step", "reference": "Forward-time centered-space finite difference"},
    {"id": "wave-equation", "name": "Wave equation (explicit FD)", "domain": "PDE / acoustics",
     "formula": "d2u/dt2 = c^2 d2u/dx2", "matrix_form": "u+ = 2u - u- + (c dt)^2 L u",
     "matmul_role": "Laplacian stencil as matmul", "recommended_backend": "tiled",
     "function": "wave_equation_step", "reference": "Leapfrog finite-difference scheme"},
    {"id": "diffusion-markov", "name": "Discrete diffusion / Markov step", "domain": "stochastic processes",
     "formula": "p_{n+1} = P p_n", "matrix_form": "transition matrix times state vector",
     "matmul_role": "matrix-vector product per step", "recommended_backend": "numpy",
     "function": "markov_diffusion_step", "reference": "Master equation / Markov chain"},
    {"id": "poisson-equation", "name": "Poisson equation", "domain": "electrostatics / gravity",
     "formula": "laplacian(phi) = -rho/eps0", "matrix_form": "solve L phi = -rho/eps0",
     "matmul_role": "Laplacian operator; solve via linear system or iterative matmul", "recommended_backend": "tiled",
     "function": None, "reference": "Poisson's equation"},
    {"id": "schrodinger-1d", "name": "Time-independent Schrodinger (1D)", "domain": "quantum mechanics",
     "formula": "H psi = E psi, H = -(hbar^2/2m) d2/dx2 + V", "matrix_form": "H = -(hbar^2/2m) L + diag(V); eigenproblem",
     "matmul_role": "build H; power iteration / eig uses matmul", "recommended_backend": "tiled",
     "function": "hamiltonian_1d", "reference": "Schrodinger equation, finite-difference discretization"},
    {"id": "tight-binding", "name": "Tight-binding band structure", "domain": "condensed matter",
     "formula": "H_ij = eps_i delta_ij - t A_ij ; E = eig(H)", "matrix_form": "eigenvalues of the hopping Hamiltonian",
     "matmul_role": "Hamiltonian assembly and diagonalization", "recommended_backend": "tiled",
     "function": None, "reference": "Tight-binding / LCAO model"},
    {"id": "density-matrix", "name": "Density-matrix expectation value", "domain": "quantum mechanics",
     "formula": "<A> = Tr(rho A)", "matrix_form": "trace of rho @ A",
     "matmul_role": "rho @ A is a matmul", "recommended_backend": "numpy",
     "function": "density_matrix_expectation", "reference": "Density-operator formalism"},
    {"id": "rotation-3d", "name": "Rigid-body 3D rotation", "domain": "classical mechanics",
     "formula": "x' = R x, R in SO(3)", "matrix_form": "point cloud X R^T",
     "matmul_role": "rotation applied as matmul over all points", "recommended_backend": "numpy",
     "function": "rotate_points", "reference": "Rodrigues' rotation formula"},
    {"id": "stress-tensor-rotation", "name": "Stress/tensor frame rotation", "domain": "continuum mechanics",
     "formula": "sigma' = R sigma R^T", "matrix_form": "two matmuls R @ sigma @ R^T",
     "matmul_role": "double contraction as chained matmul", "recommended_backend": "numpy",
     "function": "stress_tensor_rotation", "reference": "Cauchy stress transformation"},
    {"id": "state-space-step", "name": "Linear state-space step", "domain": "dynamics / control",
     "formula": "s_{n+1} = A s_n + B u", "matrix_form": "A s + B u",
     "matmul_role": "matrix-vector products per timestep", "recommended_backend": "numpy",
     "function": "state_space_step", "reference": "Linear time-invariant state-space model"},
    {"id": "verlet-integration", "name": "Position Verlet integration", "domain": "molecular dynamics",
     "formula": "x_{n+1} = 2 x_n - x_{n-1} + a dt^2", "matrix_form": "vectorized over all particles/coordinates",
     "matmul_role": "batched update; forces come from the pairwise matmul stack", "recommended_backend": "numpy",
     "function": "verlet_step", "reference": "Verlet (1967)"},
    {"id": "kalman-predict", "name": "Kalman filter predict step", "domain": "state estimation",
     "formula": "x = F x ; P = F P F^T + Q", "matrix_form": "F x and F P F^T",
     "matmul_role": "covariance propagation is chained matmul", "recommended_backend": "numpy",
     "function": "kalman_predict", "reference": "Kalman (1960)"},
    {"id": "normal-modes", "name": "Small-oscillation normal modes", "domain": "classical mechanics",
     "formula": "(K - omega^2 M) v = 0", "matrix_form": "generalized eigenproblem of M^{-1} K",
     "matmul_role": "mass-weighted stiffness assembly + diagonalization", "recommended_backend": "tiled",
     "function": None, "reference": "Normal-mode analysis"},
]


def get_physics_algorithm(algorithm_id: str) -> Optional[dict]:
    for algorithm in PHYSICS_ALGORITHMS:
        if algorithm["id"] == algorithm_id:
            return algorithm
    return None
