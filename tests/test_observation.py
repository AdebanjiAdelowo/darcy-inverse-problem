import numpy as np

from src.forward import build_mesh, solve_forward
from src.observation import grid_sensors, random_sensors, build_observation_operator
from dolfinx.fem import Function


def test_grid_sensors_shape_and_bounds():
    s = grid_sensors(4, 3, margin=0.1)
    assert s.shape == (12, 2)
    assert np.all(s[:, 0] >= 0.1) and np.all(s[:, 0] <= 0.9)
    assert np.all(s[:, 1] >= 0.1) and np.all(s[:, 1] <= 0.9)


def test_random_sensors_reproducible_with_seed():
    s1 = random_sensors(10, seed=5)
    s2 = random_sensors(10, seed=5)
    s3 = random_sensors(10, seed=6)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_observation_weight_rows_sum_to_one():
    dm = build_mesh(20)
    sensors = grid_sensors(3, 3)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.03)
    assert np.allclose(H.weight_vectors.sum(axis=1), 1.0, atol=1e-8)


def test_observation_operator_matches_known_linear_field():
    dm = build_mesh(32)
    m = Function(dm.V_m)
    m.x.array[:] = 0.0
    p = solve_forward(dm, m)  # exact p = 1-x
    sensors = grid_sensors(3, 3)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.03)
    obs = H.apply(p)
    expected = 1.0 - sensors[:, 0]
    assert np.max(np.abs(obs - expected)) < 1e-3


def test_adjoint_rhs_vector_is_linear_combination_of_weights():
    dm = build_mesh(12)
    sensors = grid_sensors(2, 2)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.05)
    residual = np.array([1.0, -2.0, 0.5, 3.0])
    result = H.adjoint_rhs_vector(residual)
    expected = residual @ H.weight_vectors
    assert np.allclose(result, expected)
