"""Tests for the adjoint gradient -- the part of this repository that must
not be trusted without direct verification. See also
scripts/verify_gradient.py for the full multi-alpha Taylor study reported
in the README; these are fast, small-mesh regression versions of the same
check plus supporting unit tests."""
import numpy as np

from src.forward import build_mesh
from src.observation import grid_sensors, build_observation_operator
from src.adjoint import InverseProblem


def _small_problem(reg_type="h1", alpha=1e-2):
    dm = build_mesh(8)
    sensors = grid_sensors(3, 3)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.05)
    rng = np.random.default_rng(1)
    y = 0.5 + 0.05 * rng.standard_normal(H.n_sensors)
    sigma = np.full(H.n_sensors, 0.02)
    ip = InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=alpha, reg_type=reg_type)
    return ip, dm


def test_objective_decomposes_into_misfit_plus_regularization():
    ip, dm = _small_problem(alpha=0.5)
    m0 = np.zeros(dm.V_m.dofmap.index_map.size_global)
    J, g, p, lam = ip.objective_and_gradient(m0)

    pred = ip.obs.apply(p)
    misfit = 0.5 * np.sum(((pred - ip.y) / ip.sigma) ** 2)
    import ufl
    from dolfinx import fem
    reg = fem.assemble_scalar(fem.form(ip._regularizer_form(ip._m)))
    J_expected = misfit + 0.5 * ip.alpha * reg
    assert abs(J - J_expected) < 1e-10


def test_taylor_remainder_is_second_order_h1_regularizer():
    ip, dm = _small_problem(reg_type="h1", alpha=1e-2)
    rng = np.random.default_rng(0)
    m0 = 0.1 * rng.standard_normal(dm.V_m.dofmap.index_map.size_global)
    direction = rng.standard_normal(m0.shape)
    direction /= np.linalg.norm(direction)

    J0, g0, _, _ = ip.objective_and_gradient(m0)
    dd = float(g0 @ direction)

    eps1, eps2 = 1e-2, 5e-3
    J1, _, _, _ = ip.objective_and_gradient(m0 + eps1 * direction)
    J2, _, _, _ = ip.objective_and_gradient(m0 + eps2 * direction)
    r1 = abs(J1 - J0 - eps1 * dd)
    r2 = abs(J2 - J0 - eps2 * dd)

    order = np.log(r1 / r2) / np.log(eps1 / eps2)
    assert 1.8 < order < 2.2


def test_taylor_remainder_is_second_order_l2_regularizer():
    ip, dm = _small_problem(reg_type="l2", alpha=1e-1)
    rng = np.random.default_rng(2)
    m0 = 0.1 * rng.standard_normal(dm.V_m.dofmap.index_map.size_global)
    direction = rng.standard_normal(m0.shape)
    direction /= np.linalg.norm(direction)

    J0, g0, _, _ = ip.objective_and_gradient(m0)
    dd = float(g0 @ direction)

    eps1, eps2 = 1e-2, 5e-3
    J1, _, _, _ = ip.objective_and_gradient(m0 + eps1 * direction)
    J2, _, _, _ = ip.objective_and_gradient(m0 + eps2 * direction)
    r1 = abs(J1 - J0 - eps1 * dd)
    r2 = abs(J2 - J0 - eps2 * dd)

    order = np.log(r1 / r2) / np.log(eps1 / eps2)
    assert 1.8 < order < 2.2


def test_gradient_is_finite_and_nonzero_away_from_optimum():
    ip, dm = _small_problem()
    m0 = np.zeros(dm.V_m.dofmap.index_map.size_global)
    J, g, p, lam = ip.objective_and_gradient(m0)
    assert np.all(np.isfinite(g))
    assert np.linalg.norm(g) > 0.0


def test_invalid_reg_type_raises():
    dm = build_mesh(6)
    sensors = grid_sensors(2, 2)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.05)
    y = np.zeros(H.n_sensors)
    sigma = np.ones(H.n_sensors)
    try:
        InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=1.0, reg_type="nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass
