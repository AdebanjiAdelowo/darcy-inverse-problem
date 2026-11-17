"""Tests for the regularisation-preconditioned deterministic inversion and the
Gauss-Newton-preconditioned MAP (src/preconditioned_inversion.py): same
objective and minimisers, correct transformed gradients, preserved gauge."""
import numpy as np
import pytest
import scipy.linalg as sla

from src.adjoint import InverseProblem
from src.bayesian_darcy import BayesianDarcyProblem
from src.bayesian_experiment import compute_map_with_diagnostics
from src.constant_mode import ReducedKL
from src.forward import build_mesh
from src.identifiability import l2_mean, mass_matrix
from src.inversion import run_lbfgs
from src.kl_prior import build_kl_prior
from src.observation import build_observation_operator, grid_sensors
from src.preconditioned_inversion import (gauss_newton_kl_hessian, observation_jacobian,
                                          regularization_preconditioner, run_lbfgs_preconditioned,
                                          run_map_preconditioned)


@pytest.fixture(scope="module")
def small():
    """Realisable data: y = H p(m_true) + noise for a smooth m_true, so that the
    regularised problems have a well-defined, well-separated minimiser."""
    dm = build_mesh(8)
    H = build_observation_operator(dm.mesh, dm.V_p, grid_sensors(3, 3), width=0.05)
    x = dm.V_m.tabulate_dof_coordinates()
    m_true = 0.5 * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    ip = InverseProblem(dm=dm, obs=H, y=np.zeros(H.n_sensors), sigma=np.ones(H.n_sensors), alpha=0.0)
    y_clean = H.apply(ip.misfit_only(m_true)[1])
    sigma = np.full(H.n_sensors, 0.01)
    y = y_clean + sigma * np.random.default_rng(3).standard_normal(H.n_sensors)
    return dm, H, y, sigma


def _ip(small, alpha=1.0):
    dm, H, y, sigma = small
    return InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=alpha, reg_type="h1")


def test_preconditioner_factor(small):
    dm = small[0]
    L, K, M = regularization_preconditioner(dm.V_m, 1.0, 1.0)
    np.testing.assert_allclose(L @ L.T, K + M, rtol=1e-12, atol=1e-14)


def test_transformed_gradient_passes_taylor_test(small):
    ip = _ip(small)
    dm = small[0]
    L, _, _ = regularization_preconditioner(dm.V_m, 1.0, 1.0)
    n = L.shape[0]
    rng = np.random.default_rng(4)
    u0, du = 0.05 * rng.standard_normal(n), rng.standard_normal(n)
    to_m = lambda u: sla.solve_triangular(L, u, lower=True, trans="T")
    J0, g_m, _, _ = ip.objective_and_gradient(to_m(u0))
    g_u = sla.solve_triangular(L, g_m, lower=True)
    r = []
    for eps in (1e-2, 5e-3):
        J1, _, _, _ = ip.objective_and_gradient(to_m(u0 + eps * du))
        r.append(abs(J1 - J0 - eps * g_u @ du))
    assert 1.8 < np.log(r[0] / r[1]) / np.log(2) < 2.2


def test_preconditioned_and_euclidean_reach_same_mean_free_minimiser(small):
    dm = small[0]
    n = dm.V_m.dofmap.index_map.size_global
    m0 = np.zeros(n)
    m_e, h = run_lbfgs(_ip(small), m0, max_iter=1000)
    L, _, _ = regularization_preconditioner(dm.V_m, 1.0, 1.0)
    m_p, diag, _, g_m, g_m0 = run_lbfgs_preconditioned(_ip(small), m0, L, max_iter=1000)
    assert diag.converged
    np.testing.assert_allclose(m_p - m_p.mean(), m_e - m_e.mean(), atol=1e-4)
    assert abs(diag.final_objective - h.diagnostics.final_objective) <= 1e-9 * abs(diag.final_objective)
    assert np.linalg.norm(g_m) < 1e-3 * np.linalg.norm(g_m0)


def test_preconditioned_steps_preserve_l2_mean(small):
    dm = small[0]
    n = dm.V_m.dofmap.index_map.size_global
    L, _, _ = regularization_preconditioner(dm.V_m, 1.0, 1.0)
    m0 = np.full(n, 0.3)
    m_p, *_ = run_lbfgs_preconditioned(_ip(small), m0, L, max_iter=30)
    # preserved up to the discrete gradient's small inconsistency in the
    # constant direction (relative ~5e-5), not to machine precision
    assert l2_mean(m_p, mass_matrix(dm.V_m)) == pytest.approx(0.3, abs=1e-5)
    assert np.ptp(m_p) > 1e-3  # it did move


def test_observation_jacobian_matches_finite_differences(small):
    dm, H, _, _ = small
    n = dm.V_m.dofmap.index_map.size_global
    rng = np.random.default_rng(5)
    m, dmv = 0.3 * rng.standard_normal(n), rng.standard_normal(n)
    Jm = observation_jacobian(dm, H, m)
    ip = InverseProblem(dm=dm, obs=H, y=np.zeros(H.n_sensors), sigma=np.ones(H.n_sensors), alpha=0.0)
    eps = 1e-5
    fd = (H.apply(ip.misfit_only(m + eps * dmv)[1]) - H.apply(ip.misfit_only(m - eps * dmv)[1])) / (2 * eps)
    np.testing.assert_allclose(Jm @ dmv, fd, rtol=1e-5, atol=1e-9)


def test_gn_preconditioned_map_matches_long_euclidean_map(small):
    dm, H, y, sigma = small
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    bp = BayesianDarcyProblem(dm=dm, obs=H, y=y, sigma=sigma, prior=prior)
    rk = ReducedKL.from_prior(prior)
    P = gauss_newton_kl_hessian(bp, rk, np.zeros(rk.r_reduced))
    assert np.allclose(P, P.T) and np.linalg.eigvalsh(P).min() >= 1.0 - 1e-10
    xi_p, diag, g, g0, _ = run_map_preconditioned(bp, rk, P, max_iter=200)
    assert diag.converged and xi_p[rk.const_index] == 0.0
    xi_e, _ = compute_map_with_diagnostics(BayesianDarcyProblem(dm=dm, obs=H, y=y, sigma=sigma, prior=prior),
                                           max_iter=2000)
    np.testing.assert_allclose(xi_p[rk.keep], xi_e[rk.keep], atol=1e-4)
