"""Tests for the L-BFGS-B termination and convergence diagnostics
(src/optim_diagnostics.py) as used by run_lbfgs and the Bayesian MAP."""
import json

import numpy as np
import pytest

from src.adjoint import InverseProblem
from src.bayesian_darcy import BayesianDarcyProblem
from src.bayesian_experiment import compute_map, compute_map_with_diagnostics
from src.forward import build_mesh
from src.inversion import run_lbfgs
from src.kl_prior import build_kl_prior
from src.observation import build_observation_operator, grid_sensors
from src.optim_diagnostics import classify_lbfgsb_termination, run_lbfgsb_with_diagnostics


@pytest.mark.parametrize("status,message,expected", [
    (0, "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL", "gradient_tolerance"),
    (0, "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL", "gradient_tolerance"),
    (0, "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH", "relative_function_reduction"),
    (0, "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH", "relative_function_reduction"),
    (1, "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", "iteration_limit"),
    (1, "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT", "iteration_limit"),
    (1, "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT", "evaluation_limit"),
    (2, "ABNORMAL_TERMINATION_IN_LNSRCH", "line_search_failure"),
    (2, "something else", "other"),
])
def test_termination_classification(status, message, expected):
    assert classify_lbfgsb_termination(status, message) == expected


def test_quadratic_reaches_gradient_tolerance_and_is_converged():
    A = np.diag([1.0, 4.0, 9.0])
    b = np.array([1.0, -2.0, 0.5])

    def f(x):
        r = A @ x - b
        return 0.5 * r @ r, A.T @ r, 0.5 * r @ r, 0.0, None

    x, d, _, _ = run_lbfgsb_with_diagnostics(f, np.zeros(3), max_iter=200, gtol=1e-10, ftol=1e-30)
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-8)
    assert d.converged and d.scipy_success and d.first_order_condition_met
    assert d.termination_reason == "gradient_tolerance"


def _small_ip(seed=3):
    dm = build_mesh(8)
    H = build_observation_operator(dm.mesh, dm.V_p, grid_sensors(3, 3), width=0.05)
    rng = np.random.default_rng(seed)
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    sigma = np.full(H.n_sensors, 0.02)
    return InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=0.1, reg_type="h1")


def test_iteration_cap_is_never_classified_as_converged():
    ip = _small_ip()
    m0 = np.zeros(ip.dm.V_m.dofmap.index_map.size_global)
    _, h = run_lbfgs(ip, m0, max_iter=3)
    d = h.diagnostics
    assert d.n_iterations == 3
    assert d.termination_reason == "iteration_limit"
    assert not d.converged and not d.scipy_success and not h.converged


def test_run_lbfgs_diagnostics_are_consistent_with_history_and_solve_counts():
    ip = _small_ip()
    m0 = np.zeros(ip.dm.V_m.dofmap.index_map.size_global)
    J0, _, _, _ = ip.objective_and_gradient(m0)
    f_before, a_before = ip.solve_counts()
    m_rec, h = run_lbfgs(ip, m0, max_iter=20)
    d = h.diagnostics
    assert d.n_function_evaluations == len(h.objective) == len(d.trace_objective)
    assert d.n_forward_solves == d.n_adjoint_solves == d.n_function_evaluations
    assert ip.solve_counts() == (f_before + d.n_forward_solves, a_before + d.n_adjoint_solves)
    assert d.initial_objective == pytest.approx(J0, rel=1e-14)
    assert d.final_objective <= d.initial_objective
    J_final, g_final, _, _ = ip.objective_and_gradient(m_rec)
    assert d.final_objective == pytest.approx(J_final, rel=1e-12)
    assert d.final_misfit + d.final_regularization == pytest.approx(d.final_objective, rel=1e-12)
    assert d.final_gradient_norm_2 == pytest.approx(np.linalg.norm(g_final), rel=1e-10)
    assert d.final_gradient_norm_inf == pytest.approx(np.max(np.abs(g_final)), rel=1e-10)
    assert d.options["maxiter"] == 20 and d.options["gtol"] == 1e-6 and d.options["ftol"] == 1e-12
    assert d.runtime_seconds > 0
    assert h.predicted_observations.shape == (ip.obs.n_sensors,)
    json.dumps(d.to_dict())  # machine-readable


def test_diagnostics_do_not_change_the_optimisation():
    """The instrumented run_lbfgs returns the same iterate as a plain SciPy call."""
    from scipy.optimize import minimize
    ip_a, ip_b = _small_ip(), _small_ip()
    m0 = np.zeros(ip_a.dm.V_m.dofmap.index_map.size_global)
    m_a, _ = run_lbfgs(ip_a, m0, max_iter=15)
    res = minimize(lambda m: ip_b.objective_and_gradient(m)[:2], m0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 15, "gtol": 1e-6, "ftol": 1e-12})
    np.testing.assert_array_equal(m_a, res.x)


def _small_bp():
    dm = build_mesh(8)
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    H = build_observation_operator(dm.mesh, dm.V_p, grid_sensors(3, 3), width=0.05)
    rng = np.random.default_rng(1)
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    return BayesianDarcyProblem(dm=dm, obs=H, y=y, sigma=np.full(H.n_sensors, 0.02), prior=prior)


def test_map_diagnostics_and_historical_interface():
    bp = _small_bp()
    xi, d = compute_map_with_diagnostics(bp, max_iter=2)
    assert d.termination_reason == "iteration_limit" and not d.converged
    assert d.final_misfit + d.final_regularization == pytest.approx(d.final_objective, rel=1e-12)
    assert d.final_regularization == pytest.approx(0.5 * xi @ xi, rel=1e-12)
    assert d.n_adjoint_solves == d.n_function_evaluations
    xi_old, nit, success = compute_map(_small_bp(), max_iter=2)
    np.testing.assert_array_equal(xi_old, xi)
    assert nit == d.n_iterations and success is False
    json.dumps(d.to_dict())
