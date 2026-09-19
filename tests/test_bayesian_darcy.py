"""Tests for the Bayesian posterior construction, including a fast Taylor
test of the xi-space chain-rule gradient (see also
scripts/verify_bayesian_gradient.py for the full multi-step version
reported in the README)."""
import numpy as np

from src.forward import build_mesh
from src.kl_prior import build_kl_prior
from src.observation import grid_sensors, build_observation_operator
from src.bayesian_darcy import BayesianDarcyProblem


def _small_problem():
    dm = build_mesh(8)
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    sensors = grid_sensors(3, 3)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.05)
    rng = np.random.default_rng(1)
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    sigma = np.full(H.n_sensors, 0.02)
    return BayesianDarcyProblem(dm=dm, obs=H, y=y, sigma=sigma, prior=prior)


def test_xi_space_gradient_taylor_remainder_is_second_order():
    bp = _small_problem()
    rng = np.random.default_rng(2)
    xi0 = 0.2 * rng.standard_normal(bp.prior.r)
    direction = rng.standard_normal(bp.prior.r)
    direction /= np.linalg.norm(direction)

    J0, g0 = bp.neg_log_posterior_and_grad(xi0)
    dd = float(g0 @ direction)

    eps1, eps2 = 1e-2, 5e-3
    J1, _ = bp.neg_log_posterior_and_grad(xi0 + eps1 * direction)
    J2, _ = bp.neg_log_posterior_and_grad(xi0 + eps2 * direction)
    r1 = abs(J1 - J0 - eps1 * dd)
    r2 = abs(J2 - J0 - eps2 * dd)
    order = np.log(r1 / r2) / np.log(eps1 / eps2)
    assert 1.8 < order < 2.2


def test_neg_log_posterior_equals_likelihood_plus_half_norm_squared():
    bp = _small_problem()
    xi = np.array([0.3, -0.1, 0.2, 0.0, 0.1, -0.2])
    misfit, _, _ = bp.neg_log_likelihood_and_grad(xi)
    J, _ = bp.neg_log_posterior_and_grad(xi)
    assert abs(J - (misfit + 0.5 * np.dot(xi, xi))) < 1e-10


def test_zero_xi_gives_prior_mean_field():
    bp = _small_problem()
    xi = np.zeros(bp.prior.r)
    m = bp.prior.sample(xi)
    assert np.allclose(m.x.array, bp.prior.m0_array)


def test_pressure_solve_matches_direct_forward_solve():
    from src.forward import solve_forward
    bp = _small_problem()
    rng = np.random.default_rng(3)
    xi = rng.standard_normal(bp.prior.r)
    p_via_bp = bp.pressure(xi)
    m = bp.prior.sample(xi)
    p_direct = solve_forward(bp.dm, m, bcs=bp._ip._bcs_p)
    assert np.allclose(p_via_bp.x.array, p_direct.x.array)
