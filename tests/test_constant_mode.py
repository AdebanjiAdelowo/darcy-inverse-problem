"""Regression tests for the exact constant-mode treatment (src/constant_mode.py)
and for the theoretical posterior-variance bound it guarantees."""
import numpy as np
import pytest
from scipy import stats

from src.bayesian_darcy import BayesianDarcyProblem
from src.constant_mode import (ReducedKL, augment_with_independent_constant_mode, lognormal_factor,
                               marginalized_moments, rao_blackwell_quantiles, reduced_neg_log_likelihood)
from src.forward import build_mesh
from src.identifiability import l2_mean, mass_matrix
from src.kl_prior import build_kl_prior
from src.observation import build_observation_operator, grid_sensors
from src.pcn_sampler import run_pcn


@pytest.fixture(scope="module")
def env():
    dm = build_mesh(8)
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    H = build_observation_operator(dm.mesh, dm.V_p, grid_sensors(3, 3), width=0.05)
    rng = np.random.default_rng(1)
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    bp = BayesianDarcyProblem(dm=dm, obs=H, y=y, sigma=np.full(H.n_sensors, 0.02), prior=prior)
    return dm, prior, bp, ReducedKL.from_prior(prior), mass_matrix(dm.V_m)


def test_non_constant_modes_span_the_mean_free_subspace(env):
    """Analytical marginalisation and mean-free sampling coincide: every
    non-constant KL mode has zero mean."""
    dm, prior, _, rk, M = env
    assert rk.const_index == 0 and rk.r_reduced == prior.r - 1
    for j in rk.keep:
        assert abs(l2_mean(prior.eigenvectors[:, j], M)) < 1e-12
    xi_r = np.random.default_rng(2).standard_normal(rk.r_reduced)
    assert abs(l2_mean(rk.mean_free_field(xi_r), M)) < 1e-12


def test_reduced_likelihood_equals_full_likelihood_for_any_constant_coefficient(env):
    _, _, bp, rk, _ = env
    phi_r = reduced_neg_log_likelihood(bp, rk)
    rng = np.random.default_rng(3)
    for _ in range(3):
        xi_r = rng.standard_normal(rk.r_reduced)
        ref = phi_r(xi_r)
        for xc in (-2.0, 0.7, 3.1):
            assert bp.neg_log_likelihood(rk.full_xi(xi_r, xc)) == pytest.approx(ref, rel=1e-12)


def test_marginalized_variance_respects_exact_floor_even_for_an_unmixed_chain(env):
    """The bound Var[m(x)|y] >= lambda_c phi_c(x)^2 holds by construction, even
    for a chain that never moved (worst case for the historical estimator)."""
    _, _, _, rk, M = env
    stuck = np.tile(np.random.default_rng(4).standard_normal(rk.r_reduced), (50, 1))
    mom = marginalized_moments(rk, stuck)
    floor = rk.constant_mode_pointwise_variance()
    assert np.all(mom["var_total"] >= floor - 1e-15)
    np.testing.assert_allclose(mom["var_total"], mom["var_mean_free"] + floor, rtol=0, atol=1e-15)
    np.testing.assert_allclose(floor, 1.0, rtol=1e-10)  # delta = 1, unit square
    assert mom["var_total"] @ (M @ np.ones_like(floor)) >= rk.lambda_c - 1e-12


def test_independent_constant_mode_draws_match_exact_marginal(env):
    _, _, _, rk, _ = env
    xi_r = 0.1 * np.random.default_rng(5).standard_normal((4000, rk.r_reduced))
    xi_c, fields = augment_with_independent_constant_mode(rk, xi_r, seed=6)
    assert stats.kstest(xi_c, "norm").pvalue > 0.01
    mom = marginalized_moments(rk, xi_r)
    var_aug = fields.var(axis=0, ddof=1)
    np.testing.assert_allclose(var_aug, mom["var_total"], rtol=0.08)


def test_rao_blackwell_quantiles(env):
    sd = 1.3
    q = rao_blackwell_quantiles(np.full(10, 0.4), sd, [0.025, 0.5, 0.975])
    np.testing.assert_allclose(q, 0.4 + sd * stats.norm.ppf([0.025, 0.5, 0.975]), atol=1e-9)
    v = np.random.default_rng(7).normal(0.2, 0.5, 200_000)
    q = rao_blackwell_quantiles(v, 1.0, [0.05, 0.95])
    np.testing.assert_allclose(q, 0.2 + np.sqrt(1.25) * stats.norm.ppf([0.05, 0.95]), atol=0.01)


def test_lognormal_factor_is_exact_for_constant_mode(env):
    _, _, _, rk, _ = env
    z = np.random.default_rng(8).standard_normal(400_000)
    mc = np.exp(np.sqrt(rk.lambda_c) * z * rk.phi_c[0]).mean()
    assert lognormal_factor(rk)[0] == pytest.approx(mc, rel=0.01)


def test_pcn_in_reduced_space_runs_and_never_touches_constant_mode(env):
    _, _, bp, rk, _ = env
    res = run_pcn(reduced_neg_log_likelihood(bp, rk), r=rk.r_reduced, n_samples=40, beta=0.3,
                  burn_in=0, seed=9)
    assert res.samples.shape == (40, rk.r_reduced)
    assert 0.0 < res.acceptance_rate <= 1.0
