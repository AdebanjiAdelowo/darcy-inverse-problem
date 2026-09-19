"""Fast regression version of scripts/validate_sampler.py's analytical
check -- smaller problem, shorter chain, but still a genuine quantitative
comparison against the closed-form posterior. The PDE posterior is never
tested by this sampler until this test (and the full validation script)
pass."""
import numpy as np

from src.linear_gaussian_validation import make_problem
from src.pcn_sampler import run_pcn, autocorrelation, effective_sample_size


def test_pcn_recovers_analytical_posterior_mean_and_variance():
    prob = make_problem(r=4, n_obs=12, noise_std=0.3, seed=5)
    mean_true, cov_true = prob.analytical_posterior()

    result = run_pcn(prob.neg_log_lik, r=prob.r, n_samples=8000, beta=0.08, burn_in=1000, seed=6)
    mc_mean = result.samples.mean(axis=0)
    mc_var = np.var(result.samples, axis=0)

    mean_rel_err = np.linalg.norm(mc_mean - mean_true) / np.linalg.norm(mean_true)
    var_rel_err = np.linalg.norm(mc_var - np.diag(cov_true)) / np.linalg.norm(np.diag(cov_true))

    assert mean_rel_err < 0.1
    assert var_rel_err < 0.3


def test_pcn_acceptance_rate_is_reasonable():
    prob = make_problem(r=4, n_obs=12, noise_std=0.3, seed=5)
    result = run_pcn(prob.neg_log_lik, r=prob.r, n_samples=4000, beta=0.08, burn_in=500, seed=7)
    assert 0.1 < result.acceptance_rate < 0.9


def test_pcn_deterministic_with_seed():
    prob = make_problem(r=3, n_obs=8, noise_std=0.3, seed=1)
    r1 = run_pcn(prob.neg_log_lik, r=prob.r, n_samples=500, beta=0.1, burn_in=100, seed=3)
    r2 = run_pcn(prob.neg_log_lik, r=prob.r, n_samples=500, beta=0.1, burn_in=100, seed=3)
    assert np.array_equal(r1.samples, r2.samples)


def test_autocorrelation_is_one_at_lag_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    acf = autocorrelation(x, max_lag=10)
    assert abs(acf[0] - 1.0) < 1e-10


def test_effective_sample_size_near_n_for_iid_samples():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)  # i.i.d., so ESS should be close to n
    ess = effective_sample_size(x)
    assert 3000 < ess <= 5000


def test_effective_sample_size_much_less_than_n_for_correlated_samples():
    rng = np.random.default_rng(0)
    n = 5000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.98 * x[i - 1] + rng.standard_normal()  # strongly autocorrelated AR(1)
    ess = effective_sample_size(x)
    assert ess < n / 10
