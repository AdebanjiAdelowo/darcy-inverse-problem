"""Validation of the multi-chain diagnostics (src/mcmc_diagnostics.py) on
cases with known answers: i.i.d. draws, AR(1) chains with analytic
integrated autocorrelation time tau = (1 + rho)/(1 - rho), and chains that
have not mixed (location or scale disagreement)."""
import numpy as np
import pytest

from src import mcmc_diagnostics as md


def _ar1(rho, m, n, seed, mean=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    x = np.empty((m, n))
    x[:, 0] = rng.standard_normal(m)
    s = np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + s * rng.standard_normal(m)
    return mean + scale * x


def test_iid_draws_have_rhat_one_and_full_ess():
    x = np.random.default_rng(0).standard_normal((4, 2000))
    assert md.rhat(x) < 1.01
    assert 0.85 * 8000 < md.ess_bulk(x) < 1.15 * 8000
    assert 0.80 * 8000 < md.ess_tail(x) < 1.20 * 8000
    assert md.mcse_mean(x) == pytest.approx(1 / np.sqrt(8000), rel=0.15)


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.98])
def test_ar1_ess_matches_analytic_value(rho):
    x = _ar1(rho, 4, 50_000, seed=1)
    true = x.size * (1 - rho) / (1 + rho)
    assert md.ess_mean(x) == pytest.approx(true, rel=0.10)
    assert md.ess_bulk(x) == pytest.approx(true, rel=0.12)
    assert md.rhat(x) < 1.01


def test_ess_not_capped_for_long_autocorrelation():
    """Unlike the historical lag-1000 estimator, tau of order 1e4 is resolved."""
    rho = 0.9998  # tau ~ 1e4
    x = _ar1(rho, 4, 200_000, seed=2)
    true = x.size * (1 - rho) / (1 + rho)  # ~80
    assert md.ess_mean(x) == pytest.approx(true, rel=0.35)


def test_rhat_detects_location_disagreement():
    x = np.random.default_rng(3).standard_normal((4, 1000))
    x[0] += 1.0
    assert md.rhat(x) > 1.1


def test_folded_rhat_detects_scale_disagreement():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 2000))
    x[0] *= 3.0
    s = md.split_chains(x)
    bulk_only = md._rhat_basic(md.rank_normalize(s))
    assert bulk_only < 1.02 and md.rhat(x) > 1.05


def test_rhat_detects_within_chain_trend_via_splitting():
    n = 2000
    trend = np.linspace(-1, 1, n)
    x = np.random.default_rng(5).standard_normal((4, n)) * 0.1 + trend
    assert md.rhat(x) > 1.5


def test_rank_based_diagnostics_invariant_to_monotone_transform():
    x = _ar1(0.8, 4, 3000, seed=6)
    assert md.rhat(np.exp(x)) == pytest.approx(md.rhat(x), rel=1e-12)
    assert md.ess_bulk(np.exp(x)) == pytest.approx(md.ess_bulk(x), rel=1e-12)


def test_mcse_mean_matches_empirical_spread_of_ar1_means():
    rho, means, mcses = 0.7, [], []
    for seed in range(150):
        x = _ar1(rho, 4, 1000, seed=100 + seed)
        means.append(x.mean())
        mcses.append(md.mcse_mean(x))
    assert np.std(means) == pytest.approx(np.mean(mcses), rel=0.15)


def test_mcse_quantile_and_sd_shrink_with_sample_size():
    rng = np.random.default_rng(7)
    small, large = rng.standard_normal((4, 500)), rng.standard_normal((4, 20000))
    assert md.mcse_quantile(large, 0.95) < md.mcse_quantile(small, 0.95)
    assert md.mcse_sd(large) < md.mcse_sd(small)
    assert md.mcse_sd(large) == pytest.approx(1 / np.sqrt(2 * large.size), rel=0.2)


def test_summary_flags_convergence_thresholds():
    good = md.summarize(np.random.default_rng(8).standard_normal((4, 1000)), "iid")
    bad = md.summarize(_ar1(0.999, 4, 1000, seed=9), "slow")
    assert good["converged"] and not bad["converged"]
    assert set(good) >= {"rhat", "ess_bulk", "ess_tail", "mcse_mean", "mcse_sd", "mcse_q05", "mcse_q95"}


def test_autocorrelation_lag_zero_is_one_and_ar1_decay():
    x = _ar1(0.9, 2, 100_000, seed=10)
    ac = md.autocorrelation(x)
    assert np.allclose(ac[:, 0], 1.0)
    assert ac[:, 5].mean() == pytest.approx(0.9**5, abs=0.02)
