"""Regression record of the effective-sample-size estimator's behaviour on
AR(1) chains with known integrated autocorrelation time
tau = (1 + rho) / (1 - rho). The estimator (src/pcn_sampler.py) is NOT changed
here; these tests pin down (a) its accuracy when tau is well below its lag cap
of 1000 and (b) the upward bias, documented in the Milestone 1 audit, when tau
exceeds the cap."""
import numpy as np

from src.pcn_sampler import effective_sample_size


def _ar1(rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    s = np.sqrt(1.0 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + s * rng.standard_normal()
    return x


def test_ess_accurate_for_moderate_autocorrelation():
    rho, n = 0.9, 200_000
    true_ess = n * (1 - rho) / (1 + rho)  # tau = 19
    est = effective_sample_size(_ar1(rho, n, seed=0))
    assert abs(est - true_ess) / true_ess < 0.15


def test_ess_biased_upward_when_tau_exceeds_lag_cap():
    """With tau ~ 10^4 > 1000 the lag cap bounds tau by about 2001, so the
    estimate cannot fall below about n/2001 even though the true ESS is ~1."""
    rho, n = 0.9998, 8000
    true_ess = n * (1 - rho) / (1 + rho)  # about 0.8
    est = effective_sample_size(_ar1(rho, n, seed=1))
    assert est >= n / 2001 - 1e-9
    assert est > 3 * true_ess
