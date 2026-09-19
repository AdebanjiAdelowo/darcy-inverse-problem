"""Preconditioned Crank-Nicolson (pCN) MCMC, for sampling a posterior whose
prior is a (whitened) standard Gaussian.

## Why pCN, not random-walk Metropolis

For a Gaussian prior xi ~ N(0, I_r) (exactly the whitened KL coefficient
representation of src/kl_prior.py) and posterior pi(xi|y) proportional to
exp(-Phi(xi)) * N(xi; 0, I_r) (Phi the negative log-likelihood / data
misfit), the pCN proposal

    xi' = sqrt(1 - beta^2) * xi + beta * eta,   eta ~ N(0, I_r), beta in (0,1]

is PRIOR-REVERSIBLE: for beta fixed, the proposal density in a Metropolis-
Hastings ratio has the prior terms cancel EXACTLY, leaving an acceptance
probability depending only on the likelihood ratio,

    alpha(xi, xi') = min(1, exp(Phi(xi) - Phi(xi'))).

This is not true of naive random-walk Metropolis (xi' = xi + beta*eta),
whose acceptance ratio also carries a prior-density ratio term; more
importantly, pCN's acceptance rate does not systematically degrade as the
parameter dimension r grows (its proposal is well-defined even in the
infinite-dimensional limit), whereas random-walk Metropolis's acceptance
rate collapses as r increases unless the step size is shrunk with
dimension. This is exactly the situation here (moderate-to-large r for a
spatially distributed coefficient field), so pCN is the appropriate choice
rather than "blindly applying generic random-walk Metropolis" (Cotter,
Roberts, Stuart & White, 2013, "MCMC methods for functions: modifying old
algorithms to make them faster", Statistical Science 28(3), 424-446).

This module is DELIBERATELY generic (it only needs a callable
Phi: R^r -> R, never seeing the PDE, the mesh, or the KL prior directly),
so the exact same code is used both for the analytically-solvable linear-
Gaussian validation problem (scripts/validate_sampler.py) and the real PDE
posterior (src/bayesian_darcy.py) -- the PDE posterior is never the first
thing this sampler is tested on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class PCNResult:
    samples: np.ndarray  # (n_samples, r), POST-burn-in, POST-thinning
    neg_log_lik: np.ndarray  # Phi(xi) at each retained sample
    acceptance_rate: float
    n_proposals: int
    beta: float
    burn_in: int
    thin: int
    raw_chain: np.ndarray = field(repr=False, default=None)  # full chain incl. burn-in, for diagnostics


def run_pcn(
    neg_log_lik: Callable[[np.ndarray], float],
    r: int,
    n_samples: int,
    beta: float,
    burn_in: int,
    thin: int = 1,
    seed: int = 0,
    x0: np.ndarray | None = None,
) -> PCNResult:
    """Run pCN for burn_in + n_samples*thin proposals, returning the
    post-burn-in, post-thinning samples (and the full raw chain for
    diagnostics)."""
    rng = np.random.default_rng(seed)
    xi = np.zeros(r) if x0 is None else x0.copy()
    phi_current = neg_log_lik(xi)

    n_total = burn_in + n_samples * thin
    raw_chain = np.zeros((n_total + 1, r))
    raw_phi = np.zeros(n_total + 1)
    raw_chain[0] = xi
    raw_phi[0] = phi_current

    n_accept = 0
    sqrt_term = np.sqrt(1.0 - beta**2)

    for i in range(1, n_total + 1):
        eta = rng.standard_normal(r)
        xi_prop = sqrt_term * xi + beta * eta
        phi_prop = neg_log_lik(xi_prop)

        log_alpha = phi_current - phi_prop
        if np.log(rng.random()) < log_alpha:
            xi = xi_prop
            phi_current = phi_prop
            n_accept += 1

        raw_chain[i] = xi
        raw_phi[i] = phi_current

    post = raw_chain[burn_in + 1 :: thin] if thin > 1 else raw_chain[burn_in + 1 :]
    post_phi = raw_phi[burn_in + 1 :: thin] if thin > 1 else raw_phi[burn_in + 1 :]

    return PCNResult(
        samples=post, neg_log_lik=post_phi, acceptance_rate=n_accept / n_total,
        n_proposals=n_total, beta=beta, burn_in=burn_in, thin=thin, raw_chain=raw_chain,
    )


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation function of a 1D chain (single coordinate), lags 0..max_lag."""
    x = x - np.mean(x)
    n = len(x)
    var = np.dot(x, x) / n
    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.dot(x[: n - lag], x[lag:]) / n / var if var > 0 else 0.0
    return acf


def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    """Standard ESS estimate via the initial positive sequence estimator
    (Geyer, 1992), summing paired autocorrelation lags until the sum turns
    non-positive."""
    n = len(x)
    if max_lag is None:
        max_lag = min(n - 1, 1000)
    acf = autocorrelation(x, max_lag)
    tau = 1.0
    k = 1
    while k + 1 <= max_lag:
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum < 0:
            break
        tau += 2.0 * pair_sum
        k += 2
    return float(n / max(tau, 1e-12))
