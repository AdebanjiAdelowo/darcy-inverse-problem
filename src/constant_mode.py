"""Exact treatment of the unidentifiable constant KL mode in the Bayesian
Darcy posterior.

## Factorisation of the posterior

With the whitened KL prior xi ~ N(0, I_r) and m(xi) = m0 + sum_j sqrt(lambda_j)
xi_j phi_j, the first mode phi_c is constant (src/identifiability.py). Pressure,
and hence the likelihood L(xi), is invariant under m -> m + const, so L depends
on xi only through the remaining coefficients xi_R. Therefore

    pi(xi | y)  propto  L(xi_R) N(xi_R; 0, I) * N(xi_c; 0, 1),

i.e. xi_c | y ~ N(0, 1) EXACTLY and independently of xi_R | y. No information is
added to or removed from the likelihood by using this; it is an identity.

## Mean-free subspace

The KL eigenfunctions are L2-orthonormal, so every non-constant mode is
orthogonal to the constants: integral phi_j dx = 0 for j != c. The span of the
non-constant modes is therefore exactly the mean-free (identifiable) part of
the parameter space, and m(xi) splits as

    m = m0 + sqrt(lambda_c) xi_c phi_c   +   sum_{j != c} sqrt(lambda_j) xi_j phi_j
        (global log-scale; prior only)       (mean-free structure; data-informed)

Consequences used here:

1. Analytical marginalisation: sample only xi_R (dimension r-1) and add the
   constant mode's exact moments: E[m(x)] = m0 + E[m_R(x)], and
   Var[m(x)] = Var[m_R(x)] + lambda_c phi_c(x)^2.
2. Sampling in the mean-free subspace: the same xi_R chain; summaries of the
   identifiable field m_R are reported without any constant-mode term.
3. Independent reconstruction of the constant mode: append i.i.d. draws
   xi_c ~ N(0, 1) to each xi_R draw, giving exact joint draws (with Monte Carlo
   error in the constant-mode contribution).

Approaches 1 and 3 target the same posterior; 1 has no Monte Carlo error in
the constant-mode term. None of them changes the pCN kernel on xi_R.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from src.identifiability import constant_mode_index


@dataclass
class ReducedKL:
    """Split of a KLPrior into its constant mode and the mean-free subspace."""
    prior: object
    const_index: int
    keep: np.ndarray  # indices of the non-constant modes, in original order

    @classmethod
    def from_prior(cls, prior) -> "ReducedKL":
        c = constant_mode_index(prior)
        if c is None:
            raise ValueError("prior has no constant KL mode")
        keep = np.array([j for j in range(prior.r) if j != c])
        return cls(prior=prior, const_index=c, keep=keep)

    @property
    def r_reduced(self) -> int:
        return len(self.keep)

    @property
    def lambda_c(self) -> float:
        return float(self.prior.eigenvalues[self.const_index])

    @property
    def phi_c(self) -> np.ndarray:
        return self.prior.eigenvectors[:, self.const_index]

    def full_xi(self, xi_r: np.ndarray, xi_c: float = 0.0) -> np.ndarray:
        xi = np.zeros(self.prior.r)
        xi[self.keep] = xi_r
        xi[self.const_index] = xi_c
        return xi

    def reduced_xi(self, xi: np.ndarray) -> np.ndarray:
        return np.asarray(xi)[self.keep]

    def mean_free_field(self, xi_r: np.ndarray) -> np.ndarray:
        """m0 + sum over non-constant modes (nodal values)."""
        V = self.prior.eigenvectors[:, self.keep]
        lam = self.prior.eigenvalues[self.keep]
        return self.prior.m0_array + V @ (np.sqrt(lam) * xi_r)

    def mean_free_fields(self, xi_r_samples: np.ndarray) -> np.ndarray:
        V = self.prior.eigenvectors[:, self.keep]
        lam = self.prior.eigenvalues[self.keep]
        return self.prior.m0_array[None, :] + (xi_r_samples * np.sqrt(lam)[None, :]) @ V.T

    def constant_mode_pointwise_variance(self) -> np.ndarray:
        return self.lambda_c * self.phi_c ** 2


def reduced_neg_log_likelihood(bp, rk: ReducedKL):
    """Phi_R(xi_R) = Phi(full_xi(xi_R, 0)); exact because Phi does not depend on xi_c."""
    def phi(xi_r):
        return bp.neg_log_likelihood(rk.full_xi(xi_r, 0.0))
    return phi


def reduced_neg_log_posterior_and_grad(bp, rk: ReducedKL):
    """Psi_R(xi_R) = Phi(xi_R) + 0.5||xi_R||^2 and its gradient (for the MAP of
    the identifiable part; the MAP of xi_c is 0 exactly)."""
    def f(xi_r):
        misfit, g, _ = bp.neg_log_likelihood_and_grad(rk.full_xi(xi_r, 0.0))
        return misfit + 0.5 * float(np.dot(xi_r, xi_r)), g[rk.keep] + xi_r
    return f


def marginalized_moments(rk: ReducedKL, xi_r_samples: np.ndarray) -> dict:
    """Posterior moments of m with the constant mode integrated analytically."""
    mf = rk.mean_free_fields(xi_r_samples)
    var_mf = mf.var(axis=0, ddof=1)
    var_c = rk.constant_mode_pointwise_variance()
    return {"mean": mf.mean(axis=0), "var_mean_free": var_mf, "var_constant_mode": var_c,
            "var_total": var_mf + var_c}


def augment_with_independent_constant_mode(rk: ReducedKL, xi_r_samples: np.ndarray, seed: int):
    """Approach 3: exact joint draws (xi_c, xi_R) with xi_c ~ N(0,1) i.i.d."""
    rng = np.random.default_rng(seed)
    xi_c = rng.standard_normal(len(xi_r_samples))
    fields = rk.mean_free_fields(xi_r_samples) + np.sqrt(rk.lambda_c) * xi_c[:, None] * rk.phi_c[None, :]
    return xi_c, fields


def rao_blackwell_quantiles(mean_free_values: np.ndarray, sd_c: float, probs) -> np.ndarray:
    """Quantiles of m(x) = m_R(x) + Z, Z ~ N(0, sd_c^2) independent of m_R, with
    the empirical distribution of m_R(x): F(t) = mean_s Phi((t - m_R^s)/sd_c)."""
    v = np.asarray(mean_free_values, dtype=float)
    out = []
    lo, hi = v.min() - 10 * sd_c, v.max() + 10 * sd_c
    for p in np.atleast_1d(probs):
        f = lambda t: stats.norm.cdf((t - v) / sd_c).mean() - p
        out.append(optimize.brentq(f, lo, hi, xtol=1e-12))
    return np.array(out)


def lognormal_factor(rk: ReducedKL) -> np.ndarray:
    """E[exp(sqrt(lambda_c) xi_c phi_c)] = exp(lambda_c phi_c^2 / 2): the exact
    factor relating E[k(x)] to E[exp(m_R(x))]."""
    return np.exp(0.5 * rk.constant_mode_pointwise_variance())
