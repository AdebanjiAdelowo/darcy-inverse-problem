"""Analytically-solvable linear-Gaussian inverse problem, used to validate
the pCN sampler implementation BEFORE it is trusted on the PDE posterior.

Model: y = G @ xi_true + eps, eps ~ N(0, Gamma), prior xi ~ N(0, I_r)
(matching the whitened KL-coefficient prior used everywhere else in this
project). Since the prior precision is I and the likelihood is Gaussian in
xi (G is linear), the posterior is available in closed form (standard
Gaussian conjugacy):

    Sigma_post = (G^T Gamma^{-1} G + I)^{-1}
    mu_post    = Sigma_post @ G^T @ Gamma^{-1} @ y

The negative log-likelihood used by the sampler is
Phi(xi) = 0.5*(G@xi - y)^T Gamma^{-1} (G@xi - y), the exact linear-Gaussian
analogue of the PDE problem's Phi(xi) = 0.5*(H p(m(xi)) - y)^T Gamma^{-1}
(H p(m(xi)) - y) with the (linear in xi, in this toy problem) map G taking
the place of the (nonlinear in xi, for the real problem) map xi -> Hp(m(xi)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearGaussianProblem:
    G: np.ndarray  # (n_obs, r)
    y: np.ndarray  # (n_obs,)
    sigma: np.ndarray  # (n_obs,) noise std
    xi_true: np.ndarray  # (r,)

    @property
    def r(self) -> int:
        return self.G.shape[1]

    def neg_log_lik(self, xi: np.ndarray) -> float:
        resid = (self.G @ xi - self.y) / self.sigma
        return 0.5 * float(np.dot(resid, resid))

    def analytical_posterior(self):
        Gamma_inv = np.diag(1.0 / self.sigma**2)
        precision_post = self.G.T @ Gamma_inv @ self.G + np.eye(self.r)
        cov_post = np.linalg.inv(precision_post)
        mean_post = cov_post @ self.G.T @ Gamma_inv @ self.y
        return mean_post, cov_post


def make_problem(r: int, n_obs: int, noise_std: float, seed: int) -> LinearGaussianProblem:
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_obs, r))
    xi_true = rng.standard_normal(r)
    sigma = np.full(n_obs, noise_std)
    y_clean = G @ xi_true
    y = y_clean + rng.normal(0.0, noise_std, size=n_obs)
    return LinearGaussianProblem(G=G, y=y, sigma=sigma, xi_true=xi_true)
