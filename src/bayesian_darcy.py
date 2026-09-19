"""The PDE Bayesian posterior, built by composing the already-verified
Project 3 forward/adjoint machinery with the KL prior of src/kl_prior.py.

## Whitened parameterisation and the MAP/Tikhonov connection

Working in the KL coefficients xi (prior xi ~ N(0, I_r)) rather than the
FEM nodal array m directly is what makes both pCN (src/pcn_sampler.py) and
the following exact connection possible. The negative log-posterior is

    Psi(xi) = Phi(xi) + (1/2)||xi||^2 + const,
    Phi(xi) = (1/2)(H p(m(xi)) - y)^T Gamma^{-1} (H p(m(xi)) - y)

(Phi is the data misfit; (1/2)||xi||^2 is exactly -log of the N(0,I_r)
prior density, up to an additive constant). Minimising Psi(xi) is
therefore EXACTLY Project 3's deterministic objective

    J(m) = (1/2)||Hp(m)-y||^2_{Gamma^{-1}} + (alpha/2)*R(m)

with alpha=1 and R(xi) = ||xi||^2 an L2 (Tikhonov) penalty in the WHITENED
KL basis -- i.e. the MAP estimate under this Gaussian prior is exactly a
Tikhonov-regularised deterministic estimate, with the prior covariance
operator determining the regulariser's metric. This is the precise,
general instance of the well-known fact that a Gaussian prior's MAP
estimate coincides with Tikhonov regularisation (Stuart, 2010, Sec. 2.3).
Because Project 3's InverseProblem/adjoint code already computes exactly
Phi(m) and its exact gradient in the NODAL basis, Phi(xi) and its gradient
in the xi basis are obtained by direct reuse plus one chain-rule step
(`neg_log_likelihood_and_grad`, below) -- no new PDE/adjoint code, and no
new correctness risk beyond that one chain rule, which is itself verified
by a dedicated Taylor test (scripts/verify_bayesian_gradient.py) before
being trusted for MAP optimisation.

## Chain rule

m_dofs(xi) = m0 + Phi_KL @ (sqrt(lambda) * xi), so
d(m_i)/d(xi_j) = sqrt(lambda_j) * Phi_KL[i,j], and for any scalar
functional Psi with nodal gradient g_i = dPsi/dm_i (exactly what
InverseProblem.objective_and_gradient already returns),

    dPsi/dxi_j = sum_i g_i * d(m_i)/d(xi_j) = sqrt(lambda_j) * (Phi_KL[:,j] . g)
               = sqrt(lambda) * (Phi_KL^T @ g)   [vector form]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.adjoint import InverseProblem
from src.forward import DarcyMesh
from src.kl_prior import KLPrior
from src.observation import ObservationOperator


@dataclass
class BayesianDarcyProblem:
    dm: DarcyMesh
    obs: ObservationOperator
    y: np.ndarray
    sigma: np.ndarray
    prior: KLPrior

    def __post_init__(self) -> None:
        # alpha=0: reuse InverseProblem purely for its (verified) misfit
        # value + adjoint-gradient computation, with NO regularisation term
        # added here -- the prior is handled entirely by the xi-space
        # (1/2)||xi||^2 term added explicitly below / by pCN's proposal.
        self._ip = InverseProblem(dm=self.dm, obs=self.obs, y=self.y, sigma=self.sigma,
                                   alpha=0.0, reg_type="h1")
        self._sqrt_lambda = np.sqrt(self.prior.eigenvalues)

    def neg_log_likelihood_and_grad(self, xi: np.ndarray):
        m_array = self.prior.sample_array(xi)
        misfit, g_nodal, p, lam = self._ip.objective_and_gradient(m_array)
        grad_xi = self._sqrt_lambda * (self.prior.eigenvectors.T @ g_nodal)
        return misfit, grad_xi, p

    def neg_log_likelihood(self, xi: np.ndarray) -> float:
        """Phi(xi): for use as pCN's Phi (gradient not needed by pCN)."""
        return self.neg_log_likelihood_and_grad(xi)[0]

    def neg_log_posterior_and_grad(self, xi: np.ndarray):
        """Psi(xi) = Phi(xi) + (1/2)||xi||^2: for gradient-based MAP optimisation."""
        misfit, grad_misfit, p = self.neg_log_likelihood_and_grad(xi)
        J = misfit + 0.5 * float(np.dot(xi, xi))
        grad = grad_misfit + xi
        return J, grad

    def pressure(self, xi: np.ndarray):
        """Solve the forward problem at m(xi) and return the pressure field
        (used for posterior-predictive checks, not the objective)."""
        m = self.prior.sample(xi)
        from src.forward import solve_forward
        return solve_forward(self.dm, m, bcs=self._ip._bcs_p)
