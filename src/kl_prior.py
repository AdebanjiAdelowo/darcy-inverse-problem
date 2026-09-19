"""Gaussian prior for the log-permeability field, via a truncated
Karhunen-Loeve (KL) expansion built from an elliptic covariance operator.

## Covariance operator

Rather than specify a covariance KERNEL C(x,x') directly (which would need
a dense n_dofs x n_dofs matrix), we specify a covariance OPERATOR as the
inverse-square of an elliptic differential operator,

    C = A^{-2},   A = -gamma*Laplacian + delta*I  (Neumann BCs),

the standard construction for Bayesian PDE inversion (Lindgren, Rue &
Lindstrom, 2011, "An explicit link between Gaussian fields and Gaussian
Markov random fields: the stochastic partial differential equation
approach", JRSS-B 73(4); Bui-Thanh, Ghattas, Martin & Stadler, 2013, "A
computational framework for infinite-dimensional Bayesian inverse
problems", SIAM J. Sci. Comput. 35(6), which uses exactly this A^{-2}
construction). The exponent 2 (rather than 1) is not arbitrary: in 2D, a
Gaussian random field needs covariance operator A^{-alpha} with
2*alpha > d = 2, i.e. alpha > 1, to have finite pointwise variance (be
"trace-class"); alpha=1 is the borderline/inadmissible case in 2D, alpha=2
is safely trace-class and is the standard choice in this literature.

gamma controls correlation length (larger gamma = smoother, more
correlated fields), delta controls the overall variance scale together
with gamma. Neumann boundary conditions on A are used (the simplest
choice); as is well documented for this construction, this causes some
variance inflation near the domain boundary relative to the interior (a
stated limitation, not corrected here via e.g. Robin boundary conditions).

## KL eigenpairs from a generalized eigenvalue problem

Since A and C = A^{-2} commute, they share eigenfunctions. Assembling A's
weak form gives a stiffness-like matrix A_h; the P1 mass matrix is M_h.
The KL eigenpairs are exactly the solutions of the generalized eigenvalue
problem

    A_h v_j = mu_j * M_h v_j,   mu_1 <= mu_2 <= ...

(the standard finite-element discretisation of a Sturm-Liouville-type
elliptic eigenproblem), M_h-orthonormalised (v_i^T M_h v_j = delta_ij, so
the corresponding FEM functions phi_j satisfy integral(phi_i*phi_j)dx =
delta_ij, i.e. are L2-orthonormal). The covariance eigenvalues are then
lambda_j = mu_j^{-2} (descending as mu_j ascends).

## KL sample / field representation

    m(x) = m0(x) + sum_{j=1}^r sqrt(lambda_j) * xi_j * phi_j(x),  xi_j ~ N(0,1) iid.

In nodal (P1 dof) form this is simply m_dofs = m0_dofs + Phi @ (sqrt(lambda) * xi),
with Phi the (n_dofs x r) matrix of retained eigenvector columns -- a cheap
linear map used for every prior/posterior sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import ufl
from dolfinx import fem
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_matrix

from src.forward import DarcyMesh


@dataclass
class KLPrior:
    dm: DarcyMesh
    r: int
    gamma: float
    delta: float
    eigenvalues: np.ndarray  # covariance eigenvalues lambda_j, length r, descending
    eigenvectors: np.ndarray  # (n_dofs, r), M-orthonormal columns
    all_mu: np.ndarray  # ALL operator eigenvalues mu_j (for variance-capture diagnostics)
    m0_array: np.ndarray  # prior mean, nodal dofs

    @property
    def n_dofs(self) -> int:
        return self.eigenvectors.shape[0]

    def sample(self, xi: np.ndarray) -> Function:
        assert xi.shape == (self.r,)
        m = Function(self.dm.V_m)
        m.x.array[:] = self.m0_array + self.eigenvectors @ (np.sqrt(self.eigenvalues) * xi)
        return m

    def sample_array(self, xi: np.ndarray) -> np.ndarray:
        return self.m0_array + self.eigenvectors @ (np.sqrt(self.eigenvalues) * xi)

    def captured_variance_fraction(self) -> float:
        """Fraction of total prior variance (trace of C, summed over ALL
        modes resolvable by this mesh) captured by the retained r modes."""
        all_lambda = self.all_mu ** (-2.0)
        return float(np.sum(self.eigenvalues) / np.sum(all_lambda))

    def pointwise_variance(self) -> Function:
        """Var[m(x)] = sum_j lambda_j * phi_j(x)^2, using the retained modes."""
        var = Function(self.dm.V_m)
        var.x.array[:] = np.sum(self.eigenvalues[None, :] * self.eigenvectors**2, axis=1)
        return var


def build_kl_prior(dm: DarcyMesh, r: int, gamma: float, delta: float, m0_value: float = 0.0) -> KLPrior:
    u = ufl.TrialFunction(dm.V_m)
    v = ufl.TestFunction(dm.V_m)
    a_form = fem.form((gamma * ufl.inner(ufl.grad(u), ufl.grad(v)) + delta * ufl.inner(u, v)) * ufl.dx)
    m_form = fem.form(ufl.inner(u, v) * ufl.dx)

    A = assemble_matrix(a_form)
    A.assemble()
    Mmat = assemble_matrix(m_form)
    Mmat.assemble()

    n = dm.V_m.dofmap.index_map.size_global
    A_dense = A.convert("dense").getDenseArray().copy()
    M_dense = Mmat.convert("dense").getDenseArray().copy()

    # generalized symmetric eigenproblem A v = mu M v, ascending mu.
    # scipy.linalg.eigh(A, M) already returns M-orthonormal eigenvectors
    # (v_i^T M v_j = delta_ij) for the generalized problem -- verified
    # directly in tests/test_kl_prior.py, no manual renormalisation needed.
    all_mu, all_v = scipy.linalg.eigh(A_dense, M_dense)

    mu_r = all_mu[:r]
    v_r = all_v[:, :r]
    lam_r = mu_r ** (-2.0)

    m0_array = np.full(n, m0_value)
    return KLPrior(dm=dm, r=r, gamma=gamma, delta=delta, eigenvalues=lam_r, eigenvectors=v_r,
                   all_mu=all_mu, m0_array=m0_array)
