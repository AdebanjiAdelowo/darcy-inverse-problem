"""Regularisation-preconditioned L-BFGS-B for the deterministic inversion, and
observation Jacobians for conditioning analysis and MAP preconditioning.

## Why the historical runs stall

L-BFGS-B works in the Euclidean geometry of whatever coordinates it is given.
In nodal P1 coordinates the Hessian of the H1-regularised objective is

    H = alpha*K + J^T Gamma^{-1} J  (+ second-order misfit terms),

with K the P1 stiffness matrix, whose spectrum spans O(h^2)..O(1) (condition
number O(h^-2)) on the mean-free subspace, plus an exactly singular constant
direction, and a misfit Gauss-Newton part of rank <= n_sensors. This is
mesh-dependent and poorly conditioned.

## Reparameterisation (objective unchanged)

Let P = alpha*K + gamma*M (M the P1 mass matrix, gamma > 0) and P = L L^T its
Cholesky factor. Optimise over u with m = m0 + L^{-T} u. Then

    J_u(u) = J(m0 + L^{-T} u),   grad_u = L^{-1} grad_m,

so the regularisation Hessian becomes L^{-1}(alpha K)L^{-T} = I - gamma L^{-1} M L^{-T},
whose spectrum lies in [0, 1] independently of the mesh, and the misfit enters
as a low-rank perturbation, which L-BFGS handles well. The objective, its
minimisers, the regulariser and the tolerances are unchanged; only the
coordinates in which L-BFGS-B takes steps change.

Gauge of the flat constant direction: the exact derivative is orthogonal to the
constant vector, and then P^{-1} grad_m has zero mass-weighted mean (because
P 1 = gamma M 1). Every step is P^{-1} times a combination of gradients, so the L2
mean of m stays at its initial value up to the small inconsistency of the
discrete gradient in the constant direction (|sum_i g_i| is about 5e-5 ||g||_2
relative, tests/test_identifiability.py); the observed drift is at most of order
1e-6. The historical Euclidean runs instead keep the nodal sum. Either way the
constant level is fixed by the initial guess, never by the data.

## First-order measures recorded

- `grad_m_inf`: the historical measure, max |dJ/dm_i| in nodal coordinates;
- `grad_dual_P`: ||grad_m||_{P^{-1}} = ||grad_u||_2, the norm of the derivative in
  the dual of the (alpha K + gamma M) energy norm, which is mesh-independent;
- relative reductions of both with respect to the initial gradient.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix

from src.adjoint import InverseProblem
from src.optim_diagnostics import run_lbfgsb_with_diagnostics


def _dense(form, n):
    A = assemble_matrix(fem.form(form))
    A.assemble()
    return A.convert("dense").getDenseArray().copy()


def stiffness_and_mass(V):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = V.dofmap.index_map.size_global
    K = _dense(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, n)
    M = _dense(ufl.inner(u, v) * ufl.dx, n)
    return K, M


def regularization_preconditioner(V, alpha: float, gamma: float):
    """Lower Cholesky factor L of P = alpha*K + gamma*M (dense; fine up to ~5e3 dofs)."""
    K, M = stiffness_and_mass(V)
    P = alpha * K + gamma * M
    return sla.cholesky(P, lower=True), K, M


def run_lbfgs_preconditioned(ip: InverseProblem, m0: np.ndarray, L: np.ndarray, max_iter: int = 100,
                             gtol: float = 1e-6, ftol: float = 1e-12):
    """L-BFGS-B on u with m = m0 + L^{-T} u (same objective, same SciPy options).
    Returns (m_opt, diagnostics, pred_at_opt, grad_m_at_opt, grad_m_at_start)."""
    n = len(m0)

    def to_m(u):
        return m0 + sla.solve_triangular(L, u, lower=True, trans="T")

    def fun(u):
        m = to_m(u)
        J, g_m, p, lam = ip.objective_and_gradient(m)
        pred = ip.obs.apply(p)
        misfit = 0.5 * np.sum(((pred - ip.y) / ip.sigma) ** 2)
        g_u = sla.solve_triangular(L, g_m, lower=True)
        return J, g_u, misfit, J - misfit, (pred, g_m)

    u_opt, diag, aux, result = run_lbfgsb_with_diagnostics(
        fun, np.zeros(n), max_iter=max_iter, gtol=gtol, ftol=ftol, solve_counter=ip.solve_counts)
    m_opt = to_m(u_opt)
    pred, g_m = aux if aux is not None else (None, None)
    g_m0 = L @ np.asarray(result["initial_gradient"])  # grad_u = L^{-1} grad_m
    return m_opt, diag, pred, g_m, g_m0


def first_order_measures(g_m: np.ndarray, g_m0: np.ndarray, L: np.ndarray) -> dict:
    gd = float(np.linalg.norm(sla.solve_triangular(L, g_m, lower=True)))
    gd0 = float(np.linalg.norm(sla.solve_triangular(L, g_m0, lower=True)))
    return {
        "grad_m_inf": float(np.max(np.abs(g_m))), "grad_m_2": float(np.linalg.norm(g_m)),
        "grad_dual_P": gd, "grad_dual_P_initial": gd0,
        "rel_grad_dual_P": gd / gd0 if gd0 > 0 else float("nan"),
        "rel_grad_m_2": float(np.linalg.norm(g_m) / np.linalg.norm(g_m0)),
    }


def observation_jacobian(dm, obs, m_array: np.ndarray) -> np.ndarray:
    """d(H p(m))/dm (n_sensors x n_dofs) via one adjoint solve per sensor, reusing
    the verified adjoint gradient: with y = H p(m) - e_s and unit sigma, the misfit
    gradient at m equals row s of the Jacobian."""
    ip = InverseProblem(dm=dm, obs=obs, y=np.zeros(obs.n_sensors), sigma=np.ones(obs.n_sensors),
                        alpha=0.0, reg_type="h1")
    _, p = ip.misfit_only(m_array)
    pred = obs.apply(p)
    rows = []
    for s in range(obs.n_sensors):
        e = np.zeros(obs.n_sensors)
        e[s] = 1.0
        ip.y = pred - e
        _, g, _, _ = ip.objective_and_gradient(m_array)
        rows.append(g)
    return np.array(rows)


# ------------------------------------------------------------------ MAP (KL space)

def gauss_newton_kl_hessian(bp, rk, xi_r: np.ndarray) -> np.ndarray:
    """I + G^T Gamma^{-1} G for the reduced (mean-free) KL coordinates at xi_r, with
    G = d(H p)/d xi_R = J_m Phi_R diag(sqrt(lambda_R)) from observation_jacobian."""
    m = rk.full_xi(xi_r, 0.0)
    m_nodal = bp.prior.sample_array(m)
    Jm = observation_jacobian(bp.dm, bp.obs, m_nodal)
    G = Jm @ (bp.prior.eigenvectors[:, rk.keep] * np.sqrt(bp.prior.eigenvalues[rk.keep])[None, :])
    W = G / bp.sigma[:, None]
    return np.eye(rk.r_reduced) + W.T @ W


def run_map_preconditioned(bp, rk, P: np.ndarray, max_iter: int = 120, gtol: float = 1e-6,
                           ftol: float = 1e-12):
    """MAP of Psi_R(xi_R) = Phi + 0.5||xi_R||^2 in the mean-free KL subspace with
    L-BFGS-B in coordinates v, xi_R = L^{-T} v, P = L L^T (same objective and SciPy
    options). Returns (xi_full_map, diagnostics, grad_xi_R_final, grad_xi_R_initial, L)."""
    from src.constant_mode import reduced_neg_log_posterior_and_grad
    L = sla.cholesky(P, lower=True)
    f = reduced_neg_log_posterior_and_grad(bp, rk)

    def to_xi(v):
        return sla.solve_triangular(L, v, lower=True, trans="T")

    def fun(v):
        xi_r = to_xi(v)
        J, g = f(xi_r)
        prior_term = 0.5 * float(np.dot(xi_r, xi_r))
        return J, sla.solve_triangular(L, g, lower=True), J - prior_term, prior_term, g

    v_opt, diag, g_final, result = run_lbfgsb_with_diagnostics(
        fun, np.zeros(rk.r_reduced), max_iter=max_iter, gtol=gtol, ftol=ftol, solve_counter=bp.solve_counts)
    g0 = L @ np.asarray(result["initial_gradient"])
    return rk.full_xi(to_xi(v_opt), 0.0), diag, g_final, g0, L
