"""Regularised objective, adjoint equation, and gradient for the Darcy
log-permeability inverse problem.

## Objective

    J(m) = (1/2) * (Hp(m) - y)^T Gamma^{-1} (Hp(m) - y)  +  (alpha/2) * R(m)

with p(m) the forward (state) map of src/forward.py, H the observation
operator of src/observation.py, Gamma = diag(sigma_1^2, ..., sigma_n^2) the
observation-noise covariance, and R a regulariser from src/regularization.py.

## Adjoint derivation

Write the forward problem as a weak residual: find p such that

    F(p, m; v) := integral_Omega exp(m) grad(p) . grad(v) dx
                  - integral_Omega f v dx  =  0   for all v in V_0

(V_0 = H^1_0, the homogeneous counterpart of the Dirichlet space p lives
in). This F is exactly the bilinear/linear weak form of src/forward.py,
re-read as a residual. Introduce the Lagrangian

    L(m, p, lambda) = J_misfit(p) + (alpha/2) R(m) + F(p, m; lambda)

(lambda plays the role of a Lagrange multiplier / adjoint variable, slotted
into F's test-function argument). Stationarity of L with respect to p, in
an arbitrary admissible direction delta_p in V_0, gives the ADJOINT
EQUATION: find lambda in V_0 such that for all delta_p in V_0,

    integral_Omega exp(m) grad(delta_p) . grad(lambda) dx
        = - dJ_misfit/dp [delta_p]
        = - (H delta_p)^T Gamma^{-1} (Hp - y)

The left-hand side is EXACTLY the same bilinear form as the forward
problem (the Darcy operator is self-adjoint), so the adjoint equation is
"solve the same PDE again, with homogeneous Dirichlet data (lambda = 0
where p was prescribed) and a source term built from the weighted
observation residual instead of f". This is why src/observation.py's H is
built directly from finite-element test-function weight vectors: the
adjoint right-hand side is simply H^T (Gamma^{-1}(Hp - y)), i.e. exactly
`ObservationOperator.adjoint_rhs_vector`.

## Gradient

Stationarity of L with respect to m, in direction delta_m, gives the
gradient (using k(m) = exp(m), dk/dm = k):

    dJ/dm [delta_m] = integral_Omega exp(m) delta_m * grad(p) . grad(lambda) dx
                       + alpha * dR/dm [delta_m]

Evaluated against the P1 nodal basis {phi_i}, this gives the gradient
vector g_i = dJ/dm[phi_i], assembled directly as a UFL linear form (the
first term via `ufl.derivative` of F with respect to m, tested with
lambda already substituted for v -- letting UFL's own symbolic
differentiation handle the exp(m) chain rule, exactly as the Jacobian is
obtained automatically in the companion `fem-cylinder-flow` project's
Newton solver).

This gradient is not trusted until it passes the Taylor-remainder check in
scripts/verify_gradient.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem, la
from dolfinx.fem import Function, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, set_bc

from src.forward import DarcyMesh, pressure_bcs, solve_forward, left_boundary, right_boundary
from src.observation import ObservationOperator
from src.regularization import h1_seminorm, l2_prior


def homogeneous_bcs(dm: DarcyMesh):
    """Same Dirichlet dofs as pressure_bcs(), but with value 0 -- used for
    the adjoint variable lambda (and for perturbation directions delta_p
    conceptually), since admissible directions/adjoint variables vanish
    wherever the state is prescribed."""
    zero = Function(dm.V_p)
    zero.x.array[:] = 0.0
    dofs_left = locate_dofs_geometrical(dm.V_p, left_boundary)
    dofs_right = locate_dofs_geometrical(dm.V_p, right_boundary)
    return [dirichletbc(zero, dofs_left), dirichletbc(zero, dofs_right)]


@dataclass
class InverseProblem:
    dm: DarcyMesh
    obs: ObservationOperator
    y: np.ndarray
    sigma: np.ndarray  # per-sensor noise std (n_sensors,)
    alpha: float
    reg_type: str = "h1"  # "h1" (smoothness) or "l2" (deviation from m0)
    m0: Function | None = None

    def __post_init__(self):
        if self.reg_type not in ("h1", "l2"):
            raise ValueError(f"unknown reg_type {self.reg_type!r}")
        self._m = Function(self.dm.V_m)
        self._bcs_p = pressure_bcs(self.dm)
        self._bcs_adj = homogeneous_bcs(self.dm)

    def _regularizer_form(self, m):
        if self.reg_type == "h1":
            return h1_seminorm(m)
        return l2_prior(m, self.m0)

    def objective_and_gradient(self, m_array: np.ndarray):
        self._m.x.array[:] = m_array
        m = self._m

        # --- forward solve ---
        p = solve_forward(self.dm, m, bcs=self._bcs_p)

        pred = self.obs.apply(p)
        weighted_residual = (pred - self.y) / self.sigma**2
        misfit = 0.5 * np.sum(((pred - self.y) / self.sigma) ** 2)

        R_form = self._regularizer_form(m)
        reg = fem.assemble_scalar(fem.form(R_form))
        J = misfit + 0.5 * self.alpha * reg

        # --- adjoint solve ---
        v = ufl.TestFunction(self.dm.V_p)
        u = ufl.TrialFunction(self.dm.V_p)
        k = ufl.exp(m)
        a_adj = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        a_form = fem.form(a_adj)

        # b_arr is already a fully-assembled load vector (a linear
        # combination of the observation weight vectors, each itself
        # assembled as integral(bump * v)*dx) -- no separate UFL linear
        # form is needed for the RHS. The adjoint BCs are homogeneous, so
        # `apply_lifting` (which corrects for nonzero Dirichlet data) would
        # be a no-op here and is skipped; `set_bc` alone enforces
        # lambda = 0 on the Dirichlet dofs.
        b_arr = -self.obs.adjoint_rhs_vector(weighted_residual)
        b_func = Function(self.dm.V_p)
        b_func.x.array[:] = b_arr

        A = assemble_matrix(a_form, bcs=self._bcs_adj)
        A.assemble()
        b_vec = b_func.x.petsc_vec.copy()
        set_bc(b_vec, self._bcs_adj)

        lam = Function(self.dm.V_p)
        ksp = PETSc.KSP().create(self.dm.mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.solve(b_vec, lam.x.petsc_vec)
        lam.x.scatter_forward()

        # --- gradient assembly ---
        phi = ufl.TestFunction(self.dm.V_m)
        grad_data = ufl.exp(m) * phi * ufl.inner(ufl.grad(p), ufl.grad(lam)) * ufl.dx
        # R'(m)[phi]: UFL differentiates R_form w.r.t. m automatically
        dR = ufl.derivative(R_form, m, phi)
        grad_form = fem.form(grad_data + 0.5 * self.alpha * dR)
        g_vec = fem.assemble_vector(grad_form)
        g_vec.scatter_reverse(la.InsertMode.add)

        return float(J), np.asarray(g_vec.array).copy(), p, lam
