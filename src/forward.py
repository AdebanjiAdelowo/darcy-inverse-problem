"""Darcy forward solver: -div(k(x) grad p(x)) = f(x) in Omega, with k = exp(m).

## Domain and boundary conditions

Omega = [0,1] x [0,1] (a "flow cell"): pressure p=1 prescribed on the left
edge (x=0, the injection boundary), p=0 prescribed on the right edge (x=1,
the production boundary), and no-flow (natural, k*dp/dn=0) on the top and
bottom edges. This drives a left-to-right flow whose pattern depends on the
heterogeneous permeability field k(x) -- exactly the "flow cell" or "single
transect" setup standard in groundwater/reservoir inverse-problem
literature (e.g. as used throughout Stuart, A. M. (2010), "Inverse
problems: a Bayesian perspective", Acta Numerica). No source term is used
for the physical inverse problem (f=0); f is used only for the
manufactured-solution verification in src/manufactured.py.

## Log-permeability parameterisation

Permeability must remain strictly positive. Rather than constraining a
bounded optimisation variable directly, we parameterise m = log(k), so
k = exp(m) is automatically positive for ANY real-valued m -- turning a
constrained problem into an unconstrained one, standard practice for this
class of inverse problem (Stuart, 2010, Sec. 2; Bui-Thanh & Ghattas et al.,
2013, "A computational framework for infinite-dimensional Bayesian inverse
problems"). m is discretised as a continuous piecewise-linear (P1) nodal
field on the same mesh as the pressure (or, for the inverse-crime-avoidance
setup, on a DIFFERENT, coarser mesh than the one used to generate synthetic
truth data -- see src/observation.py and README, "Avoiding the inverse
crime").

## Weak formulation

Find p in H^1_g(Omega) (g the Dirichlet data above) such that for all
v in H^1_0(Omega):

    a(m)(p, v) := integral_Omega exp(m) * grad(p) . grad(v) dx
                = integral_Omega f * v dx =: L(v)

This is the parameter-to-state map m |-> p(m) used throughout the rest of
this repository (src/adjoint.py differentiates it for the inverse problem;
src/observation.py composes it with the observation operator H).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem


@dataclass
class DarcyMesh:
    mesh: object
    V_p: object  # pressure function space (P2)
    V_m: object  # log-permeability function space (P1)


def build_mesh(n: int) -> DarcyMesh:
    """n x n structured triangular mesh of the unit square."""
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, n, n, dmesh.CellType.triangle)
    V_p = functionspace(msh, ("Lagrange", 2))
    V_m = functionspace(msh, ("Lagrange", 1))
    return DarcyMesh(msh, V_p, V_m)


def left_boundary(x):
    return np.isclose(x[0], 0.0)


def right_boundary(x):
    return np.isclose(x[0], 1.0)


def pressure_bcs(dm: DarcyMesh):
    p_left = Function(dm.V_p)
    p_left.x.array[:] = 1.0
    p_right = Function(dm.V_p)
    p_right.x.array[:] = 0.0

    dofs_left = locate_dofs_geometrical(dm.V_p, left_boundary)
    dofs_right = locate_dofs_geometrical(dm.V_p, right_boundary)
    return [dirichletbc(p_left, dofs_left), dirichletbc(p_right, dofs_right)]


def solve_forward(dm: DarcyMesh, m: Function, f=None, bcs=None) -> Function:
    """Solve the Darcy equation for pressure given log-permeability m.

    m: Function on dm.V_m (or any UFL expression of the mesh coordinates,
       for manufactured-solution use).
    f: optional UFL expression for the source term (default: 0).
    bcs: optional explicit list of DirichletBC (default: the flow-cell BCs
         from pressure_bcs()).
    """
    msh = dm.mesh
    p = ufl.TrialFunction(dm.V_p)
    v = ufl.TestFunction(dm.V_p)
    k = ufl.exp(m)

    a = k * ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx
    if f is None:
        f = fem.Constant(msh, PETSc.ScalarType(0.0))
    L = f * v * ufl.dx

    if bcs is None:
        bcs = pressure_bcs(dm)

    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix="darcy_forward_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    return problem.solve()
