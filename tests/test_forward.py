import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function

from src.forward import build_mesh, solve_forward, pressure_bcs, left_boundary, right_boundary


def test_homogeneous_permeability_matches_linear_exact_solution():
    """k=1 everywhere with p=1 at x=0, p=0 at x=1, no source: exact
    solution is p = 1-x (representable exactly by P2), so the FEM
    solution should match to floating-point roundoff."""
    dm = build_mesh(12)
    m = Function(dm.V_m)
    m.x.array[:] = 0.0
    p = solve_forward(dm, m)
    x = dm.V_p.tabulate_dof_coordinates()
    p_exact = 1.0 - x[:, 0]
    assert np.max(np.abs(p.x.array - p_exact)) < 1e-10


def test_positive_permeability_for_any_real_m():
    """k = exp(m) must be positive for arbitrary (including large negative
    or positive) m -- the entire point of the log-parameterisation."""
    dm = build_mesh(8)
    m = Function(dm.V_m)
    rng = np.random.default_rng(0)
    m.x.array[:] = 100.0 * rng.standard_normal(m.x.array.shape)  # extreme values
    k_expr = ufl.exp(m)
    k_h = Function(dm.V_m)
    k_h.interpolate(fem.Expression(k_expr, dm.V_m.element.interpolation_points))
    assert np.all(k_h.x.array > 0.0)
    assert np.all(np.isfinite(k_h.x.array))


def test_boundary_conditions_are_applied_correctly():
    dm = build_mesh(12)
    m = Function(dm.V_m)
    m.x.array[:] = 0.3  # nonzero uniform log-permeability
    p = solve_forward(dm, m)
    x = dm.V_p.tabulate_dof_coordinates()
    left_dofs = np.where(np.isclose(x[:, 0], 0.0))[0]
    right_dofs = np.where(np.isclose(x[:, 0], 1.0))[0]
    assert np.allclose(p.x.array[left_dofs], 1.0, atol=1e-10)
    assert np.allclose(p.x.array[right_dofs], 0.0, atol=1e-10)


def test_heterogeneous_permeability_gives_nontrivial_flow():
    """A spatially varying k should give a pressure field that is NOT the
    trivial linear profile (sanity check that the coefficient actually
    enters the weak form).

    Note: k depending on y ALONE would not be a valid test here -- for
    this flow-cell problem (Dirichlet at x=0,1, no-flow at y=0,1), p=1-x
    remains the EXACT solution for ANY k=k(y): the equation reduces to
    k(y)*d^2p/dx^2=0, satisfied by any p linear in x regardless of k(y).
    k must vary with x for the flow to differ from the homogeneous case.
    """
    dm = build_mesh(16)
    m = Function(dm.V_m)
    x = dm.V_m.tabulate_dof_coordinates()
    m.x.array[:] = 2.0 * np.sin(2 * np.pi * x[:, 0])  # varies with x
    p = solve_forward(dm, m)
    x_p = dm.V_p.tabulate_dof_coordinates()
    p_trivial = 1.0 - x_p[:, 0]
    assert np.max(np.abs(p.x.array - p_trivial)) > 1e-3


def test_deterministic_reproducibility():
    dm1 = build_mesh(12)
    m1 = Function(dm1.V_m)
    m1.x.array[:] = 0.2
    p1 = solve_forward(dm1, m1)

    dm2 = build_mesh(12)
    m2 = Function(dm2.V_m)
    m2.x.array[:] = 0.2
    p2 = solve_forward(dm2, m2)

    assert np.array_equal(p1.x.array, p2.x.array)


def test_boundary_predicates_are_disjoint_and_cover_left_right():
    dm = build_mesh(8)
    x = dm.V_p.tabulate_dof_coordinates()
    left_mask = left_boundary(x.T)
    right_mask = right_boundary(x.T)
    assert not np.any(left_mask & right_mask)
    assert np.sum(left_mask) > 0
    assert np.sum(right_mask) > 0
