"""Fast (2-resolution) regression version of scripts/verify_forward.py's
convergence check -- the full 5-resolution order estimation is too slow
for the default test budget and is run separately."""
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological
from dolfinx import mesh as dmesh

from src.forward import build_mesh, solve_forward
from src.manufactured import exact_fields


def _l2_error(n: int) -> float:
    dm = build_mesh(n)
    msh = dm.mesh
    x = ufl.SpatialCoordinate(msh)
    p_exact, m_exact, f = exact_fields(x)

    m_h = Function(dm.V_m)
    m_h.interpolate(fem.Expression(m_exact, dm.V_m.element.interpolation_points))

    p_bc = Function(dm.V_p)
    p_bc.interpolate(fem.Expression(p_exact, dm.V_p.element.interpolation_points))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dmesh.exterior_facet_indices(msh.topology)
    bdofs = locate_dofs_topological(dm.V_p, tdim - 1, boundary_facets)
    bcs = [dirichletbc(p_bc, bdofs)]

    p_h = solve_forward(dm, m_h, f=f, bcs=bcs)
    return np.sqrt(fem.assemble_scalar(fem.form((p_h - p_exact) ** 2 * ufl.dx)))


def test_forward_solver_error_decreases_under_refinement():
    e_coarse = _l2_error(8)
    e_fine = _l2_error(16)
    assert e_fine < 0.3 * e_coarse  # expect ~4x drop (O(h^2), coefficient-limited); require >3x


def test_forward_solver_matches_manufactured_solution_reasonably_well():
    e = _l2_error(16)
    assert e < 1e-2
