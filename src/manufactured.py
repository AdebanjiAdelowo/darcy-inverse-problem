"""Manufactured solution for verifying the Darcy forward solver.

A smooth, non-polynomial exact pressure and log-permeability are chosen,
and the source term f required to make them an exact solution of
-div(exp(m)*grad(p)) = f is obtained by UFL's own symbolic differentiation
(no external CAS, no hand-derived algebra -- the same approach used for the
Navier-Stokes manufactured solutions in the companion `navier-stokes-2d`
and `fem-cylinder-flow` projects). Dirichlet data on the full boundary is
taken from the manufactured p_exact itself (not the physical flow-cell
boundary conditions used elsewhere in this repository, which are unrelated
to this verification-only problem).
"""
from __future__ import annotations

import ufl


def exact_fields(x):
    """x: ufl.SpatialCoordinate. Returns (p_exact, m_exact, f) as UFL expressions."""
    p_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + 1.0
    m_exact = 0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    k_exact = ufl.exp(m_exact)
    f = -ufl.div(k_exact * ufl.grad(p_exact))
    return p_exact, m_exact, f
