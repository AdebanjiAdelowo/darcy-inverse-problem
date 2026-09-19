import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function

from src.forward import build_mesh
from src.regularization import l2_prior, h1_seminorm


def test_h1_seminorm_zero_for_constant_field():
    dm = build_mesh(10)
    m = Function(dm.V_m)
    m.x.array[:] = 3.7  # constant -> zero gradient everywhere
    val = fem.assemble_scalar(fem.form(h1_seminorm(m)))
    assert abs(val) < 1e-10


def test_h1_seminorm_positive_for_nonconstant_field():
    dm = build_mesh(10)
    m = Function(dm.V_m)
    x = dm.V_m.tabulate_dof_coordinates()
    m.x.array[:] = x[:, 0]
    val = fem.assemble_scalar(fem.form(h1_seminorm(m)))
    assert val > 0.0


def test_l2_prior_zero_when_m_equals_prior():
    dm = build_mesh(10)
    m0 = Function(dm.V_m)
    m0.x.array[:] = 0.5
    m = Function(dm.V_m)
    m.x.array[:] = 0.5
    val = fem.assemble_scalar(fem.form(l2_prior(m, m0)))
    assert abs(val) < 1e-10


def test_l2_prior_defaults_to_zero_reference():
    dm = build_mesh(10)
    m = Function(dm.V_m)
    m.x.array[:] = 1.0
    val_default = fem.assemble_scalar(fem.form(l2_prior(m)))
    m0 = Function(dm.V_m)
    m0.x.array[:] = 0.0
    val_explicit = fem.assemble_scalar(fem.form(l2_prior(m, m0)))
    assert abs(val_default - val_explicit) < 1e-10
