"""Physical quantities of interest derived from a log-permeability field.

Effective permeability of the flow cell. With p = 1 on x = 0, p = 0 on x = 1 and
no flow on y = 0, 1, the total flux through the cell is

    Q(m) = -integral_Omega exp(m) dp/dx dx

(the weak form of the boundary flux, tested with w = x; for a homogeneous medium
k = e^c it equals e^c). On the unit square with unit pressure drop, Q is the
effective (upscaled) permeability k_eff. Because p(m + c) = p(m),
Q(m + c) = e^c Q(m): log k_eff = log Q(m_R) + (constant-mode term), so the
identifiable part log Q(m_R) can be sampled and the constant mode added exactly.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function

from src.forward import solve_forward


def effective_permeability(dm, m_array: np.ndarray, bcs=None) -> float:
    m = Function(dm.V_m)
    m.x.array[:] = m_array
    p = solve_forward(dm, m, bcs=bcs)
    return float(fem.assemble_scalar(fem.form(-ufl.exp(m) * p.dx(0) * ufl.dx)))
