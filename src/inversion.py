"""L-BFGS optimisation of the regularised Darcy inverse problem, with a
recorded iteration history.

L-BFGS (Nocedal & Wright, 2006, *Numerical Optimization*, Ch. 6) is used
because it is the standard quasi-Newton method for large-scale smooth
unconstrained optimisation when only gradient (not Hessian) information is
available cheaply -- exactly the situation here, where the adjoint method
gives an exact gradient at the cost of one extra PDE solve per evaluation,
but forming the full Hessian would need many more. SciPy's
`scipy.optimize.minimize(method="L-BFGS-B")` is used as the established
implementation, wrapping the adjoint gradient from src/adjoint.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from src.adjoint import InverseProblem


@dataclass
class OptimizationHistory:
    objective: list = field(default_factory=list)
    misfit: list = field(default_factory=list)
    reg: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    message: str = ""


def run_lbfgs(
    ip: InverseProblem,
    m0: np.ndarray,
    max_iter: int = 100,
    gtol: float = 1e-6,
    ftol: float = 1e-12,
) -> tuple[np.ndarray, OptimizationHistory]:
    history = OptimizationHistory()

    def objective_and_grad(m_array):
        J, g, p, lam = ip.objective_and_gradient(m_array)
        pred = ip.obs.apply(p)
        misfit = 0.5 * np.sum(((pred - ip.y) / ip.sigma) ** 2)
        reg_val = J - misfit
        history.objective.append(J)
        history.misfit.append(misfit)
        history.reg.append(reg_val)
        history.grad_norm.append(float(np.linalg.norm(g)))
        return J, g

    result = minimize(
        objective_and_grad, m0, jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
    )
    history.n_iterations = result.nit
    history.converged = bool(result.success)
    history.message = str(result.message)
    return result.x, history
