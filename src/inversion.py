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

from src.adjoint import InverseProblem
from src.optim_diagnostics import OptimizerDiagnostics, run_lbfgsb_with_diagnostics


@dataclass
class OptimizationHistory:
    objective: list = field(default_factory=list)
    misfit: list = field(default_factory=list)
    reg: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False  # SciPy success flag (historical meaning, unchanged)
    message: str = ""
    diagnostics: OptimizerDiagnostics | None = None
    predicted_observations: np.ndarray | None = None  # H p(m) at the returned iterate


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
        return J, g, misfit, reg_val, pred

    x, diag, pred_final, result = run_lbfgsb_with_diagnostics(
        objective_and_grad, m0, max_iter=max_iter, gtol=gtol, ftol=ftol,
        solve_counter=ip.solve_counts,
    )
    history.n_iterations = result.nit
    history.converged = bool(result.success)
    history.message = str(result.message)
    history.diagnostics = diag
    history.predicted_observations = pred_final
    return x, history
