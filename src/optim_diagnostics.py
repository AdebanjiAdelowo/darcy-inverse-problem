"""Machine-readable termination and convergence diagnostics for the SciPy
L-BFGS-B runs used by the deterministic inversion (src/inversion.py) and the
Bayesian MAP estimate (src/bayesian_experiment.py).

This module only observes an optimisation; it never changes the objective,
the optimiser, its options, or its stopping rule.

Convergence vocabulary (kept deliberately strict):

- `scipy_success`: SciPy's own `OptimizeResult.success`.
- `termination_reason`: a classification of SciPy's L-BFGS-B message:
  "gradient_tolerance" (projected-gradient inf-norm <= gtol),
  "relative_function_reduction" (relative objective decrease <= ftol),
  "iteration_limit", "evaluation_limit", "line_search_failure", "other".
- `converged`: True only for the two SciPy convergence tests above. A run that
  stops at its iteration or evaluation cap is never classified as converged,
  whatever its final gradient norm.
- `first_order_condition_met`: whether the final gradient inf-norm is at or
  below gtol, reported separately because the relative-function-reduction
  test can stop a run whose gradient is still above gtol.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy
from scipy.optimize import minimize

CONVERGED_REASONS = ("gradient_tolerance", "relative_function_reduction")


def classify_lbfgsb_termination(status: int, message: str) -> str:
    """Map SciPy L-BFGS-B (status, message) to a termination reason."""
    msg = str(message).upper().replace("_", " ")
    if "PROJECTED GRADIENT" in msg or "PGTOL" in msg:
        return "gradient_tolerance"
    if "REDUCTION OF F" in msg or "FACTR" in msg:
        return "relative_function_reduction"
    if "ITERATIONS" in msg and ("LIMIT" in msg or "EXCEED" in msg):
        return "iteration_limit"
    if "FUNCTION EVALUATIONS" in msg or ("EVALUATIONS" in msg and "EXCEED" in msg):
        return "evaluation_limit"
    if "LNSRCH" in msg or "LINE SEARCH" in msg:
        return "line_search_failure"
    if status == 1:
        return "iteration_limit"
    return "other"


@dataclass
class OptimizerDiagnostics:
    optimizer: str
    scipy_version: str
    options: dict
    n_parameters: int
    # termination
    scipy_success: bool
    scipy_status: int
    scipy_message: str
    termination_reason: str
    converged: bool
    first_order_condition_met: bool
    # counts
    n_iterations: int
    n_function_evaluations: int
    n_gradient_evaluations: int
    n_forward_solves: int | None
    n_adjoint_solves: int | None
    runtime_seconds: float
    # objective values
    initial_objective: float
    initial_misfit: float
    initial_regularization: float
    final_objective: float
    final_misfit: float | None
    final_regularization: float | None
    # gradient / optimality measures
    initial_gradient_norm_2: float
    initial_gradient_norm_inf: float
    final_gradient_norm_2: float
    final_gradient_norm_inf: float
    relative_gradient_reduction: float
    # per-evaluation trace (one entry per objective+gradient evaluation)
    trace_objective: list = field(default_factory=list)
    trace_misfit: list = field(default_factory=list)
    trace_regularization: list = field(default_factory=list)
    trace_gradient_norm_2: list = field(default_factory=list)
    # per-iteration objective (accepted iterates, from the SciPy callback)
    iteration_objective: list = field(default_factory=list)
    # filled in by callers that know the truth
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def run_lbfgsb_with_diagnostics(fun_split, x0, max_iter: int, gtol: float, ftol: float,
                                solve_counter=None):
    """Run `scipy.optimize.minimize(method="L-BFGS-B", jac=True)` exactly as the
    historical code does, recording diagnostics.

    fun_split(x) -> (J, grad, misfit, regularization, aux): J and grad are what
    the optimiser sees; misfit/regularization are the two contributions of J;
    aux is any per-evaluation payload the caller wants back for the final
    iterate (e.g. predicted observations). `solve_counter()` -> (n_forward,
    n_adjoint) cumulative PDE solve counts, or None.

    Returns (x_opt, diagnostics, aux_at_x_opt, scipy_result).
    """
    evals_x, evals = [], []
    iteration_obj = []

    def wrapped(x):
        J, g, misfit, reg, aux = fun_split(x)
        evals_x.append(np.array(x, copy=True))
        evals.append((float(J), float(misfit), float(reg), np.asarray(g, dtype=float).copy(), aux))
        return J, g

    def callback(intermediate_result):
        iteration_obj.append(float(intermediate_result.fun))

    counts0 = solve_counter() if solve_counter is not None else None
    options = {"maxiter": max_iter, "gtol": gtol, "ftol": ftol}
    t0 = time.perf_counter()
    result = minimize(wrapped, x0, jac=True, method="L-BFGS-B", options=options, callback=callback)
    runtime = time.perf_counter() - t0
    counts1 = solve_counter() if solve_counter is not None else None

    # locate the evaluation corresponding to the returned iterate
    idx = None
    for i in range(len(evals_x) - 1, -1, -1):
        if np.array_equal(evals_x[i], result.x):
            idx = i
            break

    J0, mis0, reg0, g0, _ = evals[0]
    g_final = np.asarray(result.jac, dtype=float)
    reason = classify_lbfgsb_termination(int(result.status), str(result.message))
    g0_2 = float(np.linalg.norm(g0))
    diag = OptimizerDiagnostics(
        optimizer="scipy.optimize.minimize(method='L-BFGS-B')",
        scipy_version=scipy.__version__,
        options={**options, "maxcor": 10, "maxls": 20, "maxfun": 15000,
                 "note": "maxcor/maxls/maxfun are SciPy defaults (not set by this code)"},
        n_parameters=int(np.size(x0)),
        scipy_success=bool(result.success),
        scipy_status=int(result.status),
        scipy_message=str(result.message),
        termination_reason=reason,
        converged=reason in CONVERGED_REASONS and bool(result.success),
        first_order_condition_met=bool(np.max(np.abs(g_final)) <= gtol),
        n_iterations=int(result.nit),
        n_function_evaluations=int(result.nfev),
        n_gradient_evaluations=int(getattr(result, "njev", result.nfev)),
        n_forward_solves=None if counts0 is None else int(counts1[0] - counts0[0]),
        n_adjoint_solves=None if counts0 is None else int(counts1[1] - counts0[1]),
        runtime_seconds=float(runtime),
        initial_objective=J0,
        initial_misfit=mis0,
        initial_regularization=reg0,
        final_objective=float(result.fun),
        final_misfit=None if idx is None else evals[idx][1],
        final_regularization=None if idx is None else evals[idx][2],
        initial_gradient_norm_2=g0_2,
        initial_gradient_norm_inf=float(np.max(np.abs(g0))),
        final_gradient_norm_2=float(np.linalg.norm(g_final)),
        final_gradient_norm_inf=float(np.max(np.abs(g_final))),
        relative_gradient_reduction=float(np.linalg.norm(g_final) / g0_2) if g0_2 > 0 else float("nan"),
        trace_objective=[e[0] for e in evals],
        trace_misfit=[e[1] for e in evals],
        trace_regularization=[e[2] for e in evals],
        trace_gradient_norm_2=[float(np.linalg.norm(e[3])) for e in evals],
        iteration_objective=iteration_obj,
    )
    aux_final = None if idx is None else evals[idx][4]
    return result.x, diag, aux_final, result
