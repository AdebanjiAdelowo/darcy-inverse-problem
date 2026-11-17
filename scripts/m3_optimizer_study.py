"""Milestone 3 deterministic-optimiser and MAP convergence study
(pre-registered in docs/audit/M3_PREREGISTRATION.md).

Deterministic runs: the historical scripts are loaded unchanged (as in
scripts/reproduce_baselines.py) so that every configuration, seed and data set
is regenerated exactly; only the optimiser variant passed to
src.experiment.run_reconstruction changes:

  E-hist  euclidean,       historical budget
  P       preconditioned,  historical budget  (P = alpha K + alpha M)
  E-2000  euclidean,       budget 2000        (control)

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/m3_optimizer_study.py \
        deterministic --experiment regularization_study --variant P

MAP runs (both truths, historical full configuration):

    ... python scripts/m3_optimizer_study.py map

Outputs: results/m3/optimizer/VARIANT/EXPERIMENT.json and results/m3/optimizer/map.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "m3" / "optimizer"

from reproduce_baselines import SCRIPTS, _load  # noqa: E402
from src.optim_diagnostics import _jsonable  # noqa: E402

VARIANTS = {"E-hist": {"optimizer": "euclidean"}, "P": {"optimizer": "preconditioned"},
            "E-2000": {"optimizer": "euclidean", "max_iter": 2000}}
FTOL = 1e-12


def criteria(diag: dict) -> dict:
    """Pre-registered first-order criteria C0-C3 (see the pre-registration)."""
    e = diag["extra"]
    j = abs(diag["final_objective"])
    c1 = diag["termination_reason"] in ("gradient_tolerance", "relative_function_reduction")
    c2 = e["rel_grad_dual_P"] <= 1e-6
    c3 = e["grad_dual_P"] <= FTOL ** (1 / 3) * (1 + j)
    return {"C0_nodal_gtol": e["grad_m_inf"] <= 1e-6, "C1_scipy_test": c1, "C2_rel_dual_1e-6": c2,
            "C3_gmw": c3, "first_order_converged": bool(c1 and c2 and c3),
            "remaining_decrease_estimate": 0.5 * e["grad_dual_P"] ** 2,
            "objective_resolution_ftol_J": FTOL * max(j, 1.0)}


def deterministic(experiment: str, variant: str) -> dict:
    script, argv = SCRIPTS[experiment]
    mod = _load(script)
    original = mod.run_reconstruction
    overrides = VARIANTS[variant]
    runs = []

    def wrapped(*args, **kwargs):
        kw = dict(kwargs)
        kw["optimizer"] = overrides["optimizer"]
        if "max_iter" in overrides:
            kw["max_iter"] = overrides["max_iter"]
        t0 = time.perf_counter()
        res = original(*args, **kw)
        d = res.history.diagnostics.to_dict()
        for k in ("trace_objective", "trace_misfit", "trace_regularization", "trace_gradient_norm_2",
                  "iteration_objective"):
            d.pop(k, None)  # keep the study file compact; traces are in the M2 records
        d["criteria"] = criteria(d)
        d["wall_seconds"] = time.perf_counter() - t0
        d["mean_free_field"] = (res.m_rec - res.m_rec.mean()).tolist() if experiment == "reconstruction_demo" else None
        runs.append(d)
        return res

    mod.run_reconstruction = wrapped
    with tempfile.TemporaryDirectory() as tmp:
        mod.RESULTS_DIR = pathlib.Path(tmp)
        mod.FIG_DIR = pathlib.Path(tmp)
        old = sys.argv
        sys.argv = [script] + argv
        t0 = time.perf_counter()
        try:
            mod.main()
        finally:
            sys.argv = old
    return {"experiment": experiment, "variant": variant, "overrides": overrides,
            "wall_seconds": time.perf_counter() - t0, "runs": runs}


def map_study() -> dict:
    import scipy.linalg as sla
    from m3_mcmc import build
    from src.bayesian_experiment import compute_map_with_diagnostics
    from src.preconditioned_inversion import gauss_newton_kl_hessian, run_map_preconditioned

    out = {}
    for truth in ("smooth", "structured"):
        env = build(truth)
        bp, rk, prior = env["bp"], env["rk"], env["prior"]
        P = gauss_newton_kl_hessian(bp, rk, np.zeros(rk.r_reduced))
        Lp = sla.cholesky(P, lower=True)
        ev = np.linalg.eigvalsh(P)
        rec = {"gauss_newton_P": {"condition_number": float(ev[-1] / ev[0]), "eigenvalues": ev.tolist()}}

        def fo(g, g0, J, reason, nit, diag_extra):
            gd = float(np.linalg.norm(sla.solve_triangular(Lp, g, lower=True)))
            gd0 = float(np.linalg.norm(sla.solve_triangular(Lp, g0, lower=True)))
            c1 = reason in ("gradient_tolerance", "relative_function_reduction")
            return {"termination_reason": reason, "n_iterations": nit, "final_objective": J,
                    "grad_inf": float(np.abs(g).max()), "grad_dual_P": gd, "rel_grad_dual_P": gd / gd0,
                    "C0_nodal_gtol": float(np.abs(g).max()) <= 1e-6, "C1_scipy_test": c1,
                    "C2_rel_dual_1e-6": gd / gd0 <= 1e-6, "C3_gmw": gd <= FTOL ** (1 / 3) * (1 + abs(J)),
                    "first_order_converged": bool(c1 and gd / gd0 <= 1e-6 and gd <= FTOL ** (1 / 3) * (1 + abs(J))),
                    **diag_extra}

        for label, budget in (("E-hist", env["rc"]["map_max_iter"]), ("E-2000", 2000)):
            from src.bayesian_darcy import BayesianDarcyProblem
            b = BayesianDarcyProblem(dm=bp.dm, obs=bp.obs, y=bp.y, sigma=bp.sigma, prior=prior)
            t0 = time.perf_counter()
            xi, d = compute_map_with_diagnostics(b, max_iter=budget)
            # gradient of Psi in the reduced coordinates at the returned point
            _, g_full, _ = b.neg_log_likelihood_and_grad(xi)
            g = (g_full + xi)[rk.keep]
            _, g0_full, _ = b.neg_log_likelihood_and_grad(np.zeros(prior.r))
            g0 = g0_full[rk.keep]
            rec[label] = fo(g, g0, d.final_objective, d.termination_reason, d.n_iterations,
                            {"xi_map": xi.tolist(), "xi_c": float(xi[rk.const_index]),
                             "n_function_evaluations": d.n_function_evaluations,
                             "seconds": time.perf_counter() - t0})
        t0 = time.perf_counter()
        xi, d, g, g0, _ = run_map_preconditioned(bp, rk, P, max_iter=env["rc"]["map_max_iter"])
        rec["P-MAP"] = fo(g, g0, d.final_objective, d.termination_reason, d.n_iterations,
                          {"xi_map": xi.tolist(), "n_function_evaluations": d.n_function_evaluations,
                           "seconds": time.perf_counter() - t0})
        for a in ("E-hist", "E-2000"):
            rec[a]["max_abs_xi_R_difference_vs_P-MAP"] = float(np.max(np.abs(
                np.asarray(rec[a]["xi_map"])[rk.keep] - np.asarray(rec["P-MAP"]["xi_map"])[rk.keep])))
        out[truth] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deterministic")
    d.add_argument("--experiment", required=True,
                   choices=["reconstruction_demo", "regularization_study", "noise_study", "sensor_study"])
    d.add_argument("--variant", required=True, choices=list(VARIANTS))
    sub.add_parser("map")
    args = ap.parse_args()
    import os
    meta = {"thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.cmd == "deterministic":
        rec = deterministic(args.experiment, args.variant)
        path = OUT / args.variant / f"{args.experiment}.json"
    else:
        rec = map_study()
        path = OUT / "map.json"
    rec["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(rec), indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
