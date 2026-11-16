"""Flatten the Milestone 2 reproduction records into machine-readable summaries.

    python scripts/summarize_m2.py --tag single_thread

Writes
  results/m2/optimizer_diagnostics_TAG.csv   one row per L-BFGS-B run (deterministic and MAP)
  results/m2/optimizer_diagnostics_TAG.json  same rows as JSON
  results/m2/bayesian_baseline_TAG.json      MAP/pCN/variance-floor diagnostics per truth
Uses only the JSON records written by scripts/reproduce_baselines.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPRO = ROOT / "results" / "m2" / "reproduction"

FIELDS = ["experiment", "run", "truth", "alpha", "relative_noise", "n_sensors", "inv_mesh_n",
          "termination_reason", "converged", "scipy_success", "first_order_condition_met",
          "n_iterations", "max_iterations", "n_function_evaluations", "n_forward_solves", "n_adjoint_solves",
          "initial_objective", "final_objective", "final_misfit", "final_regularization",
          "initial_gradient_norm_2", "final_gradient_norm_2", "final_gradient_norm_inf",
          "relative_gradient_reduction", "gtol", "ftol", "runtime_seconds",
          "rel_error_abs_nodal", "rel_error_meanfree_nodal", "rel_error_abs_L2", "rel_error_meanfree_L2",
          "rel_error_contrast_L2", "mean_offset_L2", "rel_error_pred_vs_clean",
          "rms_residual_vs_noisy_over_sigma", "scipy_message"]


def _row(exp, i, d, truth=None, errors=None):
    e = {**d.get("extra", {}), **(errors or {})}
    return {
        "experiment": exp, "run": i, "truth": truth or "smooth",
        "alpha": e.get("alpha"), "relative_noise": e.get("relative_noise"), "n_sensors": e.get("n_sensors"),
        "inv_mesh_n": e.get("inv_mesh_n"),
        **{k: d[k] for k in ("termination_reason", "converged", "scipy_success", "first_order_condition_met",
                             "n_iterations", "n_function_evaluations", "n_forward_solves", "n_adjoint_solves",
                             "initial_objective", "final_objective", "final_misfit", "final_regularization",
                             "initial_gradient_norm_2", "final_gradient_norm_2", "final_gradient_norm_inf",
                             "relative_gradient_reduction", "runtime_seconds", "scipy_message")},
        "max_iterations": d["options"]["maxiter"], "gtol": d["options"]["gtol"], "ftol": d["options"]["ftol"],
        **{k: e.get(k) for k in ("rel_error_abs_nodal", "rel_error_meanfree_nodal", "rel_error_abs_L2",
                                 "rel_error_meanfree_L2", "rel_error_contrast_L2", "mean_offset_L2",
                                 "rel_error_pred_vs_clean", "rms_residual_vs_noisy_over_sigma")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="single_thread")
    args = ap.parse_args()
    base = REPRO / args.tag
    rows, bayes = [], {}
    for exp in ("reconstruction_demo", "regularization_study", "noise_study", "sensor_study"):
        path = base / f"{exp}.json"
        if not path.exists():
            continue
        runs = json.loads(path.read_text())["captured_runs"]
        for i, d in enumerate(runs):
            truth = ("smooth", "structured")[i] if exp == "reconstruction_demo" else "smooth"
            rows.append(_row(exp, i, d, truth))
    path = base / "posterior_demo.json"
    if path.exists():
        for i, c in enumerate(json.loads(path.read_text())["captured_runs"]):
            truth = ("smooth", "structured")[i]
            rows.append(_row("posterior_demo_map", i, c["map_diagnostics"], truth, c["map_errors"]))
            bayes[truth] = {k: c[k] for k in ("likelihood_path", "pcn_runtime_seconds", "pcn_forward_solves",
                                               "pcn_adjoint_solves", "pcn_n_proposals", "acceptance_rate",
                                               "ess_per_dim_repo_estimator", "map_errors",
                                               "posterior_mean_errors", "variance_floor_check")}
            bayes[truth]["map_termination_reason"] = c["map_diagnostics"]["termination_reason"]
            bayes[truth]["map_final_gradient_norm_inf"] = c["map_diagnostics"]["final_gradient_norm_inf"]
    out = ROOT / "results" / "m2"
    with open(out / f"optimizer_diagnostics_{args.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / f"optimizer_diagnostics_{args.tag}.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / f"bayesian_baseline_{args.tag}.json").write_text(json.dumps(bayes, indent=1) + "\n")
    n_conv = sum(r["converged"] for r in rows)
    print(f"{len(rows)} optimiser runs, {n_conv} converged; reasons:",
          {r: sum(x['termination_reason'] == r for x in rows) for r in {x['termination_reason'] for x in rows}})


if __name__ == "__main__":
    main()
