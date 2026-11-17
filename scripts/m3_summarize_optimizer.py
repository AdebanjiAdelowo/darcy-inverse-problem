"""Summarise the Milestone 3 optimiser study (results/m3/optimizer/*/*.json and
map.json) against the pre-registered first-order criteria C0-C3.

    python scripts/m3_summarize_optimizer.py
Output: results/m3/optimizer/summary.json and summary.csv
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT = ROOT / "results" / "m3" / "optimizer"
EXPS = ["reconstruction_demo", "regularization_study", "noise_study", "sensor_study"]


def main():
    rows, agg = [], {}
    for variant in ("E-hist", "P", "E-2000"):
        for exp in EXPS:
            rec = json.loads((OPT / variant / f"{exp}.json").read_text())
            for i, d in enumerate(rec["runs"]):
                e, c = d["extra"], d["criteria"]
                rows.append({
                    "variant": variant, "experiment": exp, "run": i, "alpha": e["alpha"],
                    "relative_noise": e["relative_noise"], "n_sensors": e["n_sensors"],
                    "termination_reason": d["termination_reason"], "n_iterations": d["n_iterations"],
                    "max_iterations": d["options"]["maxiter"], "n_function_evaluations": d["n_function_evaluations"],
                    "final_objective": d["final_objective"], "grad_m_inf": e["grad_m_inf"],
                    "rel_grad_dual_P": e["rel_grad_dual_P"], "grad_dual_P": e["grad_dual_P"],
                    **{k: c[k] for k in ("C0_nodal_gtol", "C1_scipy_test", "C2_rel_dual_1e-6", "C3_gmw",
                                         "first_order_converged")},
                    "remaining_decrease_estimate": c["remaining_decrease_estimate"],
                    "objective_resolution_ftol_J": c["objective_resolution_ftol_J"],
                    "rel_error_meanfree_L2": e["rel_error_meanfree_L2"],
                    "rel_error_abs_nodal": e["rel_error_abs_nodal"], "runtime_seconds": d["runtime_seconds"],
                })
    for variant in ("E-hist", "P", "E-2000"):
        r = [x for x in rows if x["variant"] == variant]
        agg[variant] = {
            "runs": len(r),
            "iteration_limit": sum(x["termination_reason"] == "iteration_limit" for x in r),
            "C0": sum(x["C0_nodal_gtol"] for x in r), "C1": sum(x["C1_scipy_test"] for x in r),
            "C2": sum(x["C2_rel_dual_1e-6"] for x in r), "C3": sum(x["C3_gmw"] for x in r),
            "first_order_converged": sum(x["first_order_converged"] for x in r),
            "median_iterations": float(np.median([x["n_iterations"] for x in r])),
            "total_function_evaluations": int(sum(x["n_function_evaluations"] for x in r)),
            "total_optimizer_seconds": float(sum(x["runtime_seconds"] for x in r)),
            "termination_reasons": {k: sum(x["termination_reason"] == k for x in r)
                                    for k in sorted({x["termination_reason"] for x in r})},
        }
    # agreement of identifiable solutions and objectives, P vs E-2000, paired by (experiment, run)
    key = lambda x: (x["experiment"], x["run"])
    p = {key(x): x for x in rows if x["variant"] == "P"}
    e2 = {key(x): x for x in rows if x["variant"] == "E-2000"}
    eh = {key(x): x for x in rows if x["variant"] == "E-hist"}
    diffs = [abs(p[k]["final_objective"] - e2[k]["final_objective"]) / abs(e2[k]["final_objective"]) for k in p]
    err_d = [abs(p[k]["rel_error_meanfree_L2"] - e2[k]["rel_error_meanfree_L2"]) for k in p]
    err_h = [abs(eh[k]["rel_error_meanfree_L2"] - e2[k]["rel_error_meanfree_L2"]) for k in p]
    lower = [k for k in p if p[k]["final_objective"] < e2[k]["final_objective"] - 1e-12 * abs(e2[k]["final_objective"])]
    agg["paired_P_vs_E-2000"] = {"max_rel_objective_difference": float(max(diffs)),
                                 "max_abs_meanfree_error_difference": float(max(err_d)),
                                 "runs_where_P_objective_lower": len(lower)}
    agg["paired_E-hist_vs_E-2000"] = {"max_abs_meanfree_error_difference": float(max(err_h))}
    agg["P_failures"] = [{k: x[k] for k in ("experiment", "run", "alpha", "termination_reason", "n_iterations",
                                            "grad_m_inf", "rel_grad_dual_P", "C1_scipy_test", "C2_rel_dual_1e-6",
                                            "C3_gmw")}
                         for x in rows if x["variant"] == "P" and not x["first_order_converged"]]
    agg["E-2000_failures"] = [{k: x[k] for k in ("experiment", "run", "alpha", "termination_reason", "n_iterations",
                                                 "rel_grad_dual_P", "C2_rel_dual_1e-6", "C3_gmw")}
                              for x in rows if x["variant"] == "E-2000" and not x["first_order_converged"]]
    agg["map"] = json.loads((OPT / "map.json").read_text())
    for t in ("smooth", "structured"):
        for v in ("E-hist", "E-2000", "P-MAP"):
            agg["map"][t][v].pop("xi_map", None)
    (OPT / "summary.json").write_text(json.dumps(agg, indent=1) + "\n")
    with open(OPT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in agg.items() if k != "map"}, indent=1))
    for t in ("smooth", "structured"):
        for v in ("E-hist", "E-2000", "P-MAP"):
            m = agg["map"][t][v]
            print(t, v, m["termination_reason"], m["n_iterations"], f"ginf={m['grad_inf']:.2e} rel_dual={m['rel_grad_dual_P']:.2e}",
                  {k: m[k] for k in ("C0_nodal_gtol", "C1_scipy_test", "C2_rel_dual_1e-6", "C3_gmw", "first_order_converged")},
                  f"dxi={m.get('max_abs_xi_R_difference_vs_P-MAP')}")
        print(t, "GN cond", agg["map"][t]["gauss_newton_P"]["condition_number"])


if __name__ == "__main__":
    main()
