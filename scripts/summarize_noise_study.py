"""Summarise noise-study result files against the classical reference estimator.

For every (tuple, Adam budget, method) entry this recomputes, on CPU and on the
identical observations, the finite-difference maximum-likelihood estimate
(src/fd_reference.py) and reports

* errors against the truth (parameters, pressure, permeability field);
* the gap to the MLE on the same data, ||theta_method - theta_MLE|| / ||theta||,
  which separates estimator/optimiser error from noise-induced error;
* per-method statistics and paired differences against ``baseline`` with a
  t-based 95% interval (descriptive at these sample sizes);
* the spread of the error within each random-effect arm (noise only, layout
  only, network only), identified from the tuple pattern around 42:42:42.

    python scripts/summarize_noise_study.py results/.../noise_sigma0p01.json ... --json out.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.deepxde_inverse_spatial_k import ALPHA_TRUE, BETA_TRUE, observation_data
from src.fd_reference import DarcyFD

THETA = np.array([ALPHA_TRUE, BETA_TRUE])
NORM = float(np.linalg.norm(THETA))
CENTER = (42, 42, 42)


def arm(tuple_):
    n, l, e = tuple_
    if tuple_ == CENTER:
        return "center"
    if (n, l) == (42, 42):
        return "noise_only"
    if (n, e) == (42, 42):
        return "layout_only"
    if (l, e) == (42, 42):
        return "network_only"
    return "other"


def describe(values):
    v = np.asarray(values, dtype=float)
    out = {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
           "min": float(v.min()), "max": float(v.max()),
           "std": float(v.std(ddof=1)) if v.size > 1 else 0.0}
    if v.size > 1:
        half = stats.t.ppf(0.975, v.size - 1) * out["std"] / np.sqrt(v.size)
        out["t95"] = [out["mean"] - half, out["mean"] + half]
    return out


def mle_cache(payloads, fd):
    cache = {}
    for payload in payloads:
        cfg = payload["config"]
        for r in payload["records"]:
            if r.get("status") != "ok":
                continue
            key = (r["layout_seed"], r["noise_seed"], cfg["observations"], cfg["noise_std"])
            if key not in cache:
                xy, values = observation_data(key[0], key[2], key[3],
                                              noise_seed=key[1] if key[3] > 0 else None)
                cache[key] = fd.mle(xy, values)[0]
    return cache


def summarize(payload, mles):
    cfg = payload["config"]
    rows = []
    for r in payload["records"]:
        if r.get("status") != "ok":
            rows.append({"tuple": [r.get("network_seed"), r.get("layout_seed"),
                                   r.get("noise_seed")], "status": r.get("status"),
                         "error": r.get("error")})
            continue
        t = (r["network_seed"], r["layout_seed"], r["noise_seed"])
        mle = mles[(t[1], t[2], cfg["observations"], cfg["noise_std"])]
        for e in r["refinements"]:
            est = np.array([e["alpha_estimated"], e["beta_estimated"]])
            rows.append({
                "tuple": list(t), "arm": arm(t), "status": "ok",
                "adam_iterations": e["adam_iterations"], "method": e["method"],
                "alpha": e["alpha_estimated"], "beta": e["beta_estimated"],
                "parameter_relative_error": e["parameter_relative_error"],
                "pressure_relative_l2": e["pressure_relative_l2"],
                "permeability_relative_l2": e["permeability_relative_l2"],
                "mle_alpha": float(mle[0]), "mle_beta": float(mle[1]),
                "mle_relative_error": float(np.linalg.norm(mle - THETA) / NORM),
                "gap_to_mle_relative": float(np.linalg.norm(est - mle) / NORM),
                "obs_mse_over_sigma2": (e["lbfgs"]["final_raw_loss_components"][2]
                                        / cfg["noise_std"] ** 2 if cfg["noise_std"] > 0 else None),
                "lbfgs_iterations": e["lbfgs"]["lbfgs_iterations"],
                "function_evaluations": e["lbfgs"]["function_evaluations"],
                "stop_reason": e["lbfgs"]["stop_reason"],
                "adam_seconds": e["adam_seconds"], "lbfgs_seconds": e["lbfgs_seconds"],
                "total_seconds": e["total_seconds"], "peak_memory_bytes": e["peak_memory_bytes"],
            })
    ok = [x for x in rows if x["status"] == "ok"]
    groups = sorted({(x["adam_iterations"], x["method"]) for x in ok})
    metrics = ("parameter_relative_error", "pressure_relative_l2", "permeability_relative_l2",
               "gap_to_mle_relative", "mle_relative_error", "lbfgs_seconds", "total_seconds",
               "lbfgs_iterations", "function_evaluations")
    per_method = {}
    for budget, method in groups:
        sel = [x for x in ok if (x["adam_iterations"], x["method"]) == (budget, method)]
        entry = {m: describe([x[m] for x in sel]) for m in metrics}
        entry["alpha"] = describe([x["alpha"] for x in sel])
        entry["beta"] = describe([x["beta"] for x in sel])
        entry["fraction_below_1e-2"] = float(np.mean([x["parameter_relative_error"] < 1e-2
                                                      for x in sel]))
        entry["stop_reasons"] = sorted({x["stop_reason"][:7] for x in sel})
        if cfg["noise_std"] > 0:
            entry["obs_mse_over_sigma2"] = describe([x["obs_mse_over_sigma2"] for x in sel])
        entry["by_arm"] = {}
        for a in ("noise_only", "layout_only", "network_only"):
            arm_sel = [x for x in sel if x["arm"] in (a, "center")]
            if len(arm_sel) > 1:
                entry["by_arm"][a] = {m: describe([x[m] for x in arm_sel])
                                      for m in ("parameter_relative_error", "alpha", "beta",
                                                "gap_to_mle_relative")}
        base = {tuple(x["tuple"]): x for x in ok
                if (x["adam_iterations"], x["method"]) == (budget, "baseline")}
        if method != "baseline" and base:
            diffs = [x["parameter_relative_error"] - base[tuple(x["tuple"])]["parameter_relative_error"]
                     for x in sel if tuple(x["tuple"]) in base]
            gaps = [x["gap_to_mle_relative"] - base[tuple(x["tuple"])]["gap_to_mle_relative"]
                    for x in sel if tuple(x["tuple"]) in base]
            entry["paired_minus_baseline"] = {
                "parameter_relative_error": describe(diffs),
                "gap_to_mle_relative": describe(gaps),
                "improved_count": int(sum(d < 0 for d in diffs)), "pairs": len(diffs)}
        per_method[f"{budget}:{method}"] = entry
    return {"config": cfg, "environment": payload["environment"],
            "failed": [x for x in rows if x["status"] != "ok"],
            "per_method": per_method, "rows": ok}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json", required=True)
    parser.add_argument("--fd-n", type=int, default=200)
    args = parser.parse_args(argv)
    payloads = {Path(f).stem: json.loads(Path(f).read_text()) for f in args.files}
    mles = mle_cache(payloads.values(), DarcyFD(args.fd_n))
    out = {name: summarize(p, mles) for name, p in payloads.items()}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
    for name, s in out.items():
        print(f"== {name} (sigma {s['config']['noise_std']}, N {s['config']['observations']})")
        for key, e in s["per_method"].items():
            p = e["paired_minus_baseline"]["parameter_relative_error"] if \
                "paired_minus_baseline" in e else None
            print(f"  {key:34s} err median {e['parameter_relative_error']['median']:.3e} "
                  f"mean {e['parameter_relative_error']['mean']:.3e} "
                  f"gap-to-MLE median {e['gap_to_mle_relative']['median']:.3e}"
                  + (f" | paired diff mean {p['mean']:+.2e}" if p else ""))


if __name__ == "__main__":
    main()
