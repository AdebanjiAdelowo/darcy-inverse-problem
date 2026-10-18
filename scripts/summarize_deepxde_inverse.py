"""Summarise DeepXDE inverse-PINN result JSON files (no training, NumPy only).

Computes per-metric descriptive statistics over seeds and the pre-registered
recovery label from docs/DEEPXDE_REMOTE_RESULTS.md, and prints a Markdown table.

    python scripts/summarize_deepxde_inverse.py results/deepxde_remote/<run>/*.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

RECOVERY_THRESHOLD = 1e-2  # pre-registered; relative parameter error

METRICS = {
    "constant_k": ["k_estimated", "relative_k_error", "pressure_relative_l2", "wall_seconds"],
    "spatial_k": ["alpha_estimated", "beta_estimated", "parameter_relative_error",
                  "pressure_relative_l2", "permeability_relative_l2", "wall_seconds"],
}
PARAMETER_ERROR = {"constant_k": "relative_k_error", "spatial_k": "parameter_relative_error"}


def describe(values):
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def recovery_label(errors, threshold=RECOVERY_THRESHOLD):
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return "no successful seeds"
    if np.all(errors < threshold):
        return "recovered"
    if np.median(errors) < threshold:
        return "partially recovered"
    return "not recovered"


def summarize(payload):
    benchmark = payload["config"]["benchmark"]
    records = payload["records"]
    ok = [r for r in records if r.get("status") == "ok"]
    summary = {
        "benchmark": benchmark, "config": payload["config"],
        "device": payload["environment"]["device"],
        "seeds_total": len(records), "seeds_ok": len(ok),
        "failed": [{"seed": r["seed"], "status": r.get("status"), "error": r.get("error")}
                   for r in records if r.get("status") != "ok"],
        "statistics": {m: describe([r[m] for r in ok]) for m in METRICS[benchmark]} if ok else {},
        "recovery": recovery_label([r[PARAMETER_ERROR[benchmark]] for r in ok]),
    }
    if len(ok) < len(records):
        summary["recovery"] +=f" ({len(records) - len(ok)} seed(s) failed)"
    return summary


def markdown(name, summary):
    lines = [f"**{name}** ({summary['benchmark']}, {summary['seeds_ok']}/{summary['seeds_total']} "
             f"seeds ok, {summary['device']}): {summary['recovery']}", "",
             "| metric | mean | std | median | min | max |", "|---|---|---|---|---|---|"]
    for metric, s in summary["statistics"].items():
        lines.append(f"| {metric} | {s['mean']:.4g} | {s['std']:.3g} | {s['median']:.4g} | "
                     f"{s['min']:.4g} | {s['max']:.4g} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json", help="also write all summaries to this file")
    args = parser.parse_args(argv)
    summaries = {}
    for f in args.files:
        summaries[Path(f).stem] = summarize(json.loads(Path(f).read_text()))
        print(markdown(Path(f).stem, summaries[Path(f).stem]) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
