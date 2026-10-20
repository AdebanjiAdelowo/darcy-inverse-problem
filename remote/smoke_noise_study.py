"""Fail-fast CUDA smoke gate for every new noise-study configuration.

Short runs only (2 Adam iterations, 20 L-BFGS iterations). Checks exit status,
finite estimates and losses for every method, function evaluations > 0, the
loss-term counts, and float64 dtypes where requested. Never trains on import.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from src.deepxde_noise_methods import METHODS

CASES = {
    "methods_sigma0p01": ["--noise-std", "0.01", "--adam-snapshots", "1"],
    "methods_sigma0": ["--noise-std", "0.0"],
    "float64_sigma0p01": ["--noise-std", "0.01", "--float64", "--methods", "baseline"],
}


def command(case, output):
    return [sys.executable, "-m", "remote.run_deepxde_noise_study", "--tuples", "42:42:42",
            "--iterations", "2", "--lbfgs-maxiter", "20", *CASES[case], "--output", str(output)]


def validate(case, payload):
    (record,) = payload["records"]
    if record.get("status") != "ok":
        raise ValueError(f"{case}: status {record.get('status')}: {record.get('error', '')}")
    if record["loss_counts"] != [576, 64, 128]:
        raise ValueError(f"{case}: unexpected loss counts {record['loss_counts']}")
    expected = ["baseline"] if "--methods" in CASES[case] else list(METHODS)
    budgets = [1, 2] if "--adam-snapshots" in CASES[case] else [2]
    seen = [(r["adam_iterations"], r["method"]) for r in record["refinements"]]
    if seen != [(b, m) for b in budgets for m in expected]:
        raise ValueError(f"{case}: refinements {seen}")
    for r in record["refinements"]:
        values = [r["alpha_estimated"], r["beta_estimated"], r["pressure_relative_l2"],
                  r["lbfgs"]["final_total_loss"], *r["lbfgs"]["final_raw_loss_components"]]
        if not all(math.isfinite(v) for v in values) or r["lbfgs"]["function_evaluations"] < 1:
            raise ValueError(f"{case}: non-finite or empty refinement {r['method']}")
    want = "torch.float64" if "--float64" in CASES[case] else "torch.float32"
    if record["dtypes"]["network"] != want or record["dtypes"]["external_variables"] != want \
            or record["dtypes"]["losses"] != want:
        raise ValueError(f"{case}: dtypes {record['dtypes']}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/noise_smoke")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if os.environ.get("DDE_BACKEND") != "pytorch":
        raise RuntimeError("Set DDE_BACKEND=pytorch")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the noise-study smoke gate")
    for case in CASES:
        out = Path(args.output_dir) / f"{case}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        result = subprocess.run(command(case, out), text=True, capture_output=True,
                                timeout=args.timeout_seconds, check=False)
        if result.returncode:
            print(result.stdout[-4000:], result.stderr[-4000:], file=sys.stderr)
            raise RuntimeError(f"{case} smoke failed")
        validate(case, json.loads(out.read_text()))
        print(f"PASS {case}", flush=True)


if __name__ == "__main__":
    main()
