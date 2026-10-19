"""Fail-fast remote smoke gate for both manufactured Darcy inverse PINNs.

Execute on CUDA only; never runs training when imported.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


BENCHMARKS = ("constant", "spatial", "spatial_lbfgs")
LBFGS_SMOKE_ITERATIONS = 20


def command(benchmark, iterations, output):
    if benchmark not in BENCHMARKS:
        raise ValueError("unknown benchmark")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    module = ("remote.run_deepxde_inverse" if benchmark == "constant"
              else "remote.run_deepxde_inverse_spatial")
    extra = (["--lbfgs-maxiter", str(LBFGS_SMOKE_ITERATIONS)]
             if benchmark == "spatial_lbfgs" else [])
    return [sys.executable, "-m", module, "--iterations", str(iterations),
            "--seeds", "42", *extra, "--output", str(output)]


def validate_record(benchmark, record):
    import math
    if record.get("seed") != 42:
        raise ValueError("incorrect seed")
    if record.get("status") != "ok":
        raise ValueError(f"run status {record.get('status')!r}: {record.get('error', '')}")
    keys = (("k_estimated", "relative_k_error", "pressure_relative_l2", "wall_seconds")
            if benchmark == "constant" else
            ("alpha_estimated", "beta_estimated", "pressure_relative_l2",
             "permeability_relative_l2", "wall_seconds"))
    if not all(isinstance(record.get(key), (int, float))
               and math.isfinite(record[key]) for key in keys):
        raise ValueError("non-finite or missing metrics")
    if benchmark == "constant" and record["k_estimated"] <= 0:
        raise ValueError("non-positive estimated permeability")
    if benchmark == "spatial_lbfgs":
        lbfgs = record.get("lbfgs") or {}
        if not (lbfgs.get("function_evaluations", 0) > 0 and lbfgs.get("finite_history")
                and "adam_stage" in record
                and math.isfinite(lbfgs.get("final_total_loss", math.nan))):
            raise ValueError("L-BFGS refinement did not run or produced non-finite values")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output-dir", default="results/deepxde_smoke")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--include-lbfgs", action="store_true",
                        help="also smoke-test the explicit adam+lbfgs spatial configuration")
    args = parser.parse_args()
    if os.environ.get("DDE_BACKEND") != "pytorch":
        raise RuntimeError("Set DDE_BACKEND=pytorch before importing DeepXDE")
    if args.timeout_seconds < 1:
        raise ValueError("timeout must be positive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for remote inverse PINN smoke tests")
    benchmarks = BENCHMARKS if args.include_lbfgs else BENCHMARKS[:2]
    for benchmark in benchmarks:
        output = Path(args.output_dir) / f"{benchmark}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        result = subprocess.run(command(benchmark, args.iterations, output),
                                text=True, capture_output=True,
                                timeout=args.timeout_seconds, check=False)
        if result.returncode:
            print(result.stdout[-4000:], file=sys.stderr)
            print(result.stderr[-4000:], file=sys.stderr)
            raise RuntimeError(f"{benchmark} inverse smoke failed")
        rows = json.loads(output.read_text())["records"]
        if len(rows) != 1:
            raise ValueError("expected exactly one result")
        validate_record(benchmark, rows[0])
        print(f"PASS {benchmark} inverse smoke", flush=True)


if __name__ == "__main__":
    main()
