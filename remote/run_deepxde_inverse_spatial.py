"""Remote-only multi-seed spatially varying Darcy inverse PINN experiment."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from src.deepxde_inverse_spatial_k import train, exact_pressure, exact_log_permeability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    parser.add_argument("--observations", type=int, default=128)
    parser.add_argument("--output", default="results/deepxde_inverse_spatial_remote.json")
    args = parser.parse_args()
    if os.environ.get("DDE_BACKEND") != "pytorch":
        raise RuntimeError("Set DDE_BACKEND=pytorch")
    if args.iterations < 1 or args.observations < 1:
        raise ValueError("iterations and observations must be positive")

    import deepxde as dde
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA GPU required; refusing heavy CPU/local training")

    grid_axis = np.linspace(0.0, 1.0, 41)
    xx, yy = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    grid = np.stack((xx.ravel(), yy.ravel()), axis=1)
    p_reference = exact_pressure(grid)
    logk_reference = exact_log_permeability(grid)
    records = []
    for seed in args.seeds:
        start = time.perf_counter()
        model, estimates = train(seed=seed, iterations=args.iterations,
                                 observations=args.observations)
        prediction = model.predict(grid)
        pressure_error = float(np.linalg.norm(prediction-p_reference) /
                               np.linalg.norm(p_reference))
        logk_estimated = exact_log_permeability(
            grid, alpha=estimates["alpha"], beta=estimates["beta"])
        k_error = float(np.linalg.norm(np.exp(logk_estimated)-np.exp(logk_reference)) /
                        np.linalg.norm(np.exp(logk_reference)))
        record = {
            "seed": seed,
            "alpha_true": 0.5, "beta_true": -0.3,
            "alpha_estimated": estimates["alpha"],
            "beta_estimated": estimates["beta"],
            "pressure_relative_l2": pressure_error,
            "permeability_relative_l2": k_error,
            "wall_seconds": time.perf_counter()-start,
            "device": str(torch.cuda.get_device_name(0)),
            "deepxde_version": dde.__version__,
            "torch_version": torch.__version__,
            "iterations": args.iterations,
            "observations": args.observations,
        }
        if not np.isfinite([record["alpha_estimated"], record["beta_estimated"],
                            pressure_error, k_error]).all():
            raise ValueError(f"Non-finite result for seed {seed}")
        records.append(record)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2)+"\n")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
