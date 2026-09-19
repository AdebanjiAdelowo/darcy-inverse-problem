"""Sensor-density and placement study: reconstruction quality vs. number of
sensors, comparing structured (grid) and random placement at matched
sensor counts.

Usage: python scripts/sensor_study.py --config smoke|local|full
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.truth_fields import smooth_truth
from src.observation import grid_sensors, random_sensors
from src.experiment import run_reconstruction

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    rc = cfg["sensor_study"]

    truth_fn = smooth_truth(seed=1)
    n_random_trials = rc.get("n_random_trials", 1)

    grid_counts, grid_errors = [], []
    random_counts, random_errors_mean, random_errors_std = [], [], []
    lines = [f"Sensor study, config={args.config}, alpha={rc['alpha']}",
             f"{'n_sensors':>10} {'grid_error':>12} {'random_error_mean':>18} {'random_error_std':>18}"]

    for ng in rc["sensor_grids"]:
        gsensors = grid_sensors(ng, ng)
        n = len(gsensors)
        result = run_reconstruction(
            truth_fn, gsensors, rc["truth_mesh_n"], rc["inv_mesh_n"], rc["sensor_width"],
            rc["relative_noise"], rc["alpha"], reg_type="h1", data_seed=200, max_iter=rc["max_iter"],
        )
        grid_counts.append(n)
        grid_errors.append(result.rel_m_error)

        r_errs = []
        for trial in range(n_random_trials):
            rsensors = random_sensors(n, seed=300 + trial)
            r_result = run_reconstruction(
                truth_fn, rsensors, rc["truth_mesh_n"], rc["inv_mesh_n"], rc["sensor_width"],
                rc["relative_noise"], rc["alpha"], reg_type="h1", data_seed=200 + trial,
                max_iter=rc["max_iter"],
            )
            r_errs.append(r_result.rel_m_error)
        random_counts.append(n)
        random_errors_mean.append(float(np.mean(r_errs)))
        random_errors_std.append(float(np.std(r_errs)))

        line = (f"{n:10d} {grid_errors[-1]:12.4f} {random_errors_mean[-1]:18.4f} "
                f"{random_errors_std[-1]:18.4f}")
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"sensor_study_{args.config}.txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # "grid" here means regularly-spaced sensor PLACEMENT, unrelated to the
    # "structured" (localised-inclusions) TRUTH field used elsewhere in
    # this project -- the two are different axes and the label is kept
    # unambiguous on purpose.
    ax.plot(grid_counts, grid_errors, "o-", label="regular grid placement")
    ax.errorbar(random_counts, random_errors_mean, yerr=random_errors_std, fmt="s--",
                label=f"random ({n_random_trials} trials, mean +/- std)", capsize=3)
    ax.set_xlabel("number of sensors")
    ax.set_ylabel(r"relative error $\|m_{rec}-m_{true}\|/\|m_{true}\|$")
    ax.set_title(f"Reconstruction error vs. sensor count/placement ({args.config})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"sensor_study_{args.config}.png", dpi=150)
    print(f"\nWrote results/sensor_study_{args.config}.txt and figures/sensor_study_{args.config}.png")


if __name__ == "__main__":
    main()
