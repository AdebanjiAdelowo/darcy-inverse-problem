"""Bayesian sensor-density study: how does posterior uncertainty (not just
reconstruction error) change with the number of sensors?

Usage: python scripts/bayesian_sensor_study.py --config smoke|local|full
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
from src.observation import grid_sensors
from src.bayesian_experiment import run_bayesian_experiment, integrated_posterior_variance

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    rc = cfg["bayesian_sensor_study"]
    truth_fn = smooth_truth(seed=1)

    n_sensors_list, ipv_list, err_list, accept_list = [], [], [], []
    lines = [f"Bayesian sensor study, config={args.config}",
             f"{'n_sensors':>10} {'integrated_post_var':>20} {'rel_post_mean_error':>20} "
             f"{'acceptance':>10}"]

    for ng in rc["sensor_grids"]:
        sensors = grid_sensors(ng, ng)
        result = run_bayesian_experiment(
            truth_fn, sensors, rc["truth_mesh_n"], rc["inv_mesh_n"], rc["sensor_width"],
            rc["relative_noise"], rc["prior_r"], rc["prior_gamma"], rc["prior_delta"],
            rc["n_samples"], rc["beta"], rc["burn_in"], map_max_iter=rc["map_max_iter"],
            data_seed=200, mcmc_seed=1,
        )
        ipv = integrated_posterior_variance(result)
        err = np.linalg.norm(result.m_post_mean - result.m_true_on_inv_mesh) / \
            np.linalg.norm(result.m_true_on_inv_mesh)
        n_sensors_list.append(len(sensors))
        ipv_list.append(ipv)
        err_list.append(err)
        accept_list.append(result.pcn.acceptance_rate)
        line = f"{len(sensors):10d} {ipv:20.5f} {err:20.4f} {result.pcn.acceptance_rate:10.3f}"
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"bayesian_sensor_study_{args.config}.txt").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(n_sensors_list, ipv_list, "o-")
    axes[0].set_xlabel("number of sensors")
    axes[0].set_ylabel(r"integrated posterior variance $\int \mathrm{Var}[m|y]\,dx$")
    axes[0].set_title("Uncertainty vs. sensor count")
    axes[0].grid(alpha=0.3)

    axes[1].plot(n_sensors_list, err_list, "o-", color="green")
    axes[1].set_xlabel("number of sensors")
    axes[1].set_ylabel("relative posterior-mean error")
    axes[1].set_title("Error vs. sensor count")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Bayesian sensor-density study ({args.config})")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"bayesian_sensor_study_{args.config}.png", dpi=150)
    print(f"\nWrote results/bayesian_sensor_study_{args.config}.txt and figure")


if __name__ == "__main__":
    main()
