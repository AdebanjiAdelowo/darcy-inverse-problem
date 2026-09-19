"""Noise-sensitivity study: reconstruction quality vs. observation noise
level, at a fixed regularisation strength and sensor network.

Usage: python scripts/noise_study.py --config smoke|local|full
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
from src.experiment import run_reconstruction

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    rc = cfg["noise_study"]

    sensors = grid_sensors(rc["sensor_grid"], rc["sensor_grid"])
    truth_fn = smooth_truth(seed=1)

    noise_levels = rc["relative_noise_levels"]
    m_errors, misfit_per_obs, p_errors = [], [], []
    lines = [f"Noise study, config={args.config}, alpha={rc['alpha']}, n_sensors={len(sensors)}",
             f"{'rel_noise':>10} {'rel_m_error':>12} {'misfit/n_sensors':>18} {'n_iter':>7}"]

    for noise in noise_levels:
        # separate data seed per noise level so different noise levels use
        # independent noise draws, but the same truth field and sensors
        result = run_reconstruction(
            truth_fn, sensors, rc["truth_mesh_n"], rc["inv_mesh_n"], rc["sensor_width"],
            noise, rc["alpha"], reg_type="h1", data_seed=100, max_iter=rc["max_iter"],
        )
        m_err = result.rel_m_error
        misfit_normalized = result.misfit_final / len(sensors)
        m_errors.append(m_err)
        misfit_per_obs.append(misfit_normalized)
        line = f"{noise:10.3f} {m_err:12.4f} {misfit_normalized:18.4e} {result.history.n_iterations:7d}"
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"noise_study_{args.config}.txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(noise_levels, m_errors, "o-")
    ax.set_xlabel("relative observation noise")
    ax.set_ylabel(r"relative error $\|m_{rec}-m_{true}\|/\|m_{true}\|$")
    ax.set_title(f"Reconstruction error vs. noise level ({args.config})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"noise_study_{args.config}.png", dpi=150)
    print(f"\nWrote results/noise_study_{args.config}.txt and figures/noise_study_{args.config}.png")


if __name__ == "__main__":
    main()
