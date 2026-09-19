"""Prior-sensitivity study: how much does the posterior change when the
prior's correlation length (gamma) or KL truncation dimension (r) changes?

This matters because, with sparse observations, Bayesian inverse solutions
can be strongly prior-informed rather than purely data-driven -- shown
here directly rather than asserted.

Usage: python scripts/bayesian_prior_sensitivity.py --config smoke|local|full
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.truth_fields import smooth_truth
from src.observation import grid_sensors
from src.bayesian_experiment import run_bayesian_experiment, integrated_posterior_variance

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def triangulation(dm):
    x = dm.mesh.geometry.x
    cells = dm.mesh.geometry.dofmap.reshape(-1, 3)
    return mtri.Triangulation(x[:, 0], x[:, 1], cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    rc = cfg["bayesian_prior_sensitivity"]
    sensors = grid_sensors(rc["sensor_grid"], rc["sensor_grid"])
    truth_fn = smooth_truth(seed=1)

    variants = rc["prior_variants"]
    results = []
    lines = [f"Bayesian prior-sensitivity study, config={args.config}, n_sensors={len(sensors)}",
             f"{'label':>16} {'r':>4} {'gamma':>7} {'delta':>7} {'captured_var':>13} "
             f"{'integrated_post_var':>20} {'rel_post_mean_error':>20}"]

    for variant in variants:
        result = run_bayesian_experiment(
            truth_fn, sensors, rc["truth_mesh_n"], rc["inv_mesh_n"], rc["sensor_width"],
            rc["relative_noise"], variant["r"], variant["gamma"], variant["delta"],
            rc["n_samples"], rc["beta"], rc["burn_in"], map_max_iter=rc["map_max_iter"],
            data_seed=400, mcmc_seed=1,
        )
        ipv = integrated_posterior_variance(result)
        err = np.linalg.norm(result.m_post_mean - result.m_true_on_inv_mesh) / \
            np.linalg.norm(result.m_true_on_inv_mesh)
        captured = result.prior.captured_variance_fraction()
        results.append((variant, result))
        line = (f"{variant['label']:>16} {variant['r']:4d} {variant['gamma']:7.3f} "
                f"{variant['delta']:7.3f} {captured:13.4f} {ipv:20.5f} {err:20.4f}")
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"bayesian_prior_sensitivity_{args.config}.txt").write_text("\n".join(lines) + "\n")

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, (variant, result) in enumerate(results):
        tri = triangulation(result.bp.dm)
        im0 = axes[0, col].tricontourf(tri, result.m_post_mean, levels=40, cmap="viridis")
        axes[0, col].set_title(f"{variant['label']}\npost. mean", fontsize=9)
        axes[0, col].set_aspect("equal")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046)

        im1 = axes[1, col].tricontourf(tri, result.m_post_std, levels=40, cmap="magma")
        axes[1, col].set_title("post. std", fontsize=9)
        axes[1, col].set_aspect("equal")
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046)

    fig.suptitle(f"Prior sensitivity: posterior mean (top) and std (bottom) ({args.config})")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"bayesian_prior_sensitivity_{args.config}.png", dpi=150)
    print(f"\nWrote results/bayesian_prior_sensitivity_{args.config}.txt and figure")


if __name__ == "__main__":
    main()
