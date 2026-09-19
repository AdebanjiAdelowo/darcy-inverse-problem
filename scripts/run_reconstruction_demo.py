"""Headline reconstruction demo: smooth and structured truths, moderate
sensor network, moderate noise, a single fixed regularisation strength
(chosen from the regularisation study -- see scripts/regularization_study.py
and the README; NOT re-tuned per truth here).

Usage: python scripts/run_reconstruction_demo.py --config smoke|local|full
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.truth_fields import smooth_truth, structured_truth
from src.observation import grid_sensors
from src.experiment import run_reconstruction

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def triangulation(dm):
    x = dm.mesh.geometry.x
    cells = dm.mesh.geometry.dofmap.reshape(-1, 3)
    return mtri.Triangulation(x[:, 0], x[:, 1], cells)


def plot_reconstruction(name: str, result, sensors, cfg_name: str):
    dm = result.inv_dm
    tri = triangulation(dm)
    vmin = min(result.m_true_on_inv_mesh.min(), result.m_rec.min())
    vmax = max(result.m_true_on_inv_mesh.max(), result.m_rec.max())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, field, title in [
        (axes[0], result.m_true_on_inv_mesh, "true $m=\\log k$ (on inversion mesh)"),
        (axes[1], result.m_rec, "reconstructed $m$"),
        (axes[2], result.m_rec - result.m_true_on_inv_mesh, "reconstruction error"),
    ]:
        if ax is axes[2]:
            e_max = np.max(np.abs(field))
            im = ax.tricontourf(tri, field, levels=40, cmap="RdBu_r", vmin=-e_max, vmax=e_max)
        else:
            im = ax.tricontourf(tri, field, levels=40, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.scatter(sensors[:, 0], sensors[:, 1], c="red", s=12, marker="x", label="sensors")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{name} truth reconstruction (rel. m error = {result.rel_m_error:.3f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"reconstruction_{name}_{cfg_name}.png", dpi=150)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    it = np.arange(len(result.history.objective))
    axes2[0].semilogy(it, result.history.objective, label="objective $J$")
    axes2[0].semilogy(it, result.history.misfit, label="data misfit")
    axes2[0].semilogy(it, result.history.reg, label="regularisation")
    axes2[0].set_xlabel("L-BFGS iteration")
    axes2[0].set_ylabel("value")
    axes2[0].set_title("Optimisation convergence")
    axes2[0].legend(fontsize=8)
    axes2[0].grid(alpha=0.3, which="both")

    axes2[1].semilogy(it, result.history.grad_norm)
    axes2[1].set_xlabel("L-BFGS iteration")
    axes2[1].set_ylabel(r"$\|\nabla J\|$")
    axes2[1].set_title("Gradient norm")
    axes2[1].grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / f"convergence_{name}_{cfg_name}.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    rc = cfg["reconstruction_demo"]

    sensors = grid_sensors(rc["sensor_grid"], rc["sensor_grid"])
    lines = [f"Reconstruction demo, config={args.config}",
             f"sensors: {rc['sensor_grid']}x{rc['sensor_grid']} grid ({len(sensors)} total), "
             f"truth_mesh_n={rc['truth_mesh_n']}, inv_mesh_n={rc['inv_mesh_n']}, "
             f"relative_noise={rc['relative_noise']}, alpha={rc['alpha']}, reg_type=h1"]

    for name, truth_fn in [("smooth", smooth_truth(seed=1)), ("structured", structured_truth())]:
        t0 = time.perf_counter()
        result = run_reconstruction(
            truth_fn, sensors, rc["truth_mesh_n"], rc["inv_mesh_n"], rc["sensor_width"],
            rc["relative_noise"], rc["alpha"], reg_type="h1", data_seed=7, max_iter=rc["max_iter"],
        )
        elapsed = time.perf_counter() - t0
        line = (f"{name}: rel_m_error={result.rel_m_error:.4f}, final_misfit={result.misfit_final:.4e}, "
                f"final_reg={result.reg_final:.4e}, n_iter={result.history.n_iterations}, "
                f"converged={result.history.converged}, time={elapsed:.2f}s")
        print(line)
        lines.append(line)
        plot_reconstruction(name, result, sensors, args.config)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"reconstruction_demo_{args.config}.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote results/reconstruction_demo_{args.config}.txt and figures/reconstruction_*.png")


if __name__ == "__main__":
    main()
