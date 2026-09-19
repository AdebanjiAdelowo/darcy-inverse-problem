"""Headline Bayesian posterior demo: MAP vs. posterior mean vs. truth,
posterior uncertainty, MCMC diagnostics, credible intervals, and posterior
predictive checks, for both the smooth and structured truths.

Usage: python scripts/run_posterior_demo.py --config smoke|local|full
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
from src.truth_fields import smooth_truth, structured_truth
from src.observation import grid_sensors
from src.bayesian_experiment import run_bayesian_experiment, integrated_posterior_variance
from src.pcn_sampler import autocorrelation, effective_sample_size

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def triangulation(dm):
    x = dm.mesh.geometry.x
    cells = dm.mesh.geometry.dofmap.reshape(-1, 3)
    return mtri.Triangulation(x[:, 0], x[:, 1], cells)


def plot_fields(name: str, result, sensors, cfg_name: str):
    dm = result.bp.dm
    tri = triangulation(dm)
    m_true = result.m_true_on_inv_mesh
    vmin = min(m_true.min(), result.m_map.min(), result.m_post_mean.min())
    vmax = max(m_true.max(), result.m_map.max(), result.m_post_mean.max())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    panels = [
        (axes[0, 0], m_true, "true $m$", "viridis", vmin, vmax),
        (axes[0, 1], result.m_map, "MAP $m$", "viridis", vmin, vmax),
        (axes[0, 2], result.m_post_mean, "posterior mean $E[m|y]$", "viridis", vmin, vmax),
        (axes[1, 0], result.m_post_std, "posterior std $\\mathrm{sd}[m|y]$", "magma", None, None),
        (axes[1, 1], result.m_map - m_true, "MAP error", "RdBu_r", None, None),
        (axes[1, 2], result.m_post_mean - m_true, "posterior mean error", "RdBu_r", None, None),
    ]
    for ax, field, title, cmap, vlo, vhi in panels:
        if cmap == "RdBu_r":
            e_max = np.max(np.abs(field))
            im = ax.tricontourf(tri, field, levels=40, cmap=cmap, vmin=-e_max, vmax=e_max)
        else:
            im = ax.tricontourf(tri, field, levels=40, cmap=cmap, vmin=vlo, vmax=vhi)
        ax.scatter(sensors[:, 0], sensors[:, 1], c="red", s=10, marker="x")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)

    rel_map = np.linalg.norm(result.m_map - m_true) / np.linalg.norm(m_true)
    rel_mean = np.linalg.norm(result.m_post_mean - m_true) / np.linalg.norm(m_true)
    fig.suptitle(f"{name} truth: Bayesian posterior (rel. MAP error={rel_map:.3f}, "
                 f"rel. posterior-mean error={rel_mean:.3f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"posterior_fields_{name}_{cfg_name}.png", dpi=150)
    return rel_map, rel_mean


def plot_diagnostics(name: str, result, cfg_name: str):
    samples = result.pcn.samples
    raw = result.pcn.raw_chain
    burn_in = result.pcn.burn_in

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    dims_to_show = [1, 2, 3] if samples.shape[1] > 3 else list(range(samples.shape[1]))

    for i, d in enumerate(dims_to_show):
        axes[0, i].plot(raw[:, d], lw=0.4)
        axes[0, i].axvline(burn_in, color="r", linestyle="--", lw=1)
        axes[0, i].set_title(f"trace, KL dim {d}")
        axes[0, i].set_xlabel("iteration")

        acf = autocorrelation(samples[:, d], max_lag=min(200, len(samples) // 2 - 1))
        axes[1, i].plot(acf)
        axes[1, i].axhline(0, color="k", lw=0.5)
        axes[1, i].set_title(f"ACF, KL dim {d}")
        axes[1, i].set_xlabel("lag")

    fig.suptitle(f"{name} truth: MCMC diagnostics (acceptance={result.pcn.acceptance_rate:.3f}, "
                 f"mean ESS={result.ess_per_dim.mean():.1f}/{len(samples)})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"posterior_diagnostics_{name}_{cfg_name}.png", dpi=150)


def credible_intervals(result, n_points: int = 5):
    """90% credible intervals at n_points spatial locations (dof indices
    spread across the domain), checking truth coverage."""
    dm = result.bp.dm
    coords = dm.V_m.tabulate_dof_coordinates()
    # pick points roughly spread across the domain interior
    targets = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.75, 0.75]])[:n_points]
    idx = [np.argmin(np.sum((coords[:, :2] - t) ** 2, axis=1)) for t in targets]

    lines = [f"{'point':>14} {'true m':>9} {'post mean':>10} {'2.5%':>8} {'97.5%':>8} {'covered':>8}"]
    n_covered = 0
    for i, target in zip(idx, targets):
        vals = result.m_samples[:, i]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        truth_val = result.m_true_on_inv_mesh[i]
        covered = lo <= truth_val <= hi
        n_covered += int(covered)
        lines.append(f"({target[0]:.2f},{target[1]:.2f}) {truth_val:9.4f} {result.m_post_mean[i]:10.4f} "
                      f"{lo:8.4f} {hi:8.4f} {str(covered):>8}")
    lines.append(f"coverage: {n_covered}/{len(idx)} (single synthetic realisation -- "
                  f"NOT a calibration study; see README)")
    return lines


def posterior_predictive_check(result, n_check_samples: int = 20, seed: int = 999):
    """Solve the forward problem at n_check_samples posterior draws and
    compare predicted observations to the actual (noisy) data."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(result.pcn.samples), size=min(n_check_samples, len(result.pcn.samples)),
                      replace=False)
    preds = []
    for i in idx:
        xi = result.pcn.samples[i]
        p = result.bp.pressure(xi)
        preds.append(result.bp.obs.apply(p))
    preds = np.array(preds)
    pred_mean = preds.mean(axis=0)
    pred_lo, pred_hi = np.percentile(preds, [2.5, 97.5], axis=0)
    y = result.data.y_noisy
    coverage = np.mean((pred_lo <= y) & (y <= pred_hi))
    return preds, pred_mean, pred_lo, pred_hi, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    rc = cfg["bayesian_posterior"]

    sensors = grid_sensors(rc["sensor_grid"], rc["sensor_grid"])
    all_lines = [f"Bayesian posterior demo, config={args.config}",
                 f"sensors={len(sensors)}, truth_mesh_n={rc['truth_mesh_n']}, "
                 f"inv_mesh_n={rc['inv_mesh_n']}, relative_noise={rc['relative_noise']}, "
                 f"prior: r={rc['prior_r']}, gamma={rc['prior_gamma']}, delta={rc['prior_delta']}",
                 f"pCN: beta={rc['beta']}, burn_in={rc['burn_in']}, n_samples={rc['n_samples']}", ""]

    for name, truth_fn in [("smooth", smooth_truth(seed=1)), ("structured", structured_truth())]:
        result = run_bayesian_experiment(
            truth_fn, sensors, rc["truth_mesh_n"], rc["inv_mesh_n"], rc["sensor_width"],
            rc["relative_noise"], rc["prior_r"], rc["prior_gamma"], rc["prior_delta"],
            rc["n_samples"], rc["beta"], rc["burn_in"], map_max_iter=rc["map_max_iter"],
            data_seed=7, mcmc_seed=1,
        )
        rel_map, rel_mean = plot_fields(name, result, sensors, args.config)
        plot_diagnostics(name, result, args.config)
        ipv = integrated_posterior_variance(result)

        all_lines.append(f"=== {name} truth ===")
        all_lines.append(f"MAP converged={result.map_converged}, n_iter={result.map_n_iter}, "
                          f"xi_map[:5]={np.round(result.xi_map[:5], 3).tolist()}")
        all_lines.append(f"pCN acceptance rate: {result.pcn.acceptance_rate:.4f}")
        all_lines.append(f"ESS per KL dim: {np.round(result.ess_per_dim, 1).tolist()}")
        all_lines.append(f"relative MAP error: {rel_map:.4f}")
        all_lines.append(f"relative posterior-mean error: {rel_mean:.4f}")
        all_lines.append(f"integrated posterior variance: {ipv:.5f}")
        all_lines.append("")
        all_lines.extend(credible_intervals(result))
        all_lines.append("")

        preds, pred_mean, pred_lo, pred_hi, coverage = posterior_predictive_check(result)
        all_lines.append(f"posterior predictive: 95% interval coverage of observed data = {coverage:.3f} "
                          f"({len(result.data.y_noisy)} sensors)")
        all_lines.append("")

        fig, ax = plt.subplots(figsize=(6, 5))
        s_idx = np.arange(len(result.data.y_noisy))
        ax.errorbar(s_idx, pred_mean, yerr=[pred_mean - pred_lo, pred_hi - pred_mean], fmt="o",
                    label="posterior predictive (95%)", capsize=3, alpha=0.7)
        ax.scatter(s_idx, result.data.y_noisy, color="red", marker="x", label="observed $y$", zorder=5)
        ax.set_xlabel("sensor index")
        ax.set_ylabel("pressure")
        ax.set_title(f"{name} truth: posterior predictive check")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"posterior_predictive_{name}_{args.config}.png", dpi=150)
        plt.close(fig)

        np.savez(RESULTS_DIR / f"posterior_samples_{name}_{args.config}.npz",
                 xi_samples=result.pcn.samples, m_map=result.m_map, m_post_mean=result.m_post_mean,
                 m_post_std=result.m_post_std, m_true=result.m_true_on_inv_mesh,
                 eigenvalues=result.prior.eigenvalues)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"posterior_demo_{args.config}.txt").write_text("\n".join(all_lines) + "\n")
    for line in all_lines:
        print(line)
    print(f"\nWrote results/posterior_demo_{args.config}.txt and figures/posterior_*.png")


if __name__ == "__main__":
    main()
