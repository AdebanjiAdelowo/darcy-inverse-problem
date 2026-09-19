"""Regularisation-strength (alpha) study: L-curve, retrospective error vs.
alpha, and the discrepancy principle as a practical (truth-free) alpha
selection rule.

## Why not just pick the alpha that minimises the true error?

Minimising ||m_rec - m_true|| over alpha requires knowing m_true -- which,
by definition, is unavailable for a real inverse problem (we only have it
here because this is a synthetic/verification study). That curve is
reported below purely as RETROSPECTIVE EVALUATION, labelled as such, to
show what the (unreachable, in practice) best case looks like.

## A practical, truth-free alpha selection rule: the discrepancy principle

Morozov's discrepancy principle (Morozov, V. A. (1966), "On the solution
of functionally ill-posed problems in a Banach space"; see also Engl,
Hanke & Neubauer, 1996, Ch. 4.3) chooses alpha so that the data misfit at
convergence matches the STATISTICALLY EXPECTED misfit under the assumed
noise model: for n independent observations with the correct noise level,
E[sum_s ((Hp-y)_s/sigma_s)^2] = n. So: pick the largest alpha (most
regularisation, most stable) whose converged weighted misfit sum_s
((Hp-y)_s/sigma_s)^2 is still <= n_sensors (equivalently, misfit
J_misfit = 0.5*sum(...) <= 0.5*n_sensors). This uses only the assumed
noise level, not the unknown truth, and is reported alongside the L-curve.

Usage: python scripts/regularization_study.py --config smoke|local|full
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
from src.data_generation import generate_synthetic_data
from src.experiment import run_reconstruction

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    rc = cfg["regularization_study"]

    sensors = grid_sensors(rc["sensor_grid"], rc["sensor_grid"])
    truth_fn = smooth_truth(seed=1)

    # generate the data ONCE, reuse across all alpha values so the study
    # isolates the effect of alpha alone
    data = generate_synthetic_data(
        truth_fn, sensors, rc["truth_mesh_n"], rc["sensor_width"], rc["relative_noise"], seed=7
    )

    alphas = rc["alphas"]
    misfits, regs, m_errors = [], [], []
    lines = [f"Regularization study, config={args.config}, n_sensors={len(sensors)}, "
             f"relative_noise={rc['relative_noise']}",
             f"{'alpha':>10} {'misfit':>14} {'reg':>14} {'rel_m_error (retrospective)':>28} "
             f"{'n_iter':>7} {'converged':>10}"]
    for alpha in alphas:
        result = run_reconstruction(
            truth_fn, sensors, rc["truth_mesh_n"], rc["inv_mesh_n"], rc["sensor_width"],
            rc["relative_noise"], alpha, reg_type="h1", max_iter=rc["max_iter"], data=data,
        )
        misfits.append(result.misfit_final)
        regs.append(result.reg_final)
        m_errors.append(result.rel_m_error)
        line = (f"{alpha:10.2e} {result.misfit_final:14.4e} {result.reg_final:14.4e} "
                f"{result.rel_m_error:28.4f} {result.history.n_iterations:7d} "
                f"{str(result.history.converged):>10}")
        print(line)
        lines.append(line)

    n_sensors = len(sensors)
    discrepancy_target = 0.5 * n_sensors
    feasible = [i for i, mf in enumerate(misfits) if mf <= discrepancy_target]
    if feasible:
        idx_discrepancy = max(feasible, key=lambda i: alphas[i])  # largest alpha still within target
    else:
        idx_discrepancy = int(np.argmin(misfits))
    alpha_discrepancy = alphas[idx_discrepancy]

    idx_retrospective_best = int(np.argmin(m_errors))
    alpha_retrospective_best = alphas[idx_retrospective_best]

    lines.append("")
    lines.append(f"Discrepancy-principle target: misfit <= 0.5*n_sensors = {discrepancy_target:.2f}")
    lines.append(f"Discrepancy-principle alpha choice (truth-free): {alpha_discrepancy:.2e} "
                 f"(misfit={misfits[idx_discrepancy]:.4e})")
    lines.append(f"Retrospective best alpha (uses m_true, NOT available in practice): "
                 f"{alpha_retrospective_best:.2e} (rel_m_error={m_errors[idx_retrospective_best]:.4f})")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"regularization_study_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines[-3:]:
        print(line)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(misfits, regs, "o-")
    axes[0].scatter([misfits[idx_discrepancy]], [regs[idx_discrepancy]], color="green", s=100,
                     zorder=5, label=f"discrepancy principle\n(alpha={alpha_discrepancy:.1e})")
    axes[0].scatter([misfits[idx_retrospective_best]], [regs[idx_retrospective_best]], color="red",
                     marker="*", s=150, zorder=5,
                     label=f"retrospective best\n(alpha={alpha_retrospective_best:.1e})")
    for i, a in enumerate(alphas):
        axes[0].annotate(f"{a:.0e}", (misfits[i], regs[i]), fontsize=6)
    axes[0].set_xlabel("data misfit")
    axes[0].set_ylabel("regularisation term")
    axes[0].set_title("L-curve")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].loglog(alphas, m_errors, "o-", color="purple", label="rel. m error (RETROSPECTIVE,\nuses m_true)")
    axes[1].axvline(alpha_discrepancy, color="green", linestyle="--", label="discrepancy-principle alpha")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("Retrospective error vs. alpha")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"regularization_study_{args.config}.png", dpi=150)
    print(f"\nWrote results/regularization_study_{args.config}.txt and "
          f"figures/regularization_study_{args.config}.png")


if __name__ == "__main__":
    main()
