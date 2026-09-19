"""Validate the pCN sampler against an analytically-solvable linear-Gaussian
posterior BEFORE trusting it on the PDE posterior.

Usage: python scripts/validate_sampler.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.linear_gaussian_validation import make_problem
from src.pcn_sampler import run_pcn, autocorrelation, effective_sample_size

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    r, n_obs, noise_std = 8, 20, 0.3
    beta, burn_in, n_samples = 0.05, 5000, 40000

    prob = make_problem(r=r, n_obs=n_obs, noise_std=noise_std, seed=1)
    mean_true, cov_true = prob.analytical_posterior()

    result = run_pcn(prob.neg_log_lik, r=r, n_samples=n_samples, beta=beta, burn_in=burn_in, seed=2)
    mc_mean = result.samples.mean(axis=0)
    mc_cov = np.cov(result.samples.T)

    mean_rel_err = np.linalg.norm(mc_mean - mean_true) / np.linalg.norm(mean_true)
    var_abs_err = np.mean(np.abs(np.diag(mc_cov) - np.diag(cov_true)))
    ess = np.array([effective_sample_size(result.samples[:, i]) for i in range(r)])

    lines = [
        f"pCN sampler validation: linear-Gaussian problem, r={r}, n_obs={n_obs}, noise_std={noise_std}",
        f"pCN settings: beta={beta}, burn_in={burn_in}, n_samples={n_samples}, "
        f"n_proposals={result.n_proposals}",
        f"acceptance rate: {result.acceptance_rate:.4f}",
        "",
        f"posterior mean relative error (MCMC vs. analytical): {mean_rel_err:.4e}",
        f"posterior variance mean absolute error: {var_abs_err:.4e}",
        f"ESS per dimension: {np.round(ess).astype(int).tolist()}",
        f"mean ESS: {ess.mean():.1f}  (ESS / n_proposals = {ess.mean()/result.n_proposals:.5f})",
        "",
        f"{'dim':>4} {'true mean':>10} {'MCMC mean':>10} {'true var':>10} {'MCMC var':>10}",
    ]
    for i in range(r):
        lines.append(f"{i:4d} {mean_true[i]:10.4f} {mc_mean[i]:10.4f} "
                      f"{cov_true[i,i]:10.4f} {mc_cov[i,i]:10.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "sampler_validation.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(result.raw_chain[:, 0], lw=0.5)
    axes[0, 0].axvline(burn_in, color="r", linestyle="--", label="end of burn-in")
    axes[0, 0].set_title("Trace plot, dim 0 (full chain incl. burn-in)")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].legend(fontsize=8)

    acf = autocorrelation(result.samples[:, 0], max_lag=200)
    axes[0, 1].plot(acf)
    axes[0, 1].set_title("Autocorrelation, dim 0 (post burn-in)")
    axes[0, 1].set_xlabel("lag")
    axes[0, 1].axhline(0, color="k", lw=0.5)

    axes[1, 0].plot(mean_true, "o-", label="analytical")
    axes[1, 0].plot(mc_mean, "s--", label="MCMC")
    axes[1, 0].set_title("Posterior mean per dimension")
    axes[1, 0].set_xlabel("KL dimension")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(np.diag(cov_true), "o-", label="analytical")
    axes[1, 1].plot(np.diag(mc_cov), "s--", label="MCMC")
    axes[1, 1].set_title("Posterior variance per dimension")
    axes[1, 1].set_xlabel("KL dimension")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("pCN sampler validation on analytically-solvable linear-Gaussian problem")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "sampler_validation.png", dpi=150)
    print(f"\nWrote results/sampler_validation.txt and figures/sampler_validation.png")


if __name__ == "__main__":
    main()
