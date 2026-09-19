"""KL prior diagnostics: eigenvalue decay, captured-variance-fraction vs.
truncation r, and representative prior samples. No MCMC/PDE-solve loop
here beyond the one-time generalized eigenproblem, so this is fast at any
resolution.

Usage: python scripts/kl_prior_study.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.forward import build_mesh
from src.kl_prior import build_kl_prior

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def triangulation(dm):
    x = dm.mesh.geometry.x
    cells = dm.mesh.geometry.dofmap.reshape(-1, 3)
    return mtri.Triangulation(x[:, 0], x[:, 1], cells)


def main() -> None:
    dm = build_mesh(16)
    gamma, delta = 0.05, 1.0
    prior_full = build_kl_prior(dm, r=40, gamma=gamma, delta=delta)

    cum = np.cumsum(prior_full.eigenvalues)
    total_all = np.sum(prior_full.all_mu ** (-2.0))
    r_values = [5, 10, 15, 20, 25, 30, 40]
    fractions = [cum[r - 1] / total_all for r in r_values]

    lines = [f"KL prior study: gamma={gamma}, delta={delta}, inversion mesh N=16",
             f"{'r':>4} {'captured_variance_fraction':>28}"]
    for r, f in zip(r_values, fractions):
        lines.append(f"{r:4d} {f:28.4f}")
    lines.append("")
    lines.append("Chosen r=15 for the main studies: captures "
                  f"{fractions[r_values.index(15)]:.1%} of prior variance, a reasonable "
                  "balance -- r=25 would only add ~1 more percentage point at roughly 1.7x "
                  "the MCMC dimension/cost.")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "kl_prior_study.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].semilogy(np.arange(1, 41), prior_full.eigenvalues, "o-", ms=3)
    axes[0].set_xlabel("KL mode index $j$")
    axes[0].set_ylabel(r"eigenvalue $\lambda_j$")
    axes[0].set_title("KL eigenvalue decay")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(r_values, fractions, "o-", color="green")
    axes[1].axvline(15, color="k", linestyle="--", alpha=0.6, label="chosen r=15")
    axes[1].set_xlabel("truncation $r$")
    axes[1].set_ylabel("captured variance fraction")
    axes[1].set_title("Captured prior variance vs. truncation")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "kl_eigenvalue_decay.png", dpi=150)

    # representative prior samples
    prior15 = build_kl_prior(dm, r=15, gamma=gamma, delta=delta)
    tri = triangulation(dm)
    rng = np.random.default_rng(0)
    fig2, axes2 = plt.subplots(1, 4, figsize=(15, 3.6))
    for i, ax in enumerate(axes2):
        xi = rng.standard_normal(prior15.r)
        m = prior15.sample(xi)
        im = ax.tricontourf(tri, m.x.array, levels=30, cmap="viridis")
        ax.set_title(f"prior sample {i+1}")
        ax.set_aspect("equal")
        fig2.colorbar(im, ax=ax, fraction=0.046)
    fig2.suptitle(f"Representative KL prior samples (r=15, gamma={gamma}, delta={delta})")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "kl_prior_samples.png", dpi=150)

    print(f"\nWrote results/kl_prior_study.txt, figures/kl_eigenvalue_decay.png, "
          f"figures/kl_prior_samples.png")


if __name__ == "__main__":
    main()
