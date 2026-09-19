"""Taylor-remainder verification of the xi-space (KL-coefficient) gradient
used for MAP optimisation of the Bayesian Darcy posterior.

This gradient reuses Project 3's already-verified nodal-space adjoint
gradient, adding one chain-rule step (src/bayesian_darcy.py). That chain
rule is new code and is verified here independently, exactly as the
nodal-space gradient was verified in scripts/verify_gradient.py, before
being trusted for MAP optimisation.

Usage: python scripts/verify_bayesian_gradient.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.forward import build_mesh
from src.kl_prior import build_kl_prior
from src.observation import grid_sensors, build_observation_operator
from src.bayesian_darcy import BayesianDarcyProblem

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main() -> None:
    dm = build_mesh(16)
    prior = build_kl_prior(dm, r=10, gamma=0.05, delta=1.0)
    sensors = grid_sensors(4, 4)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.04)
    rng = np.random.default_rng(0)
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    sigma = np.full(H.n_sensors, 0.02)

    bp = BayesianDarcyProblem(dm=dm, obs=H, y=y, sigma=sigma, prior=prior)

    xi0 = 0.3 * rng.standard_normal(prior.r)
    direction = rng.standard_normal(prior.r)
    direction /= np.linalg.norm(direction)

    J0, g0 = bp.neg_log_posterior_and_grad(xi0)
    dd = float(g0 @ direction)

    eps_list = [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4]
    lines = [f"Psi(xi0) = {J0:.6e}, directional derivative = {dd:.6e}",
             f"{'eps':>10} {'1st-order remainder':>22} {'observed order':>16}"]
    remainders = []
    for eps in eps_list:
        J1, _ = bp.neg_log_posterior_and_grad(xi0 + eps * direction)
        remainders.append(abs(J1 - J0 - eps * dd))

    for i, eps in enumerate(eps_list):
        order = float("nan") if i == 0 else (
            np.log(remainders[i - 1] / remainders[i]) / np.log(eps_list[i - 1] / eps_list[i])
        )
        line = f"{eps:10.2e} {remainders[i]:22.6e} {order:16.3f}"
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "bayesian_gradient_verification.txt").write_text("\n".join(lines) + "\n")
    print("\nWrote results/bayesian_gradient_verification.txt")


if __name__ == "__main__":
    main()
