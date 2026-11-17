"""Why the stored (historical) posterior variances violate the exact floor.

In the historical sampler the constant-mode coefficient xi_0 is part of the
pCN state. Its proposal xi_0' = sqrt(1-beta^2) xi_0 + beta eta_0 does not enter
the acceptance probability (the likelihood does not depend on xi_0), and eta_0
is independent of everything that does. Hence xi_0 evolves as an AR(1) process
on accepted steps only, driven by noise independent of the accept/reject
sequence. Conditional on that sequence (which is recoverable from the stored
retained chain), the law of xi_0's within-chain sample variance can be computed
exactly by simulation, without any PDE solve.

The burn-in accept/reject pattern was not stored; its number of accepted moves is
drawn as Binomial(burn_in, acceptance rate) and its order is irrelevant for the
AR(1) state at the end of burn-in, which depends only on that count.

    python scripts/m3_constant_mode_analysis.py
Output: results/m3/constant_mode_analysis.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BETA, BURN_IN = 0.02, 2000


def main():
    out = {}
    rng = np.random.default_rng(20261116)
    c = np.sqrt(1 - BETA**2)
    for truth, acc_total in (("smooth", 0.3998), ("structured", 0.4398)):
        d = np.load(ROOT / "results" / f"posterior_samples_{truth}_full.npz")
        xi = d["xi_samples"]
        lam = d["eigenvalues"]
        moved_mid = np.any(np.diff(xi, axis=0) != 0, axis=1)
        obs_var0 = float(xi[:, 0].var(ddof=1))
        reps = 4000
        # AR(1) from xi_0 = 0 (the MAP value) after k_burn accepted burn-in moves, vectorised over replicates
        k_burn = rng.binomial(BURN_IN, acc_total, size=reps)
        x = np.sqrt(1 - c ** (2 * k_burn)) * rng.standard_normal(reps)
        s1, s2 = x.copy(), x**2
        for t in range(1, len(xi)):
            if moved_mid[t - 1]:
                x = c * x + BETA * rng.standard_normal(reps)
            s1 += x
            s2 += x**2
        n = len(xi)
        sims = (s2 - s1**2 / n) / (n - 1)
        acc_ret = float(moved_mid.mean())
        tau0 = (2 - acc_ret * BETA**2 / 2) / (acc_ret * BETA**2 / 2)  # AR(1) with rho = 1 - acc*(1-c) ~ 1 - acc*beta^2/2
        # integrated variance decomposition of the stored chain (KL modes are L2-orthonormal, phi_0 = 1)
        var_xi = xi.var(axis=0, ddof=1)
        out[truth] = {
            "observed_sample_variance_xi0": obs_var0,
            "exact_posterior_variance_xi0": 1.0,
            "simulated_sample_variance_xi0": {"mean": float(sims.mean()), "q05": float(np.quantile(sims, 0.05)),
                                               "q50": float(np.median(sims)), "q95": float(np.quantile(sims, 0.95)),
                                               "fraction_ge_1": float(np.mean(sims >= 1.0)),
                                               "observed_quantile": float(np.mean(sims <= obs_var0))},
            "accepted_fraction_retained": acc_ret,
            "integrated_autocorrelation_time_xi0_proposals": float(tau0),
            "chain_length_over_tau0": float(len(xi) / tau0),
            "stored_integrated_variance_constant_mode_part": float(lam[0] * var_xi[0]),
            "stored_integrated_variance_mean_free_part": float(np.sum(lam[1:] * var_xi[1:])),
            "stored_integrated_variance_total_from_coefficients": float(np.sum(lam * var_xi)),
        }
        print(truth, json.dumps(out[truth], indent=1))
    (ROOT / "results" / "m3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "m3" / "constant_mode_analysis.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
