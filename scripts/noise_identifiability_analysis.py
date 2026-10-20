"""Identifiability and statistical-limit analysis for the noisy spatial benchmark.

CPU only, no training: uses the finite-difference reference solver
(src/fd_reference.py) to compute

1. solver verification against the manufactured pressure;
2. Fisher information / Cramer-Rao bound (CRLB) at the true parameters for
   several sensor layouts and counts, its eigen-structure, and the sensor
   count needed for a given RMS relative parameter error;
3. a Monte Carlo study of the nonlinear maximum-likelihood estimator (MLE)
   and of MAP estimators with zero-centred Gaussian priors, on exactly the
   observation-generation code used by the PINN runs;
4. the MLE on the datasets of earlier PINN noise runs, for direct comparison.

    python scripts/noise_identifiability_analysis.py --output results/deepxde_identifiability/analysis.json
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.deepxde_inverse_spatial_k import ALPHA_TRUE, BETA_TRUE, observation_data
from src.fd_reference import DarcyFD

THETA = np.array([ALPHA_TRUE, BETA_TRUE])
NORM = float(np.linalg.norm(THETA))


def verification():
    rows, previous = [], None
    for n in (50, 100, 200, 400):
        fd = DarcyFD(n)
        p = fd.solve(*THETA)
        xx, yy = np.meshgrid(fd.nodes, fd.nodes, indexing="ij")
        err = float(np.abs(p - np.sin(np.pi * xx) * np.sin(np.pi * yy)).max())
        rows.append({"n": n, "max_nodal_error": err,
                     "observed_order": None if previous is None else float(np.log2(previous / err))})
        previous = err
    return rows


def crlb_table(fd, layouts, counts):
    """Covariance for sigma = 1 (it scales with sigma^2)."""
    out = []
    for n_obs in counts:
        for layout in layouts:
            xy, _ = observation_data(layout, n_obs)
            jac = fd.jacobian(THETA, xy)
            info = jac.T @ jac
            cov = np.linalg.inv(info)
            evals, evecs = np.linalg.eigh(info)
            out.append({
                "observations": n_obs, "layout_seed": layout,
                "sd_alpha_per_sigma": float(np.sqrt(cov[0, 0])),
                "sd_beta_per_sigma": float(np.sqrt(cov[1, 1])),
                "correlation": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
                "rms_relative_error_per_sigma": float(np.sqrt(np.trace(cov)) / NORM),
                "fisher_eigenvalues_per_sigma2": [float(v) for v in evals],
                "weak_direction": [float(v) for v in evecs[:, 0]],
                "fisher_condition_number": float(evals[-1] / evals[0]),
            })
    return out


def map_estimate(fd, xy, values, sigma, prior_sd, theta0=(0.0, 0.0), iters=50):
    """Gauss-Newton MAP with prior N(0, prior_sd^2 I); prior_sd=None gives the MLE."""
    theta = np.asarray(theta0, dtype=float)
    lam = 0.0 if prior_sd is None else (sigma / prior_sd) ** 2
    for _ in range(iters):
        r = fd.predict(theta, xy) - values
        jac = fd.jacobian(theta, xy)
        step = np.linalg.solve(jac.T @ jac + lam * np.eye(2), -(jac.T @ r + lam * theta))
        theta = theta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return theta


def monte_carlo(fd, sigma, n_obs, layout, draws, priors):
    xy, clean = observation_data(layout, n_obs)
    clean = clean.ravel()
    rng = np.random.default_rng([layout, n_obs, int(round(sigma * 1e6))])
    est = {name: [] for name in priors}
    for _ in range(draws):
        values = clean + rng.normal(0.0, sigma, size=clean.shape)
        for name, sd in priors.items():
            est[name].append(map_estimate(fd, xy, values, sigma, sd))
    summary = {}
    for name, rows in est.items():
        rows = np.array(rows)
        rel = np.linalg.norm(rows - THETA, axis=1) / NORM
        summary[name] = {
            "prior_sd": priors[name],
            "mean_alpha": float(rows[:, 0].mean()), "mean_beta": float(rows[:, 1].mean()),
            "sd_alpha": float(rows[:, 0].std(ddof=1)), "sd_beta": float(rows[:, 1].std(ddof=1)),
            "rms_relative_error": float(np.sqrt(np.mean(rel ** 2))),
            "median_relative_error": float(np.median(rel)),
            "p90_relative_error": float(np.quantile(rel, 0.9)),
            "fraction_below_1e-2": float(np.mean(rel < 1e-2)),
        }
    return summary


def earlier_pinn_runs(fd, results_dir):
    out = []
    for sigma, name in ((0.01, "spatial_lbfgs_noise0p01.json"), (0.05, "spatial_lbfgs_noise0p05.json")):
        path = Path(results_dir) / name
        if not path.exists():
            continue
        for r in json.loads(path.read_text())["records"]:
            xy, values = observation_data(r["seed"], 128, sigma)  # coupled legacy stream
            mle = fd.mle(xy, values)[0]
            pinn = np.array([r["alpha_estimated"], r["beta_estimated"]])
            out.append({"sigma": sigma, "seed": r["seed"],
                        "mle": [float(v) for v in mle], "pinn": [float(v) for v in pinn],
                        "mle_relative_error": float(np.linalg.norm(mle - THETA) / NORM),
                        "pinn_relative_error": float(np.linalg.norm(pinn - THETA) / NORM),
                        "pinn_minus_mle_relative": float(np.linalg.norm(pinn - mle) / NORM)})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/deepxde_identifiability/analysis.json")
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--earlier-runs",
                        default="results/deepxde_remote/kaggle_b1_sensitivity_b_d3ee5b2/results")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    fd200, fd100 = DarcyFD(200), DarcyFD(100)
    counts = [16, 32, 64, 128, 256, 512, 1024]
    table = crlb_table(fd200, [42, 0, 1, 2, 3], counts)
    per_count = {}
    for n_obs in counts:
        rms = [r["rms_relative_error_per_sigma"] for r in table if r["observations"] == n_obs]
        per_count[n_obs] = float(np.mean(rms))
    # RMS relative error ~ c / sqrt(N) sigma: fit c on layouts 42, 0..3
    c = float(np.mean([per_count[n] * np.sqrt(n) for n in counts]))
    needed = {f"sigma_{s}": {f"target_{t}": int(np.ceil((c * s / t) ** 2)) for t in (1e-2, 2e-2)}
              for s in (0.01, 0.05)}
    priors = {"mle": None, "map_prior_sd_1": 1.0, "map_prior_sd_0.25": 0.25}
    mc = {f"sigma_{s}": monte_carlo(fd100, s, 128, 42, args.draws, priors) for s in (0.01, 0.05)}
    mc["sensors_sigma_0.01"] = {n: monte_carlo(fd100, 0.01, n, 42, args.draws // 2, {"mle": None})["mle"]
                                for n in (32, 256)}
    payload = {
        "description": "CPU identifiability analysis; finite-difference reference, no training",
        "true_parameters": THETA.tolist(),
        "verification": verification(),
        "noise_free_mle_bias_n200_layout42": [float(v) for v in
                                              fd200.mle(*observation_data(42, 128))[0] - THETA],
        "crlb": table,
        "crlb_mean_rms_relative_error_per_sigma_by_count": per_count,
        "crlb_scaling_constant_c": c,
        "sensors_needed_for_rms_relative_error": needed,
        "monte_carlo_layout42": mc,
        "monte_carlo_draws": args.draws,
        "earlier_pinn_runs_vs_mle": earlier_pinn_runs(fd200, args.earlier_runs),
        "seconds": time.perf_counter() - start,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} in {payload['seconds']:.0f} s")


if __name__ == "__main__":
    main()
