"""Milestone 5, Phase C: calibration machinery on linear-Gaussian problems with
known posteriors (recorded evidence; the same checks run as tests in
tests/test_calibration.py).

  1. exact posterior draws, 1,000 datasets: ranks, PIT, coverage, spread-skill
  2. known miscalibration (posterior mean shifted by 1 sd; posterior sd halved)
  3. the HpCN sampler itself (Hessian exact), 300 datasets, 4 chains x (500 + 2,000)

    OMP_NUM_THREADS=1 python scripts/m5_analytic_calibration.py
Output: results/m5/analytic_calibration.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import calibration as cal  # noqa: E402
from src.hessian_pcn import HessianInformedProposal, HessianPCNConfig, HpCNChain  # noqa: E402


def problem(d, seed):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d + 2, d)) * np.geomspace(10, 0.1, d)[None, :]
    sigma = 0.5
    return G, sigma, np.linalg.inv(G.T @ G / sigma**2 + np.eye(d))


def summarise(ranks, pits, means, sds, truths, L=99):
    return {"n": len(ranks), "rank_chi2": cal.rank_chi2(np.array(ranks), L), "pit_ks": cal.pit_ks(np.array(pits)),
            "coverage": cal.coverage_summary(np.array(pits)),
            "bias_rmse": cal.bias_rmse_spread(means, sds, truths)}


def exact(n, shift=0.0, scale=1.0, d=3, L=99, seed=0):
    G, sigma, cov = problem(d, seed)
    Lc = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed + 1)
    r, p, m, s, t = [], [], [], [], []
    for _ in range(n):
        xi = rng.standard_normal(d)
        y = G @ xi + sigma * rng.standard_normal(G.shape[0])
        mean = cov @ G.T @ y / sigma**2 + shift * np.sqrt(np.diag(cov))
        draws = mean + scale * (Lc @ rng.standard_normal((d, L))).T
        sd = scale * np.sqrt(cov[0, 0])
        r.append(cal.rank_of_truth(draws[:, 0], xi[0])); p.append(cal.gaussian_pit(mean[0], sd, xi[0]))
        m.append(mean[0]); s.append(sd); t.append(xi[0])
    return summarise(r, p, m, s, t)


def hpcn(n=300, d=3, L=99):
    G, sigma, cov = problem(d, 3)
    prop = HessianInformedProposal.from_hessian(G.T @ G / sigma**2, HessianPCNConfig(rho=0.7))
    Lc = np.linalg.cholesky(cov)
    rng = np.random.default_rng(4)
    out = {j: ([], [], [], [], []) for j in range(d)}
    for i in range(n):
        xi = rng.standard_normal(d)
        y = G @ xi + sigma * rng.standard_normal(G.shape[0])
        mean = cov @ G.T @ y / sigma**2
        phi = lambda x, y=y: 0.5 * float(np.sum((G @ x - y) ** 2)) / sigma**2
        post = []
        for c in range(4):
            ch = HpCNChain(phi, prop, mean + 3 * Lc @ rng.standard_normal(d), seed=1000 * i + c)
            ch.step(500)
            post.append(ch.step(2000)[0])
        post = np.stack(post)
        rd = cal.select_rank_draws(post, L)
        flat = post.reshape(-1, d)
        for j in range(d):
            r, p, m, s, t = out[j]
            r.append(cal.rank_of_truth(rd[:, j], xi[j])); p.append(float(np.mean(flat[:, j] < xi[j])))
            m.append(flat[:, j].mean()); s.append(flat[:, j].std(ddof=1)); t.append(xi[j])
    return {f"xi[{j}]": summarise(*out[j]) for j in range(d)}


def main():
    t0 = time.perf_counter()
    res = {"exact_posterior": exact(1000), "shifted_mean_1sd": exact(1000, shift=1.0),
           "sd_halved": exact(1000, scale=0.5), "hpcn_sampler": hpcn()}
    res["seconds"] = time.perf_counter() - t0
    out = ROOT / "results" / "m5" / "analytic_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for k in ("exact_posterior", "shifted_mean_1sd", "sd_halved"):
        v = res[k]
        print(k, "rank p=%.3g" % v["rank_chi2"]["p"], {lv: round(c["rate"], 3) for lv, c in v["coverage"].items()},
              "spread-skill %.2f" % v["bias_rmse"]["spread_skill_ratio"])
    for j, v in res["hpcn_sampler"].items():
        print("hpcn", j, "rank p=%.3g" % v["rank_chi2"]["p"], {lv: round(c["rate"], 3) for lv, c in v["coverage"].items()},
              "spread-skill %.2f" % v["bias_rmse"]["spread_skill_ratio"])


if __name__ == "__main__":
    main()
