"""Calibration statistics for Bayesian inversion: simulation-based calibration
(SBC) ranks, probability-integral transforms, credible-interval coverage with
exact binomial uncertainty, bias / RMSE / spread-skill, and Gaussian (Laplace)
counterparts.

SBC (Talts et al. 2018): if truths are drawn from the prior, data from the
likelihood, and the posterior is computed exactly, the rank of the truth among L
independent posterior draws is uniform on {0, ..., L}, and the posterior CDF at
the truth (PIT) is uniform on (0, 1). Departures indicate miscalibration of the
inference procedure under its own model.

For quantities with an exact, independent Gaussian component (the unidentifiable
constant KL mode), the PIT is computed by Rao-Blackwellisation:
F(t) = mean_s Phi((t - v_s) / sd_c), where v_s are draws of the identifiable part.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rank_of_truth(draws: np.ndarray, truth: float, rng: np.random.Generator | None = None) -> int:
    """Number of draws below the truth, with ties broken uniformly at random."""
    draws = np.asarray(draws, dtype=float)
    below = int(np.sum(draws < truth))
    ties = int(np.sum(draws == truth))
    if ties and rng is not None:
        below += int(rng.integers(0, ties + 1))
    return below


def select_rank_draws(post: np.ndarray, L: int) -> np.ndarray:
    """Equally spaced draws from each chain of post (n_chains, n_draws, ...), L in total,
    allocated as evenly as possible (e.g. 25, 25, 25, 24 for L = 99, 4 chains)."""
    m, n = post.shape[:2]
    counts = [L // m + (1 if c < L % m else 0) for c in range(m)]
    out = []
    for c, k in enumerate(counts):
        idx = np.linspace(0, n - 1, k).round().astype(int)
        out.append(post[c, idx])
    return np.concatenate(out, axis=0)


def rb_pit(values: np.ndarray, sd: float, truth: float) -> float:
    """Posterior CDF at the truth for X = V + Z, Z ~ N(0, sd^2) independent of V."""
    return float(stats.norm.cdf((truth - np.asarray(values)) / sd).mean())


def central_interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    return float(lo), float(hi)


def covered_by_pit(pit: float, level: float) -> bool:
    """Central-interval coverage expressed through the PIT: truth inside the central
    level-interval iff (1-level)/2 <= PIT <= (1+level)/2."""
    a = (1 - level) / 2
    return bool(a <= pit <= 1 - a)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def rank_chi2(ranks: np.ndarray, L: int, bins: int = 10) -> dict:
    """Chi-square test of uniformity of ranks in {0..L} with `bins` equal bins."""
    if (L + 1) % bins:
        raise ValueError("L + 1 must be divisible by bins")
    counts = np.bincount(np.asarray(ranks) // ((L + 1) // bins), minlength=bins)
    exp = len(ranks) / bins
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    return {"counts": counts.tolist(), "chi2": chi2, "df": bins - 1, "p": float(stats.chi2.sf(chi2, bins - 1))}


def pit_ks(pits: np.ndarray) -> dict:
    r = stats.kstest(np.asarray(pits), "uniform")
    return {"ks": float(r.statistic), "p": float(r.pvalue)}


def coverage_summary(pits: np.ndarray, levels=(0.5, 0.8, 0.9, 0.95)) -> dict:
    pits = np.asarray(pits)
    out = {}
    for lv in levels:
        k = int(sum(covered_by_pit(u, lv) for u in pits))
        lo, hi = clopper_pearson(k, len(pits))
        out[str(lv)] = {"covered": k, "n": len(pits), "rate": k / len(pits), "ci95": [lo, hi],
                        "nominal_inside_ci": bool(lo <= lv <= hi)}
    return out


def max_coverage_exclusions(K: int, p: float = 0.05, q: float = 0.99) -> int:
    """99th percentile of the number of nominal-excluding 95% intervals among K
    independent calibrated checks (pre-registered tolerance)."""
    return int(stats.binom.ppf(q, K, p))


def bias_rmse_spread(post_means, post_sds, truths, n_boot: int = 2000, seed: int = 0) -> dict:
    m, s, t = (np.asarray(x, dtype=float) for x in (post_means, post_sds, truths))
    err = m - t
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ratio = float(np.mean(err ** 2) / np.mean(s ** 2))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    boot = np.mean(err[idx] ** 2, axis=1) / np.mean(s[idx] ** 2, axis=1)
    z = err / s
    return {"bias": float(err.mean()), "bias_se": float(err.std(ddof=1) / np.sqrt(len(t))),
            "standardized_bias": float(z.mean()), "standardized_bias_se": float(z.std(ddof=1) / np.sqrt(len(t))),
            "rmse": rmse, "mean_posterior_sd": float(np.mean(s)),
            "spread_skill_ratio": ratio, "spread_skill_ci95": np.quantile(boot, [0.025, 0.975]).tolist()}


def gaussian_pit(mean: float, sd: float, truth: float) -> float:
    return float(stats.norm.cdf((truth - mean) / sd))
