"""Multi-chain MCMC convergence diagnostics following

    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Buerkner, P.-C. (2021),
    "Rank-normalization, folding, and localization: an improved R-hat for
    assessing convergence of MCMC", Bayesian Analysis 16(2), 667-718,

with the effective-sample-size estimator of Stan / ArviZ (Geyer's initial
monotone sequence applied to the multi-chain autocorrelation estimate).

All functions take `chains` as an array of shape (n_chains, n_draws) for one
scalar quantity (post-warmup draws). Implemented with NumPy/SciPy only and
validated against analytically known cases in tests/test_mcmc_diagnostics.py.

This module supplements, and does not replace, the historical single-chain
estimator `src.pcn_sampler.effective_sample_size` (lag cap 1000), which is kept
so that historical numbers remain reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

CONVERGENCE_RHAT = 1.01
CONVERGENCE_ESS = 400


def _as_chains(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("expected shape (n_chains, n_draws)")
    return x


def split_chains(x) -> np.ndarray:
    """Split each chain into two halves (drops the middle draw if odd)."""
    x = _as_chains(x)
    n = x.shape[1]
    h = n // 2
    return np.concatenate([x[:, :h], x[:, n - h:]], axis=0)


def rank_normalize(x) -> np.ndarray:
    """Normal scores of pooled average ranks: Phi^-1((r - 3/8) / (S + 1/4))."""
    x = _as_chains(x)
    r = stats.rankdata(x.ravel(), method="average").reshape(x.shape)
    return stats.norm.ppf((r - 0.375) / (x.size + 0.25))


def autocovariance(x: np.ndarray) -> np.ndarray:
    """Biased (divide by n) autocovariance of each row, lags 0..n-1, via FFT."""
    x = _as_chains(x)
    n = x.shape[1]
    xc = x - x.mean(axis=1, keepdims=True)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, nfft, axis=1)
    return np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n] / n


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Autocorrelation of each chain, lags 0..n-1."""
    ac = autocovariance(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ac / ac[:, :1]


def _rhat_basic(x: np.ndarray) -> float:
    x = _as_chains(x)
    n = x.shape[1]
    w = x.var(axis=1, ddof=1).mean()
    b = n * x.mean(axis=1).var(ddof=1)
    if w <= 0:
        return np.nan
    var_plus = (n - 1) / n * w + b / n
    return float(np.sqrt(var_plus / w))


def rhat(x) -> float:
    """Rank-normalised split R-hat: max of the bulk and folded (tail) versions."""
    s = split_chains(x)
    if np.ptp(s) == 0:
        return np.nan
    bulk = _rhat_basic(rank_normalize(s))
    folded = np.abs(s - np.median(s))
    tail = _rhat_basic(rank_normalize(folded))
    return float(max(bulk, tail))


def _ess(x: np.ndarray) -> float:
    """Multi-chain ESS with Geyer's initial monotone sequence (Stan/ArviZ)."""
    x = _as_chains(x)
    m, n = x.shape
    if n < 4 or np.ptp(x) == 0:
        return np.nan
    acov = autocovariance(x)
    chain_mean = x.mean(axis=1)
    mean_var = (acov[:, 0] * n / (n - 1.0)).mean()
    var_plus = mean_var * (n - 1.0) / n
    if m > 1:
        var_plus += chain_mean.var(ddof=1)
    rho = np.zeros(n)
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1.0 - (mean_var - acov[:, 1].mean()) / var_plus
    rho[1] = rho_odd
    t = 1
    while t < n - 3 and (rho_even + rho_odd) > 0.0:
        rho_even = 1.0 - (mean_var - acov[:, t + 1].mean()) / var_plus
        rho_odd = 1.0 - (mean_var - acov[:, t + 2].mean()) / var_plus
        if rho_even + rho_odd >= 0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:
        rho[max_t + 1] = rho_even
    t = 1
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2
    total = m * n
    tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + np.sum(rho[max_t + 1: max_t + 2])
    tau = max(tau, 1.0 / np.log10(total))
    return float(total / tau)


def ess_bulk(x) -> float:
    return _ess(rank_normalize(split_chains(x)))


def ess_tail(x, probs=(0.05, 0.95)) -> float:
    s = split_chains(x)
    out = []
    for p in probs:
        q = np.quantile(s, p)
        out.append(_ess(split_chains((_as_chains(x) <= q).astype(float))))
    return float(np.nanmin(out)) if not all(np.isnan(out)) else np.nan


def ess_mean(x) -> float:
    return _ess(split_chains(x))


def ess_sd(x) -> float:
    x = _as_chains(x)
    return float(np.nanmin([ess_mean(x), ess_mean((x - x.mean()) ** 2)]))


def mcse_mean(x) -> float:
    x = _as_chains(x)
    return float(x.std(ddof=1) / np.sqrt(ess_mean(x)))


def mcse_sd(x) -> float:
    """ArviZ's MCSE of the standard deviation."""
    x = _as_chains(x)
    e = ess_sd(x)
    sd = x.std(ddof=1)
    return float(sd * np.sqrt(np.e * (1 - 1 / e) ** (e - 1) - 1))


def mcse_quantile(x, prob: float) -> float:
    """MCSE of a quantile via the ESS of the indicator and a beta interval
    (ArviZ/Stan approach)."""
    x = _as_chains(x)
    q = np.quantile(x, prob)
    e = _ess(split_chains((x <= q).astype(float)))
    p = np.array([0.1586553, 0.8413447])
    a = stats.beta.ppf(p, e * prob + 1, e * (1 - prob) + 1)
    srt = np.sort(x.ravel())
    size = srt.size
    i1 = int(np.floor(np.clip(a[0] * size, 0, size - 1)))
    i2 = int(np.ceil(np.clip(a[1] * size, 0, size - 1)))
    return float((srt[i2] - srt[i1]) / 2.0)


def summarize(x, name: str = "") -> dict:
    """All diagnostics for one scalar quantity; `converged` uses the
    thresholds R-hat < 1.01 and bulk and tail ESS >= 400."""
    x = _as_chains(x)
    out = {
        "name": name, "n_chains": int(x.shape[0]), "n_draws_per_chain": int(x.shape[1]),
        "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
        "q05": float(np.quantile(x, 0.05)), "q50": float(np.quantile(x, 0.5)),
        "q95": float(np.quantile(x, 0.95)),
        "rhat": rhat(x), "ess_bulk": ess_bulk(x), "ess_tail": ess_tail(x), "ess_mean": ess_mean(x),
        "mcse_mean": mcse_mean(x), "mcse_sd": mcse_sd(x),
        "mcse_q05": mcse_quantile(x, 0.05), "mcse_q95": mcse_quantile(x, 0.95),
    }
    out["converged"] = bool(out["rhat"] < CONVERGENCE_RHAT and out["ess_bulk"] >= CONVERGENCE_ESS
                            and out["ess_tail"] >= CONVERGENCE_ESS)
    return out


def summarize_all(draws: np.ndarray, names: list[str]) -> list[dict]:
    """draws: (n_chains, n_draws, n_quantities)."""
    draws = np.asarray(draws, dtype=float)
    return [summarize(draws[:, :, k], names[k]) for k in range(draws.shape[2])]
