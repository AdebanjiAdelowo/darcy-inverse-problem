"""Milestone 3 MCMC analysis.

    python scripts/m3_analyze_mcmc.py pilot        # beta selection from results/m3/mcmc/pilot/
    python scripts/m3_analyze_mcmc.py production   # 4-chain diagnostics, constant-mode treatments,
                                                   # variance floor, figures

Monitored quantities (pre-registered): the 14 mean-free KL coefficients xi_R,
the negative log-likelihood Phi, and the mean-free field m_R at the five
historical probe points.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
MC = ROOT / "results" / "m3" / "mcmc"


def monitored(env, xi_r: np.ndarray, phi: np.ndarray):
    rk = env["rk"]
    probes = rk.mean_free_fields(xi_r)[:, env["probe_idx"]]
    names = [f"xi_R[{j}] (KL mode {k})" for j, k in enumerate(rk.keep)] + ["Phi"] + \
            [f"m_R{p}" for p in [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)]]
    return np.column_stack([xi_r, phi, probes]), names


def load(path):
    d = np.load(path)
    return d["xi_r"], d["phi"], json.loads(str(d["meta"]))


def pilot():
    from m3_mcmc import build
    from src import mcmc_diagnostics as md
    rows = []
    envs = {t: build(t) for t in ("smooth", "structured")}
    for f in sorted((MC / "pilot").glob("*.npz")):
        xi, phi, meta = load(f)
        env = envs[meta["truth"]]
        q, names = monitored(env, xi, phi)
        half = len(q) // 2 if meta["init"] == "dispersed" else 0
        qq = q[half:]
        ess = np.array([md.ess_bulk(qq[:, k]) for k in range(qq.shape[1])])
        n = len(qq)
        rows.append({"file": f.name, "truth": meta["truth"], "beta": meta["beta"], "init": meta["init"],
                     "acceptance": meta["acceptance_rate"], "ms_per_proposal": meta["ms_per_proposal"],
                     "draws_used": n, "min_bulk_ess": float(ess.min()),
                     "worst_quantity": names[int(np.argmin(ess))],
                     "min_bulk_ess_per_1000_proposals": float(1000 * ess.min() / n),
                     "median_bulk_ess_per_1000_proposals": float(1000 * np.median(ess) / n),
                     "phi_first_last_mean": [float(phi[:500].mean()), float(phi[-500:].mean())]})
    sel = {}
    for t in ("smooth", "structured"):
        cand = [r for r in rows if r["truth"] == t and r["init"] == "map"]
        best = max(cand, key=lambda r: (round(r["min_bulk_ess_per_1000_proposals"], 6), -r["beta"]))
        rate = best["min_bulk_ess_per_1000_proposals"] / 1000
        # 4 chains, warmup = half: total proposals = 2 * 600 / rate
        per_chain = int(np.ceil(2 * 600 / rate / 4 / 10_000) * 10_000)
        sel[t] = {"beta": best["beta"], "min_bulk_ess_per_proposal": rate,
                  "proposals_per_chain": per_chain,
                  "estimated_hours_per_chain": per_chain * best["ms_per_proposal"] / 3.6e6}
    out = {"rows": rows, "selection": sel,
           "rule": "beta maximising min bulk ESS per proposal over monitored quantities (MAP-started pilots); "
                   "chain length for predicted pooled bulk ESS >= 600 with 4 chains and 50% warmup"}
    (MC / "pilot_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(f"{r['truth']:10s} beta={r['beta']:<5} {r['init']:9s} acc={r['acceptance']:.3f} "
              f"{r['ms_per_proposal']:.1f}ms  minESS/1k={r['min_bulk_ess_per_1000_proposals']:.2f} "
              f"medESS/1k={r['median_bulk_ess_per_1000_proposals']:.2f} worst={r['worst_quantity']} "
              f"Phi first/last={r['phi_first_last_mean'][0]:.1f}/{r['phi_first_last_mean'][1]:.1f}")
    print(json.dumps(sel, indent=1))


def production():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    from m3_mcmc import build
    from src import mcmc_diagnostics as md
    from src.constant_mode import augment_with_independent_constant_mode, marginalized_moments, \
        rao_blackwell_quantiles
    from src.identifiability import check_posterior_variance_floor, mass_matrix, reconstruction_errors

    figdir = ROOT / "results" / "m3" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    report = {}
    for truth in ("smooth", "structured"):
        files = sorted((MC / "production").glob(f"{truth}_*.npz"))
        if not files:
            continue
        env = build(truth)
        rk, prior = env["rk"], env["prior"]
        M = mass_matrix(env["dm"].V_m)
        chains, metas = [], []
        for f in files:
            xi, phi, meta = load(f)
            q, names = monitored(env, xi, phi)
            chains.append(q)
            metas.append(meta)
        n = min(len(c) for c in chains)
        arr = np.stack([c[:n] for c in chains])  # (chains, draws, quantities)
        post = arr[:, n // 2:, :]  # discard first half as warmup
        summ = md.summarize_all(post, names)
        conv = all(s["converged"] for s in summ)
        xi_post = post[:, :, :rk.r_reduced].reshape(-1, rk.r_reduced)
        mom = marginalized_moments(rk, xi_post)
        floor_check = check_posterior_variance_floor(np.sqrt(mom["var_total"]), prior, M)
        mf_only = float(mom["var_mean_free"] @ (M @ np.ones(M.shape[0])))
        xi_c, fields_aug = augment_with_independent_constant_mode(rk, xi_post, seed=2026)
        var_aug = fields_aug.var(axis=0, ddof=1)
        ks = stats.kstest(xi_c, "norm")
        # credible intervals at probes: analytic (Rao-Blackwell) vs augmented draws
        mf_fields = rk.mean_free_fields(xi_post)
        sd_c = float(np.sqrt(rk.lambda_c) * abs(rk.phi_c[0]))
        ci = []
        for j, idx in enumerate(env["probe_idx"]):
            rb = rao_blackwell_quantiles(mf_fields[:, idx], sd_c, [0.025, 0.975])
            au = np.quantile(fields_aug[:, idx], [0.025, 0.975])
            mfq = np.quantile(mf_fields[:, idx], [0.025, 0.975])
            ci.append({"probe": names[rk.r_reduced + 1 + j], "m_true": float(env["m_true"][idx]),
                       "m_true_mean_free": float(env["m_true"][idx] - env["m_true"].mean()),
                       "absolute_95_analytic": rb.tolist(), "absolute_95_augmented": au.tolist(),
                       "mean_free_95": mfq.tolist()})
        errs_mean = reconstruction_errors(mom["mean"], env["m_true"], M)
        report[truth] = {
            "chains": [f.name for f in files], "n_draws_per_chain_stored": n,
            "thin": metas[0]["thin"], "beta": metas[0]["beta"],
            "proposals_per_chain": metas[0]["n_proposals_done"],
            "acceptance_rates": [m["acceptance_rate"] for m in metas],
            "ms_per_proposal": [m["ms_per_proposal"] for m in metas],
            "pcn_hours_per_chain": [m["pcn_seconds"] / 3600 for m in metas],
            "forward_solves": [m["forward_solves"] for m in metas],
            "adjoint_solves": [m["adjoint_solves"] for m in metas],
            "all_monitored_converged": conv,
            "max_rhat": max(s["rhat"] for s in summ),
            "min_ess_bulk": min(s["ess_bulk"] for s in summ),
            "min_ess_tail": min(s["ess_tail"] for s in summ),
            "diagnostics": summ,
            "constant_mode": {
                "lambda_c": rk.lambda_c, "phi_c": float(rk.phi_c[0]),
                "integrated_variance_mean_free_only": mf_only,
                "integrated_variance_analytic_total": floor_check["integrated_variance"],
                "pointwise_variance_min_total": floor_check["pointwise_variance_min"],
                "floor_check_analytic": floor_check,
                "augmented_variance_rel_diff_max": float(np.max(np.abs(var_aug / mom["var_total"] - 1))),
                "augmented_xi_c_mean": float(xi_c.mean()), "augmented_xi_c_var": float(xi_c.var(ddof=1)),
                "ks_pvalue_xi_c_vs_N01": float(ks.pvalue),
            },
            "posterior_mean_errors": errs_mean,
            "credible_intervals": ci,
        }
        # figures: traces (all chains, full length) and ACF for the 3 worst quantities + Phi
        worst = sorted(range(len(summ)), key=lambda k: summ[k]["ess_bulk"])[:3]
        show = worst + [rk.r_reduced] if rk.r_reduced not in worst else worst
        fig, axes = plt.subplots(len(show), 2, figsize=(12, 2.6 * len(show)))
        thin = metas[0]["thin"]
        for row, k in enumerate(show):
            for c in range(arr.shape[0]):
                axes[row, 0].plot(np.arange(n) * thin, arr[c, :, k], lw=0.4, label=f"chain {c + 1}")
            axes[row, 0].axvline(n // 2 * thin, color="k", ls="--", lw=0.8)
            s = summ[k]
            axes[row, 0].set_title(f"{names[k]}: R-hat {s['rhat']:.3f}, bulk ESS {s['ess_bulk']:.0f}, "
                                   f"tail ESS {s['ess_tail']:.0f}", fontsize=9)
            ac = md.autocorrelation(post[:, :, k])
            lags = np.arange(min(400, ac.shape[1])) * thin
            for c in range(ac.shape[0]):
                axes[row, 1].plot(lags, ac[c, :len(lags)], lw=0.8)
            axes[row, 1].axhline(0, color="k", lw=0.5)
            axes[row, 1].set_title(f"autocorrelation (post-warmup), {names[k]}", fontsize=9)
        axes[-1, 0].set_xlabel("proposal")
        axes[-1, 1].set_xlabel("lag (proposals)")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / f"mcmc_traces_{truth}.png", dpi=110)
        plt.close(fig)
    (MC / "production_summary.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    for t, r in report.items():
        print(f"{t}: converged={r['all_monitored_converged']} max_rhat={r['max_rhat']:.4f} "
              f"min_bulk={r['min_ess_bulk']:.0f} min_tail={r['min_ess_tail']:.0f} "
              f"intvar mf={r['constant_mode']['integrated_variance_mean_free_only']:.4f} "
              f"total={r['constant_mode']['integrated_variance_analytic_total']:.4f} "
              f"floor violated={r['constant_mode']['floor_check_analytic']['violates_floor']} "
              f"KS p={r['constant_mode']['ks_pvalue_xi_c_vs_N01']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "production"])
    args = ap.parse_args()
    pilot() if args.mode == "pilot" else production()


if __name__ == "__main__":
    main()
