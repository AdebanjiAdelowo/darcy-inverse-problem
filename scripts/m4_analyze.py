"""Milestone 4 analysis (pre-registered in docs/audit/M4_PREREGISTRATION.md).

    python scripts/m4_analyze.py pilot        -> results/m4/pilot_summary.json
    python scripts/m4_analyze.py production   -> results/m4/production_summary.json, figures

Monitored quantities: 14 mean-free KL coefficients xi_R, Phi, the mean-free field
m_R at the five probe points, ||m_R||_L2, and (production) log Q_R, the log
effective permeability of the mean-free field.
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
M4 = ROOT / "results" / "m4"
M3 = ROOT / "results" / "m3" / "mcmc"
PROBES = ["(0.25,0.25)", "(0.75,0.25)", "(0.5,0.5)", "(0.25,0.75)", "(0.75,0.75)"]


def quantities(env, xi, phi):
    rk = env["rk"]
    lam = rk.prior.eigenvalues[rk.keep]
    probes = rk.mean_free_fields(xi)[:, env["probe_idx"]]
    norm = np.sqrt((xi ** 2 * lam[None, :]).sum(axis=1))  # KL modes are L2-orthonormal
    names = [f"xi_R[{j}]" for j in range(rk.r_reduced)] + ["Phi"] + [f"m_R{p}" for p in PROBES] + ["||m_R||_L2"]
    return np.column_stack([xi, phi, probes, norm]), names


def _load_hpcn(path):
    from src.hessian_pcn import load_checkpoint
    draws, phis, state, extra = load_checkpoint(path)
    return draws, phis, state, extra


def group_stats(arr, names, seconds, n_proposals, forward_solves, peak_rss=None):
    from src import mcmc_diagnostics as md
    post = arr[:, arr.shape[1] // 2:, :]
    summ = md.summarize_all(post, names)
    min_bulk = min(s["ess_bulk"] for s in summ)
    return {
        "max_rhat": max(s["rhat"] for s in summ), "min_ess_bulk": min_bulk,
        "min_ess_tail": min(s["ess_tail"] for s in summ),
        "worst_quantity": min(summ, key=lambda s: s["ess_bulk"])["name"],
        "sampler_seconds_total": seconds, "proposals_total": n_proposals,
        "forward_solves_total": forward_solves,
        "min_bulk_ess_per_second": min_bulk / seconds,
        "min_bulk_ess_per_1000_proposals_post_warmup": 1000 * min_bulk / (n_proposals / 2),
        "max_rel_mcse_mean": max(s["mcse_mean"] / s["sd"] for s in summ if s["sd"] > 0),
        "peak_rss_mb": peak_rss, "diagnostics": summ,
    }


def pilot():
    from m3_mcmc import build
    out = {"hpcn": [], "pcn_reference": {}, "prior_start_transient": {}}
    for truth in ("smooth", "structured"):
        env = build(truth)
        # HpCN, Laplace-over-dispersed starts
        for rho in ("0.25", "0.5", "0.75", "1.0"):
            files = sorted((M4 / "chains" / "pilot").glob(f"{truth}_r{rho}_laplace3_s*.npz"))
            arrs, sec, nprop, fs, acc, rss = [], 0.0, 0, 0, [], []
            for f in files:
                draws, phis, state, extra = _load_hpcn(f)
                q, names = quantities(env, draws, phis)
                arrs.append(q)
                sec += state["seconds"]; nprop += state["n_steps"]; fs += extra["forward_solves"]
                acc.append(state["n_accept"] / state["n_steps"]); rss.append(extra["peak_rss_mb"])
            g = group_stats(np.stack(arrs), names, sec, nprop, fs, max(rss))
            g.update({"truth": truth, "rho": float(rho), "n_chains": len(files), "acceptance": acc})
            g.pop("diagnostics")
            out["hpcn"].append(g)
        # plain pCN reference: first 4,000 proposals of the Milestone 3 production chains (thin 10)
        arrs, sec, nprop = [], 0.0, 0
        for f in sorted((M3 / "production").glob(f"{truth}_s*.npz")):
            d = np.load(f)
            meta = json.loads(str(d["meta"]))
            q, names = quantities(env, d["xi_r"][:400], d["phi"][:400])
            arrs.append(q)
            sec += meta["pcn_seconds"] * 4000 / meta["n_proposals_done"]; nprop += 4000
        g = group_stats(np.stack(arrs), names, sec, nprop, nprop + 4)
        g.pop("diagnostics")
        g["note"] = "Milestone 3 chains, first 4,000 proposals, stored with thin 10; prior-scale starts"
        out["pcn_reference"][truth] = g
        # burn-in from prior-scale starts (rho 0.75): first proposal after which Phi stays below
        # the 99th percentile of post-warmup Phi of the Laplace-started rho 0.75 chains
        ref = np.concatenate([_load_hpcn(f)[1][2000:] for f in
                              sorted((M4 / "chains" / "pilot").glob(f"{truth}_r0.75_laplace3_s*.npz"))])
        thr = float(np.quantile(ref, 0.99))
        hits = []
        for f in sorted((M4 / "chains" / "pilot").glob(f"{truth}_r0.75_prior_s*.npz")):
            phis = _load_hpcn(f)[1]
            above = np.nonzero(phis > thr)[0]
            hits.append(int(above[-1] + 1) if len(above) else 0)
        out["prior_start_transient"][truth] = {"phi_threshold": thr, "proposals_until_phi_settles": hits}
    sel = {}
    for truth in ("smooth", "structured"):
        rows = [r for r in out["hpcn"] if r["truth"] == truth]
        best = max(rows, key=lambda r: r["min_bulk_ess_per_second"])
        ref = out["pcn_reference"][truth]
        sel[truth] = {"rho": best["rho"], "min_bulk_ess_per_second": best["min_bulk_ess_per_second"],
                      "pcn_min_bulk_ess_per_second": ref["min_bulk_ess_per_second"],
                      "ratio_vs_pcn": best["min_bulk_ess_per_second"] / ref["min_bulk_ess_per_second"],
                      "min_bulk_ess_per_proposal_post_warmup":
                          best["min_bulk_ess_per_1000_proposals_post_warmup"] / 1000}
    go = all(s["ratio_vs_pcn"] >= 10 for s in sel.values())
    rate = min(s["min_bulk_ess_per_proposal_post_warmup"] for s in sel.values())
    n = int(np.ceil(500 / rate / 5000) * 5000)
    out["selection"] = sel
    out["go_to_production"] = go
    out["production_length_rule"] = {"predicted_rate": rate, "n_uncapped": n, "n_per_chain": min(n, 50_000),
                                     "cap": 50_000}
    (M4 / "pilot_summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    for r in out["hpcn"]:
        print(f"{r['truth']:10s} rho={r['rho']:<5} acc={np.mean(r['acceptance']):.3f} maxRhat={r['max_rhat']:.3f} "
              f"minBulk={r['min_ess_bulk']:7.1f} minTail={r['min_ess_tail']:7.1f} ESS/s={r['min_bulk_ess_per_second']:.3f} "
              f"worst={r['worst_quantity']} rss={r['peak_rss_mb']:.0f}MB t={r['sampler_seconds_total']:.0f}s")
    for t, r in out["pcn_reference"].items():
        print(f"pCN ref {t}: maxRhat={r['max_rhat']:.3f} minBulk={r['min_ess_bulk']:.1f} ESS/s={r['min_bulk_ess_per_second']:.4f}")
    print(json.dumps({k: out[k] for k in ("selection", "go_to_production", "production_length_rule",
                                          "prior_start_transient")}, indent=1, default=float))


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
    from src.qoi import effective_permeability

    figdir = M4 / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    report = {}
    for truth in ("smooth", "structured"):
        files = sorted((M4 / "chains" / "production").glob(f"{truth}_s*.npz"))
        files = [f for f in files if not f.name.endswith("_qoi.npz")]
        if not files:
            continue
        env = build(truth)
        rk, prior = env["rk"], env["prior"]
        M = mass_matrix(env["dm"].V_m)
        arrs, states, extras = [], [], []
        for f in files:
            draws, phis, state, extra = _load_hpcn(f)
            q, names = quantities(env, draws, phis)
            qoi = np.load(f.with_name(f.stem + "_qoi.npz"))["log_q_r"]
            assert len(qoi) == len(q)
            arrs.append(np.column_stack([q, qoi]))
            states.append(state); extras.append(extra)
        names = names + ["log Q_R"]
        arr = np.stack(arrs)
        sec = sum(s["seconds"] for s in states)
        nprop = sum(s["n_steps"] for s in states)
        fs = sum(e["forward_solves"] for e in extras)
        g = group_stats(arr, names, sec, nprop, fs, max(e["peak_rss_mb"] for e in extras))
        post = arr[:, arr.shape[1] // 2:, :]
        Mc = post.shape[0]
        # calibrated chain-agreement test (Deviation 2) and the original statistic
        zmax, orig_max = 0.0, 0.0
        for k in range(post.shape[2]):
            pooled = post[:, :, k].mean()
            pooled_mcse = md.mcse_mean(post[:, :, k])
            for c in range(Mc):
                mc = md.mcse_mean(post[c:c + 1, :, k])
                zmax = max(zmax, abs(post[c, :, k].mean() - pooled) / (mc * np.sqrt((Mc - 1) / Mc)))
            cm = post[:, :, k].mean(axis=1)
            orig_max = max(orig_max, (cm.max() - cm.min()) / pooled_mcse)
        xi_post = post[:, :, :rk.r_reduced].reshape(-1, rk.r_reduced)
        mom = marginalized_moments(rk, xi_post)
        floor = check_posterior_variance_floor(np.sqrt(mom["var_total"]), prior, M)
        mf_int = float(mom["var_mean_free"] @ (M @ np.ones(M.shape[0])))
        xi_c, fields_aug = augment_with_independent_constant_mode(rk, xi_post, seed=4040 + len(truth))
        ks = stats.kstest(xi_c, "norm")
        var_aug = fields_aug.var(axis=0, ddof=1)
        sd_c = float(np.sqrt(rk.lambda_c) * abs(rk.phi_c[0]))
        mf_fields = rk.mean_free_fields(xi_post)
        logq = post[:, :, -1].ravel()
        q_true = float(np.log(effective_permeability(env["dm"], env["m_true"])))
        ci = []
        for j, idx in enumerate(env["probe_idx"]):
            rb = rao_blackwell_quantiles(mf_fields[:, idx], sd_c, [0.025, 0.975])
            ci.append({"probe": PROBES[j], "m_true": float(env["m_true"][idx]), "absolute_95": rb.tolist(),
                       "mean_free_95": np.quantile(mf_fields[:, idx], [0.025, 0.975]).tolist(),
                       "m_true_mean_free": float(env["m_true"][idx] - env["m_true"].mean())})
        keff_ci = rao_blackwell_quantiles(logq, sd_c, [0.025, 0.5, 0.975]).tolist()
        conv = (g["max_rhat"] < 1.01 and g["min_ess_bulk"] >= 400 and g["min_ess_tail"] >= 400 and zmax <= 4
                and ks.pvalue > 0.01 and not floor["violates_floor"]
                and all(e["n_nonfinite"] == 0 for e in states)
                and all(np.isfinite(a).all() for a in arrs))
        report[truth] = {
            "files": [f.name for f in files], "rho": states[0]["proposal"]["config"]["rho"],
            "n_proposals_per_chain": [s["n_steps"] for s in states],
            "acceptance": [s["n_accept"] / s["n_steps"] for s in states],
            "n_nonfinite": [s["n_nonfinite"] for s in states],
            "ms_per_proposal": [1e3 * s["seconds"] / s["n_steps"] for s in states],
            "forward_solves_total": fs, "adjoint_solves_total": sum(e["adjoint_solves"] for e in extras),
            **{k: g[k] for k in ("max_rhat", "min_ess_bulk", "min_ess_tail", "worst_quantity",
                                 "sampler_seconds_total", "min_bulk_ess_per_second", "max_rel_mcse_mean",
                                 "peak_rss_mb")},
            "chain_agreement_max_z": zmax, "chain_agreement_original_statistic_max": orig_max,
            "converged_all_criteria": bool(conv),
            "diagnostics": g["diagnostics"],
            "constant_mode": {"integrated_variance_mean_free": mf_int,
                              "integrated_variance_total": floor["integrated_variance"],
                              "constant_mode_integrated_variance": float(rk.lambda_c),
                              "pointwise_total_min": floor["pointwise_variance_min"],
                              "pointwise_total_max": floor["pointwise_variance_max"],
                              "violates_floor": floor["violates_floor"],
                              "augmented_vs_analytic_max_rel_diff": float(np.max(np.abs(var_aug / mom["var_total"] - 1))),
                              "ks_pvalue_xi_c": float(ks.pvalue)},
            "posterior_mean_errors": reconstruction_errors(mom["mean"], env["m_true"], M),
            "credible_intervals": ci,
            "log_effective_permeability": {"true_log_k_eff": q_true,
                                           "posterior_median_and_95": keff_ci,
                                           "mean_free_part_mean": float(logq.mean()),
                                           "mean_free_part_sd": float(logq.std(ddof=1))},
        }
        # figures
        summ = g["diagnostics"]
        show = sorted(range(len(summ)), key=lambda k: summ[k]["ess_bulk"])[:3] + [len(names) - 1]
        fig, axes = plt.subplots(len(show), 2, figsize=(12, 2.6 * len(show)))
        thin = extras[0]["thin"]
        n = arr.shape[1]
        for row, k in enumerate(show):
            for c in range(arr.shape[0]):
                axes[row, 0].plot(np.arange(n) * thin, arr[c, :, k], lw=0.4, label=f"chain {c + 1}")
            axes[row, 0].axvline(n // 2 * thin, color="k", ls="--", lw=0.8)
            s = summ[k]
            axes[row, 0].set_title(f"{names[k]}: R-hat {s['rhat']:.3f}, bulk ESS {s['ess_bulk']:.0f}, "
                                   f"tail ESS {s['ess_tail']:.0f}", fontsize=9)
            ac = md.autocorrelation(post[:, :, k])
            lags = np.arange(min(200, ac.shape[1]))
            for c in range(ac.shape[0]):
                axes[row, 1].plot(lags * thin, ac[c, :len(lags)], lw=0.8)
            axes[row, 1].axhline(0, color="k", lw=0.5)
            axes[row, 1].set_title(f"autocorrelation (post-warmup), {names[k]}", fontsize=9)
        axes[-1, 0].set_xlabel("proposal")
        axes[-1, 1].set_xlabel("lag (proposals)")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / f"hpcn_traces_{truth}.png", dpi=110)
        plt.close(fig)
    # comparison with Milestone 3 plain-pCN production
    m3 = json.loads((M3 / "production_summary.json").read_text())
    comp = {}
    for truth, r in report.items():
        p = m3[truth]
        pcn_sec = sum(h * 3600 for h in p["pcn_hours_per_chain"])
        comp[truth] = {
            "acceptance": {"pcn": float(np.mean(p["acceptance_rates"])), "hpcn": float(np.mean(r["acceptance"]))},
            "max_rhat": {"pcn": p["max_rhat"], "hpcn": r["max_rhat"]},
            "min_bulk_ess": {"pcn": p["min_ess_bulk"], "hpcn": r["min_ess_bulk"]},
            "min_tail_ess": {"pcn": p["min_ess_tail"], "hpcn": r["min_ess_tail"]},
            "min_bulk_ess_per_second": {"pcn": p["min_ess_bulk"] / pcn_sec, "hpcn": r["min_bulk_ess_per_second"]},
            "min_bulk_ess_per_1000_pde_solves": {"pcn": 1000 * p["min_ess_bulk"] / sum(p["forward_solves"]),
                                                 "hpcn": 1000 * r["min_ess_bulk"] / r["forward_solves_total"]},
            "pde_solves": {"pcn": sum(p["forward_solves"]), "hpcn": r["forward_solves_total"]},
            "sampler_seconds": {"pcn": pcn_sec, "hpcn": r["sampler_seconds_total"]},
            "constant_mode_integrated_variance": {"pcn": 1.0, "hpcn": r["constant_mode"]["constant_mode_integrated_variance"]},
            "mean_free_integrated_variance": {"pcn": p["constant_mode"]["integrated_variance_mean_free_only"],
                                              "hpcn": r["constant_mode"]["integrated_variance_mean_free"]},
            "posterior_mean_rel_error_meanfree_L2": {"pcn": p["posterior_mean_errors"]["rel_error_meanfree_L2"],
                                                    "hpcn": r["posterior_mean_errors"]["rel_error_meanfree_L2"]},
            "note": "pCN: Milestone 3 chains (4 x 100,000, prior-scale starts, analytic constant mode); "
                    "pCN monitored set excludes ||m_R|| and log Q_R",
        }
    report["comparison_vs_m3_pcn"] = comp
    (M4 / "production_summary.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    for t in ("smooth", "structured"):
        if t in report:
            r = report[t]
            print(f"{t}: converged={r['converged_all_criteria']} maxRhat={r['max_rhat']:.4f} minBulk={r['min_ess_bulk']:.0f} "
                  f"minTail={r['min_ess_tail']:.0f} zmax={r['chain_agreement_max_z']:.2f} worst={r['worst_quantity']} "
                  f"acc={np.mean(r['acceptance']):.3f} ESS/s={r['min_bulk_ess_per_second']:.2f}")
    print(json.dumps(comp, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "production"])
    pilot() if ap.parse_args().mode == "pilot" else production()


if __name__ == "__main__":
    main()
