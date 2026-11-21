"""Milestone 5 analysis (pre-registered in docs/audit/M5_PREREGISTRATION.md).

  confirmation  structured confirmation chains (results/m5/chains/confirmation):
                criteria 1-6, written to results/m5/confirmation_summary.json
  replay        compare a fresh replay of chain 21 with the archive (criterion 7)
  laplace_real  Laplace vs converged MCMC per Hessian eigen-direction (smooth: M4
                production; structured: confirmation)
  seeds         confirmation (seeds 21-24) vs Milestone 4 production (seeds 11-14)
  sbc           aggregate results/m5/sbc/datasets/*.json
  laplace_sbc   aggregate results/m5/sbc/laplace/*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
M5 = ROOT / "results" / "m5"


def _chains(env, files):
    from m4_analyze import _load_hpcn, quantities
    arrs, states, extras = [], [], []
    for f in files:
        draws, phis, state, extra = _load_hpcn(f)
        q, names = quantities(env, draws, phis)
        qoi = np.load(f.with_name(f.stem + "_qoi.npz"))["log_q_r"]
        assert len(qoi) == len(q)
        arrs.append(np.column_stack([q, qoi]))
        states.append(state); extras.append(extra)
    return np.stack(arrs), names + ["log Q_R"], states, extras


def confirmation():
    from scipy import stats
    from m3_mcmc import build
    from src import mcmc_diagnostics as md
    from src.constant_mode import augment_with_independent_constant_mode, marginalized_moments
    from src.identifiability import check_posterior_variance_floor, mass_matrix
    env = build("structured")
    rk, prior = env["rk"], env["prior"]
    M = mass_matrix(env["dm"].V_m)
    files = sorted((M5 / "chains" / "confirmation").glob("structured_s2?.npz"))
    arr, names, states, extras = _chains(env, files)
    post = arr[:, arr.shape[1] // 2:, :]
    summ = md.summarize_all(post, names)
    Mc = post.shape[0]
    zmax = 0.0
    for k in range(post.shape[2]):
        pooled = post[:, :, k].mean()
        for c in range(Mc):
            mc = md.mcse_mean(post[c:c + 1, :, k])
            zmax = max(zmax, abs(post[c, :, k].mean() - pooled) / (mc * np.sqrt((Mc - 1) / Mc)))
    xi_post = post[:, :, :rk.r_reduced].reshape(-1, rk.r_reduced)
    mom = marginalized_moments(rk, xi_post)
    floor = check_posterior_variance_floor(np.sqrt(mom["var_total"]), prior, M)
    xi_c, _ = augment_with_independent_constant_mode(rk, xi_post, seed=5021)
    ks = stats.kstest(xi_c, "norm")
    finite = bool(np.isfinite(arr).all() and all(np.isfinite([s["rhat"], s["ess_bulk"], s["ess_tail"]]).all()
                                                for s in summ))
    crit = {
        "1_max_rhat_lt_1.01": max(s["rhat"] for s in summ) < 1.01,
        "2_min_bulk_ge_400": min(s["ess_bulk"] for s in summ) >= 400,
        "3_min_tail_ge_400": min(s["ess_tail"] for s in summ) >= 400,
        "4_finite": finite and all(s["n_nonfinite"] == 0 for s in states),
        "5_chain_agreement_z_le_4": bool(zmax <= 4),
        "6_constant_mode": bool(abs(float(rk.lambda_c) - 1.0) < 1e-12 and ks.pvalue > 0.01 and not floor["violates_floor"]),
    }
    out = {"files": [f.name for f in files], "n_proposals": [s["n_steps"] for s in states],
           "acceptance": [s["n_accept"] / s["n_steps"] for s in states],
           "ms_per_proposal": [1e3 * s["seconds"] / s["n_steps"] for s in states],
           "sampler_seconds_total": sum(s["seconds"] for s in states),
           "forward_solves_total": sum(e["forward_solves"] for e in extras),
           "adjoint_solves_total": sum(e["adjoint_solves"] for e in extras),
           "peak_rss_mb": max(e["peak_rss_mb"] for e in extras),
           "max_rhat": max(s["rhat"] for s in summ), "argmax_rhat": max(summ, key=lambda s: s["rhat"])["name"],
           "min_ess_bulk": min(s["ess_bulk"] for s in summ), "min_ess_tail": min(s["ess_tail"] for s in summ),
           "chain_agreement_max_z": zmax, "constant_mode": {"lambda_c": float(rk.lambda_c), "ks_p": float(ks.pvalue),
                                                             "floor": floor, "integrated_variance_mean_free":
                                                                 float(mom["var_mean_free"] @ (M @ np.ones(M.shape[0])))},
           "criteria_1_to_6": crit, "diagnostics": summ}
    (M5 / "confirmation_summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: out[k] for k in ("max_rhat", "argmax_rhat", "min_ess_bulk", "min_ess_tail",
                                          "chain_agreement_max_z", "criteria_1_to_6", "acceptance")}, indent=1, default=float))
    s6 = [s for s in summ if s["name"] == "xi_R[6]"][0]
    print("xi_R[6]:", {k: s6[k] for k in ("rhat", "ess_bulk", "ess_tail")})


def replay():
    from src.hessian_pcn import load_checkpoint
    a, _, sa, _ = load_checkpoint(M5 / "chains" / "confirmation" / "structured_s21.npz")
    b, pb, sb, _ = load_checkpoint(M5 / "chains" / "replay" / "structured_s21_replay.npz")
    pa = load_checkpoint(M5 / "chains" / "confirmation" / "structured_s21.npz")[1]
    n = len(b)
    out = {"replay_proposals": sb["n_steps"], "replay_draws": n,
           "draws_identical": bool(np.array_equal(a[:n], b)), "phis_identical": bool(np.array_equal(pa[:n], pb))}
    (M5 / "replay_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(out)


def laplace_real():
    from m3_mcmc import build
    from src.hessian_pcn import load_checkpoint
    out = {}
    sources = {"smooth": sorted((ROOT / "results/m4/chains/production").glob("smooth_s1?.npz")),
               "structured": sorted((M5 / "chains" / "confirmation").glob("structured_s2?.npz"))}
    for truth, files in sources.items():
        rec = json.loads((ROOT / f"results/m4/hessian_{truth}.json").read_text())
        H = np.array(rec["gauss_newton_misfit_hessian"])
        lam, V = np.linalg.eigh(0.5 * (H + H.T))
        xr = np.array(rec["xi_r_map"])
        post = []
        for f in files:
            d = load_checkpoint(f)[0]
            post.append(d[len(d) // 2:])
        z = np.vstack(post) @ V  # eigen-coordinates
        zmap = V.T @ xr
        sd_lap = 1 / np.sqrt(1 + lam)
        rows = [{"lambda": float(lam[i]), "laplace_sd": float(sd_lap[i]), "mcmc_sd": float(z[:, i].std()),
                 "variance_ratio_mcmc_over_laplace": float(z[:, i].var() / sd_lap[i] ** 2),
                 "mean_offset_in_laplace_sd": float((z[:, i].mean() - zmap[i]) / sd_lap[i]),
                 "skewness": float(((z[:, i] - z[:, i].mean()) ** 3).mean() / z[:, i].std() ** 3)}
                for i in range(len(lam))]
        out[truth] = rows
        print(truth)
        for r in rows:
            print("  lam=%9.3g  var ratio %.3f  mean offset %+.3f sd  skew %+.2f" % (
                r["lambda"], r["variance_ratio_mcmc_over_laplace"], r["mean_offset_in_laplace_sd"], r["skewness"]))
    (M5 / "laplace_vs_mcmc_real.json").write_text(json.dumps(out, indent=1) + "\n")


def seeds():
    from m3_mcmc import build
    from src import mcmc_diagnostics as md
    env = build("structured")
    out = []
    a, names, _, _ = _chains(env, sorted((ROOT / "results/m4/chains/production").glob("structured_s1?.npz")))
    b, _, _, _ = _chains(env, sorted((M5 / "chains" / "confirmation").glob("structured_s2?.npz")))
    pa, pb = a[:, a.shape[1] // 2:], b[:, b.shape[1] // 2:]
    for k, n in enumerate(names):
        ma, mb = pa[:, :, k].mean(), pb[:, :, k].mean()
        se = np.hypot(md.mcse_mean(pa[:, :, k]), md.mcse_mean(pb[:, :, k]))
        out.append({"name": n, "mean_m4": float(ma), "mean_m5": float(mb), "z": float((mb - ma) / se),
                    "sd_m4": float(pa[:, :, k].std()), "sd_m5": float(pb[:, :, k].std()),
                    "sd_ratio": float(pb[:, :, k].std() / pa[:, :, k].std())})
    (M5 / "seed_robustness.json").write_text(json.dumps(out, indent=1) + "\n")
    zs = [abs(r["z"]) for r in out]
    print(f"max |z| = {max(zs):.2f}; sd ratios {min(r['sd_ratio'] for r in out):.3f}..{max(r['sd_ratio'] for r in out):.3f}")


def _agg(records, key="quantities", L=99):
    from src import calibration as cal
    names = list(records[0][key].keys())
    res = {}
    for n in names:
        rows = [r[key][n] for r in records]
        pits = np.array([x["pit"] for x in rows])
        d = {"n": len(rows), "pit_ks": cal.pit_ks(pits), "coverage": cal.coverage_summary(pits)}
        if "rank" in rows[0]:
            d["rank_chi2"] = cal.rank_chi2(np.array([x["rank"] for x in rows]), L)
        if all("mean" in x and x.get("sd", 0) > 0 for x in rows):
            d["bias_rmse"] = cal.bias_rmse_spread([x["mean"] for x in rows], [x["sd"] for x in rows],
                                                  [x["truth"] for x in rows])
        res[n] = d
    return res


def _verdict(res, alpha=0.01):
    from src import calibration as cal
    tests = []
    for n, d in res.items():
        if n == "xi_c":
            continue  # analytic, reported separately
        tests.append((n, "rank" if "rank_chi2" in d else "pit", d["rank_chi2"]["p"] if "rank_chi2" in d else d["pit_ks"]["p"]))
    thr = alpha / len(tests)
    rejected = [t for t in tests if t[2] < thr]
    pairs = [(n, lv) for n, d in res.items() if n != "xi_c" for lv, c in d["coverage"].items()]
    excl = [(n, lv) for n, d in res.items() if n != "xi_c" for lv, c in d["coverage"].items() if not c["nominal_inside_ci"]]
    kmax = cal.max_coverage_exclusions(len(pairs))
    return {"bonferroni_threshold": thr, "n_tests": len(tests), "rejected": rejected,
            "coverage_pairs": len(pairs), "coverage_exclusions": excl, "max_allowed_exclusions": kmax,
            "consistent_with_calibration": bool(not rejected and len(excl) <= kmax)}


def sbc():
    recs = [json.loads(open(f).read()) for f in sorted(glob.glob(str(M5 / "sbc/datasets/ds_*.json")))]
    res = _agg(recs)
    lap = _agg(recs, key="laplace")
    v = _verdict(res)
    cost = {k: float(np.mean([r["cost"][k] for r in recs])) for k in recs[0]["cost"]}
    diag = {"max_rhat": [r["max_rhat"] for r in recs], "min_ess_bulk": [r["min_ess_bulk"] for r in recs],
            "min_ess_tail": [r["min_ess_tail"] for r in recs],
            "n_rhat_gt_1.05": int(sum(r["max_rhat"] > 1.05 for r in recs)),
            "n_rhat_gt_1.01": int(sum(r["max_rhat"] > 1.01 for r in recs)),
            "map_converged": int(sum(r["map"]["converged"] for r in recs))}
    sens = None
    good = [r for r in recs if r["max_rhat"] <= 1.05]
    if 0 < len(good) < len(recs):
        sens = {"n": len(good), "verdict": _verdict(_agg(good))}
    mf = [r["mean_free_rel_L2_error_posterior_mean"] for r in recs]
    out = {"n_datasets": len(recs), "verdict": v, "quantities": res, "laplace_on_same_datasets": lap,
           "laplace_verdict": _verdict(lap), "per_dataset_diagnostics": diag, "sensitivity_rhat_le_1.05": sens,
           "mean_cost_per_dataset": cost, "total_cpu_seconds": float(sum(r["cost"]["seconds_total"] for r in recs)),
           "mean_free_rel_L2_error_posterior_mean": {"mean": float(np.mean(mf)), "median": float(np.median(mf))}}
    (M5 / "sbc_summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"datasets {len(recs)}; verdict {v['consistent_with_calibration']}; rejected {v['rejected']}; "
          f"exclusions {len(v['coverage_exclusions'])}/{v['coverage_pairs']} (max {v['max_allowed_exclusions']})")
    print("diag:", {k: (diag[k] if isinstance(diag[k], int) else [round(min(diag[k]), 3), round(max(diag[k]), 3)]) for k in diag})
    print("mean cost:", {k: round(v_, 1) for k, v_ in cost.items()})
    for n, d in res.items():
        cv = d["coverage"]
        tst = d.get("rank_chi2", d["pit_ks"])["p"]
        br = d.get("bias_rmse", {})
        print(f"  {n:24s} p={tst:.3f} cov50={cv['0.5']['rate']:.2f} cov80={cv['0.8']['rate']:.2f} "
              f"cov90={cv['0.9']['rate']:.2f} cov95={cv['0.95']['rate']:.2f} "
              f"zbias={br.get('standardized_bias', float('nan')):+.2f} ss={br.get('spread_skill_ratio', float('nan')):.2f}")


def laplace_sbc():
    out = {}
    for tag in ("sigma1", "sigma0.5", "sigma2"):
        recs = []
        for f in sorted(glob.glob(str(M5 / f"sbc/laplace/{tag}_*.json"))):
            recs += json.loads(open(f).read())["datasets"]
        if not recs:
            continue
        res = _agg(recs)
        out[tag] = {"n": len(recs), "verdict": _verdict(res), "quantities": res,
                    "map_converged": int(sum(r["map_converged"] for r in recs)),
                    "mean_seconds": float(np.mean([r["seconds"] for r in recs]))}
        v = out[tag]["verdict"]
        print(f"{tag}: n={len(recs)} consistent={v['consistent_with_calibration']} rejected={len(v['rejected'])} "
              f"exclusions={len(v['coverage_exclusions'])}/{v['coverage_pairs']}")
    (M5 / "laplace_sbc_summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["confirmation", "replay", "laplace_real", "seeds", "sbc", "laplace_sbc"])
    {"confirmation": confirmation, "replay": replay, "laplace_real": laplace_real, "seeds": seeds,
     "sbc": sbc, "laplace_sbc": laplace_sbc}[ap.parse_args().mode]()


if __name__ == "__main__":
    main()
