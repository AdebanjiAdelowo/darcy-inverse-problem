"""Milestone 5 supplementary analyses on the 44 completed SBC datasets (no new inference).

  e2         pre-registered E2: Laplace vs MCMC on the SBC datasets themselves
             (per-quantity posterior variance ratio and mean offset)
  subsets    POST HOC (not pre-registered as a test): descriptive calibration of the
             13 datasets with max R-hat > 1.05 and of the 31 with max R-hat <= 1.05,
             with a two-sample comparison of standardised errors
  laplace_bias  POST HOC: standardised bias of the Laplace mean (MAP) for every quantity,
             split by MAP convergence, in each E1 noise level, and Laplace vs MCMC
             standardised bias on the 44 SBC datasets (same truths and data)
  power      POST HOC: Monte Carlo power of the pre-registered verdict at D = 44,
             L = 99 for known miscalibrations (sd scaled by s; mean shifted by d sd),
             using independent exact Gaussian draws (an optimistic power bound,
             because the real rank draws are autocorrelated)

    OMP_NUM_THREADS=1 python scripts/m5_supplementary.py
Output: results/m5/sbc_supplementary.json. The primary verdict is the one in
results/m5/sbc_summary.json; nothing here changes it.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import calibration as cal  # noqa: E402

M5 = ROOT / "results" / "m5"


def load():
    return [json.loads(open(f).read()) for f in sorted(glob.glob(str(M5 / "sbc/datasets/ds_*.json")))]


def e2(recs):
    out = {}
    for n in recs[0]["laplace"]:
        vr = np.array([r["quantities"][n]["sd"] ** 2 / r["laplace"][n]["sd"] ** 2 for r in recs])
        off = np.array([(r["quantities"][n]["mean"] - r["laplace"][n]["mean"]) / r["laplace"][n]["sd"] for r in recs])
        out[n] = {"variance_ratio_mcmc_over_laplace": {"median": float(np.median(vr)), "q10": float(np.quantile(vr, 0.1)),
                                                        "q90": float(np.quantile(vr, 0.9)), "max": float(vr.max())},
                  "mean_offset_in_laplace_sd": {"median": float(np.median(off)), "max_abs": float(np.abs(off).max()),
                                                "median_abs": float(np.median(np.abs(off)))}}
    return out


def _describe(recs, names):
    res = {}
    for n in names:
        rows = [r["quantities"][n] for r in recs]
        pits = np.array([x["pit"] for x in rows])
        d = {"n": len(rows), "pit_ks_p": cal.pit_ks(pits)["p"],
             "coverage": {lv: {"rate": c["rate"], "ci95": c["ci95"]} for lv, c in cal.coverage_summary(pits).items()}}
        if "rank" in rows[0]:
            # 5 bins so that expected counts stay >= 2.6 at n = 13
            d["rank_chi2_5bins_p"] = cal.rank_chi2(np.array([x["rank"] for x in rows]), 99, bins=5)["p"]
        if all(x.get("sd", 0) > 0 for x in rows):
            z = np.array([(x["mean"] - x["truth"]) / x["sd"] for x in rows])
            d["standardized_error"] = {"mean": float(z.mean()), "rms": float(np.sqrt(np.mean(z ** 2)))}
        res[n] = d
    return res


def subsets(recs):
    names = list(recs[0]["quantities"].keys())
    hi = [r for r in recs if r["max_rhat"] > 1.05]
    lo = [r for r in recs if r["max_rhat"] <= 1.05]
    out = {"high_rhat_ids": [r["dataset"] for r in hi], "n_high": len(hi), "n_low": len(lo),
           "high_rhat_gt_1.05": _describe(hi, names), "low_rhat_le_1.05": _describe(lo, names)}
    # pooled over quantities: are |PIT - 0.5| or squared standardised errors larger in the high-R-hat subset?
    def pooled(rs, key):
        if key == "pit":
            return np.array([abs(r["quantities"][n]["pit"] - 0.5) for r in rs for n in names])
        return np.array([((r["quantities"][n]["mean"] - r["quantities"][n]["truth"]) / r["quantities"][n]["sd"]) ** 2
                         for r in rs for n in names if r["quantities"][n].get("sd", 0) > 0])
    tests = {}
    for key in ("pit", "z2"):
        a, b = pooled(hi, key), pooled(lo, key)
        # dataset-level means to respect within-dataset correlation across quantities
        da = np.array([np.mean(pooled([r], key)) for r in hi])
        db = np.array([np.mean(pooled([r], key)) for r in lo])
        tests[key] = {"mean_high": float(a.mean()), "mean_low": float(b.mean()),
                      "dataset_level_mannwhitney_p": float(stats.mannwhitneyu(da, db).pvalue)}
    out["pooled_comparison"] = tests
    rel = np.array([r["max_rel_mcse"] for r in recs])
    out["per_dataset_max_rel_mcse"] = {"median": float(np.median(rel)), "max": float(rel.max()),
                                       "n_gt_0.1": int((rel > 0.1).sum())}
    return out


def _zb(rows):
    z = np.array([(x["mean"] - x["truth"]) / x["sd"] for x in rows])
    return {"n": len(z), "mean": float(z.mean()), "se": float(z.std(ddof=1) / np.sqrt(len(z))) if len(z) > 1 else None,
            "rms": float(np.sqrt(np.mean(z ** 2)))}


def laplace_bias(recs):
    names = list(recs[0]["laplace"].keys())
    out = {"sbc_44_same_datasets": {n: {"laplace": _zb([r["laplace"][n] for r in recs]),
                                        "mcmc": _zb([r["quantities"][n] for r in recs])} for n in names}}
    for tag in ("sigma1", "sigma0.5", "sigma2"):
        rs = []
        for f in sorted(glob.glob(str(M5 / f"sbc/laplace/{tag}_*.json"))):
            rs += json.loads(open(f).read())["datasets"]
        if not rs:
            continue
        d = {"n": len(rs), "n_map_converged": int(sum(r["map_converged"] for r in rs))}
        for label, sub in (("all", rs), ("map_converged", [r for r in rs if r["map_converged"]]),
                           ("map_not_converged", [r for r in rs if not r["map_converged"]])):
            if len(sub) >= 5:
                d[label] = {n: _zb([r["quantities"][n] for r in sub]) for n in names}
        out[tag] = d
    return out


def power(D=44, L=99, n_q=27, n_rep=2000, seed=0):
    """Probability that the pre-registered verdict flags miscalibration of ONE quantity
    (rank chi-square at Bonferroni 0.01 / n_q, or a coverage interval excluding nominal),
    treating the other quantities as calibrated. Exact posterior N(mu, 1); the reported
    posterior is N(mu + d, s^2)."""
    rng = np.random.default_rng(seed)
    thr = 0.01 / n_q
    out = []
    for s, d in [(1.0, 0.0), (0.8, 0.0), (0.7, 0.0), (0.5, 0.0), (1.25, 0.0), (1.5, 0.0), (2.0, 0.0),
                 (1.0, 0.25), (1.0, 0.5), (1.0, 1.0)]:
        rank_rej = cov_excl = 0
        for _ in range(n_rep):
            # truth relative to exact posterior is N(0,1); the reported posterior is N(d, s^2)
            t = rng.standard_normal(D)
            draws = d + s * rng.standard_normal((D, L))
            ranks = (draws < t[:, None]).sum(1)
            p = cal.rank_chi2(ranks, L)["p"]
            rank_rej += p < thr
            pit = stats.norm.cdf((t - d) / s)
            cs = cal.coverage_summary(pit)
            cov_excl += any(not c["nominal_inside_ci"] for c in cs.values())
        out.append({"sd_scale": s, "mean_shift_sd": d, "p_rank_rejects": rank_rej / n_rep,
                    "p_any_coverage_interval_excludes_nominal": cov_excl / n_rep})
    return {"D": D, "L": L, "bonferroni_threshold": thr, "n_rep": n_rep, "rows": out}


def main():
    recs = load()
    assert len(recs) == 44
    res = {"e2_laplace_vs_mcmc_on_sbc": e2(recs), "post_hoc_rhat_subsets": subsets(recs), "post_hoc_laplace_bias": laplace_bias(recs),
           "post_hoc_power": power()}
    (M5 / "sbc_supplementary.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    for n, v in res["e2_laplace_vs_mcmc_on_sbc"].items():
        print(f"E2 {n:18s} var ratio median {v['variance_ratio_mcmc_over_laplace']['median']:.3f} "
              f"[q10 {v['variance_ratio_mcmc_over_laplace']['q10']:.3f}, q90 {v['variance_ratio_mcmc_over_laplace']['q90']:.3f}, "
              f"max {v['variance_ratio_mcmc_over_laplace']['max']:.2f}]  |offset| median "
              f"{v['mean_offset_in_laplace_sd']['median_abs']:.3f} max {v['mean_offset_in_laplace_sd']['max_abs']:.2f}")
    s = res["post_hoc_rhat_subsets"]
    print("subsets", s["n_high"], s["n_low"], s["pooled_comparison"], s["per_dataset_max_rel_mcse"])
    lb = res["post_hoc_laplace_bias"]
    for n, v in lb["sbc_44_same_datasets"].items():
        print(f"bias on 44 SBC datasets {n:18s} Laplace {v['laplace']['mean']:+.2f} MCMC {v['mcmc']['mean']:+.2f} "
              f"(se {v['mcmc']['se']:.2f})")
    for tag in ("sigma1", "sigma0.5", "sigma2"):
        if tag in lb and "map_converged" in lb[tag]:
            c = lb[tag]["map_converged"]
            print(tag, "MAP-converged n", lb[tag]["n_map_converged"],
                  {n: round(c[n]["mean"], 2) for n in ("xi_R[3]", "xi_R[4]", "m_R(0.5, 0.5)")})
    for r in res["post_hoc_power"]["rows"]:
        print("power", r)


if __name__ == "__main__":
    main()
