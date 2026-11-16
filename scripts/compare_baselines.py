"""Compare reproduced baseline outputs (scripts/reproduce_baselines.py) with the
stored historical artifacts in results/.

    python scripts/compare_baselines.py --tag single_thread
    python scripts/compare_baselines.py --tag historical_threads

Writes results/m2/baseline_comparison_TAG.json. Wall-clock timings ("time=...s")
are excluded from numeric comparison and reported separately, and the
"MAP-candidate (optimizer not converged)" wording (a presentation label ported
from the earlier audit branch) is normalised to "MAP" before comparison.

Verdicts per experiment:
  bitwise            - text identical after the two normalisations above, and
                       saved arrays identical;
  roundoff           - every printed number within relative 1e-9 (or absolute
                       1e-12), booleans/integers identical;
  printed_precision  - every headline number identical to its printed
                       precision, other numbers within relative 1e-3;
  discrepancy        - anything else (explained in the Milestone 2 report).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPRO = ROOT / "results" / "m2" / "reproduction"

PAIRS = {
    "verify_forward": ["forward_verification.txt"],
    "verify_gradient": ["gradient_verification.txt"],
    "verify_bayesian_gradient": ["bayesian_gradient_verification.txt"],
    "validate_sampler": ["sampler_validation.txt"],
    "reconstruction_demo": ["reconstruction_demo_full.txt"],
    "regularization_study": ["regularization_study_full.txt"],
    "noise_study": ["noise_study_full.txt"],
    "sensor_study": ["sensor_study_full.txt"],
    "posterior_demo": ["posterior_demo_full.txt", "posterior_samples_smooth_full.npz",
                       "posterior_samples_structured_full.npz"],
}

NUM = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|\bnan\b|\bTrue\b|\bFalse\b")
TIME = re.compile(r"time=([0-9.]+)s")


def _normalise(line: str) -> str:
    line = line.replace("MAP-candidate (optimizer not converged)", "MAP")
    return TIME.sub("time=<t>s", line)


def _tokens(line: str) -> list[str]:
    return NUM.findall(_normalise(line))


def _cmp_tokens(a: str, b: str):
    if a in ("True", "False", "nan") or b in ("True", "False", "nan"):
        return (0.0, 0.0) if a == b else (np.inf, np.inf)
    fa, fb = float(a), float(b)
    absd = abs(fa - fb)
    reld = absd / abs(fa) if fa != 0 else (0.0 if absd == 0 else np.inf)
    return absd, reld


def compare_text(hist: pathlib.Path, rep: pathlib.Path) -> dict:
    hl, rl = hist.read_text().splitlines(), rep.read_text().splitlines()
    out = {"historical": str(hist.relative_to(ROOT)), "reproduced": str(rep.relative_to(ROOT)),
           "n_lines": [len(hl), len(rl)], "raw_text_identical": hl == rl}
    norm_identical = [_normalise(x) for x in hl] == [_normalise(x) for x in rl]
    max_abs, max_rel, n_diff_tokens, n_tokens, structural = 0.0, 0.0, 0, 0, []
    printed_identical = True
    worst = []
    for i, (a, b) in enumerate(zip(hl, rl)):
        ta, tb = _tokens(a), _tokens(b)
        if len(ta) != len(tb):
            structural.append(i + 1)
            continue
        for x, y in zip(ta, tb):
            n_tokens += 1
            if x != y:
                printed_identical = False
                n_diff_tokens += 1
            d_abs, d_rel = _cmp_tokens(x, y)
            if d_abs > 0:
                worst.append((d_rel, i + 1, x, y))
            max_abs, max_rel = max(max_abs, d_abs), max(max_rel, d_rel)
    if len(hl) != len(rl):
        structural.append("line count")
    worst.sort(reverse=True)
    timings = [(TIME.findall(a), TIME.findall(b)) for a, b in zip(hl, rl) if TIME.search(a)]
    out.update({
        "normalised_text_identical": norm_identical,
        "printed_numbers_identical": printed_identical and not structural,
        "n_numeric_tokens": n_tokens, "n_differing_tokens": n_diff_tokens,
        "max_abs_difference": max_abs, "max_rel_difference": max_rel,
        "largest_differences": [{"rel": r, "line": ln, "historical": x, "reproduced": y}
                                for r, ln, x, y in worst[:5]],
        "structural_mismatch_lines": structural,
        "timings_seconds": [{"historical": [float(t) for t in a], "reproduced": [float(t) for t in b]}
                            for a, b in timings],
    })
    return out


def compare_npz(hist: pathlib.Path, rep: pathlib.Path) -> dict:
    a, b = np.load(hist), np.load(rep)
    keys = {}
    for k in a.files:
        if k not in b.files:
            keys[k] = {"missing": True}
            continue
        x, y = a[k], b[k]
        same_shape = x.shape == y.shape
        keys[k] = {"shape": list(x.shape), "identical": bool(same_shape and np.array_equal(x, y)),
                   "max_abs_difference": float(np.max(np.abs(x - y))) if same_shape else None}
    out = {"historical": str(hist.relative_to(ROOT)), "reproduced": str(rep.relative_to(ROOT)),
           "all_identical": all(v.get("identical", False) for v in keys.values()), "arrays": keys}
    if "xi_samples" in a.files and "xi_samples" in b.files and a["xi_samples"].shape == b["xi_samples"].shape:
        # accept/reject pattern of the retained chain: a step "moved" if any coordinate changed
        ma = np.any(np.diff(a["xi_samples"], axis=0) != 0, axis=1)
        mb = np.any(np.diff(b["xi_samples"], axis=0) != 0, axis=1)
        out["chain_accept_pattern_identical"] = bool(np.array_equal(ma, mb))
        out["chain_moves"] = [int(ma.sum()), int(mb.sum())]
        out["max_abs_difference_of_sample_means"] = float(
            np.max(np.abs(a["xi_samples"].mean(0) - b["xi_samples"].mean(0))))
    return out


def verdict(entries: list[dict]) -> str:
    texts = [e for e in entries if "normalised_text_identical" in e]
    arrays = [e for e in entries if "arrays" in e]
    if any(e.get("missing_reproduction") for e in entries):
        return "not_run"
    if all(e["normalised_text_identical"] for e in texts) and all(e["all_identical"] for e in arrays):
        return "bitwise"
    if any(e["structural_mismatch_lines"] for e in texts):
        return "discrepancy"
    if all(e["max_rel_difference"] <= 1e-9 or e["max_abs_difference"] <= 1e-12 for e in texts):
        return "roundoff"
    if all(e["max_rel_difference"] <= 1e-3 for e in texts):
        return "printed_precision"
    return "discrepancy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="single_thread")
    args = ap.parse_args()
    base = REPRO / args.tag
    report = {"tag": args.tag, "experiments": {}}
    for name, files in PAIRS.items():
        entries = []
        for f in files:
            hist, rep = ROOT / "results" / f, base / "text" / f
            if not rep.exists():
                entries.append({"file": f, "missing_reproduction": True})
            elif f.endswith(".npz"):
                entries.append(compare_npz(hist, rep))
            else:
                entries.append(compare_text(hist, rep))
        rec = base / f"{name}.json"
        meta = json.loads(rec.read_text()) if rec.exists() else {}
        report["experiments"][name] = {
            "verdict": verdict(entries), "comparisons": entries,
            "wall_seconds": meta.get("wall_seconds"), "thread_env": meta.get("thread_env"),
            "git_commit": meta.get("git_commit"),
        }
    out = ROOT / "results" / "m2" / f"baseline_comparison_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for name, e in report["experiments"].items():
        mr = [c.get("max_rel_difference") for c in e["comparisons"] if "max_rel_difference" in c]
        print(f"{name:26s} {e['verdict']:18s} max_rel={mr} wall={e['wall_seconds']}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
