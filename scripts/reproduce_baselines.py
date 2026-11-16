"""Reproduce historical baseline experiments with their ORIGINAL scripts and
settings, without overwriting any stored artifact.

Each historical script in scripts/ is loaded unchanged; only its module-level
RESULTS_DIR and FIG_DIR are redirected (text outputs to
results/m2/reproduction/text/, figures to a temporary directory that is
discarded). The experiment runners are wrapped to capture the new optimiser
diagnostics; the wrapped calls receive exactly the original arguments.

One experiment per process (use separate processes for parallelism):

    OMP_NUM_THREADS=1 /opt/anaconda3/envs/fenicsx/bin/python scripts/reproduce_baselines.py --experiment NAME

NAME is one of: verify_forward, verify_gradient, verify_bayesian_gradient,
validate_sampler, reconstruction_demo, regularization_study, noise_study,
sensor_study, posterior_demo, pcn_benchmark.

Output: results/m2/reproduction/TAG/NAME.json (machine-readable record) plus
the script's own text output under results/m2/reproduction/TAG/text/.
TAG (--tag) names the threading mode, e.g. "single_thread" (OMP_NUM_THREADS=1)
or "historical_threads" (OMP_NUM_THREADS=11, the setting that reproduces the
stored artifacts bit-for-bit; run such jobs one at a time).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "m2" / "reproduction"

SCRIPTS = {
    "verify_forward": ("verify_forward.py", []),
    "verify_gradient": ("verify_gradient.py", []),
    "verify_bayesian_gradient": ("verify_bayesian_gradient.py", []),
    "validate_sampler": ("validate_sampler.py", []),
    "reconstruction_demo": ("run_reconstruction_demo.py", ["--config", "full"]),
    "regularization_study": ("regularization_study.py", ["--config", "full"]),
    "noise_study": ("noise_study.py", ["--config", "full"]),
    "sensor_study": ("sensor_study.py", ["--config", "full"]),
    "posterior_demo": ("run_posterior_demo.py", ["--config", "full"]),
}


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _git_dirty() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def _load(script: str):
    spec = importlib.util.spec_from_file_location(f"hist_{script[:-3]}", ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jsonable(o):
    from src.optim_diagnostics import _jsonable as j
    return j(o)


def run_historical(name: str, out: pathlib.Path) -> dict:
    script, argv = SCRIPTS[name]
    mod = _load(script)
    text_dir = out / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    captured: list[dict] = []

    if hasattr(mod, "run_reconstruction"):
        original = mod.run_reconstruction

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            res = original(*args, **kwargs)
            d = res.history.diagnostics.to_dict()
            d["wall_seconds_including_data_and_setup"] = time.perf_counter() - t0
            d["historical_rel_m_error"] = res.rel_m_error
            captured.append(d)
            return res
        mod.run_reconstruction = wrapped

    if hasattr(mod, "run_bayesian_experiment"):
        original_b = mod.run_bayesian_experiment

        def wrapped_b(*args, **kwargs):
            from src.identifiability import check_posterior_variance_floor, mass_matrix, reconstruction_errors
            t0 = time.perf_counter()
            res = original_b(*args, **kwargs)
            M = mass_matrix(res.bp.dm.V_m)
            captured.append({
                "map_diagnostics": res.map_diagnostics.to_dict(),
                "map_errors": reconstruction_errors(res.m_map, res.m_true_on_inv_mesh, M),
                "posterior_mean_errors": reconstruction_errors(res.m_post_mean, res.m_true_on_inv_mesh, M),
                "variance_floor_check": check_posterior_variance_floor(res.m_post_std, res.prior, M),
                "likelihood_path": res.likelihood_path,
                "pcn_runtime_seconds": res.pcn_runtime_seconds,
                "pcn_forward_solves": res.pcn_forward_solves,
                "pcn_adjoint_solves": res.pcn_adjoint_solves,
                "pcn_n_proposals": res.pcn.n_proposals,
                "acceptance_rate": res.pcn.acceptance_rate,
                "ess_per_dim_repo_estimator": res.ess_per_dim.tolist(),
                "xi_map": res.xi_map.tolist(),
                "wall_seconds": time.perf_counter() - t0,
            })
            return res
        mod.run_bayesian_experiment = wrapped_b

    with tempfile.TemporaryDirectory() as figtmp:
        mod.RESULTS_DIR = text_dir
        mod.FIG_DIR = pathlib.Path(figtmp)
        old_argv = sys.argv
        sys.argv = [script] + argv
        t0 = time.perf_counter()
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        wall = time.perf_counter() - t0

    return {"experiment": name, "script": f"scripts/{script}", "argv": argv,
            "wall_seconds": wall, "captured_runs": captured}


def pcn_benchmark(n_proposals: int = 1000, repeats: int = 2) -> dict:
    """Original (forward+adjoint) vs forward-only likelihood path at the
    historical full Bayesian configuration (N=16, r=15, 5x5 sensors, smooth
    truth seed 1, data seed 7, beta=0.02), chains started at the MAP."""
    import yaml
    from src.bayesian_darcy import BayesianDarcyProblem
    from src.bayesian_experiment import compute_map_with_diagnostics
    from src.data_generation import generate_synthetic_data
    from src.forward import build_mesh
    from src.kl_prior import build_kl_prior
    from src.observation import build_observation_operator, grid_sensors
    from src.pcn_sampler import run_pcn
    from src.truth_fields import smooth_truth

    rc = yaml.safe_load((ROOT / "configs" / "full.yaml").read_text())["bayesian_posterior"]
    sensors = grid_sensors(rc["sensor_grid"], rc["sensor_grid"])
    data = generate_synthetic_data(smooth_truth(seed=1), sensors, rc["truth_mesh_n"],
                                   rc["sensor_width"], rc["relative_noise"], 7)
    dm = build_mesh(rc["inv_mesh_n"])
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, rc["sensor_width"])
    prior = build_kl_prior(dm, r=rc["prior_r"], gamma=rc["prior_gamma"], delta=rc["prior_delta"])

    def new_bp():
        return BayesianDarcyProblem(dm=dm, obs=H, y=data.y_noisy, sigma=data.sigma, prior=prior)

    xi_map, _ = compute_map_with_diagnostics(new_bp(), max_iter=rc["map_max_iter"])

    # 1) single-evaluation timing and value agreement at 200 prior-scale points
    rng = np.random.default_rng(123)
    pts = [xi_map + 0.3 * rng.standard_normal(prior.r) for _ in range(200)]
    bp = new_bp()
    timings = {}
    for label, fn in [("legacy", bp.neg_log_likelihood_legacy), ("forward_only", bp.neg_log_likelihood),
                      ("legacy_repeat", bp.neg_log_likelihood_legacy),
                      ("forward_only_repeat", bp.neg_log_likelihood)]:
        t0 = time.perf_counter()
        vals = np.array([fn(x) for x in pts])
        timings[label] = {"seconds_per_eval": (time.perf_counter() - t0) / len(pts), "values": vals}
    diff = np.abs(timings["legacy"]["values"] - timings["forward_only"]["values"])
    rel = diff / np.abs(timings["legacy"]["values"])

    # 2) identically seeded pCN chains, alternating order to limit drift effects
    chains = []
    for rep in range(repeats):
        order = ["legacy", "forward_only"] if rep % 2 == 0 else ["forward_only", "legacy"]
        for path in order:
            b = new_bp()
            fn = b.neg_log_likelihood_legacy if path == "legacy" else b.neg_log_likelihood
            t0 = time.perf_counter()
            res = run_pcn(fn, r=prior.r, n_samples=n_proposals, beta=rc["beta"], burn_in=0,
                          seed=1, x0=xi_map)
            chains.append({"path": path, "repeat": rep, "seconds": time.perf_counter() - t0,
                           "solves": b.solve_counts(), "acceptance_rate": res.acceptance_rate,
                           "raw_chain": res.raw_chain})
    leg = [c for c in chains if c["path"] == "legacy"]
    fast = [c for c in chains if c["path"] == "forward_only"]
    identical = all(np.array_equal(leg[0]["raw_chain"], c["raw_chain"]) for c in chains)
    max_chain_diff = max(float(np.max(np.abs(leg[0]["raw_chain"] - c["raw_chain"]))) for c in chains)
    t_leg = float(np.mean([c["seconds"] for c in leg]))
    t_fast = float(np.mean([c["seconds"] for c in fast]))
    return {
        "experiment": "pcn_benchmark",
        "configuration": {**rc, "truth": "smooth_truth(seed=1)", "data_seed": 7, "mcmc_seed": 1,
                          "x0": "MAP (map_max_iter as configured)", "n_proposals": n_proposals,
                          "burn_in": 0, "repeats_per_path": repeats},
        "single_evaluation": {
            "n_points": len(pts),
            "seconds_per_eval_legacy": [timings["legacy"]["seconds_per_eval"],
                                        timings["legacy_repeat"]["seconds_per_eval"]],
            "seconds_per_eval_forward_only": [timings["forward_only"]["seconds_per_eval"],
                                              timings["forward_only_repeat"]["seconds_per_eval"]],
            "max_abs_difference": float(diff.max()), "max_rel_difference": float(rel.max()),
            "bitwise_identical": bool(np.all(diff == 0.0)),
        },
        "chains": [{k: v for k, v in c.items() if k != "raw_chain"} for c in chains],
        "chains_bitwise_identical": bool(identical),
        "max_abs_chain_difference": max_chain_diff,
        "acceptance_rates_identical": len({c["acceptance_rate"] for c in chains}) == 1,
        "mean_seconds_per_chain": {"legacy": t_leg, "forward_only": t_fast},
        "mean_ms_per_proposal": {"legacy": 1e3 * t_leg / n_proposals, "forward_only": 1e3 * t_fast / n_proposals},
        "speedup": t_leg / t_fast,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(SCRIPTS) + ["pcn_benchmark"])
    ap.add_argument("--bench-proposals", type=int, default=1000)
    ap.add_argument("--tag", default="single_thread")
    args = ap.parse_args()
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    commit, dirty = _git_commit(), _git_dirty()
    if args.experiment == "pcn_benchmark":
        record = pcn_benchmark(n_proposals=args.bench_proposals)
    else:
        record = run_historical(args.experiment, out)
    record.update({"git_commit": commit, "uncommitted_paths_at_start": dirty, "started": started,
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "python": sys.version.split()[0], "executable_env": pathlib.Path(sys.prefix).name,
                   "tag": args.tag,
                   "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    path = out / f"{args.experiment}.json"
    path.write_text(json.dumps(_jsonable(record), indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
