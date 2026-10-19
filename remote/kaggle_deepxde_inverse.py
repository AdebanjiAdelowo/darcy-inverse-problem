"""Thin Kaggle launcher for the DeepXDE inverse-PINN smoke gate and experiments.

Runs as a Kaggle *script* kernel (GPU + Internet enabled). It clones this
repository at a pinned commit from a ``git bundle`` attached as a private
Kaggle dataset (the GitHub repository is private, and no token is ever put in
a kernel), falling back to the GitHub URL only when no bundle is attached, installs a pinned DeepXDE, records the
environment, runs the smoke gate and, only if it passes, the requested stage.
All science lives in the repository's ``remote/`` runners; this file only
orchestrates. Every outcome is written to /kaggle/working/outputs; gates skip
later steps instead of raising, so the kernel always finishes cleanly.

Before pushing, replace REF with the full 40-character commit SHA and choose
STAGE. The committed copy deliberately keeps the placeholder.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REF = "<40-character commit SHA>"
STAGE = "baseline"  # "smoke" | "baseline" | "convergence" | "lbfgs" | "sensitivity"
REPO = "https://github.com/AdebanjiAdelowo/darcy-inverse-problem.git"
BUNDLE_NAME = "darcy-inverse-problem.bundle"
DEEPXDE = "deepxde==1.15.0"

WORK = Path("/kaggle/working")
CLONE = WORK / "repo"
OUT = WORK / "outputs"
SEEDS = ["42", "0", "1", "2", "3"]

BASELINE = {
    "constant_baseline": ["remote.run_deepxde_inverse", "--iterations", "10000",
                          "--seeds", *SEEDS],
    "spatial_baseline": ["remote.run_deepxde_inverse_spatial", "--iterations", "10000",
                         "--seeds", *SEEDS, "--observations", "128"],
}

# Post-hoc convergence diagnostic (docs/DEEPXDE_REMOTE_RESULTS.md, Deviations 1):
# the baseline with 50 000 iterations; nothing else changes.
CONVERGENCE = {
    name.replace("_baseline", "_convergence50k"):
        [module, *[("50000" if prev == "--iterations" else a)
                   for prev, a in zip([None, *args], args)], "--trace-every", "500"]
    for name, (module, *args) in BASELINE.items()
}

# Explicit "adam+lbfgs" configuration (docs/DEEPXDE_REMOTE_RESULTS.md, L-BFGS
# protocol): the 50 000-iteration Adam diagnostic rerun from scratch (no network
# checkpoint exists from kernel v3), then DeepXDE's default L-BFGS, maxiter 15000.
LBFGS = {
    "spatial_adam50k_lbfgs15k": [
        *CONVERGENCE["spatial_convergence50k"], "--lbfgs-maxiter", "15000",
        "--lbfgs-record-every", "10", "--save-state", str(OUT / "states")],
}

# Pre-registered one-factor-at-a-time sensitivity study around the spatial
# baseline (docs/DEEPXDE_REMOTE_RESULTS.md). Runs only with STAGE="sensitivity".
SENSITIVITY = {
    **{f"spatial_obs{n}": ["remote.run_deepxde_inverse_spatial", "--iterations", "10000",
                           "--seeds", *SEEDS, "--observations", str(n)]
       for n in (16, 32, 64, 256)},
    **{f"spatial_noise{tag}": ["remote.run_deepxde_inverse_spatial", "--iterations", "10000",
                               "--seeds", *SEEDS, "--observations", "128",
                               "--noise-std", sigma]
       for tag, sigma in (("0p01", "0.01"), ("0p05", "0.05"))},
    "spatial_fixed_sensors": ["remote.run_deepxde_inverse_spatial", "--iterations", "10000",
                              "--seeds", *SEEDS, "--observations", "128",
                              "--observation-seed", "42"],
    "spatial_init_far": ["remote.run_deepxde_inverse_spatial", "--iterations", "10000",
                         "--seeds", *SEEDS, "--observations", "128",
                         "--alpha-initial", "-1.0", "--beta-initial", "1.0"],
    "constant_noise0p01": ["remote.run_deepxde_inverse", "--iterations", "10000",
                           "--seeds", *SEEDS, "--noise-std", "0.01"],
    "constant_init_high": ["remote.run_deepxde_inverse", "--iterations", "10000",
                           "--seeds", *SEEDS, "--k-initial", "5.0"],
}

status = {"ref": REF, "stage": STAGE, "steps": {}}


def save_status():
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")


def step(name, cmd, cwd=None, env=None):
    """Run ``cmd``, keep full stdout/stderr, record return code and duration."""
    start = time.time()
    with open(OUT / "logs" / f"{name}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            text=True).returncode
    status["steps"][name] = {"returncode": rc, "seconds": round(time.time() - start, 1),
                             "command": " ".join(map(str, cmd))}
    save_status()
    print(f"{name}: rc={rc} ({status['steps'][name]['seconds']} s)", flush=True)
    return rc == 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    if not re.fullmatch(r"[0-9a-f]{40}", REF):
        status["blocked"] = "REF is not a 40-character commit SHA"
        save_status()
        print(status["blocked"])
        return
    if STAGE not in ("smoke", "baseline", "convergence", "lbfgs", "sensitivity"):
        status["blocked"] = f"unknown STAGE {STAGE!r}"
        save_status()
        return

    shutil.rmtree(CLONE, ignore_errors=True)
    bundles = sorted(Path("/kaggle/input").rglob(BUNDLE_NAME))
    source = str(bundles[0]) if bundles else REPO
    status["source"] = source
    if not (step("clone", ["git", "clone", "--quiet", source, str(CLONE)])
            and step("checkout", ["git", "-C", str(CLONE), "checkout", "--quiet", REF])):
        return
    head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    status["head"] = head
    if head != REF:
        status["blocked"] = f"HEAD {head} != REF"
        save_status()
        return
    if not step("install_deepxde", [sys.executable, "-m", "pip", "install", "--quiet",
                                    DEEPXDE]):
        return
    step("nvidia_smi", ["nvidia-smi"])
    step("pip_freeze", [sys.executable, "-m", "pip", "freeze"])
    shutil.copy(OUT / "logs" / "pip_freeze.log", OUT / "pip_freeze.txt")

    env = {**os.environ, "DDE_BACKEND": "pytorch", "PYTHONUNBUFFERED": "1"}
    results = OUT / "results"
    results.mkdir(exist_ok=True)
    smoke_ok = step("smoke", [sys.executable, "-m", "remote.smoke_deepxde_inverse",
                              "--iterations", "2", "--output-dir", str(results / "smoke"),
                              *(["--include-lbfgs"] if STAGE == "lbfgs" else [])],
                    cwd=CLONE, env=env)
    status["smoke_passed"] = smoke_ok
    save_status()
    if not smoke_ok or STAGE == "smoke":
        return

    plan = {"baseline": BASELINE, "convergence": CONVERGENCE, "lbfgs": LBFGS,
            "sensitivity": SENSITIVITY}[STAGE]
    for name, (module, *args) in plan.items():
        step(name, [sys.executable, "-m", module, *args, "--output",
                    str(results / f"{name}.json")], cwd=CLONE, env=env)


if __name__ == "__main__":
    try:
        main()
    finally:
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if OUT.exists():
            save_status()
        shutil.rmtree(CLONE, ignore_errors=True)
