"""Thin Kaggle launcher for the DeepXDE inverse-PINN smoke gate and experiments.

Runs as a Kaggle *script* kernel (GPU + Internet enabled). It clones this
repository at a pinned commit, installs a pinned DeepXDE, records the
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
STAGE = "baseline"  # "smoke" | "baseline" | "sensitivity"
REPO = "https://github.com/AdebanjiAdelowo/darcy-inverse-problem.git"
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
    if STAGE not in ("smoke", "baseline", "sensitivity"):
        status["blocked"] = f"unknown STAGE {STAGE!r}"
        save_status()
        return

    shutil.rmtree(CLONE, ignore_errors=True)
    if not (step("clone", ["git", "clone", "--quiet", REPO, str(CLONE)])
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
                              "--iterations", "2", "--output-dir",
                              str(results / "smoke")], cwd=CLONE, env=env)
    status["smoke_passed"] = smoke_ok
    save_status()
    if not smoke_ok or STAGE == "smoke":
        return

    plan = BASELINE if STAGE == "baseline" else SENSITIVITY
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
