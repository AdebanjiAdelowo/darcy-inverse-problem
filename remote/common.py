"""Shared bookkeeping for the remote DeepXDE inverse-PINN runners.

Nothing here trains a model; it builds the held-out evaluation grid, records
the software/hardware environment, times runs with CUDA synchronisation and
writes results so that a failed seed is recorded instead of aborting the rest.
"""
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

EVALUATION_GRID_POINTS = 41


def evaluation_grid(n=EVALUATION_GRID_POINTS):
    """Uniform n x n grid on the closed unit square, independent of training data."""
    axis = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel()), axis=1)


def relative_l2(estimate, reference):
    estimate = np.asarray(estimate, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    return float(np.linalg.norm(estimate-reference) / np.linalg.norm(reference))


def all_finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def require_remote_cuda():
    """Refuse to run anywhere except a CUDA device with the PyTorch backend."""
    if os.environ.get("DDE_BACKEND") != "pytorch":
        raise RuntimeError("Set DDE_BACKEND=pytorch")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA GPU required; refusing local CPU training")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    import deepxde as dde
    import torch
    cuda = torch.cuda.is_available()
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "deepxde_version": dde.__version__,
        "deepxde_backend": os.environ.get("DDE_BACKEND"),
        "numpy_version": np.__version__,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "git_commit": git_commit(),
    }


def synchronize():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_seed(seed, fit):
    """Run ``fit(seed) -> dict`` with synchronised timing; never raise.

    Returns the record with ``status`` set to ``ok``, ``non_finite`` (when
    any metric listed in the record's ``finite_check`` is not finite) or
    ``error`` (exception text kept for the failure analysis).
    """
    synchronize()
    start = time.perf_counter()
    try:
        record = fit(seed)
        synchronize()
        record["wall_seconds"] = time.perf_counter() - start
        checked = record.pop("finite_check")
        record["status"] = ("ok" if all_finite([record[k] for k in checked])
                            else "non_finite")
    except Exception as exc:  # recorded, reported, and turned into a non-zero exit
        record = {"seed": seed, "status": "error",
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(limit=5),
                  "wall_seconds": time.perf_counter() - start}
    return record


def write_results(path, config, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "environment": environment(), "records": records}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def failed_seeds(records):
    return [r["seed"] for r in records if r.get("status") != "ok"]
