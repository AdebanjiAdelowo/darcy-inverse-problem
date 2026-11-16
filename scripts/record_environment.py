"""Record the computational environment as JSON: Git state, Python and key
package versions, OS, hardware, PETSc/MUMPS solver configuration and the
optimiser/sampler settings that are fixed in code.

Run with the FEniCSx interpreter, for example
    /opt/anaconda3/envs/fenicsx/bin/python scripts/record_environment.py --out results/m2/environment.json
Optionally pass --deepxde-python .venv-deepxde/bin/python to record that environment too.
Read-only: it never installs or modifies anything.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run(cmd: list[str], strip: bool = True) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60).stdout
        return out.strip() if strip else out
    except Exception:
        return None


def git_state() -> dict:
    status = _run(["git", "status", "--porcelain"], strip=False) or ""
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "modified_or_untracked": [line[3:] for line in status.splitlines() if line.strip()],
    }


def hardware() -> dict:
    def sysctl(key):
        return _run(["sysctl", "-n", key]) if sys.platform == "darwin" else None
    mem = sysctl("hw.memsize")
    return {
        "machine": platform.machine(),
        "cpu_brand": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "logical_cpus": os.cpu_count(),
        "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
        "memory_gib": round(int(mem) / 2**30, 1) if mem else None,
    }


def fem_stack() -> dict:
    import dolfinx, mpi4py, numpy, petsc4py, scipy, ufl, basix, ffcx, yaml, matplotlib
    from petsc4py import PETSc
    out = {
        "python": sys.version.split()[0],
        "executable_env": pathlib.Path(sys.prefix).name,
        "numpy": numpy.__version__, "scipy": scipy.__version__, "dolfinx": dolfinx.__version__,
        "ufl": ufl.__version__, "basix": basix.__version__, "ffcx": ffcx.__version__,
        "petsc4py": petsc4py.__version__, "petsc": ".".join(map(str, PETSc.Sys.getVersion())),
        "mpi4py": mpi4py.__version__, "pyyaml": yaml.__version__, "matplotlib": matplotlib.__version__,
    }
    try:
        from mpi4py import MPI
        out["mpi_library"] = MPI.Get_library_version().splitlines()[0]
    except Exception:
        pass
    cfg = PETSc.Sys.getVersionInfo()
    out["petsc_version_info"] = {k: v for k, v in cfg.items() if not isinstance(v, bytes)}
    try:
        import numpy as np
        out["numpy_blas"] = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    except Exception:
        pass
    return out


def deepxde_stack(python: str) -> dict | None:
    code = ("import json,sys,numpy,torch,deepxde;print(json.dumps({'python':sys.version.split()[0],"
            "'numpy':numpy.__version__,'torch':torch.__version__,'deepxde':deepxde.__version__}))")
    txt = _run([python, "-c", code])
    try:
        return json.loads(txt.splitlines()[-1]) if txt else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m2/environment.json")
    ap.add_argument("--deepxde-python", default=None)
    args = ap.parse_args()
    record = {
        "git": git_state(),
        "os": {"system": platform.system(), "release": platform.release(),
               "macos": platform.mac_ver()[0] or None, "platform": platform.platform()},
        "hardware": hardware(),
        "threads": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "fem_environment": fem_stack(),
        "numerical_configuration": {
            "forward_and_adjoint_solver": {"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps",
                                           "source": "src/forward.py (forward), src/adjoint.py (adjoint)"},
            "elements": {"pressure": "Lagrange P2", "log_permeability": "Lagrange P1",
                         "mesh": "dolfinx create_unit_square, triangles"},
            "optimiser": {"method": "scipy.optimize.minimize L-BFGS-B", "gtol": 1e-6, "ftol": 1e-12,
                          "maxcor": "SciPy default (10)", "maxls": "SciPy default (20)",
                          "source": "src/inversion.py, src/bayesian_experiment.py"},
            "sampler": {"method": "pCN", "rng": "numpy.random.default_rng(seed)",
                        "source": "src/pcn_sampler.py"},
        },
    }
    if args.deepxde_python:
        record["deepxde_environment"] = deepxde_stack(args.deepxde_python)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
