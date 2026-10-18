# Remote DeepXDE inverse-PINN execution handoff

Status: runners, smoke gate and Kaggle launcher prepared. Results and run status are tracked in
`docs/DEEPXDE_REMOTE_RESULTS.md`; nothing here claims a result.

These are **manufactured coefficient-identification benchmarks**, not a replacement for the original FEM adjoint/Bayesian inverse problem.

## Environment

The runners need only NumPy, PyTorch with CUDA and `deepxde==1.15.0`. They do not need FEniCSx: the
`src` package resolves its FEM exports lazily, and a unit test checks that the runners import without
`dolfinx`, `ufl` or `petsc4py`. All runners refuse to start unless `DDE_BACKEND=pytorch` is set and CUDA
is available.

## Any CUDA runtime (Kaggle, Colab or Lightning AI)

From the repository root:

```bash
pip install deepxde==1.15.0
export DDE_BACKEND=pytorch
python -m remote.smoke_deepxde_inverse --iterations 2
```

The smoke gate runs one short seed for each of the constant-permeability and two-parameter spatial-permeability benchmarks. It fails on subprocess errors, absent output, a non-`ok` status or non-finite results. It tests execution integrity, **not convergence**. Do not run heavy experiments until it passes.

## Full experiments

```bash
python -m remote.run_deepxde_inverse --iterations 10000 --seeds 42 0 1 2 3 --output results/deepxde_inverse_remote.json
python -m remote.run_deepxde_inverse_spatial --iterations 10000 --seeds 42 0 1 2 3 --observations 128 --output results/deepxde_inverse_spatial_remote.json
```

Each output is `{"config", "environment", "records"}`. `environment` holds the Python, PyTorch, CUDA and
DeepXDE versions, the GPU name and the commit. Each record holds the estimates, errors, held-out pressure
(and permeability) relative L2 on a 41 x 41 grid, the synchronised wall time, the final loss components,
the parameter trajectory and the loss history. A failing seed is recorded with `status` set to `error` or
`non_finite`, the other seeds still run, and the process exits 1 at the end.

Further options for the sensitivity study: `--observations`, `--noise-std`, `--observation-seed`, and
`--k-initial` or `--alpha-initial` / `--beta-initial`.

## Kaggle launcher

`remote/kaggle_deepxde_inverse.py` is a thin script kernel that calls the commands above. It clones the
public GitHub repository and checks `HEAD == REF`. If a Kaggle dataset containing
`darcy-inverse-problem.bundle` (a `git bundle` of `main`) is attached, it clones from that instead; kernel
v2 used this route while the repository was private. No token is ever placed in a kernel. Stage the launcher outside the repository,
set `REF` to the full 40-character SHA (the bundle must contain it), and choose `STAGE` (`smoke`,
`baseline`, `convergence`, or `sensitivity`). Each stage runs the smoke gate first.

```bash
mkdir -p ~/kaggle-staging/darcy-inverse-problem/kernel && cd ~/kaggle-staging/darcy-inverse-problem/kernel
# optional, only for the bundle route:
#   git -C <repo> bundle create <dataset dir>/darcy-inverse-problem.bundle main
#   kaggle datasets version -p <dataset dir> -m "<sha>"
cp <repo>/remote/kaggle_deepxde_inverse.py .
sed -i '' "s/<40-character commit SHA>/$(git -C <repo> rev-parse HEAD)/; s/^STAGE = .*/STAGE = \"baseline\"/" kaggle_deepxde_inverse.py
# kernel-metadata.json: script kernel, is_private true, enable_gpu true, enable_internet true,
# machine_shape NvidiaTeslaT4, code_file kaggle_deepxde_inverse.py,
# dataset_sources [] (or ["<user>/darcy-inverse-problem-src"] for the bundle route)
kaggle kernels push -p .
kaggle kernels status <user>/<kernel-slug>
kaggle kernels output <user>/<kernel-slug> -p <new empty directory>
```

On this development Mac the Kaggle CLI can stall on uploads because IPv6 is advertised but unusable. If it
stalls, run it through a wrapper that restricts name resolution to IPv4.

The kernel writes `outputs/status.json` (return code and duration of every step), `outputs/logs/*.log`,
`outputs/pip_freeze.txt`, and `outputs/results/*.json`. Gates skip later steps instead of raising.

## Rules for results

Retain the raw JSON, commit SHA, dependency versions, CUDA hardware, training logs and every failure.
Review per-seed parameters, held-out errors and runtime against the pre-registered criteria in
`docs/DEEPXDE_REMOTE_RESULTS.md`. Compare against FEM only when the forward model, observations,
parameterisation and evaluation protocol match. Never claim parameter recovery or benchmark improvements
before inspecting the numerical outputs.
