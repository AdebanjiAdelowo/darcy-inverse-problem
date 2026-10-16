# Remote DeepXDE inverse-PINN execution handoff

Status: runners and smoke validation prepared. No remote PINN training results have been obtained or verified.

These are **manufactured coefficient-identification benchmarks**, not a replacement for the original FEM adjoint/Bayesian inverse problem.

## CUDA setup

Use a remote CUDA runtime (Kaggle, Colab or Lightning AI) with compatible PyTorch and DeepXDE installed. From the repository root:

```bash
export DDE_BACKEND=pytorch
python -m remote.smoke_deepxde_inverse --iterations 2
```

The smoke gate runs one short seed for each of the constant-permeability and two-parameter spatial-permeability benchmarks. It fails on subprocess errors, absent output or non-finite results. It tests execution integrity, **not convergence**. Do not run heavy experiments until it passes.

## Full experiments

```bash
python -m remote.run_deepxde_inverse --iterations 10000 --seeds 42 0 1 2 3 --output results/deepxde_inverse_remote.json
python -m remote.run_deepxde_inverse_spatial --iterations 10000 --seeds 42 0 1 2 3 --observations 128 --output results/deepxde_inverse_spatial_remote.json
```

Retain raw JSON, commit SHA, dependency versions, CUDA hardware, training logs and all failures. Review per-seed recovered permeability/parameters, held-out errors and runtime. Include sensitivity to observations and noise in subsequent controlled experiments. Compare against FEM only when the forward model, observations, parameterization and evaluation protocol match. Never claim parameter recovery or benchmark improvements before inspecting the numerical outputs.
