# Reproducibility

This page lists the environments, settings and commands needed to reproduce the stored results.

## Environments

| Environment | Purpose | Specification files |
|---|---|---|
| conda `fenicsx` (Python 3.12.12, dolfinx 0.10.0, PETSc 3.24.3, SciPy 1.18.1, NumPy 2.5.3, OpenBLAS 0.3.34, MPICH 4.3.2) | FEM forward/adjoint, deterministic and Bayesian inversion, all non-PINN tests | `environment.yml` (unpinned, portable), `environment/fenicsx.pinned.yml` (all packages version-pinned, no build strings), `environment/fenicsx-osx-arm64.explicit.txt` (exact osx-arm64 builds) |
| uv virtual environment `.venv-deepxde` (Python 3.12.2, torch 2.14.1, DeepXDE 1.15.0, NumPy 2.5.4) | optional PINN benchmarks and their CPU tests | `environment/deepxde-requirements.txt` (pip freeze) |

Create the exact FEM environment on Apple silicon:

```bash
conda create --name fenicsx --file environment/fenicsx-osx-arm64.explicit.txt
```

On other platforms use `environment/fenicsx.pinned.yml` (same versions, platform builds resolved by conda).
The PINN environment can be recreated with `uv venv .venv-deepxde --python 3.12 && uv pip install --python .venv-deepxde/bin/python -r environment/deepxde-requirements.txt`.
A `uv.lock` was not generated because resolving it requires network access.

Run the tests by calling each environment's interpreter directly (never `uv run --active`):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /path/to/envs/fenicsx/bin/python -m pytest tests
DDE_BACKEND=pytorch .venv-deepxde/bin/python -m pytest tests/test_deepxde_*.py
```

`scripts/record_environment.py` writes the Git commit, package versions, OS, hardware, thread settings and
the fixed numerical configuration (MUMPS direct LU for forward and adjoint solves; P2 pressure and P1
log-permeability; L-BFGS-B with gtol 1e-6 and ftol 1e-12; pCN with `numpy.random.default_rng`) to JSON.
The record for the Milestone 2 runs is `results/m2/environment.json`.

## Thread settings and bit-for-bit reproduction

The results stored in `results/*.txt` and `results/*.npz` were produced with **11 OpenMP threads**
(the default on the 11-core M3 Pro). The OpenMP thread count changes the last bits of the PDE solves,
most likely through the floating-point summation order of the OpenMP-threaded sparse direct solver (MUMPS);
the mechanism was inferred, not traced:

| `OMP_NUM_THREADS` (with `OPENBLAS_NUM_THREADS=1`) | `results/gradient_verification.txt` |
|---|---|
| unset (default) or 11 | identical, byte for byte |
| 1, 2, 4 or 8 | Taylor remainders differ in the 5th to 7th significant digit (they are differences of objective values of size about 2e3, so this is last-bit roundoff amplified by cancellation); observed orders identical to printed precision |

`OPENBLAS_NUM_THREADS` alone does not change the results.

**Determinism.** With `OMP_NUM_THREADS=1` every computation tested is bitwise identical from run to run,
including full 200-iteration reconstructions and the 120-iteration MAP. With 11 threads, short computations
(the verification scripts, the noise and sensor studies) matched the stored files bit for bit, but two
identical full-configuration runs of the same code differed by up to 2.6e-5 in the reconstructed field and
in the MAP. Multi-threaded results are therefore not reproducible bit for bit in general, and the stored
optimiser-heavy artifacts (headline reconstructions, regularisation study, posterior demo) cannot be
reproduced bit for bit under any setting. Their reproduction to printed precision is documented in
`docs/audit/MILESTONE_2_REPORT.md`.

Recommended settings:

- **All new experiments:** `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, with independent jobs in at most four
  separate processes. This is deterministic and keeps the machine available for other work.
- **Bit-for-bit checks of short stored artifacts:** `OMP_NUM_THREADS=11 OPENBLAS_NUM_THREADS=1`, one process
  at a time.

The unconverged MAP estimates are sensitive to last-bit roundoff (about 1e-4 in the KL coefficients),
which shifts the start of the pCN chain. The reproduced 10,000-step chains nevertheless made exactly the
same 8,000 retained accept/reject decisions as the stored chains, with sample values within 3e-4.

## Reproducing the historical baselines

`scripts/reproduce_baselines.py` runs the original scripts unchanged with their original configurations,
redirecting their outputs to `results/m2/reproduction/TAG/` so that no stored artifact is overwritten
(figures go to a temporary directory). Each run also records the optimiser diagnostics described below.

```bash
PY=/path/to/envs/fenicsx/bin/python
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
# deterministic single-thread mode, at most 4 processes
printf "%s\n" posterior_demo sensor_study regularization_study pcn_benchmark noise_study \
  reconstruction_demo verify_forward validate_sampler verify_bayesian_gradient verify_gradient |
  OMP_NUM_THREADS=1 xargs -P 4 -I{} $PY scripts/reproduce_baselines.py --experiment {} --tag single_thread
$PY scripts/compare_baselines.py --tag single_thread
# historical threading mode, one process at a time
for e in verify_forward verify_gradient verify_bayesian_gradient validate_sampler reconstruction_demo \
         regularization_study noise_study sensor_study posterior_demo; do
  OMP_NUM_THREADS=11 $PY scripts/reproduce_baselines.py --experiment $e --tag historical_threads
done
$PY scripts/compare_baselines.py --tag historical_threads
```

## Optimiser diagnostics

Every L-BFGS-B run made through `src/inversion.py::run_lbfgs` or
`src/bayesian_experiment.py::compute_map_with_diagnostics` records an `OptimizerDiagnostics` object
(`src/optim_diagnostics.py`), serialisable with `.to_dict()`. It holds the initial and final objective and their
misfit and regularisation (or prior) parts, final gradient 2-norm and inf-norm, iteration, function and
gradient evaluation counts, forward and adjoint PDE solve counts, SciPy's status and message, a termination
reason, the options, the runtime, and per-evaluation traces. A run is classified `converged` only when SciPy
stopped on its gradient or relative-function-reduction test; a run that stops at its iteration cap is never
classified as converged. For deterministic reconstructions, `extra` also contains absolute and mean-free
error measures (`src/identifiability.py`) and the pressure-prediction error at the sensors.

## Bayesian likelihood path

Since Milestone 2, pCN evaluates the likelihood with one forward solve per proposal
(`BayesianDarcyProblem.neg_log_likelihood`). The historical path, which also solved the adjoint problem and
assembled an unused gradient, remains available as `neg_log_likelihood_legacy` and through
`run_bayesian_experiment(..., likelihood_path="legacy")`. Both give bitwise identical likelihood values and,
for the same seed, bitwise identical chains (`tests/test_likelihood_path.py`).
