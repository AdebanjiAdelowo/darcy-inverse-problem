# DeepXDE inverse-PINN benchmarks: remote experiments

Status: **PROTOCOL FIXED, NO GPU RESULTS YET.** This file is the pre-registration. Results will be
added below the protocol in a later commit; the protocol and criteria are not edited after results are
seen. Any deviation will be recorded as a deviation, with its reason.

## Scope

Two manufactured coefficient-identification benchmarks on the unit square, zero Dirichlet boundary,
manufactured pressure `p = sin(pi x) sin(pi y)`, noise-free point observations unless stated:

| Benchmark | Unknowns | Truth | Initial guess |
|---|---|---|---|
| `constant_k` (`src/deepxde_inverse_constant_k.py`) | scalar `k = exp(log_k)` | `k = 1.7` | `k = 0.8` |
| `spatial_k` (`src/deepxde_inverse_spatial_k.py`) | `alpha, beta` in `k = exp(alpha x + beta y)` | `alpha = 0.5, beta = -0.3` | `alpha = beta = 0` |

Both use a 2-64-64-64-1 tanh network (Glorot normal), Adam at learning rate 1e-3, 512 Hammersley domain
points, 64 boundary points, unit loss weights, and 10 000 iterations. The constant benchmark uses 100
observation points and the spatial benchmark 128, drawn uniformly in `[0.05, 0.95]^2`.

These are low-dimensional parametric problems. They are **not** comparable to the FEM-adjoint MAP or
pCN Bayesian study in this repository, which reconstructs a high-dimensional log-permeability field from
noisy observations on a finite-element mesh. No error or runtime from this file is to be set against those
results.

## Controls and what a seed changes

`dde.config.set_random_seed(seed)` seeds Python, NumPy and PyTorch, so the seed sets the network
initialisation. By default, the observation locations are also drawn with `numpy.random.default_rng(seed)`.
The baseline therefore varies **initialisation and sensor layout together**. The `--observation-seed`
option fixes the layout across seeds to separate the two effects. Residual collocation points (Hammersley)
are identical for every seed. GPU kernels are not forced to be deterministic, so the same seed may differ
slightly between runs or GPUs.

## Evaluation

- Parameter recovery. Constant: `|k_est - k| / k`. Spatial: `||(alpha, beta)_est - (alpha, beta)|| /
  ||(alpha, beta)||` (Euclidean), plus absolute errors of each parameter.
- Pressure: relative L2 error of the network prediction against the exact pressure on a 41 x 41 uniform
  grid on the closed unit square. The grid is fixed and independent of the observation points; a unit test
  checks that no grid point coincides with an observation.
- Permeability field (spatial only): relative L2 error of `exp(alpha x + beta y)` on the same grid.
- Cost: wall-clock seconds per seed with CUDA synchronisation, including model construction. The first
  seed in a process also pays CUDA initialisation.
- Convergence: the parameter trajectory every 100 iterations and the DeepXDE loss history (10 points).
- Failure handling: a seed that raises or gives a non-finite metric is recorded with `status` set to
  `error` or `non_finite`. The remaining seeds still run, and the runner exits non-zero at the end.

## Pre-registered criteria (set before any GPU run)

These thresholds are descriptive labels chosen for this protocol, not universal standards.

1. **Execution integrity.** The 2-iteration smoke gate passes for both benchmarks on CUDA. This says
   nothing about convergence.
2. **Stability.** All 5 baseline seeds finish with `status: ok`.
3. **Parameter recovery** (noise-free baseline, per benchmark):
   - *recovered*: relative parameter error below 1e-2 for every seed;
   - *partially recovered*: median below 1e-2 but at least one seed above;
   - *not recovered*: median at or above 1e-2.
4. **Pressure reconstruction**: reported as mean, standard deviation, median, minimum and maximum over
   seeds. No threshold.
5. **Sensitivity study** (run only after the baseline has been inspected): one factor at a time around the
   spatial baseline, 5 seeds each, 10 000 iterations. Sensor count 16, 32, 64, 256 (baseline 128).
   Observation noise standard deviation 0.01 and 0.05 (absolute; pressure peak is 1). Fixed sensor layout
   (`--observation-seed 42`). Far initialisation (`alpha = -1, beta = 1`). Constant benchmark: noise 0.01
   and initial `k = 5`. Expectation stated in advance: error grows as sensors decrease and noise increases;
   no prediction is made for the initialisation runs.

## Execution

Launcher: `remote/kaggle_deepxde_inverse.py` (Kaggle script kernel, GPU and Internet on). It clones
this repository at a pinned commit, installs `deepxde==1.15.0` on top of Kaggle's PyTorch, records
`nvidia-smi` and `pip freeze`, runs the smoke gate, and runs the chosen stage only if the smoke passes.
Exact steps are in `docs/CLAUDE_CODE_DEEPXDE_REMOTE_HANDOFF.md`.

## Results

None yet.
