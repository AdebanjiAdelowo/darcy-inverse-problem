# DeepXDE inverse-PINN benchmarks: remote experiments

Status: **smoke gate PASSED on CUDA (constant, spatial, and spatial with L-BFGS). Constant k: RECOVERED
with 50 000 Adam iterations. Spatial k: NOT recovered with Adam alone (10 000 or 50 000 iterations);
RECOVERED with the explicit `adam+lbfgs` configuration (50 000 Adam iterations, then L-BFGS), 5/5 seeds
below the pre-registered tolerance. Sensitivity study (Deviation 4) COMPLETED: recovered for every
noise-free row (16 to 256 sensors, initialisation only, layout only, far initial guess) and for both
constant-k rows; NOT recovered with observation noise 0.01 or 0.05.**

The protocol and criteria below were committed (656c591) before any GPU run and are unchanged. Deviations
are listed under *Deviations*.

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

### Run log

| Kernel | Commit | Outcome |
|---|---|---|
| v1 | 656c591 | Stopped at `git clone`: the GitHub repository was private at the time and anonymous clone failed. Nothing ran. Fixed in 5b63bb6 (clone from a git bundle). |
| v2 | 5b63bb6 | All steps exit 0: smoke gate, constant baseline, spatial baseline. Evidence in `results/deepxde_remote/kaggle_v2_5b63bb6/`. |
| v3 | 03b7744 | All steps exit 0: smoke gate, 50 000-iteration diagnostic for both benchmarks (Deviation 1). Cloned from the now-public GitHub repository. Evidence in `results/deepxde_remote/kaggle_v3_03b7744/`. |
| v4 | 5d710f7 | All steps exit 0: smoke gate (3 cases), `adam+lbfgs` spatial run (Deviation 3). Evidence in `results/deepxde_remote/kaggle_v4_5d710f7/`; the 10 network/parameter state files are not committed, their SHA-256 hashes are in `STATES_NOT_COMMITTED.sha256`. |
| v5 | d3ee5b2 | Stage `sensitivity_a`: smoke gate (3 cases) and 6 spatial rows, all exit 0. Evidence in `results/deepxde_remote/kaggle_v5_sensitivity_a_d3ee5b2/`. |
| b v1 | d3ee5b2 | Second private kernel, run in parallel. Stage `sensitivity_b`: smoke gate and 5 rows, all exit 0. Evidence in `results/deepxde_remote/kaggle_b1_sensitivity_b_d3ee5b2/`. |

Environment (v2, from the result JSON and `nvidia-smi`): Tesla T4 (2 visible, `cuda:0` used),
Python 3.13.15, PyTorch 2.11.0+cu128 (CUDA 12.8), DeepXDE 1.15.0, NumPy 2.1.3, Kaggle Linux image.
Every file is listed with its SHA-256 in `MANIFEST.sha256`; `summary.json` is produced from the raw JSON by
`scripts/summarize_deepxde_inverse.py`.

### Smoke gate (2 iterations, seed 42)

`PASS constant inverse smoke` and `PASS spatial inverse smoke`: both runners completed on CUDA, wrote their
output, and every metric was finite with `status: ok`. After 2 iterations the estimates are essentially the
initial guesses (k = 0.799, alpha = -0.002, beta = 0.002). This establishes execution integrity only.

### Baseline, 10 000 iterations, 5 seeds (42, 0, 1, 2, 3)

Criterion 2 (stability) is **met** for both benchmarks: 5/5 seeds `ok`, no non-finite values.

Criterion 3 (parameter recovery) is **not met** for either benchmark: *not recovered* (median relative
parameter error 0.040 for constant k and 0.81 for spatial k, against the 1e-2 threshold).

**Constant k (truth 1.7, start 0.8, 100 observations).**

| seed | k estimate | relative k error | pressure rel. L2 | change in k over the last 1000 iterations | wall s |
|---|---|---|---|---|---|
| 42 | 1.6323 | 0.0398 | 0.0429 | +0.0249 | 87.9 |
| 0 | 1.6418 | 0.0343 | 0.0242 | +0.0244 | 83.7 |
| 1 | 1.5960 | 0.0612 | 0.0455 | +0.0267 | 83.8 |
| 2 | 1.6434 | 0.0333 | 0.0276 | +0.0205 | 83.8 |
| 3 | 1.6181 | 0.0482 | 0.0329 | +0.0247 | 83.3 |

Over seeds: k = 1.626 ± 0.020 (mean ± sample std; median 1.632, range 1.596 to 1.643); relative error
0.043 ± 0.012; pressure relative L2 0.035 ± 0.009 (range 0.024 to 0.046).

**Spatial k (truth alpha = 0.5, beta = -0.3, start 0, 0, 128 observations).**

| seed | alpha | beta | parameter rel. error | pressure rel. L2 | k-field rel. L2 | change over last 1000 it (alpha, beta) | peak beta (iteration) | wall s |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.1664 | 0.1135 | 0.911 | 0.0592 | 0.1575 | +0.0046, -0.0148 | 0.173 (4800) | 105.1 |
| 0 | 0.1814 | 0.0987 | 0.875 | 0.0549 | 0.1514 | +0.0041, -0.0140 | 0.162 (4200) | 102.5 |
| 1 | 0.1990 | 0.0584 | 0.803 | 0.0484 | 0.1375 | +0.0067, -0.0099 | 0.104 (4600) | 102.7 |
| 2 | 0.1907 | 0.0583 | 0.812 | 0.0491 | 0.1387 | +0.0046, -0.0104 | 0.109 (4500) | 102.8 |
| 3 | 0.2141 | 0.0323 | 0.752 | 0.0448 | 0.1284 | +0.0073, -0.0126 | 0.105 (3800) | 102.1 |

Over seeds: alpha = 0.190 ± 0.018, beta = 0.072 ± 0.033 (wrong sign at every seed); parameter relative
error 0.83 ± 0.06; pressure relative L2 0.051 ± 0.006; permeability-field relative L2 0.143 ± 0.012.

**Interpretation, stated narrowly.** In every seed of both benchmarks the parameters were still moving
monotonically toward the truth at iteration 10 000 (last column but one), while the total training loss had
already fallen from about 1e2 to about 1e-3. For the spatial problem beta first moved away from the truth,
peaked between iterations 3800 and 4800, and was falling toward -0.3 at the end. The result therefore shows
that **10 000 Adam iterations at learning rate 1e-3 are not enough to converge this formulation**. It does
not show a bias at convergence, and it does not show that the parameters would be recovered with more
iterations; the convergence diagnostic below tests that. Small loss values coexisting with large parameter
error mean the loss is only weakly sensitive to the parameters near the end of training, so loss magnitude
is not a usable stopping signal here.

Seed-to-seed spread is small compared with the distance from the truth (constant k: std 0.020 against a
mean error of 0.074; spatial alpha: std 0.018 against a mean error of 0.31). Under the baseline, seed changes
initialisation and sensor layout together; the effect of each separately has not been measured.

**Cost.** Wall time per seed on one T4, CUDA-synchronised, including model construction and the 41 x 41
evaluation: constant k 84.5 ± 1.9 s (about 8.4 ms per iteration), spatial k 103.1 ± 1.2 s (about 10.3 ms
per iteration). The first seed of the constant run includes CUDA initialisation (87.9 s against 83.3 to 83.8
s). These are single-session T4 timings and are not comparable to the FEM/Bayesian runtimes.

**Warnings in the logs.** `512 points required, but 529 points sampled` (DeepXDE rounds its internal test set
up to a 23 x 23 grid; that set is not the evaluation grid) and one cuBLAS context-initialisation notice per
process. Neither affects the results.

### Convergence diagnostic, 50 000 iterations (post hoc, Deviation 1)

Same hardware type (Tesla T4), `pip freeze` byte-identical to v2, same seeds and settings; only the
iteration count and trace period differ.

**Reproducibility across sessions.** At all 21 checkpoints the two runs share (every 500 iterations up to
10 000), the recorded parameter values of v3 are identical to those of the v2 baseline for every seed of both
benchmarks (maximum absolute difference 0). The 10 000-iteration baseline is therefore exactly the start of
these runs. Only the recorded parameter values were compared, not network weights or losses.

**Constant k: recovered** under the pre-registered criterion (every seed below 1e-2).

| seed | k at 50 000 | relative k error | pressure rel. L2 | change in k over the last 5000 iterations | wall s |
|---|---|---|---|---|---|
| 42 | 1.7044 | 0.0026 | 0.0129 | +0.0005 | 413 |
| 0 | 1.7006 | 0.0004 | 0.0073 | +0.0001 | 406 |
| 1 | 1.6951 | 0.0029 | 0.0084 | +0.0012 | 406 |
| 2 | 1.6997 | 0.0001 | 0.0106 | +0.0009 | 405 |
| 3 | 1.7009 | 0.0005 | 0.0183 | +0.0008 | 406 |

Over seeds: k = 1.700 ± 0.003 (median 1.701, range 1.695 to 1.704); relative error 1.3e-3 ± 1.3e-3
(median 5.1e-4, maximum 2.9e-3); pressure relative L2 0.0115 ± 0.0044. The estimate is between 1.676 and
1.700 at iteration 20 000 and changes by at most 0.0012 over the last 5000 iterations.

**Spatial k: not recovered**, but every seed was still moving toward the truth at the last iteration.

| seed | alpha at 50 000 | beta at 50 000 | parameter rel. error | pressure rel. L2 | k-field rel. L2 | change over last 5000 it (alpha, beta) | wall s |
|---|---|---|---|---|---|---|---|
| 42 | 0.3642 | -0.1567 | 0.339 | 0.0204 | 0.0586 | +0.0172, -0.0186 | 498 |
| 0 | 0.3839 | -0.1721 | 0.296 | 0.0176 | 0.0509 | +0.0175, -0.0188 | 491 |
| 1 | 0.3776 | -0.1685 | 0.308 | 0.0212 | 0.0532 | +0.0144, -0.0163 | 490 |
| 2 | 0.3684 | -0.1590 | 0.331 | 0.0211 | 0.0570 | +0.0148, -0.0166 | 491 |
| 3 | 0.4094 | -0.2071 | 0.223 | 0.0130 | 0.0390 | +0.0133, -0.0144 | 490 |

Over seeds: alpha = 0.381 ± 0.018, beta = -0.173 ± 0.020 (sign becomes correct between iterations 20 000 and
30 000 in every seed); parameter relative error 0.30 ± 0.05 (median 0.31); pressure relative L2 0.019 ± 0.004;
permeability-field relative L2 0.052 ± 0.008.

**Interpretation, stated narrowly.** Under this optimiser and learning rate, the one-parameter problem
converges to the true value within the pre-registered tolerance in 50 000 iterations, and the 10 000-iteration
failure was a budget effect. The two-parameter problem converges far more slowly: after 50 000 iterations the
error in the parameter vector has fallen to about 30 percent of its initial value and is still moving at about 0.003 to 0.004 per
1000 iterations in each parameter. Whether it would reach the tolerance with more Adam iterations has not been
tested, and no extrapolation is claimed. Pressure is reconstructed to about 2 percent relative L2 while the
parameters are still 30 percent off, which again shows that pressure accuracy is not evidence of parameter
recovery in this benchmark.

**Cost.** Constant k 406.9 ± 3.3 s per seed (about 8.1 ms per iteration), spatial k 492.0 ± 3.3 s (about
9.8 ms per iteration), one T4, CUDA-synchronised. Kernel v3 used about 75 minutes of GPU time in total.

### Adam + L-BFGS refinement (Deviation 3)

**Smoke.** `PASS spatial_lbfgs inverse smoke` (2 Adam iterations, then 20 L-BFGS iterations), together with
the two Adam-only cases.

**Starting state.** The Adam stage was rerun from scratch because no network checkpoint existed. Its
recorded parameter values are identical to kernel v3 at all 101 checkpoints (every 500 iterations, maximum
absolute difference 0), and its final alpha, beta, pressure and permeability errors are identical to v3 for
every seed. Refinement therefore starts from the documented 50 000-iteration Adam result.

**Spatial k with `adam+lbfgs`: recovered** under the pre-registered criterion: the relative parameter error
is below 1e-2 for every seed (maximum 5.1e-3).

| seed | stage | alpha | beta | parameter rel. error | pressure rel. L2 | k-field rel. L2 | total train loss |
|---|---|---|---|---|---|---|---|
| 42 | Adam 50k | 0.3642 | -0.1567 | 0.339 | 0.0204 | 0.0586 | 2.33e-04 |
| | + L-BFGS | 0.4998 | -0.2990 | 1.75e-03 | 7.82e-04 | 4.62e-04 | 6.77e-06 |
| 0 | Adam 50k | 0.3839 | -0.1721 | 0.296 | 0.0176 | 0.0509 | 1.99e-04 |
| | + L-BFGS | 0.5012 | -0.3011 | 2.79e-03 | 1.04e-03 | 5.35e-04 | 3.37e-06 |
| 1 | Adam 50k | 0.3776 | -0.1685 | 0.308 | 0.0212 | 0.0532 | 3.59e-03 |
| | + L-BFGS | 0.5020 | -0.3022 | 5.09e-03 | 9.86e-04 | 8.92e-04 | 3.94e-06 |
| 2 | Adam 50k | 0.3684 | -0.1590 | 0.331 | 0.0211 | 0.0570 | 1.04e-03 |
| | + L-BFGS | 0.5002 | -0.3000 | 3.08e-04 | 4.43e-04 | 1.19e-04 | 4.13e-06 |
| 3 | Adam 50k | 0.4094 | -0.2071 | 0.223 | 0.0130 | 0.0390 | 1.41e-04 |
| | + L-BFGS | 0.4998 | -0.2999 | 3.68e-04 | 6.37e-04 | 1.09e-04 | 3.72e-06 |

Over seeds after refinement: alpha = 0.5006 ± 0.0010 (range 0.4998 to 0.5020), beta = -0.3004 ± 0.0012
(range -0.3022 to -0.2990); maximum absolute error 2.0e-3 in alpha and 2.2e-3 in beta. Parameter relative
error 2.1e-3 ± 2.0e-3 (median 1.8e-3) against 0.30 ± 0.05 for the same runs before refinement. Pressure
relative L2 7.8e-4 ± 2.5e-4 against 1.9e-2; permeability-field relative L2 4.2e-4 ± 3.3e-4 against 5.2e-2.

| seed | L-BFGS iterations | function evaluations | final max abs gradient | L-BFGS s | Adam s |
|---|---|---|---|---|---|
| 42 | 488 | 513 | 1.1e-04 | 12.2 | 513 |
| 0 | 615 | 649 | 3.9e-05 | 15.8 | 513 |
| 1 | 579 | 609 | 9.1e-05 | 14.7 | 520 |
| 2 | 665 | 708 | 7.9e-05 | 16.9 | 516 |
| 3 | 620 | 652 | 4.8e-05 | 15.8 | 516 |

**Stopping.** In every seed L-BFGS ended inside its first 1000-iteration step, far below `maxiter`
(15 000) and `maxfun` (18 750), and without reaching `gtol` (1e-8); the last max absolute gradient was 3.9e-5
to 1.1e-4. `torch.optim.LBFGS` returned early, which with `ftol = 0` happens when the line search can no
longer make progress; DeepXDE's documentation notes that this is expected in float32. The stop is therefore
a precision-limited stall, not convergence to the gradient tolerance. All logged function values and
parameters are finite.

**Independent check.** The saved post-refinement states were loaded on CPU into freshly built models. The
stored alpha and beta equal the reported estimates exactly, and the recomputed pressure and permeability
errors on the 41 x 41 grid agree with the GPU values to the printed precision (6 decimals for pressure, 4
significant digits for permeability).

**Interpretation, stated narrowly.** On this benchmark and under this protocol, a short L-BFGS refinement
(about 600 iterations, 15 s per seed on a T4) after 50 000 Adam iterations recovers both parameters to about
0.2 percent, while the same Adam state alone was 30 percent off. Total training loss fell by one to three
orders of magnitude during refinement, but the recovery claim rests on the parameter errors, not on the
loss. Whether a shorter Adam stage would suffice before L-BFGS has not been tested. These are noise-free
observations; the noisy case is part of the sensitivity study.

**Cost.** Per seed on one T4: Adam stage 515 ± 3 s, L-BFGS 15.1 ± 1.8 s, total 531 ± 3 s. The Adam stage
took about 5 percent longer than in v3 (492 s) on the same nominal hardware and identical software, an
example of session-to-session timing variation.

### Sensitivity study (Deviation 4)

Both kernels ran at commit d3ee5b2 on Tesla T4s with `pip freeze` identical to v4. Every row has 5/5 seeds
with `status: ok`, finite values throughout, and an L-BFGS stop by the same float32 line-search stall as in
v4. Statistics are mean ± sample standard deviation over seeds 42, 0, 1, 2, 3.

**Consistency check (fixed in advance).** In the initialisation-only and layout-only rows, the seed-42
member uses network seed 42 and observation seed 42, so it is the v4 baseline seed-42 run. In both rows it
reproduces v4 exactly: identical Adam trace, the same 488 L-BFGS iterations, and identical final alpha,
beta, pressure and permeability errors.

**Spatial k, `adam+lbfgs`, one factor changed per row** (baseline: 128 sensors, noise-free, start at 0, 0;
first line repeated from v4 for comparison).

| row | alpha | beta | parameter rel. error: median (max) | pressure rel. L2 | k-field rel. L2 | criterion 3 |
|---|---|---|---|---|---|---|
| baseline, 128 sensors (v4) | 0.5006 ± 0.0010 | -0.3004 ± 0.0012 | 1.8e-3 (5.1e-3) | 7.8e-4 ± 2.5e-4 | 4.2e-4 ± 3.3e-4 | recovered |
| 16 sensors | 0.5003 ± 0.0008 | -0.3004 ± 0.0009 | 1.7e-3 (3.6e-3) | 6.4e-4 ± 1.5e-4 | 3.2e-4 ± 2.0e-4 | recovered |
| 32 sensors | 0.4996 ± 0.0008 | -0.2999 ± 0.0005 | 1.5e-3 (2.9e-3) | 8.1e-4 ± 3.0e-4 | 3.5e-4 ± 2.5e-4 | recovered |
| 64 sensors | 0.5004 ± 0.0008 | -0.3005 ± 0.0006 | 1.2e-3 (3.5e-3) | 7.2e-4 ± 3.9e-4 | 3.2e-4 ± 2.2e-4 | recovered |
| 256 sensors | 0.5009 ± 0.0008 | -0.3009 ± 0.0006 | 2.4e-3 (4.2e-3) | 7.8e-4 ± 2.2e-4 | 5.1e-4 ± 1.9e-4 | recovered |
| initialisation only (layout fixed) | 0.5005 ± 0.0005 | -0.3004 ± 0.0009 | 1.8e-3 (2.8e-3) | 7.3e-4 ± 1.5e-4 | 3.8e-4 ± 0.9e-4 | recovered |
| layout only (network fixed) | 0.5010 ± 0.0012 | -0.3007 ± 0.0013 | 2.4e-3 (5.9e-3) | 6.4e-4 ± 1.2e-4 | 5.6e-4 ± 3.8e-4 | recovered |
| far start (alpha = -1, beta = 1) | 0.5003 ± 0.0004 | -0.3003 ± 0.0002 | 7.8e-4 (1.6e-3) | 7.5e-4 ± 2.3e-4 | 1.9e-4 ± 1.4e-4 | recovered |
| noise sigma = 0.01 | 0.5031 ± 0.0071 | -0.3026 ± 0.0117 | 1.9e-2 (3.6e-2) | 2.7e-3 ± 0.9e-3 | 3.5e-3 ± 2.4e-3 | not recovered |
| noise sigma = 0.05 | 0.5150 ± 0.0379 | -0.3121 ± 0.0615 | 0.10 (0.18) | 1.3e-2 ± 0.4e-2 | 1.9e-2 ± 1.1e-2 | not recovered |

**Constant k, 50 000 Adam iterations.**

| row | k | relative k error (mean, max) | pressure rel. L2 | criterion 3 |
|---|---|---|---|---|
| baseline, start 0.8 (v3) | 1.7002 ± 0.0033 | mean 1.3e-3, max 2.9e-3 | 1.2e-2 ± 0.4e-2 | recovered |
| noise sigma = 0.01 | 1.7004 ± 0.0035 | mean 1.6e-3, max 2.9e-3 | 9.2e-3 ± 2.2e-3 | recovered |
| start k = 5 | 1.7114 ± 0.0013 | mean 6.7e-3, max 7.8e-3 | 1.1e-2 ± 0.1e-2 | recovered |

**Findings, stated narrowly.**

- **Sensor count.** The pre-registered expectation that error grows as sensors decrease is **not
  supported** between 16 and 256 noise-free sensors. Every row is recovered, the median parameter error stays
  between 1.2e-3 and 2.4e-3, and there is no monotone trend over this range. The cause was not
  investigated.
- **Initialisation versus sensor layout.** Both are recovered when varied separately. The spread over
  seeds is somewhat smaller with the layout fixed (parameter error std 6.9e-4, max 2.8e-3) than with the
  network fixed (std 2.0e-3, max 5.9e-3), which is close to the baseline where both vary (std 2.0e-3, max
  5.1e-3). With five seeds per row this is descriptive only; no test of the difference was made.
- **Observation noise.** The expectation that error grows with noise is supported. With noisy data the
  pre-registered tolerance is not met: median parameter error 1.9e-2 at sigma 0.01 and 0.10 at sigma 0.05.
  Spread across seeds is large (at sigma 0.05, from 5.0e-3 to 0.18), and each seed draws both its sensor
  layout and its noise realisation, so the two are confounded in these rows. The noisy fit itself is close:
  pressure relative L2 2.7e-3 and 1.3e-2. No regularisation or noise-aware weighting was used.
- **Far initial guess.** Recovered (max error 1.6e-3) although the Adam stage ended further from the truth
  (parameter error 0.74 ± 0.13, against 0.30 in the baseline); L-BFGS then needed more iterations (728 to
  1654).
- **Constant k.** Recovered with noise 0.01 and from k = 5. From k = 5 the estimate approaches the truth from
  above and was still decreasing at the end (by 0.0013 to 0.0027 over the last 5000 iterations); the
  maximum error of 7.8e-3 is inside the tolerance with little margin.

**Cost.** Kernel v5: 16 139 s in 6 rows plus a 33 s smoke gate (about 4.5 GPU hours). Kernel b v1:
11 655 s in 5 rows plus a 37 s smoke gate (about 3.2 GPU hours); the two kernels ran at the same time. Mean
time per seed by row: 501 to 547 s for spatial rows and 399 to 403 s for constant rows. Mean L-BFGS time per
seed by row: 9.1 s (noise 0.05) to 26.0 s (far start).

## Open work

1. **Spatial convergence.** Resolved for this benchmark by the user's choice (b), Adam followed by
   L-BFGS (Deviation 3, results above).
2. **Sensitivity study** (criterion 5): completed (Deviation 4, results above).
3. **Noisy observations.** Spatial k does not reach the tolerance with noise 0.01 or 0.05; untested
   remedies include loss weighting by the noise level, regularisation of the parameters, and more sensors
   with noise. Noise realisation and sensor layout are confounded in the noise rows.
4. **Shorter Adam stages before L-BFGS**, float64 L-BFGS, other architectures, learning rates and loss
   weights are untested.
5. **Comparison with the FEM-adjoint study** remains out of scope: the problems differ in dimension,
   noise and discretisation.

## Adam + L-BFGS refinement protocol (Deviation 3, fixed before the run)

**Configuration `adam+lbfgs`**, separate from and not replacing the Adam-only default: the spatial
benchmark exactly as in the 50 000-iteration diagnostic (same PDE, network, data, sensor layouts, seeds 42,
0, 1, 2, 3, loss definition with unit weights, evaluation grid), then DeepXDE's PyTorch L-BFGS
(`torch.optim.LBFGS`, strong-Wolfe line search) on the same loss and variables with DeepXDE's default
options: history 100, `gtol` 1e-8, `ftol` 0, `maxiter` 15 000, `maxfun` 18 750, up to 50 line-search steps,
float32. Runner flag `--lbfgs-maxiter 15000`; launcher stage `lbfgs`.

**Why the Adam stage is rerun.** Kernel v3 stored parameter traces and metrics but no network weights, so
there is no complete state to resume from. The Adam stage is therefore reproduced from scratch in the same
process. As a check that refinement starts from the same state as the documented Adam-only result, the
Adam-stage trace is compared with kernel v3 at all 101 shared checkpoints (every 500 iterations); any
difference is reported. The Adam-only metrics of the same run (`adam_stage` in each record) are the primary
comparison, and kernel v3 is the historical reference. Network and parameter state after each stage is
saved to the kernel output (`--save-state`) and is not committed.

**Recorded.** Per seed: Adam-stage and final estimates and errors, L-BFGS iterations, function
evaluations, stop reason, final max absolute gradient, total loss at every 10th function evaluation,
parameter values per DeepXDE outer step (at most 1000 L-BFGS iterations each), final loss components, and
separate synchronised times for the two stages.

**DeepXDE detail found while implementing.** DeepXDE 1.15 computes the PyTorch per-step limits
(`iter_per_step = min(1000, maxiter)`, `fun_per_step`) only at import, from the default options; a later
`set_LBFGS_options(maxiter=...)` does not update them, so budgets below 1000 or not a multiple of 1000 are
overshot. `src/deepxde_lbfgs.py` recomputes them with DeepXDE's formula. For `maxiter` 15 000 the values
(1000, 1250) are the defaults, so the full run is unaffected; the 20-iteration smoke case is. Also, when
L-BFGS stops early, DeepXDE leaves its loop before callbacks and loss updates run; the final point and loss
components are recorded explicitly.

**Acceptance (unchanged).** Criterion 3 as pre-registered: *recovered* only if the relative parameter
error is below 1e-2 for every seed. Training loss is reported but is not evidence of recovery.

**Decision rules fixed in advance.**
- Smoke first: the 2-iteration gate for all three cases (constant, spatial, spatial with 20 L-BFGS
  iterations) must pass before the full run.
- If refinement does **not** meet criterion 3, no further GPU time is spent. The next step is an analysis of
  identifiability, loss-component balance and gradient behaviour, using the saved states on CPU (loss and
  gradient evaluations only, no training).
- The sensitivity study (criterion 5) runs only if `adam+lbfgs` meets criterion 3. It then uses `adam+lbfgs`,
  with the network-initialisation seed and the sensor-layout seed varied separately.

## Sensitivity study protocol (Deviation 4, fixed before the runs)

The pre-registered matrix (criterion 5) assumed the 10 000-iteration Adam baseline, which is now known not
to converge. Each row therefore uses the configuration that met criterion 3 for its benchmark, and changes
exactly one factor (checked by a unit test):

- **Spatial rows: `adam+lbfgs`** as in Deviation 3. Sensor count 16, 32, 64, 256 (baseline 128).
  Observation noise standard deviation 0.01 and 0.05 (absolute; pressure peak 1). Far initial guess
  (alpha = -1, beta = 1).
- **Separating initialisation from sensor layout** (added at the user's request). *Initialisation only*:
  sensor layout fixed by `--observation-seed 42`, network seeds 42, 0, 1, 2, 3. *Layout only*: network fixed
  by `--network-seed 42`, observation seeds 42, 0, 1, 2, 3. The baseline varies both together. In both rows
  seed 42 is the baseline seed-42 run and must reproduce it.
- **Constant rows: 50 000 Adam iterations** as in Deviation 1, with noise 0.01, and with initial k = 5.

Every row runs seeds 42, 0, 1, 2, 3 and reports criterion 3's label, plus mean, standard deviation, median,
minimum and maximum of every metric. The expectations stated in criterion 5 are unchanged: error grows as
sensors decrease and as noise increases. No prediction is made for the initialisation rows. With noise, the
data no longer determine the truth exactly, so criterion 3 is reported for those rows as a description,
not as a pass or fail of the method.

Execution: launcher stages `sensitivity_a` (sensor counts, initialisation only, layout only; estimated
4.4 GPU hours) and `sensitivity_b` (noise, far initial guess, constant rows; estimated 3.4 GPU hours). The
estimates come from the measured 531 s per spatial seed and 407 s per constant seed.

## Deviations

1. **Post-hoc convergence diagnostic (added after the baseline was seen).** Not part of the original
   pre-registration. Same benchmarks, seeds, network, optimiser, learning rate, data and evaluation as the
   baseline; only `--iterations 50000` (and `--trace-every 500`) change. Question: do the estimates keep
   approaching the truth and reach the pre-registered 1e-2 threshold? The pre-registered labels are applied
   unchanged and reported as diagnostic, not as a test of the original hypothesis. Launcher stage
   `convergence`.
2. **Sensitivity study postponed.** The pre-registered sensitivity matrix (criterion 5) assumes a
   converged baseline. At 10 000 iterations it would measure the rate of convergence rather than sensitivity
   to sensors, noise or initialisation, so it is not run until a converged budget or optimiser is agreed.
   Changing the optimiser (for example Adam followed by L-BFGS) would be a change of method and is left as an
   explicit decision.
3. **L-BFGS refinement** (`adam+lbfgs`), chosen by the user after the 50 000-iteration diagnostic; protocol
   in the section above.
4. **Sensitivity study configuration**: validated configurations instead of the 10 000-iteration budget,
   and an added layout-only row; see *Sensitivity study protocol*.
