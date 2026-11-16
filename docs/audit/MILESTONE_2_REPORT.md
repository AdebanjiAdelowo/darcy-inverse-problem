# Milestone 2: Baseline Reproduction and Diagnostic Infrastructure

Date: 2026-11-16
Base commit: `a7a6447` (main). Milestone 2 implementation changes are **uncommitted** and left for review.
Qualification: **Qualified, Not Strong to Discuss** (unchanged).
Milestone verdict: **PASS** (Section 10).

All numbers below come from files under `results/m2/` produced by the commands in Section 9. No
historical artifact in `results/` or `figures/` was modified.

---

## 1. Housekeeping

| Action | Result |
|---|---|
| Commit noise-runner recursion fix and its test | `3b8f282` fix: stop peak-memory helpers in the noise-study runner from recursing |
| Commit Milestone 1 audit files | `a7a6447` docs: add Milestone 1 repository and scientific audit |
| Review `origin/audit/scientific-audit` (draft PR #1) | `docs/audit/PRIOR_AUDIT_BRANCH_REVIEW.md`. Branch neither merged nor deleted. Unique content: the MAP-candidate labelling (ported to `scripts/run_posterior_demo.py`, presentation only), the local-vs-full MAP agreement observation (Section 5.6), and sampler-comparison guidance kept for Milestones 5 and 6. Everything else is superseded by Milestone 1 |
| Push, merge, tag, release | none |

---

## 2. Computational environment

| Item | Value | Source |
|---|---|---|
| Hardware | Apple M3 Pro, 11 cores (5 performance, 6 efficiency), 18 GiB | `results/m2/environment.json` |
| OS | macOS 26.6.2 (Darwin 25.6.0), arm64 | same |
| FEM environment | conda `fenicsx`: Python 3.12.12, dolfinx 0.10.0, basix 0.10.0, UFL 2025.2.1, PETSc/petsc4py 3.24.3, mpi4py 4.1.2 (MPICH 4.3.2), NumPy 2.5.3, SciPy 1.18.1, OpenBLAS 0.3.34 | same |
| PINN environment | `.venv-deepxde`: Python 3.12.2, torch 2.14.1, DeepXDE 1.15.0, NumPy 2.5.4 | same |
| Solver configuration | forward and adjoint: KSP `preonly`, PC `lu`, MUMPS; P2 pressure, P1 log-permeability; L-BFGS-B gtol 1e-6, ftol 1e-12, SciPy defaults maxcor 10, maxls 20 | same |
| Disk | 34 to 40 GiB free during the milestone; Milestone 2 outputs 5.7 MB | `df`, `du` |

New specification files (no environment was modified, nothing was installed or upgraded):

- `environment/fenicsx-osx-arm64.explicit.txt`: exact osx-arm64 builds (`conda list --explicit`).
- `environment/fenicsx.pinned.yml`: all packages version-pinned, build strings removed, local prefix removed.
- `environment/deepxde-requirements.txt`: `uv pip freeze` of the PINN venv. A `uv.lock` was not created
  because resolving it needs network access.
- `scripts/record_environment.py`: writes the provenance JSON above.

### 2.1 Thread settings and determinism (new finding)

| Setting | Behaviour |
|---|---|
| `OMP_NUM_THREADS=1` | bitwise deterministic run to run, including full 200-iteration reconstructions and the MAP (two runs compared) |
| `OMP_NUM_THREADS` = 2, 4, 8 | deterministic differences from the stored files at last-bit level |
| `OMP_NUM_THREADS` = 11 or unset (historical setting) | reproduces short computations bit for bit, but two identical full-configuration runs differed by up to 2.6e-5 (reconstruction and MAP) |
| `OPENBLAS_NUM_THREADS` | no effect on results |

The historical artifacts were produced with 11 OpenMP threads. The mechanism (summation order in the
OpenMP-threaded sparse direct solver) is inferred, not traced. **Recommendation: all future experiments
use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.** Details: `docs/REPRODUCIBILITY.md`.

---

## 3. Optimiser diagnostics

`src/optim_diagnostics.py` wraps the existing SciPy call. Objective, regularisation, options, start and
method are unchanged. Proof: in deterministic mode, instrumented and plain SciPy runs return bitwise identical iterates at
the full reconstruction and MAP configurations (Appendix B), and `tests/test_optimizer_diagnostics.py::test_diagnostics_do_not_change_the_optimisation`.

Recorded per run (`OptimizerDiagnostics.to_dict()`): initial and final objective and their misfit and
regularisation (prior) parts; final gradient 2-norm and inf-norm, relative gradient reduction, and whether
the first-order condition ||g||_inf <= gtol holds; iterations, function and gradient evaluations; forward
and adjoint solve counts; SciPy status and message; termination reason; options; runtime; per-evaluation
traces; and, when the truth is known, the error metrics of Section 6 and the sensor-prediction error.

**Convergence rule.** `converged` is true only if SciPy stopped on its projected-gradient test or its
relative-function-reduction test. An iteration-cap stop is never classified as converged.

### 3.1 Results (48 runs, single-thread mode; `results/m2/optimizer_diagnostics_single_thread.csv`)

| Experiment | Runs | Termination | Converged | ||g||_inf range | First-order condition met |
|---|---:|---|---:|---|---:|
| Reconstruction demo (alpha 30, max 200) | 2 | relative function reduction (139 and 125 it) | 2 | 6.5e-6, 6.7e-6 | 0 |
| Regularisation study (max 120) | 9 | iteration limit | 0 | 5.1e-6 to 1.5e-2 | 0 |
| Noise study (max 120) | 5 | iteration limit | 0 | 2.1e-5 to 1.2e-4 | 0 |
| Sensor study (max 120) | 30 | iteration limit | 0 | 2.0e-5 to 3.6e-4 | 0 |
| Bayesian MAP (max 120) | 2 | iteration limit | 0 | 3.6e-4 (smooth), 8.1e-3 (structured) | 0 |
| **Total** | **48** | | **2** | | **0** |

Every run made exactly one forward and one adjoint solve per function evaluation (e.g. smooth demo: 145
evaluations, 145 + 145 solves, 7.5 s).

Observations, without reclassifying anything:

- The two converged runs stopped on relative function reduction, not on the gradient test. Their gradient inf-norm
  (about 6.6e-6) is above gtol = 1e-6, while the gradient 2-norm fell by a factor of about 1e-7 from its
  initial value.
- At alpha = 30 the 120-iteration study iterate has the same objective (40.1661) and errors as the
  converged demo run, so the study's headline row is close to a converged value. Its convergence status is still "iteration_limit".
- The structured MAP stops with a gradient inf-norm of 8.1e-3, 22 times the smooth case.

---

## 4. Forward-only pCN likelihood path

`BayesianDarcyProblem.neg_log_likelihood` now performs one forward solve (`InverseProblem.misfit_only`)
instead of forward + adjoint + gradient assembly. The historical path is kept as
`neg_log_likelihood_legacy` and `run_bayesian_experiment(..., likelihood_path="legacy")`. Prior, posterior,
proposal and acceptance rule are untouched (`src/pcn_sampler.py` is unchanged).

Benchmark at the historical full configuration (N=16, r=15, 25 sensors, beta 0.02, smooth truth, data seed
7, chain started at the MAP, seed 1); `results/m2/reproduction/single_thread/pcn_benchmark.json`:

| Quantity | Legacy | Forward-only |
|---|---:|---:|
| Likelihood values at 200 points: max abs difference | | **0.0 (bitwise identical)** |
| Single evaluation (two passes) | 31.6, 30.0 ms | 12.6, 12.4 ms |
| 1,000-proposal chain, 2 repeats (alternating order) | 29.5, 29.8 s | 12.1, 12.9 s |
| ms per proposal | 29.6 | 12.5 |
| Forward / adjoint solves per chain | 1001 / 1001 | 1001 / 0 |
| Acceptance rate | 0.391 | 0.391 |
| Chains bitwise identical (all 4) | | **yes**, max difference 0.0 |
| **Speed-up** | | **2.38x** |

Full-length use: the reproduced historical posterior demo ran its 10,000 proposals in 121.8 s (smooth) and 86.5 s
(structured) with 10,001 forward and 0 adjoint solves each (single-thread mode, 4 concurrent jobs).

---

## 5. Identifiability baseline

### 5.1 Quantities distinguished (`src/identifiability.py`)

| Quantity | Identifiable from pressure data? | Metric |
|---|---|---|
| absolute log-permeability m(x) | no | `rel_error_abs_nodal` (historical), `rel_error_abs_L2` |
| mean log-permeability (1/abs(Omega)) integral of m | no (set by initial guess or prior) | `mean_offset_L2` |
| mean-free log-permeability m minus its mean | yes | `rel_error_meanfree_nodal`, `rel_error_meanfree_L2` (equals the quotient-space distance min over c of the L2 error of m + c) |
| spatial contrast k / k_g = exp(m minus its mean) | yes | `rel_error_contrast_L2` |
| pressure prediction H p(m) | yes | `rel_error_pred_vs_clean`, `rms_residual_vs_noisy_over_sigma` |

### 5.2 Regression tests (`tests/test_identifiability.py`, 14 tests, all pass)

- Full pressure field unchanged under m -> m + c for c in {-1.3, 0.4, 2.0} (max abs difference below 1e-12);
  control: a non-constant perturbation changes p by more than 1e-3.
- Misfit and H1-regularised objective unchanged to 1e-12 relative; control: the L2-to-reference objective
  changes by exactly the L2-penalty difference.
- Misfit gradient orthogonal to the constant direction (to 1e-3 relative; measured 5.4e-5 in Milestone 1).
- KL mode 0 is constant, with lambda_0 = delta^-2 (checked for delta = 1 and 2), phi_0 = 1/sqrt(abs(Omega)) = 1, so the
  pointwise variance floor is delta^-2 and the integrated floor is delta^-2.
- Likelihood independent of xi_0; its xi_0 gradient component vanishes.
- Error metrics: the mean-free and contrast errors are invariant to adding a constant; the absolute error is not.
- Historical full chains violate the exact variance floor (regression record of the Milestone 1 finding).

### 5.3 Theoretical vs stored posterior variance (reproduced chains, single-thread)

| Truth | Pointwise Var[m(x)|y], min to max | Exact floor | Fraction of nodes below floor | Integrated variance | Integrated floor |
|---|---|---:|---:|---:|---:|
| smooth | 0.089 to 0.456 | 1 | 100% | 0.295 | 1 |
| structured | 0.137 to 0.639 | 1 | 100% | 0.367 | 1 |

### 5.4 Absolute vs identifiable errors (single-thread reproduction)

| Estimate | abs nodal (historical) | abs L2 | mean-free L2 | contrast L2 | mean offset |
|---|---:|---:|---:|---:|---:|
| Smooth reconstruction (alpha 30) | 0.3489 | 0.3468 | 0.3468 | 0.1488 | -0.0021 |
| Structured reconstruction | 0.8488 | 0.8422 | 0.8417 | 0.4295 | +0.0210 |
| Smooth MAP candidate | 0.3869 | 0.3756 | 0.3756 | 0.1977 | -0.0002 |
| Structured MAP candidate | 0.8073 | 0.7691 | 0.7682 | 0.3944 | +0.0212 |
| Smooth posterior mean | 1.0010 | 0.9887 | 0.9768 | 0.4050 | +0.0655 |
| Structured posterior mean | 1.4655 | 1.4503 | **1.0080** | 0.3923 | **+0.3963** |

Both truths have near-zero mean, so the deterministic and MAP errors are almost entirely mean-free error.
The structured posterior-mean error is the exception: about a third of it is drift of the unidentifiable constant mode in the
unconverged chain.

### 5.5 Other measures recorded

The pressure-prediction error at the sensors for the smooth reconstruction is 0.96% relative to the clean data,
with an rms residual of 0.73 sigma against the noisy data. The contrast error (0.149) is less than half of the log-field error (0.347).

### 5.6 Local vs full MAP (unique observation from the earlier audit)

The stored local (80-iteration) and full (120-iteration) MAP candidates give 0.3866 and 0.3868. The full
run's final gradient inf-norm is 3.6e-4 with a relative gradient reduction of 1.1e-7, so the iterate has
nearly stopped moving although SciPy's tests are not met. Whether it is a minimiser remains for
Milestone 3 (longer budget, same objective).

---

## 6. Baseline reproduction

Original scripts and configurations were run unchanged via `scripts/reproduce_baselines.py`, in two modes:
single-thread (4 concurrent processes, 223 s total) and historical threading (11 OpenMP threads, one process at a
time, 898 s total). Automated comparison: `scripts/compare_baselines.py`, results in
`results/m2/baseline_comparison_{single_thread,historical_threads}.json`. Automated verdicts are strict.
"discrepancy" means some printed number differs beyond relative 1e-3 or an integer differs, and is explained
below.

| Experiment | Original (stored) | Reproduced (single-thread) | Abs / rel difference | Convergence | Wall time s (1T / 11T) | Verdict |
|---|---|---|---|---|---|---|
| Forward MMS convergence | L2 1.0754e-06, H1 5.2770e-04 at N=64; orders 3.000/1.999 | identical | 0 | n/a | 2.0 / 1.1 | **bitwise** (both modes) |
| Adjoint Taylor test | orders 2.000, 2.001, 2.000 | identical orders; remainders differ in 5th to 7th digit (1T) | max rel 1.9e-4 on remainders of size 5.7e-8 | n/a | 0.5 / 0.7 | **bitwise** (11T); printed-precision orders (1T) |
| Bayesian gradient Taylor test | orders 1.997 to 2.000 | identical orders | max rel 5.2e-7 | n/a | 0.4 / 0.6 | **bitwise** (11T); printed precision (1T) |
| Linear-Gaussian sampler validation | mean rel. err 5.9038e-03, var MAE 5.8597e-04, mean ESS 398.2 | identical | 0 | n/a | 1.0 / 0.9 | **bitwise** (both) |
| Smooth reconstruction | 0.3489, misfit 9.6271, reg 30.539, 138 it, converged, 12.00 s | 0.3489, 9.6271, 30.539, **139 it**, converged, 8.45 s | values identical to printed precision; +1 iteration | converged (relative reduction), first-order condition not met | 17.3 / 20.4 | **reproduced to printed precision** (iteration count +1) |
| Structured reconstruction | 0.8488, 24.074, 14.223, 125 it, converged | identical, 125 it | 0 | converged (relative reduction) | (same run) | **reproduced to printed precision** |
| Regularisation study (9 alphas) | e.g. alpha 30: 0.3489, misfit 9.6270; all 9 `False` at 120 | 8 rows identical; alpha 0.1 misfit 5.4594e-02 vs 5.4530e-02 (1T), reg 5.9333e-01 vs 5.9340e-01 | max rel 1.2e-3 (1T), 1.6e-3 (11T), only at alpha 0.1 | 9/9 iteration limit (reproduced failure) | 59.8 / 78.4 | **reproduced**; the alpha 0.1 row (least regularised, unconverged) is roundoff-sensitive; discrepancy-principle choice and retrospective best (alpha 30) unchanged |
| Noise study (5 levels) | 0.3461, 0.3544, 0.3720, 0.4422, 0.6164 | identical | 0 | 5/5 iteration limit | 35.7 / 42.4 | **bitwise** (both) |
| Sensor study (30 runs) | grid 0.6182, 0.3660, 0.3527, 0.3543, 0.3508 | identical | 0 | 30/30 iteration limit (not recorded historically) | 191.5 / 271.6 | **bitwise** (both) |
| Historical MAP (smooth / structured) | 0.3868 / 0.8074, `False`, 120 it | 0.3869 / 0.8073, `False`, 120 it | 1e-4 in last printed digit; xi_MAP differs by up to 1.3e-4 | 2/2 iteration limit (reproduced failure) | (in posterior demo) | **reproduced to last printed digit** |
| pCN chains, full config (10,000 proposals each) | acceptance 0.3998 / 0.4398; ESS 4.9 to 86.1 / 5.0 to 72.7 | identical acceptance and identical ESS list; **all 8,000 retained accept/reject decisions identical** (3,205 and 3,529 moves); sample values within 1.2e-4 / 3.0e-4 | posterior summaries differ by at most 1 in the 4th decimal (largest rel 2.7e-3, on a posterior mean of 0.0370 vs 0.0369) | not converged (reproduced failure) | 220.2 / 470.1 | **reproduced to printed precision**, same Markov path |
| Credible intervals, predictive coverage | 5/5, 5/5; 0.760, 0.760 | identical | 0 | | | reproduced (scientifically invalid, Section 7) |

**Explanation of non-bitwise rows.** In deterministic single-thread mode the code is bitwise reproducible.
The stored files came from 11-thread runs that are themselves not deterministic for long iterative
computations (Section 2.1). Last-bit differences are amplified by (a) the relative-function-reduction
stopping test (smooth demo: one more iteration, same printed values), (b) the weakly regularised
alpha = 0.1 run, which is far from convergence, and (c) the unconverged MAP. The pCN chains start at the MAP and
inherit its 1e-4 offset, but make identical accept/reject decisions. No configuration was changed and nothing
was tuned.

**Not reproduced in this milestone:** the Bayesian sensor, noise and prior-sensitivity studies (about 70 min
of the historical runtime, and scientifically invalid per Section 7), the `local`/`smoke` configurations, and
the structured-truth alpha sweep, which has no stored artifact or script.

---

## 7. Historical scientific claims

Marked in `README.md` (a "Known limitations of the stored results" section near the top plus inline audit notes; the
historical numbers are kept unchanged):

| Claim or result | Status | Evidence |
|---|---|---|
| Absolute permeability scale | not identifiable | Section 5.2 tests |
| Stored Bayesian variances, credible intervals, integrated variance | violate the exact floor (all nodes) | Section 5.3; `tests/test_identifiability.py::test_historical_full_chains_violate_exact_variance_floor` |
| Historical chains | not converged (split R-hat up to 3.4; reproduced ESS identical) | Milestone 1; Section 6 |
| Bayesian sensor, noise, prior-sensitivity trends | not validated; confounded with acceptance rate | README notes |
| "Intervals almost certainly wider than a converged chain" | wrong direction; marked | README note |
| "Aggregate quantities more robust" | not supported; marked | README note |
| Posterior predictive 76% coverage | invalid construction (20 draws, no noise); marked | README note |
| Part I studies and MAP at iteration limit | 46 of 48 runs at the cap; marked | Section 3.1; README notes |
| 2,000-iteration MAP rerun | no artifact; marked | README note |
| Structured alpha sweep 0.82 to 0.89 | no artifact; marked unverified | README note |
| MAP vs Part I "comparable settings" | settings differ; marked | README note |
| "MAP is the H1-family-equivalent Tikhonov estimate" | imprecise; marked | README note |
| Test count "56 tests" | replaced (the suite has grown) | README |

The development-status block at the top of the README ("Working status", "pending") was replaced by a factual
limitations section, consistent with the public-README policy. New README prose uses no em dashes; older
README text was not rewritten (Milestone 9).

---

## 8. Tests

| Environment | Command | Result |
|---|---|---|
| conda `fenicsx` | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/anaconda3/envs/fenicsx/bin/python -m pytest -p no:cacheprovider -q tests` | **122 passed, 15 skipped** (skips: PINN tests, torch absent), 23.2 s |
| `.venv-deepxde` | `DDE_BACKEND=pytorch .venv-deepxde/bin/python -m pytest -p no:cacheprovider -q tests/test_deepxde_*.py tests/test_ess_estimator_bias.py` | **49 passed**, 9.1 s |

137 distinct tests (103 pre-existing, 34 new), all passing:

| New test file | Tests | Covers |
|---|---:|---|
| `tests/test_identifiability.py` | 14 | invariance, controls, KL constant mode, variance floor, metrics, historical floor violation |
| `tests/test_optimizer_diagnostics.py` | 14 | termination classification (9 messages), quadratic convergence, iteration cap never converged, counts and final values, instrumentation does not change the iterate, MAP diagnostics and historical interface |
| `tests/test_likelihood_path.py` | 4 | bitwise likelihood equality, zero adjoint solves, identical seeded chains (accept and reject exercised), experiment-level equality and solve counts |
| `tests/test_ess_estimator_bias.py` | 2 | ESS accuracy at tau = 19; upward bias above the lag cap (estimator unchanged) |

Resource use: at most 4 single-thread processes, except the historical-threading pass (one process at a time, 11
threads), which was justified by its purpose (provenance) and measured load (6.6 to 8.2 on 11 cores, 34% memory free).
No PINN training and no GPU session were started.

---

## 9. Reproduction commands

```bash
cd ~/Documents/GitHub/darcy-inverse-problem
PY=/opt/anaconda3/envs/fenicsx/bin/python
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
$PY scripts/record_environment.py --out results/m2/environment.json --deepxde-python .venv-deepxde/bin/python
printf "%s\n" posterior_demo sensor_study regularization_study pcn_benchmark noise_study reconstruction_demo \
  verify_forward validate_sampler verify_bayesian_gradient verify_gradient |
  OMP_NUM_THREADS=1 xargs -P 4 -I{} $PY scripts/reproduce_baselines.py --experiment {} --tag single_thread
for e in verify_forward verify_gradient verify_bayesian_gradient validate_sampler reconstruction_demo \
         regularization_study noise_study sensor_study posterior_demo; do
  OMP_NUM_THREADS=11 $PY scripts/reproduce_baselines.py --experiment $e --tag historical_threads; done
for t in single_thread historical_threads; do
  $PY scripts/compare_baselines.py --tag $t; $PY scripts/summarize_m2.py --tag $t; done
```

Output files:

- `results/m2/environment.json`
- `results/m2/reproduction/{single_thread,historical_threads}/*.json`, with `text/` holding the scripts' own outputs and the chain archives
- `results/m2/baseline_comparison_{single_thread,historical_threads}.json`
- `results/m2/optimizer_diagnostics_{single_thread,historical_threads}.{csv,json}`
- `results/m2/bayesian_baseline_{single_thread,historical_threads}.json`
- `results/m2/reproduction/single_thread/pcn_benchmark.json`
- `results/m2/logs/`

---

## 10. Acceptance criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Existing verified functionality intact | met | all 103 pre-existing tests pass; forward/gradient/sampler verification bitwise in the historical setting |
| 2 | New regression tests pass | met | 34/34 |
| 3 | Optimiser convergence and termination diagnostics recorded correctly | met | Section 3; 48 runs; tests |
| 4 | Optimised pCN path preserves numerical behaviour | met | bitwise likelihoods and chains; 2.38x |
| 5 | Constant-mode non-identifiability verified by regression tests | met | Section 5.2 |
| 6 | Historical baselines reproduced or discrepancies explained | met | Section 6: 6 of 9 experiments bitwise, 3 to printed precision with explained roundoff amplification; chains with identical accept/reject paths |
| 7 | Invalid historical claims identified | met | Section 7; README |
| 8 | Configurations and dependency versions recorded | met, with one gap | Section 2; no `uv.lock` (network) |
| 9 | Outputs traceable to code and configuration | met | every JSON record holds the commit, uncommitted paths, thread settings, script and argv |

**Verdict: PASS.** Limitations of this pass: the three optimiser-heavy experiments are not bitwise reproducible
(structurally impossible for the stored 11-thread files); the Bayesian study scripts and the structured alpha sweep were
not rerun; there is no `uv.lock`.

---

## 11. Remaining blockers (prioritised)

| Priority | Blocker | Next milestone |
|---|---|---|
| 1 | Bayesian posterior invalid: exact variance floor violated, chains unconverged | M5 (diagnostics), M6 (exact constant-mode treatment, better sampler; approval needed) |
| 2 | 46/48 optimiser runs at the iteration cap; the 2 converged runs do not meet the first-order condition | M3 (longer budgets, same objective and tolerances) |
| 3 | Single noise realisation and truth draw for every headline number | M3 (multi-seed), M8 |
| 4 | No posterior calibration; predictive check invalid | M7 |
| 5 | Structured alpha sweep has no artifact; no edge-preserving comparison | M4 |
| 6 | No inversion-mesh refinement study | M3 |
| 7 | ESS estimator lag cap; no multi-chain R-hat or MCSE | M5 |
| 8 | Thread-dependent nondeterminism with OMP > 1 | adopt OMP_NUM_THREADS=1 for all new results (documented) |
| 9 | Draft PR #1 open; Milestone 2 changes uncommitted; no `uv.lock`; README prose cleanup | user approval; M9 |

---

## Appendix A. Files changed (uncommitted)

Modified: `README.md`, `scripts/run_posterior_demo.py` (MAP-candidate labels), `src/adjoint.py` (solve counters,
`misfit_only`; same arithmetic), `src/bayesian_darcy.py` (forward-only likelihood, legacy path, solve counts),
`src/bayesian_experiment.py` (MAP diagnostics, likelihood-path option, pCN timing and solve counts),
`src/experiment.py` (identifiability-aware errors attached to diagnostics), `src/inversion.py` (diagnostics wrapper).

New: `src/optim_diagnostics.py`, `src/identifiability.py`, `scripts/record_environment.py`,
`scripts/reproduce_baselines.py`, `scripts/compare_baselines.py`, `scripts/summarize_m2.py`, four test files,
`environment/` (3 files), `docs/REPRODUCIBILITY.md`, `docs/audit/MILESTONE_2_REPORT.md`,
`docs/audit/PRIOR_AUDIT_BRANCH_REVIEW.md`, `results/m2/` (5.7 MB).

## Appendix B. Instrumentation and determinism checks

At the full reconstruction (N=32, 36 sensors, alpha 30, 200 iterations) and the full smooth MAP (N=16, r=15, 120
iterations), comparing `run_lbfgs` / `compute_map_with_diagnostics` with a plain `scipy.optimize.minimize` call:

| Threads | Instrumented vs plain | Run 1 vs run 2 |
|---|---|---|
| `OMP_NUM_THREADS=1` | bitwise identical (both) | bitwise identical |
| `OMP_NUM_THREADS=11` | differ (MAP up to 4.8e-5) | differ (up to 2.6e-5) |

The 11-thread differences between instrumented and plain runs are of the same size as the differences between two
identical runs, so they reflect thread nondeterminism, not the instrumentation.
