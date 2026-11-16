# Milestone 1: Repository and Scientific Audit

Project: Deterministic and Bayesian Inversion of Spatially Varying Darcy Permeability
Audit date: 2026-11-16
Audited commit: `cff293a` (main, in sync with `origin/main`), plus two uncommitted user edits (Section 2.1)
Current qualification: **Qualified, Not Strong to Discuss** (unchanged by this audit)

Scope rules followed: no algorithm changed, no experiment launched beyond sub-minute timing and
read-only analysis of stored chains, no commit, no push. All new numbers in this report come from
(a) the test runs in Section 6, (b) `docs/audit/audit_stored_chains.py` applied to the committed
`results/posterior_samples_*.npz` (output: `docs/audit/audit_stored_chains_output.txt`), or (c) the
short inline checks listed in Appendix A.

---

## 1. Executive summary

1. **The deterministic core is sound and verified at the implementation level.** The forward solver passes a
   manufactured-solution study with textbook rates (P2: L2 order 3.000 and H1 order 1.999 at N=64 when the coefficient is exact), and the
   adjoint gradient passes Taylor tests with order 2.000 for both regularisers and for the KL-space
   Bayesian gradient. The full test suite passes: **103/103 across the two project environments**
   (88 pass + 15 skip in `fenicsx`; the 15 skipped plus 32 others pass in `.venv-deepxde`).
2. **A structural non-identifiability is not recognised anywhere in the formulation or documentation.**
   With f = 0 and Dirichlet/no-flow boundaries, p(m + c) = p(m) for every constant c (confirmed
   numerically: misfit changes by 3e-14 relative). Log-permeability is identifiable only up to an additive
   constant. The H1-seminorm regulariser does not penalise this direction, and KL mode 0 of the prior is
   exactly the constant function (lambda_0 = 1, phi_0 = 1).
3. **Consequence: the stored Bayesian posterior summaries are provably wrong, not just imprecise.** The
   likelihood does not depend on xi_0, so the exact (discretised) posterior has xi_0 ~ N(0,1), and
   therefore pointwise posterior variance >= 1 everywhere and integrated posterior variance >= 1.
   The stored chains give xi_0 standard deviation 0.35 (smooth) and 0.52 (structured), maximum
   pointwise posterior sd 0.68 and 0.80, and integrated posterior variances of 0.13 to 0.40 in every
   Bayesian study. All violate the analytical lower bound. The README states the credible intervals
   are "almost certainly wider than a fully-converged chain would give"; the opposite is true.
4. **MCMC non-convergence is now quantified:** split-R-hat (4 within-chain segments) ranges 1.01 to 3.38, and
   all 15 coordinates exceed 1.01 in every stored full and local chain; untruncated Geyer ESS median 5.2 to 6.6 out
   of 8,000 (the repository estimator truncates at lag 1000, which imposes an ESS floor near 4).
5. **The Bayesian "robust aggregate trends" are confounded with acceptance rate.** In both the sensor and
   noise studies, integrated posterior variance increases monotonically with pCN acceptance rate
   (a slowly moving chain has a small sample variance). Three nominally identical runs (25 sensors, 2% noise,
   r=15, differing only in data seed) report integrated variance 0.295, 0.234, 0.188 and posterior-mean
   error 1.00, 2.54, 1.01.
6. **Optimiser non-convergence is broader than previously documented.** Besides the Bayesian MAP
   (`converged=False`, 120 iterations, both truths), **all 9 regularisation-study runs** stop at the
   120-iteration cap with `converged=False`, and all 5 noise-study runs hit 120 iterations. Only the
   two headline reconstruction runs converged (138 and 125 iterations).
7. **Historical headline numbers trace to committed files** (34.9%, 84.9%, ESS 4.9 to 86.1, MAP not
   converged, linear-Gaussian validation). The "59 passing tests" figure has no source in the repository
   history: the pre-PINN suite has 56 tests (reproduced: 56 passed at `b7a190e`) and the current suite 103.
8. **The optional PINN track is well documented and manifest-backed.** One finding: a committed
   Cramer-Rao analysis (`results/deepxde_identifiability/analysis.json`) shows the 1e-2 recovery criterion is
   information-theoretically unreachable at noise 0.01 with 128 sensors (efficient-estimator rms error
   about 2.2%), and the PINN matches the classical MLE to 1e-3. This explains the "not recovered with
   noise" rows but is not yet documented. The committed noise-study runner contains an infinite
   recursion that the user's uncommitted edit fixes.

**Bottom line.** Part I is close to strong, needing converged reruns, multi-seed statistics and an
identifiability-aware error metric. Part II is the qualification blocker: its uncertainty
quantification is not supported by the stored evidence, and the main reason (an exactly unidentifiable
direction sampled with a tiny pCN step) has an exact, cheap mathematical remedy.

---

## 2. Current repository state

### 2.1 Git

| Item | State |
|---|---|
| Branch | `main` at `cff293a`, 0 ahead / 0 behind `origin/main` |
| Commits on main | 45 |
| Tags / GitHub releases | none / none |
| Uncommitted (user's work, preserved) | `remote/run_deepxde_noise_study.py`: fixes `reset_peak_memory()` and `peak_memory()`, which called themselves (infinite recursion on any CUDA host). `tests/test_deepxde_noise_study.py`: adds a regression test (passes) |
| `origin/audit/scientific-audit` | 3 commits ahead of `aa4b568`, 34 behind main, **unmerged**: `docs/scientific_audit.md`, `HANDOFF_BAYESIAN_VALIDATION.md`, and a "MAP candidate" relabel in `scripts/run_posterior_demo.py`. Its findings are consistent with, and a subset of, this audit |
| `origin/ci/diagnose-sigterm-143`, `origin/ci/scientific-validation` | stale CI branches already superseded on main (PR #2, #3 merged) |
| New, untracked (this audit) | `docs/audit/MILESTONE_1_AUDIT.md`, `docs/audit/audit_stored_chains.py`, `docs/audit/audit_stored_chains_output.txt` |

### 2.2 Components

| Area | Files | Status |
|---|---|---|
| Forward FEM (P2 pressure, P1 log-k, MUMPS LU) | `src/forward.py`, `src/manufactured.py` | complete |
| Observation operator (Gaussian-bump averages) | `src/observation.py` | complete |
| Synthetic data, truth fields | `src/data_generation.py`, `src/truth_fields.py` | complete; single truth draw (`smooth_truth(seed=1)`) |
| Adjoint objective/gradient, regularisers, L-BFGS | `src/adjoint.py`, `src/regularization.py`, `src/inversion.py`, `src/experiment.py` | complete |
| Part I studies | `scripts/run_reconstruction_demo.py`, `regularization_study.py`, `noise_study.py`, `sensor_study.py` | complete; single noise realisation per setting |
| KL prior, pCN, PDE posterior | `src/kl_prior.py`, `src/pcn_sampler.py`, `src/bayesian_darcy.py`, `src/bayesian_experiment.py` | complete; single chain; no likelihood-only path |
| Sampler validation | `src/linear_gaussian_validation.py`, `scripts/validate_sampler.py` | complete |
| Part II studies | `scripts/run_posterior_demo.py`, `bayesian_{sensor,noise}_study.py`, `bayesian_prior_sensitivity.py` | complete but unconverged (Section 5) |
| Multi-chain diagnostics, R-hat, MCSE, coverage study | none | **missing** |
| Optimiser termination diagnostics (status, message, gradient norm) | partial: Part I records `message` and gradient history; Bayesian `compute_map` keeps only `x, nit, success` (`src/bayesian_experiment.py:37-42`) | partial |
| Edge-preserving regularisation (TV etc.) | none | missing |
| Inversion-mesh refinement study | none | missing |
| DeepXDE PINN benchmarks (optional) | `src/deepxde_*.py`, `remote/`, `docs/DEEPXDE_*.md`, `results/deepxde_remote/` | complete through sensitivity study; noise-study runner implemented, not yet run |
| CRLB / FD reference for PINN | `src/fd_reference.py`, `scripts/noise_identifiability_analysis.py`, `results/deepxde_identifiability/analysis.json` | computed, undocumented |
| CI | `.github/workflows/ci.yml` (conda `fenicsx`, pytest) | present; PINN tests skip in CI |

### 2.3 Environments

| Environment | Use | Notes |
|---|---|---|
| conda `fenicsx` (`/opt/anaconda3/envs/fenicsx`) | FEM, Bayesian, tests | Python 3.12.12, dolfinx 0.10.0, PETSc 3.24.3, SciPy 1.18.1, NumPy 2.5.3: **matches** `results/environment_versions.txt`. Shared with `fem-cylinder-flow`. `environment.yml` is unpinned (versions only in comments) |
| `.venv-deepxde` (uv, 839 MB) | PINN CPU tests | torch 2.14.1, DeepXDE 1.15.0. **No `pyproject.toml` or `uv.lock`**, contrary to the project-environment rule |

Neither environment was modified. The base Anaconda environment was not used.

---

## 3. Mathematical audit

| Item | Verdict | Notes |
|---|---|---|
| Governing equation, weak form | correct | -div(e^m grad p) = f, P2 pressure, P1 m. Coercive for bounded m |
| Boundary conditions, source | correct, **with an identifiability consequence** | p=1 left, p=0 right, no-flow top/bottom, f=0. Scaling k by c > 0 leaves p unchanged, so only m modulo constants is identifiable. Verified: misfit(m) and misfit(m+0.7) agree to 3.3e-14 relative. Not stated in README or code |
| Log-k parameterisation | correct | positivity by construction |
| FE discretisation | correct | MMS rates match theory: exact coefficient L2 3.000 / H1 1.999; P1-interpolated coefficient L2 1.996 (coefficient-limited) / H1 1.999 (`results/forward_verification.txt`) |
| Observation operator | correct | normalised bump averages; H and H^T share weight vectors. Bump width (0.035) is below h=1/16, but the resulting model error is at most 0.07 sigma (rms 0.03 sigma) at N=16 (Appendix A.3), so it is negligible |
| Inverse-crime avoidance | correct but weak | truth on N=64, inversion on N=16/32. Discretisation discrepancy is at most 0.07 sigma at N=16, so this protection has little practical effect at 2% noise |
| Objective and regularisation | correct | J = 1/2 misfit + alpha/2 R; gradient term 0.5 alpha dR is the correct derivative. H1 seminorm shares the null space of the misfit (constants), so J has an exact flat direction. L-BFGS starting from m0=0 never moves along it (gradient is orthogonal to it), so the reconstructed constant level equals the initial guess. Both truths have near-zero mean (smooth: excludes the (0,0) mode; structured: nodal mean -0.019), so headline errors are only slightly affected; the metric still mixes identifiable and unidentifiable parts |
| Error metric | acceptable, should be refined | relative Euclidean norm of nodal vectors (not mass-weighted L2), and not mean-removed |
| Adjoint equation, gradient | correct | same operator, homogeneous Dirichlet, RHS -H^T Gamma^-1 (Hp - y); gradient via UFL derivative. Taylor order 2.000. Minor: sum(grad)/norm(grad) = 5.4e-5 instead of 0 in the null direction (quadrature-level inconsistency, immaterial) |
| Gradient metric | standard, mesh-dependent | Euclidean nodal gradient in L-BFGS (no mass-matrix Riesz map). Iteration counts may depend on mesh |
| Prior (KL of A^-2, A = -gamma Lap + delta I, Neumann) | correct | trace-class in 2D. Mode 0 is the constant mode with lambda_0 = 1/delta^2 = 1. Eigenpairs 1/2 and 3/4 are degenerate pairs (basis within each pair is arbitrary) |
| Likelihood | correct | Gaussian, sigma = 2% of signal RMS, nominal floor 1% |
| Posterior, MAP/Tikhonov link | correct in the KL subspace; **one README overstatement** | MAP = Tikhonov with the A^2-weighted norm restricted to 15 modes. README line 741 calls it "the H1-family-equivalent Tikhonov estimate"; README Limitations correctly says the two are not identical. Unlike H1, the prior does penalise the constant mode |
| pCN | correct algorithm | prior-reversible proposal, acceptance on likelihood ratio only; validated on linear-Gaussian target. **Mismatch with this posterior**: in the likelihood-null direction xi_0 the chain is an AR(1) with coefficient sqrt(1-beta^2) per accepted step; at beta=0.02 and acceptance 0.4 the integrated autocorrelation time is about 2/(beta^2 * acc), roughly 12,500 proposals, longer than the whole chain |
| Exact posterior for xi_0 | analytical | likelihood independent of xi_0 and prior factorises, so xi_0 given y ~ N(0,1) exactly, independent of other coordinates. This yields hard lower bounds used in Section 5 |
| ESS estimator | biased upward for slow chains | `src/pcn_sampler.py:124` truncates at lag 1000, capping tau at about 2001 and ESS at about n/2001 (4 for n=8000) |
| Posterior predictive check | **methodologically flawed** | `scripts/run_posterior_demo.py:124-141` builds 95% intervals from only 20 draws (2.5/97.5 percentiles of 20 values are close to min/max) and omits observation noise, then compares to noisy y. Under-coverage is expected by construction |
| PINN formulation (optional) | correct for a manufactured, 2-parameter problem | source uses true parameters only; protocol pre-registered; not comparable to Parts I/II (correctly stated) |

---

## 4. Numerical validation status

Legend: **IC** = implementation correctness evidence, **SP** = scientific performance evidence.

| Check | Type | Evidence | Status |
|---|---|---|---|
| Forward MMS convergence | IC | `results/forward_verification.txt`; `tests/test_forward_convergence.py` | Verified |
| Forward mesh refinement (physical problem) | IC | covered by MMS only | Verified (MMS) |
| Adjoint Taylor test (H1, L2) | IC | `results/gradient_verification.txt`; `tests/test_adjoint.py` | Verified |
| Bayesian xi-gradient Taylor test | IC | `results/bayesian_gradient_verification.txt`; `tests/test_bayesian_darcy.py` | Verified |
| Optimiser convergence, Part I headline | SP | `results/reconstruction_demo_full.txt` (converged, 138/125 it) | Verified |
| Optimiser convergence, Part I studies | SP | `results/regularization_study_full.txt` (9/9 `False` at 120), `noise_study_full.txt` (5/5 at 120), sensor study (not recorded) | **Failed / not recorded** |
| Optimiser convergence, Bayesian MAP | SP | `results/posterior_demo_full.txt` (`False`, 120) | **Failed**; README claim of a 2,000-iteration rerun (line 595) has no stored artifact |
| Smooth reconstruction accuracy | SP | 0.3489, single noise realisation, single truth | Partially verified |
| Structured reconstruction | SP | 0.8488; alpha sweep 0.82 to 0.89 claimed (README) | Partially verified; **sweep has no artifact** |
| Regularisation sensitivity / discrepancy principle | SP | U-curve and alpha=30 selection on unconverged iterates; alpha=100 misfit 19.18 vs target 18 | Partially verified |
| Inversion-mesh refinement | SP | none | Missing |
| Sampler correctness (linear-Gaussian) | IC | `results/sampler_validation.txt`: mean rel. err 0.59%, ESS 179 to 861 | Verified (well-conditioned target only; no null direction) |
| ESS | SP | repo: 4.9 to 86.1; untruncated: median 6.6 (smooth) / 5.2 (structured) | Verified as low; repo values slightly optimistic |
| Autocorrelation / mixing | SP | xi_0 ACF 0.49 to 0.74 at lag 500 (full) | Failed |
| R-hat | SP | none in repo; this audit: split-R-hat (4 segments, single chain) up to 3.38, 15/15 dims > 1.01 | **Failed** (proper multi-chain R-hat missing) |
| Credible intervals | SP | 5/5 coverage, one realisation | **Invalid**: intervals narrower than the exact lower bound |
| Coverage / calibration | SP | none | Missing |
| Posterior predictive | SP | 76% | Invalid construction |
| Multi-seed reproducibility | IC/SP | bit-exact seed reproducibility tests pass; no multi-realisation statistics for any result | Partial (implementation only) |
| PINN recovery (optional) | SP | 5 seeds, SHA-256 manifests, pre-registered | Verified for noise-free data |

---

## 5. Verified experimental results

### 5.1 Historical claims

| Claim | Source | Verified value | Verdict |
|---|---|---|---|
| Smooth reconstruction error about 34.9% | `results/reconstruction_demo_full.txt:3` | 0.3489, converged, 138 it | **Verified (traceable)**; single realisation |
| Structured reconstruction error about 84.9% | `results/reconstruction_demo_full.txt:4` | 0.8488, converged, 125 it | **Verified (traceable)**; single realisation |
| Bayesian ESS about 8 to 9 from 8,000 | `results/posterior_demo_full.txt:8,26` | repo median 7.7 / 7.0; untruncated median 6.6 / 5.2 | **Verified**, true values slightly lower |
| Non-converged MAP | `results/posterior_demo_full.txt:6,24` | `converged=False`, 120 it | **Verified** |
| Independent linear-Gaussian validation | `results/sampler_validation.txt` | mean rel. err 5.9e-3; var MAE 5.9e-4 | **Verified** (traceable; script reproducible) |
| 59 passing tests | none found | 56 (pre-PINN, reproduced at `b7a190e`), 103 (current) | **Not supported** |
| Split-chain mean difference up to 0.70 | README:572 | 0.702 (smooth), 0.760 (structured) | Verified |

### 5.2 New findings from stored chains (read-only analysis)

| Quantity | Smooth full | Structured full | Exact-posterior requirement |
|---|---:|---:|---|
| sd of xi_0 (constant mode) | 0.353 | 0.515 | 1 |
| mean of xi_0 | +0.066 | +0.375 | 0 (sd of estimate should be small) |
| max pointwise posterior sd of m | 0.675 | 0.800 | >= 1 everywhere |
| integrated posterior variance (reported) | 0.295 | 0.367 | >= 1 |
| split-R-hat max (4 segments) | 3.38 | 3.01 | < 1.01 |
| posterior-mean rel. error, raw / mean-removed | 1.001 / 0.996 | 1.466 / 1.072 | n/a |
| MAP-candidate rel. error, raw / mean-removed | 0.387 / 0.387 | 0.807 / 0.807 | n/a |

The smooth-truth posterior-mean error of about 1.0 is not caused by the constant mode (mean-removed 0.996); it reflects
non-convergence in other coordinates (xi_1, xi_2 split-R-hat 3.3). The posterior mean is as far from the truth as the zero field.

### 5.3 Confounding in Bayesian trend studies

| Study | integrated variance | acceptance |
|---|---|---|
| sensors 9 / 25 / 49 | 0.260 / 0.234 / 0.161 | 0.661 / 0.401 / 0.237 |
| noise 1% / 5% / 10% | 0.134 / 0.226 / 0.401 | 0.158 / 0.711 / 0.853 |
| same setting, data seeds 7 / 200 / 400 | 0.295 / 0.234 / 0.188 | 0.400 / 0.401 / n.r. |

Every reported value is below the exact bound of 1. The trends are monotone in acceptance rate, so the README's
"physically expected" reading (README lines 606, 678) is not separable from sampler behaviour.

### 5.4 Newer results (optional PINN track)

Verified with manifests in `results/deepxde_remote/` and documented in `docs/DEEPXDE_REMOTE_RESULTS.md`:
spatial 2-parameter k recovered by Adam 50k + L-BFGS on 5/5 seeds (max rel. err 5.1e-3), robust to 16 to 256
noise-free sensors, initialisation and layout; not recovered with noise 0.01 / 0.05. The committed CRLB analysis
shows an efficient estimator needs about 615 sensors (sigma=0.01) for 1% rms error, so the noisy "not recovered"
rows reflect an information limit, not a PINN deficiency (PINN vs MLE difference about 1e-3).

---

## 6. Test execution results

| Environment | Command (no `uv --active`, base env untouched) | Result | Time |
|---|---|---|---|
| conda `fenicsx` | `OMP_NUM_THREADS=1 /opt/anaconda3/envs/fenicsx/bin/python -m pytest -p no:cacheprovider -q tests` | **88 passed, 15 skipped** (torch absent) | 33 s |
| `.venv-deepxde` | `DDE_BACKEND=pytorch .venv-deepxde/bin/python -m pytest -p no:cacheprovider -q tests/test_deepxde_*.py` | **47 passed** (includes the user's uncommitted regression test) | 6 s |
| historical `b7a190e` (export in scratch dir) | same as first row | **56 passed** | 24 s |

Union: 103 distinct tests, all pass. Coverage gaps: no test for constant-shift invariance, no test that ESS is
unbiased for a known AR(1) process with tau > 1000, no test of R-hat or coverage machinery, no test that the
sampler recovers a posterior with a likelihood-null direction.

---

## 7. Scientific limitations

1. Exact non-identifiability of the mean log-permeability under the current boundary conditions and f=0.
2. Single synthetic realisation (one truth draw, one noise draw) for every reported number.
3. Unconverged optimisation in Part I studies and Bayesian MAP.
4. Bayesian posterior summaries invalid (Section 5.2); no multi-chain diagnostics; no calibration study.
5. Smoothness-based regulariser and prior cannot represent the structured inclusions (genuine, well-documented negative result).
6. No inversion-mesh refinement study; the gradient metric is mesh-dependent.
7. Posterior predictive check excludes noise and uses 20 draws.
8. KL truncation r=15 makes the problem 15-dimensional; "function-space" pCN advantages do not apply at this size.
9. Documentation: README test count (line 916: "56 tests") outdated; README line 659 states the wrong direction for interval bias; README line 626 compares MAP (N=16, 25 sensors, KL prior) with Part I (N=32, 36 sensors, H1) as "comparable settings"; README contains development-status language ("Working status", "pending") contrary to the public-README policy.

---

## 8. Qualification blockers (Strong to Discuss)

| # | Blocker | Severity | Fix type |
|---|---|---|---|
| B1 | Bayesian UQ violates exact analytical bounds; no converged chain | critical | method + rerun |
| B2 | Unrecognised constant-mode non-identifiability (theory, metric, sampler, docs) | critical | analysis + exact treatment |
| B3 | Unconverged optimisers in Part I studies and MAP; termination diagnostics not stored | high | diagnostics + rerun |
| B4 | No multi-seed / multi-realisation statistics anywhere | high | rerun |
| B5 | No posterior calibration (coverage / SBC); predictive check flawed | high | new experiment |
| B6 | Unsupported or untraceable claims (59 tests, 2,000-iteration MAP, structured alpha sweep, interval-bias direction, trend robustness) | medium | provenance + docs |
| B7 | No inversion-mesh refinement study | medium | rerun |
| B8 | Environment reproducibility (unpinned conda spec, no uv lockfile for `.venv-deepxde`) | medium | packaging |
| B9 | Unmerged audit branch, uncommitted bug fix, undocumented CRLB analysis | low | housekeeping (needs approval) |

The structured-field failure is **not** a blocker in itself: a carefully characterised negative result is acceptable.
It becomes one only if presented without a mean-removed, multi-seed characterisation (or an edge-preserving comparison).

---

## 9. Prioritised improvement roadmap

Runtime basis (measured today, M3 Pro, one thread, other sessions active, load average about 6.5 on 11 cores):
forward solve 12 ms (N=16), 13 ms (N=32), 84 ms (N=64); objective+gradient 20 / 31 / 175 ms; one process
about 190 MB peak RSS. A likelihood-only pCN step is about 13 ms at N=16 (current code uses about 20 ms because it
also solves the adjoint).

**Parallelisation.** Chains are sequential, so parallelise across independent chains, seeds and replicates
with separate processes (`concurrent.futures.ProcessPoolExecutor`, spawn start method, or independent shell
processes), each with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Each process gets its own size-1
`MPI.COMM_WORLD`. Meshes are far too small for MPI domain decomposition to help. Use at most 4 workers while
other sessions are active (5 P-cores, 6 E-cores). Memory is not a constraint (4 workers < 1 GB). Store chains in
xi-space only (15 x 100k x 8 B = 12 MB per chain), never the 289-dof field samples (230 MB per chain); disk has
12 GiB free at 98% usage.

**External compute.** Not needed for Parts I and II (CPU-bound, small meshes). Kaggle remains the venue for the
optional PINN track only.

### M2. Baseline reproduction and instrumentation

- Tasks: record SciPy `status`, `message`, `nit`, `nfev`, final objective, final gradient 2-norm and inf-norm for
  every optimisation (no objective change); add a likelihood-only path for pCN (removes the adjoint solve; same
  likelihood value, checked by test); add tests for constant-shift invariance and for ESS against an AR(1)
  process with known tau; rerun all verification scripts and headline demos into new versioned result files
  (historical files stay untouched); decide the fate of the audit branch and the uncommitted fix.
- Acceptance: verification outputs reproduce committed values to printed precision (MMS orders, Taylor orders,
  0.3489 / 0.8488, sampler-validation table); 103/103 tests pass plus new ones; likelihood-only Phi equals the
  current Phi to 1e-12 relative.
- Dependencies: approval for the code changes and branch handling.
- Runtime: about 30 min CPU.
- Deliverables: instrumented code, regression tests, `results/v2/` baseline set, provenance manifest (commit, config, seeds, environment).

### M3. Deterministic inversion validation

- Tasks: rerun the regularisation, noise and sensor studies to documented convergence (raise `max_iter`; report termination
  reason; no tolerance changes without justification); report raw and mean-removed L2 (mass-weighted) error; 20 noise
  realisations x 2 truths at the headline setting, plus 5 additional smooth-truth draws; inversion mesh N = 16, 24, 32, 48, 64
  at fixed data; recompute the discrepancy principle on converged runs with a finer alpha grid; H1 vs L2 regulariser comparison.
- Acceptance (pre-register before running): every run converged or explicitly reported otherwise; mean, sd and 95% CI for
  each headline error; change in mean error between N=32 and N=64 below 0.01 or reported as a mesh effect; discrepancy
  choice reported with its seed-to-seed variability.
- Runtime: about 300 reconstructions at 6 to 40 s each: about 1 h CPU, about 20 min wall with 4 workers.
- Deliverables: `results/v2/deterministic/*`, figures with error bars.

### M4. Structured permeability reconstruction

- Tasks: reproduce and store the structured alpha sweep (currently untraceable); multi-seed characterisation of the H1
  failure; optionally (method change, needs approval) smoothed total-variation regularisation with a documented epsilon,
  verified by a Taylor test before use.
- Acceptance (pre-registered): either TV reduces mean-removed structured error by a CI-separated margin over the best H1
  alpha across at least 10 noise seeds, or the negative result is reported with the same statistics.
- Runtime: about 1 h CPU (TV may need 300 to 1,000 iterations per run).
- Deliverables: sweep artifact, TV module and tests (if approved), comparison table.

### M5. Bayesian sampling diagnostics (characterise before fixing)

- Tasks: implement multi-chain rank-normalised split-R-hat, bulk and tail ESS, MCSE (Vehtari et al. 2021), tested on iid
  and AR(1) sequences; run the **unchanged** pCN (likelihood-only) with 4 chains started from dispersed prior draws, 100k
  proposals each, both truths; check xi_0 against its exact N(0,1) posterior; compute the 15 x 15 Gauss-Newton Hessian at
  the MAP (about 15 extra gradient evaluations) to identify data-informed and prior-dominated directions; beta pilot study
  reported as ESS per PDE solve.
- Acceptance: diagnostics pass their unit tests (ESS within 10% of analytical value for AR(1) with known tau); a baseline
  report that states, per coordinate, R-hat, ESS, MCSE and the xi_0 check.
- Runtime: 8 chains x 100k x 13 ms = about 3 h CPU, about 45 to 60 min wall with 4 workers.
- Deliverables: `src/mcmc_diagnostics.py` and tests, baseline diagnostic report.

### M6. Sampling efficiency improvements (method change, approval required)

- Candidates, in order of mathematical safety: (i) exact treatment of xi_0 (Gibbs draw from N(0,1), or remove it from
  the sampled coordinates and add its exact contribution analytically); (ii) Hessian-preconditioned pCN or a
  Laplace-based independence or generalised-pCN proposal using the M5 Hessian; (iii) MALA or HMC in 15 dimensions using the
  existing adjoint gradient. Keep plain pCN as the validated reference.
- Acceptance: across 4 chains, rank-normalised split-R-hat < 1.01 and bulk ESS >= 400 for all 15 coordinates and for
  m at 5 probe points; MCSE of posterior means < 0.05 posterior sd; xi_0 mean and variance consistent with N(0,1) within
  MCSE; pointwise posterior variance >= 1 everywhere; sampler re-validated on an extended linear-Gaussian problem that
  includes a likelihood-null direction; efficiency reported as ESS per PDE solve relative to pCN.
- Runtime target: at most 1 to 2 h wall per truth on 4 workers.
- Deliverables: sampler module, validation results, efficiency comparison.

### M7. Posterior uncertainty validation

- Tasks: (a) well-specified calibration: draw truths from the prior, generate data, check coverage of 50/90/95% intervals
  for KL coefficients and for spatial averages over subregions, plus simulation-based-calibration rank histograms;
  (b) Laplace-approximation coverage first (cheap), then MCMC on a subset; (c) posterior predictive with noise added and
  at least 500 draws; (d) misspecified truths (smooth, structured) reported separately.
- Acceptance (pre-registered): with R = 50 replicates, empirical 90% coverage inside the binomial 95% band [0.80, 0.98];
  SBC ranks not rejected by a chi-square uniformity test at the 1% level; predictive coverage computed with noise.
- Runtime: Laplace stage minutes. MCMC stage 50 replicates x 4 chains x about 4 min (if M6 reaches ESS 400 in 20k
  proposals) is about 14 h CPU or about 4 h wall. If resources are insufficient, leave PENDING with the exact command.
- Deliverables: calibration report and figures.

### M8. Robustness experiments

- Tasks: noise (0.5 to 10%), sensors (9/25/49), prior hyperparameters (gamma, delta) including misspecification, KL truncation
  (6/15/25), inversion mesh (16/24/32), bump width; at least 10 realisations for deterministic and 5 for Bayesian settings.
- Acceptance: a trend is claimed only when CI-separated; Bayesian variances recomputed with converged chains and exact
  xi_0 treatment.
- Runtime: deterministic about 1 h; Bayesian about 6 to 10 h wall (candidate for PENDING).
- Deliverables: robustness tables and figures.

### M9. Scientific documentation

- Tasks: README update via the `github-readme-standardizer` skill (remove development-status language, no em dashes);
  document the identifiability result, corrected UQ and calibration; move superseded numbers to a clearly labelled
  historical section with provenance; document the PINN CRLB analysis; pin the conda environment (explicit spec or
  conda-lock) and add `pyproject.toml` plus `uv.lock` for the PINN environment.
- Acceptance: every README number maps to a committed result file, config and commit; CI green.

### M10. Final qualification audit

- Fresh-context audit against the acceptance criteria above; upgrade to Strong to Discuss only if B1 to B8 are closed.

### Optional PINN track

Run the existing noise-study runner (after committing the recursion fix) on Kaggle. Any change of the recovery
criterion to a CRLB-relative one must be recorded as a pre-registered deviation and is the user's decision.

---

## 10. Acceptance criteria summary for Strong to Discuss

1. All verification evidence reproducible from a pinned environment, tests green in CI.
2. Every reported optimisation converged (or the termination reason is reported and justified).
3. Every headline number reported with multi-realisation uncertainty.
4. Identifiability of the constant mode stated, handled exactly, and reflected in the error metric.
5. Bayesian chains: multi-chain R-hat < 1.01, ESS >= 400, MCSE reported, exact-bound checks pass.
6. Posterior calibration demonstrated in the well-specified case; misspecified cases reported honestly.
7. Every README claim traceable to a stored artifact; no unsupported claims.

---

## Appendix A. Reproduction of audit checks

A.1 Stored-chain diagnostics:
`/opt/anaconda3/envs/fenicsx/bin/python -I docs/audit/audit_stored_chains.py .` (output in `docs/audit/audit_stored_chains_output.txt`).

A.2 Constant-shift invariance and timing (inline, about 1 min): build `InverseProblem` on N=16 with alpha=0,
evaluate the misfit at m and m+0.7 (relative difference 3.3e-14); time 20 forward and 20 objective+gradient
evaluations at N=16, 32, 64.

A.3 Observation-operator model error: max |H_N p_N - y_clean(N=64)| / sigma = 0.071 (N=16), 0.014 (N=32), 0.004 (N=128).

A.4 Historical test count: `git archive b7a190e` into a scratch directory, then pytest: 56 passed.
