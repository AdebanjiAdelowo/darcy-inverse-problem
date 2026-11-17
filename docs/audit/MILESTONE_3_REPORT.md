# Milestone 3: Optimisation Convergence and Bayesian Posterior Validity

Date: 2026-11-17
Base: `main` at `0019d11`. Integrated in Milestone 4, Phase 0 (commits listed in `docs/audit/MILESTONE_4_REPORT.md`). Chain archives are kept outside Git and listed with SHA-256 checksums in `results/m3/mcmc/MANIFEST.json`.
Qualification: **Qualified, Not Strong to Discuss** (unchanged).
Milestone verdict: **CONDITIONAL PASS** (Section 9).

Pre-registration: `docs/audit/M3_PREREGISTRATION.md` (criteria, selection rules and one recorded deviation).
All new runs used `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, at most four concurrent processes and fixed
seeds, and wrote only under `results/m3/`. No historical file in `results/`, `figures/` or `configs/` changed.

---

## 0. Milestone 2 integration

| Step | Result |
|---|---|
| Review, tests (both environments), historical results unchanged | 122 passed + 15 skipped; 49 passed; `git diff` on `results/ figures/ configs/` empty |
| Commits | `4f11d73` feat (diagnostics, forward-only likelihood, identifiability), `8f87625` feat (reproduction tools), `26c6cfd` test, `2e32981` build (environment specs), `511182b` docs, `0019d11` results |
| Push | `cff293a..0019d11` to `origin/main`; Scientific CI run 43970331157: success (1 m 29 s) |
| Draft PR #1 | closed with an explanation and links to the Milestone 1/2 audits and the branch review; branch `audit/scientific-audit` kept (`9d787f8`) |

---

## 1. Summary

| Question | Answer |
|---|---|
| Treatment of the constant mode | Exact factorisation: xi_c given y ~ N(0,1), independent of the mean-free coefficients. Sample only the 14 mean-free coefficients; add the constant mode analytically or by independent N(0,1) draws. No information added to the likelihood. |
| Why the stored variances violate the floor | xi_0 performs an AR(1) walk on accepted pCN steps only, with integrated autocorrelation time about 23,000 to 25,000 proposals, roughly three times the chain length. The observed xi_0 variances (0.125, 0.265) sit at the 48th and 79th percentiles of their exactly simulated distribution; P(variance >= 1) < 0.2%. |
| Corrected variance | Respects the bound by construction: pointwise total variance 1.11 to 1.94 (smooth) and 1.20 to 2.90 (structured), integrated 1.361 and 1.565 (exact constant part 1). The mean-free part is **not converged**. |
| Multi-chain diagnostics | Rank-normalised split R-hat, bulk and tail ESS, and MCSE implemented and validated on cases with known answers; 4 chains per truth from dispersed starts |
| MCMC convergence | **Not achieved.** Max R-hat 1.83 (smooth), 1.65 (structured); min bulk ESS 6 to 7 after 100,000 proposals per chain. Plain pCN would need about 27 h per chain (Section 5.4). |
| Deterministic optimiser | Regularisation-preconditioned L-BFGS-B at the **historical budgets**: 44/46 runs meet the pre-registered first-order criteria vs 2/46 historically; median 31.5 vs 120 iterations; 3.1x fewer PDE solves |
| MAP | Gauss-Newton-preconditioned L-BFGS-B in the mean-free KL space: both MAPs stop on the gradient test (64 and 33 iterations, budget 120) and meet C1 to C3; the historical runs met none of them |

---

## 2. Constant mode (A)

### 2.1 Mathematics (`src/constant_mode.py`)

The prior is xi ~ N(0, I_15) on KL coefficients; mode c = 0 is the constant phi_c = 1 with lambda_c = delta^-2 = 1.
The likelihood depends only on the remaining coefficients xi_R (pressure invariance, `tests/test_identifiability.py`),
so the posterior factorises exactly:

    pi(xi | y) = [L(xi_R) N(xi_R; 0, I_14) / Z]  x  N(xi_c; 0, 1).

The non-constant KL modes are L2-orthogonal to constants, so their span is exactly the mean-free (identifiable)
subspace (`test_non_constant_modes_span_the_mean_free_subspace`: their means are below 1e-12).

### 2.2 The three approaches compared

| Approach | What is sampled | Constant-mode contribution | Monte Carlo error in that contribution | Result here |
|---|---|---|---|---|
| Analytical marginalisation | xi_R only (pCN on 14 dims) | exact moments: Var adds lambda_c phi_c^2 = 1; E[k] gains factor exp(lambda_c phi_c^2 / 2); quantiles by Rao-Blackwellisation | none | used for the reported variance and intervals |
| Mean-free subspace sampling | xi_R only (the **same chain**) | none: reports the identifiable field m_R and contrast exp(m_R) | n/a | identical chain to the row above; differs only in which quantity is reported |
| Independent reconstruction of the constant mode | xi_R, plus xi_c ~ N(0,1) i.i.d. per draw | sampled | yes | variance within 2.0% of analytic; 95% intervals within 0.04; xi_c mean 0.000, variance 1.010, KS p = 0.31 |
| (Historical joint pCN, reference) | xi_c inside pCN | sampled by a random walk with tau ~ 2.4e4 | very large | violates the floor |

Because the KL basis is orthonormal, approaches 1 and 2 are the same sampler. The distinction between the
**identifiable structure** (m_R, contrast) and the **unidentifiable global scale** (xi_c, whose posterior equals
its prior) is kept explicit in every reported quantity. The independent draws used one fixed seed (2026) for
both truths, which is why the KS p-values coincide.

---

## 3. Posterior variance (B)

### 3.1 Why the stored estimates violate the floor (`results/m3/constant_mode_analysis.json`)

In the historical sampler, xi_0' = sqrt(1-beta^2) xi_0 + beta eta_0 never enters the acceptance ratio, and eta_0 is
independent of the accept/reject decision. So xi_0 is an AR(1) process on accepted steps, conditionally on the
accept/reject sequence, which is recoverable from the stored chains. Simulating 4,000 replicates with that sequence:

| Truth | observed var(xi_0) | simulated median [5%, 95%] | P(var >= 1) | tau(xi_0), proposals | chain length / tau |
|---|---:|---|---:|---:|---:|
| smooth | 0.125 | 0.129 [0.043, 0.435] | 0.0008 | 24,957 | 0.32 |
| structured | 0.265 | 0.146 [0.047, 0.479] | 0.0020 | 22,665 | 0.35 |

The stored integrated variance decomposes into a constant-mode part of 0.125 / 0.265 (exact value 1) plus a
mean-free part of 0.168 / 0.100. The floor violation is fully explained by sampler inefficiency in a direction the
data cannot inform. It is not a coding error.

### 3.2 Corrected values (bounded production run, Section 5)

| Truth | integrated variance: mean-free (MC) + constant (exact) = total | pointwise total, min to max | floor violated |
|---|---|---|---|
| smooth | 0.361 + 1 = **1.361** | 1.112 to 1.944 | no |
| structured | 0.565 + 1 = **1.565** | 1.202 to 2.904 | no |

Absolute 95% credible intervals at the probe points are 4.18 to 4.60 wide (exact minimum 3.92; historical about
1.9). The mean-free part comes from unconverged chains (MCSE of the probe means 0.15 to 0.27), so these totals are not
validated estimates; only the constant-mode contribution (exactly 1) and the bound are certain.

### 3.3 Regression tests (`tests/test_constant_mode.py`, 7 tests)

Mean-free span; reduced likelihood equal to the full likelihood for any xi_c (to 1e-12); marginalised variance
>= floor **even for a chain that never moves** (the bound holds by construction); independent draws match N(0,1)
(KS) and the analytic variance; Rao-Blackwell quantiles exact in limiting cases; log-normal factor; reduced pCN
runs in 14 dimensions.

---

## 4. Convergence diagnostics (C)

`src/mcmc_diagnostics.py` (NumPy/SciPy only; ArviZ is not installed and was not installed) implements, following
Vehtari et al. (2021) and the Stan/ArviZ estimators: split chains, rank normalisation, rank-normalised split R-hat
(max of bulk and folded), bulk ESS, tail ESS (5% and 95% indicators), ESS of the mean and sd, MCSE of the mean,
sd and quantiles, FFT autocorrelation, and a per-quantity summary with the thresholds R-hat < 1.01 and bulk and tail ESS >= 400.

Validation (`tests/test_mcmc_diagnostics.py`, 13 tests): i.i.d. draws (R-hat < 1.01, ESS within 15% of N); AR(1)
with rho = 0.5, 0.9, 0.98 (ESS within 10% of N(1-rho)/(1+rho)); tau of about 1e4 resolved (the historical estimator's
lag-1000 floor removed); detection of location shift, scale disagreement (folded R-hat), and within-chain trend
(split R-hat); invariance under monotone transforms; MCSE of the mean matches the empirical spread of AR(1) means
over 150 replications (within 15%). The historical estimator is kept for reproducibility.

---

## 5. MCMC experiments (C, E)

### 5.1 Setup

Historical full configuration (N=16, r=15, 25 sensors, 2% noise, data seed 7). pCN acts on the 14 mean-free
coefficients with the forward-only likelihood; the kernel is unchanged. MAP from Section 6.2.

### 5.2 Pilot (`results/m3/mcmc/pilot_summary.json`)

Single chains of 8,000 proposals per beta from the MAP, plus one dispersed start per truth (about 20 min of CPU in total, 354 s wall).

| Truth | beta | acceptance | min bulk ESS per 1,000 proposals |
|---|---:|---:|---:|
| smooth | 0.01 / 0.02 / 0.04 / 0.08 | 0.64 / 0.39 / 0.16 / 0.03 | 0.29 / 0.29 / 0.18 / 0.22 |
| structured | 0.01 / 0.02 / 0.04 / 0.08 | 0.66 / 0.42 / 0.18 / 0.05 | 0.17 / 0.25 / 0.24 / 0.20 |

Selected by the pre-registered rule: beta = 0.01 (smooth), 0.02 (structured). The pilot ESS values rest on about 2
effective draws and do not separate the betas reliably. Dispersed starts reach the typical Phi range within the
pilot (Phi 124 -> 8.4, 95 -> 11.9).

### 5.3 Bounded production run (Deviation 1)

4 chains x 100,000 proposals per truth, dispersed starts (xi_R,MAP + N(0,I)), seeds 1 to 4, thin 10, first half
discarded. `results/m3/mcmc/production_summary.json`, figures `results/m3/figures/mcmc_traces_{smooth,structured}.png`.

| Truth | acceptance (4 chains) | ms per proposal | forward / adjoint solves per chain | h per chain | max R-hat | min bulk ESS | min tail ESS | monitored quantities converged |
|---|---|---:|---|---:|---:|---:|---:|---:|
| smooth (beta 0.01) | 0.641 to 0.654 | 14.2 to 14.3 | 100,010 / 0 | 0.40 | 1.825 | 5.9 | 11.8 | 0 of 20 |
| structured (beta 0.02) | 0.437 to 0.448 | 14.6 to 14.8 | 100,010 / 0 | 0.41 | 1.649 | 6.7 | 13.4 | 0 of 20 |

Phi alone is close to the target (R-hat 1.013 and 1.016, bulk ESS 348 and 468), while the coefficients are far
from it. The likelihood value can look well mixed while the parameters are not.

**Diagnosis.** The slowest quantities are the members of degenerate KL pairs (modes 1/2, 4/5, 9/10) and the probe
values built from them. In the smooth run, chains 1 and 2 sit at (xi_1, xi_2) ~ (0.30, 0.57) and (0.49, 0.35), and chains 3 and 4 at about
(1.08, -0.32) and (1.09, -0.39). The sum xi_1 + xi_2 is similar across chains (about 0.7 to 0.9), while the
difference spans more than 1.4. The posterior is a long, narrow ridge: one combination (the dominant flow-direction mode) is
pinned by the data, and the orthogonal one (variation across the flow) is barely informed. The Gauss-Newton Hessian in KL
coordinates has a condition number of 1.2e4 (smooth) and 7.8e3 (structured). pCN's isotropic proposal must use a step
small enough for the narrow direction and so random-walks along the long one: autocorrelations are still 0.3 to 0.5 at lag
4,000.

### 5.4 Computational budget for convergence with plain pCN

Pooled bulk ESS of about 6 from 4 x 50,000 post-warmup proposals implies tau of about 33,000 proposals. The 8,000-proposal pilot
under-estimated tau about sevenfold, so the budget recorded in Deviation 1 (1.05 to 1.2 million proposals per chain) was itself too
optimistic. Reaching bulk ESS 400 would need about 1.3e7 post-warmup proposals, i.e. about 6.7 million per chain with warmup, or
about **27 h per chain** at 14.5 ms, about 27 h wall per truth on 4 local processes. This is not attempted and is not recommended (Section 8).
PENDING command, if nevertheless wanted:

```bash
for t in "smooth 0.01" "structured 0.02"; do set -- $t; for s in 1 2 3 4; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/m3_mcmc.py chain --truth $1 --beta $2 \
    --n-proposals 6700000 --seed $s --init dispersed --thin 100 --out results/m3/mcmc/long/$1_s$s.npz & done; wait; done
```

---

## 6. Optimiser convergence (D)

### 6.1 Deterministic inversion (`results/m3/optimizer/summary.{json,csv}`, `results/m3/conditioning.json`)

**Assessment.**

- *Gradient measures.* The nodal Euclidean gradient is a mesh-weighted quantity (entries scale like h^2), so the
  historical gtol = 1e-6 on its inf-norm is not a mesh-independent test. The dual norm ||g||_{P^-1} with
  P = alpha K + alpha M is mesh-independent.
- *Scaling and conditioning.* The Gauss-Newton Hessian on the mean-free subspace at the headline minimisers has
  condition number about 905 in nodal coordinates, essentially that of alpha K alone (910), i.e. O(h^-2). In
  P-coordinates it is 88 to 108, with only 5 to 6 eigenvalues outside [0.5, 2] (`results/m3/figures/hessian_spectra.png`).
  The misfit adds a rank-36 perturbation.
- *Budgets and termination.* With E-2000, Euclidean L-BFGS-B needs a median of 148.5 iterations, so a cap of 120 is
  simply too small for it. 44 of its 46 stops use SciPy's relative-function-reduction test, which in 4 cases fires before the
  dual gradient has fallen by 1e-6.
- *Remedy.* Reparameterise as m = L^-T u with P = L L^T, keeping the objective, regulariser, initial guess and SciPy
  options (gtol 1e-6, ftol 1e-12) and the historical budgets. The L2 mean of m is preserved (up to the
  discrete gradient's 5e-5 relative inconsistency in the constant direction), so the unidentifiable level stays
  fixed by the initial guess.

**Results (46 configurations; criteria pre-registered: C1 SciPy convergence test, C2 dual-gradient reduction
<= 1e-6, C3 Gill-Murray-Wright gradient test with tau_F = ftol):**

| Variant | budget | first-order converged (C1 and C2 and C3) | iteration-cap stops | C0 (historical nodal gtol) | median iterations | function evaluations | optimiser time |
|---|---|---:|---:|---:|---:|---:|---:|
| E-hist | historical | 2 | 44 | 0 | 120 | 5,832 | 249 s |
| **P** | **historical** | **44** | 2 | 7 | **31.5** | **1,857** | **80 s** |
| E-2000 | 2000 | 42 | 0 | 2 | 148.5 | 7,324 | 314 s |

- The two P failures are the weakest regularisations (alpha = 0.1 and 1; 120-iteration cap). There the data misfit dominates
  the Hessian and a regularisation-based preconditioner helps least. alpha = 1 meets C2 but not C1 or C3; alpha = 0.1 reaches
  only a 6.5e-5 reduction.
- Where P and E-2000 both converge (40 configurations), objectives agree to 1.2e-11 relative and mean-free L2 errors
  to 3.4e-6: the same minimiser. The headline mean-free L2 errors are unchanged (smooth 0.3468, structured 0.8417); the historical absolute nodal errors move from 0.3489 to 0.3488 (smooth) only because P fixes the unidentifiable level by the L2 mean instead of the nodal sum.
- The γ sensitivity pilot (γ = α/10, α, 10α; headline configuration) gave mean-free minimisers agreeing to 4e-7.
- C0 is met only by runs that stop on the gradient test (7 of 46 under P). Relative-reduction stops occur with nodal
  gradient inf-norms of a few 1e-6. A pilot-stage conjecture that these stops reflect the floating-point resolution of J
  is **not supported**: only 1 of 28 such stops has an estimated remaining decrease below ftol·|J|. They reflect SciPy's
  heuristic relative-reduction test. Satisfying C0 would require changing ftol, which was not done.

### 6.2 MAP (`results/m3/optimizer/map.json`)

| Truth | Variant | termination | iterations | max abs gradient | dual-gradient reduction | C1 / C2 / C3 | xi_R vs P-MAP |
|---|---|---|---:|---:|---:|---|---:|
| smooth | E-hist (120) | iteration limit | 120 | 3.6e-4 | 1.6e-6 | no / no / yes | 3.1e-5 |
| smooth | E-2000 | relative reduction | 162 | 8.5e-5 | 1.9e-7 | yes / yes / yes | 2.2e-6 |
| smooth | **P-MAP (120)** | **gradient tolerance** | **64** | 2.0e-5 | 1.9e-8 | **yes / yes / yes** | |
| structured | E-hist (120) | iteration limit | 120 | 8.1e-3 | 6.2e-5 | no / no / no | 2.1e-4 |
| structured | E-2000 | relative reduction | 167 | 1.1e-4 | 1.4e-6 | yes / no / yes | 8.2e-6 |
| structured | **P-MAP (120)** | **gradient tolerance** | **33** | 1.8e-5 | 8.4e-8 | **yes / yes / yes** | |

P-MAP works in the 14 mean-free coefficients (xi_c = 0 is the exact MAP of the constant mode), with L-BFGS-B in the
coordinates of P = I + G^T Gamma^-1 G (Gauss-Newton at the prior mean; condition number 1.2e4 and 7.8e3). The P-MAPs can now be called
MAP estimates under the pre-registered criteria; the historical endpoints remain MAP candidates.

---

## 7. Tests and costs

| Environment | Result |
|---|---|
| conda `fenicsx` | **148 passed, 15 skipped** (PINN tests; torch absent), 28 s |
| `.venv-deepxde` | **62 passed** (47 PINN + ESS-bias + 13 diagnostics), 8 s |

New tests (26): `test_mcmc_diagnostics.py` (13), `test_constant_mode.py` (7), `test_preconditioned_inversion.py` (6:
factor, transformed-gradient Taylor test, same minimiser as Euclidean, L2-mean preservation, observation Jacobian vs
finite differences, GN-preconditioned MAP vs long Euclidean MAP). Two test fixtures were corrected during
development: random, unrealisable data with a weak regulariser drove the fields to +-7, a nonconvex regime unsuitable
as a reference, so the fixture now uses data generated from a smooth field. One docstring claim (L2-mean
preservation "exact up to roundoff") was corrected to the measured 5e-5-limited level.

Measured costs (single thread, 4 concurrent processes; load from other sessions 6 to 20):

| Item | Cost |
|---|---|
| pCN proposal, forward-only (N=16) | 14.2 to 14.8 ms in production (12.5 ms unloaded, M2) |
| Bounded production, 8 chains x 100,000 | 3.2 h CPU (0.40 to 0.41 h per chain), about 50 min wall |
| Optimiser study, 138 deterministic runs + MAP study | about 11 min of CPU in the optimisers; about 4 min wall |
| Whole Milestone 3 queue (21 jobs) | 52 min wall |
| Gauss-Newton Hessian (KL, 25 sensors) | 0.6 to 0.7 s; nodal (36 sensors, N=32) about 3 s |
| Outputs | `results/m3/` 20 MB (18 MB chain archives) |

---

## 8. Remaining limitations and recommendation

1. **The Bayesian posterior of the identifiable field is not converged** (R-hat up to 1.83). Posterior means, mean-free
   variances, credible intervals and all Bayesian trends remain unvalidated. Only the constant-mode part is exact.
2. **Plain pCN is the wrong tool for this posterior geometry** (ridge; Gauss-Newton condition number about 1e4). Convergence by brute
   force needs about 27 h per chain. **Recommendation (requires approval; a methodological change reserved for Milestone 6):** a
   Hessian-informed, prior-reversible proposal, i.e. generalised pCN with the Gauss-Newton/Laplace covariance at the P-MAP, or
   a likelihood-informed-subspace (DILI-type) sampler, validated first on a linear-Gaussian problem with a ridge and a null
   direction, with plain pCN kept as the reference. The ingredients (P-MAP, Gauss-Newton Hessian, diagnostics) now exist.
   Its efficiency gain is expected, not measured.
3. **Two deterministic configurations (alpha = 0.1, 1) remain unconverged** at the historical budget; the historical nodal
   gtol is met in 7 of 46 runs only.
4. Single data realisation and single truth draw (calibration, repeated noise and mesh refinement are excluded from this milestone).
5. 18 MB of chain archives: before committing, decide whether to commit them, thin them further, or keep them out of Git
   with SHA-256 manifests.

---

## 9. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Mathematically justified constant-mode treatment | **met** (exact factorisation; three approaches compared) |
| 2 | Corrected posterior variance consistent with the analytical bound | **met** for the bound, by construction; the mean-free part is not converged |
| 3 | Regression tests for identifiability and posterior uncertainty | **met** |
| 4 | Reliable multi-chain convergence diagnostics | **met** (validated implementation, 4 chains); the convergence **target is not met** |
| 5 | Demonstrated improvement in deterministic optimisation convergence | **met** (2/46 -> 44/46 at the same budget; MAP 0/2 -> 2/2) |
| 6 | Reproducible configurations and fixed seeds | **met** |
| 7 | All existing and new tests passing | **met** |
| 8 | Measured computational costs | **met** |
| 9 | Complete report with remaining limitations | **met** |

**Verdict: CONDITIONAL PASS.** Every listed criterion is met, but the Section C target (R-hat < 1.01, bulk and tail
ESS >= 400) is not achieved, and achieving it needs either a sampler change (approval) or about 27 h per chain.

---

## 10. Reproduction

```bash
PY=/opt/anaconda3/envs/fenicsx/bin/python; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
# MCMC pilot (10 chains, <= 4 processes)
for t in smooth structured; do for b in 0.01 0.02 0.04 0.08; do
  $PY scripts/m3_mcmc.py chain --truth $t --beta $b --n-proposals 8000 --seed 1 --init map \
    --out results/m3/mcmc/pilot/${t}_b${b}_map_s1.npz; done
  $PY scripts/m3_mcmc.py chain --truth $t --beta 0.04 --n-proposals 8000 --seed 1 --init dispersed \
    --out results/m3/mcmc/pilot/${t}_b0.04_dispersed_s1.npz; done
$PY scripts/m3_analyze_mcmc.py pilot
# bounded production
for s in 1 2 3 4; do $PY scripts/m3_mcmc.py chain --truth smooth --beta 0.01 --n-proposals 100000 --seed $s \
  --init dispersed --thin 10 --out results/m3/mcmc/production/smooth_s$s.npz; done
for s in 1 2 3 4; do $PY scripts/m3_mcmc.py chain --truth structured --beta 0.02 --n-proposals 100000 --seed $s \
  --init dispersed --thin 10 --out results/m3/mcmc/production/structured_s$s.npz; done
$PY scripts/m3_analyze_mcmc.py production
# optimiser study
for v in E-hist P E-2000; do for e in reconstruction_demo regularization_study noise_study sensor_study; do
  $PY scripts/m3_optimizer_study.py deterministic --experiment $e --variant $v; done; done
$PY scripts/m3_optimizer_study.py map
$PY scripts/m3_summarize_optimizer.py
# analyses
$PY scripts/m3_constant_mode_analysis.py
$PY scripts/m3_conditioning.py
```

Run the chains as up to four separate processes (as in `results/m3/logs/queue.txt`).

## Appendix: files

New: `src/mcmc_diagnostics.py`, `src/constant_mode.py`, `src/preconditioned_inversion.py`,
`scripts/m3_mcmc.py`, `scripts/m3_analyze_mcmc.py`, `scripts/m3_optimizer_study.py`, `scripts/m3_summarize_optimizer.py`,
`scripts/m3_constant_mode_analysis.py`, `scripts/m3_conditioning.py`, `tests/test_mcmc_diagnostics.py`,
`tests/test_constant_mode.py`, `tests/test_preconditioned_inversion.py`, `docs/audit/M3_PREREGISTRATION.md`, this
report, `results/m3/`.
Modified: `src/experiment.py` (optional `optimizer="preconditioned"`; default unchanged; dual-norm first-order
measures in the diagnostics), `src/inversion.py` (initial and final gradient arrays on the history),
`src/optim_diagnostics.py` (initial gradient attached to the SciPy result).
