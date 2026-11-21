# Milestone 5 pre-registration

Written 2026-11-18, after Milestone 4 integration (`ee59198`) and before any Milestone 5 experiment.
All runs: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, at most four concurrent single-thread processes, fixed seeds.
Likelihood, prior, forward solver and sampler are those of Milestone 4 (unchanged). Chain archives go to
`results/m5/chains/` (outside Git) with a committed SHA-256 manifest.

---

## B. Structured posterior confirmation

**Configuration (fixed; no tuning on these chains).** Historical full configuration (`configs/full.yaml`,
`bayesian_posterior`; structured truth; data seed 7). Sampler: HpCN (`src/hessian_pcn.py`) on the 14 mean-free KL
coefficients, rho = 0.5 (the committed Milestone 4 structured selection), beta_max = 1, eigenvalue floor 0, Hessian
and MAP from the committed `results/m4/hessian_structured.json`. Constant mode exact (analytic), as in Milestone 4.

- 4 chains, **seeds 21, 22, 23, 24** (disjoint from pilot seeds 1-4 and production seeds 11-14; the start draw uses
  seed 10000 + seed, so all starts are independent of earlier chains).
- Starts: xi_R,MAP + 3 L eta, L L^T = (I + H)^-1 (Milestone 4 rule).
- **100,000 proposals per chain**, thin 5, checkpoint every 5,000, first 50% (50,000 proposals) discarded as warmup,
  all remaining retained draws used.
- Monitored (unchanged from Milestone 4, 21 quantities): 14 coefficients xi_R, Phi, m_R at the five probe points,
  ||m_R||_L2, log Q_R (one forward solve per retained draw).

**Criteria (all required):**

1. max rank-normalised split R-hat < 1.01 over the 21 quantities (Vehtari et al. 2021; `src/mcmc_diagnostics.py`);
2. min bulk ESS >= 400; 3. min tail ESS >= 400;
4. all draws, Phi values, QoIs and diagnostics finite; no non-finite likelihood rejections;
5. chain agreement: calibrated max |z| <= 4 (Milestone 4, Deviation 2);
6. constant mode: integrated constant-mode variance equals lambda_c = 1 (exact); independent N(0,1) draws pass
   KS (p > 0.01); pointwise total variance >= 1 everywhere;
7. reproducibility and integrity: a fresh replay of the first 5,000 proposals of chain 21 reproduces the archived
   draws bit for bit; all archives verify against the manifest.

**Budget:** 4 x 100,000 proposals (about 25 min wall on 4 processes) plus log Q_R post-processing (about 8 min).
**No extension.** If any criterion fails, convergence is not established, Phases C-D stop, and the failure is documented.

---

## C. Calibration machinery (before SBC)

`src/calibration.py`, validated by tests on problems with known posteriors: ranks of the truth among L posterior
draws, Rao-Blackwellised probability-integral transforms for quantities with an exact Gaussian constant-mode
component, central credible intervals, coverage with Clopper-Pearson intervals, chi-square rank-uniformity tests,
bias, RMSE, the spread-skill ratio with bootstrap intervals, and Laplace approximations. Required: exact-posterior SBC on a
linear-Gaussian problem is uniform and nominally covered; HpCN on a linear-Gaussian SBC is consistent with calibration.

Laplace reliability on the two real datasets: compare the Laplace approximation N(xi_MAP, (I + H)^-1) with the
converged MCMC (smooth: Milestone 4 production; structured: the B confirmation) per Hessian eigen-direction
(variance ratio, mean offset in posterior sd units).

---

## D. Simulation-based calibration (SBC)

**Generative model = the inference model** (SBC tests the inference procedure under its own model; inverse crime is
intended here): xi ~ N(0, I_15) on the 15-mode KL prior of the N=16 inversion mesh (constant mode included);
m = sum sqrt(lambda_j) xi_j phi_j; pressure from the N=16 P2 forward solver; 5 x 5 Gaussian-bump sensors (width
0.035); y = H p(m) + eps, eps ~ N(0, sigma^2 I) with **sigma fixed at 0.0098584**, the value used by the historical smooth
dataset (the historical scripts set sigma per dataset from the signal; SBC needs one fixed likelihood).
Dataset i uses seed 5000 + i for (xi, eps).

**Inference per dataset (fixed):** P-MAP (Gauss-Newton-preconditioned L-BFGS-B, budget 120), Gauss-Newton Hessian at
the MAP, HpCN with **rho = 0.5**, **4 chains x 10,000 proposals** (seeds 1-4 per dataset; Laplace x3 starts), 50%
warmup. Rank draws: L = 99 draws equally spaced over the pooled post-warmup draws (25, 25, 25, 24 per chain; spacing
about 200 proposals). Moments and intervals use all post-warmup draws. Per-dataset R-hat and ESS are recorded; **no
dataset is excluded** (exclusion would bias SBC). Datasets with max R-hat > 1.05 are reported as a sensitivity subset.

**Pilot (cost only):** datasets 0-3. Their outputs are part of the main study (no reselection). Study size:
D = the largest multiple of 4, at most 100, such that the projected wall time on 4 processes stays within **150 min**,
from the measured mean cost per dataset. If D < 40 the study is reported as underpowered.

**Quantities:** 14 xi_R; m_R at the 5 probes; ||m_R||_L2; log Q_R (on the L rank draws and the posterior-mean field);
the constant mode xi_c (exact N(0,1) posterior; analytic PIT); absolute m at the 5 probes and log k_eff (identifiable
part from the draws plus exact Gaussian constant-mode term; Rao-Blackwellised PIT). Mean-free reconstruction: relative
L2 error of the posterior mean.

**Statistics:** rank histograms (10 bins over ranks 0..99) with chi-square tests (df 9), Bonferroni over quantities at
family-wise 0.01; for PIT quantities, KS uniformity tests on the same Bonferroni scale. Coverage of central 50/80/90/95%
intervals with Clopper-Pearson 95% intervals. Bias (raw and standardised), RMSE, the spread-skill ratio
RMSE^2 / mean posterior variance with bootstrap 95% interval. Per-dataset relative MCSE.

**Calibration verdict (pre-registered):** "consistent with calibration" if (i) no Bonferroni-significant
rank or PIT test, and (ii) the number of (quantity, level) coverage intervals excluding the nominal level does not exceed
the 99th percentile of Binomial(K, 0.05), K the number of such pairs. Otherwise, "evidence of miscalibration", with the
quantities identified.

## E. Robustness (bounded)

1. Laplace SBC (no MCMC; MAP + Gauss-Newton Hessian; Gaussian PITs and coverage for the linear quantities xi_R and
   m_R at the probes): 200 datasets at sigma, sigma/2 and 2 sigma (seeds 7000 + i, 8000 + i, 9000 + i). Budget about 25 min wall.
2. Laplace vs MCMC on the SBC datasets themselves (per-quantity variance ratio and mean offset).
3. Seed and initialisation robustness: structured confirmation (seeds 21-24) vs Milestone 4 production (seeds 11-14):
   posterior means and sds of the 21 quantities, compared with combined MCSE.
4. Weakly identified directions: per eigen-direction MCMC vs Laplace variance (Phase C).

**Total budget for B + D + E: at most about 4.5 h wall.** No phase is extended beyond its stated budget.

## Recorded outcomes (in order, before the corresponding next step)

1. **B passed** (all seven criteria; `results/m5/confirmation_summary.json`, `results/m5/replay_check.json`), so C-D proceed.
2. **SBC size from the cost-only pilot** (datasets 0-3; mean 818 s per dataset at about 20 ms per proposal under load):
   D = 44 (projected 150 min wall). Datasets 0-43, seeds 5000-5043. Observed in the pilot (not used to change the
   design): MAP reached its 120-iteration cap in 3 of 4 datasets; per-dataset max R-hat 1.02-1.09. As fixed above,
   no dataset is excluded; the max R-hat > 1.05 subset is reported as a sensitivity analysis.
