# Milestone 5: Posterior Confirmation and Simulation-Based Calibration

Date: 2026-11-21 (experiments 2026-11-18 and 2026-11-21)
Pre-registration: `docs/audit/M5_PREREGISTRATION.md` (written 2026-11-18 after Milestone 4 integration `ee59198`,
before any Milestone 5 experiment). It was not edited after the runs, and no threshold was changed.
Qualification: **Qualified, Not Strong to Discuss** (unchanged).
Verdict: **CONDITIONAL PASS** (Section 11), accepted on review 2026-11-21.

Headline wording: the HpCN posterior inference is **consistent with calibration; gross miscalibration excluded at the
tested sensitivity**. This report does not claim that the posterior is definitively calibrated.
**The Laplace approximation fails calibration at all three tested noise levels (sigma, sigma/2, 2 sigma; Section 5).**

**Pre-registered vs post hoc.** Phases B, C, D (primary analysis and the R-hat <= 1.05 sensitivity subset) and E1-E4 are
pre-registered, and their verdicts follow the pre-registered rules. Analyses marked **[post hoc]** below were added after
seeing the results: the description of the 13 datasets with R-hat > 1.05 (Section 4.4), the power analysis (Section 4.5),
and the split of the Laplace bias by MAP convergence together with the Laplace-vs-MCMC bias comparison (Section 5). They are
descriptive, they change no verdict, and they are computed by `scripts/m5_supplementary.py`.

All runs used `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, at most four concurrent single-thread solver processes,
fixed seeds, and the Milestone 4 likelihood, prior, forward solver and sampler, all unchanged. No historical or Milestone 2-4
result file was modified. Chain archives are kept outside Git and listed with SHA-256 in
`results/m5/chains/MANIFEST.json` (Section 8).

---

## 1. Summary

| Phase | Pre-registered outcome | Result |
|---|---|---|
| B. Structured posterior confirmation (4 x 100,000 HpCN) | all 7 criteria | **passed**: max R-hat 1.0048, min bulk / tail ESS 1135 / 1460, max chain z 2.87, constant mode exact, bitwise replay |
| C. Calibration machinery on known posteriors | uniform and covered when exact; detects known miscalibration; HpCN consistent | **passed** |
| D. SBC, 44 datasets, HpCN 4 x 10,000 | "consistent with calibration" iff no Bonferroni rejection and coverage exclusions <= 99th pct of Binomial(K, 0.05) | **consistent with calibration**: 0 of 27 tests rejected; 0 of 108 coverage intervals exclude nominal (allowed 11) |
| D sensitivity: datasets with max R-hat <= 1.05 (n = 31) | reported only | consistent with calibration (0 rejected; 1 of 108 exclusions) |
| E1. Laplace SBC, 3 x 200 datasets | same verdict rule, reported (bounded) | **evidence of miscalibration at all three noise levels** (negative result for Laplace; Section 5) |
| E2. Laplace vs MCMC on the SBC datasets | reported | MCMC variance exceeds Laplace by a median 0-15% per quantity (max ratio 1.62) |
| E3. Seed robustness (seeds 21-24 vs 11-14) | reported | max abs z 2.09 over 22 quantities; sd ratios 0.982-1.057 |
| E4. Laplace vs MCMC per Hessian eigen-direction, real data | reported | agreement within about 20% in 10 of 14 directions; Laplace **underestimates the variance 2-8x in the 2 most data-informed directions** (1.2-1.4x in the next 2) |

The two most important qualifications, both quantified below: (i) the per-dataset SBC chains are short. Every one of
the 44 has max R-hat above 1.01 (median 1.038, range 1.017-1.100) and median min bulk ESS 133. So the SBC
validates the fixed procedure "HpCN, 4 x 10,000" as a whole, not a converged posterior per dataset. (ii) At D = 44 the
pre-registered test reliably detects only gross miscalibration. A post-hoc power analysis shows that a 20-30% error in a
single quantity's posterior sd, or a 0.5 sd mean shift, would usually go undetected.

---

## 2. Phase B: structured posterior confirmation

Configuration exactly as pre-registered (`configs/full.yaml`, structured truth, data seed 7; HpCN rho 0.5 on the 14 mean-free KL
coefficients; Hessian and MAP from `results/m4/hessian_structured.json`; seeds 21-24; Laplace x3 over-dispersed starts;
100,000 proposals per chain, thin 5, 50% warmup). Evidence: `results/m5/confirmation_summary.json`, `results/m5/replay_check.json`.

| Criterion | Threshold | Observed | Pass |
|---|---|---|---|
| 1. max rank-normalised split R-hat (22 quantities) | < 1.01 | 1.0048 (`xi_R[2]`) | yes |
| 2. min bulk ESS | >= 400 | 1135 | yes |
| 3. min tail ESS | >= 400 | 1460 | yes |
| 4. finiteness | all finite, no non-finite rejections | all finite | yes |
| 5. chain agreement, calibrated max abs z | <= 4 | 2.87 | yes |
| 6. constant mode | lambda_c = 1; KS p > 0.01; pointwise variance >= 1 | lambda_c = 1.0000000000001; KS p = 0.90; pointwise variance 1.23-2.65, integrated 1.73 | yes |
| 7. replay and integrity | first 5,000 proposals of chain 21 bitwise; manifest verifies | 1,000 draws and Phi values identical; 53/53 archives verify | yes |

Acceptance 0.308-0.313; 15.3-15.5 ms per proposal; 400,004 forward solves, no adjoint solves; peak RSS 202 MB.
This closes the narrow Milestone 4 structured failure (`xi_R[6]`, R-hat 1.013): with twice the chain length, `xi_R[6]` is
converged and the worst quantity is `xi_R[2]` at 1.0048. The integrated mean-free posterior variance is 0.730. Adding the exact
constant mode (variance 1) gives the integrated total of 1.730.

Note on the quantity count: the pre-registration (and Milestone 4) text says "21 quantities", but the itemised list
(14 coefficients, Phi, 5 probes, ||m_R||_L2, log Q_R) has 22, and all 22 were monitored. This is a counting error in
the text. Monitoring the extra quantity can only make criteria 1-3 harder to meet.

---

## 3. Phase C: calibration machinery

`src/calibration.py`, tested in `tests/test_calibration.py`. Recorded evidence: `results/m5/analytic_calibration.json` from
`scripts/m5_analytic_calibration.py`, run 2026-11-21 in 20 s. Problem: a linear-Gaussian model in 3 dimensions with
known posterior, L = 99 rank draws, 10 rank bins.

| Case | n | rank chi-square p | PIT KS p | 50 / 80 / 90 / 95% coverage | all 4 CIs contain nominal | spread-skill [95% CI] |
|---|---|---|---|---|---|---|
| exact posterior | 1000 | 0.997 | 0.805 | 0.510 / 0.808 / 0.912 / 0.953 | yes | 0.925 [0.848, 1.003] |
| mean shifted by 1 sd (control) | 1000 | < 1e-200 | ~0 | 0.328 / 0.609 / 0.746 / 0.833 | no (0/4) | 1.95 [1.81, 2.10] |
| sd halved (control) | 1000 | < 1e-100 | ~0 | 0.274 / 0.494 / 0.593 / 0.692 | no (0/4) | 3.70 [3.39, 4.01] |
| HpCN sampler, xi[0] | 300 | 0.699 | 0.192 | 0.517 / 0.797 / 0.897 / 0.933 | yes | 1.03 [0.87, 1.20] |
| HpCN sampler, xi[1] | 300 | 0.810 | 0.935 | 0.463 / 0.807 / 0.907 / 0.947 | yes | 1.00 [0.85, 1.15] |
| HpCN sampler, xi[2] | 300 | 0.175 | 0.568 | 0.487 / 0.803 / 0.937 / 0.960 | 3 of 4 (90%: 0.937, CI excludes 0.90) | 0.95 [0.81, 1.10] |

Verdict: the machinery is uniform and nominally covered when exact, and the two deliberate miscalibrations are detected
decisively. HpCN on the linear-Gaussian SBC is consistent with calibration: one of 12 coverage intervals excludes nominal,
against an allowance of 2 (99th percentile of Binomial(12, 0.05)), and no rank test is significant. The exact-posterior
spread-skill ratio 0.925 has an upper CI bound of 1.003. This is a chance fluctuation (rank and KS tests are uniform), noted
because it shows the spread-skill CI is near its resolution limit at n = 1000.

Tests: `tests/test_calibration.py` and the related sampler, diagnostic, constant-mode and QoI suites: 64 passed. Full
suite (conda `fenicsx`): **174 passed, 15 skipped** (the skips are the PINN tests that need `.venv-deepxde`).

---

## 4. Phase D: simulation-based calibration

### 4.1 Design as run

Generative model = inference model (the inverse crime is intended): xi ~ N(0, I_15) on the N = 16 KL prior including the
constant mode; N = 16 P2 forward solver; 5 x 5 Gaussian-bump sensors; sigma = 0.0098584 fixed. Dataset i uses seed 5000 + i.
The cost-only pilot (datasets 0-3) fixed D = 44 (Recorded outcome 2), and datasets 0-3 are part of the study. All 44 dataset
records were checked to carry exactly the pre-registered configuration: sigma 0.0098584, rho 0.5, 4 chains, 10,000 proposals,
seeds 5000-5043. **No dataset was excluded, rerun or reconfigured.** Aggregation by `scripts/m5_analyze.py sbc`, run
2026-11-21. Output `results/m5/sbc_summary.json`.

D = 44 >= 40, so the study is not underpowered by the pre-registered rule. Section 4.5 quantifies what it can detect.

### 4.2 Pre-registered verdict (primary analysis, all 44 datasets)

| Component | Value |
|---|---|
| Tests (21 rank chi-square for MCMC quantities; 6 KS on Rao-Blackwellised PITs) | 27 |
| Bonferroni threshold (family-wise 0.01) | 3.7e-4 |
| Rejected | **none** (smallest p = 0.151, `xi_R[1]`) |
| (quantity, level) coverage pairs | 108 |
| Pairs whose Clopper-Pearson 95% interval excludes nominal | **0** |
| Allowed (99th percentile of Binomial(108, 0.05)) | 11 |
| **Verdict** | **consistent with calibration** |

The constant mode `xi_c` is analytic and outside the verdict, as pre-registered. Its PIT KS p = 0.845 with nominal coverage
is reported in the table below.

### 4.3 Per-quantity results (all 44 datasets)

Coverage: empirical rate [Clopper-Pearson 95% interval]. Standardised bias: mean of (posterior mean - truth) / posterior sd
± standard error. Spread-skill: RMSE^2 / mean posterior variance [bootstrap 95% interval]; 1 is ideal.

| Quantity | Test | p | 50% | 80% | 90% | 95% | Std. bias (± se) | Spread-skill [95% CI] |
|---|---|---|---|---|---|---|---|---|
| xi_R[0] | rank χ² | 0.463 | 0.48 [0.32, 0.63] | 0.82 [0.67, 0.92] | 0.91 [0.78, 0.97] | 0.95 [0.85, 0.99] | +0.08 ± 0.15 | 1.08 [0.71, 1.50] |
| xi_R[1] | rank χ² | 0.151 | 0.50 [0.35, 0.65] | 0.86 [0.73, 0.95] | 0.91 [0.78, 0.97] | 0.91 [0.78, 0.97] | -0.11 ± 0.15 | 0.98 [0.60, 1.43] |
| xi_R[2] | rank χ² | 0.553 | 0.48 [0.32, 0.63] | 0.86 [0.73, 0.95] | 0.89 [0.75, 0.96] | 0.95 [0.85, 0.99] | +0.08 ± 0.15 | 0.80 [0.52, 1.15] |
| xi_R[3] | rank χ² | 0.784 | 0.55 [0.39, 0.70] | 0.84 [0.70, 0.93] | 0.91 [0.78, 0.97] | 0.91 [0.78, 0.97] | -0.11 ± 0.15 | 1.07 [0.58, 1.67] |
| xi_R[4] | rank χ² | 0.694 | 0.41 [0.26, 0.57] | 0.82 [0.67, 0.92] | 0.91 [0.78, 0.97] | 0.91 [0.78, 0.97] | +0.14 ± 0.16 | 1.10 [0.68, 1.64] |
| xi_R[5] | rank χ² | 0.694 | 0.43 [0.28, 0.59] | 0.80 [0.65, 0.90] | 0.89 [0.75, 0.96] | 0.95 [0.85, 0.99] | -0.06 ± 0.17 | 1.00 [0.62, 1.50] |
| xi_R[6] | rank χ² | 0.984 | 0.50 [0.35, 0.65] | 0.80 [0.65, 0.90] | 0.89 [0.75, 0.96] | 0.93 [0.81, 0.99] | +0.10 ± 0.15 | 0.96 [0.58, 1.37] |
| xi_R[7] | rank χ² | 0.463 | 0.48 [0.32, 0.63] | 0.89 [0.75, 0.96] | 0.93 [0.81, 0.99] | 0.98 [0.88, 1.00] | -0.04 ± 0.14 | 0.84 [0.44, 1.35] |
| xi_R[8] | rank χ² | 0.694 | 0.48 [0.32, 0.63] | 0.84 [0.70, 0.93] | 0.93 [0.81, 0.99] | 0.98 [0.88, 1.00] | +0.03 ± 0.14 | 0.85 [0.57, 1.15] |
| xi_R[9] | rank χ² | 0.784 | 0.45 [0.30, 0.61] | 0.82 [0.67, 0.92] | 0.93 [0.81, 0.99] | 0.95 [0.85, 0.99] | +0.04 ± 0.16 | 1.07 [0.64, 1.57] |
| xi_R[10] | rank χ² | 0.553 | 0.41 [0.26, 0.57] | 0.73 [0.57, 0.85] | 0.91 [0.78, 0.97] | 0.98 [0.88, 1.00] | -0.14 ± 0.16 | 1.10 [0.80, 1.46] |
| xi_R[11] | rank χ² | 0.694 | 0.45 [0.30, 0.61] | 0.75 [0.60, 0.87] | 0.80 [0.65, 0.90] | 0.98 [0.88, 1.00] | +0.16 ± 0.16 | 1.04 [0.67, 1.45] |
| xi_R[12] | rank χ² | 0.421 | 0.64 [0.48, 0.78] | 0.84 [0.70, 0.93] | 0.91 [0.78, 0.97] | 0.93 [0.81, 0.99] | -0.03 ± 0.14 | 0.86 [0.48, 1.29] |
| xi_R[13] | rank χ² | 0.507 | 0.41 [0.26, 0.57] | 0.75 [0.60, 0.87] | 0.84 [0.70, 0.93] | 0.98 [0.88, 1.00] | +0.05 ± 0.16 | 1.05 [0.75, 1.43] |
| m_R(0.25, 0.25) | rank χ² | 0.694 | 0.45 [0.30, 0.61] | 0.86 [0.73, 0.95] | 0.93 [0.81, 0.99] | 0.98 [0.88, 1.00] | -0.08 ± 0.15 | 0.98 [0.66, 1.34] |
| m_R(0.75, 0.25) | rank χ² | 0.984 | 0.52 [0.37, 0.68] | 0.80 [0.65, 0.90] | 0.95 [0.85, 0.99] | 0.98 [0.88, 1.00] | -0.12 ± 0.14 | 1.04 [0.61, 1.54] |
| m_R(0.5, 0.5) | rank χ² | 0.421 | 0.59 [0.43, 0.74] | 0.80 [0.65, 0.90] | 0.89 [0.75, 0.96] | 0.93 [0.81, 0.99] | -0.10 ± 0.16 | 1.06 [0.61, 1.61] |
| m_R(0.25, 0.75) | rank χ² | 0.246 | 0.50 [0.35, 0.65] | 0.86 [0.73, 0.95] | 0.93 [0.81, 0.99] | 0.98 [0.88, 1.00] | +0.13 ± 0.14 | 0.94 [0.59, 1.32] |
| m_R(0.75, 0.75) | rank χ² | 0.599 | 0.50 [0.35, 0.65] | 0.77 [0.62, 0.89] | 0.89 [0.75, 0.96] | 0.95 [0.85, 0.99] | +0.01 ± 0.15 | 1.06 [0.69, 1.46] |
| ‖m_R‖_L2 | rank χ² | 0.740 | 0.41 [0.26, 0.57] | 0.77 [0.62, 0.89] | 0.84 [0.70, 0.93] | 0.95 [0.85, 0.99] | -0.12 ± 0.17 | 1.13 [0.56, 1.93] |
| log Q_R | rank χ² | 0.865 | 0.43 [0.28, 0.59] | 0.84 [0.70, 0.93] | 0.89 [0.75, 0.96] | 0.91 [0.78, 0.97] | -0.17 ± 0.17 | 1.01 [0.51, 1.65] |
| xi_c (analytic) | PIT KS | 0.845 | 0.52 [0.37, 0.68] | 0.75 [0.60, 0.87] | 0.91 [0.78, 0.97] | 0.95 [0.85, 0.99] | -0.01 ± 0.16 | 1.04 [0.64, 1.50] |
| m(0.25, 0.25) absolute | PIT KS | 0.968 | 0.43 [0.28, 0.59] | 0.77 [0.62, 0.89] | 0.91 [0.78, 0.97] | 0.98 [0.88, 1.00] | -0.04 ± 0.16 | 1.07 [0.68, 1.53] |
| m(0.75, 0.25) absolute | PIT KS | 0.690 | 0.48 [0.32, 0.63] | 0.75 [0.60, 0.87] | 0.89 [0.75, 0.96] | 0.93 [0.81, 0.99] | -0.08 ± 0.16 | 1.21 [0.71, 1.81] |
| m(0.5, 0.5) absolute | PIT KS | 0.951 | 0.48 [0.32, 0.63] | 0.77 [0.62, 0.89] | 0.91 [0.78, 0.97] | 0.95 [0.85, 0.99] | -0.05 ± 0.15 | 1.00 [0.68, 1.38] |
| m(0.25, 0.75) absolute | PIT KS | 0.617 | 0.39 [0.24, 0.55] | 0.82 [0.67, 0.92] | 0.95 [0.85, 0.99] | 0.98 [0.88, 1.00] | +0.08 ± 0.14 | 0.89 [0.63, 1.18] |
| m(0.75, 0.75) absolute | PIT KS | 0.927 | 0.45 [0.30, 0.61] | 0.73 [0.57, 0.85] | 0.95 [0.85, 0.99] | 0.98 [0.88, 1.00] | +0.02 ± 0.15 | 0.95 [0.67, 1.23] |
| log k_eff absolute | PIT KS | 0.996 | 0.50 [0.35, 0.65] | 0.80 [0.65, 0.90] | 0.91 [0.78, 0.97] | 0.95 [0.85, 0.99] | -0.05 ± 0.15 | 1.01 [0.61, 1.47] |

Rank histograms (10 bins, expected count 4.4 per bin) are stored per quantity in `results/m5/sbc_summary.json`. No
quantity shows the U shape (under-dispersion) or hump (over-dispersion) pattern beyond chance: the largest chi-square is
13.3 (`xi_R[1]`, p = 0.15). Spread-skill ratios across the 27 quantities with a posterior sd range from 0.80 to 1.21, and every
bootstrap interval contains 1. Standardised biases are all within ±0.17, at most 1.1 standard errors from 0.

Mean-free reconstruction: relative L2 error of the posterior-mean field, mean 0.498, median 0.480 over the 44 datasets.
This reflects the information content of 25 noisy sensors for a 14-mode field. It is not a calibration statistic.

### 4.4 Per-dataset sampler diagnostics and the R-hat > 1.05 sensitivity analysis

| Per-dataset statistic (44 datasets) | median | 10th pct | 90th pct | min | max |
|---|---|---|---|---|---|
| max R-hat over monitored quantities | 1.038 | 1.024 | 1.071 | 1.017 | 1.100 |
| min bulk ESS | 133 | 55 | 241 | 35 | 275 |
| min tail ESS | 184 | 61 | 372 | 32 | 541 |
| max relative MCSE | 0.088 | | | | 0.177 |

Acceptance rates 0.19-0.41. MAP: 39 of 44 reached the 120-iteration cap; 5 converged (98-113 iterations). The MAP only sets
the Hessian preconditioner and the start distribution, and HpCN is exactly prior-reversible for any such choice, so an
unconverged MAP affects efficiency, not the target. It does affect the Laplace approximation (Section 6).

**All 44 datasets exceed R-hat 1.01. Thirteen exceed 1.05:** datasets 2, 3, 6, 7, 14, 20, 22, 28, 32, 39, 40, 41, 42.
As pre-registered, they stay in the primary analysis (Section 4.2).

*Sensitivity (pre-registered): the 31 datasets with max R-hat <= 1.05.* The verdict is consistent with calibration: 0 of 27
tests rejected (threshold 3.7e-4), 1 of 108 coverage intervals excludes nominal (`xi_R[11]` at 90%; allowed 11).

**[post hoc]** *Not pre-registered as a test; descriptive only: the 13 datasets with max R-hat > 1.05.* `scripts/m5_supplementary.py`,
output `results/m5/sbc_supplementary.json`. With n = 13, coverage intervals are very wide (one miss moves a rate by 0.077),
and rank tests use 5 bins.

- No rank (5-bin chi-square) or PIT KS test has p < 0.1 for any of the 28 quantities (smallest: m(0.25, 0.25) absolute,
  KS p = 0.100).
- Pooled over quantities, the high-R-hat datasets do not differ from the others in |PIT - 0.5| (0.259 vs 0.255;
  dataset-level Mann-Whitney p = 0.68) or in squared standardised error (1.04 vs 1.01; p = 0.84).
- One pattern to note: in this subset the two global nonlinear quantities have the largest RMS standardised error,
  ‖m_R‖_L2 1.50 and log Q_R 1.45. log Q_R has 95% coverage 10/13 = 0.77. That is suggestive of under-dispersion for these
  quantities when chains are poorly mixed. It is not significant at n = 13, and nothing comparable appears in the full
  sample (spread-skill 1.13 and 1.01).

### 4.5 What D = 44 can detect [post hoc power analysis]

`scripts/m5_supplementary.py`, 2,000 Monte Carlo replicates per row. One quantity is miscalibrated: the exact posterior is
N(mu, 1) and the reported posterior is N(mu + d, s^2). The other 26 tested quantities are calibrated. The draws are
independent, so these are **optimistic** (upper-bound) powers: the real rank draws are autocorrelated (99 rank draws per
dataset against a median min bulk ESS of 133).

| Miscalibration | P(rank χ² rejects at 3.7e-4) | P(at least one of its 4 coverage intervals excludes nominal) |
|---|---|---|
| none (s = 1, d = 0) | 0.000 | 0.10 |
| sd x 0.8 (over-confident by 20%) | 0.008 | 0.62 |
| sd x 0.7 | 0.056 | 0.90 |
| sd x 0.5 | 0.64 | 1.00 |
| sd x 1.25 (under-confident) | 0.003 | 0.49 |
| sd x 1.5 | 0.029 | 0.93 |
| sd x 2 | 0.39 | 1.00 |
| mean shift 0.25 sd | 0.010 | 0.14 |
| mean shift 0.5 sd | 0.097 | 0.28 |
| mean shift 1 sd | 0.95 | 0.94 |

The pre-registered verdict fails only on a Bonferroni rank rejection or more than 11 of 108 excluding coverage intervals.
So a single quantity miscalibrated by 20-30% in sd, or by 0.5 sd in mean, would almost always pass. Systematic
miscalibration across many quantities would push many coverage intervals out, and is much more detectable. The observed
outcome, 0 of 108 excluding intervals where about 5 would be expected even under perfect calibration, argues against
systematic miscalibration. **Conclusion supported:** no evidence of miscalibration; gross miscalibration (sd off by a factor
of 2, or bias of 1 sd) is excluded for every quantity; moderate single-quantity errors (about ±25% in sd) are not excluded.

---

## 5. Phase E1: Laplace SBC (3 x 200 datasets)

Pre-registered design: no MCMC; MAP (Gauss-Newton-preconditioned L-BFGS-B, budget 120) plus Gauss-Newton Hessian; Gaussian
PITs and coverage for the 19 linear quantities (14 xi_R, m_R at 5 probes); 200 datasets at each of sigma, sigma/2, 2 sigma (seeds
7000 + i, 8000 + i, 9000 + i). Run 2026-11-21 as 24 batches of 25 on four workers under a hard 25-minute watchdog. All
24 batches completed (exit 0) in **992 s wall**. Aggregation: `scripts/m5_analyze.py laplace_sbc` ->
`results/m5/laplace_sbc_summary.json`. Same verdict rule as Phase D (19 KS tests, Bonferroni 0.01/19 = 5.3e-4; 76
coverage pairs, allowance 9).

| Noise | MAP converged | KS rejections | Coverage exclusions (allowed 9) | Spread-skill range | **Pre-registered verdict** |
|---|---|---|---|---|---|
| sigma (0.0099) | 36 / 200 | 1: `xi_R[4]` (p = 4.7e-5) | 2 / 76 | 0.91-1.23 | **evidence of miscalibration** |
| sigma/2 | 2 / 200 | 0 (smallest p = 1.8e-3) | **60 / 76** | 1.20-2.44 | **evidence of miscalibration** |
| 2 sigma | 138 / 200 | 3: `xi_R[3]` (7.7e-7), `xi_R[4]` (2.4e-7), `m_R(0.5, 0.5)` (1.5e-7) | 11 / 76 | 0.84-1.29 | **evidence of miscalibration** |

This is a negative result for the Laplace approximation, and it is recorded as such. Two distinct failure modes appear:

1. **Over-confidence at low noise (sigma/2).** Spread-skill ratios of 1.7-2.4 for the 8 leading coefficients and the probes:
   Laplace variances are roughly half the actual error variance, and 95% intervals cover 80-89%. At this noise level the MAP
   almost never satisfies its convergence test within 120 iterations (2/200), so optimiser error and posterior non-Gaussianity
   both contribute. They are not separated here.
2. **A systematic MAP-versus-posterior-mean bias in one direction pair.** The standardised biases of `xi_R[3]` (negative),
   `xi_R[4]` (positive) and `m_R(0.5, 0.5)` (negative) have the same sign at all three noise levels: about ±0.23 at sigma and
   ±0.43 at 2 sigma. Post hoc (`scripts/m5_supplementary.py`, `post_hoc_laplace_bias`), the bias is **just as large when the MAP
   has converged** (2 sigma, n = 138: -0.39, +0.41, -0.42; sigma, n = 36: -0.24, +0.20, -0.19), so it is not an optimisation
   artefact **[post hoc]**. It is consistent with a skewed posterior whose mode differs from its mean along the near-degenerate KL pair
   (3, 4) identified in Milestone 3. That mechanism is a hypothesis, not tested here.

**[post hoc]** **The MCMC removes this bias on identical data.** On the 44 Phase D datasets (same truths, same observations), the
standardised biases are:

| Quantity | Laplace (MAP) | HpCN MCMC (± se) |
|---|---|---|
| xi_R[3] | -0.39 | -0.11 ± 0.15 |
| xi_R[4] | +0.42 | +0.14 ± 0.16 |
| m_R(0.5, 0.5) | -0.38 | -0.10 ± 0.16 |

For all other linear quantities the two agree to within 0.11. The Laplace calibration test on these same 44 datasets
nonetheless returned "consistent with calibration" (Section 6). That is a concrete instance of the power limitation in
Section 4.5: a 0.4 sd bias went undetected at D = 44 and was detected at D = 200. The MCMC residuals (about 0.1 sd, same signs)
are within one standard error of zero. With D = 44, though, a bias of that size could not be detected either way.

---

## 6. Phases E2 and E4: Laplace versus MCMC

**E2, on the 44 SBC datasets** (`results/m5/sbc_supplementary.json`). For each linear quantity, the ratio of MCMC to Laplace
posterior variance and the offset of the MCMC mean from the Laplace mean (MAP), in Laplace sd:

| Quantity group | variance ratio, median of per-quantity medians | range of per-quantity medians | largest single-dataset ratio | median abs offset | largest abs offset |
|---|---|---|---|---|---|
| xi_R[0-13] | 1.07 | 0.97-1.15 | 1.62 (`xi_R[2]`) | 0.06-0.31 | 1.37 (`xi_R[5]`) |
| m_R at 5 probes | 1.12 | 1.03-1.13 | 1.54 | 0.09-0.33 | 1.32 |

On the SBC datasets the Laplace approximation is mildly over-confident (typically 5-15% too little variance) and
its centre is sometimes displaced by more than 1 Laplace sd. The largest displacements are in `xi_R[3]`, `xi_R[4]` and `m_R(0.5, 0.5)`, where E1
(Section 5) shows a systematic MAP-versus-mean bias that persists when the MAP has converged. The Laplace calibration on these same 44 datasets (`results/m5/sbc_summary.json`,
`laplace_verdict`) is nonetheless consistent with calibration: 0 of 19 KS tests rejected, 2 of 76 coverage intervals exclude
nominal (allowed 9). Given Section 4.5, this means only that Laplace is not grossly miscalibrated here.

**E4, the two real datasets, per Hessian eigen-direction** (`results/m5/laplace_vs_mcmc_real.json`; smooth: Milestone 4
production; structured: the Phase B confirmation). Re-run on 2026-11-21 with output redirected to a scratch directory, the file
reproduces **byte for byte**.

| Hessian eigenvalue lambda | smooth: var ratio MCMC/Laplace | structured: var ratio |
|---|---|---|
| < 0.1 (3 prior-dominated directions) | 1.05-1.12 | 1.00-1.20 |
| 3-110 (7 directions) | 1.02-1.17 | 1.04-1.20 |
| 200-600 (2 directions) | 1.33, 1.40 | 1.23, 1.39 |
| about 1e3 | 2.06 | 2.38 |
| about 8e3 | **4.44** | **8.19** (skewness -0.43) |

Mean offsets are at most 0.39 Laplace sd (smooth) and 0.32 (structured). The Gauss-Newton Laplace approximation is
adequate in prior-dominated and moderately informed directions. It **substantially underestimates the posterior variance in
the 2 most data-informed directions** (2.1-8.2x) and moderately in the next 2 (1.2-1.4x), where the likelihood is most
nonlinear. Any uncertainty reported from Laplace
alone for quantities that load on those directions would be over-confident. This does not affect the MCMC results.

## 7. Phase E3: seed and initialisation robustness

Confirmation chains (seeds 21-24, 100,000 proposals) vs Milestone 4 production (seeds 11-14, 50,000 proposals), structured
data, post-warmup posterior means and sds of the 22 monitored quantities (`results/m5/seed_robustness.json`; re-run
2026-11-21 into a scratch directory, reproduces **byte for byte**).

- max abs z of the mean difference (combined MCSE) = 2.09 (`xi_R[9]`); 2 of 22 exceed 2 (`xi_R[8]` 2.03, `xi_R[9]` 2.09).
  Under independence about 1 is expected, and the Bonferroni 5% bound for 22 tests is about 3.0.
- sd ratios 0.982-1.057.
- The Milestone 4 structured chains failed R-hat narrowly, so their MCSEs are slightly optimistic and these z values are, if
  anything, overstated.

Independent seeds and starts give the same posterior to within Monte Carlo error.

---

## 8. Integrity, reproducibility and archives

| Item | Result |
|---|---|
| M4 chain manifest (input integrity) | `verify()` on `results/m4/chains/MANIFEST.json`: no problems |
| M5 manifest, before update | 9 archives (confirmation + replay), verified, backed up before regeneration |
| M5 manifest, after update | **53 archives** (9 earlier + 44 SBC, 52.9 MB), regenerated by `scripts/make_chain_manifest.py results/m5/chains`; the 9 earlier entries are unchanged (path, size, SHA-256); `shasum -a 256 -c results/m5/chains/MANIFEST.sha256`: **53/53 OK**; SHA-256 of every archive before and after the update identical (archives untouched) |
| SBC configuration audit | all 44 records: sigma 0.0098584, rho 0.5, 4 chains, 10,000 proposals, seed 5000 + i, datasets 0-43 contiguous |
| Reproduction of E3 / E4 evidence | byte-identical |
| Replay (criterion 7) | bitwise |

**Defect found and fixed (uncommitted): the chain manifests were not tracked by Git.** `.gitignore` excluded the whole
directories `results/m4/chains/` and `results/m5/chains/`, so `results/m4/chains/MANIFEST.{json,sha256}` was never
committed. This contradicts the Milestone 4 report and the commit message of `ee59198`, which describe a committed manifest.
The new M5 manifest would have been ignored the same way. The patterns are now `results/m4/chains/**/*.npz` and
`results/m5/chains/**/*.npz`, matching the Milestone 3 convention. The archives remain ignored (checked with
`git check-ignore`), and both manifests now appear as untracked files ready to commit. No archive was moved or modified.

There is still no off-machine copy of the chain archives (M3 19.3 MB, M4 34.6 MB, M5 52.9 MB). The manifests make them
verifiable but not recoverable.

---

## 9. Computational cost

| Phase | Work | CPU | Wall | Budget |
|---|---|---|---|---|
| B confirmation | 4 x 100,000 proposals (400,004 forward solves) + log Q_R post-processing | sampler 6,157 s | 34 min (01:32-02:06, 2026-11-18) | about 33 min |
| C analytic | 1000 x 3 exact cases + 300 HpCN datasets | 20 s | 20 s | n/a |
| D SBC | 44 datasets; mean per dataset 629 s (MAP + Hessian 7.5 s, sampling 619 s, QoI 2.2 s, 40,004 forward solves) | 7.69 h | 116 min (13:46-15:42, 2026-11-18) | 150 min |
| E1 Laplace SBC | 600 datasets, MAP + Hessian only (mean 5.7-7.0 s each) | about 64 min | 16.5 min (992 s) | 25 min |
| E2-E4, supplementary | aggregation only | < 2 min | < 2 min | n/a |

Peak RSS per process 161-202 MB. Each solver process showed about 240% CPU in `ps` despite `OMP_NUM_THREADS=1`, with high
system time (also visible in the test run). This is attributed to the FEniCSx/MPI stack rather than BLAS threading. The same
environment produced the bitwise-reproducible results of Milestones 2-5. It is recorded here because "single-thread" should
not be read as one core per process.

---

## 10. Limitations

1. **Short per-dataset SBC chains.** Every SBC dataset has max R-hat > 1.01, and the median min bulk ESS is 133. The SBC
   therefore certifies the fixed procedure (HpCN 4 x 10,000, rho 0.5, Laplace x3 starts) and not a converged posterior per
   dataset. Converged single-dataset inference has been shown only for the two real datasets (Milestone 4 smooth; Phase B structured).
2. **Limited power.** See Section 4.5. Moderate single-quantity miscalibration (about ±25% in sd, 0.5 sd bias) is not excluded.
   Ruling it out would need D in the hundreds at the current cost of about 10.5 CPU-min per dataset.
3. **Autocorrelated rank draws.** The L = 99 rank draws are spaced about 200 proposals apart, but per-dataset ESS is
   comparable to or below 99. Correlated rank draws reduce SBC sensitivity and can distort rank histograms. No such
   distortion is visible, but the computed powers are upper bounds.
4. **Well-specified model only (inverse crime by design).** Same mesh, prior, noise level and sensors for generation and
   inference. SBC says nothing about discretisation error, prior misspecification or noise misspecification. Those
   belong to robustness testing (Milestone 8 in the plan).
5. **One noise level for the MCMC SBC** (sigma 0.0098584). Other noise levels are covered only by the Laplace SBC (E1).
6. **Laplace is not calibrated for this problem.** E1 shows evidence of miscalibration at all three noise levels (over-confidence at
   sigma/2; a systematic bias of 0.2-0.4 sd in `xi_R[3]`, `xi_R[4]` and `m_R(0.5, 0.5)`), and on real data the variance is 2-8x too
   small in the 2 most informed directions (Section 6). Laplace-based UQ should not replace MCMC here.
7. **Unconverged MAP** in 39 of 44 SBC datasets (iteration cap 120). It does not bias HpCN. It degrades Laplace means, but it does
   not explain the E1 bias pattern, which persists in MAP-converged datasets.
8. **Multiplicity of coverage pairs.** The 108 (quantity, level) pairs are strongly correlated across levels and quantities, so
   the Binomial(108, 0.05) allowance is a heuristic tolerance, not an exact test. The observed 0 exclusions passes under any
   reasonable dependence.
9. **Archives exist on one machine only.**

---

## 11. Verdict

**CONDITIONAL PASS.**

*Why not FAIL.* Every pre-registered acceptance criterion for the MCMC inference was met, with thresholds as written: the
structured posterior confirmation passed all seven criteria (Section 2); the calibration machinery was validated on
known posteriors, including detection of deliberate miscalibration (Section 3); and the pre-registered SBC verdict for the
HpCN procedure over all 44 datasets, with none excluded, is "consistent with calibration", with 0 rejections and 0 of 108
excluding coverage intervals (Section 4). Seed robustness and both re-run evidence files reproduce exactly (Sections 6-7),
and all 53 archives verify (Section 8). On identical data, MCMC removes the systematic bias the Laplace approximation shows
(Section 5).

*Why not PASS.* The positive SBC result is genuine, but it is weaker than "the posterior is calibrated":
(a) it was obtained with per-dataset chains that are individually unconverged (all 44 above R-hat 1.01). It therefore
certifies the fixed 4 x 10,000 procedure, not per-dataset converged posteriors. (b) At D = 44 it excludes only gross
miscalibration. Errors of about 25% in sd or 0.5 sd in mean for individual quantities remain possible, and E1 shows
directly that a 0.4 sd bias can pass at D = 44. (c) It covers one noise level under a well-specified model.
(d) The archived evidence is not yet fully reproducible from Git: the manifest-tracking defect (Section 8) is fixed in the
working tree but not committed, and the archives have no off-machine copy.

*Conditions for the claim to be used.*
1. Wording: "consistent with calibration; gross miscalibration excluded at the tested sensitivity" (44-dataset SBC at the
   historical noise level; moderate, about 25%, single-quantity miscalibration is not resolved). Not "calibrated" without
   qualification.
2. The Laplace approximation must not be presented as a calibrated UQ method for this problem. E1 shows evidence of
   miscalibration at all three noise levels, and E4 shows 2-8x variance underestimation in the most informed directions on
   real data. This negative result belongs in any summary of Milestone 5.
3. Commit the `.gitignore` fix together with both chain manifests (M4 and M5), so the archive provenance claimed since
   Milestone 4 is actually in the repository.
4. Make an off-machine copy of the chain archives (107 MB in total) before they are relied on in later milestones.

None of these conditions needs new experiments. Strengthening (b) would need a larger SBC (D in the hundreds; about 10.5
CPU-min per dataset at the current settings) or longer per-dataset chains. That is a design decision for a later
milestone, not part of Milestone 5.

---

## 12. Reproduction

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/opt/anaconda3/envs/fenicsx/bin/python
$PY -m pytest -q tests                                     # 174 passed, 15 skipped
$PY scripts/m5_analytic_calibration.py                     # Phase C -> results/m5/analytic_calibration.json
$PY scripts/m5_sbc.py mcmc --start 0 --n 44                # Phase D (about 7.7 CPU-h; existing datasets are skipped)
$PY scripts/m5_analyze.py sbc                              # -> results/m5/sbc_summary.json
$PY scripts/m5_sbc.py laplace --start 0 --n 200 --seed-base 7000 --sigma-mult 1.0 --tag sigma1   # E1, also 8000/0.5, 9000/2.0
$PY scripts/m5_analyze.py laplace_sbc                      # -> results/m5/laplace_sbc_summary.json
$PY scripts/m5_supplementary.py                            # E2, post-hoc subsets and power -> results/m5/sbc_supplementary.json
$PY scripts/m5_analyze.py laplace_real; $PY scripts/m5_analyze.py seeds   # E4, E3 (overwrite with identical bytes)
shasum -a 256 -c results/m5/chains/MANIFEST.sha256         # 53/53
```

The E1 runs on 2026-11-21 used 24 batches of 25 datasets (`--start 0, 25, ..., 175`) on four workers. The files are
`results/m5/sbc/laplace/<tag>_<start>.json`, and the logs are in `results/m5/logs/laplace_*.log`.
