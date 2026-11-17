# Milestone 4 pre-registration: Hessian-informed pCN

Written 2026-11-17, before implementing the sampler and before any Milestone 4 experiment.
All runs: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, at most four concurrent single-thread processes,
fixed seeds, historical full configuration (`configs/full.yaml`, `bayesian_posterior`, data seed 7),
14 mean-free KL coefficients (constant mode treated exactly, as in Milestone 3), forward-only likelihood.

## Sampler

Hessian-informed pCN (HpCN): xi' = V( diag(a) V^T xi + diag(beta) eta ), eta ~ N(0, I_14),
a_i = sqrt(1 - beta_i^2), beta_i = min(beta_max, rho / sqrt(1 + lambda_i)), where V diag(lambda) V^T is the
Gauss-Newton misfit Hessian G^T Gamma^-1 G at the Milestone 3 P-MAP. Acceptance min(1, exp(Phi(xi) - Phi(xi'))).
Defaults: beta_max = 1, eigenvalue floor 0 (negative round-off eigenvalues clipped to 0).

## Gate before any production run

All Phase 3 tests pass in both environments.

## Pilot

Per truth, for rho in {0.25, 0.5, 0.75, 1.0}: 4 chains x 4,000 proposals, dispersed starts xi_R,MAP + N(0, I)
(seeds 1-4, the Milestone 3 start rule), first half discarded.
Plain pCN reference at a matched budget and protocol: the first 4,000 proposals of the four Milestone 3
production chains per truth (same dispersed-start rule, beta 0.01 / 0.02), first half discarded.

Monitored quantities: 14 coefficients xi_R, Phi, m_R at the five probe points, and the mean-free L2 norm
||m_R||_L2. Production additionally monitors the mean-free effective permeability log Q_R (one forward solve per
retained draw, computed after sampling).

Selection rule: the rho with the largest minimum (over monitored quantities) pooled bulk ESS per second of
sampler time. Acceptance rate is reported but not used.

Go/no-go to production: proceed only if the selected HpCN pilot has at least 10 times the minimum bulk ESS per
second of the matched pCN reference, for both truths.

## Production (fixed in advance)

4 chains per truth, dispersed starts (seeds 11-14), selected rho. Length N per chain (identical for both truths,
rounded up to 5,000) chosen from the pilot so that the predicted pooled minimum bulk ESS after 50% warmup is at
least 1,000 (2.5 x the target), with a **hard cap of 50,000 proposals per chain** (about 12 min per chain;
at most about 25 min wall for all 8 chains). Thin 5. No extension, truncation or selective discarding of chains
after launch.

Convergence (all required, after 50% warmup, for every monitored quantity including log Q_R):
max rank-normalised split R-hat < 1.01; bulk ESS >= 400; tail ESS >= 400; chain-wise posterior means within
4 pooled MCSE of each other for xi_R; constant-mode treatment exact (analytic; independent-draw check KS p > 0.01);
no non-finite likelihood values and no checkpoint inconsistencies. If any criterion fails within the budget, the
outcome is reported as failure to establish convergence.

## Scientific comparison

HpCN production vs the Milestone 3 plain-pCN production chains (4 x 100,000 proposals per truth), reported per
PDE solve and per second, with acceptance, max R-hat, min bulk/tail ESS, integrated mean-free and constant-mode
variance, and posterior-mean errors.

## Deviations (recorded before the corresponding runs)

1. **Starting points (recorded after the Phase 3 tests, before any pilot).** On the linear-Gaussian ridge test
   (`tests/test_hessian_pcn.py`), chains started at prior-scale offsets spent several thousand proposals
   descending in the most data-informed direction (posterior sd 0.003, start hundreds of posterior sds away),
   while chains started over-dispersed relative to the posterior mixed with tau of about 5 to 17 in every
   eigen-direction. The pre-registered start rule xi_R,MAP + N(0, I) is prior-scale and would make a 4,000-proposal
   pilot measure that transient rather than mixing. The pilot and production therefore start from
   **xi_R,MAP + 3 L eta, with L L^T = (I + H)^-1 the Laplace covariance at the MAP, eta ~ N(0, I)**: over-dispersed by
   3x relative to the Laplace posterior in every eigen-direction (the standard over-dispersion requirement for R-hat).
   The burn-in cost from prior-scale starts is quantified separately in the pilot (one rho, 4 chains, starts
   xi_R,MAP + N(0, I), 8,000 proposals) and reported. The plain-pCN reference (Milestone 3 chains) used prior-scale
   starts; its slow mixing (tau of about 3e4) is not driven by the start: Phi reached its typical range within the first 8,000 proposals.

2. **Chain-agreement criterion (recorded during the pilot run, before any pilot result was analysed and before
   production).** As written ("chain-wise posterior means within 4 pooled MCSE of each other") the criterion is
   statistically ill-posed: with 4 chains each chain mean has about twice the pooled MCSE, so pairwise differences
   have a standard deviation of about 2.8 pooled MCSE and the check would fail by chance even for perfectly mixed
   chains. It is replaced by a calibrated test: for every monitored quantity and chain c,
   z_c = (mean_c - pooled mean) / (MCSE_c sqrt((M - 1)/M)), with MCSE_c the chain's own MCSE, must satisfy
   max |z_c| <= 4 (family-wise false-alarm probability about 0.5% over 80 checks for independent chains).
   The original statistic is still reported.

## Pilot outcome and fixed production parameters (recorded at production launch)

From `results/m4/pilot_summary.json` by the rules above: selected rho = 0.75 (smooth) and 0.5 (structured);
min bulk ESS per second 0.438 and 0.258 vs 0.0198 for the matched pCN reference (ratios 22.1 and 13.0, both >= 10:
**GO**). Predicted rate 0.00885 min bulk ESS per post-warmup proposal gives 60,000 proposals per chain, so the
**50,000 cap binds**: 4 chains x 50,000 proposals per truth, thin 5, seeds 11-14, Laplace x3 starts. No change after launch.
