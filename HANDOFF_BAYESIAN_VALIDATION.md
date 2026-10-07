# Handoff: Bayesian MAP and pCN validation

Date: 2026-10-07

## Purpose

This handoff records what the scientific audit can establish from the committed source/results and what still requires execution in the repository's FEniCSx environment.

The deterministic Darcy formulation, adjoint gradient, Gaussian KL prior, and pCN algorithm are not presently classified as wrong. The unresolved issues are **optimizer termination** and **very low effective sample size in the PDE posterior runs**.

## Current branch / PR

- Base: `main`
- Audit branch: `audit/scientific-audit`
- Draft PR #1 contains `docs/scientific_audit.md` and presentation changes that call an unconverged endpoint a **MAP candidate (optimizer not converged)**.
- Do not merge until the rerun/diagnostic work below has been reviewed.

## Evidence already present in the repository

### Gradient

`results/bayesian_gradient_verification.txt` reports first-order Taylor remainders with observed order 1.997, 1.999, 1.999, 2.000. This strongly supports the implemented Bayesian objective gradient.

### Sampler implementation

`results/sampler_validation.txt` compares the same pCN implementation against an analytical linear-Gaussian posterior:

- acceptance: 0.3860;
- posterior-mean relative error: 5.9038e-03;
- posterior-variance mean absolute error: 5.8597e-04;
- ESS: 179--861 over 40,000 retained samples.

This supports algorithmic correctness of pCN, but it does not establish adequate mixing for the nonlinear PDE posterior.

### Full PDE posterior

Committed `results/posterior_demo_full.txt` uses 8,000 retained samples after 2,000 burn-in proposals, beta=0.02.

Smooth truth:
- optimizer: `converged=False`, 120 iterations;
- acceptance: 0.3998;
- ESS range: 4.9--86.1;
- median ESS: about 7.6;
- mean ESS: about 13.2;
- relative MAP-candidate error (historical file says MAP): 0.3868;
- posterior-mean relative error: 1.0009.

Structured truth:
- optimizer: `converged=False`, 120 iterations;
- acceptance: 0.4398;
- ESS range: 5.0--72.7;
- median ESS: about 8.9;
- mean ESS: about 12.9;
- relative MAP-candidate error: 0.8074;
- posterior-mean relative error: 1.4656.

Thus the typical coordinate has only roughly 8--9 effective samples out of 8,000 retained draws. Acceptance near 0.4 is not evidence of good mixing.

### Important comparison

The local and full MAP-candidate coefficients/errors are almost unchanged even though the iteration caps differ (80 vs 120). This suggests the optimizer may be near a stalled/flat solution, but **that cannot be called convergence** without the SciPy termination message and gradient norm. The current code stores only `result.x`, `result.nit`, and `result.success`, losing the diagnostic message and final gradient norm.

## Cheap code improvement to consider before expensive reruns

Extend MAP diagnostics so `compute_map` / `BayesianResult` preserve and report at least:

- `result.success`;
- `result.status`;
- `result.message`;
- `result.nit`, `result.nfev`, `result.njev` when available;
- final objective;
- `||grad||_2` and preferably `||grad||_inf`.

Do not redefine success manually just because the iterate looks stable. Keep SciPy's termination status and report gradient norms separately.

## Local validation sequence

From a clean worktree:

```bash
cd ~/Documents/GitHub/darcy-inverse-problem
git fetch origin
git checkout audit/scientific-audit
git status
conda activate <the environment used for this repository>

python -m pytest tests/ -v
python scripts/verify_bayesian_gradient.py
python scripts/validate_sampler.py
python scripts/run_posterior_demo.py --config smoke
```

Find the exact environment name from the repository/environment setup rather than guessing if it differs.

Before running expensive chains, add/report the MAP termination diagnostics above and rerun smoke/local MAP work.

## MAP investigation

For smooth and structured truths:

1. Record the original 80/120-iteration termination status, message, objective, and gradient norms.
2. Increase the iteration budget without changing the mathematical objective.
3. Compare the terminal coefficient vector, objective, gradient norm, reconstruction error, and number of iterations.
4. If SciPy terminates successfully, it may be called a MAP estimate.
5. If it remains unsuccessful but has a small gradient, report the exact reason and retain qualification.
6. If it materially moves with more iterations, regenerate all downstream posterior runs that start pCN from the MAP candidate.

Do not change optimizer tolerances simply to obtain `success=True` without explaining why.

## pCN investigation

Do not begin by blindly multiplying the chain length. First quantify mixing.

For each truth class, report:
- acceptance rate;
- coordinate ESS min/median/mean/max;
- ESS / retained samples;
- ACFs and integrated autocorrelation behavior;
- trace plots;
- sensitivity to initialization;
- at least several independent chains if computationally feasible.

Because beta=0.02 produces high autocorrelation, run a controlled beta study (for example values around the current beta, chosen after inspecting pilot behavior). Compare **ESS per forward solve / wall-clock cost**, not acceptance rate alone.

If pCN remains inefficient, retain it as the validated baseline and consider a likelihood-informed or gradient-informed method as a separate extension. Do not replace pCN merely to make the plots look better.

## Prior misspecification

Do not remove the structured-truth failure. The smooth Gaussian KL prior/H1-style assumptions are poorly matched to sharp inclusions. Longer MCMC can reduce Monte Carlo error conditional on the prior, but it cannot repair prior/model-class misspecification.

## Results that must remain qualified until rerun

- posterior means;
- pointwise credible intervals;
- integrated posterior variance;
- posterior-predictive interval coverage;
- any claim that the current optimizer endpoint is a MAP estimate.

The five-point truth coverage in the existing output is one synthetic realization, not a calibration experiment.

## Claude prompt

Copy this into a new Claude session at the repository root:

---

Continue the rigorous scientific audit of my `darcy-inverse-problem` repository. Work directly in the repository on my computer. Read `HANDOFF_BAYESIAN_VALIDATION.md`, `docs/scientific_audit.md`, `src/bayesian_experiment.py`, `src/bayesian_darcy.py`, `src/pcn_sampler.py`, the Bayesian tests, configs, and committed result files before changing anything.

The audit has already established that the forward Darcy problem, adjoint formulation, KL prior, and pCN algorithm are scientifically legitimate. The Bayesian gradient has a second-order Taylor-remainder check, and pCN has an independent analytical linear-Gaussian validation. The unresolved issues are:

1. SciPy L-BFGS-B reaches the configured MAP iteration cap with `success=False`, so those endpoints must currently be called MAP candidates rather than established MAP estimates.
2. The full PDE pCN runs retain 8,000 samples but typical coordinate-wise ESS is only about 8--9, despite acceptance around 0.4. Posterior summaries therefore have substantial Monte Carlo uncertainty.
3. The structured truth is deliberately poorly matched to the smooth Gaussian prior. Do not erase this negative result.

Tasks:

1. Inspect git status/branch/diff. Preserve unrelated user changes. Do not merge or force-push.
2. Run the full tests and existing forward/adjoint/Bayesian-gradient/sampler validations.
3. Extend MAP diagnostics minimally so the code records/reports SciPy status, message, iteration/evaluation counts, final objective, and final gradient norms. Do not alter the objective.
4. Run smoke/local MAP diagnostics first. Determine why `success=False`: iteration cap, tolerance, line search, or another reason.
5. Test a larger MAP iteration budget and compare objective, gradient norms, coefficients, and reconstruction error. Do not call an endpoint MAP unless convergence is actually supported.
6. Before expensive MCMC, characterize pCN mixing with the existing chains. Then perform a controlled pilot study of beta and, if feasible, multiple chains. Compare ESS per forward solve/wall time, not acceptance alone.
7. Only after choosing a defensible sampling setup should you run longer posterior experiments.
8. Keep the structured-prior-misspecification experiment as a genuine negative result.
9. Regenerate figures/results/README claims only from newly run corrected/diagnosed experiments, preserving provenance.
10. Before committing substantive result changes, show me: test results, MAP termination diagnostics, MCMC diagnostics, old-vs-new numerical comparison, changed files, and proposed commits.

Do not weaken verification tests to obtain a pass. Do not hide failed convergence. Do not silently overwrite historical results. Distinguish mathematical correctness, optimizer convergence, sampler correctness, mixing efficiency, Monte Carlo uncertainty, and prior misspecification.

---
