# Scientific Audit — Darcy Permeability Inversion

**Status:** repository/formulation audit complete; rerun of corrected presentation output pending.

## Current verdict

| Component | Verdict |
|---|---|
| Forward problem | **Supported.** Uniformly positive permeability gives a standard elliptic Darcy problem. The log parameterisation enforces positivity. |
| P2 pressure / P1 log-permeability FEM | **Supported.** The manufactured-solution study separates exact-coefficient pressure accuracy from coefficient-interpolation-limited accuracy. |
| Sparse pressure coefficient inversion | **Meaningful but ill-posed/weakly identifiable.** Regularisation or a prior is necessary; recovery quality depends strongly on truth class and observation design. |
| H1-seminorm Tikhonov regularisation | **Appropriate with qualification.** It encodes smoothness and therefore predictably suppresses sharp inclusions. Failure on structured truth is a model/regulariser mismatch, not a forward-solver failure. |
| Adjoint gradient | **Supported.** Direct Taylor-remainder verification reports second-order remainder decay. |
| L-BFGS deterministic inversion | **Appropriate.** |
| Gaussian KL prior | **Appropriate for smooth log-permeability fields; misspecified for sharp inclusions.** |
| pCN sampler | **Algorithmically supported, practically under-mixed in the reported PDE runs.** It validates on an analytical linear-Gaussian target, but many reported PDE ESS values are only single digits to tens. |
| Reported MAP | **Presentation correction required.** Full runs have `MAP converged=False`; the terminal optimiser iterate is therefore a MAP candidate, not an established MAP estimate. |
| Structured-truth Bayesian failure | **Genuine negative result.** More samples cannot repair prior support/model-class mismatch by themselves. |

## 1. Problem statement and solvability

The forward model is
[
-\nabla\cdot(k\nabla p)=f,qquad k=\exp(m)>0,
]
on the unit square, with pressure prescribed on left/right boundaries and no-flow conditions on top/bottom for the physical experiment. For bounded positive permeability this is a coercive elliptic boundary-value problem after imposing the Dirichlet conditions, so the state problem is well posed.

The inverse task—recovering a spatially varying coefficient from finitely many noisy pressure averages—is fundamentally different. Sparse observations do not uniquely or stably determine arbitrary fine-scale permeability. The repository is therefore correct to formulate the deterministic problem with regularisation and the Bayesian problem with a prior.

**Classification:** right problem, appropriate mathematical formulation.

## 2. Observation operator

Sensors are finite-width Gaussian-weighted pressure averages rather than mathematical point evaluations. This is a defensible bounded observation functional and avoids treating a Dirac distribution as an ordinary finite-element load.

The audit found no conceptual mismatch between the forward observation and its adjoint: the same assembled weight vectors define both (H) and (H^T).

## 3. Deterministic inversion

The objective combines noise-weighted pressure misfit with either an H1-seminorm or L2-to-reference penalty. H1 regularisation is a standard coefficient-identification choice, but it is a modelling assumption: it favours smooth fields.

The repository's poor recovery of sharp structured inclusions is therefore not evidence that Darcy inversion or the adjoint method is wrong. It demonstrates that a smoothness prior/regulariser is poorly matched to a discontinuous truth and that sparse pressure data do not overcome that mismatch.

A future extension could compare total variation, level-set/geometric priors, Besov-type priors, or other edge-preserving/structure-aware models. Such an extension would answer a different modelling question and should not overwrite the current negative result.

## 4. Gradient correctness

The adjoint uses the same self-adjoint Darcy bilinear form with homogeneous adjoint Dirichlet data and an observation-residual right-hand side. The gradient includes the derivative of (k=\exp(m)).

The committed Taylor-remainder evidence reaches order approximately 2.000 for both implemented regularisers. That is strong implementation-level evidence that the adjoint gradient matches the discrete objective.

## 5. Bayesian formulation

The KL parameterisation whitens a truncated Gaussian prior and pCN proposes
[
\xi' = \sqrt{1-\beta^2}\,\xi + \beta\eta,qquad \eta\sim N(0,I).
]
This is a standard prior-preserving proposal. The implementation is separately validated on a linear-Gaussian problem with an analytical posterior.

The Bayesian inverse problem is therefore **not using the wrong sampler in principle**.

## 6. Main Bayesian limitation: mixing

The full PDE run retains 8000 posterior draws after 2000 burn-in proposals with `beta=0.02`. Yet many coordinate-wise ESS values are only about 5–20. Acceptance near 0.4 does not compensate for that strong autocorrelation.

Consequences:

- the chain demonstrates a plausible posterior exploration, but not a high-precision posterior estimate;
- posterior means and pointwise quantiles have substantial Monte Carlo uncertainty;
- the reported five-point truth coverage is explicitly not a calibration study;
- posterior predictive interval coverage on one synthetic data set is diagnostic, not a frequentist coverage guarantee.

**Classification:** appropriate method, insufficient effective sampling for strong quantitative posterior-summary claims.

## 7. MAP terminology correction

The committed full result says `MAP converged=False, n_iter=120` for both truth classes. A finite L-BFGS iterate at a failed/iteration-limited termination cannot be called an established MAP estimate without qualification.

This audit branch changes the plotting/report script so that when `map_converged=False`, the result is labelled **MAP candidate (optimizer not converged)**. No numerical value is changed.

## 8. Structured truth and prior misspecification

The structured truth is poorly represented by the smooth Gaussian prior and the deterministic H1 regulariser. This is scientifically useful: Bayesian uncertainty is conditional on the assumed prior/model class. It cannot quantify uncertainty about structures that the prior effectively excludes or strongly suppresses.

The reconstruction failure should remain in the project.

## 9. What should be rerun before final publication

1. run the complete tests and forward/adjoint verification;
2. rerun the Bayesian headline experiment with corrected MAP-candidate labels;
3. increase or adapt the MAP optimisation budget and inspect termination messages/gradient norms;
4. run substantially longer or multiple pCN chains, report split-chain convergence diagnostics where meaningful, and quantify Monte Carlo error;
5. consider whether a likelihood-informed or gradient-informed sampler is warranted if pCN remains inefficient;
6. retain the structured-prior-misspecification experiment as a negative result.

## Final classification

**No wrong forward problem. No wrong primary deterministic method. No fundamental pCN implementation error identified.**

The two important scientific qualifications are:

1. the smooth H1/Gaussian assumptions are deliberately inadequate for sharp structured truth, producing a real model-class failure;
2. the reported PDE pCN chains have too little effective sampling for precise posterior summaries.

The only immediate code/presentation correction identified in this pass is to stop calling an unconverged optimisation endpoint an unqualified MAP estimate.
