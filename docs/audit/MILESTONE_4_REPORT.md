# Milestone 4: Hessian-Informed Bayesian Sampling

Date: 2026-11-17
Qualification: **Qualified, Not Strong to Discuss** (unchanged).
Decision: **CONDITIONAL GO** (Section 11).

Pre-registration: `docs/audit/M4_PREREGISTRATION.md`. It was written before implementation and records two deviations made before
the runs they affect, plus the pilot outcome at production launch. All runs used `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`, at most four concurrent single-thread processes, fixed seeds, the historical full
configuration (N=16, r=15, 25 sensors, 2% noise, data seed 7), and the forward-only likelihood. No historical, Milestone 2
or Milestone 3 result changed.

---

## 0. Milestone 3 integration (Phase 0)

| Step | Result |
|---|---|
| Inspection | uncommitted changes matched the Milestone 3 report's file list. Spot-checks of all headline numbers against the result files found one rounding error (min tail ESS 11.8, reported as 11.9), which was corrected; the report header was updated |
| Tests | 148 passed + 15 skipped (fenicsx); 62 passed (`.venv-deepxde`) |
| Historical results | unchanged (`git diff` on `results/`, `figures/`, `configs/` empty) |
| Commits | `55ff89c` feat (MCMC diagnostics), `c5f64e8` feat (constant mode, preconditioned optimisation), `b6dd37c` test, `5de024b` feat (Milestone 3 scripts, manifest tool), `6f445af` docs, `10fbe2a` chore (ignore chain archives), `40c8b76` results |
| Push | `0019d11..40c8b76` to `origin/main`; Scientific CI: success |
| **Final Milestone 3 commit** | **`40c8b76`** |
| Chain archives | the 18 Milestone 3 archives (19.3 MB) are kept unmodified at `results/m3/mcmc/{pilot,production}/` (ignored by Git), listed with SHA-256, sizes and run metadata in the committed `results/m3/mcmc/MANIFEST.json`; `shasum -a 256 -c results/m3/mcmc/MANIFEST.sha256` passes (18/18) |

---

## 1. Mathematical derivation

Target on the 14 mean-free KL coefficients (the constant mode is exact and separate, Milestone 3):
pi(d xi) ∝ exp(-Phi(xi)) mu_0(d xi), with mu_0 = N(0, I_14).

**Which Gaussian proposals are prior-reversible.** For Q(xi, .) = N(A xi, S), reversibility with respect to mu_0
means the joint law of (xi, xi') with xi ~ mu_0 is exchangeable. That joint law is N(0, [[I, A^T], [A, A A^T + S]]), which is
symmetric under swapping the blocks if and only if

    A = A^T  and  S = I - A^2   (hence ||A|| <= 1).

Consequences, all checked numerically in the tests:

- pCN is the case A = sqrt(1-beta^2) I.
- A Hessian-preconditioned random walk (A = I, S = s^2 L L^T) violates A^2 + S = I. It is reversible with respect to
  Lebesgue measure and needs the prior ratio in its acceptance.
- A proposal centred at the MAP is reversible with respect to a Gaussian centred at the MAP, not mu_0.
- The generalised pCN of Rudolf & Sprungk (2018), N(sqrt(1-s^2) xi, s^2 C), is reversible with respect to N(0, C), so it is
  prior-reversible only if C = I.

**Hessian-informed pCN (HpCN).** Let H = G^T Gamma^-1 G = V diag(lambda) V^T be the Gauss-Newton misfit Hessian at the
Milestone 3 P-MAP (whitened coordinates). Set

    A = V diag(a) V^T,  S = V diag(beta^2) V^T,  a_i = sqrt(1 - beta_i^2),  beta_i = min(beta_max, rho / sqrt(1 + lambda_i)).

Then A = A^T and A^2 + S = I exactly, so the proposal is mu_0-reversible. In eigen-coordinates z = V^T xi it is an
independent pCN step per direction, z_i' = a_i z_i + beta_i eta_i, with the step proportional to the Laplace posterior scale
(1 + lambda_i)^{-1/2} in data-informed directions and approaching the prior scale in prior-dominated ones.

**Acceptance and invariance.** Because q is mu_0-reversible, the Metropolis-Hastings ratio
pi(xi')q(xi',xi) / (pi(xi)q(xi,xi')) reduces to exp(Phi(xi) - Phi(xi')). The acceptance min(1, exp(Phi(xi) - Phi(xi')))
is therefore the correct MH acceptance, and the kernel satisfies detailed balance with respect to the posterior.
*Prior reversibility* is a property of the proposal alone. *Posterior invariance* is a property of the full kernel and
follows from prior reversibility plus the likelihood-ratio acceptance. The proposal is fixed (computed once at the MAP),
so the chain is time-homogeneous. A proposal with non-finite Phi is rejected (MH with Phi = +inf).

**Constant mode.** Unchanged: xi_c ~ N(0,1) exactly and independently. Variance, intervals and the effective permeability
include it analytically: log k_eff = log Q(m_R) + sqrt(lambda_c) phi_c xi_c.

---

## 2. Implementation

| File | Content |
|---|---|
| `src/hessian_pcn.py` | `HessianPCNConfig` (rho, beta_max, eigenvalue floor and cap, symmetry and PSD tolerances), `HessianInformedProposal` (symmetry check, `eigh` of the symmetrised Hessian, PSD check, eigenvalue clipping, proposal, exact log density, serialisation), `HpCNChain` (restartable chain: point, Phi, NumPy PCG64 state, counters, timing, proposal), atomic `save_checkpoint` / `load_checkpoint` |
| `src/qoi.py` | effective permeability Q(m) = -∫ exp(m) dp/dx dx (flux through the cell) |
| `scripts/m4_hpcn.py` | `prepare` (P-MAP and Gauss-Newton Hessian at the MAP, saved in `results/m4/hessian_*.json`), `chain` (configurable rho, beta_max, eigenvalue floor, seed, start rule, length, thinning, checkpoint and diagnostic frequency; resumes automatically), `qoi` |
| `scripts/m4_analyze.py` | pilot selection, go/no-go and production analysis |
| `scripts/make_chain_manifest.py` | now also reads Milestone 4 checkpoint metadata |
| `src/pcn_sampler.py` | **unchanged** (baseline) |

Gauss-Newton Hessians at the MAP (14 x 14): eigenvalues 0.0016 to 8,780 (smooth) and 0.0011 to 7,400 (structured); relative
asymmetry recorded in `results/m4/hessian_*.json`. No new dependencies.

---

## 3. Verification (Phase 3)

`tests/test_hessian_pcn.py` (13 tests) and `tests/test_qoi.py` (3):

| # | Requirement | Test(s) | Result |
|---|---|---|---|
| 1 | Prior invariance | A = A^T and A^2 + S = I to 1e-12; one proposal step from 50,000 prior draws returns N(0, I) (covariance within 0.03, KS); zero-likelihood chain has acceptance 1 and unit variance; control: a preconditioned random walk violates the identity | pass |
| 2 | Detailed balance / MH | mu_0(x)q(x,x') = mu_0(x')q(x',x) for 50 random pairs (to 1e-9); full MH kernel symmetric for a nonlinear Phi; with H = 0 the chain reproduces the baseline `run_pcn` **bit for bit** | pass |
| 3 | Gaussian posterior moments | 6-dimensional linear-Gaussian ridge (singular values 300 to 0.3, plus one exactly unidentified direction): 4 chains, R-hat < 1.01, bulk ESS > 1,000 per coordinate, means within 4 MCSE of the analytic mean, covariance within 5% of the largest variance; at equal cost HpCN has more than 20x the minimum ESS of pCN | pass |
| 4 | Constant mode | sampler dimension r-1 = 14; Phi of a draw identical for any xi_c | pass |
| 5 | Likelihood agreement | reduced Phi == forward-only `neg_log_likelihood` == legacy forward+adjoint path (exact equality) | pass |
| 6 | Seeded reproducibility | identical chains for identical seeds | pass |
| 7 | Checkpoint/resume | 150 + checkpoint + 250 == 400 uninterrupted (draws, Phi, counters, bitwise); also verified across separate processes with the production script | pass |
| 8 | Finite outputs | proposals with non-finite Phi are rejected and counted; draws stay finite; invalid Hessians and configs are rejected; round-off negative eigenvalues are clipped | pass |
| 9 | Deterministic results preserved | Milestone 3 preconditioned headline reconstruction: objective 40.16605132 (rel 1e-8), mean-free error 0.3468 | pass |
| QoI | effective permeability | homogeneous media exact; Q(m+c) = e^c Q(m); layered media equal the arithmetic and harmonic means (0.5%) | pass |

Development notes: two test set-ups were corrected before any pilot, without loosening an accuracy threshold.
(a) The prior-sampling test used Hessian eigenvalues up to 9e4, so a 40,000-step prior chain could not explore those
directions; it now uses moderate eigenvalues. (b) The moment and efficiency tests started at prior-scale offsets, and
the chains were still descending in the stiffest direction (posterior sd 0.003) during the "post-warmup" draws; they now start
over-dispersed relative to the posterior (3x Laplace scale). That diagnosis is the basis of Deviation 1.

**Full suites:** fenicsx **164 passed, 15 skipped** (PINN tests, torch absent); `.venv-deepxde` **62 passed**.

---

## 4. Pilot (Phase 4; `results/m4/pilot_summary.json`)

4 chains x 4,000 proposals per (truth, rho), started over-dispersed at xi_MAP + 3 L eta with L L^T = (I + H)^-1;
first half discarded. Reference: the first 4,000 proposals of the four Milestone 3 pCN chains (same protocol; prior-scale starts).

| Truth | Sampler | rho / beta | acceptance | max R-hat | min bulk ESS | min tail ESS | min bulk ESS per s | CPU s (4 chains) | peak RSS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| smooth | HpCN | 0.25 | 0.61 | 1.191 | 15.5 | 61 | 0.053 | 293 | 205 MB |
| smooth | HpCN | 0.5 | 0.33 | 1.107 | 30.5 | 108 | 0.115 | 264 | 203 MB |
| smooth | **HpCN** | **0.75** | 0.17 | **1.045** | **120.7** | 144 | **0.438** | 276 | 210 MB |
| smooth | HpCN | 1.0 | 0.08 | 1.136 | 22.7 | 28 | 0.089 | 254 | 209 MB |
| smooth | pCN | beta 0.01 | 0.64 | 3.717 | 4.5 | | 0.0198 | 228 | |
| structured | HpCN | 0.25 | 0.59 | 1.189 | 16.5 | 21 | 0.059 | 281 | 209 MB |
| structured | **HpCN** | **0.5** | 0.32 | **1.048** | **70.8** | 108 | **0.258** | 275 | 205 MB |
| structured | HpCN | 0.75 | 0.15 | 1.163 | 18.1 | 18 | 0.069 | 263 | 207 MB |
| structured | HpCN | 1.0 | 0.07 | 1.154 | 32.2 | 32 | 0.127 | 253 | 208 MB |
| structured | pCN | beta 0.02 | 0.44 | 3.142 | 4.7 | | 0.0198 | 235 | |

Selection by minimum bulk ESS per second, not acceptance: rho = 0.75 (smooth) and 0.5 (structured), with ratios over pCN of
**22.1** and **13.0** (threshold 10: **GO**). The best settings accept only 17% and 32% of proposals. Pilot estimates rest on 16 to 120
effective draws and are noisy (the best rho differs between truths).

*Burn-in from prior-scale starts* (rho 0.75, 4 chains x 8,000): Phi had not settled below the 99th percentile of its
stationary range after 6,600 to 8,000 proposals in 7 of 8 chains. Over-dispersion relative to the Laplace posterior (Deviation 1) is
essential for HpCN, whose steps in the stiff directions are about 0.01 prior units.

Pilot cost: 40 chains, 192,000 proposals, 192,040 forward solves, 0 adjoint solves, 3,185 CPU s, 839 s wall.

---

## 5. Production (Phase 5; `results/m4/production_summary.json`)

Fixed before launch: 4 chains per truth, seeds 11 to 14, Laplace x3 over-dispersed starts, 50,000 proposals per chain (the
pre-registered cap was binding; the rule asked for 60,000), thin 5, 50% warmup, no extension. Monitored: 14
coefficients, Phi, m_R at 5 probes, ||m_R||_L2, log Q_R (21 quantities).

| Criterion | Smooth (rho 0.75) | Structured (rho 0.5) |
|---|---|---|
| max rank-normalised split R-hat < 1.01 | **1.0063** (xi_R[2]) | **1.0130** (xi_R[6]); fails |
| min bulk ESS >= 400 | 840 | 633 |
| min tail ESS >= 400 | 876 | 509 |
| derived quantities (Phi, ‖m_R‖, log Q_R): R-hat / bulk ESS | 1.004 / 1,261; 1.002 / 3,053; 1.002 / 3,213 | 1.002 / 1,534; 1.003 / 2,063; 1.003 / 2,129 |
| chain agreement, calibrated max abs z <= 4 (Deviation 2) | 2.37 | 2.59 |
| original pairwise statistic (reported only) | 7.64 | 6.59 |
| constant mode: exact variance 1; independent draws KS p | 1.0000; 0.083 | 1.0000; 0.756 |
| non-finite likelihoods; checkpoint consistency | 0; consistent | 0; consistent |
| max MCSE(mean)/sd | 0.036 | 0.042 |
| **Converged on all pre-registered criteria** | **yes** | **no** (one of 21 quantities, R-hat 1.0130) |

The structured shortfall is narrow and confined to one coefficient. Its bulk ESS is 722 and every other quantity is below 1.01,
but by the pre-registered rule this is **failure to establish convergence within the budget**, and the run was not extended.
The original pairwise chain-agreement statistic would have failed even the smooth run (7.6), which confirms that it was ill-posed (Deviation 2).
Trace and autocorrelation plots: `results/m4/figures/hpcn_traces_{smooth,structured}.png` (autocorrelation decays within about 200
to 400 proposals).

Cost: 8 chains, 400,000 proposals and 400,008 forward solves (0 adjoint), plus 80,000 forward solves for log Q_R; 7,227 CPU s in total
(sampling 2,798 s smooth, 2,749 s structured, 13.7 to 14.0 ms per proposal; post-processing 190 to 230 s per chain);
**1,818 s wall**; peak memory 192 to 210 MB per process; chain archives 34.6 MB (outside Git).

---

## 6. Posterior results

| Quantity | Smooth | Structured |
|---|---|---|
| integrated variance: mean-free + constant (exact) = total | 0.673 + 1 = **1.673** | 0.705 + 1 = **1.705** |
| pointwise total variance | 1.24 to 2.60 (floor 1 respected) | 1.24 to 2.61 |
| posterior mean, relative L2 error, mean-free / contrast | 0.538 / 0.281 | 0.870 / 0.406 |
| MAP, relative L2 error, mean-free (Milestone 3) | 0.376 | 0.768 |
| log k_eff: truth / posterior median [95%] | -0.059 / 0.206 [-1.81, 2.27] | -0.021 / 0.275 [-1.74, 2.33] |
| log k_eff, identifiable part only: posterior sd | 0.28 | 0.28 |

- The width of the k_eff interval is almost entirely the unidentifiable global scale (sd 1 vs 0.28). Pressure data alone cannot
  determine the absolute flux through the cell. This is a substantive physical statement that the corrected treatment makes explicit.
- The posterior mean is further from the truth than the MAP. With converged chains, this is a property of the posterior (skewness
  and the prior's pull in weakly informed directions), not a sampling artefact. One synthetic realisation does not say which estimator is preferable.
- At all five probe points, for both truths, the true mean-free value lies inside the mean-free 95% interval. This is one realisation,
  **not** a calibration result.
- The structured posterior is not certified as converged (Section 5), so its numbers carry that qualification.

---

## 7. Comparison with plain pCN (Phase 6)

pCN: the Milestone 3 production chains (4 x 100,000 proposals per truth, prior-scale starts, beta 0.01 / 0.02, analytic constant
mode). HpCN: Milestone 4 production (4 x 50,000). The budgets differ by a factor of 2 in PDE solves, so efficiency is compared
per second and per PDE solve.

| Metric | Smooth: pCN | Smooth: HpCN | Structured: pCN | Structured: HpCN |
|---|---:|---:|---:|---:|
| Acceptance rate | 0.648 | 0.165 | 0.443 | 0.309 |
| Max R-hat | 1.825 | **1.006** | 1.649 | 1.013 |
| Min bulk ESS | 5.9 | **840** | 6.7 | **633** |
| Min tail ESS | 11.8 | **876** | 13.4 | **509** |
| Min bulk ESS per second | 0.00103 | **0.300** (291x) | 0.00114 | **0.230** (202x) |
| Min bulk ESS per 1,000 PDE solves | 0.0147 | **4.20** (285x) | 0.0168 | **3.16** (189x) |
| Total PDE solves (sampling) | 400,040 | 200,004 | 400,040 | 200,004 |
| Sampler CPU time | 5,713 s | 2,798 s | 5,876 s | 2,749 s |
| Constant-mode integrated variance | 1 (exact) | 1 (exact) | 1 (exact) | 1 (exact) |
| Mean-free integrated variance | 0.361 (not converged) | 0.673 | 0.565 (not converged) | 0.705 |
| Posterior-mean rel. error, mean-free L2 | 1.449 (not converged) | 0.538 | 1.069 (not converged) | 0.870 |

Evidence of superiority: with half the PDE solves, HpCN attains R-hat 1.006 / 1.013 against 1.83 / 1.65, and 94 to 142 times the
minimum bulk ESS. Even allowing for the large relative uncertainty of an ESS estimate near 6 (plausibly a factor of 2), the efficiency
gain per second and per PDE solve is at least about 100x for both truths. pCN's mean-free variance and posterior mean are not
converged and are shown only for completeness. Its monitored set excluded ‖m_R‖ and log Q_R.

---

## 8. Reproducibility

```bash
PY=/opt/anaconda3/envs/fenicsx/bin/python; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
for t in smooth structured; do $PY scripts/m4_hpcn.py prepare --truth $t; done
# pilot (run as up to 4 separate processes; queue in results/m4/logs/pilot_queue.txt)
$PY scripts/m4_hpcn.py chain --truth smooth --rho 0.75 --seed 1 --init laplace3 --n-proposals 4000 \
    --checkpoint-every 1000 --diagnostic-every 2000 --out results/m4/chains/pilot/smooth_r0.75_laplace3_s1.npz
$PY scripts/m4_analyze.py pilot
# production (seeds 11-14; rho 0.75 smooth, 0.5 structured)
$PY scripts/m4_hpcn.py chain --truth smooth --rho 0.75 --seed 11 --init laplace3 --n-proposals 50000 --thin 5 \
    --checkpoint-every 5000 --diagnostic-every 10000 --out results/m4/chains/production/smooth_s11.npz
$PY scripts/m4_hpcn.py qoi --truth smooth --chain results/m4/chains/production/smooth_s11.npz
$PY scripts/m4_analyze.py production
shasum -a 256 -c results/m4/chains/MANIFEST.sha256      # 56 archives
```

An interrupted chain resumes when the same `chain` command is repeated. Archives and provenance: `results/m4/chains/MANIFEST.json`
(path, bytes, SHA-256, seed, rho, start rule, counters, solve counts and timing per chain; source-file SHA-256 prefixes at generation).
The chains were produced by the uncommitted Milestone 4 code on top of `40c8b76`.

---

## 9. Remaining limitations

1. The structured posterior is not certified as converged (R-hat 1.0130 for one coefficient at the pre-set cap).
2. The proposal is fixed at the MAP's Gauss-Newton Hessian. A posterior that deviates strongly from its Laplace approximation would
   reduce efficiency (not observed here; the acceptance of 17 to 32% is acceptable).
3. HpCN needs starts that are over-dispersed relative to the Laplace posterior; from prior-scale starts its burn-in exceeds 8,000 proposals.
4. One data realisation and one truth draw; no calibration, repeated-noise or mesh-refinement study (excluded from this milestone).
5. Production and pilot estimates are single realisations of the MCMC: the rho selection between 0.5 and 0.75 is not statistically
   resolved.
6. Chain archives (34.6 MB, Milestone 3 19.3 MB) exist only in this local working copy; there is no off-machine backup.

---

## 10. Files

New: `src/hessian_pcn.py`, `src/qoi.py`, `scripts/m4_hpcn.py`, `scripts/m4_analyze.py`, `tests/test_hessian_pcn.py`,
`tests/test_qoi.py`, `docs/audit/M4_PREREGISTRATION.md`, this report, `results/m4/` (Hessians, pilot and production summaries, figures,
logs, chain manifest). Modified: `scripts/make_chain_manifest.py`. Committed in Milestone 5, Phase A (commit list in `docs/audit/MILESTONE_5_REPORT.md`); the sampler sources were verified unchanged against the SHA-256 prefixes recorded at chain generation.

---

## 11. Decision

**CONDITIONAL GO.** The sampler is mathematically valid (exact prior reversibility, verified detailed balance, exact constant mode),
reproducible (bitwise checkpoint/resume), and about 200 to 290 times more efficient per second than plain pCN. It established
converged posterior inference for the smooth truth within a pre-set budget of 30 min wall. For the structured truth it narrowly missed one
criterion (one coefficient at R-hat 1.013) and convergence is therefore not established. Adopting HpCN as the project's sampler is
justified. Certifying the structured posterior requires one more pre-registered run.

**Recommended next milestone (Milestone 5):**

1. A pre-registered confirmation run for the structured truth: 4 new chains x 100,000 proposals (about 25 min wall; predicted min
   bulk ESS about 1,300), same criteria, no other changes.
2. Then posterior calibration in the well-specified setting (truths drawn from the prior; coverage and simulation-based calibration of
   identifiable quantities). This is now affordable: one converged posterior costs about 45 CPU min (4 x 50,000 proposals). Start with a Laplace-based
   check, then MCMC on a subset of replicates.
3. Commit and push the Milestone 4 work after review, and make an off-machine backup of the chain archives.
