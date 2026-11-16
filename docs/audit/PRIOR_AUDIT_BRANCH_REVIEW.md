# Review of the earlier audit branch `audit/scientific-audit`

Reviewed: 2026-11-16, during Milestone 2.
Branch: `origin/audit/scientific-audit` (3 commits on top of `aa4b568`, dated 2026-10-07), open as draft PR #1
"Audit Darcy inversion and qualify unconverged MAP output". The branch is 34 commits behind `main`.

The branch was **not merged and not deleted**. Its history remains intact on the remote. This note
records what it contains, what the Milestone 1 audit supersedes, and which unique content has been carried
forward.

## Contents

| Commit | Content |
|---|---|
| `5f2dfbb` | `scripts/run_posterior_demo.py`: label an unconverged optimiser endpoint as "MAP candidate (optimizer not converged)" in the posterior-field figure and in the text output |
| `3bf96f5` | `docs/scientific_audit.md`: component verdicts (forward problem, FEM, adjoint, KL prior, pCN, MAP terminology, structured-truth misspecification) |
| `9d787f8` | `HANDOFF_BAYESIAN_VALIDATION.md`: evidence summary and a staged plan (MAP diagnostics, pCN beta study, multiple chains) |

## Comparison with the Milestone 1 audit

| Topic | Earlier audit | Milestone 1 / 2 status |
|---|---|---|
| Forward problem, FEM, adjoint, Taylor tests | supported | same verdict; superseded |
| pCN algorithm and linear-Gaussian validation | supported, practically under-mixed | same verdict, now quantified (split R-hat up to 3.4); superseded |
| MAP `converged=False` | "MAP candidate" terminology | same; now with full termination diagnostics (Milestone 2) |
| Structured truth as prior misspecification | genuine negative result | same; superseded |
| Constant-mode non-identifiability | **not identified** | new in Milestone 1; tested in Milestone 2 |
| Stored variances below the exact bound | **not identified** | new in Milestone 1; tested in Milestone 2 |
| ESS lag-cap bias, flawed predictive check, unconverged Part I studies | not identified | new in Milestone 1 |
| Earlier audit described intervals as "plausible exploration" | | superseded: intervals are provably too narrow |

## Unique content carried forward

1. **MAP-candidate labelling (code).** Ported unchanged into `scripts/run_posterior_demo.py` with
   `git diff main...origin/audit/scientific-audit -- scripts/run_posterior_demo.py | git apply`. It changes
   only figure titles and the text label; no numerical value changes. `scripts/compare_baselines.py`
   normalises this label when comparing with the historical text.
2. **Observation: local and full MAP candidates agree although their caps differ (80 vs 120).** This is
   consistent with a stalled or nearly stationary optimiser and is now testable with the recorded gradient norms
   (Milestone 2 report, Section 5.6). Stored values: MAP-candidate relative error 0.3866 (local) and 0.3868 (full) for the smooth truth.
3. **Methodological guidance kept for Milestones 5 and 6:** compare samplers by ESS per forward solve and
   wall time, not by acceptance rate; run a controlled beta study and multiple chains before longer chains;
   do not loosen optimiser tolerances to obtain `success=True`; keep plain pCN as the validated baseline.

## Recommendation (requires approval; not done)

Close draft PR #1 with a comment pointing to `docs/audit/MILESTONE_1_AUDIT.md`, `docs/audit/MILESTONE_2_REPORT.md`
and this note, keeping the branch for history. This needs a push of the audit documents first.
