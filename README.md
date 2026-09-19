# Darcy-Flow Inverse Problem: Adjoint-Based Log-Permeability Reconstruction

A deterministic, regularised inverse-problems study: recover a spatially varying permeability field
from sparse, noisy pressure measurements of a 2D Darcy (groundwater/reservoir) flow, using an
adjoint-derived gradient verified by a Taylor-remainder test, L-BFGS optimisation, and explicit
studies of regularisation strength, observation noise, and sensor density/placement.

## Overview

This is primarily an applied-mathematics / inverse-problems project, not a software-engineering
demonstration: the point is to show the complete workflow **elliptic PDE → finite-element forward
solver → parameter-to-observable map → ill-posed inverse problem → regularisation → PDE-constrained
optimisation (adjoint method) → noise/sensor sensitivity analysis → honest quantitative evaluation**,
using established mathematics throughout rather than a new method. No neural network, PINN, or
learned surrogate is used anywhere in this repository (that is a deliberate scope boundary set by the
project brief; a companion Bayesian extension of this same forward/inverse machinery is planned as a
separate project and is explicitly out of scope here).

## Mathematical forward problem

$$-\nabla\cdot(k(x)\nabla p(x)) = f(x) \quad \text{in } \Omega = [0,1]^2,$$

with $p=1$ prescribed on the left edge ($x=0$), $p=0$ on the right edge ($x=1$), and no-flow
(natural, $k\,\partial p/\partial n = 0$) on the top and bottom edges — a standard "flow cell" driving
left-to-right flow through a heterogeneous medium, as used throughout the groundwater/reservoir
inverse-problem literature (e.g. Stuart, A. M. (2010), "Inverse problems: a Bayesian perspective",
*Acta Numerica*). $f=0$ for the physical inverse problem (used only for manufactured-solution
verification, see below).

**Log-permeability parameterisation.** Permeability must stay strictly positive. Rather than
constrain the optimisation variable directly, we parameterise $m = \log k$, so $k = \exp(m)$ is
automatically positive for *any* real-valued $m$ — turning a constrained inverse problem into an
unconstrained one, standard practice for this problem class. $m$ is discretised on a continuous
piecewise-linear (P1) finite-element space.

**Parameter-to-observable map.** $m \mapsto p(m) \mapsto Hp(m)$, where $p(m)$ solves the weak form
below and $H$ is the sparse-pressure observation operator (see "Observation model"). Synthetic
observations are $y = Hp(m_{\mathrm{true}}) + \varepsilon$, $\varepsilon \sim \mathcal N(0,\Gamma)$.

## Weak formulation

Find $p \in H^1_g(\Omega)$ (with $g$ the Dirichlet data above) such that for all $v \in H^1_0(\Omega)$:

$$a(m)(p,v) := \int_\Omega \exp(m)\,\nabla p\cdot\nabla v\,dx = \int_\Omega f v\,dx =: L(v).$$

Pressure is discretised with continuous piecewise-quadratic (P2) elements. This weak-form/residual
reading of the forward problem, $F(p,m;v) := a(m)(p,v) - L(v) = 0\ \forall v$, is exactly what the
adjoint derivation below differentiates.

## Why this is an inverse problem, and why it is ill-posed

Recovering $m$ from sparse, noisy pressure data is a classic **ill-posed inverse problem** in the
Hadamard sense: a solution may fail to exist (for inconsistent noisy data), fail to be unique (many
different $m$ fields can produce nearly the same *sparse* pressure observations — pressure is a
smoothing, diffusive functional of the much rougher permeability field, so high-frequency / localised
structure in $m$ is strongly attenuated in $p$ before it ever reaches a handful of sensors), and fail
to depend continuously on the data (small data perturbations, i.e. noise, can produce large, physically
implausible changes in the reconstructed $m$ if unregularised). This third failure mode — instability
under regularisation-free inversion — is demonstrated directly, not just asserted: **Reconstruction
Results** and the **Regularisation study** below show fits with near-zero data misfit whose recovered
$m$ is nonetheless a poor match to the truth, the textbook signature of overfitting noise through an
ill-posed operator. Regularisation is therefore not a numerical convenience layered on top of the
mathematics; it is what converts an ill-posed problem into a well-posed one by adding side information
(smoothness, or closeness to a prior) that the data alone cannot supply.

## Regularised inverse formulation

$$J(m) = \tfrac12 \|Hp(m)-y\|^2_{\Gamma^{-1}} + \tfrac{\alpha}{2} R(m), \qquad \Gamma = \mathrm{diag}(\sigma_1^2,\dots,\sigma_n^2).$$

Two standard quadratic (Tikhonov-type) regularisers are implemented (`src/regularization.py`):

- **H1-seminorm** (default throughout the experiments below): $R(m) = \|\nabla m\|_{L^2}^2$, penalising
  spatial roughness directly; constant shifts of $m$ are unpenalised.
- **L2-to-prior**: $R(m) = \|m-m_0\|_{L^2}^2$, penalising deviation from a reference field $m_0$
  (default $m_0=0$) without penalising roughness at all.

## Observation model

Sensors measure a small, narrow Gaussian-weighted local average of pressure around each sensor
location, **not** a literal Dirac point evaluation: $H(p)_s = \int_\Omega p(x)\,\psi_s(x)\,dx$, with
$\psi_s$ a Gaussian bump normalised to $\int\psi_s\,dx=1$. This is a genuine bounded linear functional
on $H^1$ (a true point evaluation is not, in general), arguably a more physically honest model of a
real sensor (which also integrates over some small volume), and — crucially for the adjoint method —
built directly from the same finite-element test-function weights used to assemble the pressure
equation, so its adjoint $H^\top$ requires no separate implementation (`src/observation.py`). For a
bump width small relative to the domain but several mesh cells wide, this closely approximates a
point measurement.

## Adjoint derivation

Introduce the Lagrangian $L(m,p,\lambda) = J_{\mathrm{misfit}}(p) + \tfrac{\alpha}{2}R(m) + F(p,m;\lambda)$.
Stationarity with respect to $p$ gives the **adjoint equation**: find $\lambda \in H^1_0$ such that for
all $\delta p \in H^1_0$,

$$\int_\Omega \exp(m)\,\nabla\delta p\cdot\nabla\lambda\,dx = -(H\delta p)^\top \Gamma^{-1}(Hp-y).$$

The left-hand side is *exactly* the same bilinear form as the forward problem — the Darcy operator is
self-adjoint — so the adjoint equation is "solve the same PDE again, with homogeneous Dirichlet data
($\lambda=0$ where $p$ was prescribed) and a source built from the weighted observation residual
instead of $f$" (the source is exactly `ObservationOperator.adjoint_rhs_vector`, a weighted sum of the
same bump-integral weight vectors $H$ is built from). Stationarity with respect to $m$, using
$k(m)=\exp(m)$, $dk/dm=k$, gives the **gradient**:

$$\frac{dJ}{dm}[\delta m] = \int_\Omega \exp(m)\,\delta m\,\nabla p\cdot\nabla\lambda\,dx + \alpha\,R'(m)[\delta m].$$

Evaluated against the P1 nodal basis, this is assembled directly as a UFL linear form — the $\exp(m)$
chain rule and $R'(m)$ are both obtained via UFL's own automatic symbolic differentiation
(`ufl.derivative`), not hand-differentiated, mirroring the Newton-Jacobian approach used throughout
the companion `fem-cylinder-flow` project. Full derivation with inline commentary: `src/adjoint.py`.

## Gradient verification

**The adjoint gradient is not used for anything downstream until it passes this check**
(`scripts/verify_gradient.py`). For a random direction $\delta m$, the first-order Taylor remainder
$r(\epsilon) = |J(m+\epsilon\delta m) - J(m) - \epsilon\,\nabla J(m)\cdot\delta m|$ must shrink as
$O(\epsilon^2)$ if the gradient is correct (Farrell, Ham, Funke & Rognes, 2013, "Automated derivation
of the adjoint of high-level transient finite element programs", *SIAM J. Sci. Comput.* 35(4)):

| $\epsilon$ | zeroth-order $\lvert\Delta J\rvert$ | order | 1st-order remainder | order |
|---:|---:|---:|---:|---:|
| $10^{-2}$ | $2.9107\times10^{-2}$ | — | $1.4475\times10^{-5}$ | — |
| $5\times10^{-3}$ | $1.4550\times10^{-2}$ | 1.000 | $3.6184\times10^{-6}$ | **2.000** |
| $2.5\times10^{-3}$ | $7.2741\times10^{-3}$ | 1.000 | $9.0450\times10^{-7}$ | **2.000** |
| $1.25\times10^{-3}$ | $3.6368\times10^{-3}$ | 1.000 | $2.2608\times10^{-7}$ | **2.000** |
| $6.25\times10^{-4}$ | $1.8184\times10^{-3}$ | 1.000 | $5.6499\times10^{-8}$ | **2.001** |

(H1 regulariser, $\alpha=10^{-3}$; the L2-to-prior regulariser gives an identical clean order-2.000
remainder, see `results/gradient_verification.txt`.) Both the zeroth-order check (order → 1, confirming
$J$ itself is Lipschitz/differentiable along the direction) and the informative second-order check
(order → 2, confirming the gradient is *correct*, not merely finite) pass cleanly.

## Optimisation method

L-BFGS (`scipy.optimize.minimize(method="L-BFGS-B")`), the standard quasi-Newton method for
large-scale smooth unconstrained optimisation with cheap gradients but no cheap Hessian (Nocedal &
Wright, 2006, *Numerical Optimization*, Ch. 6) — exactly this situation, since the adjoint method
gives an exact gradient at the cost of one extra PDE solve, while forming the Hessian would need many
more. Every iteration's objective, data misfit, regularisation term, and gradient norm are recorded
(`src/inversion.py::OptimizationHistory`); stopping criteria are `gtol=1e-6` (gradient norm),
`ftol=1e-12` (objective change), `maxiter` per config (see below) — never "ran until it looked done."

## Forward-solver verification

The cylinder-flow project's standard applies here too: **the forward solver is verified before any
inversion is trusted, and inverse-reconstruction quality is never used as evidence the forward solver
is correct.** A manufactured solution (`src/manufactured.py`, UFL-symbolic forcing, no hand-derived
algebra) gives two informative convergence studies (`scripts/verify_forward.py`):

**(A) Exact (non-interpolated) coefficient** — isolates the P2 pressure space's own accuracy:

| $N$ | $L^2$ error | order | $H^1$ error | order |
|---:|---:|---:|---:|---:|
| 8 | $5.489\times10^{-4}$ | 2.985 | $3.344\times10^{-2}$ | 1.958 |
| 16 | $6.877\times10^{-5}$ | 2.997 | $8.423\times10^{-3}$ | 1.989 |
| 32 | $8.601\times10^{-6}$ | 2.999 | $2.110\times10^{-3}$ | 1.997 |
| 64 | $1.075\times10^{-6}$ | **3.000** | $5.277\times10^{-4}$ | **1.999** |

matching standard elliptic FEM theory exactly ($O(h^3)$/$O(h^2)$ for $L^2$/$H^1$ with P2 elements, the
$L^2$ rate one order higher via the Aubin-Nitsche duality argument).

**(B) P1-interpolated coefficient** — the *realistic* configuration used everywhere else in this
repository, since $m$ is deliberately P1 for the inverse problem:

| $N$ | $L^2$ error | order | $H^1$ error | order |
|---:|---:|---:|---:|---:|
| 8 | $2.780\times10^{-3}$ | 1.770 | $3.967\times10^{-2}$ | 1.904 |
| 16 | $7.276\times10^{-4}$ | 1.934 | $1.008\times10^{-2}$ | 1.976 |
| 32 | $1.841\times10^{-4}$ | 1.983 | $2.530\times10^{-3}$ | 1.994 |
| 64 | $4.616\times10^{-5}$ | **1.996** | $6.332\times10^{-4}$ | **1.999** |

Both norms cap at $O(h^2)$, not the P2 space's textbook $O(h^3)$/$O(h^2)$. This was diagnosed, not
assumed: raising the quadrature degree left the observed rate unchanged (ruling out quadrature error),
while switching to the exact coefficient (variant A) immediately restored $O(h^3)$. The cause is the
P1 interpolation error in the coefficient $k=\exp(m_h)$ itself, which dominates the discretisation
error regardless of the (higher-order) pressure space. Since $m$ genuinely is P1 throughout the rest
of this repository, variant (B) — not the more flattering variant (A) — is the solver's actual verified
behaviour, and is reported as such.

## Synthetic experiment design: truth fields

Two qualitatively different truths (`src/truth_fields.py`), so reconstruction difficulty is shown to
depend on the *class* of unknown field, not a single cherry-picked case:

- **Smooth**: a truncated Karhunen-Loeve-like random field — a finite sum of low-frequency cosine
  modes with amplitudes decaying with wavenumber (seeded, reproducible).
- **Structured**: two localised, smoothly-blended circular inclusions (one high-, one
  low-permeability) on a uniform background, modelling channels/lenses; deliberately not representable
  by a handful of smooth global modes.

## Avoiding the inverse crime

Synthetic truth pressure — and hence the observations $y$ — is generated on a mesh **strictly finer**
than any mesh used later for inversion (`src/data_generation.py`): truth meshes use $N=24/40/64$
(smoke/local/full), inversion meshes use $N=12/20/32$. Using the same discretisation for both would
let the inversion implicitly match a discretisation artefact rather than genuine physics, and could
mask bugs that would surface against more realistic data (Kaipio & Somersalo, 2007, "Statistical
inverse problems: discretization, model reduction and inverse crimes", *J. Comput. Appl. Math.* 198(2)).
The true field is defined as a plain coordinate-based Python function, not tied to either mesh, so it
can be evaluated consistently at both resolutions.

## Reconstruction results

`python scripts/run_reconstruction_demo.py --config full` — 6×6 sensor grid (36 sensors), truth mesh
$N=64$, inversion mesh $N=32$ (1,089 $m$-dofs / 4,225 $p$-dofs), 2% relative noise,
$\alpha=30$ (chosen by the regularisation study below, **not** re-tuned per truth):

| Truth | rel. $m$ error | final misfit | final reg. | L-BFGS iterations | converged | time |
|---|---:|---:|---:|---:|---:|---:|
| smooth | **0.349** | 9.627 | 30.54 | 138 | yes | 12.0 s |
| structured | **0.849** | 24.07 | 14.22 | 125 | yes | 10.1 s |

(figures: `figures/reconstruction_{smooth,structured}_full.png`, true/reconstructed/error fields with
sensor locations, same colour scale for true and reconstructed; convergence histories in
`figures/convergence_{smooth,structured}_full.png`.) The smooth truth's broad left-to-right structure
is recovered reasonably well (though vertical variation orthogonal to the dominant flow direction is
visibly under-recovered — see "Sensor-density/placement study" for why). **The structured truth is
recovered very poorly at this $\alpha$**: the reconstruction is nearly featureless, missing both
inclusions almost entirely. This is not a bug or a mesh issue — the same qualitative failure appears
at `local` resolution (rel. error 0.875, `figures/reconstruction_structured_local.png`) — and a
supplementary $\alpha$-sweep specifically for the structured truth (0.01 to 30) found the error stays
in the 0.82–0.89 range across the *entire* range, i.e. **no choice of H1-seminorm regularisation
strength recovers these inclusions well**. This is an honest, informative negative result, not a
parameter-tuning failure: H1-seminorm regularisation directly penalises the sharp gradients that
define a localised inclusion, so it structurally biases reconstructions toward smooth fields regardless
of $\alpha$; recovering sharp, localised structure would need a different regulariser (e.g. total
variation) or a prior informed by the true field's structure — genuine limitations of the
Tikhonov-type regularisers implemented here, stated plainly rather than hidden by only showing the
smooth-truth result.

## Regularisation study

`python scripts/regularization_study.py --config full` — smooth truth, 6×6 sensors, 2% noise, 9 values
of $\alpha$ from $10^{-1}$ to $10^5$:

| $\alpha$ | misfit | reg. | rel. $m$ error (**retrospective**) |
|---:|---:|---:|---:|
| $10^{-1}$ | 0.055 | 0.593 | 0.534 |
| $1$ | 1.283 | 2.938 | 0.434 |
| $10$ | 6.275 | 12.14 | 0.354 |
| **30** | **9.627** | **30.54** | **0.349** |
| $10^2$ | 19.18 | 85.62 | 0.351 |
| $3\times10^2$ | 53.03 | 202.0 | 0.374 |
| $10^3$ | 189.3 | 441.9 | 0.451 |
| $10^4$ | 1462 | 550.3 | 0.783 |
| $10^5$ | 2583 | 101.2 | 0.970 |

a textbook bias-variance U-curve in the retrospective error (figure: `figures/regularization_study_full.png`,
L-curve and error-vs-$\alpha$ side by side). **The retrospective-best column requires $m_{\mathrm{true}}$
and is never available for a real inverse problem** — it is reported here purely to show what the
(unreachable in practice) best case looks like.

**Practical, truth-free selection: Morozov's discrepancy principle.** Choose the largest $\alpha$
whose converged weighted misfit stays at or below the statistically expected value under the assumed
noise model, $\tfrac12\sum_s((Hp-y)_s/\sigma_s)^2 \le \tfrac12 n_{\mathrm{sensors}}$ (Morozov, 1966;
Engl, Hanke & Neubauer, 1996, Ch. 4.3) — uses only the assumed noise level, never the unknown truth.
For $n=36$ sensors the target is misfit $\le 18.0$; this selects $\alpha=30$, which is **exactly the
retrospective-best value found independently at all three tested resolutions** (smoke: $n=16$,
discrepancy target 8.0, selects $\alpha=10$ = retrospective best; local: $n=25$, target 12.5, selects
$\alpha=30$ = retrospective best; full: as above) — a genuinely validated, non-cherry-picked agreement,
not an artefact of one lucky run.

## Noise sensitivity

`python scripts/noise_study.py --config full` — smooth truth, 6×6 sensors, $\alpha=30$ fixed, 5
relative-noise levels:

| relative noise | rel. $m$ error | misfit / $n_{\mathrm{sensors}}$ |
|---:|---:|---:|
| 0.00 | 0.346 | 0.0154 |
| 0.01 | 0.354 | 0.271 |
| 0.05 | 0.372 | 0.528 |
| 0.10 | 0.442 | 0.592 |
| 0.20 | 0.616 | 0.610 |

(figure: `figures/noise_study_full.png`.) Monotonic, honestly reported degradation — no instability is
hidden. (An earlier version of this study had a real bug here: the noise-weighting $\sigma$ was floored
at $10^{-12}$ for the noiseless case, making the misfit term explode to $O(10^{11})$ and destabilising
the noiseless-case optimisation; fixed by using a sane nominal noise floor — assumed measurement
precision, not literal injected noise — for the weighting even when the synthetic data happen to be
exact. Regression test: `tests/test_data_generation.py::test_sigma_uses_nominal_floor_for_noiseless_data`.)

## Sensor-density/placement study

`python scripts/sensor_study.py --config full` — smooth truth, $\alpha=30$, 2% noise, comparing
regularly-spaced ("grid") placement against 5 independent random-placement trials at matched sensor
counts:

| $n_{\mathrm{sensors}}$ | grid (regular placement) | random (mean $\pm$ std, 5 trials) |
|---:|---:|---:|
| 4 | 0.618 | $0.385 \pm 0.014$ |
| 9 | 0.366 | $0.374 \pm 0.008$ |
| 25 | 0.353 | $0.359 \pm 0.009$ |
| 49 | 0.354 | $0.358 \pm 0.008$ |
| 81 | 0.351 | $0.357 \pm 0.007$ |

(figure: `figures/sensor_study_full.png`.) Two clear findings, neither of which is an identifiability
claim beyond what was actually tested:

- **Diminishing returns beyond a modest sensor count.** Error drops sharply from 4 to 9 sensors
  (0.618 → 0.366 for the grid), then is essentially flat from 9 through 81 sensors (0.35–0.37
  throughout). Since the smooth truth field is a sum of only $4\times4-1=15$ low-frequency modes, a
  handful of well-placed sensors already captures most of the recoverable information; adding more
  sensors past that point does not meaningfully improve the reconstruction *for this truth field and
  this regulariser* — it does not follow that 9 sensors would be enough for a different (e.g. higher
  wavenumber, or structured/localised) truth.
- **Placement matters most when sensors are very sparse.** At 4 sensors, random placement (0.385)
  outperforms the regular grid (0.618) on average — a poorly-placed regular grid at very low density
  can miss more of the domain than a random scatter of the same size — but the gap closes almost
  completely by 9+ sensors, where grid and random placement give statistically indistinguishable
  results (well within the random trials' spread).

**What this does and does not establish about identifiability.** These results characterise recoverability of a *specific, low-dimensional smooth field* from sparse pressure data under H1-seminorm
regularisation; they are not a general identifiability statement for arbitrary permeability fields.
Combined with the "Reconstruction results" section above, sparse pressure observations of this Darcy
flow can identify broad, large-scale (low-wavenumber) permeability structure reasonably well, but
cannot — regardless of how many sensors are added — identify sharp, spatially localised features (the
structured truth's inclusions) with this observation model and regulariser; that limitation is
governed by the smoothing nature of the elliptic forward map itself, not by an insufficient sensor
count.

## Computational performance

Measured on: Apple M3 Pro, macOS 26.6.2-arm64, dolfinx 0.10.0, PETSc/MUMPS direct LU solves (no GPU),
same conda-forge `fenicsx` environment as the companion `fem-cylinder-flow` project
(`results/environment_versions.txt` for the exact pinned versions).

- Single forward solve (inversion mesh, $N=32$): well under 0.1 s.
- Single reconstruction (36 sensors, $\alpha=30$, `full` config, ~130 L-BFGS iterations, each one
  forward + one adjoint PDE solve): 10–12 s.
- Full regularisation study (9 $\alpha$ values, shared synthetic data): ~11 min.
- Full noise study (5 noise levels, shared truth/sensors): ~1 min.
- Full sensor study (5 sensor counts × (1 grid + 5 random trials) = 30 reconstructions,
  independent data generation per case): ~8 min.

## Limitations

- **H1-seminorm regularisation cannot recover sharp, localised structure** (the structured-truth
  result above), regardless of $\alpha$ — a fundamental mismatch between this regulariser and that
  class of truth field, not a tuning failure. Total-variation or structure-informed priors are the
  natural fix and are not implemented here (they are a substantially different, non-quadratic
  optimisation problem).
- **Only two truth-field families were tested**, both isotropic in construction; no attempt was made
  to test truths with directional/anisotropic structure.
- **The observation operator uses a fixed Gaussian bump width** (not itself studied for sensitivity);
  a much wider or narrower bump would change the effective information content per sensor.
- **Sensor locations are confined to $[0.1,0.9]^2$** (a margin from the boundary); sensors very close
  to the Dirichlet boundaries were not tested and would carry different (likely less) information,
  since $p$ is already known there.
- **No attempt was made to choose the regulariser type (H1 vs. L2-to-prior) systematically**; H1 was
  used as the default throughout based on general suitability for spatially distributed coefficients,
  not a comparison study between the two.
- CPU-only; no GPU or distributed-memory benchmarking was performed.
- This is the deterministic (MAP-only) inverse problem. Posterior uncertainty, Laplace approximations,
  and sampling-based inference are explicitly out of scope here and are planned for a Bayesian
  extension (Project 4) that reuses this repository's forward model, parameterisation, observation
  operator, noise model, and regularisation operators without modification.

## Reproducibility

```bash
conda env create -f environment.yml && conda activate fenicsx   # or reuse the fem-cylinder-flow env

pytest tests/ -v                              # 34 tests

python scripts/verify_forward.py              # forward-solver manufactured-solution convergence
python scripts/verify_gradient.py             # adjoint Taylor-remainder check (run before trusting
                                                # anything below)

python scripts/run_reconstruction_demo.py --config smoke   # <1 min sanity check
python scripts/run_reconstruction_demo.py --config local   # development results, ~15 s
python scripts/run_reconstruction_demo.py --config full    # reported results, ~25 s

python scripts/regularization_study.py --config full       # ~11 min
python scripts/noise_study.py --config full                 # ~1 min
python scripts/sensor_study.py --config full                 # ~8 min
```

Raw numerical results are written to `results/*.txt`; figures are generated from those saved results,
not hand-drawn, so every number and figure in this README is traceable to a specific script and
output file. `configs/{smoke,local,full}.yaml` are never silently substituted for each other — every
result above is explicitly labelled with the config that produced it.

## Repository structure

```
darcy-inverse-problem/
├── README.md, environment.yml
├── src/
│   ├── forward.py            Darcy weak form, mesh, boundary conditions, forward solve
│   ├── manufactured.py         UFL-symbolic manufactured solution for forward-solver verification
│   ├── truth_fields.py          smooth (KL-like) and structured (inclusions) synthetic truths
│   ├── observation.py            sensor placement, Gaussian-bump observation operator H
│   ├── data_generation.py         synthetic data on a finer truth mesh (inverse-crime avoidance)
│   ├── regularization.py           H1-seminorm and L2-to-prior regularisers
│   ├── adjoint.py                   objective, adjoint equation, UFL-differentiated gradient
│   ├── inversion.py                  L-BFGS wrapper with recorded iteration history
│   └── experiment.py                  shared single-reconstruction runner for the study scripts
├── scripts/
│   ├── verify_forward.py, verify_gradient.py    verification (run before trusting anything else)
│   ├── run_reconstruction_demo.py                headline smooth/structured reconstructions
│   ├── regularization_study.py, noise_study.py, sensor_study.py
│   └── (each accepts --config smoke|local|full)
├── tests/                    34 pytest tests
├── configs/                  smoke.yaml / local.yaml / full.yaml
├── figures/, results/        generated outputs backing every number in this README
```

## Installation

```bash
cd darcy-inverse-problem
conda env create -f environment.yml
conda activate fenicsx
```

## References

- Stuart, A. M. (2010). "Inverse problems: a Bayesian perspective." *Acta Numerica* 19, 451-559.
- Engl, H. W., Hanke, M., Neubauer, A. (1996). *Regularization of Inverse Problems*. Kluwer.
- Morozov, V. A. (1966). "On the solution of functionally ill-posed problems in a Banach space."
- Kaipio, J., Somersalo, E. (2007). "Statistical inverse problems: discretization, model reduction and
  inverse crimes." *J. Comput. Appl. Math.* 198(2), 493-504.
- Farrell, P. E., Ham, D. A., Funke, S. W., Rognes, M. E. (2013). "Automated derivation of the adjoint
  of high-level transient finite element programs." *SIAM J. Sci. Comput.* 35(4), C369-C393.
- Nocedal, J., Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer. (L-BFGS, Ch. 6.)
- Bui-Thanh, T., Ghattas, O., Martin, J., Stadler, G. (2013). "A computational framework for
  infinite-dimensional Bayesian inverse problems." *SIAM J. Sci. Comput.* 35(6), A2494-A2523.

**Prior related work.** `Image_inpainting` (a pre-existing repository) demonstrates the same
regularised-inverse-problem skillset — total-variation regularisation solved via Douglas-Rachford
splitting — applied to a different problem class (image denoising, no PDE forward operator). It is
cited here as prior evidence of that skillset, not reused: this repository's forward operator, adjoint
derivation, and regularisation machinery are implemented independently for the PDE-constrained
setting.
