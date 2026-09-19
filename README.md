# Darcy-Flow Inverse Problem: Deterministic and Bayesian Log-Permeability Reconstruction

A complete inverse-problems study of a 2D Darcy (groundwater/reservoir) flow, in two parts. **Part I**
is a deterministic, adjoint-based regularised reconstruction (MAP-only): recover a spatially varying
permeability field from sparse, noisy pressure measurements, with an adjoint gradient verified by a
Taylor-remainder test, L-BFGS optimisation, and explicit studies of regularisation strength,
observation noise, and sensor density/placement. **Part II** extends this to full Bayesian inference
over the permeability field, $\pi(m\mid y) \propto \pi(y\mid m)\pi_0(m)$, using a Karhunen-Loeve
Gaussian prior, a preconditioned Crank-Nicolson (pCN) MCMC sampler independently validated against an
analytically solvable problem, and quantitative uncertainty studies.

## Overview

This is primarily an applied-mathematics / inverse-problems project, not a software-engineering
demonstration: Part I shows **elliptic PDE → finite-element forward solver → parameter-to-observable
map → ill-posed inverse problem → regularisation → PDE-constrained optimisation (adjoint method) →
noise/sensor sensitivity analysis → honest quantitative evaluation**; Part II extends this to
**probability → priors → likelihood → posterior inference → uncertainty quantification →
identifiability / information content**, using established mathematics throughout rather than a new
method. No neural network, PINN, learned surrogate, or neural posterior estimator is used anywhere in
this repository (a deliberate scope boundary set by the project brief).

# Part I — Deterministic Inversion

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
- This is the deterministic (MAP-only) inverse problem. Posterior uncertainty, credible intervals, and
  a full Bayesian treatment are taken up in **Part II** below, which reuses this repository's forward
  model, parameterisation, observation operator, noise model, and regularisation operators without
  modification.

# Part II — Bayesian Inversion

## Bayesian formulation

Part I finds a single point estimate $m_{\mathrm{MAP}}$. Part II instead characterises the full
posterior distribution over $m$ given the data:

$$\pi(m\mid y) \propto \pi(y\mid m)\,\pi_0(m), \qquad y = Hp(m) + \varepsilon,\ \varepsilon\sim\mathcal N(0,\Gamma),$$

$$\pi(y\mid m) \propto \exp\!\left[-\tfrac12 (Hp(m)-y)^\top\Gamma^{-1}(Hp(m)-y)\right].$$

This answers a different question than Part I: not just "what is the best-fitting permeability
field?" but "given the data and our modelling assumptions, what range of permeability fields remain
plausible, and where?"

## Prior construction and KL parameterisation

A Gaussian prior $\pi_0 = \mathcal N(m_0, \mathcal C)$ is built from the covariance operator
$\mathcal C = \mathcal A^{-2}$, $\mathcal A = -\gamma\Delta + \delta I$ (Neumann BCs) — the standard
construction for Bayesian PDE inversion (Lindgren, Rue & Lindström, 2011; Bui-Thanh, Ghattas, Martin &
Stadler, 2013). The exponent 2, not 1, is not arbitrary: in 2D a Gaussian field needs
$2\alpha > d = 2$ to have finite pointwise variance, so $\mathcal A^{-1}$ is the inadmissible
borderline case and $\mathcal A^{-2}$ is the standard safely-trace-class choice. $\mathcal A$ and
$\mathcal C$ commute, so the KL eigenpairs solve the generalised FEM eigenproblem
$A_h v_j = \mu_j M_h v_j$ ($A_h$, $M_h$ the stiffness/mass matrices of $\mathcal A$'s weak form), with
$\lambda_j = \mu_j^{-2}$ and $M_h$-orthonormal eigenvectors (verified directly to $10^{-8}$ absolute
error, `tests/test_kl_prior.py`). A truncated field is

$$m(x) = m_0(x) + \sum_{j=1}^r \sqrt{\lambda_j}\,\xi_j\,\phi_j(x), \qquad \xi_j\sim\mathcal N(0,1)\text{ i.i.d.}$$

**Eigenvalue decay and truncation choice** ($\gamma=0.05$, $\delta=1.0$, inversion mesh $N=16$):

| $r$ | 5 | 10 | 15 | 20 | 25 | 30 | 40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| captured variance | 0.783 | 0.902 | **0.944** | 0.962 | 0.972 | 0.979 | 0.986 |

(figure: `figures/kl_eigenvalue_decay.png`.) **$r=15$ is used for all studies below**: it captures
94.4% of the prior variance, and going to $r=25$ (67% more MCMC dimensions, directly increasing
per-sample cost) would only add one more percentage point. This was a genuine
computational/informativeness trade-off, not chosen purely for MCMC speed. Representative prior
samples (figure: `figures/kl_prior_samples.png`) show smooth, spatially correlated fields with
visible variance inflation near the domain boundary — the expected, documented artefact of the
Neumann boundary condition on $\mathcal A$ (not corrected here; see Limitations).

## Likelihood

As stated above; identical noise model to Part I ($\Gamma = \mathrm{diag}(\sigma_1^2,\dots,\sigma_n^2)$,
same Gaussian-bump observation operator $H$, same synthetic-data generation with inverse-crime
avoidance).

## Relationship between MAP and Tikhonov regularisation

Working in the whitened KL coefficients $\xi$ (prior $\xi\sim\mathcal N(0,I_r)$) makes this connection
exact and explicit. The negative log-posterior is

$$\Psi(\xi) = \Phi(\xi) + \tfrac12\|\xi\|^2 + \text{const}, \qquad \Phi(\xi) = \tfrac12\|Hp(m(\xi))-y\|^2_{\Gamma^{-1}},$$

and $\tfrac12\|\xi\|^2$ is exactly $-\log$ of the $\mathcal N(0,I_r)$ prior density up to an additive
constant. **Minimising $\Psi$ is therefore exactly Part I's regularised objective
$J(m)=\tfrac12\|Hp(m)-y\|^2_{\Gamma^{-1}} + \tfrac{\alpha}{2}R(m)$ with $\alpha=1$ and an L2 (Tikhonov)
penalty $R(\xi)=\|\xi\|^2$ in the whitened KL basis** — the MAP estimate under this Gaussian prior
*is* a Tikhonov-regularised estimate, with the prior covariance operator determining the
regulariser's metric (Stuart, 2010, Sec. 2.3). This is implemented directly by reusing Part I's
`InverseProblem` class with `alpha=0` for the misfit-only term, then adding the $\tfrac12\|\xi\|^2$
prior term explicitly (`src/bayesian_darcy.py`) — no PDE/adjoint code was rewritten; only one new
chain-rule step (nodal gradient → KL-coefficient gradient) was added and independently Taylor-tested
(see "Sampler validation" below).

## Posterior sampling: preconditioned Crank-Nicolson (pCN)

Generic random-walk Metropolis is not appropriate here: its acceptance rate degrades as the parameter
dimension $r$ grows unless the step size shrinks with dimension. **pCN** uses the proposal
$\xi' = \sqrt{1-\beta^2}\,\xi + \beta\,\eta$, $\eta\sim\mathcal N(0,I_r)$, which is exactly
prior-reversible: the prior-density terms in the Metropolis-Hastings ratio cancel identically, leaving

$$\alpha(\xi,\xi') = \min\!\big(1,\ \exp[\Phi(\xi)-\Phi(\xi')]\big),$$

an acceptance probability depending only on the likelihood ratio, well-behaved even as $r$ grows
(Cotter, Roberts, Stuart & White, 2013). $\beta$ was tuned empirically on the actual PDE posterior
(not assumed): a sweep from $\beta=0.3$ (0.6% acceptance, far too small) down to $\beta=0.005$ (75%
acceptance, too-small steps) identified $\beta=0.02$ as giving the standard textbook target range
(~35-40% acceptance) while still moving efficiently; this value is used throughout.

## Sampler validation

**The PDE posterior is not the first thing this sampler is tested on.** `src/linear_gaussian_validation.py`
defines a linear-Gaussian inverse problem ($y=G\xi_{\mathrm{true}}+\varepsilon$, prior
$\xi\sim\mathcal N(0,I_r)$) whose posterior is available in closed form
($\Sigma_{\mathrm{post}}=(G^\top\Gamma^{-1}G+I)^{-1}$, $\mu_{\mathrm{post}}=\Sigma_{\mathrm{post}}G^\top\Gamma^{-1}y$).
`scripts/validate_sampler.py` ($r=8$, 20 observations, 40,000 post-burn-in samples, $\beta=0.05$):

| Diagnostic | Result |
|---|---:|
| acceptance rate | 38.6% |
| posterior mean relative error vs. analytical | 0.59% |
| posterior variance mean absolute error | $5.9\times10^{-4}$ |
| mean effective sample size (8 dims) | 398 (out of 45,000 proposals) |

(figure: `figures/sampler_validation.png` — trace plot, autocorrelation, and mean/variance overlay,
all visually matching the analytical posterior.) The sampler is trustworthy before it is ever pointed
at the PDE problem.

**The xi-space MAP gradient** (the one piece of genuinely new derivative code, reusing Part I's
verified nodal gradient via one chain-rule step) was separately Taylor-tested
(`scripts/verify_bayesian_gradient.py`): observed remainder order **2.000** across five step-size
halvings, before being trusted for MAP optimisation.

## MCMC diagnostics

`python scripts/run_posterior_demo.py --config full` — smooth truth, 25 sensors, 2% noise, $r=15$,
$\beta=0.02$, 2,000 burn-in + 8,000 retained samples (10,000 total pCN proposals), chain initialised
at the MAP:

| Diagnostic | Value |
|---|---:|
| acceptance rate | 40.0% |
| ESS per KL dimension | 5–86 (mean 13.4) out of 8,000 samples |
| mean ESS / n_proposals | ~0.17% |

**This is a genuinely important, diagnosed limitation, not a footnote.** ESS this low (most dimensions
under 1% efficiency, vs. ~40% for the toy linear-Gaussian validation problem at a comparable acceptance
rate) means the number of *effectively independent* draws is small. Two further checks confirm this
concretely, rather than just citing the ESS number:

- **Split-chain comparison**: mean of the first half of the retained chain vs. the second half differs
  by up to 0.70 in individual KL coefficients (dimension-dependent; some coefficients even change sign
  between halves), against a prior standard deviation of 1.0 — a large disagreement for a converged
  estimate.
- **Trace and autocorrelation plots** (figure: `figures/posterior_diagnostics_smooth_full.png`) show
  this directly and unevenly across dimensions: KL dimensions 1 and 2 show classic slow random-walk
  behaviour (a long, smooth excursion up or down over the full 10,000 iterations, with autocorrelation
  still above 0.9 at lag 200 — barely decayed at all), while dimension 3 mixes well by comparison (a
  visibly stationary-looking trace, autocorrelation decayed to ~0 by lag ~150). Mixing quality is
  genuinely heterogeneous across the 15 KL dimensions, not uniformly bad — which is itself useful
  information (it points at *which* directions in parameter space are hard for this proposal to
  explore) but means a single scalar ESS or acceptance-rate summary would have hidden this pattern.
- Posterior **standard deviation** is comparatively (not perfectly) more stable under the same
  split-chain check: mean relative disagreement 26% between halves, vs. 75% for the mean — still
  imperfect, but the second moment is less corrupted by the slow mixing than the first moment.

**Ruled out as the cause**: the MAP starting point itself. Running L-BFGS-B to a much tighter tolerance
(2,000 iterations instead of 120) changes the negative-log-posterior gradient norm from $1.5\times10^{-3}$
to $3\times10^{-5}$ but leaves every reported $\xi_{\mathrm{MAP}}$ coefficient unchanged to 3 decimal
places — the chain is started at an accurate MAP, so the drift is a genuine mixing-rate property of the
PDE posterior's geometry (attributed to correlation between KL modes induced by the shared, nonlinear
forward map), not an artefact of a poor initial point.

**Practical consequence, applied consistently below**: MAP is used as the primary trustworthy point
estimate throughout. Posterior mean, standard deviation, and credible intervals are still reported (the
sampler's correctness is independently established; the issue is chain length, not algorithm
correctness), but are explicitly flagged wherever the split-chain check indicates they should be read
with caution, and *aggregate/integrated* quantities (below) turn out to give more robust, physically
sensible trends than pointwise or per-coefficient values.

## Posterior mean, MAP, and uncertainty

| Quantity | smooth truth | structured truth |
|---|---:|---:|
| relative MAP error | **0.387** | **0.807** |
| relative posterior-mean error | 1.001 (see caveat above) | 1.466 (see caveat above) |
| integrated posterior variance $\int\mathrm{Var}[m\mid y]\,dx$ | 0.295 | 0.367 |

(figure: `figures/posterior_fields_{smooth,structured}_full.png` — truth / MAP / posterior mean /
posterior std / MAP error / posterior-mean error, all on shared colour scales.) The MAP errors closely
match Part I's deterministic reconstruction errors at comparable settings (0.35 smooth / 0.85
structured there), as expected since MAP is exactly the Part-I-equivalent point estimate under this
prior (see "Relationship between MAP and Tikhonov regularisation"). **The posterior-mean errors are
reported for completeness but should not be read as "the posterior mean is worse than MAP"** — given
the split-chain diagnostic above, they are better interpreted as showing that 8,000 post-burn-in pCN
samples are not enough to pin down the posterior mean to the precision needed for a fair comparison
against MAP, rather than as a real difference between the two estimators. This is exactly why the
project's own standard ("do not present raw MCMC samples as trustworthy without convergence/mixing
diagnostics") is being applied here to Part II's own results, not only to the toy validation problem.
Posterior standard deviation is visibly higher in regions further from sensor locations and, for the
structured truth, roughly uniform (the Gaussian prior does not "know" where the inclusions are, so it
does not concentrate uncertainty around them the way a correctly-specified prior would).

## Credible intervals

95% credible intervals ($2.5$/$97.5$ percentiles of the retained pCN samples) at 5 spatial points,
smooth truth, full config:

| point | true $m$ | post. mean | 2.5% | 97.5% | covered |
|---|---:|---:|---:|---:|:---:|
| (0.25, 0.25) | -0.179 | -0.209 | -1.258 | 0.626 | yes |
| (0.75, 0.25) | 0.358 | 0.037 | -1.193 | 0.975 | yes |
| (0.50, 0.50) | -0.169 | 0.212 | -0.563 | 0.989 | yes |
| (0.25, 0.75) | -0.639 | -0.427 | -1.607 | 0.472 | yes |
| (0.75, 0.75) | 0.460 | 0.853 | -0.301 | 1.741 | yes |

Coverage: **5/5** for the smooth truth and 5/5 for the structured truth (full config;
`results/posterior_demo_full.txt`). **This is explicitly not a calibration study, and 5/5 coverage
from one synthetic realisation is weak evidence of anything on its own** — with only 5 points and
intervals this wide (up to ~1.9 units against a prior standard deviation of 1.0 per coefficient), high
coverage is close to guaranteed regardless of whether the posterior is well-calibrated; it is reported
because the brief asks for it, not presented as a validated calibration result. The intervals
themselves are also affected by the chain-length limitation discussed above (they are almost certainly
wider than a fully-converged chain would give, which would trivially inflate coverage further) — stated
plainly rather than glossed over.

## Sensor-information study

`python scripts/bayesian_sensor_study.py --config full` — smooth truth, $\alpha$/prior fixed, same
chain settings as above per sensor count:

| $n_{\mathrm{sensors}}$ | integrated posterior variance | acceptance rate |
|---:|---:|---:|
| 9 | 0.260 | 66.1% |
| 25 | 0.234 | 40.1% |
| 49 | 0.161 | 23.7% |

(figure: `figures/bayesian_sensor_study_full.png`.) **Integrated posterior variance decreases
monotonically as sensor count increases** (0.260 → 0.234 → 0.161), the physically expected direction —
more observations reduce uncertainty. This aggregate, domain-integrated quantity is noticeably more
robust to the chain-mixing limitation than the pointwise posterior-mean values above (it is a spatial
average of the posterior *second moment*, which the split-chain check found comparatively more stable
than the mean), so this monotonic trend is read as a genuine, if not high-precision, finding rather than
chain noise. This complements, and answers a different question from, Part I's deterministic sensor
study: not just "does more sensors reduce reconstruction *error*" (Part I: yes, with diminishing
returns past ~9–25 sensors) but "does more sensors reduce reconstruction *uncertainty*" (here: yes,
continuing to decrease through the largest tested count, without the same clear plateau Part I found
for error) — consistent with the two questions being related but not identical. Acceptance rate falling
as sensor count grows (66% → 40% → 24%) is itself informative: a more informative (more constraining)
likelihood sharpens the posterior, which pCN feels as a smaller effective step relative to the
now-tighter target distribution at fixed $\beta$.

## Noise sensitivity

`python scripts/bayesian_noise_study.py --config full` — smooth truth, 25 sensors, same chain settings
per noise level:

| relative noise | integrated posterior variance | acceptance rate |
|---:|---:|---:|
| 0.01 | 0.134 | 15.8% |
| 0.05 | 0.226 | 71.1% |
| 0.10 | 0.401 | 85.3% |

(figure: `figures/bayesian_noise_study_full.png`.) **Integrated posterior variance increases
monotonically with noise level** (0.134 → 0.226 → 0.401), more than tripling from the lowest to the
highest tested noise — the expected direction, quantified rather than only shown as different pictures.
As with the sensor study, this integrated/aggregate quantity is used as the headline result specifically
because it is more robust to the diagnosed chain-mixing limitation than pointwise posterior-mean
values. The acceptance-rate pattern is the mirror image of the sensor study's: noisier data gives a
*less* informative (flatter, more prior-like) likelihood, which pCN — proposing relative to the fixed
prior scale — accepts more readily.

## Prior sensitivity

`python scripts/bayesian_prior_sensitivity.py --config full` — smooth truth, 25 sensors, 2% noise,
varying the prior's correlation-length parameter $\gamma$ and KL truncation $r$:

| variant | $r$ | $\gamma$ | captured variance | integrated post. var. | rel. posterior-mean error |
|---|---:|---:|---:|---:|---:|
| baseline | 15 | 0.05 | 94.4% | 0.188 | 1.011 |
| short_corr_len | 15 | 0.01 | 72.7% | **0.143** | 1.327 |
| long_corr_len | 15 | 0.20 | 99.2% | 0.145 | **0.556** |
| fewer_modes | 6 | 0.05 | 82.2% | 0.285 | 3.427 |

(figure: `figures/bayesian_prior_sensitivity_full.png`.) Given the posterior-mean caveat above, these
error figures should be read as indicative rather than precise, but the pattern across variants is
large enough (0.56 to 3.43) to be informative regardless: **the long-correlation-length prior gives
both the lowest error and among the lowest variance**, because its smoothness assumption actually
matches the smooth truth well, while **the short-correlation-length prior gives the lowest variance of
all but a substantially worse error** than the baseline — a prior that is "confident" (tight posterior)
is not the same as a prior that is *right*, precisely the distinction Part II's brief asks to
demonstrate. The **fewer_modes** variant ($r=6$, only 82.2% captured variance) is markedly worse on
both counts, showing directly that under-truncating the KL expansion is not merely "cheaper" but loses
real representational capacity needed to fit this truth. With only 25 sparse, noisy sensors, the
posterior is visibly prior-informed: four different — all "reasonable" — prior choices give
integrated variances spanning a factor of 2 and mean errors spanning a factor of 6, for the *same*
data.

## Structured-truth / prior-misspecification experiment

The structured truth (two localised circular inclusions) is, by construction, outside the class of
fields the smooth Gaussian KL prior assigns much probability to — the same mismatch documented for
Part I's H1-seminorm regulariser (Section "Reconstruction results"), now examined from the Bayesian
side. MAP error is **0.807** (vs. 0.387 for the smooth truth at matched settings), and the posterior
mean/std fields (figure: `figures/posterior_fields_structured_full.png`) show the reconstruction
smoothing over both inclusions almost entirely — visually similar to Part I's H1-regularised failure,
which is expected since MAP under this prior **is** the H1-family-equivalent Tikhonov estimate (see
"Relationship between MAP and Tikhonov regularisation").

**The key Bayesian-specific question is whether the posterior is at least honestly uncertain about
this failure, or whether it is confidently wrong.** Integrated posterior variance for the structured
truth is 0.367, *higher* than the smooth truth's 0.295 at matched settings — so the posterior is not
simply confident-and-wrong here; it does carry more spread when the truth is harder to represent.
However, this is at most partial reassurance: the posterior mean is still badly biased toward the
smooth prior's characteristic length scale, and no amount of variance in the *KL coefficient
directions the prior can represent* can express uncertainty about the missing directions (sharp,
localised structure) that the truncated Gaussian KL prior cannot represent at all. **This is the
distinction the brief asks to draw explicitly: posterior variance quantifies uncertainty within the
model class the prior defines; it does not, and cannot, quantify uncertainty about whether that model
class is the right one.** A confidently-wrong-but-not-modestly-so posterior for the inclusions would
have been the more concerning failure mode, and is not quite what is observed here, but the
reconstruction failure itself is real and is not something more posterior samples would fix — it is a
property of the prior's support, not of the chain length.

## Posterior predictive checks

For 20 posterior draws $m^{(s)}$, the forward problem is re-solved to obtain $p^{(s)} = p(m^{(s)})$
and predicted observations $Hp^{(s)}$, compared against the actual observed (noisy) data $y$ (figure:
`figures/posterior_predictive_{smooth,structured}_full.png`). This checks uncertainty in the
*observable* quantity (pressure at sensors), distinct from uncertainty in the *inferred coefficient*
$m$ examined above — a model can have a well-behaved predictive distribution even where the underlying
coefficient field is poorly constrained, since many different $m$ fields can produce similar pressure
data (that is precisely the ill-posedness Part I demonstrates).

| truth | 95% predictive-interval coverage of observed data (25 sensors) |
|---|---:|
| smooth | 76.0% |
| structured | 76.0% |

Both are somewhat below the nominal 95% target. Given the same chain-mixing limitation discussed
above, the predictive intervals built from only 20 (further sub-sampled from the already
under-mixed) posterior draws are themselves an approximate, noisy estimate of the true posterior
predictive distribution — this figure is not read as a precise calibration statement, but the fact
that both truths give a similar, moderately-below-nominal coverage (rather than, say, near-100% or
near-0%) is at least consistent with the predictive distribution being in the right regime rather than
badly mis-specified.

## Computational cost (Part II)

Measured on: Apple M3 Pro, macOS 26.6.2-arm64, same environment as Part I and `fem-cylinder-flow`.

- **Per-proposal cost**: each pCN proposal calls `objective_and_gradient`, which performs one forward
  PDE solve and one adjoint PDE solve (2 solves), even though pCN itself only needs the likelihood
  value, not the gradient — a real, acknowledged inefficiency (noted in Limitations) from reusing
  Part I's combined objective+gradient routine rather than writing a likelihood-only fast path.
  Measured at ~42-49 ms/proposal (inversion mesh $N=16$, 441 $m$-dofs / 1,681 $p$-dofs), i.e.
  ~21-25 ms per PDE solve.
- **Chain lengths**: `full` config uses 2,000 burn-in + 8,000 retained samples = 10,000 proposals =
  20,000 PDE solves per chain; `local` uses 1,000 + 3,000 = 4,000 proposals = 8,000 PDE solves.
- **Total full-resolution Bayesian study runtime**: 86.2 minutes (12:12 to 13:39 wall-clock),
  comprising `run_posterior_demo` (14.0 min, 2 truths), `bayesian_sensor_study` (22.4 min, 3 sensor
  counts), `bayesian_noise_study` (20.9 min, 3 noise levels), and `bayesian_prior_sensitivity`
  (28.9 min, 4 prior variants) — roughly 240,000 total PDE solves across the full-resolution study
  suite. The `local` config sequence (development-scale) took 35.3 minutes total.
- All posterior samples are saved to `results/posterior_samples_*.npz` specifically so that figures
  and diagnostics (including the split-chain check documented above) can be regenerated without
  rerunning any MCMC.

## Limitations (Part II)

- **The single most important limitation: at the tested chain lengths (up to 10,000 proposals), the
  pCN chain has not fully converged for the posterior MEAN**, diagnosed directly (not assumed) via a
  split-chain comparison (up to 0.70 disagreement between first-half and second-half means, against a
  prior standard deviation of 1.0, with some coefficients even changing sign between halves) and a
  visibly still-drifting running-mean trace. Posterior standard deviation is comparatively more stable
  (26% vs. 75% mean relative split-chain disagreement) but not perfectly converged either. The sampler
  itself was independently validated as correct on an analytically solvable problem (see "Sampler
  validation") — this is a chain-length/budget limitation for this specific, harder posterior
  geometry, not an algorithm-correctness issue. **Practical consequence applied throughout this
  README**: MAP is treated as the reliable point estimate; posterior-mean values are reported but
  explicitly flagged; aggregate/integrated quantities (integrated posterior variance) were found to be
  more robust and are used for the headline sensor/noise trend results. ESS per KL dimension is
  reported honestly (5-86 out of 8,000 retained samples) rather than a single favourable-looking
  average.
- **Each pCN proposal costs one forward PLUS one adjoint PDE solve** (via reuse of Part I's combined
  `objective_and_gradient`), even though pCN itself needs only the likelihood value, not the gradient
  — a real, acknowledged inefficiency (roughly 2x more PDE solves than a likelihood-only fast path
  would need), not fixed here given the time already invested in this diagnosis.
- **A single MCMC chain is used throughout** (no multi-chain $\hat R$/Gelman-Rubin convergence
  diagnostic); mixing is instead assessed via the split-chain/running-mean checks above, trace plots,
  autocorrelation, and effective sample size on the single chain. A multi-chain comparison was not run
  given the already-substantial single-chain cost at `full` resolution (86 minutes for the full study
  suite) and would be the natural next step to further corroborate (or contradict) the convergence
  diagnosis above.
- **Neumann boundary conditions on the prior covariance operator** cause visible variance inflation
  near the domain boundary (visible in `figures/kl_prior_samples.png`); a Robin-BC correction (used in
  some of this literature specifically to reduce this artefact) was not implemented.
- **Credible-interval coverage is reported for a single synthetic realisation only** (5/5 for each
  truth) — explicitly not a calibration study; no repeated-realisation empirical coverage experiment
  was run, and the wide intervals resulting from the chain-length limitation above make high coverage
  from one realisation weak evidence of good calibration specifically.
- **The KL prior/basis is shared with Part I's regulariser family only in spirit** (both encode
  smoothness), not literally: Part I's H1-seminorm regulariser and Part II's $\mathcal A^{-2}$-based
  Gaussian prior are related but not identical mathematical objects; no attempt was made to make them
  exactly correspond.
- **The prior-sensitivity study's error comparison uses the (diagnosed-as-imperfectly-converged)
  posterior mean**, not MAP; the qualitative pattern (short correlation length = lower variance but
  worse error) is large enough to plausibly survive this caveat, but the exact magnitudes should be
  read with the same caution as the rest of the posterior-mean results.
- Same deterministic-inversion limitations as Part I apply where relevant (fixed observation bump
  width untested for sensitivity, sensors kept away from the Dirichlet boundary, CPU-only).

## Reproducibility (Part I)

```bash
conda env create -f environment.yml && conda activate fenicsx   # or reuse the fem-cylinder-flow env

pytest tests/ -v                              # 56 tests

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

## Reproducibility (Part II)

```bash
python scripts/kl_prior_study.py                            # eigenvalue decay, prior samples (fast)
python scripts/validate_sampler.py                          # pCN vs. analytical linear-Gaussian
python scripts/verify_bayesian_gradient.py                  # Taylor test of the xi-space MAP gradient

python scripts/run_posterior_demo.py --config smoke         # <2 min sanity check
python scripts/run_posterior_demo.py --config local         # development results
python scripts/run_posterior_demo.py --config full          # reported results (see Computational cost)

python scripts/bayesian_sensor_study.py --config full
python scripts/bayesian_noise_study.py --config full
python scripts/bayesian_prior_sensitivity.py --config full
```

Raw numerical results are written to `results/*.txt` (and posterior samples to
`results/posterior_samples_*.npz`, so figures/diagnostics can be regenerated without rerunning MCMC);
figures are generated from those saved results, not hand-drawn, so every number and figure in this
README is traceable to a specific script and output file. `configs/{smoke,local,full}.yaml` are never
silently substituted for each other — every result above is explicitly labelled with the config that
produced it.

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
│   ├── experiment.py                  shared single-reconstruction runner for Part I study scripts
│   ├── kl_prior.py                   Part II: Karhunen-Loeve Gaussian prior (covariance operator,
│   │                                   eigenpairs, sampling)
│   ├── pcn_sampler.py                 Part II: preconditioned Crank-Nicolson MCMC (generic; also
│   │                                   used for sampler validation, not PDE-specific)
│   ├── linear_gaussian_validation.py   Part II: analytically-solvable posterior for sampler validation
│   ├── bayesian_darcy.py                Part II: PDE posterior (Phi, xi-space gradient), reusing
│   │                                      Part I's forward/adjoint code
│   └── bayesian_experiment.py            Part II: shared MAP + pCN experiment runner
├── scripts/
│   ├── verify_forward.py, verify_gradient.py    Part I verification (run before trusting anything else)
│   ├── run_reconstruction_demo.py                Part I: headline smooth/structured reconstructions
│   ├── regularization_study.py, noise_study.py, sensor_study.py      Part I studies
│   ├── kl_prior_study.py, validate_sampler.py, verify_bayesian_gradient.py   Part II verification
│   ├── run_posterior_demo.py                      Part II: headline MAP/posterior/UQ demo
│   ├── bayesian_sensor_study.py, bayesian_noise_study.py, bayesian_prior_sensitivity.py
│   └── (each accepts --config smoke|local|full)
├── tests/                    56 pytest tests
├── configs/                  smoke.yaml / local.yaml / full.yaml (both Part I and Part II sections)
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
- Lindgren, F., Rue, H., Lindström, J. (2011). "An explicit link between Gaussian fields and Gaussian
  Markov random fields: the stochastic partial differential equation approach." *J. R. Stat. Soc. B*
  73(4), 423-498.
- Cotter, S. L., Roberts, G. O., Stuart, A. M., White, D. (2013). "MCMC methods for functions:
  modifying old algorithms to make them faster." *Statistical Science* 28(3), 424-446.
- Geyer, C. J. (1992). "Practical Markov Chain Monte Carlo." *Statistical Science* 7(4), 473-483.
  (Initial positive sequence effective-sample-size estimator.)

**Prior related work.** `Image_inpainting` (a pre-existing repository) demonstrates the same
regularised-inverse-problem skillset — total-variation regularisation solved via Douglas-Rachford
splitting — applied to a different problem class (image denoising, no PDE forward operator). It is
cited here as prior evidence of that skillset, not reused: this repository's forward operator, adjoint
derivation, and regularisation machinery are implemented independently for the PDE-constrained
setting.
