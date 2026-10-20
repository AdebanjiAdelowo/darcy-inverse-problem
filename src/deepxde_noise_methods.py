"""Candidate L-BFGS-stage objectives for the noisy spatial benchmark.

DeepXDE's loss is a weighted sum of mean squared residuals,

    L = w_r mean_r(r^2) + w_b mean_b(b^2) + w_o mean_o((p - y)^2),

over N_r collocation points (PDE residual r), N_b boundary points (b) and N_o
observations. With i.i.d. N(0, sigma^2) observation noise and a residual
"model error" scale tau for the PDE and the boundary condition, the negative
log-likelihood (up to constants) is

    sum_o (p - y)^2 / (2 sigma^2) + sum_r r^2 / (2 tau^2) + sum_b b^2 / (2 tau^2),

and multiplying by 2 sigma^2 / N_o gives the same minimiser with

    w_o = 1,   w_r = (N_r / N_o) sigma^2 / tau^2,   w_b = (N_b / N_o) sigma^2 / tau^2.

A Gaussian prior N(0, s^2 I) on (alpha, beta) adds |theta|^2 / (2 s^2) to the
negative log-posterior, i.e. a penalty (sigma^2 / (N_o s^2)) |theta|^2 in the
same units. Unit weights (the validated baseline) correspond to
tau = sigma sqrt(N_r / N_o) for the PDE term, a choice that depends on sigma.

Methods (applied in the L-BFGS stage only; the Adam stage is always the
validated unit-weight baseline):

* ``baseline``: unit weights (validated configuration).
* ``naive_inverse_variance``: w_o = 1 / sigma^2, w_r = w_b = 1 (observation term
  multiplied by 1 / sigma^2, everything else unchanged). Included as a control
  for the "1 / sigma^2 automatically helps" assumption.
* ``likelihood_weighted``: the weights above with a fixed tau (TAU, fixed before
  any noisy run from the residual level the network reaches on exact data).
* ``likelihood_prior_sd1`` / ``likelihood_prior_sd0.25``: ``likelihood_weighted``
  plus a zero-centred Gaussian prior on (alpha, beta) with s = 1 or 0.25. Zero
  is a homogeneous medium, not the manufactured truth.

For noise-free data sigma is replaced by the nominal ``NOMINAL_SIGMA``.
"""
import math

TAU = 2e-3
NOMINAL_SIGMA = 0.01
METHODS = ("baseline", "naive_inverse_variance", "likelihood_weighted",
           "likelihood_prior_sd1", "likelihood_prior_sd0.25")
PRIOR_SD = {"likelihood_prior_sd1": 1.0, "likelihood_prior_sd0.25": 0.25}


def loss_counts(model):
    """(N_r, N_b, N_o) as used by DeepXDE's loss for the spatial benchmark."""
    data = model.data
    n_bc, n_obs = (int(v) for v in data.num_bcs)
    n_res = int(len(data.train_x)) - n_bc - n_obs
    return n_res, n_bc, n_obs


def method_objective(name, sigma, counts, tau=TAU):
    """Return ``(loss_weights or None, prior_coefficient, assumed_sigma)``.

    ``prior_coefficient`` c multiplies |theta|^2 in DeepXDE's loss units.
    """
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}")
    sig = sigma if sigma > 0 else NOMINAL_SIGMA
    n_res, n_bc, n_obs = counts
    if name == "baseline":
        return None, 0.0, sig
    if name == "naive_inverse_variance":
        return [1.0, 1.0, 1.0 / sig**2], 0.0, sig
    ratio = sig**2 / tau**2
    weights = [n_res / n_obs * ratio, n_bc / n_obs * ratio, 1.0]
    prior = 0.0
    if name in PRIOR_SD:
        prior = sig**2 / (n_obs * PRIOR_SD[name] ** 2)
    return weights, prior, sig


def prior_penalty(variables, coefficient):
    """Callable returning c * sum(theta^2) as a tensor, or None when c == 0."""
    if coefficient == 0:
        return None
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("prior coefficient must be finite and non-negative")

    def penalty():
        return coefficient * sum(v * v for v in variables)
    return penalty
