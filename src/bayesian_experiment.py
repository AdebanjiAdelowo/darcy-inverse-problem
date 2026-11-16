"""Shared PDE-posterior experiment runner: MAP estimation + pCN sampling +
posterior-field summaries, reused by all the Bayesian study scripts
(mirrors src/experiment.py's role for the deterministic studies)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from dolfinx.fem import Function

from src.forward import build_mesh, solve_forward
from src.kl_prior import build_kl_prior, KLPrior
from src.observation import build_observation_operator
from src.data_generation import generate_synthetic_data, SyntheticData
from src.bayesian_darcy import BayesianDarcyProblem
from src.pcn_sampler import run_pcn, PCNResult, effective_sample_size
from src.optim_diagnostics import OptimizerDiagnostics, run_lbfgsb_with_diagnostics


@dataclass
class BayesianResult:
    bp: BayesianDarcyProblem
    prior: KLPrior
    data: SyntheticData
    xi_map: np.ndarray
    m_map: np.ndarray
    map_n_iter: int
    map_converged: bool
    pcn: PCNResult
    m_samples: np.ndarray  # (n_samples, n_dofs), FEM nodal fields for each retained xi sample
    m_post_mean: np.ndarray
    m_post_std: np.ndarray
    m_true_on_inv_mesh: np.ndarray
    ess_per_dim: np.ndarray
    # Milestone 2 diagnostics (defaults keep older constructors valid)
    map_diagnostics: OptimizerDiagnostics | None = None
    likelihood_path: str = "legacy"
    pcn_runtime_seconds: float | None = None
    pcn_forward_solves: int | None = None
    pcn_adjoint_solves: int | None = None
    extra: dict = field(default_factory=dict)


def compute_map_with_diagnostics(bp: BayesianDarcyProblem, xi0: np.ndarray | None = None,
                                 max_iter: int = 100):
    """MAP of Psi(xi) = Phi(xi) + 0.5||xi||^2 with SciPy L-BFGS-B, exactly as in
    compute_map (same objective arithmetic, options and start), plus diagnostics.
    'misfit' is Phi and 'regularization' is the prior term 0.5||xi||^2."""
    r = bp.prior.r
    x0 = np.zeros(r) if xi0 is None else xi0

    def fun_split(xi):
        # same operations as BayesianDarcyProblem.neg_log_posterior_and_grad
        misfit, grad_misfit, p = bp.neg_log_likelihood_and_grad(xi)
        prior_term = 0.5 * float(np.dot(xi, xi))
        J = misfit + prior_term
        grad = grad_misfit + xi
        return J, grad, misfit, prior_term, bp.obs.apply(p)

    xi_map, diag, pred, result = run_lbfgsb_with_diagnostics(
        fun_split, x0, max_iter=max_iter, gtol=1e-6, ftol=1e-12, solve_counter=bp.solve_counts)
    diag.extra["predicted_observations_at_map"] = None if pred is None else np.asarray(pred).tolist()
    return xi_map, diag


def compute_map(bp: BayesianDarcyProblem, xi0: np.ndarray | None = None, max_iter: int = 100):
    """Historical interface: returns (xi_map, n_iter, scipy_success)."""
    xi_map, diag = compute_map_with_diagnostics(bp, xi0=xi0, max_iter=max_iter)
    return xi_map, diag.n_iterations, diag.scipy_success


def run_bayesian_experiment(
    truth_fn,
    sensor_locations: np.ndarray,
    truth_mesh_n: int,
    inv_mesh_n: int,
    sensor_width: float,
    relative_noise: float,
    prior_r: int,
    prior_gamma: float,
    prior_delta: float,
    n_samples: int,
    beta: float,
    burn_in: int,
    map_max_iter: int = 100,
    data_seed: int = 0,
    mcmc_seed: int = 1,
    data: SyntheticData | None = None,
    likelihood_path: str = "forward_only",
) -> BayesianResult:
    """likelihood_path: "forward_only" (default since Milestone 2; one forward
    solve per pCN proposal) or "legacy" (the historical forward+adjoint path).
    Both evaluate the same likelihood with the same arithmetic."""
    if likelihood_path not in ("forward_only", "legacy"):
        raise ValueError(f"unknown likelihood_path {likelihood_path!r}")
    if data is None:
        data = generate_synthetic_data(truth_fn, sensor_locations, truth_mesh_n, sensor_width,
                                        relative_noise, data_seed)

    inv_dm = build_mesh(inv_mesh_n)
    H_inv = build_observation_operator(inv_dm.mesh, inv_dm.V_p, sensor_locations, sensor_width)
    prior = build_kl_prior(inv_dm, r=prior_r, gamma=prior_gamma, delta=prior_delta)
    bp = BayesianDarcyProblem(dm=inv_dm, obs=H_inv, y=data.y_noisy, sigma=data.sigma, prior=prior)

    xi_map, map_diag = compute_map_with_diagnostics(bp, max_iter=map_max_iter)
    map_nit, map_converged = map_diag.n_iterations, map_diag.scipy_success
    m_map = prior.sample_array(xi_map)

    nll = bp.neg_log_likelihood if likelihood_path == "forward_only" else bp.neg_log_likelihood_legacy
    f0, a0 = bp.solve_counts()
    t0 = time.perf_counter()
    pcn_result = run_pcn(nll, r=prior.r, n_samples=n_samples, beta=beta,
                          burn_in=burn_in, seed=mcmc_seed, x0=xi_map)
    pcn_runtime = time.perf_counter() - t0
    f1, a1 = bp.solve_counts()

    m_samples = np.array([prior.sample_array(xi) for xi in pcn_result.samples])
    m_post_mean = m_samples.mean(axis=0)
    m_post_std = m_samples.std(axis=0)

    m_true_on_inv = Function(inv_dm.V_m)
    m_true_on_inv.interpolate(truth_fn)

    ess = np.array([effective_sample_size(pcn_result.samples[:, j]) for j in range(prior.r)])

    return BayesianResult(
        bp=bp, prior=prior, data=data, xi_map=xi_map, m_map=m_map,
        map_n_iter=map_nit, map_converged=map_converged, pcn=pcn_result, m_samples=m_samples,
        m_post_mean=m_post_mean, m_post_std=m_post_std,
        m_true_on_inv_mesh=m_true_on_inv.x.array.copy(), ess_per_dim=ess,
        map_diagnostics=map_diag, likelihood_path=likelihood_path,
        pcn_runtime_seconds=pcn_runtime, pcn_forward_solves=f1 - f0, pcn_adjoint_solves=a1 - a0,
    )


def integrated_posterior_variance(result: BayesianResult) -> float:
    """integral_Omega Var[m(x)|y] dx, approximated via the FEM mass matrix
    on the posterior-std field (consistent with the L2 norms used
    throughout this project)."""
    import ufl
    from dolfinx import fem
    var_field = Function(result.bp.dm.V_m)
    var_field.x.array[:] = result.m_post_std ** 2
    return float(fem.assemble_scalar(fem.form(var_field * ufl.dx)))
