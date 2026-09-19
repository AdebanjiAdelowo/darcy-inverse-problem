"""Shared PDE-posterior experiment runner: MAP estimation + pCN sampling +
posterior-field summaries, reused by all the Bayesian study scripts
(mirrors src/experiment.py's role for the deterministic studies)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from dolfinx.fem import Function

from src.forward import build_mesh, solve_forward
from src.kl_prior import build_kl_prior, KLPrior
from src.observation import build_observation_operator
from src.data_generation import generate_synthetic_data, SyntheticData
from src.bayesian_darcy import BayesianDarcyProblem
from src.pcn_sampler import run_pcn, PCNResult, effective_sample_size


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


def compute_map(bp: BayesianDarcyProblem, xi0: np.ndarray | None = None, max_iter: int = 100):
    r = bp.prior.r
    x0 = np.zeros(r) if xi0 is None else xi0
    result = minimize(bp.neg_log_posterior_and_grad, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": max_iter, "gtol": 1e-6, "ftol": 1e-12})
    return result.x, result.nit, bool(result.success)


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
) -> BayesianResult:
    if data is None:
        data = generate_synthetic_data(truth_fn, sensor_locations, truth_mesh_n, sensor_width,
                                        relative_noise, data_seed)

    inv_dm = build_mesh(inv_mesh_n)
    H_inv = build_observation_operator(inv_dm.mesh, inv_dm.V_p, sensor_locations, sensor_width)
    prior = build_kl_prior(inv_dm, r=prior_r, gamma=prior_gamma, delta=prior_delta)
    bp = BayesianDarcyProblem(dm=inv_dm, obs=H_inv, y=data.y_noisy, sigma=data.sigma, prior=prior)

    xi_map, map_nit, map_converged = compute_map(bp, max_iter=map_max_iter)
    m_map = prior.sample_array(xi_map)

    pcn_result = run_pcn(bp.neg_log_likelihood, r=prior.r, n_samples=n_samples, beta=beta,
                          burn_in=burn_in, seed=mcmc_seed, x0=xi_map)

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
