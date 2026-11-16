"""Equivalence of the forward-only pCN likelihood path with the historical
forward+adjoint path. The transition kernel, prior and proposal are unchanged;
only the unused adjoint solve is removed."""
import numpy as np
import pytest

from src.bayesian_darcy import BayesianDarcyProblem
from src.bayesian_experiment import run_bayesian_experiment
from src.forward import build_mesh
from src.kl_prior import build_kl_prior
from src.observation import build_observation_operator, grid_sensors
from src.pcn_sampler import run_pcn
from src.truth_fields import smooth_truth


def _bp():
    dm = build_mesh(8)
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    H = build_observation_operator(dm.mesh, dm.V_p, grid_sensors(3, 3), width=0.05)
    rng = np.random.default_rng(1)
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    return BayesianDarcyProblem(dm=dm, obs=H, y=y, sigma=np.full(H.n_sensors, 0.02), prior=prior)


def test_likelihood_values_identical_between_paths():
    bp = _bp()
    rng = np.random.default_rng(7)
    for _ in range(10):
        xi = rng.standard_normal(bp.prior.r)
        assert bp.neg_log_likelihood(xi) == bp.neg_log_likelihood_legacy(xi)


def test_forward_only_path_performs_no_adjoint_solves():
    bp = _bp()
    xi = np.zeros(bp.prior.r)
    bp.neg_log_likelihood(xi)
    assert bp.solve_counts() == (1, 0)
    bp.neg_log_likelihood_legacy(xi)
    assert bp.solve_counts() == (2, 1)


def test_identically_seeded_pcn_chains_are_identical():
    """Same seed, same start: the proposals, the accept/reject decisions and
    the chain are identical, because the likelihood values are identical."""
    bp_fast, bp_old = _bp(), _bp()
    kw = dict(r=bp_fast.prior.r, n_samples=60, beta=0.3, burn_in=20, seed=4)
    fast = run_pcn(bp_fast.neg_log_likelihood, **kw)
    old = run_pcn(bp_old.neg_log_likelihood_legacy, **kw)
    np.testing.assert_array_equal(fast.raw_chain, old.raw_chain)
    np.testing.assert_array_equal(fast.neg_log_lik, old.neg_log_lik)
    assert fast.acceptance_rate == old.acceptance_rate
    assert 0.0 < fast.acceptance_rate < 1.0  # the test exercises both accept and reject
    f_fast, a_fast = bp_fast.solve_counts()
    f_old, a_old = bp_old.solve_counts()
    assert a_fast == 0 and f_fast == fast.n_proposals + 1
    assert a_old == f_old == old.n_proposals + 1


def test_bayesian_experiment_paths_agree_and_record_solve_counts():
    kw = dict(truth_fn=smooth_truth(seed=1), sensor_locations=grid_sensors(3, 3), truth_mesh_n=16,
              inv_mesh_n=8, sensor_width=0.05, relative_noise=0.02, prior_r=5, prior_gamma=0.05,
              prior_delta=1.0, n_samples=40, beta=0.2, burn_in=10, map_max_iter=10)
    fast = run_bayesian_experiment(**kw)
    old = run_bayesian_experiment(**kw, likelihood_path="legacy")
    np.testing.assert_array_equal(fast.xi_map, old.xi_map)
    np.testing.assert_array_equal(fast.pcn.samples, old.pcn.samples)
    assert fast.likelihood_path == "forward_only" and old.likelihood_path == "legacy"
    assert fast.pcn_adjoint_solves == 0 and fast.pcn_forward_solves == fast.pcn.n_proposals + 1
    assert old.pcn_adjoint_solves == old.pcn.n_proposals + 1
    assert fast.map_diagnostics is not None
    with pytest.raises(ValueError):
        run_bayesian_experiment(**kw, likelihood_path="bogus")
