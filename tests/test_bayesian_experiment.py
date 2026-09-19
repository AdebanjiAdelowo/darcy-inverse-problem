"""Tests for the end-to-end Bayesian experiment runner: posterior sample
shapes/statistics and the KL-coefficient <-> FEM-field conversion, on a
tiny, fast problem (not a full PDE-posterior MCMC study)."""
import numpy as np

from src.truth_fields import smooth_truth
from src.observation import grid_sensors
from src.bayesian_experiment import run_bayesian_experiment, integrated_posterior_variance


def _tiny_result():
    sensors = grid_sensors(3, 3)
    truth_fn = smooth_truth(seed=1)
    return run_bayesian_experiment(
        truth_fn, sensors, truth_mesh_n=12, inv_mesh_n=8, sensor_width=0.05,
        relative_noise=0.02, prior_r=5, prior_gamma=0.05, prior_delta=1.0,
        n_samples=100, beta=0.05, burn_in=50, map_max_iter=20, data_seed=1, mcmc_seed=2,
    )


def test_posterior_sample_shapes():
    result = _tiny_result()
    n_dofs = result.bp.dm.V_m.dofmap.index_map.size_global
    assert result.pcn.samples.shape == (100, 5)
    assert result.m_samples.shape == (100, n_dofs)
    assert result.m_post_mean.shape == (n_dofs,)
    assert result.m_post_std.shape == (n_dofs,)
    assert result.m_map.shape == (n_dofs,)
    assert result.ess_per_dim.shape == (5,)


def test_kl_to_fem_conversion_is_consistent_between_prior_and_result():
    result = _tiny_result()
    m_from_prior = result.prior.sample_array(result.xi_map)
    assert np.allclose(m_from_prior, result.m_map)


def test_posterior_mean_matches_manual_average_of_samples():
    result = _tiny_result()
    manual_mean = result.m_samples.mean(axis=0)
    assert np.allclose(manual_mean, result.m_post_mean)


def test_posterior_std_is_nonnegative():
    result = _tiny_result()
    assert np.all(result.m_post_std >= 0.0)


def test_integrated_posterior_variance_is_nonnegative():
    result = _tiny_result()
    assert integrated_posterior_variance(result) >= 0.0


def test_reproducible_with_fixed_seeds():
    def run():
        sensors = grid_sensors(3, 3)
        truth_fn = smooth_truth(seed=1)
        return run_bayesian_experiment(
            truth_fn, sensors, truth_mesh_n=12, inv_mesh_n=8, sensor_width=0.05,
            relative_noise=0.02, prior_r=5, prior_gamma=0.05, prior_delta=1.0,
            n_samples=50, beta=0.05, burn_in=20, map_max_iter=15, data_seed=1, mcmc_seed=2,
        )

    r1 = run()
    r2 = run()
    assert np.array_equal(r1.pcn.samples, r2.pcn.samples)
    assert np.array_equal(r1.m_map, r2.m_map)
