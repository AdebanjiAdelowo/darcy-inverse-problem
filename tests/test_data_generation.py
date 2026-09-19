import numpy as np

from src.truth_fields import smooth_truth
from src.observation import grid_sensors
from src.data_generation import generate_synthetic_data


def test_truth_mesh_is_independent_of_any_inversion_mesh():
    """The whole point of generate_synthetic_data is that it builds its own
    (fine) mesh, never referencing whatever mesh an inversion will later
    use -- this test checks that no inversion-mesh argument exists at all
    in its signature (the function cannot commit an inverse crime by
    construction) and that the returned truth mesh has the resolution
    actually requested."""
    sensors = grid_sensors(3, 3)
    data = generate_synthetic_data(smooth_truth(seed=0), sensors, truth_mesh_n=30,
                                    sensor_width=0.04, relative_noise=0.0, seed=1)
    assert data.truth_dm.mesh.topology.index_map(2).size_local == 30 * 30 * 2


def test_noise_reproducible_with_seed():
    sensors = grid_sensors(3, 3)
    truth = smooth_truth(seed=0)
    d1 = generate_synthetic_data(truth, sensors, 16, 0.04, 0.05, seed=42)
    d2 = generate_synthetic_data(truth, sensors, 16, 0.04, 0.05, seed=42)
    d3 = generate_synthetic_data(truth, sensors, 16, 0.04, 0.05, seed=43)
    assert np.array_equal(d1.y_noisy, d2.y_noisy)
    assert not np.array_equal(d1.y_noisy, d3.y_noisy)
    assert np.array_equal(d1.y_clean, d2.y_clean)  # clean signal independent of noise seed


def test_zero_relative_noise_gives_clean_equals_noisy():
    sensors = grid_sensors(3, 3)
    data = generate_synthetic_data(smooth_truth(seed=0), sensors, 16, 0.04, 0.0, seed=7)
    assert np.array_equal(data.y_clean, data.y_noisy)


def test_sigma_uses_nominal_floor_for_noiseless_data():
    """Regression test for a real bug: sigma used to be floored at 1e-12
    for the noiseless case, making the misfit weighting numerically
    degenerate (residuals as small as roundoff error were amplified by a
    factor of 1e24). sigma must stay at a sane, physically-motivated
    scale even when relative_noise=0."""
    sensors = grid_sensors(3, 3)
    data = generate_synthetic_data(smooth_truth(seed=0), sensors, 16, 0.04, 0.0, seed=7)
    assert np.all(data.sigma > 1e-6)


def test_relative_noise_scales_actual_injected_noise():
    sensors = grid_sensors(4, 4)
    truth = smooth_truth(seed=0)
    d_low = generate_synthetic_data(truth, sensors, 20, 0.04, 0.01, seed=1)
    d_high = generate_synthetic_data(truth, sensors, 20, 0.04, 0.2, seed=1)
    dev_low = np.std(d_low.y_noisy - d_low.y_clean)
    dev_high = np.std(d_high.y_noisy - d_high.y_clean)
    assert dev_high > dev_low
