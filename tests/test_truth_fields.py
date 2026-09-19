import numpy as np

from src.truth_fields import smooth_truth, structured_truth
from src.forward import build_mesh
from dolfinx.fem import Function


def test_smooth_truth_reproducible_with_seed():
    dm = build_mesh(10)
    m1 = Function(dm.V_m); m1.interpolate(smooth_truth(seed=3))
    m2 = Function(dm.V_m); m2.interpolate(smooth_truth(seed=3))
    m3 = Function(dm.V_m); m3.interpolate(smooth_truth(seed=4))
    assert np.array_equal(m1.x.array, m2.x.array)
    assert not np.array_equal(m1.x.array, m3.x.array)


def test_smooth_truth_is_bounded_and_finite():
    dm = build_mesh(20)
    m = Function(dm.V_m)
    m.interpolate(smooth_truth(seed=0, amplitude=1.0))
    assert np.all(np.isfinite(m.x.array))
    assert np.max(np.abs(m.x.array)) < 10.0  # sanity bound, not tight


def test_structured_truth_has_two_localized_extrema():
    dm = build_mesh(40)
    m = Function(dm.V_m)
    m.interpolate(structured_truth())
    x = dm.V_m.tabulate_dof_coordinates()
    # near the two specified centres, m should be close to the specified amplitudes
    idx_center1 = np.argmin((x[:, 0] - 0.3) ** 2 + (x[:, 1] - 0.65) ** 2)
    idx_center2 = np.argmin((x[:, 0] - 0.7) ** 2 + (x[:, 1] - 0.3) ** 2)
    assert m.x.array[idx_center1] > 1.0
    assert m.x.array[idx_center2] < -0.9


def test_structured_truth_background_is_near_zero():
    dm = build_mesh(20)
    m = Function(dm.V_m)
    m.interpolate(structured_truth())
    x = dm.V_m.tabulate_dof_coordinates()
    idx_far = np.argmin((x[:, 0] - 0.05) ** 2 + (x[:, 1] - 0.05) ** 2)  # far corner
    assert abs(m.x.array[idx_far]) < 0.1
