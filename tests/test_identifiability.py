"""Regression tests for the constant-scaling non-identifiability of the
zero-source Darcy problem and for the KL prior's constant mode
(src/identifiability.py). These document mathematical facts about the
existing formulation; nothing in the inverse problem, prior or sampler is
changed by them."""
import pathlib

import numpy as np
import pytest
from dolfinx.fem import Function

from src.adjoint import InverseProblem
from src.bayesian_darcy import BayesianDarcyProblem
from src.forward import build_mesh, solve_forward
from src.identifiability import (
    check_posterior_variance_floor, constant_mode_index, constant_mode_variance_floor,
    l2_mean, mass_matrix, reconstruction_errors,
)
from src.kl_prior import build_kl_prior
from src.observation import build_observation_operator, grid_sensors

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIFTS = [-1.3, 0.4, 2.0]


@pytest.fixture(scope="module")
def setup():
    dm = build_mesh(8)
    sensors = grid_sensors(3, 3)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.05)
    rng = np.random.default_rng(11)
    n = dm.V_m.dofmap.index_map.size_global
    m = 0.6 * rng.standard_normal(n)  # strongly heterogeneous, non-constant
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    sigma = np.full(H.n_sensors, 0.02)
    return dm, H, m, y, sigma


def _pressure(dm, m_array):
    m = Function(dm.V_m)
    m.x.array[:] = m_array
    return solve_forward(dm, m).x.array.copy()


@pytest.mark.parametrize("c", SHIFTS)
def test_pressure_field_invariant_under_constant_log_shift(setup, c):
    """k -> exp(c) k (positive constant scaling) leaves the full pressure field unchanged."""
    dm, _, m, _, _ = setup
    p0 = _pressure(dm, m)
    p1 = _pressure(dm, m + c)
    assert np.max(np.abs(p1 - p0)) < 1e-12
    assert np.ptp(p0) > 0.5  # non-trivial flow: p spans the 1 -> 0 Dirichlet drop


def test_pressure_changes_under_nonconstant_perturbation(setup):
    """Control: the invariance is specific to constant shifts."""
    dm, _, m, _, _ = setup
    x = dm.V_m.tabulate_dof_coordinates()[:, 0]
    p0 = _pressure(dm, m)
    p1 = _pressure(dm, m + 0.5 * x)
    assert np.max(np.abs(p1 - p0)) > 1e-3


@pytest.mark.parametrize("c", SHIFTS)
def test_misfit_and_h1_objective_invariant_under_constant_shift(setup, c):
    dm, H, m, y, sigma = setup
    ip = InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=0.3, reg_type="h1")
    mis0, _ = ip.misfit_only(m)
    mis1, _ = ip.misfit_only(m + c)
    assert abs(mis1 - mis0) <= 1e-12 * abs(mis0)
    J0, *_ = ip.objective_and_gradient(m)
    J1, *_ = ip.objective_and_gradient(m + c)
    assert abs(J1 - J0) <= 1e-12 * abs(J0)


def test_l2_regulariser_is_not_shift_invariant(setup):
    """Control: the L2-to-reference regulariser does penalise the constant direction."""
    dm, H, m, y, sigma = setup
    ip = InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=0.3, reg_type="l2")
    J0, *_ = ip.objective_and_gradient(m)
    J1, *_ = ip.objective_and_gradient(m + 0.4)
    # the misfit is unchanged, so the whole change is the L2 penalty change
    # (alpha/2)(||m + c||^2 - ||m||^2), far above the 1e-12 invariance tolerance
    M = mass_matrix(dm.V_m)
    expected = 0.5 * 0.3 * ((m + 0.4) @ (M @ (m + 0.4)) - m @ (M @ m))
    assert abs(J1 - J0) > 1e-6 * abs(J0)
    assert (J1 - J0) == pytest.approx(expected, rel=1e-8)


def test_misfit_gradient_is_orthogonal_to_constant_direction(setup):
    """The directional derivative of the misfit along m -> m + c*1 vanishes
    (to the quadrature-level consistency of the discrete gradient)."""
    dm, H, m, y, sigma = setup
    ip = InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=0.0, reg_type="h1")
    _, g, _, _ = ip.objective_and_gradient(m)
    ones = np.ones_like(g) / np.sqrt(g.size)
    assert abs(g @ ones) <= 1e-3 * np.linalg.norm(g)


@pytest.mark.parametrize("delta", [1.0, 2.0])
def test_kl_prior_first_mode_is_constant_with_lambda_delta_minus_two(setup, delta):
    dm = setup[0]
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=delta)
    M = mass_matrix(dm.V_m)
    assert constant_mode_index(prior) == 0
    floor = constant_mode_variance_floor(prior, M)
    assert floor["lambda"] == pytest.approx(delta ** -2, rel=1e-10)
    assert abs(floor["phi_value"]) == pytest.approx(1.0, rel=1e-10)  # 1/sqrt(|Omega|), |Omega|=1
    assert floor["area"] == pytest.approx(1.0, rel=1e-12)
    assert floor["integrated_floor"] == pytest.approx(delta ** -2, rel=1e-10)
    np.testing.assert_allclose(floor["pointwise_floor"], delta ** -2, rtol=1e-10)


def test_likelihood_independent_of_constant_kl_coefficient(setup):
    dm, H, _, y, sigma = setup
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    bp = BayesianDarcyProblem(dm=dm, obs=H, y=y, sigma=sigma, prior=prior)
    xi = np.random.default_rng(5).standard_normal(prior.r)
    phi0 = bp.neg_log_likelihood(xi)
    for c in SHIFTS:
        xi_c = xi.copy()
        xi_c[0] += c
        assert abs(bp.neg_log_likelihood(xi_c) - phi0) <= 1e-12 * abs(phi0)
    _, grad, _ = bp.neg_log_likelihood_and_grad(xi)
    assert abs(grad[0]) <= 1e-3 * np.linalg.norm(grad)


def test_reconstruction_error_metrics_separate_mean_and_mean_free_parts(setup):
    dm = setup[0]
    M = mass_matrix(dm.V_m)
    rng = np.random.default_rng(3)
    m_true = rng.standard_normal(M.shape[0])
    m_est = m_true + 0.1 * rng.standard_normal(M.shape[0])
    base = reconstruction_errors(m_est, m_true, M)
    shifted = reconstruction_errors(m_est + 0.8, m_true, M)
    for key in ("rel_error_meanfree_nodal", "rel_error_meanfree_L2", "rel_error_contrast_L2"):
        assert shifted[key] == pytest.approx(base[key], rel=1e-10)
    assert shifted["rel_error_abs_L2"] > base["rel_error_abs_L2"]
    assert shifted["mean_offset_L2"] == pytest.approx(base["mean_offset_L2"] + 0.8, rel=1e-10)
    assert l2_mean(np.full(M.shape[0], 2.5), M) == pytest.approx(2.5, rel=1e-12)
    exact = reconstruction_errors(m_true + 0.3, m_true, M)
    assert exact["rel_error_meanfree_L2"] < 1e-12 and exact["rel_error_contrast_L2"] < 1e-12


def test_historical_full_chains_violate_exact_variance_floor():
    """Regression record of the Milestone 1 finding: the stored full-config
    posterior standard deviations fall below the exact constant-mode floor
    (Var[m(x)|y] >= 1 for delta=1), so they cannot be correct estimates of the
    exact discretised posterior variance."""
    dm = build_mesh(16)  # historical Bayesian inversion mesh (289 P1 dofs)
    prior = build_kl_prior(dm, r=15, gamma=0.05, delta=1.0)
    M = mass_matrix(dm.V_m)
    for truth in ("smooth", "structured"):
        d = np.load(ROOT / "results" / f"posterior_samples_{truth}_full.npz")
        np.testing.assert_allclose(d["eigenvalues"], prior.eigenvalues, rtol=1e-8)
        chk = check_posterior_variance_floor(d["m_post_std"], prior, M)
        assert chk["violates_floor"]
        assert chk["pointwise_variance_max"] < 1.0
        assert chk["integrated_variance"] < chk["integrated_floor"] == pytest.approx(1.0, rel=1e-10)
