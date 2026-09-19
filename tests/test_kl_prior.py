import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix

from src.forward import build_mesh
from src.kl_prior import build_kl_prior


def _mass_matrix(dm):
    u = ufl.TrialFunction(dm.V_m)
    v = ufl.TestFunction(dm.V_m)
    Mmat = assemble_matrix(fem.form(ufl.inner(u, v) * ufl.dx))
    Mmat.assemble()
    return Mmat.convert("dense").getDenseArray()


def test_eigenvectors_are_M_orthonormal():
    dm = build_mesh(10)
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    M = _mass_matrix(dm)
    gram = prior.eigenvectors.T @ M @ prior.eigenvectors
    assert np.max(np.abs(gram - np.eye(prior.r))) < 1e-8


def test_eigenvalues_are_positive_and_descending():
    dm = build_mesh(10)
    prior = build_kl_prior(dm, r=8, gamma=0.05, delta=1.0)
    assert np.all(prior.eigenvalues > 0)
    assert np.all(np.diff(prior.eigenvalues) <= 1e-12)


def test_captured_variance_increases_with_r():
    dm = build_mesh(12)
    fracs = []
    for r in [5, 10, 20]:
        prior = build_kl_prior(dm, r=r, gamma=0.05, delta=1.0)
        fracs.append(prior.captured_variance_fraction())
    assert fracs[0] < fracs[1] < fracs[2] <= 1.0 + 1e-9


def test_sample_reproducible_with_seed():
    dm = build_mesh(10)
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    xi1 = rng1.standard_normal(prior.r)
    xi2 = rng2.standard_normal(prior.r)
    m1 = prior.sample(xi1)
    m2 = prior.sample(xi2)
    assert np.array_equal(m1.x.array, m2.x.array)


def test_sample_mean_matches_m0_for_large_ensemble():
    """The KL sample mean over many draws should converge toward m0 (law of
    large numbers on the xi_j ~ N(0,1) coefficients)."""
    dm = build_mesh(10)
    prior = build_kl_prior(dm, r=10, gamma=0.05, delta=1.0, m0_value=0.3)
    rng = np.random.default_rng(0)
    n_draws = 500
    total = np.zeros(prior.n_dofs)
    for _ in range(n_draws):
        xi = rng.standard_normal(prior.r)
        total += prior.sample_array(xi)
    empirical_mean = total / n_draws
    # mean absolute deviation over all dofs, not max (max-over-many-correlated
    # -nodes is extreme-value-inflated and not a stable MC convergence check)
    assert np.mean(np.abs(empirical_mean - 0.3)) < 0.1


def test_larger_gamma_gives_smoother_samples():
    """Larger correlation-length parameter gamma should give samples with
    smaller H1-seminorm (less high-frequency content) for the same xi."""
    dm = build_mesh(16)
    rng = np.random.default_rng(0)
    prior_rough = build_kl_prior(dm, r=15, gamma=0.01, delta=1.0)
    prior_smooth = build_kl_prior(dm, r=15, gamma=0.5, delta=1.0)
    xi = rng.standard_normal(15)

    def h1_seminorm_value(prior, xi):
        m = prior.sample(xi)
        form = fem.form(ufl.inner(ufl.grad(m), ufl.grad(m)) * ufl.dx)
        return fem.assemble_scalar(form)

    assert h1_seminorm_value(prior_smooth, xi) < h1_seminorm_value(prior_rough, xi)
