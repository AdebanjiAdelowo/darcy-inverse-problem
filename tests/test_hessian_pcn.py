"""Mathematical and regression tests for the Hessian-informed pCN sampler
(src/hessian_pcn.py)."""
import json

import numpy as np
import pytest
from scipy import stats

from src import mcmc_diagnostics as md
from src.hessian_pcn import (HessianInformedProposal, HessianPCNConfig, HpCNChain, load_checkpoint,
                             save_checkpoint)
from src.pcn_sampler import run_pcn


def _ridge_problem(d=6, seed=0):
    """Linear-Gaussian problem with a ridge (singular values 300..0.3) and one
    exactly unidentified direction: y = G xi + eps, prior N(0, I)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.r_[np.geomspace(300.0, 0.3, d - 1), 0.0]
    G = (U * s) @ np.linalg.qr(rng.standard_normal((d, d)))[0].T
    xi_true = rng.standard_normal(d)
    sigma = 1.0
    y = G @ xi_true + sigma * rng.standard_normal(d)
    H = G.T @ G / sigma**2
    cov = np.linalg.inv(H + np.eye(d))
    mean = cov @ G.T @ y / sigma**2
    phi = lambda xi: 0.5 * float(np.sum((G @ xi - y) ** 2) / sigma**2)
    return phi, H, mean, cov


def _prop(H, rho=0.5, **kw):
    return HessianInformedProposal.from_hessian(H, HessianPCNConfig(rho=rho, **kw))


# 1. prior invariance --------------------------------------------------------------

def test_proposal_operator_satisfies_prior_reversibility_identities():
    _, H, _, _ = _ridge_problem()
    p = _prop(H, rho=0.7)
    A, S = p.matrices()
    np.testing.assert_allclose(A, A.T, atol=1e-13)
    np.testing.assert_allclose(A @ A + S, np.eye(p.dim), atol=1e-12)
    assert np.all(p.beta > 0) and np.all(p.beta <= 1)


def test_one_step_from_prior_draws_returns_prior_draws():
    _, H, _, _ = _ridge_problem(d=4)
    p = _prop(H, rho=0.8)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50_000, 4))
    y = np.array([p.propose(xi, rng) for xi in x])
    np.testing.assert_allclose(np.cov(y.T), np.eye(4), atol=0.03)
    np.testing.assert_allclose(y.mean(0), 0.0, atol=0.02)
    for j in range(4):
        assert stats.kstest(y[:, j], "norm").pvalue > 1e-3


def test_chain_with_zero_likelihood_samples_the_prior():
    # moderate eigenvalues so that every direction takes O(1) steps under the prior
    H = np.diag([0.0, 0.5, 2.0, 3.0])
    ch = HpCNChain(lambda xi: 0.0, _prop(H, rho=0.9), np.zeros(4), seed=2)
    draws, _ = ch.step(40_000)
    assert ch.acceptance_rate == 1.0
    np.testing.assert_allclose(draws.var(axis=0), 1.0, atol=0.1)


def test_hessian_preconditioned_random_walk_is_not_prior_reversible():
    """Control: A = I, S = L L^T violates A^2 + S = I."""
    _, H, _, _ = _ridge_problem(d=4)
    L = np.linalg.cholesky(np.linalg.inv(H + np.eye(4)))
    assert np.max(np.abs(np.eye(4) + L @ L.T - np.eye(4))) > 1e-3


# 2. detailed balance --------------------------------------------------------------

def test_detailed_balance_with_respect_to_prior_and_posterior():
    phi, H, _, _ = _ridge_problem(d=5)
    p = _prop(H, rho=0.6)
    rng = np.random.default_rng(3)
    logmu = lambda x: -0.5 * x @ x
    for _ in range(50):
        x, x2 = rng.standard_normal(5), rng.standard_normal(5)
        assert logmu(x) + p.log_q(x, x2) == pytest.approx(logmu(x2) + p.log_q(x2, x), abs=1e-9)
        # full MH kernel, nonlinear Phi: pi(x) q(x,x') alpha(x,x') symmetric
        f = lambda z: phi(z) + 0.3 * np.sin(z).sum()
        lp = lambda z: -f(z) + logmu(z)
        la = lambda a, b: min(0.0, f(a) - f(b))
        assert lp(x) + p.log_q(x, x2) + la(x, x2) == pytest.approx(lp(x2) + p.log_q(x2, x) + la(x2, x), abs=1e-9)


def test_reduces_to_baseline_pcn_bitwise_when_hessian_is_zero():
    phi, _, _, _ = _ridge_problem(d=5)
    p = _prop(np.zeros((5, 5)), rho=0.3)
    ch = HpCNChain(phi, p, np.full(5, 0.2), seed=4)
    draws, _ = ch.step(300)
    ref = run_pcn(phi, r=5, n_samples=300, beta=0.3, burn_in=0, seed=4, x0=np.full(5, 0.2))
    np.testing.assert_array_equal(draws, ref.samples)
    assert ch.acceptance_rate == ref.acceptance_rate


# 3. Gaussian posterior moments ----------------------------------------------------

def _overdispersed(mean, cov, seed, factor=3.0):
    """Start over-dispersed relative to the posterior in every direction."""
    return mean + factor * np.linalg.cholesky(cov) @ np.random.default_rng(seed).standard_normal(len(mean))


def test_recovers_analytical_gaussian_posterior_on_a_ridge():
    phi, H, mean, cov = _ridge_problem(d=6)
    p = _prop(H, rho=0.9)
    chains = []
    for s in range(4):
        x0 = _overdispersed(mean, cov, 100 + s)
        ch = HpCNChain(phi, p, x0, seed=s)
        ch.step(1000)
        draws, _ = ch.step(8000)
        chains.append(draws)
    arr = np.stack(chains)
    for j in range(6):
        s = md.summarize(arr[:, :, j])
        assert s["rhat"] < 1.01 and s["ess_bulk"] > 1000
        assert abs(s["mean"] - mean[j]) < 4 * s["mcse_mean"] + 1e-3
    np.testing.assert_allclose(np.cov(arr.reshape(-1, 6).T), cov, atol=0.05 * np.max(np.diag(cov)))


def test_more_efficient_than_pcn_on_a_ridge_at_equal_cost():
    """Same starts, same number of likelihood evaluations; pCN beta is chosen for
    the narrowest direction, as it must be to have a usable acceptance rate."""
    phi, H, mean, cov = _ridge_problem(d=6)
    ess_h, ess_p = [], []
    hp = _prop(H, rho=0.9)
    for s in range(4):
        x0 = _overdispersed(mean, cov, 200 + s)
        ess_h.append(HpCNChain(phi, hp, x0, seed=s).step(4000)[0])
        ess_p.append(run_pcn(phi, r=6, n_samples=4000, beta=0.003, burn_in=0, seed=s, x0=x0).samples)
    h = min(md.ess_bulk(np.stack(ess_h)[:, 2000:, j]) for j in range(6))
    q = min(md.ess_bulk(np.stack(ess_p)[:, 2000:, j]) for j in range(6))
    assert h > 20 * q


# 4/5. constant mode and likelihood agreement --------------------------------------

def test_constant_mode_and_likelihood_agreement_on_darcy_problem():
    from src.bayesian_darcy import BayesianDarcyProblem
    from src.constant_mode import ReducedKL, reduced_neg_log_likelihood
    from src.forward import build_mesh
    from src.kl_prior import build_kl_prior
    from src.observation import build_observation_operator, grid_sensors
    dm = build_mesh(8)
    prior = build_kl_prior(dm, r=6, gamma=0.05, delta=1.0)
    H = build_observation_operator(dm.mesh, dm.V_p, grid_sensors(3, 3), width=0.05)
    rng = np.random.default_rng(1)
    bp = BayesianDarcyProblem(dm=dm, obs=H, y=0.5 + 0.02 * rng.standard_normal(9),
                              sigma=np.full(9, 0.02), prior=prior)
    rk = ReducedKL.from_prior(prior)
    phi_r = reduced_neg_log_likelihood(bp, rk)
    p = _prop(np.eye(rk.r_reduced) * 5.0, rho=0.5)
    assert p.dim == rk.r_reduced == prior.r - 1
    ch = HpCNChain(phi_r, p, np.zeros(rk.r_reduced), seed=5)
    draws, phis = ch.step(20)
    for x, f in zip(draws[-3:], phis[-3:]):
        full = rk.full_xi(x, 1.7)  # any constant-mode value
        assert f == bp.neg_log_likelihood(rk.full_xi(x, 0.0))
        assert bp.neg_log_likelihood(full) == pytest.approx(f, rel=1e-12)
        assert bp.neg_log_likelihood_legacy(rk.full_xi(x, 0.0)) == f


# 6/7. reproducibility and checkpoint/resume ---------------------------------------

def test_seeded_reproducibility_and_checkpoint_resume_equivalence(tmp_path):
    phi, H, _, _ = _ridge_problem(d=5)
    p = _prop(H, rho=0.5)
    x0 = np.full(5, 0.1)
    a = HpCNChain(phi, p, x0, seed=7)
    full_draws, full_phis = a.step(400, thin=2)
    b = HpCNChain(phi, p, x0, seed=7)
    np.testing.assert_array_equal(b.step(400, thin=2)[0], full_draws)
    c = HpCNChain(phi, p, x0, seed=7)
    d1, p1 = c.step(150, thin=2)
    save_checkpoint(tmp_path / "ck.npz", c, d1, p1, extra={"note": "x"})
    draws, phis, state, extra = load_checkpoint(tmp_path / "ck.npz")
    json.dumps(state)
    c2 = HpCNChain.from_state(phi, state)
    d2, p2 = c2.step(250, thin=2)
    np.testing.assert_array_equal(np.vstack([draws, d2]), full_draws)
    np.testing.assert_array_equal(np.r_[phis, p2], full_phis)
    assert (c2.n_steps, c2.n_accept) == (a.n_steps, a.n_accept)
    assert c2.n_lik_evals == a.n_lik_evals and extra == {"note": "x"}


# 8. finite outputs and input validation -------------------------------------------

def test_nonfinite_likelihood_is_rejected_and_outputs_stay_finite():
    phi = lambda x: np.inf if x[0] > 0.5 else 0.5 * x @ x
    ch = HpCNChain(phi, _prop(np.eye(3), rho=0.9), np.zeros(3), seed=8)
    draws, phis = ch.step(2000)
    assert np.all(np.isfinite(draws)) and np.all(np.isfinite(phis))
    assert ch.n_nonfinite > 0 and np.all(draws[:, 0] <= 0.5)


def test_rejects_invalid_hessians_and_configs():
    with pytest.raises(ValueError):
        _prop(np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        _prop(np.diag([1.0, -1.0]))
    with pytest.raises(ValueError):
        _prop(np.eye(2), rho=1.5)
    p = _prop(np.diag([1.0, -1e-12]))  # round-off negative eigenvalue is clipped
    assert p.lam.min() == 0.0


# 9. existing deterministic optimisation results preserved -------------------------

def test_preconditioned_headline_reconstruction_unchanged():
    """Milestone 3 headline smooth reconstruction (results/m3/optimizer/P)."""
    from src.experiment import run_reconstruction
    from src.observation import grid_sensors
    from src.truth_fields import smooth_truth
    r = run_reconstruction(smooth_truth(seed=1), grid_sensors(6, 6), 64, 32, 0.03, 0.02, 30.0,
                           data_seed=7, max_iter=200, optimizer="preconditioned")
    d = r.history.diagnostics
    assert d.converged
    assert d.final_objective == pytest.approx(40.16605132, rel=1e-8)
    assert d.extra["rel_error_meanfree_L2"] == pytest.approx(0.3468, abs=5e-5)
