"""Tests for the calibration machinery (src/calibration.py) on problems with
known posteriors, plus archive verification and confirmation-chain
initialisation."""
import importlib.util
import pathlib

import numpy as np
import pytest
from scipy import stats

from src import calibration as cal
from src.hessian_pcn import HessianInformedProposal, HessianPCNConfig, HpCNChain

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lin_gauss(d, seed):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d + 2, d)) * np.geomspace(10, 0.1, d)[None, :]
    sigma = 0.5
    cov = np.linalg.inv(G.T @ G / sigma**2 + np.eye(d))
    return G, sigma, cov


def _sbc_exact(n_data, d=3, L=99, shift=0.0, scale=1.0, seed=0):
    """SBC with exact posterior draws; shift/scale introduce known miscalibration."""
    G, sigma, cov = _lin_gauss(d, seed)
    Lc = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed + 1)
    ranks, pits, means, sds, truths = [], [], [], [], []
    for _ in range(n_data):
        xi = rng.standard_normal(d)
        y = G @ xi + sigma * rng.standard_normal(G.shape[0])
        mean = cov @ G.T @ y / sigma**2 + shift * np.sqrt(np.diag(cov))
        draws = mean + scale * (Lc @ rng.standard_normal((d, L))).T
        ranks.append(cal.rank_of_truth(draws[:, 0], xi[0]))
        sd = scale * np.sqrt(cov[0, 0])
        pits.append(cal.gaussian_pit(mean[0], sd, xi[0]))
        means.append(mean[0]); sds.append(sd); truths.append(xi[0])
    return np.array(ranks), np.array(pits), np.array(means), np.array(sds), np.array(truths)


def test_exact_posterior_sbc_is_uniform_and_nominally_covered():
    ranks, pits, means, sds, truths = _sbc_exact(1000)
    assert cal.rank_chi2(ranks, 99)["p"] > 1e-3
    assert cal.pit_ks(pits)["p"] > 1e-3
    cov = cal.coverage_summary(pits)
    assert all(v["nominal_inside_ci"] for v in cov.values())
    s = cal.bias_rmse_spread(means, sds, truths)
    assert s["spread_skill_ci95"][0] < 1.0 < s["spread_skill_ci95"][1]
    assert abs(s["standardized_bias"]) < 4 * s["standardized_bias_se"]


@pytest.mark.parametrize("shift,scale", [(1.0, 1.0), (0.0, 0.5)])
def test_miscalibrated_posteriors_are_detected(shift, scale):
    ranks, pits, means, sds, truths = _sbc_exact(1000, shift=shift, scale=scale)
    assert cal.rank_chi2(ranks, 99)["p"] < 1e-4
    cov = cal.coverage_summary(pits)
    assert any(not v["nominal_inside_ci"] for v in cov.values())
    if scale < 1:
        assert cal.bias_rmse_spread(means, sds, truths)["spread_skill_ci95"][0] > 1.5


def test_hpcn_on_linear_gaussian_sbc_is_consistent_with_calibration():
    """SBC of the actual HpCN sampler (Hessian = exact G^T G / sigma^2)."""
    d, L = 3, 99
    G, sigma, cov = _lin_gauss(d, 3)
    prop = HessianInformedProposal.from_hessian(G.T @ G / sigma**2, HessianPCNConfig(rho=0.7))
    Lc = np.linalg.cholesky(cov)
    rng = np.random.default_rng(4)
    ranks = []
    for i in range(200):
        xi = rng.standard_normal(d)
        y = G @ xi + sigma * rng.standard_normal(G.shape[0])
        mean = cov @ G.T @ y / sigma**2
        phi = lambda x, y=y: 0.5 * float(np.sum((G @ x - y) ** 2)) / sigma**2
        post = []
        for c in range(4):
            ch = HpCNChain(phi, prop, mean + 3 * Lc @ rng.standard_normal(d), seed=1000 * i + c)
            ch.step(300)
            post.append(ch.step(1000)[0])
        draws = cal.select_rank_draws(np.stack(post), L)
        ranks.append(cal.rank_of_truth(draws[:, 0], xi[0]))
    assert cal.rank_chi2(np.array(ranks), L)["p"] > 1e-3


def test_rank_and_selection_helpers():
    assert cal.rank_of_truth(np.array([0.1, 0.2, 0.3]), 0.25) == 2
    r = [cal.rank_of_truth(np.array([1.0, 1.0]), 1.0, np.random.default_rng(s)) for s in range(200)]
    assert set(r) == {0, 1, 2}
    post = np.arange(4 * 100).reshape(4, 100)
    sel = cal.select_rank_draws(post, 99)
    assert len(sel) == 99 and len(set(sel.tolist())) == 99
    assert [int(np.sum((sel >= 100 * c) & (sel < 100 * (c + 1)))) for c in range(4)] == [25, 25, 25, 24]
    with pytest.raises(ValueError):
        cal.rank_chi2(np.zeros(10, int), 98)


def test_clopper_pearson_and_exclusion_threshold():
    lo, hi = cal.clopper_pearson(45, 50)
    assert lo == pytest.approx(0.7819, abs=1e-3) and hi == pytest.approx(0.9667, abs=1e-3)
    assert cal.clopper_pearson(0, 10)[0] == 0.0 and cal.clopper_pearson(10, 10)[1] == 1.0
    assert cal.max_coverage_exclusions(100) == int(stats.binom.ppf(0.99, 100, 0.05))


def test_rao_blackwell_pit_and_exact_constant_mode():
    # constant-mode-only quantity: posterior N(0, 1) exactly -> PIT = Phi(truth), uniform under the prior
    xs = np.random.default_rng(5).standard_normal(5000)
    pits = np.array([cal.rb_pit(np.zeros(1), 1.0, x) for x in xs])
    np.testing.assert_allclose(pits, stats.norm.cdf(xs), atol=1e-14)
    assert cal.pit_ks(pits)["p"] > 1e-3
    # identifiable part ~ N(0.3, 0.5^2) by draws plus exact N(0, 1): PIT of X = V + Z
    v = np.random.default_rng(6).normal(0.3, 0.5, 200_000)
    assert cal.rb_pit(v, 1.0, 1.0) == pytest.approx(stats.norm.cdf((1.0 - 0.3) / np.sqrt(1.25)), abs=2e-3)
    assert cal.covered_by_pit(0.5, 0.5) and not cal.covered_by_pit(0.99, 0.95)


def test_manifest_verification_detects_changes(tmp_path):
    mod = _load_script("make_chain_manifest")
    d = tmp_path / "chains"
    d.mkdir()
    for k in range(2):
        np.savez(d / f"c{k}.npz", draws=np.arange(5.0) + k)
    entries = [{"path": f"chains/c{k}.npz", "bytes": (d / f"c{k}.npz").stat().st_size,
                "sha256": mod.sha256(d / f"c{k}.npz")} for k in range(2)]
    import json
    man = tmp_path / "MANIFEST.json"
    man.write_text(json.dumps({"files": entries}))
    assert mod.verify(man, root=tmp_path) == []
    np.savez(d / "c1.npz", draws=np.arange(5.0) + 7)
    (d / "c0.npz").unlink()
    probs = mod.verify(man, root=tmp_path)
    assert any("missing" in p for p in probs) and any("mismatch" in p for p in probs)


def test_confirmation_chain_starts_are_independent_and_deterministic():
    mod = _load_script("m4_hpcn")
    H = np.diag([0.0, 1.0, 100.0, 1e4])
    rec = {"xi_r_map": [0.1, -0.2, 0.3, 0.4]}
    a = mod._start(rec, H, "laplace3", 21)
    np.testing.assert_array_equal(a, mod._start(rec, H, "laplace3", 21))
    starts = np.array([mod._start(rec, H, "laplace3", s) for s in range(21, 2021)])
    assert len({tuple(x) for x in starts[:4]}) == 4
    offs = starts - np.array(rec["xi_r_map"])
    np.testing.assert_allclose(offs.var(axis=0), 9.0 / (1.0 + np.diag(H)), rtol=0.12)
    # disjoint from the Milestone 4 production starts (seeds 11-14)
    m4 = {tuple(mod._start(rec, H, "laplace3", s)) for s in (11, 12, 13, 14)}
    assert not m4 & {tuple(mod._start(rec, H, "laplace3", s)) for s in (21, 22, 23, 24)}


def test_checkpoint_records_configuration_provenance():
    prop = HessianInformedProposal.from_hessian(np.eye(2), HessianPCNConfig(rho=0.5))
    st = HpCNChain(lambda x: 0.0, prop, np.zeros(2), seed=21).state_dict()
    assert st["seed"] == 21 and st["proposal"]["config"]["rho"] == 0.5
    assert "rng_state" in st and st["proposal"]["eigenvalues"] == [1.0, 1.0]
