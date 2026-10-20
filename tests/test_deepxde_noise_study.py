"""CPU checks for the noise-robustness study (no training beyond a few steps)."""
import json
import math
import os

import numpy as np
import pytest

from src import deepxde_inverse_spatial_k as spatial
from src.deepxde_noise_methods import METHODS, TAU, method_objective


# ---------------------------------------------------------------- numpy only

def test_legacy_noise_stream_is_unchanged():
    xy, values = spatial.observation_data(3, 64, noise_std=0.01)
    rng = np.random.default_rng(3)
    xy_ref = rng.uniform(0.05, 0.95, (64, 2))
    ref = spatial.exact_pressure(xy_ref) + rng.normal(0.0, 0.01, size=(64, 1))
    assert np.array_equal(xy, xy_ref) and np.array_equal(values, ref)


def test_independent_noise_stream_separates_layout_and_noise():
    clean_xy, clean = spatial.observation_data(42, 128)
    xy_a, va = spatial.observation_data(42, 128, 0.01, noise_seed=0)
    xy_b, vb = spatial.observation_data(42, 128, 0.01, noise_seed=1)
    xy_c, vc = spatial.observation_data(7, 128, 0.01, noise_seed=0)
    assert np.array_equal(xy_a, clean_xy) and np.array_equal(xy_a, xy_b)  # layout fixed
    assert not np.allclose(va, vb)                                          # noise varies
    assert np.allclose(va - clean, vc - spatial.exact_pressure(xy_c))       # same noise, new layout
    same_seed = spatial.observation_data(42, 128, 0.01, noise_seed=42)[1] - clean
    legacy = spatial.observation_data(42, 128, 0.01)[1] - clean
    assert not np.allclose(same_seed, legacy)  # equal seeds still give independent streams
    z = spatial.noise_draws(5, 256)
    assert np.array_equal(spatial.noise_draws(5, 32), z[:32])               # prefix-nested
    assert np.allclose(spatial.observation_data(42, 128, 0.05, noise_seed=0)[1] - clean,
                       5 * (va - clean))                                    # paired across sigma
    assert abs(z.std() - 1) < 0.15


def test_method_objectives_match_the_formulas():
    counts = (576, 64, 128)
    assert method_objective("baseline", 0.05, counts) == (None, 0.0, 0.05)
    w, c, s = method_objective("naive_inverse_variance", 0.01, counts)
    assert w == [1.0, 1.0, 1.0 / 0.01**2] and c == 0.0
    w, c, s = method_objective("likelihood_weighted", 0.05, counts)
    assert math.isclose(w[0], 576 / 128 * 0.05**2 / TAU**2) and w[2] == 1.0
    assert math.isclose(w[1], 64 / 128 * 0.05**2 / TAU**2)
    _, c1, _ = method_objective("likelihood_prior_sd1", 0.05, counts)
    _, c2, _ = method_objective("likelihood_prior_sd0.25", 0.05, counts)
    assert math.isclose(c1, 0.05**2 / 128) and math.isclose(c2, 16 * c1)
    assert method_objective("likelihood_weighted", 0.0, counts)[2] == 0.01  # nominal sigma
    with pytest.raises(ValueError):
        method_objective("other", 0.01, counts)


def test_noise_runner_arguments():
    from remote.run_deepxde_noise_study import parse_args
    a = parse_args(["--tuples", "42:42:42", "1:42:42", "--noise-std", "0.01", "--output", "x"])
    assert a.tuples == [(42, 42, 42), (1, 42, 42)] and a.methods == list(METHODS)
    for bad in (["--tuples", "42:42"], ["--tuples", "1:1:1", "1:1:1"],
                ["--tuples", "1:1:1", "--methods", "nope"],
                ["--tuples", "1:1:1", "--adam-snapshots", "50000"],
                ["--tuples", "1:1:1", "--init-from", "d", "--adam-snapshots", "10"]):
        with pytest.raises(ValueError):
            parse_args([*bad, "--noise-std", "0.01", "--output", "x"])


# ------------------------------------------------------------ DeepXDE on CPU

@pytest.fixture(scope="module")
def dde_cpu():
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    torch = pytest.importorskip("torch")
    dde = pytest.importorskip("deepxde")
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    yield dde, torch
    torch.set_default_device(previous)
    dde.config.set_default_float("float32")


def test_loss_counts_and_penalty_gradient(dde_cpu):
    from src.deepxde_lbfgs import refine_lbfgs
    from src.deepxde_noise_methods import loss_counts, prior_penalty
    dde, torch = dde_cpu
    model, alpha, beta = spatial.build_model(seed=1, noise_std=0.01, noise_seed=0)
    assert loss_counts(model) == (576, 64, 128)
    pen = prior_penalty([alpha, beta], 0.5)
    with torch.no_grad():
        alpha.fill_(0.2)
        beta.fill_(-0.4)
    value = pen()
    value.backward()
    assert math.isclose(float(value.detach()), 0.5 * (0.04 + 0.16), rel_tol=1e-6)
    assert math.isclose(float(alpha.grad), 0.2, rel_tol=1e-5)  # d/dalpha 0.5 alpha^2 * 2
    alpha.grad = None
    beta.grad = None
    assert prior_penalty([alpha, beta], 0.0) is None
    # A dominant penalty drives the parameters towards zero during refinement.
    model.train(iterations=2, display_every=1, verbose=0)
    info = refine_lbfgs(model, {"alpha": alpha, "beta": beta}, maxiter=20,
                        penalty=prior_penalty([alpha, beta], 1e3))
    assert abs(float(alpha.detach())) < 1e-2 and abs(float(beta.detach())) < 1e-2
    assert info["final_penalty"] >= 0 and info["loss_weights"] is None


def test_runner_pairs_methods_and_snapshots_on_cpu(dde_cpu, tmp_path, monkeypatch):
    from remote import run_deepxde_noise_study as runner
    from src.deepxde_lbfgs import refine_lbfgs
    dde, torch = dde_cpu
    monkeypatch.setattr(runner, "require_remote_cuda", lambda: None)
    out = tmp_path / "noise.json"
    runner.main(["--tuples", "42:42:0", "--noise-std", "0.01", "--iterations", "3",
                 "--adam-snapshots", "2", "--lbfgs-maxiter", "4", "--trace-every", "1",
                 "--save-state", str(tmp_path / "st"), "--output", str(out)])
    (record,) = json.loads(out.read_text())["records"]
    assert record["status"] == "ok" and record["loss_counts"] == [576, 64, 128]
    seen = [(r["adam_iterations"], r["method"]) for r in record["refinements"]]
    assert seen == [(b, m) for b in (2, 3) for m in METHODS]
    weighted = [r for r in record["refinements"] if r["method"] == "likelihood_weighted"][0]
    assert weighted["lbfgs"]["loss_weights"][0] > 100
    raw, comp = weighted["lbfgs"]["final_raw_loss_components"], weighted["lbfgs"]["final_loss_components"]
    assert math.isclose(raw[0] * weighted["lbfgs"]["loss_weights"][0], comp[0], rel_tol=1e-6)
    # Each method starts from the identical Adam state: the logged first
    # evaluation of every 3-iteration refinement has the same parameters.
    starts = {tuple(r["lbfgs"]["evaluation_history"]["values"][0][1:])
              for r in record["refinements"] if r["adam_iterations"] == 3}
    assert len(starts) == 1
    assert list(starts)[0] == (record["adam"]["alpha_estimated"], record["adam"]["beta_estimated"])
    # Baseline from the 3-iteration state equals a plain Adam(3) + refinement run.
    model, alpha, beta = spatial.build_model(seed=42, observation_seed=42, noise_std=0.01,
                                             noise_seed=0)
    model.train(iterations=3, display_every=1, verbose=0)
    refine_lbfgs(model, {"alpha": alpha, "beta": beta}, maxiter=4)
    base3 = [r for r in record["refinements"] if r["adam_iterations"] == 3
             and r["method"] == "baseline"][0]
    assert base3["alpha_estimated"] == float(alpha.detach())
    # The 2-iteration snapshot equals a separate 2-iteration Adam run.
    model, alpha, beta = spatial.build_model(seed=42, observation_seed=42, noise_std=0.01,
                                             noise_seed=0)
    model.train(iterations=2, display_every=1, verbose=0)
    refine_lbfgs(model, {"alpha": alpha, "beta": beta}, maxiter=4)
    base2 = [r for r in record["refinements"] if r["adam_iterations"] == 2
             and r["method"] == "baseline"][0]
    assert base2["alpha_estimated"] == float(alpha.detach())
    assert (tmp_path / "st" / "tuple42_42_0_adam.pt").exists()


def test_float64_dtypes_and_checkpoint_rules(dde_cpu, tmp_path, monkeypatch):
    from remote import run_deepxde_noise_study as runner
    dde, torch = dde_cpu
    monkeypatch.setattr(runner, "require_remote_cuda", lambda: None)
    out = tmp_path / "f64.json"
    runner.main(["--tuples", "42:42:0", "--noise-std", "0.01", "--iterations", "2",
                 "--lbfgs-maxiter", "3", "--methods", "baseline", "--float64",
                 "--output", str(out)])
    (record,) = json.loads(out.read_text())["records"]
    assert record["status"] == "ok"
    assert record["dtypes"] == {"network": "torch.float64", "external_variables": "torch.float64",
                                "inputs": "float64", "losses": "torch.float64",
                                "torch_default": "torch.float64"}
    dde.config.set_default_float("float32")
    torch.set_default_dtype(torch.float32)
    # Checkpoint mismatch is refused.
    ckdir = tmp_path / "ck"
    ckdir.mkdir()
    model, alpha, beta = spatial.build_model(seed=0)
    torch.save({"net": model.net.state_dict(), "alpha": 0.1, "beta": 0.2, "seed": 0,
                "stage": "adam", "config": {"observations": 128, "noise_std": 0.0,
                                            "iterations": 50000, "observation_seed": None,
                                            "network_seed": None}}, ckdir / "seed0_adam.pt")
    args = runner.parse_args(["--tuples", "0:0:0", "--noise-std", "0.0", "--init-from",
                              str(ckdir), "--output", "x"])
    assert runner.load_checkpoint(ckdir, 0, 0, 0, args)["alpha"] == 0.1
    with pytest.raises(ValueError, match="seeds"):
        runner.load_checkpoint(ckdir, 0, 1, 0, args)
    noisy = runner.parse_args(["--tuples", "0:0:0", "--noise-std", "0.01", "--init-from",
                               str(ckdir), "--output", "x"])
    with pytest.raises(ValueError, match="noise"):
        runner.load_checkpoint(ckdir, 0, 0, 0, noisy)


def test_noise_summary_pairs_methods_and_compares_with_mle(dde_cpu, tmp_path, monkeypatch):
    from remote import run_deepxde_noise_study as runner
    from scripts.summarize_noise_study import arm, main as summarize_main
    monkeypatch.setattr(runner, "require_remote_cuda", lambda: None)
    out = tmp_path / "n.json"
    runner.main(["--tuples", "42:42:42", "42:42:0", "--noise-std", "0.01", "--iterations", "2",
                 "--lbfgs-maxiter", "2", "--methods", "baseline", "likelihood_weighted",
                 "--output", str(out)])
    summary_path = tmp_path / "s.json"
    summarize_main([str(out), "--json", str(summary_path), "--fd-n", "24"])
    s = json.loads(summary_path.read_text())["n"]
    assert set(s["per_method"]) == {"2:baseline", "2:likelihood_weighted"}
    lw = s["per_method"]["2:likelihood_weighted"]
    assert lw["paired_minus_baseline"]["pairs"] == 2
    assert {r["arm"] for r in s["rows"]} == {"center", "noise_only"}
    assert all(r["mle_relative_error"] < 0.2 for r in s["rows"])
    assert arm((1, 42, 42)) == "network_only" and arm((42, 3, 42)) == "layout_only"


def test_fd_reference_converges_and_recovers_truth():
    from src.fd_reference import DarcyFD
    errors = []
    for n in (16, 32):
        fd = DarcyFD(n)
        p = fd.solve(spatial.ALPHA_TRUE, spatial.BETA_TRUE)
        xx, yy = np.meshgrid(fd.nodes, fd.nodes, indexing="ij")
        errors.append(np.abs(p - np.sin(np.pi * xx) * np.sin(np.pi * yy)).max())
    assert 3.5 < errors[0] / errors[1] < 4.5  # second order
    fd = DarcyFD(64)
    xy, values = spatial.observation_data(42, 64)
    theta, mse, _ = fd.mle(xy, values)
    assert np.allclose(theta, [spatial.ALPHA_TRUE, spatial.BETA_TRUE], atol=2e-3)
    info, cov = fd.fisher(np.array([0.5, -0.3]), xy, 0.01)
    assert np.all(np.linalg.eigvalsh(info) > 0) and cov[0, 1] < 0
