"""CPU-only scientific checks of the DeepXDE inverse-PINN benchmarks.

No training beyond a single optimiser step. Tests that need DeepXDE/PyTorch
are skipped when those packages are absent (for example in the FEniCSx CI env).
"""
import math
import os

import numpy as np
import pytest

from remote.common import evaluation_grid, relative_l2, run_seed
from src import deepxde_inverse_constant_k as const
from src import deepxde_inverse_spatial_k as spatial


# ---------------------------------------------------------------- numpy only

@pytest.mark.parametrize("module,kwargs", [(const, {"n_observations": 100}),
                                           (spatial, {"observations": 128})])
def test_noise_free_observations_match_original_protocol(module, kwargs):
    n = next(iter(kwargs.values()))
    xy, values = module.observation_data(42, n)
    expected = np.random.default_rng(42).uniform(0.05, 0.95, size=(n, 2))
    assert np.array_equal(xy, expected)
    assert np.array_equal(values, module.exact_pressure(expected))


def test_noise_changes_values_not_locations():
    xy0, v0 = spatial.observation_data(3, 64)
    xy1, v1 = spatial.observation_data(3, 64, noise_std=0.01)
    assert np.array_equal(xy0, xy1)
    assert 0.005 < np.std(v1 - v0) < 0.02
    with pytest.raises(ValueError):
        spatial.observation_data(3, 64, noise_std=-1.0)


@pytest.mark.parametrize("seed", [42, 0, 1, 2, 3])
def test_evaluation_grid_is_disjoint_from_observations(seed):
    grid = evaluation_grid()
    assert grid.shape == (41 * 41, 2)
    xy, _ = spatial.observation_data(seed, 256)
    distances = np.linalg.norm(grid[:, None, :] - xy[None, :, :], axis=2)
    assert distances.min() > 1e-6
    assert np.all((xy > 0.05) & (xy < 0.95))


def test_constant_manufactured_forcing_sign():
    # -k * laplacian(p) with laplacian(sin sin) = -2 pi^2 sin sin.
    xy = np.array([[0.3, 0.4], [0.7, 0.2]])
    h = 1e-4
    lap = sum((const.exact_pressure(xy + d) - 2 * const.exact_pressure(xy)
               + const.exact_pressure(xy - d)) / h**2
              for d in (np.array([h, 0.0]), np.array([0.0, h])))
    forcing = 2 * np.pi**2 * const.K_TRUE * const.exact_pressure(xy)
    assert np.allclose(-const.K_TRUE * lap, forcing, rtol=1e-6)


def test_relative_l2():
    assert relative_l2([1.0, 1.0], [1.0, 1.0]) == 0.0
    assert math.isclose(relative_l2([2.0, 0.0], [1.0, 0.0]), 1.0)


def test_run_seed_records_failures_instead_of_raising():
    ok = run_seed(1, lambda s: {"seed": s, "x": 1.0, "finite_check": ["x"]})
    assert ok["status"] == "ok" and "finite_check" not in ok and ok["wall_seconds"] >= 0
    nan = run_seed(2, lambda s: {"seed": s, "x": float("nan"), "finite_check": ["x"]})
    assert nan["status"] == "non_finite"

    def boom(seed):
        raise FloatingPointError("diverged")
    err = run_seed(3, boom)
    assert err["status"] == "error" and "diverged" in err["error"]


def test_runner_argument_validation():
    from remote.run_deepxde_inverse import parse_args as parse_constant
    from remote.run_deepxde_inverse_spatial import parse_args as parse_spatial
    assert parse_constant([]).seeds == [42, 0, 1, 2, 3]
    assert parse_spatial([]).observations == 128
    for bad in (["--seeds", "1", "1"], ["--noise-std", "-0.1"], ["--iterations", "0"]):
        with pytest.raises(ValueError):
            parse_spatial(bad)
    with pytest.raises(ValueError):
        parse_constant(["--k-initial", "0"])


def test_runners_refuse_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from remote.common import require_remote_cuda
    monkeypatch.delenv("DDE_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="DDE_BACKEND"):
        require_remote_cuda()
    if not torch.cuda.is_available():
        monkeypatch.setenv("DDE_BACKEND", "pytorch")
        with pytest.raises(RuntimeError, match="CUDA"):
            require_remote_cuda()


# ------------------------------------------------------------ DeepXDE on CPU

@pytest.fixture(scope="module")
def dde_cpu():
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    torch = pytest.importorskip("torch")
    dde = pytest.importorskip("deepxde")
    if dde.backend.backend_name != "pytorch":
        pytest.skip("DeepXDE backend is not pytorch")
    previous = torch.get_default_device()
    torch.set_default_device("cpu")  # DeepXDE selects MPS/CUDA at import time
    yield dde, torch
    torch.set_default_device(previous)


def _residual(dde, torch, model, n=64):
    rng = np.random.default_rng(7)
    x = torch.tensor(rng.uniform(0.0, 1.0, (n, 2)), dtype=torch.float32,
                     requires_grad=True)
    p = torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])
    dde.grad.clear()
    r = model.data.pde(x, p)
    dde.grad.clear()
    return float(r.detach().abs().max())


def test_constant_residual_vanishes_only_at_true_k(dde_cpu):
    dde, torch = dde_cpu
    model, log_k = const.build_inverse_model(seed=42)
    with torch.no_grad():
        log_k.fill_(math.log(const.K_TRUE))
    assert _residual(dde, torch, model) < 1e-3
    with torch.no_grad():
        log_k.fill_(math.log(0.8))
    assert _residual(dde, torch, model) > 1.0


def test_spatial_residual_vanishes_only_at_true_parameters(dde_cpu):
    dde, torch = dde_cpu
    model, alpha, beta = spatial.build_model(seed=42)
    with torch.no_grad():
        alpha.fill_(spatial.ALPHA_TRUE)
        beta.fill_(spatial.BETA_TRUE)
    assert _residual(dde, torch, model) < 1e-3
    for a, b in ((0.0, spatial.BETA_TRUE), (spatial.ALPHA_TRUE, 0.0)):
        with torch.no_grad():
            alpha.fill_(a)
            beta.fill_(b)
        assert _residual(dde, torch, model) > 0.1


def _optimised_ids(model):
    return {id(p) for group in model.opt.param_groups for p in group["params"]}


def test_external_variables_are_optimised_and_receive_gradients(dde_cpu):
    from src.deepxde_tracking import parameter_trace
    dde, torch = dde_cpu
    model, alpha, beta = spatial.build_model(seed=0)
    assert {id(alpha), id(beta)} <= _optimised_ids(model)
    trace = parameter_trace({"alpha": alpha, "beta": beta}, period=1)
    model.train(iterations=1, display_every=1, verbose=0, callbacks=[trace])
    assert trace.trace["epoch"] == [0, 1]
    assert trace.trace["alpha"][0] == 0.0 and trace.trace["alpha"][1] != 0.0
    assert trace.trace["beta"][1] != 0.0
    assert np.isfinite(trace.trace["alpha"] + trace.trace["beta"]).all()

    model, log_k = const.build_inverse_model(seed=0)
    assert id(log_k) in _optimised_ids(model)
    before = float(log_k.detach())
    model.train(iterations=1, display_every=1, verbose=0)
    assert float(log_k.detach()) != before
    assert math.exp(float(log_k.detach())) > 0


def test_build_is_reproducible_per_seed(dde_cpu):
    dde, torch = dde_cpu

    def first_layer(seed):
        model, _, _ = spatial.build_model(seed=seed)
        return next(model.net.parameters()).detach().clone()

    assert torch.equal(first_layer(1), first_layer(1))
    assert not torch.equal(first_layer(1), first_layer(2))


def test_initialisation_and_observation_seed_controls(dde_cpu):
    dde, torch = dde_cpu
    _, alpha, beta = spatial.build_model(seed=0, alpha_initial=1.0, beta_initial=-1.0)
    assert float(alpha.detach()) == 1.0 and float(beta.detach()) == -1.0
    m1, _, _ = spatial.build_model(seed=0, observation_seed=9)
    m2, _, _ = spatial.build_model(seed=1, observation_seed=9)
    assert np.array_equal(m1.data.bcs[1].points, m2.data.bcs[1].points)
    _, log_k = const.build_inverse_model(k_initial=2.5)
    assert math.isclose(math.exp(float(log_k.detach())), 2.5, rel_tol=1e-6)
    with pytest.raises(ValueError):
        const.build_inverse_model(k_initial=-1.0)


def test_deepxde_modules_import_without_fenicsx():
    # Remote CUDA images have no dolfinx; the runners must not import the FEM stack.
    import subprocess
    import sys
    code = ("import sys; import remote.run_deepxde_inverse, remote.run_deepxde_inverse_spatial, "
            "remote.smoke_deepxde_inverse; "
            "bad = [m for m in ('dolfinx', 'ufl', 'petsc4py', 'deepxde', 'torch') "
            "if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lbfgs_refinement_logs_without_changing_the_optimisation(dde_cpu):
    from src.deepxde_lbfgs import refine_lbfgs, set_lbfgs_options
    dde, torch = dde_cpu

    def adam_then_plain_lbfgs():
        model, alpha, beta = spatial.build_model(seed=5)
        model.train(iterations=3, display_every=1, verbose=0)
        set_lbfgs_options(maxiter=5)
        model.compile("L-BFGS", external_trainable_variables=[alpha, beta])
        model.train(display_every=1, verbose=0)
        return float(alpha.detach()), float(beta.detach())

    model, alpha, beta = spatial.build_model(seed=5)
    model.train(iterations=3, display_every=1, verbose=0)
    adam = (float(alpha.detach()), float(beta.detach()))
    info = refine_lbfgs(model, {"alpha": alpha, "beta": beta}, maxiter=5, record_every=2)
    logged = (float(alpha.detach()), float(beta.detach()))
    assert logged == adam_then_plain_lbfgs()  # the logging wrapper is read-only
    assert logged != adam
    assert 0 < info["lbfgs_iterations"] <= 5
    assert info["function_evaluations"] == info["logged_evaluations"] > 0
    assert info["finite_history"] and info["options"]["maxiter"] == 5
    assert info["evaluation_history"]["columns"] == ["total_loss", "alpha", "beta"]
    assert info["evaluation_history"]["index"][-1] == info["logged_evaluations"] - 1
    assert info["start_iteration"] == 3


def test_spatial_runner_adam_lbfgs_end_to_end_on_cpu(dde_cpu, tmp_path, monkeypatch):
    import json
    from remote import run_deepxde_inverse_spatial as runner
    monkeypatch.setattr(runner, "require_remote_cuda", lambda: None)  # CPU, 2+3 iterations only
    out = tmp_path / "r.json"
    runner.main(["--iterations", "2", "--seeds", "42", "--lbfgs-maxiter", "3",
                 "--trace-every", "1", "--save-state", str(tmp_path / "state"),
                 "--output", str(out)])
    payload = json.loads(out.read_text())
    record = payload["records"][0]
    assert payload["config"]["optimizer"] == "adam+lbfgs"
    assert record["status"] == "ok" and record["parameter_trace"]["epoch"] == [0, 1, 2]
    assert record["adam_stage"]["alpha_estimated"] == record["parameter_trace"]["alpha"][-1]
    assert record["alpha_estimated"] == record["lbfgs"]["outer_step_trace"]["alpha"][-1]
    assert record["lbfgs"]["outer_step_trace"]["epoch"][-1] == 2 + record["lbfgs"]["lbfgs_iterations"]
    assert len(record["final_loss_train"]) == 3 and record["lbfgs"]["lbfgs_iterations"] <= 3
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == \
        ["seed42_adam.pt", "seed42_adam_lbfgs.pt"]
    from remote.smoke_deepxde_inverse import validate_record
    validate_record("spatial_lbfgs", record)


def test_spatial_runner_default_is_adam_only(dde_cpu, tmp_path, monkeypatch):
    import json
    from remote import run_deepxde_inverse_spatial as runner
    monkeypatch.setattr(runner, "require_remote_cuda", lambda: None)
    out = tmp_path / "r.json"
    runner.main(["--iterations", "2", "--seeds", "42", "--output", str(out)])
    payload = json.loads(out.read_text())
    assert payload["config"]["optimizer"] == "adam" and payload["config"]["lbfgs_maxiter"] == 0
    assert "lbfgs" not in payload["records"][0] and "adam_stage" not in payload["records"][0]


def test_lbfgs_options_respect_maxiter_and_keep_defaults(dde_cpu):
    from deepxde.optimizers.config import LBFGS_options
    from src.deepxde_lbfgs import set_lbfgs_options
    dde, _ = dde_cpu
    before = LBFGS_options["iter_per_step"]
    dde.optimizers.set_LBFGS_options(maxiter=before + 7)
    assert LBFGS_options["iter_per_step"] == before  # upstream 1.15: not recomputed
    assert set_lbfgs_options(maxiter=5)["iter_per_step"] == 5
    default = set_lbfgs_options()
    assert (default["maxiter"], default["iter_per_step"], default["fun_per_step"]) == \
        (15000, 1000, 1250)
