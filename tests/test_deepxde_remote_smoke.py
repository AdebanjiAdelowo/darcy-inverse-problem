"""CPU-only validation of the remote inverse smoke orchestration."""
import pytest
from remote.smoke_deepxde_inverse import command, validate_record


def test_commands():
    assert "remote.run_deepxde_inverse" in command("constant", 2, "out.json")
    assert "remote.run_deepxde_inverse_spatial" in command("spatial", 2, "out.json")
    with pytest.raises(ValueError):
        command("spatial", 0, "out.json")
    lbfgs = command("spatial_lbfgs", 2, "out.json")
    assert "remote.run_deepxde_inverse_spatial" in lbfgs
    assert lbfgs[lbfgs.index("--lbfgs-maxiter") + 1] == "20"
    assert "--lbfgs-maxiter" not in command("spatial", 2, "out.json")


def test_constant_validation():
    record={"seed":42,"status":"ok","k_estimated":1.6,"relative_k_error":0.1,
            "pressure_relative_l2":0.05,"wall_seconds":1.0}
    assert validate_record("constant",record)==record
    with pytest.raises(ValueError):
        validate_record("constant",{**record,"k_estimated":float("nan")})
    with pytest.raises(ValueError):
        validate_record("constant",{**record,"k_estimated":-1.0})
    with pytest.raises(ValueError):
        validate_record("constant",{**record,"status":"non_finite"})


def test_spatial_validation():
    record={"seed":42,"status":"ok","alpha_estimated":0.4,"beta_estimated":-0.2,
            "pressure_relative_l2":0.1,"permeability_relative_l2":0.2,
            "wall_seconds":1.0}
    assert validate_record("spatial",record)==record
    with pytest.raises(ValueError):
        validate_record("spatial",{**record,"seed":1})
    with pytest.raises(ValueError, match="L-BFGS"):
        validate_record("spatial_lbfgs", record)  # no refinement recorded


def test_kaggle_launcher_commands_parse():
    from remote import kaggle_deepxde_inverse as launcher
    from remote.run_deepxde_inverse import parse_args as parse_constant
    from remote.run_deepxde_inverse_spatial import parse_args as parse_spatial
    parsers = {"remote.run_deepxde_inverse": parse_constant,
               "remote.run_deepxde_inverse_spatial": parse_spatial}
    assert launcher.REF == "<40-character commit SHA>"  # committed copy is never pinned
    for name, (module, *args) in {**launcher.BASELINE, **launcher.SENSITIVITY}.items():
        parsed = parsers[module]([*args, "--output", f"{name}.json"])
        assert parsed.iterations == 10000 and parsed.seeds == [42, 0, 1, 2, 3]
    for name, (module, *args) in launcher.CONVERGENCE.items():
        parsed = parsers[module]([*args, "--output", f"{name}.json"])
        base = parsers[module]([*launcher.BASELINE[name.replace("_convergence50k", "_baseline")][1:],
                                "--output", "x.json"])
        assert parsed.iterations == 50000 and parsed.trace_every == 500
        assert {**vars(parsed), "iterations": 0, "trace_every": 0, "output": ""} == \
            {**vars(base), "iterations": 0, "trace_every": 0, "output": ""}


def test_kaggle_lbfgs_stage_is_the_convergence_run_plus_refinement():
    from remote import kaggle_deepxde_inverse as launcher
    from remote.run_deepxde_inverse_spatial import parse_args
    (module, *args), = launcher.LBFGS.values()
    lbfgs = parse_args([*args, "--output", "x.json"])
    adam = parse_args([*launcher.CONVERGENCE["spatial_convergence50k"][1:], "--output", "x.json"])
    assert module == "remote.run_deepxde_inverse_spatial"
    assert lbfgs.lbfgs_maxiter == 15000 and adam.lbfgs_maxiter == 0
    ignore = {"lbfgs_maxiter": 0, "lbfgs_record_every": 0, "save_state": None, "output": ""}
    assert {**vars(lbfgs), **ignore} == {**vars(adam), **ignore}


def test_sensitivity_rows_change_one_factor_from_the_validated_configs():
    from remote import kaggle_deepxde_inverse as launcher
    from remote.run_deepxde_inverse import parse_args as parse_constant
    from remote.run_deepxde_inverse_spatial import parse_args as parse_spatial
    lbfgs = vars(parse_spatial([*launcher.LBFGS["spatial_adam50k_lbfgs15k"][1:], "--output", "x"]))
    lbfgs["save_state"] = None
    const = vars(parse_constant([*launcher.CONVERGENCE["constant_convergence50k"][1:],
                                 "--output", "x"]))
    expected = {
        "spatial_lbfgs_obs16": {"observations": 16}, "spatial_lbfgs_obs32": {"observations": 32},
        "spatial_lbfgs_obs64": {"observations": 64}, "spatial_lbfgs_obs256": {"observations": 256},
        "spatial_lbfgs_init_only": {"observation_seed": 42},
        "spatial_lbfgs_layout_only": {"network_seed": 42},
        "spatial_lbfgs_noise0p01": {"noise_std": 0.01}, "spatial_lbfgs_noise0p05": {"noise_std": 0.05},
        "spatial_lbfgs_init_far": {"alpha_initial": -1.0, "beta_initial": 1.0},
        "constant_adam50k_noise0p01": {"noise_std": 0.01},
        "constant_adam50k_init_high": {"k_initial": 5.0},
    }
    rows = {**launcher.SENSITIVITY_A, **launcher.SENSITIVITY_B}
    assert set(rows) == set(expected)
    for name, (module, *args) in rows.items():
        assert args.count("--observations") <= 1
        spatial = module.endswith("_spatial")
        parsed = vars((parse_spatial if spatial else parse_constant)([*args, "--output", "x"]))
        assert {k: v for k, v in parsed.items() if v != (lbfgs if spatial else const)[k]} == \
            expected[name], name
