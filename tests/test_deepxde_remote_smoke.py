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
