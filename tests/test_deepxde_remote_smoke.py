"""CPU-only validation of the remote inverse smoke orchestration."""
import pytest
from remote.smoke_deepxde_inverse import command, validate_record


def test_commands():
    assert "remote.run_deepxde_inverse" in command("constant", 2, "out.json")
    assert "remote.run_deepxde_inverse_spatial" in command("spatial", 2, "out.json")
    with pytest.raises(ValueError):
        command("spatial", 0, "out.json")


def test_constant_validation():
    record={"seed":42,"k_estimated":1.6,"relative_k_error":0.1,
            "wall_seconds":1.0}
    assert validate_record("constant",record)==record
    with pytest.raises(ValueError):
        validate_record("constant",{**record,"k_estimated":float("nan")})


def test_spatial_validation():
    record={"seed":42,"alpha_estimated":0.4,"beta_estimated":-0.2,
            "pressure_relative_l2":0.1,"permeability_relative_l2":0.2,
            "wall_seconds":1.0}
    assert validate_record("spatial",record)==record
    with pytest.raises(ValueError):
        validate_record("spatial",{**record,"seed":1})
