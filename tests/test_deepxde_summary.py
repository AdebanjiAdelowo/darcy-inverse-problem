"""The results summariser computes statistics and pre-registered labels correctly."""
import math

from scripts.summarize_deepxde_inverse import describe, recovery_label, summarize


def test_describe():
    s = describe([1.0, 2.0, 3.0, 10.0])
    assert s["n"] == 4 and s["mean"] == 4.0 and s["median"] == 2.5
    assert s["min"] == 1.0 and s["max"] == 10.0
    assert math.isclose(s["std"], 4.0824829, rel_tol=1e-6)  # sample std (ddof=1)


def test_recovery_labels():
    assert recovery_label([1e-3, 5e-3, 9e-3]) == "recovered"
    assert recovery_label([1e-3, 2e-3, 0.5]) == "partially recovered"
    assert recovery_label([0.02, 0.03, 1e-3]) == "not recovered"
    assert recovery_label([]) == "no successful seeds"


def test_summary_excludes_failed_seeds_and_flags_them():
    payload = {"config": {"benchmark": "constant_k"}, "environment": {"device": "Tesla T4"},
               "records": [
                   {"seed": 42, "status": "ok", "k_estimated": 1.69, "relative_k_error": 0.005,
                    "pressure_relative_l2": 0.01, "wall_seconds": 60.0},
                   {"seed": 0, "status": "non_finite", "k_estimated": float("nan")}]}
    s = summarize(payload)
    assert s["seeds_ok"] == 1 and s["failed"][0]["seed"] == 0
    assert s["statistics"]["relative_k_error"]["n"] == 1
    assert s["recovery"] == "recovered (1 seed(s) failed)"


def test_summary_reports_adam_stage_and_lbfgs_separately():
    def rec(seed, final, adam):
        base = {"alpha_estimated": 0.5, "beta_estimated": -0.3, "pressure_relative_l2": 0.01,
                "permeability_relative_l2": 0.01, "wall_seconds": 1.0}
        return {"seed": seed, "status": "ok", **base, "parameter_relative_error": final,
                "adam_stage": {**base, "parameter_relative_error": adam},
                "lbfgs": {"lbfgs_iterations": 900, "function_evaluations": 1100, "seconds": 9.0,
                          "final_max_abs_gradient": 1e-6, "final_total_loss": 1e-7,
                          "stop_reason": "gtol"}}
    payload = {"config": {"benchmark": "spatial_k", "optimizer": "adam+lbfgs"},
               "environment": {"device": "Tesla T4"},
               "records": [rec(0, 1e-3, 0.3), rec(1, 2e-3, 0.25)]}
    s = summarize(payload)
    assert s["recovery"] == "recovered" and s["adam_stage_recovery"] == "not recovered"
    assert s["adam_stage_statistics"]["parameter_relative_error"]["mean"] == 0.275
    assert s["lbfgs_stop_reasons"] == ["gtol"] and s["optimizer"] == "adam+lbfgs"
