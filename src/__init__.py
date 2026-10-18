# Package-level names are resolved lazily (PEP 562) so that FEniCSx-free
# submodules, such as the DeepXDE inverse-PINN benchmarks run on remote CUDA
# images without dolfinx, can be imported without importing the FEM stack.
from importlib import import_module

_EXPORTS = {
    "build_mesh": "forward", "solve_forward": "forward", "pressure_bcs": "forward",
    "DarcyMesh": "forward",
    "exact_fields": "manufactured",
    "smooth_truth": "truth_fields", "structured_truth": "truth_fields",
    "grid_sensors": "observation", "random_sensors": "observation",
    "build_observation_operator": "observation", "ObservationOperator": "observation",
    "generate_synthetic_data": "data_generation", "SyntheticData": "data_generation",
    "l2_prior": "regularization", "h1_seminorm": "regularization",
    "InverseProblem": "adjoint", "homogeneous_bcs": "adjoint",
    "run_lbfgs": "inversion", "OptimizationHistory": "inversion",
    "run_reconstruction": "experiment", "ReconstructionResult": "experiment",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
