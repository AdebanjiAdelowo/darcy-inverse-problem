from .forward import build_mesh, solve_forward, pressure_bcs, DarcyMesh
from .manufactured import exact_fields
from .truth_fields import smooth_truth, structured_truth
from .observation import grid_sensors, random_sensors, build_observation_operator, ObservationOperator
from .data_generation import generate_synthetic_data, SyntheticData
from .regularization import l2_prior, h1_seminorm
from .adjoint import InverseProblem, homogeneous_bcs
from .inversion import run_lbfgs, OptimizationHistory
from .experiment import run_reconstruction, ReconstructionResult

__all__ = [
    "build_mesh", "solve_forward", "pressure_bcs", "DarcyMesh",
    "exact_fields",
    "smooth_truth", "structured_truth",
    "grid_sensors", "random_sensors", "build_observation_operator", "ObservationOperator",
    "generate_synthetic_data", "SyntheticData",
    "l2_prior", "h1_seminorm",
    "InverseProblem", "homogeneous_bcs",
    "run_lbfgs", "OptimizationHistory",
    "run_reconstruction", "ReconstructionResult",
]
