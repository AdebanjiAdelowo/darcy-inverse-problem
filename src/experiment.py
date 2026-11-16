"""Reusable single-reconstruction experiment runner, shared by the
regularisation, noise, and sensor studies (avoids duplicating the
data-generation / inversion-mesh / L-BFGS wiring in every study script)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx.fem import Function

from src.forward import build_mesh, solve_forward
from src.observation import build_observation_operator
from src.data_generation import generate_synthetic_data, SyntheticData
from src.adjoint import InverseProblem
from src.inversion import run_lbfgs, OptimizationHistory


@dataclass
class ReconstructionResult:
    m_rec: np.ndarray
    m_true_on_inv_mesh: np.ndarray
    history: OptimizationHistory
    data: SyntheticData
    inv_dm: object
    ip: InverseProblem
    rel_m_error: float
    misfit_final: float
    reg_final: float


def run_reconstruction(
    truth_fn,
    sensor_locations: np.ndarray,
    truth_mesh_n: int,
    inv_mesh_n: int,
    sensor_width: float,
    relative_noise: float,
    alpha: float,
    reg_type: str = "h1",
    data_seed: int = 0,
    max_iter: int = 150,
    data: SyntheticData | None = None,
) -> ReconstructionResult:
    data_provided = data is not None
    if data is None:
        data = generate_synthetic_data(
            truth_fn, sensor_locations, truth_mesh_n, sensor_width, relative_noise, data_seed
        )

    inv_dm = build_mesh(inv_mesh_n)
    H_inv = build_observation_operator(inv_dm.mesh, inv_dm.V_p, sensor_locations, sensor_width)
    ip = InverseProblem(dm=inv_dm, obs=H_inv, y=data.y_noisy, sigma=data.sigma, alpha=alpha, reg_type=reg_type)

    m0 = np.zeros(inv_dm.V_m.dofmap.index_map.size_global)
    m_rec, history = run_lbfgs(ip, m0, max_iter=max_iter)

    m_true_on_inv = Function(inv_dm.V_m)
    m_true_on_inv.interpolate(truth_fn)
    rel_m_error = float(
        np.linalg.norm(m_rec - m_true_on_inv.x.array) / np.linalg.norm(m_true_on_inv.x.array)
    )

    # diagnostics only: identifiability-aware errors and pressure prediction at
    # the returned iterate (no extra PDE solve; the historical rel_m_error above
    # is unchanged and remains the reported metric of the historical scripts)
    if history.diagnostics is not None:
        from src.identifiability import mass_matrix, observation_prediction_error, reconstruction_errors
        extra = {"truth_available": True, "reg_type": reg_type, "alpha": alpha,
                 "inv_mesh_n": inv_mesh_n, "truth_mesh_n": truth_mesh_n,
                 "n_sensors": int(len(sensor_locations)), "relative_noise": relative_noise,
                 "data_seed": "supplied by caller" if data_provided else data_seed,
                 "initial_guess": "zeros"}
        extra.update(reconstruction_errors(m_rec, m_true_on_inv.x.array, mass_matrix(inv_dm.V_m)))
        if history.predicted_observations is not None:
            extra.update(observation_prediction_error(history.predicted_observations, data.y_clean,
                                                      data.y_noisy, data.sigma))
        history.diagnostics.extra.update(extra)

    return ReconstructionResult(
        m_rec=m_rec, m_true_on_inv_mesh=m_true_on_inv.x.array.copy(), history=history, data=data,
        inv_dm=inv_dm, ip=ip, rel_m_error=rel_m_error,
        misfit_final=history.misfit[-1], reg_final=history.reg[-1],
    )
