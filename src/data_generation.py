"""Synthetic observation generation, with explicit avoidance of the inverse crime.

The "inverse crime" is generating synthetic data with the SAME
discretisation (mesh, function space) that the inverse solver then uses.
Doing so lets the inversion implicitly "cheat" by exactly matching a
discretisation artefact rather than genuine physics, and can hide bugs
that would surface against real (or more realistically simulated) data
(Kaipio, J., Somersalo, E. (2007), "Statistical inverse problems:
discretization, model reduction and inverse crimes", J. Comput. Appl.
Math. 198(2)).

This module generates truth pressure fields, and hence synthetic
observations, on a mesh that is DELIBERATELY FINER than any mesh used
later for inversion (see configs/*.yaml: `truth_mesh_n` is always larger
than `inversion_mesh_n`). The true log-permeability field is evaluated
directly from its analytic/callable definition (src/truth_fields.py) on
whichever mesh is asked for, so the same physical truth can be represented
at different resolutions without being tied to a single discretisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx.fem import Function

from src.forward import build_mesh, solve_forward, DarcyMesh
from src.observation import ObservationOperator, build_observation_operator


@dataclass
class SyntheticData:
    y_clean: np.ndarray
    y_noisy: np.ndarray
    sigma: np.ndarray
    m_true_values_on_truth_mesh: np.ndarray
    truth_dm: DarcyMesh
    p_true: Function


def generate_synthetic_data(
    truth_fn,
    sensor_locations: np.ndarray,
    truth_mesh_n: int,
    sensor_width: float,
    relative_noise: float,
    seed: int,
    nominal_noise_floor: float = 0.01,
) -> SyntheticData:
    """truth_fn: callable m(x) as in src/truth_fields.py.
    relative_noise: noise std ACTUALLY INJECTED into y, as a fraction of
        the clean-signal RMS amplitude (0.0 = noiseless data).
    nominal_noise_floor: the noise standard deviation `sigma` returned
        (used to weight the data-misfit term in the objective, i.e. what
        the inversion ASSUMES about measurement precision) is
        max(relative_noise, nominal_noise_floor) * signal_rms, never less
        than this floor. This matters specifically for relative_noise=0.0
        (the noiseless-data case, used to isolate other error sources in
        some studies): weighting by an essentially-zero sigma would divide
        the misfit by numbers as small as 1e-12 squared, making the
        objective numerically degenerate long before any real ill-posedness
        is visible. A real analyst never assumes literally perfect
        measurement precision either; `nominal_noise_floor` represents a
        modest assumed sensor precision (default 1% of signal RMS) used for
        weighting even when the synthetic data happens to be exact.
    """
    truth_dm = build_mesh(truth_mesh_n)
    m_true = Function(truth_dm.V_m)
    m_true.interpolate(truth_fn)

    p_true = solve_forward(truth_dm, m_true)

    H_truth = build_observation_operator(truth_dm.mesh, truth_dm.V_p, sensor_locations, sensor_width)
    y_clean = H_truth.apply(p_true)

    rng = np.random.default_rng(seed)
    signal_rms = np.sqrt(np.mean(y_clean**2))
    injected_noise_val = relative_noise * signal_rms
    weighting_sigma_val = max(relative_noise, nominal_noise_floor) * signal_rms
    sigma = np.full(y_clean.shape, weighting_sigma_val)
    noise = rng.normal(0.0, injected_noise_val, size=y_clean.shape) if injected_noise_val > 0 else 0.0
    y_noisy = y_clean + noise

    return SyntheticData(
        y_clean=y_clean, y_noisy=y_noisy, sigma=sigma,
        m_true_values_on_truth_mesh=m_true.x.array.copy(),
        truth_dm=truth_dm, p_true=p_true,
    )
