"""Sensor placement and the pressure observation operator H.

Sensors measure a small local average of the pressure field around each
sensor location (a narrow Gaussian-weighted average), not a literal Dirac
point evaluation -- both a more robust finite-element construction (the
weak form of a point evaluation is a distribution, not an L2 function; a
narrow but finite-width average is a genuine bounded linear functional on
H^1) and, arguably, a more physically honest model of a real pressure
sensor, which also integrates over some small finite volume. For a bump
width small relative to the domain but a few mesh cells wide, this closely
approximates a point measurement while remaining exactly differentiable
for the adjoint method below.

For a pressure field p, H is the linear map

    H(p)_s = integral_Omega p(x) * psi_s(x) dx ,  s = 1, ..., n_sensors

with psi_s a Gaussian bump centred at the s-th sensor location, normalised
so integral psi_s dx = 1 (implemented by assembling the linear functional
directly against the FE test-function basis and dividing by its sum --
exact for a partition-of-unity nodal basis, see `sensor_weight_vectors`).
Because H is linear and represented by the same finite-element test-space
weights used to assemble the pressure equation itself, its adjoint H^T
(needed for the gradient in src/adjoint.py) is simply a weighted sum of the
same weight vectors -- no separate implementation is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem


def grid_sensors(nx: int, ny: int, margin: float = 0.1) -> np.ndarray:
    xs = np.linspace(margin, 1.0 - margin, nx)
    ys = np.linspace(margin, 1.0 - margin, ny)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


def random_sensors(n: int, seed: int, margin: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return margin + (1.0 - 2 * margin) * rng.random((n, 2))


@dataclass
class ObservationOperator:
    sensor_locations: np.ndarray  # (n_sensors, 2)
    weight_vectors: np.ndarray  # (n_sensors, n_dofs)
    V_p: object

    @property
    def n_sensors(self) -> int:
        return self.sensor_locations.shape[0]

    def apply(self, p: fem.Function) -> np.ndarray:
        return self.weight_vectors @ p.x.array

    def adjoint_rhs_vector(self, residual: np.ndarray) -> np.ndarray:
        """Returns sum_s residual[s] * weight_vectors[s], the dof vector to
        use as the adjoint equation's RHS load."""
        return residual @ self.weight_vectors


def build_observation_operator(mesh, V_p, sensor_locations: np.ndarray, width: float) -> ObservationOperator:
    x = ufl.SpatialCoordinate(mesh)
    v = ufl.TestFunction(V_p)
    n_dofs = V_p.dofmap.index_map.size_global * V_p.dofmap.index_map_bs
    weights = np.zeros((len(sensor_locations), n_dofs))
    for s, (sx, sy) in enumerate(sensor_locations):
        bump = ufl.exp(-((x[0] - sx) ** 2 + (x[1] - sy) ** 2) / (2.0 * width**2))
        w = fem.assemble_vector(fem.form(bump * v * ufl.dx))
        w_arr = w.array.copy()
        total = w_arr.sum()
        weights[s] = w_arr / total
    return ObservationOperator(sensor_locations=sensor_locations, weight_vectors=weights, V_p=V_p)
