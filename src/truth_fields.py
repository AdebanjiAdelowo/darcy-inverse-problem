"""Synthetic truth log-permeability fields.

Two qualitatively different truths are provided, both as plain
NumPy-callable functions of physical coordinates (independent of any
particular mesh, so the same truth can be evaluated on a fine mesh for
generating synthetic data and, separately, compared against a
reconstruction on a coarser inversion mesh -- see "Avoiding the inverse
crime" in the README).

- `smooth_truth`: a truncated Karhunen-Loeve-like random field, a finite
  sum of low-frequency cosine modes with amplitudes decaying with
  wavenumber, similar in spirit to the synthetic truths used throughout
  the Bayesian-inverse-problems literature (e.g. Stuart, A. M. (2010),
  "Inverse problems: a Bayesian perspective", Acta Numerica, Sec. 5).
- `structured_truth`: two localised, smoothly-blended circular inclusions
  (one high-permeability, one low-permeability) on a uniform background --
  a simple model of channels/lenses, deliberately NOT representable by a
  handful of smooth global modes, to demonstrate that reconstruction
  difficulty depends on the class of unknown field, not just its overall
  amplitude.

Both are deterministic given a random seed, so the whole pipeline (truth
generation -> synthetic data -> reconstruction) is exactly reproducible.
"""
from __future__ import annotations

import numpy as np


def smooth_truth(seed: int = 0, n_modes: int = 4, amplitude: float = 1.0):
    rng = np.random.default_rng(seed)
    ij = [(i, j) for i in range(n_modes) for j in range(n_modes) if not (i == 0 and j == 0)]
    coeffs = {}
    for (i, j) in ij:
        decay = 1.0 / (1.0 + i**2 + j**2) ** 1.25
        coeffs[(i, j)] = rng.standard_normal() * amplitude * decay

    def m(x: np.ndarray) -> np.ndarray:
        # x: shape (2, npoints) or (2,) -- dolfinx interpolation convention
        val = np.zeros(x.shape[1] if x.ndim == 2 else ())
        for (i, j), c in coeffs.items():
            val = val + c * np.cos(i * np.pi * x[0]) * np.cos(j * np.pi * x[1])
        return val

    return m


def structured_truth(
    centers=((0.3, 0.65), (0.7, 0.3)),
    radii=(0.12, 0.15),
    amplitudes=(1.4, -1.2),
    sharpness: float = 40.0,
):
    """Two smoothly-blended circular inclusions on a zero background.

    A tanh-based blend (rather than a hard indicator function) keeps m
    continuous, which is required for it to be meaningfully represented on
    the P1 finite-element space used for the inversion parameter and for
    the adjoint gradient (a discontinuous truth would not be exactly
    representable by P1 regardless of mesh, similar in spirit to the
    Taylor-Hood divergence discussion in the companion `fem-cylinder-flow`
    project); `sharpness` controls how close to a hard indicator the blend
    is.
    """

    def m(x: np.ndarray) -> np.ndarray:
        val = np.zeros(x.shape[1] if x.ndim == 2 else ())
        for (cx, cy), r, amp in zip(centers, radii, amplitudes):
            dist = np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2)
            blend = 0.5 * (1.0 - np.tanh(sharpness * (dist - r)))  # ~1 inside, ~0 outside
            val = val + amp * blend
        return val

    return m
