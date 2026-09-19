"""Tikhonov-type regularisers for the log-permeability parameter m.

Both are standard, convex, quadratic regularisers (Engl, Hanke & Neubauer,
1996, *Regularization of Inverse Problems*, Ch. 5), returned as UFL forms
so their contribution to the objective and its gradient can be assembled
and differentiated exactly the same way as the data-misfit term.

- `l2_prior`: R(m) = ||m - m0||_{L2}^2 penalises deviation from a reference
  (prior-mean) field m0, pulling the reconstruction toward m0 wherever the
  data are uninformative. It does NOT penalise roughness: a highly
  oscillatory m with small L2 norm is not penalised more than a smooth one
  with the same norm.
- `h1_seminorm`: R(m) = ||grad(m)||_{L2}^2 penalises spatial roughness
  (large gradients) directly, without penalising the mean level of m at
  all (a constant shift of m is in its null space). This is the more
  common choice for spatially distributed coefficient-inversion problems,
  since it encodes the modelling assumption that permeability varies
  smoothly except where genuine structure exists, and is used as the
  default in this repository's experiments.
"""
from __future__ import annotations

import ufl
from dolfinx import fem


def l2_prior(m, m0=None):
    if m0 is None:
        return ufl.inner(m, m) * ufl.dx
    return ufl.inner(m - m0, m - m0) * ufl.dx


def h1_seminorm(m):
    return ufl.inner(ufl.grad(m), ufl.grad(m)) * ufl.dx
