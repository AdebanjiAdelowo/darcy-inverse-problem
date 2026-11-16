"""Identifiability-aware diagnostics for the zero-source Darcy inverse problem.

## The constant-scaling invariance

For the physical problem used throughout this repository,

    -div(k grad p) = 0,  p = 1 (x=0), p = 0 (x=1),  k dp/dn = 0 (y=0, y=1),

replacing k by c*k for any constant c > 0 leaves p unchanged: the operator is
multiplied by c, the right-hand side is zero, and the Dirichlet data do not
depend on k. In log form, m -> m + log(c) leaves p(m), and therefore every
pressure observation, unchanged. The same holds exactly for the discrete
problem, because the stiffness matrix is assembled from exp(m) pointwise.

Consequently the quantities involved separate as follows:

- absolute log-permeability m(x): NOT identifiable from pressure data;
- mean log-permeability  mean(m) = (1/|Omega|) integral m dx: NOT identifiable
  (it is the only direction lost: shifting m by a constant changes only this);
- mean-free log-permeability m - mean(m): the identifiable part of m;
- spatial permeability contrast k(x)/k_g with k_g = exp(mean(m)) (geometric
  mean): identifiable, equal to exp(m - mean(m));
- pressure prediction p(m) and H p(m): identifiable functions of the
  mean-free part only.

Under H1-seminorm regularisation the constant direction is also unpenalised,
so the deterministic objective has an exactly flat direction; with a Gaussian
prior the constant mode is constrained by the prior alone.

## The KL prior's constant mode

With covariance C = A^{-2}, A = -gamma*Laplacian + delta*I and Neumann
boundary conditions, the constant function is an eigenfunction of A with
eigenvalue delta, which is the smallest. The first KL mode is therefore the
L2-normalised constant phi_0 = 1/sqrt(|Omega|) with covariance eigenvalue
lambda_0 = delta^{-2}. Because the likelihood does not depend on xi_0 and the
whitened prior factorises, the exact posterior has xi_0 | y ~ N(0, 1),
independent of the other coefficients. Hence for every x

    Var[m(x) | y] >= lambda_0 * phi_0(x)^2 = delta^{-2} / |Omega|,

and integral Var[m(x)|y] dx >= lambda_0. Any posterior summary below these
bounds is a sampling artefact. This module only diagnoses; it changes nothing
in the inverse problem, the prior or the sampler.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix


def mass_matrix(V) -> sp.csr_matrix:
    """Consistent P1 mass matrix of V as a SciPy CSR matrix."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    M = assemble_matrix(fem.form(ufl.inner(u, v) * ufl.dx))
    M.assemble()
    indptr, indices, data = M.getValuesCSR()
    n = V.dofmap.index_map.size_global
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def l2_mean(m: np.ndarray, M: sp.spmatrix) -> float:
    """(1/|Omega|) integral m dx for a nodal field m."""
    ones = np.ones(M.shape[0])
    return float(ones @ (M @ m) / (ones @ (M @ ones)))


def l2_norm(m: np.ndarray, M: sp.spmatrix) -> float:
    return float(np.sqrt(max(m @ (M @ m), 0.0)))


def reconstruction_errors(m_est: np.ndarray, m_true: np.ndarray, M: sp.spmatrix | None = None) -> dict:
    """Absolute and constant-invariant error measures for a log-permeability
    estimate. Keys:

    - rel_error_abs_nodal: ||m_est - m_true||_2 / ||m_true||_2 over nodal values
      (the historical metric used in all stored results);
    - rel_error_meanfree_nodal: the same after removing each field's nodal mean;
    - with M: rel_error_abs_L2, rel_error_meanfree_L2 (mass-weighted mean
      removed; equals min_c ||m_est + c - m_true||_L2 / ||m_true - mean||_L2,
      the distance in the quotient space that pressure data can identify),
      mean_true_L2, mean_est_L2, mean_offset_L2, rel_error_contrast_L2 for the
      contrast field exp(m - mean(m)).
    """
    m_est = np.asarray(m_est, dtype=float)
    m_true = np.asarray(m_true, dtype=float)
    c = lambda a: a - a.mean()
    out = {
        "rel_error_abs_nodal": float(np.linalg.norm(m_est - m_true) / np.linalg.norm(m_true)),
        "rel_error_meanfree_nodal": float(np.linalg.norm(c(m_est) - c(m_true)) / np.linalg.norm(c(m_true))),
        "mean_true_nodal": float(m_true.mean()),
        "mean_est_nodal": float(m_est.mean()),
    }
    if M is not None:
        mt, me = l2_mean(m_true, M), l2_mean(m_est, M)
        dt, de = m_true - mt, m_est - me
        kt, ke = np.exp(dt), np.exp(de)
        out.update({
            "rel_error_abs_L2": l2_norm(m_est - m_true, M) / l2_norm(m_true, M),
            "rel_error_meanfree_L2": l2_norm(de - dt, M) / l2_norm(dt, M),
            "mean_true_L2": mt,
            "mean_est_L2": me,
            "mean_offset_L2": me - mt,
            "rel_error_contrast_L2": l2_norm(ke - kt, M) / l2_norm(kt, M),
        })
    return out


def observation_prediction_error(pred: np.ndarray, y_clean: np.ndarray, y_noisy: np.ndarray,
                                 sigma: np.ndarray) -> dict:
    """Pressure-prediction diagnostics at the sensors (an identifiable quantity)."""
    return {
        "rel_error_pred_vs_clean": float(np.linalg.norm(pred - y_clean) / np.linalg.norm(y_clean)),
        "rms_residual_vs_noisy_over_sigma": float(np.sqrt(np.mean(((pred - y_noisy) / sigma) ** 2))),
    }


# ---------------------------------------------------------------- prior side

def constant_mode_index(prior, rtol: float = 1e-8) -> int | None:
    """Index of the KL mode whose eigenvector is constant, or None."""
    V = prior.eigenvectors
    for j in range(V.shape[1]):
        v = V[:, j]
        scale = np.max(np.abs(v))
        if scale > 0 and np.ptp(v) <= rtol * scale:
            return j
    return None


def constant_mode_variance_floor(prior, M: sp.spmatrix) -> dict:
    """Exact posterior-variance floor contributed by the unidentifiable constant
    mode (valid because the likelihood does not depend on its coefficient).

    Returns pointwise floor lambda_c*phi_c(x)^2 (nodal), its integral, the
    domain area, and the theoretical values delta^-2 and delta^-2/|Omega|."""
    j = constant_mode_index(prior)
    if j is None:
        raise ValueError("prior has no constant KL mode")
    lam = float(prior.eigenvalues[j])
    phi = prior.eigenvectors[:, j]
    area = float(np.ones(M.shape[0]) @ (M @ np.ones(M.shape[0])))
    return {
        "mode_index": j,
        "lambda": lam,
        "lambda_theory": 1.0 / prior.delta**2,
        "phi_value": float(phi.mean()),
        "pointwise_floor": lam * phi**2,
        "pointwise_floor_theory": 1.0 / (prior.delta**2 * area),
        "integrated_floor": lam * float(phi @ (M @ phi)),
        "area": area,
    }


def check_posterior_variance_floor(m_post_std: np.ndarray, prior, M: sp.spmatrix) -> dict:
    """Compare a stored/estimated posterior standard deviation field with the
    exact constant-mode floor. A violation means the posterior summary cannot
    be a correct estimate of the exact (discretised) posterior variance."""
    floor = constant_mode_variance_floor(prior, M)
    var = np.asarray(m_post_std, dtype=float) ** 2
    ratio = var / floor["pointwise_floor"]
    integrated = float(var @ (M @ np.ones_like(var)))
    return {
        "pointwise_variance_min": float(var.min()),
        "pointwise_variance_max": float(var.max()),
        "pointwise_floor": float(np.median(floor["pointwise_floor"])),
        "min_ratio_to_floor": float(ratio.min()),
        "max_ratio_to_floor": float(ratio.max()),
        "fraction_of_nodes_below_floor": float(np.mean(ratio < 1.0)),
        "integrated_variance": integrated,
        "integrated_floor": floor["integrated_floor"],
        "violates_floor": bool(np.any(ratio < 1.0) or integrated < floor["integrated_floor"]),
    }
