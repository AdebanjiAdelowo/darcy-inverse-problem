"""Conditioning of the deterministic inversion Hessian in nodal (Euclidean) and
preconditioned coordinates, at the headline smooth and structured minimisers
(N=32, 36 sensors, alpha=30, data seed 7).

The Gauss-Newton Hessian H = alpha K + J^T Gamma^{-1} J is formed explicitly
(J from one adjoint solve per sensor); second-order misfit terms are omitted.
Spectra are reported on the complement of the constant vector, which is an exact
null direction of both terms.

    python scripts/m3_conditioning.py
Output: results/m3/conditioning.json and results/m3/figures/hessian_spectra.png
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import scipy.linalg as sla

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def spectrum_on_mean_free(H, B=None):
    """Eigenvalues of H restricted to the Euclidean complement of the constant vector
    (B=None) or of the generalised pencil with an orthonormal basis Q of 1-perp."""
    n = H.shape[0]
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, :n - 1]]))
    Q = Q[:, 1:]
    return np.linalg.eigvalsh(Q.T @ H @ Q)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.adjoint import InverseProblem
    from src.data_generation import generate_synthetic_data
    from src.forward import build_mesh
    from src.observation import build_observation_operator, grid_sensors
    from src.preconditioned_inversion import (observation_jacobian, regularization_preconditioner,
                                              run_lbfgs_preconditioned)
    from src.truth_fields import smooth_truth, structured_truth

    alpha, S = 30.0, grid_sensors(6, 6)
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for col, (name, tf) in enumerate((("smooth", smooth_truth(seed=1)), ("structured", structured_truth()))):
        data = generate_synthetic_data(tf, S, 64, 0.03, 0.02, 7)
        dm = build_mesh(32)
        H_obs = build_observation_operator(dm.mesh, dm.V_p, S, 0.03)
        ip = InverseProblem(dm=dm, obs=H_obs, y=data.y_noisy, sigma=data.sigma, alpha=alpha)
        L, K, M = regularization_preconditioner(dm.V_m, alpha, alpha)
        n = K.shape[0]
        m_opt, *_ = run_lbfgs_preconditioned(ip, np.zeros(n), L, max_iter=200)
        Jm = observation_jacobian(dm, H_obs, m_opt)
        W = Jm / data.sigma[:, None]
        H = alpha * K + W.T @ W
        ev_e = spectrum_on_mean_free(H)
        Linv = sla.solve_triangular(L, np.eye(n), lower=True)
        Hp = Linv @ H @ Linv.T
        ev_p = np.linalg.eigvalsh(0.5 * (Hp + Hp.T))
        ev_p_mf = np.sort(ev_p)[1:]  # drop the one eigenvalue near 0 (constant direction, P-gauge)
        ev_reg = spectrum_on_mean_free(alpha * K)
        out[name] = {
            "n_dofs": n, "n_sensors": len(S),
            "euclidean": {"min": float(ev_e.min()), "max": float(ev_e.max()),
                          "condition_number": float(ev_e.max() / ev_e.min())},
            "regularization_only_euclidean_condition_number": float(ev_reg.max() / ev_reg.min()),
            "preconditioned": {"min": float(ev_p_mf.min()), "max": float(ev_p_mf.max()),
                               "condition_number": float(ev_p_mf.max() / ev_p_mf.min()),
                               "n_eigenvalues_outside_0.5_2": int(np.sum((ev_p_mf < 0.5) | (ev_p_mf > 2.0))),
                               "smallest_eigenvalue_constant_direction": float(np.sort(ev_p)[0])},
            "gauss_newton_rank": int(np.linalg.matrix_rank(W)),
        }
        ax[col].semilogy(np.sort(ev_e)[::-1] / ev_e.max(), label=f"nodal (cond {out[name]['euclidean']['condition_number']:.1e})")
        ax[col].semilogy(np.sort(ev_p_mf)[::-1] / ev_p_mf.max(),
                         label=f"preconditioned (cond {out[name]['preconditioned']['condition_number']:.1e})")
        ax[col].set_title(f"{name} truth: Gauss-Newton Hessian spectrum (mean-free)", fontsize=9)
        ax[col].set_xlabel("index")
        ax[col].set_ylabel("eigenvalue / max")
        ax[col].legend(fontsize=8)
        print(name, json.dumps(out[name], indent=1))
    fig.tight_layout()
    (ROOT / "results" / "m3" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "m3" / "figures" / "hessian_spectra.png", dpi=110)
    (ROOT / "results" / "m3" / "conditioning.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
