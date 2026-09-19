"""Taylor-remainder verification of the adjoint gradient.

The adjoint-based gradient in src/adjoint.py is not trusted for anything
downstream until it passes this check.

For a random direction dm and step sizes eps, define the first-order Taylor
remainder

    r(eps) = | J(m + eps*dm) - J(m) - eps * grad(J)(m).dm |

If the gradient is correct, r(eps) = O(eps^2) as eps -> 0 (the linear term
cancels exactly; what remains is the quadratic term of the Taylor
expansion). The convergence RATE (not just decreasing r) is the actual
test: computing log(r(eps_i)/r(eps_{i+1})) / log(eps_i/eps_{i+1}) over
several successively halved eps should approach 2. This is the standard
verification technique for adjoint-based gradients (e.g. as used by
dolfin-adjoint's taylor_test; see Farrell, P. E., Ham, D. A., Funke, S. W.,
Rognes, M. E. (2013), "Automated derivation of the adjoint of
high-level transient finite element programs", SIAM J. Sci. Comput. 35(4)).

Usage: python scripts/verify_gradient.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.forward import build_mesh
from src.observation import grid_sensors, build_observation_operator
from src.adjoint import InverseProblem

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def taylor_test(ip: InverseProblem, m0: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    dm_dir = rng.standard_normal(m0.shape)
    dm_dir /= np.linalg.norm(dm_dir)

    J0, g0, _, _ = ip.objective_and_gradient(m0)
    directional_deriv = float(g0 @ dm_dir)

    eps_list = [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4]
    zeroth_residuals = []
    first_residuals = []
    for eps in eps_list:
        J_pert, _, _, _ = ip.objective_and_gradient(m0 + eps * dm_dir)
        zeroth_residuals.append(abs(J_pert - J0))
        first_residuals.append(abs(J_pert - J0 - eps * directional_deriv))

    return eps_list, zeroth_residuals, first_residuals, J0, directional_deriv


def main() -> None:
    dm = build_mesh(16)
    sensors = grid_sensors(4, 4)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.03)
    rng = np.random.default_rng(42)
    y = 0.5 + 0.05 * rng.standard_normal(H.n_sensors)
    sigma = np.full(H.n_sensors, 0.02)

    lines = []
    for reg_type, alpha in [("h1", 1e-3), ("l2", 1e-2)]:
        ip = InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=alpha, reg_type=reg_type)
        m0 = 0.1 * rng.standard_normal(dm.V_m.dofmap.index_map.size_global)

        eps_list, r0, r1, J0, dd = taylor_test(ip, m0)

        lines.append(f"--- reg_type={reg_type}, alpha={alpha} ---")
        lines.append(f"J(m0) = {J0:.6e}, directional derivative = {dd:.6e}")
        lines.append(f"{'eps':>10} {'|J(m+eps*dm)-J(m)|':>22} {'zeroth order':>14}   "
                      f"{'1st-order remainder':>22} {'observed order':>16}")
        for i, eps in enumerate(eps_list):
            if i == 0:
                order0 = order1 = float("nan")
            else:
                order0 = np.log(r0[i - 1] / r0[i]) / np.log(eps_list[i - 1] / eps_list[i])
                order1 = np.log(r1[i - 1] / r1[i]) / np.log(eps_list[i - 1] / eps_list[i])
            lines.append(f"{eps:10.2e} {r0[i]:22.6e} {order0:14.3f}   {r1[i]:22.6e} {order1:16.3f}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "gradient_verification.txt").write_text(text + "\n")
    print(f"Wrote results/gradient_verification.txt")


if __name__ == "__main__":
    main()
