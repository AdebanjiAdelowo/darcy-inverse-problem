"""Verify the Darcy forward solver against a manufactured solution and
perform a mesh-refinement convergence study.

Two variants are run, because they isolate different error sources:

(A) "Exact coefficient": k = exp(m_exact) is used directly as a continuous
    UFL expression in the weak form (never projected onto any finite
    element space). This isolates the accuracy of the PRESSURE
    discretisation alone. For smooth data and a convex domain, standard
    elliptic FEM a priori estimates predict ||p-p_h||_{L2} = O(h^3) and
    ||p-p_h||_{H1} = O(h^2) for P2 pressure elements (the L2 rate is one
    order higher via the Aubin-Nitsche duality argument).

(B) "P1-interpolated coefficient": k = exp(m_h), with m_h the P1 nodal
    interpolant of m_exact -- i.e. exactly how the coefficient will
    actually be represented throughout the rest of this repository, since
    log-permeability is deliberately parameterised on a P1 space for the
    inverse problem (see src/forward.py). Interpolating m onto P1
    introduces an O(h^2) coefficient error that DOMINATES the discretely
    solved problem's accuracy, capping both norms at O(h^2) regardless of
    the (higher-order) pressure space. This was confirmed directly: running
    with a spuriously high quadrature degree left the observed L2 rate
    unchanged at ~2.0, ruling out quadrature error as the cause, while
    switching to the exact (non-interpolated) coefficient in variant (A)
    immediately restored the textbook O(h^3) rate.

Both are reported so the forward solver's behaviour in its ACTUAL
configuration (variant B) is verified and explained, rather than only
showing the more flattering, but less representative, variant (A) rate.

Usage: python scripts/verify_forward.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.forward import build_mesh, solve_forward
from src.manufactured import exact_fields

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def _boundary_bcs(dm, p_exact):
    msh = dm.mesh
    p_bc = Function(dm.V_p)
    p_bc.interpolate(fem.Expression(p_exact, dm.V_p.element.interpolation_points))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dmesh.exterior_facet_indices(msh.topology)
    bdofs = locate_dofs_topological(dm.V_p, tdim - 1, boundary_facets)
    return [dirichletbc(p_bc, bdofs)]


def run_case_interpolated_k(n: int):
    """Variant (B): the realistic configuration used throughout this repo."""
    dm = build_mesh(n)
    x = ufl.SpatialCoordinate(dm.mesh)
    p_exact, m_exact, f = exact_fields(x)
    m_h = Function(dm.V_m)
    m_h.interpolate(fem.Expression(m_exact, dm.V_m.element.interpolation_points))
    bcs = _boundary_bcs(dm, p_exact)
    p_h = solve_forward(dm, m_h, f=f, bcs=bcs)
    e_l2 = np.sqrt(fem.assemble_scalar(fem.form((p_h - p_exact) ** 2 * ufl.dx)))
    e_h1 = np.sqrt(
        fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(p_h - p_exact), ufl.grad(p_h - p_exact)) * ufl.dx))
    )
    return 1.0 / n, e_l2, e_h1


def run_case_exact_k(n: int):
    """Variant (A): isolates the pressure discretisation's own accuracy."""
    dm = build_mesh(n)
    msh = dm.mesh
    x = ufl.SpatialCoordinate(msh)
    p_exact, m_exact, f = exact_fields(x)
    k_exact = ufl.exp(m_exact)
    bcs = _boundary_bcs(dm, p_exact)

    p = ufl.TrialFunction(dm.V_p)
    v = ufl.TestFunction(dm.V_p)
    a = k_exact * ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=f"exactk_verify_{n}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    p_h = problem.solve()
    e_l2 = np.sqrt(fem.assemble_scalar(fem.form((p_h - p_exact) ** 2 * ufl.dx)))
    e_h1 = np.sqrt(
        fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(p_h - p_exact), ufl.grad(p_h - p_exact)) * ufl.dx))
    )
    return 1.0 / n, e_l2, e_h1


def _sweep(run_fn, Ns):
    hs, e_l2s, e_h1s = [], [], []
    lines = [f"{'N':>4} {'h':>8} {'L2 error':>12} {'order':>7} {'H1 error':>12} {'order':>7}"]
    for N in Ns:
        h, e_l2, e_h1 = run_fn(N)
        hs.append(h); e_l2s.append(e_l2); e_h1s.append(e_h1)
    for i, N in enumerate(Ns):
        if i == 0:
            o_l2 = o_h1 = float("nan")
        else:
            o_l2 = np.log(e_l2s[i - 1] / e_l2s[i]) / np.log(hs[i - 1] / hs[i])
            o_h1 = np.log(e_h1s[i - 1] / e_h1s[i]) / np.log(hs[i - 1] / hs[i])
        line = f"{N:4d} {hs[i]:8.4f} {e_l2s[i]:12.4e} {o_l2:7.3f} {e_h1s[i]:12.4e} {o_h1:7.3f}"
        print(line)
        lines.append(line)
    return hs, e_l2s, e_h1s, lines


def main() -> None:
    Ns = [4, 8, 16, 32, 64]

    print("=== Variant (A): exact (non-interpolated) coefficient ===")
    hs_a, e_l2_a, e_h1_a, lines_a = _sweep(run_case_exact_k, Ns)

    print("\n=== Variant (B): P1-interpolated coefficient (realistic configuration) ===")
    hs_b, e_l2_b, e_h1_b, lines_b = _sweep(run_case_interpolated_k, Ns)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = (
        ["Variant (A): exact (non-interpolated) coefficient -- isolates pressure-space accuracy"]
        + lines_a
        + ["", "Variant (B): P1-interpolated coefficient -- realistic configuration used elsewhere "
               "in this repository"]
        + lines_b
    )
    (RESULTS_DIR / "forward_verification.txt").write_text("\n".join(out) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.loglog(hs_a, e_l2_a, "o-", label=r"$\|p-p_h\|_{L^2}$ (expect $O(h^3)$)")
    ax.loglog(hs_a, e_h1_a, "s-", label=r"$\|p-p_h\|_{H^1}$ (expect $O(h^2)$)")
    ax.loglog(hs_a, e_l2_a[0] * (np.array(hs_a) / hs_a[0]) ** 3, "k--", alpha=0.5, label=r"$O(h^3)$")
    ax.loglog(hs_a, e_h1_a[0] * (np.array(hs_a) / hs_a[0]) ** 2, "k:", alpha=0.5, label=r"$O(h^2)$")
    ax.set_xlabel("mesh size $h$"); ax.set_ylabel("error")
    ax.set_title("(A) Exact coefficient:\npressure-space accuracy")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.loglog(hs_b, e_l2_b, "o-", label=r"$\|p-p_h\|_{L^2}$")
    ax.loglog(hs_b, e_h1_b, "s-", label=r"$\|p-p_h\|_{H^1}$")
    ax.loglog(hs_b, e_h1_b[0] * (np.array(hs_b) / hs_b[0]) ** 2, "k:", alpha=0.5, label=r"$O(h^2)$")
    ax.set_xlabel("mesh size $h$"); ax.set_ylabel("error")
    ax.set_title("(B) P1-interpolated coefficient\n(realistic config.): coefficient-error-limited")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Darcy forward solver: manufactured-solution convergence")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "forward_convergence.png", dpi=150)
    print(f"\nWrote results/forward_verification.txt and figures/forward_convergence.png")


if __name__ == "__main__":
    main()
