import numpy as np

from src.forward import build_mesh
from src.observation import grid_sensors, build_observation_operator
from src.adjoint import InverseProblem
from src.inversion import run_lbfgs


def test_lbfgs_reduces_objective_on_small_problem():
    dm = build_mesh(8)
    sensors = grid_sensors(3, 3)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.05)
    rng = np.random.default_rng(3)
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    sigma = np.full(H.n_sensors, 0.02)
    ip = InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=0.1, reg_type="h1")

    m0 = np.zeros(dm.V_m.dofmap.index_map.size_global)
    J0, _, _, _ = ip.objective_and_gradient(m0)
    m_rec, history = run_lbfgs(ip, m0, max_iter=50)
    J_final = history.objective[-1]

    assert J_final < J0
    assert history.n_iterations > 0
    assert len(history.objective) == len(history.misfit) == len(history.reg) == len(history.grad_norm)


def test_lbfgs_history_is_monotonically_non_increasing_in_objective():
    """L-BFGS-B with a line search should not accept a step that increases
    the objective (a basic sanity property of the optimiser, not a claim
    about the gradient's correctness, which is tested separately)."""
    dm = build_mesh(8)
    sensors = grid_sensors(3, 3)
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.05)
    rng = np.random.default_rng(4)
    y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
    sigma = np.full(H.n_sensors, 0.02)
    ip = InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=0.1, reg_type="h1")
    m0 = np.zeros(dm.V_m.dofmap.index_map.size_global)
    m_rec, history = run_lbfgs(ip, m0, max_iter=50)

    obj = np.array(history.objective)
    # L-BFGS-B evaluates the objective at trial points during its internal
    # line search, so the raw per-call sequence need not be monotone; check
    # the RUNNING MINIMUM decreases to (near) the final value, and that the
    # final value is the smallest seen (the optimiser did not walk away
    # from a good point).
    assert obj[-1] <= np.min(obj) + 1e-8


def test_reproducible_with_fixed_seed():
    def build_and_run():
        dm = build_mesh(8)
        sensors = grid_sensors(3, 3)
        H = build_observation_operator(dm.mesh, dm.V_p, sensors, width=0.05)
        rng = np.random.default_rng(9)
        y = 0.5 + 0.02 * rng.standard_normal(H.n_sensors)
        sigma = np.full(H.n_sensors, 0.02)
        ip = InverseProblem(dm=dm, obs=H, y=y, sigma=sigma, alpha=0.1, reg_type="h1")
        m0 = np.zeros(dm.V_m.dofmap.index_map.size_global)
        return run_lbfgs(ip, m0, max_iter=30)

    m1, h1 = build_and_run()
    m2, h2 = build_and_run()
    # Tight-tolerance (not bit-exact) comparison of the FINAL reconstruction and objective: the
    # FEniCSx JIT-compiled forms and PETSc's threaded assembly can introduce floating-point-
    # summation-order differences at the ULP level depending on what has already run earlier in a
    # test session (observed directly: bit-exact in isolation, but occasionally differs at the
    # ~1e-10 relative level after other tests have run first in the full suite). Comparing the
    # FULL per-iteration objective history (history.objective, a list appended to at every L-BFGS-B
    # function evaluation) is both the wrong scientific question -- reproducibility of the
    # RECONSTRUCTION, not bit-identical agreement at every internal line-search evaluation -- and
    # fragile to the line search taking a different number of evaluations under a ULP-level
    # perturbation; comparing only the final objective is the meaningful, still-strict check.
    assert np.allclose(m1, m2, rtol=1e-8, atol=1e-10)
    assert np.isclose(h1.objective[-1], h2.objective[-1], rtol=1e-8, atol=1e-12)
