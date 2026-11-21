"""Milestone 5 simulation-based calibration (pre-registered in docs/audit/M5_PREREGISTRATION.md).

Generative model = inference model: xi ~ N(0, I_15) on the N=16 KL prior (constant
mode included), pressure by the N=16 forward solver, 5x5 Gaussian-bump sensors,
y = H p(m) + N(0, sigma^2 I), sigma = 0.0098584 fixed.

  mcmc     full HpCN inference for datasets [start, start + n): writes
           results/m5/sbc/datasets/ds_XXXX.json and a thinned chain archive
           results/m5/chains/sbc/ds_XXXX.npz (outside Git)
  laplace  Laplace-only inference (MAP + Gauss-Newton Hessian) for a batch:
           results/m5/sbc/laplace/<tag>.json

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/m5_sbc.py mcmc --start 0 --n 4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "m5" / "sbc"
SIGMA = 0.0098584
PROBES = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)]
RHO, N_CHAINS, N_PROP, L_RANK = 0.5, 4, 10_000, 99


def build_env():
    import yaml
    from src.forward import build_mesh
    from src.identifiability import mass_matrix
    from src.kl_prior import build_kl_prior
    from src.observation import build_observation_operator, grid_sensors
    from src.constant_mode import ReducedKL
    rc = yaml.safe_load((ROOT / "configs" / "full.yaml").read_text())["bayesian_posterior"]
    dm = build_mesh(rc["inv_mesh_n"])
    sensors = grid_sensors(rc["sensor_grid"], rc["sensor_grid"])
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, rc["sensor_width"])
    prior = build_kl_prior(dm, r=rc["prior_r"], gamma=rc["prior_gamma"], delta=rc["prior_delta"])
    rk = ReducedKL.from_prior(prior)
    coords = dm.V_m.tabulate_dof_coordinates()[:, :2]
    probe_idx = np.array([int(np.argmin(np.linalg.norm(coords - np.array(p), axis=1))) for p in PROBES])
    return {"rc": rc, "dm": dm, "H": H, "prior": prior, "rk": rk, "M": mass_matrix(dm.V_m), "probe_idx": probe_idx}


def make_dataset(env, seed: int, sigma: float):
    from src.bayesian_darcy import BayesianDarcyProblem
    from src.forward import solve_forward
    from dolfinx.fem import Function
    rng = np.random.default_rng(seed)
    prior = env["prior"]
    xi = rng.standard_normal(prior.r)
    m = Function(env["dm"].V_m)
    m.x.array[:] = prior.sample_array(xi)
    y_clean = env["H"].apply(solve_forward(env["dm"], m))
    y = y_clean + sigma * rng.standard_normal(len(y_clean))
    bp = BayesianDarcyProblem(dm=env["dm"], obs=env["H"], y=y, sigma=np.full(len(y), sigma), prior=prior)
    return xi, m.x.array.copy(), bp


def map_and_hessian(env, bp):
    from src.preconditioned_inversion import gauss_newton_kl_hessian, run_map_preconditioned
    rk = env["rk"]
    P0 = gauss_newton_kl_hessian(bp, rk, np.zeros(rk.r_reduced))
    xi_map, diag, *_ = run_map_preconditioned(bp, rk, P0, max_iter=env["rc"]["map_max_iter"])
    xr = rk.reduced_xi(xi_map)
    Hm = gauss_newton_kl_hessian(bp, rk, xr) - np.eye(rk.r_reduced)
    return xr, 0.5 * (Hm + Hm.T), diag


def linear_maps(env):
    """xi_R -> m_R at probes (linear) and ||m_R||^2 weights."""
    rk = env["rk"]
    lam = rk.prior.eigenvalues[rk.keep]
    B = rk.prior.eigenvectors[env["probe_idx"]][:, rk.keep] * np.sqrt(lam)[None, :]  # (5, 14)
    return B, lam


def laplace_quantities(env, xr, Hm, xi_true):
    from src import calibration as cal
    rk = env["rk"]
    B, _ = linear_maps(env)
    cov = np.linalg.inv(np.eye(rk.r_reduced) + Hm)
    tr = xi_true[rk.keep]
    out = {}
    for j in range(rk.r_reduced):
        sd = float(np.sqrt(cov[j, j]))
        out[f"xi_R[{j}]"] = {"mean": float(xr[j]), "sd": sd, "truth": float(tr[j]),
                             "pit": cal.gaussian_pit(xr[j], sd, tr[j])}
    pm, pc = B @ xr, B @ cov @ B.T
    pt = B @ tr
    for k in range(len(PROBES)):
        sd = float(np.sqrt(pc[k, k]))
        out[f"m_R{PROBES[k]}"] = {"mean": float(pm[k]), "sd": sd, "truth": float(pt[k]),
                                  "pit": cal.gaussian_pit(pm[k], sd, pt[k])}
    return out


def run_mcmc_dataset(env, i: int, n_prop: int = N_PROP, chain_dir=None) -> dict:
    from src import calibration as cal
    from src import mcmc_diagnostics as md
    from src.constant_mode import reduced_neg_log_likelihood
    from src.hessian_pcn import HessianInformedProposal, HessianPCNConfig, HpCNChain
    from src.identifiability import l2_norm, reconstruction_errors
    from src.qoi import effective_permeability
    from m4_hpcn import _start
    rk, prior = env["rk"], env["prior"]
    t0 = time.perf_counter()
    xi_true, m_true, bp = make_dataset(env, 5000 + i, SIGMA)
    t_data = time.perf_counter() - t0
    t1 = time.perf_counter()
    xr, Hm, mdiag = map_and_hessian(env, bp)
    t_map = time.perf_counter() - t1
    f0, _ = bp.solve_counts()
    prop = HessianInformedProposal.from_hessian(Hm, HessianPCNConfig(rho=RHO))
    phi = reduced_neg_log_likelihood(bp, rk)
    chains, phis, accs, secs = [], [], [], 0.0
    for c in range(N_CHAINS):
        ch = HpCNChain(phi, prop, _start({"xi_r_map": xr.tolist()}, Hm, "laplace3", c + 1), seed=c + 1)
        d, p = ch.step(n_prop)
        chains.append(d); phis.append(p); accs.append(ch.acceptance_rate); secs += ch.seconds
    f1, _ = bp.solve_counts()
    arr = np.stack(chains)
    post = arr[:, n_prop // 2:, :]
    phipost = np.stack(phis)[:, n_prop // 2:]
    # per-dataset diagnostics (xi_R and Phi)
    rh = [md.rhat(post[:, :, j]) for j in range(rk.r_reduced)] + [md.rhat(phipost)]
    eb = [md.ess_bulk(post[:, :, j]) for j in range(rk.r_reduced)] + [md.ess_bulk(phipost)]
    et = [md.ess_tail(post[:, :, j]) for j in range(rk.r_reduced)] + [md.ess_tail(phipost)]
    rng_tie = np.random.default_rng(123 + i)
    flat = post.reshape(-1, rk.r_reduced)
    rank_xi = cal.select_rank_draws(post, L_RANK)
    B, lam = linear_maps(env)
    tr = xi_true[rk.keep]
    sd_c = float(np.sqrt(rk.lambda_c) * abs(rk.phi_c[0]))
    q = {}

    def add(name, all_draws, rank_draws, truth):
        q[name] = {"truth": float(truth), "mean": float(np.mean(all_draws)), "sd": float(np.std(all_draws, ddof=1)),
                   "rank": cal.rank_of_truth(rank_draws, truth, rng_tie),
                   "pit": float(np.mean(all_draws < truth))}

    for j in range(rk.r_reduced):
        add(f"xi_R[{j}]", flat[:, j], rank_xi[:, j], tr[j])
    probes_all, probes_rank, probes_true = flat @ B.T, rank_xi @ B.T, B @ tr
    for k in range(len(PROBES)):
        add(f"m_R{PROBES[k]}", probes_all[:, k], probes_rank[:, k], probes_true[k])
    norm_all = np.sqrt((flat ** 2 * lam).sum(1))
    add("||m_R||_L2", norm_all, np.sqrt((rank_xi ** 2 * lam).sum(1)), np.sqrt((tr ** 2 * lam).sum()))
    # log Q_R on the rank draws (one forward solve each) and the truth
    t2 = time.perf_counter()
    logq = np.array([np.log(effective_permeability(env["dm"], rk.mean_free_field(x))) for x in rank_xi])
    mR_true = rk.mean_free_field(tr)
    logq_true = float(np.log(effective_permeability(env["dm"], mR_true)))
    logk_true = float(np.log(effective_permeability(env["dm"], m_true)))
    t_qoi = time.perf_counter() - t2
    add("log Q_R", logq, logq, logq_true)
    # quantities with the exact constant mode (Rao-Blackwellised PIT)
    xc_true = float(xi_true[rk.const_index])
    q["xi_c"] = {"truth": xc_true, "mean": 0.0, "sd": 1.0, "pit": cal.rb_pit(np.zeros(1), 1.0, xc_true)}
    for k in range(len(PROBES)):
        idx = env["probe_idx"][k]
        q[f"m{PROBES[k]} (absolute)"] = {"truth": float(m_true[idx]), "mean": float(probes_all[:, k].mean()),
                                         "sd": float(np.sqrt(probes_all[:, k].var(ddof=1) + sd_c ** 2)),
                                         "pit": cal.rb_pit(probes_all[:, k], sd_c, float(m_true[idx]))}
    q["log k_eff (absolute)"] = {"truth": logk_true, "mean": float(logq.mean()),
                                 "sd": float(np.sqrt(logq.var(ddof=1) + sd_c ** 2)),
                                 "pit": cal.rb_pit(logq, sd_c, logk_true)}
    # mean-free reconstruction error of the posterior mean
    pm_field = rk.mean_free_field(flat.mean(0))
    err = l2_norm(pm_field - mR_true, env["M"]) / l2_norm(mR_true, env["M"])  # m_R is mean-free by construction
    rec = {
        "dataset": i, "seed": 5000 + i, "sigma": SIGMA, "rho": RHO, "n_chains": N_CHAINS, "n_proposals": n_prop,
        "map": {"termination": mdiag.termination_reason, "iterations": mdiag.n_iterations, "converged": mdiag.converged},
        "hessian_eigen_range": [float(np.linalg.eigvalsh(Hm).min()), float(np.linalg.eigvalsh(Hm).max())],
        "acceptance": accs, "max_rhat": float(np.nanmax(rh)), "min_ess_bulk": float(np.nanmin(eb)),
        "min_ess_tail": float(np.nanmin(et)),
        "max_rel_mcse": float(max(md.mcse_mean(post[:, :, j]) / post[:, :, j].std() for j in range(rk.r_reduced))),
        "quantities": q, "laplace": laplace_quantities(env, xr, Hm, xi_true),
        "mean_free_rel_L2_error_posterior_mean": float(err),
        "cost": {"seconds_data": t_data, "seconds_map_hessian": t_map, "seconds_sampling": secs,
                 "seconds_qoi": t_qoi, "seconds_total": time.perf_counter() - t0,
                 "forward_solves_sampling": int(f1 - f0), "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6},
    }
    chain_dir = ROOT / "results" / "m5" / "chains" / "sbc" if chain_dir is None else pathlib.Path(chain_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)
    np.savez(chain_dir / f"ds_{i:04d}.npz", draws_thin5=arr[:, ::5, :],
             phi_thin5=np.stack(phis)[:, ::5], xi_true=xi_true, xr_map=xr, hessian=Hm,
             meta=json.dumps({k: rec[k] for k in ("dataset", "seed", "sigma", "rho", "n_chains", "n_proposals")}))
    return rec


def run_laplace_batch(env, start, n, seed_base, sigma_mult, tag):
    out = []
    for i in range(start, start + n):
        t0 = time.perf_counter()
        xi_true, _, bp = make_dataset(env, seed_base + i, SIGMA * sigma_mult)
        xr, Hm, mdiag = map_and_hessian(env, bp)
        out.append({"dataset": i, "seed": seed_base + i, "map_converged": mdiag.converged,
                    "map_termination": mdiag.termination_reason,
                    "quantities": laplace_quantities(env, xr, Hm, xi_true), "seconds": time.perf_counter() - t0})
    path = OUT / "laplace" / f"{tag}_{start:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sigma": SIGMA * sigma_mult, "sigma_mult": sigma_mult, "seed_base": seed_base,
                                "datasets": out}, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("mcmc")
    a.add_argument("--start", type=int, required=True)
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--n-proposals", type=int, default=N_PROP)
    a.add_argument("--out-dir", default=None, help="smoke runs only; default results/m5/sbc")
    b = sub.add_parser("laplace")
    b.add_argument("--start", type=int, required=True)
    b.add_argument("--n", type=int, required=True)
    b.add_argument("--seed-base", type=int, required=True)
    b.add_argument("--sigma-mult", type=float, required=True)
    b.add_argument("--tag", required=True)
    args = ap.parse_args()
    env = build_env()
    if args.cmd == "mcmc":
        base = OUT if args.out_dir is None else pathlib.Path(args.out_dir)
        (base / "datasets").mkdir(parents=True, exist_ok=True)
        for i in range(args.start, args.start + args.n):
            path = base / "datasets" / f"ds_{i:04d}.json"
            if path.exists():
                print(f"skip {path.name} (exists)")
                continue
            rec = run_mcmc_dataset(env, i, args.n_proposals,
                                   None if args.out_dir is None else pathlib.Path(args.out_dir) / "chains")
            path.write_text(json.dumps(rec, indent=1) + "\n")
            print(f"ds {i}: maxRhat={rec['max_rhat']:.3f} minBulk={rec['min_ess_bulk']:.0f} "
                  f"t={rec['cost']['seconds_total']:.0f}s", flush=True)
    else:
        run_laplace_batch(env, args.start, args.n, args.seed_base, args.sigma_mult, args.tag)


if __name__ == "__main__":
    main()
