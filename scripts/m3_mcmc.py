"""Milestone 3 MCMC: pCN in the mean-free (identifiable) KL subspace, one chain
per process, at the historical full Bayesian configuration.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/m3_mcmc.py chain \
        --truth smooth --beta 0.02 --n-proposals 8000 --seed 1 --init map --thin 1 \
        --out results/m3/mcmc/pilot/smooth_b0.02_map_s1.npz

The pCN kernel (src/pcn_sampler.py) is unchanged; it acts on the 14 coefficients
xi_R, whose exact marginal posterior is L(xi_R) N(xi_R; 0, I), because the
likelihood does not depend on the constant-mode coefficient (src/constant_mode.py).
The chain runs in blocks (default 10,000 proposals); block b uses seed
seed*100000 + b and starts from the last state of block b-1, and a checkpoint is
written after every block. Draws are stored after thinning.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBES = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)]


def build(truth: str):
    """Historical full configuration (configs/full.yaml: bayesian_posterior, data seed 7)."""
    from src.bayesian_darcy import BayesianDarcyProblem
    from src.constant_mode import ReducedKL
    from src.data_generation import generate_synthetic_data
    from src.forward import build_mesh
    from src.kl_prior import build_kl_prior
    from src.observation import build_observation_operator, grid_sensors
    from src.truth_fields import smooth_truth, structured_truth

    rc = yaml.safe_load((ROOT / "configs" / "full.yaml").read_text())["bayesian_posterior"]
    truth_fn = {"smooth": smooth_truth(seed=1), "structured": structured_truth()}[truth]
    sensors = grid_sensors(rc["sensor_grid"], rc["sensor_grid"])
    data = generate_synthetic_data(truth_fn, sensors, rc["truth_mesh_n"], rc["sensor_width"],
                                   rc["relative_noise"], 7)
    dm = build_mesh(rc["inv_mesh_n"])
    H = build_observation_operator(dm.mesh, dm.V_p, sensors, rc["sensor_width"])
    prior = build_kl_prior(dm, r=rc["prior_r"], gamma=rc["prior_gamma"], delta=rc["prior_delta"])
    bp = BayesianDarcyProblem(dm=dm, obs=H, y=data.y_noisy, sigma=data.sigma, prior=prior)
    rk = ReducedKL.from_prior(prior)
    coords = dm.V_m.tabulate_dof_coordinates()[:, :2]
    probe_idx = []
    for p in PROBES:
        d = np.linalg.norm(coords - np.array(p), axis=1)
        i = int(np.argmin(d))
        assert d[i] < 1e-12, "probe points must be mesh nodes"
        probe_idx.append(i)
    from dolfinx.fem import Function
    mt = Function(dm.V_m)
    mt.interpolate(truth_fn)
    return {"rc": rc, "data": data, "dm": dm, "prior": prior, "bp": bp, "rk": rk,
            "probe_idx": np.array(probe_idx), "m_true": mt.x.array.copy()}


def p_map(env):
    from src.preconditioned_inversion import gauss_newton_kl_hessian, run_map_preconditioned
    bp, rk = env["bp"], env["rk"]
    P = gauss_newton_kl_hessian(bp, rk, np.zeros(rk.r_reduced))
    xi, diag, g, g0, L = run_map_preconditioned(bp, rk, P, max_iter=env["rc"]["map_max_iter"])
    return xi, diag


def run_chain(args):
    from src.constant_mode import reduced_neg_log_likelihood
    from src.pcn_sampler import run_pcn
    env = build(args.truth)
    bp, rk = env["bp"], env["rk"]
    xi_map, map_diag = p_map(env)
    xr_map = rk.reduced_xi(xi_map)
    if args.init == "map":
        x0 = xr_map.copy()
    else:
        x0 = xr_map + np.random.default_rng(10_000 + args.seed).standard_normal(rk.r_reduced)
    phi = reduced_neg_log_likelihood(bp, rk)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    stored_x, stored_phi = [], []
    n_done, n_acc, t_run = 0, 0, 0.0
    f0, a0 = bp.solve_counts()
    x = x0
    block = 0
    meta = {"truth": args.truth, "beta": args.beta, "seed": args.seed, "init": args.init, "thin": args.thin,
            "block": args.block, "n_proposals_target": args.n_proposals, "x0": x0.tolist(),
            "xi_map_reduced": xr_map.tolist(), "map_termination": map_diag.termination_reason,
            "map_iterations": map_diag.n_iterations, "sampler": "pCN on mean-free KL subspace (r-1=14)",
            "likelihood_path": "forward_only", "config": env["rc"], "data_seed": 7}
    while n_done < args.n_proposals:
        nb = min(args.block, args.n_proposals - n_done)
        t0 = time.perf_counter()
        res = run_pcn(phi, r=rk.r_reduced, n_samples=nb, beta=args.beta, burn_in=0,
                      seed=args.seed * 100000 + block, x0=x)
        t_run += time.perf_counter() - t0
        chain = res.raw_chain[1:]
        n_acc += int(round(res.acceptance_rate * res.n_proposals))
        offset = (-n_done) % args.thin  # keep global thinning phase
        stored_x.append(chain[offset::args.thin])
        stored_phi.append(res.neg_log_lik[offset::args.thin])
        x = chain[-1]
        n_done += nb
        block += 1
        f1, a1 = bp.solve_counts()
        meta.update({"n_proposals_done": n_done, "acceptance_rate": n_acc / n_done,
                     "pcn_seconds": t_run, "ms_per_proposal": 1e3 * t_run / n_done,
                     "forward_solves": f1 - f0, "adjoint_solves": a1 - a0, "blocks": block})
        np.savez(out, xi_r=np.concatenate(stored_x), phi=np.concatenate(stored_phi),
                 meta=json.dumps(meta))
        print(f"{out.name}: {n_done}/{args.n_proposals} proposals, acc={n_acc / n_done:.3f}, "
              f"{1e3 * t_run / n_done:.2f} ms/proposal", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chain")
    c.add_argument("--truth", choices=["smooth", "structured"], required=True)
    c.add_argument("--beta", type=float, required=True)
    c.add_argument("--n-proposals", type=int, required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--init", choices=["map", "dispersed"], default="dispersed")
    c.add_argument("--thin", type=int, default=1)
    c.add_argument("--block", type=int, default=10_000)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "chain":
        run_chain(args)


if __name__ == "__main__":
    main()
