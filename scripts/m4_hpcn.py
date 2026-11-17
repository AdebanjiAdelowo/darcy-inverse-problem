"""Milestone 4: Hessian-informed pCN (src/hessian_pcn.py) on the 14 mean-free KL
coefficients of the historical full Bayesian configuration.

  prepare  compute the Milestone 3 P-MAP and the Gauss-Newton misfit Hessian at
           the MAP for one truth; writes results/m4/hessian_TRUTH.json
  chain    run (or resume) one chain with checkpoints
  qoi      log effective permeability of the mean-free field for every
           retained draw of a finished chain (one forward solve per draw)

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/m4_hpcn.py chain --truth smooth \
        --rho 0.75 --seed 1 --init laplace3 --n-proposals 4000 --thin 1 \
        --checkpoint-every 1000 --out results/m4/chains/pilot/smooth_r0.75_s1.npz

Chains (results/m4/chains/, outside Git) are complete checkpoints: retained draws,
their Phi values, and the full chain state (point, Phi, RNG state, counters,
proposal). Re-running the same command on an existing file resumes it.
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
OUT = ROOT / "results" / "m4"


def prepare(truth: str) -> dict:
    from m3_mcmc import build, p_map
    from src.preconditioned_inversion import gauss_newton_kl_hessian
    env = build(truth)
    t0 = time.perf_counter()
    xi_map, diag = p_map(env)
    rk = env["rk"]
    xr = rk.reduced_xi(xi_map)
    P = gauss_newton_kl_hessian(env["bp"], rk, xr)
    H = P - np.eye(rk.r_reduced)
    rec = {"truth": truth, "xi_r_map": xr.tolist(), "map_termination": diag.termination_reason,
           "map_iterations": diag.n_iterations, "map_converged": diag.converged,
           "gauss_newton_misfit_hessian": H.tolist(),
           "hessian_eigenvalues": np.linalg.eigvalsh(0.5 * (H + H.T)).tolist(),
           "hessian_asymmetry_rel": float(np.max(np.abs(H - H.T)) / np.max(np.abs(H))),
           "seconds": time.perf_counter() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"hessian_{truth}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{truth}: MAP {diag.termination_reason} in {diag.n_iterations} it; "
          f"eigenvalues {min(rec['hessian_eigenvalues']):.3g}..{max(rec['hessian_eigenvalues']):.3g}")
    return rec


def _start(rec, H, init, seed):
    xr = np.array(rec["xi_r_map"])
    rng = np.random.default_rng(10_000 + seed)
    if init == "prior":
        return xr + rng.standard_normal(len(xr))
    lam, V = np.linalg.eigh(0.5 * (H + H.T))
    L = V / np.sqrt(1.0 + np.maximum(lam, 0.0))  # L L^T = (I + H)^-1
    return xr + 3.0 * L @ rng.standard_normal(len(xr))


def run_chain(a):
    from m3_mcmc import build
    from src import mcmc_diagnostics as md
    from src.constant_mode import reduced_neg_log_likelihood
    from src.hessian_pcn import (HessianInformedProposal, HessianPCNConfig, HpCNChain, load_checkpoint,
                                 save_checkpoint)
    rec = json.loads((OUT / f"hessian_{a.truth}.json").read_text())
    H = np.array(rec["gauss_newton_misfit_hessian"])
    env = build(a.truth)
    bp, rk = env["bp"], env["rk"]
    phi = reduced_neg_log_likelihood(bp, rk)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    f0, a0 = bp.solve_counts()
    if out.exists():
        draws, phis, state, extra = load_checkpoint(out)
        chain = HpCNChain.from_state(phi, state)
        log = extra.get("diagnostic_log", [])
        solves_prev = extra.get("forward_solves", 0)
        print(f"resuming {out.name} at {chain.n_steps} proposals", flush=True)
    else:
        cfg = HessianPCNConfig(rho=a.rho, beta_max=a.beta_max, eig_floor=a.eig_floor)
        prop = HessianInformedProposal.from_hessian(H, cfg)
        chain = HpCNChain(phi, prop, _start(rec, H, a.init, a.seed), seed=a.seed)
        draws, phis = np.zeros((0, rk.r_reduced)), np.zeros(0)
        log, solves_prev = [], 0
    while chain.n_steps < a.n_proposals:
        n = min(a.checkpoint_every, a.n_proposals - chain.n_steps)
        d, p = chain.step(n, thin=a.thin)
        draws, phis = np.vstack([draws, d]), np.r_[phis, p]
        if a.diagnostic_every and (chain.n_steps % a.diagnostic_every == 0 or chain.n_steps == a.n_proposals):
            h = len(draws) // 2
            post = draws[h:]
            log.append({"n_steps": chain.n_steps, "acceptance": chain.acceptance_rate,
                        "single_chain_max_split_rhat": float(max(md.rhat(post[:, j]) for j in range(post.shape[1]))),
                        "single_chain_min_bulk_ess": float(min(md.ess_bulk(post[:, j]) for j in range(post.shape[1])))})
        f1, a1 = bp.solve_counts()
        extra = {"truth": a.truth, "rho": chain.proposal.config.rho, "init": a.init, "thin": a.thin,
                 "n_proposals_target": a.n_proposals, "seed": chain.seed, "x0_rule": a.init,
                 "forward_solves": solves_prev + (f1 - f0), "adjoint_solves": a1 - a0,
                 "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                 "diagnostic_log": log, "hessian_file": f"results/m4/hessian_{a.truth}.json",
                 "data_seed": 7, "sampler": "HpCN on mean-free KL subspace (14 dims)"}
        save_checkpoint(out, chain, draws, phis, extra)
        print(f"{out.name}: {chain.n_steps}/{a.n_proposals}, acc={chain.acceptance_rate:.3f}, "
              f"{1e3 * chain.seconds / chain.n_steps:.2f} ms/proposal", flush=True)


def qoi(a):
    from m3_mcmc import build
    from src.hessian_pcn import load_checkpoint
    from src.qoi import effective_permeability
    env = build(a.truth)
    rk = env["rk"]
    path = ROOT / a.chain
    draws, _, state, _ = load_checkpoint(path)
    t0 = time.perf_counter()
    logq = np.array([np.log(effective_permeability(env["dm"], rk.mean_free_field(x))) for x in draws])
    out = path.with_name(path.stem + "_qoi.npz")
    np.savez(out, log_q_r=logq, seconds=time.perf_counter() - t0, n_solves=len(draws))
    print(f"{out.name}: {len(draws)} draws, {time.perf_counter() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--truth", choices=["smooth", "structured"], required=True)
    c = sub.add_parser("chain")
    c.add_argument("--truth", choices=["smooth", "structured"], required=True)
    c.add_argument("--rho", type=float, required=True)
    c.add_argument("--beta-max", type=float, default=1.0)
    c.add_argument("--eig-floor", type=float, default=0.0)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--init", choices=["laplace3", "prior"], default="laplace3")
    c.add_argument("--n-proposals", type=int, required=True)
    c.add_argument("--thin", type=int, default=1)
    c.add_argument("--checkpoint-every", type=int, default=1000)
    c.add_argument("--diagnostic-every", type=int, default=0)
    c.add_argument("--out", required=True)
    q = sub.add_parser("qoi")
    q.add_argument("--truth", choices=["smooth", "structured"], required=True)
    q.add_argument("--chain", required=True)
    a = ap.parse_args()
    {"prepare": lambda: prepare(a.truth), "chain": lambda: run_chain(a), "qoi": lambda: qoi(a)}[a.cmd]()


if __name__ == "__main__":
    main()
