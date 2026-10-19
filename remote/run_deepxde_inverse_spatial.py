"""Remote-only multi-seed spatially varying Darcy inverse PINN experiment."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

from remote.common import (evaluation_grid, failed_seeds, relative_l2,
                           require_remote_cuda, run_seed, synchronize, write_results)
from src.deepxde_inverse_spatial_k import (ALPHA_TRUE, BETA_TRUE, build_model,
                                           exact_log_permeability, exact_pressure)
from src.deepxde_tracking import loss_history, parameter_trace


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    parser.add_argument("--observations", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--alpha-initial", type=float, default=0.0)
    parser.add_argument("--beta-initial", type=float, default=0.0)
    parser.add_argument("--observation-seed", type=int, default=None,
                        help="fix sensor layout across seeds (default: layout follows --seeds)")
    parser.add_argument("--trace-every", type=int, default=100)
    parser.add_argument("--lbfgs-maxiter", type=int, default=0,
                        help="0 (default): Adam only. >0: explicit 'adam+lbfgs' configuration, "
                             "L-BFGS refinement after the Adam stage")
    parser.add_argument("--lbfgs-record-every", type=int, default=10,
                        help="keep every n-th L-BFGS function evaluation in the history")
    parser.add_argument("--save-state", default=None,
                        help="directory for per-seed network and parameter state (not for Git)")
    parser.add_argument("--output", default="results/deepxde_inverse_spatial_remote.json")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.observations < 1 or args.trace_every < 1:
        raise ValueError("iterations, observations and trace-every must be positive")
    if args.noise_std < 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("noise-std must be non-negative and seeds unique")
    if args.lbfgs_maxiter < 0 or args.lbfgs_record_every < 1:
        raise ValueError("lbfgs-maxiter must be >= 0 and lbfgs-record-every positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    require_remote_cuda()
    grid = evaluation_grid()
    p_reference = exact_pressure(grid)
    k_reference = np.exp(exact_log_permeability(grid))

    def evaluate(model, alpha, beta):
        a_est = float(alpha.detach().cpu())
        b_est = float(beta.detach().cpu())
        k_estimated = np.exp(exact_log_permeability(grid, alpha=a_est, beta=b_est))
        return {
            "alpha_estimated": a_est, "beta_estimated": b_est,
            "alpha_abs_error": abs(a_est-ALPHA_TRUE), "beta_abs_error": abs(b_est-BETA_TRUE),
            "parameter_relative_error": float(np.hypot(a_est-ALPHA_TRUE, b_est-BETA_TRUE)
                                              / np.hypot(ALPHA_TRUE, BETA_TRUE)),
            "pressure_relative_l2": relative_l2(model.predict(grid), p_reference),
            "permeability_relative_l2": relative_l2(k_estimated, k_reference),
            "final_loss_train": [float(v) for v in model.train_state.loss_train],
        }

    def save_state(model, alpha, beta, seed, stage):
        if args.save_state:
            import torch
            directory = Path(args.save_state)
            directory.mkdir(parents=True, exist_ok=True)
            torch.save({"net": model.net.state_dict(), "alpha": float(alpha.detach()),
                        "beta": float(beta.detach()), "seed": seed, "stage": stage,
                        "config": config}, directory / f"seed{seed}_{stage}.pt")

    def fit(seed):
        model, alpha, beta = build_model(
            seed=seed, observations=args.observations, noise_std=args.noise_std,
            observation_seed=args.observation_seed, alpha_initial=args.alpha_initial,
            beta_initial=args.beta_initial)
        trace = parameter_trace({"alpha": alpha, "beta": beta}, period=args.trace_every)
        start = time.perf_counter()
        model.train(iterations=args.iterations,
                    display_every=max(1, args.iterations // 10), callbacks=[trace])
        synchronize()
        adam_seconds = time.perf_counter() - start
        record = {"seed": seed, "alpha_true": ALPHA_TRUE, "beta_true": BETA_TRUE,
                  **evaluate(model, alpha, beta), "parameter_trace": trace.trace}
        finite = ["alpha_estimated", "beta_estimated", "pressure_relative_l2",
                  "permeability_relative_l2"]
        save_state(model, alpha, beta, seed, "adam")
        if args.lbfgs_maxiter:
            from src.deepxde_lbfgs import refine_lbfgs
            adam_stage = {k: record[k] for k in (*finite, "alpha_abs_error", "beta_abs_error",
                                                 "parameter_relative_error", "final_loss_train")}
            adam_stage["seconds"] = adam_seconds
            start = time.perf_counter()
            lbfgs = refine_lbfgs(model, {"alpha": alpha, "beta": beta}, args.lbfgs_maxiter,
                                 record_every=args.lbfgs_record_every)
            synchronize()
            lbfgs["seconds"] = time.perf_counter() - start
            record.update(evaluate(model, alpha, beta))
            record["final_loss_train"] = lbfgs["final_loss_components"]
            record["adam_stage"] = adam_stage
            record["lbfgs"] = lbfgs
            record["lbfgs_final_total_loss"] = lbfgs["final_total_loss"]
            finite.append("lbfgs_final_total_loss")
            save_state(model, alpha, beta, seed, "adam_lbfgs")
        record["adam_seconds"] = adam_seconds
        record["loss_history"] = loss_history(model)
        record["finite_check"] = finite
        return record

    config = {"benchmark": "spatial_k",
              "optimizer": "adam+lbfgs" if args.lbfgs_maxiter else "adam",
              "lbfgs_maxiter": args.lbfgs_maxiter,
              "iterations": args.iterations, "seeds": args.seeds,
              "observations": args.observations, "noise_std": args.noise_std,
              "alpha_initial": args.alpha_initial, "beta_initial": args.beta_initial,
              "observation_seed": args.observation_seed,
              "evaluation_grid": "41x41 uniform on [0,1]^2", "trace_every": args.trace_every}
    records = []
    for seed in args.seeds:
        records.append(run_seed(seed, fit))
        write_results(args.output, config, records)
    failed = failed_seeds(records)
    print(f"Wrote {len(records)} records to {args.output}; failed seeds: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
