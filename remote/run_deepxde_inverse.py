"""Remote-only manufactured constant-k Darcy PINN experiment."""
import argparse
import math
import sys

from remote.common import (evaluation_grid, failed_seeds, relative_l2,
                           require_remote_cuda, run_seed, write_results)
from src.deepxde_inverse_constant_k import K_TRUE, build_inverse_model, exact_pressure
from src.deepxde_tracking import loss_history, parameter_trace


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    p.add_argument("--observations", type=int, default=100)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--k-initial", type=float, default=0.8)
    p.add_argument("--observation-seed", type=int, default=None,
                   help="fix sensor layout across seeds (default: layout follows --seeds)")
    p.add_argument("--trace-every", type=int, default=100)
    p.add_argument("--output", default="results/deepxde_inverse_remote.json")
    a = p.parse_args(argv)
    if a.iterations < 1 or len(set(a.seeds)) != len(a.seeds):
        raise ValueError("iterations must be positive and seeds unique")
    if a.observations < 1 or a.noise_std < 0 or a.k_initial <= 0 or a.trace_every < 1:
        raise ValueError("observations/trace-every positive, noise-std >= 0, k-initial > 0")
    return a


def main(argv=None):
    a = parse_args(argv)
    require_remote_cuda()
    grid = evaluation_grid()
    p_reference = exact_pressure(grid)

    def fit(seed):
        model, log_k = build_inverse_model(
            seed=seed, n_observations=a.observations, noise_std=a.noise_std,
            k_initial=a.k_initial, observation_seed=a.observation_seed)
        trace = parameter_trace({"k": log_k}, {"k": math.exp}, period=a.trace_every)
        model.train(iterations=a.iterations, display_every=max(1, a.iterations // 10),
                    callbacks=[trace])
        estimated = math.exp(float(log_k.detach().cpu().item()))
        pressure_error = relative_l2(model.predict(grid), p_reference)
        return {"seed": seed, "k_true": K_TRUE, "k_estimated": estimated,
                "relative_k_error": abs(estimated-K_TRUE)/K_TRUE,
                "pressure_relative_l2": pressure_error,
                "final_loss_train": [float(v) for v in model.train_state.loss_train],
                "parameter_trace": trace.trace, "loss_history": loss_history(model),
                "finite_check": ["k_estimated", "relative_k_error", "pressure_relative_l2"]}

    config = {"benchmark": "constant_k", "iterations": a.iterations, "seeds": a.seeds,
              "observations": a.observations, "noise_std": a.noise_std,
              "k_initial": a.k_initial, "observation_seed": a.observation_seed,
              "evaluation_grid": "41x41 uniform on [0,1]^2", "trace_every": a.trace_every}
    records = []
    for seed in a.seeds:
        records.append(run_seed(seed, fit))
        write_results(a.output, config, records)
    failed = failed_seeds(records)
    print(f"Wrote {len(records)} records to {a.output}; failed seeds: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
