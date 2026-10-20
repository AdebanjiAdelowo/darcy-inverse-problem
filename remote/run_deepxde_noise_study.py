"""Remote-only paired study of L-BFGS-stage objectives for the noisy spatial benchmark.

For every (network seed, layout seed, noise seed) tuple, one Adam stage (the
validated unit-weight configuration) is run, or loaded from a validated
checkpoint, and every requested L-BFGS-stage objective
(src/deepxde_noise_methods.py) is then started from that identical Adam state.
Comparisons between methods are therefore paired by construction. Optional
Adam snapshots run the same refinements from earlier Adam iterations of the same
run (training-efficiency study); ``--float64`` builds the model in float64.

Each record keeps the Adam-stage result and one entry per (snapshot, method).
Failures are recorded per tuple; the process exits 1 at the end if any occurred.
"""
import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np

from remote.common import (evaluation_grid, failed_seeds, relative_l2, require_remote_cuda,
                           run_seed, synchronize, write_results)
from src.deepxde_inverse_spatial_k import (ALPHA_TRUE, BETA_TRUE, build_model,
                                           exact_log_permeability, exact_pressure)
from src.deepxde_noise_methods import (METHODS, TAU, loss_counts, method_objective,
                                       prior_penalty)
from src.deepxde_tracking import loss_history, parameter_trace


def reset_peak_memory():
    import torch
    if torch.cuda.is_available():
        reset_peak_memory()


def peak_memory():
    import torch
    return peak_memory() if torch.cuda.is_available() else 0


def parse_tuple(text):
    try:
        network, layout, noise = (int(v) for v in text.split(":"))
    except ValueError as exc:
        raise ValueError(f"tuple {text!r} is not network:layout:noise") from exc
    return network, layout, noise


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tuples", nargs="+", required=True,
                   help="network:layout:noise seed triples, e.g. 42:42:42")
    p.add_argument("--noise-std", type=float, required=True,
                   help="absolute Gaussian noise standard deviation (pressure peak is 1)")
    p.add_argument("--observations", type=int, default=128)
    p.add_argument("--iterations", type=int, default=50000, help="Adam iterations")
    p.add_argument("--adam-snapshots", type=int, nargs="*", default=[],
                   help="extra Adam iterations to refine from (efficiency study)")
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--lbfgs-maxiter", type=int, default=15000)
    p.add_argument("--lbfgs-record-every", type=int, default=10)
    p.add_argument("--init-from", default=None,
                   help="directory of validated seed<N>_adam.pt checkpoints; skips Adam")
    p.add_argument("--float64", action="store_true")
    p.add_argument("--trace-every", type=int, default=500)
    p.add_argument("--save-state", default=None)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    a.tuples = [parse_tuple(t) for t in a.tuples]
    if len(set(a.tuples)) != len(a.tuples):
        raise ValueError("tuples must be unique")
    if a.noise_std < 0 or a.observations < 1 or a.iterations < 1 or a.lbfgs_maxiter < 1:
        raise ValueError("invalid noise, observation, iteration or L-BFGS settings")
    if any(m not in METHODS for m in a.methods) or len(set(a.methods)) != len(a.methods):
        raise ValueError(f"methods must be unique names from {METHODS}")
    if any(s < 1 or s >= a.iterations for s in a.adam_snapshots):
        raise ValueError("snapshots must lie strictly inside the Adam stage")
    if a.init_from and a.adam_snapshots:
        raise ValueError("snapshots need the Adam stage; not with --init-from")
    return a


def load_checkpoint(directory, network, layout, noise, args):
    """Load a validated Adam checkpoint, refusing any mismatch with this run's data."""
    import torch
    ckpt = torch.load(Path(directory) / f"seed{network}_adam.pt", map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    ck_network = cfg.get("network_seed") if cfg.get("network_seed") is not None else ckpt["seed"]
    ck_layout = (cfg.get("observation_seed") if cfg.get("observation_seed") is not None
                 else ckpt["seed"])
    problems = []
    if ckpt.get("stage") != "adam":
        problems.append("not an Adam-stage checkpoint")
    if (ck_network, ck_layout) != (network, layout):
        problems.append(f"checkpoint seeds {(ck_network, ck_layout)} != {(network, layout)}")
    if cfg.get("observations") != args.observations or cfg.get("noise_std") != args.noise_std:
        problems.append("observation count or noise level differs")
    if args.noise_std > 0:
        problems.append("validated checkpoints are noise-free only")
    if cfg.get("iterations") != args.iterations:
        problems.append("Adam iteration count differs")
    if problems:
        raise ValueError("refusing checkpoint: " + "; ".join(problems))
    return ckpt


def main(argv=None):
    args = parse_args(argv)
    require_remote_cuda()
    import deepxde as dde
    import torch
    if args.float64:
        dde.config.set_default_float("float64")
    grid = evaluation_grid()
    p_reference = exact_pressure(grid)
    k_reference = np.exp(exact_log_permeability(grid))
    theta_true = np.array([ALPHA_TRUE, BETA_TRUE])

    def evaluate(model, alpha, beta):
        a_est, b_est = float(alpha.detach().cpu()), float(beta.detach().cpu())
        k_est = np.exp(exact_log_permeability(grid, alpha=a_est, beta=b_est))
        return {"alpha_estimated": a_est, "beta_estimated": b_est,
                "parameter_relative_error": float(np.linalg.norm([a_est, b_est] - theta_true)
                                                  / np.linalg.norm(theta_true)),
                "pressure_relative_l2": relative_l2(model.predict(grid), p_reference),
                "permeability_relative_l2": relative_l2(k_est, k_reference)}

    def dtype_report(model, alpha):
        _, losses = model.outputs_losses_train(model.train_state.X_train,
                                               model.train_state.y_train,
                                               model.train_state.train_aux_vars)
        return {"network": str(next(model.net.parameters()).dtype),
                "external_variables": str(alpha.dtype),
                "inputs": str(np.asarray(model.train_state.X_train).dtype),
                "losses": str(losses.dtype), "torch_default": str(torch.get_default_dtype())}

    def fit(index):
        network, layout, noise = args.tuples[index]
        reset_peak_memory()
        model, alpha, beta = build_model(seed=network, observation_seed=layout,
                                         observations=args.observations,
                                         noise_std=args.noise_std, noise_seed=noise)
        record = {"seed": index, "network_seed": network, "layout_seed": layout,
                  "noise_seed": noise, "loss_counts": list(loss_counts(model))}
        snapshots = {}
        if args.init_from:
            ckpt = load_checkpoint(args.init_from, network, layout, noise, args)
            model.net.load_state_dict(ckpt["net"])
            with torch.no_grad():
                alpha.fill_(ckpt["alpha"])
                beta.fill_(ckpt["beta"])
            model.train_state.set_data_train(*model.data.train_next_batch(model.batch_size))
            record["adam"] = {"source": f"checkpoint {args.init_from}", "seconds": 0.0,
                              **evaluate(model, alpha, beta)}
            snapshots[args.iterations] = (copy.deepcopy(model.net.state_dict()),
                                          ckpt["alpha"], ckpt["beta"], 0.0)
        else:
            trace = parameter_trace({"alpha": alpha, "beta": beta}, period=args.trace_every)

            class Snapshot(dde.callbacks.Callback):
                def on_epoch_end(self):
                    it = int(self.model.train_state.iteration)
                    if it in args.adam_snapshots:
                        synchronize()
                        snapshots[it] = (copy.deepcopy(self.model.net.state_dict()),
                                         float(alpha.detach()), float(beta.detach()),
                                         time.perf_counter() - start)

            synchronize()
            start = time.perf_counter()
            model.train(iterations=args.iterations,
                        display_every=max(1, args.iterations // 10),
                        callbacks=[trace, Snapshot()])
            synchronize()
            adam_seconds = time.perf_counter() - start
            snapshots[args.iterations] = (copy.deepcopy(model.net.state_dict()),
                                          float(alpha.detach()), float(beta.detach()),
                                          adam_seconds)
            record["adam"] = {"source": "trained", "seconds": adam_seconds,
                              **evaluate(model, alpha, beta),
                              "final_loss_train": [float(v) for v in model.train_state.loss_train],
                              "parameter_trace": trace.trace, "loss_history": loss_history(model)}
            if args.save_state:
                directory = Path(args.save_state)
                directory.mkdir(parents=True, exist_ok=True)
                torch.save({"net": model.net.state_dict(), "alpha": snapshots[args.iterations][1],
                            "beta": snapshots[args.iterations][2], "stage": "adam",
                            "tuple": [network, layout, noise], "config": config},
                           directory / f"tuple{network}_{layout}_{noise}_adam.pt")
        record["adam"]["peak_memory_bytes"] = peak_memory()
        record["dtypes"] = dtype_report(model, alpha)

        from src.deepxde_lbfgs import refine_lbfgs
        record["refinements"] = []
        finite = []
        for budget in sorted(snapshots):
            state, a0, b0, adam_time = snapshots[budget]
            for method in args.methods:
                model.net.load_state_dict(state)
                with torch.no_grad():
                    alpha.fill_(a0)
                    beta.fill_(b0)
                weights, prior, sigma_used = method_objective(method, args.noise_std,
                                                              record["loss_counts"])
                reset_peak_memory()
                synchronize()
                t0 = time.perf_counter()
                lbfgs = refine_lbfgs(model, {"alpha": alpha, "beta": beta}, args.lbfgs_maxiter,
                                     record_every=args.lbfgs_record_every, loss_weights=weights,
                                     penalty=prior_penalty([alpha, beta], prior))
                synchronize()
                seconds = time.perf_counter() - t0
                entry = {"adam_iterations": budget, "method": method, "assumed_sigma": sigma_used,
                         "prior_coefficient": prior, "adam_seconds": adam_time,
                         "lbfgs_seconds": seconds, "total_seconds": adam_time + seconds,
                         "peak_memory_bytes": peak_memory(),
                         **evaluate(model, alpha, beta), "lbfgs": lbfgs}
                record["refinements"].append(entry)
                key = f"{budget}:{method}"
                record[f"check:{key}"] = entry["parameter_relative_error"]
                finite.append(f"check:{key}")
        record["finite_check"] = finite
        return record

    config = {"benchmark": "spatial_k_noise_study", "noise_std": args.noise_std,
              "noise_kind": "absolute i.i.d. Gaussian; independent stream default_rng([noise_seed, 1])",
              "observations": args.observations, "iterations": args.iterations,
              "adam_snapshots": args.adam_snapshots, "methods": args.methods,
              "tau": TAU, "lbfgs_maxiter": args.lbfgs_maxiter, "float64": args.float64,
              "init_from": args.init_from, "tuples": [list(t) for t in args.tuples],
              "evaluation_grid": "41x41 uniform on [0,1]^2", "trace_every": args.trace_every}
    records = []
    for index in range(len(args.tuples)):
        records.append(run_seed(index, fit))
        write_results(args.output, config, records)
    failed = failed_seeds(records)
    print(f"Wrote {len(records)} records to {args.output}; failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
