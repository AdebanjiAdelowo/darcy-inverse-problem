"""Optional L-BFGS refinement stage for the DeepXDE inverse benchmarks.

Explicitly separate configuration ("adam+lbfgs"): the default training method
of every benchmark stays Adam only. The refinement recompiles an already
Adam-trained model with DeepXDE's PyTorch L-BFGS (``torch.optim.LBFGS`` with
a strong-Wolfe line search) over the same loss, data and trainable variables.

Every function evaluation is logged by wrapping the closure passed to the
optimiser; the wrapper only reads values and does not change the optimisation.
"""
import math

DEFAULT_LBFGS = {"maxcor": 100, "ftol": 0.0, "gtol": 1e-8, "maxls": 50}


def set_lbfgs_options(**options):
    """``dde.optimizers.set_LBFGS_options`` plus the per-step limits for PyTorch.

    DeepXDE 1.15 computes ``iter_per_step = min(1000, maxiter)`` and
    ``fun_per_step`` only once, at import, from the default options, so a later
    ``set_LBFGS_options(maxiter=...)`` is overshot to the next multiple of 1000
    iterations. This recomputes both with DeepXDE's own formula; for the default
    ``maxiter = 15000`` the values (1000, 1250) are unchanged.
    """
    import deepxde as dde
    from deepxde.optimizers.config import LBFGS_options

    dde.optimizers.set_LBFGS_options(**options)
    LBFGS_options["iter_per_step"] = min(1000, LBFGS_options["maxiter"])
    LBFGS_options["fun_per_step"] = (LBFGS_options["maxfun"] * LBFGS_options["iter_per_step"]
                                     // LBFGS_options["maxiter"])
    return dict(LBFGS_options)


def refine_lbfgs(model, variables, maxiter, record_every=10, loss_weights=None,
                 penalty=None, **options):
    """Run L-BFGS on ``model`` (already compiled and trained with Adam).

    ``variables`` maps names to the external ``dde.Variable`` objects. Returns
    a JSON-ready dict with the configuration, stopping information, function
    evaluation count and the per-evaluation history (every ``record_every``-th
    evaluation plus the last one).
    """
    import deepxde as dde
    import torch

    from src.deepxde_tracking import iteration_of, parameter_trace

    if maxiter < 1 or record_every < 1:
        raise ValueError("maxiter and record_every must be positive")
    settings = {**DEFAULT_LBFGS, **options}
    set_lbfgs_options(maxiter=maxiter, **settings)
    from deepxde.optimizers.config import LBFGS_options
    config = dict(LBFGS_options)

    start_iteration = iteration_of(model.train_state)
    model.compile("L-BFGS", loss_weights=loss_weights,
                  external_trainable_variables=list(variables.values()))
    evaluations = []
    original_step = model.opt.step

    def logged_step(closure):
        def logged_closure():
            loss = closure()
            if penalty is not None:
                extra = penalty()
                extra.backward()  # closure() already zeroed and filled the gradients
                loss = loss + extra.detach()
            evaluations.append(torch.stack(
                [loss.detach().reshape(())] + [v.detach().reshape(()) for v in variables.values()]))
            return loss
        return original_step(logged_closure)

    model.opt.step = logged_step
    trace = parameter_trace(variables, period=1)  # one sample per DeepXDE outer step
    model.train(display_every=1, callbacks=[trace], verbose=0)

    state = model.opt.state[model.opt.param_groups[0]["params"][0]]
    n_iter = int(state.get("n_iter", 0))
    # DeepXDE leaves its outer loop on convergence before callbacks and loss
    # updates run, so the last step is missing from the trace and from
    # train_state. Record the final point and recompute the loss components.
    final = [float(v.detach().cpu().item()) for v in variables.values()]
    if [trace.trace[name][-1] for name in variables] != final:
        trace.trace["epoch"].append(start_iteration + n_iter)
        for name, value in zip(variables, final):
            trace.trace[name].append(value)
    _, final_losses = model.outputs_losses_train(
        model.train_state.X_train, model.train_state.y_train, model.train_state.train_aux_vars)
    final_loss_components = [float(v) for v in final_losses.detach().cpu().reshape(-1)]
    final_penalty = float(penalty().detach()) if penalty is not None else 0.0
    # DeepXDE applies loss_weights inside outputs_losses; keep raw components too.
    raw_components = (final_loss_components if loss_weights is None else
                      [c / w for c, w in zip(final_loss_components, loss_weights)])
    func_evals = int(state.get("func_evals", 0))
    grad = state.get("prev_flat_grad")
    grad_max = float(grad.abs().max()) if grad is not None else math.nan
    history = torch.stack(evaluations).cpu().tolist() if evaluations else []
    keep = sorted(set(range(0, len(history), record_every)) | ({len(history) - 1} if history else set()))
    if n_iter >= config["maxiter"]:
        reason = "maxiter"
    elif func_evals >= config["maxfun"]:
        reason = "maxfun"
    elif grad_max <= config["gtol"]:
        reason = "gtol"
    else:
        reason = "stalled (no progress in an outer step: line search or tolerance_change)"
    losses = [row[0] for row in history]
    return {
        "optimizer": "torch.optim.LBFGS via DeepXDE (strong_wolfe)",
        "options": config,
        "lbfgs_iterations": n_iter,
        "function_evaluations": func_evals,
        "logged_evaluations": len(history),
        "stop_reason": reason,
        "final_max_abs_gradient": grad_max,
        "final_total_loss": losses[-1] if losses else math.nan,
        "final_loss_components": final_loss_components,
        "final_raw_loss_components": raw_components,
        "final_penalty": final_penalty,
        "loss_weights": loss_weights,
        "initial_total_loss": losses[0] if losses else math.nan,
        "finite_history": all(math.isfinite(v) for row in history for v in row),
        "start_iteration": start_iteration,
        "evaluation_history": {"columns": ["total_loss", *variables],
                               "every": record_every, "index": keep,
                               "values": [history[i] for i in keep]},
        "outer_step_trace": trace.trace,
    }
