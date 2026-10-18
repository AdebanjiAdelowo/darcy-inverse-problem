"""Lightweight training diagnostics for the DeepXDE inverse benchmarks.

Records the trajectory of external trainable variables during training so
convergence behaviour can be inspected after a remote run. Observation only:
nothing here changes the optimisation.
"""


def parameter_trace(variables, transforms=None, period=100):
    """Return a DeepXDE callback that samples ``variables`` every ``period`` epochs.

    ``variables`` maps a name to a ``dde.Variable``; ``transforms`` optionally
    maps the same name to a function applied to the float value (for example
    ``math.exp`` to report k instead of log k). The samples are stored on the
    callback as ``trace = {"epoch": [...], name: [...]}``.
    """
    import deepxde as dde

    if period < 1:
        raise ValueError("period must be positive")
    transforms = transforms or {}

    def iteration(state):
        # DeepXDE >= 1.15 renamed TrainState.epoch to TrainState.iteration.
        return int(state.iteration if hasattr(state, "iteration") else state.epoch)

    class ParameterTrace(dde.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.trace = {"epoch": [], **{name: [] for name in variables}}

        def _record(self):
            self.trace["epoch"].append(iteration(self.model.train_state))
            for name, var in variables.items():
                value = float(var.detach().cpu().item())
                self.trace[name].append(transforms.get(name, lambda v: v)(value))

        def on_train_begin(self):
            if not self.trace["epoch"]:
                self._record()

        def on_epoch_end(self):
            if iteration(self.model.train_state) % period == 0:
                self._record()

        def on_train_end(self):
            if self.trace["epoch"][-1] != iteration(self.model.train_state):
                self._record()

    return ParameterTrace()


def loss_history(model):
    """Return the recorded DeepXDE loss history as plain lists."""
    history = model.losshistory
    return {
        "steps": [int(s) for s in history.steps],
        "loss_train": [[float(v) for v in row] for row in history.loss_train],
        "loss_test": [[float(v) for v in row] for row in history.loss_test],
    }
