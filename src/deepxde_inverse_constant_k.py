"""DeepXDE inverse Darcy coefficient identification (manufactured 2D test).

This is a separately identified constant-permeability calibration benchmark,
NOT a replacement for the spatial log-permeability FEM/adjoint inverse study.
PDE: -div(k grad p)=f, k>0 constant; p=sin(pi*x)sin(pi*y).
"""
import numpy as np


def exact_pressure(x):
    x = np.asarray(x)
    return np.sin(np.pi*x[:, 0:1]) * np.sin(np.pi*x[:, 1:2])


def build_inverse_model(k_true=1.7, k_initial=0.8, seed=42, n_observations=100):
    import deepxde as dde
    import torch

    if k_true <= 0 or k_initial <= 0:
        raise ValueError("permeability must be positive")
    dde.config.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    geometry = dde.geometry.Rectangle([0, 0], [1, 1])
    log_k = dde.Variable(float(np.log(k_initial)))

    def pde(x, p):
        laplace = (dde.grad.hessian(p, x, i=0, j=0)
                   + dde.grad.hessian(p, x, i=1, j=1))
        forcing = (2 * np.pi**2 * k_true * torch.sin(np.pi*x[:, 0:1])
                   * torch.sin(np.pi*x[:, 1:2]))
        return -torch.exp(log_k) * laplace - forcing

    bc = dde.icbc.DirichletBC(geometry, lambda _: 0.0,
                              lambda _, on_boundary: on_boundary)
    xy = rng.uniform(0.05, 0.95, size=(n_observations, 2))
    observed = dde.icbc.PointSetBC(xy, exact_pressure(xy), component=0)
    data = dde.data.PDE(geometry, pde, [bc, observed],
                        num_domain=512, num_boundary=64, num_test=512)
    net = dde.nn.FNN([2, 64, 64, 64, 1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, external_trainable_variables=[log_k])
    return model, log_k


def train(iterations=10000, **kwargs):
    import torch

    model, log_k = build_inverse_model(**kwargs)
    model.train(iterations=iterations, display_every=max(1, iterations // 10))
    return model, float(torch.exp(log_k).detach().cpu().item())
