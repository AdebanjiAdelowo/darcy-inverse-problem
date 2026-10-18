"""DeepXDE inverse Darcy coefficient identification (manufactured 2D test).

This is a separately identified constant-permeability calibration benchmark,
NOT a replacement for the spatial log-permeability FEM/adjoint inverse study.
PDE: -div(k grad p)=f, k>0 constant; p=sin(pi*x)sin(pi*y).
"""
import numpy as np

K_TRUE = 1.7


def exact_pressure(x):
    x = np.asarray(x)
    return np.sin(np.pi*x[:, 0:1]) * np.sin(np.pi*x[:, 1:2])


def observation_data(seed, n_observations, noise_std=0.0):
    """Sensor locations and (optionally noisy) pressure values.

    With ``noise_std == 0`` the draws are identical to the original protocol
    (one ``uniform`` call on ``default_rng(seed)``).
    """
    if n_observations < 1 or noise_std < 0:
        raise ValueError("n_observations must be positive and noise_std non-negative")
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.05, 0.95, size=(n_observations, 2))
    values = exact_pressure(xy)
    if noise_std > 0:
        values = values + rng.normal(0.0, noise_std, size=values.shape)
    return xy, values


def build_inverse_model(k_true=K_TRUE, k_initial=0.8, seed=42, n_observations=100,
                        noise_std=0.0, observation_seed=None):
    import deepxde as dde
    import torch

    if k_true <= 0 or k_initial <= 0:
        raise ValueError("permeability must be positive")
    dde.config.set_random_seed(seed)
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
    xy, values = observation_data(seed if observation_seed is None else observation_seed,
                                  n_observations, noise_std)
    observed = dde.icbc.PointSetBC(xy, values, component=0)
    data = dde.data.PDE(geometry, pde, [bc, observed],
                        num_domain=512, num_boundary=64, num_test=512)
    net = dde.nn.FNN([2, 64, 64, 64, 1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, external_trainable_variables=[log_k])
    return model, log_k


def train(iterations=10000, callbacks=None, **kwargs):
    import torch

    model, log_k = build_inverse_model(**kwargs)
    model.train(iterations=iterations, display_every=max(1, iterations // 10),
                callbacks=callbacks)
    return model, float(torch.exp(log_k).detach().cpu().item())
