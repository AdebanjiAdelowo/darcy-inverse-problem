"""Manufactured spatially varying Darcy inverse-PINN benchmark.

p(x,y)=sin(pi*x)sin(pi*y); k(x,y)=exp(alpha*x+beta*y).
Unknown alpha,beta are inferred from pressure observations.
This standalone parametric inverse benchmark is not the original FEM KL field.
"""
import numpy as np

ALPHA_TRUE = 0.5
BETA_TRUE = -0.3


def exact_pressure(x):
    x = np.asarray(x)
    return np.sin(np.pi*x[:, 0:1])*np.sin(np.pi*x[:, 1:2])


def exact_log_permeability(x, alpha=ALPHA_TRUE, beta=BETA_TRUE):
    x = np.asarray(x)
    return alpha*x[:, 0:1] + beta*x[:, 1:2]


def manufactured_forcing(x, alpha=ALPHA_TRUE, beta=BETA_TRUE):
    x = np.asarray(x)
    px = np.pi*np.cos(np.pi*x[:, 0:1])*np.sin(np.pi*x[:, 1:2])
    py = np.pi*np.sin(np.pi*x[:, 0:1])*np.cos(np.pi*x[:, 1:2])
    p = exact_pressure(x)
    k = np.exp(exact_log_permeability(x, alpha, beta))
    return k*(2*np.pi**2*p-alpha*px-beta*py)


def noise_draws(noise_seed, observations):
    """Standard-normal noise from a stream independent of every layout stream.

    ``default_rng([noise_seed, 1])`` is seeded through a different SeedSequence
    entropy from ``default_rng(layout_seed)``, so it is independent of the sensor
    layout even when the two seeds are equal. Draws are prefix-nested: the first
    m draws for n > m observations equal the draws for m observations.
    """
    return np.random.default_rng([noise_seed, 1]).standard_normal((observations, 1))


def observation_data(seed, observations, noise_std=0.0, noise_seed=None):
    """Sensor locations and (optionally noisy) pressure values.

    Noise is absolute: ``noise_std`` is the standard deviation of i.i.d. Gaussian
    noise added to the pressure (whose maximum is 1). With ``noise_std == 0`` the
    draws are identical to the original protocol. With ``noise_seed=None`` (the
    original protocol) the noise continues the layout stream, so layout and noise
    realisation are coupled; with a ``noise_seed`` they are independent and the
    noise is ``noise_std * noise_draws(noise_seed, observations)``.
    """
    if observations < 1 or noise_std < 0:
        raise ValueError("observations must be positive and noise_std non-negative")
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.05, 0.95, (observations, 2))
    values = exact_pressure(xy)
    if noise_std > 0:
        if noise_seed is None:
            values = values + rng.normal(0.0, noise_std, size=values.shape)
        else:
            values = values + noise_std * noise_draws(noise_seed, observations)
    return xy, values


def build_model(alpha_true=ALPHA_TRUE, beta_true=BETA_TRUE, seed=42, observations=128,
                noise_std=0.0, observation_seed=None, alpha_initial=0.0,
                beta_initial=0.0, noise_seed=None):
    import deepxde as dde
    import torch

    dde.config.set_random_seed(seed)
    geometry = dde.geometry.Rectangle([0,0],[1,1])
    alpha = dde.Variable(float(alpha_initial))
    beta = dde.Variable(float(beta_initial))

    def pde(x, p):
        px = dde.grad.jacobian(p, x, j=0)
        py = dde.grad.jacobian(p, x, j=1)
        pxx = dde.grad.hessian(p, x, i=0, j=0)
        pyy = dde.grad.hessian(p, x, i=1, j=1)
        k = torch.exp(alpha*x[:,0:1]+beta*x[:,1:2])
        true_k = torch.exp(alpha_true*x[:,0:1]+beta_true*x[:,1:2])
        true_p = torch.sin(np.pi*x[:,0:1])*torch.sin(np.pi*x[:,1:2])
        true_px = np.pi*torch.cos(np.pi*x[:,0:1])*torch.sin(np.pi*x[:,1:2])
        true_py = np.pi*torch.sin(np.pi*x[:,0:1])*torch.cos(np.pi*x[:,1:2])
        source = true_k*(2*np.pi**2*true_p-alpha_true*true_px-beta_true*true_py)
        return -k*(pxx+pyy+alpha*px+beta*py)-source

    xy, values = observation_data(seed if observation_seed is None else observation_seed,
                                  observations, noise_std, noise_seed)
    bc = dde.icbc.DirichletBC(geometry,lambda _:0.0,lambda _,on_boundary:on_boundary)
    obs = dde.icbc.PointSetBC(xy,values)
    data = dde.data.PDE(geometry,pde,[bc,obs],num_domain=512,
                        num_boundary=64,num_test=512)
    net = dde.nn.FNN([2,64,64,64,1],"tanh","Glorot normal")
    model = dde.Model(data,net)
    model.compile("adam",lr=1e-3,external_trainable_variables=[alpha,beta])
    return model,alpha,beta


def train(iterations=10000,callbacks=None,**kwargs):
    model,alpha,beta=build_model(**kwargs)
    model.train(iterations=iterations,display_every=max(1,iterations//10),
                callbacks=callbacks)
    return model,{"alpha":float(alpha.detach().cpu()),"beta":float(beta.detach().cpu())}
