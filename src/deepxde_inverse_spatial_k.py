"""Manufactured spatially varying Darcy inverse-PINN benchmark.

p(x,y)=sin(pi*x)sin(pi*y); k(x,y)=exp(alpha*x+beta*y).
Unknown alpha,beta are inferred from pressure observations.
This standalone parametric inverse benchmark is not the original FEM KL field.
"""
import numpy as np


def exact_pressure(x):
    x = np.asarray(x)
    return np.sin(np.pi*x[:, 0:1])*np.sin(np.pi*x[:, 1:2])


def exact_log_permeability(x, alpha=0.5, beta=-0.3):
    x = np.asarray(x)
    return alpha*x[:, 0:1] + beta*x[:, 1:2]


def manufactured_forcing(x, alpha=0.5, beta=-0.3):
    x = np.asarray(x)
    px = np.pi*np.cos(np.pi*x[:, 0:1])*np.sin(np.pi*x[:, 1:2])
    py = np.pi*np.sin(np.pi*x[:, 0:1])*np.cos(np.pi*x[:, 1:2])
    p = exact_pressure(x)
    k = np.exp(exact_log_permeability(x, alpha, beta))
    return k*(2*np.pi**2*p-alpha*px-beta*py)


def build_model(alpha_true=0.5, beta_true=-0.3, seed=42, observations=128):
    import deepxde as dde
    import torch

    dde.config.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    geometry = dde.geometry.Rectangle([0,0],[1,1])
    alpha = dde.Variable(0.0)
    beta = dde.Variable(0.0)

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

    xy = rng.uniform(0.05,0.95,(observations,2))
    bc = dde.icbc.DirichletBC(geometry,lambda _:0.0,lambda _,on_boundary:on_boundary)
    obs = dde.icbc.PointSetBC(xy,exact_pressure(xy))
    data = dde.data.PDE(geometry,pde,[bc,obs],num_domain=512,
                        num_boundary=64,num_test=512)
    net = dde.nn.FNN([2,64,64,64,1],"tanh","Glorot normal")
    model = dde.Model(data,net)
    model.compile("adam",lr=1e-3,external_trainable_variables=[alpha,beta])
    return model,alpha,beta


def train(iterations=10000,**kwargs):
    import torch
    model,alpha,beta=build_model(**kwargs)
    model.train(iterations=iterations,display_every=max(1,iterations//10))
    return model,{"alpha":float(alpha.detach().cpu()),"beta":float(beta.detach().cpu())}
