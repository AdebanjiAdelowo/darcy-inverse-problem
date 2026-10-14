import numpy as np
from src.deepxde_inverse_spatial_k import exact_pressure, exact_log_permeability, manufactured_forcing


def test_pressure_zero_on_boundary():
    x=np.linspace(0,1,17)
    for points in (np.stack([x,np.zeros_like(x)],axis=1),
                   np.stack([np.ones_like(x),x],axis=1)):
        assert np.max(np.abs(exact_pressure(points)))<1e-12


def test_log_k_and_forcing_are_finite():
    xy=np.array([[.1,.2],[.4,.8],[.9,.6]])
    assert np.allclose(exact_log_permeability(xy),.5*xy[:,0:1]-.3*xy[:,1:2])
    assert np.isfinite(manufactured_forcing(xy)).all()


def test_manufactured_source_finite_difference():
    # Independent finite-difference divergence approximation.
    xy=np.array([[.3,.4],[.6,.7]])
    h=1e-5
    def k(z):
        return np.exp(exact_log_permeability(z))
    def flux(z,axis):
        d=np.zeros_like(z);d[:,axis]=h
        return k(z)*(exact_pressure(z+d)-exact_pressure(z-d))/(2*h)
    divergence=0
    for axis in (0,1):
        d=np.zeros_like(xy);d[:,axis]=h
        divergence += (flux(xy+d,axis)-flux(xy-d,axis))/(2*h)
    assert np.max(np.abs(-divergence-manufactured_forcing(xy)))<1e-6
