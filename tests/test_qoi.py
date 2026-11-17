"""Tests for the effective-permeability quantity of interest (src/qoi.py)."""
import numpy as np
import pytest

from src.forward import build_mesh
from src.qoi import effective_permeability


def test_homogeneous_medium_gives_its_permeability():
    dm = build_mesh(8)
    n = dm.V_m.dofmap.index_map.size_global
    for c in (-0.7, 0.0, 1.2):
        assert effective_permeability(dm, np.full(n, c)) == pytest.approx(np.exp(c), rel=1e-10)


def test_constant_shift_scales_flux_exactly():
    dm = build_mesh(8)
    m = 0.5 * np.random.default_rng(0).standard_normal(dm.V_m.dofmap.index_map.size_global)
    q0 = effective_permeability(dm, m)
    assert effective_permeability(dm, m + 0.8) == pytest.approx(np.exp(0.8) * q0, rel=1e-10)


def test_layered_medium_matches_series_and_parallel_bounds():
    """Effective permeability lies between the harmonic and arithmetic means;
    for layers varying only in y (parallel to the flow) it equals the arithmetic mean."""
    dm = build_mesh(16)
    x = dm.V_m.tabulate_dof_coordinates()
    m = 0.6 * np.cos(np.pi * x[:, 1])
    q = effective_permeability(dm, m)
    k = np.exp(0.6 * np.cos(np.pi * np.linspace(0, 1, 20001)))
    assert q == pytest.approx(k.mean(), rel=5e-3)
    m2 = 0.6 * np.cos(np.pi * x[:, 0])  # series layers: harmonic mean
    assert effective_permeability(dm, m2) == pytest.approx(1 / (1 / k).mean(), rel=5e-3)
