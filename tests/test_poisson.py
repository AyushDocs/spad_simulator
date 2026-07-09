"""Smoke tests for Poisson solver."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.grid import Grid1D
from src.core.layer import Layer
from src.core.doping import DopingProfile
from src.core.material import Material
from src.core.constants import q, VT
from src.core.absorption import InterpolatedAbsorption
from src.utils.loaders import MaterialData, AbsorptionData
from src.poisson.solver import PoissonSolver


def _make_device(absorption_data):
    layers = [
        Layer(2.5e-4, "acceptor", 2e18, 0, material="InP"),
        Layer(0.5e-4, "donor", 0, 0, material="InP"),
        Layer(0.2e-4, "donor", 1e17, 0, material="InP"),
        Layer(0.12e-4, "donor", 0, 0, material="InGaAsP"),
        Layer(1.5e-4, "donor", 0, 0, material="InGaAs"),
        Layer(0.5e-4, "donor", 5e16, 0, material="InP"),
        Layer(2.0e-4, "donor", 1e18, 0, material="InP"),
    ]
    grid = Grid1D(L=sum(l.thickness for l in layers), N=500)
    doping = DopingProfile._from_layers(layers, grid)
    data = MaterialData(
        name="InP", eps_r=12.5, mu_n=5400, mu_p=2000,
        vsat_n=1e7, vsat_p=1e7, mc=0.077, mh=0.64,
        tau_n=1e-6, tau_p=1e-6, Eg_0K=1.42,
        varshni_alpha=4.9e-4, varshni_beta=327,
        Nc_300K=5.7e17, Nv_300K=1.1e19, dos_gamma=1.5,
        ionization_e={"Eth": 2.1, "lambda0": 4e-7, "ER0": 3.5e-2, "hw_meV": 42},
        ionization_h={"Eth": 2.1, "lambda0": 4e-7, "ER0": 3.5e-2, "hw_meV": 42},
    )
    mat = Material(data, absorption=InterpolatedAbsorption(absorption_data), T=300.0)
    eps_grid = np.full(grid.no_of_nodes, mat.eps_r * 8.854187817e-14)
    ni_grid = np.full(grid.no_of_nodes, mat.ni())
    return grid, doping, eps_grid, ni_grid


def test_poisson_solver_converges(absorption_data):
    grid, doping, eps_grid, ni_grid = _make_device(absorption_data)
    solver = PoissonSolver(grid, T=300.0, doping=doping,
                           eps_grid=eps_grid, ni_grid=ni_grid)
    phi, info = solver.solve(Vbias=0.0)
    assert info["converged"] is True
    assert len(phi) == grid.no_of_nodes
    assert np.all(np.isfinite(phi))


def test_poisson_solver_biased(absorption_data):
    grid, doping, eps_grid, ni_grid = _make_device(absorption_data)
    solver = PoissonSolver(grid, T=300.0, doping=doping,
                           eps_grid=eps_grid, ni_grid=ni_grid)
    phi = None
    for V in np.linspace(0.0, 20.0, 11):
        phi, info = solver.solve(V, guess=phi)
    assert info["converged"] is True
    assert np.all(np.isfinite(phi))
    assert np.max(np.abs(phi)) > 0
