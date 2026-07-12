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
from src.core.units import Q


def _make_device(absorption_data):
    layers = [
        Layer(thickness=2.5e-4, doping_type="acceptor", doping_A=2e18, doping_m=0, material="InP"),
        Layer(thickness=0.5e-4, doping_type="donor", doping_A=0, doping_m=0, material="InP"),
        Layer(thickness=0.2e-4, doping_type="donor", doping_A=1e17, doping_m=0, material="InP"),
        Layer(thickness=0.12e-4, doping_type="donor", doping_A=0, doping_m=0, material="InGaAsP"),
        Layer(thickness=1.5e-4, doping_type="donor", doping_A=0, doping_m=0, material="InGaAs"),
        Layer(thickness=0.5e-4, doping_type="donor", doping_A=5e16, doping_m=0, material="InP"),
        Layer(thickness=2.0e-4, doping_type="donor", doping_A=1e18, doping_m=0, material="InP"),
    ]
    grid = Grid1D(L=sum(l.thickness for l in layers), N=500)
    doping = DopingProfile._from_layers(layers)
    data = MaterialData(
        name="InP",
        eps_r=Q(12.5, "1"),
        mu_n=Q(5400, "cm**2/(V*s)"),
        mu_p=Q(2000, "cm**2/(V*s)"),
        vsat_n=Q(1e7, "cm/s"),
        vsat_p=Q(1e7, "cm/s"),
        mc=Q(0.077, "m0"),
        mh=Q(0.64, "m0"),
        tau_n=Q(1e-6, "s"),
        tau_p=Q(1e-6, "s"),
        Eg_0K=Q(1.42, "eV"),
        varshni_alpha=Q(4.9e-4, "eV/K"),
        varshni_beta=Q(327, "K"),
        Nc_300K=Q(5.7e17, "cm**-3"),
        Nv_300K=Q(1.1e19, "cm**-3"),
        dos_gamma=Q(1.5, "1"),
        ionization_e={"Eth": Q(2.1, "eV"), "lambda0": Q(4e-7, "cm"), "ER0": Q(3.5e-2, "eV"), "hw_meV": Q(42, "meV")},
        ionization_h={"Eth": Q(2.1, "eV"), "lambda0": Q(4e-7, "cm"), "ER0": Q(3.5e-2, "eV"), "hw_meV": Q(42, "meV")},
    )
    mat = Material(data, absorption=InterpolatedAbsorption(absorption_data), T=300.0)
    eps_grid = np.full(grid.no_of_nodes, mat.eps_r * 8.854187817e-14)
    ni_grid = np.full(grid.no_of_nodes, mat.ni())
    return grid, doping, eps_grid, ni_grid


def test_poisson_solver_converges(absorption_data):
    grid, doping, eps_grid, ni_grid = _make_device(absorption_data)
    solver = PoissonSolver(grid=grid, T=300.0, doping=doping,
                           eps_grid=eps_grid, ni_grid=ni_grid)
    phi, info = solver.solve(Vbias=0.0)
    assert info["converged"] is True
    assert len(phi) == grid.no_of_nodes
    assert np.all(np.isfinite(phi))


def test_poisson_solver_biased(absorption_data):
    grid, doping, eps_grid, ni_grid = _make_device(absorption_data)
    solver = PoissonSolver(grid=grid, T=300.0, doping=doping,
                           eps_grid=eps_grid, ni_grid=ni_grid)
    phi = None
    for V in np.linspace(0.0, 20.0, 11):
        phi, info = solver.solve(V, guess=phi)
    assert info["converged"] is True
    assert np.all(np.isfinite(phi))
    assert np.max(np.abs(phi)) > 0
