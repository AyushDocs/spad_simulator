"""Smoke tests for core module."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.constants import q, kB, eps0, VT, thermal_energy
from src.core.grid import Grid1D
from src.core.layer import Layer
from src.core.doping import DopingProfile
from src.core.material import Material
from src.core.material_grid import MaterialGrid
from src.core.device import Device
from src.core.absorption import InterpolatedAbsorption
from src.utils.loaders import MaterialData, AbsorptionData


def test_constants():
    assert q == pytest.approx(1.602176634e-19)
    assert kB == pytest.approx(1.380649e-23)
    assert eps0 == pytest.approx(8.854187817e-14)
    assert VT(300.0) == pytest.approx(kB * 300.0 / q)
    assert thermal_energy(300.0) == pytest.approx(kB * 300.0)


def test_grid1d():
    grid = Grid1D(L=7.3e-4, N=100)
    assert grid.no_of_nodes == 100
    assert grid.x[0] == pytest.approx(0.0)
    assert grid.x[-1] == pytest.approx(grid.L)
    assert grid.dx == pytest.approx(grid.L / 99)

    phi = grid.x ** 2
    grad = grid.gradient(phi)
    assert len(grad) == 100
    # gradient() returns -d(phi)/dx, so grad[50] ≈ -2*x[50]
    assert abs(grad[50]) == pytest.approx(2 * grid.x[50], rel=0.05)


def test_layer():
    lyr = Layer(thickness=1e-4, doping_type="donor", doping_A=1e16,
                doping_m=0, material="InP")
    assert lyr.thickness == 1e-4
    assert lyr.is_donor is True
    assert lyr.material == "InP"


def test_doping_profile():
    layers = [
        Layer(1e-4, "acceptor", 1e18, 0, material="InP"),
        Layer(1e-4, "donor", 0, 0, material="InP"),
        Layer(1e-4, "donor", 1e18, 0, material="InP"),
    ]
    grid = Grid1D(L=3e-4, N=60)
    dp = DopingProfile._from_layers(layers, grid)
    net = dp.net_doping(grid.x)
    assert len(net) == 60
    assert net[0] < 0
    assert net[-1] > 0


def _make_absorption():
    wl = np.linspace(400e-9, 2000e-9, 50)
    alpha = np.where(wl < 920e-9, 5000.0,
                     np.where(wl < 1650e-9, 4000.0 * np.exp(-(wl - 920e-9) / 500e-9), 10.0))
    return AbsorptionData(material="InP", wavelengths=wl, alphas=alpha)


def test_material():
    data = MaterialData(
        name="InP", eps_r=12.5, mu_n=5400, mu_p=2000,
        vsat_n=1e7, vsat_p=1e7, mc=0.077, mh=0.64,
        tau_n=1e-6, tau_p=1e-6, Eg_0K=1.42,
        varshni_alpha=4.9e-4, varshni_beta=327,
        Nc_300K=5.7e17, Nv_300K=1.1e19, dos_gamma=1.5,
        ionization_e={"Eth": 2.1, "lambda0": 4e-7, "ER0": 3.5e-2, "hw_meV": 42},
        ionization_h={"Eth": 2.1, "lambda0": 4e-7, "ER0": 3.5e-2, "hw_meV": 42},
    )
    abs_data = _make_absorption()
    mat = Material(data, absorption=InterpolatedAbsorption(abs_data), T=300.0)
    assert mat.name == "InP"
    assert mat.eps_r == 12.5
    assert mat.Eg() > 1.0
    assert mat.ni() > 0
    assert mat.absorption_coefficient(1550e-9) >= 0
