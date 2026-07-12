"""Tests for SelfConsistentLoop, CircuitSolver, and ParticleMesh."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.self_consistent.circuit import CircuitSolver
from src.self_consistent.particle_mesh import ParticleMesh
from src.core.grid import Grid1D
from src.core.doping import DopingProfile
from src.core.material import Material
from src.core.layer import Layer
from src.self_consistent.particle_mesh import Carrier
from src.transport.drift_diffusion import DriftDiffusionSolver, DriftDiffusion


def _make_material():
    from src.core.absorption import InterpolatedAbsorption
    from src.utils.loaders import MaterialData, AbsorptionData
    from src.core.units import Q
    wl = np.linspace(400e-9, 2000e-9, 50)
    alpha = np.where(wl < 920e-9, 5000.0,
                     np.where(wl < 1650e-9, 4000.0, 10.0))
    abs_data = AbsorptionData(material="InP", wavelengths=wl, alphas=alpha)
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
    return Material(data, absorption=InterpolatedAbsorption(abs_data), T=300.0)


def _make_grid_and_doping():
    layers = [
        Layer(thickness=0.5e-4, doping_type="donor", doping_A=3e16, doping_m=0, doping_x0=0, material="InP"),
        Layer(thickness=1.8e-4, doping_type="acceptor", doping_A=1e14, doping_m=0, doping_x0=0, material="InP"),
        Layer(thickness=1.0e-4, doping_type="acceptor", doping_A=1e17, doping_m=0, doping_x0=0, material="InGaAs"),
    ]
    L = sum(ly.thickness for ly in layers)
    grid = Grid1D(L=L, N=300)
    doping = DopingProfile._from_layers(layers)
    return grid, doping


def test_circuit_solver_init():
    cs = CircuitSolver(Vbias=75.0, Rq=1e5, Cspad=1e-15, Vbr=70.0)
    assert cs.Vspad == 75.0
    assert not cs.is_quenched


def test_circuit_quench():
    cs = CircuitSolver(Vbias=75.0, Rq=1e5, Cspad=1e-15, Vbr=70.0)
    cs.update(1e-3, 1e-12)
    assert cs.Vspad < 75.0


def test_circuit_recharge():
    cs = CircuitSolver(Vbias=75.0, Rq=1e5, Cspad=1e-15, Vbr=70.0)
    V0 = 68.0
    V_recharge = cs.recharge_voltage(1e-10, V0)
    assert V_recharge > V0


def test_particle_mesh_deposit():
    grid, _ = _make_grid_and_doping()
    pm = ParticleMesh(grid)
    c = Carrier(x=grid.x[50], typ="electron")
    c.alive = True
    rho = pm.deposit_charge([c])
    assert np.any(rho != 0.0)
    assert np.isfinite(rho).all()


def test_particle_mesh_interpolate():
    grid, _ = _make_grid_and_doping()
    pm = ParticleMesh(grid)
    E_grid = np.ones(grid.no_of_nodes) * 1e5
    c = Carrier(x=grid.x[50], typ="electron")
    c.alive = True
    E_vals = pm.interpolate_field([c], E_grid)
    assert len(E_vals) == 1
    assert abs(E_vals[0] - 1e5) < 1.0


def test_self_consistent_loop_step():
    grid, doping = _make_grid_and_doping()
    mat = _make_material()
    pm = ParticleMesh(grid)
    transport = DriftDiffusion(mat)

    ion = MagicMock()
    ion.alpha.return_value = np.zeros(grid.no_of_nodes)
    ion.beta.return_value = np.zeros(grid.no_of_nodes)
    ion.dead_space_length.return_value = 1e-6

    ps = MagicMock()
    phi_guess = np.linspace(0, 0.01, grid.no_of_nodes)
    ps.solve.return_value = (phi_guess, {"converged": True, "iterations": 1})

    cs = CircuitSolver(Vbias=0.01, Rq=1e5, Cspad=1e-15, Vbr=0.005)

    loop = SelfConsistentLoop(grid, doping, ps, pm, transport, ion, cs, dt=1e-15)
    loop.inject_carrier(grid.x[50])
    assert len(loop.carriers) == 1
    info = loop.step()
    assert "t" in info
    assert "N" in info
    assert info["t"] > 0
    ps.solve.assert_called_once()


from src.self_consistent.loop import SelfConsistentLoop


def test_self_consistent_loop_run():
    grid, doping = _make_grid_and_doping()
    mat = _make_material()
    pm = ParticleMesh(grid)
    transport = DriftDiffusion(mat)

    ion = MagicMock()
    ion.alpha.return_value = np.zeros(grid.no_of_nodes)
    ion.beta.return_value = np.zeros(grid.no_of_nodes)
    ion.dead_space_length.return_value = 1e-6

    ps = MagicMock()
    phi_guess = np.linspace(0, 0.01, grid.no_of_nodes)
    ps.solve.return_value = (phi_guess, {"converged": True, "iterations": 1})

    cs = CircuitSolver(Vbias=0.01, Rq=1e5, Cspad=1e-15, Vbr=0.005)

    loop = SelfConsistentLoop(grid, doping, ps, pm, transport, ion, cs, dt=1e-15)
    loop.inject_carrier(grid.x[50])
    history = loop.run(5)
    assert len(history) == 5
    assert history[-1]["t"] > history[0]["t"]
    assert ps.solve.call_count == 5
