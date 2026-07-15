"""Smoke tests for avalanche module."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.grid import Grid1D
from src.core.layer import Layer
from src.core.material import Material
from src.core.constants import q
from src.avalanche.ionization import OkutoCrowellModel, IonizationCoefficients
from src.avalanche.trigger import TriggerSolver
from src.avalanche.tunneling import TunnelingModel
from src.avalanche.dark_current import DarkCurrentModel
from src.avalanche.pde import PDEModel
from src.avalanche.afterpulsing import AfterpulsingModel
from src.avalanche.excess_noise import ExcessNoiseFactor


def test_okuto_crowell(inp_material):
    model = OkutoCrowellModel()
    mat = inp_material
    assert model.alpha(5e5, mat, 300.0) > 0
    assert model.beta(5e5, mat, 300.0) > 0
    assert model.alpha(1e3, mat, 300.0) == pytest.approx(0.0, abs=1e-20)
    assert model.alpha(5e5, mat, 300.0) > model.alpha(5e5, mat, 400.0)


def test_ionization_coefficients(inp_material):
    mat = inp_material
    model = OkutoCrowellModel()
    x = np.linspace(0, 7e-4, 500)
    Eg = np.full(500, 1.34 * q)
    Eth = np.full(500, 2.1 * q)
    mc = np.full(500, 0.077)
    mh = np.full(500, 0.64)
    E_ie = np.full(500, 2.1 * q)
    E_ih = np.full(500, 2.1 * q)
    mat_names = np.array(["InP"] * 500)
    materials = {"InP": mat}

    ic = IonizationCoefficients(
        model, materials,
        Eg_grid=Eg, Eth_grid=Eth, mc_grid=mc, mh_grid=mh,
        E_ie_grid=E_ie, E_ih_grid=E_ih,
        grid_x=x, mat_names=mat_names, T=300.0,
    )
    E = np.full(500, 5e5)
    alpha = ic.alpha(E)
    beta = ic.beta(E)
    assert alpha.shape == (500,)
    assert np.all(alpha >= 0)
    assert np.all(beta >= 0)
    assert np.max(alpha) > 0


def test_trigger_solver():
    grid = Grid1D(L=7.3e-4, N=500)
    solver = TriggerSolver(grid)
    E = np.full(500, 5e5)
    alpha = np.full(500, 1e4)
    beta = np.full(500, 5e3)
    x = grid.x
    Pe, Ph = solver.solve(E, alpha, beta, x)
    assert Pe.shape == (500,)
    assert Ph.shape == (500,)
    assert np.all(Pe >= 0)
    assert np.all(Pe <= 1)
    assert np.all(Ph >= 0)
    assert np.all(Ph <= 1)


def test_tunneling_model():
    model = TunnelingModel(T=300.0, N_T=1e12, Eg_mulp=1.35, mc_mulp=0.041, mh_mulp=0.4)
    E = np.full(500, 5e5)
    J_btbt = model.btbt_current(E)
    assert J_btbt.shape == (500,)
    assert np.all(J_btbt >= 0)

    # Hurkx TAT enhancement factor
    gamma = TunnelingModel.hurkx_gamma(E, m_eff=0.08, E_barrier_eV=0.334, T=300.0)
    assert gamma.shape == (500,)
    assert np.all(gamma >= 0)
    assert np.all(gamma <= 1e6)

    # TAT current uses the new Hurkx model
    J_tat = model.tat_current(E, Eg_mulp=1.35, mc_mulp=0.08, mh_mulp=0.86,
                              ni_val=1.3e10, tau_val=1e-9)
    assert J_tat.shape == (500,)
    assert np.all(J_tat >= 0)


def test_hurkx_gamma():
    """Verify Hurkx enhancement factor behavior."""
    F = np.array([0.0, 1e4, 5e5, 1e6, 2e6])
    gamma = TunnelingModel.hurkx_gamma(F, m_eff=0.08, E_barrier_eV=0.334, T=300.0)
    assert gamma[0] == 0.0  # Zero field → no enhancement
    assert gamma[2] > 0.0   # Typical operating field → positive
    assert gamma[3] > gamma[2]  # Higher field → larger enhancement
    assert gamma[4] > gamma[3]
    assert gamma[4] <= 1e20  # Capped


def test_dark_current_model():
    x = np.linspace(0, 7e-4, 500)
    ni_arr = np.full(500, 1e10)
    Eg_arr = np.full(500, 1.34 * q)
    mc_arr = np.full(500, 0.077)
    mh_arr = np.full(500, 0.64)

    mat_name_grid = np.where(x > 5e-4, "InGaAs", "InP")

    model = DarkCurrentModel(
        T=300.0,
        grid_x=x,
        N_T=1e12,
        mat_name_grid=mat_name_grid,
        ni_absorber=1e10,
        tau_n_absorber=1e-6,
        tau_p_absorber=1e-6,
        Eg_mulp=1.35,
        mc_mulp=0.041,
        mh_mulp=0.4,
    )
    F = np.full(500, 5e5)
    J = model.total_dark_current_density(x, F, ni_arr, Eg_arr, mc_arr, mh_arr)
    assert J.shape == (500,)

    G = model.generation_rate(x, F, ni_arr, Eg_arr, mc_arr, mh_arr)
    assert G.shape == (500,)

    dcr = model.compute_dcr(x, F, np.ones(500), ni_arr, Eg_arr, mc_arr, mh_arr)
    assert dcr >= 0


def test_pdp_model(inp_material):
    mat_inp = inp_material
    materials = {"InGaAs": mat_inp, "InP": mat_inp, "InGaAsP": mat_inp}
    pde = PDEModel(materials=materials, reflectivity=0.1)
    layers = [
        Layer(thickness=2.5e-4, doping_type="acceptor", doping_A=2e18, doping_m=0, material="InP"),
        Layer(thickness=0.5e-4, doping_type="donor", doping_A=0, doping_m=0, material="InP"),
        Layer(thickness=0.2e-4, doping_type="donor", doping_A=1e17, doping_m=0, material="InP"),
        Layer(thickness=0.12e-4, doping_type="donor", doping_A=0, doping_m=0, material="InGaAsP"),
        Layer(thickness=1.5e-4, doping_type="donor", doping_A=0, doping_m=0, material="InGaAs"),
    ]
    dead_zone, absorber = pde.find_absorber(layers, "InGaAs")
    assert len(dead_zone) == 4
    assert absorber.thickness == 1.5e-4

    x = np.linspace(0, absorber.thickness, 100)
    Ptr = np.ones(100) * 0.9
    pde_val = pde.pde_integral(1550e-9, x, Ptr, x[1] - x[0])
    assert pde_val >= 0
    assert pde_val <= 1


def test_afterpulsing():
    ap = AfterpulsingModel(N_T=1e12, tau_c=1e-6)
    assert ap.afterpulsing_probability(0.0) == pytest.approx(0.0)
    assert ap.afterpulsing_probability(1e-3) > 0.0
    assert ap.holdoff_optimal(0.5) > 0
    assert ap.effective_dcr(1e6, 1e-6) >= 1e6


def test_afterpulsing_sweep():
    ap = AfterpulsingModel(N_T=1e6, tau_c=1e-6)
    holdoffs = np.logspace(-9, -4, 20)
    P_ap = np.array([ap.afterpulsing_probability(t) for t in holdoffs])
    assert np.all(np.diff(P_ap) >= 0)
    assert P_ap[0] < 0.01
    assert P_ap[-1] > P_ap[0]


def test_excess_noise():
    en = ExcessNoiseFactor(k_eff=0.5)
    assert en.f(1.0) == pytest.approx(1.0)
    assert en.f(10.0) == pytest.approx(5.95)
    assert en.f(1.0) >= 1.0

    en2 = ExcessNoiseFactor.from_ionization(1e4, 5e3)
    assert en2.k_eff == pytest.approx(0.5)


def test_excess_noise_sweep():
    en = ExcessNoiseFactor(k_eff=0.3)
    M_vals = np.linspace(1, 50, 20)
    F_vals = en.f(M_vals)
    assert np.all(np.diff(F_vals) >= 0)
    assert F_vals[0] == pytest.approx(1.0)
    assert np.all(F_vals[1:] > 1.0)
