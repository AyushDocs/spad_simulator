"""Smoke tests for avalanche module."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.grid import Grid1D
from src.core.layer import Layer
from src.core.material import Material
from src.core.constants import q
from src.core.absorption import InterpolatedAbsorption
from src.utils.loaders import MaterialData, AbsorptionData
from src.avalanche.ionization import OkutoCrowellModel, IonizationCoefficients
from src.avalanche.trigger import TriggerSolver
from src.avalanche.tunneling import TunnelingModel
from src.avalanche.dark_current import DarkCurrentModel
from src.avalanche.pdp import PDPModel
from src.avalanche.afterpulsing import AfterpulsingModel
from src.avalanche.excess_noise import ExcessNoiseFactor


def _make_absorption():
    wl = np.linspace(400e-9, 2000e-9, 50)
    alpha = np.where(wl < 920e-9, 5000.0,
                     np.where(wl < 1650e-9, 4000.0 * np.exp(-(wl - 920e-9) / 500e-9), 10.0))
    return AbsorptionData(material="InP", wavelengths=wl, alphas=alpha)


def _make_material():
    data = MaterialData(
        name="InP", eps_r=12.5, mu_n=5400, mu_p=2000,
        vsat_n=1e7, vsat_p=1e7, mc=0.077, mh=0.64,
        tau_n=1e-6, tau_p=1e-6, Eg_0K=1.42,
        varshni_alpha=4.9e-4, varshni_beta=327,
        Nc_300K=5.7e17, Nv_300K=1.1e19, dos_gamma=1.5,
        ionization_e={"Eth": 2.1, "lambda0": 4e-7, "ER0": 3.5e-2, "hw_meV": 42},
        ionization_h={"Eth": 2.1, "lambda0": 4e-7, "ER0": 3.5e-2, "hw_meV": 42},
    )
    return Material(data, absorption=InterpolatedAbsorption(_make_absorption()), T=300.0)


def test_okuto_crowell():
    model = OkutoCrowellModel()
    mat = _make_material()
    assert model.alpha(5e5, mat, 300.0) > 0
    assert model.beta(5e5, mat, 300.0) > 0
    assert model.alpha(1e3, mat, 300.0) == pytest.approx(0.0, abs=1e-20)
    assert model.alpha(5e5, mat, 300.0) > model.alpha(5e5, mat, 400.0)


def test_ionization_coefficients():
    mat = _make_material()
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
    Eg_grid = np.full(500, 1.34 * q)
    model = TunnelingModel(T=300.0, N_T=1e12, Eg_grid=Eg_grid)
    E = np.full(500, 5e5)
    mc = np.full(500, 0.077)
    mh = np.full(500, 0.64)
    J_btbt = model.btbt_current(E, Eg_grid, mc, mh)
    J_tat = model.tat_current(E, Eg_grid, mc, mh)
    assert J_btbt.shape == (500,)
    assert J_tat.shape == (500,)
    assert np.all(J_btbt >= 0)
    assert np.all(J_tat >= 0)


def test_dark_current_model():
    Eg_grid = np.full(500, 1.34 * q)
    mc_grid = np.full(500, 0.077)
    mh_grid = np.full(500, 0.64)
    x = np.linspace(0, 7e-4, 500)
    tau_n = np.full(500, 1e-6)
    tau_p = np.full(500, 1e-6)
    ni = np.full(500, 1e10)

    model = DarkCurrentModel(
        T=300.0, Eg_grid=Eg_grid, mc_grid=mc_grid, mh_grid=mh_grid,
        grid_x=x, tau_n_grid=tau_n, tau_p_grid=tau_p,
    )
    F = np.full(500, 5e5)
    J = model.total_dark_current_density(x, F, ni, Eg_grid, mc_grid, mh_grid)
    assert J.shape == (500,)

    G = model.generation_rate(x, F, ni, Eg_grid, mc_grid, mh_grid)
    assert G.shape == (500,)

    dcr = model.compute_dcr(x, F, np.ones(500), ni, Eg_grid, mc_grid, mh_grid)
    assert dcr >= 0


def test_pdp_model():
    mat_inp = _make_material()
    materials = {"InGaAs": mat_inp, "InP": mat_inp, "InGaAsP": mat_inp}
    pdp = PDPModel(materials, reflectivity=0.1)
    layers = [
        Layer(2.5e-4, "acceptor", 2e18, 0, material="InP"),
        Layer(0.5e-4, "donor", 0, 0, material="InP"),
        Layer(0.2e-4, "donor", 1e17, 0, material="InP"),
        Layer(0.12e-4, "donor", 0, 0, material="InGaAsP"),
        Layer(1.5e-4, "donor", 0, 0, material="InGaAs"),
    ]
    dead_zone, absorber = pdp.find_absorber(layers, "InGaAs")
    assert len(dead_zone) == 4
    assert absorber.thickness == 1.5e-4

    trans = pdp.dead_zone_transmission(1550e-9, dead_zone)
    assert 0 <= trans <= 1

    x = np.linspace(0, absorber.thickness, 100)
    Ptr = np.ones(100) * 0.9
    pdp_val = pdp.pdp_integral(1550e-9, x, Ptr, trans, x[1] - x[0])
    assert pdp_val >= 0
    assert pdp_val <= 1


def test_afterpulsing():
    ap = AfterpulsingModel(N_T=1e12, tau_c=1e-6)
    # At t=0 (holdoff=0), probability is 0 (no time to trap)
    assert ap.afterpulsing_probability(0.0) == pytest.approx(0.0)
    # At large holdoff, probability saturates near 1 - exp(-N_T * tau_c) ≈ 1.0
    assert ap.afterpulsing_probability(1e-3) > 0.0
    assert ap.holdoff_optimal(0.5) > 0
    assert ap.effective_dcr(1e6, 1e-6) >= 1e6


def test_excess_noise():
    en = ExcessNoiseFactor(k_eff=0.5)
    # At M=1: F = k*1 + (1-k)*(2-1) = k + 1-k = 1
    assert en.f(1.0) == pytest.approx(1.0)
    # At M=10: F = 0.5*10 + 0.5*(2-0.1) = 5 + 0.95 = 5.95
    assert en.f(10.0) == pytest.approx(5.95)
    assert en.f(1.0) >= 1.0

    en2 = ExcessNoiseFactor.from_ionization(1e4, 5e3)
    assert en2.k_eff == pytest.approx(0.5)
