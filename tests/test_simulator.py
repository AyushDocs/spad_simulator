"""Smoke tests for simulator facade."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.main import build_sagcm_spad, _write_json_output
from src.avalanche.afterpulsing import AfterpulsingModel
from src.avalanche.excess_noise import ExcessNoiseFactor
from src.simulator import SPADSimulator


@pytest.fixture
def sim():
    device = build_sagcm_spad()
    return SPADSimulator(device)


def test_simulator_init(sim):
    assert sim.grid.no_of_nodes == 500
    assert sim.T == 300.0
    assert sim.detector_area > 0


def test_poisson_solve(sim):
    phi, E, info = sim.solve_poisson(0.0)
    assert len(phi) == sim.grid.no_of_nodes
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(E))


def test_depletion_width(sim):
    xl, xr, Vdep = sim.depletion_width(20.0)
    assert xl < xr
    assert Vdep > 0


def test_get_fields(sim):
    phi, E, Pe, Ph, xl, xr = sim.get_fields(20.0)
    assert len(phi) == sim.grid.no_of_nodes
    assert np.all(np.isfinite(Pe))
    assert np.all(np.isfinite(Ph))


def test_find_breakdown(sim):
    Vbr, info = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    assert Vbr is not None
    assert 50 < Vbr < 100


def test_dark_current(sim):
    Vbr, _ = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    dc = sim.compute_dark_current(Vbr + 1.0)
    assert dc["I_dark"] > 0
    assert dc["DCR"] > 0
    assert np.all(np.isfinite(dc["Pe"]))


def test_json_output(sim):
    Vbr, _ = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    ap = AfterpulsingModel(N_T=1e12, tau_c=1e-6, Vbr=Vbr)
    en = ExcessNoiseFactor(k_eff=0.5)
    ap_metrics = {"N_T": ap.N_T, "tau_c": ap.tau_c,
                  "P_ap_1us": ap.afterpulsing_probability(1e-6),
                  "holdoff_optimal_1pct_s": ap.holdoff_optimal(0.01)}
    en_metrics = {"M_max": 10.0, "F_max": en.f(10.0), "k_eff": 0.5}
    pde_metrics = {"pde_max": 0.68, "wavelength_nm": 1310}
    jitter_metrics = {"sigma_s": 1e-12, "fwhm_s": 2e-12}
    dc_metrics = {"I_dark_A": 3e-8, "DCR_cps": 1e9, "Vex_V": 3.0}
    pdp_metrics = {"905nm": 0.5, "1310nm": 0.74, "1550nm": 0.65}

    with tempfile.TemporaryDirectory() as tmpdir:
        import src.main as main_mod
        old_dir = main_mod._plot_dir
        main_mod._plot_dir = tmpdir
        try:
            _write_json_output(Vbr, sim, ap_metrics, en_metrics,
                               pde_metrics, jitter_metrics,
                               dark_current=dc_metrics, pdp_max=pdp_metrics)
            path = os.path.join(tmpdir, "sim_results.json")
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "device" in data
            assert data["device"]["Vbr_V"] == Vbr
            assert data["device"]["T_K"] == 300.0
            assert "dark_current" in data
            assert data["dark_current"]["I_dark_A"] == 3e-8
            assert "pdp_max" in data
            assert data["pdp_max"]["1310nm"] == 0.74
            assert "afterpulsing" in data
            assert data["afterpulsing"]["N_T"] == 1e12
            assert "excess_noise" in data
            assert data["excess_noise"]["k_eff"] == 0.5
            assert "jitter" in data
            assert data["jitter"]["fwhm_s"] == 2e-12
        finally:
            main_mod._plot_dir = old_dir
