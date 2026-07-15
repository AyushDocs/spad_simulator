"""Smoke tests for all study modules using mock simulator objects."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.studies.fields import (
    find_breakdown, plot_device_structure, run_field_sweep, run_trigger_profiles,
    run_trigger_vs_vex,
)
from src.studies.dark_current import (
    run_dark_current_sweep, collect_dark_current_metrics,
)
from src.studies.iv import run_iv_characteristic, run_comprehensive_iv
from src.studies.pde import (
    run_pde_spectrum, run_pde_vs_vex, collect_pde_max_metrics,
)
from src.studies.avalanche import (
    run_afterpulsing, run_excess_noise, run_jitter,
)


def _mock_sim():
    sim = MagicMock()
    sim.grid.x = np.linspace(0, 5e-4, 500)
    sim.grid.no_of_nodes = 500
    sim.grid.dx = 1e-6
    sim.grid.gradient.return_value = np.zeros(500)
    sim.T = 300.0
    sim.detector_area = 1e-6
    sim.device.layers = []
    sim.device.material.mat_name = "InP"
    sim.device.net_doping_on_grid = np.zeros(500)
    mock_ingaas = MagicMock()
    mock_ingaas.absorption_coefficient.return_value = 7500.0  # cm⁻¹ at 1550 nm
    sim.materials = {"InGaAs": mock_ingaas}

    phi = np.linspace(0, 75.0, 500)
    E = np.zeros(500)
    Pe = np.full(500, 0.5)
    Ph = np.full(500, 0.3)
    sim.solve_poisson.return_value = (phi, E, {"Vbr": 75.0, "iterations": 5})
    sim.get_fields.return_value = (phi, E, Pe, Ph, 0.0, 3e-4)
    sim.solve_trigger.return_value = (Pe, Ph, E)
    sim.trigger_for_pdp.return_value = (Pe, Ph)
    sim.depletion_width.return_value = (0.0, 3e-4, 70.0)
    sim.find_breakdown.return_value = (75.0, {"Vbr": 75.0})
    sim.compute_dark_current.return_value = {
        "I_dark": 1e-8, "DCR": 1e9, "Pe": Pe, "Ph": Ph, "E": E,
    }
    sim.compute_photocurrent.return_value = 5e-6
    sim.compute_pde_spectrum.return_value = np.array([0.74])

    sim.pde_model.find_absorber.return_value = (
        [MagicMock(thickness=0.5e-4)], MagicMock(thickness=1.5e-4))
    sim.pde_model.pde_integral.return_value = 0.7
    sim.pde_model.photocurrent_density.return_value = np.full(500, 1e2)

    sim.ionization.alpha.return_value = np.full(500, 1e4)
    sim.ionization.beta.return_value = np.full(500, 5e3)

    sim.run_mc_ensemble.return_value = [
        {"detection_times": np.array([1e-12, 2e-12, 3e-12])},
    ]
    return sim


@patch("src.studies.fields.get_plotter")
def test_find_breakdown(mock_plotter):
    sim = _mock_sim()
    Vbr = find_breakdown(sim)
    assert Vbr == 75.0


@patch("src.studies.fields.get_plotter")
def test_run_field_sweep(mock_plotter):
    sim = _mock_sim()
    run_field_sweep(sim, 75.0)
    assert sim.solve_poisson.call_count == 6


@patch("src.studies.fields.get_plotter")
def test_run_trigger_profiles(mock_plotter):
    sim = _mock_sim()
    run_trigger_profiles(sim, 75.0)
    assert sim.solve_trigger.call_count == 8


@patch("src.studies.fields.get_plotter")
def test_run_trigger_vs_vex(mock_plotter):
    sim = _mock_sim()
    run_trigger_vs_vex(sim, 75.0, n_pts=5)
    assert sim.solve_trigger.call_count == 5


@patch("src.studies.dark_current.get_plotter")
def test_run_dark_current_sweep(mock_plotter):
    sim = _mock_sim()
    run_dark_current_sweep(sim, 75.0)
    assert sim.compute_dark_current.call_count == 11


def test_collect_dark_current_metrics():
    sim = _mock_sim()
    result = collect_dark_current_metrics(sim, 75.0)
    assert result["I_dark_A"] == 1e-8
    assert result["DCR_cps"] == 1e9


@patch("src.studies.iv.get_plotter")
def test_run_iv_characteristic(mock_plotter):
    sim = _mock_sim()
    run_iv_characteristic(sim, 75.0)
    assert sim.get_fields.call_count > 0


@patch("src.studies.pde.get_plotter")
def test_run_pde_spectrum(mock_plotter):
    sim = _mock_sim()
    run_pde_spectrum(sim, 75.0)
    assert sim.get_fields.call_count > 0


@patch("src.studies.pde.get_plotter")
def test_run_pde_vs_vex(mock_plotter):
    sim = _mock_sim()
    run_pde_vs_vex(sim, 75.0)
    assert sim.get_fields.call_count > 0


def test_collect_pde_max_metrics():
    sim = _mock_sim()
    result = collect_pde_max_metrics(sim, [1310, 1550])
    assert "1310nm" in result
    assert "1550nm" in result


@patch("src.studies.avalanche.get_plotter")
def test_run_afterpulsing(mock_plotter):
    sim = _mock_sim()
    result = run_afterpulsing(sim, 75.0)
    assert "P_ap_1us" in result
    assert result["P_ap_1us"] >= 0


@patch("src.studies.avalanche.get_plotter")
def test_run_excess_noise(mock_plotter):
    sim = _mock_sim()
    result = run_excess_noise(sim, 75.0)
    assert "M_max" in result
    assert "F_max" in result


@patch("src.studies.avalanche.get_plotter")
def test_run_jitter(mock_plotter):
    sim = _mock_sim()
    result = run_jitter(sim, 75.0)
    assert "sigma_s" in result
