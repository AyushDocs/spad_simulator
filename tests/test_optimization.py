"""Tests for PSOOptimizer and CostFunction."""
from __future__ import annotations

import numpy as np
import pytest

from src.optimization.pso import PSOOptimizer
from src.optimization.cost import CostFunction, PDE, DCR


def test_pso_minimizes_sphere():
    def sphere_cost(x):
        val = -float(np.sum(x ** 2))
        return val, {}

    opt = PSOOptimizer(n_particles=10, n_dims=2,
                       bounds=[(-10, 10), (-10, 10)], max_iter=50)
    best, best_val, history = opt.optimize(sphere_cost, verbose=False)
    assert len(history) == 50
    assert history[-1] >= history[0]
    assert np.all(np.abs(best) < 5.0)


def test_pso_respects_bounds():
    def neg_cost(x):
        return -float(np.sum(x ** 2)), {}

    opt = PSOOptimizer(n_particles=5, n_dims=3,
                       bounds=[(0, 1), (0, 1), (0, 1)], max_iter=20)
    best, _, _ = opt.optimize(neg_cost, verbose=False)
    assert np.all(best >= 0.0)
    assert np.all(best <= 1.0)


def test_pde_function():
    BrP = np.array([0.5, 0.7, 0.9])
    assert abs(PDE(BrP) - 0.7) < 1e-10


def test_dcr_function():
    BrP = np.array([0.5, 0.7, 0.9])
    dcr = DCR(BrP, dark_gen_rate=1e3, area=1e-6)
    expected = 1e3 * 1e-6 * np.mean(BrP)
    assert abs(dcr - expected) < 1e-15


def test_cost_function_evaluate():
    from unittest.mock import MagicMock
    from src.core.layer import Layer
    mock_layer = Layer(thickness=1e-4, doping_type="donor", doping_A=1e16,
                       doping_m=0, doping_x0=0, material="InP")
    mock_mat = MagicMock()
    mock_mat.absorption_coefficient.return_value = 7500.0
    mock_sim = MagicMock()
    mock_sim.device.layers = [mock_layer]
    mock_sim.grid.x = np.linspace(0, 4e-4, 100)
    mock_sim.materials = {"InGaAs": mock_mat}
    mock_sim.find_breakdown.return_value = (20.0, {})
    mock_sim.solve_trigger.return_value = (
        np.full(100, 0.8), np.full(100, 0.4), np.zeros(100))

    cost = CostFunction(BV_target=20.0)
    J, details = cost.evaluate(mock_sim, {"0": 1e16})
    assert "PDE" in details
    assert "DCR" in details
    assert "BV" in details
    assert details["BV"] == 20.0
    assert abs(J) < 10.0


def test_cost_function_bv_penalty():
    from unittest.mock import MagicMock
    from src.core.layer import Layer
    mock_layer = Layer(thickness=1e-4, doping_type="donor", doping_A=1e16,
                       doping_m=0, doping_x0=0, material="InP")
    mock_mat = MagicMock()
    mock_mat.absorption_coefficient.return_value = 7500.0

    def _make_sim(Vbr_val):
        m = MagicMock()
        m.device.layers = [mock_layer]
        m.grid.x = np.linspace(0, 4e-4, 10)
        m.materials = {"InGaAs": mock_mat}
        m.find_breakdown.return_value = (Vbr_val, {})
        m.solve_trigger.return_value = (
            np.full(10, 0.8), np.full(10, 0.4), np.zeros(10))
        return m

    cost = CostFunction(BV_target=20.0)
    J_ok, _ = cost.evaluate(_make_sim(20.0), {})
    J_bad, _ = cost.evaluate(_make_sim(50.0), {})
    assert J_bad < J_ok
