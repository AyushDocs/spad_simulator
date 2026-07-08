"""Smoke tests for simulator facade."""
from __future__ import annotations

import numpy as np
import pytest

from src.main import build_sagcm_spad
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
