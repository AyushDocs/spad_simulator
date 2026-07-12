"""Smoke tests for transport module."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.constants import q, VT
from src.self_consistent.particle_mesh import Carrier
from src.transport.drift_diffusion import DriftDiffusion
from src.transport.jitter import TimingJitter


def test_carrier():
    c = Carrier(x=1e-4, typ="electron", dead_space=1e-5)
    assert c.x == 1e-4
    assert c.typ == "electron"
    assert c.alive is True
    assert c.is_electron is True

    c.move(1e-6, 1e-15)
    assert c.x == pytest.approx(1.01e-4)

    c.exit_check(0, 1e-3)
    assert c.alive is True

    c.exit_check(0, 1e-4)
    assert c.alive is False


def test_drift_diffusion(inp_material):
    dd = DriftDiffusion(inp_material)

    v_e = dd.drift_velocity(1e5, "electron")
    assert v_e < 0
    assert abs(v_e) > 0

    v_h = dd.drift_velocity(1e5, "hole")
    assert v_h > 0

    D_e = dd.diffusion_coefficient(1e5, "electron")
    assert D_e > 0

    D_h = dd.diffusion_coefficient(1e5, "hole")
    assert D_h > 0


def test_timing_jitter():
    detection_times = [1e-12, 1.2e-12, 1.1e-12, 1.3e-12, 0.9e-12]
    stats = TimingJitter.statistics(detection_times)
    assert "mean" in stats
    assert "std" in stats
    assert stats["std"] > 0


def test_fwhm():
    np.random.seed(42)
    t_detect = np.random.normal(10e-12, 1e-12, 500)
    fwhm_val = TimingJitter.fwhm(t_detect, bins=50)
    assert np.isfinite(fwhm_val)
    assert fwhm_val > 0
    assert fwhm_val < 10e-12

    assert np.isnan(TimingJitter.fwhm(np.array([])))
    assert np.isnan(TimingJitter.fwhm(np.array([1e-12])))


def test_fwhm_percentiles():
    t_detect = np.array([1e-12, 2e-12, 3e-12, 4e-12, 5e-12])
    pcts = TimingJitter.percentiles(t_detect, (10, 50, 90))
    assert "t10" in pcts
    assert "t50" in pcts
    assert "t90" in pcts
    assert pcts["t50"] == pytest.approx(3e-12)
