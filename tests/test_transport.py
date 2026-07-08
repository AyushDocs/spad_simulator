"""Smoke tests for transport module."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.material import Material
from src.core.constants import q, VT
from src.core.absorption import InterpolatedAbsorption
from src.utils.loaders import MaterialData, AbsorptionData
from src.transport.carrier import Carrier
from src.transport.drift_diffusion import DriftDiffusion
from src.transport.jitter import TimingJitter


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


def test_drift_diffusion():
    mat = _make_material()
    dd = DriftDiffusion(mat)

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
