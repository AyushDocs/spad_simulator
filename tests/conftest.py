"""Shared test fixtures — eliminates _make_absorption/_make_material duplication."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.material import Material
from src.core.absorption import InterpolatedAbsorption
from src.utils.loaders import MaterialData, AbsorptionData


@pytest.fixture
def absorption_data() -> AbsorptionData:
    wl = np.linspace(400e-9, 2000e-9, 50)
    alpha = np.where(wl < 920e-9, 5000.0,
                     np.where(wl < 1650e-9, 4000.0 * np.exp(-(wl - 920e-9) / 500e-9), 10.0))
    return AbsorptionData(material="InP", wavelengths=wl, alphas=alpha)


@pytest.fixture
def inp_material(absorption_data: AbsorptionData) -> Material:
    data = MaterialData(
        name="InP", eps_r=12.5, mu_n=5400, mu_p=2000,
        vsat_n=1e7, vsat_p=1e7, mc=0.077, mh=0.64,
        tau_n=1e-6, tau_p=1e-6, Eg_0K=1.42,
        varshni_alpha=4.9e-4, varshni_beta=327,
        Nc_300K=5.7e17, Nv_300K=1.1e19, dos_gamma=1.5,
        ionization_e={"Eth": 2.1, "lambda0": 4e-7, "ER0": 3.5e-2, "hw_meV": 42},
        ionization_h={"Eth": 2.1, "lambda0": 4e-7, "ER0": 3.5e-2, "hw_meV": 42},
    )
    return Material(data, absorption=InterpolatedAbsorption(absorption_data), T=300.0)
