"""Shared test fixtures — eliminates _make_absorption/_make_material duplication."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.material import Material
from src.core.absorption import InterpolatedAbsorption
from src.utils.loaders import MaterialData, AbsorptionData
from src.core.units import Q


@pytest.fixture
def absorption_data() -> AbsorptionData:
    wl = np.linspace(400e-9, 2000e-9, 50)
    alpha = np.where(wl < 920e-9, 5000.0,
                     np.where(wl < 1650e-9, 4000.0 * np.exp(-(wl - 920e-9) / 500e-9), 10.0))
    return AbsorptionData(material="InP", wavelengths=wl, alphas=alpha)


@pytest.fixture
def inp_material(absorption_data: AbsorptionData) -> Material:
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
    return Material(data, absorption=InterpolatedAbsorption(absorption_data), T=300.0)
