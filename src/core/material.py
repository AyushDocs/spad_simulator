from __future__ import annotations

import numpy as np

from ..utils._logging import get_logger
from ..utils.loaders import MaterialData, AbsorptionData
from .constants import q, VT
from .absorption import AbsorptionModel, InterpolatedAbsorption
from .fermi_dirac import BandgapNarrowing

log = get_logger("material")

# Default BGN models per material
_BGN_MODELS: dict[str, BandgapNarrowing] = {
    "InP": BandgapNarrowing.for_inp(),
    "InGaAs": BandgapNarrowing.for_ingaas(),
}


class Material:
    """
    Single-material parameter container.

    Wraps ``MaterialData`` (parsed from XML) with temperature-dependent
    getters and a pluggable ``AbsorptionModel``.
    """

    def __init__(self,
                 data: MaterialData,
                 absorption: AbsorptionModel | None = None,
                 T: float = 300.0,
                 bandgap_narrowing: BandgapNarrowing | None = None) -> None:
        self._data = data
        self._absorption = absorption if absorption is not None else InterpolatedAbsorption(AbsorptionData(material="unknown", wavelengths=np.zeros(1), alphas=np.zeros(1)))
        self._T = T
        self._bgn = bandgap_narrowing or _BGN_MODELS.get(data.name)
        log.info("Material %s  T=%.0f K", self.name, T)

    @property
    def name(self) -> str:
        return self._data.name

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        self._T = value

    @property
    def eps_r(self) -> float:
        return float(self._data.eps_r)

    @property
    def mu_n(self) -> float:
        return float(self._data.mu_n)

    @property
    def mu_p(self) -> float:
        return float(self._data.mu_p)

    @property
    def vsat_n(self) -> float:
        return float(self._data.vsat_n)

    @property
    def vsat_p(self) -> float:
        return float(self._data.vsat_p)

    @property
    def mc(self) -> float:
        return float(self._data.mc)

    @property
    def mh(self) -> float:
        return float(self._data.mh)

    @property
    def tau_n(self) -> float:
        return float(self._data.tau_n)

    @property
    def tau_p(self) -> float:
        return float(self._data.tau_p)

    @property
    def E_ie(self) -> float:
        return self._data.ionization_e.get("Eth", 2.16) * q

    @property
    def E_ih(self) -> float:
        return self._data.ionization_h.get("Eth", 2.16) * q

    def Eg(self, T: float | None = None) -> float:
        T_use = T if T is not None else self._T
        return float(self._data.Eg_0K
                     - self._data.varshni_alpha * T_use ** 2
                     / (T_use + self._data.varshni_beta))

    def Eg_bgn(self, N_doping: float, T: float | None = None) -> float:
        """Bandgap with narrowing correction for doping N_doping (cm⁻³)."""
        Eg = self.Eg(T)
        if self._bgn is not None:
            Eg -= self._bgn.delta_eg(N_doping)
        return Eg

    def Nc(self, T: float | None = None) -> float:
        T_use = T if T is not None else self._T
        return float(self._data.Nc_300K * (T_use / 300.0) ** self._data.dos_gamma)

    def Nv(self, T: float | None = None) -> float:
        T_use = T if T is not None else self._T
        return float(self._data.Nv_300K * (T_use / 300.0) ** self._data.dos_gamma)

    def ni(self, T: float | None = None) -> float:
        T_use = T if T is not None else self._T
        Eg = self.Eg(T_use)
        Nc = self.Nc(T_use)
        Nv = self.Nv(T_use)
        return float(np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * VT(T_use))))

    def ionization_params(self, carrier: str) -> dict:
        if carrier == "electron":
            return dict(self._data.ionization_e)
        return dict(self._data.ionization_h)

    def absorption_coefficient(self, lam: float) -> float:
        return self._absorption.coefficient(lam)
