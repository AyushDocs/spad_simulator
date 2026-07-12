from __future__ import annotations

import numpy as np

from ..utils._logging import get_logger
from ..utils.loaders import MaterialData, AbsorptionData
from .constants import q, VT
from .absorption import AbsorptionModel, InterpolatedAbsorption
from .fermi_dirac import BandgapNarrowing
from .units import Q_

log = get_logger("material")

# Default BGN models per material
_BGN_MODELS: dict[str, BandgapNarrowing] = {
    "InP": BandgapNarrowing.for_inp(),
    "InGaAs": BandgapNarrowing.for_ingaas(),
}


class Material:
    """
    Single-material parameter container.

    Wraps ``MaterialData`` (parsed from XML with ``pint.Quantity`` fields)
    with temperature-dependent getters and a pluggable ``AbsorptionModel``.

    All properties return pint Quantities in the project's mixed unit system
    (eV, cm, cm⁻³, cm²/V·s, cm/s, s, etc.).
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
    def data(self) -> MaterialData:
        return self._data

    @property
    def name(self) -> str:
        return self._data.name

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        self._T = value

    # -- properties returning pint Quantities --------------------------------

    @property
    def eps_r(self) -> float:
        return float(self._data.eps_r.magnitude)

    @property
    def mu_n(self) -> float:
        return float(self._data.mu_n.to("cm**2/(V*s)").magnitude)

    @property
    def mu_p(self) -> float:
        return float(self._data.mu_p.to("cm**2/(V*s)").magnitude)

    @property
    def vsat_n(self) -> float:
        return float(self._data.vsat_n.to("cm/s").magnitude)

    @property
    def vsat_p(self) -> float:
        return float(self._data.vsat_p.to("cm/s").magnitude)

    @property
    def mc(self) -> float:
        return float(self._data.mc.magnitude)

    @property
    def mh(self) -> float:
        return float(self._data.mh.magnitude)

    @property
    def tau_n(self) -> float:
        return float(self._data.tau_n.to("s").magnitude)

    @property
    def tau_p(self) -> float:
        return float(self._data.tau_p.to("s").magnitude)

    @property
    def E_ie(self) -> float:
        eth = self._data.ionization_e.get("Eth", Q_(2.16, "eV"))
        return float(eth.to("eV").magnitude) * float(q.magnitude)

    @property
    def E_ih(self) -> float:
        eth = self._data.ionization_h.get("Eth", Q_(2.16, "eV"))
        return float(eth.to("eV").magnitude) * float(q.magnitude)

    def Eg(self, T: float | None = None) -> float:
        T_use = T if T is not None else self._T
        alpha = float(self._data.varshni_alpha.to("eV/K").magnitude)
        beta = float(self._data.varshni_beta.to("K").magnitude)
        eg0 = float(self._data.Eg_0K.to("eV").magnitude)
        return float(eg0 - alpha * T_use ** 2 / (T_use + beta))

    def Eg_bgn(self, N_doping: float, T: float | None = None) -> float:
        Eg = self.Eg(T)
        if self._bgn is not None:
            Eg -= self._bgn.delta_eg(N_doping)
        return Eg

    def Nc(self, T: float | None = None) -> float:
        T_use = T if T is not None else self._T
        nc300 = float(self._data.Nc_300K.to("cm**-3").magnitude)
        gamma = float(self._data.dos_gamma.magnitude)
        return float(nc300 * (T_use / 300.0) ** gamma)

    def Nv(self, T: float | None = None) -> float:
        T_use = T if T is not None else self._T
        nv300 = float(self._data.Nv_300K.to("cm**-3").magnitude)
        gamma = float(self._data.dos_gamma.magnitude)
        return float(nv300 * (T_use / 300.0) ** gamma)

    def ni(self, T: float | None = None) -> float:
        T_use = T if T is not None else self._T
        Eg = self.Eg(T_use)
        Nc = self.Nc(T_use)
        Nv = self.Nv(T_use)
        return float(np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * VT(T_use))))

    def ionization_params(self, carrier: str) -> dict[str, float]:
        raw = self._data.ionization_e if carrier == "electron" else self._data.ionization_h
        return {k: float(v.to(v.units).magnitude) for k, v in raw.items()}

    def absorption_coefficient(self, lam: float) -> float:
        return self._absorption.coefficient(lam)
