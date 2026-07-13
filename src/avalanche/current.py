from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic.dataclasses import dataclass

from ..core.constants import q
from ..utils.pydantic_types import NDArray
from .tunneling import TunnelingModel


class CurrentDensityComponent(ABC):
    """Interface for current density components returning A/cm³."""

    @abstractmethod
    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


@dataclass(config=dict(arbitrary_types_allowed=True))
class SRHCurrentDensity(CurrentDensityComponent):
    """Shockley-Read-Hall thermal generation current density.

    Only active inside the InGaAs absorption region.  Uses a single
    ni value and tau_n, tau_p for the absorption layer only.
    """

    tau_n_absorber: float
    tau_p_absorber: float
    mat_name_grid: NDArray
    ni_absorber: float

    @property
    def name(self) -> str:
        return "SRH"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        tau = (self.tau_n_absorber + self.tau_p_absorber) / 2.0
        G = self.ni_absorber / (2.0 * tau)
        J = float(q.to("C").magnitude) * G
        in_absorber = self.mat_name_grid == "InGaAs"
        return J * in_absorber * (F > 1e4)


@dataclass(config=dict(arbitrary_types_allowed=True))
class BTBTCurrentDensity(CurrentDensityComponent):
    """Band-to-band tunneling current density (Kane model).

    A and B are derived from first principles using the multiplication
    layer bandgap and conductivity effective masses.
    """

    Eg_mulp: float
    mc_mulp: float
    mh_mulp: float
    T: float = 300.0
    N_T: float = 1e12

    _tunnel: TunnelingModel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tunnel", TunnelingModel(
            T=self.T, N_T=self.N_T,
            Eg_mulp=self.Eg_mulp, mc_mulp=self.mc_mulp, mh_mulp=self.mh_mulp))

    @property
    def name(self) -> str:
        return "BTBT"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        J = self._tunnel.btbt_current(F)  # A/cm³
        return J


@dataclass(config=dict(arbitrary_types_allowed=True))
class TATCurrentDensity(CurrentDensityComponent):
    """Trap-assisted tunneling current density (Hurkx model).

    Computes the field-enhanced portion of the SRH generation rate using
    the Hurkx phonon-assisted tunneling model:

        J_TAT = q * (n_i / (2*tau)) * (Gamma_e + Gamma_h)

    where Gamma_e, Gamma_h are the Hurkx enhancement factors.

    For InGaAs this is the field enhancement ONLY (zero-field SRH is
    handled by SRHCurrentDensity).  For InP and grading layers this
    provides the full SRH+TAT generation since no other component
    accounts for zero-field SRH in those materials.
    """

    mat_name_grid: NDArray
    materials: dict  # str -> Material
    T: float = 300.0
    a_frac: float = 0.75

    @property
    def name(self) -> str:
        return "TAT"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        q_val = float(q.to("C").magnitude)
        J = np.zeros_like(x)

        for mat_name in self.materials:
            mat = self.materials[mat_name]
            mask = self.mat_name_grid == mat_name
            if not np.any(mask):
                continue

            ni = mat.ni(self.T)
            tau = (mat.tau_n + mat.tau_p) / 2.0
            Eg = mat.Eg(self.T)
            mc = mat.mc
            mh = mat.mh

            if mat_name == "InGaAs":
                # SRHCurrentDensity handles zero-field in InGaAs,
                # so only add the field-enhanced portion.
                gamma_e = TunnelingModel.hurkx_gamma(F[mask], mc, (1.0 - self.a_frac) * Eg, self.T)
                gamma_h = TunnelingModel.hurkx_gamma(F[mask], mh, self.a_frac * Eg, self.T)
                Gamma_total = gamma_e + gamma_h
            else:
                # Full SRH + TAT generation for materials not covered by SRH
                gamma_e = TunnelingModel.hurkx_gamma(F[mask], mc, (1.0 - self.a_frac) * Eg, self.T)
                gamma_h = TunnelingModel.hurkx_gamma(F[mask], mh, self.a_frac * Eg, self.T)
                Gamma_total = 1.0 + gamma_e + gamma_h

            J[mask] = q_val * (ni / (2.0 * tau)) * Gamma_total

        return J


class CompositeCurrentDensity(CurrentDensityComponent):
    """Sum of multiple current components."""

    def __init__(self) -> None:
        self._components: list[CurrentDensityComponent] = []

    def add(self, component: CurrentDensityComponent) -> None:
        self._components.append(component)

    @property
    def components(self) -> list[CurrentDensityComponent]:
        return list(self._components)

    @property
    def name(self) -> str:
        return "Composite"

    def compute(self, x: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
        total = np.zeros_like(x)
        for comp in self._components:
            total += comp.compute(x, F, **kwargs)
        return total
