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
    """Trap-assisted tunneling current density (Hurkx model)."""

    Eg_mulp: float
    mc_mulp: float
    mh_mulp: float
    T: float = 300.0
    N_T: float = 1e12

    _tunnel: TunnelingModel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tunnel", TunnelingModel(T=self.T, N_T=self.N_T))

    @property
    def name(self) -> str:
        return "TAT"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        J = self._tunnel.tat_current(
            F, self.Eg_mulp, self.mc_mulp, self.mh_mulp, N_T=self.N_T)  # A/cm³
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
