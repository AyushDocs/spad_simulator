from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.constants import q
from .tunneling import TunnelingModel


class CurrentComponent(ABC):
    """Interface for current density components returning A/cm³."""

    @abstractmethod
    def compute(self, x: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class SRHCurrent(CurrentComponent):
    """Shockley-Read-Hall thermal generation current density."""

    def __init__(self,
                 tau_n_grid: np.ndarray | None = None,
                 tau_p_grid: np.ndarray | None = None,
                 tau_default: float = 1e-6) -> None:
        self._tau_n = tau_n_grid
        self._tau_p = tau_p_grid
        self._tau_default = tau_default

    @property
    def name(self) -> str:
        return "SRH"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        ni_arr = kwargs.get("ni_arr", None)
        if self._tau_n is not None and self._tau_p is not None:
            tau = (self._tau_n + self._tau_p) / 2.0
        elif ni_arr is not None:
            tau = np.full_like(ni_arr, self._tau_default)
        else:
            return np.zeros_like(x)
        G = ni_arr / (2.0 * tau)
        return q * G


class BTBTCurrent(CurrentComponent):
    """Band-to-band tunneling current density (Kane model)."""

    def __init__(self, T: float = 300.0,
                 N_T: float = 1e12) -> None:
        self._tunnel = TunnelingModel(T=T, N_T=N_T)

    @property
    def name(self) -> str:
        return "BTBT"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        J = self._tunnel.btbt_current(
            F, kwargs.get("Eg_arr", np.zeros_like(F)),
            kwargs.get("mc_arr", np.zeros_like(F)),
            kwargs.get("mh_arr", np.zeros_like(F)))
        dx = x[1] - x[0]
        return J / dx


class TATCurrent(CurrentComponent):
    """Trap-assisted tunneling current density (Hurkx model)."""

    def __init__(self, T: float = 300.0,
                 N_T: float = 1e12) -> None:
        self._tunnel = TunnelingModel(T=T, N_T=N_T)

    @property
    def name(self) -> str:
        return "TAT"

    def compute(self, x: np.ndarray, F: np.ndarray,
                **kwargs) -> np.ndarray:
        J = self._tunnel.tat_current(
            F, kwargs.get("Eg_arr", np.zeros_like(F)),
            kwargs.get("mc_arr", np.zeros_like(F)),
            kwargs.get("mh_arr", np.zeros_like(F)))
        dx = x[1] - x[0]
        return J / dx


class CompositeCurrent(CurrentComponent):
    """Sum of multiple current components."""

    def __init__(self) -> None:
        self._components: list[CurrentComponent] = []

    def add(self, component: CurrentComponent) -> None:
        self._components.append(component)

    @property
    def components(self) -> list[CurrentComponent]:
        return list(self._components)

    @property
    def name(self) -> str:
        return "Composite"

    def compute(self, x: np.ndarray, F: np.ndarray, **kwargs) -> np.ndarray:
        total = np.zeros_like(x)
        for comp in self._components:
            total += comp.compute(x, F, **kwargs)
        return total
