from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.material import Material
from ..core.constants import q, kB


class IonizationModel(ABC):
    """Pluggable impact-ionisation coefficient model."""

    @abstractmethod
    def alpha(self, E: float, material: Material, T: float) -> float: ...
    @abstractmethod
    def beta(self, E: float, material: Material, T: float) -> float: ...


class OkutoCrowellModel(IonizationModel):
    """
    Okuto-Crowell impact ionisation coefficients with temperature dependence.

        alpha(E,T) = (qE / Eth_alpha) * exp{
            S - sqrt(S^2 + (Eth_alpha / (qE lambda_alpha))^2)
        }
        S = 0.217 * (Eth_alpha / ER_alpha)^1.14
    """

    def _okuto_temp_factor(self, params: dict, T: float) -> float:
        hw_meV = params.get("hw_meV", 42.0)
        hw_J = hw_meV * 1e-3 * q
        return float(np.tanh(hw_J / (2.0 * kB * T)))

    def _coeff(self, E_abs: float, material: Material,
               T: float, carrier: str) -> float:
        params = material.ionization_params(carrier)
        Eth = params.get("Eth", 2.1)
        factor = self._okuto_temp_factor(params, T)
        lam = params.get("lambda0", 4e-7) * factor
        ER = params.get("ER0", 3.5e-2) * factor
        if E_abs < 1e-10 or lam < 1e-20 or ER < 1e-20:
            return 0.0
        S = 0.217 * (Eth / ER) ** 1.14
        term = Eth / (E_abs * lam)
        arg = S * S + term * term
        return float((E_abs / Eth) * np.exp(S - np.sqrt(arg)))

    def alpha(self, E: float, material: Material, T: float) -> float:
        return self._coeff(abs(E), material, T, "electron")

    def beta(self, E: float, material: Material, T: float) -> float:
        return self._coeff(abs(E), material, T, "hole")


class IonizationCoefficients:
    """
    Per-grid-point impact ionisation coefficients.

    Uses a pluggable ``IonizationModel`` (default Okuto-Crowell)
    and a ``{name: Material}`` dict for per-material parameters.

    Handles dead-space correction (alpha_eff, beta_eff).
    """

    def __init__(self, model: IonizationModel,
                 materials: dict[str, Material],
                 Eg_grid: np.ndarray | None = None,
                 Eth_grid: np.ndarray | None = None,
                 mc_grid: np.ndarray | None = None,
                 mh_grid: np.ndarray | None = None,
                 E_ie_grid: np.ndarray | None = None,
                 E_ih_grid: np.ndarray | None = None,
                 grid_x: np.ndarray | None = None,
                 mat_names: np.ndarray | None = None,
                 T: float = 300.0,
                 use_dead_space: bool = True) -> None:
        self._model = model
        self._materials = materials
        self.use_dead_space = use_dead_space
        self._x = grid_x
        self._Eg = Eg_grid
        self._Eth = Eth_grid
        self._mc = mc_grid
        self._mh = mh_grid
        self._E_ie = E_ie_grid
        self._E_ih = E_ih_grid
        self._mat_names = mat_names
        self._T = T
        self._q = q

    def _material_at(self, idx: int) -> Material:
        name = "InP"
        if self._mat_names is not None and idx < len(self._mat_names):
            name = self._mat_names[idx]
        return self._materials.get(name, list(self._materials.values())[0])

    def _alpha_at_point(self, E_abs: float, idx: int) -> float:
        return self._model.alpha(E_abs, self._material_at(idx), self._T)

    def _beta_at_point(self, E_abs: float, idx: int) -> float:
        return self._model.beta(E_abs, self._material_at(idx), self._T)

    def alpha(self, E: np.ndarray) -> np.ndarray:
        E_abs = np.abs(E)
        out = np.zeros_like(E_abs)
        if self._x is not None and self._Eg is not None:
            active = E_abs > 1e-10
            if not np.any(active):
                return out
            indices = np.where(active)[0]
            for i in indices:
                out[i] = self._alpha_at_point(E_abs[i], i)
        else:
            with np.errstate(divide="ignore", over="ignore"):
                mat_default = next(iter(self._materials.values()))
                out = np.where(E_abs > 1e-10,
                               np.array([self._model.alpha(float(e), mat_default, self._T)
                                         for e in E_abs]), 0.0)
        return out

    def beta(self, E: np.ndarray) -> np.ndarray:
        E_abs = np.abs(E)
        out = np.zeros_like(E_abs)
        if self._x is not None and self._Eg is not None:
            active = E_abs > 1e-10
            if not np.any(active):
                return out
            indices = np.where(active)[0]
            for i in indices:
                out[i] = self._beta_at_point(E_abs[i], i)
        else:
            with np.errstate(divide="ignore", over="ignore"):
                mat_default = next(iter(self._materials.values()))
                out = np.where(E_abs > 1e-10,
                               np.array([self._model.beta(float(e), mat_default, self._T)
                                         for e in E_abs]), 0.0)
        return out

    def alpha_at(self, x: float, E: float, T: float | None = None) -> float:
        E_abs = abs(E)
        if E_abs < 1e-10:
            return 0.0
        mat = next(iter(self._materials.values()))
        if self._mat_names is not None and self._x is not None:
            idx = int(np.clip(np.searchsorted(self._x, x), 0, len(self._x) - 1))
            mat = self._material_at(idx)
        return self._model.alpha(E_abs, mat, T or self._T)

    def beta_at(self, x: float, E: float, T: float | None = None) -> float:
        E_abs = abs(E)
        if E_abs < 1e-10:
            return 0.0
        mat = next(iter(self._materials.values()))
        if self._mat_names is not None and self._x is not None:
            idx = int(np.clip(np.searchsorted(self._x, x), 0, len(self._x) - 1))
            mat = self._material_at(idx)
        return self._model.beta(E_abs, mat, T or self._T)

    def _eth_at(self, x: float, carrier: str = "electron") -> float:
        if carrier == "electron":
            grid = self._E_ie
        else:
            grid = self._E_ih
        if grid is None or self._x is None:
            return 2.16
        return float(np.interp(x, self._x, grid))

    def dead_space_length(self, E: float | np.ndarray,
                          carrier: str = "electron") -> float | np.ndarray:
        E_abs = np.abs(E)
        scalar = np.ndim(E) == 0
        if scalar:
            if E_abs < 1e-10:
                return float(np.inf)
            Eth = self._eth_at(0.0, carrier)
            return float(Eth / (self._q * E_abs))
        if carrier == "electron" and self._E_ie is not None:
            with np.errstate(divide="ignore", over="ignore"):
                return np.where(E_abs > 1e-10,
                                self._E_ie / (self._q * E_abs), np.inf)
        if carrier == "hole" and self._E_ih is not None:
            with np.errstate(divide="ignore", over="ignore"):
                return np.where(E_abs > 1e-10,
                                self._E_ih / (self._q * E_abs), np.inf)
        with np.errstate(divide="ignore", over="ignore"):
            return np.where(E_abs > 1e-10,
                            2.16 / (self._q * E_abs), np.inf)

    def dead_space_at(self, x: float, E: float,
                      carrier: str = "electron") -> float:
        E_abs = abs(E)
        if E_abs < 1e-10:
            return float(np.inf)
        Eth = self._eth_at(x, carrier)
        return float(Eth / (self._q * E_abs))

    def alpha_eff(self, E: np.ndarray, x: np.ndarray,
                  x_ion_start: float | None = None) -> np.ndarray:
        a0 = self.alpha(E)
        if not self.use_dead_space:
            return a0
        start = x_ion_start if x_ion_start is not None else 0.0
        dead_end = start + np.abs(self.dead_space_length(E, "electron"))
        return np.where(x >= dead_end, a0, 0.0)

    def beta_eff(self, E: np.ndarray, x: np.ndarray,
                 x_ion_start: float | None = None) -> np.ndarray:
        b0 = self.beta(E)
        if not self.use_dead_space:
            return b0
        start = x_ion_start if x_ion_start is not None else 0.0
        dead_end = start + np.abs(self.dead_space_length(E, "hole"))
        return np.where(x >= dead_end, b0, 0.0)
