from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from ..core.constants import q, kB
from ..core.material import Material
from ..utils.pydantic_types import NDArray


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
        hw_J = hw_meV * 1e-3 * float(q.magnitude)
        return float(np.tanh(hw_J / (2.0 * float(kB.magnitude) * T)))

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


class OkutoCrowellCoefficients:
    """Vectorized Okuto-Crowell ionization coefficients for a single material.

    Provides the same ``alpha_n(F)`` / ``alpha_p(F)`` array interface as
    ``IonizationCoefficients``, but uses the Okuto-Crowell model with
    material-specific parameters loaded from XML (via ``Material``).

    This is the preferred class for the InP multiplication layer in the
    SAGCM SPAD simulator, since the Okuto-Crowell parameters are calibrated
    against measured InP/InGaAs APD data.
    """

    def __init__(self, material: Material, T: float = 300.0) -> None:
        self._model = OkutoCrowellModel()
        self._mat = material
        self._T = T
        # expose same attributes as IonizationCoefficients for compatibility
        self.use_dead_space = False
        self._x = None
        self._Eg = None
        # Fake Chynoweth attributes (unused in OC path) for duck-typing
        self.alpha_n0 = 1.0
        self.alpha_p0 = 1.0
        self.E_n = 0.0
        self.E_p = 0.0
        self.n_n = 1.0
        self.n_p = 1.0
        self._materials = None
        self._model_ref = self._model  # keep OC model reference

    def alpha_n(self, F: NDArray) -> NDArray:
        """Electron ionization coefficient (cm⁻¹) via Okuto-Crowell."""
        F = np.asarray(F, dtype=float)
        return np.array([
            self._model.alpha(float(f), self._mat, self._T) if f > 1e4 else 0.0
            for f in F
        ])

    def alpha_p(self, F: NDArray) -> NDArray:
        """Hole ionization coefficient (cm⁻¹) via Okuto-Crowell."""
        F = np.asarray(F, dtype=float)
        return np.array([
            self._model.beta(float(f), self._mat, self._T) if f > 1e4 else 0.0
            for f in F
        ])

    # Compatibility methods for tests (same interface as IonizationCoefficients)
    def alpha(self, E: np.ndarray) -> np.ndarray:
        return self.alpha_n(np.abs(E))

    def beta(self, E: np.ndarray) -> np.ndarray:
        return self.alpha_p(np.abs(E))

    def dead_space_length(self, E: float | np.ndarray, carrier: str = "electron",
                          Eg: float = 1.35,
                          Eth: float | None = None) -> float | np.ndarray:
        """Dead-space length using material threshold energies."""
        E_abs = np.abs(np.asarray(E, dtype=float))
        if Eth is not None:
            E_th = Eth
        else:
            E_th = (1.5 * Eg) if carrier == "electron" else (1.0 * Eg)
        with np.errstate(divide="ignore", invalid="ignore"):
            l_dead = np.where(E_abs > 1e4, E_th / E_abs, 0.0)
        return float(l_dead) if np.ndim(E) == 0 else l_dead

    def effective_alpha_n(self, F: np.ndarray, Eg: float = 1.35) -> np.ndarray:
        """Effective electron ionization coefficient with material Eth."""
        alpha = self.alpha_n(np.abs(F))
        params = self._mat.ionization_params("electron")
        Eth = params.get("Eth", 1.5 * Eg)
        ld = self.dead_space_length(F, "electron", Eg, Eth=Eth)
        return np.where(ld > 0, alpha / (1.0 + alpha * ld), alpha)

    def effective_alpha_p(self, F: np.ndarray, Eg: float = 1.35) -> np.ndarray:
        """Effective hole ionization coefficient with material Eth."""
        alpha = self.alpha_p(np.abs(F))
        params = self._mat.ionization_params("hole")
        Eth = params.get("Eth", 1.0 * Eg)
        ld = self.dead_space_length(F, "hole", Eg, Eth=Eth)
        return np.where(ld > 0, alpha / (1.0 + alpha * ld), alpha)


class IonizationCoefficients:
    """Impact ionization coefficients (electrons and holes).

    Uses the Chynoweth model:
        α(F) = α_n0 * exp(- E_n * 10 / F)

    All inputs/outputs in cm/V units.
    """

    def __init__(self, *args, **kwargs):
        # Compatibility properties/methods for tests
        self.use_dead_space = True
        self._x = None
        self._Eg = None
        self._T = kwargs.get("T", 300.0)

        if len(args) >= 2 and not isinstance(args[0], (int, float)):
            # Test style signature: IonizationCoefficients(model, materials, Eg_grid=Eg, ...)
            self.alpha_n0 = args[0]  # model
            self.alpha_p0 = args[1]  # materials
            self.E_n = 0.0
            self.E_p = 0.0
            self.n_n = 1.0
            self.n_p = 1.0
            self._model = args[0]
            self._materials = args[1]
        else:
            # Standard style signature
            self.alpha_n0 = kwargs.get("alpha_n0") if "alpha_n0" in kwargs else (args[0] if len(args) > 0 else 1.16e6)
            self.alpha_p0 = kwargs.get("alpha_p0") if "alpha_p0" in kwargs else (args[1] if len(args) > 1 else 5.94e5)
            self.E_n = kwargs.get("E_n") if "E_n" in kwargs else (args[2] if len(args) > 2 else 1.77e5)
            self.E_p = kwargs.get("E_p") if "E_p" in kwargs else (args[3] if len(args) > 3 else 2.41e5)
            self.n_n = kwargs.get("n_n") if "n_n" in kwargs else (args[4] if len(args) > 4 else 1.0)
            self.n_p = kwargs.get("n_p") if "n_p" in kwargs else (args[5] if len(args) > 5 else 1.0)
            self._model = None
            self._materials = None

    def alpha_n(self, F: NDArray) -> NDArray:
        """Electron ionization coefficient (cm⁻¹).

        Chynoweth model: α_n(F) = α_n0 · exp(-(E_n/F)^n_n)
        """
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            result = np.where(
                F > 1e4,
                self.alpha_n0 * np.exp(- (self.E_n / F) ** self.n_n),
                0.0)
        return result

    def alpha_p(self, F: NDArray) -> NDArray:
        """Hole ionization coefficient (cm⁻¹).

        Chynoweth model: α_p(F) = α_p0 · exp(-(E_p/F)^n_p)
        """
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            result = np.where(
                F > 1e4,
                self.alpha_p0 * np.exp(- (self.E_p / F) ** self.n_p),
                0.0)
        return result

    # Compatibility methods for tests
    def alpha(self, E: np.ndarray) -> np.ndarray:
        if self._model is not None:
            model = self.alpha_n0
            materials = self.alpha_p0
            E_abs = np.abs(E)
            mat_default = next(iter(materials.values()))
            return np.where(E_abs > 1e-10,
                            np.array([model.alpha(float(e), mat_default, self._T)
                                      for e in E_abs]), 0.0)
        else:
            return self.alpha_n(np.abs(E))

    def beta(self, E: np.ndarray) -> np.ndarray:
        if self._model is not None:
            model = self.alpha_n0
            materials = self.alpha_p0
            E_abs = np.abs(E)
            mat_default = next(iter(materials.values()))
            return np.where(E_abs > 1e-10,
                            np.array([model.beta(float(e), mat_default, self._T)
                                      for e in E_abs]), 0.0)
        else:
            return self.alpha_p(np.abs(E))

    def dead_space_length(self, E: float | np.ndarray, carrier: str = "electron",
                          Eg: float = 1.35,
                          Eth: float | None = None) -> float | np.ndarray:
        E_abs = np.abs(np.asarray(E, dtype=float))
        if Eth is not None:
            E_th = Eth
        else:
            E_th = (1.5 * Eg) if carrier == "electron" else (1.0 * Eg)
        with np.errstate(divide="ignore", invalid="ignore"):
            l_dead = np.where(E_abs > 1e4, E_th / E_abs, 0.0)
        return float(l_dead) if np.ndim(E) == 0 else l_dead

    def effective_alpha_n(self, F: np.ndarray, Eg: float = 1.35) -> np.ndarray:
        alpha = self.alpha_n(np.abs(F))
        ld = self.dead_space_length(F, "electron", Eg)
        return np.where(ld > 0, alpha / (1.0 + alpha * ld), alpha)

    def effective_alpha_p(self, F: np.ndarray, Eg: float = 1.35) -> np.ndarray:
        alpha = self.alpha_p(np.abs(F))
        ld = self.dead_space_length(F, "hole", Eg)
        return np.where(ld > 0, alpha / (1.0 + alpha * ld), alpha)
