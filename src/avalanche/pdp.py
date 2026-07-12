"""Photon detection probability model."""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ..core.material import Material

# elementary charge (C) — avoid pint import to prevent UnitStrippedWarning
_Q_C = 1.602176634e-19
from ..core.layer import Layer
from ..utils._logging import get_logger
from ..utils.pydantic_types import NDArray

log = get_logger("avalanche.pdp")


class PDPModel(BaseModel):
    """Photon detection probability (PDP).

    PDP(λ, V) = η_abs(λ) × P_trigger(V)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: Optional[NDArray] = None
    x_abs_start: float = 1e-5
    x_abs_stop: float = 1.2e-5
    QE: float = 0.95
    PCE: float = 1.0
    materials: Optional[dict[str, Material]] = None
    reflectivity: float = 0.1

    _x: np.ndarray = PrivateAttr()

    @model_validator(mode="after")
    def _init_grid(self):
        if self.grid is not None:
            self._x = np.asarray(self.grid, dtype=float)
        else:
            self._x = np.array([])
        return self

    def absorption_efficiency(self, abs_coeff: float) -> float:
        """Absorption efficiency.

        η_abs = 1 - exp(-α × (x_stop - x_start))
        """
        return self.QE * (1.0 - np.exp(-abs_coeff * (self.x_abs_stop - self.x_abs_start))) * self.PCE

    def compute(self, abs_coeff: float, trigger_prob: float) -> float:
        """Compute PDP."""
        return self.absorption_efficiency(abs_coeff) * trigger_prob

    # Original methods for test/photocurrent compatibility
    @property
    def _r(self) -> float:
        return self.reflectivity

    def find_absorber(self, layers: List[Layer],
                      material_name: str = "InGaAs"
                      ) -> Tuple[List[Layer], Layer]:
        dead_zone: List[Layer] = []
        absorber: Layer | None = None
        for lyr in layers:
            if lyr.material == material_name and lyr.doping_A < 1e16:
                absorber = lyr
                break
            dead_zone.append(lyr)
        if absorber is None:
            raise ValueError(f"No intrinsic absorber layer '{material_name}' found")
        return dead_zone, absorber

    def dead_zone_transmission(self, lam: float,
                                dead_zone_layers: List[Layer]) -> float:
        trans = 1.0
        for lyr in dead_zone_layers:
            mat = self.materials[lyr.material]
            alpha = mat.absorption_coefficient(lam)
            trans *= np.exp(-alpha * lyr.thickness)
        return trans

    def pdp_integral(self, lam: float,
                     xx: np.ndarray,
                     P_trigger_slice: np.ndarray,
                     dead_zone_trans_or_dx: float,
                     dx: Optional[float] = None,
                     material_name: str = "InGaAs") -> float:
        if dx is None:
            dx = dead_zone_trans_or_dx
            dead_zone_trans = 1.0
        else:
            dead_zone_trans = dead_zone_trans_or_dx
        mat = self.materials[material_name]
        alpha = mat.absorption_coefficient(lam)
        integrand = alpha * np.exp(-alpha * xx) * P_trigger_slice
        apt = float(np.trapezoid(integrand, dx=dx))
        return (1.0 - self.reflectivity) * dead_zone_trans * apt

    def absorption_probability(self, lam: float,
                                L_abs: float,
                                material_name: str = "Si") -> float:
        mat = self.materials[material_name]
        alpha = mat.absorption_coefficient(lam)
        return 1.0 - np.exp(-alpha * L_abs)

    def pdp(self, lam: float, L_abs: float,
            P_trigger: float,
            material_name: str = "Si",
            dead_zone: float = 0.0) -> float:
        mat = self.materials[material_name]
        alpha = mat.absorption_coefficient(lam)
        absorption = 1.0 - np.exp(-alpha * max(L_abs, 0.0))
        return (1.0 - self.reflectivity) * absorption * P_trigger

    def photocurrent_density(self, x: np.ndarray,
                             alpha: np.ndarray,
                             phi_photon: float,
                             x_start: float,
                             x_end: float) -> np.ndarray:
        mask = (x >= x_start - 1e-16) & (x <= x_end + 1e-16)
        if not np.any(mask):
            return np.zeros_like(x)
        x_rel = x[mask] - x_start
        alpha_m = alpha[mask]
        G_opt = phi_photon * alpha_m * np.exp(-alpha_m * x_rel)
        J = np.zeros_like(x)
        J[mask] = _Q_C * G_opt
        return J
