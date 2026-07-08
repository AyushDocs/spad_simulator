from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.constants import q
from ..core.material import Material
from ..core.layer import Layer


class PDPModel:
    """
    Photon Detection Probability using Beer-Lambert absorption.

        PDP(lambda, V) = (1 - r) * T_dead(lambda)
                         * int alpha * exp(-alpha * (x - x_abs))
                         * P_trigger(x) dx
    """

    def __init__(self, materials: dict[str, Material],
                 reflectivity: float = 0.1) -> None:
        self.materials = materials
        self._r = reflectivity

    def find_absorber(self, layers: List[Layer],
                      material_name: str = "InGaAs"
                      ) -> Tuple[List[Layer], Layer]:
        dead_zone: List[Layer] = []
        absorber: Layer | None = None
        for lyr in layers:
            if lyr.material == material_name and lyr.doping_A < 1e14:
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
                     dead_zone_trans: float,
                     dx: float,
                     material_name: str = "InGaAs") -> float:
        mat = self.materials[material_name]
        alpha = mat.absorption_coefficient(lam)
        integrand = alpha * np.exp(-alpha * xx) * P_trigger_slice
        # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
        apt = float(np.trapezoid(integrand, dx=dx))
        return (1.0 - self._r) * dead_zone_trans * apt

    def absorption_probability(self, lam: float,
                                L_abs: float,
                                material_name: str = "Si") -> float:
        mat = self.materials[material_name]
        alpha = mat.absorption_coefficient(lam)
        return 1.0 - np.exp(-alpha * L_abs)

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
        J[mask] = q * G_opt
        return J

    def pdp(self, lam: float, L_abs: float,
            P_trigger: float,
            material_name: str = "Si",
            dead_zone: float = 0.0) -> float:
        mat = self.materials[material_name]
        alpha = mat.absorption_coefficient(lam)
        absorption = 1.0 - np.exp(-alpha * max(L_abs, 0.0))
        return (1.0 - self._r) * absorption * P_trigger
