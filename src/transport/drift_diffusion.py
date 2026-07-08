from __future__ import annotations

import numpy as np

from ..core.material import Material
from ..core.constants import VT
from .carrier import Carrier


class DriftDiffusion:
    """
    Advection-diffusion transport step in a single material.

        x -> x + v_d dt + sqrt(2 D dt) xi

    Drift velocity saturates at high field:
        v_d = mu E / (1 + E / E_sat)
    """

    def __init__(self, material: Material,
                 use_high_field_diffusion: bool = True) -> None:
        self._mat = material
        self.use_high_field_diffusion = use_high_field_diffusion

    def drift_velocity(self, E: float, carrier_type: str) -> float:
        E_abs = abs(E)
        if carrier_type == "electron":
            mu = self._mat.mu_n
            vsat = self._mat.vsat_n
            sign = -np.sign(E)
        else:
            mu = self._mat.mu_p
            vsat = self._mat.vsat_p
            sign = np.sign(E) if E != 0 else 0.0
        if E_abs < 1e-10:
            return 0.0
        return float(sign * mu * E_abs / (1.0 + E_abs / vsat))

    def diffusion_coefficient(self, E: float, carrier_type: str) -> float:
        if carrier_type == "electron":
            mu = self._mat.mu_n
            vsat = self._mat.vsat_n
        else:
            mu = self._mat.mu_p
            vsat = self._mat.vsat_p
        D0 = mu * VT(self._mat.T)
        if not self.use_high_field_diffusion or abs(E) < 1e-10:
            return D0
        return D0 * (1.0 + 2.0 * (abs(E) / vsat) ** 2)

    def step(self, carrier: Carrier, E_local: float,
             dt: float, x_left: float, x_right: float) -> float:
        carrier.E = E_local
        v_d = self.drift_velocity(E_local, carrier.typ)
        D = self.diffusion_coefficient(E_local, carrier.typ)
        dx = v_d * dt + np.sqrt(2.0 * D * dt) * np.random.randn()
        carrier.move(dx, dt)
        if carrier.dead_space_remaining > 0:
            carrier.dead_space_remaining -= abs(dx)
            if carrier.dead_space_remaining < 0:
                carrier.dead_space_remaining = 0.0
        carrier.exit_check(x_left, x_right)
        return dx
