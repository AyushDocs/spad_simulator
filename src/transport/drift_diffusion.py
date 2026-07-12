"""Drift-diffusion transport solver."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from pydantic.dataclasses import dataclass

from typing import TYPE_CHECKING

from ..core.constants import q, kB, VT
from ..utils._logging import get_logger
from ..utils.pydantic_types import NDArray

if TYPE_CHECKING:
    from ..self_consistent.particle_mesh import Carrier

log = get_logger("transport.drift_diffusion")


class DriftDiffusionSolver(BaseModel):
    """1-D drift-diffusion solver for carrier transport.

    Solves the continuity equations using a Scharfetter-Gummel scheme.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: NDArray
    T: float = 300.0

    _x: np.ndarray = PrivateAttr()
    _dx: float = PrivateAttr()

    @model_validator(mode="after")
    def _init_grid(self):
        self._x = np.asarray(self.grid, dtype=float)
        self._dx = float(self._x[1] - self._x[0])
        return self

    def scharfetter_gummel_flux(self, n: np.ndarray, mu: np.ndarray,
                                 F: np.ndarray) -> np.ndarray:
        """Scharfetter-Gummel flux for electron current.

        J_n = q * n * mu * F * Bernoulli(F * dx / (k_B * T / q))
        """
        vth = float((kB * T / q).magnitude)  # V (thermal voltage)
        dx = self._dx

        # Bernoulli function: B(x) = x / (exp(x) - 1)
        x = F * dx / vth
        with np.errstate(over="ignore", invalid="ignore"):
            B = np.where(np.abs(x) > 1e-6,
                         x / (np.exp(x) - 1.0),
                         1.0 - 0.5 * x)

        flux = float(q.to("C").magnitude) * n * mu * F * B
        return flux


class DriftDiffusion:
    """
    Advection-diffusion transport step in a single material.

        x -> x + v_d dt + sqrt(2 D dt) xi

    Drift velocity saturates at high field:
        v_d = mu E / (1 + E / E_sat)
    """

    def __init__(self, material,
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
        vth = float(VT(self._mat.T))
        D0 = mu * vth
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
