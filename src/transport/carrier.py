"""Carrier transport module."""

from __future__ import annotations

import numpy as np

from ..core.constants import q, kB


class CarrierTransport:
    """Handles drift-diffusion transport equations.

    Continuity equation:
        ∂n/∂t = (1/q) ∂J_n/∂x + G - R
        ∂p/∂t = -(1/q) ∂J_p/∂x + G - R

    Current densities:
        J_n = q * n * mu_n * F + q * D_n * ∂n/∂x
        J_p = q * p * mu_p * F - q * D_p * ∂p/∂x
    """

    def drift_velocity(self, F: float, mu: float, vsat: float) -> float:
        """Drift velocity (cm/s).

        v = mu * F / sqrt(1 + (mu * F / vsat)²)
        """
        muF = mu * F
        return muF / np.sqrt(1.0 + (muF / vsat) ** 2)

    def diffusion_coefficient(self, mu: float) -> float:
        """Diffusion coefficient via Einstein relation.

        D = (k_B * T / q) * mu
        """
        return float(kB.magnitude) * 300.0 / float(q.magnitude) * mu
