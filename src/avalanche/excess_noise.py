"""Excess noise factor model."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from pydantic.dataclasses import dataclass

from ..utils._logging import get_logger

log = get_logger("avalanche.excess_noise")


class ExcessNoiseModel(BaseModel):
    """Excess noise factor F(M).

    Uses the McIntyre model:
        F(M) = k × M + (1-k) × (2 - 1/M)
    where k = α_p / α_n is the ionization ratio.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: float = 0.02  # ionization ratio (InP)

    @field_validator("k")
    @classmethod
    def _k_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Ionization ratio k must be in [0, 1], got {v}")
        return v

    def compute(self, gain: float) -> float:
        """Compute excess noise factor."""
        if gain <= 1.0:
            return 1.0
        return self.k * gain + (1.0 - self.k) * (2.0 - 1.0 / gain)


class ExcessNoiseFactor:
    """McIntyre excess noise factor F(M) compatibility class for tests."""

    def __init__(self, k_eff: float = 0.5) -> None:
        self.k_eff = k_eff

    def f(self, M: float | np.ndarray) -> float | np.ndarray:
        M = np.asarray(M, dtype=float)
        k = self.k_eff
        F = k * M + (1.0 - k) * (2.0 - 1.0 / np.clip(M, 1.0, None))
        return float(F) if F.ndim == 0 else F

    @classmethod
    def from_ionization(cls, alpha: float, beta: float) -> ExcessNoiseFactor:
        k = beta / alpha if alpha > 1e-20 else 1.0
        return cls(k_eff=k)
