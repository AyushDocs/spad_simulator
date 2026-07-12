"""Afterpulsing model."""

from __future__ import annotations

from typing import Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ..utils._logging import get_logger

log = get_logger("avalanche.afterpulsing")


class AfterpulsingModel(BaseModel):
    """Trap-related afterpulsing probability.

    Uses a multi-trap model with exponential decay components.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trap_counts: list[int] = [1, 2, 3]
    trap_lifetimes: list[float] = [1e-8, 1e-7, 1e-6]
    trap_probs: list[float] = [0.1, 0.2, 0.3]

    # Compatibility fields for single-trap model in tests
    N_T: Optional[float] = None
    tau_c: Optional[float] = None
    Vbr: float = 76.0

    @model_validator(mode="after")
    def _validate_lengths(self):
        if self.N_T is not None and self.tau_c is not None:
            return self
        n = len(self.trap_counts)
        if len(self.trap_lifetimes) != n:
            raise ValueError("trap_counts and trap_lifetimes must have same length")
        if len(self.trap_probs) != n:
            raise ValueError("trap_counts and trap_probs must have same length")
        return self

    def compute(self, holdoff: float) -> float:
        """Afterpulsing probability for given holdoff time."""
        if self.N_T is not None and self.tau_c is not None:
            return self.afterpulsing_probability(holdoff)
        total = 0.0
        for count, tau, prob in zip(self.trap_counts, self.trap_lifetimes, self.trap_probs):
            total += count * prob * np.exp(-holdoff / tau)
        return total

    # Compatibility methods
    def afterpulsing_probability(self, holdoff: float) -> float:
        if self.N_T is None or self.tau_c is None:
            return self.compute(holdoff)
        return float(1.0 - np.exp(
            -self.N_T * self.tau_c * (1.0 - np.exp(-holdoff / self.tau_c))
        ))

    def effective_dcr(self, raw_dcr: float, holdoff: float) -> float:
        return raw_dcr * (1.0 + self.afterpulsing_probability(holdoff))

    def holdoff_optimal(self, target_ap: float = 0.01) -> float:
        nt = self.N_T if self.N_T is not None else 1e12
        tc = self.tau_c if self.tau_c is not None else 1e-6
        if nt * tc < 1e-30:
            return 0.0
        arg = 1.0 + np.log(1.0 - target_ap) / (nt * tc)
        if arg <= 0:
            return tc * 10.0
        return float(-tc * np.log(max(arg, 1e-30)))
