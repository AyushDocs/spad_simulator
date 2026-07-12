"""Geiger-mode APD model."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from pydantic.dataclasses import dataclass

from ..core.constants import q
from ..utils._logging import get_logger

log = get_logger("avalanche.geiger")


class GeigerModel(BaseModel):
    """Geiger-mode APD operation model.

    Models the carrier dynamics in a SPAD during Geiger-mode operation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dead_time: float = 1e-9  # dead time (s)
    quenching_resistance: float = 1e6  # quenching resistance (ohm)

    _q_val: float = PrivateAttr()

    @model_validator(mode="after")
    def _init_q(self):
        self._q_val = float(q.to("C").magnitude)
        return self

    def overvoltage(self, V_bias: float, V_br: float) -> float:
        """Overvoltage ΔV = V_bias - V_br."""
        return V_bias - V_br

    def reset_time(self, C_j: float) -> float:
        """Reset time τ_reset = R_q × C_j."""
        return self.quenching_resistance * C_j


@dataclass
class AvalancheResult:
    triggered: bool
    Q_av: float


def avalanche_charge(V_bias: float, V_br: float, C_j: float | Quantity) -> AvalancheResult:
    """Compute avalanche charge Q = C_j * overvoltage in Geiger mode."""
    c_val = float(C_j.magnitude) if hasattr(C_j, "magnitude") else float(C_j)
    if V_bias > V_br:
        delta_V = V_bias - V_br
        Q_av = c_val * delta_V
        return AvalancheResult(triggered=True, Q_av=Q_av)
    return AvalancheResult(triggered=False, Q_av=0.0)


def effective_gain(Q_av: float) -> float:
    """Compute effective gain as Q_av / q."""
    q_val = 1.602176634e-19
    return Q_av / q_val
