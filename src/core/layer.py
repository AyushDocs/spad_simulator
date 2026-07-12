from __future__ import annotations

from typing import Literal

import pint
from pydantic import BaseModel, ConfigDict, field_validator

from .units import Q_

DopingType = Literal["donor", "acceptor"]


class Layer(BaseModel, frozen=True):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    thickness: float
    doping_type: DopingType = "acceptor"
    doping_A: float = 1e15
    doping_m: float = 0.0
    doping_x0: float | None = None
    material: str = "Si"

    @field_validator("thickness")
    @classmethod
    def _thickness_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Layer thickness must be positive, got {v}")
        return v

    @field_validator("doping_A")
    @classmethod
    def _doping_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"doping_A must be non-negative, got {v}")
        return v

    @property
    def is_donor(self) -> bool:
        return self.doping_type == "donor"

    @property
    def is_acceptor(self) -> bool:
        return self.doping_type == "acceptor"
