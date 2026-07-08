from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DopingType = Literal["donor", "acceptor"]


@dataclass(frozen=True)
class Layer:
    thickness: float
    doping_type: DopingType = "acceptor"
    doping_A: float = 1e15
    doping_m: float = 0.0
    doping_x0: float | None = None
    material: str = "Si"

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError(
                f"Layer thickness must be positive, got {self.thickness}")
        if self.doping_A < 0:
            raise ValueError(
                f"doping_A must be non-negative, got {self.doping_A}")

    @property
    def is_donor(self) -> bool:
        return self.doping_type == "donor"

    @property
    def is_acceptor(self) -> bool:
        return self.doping_type == "acceptor"
