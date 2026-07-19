from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..utils._exceptions import ConfigError
from ..utils.pydantic_types import NDArray

class Grid1D(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    L: float
    N: int

    dx: float = 0.0
    x: NDArray = np.array([])
    no_of_nodes: int = 0

    @field_validator("L")
    @classmethod
    def _L_positive(cls, v: float) -> float:
        if v <= 0:
            raise ConfigError(f"Device length must be positive, got {v}")
        return v

    @field_validator("N")
    @classmethod
    def _N_min(cls, v: int) -> int:
        if v < 3:
            raise ConfigError(f"Need at least 3 grid nodes, got {v}")
        return v

    @model_validator(mode="after")
    def _compute_grid(self):
        self.dx = self.L / (self.N - 1)
        self.x = np.linspace(0, self.L, self.N)
        self.no_of_nodes = self.N
        return self

    def gradient(self, phi: np.ndarray) -> np.ndarray:
        return -np.gradient(phi, self.dx)
