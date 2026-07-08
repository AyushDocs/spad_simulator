from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils._exceptions import ConfigError
from ..utils._logging import get_logger

log = get_logger("grid")


@dataclass
class Grid1D:
    L: float
    N: int

    x: np.ndarray = field(init=False)
    dx: float = field(init=False)
    no_of_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        if self.L <= 0:
            raise ConfigError(f"Device length must be positive, got {self.L}")
        if self.N < 3:
            raise ConfigError(f"Need at least 3 grid nodes, got {self.N}")

        self.dx = self.L / (self.N - 1)
        self.x = np.linspace(0, self.L, self.N)
        self.no_of_nodes = self.N
        log.info("Grid1D  L=%.1f um  N=%d  dx=%.2f nm",
                 self.L * 1e4, self.N, self.dx * 1e7)

    def gradient(self, phi: np.ndarray) -> np.ndarray:
        return -np.gradient(phi, self.dx)
