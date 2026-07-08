from __future__ import annotations

from typing import List

import numpy as np

from ..core.grid import Grid1D
from ..core.constants import q
from ..transport.carrier import Carrier


class ParticleMesh:
    """
    Particle-in-cell coupling.

    Deposition:
        Scatter carrier charges to grid nodes via linear (CIC) weighting.
    Interpolation:
        Gather electric field from grid nodes to each carrier.
    """

    def __init__(self, grid: Grid1D) -> None:
        self.grid = grid

    def deposit_charge(self, carriers: List[Carrier]) -> np.ndarray:
        N = self.grid.no_of_nodes
        x = self.grid.x
        dx = self.grid.dx
        rho = np.zeros(N)
        for c in carriers:
            if not c.alive:
                continue
            idx = int(np.clip(np.floor((c.x - x[0]) / dx), 0, N - 2))
            w_right = (c.x - x[idx]) / dx
            w_left = 1.0 - w_right
            charge = q if c.is_hole else -q
            rho[idx] += charge * w_left / dx
            rho[idx + 1] += charge * w_right / dx
        return rho

    def interpolate_field(self, carriers: List[Carrier],
                          E_grid: np.ndarray) -> np.ndarray:
        return np.array([np.interp(c.x, self.grid.x, E_grid)
                         if c.alive else 0.0 for c in carriers])
