"""Particle-in-cell charge deposition and field interpolation."""

from __future__ import annotations

import numpy as np
from pydantic.dataclasses import dataclass

from ..core.grid import Grid1D
from ..core.constants import q
from ..utils.pydantic_types import NDArray


@dataclass(config=dict(arbitrary_types_allowed=True))
class Carrier:
    """Represents a single carrier (electron or hole)."""
    x: float
    typ: str = "electron"
    v: float = 0.0
    dead_space: float = 0.0
    alive: bool = True
    E: float = 0.0
    age: float = 0.0
    ionized: bool = False

    @property
    def is_electron(self) -> bool:
        return self.typ == "electron"

    @property
    def is_hole(self) -> bool:
        return self.typ == "hole"

    @property
    def dead_space_remaining(self) -> float:
        return self.dead_space

    @dead_space_remaining.setter
    def dead_space_remaining(self, val: float) -> None:
        self.dead_space = val

    def reset_dead_space(self, ld: float) -> None:
        self.dead_space = ld
        self.ionized = True

    def move(self, dx: float, dt: float) -> None:
        self.x += dx
        self.age += dt

    def exit_check(self, x_left: float, x_right: float) -> None:
        if self.x < x_left or self.x > x_right:
            self.alive = False


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
        self._q_val = float(q.to("C").magnitude)

    def deposit_charge(self, carriers: list[Carrier]) -> np.ndarray:
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
            charge = self._q_val if c.is_hole else -self._q_val
            rho[idx] += charge * w_left / dx
            rho[idx + 1] += charge * w_right / dx
        return rho

    def interpolate_field(self, carriers: list[Carrier],
                          E_grid: np.ndarray) -> np.ndarray:
        return np.array([np.interp(c.x, self.grid.x, E_grid)
                         if c.alive else 0.0 for c in carriers])
