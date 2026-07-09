"""Poisson + field service."""

from __future__ import annotations

import numpy as np

from ..core.grid import Grid1D
from ..utils._logging import get_logger
from .solver import PoissonSolver
from .field import DepletionWidth

log = get_logger("poisson")


class PoissonService:
    """Facade over PoissonSolver and field computation.

    No implicit state between calls — the caller provides initial
    guesses explicitly.
    """

    def __init__(self, poisson_solver: PoissonSolver, grid: Grid1D,
                 depletion: DepletionWidth) -> None:
        self.poisson = poisson_solver
        self.grid = grid
        self.depletion = depletion

    def solve(self, Vbias: float, phi_n: float | None = None,
              phi_p: float = 0.0,
              guess: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve Poisson at *Vbias* with optional initial *guess*.

        When no *guess* is provided the solver ramps from 0 V in 2 V
        steps to build up a physical initial condition.
        """
        if guess is not None:
            phi, info = self.poisson.solve(Vbias, phi_n, phi_p, guess=guess)
        else:
            n_ramp = max(1, int(abs(Vbias) / 2.0))
            for Vr in np.linspace(0.0, Vbias, n_ramp):
                phi, info = self.poisson.solve(
                    Vr, phi_n, phi_p,
                    guess=(phi if Vr > 0.0 else None))
        E = self.grid.gradient(phi)
        return phi, E, info

    def depletion_width(self, Vbias: float) -> tuple[float, float, float]:
        _, E = self.solve(Vbias)[:2]
        return self.depletion.from_field(E)
