"""Poisson + field service with breakdown detection."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.grid import Grid1D
from .solver import PoissonSolver
from .field import DepletionWidth
from ..avalanche.breakdown import BreakdownVoltage, BreakdownCriterion


class PoissonService:
    """
    Facade over PoissonSolver, field computation, and breakdown detection.

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
              guess: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Solve Poisson at *Vbias* with optional initial *guess*.

        When no *guess* is provided the solver ramps from 0 V in 2 V
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

    def depletion_width(self, Vbias: float) -> Tuple[float, float, float]:
        _, E = self.solve(Vbias)[:2]
        return self.depletion.from_field(E)

    def find_breakdown(self, V_start: float, V_max: float,
                       criterion: BreakdownCriterion,
                       V_step: float = 0.1,
                       phi_n: float | None = None, phi_p: float = 0.0
                       ) -> Tuple[float | None, List[dict]]:
        bv = BreakdownVoltage(self.poisson, self.grid, criterion, V_step)
        return bv.find(V_start, V_max, phi_n, phi_p)
