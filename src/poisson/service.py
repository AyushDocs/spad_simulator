"""Poisson + field service."""

from __future__ import annotations

import numpy as np

from ..core.grid import Grid1D
from ..utils._logging import get_logger
from .solver import PoissonSolver
from .field import DepletionWidth

log = get_logger("poisson")


class PoissonService:
    """Facade over PoissonSolver and field computation."""

    RAMP_STEP = 2.0

    def __init__(self, poisson_solver: PoissonSolver, grid: Grid1D,
                 depletion: DepletionWidth) -> None:
        self.poisson = poisson_solver
        self.grid = grid
        self.depletion = depletion

    def solve(self, Vbias: float, phi_n: float | None = None,
              phi_p: float = 0.0,
              guess: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve Poisson's equation for a given bias voltage.

        Returns
        -------
        phi : np.ndarray
            Potential array in V.
        E : np.ndarray
            Electric field array in V/cm.
        info : dict
            Solver diagnostic info.
        """
        if guess is not None:
            phi, info = self.poisson.solve(Vbias, phi_n, phi_p, guess=guess)
            E_raw = self.grid.gradient(phi)
            return phi, E_raw, info

        phi_guess = self.poisson._step_guess(Vbias)
        phi, info = self.poisson.solve(Vbias, phi_n, phi_p, guess=phi_guess)
        E_raw = self.grid.gradient(phi)
        return phi, E_raw, info

    def depletion_width(self, Vbias: float) -> tuple[float, float, float]:
        _, E = self.solve(Vbias)[:2]
        return self.depletion.from_field(E)
