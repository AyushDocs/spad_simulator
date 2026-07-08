from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.grid import Grid1D
from ..utils._logging import get_logger

log = get_logger("field")


class DepletionWidth:
    """Extract depletion width from field."""

    def __init__(self, grid: Grid1D) -> None:
        self.grid = grid

    def from_field(
        self, E: np.ndarray, eps_field: float = 1e4
    ) -> Tuple[float, float, float]:
        mask = np.abs(E) > eps_field
        x = self.grid.x
        if not np.any(mask):
            return 0.0, 0.0, 0.0
        xl = float(x[np.argmax(mask)])
        xr = float(x[len(x) - 1 - np.argmax(mask[::-1])])
        return xl, xr, xr - xl
