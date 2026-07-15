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

        # Find contiguous depleted regions
        edges = np.diff(mask.astype(np.int8))
        rises = np.where(edges == 1)[0] + 1
        falls = np.where(edges == -1)[0] + 1

        if mask[0]:
            rises = np.concatenate([[0], rises])
        if mask[-1]:
            falls = np.concatenate([falls, [len(mask)]])

        if len(rises) == 0:
            return 0.0, 0.0, 0.0

        # First contiguous depletion region from the left
        xl = float(x[rises[0]])
        i_end = min(falls[0], len(x) - 1)
        xr = float(x[i_end - 1]) if i_end > rises[0] else xl
        return xl, xr, xr - xl
