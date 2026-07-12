from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .layer import Layer


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """Specification for a single doping layer."""
    type: str
    A: float
    m: float
    x0: float
    x_start: float
    x_end: float


class DopingProfile:
    """
    Piecewise exponential doping profile assembled from ``Layer`` s.

        N(x) = A * exp(m * (x - x0))
    """

    def __init__(self, layer_specs: List[LayerSpec]) -> None:
        self._specs = list(layer_specs)

    @classmethod
    def _from_layers(cls, layers: List[Layer]) -> DopingProfile:
        specs = []
        x_start = 0.0
        for lyr in layers:
            x_end = x_start + lyr.thickness
            x0 = lyr.doping_x0 if lyr.doping_x0 is not None else x_start
            specs.append(LayerSpec(
                type=lyr.doping_type,
                A=lyr.doping_A,
                m=lyr.doping_m,
                x0=x0,
                x_start=x_start,
                x_end=x_end,
            ))
            x_start = x_end
        return cls(specs)

    def _eval(self, x: float | np.ndarray,
              dtype: str) -> float | np.ndarray:
        scalar_input = np.isscalar(x)
        if scalar_input:
            x = np.array([x])
        val = np.zeros_like(x, dtype=float)
        for L in self._specs:
            if L.type != dtype:
                continue
            mask = (x >= L.x_start) & (x <= L.x_end)
            if not np.any(mask):
                continue
            arg = L.m * (x[mask] - L.x0)
            arg = np.clip(arg, -700, 700)
            val[mask] += L.A * np.exp(arg)
        return float(val[0]) if scalar_input else val

    def nd(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._eval(x, "donor")

    def na(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._eval(x, "acceptor")

    def net_doping(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.nd(x) - self.na(x)
