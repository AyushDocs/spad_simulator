"""LRU field cache for SPADSimulator."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


class FieldCache:
    """Fixed-size LRU cache of (phi, E, Pe, Ph, xl, xr) tuples keyed by bias voltage."""

    def __init__(self, maxlen: int = 200) -> None:
        self._cache: OrderedDict[
            float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]
        ] = OrderedDict()
        self._maxlen = maxlen

    def get(self, Vbias: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float] | None:
        Vkey = round(Vbias, 6)
        cached = self._cache.get(Vkey)
        if cached is not None:
            self._cache.move_to_end(Vkey)
        return cached

    def put(self, Vbias: float, value: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]) -> None:
        Vkey = round(Vbias, 6)
        self._cache[Vkey] = value
        if len(self._cache) > self._maxlen:
            self._cache.popitem(last=False)

    def interpolate_guess(self, Vbias: float) -> np.ndarray | None:
        """Return nearest-neighbor phi guess if close enough, else None."""
        if not self._cache:
            return None
        biases = np.array(list(self._cache.keys()))
        nearest = biases[np.argmin(np.abs(biases - Vbias))]
        if abs(nearest - Vbias) < 10.0:
            return self._cache[nearest][0]
        return None

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
