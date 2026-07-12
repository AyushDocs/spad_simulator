from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..utils.loaders import AbsorptionData


class AbsorptionModel(ABC):
    """Pluggable optical absorption coefficient model."""

    @abstractmethod
    def coefficient(self, wavelength: float) -> float: ...


class InterpolatedAbsorption(AbsorptionModel):
    """Linear interpolation of tabulated absorption data."""

    def __init__(self, data: AbsorptionData) -> None:
        self._data = data

    def coefficient(self, wavelength: float) -> float:
        return float(
            np.interp(
                wavelength, self._data.wavelengths, self._data.alphas, left=0.0, right=0.0
            )
        )
