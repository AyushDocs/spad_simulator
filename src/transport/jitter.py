"""Timing jitter model."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from pydantic.dataclasses import dataclass

from ..core.constants import q
from ..utils._logging import get_logger

log = get_logger("transport.jitter")


class JitterModel(BaseModel):
    """Timing jitter model for SPADs.

    Accounts for:
    - Transit time jitter
    - Avalanche build-up time jitter
    - Electronic jitter
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transit_time: float = 1e-12  # transit time (s)
    avalanche_buildup_time: float = 50e-12  # avalanche buildup time (s)
    electronic_jitter: float = 10e-12  # electronic jitter (s)
    absorption_depth: float = 1e-5  # absorption depth (cm)

    _q_val: float = PrivateAttr()

    @model_validator(mode="after")
    def _init_q(self):
        self._q_val = float(q.to("C").magnitude)
        return self

    def compute(self, wavelength: float) -> float:
        """Compute total timing jitter.

        Returns total jitter in seconds.
        """
        # Transit time jitter (sqrt(2 * D * t) / v)
        transit_jitter = self.transit_time / np.sqrt(2)

        # Avalanche buildup jitter (sqrt(t_build))
        avalanche_jitter = self.avalanche_buildup_time / np.sqrt(2)

        # Total jitter (quadrature sum)
        total = np.sqrt(transit_jitter**2 +
                        avalanche_jitter**2 +
                        self.electronic_jitter**2)
        return total


class TimingJitter:
    """Timing jitter statistics compatibility class for tests."""

    @staticmethod
    def extract_detection_times(ensemble_result: dict) -> np.ndarray:
        return ensemble_result.get("t_detect", np.array([]))

    @staticmethod
    def statistics(t_detect: np.ndarray) -> dict:
        if len(t_detect) == 0:
            return {"mean": np.nan, "std": np.nan,
                    "min": np.nan, "max": np.nan, "N": 0}
        return {"mean": float(np.mean(t_detect)),
                "std": float(np.std(t_detect)),
                "min": float(np.min(t_detect)),
                "max": float(np.max(t_detect)),
                "N": len(t_detect)}

    @staticmethod
    def percentiles(t_detect: np.ndarray,
                    p: tuple = (10, 50, 90)) -> dict:
        if len(t_detect) == 0:
            return {f"t{pp}": np.nan for pp in p}
        vals = np.percentile(t_detect, list(p))
        return {f"t{pp}": float(v) for pp, v in zip(p, vals)}

    @staticmethod
    def fwhm(t_detect: np.ndarray, bins: int = 100) -> float:
        """Full width at half maximum of detection time histogram."""
        if len(t_detect) < 2:
            return np.nan
        counts, edges = np.histogram(t_detect, bins=bins)
        half_max = counts.max() / 2.0
        above = edges[:-1][counts >= half_max]
        if len(above) < 2:
            return np.nan
        return float(above[-1] - above[0])
