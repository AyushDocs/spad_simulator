from __future__ import annotations

import numpy as np


class TimingJitter:
    """
    Timing jitter statistics from Monte Carlo avalanche histories.

    Extracts:
        mean build-up time mu_t
        jitter (std)       sigma_t
        percentile thresholds t_10, t_50, t_90
    """

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
