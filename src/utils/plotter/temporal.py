"""Trajectory, jitter, and population plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class TrajectoryPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "trajectory"

    def plot(self, trajectories: list[np.ndarray],
             times: list[np.ndarray] | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, traj in enumerate(trajectories):
            t = times[i] if times is not None else np.arange(len(traj))
            ax.plot(t, traj * 1e4, lw=0.5, alpha=0.7)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Position (µm)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("carrier_trajectories.png")


class JitterPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "jitter"

    def plot(self, detection_times: np.ndarray,
             bins: int = 50) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(detection_times * 1e12, bins=bins, alpha=0.7, edgecolor="k")
        ax.set_xlabel("Detection Time (ps)")
        ax.set_ylabel("Counts")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("timing_jitter.png")


class JitterHistogramPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "jitter_histogram"

    def plot(self, detection_times: np.ndarray,
             bins: int = 50,
             fwhm: float | None = None,
             sigma: float | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        t_ps = detection_times * 1e12
        ax.hist(t_ps, bins=bins, alpha=0.7, edgecolor="k", density=True)
        label_parts = []
        if fwhm is not None and np.isfinite(fwhm):
            label_parts.append(f"FWHM = {fwhm*1e12:.1f} ps")
        if sigma is not None and np.isfinite(sigma):
            label_parts.append(f"σ = {sigma*1e12:.1f} ps")
        if label_parts:
            ax.set_title("Timing Jitter (SPTR)\n" + "  ".join(label_parts), fontsize=10)
        else:
            ax.set_title("Timing Jitter (SPTR)", fontsize=10)
        ax.set_xlabel("Detection Time (ps)")
        ax.set_ylabel("Probability Density")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("timing_jitter_histogram.png")


class PopulationPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "population"

    def plot(self, time: np.ndarray, n_electrons: np.ndarray,
             n_holes: np.ndarray) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(time, n_electrons, label="Electrons", lw=2)
        ax.semilogy(time, n_holes, label="Holes", lw=2)
        ax.semilogy(time, n_electrons + n_holes, "k--",
                    label="Total", lw=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Carrier Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("carrier_population.png")
