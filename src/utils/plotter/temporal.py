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
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Carrier Position Trajectories", fontsize=12, pad=12)
        for i, traj in enumerate(trajectories):
            t = times[i] if times is not None else np.arange(len(traj))
            ax.plot(t, traj * 1e4, lw=0.5, alpha=0.7)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Position (µm)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("carrier_trajectories.png", plt)


class JitterPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "jitter"

    def plot(self, detection_times: np.ndarray,
             bins: int = 50) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Timing Jitter (SPTR) Histogram", fontsize=12, pad=12)
        ax.hist(detection_times * 1e12, bins=bins, alpha=0.7, edgecolor="k")
        ax.set_xlabel("Detection Time (ps)")
        ax.set_ylabel("Counts")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("timing_jitter.png", plt)


class JitterHistogramPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "jitter_histogram"

    def plot(self, detection_times: np.ndarray,
             bins: int = 50,
             fwhm: float | None = None,
             sigma: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        t_ps = detection_times * 1e12
        ax.hist(t_ps, bins=bins, alpha=0.7, edgecolor="k", density=True)
        label_parts = []
        if fwhm is not None and np.isfinite(fwhm):
            label_parts.append(f"FWHM = {fwhm*1e12:.1f} ps")
        if sigma is not None and np.isfinite(sigma):
            label_parts.append(f"σ = {sigma*1e12:.1f} ps")
        if label_parts:
            ax.set_title("Timing Jitter (SPTR)\n" + "  ".join(label_parts), fontsize=12, pad=12)
        else:
            ax.set_title("Timing Jitter (SPTR)", fontsize=12, pad=12)
        ax.set_xlabel("Detection Time (ps)")
        ax.set_ylabel("Probability Density")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("timing_jitter_histogram.png", plt)


class PopulationPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "population"

    def plot(self, time: np.ndarray, n_electrons: np.ndarray,
             n_holes: np.ndarray) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Carrier Population Dynamics", fontsize=12, pad=12)
        ax.semilogy(time, n_electrons, label="Electrons", lw=2)
        ax.semilogy(time, n_holes, label="Holes", lw=2)
        ax.semilogy(time, n_electrons + n_holes, "k--",
                    label="Total", lw=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Carrier Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("carrier_population.png", plt)


class AvalancheCurrentPulsePlotter(BasePlotter):
    """Plot avalanche current pulse I(t) from self-consistent simulation."""

    @property
    def name(self) -> str:
        return "avalanche_current_pulse"

    def plot(self, time: np.ndarray, current: np.ndarray,
             Vbr: float | None = None, Vex: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time * 1e12, current * 1e6, lw=2, color="#1f77b4")
        ax.fill_between(time * 1e12, current * 1e6, alpha=0.15, color="#1f77b4")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Avalanche Current (µA)")
        title = "Avalanche Current Pulse"
        parts = []
        if Vbr is not None:
            parts.append(f"Vbr = {Vbr:.1f} V")
        if Vex is not None:
            parts.append(f"Vex = {Vex:.1f} V")
        if parts:
            title += f"  ({', '.join(parts)})"
        ax.set_title(title, fontsize=12, pad=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("avalanche_current_pulse.png", plt)


class DeadSpaceDistributionPlotter(BasePlotter):
    """Histogram of dead-space values from Monte Carlo carriers."""

    @property
    def name(self) -> str:
        return "dead_space_dist"

    def plot(self, dead_spaces: np.ndarray, carrier_type: str = "all",
             bins: int = 30) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ds_um = dead_spaces * 1e4
        ax.hist(ds_um, bins=bins, alpha=0.7, edgecolor="k", color="#1f77b4")
        mean_ds = np.mean(ds_um)
        std_ds = np.std(ds_um)
        ax.axvline(mean_ds, color="r", ls="--", lw=1.5,
                   label=f"μ = {mean_ds:.3f} µm")
        ax.axvline(mean_ds + std_ds, color="orange", ls=":", lw=1.5,
                   label=f"σ = {std_ds:.3f} µm")
        ax.axvline(mean_ds - std_ds, color="orange", ls=":", lw=1.5)
        ax.set_xlabel("Dead Space (µm)")
        ax.set_ylabel("Counts")
        ax.set_title(f"Dead-Space Distribution ({carrier_type})", fontsize=12, pad=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("dead_space_distribution.png", plt)


class QuenchingWaveformPlotter(BasePlotter):
    """Plot quenching waveform: Vspad(t) and I(t) from self-consistent loop."""

    @property
    def name(self) -> str:
        return "quenching_waveform"

    def plot(self, time: np.ndarray, Vspad: np.ndarray,
             current: np.ndarray, Vbr: float | None = None,
             Vbias: float | None = None) -> None:
        plt = self._import()
        fig, ax1 = plt.subplots(figsize=(10, 5))
        t_ps = time * 1e12

        color1 = "#1f77b4"
        ax1.plot(t_ps, Vspad, color=color1, lw=2, label="V_spad")
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("SPAD Voltage (V)", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        if Vbr is not None:
            ax1.axhline(y=Vbr, color=color1, ls=":", alpha=0.4,
                        label=f"Vbr = {Vbr:.1f} V")
        if Vbias is not None:
            ax1.axhline(y=Vbias, color="gray", ls="--", alpha=0.3,
                        label=f"Vbias = {Vbias:.1f} V")

        ax2 = ax1.twinx()
        color2 = "#d62728"
        ax2.plot(t_ps, current * 1e6, color=color2, lw=2,
                 ls="--", label="I_av")
        ax2.set_ylabel("Avalanche Current (µA)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        ax1.set_title("Quenching Waveform", fontsize=12, pad=12)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("quenching_waveform.png", plt)
