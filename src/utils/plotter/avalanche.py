"""Trigger probability, afterpulsing, and excess noise plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class TriggerProbabilityPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "trigger_probability"

    def plot(self, x: np.ndarray, Pe: np.ndarray,
             Ph: np.ndarray | None = None,
             V_list: list[float] | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        x_um = x * 1e4
        if V_list is not None and Pe.ndim > 1:
            for i, (Pe_i, V) in enumerate(zip(Pe, V_list)):
                ax.plot(x_um, Pe_i, label=f"Pe V={V:.1f} V",
                        color=plt.cm.viridis(i / len(Pe)), lw=1.5)
                if Ph is not None and Ph.ndim > 1:
                    ax.plot(x_um, Ph[i], "--",
                            color=plt.cm.viridis(i / len(Ph)), lw=1.5)
        else:
            ax.plot(x_um, Pe, label="Pe")
            if Ph is not None:
                ax.plot(x_um, Ph, "--", label="Ph")
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Trigger Probability")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("trigger_probability.png")


class AfterpulsingPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "afterpulsing"

    def plot(self, holdoff: np.ndarray, P_ap: np.ndarray,
             N_T: float | None = None, tau_c: float | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(holdoff * 1e6, P_ap * 100, lw=2)
        ax.set_xlabel("Holdoff Time (µs)")
        ax.set_ylabel("Afterpulsing Probability (%)")
        title = "Afterpulsing vs Holdoff"
        if N_T is not None and tau_c is not None:
            title += f"\nN_T={N_T:.1e} cm⁻³, τ_c={tau_c*1e6:.1f} µs"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("afterpulsing_vs_holdoff.png")


class ExcessNoisePlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "excess_noise"

    def plot(self, M: np.ndarray, F: np.ndarray,
             k_eff: float | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(M, F, lw=2)
        ax.axhline(y=1.0, color="k", ls="--", alpha=0.3, label="F=1 (shot noise)")
        if k_eff is not None:
            ax.plot(M, k_eff * M + (1.0 - k_eff) * (2.0 - 1.0 / np.clip(M, 1, None)),
                    ":", color="gray", alpha=0.6,
                    label=f"k_eff={k_eff:.2f}")
        ax.set_xlabel("Multiplication M")
        ax.set_ylabel("Excess Noise Factor F(M)")
        ax.set_title("McIntyre Excess Noise", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("excess_noise.png")
