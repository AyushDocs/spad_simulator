"""Potential, electric field, and breakdown sweep plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class PotentialProfilePlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "potential_profile"

    def plot(self, x: np.ndarray, phi: np.ndarray,
             V_list: list[float] | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        x_um = x * 1e4
        if V_list is not None:
            for i, (phi_i, V) in enumerate(zip(phi, V_list)):
                ax.plot(x_um, phi_i, label=f"V = {V:.1f} V",
                        color=plt.cm.viridis(i / len(phi)))
        else:
            ax.plot(x_um, phi)
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Potential φ (V)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("potential_profile.png")


class ElectricFieldPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "electric_field"

    def plot(self, x: np.ndarray, E: np.ndarray,
             V_list: list[float] | None = None,
             Vbr: float | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        x_um = x * 1e4

        if V_list is not None:
            Vex_list = (
                [v - Vbr for v in V_list]
                if Vbr is not None and Vbr > 0
                else V_list
            )
            for i, (E_i, Vex) in enumerate(zip(E, Vex_list)):
                ax.plot(x_um, -E_i / 1e5, label=f"Vex = {Vex:.0f} V",
                        color=plt.cm.viridis(i / len(E)))
        else:
            ax.plot(x_um, -np.atleast_1d(E) / 1e5)

        ax.set_xlim(2, 5)
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Electric Field (×10⁵ V/cm)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("electric_field_profile.png")


class BreakdownSweepPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "breakdown_sweep"

    def plot(self, Vbias: np.ndarray, Pe_max: np.ndarray) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Vbias, Pe_max, "o-", lw=2)
        ax.axhline(y=0.99, color="r", ls="--", alpha=0.5, label="Pe=0.99")
        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("Max P_e")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("breakdown_sweep.png")
