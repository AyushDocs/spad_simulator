"""Device structure and doping plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class DeviceStructurePlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "device_structure"

    def plot(self, x: np.ndarray, mat_names: np.ndarray,
             doping: np.ndarray, net_doping: np.ndarray) -> None:
        plt = self._import()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        x_um = x * 1e4
        mat_colors = {"InP": "#1f77b4", "InGaAs": "#ff7f0e",
                      "InGaAsP": "#2ca02c", "Si": "#d62728"}
        unique_mats = sorted(set(mat_names))
        for mat in unique_mats:
            mask = mat_names == mat
            ax1.fill_between(x_um, 0, np.where(mask, 1, 0),
                             alpha=0.3, color=mat_colors.get(mat, "gray"))
        ax1.set_ylabel("Material")
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        for mat in unique_mats:
            mid = np.mean(x_um[mat_names == mat]) if np.any(mat_names == mat) else 0
            ax1.text(mid, 0.5, mat, ha="center", va="center",
                     fontsize=8, fontweight="bold")

        ax2.semilogy(x_um, np.abs(net_doping + 1e10), label="|Net Doping|")
        ax2.semilogy(x_um, doping, "--", label="Doping", alpha=0.6)
        ax2.set_xlabel("Depth (µm)")
        ax2.set_ylabel("Doping (cm⁻³)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Device Schematic and Net Doping Profile", fontsize=12)
        plt.tight_layout()
        self._save("device_structure.png", plt)


class DopingPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "doping"

    def plot(self, x: np.ndarray, nd: np.ndarray, na: np.ndarray) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Doping Concentrations vs Depth", fontsize=12, pad=12)
        x_um = x * 1e4
        ax.semilogy(x_um, np.abs(nd) + 1e10, label="Donors")
        ax.semilogy(x_um, np.abs(na) + 1e10, label="Acceptors")
        ax.semilogy(x_um, np.abs(nd - na) + 1e10, "k--",
                    label="|Net|", lw=1.5)
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Doping (cm⁻³)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("doping_profile.png", plt)
