"""DCR vs temperature and PDP vs temperature plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class DCRvsTempPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "dcr_vs_temp"

    def plot(self, temperatures: np.ndarray, DCR: np.ndarray,
             Vex: float | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(temperatures, DCR + 1e-10, "o-", lw=2)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("DCR (cps)")
        title = "Dark Count Rate vs Temperature"
        if Vex is not None:
            title += f" (Vex = {Vex:.1f} V)"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("dcr_vs_temperature.png")


class PDPvsTempPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "pdp_vs_temp"

    def plot(self, temperatures: np.ndarray, pdp_dict: dict,
             wavelengths_nm: np.ndarray | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(pdp_dict)))
        for (lam, pdp), c in zip(sorted(pdp_dict.items()), colors):
            ax.plot(temperatures, pdp * 100, "o-", lw=2, color=c,
                    label=f"{lam:.0f} nm")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("PDP (%)")
        ax.set_title("PDP vs Temperature", fontsize=10)
        ax.legend(fontsize=8, title="Wavelength")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        self._save("pdp_vs_temperature.png")
