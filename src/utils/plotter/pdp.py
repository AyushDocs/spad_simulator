"""PDP spectrum, PDP vs excess voltage, and PDE plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class PDPPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "pdp"

    def plot(self, wavelengths: np.ndarray, pdp: np.ndarray,
             Vex_list: list[float] | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        if Vex_list is not None and pdp.ndim > 1:
            for i, (pdp_i, Vex) in enumerate(zip(pdp, Vex_list)):
                ax.plot(wavelengths, pdp_i * 100,
                        label=f"Vex={Vex:.1f}V",
                        lw=2, color=plt.cm.viridis(i / len(pdp)))
        else:
            ax.plot(wavelengths, np.atleast_1d(pdp) * 100, lw=2)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("PDP (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        self._save("pdp_spectrum.png")


class PDPVsExcessVoltagePlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "pdp_vs_vex"

    def plot(self, Vex: np.ndarray, pdp_dict: dict,
             wavelengths_nm: np.ndarray | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(pdp_dict)))
        for (lam, pdp), c in zip(sorted(pdp_dict.items()), colors):
            ax.plot(Vex, pdp * 100, "o-", lw=2, color=c,
                    label=f"{lam:.0f} nm")
        ax.set_xlabel("Excess Voltage (V)")
        ax.set_ylabel("PDP (%)")
        ax.legend(fontsize=8, title="Wavelength")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        self._save("pdp_vs_excess_voltage.png")


class PDEPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "pde"

    def plot(self, Vbias: np.ndarray, PDE: np.ndarray) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Vbias, PDE * 100, lw=2)
        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("PDE (%)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("pde_vs_bias.png")
