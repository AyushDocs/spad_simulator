"""Dark current, DCR, I-V, and comprehensive I-V plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class DarkCurrentPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "dark_current"

    def plot(self, Vbias: np.ndarray, I_dark: np.ndarray,
             J_th: np.ndarray | None = None,
             J_btbt: np.ndarray | None = None,
             J_tat: np.ndarray | None = None) -> None:
        self._import()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.semilogy(Vbias, np.abs(I_dark) + 1e-20)
        ax1.set_xlabel("Bias (V)")
        ax1.set_ylabel("Dark Current (A)")
        ax1.grid(True, alpha=0.3)

        if J_th is not None and J_btbt is not None and J_tat is not None:
            ax2.semilogy(Vbias, np.abs(J_th) + 1e-20, label="Thermal", lw=2)
            ax2.semilogy(Vbias, np.abs(J_btbt) + 1e-20, label="BTBT", lw=2)
            ax2.semilogy(Vbias, np.abs(J_tat) + 1e-20, label="TAT", lw=2)
            total = np.abs(J_th + J_btbt + J_tat) + 1e-20
            ax2.semilogy(Vbias, total, "k--", label="Total", lw=1.5)
            ax2.legend(fontsize=8)
        ax2.set_xlabel("Bias (V)")
        ax2.set_ylabel("Current Density (A/cm²)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save("dark_current_vs_bias.png")


class DCRPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "dcr"

    def plot(self, Vbias: np.ndarray, DCR: np.ndarray) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(Vbias, DCR + 1e-10)
        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("DCR (cps)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("dcr_vs_bias.png")


class IVCharacteristicPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "iv_characteristic"

    def plot(self, Vbias: np.ndarray, I_dark: np.ndarray,
             I_light: np.ndarray | None = None,
             optical_power: float | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(Vbias, np.abs(I_dark) + 1e-20, "b-",
                    label="Dark", lw=2)
        if I_light is not None:
            label = "Illuminated"
            if optical_power is not None:
                label += f" ({optical_power*1e6:.0f} µW)"
            ax.semilogy(Vbias, np.abs(I_light) + 1e-20, "r--",
                        label=label, lw=2)
        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("Current (A)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        self._save("iv_characteristic.png")


class ComprehensiveIVPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "comprehensive_iv"

    def plot(self, Vbias: np.ndarray, I_dark: np.ndarray,
             I_photo_primary: np.ndarray | None = None,
             I_total_illuminated: np.ndarray | None = None,
             gain: np.ndarray | None = None,
             Vbr: float | None = None) -> None:
        self._import()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        eps = 1e-20
        ax1.semilogy(Vbias, np.abs(I_dark) + eps, "b-",
                     label="Dark (primary)", lw=2)
        if I_photo_primary is not None:
            ax1.semilogy(Vbias, np.abs(I_photo_primary) + eps, "orange",
                         label="Photo-primary", lw=2, ls="--")
        if I_total_illuminated is not None:
            ax1.semilogy(Vbias, np.abs(I_total_illuminated) + eps, "r-.",
                         label="Total illuminated", lw=2)
        if Vbr is not None:
            ax1.axvline(x=Vbr, color="k", ls=":", alpha=0.5,
                        label=f"Vbr = {Vbr:.1f} V")
        ax1.set_xlabel("Bias (V)")
        ax1.set_ylabel("Current (A)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(Vbias, np.abs(I_dark) + eps, "b-",
                     label="Dark (primary)", lw=2)
        if I_photo_primary is not None:
            ax2.semilogy(Vbias, np.abs(I_photo_primary) + eps, "orange",
                         label="Photo-primary", lw=2, ls="--")
        if I_total_illuminated is not None:
            ax2.semilogy(Vbias, np.abs(I_total_illuminated) + eps, "r-.",
                         label="Total illuminated", lw=2)
        if gain is not None:
            ax_twin = ax2.twinx()
            ax_twin.plot(Vbias, gain, "g:", lw=1.5, alpha=0.7)
            ax_twin.set_ylabel("Gain M", color="g")
            ax_twin.tick_params(axis="y", labelcolor="g")
        if Vbr is not None:
            ax2.axvline(x=Vbr, color="k", ls=":", alpha=0.5)
        ax2.set_xlabel("Bias (V)")
        ax2.set_ylabel("Current (A)")
        ax2.set_xlim(0, Vbr + 5 if Vbr else Vbias[-1])
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Comprehensive I-V Characteristic", fontsize=12)
        plt.tight_layout()
        self._save("comprehensive_iv.png")
