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
             J_tat: np.ndarray | None = None,
             Vbr: float | None = None,
             gain: np.ndarray | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        title = f"Dark Current Density vs Excess Voltage (Vbr = {Vbr:.1f} V)" if Vbr else "Dark Current Density vs Excess Voltage"
        ax.set_title(title, fontsize=12, pad=12)

        eps = 1e-20
        if J_th is not None and J_btbt is not None and J_tat is not None:
            if gain is not None:
                ax.semilogy(Vbias, J_th * gain + eps, label="Thermal × M", lw=2)
                ax.semilogy(Vbias, J_btbt * gain + eps, label="BTBT × M", lw=2)
                ax.semilogy(Vbias, J_tat * gain + eps, label="TAT × M", lw=2)
                J_total_mult = (J_th + J_btbt + J_tat) * gain
                ax.semilogy(Vbias, J_total_mult + eps, "k--", label="Total × M", lw=1.5)
            else:
                ax.semilogy(Vbias, np.abs(J_th) + eps, label="Thermal", lw=2)
                ax.semilogy(Vbias, np.abs(J_btbt) + eps, label="BTBT", lw=2)
                ax.semilogy(Vbias, np.abs(J_tat) + eps, label="TAT", lw=2)
                J_total = np.abs(J_th + J_btbt + J_tat)
                ax.semilogy(Vbias, J_total + eps, "k--", label="Total", lw=1.5)
            ax.legend(fontsize=7)
        else:
            # Fallback: plot aggregate I_dark when components aren't available
            ax.semilogy(Vbias, np.abs(I_dark) + eps, "b-o", lw=2, ms=4,
                        label="Dark Current")
            ax.legend(fontsize=8)
        ax.set_xlabel("Excess Voltage (V)")
        ax.set_ylabel("Current Density (A/cm²)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save("dark_current_vs_bias.png", plt)


class DarkCurrentComponentsPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "dark_current_components"

    def plot(self, Vex: np.ndarray, I_srh: np.ndarray,
             I_btbt: np.ndarray, I_tat: np.ndarray,
             Vbr: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        eps = 1e-20
        ax.semilogy(Vex, np.abs(I_srh) + eps, label="SRH", lw=2)
        ax.semilogy(Vex, np.abs(I_btbt) + eps, label="BTBT", lw=2)
        ax.semilogy(Vex, np.abs(I_tat) + eps, label="TAT", lw=2)
        I_total = np.abs(I_srh + I_btbt + I_tat)
        ax.semilogy(Vex, I_total + eps, "k--", label="Total", lw=1.5)
        if Vbr is not None:
            ax.axvline(x=0, color="k", ls=":", alpha=0.5, label=f"Vbr = {Vbr:.1f} V")
        ax.set_xlabel("Excess Voltage (V)")
        ax.set_ylabel("Primary Dark Current (A)")
        ax.set_title("Primary Dark Generation Current vs Excess Voltage", fontsize=12, pad=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("dark_current_components.png", plt)


class DCRPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "dcr"

    def plot(self, Vbias: np.ndarray, DCR: np.ndarray,
             Vbr: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        eps = 1e-10
        ax.semilogy(Vbias, np.abs(DCR) + eps, "o-", lw=2)
        if Vbr is not None:
            ax.axvline(x=0, color="k", ls=":", alpha=0.5, label=f"Vbr = {Vbr:.1f} V")
            ax.legend(fontsize=8)
        ax.set_title("Dark Count Rate (DCR) vs Excess Voltage", fontsize=12, pad=12)
        ax.set_xlabel("Excess Voltage (V)")
        ax.set_ylabel("DCR (cps)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("dcr_vs_bias.png", plt)


class IVCharacteristicPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "iv_characteristic"

    def plot(self, Vbias: np.ndarray, I_dark: np.ndarray,
             I_light: np.ndarray | None = None,
             optical_power: float | None = None,
             Vbr: float | None = None) -> None:
        plt = self._import()
        fig, (ax_log, ax_lin) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        eps = 1e-20
        logI_dark = np.log10(np.abs(I_dark) + eps)
        ax_log.plot(Vbias, logI_dark, "b-", label="Dark", lw=2)
        if I_light is not None:
            label = "Illuminated"
            if optical_power is not None:
                label += f" ({optical_power*1e6:.0f} µW)"
            logI_light = np.log10(np.abs(I_light) + eps)
            ax_log.plot(Vbias, logI_light, color="tab:orange", ls="-", label=label, lw=2.5, alpha=0.9)
        if Vbr is not None:
            ax_log.axvline(x=Vbr, color="k", ls=":", alpha=0.5, label=f"Vbr = {Vbr:.1f} V")
        ax_log.set_ylabel("log₁₀ I (A)")
        ax_log.legend(fontsize=8)
        ax_log.grid(True, alpha=0.3)
        ax_log.set_title("I-V Characteristic — Log Scale", fontsize=11, pad=10)

        ax_lin.plot(Vbias, np.abs(I_dark) * 1e9, "b-", label="Dark", lw=2)
        if I_light is not None:
            I_light_lin = np.abs(I_light) * 1e9
            ax_lin.fill_between(Vbias, 0, I_light_lin,
                                color="tab:orange", alpha=0.15, label=None)
            ax_lin.plot(Vbias, I_light_lin, color="tab:orange", ls="-",
                        label=label if I_light is not None else "", lw=2.5, alpha=0.9)
        if Vbr is not None:
            ax_lin.axvline(x=Vbr, color="k", ls=":", alpha=0.5, label=f"Vbr = {Vbr:.1f} V")
        ax_lin.set_xlabel("Bias (V)")
        ax_lin.set_ylabel("I (nA)")
        ax_lin.legend(fontsize=8)
        ax_lin.grid(True, alpha=0.3)
        ax_lin.set_title("I-V Characteristic — Linear Scale", fontsize=11, pad=10)

        fig.suptitle("I-V Characteristic (passive quenching load assumed)", fontsize=13, y=1.01)
        plt.tight_layout()
        self._save("iv_characteristic.png", plt)


class ComprehensiveIVPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "comprehensive_iv"

    def plot(self, Vbias: np.ndarray, I_dark: np.ndarray,
             gain: np.ndarray | None = None,
             Vbr: float | None = None) -> None:
        plt = self._import()
        fig, (ax_log, ax_lin) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        eps = 1e-20
        ax_log.plot(Vbias, np.log10(np.abs(I_dark) + eps), "b-", label="Dark", lw=2)
        if Vbr is not None:
            ax_log.axvline(x=Vbr, color="k", ls=":", alpha=0.5, label=f"Vbr = {Vbr:.1f} V")
        ax_log.set_ylabel("log₁₀ I (A)")
        ax_log.legend(fontsize=8)
        ax_log.grid(True, alpha=0.3)
        ax_log.set_title("Comprehensive I-V — Log Scale", fontsize=11, pad=10)

        mask_lin = Vbias < Vbr if Vbr is not None else slice(None)
        ax_lin.plot(Vbias[mask_lin], np.abs(I_dark[mask_lin]) * 1e9, "b-", label="Dark", lw=2)
        if Vbr is not None:
            ax_lin.axvline(x=Vbr, color="k", ls=":", alpha=0.5, label=f"Vbr = {Vbr:.1f} V")
        ax_lin.set_xlabel("Bias (V)")
        ax_lin.set_ylabel("I (nA)")
        ax_lin.legend(fontsize=8)
        ax_lin.grid(True, alpha=0.3)
        ax_lin.set_title("Comprehensive I-V — Linear Scale (pre-breakdown)", fontsize=11, pad=10)

        fig.suptitle("Comprehensive I-V Characteristic", fontsize=13, y=1.01)
        plt.tight_layout()
        self._save("comprehensive_iv.png", plt)


class TrapDensityIVPlotter(BasePlotter):
    """2×2 subplot: each subplot shows 5 current components vs reverse
    voltage for a given trap density N_T."""

    @property
    def name(self) -> str:
        return "trap_density_iv"

    def plot(self, subplots_data: list[dict], Vbr: float | None = None) -> None:
        plt = self._import()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        eps = 1e-25

        colors = {
            "dark": "black",
            "optical": "red",
            "srh": "blue",
            "btbt": "green",
            "avalanche": "orange",
        }
        styles = {
            "dark": ("-", "o"),
            "optical": ("--", "s"),
            "srh": (":", "^"),
            "btbt": ("-.", "D"),
            "avalanche": ((0, (3, 1, 1, 1)), "v"),
        }

        for ax, sd in zip(axes.flat, subplots_data):
            lbl = sd.get("label", "")
            V = sd["V"]

            ax.set_title(lbl, fontsize=11, pad=8)
            for key in ("dark", "optical", "srh", "btbt", "avalanche"):
                if key in sd:
                    y = np.abs(sd[key]) + eps
                    ls, marker = styles[key]
                    ax.semilogy(V, y, color=colors[key], ls=ls,
                                marker=marker, ms=3, markevery=5,
                                lw=1.5, label=key.capitalize())

            if Vbr is not None:
                ax.axvline(x=Vbr, color="gray", ls=":", alpha=0.6, lw=1)

            ax.set_xlabel("Reverse Bias (V)", fontsize=10)
            ax.set_ylabel("Current (A)", fontsize=10)
            ax.set_xlim(0.0, 90.0)
            ax.set_ylim(1e-18, 1e2)
            ax.tick_params(labelbottom=True)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)

        fig.suptitle("Dark & Optical Current vs Trap Density",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        self._save("trap_density_iv.png", plt)
