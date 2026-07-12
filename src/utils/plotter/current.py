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
        ax.set_ylabel("Dark Current (A)")
        ax.set_title("Dark Current Components vs Excess Voltage", fontsize=12, pad=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("dark_current_components.png", plt)


class DCRPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "dcr"

    def plot(self, Vbias: np.ndarray, DCR: np.ndarray) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(Vbias, DCR + 1e-10)
        ax.set_title("Dark Count Rate (DCR) vs Bias Voltage", fontsize=12, pad=12)
        ax.set_xlabel("Bias (V)")
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
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("I-V Characteristic", fontsize=12, pad=12)
        ax.semilogy(Vbias, np.abs(I_dark) + 1e-20, "b-",
                    label="Dark", lw=2)
        if I_light is not None:
            label = "Illuminated"
            if optical_power is not None:
                label += f" ({optical_power*1e6:.0f} µW)"
            ax.semilogy(Vbias, np.abs(I_light) + 1e-20, "r--",
                        label=label, lw=2)
        if Vbr is not None:
            ax.axvline(x=Vbr, color="k", ls=":", alpha=0.5,
                        label=f"Vbr = {Vbr:.1f} V")
        # Transition from linear to exponential growth:
        # find the first point where d(log I)/dV exceeds the low-bias
        # median by a significant margin
        eps = 1e-20
        logI = np.log(np.abs(I_dark) + eps)
        dlogI_dV = np.gradient(logI, Vbias)
        baseline = np.median(dlogI_dV[:len(Vbias)//4])
        threshold = baseline + 2.0 * np.std(dlogI_dV[:len(Vbias)//4])
        above = np.where(dlogI_dV > threshold)[0]
        V_trans = float(Vbias[above[0]]) if len(above) > 0 else float(Vbias[-1])
        ax.axvline(x=V_trans, color="orange", ls="--", lw=2, alpha=0.9,
                    label=f"Transition ≈ {V_trans:.1f} V", zorder=5)
        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("log I (A)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
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
        fig, ax = plt.subplots(figsize=(8, 5))

        eps = 1e-20
        ax.semilogy(Vbias, np.abs(I_dark) + eps, "b-",
                     label="Dark", lw=2)
        # Transition from linear to exponential growth:
        # first point where d(log I)/dV exceeds low-bias baseline
        logI = np.log(np.abs(I_dark) + eps)
        dlogI_dV = np.gradient(logI, Vbias)
        baseline = np.median(dlogI_dV[:len(Vbias)//4])
        threshold = baseline + 2.0 * np.std(dlogI_dV[:len(Vbias)//4])
        above = np.where(dlogI_dV > threshold)[0]
        V_trans = float(Vbias[above[0]]) if len(above) > 0 else float(Vbias[-1])
        ax.axvline(x=V_trans, color="orange", ls="--", lw=2, alpha=0.9,
                    label=f"Transition ≈ {V_trans:.1f} V", zorder=5)
        if Vbr is not None:
            ax.axvline(x=Vbr, color="k", ls=":", alpha=0.5,
                        label=f"Vbr = {Vbr:.1f} V")

        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("log I (A)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Comprehensive I-V Characteristic", fontsize=12)
        plt.tight_layout()
        self._save("comprehensive_iv.png", plt)
