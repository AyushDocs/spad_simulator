"""Temperature-dependent plotters: DCR, PDE, dark current components, breakdown voltage."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class DCRvsTempPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "dcr_vs_temp"

    def plot(self, temperatures: np.ndarray, DCR: np.ndarray,
             Vex: float | None = None) -> None:
        plt = self._import()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: DCR vs T (log scale)
        eps = 1e-10
        ax1.semilogy(temperatures, np.abs(DCR) + eps, "o-", lw=2)
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("DCR (cps)")
        title = "DCR vs Temperature"
        if Vex is not None:
            title += f" (Vex = {Vex:.1f} V)"
        ax1.set_title(title, fontsize=12, pad=12)
        ax1.grid(True, alpha=0.3)

        # Right: Arrhenius plot — ln(DCR) vs 1000/T
        valid = np.abs(DCR) > eps
        if np.sum(valid) >= 2:
            T_valid = temperatures[valid]
            DCR_valid = np.abs(DCR[valid])
            inv_T = 1000.0 / T_valid  # 1000/T (K⁻¹)
            ln_DCR = np.log(DCR_valid)
            ax2.plot(inv_T, ln_DCR, "s-", lw=2, color="#d62728")
            # Linear fit to extract activation energy
            if len(inv_T) >= 2:
                coeffs = np.polyfit(inv_T, ln_DCR, 1)
                E_a = -coeffs[0] * 8.617e-5  # convert from K to eV (k_B = 8.617e-5 eV/K)
                fit_line = np.polyval(coeffs, inv_T)
                ax2.plot(inv_T, fit_line, "--", color="gray", alpha=0.6,
                         label=f"E_a = {E_a:.2f} eV")
                ax2.legend(fontsize=9)
        ax2.set_xlabel("1000 / T (K⁻¹)")
        ax2.set_ylabel("ln(DCR)")
        ax2.set_title("Arrhenius Plot", fontsize=12, pad=12)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Dark Count Rate vs Temperature" + (f"  (Vex = {Vex:.1f} V)" if Vex else ""),
                     fontsize=13, y=1.02)
        plt.tight_layout()
        self._save("dcr_vs_temperature.png", plt)


class PDEvsTempPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "pde_vs_temp"

    def plot(self, temperatures: np.ndarray, pde_dict: dict,
             wavelengths_nm: np.ndarray | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pde_dict)))
        for (lam, pde), c in zip(sorted(pde_dict.items()), colors):
            ax.plot(temperatures, pde * 100, "o-", lw=2, color=c,
                    label=f"{lam:.0f} nm")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("PDE (%)")
        ax.set_title("Photon Detection Efficiency (PDE) vs Temperature", fontsize=12, pad=12)
        ax.legend(fontsize=8, title="Wavelength")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        self._save("pde_vs_temperature.png", plt)


class DarkCurrentComponentsVsTempPlotter(BasePlotter):
    """Plot each dark current component (SRH, BTBT, TAT) vs temperature."""

    @property
    def name(self) -> str:
        return "dark_current_vs_temp_components"

    def plot(self, temperatures: np.ndarray, J_srh: np.ndarray,
             J_btbt: np.ndarray, J_tat: np.ndarray,
             J_total: np.ndarray | None = None,
             Vex: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        eps = 1e-25
        ax.semilogy(temperatures, np.abs(J_srh) + eps, "o-", lw=2,
                    label="SRH (thermal)", color="#1f77b4")
        ax.semilogy(temperatures, np.abs(J_btbt) + eps, "s-", lw=2,
                    label="BTBT", color="#d62728")
        ax.semilogy(temperatures, np.abs(J_tat) + eps, "^-", lw=2,
                    label="TAT", color="#2ca02c")
        if J_total is not None:
            ax.semilogy(temperatures, np.abs(J_total) + eps, "k--",
                        lw=1.5, label="Total")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Current Density (A/cm²)")
        title = "Dark Current Components vs Temperature"
        if Vex is not None:
            title += f"  (Vex = {Vex:.1f} V)"
        ax.set_title(title, fontsize=12, pad=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("dark_current_components_vs_temp.png", plt)


class BreakdownVoltageVsTempPlotter(BasePlotter):
    """Plot breakdown voltage vs temperature."""

    @property
    def name(self) -> str:
        return "breakdown_voltage_vs_temp"

    def plot(self, temperatures: np.ndarray, Vbr_arr: np.ndarray) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(temperatures, Vbr_arr, "o-", lw=2, color="#1f77b4")
        if len(temperatures) > 1:
            dvdt = (Vbr_arr[-1] - Vbr_arr[0]) / (temperatures[-1] - temperatures[0])
            ax.annotate(f"dVbr/dT = {dvdt*1e3:.1f} mV/K",
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Breakdown Voltage Vbr (V)")
        ax.set_title("Breakdown Voltage vs Temperature", fontsize=12, pad=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("breakdown_voltage_vs_temp.png", plt)
