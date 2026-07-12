"""PDP spectrum, PDP vs excess voltage, and PDP 3D plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class PDPPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "pdp"

    def plot(self, wavelengths: np.ndarray, pdp: np.ndarray,
             Vex_list: list[float] | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Photon Detection Probability (PDP) Spectrum", fontsize=12, pad=12)
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
        self._save("pdp_spectrum.png", plt)


class PDPVsExcessVoltagePlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "pdp_vs_vex"

    def plot(self, Vex: np.ndarray, pdp_dict: dict,
             wavelengths_nm: np.ndarray | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("PDP vs Excess Voltage", fontsize=12, pad=12)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pdp_dict)))
        for (lam, pdp), c in zip(sorted(pdp_dict.items()), colors):
            ax.plot(Vex, pdp * 100, "o-", lw=2, color=c,
                    label=f"{lam:.0f} nm")

        # Add vertical line at breakdown Vex = 0
        ax.axvline(x=0.0, color="k", ls="--", lw=1.5, alpha=0.6, label="Breakdown (Vbr)")

        # Shade Geiger region (Vex > 0)
        ax.axvspan(0, max(Vex) + 1, alpha=0.04, color="green", label="Geiger mode")

        ax.set_xlabel("Excess Voltage (V)")
        ax.set_ylabel("PDP (%)")
        ax.legend(fontsize=8, title="Wavelength")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        ax.set_xlim(min(Vex), max(Vex))
        plt.tight_layout()
        self._save("pdp_vs_excess_voltage.png", plt)


class AbsorptionProfilePlotter(BasePlotter):
    """Beer-Lambert absorption profile G(x) = α·exp(-α·x) in absorber."""

    @property
    def name(self) -> str:
        return "absorption_profile"

    def plot(self, x_um: np.ndarray, G_dict: dict[str, np.ndarray],
             material_name: str = "InGaAs") -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(G_dict)))
        for (lam_nm, G), c in zip(sorted(G_dict.items(), key=lambda kv: float(kv[0])), colors):
            ax.plot(x_um, G, lw=2, color=c, label=f"{lam_nm} nm")
            ax.fill_between(x_um, G, alpha=0.1, color=c)
        ax.set_xlabel("Depth into Absorber (µm)")
        ax.set_ylabel("G(x) = α·exp(−αx)  (cm⁻¹)")
        ax.set_title(f"Beer-Lambert Absorption Profile — {material_name}\n"
                     r"$G(x) = \alpha \, e^{-\alpha x}$", fontsize=12, pad=12)
        ax.legend(fontsize=8, title="Wavelength")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("absorption_profile.png", plt)


class PDP3DPlotter(BasePlotter):
    """3D surface plot: wavelength × excess voltage × PDP."""

    @property
    def name(self) -> str:
        return "pdp_3d"

    def plot(self, wavelengths_nm: np.ndarray, Vex_arr: np.ndarray,
             pdp_2d: np.ndarray) -> None:
        plt = self._import()
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")
        WL, VEX = np.meshgrid(wavelengths_nm, Vex_arr)
        surf = ax.plot_surface(WL, VEX, pdp_2d * 100, cmap="viridis",
                               edgecolor="none", alpha=0.85)
        fig.colorbar(surf, ax=ax, shrink=0.5, label="PDP (%)")
        ax.set_xlabel("Wavelength (nm)", labelpad=10)
        ax.set_ylabel("Excess Voltage (V)", labelpad=10)
        ax.set_zlabel("PDP (%)", labelpad=10)
        ax.set_title("Photon Detection Probability", fontsize=12, pad=15)
        ax.view_init(elev=25, azim=135)
        plt.tight_layout()
        self._save("pdp_3d_surface.png", plt)
