from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------
# Plotter interface
# ----------------------------------------------------------------

class Plotter(ABC):
    """Interface for all plotters. Each subclass implements one plot type."""

    @abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ----------------------------------------------------------------
# concrete plotters
# ----------------------------------------------------------------

class _BasePlotter(Plotter):
    """Shared helpers for matplotlib-based plotters."""

    def __init__(self, plot_dir: str = "plots") -> None:
        self.plot_dir = plot_dir
        self._imported = False

    def _import(self) -> None:
        if not self._imported:
            global plt
            import matplotlib.pyplot as plt  # type: ignore[import-untyped]
            self._imported = True

    def _save(self, fname: str) -> None:
        import os
        path = os.path.join(self.plot_dir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        import logging
        logging.getLogger("spad.plots").info("saved  %s", path)


class DeviceStructurePlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "device_structure"

    def plot(self, x: np.ndarray, mat_names: np.ndarray,
             doping: np.ndarray, net_doping: np.ndarray) -> None:
        self._import()
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

        fig.suptitle("Device Structure", fontsize=12)
        plt.tight_layout()
        self._save("device_structure.png")


class PotentialProfilePlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "potential_profile"

    def plot(self, x: np.ndarray, phi: np.ndarray,
             V_list: List[float] | None = None) -> None:
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


class ElectricFieldPlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "electric_field"

    def plot(self, x: np.ndarray, E: np.ndarray,
             V_list: List[float] | None = None,
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


class DarkCurrentPlotter(_BasePlotter):
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


class DCRPlotter(_BasePlotter):
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


class PDPPlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "pdp"

    def plot(self, wavelengths: np.ndarray, pdp: np.ndarray,
             Vex_list: List[float] | None = None) -> None:
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


class TriggerProbabilityPlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "trigger_probability"

    def plot(self, x: np.ndarray, Pe: np.ndarray,
             Ph: np.ndarray | None = None,
             V_list: List[float] | None = None) -> None:
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


class IVCharacteristicPlotter(_BasePlotter):
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


class BreakdownSweepPlotter(_BasePlotter):
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


class PDPVsExcessVoltagePlotter(_BasePlotter):
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


class ComprehensiveIVPlotter(_BasePlotter):
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


class TrajectoryPlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "trajectory"

    def plot(self, trajectories: List[np.ndarray],
             times: List[np.ndarray] | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, traj in enumerate(trajectories):
            t = times[i] if times is not None else np.arange(len(traj))
            ax.plot(t, traj * 1e4, lw=0.5, alpha=0.7)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Position (µm)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("carrier_trajectories.png")


class JitterPlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "jitter"

    def plot(self, detection_times: np.ndarray,
             bins: int = 50) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(detection_times * 1e12, bins=bins, alpha=0.7, edgecolor="k")
        ax.set_xlabel("Detection Time (ps)")
        ax.set_ylabel("Counts")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("timing_jitter.png")


class PopulationPlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "population"

    def plot(self, time: np.ndarray, n_electrons: np.ndarray,
             n_holes: np.ndarray) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(time, n_electrons, label="Electrons", lw=2)
        ax.semilogy(time, n_holes, label="Holes", lw=2)
        ax.semilogy(time, n_electrons + n_holes, "k--",
                    label="Total", lw=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Carrier Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("carrier_population.png")


class DopingPlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "doping"

    def plot(self, x: np.ndarray, nd: np.ndarray, na: np.ndarray) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
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
        self._save("doping_profile.png")


class PDEPlotter(_BasePlotter):
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


class AfterpulsingPlotter(_BasePlotter):
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


class ExcessNoisePlotter(_BasePlotter):
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


class DCRvsTempPlotter(_BasePlotter):
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


class PDPvsTempPlotter(_BasePlotter):
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


class JitterHistogramPlotter(_BasePlotter):
    @property
    def name(self) -> str:
        return "jitter_histogram"

    def plot(self, detection_times: np.ndarray,
             bins: int = 50,
             fwhm: float | None = None,
             sigma: float | None = None) -> None:
        self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        t_ps = detection_times * 1e12
        ax.hist(t_ps, bins=bins, alpha=0.7, edgecolor="k", density=True)
        label_parts = []
        if fwhm is not None and np.isfinite(fwhm):
            label_parts.append(f"FWHM = {fwhm*1e12:.1f} ps")
        if sigma is not None and np.isfinite(sigma):
            label_parts.append(f"σ = {sigma*1e12:.1f} ps")
        if label_parts:
            ax.set_title("Timing Jitter (SPTR)\n" + "  ".join(label_parts), fontsize=10)
        else:
            ax.set_title("Timing Jitter (SPTR)", fontsize=10)
        ax.set_xlabel("Detection Time (ps)")
        ax.set_ylabel("Probability Density")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("timing_jitter_histogram.png")


# ----------------------------------------------------------------
# registry
# ----------------------------------------------------------------

_BUILTIN_PLOTTERS: Dict[str, type[Plotter]] = {
    "device_structure": DeviceStructurePlotter,
    "potential_profile": PotentialProfilePlotter,
    "electric_field": ElectricFieldPlotter,
    "dark_current": DarkCurrentPlotter,
    "dcr": DCRPlotter,
    "pdp": PDPPlotter,
    "pdp_vs_vex": PDPVsExcessVoltagePlotter,
    "trigger_probability": TriggerProbabilityPlotter,
    "iv_characteristic": IVCharacteristicPlotter,
    "comprehensive_iv": ComprehensiveIVPlotter,
    "breakdown_sweep": BreakdownSweepPlotter,
    "trajectory": TrajectoryPlotter,
    "jitter": JitterPlotter,
    "population": PopulationPlotter,
    "doping": DopingPlotter,
    "pde": PDEPlotter,
    "afterpulsing": AfterpulsingPlotter,
    "excess_noise": ExcessNoisePlotter,
    "dcr_vs_temp": DCRvsTempPlotter,
    "pdp_vs_temp": PDPvsTempPlotter,
    "jitter_histogram": JitterHistogramPlotter,
}


def get_plotter(name: str, plot_dir: str = "plots",
                **kwargs: Any) -> Plotter:
    cls = _BUILTIN_PLOTTERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown plotter '{name}'. "
                         f"Available: {list(_BUILTIN_PLOTTERS)}")
    return cls(plot_dir=plot_dir, **kwargs)


def register_plotter(name: str, plotter_cls: type[Plotter]) -> None:
    _BUILTIN_PLOTTERS[name] = plotter_cls
