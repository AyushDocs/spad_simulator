"""Avalanche-related plotters: trigger, afterpulsing, noise, ionization, multiplication, maps."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class TriggerProbabilityPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "trigger_probability"

    def plot(self, x: np.ndarray, Pe: np.ndarray,
             Ph: np.ndarray | None = None,
             V_list: list[float] | None = None,
             Vbr: float | None = None,
             doping: np.ndarray | None = None,
             E_field: np.ndarray | None = None,
             filename: str = "trigger_probability.png") -> None:
        plt = self._import()
        x_um = x * 1e4
        n_plots = Pe.shape[0] if Pe.ndim > 1 else 1

        # Arrange subplots: use 2 rows when > 3 panels
        if n_plots <= 3:
            nrows, ncols = 1, n_plots
        else:
            ncols = min(3, n_plots)
            nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4.5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for i in range(n_plots):
            ax = axes[i]
            pe_i = Pe[i] if Pe.ndim > 1 else Pe
            ph_i = Ph[i] if (Ph is not None and Ph.ndim > 1) else Ph
            E_i = E_field[i] if (E_field is not None and E_field.ndim > 1) else E_field
            Vex = (V_list[i] - Vbr) if (Vbr is not None and V_list) else (V_list[i] if V_list else i)

            # Plot trigger probabilities
            ax.plot(x_um, pe_i, "r-", lw=2, label="Pe (electron)")
            if ph_i is not None:
                ax.plot(x_um, ph_i, "b--", lw=2, label="Ph (hole)")
                # Ptr = Pe + Ph - Pe * Ph (pair triggering probability)
                ptr_i = pe_i + ph_i - pe_i * ph_i
                ax.plot(x_um, ptr_i, "g-.", lw=2, label="Pp (pair)")

            # Auto-zoom to the multiplication transition region
            changing = (pe_i > 0.001) & (pe_i < 0.999 * np.max(pe_i))
            if np.any(changing):
                x_changing = x_um[changing]
                pad_left = 0.5  # um
                pad_right = 1.0  # um (slightly wider right side to show grading/absorber junction entry)
                ax.set_xlim(max(0, x_changing[0] - pad_left), min(x_um[-1], x_changing[-1] + pad_right))
            else:
                # Fallback range covering the active multiplication region for SAGCM SPADs
                ax.set_xlim(2.0, 5.0)

            ax.set_xlabel("Depth (µm)", fontsize=10)
            ax.set_ylabel("Probabilities", fontsize=10)
            # Distinguish sub-breakdown vs Geiger titles
            title_color = "#cc4400" if Vex < 0 else "#006600"
            ax.set_title(f"Vex = {Vex:.1f} V", fontsize=11, color=title_color)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.02, 1.02)

            # Overlay electric field on a twin y-axis (right side)
            if E_i is not None:
                ax2 = ax.twinx()
                ax2.plot(x_um, np.abs(E_i) / 1e5, color="purple", ls="-", lw=1.2, alpha=0.7, label="|E| field")
                ax2.set_ylabel("|E| (×10⁵ V/cm)", color="purple", fontsize=9)
                ax2.tick_params(axis="y", labelcolor="purple", labelsize=8)
                e_max = float(np.max(np.abs(E_i))) / 1e5
                ax2.set_ylim(0, e_max * 1.15)
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="lower left")
            elif doping is not None:
                # Fallback: plot doping on twin y-axis when no E-field
                ax2 = ax.twinx()
                ax2.semilogy(x_um, np.abs(doping) + 1e10, color="green", ls=":", lw=1.5, label="Doping")
                ax2.set_ylabel("Doping ($cm^{-3}$)", color="green", fontsize=9)
                ax2.tick_params(axis="y", labelcolor="green", labelsize=8)
                ax2.set_ylim(1e14, 1e19)

                # Combine legends from both axes
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="lower left")
            else:
                ax.legend(fontsize=7, loc="lower left")

        # Hide unused axes
        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Trigger Probability Profiles & Doping Profile vs Depth", fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save(filename, plt)


class TriggerVsVexPlotter(BasePlotter):
    """Plot max Pe, max Ph, and avalanche trigger probability Ptr as a function of excess voltage Vex."""

    @property
    def name(self) -> str:
        return "trigger_vs_vex"

    def plot(self, Vex_arr: np.ndarray, Pe_max: np.ndarray,
             Ph_max: np.ndarray, Vbr: float | None = None,
             Ptr_max: np.ndarray | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))

        valid_e = np.isfinite(Pe_max)
        valid_h = np.isfinite(Ph_max)

        ax.plot(Vex_arr[valid_e], Pe_max[valid_e],
                "b-o", lw=2, ms=4, label="⟨Pe⟩ electron (mult. layer)")
        ax.plot(Vex_arr[valid_h], Ph_max[valid_h],
                "r--s", lw=2, ms=4, label="⟨Ph⟩ hole (mult. layer)")

        if Ptr_max is None:
            Ptr_max = Pe_max + Ph_max - Pe_max * Ph_max
        valid_tr = np.isfinite(Ptr_max)
        ax.plot(Vex_arr[valid_tr], Ptr_max[valid_tr],
                "g-.d", lw=2, ms=4, label="⟨Ptr⟩ avalanche = Pe + Ph - Pe·Ph")

        # Shade area between Pe and Ph
        if np.any(valid_e) and np.any(valid_h):
            ax.fill_between(Vex_arr[valid_e & valid_h],
                            Pe_max[valid_e & valid_h],
                            Ph_max[valid_e & valid_h],
                            alpha=0.10, color="purple",
                            label="⟨Pe⟩ − ⟨Ph⟩ spread")

        ax.axhline(y=1.0, color="gray", ls=":", lw=1, alpha=0.6)
        ax.axvline(x=0.0, color="k", ls="--", lw=1.5, alpha=0.6,
                   label=f"Vbr = {Vbr:.1f} V" if Vbr else "Vex = 0")

        # Shade Geiger region (Vex > 0)
        ax.axvspan(0, Vex_arr[-1] + 1, alpha=0.04, color="green",
                   label="Geiger mode (Vex > 0)")

        ax.set_xlabel("Excess Voltage Vex (V)")
        ax.set_ylabel("Mean Trigger Probability (mult. layer)")
        title = "Trigger Probability vs Excess Voltage"
        if Vbr is not None:
            title += f"  (Vbr = {Vbr:.1f} V)"
        ax.set_title(title, fontsize=12, pad=12)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlim(Vex_arr[0], Vex_arr[-1])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("trigger_vs_vex.png", plt)


class AfterpulsingPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "afterpulsing"

    def plot(self, holdoff: np.ndarray, P_ap: np.ndarray,
             N_T: float | None = None, tau_c: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(holdoff * 1e6, P_ap * 100, lw=2)
        ax.set_xlabel("Holdoff Time (µs)")
        ax.set_ylabel("Afterpulsing Probability (%)")
        title = "Afterpulsing Probability vs Holdoff Time"
        if N_T is not None and tau_c is not None:
            title += f"\nN_T={N_T:.1e} cm⁻³, τ_c={tau_c*1e6:.1f} µs"
        ax.set_title(title, fontsize=12, pad=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("afterpulsing_vs_holdoff.png", plt)


class ExcessNoisePlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "excess_noise"

    def plot(self, M: np.ndarray, F: np.ndarray,
             k_eff: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(M, F, lw=2)
        ax.axhline(y=1.0, color="k", ls="--", alpha=0.3, label="F=1 (shot noise)")
        if k_eff is not None:
            ax.plot(M, k_eff * M + (1.0 - k_eff) * (2.0 - 1.0 / np.clip(M, 1, None)),
                    ":", color="gray", alpha=0.6,
                    label=f"k_eff={k_eff:.2f}")
        ax.set_xlabel("Multiplication M")
        ax.set_ylabel("Excess Noise Factor F(M)")
        ax.set_title("McIntyre Excess Noise Factor vs Multiplication", fontsize=12, pad=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("excess_noise.png", plt)


class IonizationCoefficientsVsFieldPlotter(BasePlotter):
    """Plot ionization coefficients α(E) and β(E) for electrons and holes."""

    @property
    def name(self) -> str:
        return "ionization_vs_field"

    def plot(self, E_arr: np.ndarray, alpha_e: dict[str, np.ndarray],
             beta_h: dict[str, np.ndarray],
             material_name: str = "InP") -> None:
        plt = self._import()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        linestyles = ["-", "--", ":"]
        colors = ["#1f77b4", "#d62728", "#2ca02c"]

        for i, (model_name, alpha_vals) in enumerate(alpha_e.items()):
            mask = alpha_vals > 0
            ax1.loglog(E_arr[mask] / 1e6, alpha_vals[mask],
                       ls=linestyles[i % len(linestyles)],
                       color=colors[i % len(colors)], lw=2, label=model_name)
        ax1.set_xlabel("Electric Field (×10⁶ V/m)")
        ax1.set_ylabel("α_e (cm⁻¹)")
        ax1.set_title(f"Electron Ionization — {material_name}", fontsize=12, pad=12)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, which="both")

        for i, (model_name, beta_vals) in enumerate(beta_h.items()):
            mask = beta_vals > 0
            ax2.loglog(E_arr[mask] / 1e6, beta_vals[mask],
                       ls=linestyles[i % len(linestyles)],
                       color=colors[i % len(colors)], lw=2, label=model_name)
        ax2.set_xlabel("Electric Field (×10⁶ V/m)")
        ax2.set_ylabel("β_h (cm⁻¹)")
        ax2.set_title(f"Hole Ionization — {material_name}", fontsize=12, pad=12)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, which="both")

        fig.suptitle("Impact Ionization Coefficients vs Electric Field", fontsize=14, y=0.98)
        plt.tight_layout()
        self._save("ionization_vs_field.png", plt)


class IonizationRatioVsFieldPlotter(BasePlotter):
    """Plot ionization ratio k = β/α vs electric field."""

    @property
    def name(self) -> str:
        return "ionization_ratio"

    def plot(self, E_arr: np.ndarray, k_ratio: dict[str, np.ndarray],
             material_name: str = "InP",
             peak_field: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        linestyles = ["-", "--", ":"]
        colors = ["#1f77b4", "#d62728", "#2ca02c"]

        for i, (model_name, k_vals) in enumerate(k_ratio.items()):
            mask = (k_vals > 0) & np.isfinite(k_vals)
            ax.loglog(E_arr[mask] / 1e6, k_vals[mask],
                      ls=linestyles[i % len(linestyles)],
                      color=colors[i % len(colors)], lw=2, label=model_name)
        ax.axhline(y=1.0, color="gray", ls="--", alpha=0.5, label="k = 1")

        if peak_field is not None:
            ax.axvline(x=peak_field / 1e6, color="purple", ls="--", lw=1.5,
                       alpha=0.7, label=f"Peak field = {peak_field:.2e} V/cm")

        ax.set_xlabel("Electric Field (×10⁶ V/m)")
        ax.set_ylabel("Ionization Ratio k = β / α")
        ax.set_title(f"Ionization Ratio vs Field — {material_name}", fontsize=12, pad=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        self._save("ionization_ratio_vs_field.png", plt)


class MultiplicationVsExcessBiasPlotter(BasePlotter):
    """Plot multiplication factor M vs excess bias voltage."""

    @property
    def name(self) -> str:
        return "multiplication_vs_vex"

    def plot(self, Vex_arr: np.ndarray, M_arr: np.ndarray,
             Vbr: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        mask = np.isfinite(M_arr) & (M_arr > 0)
        ax.semilogy(Vex_arr[mask], M_arr[mask], "o-", lw=2, color="#1f77b4")
        if Vbr is not None:
            ax.axvline(x=0, color="k", ls=":", alpha=0.5, label=f"Vbr = {Vbr:.1f} V")
        ax.set_xlabel("V - Vbr (V)")
        ax.set_ylabel("Multiplication Factor M")
        ax.set_title("Avalanche Multiplication vs Bias (below breakdown)", fontsize=12, pad=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("multiplication_vs_vex.png", plt)


class AvalancheProbabilityMapPlotter(BasePlotter):
    """2D heatmap: trigger probability vs depth and excess voltage."""

    @property
    def name(self) -> str:
        return "avalanche_map"

    def plot(self, x_um: np.ndarray, Vex_arr: np.ndarray,
             Pe_2d: np.ndarray, Vbr: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 6))
        extent = [x_um[0], x_um[-1], Vex_arr[0], Vex_arr[-1]]
        im = ax.imshow(Pe_2d, aspect="auto", origin="lower",
                       extent=extent, cmap="hot", vmin=0, vmax=1,
                       interpolation="bilinear")
        fig.colorbar(im, ax=ax, label="Trigger Probability P_e")
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Excess Voltage (V)")
        title = "Avalanche Probability Map"
        if Vbr is not None:
            title += f"  (Vbr = {Vbr:.1f} V)"
        ax.set_title(title, fontsize=12, pad=12)
        plt.tight_layout()
        self._save("avalanche_probability_map.png", plt)


class BreakdownProbVsExcessBiasPlotter(BasePlotter):
    """Plot breakdown probability vs excess bias from Monte Carlo."""

    @property
    def name(self) -> str:
        return "breakdown_prob_vs_vex"

    def plot(self, Vex_arr: np.ndarray, BrP_arr: np.ndarray,
             N_sim: int | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(Vex_arr, BrP_arr, where="mid", lw=2, color="#1f77b4")
        ax.plot(Vex_arr, BrP_arr, "o", color="#1f77b4", markersize=6)
        ax.axhline(y=0.5, color="r", ls="--", alpha=0.5, label="P = 0.5")
        ax.set_xlabel("Excess Voltage (V)")
        ax.set_ylabel("Breakdown Probability")
        title = "Breakdown Probability vs Excess Bias"
        if N_sim is not None:
            title += f"  (N = {N_sim} per point)"
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("breakdown_prob_vs_vex.png", plt)


class ATPPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "atp"

    def plot(self, x_um: np.ndarray, Vex_list: list[float],
             ATP_list: list[np.ndarray], Vbr: float) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 6))
        for Vex, ATP in zip(Vex_list, ATP_list):
            ax.plot(x_um, ATP, label=f'Vex={Vex}V')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('ATP = Pe + Ph - Pe\u00b7Ph')
        ax.set_title(f'Avalanche Triggering Probability vs position (Vbr={Vbr}V)', fontsize=12, pad=12)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        self._save("atp_vs_position.png", plt)


class TriggerBackCalculatePlotter(BasePlotter):
    """Validate trigger probability: spatial profiles + absorption-weighted Ptr(Vex)."""

    @property
    def name(self) -> str:
        return "trigger_back_calculate"

    def plot(self, x_um: np.ndarray,
             Pe_spatial: list[np.ndarray], Ph_spatial: list[np.ndarray],
             Ptr_spatial: list[np.ndarray], E_spatial: list[np.ndarray],
             labels: list[str],
             Vex_arr: np.ndarray, Pe_mean: np.ndarray,
             Ph_mean: np.ndarray, Ptr_mean: np.ndarray,
             Vbr: float = 0.0) -> None:
        plt = self._import()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: spatial profiles
        ax1 = axes[0]
        cmap = plt.cm.tab10
        for i, (pe, ph, ptr, lbl) in enumerate(zip(Pe_spatial, Ph_spatial, Ptr_spatial, labels)):
            c = cmap(i)
            ax1.plot(x_um, pe, "-", color=c, lw=1.5, label=f"Pe {lbl}")
            ax1.plot(x_um, ph, "--", color=c, lw=1.5, label=f"Ph {lbl}")
            ax1.plot(x_um, ptr, ":", color=c, lw=2.0, label=f"Ptr {lbl}")
        ax1.set_xlabel("Depth (µm)", fontsize=11)
        ax1.set_ylabel("Probability", fontsize=11)
        ax1.set_title("Spatial Trigger Profiles", fontsize=12, pad=10)
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        # Right: absorption-weighted Ptr vs Vex
        ax2 = axes[1]
        mask_pe = np.isfinite(Pe_mean)
        mask_ph = np.isfinite(Ph_mean)
        mask_ptr = np.isfinite(Ptr_mean)
        ax2.plot(Vex_arr[mask_pe], Pe_mean[mask_pe], "r-", lw=2, label="Pe (abs-weighted)")
        ax2.plot(Vex_arr[mask_ph], Ph_mean[mask_ph], "b--", lw=2, label="Ph (abs-weighted)")
        ax2.plot(Vex_arr[mask_ptr], Ptr_mean[mask_ptr], "k-", lw=2.5, label="Ptr (abs-weighted)")
        ax2.axvline(x=0, color="gray", ls=":", alpha=0.5)
        ax2.set_xlabel("Excess Voltage Vex (V)", fontsize=11)
        ax2.set_ylabel("Trigger Probability", fontsize=11)
        ax2.set_title("Absorption-Weighted Ptr vs Vex", fontsize=12, pad=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        self._save("trigger_back_calculate.png", plt)
