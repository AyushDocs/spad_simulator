"""Plotters for parameter-sweep and I-V decomposition studies."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class IVSweepPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "iv_sweep"

    def plot(self, V: np.ndarray, I_dark: np.ndarray,
             I_srh: np.ndarray | None = None,
             I_btbt: np.ndarray | None = None,
             I_tat: np.ndarray | None = None,
             Vbr: float | None = None) -> None:
        plt = self._import()
        fig, (ax_log, ax_comp) = plt.subplots(1, 2, figsize=(14, 5))

        eps = 1e-20
        ax_log.semilogy(V, np.abs(I_dark) + eps, "k-", lw=2, label="Total")
        if Vbr is not None:
            ax_log.axvline(x=Vbr, color="k", ls=":", alpha=0.5, label=f"Vbr={Vbr:.1f}V")
        ax_log.set_xlabel("Bias (V)")
        ax_log.set_ylabel("Dark Current (A)")
        ax_log.set_title("I-V Characteristic (log)")
        ax_log.legend(fontsize=8)
        ax_log.grid(True, alpha=0.3)

        if I_srh is not None and I_btbt is not None and I_tat is not None:
            ax_comp.semilogy(V, np.abs(I_srh) + eps, label="SRH", lw=2)
            ax_comp.semilogy(V, np.abs(I_btbt) + eps, label="BTBT", lw=2)
            ax_comp.semilogy(V, np.abs(I_tat) + eps, label="TAT", lw=2)
            ax_comp.semilogy(V, np.abs(I_dark) + eps, "k--", label="Total", lw=1.5)
            ax_comp.set_title("Decomposed Dark Current")
        else:
            ax_comp.semilogy(V, np.abs(I_dark) + eps, "b-o", ms=3, lw=1.5)
            ax_comp.set_title("Dark Current")
        if Vbr is not None:
            ax_comp.axvline(x=Vbr, color="k", ls=":", alpha=0.5)
        ax_comp.set_xlabel("Bias (V)")
        ax_comp.set_ylabel("Current (A)")
        ax_comp.legend(fontsize=8)
        ax_comp.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save("iv_sweep.png", plt)


class ParamSweepPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "param_sweep"

    def plot(self, values: np.ndarray, y: np.ndarray,
             xlabel: str = "", ylabel: str = "",
             title: str = "", fname: str = "param_sweep.png",
             **kwargs) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(values, y, "b-o", ms=5, lw=2)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fname, plt)


class ParamSweepIVPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "param_sweep_iv"

    def plot(self, param_values: np.ndarray, V_range: np.ndarray,
             I_dark_2d: np.ndarray,
             param_label: str = "parameter") -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(9, 6))
        eps = 1e-20
        for i, pv in enumerate(param_values):
            label = f"{param_label}={pv:.2e}"
            ax.semilogy(V_range, np.abs(I_dark_2d[i]) + eps, lw=1.5, label=label)
        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("Dark Current (A)")
        ax.set_title(f"I-V Curves for Different {param_label}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("param_sweep_iv.png", plt)


class PunchBreakdownSweepPlotter(BasePlotter):
    """2×3 subplot: each panel shows V_pt and V_br vs one parameter."""

    @property
    def name(self) -> str:
        return "punch_breakdown_sweep"

    def plot(self, panels: list[dict]) -> None:
        plt = self._import()
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        for ax, p in zip(axes.flat, panels):
            x = p["values"]
            ax.plot(x, p["V_pt"], "b-o", ms=5, lw=2, label="V$_{\\mathrm{pt}}$")
            ax.plot(x, p["V_br"], "r-s", ms=5, lw=2, label="V$_{\\mathrm{br}}$")
            ax.set_xlabel(p.get("xlabel", ""), fontsize=10)
            ax.set_ylabel("Voltage (V)", fontsize=10)
            ax.set_title(p.get("label", ""), fontsize=11, pad=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save("punch_breakdown_sweep.png", plt)


class DCRvsPDEPlotter(BasePlotter):
    """DCR (y, log, Hz/cm²) vs PDE (x, fraction 0–1) for multiple absorption widths."""

    @property
    def name(self) -> str:
        return "dcr_vs_pde"

    def plot(self, curves: dict[str, dict], Vex: float = 3.0,
             detector_area_cm2: float = 4.91e-6) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(9, 6))

        cmap = plt.cm.viridis
        n = max(len(curves), 1)
        for i, (label, data) in enumerate(curves.items()):
            pde = np.asarray(data["PDE"])
            dcr = np.asarray(data["DCR"])  # keep in cps
            vex = np.asarray(data.get("Vex", np.linspace(0.5, 10, len(pde))))
            mask = (pde > 0) & (dcr > 0) & np.isfinite(pde) & np.isfinite(dcr)
            if not np.any(mask):
                continue
            color = cmap(i / max(n - 1, 1))
            ax.semilogy(pde[mask], dcr[mask], "-", lw=1.8, color=color, label=label)
            # Mark Vex = 3 V with a star
            idx3 = np.argmin(np.abs(vex[mask] - 3.0))
            pde_masked = pde[mask]
            dcr_masked = dcr[mask]
            ax.plot(pde_masked[idx3], dcr_masked[idx3], "*", ms=12, color=color,
                    markeredgecolor="k", markeredgewidth=0.5)
            # Mark Vex = 1, 5, 8 with small dots
            for v_mark in [1.0, 5.0, 8.0]:
                idx_m = np.argmin(np.abs(vex[mask] - v_mark))
                ax.plot(pde_masked[idx_m], dcr_masked[idx_m], "o", ms=4, color=color,
                        markeredgecolor="k", markeredgewidth=0.3)

        ax.set_xlabel("PDE", fontsize=12)
        ax.set_ylabel("DCR (cps)", fontsize=12)
        ax.set_title("DCR vs PDE — Absorption Layer Width Sweep (★ = Vex 3 V)", fontsize=12, pad=10)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        self._save("dcr_vs_pde.png", plt)


class DCRPDEVsVexPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "dcr_pde_vs_vex"

    def plot(self, Vex: np.ndarray, DCR: np.ndarray, PDE: np.ndarray,
             wavelength_nm: int = 1550) -> None:
        plt = self._import()
        fig, ax_dcr = plt.subplots(figsize=(8, 5))
        ax_pde = ax_dcr.twinx()

        eps = 1e-10
        ax_dcr.semilogy(Vex, np.abs(DCR) + eps, "b-o", ms=4, lw=2, label="DCR")
        ax_pde.plot(Vex, PDE * 100, "r-s", ms=4, lw=2, label=f"PDE ({wavelength_nm} nm)")

        ax_dcr.set_xlabel("Excess Voltage (V)")
        ax_dcr.set_ylabel("DCR (cps)", color="b")
        ax_pde.set_ylabel("PDE (%)", color="r")
        ax_dcr.tick_params(axis="y", labelcolor="b")
        ax_pde.tick_params(axis="y", labelcolor="r")

        lines_dcr, labels_dcr = ax_dcr.get_legend_handles_labels()
        lines_pde, labels_pde = ax_pde.get_legend_handles_labels()
        ax_dcr.legend(lines_dcr + lines_pde, labels_dcr + labels_pde, fontsize=8, loc="upper left")

        ax_dcr.set_title("DCR and PDE vs Excess Voltage", fontsize=12, pad=12)
        ax_dcr.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("dcr_pde_vs_vex.png", plt)
