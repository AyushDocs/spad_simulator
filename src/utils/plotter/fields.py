"""Potential, electric field, and breakdown sweep plotters."""
from __future__ import annotations

import numpy as np

from ._base import BasePlotter


class PotentialProfilePlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "potential_profile"

    def plot(self, x: np.ndarray, phi: np.ndarray,
             V_list: list[float] | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Electrostatic Potential Profiles", fontsize=12, pad=12)
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
        self._save("potential_profile.png", plt)


class ElectricFieldPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "electric_field"

    def plot(self, x: np.ndarray, E: np.ndarray,
             V_list: list[float] | None = None,
             Vbr: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Electric Field Profiles", fontsize=12, pad=12)
        x_um = x * 1e4

        if V_list is not None:
            Vex_list = (
                [v - Vbr for v in V_list]
                if Vbr is not None and Vbr > 0
                else V_list
            )
            for i, (E_i, Vex) in enumerate(zip(E, Vex_list)):
                ax.plot(x_um, -E_i / 1e4, label=f"Vex = {Vex:.0f} V",
                        color=plt.cm.viridis(i / len(E)))
        else:
            ax.plot(x_um, -np.atleast_1d(E) / 1e4)

        ax.set_xlim(2, 5)
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Electric Field (V/µm)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("electric_field_profile.png", plt)


class BreakdownSweepPlotter(BasePlotter):
    @property
    def name(self) -> str:
        return "breakdown_sweep"

    def plot(self, Vbias: np.ndarray, Pe_max: np.ndarray) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Breakdown Sweep: Max Electron Trigger Probability", fontsize=12, pad=12)
        ax.plot(Vbias, Pe_max, "o-", lw=2)
        ax.axhline(y=0.99, color="r", ls="--", alpha=0.5, label="Pe=0.99")
        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("Max P_e")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("breakdown_sweep.png", plt)


class PeakFieldVsBiasPlotter(BasePlotter):
    """Plot peak electric field magnitude vs bias voltage."""

    @property
    def name(self) -> str:
        return "peak_field_vs_bias"

    def plot(self, Vbias: np.ndarray, E_peak: np.ndarray,
             Vbr: float | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Vbias, E_peak / 1e4, "o-", lw=2, color="#1f77b4")
        if Vbr is not None:
            ax.axvline(x=Vbr, color="k", ls=":", alpha=0.5,
                        label=f"Vbr = {Vbr:.1f} V")
        ax.set_xlabel("Bias Voltage (V)")
        ax.set_ylabel("Peak Electric Field (V/µm)")
        ax.set_title("Peak Electric Field vs Bias", fontsize=12, pad=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("peak_field_vs_bias.png", plt)


class EFieldVsAbsorptionPlotter(BasePlotter):
    """Plot E-field profiles for varying absorption-layer widths."""

    @property
    def name(self) -> str:
        return "efield_vs_absorption"

    def plot(self, results: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Electric Field vs Absorption Layer Width", fontsize=12, pad=12)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i, (w_um, x, E) in enumerate(results):
            x_um = x * 1e4
            ax.plot(x_um, -E / 1e4, lw=2, color=colors[i % len(colors)],
                    label=f"W_abs = {w_um:.1f} µm")
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Electric Field (V/µm)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("efield_vs_absorption.png", plt)


class EFieldVsMultiplicationPlotter(BasePlotter):
    """Plot E-field profiles for varying multiplication-layer widths."""

    @property
    def name(self) -> str:
        return "efield_vs_multiplication"

    def plot(self, results: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Electric Field vs Multiplication Layer Width", fontsize=12, pad=12)
        colors = ["#d62728", "#9467bd", "#8c564b"]
        for i, (w_um, x, E) in enumerate(results):
            x_um = x * 1e4
            ax.plot(x_um, -E / 1e4, lw=2, color=colors[i % len(colors)],
                    label=f"W_mult = {w_um:.1f} µm")
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Electric Field (V/µm)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("efield_vs_multiplication.png", plt)


class BandDiagramPlotter(BasePlotter):
    """Plot conduction band (Ec), valence band (Ev), and Fermi level vs depth."""

    @property
    def name(self) -> str:
        return "band_diagram"

    def plot(self, x: np.ndarray, Ec: np.ndarray, Ev: np.ndarray,
             Eg: np.ndarray,
             Ef: float | None = None,
             layer_bounds_um: list[float] | None = None) -> None:
        plt = self._import()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Equilibrium Energy Band Diagram", fontsize=12, pad=12)
        x_um = x * 1e4

        ax.plot(x_um, Ec, lw=2, color="#1f77b4", label="$E_C$")
        ax.plot(x_um, Ev, lw=2, color="#d62728", label="$E_V$")

        if Ef is not None:
            ax.axhline(y=Ef, color="#2ca02c", ls="--", lw=1.5, label=f"$E_F$ = {Ef:.3f} eV")

        eg_mid = Eg[len(Eg) // 2]
        ax.fill_between(x_um, Ev, Ec, alpha=0.08, color="gray",
                        label=f"$E_g$ = {eg_mid:.2f} eV")

        if layer_bounds_um is not None:
            for xb in layer_bounds_um:
                ax.axvline(x=xb, color="k", ls=":", alpha=0.3)

        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("Energy (eV)")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("band_diagram.png", plt)
