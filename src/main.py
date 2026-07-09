#!/usr/bin/env python3
"""InGaAs/InP SAGCM SPAD Simulator (1D center-axis)."""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .core.constants import h, c
from .core.material import Material
from .core.absorption import InterpolatedAbsorption
from .core.layer import Layer
from .core.device import Device
from .simulator import SPADSimulator
from .avalanche.afterpulsing import AfterpulsingModel
from .avalanche.excess_noise import ExcessNoiseFactor
from .transport.jitter import TimingJitter
from .utils._logging import get_logger, set_log_level
from .utils.loaders import load_materials, load_absorption, load_device
from .utils.plotter import get_plotter

log = get_logger()

_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
_plot_dir = os.path.normpath(os.path.join(_data_dir, "..", "plots", "spad"))
_optical_power = 1e-6


# ================================================================
# 1. DATA INGESTION CONFIG
# ================================================================

@dataclass
class DataIngestionConfig:
    """Configuration for all input data paths and simulation parameters."""

    device_xml: str = ""
    materials_xml: str = ""
    absorption_xml: str = ""
    output_dir: str = ""

    # Simulation parameters
    optical_power_W: float = 1e-6
    detector_area_cm2: float = 1e-6
    target_wavelengths_nm: List[int] = field(default_factory=lambda: [905, 1310, 1550])
    excess_voltages_V: List[float] = field(default_factory=lambda: [1, 3, 5, 8])
    temperature_K: float = 300.0
    mc_N_sim: int = 20
    mc_N_threshold: int = 30
    mc_dt: float = 5e-15
    temp_sweep_K: List[int] = field(default_factory=lambda: [285, 315])

    @classmethod
    def from_defaults(cls) -> DataIngestionConfig:
        base = os.path.join(os.path.dirname(__file__), "..", "data")
        out = os.path.normpath(os.path.join(base, "..", "plots", "spad"))
        return cls(
            device_xml=os.path.join(base, "device_sagcm.xml"),
            materials_xml=os.path.join(base, "materials.xml"),
            absorption_xml=os.path.join(base, "absorption.xml"),
            output_dir=out,
        )


# ================================================================
# 2. DATA INGESTION SERVICE
# ================================================================

class DataIngestionService:
    """Loads device, material, and absorption data; builds Device objects."""

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def load_materials(self) -> Dict[str, MaterialData]:
        from .utils.loaders import MaterialData as _MD
        return load_materials(self.config.materials_xml)

    def load_absorption(self) -> Dict[str, AbsorptionData]:
        from .utils.loaders import AbsorptionData as _AD
        return load_absorption(self.config.absorption_xml)

    def load_device_spec(self) -> DeviceSpec:
        from .utils.loaders import DeviceSpec as _DS
        return load_device(self.config.device_xml)

    def build_device(self, T: float | None = None) -> Device:
        cfg = self.load_device_spec()
        mat_data = self.load_materials()
        abs_data = self.load_absorption()
        T_use = T if T is not None else cfg.temperature

        materials = {
            name: Material(data, absorption=InterpolatedAbsorption(abs_data.get(name)),
                           T=T_use)
            for name, data in mat_data.items()
        }
        layers = [
            Layer(
                thickness=lyr["thickness_cm"],
                doping_type=lyr["doping_type"],
                doping_A=lyr.get("doping_A", 0.0),
                doping_m=lyr.get("doping_m", 0.0),
                material=lyr["material"],
            )
            for lyr in cfg.layers
        ]
        return Device(layers, materials, no_of_nodes=cfg.nx)

    def build_simulator(self, T: float | None = None) -> SPADSimulator:
        dev = self.build_device(T)
        return SPADSimulator(dev, detector_area=self.config.detector_area_cm2)

    def build_simulator_at_temp(self, T: float) -> tuple[SPADSimulator, float]:
        sim = self.build_simulator(T)
        Vbr, _ = sim.find_breakdown(V_start=0, V_max=100, V_step=5.0)
        return sim, Vbr if Vbr else 75.0


# ================================================================
# 3. SIMULATION ARTIFACT (output container)
# ================================================================

@dataclass
class SimulationArtifact:
    """Structured container for all simulation results."""

    # Device info
    Vbr_V: float = 0.0
    T_K: float = 300.0
    detector_area_cm2: float = 1e-6
    grid_N: int = 0
    grid_dx_cm: float = 0.0
    total_thickness_cm: float = 0.0
    n_layers: int = 0

    # Dark current
    I_dark_A: float = 0.0
    DCR_cps: float = 0.0

    # PDP max at key wavelengths
    pdp_max: Dict[str, float] = field(default_factory=dict)

    # Afterpulsing
    ap_N_T: float = 1e12
    ap_tau_c_s: float = 1e-6
    ap_P_ap_1us: float = 0.0
    ap_holdoff_1pct_s: float = 0.0

    # Excess noise
    en_M_max: float = 0.0
    en_F_max: float = 0.0
    en_k_eff: float = 0.5

    # PDE
    pde_max: float = 0.0
    pde_wavelength_nm: int = 1310

    # Jitter
    jitter_sigma_s: float = 0.0
    jitter_fwhm_s: float = 0.0

    # Temperature sweeps
    dcr_vs_temp: Dict[str, Any] = field(default_factory=dict)
    pdp_vs_temp: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": {
                "Vbr_V": self.Vbr_V,
                "T_K": self.T_K,
                "detector_area_cm2": self.detector_area_cm2,
                "grid_N": self.grid_N,
                "grid_dx_cm": self.grid_dx_cm,
                "total_thickness_cm": self.total_thickness_cm,
                "n_layers": self.n_layers,
            },
            "dark_current": {
                "I_dark_A": self.I_dark_A,
                "DCR_cps": self.DCR_cps,
                "Vex_V": 3.0,
            },
            "pdp_max": self.pdp_max,
            "afterpulsing": {
                "N_T": self.ap_N_T,
                "tau_c": self.ap_tau_c_s,
                "P_ap_1us": self.ap_P_ap_1us,
                "holdoff_optimal_1pct_s": self.ap_holdoff_1pct_s,
            },
            "excess_noise": {
                "M_max": self.en_M_max,
                "F_max": self.en_F_max,
                "k_eff": self.en_k_eff,
            },
            "pde_1310nm": {
                "pde_max": self.pde_max,
                "wavelength_nm": self.pde_wavelength_nm,
            },
            "jitter": {
                "sigma_s": self.jitter_sigma_s,
                "fwhm_s": self.jitter_fwhm_s,
            },
            "dcr_vs_temperature": self.dcr_vs_temp,
            "pdp_vs_temperature": self.pdp_vs_temp,
        }


# ================================================================
# 4. ARTIFACT WRITER (saves to XML)
# ================================================================

class ArtifactWriter:
    """Writes SimulationArtifact to XML and optionally JSON."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _add_element(self, parent: ET.Element, tag: str,
                     text: str | float | int | None = None,
                     attrib: Dict[str, str] | None = None) -> ET.Element:
        el = ET.SubElement(parent, tag, attrib or {})
        if text is not None:
            el.text = str(text)
        return el

    def write_xml(self, artifact: SimulationArtifact,
                  filename: str = "sim_results.xml") -> str:
        root = ET.Element("spad_simulation")

        # Device section
        dev_el = self._add_element(root, "device")
        self._add_element(dev_el, "breakdown_voltage_V", f"{artifact.Vbr_V:.2f}")
        self._add_element(dev_el, "temperature_K", f"{artifact.T_K:.1f}")
        self._add_element(dev_el, "detector_area_cm2", f"{artifact.detector_area_cm2:.6e}")
        self._add_element(dev_el, "grid_N", artifact.grid_N)
        self._add_element(dev_el, "grid_dx_cm", f"{artifact.grid_dx_cm:.6e}")
        self._add_element(dev_el, "total_thickness_cm", f"{artifact.total_thickness_cm:.6e}")
        self._add_element(dev_el, "n_layers", artifact.n_layers)

        # Dark current section
        dc_el = self._add_element(root, "dark_current")
        self._add_element(dc_el, "I_dark_A", f"{artifact.I_dark_A:.6e}")
        self._add_element(dc_el, "DCR_cps", f"{artifact.DCR_cps:.6e}")
        self._add_element(dc_el, "excess_voltage_V", "3.0")

        # PDP max section
        pdp_el = self._add_element(root, "pdp_max")
        for wl_key, val in artifact.pdp_max.items():
            self._add_element(pdp_el, f"PDP_{wl_key}", f"{val:.6f}",
                              attrib={"wavelength": wl_key})

        # Afterpulsing section
        ap_el = self._add_element(root, "afterpulsing")
        self._add_element(ap_el, "trap_density_cm3", f"{artifact.ap_N_T:.3e}")
        self._add_element(ap_el, "emission_time_constant_s", f"{artifact.ap_tau_c_s:.3e}")
        self._add_element(ap_el, "P_ap_at_1us", f"{artifact.ap_P_ap_1us:.6f}")
        self._add_element(ap_el, "holdoff_for_1pct_s", f"{artifact.ap_holdoff_1pct_s:.6e}")

        # Excess noise section
        en_el = self._add_element(root, "excess_noise")
        self._add_element(en_el, "M_max", f"{artifact.en_M_max:.2f}")
        self._add_element(en_el, "F_max", f"{artifact.en_F_max:.4f}")
        self._add_element(en_el, "k_eff", f"{artifact.en_k_eff:.4f}")

        # PDE section
        pde_el = self._add_element(root, "photon_detection_efficiency")
        self._add_element(pde_el, "PDE_max", f"{artifact.pde_max:.6f}")
        self._add_element(pde_el, "wavelength_nm", artifact.pde_wavelength_nm)

        # Jitter section
        jit_el = self._add_element(root, "timing_jitter")
        self._add_element(jit_el, "sigma_s", f"{artifact.jitter_sigma_s:.6e}")
        self._add_element(jit_el, "FWHM_s", f"{artifact.jitter_fwhm_s:.6e}")

        # DCR vs temperature section
        if artifact.dcr_vs_temp:
            dcrT_el = self._add_element(root, "dcr_vs_temperature")
            temps = artifact.dcr_vs_temp.get("temperatures_K", [])
            dcr_vals = artifact.dcr_vs_temp.get("DCR_cps", [])
            for t, d in zip(temps, dcr_vals):
                self._add_element(dcrT_el, "data_point",
                                  attrib={"temperature_K": str(t),
                                          "DCR_cps": str(d)})

        # PDP vs temperature section
        if artifact.pdp_vs_temp:
            pdpT_el = self._add_element(root, "pdp_vs_temperature")
            temps = artifact.pdp_vs_temp.get("temperatures_K", [])
            pdp_data = artifact.pdp_vs_temp.get("pdp", {})
            for wl, vals in pdp_data.items():
                wl_el = self._add_element(pdpT_el, "wavelength",
                                          attrib={"nm": str(wl)})
                for t, v in zip(temps, vals):
                    self._add_element(wl_el, "data_point",
                                      attrib={"temperature_K": str(t),
                                              "PDP": str(v)})

        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        path = os.path.join(self.output_dir, filename)
        tree.write(path, encoding="unicode", xml_declaration=True)
        log.info("  XML artifact saved to %s", path)
        return path


def build_sagcm_spad() -> Device:
    """Build the default SAGCM SPAD device from data files."""
    cfg = DataIngestionConfig.from_defaults()
    svc = DataIngestionService(cfg)
    return svc.build_device()


def _find_breakdown(sim: SPADSimulator) -> float:
    Vbr, _ = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    if Vbr is None:
        raise ValueError("No breakdown detected")
    log.info(f"Vbr = {Vbr:.1f} V")
    return Vbr


def _plot_device_structure(sim: SPADSimulator) -> None:
    for i, lyr in enumerate(sim.device.layers):
        label = "i"
        if lyr.doping_A > 1e14:
            label = "n+" if lyr.is_donor else "p+"
        log.info(f"Layer {i}:  {label:3s}  {lyr.thickness * 1e4:.2f} um  "
                 f"{lyr.material:8s}  A={lyr.doping_A:.1e}  ({lyr.doping_type})")

    p = get_plotter("device_structure", plot_dir=_plot_dir)
    p.plot(sim.grid.x, sim.device.material.mat_name,
           sim.device.net_doping_on_grid, sim.device.net_doping_on_grid)


def _run_field_sweep(sim: SPADSimulator, Vbr: float) -> None:
    Vex_list = [0, 1, 2, 3, 4, 5]
    V_list = [Vbr + vex for vex in Vex_list]
    phi_list, E_list = [], []
    for V in V_list:
        phi, E, _ = sim.solve_poisson(float(V))
        phi_list.append(phi)
        E_list.append(E)
        log.info(f"Vex = {V - Vbr:.0f} V  phi_max = {phi.max():.1f}  "
                 f"|E|_max = {np.max(np.abs(E)):.2e}")

    get_plotter("potential_profile", plot_dir=_plot_dir).plot(
        sim.grid.x, np.array(phi_list), V_list)
    get_plotter("electric_field", plot_dir=_plot_dir).plot(
        sim.grid.x, np.array(E_list), V_list, Vbr=Vbr)


def _run_dark_current_sweep(sim: SPADSimulator, Vbr: float) -> None:
    Vex_range = np.linspace(0, 10, 11)
    I_dark, dcr = [], []
    for Vex in Vex_range:
        try:
            dc = sim.compute_dark_current(float(Vbr + Vex))
            I_dark.append(dc["I_dark"])
            dcr.append(dc["DCR"])
        except Exception:
            I_dark.append(np.nan)
            dcr.append(np.nan)

    I_dark, dcr = np.array(I_dark), np.array(dcr)
    mask = np.isfinite(I_dark)
    if np.any(mask):
        log.info(f"I_dark: {np.nanmin(I_dark):.2e} - {np.nanmax(I_dark):.2e} A")
        log.info(f"DCR:    {np.nanmin(dcr):.2e} - {np.nanmax(dcr):.2e} cps")
        get_plotter("dark_current", plot_dir=_plot_dir).plot(Vex_range[mask], I_dark[mask])
        get_plotter("dcr", plot_dir=_plot_dir).plot(Vex_range[mask], dcr[mask])


def _run_iv_characteristic(sim: SPADSimulator, Vbr: float) -> None:
    V_sweep = np.linspace(0, Vbr + 10, 61)
    I_dark, I_light = [], []
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)
            I_dark.append(dc["I_dark"])
            photo = sim.compute_photocurrent(float(V), power=_optical_power,
                                             E=E, Pe=Pe, Ph=Ph, xr=xr)
            I_light.append(photo + dc["I_dark"])
        except Exception:
            I_dark.append(np.nan)
            I_light.append(np.nan)

    I_dark, I_light = np.array(I_dark), np.array(I_light)
    mask = np.isfinite(I_dark)
    if np.any(mask):
        get_plotter("iv_characteristic", plot_dir=_plot_dir).plot(
            V_sweep[mask], I_dark[mask], I_light=I_light[mask],
            optical_power=_optical_power)


def _run_pdp_spectrum(sim: SPADSimulator, Vbr: float) -> None:
    pdp_wavelengths = np.linspace(900, 1700, 41) * 1e-9
    pdp_spectra, vex_list_pdp = [], []
    for Vex in [1, 3, 5, 8]:
        try:
            _, E, _ = sim.solve_poisson(Vbr + Vex)
            Pe, Ph = sim._trigger_for_pdp(E)
            _, xr, _ = sim.depletion_width(Vbr + Vex)
            pdp = sim.compute_pdp_spectrum(
                pdp_wavelengths, float(Vex), material_name="InGaAs",
                E=E, Pe=Pe, Ph=Ph, xr=xr)
            pdp = np.clip(pdp, 0, 1)
            pdp_spectra.append(pdp)
            vex_list_pdp.append(Vex)
            log.info(f"  Vex = {Vex} V: PDP max = {np.max(pdp) * 100:.4f}%")
        except Exception as e:
            log.info(f"  Vex = {Vex} V: {e}")

    if pdp_spectra:
        get_plotter("pdp", plot_dir=_plot_dir).plot(
            pdp_wavelengths, np.array(pdp_spectra), vex_list_pdp)


def _run_pdp_vs_vex(sim: SPADSimulator, Vbr: float) -> None:
    dead_zone_layers, absorber = sim.pdp_model.find_absorber(
        sim.device.layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)

    wl_apt = np.array([1100, 1310, 1550, 1610])
    vex_pts = np.linspace(0, 10, 11)
    pdp_dict = {lam: [] for lam in wl_apt}
    for Vex in vex_pts:
        try:
            _, E, _ = sim.solve_poisson(Vbr + Vex)
            Pe, Ph = sim._trigger_for_pdp(E)
            _, xr, _ = sim.depletion_width(Vbr + Vex)
            Ptr = Pe + Ph - Pe * Ph
            x_end = min(xr, dead_zone + absorber.thickness)
            mask = (sim.grid.x >= dead_zone) & (sim.grid.x <= x_end)
            xx = sim.grid.x[mask] - dead_zone
            for lam in wl_apt:
                trans = sim.pdp_model.dead_zone_transmission(lam * 1e-9, dead_zone_layers)
                pdp = sim.pdp_model.pdp_integral(
                    lam * 1e-9, xx, Ptr[mask], trans, sim.grid.dx,
                    material_name="InGaAs")
                pdp_dict[lam].append(pdp)
        except Exception:
            for lam in wl_apt:
                pdp_dict[lam].append(0.0)

    get_plotter("pdp_vs_vex", plot_dir=_plot_dir).plot(
        vex_pts, {lam: np.array(v) for lam, v in pdp_dict.items()},
        wavelengths_nm=wl_apt)


def _run_comprehensive_iv(sim: SPADSimulator, Vbr: float) -> None:
    dead_zone_layers, absorber = sim.pdp_model.find_absorber(
        sim.device.layers, "InGaAs")
    dead_zone = sum(l.thickness for l in dead_zone_layers)

    alpha_arr = np.array([
        sim.materials[lyr.material].absorption_coefficient(1310e-9)
        for lyr in sim.device.layers
    ])
    alpha_grid = np.zeros_like(sim.grid.x)
    xs = 0.0
    for lyr, alpha_val in zip(sim.device.layers, alpha_arr):
        xe = xs + lyr.thickness
        mask = (sim.grid.x >= xs - 1e-16) & (sim.grid.x <= xe + 1e-16)
        alpha_grid[mask] = alpha_val
        xs = xe

    Eph = h * c / 1310e-9
    phi_photon = _optical_power / (Eph * sim.detector_area)
    absorber_start = dead_zone
    absorber_end = dead_zone + absorber.thickness

    V_sweep = np.linspace(Vbr - 5, Vbr + 10, 11)
    I_dark, I_photo_prim, I_total, M_vals = [], [], [], []
    for V in V_sweep:
        try:
            _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
            dc = sim.compute_dark_current(float(V), E=E)
            Ptr = Pe + Ph - Pe * Ph
            M = min(1.0 / (1.0 - float(np.max(Ptr)) + 1e-15), 10000.0)
            M_vals.append(M)
            I_dark.append(dc["I_dark"])

            J_pp = sim.pdp_model.photocurrent_density(
                sim.grid.x, alpha_grid, phi_photon, absorber_start, absorber_end)
            # Trapezoidal rule: O(h²), handles non-uniform grid near heterojunctions
            I_pp = float(np.trapezoid(J_pp, sim.grid.x) * sim.detector_area)
            I_photo_prim.append(I_pp)
            I_total.append(dc["I_dark"] + I_pp * M)
        except Exception:
            I_dark.append(np.nan)
            I_photo_prim.append(np.nan)
            I_total.append(np.nan)
            M_vals.append(np.nan)

    I_dark, I_pp, I_total, M_vals = [np.array(a) for a in
                                      (I_dark, I_photo_prim, I_total, M_vals)]
    mask = np.isfinite(I_dark)
    if np.any(mask):
        get_plotter("comprehensive_iv", plot_dir=_plot_dir).plot(
            V_sweep[mask], I_dark[mask],
            I_photo_primary=I_pp[mask], I_total_illuminated=I_total[mask],
            gain=M_vals[mask], Vbr=Vbr)


def _run_trigger_profiles(sim: SPADSimulator, Vbr: float) -> None:
    Vex_list = [1, 3, 5]
    Pe_list, Ph_list, V_list = [], [], []
    for Vex in Vex_list:
        try:
            Pe, Ph, E = sim.solve_trigger(Vbr + Vex)
            Pe_list.append(Pe)
            Ph_list.append(Ph)
            V_list.append(Vbr + Vex)
            log.info(f"  Trigger Vex={Vex}V  Pe_max={np.max(Pe):.4f}  "
                     f"Ph_max={np.max(Ph):.4f}")
        except Exception as e:
            log.info(f"  Trigger Vex={Vex}V failed: {e}")

    if Pe_list:
        get_plotter("trigger_probability", plot_dir=_plot_dir).plot(
            sim.grid.x, np.array(Pe_list), np.array(Ph_list), V_list)


def _run_afterpulsing(sim: SPADSimulator, Vbr: float) -> dict:
    ap = AfterpulsingModel(N_T=1e12, tau_c=1e-6, Vbr=Vbr)
    holdoff_pts = np.logspace(-9, -4, 100)
    P_ap = np.array([ap.afterpulsing_probability(t) for t in holdoff_pts])

    get_plotter("afterpulsing", plot_dir=_plot_dir).plot(
        holdoff_pts, P_ap, N_T=ap.N_T, tau_c=ap.tau_c)

    holdoff_1us = ap.afterpulsing_probability(1e-6)
    holdoff_opt = ap.holdoff_optimal(0.01)
    log.info(f"  Afterpulsing: P_ap(1µs)={holdoff_1us*100:.1f}%  "
             f"holdoff_1%={holdoff_opt*1e6:.1f}µs")
    return {"N_T": ap.N_T, "tau_c": ap.tau_c,
            "P_ap_1us": holdoff_1us, "holdoff_optimal_1pct_s": holdoff_opt}


def _run_excess_noise(sim: SPADSimulator, Vbr: float) -> dict:
    Vex_range = np.linspace(0.5, 10, 20)
    M_vals, F_vals = [], []
    k_eff = None

    for Vex in Vex_range:
        try:
            _, E, Pe, Ph, _, _ = sim.get_fields(Vbr + Vex)
            Ptr = Pe + Ph - Pe * Ph
            Ptr_max = float(np.max(Ptr))
            M = min(1.0 / (1.0 - Ptr_max + 1e-15), 10000.0)

            alpha = sim.ionization.alpha(E)
            beta = sim.ionization.beta(E)
            active = np.abs(E) > 1e4
            if np.any(active):
                k_eff = float(np.mean(beta[active]) / np.mean(alpha[active]))
            else:
                k_eff = 0.5

            en = ExcessNoiseFactor(k_eff=k_eff)
            F = en.f(M)
            M_vals.append(M)
            F_vals.append(F)
        except Exception:
            M_vals.append(np.nan)
            F_vals.append(np.nan)

    M_arr, F_arr = np.array(M_vals), np.array(F_vals)
    mask = np.isfinite(M_arr) & np.isfinite(F_arr)
    if np.any(mask):
        get_plotter("excess_noise", plot_dir=_plot_dir).plot(
            M_arr[mask], F_arr[mask], k_eff=k_eff)

    M_max = float(np.nanmax(M_arr[mask])) if np.any(mask) else 0.0
    F_max = float(np.nanmax(F_arr[mask])) if np.any(mask) else 0.0
    log.info(f"  Excess noise: M_max={M_max:.1f}  F_max={F_max:.2f}  k_eff={k_eff:.3f}")
    return {"M_max": M_max, "F_max": F_max, "k_eff": k_eff}


def _run_pde_vs_bias(sim: SPADSimulator, Vbr: float) -> dict:
    Vex_range = np.linspace(0, 10, 21)
    wavelength = 1310e-9
    PDE_vals = []

    for Vex in Vex_range:
        try:
            pdp_spectrum = sim.compute_pdp_spectrum(
                np.array([wavelength]), float(Vex),
                material_name="InGaAs")
            PDE_vals.append(float(pdp_spectrum[0]))
        except Exception:
            PDE_vals.append(0.0)

    PDE_arr = np.array(PDE_vals)
    get_plotter("pde", plot_dir=_plot_dir).plot(Vex_range, PDE_arr)

    pde_max = float(np.max(PDE_arr))
    log.info(f"  PDE(1310nm): max={pde_max*100:.1f}%")
    return {"pde_max": pde_max, "wavelength_nm": 1310}


def _run_jitter(sim: SPADSimulator, Vbr: float) -> dict:
    try:
        ens = sim.run_mc_ensemble(Vbr + 3.0, N_sim=20, N_threshold=30, dt=5e-15)
        t_detect = TimingJitter.extract_detection_times(ens)

        if len(t_detect) == 0:
            log.info("  Jitter: no successful avalanches")
            return {"sigma_s": np.nan, "fwhm_s": np.nan}

        stats = TimingJitter.statistics(t_detect)
        fwhm_val = TimingJitter.fwhm(t_detect)

        get_plotter("jitter_histogram", plot_dir=_plot_dir).plot(
            t_detect, bins=30, fwhm=fwhm_val, sigma=stats["std"])

        log.info(f"  Jitter: σ={stats['std']*1e12:.1f}ps  "
                 f"FWHM={fwhm_val*1e12:.1f}ps  N={stats['N']}")
        return {"sigma_s": stats["std"], "fwhm_s": fwhm_val,
                "mean_s": stats["mean"], "N": stats["N"]}
    except Exception as e:
        log.info(f"  Jitter simulation failed: {e}")
        return {"sigma_s": np.nan, "fwhm_s": np.nan}


def _build_sim_at_temp(T: float) -> tuple[SPADSimulator, float]:
    """Build a simulator at temperature T and return (sim, Vbr)."""
    cfg = DataIngestionConfig.from_defaults()
    svc = DataIngestionService(cfg)
    return svc.build_simulator_at_temp(T)


def _run_dcr_vs_temp(sim: SPADSimulator, Vbr: float) -> dict:
    temps = np.array([285, 315])
    Vex = 3.0
    DCR_vals = []

    for T in temps:
        try:
            sim_T, Vbr_T = _build_sim_at_temp(T)
            dc = sim_T.compute_dark_current(Vbr_T + Vex)
            DCR_vals.append(dc["DCR"])
            log.info(f"  T={T}K  Vbr={Vbr_T:.1f}V  DCR={dc['DCR']:.2e} cps")
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            DCR_vals.append(np.nan)

    DCR_arr = np.array(DCR_vals)
    mask = np.isfinite(DCR_arr)
    if np.any(mask):
        get_plotter("dcr_vs_temp", plot_dir=_plot_dir).plot(
            temps[mask], DCR_arr[mask], Vex=Vex)

    return {"temperatures_K": temps.tolist(), "DCR_cps": DCR_arr.tolist(),
            "Vex": Vex}


def _run_pdp_vs_temp(sim: SPADSimulator, Vbr: float) -> dict:
    temps = np.array([285, 315])
    Vex = 3.0
    wavelengths = [1310, 1550]
    pdp_dict = {wl: [] for wl in wavelengths}

    for T in temps:
        try:
            sim_T, Vbr_T = _build_sim_at_temp(T)
            for wl in wavelengths:
                pdp_spectrum = sim_T.compute_pdp_spectrum(
                    np.array([wl * 1e-9]), float(Vex), material_name="InGaAs")
                pdp_dict[wl].append(float(pdp_spectrum[0]))
            log.info(f"  T={T}K  Vbr={Vbr_T:.1f}V  "
                     f"PDP1310={pdp_dict[1310][-1]*100:.1f}%  "
                     f"PDP1550={pdp_dict[1550][-1]*100:.1f}%")
        except Exception as e:
            log.info(f"  T={T}K failed: {e}")
            for wl in wavelengths:
                pdp_dict[wl].append(0.0)

    pdp_plot = {wl: np.array(vals) for wl, vals in pdp_dict.items()}
    get_plotter("pdp_vs_temp", plot_dir=_plot_dir).plot(
        temps, pdp_plot, wavelengths_nm=np.array(wavelengths))

    return {"temperatures_K": temps.tolist(), "pdp": pdp_dict, "Vex": Vex}


def _collect_artifact(Vbr: float, sim: SPADSimulator,
                      afterpulsing: dict, excess_noise: dict,
                      pde: dict, jitter: dict,
                      dark_current: dict | None = None,
                      pdp_max: dict | None = None,
                      dcr_temp: dict | None = None,
                      pdp_temp: dict | None = None) -> SimulationArtifact:
    """Collect all metrics into a SimulationArtifact."""
    dc = dark_current or {}
    return SimulationArtifact(
        Vbr_V=Vbr,
        T_K=sim.T,
        detector_area_cm2=sim.detector_area,
        grid_N=sim.grid.no_of_nodes,
        grid_dx_cm=sim.grid.dx,
        total_thickness_cm=sim.device.L,
        n_layers=len(sim.device.layers),
        I_dark_A=dc.get("I_dark_A", 0.0),
        DCR_cps=dc.get("DCR_cps", 0.0),
        pdp_max=pdp_max or {},
        ap_N_T=afterpulsing.get("N_T", 1e12),
        ap_tau_c_s=afterpulsing.get("tau_c", 1e-6),
        ap_P_ap_1us=afterpulsing.get("P_ap_1us", 0.0),
        ap_holdoff_1pct_s=afterpulsing.get("holdoff_optimal_1pct_s", 0.0),
        en_M_max=excess_noise.get("M_max", 0.0),
        en_F_max=excess_noise.get("F_max", 0.0),
        en_k_eff=excess_noise.get("k_eff", 0.5),
        pde_max=pde.get("pde_max", 0.0),
        pde_wavelength_nm=pde.get("wavelength_nm", 1310),
        jitter_sigma_s=jitter.get("sigma_s", 0.0),
        jitter_fwhm_s=jitter.get("fwhm_s", 0.0),
        dcr_vs_temp=dcr_temp or {},
        pdp_vs_temp=pdp_temp or {},
    )


def main() -> None:
    set_log_level(logging.INFO)
    cfg = DataIngestionConfig.from_defaults()
    svc = DataIngestionService(cfg)
    sim = svc.build_simulator()

    _plot_device_structure(sim)
    Vbr = _find_breakdown(sim)
    _run_field_sweep(sim, Vbr)
    _run_dark_current_sweep(sim, Vbr)
    _run_iv_characteristic(sim, Vbr)
    _run_pdp_spectrum(sim, Vbr)
    _run_pdp_vs_vex(sim, Vbr)
    _run_comprehensive_iv(sim, Vbr)
    _run_trigger_profiles(sim, Vbr)
    afterpulsing = _run_afterpulsing(sim, Vbr)
    excess_noise = _run_excess_noise(sim, Vbr)
    pde = _run_pde_vs_bias(sim, Vbr)
    jitter = _run_jitter(sim, Vbr)

    # Collect dark current metrics at Vex = 3 V
    dark_current_metrics = {}
    try:
        dc3 = sim.compute_dark_current(Vbr + 3.0)
        dark_current_metrics = {
            "I_dark_A": dc3["I_dark"],
            "DCR_cps": dc3["DCR"],
            "Vex_V": 3.0,
        }
        log.info(f"  Dark current @ Vex=3V: I={dc3['I_dark']:.2e} A  "
                 f"DCR={dc3['DCR']:.2e} cps")
    except Exception as e:
        log.info(f"  Dark current collection failed: {e}")

    # Collect PDP max at key wavelengths
    pdp_max_metrics = {}
    for wl_nm in cfg.target_wavelengths_nm:
        try:
            pdp_spectrum = sim.compute_pdp_spectrum(
                np.array([wl_nm * 1e-9]), 3.0, material_name="InGaAs")
            pdp_max_metrics[f"{wl_nm}nm"] = float(np.max(pdp_spectrum))
            log.info(f"  PDP max @ {wl_nm}nm, Vex=3V: {np.max(pdp_spectrum)*100:.1f}%")
        except Exception:
            pdp_max_metrics[f"{wl_nm}nm"] = 0.0

    # Write XML artifact immediately (before slow temp sweeps)
    artifact = _collect_artifact(Vbr, sim, afterpulsing, excess_noise,
                                 pde, jitter, dark_current_metrics,
                                 pdp_max_metrics)
    writer = ArtifactWriter(cfg.output_dir)
    writer.write_xml(artifact)

    # Slow temperature sweeps — results update the artifact when complete
    dcr_temp = _run_dcr_vs_temp(sim, Vbr)
    pdp_temp = _run_pdp_vs_temp(sim, Vbr)

    # Update XML with temperature sweep results
    if dcr_temp or pdp_temp:
        artifact.dcr_vs_temp = dcr_temp
        artifact.pdp_vs_temp = pdp_temp
        writer.write_xml(artifact)

    log.info("\n  Plots saved to %s/", cfg.output_dir)


if __name__ == "__main__":
    main()
