"""Simulation artifact data container and XML writer."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict

from ._logging import get_logger

log = get_logger("artifact")


@dataclass
class SimulationArtifact:
    """Structured container for all simulation results."""

    # Device info
    Vbr_V: float = 0.0
    T_K: float = 300.0
    detector_area_cm2: float = 4.91e-6
    grid_N: int = 0
    grid_dx_cm: float = 0.0
    total_thickness_cm: float = 0.0
    n_layers: int = 0

    # Dark current
    I_dark_A: float = 0.0
    DCR_cps: float = 0.0

    # PDE max at key wavelengths
    pde_max: Dict[str, float] = field(default_factory=dict)

    # Afterpulsing
    ap_N_T: float = 1e12
    ap_tau_c_s: float = 1e-6
    ap_P_ap_1us: float = 0.0
    ap_holdoff_1pct_s: float = 0.0

    # Excess noise
    en_M_max: float = 0.0
    en_F_max: float = 0.0
    en_k_eff: float = 0.5

    # Jitter
    jitter_sigma_s: float = 0.0
    jitter_fwhm_s: float = 0.0

    # Temperature sweeps
    dcr_vs_temp: Dict[str, Any] = field(default_factory=dict)
    pde_vs_temp: Dict[str, Any] = field(default_factory=dict)

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
            "pde_max": self.pde_max,
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
            "jitter": {
                "sigma_s": self.jitter_sigma_s,
                "fwhm_s": self.jitter_fwhm_s,
            },
            "dcr_vs_temperature": self.dcr_vs_temp,
            "pde_vs_temperature": self.pde_vs_temp,
        }


def collect_artifact(Vbr: float, sim: Any, afterpulsing: dict,
                     excess_noise: dict, jitter: dict,
                     dark_current: dict | None = None,
                      pde_max: dict | None = None,
                      dcr_temp: dict | None = None,
                      pde_temp: dict | None = None) -> SimulationArtifact:
    """Collect all simulation metrics into a SimulationArtifact."""
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
        pde_max=pde_max or {},
        ap_N_T=afterpulsing.get("N_T", 1e12),
        ap_tau_c_s=afterpulsing.get("tau_c", 1e-6),
        ap_P_ap_1us=afterpulsing.get("P_ap_1us", 0.0),
        ap_holdoff_1pct_s=afterpulsing.get("holdoff_optimal_1pct_s", 0.0),
        en_M_max=excess_noise.get("M_max", 0.0),
        en_F_max=excess_noise.get("F_max", 0.0),
        en_k_eff=excess_noise.get("k_eff", 0.5),
        jitter_sigma_s=jitter.get("sigma_s", 0.0),
        jitter_fwhm_s=jitter.get("fwhm_s", 0.0),
        dcr_vs_temp=dcr_temp or {},
        pde_vs_temp=pde_temp or {},
    )


class ArtifactWriter:
    """Writes SimulationArtifact to XML."""

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

        dev_el = self._add_element(root, "device")
        self._add_element(dev_el, "breakdown_voltage_V", f"{artifact.Vbr_V:.2f}")
        self._add_element(dev_el, "temperature_K", f"{artifact.T_K:.1f}")
        self._add_element(dev_el, "detector_area_cm2", f"{artifact.detector_area_cm2:.6e}")
        self._add_element(dev_el, "grid_N", artifact.grid_N)
        self._add_element(dev_el, "grid_dx_cm", f"{artifact.grid_dx_cm:.6e}")
        self._add_element(dev_el, "total_thickness_cm", f"{artifact.total_thickness_cm:.6e}")
        self._add_element(dev_el, "n_layers", artifact.n_layers)

        dc_el = self._add_element(root, "dark_current")
        self._add_element(dc_el, "I_dark_A", f"{artifact.I_dark_A:.6e}")
        self._add_element(dc_el, "DCR_cps", f"{artifact.DCR_cps:.6e}")
        self._add_element(dc_el, "excess_voltage_V", "3.0")

        pde_el = self._add_element(root, "pde_max")
        for wl_key, val in artifact.pde_max.items():
            self._add_element(pde_el, f"PDE_{wl_key}", f"{val:.6f}",
                              attrib={"wavelength": wl_key})

        ap_el = self._add_element(root, "afterpulsing")
        self._add_element(ap_el, "trap_density_cm3", f"{artifact.ap_N_T:.3e}")
        self._add_element(ap_el, "emission_time_constant_s", f"{artifact.ap_tau_c_s:.3e}")
        self._add_element(ap_el, "P_ap_at_1us", f"{artifact.ap_P_ap_1us:.6f}")
        self._add_element(ap_el, "holdoff_for_1pct_s", f"{artifact.ap_holdoff_1pct_s:.6e}")

        en_el = self._add_element(root, "excess_noise")
        self._add_element(en_el, "M_max", f"{artifact.en_M_max:.2f}")
        self._add_element(en_el, "F_max", f"{artifact.en_F_max:.4f}")
        self._add_element(en_el, "k_eff", f"{artifact.en_k_eff:.4f}")

        jit_el = self._add_element(root, "timing_jitter")
        self._add_element(jit_el, "sigma_s", f"{artifact.jitter_sigma_s:.6e}")
        self._add_element(jit_el, "FWHM_s", f"{artifact.jitter_fwhm_s:.6e}")

        if artifact.dcr_vs_temp:
            dcrT_el = self._add_element(root, "dcr_vs_temperature")
            temps = artifact.dcr_vs_temp.get("temperatures_K", [])
            dcr_vals = artifact.dcr_vs_temp.get("DCR_cps", [])
            for t, d in zip(temps, dcr_vals):
                self._add_element(dcrT_el, "data_point",
                                  attrib={"temperature_K": str(t),
                                          "DCR_cps": str(d)})

        if artifact.pde_vs_temp:
            pdeT_el = self._add_element(root, "pde_vs_temperature")
            temps = artifact.pde_vs_temp.get("temperatures_K", [])
            pde_data = artifact.pde_vs_temp.get("pde", {})
            for wl, vals in pde_data.items():
                wl_el = self._add_element(pdeT_el, "wavelength",
                                          attrib={"nm": str(wl)})
                for t, v in zip(temps, vals):
                    self._add_element(wl_el, "data_point",
                                      attrib={"temperature_K": str(t),
                                              "PDE": str(v)})

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        path = os.path.join(self.output_dir, filename)
        tree.write(path, encoding="unicode", xml_declaration=True)
        log.info("  XML artifact saved to %s", path)
        return path
