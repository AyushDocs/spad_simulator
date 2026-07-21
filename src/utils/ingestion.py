"""Data ingestion config and service for loading device/material/absorption data."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..core.material import Material
from ..core.absorption import InterpolatedAbsorption
from ..core.layer import Layer
from ..core.device import Device
from ..simulator import SPADSimulator
from .loaders import (load_materials, load_absorption, load_device,
                       MaterialData, AbsorptionData, DeviceSpec,
                       PlotConfig, load_plot_config)
from ._logging import get_logger

log = get_logger("ingestion")


@dataclass
class DataIngestionConfig:
    """Configuration for all input data paths and simulation parameters."""

    device_xml: str = ""
    materials_xml: str = ""
    absorption_xml: str = ""
    plots_xml: str = ""
    output_dir: str = ""

    # Simulation parameters
    optical_power_W: float = 1e-6
    detector_area_cm2: float = 4.91e-6
    target_wavelengths_nm: List[int] = field(default_factory=lambda: [905, 1310, 1550])
    excess_voltages_V: List[float] = field(default_factory=lambda: [1, 3, 5, 8])
    temperature_K: float = 300.0
    mc_N_sim: int = 20
    mc_N_threshold: int = 30
    mc_dt: float = 5e-15
    temp_sweep_K: List[int] = field(default_factory=lambda: [285, 315])

    @classmethod
    def from_defaults(cls) -> DataIngestionConfig:
        base = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        out = os.path.normpath(os.path.join(base, "..", "plots", "spad"))
        return cls(
            device_xml=os.path.join(base, "device_sagcm.xml"),
            materials_xml=os.path.join(base, "materials.xml"),
            absorption_xml=os.path.join(base, "absorption.xml"),
            plots_xml=os.path.join(base, "plots_config.xml"),
            output_dir=out,
        )


class DataIngestionService:
    """Loads device, material, and absorption data; builds Device objects."""

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def load_materials(self) -> Dict[str, MaterialData]:
        return load_materials(self.config.materials_xml)

    def load_absorption(self) -> Dict[str, AbsorptionData]:
        return load_absorption(self.config.absorption_xml)

    def load_device_spec(self) -> DeviceSpec:
        return load_device(self.config.device_xml)

    def load_plot_config(self) -> PlotConfig:
        return load_plot_config(self.config.plots_xml)

    def build_device(self, T: float | None = None) -> Device:
        cfg = self.load_device_spec()
        mat_data = self.load_materials()
        abs_data = self.load_absorption()
        T_use = T if T is not None else cfg.temperature

        materials = {
            name: Material(data, absorption=InterpolatedAbsorption(abs_data.get(name, abs_data[list(abs_data.keys())[0]])),
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
        return Device(layers=layers, materials=materials, no_of_nodes=cfg.nx)

    def build_simulator(self, T: float | None = None) -> SPADSimulator:
        dev = self.build_device(T)
        return SPADSimulator(dev, detector_area=self.config.detector_area_cm2)

    def build_simulator_at_temp(self, T: float) -> Tuple[SPADSimulator, float]:
        sim = self.build_simulator(T)
        try:
            Vbr, _ = sim.find_breakdown(V_start=30.0, V_max=120, V_step=1.0)
            return sim, Vbr  # type: ignore[return-value]
        except Exception:
            dVbr = (T - 300.0) * 0.002
            return sim, 60.0 + dVbr
