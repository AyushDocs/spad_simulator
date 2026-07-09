"""Plotter registry — name → class mapping, get_plotter, register_plotter."""
from __future__ import annotations

from typing import Any

from ._base import Plotter

_BUILTIN_PLOTTERS: dict[str, type[Plotter]] = {}


def _register_builtins() -> None:
    from .structure import DeviceStructurePlotter, DopingPlotter
    from .fields import PotentialProfilePlotter, ElectricFieldPlotter, BreakdownSweepPlotter
    from .current import (DarkCurrentPlotter, DCRPlotter,
                          IVCharacteristicPlotter, ComprehensiveIVPlotter)
    from .pdp import PDPPlotter, PDPVsExcessVoltagePlotter, PDEPlotter
    from .avalanche import (TriggerProbabilityPlotter, AfterpulsingPlotter,
                            ExcessNoisePlotter)
    from .temporal import (TrajectoryPlotter, JitterPlotter,
                           JitterHistogramPlotter, PopulationPlotter)
    from .temperature import DCRvsTempPlotter, PDPvsTempPlotter

    _BUILTIN_PLOTTERS.update({
        "device_structure": DeviceStructurePlotter,
        "doping": DopingPlotter,
        "potential_profile": PotentialProfilePlotter,
        "electric_field": ElectricFieldPlotter,
        "breakdown_sweep": BreakdownSweepPlotter,
        "dark_current": DarkCurrentPlotter,
        "dcr": DCRPlotter,
        "iv_characteristic": IVCharacteristicPlotter,
        "comprehensive_iv": ComprehensiveIVPlotter,
        "pdp": PDPPlotter,
        "pdp_vs_vex": PDPVsExcessVoltagePlotter,
        "pde": PDEPlotter,
        "trigger_probability": TriggerProbabilityPlotter,
        "afterpulsing": AfterpulsingPlotter,
        "excess_noise": ExcessNoisePlotter,
        "trajectory": TrajectoryPlotter,
        "jitter": JitterPlotter,
        "jitter_histogram": JitterHistogramPlotter,
        "population": PopulationPlotter,
        "dcr_vs_temp": DCRvsTempPlotter,
        "pdp_vs_temp": PDPvsTempPlotter,
    })


def get_plotter(name: str, plot_dir: str = "plots", **kwargs: Any) -> Plotter:
    if not _BUILTIN_PLOTTERS:
        _register_builtins()
    cls = _BUILTIN_PLOTTERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown plotter '{name}'. Available: {list(_BUILTIN_PLOTTERS)}")
    return cls(plot_dir=plot_dir, **kwargs)


def register_plotter(name: str, plotter_cls: type[Plotter]) -> None:
    _BUILTIN_PLOTTERS[name] = plotter_cls
