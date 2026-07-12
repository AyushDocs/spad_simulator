"""Plotter registry — name → class mapping, get_plotter, register_plotter."""
from __future__ import annotations

from typing import Any

from ._base import Plotter

_BUILTIN_PLOTTERS: dict[str, type[Plotter]] = {}


def _register_builtins() -> None:
    from .structure import DeviceStructurePlotter, DopingPlotter
    from .fields import (PotentialProfilePlotter, ElectricFieldPlotter,
                         BreakdownSweepPlotter, PeakFieldVsBiasPlotter)
    from .current import (DarkCurrentPlotter, DarkCurrentComponentsPlotter,
                          DCRPlotter, IVCharacteristicPlotter, ComprehensiveIVPlotter)
    from .pdp import (PDPPlotter, PDPVsExcessVoltagePlotter,
                      AbsorptionProfilePlotter, PDP3DPlotter)
    from .avalanche import (TriggerProbabilityPlotter, TriggerVsVexPlotter,
                            AfterpulsingPlotter,
                            ExcessNoisePlotter, IonizationCoefficientsVsFieldPlotter,
                            IonizationRatioVsFieldPlotter,
                            MultiplicationVsExcessBiasPlotter,
                            AvalancheProbabilityMapPlotter,
                            BreakdownProbVsExcessBiasPlotter,
                            ATPPlotter)
    from .temporal import (TrajectoryPlotter, JitterPlotter,
                           JitterHistogramPlotter, PopulationPlotter,
                           AvalancheCurrentPulsePlotter,
                           DeadSpaceDistributionPlotter,
                           QuenchingWaveformPlotter)
    from .temperature import (DCRvsTempPlotter, PDPvsTempPlotter,
                              DarkCurrentComponentsVsTempPlotter,
                              BreakdownVoltageVsTempPlotter)

    _BUILTIN_PLOTTERS.update({
        "device_structure": DeviceStructurePlotter,
        "doping": DopingPlotter,
        "potential_profile": PotentialProfilePlotter,
        "electric_field": ElectricFieldPlotter,
        "breakdown_sweep": BreakdownSweepPlotter,
        "peak_field_vs_bias": PeakFieldVsBiasPlotter,
        "dark_current": DarkCurrentPlotter,
        "dark_current_components": DarkCurrentComponentsPlotter,
        "dcr": DCRPlotter,
        "iv_characteristic": IVCharacteristicPlotter,
        "comprehensive_iv": ComprehensiveIVPlotter,
        "pdp": PDPPlotter,
        "pdp_vs_vex": PDPVsExcessVoltagePlotter,
        "absorption_profile": AbsorptionProfilePlotter,
        "pdp_3d": PDP3DPlotter,
        "trigger_probability": TriggerProbabilityPlotter,
        "trigger_vs_vex": TriggerVsVexPlotter,
        "afterpulsing": AfterpulsingPlotter,
        "excess_noise": ExcessNoisePlotter,
        "ionization_vs_field": IonizationCoefficientsVsFieldPlotter,
        "ionization_ratio": IonizationRatioVsFieldPlotter,
        "multiplication_vs_vex": MultiplicationVsExcessBiasPlotter,
        "avalanche_map": AvalancheProbabilityMapPlotter,
        "breakdown_prob_vs_vex": BreakdownProbVsExcessBiasPlotter,
        "atp": ATPPlotter,
        "trajectory": TrajectoryPlotter,
        "jitter": JitterPlotter,
        "jitter_histogram": JitterHistogramPlotter,
        "population": PopulationPlotter,
        "avalanche_current_pulse": AvalancheCurrentPulsePlotter,
        "dead_space_dist": DeadSpaceDistributionPlotter,
        "quenching_waveform": QuenchingWaveformPlotter,
        "dcr_vs_temp": DCRvsTempPlotter,
        "pdp_vs_temp": PDPvsTempPlotter,
        "dark_current_vs_temp_components": DarkCurrentComponentsVsTempPlotter,
        "breakdown_voltage_vs_temp": BreakdownVoltageVsTempPlotter,
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
