"""Plotter registry — name → class mapping, get_plotter, register_plotter."""
from __future__ import annotations

from typing import Any

from ._base import Plotter

_BUILTIN_PLOTTERS: dict[str, type[Plotter]] = {}


def _register_builtins() -> None:
    from .structure import DeviceStructurePlotter, DopingPlotter
    from .fields import (PotentialProfilePlotter, ElectricFieldPlotter,
                          BreakdownSweepPlotter, PeakFieldVsBiasPlotter,
                          EFieldVsAbsorptionPlotter,
                          EFieldVsMultiplicationPlotter,
                          BandDiagramPlotter)
    from .current import (DarkCurrentPlotter, DarkCurrentComponentsPlotter,
                          DCRPlotter, IVCharacteristicPlotter, ComprehensiveIVPlotter,
                          GenerationRateProfilePlotter, TrapDensityIVPlotter)
    from .pde import (PDEPlotter, PDEVsExcessVoltagePlotter,
                      AbsorptionProfilePlotter, PDEVsAreaPlotter, PDE3DPlotter)
    from .avalanche import (TriggerProbabilityPlotter, TriggerVsVexPlotter,
                            AfterpulsingPlotter,
                            ExcessNoisePlotter, IonizationCoefficientsVsFieldPlotter,
                            IonizationRatioVsFieldPlotter,
                            MultiplicationVsExcessBiasPlotter,
                            AvalancheProbabilityMapPlotter,
                            BreakdownProbVsExcessBiasPlotter,
                            ATPPlotter,
                            TriggerBackCalculatePlotter)
    from .param_sweep import (IVSweepPlotter, ParamSweepPlotter,
                               ParamSweepIVPlotter,
                               PunchBreakdownSweepPlotter,
                               DCRvsPDEPlotter,
                               DCRPDEVsVexPlotter)
    from .temporal import (TrajectoryPlotter, JitterPlotter,
                           JitterHistogramPlotter, PopulationPlotter,
                           AvalancheCurrentPulsePlotter,
                           DeadSpaceDistributionPlotter,
                           QuenchingWaveformPlotter)
    from .temperature import (DCRvsTempPlotter, PDEvsTempPlotter,
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
        "generation_rate_profile": GenerationRateProfilePlotter,
        "dcr": DCRPlotter,
        "iv_characteristic": IVCharacteristicPlotter,
        "comprehensive_iv": ComprehensiveIVPlotter,
        "pde": PDEPlotter,
        "pde_vs_vex": PDEVsExcessVoltagePlotter,
        "absorption_profile": AbsorptionProfilePlotter,
        "pde_vs_area": PDEVsAreaPlotter,
        "pde_3d": PDE3DPlotter,
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
        "trigger_back_calculate": TriggerBackCalculatePlotter,
        "trajectory": TrajectoryPlotter,
        "jitter": JitterPlotter,
        "jitter_histogram": JitterHistogramPlotter,
        "population": PopulationPlotter,
        "avalanche_current_pulse": AvalancheCurrentPulsePlotter,
        "dead_space_dist": DeadSpaceDistributionPlotter,
        "quenching_waveform": QuenchingWaveformPlotter,
        "dcr_vs_temp": DCRvsTempPlotter,
        "pde_vs_temp": PDEvsTempPlotter,
        "dark_current_vs_temp_components": DarkCurrentComponentsVsTempPlotter,
        "breakdown_voltage_vs_temp": BreakdownVoltageVsTempPlotter,
        "efield_vs_absorption": EFieldVsAbsorptionPlotter,
        "efield_vs_multiplication": EFieldVsMultiplicationPlotter,
        "band_diagram": BandDiagramPlotter,
        "iv_sweep": IVSweepPlotter,
        "param_sweep": ParamSweepPlotter,
        "param_sweep_iv": ParamSweepIVPlotter,
        "trap_density_iv": TrapDensityIVPlotter,
        "punch_breakdown_sweep": PunchBreakdownSweepPlotter,
        "dcr_vs_pde": DCRvsPDEPlotter,
        "dcr_pde_vs_vex": DCRPDEVsVexPlotter,
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
