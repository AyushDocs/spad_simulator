#!/usr/bin/env python3
"""InGaAs/InP SAGCM SPAD Simulator (1D center-axis)."""

from __future__ import annotations

import logging
import sys

from .utils._logging import set_log_level, get_logger
from .utils.ingestion import DataIngestionConfig, DataIngestionService
from .utils.artifacts import collect_artifact, ArtifactWriter
from .utils.loaders import PlotConfig
from .studies.fields import (find_breakdown, plot_device_structure,
                              run_field_sweep, run_trigger_profiles,
                              run_trigger_vs_vex, run_depletion_vs_bias,
                              run_peak_field_vs_bias, run_avalanche_probability_map,
                              run_breakdown_vs_temp, run_band_diagram)
from .studies.dark_current import (run_dark_current_sweep,
                                   run_dark_current_component_sweep,
                                   run_dcr_vs_temp,
                                   run_dcr_pde_vs_vex,
                                   collect_dark_current_metrics,
                                   run_dark_current_components_vs_temp)
from .studies.iv import run_iv_characteristic, run_comprehensive_iv
from .studies.pde import (run_pde_spectrum, run_pde_vs_vex, run_pde_vs_temp,
                          run_pde_vs_area,
                          collect_pde_max_metrics,
                          run_absorption_profile, run_pde_3d)
from .studies.avalanche import (run_afterpulsing, run_excess_noise, run_jitter,
                                run_breakdown_prob_vs_vex,
                                run_dead_space_distribution,
                                run_avalanche_current_pulse,
                                run_quenching_waveform)
from .studies.ionization import run_ionization_vs_field, run_multiplication_vs_vex
from .studies.trigger_back_calculate import run_trigger_back_calculate
from .studies.trap_density import run_trap_density_iv
from .studies.punch_breakdown import run_punch_breakdown_sweep
from .studies.field_sweep import (run_efield_vs_absorption_width,
                                 run_efield_vs_multiplication_width)
from .studies.param_sweep import (run_iv_sweep,
                                    sweep_absorption_thickness,
                                    sweep_multiplication_thickness,
                                    sweep_charge_density,
                                    sweep_p_layer_doping)
from .studies.dcr_vs_pde import run_dcr_vs_pde
from .studies.optimization import run_optimize_device

log = get_logger()


def main() -> None:
    iv_only = "--iv-only" in sys.argv
    set_log_level(logging.INFO)
    log.info("=" * 60)
    log.info("SPAD Simulator — starting")
    log.info("=" * 60)
    cfg = DataIngestionConfig.from_defaults()
    svc = DataIngestionService(cfg)
    plot_cfg = svc.load_plot_config()
    sim = svc.build_simulator()
    log.info("Device built  L=%.1f um  N=%d", sim.device.L * 1e4, sim.grid.no_of_nodes)

    plot_device_structure(sim, plot_cfg=plot_cfg)
    log.info("─" * 50)
    log.info("Finding breakdown voltage …")
    Vbr = find_breakdown(sim)
    log.info("Breakdown voltage: Vbr = %.2f V", Vbr)

    if iv_only:
        log.info("IV-only mode")
        run_iv_characteristic(sim, Vbr, plot_cfg=plot_cfg)
        run_comprehensive_iv(sim, Vbr, plot_cfg=plot_cfg)
        log.info("Done.")
        return

    # -- Block 1: Fields, band diagram, depletion ---------------------------------
    log.info("─" * 50)
    log.info("Block 1/6 — Fields, band diagram, depletion")
    run_field_sweep(sim, Vbr, plot_cfg=plot_cfg)
    run_band_diagram(sim, Vbr, plot_cfg=plot_cfg)
    run_depletion_vs_bias(sim, Vbr, plot_cfg=plot_cfg)

    # -- Block 2: Dark current, IV -------------------------------------------------
    log.info("Block 2/6 — Dark current, IV characteristic")
    run_dark_current_sweep(sim, Vbr, plot_cfg=plot_cfg)
    run_dark_current_component_sweep(sim, Vbr, plot_cfg=plot_cfg)
    run_dcr_pde_vs_vex(sim, Vbr, plot_cfg=plot_cfg)
    run_iv_characteristic(sim, Vbr, plot_cfg=plot_cfg)
    run_trap_density_iv(sim, Vbr, plot_cfg=plot_cfg)
    run_punch_breakdown_sweep(sim, Vbr, plot_cfg=plot_cfg)

    # -- Block 3: PDE, trigger -----------------------------------------------------
    log.info("Block 3/6 — PDE spectrum, trigger probability")
    run_pde_spectrum(sim, Vbr, plot_cfg=plot_cfg)
    run_comprehensive_iv(sim, Vbr, plot_cfg=plot_cfg)
    run_trigger_profiles(sim, Vbr, plot_cfg=plot_cfg)
    run_trigger_vs_vex(sim, Vbr, plot_cfg=plot_cfg)
    run_trigger_back_calculate(sim, Vbr, plot_cfg=plot_cfg)

    # -- Block 4: Avalanche dynamics, metrics, artifact ----------------------------
    log.info("Block 4/6 — Avalanche dynamics, metrics, artifact")
    afterpulsing = run_afterpulsing(sim, Vbr, plot_cfg=plot_cfg)
    excess_noise = run_excess_noise(sim, Vbr, plot_cfg=plot_cfg)
    jitter = run_jitter(sim, Vbr)

    dark_current_metrics = collect_dark_current_metrics(sim, Vbr)
    pde_max_metrics = collect_pde_max_metrics(sim, cfg.target_wavelengths_nm)

    artifact = collect_artifact(Vbr, sim, afterpulsing, excess_noise,
                                jitter, dark_current_metrics, pde_max_metrics)
    writer = ArtifactWriter(cfg.output_dir)
    writer.write_xml(artifact)

    # -- Block 5: Temperature sweeps -----------------------------------------------
    log.info("Block 5/6 — Temperature sweeps")
    dcr_temp = run_dcr_vs_temp(svc, Vbr, plot_cfg=plot_cfg)
    pde_temp = run_pde_vs_temp(svc, Vbr, plot_cfg=plot_cfg)

    if dcr_temp or pde_temp:
        artifact.dcr_vs_temp = dcr_temp
        artifact.pde_vs_temp = pde_temp
        writer.write_xml(artifact)

    run_ionization_vs_field(sim, Vbr, plot_cfg=plot_cfg)
    run_multiplication_vs_vex(sim, Vbr, plot_cfg=plot_cfg)
    run_absorption_profile(sim, Vbr, plot_cfg=plot_cfg)
    run_pde_vs_area(sim, Vbr, plot_cfg=plot_cfg)
    run_pde_3d(sim, Vbr)
    run_peak_field_vs_bias(sim, Vbr, plot_cfg=plot_cfg)
    run_efield_vs_absorption_width(sim, Vbr, plot_cfg=plot_cfg)
    run_efield_vs_multiplication_width(sim, Vbr, plot_cfg=plot_cfg)
    run_avalanche_probability_map(sim, Vbr, plot_cfg=plot_cfg)
    run_breakdown_prob_vs_vex(sim, Vbr, plot_cfg=plot_cfg)
    run_dead_space_distribution(sim, Vbr)
    run_avalanche_current_pulse(sim, Vbr, plot_cfg=plot_cfg)
    run_quenching_waveform(sim, Vbr, plot_cfg=plot_cfg)

    # -- Block 6: Parameter sweeps, optimization -----------------------------------
    log.info("Block 6/6 — Parameter sweeps, optimization")
    try:
        _iv_dec = run_iv_sweep(sim, Vbr, decompose=True, plot=True,
                               plot_cfg=plot_cfg)
        if _iv_dec:
            log.info("I-V sweep with decomposition: %d points", len(_iv_dec["V"]))
    except Exception as e:
        log.warning("I-V sweep+decomposition failed: %s", e)

    for _sweep_fn, _name in [
        (sweep_absorption_thickness, "absorption thickness"),
        (sweep_multiplication_thickness, "multiplication thickness"),
        (sweep_charge_density, "charge density"),
        (sweep_p_layer_doping, "p-layer doping"),
    ]:
        try:
            log.info("  Sweep: %s", _name)
            _res = _sweep_fn(sim, Vbr, plot_cfg=plot_cfg)
            if len(_res.get("Vbr", [])) > 0:
                log.info("  Sweep %s done (%d pts)", _name, len(_res["Vbr"]))
        except Exception as e:
            log.warning("  Sweep %s failed: %s", _name, e)

    bv_temp = run_breakdown_vs_temp(svc, Vbr, plot_cfg=plot_cfg)
    dc_comp_temp = run_dark_current_components_vs_temp(svc, Vbr, plot_cfg=plot_cfg)

    if bv_temp or dc_comp_temp:
        writer.write_xml(artifact)

    original_layers_before_opt = list(sim.device.layers)
    try:
        log.info("  Device optimization …")
        opt = run_optimize_device(sim, Vbr, BV_target=Vbr, plot_cfg=plot_cfg)
        if opt:
            log.info("  Optimization: best J = %.4e", opt["best_Vbr"])
    except Exception as e:
        log.warning("  Optimization failed: %s", e)

    try:
        sim.set_layers(original_layers_before_opt)
        Vbr_restored, _ = sim.find_breakdown(force=True)
        run_dcr_vs_pde(sim, Vbr_restored, plot_cfg=plot_cfg)
    except Exception as e:
        log.warning("DCR vs PDE study failed: %s", e)

    log.info("=" * 60)
    log.info("All studies complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
