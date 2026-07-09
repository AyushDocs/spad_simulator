#!/usr/bin/env python3
"""InGaAs/InP SAGCM SPAD Simulator (1D center-axis)."""

from __future__ import annotations

import logging

from .utils._logging import set_log_level
from .utils.ingestion import DataIngestionConfig, DataIngestionService
from .utils.artifacts import collect_artifact, ArtifactWriter
from .studies.fields import find_breakdown, plot_device_structure, run_field_sweep, run_trigger_profiles
from .studies.dark_current import run_dark_current_sweep, run_dcr_vs_temp, collect_dark_current_metrics
from .studies.iv import run_iv_characteristic, run_comprehensive_iv
from .studies.pdp import (run_pdp_spectrum, run_pdp_vs_vex, run_pdp_vs_temp,
                           run_pde_vs_bias, collect_pdp_max_metrics)
from .studies.avalanche import run_afterpulsing, run_excess_noise, run_jitter


def main() -> None:
    set_log_level(logging.INFO)
    cfg = DataIngestionConfig.from_defaults()
    svc = DataIngestionService(cfg)
    sim = svc.build_simulator()

    plot_device_structure(sim)
    Vbr = find_breakdown(sim)

    run_field_sweep(sim, Vbr)
    run_dark_current_sweep(sim, Vbr)
    run_iv_characteristic(sim, Vbr)
    run_pdp_spectrum(sim, Vbr)
    run_pdp_vs_vex(sim, Vbr)
    run_comprehensive_iv(sim, Vbr)
    run_trigger_profiles(sim, Vbr)

    afterpulsing = run_afterpulsing(sim, Vbr)
    excess_noise = run_excess_noise(sim, Vbr)
    pde = run_pde_vs_bias(sim, Vbr)
    jitter = run_jitter(sim, Vbr)

    dark_current_metrics = collect_dark_current_metrics(sim, Vbr)
    pdp_max_metrics = collect_pdp_max_metrics(sim, cfg.target_wavelengths_nm)

    artifact = collect_artifact(Vbr, sim, afterpulsing, excess_noise,
                                pde, jitter, dark_current_metrics, pdp_max_metrics)
    writer = ArtifactWriter(cfg.output_dir)
    writer.write_xml(artifact)

    dcr_temp = run_dcr_vs_temp(svc, Vbr)
    pdp_temp = run_pdp_vs_temp(svc, Vbr)

    if dcr_temp or pdp_temp:
        artifact.dcr_vs_temp = dcr_temp
        artifact.pdp_vs_temp = pdp_temp
        writer.write_xml(artifact)


if __name__ == "__main__":
    main()
