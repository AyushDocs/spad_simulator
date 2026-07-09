"""Smoke tests for simulator facade."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.utils.artifacts import SimulationArtifact, ArtifactWriter, collect_artifact
from src.avalanche.afterpulsing import AfterpulsingModel
from src.avalanche.excess_noise import ExcessNoiseFactor
from src.simulator import SPADSimulator


@pytest.fixture
def sim():
    cfg = DataIngestionConfig.from_defaults()
    svc = DataIngestionService(cfg)
    device = svc.build_device()
    return SPADSimulator(device)


def test_simulator_init(sim):
    assert sim.grid.no_of_nodes == 500
    assert sim.T == 300.0
    assert sim.detector_area > 0


def test_poisson_solve(sim):
    phi, E, info = sim.solve_poisson(0.0)
    assert len(phi) == sim.grid.no_of_nodes
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(E))


def test_depletion_width(sim):
    xl, xr, Vdep = sim.depletion_width(20.0)
    assert xl < xr
    assert Vdep > 0


def test_get_fields(sim):
    phi, E, Pe, Ph, xl, xr = sim.get_fields(20.0)
    assert len(phi) == sim.grid.no_of_nodes
    assert np.all(np.isfinite(Pe))
    assert np.all(np.isfinite(Ph))


def test_find_breakdown(sim):
    Vbr, info = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    assert Vbr is not None
    assert 50 < Vbr < 100


def test_dark_current(sim):
    Vbr, _ = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    dc = sim.compute_dark_current(Vbr + 1.0)
    assert dc["I_dark"] > 0
    assert dc["DCR"] > 0
    assert np.all(np.isfinite(dc["Pe"]))


def test_json_output(sim):
    Vbr, _ = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
    ap = AfterpulsingModel(N_T=1e12, tau_c=1e-6, Vbr=Vbr)
    en = ExcessNoiseFactor(k_eff=0.5)
    ap_metrics = {"N_T": ap.N_T, "tau_c": ap.tau_c,
                  "P_ap_1us": ap.afterpulsing_probability(1e-6),
                  "holdoff_optimal_1pct_s": ap.holdoff_optimal(0.01)}
    en_metrics = {"M_max": 10.0, "F_max": en.f(10.0), "k_eff": 0.5}
    pde_metrics = {"pde_max": 0.68, "wavelength_nm": 1310}
    jitter_metrics = {"sigma_s": 1e-12, "fwhm_s": 2e-12}
    dc_metrics = {"I_dark_A": 3e-8, "DCR_cps": 1e9, "Vex_V": 3.0}
    pdp_metrics = {"905nm": 0.5, "1310nm": 0.74, "1550nm": 0.65}

    artifact = collect_artifact(Vbr, sim, ap_metrics, en_metrics,
                                 pde_metrics, jitter_metrics,
                                 dc_metrics, pdp_metrics)

    assert artifact.Vbr_V == Vbr
    assert artifact.T_K == 300.0
    assert artifact.I_dark_A == 3e-8
    assert artifact.pdp_max["1310nm"] == 0.74
    assert artifact.ap_N_T == 1e12
    assert artifact.en_k_eff == 0.5
    assert artifact.jitter_fwhm_s == 2e-12

    d = artifact.to_dict()
    assert d["device"]["Vbr_V"] == Vbr
    assert d["dark_current"]["I_dark_A"] == 3e-8
    assert d["pdp_max"]["1310nm"] == 0.74
    assert d["afterpulsing"]["N_T"] == 1e12

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ArtifactWriter(tmpdir)
        path = writer.write_xml(artifact)
        assert os.path.exists(path)
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "spad_simulation"
        assert root.find("device") is not None
        assert root.find("dark_current") is not None
        assert root.find("pdp_max") is not None
        assert root.find("afterpulsing") is not None
        assert root.find("excess_noise") is not None
        assert root.find("photon_detection_efficiency") is not None
        assert root.find("timing_jitter") is not None


def test_data_ingestion_service():
    cfg = DataIngestionConfig.from_defaults()
    assert os.path.exists(cfg.device_xml)
    assert os.path.exists(cfg.materials_xml)
    assert os.path.exists(cfg.absorption_xml)
    svc = DataIngestionService(cfg)
    dev = svc.build_device()
    assert dev.T == 300.0
    assert len(dev.layers) > 0
