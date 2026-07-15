"""Avalanche studies: afterpulsing, excess noise, jitter, breakdown prob, pulse, quenching."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..avalanche.afterpulsing import AfterpulsingModel
from ..avalanche.excess_noise import ExcessNoiseModel
from ..transport.jitter import JitterModel
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from . import _config as _cfg
from ._config import PLOT_DIR

log = get_logger()


def run_afterpulsing(sim: SPADSimulator, Vbr: float) -> dict:
    ap = AfterpulsingModel()
    holdoff_pts = np.logspace(-9, -4, 100)
    P_ap = np.array([ap.compute(t) for t in holdoff_pts])

    get_plotter("afterpulsing", plot_dir=PLOT_DIR).plot(
        holdoff_pts, P_ap)

    holdoff_1us = ap.compute(1e-6)
    log.info(f"  Afterpulsing: P_ap(1us)={holdoff_1us*100:.1f}%")
    return {"P_ap_1us": holdoff_1us}


def run_excess_noise(sim: SPADSimulator, Vbr: float) -> dict:
    en = ExcessNoiseModel()
    M_vals = np.linspace(2, 20, 20)
    F_vals = np.array([en.compute(M) for M in M_vals])

    get_plotter("excess_noise", plot_dir=PLOT_DIR).plot(
        M_vals, F_vals, k_eff=en.k)

    M_max = float(np.nanmax(M_vals))
    F_max = float(np.nanmax(F_vals))
    log.info(f"  Excess noise: M_max={M_max:.1f}  F_max={F_max:.2f}  k_eff={en.k:.3f}")
    return {"M_max": M_max, "F_max": F_max, "k_eff": en.k}


def run_jitter(sim: SPADSimulator, Vbr: float) -> dict:
    jm = JitterModel()
    sigma = jm.compute(1310e-9)
    log.info(f"  Jitter: sigma={sigma*1e12:.1f}ps")
    return {"sigma_s": sigma, "fwhm_s": sigma * 2.355}


def run_dead_space_distribution(sim: SPADSimulator, Vbr: float,
                                 N_sim: int = 20) -> None:
    log.info("  Dead space: skipped (MC not available)")


def run_breakdown_prob_vs_vex(sim: SPADSimulator, Vbr: float,
                               N_sim: int = 20) -> None:
    """Breakdown probability vs excess bias.

    Uses the McIntyre trigger probability averaged over the multiplication
    region (same physics as ``run_trigger_vs_vex`` but reported as a single
    breakdown-probability curve suitable for Figure 7).
    """
    Vex_arr = np.linspace(-10, 10, 41)
    BrP = np.full(len(Vex_arr), float("nan"))

    for j, Vex in enumerate(Vex_arr):
        Vbias = Vbr + Vex
        if Vbias <= 0:
            continue
        try:
            _, E, _, _, _, _ = sim.get_fields(Vbias)
            BrP[j] = _cfg.absorption_weighted_trigger(sim, E)
        except Exception as e:
            log.info(f"  breakdown prob Vex={Vex:.1f}V failed: {e}")
        except Exception as e:
            log.info(f"  breakdown prob Vex={Vex:.1f}V failed: {e}")

    valid = np.isfinite(BrP)
    if np.any(valid):
        log.info(f"  Breakdown prob range: {np.nanmin(BrP):.4f} – {np.nanmax(BrP):.4f}")
        get_plotter("breakdown_prob_vs_vex", plot_dir=PLOT_DIR).plot(
            Vex_arr[valid], BrP[valid], N_sim=N_sim)


def run_avalanche_current_pulse(sim: SPADSimulator, Vbr: float) -> None:
    Vex = 3.0
    try:
        loop = sim.build_self_consistent(Vbr + Vex, Rq=1e5, Cspad=1e-15, Vbr=Vbr)
        xl, xr, _ = sim.depletion_width(Vbr + Vex)
        x_inject = (xl + xr) / 2.0
        loop.inject_carrier(x_inject, "electron")
        history = loop.run(2000)

        t_arr = np.array([h["t"] for h in history])
        I_arr = np.array([h["I"] for h in history])

        get_plotter("avalanche_current_pulse", plot_dir=PLOT_DIR).plot(
            t_arr, I_arr, Vbr=Vbr, Vex=Vex)
        log.info(f"  Current pulse: I_peak={I_arr.max()*1e6:.2f} uA  "
                 f"duration={t_arr[-1]*1e12:.1f} ps")
    except Exception as e:
        log.info(f"  Avalanche current pulse failed: {e}")


def run_quenching_waveform(sim: SPADSimulator, Vbr: float) -> None:
    Vex = 3.0
    Vbias = Vbr + Vex
    try:
        loop = sim.build_self_consistent(Vbias, Rq=1e5, Cspad=1e-15, Vbr=Vbr)
        xl, xr, _ = sim.depletion_width(Vbias)
        x_inject = (xl + xr) / 2.0
        loop.inject_carrier(x_inject, "electron")
        history = loop.run(3000)

        t_arr = np.array([h["t"] for h in history])
        Vspad_arr = np.array([h["Vspad"] for h in history])
        I_arr = np.array([h["I"] for h in history])

        get_plotter("quenching_waveform", plot_dir=PLOT_DIR).plot(
            t_arr, Vspad_arr, I_arr, Vbr=Vbr, Vbias=Vbias)
        log.info(f"  Quenching: V_drop={Vbias - Vspad_arr.min():.2f}V  "
                 f"recharge_time~{sim.loop.circuit.tau*1e12:.1f}ps")
    except Exception as e:
        log.info(f"  Quenching waveform failed: {e}")
