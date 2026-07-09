"""Avalanche studies: afterpulsing, excess noise, timing jitter."""
from __future__ import annotations

import numpy as np

from ..simulator import SPADSimulator
from ..avalanche.afterpulsing import AfterpulsingModel
from ..avalanche.excess_noise import ExcessNoiseFactor
from ..transport.jitter import TimingJitter
from ..utils._logging import get_logger
from ..utils.plotter import get_plotter
from ._config import PLOT_DIR

log = get_logger()


def run_afterpulsing(sim: SPADSimulator, Vbr: float) -> dict:
    ap = AfterpulsingModel(N_T=1e12, tau_c=1e-6, Vbr=Vbr)
    holdoff_pts = np.logspace(-9, -4, 100)
    P_ap = np.array([ap.afterpulsing_probability(t) for t in holdoff_pts])

    get_plotter("afterpulsing", plot_dir=PLOT_DIR).plot(
        holdoff_pts, P_ap, N_T=ap.N_T, tau_c=ap.tau_c)

    holdoff_1us = ap.afterpulsing_probability(1e-6)
    holdoff_opt = ap.holdoff_optimal(0.01)
    log.info(f"  Afterpulsing: P_ap(1µs)={holdoff_1us*100:.1f}%  "
             f"holdoff_1%={holdoff_opt*1e6:.1f}µs")
    return {"N_T": ap.N_T, "tau_c": ap.tau_c,
            "P_ap_1us": holdoff_1us, "holdoff_optimal_1pct_s": holdoff_opt}


def run_excess_noise(sim: SPADSimulator, Vbr: float) -> dict:
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
        get_plotter("excess_noise", plot_dir=PLOT_DIR).plot(
            M_arr[mask], F_arr[mask], k_eff=k_eff)

    M_max = float(np.nanmax(M_arr[mask])) if np.any(mask) else 0.0
    F_max = float(np.nanmax(F_arr[mask])) if np.any(mask) else 0.0
    log.info(f"  Excess noise: M_max={M_max:.1f}  F_max={F_max:.2f}  k_eff={k_eff:.3f}")
    return {"M_max": M_max, "F_max": F_max, "k_eff": k_eff}


def run_jitter(sim: SPADSimulator, Vbr: float) -> dict:
    try:
        ens = sim.run_mc_ensemble(Vbr + 3.0, N_sim=20, N_threshold=30, dt=5e-15)
        t_detect = TimingJitter.extract_detection_times(ens)

        if len(t_detect) == 0:
            log.info("  Jitter: no successful avalanches")
            return {"sigma_s": np.nan, "fwhm_s": np.nan}

        stats = TimingJitter.statistics(t_detect)
        fwhm_val = TimingJitter.fwhm(t_detect)

        get_plotter("jitter_histogram", plot_dir=PLOT_DIR).plot(
            t_detect, bins=30, fwhm=fwhm_val, sigma=stats["std"])

        log.info(f"  Jitter: σ={stats['std']*1e12:.1f}ps  "
                 f"FWHM={fwhm_val*1e12:.1f}ps  N={stats['N']}")
        return {"sigma_s": stats["std"], "fwhm_s": fwhm_val,
                "mean_s": stats["mean"], "N": stats["N"]}
    except Exception as e:
        log.info(f"  Jitter simulation failed: {e}")
        return {"sigma_s": np.nan, "fwhm_s": np.nan}
