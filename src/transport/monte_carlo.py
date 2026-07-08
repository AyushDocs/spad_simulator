from __future__ import annotations

from typing import List

import numpy as np

from ..core.grid import Grid1D
from ..avalanche.ionization import IonizationCoefficients
from .carrier import Carrier
from .drift_diffusion import DriftDiffusion
from ..utils._logging import get_logger

log = get_logger("mc")


class MonteCarloSimulator:
    """
    Device-level Monte Carlo.

    For each history:
        1. Inject a single carrier
        2. Advection-diffusion step
        3. Sample ionisation: P_ion = 1 - exp(-alpha dx)
        4. Spawn secondary e-h pair on ionisation
        5. Stop when population >= N_threshold or all carriers exit
    """

    def __init__(self, transport: DriftDiffusion,
                 ionization: IonizationCoefficients,
                 grid: Grid1D) -> None:
        self.transport = transport
        self.ionization = ionization
        self.grid = grid
        self.x_left = float(grid.x[0])
        self.x_right = float(grid.x[-1])

    def run_single(self, x0: float, E_grid: np.ndarray,
                   typ: str = "electron",
                   N_threshold: int = 100, dt: float = 1e-15,
                   max_steps: int = 100000) -> dict:
        E0 = float(np.interp(x0, self.grid.x, E_grid))
        l_dead = self.ionization.dead_space_at(x0, E0)
        queue = [Carrier(x0, typ, dead_space=l_dead)]
        t = 0.0
        t_detect = None

        for _ in range(max_steps):
            new_queue: List[Carrier] = []
            for c in queue:
                if not c.alive:
                    continue
                E_loc = float(np.interp(c.x, self.grid.x, E_grid))
                self.transport.step(c, E_loc, dt,
                                    self.x_left, self.x_right)
                if not c.alive:
                    continue
                if c.dead_space_remaining <= 0:
                    dx = abs(self.transport.drift_velocity(E_loc, c.typ)) * dt
                    coeff = (self.ionization.alpha if c.is_electron
                             else self.ionization.beta)
                    P_ion = 1.0 - np.exp(-float(coeff(np.array([E_loc]))[0]) * dx)
                    if np.random.rand() < P_ion:
                        l_d = self.ionization.dead_space_at(c.x, E_loc)
                        new_queue.append(Carrier(c.x, "electron", dead_space=l_d))
                        new_queue.append(Carrier(c.x, "hole", dead_space=l_d))
                        c.reset_dead_space(l_d)
                new_queue.append(c)
            queue = new_queue
            t += dt
            N = sum(1 for c in queue if c.alive)
            if N >= N_threshold:
                t_detect = t
                break

        return {"avalanche": t_detect is not None, "N_max": N,
                "t_detect": t_detect, "t_total": t}

    def run_ensemble(self, N_sim: int, x0: float, E_grid: np.ndarray,
                     typ: str = "electron",
                     N_threshold: int = 100, dt: float = 1e-15,
                     verbose: bool = True) -> dict:
        results: List[dict] = []
        n_av = 0
        for i in range(N_sim):
            res = self.run_single(x0, E_grid, typ, N_threshold, dt)
            results.append(res)
            if res["avalanche"]:
                n_av += 1

        t_detect = np.array([r["t_detect"] for r in results
                             if r["avalanche"] and r["t_detect"] is not None])

        if verbose:
            log.info("MC ensemble  N=%d  BrP=%d/%d=%.3f  t_mean=%.2f ps",
                     N_sim, n_av, N_sim, n_av / max(N_sim, 1),
                     np.mean(t_detect) * 1e12 if len(t_detect) else 0)

        return {"BrP": n_av / N_sim,
                "t_detect": t_detect,
                "n_avalanches": n_av, "n_sim": N_sim,
                "results": results}
