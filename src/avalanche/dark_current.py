from __future__ import annotations

import numpy as np

from ..core.constants import q
from .tunneling import TunnelingModel
from .current import SRHCurrent, BTBTCurrent, TATCurrent, CompositeCurrent


class DarkCurrentModel:
    """
    SPAD dark current model combining thermal generation, BTBT, and TAT
    via the CurrentComponent interface.

    Thermal generation (mid-gap traps, depletion approximation):
        G_thermal = n_i / (2*tau)
        J_thermal = q * G_thermal
        where tau = (tau_n + tau_p)/2 assumed equal for mid-gap

    BTBT and TAT from TunnelingModel wrapped as components.

    DCR = A_det * int (G_thermal + G_BTBT/q + G_TAT/q) * P_trigger(x) dx
    """

    def __init__(self, T: float = 300.0,
                 Eg_grid: np.ndarray | None = None,
                 mc_grid: np.ndarray | None = None,
                 mh_grid: np.ndarray | None = None,
                 grid_x: np.ndarray | None = None,
                 N_T: float = 1e12,
                 tau_n_grid: np.ndarray | None = None,
                 tau_p_grid: np.ndarray | None = None) -> None:
        self.tunneling = TunnelingModel(T=T, N_T=N_T,
                                        Eg_grid=Eg_grid)
        self._Eg_grid = Eg_grid
        self._mc_grid = mc_grid
        self._mh_grid = mh_grid
        self._x = grid_x
        self._N_T = N_T
        self._tau_n = tau_n_grid
        self._tau_p = tau_p_grid
        self._q = q

        self._current = CompositeCurrent()
        self._current.add(SRHCurrent(tau_n_grid, tau_p_grid))
        self._current.add(BTBTCurrent(T=T, N_T=N_T))
        self._current.add(TATCurrent(T=T, N_T=N_T))

    @property
    def current_model(self) -> CompositeCurrent:
        return self._current

    def thermal_generation(self, x: np.ndarray,
                            ni_arr: np.ndarray) -> np.ndarray:
        if self._tau_n is not None and self._tau_p is not None:
            tau = (self._tau_n + self._tau_p) / 2.0
        elif self._x is not None and ni_arr is not None:
            tau = np.full_like(ni_arr, 1e-6)
        else:
            tau = 1e-6
        return ni_arr / (2.0 * tau)

    def thermal_current_density(self, x: np.ndarray,
                                ni_arr: np.ndarray) -> float:
        G = self.thermal_generation(x, ni_arr)
        if self._x is not None:
            return float(self._q * np.trapezoid(G, self._x))
        return float(self._q * np.trapezoid(G, x))

    def total_dark_current_density(self, x: np.ndarray,
                                   F: np.ndarray,
                                   ni_arr: np.ndarray,
                                   Eg_arr: np.ndarray,
                                   mc_arr: np.ndarray,
                                   mh_arr: np.ndarray) -> np.ndarray:
        return self._current.compute(
            x, F, ni_arr=ni_arr, Eg_arr=Eg_arr,
            mc_arr=mc_arr, mh_arr=mh_arr)

    def generation_rate(self, x: np.ndarray,
                        F: np.ndarray,
                        ni_arr: np.ndarray,
                        Eg_arr: np.ndarray,
                        mc_arr: np.ndarray,
                        mh_arr: np.ndarray) -> np.ndarray:
        J = self.total_dark_current_density(x, F, ni_arr, Eg_arr, mc_arr, mh_arr)
        return J / self._q

    def compute_dcr(self, x: np.ndarray,
                    F: np.ndarray,
                    Pe: np.ndarray,
                    ni_arr: np.ndarray,
                    Eg_arr: np.ndarray,
                    mc_arr: np.ndarray,
                    mh_arr: np.ndarray,
                    A_det: float = 1e-6) -> float:
        G = self.generation_rate(x, F, ni_arr, Eg_arr, mc_arr, mh_arr)
        integrand = G * Pe
        total_gen = float(np.trapezoid(integrand, x))
        return A_det * total_gen


