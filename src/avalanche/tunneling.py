from __future__ import annotations

import numpy as np

from ..core.constants import q, hbar, m0


class TunnelingModel:
    """
    Band-to-band (BTBT) and trap-assisted (TAT) tunneling currents.

    BTBT (Kane model):
        J_BTBT = A * F^2 * exp(-B * Eg^{3/2} / F)

    TAT (Hurkx model):
        J_TAT = A * F^2 * N_T * exp(-(B1*E_B1^{3/2} + B2*E_B2^{3/2}) / F) /
                (N_v * exp(-B1*E_B1^{3/2}/F) + N_c * exp(-B2*E_B2^{3/2}/F))
    """

    def __init__(self, T: float = 300.0,
                 N_T: float = 1e12,
                 a_frac: float = 0.5,
                 Eg_grid: np.ndarray | None = None,
                 Nc_grid: np.ndarray | None = None,
                 Nv_grid: np.ndarray | None = None) -> None:
        self._T = T
        self._N_T = N_T
        self._a = a_frac
        self._Eg_grid = Eg_grid
        self._Nc_grid = Nc_grid
        self._Nv_grid = Nv_grid
        self._q = q
        self._hbar = hbar

    def _m_r(self, mc: float, mh: float) -> float:
        return mc * m0 * mh * m0 / ((mc + mh) * m0)

    def _A_coeff(self, Eg: float, mr: float) -> float:
        return (self._q ** 3 / (np.sqrt(2.0 * mr * Eg * self._q)
                                * 4.0 * np.pi ** 3 * self._hbar ** 2))

    def _B_coeff(self, mr: float) -> float:
        me = m0
        return float(np.pi / (2.0 * self._q * self._hbar) * np.sqrt(mr * me / 2.0))

    def btbt_current(self, F: np.ndarray,
                     Eg_arr: np.ndarray,
                     mc_arr: np.ndarray,
                     mh_arr: np.ndarray) -> np.ndarray:
        F_abs = np.abs(F)
        mr = self._m_r(np.mean(mc_arr), np.mean(mh_arr))
        Eg_mean = np.mean(Eg_arr)
        A = self._A_coeff(Eg_mean, mr)
        B = self._B_coeff(mr)
        with np.errstate(divide="ignore", over="ignore"):
            return np.where(F_abs > 1e3,
                            A * F_abs ** 2 * np.exp(-B * Eg_mean ** 1.5 / F_abs),
                            0.0)

    def tat_current(self, F: np.ndarray,
                    Eg_arr: np.ndarray,
                    mc_arr: np.ndarray,
                    mh_arr: np.ndarray,
                    N_T: float | None = None,
                    a_frac: float | None = None) -> np.ndarray:
        F_abs = np.abs(F)
        N_T = N_T if N_T is not None else self._N_T
        a = a_frac if a_frac is not None else self._a
        Eg_mean = np.mean(Eg_arr)

        mc_mean = np.mean(mc_arr) * m0
        mh_mean = np.mean(mh_arr) * m0

        B1 = np.pi / (2.0 * self._q * self._hbar) * np.sqrt(mh_mean / 2.0)
        B2 = np.pi / (2.0 * self._q * self._hbar) * np.sqrt(mc_mean / 2.0)

        E_B1 = a * Eg_mean
        E_B2 = (1.0 - a) * Eg_mean

        mr = self._m_r(np.mean(mc_arr), np.mean(mh_arr))
        A = self._A_coeff(Eg_mean, mr)

        if self._Nc_grid is not None:
            Nc_mean = float(np.mean(self._Nc_grid))
        else:
            Nc_mean = 1e19
        if self._Nv_grid is not None:
            Nv_mean = float(np.mean(self._Nv_grid))
        else:
            Nv_mean = 1e19

        Nc = Nc_mean * 1e-6
        Nv = Nv_mean * 1e-6

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            num = (A * F_abs ** 2 * N_T
                   * np.exp(-(B1 * E_B1 ** 1.5 + B2 * E_B2 ** 1.5) / F_abs))
            den = (Nv * np.exp(-B1 * E_B1 ** 1.5 / F_abs)
                   + Nc * np.exp(-B2 * E_B2 ** 1.5 / F_abs))
        safe = (F_abs > 1e3) & (den > 1e-300) & (np.isfinite(num)) & (np.isfinite(den))
        result = np.zeros_like(num)
        result[safe] = num[safe] / den[safe]
        return result
