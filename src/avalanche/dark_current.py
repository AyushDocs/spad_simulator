"""Dark current density components."""

from __future__ import annotations


import numpy as np

from ..core.constants import q
from .current import CurrentDensityComponent


class DarkCurrentModel:
    """DarkCurrentModel compatibility class for tests."""

    def __init__(self, T, grid_x, N_T, mat_name_grid, ni_absorber, tau_n_absorber, tau_p_absorber, Eg_mulp, mc_mulp, mh_mulp):
        self.T = T
        self.grid_x = grid_x
        self.N_T = N_T
        self.mat_name_grid = mat_name_grid
        self.ni_absorber = ni_absorber
        self.tau_n_absorber = tau_n_absorber
        self.tau_p_absorber = tau_p_absorber
        self.Eg_mulp = Eg_mulp
        self.mc_mulp = mc_mulp
        self.mh_mulp = mh_mulp

        from .current import CompositeCurrentDensity, BTBTCurrentDensity, SRHCurrentDensity
        self.current = CompositeCurrentDensity()
        self.current.add(SRHCurrentDensity(
            tau_n_absorber=tau_n_absorber,
            tau_p_absorber=tau_p_absorber,
            mat_name_grid=mat_name_grid,
            ni_absorber=ni_absorber,
        ))
        self.current.add(BTBTCurrentDensity(
            Eg_mulp=Eg_mulp, mc_mulp=mc_mulp, mh_mulp=mh_mulp, T=T, N_T=N_T
        ))

    def total_dark_current_density(self, x, F, ni_arr, Eg_arr, mc_arr, mh_arr):
        return self.current.compute(x, F)

    def generation_rate(self, x, F, ni_arr, Eg_arr, mc_arr, mh_arr):
        J_total = self.total_dark_current_density(x, F, ni_arr, Eg_arr, mc_arr, mh_arr)
        q_val = float(q.to("C").magnitude)
        return J_total / q_val

    def compute_dcr(self, x, F, Pe, ni_arr, Eg_arr, mc_arr, mh_arr, detector_area=1e-6):
        G = self.generation_rate(x, F, ni_arr, Eg_arr, mc_arr, mh_arr)
        return float(np.trapezoid(G * Pe, x) * detector_area)
