"""Afterpulsing model for SPAD carriers trapped during avalanche."""
from __future__ import annotations

import numpy as np

from ..core.constants import q, kB


class AfterpulsingModel:
    """
    Trap-mediated afterpulsing model.

    During avalanche, carriers are captured by traps with concentration N_T.
    Trapped carriers are released with time constant tau_c:
        n_trapped(t) = N_T * (1 - exp(-t / tau_c))
        release_rate = n_trapped / tau_c

    Afterpulsing probability:
        P_ap(t) = 1 - exp(-integral_0^t release_rate(s) ds)

    For a single trap level:
        P_ap(t) = 1 - exp(-N_T * tau_c * (1 - exp(-t/tau_c)))

    Parameters
    ----------
    N_T : float
        Effective trap concentration (cm⁻³).
    tau_c : float
        Characteristic emission time constant (s).
    Vbr : float
        Breakdown voltage for normalization.
    """

    def __init__(self, N_T: float = 1e12, tau_c: float = 1e-6,
                 Vbr: float = 76.0) -> None:
        self.N_T = N_T
        self.tau_c = tau_c
        self.Vbr = Vbr

    def capture_probability(self, Vex: float, dt: float) -> float:
        """
        Probability a carrier is trapped during one time step.

        P_capture = N_T * sigma * v_th * dt
        where sigma ~ 1/N_T^(2/3) for geometric cross-section.
        """
        if self.N_T <= 0:
            return 0.0
        cross_section = self.N_T ** (-2.0 / 3.0)
        v_th = np.sqrt(8.0 * kB * 300.0 / (np.pi * 0.26 * 9.109e-31))
        return float(np.clip(self.N_T * cross_section * v_th * dt, 0.0, 1.0))

    def release_rate(self, t_since_capture: float) -> float:
        """
        Trap release rate at time t after capture.

        Rate = (1/tau_c) * exp(-t/tau_c)
        """
        return (1.0 / self.tau_c) * np.exp(-t_since_capture / self.tau_c)

    def afterpulsing_probability(self, holdoff: float) -> float:
        """
        Afterpulsing probability given a holdoff time after detection.

        P_ap = 1 - exp(-N_T * tau_c * (1 - exp(-holdoff/tau_c)))
        """
        return float(1.0 - np.exp(
            -self.N_T * self.tau_c * (1.0 - np.exp(-holdoff / self.tau_c))
        ))

    def effective_dcr(self, raw_dcr: float, holdoff: float) -> float:
        """
        DCR corrected for afterpulsing.

        DCR_eff = DCR_raw * (1 + P_ap)
        """
        return raw_dcr * (1.0 + self.afterpulsing_probability(holdoff))

    def holdoff_optimal(self, target_ap: float = 0.01) -> float:
        """
        Minimum holdoff time to achieve target afterpulsing probability.

        Solves: target = 1 - exp(-N_T * tau_c * (1 - exp(-t/tau_c)))
        """
        if self.N_T * self.tau_c < 1e-30:
            return 0.0
        arg = 1.0 + np.log(1.0 - target_ap) / (self.N_T * self.tau_c)
        if arg <= 0:
            return self.tau_c * 10.0
        return float(-self.tau_c * np.log(max(arg, 1e-30)))
