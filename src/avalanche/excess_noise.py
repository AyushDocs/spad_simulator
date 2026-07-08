"""Excess noise factor for avalanche multiplication (McIntyre 1966)."""
from __future__ import annotations

import numpy as np


class ExcessNoiseFactor:
    """
    McIntyre excess noise factor F(M) for APD/SPAD multiplication.

    For electron-initiated multiplication:
        F(M) = k * M + (1-k) * (2 - 1/M)

    where k = beta/alpha (ionization rate ratio).

    For arbitrary injection:
        F(M) = k_eff * M + (1 - k_eff) * (2 - 1/M)

    References
    ----------
    McIntyre, R.J. (1966). "Multiplication noise in uniform avalanche
    diodes." IEEE Trans. Electron Devices, 13(1), 164-168.
    """

    def __init__(self, k_eff: float = 0.5) -> None:
        self.k_eff = k_eff

    def f(self, M: float | np.ndarray) -> float | np.ndarray:
        """
        Excess noise factor as function of multiplication M.

        Parameters
        ----------
        M : float or array
            Multiplication factor (M >= 1).

        Returns
        -------
        float or array
            F(M) >= 1.
        """
        M = np.asarray(M, dtype=float)
        k = self.k_eff
        F = k * M + (1.0 - k) * (2.0 - 1.0 / np.clip(M, 1.0, None))
        return float(F) if F.ndim == 0 else F

    @classmethod
    def from_ionization(cls, alpha: float, beta: float) -> ExcessNoiseFactor:
        """Create from ionization coefficients (k = beta/alpha)."""
        k = beta / alpha if alpha > 1e-20 else 1.0
        return cls(k_eff=k)

    def excess_noise(self, M: float, alpha: float, beta: float) -> float:
        """Compute F(M) for given M and ionization coefficients."""
        k = beta / alpha if alpha > 1e-20 else 1.0
        return float(k * M + (1.0 - k) * (2.0 - 1.0 / max(M, 1.0)))

    def bandwidth_gain_product(self, M: float, alpha: float, beta: float) -> float:
        """
        Gain-bandwidth product approximation.

        GBP ≈ 1 / (2 * pi * tau_tr * M * (1 - k) * (1 + 1/k))
        where tau_tr is transit time through multiplication region.
        """
        if M <= 1 or alpha <= 0:
            return 0.0
        k = beta / alpha
        if k <= 0 or k >= 1:
            return 0.0
        return 1.0 / (2.0 * np.pi * M * (1.0 - k) * (1.0 + 1.0 / k))
