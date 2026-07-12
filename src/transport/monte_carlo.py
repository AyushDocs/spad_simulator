"""Monte Carlo transport module."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from pydantic.dataclasses import dataclass

from ..core.constants import q, hbar, m0, kB
from ..utils._logging import get_logger

log = get_logger("transport.monte_carlo")


class MonteCarloTransport(BaseModel):
    """Monte Carlo transport simulation.

    Simulates carrier transport using a simplified Monte Carlo approach.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: NDArray
    T: float = 300.0
    n_particles: int = 1000
    dt: float = 1e-15  # time step (s)
    n_steps: int = 1000

    _q_val: float = PrivateAttr()
    _kB_val: float = PrivateAttr()

    @model_validator(mode="after")
    def _init_constants(self):
        self._q_val = float(q.to("C").magnitude)
        self._kB_val = float(kB.to("J/K").magnitude)
        return self

    def _init_particles(self, n_particles: int, x_min: float, x_max: float) -> np.ndarray:
        """Initialize particle positions uniformly."""
        return np.random.uniform(x_min, x_max, n_particles)

    def _init_velocities(self, n_particles: int) -> np.ndarray:
        """Initialize particle velocities from Maxwell-Boltzmann distribution."""
        vth = np.sqrt(self._kB_val * self.T / (0.26 * float(m0.to("kg").magnitude)))
        return np.random.normal(0, vth, n_particles)
