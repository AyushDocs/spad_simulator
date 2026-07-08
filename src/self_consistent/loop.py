from __future__ import annotations

from typing import List

import numpy as np

from ..core.grid import Grid1D
from ..core.doping import DopingProfile
from ..core.constants import q
from ..poisson.solver import PoissonSolver
from ..avalanche.ionization import IonizationCoefficients
from ..transport.carrier import Carrier
from ..transport.drift_diffusion import DriftDiffusion
from .particle_mesh import ParticleMesh
from .circuit import CircuitSolver
from ..utils._logging import get_logger

log = get_logger("pic")


class SelfConsistentLoop:
    """
    Self-consistent Monte Carlo-Poisson loop.

    At each time step:
        1. Move carriers (drift-diffusion)
        2. Sample ionisation -> spawn secondaries
        3. Deposit charge onto grid -> rho_ext
        4. Solve Poisson with rho_ext
        5. Update external circuit
    """

    def __init__(self, grid: Grid1D,
                 doping: DopingProfile,
                 poisson_solver: PoissonSolver,
                 particle_mesh: ParticleMesh,
                 transport: DriftDiffusion,
                 ionization: IonizationCoefficients,
                 circuit: CircuitSolver,
                 dt: float = 1e-15) -> None:
        self.grid = grid
        self.doping = doping
        self.poisson = poisson_solver
        self.mesh = particle_mesh
        self.transport = transport
        self.ionization = ionization
        self.circuit = circuit
        self.dt = dt

        self.carriers: List[Carrier] = []
        self.t = 0.0
        self.history: List[dict] = []
        self._phi_grid = np.linspace(0, circuit.Vbias, grid.no_of_nodes)
        self._E_grid = self.grid.gradient(self._phi_grid)

    def inject_carrier(self, x0: float, typ: str = "electron") -> None:
        E0 = float(np.interp(x0, self.grid.x, self._E_grid))
        l_dead = float(self.ionization.dead_space_length(E0, typ))
        self.carriers.append(Carrier(x0, typ, dead_space=l_dead))

    def _current(self) -> float:
        L = self.grid.L
        return float(sum(q * c.v / L for c in self.carriers if c.alive))

    def step(self) -> dict:
        new_carriers: List[Carrier] = []
        for c in self.carriers:
            if not c.alive:
                continue
            E_loc = float(np.interp(c.x, self.grid.x, self._E_grid))
            self.transport.step(c, E_loc, self.dt,
                                self.grid.x[0], self.grid.x[-1])
            if not c.alive:
                continue
            if c.dead_space_remaining <= 0:
                E_loc = float(np.interp(c.x, self.grid.x, self._E_grid))
                dx = abs(self.transport.drift_velocity(E_loc, c.typ)) * self.dt
                coeff = (self.ionization.alpha if c.is_electron
                         else self.ionization.beta)
                P = 1.0 - np.exp(-float(coeff(np.array([E_loc]))[0]) * dx)
                if np.random.rand() < P:
                    ld = float(self.ionization.dead_space_length(E_loc, c.typ))
                    new_carriers.extend([Carrier(c.x, "electron", dead_space=ld),
                                         Carrier(c.x, "hole", dead_space=ld)])
                    c.reset_dead_space(ld)
            new_carriers.append(c)
        self.carriers.extend(new_carriers)

        rho_ext = self.mesh.deposit_charge(self.carriers)
        phi, _ = self.poisson.solve(self.circuit.Vspad,
                                    phi_n=0.0, phi_p=0.0,
                                    guess=self._phi_grid,
                                    rho_ext=rho_ext)
        self._phi_grid = phi
        self._E_grid = self.grid.gradient(phi)

        I_av = self._current()
        self.circuit.update(I_av, self.dt)
        self.t += self.dt

        info = {"t": self.t,
                "N": sum(1 for c in self.carriers if c.alive),
                "Vspad": self.circuit.Vspad, "I": I_av}
        self.history.append(info)
        return info

    def run(self, N_steps: int) -> List[dict]:
        for _ in range(N_steps):
            self.step()
        return self.history
