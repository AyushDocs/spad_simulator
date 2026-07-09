"""SPADSimulator subsystem construction."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.device import Device
from ..poisson.solver import PoissonSolver
from ..poisson.field import DepletionWidth
from ..poisson.service import PoissonService
from ..avalanche.ionization import IonizationCoefficients, OkutoCrowellModel
from ..avalanche.trigger import TriggerSolver
from ..avalanche.dark_current import DarkCurrentModel
from ..avalanche.pdp import PDPModel

if TYPE_CHECKING:
    from ..core.grid import Grid1D


def build_subsystems(
    device: Device,
    grid: Grid1D,
    materials: dict,
    T: float,
) -> dict:
    """Create all physics subsystem objects from a Device.

    Returns a dict of subsystem instances keyed by name.
    """
    poisson_solver = PoissonSolver(
        grid, T, device.doping, device.material.eps, device.material.ni,
        max_iter=200,
    )
    depletion = DepletionWidth(grid)
    poisson_service = PoissonService(poisson_solver, grid, depletion)

    ionization = IonizationCoefficients(
        OkutoCrowellModel(),
        materials,
        Eg_grid=device.material.Eg,
        Eth_grid=device.material.Eth,
        mc_grid=device.material.mc,
        mh_grid=device.material.mh,
        E_ie_grid=device.material.E_ie,
        E_ih_grid=device.material.E_ih,
        grid_x=grid.x,
        mat_names=device.material.mat_name,
        T=T,
    )
    trigger = TriggerSolver(grid)
    dark_current = DarkCurrentModel(
        T=T,
        Eg_grid=device.material.Eg,
        mc_grid=device.material.mc,
        mh_grid=device.material.mh,
        grid_x=grid.x,
        tau_n_grid=device.material.tau_n,
        tau_p_grid=device.material.tau_p,
    )
    pdp_model = PDPModel(materials)

    return {
        "poisson_service": poisson_service,
        "ionization": ionization,
        "trigger": trigger,
        "dark_current": dark_current,
        "pdp_model": pdp_model,
    }
