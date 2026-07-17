"""SPADSimulator subsystem construction."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.device import Device
from ..poisson.solver import PoissonSolver
from ..poisson.field import DepletionWidth
from ..poisson.service import PoissonService
from ..avalanche.ionization import OkutoCrowellCoefficients

from ..avalanche.current import (
    CurrentDecompositionManager,
    BTBTCurrentDensity,
    TATCurrentDensity,
    SRHCurrentDensity,
)
from ..avalanche.pde import PDEModel
from ..transport.continuum import ContinuumSolver

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
    # Poisson solver needs absolute permittivity in F/cm
    # device.material.eps is already eps_r * eps0 (F/cm) from MaterialGrid.build()
    eps_grid = device.material.eps

    poisson_solver = PoissonSolver(
        grid=grid, T=T, doping=device.doping,
        eps_grid=eps_grid,
        ni_grid=device.material.ni,
        max_iter=200,
    )
    depletion = DepletionWidth(grid)
    poisson_service = PoissonService(poisson_solver, grid, depletion)

    # Ionization coefficients — Okuto-Crowell model using InP material params from XML.
    # The OC parameters (lambda0, ER0, hw_meV, Eth) for electrons and holes in InP
    # are calibrated from literature (Osaka et al. 1984, Hamoui & Zavalichin 1992).
    # mat_inp is extracted just below, so we build ionization after it.
    # Current components
    mat_inp = materials.get("InP")
    if mat_inp is not None:
        Eg_mulp = mat_inp.Eg()
        mc_mulp = mat_inp.mc
        mh_mulp = mat_inp.mh
        # Build ionization with Okuto-Crowell using InP XML parameters
        ionization = OkutoCrowellCoefficients(mat_inp, T=T)
    else:
        raise ValueError("InP material data required for multiplication layer")

    mat_ingaas = materials.get("InGaAs")
    if mat_ingaas is None:
        raise ValueError("InGaAs material data required for absorption layer")

    current = CurrentDecompositionManager()
    current.add(SRHCurrentDensity(
        mat_name_grid=device.material.mat_name,
        materials=materials,
        T=T,
    ))
    current.add(BTBTCurrentDensity(
        Eg_mulp=Eg_mulp, mc_mulp=mc_mulp, mh_mulp=mh_mulp, T=T))
    current.add(TATCurrentDensity(
        mat_name_grid=device.material.mat_name, materials=materials,
        N_T=1e12, T=T))

    # PDE model - compute absorber layer position from device layers
    x_start_acc = 0.0
    x_abs_start, x_abs_stop = 0.0, 0.0
    for lyr in device.layers:
        if lyr.material == "InGaAs":
            x_abs_start = x_start_acc
            x_abs_stop = x_start_acc + lyr.thickness
            break
        x_start_acc += lyr.thickness

    pde_model = PDEModel(
        grid=grid.x,
        x_abs_start=x_abs_start,
        x_abs_stop=x_abs_stop,
        materials=materials,
    )


    # Continuum drift-diffusion solver (built on demand; can be None)
    continuum = ContinuumSolver(
        grid=grid, doping=device.doping,
        material=device.material, T=T,
    )

    return {
        "poisson_service": poisson_service,
        "ionization": ionization,
        "current": current,
        "pde_model": pde_model,
        "continuum": continuum,
    }
