from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .grid import Grid1D
from .layer import Layer
from .material import Material
from .doping import DopingProfile
from .material_grid import MaterialGrid
from ..utils._exceptions import ConfigError
from ..utils._logging import get_logger

log = get_logger("device")


@dataclass
class Device:
    layers: List[Layer]
    materials: Dict[str, Material]
    no_of_nodes: int = 200

    _total_L: float = field(init=False)
    grid: Grid1D = field(init=False)
    material: MaterialGrid = field(init=False)
    doping: DopingProfile = field(init=False)
    nd_on_grid: np.ndarray = field(init=False)
    na_on_grid: np.ndarray = field(init=False)
    net_doping_on_grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ConfigError("Device requires at least one layer")

        self.layers = list(self.layers)
        self._total_L = sum(l.thickness for l in self.layers)
        self.grid = Grid1D(self._total_L, self.no_of_nodes)
        self.doping = DopingProfile._from_layers(self.layers, self.grid)
        self.material = MaterialGrid.build(self.layers, self.materials, self.grid)
        self.nd_on_grid = np.asarray(self.doping.nd(self.grid.x))
        self.na_on_grid = np.asarray(self.doping.na(self.grid.x))
        self.net_doping_on_grid = np.asarray(self.doping.net_doping(self.grid.x))

    @property
    def T(self) -> float:
        return next(iter(self.materials.values())).T

    @property
    def L(self) -> float:
        return self._total_L
