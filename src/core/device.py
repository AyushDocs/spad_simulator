from __future__ import annotations

from typing import Dict, List

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from .grid import Grid1D
from .layer import Layer
from .material import Material
from .doping import DopingProfile
from .material_grid import MaterialGrid
from ..utils._exceptions import ConfigError
from ..utils._logging import get_logger

log = get_logger("device")


class Device(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    layers: List[Layer]
    materials: Dict[str, Material]
    no_of_nodes: int = 200

    grid: Grid1D = None
    doping: DopingProfile = None
    material: MaterialGrid = None
    nd_on_grid: np.ndarray = None
    na_on_grid: np.ndarray = None
    net_doping_on_grid: np.ndarray = None

    _total_L: float = 0.0

    @field_validator("layers")
    @classmethod
    def _layers_nonempty(cls, v: List[Layer]) -> List[Layer]:
        if not v:
            raise ConfigError("Device requires at least one layer")
        return list(v)

    def model_post_init(self, __context) -> None:
        self._total_L = sum(l.thickness for l in self.layers)
        self.grid = Grid1D(L=self._total_L, N=self.no_of_nodes)
        self.doping = DopingProfile._from_layers(self.layers)
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
