from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .grid import Grid1D
from .layer import Layer
from .material import Material
from .constants import eps0, q


_MATERIAL_GETTERS = [
    ("eps", lambda mat: mat.eps_r * eps0),
    ("ni", lambda mat: mat.ni()),
    ("Eg", lambda mat: mat.Eg()),
    ("Eth", lambda mat: mat.Eg() * 1.5 * q),
    ("mu_n", lambda mat: mat.mu_n),
    ("mu_p", lambda mat: mat.mu_p),
    ("vsat_n", lambda mat: mat.vsat_n),
    ("vsat_p", lambda mat: mat.vsat_p),
    ("mc", lambda mat: mat.mc),
    ("mh", lambda mat: mat.mh),
    ("E_ie", lambda mat: mat.E_ie),
    ("E_ih", lambda mat: mat.E_ih),
    ("tau_n", lambda mat: mat.tau_n),
    ("tau_p", lambda mat: mat.tau_p),
    ("Nc", lambda mat: mat.Nc()),
    ("Nv", lambda mat: mat.Nv()),
]


@dataclass
class MaterialGrid:
    eps: np.ndarray = field(repr=False)
    ni: np.ndarray = field(repr=False)
    Eg: np.ndarray = field(repr=False)
    Eth: np.ndarray = field(repr=False)
    mu_n: np.ndarray = field(repr=False)
    mu_p: np.ndarray = field(repr=False)
    vsat_n: np.ndarray = field(repr=False)
    vsat_p: np.ndarray = field(repr=False)
    mc: np.ndarray = field(repr=False)
    mh: np.ndarray = field(repr=False)
    E_ie: np.ndarray = field(repr=False)
    E_ih: np.ndarray = field(repr=False)
    tau_n: np.ndarray = field(repr=False)
    tau_p: np.ndarray = field(repr=False)
    Nc: np.ndarray = field(repr=False)
    Nv: np.ndarray = field(repr=False)
    mat_name: np.ndarray = field(repr=False)

    @classmethod
    def build(cls, layers: List[Layer], materials: Dict[str, Material],
              grid: Grid1D) -> MaterialGrid:
        nn = grid.no_of_nodes
        data: dict[str, np.ndarray] = {}
        for attr, _ in _MATERIAL_GETTERS:
            data[attr] = np.zeros(nn)
        mat_name = np.empty(nn, dtype="<U12")

        x_start = 0.0
        for lyr in layers:
            x_end = x_start + lyr.thickness
            mat = materials[lyr.material]
            mask = (grid.x >= x_start - 1e-16) & (grid.x <= x_end + 1e-16)
            if not np.any(mask):
                x_start = x_end
                continue
            mat_name[mask] = lyr.material
            for attr, getter in _MATERIAL_GETTERS:
                data[attr][mask] = getter(mat)
            x_start = x_end

        return cls(**data, mat_name=mat_name)
