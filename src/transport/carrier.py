from __future__ import annotations

import numpy as np


class Carrier:
    """A single charge carrier (electron or hole)."""

    def __init__(self, x: float, typ: str, v: float = 0.0,
                 dead_space: float = 0.0) -> None:
        self._x = float(x)
        self.typ = typ
        self.v = float(v)
        self.E = 0.0
        self.dead_space_remaining = float(dead_space)
        self.ionized = False
        self.alive = True
        self.age = 0.0

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        self._x = float(value)

    @property
    def is_electron(self) -> bool:
        return self.typ == "electron"

    @property
    def is_hole(self) -> bool:
        return self.typ == "hole"

    def move(self, dx: float, dt: float) -> None:
        self._x += dx
        self.age += dt

    def exit_check(self, x_left: float, x_right: float) -> None:
        if self._x < x_left or self._x > x_right:
            self.alive = False

    def reset_dead_space(self, l_dead: float) -> None:
        self.dead_space_remaining = l_dead
        self.ionized = True

    def __repr__(self) -> str:
        return (f"Carrier({self.typ}, x={self._x:.3e}, "
                f"v={self.v:.3e}, alive={self.alive})")
