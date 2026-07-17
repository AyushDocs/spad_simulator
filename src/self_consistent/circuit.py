from __future__ import annotations

import numpy as np
from pydantic import BaseModel, computed_field, model_validator

from ..utils._logging import get_logger

log = get_logger("circuit")


class CircuitSolver(BaseModel):
    """
    External quenching circuit.

    ``Vbias - Rq - Vspad`` with parallel capacitance ``Cspad``.

    KCL:
        Cspad dVspad/dt = (Vbias - Vspad) / Rq - I_av
    """

    model_config = {"arbitrary_types_allowed": True}

    Vbias: float
    Rq: float = 1e5
    Cspad: float = 1e-15
    Vbr: float = 0.0
    I_av: float = 0.0

    Vspad: float = 0.0

    @model_validator(mode="after")
    def _init_vspad(self):
        self.Vspad = self.Vbias
        return self

    @computed_field
    def tau(self) -> float:
        return self.Rq * self.Cspad

    @property
    def Vex(self) -> float:
        return self.Vspad - self.Vbr

    @property
    def is_quenched(self) -> bool:
        return self.Vspad < self.Vbr

    def update(self, I_av: float, dt: float) -> None:
        self.I_av = I_av
        dV = (self.Vbias - self.Vspad) / self.Rq - self.I_av
        self.Vspad += (dt / self.Cspad) * dV

    def recharge_voltage(self, t: float, V0: float | None = None) -> float:
        V0 = V0 if V0 is not None else self.Vspad
        return self.Vbias - (self.Vbias - V0) * np.exp(-t / self.tau)

    def quench_time(self, V0: float | None = None,
                    Vtarget: float | None = None) -> float:
        V0 = V0 if V0 is not None else self.Vspad
        Vt = Vtarget if Vtarget is not None else self.Vbr + 0.1
        num = self.Vbias - V0
        den = max(self.Vbias - Vt, 1e-30)
        if num <= 0 or den <= 0 or num / den <= 0:
            return 0.0
        return float(self.tau * np.log(num / den))
