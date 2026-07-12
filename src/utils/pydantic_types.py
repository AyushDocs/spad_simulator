from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import BeforeValidator


def _to_ndarray(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v)


NDArray = Annotated[np.ndarray, BeforeValidator(_to_ndarray)]
