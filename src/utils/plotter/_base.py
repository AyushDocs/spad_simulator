"""Plotter base class and ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  # imported for subclass use via _import()


class Plotter(ABC):
    """Interface for all plotters. Each subclass implements one plot type."""

    @abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class BasePlotter(Plotter):
    """Shared helpers for matplotlib-based plotters."""

    def __init__(self, plot_dir: str = "plots") -> None:
        self.plot_dir = plot_dir

    def _import(self) -> Any:
        return plt

    def _save(self, fname: str, plt: Any = None) -> None:
        import os
        from .._logging import get_logger
        if plt is None:
            plt = self._import()
        log = get_logger("plots")
        path = os.path.join(self.plot_dir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("saved  %s", path)
