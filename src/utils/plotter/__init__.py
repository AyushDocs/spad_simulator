"""Plotter package — re-exports the public API."""
from ._base import Plotter, BasePlotter
from ._registry import get_plotter, register_plotter

__all__ = ["Plotter", "BasePlotter", "get_plotter", "register_plotter"]
