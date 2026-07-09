"""Shared constants for study runners."""
from __future__ import annotations

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PLOT_DIR = os.path.normpath(os.path.join(DATA_DIR, "..", "plots", "spad"))
OPTICAL_POWER = 1e-6
