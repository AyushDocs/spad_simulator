"""Regenerate trigger probability plot with fixed plotter."""
import sys
sys.path.insert(0, "/home/ayush/Desktop/code/iitd/spad_simulator")

import numpy as np
from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.studies.fields import run_trigger_profiles

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
sim = svc.build_simulator()

Vbr, _ = sim.find_breakdown(V_start=0, V_max=150, V_step=1.0)
print(f"Vbr = {Vbr} V")

run_trigger_profiles(sim, Vbr)
print("Done — check plots/spad/trigger_probability.png")
