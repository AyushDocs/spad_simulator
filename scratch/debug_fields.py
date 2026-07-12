import numpy as np
from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.simulator import SPADSimulator

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
device = svc.build_device()
sim = SPADSimulator(device)

print("Bias (V) | xl (um) | xr (um) | Width (um) | I_dark (A)")
print("-" * 55)

for V in range(0, 95, 5):
    phi, E, Pe, Ph, xl, xr = sim.get_fields(float(V))
    dc = sim.compute_dark_current(float(V), E=E)
    print(f"{V:8.1f} | {xl*1e4:7.3f} | {xr*1e4:7.3f} | {(xr-xl)*1e4:10.3f} | {dc['I_dark']:.3e}")
