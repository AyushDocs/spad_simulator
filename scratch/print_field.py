import numpy as np
from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.simulator import SPADSimulator

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
device = svc.build_device()
sim = SPADSimulator(device)

phi, E, Pe, Ph, xl, xr = sim.get_fields(0.0)
x = sim.grid.x

print("Node | x (um) | E (V/cm) | Depleted (F > 1e4)?")
print("-" * 50)
for i in [0, 50, 100, 150, 153, 160, 180, 200, 220, 250, 300, 350, 400, 450, 499]:
    E_val = abs(E[i])
    dep = E_val > 1e4
    print(f"{i:4d} | {x[i]*1e4:5.2f}  | {E_val:.2e} | {dep}")
