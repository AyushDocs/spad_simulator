import numpy as np
from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.simulator import SPADSimulator

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
device = svc.build_device()
sim = SPADSimulator(device)

Vbr = 66.0
V_sweep = np.arange(0, Vbr + 30, 1.0)

for V in V_sweep:
    try:
        _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
        dc = sim.compute_dark_current(float(V), E=E)
    except Exception as e:
        print(f"Voltage {V} V failed with exception: {type(e).__name__}: {e}")
