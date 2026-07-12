import numpy as np
from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.simulator import SPADSimulator
from src.simulator.photocurrent import compute_photocurrent

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
device = svc.build_device()
sim = SPADSimulator(device)

Vbr = 66.0
V_sweep = np.arange(0, Vbr + 30, 1.0)
OPTICAL_POWER = 1e-6

for V in V_sweep:
    try:
        _, E, Pe, Ph, _, xr = sim.get_fields(float(V))
        dc = sim.compute_dark_current(float(V), E=E)
        I_photo = compute_photocurrent(
            grid_x=sim.grid.x,
            layers=sim.device.layers,
            materials=sim.materials,
            pdp_model=sim.pdp_model,
            detector_area=sim.detector_area,
            wavelength=1550e-9,
            power=OPTICAL_POWER,
            E=E,
            Pe=Pe,
            Ph=Ph,
            xr=xr,
            multiply=True,
            V_bias=float(V),
            V_br=Vbr,
        )
    except Exception as e:
        print(f"Voltage {V} V failed with exception: {type(e).__name__}: {e}")
