import numpy as np
from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.simulator import SPADSimulator

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
device = svc.build_device()
sim = SPADSimulator(device)

for V in [23.0, 25.0]:
    phi, E, Pe, Ph, xl, xr = sim.get_fields(V)
    srh_comp = sim.current.components[0]
    J_srh = srh_comp.compute(sim.grid.x, np.abs(E))
    I_srh = float(np.trapezoid(J_srh, sim.grid.x) * sim.detector_area)
    
    # Print nodes contributing to SRH
    contrib_nodes = np.where(J_srh > 0)[0]
    print(f"Bias V = {V:.1f} V | Total SRH current = {I_srh:.3e} A")
    print(f"Number of contributing nodes: {len(contrib_nodes)}")
    for idx in contrib_nodes:
        x_val = sim.grid.x[idx] * 1e4
        F_val = abs(E[idx])
        print(f"  Node {idx:3d} | x = {x_val:5.2f} um | E = {F_val:.2e} V/cm")
