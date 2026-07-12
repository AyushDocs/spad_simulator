import numpy as np
from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.simulator import SPADSimulator

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
device = svc.build_device()
sim = SPADSimulator(device)

print("Bias (V) | SRH (A) | BTBT (A) | TAT (A) | Total (A)")
print("-" * 55)

for V in range(0, 70, 5):
    phi, E, Pe, Ph, xl, xr = sim.get_fields(float(V))
    x = sim.grid.x
    
    # Compute each component manually to see their values
    srh_comp = sim.current.components[0]
    btbt_comp = sim.current.components[1]
    tat_comp = sim.current.components[2]
    
    J_srh = srh_comp.compute(x, np.abs(E))
    J_btbt = btbt_comp.compute(x, np.abs(E))
    J_tat = tat_comp.compute(x, np.abs(E))
    
    I_srh = float(np.trapezoid(J_srh, x) * sim.detector_area)
    I_btbt = float(np.trapezoid(J_btbt, x) * sim.detector_area)
    I_tat = float(np.trapezoid(J_tat, x) * sim.detector_area)
    I_total = I_srh + I_btbt + I_tat
    
    # Let's also print number of nodes where F > 1e4 in InGaAs
    in_absorber = srh_comp.mat_name_grid == "InGaAs"
    nodes_dep = np.sum(in_absorber & (np.abs(E) > 1e4))
    total_nodes = np.sum(in_absorber)
    
    print(f"{V:8.1f} | {I_srh:.3e} | {I_btbt:.3e} | {I_tat:.3e} | {I_total:.3e} | Dep nodes: {nodes_dep}/{total_nodes}")
