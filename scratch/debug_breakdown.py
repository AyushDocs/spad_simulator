import numpy as np
from src.utils.ingestion import DataIngestionConfig, DataIngestionService
from src.simulator import SPADSimulator

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
device = svc.build_device()
sim = SPADSimulator(device)

print("Grid length:", sim.grid.L)
print("Grid nodes:", sim.grid.no_of_nodes)

V = 70.0
phi, E, info = sim.solve_poisson(V)
print(f"At V={V}V, max field E: {np.max(np.abs(E)):.3e} V/cm")

alpha = sim.ionization.alpha_n(np.abs(E))
beta = sim.ionization.alpha_p(np.abs(E))
print(f"Max alpha: {np.max(alpha):.3e}, Max beta: {np.max(beta):.3e}")

# Calculate McIntyre integral manually
x = sim.grid.x
dx = np.diff(x)
diff = alpha - beta
cum = np.zeros_like(x)
for i in range(len(x) - 2, -1, -1):
    cum[i] = cum[i + 1] + diff[i] * dx[i]

integrand = beta * np.exp(cum)
integral = float(np.trapezoid(integrand, x))
denom = 1.0 - integral
print(f"McIntyre integral: {integral:.4f}, denom: {denom:.4f}")
