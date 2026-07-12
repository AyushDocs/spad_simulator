from src.utils.ingestion import DataIngestionConfig, DataIngestionService

cfg = DataIngestionConfig.from_defaults()
svc = DataIngestionService(cfg)
device = svc.build_device()

print("Layer indices and materials:")
for idx, lyr in enumerate(device.layers):
    print(f"Layer {idx}: {lyr.material}, thickness {lyr.thickness*1e4:.3f} um, doping {lyr.doping_A:.2e} ({lyr.doping_type})")

print("\nDoping at grid points:")
x = device.grid.x
net_doping = device.doping.net_doping(x)
for i in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499]:
    print(f"Node {i:3d} | x = {x[i]*1e4:5.2f} um | Net doping = {net_doping[i]:.2e}")
