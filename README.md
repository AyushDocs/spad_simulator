# SPAD Simulator

1D Poisson–Drift–Diffusion simulator for Separate Absorption, Grading, Charge-sheet, and Multiplication (SAGCM) InGaAs/InP Single-Photon Avalanche Diodes.

## Physics

### Poisson Equation

Nonlinear Poisson solved via Newton–Raphson with tridiagonal factorization:

```
d/dx (eps dphi/dx) = -q(p - n + N_D - N_A)
```

Carrier statistics (Boltzmann):
```
n = n_i exp((phi - phi_n) / V_T)
p = n_i exp((phi_p - phi) / V_T)
```

### Impact Ionization (Okuto–Crowell)

```
alpha(E) = (qE / Eth) * exp{S - sqrt(S^2 + (Eth / (qE lambda))^2)}
S = 0.217 * (Eth / ER)^1.14
```

Temperature-dependent via tanh phonon factor.

### Trigger Probability (McIntyre)

Electron trigger:
```
mu(x) = exp(int_x^W (alpha - beta) ds)
M(x) = mu(x) / (1 - int_x^W beta mu ds)
P_e(x) = 1 - 1/M(x)
```

Hole trigger (mirror integral from left contact).

### Tunneling

- **BTBT**: Kane model with exponential field dependence
- **TAT**: Hurkx model with trap-assisted tunneling

### Dark Current

```
J_total = J_SRH + J_BTBT + J_TAT
DCR = A_det * int G(x) * P_trigger(x) dx
```

### PDP Spectrum

```
PDP(lambda) = (1-r) * T_dead(lambda) * int alpha_abs * exp(-alpha_abs * x) * P_trigger(x) dx
```

### Drift-Diffusion Transport

Velocity saturation: `v_d = mu*E / (1 + E/E_sat)`

High-field diffusion: `D = D_0 * (1 + 2*(E/v_sat)^2)`

### Numerical Integration

All spatial integrations (dark current, photocurrent, generation rate, PDP) use the **composite trapezoidal rule** (`np.trapezoid`) rather than rectangular (midpoint/left-Riemann) quadrature:

- **O(h²) accuracy** vs O(h) for rectangular — the trapezoid rule captures the linear slope between adjacent grid points, which is critical in regions where J(x) varies steeply (e.g. the high-field multiplication region).
- **Converges as ~1/h²** — halving the grid spacing reduces the error by ~4×, vs only ~2× for rectangular integration.
- **Handles non-uniform grids** — the trapezoidal rule naturally accommodates variable Δx, which arises from the adaptive grid near heterojunctions.
- Simpson's rule (O(h⁴)) is unnecessary overhead here: the integrands are smooth and the grid is already fine enough that trapezoidal integration converges to < 0.1% of the total current.

### Timing Jitter

Monte Carlo ensemble simulation tracking carrier transit times.

## Device Structure

| Layer | Material | Thickness | Doping |
|-------|----------|-----------|--------|
| p+ cap | InP | 2.5 um | 2e18 cm⁻³ |
| Multiplication | InP | 0.5 um | i |
| Charge sheet | InP | 0.2 um | 1e17 cm⁻³ |
| Grading | InGaAsP | 0.12 um | i |
| Absorber | InGaAs | 1.5 um | i |
| Buffer | InP | 0.5 um | 5e16 cm⁻³ |
| n+ substrate | InP | 2.0 um | 1e18 cm⁻³ |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
python -m src
```

Or programmatically:

```python
from src.main import build_sagcm_spad
from src.simulator import SPADSimulator

device = build_sagcm_spad()
sim = SPADSimulator(device)
Vbr, info = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
print(f"Breakdown voltage: {Vbr:.1f} V")

# PDP spectrum
import numpy as np
wavelengths = np.linspace(900, 1700, 41) * 1e-9
pdp = sim.compute_pdp_spectrum(wavelengths, Vex=3.0)
```

## Running Tests

```bash
pytest
```

## Module Structure

```
src/
  core/           Grid, layers, materials, doping, device
  poisson/        Nonlinear Poisson solver, depletion width
  avalanche/      Impact ionization, trigger, breakdown, dark current, PDP
  transport/      Drift-diffusion, Monte Carlo, timing jitter
  self_consistent/ PIC loop, particle-mesh, circuit quenching
  optimization/   PSO optimizer
  utils/          Exceptions, logging, XML loaders, plotters
```

## License

MIT
